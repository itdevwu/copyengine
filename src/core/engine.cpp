#include "copyengine/engine.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <list>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "backends/executor.hpp"
#include "backends/host/host.hpp"
#include "copyengine/error.hpp"
#include "copyengine/probe.hpp"
#include "core/json.hpp"
#include "core/plan.hpp"
#include "core/tuner.hpp"

#if CE_WITH_CUDA
#include "backends/cuda/executor.hpp"
#endif

namespace ce {

// ---- options --------------------------------------------------------------------

EngineOptions EngineOptions::from_env(EngineOptions o) {
  auto env = [](const char* k) -> std::string {
    const char* v = std::getenv(k);
    return v ? v : "";
  };
  auto truthy = [](const std::string& v) { return !v.empty() && v != "0" && v != "false" && v != "off"; };
  if (truthy(env("CE_DISABLE"))) o.disable = true;
  if (env("CE_LOG").find("plan") != std::string::npos) o.log_plan = true;
  if (auto v = env("CE_STAGING_MB"); !v.empty()) o.staging_budget = std::stoll(v) << 20;
  if (auto v = env("CE_EXPLORE"); !v.empty()) o.explore = truthy(v);
  if (truthy(env("CE_SYNC"))) o.sync_always = true;
  if (auto v = env("CE_MAX_SMS"); !v.empty()) o.max_sms = std::stoi(v);
  return o;
}

EngineOptions EngineOptions::from_env() { return from_env(EngineOptions{}); }

// ---- engine ---------------------------------------------------------------------

namespace {

// What one flush executed: the plan and the strategy picked per group.
struct Ran {
  std::shared_ptr<const Plan> plan;
  std::vector<Strategy> strategies;
};

using Clock = std::chrono::steady_clock;

double us_since(Clock::time_point t0) { return std::chrono::duration<double, std::micro>(Clock::now() - t0).count(); }

int lg2(int64_t x) { return x > 0 ? 63 - std::countl_zero(static_cast<uint64_t>(x)) : -1; }

// An allocation touched by queued work.
struct Alloc {
  Device device;
  int64_t lo, hi;
};

bool overlaps(const std::vector<Alloc>& set, Device d, int64_t lo, int64_t hi) {
  for (const Alloc& a : set)
    if (a.device == d && a.lo < hi && lo < a.hi) return true;
  return false;
}

void add_alloc(std::vector<Alloc>& set, Device d, int64_t lo, int64_t hi) {
  for (Alloc& a : set)
    if (a.device == d && a.lo <= hi && lo <= a.hi) {  // overlapping or touching: widen
      a.lo = std::min(a.lo, lo), a.hi = std::max(a.hi, hi);
      return;
    }
  set.push_back({d, lo, hi});
}

// What interception has learned about one op class.
struct KeyStat {
  int64_t seen = 0, merged = 0;  // ops flushed; of those, in a group with others
  int single = -1;               // a lone op's locked choice: -1 unknown, 0 ce, 1 native
  int64_t since = 0;             // ops run natively since one last went through the tuner
};

// A timed run whose GPU part is still in flight.
struct PendingSample {
  std::string bucket;
  Strategy strategy;
  double host, submit;
  int64_t a, b;
  Executor* ex;
};

// A bounded map that evicts the least recently used entry: a working set
// slightly larger than the cache degrades gradually instead of losing every
// entry at once.
template <class V>
class Lru {
 public:
  explicit Lru(size_t cap = 1) : cap_(cap) {}
  void set_capacity(size_t cap) { cap_ = std::max<size_t>(cap, 1); }
  V* find(const Hash128& k) {
    auto it = map_.find(k);
    if (it == map_.end()) return nullptr;
    order_.splice(order_.begin(), order_, it->second);  // most recent first
    return &it->second->second;
  }
  V& insert(const Hash128& k, V v) {
    if (V* hit = find(k)) return *hit = std::move(v);
    while (map_.size() >= cap_) {
      map_.erase(order_.back().first);
      order_.pop_back();
    }
    order_.emplace_front(k, std::move(v));
    map_.emplace(k, order_.begin());
    return order_.front().second;
  }
  size_t size() const { return map_.size(); }
  void clear() { map_.clear(), order_.clear(); }

 private:
  size_t cap_;
  std::list<std::pair<Hash128, V>> order_;
  std::unordered_map<Hash128, typename std::list<std::pair<Hash128, V>>::iterator, Hash128Hasher> map_;
};

}  // namespace

struct Engine::Impl {
  EngineOptions opt;
  mutable std::recursive_mutex mu;
  int depth = 0;
  std::vector<bool> scope_async;  // per open scope: begin_scope(host_async)
  std::vector<double> scope_bw;   // per open scope: its bandwidth limit (0: inherited)
  int async_depth = 0;            // open scopes with host_async
  std::vector<Op> queue;
  uintptr_t stream = 0;
  Lru<std::shared_ptr<const Plan>> plans;
  Lru<char> safe;  // ops already checked (check_safe)
  Tuner tuner;
  std::unordered_map<Hash128, KeyStat, Hash128Hasher> keys;
  std::vector<Alloc> q_reads, q_writes;  // allocations the queue reads / writes
  std::vector<PendingSample> samples;
  int64_t native_ops = 0;  // ops run natively at interception
  std::map<std::string, int64_t, std::less<>> declined;  // intercepted copies left to the frontend, by reason
  bool last_capture = false;  // the last flush was recorded into a CUDA graph
  HostExecutor host;
  std::unique_ptr<Executor> cuda;
  std::string cuda_error;
  // Streams with issued, unfinished CUDA work, and whether any of it touches
  // host memory (only then must the host wait before returning or reading).
  std::vector<std::pair<uintptr_t, int>> cuda_streams;  // (stream, device) with work synchronize() waits for
  std::vector<std::pair<uintptr_t, int>> pending_at;  // (stream, device) with pending work, for mark_pending
  std::unordered_map<int64_t, std::vector<int64_t>> tokens;  // mark_pending token -> executor marks
  int64_t next_token = 0;
  std::thread::id owner;  // thread with the open scope (depth > 0)

  // Objects whose destruction may run frontend code (Python objects, torch
  // tensors): destroyed only once the engine lock is released (see Locked),
  // so no thread ever waits for the GIL while holding the engine lock.
  std::vector<std::shared_ptr<void>> graveyard;
  int lock_depth = 0;
  void bury(std::shared_ptr<void> p) {
    if (p) graveyard.push_back(std::move(p));
  }

  // Copies whose host side is not waited for (an intercepted non_blocking
  // copy_, or allow_async): their payloads hold the frontend's buffers, which
  // must outlive the copy (torch would otherwise reuse pinned memory early).
  struct Kept {
    std::vector<int64_t> marks;
    std::vector<std::shared_ptr<NativeOp>> payloads;
  };
  std::deque<Kept> kept;

  void keep(const std::vector<Op>& ops, uintptr_t strm) {
    if (!cuda || capturing) return;
    Kept k;
    std::vector<int> devs;
    for (const Op& op : ops) {
      const bool host = op.src.device.kind == DeviceKind::host || op.dst.device.kind == DeviceKind::host;
      const bool cuda_side = op.src.device.kind == DeviceKind::cuda || op.dst.device.kind == DeviceKind::cuda;
      if (!op.native || !host || !cuda_side || !(op.host_async || opt.allow_async)) continue;
      k.payloads.push_back(op.native);
      const int dev = op.src.device.kind == DeviceKind::cuda ? op.src.device.index : op.dst.device.index;
      if (std::find(devs.begin(), devs.end(), dev) == devs.end()) devs.push_back(dev);
    }
    if (k.payloads.empty()) return;
    for (int d : devs) k.marks.push_back(cuda->mark(strm, d));
    kept.push_back(std::move(k));
  }

  // Lets go of kept payloads whose copies have completed (oldest first).
  void reap(bool all = false) {
    while (!kept.empty()) {
      Kept& k = kept.front();
      if (!all)
        for (int64_t m : k.marks)
          if (!cuda->ready(m)) return;
      for (int64_t m : k.marks) cuda->release(m);
      for (auto& p : k.payloads) bury(std::move(p));
      kept.pop_front();
    }
  }

  void check_thread() const {
    if (depth > 0 && owner != std::this_thread::get_id())
      fail(Errc::invalid_argument,
           "this engine has a scope open on another thread; give each thread its own engine");
  }
  bool host_pending = false;
  FlushStats last;
  int64_t total_flushes = 0, total_ops = 0, total_calls = 0, total_bytes = 0, plan_hits = 0;
  int64_t tuning_runs = 0;  // see Engine::tuning_runs
  double total_plan_us = 0;  // planning on plan-cache misses
  bool recording = false;   // see Engine::set_recording
  std::shared_ptr<std::vector<Op>> last_ops;
  Ran last_ran;

  explicit Impl(EngineOptions o)
      : opt(o),
        tuner([&] {
          TunerConfig c;
          c.explore = o.explore, c.prior_only = o.prior_only, c.trials = o.tune_trials, c.topk = o.tune_topk;
          c.pairs = o.ab_pairs, c.max_pairs = o.ab_max_pairs, c.online_every = o.online_every;
          return c;
        }()),
        host(o.host_threads > 0 ? o.host_threads
                                : static_cast<int>(std::min(8u, std::max(1u, std::thread::hardware_concurrency())))) {
    const size_t cap = static_cast<size_t>(std::max(1, o.plan_cache));
    plans.set_capacity(cap);
    stmt_cache.set_capacity(cap * 4);
    safe.set_capacity(cap * 4);
#if CE_WITH_CUDA
    try {
      cuda = make_cuda_executor(opt);
    } catch (const std::exception& e) {
      cuda_error = e.what();
    }
#else
    cuda_error = "built without CUDA support";
#endif
  }

  Executor& exec_for(const Side& s, const Side& d) {
    if (s.device.kind == DeviceKind::host && d.device.kind == DeviceKind::host) return host;
    if (!cuda) fail(Errc::unsupported, "CUDA copy requested but the CUDA backend is unavailable: " + cuda_error);
    return *cuda;
  }

  // Validated, absolute, safety-checked op ready to queue (runs may be empty).
  Op make_op(const Endpoint& src, const Endpoint& dst, Runs runs, const Elementwise& fn, const Hints& hints,
             std::shared_ptr<const Form> form = nullptr) {
    const i64 ssz = static_cast<i64>(itemsize(src.dtype)), dsz = static_cast<i64>(itemsize(dst.dtype));
    for (const Endpoint* e : {&src, &dst}) {
      if (e->base % itemsize(e->dtype) != 0) fail(Errc::invalid_argument, "buffer base is not aligned to its dtype");
      if (e->nbytes < 0) fail(Errc::invalid_argument, "buffer size is negative");
      if (e->device.kind == DeviceKind::cuda && e->pinned) fail(Errc::invalid_argument, "pinned applies to host buffers only");
    }
    if (hints.has_valid) runs = runs.clip_src(hints.valid_start, hints.valid_len);
    const i64 s_lim = src.nbytes / ssz, d_lim = dst.nbytes / dsz;
    for (size_t i = 0; i < runs.size(); ++i) {
      if (runs.src[i] < 0 || runs.src[i] + runs.len[i] > s_lim)
        fail(Errc::invalid_argument, "source access out of bounds: [" + std::to_string(runs.src[i]) + ", " +
                                         std::to_string(runs.src[i] + runs.len[i]) + ") of " + std::to_string(s_lim));
      if (runs.dst[i] < 0 || runs.dst[i] + runs.len[i] > d_lim)
        fail(Errc::invalid_argument, "destination access out of bounds: [" + std::to_string(runs.dst[i]) + ", " +
                                         std::to_string(runs.dst[i] + runs.len[i]) + ") of " + std::to_string(d_lim));
    }
    Op op;
    op.src = {src.device, src.dtype, src.pinned};
    op.dst = {dst.device, dst.dtype, dst.pinned};
    op.fn = fn;
    op.ctx = hints.ctx;
    op.src_lo = static_cast<int64_t>(src.base);
    op.src_hi = static_cast<int64_t>(src.base) + src.nbytes;
    op.dst_lo = static_cast<int64_t>(dst.base);
    op.dst_hi = static_cast<int64_t>(dst.base) + dst.nbytes;
    // Absolute element indices. Pure copies become bytes so any dtypes merge.
    const bool pure = fn.identity() && src.dtype == dst.dtype;
    if (!pure && (!convertible(src.dtype) || !convertible(dst.dtype)))
      fail(Errc::unsupported, std::string(dtype_name(convertible(src.dtype) ? dst.dtype : src.dtype)) +
                                  " is only copied to the same dtype, unconverted");
    const i64 scale = pure ? ssz : 1;
    const i64 sb = static_cast<i64>(src.base) / (pure ? 1 : ssz), db = static_cast<i64>(dst.base) / (pure ? 1 : dsz);
    for (size_t i = 0; i < runs.size(); ++i) {
      runs.src[i] = sb + runs.src[i] * scale;
      runs.dst[i] = db + runs.dst[i] * scale;
      runs.len[i] *= scale;
    }
    if (pure) op.src.dtype = op.dst.dtype = DType::u8;
    if (runs.empty()) return op;
    auto body = std::make_shared<OpBody>();
    body->runs = std::move(runs);
    op.body = std::move(body);
    exec_for(op.src, op.dst);  // fail early when the platform is missing
    op.form = std::move(form);
    op.fp = fingerprint(op);
    Hasher kh;
    kh.add(fingerprint_sides(op));
    kh.add(static_cast<uint64_t>(lg2(op.runs().total() * static_cast<int64_t>(itemsize(op.src.dtype)))));
    kh.add(static_cast<uint64_t>(lg2(static_cast<int64_t>(op.runs().size()))));
    kh.add(uint64_t{op.form != nullptr});
    op.key = kh.done();
    if (!safe.find(op.fp)) {
      check_safe(op);
      safe.insert(op.fp, 1);
    }
    return op;
  }

  // Structured form of a table-free statement, or null if it has none.
  // `stab` / `dtab`: where to put the device table pointers, if the statement
  // has them (at most one dimension per side), else they make it formless.
  // Merges adjacent dimensions that are one on both sides (k is outer to
  // k + 1): the same elements in the same order, with less index arithmetic
  // per element and wider vectors. Table dimensions stay.
  static void coalesce(Form& f) {
    for (int k = f.rank - 2; k >= 0; --k) {
      const int j = k + 1;
      if (k == f.stab_dim || j == f.stab_dim || k == f.dtab_dim || j == f.dtab_dim) continue;
      if (f.ss[k] != f.ss[j] * f.shape[j] || f.ds[k] != f.ds[j] * f.shape[j]) continue;
      f.shape[j] *= f.shape[k];
      for (int m = k; m + 1 < f.rank; ++m) f.shape[m] = f.shape[m + 1], f.ss[m] = f.ss[m + 1], f.ds[m] = f.ds[m + 1];
      --f.rank;
      f.shape[f.rank] = 0, f.ss[f.rank] = 0, f.ds[f.rank] = 0;
      if (f.stab_dim > k) --f.stab_dim;
      if (f.dtab_dim > k) --f.dtab_dim;
    }
  }

  static std::shared_ptr<const Form> make_form(const Endpoint& src, const Endpoint& dst, const Stmt& stmt,
                                               uintptr_t* stab = nullptr, uintptr_t* dtab = nullptr) {
    if (stmt.src.has_table() || stmt.dst.has_table()) return nullptr;
    Form f;
    f.sdt = src.dtype, f.ddt = dst.dtype;
    f.s_base = static_cast<i64>(src.base / itemsize(src.dtype)) + stmt.src.base();
    f.d_base = static_cast<i64>(dst.base / itemsize(dst.dtype)) + stmt.dst.base();
    const auto &st = stmt.src.terms(), &dt = stmt.dst.terms();
    for (size_t k = 0; k < st.size(); ++k) {
      const bool sdev = st[k].dev.ptr != 0, ddev = dt[k].dev.ptr != 0;
      if (st[k].size == 1 && !sdev && !ddev) continue;  // contributes nothing but its (already folded) base
      if (f.rank == Form::kMaxRank) return nullptr;
      if (sdev) {
        if (f.stab_dim >= 0 || !stab) return nullptr;
        f.stab_dim = f.rank, f.stab_step = st[k].dev.step, f.sbound = st[k].dev.bound, *stab = st[k].dev.ptr;
        f.stab_host = st[k].dev.host();
      }
      if (ddev) {
        if (f.dtab_dim >= 0 || !dtab) return nullptr;
        f.dtab_dim = f.rank, f.dtab_step = dt[k].dev.step, f.dbound = dt[k].dev.bound, *dtab = dt[k].dev.ptr;
        f.dtab_host = dt[k].dev.host();
      }
      f.shape[f.rank] = st[k].size, f.ss[f.rank] = st[k].stride, f.ds[f.rank] = dt[k].stride;
      f.numel *= st[k].size;
      ++f.rank;
    }
    if (f.numel <= 0 || f.numel >= (int64_t{1} << 31)) return nullptr;
    coalesce(f);
    return std::make_shared<const Form>(f);
  }

  // Adjacent to the queue's last op on both allocations: merging it is a
  // dominance-safe transform, so it always queues.
  bool merges_with_tail(const Op& op) const {
    if (queue.empty() || op.opaque || queue.back().opaque) return false;
    const Op& t = queue.back();
    if (!(t.src == op.src && t.dst == op.dst && t.fn == op.fn && t.ctx == op.ctx && t.src_lo == op.src_lo &&
          t.dst_lo == op.dst_lo))
      return false;
    const Runs &a = t.runs(), &b = op.runs();
    return a.dst.back() + a.len.back() == b.dst.front() || a.src.back() + a.len.back() == b.src.front();
  }

  // Interception-time decision: run natively now instead of queueing. Only
  // for an op class whose lone ops were measured slower in ce and that rarely
  // merges (so its first op is not the head of a mergeable run).
  // One op in online_every still goes through the queue and the tuner, which
  // keeps re-checking the decision (it stands for the runs that skipped it).
  bool native_now(Op& op) {
    if (!op.native || opt.prior_only || recording) return false;
    if (opt.disable) return true;
    if (merges_with_tail(op)) return false;
    auto it = keys.find(op.key);
    if (it == keys.end() || it->second.single != 1) return false;
    KeyStat& ks = it->second;
    if (!(depth == 0 || (ks.seen >= 16 && ks.merged * 10 < ks.seen))) return false;
    if (opt.online_every > 0 && ++ks.since >= opt.online_every) {
      op.weight = ks.since;
      ks.since = 0;
      return false;
    }
    return true;
  }

  bool conflicts(Device sd, int64_t slo, int64_t shi, Device dd, int64_t dlo, int64_t dhi) const {
    return overlaps(q_writes, sd, slo, shi) || overlaps(q_writes, dd, dlo, dhi) || overlaps(q_reads, dd, dlo, dhi);
  }

  // `host_wait`: the caller expects the host side complete when it returns.
  // CUDA Graph capture of `strm` (torch.cuda.graph): work is recorded into
  // the graph, never run now, so nothing may be timed, waited for or kept
  // alive by the engine (the graph's owner keeps the tensors).
  bool capturing = false;
  bool is_capturing(uintptr_t strm) { return cuda && cuda->capturing(strm); }

  // What runs a group during capture: the locked or given strategy if it only
  // enqueues stream work, else torch's own copy (capturable), else plain DMA.
  Strategy capture_choice(const Group& g, const std::vector<Op>& ops, std::optional<Strategy> want) {
    Executor& ex = exec_for(g.src, g.dst);
    if (g.bandwidth > 0) {  // the limit holds in graphs too: paced or nothing
      if (ex.capture_safe(Strategy::paced, g)) return Strategy::paced;
      fail(Errc::unsupported,
           "copyengine cannot record this bandwidth-limited copy in a CUDA graph: run it once before capturing "
           "(pacing state and tables are set up on first use); pageable host memory cannot be captured");
    }
    if (!want) want = tuner.locked_strategy(g.bucket);
    if (want && *want != Strategy::native && ex.capture_safe(*want, g)) return *want;
    if (std::all_of(g.ops.begin(), g.ops.end(), [&](int k) { return ops[k].native != nullptr; })) return Strategy::native;
    for (Strategy s : {Strategy::runs, Strategy::passthrough, Strategy::fused})
      if (ex.capture_safe(s, g)) return s;
    fail(Errc::unsupported,
         "copyengine cannot record this copy in a CUDA graph: torch has no equivalent and its strategy needs "
         "resources that are set up on first use; run it once before capturing");
  }

  void note_pending(const Side& s, const Side& d, uintptr_t strm, bool host_wait) {
    if (capturing) return;  // the graph's work, not the engine's
    if (s.device.kind == DeviceKind::host && d.device.kind == DeviceKind::host) return;
    const int dev = s.device.kind == DeviceKind::cuda ? s.device.index : d.device.index;
    if (std::find(cuda_streams.begin(), cuda_streams.end(), std::make_pair(strm, dev)) == cuda_streams.end())
      cuda_streams.emplace_back(strm, dev);
    if (std::find(pending_at.begin(), pending_at.end(), std::make_pair(strm, dev)) == pending_at.end())
      pending_at.emplace_back(strm, dev);
    if (host_wait && (s.device.kind == DeviceKind::host || d.device.kind == DeviceKind::host)) host_pending = true;
  }

  // torch's own copy keeps its buffers safe itself (it records events on pinned memory).
  void run_native_now(Op& op, uintptr_t strm) {
    if (!queue.empty() && (strm != stream || !op.extras.empty() ||
                           conflicts(op.src.device, op.src_lo, op.src_hi, op.dst.device, op.dst_lo, op.dst_hi)))
      flush();
    const bool host_only = op.src.device.kind == DeviceKind::host && op.dst.device.kind == DeviceKind::host;
    if (host_only && host_pending && !capturing) synchronize();  // host work may touch what CUDA work still uses
    op.native->run(strm, op.host_async);
    note_pending(op.src, op.dst, strm, !op.host_async);
    bury(std::move(op.native));
    ++native_ops;
    if (depth == 0) finish(strm);
  }

  // The limit for copies queued now: the innermost scope's, else the engine's.
  double bandwidth_now() const {
    for (auto it = scope_bw.rbegin(); it != scope_bw.rend(); ++it)
      if (*it > 0) return *it;
    return opt.bandwidth;
  }

  void push(Op op, uintptr_t strm) {
    if (op.runs().empty() && !op.opaque) return;
    if (async_depth > 0) op.host_async = true;
    if (const double bw = bandwidth_now();
        bw > 0 && !opt.disable && !op.opaque &&
        (op.src.device.kind == DeviceKind::cuda || op.dst.device.kind == DeviceKind::cuda)) {
      // Paced: never the frontend's full-speed copy, and its own plans.
      // Not copies with device indices: pacing cuts runs, and they have none.
      op.bandwidth = bw;
      op.native.reset();
      op.fp.a ^= std::bit_cast<uint64_t>(bw) * 0x9E3779B97F4A7C15ull;
      op.fp.b += 0x5EED;
    }
    // Only work that touches the device now asks about capture: a copy queued
    // in a scope costs nothing extra.
    std::optional<Capture> cap;
    if (depth == 0 || op.native || !kept.empty()) cap.emplace(*this, strm);
    if (!kept.empty() && !capturing) reap();
    if (native_now(op)) {
      run_native_now(op, strm);
      return;
    }
    if (!queue.empty() && strm != stream) flush();
    stream = strm;
    add_alloc(q_reads, op.src.device, op.src_lo, op.src_hi);
    add_alloc(q_writes, op.dst.device, op.dst_lo, op.dst_hi);
    for (const Op::Extra& x : op.extras) add_alloc(x.write ? q_writes : q_reads, x.device, x.lo, x.hi);
    queue.push_back(std::move(op));
    if (depth == 0) {
      flush();
      finish(strm);
    }
  }

  void enqueue(const Endpoint& src, const Endpoint& dst, Runs runs, const Elementwise& fn, const Hints& hints,
               uintptr_t strm, Clock::time_point t0) {
    Op op = make_op(src, dst, std::move(runs), fn, hints);
    op.native = hints.native;
    op.host_async = hints.host_async;
    op.enqueue_us = us_since(t0);
    push(std::move(op), strm);
  }

  void fence(std::span<const Range> reads, std::span<const Range> writes) {
    if (queue.empty()) return;
    bool hit = false, host_side = false;
    for (const Range& r : reads)
      if (overlaps(q_writes, r.device, static_cast<int64_t>(r.lo), static_cast<int64_t>(r.hi)))
        hit = true, host_side |= r.device.kind == DeviceKind::host;
    for (const Range& r : writes)
      if (overlaps(q_writes, r.device, static_cast<int64_t>(r.lo), static_cast<int64_t>(r.hi)) ||
          overlaps(q_reads, r.device, static_cast<int64_t>(r.lo), static_cast<int64_t>(r.hi)))
        hit = true, host_side |= r.device.kind == DeviceKind::host;
    if (!hit) return;
    flush();
    if (host_side) synchronize();  // the CPU is about to touch memory queued CUDA work uses
  }

  // Statement cache: an identical copy (same buffers, layouts, tables,
  // function) reuses the lowered, checked op without touching its runs.
  Lru<Op> stmt_cache;

  static void add_access(Hasher& h, const Access& a) {
    h.add(static_cast<uint64_t>(a.base()));
    h.add(static_cast<uint64_t>(a.rank()));
    for (const Term& t : a.terms()) {
      h.add(static_cast<uint64_t>(t.size));
      h.add(static_cast<uint64_t>(t.stride));
      if (t.table) h.add(std::span<const i64>(*t.table));
      else h.add(uint64_t{0xA77});
    }
  }

  static void add_endpoint(Hasher& h, const Endpoint& e) {
    h.add((static_cast<uint64_t>(e.device.kind) << 40) | (static_cast<uint64_t>(e.device.index) << 8) |
          (static_cast<uint64_t>(e.dtype) << 1) | (e.pinned ? 1u : 0u));
    h.add(static_cast<uint64_t>(e.base));
    h.add(static_cast<uint64_t>(e.nbytes));
  }

  // A statement whose indices live on the device: no runs, its form is how it
  // runs (a statement kernel reads the indices). Its footprint is its form's
  // extent, a table dimension spanning its bound; the kernel checks the values.
  void enqueue_device_stmt(const Endpoint& src, const Endpoint& dst, const Stmt& stmt, const Elementwise& fn,
                           const Hints& hints, uintptr_t strm, Clock::time_point t0) {
    if (hints.has_valid) fail(Errc::unsupported, "valid_src with device indices");
    bool on_gpu = false, on_host = false;
    for (const Access* a : {&stmt.src, &stmt.dst})
      for (const Term& t : a->terms())
        if (t.dev.ptr) (t.dev.host() ? on_host : on_gpu) = true;
    const bool scuda = src.device.kind == DeviceKind::cuda, dcuda = dst.device.kind == DeviceKind::cuda;
    if (on_gpu) {
      if (!scuda || !(src.device == dst.device))
        fail(Errc::unsupported, "device indices need both tensors on the GPU that holds the indices");
      for (const Access* a : {&stmt.src, &stmt.dst})
        for (const Term& t : a->terms())
          if (t.dev.ptr && !t.dev.host() && t.dev.device != src.device.index)
            fail(Errc::unsupported, "device indices on another GPU than the tensors");
    }
    if (on_host && !scuda && !dcuda) fail(Errc::unsupported, "host index tables need a GPU on one side");
    if (scuda && dcuda && !(src.device == dst.device))
      fail(Errc::unsupported, "host index tables between two GPUs");
    Op op;
    std::shared_ptr<const Form> form = make_form(src, dst, stmt, &op.stab, &op.dtab);
    if (!form)
      fail(Errc::needs_materialize, "a copy with device indices needs at most 6 dimensions and one index table per side");
    const auto [slo, shi] = form->src_extent();
    const auto [dlo, dhi] = form->dst_extent();
    const i64 ss = static_cast<i64>(itemsize(src.dtype)), ds = static_cast<i64>(itemsize(dst.dtype));
    const i64 s0 = static_cast<i64>(src.base) / ss, d0 = static_cast<i64>(dst.base) / ds;
    if (slo < s0 || shi >= s0 + src.nbytes / ss) fail(Errc::invalid_argument, "source access out of bounds");
    if (dlo < d0 || dhi >= d0 + dst.nbytes / ds) fail(Errc::invalid_argument, "destination access out of bounds");
    exec_for({src.device, src.dtype, src.pinned}, {dst.device, dst.dtype, dst.pinned});
    op.src = {src.device, src.dtype, src.pinned};
    op.dst = {dst.device, dst.dtype, dst.pinned};
    op.fn = fn;
    op.ctx = hints.ctx;
    op.src_lo = static_cast<int64_t>(src.base), op.src_hi = static_cast<int64_t>(src.base) + src.nbytes;
    op.dst_lo = static_cast<int64_t>(dst.base), op.dst_hi = static_cast<int64_t>(dst.base) + dst.nbytes;
    op.body = std::make_shared<OpBody>();
    op.form = std::move(form);
    op.opaque = true;
    op.fp = fingerprint(op);
    Hasher kh;
    kh.add(fingerprint_sides(op));
    kh.add(uint64_t{0x0DE7});
    kh.add(static_cast<uint64_t>(lg2(op.form->numel * ss)));
    op.key = kh.done();
    op.native = hints.native;
    op.host_async = hints.host_async;
    op.enqueue_us = us_since(t0);
    push(std::move(op), strm);
  }

  // Byte range [lo, hi) a tile tensor's layout touches: rows, then `count`
  // steps of `step` bytes, each `width` bytes wide.
  static std::pair<int64_t, int64_t> tile_extent(const TileSpec& t, const TileSpec::Side& s, int64_t count,
                                                 int64_t width) {
    int64_t lo = static_cast<int64_t>(s.ep.base) + s.offset, hi = lo;
    for (int k = 0; k < 3; ++k) {
      const int64_t span = (t.rows[k] - 1) * s.row_stride[k];
      lo += std::min<int64_t>(0, span), hi += std::max<int64_t>(0, span);
    }
    const int64_t span = (count - 1) * s.step;
    lo += std::min<int64_t>(0, span), hi += std::max<int64_t>(0, span);
    return {lo, hi + width};
  }

  void enqueue_tile(const TileSpec& t, const Hints& hints, uintptr_t strm, Clock::time_point t0) {
    using cv::QFmt;
    const auto f = static_cast<QFmt>(t.fmt);
    if (t.fmt < 0 || t.fmt > static_cast<int>(QFmt::uint4)) fail(Errc::invalid_argument, "unknown quantization format");
    const bool packed = cv::qfmt_packed(f), mx = cv::qfmt_mx(f), byte_scale = mx || f == QFmt::nvfp4;
    if (t.g <= 0 || (packed && t.g % 2) || t.upr <= 0) fail(Errc::invalid_argument, "bad unit size");
    if (mx && t.g != 32) fail(Errc::invalid_argument, "MX formats use units of 32 values");
    if (f == QFmt::nvfp4 && t.g != 16) fail(Errc::invalid_argument, "NVFP4 uses units of 16 values");
    const DType xd = t.x.ep.dtype;
    if (xd != DType::f32 && xd != DType::f16 && xd != DType::bf16)
      fail(Errc::invalid_argument, "the full-precision tensor must be float32, float16 or bfloat16");
    const DType want_q = f == QFmt::fp8_e4m3 || f == QFmt::mxfp8_e4m3 ? DType::f8e4m3
                         : f == QFmt::fp8_e5m2 || f == QFmt::mxfp8_e5m2 ? DType::f8e5m2
                         : f == QFmt::int8                               ? DType::i8
                                                                         : DType::u8;
    const DType qd = t.q.ep.dtype;
    if (!(qd == want_q || (packed && (qd == DType::u8 || qd == DType::i8 || qd == DType::f4x2))))
      fail(Errc::invalid_argument, "the quantized tensor's dtype does not match the format");
    const DType sd = t.scale.ep.dtype;
    if (byte_scale ? itemsize(sd) != 1 : sd != DType::f32)
      fail(Errc::invalid_argument, byte_scale ? "the scale must be a one-byte tensor (E8M0 for MX, E4M3 for NVFP4)"
                                              : "the scale must be float32");
    const bool zero = f == QFmt::uint4;
    if (zero && itemsize(t.zero.ep.dtype) != 1) fail(Errc::invalid_argument, "the zero point must be uint8");
    // All on the host, or on one GPU with the rest in pinned host memory (the
    // kernel reads and writes it directly: quantize and move in one pass).
    std::vector<const TileSpec::Side*> sides{&t.x, &t.q, &t.scale};
    if (zero) sides.push_back(&t.zero);
    Device dev{DeviceKind::host, 0};
    for (const auto* s : sides)
      if (s->ep.device.kind == DeviceKind::cuda) {
        if (dev.kind == DeviceKind::cuda && !(s->ep.device == dev))
          fail(Errc::unsupported, "quantization tensors on two GPUs");
        dev = s->ep.device;
      }
    if (dev.kind == DeviceKind::cuda && t.x.ep.device.kind != DeviceKind::cuda && t.q.ep.device.kind != DeviceKind::cuda)
      fail(Errc::unsupported, "quantization with only the scale on a GPU");
    if (dev.kind == DeviceKind::cuda)
      for (const auto* s : sides)
        if (s->ep.device.kind == DeviceKind::host && !s->ep.pinned)
          fail(Errc::unsupported, "quantization between a GPU and pageable host memory (pin it)");

    for (const TileSpec::Index* ix : {&t.x_index, &t.index}) {
      if (!ix->used()) continue;
      if (ix->dev && dev.kind != DeviceKind::cuda) fail(Errc::unsupported, "GPU index tables need GPU tensors");
      if (!ix->dev && !ix->host_ptr && static_cast<int64_t>(ix->host.size()) != t.rows[0])
        fail(Errc::invalid_argument, "an index table's length is not rows[0]");
      for (int64_t v : ix->host)
        if (v < 0 || v >= ix->bound) fail(Errc::invalid_argument, "index out of range");
    }
    const int64_t vals = t.upr * t.g;
    const auto bound = [](const TileSpec::Side& s, std::pair<int64_t, int64_t> e, const char* what) {
      const int64_t b = static_cast<int64_t>(s.ep.base);
      if (e.first < b || e.second > b + s.ep.nbytes) fail(Errc::invalid_argument, std::string(what) + " out of bounds");
      return e;
    };
    // An indexed side may touch any row its index can reach: its whole bound.
    TileSpec tx = t, tc = t;
    if (t.x_index.used()) tx.rows[0] = t.x_index.bound;
    if (t.index.used()) tc.rows[0] = t.index.bound;
    const auto xr = bound(t.x, tile_extent(tx, t.x, vals, static_cast<int64_t>(itemsize(xd))), "x");
    const auto qr = bound(t.q, tile_extent(tc, t.q, packed ? vals / 2 : vals, 1), "q");
    const auto sr = bound(t.scale, tile_extent(tc, t.scale, t.upr, byte_scale ? 1 : 4), "scale");
    std::pair<int64_t, int64_t> zr{0, 0};
    if (zero) zr = bound(t.zero, tile_extent(tc, t.zero, t.upr, 1), "zero");

    auto k = std::make_shared<cv::TileK>();
    auto at = [](const TileSpec::Side& s) { return static_cast<long long>(s.ep.base) + s.offset; };
    k->x = at(t.x), k->q = at(t.q), k->s = at(t.scale), k->z = zero ? at(t.zero) : 0;
    for (int i = 0; i < 3; ++i) {
      k->rows[i] = t.rows[i];
      k->xr[i] = t.x.row_stride[i], k->qr[i] = t.q.row_stride[i], k->sr[i] = t.scale.row_stride[i];
      k->zr[i] = zero ? t.zero.row_stride[i] : 0;
    }
    k->xe = t.x.step, k->qe = t.q.step, k->se = t.scale.step, k->ze = zero ? t.zero.step : 0;
    k->g = t.g, k->upr = static_cast<int>(t.upr), k->fmt = t.fmt, k->xdt = static_cast<int>(xd);
    k->scale_byte = byte_scale, k->quantize = t.quantize, k->eps = t.eps, k->gs = t.gs;
    k->static_scale = t.quantize && t.static_scale;
    {  // the kernels' vector path (TileK::vec)
      const long long xa = 4 * static_cast<long long>(itemsize(xd)), qa = packed ? 2 : 4;
      bool v = t.g % 4 == 0 && k->xe == static_cast<long long>(itemsize(xd)) && k->qe == 1 && k->x % xa == 0 &&
               k->q % qa == 0;
      for (int i = 0; i < 3; ++i) v = v && k->xr[i] % xa == 0 && k->qr[i] % qa == 0;
      k->vec = v;
    }
    // Index tables: host values live with the op (the executor reads them, or
    // uploads them for a kernel); GPU ones are this call's pointer.
    std::shared_ptr<std::vector<int64_t>> hx, hc;
    bool by_ptr[2] = {false, false};
    if (t.x_index.used()) {
      k->xb = t.x_index.bound;
      if (t.x_index.dev) k->xi = static_cast<long long>(t.x_index.dev);
      else if (t.x_index.host_ptr) k->xi = static_cast<long long>(t.x_index.host_ptr), by_ptr[0] = true;
      else hx = std::make_shared<std::vector<int64_t>>(t.x_index.host), k->xi = reinterpret_cast<long long>(hx->data());
    }
    if (t.index.used()) {
      k->cb = t.index.bound;
      if (t.index.dev) k->ci = static_cast<long long>(t.index.dev);
      else if (t.index.host_ptr) k->ci = static_cast<long long>(t.index.host_ptr), by_ptr[1] = true;
      else hc = std::make_shared<std::vector<int64_t>>(t.index.host), k->ci = reinterpret_cast<long long>(hc->data());
    }

    const Side xs{t.x.ep.device, xd, t.x.ep.pinned}, qs{t.q.ep.device, qd, t.q.ep.pinned};
    exec_for(xs, qs);
    Op op;
    op.src = t.quantize ? xs : qs;
    op.dst = t.quantize ? qs : xs;
    const auto& src_ep = t.quantize ? t.x.ep : t.q.ep;
    const auto& dst_ep = t.quantize ? t.q.ep : t.x.ep;
    op.src_lo = static_cast<int64_t>(src_ep.base), op.src_hi = op.src_lo + src_ep.nbytes;
    op.dst_lo = static_cast<int64_t>(dst_ep.base), op.dst_hi = op.dst_lo + dst_ep.nbytes;
    auto body = std::make_shared<OpBody>();
    const auto rd = t.quantize ? xr : qr, wr = t.quantize ? qr : xr;
    body->rd = std::make_unique<Intervals>(), body->wr = std::make_unique<Intervals>();
    body->rd->start.push_back(rd.first), body->rd->end.push_back(rd.second);
    body->wr->start.push_back(wr.first), body->wr->end.push_back(wr.second);
    op.body = std::move(body);
    const bool writes_scale = t.quantize && !t.static_scale;
    op.extras.push_back({t.scale.ep.device, sr.first, sr.second, writes_scale});
    if (zero) op.extras.push_back({t.zero.ep.device, zr.first, zr.second, writes_scale});
    op.tile = std::move(k);
    op.tile_index[0] = std::move(hx), op.tile_index[1] = std::move(hc);
    op.tile_host_ptr[0] = by_ptr[0], op.tile_host_ptr[1] = by_ptr[1];
    {  // device staging for host sides (Op::tile_staged_ok)
      // A host side is staged whole, or row by row where a host index table
      // selects its rows: an input's extent (per row) holds at most twice the
      // bytes it reads, an output's exactly the bytes it writes (so copying
      // it back writes nothing else). An index shared by host and GPU sides,
      // or a GPU table selecting host rows, keeps the op zero-copy.
      const int64_t xw = static_cast<int64_t>(itemsize(xd)), sw = byte_scale ? 1 : 4;
      TileSpec t1 = t;  // one row of the first dimension
      t1.rows[0] = 1;
      struct S {
        const TileSpec::Side& side;
        const TileSpec::Index& index;
        std::pair<int64_t, int64_t> e;
        int64_t width, count;  // bytes per value or scale, and how many per row (rows[1], rows[2] aside)
        bool input;
      };
      const bool reads_scale = !t.quantize || t.static_scale;
      std::vector<S> ss{{t.x, t.x_index, xr, xw, vals, t.quantize},
                        {t.q, t.index, qr, 1, packed ? vals / 2 : vals, !t.quantize},
                        {t.scale, t.index, sr, sw, t.upr, reads_scale}};
      if (zero) ss.push_back({t.zero, t.index, zr, 1, t.upr, reads_scale});
      bool host = false, ok = true, c_host = false, c_dev = false;
      for (const S& s : ss) {
        if (&s.index == &t.index && s.index.used()) (s.side.ep.device.kind == DeviceKind::host ? c_host : c_dev) = true;
        if (s.side.ep.device.kind != DeviceKind::host) continue;
        host = true;
        int64_t span = s.e.second - s.e.first, need = t.rows[0] * t.rows[1] * t.rows[2] * s.count * s.width;
        if (s.index.used()) {
          ok = ok && !s.index.dev;
          const auto e1 = tile_extent(t1, s.side, s.count, s.width);
          span = e1.second - e1.first, need = t.rows[1] * t.rows[2] * s.count * s.width;
        }
        ok = ok && (s.input ? span <= 2 * need : span == need);
      }
      ok = ok && !(c_host && c_dev);
      op.tile_host = host;
      op.tile_staged_ok = host && ok && dev.kind == DeviceKind::cuda;
    }
    op.tile_dev_index = t.x_index.dev || t.index.dev;
    op.opaque = true;
    op.ctx = hints.ctx;
    op.fp = fingerprint(op);
    Hasher kh;
    kh.add(fingerprint_sides(op));
    kh.add(uint64_t{0x711E});
    kh.add(static_cast<uint64_t>(t.fmt) | (uint64_t{t.quantize} << 8) | (uint64_t{t.static_scale} << 9));
    kh.add(static_cast<uint64_t>(lg2(op.tile->units() * t.g)));
    op.key = kh.done();
    op.native = hints.native;
    op.host_async = hints.host_async;
    op.enqueue_us = us_since(t0);
    push(std::move(op), strm);
  }

  void enqueue_stmt(const Endpoint& src, const Endpoint& dst, const Stmt& stmt, const Elementwise& fn,
                    const Hints& hints, uintptr_t strm, Clock::time_point t0) {
    if (stmt.src.has_device_table() || stmt.dst.has_device_table()) {
      enqueue_device_stmt(src, dst, stmt, fn, hints, strm, t0);
      return;
    }
    Hasher h;
    add_endpoint(h, src);
    add_endpoint(h, dst);
    add_access(h, stmt.src);
    add_access(h, stmt.dst);
    h.add(std::bit_cast<uint32_t>(fn.scale));
    h.add(std::bit_cast<uint32_t>(fn.bias));
    h.add((fn.round ? 1u : 0u) | (fn.clamp ? 2u : 0u));
    h.add(std::bit_cast<uint32_t>(fn.lo));
    h.add(std::bit_cast<uint32_t>(fn.hi));
    h.add(hints.ctx);
    h.add(uint64_t{hints.has_valid});
    h.add(std::span<const i64>(hints.valid_start));
    h.add(std::span<const i64>(hints.valid_len));
    const Hash128 key = h.done();
    if (const Op* cached = stmt_cache.find(key)) {
      Op op = *cached;
      op.native = hints.native;
      op.host_async = hints.host_async;
      op.enqueue_us = us_since(t0);
      push(std::move(op), strm);
      return;
    }
    std::shared_ptr<const Form> form = hints.has_valid ? nullptr : make_form(src, dst, stmt);
    Op op = make_op(src, dst, lower(stmt), fn, hints, std::move(form));
    stmt_cache.insert(key, op);
    op.native = hints.native;
    op.host_async = hints.host_async;
    op.enqueue_us = us_since(t0);
    push(std::move(op), strm);
  }

  void log(const Plan& p, size_t nops) {
    std::fprintf(stderr, "[copyengine] plan: %zu ops -> %d waves, %zu groups\n", nops, p.waves, p.groups.size());
    for (const Group& g : p.groups) {
      std::fprintf(stderr, "[copyengine]   wave %d  %s  ops=%zu runs=%zu rects=%zu bytes=%lld  cands=", g.wave,
                   g.bucket.c_str(), g.ops.size(), g.runs.size(), g.rects.size(), static_cast<long long>(g.bytes()));
      for (Strategy s : g.candidates) std::fprintf(stderr, "%s ", std::string(strategy_name(s)).c_str());
      std::fprintf(stderr, "\n");
    }
  }

  static int device_of(const Group& g) {
    return g.src.device.kind == DeviceKind::cuda ? g.src.device.index : g.dst.device.index;
  }

  // Dominance-safe: merged runs that at least halve the call count. On the
  // host a 2-D block is the same row copies, issued once; on CUDA a 2-D DMA can
  // be slower than its rows, so only 1-D merges qualify there.
  static bool dominance_safe(const Group& g) {
    const bool host_only = g.src.device.kind == DeviceKind::host && g.dst.device.kind == DeviceKind::host;
    return !g.opaque() && g.pure() && (host_only || !g.rects.has_2d()) && g.rects.size() * 2 <= g.ops.size();
  }

  // Passthrough issues every submitted run separately: beyond this many calls
  // it is never worth measuring (a gather of 100k rows takes seconds).
  static constexpr int64_t kPassthroughCalls = 4096;

  Choice choice_for(const Group& g, const std::vector<Op>& ops) const {
    Choice c;
    c.cands = g.candidates;
    const bool nat = std::all_of(g.ops.begin(), g.ops.end(), [&](int k) { return ops[k].native != nullptr; });
    c.baseline = nat ? Strategy::native : Strategy::passthrough;
    if (dominance_safe(g)) c.safe = Strategy::runs;
    if (!nat && !c.safe && !c.cands.empty()) {
      // No native copy: nothing the caller would otherwise run. The prior's
      // first choice is the reference, passthrough a candidate if it is cheap.
      c.lead = c.cands.front();
      if (g.op_runs <= kPassthroughCalls && !g.opaque()) c.cands.push_back(Strategy::passthrough);
    }
    return c;
  }

  // Reports every timed run whose GPU part has finished.
  void poll_samples() {
    std::erase_if(samples, [&](const PendingSample& p) {
      const double gpu = p.ex->elapsed(p.a, p.b);
      if (gpu < 0) return false;
      tuner.report(p.bucket, p.strategy, {p.host, gpu, p.submit});
      return true;
    });
  }

  int64_t run_group(const Group& g, Strategy s, const std::vector<Op>& ops, uintptr_t strm, bool measure,
                    FlushStats* st) {
    Executor& ex = exec_for(g.src, g.dst);
    if (&ex == &host && host_pending && !capturing) synchronize();  // host work may touch what CUDA work still uses
    const int64_t a = measure ? ex.mark(strm, device_of(g)) : -1;
    // Setup on a strategy's first run for this group (tables made resident,
    // descriptors built) is paid once, not per run: such a run is no sample.
    const bool had_cache = g.exec_cache[static_cast<int>(s)] != nullptr;
    const auto t0 = Clock::now();
    int64_t calls = 0;
    if (s == Strategy::native) {
      for (int k : g.ops) ops[k].native->run(strm, ops[k].host_async);
      calls = static_cast<int64_t>(g.ops.size());
    } else {
      calls = ex.run(s, g, ops, strm);
    }
    if (measure && !had_cache && g.exec_cache[static_cast<int>(s)] != nullptr) {
      ex.release(a);
      tuner.discard(g.bucket, s);
    } else if (measure) {
      // Submitting the ops into the engine is paid whatever runs them: torch's
      // copy is only chosen after interception (or the explicit call) went
      // through it. Counting it on one side only biased the A/B toward native.
      double submit = 0;
      for (int k : g.ops) submit += ops[k].enqueue_us;
      const int64_t b = ex.mark(strm, device_of(g));
      samples.push_back({g.bucket, s, us_since(t0) + submit, submit, a, b, &ex});
    }
    note_pending(g.src, g.dst, strm,
                 std::any_of(g.ops.begin(), g.ops.end(), [&](int k) { return !ops[k].host_async; }));
    if (st) {
      st->calls += calls;
      st->bytes += g.bytes();
      st->strategies.emplace_back(strategy_name(s));
    }
    return calls;
  }

  // The hot depth-0 case: the same lone op as last time, its strategy locked.
  // Runs it without hashing, tuner lookups, key statistics or stats strings.
  // It hands one run in online_every back to the tuner, with the count of
  // runs it served, so the decision keeps being checked.
  struct Memo {
    bool valid = false;
    Hash128 fp;
    std::shared_ptr<const Plan> plan;
    Strategy strategy = Strategy::passthrough;
    int64_t served = 0;
  } memo;

  bool flush_memo() {
    if (!memo.valid || queue.size() != 1) return false;
    Op& op = queue.front();
    if (!(op.fp == memo.fp) || (memo.strategy == Strategy::native && !op.native)) return false;
    if (opt.online_every > 0 && memo.served + 1 >= opt.online_every) {
      op.weight += memo.served;
      memo.valid = false;
      return false;
    }
    ++memo.served;
    const Group& g = memo.plan->groups.front();
    const int64_t calls = run_group(g, memo.strategy, queue, stream, false, nullptr);
    keep(queue, stream);
    if (memo.strategy != Strategy::native) bury(std::move(op.native));
    ++total_flushes, ++total_ops, ++plan_hits;
    total_calls += calls;
    total_bytes += g.bytes();
    if (last_ops && last_ops.use_count() == 1 && last_ops->size() == 1) {
      bury(std::move((*last_ops)[0].native));
      (*last_ops)[0] = std::move(op);
    } else {
      bury(std::move(last_ops));
      last_ops = std::make_shared<std::vector<Op>>(1, std::move(op));
    }
    if (last_ran.plan != memo.plan) last_ran = Ran{memo.plan, {memo.strategy}};
    queue.clear();
    q_reads.clear();
    q_writes.clear();
    return true;
  }

  void flush() {
    if (queue.empty()) return;
    const Capture cap(*this, stream);
    if (!capturing) {
      if (flush_memo()) return;
      poll_samples();
    }
    std::vector<Op> ops;
    ops.swap(queue);
    q_reads.clear();
    q_writes.clear();
    Hasher h;
    for (const Op& op : ops) h.add(op.fp.a), h.add(op.fp.b);
    const Hash128 key = h.done();
    const std::shared_ptr<const Plan>* found = plans.find(key);
    const bool hit = found != nullptr;
    std::shared_ptr<const Plan> pp;
    double plan_us = 0;
    if (hit) {
      pp = *found;
    } else {
      const auto tp = Clock::now();
      auto p = std::make_shared<Plan>(build_plan(ops));
      for (Group& g : p->groups) {
        // Candidates may initialize the device (allocations): not while capturing,
        // and a plan built then is not kept.
        if (g.bandwidth > 0) g.candidates = {Strategy::paced};
        else if (!capturing) g.candidates = tuner.shortlist(exec_for(g.src, g.dst).candidates(g));
        g.bucket = bucket_key(g, static_cast<int>(g.ops.size()));
      }
      if (opt.log_plan) log(*p, ops.size());
      if (!capturing) {
        plans.insert(key, p);
      }
      pp = std::move(p);
      plan_us = us_since(tp);
      total_plan_us += plan_us;
    }
    const Plan& plan = *pp;
    FlushStats st;
    st.plan_us = plan_us;
    st.ops = static_cast<int>(ops.size());
    st.waves = plan.waves;
    st.groups = static_cast<int>(plan.groups.size());
    st.plan_hit = hit;
    Ran ran{pp, {}};
    ran.strategies.reserve(plan.groups.size());
    memo.valid = false;
    for (const Group& g : plan.groups) {
      const Choice c = choice_for(g, ops);
      Tuner::Pick p{c.baseline, false};
      int64_t runs = 1;
      for (int k : g.ops) runs = std::max(runs, ops[k].weight);
      // A bandwidth-limited group is not raced against full-speed copies: the
      // executor paces it (its objective is the limit, not the fastest time).
      if (capturing) p = {capture_choice(g, ops, opt.disable ? std::optional<Strategy>(c.baseline) : std::nullopt), false};
      else if (g.bandwidth > 0) p = {Strategy::paced, false};
      else if (!opt.disable) p = tuner.pick(g.bucket, c, runs);
      if (!capturing && ops.size() == 1 && plan.groups.size() == 1 && !p.measure &&
          (opt.disable || g.bandwidth > 0 || tuner.locked(g.bucket)))
        memo = {true, ops.front().fp, pp, p.strategy, 0};
      run_group(g, p.strategy, ops, stream, p.measure, &st);
      tuning_runs += p.measure;
      ran.strategies.push_back(p.strategy);
      for (int k : g.ops) {
        if (capturing) break;
        KeyStat& ks = keys[ops[k].key];
        ++ks.seen;
        ks.merged += g.ops.size() > 1;
      }
      if (g.ops.size() == 1)
        if (auto l = tuner.locked_strategy(g.bucket)) keys[ops[g.ops[0]].key].single = *l == Strategy::native;
    }
    // Payloads hold the frontend's tensors: release them where the decision
    // is a ce strategy. Kept while undecided or native, so freeze() can replay
    // the reference.
    keep(ops, stream);
    for (const Group& g : plan.groups) {
      const auto l = tuner.locked_strategy(g.bucket);
      if (l && *l != Strategy::native)
        for (int k : g.ops) bury(std::move(ops[k].native));
    }
    last = st;
    last_capture = capturing;
    ++total_flushes;
    total_ops += st.ops;
    total_calls += st.calls;
    total_bytes += st.bytes;
    plan_hits += hit;
    bury(std::move(last_ops));
    last_ops = std::make_shared<std::vector<Op>>(std::move(ops));
    last_ran = std::move(ran);
  }

  void synchronize() {
    if (cuda_streams.empty()) return;
    cuda->sync(cuda_streams);
    cuda_streams.clear();
    pending_at.clear();
    reap(true);
    host_pending = false;
    poll_samples();
  }

  // End of a top-level copy or scope. Device-only work stays queued on the
  // caller's stream, which orders it like torch's own copies.
  // `strm`: the stream of the work that just ended.
  void finish(uintptr_t strm) {
    if (capturing) return;  // recorded, not run: nothing to wait for
    if (!(opt.sync_always || (host_pending && !opt.allow_async))) return;
    if (is_capturing(strm)) return;  // earlier work stays pending: no waiting during a capture
    synchronize();
  }

  // Sets `capturing` for one flush or replay on `strm`.
  struct Capture {
    Impl& m;
    bool was;
    Capture(Impl& impl, uintptr_t strm) : m(impl), was(impl.capturing) { m.capturing = m.is_capturing(strm); }
    ~Capture() { m.capturing = was; }
  };
};

namespace {
// The engine lock. When the outermost holder lets go, the graveyard is
// emptied after unlocking (see Impl::graveyard).
class Locked {
 public:
  explicit Locked(Engine::Impl& m) : m_(m), lk_(m.mu) { ++m_.lock_depth; }
  ~Locked() {
    std::vector<std::shared_ptr<void>> drop;
    if (--m_.lock_depth == 0) drop.swap(m_.graveyard);
    lk_.unlock();
  }  // `drop` is destroyed here, unlocked

 private:
  Engine::Impl& m_;
  std::unique_lock<std::recursive_mutex> lk_;
};
}  // namespace

Engine::Engine(EngineOptions options) : impl_(std::make_unique<Impl>(options)) {}
Engine::~Engine() {
  try {
    Locked lk(*impl_);
    impl_->flush();
    impl_->synchronize();
  } catch (...) {
  }
  // Plans own device tables freed through the executor: drop them while it
  // still exists (members would otherwise outlive it and leak the memory).
  Impl& m = *impl_;
  m.memo = {};
  m.last_ran = {};
  m.last_ops.reset();
  m.stmt_cache.clear();
  m.plans.clear();
  m.graveyard.clear();
}

void Engine::copy(const Endpoint& src, const Endpoint& dst, const Runs& runs, const Elementwise& fn,
                  const Hints& hints, uintptr_t stream) {
  const auto t0 = Clock::now();
  Locked lk(*impl_);
  impl_->check_thread();
  impl_->enqueue(src, dst, runs, fn, hints, stream, t0);
}

void Engine::copy(const Endpoint& src, const Endpoint& dst, const Stmt& stmt, const Elementwise& fn,
                  const Hints& hints, uintptr_t stream) {
  const auto t0 = Clock::now();
  Locked lk(*impl_);
  impl_->check_thread();
  impl_->enqueue_stmt(src, dst, stmt, fn, hints, stream, t0);
}

void Engine::tile(const TileSpec& spec, const Hints& hints, uintptr_t stream) {
  const auto t0 = Clock::now();
  Locked lk(*impl_);
  impl_->check_thread();
  impl_->enqueue_tile(spec, hints, stream, t0);
}

void Engine::fence(std::span<const Range> reads, std::span<const Range> writes) {
  Locked lk(*impl_);
  impl_->fence(reads, writes);
}

void Engine::decline(std::string_view reason) {
  Locked lk(*impl_);
  auto it = impl_->declined.find(reason);
  if (it == impl_->declined.end()) it = impl_->declined.emplace(std::string(reason), 0).first;
  ++it->second;
}

void Engine::begin_scope() { begin_scope(false); }

void Engine::begin_scope(bool host_async) { begin_scope(host_async, 0); }

void Engine::begin_scope(bool host_async, double bandwidth) {
  if (!(bandwidth >= 0)) fail(Errc::invalid_argument, "bandwidth must be >= 0 bytes per second");
  Locked lk(*impl_);
  impl_->check_thread();
  if (impl_->depth++ == 0) impl_->owner = std::this_thread::get_id();
  impl_->scope_async.push_back(host_async);
  impl_->scope_bw.push_back(bandwidth);
  impl_->async_depth += host_async;
}

int64_t Engine::mark_pending() {
  Locked lk(*impl_);
  Impl& m = *impl_;
  if (m.pending_at.empty() || !m.cuda) return -1;
  std::vector<int64_t> marks;
  for (auto [st, dev] : m.pending_at) marks.push_back(m.cuda->mark(st, dev));
  m.pending_at.clear();  // later work gets a later token
  const int64_t t = m.next_token++;
  m.tokens.emplace(t, std::move(marks));
  return t;
}

bool Engine::done(int64_t token) {
  Locked lk(*impl_);
  Impl& m = *impl_;
  auto it = m.tokens.find(token);
  if (it == m.tokens.end()) return true;
  for (int64_t mk : it->second)
    if (!m.cuda->ready(mk)) return false;
  for (int64_t mk : it->second) m.cuda->release(mk);
  m.tokens.erase(it);
  return true;
}

void Engine::end_scope() {
  Locked lk(*impl_);
  impl_->check_thread();
  if (impl_->depth == 0) fail(Errc::invalid_argument, "end_scope without begin_scope");
  impl_->async_depth -= impl_->scope_async.back();
  impl_->scope_async.pop_back();
  impl_->scope_bw.pop_back();
  if (--impl_->depth == 0) {
    impl_->flush();
    impl_->finish(impl_->stream);
  }
}

void Engine::flush() {
  Locked lk(*impl_);
  impl_->flush();
}

void Engine::synchronize() {
  Locked lk(*impl_);
  impl_->synchronize();
}

int64_t Engine::tuning_runs() const {
  Locked lk(*impl_);
  return impl_->tuning_runs;
}

int Engine::depth() const {
  Locked lk(*impl_);
  return impl_->depth;
}

size_t Engine::queued() const {
  Locked lk(*impl_);
  return impl_->queue.size();
}

const EngineOptions& Engine::options() const { return impl_->opt; }

FlushStats Engine::last() const {
  Locked lk(*impl_);
  return impl_->last;
}

std::string Engine::stats_json() const {
  Locked lk(*impl_);
  const Impl& m = *impl_;
  Json strategies = Json::array();
  for (auto& s : m.last.strategies) strategies.push(s);
  Json last = Json::object()
                  .set("ops", m.last.ops)
                  .set("waves", m.last.waves)
                  .set("groups", m.last.groups)
                  .set("calls", m.last.calls)
                  .set("bytes", m.last.bytes)
                  .set("plan_cache", m.last.plan_hit ? "hit" : "miss")
                  .set("plan_us", m.last.plan_us)
                  .set("strategies", std::move(strategies));
  // Why each group of the last flush ran what it did (worked out here, not per flush).
  Json decisions = Json::array();
  if (m.last_ran.plan && m.last_ops) {
    const auto& groups = m.last_ran.plan->groups;
    for (size_t i = 0; i < groups.size() && i < m.last_ran.strategies.size(); ++i) {
      const Group& g = groups[i];
      const Strategy s = m.last_ran.strategies[i];
      std::string why;
      if (m.last_capture) why = "CUDA graph capture: a strategy that only enqueues stream work";
      else if (m.opt.disable) why = "CE_DISABLE";
      else if (g.bandwidth > 0) why = "bandwidth limit: paced on the GPU, not raced against full speed";
      else why = m.tuner.explain(g.bucket, m.choice_for(g, *m.last_ops), s);
      decisions.push(Json::object()
                         .set("strategy", std::string(strategy_name(s)))
                         .set("why", why)
                         .set("ops", static_cast<uint64_t>(g.ops.size()))
                         .set("bytes", g.bytes())
                         .set("bucket", g.bucket));
    }
  }
  last.set("decisions", std::move(decisions));
  Json declined = Json::object();
  for (const auto& [k, v] : m.declined) declined.set(k, v);
  return Json::object()
      .set("last", std::move(last))
      .set("flushes", m.total_flushes)
      .set("ops", m.total_ops)
      .set("calls", m.total_calls)
      .set("bytes", m.total_bytes)
      .set("plan_hits", m.plan_hits)
      .set("plan_us", m.total_plan_us)
      .set("native_ops", m.native_ops)
      .set("declined", std::move(declined))
      .set("kept", static_cast<uint64_t>(m.kept.size()))  // in-flight copies whose buffers are held
      .set("plans", static_cast<uint64_t>(m.plans.size()))
      .set("buckets", static_cast<uint64_t>(m.tuner.size()))
      .set("cuda", m.cuda ? m.cuda->info() : Json(m.cuda_error))
      .dump();
}

std::string Engine::dump_profile() const {
  Locked lk(*impl_);
  Json t = impl_->tuner.dump();
  return Json::object()
      .set("copyengine_profile", 2)
      .set("version", t["version"])
      .set("hardware", hardware_fingerprint())
      .set("buckets", t["buckets"])
      .set("uncovered", t["uncovered"])
      .dump();
}

void Engine::load_profile(const std::string& text) {
  Json j;
  try {
    j = Json::parse(text);
  } catch (const std::exception& e) {
    fail(Errc::invalid_argument, std::string("profile: ") + e.what());
  }
  if (j["copyengine_profile"].is_null()) fail(Errc::invalid_argument, "profile: not a copyengine profile");
  if (!j["copyengine_profile"].is_number() || j["copyengine_profile"].num() != 2)
    fail(Errc::invalid_argument, "profile: unsupported version (re-run warmup)");
  if (!j["hardware"].is_string() || j["hardware"].str() != hardware_fingerprint())
    fail(Errc::invalid_argument, "profile: recorded on different hardware");
  Locked lk(*impl_);
  impl_->tuner.load(j);
  impl_->memo.valid = false;  // locked choices may have changed
}

namespace {
class FrozenImpl final : public Frozen {
 public:
  FrozenImpl(Engine::Impl* eng, std::shared_ptr<std::vector<Op>> ops, Ran ran)
      : eng_(eng), ops_(std::move(ops)), ran_(std::move(ran)) {}
  void run(uintptr_t stream) override { run(stream, {}); }
  void run(uintptr_t stream, std::span<const Rebind> rebinds) override {
    Locked lk(*eng_);
    const std::vector<Op>* ops = ops_.get();
    if (!rebinds.empty()) {
      if (work_.empty()) work_ = *ops_;
      auto remap = [&](uintptr_t p) {
        for (const Rebind& r : rebinds)
          if (p >= r.lo && p < r.hi) return r.to + (p - r.lo);
        return p;
      };
      for (size_t k = 0; k < work_.size(); ++k) {
        const Op& op = (*ops_)[k];
        if (op.opaque) work_[k].stab = remap(op.stab), work_[k].dtab = remap(op.dtab);
        if (op.tile && op.tile_dev_index) {  // a tile op's GPU index tables
          auto t = std::make_shared<cv::TileK>(*op.tile);
          if (t->xi && !op.tile_index[0] && !op.tile_host_ptr[0])
            t->xi = static_cast<long long>(remap(static_cast<uintptr_t>(t->xi)));
          if (t->ci && !op.tile_index[1] && !op.tile_host_ptr[1])
            t->ci = static_cast<long long>(remap(static_cast<uintptr_t>(t->ci)));
          work_[k].tile = std::move(t);
        }
      }
      ops = &work_;
    }
    const Engine::Impl::Capture cap(*eng_, stream);
    const auto& groups = ran_.plan->groups;
    for (size_t i = 0; i < groups.size(); ++i) {
      const Strategy s = eng_->capturing ? eng_->capture_choice(groups[i], *ops, ran_.strategies[i]) : ran_.strategies[i];
      eng_->run_group(groups[i], s, *ops, stream, false, nullptr);
    }
    eng_->finish(stream);
  }
  std::vector<uintptr_t> tables() const override {
    std::vector<uintptr_t> t;
    for (const Op& op : *ops_) {
      if (op.stab) t.push_back(op.stab);
      if (op.dtab) t.push_back(op.dtab);
      if (op.tile && op.tile_dev_index) {  // GPU index tables of tile ops (host ones are values in the plan)
        if (op.tile->xi && !op.tile_index[0] && !op.tile_host_ptr[0]) t.push_back(static_cast<uintptr_t>(op.tile->xi));
        if (op.tile->ci && !op.tile_index[1] && !op.tile_host_ptr[1]) t.push_back(static_cast<uintptr_t>(op.tile->ci));
      }
    }
    return t;
  }
  std::vector<std::array<int64_t, 4>> allocations() const override {
    std::vector<std::array<int64_t, 4>> a;
    auto add = [&](const Side& s, int64_t lo, int64_t hi) {
      std::array<int64_t, 4> x{static_cast<int64_t>(s.device.kind), s.device.index, lo, hi};
      if (std::find(a.begin(), a.end(), x) == a.end()) a.push_back(x);
    };
    for (const Op& op : *ops_) add(op.src, op.src_lo, op.src_hi), add(op.dst, op.dst_lo, op.dst_hi);
    return a;
  }

 private:
  Engine::Impl* eng_;
  std::shared_ptr<std::vector<Op>> ops_;
  std::vector<Op> work_;  // ops_ with this call's table pointers
  Ran ran_;
};
}  // namespace

std::shared_ptr<Frozen> Engine::freeze() const { return freeze(false); }

void Engine::set_recording(bool on) {
  Locked lk(*impl_);
  impl_->recording = on;
}

int64_t Engine::flushes() const {
  Locked lk(*impl_);
  return impl_->total_flushes;
}

std::shared_ptr<Frozen> Engine::freeze(bool rebindable) const {
  Locked lk(*impl_);
  if (!impl_->last_ops) fail(Errc::invalid_argument, "freeze: nothing has been flushed yet");
  // What each group should run from now on, not what the last flush happened
  // to run (an exploration or online sample): the locked strategy, else the
  // reference, else what ran. Native needs every op's frontend payload.
  const Impl& m = *impl_;
  const auto& ops = *m.last_ops;
  Ran ran = m.last_ran;
  for (size_t i = 0; i < ran.plan->groups.size(); ++i) {
    const Group& g = ran.plan->groups[i];
    auto runnable = [&](Strategy s) {
      return s != Strategy::native || std::all_of(g.ops.begin(), g.ops.end(), [&](int k) { return ops[k].native; });
    };
    const Choice c = m.choice_for(g, ops);
    if (auto l = m.tuner.locked_strategy(g.bucket); l && runnable(*l)) ran.strategies[i] = *l;
    else if (runnable(c.reference())) ran.strategies[i] = c.reference();
    if (rebindable && g.opaque() && ran.strategies[i] == Strategy::native) {
      if (c.cands.empty()) fail(Errc::unsupported, "freeze: a copy with device indices has no strategy but torch's");
      ran.strategies[i] = c.cands.front();
    }
  }
  return std::make_shared<FrozenImpl>(impl_.get(), impl_->last_ops, std::move(ran));
}

}  // namespace ce
