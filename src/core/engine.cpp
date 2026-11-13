#include "copyengine/engine.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
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

}  // namespace

struct Engine::Impl {
  EngineOptions opt;
  mutable std::recursive_mutex mu;
  int depth = 0;
  std::vector<Op> queue;
  uintptr_t stream = 0;
  std::unordered_map<Hash128, std::shared_ptr<const Plan>, Hash128Hasher> plans;
  std::unordered_set<Hash128, Hash128Hasher> safe;
  Tuner tuner;
  std::unordered_map<Hash128, KeyStat, Hash128Hasher> keys;
  std::vector<Alloc> q_reads, q_writes;  // allocations the queue reads / writes
  std::vector<PendingSample> samples;
  int64_t native_ops = 0;  // ops run natively at interception
  HostExecutor host;
  std::unique_ptr<Executor> cuda;
  std::string cuda_error;
  // Streams with issued, unfinished CUDA work, and whether any of it touches
  // host memory (only then must the host wait before returning or reading).
  std::vector<uintptr_t> cuda_streams;
  bool host_pending = false;
  FlushStats last;
  int64_t total_flushes = 0, total_ops = 0, total_calls = 0, total_bytes = 0, plan_hits = 0;
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
    if (!safe.count(op.fp)) {
      check_safe(op);
      if (static_cast<int>(safe.size()) >= opt.plan_cache * 4) safe.clear();
      safe.insert(op.fp);
    }
    return op;
  }

  // Structured form of a table-free statement, or null if it has none.
  static std::shared_ptr<const Form> make_form(const Endpoint& src, const Endpoint& dst, const Stmt& stmt) {
    if (stmt.src.has_table() || stmt.dst.has_table()) return nullptr;
    Form f;
    f.sdt = src.dtype, f.ddt = dst.dtype;
    f.s_base = static_cast<i64>(src.base / itemsize(src.dtype)) + stmt.src.base();
    f.d_base = static_cast<i64>(dst.base / itemsize(dst.dtype)) + stmt.dst.base();
    const auto &st = stmt.src.terms(), &dt = stmt.dst.terms();
    for (size_t k = 0; k < st.size(); ++k) {
      if (st[k].size == 1) continue;  // contributes nothing but its (already folded) base
      if (f.rank == Form::kMaxRank) return nullptr;
      f.shape[f.rank] = st[k].size, f.ss[f.rank] = st[k].stride, f.ds[f.rank] = dt[k].stride;
      f.numel *= st[k].size;
      ++f.rank;
    }
    if (f.numel <= 0 || f.numel >= (int64_t{1} << 31)) return nullptr;
    return std::make_shared<const Form>(f);
  }

  // Adjacent to the queue's last op on both allocations: merging it is a
  // dominance-safe transform, so it always queues.
  bool merges_with_tail(const Op& op) const {
    if (queue.empty()) return false;
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
    if (!op.native || opt.prior_only) return false;
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
  void note_pending(const Side& s, const Side& d, uintptr_t strm, bool host_wait) {
    if (s.device.kind == DeviceKind::host && d.device.kind == DeviceKind::host) return;
    if (std::find(cuda_streams.begin(), cuda_streams.end(), strm) == cuda_streams.end()) cuda_streams.push_back(strm);
    if (host_wait && (s.device.kind == DeviceKind::host || d.device.kind == DeviceKind::host)) host_pending = true;
  }

  void run_native_now(const Op& op, uintptr_t strm) {
    if (!queue.empty() && (strm != stream || conflicts(op.src.device, op.src_lo, op.src_hi, op.dst.device, op.dst_lo,
                                                      op.dst_hi)))
      flush();
    const bool host_only = op.src.device.kind == DeviceKind::host && op.dst.device.kind == DeviceKind::host;
    if (host_only && host_pending) synchronize();  // host work may touch what CUDA work still uses
    op.native->run(strm);
    note_pending(op.src, op.dst, strm, !op.host_async);
    ++native_ops;
    if (depth == 0) finish();
  }

  void push(Op op, uintptr_t strm) {
    if (op.runs().empty()) return;
    if (native_now(op)) {
      run_native_now(op, strm);
      return;
    }
    if (!queue.empty() && strm != stream) flush();
    stream = strm;
    add_alloc(q_reads, op.src.device, op.src_lo, op.src_hi);
    add_alloc(q_writes, op.dst.device, op.dst_lo, op.dst_hi);
    queue.push_back(std::move(op));
    if (depth == 0) {
      flush();
      finish();
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
  std::unordered_map<Hash128, Op, Hash128Hasher> stmt_cache;

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

  void enqueue_stmt(const Endpoint& src, const Endpoint& dst, const Stmt& stmt, const Elementwise& fn,
                    const Hints& hints, uintptr_t strm, Clock::time_point t0) {
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
    if (auto it = stmt_cache.find(key); it != stmt_cache.end()) {
      Op op = it->second;
      op.native = hints.native;
      op.host_async = hints.host_async;
      op.enqueue_us = us_since(t0);
      push(std::move(op), strm);
      return;
    }
    std::shared_ptr<const Form> form = hints.has_valid ? nullptr : make_form(src, dst, stmt);
    Op op = make_op(src, dst, lower(stmt), fn, hints, std::move(form));
    if (static_cast<int>(stmt_cache.size()) >= opt.plan_cache * 4) stmt_cache.clear();
    stmt_cache.emplace(key, op);
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
    return g.pure() && (host_only || !g.rects.has_2d()) && g.rects.size() * 2 <= g.ops.size();
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
      if (g.op_runs <= kPassthroughCalls) c.cands.push_back(Strategy::passthrough);
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
    if (&ex == &host && host_pending) synchronize();  // host work may touch what CUDA work still uses
    const int64_t a = measure ? ex.mark(strm, device_of(g)) : -1;
    const auto t0 = Clock::now();
    int64_t calls = 0;
    if (s == Strategy::native) {
      for (int k : g.ops) ops[k].native->run(strm);
      calls = static_cast<int64_t>(g.ops.size());
    } else {
      calls = ex.run(s, g, ops, strm);
    }
    if (measure) {
      double submit = 0;
      if (s != Strategy::native)  // ce also paid for submitting the ops; native would not have
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
    if (memo.strategy != Strategy::native) op.native.reset();
    ++total_flushes, ++total_ops, ++plan_hits;
    total_calls += calls;
    total_bytes += g.bytes();
    if (last_ops && last_ops.use_count() == 1 && last_ops->size() == 1) (*last_ops)[0] = std::move(op);
    else last_ops = std::make_shared<std::vector<Op>>(1, std::move(op));
    if (last_ran.plan != memo.plan) last_ran = Ran{memo.plan, {memo.strategy}};
    queue.clear();
    q_reads.clear();
    q_writes.clear();
    return true;
  }

  void flush() {
    if (queue.empty()) return;
    if (flush_memo()) return;
    poll_samples();
    std::vector<Op> ops;
    ops.swap(queue);
    q_reads.clear();
    q_writes.clear();
    Hasher h;
    for (const Op& op : ops) h.add(op.fp.a), h.add(op.fp.b);
    const Hash128 key = h.done();
    auto it = plans.find(key);
    const bool hit = it != plans.end();
    if (!hit) {
      auto p = std::make_shared<Plan>(build_plan(ops));
      for (Group& g : p->groups) {
        g.candidates = tuner.shortlist(exec_for(g.src, g.dst).candidates(g));
        g.bucket = bucket_key(g, static_cast<int>(g.ops.size()));
      }
      if (opt.log_plan) log(*p, ops.size());
      if (static_cast<int>(plans.size()) >= opt.plan_cache) plans.clear();
      it = plans.emplace(key, std::move(p)).first;
    }
    const std::shared_ptr<const Plan> pp = it->second;
    const Plan& plan = *pp;
    FlushStats st;
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
      if (!opt.disable) p = tuner.pick(g.bucket, c, runs);
      if (ops.size() == 1 && plan.groups.size() == 1 && !p.measure && (opt.disable || tuner.locked(g.bucket)))
        memo = {true, ops.front().fp, pp, p.strategy, 0};
      run_group(g, p.strategy, ops, stream, p.measure, &st);
      ran.strategies.push_back(p.strategy);
      for (int k : g.ops) {
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
    for (const Group& g : plan.groups) {
      const auto l = tuner.locked_strategy(g.bucket);
      if (l && *l != Strategy::native)
        for (int k : g.ops) ops[k].native.reset();
    }
    last = st;
    ++total_flushes;
    total_ops += st.ops;
    total_calls += st.calls;
    total_bytes += st.bytes;
    plan_hits += hit;
    last_ops = std::make_shared<std::vector<Op>>(std::move(ops));
    last_ran = std::move(ran);
  }

  void synchronize() {
    if (cuda_streams.empty()) return;
    cuda->sync(cuda_streams);
    cuda_streams.clear();
    host_pending = false;
    poll_samples();
  }

  // End of a top-level copy or scope. Device-only work stays queued on the
  // caller's stream, which orders it like torch's own copies.
  void finish() {
    if (opt.sync_always || (host_pending && !opt.allow_async)) synchronize();
  }
};

Engine::Engine(EngineOptions options) : impl_(std::make_unique<Impl>(options)) {}
Engine::~Engine() {
  try {
    std::lock_guard lk(impl_->mu);
    impl_->flush();
    impl_->synchronize();
  } catch (...) {
  }
}

void Engine::copy(const Endpoint& src, const Endpoint& dst, const Runs& runs, const Elementwise& fn,
                  const Hints& hints, uintptr_t stream) {
  const auto t0 = Clock::now();
  std::lock_guard lk(impl_->mu);
  impl_->enqueue(src, dst, runs, fn, hints, stream, t0);
}

void Engine::copy(const Endpoint& src, const Endpoint& dst, const Stmt& stmt, const Elementwise& fn,
                  const Hints& hints, uintptr_t stream) {
  const auto t0 = Clock::now();
  std::lock_guard lk(impl_->mu);
  impl_->enqueue_stmt(src, dst, stmt, fn, hints, stream, t0);
}

void Engine::fence(std::span<const Range> reads, std::span<const Range> writes) {
  std::lock_guard lk(impl_->mu);
  impl_->fence(reads, writes);
}

void Engine::begin_scope() {
  std::lock_guard lk(impl_->mu);
  ++impl_->depth;
}

void Engine::end_scope() {
  std::lock_guard lk(impl_->mu);
  if (impl_->depth == 0) fail(Errc::invalid_argument, "end_scope without begin_scope");
  if (--impl_->depth == 0) {
    impl_->flush();
    impl_->finish();
  }
}

void Engine::flush() {
  std::lock_guard lk(impl_->mu);
  impl_->flush();
}

void Engine::synchronize() {
  std::lock_guard lk(impl_->mu);
  impl_->synchronize();
}

int Engine::depth() const {
  std::lock_guard lk(impl_->mu);
  return impl_->depth;
}

size_t Engine::queued() const {
  std::lock_guard lk(impl_->mu);
  return impl_->queue.size();
}

const EngineOptions& Engine::options() const { return impl_->opt; }

FlushStats Engine::last() const {
  std::lock_guard lk(impl_->mu);
  return impl_->last;
}

std::string Engine::stats_json() const {
  std::lock_guard lk(impl_->mu);
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
                  .set("strategies", std::move(strategies));
  return Json::object()
      .set("last", std::move(last))
      .set("flushes", m.total_flushes)
      .set("ops", m.total_ops)
      .set("calls", m.total_calls)
      .set("bytes", m.total_bytes)
      .set("plan_hits", m.plan_hits)
      .set("native_ops", m.native_ops)
      .set("plans", static_cast<uint64_t>(m.plans.size()))
      .set("buckets", static_cast<uint64_t>(m.tuner.size()))
      .set("cuda", m.cuda ? m.cuda->info() : Json(m.cuda_error))
      .dump();
}

std::string Engine::dump_profile() const {
  std::lock_guard lk(impl_->mu);
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
  std::lock_guard lk(impl_->mu);
  impl_->tuner.load(j);
  impl_->memo.valid = false;  // locked choices may have changed
}

namespace {
class FrozenImpl final : public Frozen {
 public:
  FrozenImpl(Engine::Impl* eng, std::shared_ptr<std::vector<Op>> ops, Ran ran)
      : eng_(eng), ops_(std::move(ops)), ran_(std::move(ran)) {}
  void run(uintptr_t stream) override {
    std::lock_guard lk(eng_->mu);
    const auto& groups = ran_.plan->groups;
    for (size_t i = 0; i < groups.size(); ++i)
      eng_->run_group(groups[i], ran_.strategies[i], *ops_, stream, false, nullptr);
    eng_->finish();
  }

 private:
  Engine::Impl* eng_;
  std::shared_ptr<std::vector<Op>> ops_;
  Ran ran_;
};
}  // namespace

std::shared_ptr<Frozen> Engine::freeze() const {
  std::lock_guard lk(impl_->mu);
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
  }
  return std::make_shared<FrozenImpl>(impl_.get(), impl_->last_ops, std::move(ran));
}

}  // namespace ce
