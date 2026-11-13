#include "copyengine/engine.hpp"

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
  if (auto v = env("CE_TUNE"); !v.empty()) o.tune = truthy(v);
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
  HostExecutor host;
  std::unique_ptr<Executor> cuda;
  std::string cuda_error;
  bool cuda_pending = false;
  uintptr_t cuda_stream = 0;
  FlushStats last;
  int64_t total_flushes = 0, total_ops = 0, total_calls = 0, total_bytes = 0, plan_hits = 0;
  std::shared_ptr<std::vector<Op>> last_ops;
  Ran last_ran;

  explicit Impl(EngineOptions o)
      : opt(o),
        tuner(o.tune ? o.tune_trials : 0, o.tune_topk),
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

  void push(Op op, uintptr_t strm) {
    if (op.runs().empty()) return;
    if (!queue.empty() && strm != stream) flush();
    stream = strm;
    queue.push_back(std::move(op));
    if (depth == 0) {
      flush();
      if (!opt.allow_async) synchronize();
    }
  }

  void enqueue(const Endpoint& src, const Endpoint& dst, Runs runs, const Elementwise& fn, const Hints& hints,
               uintptr_t strm) {
    push(make_op(src, dst, std::move(runs), fn, hints), strm);
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
                    const Hints& hints, uintptr_t strm) {
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
      push(it->second, strm);
      return;
    }
    std::shared_ptr<const Form> form = hints.has_valid ? nullptr : make_form(src, dst, stmt);
    Op op = make_op(src, dst, lower(stmt), fn, hints, std::move(form));
    if (static_cast<int>(stmt_cache.size()) >= opt.plan_cache * 4) stmt_cache.clear();
    stmt_cache.emplace(key, op);
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

  void run_group(const Group& g, Strategy s, const std::vector<Op>& ops, uintptr_t strm, bool measure,
                 FlushStats* st) {
    Executor& ex = exec_for(g.src, g.dst);
    if (&ex == &host && cuda_pending) {
      cuda->sync(cuda_stream);  // host work may read what earlier CUDA work wrote
      cuda_pending = false;
    }
    const auto t0 = Clock::now();
    const int64_t calls = ex.run(s, g, ops, strm);
    if (measure) {
      ex.sync(strm);
      tuner.report(g.bucket, s, std::chrono::duration<double>(Clock::now() - t0).count());
    }
    if (ex.async()) {
      cuda_pending = true;
      cuda_stream = strm;
    }
    if (st) {
      st->calls += calls;
      st->bytes += g.bytes();
      st->strategies.emplace_back(strategy_name(s));
    }
  }

  void flush() {
    if (queue.empty()) return;
    std::vector<Op> ops;
    ops.swap(queue);
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
    for (const Group& g : plan.groups) {
      const Strategy s = opt.disable ? Strategy::passthrough : tuner.pick(g.bucket, g.candidates);
      const bool measure = !opt.disable && tuner.exploring(g.bucket);
      run_group(g, s, ops, stream, measure, &st);
      ran.strategies.push_back(s);
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
    if (cuda_pending) {
      cuda->sync(cuda_stream);
      cuda_pending = false;
    }
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
  std::lock_guard lk(impl_->mu);
  impl_->enqueue(src, dst, runs, fn, hints, stream);
}

void Engine::copy(const Endpoint& src, const Endpoint& dst, const Stmt& stmt, const Elementwise& fn,
                  const Hints& hints, uintptr_t stream) {
  std::lock_guard lk(impl_->mu);
  impl_->enqueue_stmt(src, dst, stmt, fn, hints, stream);
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
    if (!impl_->opt.allow_async) impl_->synchronize();
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
      .set("plans", static_cast<uint64_t>(m.plans.size()))
      .set("buckets", static_cast<uint64_t>(m.tuner.size()))
      .set("cuda", m.cuda ? m.cuda->info() : Json(m.cuda_error))
      .dump();
}

std::string Engine::dump_profile() const {
  std::lock_guard lk(impl_->mu);
  return Json::object()
      .set("copyengine_profile", 1)
      .set("hardware", hardware_fingerprint())
      .set("buckets", impl_->tuner.dump())
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
  if (!j["hardware"].is_string() || j["hardware"].str() != hardware_fingerprint())
    fail(Errc::invalid_argument, "profile: recorded on different hardware");
  std::lock_guard lk(impl_->mu);
  impl_->tuner.load(j["buckets"]);
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
    if (!eng_->opt.allow_async) eng_->synchronize();
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
  return std::make_shared<FrozenImpl>(impl_.get(), impl_->last_ops, impl_->last_ran);
}

}  // namespace ce
