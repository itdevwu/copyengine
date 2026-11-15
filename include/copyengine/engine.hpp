#pragma once

#include <cstdint>
#include <memory>
#include <array>
#include <span>
#include <string>
#include <vector>

#include "copyengine/ir.hpp"
#include "copyengine/types.hpp"

namespace ce {

enum class Strategy : uint8_t {
  passthrough,    // each op's runs exactly as submitted: the no-regression baseline
  runs,           // merged runs / 2-D blocks, one call each
  runs_parallel,  // host: merged runs split across worker threads
  batch,          // CUDA: one cuMemcpyBatchAsync (+ 2-D copies)
  kernel,         // CUDA: table-driven gather/scatter kernel (with cast / affine)
  staged,         // CUDA: dense raw copy to/from device staging + kernel on the device side
  bounce,         // CUDA: pageable host side packed through pinned buffers on the CPU
  fused,          // CUDA: structured statements (strides, broadcast) evaluated by one kernel
  native,         // each op through the frontend's own implementation (torch's copy_): the baseline
  paced,          // CUDA: a bandwidth-limited group in chunks, a GPU pacing kernel before each (QoS)
  lowered,        // CUDA: host index tables expanded into DMA descriptors at each run
};
inline constexpr int kNumStrategies = 11;
std::string_view strategy_name(Strategy s);
Strategy parse_strategy(std::string_view name);

struct EngineOptions {
  int64_t staging_budget = 256ll << 20;  // bytes of device staging, never exceeded
  // Measure speculative strategies against the baseline and keep one only if
  // it is significantly faster. false (CE_EXPLORE=0): use the loaded profile;
  // buckets it does not cover take the dominance-safe choice or the baseline
  // and are recorded for the next warmup.
  bool explore = true;
  bool prior_only = false;               // testing: run the prior's first candidate, never measure
  int tune_trials = 5;                   // samples per speculative candidate before the A/B
  int tune_topk = 3;                     // speculative candidates explored
  int ab_pairs = 20;                     // paired samples before an A/B decision
  int ab_max_pairs = 50;                 // an inconclusive A/B keeps sampling up to this
  // Once locked, one run in `online_every` is timed (never synchronizing) to
  // keep checking the decision; 0 turns this off.
  int online_every = 64;
  bool allow_async = false;              // scope exit may return before host-side copies finish
  bool sync_always = false;              // wait after every flush, also device-only (CE_SYNC; debugging)
  // Copy kernels use at most this many SMs per GPU (0: no limit; CE_MAX_SMS).
  // Enforced by a green context where the driver confines it (verified on
  // start), else by launching at most this many thread blocks. DMA copies use
  // the copy engines, not SMs, and are not limited.
  int max_sms = 0;
  // Copies touching a GPU run at most this many bytes per second (0: no
  // limit), paced on the GPU; a scope's own limit (begin_scope) overrides it.
  double bandwidth = 0;
  bool disable = false;                  // every op natively, else as submitted (CE_DISABLE)
  bool log_plan = false;                 // print plans to stderr (CE_LOG=plan)
  int host_threads = 0;                  // 0: min(hardware threads, 8)
  int plan_cache = 4096;                 // max cached plans

  // Applies CE_DISABLE, CE_LOG, CE_STAGING_MB, CE_EXPLORE, CE_SYNC, CE_MAX_SMS on top of `base`.
  static EngineOptions from_env(EngineOptions base);
  static EngineOptions from_env();
};

struct FlushStats {
  int ops = 0, waves = 0, groups = 0;
  int64_t calls = 0;     // backend calls issued
  int64_t bytes = 0;
  bool plan_hit = false;
  double plan_us = 0;  // planning on a plan-cache miss (outside the strategies' measured times)
  std::vector<std::string> strategies;
};

class Frozen;

// A byte range of one allocation, for fence().
struct Range {
  Device device;
  uintptr_t lo = 0, hi = 0;
};

// Lazy copy queue. Methods take an internal lock, so threads may share an
// engine; but while one thread has a scope open, another thread's copies or
// scopes on the same engine fail (they would be queued into that scope):
// give each thread its own engine.
class Engine {
 public:
  explicit Engine(EngineOptions options = EngineOptions::from_env());
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  // L1 entry: runs relative to each endpoint's base, in elements.
  void copy(const Endpoint& src, const Endpoint& dst, const Runs& runs, const Elementwise& fn = {},
            const Hints& hints = {}, uintptr_t stream = 0);
  // L3 entry: structured statement; offsets relative to the endpoints' bases.
  void copy(const Endpoint& src, const Endpoint& dst, const Stmt& stmt, const Elementwise& fn = {},
            const Hints& hints = {}, uintptr_t stream = 0);

  // Before other work reads `reads` and writes `writes` (whole allocations):
  // issues queued copies that conflict with them. Free when nothing is queued.
  void fence(std::span<const Range> reads, std::span<const Range> writes);

  void begin_scope();
  // A scope whose copies the host never waits for, as if every one were
  // non_blocking: buffers stay alive until the GPU is done (Python's
  // scope(stream=...)). Scopes nest; one async scope makes its inner ones async.
  void begin_scope(bool host_async);
  // `bandwidth` > 0: copies queued in this scope (and inner scopes without
  // their own limit) run at most this many bytes per second, per GPU and
  // direction, as a paced background transfer instead of at full speed.
  void begin_scope(bool host_async, double bandwidth);
  // Flushes at depth 0. Copies are ordered on the caller's stream; the call
  // waits only when host memory is involved (unless allow_async).
  void end_scope();
  void flush();         // plan + issue everything queued
  void synchronize();   // wait for all issued work, device-only copies included
  // A token for the work issued so far (-1: none is pending), and whether it
  // has completed, without blocking. done() forgets the token once it is true.
  // Callers keep a copy's buffers alive until then (allow_async).
  int64_t mark_pending();
  bool done(int64_t token);

  // An intercepted copy the frontend left to the framework, and why (counted
  // per reason in stats_json()["declined"]).
  void decline(std::string_view reason);
  int depth() const;
  size_t queued() const;
  // Group runs so far that were tuning measurements (exploration or A/B), not
  // decided strategies: a frontend compares it before and after a region to
  // tell whether the region ran settled.
  int64_t tuning_runs() const;

  // Tuning table, keyed by hardware fingerprint inside the JSON.
  std::string dump_profile() const;
  void load_profile(const std::string& json);

  FlushStats last() const;
  std::string stats_json() const;
  const EngineOptions& options() const;

  // Freezes the last flush: replay without planning or checks. The caller
  // guarantees the same buffers, layouts and tables stay valid.
  std::shared_ptr<Frozen> freeze() const;
  // The same, replayable with other device index tables (Frozen::run's
  // rebinds): groups reading device tables never take the frontend's native
  // copy, whose payload holds the traced index tensors.
  std::shared_ptr<Frozen> freeze(bool rebindable) const;
  int64_t flushes() const;
  // While on, every copy goes through the queue (none runs as torch's copy at
  // interception time): a frontend tracing a region for freeze() sets it.
  void set_recording(bool on);  // flushes so far (a region that flushed more than once was not one plan)

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

// Device index tables a replay reads elsewhere: a table pointer in [lo, hi)
// at freeze time reads from `to + (pointer - lo)` instead.
struct Rebind {
  uintptr_t lo = 0, hi = 0, to = 0;
};

class Frozen {
 public:
  virtual ~Frozen() = default;
  virtual void run(uintptr_t stream = 0) = 0;
  virtual void run(uintptr_t stream, std::span<const Rebind> rebinds) = 0;
  // The device table pointers the frozen copies read (for frontends that map
  // them to their arguments).
  virtual std::vector<uintptr_t> tables() const = 0;
  // Every allocation the frozen copies touch: {device kind, device index, lo, hi} (bytes).
  virtual std::vector<std::array<int64_t, 4>> allocations() const = 0;
};

}  // namespace ce
