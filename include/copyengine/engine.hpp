#pragma once

#include <cstdint>
#include <memory>
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
};
inline constexpr int kNumStrategies = 9;
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
  int tune_trials = 3;                   // samples per speculative candidate before the A/B
  int tune_topk = 3;                     // speculative candidates explored
  int ab_pairs = 20;                     // paired samples before an A/B decision
  int ab_max_pairs = 50;                 // an inconclusive A/B keeps sampling up to this
  bool allow_async = false;              // scope exit may return before host-side copies finish
  bool sync_always = false;              // wait after every flush, also device-only (CE_SYNC; debugging)
  bool disable = false;                  // every op natively, else as submitted (CE_DISABLE)
  bool log_plan = false;                 // print plans to stderr (CE_LOG=plan)
  int host_threads = 0;                  // 0: min(hardware threads, 8)
  int plan_cache = 4096;                 // max cached plans

  // Applies CE_DISABLE, CE_LOG, CE_STAGING_MB, CE_EXPLORE, CE_SYNC on top of `base`.
  static EngineOptions from_env(EngineOptions base);
  static EngineOptions from_env();
};

struct FlushStats {
  int ops = 0, waves = 0, groups = 0;
  int64_t calls = 0;     // backend calls issued
  int64_t bytes = 0;
  bool plan_hit = false;
  std::vector<std::string> strategies;
};

class Frozen;

// The frontend's own implementation of a copy (for torch: aten::copy_), used as
// the no-regression baseline. Each op carries its payload in Hints::native.
class NativeRunner {
 public:
  virtual ~NativeRunner() = default;
  // Runs one op on `stream`, ordered like any other work on it.
  virtual void run(const std::shared_ptr<void>& payload, uintptr_t stream) = 0;
};

// A byte range of one allocation, for fence().
struct Range {
  Device device;
  uintptr_t lo = 0, hi = 0;
};

// Lazy copy queue. Not thread-safe across scopes: give each thread its own
// engine, or serialize access (methods take an internal lock).
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

  // Enables the native baseline for ops whose Hints::native is set.
  void set_native(std::shared_ptr<NativeRunner> runner);

  // Before other work reads `reads` and writes `writes` (whole allocations):
  // issues queued copies that conflict with them. Free when nothing is queued.
  void fence(std::span<const Range> reads, std::span<const Range> writes);

  void begin_scope();
  // Flushes at depth 0. Copies are ordered on the caller's stream; the call
  // waits only when host memory is involved (unless allow_async).
  void end_scope();
  void flush();         // plan + issue everything queued
  void synchronize();   // wait for all issued work, device-only copies included
  int depth() const;
  size_t queued() const;

  // Tuning table, keyed by hardware fingerprint inside the JSON.
  std::string dump_profile() const;
  void load_profile(const std::string& json);

  FlushStats last() const;
  std::string stats_json() const;
  const EngineOptions& options() const;

  // Freezes the last flush: replay without planning or checks. The caller
  // guarantees the same buffers, layouts and tables stay valid.
  std::shared_ptr<Frozen> freeze() const;

  struct Impl;

 private:
  std::unique_ptr<Impl> impl_;
};

class Frozen {
 public:
  virtual ~Frozen() = default;
  virtual void run(uintptr_t stream = 0) = 0;
};

}  // namespace ce
