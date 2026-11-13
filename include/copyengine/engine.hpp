#pragma once

#include <cstdint>
#include <memory>
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
  staged,         // CUDA: raw copy to/from device staging + kernel on the device side
};
inline constexpr int kNumStrategies = 6;
std::string_view strategy_name(Strategy s);
Strategy parse_strategy(std::string_view name);

struct EngineOptions {
  int64_t staging_budget = 256ll << 20;  // bytes of device staging, never exceeded
  bool tune = true;                      // explore candidates per bucket, then lock
  int tune_trials = 3;                   // samples per candidate before locking
  int tune_topk = 3;                     // candidates explored (plus passthrough)
  bool allow_async = false;              // scope exit may return before copies finish
  bool disable = false;                  // run every op as submitted (CE_DISABLE)
  bool log_plan = false;                 // print plans to stderr (CE_LOG=plan)
  int host_threads = 0;                  // 0: min(hardware threads, 8)
  int plan_cache = 4096;                 // max cached plans

  // Applies CE_DISABLE, CE_LOG, CE_STAGING_MB, CE_TUNE on top of `base`.
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

  void begin_scope();
  void end_scope();     // flushes at depth 0, then waits unless allow_async
  void flush();         // plan + issue everything queued
  void synchronize();   // wait for issued work
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
