#pragma once

#include <array>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "copyengine/engine.hpp"
#include "core/json.hpp"

namespace ce {

// One timed execution: host time from enqueue to return, GPU time between
// stream events (0 for host-only work). Microseconds.
struct Sample {
  double host = 0, gpu = 0;
  double total() const { return host + gpu; }
};

// What a bucket may run. `baseline` is the reference implementation (the
// frontend's native copy, else passthrough); `safe` is a dominance-safe
// strategy applied without proof; `cands` are speculative strategies in prior
// order.
struct Choice {
  std::vector<Strategy> cands;
  Strategy baseline = Strategy::passthrough;
  std::optional<Strategy> safe;
};

struct TunerConfig {
  bool explore = true;   // measure; false: loaded profile only, uncovered buckets take the safe choice
  bool prior_only = false;  // testing: always the first candidate, never measure
  int trials = 3;        // samples per speculative candidate before the A/B
  int topk = 3;          // speculative candidates explored
  int pairs = 20;        // paired A/B samples before a decision
  int max_pairs = 50;    // keep sampling an inconclusive A/B up to this many pairs
  double margin = 0.05;  // the challenger's median total must be this much faster
};

// Per bucket: explore the speculative candidates, then A/B the best one
// against the reference (the safe strategy if there is one, else the
// baseline) in alternating, paired runs. The challenger wins only if neither
// host nor GPU time is worse, its median total is faster by max(margin, noise)
// and it wins a significant share of the pairs; ties keep the reference.
class Tuner {
 public:
  explicit Tuner(TunerConfig cfg) : cfg_(cfg) {}

  // Speculative candidates worth exploring (prior order, top-k).
  std::vector<Strategy> shortlist(std::vector<Strategy> prior) const;

  // The strategy to run next, and whether that run is a sample to report.
  struct Pick {
    Strategy strategy;
    bool measure = false;
  };
  Pick pick(const std::string& bucket, const Choice& c);
  void report(const std::string& bucket, Strategy s, Sample sample);
  bool locked(const std::string& bucket) const;

  Json dump() const;
  void load(const Json& buckets);
  size_t size() const { return table_.size(); }
  const TunerConfig& config() const { return cfg_; }

  // The A/B rule, exposed for tests: paired samples (same index = same pair).
  static bool challenger_wins(const std::vector<Sample>& challenger, const std::vector<Sample>& reference,
                              double margin);

 private:
  struct Entry {
    int locked = -1;
    int challenger = -1;  // best speculative candidate once exploration is done
    int issued = 0;       // A/B runs handed out (alternating)
    std::array<std::vector<Sample>, kNumStrategies> samples;
    std::array<int, kNumStrategies> pending{};  // issued, not yet reported
  };
  Strategy decide(Entry& e, const Choice& c);
  TunerConfig cfg_;
  std::unordered_map<std::string, Entry> table_;
  std::set<std::string> uncovered_;  // buckets met without a profile entry while not exploring
};

}  // namespace ce
