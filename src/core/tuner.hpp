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
  double submit = 0;  // part of `host` spent submitting the ops (the same for every ce strategy)
  double total() const { return host + gpu; }
  // What tells ce strategies apart: their common submit cost is only noise here.
  double own() const { return host - submit + gpu; }
};

// What a bucket may run. `baseline` is what runs without copyengine's
// choices (the frontend's native copy, else passthrough); `safe` is a
// dominance-safe strategy applied without proof; `cands` are speculative
// strategies in prior order. `lead` is set when no native copy exists: there
// is no original to protect, so all candidates (lead first, passthrough only if
// cheap) are explored alike and the fastest locks without an A/B; the lead is
// what runs where nothing is measured. Otherwise the A/B reference is `safe`,
// else the baseline.
struct Choice {
  std::vector<Strategy> cands;
  Strategy baseline = Strategy::passthrough;
  std::optional<Strategy> safe;
  std::optional<Strategy> lead;
  Strategy reference() const { return lead ? *lead : safe ? *safe : baseline; }
};

struct TunerConfig {
  bool explore = true;   // measure; false: loaded profile only, uncovered buckets take the safe choice
  bool prior_only = false;  // testing: always the first candidate, never measure
  int trials = 5;        // samples per speculative candidate (round-robin) before the A/B
  int topk = 3;          // speculative candidates explored
  int pairs = 20;        // paired A/B samples before a decision
  int max_pairs = 50;    // keep sampling an inconclusive A/B up to this many pairs
  double margin = 0.05;  // the challenger's median total must be this much faster
  // Once locked: every `online_every`-th run is timed (0: never); every
  // `alt_every`-th of those runs the other finalist instead. A window of
  // `window` samples each decides a flip with the A/B rule; the locked side
  // drifting `drift` x slower than when it locked re-opens exploration.
  int online_every = 64;
  int alt_every = 4;
  int window = 8;
  double drift = 1.5;
};

// Per bucket: explore the speculative candidates round-robin, then A/B the
// best one against the reference (the safe strategy if there is one, else the
// baseline) in alternating, paired runs. The challenger wins only if neither
// host nor GPU time is worse, its median total is faster by max(margin, noise)
// and it wins a significant share of the pairs; ties keep the reference.
// After locking, low-frequency online samples keep checking both finalists
// (with the same rule, so flips need significant evidence) and re-open the
// bucket when the locked side drifts.
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
  // `runs`: executions this pick stands for (callers that skip the tuner for
  // a while report how many runs they skipped, so online sampling keeps pace).
  Pick pick(const std::string& bucket, const Choice& c, int64_t runs = 1);
  void report(const std::string& bucket, Strategy s, Sample sample);
  // A measured run that does not count (it paid one-time setup): the next
  // run of `s` is measured instead.
  void discard(const std::string& bucket, Strategy s);
  bool locked(const std::string& bucket) const;
  // Why `ran` ran for `bucket` (after pick): the evidence behind a locked
  // choice, or which phase of exploration a sample belongs to.
  std::string explain(const std::string& bucket, const Choice& c, Strategy ran) const;
  // The locked strategy (not what an online sample just ran), if locked.
  std::optional<Strategy> locked_strategy(const std::string& bucket) const;

  // Profiles record the copyengine version: entries from another version are
  // not trusted, and their buckets count as uncovered.
  Json dump() const;
  void load(const Json& profile);
  size_t size() const { return table_.size(); }
  // Buckets per locked strategy, those still exploring, and the online
  // re-checks' flips and reopens so far (stats()["locks"]).
  Json lock_counts() const;
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
    // online, once locked
    int alt = -1;                          // the other finalist
    bool alt_close = false;                // within 2x of the locked side: worth re-checking online
    double lock_total = 0;                 // locked side's median total when it locked (drift reference)
    int64_t since = 0, taken = 0;          // runs since the last online sample; online samples so far
    std::vector<Sample> on_locked, on_alt;  // recent online samples of each side
    int64_t flips = 0, reopens = 0;
  };
  Strategy decide(Entry& e, const Choice& c);
  void lock(Entry& e, int winner, int other);
  void online(Entry& e, Strategy s, Sample sample);
  TunerConfig cfg_;
  std::unordered_map<std::string, Entry> table_;
  std::set<std::string> uncovered_;  // buckets met without a profile entry while not exploring
};

}  // namespace ce
