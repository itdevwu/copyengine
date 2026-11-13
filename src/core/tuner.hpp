#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "copyengine/engine.hpp"
#include "core/json.hpp"

namespace ce {

// Per bucket: try each candidate `trials` times, then lock the one with the
// lowest median (FFTW MEASURE style). trials == 0 means never explore.
class Tuner {
 public:
  Tuner(int trials, int topk) : trials_(trials), topk_(topk) {}

  // Narrows prior-ordered candidates to what will be explored (top-k + passthrough).
  std::vector<Strategy> shortlist(std::vector<Strategy> prior) const;
  Strategy pick(const std::string& bucket, const std::vector<Strategy>& cands);
  bool exploring(const std::string& bucket) const;
  void report(const std::string& bucket, Strategy s, double seconds);

  Json dump() const;
  void load(const Json& buckets);
  size_t size() const { return table_.size(); }

 private:
  struct Entry {
    int locked = -1;
    std::array<std::vector<double>, kNumStrategies> samples;
  };
  int trials_, topk_;
  std::unordered_map<std::string, Entry> table_;
};

}  // namespace ce
