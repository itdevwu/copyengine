#include "core/tuner.hpp"

#include <algorithm>

#include "copyengine/error.hpp"

namespace ce {

namespace {
constexpr std::string_view kNames[kNumStrategies] = {"passthrough", "runs", "runs_parallel", "batch", "kernel", "staged"};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
}  // namespace

std::string_view strategy_name(Strategy s) { return kNames[static_cast<int>(s)]; }

Strategy parse_strategy(std::string_view name) {
  for (int i = 0; i < kNumStrategies; ++i)
    if (kNames[i] == name) return static_cast<Strategy>(i);
  fail(Errc::invalid_argument, "unknown strategy: " + std::string(name));
}

std::vector<Strategy> Tuner::shortlist(std::vector<Strategy> prior) const {
  std::erase(prior, Strategy::passthrough);
  if (static_cast<int>(prior.size()) > std::max(1, topk_ - 1)) prior.resize(std::max(1, topk_ - 1));
  prior.push_back(Strategy::passthrough);
  return prior;
}

Strategy Tuner::pick(const std::string& bucket, const std::vector<Strategy>& cands) {
  Entry& e = table_[bucket];
  if (e.locked >= 0) {
    const auto s = static_cast<Strategy>(e.locked);
    if (std::find(cands.begin(), cands.end(), s) != cands.end()) return s;
  }
  if (trials_ <= 0) return cands.front();
  for (Strategy s : cands)
    if (static_cast<int>(e.samples[static_cast<int>(s)].size()) < trials_) return s;
  Strategy best = cands.front();
  double best_t = median(e.samples[static_cast<int>(best)]);
  for (Strategy s : cands) {
    const double t = median(e.samples[static_cast<int>(s)]);
    if (t < best_t) best = s, best_t = t;
  }
  e.locked = static_cast<int>(best);
  return best;
}

bool Tuner::exploring(const std::string& bucket) const {
  if (trials_ <= 0) return false;
  auto it = table_.find(bucket);
  return it == table_.end() || it->second.locked < 0;
}

void Tuner::report(const std::string& bucket, Strategy s, double seconds) {
  Entry& e = table_[bucket];
  if (e.locked < 0) e.samples[static_cast<int>(s)].push_back(seconds);
}

Json Tuner::dump() const {
  std::vector<std::string> keys;
  for (auto& [k, _] : table_) keys.push_back(k);
  std::sort(keys.begin(), keys.end());
  Json out = Json::object();
  for (auto& k : keys) {
    const Entry& e = table_.at(k);
    Json samples = Json::object();
    for (int i = 0; i < kNumStrategies; ++i) {
      if (e.samples[i].empty()) continue;
      Json arr = Json::array();
      for (double t : e.samples[i]) arr.push(t);
      samples.set(std::string(kNames[i]), std::move(arr));
    }
    out.set(k, Json::object()
                   .set("locked", e.locked >= 0 ? Json(kNames[e.locked]) : Json(nullptr))
                   .set("samples", std::move(samples)));
  }
  return out;
}

void Tuner::load(const Json& buckets) {
  if (!buckets.is_object()) fail(Errc::invalid_argument, "profile: buckets must be an object");
  for (auto& [k, v] : buckets.items()) {
    Entry e;
    if (v["locked"].is_string()) e.locked = static_cast<int>(parse_strategy(v["locked"].str()));
    if (v["samples"].is_object())
      for (auto& [name, arr] : v["samples"].items())
        if (arr.is_array())
          for (auto& t : arr.elements())
            if (t.is_number()) e.samples[static_cast<int>(parse_strategy(name))].push_back(t.num());
    table_[k] = std::move(e);
  }
}

}  // namespace ce
