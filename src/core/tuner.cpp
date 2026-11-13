#include "core/tuner.hpp"

#include <algorithm>
#include <cmath>

#include "copyengine/error.hpp"

namespace ce {

namespace {
constexpr std::string_view kNames[kNumStrategies] = {"passthrough", "runs",  "runs_parallel", "batch", "kernel",
                                                     "staged",      "bounce", "fused",         "native"};

double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

// Relative median absolute deviation: the reference's own noise level.
double rel_noise(const std::vector<double>& v) {
  const double m = median(v);
  if (m <= 0) return 0;
  std::vector<double> dev(v.size());
  for (size_t i = 0; i < v.size(); ++i) dev[i] = std::abs(v[i] - m);
  return median(dev) / m;
}

// Smallest k with P(Binomial(n, 1/2) >= k) <= 0.05: a one-sided sign test.
int sign_test_threshold(int n) {
  std::vector<double> pmf(n + 1);
  double c = 1;  // C(n, k)
  for (int k = 0; k <= n; ++k) {
    pmf[k] = c * std::pow(0.5, n);
    c = c * (n - k) / (k + 1);
  }
  double tail = 0;
  for (int k = n; k >= 0; --k) {
    tail += pmf[k];
    if (tail > 0.05) return k + 1;
  }
  return n + 1;
}

template <class F>
std::vector<double> col(const std::vector<Sample>& v, size_t n, F f) {
  std::vector<double> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = f(v[i]);
  return out;
}

Json sample_list(const std::vector<Sample>& v) {
  Json arr = Json::array();
  for (const Sample& s : v) arr.push(Json::array().push(s.host).push(s.gpu));
  return arr;
}

std::vector<Sample> parse_samples(const Json& arr) {
  std::vector<Sample> out;
  if (!arr.is_array()) return out;
  for (auto& p : arr.elements())
    if (p.is_array() && p.elements().size() == 2 && p.elements()[0].is_number() && p.elements()[1].is_number())
      out.push_back({p.elements()[0].num(), p.elements()[1].num()});
  return out;
}

bool allowed(Strategy s, const Choice& c) {
  return s == c.baseline || (c.safe && s == *c.safe) || std::find(c.cands.begin(), c.cands.end(), s) != c.cands.end();
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
  std::erase(prior, Strategy::native);
  if (static_cast<int>(prior.size()) > std::max(1, cfg_.topk)) prior.resize(std::max(1, cfg_.topk));
  return prior;
}

bool Tuner::challenger_wins(const std::vector<Sample>& c, const std::vector<Sample>& r, double margin) {
  const size_t n = std::min(c.size(), r.size());
  if (n == 0) return false;
  constexpr double kEps = 0.25;  // µs: below timer resolution, "not worse" for near-zero metrics
  const auto host = [](const Sample& s) { return s.host; };
  const auto gpu = [](const Sample& s) { return s.gpu; };
  const auto tot = [](const Sample& s) { return s.total(); };
  const auto rh = col(r, n, host), rg = col(r, n, gpu), rt = col(r, n, tot);
  const auto ch = col(c, n, host), cg = col(c, n, gpu), ct = col(c, n, tot);
  if (median(ch) > median(rh) * (1 + rel_noise(rh)) + kEps) return false;
  if (median(cg) > median(rg) * (1 + rel_noise(rg)) + kEps) return false;
  const double need = std::min(0.5, std::max(margin, rel_noise(rt)));
  if (median(ct) > median(rt) * (1 - need)) return false;
  int wins = 0;
  for (size_t i = 0; i < n; ++i) wins += ct[i] < rt[i];
  return wins >= sign_test_threshold(static_cast<int>(n));
}

Tuner::Pick Tuner::pick(const std::string& bucket, const Choice& c) {
  const Strategy reference = c.safe ? *c.safe : c.baseline;
  if (cfg_.prior_only) return {c.cands.empty() ? reference : c.cands.front(), false};
  auto it = table_.find(bucket);
  if (!cfg_.explore) {
    if (it != table_.end() && it->second.locked >= 0 && allowed(static_cast<Strategy>(it->second.locked), c))
      return {static_cast<Strategy>(it->second.locked), false};
    uncovered_.insert(bucket);
    return {reference, false};
  }
  Entry& e = it != table_.end() ? it->second : table_[bucket];
  if (e.locked >= 0 && allowed(static_cast<Strategy>(e.locked), c)) return {static_cast<Strategy>(e.locked), false};
  e.locked = -1;
  const Strategy s = decide(e, c);
  if (e.locked >= 0) return {s, false};
  ++e.pending[static_cast<int>(s)];
  return {s, true};
}

// Next run for an unlocked entry; may lock it.
Strategy Tuner::decide(Entry& e, const Choice& c) {
  const Strategy reference = c.safe ? *c.safe : c.baseline;
  std::vector<Strategy> cands;
  for (Strategy s : c.cands)
    if (s != reference) cands.push_back(s);
  if (cands.empty()) {
    e.locked = static_cast<int>(reference);
    return reference;
  }
  if (e.challenger < 0 || !allowed(static_cast<Strategy>(e.challenger), c)) {
    e.challenger = -1;
    for (Strategy s : cands) {
      const int i = static_cast<int>(s);
      if (static_cast<int>(e.samples[i].size()) + e.pending[i] < cfg_.trials) return s;
    }
    for (Strategy s : cands)
      if (e.pending[static_cast<int>(s)] > 0) return reference;  // wait for outstanding samples
    Strategy best = cands.front();
    double best_t = 1e300;
    for (Strategy s : cands) {
      std::vector<double> t;
      for (const Sample& x : e.samples[static_cast<int>(s)]) t.push_back(x.total());
      if (median(t) < best_t) best = s, best_t = median(t);
    }
    e.challenger = static_cast<int>(best);
    e.samples[static_cast<int>(reference)].clear();  // A/B pairs start now
    e.samples[e.challenger].clear();
    e.issued = 0;
  }
  const Strategy ch = static_cast<Strategy>(e.challenger);
  const auto& cs = e.samples[e.challenger];
  const auto& rs = e.samples[static_cast<int>(reference)];
  const int n = static_cast<int>(std::min(cs.size(), rs.size()));
  if (n >= cfg_.pairs) {
    std::vector<double> ct, rt;
    for (int i = 0; i < n; ++i) ct.push_back(cs[i].total()), rt.push_back(rs[i].total());
    if (challenger_wins(cs, rs, cfg_.margin)) {
      e.locked = e.challenger;
      return ch;
    }
    if (n >= cfg_.max_pairs || median(ct) >= median(rt)) {
      e.locked = static_cast<int>(reference);
      return reference;
    }
  }
  return e.issued++ % 2 == 0 ? ch : reference;
}

void Tuner::report(const std::string& bucket, Strategy s, Sample sample) {
  auto it = table_.find(bucket);
  if (it == table_.end()) return;
  Entry& e = it->second;
  int& p = e.pending[static_cast<int>(s)];
  if (p > 0) --p;
  if (e.locked < 0) e.samples[static_cast<int>(s)].push_back(sample);
}

bool Tuner::locked(const std::string& bucket) const {
  auto it = table_.find(bucket);
  return it != table_.end() && it->second.locked >= 0;
}

Json Tuner::dump() const {
  std::vector<std::string> keys;
  for (auto& [k, _] : table_) keys.push_back(k);
  std::sort(keys.begin(), keys.end());
  Json buckets = Json::object();
  for (auto& k : keys) {
    const Entry& e = table_.at(k);
    Json samples = Json::object();
    for (int i = 0; i < kNumStrategies; ++i)
      if (!e.samples[i].empty()) samples.set(std::string(kNames[i]), sample_list(e.samples[i]));
    buckets.set(k, Json::object()
                       .set("locked", e.locked >= 0 ? Json(kNames[e.locked]) : Json(nullptr))
                       .set("samples", std::move(samples)));
  }
  Json uncovered = Json::array();
  for (auto& k : uncovered_) uncovered.push(k);
  return Json::object().set("buckets", std::move(buckets)).set("uncovered", std::move(uncovered));
}

void Tuner::load(const Json& j) {
  const Json& buckets = j["buckets"];
  if (!buckets.is_object()) fail(Errc::invalid_argument, "profile: buckets must be an object");
  for (auto& [k, v] : buckets.items()) {
    Entry e;
    if (v["locked"].is_string()) e.locked = static_cast<int>(parse_strategy(v["locked"].str()));
    if (v["samples"].is_object())
      for (auto& [name, arr] : v["samples"].items()) e.samples[static_cast<int>(parse_strategy(name))] = parse_samples(arr);
    table_[k] = std::move(e);
    uncovered_.erase(k);
  }
  if (j["uncovered"].is_array())
    for (auto& k : j["uncovered"].elements())
      if (k.is_string() && !table_.count(k.str())) uncovered_.insert(k.str());
}

}  // namespace ce
