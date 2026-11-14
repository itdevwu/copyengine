#include "core/tuner.hpp"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cmath>

#include "copyengine/copyengine.hpp"
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
  for (const Sample& s : v) arr.push(Json::array().push(s.host).push(s.gpu).push(s.submit));
  return arr;
}

std::vector<Sample> parse_samples(const Json& arr) {
  std::vector<Sample> out;
  if (!arr.is_array()) return out;
  for (auto& p : arr.elements())
    if (p.is_array() && p.elements().size() == 3 && p.elements()[0].is_number() && p.elements()[1].is_number() &&
        p.elements()[2].is_number())
      out.push_back({p.elements()[0].num(), p.elements()[1].num(), p.elements()[2].num()});
  return out;
}

bool allowed(Strategy s, const Choice& c) {
  return s == c.baseline || s == c.reference() || (c.safe && s == *c.safe) ||
         std::find(c.cands.begin(), c.cands.end(), s) != c.cands.end();
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

Tuner::Pick Tuner::pick(const std::string& bucket, const Choice& c, int64_t runs) {
  const Strategy reference = c.reference();
  if (cfg_.prior_only) return {c.cands.empty() ? reference : c.cands.front(), false};
  auto it = table_.find(bucket);
  if (!cfg_.explore) {
    if (it != table_.end() && it->second.locked >= 0 && allowed(static_cast<Strategy>(it->second.locked), c))
      return {static_cast<Strategy>(it->second.locked), false};
    uncovered_.insert(bucket);
    return {reference, false};
  }
  Entry& e = it != table_.end() ? it->second : table_[bucket];
  if (e.locked >= 0 && allowed(static_cast<Strategy>(e.locked), c)) {
    const auto locked = static_cast<Strategy>(e.locked);
    if (cfg_.online_every <= 0 || (e.since += runs) < cfg_.online_every) return {locked, false};
    e.since = 0;
    // An online sample: mostly the locked side, every alt_every-th the other
    // finalist, if it was close enough to ever flip (a far slower one would
    // stall the run it samples; drift of the locked side still re-opens).
    ++e.taken;
    if (e.alt >= 0 && e.alt_close && allowed(static_cast<Strategy>(e.alt), c) && e.taken % cfg_.alt_every == 0)
      return {static_cast<Strategy>(e.alt), true};
    return {locked, true};
  }
  e.locked = -1;
  const Strategy s = decide(e, c);
  if (e.locked >= 0) return {s, false};
  ++e.pending[static_cast<int>(s)];
  return {s, true};
}

// Next run for an unlocked entry; may lock it.
Strategy Tuner::decide(Entry& e, const Choice& c) {
  const Strategy reference = c.reference();
  // With a baseline to protect (native, or a dominance-safe merge), the best
  // speculative candidate must then beat it in the A/B. Without one (`lead`),
  // every candidate is explored alike and the fastest locks: no A/B.
  const bool protect = !c.lead;
  std::vector<Strategy> cands;
  for (Strategy s : c.cands)
    if (!protect || s != reference) cands.push_back(s);
  if (!protect && std::find(cands.begin(), cands.end(), reference) == cands.end()) cands.insert(cands.begin(), reference);
  if (cands.empty()) {
    lock(e, static_cast<int>(reference), -1);
    return reference;
  }
  if (e.challenger < 0 || !allowed(static_cast<Strategy>(e.challenger), c)) {
    e.challenger = -1;
    // Round-robin, so clock or load changes during exploration hit every
    // candidate alike: the least-sampled one goes next (prior order on ties).
    // A candidate already 4x slower than the best one is not sampled again.
    double best_seen = 1e300;
    for (Strategy s : cands) {
      const auto& v = e.samples[static_cast<int>(s)];
      if (v.empty()) continue;
      std::vector<double> t;
      for (const Sample& x : v) t.push_back(x.own());
      best_seen = std::min(best_seen, median(t));
    }
    auto hopeless = [&](Strategy s) {
      const auto& v = e.samples[static_cast<int>(s)];
      if (v.empty()) return false;
      std::vector<double> t;
      for (const Sample& x : v) t.push_back(x.own());
      return median(t) > 4 * best_seen;
    };
    std::erase_if(cands, hopeless);
    Strategy next = cands.front();
    int fewest = INT32_MAX;
    for (Strategy s : cands) {
      const int i = static_cast<int>(s), have = static_cast<int>(e.samples[i].size()) + e.pending[i];
      if (have < fewest) next = s, fewest = have;
    }
    if (fewest < cfg_.trials) return next;
    // By median own time; candidates still without a reported sample come last.
    auto rank = [&](Strategy s) {
      std::vector<double> t;
      for (const Sample& x : e.samples[static_cast<int>(s)]) t.push_back(x.own());
      return t.empty() ? 1e300 : median(t);
    };
    std::vector<Strategy> by = cands;
    std::stable_sort(by.begin(), by.end(), [&](Strategy a, Strategy b) { return rank(a) < rank(b); });
    for (Strategy s : cands)
      if (e.pending[static_cast<int>(s)] > 0)  // wait for outstanding samples, running something cheap
        return protect ? reference : by.front();
    const Strategy best = by.front();
    if (!protect) {
      lock(e, static_cast<int>(best), by.size() > 1 ? static_cast<int>(by[1]) : -1);
      return best;
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
  // One side at least 2x faster in every pair so far: decided early, so warmup
  // does not keep running a far slower side (6 of 6: sign test p ~ 1/64).
  if (n >= 6 && n < cfg_.pairs) {
    if (challenger_wins(cs, rs, 0.5)) {
      lock(e, e.challenger, static_cast<int>(reference));
      return ch;
    }
    if (challenger_wins(rs, cs, 0.5)) {
      lock(e, static_cast<int>(reference), e.challenger);
      return reference;
    }
  }
  if (n >= cfg_.pairs) {
    std::vector<double> ct, rt;
    for (int i = 0; i < n; ++i) ct.push_back(cs[i].total()), rt.push_back(rs[i].total());
    if (challenger_wins(cs, rs, cfg_.margin)) {
      lock(e, e.challenger, static_cast<int>(reference));
      return ch;
    }
    if (n >= cfg_.max_pairs || median(ct) >= median(rt)) {
      lock(e, static_cast<int>(reference), e.challenger);
      return reference;
    }
  }
  return e.issued++ % 2 == 0 ? ch : reference;
}

void Tuner::lock(Entry& e, int winner, int other) {
  e.locked = winner;
  e.alt = other;
  std::vector<double> t;
  for (const Sample& x : e.samples[winner]) t.push_back(x.total());
  e.lock_total = median(t);
  e.alt_close = false;
  if (other >= 0 && !e.samples[other].empty()) {
    std::vector<double> a;
    for (const Sample& x : e.samples[other]) a.push_back(x.total());
    e.alt_close = e.lock_total > 0 && median(a) <= 2 * e.lock_total;
  }
  e.since = e.taken = 0;
  e.on_locked.clear();
  e.on_alt.clear();
}

void Tuner::online(Entry& e, Strategy s, Sample sample) {
  const size_t w = static_cast<size_t>(std::max(2, cfg_.window));
  if (static_cast<int>(s) == e.locked) {
    e.on_locked.push_back(sample);
    if (e.on_locked.size() > w) e.on_locked.erase(e.on_locked.begin());
  } else if (static_cast<int>(s) == e.alt) {
    e.on_alt.push_back(sample);
  } else {
    return;
  }
  // The other finalist got clearly faster: swap, with the A/B's own rule.
  if (e.on_alt.size() >= w && e.on_locked.size() >= w) {
    if (challenger_wins(e.on_alt, e.on_locked, cfg_.margin)) {
      const int was = e.locked;
      e.samples[e.alt] = e.on_alt;
      lock(e, e.alt, was);
      ++e.flips;
      return;
    }
    e.on_alt.clear();
  }
  // The locked side itself drifted well away from what won: explore again.
  if (e.on_locked.size() >= w) {
    std::vector<double> t;
    for (const Sample& x : e.on_locked) t.push_back(x.total());
    if (e.lock_total <= 0) e.lock_total = median(t);  // locked without samples: this is the reference
    if (median(t) > e.lock_total * cfg_.drift) {
      e.locked = e.challenger = e.alt = -1;
      for (auto& v : e.samples) v.clear();
      e.on_locked.clear();
      e.on_alt.clear();
      ++e.reopens;
    }
  }
}

void Tuner::discard(const std::string& bucket, Strategy s) {
  auto it = table_.find(bucket);
  if (it == table_.end()) return;
  int& p = it->second.pending[static_cast<int>(s)];
  if (p > 0) --p;
}

void Tuner::report(const std::string& bucket, Strategy s, Sample sample) {
  auto it = table_.find(bucket);
  if (it == table_.end()) return;
  Entry& e = it->second;
  int& p = e.pending[static_cast<int>(s)];
  if (p > 0) --p;
  if (e.locked >= 0) online(e, s, sample);
  else e.samples[static_cast<int>(s)].push_back(sample);
}

namespace {
std::string fmt_us(double v) {
  char b[32];
  std::snprintf(b, sizeof b, v < 10 ? "%.1f us" : "%.0f us", v);
  return b;
}
double median_total(const std::vector<Sample>& v) {
  std::vector<double> t;
  for (const Sample& x : v) t.push_back(x.total());
  return median(t);
}
}  // namespace

std::string Tuner::explain(const std::string& bucket, const Choice& c, Strategy ran) const {
  const std::string name(strategy_name(ran));
  if (cfg_.prior_only) return "prior's first choice (no measurement)";
  auto it = table_.find(bucket);
  if (it == table_.end() || it->second.locked < 0) {
    if (!cfg_.explore) return "not in the loaded profile: " + std::string(c.lead ? "copyengine's default" : "the baseline");
    if (it == table_.end()) return "exploring";
    const Entry& e = it->second;
    if (e.challenger < 0) return "exploring: sampling " + name;
    return "A/B: " + std::string(strategy_name(static_cast<Strategy>(e.challenger))) + " vs " +
           std::string(strategy_name(c.reference())) + ", sampling " + name;
  }
  const Entry& e = it->second;
  const auto locked = static_cast<Strategy>(e.locked);
  if (ran != locked) return "online re-check of " + name + " (locked: " + std::string(strategy_name(locked)) + ")";
  std::string why = "locked";
  if (e.alt >= 0 && !e.samples[e.alt].empty() && !e.samples[e.locked].empty())
    why += ": " + fmt_us(median_total(e.samples[e.locked])) + " vs " +
           std::string(strategy_name(static_cast<Strategy>(e.alt))) + " " + fmt_us(median_total(e.samples[e.alt]));
  else if (c.cands.empty() || (c.cands.size() == 1 && c.cands[0] == c.reference()))
    why += ": the only choice";
  if (e.flips) why += ", flipped online " + std::to_string(e.flips) + "x";
  return why;
}

std::optional<Strategy> Tuner::locked_strategy(const std::string& bucket) const {
  auto it = table_.find(bucket);
  if (it == table_.end() || it->second.locked < 0) return std::nullopt;
  return static_cast<Strategy>(it->second.locked);
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
                       .set("alt", e.alt >= 0 ? Json(kNames[e.alt]) : Json(nullptr))
                       .set("lock_total", e.lock_total)
                       .set("samples", std::move(samples))
                       .set("online", Json::object()
                                          .set("samples", e.taken)
                                          .set("flips", e.flips)
                                          .set("reopens", e.reopens)));
  }
  Json uncovered = Json::array();
  for (auto& k : uncovered_) uncovered.push(k);
  return Json::object()
      .set("version", std::string(version_string()))
      .set("buckets", std::move(buckets))
      .set("uncovered", std::move(uncovered));
}

void Tuner::load(const Json& j) {
  const Json& buckets = j["buckets"];
  if (!buckets.is_object()) fail(Errc::invalid_argument, "profile: buckets must be an object");
  // Strategies and kernels change between versions: their measurements are re-taken.
  const bool same = j["version"].is_string() && j["version"].str() == version_string();
  for (auto& [k, v] : buckets.items()) {
    if (!same) {
      if (!table_.count(k)) uncovered_.insert(k);
      continue;
    }
    Entry e;
    if (v["locked"].is_string()) e.locked = static_cast<int>(parse_strategy(v["locked"].str()));
    if (v["alt"].is_string()) e.alt = static_cast<int>(parse_strategy(v["alt"].str()));
    if (v["lock_total"].is_number()) e.lock_total = v["lock_total"].num();
    if (v["samples"].is_object())
      for (auto& [name, arr] : v["samples"].items()) e.samples[static_cast<int>(parse_strategy(name))] = parse_samples(arr);
    if (e.alt >= 0 && !e.samples[e.alt].empty() && e.lock_total > 0) {
      std::vector<double> a;
      for (const Sample& x : e.samples[e.alt]) a.push_back(x.total());
      e.alt_close = median(a) <= 2 * e.lock_total;
    }
    table_[k] = std::move(e);
    uncovered_.erase(k);
  }
  if (j["uncovered"].is_array())
    for (auto& k : j["uncovered"].elements())
      if (k.is_string() && !table_.count(k.str())) uncovered_.insert(k.str());
}

}  // namespace ce
