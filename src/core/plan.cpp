#include "core/plan.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <map>
#include <tuple>

#include "copyengine/error.hpp"

namespace ce {

// ---- hashing -------------------------------------------------------------------

namespace {
uint64_t mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
}  // namespace

void Hasher::add(uint64_t v) {
  a_ = mix(a_ ^ v);
  b_ = mix(b_ + std::rotl(v, 29) + 0x632BE59BD9B4E019ull);
}

void Hasher::add(std::span<const i64> v) {
  add(static_cast<uint64_t>(v.size()));
  for (i64 x : v) add(static_cast<uint64_t>(x));
}

void Hasher::add(std::string_view s) {
  add(static_cast<uint64_t>(s.size()));
  for (unsigned char c : s) add(c);
}

static void add_side(Hasher& h, const Side& s) {
  h.add((static_cast<uint64_t>(s.device.kind) << 40) | (static_cast<uint64_t>(s.device.index) << 8) |
        (static_cast<uint64_t>(s.dtype) << 1) | (s.pinned ? 1u : 0u));
}

Hash128 fingerprint(const Op& op) {
  Hasher h;
  add_side(h, op.src);
  add_side(h, op.dst);
  h.add(std::bit_cast<uint32_t>(op.fn.scale));
  h.add(std::bit_cast<uint32_t>(op.fn.bias));
  h.add((op.fn.round ? 1u : 0u) | (op.fn.clamp ? 2u : 0u));
  h.add(std::bit_cast<uint32_t>(op.fn.lo));
  h.add(std::bit_cast<uint32_t>(op.fn.hi));
  h.add(op.ctx);
  h.add(uint64_t{op.form != nullptr});
  h.add(static_cast<uint64_t>(op.src_lo));
  h.add(static_cast<uint64_t>(op.src_hi));
  h.add(op.runs().src);
  h.add(op.runs().dst);
  h.add(op.runs().len);
  return h.done();
}

static std::pair<i64, i64> extent_of(i64 base, const std::array<i64, Form::kMaxRank>& shape,
                                     const std::array<i64, Form::kMaxRank>& stride, int rank) {
  i64 lo = base, hi = base;
  for (int k = 0; k < rank; ++k) {
    const i64 span = (shape[k] - 1) * stride[k];
    lo += std::min<i64>(0, span), hi += std::max<i64>(0, span);
  }
  return {lo, hi};
}

std::pair<i64, i64> Form::src_extent() const { return extent_of(s_base, shape, ss, rank); }
std::pair<i64, i64> Form::dst_extent() const { return extent_of(d_base, shape, ds, rank); }

// ---- safety ----------------------------------------------------------------------

const Runs& Op::runs() const {
  static const Runs kEmpty;
  return body ? body->runs : kEmpty;
}

const Intervals& Op::read() const {
  const Runs& r = runs();
  if (!body->rd) body->rd = std::make_unique<Intervals>(interval_union(r.src, r.len, itemsize(src.dtype)));
  return *body->rd;
}

const Intervals& Op::write() const {
  const Runs& r = runs();
  if (!body->wr) body->wr = std::make_unique<Intervals>(interval_union(r.dst, r.len, itemsize(dst.dtype)));
  return *body->wr;
}

void check_safe(const Op& op) {
  const Runs& r = op.runs();
  if (r.empty()) return;
  std::vector<size_t> o(r.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = i;
  std::sort(o.begin(), o.end(), [&](size_t a, size_t b) { return r.dst[a] < r.dst[b]; });
  for (size_t k = 1; k < o.size(); ++k)
    if (r.dst[o[k]] < r.dst[o[k - 1]] + r.len[o[k - 1]])
      fail(Errc::unsafe_copy, "destination written more than once (duplicate scatter index or stride-0 dst)");
  if (op.src.device == op.dst.device && intersects(op.read(), op.write()))
    fail(Errc::unsafe_copy, "source and destination overlap within one copy (needs memmove semantics)");
}

// ---- dependencies ---------------------------------------------------------------

std::vector<int> waves(const std::vector<Op>& ops) {
  const int n = static_cast<int>(ops.size());
  // Every exact read / write interval, per device. Overlap of two intervals
  // with at least one writer is a dependency by definition (flow, anti or
  // output), so no pair needs a second, exact check.
  struct Item {
    i64 lo, hi;
    int k;
    bool write;
  };
  std::map<std::pair<int, int>, std::vector<Item>> by_dev;
  for (int k = 0; k < n; ++k) {
    const Op& op = ops[k];
    if (op.runs().empty()) continue;
    auto& r = by_dev[{int(op.src.device.kind), op.src.device.index}];
    const Intervals& rd = op.read();
    for (size_t i = 0; i < rd.start.size(); ++i) r.push_back({rd.start[i], rd.end[i], k, false});
    auto& w = by_dev[{int(op.dst.device.kind), op.dst.device.index}];
    const Intervals& wr = op.write();
    for (size_t i = 0; i < wr.start.size(); ++i) w.push_back({wr.start[i], wr.end[i], k, true});
  }
  // (pred, succ) pairs with pred < succ, possibly repeated.
  std::vector<std::pair<int, int>> edges;
  for (auto& [dev, items] : by_dev) {
    if (std::none_of(items.begin(), items.end(), [](const Item& i) { return i.write; })) continue;
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) { return a.lo < b.lo; });
    // Intervals still open at the sweep position. Each op's own intervals are
    // disjoint per side, so an active set only grows with real overlap.
    std::vector<Item> act_r, act_w;
    auto prune = [](std::vector<Item>& v, i64 s) {
      std::erase_if(v, [&](const Item& a) { return a.hi <= s; });
    };
    auto link = [&](const std::vector<Item>& act, const Item& it) {
      for (const Item& a : act)
        if (a.k != it.k) edges.emplace_back(std::min(a.k, it.k), std::max(a.k, it.k));
    };
    for (const Item& it : items) {
      prune(act_w, it.lo);
      link(act_w, it);
      if (it.write) {
        prune(act_r, it.lo);
        link(act_r, it);
        act_w.push_back(it);
      } else {
        act_r.push_back(it);
      }
    }
  }
  std::sort(edges.begin(), edges.end(), [](const auto& a, const auto& b) {
    return a.second != b.second ? a.second < b.second : a.first < b.first;
  });
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  std::vector<int> level(n, 0);
  for (const auto& [i, j] : edges) level[j] = std::max(level[j], level[i] + 1);  // i < j: level[i] is final
  return level;
}

// ---- planning --------------------------------------------------------------------

Plan build_plan(const std::vector<Op>& ops) {
  Plan plan;
  if (ops.empty()) return plan;
  const std::vector<int> level = waves(ops);
  plan.waves = *std::max_element(level.begin(), level.end()) + 1;
  for (int lv = 0; lv < plan.waves; ++lv) {
    std::vector<Group> groups;
    for (int k = 0; k < static_cast<int>(ops.size()); ++k) {
      if (level[k] != lv) continue;
      const Op& op = ops[k];
      auto it = std::find_if(groups.begin(), groups.end(), [&](const Group& g) {
        return g.src == op.src && g.dst == op.dst && g.fn == op.fn && g.ctx == op.ctx;
      });
      if (it == groups.end()) {
        groups.push_back({});
        it = groups.end() - 1;
        it->src = op.src, it->dst = op.dst, it->fn = op.fn, it->ctx = op.ctx, it->wave = lv;
      }
      it->ops.push_back(k);
      it->runs.append(op.runs());
    }
    for (Group& g : groups) {
      std::vector<i64> lo, len;
      for (int k : g.ops) lo.push_back(ops[k].src_lo), len.push_back(ops[k].src_hi - ops[k].src_lo);
      g.src_extent = interval_union(lo, len);
      if (std::all_of(g.ops.begin(), g.ops.end(), [&](int k) { return ops[k].form != nullptr; }))
        for (int k : g.ops) g.forms.push_back(ops[k].form);
      g.runs = g.runs.coalesce();
      g.rects = recover_2d(g.runs);
      if (!g.runs.empty()) plan.groups.push_back(std::move(g));
    }
  }
  return plan;
}

static int lg(int64_t x) { return x > 0 ? 63 - std::countl_zero(static_cast<uint64_t>(x)) : -1; }

std::string bucket_key(const Group& g, int nops) {
  auto side = [](const Side& s) {
    std::string r = s.device.kind == DeviceKind::cuda ? "cuda" : (s.pinned ? "pinned" : "host");
    return r + ":" + std::string(dtype_name(s.dtype));
  };
  std::string fn = g.fn.identity() ? "id" : (std::string("aff") + (g.fn.round ? "r" : "") + (g.fn.clamp ? "c" : ""));
  return side(g.src) + ">" + side(g.dst) + "|" + fn + "|b" + std::to_string(lg(g.bytes())) + "|n" +
         std::to_string(lg(static_cast<int64_t>(g.rects.size()))) + "|o" + std::to_string(lg(nops)) +
         (g.rects.has_2d() ? "|2d" : "") + (g.ctx.empty() ? "" : "|" + g.ctx);
}

}  // namespace ce
