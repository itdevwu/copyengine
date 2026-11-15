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

uint64_t fingerprint_sides(const Op& op) {
  Hasher h;
  add_side(h, op.src);
  add_side(h, op.dst);
  h.add(std::bit_cast<uint32_t>(op.fn.scale));
  h.add(std::bit_cast<uint32_t>(op.fn.bias));
  h.add((op.fn.round ? 1u : 0u) | (op.fn.clamp ? 2u : 0u));
  h.add(std::bit_cast<uint32_t>(op.fn.lo));
  h.add(std::bit_cast<uint32_t>(op.fn.hi));
  h.add(op.ctx);
  return h.done().a;
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
  if (op.tile) {  // the whole description: tile ops are planned by their layout and addresses
    const cv::TileK& t = *op.tile;
    for (long long v : {t.x, t.q, t.s, t.z, t.rows[0], t.rows[1], t.rows[2], t.xr[0], t.xr[1], t.xr[2], t.qr[0], t.qr[1],
                        t.qr[2], t.sr[0], t.sr[1], t.sr[2], t.zr[0], t.zr[1], t.zr[2], t.xe, t.qe, t.se, t.ze, t.xb, t.cb,
                        static_cast<long long>(t.xi != 0), static_cast<long long>(t.ci != 0)})
      h.add(static_cast<uint64_t>(v));  // field by field: no padding bytes in the hash
    // Index tables (host values, GPU pointers) are this call's: footprints
    // cover the index bounds and executors read the tables from the ops they
    // run, so new indices hit the same plan.
    for (int v : {t.g, t.upr, t.fmt, t.xdt, t.scale_byte, t.quantize, t.static_scale}) h.add(static_cast<uint64_t>(v));
    h.add(std::bit_cast<uint32_t>(t.eps)), h.add(std::bit_cast<uint32_t>(t.gs));
  }
  if (op.opaque && op.form) {  // the structure, not this call's table pointers: new indices hit the same plan
    const Form& f = *op.form;
    h.add(static_cast<uint64_t>(op.dst_lo));
    h.add(static_cast<uint64_t>(op.dst_hi));
    h.add(static_cast<uint64_t>(f.rank));
    for (int k = 0; k < f.rank; ++k)
      h.add(static_cast<uint64_t>(f.shape[k])), h.add(static_cast<uint64_t>(f.ss[k])), h.add(static_cast<uint64_t>(f.ds[k]));
    for (i64 v : {f.s_base, f.d_base, static_cast<i64>(f.stab_dim), static_cast<i64>(f.dtab_dim), f.stab_step,
                  f.dtab_step, f.sbound, f.dbound, static_cast<i64>(f.stab_host), static_cast<i64>(f.dtab_host)})
      h.add(static_cast<uint64_t>(v));
  }
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

Runs lower_form(const Form& f, const int64_t* stab, const int64_t* dtab) {
  Runs r;
  if (f.rank == 0) {
    r.src.push_back(f.s_base), r.dst.push_back(f.d_base), r.len.push_back(1);
    return r;
  }
  const int in = f.rank - 1;
  const bool whole = f.ss[in] == 1 && f.ds[in] == 1 && in != f.stab_dim && in != f.dtab_dim;
  const int outer = whole ? in : f.rank;  // dimensions walked one index at a time
  const i64 len = whole ? f.shape[in] : 1;
  std::array<i64, Form::kMaxRank> i{};
  i64 count = 1;
  for (int k = 0; k < outer; ++k) count *= f.shape[k];
  for (i64 n = 0; n < count; ++n) {
    i64 s = f.s_base, d = f.d_base;
    for (int k = 0; k < outer; ++k) {
      i64 is = i[k], id = i[k];
      if (k == f.stab_dim) {
        is = stab[i[k] * f.stab_step];
        if (is < 0 || is >= f.sbound) fail(Errc::invalid_argument, "a host index table held an index out of range");
      }
      if (k == f.dtab_dim) {
        id = dtab[i[k] * f.dtab_step];
        if (id < 0 || id >= f.dbound) fail(Errc::invalid_argument, "a host index table held an index out of range");
      }
      s += is * f.ss[k], d += id * f.ds[k];
    }
    if (!r.empty() && r.src.back() + r.len.back() == s && r.dst.back() + r.len.back() == d) r.len.back() += len;
    else r.src.push_back(s), r.dst.push_back(d), r.len.push_back(len);
    for (int k = outer - 1; k >= 0 && ++i[k] == f.shape[k]; --k) i[k] = 0;
  }
  return r;
}

std::pair<i64, i64> Form::src_extent() const {
  auto sh = shape;
  if (stab_dim >= 0) sh[stab_dim] = sbound;  // any index within its bound
  return extent_of(s_base, sh, ss, rank);
}
std::pair<i64, i64> Form::dst_extent() const {
  auto sh = shape;
  if (dtab_dim >= 0) sh[dtab_dim] = dbound;
  return extent_of(d_base, sh, ds, rank);
}

// ---- safety ----------------------------------------------------------------------

const Runs& Op::runs() const {
  static const Runs kEmpty;
  return body ? body->runs : kEmpty;
}

const Intervals& Op::read() const {
  const Runs& r = runs();
  if (opaque && !body->rd) {  // the indices are not known on the host: every element an index may reach
    const auto [lo, hi] = form->src_extent();
    const i64 sz = static_cast<i64>(itemsize(src.dtype));
    auto iv = std::make_unique<Intervals>();
    iv->start.push_back(lo * sz), iv->end.push_back((hi + 1) * sz);
    body->rd = std::move(iv);
  }
  if (!body->rd) body->rd = std::make_unique<Intervals>(interval_union(r.src, r.len, itemsize(src.dtype)));
  return *body->rd;
}

const Intervals& Op::write() const {
  const Runs& r = runs();
  if (opaque && !body->wr) {  // exact when the destination has no table; else its whole bound
    const auto [lo, hi] = form->dst_extent();
    const i64 sz = static_cast<i64>(itemsize(dst.dtype));
    auto iv = std::make_unique<Intervals>();
    iv->start.push_back(lo * sz), iv->end.push_back((hi + 1) * sz);
    body->wr = std::move(iv);
  }
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
    if (op.runs().empty() && !op.opaque) continue;
    auto& r = by_dev[{int(op.src.device.kind), op.src.device.index}];
    const Intervals& rd = op.read();
    for (size_t i = 0; i < rd.start.size(); ++i) r.push_back({rd.start[i], rd.end[i], k, false});
    auto& w = by_dev[{int(op.dst.device.kind), op.dst.device.index}];
    const Intervals& wr = op.write();
    for (size_t i = 0; i < wr.start.size(); ++i) w.push_back({wr.start[i], wr.end[i], k, true});
    for (const Op::Extra& x : op.extras)  // a tile op's scale / zero point
      by_dev[{int(x.device.kind), x.device.index}].push_back({x.lo, x.hi, k, x.write});
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
      auto same_tile = [&](const Group& g) {
        const cv::TileK* a = g.ops.empty() ? nullptr : ops[g.ops[0]].tile.get();
        const cv::TileK* b = op.tile.get();
        if (!a || !b) return a == b;
        return a->fmt == b->fmt && a->quantize == b->quantize && a->static_scale == b->static_scale && a->g == b->g &&
               a->xdt == b->xdt &&
               a->scale_byte == b->scale_byte && a->eps == b->eps && a->gs == b->gs;
      };
      auto it = std::find_if(groups.begin(), groups.end(), [&](const Group& g) {
        return g.src == op.src && g.dst == op.dst && g.fn == op.fn && g.ctx == op.ctx && g.bandwidth == op.bandwidth &&
               same_tile(g);
      });
      if (it == groups.end()) {
        groups.push_back({});
        it = groups.end() - 1;
        it->src = op.src, it->dst = op.dst, it->fn = op.fn, it->ctx = op.ctx, it->wave = lv;
        it->bandwidth = op.bandwidth;
      }
      it->ops.push_back(k);
      it->runs.append(op.runs());
      if (op.tile) {
        const bool first = it->ops.size() == 1;
        it->opaque_elems += op.tile->units() * op.tile->g, it->tiles = true;
        it->tile_host_sides = it->tile_host_sides || op.tile_host;
        it->tile_staged_ok = (first || it->tile_staged_ok) && op.tile_staged_ok;
      }
      else if (op.opaque) it->opaque_elems += op.form->numel;
    }
    for (Group& g : groups) {
      const i64 ss = static_cast<i64>(itemsize(g.src.dtype)), ds = static_cast<i64>(itemsize(g.dst.dtype));
      std::vector<i64> sc, dc, s_lo, s_len, d_lo, d_len;  // cuts; allocation ranges (run units)
      for (int k : g.ops) {
        const Op& op = ops[k];
        g.op_runs += static_cast<int64_t>(op.runs().size());
        g.src_allocs.emplace_back(op.src_lo, op.src_hi);
        sc.push_back(op.src_lo / ss), sc.push_back(op.src_hi / ss);
        dc.push_back(op.dst_lo / ds), dc.push_back(op.dst_hi / ds);
        s_lo.push_back(op.src_lo / ss), s_len.push_back(op.src_hi / ss - op.src_lo / ss);
        d_lo.push_back(op.dst_lo / ds), d_len.push_back(op.dst_hi / ds - op.dst_lo / ds);
      }
      for (auto* v : {&sc, &dc}) {
        std::sort(v->begin(), v->end());
        v->erase(std::unique(v->begin(), v->end()), v->end());
      }
      std::sort(g.src_allocs.begin(), g.src_allocs.end());
      g.src_allocs.erase(std::unique(g.src_allocs.begin(), g.src_allocs.end()), g.src_allocs.end());
      // Real allocations never overlap: a cut inside one described allocation
      // is the same memory described twice, not a boundary.
      g.src_cuts = real_cuts(sc, s_lo, s_len);
      g.dst_cuts = real_cuts(dc, d_lo, d_len);
      if (std::all_of(g.ops.begin(), g.ops.end(), [&](int k) { return ops[k].form != nullptr; }))
        for (int k : g.ops) g.forms.push_back(ops[k].form);
      g.runs = g.runs.coalesce(g.src_cuts, g.dst_cuts);
      g.rects = recover_2d(g.runs, 2, g.src_cuts, g.dst_cuts);
      if (!g.runs.empty() || g.opaque_elems) plan.groups.push_back(std::move(g));
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
         (g.rects.has_2d() ? "|2d" : "") + (g.ctx.empty() ? "" : "|" + g.ctx) + (g.bandwidth > 0 ? "|paced" : "");
}

}  // namespace ce
