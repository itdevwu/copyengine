#include "copyengine/ir.hpp"

#include <algorithm>
#include <numeric>
#include <string>

#include "copyengine/error.hpp"

namespace ce {
namespace {

std::vector<size_t> argsort(const std::vector<i64>& key) {
  std::vector<size_t> o(key.size());
  std::iota(o.begin(), o.end(), size_t{0});
  std::stable_sort(o.begin(), o.end(), [&](size_t a, size_t b) { return key[a] < key[b]; });
  return o;
}

void require(bool ok, const char* what) {
  if (!ok) fail(Errc::invalid_argument, what);
}

}  // namespace

// ---- Runs --------------------------------------------------------------------

i64 Runs::total() const { return std::accumulate(len.begin(), len.end(), i64{0}); }

void Runs::push(i64 s, i64 d, i64 n) {
  src.push_back(s);
  dst.push_back(d);
  len.push_back(n);
}

void Runs::reserve(size_t n) {
  src.reserve(n);
  dst.reserve(n);
  len.reserve(n);
}

void Runs::append(const Runs& o) {
  src.insert(src.end(), o.src.begin(), o.src.end());
  dst.insert(dst.end(), o.dst.begin(), o.dst.end());
  len.insert(len.end(), o.len.begin(), o.len.end());
}

Runs Runs::of(std::span<const i64> s, std::span<const i64> d, std::span<const i64> n) {
  require(s.size() == d.size() && d.size() == n.size(), "runs: src/dst/len lengths differ");
  Runs r;
  r.reserve(n.size());
  for (size_t i = 0; i < n.size(); ++i) {
    require(n[i] >= 0, "runs: negative length");
    require(s[i] >= 0 && d[i] >= 0, "runs: negative offset");
    if (n[i] > 0) r.push(s[i], d[i], n[i]);
  }
  return r;
}

Runs Runs::from_indices(std::span<const i64> s, std::span<const i64> d) {
  require(s.size() == d.size(), "indices: src/dst lengths differ");
  Runs r;
  for (size_t i = 0; i < s.size(); ++i) {
    require(s[i] >= 0 && d[i] >= 0, "indices: negative index");
    if (!r.empty() && s[i] == r.src.back() + r.len.back() && d[i] == r.dst.back() + r.len.back()) {
      ++r.len.back();
    } else {
      r.push(s[i], d[i], 1);
    }
  }
  return r;
}

Runs Runs::coalesce() const {
  if (size() <= 1) return *this;
  Runs r;
  r.reserve(size());
  for (size_t k : argsort(dst)) {
    if (!r.empty() && src[k] == r.src.back() + r.len.back() && dst[k] == r.dst.back() + r.len.back()) {
      r.len.back() += len[k];
    } else {
      r.push(src[k], dst[k], len[k]);
    }
  }
  return r;
}

Runs Runs::clip_src(std::span<const i64> vstart, std::span<const i64> vlen) const {
  require(vstart.size() == vlen.size(), "valid_src: start/len lengths differ");
  for (i64 n : vlen) require(n >= 0, "valid_src: negative length");
  const Intervals v = interval_union(vstart, vlen);
  Runs r;
  for (size_t i = 0; i < size(); ++i) {
    const i64 so = src[i], end = src[i] + len[i];
    // first valid interval that ends after so
    auto it = std::upper_bound(v.end.begin(), v.end.end(), so);
    for (size_t j = it - v.end.begin(); j < v.start.size() && v.start[j] < end; ++j) {
      const i64 lo = std::max(so, v.start[j]), hi = std::min(end, v.end[j]);
      if (lo < hi) r.push(lo, dst[i] + (lo - so), hi - lo);
    }
  }
  return r;
}

std::vector<Runs> Runs::chunks(i64 max_elems) const {
  require(max_elems > 0, "chunks: max_elems must be positive");
  std::vector<Runs> out(1);
  i64 acc = 0;
  for (size_t i = 0; i < size(); ++i) {
    i64 s = src[i], d = dst[i], n = len[i];
    while (n > 0) {
      const i64 take = std::min(n, max_elems - acc);
      out.back().push(s, d, take);
      s += take, d += take, n -= take, acc += take;
      if (acc == max_elems) {
        out.emplace_back();
        acc = 0;
      }
    }
  }
  if (out.back().empty()) out.pop_back();
  return out;
}

void Runs::expand(std::vector<i64>& si, std::vector<i64>& di) const {
  for (size_t i = 0; i < size(); ++i)
    for (i64 k = 0; k < len[i]; ++k) {
      si.push_back(src[i] + k);
      di.push_back(dst[i] + k);
    }
}

// ---- Intervals -----------------------------------------------------------------

Intervals interval_union(std::span<const i64> off, std::span<const i64> len, i64 scale) {
  Intervals out;
  std::vector<i64> s(off.begin(), off.end());
  for (size_t k : argsort(s)) {
    if (len[k] <= 0) continue;
    const i64 a = off[k] * scale, b = (off[k] + len[k]) * scale;
    if (!out.empty() && a <= out.end.back()) {
      out.end.back() = std::max(out.end.back(), b);
    } else {
      out.start.push_back(a);
      out.end.push_back(b);
    }
  }
  return out;
}

bool intersects(const Intervals& a, const Intervals& b) {
  if (a.empty() || b.empty() || a.hi() <= b.lo() || b.hi() <= a.lo()) return false;
  size_t i = 0, j = 0;
  while (i < a.start.size() && j < b.start.size()) {
    if (a.start[i] < b.end[j] && b.start[j] < a.end[i]) return true;
    if (a.end[i] <= b.end[j]) ++i;
    else ++j;
  }
  return false;
}

// ---- Rects ---------------------------------------------------------------------

void Rects::push(i64 s, i64 d, i64 w, i64 h, i64 sp, i64 dp) {
  src.push_back(s), dst.push_back(d), width.push_back(w), height.push_back(h);
  src_pitch.push_back(sp), dst_pitch.push_back(dp);
}

bool Rects::has_2d() const {
  return std::any_of(height.begin(), height.end(), [](i64 h) { return h > 1; });
}

void Rects::expand(std::vector<i64>& si, std::vector<i64>& di) const {
  for (size_t i = 0; i < size(); ++i)
    for (i64 r = 0; r < height[i]; ++r)
      for (i64 c = 0; c < width[i]; ++c) {
        si.push_back(src[i] + r * src_pitch[i] + c);
        di.push_back(dst[i] + r * dst_pitch[i] + c);
      }
}

Rects recover_2d(const Runs& runs, int min_height) {
  const auto &s = runs.src, &d = runs.dst, &n = runs.len;
  const size_t m = runs.size();
  const size_t need = static_cast<size_t>(std::max(min_height, 2));
  Rects out;
  size_t i = 0;
  while (i < m) {
    size_t j = i;
    i64 sp = 0, dp = 0;
    if (i + 1 < m && n[i + 1] == n[i]) {
      sp = s[i + 1] - s[i];
      dp = d[i + 1] - d[i];
      if (sp >= n[i] && dp >= n[i]) {
        j = i + 1;
        while (j + 1 < m && n[j + 1] == n[i] && s[j + 1] - s[j] == sp && d[j + 1] - d[j] == dp) ++j;
      }
    }
    if (j - i + 1 >= need) {
      out.push(s[i], d[i], n[i], static_cast<i64>(j - i + 1), sp, dp);
      i = j + 1;
    } else {
      out.push(s[i], d[i], n[i], 1, n[i], n[i]);
      ++i;
    }
  }
  return out;
}

// ---- Access --------------------------------------------------------------------

Access Access::contiguous(std::span<const i64> shape, i64 offset) {
  std::vector<i64> strides(shape.size());
  i64 acc = 1;
  for (size_t k = shape.size(); k-- > 0;) {
    strides[k] = acc;
    acc *= shape[k];
  }
  return strided(shape, strides, offset);
}

Access Access::strided(std::span<const i64> shape, std::span<const i64> strides, i64 offset) {
  require(shape.size() == strides.size(), "strided: shape/strides rank differ");
  std::vector<Term> t;
  for (size_t k = 0; k < shape.size(); ++k) {
    require(shape[k] >= 0, "strided: negative size");
    t.push_back({shape[k], strides[k], nullptr});
  }
  return Access(offset, std::move(t));
}

std::vector<i64> Access::shape() const {
  std::vector<i64> s;
  for (auto& t : terms_) s.push_back(t.size);
  return s;
}

bool Access::has_table() const {
  return std::any_of(terms_.begin(), terms_.end(), [](const Term& t) { return t.table != nullptr; });
}

i64 Access::numel() const {
  i64 n = 1;
  for (auto& t : terms_) n *= t.size;
  return n;
}

int Access::norm_dim(int dim, size_t rank) const {
  const int r = static_cast<int>(rank);
  if (dim < -r || dim >= r) fail(Errc::invalid_argument, "dim " + std::to_string(dim) + " out of range");
  return dim < 0 ? dim + r : dim;
}

Access Access::permute(std::span<const int> perm) const {
  require(perm.size() == rank(), "permute: wrong number of dims");
  std::vector<bool> seen(rank());
  std::vector<Term> t;
  for (int p : perm) {
    const int d = norm_dim(p, rank());
    require(!seen[d], "permute: repeated dim");
    seen[d] = true;
    t.push_back(terms_[d]);
  }
  return Access(base_, std::move(t));
}

Access Access::slice(int dim, std::optional<i64> start, std::optional<i64> stop, i64 step) const {
  dim = norm_dim(dim, rank());
  require(step != 0, "slice: step must be nonzero");
  const Term& t = terms_[dim];
  const i64 n = t.size;
  // Python slice.indices semantics.
  auto adjust = [&](std::optional<i64> v, i64 dflt, i64 lo, i64 hi) {
    if (!v) return dflt;
    i64 x = *v;
    if (x < 0) x = std::max(x + n, lo);
    else x = std::min(x, hi);
    return x;
  };
  const i64 b = step > 0 ? adjust(start, 0, 0, n) : adjust(start, n - 1, -1, n - 1);
  const i64 e = step > 0 ? adjust(stop, n, 0, n) : adjust(stop, -1, -1, n - 1);
  i64 count = 0;
  if (step > 0 && b < e) count = (e - b + step - 1) / step;
  if (step < 0 && e < b) count = (b - e + (-step) - 1) / (-step);

  std::vector<Term> terms = terms_;
  i64 base = base_;
  if (!t.table) {
    if (count > 0) base += b * t.stride;
    terms[dim] = {count, t.stride * step, nullptr};
  } else {
    auto v = std::make_shared<std::vector<i64>>();
    for (i64 k = 0; k < count; ++k) v->push_back((*t.table)[b + k * step]);
    terms[dim] = {count, t.stride, std::move(v)};
  }
  return Access(base, std::move(terms));
}

Access Access::select(int dim, i64 index) const {
  dim = norm_dim(dim, rank());
  const Term& t = terms_[dim];
  if (index < -t.size || index >= t.size) fail(Errc::invalid_argument, "select: index out of range");
  if (index < 0) index += t.size;
  const i64 base = base_ + t.stride * (t.table ? (*t.table)[index] : index);
  std::vector<Term> terms = terms_;
  terms.erase(terms.begin() + dim);
  return Access(base, std::move(terms));
}

Access Access::flip(int dim) const {
  dim = norm_dim(dim, rank());
  const Term& t = terms_[dim];
  std::vector<Term> terms = terms_;
  i64 base = base_;
  if (!t.table) {
    if (t.size > 0) base += (t.size - 1) * t.stride;
    terms[dim] = {t.size, -t.stride, nullptr};
  } else {
    auto v = std::make_shared<std::vector<i64>>(t.table->rbegin(), t.table->rend());
    terms[dim] = {t.size, t.stride, std::move(v)};
  }
  return Access(base, std::move(terms));
}

Access Access::index_select(int dim, std::span<const i64> index) const {
  dim = norm_dim(dim, rank());
  const Term& t = terms_[dim];
  auto v = std::make_shared<std::vector<i64>>();
  v->reserve(index.size());
  for (i64 i : index) {
    if (i < -t.size || i >= t.size) fail(Errc::invalid_argument, "index_select: index out of range");
    if (i < 0) i += t.size;
    v->push_back(t.table ? (*t.table)[i] : i);
  }
  std::vector<Term> terms = terms_;
  terms[dim] = {static_cast<i64>(index.size()), t.stride, std::move(v)};
  return Access(base_, std::move(terms));
}

Access Access::unsqueeze(int dim) const {
  dim = norm_dim(dim, rank() + 1);
  std::vector<Term> terms = terms_;
  terms.insert(terms.begin() + dim, Term{1, 0, nullptr});
  return Access(base_, std::move(terms));
}

Access Access::expand(int dim, i64 n) const {
  dim = norm_dim(dim, rank());
  const Term& t = terms_[dim];
  require(t.size == 1, "expand: dim must have size 1");
  require(n >= 0, "expand: negative size");
  const i64 base = base_ + t.stride * (t.table ? (*t.table)[0] : 0);
  std::vector<Term> terms = terms_;
  terms[dim] = {n, 0, nullptr};
  return Access(base, std::move(terms));
}

namespace {
// Port of torch's computeStride: can (oshape, ostrides) be viewed as nshape?
std::optional<std::vector<i64>> view_strides(const std::vector<i64>& oshape,
                                             const std::vector<i64>& ostrides,
                                             std::span<const i64> nshape) {
  i64 numel = 1;
  for (i64 s : oshape) numel *= s;
  if (oshape.empty() || numel == 0) return std::nullopt;
  std::vector<i64> nstrides(nshape.size(), 0);
  i64 view_d = static_cast<i64>(nshape.size()) - 1;
  i64 chunk_base = ostrides.back();
  i64 tensor_numel = 1, view_numel = 1;
  for (i64 d = static_cast<i64>(oshape.size()) - 1; d >= 0; --d) {
    tensor_numel *= oshape[d];
    if (d == 0 || (oshape[d - 1] != 1 && ostrides[d - 1] != tensor_numel * chunk_base)) {
      while (view_d >= 0 && (view_numel < tensor_numel || nshape[view_d] == 1)) {
        nstrides[view_d] = view_numel * chunk_base;
        view_numel *= nshape[view_d];
        --view_d;
      }
      if (view_numel != tensor_numel) return std::nullopt;
      if (d > 0) {
        chunk_base = ostrides[d - 1];
        tensor_numel = view_numel = 1;
      }
    }
  }
  if (view_d != -1) return std::nullopt;
  return nstrides;
}
}  // namespace

Access Access::reshape(std::span<const i64> nshape) const {
  i64 n = 1;
  for (i64 s : nshape) {
    require(s >= 0, "reshape: negative size");
    n *= s;
  }
  require(n == numel(), "reshape: element count differs");
  if (has_table()) fail(Errc::needs_materialize, "reshape across a table dimension");
  std::vector<i64> st;
  for (auto& t : terms_) st.push_back(t.stride);
  auto ns = view_strides(shape(), st, nshape);
  if (!ns) fail(Errc::needs_materialize, "reshape is not view-compatible");
  return strided(nshape, *ns, base_);
}

namespace {
std::vector<i64> term_values(const Term& t) {
  std::vector<i64> v(t.size);
  for (i64 i = 0; i < t.size; ++i) v[i] = t.stride * (t.table ? (*t.table)[i] : i);
  return v;
}

// Row-major outer sum of the first k dims' values plus base.
std::vector<i64> outer_offsets(const std::vector<std::vector<i64>>& vals, size_t k, i64 base) {
  std::vector<i64> off{base};
  for (size_t d = 0; d < k; ++d) {
    std::vector<i64> next;
    next.reserve(off.size() * vals[d].size());
    for (i64 o : off)
      for (i64 v : vals[d]) next.push_back(o + v);
    off.swap(next);
  }
  return off;
}
}  // namespace

std::vector<i64> Access::offsets() const {
  std::vector<std::vector<i64>> vals;
  for (auto& t : terms_) vals.push_back(term_values(t));
  return outer_offsets(vals, vals.size(), base_);
}

std::pair<i64, i64> Access::extent() const {
  i64 lo = base_, hi = base_;
  for (auto& t : terms_) {
    if (t.size == 0) continue;
    i64 mn = INT64_MAX, mx = INT64_MIN;
    if (t.table) {
      for (i64 v : *t.table) mn = std::min(mn, v * t.stride), mx = std::max(mx, v * t.stride);
    } else {
      mn = std::min<i64>(0, (t.size - 1) * t.stride);
      mx = std::max<i64>(0, (t.size - 1) * t.stride);
    }
    lo += mn, hi += mx;
  }
  return {lo, hi};
}

Access broadcast_to(const Access& a, std::span<const i64> shape) {
  if (a.rank() > shape.size()) fail(Errc::invalid_argument, "cannot broadcast to a lower rank");
  Access r = a;
  while (r.rank() < shape.size()) r = r.unsqueeze(0);
  for (size_t d = 0; d < shape.size(); ++d) {
    const i64 n = r.terms()[d].size;
    if (n == shape[d]) continue;
    if (n != 1) fail(Errc::invalid_argument, "shapes are not broadcastable");
    r = r.expand(static_cast<int>(d), shape[d]);
  }
  return r;
}

// ---- Stmt / lowering -----------------------------------------------------------

Stmt::Stmt(Access d, Access s) : dst(std::move(d)), src(std::move(s)) {
  if (dst.shape() != src.shape()) fail(Errc::invalid_argument, "stmt: src and dst domains differ");
}

Runs lower(const Stmt& stmt) {
  const auto& st = stmt.src.terms();
  const auto& dt = stmt.dst.terms();
  const size_t rank = st.size();
  Runs out;
  for (auto& t : st)
    if (t.size == 0) return out;

  std::vector<std::vector<i64>> sv, dv;
  for (size_t k = 0; k < rank; ++k) {
    sv.push_back(term_values(st[k]));
    dv.push_back(term_values(dt[k]));
  }
  // Longest trailing block that is row-major on both sides.
  size_t k = rank;
  i64 inner = 1;
  auto is_rowmajor = [&](const std::vector<i64>& v) {
    for (size_t i = 0; i < v.size(); ++i)
      if (v[i] != static_cast<i64>(i) * inner) return false;
    return true;
  };
  while (k > 0 && is_rowmajor(sv[k - 1]) && is_rowmajor(dv[k - 1])) {
    inner *= st[k - 1].size;
    --k;
  }
  const std::vector<i64> so = outer_offsets(sv, k, stmt.src.base());
  const std::vector<i64> d_o = outer_offsets(dv, k, stmt.dst.base());
  for (size_t i = 0; i < so.size(); ++i) {
    if (!out.empty() && so[i] == out.src.back() + out.len.back() &&
        d_o[i] == out.dst.back() + out.len.back()) {
      out.len.back() += inner;
    } else {
      out.push(so[i], d_o[i], inner);
    }
  }
  return out;
}

}  // namespace ce
