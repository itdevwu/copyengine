#pragma once

// Address-relation IR. A copy is dst[D(i)] = f(src[S(i)]) over a shared domain.
// Five levels, highest first; anything lowers downward, and structure is
// recovered upward after merging (failing to recover is slow, never wrong):
//
//   L4 Access  separable affine + table offsets per side
//   L3 Stmt    a pair of Accesses over one domain
//   L2 Rects   2-D blocks (off, width, height, pitch)
//   L1 Runs    1-D contiguous runs (src_off, dst_off, len)
//   L0 index pairs (src_idx, dst_idx)
//
// All offsets and lengths are in elements of the respective buffer.

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ce {

using i64 = int64_t;

// ---- L1 ----------------------------------------------------------------------

struct Runs {
  std::vector<i64> src, dst, len;

  size_t size() const { return len.size(); }
  bool empty() const { return len.empty(); }
  i64 total() const;
  void push(i64 s, i64 d, i64 n);
  void append(const Runs& other);
  void reserve(size_t n);

  // Drops zero-length runs; rejects negative lengths and offsets.
  static Runs of(std::span<const i64> src, std::span<const i64> dst, std::span<const i64> len);
  // L0 -> L1: maximal runs where both sides advance by one.
  static Runs from_indices(std::span<const i64> src, std::span<const i64> dst);

  // Sort by dst and merge runs that are adjacent on both sides.
  Runs coalesce() const;
  // Keep only the parts whose source lies in the given (possibly overlapping) intervals.
  Runs clip_src(std::span<const i64> valid_start, std::span<const i64> valid_len) const;
  // Split into pieces of at most max_elems elements, cutting long runs.
  std::vector<Runs> chunks(i64 max_elems) const;
  // L1 -> L0, for verification.
  void expand(std::vector<i64>& src_idx, std::vector<i64>& dst_idx) const;
};

// Disjoint, sorted half-open intervals [start, end).
struct Intervals {
  std::vector<i64> start, end;
  bool empty() const { return start.empty(); }
  i64 lo() const { return start.front(); }
  i64 hi() const { return end.back(); }
};

// Union of [off*scale, (off+len)*scale) — scale converts elements to bytes.
Intervals interval_union(std::span<const i64> off, std::span<const i64> len, i64 scale = 1);
bool intersects(const Intervals& a, const Intervals& b);

// ---- L2 ----------------------------------------------------------------------

struct Rects {
  std::vector<i64> src, dst, width, height, src_pitch, dst_pitch;
  size_t size() const { return width.size(); }
  void push(i64 s, i64 d, i64 w, i64 h, i64 sp, i64 dp);
  void expand(std::vector<i64>& src_idx, std::vector<i64>& dst_idx) const;
  bool has_2d() const;
};

// L1 -> L2: consecutive equal-length runs with constant src and dst pitch
// (pitch >= width on both sides) become one block of >= min_height rows.
Rects recover_2d(const Runs& runs, int min_height = 2);

// ---- L4 / L3 -------------------------------------------------------------------

using Table = std::shared_ptr<const std::vector<i64>>;

// One dimension's contribution to the offset: stride * i, or stride * table[i].
struct Term {
  i64 size = 0;
  i64 stride = 0;
  Table table;  // null: affine
};

// offset(i) = base + sum_d term_d(i_d). Tables are concrete runtime values
// (inspector-executor): composition materializes them immediately.
class Access {
 public:
  Access() = default;
  Access(i64 base, std::vector<Term> terms) : base_(base), terms_(std::move(terms)) {}

  static Access contiguous(std::span<const i64> shape, i64 offset = 0);
  static Access strided(std::span<const i64> shape, std::span<const i64> strides, i64 offset = 0);

  i64 base() const { return base_; }
  const std::vector<Term>& terms() const { return terms_; }
  std::vector<i64> shape() const;
  size_t rank() const { return terms_.size(); }
  bool has_table() const;
  i64 numel() const;

  // Data-movement operators: pure algebra, no data is touched.
  Access permute(std::span<const int> perm) const;
  Access slice(int dim, std::optional<i64> start, std::optional<i64> stop, i64 step = 1) const;
  Access select(int dim, i64 index) const;
  Access flip(int dim) const;
  Access index_select(int dim, std::span<const i64> index) const;
  Access unsqueeze(int dim) const;
  Access expand(int dim, i64 n) const;           // size-1 dim -> stride 0 (source side only)
  Access reshape(std::span<const i64> shape) const;  // view-compatible only, else needs_materialize

  // Per-element offsets in row-major domain order (verification only).
  std::vector<i64> offsets() const;
  // Min / max element offset touched (for bounds checks). Requires numel() > 0.
  std::pair<i64, i64> extent() const;

 private:
  int norm_dim(int dim, size_t rank) const;
  i64 base_ = 0;
  std::vector<Term> terms_;
};

struct Stmt {
  Access dst, src;
  Stmt(Access dst_, Access src_);
  std::vector<i64> domain() const { return dst.shape(); }
};

// L3 -> L1. Enumerates only the outer dimensions: the longest trailing block
// that is row-major on both sides becomes the run length.
Runs lower(const Stmt& stmt);

}  // namespace ce
