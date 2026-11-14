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
  // Never merges at a cut (sorted positions on each side: allocation
  // boundaries, which one copy call must not cross).
  Runs coalesce(std::span<const i64> src_cuts = {}, std::span<const i64> dst_cuts = {}) const;
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
// Splits every interval at the cuts strictly inside it (cuts sorted).
Intervals split_at(const Intervals& iv, std::span<const i64> cuts);
// The cuts that are real boundaries: those not strictly inside any of the
// half-open ranges [start, start + len) (each known to be one allocation).
std::vector<i64> real_cuts(std::span<const i64> cuts, std::span<const i64> start, std::span<const i64> len);

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
// A block's rows all lie between the same two cuts on each side.
Rects recover_2d(const Runs& runs, int min_height = 2, std::span<const i64> src_cuts = {},
                 std::span<const i64> dst_cuts = {});

// ---- L4 / L3 -------------------------------------------------------------------

using Table = std::shared_ptr<const std::vector<i64>>;

// Indices that live in device memory: entry i is the int64 at
// ptr + i * step * 8 on `device`, valid in [0, bound). Their values are
// never known on the host: kernels read them.
struct DeviceTable {
  uintptr_t ptr = 0;  // 0: not a device table
  i64 step = 1;
  i64 bound = 0;
  int device = 0;
  bool operator==(const DeviceTable&) const = default;
};

// One dimension's contribution to the offset: stride * i, stride * table[i]
// (host values), or stride * dev[i] (a device table).
struct Term {
  i64 size = 0;
  i64 stride = 0;
  Table table;      // null: no host table
  DeviceTable dev;  // dev.ptr == 0: no device table
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
  bool has_table() const;         // host-valued table on some dimension
  bool has_device_table() const;  // device table on some dimension
  i64 numel() const;

  // Data-movement operators: pure algebra, no data is touched.
  Access permute(std::span<const int> perm) const;
  Access slice(int dim, std::optional<i64> start, std::optional<i64> stop, i64 step = 1) const;
  Access select(int dim, i64 index) const;
  Access flip(int dim) const;
  Access index_select(int dim, std::span<const i64> index) const;
  // Indices in device memory (n int64 values at `ptr`, contiguous) on `device`.
  // Only an affine dimension can be selected this way.
  Access index_select_device(int dim, uintptr_t ptr, i64 n, int device) const;
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

// Numpy-style broadcast of `a` to `shape` (leading unsqueezes, size-1 expands).
Access broadcast_to(const Access& a, std::span<const i64> shape);

struct Stmt {
  Access dst, src;
  Stmt(Access dst_, Access src_);
  std::vector<i64> domain() const { return dst.shape(); }
};

// L3 -> L1. Enumerates only the outer dimensions: the longest trailing block
// that is row-major on both sides becomes the run length.
Runs lower(const Stmt& stmt);

}  // namespace ce
