#pragma once

// The structures the CUDA executor writes and the kernels read: defined once,
// here, and included by both (executor.cpp with the host compiler,
// kernels.cu with nvcc). Rules, so the two sides agree byte for byte
// whatever the compilers and CUDA headers:
// - fixed-width types only; device addresses are uint64_t (the kernels cast);
// - no CUDA header types: driver objects (a CUtensorMap) are bytes;
// - explicit alignment where it matters, and static_asserts on sizes and
//   offsets, checked by both compilers;
// - layout_fingerprint() folds every size and offset into one number: the
//   executor compares the kernels' own (ce_layout) with its own before it
//   admits them, so a mismatch disables the kernels instead of corrupting
//   their tables.

#include <cstddef>
#include <cstdint>

#include "core/tile.hpp"  // cv::TileK, shared the same way

namespace ce::cuabi {

// Rows x width elements at constant pitches (element indices, see the executor).
struct Chunk {
  int64_t src, dst, width, rows, spitch, dpitch;
};
static_assert(sizeof(Chunk) == 48);

// A structured statement: up to 6 dimensions, shapes with their division
// constants, strides per side, device index tables per side.
struct StmtDesc {
  int32_t rank, sdt, ddt, mode;  // mode 0: raw element copy, 1: cast, 2: apply fn
  int32_t vec;                   // destination elements per thread (innermost dimension)
  uint32_t numel;
  uint32_t shape[6];
  uint32_t magic[6], shift[6];  // n / shape[k] == (umulhi(n, magic) + n) >> shift, for n < 2^31
  int64_t ss[6], ds[6];
  // Device index tables: on dimension stab_dim (dtab_dim) the index is read
  // from Base::stab (dtab), step int64s apart, and must lie in [0, bound);
  // an index outside it skips the element and sets the word at `err`. -1: no table.
  int32_t stab_dim, dtab_dim;
  int64_t stab_step, dtab_step, sbound, dbound;
  uint64_t err;
};
static_assert(sizeof(StmtDesc) == 240 && offsetof(StmtDesc, ss) == 96 && offsetof(StmtDesc, stab_dim) == 192 &&
              offsetof(StmtDesc, err) == 232);

// A statement's bases and its device index tables (0: none).
struct Base {
  int64_t src, dst;
  uint64_t stab, dtab;
};
static_assert(sizeof(Base) == 32);

// ce_put: words carried in the kernel's parameters (under the 4 KB limit).
constexpr int kPutWords = 480;
struct Put {
  uint64_t dst;
  int32_t n;
  uint64_t w[kPutWords];
};
static_assert(offsetof(Put, w) == 16 && sizeof(Put) == 16 + 8 * kPutWords && sizeof(Put) <= 4096);

// Bulk copies: segments in flight per block.
constexpr int kBulkBufs = 4;

// tma2d: a source and a destination tensor map (CUtensorMap's 128 bytes; its
// own alignment is 64 in some CUDA headers, 128 in others), the block (bytes
// x rows) and the box.
constexpr int kTmaBox = 8192;  // bytes per box, two buffers per block
struct alignas(128) TmaBlock {
  unsigned char src[128], dst[128];
  int32_t width, height, bw, bh;
};
static_assert(sizeof(TmaBlock) == 384 && offsetof(TmaBlock, dst) == 128 && offsetof(TmaBlock, width) == 256);

// Rows copied within one pool whose indices may overlap (pool[d] = pool[s],
// torch's gather-first result), in three kernels on one stream: ce_ov_mark
// marks the destination rows, ce_ov_stage copies the source rows that are
// also destination rows to staging, ce_ov_scatter writes every destination
// row from staging or the pool. Marks are epochs: the current one lives in
// device memory (epoch[0]; epoch[1] is the next, written by ce_ov_stage and
// made current by ce_ov_scatter), so a CUDA graph replay advances it too.
// Rows are row_bytes contiguous bytes, row_stride bytes apart from `pool`.
struct RowsOv {
  uint64_t pool;
  int64_t row_stride, row_bytes;
  uint64_t stab, dtab;  // int64 indices, sstep / dstep apart
  int64_t sstep, dstep;
  int64_t n, rows;      // indices per side; every index lies in [0, rows)
  uint64_t marks;       // uint64 per row
  uint64_t epoch;       // two uint64 words
  uint64_t staging;     // n * row_bytes
  uint64_t err;         // set (and the element skipped) for an index outside [0, rows)
};
static_assert(sizeof(RowsOv) == 104 && offsetof(RowsOv, marks) == 72 && offsetof(RowsOv, err) == 96);

// Every size and offset above, and the tile descriptor's, in one number.
#if defined(__CUDACC__)
__host__ __device__
#endif
constexpr uint64_t layout_fingerprint() {
  const uint64_t v[] = {sizeof(Chunk),          sizeof(StmtDesc),           offsetof(StmtDesc, ss),
                        offsetof(StmtDesc, stab_dim), offsetof(StmtDesc, err), sizeof(Base),
                        sizeof(Put),            offsetof(Put, w),           sizeof(TmaBlock),
                        offsetof(TmaBlock, width), alignof(TmaBlock),       sizeof(cv::TileK),
                        offsetof(cv::TileK, rows), offsetof(cv::TileK, g),  offsetof(cv::TileK, xi),
                        offsetof(cv::TileK, err), kPutWords,                kBulkBufs,
                        kTmaBox,                sizeof(RowsOv),             offsetof(RowsOv, marks),
                        offsetof(RowsOv, err)};
  uint64_t h = 1469598103934665603ull;
  for (uint64_t x : v) h = (h ^ x) * 1099511628211ull;
  return h;
}

}  // namespace ce::cuabi
