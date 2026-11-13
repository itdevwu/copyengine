// Table-driven copy kernels, compiled AOT into a fatbin and loaded through the
// driver API. One warp per chunk; chunks are (src, dst, len) in absolute
// element indices, so a pointer is index * sizeof(T).
//
// Build with --fmad=false: conversions must match the host backend bit for bit.

#include "core/convert.hpp"

namespace {

struct Chunk {
  long long src, dst, len;
};

using namespace ce::cv;

template <class D, class S, bool kIdentity>
__device__ __forceinline__ void typed(const Chunk* __restrict__ chunks, int n, Fn fn) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  const Chunk c = chunks[warp];
  const S* __restrict__ s = reinterpret_cast<const S*>(static_cast<uintptr_t>(c.src) * sizeof(S));
  D* __restrict__ d = reinterpret_cast<D*>(static_cast<uintptr_t>(c.dst) * sizeof(D));
  for (long long i = lane; i < c.len; i += 32) {
    if constexpr (kIdentity) d[i] = cast<D>(s[i]);
    else d[i] = apply<D>(s[i], fn);
  }
}

template <class V>
__device__ __forceinline__ void move(uintptr_t s, uintptr_t d, long long nbytes, int lane) {
  const V* __restrict__ sv = reinterpret_cast<const V*>(s);
  V* __restrict__ dv = reinterpret_cast<V*>(d);
  const long long n = nbytes / static_cast<long long>(sizeof(V));
  for (long long i = lane; i < n; i += 32) dv[i] = sv[i];
}

}  // namespace

// Pure byte copies: widest vector the chunk's alignment allows.
extern "C" __global__ void ce_bytes(const Chunk* __restrict__ chunks, int n) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  const Chunk c = chunks[warp];
  const uintptr_t s = static_cast<uintptr_t>(c.src), d = static_cast<uintptr_t>(c.dst);
  const uintptr_t a = s | d | static_cast<uintptr_t>(c.len);
  if ((a & 15) == 0) move<uint4>(s, d, c.len, lane);
  else if ((a & 7) == 0) move<uint2>(s, d, c.len, lane);
  else if ((a & 3) == 0) move<uint32_t>(s, d, c.len, lane);
  else move<uint8_t>(s, d, c.len, lane);
}

#define CE_TYPES(X, ...)            \
  X(__VA_ARGS__, f64, double)       \
  X(__VA_ARGS__, f32, float)        \
  X(__VA_ARGS__, f16, half_t)       \
  X(__VA_ARGS__, bf16, bf16_t)      \
  X(__VA_ARGS__, i64, int64_t)      \
  X(__VA_ARGS__, i32, int32_t)      \
  X(__VA_ARGS__, i16, int16_t)      \
  X(__VA_ARGS__, i8, int8_t)        \
  X(__VA_ARGS__, u8, uint8_t)       \
  X(__VA_ARGS__, b8, bool_t)

// The preprocessor won't re-expand CE_TYPES inside itself, hence a twin.
#define CE_TYPES_INNER(X, ...)            \
  X(__VA_ARGS__, f64, double)       \
  X(__VA_ARGS__, f32, float)        \
  X(__VA_ARGS__, f16, half_t)       \
  X(__VA_ARGS__, bf16, bf16_t)      \
  X(__VA_ARGS__, i64, int64_t)      \
  X(__VA_ARGS__, i32, int32_t)      \
  X(__VA_ARGS__, i16, int16_t)      \
  X(__VA_ARGS__, i8, int8_t)        \
  X(__VA_ARGS__, u8, uint8_t)       \
  X(__VA_ARGS__, b8, bool_t)

#define CE_KERNEL(sn, st, dn, dt)                                                               \
  extern "C" __global__ void ce_cast_##sn##_##dn(const Chunk* __restrict__ c, int n, Fn fn) { \
    typed<dt, st, true>(c, n, fn);                                                             \
  }                                                                                             \
  extern "C" __global__ void ce_apply_##sn##_##dn(const Chunk* __restrict__ c, int n, Fn fn) { \
    typed<dt, st, false>(c, n, fn);                                                            \
  }
#define CE_ROW(sn, st) CE_TYPES_INNER(CE_KERNEL, sn, st)
#define CE_ALL(unused, sn, st) CE_ROW(sn, st)
CE_TYPES(CE_ALL, _)
