// Table-driven copy kernels, compiled AOT into a fatbin and loaded through the
// driver API. One warp per chunk. A chunk is a 2-D block in absolute element
// indices (a pointer is index * sizeof(T)): `rows` rows of `width` elements,
// row r at src + r * spitch / dst + r * dpitch. spitch may be 0 (broadcast) or
// smaller than width; rows never overlap on the destination.
//
// Build with --fmad=false: conversions must match the host backend bit for bit.

#include "core/convert.hpp"

namespace {

struct Chunk {
  long long src, dst, width, rows, spitch, dpitch;
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
  auto one = [&](long long so, long long dof) {
    if constexpr (kIdentity) d[dof] = cast<D>(s[so]);
    else d[dof] = apply<D>(s[so], fn);
  };
  if (c.width >= 32 || c.rows == 1) {
    for (long long r = 0; r < c.rows; ++r)
      for (long long i = lane; i < c.width; i += 32) one(r * c.spitch + i, r * c.dpitch + i);
  } else {  // narrow rows (broadcast, element transposes): flatten so all lanes work
    const long long n = c.width * c.rows;
    for (long long e = lane; e < n; e += 32) {
      const long long r = e / c.width, i = e - r * c.width;
      one(r * c.spitch + i, r * c.dpitch + i);
    }
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
  const uintptr_t a = static_cast<uintptr_t>(c.src | c.dst | c.width | c.spitch | c.dpitch);
  for (long long r = 0; r < c.rows; ++r) {
    const uintptr_t s = static_cast<uintptr_t>(c.src + r * c.spitch), d = static_cast<uintptr_t>(c.dst + r * c.dpitch);
    if ((a & 15) == 0) move<uint4>(s, d, c.width, lane);
    else if ((a & 7) == 0) move<uint2>(s, d, c.width, lane);
    else if ((a & 3) == 0) move<uint32_t>(s, d, c.width, lane);
    else move<uint8_t>(s, d, c.width, lane);
  }
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
