// Table-driven copy kernels, compiled AOT into a fatbin and loaded through the
// driver API. Tables stay resident in device memory; `sadd` / `dadd` are added
// to every source / destination index (the staging base, which can move).
// One warp per chunk. A chunk is a 2-D block in absolute element
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

// An unsigned integer of exactly N bytes, for vector loads and stores.
template <int N> struct Word;
template <> struct Word<1> { using type = uint8_t; };
template <> struct Word<2> { using type = uint16_t; };
template <> struct Word<4> { using type = uint32_t; };
template <> struct Word<8> { using type = uint2; };
template <> struct Word<16> { using type = uint4; };

// V elements of T moved as one access.
template <class T, int V>
union Pack {
  typename Word<sizeof(T) * V>::type word;
  T v[V];
};

template <class D, class S, bool kIdentity>
__device__ __forceinline__ D convert(S x, const Fn& fn) {
  if constexpr (kIdentity) return cast<D>(x);
  else return apply<D>(x, fn);
}

template <class D, class S, bool kIdentity>
__device__ __forceinline__ void typed(const Chunk* __restrict__ chunks, int n, Fn fn, long long sadd, long long dadd) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  Chunk c = chunks[warp];
  c.src += sadd, c.dst += dadd;
  const S* __restrict__ s = reinterpret_cast<const S*>(static_cast<uintptr_t>(c.src) * sizeof(S));
  D* __restrict__ d = reinterpret_cast<D*>(static_cast<uintptr_t>(c.dst) * sizeof(D));
  auto one = [&](long long so, long long dof) { d[dof] = convert<D, S, kIdentity>(s[so], fn); };

  // Each lane moves V elements per access: 16 bytes on the wider side. The
  // conversion is still per element, so results do not change.
  constexpr int V = 16 / (sizeof(S) > sizeof(D) ? sizeof(S) : sizeof(D));
  const bool aligned = ((c.src | c.dst | c.spitch | c.dpitch) & (V - 1)) == 0;
  if (aligned && (c.width >= 32 * V || c.rows == 1)) {
    const long long nv = c.width / V;
    for (long long r = 0; r < c.rows; ++r) {
      const auto* sv = reinterpret_cast<const Pack<S, V>*>(s + r * c.spitch);
      auto* dv = reinterpret_cast<Pack<D, V>*>(d + r * c.dpitch);
#pragma unroll 4
      for (long long i = lane; i < nv; i += 32) {
        Pack<S, V> in;
        in.word = sv[i].word;
        Pack<D, V> out;
#pragma unroll
        for (int k = 0; k < V; ++k) out.v[k] = convert<D, S, kIdentity>(in.v[k], fn);
        dv[i].word = out.word;
      }
      for (long long i = nv * V + lane; i < c.width; i += 32) one(r * c.spitch + i, r * c.dpitch + i);
    }
  } else if (c.width >= 32 || c.rows == 1) {
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
extern "C" __global__ void ce_bytes(const Chunk* __restrict__ chunks, int n, long long sadd, long long dadd) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n) return;
  Chunk c = chunks[warp];
  c.src += sadd, c.dst += dadd;
  const uintptr_t a = static_cast<uintptr_t>(c.src | c.dst | c.width | c.spitch | c.dpitch);
  for (long long r = 0; r < c.rows; ++r) {
    const uintptr_t s = static_cast<uintptr_t>(c.src + r * c.spitch), d = static_cast<uintptr_t>(c.dst + r * c.dpitch);
    if ((a & 15) == 0) move<uint4>(s, d, c.width, lane);
    else if ((a & 7) == 0) move<uint2>(s, d, c.width, lane);
    else if ((a & 3) == 0) move<uint32_t>(s, d, c.width, lane);
    else move<uint8_t>(s, d, c.width, lane);
  }
}

// Structured statements: dst[D(i)] = f(src[S(i)]) over a shared domain, for a
// batch of statements with the same shape and strides but different bases.
// One thread per element; strides may be 0 (broadcast) or negative. Dtypes are
// dispatched at runtime (same bits as the typed kernels, see convert.hpp).
struct StmtDesc {
  int rank, sdt, ddt, mode;  // mode 0: raw element copy, 1: cast, 2: apply fn
  unsigned numel;
  unsigned shape[6];
  long long ss[6], ds[6];
};

struct Base {
  long long src, dst;
};

extern "C" __global__ void ce_stmt(const Base* __restrict__ bases, long long total, StmtDesc D, Fn fn, long long sadd,
                                   long long dadd) {
  const long long e = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  if (e >= total) return;
  const long long op = e / D.numel;
  unsigned r = static_cast<unsigned>(e - op * D.numel);
  long long so = bases[op].src + sadd, dof = bases[op].dst + dadd;
  for (int k = D.rank - 1; k >= 0; --k) {
    const unsigned q = r / D.shape[k], i = r - q * D.shape[k];
    r = q;
    so += static_cast<long long>(i) * D.ss[k];
    dof += static_cast<long long>(i) * D.ds[k];
  }
  const void* sp = reinterpret_cast<const void*>(static_cast<uintptr_t>(so) * dt_size(D.sdt));
  void* dp = reinterpret_cast<void*>(static_cast<uintptr_t>(dof) * dt_size(D.ddt));
  if (D.mode == 0) {
    switch (dt_size(D.sdt)) {
      case 8: wr<uint64_t>(dp, rd<uint64_t>(sp)); break;
      case 4: wr<uint32_t>(dp, rd<uint32_t>(sp)); break;
      case 2: wr<uint16_t>(dp, rd<uint16_t>(sp)); break;
      default: wr<uint8_t>(dp, rd<uint8_t>(sp)); break;
    }
  } else if (D.mode == 1) {
    dyn_cast(sp, D.sdt, dp, D.ddt);
  } else {
    dyn_apply(sp, D.sdt, dp, D.ddt, fn);
  }
}

// The same statements when the destination's contiguous dimension `a` is not
// the source's `b` (transposes): 32x32 tiles through shared memory, read
// along b and written along a, so both sides are coalesced. Converted on the
// way in; each slot holds one destination element. Block (32, 8); grid
// (tiles of b, tiles of a, statements x remaining index, strided).
extern "C" __global__ void ce_stmt_tile(const Base* __restrict__ bases, long long count, StmtDesc D, Fn fn, int a,
                                        int b, long long sadd, long long dadd) {
  __shared__ uint64_t tile[32][33];
  const long long sa = D.shape[a], sb = D.shape[b];
  const long long outer = static_cast<long long>(D.numel) / (sa * sb);
  const long long ia0 = static_cast<long long>(blockIdx.y) * 32, ib0 = static_cast<long long>(blockIdx.x) * 32;
  const int ssz = dt_size(D.sdt), dsz = dt_size(D.ddt);
  for (long long z = blockIdx.z; z < count; z += gridDim.z) {
    const long long op = z / outer;
    long long r = z - op * outer;
    long long so = bases[op].src + sadd, dof = bases[op].dst + dadd;
    for (int k = D.rank - 1; k >= 0; --k) {
      if (k == a || k == b) continue;
      const long long q = r / D.shape[k], i = r - q * D.shape[k];
      r = q;
      so += i * D.ss[k];
      dof += i * D.ds[k];
    }
    for (int j = threadIdx.y; j < 32; j += 8) {
      const long long ia = ia0 + j, ib = ib0 + threadIdx.x;
      if (ia < sa && ib < sb) {
        const void* sp = reinterpret_cast<const void*>(static_cast<uintptr_t>(so + ia * D.ss[a] + ib * D.ss[b]) * ssz);
        void* slot = &tile[j][threadIdx.x];
        if (D.mode == 0) {
          switch (ssz) {
            case 8: wr<uint64_t>(slot, rd<uint64_t>(sp)); break;
            case 4: wr<uint32_t>(slot, rd<uint32_t>(sp)); break;
            case 2: wr<uint16_t>(slot, rd<uint16_t>(sp)); break;
            default: wr<uint8_t>(slot, rd<uint8_t>(sp)); break;
          }
        } else if (D.mode == 1) {
          dyn_cast(sp, D.sdt, slot, D.ddt);
        } else {
          dyn_apply(sp, D.sdt, slot, D.ddt, fn);
        }
      }
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += 8) {
      const long long ib = ib0 + j, ia = ia0 + threadIdx.x;
      if (ia < sa && ib < sb) {
        void* dp = reinterpret_cast<void*>(static_cast<uintptr_t>(dof + ia * D.ds[a] + ib * D.ds[b]) * dsz);
        const void* slot = &tile[threadIdx.x][j];
        switch (dsz) {
          case 8: wr<uint64_t>(dp, rd<uint64_t>(slot)); break;
          case 4: wr<uint32_t>(dp, rd<uint32_t>(slot)); break;
          case 2: wr<uint16_t>(dp, rd<uint16_t>(slot)); break;
          default: wr<uint8_t>(dp, rd<uint8_t>(slot)); break;
        }
      }
    }
    __syncthreads();
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
  extern "C" __global__ void ce_cast_##sn##_##dn(const Chunk* __restrict__ c, int n, Fn fn, long long sa, \
                                                  long long da) {                                         \
    typed<dt, st, true>(c, n, fn, sa, da);                                                     \
  }                                                                                             \
  extern "C" __global__ void ce_apply_##sn##_##dn(const Chunk* __restrict__ c, int n, Fn fn, long long sa, \
                                                   long long da) {                                         \
    typed<dt, st, false>(c, n, fn, sa, da);                                                    \
  }
#define CE_ROW(sn, st) CE_TYPES_INNER(CE_KERNEL, sn, st)
#define CE_ALL(unused, sn, st) CE_ROW(sn, st)
CE_TYPES(CE_ALL, _)
