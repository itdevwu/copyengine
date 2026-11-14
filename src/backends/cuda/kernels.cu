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
__device__ __forceinline__ void typed_one(Chunk c, int lane, const Fn& fn) {
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

// Chunk kernels take any grid: each warp strides over the chunks (the grid
// may be capped to bound the SMs a copy occupies).
#define CE_FOR_CHUNKS(chunks, n, sadd, dadd)                                  \
  const int lane = threadIdx.x & 31;                                         \
  const int warps = static_cast<int>((gridDim.x * blockDim.x) >> 5);         \
  for (int w = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5); w < (n); w += warps) \
    if (Chunk c = (chunks)[w]; (c.src += (sadd), c.dst += (dadd), true))

template <class D, class S, bool kIdentity>
__device__ __forceinline__ void typed(const Chunk* __restrict__ chunks, int n, Fn fn, long long sadd, long long dadd) {
  CE_FOR_CHUNKS(chunks, n, sadd, dadd) typed_one<D, S, kIdentity>(c, lane, fn);
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
__device__ __forceinline__ void bytes_one(Chunk c, int lane) {
  const uintptr_t a = static_cast<uintptr_t>(c.src | c.dst | c.width | c.spitch | c.dpitch);
  for (long long r = 0; r < c.rows; ++r) {
    const uintptr_t s = static_cast<uintptr_t>(c.src + r * c.spitch), d = static_cast<uintptr_t>(c.dst + r * c.dpitch);
    if ((a & 15) == 0) move<uint4>(s, d, c.width, lane);
    else if ((a & 7) == 0) move<uint2>(s, d, c.width, lane);
    else if ((a & 3) == 0) move<uint32_t>(s, d, c.width, lane);
    else move<uint8_t>(s, d, c.width, lane);
  }
}

extern "C" __global__ void ce_bytes(const Chunk* __restrict__ chunks, int n, long long sadd, long long dadd) {
  CE_FOR_CHUNKS(chunks, n, sadd, dadd) bytes_one(c, lane);
}

// Structured statements: dst[D(i)] = f(src[S(i)]) over a shared domain, for a
// batch of statements with the same shape and strides but different bases.
// Strides may be 0 (broadcast) or negative. Dtypes are dispatched at runtime
// (same bits as the typed kernels, see convert.hpp). Each thread takes `vec`
// consecutive elements of the innermost dimension (the host checks it divides
// that dimension, which is contiguous and vec-aligned on the destination) and
// stores them as one access.
struct StmtDesc {
  int rank, sdt, ddt, mode;  // mode 0: raw element copy, 1: cast, 2: apply fn
  int vec;
  unsigned numel;
  unsigned shape[6];
  unsigned magic[6], shift[6];  // n / shape[k] == (umulhi(n, magic) + n) >> shift, for n < 2^31
  long long ss[6], ds[6];
  // Device index tables: on dimension stab_dim (dtab_dim) the index is read
  // from Base::stab (dtab), step int64s apart, and must lie in [0, bound);
  // an index outside it skips the element and sets *err. -1: no table.
  int stab_dim, dtab_dim;
  long long stab_step, dtab_step, sbound, dbound;
  unsigned long long* err;
};

struct Base {
  long long src, dst;
  const long long* stab;
  const long long* dtab;
};

constexpr int kPutWords = 480;  // ce_put's parameters stay under the 4 KB limit

__device__ __forceinline__ void element(const void* sp, void* dp, const StmtDesc& D, const Fn& fn) {
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

// Element conversion for statements: from the descriptor at run time ...
struct DynConv {
  const StmtDesc& D;
  const Fn& fn;
  __device__ __forceinline__ void operator()(const void* sp, void* dp) const { element(sp, dp, D, fn); }
};

// ... or compiled for one dtype pair (kMode 0: raw copy, 1: cast, 2: apply),
// for the common pairs where the run-time dispatch would dominate.
template <class Dt, class St, int kMode>
struct TypedConv {
  const Fn& fn;
  __device__ __forceinline__ void operator()(const void* sp, void* dp) const {
    const St x = rd<St>(sp);
    if constexpr (kMode == 0) wr<Dt>(dp, x);
    else wr<Dt>(dp, convert<Dt, St, kMode == 1>(x, fn));
  }
};

// Grid: x covers one statement's elements (vec per thread), y the statements
// (strided when there are more than gridDim.y), so indexing stays 32-bit.
template <class Conv>
__device__ __forceinline__ void stmt_body(const Base* __restrict__ bases, long long count, const StmtDesc& D,
                                          long long sadd, long long dadd, int ssz, int dsz, Conv conv) {
  const long long sstep = D.rank > 0 ? D.ss[D.rank - 1] : 0;
  const unsigned stride = gridDim.x * blockDim.x * static_cast<unsigned>(D.vec);
  for (unsigned e = (blockIdx.x * blockDim.x + threadIdx.x) * static_cast<unsigned>(D.vec); e < D.numel; e += stride)
  for (long long op = blockIdx.y; op < count; op += gridDim.y) {
    unsigned r = e;
    const Base B = bases[op];
    long long so = B.src + sadd, dof = B.dst + dadd;
    bool skip = false;
    for (int k = D.rank - 1; k >= 0; --k) {
      const unsigned q = (__umulhi(r, D.magic[k]) + r) >> D.shift[k], i = r - q * D.shape[k];
      r = q;
      long long is = i, id = i;
      if (k == D.stab_dim) {
        is = B.stab[static_cast<long long>(i) * D.stab_step];
        skip |= is < 0 || is >= D.sbound;
      }
      if (k == D.dtab_dim) {
        id = B.dtab[static_cast<long long>(i) * D.dtab_step];
        skip |= id < 0 || id >= D.dbound;
      }
      so += is * D.ss[k];
      dof += id * D.ds[k];
    }
    if (skip) {
      atomicOr(D.err, 1ull);
      continue;
    }
    if (D.vec == 1) {
      conv(reinterpret_cast<const void*>(static_cast<uintptr_t>(so) * ssz),
           reinterpret_cast<void*>(static_cast<uintptr_t>(dof) * dsz));
      continue;
    }
    alignas(16) unsigned char buf[16];
    for (int k = 0; k < D.vec; ++k)
      conv(reinterpret_cast<const void*>(static_cast<uintptr_t>(so + k * sstep) * ssz), buf + k * dsz);
    void* dp = reinterpret_cast<void*>(static_cast<uintptr_t>(dof) * dsz);
    switch (D.vec * dsz) {
      case 16: wr<uint4>(dp, *reinterpret_cast<const uint4*>(buf)); break;
      case 8: wr<uint2>(dp, *reinterpret_cast<const uint2*>(buf)); break;
      case 4: wr<uint32_t>(dp, *reinterpret_cast<const uint32_t*>(buf)); break;
      default: wr<uint16_t>(dp, *reinterpret_cast<const uint16_t*>(buf)); break;
    }
  }
}

// Writes words carried in the kernel's parameters to device memory: inside a
// CUDA graph capture the values become part of the kernel node, so no host
// buffer has to outlive the capture (pinned allocation is illegal there).
struct Put {
  unsigned long long* dst;
  int n;
  unsigned long long w[kPutWords];
};

extern "C" __global__ void ce_put(Put p) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.n; i += gridDim.x * blockDim.x) p.dst[i] = p.w[i];
}

extern "C" __global__ void ce_stmt(const Base* __restrict__ bases, long long count, StmtDesc D, Fn fn, long long sadd,
                                   long long dadd) {
  stmt_body(bases, count, D, sadd, dadd, dt_size(D.sdt), dt_size(D.ddt), DynConv{D, fn});
}

#define CE_STMT(name, Dt, St, mode)                                                                    \
  extern "C" __global__ void ce_stmt_##name(const Base* __restrict__ bases, long long count, StmtDesc D, Fn fn, \
                                            long long sadd, long long dadd) {                          \
    stmt_body(bases, count, D, sadd, dadd, sizeof(St), sizeof(Dt), TypedConv<Dt, St, mode>{fn});       \
  }
CE_STMT(b1, uint8_t, uint8_t, 0)
CE_STMT(b2, uint16_t, uint16_t, 0)
CE_STMT(b4, uint32_t, uint32_t, 0)
CE_STMT(b8, uint64_t, uint64_t, 0)
CE_STMT(cast_f32_f16, half_t, float, 1)
CE_STMT(cast_f32_bf16, bf16_t, float, 1)
CE_STMT(cast_f16_f32, float, half_t, 1)
CE_STMT(cast_bf16_f32, float, bf16_t, 1)
CE_STMT(cast_f16_bf16, bf16_t, half_t, 1)
CE_STMT(cast_bf16_f16, half_t, bf16_t, 1)
CE_STMT(cast_i8_f16, half_t, int8_t, 1)
CE_STMT(cast_i8_bf16, bf16_t, int8_t, 1)
CE_STMT(cast_i8_f32, float, int8_t, 1)
CE_STMT(cast_u8_f16, half_t, uint8_t, 1)
CE_STMT(cast_u8_bf16, bf16_t, uint8_t, 1)
CE_STMT(cast_u8_f32, float, uint8_t, 1)
CE_STMT(apply_i8_f16, half_t, int8_t, 2)
CE_STMT(apply_i8_bf16, bf16_t, int8_t, 2)
CE_STMT(apply_i8_f32, float, int8_t, 2)
CE_STMT(apply_f16_i8, int8_t, half_t, 2)
CE_STMT(apply_bf16_i8, int8_t, bf16_t, 2)
CE_STMT(apply_f32_i8, int8_t, float, 2)
CE_STMT(cast_f32_f8e4m3, f8e4m3_t, float, 1)
CE_STMT(cast_f8e4m3_f32, float, f8e4m3_t, 1)
CE_STMT(apply_f32_f8e4m3, f8e4m3_t, float, 2)
CE_STMT(apply_f8e4m3_f32, float, f8e4m3_t, 2)
CE_STMT(cast_f16_f8e4m3, f8e4m3_t, half_t, 1)
CE_STMT(cast_f8e4m3_f16, half_t, f8e4m3_t, 1)
CE_STMT(apply_f16_f8e4m3, f8e4m3_t, half_t, 2)
CE_STMT(apply_f8e4m3_f16, half_t, f8e4m3_t, 2)
CE_STMT(cast_bf16_f8e4m3, f8e4m3_t, bf16_t, 1)
CE_STMT(cast_f8e4m3_bf16, bf16_t, f8e4m3_t, 1)
CE_STMT(apply_bf16_f8e4m3, f8e4m3_t, bf16_t, 2)
CE_STMT(apply_f8e4m3_bf16, bf16_t, f8e4m3_t, 2)
CE_STMT(cast_f32_f8e5m2, f8e5m2_t, float, 1)
CE_STMT(cast_f8e5m2_f32, float, f8e5m2_t, 1)
CE_STMT(apply_f32_f8e5m2, f8e5m2_t, float, 2)
CE_STMT(apply_f8e5m2_f32, float, f8e5m2_t, 2)
CE_STMT(cast_f16_f8e5m2, f8e5m2_t, half_t, 1)
CE_STMT(cast_f8e5m2_f16, half_t, f8e5m2_t, 1)
CE_STMT(apply_f16_f8e5m2, f8e5m2_t, half_t, 2)
CE_STMT(apply_f8e5m2_f16, half_t, f8e5m2_t, 2)
CE_STMT(cast_bf16_f8e5m2, f8e5m2_t, bf16_t, 1)
CE_STMT(cast_f8e5m2_bf16, bf16_t, f8e5m2_t, 1)
CE_STMT(apply_bf16_f8e5m2, f8e5m2_t, bf16_t, 2)
CE_STMT(apply_f8e5m2_bf16, bf16_t, f8e5m2_t, 2)

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
  const long long tb = (sb + 31) / 32, tiles = tb * ((sa + 31) / 32);
  const int ssz = dt_size(D.sdt), dsz = dt_size(D.ddt);
  for (long long t = blockIdx.x; t < tiles; t += gridDim.x)  // grid-stride over tiles (the grid may be capped)
  for (long long z = blockIdx.z; z < count; z += gridDim.z) {
    const long long ia0 = (t / tb) * 32, ib0 = (t % tb) * 32;
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
// FP8 is compiled only against the types it is commonly converted with; the
// other pairs run ce_convert_dyn (same bits, no vector accesses).
CE_KERNEL(f32, float, f8e4m3, f8e4m3_t)
CE_KERNEL(f8e4m3, f8e4m3_t, f32, float)
CE_KERNEL(f16, half_t, f8e4m3, f8e4m3_t)
CE_KERNEL(f8e4m3, f8e4m3_t, f16, half_t)
CE_KERNEL(bf16, bf16_t, f8e4m3, f8e4m3_t)
CE_KERNEL(f8e4m3, f8e4m3_t, bf16, bf16_t)
CE_KERNEL(f32, float, f8e5m2, f8e5m2_t)
CE_KERNEL(f8e5m2, f8e5m2_t, f32, float)
CE_KERNEL(f16, half_t, f8e5m2, f8e5m2_t)
CE_KERNEL(f8e5m2, f8e5m2_t, f16, half_t)
CE_KERNEL(bf16, bf16_t, f8e5m2, f8e5m2_t)
CE_KERNEL(f8e5m2, f8e5m2_t, bf16, bf16_t)
CE_KERNEL(f32, float, f8e8m0, f8e8m0_t)
CE_KERNEL(f8e8m0, f8e8m0_t, f32, float)

// Any convertible pair, dtypes read at run time (sdt / ddt: the DType codes).
extern "C" __global__ void ce_convert_dyn(const Chunk* __restrict__ chunks, int n, Fn fn, long long sadd,
                                          long long dadd, int sdt, int ddt, int identity) {
  const int ss = dt_size(sdt), ds = dt_size(ddt);
  CE_FOR_CHUNKS(chunks, n, sadd, dadd) {
    const long long total = c.width * c.rows;
    for (long long e = lane; e < total; e += 32) {
      const long long r = e / c.width, i = e - r * c.width;
      const void* sp = reinterpret_cast<const void*>(static_cast<uintptr_t>(c.src + r * c.spitch + i) * ss);
      void* dp = reinterpret_cast<void*>(static_cast<uintptr_t>(c.dst + r * c.dpitch + i) * ds);
      if (identity) dyn_cast(sp, sdt, dp, ddt);
      else dyn_apply(sp, sdt, dp, ddt, fn);
    }
  }
}

// SM-limit admission: each block records the SM it ran on, after spinning a
// little so that an unconfined launch spreads over every SM.
extern "C" __global__ void ce_smid(unsigned* out, long long spin) {
  if (threadIdx.x != 0) return;
  const long long t0 = clock64();
  while (clock64() - t0 < spin) {
  }
  unsigned id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(id));
  out[blockIdx.x] = id;
}
