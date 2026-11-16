// Table-driven copy kernels, compiled AOT into a fatbin and loaded through the
// driver API. Tables stay resident in device memory; `sadd` / `dadd` are added
// to every source / destination index (the staging base, which can move).
// One warp per chunk. A chunk is a 2-D block in absolute element
// indices (a pointer is index * sizeof(T)): `rows` rows of `width` elements,
// row r at src + r * spitch / dst + r * dpitch. spitch may be 0 (broadcast) or
// smaller than width; rows never overlap on the destination.
//
// Build with --fmad=false: conversions must match the host backend bit for bit.

#include "backends/cuda/abi.hpp"
#include "core/convert.hpp"
#include "core/tile.hpp"

namespace {

using Chunk = ce::cuabi::Chunk;  // abi.hpp: the executor's layout

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
  if (aligned && c.rows > 1 && c.width < 32 * V && c.width % V == 0) {
    // Narrow rows (a head of 128 values, say): lanes span rows, V per access.
    const long long nv = c.width / V, n = nv * c.rows;
#pragma unroll 4
    for (long long e = lane; e < n; e += 32) {
      const long long r = e / nv, i = e - r * nv;
      Pack<S, V> in;
      in.word = reinterpret_cast<const Pack<S, V>*>(s + r * c.spitch)[i].word;
      Pack<D, V> out;
#pragma unroll
      for (int k = 0; k < V; ++k) out.v[k] = convert<D, S, kIdentity>(in.v[k], fn);
      reinterpret_cast<Pack<D, V>*>(d + r * c.dpitch)[i].word = out.word;
    }
  } else if (aligned && (c.width >= 32 * V || c.rows == 1)) {
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

// kStream: streaming (evict-first) loads and stores, ld.global.cs / st.global.cs,
// for background copies that should not evict the compute's data from L2.
template <class V, bool kStream = false>
__device__ __forceinline__ void move(uintptr_t s, uintptr_t d, long long nbytes, int lane) {
  const V* __restrict__ sv = reinterpret_cast<const V*>(s);
  V* __restrict__ dv = reinterpret_cast<V*>(d);
  const long long n = nbytes / static_cast<long long>(sizeof(V));
  for (long long i = lane; i < n; i += 32) {
    if constexpr (kStream) __stcs(dv + i, __ldcs(sv + i));
    else dv[i] = sv[i];
  }
}

}  // namespace

// Pure byte copies: widest vector the chunk's alignment allows.
template <bool kStream>
__device__ __forceinline__ void bytes_one(Chunk c, int lane) {
  const uintptr_t a = static_cast<uintptr_t>(c.src | c.dst | c.width | c.spitch | c.dpitch);
  for (long long r = 0; r < c.rows; ++r) {
    const uintptr_t s = static_cast<uintptr_t>(c.src + r * c.spitch), d = static_cast<uintptr_t>(c.dst + r * c.dpitch);
    if ((a & 15) == 0) move<uint4, kStream>(s, d, c.width, lane);
    else if ((a & 7) == 0) move<uint2, kStream>(s, d, c.width, lane);
    else if ((a & 3) == 0) move<unsigned int, kStream>(s, d, c.width, lane);
    else move<unsigned char, kStream>(s, d, c.width, lane);
  }
}

// ---- bulk copies through shared memory (sm80+: cp.async, sm90+: TMA) ------------------
//
// One warp per block per chunk; a chunk's rows are cut into segments of at
// most `seg` bytes (a launch parameter: the executor sizes it so that enough
// blocks fit on an SM), moved global -> shared -> global through kBufs
// buffers so later segments load while earlier ones store. On sm90+ one thread drives
// it with TMA bulk copies (cp.async.bulk + mbarrier): few threads keep many
// bytes in flight, which matters where copy kernels get few SMs (max_sms). On
// sm80-89 the warp uses cp.async. Chunks not 16-byte aligned (address, width,
// pitches) take the plain warp copy. Other architectures: the plain copy (the
// executor only offers this kernel where it was admitted).

constexpr int kBufs = ce::cuabi::kBulkBufs;  // segments in flight per block

struct Segs {  // the segments of one chunk
  const Chunk& c;
  long long seg, per_row;
  __device__ long long count() const { return c.rows * per_row; }
  __device__ void at(long long i, long long* s, long long* d, unsigned* n) const {
    const long long r = i / per_row, o = (i - r * per_row) * seg;
    *s = c.src + r * c.spitch + o, *d = c.dst + r * c.dpitch + o;
    *n = static_cast<unsigned>(min(seg, c.width - o));
  }
};

__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
__device__ __forceinline__ void mbar_init(unsigned long long* b) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_addr(b)) : "memory");
}
__device__ __forceinline__ void mbar_expect(unsigned long long* b, unsigned bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_addr(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(unsigned long long* b, unsigned phase) {
  unsigned done = 0;
  do {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
        : "=r"(done)
        : "r"(smem_addr(b)), "r"(phase)
        : "memory");
  } while (!done);
}
__device__ __forceinline__ void tma_load(void* sm, long long g, unsigned n, unsigned long long* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                   smem_addr(sm)),
               "l"(g), "r"(n), "r"(smem_addr(b))
               : "memory");
}
__device__ __forceinline__ void tma_store(long long g, const void* sm, unsigned n) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(g), "r"(smem_addr(sm)), "r"(n)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
__device__ __forceinline__ void async16(void* sm, long long g) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(sm)), "l"(g) : "memory");
}
#endif

extern "C" __global__ void __launch_bounds__(32) ce_bytes_bulk(const Chunk* __restrict__ chunks, int n,
                                                               long long sadd, long long dadd, int seg) {
  extern __shared__ __align__(128) unsigned char smem[];  // kBufs * seg bytes
  const auto buf = [&](long long i) { return smem + (i % kBufs) * seg; };
  const int lane = threadIdx.x & 31;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  __shared__ alignas(8) unsigned long long bar[kBufs];
  unsigned phase[kBufs] = {0, 0, 0, 0};
  if (lane == 0) {
    for (int b = 0; b < kBufs; ++b) mbar_init(&bar[b]);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncwarp();
#endif
  for (int w = blockIdx.x; w < n; w += gridDim.x) {
    Chunk c = chunks[w];
    c.src += sadd, c.dst += dadd;
    if ((c.src | c.dst | c.width | c.spitch | c.dpitch) & 15) {  // not 16-byte aligned: the plain warp copy
      bytes_one<false>(c, lane);
      continue;
    }
    const Segs sg{c, seg, (c.width + seg - 1) / seg};
    const long long count = sg.count();
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    // Loads run kBufs - 1 segments ahead. The buffer reloaded at step i is
    // the one stored at step i - 1: waiting until at most one store (step
    // i's) is still reading shared memory keeps a store in flight.
    if (lane == 0) {
      auto load = [&](long long i) {
        long long s, d;
        unsigned len;
        sg.at(i, &s, &d, &len);
        const int b = static_cast<int>(i % kBufs);
        mbar_expect(&bar[b], len);
        tma_load(buf(i), s, len, &bar[b]);
      };
      for (long long i = 0; i < count && i < kBufs - 1; ++i) load(i);
      for (long long i = 0; i < count; ++i) {
        const int b = static_cast<int>(i % kBufs);
        mbar_wait(&bar[b], phase[b]);
        phase[b] ^= 1u;
        long long s, d;
        unsigned len;
        sg.at(i, &s, &d, &len);
        tma_store(d, buf(i), len);
        if (i + kBufs - 1 < count) {
          asm volatile("cp.async.bulk.wait_group.read 1;" ::: "memory");  // step i - 1's store has read its buffer
          load(i + kBufs - 1);
        }
      }
      asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");  // buffers free for the next chunk
    }
    __syncwarp();
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    auto issue = [&](long long i) {
      long long s, d;
      unsigned len;
      sg.at(i, &s, &d, &len);
      unsigned char* b = buf(i);
      for (unsigned o = lane * 16u; o < len; o += 512u) async16(b + o, s + o);
    };
    for (long long i = 0; i < kBufs - 1; ++i) {  // kBufs - 1 segments ahead (empty groups past the end)
      if (i < count) issue(i);
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    for (long long i = 0; i < count; ++i) {
      if (i + kBufs - 1 < count) issue(i + kBufs - 1);
      asm volatile("cp.async.commit_group;" ::: "memory");
      asm volatile("cp.async.wait_group %0;" ::"n"(kBufs - 1) : "memory");  // segment i has landed
      __syncwarp();
      long long s, d;
      unsigned len;
      sg.at(i, &s, &d, &len);
      const unsigned char* b = buf(i);
      for (unsigned o = lane * 16u; o < len; o += 512u)
        *reinterpret_cast<uint4*>(static_cast<uintptr_t>(d + o)) = *reinterpret_cast<const uint4*>(b + o);
      __syncwarp();
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#else
    bytes_one<false>(c, lane);
#endif
  }
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if (lane == 0) asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");  // stores done before exit
#endif
  (void)buf;
  (void)smem;
}

// ---- 2-D TMA (sm90+): a block's rows as tensor maps -----------------------------------
//
// Each 2-D block of a pure copy (width bytes x height rows, a pitch per side)
// is a pair of uint8 tensor maps (encoded on the host, resident with the
// plan) moved box by box (TmaBlock::bw bytes x bh rows) through two shared
// buffers: a whole box per TMA operation, where the row-wise bulk copy needs
// one per row segment. Loads past the block fill zeros, stores past it are
// clipped by the hardware, so edge boxes need no special case.

using TmaBlock = ce::cuabi::TmaBlock;  // abi.hpp

extern "C" __global__ void __launch_bounds__(32) ce_bytes_tma2d(const TmaBlock* __restrict__ blocks, int n) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  constexpr int kTmaBox = ce::cuabi::kTmaBox;  // bytes per box (bw x bh), two per block
  __shared__ alignas(128) unsigned char buf[2][kTmaBox];
  __shared__ alignas(8) unsigned long long bar[2];
  if (threadIdx.x != 0) return;
  for (int b = 0; b < 2; ++b) mbar_init(&bar[b]);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  unsigned phase[2] = {0, 0};
  for (int k = blockIdx.x; k < n; k += gridDim.x) {
    const TmaBlock& B = blocks[k];
    const unsigned long long sm = reinterpret_cast<unsigned long long>(&B.src);
    const unsigned long long dm = reinterpret_cast<unsigned long long>(&B.dst);
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(sm) : "memory");
    asm volatile("fence.proxy.tensormap::generic.acquire.gpu [%0], 128;" ::"l"(dm) : "memory");
    const int nx = (B.width + B.bw - 1) / B.bw, ny = (B.height + B.bh - 1) / B.bh, tiles = nx * ny;
    const unsigned bytes = static_cast<unsigned>(B.bw * B.bh);
    auto load = [&](int i) {
      const int b = i & 1, x = (i % nx) * B.bw, y = (i / nx) * B.bh;
      mbar_expect(&bar[b], bytes);
      asm volatile(
          "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
          ::"r"(smem_addr(buf[b])), "l"(sm), "r"(x), "r"(y), "r"(smem_addr(&bar[b]))
          : "memory");
    };
    for (int i = 0; i < tiles && i < 2; ++i) load(i);
    for (int i = 0; i < tiles; ++i) {
      const int b = i & 1, x = (i % nx) * B.bw, y = (i / nx) * B.bh;
      mbar_wait(&bar[b], phase[b]);
      phase[b] ^= 1u;
      asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [%0, {%1, %2}], [%3];" ::"l"(dm),
                   "r"(x), "r"(y), "r"(smem_addr(buf[b]))
                   : "memory");
      asm volatile("cp.async.bulk.commit_group;" ::: "memory");
      if (i + 2 < tiles) {
        asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");  // buffer b read by its store
        load(i + 2);
      }
    }
    asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");  // buffers free for the next block
  }
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");  // stores done before exit
#else
  (void)blocks, (void)n;  // admitted on sm90+ only
#endif
}

extern "C" __global__ void ce_bytes(const Chunk* __restrict__ chunks, int n, long long sadd, long long dadd) {
  CE_FOR_CHUNKS(chunks, n, sadd, dadd) bytes_one<false>(c, lane);
}

// The same with streaming accesses: zero-copy over pinned host memory in
// bandwidth-limited (paced) copies.
extern "C" __global__ void ce_bytes_cs(const Chunk* __restrict__ chunks, int n, long long sadd, long long dadd) {
  CE_FOR_CHUNKS(chunks, n, sadd, dadd) bytes_one<true>(c, lane);
}

// Bandwidth pacing: one thread, launched before each chunk of a paced copy on
// its stream. `*next` (per GPU and direction) is the earliest time, on the
// GPU's global timer, at which the next chunk may start; this chunk reserves
// [start, start + ns) and the stream waits until start. The reservation is a
// CAS loop, so concurrent paced streams share the budget. Needs sm_70
// (__nanosleep); older GPUs only reserve (no wait).
extern "C" __global__ void ce_pace(unsigned long long* next, unsigned long long ns) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  unsigned long long now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  unsigned long long old = *reinterpret_cast<volatile unsigned long long*>(next), start;
  for (;;) {
    start = old > now ? old : now;
    const unsigned long long seen = atomicCAS(next, old, start + ns);
    if (seen == old) break;
    old = seen;
  }
#if __CUDA_ARCH__ >= 700
  while (now < start) {
    const unsigned long long left = start - now;
    __nanosleep(static_cast<unsigned>(left < 500000ull ? left : 500000ull));
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  }
#endif
}

// Structured statements: dst[D(i)] = f(src[S(i)]) over a shared domain, for a
// batch of statements with the same shape and strides but different bases.
// Strides may be 0 (broadcast) or negative. Dtypes are dispatched at runtime
// (same bits as the typed kernels, see convert.hpp). Each thread takes `vec`
// consecutive elements of the innermost dimension (the host checks it divides
// that dimension, which is contiguous and vec-aligned on the destination) and
// stores them as one access.
using StmtDesc = ce::cuabi::StmtDesc;  // abi.hpp

using Base = ce::cuabi::Base;  // abi.hpp

constexpr int kPutWords = ce::cuabi::kPutWords;  // ce_put's parameters stay under the 4 KB limit

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
        is = reinterpret_cast<const long long*>(B.stab)[static_cast<long long>(i) * D.stab_step];
        skip |= is < 0 || is >= D.sbound;
      }
      if (k == D.dtab_dim) {
        id = reinterpret_cast<const long long*>(B.dtab)[static_cast<long long>(i) * D.dtab_step];
        skip |= id < 0 || id >= D.dbound;
      }
      so += is * D.ss[k];
      dof += id * D.ds[k];
    }
    if (skip) {
      atomicOr(reinterpret_cast<unsigned long long*>(D.err), 1ull);
      continue;
    }
    if (D.vec == 1) {
      conv(reinterpret_cast<const void*>(static_cast<uintptr_t>(so) * ssz),
           reinterpret_cast<void*>(static_cast<uintptr_t>(dof) * dsz));
      continue;
    }
    void* dp0 = reinterpret_cast<void*>(static_cast<uintptr_t>(dof) * dsz);
    const uintptr_t sp0 = static_cast<uintptr_t>(so) * ssz;
    if (D.mode == 0 && sstep == 1 && ssz == dsz && (sp0 & static_cast<uintptr_t>(D.vec * dsz - 1)) == 0) {
      const void* sp = reinterpret_cast<const void*>(sp0);  // raw copy, aligned on both sides: one load
      switch (D.vec * dsz) {  // aligned: plain typed accesses (memcpy-based rd / wr may go bytewise)
        case 16: *static_cast<uint4*>(dp0) = *static_cast<const uint4*>(sp); break;
        case 8: *static_cast<uint2*>(dp0) = *static_cast<const uint2*>(sp); break;
        case 4: *static_cast<uint32_t*>(dp0) = *static_cast<const uint32_t*>(sp); break;
        default: *static_cast<uint16_t*>(dp0) = *static_cast<const uint16_t*>(sp); break;
      }
      continue;
    }
    alignas(16) unsigned char buf[16];
    for (int k = 0; k < D.vec; ++k)
      conv(reinterpret_cast<const void*>(static_cast<uintptr_t>(so + k * sstep) * ssz), buf + k * dsz);
    switch (D.vec * dsz) {  // the host checked the destination's alignment
      case 16: *static_cast<uint4*>(dp0) = *reinterpret_cast<const uint4*>(buf); break;
      case 8: *static_cast<uint2*>(dp0) = *reinterpret_cast<const uint2*>(buf); break;
      case 4: *static_cast<uint32_t*>(dp0) = *reinterpret_cast<const uint32_t*>(buf); break;
      default: *static_cast<uint16_t*>(dp0) = *reinterpret_cast<const uint16_t*>(buf); break;
    }
  }
}

// Writes words carried in the kernel's parameters to device memory: inside a
// CUDA graph capture the values become part of the kernel node, so no host
// buffer has to outlive the capture (pinned allocation is illegal there).
using Put = ce::cuabi::Put;  // abi.hpp

// The layout fingerprint of the structures above as this compilation sees them
// (abi.hpp): the executor admits the kernels only if it equals its own.
extern "C" __global__ void ce_layout(unsigned long long* out) { *out = ce::cuabi::layout_fingerprint(); }

extern "C" __global__ void ce_put(Put p) {
  auto* dst = reinterpret_cast<unsigned long long*>(p.dst);
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < p.n; i += gridDim.x * blockDim.x) dst[i] = p.w[i];
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


namespace cv = ce::cv;

// ---- tile ops: quantization while copying (docs/design/v0.4.md) ----------------------

namespace {

__device__ __forceinline__ float warp_amax(float v) {
  for (int o = 16; o; o >>= 1) v = cv::amax_step(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}
__device__ __forceinline__ float warp_amin(float v) {
  for (int o = 16; o; o >>= 1) v = cv::amin_step(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
  return v;
}

// One unit on one warp: the same unit functions as the host path (the
// reductions are exact and NaN-propagating, so their order does not matter).
// The format and direction are compile-time constants: one kernel each keeps
// registers and branches down.
template <int F, bool Q>
__device__ __forceinline__ void tile_unit_warp(const cv::TileK& t, long long u, int lane) {
  using cv::QFmt;
  constexpr QFmt f = static_cast<QFmt>(F);
  const cv::UnitAt a = cv::unit_at(t, u);
  if (a.bad) {  // an index out of range: skipped, reported at synchronize()
    if (lane == 0 && t.err) atomicOr(reinterpret_cast<unsigned long long*>(t.err), 1ull);
    return;
  }
  auto xv = [&](int k) { return cv::val_float(cv::load(cv::addr(cv::x_at(t, a, k)), t.xdt)); };
  if (Q && t.static_scale) {  // the given scale (and zero point): read, not computed
    const float s = t.scale_byte ? 0.0f : cv::rd<float>(cv::addr(a.s));
    const uint8_t sb = t.scale_byte ? cv::rd<uint8_t>(cv::addr(a.s)) : 0;
    const uint8_t z = f == QFmt::uint4 ? cv::rd<uint8_t>(cv::addr(a.z)) : 0;
    const cv::UnitQ p = cv::unit_from_scale(f, s, sb, z, t.gs);
    if constexpr (cv::qfmt_packed(f)) {
      for (int i = lane; i < t.g / 2; i += 32)
        cv::wr<uint8_t>(cv::addr(cv::q_at(t, a, 2 * i)),
                        static_cast<uint8_t>(cv::quant_elem(f, xv(2 * i), p) | (cv::quant_elem(f, xv(2 * i + 1), p) << 4)));
    } else {
      for (int k = lane; k < t.g; k += 32) cv::wr<uint8_t>(cv::addr(cv::q_at(t, a, k)), cv::quant_elem(f, xv(k), p));
    }
    return;
  }
  if constexpr (Q) {
    constexpr bool range = f == QFmt::uint4;
    constexpr int kCache = 8;  // values per lane kept in registers (units up to 256)
    float c[kCache];
    float amax = 0.0f, lo = 0.0f, hi = 0.0f;
    int n = 0;
    for (int k = lane; k < t.g; k += 32, ++n) {
      const float v = xv(k);
      if (n < kCache) c[n] = v;
      amax = cv::amax_step(amax, cv::abs_f(v));
      if (range) lo = cv::amin_step(lo, v), hi = cv::amax_step(hi, v);
    }
    amax = warp_amax(amax);
    if (range) lo = warp_amin(lo), hi = warp_amax(hi);
    const cv::UnitQ p = cv::unit_params(f, amax, lo, hi, t.eps, t.gs);
    if (lane == 0) {
      if (t.scale_byte) cv::wr<uint8_t>(cv::addr(a.s), p.bits);
      else cv::wr<float>(cv::addr(a.s), p.scale);
      if (f == QFmt::uint4) cv::wr<uint8_t>(cv::addr(a.z), p.bits);
    }
    if constexpr (cv::qfmt_packed(f)) {
      for (int i = lane; i < t.g / 2; i += 32)
        cv::wr<uint8_t>(cv::addr(cv::q_at(t, a, 2 * i)),
                        static_cast<uint8_t>(cv::quant_elem(f, xv(2 * i), p) | (cv::quant_elem(f, xv(2 * i + 1), p) << 4)));
    } else {
      n = 0;
      for (int k = lane; k < t.g; k += 32, ++n)
        cv::wr<uint8_t>(cv::addr(cv::q_at(t, a, k)), cv::quant_elem(f, n < kCache ? c[n] : xv(k), p));
    }
  } else {
    const float s = t.scale_byte ? 0.0f : cv::rd<float>(cv::addr(a.s));
    const uint8_t bits = t.scale_byte ? cv::rd<uint8_t>(cv::addr(a.s)) : (f == QFmt::uint4 ? cv::rd<uint8_t>(cv::addr(a.z)) : 0);
    const float factor = cv::unit_scale(f, s, bits, t.gs);
    const uint8_t* q0 = static_cast<const uint8_t*>(cv::addr(cv::q_at(t, a, 0)));
    for (int k = lane; k < t.g; k += 32)
      cv::store_float(cv::addr(cv::x_at(t, a, k)), t.xdt, cv::dequantize_at(q0, t.qe, k, f, factor, bits));
  }
}

// ---- the vector path: 4 consecutive values per thread, several units per warp ----------

__device__ __forceinline__ void load4(long long p, int dt, float v[4]) {
  if (dt == cv::kF32) {
    const float4 w = *reinterpret_cast<const float4*>(static_cast<uintptr_t>(p));
    v[0] = w.x, v[1] = w.y, v[2] = w.z, v[3] = w.w;
    return;
  }
  const uint2 w = *reinterpret_cast<const uint2*>(static_cast<uintptr_t>(p));
  const uint16_t h[4] = {static_cast<uint16_t>(w.x), static_cast<uint16_t>(w.x >> 16), static_cast<uint16_t>(w.y),
                         static_cast<uint16_t>(w.y >> 16)};
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = dt == cv::kF16 ? cv::half_to_float(h[i]) : cv::bf16_to_float(h[i]);
}

__device__ __forceinline__ void store4(long long p, int dt, const float v[4]) {
  if (dt == cv::kF32) {
    *reinterpret_cast<float4*>(static_cast<uintptr_t>(p)) = make_float4(v[0], v[1], v[2], v[3]);
    return;
  }
  uint16_t h[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = dt == cv::kF16 ? cv::float_to_half(v[i]) : cv::float_to_bf16(v[i]);
  *reinterpret_cast<uint2*>(static_cast<uintptr_t>(p)) =
      make_uint2(h[0] | (static_cast<uint32_t>(h[1]) << 16), h[2] | (static_cast<uint32_t>(h[3]) << 16));
}

template <int W>
__device__ __forceinline__ void loadw(long long p, int dt, float* v) {
  const long long step = 4LL * (dt == cv::kF32 ? 4 : 2);
#pragma unroll
  for (int h = 0; h < W / 4; ++h) load4(p + h * step, dt, v + 4 * h);
}
template <int W>
__device__ __forceinline__ void storew(long long p, int dt, const float* v) {
  const long long step = 4LL * (dt == cv::kF32 ? 4 : 2);
#pragma unroll
  for (int h = 0; h < W / 4; ++h) store4(p + h * step, dt, v + 4 * h);
}

template <int F, bool Q, int W>
__device__ void tile_vec(const cv::TileK& t, int lane, long long warp0, long long warps, int tpu) {
  using cv::QFmt;
  constexpr QFmt f = static_cast<QFmt>(F);
  constexpr bool range = f == QFmt::uint4;
  constexpr bool packed = cv::qfmt_packed(f);
  // Values per thread kept in registers (unrolled: no local memory). More
  // would cost occupancy, which is what keeps loads in flight where few SMs
  // run these kernels.
  constexpr int kC = 8 / W;
  const int upw = 32 / tpu, sub = lane & (tpu - 1), cpt = (t.g / W) / tpu, stride = W * tpu;
  const long long units = t.units();
  const float igs = cv::div_rn(1.0f, t.gs);  // once per thread, not per unit
  // 4 codes per access: a 32-bit word (bytes) or 16 bits (nibbles).
  auto codes = [&](const cv::UnitAt& a, int base, uint32_t* c4) {
#pragma unroll
    for (int h = 0; h < W / 4; ++h) {
      const uintptr_t qp = static_cast<uintptr_t>(cv::q_at(t, a, base + 4 * h));
      c4[h] = packed ? *reinterpret_cast<const uint16_t*>(qp) : *reinterpret_cast<const uint32_t*>(qp);
    }
  };
  for (long long w = warp0; w * upw < units; w += warps) {
    const long long u = w * upw + lane / tpu;
    cv::UnitAt a{};
    a.bad = true;
    if (u < units) {
      a = cv::unit_at(t, u);
      if (a.bad && sub == 0 && t.err) atomicOr(reinterpret_cast<unsigned long long*>(t.err), 1ull);
    }
    const bool ok = u < units && !a.bad;  // the same for a unit's threads: shuffles stay within its group
    if (!Q) {  // dequantize
      if (!ok) continue;
      const float s = t.scale_byte ? 0.0f : cv::rd<float>(cv::addr(a.s));
      const uint8_t bits =
          t.scale_byte ? cv::rd<uint8_t>(cv::addr(a.s)) : (f == QFmt::uint4 ? cv::rd<uint8_t>(cv::addr(a.z)) : 0);
      const float factor = cv::unit_scale(f, s, bits, t.gs);
      for (int j = 0; j < cpt; ++j) {
        const int base = sub * W + j * stride;
        uint32_t c4[W / 4];
        codes(a, base, c4);
        float v[W];
#pragma unroll
        for (int i = 0; i < W; ++i) {
          const uint32_t c = c4[i / 4];
          v[i] = cv::dequantize_code(
              f, static_cast<uint8_t>(packed ? (c >> (4 * (i % 4))) & 0xFu : (c >> (8 * (i % 4))) & 0xFFu), factor,
              bits);
        }
        storew<W>(cv::x_at(t, a, base), t.xdt, v);
      }
      continue;
    }
    float c[kC][W];
    float amax = 0.0f, lo = 0.0f, hi = 0.0f;
    auto take = [&](const float* v) {
#pragma unroll
      for (int i = 0; i < W; ++i) {
        amax = cv::amax_step(amax, cv::abs_f(v[i]));
        if (range) lo = cv::amin_step(lo, v[i]), hi = cv::amax_step(hi, v[i]);
      }
    };
#pragma unroll
    for (int j = 0; j < kC; ++j) {
      if (j >= cpt) break;
#pragma unroll
      for (int i = 0; i < W; ++i) c[j][i] = 0.0f;
      if (ok) loadw<W>(cv::x_at(t, a, sub * W + j * stride), t.xdt, c[j]);
      take(c[j]);
    }
    for (int j = kC; j < cpt; ++j) {  // long units: read again below
      float v[W];
#pragma unroll
      for (int i = 0; i < W; ++i) v[i] = 0.0f;
      if (ok) loadw<W>(cv::x_at(t, a, sub * W + j * stride), t.xdt, v);
      take(v);
    }
    for (int o = tpu >> 1; o; o >>= 1) {
      amax = cv::amax_step(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
      if (range) {
        lo = cv::amin_step(lo, __shfl_xor_sync(0xFFFFFFFFu, lo, o));
        hi = cv::amax_step(hi, __shfl_xor_sync(0xFFFFFFFFu, hi, o));
      }
    }
    if (!ok) continue;
    cv::UnitQ p;
    if (t.static_scale) {
      const float s = t.scale_byte ? 0.0f : cv::rd<float>(cv::addr(a.s));
      const uint8_t sb = t.scale_byte ? cv::rd<uint8_t>(cv::addr(a.s)) : 0;
      const uint8_t z = f == QFmt::uint4 ? cv::rd<uint8_t>(cv::addr(a.z)) : 0;
      p = cv::unit_from_scale(f, s, sb, z, t.gs);
    } else {
      p = cv::unit_params(f, amax, lo, hi, t.eps, t.gs, igs);
      if (sub == 0) {
        if (t.scale_byte) cv::wr<uint8_t>(cv::addr(a.s), p.bits);
        else cv::wr<float>(cv::addr(a.s), p.scale);
        if (f == QFmt::uint4) cv::wr<uint8_t>(cv::addr(a.z), p.bits);
      }
    }
    auto put = [&](int base, const float* v) {
#pragma unroll
      for (int h = 0; h < W / 4; ++h) {
        uint8_t k[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) k[i] = cv::quant_elem(f, v[4 * h + i], p);
        const uintptr_t qp = static_cast<uintptr_t>(cv::q_at(t, a, base + 4 * h));
        if constexpr (packed) {
          *reinterpret_cast<uint16_t*>(qp) = static_cast<uint16_t>((k[0] | (k[1] << 4)) | ((k[2] | (k[3] << 4)) << 8));
        } else {
          *reinterpret_cast<uint32_t*>(qp) = k[0] | (static_cast<uint32_t>(k[1]) << 8) |
                                             (static_cast<uint32_t>(k[2]) << 16) | (static_cast<uint32_t>(k[3]) << 24);
        }
      }
    };
#pragma unroll
    for (int j = 0; j < kC; ++j) {
      if (j >= cpt) break;
      put(sub * W + j * stride, c[j]);
    }
    for (int j = kC; j < cpt; ++j) {
      float v[W];
      loadw<W>(cv::x_at(t, a, sub * W + j * stride), t.xdt, v);
      put(sub * W + j * stride, v);
    }
  }
}

// Box units (byte formats, float32 scales; block FP8 / INT8): one warp per
// box, a pass for the amax and one for the codes (or one to dequantize), 4
// consecutive values per access where the layout allows (TileK::vec).
template <int F, bool Q>
__device__ void tile_box(const cv::TileK& t, int lane, long long warp0, long long warps) {
  using cv::QFmt;
  constexpr QFmt f = static_cast<QFmt>(F);
  const long long units = t.units();
  const float igs = cv::div_rn(1.0f, t.gs);
  const bool v4 = t.vec != 0;
  for (long long u = warp0; u < units; u += warps) {
    const cv::UnitAt a = cv::unit_at(t, u);
    if (a.bad) {
      if (lane == 0 && t.err) atomicOr(reinterpret_cast<unsigned long long*>(t.err), 1ull);
      continue;
    }
    if (!Q) {
      const float factor = cv::unit_scale(f, cv::rd<float>(cv::addr(a.s)), 0, t.gs);
      for (int r = 0; r < t.bh; ++r) {
        if (v4) {
          for (int c = lane * 4; c < t.g; c += 128) {
            const uint32_t c4 = *reinterpret_cast<const uint32_t*>(static_cast<uintptr_t>(cv::qb_at(t, a, r, c)));
            float v[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) v[i] = cv::dequantize_code(f, static_cast<uint8_t>(c4 >> (8 * i)), factor);
            store4(cv::xb_at(t, a, r, c), t.xdt, v);
          }
        } else {
          for (int c = lane; c < t.g; c += 32)
            cv::store_float(cv::addr(cv::xb_at(t, a, r, c)), t.xdt,
                            cv::dequantize_code(f, cv::rd<uint8_t>(cv::addr(cv::qb_at(t, a, r, c))), factor));
        }
      }
      continue;
    }
    cv::UnitQ p;
    if (t.static_scale) {
      p = cv::unit_from_scale(f, cv::rd<float>(cv::addr(a.s)), 0, 0, t.gs);
    } else {
      float amax = 0.0f;
      for (int r = 0; r < t.bh; ++r) {
        if (v4) {
          for (int c = lane * 4; c < t.g; c += 128) {
            float v[4];
            load4(cv::xb_at(t, a, r, c), t.xdt, v);
#pragma unroll
            for (int i = 0; i < 4; ++i) amax = cv::amax_step(amax, cv::abs_f(v[i]));
          }
        } else {
          for (int c = lane; c < t.g; c += 32)
            amax = cv::amax_step(amax, cv::abs_f(cv::val_float(cv::load(cv::addr(cv::xb_at(t, a, r, c)), t.xdt))));
        }
      }
      amax = warp_amax(amax);
      p = cv::unit_params(f, amax, 0.0f, 0.0f, t.eps, t.gs, igs);
      if (lane == 0) cv::wr<float>(cv::addr(a.s), p.scale);
    }
    for (int r = 0; r < t.bh; ++r) {
      if (v4) {
        for (int c = lane * 4; c < t.g; c += 128) {
          float v[4];
          load4(cv::xb_at(t, a, r, c), t.xdt, v);
          uint32_t w = 0;
#pragma unroll
          for (int i = 0; i < 4; ++i) w |= static_cast<uint32_t>(cv::quant_elem(f, v[i], p)) << (8 * i);
          *reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(cv::qb_at(t, a, r, c))) = w;
        }
      } else {
        for (int c = lane; c < t.g; c += 32)
          cv::wr<uint8_t>(cv::addr(cv::qb_at(t, a, r, c)),
                          cv::quant_elem(f, cv::val_float(cv::load(cv::addr(cv::xb_at(t, a, r, c)), t.xdt)), p));
      }
    }
  }
}

}  // namespace

// Grid: x over warps (8 per block, grid-stride over units), y over the group's ops.
template <int F, bool Q>
__device__ __forceinline__ void tile_kernel(const cv::TileK* __restrict__ tiles, int count) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, wpb = blockDim.x >> 5;
  const long long warp0 = static_cast<long long>(blockIdx.x) * wpb + warp;
  const long long warps = static_cast<long long>(gridDim.x) * wpb;
  // The descriptor stays in shared memory (read where used): a copy in
  // registers would take some 60 of them and push the rest into local memory.
  __shared__ cv::TileK sh;
  static_assert(sizeof(cv::TileK) % 8 == 0 && sizeof(cv::TileK) / 8 <= 256);
  for (int op = blockIdx.y; op < count; op += gridDim.y) {
    __syncthreads();  // the previous op's readers are done
    if (threadIdx.x < sizeof(cv::TileK) / 8)
      reinterpret_cast<unsigned long long*>(&sh)[threadIdx.x] =
          reinterpret_cast<const unsigned long long*>(&tiles[op])[threadIdx.x];
    __syncthreads();
    const cv::TileK& t = sh;
    if constexpr (F <= static_cast<int>(cv::QFmt::int8)) {  // box units: byte formats only
      if (t.bh > 1) {
        tile_box<F, Q>(t, lane, warp0, warps);
        continue;
      }
    }
    if (const cv::VecShape v = cv::tile_shape(t); v.tpu) {
      // NVFP4's units are 16 values: its kernels hold only the 4-wide path.
      if constexpr (F != static_cast<int>(cv::QFmt::nvfp4)) {
        if (v.w == 8) {
          tile_vec<F, Q, 8>(t, lane, warp0, warps, v.tpu);
          continue;
        }
      }
      tile_vec<F, Q, 4>(t, lane, warp0, warps, v.tpu);
      continue;
    }
    const long long units = t.units();
    for (long long u = warp0; u < units; u += warps) tile_unit_warp<F, Q>(t, u, lane);
  }
}

// Byte formats: at least 3 blocks of 256 per SM (occupancy keeps loads in
// flight when few SMs run them); the packed formats need their registers.
#define CE_TILE_K(F, BOUNDS)                                                                             \
  extern "C" __global__ void BOUNDS ce_tile_q##F(const cv::TileK* __restrict__ t, int n) { tile_kernel<F, true>(t, n); } \
  extern "C" __global__ void BOUNDS ce_tile_d##F(const cv::TileK* __restrict__ t, int n) { tile_kernel<F, false>(t, n); }
#define CE_TILE(F) CE_TILE_K(F, __launch_bounds__(256, 3))
#define CE_TILE_PACKED(F) CE_TILE_K(F, )
CE_TILE(0)
CE_TILE(1)
CE_TILE(2)
CE_TILE(3)
CE_TILE(4)
CE_TILE_PACKED(5)
CE_TILE_PACKED(6)
CE_TILE_PACKED(7)
