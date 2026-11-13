#include "backends/cuda/executor.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/cuda/driver.hpp"
#include "backends/host/host.hpp"
#include "copyengine/error.hpp"

#if CE_CUDA_KERNELS
extern "C" const unsigned char ce_kernels_fatbin[];
#endif

namespace ce {
namespace {

using cuda::Driver;

// Mirrors the kernel's chunk: rows x width elements at constant pitches.
struct Chunk {
  long long src, dst, width, rows, spitch, dpitch;
};

// Packs (src, dst, len) parts, in order, into kernel chunks: consecutive parts
// of equal length at constant pitches (any src pitch, including 0) share a
// chunk; nothing exceeds `max` elements.
class ChunkBuilder {
 public:
  explicit ChunkBuilder(long long max) : max_(max) {}
  void add(long long s, long long d, long long len) {
    if (len <= 0) return;
    if (len > max_) {
      close();
      for (long long k = 0; k < len; k += max_) out_.push_back({s + k, d + k, std::min(max_, len - k), 1, 0, 0});
      return;
    }
    if (open_) {
      Chunk& c = out_.back();
      if (len == c.width && (c.rows + 1) * c.width <= max_) {
        // Destination parts never overlap (checked at enqueue), so any
        // constant destination pitch keeps rows disjoint.
        if (c.rows == 1) {
          c.spitch = s - c.src, c.dpitch = d - c.dst, c.rows = 2;
          return;
        }
        const long long ls = c.src + (c.rows - 1) * c.spitch, ld = c.dst + (c.rows - 1) * c.dpitch;
        if (s - ls == c.spitch && d - ld == c.dpitch) {
          ++c.rows;
          return;
        }
      }
    }
    out_.push_back({s, d, len, 1, 0, 0});
    open_ = true;
  }
  void close() { open_ = false; }
  std::vector<Chunk> take() { return std::move(out_); }

 private:
  long long max_;
  bool open_ = false;
  std::vector<Chunk> out_;
};

// A raw byte copy between two addresses on known sides.
struct Raw {
  uintptr_t src, dst;
  size_t bytes;
};

// Mirror the kernel's structured-statement parameters (kernels.cu).
struct StmtDesc {
  int rank, sdt, ddt, mode;
  int vec;  // destination elements per thread (innermost dim)
  unsigned numel;
  unsigned shape[6];
  unsigned magic[6], shift[6];  // division by shape[k] as multiply + shift (see kernels.cu)
  long long ss[6], ds[6];
};

// Magic number and shift such that n / d == (umulhi(n, magic) + n) >> shift
// for every n < 2^31 (the divider PyTorch's IntDivider uses).
void int_divider(unsigned d, unsigned* magic, unsigned* shift) {
  unsigned s = 0;
  while (s < 32 && (uint64_t{1} << s) < d) ++s;
  *shift = s;
  *magic = static_cast<unsigned>(((uint64_t{1} << 32) * ((uint64_t{1} << s) - d)) / d + 1);
}

struct Base {
  long long src, dst;
};

struct FusedBatch {
  StmtDesc desc{};
  int tile_a = -1, tile_b = -1;  // destination / source contiguous dims when they differ (tiled kernel)
  i64 link_size = 1;  // itemsize of the side that lives in staging (cross-device only)
  std::vector<Base> bases;
  std::shared_ptr<void> table;  // the bases, resident on the device (set on first run)
};

// Structured statements grouped by structure; for cross-device groups the
// link side is addressed relative to the staging base.
struct FusedPlan {
  bool ok = false;
  bool cross = false, h2d = false;
  size_t stg_bytes = 0;
  std::vector<Raw> raws;
  std::vector<FusedBatch> batches;
};

constexpr size_t kMaxFusedBatches = 64;

constexpr int kSlots = 16;
constexpr long long kTypedChunk = 4096;      // elements per warp
constexpr long long kByteChunk = 16 << 10;   // bytes per warp
constexpr size_t kBatchMax = 1 << 16;        // copies per cuMemcpyBatchAsync call
constexpr int64_t kSmallRect = 4 << 10;      // rows below this are "fragmented"
constexpr int64_t kGapBytes = 4 << 10;       // read gaps below this are copied, not skipped

void check(CUresult r, const char* what) {
  if (r != CUDA_SUCCESS) fail(Errc::backend, std::string(what) + ": " + cuda::error_name(r));
}

std::string short_name(DType t) {
  switch (t) {
    case DType::f64: return "f64";
    case DType::f32: return "f32";
    case DType::f16: return "f16";
    case DType::bf16: return "bf16";
    case DType::i64: return "i64";
    case DType::i32: return "i32";
    case DType::i16: return "i16";
    case DType::i8: return "i8";
    case DType::u8: return "u8";
    case DType::b8: return "b8";
  }
  return "?";
}

class CudaExecutor final : public Executor {
 public:
  explicit CudaExecutor(const EngineOptions& opt) : opt_(opt) {
    std::string err;
    d_ = cuda::driver(&err);
    if (!d_) fail(Errc::unsupported, err);
    int count = 0;
    check(d_->cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count == 0) fail(Errc::unsupported, "no CUDA devices");
    devs_.resize(count);
  }

  ~CudaExecutor() override {
    alive_.reset();  // resident tables outliving the executor must not touch the driver
    for (Mark& m : marks_) {
      if (!m.ev) continue;
      d_->cuCtxPushCurrent(devs_[m.dev].ctx);
      d_->cuEventDestroy(m.ev);
      CUcontext dummy{};
      d_->cuCtxPopCurrent(&dummy);
    }
    for (size_t i = 0; i < devs_.size(); ++i) {
      Dev& s = devs_[i];
      if (!s.init) continue;
      d_->cuCtxPushCurrent(s.ctx);
      for (Slot& sl : s.slots) {
        if (sl.used) d_->cuEventSynchronize(sl.done);
        if (sl.host) d_->cuMemFreeHost(sl.host);
        if (sl.dev) d_->cuMemFree(sl.dev);
        if (sl.done) d_->cuEventDestroy(sl.done);
      }
      for (Bounce& b : s.bounce) {
        if (b.used) d_->cuEventSynchronize(b.done);
        if (b.host) d_->cuMemFreeHost(b.host);
        if (b.done) d_->cuEventDestroy(b.done);
      }
      if (s.staging_used) d_->cuEventSynchronize(s.staging_done);
      if (s.staging) d_->cuMemFree(s.staging);
      for (CUevent e : {s.ev_in, s.ev_out, s.staging_done})
        if (e) d_->cuEventDestroy(e);
      if (s.side) d_->cuStreamDestroy(s.side);
      CUcontext dummy{};
      d_->cuCtxPopCurrent(&dummy);
      d_->cuDevicePrimaryCtxRelease(s.dev);
    }
  }

  std::vector<Strategy> candidates(const Group& g) override {
    const int di = device_of(g);
    Dev& s = state(di);
    const bool src_cuda = g.src.device.kind == DeviceKind::cuda, dst_cuda = g.dst.device.kind == DeviceKind::cuda;
    const bool same_dev = src_cuda && dst_cuda && g.src.device.index == g.dst.device.index;
    const size_t rects = g.rects.size();
    std::vector<Strategy> c;
    const bool use_fused = s.kernels && fused_worthwhile(g);
    auto with_fused = [&](std::vector<Strategy> v) {
      if (use_fused) v.insert(v.begin(), Strategy::fused);
      return v;
    };
    if (g.pure()) {
      if (same_dev) {
        if (s.kernels) c.push_back(Strategy::kernel);
        if (s.batch) c.push_back(Strategy::batch);
        c.push_back(Strategy::runs);
        if (rects <= 2) std::stable_partition(c.begin(), c.end(), [](Strategy x) { return x == Strategy::runs; });
      } else if (src_cuda && dst_cuda) {
        if (s.batch) c.push_back(Strategy::batch);
        c.push_back(Strategy::runs);
        if (rects <= 2) std::reverse(c.begin(), c.end());
      } else {
        const bool host_ok = (src_cuda || g.src.pinned) && (dst_cuda || g.dst.pinned);
        int64_t rows = 0;
        for (i64 h : g.rects.height) rows += h;
        const bool fragmented = rows > 2 && g.bytes() / rows < kSmallRect;  // small rows DMA poorly
        if (fragmented && s.kernels) c.push_back(Strategy::staged);  // dense link copy + device-side kernel
        if (s.batch && host_ok) c.push_back(Strategy::batch);
        if (!host_ok) c.push_back(Strategy::bounce);
        if (!fragmented && s.kernels && rects > 2) c.push_back(Strategy::staged);
        c.push_back(Strategy::runs);
        if (rects <= 2) std::stable_partition(c.begin(), c.end(), [](Strategy x) { return x == Strategy::runs; });
      }
      return with_fused(std::move(c));
    }
    if (!s.kernels) fail(Errc::unsupported, "dtype conversion on CUDA needs the kernel module: " + s.kernel_error);
    if (src_cuda && dst_cuda) {
      if (!same_dev) fail(Errc::unsupported, "converting copies between two GPUs are not supported yet");
      return with_fused({Strategy::kernel});
    }
    return with_fused({Strategy::staged});
  }

  int64_t run(Strategy strat, const Group& g, const std::vector<Op>& ops, uintptr_t stream) override {
    const int di = device_of(g);
    Dev& s = state(di);
    Push push(*d_, s.ctx);
    const CUstream st = reinterpret_cast<CUstream>(stream);
    switch (strat) {
      case Strategy::passthrough: return passthrough(s, g, ops, st);
      case Strategy::runs: return runs(g, st);
      case Strategy::batch: return batch(s, g.src, g.dst, cached_rows(g, Strategy::batch), st);
      case Strategy::kernel: return kernel(s, g, st);
      case Strategy::staged: return staged(s, g, st);
      case Strategy::bounce: return bounce(s, g.src, g.dst, cached_rows(g, Strategy::bounce), st);
      case Strategy::fused: return fused(s, g, st);
      default: fail(Errc::unsupported, "CUDA backend cannot run strategy " + std::string(strategy_name(strat)));
    }
  }

  // A stream handle names its own context; stream 0 is each device's legacy
  // stream, so it is synchronized on every device in use.
  void sync(std::span<const uintptr_t> streams) override {
    bool first = true;
    for (Dev& s : devs_) {
      if (!s.init) continue;
      Push push(*d_, s.ctx);
      for (uintptr_t st : streams)
        if (st == 0 || first) check(d_->cuStreamSynchronize(reinterpret_cast<CUstream>(st)), "cuStreamSynchronize");
      first = false;
    }
  }

  bool async() const override { return true; }

  int64_t mark(uintptr_t stream, int device) override {
    Dev& s = state(device);
    Push push(*d_, s.ctx);
    int64_t id;
    if (!free_marks_.empty()) {
      id = free_marks_.back();
      free_marks_.pop_back();
    } else {
      id = static_cast<int64_t>(marks_.size());
      marks_.push_back({});
    }
    Mark& m = marks_[id];
    if (!m.ev || m.dev != device) {
      if (m.ev) d_->cuEventDestroy(m.ev);
      m.ev = nullptr;
      check(d_->cuEventCreate(&m.ev, CU_EVENT_DEFAULT), "cuEventCreate");
      m.dev = device;
    }
    check(d_->cuEventRecord(m.ev, reinterpret_cast<CUstream>(stream)), "cuEventRecord");
    return id;
  }

  double elapsed(int64_t a, int64_t b) override {
    if (a < 0 || b < 0) return 0;
    Push push(*d_, devs_[marks_[b].dev].ctx);
    const CUresult q = d_->cuEventQuery(marks_[b].ev);
    if (q == CUDA_ERROR_NOT_READY) return -1;
    check(q, "cuEventQuery");
    float ms = 0;
    check(d_->cuEventElapsedTime(&ms, marks_[a].ev, marks_[b].ev), "cuEventElapsedTime");
    release(a);
    release(b);
    return static_cast<double>(ms) * 1000.0;
  }

  void release(int64_t m) override {
    if (m >= 0) free_marks_.push_back(m);
  }

  Json info() override {
    Json devs = Json::array();
    for (size_t i = 0; i < devs_.size(); ++i) {
      Dev& s = state(static_cast<int>(i));
      devs.push(Json::object()
                    .set("index", static_cast<int>(i))
                    .set("kernels", s.kernels ? Json(true) : Json(s.kernel_error))
                    .set("batch", s.batch ? Json(true) : Json(s.batch_error)));
    }
    return devs;
  }

 private:
  struct Slot {
    void* host = nullptr;
    CUdeviceptr dev = 0;
    size_t cap = 0;
    CUevent done = nullptr;
    bool used = false;
  };

  struct Bounce {
    void* host = nullptr;
    size_t cap = 0;
    CUevent done = nullptr;
    bool used = false;
  };

  struct Dev {
    bool init = false;
    int index = 0;
    CUdevice dev{};
    CUcontext ctx{};
    CUmodule mod{};
    bool kernels = false, batch = false;
    std::string kernel_error = "not built", batch_error;
    std::unordered_map<std::string, CUfunction> fns;
    CUstream side{};
    CUevent ev_in{}, ev_out{};
    std::array<Slot, kSlots> slots;
    int next = 0;
    CUdeviceptr staging = 0;
    size_t staging_cap = 0;
    CUevent staging_done{};
    CUstream staging_stream{};
    bool staging_used = false;
    std::array<Bounce, 2> bounce;
  };

  // Makes `c` current for a scope. torch and ce share each device's primary
  // context, so it usually already is: then nothing is pushed or popped.
  struct Push {
    const Driver& d;
    bool pushed = false;
    Push(const Driver& drv, CUcontext c) : d(drv) {
      CUcontext cur{};
      if (d.cuCtxGetCurrent(&cur) == CUDA_SUCCESS && cur == c) return;
      check(d.cuCtxPushCurrent(c), "cuCtxPushCurrent");
      pushed = true;
    }
    ~Push() {
      if (!pushed) return;
      CUcontext dummy{};
      d.cuCtxPopCurrent(&dummy);
    }
  };

  static int device_of(const Group& g) {
    return g.src.device.kind == DeviceKind::cuda ? g.src.device.index : g.dst.device.index;
  }

  Dev& state(int di) {
    if (di < 0 || di >= static_cast<int>(devs_.size())) fail(Errc::invalid_argument, "CUDA device index out of range");
    Dev& s = devs_[di];
    if (s.init) return s;
    s.index = di;
    check(d_->cuDeviceGet(&s.dev, di), "cuDeviceGet");
    check(d_->cuDevicePrimaryCtxRetain(&s.ctx, s.dev), "cuDevicePrimaryCtxRetain");
    s.init = true;
    Push push(*d_, s.ctx);
    check(d_->cuStreamCreate(&s.side, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    check(d_->cuEventCreate(&s.ev_in, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    check(d_->cuEventCreate(&s.ev_out, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    check(d_->cuEventCreate(&s.staging_done, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    for (Slot& sl : s.slots) check(d_->cuEventCreate(&sl.done, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    admit_kernels(s);
    admit_batch(s);
    return s;
  }

  // ---- admission: call once, verify bit-exact ------------------------------------

  void admit_kernels(Dev& s) {
#if CE_CUDA_KERNELS
    CUresult r = d_->cuModuleLoadData(&s.mod, ce_kernels_fatbin);
    if (r != CUDA_SUCCESS) {
      s.kernel_error = "cuModuleLoadData: " + cuda::error_name(r) + " (no SASS/PTX for this GPU?)";
      return;
    }
    s.kernels = true;  // provisionally, so the verification can launch
    try {
      const int n = 1000;
      std::vector<float> in(n);
      for (int i = 0; i < n; ++i) in[i] = static_cast<float>(i) * 0.37f - 100.0f;
      std::vector<uint16_t> want(n), got(n);
      host_copy(in.data(), DType::f32, want.data(), DType::f16, n, {});
      CUdeviceptr a = 0, b = 0;
      check(d_->cuMemAlloc(&a, n * 4), "cuMemAlloc");
      check(d_->cuMemAlloc(&b, n * 2), "cuMemAlloc");
      check(d_->cuMemcpyHtoDAsync(a, in.data(), n * 4, s.side), "cuMemcpyHtoDAsync");
      std::vector<Chunk> ch{{static_cast<long long>(a / 4), static_cast<long long>(b / 2), n, 1, 0, 0}};
      launch(s, "ce_cast_f32_f16", ch, {}, s.side);
      check(d_->cuMemcpyDtoHAsync(got.data(), b, n * 2, s.side), "cuMemcpyDtoHAsync");
      check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
      d_->cuMemFree(a);
      d_->cuMemFree(b);
      if (got != want) {
        s.kernels = false;
        s.kernel_error = "kernel admission: f32->f16 result differs from host conversion";
      }
    } catch (const std::exception& e) {
      s.kernels = false;
      s.kernel_error = std::string("kernel admission: ") + e.what();
    }
#else
    s.kernel_error = "built without nvcc: no kernel module";
#endif
  }

  void admit_batch(Dev& s) {
    if (!d_->cuMemcpyBatchAsync) {
      s.batch_error = "cuMemcpyBatchAsync: " + cuda::missing_reason("cuMemcpyBatchAsync");
      return;
    }
    s.batch = true;
    try {
      const size_t segs = 16, len = 1000, pitch = 1024;
      void* h = nullptr;
      check(d_->cuMemHostAlloc(&h, segs * len + segs * pitch, 0), "cuMemHostAlloc");
      CUdeviceptr dp = 0;
      check(d_->cuMemAlloc(&dp, segs * pitch), "cuMemAlloc");
      auto* src = static_cast<uint8_t*>(h);
      auto* back = src + segs * len;
      for (size_t i = 0; i < segs * len; ++i) src[i] = static_cast<uint8_t>(i * 131 + 7);
      check(d_->cuMemsetD8Async(dp, 0, segs * pitch, s.side), "cuMemsetD8Async");
      std::vector<Raw> raws;
      for (size_t i = 0; i < segs; ++i)
        raws.push_back({reinterpret_cast<uintptr_t>(src + i * len), dp + (segs - 1 - i) * pitch, len});
      Side hs{{DeviceKind::host, 0}, DType::u8, true}, ds{{DeviceKind::cuda, s.index}, DType::u8, false};
      batch(s, hs, ds, raws, s.side);
      check(d_->cuMemcpyDtoHAsync(back, dp, segs * pitch, s.side), "cuMemcpyDtoHAsync");
      check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
      bool ok = true;
      for (size_t i = 0; i < segs && ok; ++i) {
        ok = std::memcmp(back + (segs - 1 - i) * pitch, src + i * len, len) == 0;
        for (size_t k = len; k < pitch && ok; ++k) ok = back[(segs - 1 - i) * pitch + k] == 0;
      }
      d_->cuMemFree(dp);
      d_->cuMemFreeHost(h);
      if (!ok) s.batch = false, s.batch_error = "batch admission: wrong bytes";
    } catch (const std::exception& e) {
      s.batch = false;
      s.batch_error = std::string("batch admission: ") + e.what();
    }
  }

  // ---- primitives -------------------------------------------------------------------

  void copy1d(const Side& src, const Side& dst, uintptr_t s, uintptr_t d, size_t n, CUstream st) {
    const bool sc = src.device.kind == DeviceKind::cuda, dc = dst.device.kind == DeviceKind::cuda;
    if (sc && dc) check(d_->cuMemcpyDtoDAsync(d, s, n, st), "cuMemcpyDtoDAsync");
    else if (dc) check(d_->cuMemcpyHtoDAsync(d, reinterpret_cast<const void*>(s), n, st), "cuMemcpyHtoDAsync");
    else check(d_->cuMemcpyDtoHAsync(reinterpret_cast<void*>(d), s, n, st), "cuMemcpyDtoHAsync");
  }

  // Returns false when the driver rejects the 2-D shape (caller falls back to rows).
  bool copy2d(const Side& src, const Side& dst, uintptr_t s, uintptr_t d, size_t width, size_t height,
              size_t spitch, size_t dpitch, CUstream st) {
    CUDA_MEMCPY2D p{};
    if (src.device.kind == DeviceKind::cuda) p.srcMemoryType = CU_MEMORYTYPE_DEVICE, p.srcDevice = s;
    else p.srcMemoryType = CU_MEMORYTYPE_HOST, p.srcHost = reinterpret_cast<const void*>(s);
    if (dst.device.kind == DeviceKind::cuda) p.dstMemoryType = CU_MEMORYTYPE_DEVICE, p.dstDevice = d;
    else p.dstMemoryType = CU_MEMORYTYPE_HOST, p.dstHost = reinterpret_cast<void*>(d);
    p.srcPitch = spitch, p.dstPitch = dpitch, p.WidthInBytes = width, p.Height = height;
    const CUresult r = d_->cuMemcpy2DAsync(&p, st);
    if (r == CUDA_ERROR_INVALID_VALUE) return false;
    check(r, "cuMemcpy2DAsync");
    return true;
  }

  // Rows of the group's rects as raw byte copies.
  static std::vector<Raw> rows_raw(const Group& g) {
    std::vector<Raw> out;
    const size_t ss = itemsize(g.src.dtype), ds = itemsize(g.dst.dtype);
    const Rects& R = g.rects;
    for (size_t r = 0; r < R.size(); ++r)
      for (i64 h = 0; h < R.height[r]; ++h)
        out.push_back({static_cast<uintptr_t>(R.src[r] + h * R.src_pitch[r]) * ss,
                       static_cast<uintptr_t>(R.dst[r] + h * R.dst_pitch[r]) * ds,
                       static_cast<size_t>(R.width[r]) * ss});
    return out;
  }

  static const std::vector<Raw>& cached_rows(const Group& g, Strategy which) {
    auto& slot = g.exec_cache[static_cast<int>(which)];
    if (!slot) slot = std::make_shared<std::vector<Raw>>(rows_raw(g));
    return *static_cast<std::vector<Raw>*>(slot.get());
  }

  int64_t runs(const Group& g, CUstream st) {
    const size_t ss = itemsize(g.src.dtype);
    const Rects& R = g.rects;
    int64_t calls = 0;
    for (size_t r = 0; r < R.size(); ++r) {
      const uintptr_t s = static_cast<uintptr_t>(R.src[r]) * ss, d = static_cast<uintptr_t>(R.dst[r]) * ss;
      const size_t w = static_cast<size_t>(R.width[r]) * ss;
      if (R.height[r] > 1 &&
          copy2d(g.src, g.dst, s, d, w, R.height[r], R.src_pitch[r] * ss, R.dst_pitch[r] * ss, st)) {
        ++calls;
        continue;
      }
      for (i64 h = 0; h < R.height[r]; ++h, ++calls)
        copy1d(g.src, g.dst, s + h * R.src_pitch[r] * ss, d + h * R.dst_pitch[r] * ss, w, st);
    }
    return calls;
  }

  // Batch API on a non-legacy stream; stream 0 is fenced through the side stream.
  int64_t batch(Dev& s, const Side& src, const Side& dst, const std::vector<Raw>& raws, CUstream st) {
    if (raws.empty()) return 0;
    CUstream run = st;
    if (!st) {
      check(d_->cuEventRecord(s.ev_in, st), "cuEventRecord");
      check(d_->cuStreamWaitEvent(s.side, s.ev_in, 0), "cuStreamWaitEvent");
      run = s.side;
    }
    CUmemcpyAttributes attr{};
    attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    attr.srcLocHint.type = src.device.kind == DeviceKind::cuda ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST;
    attr.srcLocHint.id = src.device.kind == DeviceKind::cuda ? src.device.index : 0;
    attr.dstLocHint.type = dst.device.kind == DeviceKind::cuda ? CU_MEM_LOCATION_TYPE_DEVICE : CU_MEM_LOCATION_TYPE_HOST;
    attr.dstLocHint.id = dst.device.kind == DeviceKind::cuda ? dst.device.index : 0;
    int64_t calls = 0;
    std::vector<CUdeviceptr> ds, ss;
    std::vector<size_t> sizes;
    for (size_t at = 0; at < raws.size(); at += kBatchMax) {
      const size_t n = std::min(kBatchMax, raws.size() - at);
      ds.resize(n), ss.resize(n), sizes.resize(n);
      for (size_t i = 0; i < n; ++i) ds[i] = raws[at + i].dst, ss[i] = raws[at + i].src, sizes[i] = raws[at + i].bytes;
      size_t idx = 0, fail_idx = SIZE_MAX;
      const CUresult r =
          d_->cuMemcpyBatchAsync(ds.data(), ss.data(), sizes.data(), n, &attr, &idx, 1, &fail_idx, run);
      if (r != CUDA_SUCCESS)
        fail(Errc::backend, "cuMemcpyBatchAsync: " + cuda::error_name(r) + " (failIdx " + std::to_string(fail_idx) + ")");
      ++calls;
    }
    if (!st) {
      check(d_->cuEventRecord(s.ev_out, s.side), "cuEventRecord");
      check(d_->cuStreamWaitEvent(st, s.ev_out, 0), "cuStreamWaitEvent");
    }
    return calls;
  }

  void copy_raws(Dev& s, const Side& src, const Side& dst, const std::vector<Raw>& raws, CUstream st) {
    const bool host_ok = (src.device.kind == DeviceKind::cuda || src.pinned) && (dst.device.kind == DeviceKind::cuda || dst.pinned);
    if (!host_ok && raws.size() > 2) {
      bounce(s, src, dst, raws, st);
      return;
    }
    if (s.batch && host_ok && raws.size() > 2) {
      batch(s, src, dst, raws, st);
      return;
    }
    for (const Raw& r : raws) copy1d(src, dst, r.src, r.dst, r.bytes, st);
  }

  // Pageable host side: pack runs into pinned double buffers on the host and
  // move each buffer with batched DMA, overlapping packing with transfers.
  // D2H must wait for each buffer before unpacking, so it blocks like any
  // pageable D2H does.
  int64_t bounce(Dev& s, const Side& src, const Side& dst, const std::vector<Raw>& raws, CUstream st) {
    const bool h2d = dst.device.kind == DeviceKind::cuda;
    // The CPU is about to touch host memory that earlier work on this stream may still use.
    check(d_->cuStreamSynchronize(st), "cuStreamSynchronize");
    const size_t cap = static_cast<size_t>(std::clamp<int64_t>(opt_.staging_budget / 2, 1 << 20, 64 << 20));
    for (Bounce& b : s.bounce) {
      if (b.cap >= cap) continue;
      if (b.used) check(d_->cuEventSynchronize(b.done), "cuEventSynchronize");
      if (b.host) d_->cuMemFreeHost(b.host);
      b.host = nullptr;
      check(d_->cuMemHostAlloc(&b.host, cap, 0), "cuMemHostAlloc (bounce)");
      if (!b.done) check(d_->cuEventCreate(&b.done, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
      b.cap = cap, b.used = false;
    }
    const Side pinned_host{{DeviceKind::host, 0}, DType::u8, true};
    struct Pending {
      int buf;
      std::vector<Raw> unpack;  // D2H: bounce -> pageable
    };
    std::vector<Pending> pending;
    auto drain = [&](Pending& p) {
      check(d_->cuEventSynchronize(s.bounce[p.buf].done), "cuEventSynchronize");
      for (const Raw& r : p.unpack) std::memcpy(reinterpret_cast<void*>(r.dst), reinterpret_cast<const void*>(r.src), r.bytes);
    };
    int64_t calls = 0;
    int k = 0;
    size_t i = 0;
    while (i < raws.size()) {
      Bounce& b = s.bounce[k % 2];
      if (!pending.empty() && pending.front().buf == k % 2) {
        drain(pending.front());
        pending.erase(pending.begin());
      } else if (b.used) {
        check(d_->cuEventSynchronize(b.done), "cuEventSynchronize");
      }
      std::vector<Raw> dev_raws, unpack;
      size_t off = 0;
      for (; i < raws.size(); ++i) {
        const Raw& r = raws[i];
        if (r.bytes > cap) {  // a single huge run: plain async copy, the driver stages it
          if (off) break;
          copy1d(src, dst, r.src, r.dst, r.bytes, st);
          ++calls;
          continue;
        }
        if (off + r.bytes > cap) break;
        const uintptr_t at = reinterpret_cast<uintptr_t>(b.host) + off;
        if (h2d) {
          std::memcpy(reinterpret_cast<void*>(at), reinterpret_cast<const void*>(r.src), r.bytes);
          dev_raws.push_back({at, r.dst, r.bytes});
        } else {
          dev_raws.push_back({r.src, at, r.bytes});
          unpack.push_back({at, r.dst, r.bytes});
        }
        off += r.bytes;
      }
      if (dev_raws.empty()) continue;
      if (h2d) copy_raws(s, pinned_host, dst, dev_raws, st);
      else copy_raws(s, src, pinned_host, dev_raws, st);
      check(d_->cuEventRecord(b.done, st), "cuEventRecord");
      b.used = true;
      ++calls;
      if (!h2d) pending.push_back({k % 2, std::move(unpack)});
      ++k;
    }
    for (Pending& p : pending) drain(p);
    return calls;
  }

  // ---- kernels ------------------------------------------------------------------------

  // A kernel the fatbin may not have (specializations): null if missing.
  CUfunction find_function(Dev& s, const std::string& name) {
    auto it = s.fns.find(name);
    if (it != s.fns.end()) return it->second;
    CUfunction f{};
    if (d_->cuModuleGetFunction(&f, s.mod, name.c_str()) != CUDA_SUCCESS) f = nullptr;
    s.fns.emplace(name, f);
    return f;
  }

  // The specialized ce_stmt for this dtype pair and mode, if compiled.
  static std::string stmt_kernel(const StmtDesc& d) {
    const DType sd = static_cast<DType>(d.sdt), dd = static_cast<DType>(d.ddt);
    if (d.mode == 0) return "ce_stmt_b" + std::to_string(itemsize(sd));
    return std::string("ce_stmt_") + (d.mode == 1 ? "cast_" : "apply_") + short_name(sd) + "_" + short_name(dd);
  }

  CUfunction function(Dev& s, const std::string& name) {
    auto it = s.fns.find(name);
    if (it != s.fns.end()) return it->second;
    CUfunction f{};
    check(d_->cuModuleGetFunction(&f, s.mod, name.c_str()), ("cuModuleGetFunction " + name).c_str());
    s.fns.emplace(name, f);
    return f;
  }

  // Copies a parameter table to the device through the pinned/device slot
  // ring. The caller launches, then records slot.done on the stream.
  Slot& upload(Dev& s, const void* data, size_t need, CUstream st) {
    Slot& sl = s.slots[s.next++ % kSlots];
    if (sl.used) check(d_->cuEventSynchronize(sl.done), "cuEventSynchronize");
    if (sl.cap < need) {
      if (sl.host) d_->cuMemFreeHost(sl.host);
      if (sl.dev) d_->cuMemFree(sl.dev);
      sl.host = nullptr, sl.dev = 0;
      size_t cap = 64 << 10;
      while (cap < need) cap <<= 1;
      check(d_->cuMemHostAlloc(&sl.host, cap, 0), "cuMemHostAlloc");
      check(d_->cuMemAlloc(&sl.dev, cap), "cuMemAlloc");
      sl.cap = cap;
    }
    std::memcpy(sl.host, data, need);
    check(d_->cuMemcpyHtoDAsync(sl.dev, sl.host, need, st), "cuMemcpyHtoDAsync");
    return sl;
  }

  // A parameter table in device memory for the life of the group that built
  // it: launches read it in place, so the hot path uploads nothing.
  struct DevTable {
    const Driver* d = nullptr;
    CUcontext ctx{};
    CUdeviceptr ptr = 0;
    size_t count = 0;
    std::weak_ptr<int> alive;  // the executor
    ~DevTable() {
      if (!ptr || alive.expired()) return;
      d->cuCtxPushCurrent(ctx);
      d->cuMemFree(ptr);
      CUcontext dummy{};
      d->cuCtxPopCurrent(&dummy);
    }
  };

  // Built once, when a group first runs a strategy (planning, not the hot path).
  template <class T>
  std::shared_ptr<DevTable> resident(Dev& s, const std::vector<T>& v) {
    auto t = std::make_shared<DevTable>();
    t->d = d_, t->ctx = s.ctx, t->alive = alive_, t->count = v.size();
    if (v.empty()) return t;
    const size_t bytes = v.size() * sizeof(T);
    check(d_->cuMemAlloc(&t->ptr, bytes), "cuMemAlloc (table)");
    check(d_->cuMemcpyHtoDAsync(t->ptr, v.data(), bytes, s.side), "cuMemcpyHtoDAsync");
    check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
    return t;
  }

  // Chunk kernels on a device table; `sadd` / `dadd` offset every chunk.
  void launch_table(Dev& s, const std::string& name, CUdeviceptr table, size_t chunks, const cv::Fn& fn,
                    long long sadd, long long dadd, CUstream st) {
    if (chunks == 0) return;
    CUfunction f = function(s, name);
    const bool bytes = name == "ce_bytes";  // no function argument
    for (size_t at = 0; at < chunks;) {
      const size_t n = std::min<size_t>(chunks - at, size_t{1} << 24);
      int count = static_cast<int>(n);
      CUdeviceptr t = table + at * sizeof(Chunk);
      cv::Fn fcopy = fn;
      void* typed_args[] = {&t, &count, &fcopy, &sadd, &dadd};
      void* bytes_args[] = {&t, &count, &sadd, &dadd};
      const unsigned threads = 256, warps = threads / 32;
      const unsigned blocks = static_cast<unsigned>((n + warps - 1) / warps);
      check(d_->cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, st, bytes ? bytes_args : typed_args, nullptr),
            ("cuLaunchKernel " + name).c_str());
      at += n;
    }
  }

  // A one-off table (admission, passthrough of a single op) through the upload ring.
  void launch(Dev& s, const std::string& name, const std::vector<Chunk>& chunks, const cv::Fn& fn, CUstream st) {
    if (chunks.empty()) return;
    Slot& sl = upload(s, chunks.data(), chunks.size() * sizeof(Chunk), st);
    launch_table(s, name, sl.dev, chunks.size(), fn, 0, 0, st);
    check(d_->cuEventRecord(sl.done, st), "cuEventRecord");
    sl.used = true;
  }

  // Destination elements one thread of ce_stmt stores at once: the innermost
  // dimension is contiguous on the destination, divisible by the width, and
  // every statement's destination base (plus `dadd`) is aligned to it.
  static int vector_width(const FusedBatch& b, long long dadd) {
    const StmtDesc& d = b.desc;
    if (d.rank == 0 || d.ds[d.rank - 1] != 1 || b.tile_a >= 0) return 1;
    const int dsz = static_cast<int>(itemsize(static_cast<DType>(d.ddt)));
    for (int v = 16 / dsz; v >= 2; v /= 2) {
      if (d.shape[d.rank - 1] % v != 0) continue;
      bool ok = true;
      for (const Base& x : b.bases) ok = ok && ((x.dst + dadd) % v) == 0;
      if (ok) return v;
    }
    return 1;
  }

  // Destination and source each have a unit-stride dimension, and they
  // differ (a transpose): the tiled kernel keeps both sides coalesced.
  static void transpose_dims(const Form& f, int* a, int* b) {
    int da = -1, sb = -1;
    for (int k = 0; k < f.rank; ++k) {
      if (f.ds[k] == 1 || f.ds[k] == -1) da = k;
      if (f.ss[k] == 1 || f.ss[k] == -1) sb = k;
    }
    if (da >= 0 && sb >= 0 && da != sb && f.shape[da] >= 32 && f.shape[sb] >= 32) *a = da, *b = sb;  // full tiles only
  }

  void launch_stmt(const DevTable& table, const FusedBatch& batch, const cv::Fn& fn, long long sadd, long long dadd,
                   Dev& s, CUstream st) {
    if (table.count == 0) return;
    const StmtDesc& desc = batch.desc;
    const bool tiled = batch.tile_a >= 0;
    CUfunction f = tiled ? function(s, "ce_stmt_tile") : find_function(s, stmt_kernel(desc));
    if (!f) f = function(s, "ce_stmt");
    CUdeviceptr t = table.ptr;
    StmtDesc dcopy = desc;
    cv::Fn fcopy = fn;
    if (tiled) {
      const long long sa = desc.shape[batch.tile_a], sb = desc.shape[batch.tile_b];
      long long count = static_cast<long long>(table.count) * (static_cast<long long>(desc.numel) / (sa * sb));
      int a = batch.tile_a, b = batch.tile_b;
      void* args[] = {&t, &count, &dcopy, &fcopy, &a, &b, &sadd, &dadd};
      const unsigned gx = static_cast<unsigned>((sb + 31) / 32), gy = static_cast<unsigned>((sa + 31) / 32);
      const unsigned gz = static_cast<unsigned>(std::min<long long>(count, 65535));
      check(d_->cuLaunchKernel(f, gx, gy, gz, 32, 8, 1, 0, st, args, nullptr), "cuLaunchKernel ce_stmt_tile");
    } else {
      long long count = static_cast<long long>(table.count);
      void* args[] = {&t, &count, &dcopy, &fcopy, &sadd, &dadd};
      const unsigned threads = 256;
      const unsigned per = (desc.numel + desc.vec - 1) / desc.vec;
      const unsigned gx = (per + threads - 1) / threads;
      const unsigned gy = static_cast<unsigned>(std::min<long long>(count, 65535));
      check(d_->cuLaunchKernel(f, gx, gy, 1, threads, 1, 1, 0, st, args, nullptr), "cuLaunchKernel ce_stmt");
    }
  }

  // ---- fused structured statements --------------------------------------------------------

  FusedPlan build_fused(const Group& g) const {
    FusedPlan P;
    if (g.forms.empty()) return P;
    const bool src_cuda = g.src.device.kind == DeviceKind::cuda, dst_cuda = g.dst.device.kind == DeviceKind::cuda;
    if (src_cuda && dst_cuda && g.src.device.index != g.dst.device.index) return P;
    P.cross = !(src_cuda && dst_cuda);
    P.h2d = dst_cuda;
    const int mode = g.pure() ? 0 : (g.fn.identity() ? 1 : 2);
    std::vector<const Form*> reps;
    std::vector<size_t> batch_of(g.forms.size());
    for (size_t i = 0; i < g.forms.size(); ++i) {
      const Form& f = *g.forms[i];
      size_t j = 0;
      while (j < reps.size() && !reps[j]->same_structure(f)) ++j;
      if (j == reps.size()) {
        if (reps.size() == kMaxFusedBatches) return P;
        reps.push_back(&f);
        FusedBatch b;
        b.desc.rank = f.rank, b.desc.sdt = static_cast<int>(f.sdt), b.desc.ddt = static_cast<int>(f.ddt);
        b.desc.mode = mode, b.desc.numel = static_cast<unsigned>(f.numel);
        for (int k = 0; k < f.rank; ++k) {
          b.desc.shape[k] = static_cast<unsigned>(f.shape[k]), b.desc.ss[k] = f.ss[k], b.desc.ds[k] = f.ds[k];
          int_divider(b.desc.shape[k], &b.desc.magic[k], &b.desc.shift[k]);
        }
        b.link_size = static_cast<i64>(itemsize(P.h2d ? f.sdt : f.ddt));
        transpose_dims(f, &b.tile_a, &b.tile_b);
        b.desc.vec = 1;
        P.batches.push_back(std::move(b));
      }
      batch_of[i] = j;
      P.batches[j].bases.push_back({f.s_base, f.d_base});
    }
    if (!P.cross) {
      for (FusedBatch& b : P.batches) b.desc.vec = vector_width(b, 0);
      P.ok = true;
      return P;
    }
    // Link-side byte intervals: the whole extent of every statement.
    std::vector<i64> lo(g.forms.size()), len(g.forms.size());
    for (size_t i = 0; i < g.forms.size(); ++i) {
      const Form& f = *g.forms[i];
      const i64 sz = P.batches[batch_of[i]].link_size;
      const auto [a, b] = P.h2d ? f.src_extent() : f.dst_extent();
      // D2H writes the staged bytes back verbatim: the destination must have no holes.
      if (!P.h2d && b - a + 1 != f.numel) return FusedPlan{};
      lo[i] = a * sz, len[i] = (b - a + 1) * sz;
    }
    // One raw copy per interval: never across an allocation boundary. A cut
    // inside a statement's extent would split its staging; that is the same
    // memory described twice, but leave such groups to other strategies.
    const std::vector<i64>& cuts = P.h2d ? g.src_cuts : g.dst_cuts;
    const i64 cut_scale = static_cast<i64>(itemsize(P.h2d ? g.src.dtype : g.dst.dtype));
    std::vector<i64> byte_cuts;
    for (i64 c : cuts) byte_cuts.push_back(c * cut_scale);
    if (real_cuts(byte_cuts, lo, len).size() != byte_cuts.size()) return FusedPlan{};
    const Intervals iv = split_at(interval_union(lo, len), byte_cuts);
    // Staging offsets keep each interval's alignment modulo 8.
    std::vector<i64> off(iv.start.size());
    i64 cur = 0, payload = 0;
    for (size_t k = 0; k < iv.start.size(); ++k) {
      cur = ((cur + 7) & ~i64{7}) + (iv.start[k] & 7);
      off[k] = cur;
      cur += iv.end[k] - iv.start[k];
      payload += iv.end[k] - iv.start[k];
    }
    const int64_t budget = std::max<int64_t>(opt_.staging_budget, 1 << 20);
    if (cur > budget) return FusedPlan{};
    if (P.h2d && payload > 2 * g.bytes() + (4 << 10)) return FusedPlan{};  // sparse source: not worth moving whole extents
    P.stg_bytes = static_cast<size_t>(cur);
    for (size_t k = 0; k < iv.start.size(); ++k) {
      const size_t n = static_cast<size_t>(iv.end[k] - iv.start[k]);
      if (P.h2d) P.raws.push_back({static_cast<uintptr_t>(iv.start[k]), static_cast<uintptr_t>(off[k]), n});
      else P.raws.push_back({static_cast<uintptr_t>(off[k]), static_cast<uintptr_t>(iv.start[k]), n});
    }
    // Rebase the link side of every statement into staging (relative elements).
    std::vector<size_t> seen(P.batches.size(), 0);
    for (size_t i = 0; i < g.forms.size(); ++i) {
      FusedBatch& b = P.batches[batch_of[i]];
      Base& base = b.bases[seen[batch_of[i]]++];
      long long& link = P.h2d ? base.src : base.dst;
      const i64 byte = static_cast<i64>(link) * b.link_size;
      const size_t k = static_cast<size_t>(std::upper_bound(iv.start.begin(), iv.start.end(), byte) - iv.start.begin()) - 1;
      link = (off[k] + (byte - iv.start[k])) / b.link_size;
    }
    // D2H writes into staging (based at 256-byte aligned addresses); H2D's
    // destination keeps its own alignment.
    for (FusedBatch& b : P.batches) b.desc.vec = vector_width(b, 0);
    P.ok = true;
    return P;
  }

  const FusedPlan& fused_plan(const Group& g) const {
    auto& slot = g.exec_cache[static_cast<int>(Strategy::fused)];
    if (!slot) slot = std::make_shared<FusedPlan>(build_fused(g));
    return *static_cast<FusedPlan*>(slot.get());
  }

  // Short runs are where evaluating the statement beats copying runs.
  bool fused_worthwhile(const Group& g) const {
    if (g.forms.empty() || g.runs.empty()) return false;
    const i64 avg = g.runs.total() / static_cast<i64>(g.runs.size());  // bytes for pure groups, else elements
    return avg < (g.pure() ? 256 : 64) && fused_plan(g).ok;
  }

  int64_t fused(Dev& s, const Group& g, CUstream st) {
    const FusedPlan& P = fused_plan(g);
    if (!P.ok) fail(Errc::unsupported, "fused strategy is not applicable to this group");
    const cv::Fn fn = to_cv(g.fn);
    CUdeviceptr stg = 0;
    std::vector<Raw> raws;
    const Side dev_side{P.h2d ? g.dst.device : g.src.device, DType::u8, false};
    const Side host_side{{DeviceKind::host, 0}, DType::u8, P.h2d ? g.src.pinned : g.dst.pinned};
    if (P.cross) {
      stg = staging(s, P.stg_bytes, st);
      raws = P.raws;
      for (Raw& r : raws) (P.h2d ? r.dst : r.src) += stg;
      if (P.h2d) copy_raws(s, host_side, dev_side, raws, st);
    }
    for (const FusedBatch& b : P.batches) {
      auto& table = const_cast<FusedBatch&>(b).table;
      if (!table) table = resident(s, b.bases);
      // Cross-device: the link side's indices are relative to the staging base.
      const long long add = P.cross ? static_cast<long long>(stg) / b.link_size : 0;
      launch_stmt(*static_cast<DevTable*>(table.get()), b, fn, P.h2d ? add : 0, P.h2d ? 0 : add, s, st);
    }
    if (P.cross) {
      if (!P.h2d) copy_raws(s, dev_side, host_side, raws, st);
      staging_release(s, st);
    }
    return static_cast<int64_t>(P.batches.size()) + (P.cross ? static_cast<int64_t>(P.raws.size() > 2 ? 1 : P.raws.size()) : 0);
  }

  static std::string kernel_name(const Group& g) {
    if (g.pure()) return "ce_bytes";
    return std::string(g.fn.identity() ? "ce_cast_" : "ce_apply_") + short_name(g.src.dtype) + "_" + short_name(g.dst.dtype);
  }

  static std::vector<Chunk> chunks_of(const Group& g) {
    ChunkBuilder cb(g.pure() ? kByteChunk : kTypedChunk);
    for (size_t i = 0; i < g.runs.size(); ++i) cb.add(g.runs.src[i], g.runs.dst[i], g.runs.len[i]);
    return cb.take();
  }

  // `resident`: the group belongs to a cached plan, so its table can live on
  // the device; a one-off group (passthrough of one op) uploads it per call.
  int64_t kernel(Dev& s, const Group& g, CUstream st, bool resident_table = true) {
    if (!resident_table) {
      launch(s, kernel_name(g), chunks_of(g), to_cv(g.fn), st);
      return 1;
    }
    auto& slot = g.exec_cache[static_cast<int>(Strategy::kernel)];
    if (!slot) slot = resident(s, chunks_of(g));
    const auto& t = *static_cast<DevTable*>(slot.get());
    launch_table(s, kernel_name(g), t.ptr, t.count, to_cv(g.fn), 0, 0, st);
    return 1;
  }

  // ---- staging --------------------------------------------------------------------------

  CUdeviceptr staging(Dev& s, size_t bytes, CUstream st) {
    if (s.staging_cap < bytes) {
      if (s.staging_used) check(d_->cuEventSynchronize(s.staging_done), "cuEventSynchronize");
      if (s.staging) d_->cuMemFree(s.staging);
      s.staging = 0;
      size_t cap = 1 << 20;
      while (cap < bytes) cap <<= 1;
      cap = std::min<size_t>(cap, static_cast<size_t>(std::max<int64_t>(opt_.staging_budget, 1 << 20)));
      if (cap < bytes) fail(Errc::invalid_argument, "staging request exceeds the budget");
      check(d_->cuMemAlloc(&s.staging, cap), "cuMemAlloc (staging)");
      s.staging_cap = cap;
      s.staging_used = false;
    }
    if (s.staging_used && s.staging_stream != st) check(d_->cuStreamWaitEvent(st, s.staging_done, 0), "cuStreamWaitEvent");
    return s.staging;
  }

  void staging_release(Dev& s, CUstream st) {
    check(d_->cuEventRecord(s.staging_done, st), "cuEventRecord");
    s.staging_used = true;
    s.staging_stream = st;
  }

  // Host <-> device through device staging. Only dense intervals cross the
  // link (read gaps below kGapBytes inside one source allocation are copied
  // too); a kernel on the device side scatters / gathers and converts. This is
  // the "cut point on the device" transform: layout changes happen in HBM.
  // One staging round: raw link copies and kernel chunks, with staging-side
  // addresses relative to the staging base (it may be reallocated).
  struct StagedStep {
    size_t bytes = 0;
    std::vector<Raw> raws;
    std::vector<Chunk> chunks;
    std::shared_ptr<DevTable> table;  // the chunks, resident on the device (set on first run)
  };

  std::vector<StagedStep> build_staged(const Group& g) const {
    const bool h2d = g.dst.device.kind == DeviceKind::cuda;
    const DType lt = h2d ? g.src.dtype : g.dst.dtype;  // dtype crossing the link
    const i64 isz = static_cast<i64>(itemsize(lt));
    const int64_t budget = std::max<int64_t>(opt_.staging_budget, 1 << 20);
    const i64 per = std::max<i64>(1, budget / isz);

    // Link-side intervals (elements of lt), sorted.
    const std::vector<i64>& link = h2d ? g.runs.src : g.runs.dst;
    // One raw copy per interval: never across an allocation boundary.
    Intervals iv = split_at(interval_union(link, g.runs.len), h2d ? g.src_cuts : g.dst_cuts);
    if (h2d) {  // widen across small read gaps that stay inside one source allocation
      // Allocations by start, with the furthest end among each prefix.
      std::vector<i64> alo, reach;
      for (const auto& [lo, hi] : g.src_allocs) {
        alo.push_back(lo);
        reach.push_back(reach.empty() ? hi : std::max(reach.back(), hi));
      }
      Intervals m;
      for (size_t k = 0; k < iv.start.size(); ++k) {
        if (!m.empty()) {
          const i64 gap_lo = m.end.back() * isz, gap_hi = iv.start[k] * isz;
          // One allocation holds the element before the gap and the one after it.
          const size_t a = static_cast<size_t>(std::lower_bound(alo.begin(), alo.end(), gap_lo) - alo.begin());
          const bool inside = a > 0 && reach[a - 1] > gap_hi;
          if (gap_hi - gap_lo < kGapBytes && inside) {
            m.end.back() = iv.end[k];
            continue;
          }
        }
        m.start.push_back(iv.start[k]);
        m.end.push_back(iv.end[k]);
      }
      iv = std::move(m);
    }
    std::vector<std::pair<i64, i64>> pieces;
    for (size_t k = 0; k < iv.start.size(); ++k)
      for (i64 a = iv.start[k]; a < iv.end[k]; a += per) pieces.push_back({a, std::min(iv.end[k], a + per)});

    // Runs ordered by link offset; stable so runs sharing a source (broadcast)
    // keep destination order and pack into 2-D chunks.
    std::vector<size_t> order(g.runs.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return link[a] < link[b]; });

    std::vector<StagedStep> steps;
    size_t p = 0, ri = 0;
    i64 r_done = 0;
    while (p < pieces.size()) {
      size_t q = p;
      i64 total = 0;
      while (q < pieces.size() && total + (pieces[q].second - pieces[q].first) <= per) total += pieces[q].second - pieces[q].first, ++q;
      if (q == p) ++q, total = pieces[p].second - pieces[p].first;
      StagedStep step;
      step.bytes = static_cast<size_t>(total * isz);
      ChunkBuilder cb(g.pure() ? kByteChunk : kTypedChunk);
      i64 off = 0;
      for (size_t k = p; k < q; ++k) {
        const auto [a, b] = pieces[k];
        if (h2d) step.raws.push_back({static_cast<uintptr_t>(a * isz), static_cast<uintptr_t>(off * isz), static_cast<size_t>((b - a) * isz)});
        else step.raws.push_back({static_cast<uintptr_t>(off * isz), static_cast<uintptr_t>(a * isz), static_cast<size_t>((b - a) * isz)});
        while (ri < order.size()) {
          const size_t r = order[ri];
          const i64 lo = link[r] + r_done, hi = link[r] + g.runs.len[r];
          if (lo >= b) break;
          const i64 take = std::min(hi, b) - lo;
          const i64 stg_idx = off + (lo - a);  // relative to the staging base
          const i64 other = (h2d ? g.runs.dst[r] : g.runs.src[r]) + r_done;
          if (h2d) cb.add(stg_idx, other, take);
          else cb.add(other, stg_idx, take);
          r_done += take;
          if (r_done == g.runs.len[r]) ++ri, r_done = 0;
          else break;
        }
        off += b - a;
      }
      step.chunks = cb.take();
      steps.push_back(std::move(step));
      p = q;
    }
    return steps;
  }

  // Host <-> device through device staging. Only dense intervals cross the
  // link (read gaps below kGapBytes inside one source allocation are copied
  // too); a kernel on the device side scatters / gathers and converts. This is
  // the "cut point on the device" transform: layout changes happen in HBM.
  int64_t staged(Dev& s, const Group& g, CUstream st, bool resident_table = true) {
    auto& slot = g.exec_cache[static_cast<int>(Strategy::staged)];
    if (!slot) slot = std::make_shared<std::vector<StagedStep>>(build_staged(g));
    const auto& steps = *static_cast<std::vector<StagedStep>*>(slot.get());
    const bool h2d = g.dst.device.kind == DeviceKind::cuda;
    const DType lt = h2d ? g.src.dtype : g.dst.dtype;
    const i64 isz = static_cast<i64>(itemsize(lt));
    const std::string name = g.pure() ? "ce_bytes" : kernel_name(g);
    const cv::Fn fn = to_cv(g.fn);
    const Side dev_side{h2d ? g.dst.device : g.src.device, lt, false};
    int64_t calls = 0;
    std::vector<Raw> raws;
    for (const StagedStep& step : steps) {
      const CUdeviceptr stg = staging(s, step.bytes, st);
      const long long base = static_cast<long long>(stg / isz);  // staging-side chunk indices are relative
      raws = step.raws;
      for (Raw& r : raws) (h2d ? r.dst : r.src) += stg;
      auto run_kernel = [&] {
        if (!resident_table) {
          std::vector<Chunk> ch = step.chunks;
          for (Chunk& c : ch) (h2d ? c.src : c.dst) += base;
          launch(s, name, ch, fn, st);
          return;
        }
        auto& table = const_cast<StagedStep&>(step).table;
        if (!table) table = resident(s, step.chunks);
        launch_table(s, name, table->ptr, table->count, fn, h2d ? base : 0, h2d ? 0 : base, st);
      };
      if (h2d) {
        copy_raws(s, g.src, dev_side, raws, st);
        run_kernel();
      } else {
        run_kernel();
        copy_raws(s, dev_side, g.dst, raws, st);
      }
      staging_release(s, st);
      calls += 2;
    }
    return calls;
  }

  // ---- passthrough: every op exactly as submitted ------------------------------------------

  int64_t passthrough(Dev& s, const Group& g, const std::vector<Op>& ops, CUstream st) {
    int64_t calls = 0;
    for (int k : g.ops) {
      const Op& op = ops[k];
      if (g.pure()) {
        for (size_t i = 0; i < op.runs().size(); ++i, ++calls)
          copy1d(op.src, op.dst, static_cast<uintptr_t>(op.runs().src[i]), static_cast<uintptr_t>(op.runs().dst[i]),
                 static_cast<size_t>(op.runs().len[i]), st);
        continue;
      }
      Group one;
      one.src = op.src, one.dst = op.dst, one.fn = op.fn;
      one.runs = op.runs();
      one.rects = recover_2d(op.runs(), 1 << 30);  // rows only
      calls += (op.src.device.kind == DeviceKind::cuda && op.dst.device.kind == DeviceKind::cuda)
                   ? kernel(s, one, st, false)
                   : staged(s, one, st, false);
    }
    return calls;
  }

  struct Mark {
    CUevent ev = nullptr;
    int dev = -1;
  };
  std::shared_ptr<int> alive_ = std::make_shared<int>(0);
  std::vector<Mark> marks_;
  std::vector<int64_t> free_marks_;

  EngineOptions opt_;
  const Driver* d_ = nullptr;
  std::vector<Dev> devs_;
};

}  // namespace

std::unique_ptr<Executor> make_cuda_executor(const EngineOptions& opt) { return std::make_unique<CudaExecutor>(opt); }

}  // namespace ce
