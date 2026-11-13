#include "backends/cuda/executor.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <mutex>
#include <set>
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
      return c;
    }
    if (!s.kernels) fail(Errc::unsupported, "dtype conversion on CUDA needs the kernel module: " + s.kernel_error);
    if (src_cuda && dst_cuda) {
      if (!same_dev) fail(Errc::unsupported, "converting copies between two GPUs are not supported yet");
      return {Strategy::kernel};
    }
    return {Strategy::staged};
  }

  int64_t run(Strategy strat, const Group& g, const std::vector<Op>& ops, uintptr_t stream) override {
    const int di = device_of(g);
    Dev& s = state(di);
    Push push(*d_, s.ctx);
    pending_.insert(di);
    const CUstream st = reinterpret_cast<CUstream>(stream);
    switch (strat) {
      case Strategy::passthrough: return passthrough(s, g, ops, st);
      case Strategy::runs: return runs(g, st);
      case Strategy::batch: return batch(s, g.src, g.dst, cached_rows(g, Strategy::batch), st);
      case Strategy::kernel: return kernel(s, g, st);
      case Strategy::staged: return staged(s, g, st);
      case Strategy::bounce: return bounce(s, g.src, g.dst, cached_rows(g, Strategy::bounce), st);
      default: fail(Errc::unsupported, "CUDA backend cannot run strategy " + std::string(strategy_name(strat)));
    }
  }

  void sync(uintptr_t stream) override {
    for (int di : pending_) {
      Push push(*d_, devs_[di].ctx);
      check(d_->cuStreamSynchronize(reinterpret_cast<CUstream>(stream)), "cuStreamSynchronize");
    }
    pending_.clear();
  }

  bool async() const override { return true; }

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

  struct Push {
    const Driver& d;
    Push(const Driver& drv, CUcontext c) : d(drv) { check(d.cuCtxPushCurrent(c), "cuCtxPushCurrent"); }
    ~Push() {
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

  CUfunction function(Dev& s, const std::string& name) {
    auto it = s.fns.find(name);
    if (it != s.fns.end()) return it->second;
    CUfunction f{};
    check(d_->cuModuleGetFunction(&f, s.mod, name.c_str()), ("cuModuleGetFunction " + name).c_str());
    s.fns.emplace(name, f);
    return f;
  }

  // Uploads the chunk table through a pinned/device slot ring and launches.
  void launch(Dev& s, const std::string& name, const std::vector<Chunk>& chunks, const cv::Fn& fn, CUstream st) {
    if (chunks.empty()) return;
    CUfunction f = function(s, name);
    const size_t bytes = chunks.size() * sizeof(Chunk);
    for (size_t at = 0; at < chunks.size();) {
      const size_t n = std::min<size_t>(chunks.size() - at, size_t{1} << 24);
      Slot& sl = s.slots[s.next++ % kSlots];
      if (sl.used) check(d_->cuEventSynchronize(sl.done), "cuEventSynchronize");
      const size_t need = n * sizeof(Chunk);
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
      std::memcpy(sl.host, chunks.data() + at, need);
      check(d_->cuMemcpyHtoDAsync(sl.dev, sl.host, need, st), "cuMemcpyHtoDAsync");
      int count = static_cast<int>(n);
      CUdeviceptr table = sl.dev;
      cv::Fn fcopy = fn;
      void* args[] = {&table, &count, &fcopy};
      const unsigned threads = 256, warps = threads / 32;
      const unsigned blocks = static_cast<unsigned>((n + warps - 1) / warps);
      check(d_->cuLaunchKernel(f, blocks, 1, 1, threads, 1, 1, 0, st, args, nullptr), ("cuLaunchKernel " + name).c_str());
      check(d_->cuEventRecord(sl.done, st), "cuEventRecord");
      sl.used = true;
      at += n;
    }
    (void)bytes;
  }


  static std::string kernel_name(const Group& g) {
    if (g.pure()) return "ce_bytes";
    return std::string(g.fn.identity() ? "ce_cast_" : "ce_apply_") + short_name(g.src.dtype) + "_" + short_name(g.dst.dtype);
  }

  int64_t kernel(Dev& s, const Group& g, CUstream st) {
    auto& slot = g.exec_cache[static_cast<int>(Strategy::kernel)];
    if (!slot) {
      ChunkBuilder cb(g.pure() ? kByteChunk : kTypedChunk);
      for (size_t i = 0; i < g.runs.size(); ++i) cb.add(g.runs.src[i], g.runs.dst[i], g.runs.len[i]);
      slot = std::make_shared<std::vector<Chunk>>(cb.take());
    }
    launch(s, kernel_name(g), *static_cast<std::vector<Chunk>*>(slot.get()), to_cv(g.fn), st);
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
  };

  std::vector<StagedStep> build_staged(const Group& g) const {
    const bool h2d = g.dst.device.kind == DeviceKind::cuda;
    const DType lt = h2d ? g.src.dtype : g.dst.dtype;  // dtype crossing the link
    const i64 isz = static_cast<i64>(itemsize(lt));
    const int64_t budget = std::max<int64_t>(opt_.staging_budget, 1 << 20);
    const i64 per = std::max<i64>(1, budget / isz);

    // Link-side intervals (elements of lt), sorted.
    const std::vector<i64>& link = h2d ? g.runs.src : g.runs.dst;
    Intervals iv = interval_union(link, g.runs.len);
    if (h2d) {  // widen across small read gaps that stay inside one source allocation
      const Intervals& ext = g.src_extent;
      Intervals m;
      for (size_t k = 0; k < iv.start.size(); ++k) {
        if (!m.empty()) {
          const i64 gap_lo = m.end.back() * isz, gap_hi = iv.start[k] * isz;
          auto e = std::upper_bound(ext.start.begin(), ext.start.end(), gap_lo);
          const bool inside = e != ext.start.begin() && ext.end[(e - ext.start.begin()) - 1] >= gap_hi;
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
  int64_t staged(Dev& s, const Group& g, CUstream st) {
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
    std::vector<Chunk> ch;
    for (const StagedStep& step : steps) {
      const CUdeviceptr stg = staging(s, step.bytes, st);
      const long long base = static_cast<long long>(stg / isz);
      raws = step.raws;
      ch = step.chunks;
      for (Raw& r : raws) (h2d ? r.dst : r.src) += stg;
      for (Chunk& c : ch) (h2d ? c.src : c.dst) += base;
      if (h2d) {
        copy_raws(s, g.src, dev_side, raws, st);
        launch(s, name, ch, fn, st);
      } else {
        launch(s, name, ch, fn, st);
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
        for (size_t i = 0; i < op.runs.size(); ++i, ++calls)
          copy1d(op.src, op.dst, static_cast<uintptr_t>(op.runs.src[i]), static_cast<uintptr_t>(op.runs.dst[i]),
                 static_cast<size_t>(op.runs.len[i]), st);
        continue;
      }
      Group one;
      one.src = op.src, one.dst = op.dst, one.fn = op.fn;
      one.runs = op.runs;
      one.rects = recover_2d(op.runs, 1 << 30);  // rows only
      calls += (op.src.device.kind == DeviceKind::cuda && op.dst.device.kind == DeviceKind::cuda) ? kernel(s, one, st)
                                                                                                    : staged(s, one, st);
    }
    return calls;
  }

  EngineOptions opt_;
  const Driver* d_ = nullptr;
  std::vector<Dev> devs_;
  std::set<int> pending_;
};

}  // namespace

std::unique_ptr<Executor> make_cuda_executor(const EngineOptions& opt) { return std::make_unique<CudaExecutor>(opt); }

}  // namespace ce
