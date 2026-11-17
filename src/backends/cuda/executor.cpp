#include "backends/cuda/executor.hpp"

#include <algorithm>
#include <numeric>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "backends/cuda/abi.hpp"
#include "backends/cuda/driver.hpp"
#include "copyengine/frontend.hpp"
#include "backends/host/host.hpp"
#include "copyengine/error.hpp"

#if CE_CUDA_KERNELS
extern "C" const unsigned char ce_kernels_fatbin[];
#endif

namespace ce {
namespace {

using cuda::Driver;

// The structures the kernels read (abi.hpp: one definition for both sides).
using cuabi::Base;
using cuabi::Chunk;
using cuabi::StmtDesc;

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


// Magic number and shift such that n / d == (umulhi(n, magic) + n) >> shift
// for every n < 2^31 (the divider PyTorch's IntDivider uses).
void int_divider(unsigned d, unsigned* magic, unsigned* shift) {
  unsigned s = 0;
  while (s < 32 && (uint64_t{1} << s) < d) ++s;
  *shift = s;
  *magic = static_cast<unsigned>(((uint64_t{1} << 32) * ((uint64_t{1} << s) - d)) / d + 1);
}


struct BaseEntry {
  std::shared_ptr<void> table;    // DevTable holding the bases
  std::vector<unsigned long long> ptrs;  // the table pointers they were written with
  CUstream stream{};
  uint64_t used = 0;
};

struct FusedBatch {
  StmtDesc desc{};
  int tile_a = -1, tile_b = -1;  // destination / source contiguous dims when they differ (tiled kernel)
  i64 link_size = 1;  // itemsize of the side that lives in staging (cross-device only)
  std::vector<Base> bases;
  std::shared_ptr<void> table;  // the bases, resident on the device (set on first run)
  // Device index tables: their pointers change per call, so the bases go to
  // the device each call; members[i]: position in the group of bases[i].
  bool device_tables = false;
  // Tables in host memory (step plan parameters): each call packs their values
  // into a device mirror (per stream) the kernel reads with step 1.
  bool stab_host = false, dtab_host = false;
  struct Mirror {
    CUstream stream{};
    std::shared_ptr<void> table;  // DevTable
    size_t words = 0;
  };
  std::vector<Mirror> mirrors;
  std::vector<int> members;
  std::vector<BaseEntry> recent;                // eager: bases of recent pointer sets (see current_bases)
  uint64_t clock = 0;
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

// An NVTX range per group, named after its strategy ("copyengine:fused"),
// visible in Nsight Systems; free when no tool is attached. NVTX is header
// only: without its headers the range does nothing.
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
struct NvtxRange {
  explicit NvtxRange(Strategy s) {
    static const auto names = [] {
      std::array<std::string, kNumStrategies> n;
      for (int i = 0; i < kNumStrategies; ++i) n[i] = "copyengine:" + std::string(strategy_name(static_cast<Strategy>(i)));
      return n;
    }();
    nvtxRangePushA(names[static_cast<int>(s)].c_str());
  }
  ~NvtxRange() { nvtxRangePop(); }
};
#else
struct NvtxRange {
  explicit NvtxRange(Strategy) {}
};
#endif

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
    case DType::f8e4m3: return "f8e4m3";
    case DType::f8e5m2: return "f8e5m2";
    case DType::f8e8m0: return "f8e8m0";
    case DType::f4x2: return "f4x2";
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
      if (s.light_ctx) d_->cuDevicePrimaryCtxRelease(s.light_dev);
      if (!s.init) continue;
      d_->cuCtxPushCurrent(s.ctx);
      for (Slot& sl : s.slots) {
        if (sl.used) d_->cuEventSynchronize(sl.done);
        pinned_free(sl.host, sl.host_handle);
        dev_free(s, sl.dev);
        if (sl.done) d_->cuEventDestroy(sl.done);
      }
      // Device memory first: freeing orders it after every stream, so they must still exist.
      for (Staging& g : s.stagings) {
        if (g.used) d_->cuEventSynchronize(g.done);
        dev_free(s, g.mem);
      }
      dev_free(s, s.err);
      dev_free(s, s.capture_arena);
      dev_free(s, s.pace);
      for (Lane& l : s.lanes) {  // lanes[0] is sm_stream
        d_->cuStreamSynchronize(l.sm);
        d_->cuStreamDestroy(l.sm);
        d_->cuEventDestroy(l.in);
        d_->cuEventDestroy(l.out);
      }
      if (s.green) d_->cuGreenCtxDestroy(s.green);
      for (Bounce& b : s.bounce) {
        if (b.used) d_->cuEventSynchronize(b.done);
        pinned_free(b.host, b.host_handle);
        if (b.done) d_->cuEventDestroy(b.done);
      }
      if (s.ev_free) d_->cuEventDestroy(s.ev_free);
      if (s.ev_alloc) d_->cuEventDestroy(s.ev_alloc);
      for (Staging& g : s.stagings) {
        if (g.done) d_->cuEventDestroy(g.done);
        if (g.aux) {
          d_->cuStreamSynchronize(g.aux);
          d_->cuStreamDestroy(g.aux);
          for (CUevent e : {g.fork, g.join, g.in, g.kern, g.out[0], g.out[1]}) d_->cuEventDestroy(e);
        }
      }
      for (CUevent e : {s.ev_in, s.ev_out})
        if (e) d_->cuEventDestroy(e);
      if (s.side) d_->cuStreamDestroy(s.side);
      CUcontext dummy{};
      d_->cuCtxPopCurrent(&dummy);
      d_->cuDevicePrimaryCtxRelease(s.dev);
    }
  }

  // What each strategy needs of the memory it touches, in one place (the
  // branches of offered() choose; this only refuses). Kernels and batched DMA
  // take device-accessible addresses: host memory must be mapped at its own;
  // the device-only kernels need both sides on one GPU.
  static bool admissible(Strategy s, const Group& g) {
    const bool sc = g.src.device.kind == DeviceKind::cuda, dc = g.dst.device.kind == DeviceKind::cuda;
    const bool one_gpu = sc && dc && g.src.device.index == g.dst.device.index;
    const bool reach = kernel_reach(g.src) && kernel_reach(g.dst);
    switch (s) {
      case Strategy::kernel:
      case Strategy::bulk:
      case Strategy::tma2d: return one_gpu;
      case Strategy::batch: return reach;
      case Strategy::fused: return one_gpu || !g.opaque() || reach;  // non-opaque host sides go through staging
      case Strategy::tile: return !g.tile_host_sides || g.tile_mapped;
      default: return true;  // DMA through the driver (any pinned or pageable host memory), or staging
    }
  }

  std::vector<Strategy> candidates(const Group& g) override {
    std::vector<Strategy> c = offered(g);
    std::erase_if(c, [&](Strategy s) { return !admissible(s, g); });
    if (c.empty()) fail(Errc::unsupported, "no strategy can reach this memory (see stats()[\"cuda\"]: zero_copy, batch)");
    return c;
  }

  std::vector<Strategy> offered(const Group& g) {
    const int di = device_of(g);
    Dev& s = state(di);
    if (g.tiles) {  // quantization while copying: the tile kernel on this GPU
      if (!s.kernels) fail(Errc::unsupported, "quantization on the GPU needs the copy kernels: " + s.kernel_error);
      if (!g.tile_host_sides) return {Strategy::tile};
      // Host sides: the kernel reading / writing pinned memory directly, or
      // through device staging (DMA in, kernel, DMA out); measured.
      std::vector<Strategy> c;
      if (g.tile_mapped && zero_copy_ok(s)) c.push_back(Strategy::tile);
      if (g.tile_staged_ok) c.push_back(Strategy::tile_staged);
      if (c.empty())
        fail(Errc::unsupported, "quantizing to or from host memory needs zero-copy kernels (" +
                                    (g.tile_mapped ? s.zero_copy_error
                                                   : std::string("host memory not mapped for kernels at its address")) +
                                    ") or compact host tensors for staging");
      return c;
    }
    if (g.opaque()) {  // indices read at run time: the statement kernel, or (host tables) DMA expanded per run
      std::vector<Strategy> c;
      if (g.pure() && host_tables_only(g)) c.push_back(Strategy::lowered);
      const bool sc = g.src.device.kind == DeviceKind::cuda, dc = g.dst.device.kind == DeviceKind::cuda;
      const bool reach = kernel_reach(g.src) && kernel_reach(g.dst);  // the statement kernel touches host memory itself
      if (s.kernels && fused_plan(g).ok && ((sc && dc) || (reach && zero_copy_ok(s)))) c.push_back(Strategy::fused);
      if (c.empty())
        fail(Errc::unsupported, "copies with indices read at run time need the statement kernel (" +
                                    (s.kernels ? std::string("this layout or memory is not supported") : s.kernel_error) +
                                    ") or, for host index tables, a pure copy");
      return c;
    }
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
        // Large groups: first among the candidates, so the shortlist keeps it (measured like the rest).
        if (!s.bulk.empty() && g.bytes() >= kBulkMinBytes) c.insert(c.begin(), Strategy::bulk);
        if (s.tma2d && g.bytes() >= kBulkMinBytes && tma2d_fits(g)) c.insert(c.begin(), Strategy::tma2d);
      } else if (src_cuda && dst_cuda) {  // two GPUs: batch is admitted per device, not per pair yet (P2P)
        c.push_back(Strategy::runs);
        if (rects <= 2) std::reverse(c.begin(), c.end());
      } else {
        const bool host_ok = (src_cuda || g.src.pinned) && (dst_cuda || g.dst.pinned);
        int64_t rows = 0;
        for (i64 h : g.rects.height) rows += h;
        const bool fragmented = rows > 2 && g.bytes() / rows < kSmallRect;  // small rows DMA poorly
        if (fragmented && s.kernels) c.push_back(Strategy::staged);  // dense link copy + device-side kernel
        // cuMemcpyBatchAsync takes device-accessible addresses: host memory must be mapped at its own.
        if (s.batch && kernel_reach(g.src) && kernel_reach(g.dst)) c.push_back(Strategy::batch);
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

  bool capturing(uintptr_t stream) override {
    if (stream == 0) return false;  // the legacy stream cannot be captured
    CUstreamCaptureStatus st = CU_STREAM_CAPTURE_STATUS_NONE;
    check(d_->cuStreamIsCapturing(reinterpret_cast<CUstream>(stream), &st), "cuStreamIsCapturing");
    return st != CU_STREAM_CAPTURE_STATUS_NONE;
  }

  bool capture_safe(Strategy strat, const Group& g) override {
    const int di = device_of(g);
    if (di < 0 || di >= static_cast<int>(devs_.size()) || !devs_[di].init) return false;  // first use sets up the device
    const bool src_ok = g.src.device.kind == DeviceKind::cuda || g.src.pinned;
    const bool dst_ok = g.dst.device.kind == DeviceKind::cuda || g.dst.pinned;
    const bool on_device = g.src.device.kind == DeviceKind::cuda && g.dst.device.kind == DeviceKind::cuda &&
                           g.src.device.index == g.dst.device.index;
    switch (strat) {
      case Strategy::runs: return !g.opaque() && g.pure() && src_ok && dst_ok;  // plain DMA calls
      case Strategy::passthrough: return !g.opaque() && g.pure() && src_ok && dst_ok;
      case Strategy::kernel: return on_device && g.exec_cache[static_cast<int>(Strategy::kernel)] != nullptr;
      case Strategy::bulk: return on_device && g.exec_cache[static_cast<int>(Strategy::bulk)] != nullptr;
      case Strategy::tma2d: return on_device && g.exec_cache[static_cast<int>(Strategy::tma2d)] != nullptr;
      case Strategy::tile: {  // descriptors resident; indexed ones are written per call (allocation): not yet
        const auto& slot = g.exec_cache[static_cast<int>(Strategy::tile)];
        return on_device && slot && static_cast<const TileExec*>(slot.get())->table != nullptr;
      }
      case Strategy::fused: {
        if (!on_device || !devs_[di].kernels) return false;
        if (g.opaque()) fused_plan(g);  // host-only planning: fine under capture
        const auto& slot = g.exec_cache[static_cast<int>(Strategy::fused)];
        if (!slot) return false;
        const auto& P = *static_cast<const FusedPlan*>(slot.get());
        // Resident tables are built on first use (a host wait); device-table
        // bases are written by kernel nodes of the capture itself.
        if (g.opaque() && std::any_of(g.forms.begin(), g.forms.end(), [](const auto& f) { return f->host_tables(); }))
          return false;  // host index values would be baked into the graph
        const Dev& s = devs_[di];
        size_t need = 0;
        for (const FusedBatch& b : P.batches)
          if (b.device_tables) need += (b.bases.size() * sizeof(Base) + 255) & ~size_t{255};
        return P.ok && !P.cross && s.err.ptr != 0 /* made at device setup */ &&
               s.capture_used + need <= s.capture_arena_bytes &&
               std::all_of(P.batches.begin(), P.batches.end(), [](const FusedBatch& b) {
                 return b.device_tables || b.table != nullptr;
               });
      }
      case Strategy::paced: {
        const auto& slot = g.exec_cache[static_cast<int>(Strategy::paced)];
        if (!slot || !devs_[di].pace.ptr) return false;
        const auto& P = *static_cast<const PacedPlan*>(slot.get());
        return std::all_of(P.chunks.begin(), P.chunks.end(), [](const PacedChunk& c) {
          if (c.inner == Strategy::runs || c.inner == Strategy::batch) return true;  // batch records as runs
          return c.inner == Strategy::kernel && c.g.exec_cache[static_cast<int>(Strategy::kernel)] != nullptr;
        });
      }
      default: return false;  // staging, bounce buffers, batch descriptors: allocate or wait on the host
    }
  }

  int64_t run(Strategy strat, const Group& g, const std::vector<Op>& ops, uintptr_t stream) override {
    const NvtxRange range(strat);
    const int di = device_of(g);
    Dev& s = state(di);
    note_stream(s, stream);
    Push push(*d_, s.ctx);
    const CUstream st = reinterpret_cast<CUstream>(stream);
    switch (strat) {
      case Strategy::passthrough: return passthrough(s, g, ops, st);
      case Strategy::runs: return runs(g, st);
      case Strategy::batch: return batch(s, g.src, g.dst, cached_rows(g, Strategy::batch), st);
      case Strategy::kernel: return kernel(s, g, st);
      case Strategy::staged: return staged(s, g, st);
      case Strategy::bounce: return bounce(s, g.src, g.dst, cached_rows(g, Strategy::bounce), st);
      case Strategy::fused: return fused(s, g, ops, st);
      case Strategy::paced: return paced(s, g, st);
      case Strategy::lowered: return lowered(s, g, ops, st);
      case Strategy::tile: return tile(s, g, ops, st);
      case Strategy::tile_staged: return tile_staged(s, g, ops, st);
      case Strategy::bulk: return bulk(s, g, st);
      case Strategy::tma2d: return tma2d(s, g, st);
      default: fail(Errc::unsupported, "CUDA backend cannot run strategy " + std::string(strategy_name(strat)));
    }
  }

  // A device's primary context without setting the device up (retained once).
  CUcontext primary_ctx(int dev) {
    Dev& s = devs_[dev];
    if (s.init) return s.ctx;
    if (!s.light_ctx) {
      check(d_->cuDeviceGet(&s.light_dev, dev), "cuDeviceGet");
      check(d_->cuDevicePrimaryCtxRetain(&s.light_ctx, s.light_dev), "cuDevicePrimaryCtxRetain");
    }
    return s.light_ctx;
  }

  // Each stream on the context of its device; stream 0 is that device's legacy stream.
  void sync(std::span<const std::pair<uintptr_t, int>> streams) override {
    for (const auto& [st, dev] : streams) {
      // The device's primary context (torch's too), set up here or not: work
      // left to the framework never set the device up, and must be waited for.
      if (dev < 0 || dev >= static_cast<int>(devs_.size())) continue;
      Push push(*d_, primary_ctx(dev));
      check(d_->cuStreamSynchronize(reinterpret_cast<CUstream>(st)), "cuStreamSynchronize");
    }
    for (Dev& s : devs_) {  // the work is done: report device-index errors it found
      if (!s.init || !(s.err_armed || s.err_graphs)) continue;
      Push push(*d_, s.ctx);
      s.err_armed = false;
      unsigned long long flag = 0;
      check(d_->cuMemcpyDtoHAsync(&flag, s.err.ptr, 8, s.side), "cuMemcpyDtoHAsync");
      check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
      if (flag) {
        check(d_->cuMemsetD8Async(s.err.ptr, 0, 8, s.side), "cuMemsetD8Async");
        check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
        fail(Errc::invalid_argument, "a device index table held an index out of range (those elements were skipped)");
      }
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

  bool ready(int64_t m) override {
    if (m < 0) return true;
    Push push(*d_, devs_[marks_[m].dev].ctx);
    const CUresult q = d_->cuEventQuery(marks_[m].ev);
    if (q == CUDA_ERROR_NOT_READY) return false;
    check(q, "cuEventQuery");
    return true;
  }

  Json info() override {
    Json devs = Json::array();
    for (size_t i = 0; i < devs_.size(); ++i) {
      Dev& s = state(static_cast<int>(i));
      devs.push(Json::object()
                    .set("index", static_cast<int>(i))
                    .set("kernels", s.kernels ? Json(true) : Json(s.kernel_error))
                    .set("batch", s.batch ? Json(true) : Json(s.batch_error))
                    .set("sm_limit", s.sm_limit)
                    .set("bulk", s.bulk.empty() ? Json(s.bulk_error) : Json(s.bulk))
                    .set("tma2d", s.tma2d ? Json(true) : Json(s.tma2d_error))
                    .set("zero_copy", s.zero_copy < 0 ? Json("unchecked")
                                                      : (s.zero_copy ? Json(true) : Json(s.zero_copy_error)))
                    .set("allocator", device_allocator() ? device_allocator()->name : "cuda")
                    .set("host_allocator",
                         device_allocator() && device_allocator()->host_alloc ? device_allocator()->name : "cuda"));
    }
    return devs;
  }

 private:
  // Device memory of copyengine's own: from the framework's allocator when one
  // is registered (its handle, and the stream that allocator ties it to), else
  // from cuMemAlloc.
  struct DevMem {
    CUdeviceptr ptr = 0;
    void* handle = nullptr;
    uintptr_t stream = 0;
  };

  struct Slot {
    void* host = nullptr;
    void* host_handle = nullptr;  // from the framework's pinned allocator
    DevMem dev;
    size_t cap = 0;
    CUevent done = nullptr;
    bool used = false;
  };

  struct Bounce {
    void* host = nullptr;
    void* host_handle = nullptr;
    size_t cap = 0;
    CUevent done = nullptr;
    bool used = false;
    bool mapped = false;  // host_memory(host): the allocator decides (a framework's may register memory)
  };

  struct Lane {
    CUstream caller{};  // the stream it serves (null: not yet assigned)
    CUstream sm{};
    CUevent in{}, out{};
  };

  struct Staging {
    CUstream caller{};
    DevMem mem;
    size_t cap = 0;
    CUevent done{};
    bool used = false;
    uint64_t last = 0;
    // Chunked tile staging: DMA on `aux` while kernels run on the caller's
    // stream (joined back before the call returns). Created on first use.
    CUstream aux{};
    CUevent fork{}, join{}, in{}, kern{}, out[2]{};
  };

  struct Dev {
    bool init = false;
    int index = 0;
    CUdevice dev{};
    CUcontext ctx{};
    CUmodule mod{};
    bool kernels = false, batch = false;
    std::string kernel_error = "not built", batch_error;
    // Bulk copies through shared memory: "tma" (sm90+), "cp.async" (sm80-89),
    // admitted by a verified test copy; empty: not available (bulk_error says why).
    std::string bulk, bulk_error;
    int bulk_seg = 8192;  // bytes per segment: kBulkBufs of them per block (shared memory)
    // 2-D blocks as tensor maps (sm90+), admitted by a verified test copy.
    bool tma2d = false;
    std::string tma2d_error = "not admitted";
    std::unordered_map<std::string, CUfunction> fns;
    CUstream side{};
    CUevent ev_in{}, ev_out{};
    std::array<Slot, kSlots> slots;
    int next = 0;
    // Device staging, one buffer per caller stream (kLanes, then shared):
    // copies on one stream never wait for another stream's use of it.
    std::vector<Staging> stagings;
    uint64_t staging_clock = 0;
    std::array<Bounce, 2> bounce;
    // SM limit (EngineOptions::max_sms)
    CUgreenCtx green{};
    CUstream sm_stream{};       // set when the limit is hard: the verified green-context stream (lanes[0])
    CUevent sm_in{}, sm_out{};
    // Copy kernels run on a green-context lane per caller stream (kLanes,
    // then shared), so one stream's copies never queue behind another's.
    std::vector<Lane> lanes;
    unsigned grid_cap = 0;      // soft limit: max thread blocks per launch (0: none)
    std::string sm_limit = "none";
    // Streams this device's work ran on, for ordering a framework allocator's
    // reuse of freed memory after it (kMaxStreams, then: the whole device).
    std::vector<uintptr_t> streams;
    bool many_streams = false;
    CUevent ev_free{};
    CUevent ev_alloc{};  // dev_alloc: orders a first use after the framework's allocation stream
    DevMem err;               // set by kernels that meet an out-of-range device index
    bool err_armed = false;   // such kernels ran since the last check
    bool err_graphs = false;  // ... or were captured: graph replays may set it at any time
    CUcontext light_ctx{};    // primary context retained only to synchronize (device not set up)
    CUdevice light_dev{};
    DevMem capture_arena;     // bases recorded by CUDA graph captures (see captured_bases)
    size_t capture_arena_bytes = 0, capture_used = 0;
    // Bandwidth limit: the pacing words (the next start time per direction,
    // on the device), and whether kernels may read and write pinned host
    // memory directly (zero-copy), checked by a test copy.
    DevMem pace;
    int zero_copy = -1;  // -1: not checked yet
    std::string zero_copy_error;
  };

  CUdeviceptr error_word(Dev& s) {
    if (!s.err.ptr) {
      s.err = dev_alloc(s, 8, "error word", s.side);
      check(d_->cuMemsetD8Async(s.err.ptr, 0, 8, s.side), "cuMemsetD8Async");
      check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
    }
    return s.err.ptr;
  }
  static constexpr size_t kMaxStreams = 32;
  static constexpr size_t kLanes = 8;

  // Device memory for copyengine itself. From the framework's allocator it
  // is ordered on m.stream only: the block may have been freed by a tensor
  // whose kernels on m.stream still read or write it. So `use`, the stream
  // that touches it first, waits for m.stream first (null: the host waits;
  // setup only). The mirror of dev_free's ordering.
  DevMem dev_alloc(Dev& s, size_t bytes, const char* what, CUstream use = nullptr) {
    if (const DeviceAllocator* a = device_allocator()) {
      DevMem m;
      m.ptr = static_cast<CUdeviceptr>(a->alloc(s.index, bytes, &m.handle, &m.stream));
      if (!m.ptr) fail(Errc::backend, std::string(what) + ": out of memory (" + a->name + " allocator, " +
                                          std::to_string(bytes) + " bytes)");
      if (reinterpret_cast<uintptr_t>(use) != m.stream || !use) {
        if (!s.ev_alloc) check(d_->cuEventCreate(&s.ev_alloc, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
        check(d_->cuEventRecord(s.ev_alloc, reinterpret_cast<CUstream>(m.stream)), "cuEventRecord");
        if (use) check(d_->cuStreamWaitEvent(use, s.ev_alloc, 0), "cuStreamWaitEvent");
        else check(d_->cuEventSynchronize(s.ev_alloc), "cuEventSynchronize");
      }
      return m;
    }
    DevMem m;
    check(d_->cuMemAlloc(&m.ptr, bytes), what);
    return m;
  }

  // Releases `m`. Memory from the framework's allocator is reused only by later
  // allocations on m.stream, so that stream first waits for every stream this
  // device's copies used: nothing reuses it before they are done. Never blocks
  // the host. Plain cuMemFree waits for the device itself.
  void dev_free(Dev& s, DevMem& m) {
    if (!m.ptr) return;
    if (m.handle) {
      const DeviceAllocator* a = device_allocator();
      if (s.many_streams) {
        d_->cuCtxSynchronize();
      } else {
        if (!s.ev_free) d_->cuEventCreate(&s.ev_free, CU_EVENT_DISABLE_TIMING);
        std::vector<uintptr_t> all = s.streams;
        all.push_back(reinterpret_cast<uintptr_t>(s.side));
        for (const Lane& l : s.lanes) all.push_back(reinterpret_cast<uintptr_t>(l.sm));
        for (const Staging& g : s.stagings) {
          all.push_back(reinterpret_cast<uintptr_t>(g.caller));
          if (g.aux) all.push_back(reinterpret_cast<uintptr_t>(g.aux));
        }
        for (uintptr_t x : all) {
          if (x == m.stream || !s.ev_free) continue;
          d_->cuEventRecord(s.ev_free, reinterpret_cast<CUstream>(x));
          d_->cuStreamWaitEvent(reinterpret_cast<CUstream>(m.stream), s.ev_free, 0);
        }
      }
      if (a) a->free(m.handle);
    } else {
      d_->cuMemFree(m.ptr);
    }
    m = DevMem{};
  }

  // Pinned host memory for copyengine's own buffers: from the framework's
  // allocator when it has one. Freed only once the work using it is done.
  void* pinned_alloc(size_t bytes, void** handle, const char* what) {
    *handle = nullptr;
    if (const DeviceAllocator* a = device_allocator(); a && a->host_alloc) {
      void* p = a->host_alloc(bytes, handle);
      if (!p) fail(Errc::backend, std::string(what) + ": out of pinned memory (" + a->name + " allocator, " +
                                      std::to_string(bytes) + " bytes)");
      return p;
    }
    void* p = nullptr;
    check(d_->cuMemHostAlloc(&p, bytes, 0), what);
    return p;
  }

  void pinned_free(void*& p, void*& handle) {
    if (!p) return;
    if (handle) {
      if (const DeviceAllocator* a = device_allocator(); a && a->host_free) a->host_free(handle);
    } else {
      d_->cuMemFreeHost(p);
    }
    p = nullptr, handle = nullptr;
  }

  void note_stream(Dev& s, uintptr_t st) {
    if (s.many_streams || std::find(s.streams.begin(), s.streams.end(), st) != s.streams.end()) return;
    if (s.streams.size() == kMaxStreams) s.many_streams = true;
    else s.streams.push_back(st);
  }

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
    s.stagings.reserve(kLanes);
    for (Slot& sl : s.slots) check(d_->cuEventCreate(&sl.done, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
    {  // once per device and process: the module and the verdicts of the test copies
      Admitted& a = admitted(di);
      std::lock_guard<std::mutex> lock(a.m);
      if (!a.done) {
        admit_kernels(s);
        admit_batch(s);
        admit_bulk(s);
        admit_tma2d(s);
        a.done = true;
        a.tma2d = s.tma2d, a.tma2d_error = s.tma2d_error;
        a.mod = s.mod, a.kernels = s.kernels, a.kernel_error = s.kernel_error;
        a.batch = s.batch, a.batch_error = s.batch_error;
        a.bulk = s.bulk, a.bulk_error = s.bulk_error, a.bulk_seg = s.bulk_seg;
      } else {
        s.mod = a.mod, s.kernels = a.kernels, s.kernel_error = a.kernel_error;
        s.batch = a.batch, s.batch_error = a.batch_error;
        s.bulk = a.bulk, s.bulk_error = a.bulk_error, s.bulk_seg = a.bulk_seg;
        s.tma2d = a.tma2d, s.tma2d_error = a.tma2d_error;
      }
    }
    if (s.kernels) {  // before any capture can need them
      error_word(s);
      s.capture_arena = dev_alloc(s, kCaptureArena, "capture tables");
      s.capture_arena_bytes = kCaptureArena;
    }
    if (opt_.max_sms > 0 && s.kernels) setup_sm_limit(s);
    return s;
  }

  // ---- admission: call once, verify bit-exact ------------------------------------

  // Admission is per device and process: engines share the primary context,
  // so they share the kernels' module (loaded once, kept for the process) and
  // the verdicts of the test copies.
  struct Admitted {
    std::mutex m;
    bool done = false;
    CUmodule mod{};
    bool kernels = false, batch = false;
    std::string kernel_error, batch_error, bulk, bulk_error;
    int bulk_seg = 8192;
    bool tma2d = false;
    std::string tma2d_error;
  };
  static Admitted& admitted(int device) {
    static std::mutex m;
    static std::map<int, std::unique_ptr<Admitted>> all;
    std::lock_guard<std::mutex> lock(m);
    auto& a = all[device];
    if (!a) a = std::make_unique<Admitted>();
    return *a;
  }

  void admit_kernels(Dev& s) {
#if CE_CUDA_KERNELS
    CUresult r = d_->cuModuleLoadData(&s.mod, ce_kernels_fatbin);
    if (r != CUDA_SUCCESS) {
      s.kernel_error = "cuModuleLoadData: " + cuda::error_name(r) + " (no SASS/PTX for this GPU?)";
      return;
    }
    s.kernels = true;  // provisionally, so the verification can launch
    try {
      {  // the tables' layout as the kernels see it (abi.hpp): else nothing they read is right
        CUdeviceptr out = 0;
        check(d_->cuMemAlloc(&out, 8), "cuMemAlloc");
        CUfunction f = function(s, "ce_layout");
        void* args[] = {&out};
        unsigned long long got = 0;
        check(d_->cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, s.side, args, nullptr), "cuLaunchKernel ce_layout");
        check(d_->cuMemcpyDtoHAsync(&got, out, 8, s.side), "cuMemcpyDtoHAsync");
        check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
        d_->cuMemFree(out);
        if (got != cuabi::layout_fingerprint()) {
          s.kernels = false;
          s.kernel_error = "kernel admission: the kernels' table layout differs from the executor's (abi.hpp)";
          return;
        }
      }
      {  // the float8_e4m3fn overflow mode of the installed torch (convert.hpp)
        CUdeviceptr g = 0;
        size_t sz = 0;
        check(d_->cuModuleGetGlobal(&g, &sz, s.mod, "ce_f8e4m3_saturate"), "cuModuleGetGlobal");
        const int sat = cv::f8e4m3_saturate;
        check(d_->cuMemcpyHtoDAsync(g, &sat, sizeof(sat), s.side), "cuMemcpyHtoDAsync");
        check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");  // admission: `sat` is on the stack
      }
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

  void admit_bulk(Dev& s) {
    if (!s.kernels) {
      s.bulk_error = "no kernels";
      return;
    }
    int major = 0;
    check(d_->cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, s.dev), "cuDeviceGetAttribute");
    if (major < 8) {
      s.bulk_error = "needs sm80 or later";
      return;
    }
    const std::string kind = major >= 9 ? "tma" : "cp.async";
    // Segments small enough that kBulkBlocksPerSm blocks fit in an SM's
    // shared memory: under max_sms, the bytes in flight come from the blocks
    // that fit on the few SMs.
    int smem = 0;
    check(d_->cuDeviceGetAttribute(&smem, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, s.dev),
          "cuDeviceGetAttribute");
    s.bulk_seg = 8192;
    while (s.bulk_seg > 1024 && (s.bulk_seg * kBulkBufs + 1024) * kBulkBlocksPerSm > smem) s.bulk_seg /= 2;
    if (const char* e = std::getenv("CE_BULK_SEG")) s.bulk_seg = std::atoi(e);  // diagnostics: sweep on new hardware
    try {  // rows of several segments, 16-byte aligned, with pitches: verified byte for byte
      const size_t rows = 5, width = 3 * 16384 + 4096, spitch = width + 256, dpitch = width + 512;
      const size_t sbytes = rows * spitch, dbytes = rows * dpitch;
      CUdeviceptr a = 0, b = 0;
      check(d_->cuMemAlloc(&a, sbytes), "cuMemAlloc");
      check(d_->cuMemAlloc(&b, dbytes), "cuMemAlloc");
      std::vector<uint8_t> in(sbytes), out(dbytes, 0);
      for (size_t i = 0; i < sbytes; ++i) in[i] = static_cast<uint8_t>(i * 131 + 7);
      check(d_->cuMemcpyHtoDAsync(a, in.data(), sbytes, s.side), "cuMemcpyHtoDAsync");
      check(d_->cuMemsetD8Async(b, 0, dbytes, s.side), "cuMemsetD8Async");
      std::vector<Chunk> ch{{static_cast<long long>(a), static_cast<long long>(b), static_cast<long long>(width),
                             static_cast<long long>(rows), static_cast<long long>(spitch),
                             static_cast<long long>(dpitch)}};
      Slot& sl = upload(s, ch.data(), ch.size() * sizeof(Chunk), s.side);
      CUfunction f = function(s, "ce_bytes_bulk");
      CUdeviceptr t = sl.dev.ptr;
      int n = 1;
      long long zero = 0;
      int seg = s.bulk_seg;
      void* args[] = {&t, &n, &zero, &zero, &seg};
      check(d_->cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, kBulkBufs * seg, s.side, args, nullptr),
            "cuLaunchKernel ce_bytes_bulk");
      check(d_->cuEventRecord(sl.done, s.side), "cuEventRecord");
      sl.used = true;
      check(d_->cuMemcpyDtoHAsync(out.data(), b, dbytes, s.side), "cuMemcpyDtoHAsync");
      check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
      d_->cuMemFree(a);
      d_->cuMemFree(b);
      bool ok = true;
      for (size_t r = 0; r < rows && ok; ++r)
        ok = std::memcmp(out.data() + r * dpitch, in.data() + r * spitch, width) == 0;
      if (ok) s.bulk = kind;
      else s.bulk_error = kind + " test copy came back wrong";
    } catch (const std::exception& e) {
      s.bulk_error = kind + ": " + e.what();
    }
  }

  void admit_batch(Dev& s) {
    if (!d_->cuMemcpyBatchAsync) {
      s.batch_error = "cuMemcpyBatchAsync: " + cuda::missing_reason("cuMemcpyBatchAsync");
      return;
    }
    s.batch = true;
    try {
      // Every direction batch is offered for, each its own test copy: H2D into
      // reversed padded slots, D2D (same device) reversing them again, D2H back
      // into host slots; then the bytes and the untouched padding are checked.
      const size_t segs = 16, len = 1000, pitch = 1024;
      void* h = nullptr;
      check(d_->cuMemHostAlloc(&h, segs * len + segs * pitch, 0), "cuMemHostAlloc");
      CUdeviceptr dp = 0, dq = 0;
      check(d_->cuMemAlloc(&dp, segs * pitch), "cuMemAlloc");
      check(d_->cuMemAlloc(&dq, segs * pitch), "cuMemAlloc");
      auto* src = static_cast<uint8_t*>(h);
      auto* back = src + segs * len;
      for (size_t i = 0; i < segs * len; ++i) src[i] = static_cast<uint8_t>(i * 131 + 7);
      std::memset(back, 0, segs * pitch);
      check(d_->cuMemsetD8Async(dp, 0, segs * pitch, s.side), "cuMemsetD8Async");
      check(d_->cuMemsetD8Async(dq, 0, segs * pitch, s.side), "cuMemsetD8Async");
      const Side hs{{DeviceKind::host, 0}, DType::u8, true, true}, ds{{DeviceKind::cuda, s.index}, DType::u8, false};
      std::vector<Raw> h2d, d2d, d2h;
      for (size_t i = 0; i < segs; ++i) {
        const size_t r = segs - 1 - i;
        h2d.push_back({reinterpret_cast<uintptr_t>(src + i * len), dp + r * pitch, len});  // slot r: segment i
        d2d.push_back({dp + r * pitch, dq + i * pitch, len});                              // slot i: segment i
        d2h.push_back({dq + i * pitch, reinterpret_cast<uintptr_t>(back + r * pitch), len});  // host slot r
      }
      batch(s, hs, ds, h2d, s.side);
      batch(s, ds, ds, d2d, s.side);
      batch(s, ds, hs, d2h, s.side);
      check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
      bool ok = true;
      for (size_t i = 0; i < segs && ok; ++i) {
        const size_t r = segs - 1 - i;
        ok = std::memcmp(back + r * pitch, src + i * len, len) == 0;
        for (size_t k = len; k < pitch && ok; ++k) ok = back[r * pitch + k] == 0;
      }
      d_->cuMemFree(dp);
      d_->cuMemFree(dq);
      d_->cuMemFreeHost(h);
      if (!ok) s.batch = false, s.batch_error = "batch admission: wrong bytes (H2D, D2D or D2H)";
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
          cuda::memcpy_batch(*d_, ds.data(), ss.data(), sizes.data(), n, &attr, &idx, 1, &fail_idx, run);
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
    if (s.batch && kernel_reach(src) && kernel_reach(dst) && raws.size() > 2) {  // device-accessible addresses
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
      pinned_free(b.host, b.host_handle);
      b.host = pinned_alloc(cap, &b.host_handle, "bounce buffer");
      if (!b.done) check(d_->cuEventCreate(&b.done, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
      b.cap = cap, b.used = false;
      b.mapped = host_memory(reinterpret_cast<uintptr_t>(b.host)) == HostMemory::mapped;
    }
    const Side pinned_host{{DeviceKind::host, 0}, DType::u8, true, s.bounce[0].mapped && s.bounce[1].mapped};
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

  // "ce_cast_<s>_<d>" / "ce_apply_<s>_<d>" back to DType codes.
  static bool parse_kernel(const std::string& name, int* sdt, int* ddt, int* identity) {
    std::string rest;
    if (name.rfind("ce_cast_", 0) == 0) *identity = 1, rest = name.substr(8);
    else if (name.rfind("ce_apply_", 0) == 0) *identity = 0, rest = name.substr(9);
    else return false;
    for (int a = 0; a < kNumConvertible; ++a)
      for (int b = 0; b < kNumConvertible; ++b)
        if (rest == short_name(static_cast<DType>(a)) + "_" + short_name(static_cast<DType>(b))) {
          *sdt = a, *ddt = b;
          return true;
        }
    return false;
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
      pinned_free(sl.host, sl.host_handle);
      dev_free(s, sl.dev);
      size_t cap = 64 << 10;
      while (cap < need) cap <<= 1;
      sl.host = pinned_alloc(cap, &sl.host_handle, "upload buffer");
      sl.dev = dev_alloc(s, cap, "upload buffer", st);
      sl.cap = cap;
    }
    std::memcpy(sl.host, data, need);
    check(d_->cuMemcpyHtoDAsync(sl.dev.ptr, sl.host, need, st), "cuMemcpyHtoDAsync");
    return sl;
  }

  // Every copy kernel launches here, within the engine's SM limit: on the
  // green-context stream (ordered after and before `st` by events, the host
  // never waits), or with the grid capped to that many blocks. All kernels
  // stride over their work, so any grid is correct.
  void run_kernel(Dev& s, CUfunction f, std::array<unsigned, 3> grid, std::array<unsigned, 3> block, void** args,
                  CUstream st, const char* what, unsigned smem = 0) {
    CUstream on = st;
    Lane* lane = s.sm_stream ? &lane_for(s, st) : nullptr;
    if (lane) {
      check(d_->cuEventRecord(lane->in, st), "cuEventRecord");
      check(d_->cuStreamWaitEvent(lane->sm, lane->in, 0), "cuStreamWaitEvent");
      on = lane->sm;
    } else if (s.grid_cap) {
      const unsigned cap = s.grid_cap;
      grid[2] = std::min(grid[2], cap);
      grid[1] = std::min(grid[1], std::max(1u, cap / grid[2]));
      grid[0] = std::min(grid[0], std::max(1u, cap / (grid[1] * grid[2])));
    }
    check(d_->cuLaunchKernel(f, grid[0], grid[1], grid[2], block[0], block[1], block[2], smem, on, args, nullptr),
          (std::string("cuLaunchKernel ") + what).c_str());
    if (lane) {
      check(d_->cuEventRecord(lane->out, lane->sm), "cuEventRecord");
      check(d_->cuStreamWaitEvent(st, lane->out, 0), "cuStreamWaitEvent");
    }
  }

  // The green-context lane serving `st`: its own while there are lanes to
  // spare, then one shared by hash (ordered, still correct).
  Lane& lane_for(Dev& s, CUstream st) {
    for (Lane& l : s.lanes)
      if (l.caller == st) return l;
    for (Lane& l : s.lanes)
      if (!l.caller) return l.caller = st, l;
    if (s.lanes.size() < kLanes) {
      Lane l;
      l.caller = st;
      check(d_->cuGreenCtxStreamCreate(&l.sm, s.green, CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate");
      check(d_->cuEventCreate(&l.in, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
      check(d_->cuEventCreate(&l.out, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
      s.lanes.push_back(l);
      return s.lanes.back();
    }
    return s.lanes[std::hash<const void*>{}(st) % s.lanes.size()];
  }

  // ---- SM limit ------------------------------------------------------------------------

  // Distinct SMs the blocks of one ce_smid launch on `st` ran on.
  int smids_used(Dev& s, CUstream st, int blocks) {
    CUdeviceptr out = 0;
    check(d_->cuMemAlloc(&out, blocks * sizeof(unsigned)), "cuMemAlloc");
    struct Free {
      const Driver* d;
      CUdeviceptr p;
      ~Free() { d->cuMemFree(p); }
    } free_out{d_, out};
    long long spin = 2'000'000;  // ~1 ms of clocks per block: long enough for every block to be resident at once
    void* args[] = {&out, &spin};
    check(d_->cuLaunchKernel(function(s, "ce_smid"), blocks, 1, 1, 32, 1, 1, 0, st, args, nullptr),
          "cuLaunchKernel ce_smid");
    std::vector<unsigned> ids(blocks);
    check(d_->cuMemcpyDtoHAsync(ids.data(), out, blocks * sizeof(unsigned), st), "cuMemcpyDtoHAsync");
    check(d_->cuStreamSynchronize(st), "cuStreamSynchronize");
    std::sort(ids.begin(), ids.end());
    return static_cast<int>(std::unique(ids.begin(), ids.end()) - ids.begin());
  }

  // Hard limit through a green context, admitted only if a launch on it stays
  // within its SMs while the same launch unconfined does not; else the soft
  // limit (a block cap). Planning time: called once per device.
  void setup_sm_limit(Dev& s) {
    const int want = opt_.max_sms;
    int total = 0;
    check(d_->cuDeviceGetAttribute(&total, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, s.dev), "cuDeviceGetAttribute");
    if (want >= total) {
      s.sm_limit = "none (the limit is not below the GPU's " + std::to_string(total) + " SMs)";
      return;
    }
    s.grid_cap = static_cast<unsigned>(std::max(1, want));
    s.sm_limit = "soft: at most " + std::to_string(s.grid_cap) + " thread blocks per copy kernel";
    try {
      for (const char* sym : {"cuDeviceGetDevResource", "cuDevSmResourceSplitByCount", "cuDevResourceGenerateDesc",
                              "cuGreenCtxCreate", "cuGreenCtxStreamCreate"})
        if (auto m = cuda::missing_reason(sym); !m.empty()) throw std::runtime_error(std::string(sym) + ": " + m);
      CUdevResource all{}, part{}, rest{};
      check(d_->cuDeviceGetDevResource(s.dev, &all, CU_DEV_RESOURCE_TYPE_SM), "cuDeviceGetDevResource");
      // The driver rounds a group up to its granularity: if that exceeds the
      // limit, ask for as much less as it rounded up (at most the limit, then).
      auto split = [&](int n) {
        unsigned groups = 1;
        check(d_->cuDevSmResourceSplitByCount(&part, &groups, &all, &rest, 0, static_cast<unsigned>(n)),
              "cuDevSmResourceSplitByCount");
        if (groups != 1) throw std::runtime_error("no SM group of " + std::to_string(n));
        return static_cast<int>(part.sm.smCount);
      };
      int granted = split(want);
      if (granted > want && want - (granted - want) > 0) granted = split(want - (granted - want));
      if (granted > want)
        throw std::runtime_error("the driver's SM groups (" + std::to_string(granted) + ") exceed the limit");
      CUdevResourceDesc desc{};
      check(d_->cuDevResourceGenerateDesc(&desc, &part, 1), "cuDevResourceGenerateDesc");
      check(d_->cuGreenCtxCreate(&s.green, desc, s.dev, CU_GREEN_CTX_DEFAULT_STREAM), "cuGreenCtxCreate");
      check(d_->cuGreenCtxStreamCreate(&s.sm_stream, s.green, CU_STREAM_NON_BLOCKING, 0), "cuGreenCtxStreamCreate");
      const int blocks = 8 * total;
      const int confined = smids_used(s, s.sm_stream, blocks), free = smids_used(s, s.side, blocks);
      if (confined > granted || free <= granted)
        throw std::runtime_error("confinement not verified (" + std::to_string(confined) + " SMs used on the green "
                                 "context, " + std::to_string(free) + " without)");
      check(d_->cuEventCreate(&s.sm_in, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
      check(d_->cuEventCreate(&s.sm_out, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
      s.grid_cap = 0;
      s.lanes.push_back({nullptr, s.sm_stream, s.sm_in, s.sm_out});
      s.sm_limit = "hard: green context of " + std::to_string(granted) + " SMs (verified)";
    } catch (const std::exception& e) {
      if (s.sm_stream) d_->cuStreamDestroy(s.sm_stream);
      if (s.green) d_->cuGreenCtxDestroy(s.green);
      s.sm_stream = nullptr, s.green = nullptr;
      s.sm_limit += std::string("; no green context: ") + e.what();
    }
  }

  // A parameter table in device memory for the life of the group that built
  // it: launches read it in place, so the hot path uploads nothing.
  struct DevTable {
    CudaExecutor* owner = nullptr;
    int device = 0;
    DevMem mem;
    CUdeviceptr ptr = 0;
    size_t count = 0;
    std::weak_ptr<int> alive;  // the executor
    ~DevTable() {
      if (!mem.ptr || alive.expired()) return;
      try {  // a destructor must not throw (std::terminate): after a sticky CUDA error the push fails
        Dev& s = owner->devs_[device];
        Push push(*owner->d_, s.ctx);
        owner->dev_free(s, mem);
      } catch (...) {  // NOLINT(bugprone-empty-catch): the memory is lost with the context
      }
    }
  };

  static constexpr int kPutWords = cuabi::kPutWords;
  static constexpr size_t kCaptureArena = 256 << 10;  // bytes of bases CUDA graph captures may record per device
  using Put = cuabi::Put;  // abi.hpp

  // A batch's bases for one CUDA graph capture, written by ce_put nodes so
  // replays read the bases recorded at capture time. They live in an arena
  // made at device setup (nothing may be allocated during a capture) until
  // the executor goes: a graph may replay any time.
  CUdeviceptr captured_bases(Dev& s, const std::vector<Base>& bases, CUstream st) {
    const size_t bytes = bases.size() * sizeof(Base);
    if (s.capture_used + bytes > s.capture_arena_bytes)
      fail(Errc::unsupported, "CUDA graph captures of copies with device indices used up their tables");
    const CUdeviceptr t = s.capture_arena.ptr + s.capture_used;
    s.capture_used += (bytes + 255) & ~size_t{255};
    put(s, t, bases, st);
    return t;
  }

  // Writes `bases` to device memory at `dst` in stream order, from kernel
  // parameters: no host buffer, no host wait.
  void put(Dev& s, CUdeviceptr dst, const std::vector<Base>& bases, CUstream st) {
    put_words(s, dst, reinterpret_cast<const unsigned long long*>(bases.data()), bases.size() * sizeof(Base) / 8, st);
  }

  void put_words(Dev& s, CUdeviceptr dst, const unsigned long long* w, size_t words, CUstream st) {
    CUfunction f = function(s, "ce_put");
    for (size_t at = 0; at < words; at += kPutWords) {
      Put p{};
      p.dst = dst + at * 8;
      p.n = static_cast<int>(std::min<size_t>(kPutWords, words - at));
      std::memcpy(p.w, w + at, static_cast<size_t>(p.n) * 8);
      void* args[] = {&p};
      check(d_->cuLaunchKernel(f, 1, 1, 1, 128, 1, 1, 0, st, args, nullptr), "cuLaunchKernel ce_put");
    }
  }

  // Host index tables of a batch: their current values (checked against the
  // bounds here) go to the batch's device mirror for this stream through
  // kernel parameters, read on the host now, so the caller may overwrite its
  // buffer as soon as the call returns; the bases point into the mirror.
  void pack_host_tables(Dev& s, FusedBatch& b, const Group& g, const std::vector<Op>& ops, std::vector<Base>& bases,
                        CUstream st) {
    const StmtDesc& d = b.desc;
    const Form& f0 = *g.forms[b.members[0]];
    const i64 ns = b.stab_host ? f0.shape[f0.stab_dim] : 0, nd = b.dtab_host ? f0.shape[f0.dtab_dim] : 0;
    std::vector<unsigned long long> w;
    w.reserve(bases.size() * static_cast<size_t>(ns + nd));
    auto take = [&](uintptr_t p, i64 n, i64 step, i64 bound) {
      const auto* v = reinterpret_cast<const int64_t*>(p);
      for (i64 i = 0; i < n; ++i) {
        const int64_t x = v[i * step];
        if (x < 0 || x >= bound) fail(Errc::invalid_argument, "a host index table held an index out of range");
        w.push_back(static_cast<unsigned long long>(x));
      }
    };
    for (size_t i = 0; i < bases.size(); ++i) {
      const Op& op = ops[g.ops[b.members[i]]];
      if (b.stab_host) take(op.stab, ns, op.form->stab_step, d.sbound);
      if (b.dtab_host) take(op.dtab, nd, op.form->dtab_step, d.dbound);
    }
    FusedBatch::Mirror* m = nullptr;
    for (auto& x : b.mirrors)
      if (x.stream == st) m = &x;
    if (!m) {
      if (b.mirrors.size() == kLanes) b.mirrors.erase(b.mirrors.begin());  // freed in order after its users
      m = &b.mirrors.emplace_back();
      m->stream = st;
    }
    if (m->words < w.size()) {
      auto t = std::make_shared<DevTable>();
      t->owner = this, t->device = s.index, t->alive = alive_, t->count = w.size();
      t->mem = dev_alloc(s, w.size() * 8, "host index mirror", st);
      t->ptr = t->mem.ptr;
      m->table = t, m->words = w.size();
    }
    const CUdeviceptr base = static_cast<DevTable*>(m->table.get())->ptr;
    put_words(s, base, w.data(), w.size(), st);
    size_t at = 0;
    for (Base& x : bases) {
      if (b.stab_host) x.stab = base + at * 8, at += static_cast<size_t>(ns);
      if (b.dtab_host) x.dtab = base + at * 8, at += static_cast<size_t>(nd);
    }
  }

  static bool host_tables_only(const Group& g) {
    return !g.forms.empty() && std::all_of(g.forms.begin(), g.forms.end(), [](const auto& f) {
             return (f->stab_dim < 0 || f->stab_host) && (f->dtab_dim < 0 || f->dtab_host) && f->host_tables();
           });
  }

  static_assert(sizeof(cv::TileK) % 8 == 0, "TileK is written as 8-byte words");
  struct TileExec {
    std::shared_ptr<void> table;  // DevTable of TileK: groups without index tables
    struct Mirror {
      CUstream stream{};
      std::shared_ptr<void> table;  // DevTable: this call's host index values, then the TileKs
      size_t words = 0;
    };
    std::vector<Mirror> mirrors;  // groups with index tables, per stream
  };

  static bool tile_indexed(const Group& g, const std::vector<Op>& ops) {
    return std::any_of(g.ops.begin(), g.ops.end(), [&](int k) { return ops[k].tile->xi || ops[k].tile->ci; });
  }

  // Pure device copies through shared memory: the chunk table of `kernel`,
  // one warp per block (the shared buffers are per block).
  static constexpr int64_t kBulkMinBytes = 1 << 20;
  static constexpr int kBulkBufs = cuabi::kBulkBufs;
  static constexpr int kBulkBlocksPerSm = 8;
  int64_t bulk(Dev& s, const Group& g, CUstream st) {
    auto& slot = g.exec_cache[static_cast<int>(Strategy::bulk)];
    if (!slot) slot = resident(s, chunks_of(g));
    const auto& t = *static_cast<DevTable*>(slot.get());
    if (t.count == 0) return 0;
    CUfunction f = function(s, "ce_bytes_bulk");
    CUdeviceptr table = t.ptr;
    int n = static_cast<int>(t.count);
    long long zero = 0;
    int seg = s.bulk_seg;
    void* args[] = {&table, &n, &zero, &zero, &seg};
    run_kernel(s, f, {static_cast<unsigned>(std::min<size_t>(t.count, 1 << 16)), 1, 1}, {32, 1, 1}, args, st,
               "ce_bytes_bulk", static_cast<unsigned>(kBulkBufs * seg));
    return 1;
  }

  // ---- 2-D blocks through tensor maps (sm90+) ---------------------------------------------

  using TmaBlock = cuabi::TmaBlock;  // abi.hpp
  static_assert(sizeof(CUtensorMap) == sizeof(TmaBlock::src), "a tensor map's bytes");
  static constexpr int kTmaBox = cuabi::kTmaBox;

  // What the hardware takes: 16-byte aligned addresses and pitches, widths up
  // to 2^32, and the box from the width (up to 256 bytes) and kTmaBox.
  static bool tma2d_block_ok(i64 s, i64 d, i64 w, i64 h, i64 sp, i64 dp) {
    if (((s | d) & 15) || w <= 0 || h <= 0 || w >= (i64{1} << 32) || h >= (i64{1} << 32)) return false;
    if (h > 1 && ((sp | dp) & 15)) return false;
    return true;
  }
  static bool tma2d_fits(const Group& g) {
    const Rects& r = g.rects;
    if (r.size() == 0 || r.size() > 4096) return false;  // a tensor map pair per block
    bool rows = false;
    for (size_t i = 0; i < r.size(); ++i) {
      if (!tma2d_block_ok(r.src[i], r.dst[i], r.width[i], r.height[i], r.src_pitch[i], r.dst_pitch[i])) return false;
      rows = rows || r.height[i] > 1;
    }
    return rows;  // 2-D work: 1-D runs are the row-wise bulk copy's
  }

  bool encode_tma(unsigned char* out, i64 addr, i64 w, i64 h, i64 pitch, int bw, int bh) {
    CUtensorMap map;
    CUtensorMap* m = &map;
    const cuuint64_t dims[2] = {static_cast<cuuint64_t>(w), static_cast<cuuint64_t>(h)};
    const cuuint64_t strides[1] = {static_cast<cuuint64_t>(h > 1 ? pitch : ((w + 15) & ~i64{15}))};
    const cuuint32_t box[2] = {static_cast<cuuint32_t>(bw), static_cast<cuuint32_t>(bh)}, el[2] = {1, 1};
    const bool ok = d_->cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, reinterpret_cast<void*>(addr), dims,
                                               strides, box, el, CU_TENSOR_MAP_INTERLEAVE_NONE,
                                               CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
    std::memcpy(out, &map, sizeof(map));
    return ok;
  }

  // The group's blocks as tensor map pairs; empty if the driver refuses one.
  std::vector<TmaBlock> tma_blocks(const Rects& r, i64 sadd = 0, i64 dadd = 0) {
    std::vector<TmaBlock> v(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
      TmaBlock& b = v[i];
      b.width = static_cast<int>(r.width[i]), b.height = static_cast<int>(r.height[i]);
      b.bw = static_cast<int>(std::min<i64>(256, (r.width[i] + 15) & ~i64{15}));
      b.bh = static_cast<int>(std::clamp<i64>(kTmaBox / b.bw, 1, 256));
      if (!encode_tma(b.src, r.src[i] + sadd, r.width[i], r.height[i], r.src_pitch[i], b.bw, b.bh) ||
          !encode_tma(b.dst, r.dst[i] + dadd, r.width[i], r.height[i], r.dst_pitch[i], b.bw, b.bh))
        return {};
    }
    return v;
  }

  int64_t tma2d(Dev& s, const Group& g, CUstream st) {
    auto& slot = g.exec_cache[static_cast<int>(Strategy::tma2d)];
    if (!slot) {
      const std::vector<TmaBlock> v = tma_blocks(g.rects);
      if (v.empty()) fail(Errc::backend, "cuTensorMapEncodeTiled refused a block");
      slot = resident(s, v);
    }
    const auto& t = *static_cast<DevTable*>(slot.get());
    CUfunction f = function(s, "ce_bytes_tma2d");
    CUdeviceptr table = t.ptr;
    int n = static_cast<int>(t.count);
    void* args[] = {&table, &n};
    run_kernel(s, f, {static_cast<unsigned>(std::min<size_t>(t.count, 1 << 16)), 1, 1}, {32, 1, 1}, args, st,
               "ce_bytes_tma2d");
    return 1;
  }

  void admit_tma2d(Dev& s) {
    s.tma2d = false;
    if (!s.kernels) return void(s.tma2d_error = "no kernels");
    int major = 0;
    check(d_->cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, s.dev), "cuDeviceGetAttribute");
    if (major < 9) return void(s.tma2d_error = "needs sm90 or later");
    if (!d_->cuTensorMapEncodeTiled)
      return void(s.tma2d_error = "cuTensorMapEncodeTiled: " + cuda::missing_reason("cuTensorMapEncodeTiled"));
    try {  // narrow rows (two boxes across, edge boxes on both axes), pitched: verified byte for byte
      const i64 rows = 77, width = 272, sp = 384, dp = 320;
      const size_t sbytes = rows * sp, dbytes = rows * dp;
      CUdeviceptr a = 0, b = 0;
      check(d_->cuMemAlloc(&a, sbytes), "cuMemAlloc");
      check(d_->cuMemAlloc(&b, dbytes), "cuMemAlloc");
      std::vector<uint8_t> in(sbytes), out(dbytes, 0);
      for (size_t i = 0; i < sbytes; ++i) in[i] = static_cast<uint8_t>(i * 131 + 7);
      check(d_->cuMemcpyHtoDAsync(a, in.data(), sbytes, s.side), "cuMemcpyHtoDAsync");
      check(d_->cuMemsetD8Async(b, 0, dbytes, s.side), "cuMemsetD8Async");
      Rects r;  // three blocks (a table of several: its layout must match the kernel's)
      for (i64 k = 0, row = 0; k < 3; ++k) {
        const i64 h = k == 2 ? rows - row : rows / 3;
        r.push(static_cast<i64>(a) + row * sp, static_cast<i64>(b) + row * dp, width, h, sp, dp);
        row += h;
      }
      const std::vector<TmaBlock> v = tma_blocks(r);
      if (v.empty()) {
        d_->cuMemFree(a), d_->cuMemFree(b);
        return void(s.tma2d_error = "cuTensorMapEncodeTiled refused the test block");
      }
      Slot& sl = upload(s, v.data(), v.size() * sizeof(TmaBlock), s.side);
      CUfunction f = function(s, "ce_bytes_tma2d");
      CUdeviceptr t = sl.dev.ptr;
      int n = static_cast<int>(v.size());
      void* args[] = {&t, &n};
      check(d_->cuLaunchKernel(f, 1, 1, 1, 32, 1, 1, 0, s.side, args, nullptr), "cuLaunchKernel ce_bytes_tma2d");
      check(d_->cuEventRecord(sl.done, s.side), "cuEventRecord");
      sl.used = true;
      check(d_->cuMemcpyDtoHAsync(out.data(), b, dbytes, s.side), "cuMemcpyDtoHAsync");
      check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
      d_->cuMemFree(a);
      d_->cuMemFree(b);
      bool ok = true;
      for (i64 row = 0; row < rows && ok; ++row) {
        ok = std::memcmp(out.data() + row * dp, in.data() + row * sp, width) == 0;
        for (i64 k = width; k < dp && ok; ++k) ok = out[row * dp + k] == 0;  // nothing past the block
      }
      if (ok) s.tma2d = true, s.tma2d_error.clear();
      else s.tma2d_error = "tma2d test copy came back wrong";
    } catch (const std::exception& e) {
      s.tma2d_error = std::string("tma2d: ") + e.what();
    }
  }

  // Tile ops: their descriptors resident with the plan, one launch per group.
  // With index tables the descriptors carry this call's tables (host values
  // copied next to them), written in stream order through kernel parameters.
  int64_t tile(Dev& s, const Group& g, const std::vector<Op>& ops, CUstream st) {
    auto& slot = g.exec_cache[static_cast<int>(Strategy::tile)];
    if (!slot) slot = std::make_shared<TileExec>();
    auto& x = *static_cast<TileExec*>(slot.get());
    CUdeviceptr table = 0;
    if (!tile_indexed(g, ops)) {
      if (!x.table) {
        std::vector<cv::TileK> v;
        for (int k : g.ops) v.push_back(*ops[k].tile);
        x.table = resident(s, v);
      }
      table = static_cast<DevTable*>(x.table.get())->ptr;
    } else {
      std::vector<unsigned long long> vals;  // host index values first, then the TileKs
      std::vector<std::pair<size_t, size_t>> at;
      host_index_values(g, ops, vals, at);
      std::vector<cv::TileK> v;
      for (int k : g.ops) v.push_back(*ops[k].tile);
      table = upload_tiles(s, x, std::move(vals), v, &at, st);
    }
    launch_tiles(s, g, ops, table, st);
    return 1;
  }

  // The host index values of a group's ops (this call's, for a plan's
  // parameter buffers, checked now), appended to `vals`; `at` gets each op's
  // x / c offsets in it (kNoIndex: none, or a GPU table).
  static constexpr size_t kNoIndex = ~size_t{0};
  static void host_index_values(const Group& g, const std::vector<Op>& ops, std::vector<unsigned long long>& vals,
                                std::vector<std::pair<size_t, size_t>>& at) {
    for (int k : g.ops) {
      const Op& op = ops[k];
      std::pair<size_t, size_t> o{kNoIndex, kNoIndex};
      for (int w = 0; w < 2; ++w)
        if (op.tile_index[w]) {
          (w ? o.second : o.first) = vals.size();
          for (int64_t i : *op.tile_index[w]) vals.push_back(static_cast<unsigned long long>(i));
        } else if (op.tile_host_ptr[w]) {
          (w ? o.second : o.first) = vals.size();
          const auto* p = reinterpret_cast<const int64_t*>(static_cast<uintptr_t>(w ? op.tile->ci : op.tile->xi));
          const long long bound = w ? op.tile->cb : op.tile->xb;
          for (long long r = 0; r < op.tile->rows[0]; ++r) {
            if (p[r] < 0 || p[r] >= bound) fail(Errc::invalid_argument, "index out of range");
            vals.push_back(static_cast<unsigned long long>(p[r]));
          }
        }
      at.push_back(o);
    }
  }

  // This call's TileKs (after `vals`, host index values they point into:
  // `at` gives each op's x / c offsets in vals), written in stream order to
  // the group's mirror for this stream; returns the TileK table's address.
  CUdeviceptr upload_tiles(Dev& s, TileExec& x, std::vector<unsigned long long> vals, std::vector<cv::TileK>& v,
                           const std::vector<std::pair<size_t, size_t>>* at, CUstream st) {
    constexpr size_t kNone = kNoIndex;
    {
      const size_t tk_words = sizeof(cv::TileK) / 8, words = vals.size() + v.size() * tk_words;
      TileExec::Mirror* m = nullptr;
      for (auto& y : x.mirrors)
        if (y.stream == st) m = &y;
      if (!m) {
        if (x.mirrors.size() == kLanes) x.mirrors.erase(x.mirrors.begin());  // freed in order after its users
        m = &x.mirrors.emplace_back();
        m->stream = st;
      }
      if (m->words < words) {
        auto t = std::make_shared<DevTable>();
        t->owner = this, t->device = s.index, t->alive = alive_, t->count = words;
        t->mem = dev_alloc(s, words * 8, "tile descriptors", st);
        t->ptr = t->mem.ptr;
        m->table = t, m->words = words;
      }
      const CUdeviceptr base = static_cast<DevTable*>(m->table.get())->ptr;
      if (at) {
        const CUdeviceptr err = error_word(s);
        for (size_t i = 0; i < v.size(); ++i) {
          if ((*at)[i].first != kNone) v[i].xi = static_cast<long long>(base + (*at)[i].first * 8);
          if ((*at)[i].second != kNone) v[i].ci = static_cast<long long>(base + (*at)[i].second * 8);
          v[i].err = static_cast<long long>(err);
        }
        s.err_armed = true;
      }
      const auto* tw = reinterpret_cast<const unsigned long long*>(v.data());
      vals.insert(vals.end(), tw, tw + v.size() * tk_words);
      put_words(s, base, vals.data(), vals.size(), st);
      return base + (words - v.size() * tk_words) * 8;
    }
  }

  void launch_tiles(Dev& s, const Group& g, const std::vector<Op>& ops, CUdeviceptr table, CUstream st) {
    long long most = 0;
    for (int k : g.ops) most = std::max(most, cv::tile_warps(*ops[k].tile));
    launch_tiles(s, *ops[g.ops[0]].tile, table, static_cast<int>(g.ops.size()), most, st);
  }

  // `count` TileKs at `table`, of t0's format and direction (a group shares
  // them); `most`: the most warps one of them has work for (cv::tile_warps).
  void launch_tiles(Dev& s, const cv::TileK& t0, CUdeviceptr table, int count, long long most, CUstream st) {
    const std::string name = std::string(t0.quantize ? "ce_tile_q" : "ce_tile_d") + std::to_string(t0.fmt);
    CUfunction f = function(s, name);
    void* args[] = {&table, &count};
    const unsigned gx = static_cast<unsigned>(std::min<long long>((most + 7) / 8, 1 << 20));
    const unsigned gy = static_cast<unsigned>(std::min<long long>(count, 65535));
    run_kernel(s, f, {std::max(1u, gx), gy, 1}, {256, 1, 1}, args, st, name.c_str());
  }

  // ---- tile ops with host sides: through device staging ----------------------------------
  //
  // Host sides are staged in device memory: inputs copied in before the
  // kernel, outputs copied back after it. A side whose rows a host index
  // table selects is staged as compact rows (one batched DMA per side and
  // direction), and the staged TileK drops that index; GPU sides keep theirs.

  struct TileSide {
    long long cv::TileK::*field;     // the base it rewrites
    long long (cv::TileK::*rs)[3];  // its row strides
    int idx;                         // the index selecting its rows: 0 xi, 1 ci, -1 none
    bool host, input;
    long long lo, hi;  // byte extent of one row of the first dimension, from the base
  };

  // A TileK's four tensors: extent, whether on the host, whether the kernel reads them.
  static std::vector<TileSide> tile_sides(const cv::TileK& t, const Op& op) {
    auto ext = [&](const long long r[3], long long count, long long step, long long width) {
      long long lo = 0, hi = 0;
      for (int k = 1; k < 3; ++k) {
        const long long span = (t.rows[k] - 1) * r[k];
        lo += std::min(0LL, span), hi += std::max(0LL, span);
      }
      const long long span = (count - 1) * step;
      lo += std::min(0LL, span), hi += std::max(0LL, span);
      return std::pair<long long, long long>{lo, hi + width};
    };
    const bool packed = cv::qfmt_packed(static_cast<cv::QFmt>(t.fmt));
    const long long vals = static_cast<long long>(t.g) * t.upr;
    const long long xw = static_cast<long long>(cv::dt_size(t.xdt)), sw = t.scale_byte ? 1 : 4;
    const Side& xs = t.quantize ? op.src : op.dst;
    const Side& qs = t.quantize ? op.dst : op.src;
    const bool writes_scale = t.quantize && !t.static_scale;
    const int xi = t.xi ? 0 : -1, ci = t.ci ? 1 : -1;
    std::vector<TileSide> v;
    auto add = [&](long long cv::TileK::*f, long long (cv::TileK::*rs)[3], int idx, bool host, bool input,
                   long long count, long long step, long long width, long long pitch = 0) {
      auto e = ext(t.*rs, count, step, width);
      if (t.bh > 1 && pitch) {  // a box's rows
        const long long span = (t.bh - 1) * pitch;
        e.first += std::min(0LL, span), e.second += std::max(0LL, span);
      }
      v.push_back({f, rs, idx, host, input, e.first, e.second});
    };
    add(&cv::TileK::x, &cv::TileK::xr, xi, xs.device.kind == DeviceKind::host, t.quantize != 0, vals, t.xe, xw,
        t.xp);
    add(&cv::TileK::q, &cv::TileK::qr, ci, qs.device.kind == DeviceKind::host, !t.quantize, packed ? vals / 2 : vals,
        t.qe, 1, t.qp);
    add(&cv::TileK::s, &cv::TileK::sr, ci, op.extras[0].device.kind == DeviceKind::host, !writes_scale, t.upr, t.se,
        sw);
    if (op.extras.size() > 1)
      add(&cv::TileK::z, &cv::TileK::zr, ci, op.extras[1].device.kind == DeviceKind::host, !writes_scale, t.upr, t.ze,
          1);
    return v;
  }

  static size_t align256(long long n) { return (static_cast<size_t>(n) + 255) & ~size_t{255}; }

  // Rows [a, b) of the first dimension of a host side: the bytes they span
  // (unindexed: one interval; indexed: one staged row each).
  static std::pair<long long, long long> rows_extent(const cv::TileK& t, const TileSide& x, long long a, long long b) {
    const long long base = t.*x.field, r0 = (t.*x.rs)[0];
    return {base + std::min(a * r0, (b - 1) * r0) + x.lo, base + std::max(a * r0, (b - 1) * r0) + x.hi};
  }
  static size_t rows_staged(const cv::TileK& t, const TileSide& x, long long a, long long b) {
    if (x.idx >= 0) return static_cast<size_t>(b - a) * align256(x.hi - x.lo);
    const auto e = rows_extent(t, x, a, b);
    return align256(e.second - e.first);
  }

  // Tile ops with host sides through device staging, in chunks of rows of
  // their first dimension: a chunk's inputs move in on the aux stream while
  // the previous chunk's kernel runs, and its outputs move out while the next
  // one's does. Chunks alternate between two halves of the staging buffer.
  // A host side is staged as rows [a, b) (one DMA; or, selected by host index
  // values, one DMA per row, batched), and the chunk's TileK works on it.
  static constexpr size_t kStageOneShot = 2 << 20;  // below: one chunk, everything on the caller's stream
  static constexpr int kStageChunks = 4;

  int64_t tile_staged(Dev& s, const Group& g, const std::vector<Op>& ops, CUstream st) {
    std::vector<unsigned long long> vals;
    std::vector<std::pair<size_t, size_t>> at;
    host_index_values(g, ops, vals, at);
    const size_t n = g.ops.size();
    std::vector<std::vector<TileSide>> sides(n);
    std::vector<size_t> whole(n);
    std::vector<bool> split(n);
    size_t total = 0;
    for (size_t i = 0; i < n; ++i) {
      const Op& op = ops[g.ops[i]];
      const cv::TileK& t = *op.tile;
      sides[i] = tile_sides(t, op);
      bool sp = t.rows[0] > 1;
      for (const TileSide& x : sides[i]) {
        if (!x.host) continue;
        whole[i] += rows_staged(t, x, 0, t.rows[0]);
        if (x.idx < 0) sp = sp && std::llabs((t.*x.rs)[0]) >= x.hi - x.lo;  // rows' extents do not interleave
      }
      split[i] = sp;
      total += whole[i];
    }
    // Pieces (rows [a, b) of op i) of about `target` bytes, packed into chunks.
    const size_t half = static_cast<size_t>(std::max<int64_t>(opt_.staging_budget, 2 << 20)) / 2;
    const size_t target =
        total <= kStageOneShot ? total : std::clamp(total / kStageChunks, size_t{1} << 20, std::max(half, size_t{1}));
    struct Piece {
      size_t i;
      long long a, b;
      size_t bytes;
    };
    auto piece_bytes = [&](size_t i, long long a, long long b) {
      size_t sum = 0;
      for (const TileSide& x : sides[i])
        if (x.host) sum += rows_staged(*ops[g.ops[i]].tile, x, a, b);
      return sum;
    };
    std::vector<std::vector<Piece>> chunks(1);
    size_t fill = 0, most = 0;
    for (size_t i = 0; i < n; ++i) {
      const long long rows = ops[g.ops[i]].tile->rows[0];
      long long step = rows;
      if (split[i] && whole[i] > target)
        step = std::max<long long>(1, static_cast<long long>(static_cast<double>(rows) * static_cast<double>(target) / static_cast<double>(whole[i])));
      for (long long a = 0; a < rows; a += step) {
        const long long b = std::min(rows, a + step);
        const size_t by = whole[i] && step == rows ? whole[i] : piece_bytes(i, a, b);
        if (fill > 0 && fill + by > target) chunks.emplace_back(), fill = 0;
        chunks.back().push_back({i, a, b, by});
        fill += by;
        most = std::max(most, fill);
      }
    }
    const size_t nch = chunks.size();
    if (nch > 1 && most > half) chunks = {[&] {  // a piece too large to double-buffer: one round
      std::vector<Piece> all;
      for (auto& c : chunks) all.insert(all.end(), c.begin(), c.end());
      return all;
    }()};
    const bool piped = chunks.size() > 1;
    const size_t H = piped ? most : total;
    const CUdeviceptr stg = staging(s, piped ? 2 * H : H, st);
    // The user's host tensors: mapped only if all of them are (batched DMA takes device addresses).
    const Side dev_side{{DeviceKind::cuda, s.index}, DType::u8, false},
        host_side{{DeviceKind::host, 0}, DType::u8, true, g.tile_mapped};

    // The chunks' TileKs (one table) and DMA.
    std::vector<cv::TileK> v;
    std::vector<std::pair<size_t, size_t>> vat;
    struct Round {
      size_t first, count;
      long long units;
      std::vector<Raw> in, out;
    };
    std::vector<Round> rounds;
    bool indexed = false;
    for (size_t c = 0; c < chunks.size(); ++c) {
      Round r{v.size(), chunks[c].size(), 0, {}, {}};
      const CUdeviceptr d0 = stg + (c % 2) * H;
      size_t off = 0;
      for (const Piece& p : chunks[c]) {
        const cv::TileK& t0 = *ops[g.ops[p.i]].tile;
        cv::TileK t = t0;
        std::pair<size_t, size_t> o = at[p.i];
        bool drop[2] = {false, false};
        for (const TileSide& x : sides[p.i]) {
          const long long base = t0.*x.field, r0 = (t0.*x.rs)[0];
          if (!x.host) {
            if (x.idx < 0) t.*x.field = base + p.a * r0;
            continue;
          }
          const CUdeviceptr d = d0 + off;
          if (x.idx < 0) {
            const auto e = rows_extent(t0, x, p.a, p.b);
            const size_t len = static_cast<size_t>(e.second - e.first);
            const Raw raw = x.input ? Raw{static_cast<uintptr_t>(e.first), static_cast<uintptr_t>(d), len}
                                    : Raw{static_cast<uintptr_t>(d), static_cast<uintptr_t>(e.first), len};
            (x.input ? r.in : r.out).push_back(raw);
            t.*x.field = static_cast<long long>(d) + (base + p.a * r0 - e.first);
            off += align256(static_cast<long long>(len));
          } else {  // host index values (the engine stages no host side a GPU table selects)
            const size_t w = align256(x.hi - x.lo), len = static_cast<size_t>(x.hi - x.lo);
            const size_t vo = x.idx ? at[p.i].second : at[p.i].first;
            for (long long k = p.a; k < p.b; ++k) {
              const uintptr_t h = static_cast<uintptr_t>(base + static_cast<long long>(vals[vo + k]) * r0 + x.lo);
              const uintptr_t dr = static_cast<uintptr_t>(d + (k - p.a) * w);
              (x.input ? r.in : r.out).push_back(x.input ? Raw{h, dr, len} : Raw{dr, h, len});
            }
            t.*x.field = static_cast<long long>(d) - x.lo;
            (t.*x.rs)[0] = static_cast<long long>(w);
            drop[x.idx] = true;
            off += w * static_cast<size_t>(p.b - p.a);
          }
        }
        t.rows[0] = p.b - p.a;
        for (int w = 0; w < 2; ++w) {
          long long& tab = w ? t.ci : t.xi;
          size_t& slot = w ? o.second : o.first;
          if (!tab) continue;
          if (drop[w]) tab = 0, slot = kNoIndex;
          else if (slot != kNoIndex) slot += static_cast<size_t>(p.a);  // host values: from row a on
          else tab += p.a * 8;                                          // a GPU table
        }
        r.units = std::max(r.units, cv::tile_warps(t));  // (warps, for the launch)
        v.push_back(t);
        vat.push_back(o);
      }
      rounds.push_back(std::move(r));
    }
    for (size_t k = 0; k < v.size(); ++k) indexed = indexed || v[k].xi || v[k].ci;
    auto& slot = g.exec_cache[static_cast<int>(Strategy::tile_staged)];
    if (!slot) slot = std::make_shared<TileExec>();
    const CUdeviceptr table =
        upload_tiles(s, *static_cast<TileExec*>(slot.get()), std::move(vals), v, indexed ? &vat : nullptr, st);
    const cv::TileK& t0 = *ops[g.ops[0]].tile;
    auto launch = [&](const Round& r) {
      launch_tiles(s, t0, table + r.first * sizeof(cv::TileK), static_cast<int>(r.count), r.units, st);
    };
    int64_t calls = 0;
    auto dma = [&](const std::vector<Raw>& raws, bool in, CUstream on) {
      if (raws.empty()) return;
      copy_raws(s, in ? host_side : dev_side, in ? dev_side : host_side, raws, on);
      calls += raws.size() > 2 && s.batch ? 1 : static_cast<int64_t>(raws.size());
    };
    if (!piped) {
      dma(rounds[0].in, true, st);
      launch(rounds[0]);
      dma(rounds[0].out, false, st);
    } else {
      Staging& sg = staging_of(s, st);
      if (!sg.aux) {
        check(d_->cuStreamCreate(&sg.aux, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
        for (CUevent* e : {&sg.fork, &sg.join, &sg.in, &sg.kern, &sg.out[0], &sg.out[1]})
          check(d_->cuEventCreate(e, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
      }
      auto rec = [&](CUevent e, CUstream on) { check(d_->cuEventRecord(e, on), "cuEventRecord"); };
      auto wait = [&](CUstream on, CUevent e) { check(d_->cuStreamWaitEvent(on, e, 0), "cuStreamWaitEvent"); };
      // aux: in_0, then per chunk c: in_{c+1}, [kernel c], out_c. A half is
      // refilled (in_{c+2}) after kernel c and out_c, both earlier on aux;
      // kernel c writes its half after out_{c-2} has read it.
      rec(sg.fork, st), wait(sg.aux, sg.fork);
      dma(rounds[0].in, true, sg.aux);
      rec(sg.in, sg.aux);
      for (size_t c = 0; c < rounds.size(); ++c) {
        wait(st, sg.in);
        if (c >= 2) wait(st, sg.out[c % 2]);
        launch(rounds[c]);
        rec(sg.kern, st);
        if (c + 1 < rounds.size()) {
          dma(rounds[c + 1].in, true, sg.aux);
          rec(sg.in, sg.aux);
        }
        wait(sg.aux, sg.kern);
        dma(rounds[c].out, false, sg.aux);
        rec(sg.out[c % 2], sg.aux);
      }
      rec(sg.join, sg.aux), wait(st, sg.join);
    }
    staging_release(s, st);
    return calls + static_cast<int64_t>(rounds.size());
  }

  // Host index tables expanded into copy descriptors now (DMA; pure copies).
  int64_t lowered(Dev& s, const Group& g, const std::vector<Op>& ops, CUstream st) {
    const size_t isz = itemsize(g.src.dtype);
    std::vector<Raw> raws;
    for (int k : g.ops) {
      const Op& op = ops[k];
      const Runs r = lower_form(*op.form, reinterpret_cast<const int64_t*>(op.stab),
                                reinterpret_cast<const int64_t*>(op.dtab));
      for (size_t i = 0; i < r.size(); ++i)
        raws.push_back({static_cast<uintptr_t>(r.src[i]) * isz, static_cast<uintptr_t>(r.dst[i]) * isz,
                        static_cast<size_t>(r.len[i]) * isz});
    }
    copy_raws(s, g.src, g.dst, raws, st);
    return static_cast<int64_t>(raws.size());
  }

  // Eager calls: the bases of the last few pointer sets stay on the device, so
  // a call with the same index tensors (a static index buffer) launches only
  // the copy kernel. An entry is read and rewritten on one stream only, which
  // orders a rewrite after the kernels that read it.
  static constexpr size_t kBaseEntries = 4;
  CUdeviceptr current_bases(Dev& s, FusedBatch& b, const std::vector<Base>& bases, CUstream st) {
    const auto sameptrs = [&](const BaseEntry& e) {
      for (size_t i = 0; i < bases.size(); ++i)
        if (e.ptrs[2 * i] != bases[i].stab || e.ptrs[2 * i + 1] != bases[i].dtab) return false;
      return true;
    };
    BaseEntry* hit = nullptr;
    for (BaseEntry& e : b.recent)
      if (e.stream == st && sameptrs(e)) hit = &e;
    if (!hit) {
      for (BaseEntry& e : b.recent)
        if (e.stream == st && (!hit || e.used < hit->used)) hit = &e;  // rewrite in stream order
      if (!hit && b.recent.size() < kBaseEntries) hit = &b.recent.emplace_back();
      if (!hit) {  // all on other streams: a fresh table; the old one is freed in order
        hit = &*std::min_element(b.recent.begin(), b.recent.end(),
                                 [](const BaseEntry& x, const BaseEntry& y) { return x.used < y.used; });
        *hit = BaseEntry{};
      }
      if (!hit->table) {
        auto t = std::make_shared<DevTable>();
        t->owner = this, t->device = s.index, t->alive = alive_, t->count = bases.size();
        t->mem = dev_alloc(s, bases.size() * sizeof(Base), "copy table", st);
        t->ptr = t->mem.ptr;
        hit->table = t;
      }
      hit->stream = st;
      hit->ptrs.resize(2 * bases.size());
      for (size_t i = 0; i < bases.size(); ++i) hit->ptrs[2 * i] = bases[i].stab, hit->ptrs[2 * i + 1] = bases[i].dtab;
      put(s, static_cast<DevTable*>(hit->table.get())->ptr, bases, st);
    }
    hit->used = ++b.clock;
    return static_cast<DevTable*>(hit->table.get())->ptr;
  }

  // Built once, when a group first runs a strategy (planning, not the hot path).
  template <class T>
  std::shared_ptr<DevTable> resident(Dev& s, const std::vector<T>& v) {
    auto t = std::make_shared<DevTable>();
    t->owner = this, t->device = s.index, t->alive = alive_, t->count = v.size();
    if (v.empty()) return t;
    const size_t bytes = v.size() * sizeof(T);
    t->mem = dev_alloc(s, bytes, "copy table", s.side);
    t->ptr = t->mem.ptr;
    check(d_->cuMemcpyHtoDAsync(t->ptr, v.data(), bytes, s.side), "cuMemcpyHtoDAsync");
    check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
    return t;
  }

  // Chunk kernels on a device table; `sadd` / `dadd` offset every chunk.
  void launch_table(Dev& s, const std::string& name, CUdeviceptr table, size_t chunks, const cv::Fn& fn,
                    long long sadd, long long dadd, CUstream st) {
    if (chunks == 0) return;
    CUfunction f = find_function(s, name);
    const bool bytes = name == "ce_bytes" || name == "ce_bytes_cs";  // no function argument
    // A pair without a compiled kernel: the run-time dispatched one.
    int sdt = 0, ddt = 0, identity = 0;
    const bool dyn = !f && parse_kernel(name, &sdt, &ddt, &identity);
    if (dyn) f = function(s, "ce_convert_dyn");
    if (!f) f = function(s, name);  // reports the missing kernel
    for (size_t at = 0; at < chunks;) {
      const size_t n = std::min<size_t>(chunks - at, size_t{1} << 24);
      int count = static_cast<int>(n);
      CUdeviceptr t = table + at * sizeof(Chunk);
      cv::Fn fcopy = fn;
      void* typed_args[] = {&t, &count, &fcopy, &sadd, &dadd, &sdt, &ddt, &identity};  // dyn reads the last three
      void* bytes_args[] = {&t, &count, &sadd, &dadd};
      const unsigned threads = 256, warps = threads / 32;
      const unsigned blocks = static_cast<unsigned>((n + warps - 1) / warps);
      run_kernel(s, f, {blocks, 1, 1}, {threads, 1, 1}, bytes ? bytes_args : typed_args, st, name.c_str());
      at += n;
    }
  }

  // A one-off table (admission, passthrough of a single op) through the upload ring.
  void launch(Dev& s, const std::string& name, const std::vector<Chunk>& chunks, const cv::Fn& fn, CUstream st) {
    if (chunks.empty()) return;
    Slot& sl = upload(s, chunks.data(), chunks.size() * sizeof(Chunk), st);
    launch_table(s, name, sl.dev.ptr, chunks.size(), fn, 0, 0, st);
    check(d_->cuEventRecord(sl.done, st), "cuEventRecord");
    sl.used = true;
  }

  // Destination elements one thread of ce_stmt stores at once: the innermost
  // dimension is contiguous on the destination and divisible by the width, and
  // every row start is aligned to it. A row starts at a statement's base (plus
  // `dadd`) plus any combination of the outer strides, so the width must divide
  // the base and every outer stride whose dimension has more than one index
  // (row_align, in elements); a table dimension's stride counts too (any index).
  static long long row_align(const StmtDesc& d, long long base) {
    long long g = base < 0 ? -base : base;
    for (int k = 0; k + 1 < d.rank; ++k)
      if (d.shape[k] > 1 || k == d.dtab_dim) g = std::gcd(g, d.ds[k] < 0 ? -d.ds[k] : d.ds[k]);
    return g;  // 0: every row starts at element 0 (aligned to anything)
  }
  static int vector_width(const FusedBatch& b, long long dadd) {
    const StmtDesc& d = b.desc;
    if (d.rank == 0 || d.ds[d.rank - 1] != 1 || b.tile_a >= 0) return 1;
    if (d.stab_dim == d.rank - 1 || d.dtab_dim == d.rank - 1) return 1;  // indices, not consecutive elements
    const int dsz = static_cast<int>(itemsize(static_cast<DType>(d.ddt)));
    for (int v = 16 / dsz; v >= 2; v /= 2) {
      if (d.shape[d.rank - 1] % v != 0) continue;
      bool ok = true;
      for (const Base& x : b.bases) ok = ok && row_align(d, x.dst + dadd) % v == 0;
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

  // `table`: `count` Base entries in device memory.
  void launch_stmt(CUdeviceptr table, size_t count_, const FusedBatch& batch, const cv::Fn& fn, long long sadd,
                   long long dadd, Dev& s, CUstream st) {
    if (count_ == 0) return;
    const StmtDesc& desc = batch.desc;
    const bool tiled = batch.tile_a >= 0;
    CUfunction f = tiled ? function(s, "ce_stmt_tile") : find_function(s, stmt_kernel(desc));
    if (!f) f = function(s, "ce_stmt");
    CUdeviceptr t = table;
    StmtDesc dcopy = desc;
    if (desc.stab_dim >= 0 || desc.dtab_dim >= 0) {
      dcopy.err = static_cast<unsigned long long>(error_word(s));
      s.err_armed = true;
      if (capturing(reinterpret_cast<uintptr_t>(st))) s.err_graphs = true;
    }
    cv::Fn fcopy = fn;
    if (tiled) {
      const long long sa = desc.shape[batch.tile_a], sb = desc.shape[batch.tile_b];
      long long count = static_cast<long long>(count_) * (static_cast<long long>(desc.numel) / (sa * sb));
      int a = batch.tile_a, b = batch.tile_b;
      void* args[] = {&t, &count, &dcopy, &fcopy, &a, &b, &sadd, &dadd};
      const long long tiles = ((sb + 31) / 32) * ((sa + 31) / 32);
      const unsigned gx = static_cast<unsigned>(std::min<long long>(tiles, 1u << 30));
      const unsigned gz = static_cast<unsigned>(std::min<long long>(count, 65535));
      run_kernel(s, f, {gx, 1, gz}, {32, 8, 1}, args, st, "ce_stmt_tile");
    } else {
      long long count = static_cast<long long>(count_);
      void* args[] = {&t, &count, &dcopy, &fcopy, &sadd, &dadd};
      const unsigned threads = 256;
      const unsigned per = (desc.numel + desc.vec - 1) / desc.vec;
      const unsigned gx = (per + threads - 1) / threads;
      const unsigned gy = static_cast<unsigned>(std::min<long long>(count, 65535));
      run_kernel(s, f, {gx, gy, 1}, {threads, 1, 1}, args, st, "ce_stmt");
    }
  }

  // ---- fused structured statements --------------------------------------------------------

  FusedPlan build_fused(const Group& g) const {
    FusedPlan P;
    if (g.forms.empty()) return P;
    const bool src_cuda = g.src.device.kind == DeviceKind::cuda, dst_cuda = g.dst.device.kind == DeviceKind::cuda;
    if (src_cuda && dst_cuda && g.src.device.index != g.dst.device.index) return P;
    // Copies with host index tables read or write host memory directly
    // (zero-copy, admitted in candidates()), never through staging.
    P.cross = !(src_cuda && dst_cuda) && !g.opaque();
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
        b.desc.stab_dim = f.stab_dim, b.desc.dtab_dim = f.dtab_dim;
        b.desc.stab_step = f.stab_step, b.desc.dtab_step = f.dtab_step;
        b.desc.sbound = f.sbound, b.desc.dbound = f.dbound;
        b.desc.err = 0;
        b.device_tables = f.device_tables();
        b.stab_host = f.stab_dim >= 0 && f.stab_host, b.dtab_host = f.dtab_dim >= 0 && f.dtab_host;
        if (b.stab_host) b.desc.stab_step = 1;  // packed in the mirror
        if (b.dtab_host) b.desc.dtab_step = 1;
        if (!b.device_tables) transpose_dims(f, &b.tile_a, &b.tile_b);
        b.desc.vec = 1;
        P.batches.push_back(std::move(b));
      }
      batch_of[i] = j;
      P.batches[j].bases.push_back({f.s_base, f.d_base, 0, 0});
      P.batches[j].members.push_back(static_cast<int>(i));
    }
    if (!P.cross) {
      for (FusedBatch& b : P.batches) b.desc.vec = vector_width(b, 0);  // 1 when a table dimension is innermost
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
      int64_t& link = P.h2d ? base.src : base.dst;
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
    if (g.opaque()) return fused_plan(g).ok;  // its only way to run
    if (g.forms.empty() || g.runs.empty()) return false;
    const i64 avg = g.runs.total() / static_cast<i64>(g.runs.size());  // bytes for pure groups, else elements
    return avg < (g.pure() ? 256 : 64) && fused_plan(g).ok;
  }

  int64_t fused(Dev& s, const Group& g, const std::vector<Op>& ops, CUstream st) {
    const FusedPlan& P = fused_plan(g);
    if (!P.ok) fail(Errc::unsupported, "fused strategy is not applicable to this group");
    const cv::Fn fn = to_cv(g.fn);
    CUdeviceptr stg = 0;
    std::vector<Raw> raws;
    const Side dev_side{P.h2d ? g.dst.device : g.src.device, DType::u8, false};
    const Side host_side{{DeviceKind::host, 0}, DType::u8, P.h2d ? g.src.pinned : g.dst.pinned,
                         P.h2d ? g.src.mapped : g.dst.mapped};
    if (P.cross) {
      stg = staging(s, P.stg_bytes, st);
      raws = P.raws;
      for (Raw& r : raws) (P.h2d ? r.dst : r.src) += stg;
      if (P.h2d) copy_raws(s, host_side, dev_side, raws, st);
    }
    for (const FusedBatch& b : P.batches) {
      // Cross-device: the link side's indices are relative to the staging base.
      const long long add = P.cross ? static_cast<long long>(stg) / b.link_size : 0;
      if (b.device_tables) {  // this call's table pointers: the bases go up each call
        std::vector<Base> bases = b.bases;
        for (size_t i = 0; i < bases.size(); ++i) {
          const Op& op = ops[g.ops[b.members[i]]];
          bases[i].stab = op.stab, bases[i].dtab = op.dtab;
        }
        if (b.stab_host || b.dtab_host) pack_host_tables(s, const_cast<FusedBatch&>(b), g, ops, bases, st);
        if (capturing(reinterpret_cast<uintptr_t>(st))) {  // the graph owns these bases
          const CUdeviceptr t = captured_bases(s, bases, st);
          launch_stmt(t, bases.size(), b, fn, P.h2d ? add : 0, P.h2d ? 0 : add, s, st);
          continue;
        }
        launch_stmt(current_bases(s, const_cast<FusedBatch&>(b), bases, st), bases.size(), b, fn,
                    P.h2d ? add : 0, P.h2d ? 0 : add, s, st);
        continue;
      }
      auto& table = const_cast<FusedBatch&>(b).table;
      if (!table) table = resident(s, b.bases);
      const auto& t = *static_cast<DevTable*>(table.get());
      launch_stmt(t.ptr, t.count, b, fn, P.h2d ? add : 0, P.h2d ? 0 : add, s, st);
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
  // The kernel strategy with a given chunk kernel (zero-copy: streaming accesses).
  int64_t kernel_named(Dev& s, const Group& g, const std::string& name, CUstream st) {
    auto& slot = g.exec_cache[static_cast<int>(Strategy::kernel)];
    if (!slot) slot = resident(s, chunks_of(g));
    const auto& t = *static_cast<DevTable*>(slot.get());
    launch_table(s, name, t.ptr, t.count, to_cv(g.fn), 0, 0, st);
    return 1;
  }

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

  // `st`'s staging buffer: reused in stream order on its own stream; another
  // stream takes one over (and waits for its last use) only beyond kLanes.
  Staging& staging_of(Dev& s, CUstream st) {
    for (Staging& g : s.stagings)
      if (g.caller == st) return g;
    if (s.stagings.size() < kLanes) {
      Staging g;
      g.caller = st;
      check(d_->cuEventCreate(&g.done, CU_EVENT_DISABLE_TIMING), "cuEventCreate");
      s.stagings.push_back(g);
      return s.stagings.back();
    }
    Staging& g = *std::min_element(s.stagings.begin(), s.stagings.end(),
                                   [](const Staging& a, const Staging& b) { return a.last < b.last; });
    if (g.used) check(d_->cuStreamWaitEvent(st, g.done, 0), "cuStreamWaitEvent");
    g.caller = st;
    return g;
  }

  CUdeviceptr staging(Dev& s, size_t bytes, CUstream st) {
    Staging& g = staging_of(s, st);
    g.last = ++s.staging_clock;
    if (g.cap < bytes) {
      if (g.used && !g.mem.handle) check(d_->cuEventSynchronize(g.done), "cuEventSynchronize");
      dev_free(s, g.mem);  // a framework allocator's memory is ordered, not waited for
      size_t cap = 1 << 20;
      while (cap < bytes) cap <<= 1;
      cap = std::min<size_t>(cap, static_cast<size_t>(std::max<int64_t>(opt_.staging_budget, 1 << 20)));
      if (cap < bytes) fail(Errc::invalid_argument, "staging request exceeds the budget");
      g.mem = dev_alloc(s, cap, "staging", st);
      g.cap = cap;
      g.used = false;
    }
    return g.mem.ptr;
  }

  void staging_release(Dev& s, CUstream st) {
    Staging& g = staging_of(s, st);
    check(d_->cuEventRecord(g.done, st), "cuEventRecord");
    g.used = true;
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

  // ---- bandwidth limit: paced chunks ------------------------------------------------------
  //
  // A limited group is cut into chunks of about kPaceUs at the limit; before
  // each chunk a one-thread kernel waits on the GPU until the chunk's slot
  // (ce_pace). The stream carries the whole schedule: no host thread, no host
  // waits, and a CUDA graph records it like any other launch. Each chunk runs
  // the cheapest way that suits its layout: DMA (runs / batch) where rows are
  // large, else a kernel on the device or zero-copy over pinned host memory,
  // else staging / bounce buffers (pageable memory).

  static constexpr double kPaceUs = 250;  // chunk duration at the limit
  static constexpr int64_t kPaceMin = 64 << 10, kPaceMax = 16 << 20;

  struct PacedChunk {
    Group g;  // this chunk's runs, with the parent's sides and cuts (and its own tables)
    Strategy inner = Strategy::runs;
    bool zero_copy = false;     // kernel over pinned host memory
    unsigned long long ns = 0;  // the chunk's time at the limit
  };
  struct PacedPlan {
    int dir = 0;  // pacing word: 0 H2D, 1 D2H, 2 on / between GPUs
    bool has_batch = false;
    std::vector<PacedChunk> chunks;
  };

  // Zero-copy admission: a kernel copies through pinned host memory both
  // ways, and the bytes must come back exact.
  bool zero_copy_ok(Dev& s) {
    if (s.zero_copy >= 0) return s.zero_copy == 1;
    s.zero_copy = 0;
    if (!s.kernels) {
      s.zero_copy_error = "no kernels: " + s.kernel_error;
      return false;
    }
    const size_t n = 1 << 16;
    void* host = nullptr;
    CUdeviceptr dev = 0;
    try {
      check(d_->cuMemHostAlloc(&host, 2 * n, 0), "cuMemHostAlloc");
      check(d_->cuMemAlloc(&dev, n), "cuMemAlloc");
      auto* h = static_cast<unsigned char*>(host);
      for (size_t i = 0; i < n; ++i) h[i] = static_cast<unsigned char>(i * 131 + 7), h[n + i] = 0;
      const long long hp = static_cast<long long>(reinterpret_cast<uintptr_t>(host));
      const long long dp = static_cast<long long>(dev), ln = static_cast<long long>(n);
      launch(s, "ce_bytes_cs", {Chunk{hp, dp, ln, 1, 0, 0}}, {}, s.side);       // host -> device
      launch(s, "ce_bytes_cs", {Chunk{dp, hp + ln, ln, 1, 0, 0}}, {}, s.side);  // device -> host
      check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
      s.zero_copy = std::memcmp(h, h + n, n) == 0 ? 1 : 0;
      if (!s.zero_copy) s.zero_copy_error = "test copy through pinned host memory came back wrong";
    } catch (const Error& e) {
      s.zero_copy_error = e.what();
      s.zero_copy = 0;
    }
    if (dev) d_->cuMemFree(dev);
    if (host) d_->cuMemFreeHost(host);
    return s.zero_copy == 1;
  }

  PacedPlan build_paced(Dev& s, const Group& g) {
    if (!s.kernels) fail(Errc::unsupported, "a bandwidth limit needs the kernel module (pacing): " + s.kernel_error);
    if (!(g.bandwidth > 0)) fail(Errc::backend, "paced: the group has no bandwidth limit");
    const bool sc = g.src.device.kind == DeviceKind::cuda, dc = g.dst.device.kind == DeviceKind::cuda;
    const bool same_dev = sc && dc && g.src.device.index == g.dst.device.index;
    if (!g.pure() && sc && dc && !same_dev)
      fail(Errc::unsupported, "converting copies between two GPUs are not supported yet");
    if (!s.pace.ptr) {
      s.pace = dev_alloc(s, 3 * sizeof(unsigned long long), "pacing words", s.side);
      check(d_->cuMemsetD8Async(s.pace.ptr, 0, 3 * sizeof(unsigned long long), s.side), "cuMemsetD8Async");
      check(d_->cuStreamSynchronize(s.side), "cuStreamSynchronize");
    }
    PacedPlan P;
    P.dir = sc && dc ? 2 : (dc ? 0 : 1);
    // The bytes that count: the host side's across the link, the destination's on the device.
    const Side& link = sc && dc ? g.dst : (dc ? g.src : g.dst);
    const i64 lsz = static_cast<i64>(itemsize(link.dtype));
    const double bw = g.bandwidth;
    const int64_t chunk = std::clamp<int64_t>(static_cast<int64_t>(bw * kPaceUs * 1e-6), kPaceMin, kPaceMax);
    const bool host_ok = (sc || g.src.pinned) && (dc || g.dst.pinned);
    for (Runs& piece : g.runs.chunks(std::max<i64>(1, chunk / lsz))) {
      PacedChunk c;
      Group& sub = c.g;
      sub.src = g.src, sub.dst = g.dst, sub.fn = g.fn, sub.ctx = g.ctx, sub.wave = g.wave;
      sub.src_cuts = g.src_cuts, sub.dst_cuts = g.dst_cuts, sub.src_allocs = g.src_allocs;
      sub.op_runs = static_cast<int64_t>(piece.size());
      sub.rects = recover_2d(piece, 2, sub.src_cuts, sub.dst_cuts);
      sub.runs = std::move(piece);
      c.ns = static_cast<unsigned long long>(static_cast<double>(sub.runs.total() * lsz) / bw * 1e9);
      int64_t rows = 0;
      for (i64 h : sub.rects.height) rows += h;
      const bool fragmented = rows > 2 && sub.bytes() / rows < kSmallRect;
      auto zero_copy = [&] { return sc != dc && kernel_reach(g.src) && kernel_reach(g.dst) && zero_copy_ok(s); };
      if (g.pure()) {
        if (!host_ok) c.inner = Strategy::bounce;
        else if (fragmented && same_dev) c.inner = Strategy::kernel;
        else if (fragmented && zero_copy()) c.inner = Strategy::kernel, c.zero_copy = true;
        else if (sub.rects.size() > 2 && s.batch && kernel_reach(g.src) && kernel_reach(g.dst)) c.inner = Strategy::batch;
        else c.inner = Strategy::runs;
      } else if (same_dev) {
        c.inner = Strategy::kernel;
      } else if (zero_copy()) {
        c.inner = Strategy::kernel, c.zero_copy = true;
      } else {
        c.inner = Strategy::staged;
      }
      P.has_batch |= c.inner == Strategy::batch;
      P.chunks.push_back(std::move(c));
    }
    return P;
  }

  int64_t paced(Dev& s, const Group& g, CUstream st) {
    auto& slot = g.exec_cache[static_cast<int>(Strategy::paced)];
    if (!slot) slot = std::make_shared<PacedPlan>(build_paced(s, g));
    const auto& P = *static_cast<const PacedPlan*>(slot.get());
    CUfunction pace = function(s, "ce_pace");
    CUdeviceptr word = s.pace.ptr + static_cast<CUdeviceptr>(P.dir) * sizeof(unsigned long long);
    // Batch descriptors are not recorded into graphs: those chunks go as plain DMA calls.
    const bool rec = P.has_batch && capturing(reinterpret_cast<uintptr_t>(st));
    int64_t calls = 0;
    for (const PacedChunk& c : P.chunks) {
      unsigned long long ns = c.ns;
      void* args[] = {&word, &ns};
      check(d_->cuLaunchKernel(pace, 1, 1, 1, 32, 1, 1, 0, st, args, nullptr), "cuLaunchKernel ce_pace");
      ++calls;
      const Group& sub = c.g;
      switch (rec && c.inner == Strategy::batch ? Strategy::runs : c.inner) {
        case Strategy::runs: calls += runs(sub, st); break;
        case Strategy::batch: calls += batch(s, sub.src, sub.dst, cached_rows(sub, Strategy::batch), st); break;
        case Strategy::kernel:
          calls += c.zero_copy ? kernel_named(s, sub, sub.pure() ? "ce_bytes_cs" : kernel_name(sub), st)
                               : kernel(s, sub, st);
          break;
        case Strategy::staged: calls += staged(s, sub, st); break;
        case Strategy::bounce: calls += bounce(s, sub.src, sub.dst, cached_rows(sub, Strategy::bounce), st); break;
        default: fail(Errc::backend, "paced: unexpected chunk strategy");
      }
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
