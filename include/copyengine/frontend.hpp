#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "copyengine/engine.hpp"

namespace ce {

// What the driver says of host memory at `p` (one query): pageable, pinned
// (DMA reaches it), or mapped (kernels too, at `p`). The one source of
// Endpoint::pinned / mapped for every frontend.
HostMemory host_memory(uintptr_t p);
inline bool host_is_pinned(uintptr_t p) { return host_memory(p) != HostMemory::pageable; }
// Whether torch's own copy between these endpoints is one memcpy in stream
// order (a GPU and pinned host memory, one dtype, both dense with one shape);
// otherwise it touches host memory on the CPU when called (NativeOp::cpu_host_access).
inline bool torch_copy_stream_ordered(const Endpoint& dst, const Endpoint& src, bool same_dtype, bool dense) {
  const bool dc = dst.device.kind == DeviceKind::cuda, sc = src.device.kind == DeviceKind::cuda;
  if (dc && sc) return true;
  if (!dc && !sc) return false;
  return same_dtype && dense && (dc ? src.pinned : dst.pinned);
}

// An endpoint's host memory facts from the driver (host endpoints only).
inline void set_host_memory(Endpoint& e) {
  const HostMemory m = host_memory(e.base);
  e.pinned = m != HostMemory::pageable, e.mapped = m == HostMemory::mapped;
}

// Where copyengine's own device memory (staging, resident tables, upload
// buffers) and pinned host memory (bounce and upload buffers) come from, when
// a framework provides it: then it counts in the framework's statistics and
// limits. Process-wide; null means cuMemAlloc / cuMemHostAlloc.
struct DeviceAllocator {
  // `bytes` on `device`; returns the address (0 when out of memory), the
  // owning handle, and the stream the allocator associated it with (memory is
  // reused only by later allocations on that stream).
  uintptr_t (*alloc)(int device, size_t bytes, void** handle, uintptr_t* stream) = nullptr;
  void (*free)(void* handle) = nullptr;
  // Pinned host memory (null: cuMemHostAlloc). copyengine frees it only after
  // its own work on it is done, so the framework may reuse it at once.
  void* (*host_alloc)(size_t bytes, void** handle) = nullptr;
  void (*host_free)(void* handle) = nullptr;
  const char* name = "";
};
void set_device_allocator(const DeviceAllocator* a);
const DeviceAllocator* device_allocator();

// The framework's current stream of GPU `device` (torch's current CUDA
// stream): a copy between two GPUs runs on the source GPU's, ordered after
// the work queued on both GPUs' and before their later work, as torch's own
// copy. Returns false when it cannot tell. Without a provider (the C ABI), a
// copy between two GPUs runs on the stream it is given, ordered with nothing
// else: the caller orders the other GPU's work.
using StreamProvider = bool (*)(int device, uintptr_t* stream);
void set_stream_provider(StreamProvider f);
StreamProvider stream_provider();

// Makes the framework give GPU `a` access to GPU `b`'s memory as for its own
// copies (torch's copy from a to b enables it and maps b's expandable segments
// for a), before copyengine's kernels on `a` touch b's memory. Without one (the
// C ABI), memory from cuMemCreate must be mapped for `a` by the caller
// (cuMemSetAccess); copyengine enables peer access for cuMemAlloc'ed memory.
using PeerEnabler = void (*)(int a, int b);
void set_peer_enabler(PeerEnabler f);
PeerEnabler peer_enabler();
void run_peer_enabler(int a, int b);

// While copyengine runs the framework's own copy on a thread (a native: the
// baseline, a copy torch does better, the peer enabler's), nothing is
// intercepted on that thread: interception (the shim, the Python mode) lets
// every op through to the framework, and an engine runs a copy handed to it
// as its native at once. Otherwise the framework's copy, intercepted inside a
// scope, would come back into the engine and be queued instead of done.
class FrameworkCopy {
 public:
  FrameworkCopy();
  ~FrameworkCopy();
  FrameworkCopy(const FrameworkCopy&) = delete;
  FrameworkCopy& operator=(const FrameworkCopy&) = delete;

 private:
  bool was_;
};
bool framework_copy_running();

// Entry points for framework extensions (copyengine._torch). They receive this
// table from copyengine._core, so the process holds one copy of the engine and
// extensions need not link the core. Both sides come from the same build;
// errors cross as codes because exception types do not cross shared objects.
struct FrontendApi {
  static constexpr uint32_t kVersion = 9;
  enum Flags : uint32_t {
    kHostAsync = 1,  // see Hints::host_async
  };
  uint32_t version = kVersion;
  // dst = fn(src broadcast to the destination shape). Shapes, strides and
  // offsets are in elements of each endpoint's dtype, relative to its base.
  // Returns 0, or 1 + Errc with the message in `error`.
  int (*copy_strided)(Engine& engine, const Endpoint& src, const Endpoint& dst, const int64_t* s_shape,
                      const int64_t* s_strides, int s_rank, int64_t s_off, const int64_t* d_shape,
                      const int64_t* d_strides, int d_rank, int64_t d_off, const Elementwise& fn,
                      std::shared_ptr<NativeOp> native, uint32_t flags, uintptr_t stream,
                      std::string* error) = nullptr;
  HostMemory (*host_memory)(uintptr_t p) = nullptr;
  // Engine::fence and Engine::queued.
  int (*fence)(Engine& engine, const Range* reads, size_t n_reads, const Range* writes, size_t n_writes,
               std::string* error) = nullptr;
  size_t (*queued)(const Engine& engine) = nullptr;
  // Engine::decline.
  void (*decline)(Engine& engine, const char* reason) = nullptr;
  void (*set_device_allocator)(const DeviceAllocator* a) = nullptr;
  // copy_strided with one side indexed along `dim` by `n` int64 indices in
  // device memory at `table` on GPU `table_device` (Access::index_select_device):
  // `side` 0 indexes the source (gather; the source shape is before indexing),
  // 1 the destination (scatter; the source broadcasts to the indexed shape).
  int (*copy_indexed)(Engine& engine, const Endpoint& src, const Endpoint& dst, const int64_t* s_shape,
                      const int64_t* s_strides, int s_rank, int64_t s_off, const int64_t* d_shape,
                      const int64_t* d_strides, int d_rank, int64_t d_off, int side, int dim, uintptr_t table,
                      int64_t n, int table_device, const Elementwise& fn, std::shared_ptr<NativeOp> native,
                      uint32_t flags, uintptr_t stream, std::string* error) = nullptr;
  // The process-wide engine of calls that name none (engine handle 0: the
  // explicit API inside torch.compile regions), created on first use.
  Engine* (*shared_engine)() = nullptr;
  // Engine::tile (quantization while copying).
  int (*tile)(Engine& engine, const TileSpec& spec, uintptr_t stream, std::string* error) = nullptr;
  // Engine::begin_scope / end_scope: calls between them plan together.
  void (*begin_scope)(Engine& engine) = nullptr;
  int (*end_scope)(Engine& engine, std::string* error) = nullptr;
  void (*set_stream_provider)(StreamProvider f) = nullptr;
  void (*set_peer_enabler)(PeerEnabler f) = nullptr;
  bool (*framework_copy_running)() = nullptr;
};

const FrontendApi* frontend_api();

}  // namespace ce
