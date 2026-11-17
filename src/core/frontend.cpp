#include "copyengine/frontend.hpp"

#include <atomic>
#include <span>
#include <vector>

#include "copyengine/error.hpp"
#include "copyengine/ir.hpp"

#if CE_WITH_CUDA
#include "backends/cuda/driver.hpp"
#endif

namespace ce {

HostMemory host_memory(uintptr_t p) {
#if CE_WITH_CUDA
  const cuda::Driver* d = cuda::driver(nullptr);
  if (!d || p == 0) return HostMemory::pageable;
  // The memory type needs no context; not known to the driver: pageable.
  CUmemorytype type{};
  if (d->cuPointerGetAttribute(&type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, static_cast<CUdeviceptr>(p)) != CUDA_SUCCESS ||
      type != CU_MEMORYTYPE_HOST)
    return HostMemory::pageable;
  // The device address is a context's view: without one current on this thread
  // (a fresh worker thread), ask in GPU 0's primary context (retained once).
  CUcontext cur = nullptr;
  bool pushed = false;
  if (d->cuCtxGetCurrent(&cur) != CUDA_SUCCESS || !cur) {
    static CUcontext primary = [d] {
      CUdevice dev0 = 0;
      CUcontext c = nullptr;
      if (d->cuDeviceGet(&dev0, 0) != CUDA_SUCCESS || d->cuDevicePrimaryCtxRetain(&c, dev0) != CUDA_SUCCESS) c = nullptr;
      return c;
    }();
    pushed = primary && d->cuCtxPushCurrent(primary) == CUDA_SUCCESS;
  }
  CUdeviceptr dev = 0;
  const CUresult r = d->cuPointerGetAttribute(&dev, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, static_cast<CUdeviceptr>(p));
  if (pushed) {
    CUcontext popped = nullptr;
    d->cuCtxPopCurrent(&popped);
  }
  return r == CUDA_SUCCESS && dev == static_cast<CUdeviceptr>(p) ? HostMemory::mapped : HostMemory::pinned;
#else
  (void)p;
  return HostMemory::pageable;
#endif
}

namespace {
std::atomic<const DeviceAllocator*> g_allocator{nullptr};
std::atomic<StreamProvider> g_streams{nullptr};
std::atomic<PeerEnabler> g_peers{nullptr};
}  // namespace

void set_peer_enabler(PeerEnabler f) { g_peers.store(f); }
PeerEnabler peer_enabler() { return g_peers.load(); }

namespace {
thread_local bool t_enabling_peer = false;
}
void run_peer_enabler(int a, int b) {
  const PeerEnabler f = peer_enabler();
  if (!f) return;
  t_enabling_peer = true;
  f(a, b);  // noexcept
  t_enabling_peer = false;
}
bool enabling_peer() { return t_enabling_peer; }

void set_stream_provider(StreamProvider f) { g_streams.store(f); }
StreamProvider stream_provider() { return g_streams.load(); }

void set_device_allocator(const DeviceAllocator* a) { g_allocator.store(a); }
const DeviceAllocator* device_allocator() { return g_allocator.load(); }

namespace {

int copy_strided(Engine& engine, const Endpoint& src, const Endpoint& dst, const int64_t* s_shape,
                 const int64_t* s_strides, int s_rank, int64_t s_off, const int64_t* d_shape, const int64_t* d_strides,
                 int d_rank, int64_t d_off, const Elementwise& fn, std::shared_ptr<NativeOp> native, uint32_t flags,
                 uintptr_t stream, std::string* error) {
  try {
    const std::span<const i64> ds(d_shape, static_cast<size_t>(d_rank));
    Access da = Access::strided(ds, std::span<const i64>(d_strides, static_cast<size_t>(d_rank)), d_off);
    Access sa = broadcast_to(Access::strided(std::span<const i64>(s_shape, static_cast<size_t>(s_rank)),
                                             std::span<const i64>(s_strides, static_cast<size_t>(s_rank)), s_off),
                             ds);
    Hints h;
    h.native = std::move(native);
    h.host_async = (flags & FrontendApi::kHostAsync) != 0;
    engine.copy(src, dst, Stmt(std::move(da), std::move(sa)), fn, h, stream);
    return 0;
  } catch (const Error& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(e.code());
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(Errc::backend);
  }
}

int copy_indexed(Engine& engine, const Endpoint& src, const Endpoint& dst, const int64_t* s_shape,
                 const int64_t* s_strides, int s_rank, int64_t s_off, const int64_t* d_shape, const int64_t* d_strides,
                 int d_rank, int64_t d_off, int side, int dim, uintptr_t table, int64_t n, int table_device,
                 const Elementwise& fn, std::shared_ptr<NativeOp> native, uint32_t flags, uintptr_t stream,
                 std::string* error) {
  try {
    Access da = Access::strided(std::span<const i64>(d_shape, static_cast<size_t>(d_rank)),
                                std::span<const i64>(d_strides, static_cast<size_t>(d_rank)), d_off);
    Access sa = Access::strided(std::span<const i64>(s_shape, static_cast<size_t>(s_rank)),
                                std::span<const i64>(s_strides, static_cast<size_t>(s_rank)), s_off);
    if (side == 0) {
      sa = sa.index_select_device(dim, table, n, table_device);
      if (sa.shape() != da.shape()) fail(Errc::invalid_argument, "gather: output shape does not match");
    } else {
      da = da.index_select_device(dim, table, n, table_device);
      sa = broadcast_to(sa, da.shape());
    }
    Hints h;
    h.native = std::move(native);
    h.host_async = (flags & FrontendApi::kHostAsync) != 0;
    engine.copy(src, dst, Stmt(std::move(da), std::move(sa)), fn, h, stream);
    return 0;
  } catch (const Error& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(e.code());
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(Errc::backend);
  }
}

int fence(Engine& engine, const Range* reads, size_t n_reads, const Range* writes, size_t n_writes,
          std::string* error) {
  try {
    engine.fence(std::span<const Range>(reads, n_reads), std::span<const Range>(writes, n_writes));
    return 0;
  } catch (const Error& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(e.code());
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(Errc::backend);
  }
}

size_t queued(const Engine& engine) { return engine.queued(); }

void decline(Engine& engine, const char* reason) {
  try {
    engine.decline(reason);
  } catch (...) {  // NOLINT(bugprone-empty-catch): statistics only, never fails the copy
  }
}

Engine* shared_engine() {
  static Engine* e = new Engine(EngineOptions::from_env());  // lives as long as the process
  return e;
}

int tile(Engine& engine, const TileSpec& spec, uintptr_t stream, std::string* error) {
  try {
    engine.tile(spec, {}, stream);
    return 0;
  } catch (const Error& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(e.code());
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(Errc::backend);
  }
}

void begin_scope(Engine& engine) { engine.begin_scope(); }

int end_scope(Engine& engine, std::string* error) {
  try {
    engine.end_scope();
    return 0;
  } catch (const Error& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(e.code());
  } catch (const std::exception& e) {
    if (error) *error = e.what();
    return 1 + static_cast<int>(Errc::backend);
  }
}

}  // namespace

const FrontendApi* frontend_api() {
  static const FrontendApi api = [] {
    FrontendApi a;
    a.copy_strided = &copy_strided;
    a.host_memory = &host_memory;
    a.fence = &fence;
    a.queued = &queued;
    a.decline = &decline;
    a.set_device_allocator = &set_device_allocator;
    a.set_stream_provider = &set_stream_provider;
    a.set_peer_enabler = &set_peer_enabler;
    a.copy_indexed = &copy_indexed;
    a.shared_engine = &shared_engine;
    a.tile = &tile;
    a.begin_scope = &begin_scope;
    a.end_scope = &end_scope;
    return a;
  }();
  return &api;
}

}  // namespace ce
