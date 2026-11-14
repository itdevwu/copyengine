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

bool host_is_pinned(uintptr_t p) {
#if CE_WITH_CUDA
  const cuda::Driver* d = cuda::driver(nullptr);
  if (!d || p == 0) return false;
  CUmemorytype t{};
  return d->cuPointerGetAttribute(&t, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, static_cast<CUdeviceptr>(p)) == CUDA_SUCCESS &&
         t == CU_MEMORYTYPE_HOST;
#else
  (void)p;
  return false;
#endif
}

namespace {
std::atomic<const DeviceAllocator*> g_allocator{nullptr};
}  // namespace

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
  } catch (...) {  // statistics only: never fails the copy
  }
}

}  // namespace

const FrontendApi* frontend_api() {
  static const FrontendApi api = [] {
    FrontendApi a;
    a.copy_strided = &copy_strided;
    a.host_is_pinned = &host_is_pinned;
    a.fence = &fence;
    a.queued = &queued;
    a.decline = &decline;
    a.set_device_allocator = &set_device_allocator;
    return a;
  }();
  return &api;
}

}  // namespace ce
