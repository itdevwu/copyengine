#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "copyengine/engine.hpp"

namespace ce {

// Whether host memory at `p` is page-locked by the CUDA driver (false
// without a CUDA driver).
bool host_is_pinned(uintptr_t p);

// Entry points for framework extensions (copyengine._torch). They receive this
// table from copyengine._core, so the process holds one copy of the engine and
// extensions need not link the core. Both sides come from the same build;
// errors cross as codes because exception types do not cross shared objects.
struct FrontendApi {
  static constexpr uint32_t kVersion = 3;
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
  bool (*host_is_pinned)(uintptr_t p) = nullptr;
  // Engine::fence and Engine::queued.
  int (*fence)(Engine& engine, const Range* reads, size_t n_reads, const Range* writes, size_t n_writes,
               std::string* error) = nullptr;
  size_t (*queued)(const Engine& engine) = nullptr;
  // Engine::decline.
  void (*decline)(Engine& engine, const char* reason) = nullptr;
};

const FrontendApi* frontend_api();

}  // namespace ce
