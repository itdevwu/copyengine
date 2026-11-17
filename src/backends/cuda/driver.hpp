#pragma once

// CUDA driver API loaded at runtime. copyengine never links libcuda or cudart:
// every entry point is fetched with cuGetProcAddress at CUDA_VERSION of the
// header we compile against. The function-pointer types also come from that
// header, so signature and ABI always match, even on newer drivers where an
// entry point changed shape (cuMemcpyBatchAsync did in 13.0).

#include <cuda.h>

#include <string>

namespace ce::cuda {

#define CE_CUDA_REQUIRED(X)            \
  X(cuDriverGetVersion)         \
  X(cuGetErrorName)             \
  X(cuDeviceGetCount)           \
  X(cuDeviceGet)                \
  X(cuDeviceGetName)            \
  X(cuDeviceGetAttribute)       \
  X(cuDeviceTotalMem)           \
  X(cuDeviceGetPCIBusId)        \
  X(cuDevicePrimaryCtxRetain)   \
  X(cuDevicePrimaryCtxRelease)  \
  X(cuCtxPushCurrent)           \
  X(cuCtxPopCurrent)            \
  X(cuCtxGetCurrent)            \
  X(cuCtxGetDevice)             \
  X(cuStreamGetCtx)             \
  X(cuDeviceCanAccessPeer)      \
  X(cuDeviceGetP2PAttribute)    \
  X(cuCtxEnablePeerAccess)      \
  X(cuCtxGetStreamPriorityRange) \
  X(cuStreamCreate)             \
  X(cuStreamDestroy)            \
  X(cuStreamSynchronize)        \
  X(cuCtxSynchronize)           \
  X(cuEventCreate)              \
  X(cuEventDestroy)             \
  X(cuEventRecord)              \
  X(cuEventSynchronize)         \
  X(cuEventElapsedTime)         \
  X(cuMemAlloc)                 \
  X(cuMemFree)                  \
  X(cuMemHostAlloc)             \
  X(cuMemFreeHost)              \
  X(cuMemsetD8Async)            \
  X(cuMemcpyHtoDAsync)          \
  X(cuMemcpyDtoHAsync)          \
  X(cuMemcpyDtoDAsync)          \
  X(cuMemcpyAsync)              \
  X(cuMemcpy2DAsync)            \
  X(cuStreamWaitEvent)          \
  X(cuStreamIsCapturing)        \
  X(cuEventQuery)               \
  X(cuPointerGetAttribute)      \
  X(cuModuleLoadData)           \
  X(cuModuleGetFunction)        \
  X(cuModuleGetGlobal)          \
  X(cuLaunchKernel)

// Present only on newer drivers; null when missing.
#define CE_CUDA_OPTIONAL(X)              \
  X(cuMemcpyBatchAsync)           \
  X(cuDeviceGetDevResource)       \
  X(cuDevSmResourceSplitByCount)  \
  X(cuDevResourceGenerateDesc)    \
  X(cuGreenCtxCreate)             \
  X(cuGreenCtxDestroy)            \
  X(cuGreenCtxStreamCreate)       \
  X(cuTensorMapEncodeTiled)

struct Driver {
#define CE_CUDA_DECL(name) decltype(&::name) name = nullptr;
  CE_CUDA_REQUIRED(CE_CUDA_DECL)
  CE_CUDA_OPTIONAL(CE_CUDA_DECL)
#undef CE_CUDA_DECL
};

// cuMemcpyBatchAsync lost its failIdx parameter in CUDA 13.0: one call site
// for both header generations (`fail_idx` is left unset by 13.x).
template <class D, class S, class Z, class A, class I>
CUresult memcpy_batch(const Driver& d, D dsts, S srcs, Z sizes, size_t count, A attrs, I attr_idxs, size_t n_attrs,
                      size_t* fail_idx, CUstream st) {
#if CUDA_VERSION >= 13000
  (void)fail_idx;
  return d.cuMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attr_idxs, n_attrs, st);
#else
  return d.cuMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attr_idxs, n_attrs, fail_idx, st);
#endif
}

// Loads and initializes the driver once. Returns null and fills *error on failure.
const Driver* driver(std::string* error);

// Why an optional entry point is null ("symbol not found", "driver too old", ...).
std::string missing_reason(const char* symbol);

std::string error_name(CUresult r);

}  // namespace ce::cuda
