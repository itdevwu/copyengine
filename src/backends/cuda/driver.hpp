#pragma once

// CUDA driver API loaded at runtime. copyengine never links libcuda or cudart.
// Every entry point is listed with the version of its prototype: its type is
// cudaTypedefs.h's PFN_cuX_vNNNNN and it is fetched with cuGetProcAddress at
// exactly that version, so what is called is the prototype that was fetched,
// whatever the headers and the driver. (A name's prototype in cuda.h is not
// what the driver returns for it at a newer version: cuStreamGetCtx stays
// two-argument in cuda.h, while from 12.5 the driver's newest is the
// three-argument _v2; CUDA 13's cudaTypedefs.h has only versioned names.)
// Versions are chosen, not the headers' newest, and exist in the 12.x and
// 13.x headers alike: the original cuStreamGetCtx (copyengine needs no green
// context), cuMemcpyBatchAsync with failIdx (12.8; 13.0 dropped it).

#include <cuda.h>
#include <cudaTypedefs.h>

#include <string>

namespace ce::cuda {

#define CE_CUDA_REQUIRED(X)            \
  X(cuDriverGetVersion, 2020)         \
  X(cuGetErrorName, 6000)             \
  X(cuDeviceGetCount, 2000)           \
  X(cuDeviceGet, 2000)                \
  X(cuDeviceGetName, 2000)            \
  X(cuDeviceGetAttribute, 2000)       \
  X(cuDeviceTotalMem, 3020)           \
  X(cuDeviceGetPCIBusId, 4010)        \
  X(cuDevicePrimaryCtxRetain, 7000)   \
  X(cuDevicePrimaryCtxRelease, 11000)  \
  X(cuCtxPushCurrent, 4000)           \
  X(cuCtxPopCurrent, 4000)            \
  X(cuCtxGetCurrent, 4000)            \
  X(cuCtxGetDevice, 2000)             \
  X(cuStreamGetCtx, 9020)             \
  X(cuDeviceCanAccessPeer, 4000)      \
  X(cuDeviceGetP2PAttribute, 8000)    \
  X(cuCtxEnablePeerAccess, 4000)      \
  X(cuCtxGetStreamPriorityRange, 5050) \
  X(cuStreamCreate, 2000)             \
  X(cuStreamDestroy, 4000)            \
  X(cuStreamSynchronize, 2000)        \
  X(cuCtxSynchronize, 2000)           \
  X(cuEventCreate, 2000)              \
  X(cuEventDestroy, 4000)             \
  X(cuEventRecord, 2000)              \
  X(cuEventSynchronize, 2000)         \
  X(cuEventElapsedTime, 2000)         \
  X(cuMemAlloc, 3020)                 \
  X(cuMemFree, 3020)                  \
  X(cuMemHostAlloc, 2020)             \
  X(cuMemFreeHost, 2000)              \
  X(cuMemsetD8Async, 3020)            \
  X(cuMemcpyHtoDAsync, 3020)          \
  X(cuMemcpyDtoHAsync, 3020)          \
  X(cuMemcpyDtoDAsync, 3020)          \
  X(cuMemcpyAsync, 4000)              \
  X(cuMemcpy2DAsync, 3020)            \
  X(cuStreamWaitEvent, 3020)          \
  X(cuStreamIsCapturing, 10000)        \
  X(cuEventQuery, 2000)               \
  X(cuPointerGetAttribute, 4000)      \
  X(cuModuleLoadData, 2000)           \
  X(cuModuleGetFunction, 2000)        \
  X(cuModuleGetGlobal, 3020)          \
  X(cuLaunchKernel, 4000)

// Present only on newer drivers; null when missing.
#define CE_CUDA_OPTIONAL(X)              \
  X(cuMemcpyBatchAsync, 12080)           \
  X(cuDeviceGetDevResource, 12040)       \
  X(cuDevSmResourceSplitByCount, 12040)  \
  X(cuDevResourceGenerateDesc, 12040)    \
  X(cuGreenCtxCreate, 12040)             \
  X(cuGreenCtxDestroy, 12040)            \
  X(cuGreenCtxStreamCreate, 12050)       \
  X(cuTensorMapEncodeTiled, 12000)

struct Driver {
#define CE_CUDA_DECL(name, version) PFN_##name##_v##version name = nullptr;
  CE_CUDA_REQUIRED(CE_CUDA_DECL)
  CE_CUDA_OPTIONAL(CE_CUDA_DECL)
#undef CE_CUDA_DECL
};

// cuMemcpyBatchAsync at 12080 (with failIdx), with 12.x and 13.x headers alike.
template <class D, class S, class Z, class A, class I>
CUresult memcpy_batch(const Driver& d, D dsts, S srcs, Z sizes, size_t count, A attrs, I attr_idxs, size_t n_attrs,
                      size_t* fail_idx, CUstream st) {
  return d.cuMemcpyBatchAsync(dsts, srcs, sizes, count, attrs, attr_idxs, n_attrs, fail_idx, st);
}

// Loads and initializes the driver once. Returns null and fills *error on failure.
const Driver* driver(std::string* error);

// Why an optional entry point is null ("symbol not found", "driver too old", ...).
std::string missing_reason(const char* symbol);

std::string error_name(CUresult r);

}  // namespace ce::cuda
