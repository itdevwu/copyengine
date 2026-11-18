// The driver table: every entry point is called with the prototype it was
// fetched at (cuStreamGetCtx: a newer version has another argument; calling
// it with the old one wrote through a garbage pointer). Skips without a GPU.
#include <cstdio>
#include <string>

#include "backends/cuda/driver.hpp"

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
      return 1;                                                              \
    }                                                                        \
  } while (0)

int main() {
  std::string err;
  const ce::cuda::Driver* d = ce::cuda::driver(&err);
  if (!d) {
    std::printf("skipped: %s\n", err.c_str());
    return 0;
  }
  int n = 0;
  CHECK(d->cuDeviceGetCount(&n) == CUDA_SUCCESS);
  if (n == 0) {
    std::printf("skipped: no CUDA device\n");
    return 0;
  }
  CUdevice dev{};
  CUcontext ctx{}, got{}, popped{};
  CUstream st{};
  CHECK(d->cuDeviceGet(&dev, 0) == CUDA_SUCCESS);
  CHECK(d->cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS);
  CHECK(d->cuCtxPushCurrent(ctx) == CUDA_SUCCESS);
  CHECK(d->cuStreamCreate(&st, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);
  CHECK(d->cuStreamGetCtx(st, &got) == CUDA_SUCCESS);
  CHECK(got == ctx);
  CUdevice back{};
  CHECK(d->cuCtxGetDevice(&back) == CUDA_SUCCESS);
  CHECK(back == dev);
  if (n > 1) {
    CUdevice other{};
    int can = -1, rank = -1;
    CHECK(d->cuDeviceGet(&other, 1) == CUDA_SUCCESS);
    CHECK(d->cuDeviceCanAccessPeer(&can, dev, other) == CUDA_SUCCESS);
    CHECK(can == 0 || can == 1);
    CHECK(d->cuDeviceGetP2PAttribute(&rank, CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK, dev, other) == CUDA_SUCCESS);
  }
  CHECK(d->cuStreamDestroy(st) == CUDA_SUCCESS);
  CHECK(d->cuCtxPopCurrent(&popped) == CUDA_SUCCESS);
  CHECK(d->cuDevicePrimaryCtxRelease(dev) == CUDA_SUCCESS);
  std::printf("driver table: ok\n");
  return 0;
}
