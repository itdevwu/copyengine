// CUDA backend probe: per device, check each copy-related capability by
// actually using it and verifying the bytes, then optionally microbenchmark.

#include "backends/cuda/probe.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#include "backends/cuda/driver.hpp"

namespace ce::cuda {
namespace {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void check(CUresult r, const char* call) {
  if (r == CUDA_SUCCESS) return;
  std::string what(call);
  what = what.substr(0, what.find('('));  // "d.cuMemAlloc(&p, n)" -> "d.cuMemAlloc"
  what = what.substr(what.find_last_of(".>") + 1);
  throw Error(what + ": " + error_name(r));
}
#define CE_CU(call) check((call), #call)

class Defer {
 public:
  explicit Defer(std::function<void()> f) : f_(std::move(f)) {}
  ~Defer() { f_(); }
  Defer(const Defer&) = delete;
  Defer& operator=(const Defer&) = delete;

 private:
  std::function<void()> f_;
};

// The "arange probe": a position-dependent byte pattern so misplaced or
// missing bytes are always detected.
uint8_t pattern(size_t i, uint8_t seed) { return static_cast<uint8_t>(i * 131u + seed); }

void fill(uint8_t* p, size_t n, uint8_t seed) {
  for (size_t i = 0; i < n; ++i) p[i] = pattern(i, seed);
}

Json ok(Json detail = nullptr) {
  Json j = Json::object().set("ok", true);
  if (!detail.is_null()) j.set("detail", std::move(detail));
  return j;
}

Json fail(std::string why) { return Json::object().set("ok", false).set("reason", std::move(why)); }

template <class F>
Json feature(F&& f) {
  try {
    return f();
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

struct Dev {
  const Driver& d;
  CUdevice dev;
  CUstream stream = nullptr;
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

// Wall time of fn() + stream sync, median over reps, in seconds.
double time_s(const Dev& x, const std::function<void()>& fn, int reps = 5) {
  fn();
  CE_CU(x.d.cuStreamSynchronize(x.stream));
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    fn();
    CE_CU(x.d.cuStreamSynchronize(x.stream));
    t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
  return median(std::move(t));
}

// ---- features ---------------------------------------------------------------

Json probe_async_copy(const Dev& x) {
  const size_t n = 1 << 20;
  const Driver& d = x.d;
  void* h = nullptr;
  CE_CU(d.cuMemHostAlloc(&h, 2 * n, 0));
  Defer fh([&] { d.cuMemFreeHost(h); });
  CUdeviceptr dp = 0;
  CE_CU(d.cuMemAlloc(&dp, n));
  Defer fd([&] { d.cuMemFree(dp); });
  auto* src = static_cast<uint8_t*>(h);
  auto* back = src + n;
  fill(src, n, 7);
  std::memset(back, 0, n);
  CE_CU(d.cuMemcpyHtoDAsync(dp, src, n, x.stream));
  CE_CU(d.cuMemcpyDtoHAsync(back, dp, n, x.stream));
  CE_CU(d.cuStreamSynchronize(x.stream));
  if (std::memcmp(src, back, n) != 0) return fail("pinned H2D/D2H round trip corrupted data");
  return ok();
}

Json probe_memcpy_batch(const Dev& x) {
  const Driver& d = x.d;
  if (!d.cuMemcpyBatchAsync) return fail("cuMemcpyBatchAsync: " + missing_reason("cuMemcpyBatchAsync"));
  // Scatter 64 host segments of 3000 B into a device buffer at a 4096 B pitch,
  // in reverse order, then read everything back and check gaps stayed zero.
  const size_t segs = 64, len = 3000, pitch = 4096, total = segs * pitch;
  void* h = nullptr;
  CE_CU(d.cuMemHostAlloc(&h, segs * len + total, 0));
  Defer fh([&] { d.cuMemFreeHost(h); });
  CUdeviceptr dp = 0;
  CE_CU(d.cuMemAlloc(&dp, total));
  Defer fd([&] { d.cuMemFree(dp); });
  auto* src = static_cast<uint8_t*>(h);
  auto* back = src + segs * len;
  fill(src, segs * len, 11);
  CE_CU(d.cuMemsetD8Async(dp, 0, total, x.stream));

  std::vector<CUdeviceptr> dsts(segs), srcs(segs);
  std::vector<size_t> sizes(segs, len);
  for (size_t i = 0; i < segs; ++i) {
    srcs[i] = reinterpret_cast<CUdeviceptr>(src + i * len);
    dsts[i] = dp + (segs - 1 - i) * pitch;
  }
  CUmemcpyAttributes attr{};
  attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
  attr.srcLocHint.type = CU_MEM_LOCATION_TYPE_HOST;
  attr.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
  attr.dstLocHint.id = static_cast<int>(x.dev);
  size_t attr_idx = 0, fail_idx = SIZE_MAX;
  CUresult r = d.cuMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), segs, &attr, &attr_idx,
                                    1, &fail_idx, x.stream);
  if (r != CUDA_SUCCESS) {
    std::string why = "cuMemcpyBatchAsync: " + error_name(r);
    if (fail_idx != SIZE_MAX) why += " (failIdx " + std::to_string(fail_idx) + ")";
    return fail(why);
  }
  CE_CU(d.cuMemcpyDtoHAsync(back, dp, total, x.stream));
  CE_CU(d.cuStreamSynchronize(x.stream));
  for (size_t i = 0; i < segs; ++i) {
    const uint8_t* got = back + (segs - 1 - i) * pitch;
    if (std::memcmp(got, src + i * len, len) != 0) return fail("batch copy wrote wrong bytes");
    for (size_t k = len; k < pitch; ++k)
      if (got[k] != 0) return fail("batch copy wrote outside its segments");
  }
  return ok();
}

Json probe_green_ctx(const Dev& x) {
  const Driver& d = x.d;
  for (const char* s : {"cuDeviceGetDevResource", "cuDevSmResourceSplitByCount",
                        "cuDevResourceGenerateDesc", "cuGreenCtxCreate", "cuGreenCtxDestroy",
                        "cuGreenCtxStreamCreate"}) {
    if (auto why = missing_reason(s); !why.empty()) return fail(std::string(s) + ": " + why);
  }
  CUdevResource all{};
  CE_CU(d.cuDeviceGetDevResource(x.dev, &all, CU_DEV_RESOURCE_TYPE_SM));
  const unsigned total = all.sm.smCount;
  unsigned want = std::max(2u, (total / 4) & ~1u);  // a copy-sized slice; even for sm_8x granularity
  CUdevResource part{}, rest{};
  unsigned groups = 1;
  CE_CU(d.cuDevSmResourceSplitByCount(&part, &groups, &all, &rest, 0, want));
  if (groups != 1) return fail("SM split produced no group");
  CUdevResourceDesc desc{};
  CE_CU(d.cuDevResourceGenerateDesc(&desc, &part, 1));
  CUgreenCtx g{};
  CE_CU(d.cuGreenCtxCreate(&g, desc, x.dev, CU_GREEN_CTX_DEFAULT_STREAM));
  Defer fg([&] { d.cuGreenCtxDestroy(g); });
  CUstream s{};
  CE_CU(d.cuGreenCtxStreamCreate(&s, g, CU_STREAM_NON_BLOCKING, 0));
  Defer fs([&] { d.cuStreamDestroy(s); });
  CE_CU(d.cuStreamSynchronize(s));
  return ok(Json::object()
                .set("sm_total", total)
                .set("sm_requested", want)
                .set("sm_granted", part.sm.smCount)
                .set("note", "created and stream usable; SM confinement not yet verified by a kernel"));
}

Json probe_priority(const Dev& x) {
  int least = 0, greatest = 0;
  CE_CU(x.d.cuCtxGetStreamPriorityRange(&least, &greatest));
  if (least == greatest) return fail("single stream priority level");
  return ok(Json::object().set("least", least).set("greatest", greatest));
}

// ---- PCIe link via NVML (optional; absent on some platforms) -----------------

Json probe_pcie(const char* bus_id) {
  using Ret = int;
  using Handle = void*;
  void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib) return fail("libnvidia-ml.so.1 not available");
  Defer fl([&] { dlclose(lib); });
  auto sym = [&](const char* n) { return dlsym(lib, n); };
  auto init = reinterpret_cast<Ret (*)()>(sym("nvmlInit_v2"));
  auto shutdown = reinterpret_cast<Ret (*)()>(sym("nvmlShutdown"));
  auto by_bus = reinterpret_cast<Ret (*)(const char*, Handle*)>(sym("nvmlDeviceGetHandleByPciBusId_v2"));
  using Q = Ret (*)(Handle, unsigned*);
  const char* names[] = {"nvmlDeviceGetCurrPcieLinkGeneration", "nvmlDeviceGetCurrPcieLinkWidth",
                         "nvmlDeviceGetMaxPcieLinkGeneration", "nvmlDeviceGetMaxPcieLinkWidth"};
  const char* keys[] = {"gen_current", "width_current", "gen_max", "width_max"};
  if (!init || !shutdown || !by_bus) return fail("NVML lacks required symbols");
  if (init() != 0) return fail("nvmlInit failed");
  Defer fs([&] { shutdown(); });
  Handle h = nullptr;
  if (by_bus(bus_id, &h) != 0) return fail("NVML cannot find device by PCI bus id");
  Json out = Json::object().set("ok", true);
  for (int i = 0; i < 4; ++i) {
    auto q = reinterpret_cast<Q>(sym(names[i]));
    unsigned v = 0;
    if (q && q(h, &v) == 0) out.set(keys[i], v);
    else out.set(keys[i], nullptr);
  }
  return out;
}

// ---- microbenchmarks -------------------------------------------------------

Json bench(const Dev& x, bool have_batch) {
  const Driver& d = x.d;
  const size_t big = 64u << 20;
  Json out = Json::object();

  void* pinned = nullptr;
  CE_CU(d.cuMemHostAlloc(&pinned, big, 0));
  Defer f1([&] { d.cuMemFreeHost(pinned); });
  std::vector<uint8_t> pageable(big, 1);
  CUdeviceptr a = 0, b = 0;
  CE_CU(d.cuMemAlloc(&a, big));
  Defer f2([&] { d.cuMemFree(a); });
  CE_CU(d.cuMemAlloc(&b, big));
  Defer f3([&] { d.cuMemFree(b); });
  std::memset(pinned, 1, big);

  auto gbps = [&](double s) { return big / s / 1e9; };
  Json bw = Json::object();
  bw.set("bytes", static_cast<uint64_t>(big));
  bw.set("h2d_pinned", gbps(time_s(x, [&] { CE_CU(d.cuMemcpyHtoDAsync(a, pinned, big, x.stream)); })));
  bw.set("d2h_pinned", gbps(time_s(x, [&] { CE_CU(d.cuMemcpyDtoHAsync(pinned, a, big, x.stream)); })));
  bw.set("h2d_pageable",
         gbps(time_s(x, [&] { CE_CU(d.cuMemcpyHtoDAsync(a, pageable.data(), big, x.stream)); })));
  bw.set("d2h_pageable",
         gbps(time_s(x, [&] { CE_CU(d.cuMemcpyDtoHAsync(pageable.data(), a, big, x.stream)); })));
  // D2D reads and writes: report bytes moved once (copy throughput, not DRAM traffic).
  bw.set("d2d", gbps(time_s(x, [&] { CE_CU(d.cuMemcpyDtoDAsync(b, a, big, x.stream)); })));
  out.set("bandwidth_gbps", std::move(bw));

  // The workload copyengine exists for: many small scattered H2D copies.
  Json small = Json::array();
  for (size_t seg : {size_t{4096}, size_t{65536}}) {
    const size_t count = std::min<size_t>(4096, (big / 2) / seg);
    std::vector<CUdeviceptr> dsts(count), srcs(count);
    std::vector<size_t> sizes(count, seg);
    for (size_t i = 0; i < count; ++i) {
      srcs[i] = reinterpret_cast<CUdeviceptr>(static_cast<uint8_t*>(pinned) + i * seg);
      dsts[i] = a + ((i * 7919) % count) * 2 * seg;  // scattered, non-overlapping
    }
    const double bytes = double(count) * seg;
    Json row = Json::object().set("segment_bytes", static_cast<uint64_t>(seg)).set("segments", static_cast<uint64_t>(count));
    double loop = time_s(x, [&] {
      for (size_t i = 0; i < count; ++i)
        CE_CU(d.cuMemcpyHtoDAsync(dsts[i], reinterpret_cast<void*>(srcs[i]), seg, x.stream));
    });
    double one = time_s(x, [&] { CE_CU(d.cuMemcpyHtoDAsync(a, pinned, count * seg, x.stream)); });
    row.set("loop_us", loop * 1e6).set("loop_gbps", bytes / loop / 1e9);
    if (have_batch) {
      CUmemcpyAttributes attr{};
      attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
      attr.srcLocHint.type = CU_MEM_LOCATION_TYPE_HOST;
      attr.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
      attr.dstLocHint.id = static_cast<int>(x.dev);
      size_t idx = 0, fail_idx = 0;
      double batch = time_s(x, [&] {
        CE_CU(d.cuMemcpyBatchAsync(dsts.data(), srcs.data(), sizes.data(), count, &attr, &idx, 1,
                                   &fail_idx, x.stream));
      });
      row.set("batch_us", batch * 1e6).set("batch_gbps", bytes / batch / 1e9);
    }
    row.set("contiguous_us", one * 1e6).set("contiguous_gbps", bytes / one / 1e9);
    small.push(std::move(row));
  }
  out.set("small_h2d", std::move(small));
  return out;
}

Json probe_device(const Driver& d, int ordinal, const ProbeOptions& opt) {
  CUdevice dev{};
  CE_CU(d.cuDeviceGet(&dev, ordinal));
  char name[256] = {};
  CE_CU(d.cuDeviceGetName(name, sizeof name, dev));
  auto attr = [&](CUdevice_attribute a) {
    int v = 0;
    CE_CU(d.cuDeviceGetAttribute(&v, a, dev));
    return v;
  };
  size_t mem = 0;
  CE_CU(d.cuDeviceTotalMem(&mem, dev));
  char bus[32] = {};
  CE_CU(d.cuDeviceGetPCIBusId(bus, sizeof bus, dev));

  Json j = Json::object();
  j.set("index", ordinal).set("name", name);
  j.set("compute_capability", std::to_string(attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)) + "." +
                                  std::to_string(attr(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)));
  j.set("sm_count", attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
  j.set("memory_bytes", static_cast<uint64_t>(mem));
  j.set("copy_engines", attr(CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT));
  j.set("integrated", attr(CU_DEVICE_ATTRIBUTE_INTEGRATED) != 0);
  j.set("host_register", attr(CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED) != 0);
  j.set("pcie", probe_pcie(bus));

  CUcontext ctx{};
  CE_CU(d.cuDevicePrimaryCtxRetain(&ctx, dev));
  Defer fr([&] { d.cuDevicePrimaryCtxRelease(dev); });
  CE_CU(d.cuCtxPushCurrent(ctx));
  Defer fp([&] {
    CUcontext dummy{};
    d.cuCtxPopCurrent(&dummy);
  });
  Dev x{d, dev};
  CE_CU(d.cuStreamCreate(&x.stream, CU_STREAM_NON_BLOCKING));
  Defer fs([&] { d.cuStreamDestroy(x.stream); });

  Json features = Json::object();
  features.set("async_copy", feature([&] { return probe_async_copy(x); }));
  Json batch = feature([&] { return probe_memcpy_batch(x); });
  const bool have_batch = batch.flag("ok");
  features.set("memcpy_batch", std::move(batch));
  features.set("green_ctx", feature([&] { return probe_green_ctx(x); }));
  features.set("stream_priority", feature([&] { return probe_priority(x); }));
  j.set("features", std::move(features));

  if (opt.bandwidth) j.set("bench", feature([&] { return bench(x, have_batch); }));
  return j;
}

}  // namespace

Json probe(const ProbeOptions& opt) {
  Json out = Json::object();
  std::string err;
  const Driver* d = driver(&err);
  if (!d) return out.set("available", false).set("reason", err);
  int version = 0, count = 0;
  try {
    CE_CU(d->cuDriverGetVersion(&version));
    CE_CU(d->cuDeviceGetCount(&count));
  } catch (const std::exception& e) {
    return out.set("available", false).set("reason", e.what());
  }
  out.set("available", count > 0);
  if (count == 0) out.set("reason", "no CUDA devices");
  out.set("driver_version", version);
  Json devs = Json::array();
  for (int i = 0; i < count; ++i) {
    devs.push(feature([&] { return probe_device(*d, i, opt); }));
  }
  out.set("devices", std::move(devs));
  return out;
}

}  // namespace ce::cuda
