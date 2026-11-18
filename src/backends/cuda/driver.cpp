#include "backends/cuda/driver.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <map>
#include <string>
#include <mutex>

namespace ce::cuda {
namespace {

using GetProcAddressFn = CUresult(CUDAAPI*)(const char*, void**, int, cuuint64_t,
                                            CUdriverProcAddressQueryResult*);
using InitFn = CUresult(CUDAAPI*)(unsigned int);

struct State {
  Driver drv;
  std::string error;
  std::map<std::string, std::string> missing;
  bool ok = false;
};

std::string status_text(CUresult r, CUdriverProcAddressQueryResult s) {
  if (s == CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT) return "driver too old for this entry point";
  if (s == CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND) return "symbol not found in driver";
  // Not error_name(): this runs while state() is still being constructed.
  return "cuGetProcAddress failed: CUresult " + std::to_string(static_cast<int>(r));
}

State load() {
  State st;
  void* lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    const char* e = dlerror();
    st.error = std::string("cannot load libcuda.so.1: ") + (e ? e : "unknown error");
    return st;
  }
  // Deliberately never dlclose: the driver must outlive every handle we give out.
  auto init = reinterpret_cast<InitFn>(dlsym(lib, "cuInit"));
  auto gpa = reinterpret_cast<GetProcAddressFn>(dlsym(lib, "cuGetProcAddress_v2"));
  if (!init || !gpa) {
    st.error = "libcuda.so.1 lacks cuInit / cuGetProcAddress_v2 (driver older than CUDA 12)";
    return st;
  }
  if (CUresult r = init(0); r != CUDA_SUCCESS) {
    st.error = "cuInit failed: " + std::to_string(static_cast<int>(r));
    return st;
  }

  auto fetch = [&](const char* name, int version, void** out) -> std::string {
    CUdriverProcAddressQueryResult status{};
    CUresult r = gpa(name, out, version, CU_GET_PROC_ADDRESS_DEFAULT, &status);
    if (r == CUDA_SUCCESS && status == CU_GET_PROC_ADDRESS_SUCCESS && *out) return {};
    *out = nullptr;
    return status_text(r, status);
  };

#define CE_CUDA_FETCH_REQUIRED(name, version)                                            \
  if (auto why = fetch(#name, version, reinterpret_cast<void**>(&st.drv.name)); !why.empty()) { \
    st.error = std::string(#name) + ": " + why;                                        \
    return st;                                                                         \
  }
  CE_CUDA_REQUIRED(CE_CUDA_FETCH_REQUIRED)
#undef CE_CUDA_FETCH_REQUIRED

#define CE_CUDA_FETCH_OPTIONAL(name, version)                                            \
  if (auto why = fetch(#name, version, reinterpret_cast<void**>(&st.drv.name)); !why.empty()) { \
    st.missing[#name] = why;                                                           \
  }
  CE_CUDA_OPTIONAL(CE_CUDA_FETCH_OPTIONAL)
#undef CE_CUDA_FETCH_OPTIONAL

  st.ok = true;
  return st;
}

const State& state() {
  static const State st = load();
  return st;
}

}  // namespace

const Driver* driver(std::string* error) {
  const State& st = state();
  if (!st.ok) {
    if (error) *error = st.error;
    return nullptr;
  }
  return &st.drv;
}

std::string missing_reason(const char* symbol) {
  const State& st = state();
  auto it = st.missing.find(symbol);
  return it == st.missing.end() ? std::string() : it->second;
}

std::string error_name(CUresult r) {
  const State& st = state();
  const char* name = nullptr;
  if (st.drv.cuGetErrorName && st.drv.cuGetErrorName(r, &name) == CUDA_SUCCESS && name) return name;
  return "CUresult " + std::to_string(static_cast<int>(r));
}

}  // namespace ce::cuda
