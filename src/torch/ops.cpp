// copyengine._torch: torch tensors to engine descriptors, and the explicit
// copy API as torch custom ops. Built against the stable libtorch ABI (torch
// 2.9+), so one binary serves every supported torch. All logic lives in the
// core; this file only converts and forwards through the FrontendApi table.

#include <dlfcn.h>

#include <array>
#include <memory>
#include <string>

#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "copyengine/frontend.hpp"

namespace {

using torch::stable::Tensor;
using ST = torch::headeronly::ScalarType;

const ce::FrontendApi* g_api = nullptr;

bool to_dtype(ST t, ce::DType* out) {
  switch (t) {
    case ST::Double: *out = ce::DType::f64; return true;
    case ST::Float: *out = ce::DType::f32; return true;
    case ST::Half: *out = ce::DType::f16; return true;
    case ST::BFloat16: *out = ce::DType::bf16; return true;
    case ST::Long: *out = ce::DType::i64; return true;
    case ST::Int: *out = ce::DType::i32; return true;
    case ST::Short: *out = ce::DType::i16; return true;
    case ST::Char: *out = ce::DType::i8; return true;
    case ST::Byte: *out = ce::DType::u8; return true;
    case ST::Bool: *out = ce::DType::b8; return true;
    default: return false;
  }
}

bool is_float(ST t) { return t == ST::Half || t == ST::BFloat16 || t == ST::Float || t == ST::Double; }

// torch's copy_ gives ce's bits for these pairs (tests/python/test_cuda.py).
bool same_bits(ST d, ST s) { return d == s || (is_float(d) && s != ST::Bool); }

// The whole storage is the endpoint; the view is (shape, strides, offset) in elements.
ce::Endpoint endpoint(const Tensor& t) {
  ce::Endpoint e;
  STD_TORCH_CHECK(to_dtype(t.scalar_type(), &e.dtype), "[copyengine:3] unsupported dtype");
  const int64_t isz = static_cast<int64_t>(t.element_size());
  e.base = reinterpret_cast<uintptr_t>(t.data_ptr()) - static_cast<uintptr_t>(t.storage_offset() * isz);
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(t.get(), &e.nbytes));
  if (t.is_cuda()) {
    e.device = {ce::DeviceKind::cuda, static_cast<int>(t.get_device_index())};
  } else {
    STD_TORCH_CHECK(t.is_cpu(), "[copyengine:3] only CPU and CUDA tensors are supported");
    e.device = {ce::DeviceKind::host, 0};
    e.pinned = g_api->host_is_pinned(e.base);
  }
  return e;
}

// torch's current CUDA stream. Resolved at run time from the already loaded
// libtorch_cuda (Python loads it RTLD_LOCAL, so the global scope cannot see
// it), so this library does not depend on torch's CUDA build.
uintptr_t current_stream(int device) {
  using Fn = AOTITorchError (*)(int32_t, void**);
  static const Fn fn = [] {
    void* lib = dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_NOLOAD);
    return lib ? reinterpret_cast<Fn>(dlsym(lib, "aoti_torch_get_current_cuda_stream")) : nullptr;
  }();
  STD_TORCH_CHECK(fn, "[copyengine:3] torch was built without CUDA");
  void* s = nullptr;
  TORCH_ERROR_CODE_CHECK(fn(device, &s));
  return reinterpret_cast<uintptr_t>(s);
}

// The baseline: the same copy through torch's own aten::copy_.
class TorchCopy final : public ce::NativeOp {
 public:
  TorchCopy(Tensor dst, Tensor src) : dst_(std::move(dst)), src_(std::move(src)) {}
  void run(uintptr_t) override { torch::stable::copy_(dst_, src_, true); }

 private:
  Tensor dst_, src_;
};

void copy_with(Tensor dst, Tensor src, int64_t engine, const ce::Elementwise& fn) {
  STD_TORCH_CHECK(g_api, "[copyengine:3] copyengine._torch is not bound to copyengine._core");
  if (dst.numel() == 0) return;
  const ce::Endpoint de = endpoint(dst), se = endpoint(src);
  std::shared_ptr<ce::NativeOp> native;
  if (fn.identity() && same_bits(dst.scalar_type(), src.scalar_type())) native = std::make_shared<TorchCopy>(dst, src);
  uintptr_t stream = 0;
  if (dst.is_cuda()) stream = current_stream(de.device.index);
  else if (src.is_cuda()) stream = current_stream(se.device.index);
  const auto ds = dst.sizes(), dt = dst.strides(), ss = src.sizes(), st = src.strides();
  std::string err;
  const int rc = g_api->copy_strided(*reinterpret_cast<ce::Engine*>(engine), se, de, ss.data(), st.data(),
                                     static_cast<int>(ss.size()), src.storage_offset(), ds.data(), dt.data(),
                                     static_cast<int>(ds.size()), dst.storage_offset(), fn, std::move(native), stream,
                                     &err);
  STD_TORCH_CHECK(rc == 0, "[copyengine:", rc - 1, "] ", err);
}

void copy_(Tensor dst, Tensor src, int64_t engine) { copy_with(std::move(dst), std::move(src), engine, {}); }

void copy_fn_(Tensor dst, Tensor src, int64_t engine, double scale, double bias, bool round, bool clamp, double lo,
              double hi) {
  const ce::Elementwise fn{static_cast<float>(scale), static_cast<float>(bias), round, clamp, static_cast<float>(lo),
                           static_cast<float>(hi)};
  copy_with(std::move(dst), std::move(src), engine, fn);
}

void copy_meta(Tensor, Tensor, int64_t) {}
void copy_fn_meta(Tensor, Tensor, int64_t, double, double, bool, bool, double, double) {}

}  // namespace

// Called once by copyengine with the address of copyengine._core's table.
extern "C" __attribute__((visibility("default"))) int copyengine_torch_bind(uintptr_t api) {
  const auto* a = reinterpret_cast<const ce::FrontendApi*>(api);
  if (!a || a->version != ce::FrontendApi::kVersion) return 1;
  g_api = a;
  return 0;
}

STABLE_TORCH_LIBRARY(copyengine, m) {
  m.def("copy_(Tensor(a!) dst, Tensor src, int engine) -> ()");
  m.def("copy_fn_(Tensor(a!) dst, Tensor src, int engine, float scale, float bias, bool round, bool clamp, "
        "float lo, float hi) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(copyengine, CompositeExplicitAutograd, m) {
  m.impl("copy_", TORCH_BOX(&copy_));
  m.impl("copy_fn_", TORCH_BOX(&copy_fn_));
}

STABLE_TORCH_LIBRARY_IMPL(copyengine, Meta, m) {
  m.impl("copy_", TORCH_BOX(&copy_meta));
  m.impl("copy_fn_", TORCH_BOX(&copy_fn_meta));
}
