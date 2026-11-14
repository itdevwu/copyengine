// copyengine._torch: torch tensors to engine descriptors, and the explicit
// copy API as torch custom ops. Built against the stable libtorch ABI (torch
// 2.10+), so one binary serves every supported torch. All logic lives in the
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
    case ST::Float8_e4m3fn: *out = ce::DType::f8e4m3; return true;
    case ST::Float8_e5m2: *out = ce::DType::f8e5m2; return true;
    case ST::Float8_e8m0fnu: *out = ce::DType::f8e8m0; return true;
    case ST::Float4_e2m1fn_x2: *out = ce::DType::f4x2; return true;
    default: return false;
  }
}

bool is_float(ST t) {
  return t == ST::Half || t == ST::BFloat16 || t == ST::Float || t == ST::Double || t == ST::Float8_e4m3fn ||
         t == ST::Float8_e5m2 || t == ST::Float8_e8m0fnu;
}

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
                                     static_cast<int>(ds.size()), dst.storage_offset(), fn, std::move(native), 0,
                                     stream, &err);
  STD_TORCH_CHECK(rc == 0, "[copyengine:", rc - 1, "] ", err);
}

void copy_(Tensor dst, Tensor src, int64_t engine) { copy_with(std::move(dst), std::move(src), engine, {}); }

void copy_fn_(Tensor dst, Tensor src, int64_t engine, double scale, double bias, bool round, bool clamp, double lo,
              double hi) {
  const ce::Elementwise fn{static_cast<float>(scale), static_cast<float>(bias), round, clamp, static_cast<float>(lo),
                           static_cast<float>(hi)};
  copy_with(std::move(dst), std::move(src), engine, fn);
}

// copyengine's own device memory from torch's caching allocator, as uint8
// tensors: it counts in torch.cuda.memory_allocated() and torch's limits, and
// freeing it never synchronizes. Tied to torch's current stream at allocation
// (the core orders that stream after its own use before freeing).
bool g_shutdown = false;  // torch may be gone at interpreter exit: then leave memory to the process

uintptr_t torch_alloc(int device, size_t bytes, void** handle, uintptr_t* stream) {
  try {
    const int64_t size = static_cast<int64_t>(bytes), stride = 1;
    AtenTensorHandle t = nullptr;
    if (aoti_torch_empty_strided(1, &size, &stride, aoti_torch_dtype_uint8(), aoti_torch_device_type_cuda(), device,
                                 &t) != AOTI_TORCH_SUCCESS || !t)
      return 0;
    void* p = nullptr;
    if (aoti_torch_get_data_ptr(t, &p) != AOTI_TORCH_SUCCESS || !p) {
      aoti_torch_delete_tensor_object(t);
      return 0;
    }
    *handle = t;
    *stream = current_stream(device);
    return reinterpret_cast<uintptr_t>(p);
  } catch (...) {
    return 0;
  }
}

void torch_free(void* handle) {
  if (!g_shutdown) aoti_torch_delete_tensor_object(static_cast<AtenTensorHandle>(handle));
}

const ce::DeviceAllocator kTorchAllocator = [] {
  ce::DeviceAllocator a;
  a.alloc = &torch_alloc;
  a.free = &torch_free;
  a.name = "torch";
  return a;
}();

// ---- gather / scatter with CUDA indices -----------------------------------------------

Tensor dispatch(const char* name, const char* overload, StableIValue* stack) {
#if TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(name, overload, stack, TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(aoti_torch_call_dispatcher(name, overload, stack));
#endif
  return torch::stable::detail::to<Tensor>(stack[0]);
}

// Baselines: what a torch user writes.
class TorchGather final : public ce::NativeOp {  // index_select(src, dim, index, out=out)
 public:
  TorchGather(Tensor out, Tensor src, int64_t dim, Tensor index)
      : out_(std::move(out)), src_(std::move(src)), index_(std::move(index)), dim_(dim) {}
  void run(uintptr_t) override {
    std::array<StableIValue, 4> st{torch::stable::detail::from(src_), torch::stable::detail::from(dim_),
                                   torch::stable::detail::from(index_), torch::stable::detail::from(out_)};
    dispatch("aten::index_select", "out", st.data());
  }

 private:
  Tensor out_, src_, index_;
  int64_t dim_;
};

class TorchScatter final : public ce::NativeOp {  // dst.index_copy_(dim, index, src)
 public:
  TorchScatter(Tensor dst, Tensor src, int64_t dim, Tensor index)
      : dst_(std::move(dst)), src_(std::move(src)), index_(std::move(index)), dim_(dim) {}
  void run(uintptr_t) override {
    std::array<StableIValue, 4> st{torch::stable::detail::from(dst_), torch::stable::detail::from(dim_),
                                   torch::stable::detail::from(index_), torch::stable::detail::from(src_)};
    dispatch("aten::index_copy_", "", st.data());
  }

 private:
  Tensor dst_, src_, index_;
  int64_t dim_;
};

// Indices this path takes: 1-D contiguous int64 on the tensors' GPU; anything
// else is the Python layer's (it converts on the device).
void check_index(const Tensor& index, const Tensor& on) {
  STD_TORCH_CHECK(index.is_cuda() && on.is_cuda() && index.get_device_index() == on.get_device_index() &&
                      index.scalar_type() == ST::Long && index.dim() == 1 && index.stride(0) == 1,
                  "[copyengine:3] index: not a contiguous int64 CUDA vector on the tensors' GPU");
}

void indexed(Tensor dst, Tensor src, int side, int64_t dim, Tensor index, int64_t engine) {
  STD_TORCH_CHECK(g_api && g_api->copy_indexed, "[copyengine:3] copyengine._torch is not bound to copyengine._core");
  check_index(index, side == 0 ? src : dst);
  if (dim < 0) dim += side == 0 ? src.dim() : dst.dim();
  const ce::Endpoint de = endpoint(dst), se = endpoint(src);
  std::shared_ptr<ce::NativeOp> native;
  if (dst.scalar_type() == src.scalar_type()) {
    if (side == 0 && dst.is_contiguous()) native = std::make_shared<TorchGather>(dst, src, dim, index);
    if (side == 1 && src.dim() == dst.dim() && dim >= 0 && dim < dst.dim()) {  // index_copy_ does not broadcast
      bool same = true;
      for (int64_t k = 0; k < dst.dim(); ++k)
        same = same && src.size(k) == (k == dim ? index.numel() : dst.size(k));
      if (same) native = std::make_shared<TorchScatter>(dst, src, dim, index);
    }
  }
  const auto ds = dst.sizes(), dt = dst.strides(), ss = src.sizes(), st = src.strides();
  std::string err;
  const int rc = g_api->copy_indexed(*reinterpret_cast<ce::Engine*>(engine), se, de, ss.data(), st.data(),
                                     static_cast<int>(ss.size()), src.storage_offset(), ds.data(), dt.data(),
                                     static_cast<int>(ds.size()), dst.storage_offset(), side, dim,
                                     reinterpret_cast<uintptr_t>(index.data_ptr()), index.numel(),
                                     static_cast<int>(index.get_device_index()), {}, std::move(native), 0,
                                     current_stream(de.device.index), &err);
  STD_TORCH_CHECK(rc == 0, "[copyengine:", rc - 1, "] ", err);
}

void gather_(Tensor out, Tensor src, int64_t dim, Tensor index, int64_t engine) {
  if (out.numel() == 0) return;
  indexed(std::move(out), std::move(src), 0, dim, std::move(index), engine);
}

void scatter_(Tensor dst, Tensor index, Tensor src, int64_t dim, int64_t engine) {
  if (index.numel() == 0) return;
  indexed(std::move(dst), std::move(src), 1, dim, std::move(index), engine);
}

void gather_meta(Tensor, Tensor, int64_t, Tensor, int64_t) {}
void scatter_meta(Tensor, Tensor, Tensor, int64_t, int64_t) {}

void copy_meta(Tensor, Tensor, int64_t) {}
void copy_fn_meta(Tensor, Tensor, int64_t, double, double, bool, bool, double, double) {}

}  // namespace

// Called once by copyengine with the address of copyengine._core's table;
// `allocator`: also give copyengine's device memory to torch's allocator.
extern "C" __attribute__((visibility("default"))) int copyengine_torch_bind(uintptr_t api, int allocator) {
  const auto* a = reinterpret_cast<const ce::FrontendApi*>(api);
  if (!a || a->version != ce::FrontendApi::kVersion) return 1;
  g_api = a;
  if (allocator && a->set_device_allocator) a->set_device_allocator(&kTorchAllocator);
  return 0;
}

// At interpreter exit (atexit, before torch's own teardown completes).
extern "C" __attribute__((visibility("default"))) void copyengine_torch_shutdown() { g_shutdown = true; }

STABLE_TORCH_LIBRARY(copyengine, m) {
  m.def("copy_(Tensor(a!) dst, Tensor src, int engine) -> ()");
  m.def("gather_(Tensor(a!) out, Tensor src, int dim, Tensor index, int engine) -> ()");
  m.def("scatter_(Tensor(a!) dst, Tensor index, Tensor src, int dim, int engine) -> ()");
  m.def("copy_fn_(Tensor(a!) dst, Tensor src, int engine, float scale, float bias, bool round, bool clamp, "
        "float lo, float hi) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(copyengine, CompositeExplicitAutograd, m) {
  m.impl("copy_", TORCH_BOX(&copy_));
  m.impl("copy_fn_", TORCH_BOX(&copy_fn_));
  m.impl("gather_", TORCH_BOX(&gather_));
  m.impl("scatter_", TORCH_BOX(&scatter_));
}

STABLE_TORCH_LIBRARY_IMPL(copyengine, Meta, m) {
  m.impl("copy_", TORCH_BOX(&copy_meta));
  m.impl("copy_fn_", TORCH_BOX(&copy_fn_meta));
  m.impl("gather_", TORCH_BOX(&gather_meta));
  m.impl("scatter_", TORCH_BOX(&scatter_meta));
}
