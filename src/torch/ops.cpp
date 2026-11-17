// copyengine._torch: torch tensors to engine descriptors, and the explicit
// copy API as torch custom ops. Built against the stable libtorch ABI (torch
// 2.10+), so one binary serves every supported torch. All logic lives in the
// core; this file only converts and forwards through the FrontendApi table.

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

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
    const ce::HostMemory m = g_api->host_memory(e.base);
    e.pinned = m != ce::HostMemory::pageable, e.mapped = m == ce::HostMemory::mapped;
  }
  return e;
}

// An op's engine: a Python Engine's handle, or 0 for the process-wide one
// (the explicit API inside torch.compile regions, where no Engine is traced).
ce::Engine& engine_of(int64_t handle) {
  if (handle) return *reinterpret_cast<ce::Engine*>(handle);
  STD_TORCH_CHECK(g_api->shared_engine, "[copyengine:3] copyengine._core has no shared engine");
  return *g_api->shared_engine();
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

bool g_shutdown = false;  // torch may be gone at interpreter exit: then leave memory to the process

// copyengine's StreamProvider: torch's current stream of a GPU (copies between two GPUs).
bool provide_stream(int device, uintptr_t* stream) noexcept {
  if (g_shutdown) return false;
  try {
    *stream = current_stream(device);
    return true;
  } catch (...) {  // no CUDA build, or an invalid device: copyengine reports it
    return false;
  }
}

// The baseline: the same copy through torch's own aten::copy_.
class TorchCopy final : public ce::NativeOp {
 public:
  TorchCopy(Tensor dst, Tensor src, bool cpu) : dst_(std::move(dst)), src_(std::move(src)), cpu_(cpu) {}
  void run(uintptr_t) override { torch::stable::copy_(dst_, src_, true); }
  bool cpu_host_access() const override { return cpu_; }

 private:
  Tensor dst_, src_;
  bool cpu_;
};

void copy_with(Tensor dst, Tensor src, int64_t engine, const ce::Elementwise& fn) {
  STD_TORCH_CHECK(g_api, "[copyengine:3] copyengine._torch is not bound to copyengine._core");
  if (dst.numel() == 0) return;
  const ce::Endpoint de = endpoint(dst), se = endpoint(src);
  std::shared_ptr<ce::NativeOp> native;
  if (fn.identity() && same_bits(dst.scalar_type(), src.scalar_type())) {
    const bool dense = dst.is_contiguous() && src.is_contiguous() && dst.numel() == src.numel();
    native = std::make_shared<TorchCopy>(
        dst, src, !ce::torch_copy_stream_ordered(de, se, dst.scalar_type() == src.scalar_type(), dense));
  }
  uintptr_t stream = 0;
  if (dst.is_cuda()) stream = current_stream(de.device.index);
  else if (src.is_cuda()) stream = current_stream(se.device.index);
  const auto ds = dst.sizes(), dt = dst.strides(), ss = src.sizes(), st = src.strides();
  std::string err;
  const int rc = g_api->copy_strided(engine_of(engine), se, de, ss.data(), st.data(),
                                     static_cast<int>(ss.size()), src.storage_offset(), ds.data(), dt.data(),
                                     static_cast<int>(ds.size()), dst.storage_offset(), fn, std::move(native), 0,
                                     stream, &err);
  STD_TORCH_CHECK(rc == 0, "[copyengine:", rc - 1, "] ", err);
}

void copy_(Tensor dst, Tensor src, int64_t engine) { copy_with(std::move(dst), std::move(src), engine, {}); }

// Many pairs in one call, inside a scope of their own (nested in the
// caller's, if any): they plan together.
void copy_many_(std::vector<Tensor> dsts, std::vector<Tensor> srcs, int64_t engine) {
  STD_TORCH_CHECK(dsts.size() == srcs.size(), "[copyengine:0] copy_many: ", dsts.size(), " destinations, ",
                  srcs.size(), " sources");
  STD_TORCH_CHECK(g_api && g_api->begin_scope, "[copyengine:3] copyengine._torch is not bound to copyengine._core");
  ce::Engine& e = engine_of(engine);
  g_api->begin_scope(e);
  std::string err;
  try {
    for (size_t i = 0; i < dsts.size(); ++i) copy_with(dsts[i], srcs[i], engine, {});
  } catch (...) {
    g_api->end_scope(e, &err);  // the pairs queued so far still run
    throw;
  }
  const int rc = g_api->end_scope(e, &err);
  STD_TORCH_CHECK(rc == 0, "[copyengine:", rc - 1, "] ", err);
}

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

// copyengine's PeerEnabler: one byte copied by torch from GPU a to GPU b, so
// torch enables a's access to b's memory as for its own copies (its allocator
// then maps b's expandable segments for a as well).
void enable_peer(int a, int b) noexcept {
  if (g_shutdown) return;
  try {
    const int64_t size = 1, stride = 1;
    AtenTensorHandle ta = nullptr, tb = nullptr;
    if (aoti_torch_empty_strided(1, &size, &stride, aoti_torch_dtype_uint8(), aoti_torch_device_type_cuda(), a, &ta) !=
        AOTI_TORCH_SUCCESS)
      return;
    Tensor src(ta);
    if (aoti_torch_empty_strided(1, &size, &stride, aoti_torch_dtype_uint8(), aoti_torch_device_type_cuda(), b, &tb) !=
        AOTI_TORCH_SUCCESS)
      return;
    Tensor dst(tb);
    torch::stable::copy_(dst, src, false);
  } catch (...) {  // copyengine's own admission still decides
  }
}

void torch_free(void* handle) {
  if (!g_shutdown) aoti_torch_delete_tensor_object(static_cast<AtenTensorHandle>(handle));
}

// Pinned host memory from torch's caching host allocator (what pin_memory()
// uses), so it counts there. Resolved at run time: the entry point is newer
// than the oldest torch this library supports.
using PinnedFn = AOTITorchError (*)(int64_t, const int64_t*, const int64_t*, int32_t, int32_t, int32_t,
                                    AtenTensorHandle*);
PinnedFn pinned_fn() {
  static const PinnedFn fn = [] {
    for (const char* lib : {"libtorch_cpu.so", "libtorch.so"}) {
      void* h = dlopen(lib, RTLD_NOW | RTLD_NOLOAD);
      if (void* f = h ? dlsym(h, "aoti_torch_empty_strided_pinned") : nullptr) return reinterpret_cast<PinnedFn>(f);
    }
    return static_cast<PinnedFn>(nullptr);
  }();
  return fn;
}

void* torch_host_alloc(size_t bytes, void** handle) {
  try {
    const int64_t size = static_cast<int64_t>(bytes), stride = 1;
    AtenTensorHandle t = nullptr;
    if (pinned_fn()(1, &size, &stride, aoti_torch_dtype_uint8(), aoti_torch_device_type_cpu(), 0, &t) !=
            AOTI_TORCH_SUCCESS ||
        !t)
      return nullptr;
    void* p = nullptr;
    if (aoti_torch_get_data_ptr(t, &p) != AOTI_TORCH_SUCCESS || !p) {
      aoti_torch_delete_tensor_object(t);
      return nullptr;
    }
    *handle = t;
    return p;
  } catch (...) {
    return nullptr;
  }
}

const ce::DeviceAllocator kTorchAllocator = [] {
  ce::DeviceAllocator a;
  a.alloc = &torch_alloc;
  a.free = &torch_free;
  if (pinned_fn()) {
    a.host_alloc = &torch_host_alloc;
    a.host_free = &torch_free;  // also just drops the tensor
  }
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

// A contiguous copy of `t` (host) on `like`'s GPU.
Tensor on_gpu_like(const Tensor& t, const Tensor& like) {
  const int64_t r = t.dim();
  std::vector<int64_t> sizes(static_cast<size_t>(r)), strides(static_cast<size_t>(r));
  int64_t st = 1;
  for (int64_t k = r - 1; k >= 0; --k) sizes[k] = t.size(k), strides[k] = st, st *= std::max<int64_t>(1, t.size(k));
  int32_t dtype = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(t.get(), &dtype));
  AtenTensorHandle h = nullptr;
  TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(r, sizes.data(), strides.data(), dtype,
                                                  aoti_torch_device_type_cuda(),
                                                  static_cast<int32_t>(like.get_device_index()), &h));
  Tensor out(h);
  torch::stable::copy_(out, t, true);
  return out;
}

// Baselines: what a torch user writes.
class TorchGather final : public ce::NativeOp {  // index_select(src, dim, index, out=out)
 public:
  TorchGather(Tensor out, Tensor src, int64_t dim, Tensor index)
      : out_(std::move(out)), src_(std::move(src)), index_(std::move(index)), dim_(dim) {}
  void run(uintptr_t) override {
    if (out_.is_cuda()) {
      std::array<StableIValue, 4> st{torch::stable::detail::from(src_), torch::stable::detail::from(dim_),
                                     torch::stable::detail::from(index_), torch::stable::detail::from(out_)};
      dispatch("aten::index_select", "out", st.data());
      return;
    }
    // Into host memory: gather on the GPU, then copy (torch's index_select needs one device).
    std::array<StableIValue, 3> st{torch::stable::detail::from(src_), torch::stable::detail::from(dim_),
                                   torch::stable::detail::from(index_)};
    Tensor tmp = dispatch("aten::index_select", "", st.data());
    torch::stable::copy_(out_, tmp, true);
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
    Tensor src = src_;
    if (!src.is_cuda()) src = on_gpu_like(src_, dst_);  // index_copy_ needs one device
    std::array<StableIValue, 4> st{torch::stable::detail::from(dst_), torch::stable::detail::from(dim_),
                                   torch::stable::detail::from(index_), torch::stable::detail::from(src)};
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
  const int64_t rank = side == 0 ? src.dim() : dst.dim();
  if (dim < 0) dim += rank;
  STD_TORCH_CHECK(dim >= 0 && dim < rank, "[copyengine:0] dim out of range");
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
  const int rc = g_api->copy_indexed(engine_of(engine), se, de, ss.data(), st.data(),
                                     static_cast<int>(ss.size()), src.storage_offset(), ds.data(), dt.data(),
                                     static_cast<int>(ds.size()), dst.storage_offset(), side, static_cast<int>(dim),
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

// ---- quantization while copying ------------------------------------------------------

// One tensor of a tile op: its storage, and byte offset / row strides / step
// of the view; `bh` > 1: x or q of box units (rows of boxes bh rows apart,
// a box's rows one row apart).
ce::TileSpec::Side tile_side(const Tensor& t, int64_t bh = 1) {
  ce::TileSpec::Side s;
  s.ep = endpoint(t);
  const int64_t isz = static_cast<int64_t>(t.element_size()), r = t.dim();
  s.offset = t.storage_offset() * isz;
  if (bh > 1) {
    for (int64_t k = 0; k + 2 < r && k < 2; ++k) s.row_stride[static_cast<size_t>(k)] = t.stride(k) * isz;
    s.row_stride[static_cast<size_t>(std::min<int64_t>(r - 2, 2))] = bh * t.stride(r - 2) * isz;
    s.pitch = t.stride(r - 2) * isz;
  } else {
    for (int64_t k = 0; k + 1 < r && k < 3; ++k) s.row_stride[static_cast<size_t>(k)] = t.stride(k) * isz;
  }
  s.step = t.stride(r - 1) * isz;
  return s;
}

// An index over the first dimension of `of`: GPU values stay there, host values are read now.
ce::TileSpec::Index tile_index(const std::optional<Tensor>& ix, const Tensor& of) {
  ce::TileSpec::Index i;
  if (!ix) return i;
  const Tensor& t = *ix;
  STD_TORCH_CHECK(t.scalar_type() == ST::Long && t.dim() == 1 && t.stride(0) == 1,
                  "[copyengine:0] an index is a contiguous int64 vector");
  i.bound = of.size(0);
  i.count = t.numel();
  if (t.is_cuda()) {
    i.dev = reinterpret_cast<uintptr_t>(t.data_ptr());
  } else {
    const auto* p = static_cast<const int64_t*>(t.data_ptr());
    i.host.assign(p, p + t.numel());
  }
  return i;
}

// The unit size of `fmt` (MX: 32, NVFP4: 16; the others: `group`); copyengine/_quant.py UNIT.
int64_t unit_of(int64_t fmt, int64_t group) { return fmt == 3 || fmt == 4 ? 32 : fmt == 5 ? 16 : group; }

void tile_op(bool quantize, Tensor x, Tensor q, Tensor scale, const std::optional<Tensor>& zero, int64_t fmt,
             int64_t group, int64_t bh, double eps, double gs, bool static_scale, const std::optional<Tensor>& x_index,
             const std::optional<Tensor>& index, int64_t engine) {
  STD_TORCH_CHECK(g_api && g_api->tile, "[copyengine:3] copyengine._torch is not bound to copyengine._core");
  const int64_t lead_dims = x.dim() - (bh > 1 ? 2 : 1);  // box units: rows of boxes are a leading dimension too
  STD_TORCH_CHECK(lead_dims >= (bh > 1 ? 0 : 1) && lead_dims + (bh > 1 ? 1 : 0) <= 3,
                  "[copyengine:0] x needs one to three leading dimensions");
  const int64_t D = x.size(x.dim() - 1), g = unit_of(fmt, group);
  STD_TORCH_CHECK(g > 0 && D % g == 0, "[copyengine:0] the last dimension is not a multiple of the unit size");
  ce::TileSpec t;
  t.fmt = static_cast<int>(fmt);
  t.quantize = quantize;
  t.static_scale = static_scale;
  t.g = static_cast<int>(g);
  t.bh = static_cast<int>(bh);
  t.eps = static_cast<float>(eps);
  t.gs = static_cast<float>(gs);
  t.upr = D / g;
  for (int64_t k = 0; k < lead_dims; ++k) t.rows[static_cast<size_t>(k)] = x.size(k);
  if (bh > 1) {
    const int64_t M = x.size(x.dim() - 2);
    STD_TORCH_CHECK(M % bh == 0, "[copyengine:0] the rows are not a multiple of the block");
    t.rows[static_cast<size_t>(lead_dims)] = M / bh;
  }
  if (x_index) t.rows[0] = x_index->numel();
  else if (index) t.rows[0] = index->numel();
  t.x = tile_side(x, bh), t.q = tile_side(q, bh), t.scale = tile_side(scale);
  if (zero) t.zero = tile_side(*zero);
  t.x_index = tile_index(x_index, x);
  t.index = tile_index(index, q);
  uintptr_t stream = 0;
  if (x.is_cuda()) stream = current_stream(static_cast<int>(x.get_device_index()));
  else if (q.is_cuda()) stream = current_stream(static_cast<int>(q.get_device_index()));
  std::string err;
  const int rc = g_api->tile(engine_of(engine), t, stream, &err);
  STD_TORCH_CHECK(rc == 0, "[copyengine:", rc - 1, "] ", err);
}

void quantize_(Tensor x, Tensor q, Tensor scale, std::optional<Tensor> zero, int64_t fmt, int64_t group, int64_t bh,
               double eps, double gs, bool static_scale, std::optional<Tensor> x_index, std::optional<Tensor> index,
               int64_t engine) {
  tile_op(true, std::move(x), std::move(q), std::move(scale), zero, fmt, group, bh, eps, gs, static_scale, x_index,
          index, engine);
}

void dequantize_(Tensor q, Tensor scale, Tensor out, std::optional<Tensor> zero, int64_t fmt, int64_t group,
                 int64_t bh, double gs, std::optional<Tensor> x_index, std::optional<Tensor> index, int64_t engine) {
  tile_op(false, std::move(out), std::move(q), std::move(scale), zero, fmt, group, bh, 1e-12, gs, false, x_index,
          index, engine);
}

void quantize_meta(Tensor, Tensor, Tensor, std::optional<Tensor>, int64_t, int64_t, int64_t, double, double, bool,
                   std::optional<Tensor>, std::optional<Tensor>, int64_t) {}
void dequantize_meta(Tensor, Tensor, Tensor, std::optional<Tensor>, int64_t, int64_t, int64_t, double,
                     std::optional<Tensor>, std::optional<Tensor>, int64_t) {}

void gather_meta(Tensor, Tensor, int64_t, Tensor, int64_t) {}
void scatter_meta(Tensor, Tensor, Tensor, int64_t, int64_t) {}

void copy_meta(Tensor, Tensor, int64_t) {}
void copy_many_meta(std::vector<Tensor>, std::vector<Tensor>, int64_t) {}
void copy_fn_meta(Tensor, Tensor, int64_t, double, double, bool, bool, double, double) {}

}  // namespace

// Called once by copyengine with the address of copyengine._core's table;
// `allocator`: also give copyengine's device memory to torch's allocator.
extern "C" __attribute__((visibility("default"))) int copyengine_torch_bind(uintptr_t api, int allocator) {
  const auto* a = reinterpret_cast<const ce::FrontendApi*>(api);
  if (!a || a->version != ce::FrontendApi::kVersion) return 1;
  g_api = a;
  if (allocator && a->set_device_allocator) a->set_device_allocator(&kTorchAllocator);
  if (a->set_stream_provider) a->set_stream_provider(&provide_stream);
  if (a->set_peer_enabler) a->set_peer_enabler(&enable_peer);
  return 0;
}

// At interpreter exit (atexit, before torch's own teardown completes).
extern "C" __attribute__((visibility("default"))) void copyengine_torch_shutdown() { g_shutdown = true; }

STABLE_TORCH_LIBRARY(copyengine, m) {
  m.def("copy_(Tensor(a!) dst, Tensor src, int engine) -> ()");
  m.def("copy_many_(Tensor(a!)[] dsts, Tensor[] srcs, int engine) -> ()");
  m.def("gather_(Tensor(a!) out, Tensor src, int dim, Tensor index, int engine) -> ()");
  m.def("scatter_(Tensor(a!) dst, Tensor index, Tensor src, int dim, int engine) -> ()");
  m.def("copy_fn_(Tensor(a!) dst, Tensor src, int engine, float scale, float bias, bool round, bool clamp, "
        "float lo, float hi) -> ()");
  m.def("quantize_(Tensor x, Tensor(a!) q, Tensor(b!) scale, Tensor(c!)? zero, int fmt, int group, int block_rows, "
        "float eps, float gs, bool static_scale, Tensor? x_index, Tensor? index, int engine) -> ()");
  m.def("dequantize_(Tensor q, Tensor scale, Tensor(a!) out, Tensor? zero, int fmt, int group, int block_rows, "
        "float gs, Tensor? x_index, Tensor? index, int engine) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(copyengine, CompositeExplicitAutograd, m) {
  m.impl("copy_", TORCH_BOX(&copy_));
  m.impl("copy_fn_", TORCH_BOX(&copy_fn_));
  m.impl("copy_many_", TORCH_BOX(&copy_many_));
  m.impl("gather_", TORCH_BOX(&gather_));
  m.impl("scatter_", TORCH_BOX(&scatter_));
  m.impl("quantize_", TORCH_BOX(&quantize_));
  m.impl("dequantize_", TORCH_BOX(&dequantize_));
}

STABLE_TORCH_LIBRARY_IMPL(copyengine, Meta, m) {
  m.impl("copy_", TORCH_BOX(&copy_meta));
  m.impl("copy_fn_", TORCH_BOX(&copy_fn_meta));
  m.impl("copy_many_", TORCH_BOX(&copy_many_meta));
  m.impl("gather_", TORCH_BOX(&gather_meta));
  m.impl("scatter_", TORCH_BOX(&scatter_meta));
  m.impl("quantize_", TORCH_BOX(&quantize_meta));
  m.impl("dequantize_", TORCH_BOX(&dequantize_meta));
}
