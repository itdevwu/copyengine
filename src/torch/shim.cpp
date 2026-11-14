// copyengine shim: inside a copyengine scope, aten::copy_ is queued in the
// engine and every other op is ordered after the queued copies it touches.
// Built per torch version (it uses torch's unstable C++ dispatcher API), so it
// holds only dispatcher registration and redispatch; everything else is in the
// torch-free core, reached through the FrontendApi table.
//
// The scope is a thread-local included dispatch key (DeferredInit, used out of
// tree only by torchdistx; init() refuses to load if anything else holds it).
// Outside a scope nothing is intercepted.

#include <dlfcn.h>

#include <memory>
#include <string>
#include <vector>

#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/GradMode.h>
#include <c10/core/impl/LocalDispatchKeySet.h>
#include <torch/library.h>

#include "copyengine/frontend.hpp"

namespace {

constexpr c10::DispatchKey kKey = c10::DispatchKey::DeferredInit;
const c10::DispatchKeySet kAfter(c10::DispatchKeySet::FULL_AFTER, kKey);

const ce::FrontendApi* g_api = nullptr;
std::vector<std::unique_ptr<torch::Library>> g_libs;  // registrations made by init()

struct Scope {
  ce::Engine* engine;
  bool was_included;
};
thread_local std::vector<Scope> t_scopes;

ce::Engine* engine() { return t_scopes.empty() ? nullptr : t_scopes.back().engine; }

bool to_dtype(at::ScalarType t, ce::DType* out) {
  switch (t) {
    case at::kDouble: *out = ce::DType::f64; return true;
    case at::kFloat: *out = ce::DType::f32; return true;
    case at::kHalf: *out = ce::DType::f16; return true;
    case at::kBFloat16: *out = ce::DType::bf16; return true;
    case at::kLong: *out = ce::DType::i64; return true;
    case at::kInt: *out = ce::DType::i32; return true;
    case at::kShort: *out = ce::DType::i16; return true;
    case at::kChar: *out = ce::DType::i8; return true;
    case at::kByte: *out = ce::DType::u8; return true;
    case at::kBool: *out = ce::DType::b8; return true;
    case at::kFloat8_e4m3fn: *out = ce::DType::f8e4m3; return true;
    case at::kFloat8_e5m2: *out = ce::DType::f8e5m2; return true;
    case at::kFloat8_e8m0fnu: *out = ce::DType::f8e8m0; return true;
    case at::kFloat4_e2m1fn_x2: *out = ce::DType::f4x2; return true;
    default: return false;
  }
}

bool usable(const at::Tensor& t) {
  ce::DType d;
  return t.defined() && t.layout() == at::kStrided && (t.is_cpu() || t.is_cuda()) && t.has_storage() &&
         to_dtype(t.scalar_type(), &d) && !t.is_conj() && !t.is_neg();
}

// torch's copy_ gives ce's bits for these pairs (tests/python/test_cuda.py).
bool is_float(at::ScalarType t) {
  return t == at::kHalf || t == at::kBFloat16 || t == at::kFloat || t == at::kDouble || t == at::kFloat8_e4m3fn ||
         t == at::kFloat8_e5m2 || t == at::kFloat8_e8m0fnu;
}
bool same_bits(at::ScalarType d, at::ScalarType s) { return d == s || (is_float(d) && s != at::kBool); }

ce::Device ce_device(const at::Tensor& t) {
  return t.is_cuda() ? ce::Device{ce::DeviceKind::cuda, t.get_device()} : ce::Device{ce::DeviceKind::host, 0};
}

ce::Range range_of(const at::Tensor& t) {
  const auto lo = reinterpret_cast<uintptr_t>(t.storage().data());
  return {ce_device(t), lo, lo + t.storage().nbytes()};
}

ce::Endpoint endpoint(const at::Tensor& t) {
  ce::Endpoint e;
  to_dtype(t.scalar_type(), &e.dtype);
  e.device = ce_device(t);
  e.base = reinterpret_cast<uintptr_t>(t.storage().data());
  e.nbytes = static_cast<int64_t>(t.storage().nbytes());
  e.pinned = t.is_cpu() && g_api->host_is_pinned(e.base);
  return e;
}

// torch's current CUDA stream, from the already loaded libtorch_cuda (no
// CUDA headers or libraries are needed to build this file).
uintptr_t current_stream(int device) {
  using Fn = int32_t (*)(int32_t, void**);
  static const Fn fn = [] {
    void* lib = dlopen("libtorch_cuda.so", RTLD_NOW | RTLD_NOLOAD);
    return lib ? reinterpret_cast<Fn>(dlsym(lib, "aoti_torch_get_current_cuda_stream")) : nullptr;
  }();
  TORCH_CHECK(fn, "copyengine: torch was built without CUDA");
  void* s = nullptr;
  TORCH_CHECK(fn(device, &s) == 0, "copyengine: cannot query the current CUDA stream");
  return reinterpret_cast<uintptr_t>(s);
}

void fence(ce::Engine& e, const std::vector<ce::Range>& reads, const std::vector<ce::Range>& writes) {
  std::string err;
  TORCH_CHECK(g_api->fence(e, reads.data(), reads.size(), writes.data(), writes.size(), &err) == 0,
              "copyengine: ", err);
}

const c10::TypedOperatorHandle<at::Tensor&(at::Tensor&, const at::Tensor&, bool)>& copy_op() {
  static const auto op = c10::Dispatcher::singleton()
                             .findSchemaOrThrow("aten::copy_", "")
                             .typed<at::Tensor&(at::Tensor&, const at::Tensor&, bool)>();
  return op;
}

// The copy as torch runs it: the baseline. Runs with the shim's key excluded
// so it does not come back here.
class TorchCopy final : public ce::NativeOp {
 public:
  TorchCopy(at::Tensor dst, at::Tensor src, bool non_blocking)
      : dst_(std::move(dst)), src_(std::move(src)), non_blocking_(non_blocking) {}
  void run(uintptr_t) override { run(0, false); }
  void run(uintptr_t, bool host_async) override {
    c10::impl::ExcludeDispatchKeyGuard guard(kKey);
    copy_op().call(dst_, src_, non_blocking_ || host_async);
  }

 private:
  at::Tensor dst_, src_;
  bool non_blocking_;
};

at::Tensor& native_copy(c10::DispatchKeySet ks, at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  if (ce::Engine* e = engine(); e && g_api->queued(*e) > 0) {
    std::vector<ce::Range> reads, writes{range_of(self)};
    if (src.defined() && src.has_storage()) reads.push_back(range_of(src));
    fence(*e, reads, writes);
  }
  // Ops torch runs inside this one are its own business, not the user's.
  c10::impl::ExcludeDispatchKeyGuard guard(kKey);
  return copy_op().redispatch(ks & kAfter, self, src, non_blocking);
}

// Why copyengine leaves a copy to torch (the same texts as python/copyengine/_mode.py).
const char* decline_reason(const at::Tensor& self, const at::Tensor& src) {
  if (!usable(self) || !usable(src)) return "layout, device or dtype copyengine does not handle";
  if (!same_bits(self.scalar_type(), src.scalar_type())) return "dtype pair where torch's result differs";
  if (c10::GradMode::is_enabled() && (self.requires_grad() || src.requires_grad())) return "autograd";
  return nullptr;
}

// rc is 1 + ce::Errc (copyengine/error.hpp, not shipped with the shim):
// 1 needs_materialize, 2 unsafe_copy, 3 unsupported.
const char* refusal(int rc) {
  switch (rc - 1) {
    case 2: return "overlapping source and destination";
    case 1: return "view copyengine cannot describe";
    case 3: return "unsupported by copyengine";
    default: return "refused by copyengine";
  }
}

at::Tensor& intercepted_copy(c10::DispatchKeySet ks, at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  ce::Engine* e = engine();
  if (!e || self.numel() == 0) return native_copy(ks, self, src, non_blocking);
  if (const char* why = decline_reason(self, src)) {
    g_api->decline(*e, why);
    return native_copy(ks, self, src, non_blocking);
  }
  const ce::Endpoint de = endpoint(self), se = endpoint(src);
  uintptr_t stream = 0;
  if (self.is_cuda()) stream = current_stream(self.get_device());
  else if (src.is_cuda()) stream = current_stream(src.get_device());
  const auto ds = self.sizes(), dt = self.strides(), ss = src.sizes(), st = src.strides();
  std::string err;
  const int rc = g_api->copy_strided(
      *e, se, de, ss.data(), st.data(), static_cast<int>(ss.size()), src.storage_offset(), ds.data(), dt.data(),
      static_cast<int>(ds.size()), self.storage_offset(), ce::Elementwise{},
      std::make_shared<TorchCopy>(self, src, non_blocking), non_blocking ? ce::FrontendApi::kHostAsync : 0u, stream,
      &err);
  if (rc != 0) {
    g_api->decline(*e, refusal(rc));
    return native_copy(ks, self, src, non_blocking);  // torch decides (and raises) as it would
  }
  // The ADInplaceOrView kernel below this key would have done this.
  if (!self.is_inference()) self.unsafeGetTensorImpl()->bump_version();
  return self;
}

void add_tensors(const c10::IValue& v, std::vector<at::Tensor>& out) {
  if (v.isTensor()) {
    out.push_back(v.toTensor());
  } else if (v.isTensorList()) {
    for (const at::Tensor& t : v.toTensorList()) out.push_back(t);
  } else if (v.isList()) {
    for (const c10::IValue& x : v.toListRef())
      if (x.isTensor()) out.push_back(x.toTensor());
  }
}

// Every other op inside a scope: if copies are queued, order it after the
// ones whose memory it reads or writes, then hand it to torch.
void fallback(const c10::OperatorHandle& op, c10::DispatchKeySet ks, torch::jit::Stack* stack) {
  ce::Engine* e = engine();
  if (e && g_api->queued(*e) > 0) {
    const auto& args = op.schema().arguments();
    const size_t n = args.size(), base = stack->size() - n;
    std::vector<ce::Range> reads, writes;
    std::vector<at::Tensor> ts;
    for (size_t i = 0; i < n; ++i) {
      ts.clear();
      add_tensors((*stack)[base + i], ts);
      const bool w = args[i].alias_info() && args[i].alias_info()->isWrite();
      for (const at::Tensor& t : ts) {
        if (!t.defined() || !t.has_storage() || !(t.is_cpu() || t.is_cuda()) || !t.storage().data()) continue;
        (w ? writes : reads).push_back(range_of(t));
      }
    }
    fence(*e, reads, writes);
  }
  // Ops torch runs inside this one (composite ops, kernels calling copy_) are
  // its own business: only what the user calls is intercepted.
  c10::impl::ExcludeDispatchKeyGuard guard(kKey);
  op.redispatchBoxed(ks & kAfter, stack);
}

// Views only alias their input: they touch no memory, so they skip the shim.
bool is_view(const c10::FunctionSchema& s) {
  for (const auto& a : s.arguments())
    if (a.alias_info() && a.alias_info()->isWrite()) return false;
  if (s.returns().empty()) return false;
  for (const auto& r : s.returns())
    if (!r.alias_info()) return false;
  return true;
}

}  // namespace

extern "C" {

// Registers the shim. 0: ok; 1: table from a different build; 2: another
// library already uses the dispatch key; 3: registration failed.
__attribute__((visibility("default"))) int copyengine_shim_init(uintptr_t api) {
  const auto* a = reinterpret_cast<const ce::FrontendApi*>(api);
  if (!a || a->version != ce::FrontendApi::kVersion) return 1;
  if (!g_libs.empty()) return 0;
  auto& d = c10::Dispatcher::singleton();
  if (d.hasBackendFallbackForDispatchKey(kKey)) return 2;
  g_api = a;
  try {
    auto any = std::make_unique<torch::Library>(torch::Library::IMPL, "_", kKey, __FILE__, __LINE__);
    any->fallback(torch::CppFunction::makeFromBoxedFunction<&fallback>());
    auto aten = std::make_unique<torch::Library>(torch::Library::IMPL, "aten", kKey, __FILE__, __LINE__);
    aten->impl("copy_", TORCH_FN(intercepted_copy));
    // Skip the shim: views (they touch no memory), metadata queries, and
    // copyengine's own ops (explicit copies, already queued in order).
    auto ours = std::make_unique<torch::Library>(torch::Library::IMPL, "copyengine", kKey, __FILE__, __LINE__);
    for (const c10::OperatorName& name : d.getAllOpNames()) {
      const std::string n = name.name.substr(name.name.find("::") + 2) +
                            (name.overload_name.empty() ? "" : "." + name.overload_name);
      if (name.name.rfind("copyengine::", 0) == 0) {
        ours->impl(n.c_str(), torch::CppFunction::makeFallthrough());
        continue;
      }
      if (name.name.rfind("aten::", 0) != 0 || name.name == "aten::copy_") continue;
      auto h = d.findOp(name);
      if (!h || !h->hasSchema()) continue;
      if (is_view(h->schema()) || name.name == "aten::is_pinned")
        aten->impl(n.c_str(), torch::CppFunction::makeFallthrough());
    }
    g_libs.push_back(std::move(any));
    g_libs.push_back(std::move(aten));
    g_libs.push_back(std::move(ours));
  } catch (const std::exception&) {
    g_libs.clear();
    g_api = nullptr;
    return 3;
  }
  return 0;
}

// Removes every registration init() made (verification failed).
__attribute__((visibility("default"))) void copyengine_shim_disable() {
  g_libs.clear();
  g_api = nullptr;
}

// Scopes nest; the innermost engine receives the copies.
__attribute__((visibility("default"))) void copyengine_shim_enter(uintptr_t engine) {
  t_scopes.push_back({reinterpret_cast<ce::Engine*>(engine), c10::impl::tls_is_dispatch_key_included(kKey)});
  c10::impl::tls_set_dispatch_key_included(kKey, true);
}

__attribute__((visibility("default"))) void copyengine_shim_exit() {
  if (t_scopes.empty()) return;
  c10::impl::tls_set_dispatch_key_included(kKey, t_scopes.back().was_included);
  t_scopes.pop_back();
}

}  // extern "C"
