#include <atomic>

#include <nanobind/nanobind.h>
#include <cstring>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/array.h>

#include "core/quant.hpp"
#include "copyengine/copyengine.h"
#include "copyengine/copyengine.hpp"
#include "copyengine/engine.hpp"
#include "copyengine/error.hpp"
#include "copyengine/frontend.hpp"
#include "copyengine/ir.hpp"
#include "copyengine/probe.hpp"

namespace nb = nanobind;
using namespace nb::literals;
using ce::i64;

namespace {

// int64 C-contiguous 1-D host array; the Python layer normalizes inputs to this.
using I64In = nb::ndarray<const int64_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using I64Out = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>;

std::span<const i64> span_of(const I64In& a) { return {a.data(), a.shape(0)}; }

// Accepts anything numpy can turn into int64 (lists, arrays, tensors on CPU).
struct I64Arg {
  nb::object keep;
  I64In arr;
  std::span<const i64> span() const { return span_of(arr); }
};

I64Arg i64_arg(nb::handle obj) {
  // Leaked on purpose: a static nb::object would be destroyed after the
  // interpreter has finalized.
  static PyObject* asarray = nb::object(nb::module_::import_("numpy").attr("ascontiguousarray")).release().ptr();
  nb::object a = nb::handle(asarray)(obj, "dtype"_a = "int64").attr("reshape")(-1);
  return {a, nb::cast<I64In>(a)};
}

// Returned as a plain object so property getters don't apply reference_internal.
nb::object to_numpy(std::vector<i64> v) {
  auto* heap = new std::vector<i64>(std::move(v));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<i64>*>(p); });
  return nb::cast(I64Out(heap->data(), {heap->size()}, owner), nb::rv_policy::move);
}

// Set by Py_AtExit, which runs at the very end of finalization. Before that
// (module teardown included) releasing a reference is still safe, although
// Py_IsInitialized() already reports false.
std::atomic<bool> g_python_gone{false};

// Native baseline implemented in Python: a tuple (fn, *args) run as fn(*args).
// The engine holds the tuple only while the op is queued or frozen.
class PyNativeOp final : public ce::NativeOp {
 public:
  explicit PyNativeOp(PyObject* t) : t_(t) { Py_INCREF(t_); }
  ~PyNativeOp() override {
    if (g_python_gone.load()) return;  // the interpreter is gone: nothing to release
    const PyGILState_STATE g = PyGILState_Ensure();
    Py_DECREF(t_);
    PyGILState_Release(g);
  }
  void run(uintptr_t) override {
    nb::gil_scoped_acquire gil;
    nb::object args = nb::steal(PyTuple_GetSlice(t_, 1, PyTuple_Size(t_)));
    if (!args.is_valid()) throw nb::python_error();
    nb::object r = nb::steal(PyObject_Call(PyTuple_GetItem(t_, 0), args.ptr(), nullptr));
    if (!r.is_valid()) throw nb::python_error();
  }

 private:
  PyObject* t_;
};

std::shared_ptr<ce::NativeOp> py_payload(nb::handle obj) {
  if (obj.is_none()) return nullptr;
  return std::make_shared<PyNativeOp>(obj.ptr());
}

}  // namespace

NB_MODULE(_core, m) {
  const std::string_view v = ce::version_string();
  m.attr("__version__") = nb::str(v.data(), v.size());
  m.attr("ABI_VERSION") = ce_abi_version();
  Py_AtExit([] { g_python_gone.store(true); });

  // ---- errors ----
  // Exception types are held as raw pointers for the life of the process: the
  // translator may run during shutdown, and static nb::objects would be
  // destroyed after the interpreter has finalized.
  static PyObject* base_exc = PyErr_NewException("copyengine.CopyEngineError", PyExc_RuntimeError, nullptr);
  static PyObject* materialize_exc = PyErr_NewException("copyengine.NeedsMaterialize", base_exc, nullptr);
  static PyObject* unsafe_exc = PyErr_NewException("copyengine.UnsafeCopy", base_exc, nullptr);
  static PyObject* unsupported_exc = PyErr_NewException("copyengine.Unsupported", base_exc, nullptr);
  static PyObject* backend_exc = PyErr_NewException("copyengine.BackendError", base_exc, nullptr);
  m.attr("CopyEngineError") = nb::handle(base_exc);
  m.attr("NeedsMaterialize") = nb::handle(materialize_exc);
  m.attr("UnsafeCopy") = nb::handle(unsafe_exc);
  m.attr("Unsupported") = nb::handle(unsupported_exc);
  m.attr("BackendError") = nb::handle(backend_exc);
  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const ce::Error& e) {
      PyObject* type = PyExc_ValueError;
      switch (e.code()) {
        case ce::Errc::invalid_argument: type = PyExc_ValueError; break;
        case ce::Errc::needs_materialize: type = materialize_exc; break;
        case ce::Errc::unsafe_copy: type = unsafe_exc; break;
        case ce::Errc::unsupported: type = unsupported_exc; break;
        case ce::Errc::backend: type = backend_exc; break;
      }
      PyErr_SetString(type, e.what());
    }
  });

  m.def(
      "probe_json",
      [](bool bandwidth) {
        std::string s;
        {
          nb::gil_scoped_release release;
          s = ce::probe_json({.bandwidth = bandwidth});
        }
        return s;
      },
      "bandwidth"_a = false);

  // ---- IR ----
  nb::class_<ce::Runs>(m, "Runs")
      .def_static("of", [](nb::handle s, nb::handle d, nb::handle n) {
        auto a = i64_arg(s), b = i64_arg(d), c = i64_arg(n);
        return ce::Runs::of(a.span(), b.span(), c.span());
      }, "src_off"_a, "dst_off"_a, "length"_a)
      .def_static("from_indices", [](nb::handle s, nb::handle d) {
        auto a = i64_arg(s), b = i64_arg(d);
        return ce::Runs::from_indices(a.span(), b.span());
      }, "src_idx"_a, "dst_idx"_a)
      .def("__len__", &ce::Runs::size)
      .def_prop_ro("total", &ce::Runs::total)
      .def_prop_ro("src_off", [](const ce::Runs& r) { return to_numpy(r.src); })
      .def_prop_ro("dst_off", [](const ce::Runs& r) { return to_numpy(r.dst); })
      .def_prop_ro("length", [](const ce::Runs& r) { return to_numpy(r.len); })
      .def("coalesce", [](const ce::Runs& r) { return r.coalesce(); })
      .def("clip_src", [](const ce::Runs& r, nb::handle s, nb::handle n) {
        auto a = i64_arg(s), b = i64_arg(n);
        return r.clip_src(a.span(), b.span());
      })
      .def("chunks", &ce::Runs::chunks)
      .def("expand", [](const ce::Runs& r) {
        std::vector<i64> s, d;
        r.expand(s, d);
        return std::make_pair(to_numpy(std::move(s)), to_numpy(std::move(d)));
      })
      .def("recover_2d", [](const ce::Runs& r, int min_height) { return ce::recover_2d(r, min_height); },
           "min_height"_a = 2);

  nb::class_<ce::Rects>(m, "Rects")
      .def("__len__", &ce::Rects::size)
      .def_prop_ro("src_off", [](const ce::Rects& r) { return to_numpy(r.src); })
      .def_prop_ro("dst_off", [](const ce::Rects& r) { return to_numpy(r.dst); })
      .def_prop_ro("width", [](const ce::Rects& r) { return to_numpy(r.width); })
      .def_prop_ro("height", [](const ce::Rects& r) { return to_numpy(r.height); })
      .def_prop_ro("src_pitch", [](const ce::Rects& r) { return to_numpy(r.src_pitch); })
      .def_prop_ro("dst_pitch", [](const ce::Rects& r) { return to_numpy(r.dst_pitch); })
      .def("expand", [](const ce::Rects& r) {
        std::vector<i64> s, d;
        r.expand(s, d);
        return std::make_pair(to_numpy(std::move(s)), to_numpy(std::move(d)));
      });

  nb::class_<ce::Access>(m, "Access")
      .def_static("contiguous", [](const std::vector<i64>& shape, i64 offset) {
        return ce::Access::contiguous(shape, offset);
      }, "shape"_a, "offset"_a = 0)
      .def_static("strided", [](const std::vector<i64>& shape, const std::vector<i64>& strides, i64 offset) {
        return ce::Access::strided(shape, strides, offset);
      }, "shape"_a, "strides"_a, "offset"_a = 0)
      .def_prop_ro("shape", &ce::Access::shape)
      .def_prop_ro("base", &ce::Access::base)
      .def_prop_ro("has_table", &ce::Access::has_table)
      .def("permute", [](const ce::Access& a, const std::vector<int>& p) { return a.permute(p); })
      .def("slice", &ce::Access::slice, "dim"_a, "start"_a = nb::none(), "stop"_a = nb::none(), "step"_a = 1)
      .def("select", &ce::Access::select)
      .def("flip", &ce::Access::flip)
      .def("index_select", [](const ce::Access& a, int dim, nb::handle idx) {
        auto t = i64_arg(idx);
        return a.index_select(dim, t.span());
      })
      .def("index_select_device", &ce::Access::index_select_device, "dim"_a, "ptr"_a, "n"_a, "device"_a,
           "index_select whose int64 indices stay in device memory at `ptr` (read by the copy kernel)")
      .def("unsqueeze", &ce::Access::unsqueeze)
      .def("expand", &ce::Access::expand)
      .def("reshape", [](const ce::Access& a, const std::vector<i64>& s) { return a.reshape(s); })
      .def("offsets", [](const ce::Access& a) { return to_numpy(a.offsets()); })
      .def("extent", &ce::Access::extent);

  nb::class_<ce::TileSpec::Side>(m, "TileSide")
      .def("__init__", [](ce::TileSpec::Side* s, const ce::Endpoint& ep, int64_t offset,
                          const std::array<int64_t, 3>& row_stride, int64_t step, int64_t pitch) {
        new (s) ce::TileSpec::Side{ep, offset, row_stride, step, pitch};
      }, "ep"_a, "offset"_a, "row_stride"_a, "step"_a, "pitch"_a = 0);
  nb::class_<ce::TileSpec>(m, "TileSpec")
      .def(nb::init<>())
      .def_rw("fmt", &ce::TileSpec::fmt)
      .def_rw("quantize", &ce::TileSpec::quantize)
      .def_rw("static_scale", &ce::TileSpec::static_scale)
      .def_rw("g", &ce::TileSpec::g)
      .def_rw("bh", &ce::TileSpec::bh)
      .def_rw("eps", &ce::TileSpec::eps)
      .def_rw("gs", &ce::TileSpec::gs)
      .def_rw("rows", &ce::TileSpec::rows)
      .def_rw("upr", &ce::TileSpec::upr)
      .def_rw("x", &ce::TileSpec::x)
      .def_rw("q", &ce::TileSpec::q)
      .def_rw("scale", &ce::TileSpec::scale)
      .def_rw("zero", &ce::TileSpec::zero)
      .def("set_index", [](ce::TileSpec& t, bool x_side, uintptr_t dev, const std::vector<int64_t>& host,
                           int64_t bound, uintptr_t host_ptr) {
        ce::TileSpec::Index& ix = x_side ? t.x_index : t.index;
        ix.dev = dev, ix.host = host, ix.bound = bound, ix.host_ptr = host_ptr;
      }, "x_side"_a, "dev"_a, "host"_a, "bound"_a, "host_ptr"_a = 0);

  nb::class_<ce::Stmt>(m, "Stmt")
      .def(nb::init<ce::Access, ce::Access>(), "dst"_a, "src"_a)
      .def_ro("dst", &ce::Stmt::dst)
      .def_ro("src", &ce::Stmt::src)
      .def_prop_ro("domain", &ce::Stmt::domain);

  m.def("lower", &ce::lower, "stmt"_a, nb::call_guard<nb::gil_scoped_release>());

  // ---- engine ----
  nb::class_<ce::Endpoint>(m, "Endpoint")
      .def("__init__", [](ce::Endpoint* e, const std::string& kind, int index, uintptr_t base, int64_t nbytes,
                          const std::string& dtype, bool pinned) {
        ce::Device dev;
        if (kind == "host" || kind == "cpu") dev = {ce::DeviceKind::host, 0};
        else if (kind == "cuda") dev = {ce::DeviceKind::cuda, index};
        else throw ce::Error(ce::Errc::unsupported, "unsupported device kind: " + kind);
        new (e) ce::Endpoint{dev, base, nbytes, ce::parse_dtype(dtype), pinned};
      }, "kind"_a, "index"_a, "base"_a, "nbytes"_a, "dtype"_a, "pinned"_a = false)
      .def_prop_ro("base", [](const ce::Endpoint& e) { return e.base; })
      .def_prop_ro("nbytes", [](const ce::Endpoint& e) { return e.nbytes; })
      .def_prop_ro("pinned", [](const ce::Endpoint& e) { return e.pinned; })
      .def_prop_ro("dtype", [](const ce::Endpoint& e) { return std::string(ce::dtype_name(e.dtype)); });

  nb::class_<ce::Elementwise>(m, "Elementwise")
      .def("__init__", [](ce::Elementwise* f, float scale, float bias, bool round, std::optional<std::pair<float, float>> clamp) {
        new (f) ce::Elementwise{scale, bias, round, clamp.has_value(), clamp ? clamp->first : 0.f, clamp ? clamp->second : 0.f};
      }, "scale"_a = 1.0f, "bias"_a = 0.0f, "round"_a = false, "clamp"_a = nb::none())
      .def_ro("scale", &ce::Elementwise::scale)
      .def_ro("bias", &ce::Elementwise::bias)
      .def_ro("round", &ce::Elementwise::round)
      .def_prop_ro("clamp", [](const ce::Elementwise& f) -> std::optional<std::pair<float, float>> {
        if (!f.clamp) return std::nullopt;
        return std::make_pair(f.lo, f.hi);
      })
      .def_prop_ro("identity", &ce::Elementwise::identity)
      .def("__repr__", [](const ce::Elementwise& f) {
        std::string r = "Elementwise(scale=" + std::to_string(f.scale) + ", bias=" + std::to_string(f.bias);
        if (f.round) r += ", round=True";
        if (f.clamp) r += ", clamp=(" + std::to_string(f.lo) + ", " + std::to_string(f.hi) + ")";
        return r + ")";
      });

  nb::class_<ce::EngineOptions>(m, "EngineOptions")
      .def(nb::init<>())
      .def_rw("staging_budget", &ce::EngineOptions::staging_budget)
      .def_rw("explore", &ce::EngineOptions::explore)
      .def_rw("prior_only", &ce::EngineOptions::prior_only)
      .def_rw("tune_trials", &ce::EngineOptions::tune_trials)
      .def_rw("tune_topk", &ce::EngineOptions::tune_topk)
      .def_rw("ab_pairs", &ce::EngineOptions::ab_pairs)
      .def_rw("ab_max_pairs", &ce::EngineOptions::ab_max_pairs)
      .def_rw("online_every", &ce::EngineOptions::online_every)
      .def_rw("allow_async", &ce::EngineOptions::allow_async)
      .def_rw("sync_always", &ce::EngineOptions::sync_always)
      .def_rw("max_sms", &ce::EngineOptions::max_sms)
      .def_rw("bandwidth", &ce::EngineOptions::bandwidth)
      .def_rw("disable", &ce::EngineOptions::disable)
      .def_rw("log_plan", &ce::EngineOptions::log_plan)
      .def_rw("host_threads", &ce::EngineOptions::host_threads)
      .def_rw("plan_cache", &ce::EngineOptions::plan_cache)
      .def_static("from_env", [](ce::EngineOptions base) { return ce::EngineOptions::from_env(base); }, "base"_a);

  nb::class_<ce::Frozen>(m, "Frozen")
      .def("run", nb::overload_cast<uintptr_t>(&ce::Frozen::run), "stream"_a = 0,
           nb::call_guard<nb::gil_scoped_release>())
      .def("run_rebound",
           [](ce::Frozen& f, uintptr_t stream, const std::vector<std::array<uintptr_t, 3>>& rebinds) {
             std::vector<ce::Rebind> r;
             r.reserve(rebinds.size());
             for (const auto& x : rebinds) r.push_back({x[0], x[1], x[2]});
             nb::gil_scoped_release release;
             f.run(stream, r);
           },
           "stream"_a, "rebinds"_a)
      .def_prop_ro("tables", &ce::Frozen::tables)
      .def_prop_ro("allocations", &ce::Frozen::allocations);

  auto make_hints = [](nb::handle valid_start, nb::handle valid_len, const std::string& ctx) {
    ce::Hints h;
    h.ctx = ctx;
    if (!valid_start.is_none()) {
      auto a = i64_arg(valid_start), b = i64_arg(valid_len);
      h.valid_start.assign(a.span().begin(), a.span().end());
      h.valid_len.assign(b.span().begin(), b.span().end());
      h.has_valid = true;
    }
    return h;
  };

  nb::class_<ce::Engine>(m, "Engine")
      .def(nb::init<ce::EngineOptions>(), "options"_a)
      .def("copy_runs", [make_hints](ce::Engine& e, const ce::Endpoint& src, const ce::Endpoint& dst, nb::handle so,
                                     nb::handle d_o, nb::handle n, const ce::Elementwise& fn, nb::handle valid_start,
                                     nb::handle valid_len, const std::string& ctx, uintptr_t stream, nb::handle native) {
        auto a = i64_arg(so), b = i64_arg(d_o), c = i64_arg(n);
        ce::Runs runs = ce::Runs::of(a.span(), b.span(), c.span());
        ce::Hints h = make_hints(valid_start, valid_len, ctx);
        h.native = py_payload(native);
        nb::gil_scoped_release release;
        e.copy(src, dst, runs, fn, h, stream);
      }, "src"_a, "dst"_a, "src_off"_a, "dst_off"_a, "length"_a, "fn"_a, "valid_start"_a = nb::none(),
         "valid_len"_a = nb::none(), "ctx"_a = "", "stream"_a = 0, "native"_a = nb::none())
      .def("copy_indices", [make_hints](ce::Engine& e, const ce::Endpoint& src, const ce::Endpoint& dst, nb::handle si,
                                        nb::handle di, const ce::Elementwise& fn, const std::string& ctx, uintptr_t stream,
                                        nb::handle native) {
        auto a = i64_arg(si), b = i64_arg(di);
        ce::Runs runs = ce::Runs::from_indices(a.span(), b.span());
        ce::Hints h = make_hints(nb::none(), nb::none(), ctx);
        h.native = py_payload(native);
        nb::gil_scoped_release release;
        e.copy(src, dst, runs, fn, h, stream);
      }, "src"_a, "dst"_a, "src_idx"_a, "dst_idx"_a, "fn"_a, "ctx"_a = "", "stream"_a = 0, "native"_a = nb::none())
      .def("copy_stmt", [make_hints](ce::Engine& e, const ce::Endpoint& src, const ce::Endpoint& dst, const ce::Stmt& stmt,
                                     const ce::Elementwise& fn, nb::handle valid_start, nb::handle valid_len,
                                     const std::string& ctx, uintptr_t stream, nb::handle native) {
        ce::Hints h = make_hints(valid_start, valid_len, ctx);
        h.native = py_payload(native);
        nb::gil_scoped_release release;
        e.copy(src, dst, stmt, fn, h, stream);
      }, "src"_a, "dst"_a, "stmt"_a, "fn"_a, "valid_start"_a = nb::none(), "valid_len"_a = nb::none(),
         "ctx"_a = "", "stream"_a = 0, "native"_a = nb::none())
      .def("tile", [](ce::Engine& e, const ce::TileSpec& spec, const std::string& ctx, uintptr_t stream,
                      nb::handle native) {
        ce::Hints h;
        h.ctx = ctx;
        h.native = py_payload(native);
        nb::gil_scoped_release release;
        e.tile(spec, h, stream);
      }, "spec"_a, "ctx"_a = "", "stream"_a = 0, "native"_a = nb::none())
      .def("copy_strided", [](ce::Engine& e, const ce::Endpoint& src, const ce::Endpoint& dst,
                              const std::vector<i64>& s_shape, const std::vector<i64>& s_strides, i64 s_off,
                              const std::vector<i64>& d_shape, const std::vector<i64>& d_strides, i64 d_off,
                              const ce::Elementwise& fn, uintptr_t stream, nb::handle native, bool host_async) {
        ce::Access da = ce::Access::strided(d_shape, d_strides, d_off);
        ce::Access sa = ce::broadcast_to(ce::Access::strided(s_shape, s_strides, s_off), d_shape);
        ce::Stmt stmt(std::move(da), std::move(sa));
        ce::Hints h;
        h.native = py_payload(native);
        h.host_async = host_async;
        nb::gil_scoped_release release;
        e.copy(src, dst, stmt, fn, h, stream);
      }, "src"_a, "dst"_a, "s_shape"_a, "s_strides"_a, "s_off"_a, "d_shape"_a, "d_strides"_a, "d_off"_a, "fn"_a,
         "stream"_a, "native"_a = nb::none(), "host_async"_a = false)
      .def_prop_ro("handle", [](ce::Engine& e) { return reinterpret_cast<uintptr_t>(&e); })
      .def("fence", [](ce::Engine& e, const std::vector<std::tuple<std::string, int, uintptr_t, uintptr_t>>& reads,
                       const std::vector<std::tuple<std::string, int, uintptr_t, uintptr_t>>& writes) {
        auto conv = [](const auto& v) {
          std::vector<ce::Range> out;
          for (auto& [kind, index, lo, hi] : v)
            out.push_back({kind == "cuda" ? ce::Device{ce::DeviceKind::cuda, index} : ce::Device{ce::DeviceKind::host, 0},
                           lo, hi});
          return out;
        };
        const auto r = conv(reads), w = conv(writes);
        nb::gil_scoped_release release;
        e.fence(r, w);
      }, "reads"_a, "writes"_a)
      .def("begin_scope", nb::overload_cast<bool, double>(&ce::Engine::begin_scope), "host_async"_a = false,
           "bandwidth"_a = 0.0)
      .def("end_scope", &ce::Engine::end_scope, nb::call_guard<nb::gil_scoped_release>())
      .def("flush", &ce::Engine::flush, nb::call_guard<nb::gil_scoped_release>())
      .def("synchronize", &ce::Engine::synchronize, nb::call_guard<nb::gil_scoped_release>())
      .def("decline", [](ce::Engine& e, const std::string& why) { e.decline(why); }, "reason"_a)
      .def("mark_pending", &ce::Engine::mark_pending)
      .def("done", &ce::Engine::done, "token"_a)
      .def_prop_ro("depth", &ce::Engine::depth)
      .def_prop_ro("queued", &ce::Engine::queued)
      .def("stats_json", &ce::Engine::stats_json)
      .def_prop_ro("tuning_runs", &ce::Engine::tuning_runs)
      .def("dump_profile", &ce::Engine::dump_profile)
      .def("load_profile", &ce::Engine::load_profile)
      .def("freeze", [](const ce::Engine& e, bool rebindable) { return e.freeze(rebindable); }, "rebindable"_a = false)
      .def("freeze_composed",
           [](const ce::Engine& e, const std::vector<std::array<int64_t, 3>>& dead, bool rebindable) {
             return e.freeze_composed(dead, rebindable);
           },
           "dead"_a, "rebindable"_a = false)
      .def("release_payloads", &ce::Engine::release_payloads)
      .def_prop_ro("flushes", &ce::Engine::flushes)
      .def("set_recording", &ce::Engine::set_recording);

  m.def("hardware_fingerprint", &ce::hardware_fingerprint);
  m.def("host_is_pinned", &ce::host_is_pinned, "ptr"_a);
  // torch's float8_e4m3fn overflow mode (convert.hpp): set once at import, before any copy.
  m.def("_set_f8e4m3_saturate", [](bool v) { ce::cv::f8e4m3_saturate = v ? 1 : 0; });
  m.def("_f8e4m3_saturate", [] { return ce::cv::f8e4m3_saturate != 0; });
  // Address of the FrontendApi table handed to copyengine._torch.
  m.def("_frontend_api", [] { return reinterpret_cast<uintptr_t>(ce::frontend_api()); });

  // Test hooks: the quantization catalog's host implementation, unit by unit
  // (tests/python/test_quant_ref.py checks it against the torch recipes).
  m.def("_quant_ref", [](int fmt, nb::ndarray<const float, nb::ndim<1>, nb::c_contig, nb::device::cpu> x, int group,
                         float eps, float gs) {
    const auto f = static_cast<ce::cv::QFmt>(fmt);
    const size_t n = x.shape(0), units = n / static_cast<size_t>(group);
    const size_t per = ce::cv::qfmt_packed(f) ? static_cast<size_t>(group) / 2 : static_cast<size_t>(group);
    std::string q(units * per, '\0'), scale(units * 4, '\0'), bits(units, '\0');
    for (size_t u = 0; u < units; ++u) {
      const float* xp = x.data() + u * static_cast<size_t>(group);
      float s = 0;
      uint8_t sb = 0;
      ce::cv::quantize_unit([&](long long i) { return xp[i]; }, 1, group, f, eps, gs,
                            reinterpret_cast<uint8_t*>(q.data()) + u * per, 1, &s, &sb);
      std::memcpy(scale.data() + u * 4, &s, 4);
      bits[u] = static_cast<char>(sb);
    }
    return nb::make_tuple(nb::bytes(q.data(), q.size()), nb::bytes(scale.data(), scale.size()),
                          nb::bytes(bits.data(), bits.size()));
  }, "fmt"_a, "x"_a, "group"_a, "eps"_a = 1e-12f, "gs"_a = 1.0f);
  m.def("_dequant_ref", [](int fmt, nb::bytes q, nb::bytes scale, nb::bytes bits, int group, float gs) {
    const auto f = static_cast<ce::cv::QFmt>(fmt);
    const size_t units = bits.size(), n = units * static_cast<size_t>(group);
    const size_t per = ce::cv::qfmt_packed(f) ? static_cast<size_t>(group) / 2 : static_cast<size_t>(group);
    std::string y(n * 4, '\0');
    const auto* qp = reinterpret_cast<const uint8_t*>(q.c_str());
    for (size_t u = 0; u < units; ++u) {
      float s;
      std::memcpy(&s, scale.c_str() + u * 4, 4);
      const float factor = ce::cv::unit_scale(f, s, static_cast<uint8_t>(bits.c_str()[u]), gs);
      for (int i = 0; i < group; ++i) {
        const float v = ce::cv::dequantize_at(qp + u * per, 1, i, f, factor, static_cast<uint8_t>(bits.c_str()[u]));
        std::memcpy(y.data() + (u * static_cast<size_t>(group) + static_cast<size_t>(i)) * 4, &v, 4);
      }
    }
    return nb::bytes(y.data(), y.size());
  }, "fmt"_a, "q"_a, "scale"_a, "bits"_a, "group"_a, "gs"_a = 1.0f);
}
