#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "copyengine/copyengine.h"
#include "copyengine/copyengine.hpp"
#include "copyengine/error.hpp"
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
  static nb::object asarray = nb::module_::import_("numpy").attr("ascontiguousarray");
  nb::object a = asarray(obj, "dtype"_a = "int64").attr("reshape")(-1);
  return {a, nb::cast<I64In>(a)};
}

// Returned as a plain object so property getters don't apply reference_internal.
nb::object to_numpy(std::vector<i64> v) {
  auto* heap = new std::vector<i64>(std::move(v));
  nb::capsule owner(heap, [](void* p) noexcept { delete static_cast<std::vector<i64>*>(p); });
  return nb::cast(I64Out(heap->data(), {heap->size()}, owner), nb::rv_policy::move);
}

}  // namespace

NB_MODULE(_core, m) {
  const std::string_view v = ce::version_string();
  m.attr("__version__") = nb::str(v.data(), v.size());
  m.attr("ABI_VERSION") = ce_abi_version();

  // ---- errors ----
  static nb::object base_exc = nb::steal(PyErr_NewException("copyengine.CopyEngineError", PyExc_RuntimeError, nullptr));
  static nb::object materialize_exc = nb::steal(PyErr_NewException("copyengine.NeedsMaterialize", base_exc.ptr(), nullptr));
  static nb::object unsafe_exc = nb::steal(PyErr_NewException("copyengine.UnsafeCopy", base_exc.ptr(), nullptr));
  static nb::object unsupported_exc = nb::steal(PyErr_NewException("copyengine.Unsupported", base_exc.ptr(), nullptr));
  static nb::object backend_exc = nb::steal(PyErr_NewException("copyengine.BackendError", base_exc.ptr(), nullptr));
  m.attr("CopyEngineError") = base_exc;
  m.attr("NeedsMaterialize") = materialize_exc;
  m.attr("UnsafeCopy") = unsafe_exc;
  m.attr("Unsupported") = unsupported_exc;
  m.attr("BackendError") = backend_exc;
  nb::register_exception_translator([](const std::exception_ptr& p, void*) {
    try {
      std::rethrow_exception(p);
    } catch (const ce::Error& e) {
      PyObject* type = PyExc_ValueError;
      switch (e.code()) {
        case ce::Errc::invalid_argument: type = PyExc_ValueError; break;
        case ce::Errc::needs_materialize: type = materialize_exc.ptr(); break;
        case ce::Errc::unsafe_copy: type = unsafe_exc.ptr(); break;
        case ce::Errc::unsupported: type = unsupported_exc.ptr(); break;
        case ce::Errc::backend: type = backend_exc.ptr(); break;
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
      .def("coalesce", &ce::Runs::coalesce)
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
      .def("unsqueeze", &ce::Access::unsqueeze)
      .def("expand", &ce::Access::expand)
      .def("reshape", [](const ce::Access& a, const std::vector<i64>& s) { return a.reshape(s); })
      .def("offsets", [](const ce::Access& a) { return to_numpy(a.offsets()); })
      .def("extent", &ce::Access::extent);

  nb::class_<ce::Stmt>(m, "Stmt")
      .def(nb::init<ce::Access, ce::Access>(), "dst"_a, "src"_a)
      .def_ro("dst", &ce::Stmt::dst)
      .def_ro("src", &ce::Stmt::src)
      .def_prop_ro("domain", &ce::Stmt::domain);

  m.def("lower", &ce::lower, "stmt"_a, nb::call_guard<nb::gil_scoped_release>());
}
