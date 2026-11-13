#include <nanobind/nanobind.h>

#include "copyengine/copyengine.h"
#include "copyengine/copyengine.hpp"
#include "copyengine/probe.hpp"

namespace nb = nanobind;

NB_MODULE(_core, m) {
  const std::string_view v = ce::version_string();
  m.attr("__version__") = nb::str(v.data(), v.size());
  m.attr("ABI_VERSION") = ce_abi_version();
  m.def(
      "probe_json",
      [](bool bandwidth) {
        std::string s;
        {
          nb::gil_scoped_release release;
          s = ce::probe_json({.bandwidth = bandwidth});
        }
        return nb::str(s.data(), s.size());
      },
      nb::arg("bandwidth") = false);
}
