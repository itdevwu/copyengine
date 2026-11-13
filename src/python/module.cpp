#include <nanobind/nanobind.h>

#include "copyengine/copyengine.h"
#include "copyengine/copyengine.hpp"

namespace nb = nanobind;

NB_MODULE(_core, m) {
  const std::string_view v = ce::version_string();
  m.attr("__version__") = nb::str(v.data(), v.size());
  m.attr("ABI_VERSION") = ce_abi_version();
}
