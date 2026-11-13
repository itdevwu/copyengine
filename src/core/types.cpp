#include "copyengine/types.hpp"

#include <string>

#include "copyengine/error.hpp"

namespace ce {

namespace {
struct Info {
  std::string_view name;
  size_t size;
};
constexpr Info kInfo[kNumDTypes] = {
    {"float64", 8}, {"float32", 4}, {"float16", 2}, {"bfloat16", 2}, {"int64", 8},
    {"int32", 4},   {"int16", 2},   {"int8", 1},    {"uint8", 1},    {"bool", 1},
};
}  // namespace

size_t itemsize(DType t) { return kInfo[static_cast<int>(t)].size; }

std::string_view dtype_name(DType t) { return kInfo[static_cast<int>(t)].name; }

DType parse_dtype(std::string_view name) {
  for (int i = 0; i < kNumDTypes; ++i)
    if (kInfo[i].name == name) return static_cast<DType>(i);
  fail(Errc::invalid_argument, "unsupported dtype: " + std::string(name));
}

}  // namespace ce
