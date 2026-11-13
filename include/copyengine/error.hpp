#pragma once

#include <stdexcept>
#include <string>

namespace ce {

enum class Errc {
  invalid_argument,  // caller error: shapes, ranges, dtypes
  needs_materialize, // the view cannot be expressed without a contiguous copy first
  unsafe_copy,       // semantics forbid optimizing: duplicate scatter targets, src/dst overlap
  unsupported,       // valid request this build / platform cannot run
  backend,           // a platform call failed
};

class Error : public std::runtime_error {
 public:
  Error(Errc code, const std::string& what) : std::runtime_error(what), code_(code) {}
  Errc code() const noexcept { return code_; }

 private:
  Errc code_;
};

[[noreturn]] inline void fail(Errc code, const std::string& what) { throw Error(code, what); }

}  // namespace ce
