#include <cstdio>
#include <cstring>
#include <string>

#include "copyengine/copyengine.h"
#include "copyengine/probe.hpp"

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
      return 1;                                                              \
    }                                                                        \
  } while (0)

int main() {
  const std::string s = ce::probe_json();
  CHECK(s.front() == '{' && s.back() == '}');
  CHECK(s.find("\"cuda\":{\"available\":") != std::string::npos);

  // C ABI: size query, then truncation stays NUL-terminated.
  const size_t n = ce_probe_json(0, nullptr, 0);
  CHECK(n != static_cast<size_t>(-1) && n > 2);
  char small[8];
  CHECK(ce_probe_json(0, small, sizeof small) == n);
  CHECK(std::strlen(small) == sizeof small - 1);
  return 0;
}
