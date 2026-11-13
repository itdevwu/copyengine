#include <cstdio>
#include <cstring>

#include "copyengine/copyengine.h"
#include "copyengine/copyengine.hpp"

#define CHECK(cond)                                                     \
  do {                                                                  \
    if (!(cond)) {                                                      \
      std::fprintf(stderr, "%s:%d: CHECK(%s)\n", __FILE__, __LINE__, #cond); \
      return 1;                                                         \
    }                                                                   \
  } while (0)

int main() {
  CHECK(ce_abi_version() == CE_ABI_VERSION);
  CHECK(std::strcmp(ce_version_string(), ce::version_string().data()) == 0);
  auto v = ce::version();
  char buf[32];
  std::snprintf(buf, sizeof buf, "%d.%d.%d", v.major, v.minor, v.patch);
  CHECK(ce::version_string() == buf);
  return 0;
}
