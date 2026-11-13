#include "copyengine/copyengine.h"

#include <cstring>
#include <string>

#include "copyengine/copyengine.hpp"
#include "copyengine/probe.hpp"

extern "C" {

uint32_t ce_abi_version(void) { return CE_ABI_VERSION; }

const char* ce_version_string(void) { return ce::version_string().data(); }

size_t ce_probe_json(uint32_t flags, char* buf, size_t cap) {
  try {
    const std::string s = ce::probe_json({.bandwidth = (flags & CE_PROBE_BANDWIDTH) != 0});
    if (buf && cap) {
      const size_t n = s.size() < cap ? s.size() : cap - 1;
      std::memcpy(buf, s.data(), n);
      buf[n] = '\0';
    }
    return s.size();
  } catch (...) {
    return static_cast<size_t>(-1);
  }
}

}  // extern "C"
