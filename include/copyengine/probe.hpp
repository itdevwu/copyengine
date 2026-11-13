#pragma once

#include <string>

namespace ce {

struct ProbeOptions {
  // Also run bandwidth / small-copy microbenchmarks (takes ~1 s per device).
  bool bandwidth = false;
};

// Platform probe (design: query symbols, call once, verify bit-exact, then admit).
// Returns a JSON document. Unsupported features are reported with a reason,
// never silently treated as absent.
std::string probe_json(const ProbeOptions& options = {});

}  // namespace ce
