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

// Stable identity of the hardware a tuning profile applies to: CPU model,
// GPU models and compute capabilities, PCIe link, driver version.
std::string hardware_fingerprint();

}  // namespace ce
