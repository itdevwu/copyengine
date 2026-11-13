#pragma once

#include "copyengine/probe.hpp"
#include "core/json.hpp"

namespace ce::cuda {

// {"available": bool, "reason"?: str, "driver_version": int, "devices": [...]}
Json probe(const ProbeOptions& options);

// "driver=13010;gpu0=<name>/8.6/pcie4x8;..." or "cuda=none".
std::string fingerprint();

}  // namespace ce::cuda
