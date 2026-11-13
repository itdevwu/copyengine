#pragma once

#include "copyengine/probe.hpp"
#include "core/json.hpp"

namespace ce::cuda {

// {"available": bool, "reason"?: str, "driver_version": int, "devices": [...]}
Json probe(const ProbeOptions& options);

}  // namespace ce::cuda
