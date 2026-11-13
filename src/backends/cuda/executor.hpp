#pragma once

#include <memory>

#include "backends/executor.hpp"

namespace ce {

// Throws if no usable CUDA driver / device is present.
std::unique_ptr<Executor> make_cuda_executor(const EngineOptions& opt);

}  // namespace ce
