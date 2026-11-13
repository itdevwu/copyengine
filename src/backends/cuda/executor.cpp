#include "backends/cuda/executor.hpp"

#include "copyengine/error.hpp"

namespace ce {

std::unique_ptr<Executor> make_cuda_executor(const EngineOptions&) {
  fail(Errc::unsupported, "CUDA executor not implemented yet");
}

}  // namespace ce
