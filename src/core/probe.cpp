#include "copyengine/probe.hpp"

#include <fstream>
#include <string>
#include <thread>

#include "copyengine/copyengine.hpp"
#include "core/json.hpp"

#if CE_WITH_CUDA
#include "backends/cuda/probe.hpp"
#endif

namespace ce {
namespace {

Json probe_host() {
  Json j = Json::object();
  std::string model;
  std::ifstream cpuinfo("/proc/cpuinfo");
  for (std::string line; std::getline(cpuinfo, line);) {
    if (line.rfind("model name", 0) == 0) {
      model = line.substr(line.find(':') + 2);
      break;
    }
  }
  j.set("cpu", model.empty() ? Json(nullptr) : Json(model));
  j.set("logical_cpus", std::thread::hardware_concurrency());
  return j;
}

}  // namespace

std::string probe_json(const ProbeOptions& options) {
  Json out = Json::object();
  out.set("copyengine", version_string());
  out.set("host", probe_host());
#if CE_WITH_CUDA
  out.set("cuda", cuda::probe(options));
#else
  (void)options;
  out.set("cuda", Json::object().set("available", false).set("reason", "built without CUDA support"));
#endif
  return out.dump();
}

}  // namespace ce
