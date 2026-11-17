#include "copyengine/copyengine.h"

#include <cstring>
#include <string>

#include "copyengine/copyengine.hpp"
#include "copyengine/engine.hpp"
#include "copyengine/error.hpp"
#include "copyengine/frontend.hpp"
#include "copyengine/probe.hpp"

struct ce_engine {
  ce::Engine engine;
  explicit ce_engine(const ce::EngineOptions& o) : engine(o) {}
};

namespace {

thread_local std::string g_error;

size_t emit(const std::string& s, char* buf, size_t cap) {
  if (buf && cap) {
    const size_t n = s.size() < cap ? s.size() : cap - 1;
    std::memcpy(buf, s.data(), n);
    buf[n] = '\0';
  }
  return s.size();
}

template <class F>
ce_status guard(F&& f) {
  try {
    f();
    g_error.clear();
    return CE_OK;
  } catch (const ce::Error& e) {
    g_error = e.what();
    switch (e.code()) {
      case ce::Errc::invalid_argument: return CE_ERR_INVALID;
      case ce::Errc::needs_materialize: return CE_ERR_MATERIALIZE;
      case ce::Errc::unsafe_copy: return CE_ERR_UNSAFE;
      case ce::Errc::unsupported: return CE_ERR_UNSUPPORTED;
      case ce::Errc::backend: return CE_ERR_BACKEND;
    }
    return CE_ERR_INTERNAL;
  } catch (const std::exception& e) {
    g_error = e.what();
    return CE_ERR_INTERNAL;
  } catch (...) {
    g_error = "unknown error";
    return CE_ERR_INTERNAL;
  }
}

ce::Endpoint endpoint(const ce_endpoint* e) {
  if (!e) ce::fail(ce::Errc::invalid_argument, "null endpoint");
  if (e->dtype < 0 || e->dtype >= ce::kNumDTypes) ce::fail(ce::Errc::invalid_argument, "bad dtype");
  ce::Device dev;
  if (e->device_kind == CE_DEVICE_HOST) dev = {ce::DeviceKind::host, 0};
  else if (e->device_kind == CE_DEVICE_CUDA) dev = {ce::DeviceKind::cuda, e->device_index};
  else ce::fail(ce::Errc::unsupported, "bad device kind");
  ce::Endpoint out{dev, static_cast<uintptr_t>(e->base), e->nbytes, static_cast<ce::DType>(e->dtype), false, false};
  // Host memory facts come from the driver (`pinned` in the C struct is a hint the driver overrides).
  if (dev.kind == ce::DeviceKind::host) ce::set_host_memory(out);
  return out;
}

}  // namespace

extern "C" {

uint32_t ce_abi_version(void) { return CE_ABI_VERSION; }

const char* ce_version_string(void) { return ce::version_string().data(); }

size_t ce_probe_json(uint32_t flags, char* buf, size_t cap) {
  try {
    return emit(ce::probe_json({.bandwidth = (flags & CE_PROBE_BANDWIDTH) != 0}), buf, cap);
  } catch (...) {
    return static_cast<size_t>(-1);
  }
}

void ce_engine_options_default(ce_engine_options* o) {
  if (!o) return;
  const ce::EngineOptions d = ce::EngineOptions::from_env();
  o->struct_size = sizeof(ce_engine_options);
  o->staging_budget = d.staging_budget;
  o->explore = d.explore, o->tune_trials = d.tune_trials, o->tune_topk = d.tune_topk;
  o->allow_async = d.allow_async, o->disable = d.disable, o->log_plan = d.log_plan;
  o->host_threads = d.host_threads;
}

ce_status ce_engine_create(const ce_engine_options* o, ce_engine** out) {
  return guard([&] {
    if (!out) ce::fail(ce::Errc::invalid_argument, "null output pointer");
    ce::EngineOptions opt = ce::EngineOptions::from_env();
    if (o) {
      if (o->struct_size < sizeof(ce_engine_options)) ce::fail(ce::Errc::invalid_argument, "options struct_size too small");
      opt.staging_budget = o->staging_budget;
      opt.explore = o->explore != 0, opt.tune_trials = o->tune_trials, opt.tune_topk = o->tune_topk;
      opt.allow_async = o->allow_async, opt.disable = o->disable, opt.log_plan = o->log_plan;
      opt.host_threads = o->host_threads;
    }
    *out = new ce_engine(opt);
  });
}

void ce_engine_destroy(ce_engine* e) { delete e; }

ce_status ce_copy_runs(ce_engine* e, const ce_endpoint* src, const ce_endpoint* dst, const int64_t* so,
                       const int64_t* d_o, const int64_t* len, size_t n, const ce_elementwise* fn, uint64_t stream) {
  return guard([&] {
    if (!e) ce::fail(ce::Errc::invalid_argument, "null engine");
    if (n && (!so || !d_o || !len)) ce::fail(ce::Errc::invalid_argument, "null run arrays");
    ce::Runs runs = ce::Runs::of({so, n}, {d_o, n}, {len, n});
    ce::Elementwise f;
    if (fn) f = {fn->scale, fn->bias, fn->round != 0, fn->clamp != 0, fn->lo, fn->hi};
    e->engine.copy(endpoint(src), endpoint(dst), runs, f, {}, static_cast<uintptr_t>(stream));
  });
}

ce_status ce_scope_begin(ce_engine* e) {
  return guard([&] { e->engine.begin_scope(); });
}
ce_status ce_scope_end(ce_engine* e) {
  return guard([&] { e->engine.end_scope(); });
}
ce_status ce_flush(ce_engine* e) {
  return guard([&] { e->engine.flush(); });
}
ce_status ce_synchronize(ce_engine* e) {
  return guard([&] { e->engine.synchronize(); });
}

size_t ce_engine_stats_json(ce_engine* e, char* buf, size_t cap) {
  try {
    return emit(e->engine.stats_json(), buf, cap);
  } catch (...) {
    return static_cast<size_t>(-1);
  }
}

size_t ce_profile_dump(ce_engine* e, char* buf, size_t cap) {
  try {
    return emit(e->engine.dump_profile(), buf, cap);
  } catch (...) {
    return static_cast<size_t>(-1);
  }
}

ce_status ce_profile_load(ce_engine* e, const char* json) {
  return guard([&] { e->engine.load_profile(json ? json : ""); });
}

const char* ce_last_error(void) { return g_error.c_str(); }

}  // extern "C"
