/* copyengine stable C ABI. Every symbol is prefixed `ce_`. */
#ifndef COPYENGINE_COPYENGINE_H
#define COPYENGINE_COPYENGINE_H

#include <stddef.h>
#include <stdint.h>

#if defined(CE_BUILDING_LIBRARY)
#define CE_API __attribute__((visibility("default")))
#else
#define CE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any incompatible change to this header. */
#define CE_ABI_VERSION 0u

CE_API uint32_t ce_abi_version(void);
CE_API const char* ce_version_string(void);

/* Flags for ce_probe_json. */
#define CE_PROBE_BANDWIDTH 1u

/* Writes the platform probe report (JSON, NUL-terminated) into buf.
 * Returns the full length excluding the NUL; if that is >= cap, the output was
 * truncated and the call should be repeated with a larger buffer.
 * Returns (size_t)-1 on internal error. */
CE_API size_t ce_probe_json(uint32_t flags, char* buf, size_t cap);

/* ---- engine ------------------------------------------------------------------
 * All functions are thread-safe per engine. Errors return a ce_status and set
 * a thread-local message readable with ce_last_error(). */

typedef enum ce_status {
  CE_OK = 0,
  CE_ERR_INVALID = 1,      /* bad argument: shapes, bounds, dtypes */
  CE_ERR_MATERIALIZE = 2,  /* view needs a contiguous copy first */
  CE_ERR_UNSAFE = 3,       /* duplicate destination or src/dst overlap: run it yourself */
  CE_ERR_UNSUPPORTED = 4,  /* this build / platform cannot run the request */
  CE_ERR_BACKEND = 5,      /* a platform call failed */
  CE_ERR_INTERNAL = 6
} ce_status;

typedef enum ce_dtype {
  CE_F64 = 0, CE_F32 = 1, CE_F16 = 2, CE_BF16 = 3, CE_I64 = 4,
  CE_I32 = 5, CE_I16 = 6, CE_I8 = 7, CE_U8 = 8, CE_BOOL = 9
} ce_dtype;

typedef enum ce_device_kind { CE_DEVICE_HOST = 0, CE_DEVICE_CUDA = 1 } ce_device_kind;

/* One side of a copy: [base, base + nbytes) of an allocation, viewed as dtype. */
typedef struct ce_endpoint {
  int32_t device_kind;  /* ce_device_kind */
  int32_t device_index;
  uint64_t base;
  int64_t nbytes;
  int32_t dtype;        /* ce_dtype */
  int32_t pinned;       /* host memory is page-locked */
} ce_endpoint;

/* y = clamp(round(x * scale + bias)) converted to the destination dtype. */
typedef struct ce_elementwise {
  float scale, bias;
  int32_t round, clamp;
  float lo, hi;
} ce_elementwise;

typedef struct ce_engine_options {
  uint32_t struct_size; /* sizeof(ce_engine_options), for forward compatibility */
  int64_t staging_budget;
  int32_t explore; /* measure against the baseline; 0: loaded profile only */
  int32_t tune_trials, tune_topk;
  int32_t allow_async, disable, log_plan;
  int32_t host_threads;
} ce_engine_options;

typedef struct ce_engine ce_engine;

/* Defaults with CE_* environment variables applied. */
CE_API void ce_engine_options_default(ce_engine_options* options);
CE_API ce_status ce_engine_create(const ce_engine_options* options, ce_engine** out);
CE_API void ce_engine_destroy(ce_engine* engine);

/* dst[dst_off[i] + k] = fn(src[src_off[i] + k]) for k < len[i]; element offsets
 * relative to each endpoint's base. fn may be NULL (identity). stream is a
 * CUstream (0 = legacy default stream) or 0 for host-only copies. */
CE_API ce_status ce_copy_runs(ce_engine* engine, const ce_endpoint* src, const ce_endpoint* dst,
                              const int64_t* src_off, const int64_t* dst_off, const int64_t* len, size_t n,
                              const ce_elementwise* fn, uint64_t stream);

CE_API ce_status ce_scope_begin(ce_engine* engine);
CE_API ce_status ce_scope_end(ce_engine* engine);  /* plans, issues, and waits unless allow_async */
CE_API ce_status ce_flush(ce_engine* engine);
CE_API ce_status ce_synchronize(ce_engine* engine);

/* JSON outputs follow ce_probe_json's buffer convention. */
CE_API size_t ce_engine_stats_json(ce_engine* engine, char* buf, size_t cap);
CE_API size_t ce_profile_dump(ce_engine* engine, char* buf, size_t cap);
CE_API ce_status ce_profile_load(ce_engine* engine, const char* json);

/* Message of the last failed call on this thread ("" if none). */
CE_API const char* ce_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* COPYENGINE_COPYENGINE_H */
