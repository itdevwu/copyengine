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

#ifdef __cplusplus
}
#endif

#endif /* COPYENGINE_COPYENGINE_H */
