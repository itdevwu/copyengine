/* copyengine stable C ABI. Every symbol is prefixed `ce_`. */
#ifndef COPYENGINE_COPYENGINE_H
#define COPYENGINE_COPYENGINE_H

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

#ifdef __cplusplus
}
#endif

#endif /* COPYENGINE_COPYENGINE_H */
