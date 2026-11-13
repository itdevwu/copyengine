#include "copyengine/copyengine.h"

#include "copyengine/copyengine.hpp"

extern "C" {

uint32_t ce_abi_version(void) { return CE_ABI_VERSION; }

const char* ce_version_string(void) { return ce::version_string().data(); }

}  // extern "C"
