/* Plain C: also proves the public header is valid C. */
#include <stdio.h>
#include <string.h>

#include "copyengine/copyengine.h"

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      fprintf(stderr, "%s:%d: CHECK(%s) %s\n", __FILE__, __LINE__, #cond, ce_last_error()); \
      return 1;                                                              \
    }                                                                        \
  } while (0)

int main(void) {
  float src[12 * 64], dst[12 * 100];
  for (int i = 0; i < 12 * 64; ++i) src[i] = (float)i;
  memset(dst, 0, sizeof dst);

  ce_engine_options o;
  ce_engine_options_default(&o);
  o.explore = 0;
  ce_engine* e = NULL;
  CHECK(ce_engine_create(&o, &e) == CE_OK);

  ce_endpoint s = {CE_DEVICE_HOST, 0, (uint64_t)(uintptr_t)src, sizeof src, CE_F32, 0};
  ce_endpoint d = {CE_DEVICE_HOST, 0, (uint64_t)(uintptr_t)dst, sizeof dst, CE_F32, 0};
  CHECK(ce_scope_begin(e) == CE_OK);
  for (int64_t l = 0; l < 12; ++l) {
    int64_t so = l * 64, dofs = l * 100, n = 64;
    CHECK(ce_copy_runs(e, &s, &d, &so, &dofs, &n, 1, NULL, 0) == CE_OK);
  }
  CHECK(ce_scope_end(e) == CE_OK);
  for (int l = 0; l < 12; ++l)
    for (int k = 0; k < 64; ++k) CHECK(dst[l * 100 + k] == src[l * 64 + k]);

  char stats[4096];
  CHECK(ce_engine_stats_json(e, stats, sizeof stats) < sizeof stats);
  CHECK(strstr(stats, "\"calls\":1,") != NULL);  /* 12 layer copies -> one 2-D copy */

  /* errors: out of bounds, duplicate destination */
  int64_t so = 0, dofs = 12 * 100 - 1, n = 2;
  CHECK(ce_copy_runs(e, &s, &d, &so, &dofs, &n, 1, NULL, 0) == CE_ERR_INVALID);
  CHECK(strstr(ce_last_error(), "out of bounds") != NULL);
  int64_t so2[2] = {0, 8}, do2[2] = {0, 0}, n2[2] = {8, 8};
  CHECK(ce_copy_runs(e, &s, &d, so2, do2, n2, 2, NULL, 0) == CE_ERR_UNSAFE);

  /* quantize f32 -> i8 */
  signed char q[64];
  ce_endpoint qd = {CE_DEVICE_HOST, 0, (uint64_t)(uintptr_t)q, sizeof q, CE_I8, 0};
  ce_elementwise fn = {0.5f, 0.0f, 1, 1, -127.0f, 127.0f};
  int64_t z = 0, len = 64;
  CHECK(ce_copy_runs(e, &s, &qd, &z, &z, &len, 1, &fn, 0) == CE_OK);
  CHECK(q[3] == 2 && q[5] == 2 && q[63] == 32);  /* 1.5 -> 2, 2.5 -> 2 (half-even), 31.5 -> 32 */

  ce_engine_destroy(e);
  return 0;
}
