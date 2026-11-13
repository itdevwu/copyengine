<div align="center">

# copyengine

_Deliver optimal copy performance effortless._

</div>

A software copy planner for ML training and inference infra. It batches, fuses
and schedules tensor copies (gather / scatter, H2D / D2H, layout changes, casts).

Copies issued inside a scope are queued, analyzed together (exact address-range
dependencies, merging across tensors, 2-D structure recovery), and issued with
the strategy measured fastest on this machine: merged DMA, batched DMA
(`cuMemcpyBatchAsync`), 2-D copies, table-driven copy kernels that also cast and
scale, dense link copies with the layout change done on the GPU, or pinned
bounce buffers for pageable memory.

The framework's own copy (torch's `copy_`, numpy's `copyto`) is the baseline.
Merges that at least halve the number of calls are applied directly; every
other strategy is measured against the baseline in paired runs, including the
host time ce spends planning and submitting, and kept only if it is
significantly faster. Otherwise the baseline runs, and a lone copy whose class
loses runs natively without entering the queue.

## Install

```bash
pip install .                      # needs CMake >= 3.26, a C++20 compiler
                                   # CUDA backend: CUDA toolkit headers + nvcc at build time
```

With torch importable at build time (`pip install --no-build-isolation .`), the
wheel also carries the scope's interception shim, built for that torch, and,
with torch 2.9 or newer, the explicit copy API as custom ops on torch's stable
ABI. The shim follows torch's C++ API, so for any other torch it is compiled on
first import (a C++ compiler, a few seconds, cached per torch build). Without
it, scopes intercept in Python: the same results, only slower.

The CUDA driver is loaded at runtime; nothing links against `libcuda` or
`cudart`, and one build runs on CUDA 12.x and 13.x drivers.

## Use

```python
import copyengine as ce
import torch

# Unchanged torch code: inside a scope, copy_ is queued and merged.
with ce.scope():
    for l in range(num_layers):
        for i, b in enumerate(block_ids):
            kv_cache[l, b].copy_(host_slot[l, i])

# Copies torch cannot express as one view: gather + permute + cast in one pass.
src = ce.lazy(cache).index_select(0, ids).permute(0, 2, 1, 3)
ce.copy_(out_fp16, src)

# Quantize / dequantize on the way.
ce.copy_(q_int8, x, fn=ce.Elementwise(scale=1 / qs, round=True, clamp=(-127, 127)))

# Explicit run lists (ragged / CSR layouts): one call for many segments.
ce.segments(src, dst, src_off, dst_off, length)

# Replay a planned copy without re-planning.
eng = ce.Engine()
with eng.scope():
    ...
plan = eng.freeze()
plan.run()
```

Tuning results persist per machine: `eng.save_profile(path)` /
`eng.load_profile(open(path).read())`. `ce.probe()` reports which copy
mechanisms this platform supports, each verified by an actual copy.

C and C++ callers use `include/copyengine/copyengine.h` (stable C ABI) or
`copyengine::copyengine` from CMake.

## Semantics

- Each destination element is written once per copy; duplicate scatter indices
  and overlapping source/destination raise `UnsafeCopy`.
- Inside a scope, `Tensor.copy_` is queued when ce can produce torch's exact
  result (otherwise torch runs it), and every other op is ordered after the
  queued copies whose memory it reads or writes. Results are torch's; only
  timing differs. Intercepted copies keep their `non_blocking` semantics.
  Memory read without an op (`.numpy()`, raw pointers) needs `barrier()` first.
- CUDA work is ordered on the caller's current stream (torch's, if any), like
  torch's own copies: device-to-device copies are not waited for. Copies that
  touch host memory have completed when the copy or scope returns, unless the
  engine was created with `allow_async=True`.
- Host and GPU conversions produce identical bits; see `include/copyengine/types.hpp`.

## Environment

| Variable | Effect |
|---|---|
| `CE_DISABLE=1` | run every copy natively (as submitted where there is no native equivalent) |
| `CE_LOG=plan` | print plans to stderr |
| `CE_STAGING_MB` | device staging budget (default 256) |
| `CE_EXPLORE=0` | never measure: use the loaded profile; buckets it does not cover take the baseline and are listed in the profile for the next warmup |
| `CE_SYNC=1` | wait for every copy, device-to-device included (debugging) |
| `CE_SHIM=0` | do not load the C++ interception shim (scopes intercept in Python) |
| `CE_CACHE_DIR` | where shims compiled on import are cached (default `~/.cache/copyengine`) |

Licensed under Apache-2.0.
