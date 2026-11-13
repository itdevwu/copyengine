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
bounce buffers for pageable memory. The unmerged original is always a candidate,
so tuning cannot make a copy slower than issuing it as written.

## Install

```bash
pip install .                      # needs CMake >= 3.26, a C++20 compiler
                                   # CUDA backend: CUDA toolkit headers + nvcc at build time
```

The CUDA driver is loaded at runtime; nothing links against `libcuda` or
`cudart`, and one build runs on CUDA 12.x and 13.x drivers.

## Use

```python
import copyengine as ce
import torch

# Merge a per-layer, per-block loop into a few DMA calls.
with ce.scope():
    for l in range(num_layers):
        for i, b in enumerate(block_ids):
            ce.copy_(kv_cache[l, b], host_slot[l, i])

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
- Buffers used inside a scope must not be touched by other code until the scope
  exits (or `barrier()`); on exit all copies have completed unless the engine
  was created with `allow_async=True`.
- CUDA work is ordered on the caller's current stream (torch's, if any).
- Host and GPU conversions produce identical bits; see `include/copyengine/types.hpp`.

## Environment

| Variable | Effect |
|---|---|
| `CE_DISABLE=1` | run every copy exactly as submitted |
| `CE_LOG=plan` | print plans to stderr |
| `CE_STAGING_MB` | device staging budget (default 256) |
| `CE_TUNE=0` | use the prior ranking only, never measure |

Licensed under Apache-2.0.
