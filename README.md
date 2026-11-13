<div align="center">

# copyengine

_Deliver optimal copy performance effortless._

</div>

Hand the copies scattered through your code (`copy_`, gather / scatter, host ↔
GPU transfers, layout changes, casts) to copyengine. It merges them, decides
where to reorder and convert, and tunes itself to the hardware it runs on.
Existing code mostly stays as it is.

## Before / after

```python
# Before: layers × blocks small copies, each its own call (tens of thousands of calls)
for layer in range(L):
    for blk in blocks:
        scale_k[layer][blk].copy_(k_qs_buf[blk].expand(N, S))

# After: one line of wrapping. The loop is unchanged; its copies are merged
# into a few transfers and kernels.
with ce.scope():
    for layer in range(L):
        for blk in blocks:
            scale_k[layer][blk].copy_(k_qs_buf[blk].expand(N, S))
```

Loops like this are usually rewritten by hand into a few vectorized copies or a
custom kernel. copyengine aims to get that result without the rewrite, and
without a new kernel for every new model or cache layout.

## Where it helps

- **Inference**: loading and offloading KV cache blocks between host and GPU,
  paged caches (gather / scatter by block id), quantizing or dequantizing KV on
  the way (int8 with a scale), swapping LoRA weights in and out.
- **Training**: offloading optimizer state and activations, dataloader H2D,
  checkpoint save / load and resharding (pure reordering).
- **Model code**: `x.permute(...).contiguous().cuda()`, copying after
  `index_select`, permute-then-cast: write it as one `ce.copy_` into the
  destination (with `ce.lazy` for gathers), and it happens in one pass, with
  the reordering done on the GPU.

copyengine moves and converts data. Computation such as reductions is out of
scope: it runs as usual, ordered correctly around the copies.

## What it guarantees

1. **Not slower than the original.** The framework's own copy (torch's
   `copy_`) is always a candidate. Anything else is measured against it,
   including the time copyengine itself spends planning and submitting, and is
   kept only if it is clearly faster. Decisions keep being re-checked while
   your program runs.
2. **The same results.** copyengine only takes a copy when it can produce
   exactly torch's result; otherwise torch runs it. Host and GPU conversions
   are bit-identical.
3. **Easy to turn off.** `CE_DISABLE=1` turns copyengine's choices off: every
   copy runs through torch's own `copy_`, or exactly as submitted where torch
   has no equivalent.
4. **Bounded resources.** Device staging memory has a budget (`CE_STAGING_MB`).

The first runs of each new copy pattern are a warmup: copyengine measures its
options, then locks the fastest. Save the result with `eng.save_profile(path)`
to skip the warmup next time on the same machine.

## Install

```bash
pip install .                      # needs CMake >= 3.26 and a C++20 compiler
                                   # GPU support: CUDA toolkit headers + nvcc at build time
pip install --no-build-isolation . # same, with torch already installed: also builds the torch integration
```

Requirements: Linux x86_64, Python ≥ 3.10. GPU support runs on CUDA 12.x and
13.x drivers; the driver is loaded at runtime, nothing links against `libcuda`
or `cudart`.

The torch integration is built against the torch you have installed. If you
later switch torch versions, the part that intercepts `copy_` is recompiled
once on import (needs a C++ compiler, takes seconds, cached). If that is not
possible, scopes fall back to a pure-Python interception with the same
results, only slower, and a warning says so.

## Use

```python
import copyengine as ce
import torch

# 1. Unchanged torch code: copy_ inside a scope is queued and merged.
with ce.scope():
    for l in range(num_layers):
        for i, b in enumerate(block_ids):
            kv_cache[l, b].copy_(host_slot[l, i])

# 2. A copy torch cannot express as one view: gather + permute + cast in one pass.
src = ce.lazy(cache).index_select(0, ids).permute(0, 2, 1, 3)
ce.copy_(out_fp16, src)

# 3. Quantize / dequantize on the way.
ce.copy_(q_int8, x, fn=ce.Elementwise(scale=1 / qs, round=True, clamp=(-127, 127)))

# 4. Ragged or CSR layouts: many segments in one call.
ce.segments(src, dst, src_off, dst_off, length)

# 5. Replay a planned copy without planning again.
eng = ce.Engine()
with eng.scope():
    ...
plan = eng.freeze()
plan.run()
```

More: `ce.gather` / `ce.scatter_` for index-based copies, `ce.barrier()` to
wait for queued work, `ce.probe()` to see which copy mechanisms this machine
supports (each verified by a real copy). C and C++ callers use
`include/copyengine/copyengine.h` (a stable C ABI) or `copyengine::copyengine`
from CMake.

To compare against torch on your GPU: `python bench/bench_torch.py`.

## How it works

Copies issued inside a scope are queued instead of run. When the scope ends,
copyengine looks at them together: which ones depend on each other (exact
address ranges), which can be merged across tensors, and what structure they
form (2-D blocks, strides, broadcasts). For each group it picks among merged
DMA calls, batched DMA (`cuMemcpyBatchAsync`), 2-D copies, copy kernels that
also cast and scale, moving data densely over the link and reordering it on the
GPU, or pinned bounce buffers for pageable memory, and measures which is
fastest.

## Rules worth knowing

- Inside a scope, every other op is ordered after the queued copies whose
  memory it reads or writes, so code in the scope sees the copied data.
  Memory read without going through torch (`.numpy()`, raw pointers) needs
  `ce.barrier()` first.
- GPU work runs on the current stream, like torch's own copies. GPU-to-GPU
  copies are not waited for; intercepted `copy_` keeps its `non_blocking`
  behavior; the explicit `ce.copy_` returns with any host-side copy complete
  (unless the engine was created with `allow_async=True`).
- Each destination element must be written once per copy: duplicate scatter
  indices and a source overlapping its own destination raise `UnsafeCopy` in
  the explicit API (an intercepted `copy_` leaves those to torch).

## Environment

| Variable | Effect |
|---|---|
| `CE_DISABLE=1` | turn copyengine's choices off: torch's own copy, or each copy as submitted |
| `CE_EXPLORE=0` | never measure: use the loaded profile; copy patterns it does not cover run natively (or with copyengine's default where torch has no equivalent) and are listed in the profile for the next warmup |
| `CE_STAGING_MB` | device staging budget in MB (default 256) |
| `CE_LOG=plan` | print plans to stderr |
| `CE_SYNC=1` | wait for every copy to finish (debugging) |
| `CE_SHIM=0` | skip the compiled `copy_` interception; scopes intercept in Python |
| `CE_CACHE_DIR` | where interception builds compiled on import are cached (default `~/.cache/copyengine`) |

Licensed under Apache-2.0.
