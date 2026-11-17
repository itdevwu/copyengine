<div align="center">

# copyengine

_Deliver optimal copy performance effortless._

</div>

Hand the copies scattered through your code (`copy_`, gather / scatter, host ↔
GPU transfers, layout changes, casts, quantization on the way) to copyengine.
It merges them, picks how to move them on this machine, and keeps a choice
only where it measured faster than torch's own copy, with torch's results.

## Quick start

```bash
pip install copyengine
```

```python
import copyengine as ce

# A loop of copy_ calls: wrap it. The code stays as it is; its copies are
# queued and merged into a few transfers and kernels when the scope ends.
with ce.scope():
    for l in range(num_layers):
        for i, b in enumerate(block_ids):
            kv_cache[l, b].copy_(host_kv[l, i])

# Block ids that change every step (paged KV): plan the step once, pass the
# whole pools and the ids; later calls replay it with the new ids.
def load(dev_pools, host_pools, dev_ids, host_ids):
    for d, h in zip(dev_pools, host_pools):
        ce.copy_(ce.lazy(d)[dev_ids], ce.lazy(h)[host_ids])

step = ce.plan(load)
step(dev_pools, host_pools, dev_ids, host_ids)   # ids: lists, numpy arrays or int64 tensors

# Quantize while copying (FP8 / INT8 / INT4 / MXFP8 / NVFP4, a scale per group).
ce.quantize(kv, q_cache, scales, fmt="fp8_e4m3", group=128, x_index=pages, index=slots)
```

## Pick by situation

| Your code | Use |
|---|---|
| A loop of `copy_` calls you do not want to touch | `with ce.scope():` around it |
| The same step every iteration, block ids changing (paged caches, offload) | `step = ce.plan(fn)`, with `ce.lazy(pool)[ids]` inside; pass pools and ids as arguments |
| One copy per layer, the same tensors every step | `ce.copy_many(dsts, srcs)` |
| Gather + permute + cast before a copy (`x[ids].permute(...).to(...)`) | `ce.copy_(out, ce.lazy(x)[ids].permute(...))`: one pass, no temporaries |
| einops patterns on the way | `ce.rearrange(ce.lazy(x), "b h s d -> b s h d")`, also inside `torch.compile` |
| Quantize / dequantize while copying | `ce.quantize(...)`, `ce.dequantize(...)`; a single scale: `ce.copy_(q, x, fn=ce.Elementwise(scale=...))` |
| Copies that should overlap compute | `with ce.scope(stream="copy") as h:` ... `h.wait()` |
| Background copies that must leave SMs or bandwidth to compute | `ce.Engine(max_sms=4)`, `ce.scope(bandwidth=8e9)` |
| Ragged / CSR segments | `ce.segments(src, dst, src_off, dst_off, length)` |
| Exactly the same copies every step | `eng.freeze()` once, then `plan.run()` (also inside CUDA graphs) |

Inside `torch.compile`, the explicit calls (`ce.copy_`, `ce.copy_many`,
`ce.gather`, `ce.scatter_`, `ce.quantize`, lazy views) are graph nodes, each
measured against what torch.compile generates for the same operation, and the
faster one is kept: adding copyengine never makes a compiled model slower.

## Check that it works

```python
eng = ce.Engine()                         # or use the module functions above
...                                       # run your step a few times
eng.stats()["last"]["decisions"]          # per group: strategy and why ("locked: 63 us vs native 2632 us")
```

- `CE_DISABLE=1` runs every copy as torch's own: compare results and time.
- The first runs of each copy pattern measure the options, then lock the
  fastest. Warm up with `eng.warmup(step)` and `eng.save_profile(path)`; load
  it in serving with `CE_EXPLORE=0` to never measure there.
- `python bench/bench_torch.py` compares against torch on your GPU.

## What it guarantees

1. **Not slower than torch.** Torch's own copy is always a candidate where
   torch has the same copy; anything else is kept only if it measured clearly
   faster, counting copyengine's own planning and submission time, and is
   re-checked while the program runs. A scope call site, a `copy_many` call
   and a step plan are each compared with not using copyengine at all, and
   fall back where that does not clearly win (`stats()["scopes"]`,
   `stats()["copy_many"]`). Where torch has no equivalent (a conversion with
   `fn=`, a reversed slice), copyengine picks the fastest of its own ways.
2. **The same results.** Copies (moving and converting values) give exactly
   torch's bits, on the host and every GPU (NaN stays NaN; its payload bits are
   not kept, torch's own differ between CPU and CUDA). Torch's copy is used as
   the baseline only where its bits are copyengine's. Quantization computes,
   and computation has no single torch result: copyengine follows its written
   recipe (`copyengine/_quant.py`), the same bits everywhere.
3. **Easy to turn off.** `CE_DISABLE=1`: every copy through torch's `copy_`, or
   exactly as submitted where torch has no equivalent.
4. **Bounded resources.** With torch, copyengine's device memory and pinned
   buffers come from torch's allocators (they show in
   `torch.cuda.memory_allocated()`); device staging has a budget
   (`CE_STAGING_MB`); `Engine(max_sms=N)` keeps copy kernels on at most N SMs
   (a green context where the driver supports it, verified on start); DMA uses
   the copy engines, and `bandwidth=` paces it.

## Install details

The wheels include the GPU kernels and the torch integration and use the torch
you have. From source:

```bash
pip install .                      # CMake >= 3.26, a C++20 compiler; GPU: CUDA headers + nvcc
pip install --no-build-isolation . # with torch installed: also builds the torch integration
```

- Linux on x86_64 or aarch64 (server ARM such as Grace; not Jetson), Python ≥
  3.10, CUDA 12.x or 13.x drivers (loaded at runtime; nothing links `libcuda`
  or `cudart`).
- torch 2.9 or newer gives the same results; **2.10 or newer** runs the
  explicit API as a compiled torch op (a few microseconds less per call).
  After a torch upgrade, the `copy_` interception recompiles once on import
  (needs a C++ compiler; else scopes intercept in Python, slower, with a
  warning).
- dtypes: float64/32/16, bfloat16, float8 (e4m3fn, e5m2, e8m0fnu),
  int64/32/16/8, uint8 and bool convert into each other; packed FP4
  (`float4_e2m1fn_x2`) is copied unconverted.
- Host memory may be pinned (`pin_memory()`), registered
  (`cudaHostRegister`) or pageable; pinned is fastest.

More: `ce.gather` / `ce.scatter_` for index-based copies, `ce.indices`,
`ce.barrier()` to wait for queued work, `ce.probe()` for the copy mechanisms
this machine has (each verified by a real copy). C and C++ callers use
`include/copyengine/copyengine.h` (a stable C ABI) or `copyengine::copyengine`
from CMake. In Nsight Systems every group is an NVTX range
`copyengine:<strategy>`.

## With an AI coding agent

Two [Agent Skills](skills/) teach coding agents (Claude Code, Codex, Cursor,
...) to adopt and debug copyengine: `copyengine-adopt` finds copy hotspots,
picks the integration, warms it up and proves same results and the measured
gain; `copyengine-diagnose` explains slow, declined or failing copies from
copyengine's own stats.

```bash
npx skills@1 add https://github.com/itdevwu/copyengine/tree/v0.4.2/skills/copyengine-adopt --global
npx skills@1 add https://github.com/itdevwu/copyengine/tree/v0.4.2/skills/copyengine-diagnose --global
```

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
- Threads: an `Engine` may be shared by threads. Each thread has its own
  scopes, queue and stream, and its own last flush for `freeze()`; plans,
  tuning and the profile are shared. A scope intercepts copies only on the
  thread that opened it. `ce.scope()` and `ce.copy_` use a per-thread default
  engine; share one `Engine` explicitly to share its tuning.
- Buffers of copies that are not waited for (intercepted `non_blocking`
  copies, `allow_async`) are kept alive until the GPU is done with them.
- CUDA Graphs: copies inside `torch.cuda.graph` (scopes, `ce.copy_`,
  `FrozenPlan.run()`) are recorded and replay correctly. Copies torch has no
  equivalent for must run once before capturing.
- `scope(stream=...)`: the scope's work (its copies, and any other op written
  in it) runs on that stream, after what the current stream had queued. The
  host waits for none of it, even for copies that would otherwise block; the
  returned handle (`wait()`, `synchronize()`, `done()`, `event`) orders it
  against the rest. Its buffers stay alive until the copies are done. Inside a
  CUDA graph capture this is a fork, and `h.wait()` the join. Copies
  copyengine leaves to torch (for example a float-to-int cast) keep torch's
  own semantics.
- Index tensors (`ce.lazy(...).index_select`, `ce.gather`, `ce.scatter_`) may
  be host or CUDA tensors. CUDA indices stay on the GPU: the copy kernel reads
  them, so the host never waits and new indices reuse the same plan. An
  out-of-range CUDA index skips that element and raises `ValueError` at the
  next `synchronize()`; duplicate CUDA scatter indices are not detected
  (undefined, as in torch's `index_copy_`).
- Each destination element must be written once per copy: duplicate scatter
  indices raise `UnsafeCopy` in the explicit API (an intercepted `copy_`
  leaves those to torch).
- A source that shares bytes with its destination (judged by storage and
  byte ranges, so views of one buffer count): a lazy copy
  (`ce.copy_(ce.lazy(pool)[d], ce.lazy(pool)[s])`) gathers first and then
  writes, as torch's `pool[d] = pool[s]` does, in two passes where the sides
  may overlap and one where they cannot (disjoint column slices of one pool
  stay one pass; with host indices the check uses this call's values).
  `ce.gather` / `ce.scatter_` and plain views refuse it with `UnsafeCopy`, as
  torch's `index_select(out=)`, `index_copy_` and `copy_` do.

## Environment

| Variable | Effect |
|---|---|
| `CE_DISABLE=1` | turn copyengine's choices off: torch's own copy, or each copy as submitted |
| `CE_EXPLORE=0` | never measure: use the loaded profile; copy patterns it does not cover run natively (or with copyengine's default where torch has no equivalent) and are listed in the profile for the next warmup |
| `CE_STAGING_MB` | device staging budget in MB per stream that stages copies (default 256) |
| `CE_LOG=plan` | print plans to stderr |
| `CE_SYNC=1` | wait for every copy to finish (debugging) |
| `CE_MAX_SMS` | at most this many SMs per GPU for copy kernels (see guarantee 4) |
| `CE_SHIM=0` | skip the compiled `copy_` interception; scopes intercept in Python |
| `CE_TORCH_ALLOCATOR=0` | take copyengine's device and pinned memory from `cuMemAlloc` / `cuMemHostAlloc` instead of torch's allocators |
| `CE_CACHE_DIR` | where interception builds compiled on import are cached (default `~/.cache/copyengine`) |

Licensed under Apache-2.0.
