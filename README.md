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
  the way (FP8 or int8 with a scale), swapping LoRA weights in and out.
- **Training**: offloading optimizer state and activations, dataloader H2D,
  checkpoint save / load and resharding (pure reordering).
- **Model code**: `x.permute(...).contiguous().cuda()`, copying after
  `index_select`, permute-then-cast: write it as one `ce.copy_` into the
  destination (with `ce.lazy` for gathers), and it happens in one pass, with
  the reordering done on the GPU.

copyengine moves and converts data. Computation such as reductions is out of
scope: it runs as usual, ordered correctly around the copies.

## What it guarantees

1. **Not slower than the original.** Wherever torch has the same copy, torch's
   own `copy_` is always a candidate. Anything else is measured against it,
   including the time copyengine itself spends planning and submitting, and is
   kept only if it is clearly faster. Decisions keep being re-checked while
   your program runs. The explicit API has the same protection where torch can
   express the copy: `ce.lazy` views, `ce.gather`, `ce.scatter_`,
   `ce.segments` and `ce.indices` are measured against the same copy written
   with torch ops (indexing, `index_copy_`, slices) from the same host
   indices. Conversions with `fn=` and views torch has no form for (a reversed
   slice) have no torch equivalent: there copyengine picks the fastest of its
   own strategies. Either way it cannot beat torch code that does less work,
   for example indices already on the GPU, and each explicit call costs a few
   microseconds of host time: for a few small copies per step, `freeze()` or
   a CUDA graph is the fast path.
2. **The same results.** copyengine only takes a copy when it can produce
   exactly torch's result; otherwise torch runs it. Host and GPU conversions
   are bit-identical. (NaN stays NaN, but its sign and payload bits are not
   kept: torch's own differ between its CPU and CUDA conversions.)
3. **Easy to turn off.** `CE_DISABLE=1` turns copyengine's choices off: every
   copy runs through torch's own `copy_`, or exactly as submitted where torch
   has no equivalent.
4. **Bounded resources.** With torch installed, copyengine's own device memory
   (staging, copy tables) comes from torch's caching allocator, so it shows in
   `torch.cuda.memory_allocated()` and obeys torch's limits. Device staging
   has a budget (`CE_STAGING_MB`),
   and `Engine(max_sms=N)` (or `CE_MAX_SMS`) keeps copy kernels on at most N
   SMs per GPU, leaving the rest to your compute. Where the driver supports
   green contexts the limit is hard (checked on start); otherwise copyengine
   launches at most N thread blocks. DMA copies use the copy engines and no
   SMs; memory bandwidth is still shared.

The first runs of each new copy pattern are a warmup: copyengine measures its
options, then locks the fastest. Save the result with `eng.save_profile(path)`
to skip the warmup next time on the same machine.

## Install

```bash
pip install copyengine
```

The wheels include the GPU kernels and the torch integration; they use the
torch you already have. To build from source instead:

```bash
pip install .                      # needs CMake >= 3.26 and a C++20 compiler
                                   # GPU support: CUDA toolkit headers + nvcc at build time
pip install --no-build-isolation . # same, with torch already installed: also builds the torch integration
```

dtypes: float64/32/16, bfloat16, float8 (e4m3fn, e5m2, e8m0fnu), int64/32/16/8,
uint8 and bool convert into each other; packed FP4 (`float4_e2m1fn_x2`) is
copied unconverted.

Requirements: Linux x86_64, Python ≥ 3.10. GPU support runs on CUDA 12.x and
13.x drivers; the driver is loaded at runtime, nothing links against `libcuda`
or `cudart`.

The torch integration is built against the torch you have installed. If you
later switch torch versions, the part that intercepts `copy_` is recompiled
once on import (needs a C++ compiler, takes seconds, cached). If that is not
possible, scopes fall back to a pure-Python interception with the same
results, only slower, and a warning says so.

**Use torch 2.10 or newer for the best performance.** Every torch from 2.9 on
gives the same results, but from 2.10 the explicit API (`ce.copy_`,
`ce.lazy`, ...) runs as a compiled torch op; on 2.9 it goes through Python and
costs a few more microseconds per call, which matters for many small copies.
`scope()` interception is compiled on every supported version.

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

# 5. Overlap copies with compute: run them on a copy stream, the host never waits.
with ce.scope(stream="copy") as h:      # or stream=<a torch.cuda.Stream>
    for l in range(num_layers):
        next_kv[l].copy_(host_kv[l, ids])
...                                     # compute keeps running on the current stream
h.wait()                                # the current stream waits for the copies (GPU side)

# 6. Replay a planned copy without planning again.
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

To see what copyengine did: `eng.stats()["last"]["decisions"]` lists each
group of the last flush with its strategy and why (still exploring, the A/B
result it locked on, an online re-check, a CUDA graph capture), and
`stats()["declined"]` counts the intercepted copies it left to torch, by
reason. In Nsight Systems every group is an NVTX range named
`copyengine:<strategy>`.

To compare against torch on your GPU: `python bench/bench_torch.py`.

## With an AI coding agent

Two [Agent Skills](skills/) teach coding agents (Claude Code, Codex, Cursor,
...) to adopt and debug copyengine: `copyengine-adopt` finds copy hotspots,
picks the integration, warms it up and proves same results and the measured
gain; `copyengine-diagnose` explains slow, declined or failing copies from
copyengine's own stats.

```bash
npx skills@1 add https://github.com/itdevwu/copyengine/tree/v0.2.0/skills/copyengine-adopt --global
npx skills@1 add https://github.com/itdevwu/copyengine/tree/v0.2.0/skills/copyengine-diagnose --global
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
- Threads: `ce.scope()` and `ce.copy_` use a per-thread default engine. An
  `Engine` you share between threads works while none of them has a scope
  open on it; a copy or scope from another thread during one raises an error.
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
| `CE_MAX_SMS` | at most this many SMs per GPU for copy kernels (see guarantee 4) |
| `CE_SHIM=0` | skip the compiled `copy_` interception; scopes intercept in Python |
| `CE_TORCH_ALLOCATOR=0` | take copyengine's device memory from `cuMemAlloc` instead of torch's allocator |
| `CE_CACHE_DIR` | where interception builds compiled on import are cached (default `~/.cache/copyengine`) |

Licensed under Apache-2.0.
