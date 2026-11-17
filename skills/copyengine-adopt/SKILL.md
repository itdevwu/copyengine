---
name: copyengine-adopt
description: Make tensor copies in a PyTorch codebase faster with copyengine, without changing results. Use to find copy hotspots (loops of copy_, gathers and scatters by block id, host <-> GPU transfers, permute / contiguous / cast before a copy, quantize or dequantize on the way), pick the right integration (scope, explicit API, frozen replay, async copy stream, SM limit), warm it up, prove the results are unchanged and measure the gain. Also for KV-cache offload / load, paged caches, LoRA or weight reload and per-step input refresh before CUDA graph replay.
---

# Adopt copyengine

copyengine plans and merges copies at run time and keeps a strategy only if it
measured faster than torch's own copy. Your job: find copies worth handing
over, hand them over with the smallest change, and prove both claims (same
bits, faster) on the user's machine. Never claim a speedup you did not measure.

## Check the installation

This Skill supports copyengine `0.4.x`:

```bash
python -c "import copyengine as ce, torch; print(ce.__version__, torch.__version__)"
python -m pip install 'copyengine~=0.4.0'   # if missing or another series
python -c "import copyengine as ce; print('shim:', ce._shim.lib is not None, ce._shim.error); print('ops:', ce._torch_ops.copy_ is not None, ce._torch_ops.error)"
```

`shim: False` means scopes intercept in Python: same results, slower. It
usually needs a C++ compiler for the installed torch; read the
copyengine-diagnose Skill if the user wants it fixed. `ops: False` on torch
2.9 is expected (2.10+ compiles the explicit API). Linux x86_64 and aarch64.

## Find what to hand over

Profile one representative step after warmup, not startup. Prefer evidence
over reading code: `torch.profiler` (count `aten::copy_`, `Memcpy HtoD/DtoH/DtoD`
and small elementwise kernels per step), or the xprobe Skill if installed (its
GPU memcpy / kernel inventory). Good candidates, in order:

1. Many small copies per step: loops over layers x blocks, per-request rows.
2. Gather or scatter by an index list (paged KV, offload by block id).
3. Layout or dtype changes around a copy: `permute(...).contiguous()`,
   `.to(dtype)` then copy, quantize with a scale, dequantize.
4. Host <-> GPU transfers split into many pieces, or from pageable memory.

Skip: one large contiguous copy (already at bandwidth), copies fused into
compute kernels, anything needing autograd through the copy.

## copyengine or torch.compile

They do different jobs and combine: compile the model, hand the data
movement around it to copyengine. copyengine leaves compiled graphs alone
(importing it or running a compiled model inside `ce.scope()` changes neither
results, graphs nor speed).

| Work | Better with |
|---|---|
| Elementwise math or quantization on one GPU, fused into the model, static shapes | torch.compile (shape-specialized kernels) |
| Host <-> GPU transfers, paged offload / load, gathers and scatters whose indices change every step | copyengine (no recompiles; DMA engines; overlap) |
| Hundreds of small tensors per step (one per layer and kind) | copyengine (`ce.copy_many`, one launch) |
| Background copies that must not take SMs or bandwidth from compute | copyengine (`max_sms`, `bandwidth=`) |
| Paths where a compile (seconds, again per new shape) is not affordable | copyengine |

Inside a `torch.compile`d function, `ce.copy_`, `ce.copy_many`,
`ce.lazy(...).index_select(...)` copies (GPU indices), `ce.gather`,
`ce.scatter_`, `ce.quantize` and `ce.dequantize` are one graph node each
(no graph break; CUDA graphs work); other forms (host index lists, a lazy
`flip`) break the graph and run eagerly with the same results. Scopes stay
outside compiled regions; there, write einops patterns on lazy views with
`ce.rearrange` / `ce.repeat` (same signature as einops). There each call is measured once per layout
against torch.compile's own code for it and keeps whichever is faster
(`ce.compiled_stats()["baselines"]` says which and why), so adding ce
inside a compiled model never makes it slower than compiling it alone. When unsure, measure both on the real step;
`bench/bench_compile.py` in the repository compares the common patterns.

## Pick the integration

Read [patterns](references/patterns.md) for code. In short:

| Situation | Use |
|---|---|
| Loop of `copy_` calls, code must stay as is | `with ce.scope():` around the loop |
| Gather / permute / slice / cast torch can't express as one view | `ce.copy_(out, ce.lazy(src).index_select(...).permute(...))` |
| Quantize / dequantize while copying, one scale | `ce.copy_(dst, src, fn=ce.Elementwise(scale=..., clamp=(lo, hi)))` |
| Quantize per group (FP8 / INT8 / INT4 / MXFP8 / NVFP4), scales computed or given | `ce.quantize(x, q, scale, fmt=..., group=...)`, `ce.dequantize(q, scale, out, ...)`; `x_index=` / `index=` gather pages and write cache blocks |
| Ragged / CSR segments | `ce.segments(src, dst, src_off, dst_off, length)` |
| One copy per layer from Python lists of the same tensors every step | `ce.copy_many(dsts, srcs)` instead of a `copy_` loop |
| The same copies every step, same buffers | `plan = eng.freeze()` once, then `plan.run()` (also inside `torch.cuda.graph`) |
| The same step every iteration, new index tensors or host index arrays (paged pools) | `step = eng.plan(fn)` once, then `step(*args)`; pass whole pools and the indices, not views picked per step |
| Copies should overlap compute | `with ce.scope(stream="copy") as h:` ... `h.wait()` before use |
| Copies must not take SMs from compute | `ce.Engine(max_sms=N)` |
| Background offload must leave PCIe / memory bandwidth to other work | `with ce.scope(stream="copy", bandwidth=8e9) as h:` |

An `Engine` may be shared by threads (each has its own scopes and queue;
tuning is shared); the module-level functions use one engine per thread.
Index tensors may stay on the GPU (no sync): `ce.gather`, `ce.scatter_` and
`ce.lazy(...).index_select` read them there. Host indices (lists, numpy
arrays) are as good in a step plan; do not turn them into CUDA tensors per
step with `torch.tensor(ids, device="cuda")`, which waits for that copy.

Block ids that change every step: index the pools, do not slice them.
`ce.copy_many([pool[i] for i in ids], ...)` makes new views each step: each
call is planned anew, and `copy_many` keeps torch's `_foreach_copy_` where
that is faster (`stats()["copy_many"]`), so it gains little. Pass the pools
and the ids to a step plan instead (planned once, replayed with new ids):
`ce.lazy(pool)[ids]` (see [patterns](references/patterns.md)). Host pools may
be pinned or pageable; pinned is faster. Layer slices of one block-major pool
(`pool[:, a:b]`) are fine as destinations: copies into disjoint slices run
together.

## Warm up, then serve without exploring

copyengine explores strategies on the first runs of each copy pattern. Warm up
with representative shapes, save the decisions, load them in serving:

```python
n = eng.warmup(step)   # step(engine) issues the copies; runs until every pattern has decided
eng.save_profile("ce_profile.json")
# serving process:
eng.load_profile(open("ce_profile.json").read())   # with CE_EXPLORE=0 to never measure
```

A profile is valid only on the same hardware and copyengine version.

Exploring includes one run of torch's own copy per new pattern (the baseline
copyengine must beat): a few milliseconds once, then never again for that
pattern. Patterns are bucketed by sizes in powers of two, so warm up with the
block counts serving will see (n = 1, 2, 4, 8, ...) and check that every
decision's `why` reads "locked" before taking traffic.

## Prove it

1. **Same results.** Run the original and the adopted code on identical inputs
   and compare bit for bit (`torch.equal` on `.view(torch.int16/int32)` etc.).
   NaN payloads may differ (torch's own differ between CPU and CUDA); compare
   NaN positions instead. Also run once with `CE_DISABLE=1` to confirm the
   integration itself is correct.
2. **Faster.** After warmup, time many steps with CUDA events (GPU time) and
   `time.perf_counter` around a synchronized step (wall time). Report both,
   with the number of runs and the spread; laptop GPUs change clocks, so
   alternate original / adopted runs.
3. **What it chose.** `eng.stats()["last"]["decisions"]` (strategy and why per
   group) and `eng.stats()["declined"]` (copies left to torch, by reason).
   "locked ... native" means torch's copy was already best there: that copy
   gains nothing; say so instead of forcing it.

## Rules that keep results right

- Inside a scope, other torch ops are ordered after the queued copies they
  touch; memory read without torch (`.numpy()`, raw pointers) needs
  `ce.barrier()` first.
- After `scope(stream=...)`, call `h.wait()` (GPU) or `h.synchronize()`
  (host) before using the destination.
- A copy with no torch equivalent (lazy gather, segments) must run once before
  a CUDA graph capture.
- `CE_DISABLE=1` turns copyengine's choices off for an instant rollback.

## SM limits for background work

`ce.Engine(max_sms=N)` bounds the SMs a background engine's kernels take;
DMA copies take none but share memory bandwidth (`bandwidth=` paces them).
The right N depends on the foreground: compute-bound work (prefill) tolerates
more than bandwidth-bound work (decode), which a quantizing background slows
through memory traffic even on few SMs. Measure the foreground with the
background running at a few settings (e.g. N = 16, 8, 4) and pick the
largest N that keeps it within budget; the default is no limit.
