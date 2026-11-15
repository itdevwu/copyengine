---
name: copyengine-diagnose
description: Diagnose copyengine in a PyTorch program - why copies did not get faster, why it fell back to torch, errors such as Unsupported / UnsafeCopy / NeedsMaterialize, interception shim warnings or build failures, CUDA graph capture errors, async-scope ordering mistakes, suspected wrong results, SM-limit or memory questions. Use when copyengine is already in use and something is slower, failing or unclear.
---

# Diagnose copyengine

Diagnose from copyengine's own evidence before changing code. This Skill
supports copyengine `0.3.x`; check `copyengine.__version__` first.

## First look

```python
import copyengine as ce
st = eng.stats()               # or ce.stats() for the default engine
st["last"]["decisions"]        # per group of the last flush: strategy, why, ops, bytes
st["declined"]                 # intercepted copies left to torch, by reason
st["cuda"]                     # kernels / batch admission, sm_limit per GPU
print(ce._shim.lib is not None, ce._shim.error)          # C++ interception?
print(ce._torch_ops.copy_ is not None, ce._torch_ops.error)  # compiled explicit API?
```

`CE_LOG=plan` prints every new plan to stderr. In Nsight Systems each group is
an NVTX range `copyengine:<strategy>`.

## Not faster

Read `why` in the decisions:

- **"exploring" / "A/B"**: still warming up. Use `eng.warmup(step)` (runs until
  locked) or more representative steps; then `save_profile` / `load_profile`.
- **"locked ... native"**: torch's own copy won the measurement for that
  pattern. Nothing to gain there unless the copies can merge: put more of the
  loop in one scope, or describe the copy in one call (`ce.lazy`, `ce.segments`).
- **"not in the loaded profile"**: running with `CE_EXPLORE=0` and a profile
  that never saw this pattern (other shapes, other version, other GPU).
  Re-run warmup on this machine.
- Many entries in `declined`: those copies are torch's. "dtype pair where
  torch's result differs" (e.g. float to int) and "autograd" are by design;
  "layout, device or dtype copyengine does not handle" lists sparse, complex,
  conjugated or non-CPU/CUDA tensors.
- `stats()["scopes"]` with `"intercept": false`: that scope site measured
  intercepting against not intercepting and torch won, so it runs torch's
  copies directly (no cost, no gain). Typical when the scope holds no `copy_`
  (an `index_copy_`, a vectorized op: use `ce.scatter_` / `ce.lazy` instead) or
  only copies torch already does best. `null`: still deciding.
- `shim: False`: scopes intercept in Python, ~9 us per op, so fewer sites win.
  Fix the shim (below) for loops of many small copies.
- Host overhead dominates tiny copies (a few us each): batch them in one scope,
  or `freeze()` a fixed set and `plan.run()` it.
- A CUDA index tensor stays on the GPU (no sync); a `ValueError` at
  `synchronize()` means one of its indices was out of range.

## Errors

- `Unsupported: ... CUDA graph ... run it once before capturing`: a copy with
  no torch equivalent met a capture before its strategy was set up. Run the
  step (or `eng.warmup`) before `torch.cuda.graph`. Also create
  `eng.copy_stream()` before capturing async scopes.
- `UnsafeCopy`: the explicit API refuses duplicate scatter indices or a source
  overlapping its destination (an intercepted `copy_` leaves those to torch).
- `NeedsMaterialize`: a `reshape` of a lazy view that no strides express;
  materialize with torch first or restructure the view.
- "a scope open on another thread": one engine per thread (module-level
  functions already use a per-thread engine).
- Shim warning at import, or compile errors: the shim is compiled for the
  installed torch on first import (needs `c++`/`g++`; set `CXX`), cached under
  `CE_CACHE_DIR` (default `~/.cache/copyengine`). Delete that cache after a
  compiler or torch change; `CE_SHIM=0` silences it and keeps the Python mode.

## Results look wrong

1. Rerun with `CE_DISABLE=1`. If it is still wrong, the bug is outside
   copyengine (or in how it was wired in).
2. Rerun with `CE_SYNC=1` (waits after every copy). If that fixes it, an
   ordering is missing: a `h.wait()` / `h.synchronize()` after
   `scope(stream=...)`, a `ce.barrier()` before reading memory without torch,
   or a cross-stream dependency.
3. Compare bit for bit on small inputs; NaN payloads may legitimately differ.
   Report a reproducer with `CE_LOG=plan` output and `eng.stats()`.

## Resources

- `CE_STAGING_MB` caps device staging (default 256 MB, allocated on demand).
- `Engine(max_sms=N)` / `CE_MAX_SMS`: `st["cuda"][i]["sm_limit"]` says whether
  the limit is hard (green context, verified) or soft (thread-block cap).
- `scope(bandwidth=...)`: limited groups show strategy `paced` and the
  decision "bandwidth limit"; `st["cuda"][i]["zero_copy"]` says whether
  small rows may use kernels over pinned host memory. A limited copy inside
  a CUDA graph capture must run once before capturing.
