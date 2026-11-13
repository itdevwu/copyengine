# copyengine engineering rules

## Layout

- `include/copyengine/copyengine.h` is the stable C ABI (`ce_` prefix). Any
  incompatible change bumps `CE_ABI_VERSION`. Until the first release, the C
  ABI, Python API and profile format may change freely; from then on they
  follow semver.
- `include/copyengine/*.hpp` + `src/core`: the platform-independent planner
  (namespace `ce`): IR and lowering (`ir`), planning and dependencies (`plan`),
  tuner, engine, probe. It must not include torch or CUDA headers.
- `src/backends/host`: host executor and the conversion table.
- `src/backends/cuda`: driver loader (dlopen + `cuGetProcAddress` at the
  header's `CUDA_VERSION`), probe, executor, and `kernels.cu` (AOT fatbin,
  embedded by `cmake/embed.cmake`).
- `src/core/convert.hpp` is shared verbatim by host code and CUDA kernels;
  host and device conversions must stay bit-identical (no FMA: host files use
  `-ffp-contract=off`, kernels `--fmad=false`).
- `src/capi` implements the C ABI; `src/python` is the nanobind extension
  `copyengine._core`; `python/copyengine` is the Python layer.
- torch integration is layered, and all logic stays in the torch-free core:
  `_torch` on the stable libtorch ABI (descriptors, explicit APIs as custom
  ops; one build for every supported torch), a per-torch-version interception
  shim holding only dispatch-key registration and redispatch, and
  TorchDispatchMode as the fallback when the shim is missing.
- The version lives only in `project(... VERSION ...)` in `CMakeLists.txt`.

## Invariants

- Ops carry absolute element indices (address / itemsize); pure copies are
  normalized to bytes. Dependencies and merging rely on this.
- Every Executor strategy must be correct for any group it is offered;
  `passthrough` (each op as submitted) is always a candidate.
- New platform features are admitted only after a verified test copy.
- Backend tables built per group go in `Group::exec_cache`; hot paths do not
  re-plan.
- No regression: the framework's own copy is the baseline. Dominance-safe
  transforms apply directly; a speculative strategy is used only after it wins
  a paired A/B against the baseline that counts host and GPU time. Ties keep
  the baseline.
- The native baseline is offered only for dtype pairs whose bits match ce's,
  backed by a test.
- Hot paths never synchronize; only warmup exploration and validation may.
- Intercepted `copy_` keeps the caller's `non_blocking` semantics; explicit
  `ce.copy_` returns with host-side copies complete.

## Scope

- Commit code only: no design notes, benchmark results, or machine-specific
  data (hostnames, paths, GPU models, driver versions).
- copyengine is a library; do not add a CLI.

## Verification

- `just test` builds the C++ library, runs ctest, installs the Python package
  and runs pytest (CUDA tests skip without a GPU).
- GPU work runs inside `just docker-shell`.
- `python bench/bench_torch.py` compares against torch on a GPU.

## Commits

- Emoji conventional commits on `master`, for example `✨ feat: add Runs IR`.
  Rebase, no merge commits.
