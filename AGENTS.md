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
- torch integration is layered, and all logic stays in the torch-free core,
  reached through the `FrontendApi` table that `_core` exports:
  `src/torch/ops.cpp` on the stable libtorch ABI (descriptors, explicit APIs as
  custom ops; one build for every torch >= 2.10), and `src/torch/shim.cpp`, the
  scope interception, holding only dispatch-key registration and redispatch.
  The shim uses the DeferredInit key, refuses to load if something else holds
  it, and is admitted only after a verified copy; it is prebuilt for the
  build-time torch and otherwise compiled on import, cached per torch build.
  Its fallback is `python/copyengine/_mode.py` (TorchDispatchMode) with
  identical semantics: behavior may differ only in speed, never in results,
  and `tests/python/test_scope.py` runs under both.
- The version lives only in `project(... VERSION ...)` in `CMakeLists.txt`.
- `skills/` holds the Agent Skills users install by release tag. Each states
  the copyengine minor series it supports; update them with any API or
  behavior change they describe, and keep every snippet in them runnable.

## Invariants

- Ops carry absolute element indices (address / itemsize); pure copies are
  normalized to bytes. Dependencies and merging rely on this.
- Every Executor strategy must be correct for any group it is offered.
  `passthrough` (each op as submitted) is always correct, but it is measured
  only when its call count is bounded: it is never a multi-second baseline.
- New platform features are admitted only after a verified test copy.
- Backend tables built per group go in `Group::exec_cache`; hot paths do not
  re-plan.
- No regression: the framework's own copy is the baseline. Dominance-safe
  transforms apply directly; a speculative strategy is used only after it wins
  a paired A/B against the baseline that counts host and GPU time. Ties keep
  the baseline. Where no native copy exists there is no original to protect:
  candidates are explored alike and the fastest locks.
- Measuring must not stall a run: hopeless candidates are dropped after one
  sample, and online re-checks only run finalists within 2x of the locked one.
- A group touches several allocations; no copy call, 2-D block or staged
  interval may cross an allocation boundary (the driver rejects it).
- The native baseline is offered only for dtype pairs whose bits match ce's,
  backed by a test. "Match" means every non-NaN value bit for bit and NaN
  for NaN: torch's NaN payloads differ between its CPU and CUDA conversions.
- Hot paths never synchronize; only warmup exploration and validation may.
- copyengine's own long-lived device memory goes through `DevMem`
  (`dev_alloc` / `dev_free` in the CUDA executor), which uses the framework's
  allocator when one is registered. Before such memory goes back, its
  allocation stream is ordered after every stream copyengine used on that
  device; a test checks the ordering, not just the data.
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

## Branches, commits and releases

- `master` is the trunk and only receives pull requests; never push to it
  directly. Do each piece of work on its own branch named
  `<type>/v<next version>-<topic>`, for example `feat/v0.2.0-torch-allocator`
  or `fix/v0.1.2-capture`, open a pull request, and rebase it into `master`
  once CI passes (the repository allows only rebase merges).
- Emoji conventional commits, for example `✨ feat: add Runs IR`.
- CI has no GPU. A pull request that touches GPU paths (kernels, CUDA graphs,
  async scopes, SM limits) states which GPU tests ran locally, on what GPU,
  and their result.
- A release is a pull request whose last commit is
  `🔖 chore: prepare version X.Y.Z`: the version in `CMakeLists.txt`, the
  CHANGELOG heading, and every tag the README and `skills/` refer to. After it
  is merged, tag `vX.Y.Z` on `master`; the tag builds and publishes the wheels.
- Release from `master`. Only when an older minor needs a fix while `master`
  holds unreleased incompatible changes, branch `release/0.X` from that
  minor's last tag, release the fix there, and port it to `master` through a
  pull request.
