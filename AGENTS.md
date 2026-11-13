# copyengine engineering rules

## Layout

- `include/copyengine/copyengine.h` is the stable C ABI (`ce_` prefix). Any
  incompatible change bumps `CE_ABI_VERSION`.
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
- The version lives only in `project(... VERSION ...)` in `CMakeLists.txt`.

## Invariants

- Ops carry absolute element indices (address / itemsize); pure copies are
  normalized to bytes. Dependencies and merging rely on this.
- Every Executor strategy must be correct for any group it is offered;
  `passthrough` (each op as submitted) is always a candidate.
- New platform features are admitted only after a verified test copy.
- Backend tables built per group go in `Group::exec_cache`; hot paths do not
  re-plan.

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
