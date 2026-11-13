# copyengine engineering rules

## Layout

- `include/copyengine/copyengine.h` is the stable C ABI (`ce_` prefix). Any
  incompatible change bumps `CE_ABI_VERSION`.
- `include/copyengine/*.hpp` and `src/core` hold the C++20 planner core
  (namespace `ce`). It is platform-independent and must not depend on torch
  or CUDA headers.
- `src/capi` implements the C ABI on top of the core.
- `src/python` is the nanobind extension `copyengine._core`; `python/copyengine`
  is the pure-Python layer above it.
- The version lives only in `project(... VERSION ...)` in `CMakeLists.txt`.

## Scope

- Commit code only: no design notes, benchmark results, or machine-specific
  data (hostnames, paths, GPU models, driver versions).
- copyengine is a library; do not add a CLI.

## Verification

- `just test` builds the C++ library, runs ctest, installs the Python package
  and runs pytest.
- GPU work runs inside `just docker-shell`.

## Commits

- Emoji conventional commits on `master`, for example `✨ feat: add Runs IR`.
  Rebase, no merge commits.
