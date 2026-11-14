# Changelog

copyengine follows [semantic versioning](https://semver.org). From 0.1.0 on,
the Python API, the C ABI (`CE_ABI_VERSION`) and the tuning profile format
change incompatibly only in a new minor (0.x) or major release.

## Unreleased

- FP8 dtypes: `float8_e4m3fn`, `float8_e5m2`, `float8_e8m0fnu`, converted to
  and from every other dtype with torch's exact bits (round to nearest even;
  e4m3fn overflow becomes NaN as in torch, so clamp first to saturate), on
  host and GPU, inside scopes and in the explicit API. Quantization and
  dequantization with a scale run in one pass.
- `float4_e2m1fn_x2` (packed FP4): copied between tensors of that dtype,
  any layout; never converted.
- `Engine(max_sms=N)` / `CE_MAX_SMS`: copy kernels occupy at most N SMs per
  GPU, through a green context where its confinement verifies on start
  (rounded down to the driver's SM granularity), else by launching at most N
  thread blocks. `stats()["cuda"]` reports which. DMA copies are unaffected.
- `scope(stream="copy")` (or a `torch.cuda.Stream`): the scope runs on that
  stream after the current one's queued work, the host never waits, and the
  `Pending` handle it binds (`wait()`, `synchronize()`, `done()`, `event`)
  orders it against other streams; buffers are kept alive until done. Works
  inside CUDA graph capture (fork / join).
- Observability: `stats()["last"]["decisions"]` gives every group's strategy
  and why; `stats()["declined"]` counts intercepted copies left to torch, by
  reason (the same in the C++ shim and the Python interception); each group
  is an NVTX range `copyengine:<strategy>`.
- The no-regression and same-results guarantees now state what holds for
  NaN: NaN stays NaN, its payload is not kept (torch's differs per device).
## 0.1.1

- CUDA Graphs: copies issued while a stream is being captured
  (`torch.cuda.graph`) are recorded and replay correctly, in scopes, through
  the explicit API and from `FrozenPlan.run()`. During a capture copyengine
  does not explore, wait, time or allocate: it uses the decided strategy when
  that only enqueues stream work, else torch's own copy. A copy torch has no
  equivalent for (a `ce.lazy` gather, `ce.segments`) must run once before
  capture; otherwise copyengine raises `Unsupported` instead of breaking the
  capture.

## 0.1.0

First release.

- `ce.scope()`: inside it, unchanged torch `copy_` calls are queued, merged
  and issued together; every other op is ordered after the queued copies it
  touches. A C++ dispatcher shim does the interception (prebuilt for torch
  2.9, 2.10 and 2.11, compiled on import for others); a pure-Python
  interception with the same results is the fallback.
- Explicit API: `ce.copy_` for any strided views with broadcast,
  `ce.lazy(...)` views torch cannot express (gather + permute + slice),
  `ce.gather` / `ce.scatter_`, `ce.segments`, conversions and quantization
  with `ce.Elementwise`, `Engine.freeze()` replay.
- Strategies on CUDA: merged and batched DMA (`cuMemcpyBatchAsync`), 2-D
  copies, vectorized copy and conversion kernels, dense link copies with the
  reordering done on the GPU, pinned bounce buffers for pageable memory. Host
  copies run on worker threads.
- Measured choices: torch's own copy is the baseline; other strategies must
  win a paired A/B, and locked decisions are re-checked online.
- dtypes: float64/32/16, bfloat16, int64/32/16/8, uint8, bool. Host and GPU
  conversions are bit-identical.
- Platforms: Linux x86_64, Python 3.10+, CUDA 12.x and 13.x drivers (loaded at
  runtime), torch 2.9+ for `scope()` interception, 2.10+ for the explicit API
  as torch custom ops (older torch runs it from Python). A stable C ABI and CMake
  package for C and C++ callers.

Known limits: no fp8 dtypes; copies between two GPUs are implemented but have
not been tested on a multi-GPU machine; small explicit copies pay a few
microseconds of host overhead more than torch.
