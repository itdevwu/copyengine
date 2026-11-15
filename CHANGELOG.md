# Changelog

copyengine follows [semantic versioning](https://semver.org). From 0.1.0 on,
the Python API, the C ABI (`CE_ABI_VERSION`) and the tuning profile format
change incompatibly only in a new minor (0.x) or major release.

## Unreleased

- Quantization while copying (`ce.quantize`, `ce.dequantize`): FP8 (E4M3,
  E5M2), INT8, INT4 (symmetric) and UINT4 (zero points) per group of values,
  MXFP8 with E8M0 scales, NVFP4 with E4M3 scales and a global scale; dynamic
  or static scales; gathers and scatters by index (host or GPU indices);
  tensors on one GPU and in pinned host memory (quantize and move in one
  pass); inside scopes and step plans. Each format's result is defined by
  its torch recipe in `copyengine/_quant.py`, with the same bits on the host
  and every GPU. Copies keep matching torch itself bit for bit; computation
  matches its recipe (torch's own arithmetic differs across devices).

## 0.3.0

- Step plans take host index arguments (numpy arrays, lists of ints, CPU
  integer tensors) as parameters too: new values each call replay the same
  plan. Two strategies compete for them in the tuner: `lowered` (DMA
  descriptors built from the values at each run) and `fused` (values sent to
  the GPU, host memory read or written directly by the statement kernel).
- Fix: an engine whose copies all ran as torch's own never waited for them in
  `synchronize()` (its devices were never set up), so an explicit copy into
  host memory could return before the data arrived.
- `ce.copy_many(dsts, srcs)`: a list of copies (one per layer, say) in one
  call into C++, planned together, instead of a Python loop of `copy_`.
- Fixes: building with CUDA 13 headers (`cuMemcpyBatchAsync` lost a
  parameter); the torch baselines of explicit copies between host and GPU
  (`ce.lazy(...).index_select` destinations, `scatter_`, `indices`) moved the
  data to the wrong device; the baseline of `ce.segments` with many long
  segments built element-index tables as large as the data (now slices).
- Copies on one stream no longer wait for the same engine's copies on another
  stream: with `max_sms`, each caller stream gets its own green-context lane,
  and device staging is one buffer per caller stream (up to 8 of each, then
  shared). Before, a copy on the compute stream could queue behind a large
  background copy.
- Step plans: `ce.plan(step)` / `eng.plan(step)` trace a step of copies per
  argument layout and replay it without running its Python or planning; CUDA
  int64 index tensors are read anew each call, other buffers are guarded by
  identity. A replay is kept only where it beats running the step plainly.
- Copies with device indices depend on each other only through their
  reachable extents, so gathers into disjoint slices run in one launch.
- The plan, statement and safety caches evict the least recently used entry
  instead of clearing when full. `stats()` reports planning time on cache
  misses (`plan_us`, total and last flush).
- copyengine's pinned host buffers (bounce and upload buffers) come from
  torch's pinned allocator when torch provides it, so they count in its host
  memory statistics; `stats()["cuda"][i]["host_allocator"]` reports it.
- CUDA index tensors stay on the GPU in `ce.gather`, `ce.scatter_` and
  `ce.lazy(...).index_select`: the statement kernel reads them (no device
  synchronization), and new indices reuse the same plan. Out-of-range indices
  are skipped and raise `ValueError` at the next `synchronize()`.
- Bandwidth limits: `ce.scope(bandwidth=bytes_per_s)` and
  `Engine(bandwidth=...)` run copies that touch a GPU as a paced background
  transfer (a GPU pacing kernel between chunks, per GPU and direction; no
  host waits, recorded into CUDA graphs). Small rows use zero-copy kernels
  over pinned host memory with streaming accesses. Results do not change.

## 0.2.2

- A scope no longer costs time where copyengine does not help: each scope
  call site compares intercepting with not intercepting at all (end to end,
  host and GPU), after the engine has settled there, and stops intercepting
  unless that clearly wins. Before, with the compiled shim, a scope always
  intercepted, so a scope around ops copyengine does not take, or around
  copies torch already did best, ran slower than torch. `stats()["scopes"]`
  shows each site's decision.

## 0.2.1

- Wheels for Linux aarch64 (manylinux_2_28; server ARM such as Grace: GH200,
  GB200, GB10), built and tested in CI next to x86_64.
- Fix: destroying an engine with `max_sms` whose device staging came from
  torch's allocator could crash (a stream was destroyed before the memory
  ordered after it was freed).

## 0.2.0

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
- `Engine.warmup(fn)` runs until every copy pattern `fn` hits has locked its
  decision (it ran 4 times regardless) and returns the number of runs.
- Agent Skills `copyengine-adopt` and `copyengine-diagnose` (`skills/`).
- Torch baselines for the explicit API: `ce.lazy` views (index_select,
  permute, slice, select, flip, unsqueeze, expand, reshape) as source, views
  or a single index_select as destination, and `ce.gather`, `ce.scatter_`,
  `ce.segments`, `ce.indices` now carry the same copy written with torch ops,
  so the A/B protects them too and `CE_DISABLE` runs them natively. Copies
  with a torch baseline record into CUDA graphs without a prior run.
- A/B fixes: a run that set a strategy up (resident tables) is no longer a
  sample, and submit cost counts on both sides; both had biased decisions
  toward torch's copy.
- copyengine's own device memory (staging, resident copy tables, upload
  buffers) comes from torch's caching allocator when torch 2.10+ is
  installed: it counts in torch's memory statistics, limits and snapshots, and
  freeing it never synchronizes the device. Before memory goes back to torch,
  the stream it belongs to is ordered after every stream copyengine used.
  `CE_TORCH_ALLOCATOR=0` restores `cuMemAlloc`. Engines also no longer leak
  their copy tables when destroyed.
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
