# Changelog

copyengine follows [semantic versioning](https://semver.org). From 0.1.0 on,
the Python API, the C ABI (`CE_ABI_VERSION`) and the tuning profile format
change incompatibly only in a new minor (0.x) or major release.

## Unreleased

- Many small rows between pinned host memory (mapped at its address) and a
  GPU can run as one kernel reading or writing them over the link itself
  (zero-copy), measured beside DMA per row, batched DMA and staging (token
  rows of a page-first host KV pool, say).
- A GPU index table is a read of the copy that uses it: a copy queued in the
  same scope that writes the table runs first (before: the kernel could read
  the table before it was written), and a copy whose destination overlaps
  its own index table is refused (UnsafeCopy). `disable=True` engines run
  copies with run-time indices and no torch equivalent themselves (before:
  skipped).
- GPU-index gathers and scatters between two GPUs are no longer refused as
  "host index tables between two GPUs"; their torch baselines gather on the
  source GPU and scatter on the destination's; a gather into host memory runs
  on the source GPU's current stream.
- A lazy copy with host index values on one side and a GPU index table on
  the other runs (before: refused); the host values are snapshotted at the
  call; duplicate destination rows are refused (UnsafeCopy).
- A step plan's first run executes the step once (before: a copy that could
  not take a host index parameter made it run the step a second time, after
  the copies queued before it had already run: wrong results for steps that
  are not idempotent, fuzzer seed 17). Overlapping block copies within one
  pool run up to 36 copies per launch of the three kernels.
- From an independent review of this release: overlapping block copies under
  a soft SM limit ran only the first copies of a batch (the grid's second
  dimension was capped; the kernels now stride over the batch); a CUDA graph
  that captured them could replay into buffers a later, larger eager batch
  had freed, and two graphs captured on one stream shared them (a captured
  graph's buffers are now its own); a step plan replay
  whose index tensor lies in memory the step writes runs the step plainly
  (its frozen order did not cover that); quantization writing its own GPU
  index table and copies into broadcast (stride-0) destinations are refused
  (UnsafeCopy); a scoped copy between two GPUs runs on the source GPU's
  stream current when it was made; a CPU index tensor rewritten after a lazy
  view was built no longer changes the view; `gather` / `scatter_` in a step
  plan take index values where a host index parameter cannot be read.
- `bench/bench_serving.py` draws new block ids every step, as serving does,
  and adds a step-plan column (`ce_plan`) and per-case profiling.

- `bench/bench_workloads.py`: copyengine on its target workloads, one suite
  per kind (rooflines, segment sweeps, paged KV, reblock, resharding, SM
  caps, interference, quantization with paged gathers and across PCIe, block
  FP8, every quantization format, small-copy call overhead, two GPUs), one
  JSON report that compares release against release (`--compare`); the kit
  runs it with `--bench`.

- Block copies within one pool by GPU indices that may overlap
  (`ce.copy_(ce.lazy(pool)[d], ce.lazy(pool)[s])`, rows of one layout) run as
  three kernels: mark the destination rows, stage only the source rows that
  are also destination rows, scatter; before, two full passes through a
  temporary. Fuzzed by a `pool` mode (explicit, frozen and step-plan runs).
- No host wait on the hot path: tables made resident for a new plan are
  written on the caller's stream or uploaded on copyengine's side stream with
  an event their users wait for (before, each new plan waited; with a copy
  at new addresses every call, every call did); copyengine's buffers
  allocated on the default stream (handle 0) no longer wait for the device.
  A queue flushed because work arrives on another stream now orders that
  stream after the flushed copies too (it waited only for earlier work).
- Inside torch.compile regions, `dst = fn(src)` copies are measured against
  Inductor's code for the same steps (taken only where its bits equal the
  catalog's on edge values), and copies with a host side are measured with
  pinned and pageable memory (the trace cannot tell them apart): copyengine is
  taken only if it wins with both.
- `stats()["locks"]`: buckets per locked strategy, those still exploring, and
  the online re-checks' flips and reopens. A lazy reshape that merges
  dimensions whose strides nest keeps them strided where another dimension
  has an index table (before: a table of every offset). Quantizing a
  one-dimensional `x` works inside torch.compile regions as it does eagerly.
- torch's own copy run by copyengine inside an intercepting scope is no
  longer intercepted again (it was queued instead of done: a later copy could
  run first; fuzzer seed 1963). The CUDA driver's entry points are fetched at
  the version of the prototype they are called with (cuStreamGetCtx's newer
  version takes three arguments: copies between two GPUs crashed).

- Copies between two GPUs are ordered as torch's: on the source GPU's
  current stream, after both GPUs' queued work and before their later work
  (before: only the destination's stream, and a default stream could name
  the wrong GPU). Each ordered pair of GPUs is admitted once with verified
  test copies (`stats()["cuda"][i]["peers"]`); where it passes, conversions,
  many-fragment copies and GPU-index gathers / scatters between two GPUs run
  as one kernel on the source GPU (before: refused or one DMA per fragment),
  and batched DMA is offered. The kit runs the peer tests on machines with
  two GPUs or more.

- Threads may share an `Engine`: each thread has its own session (scopes,
  queue, stream, kept buffers, last flush for `freeze()`), while plans,
  tuning and the profile are shared (before: a scope open on one thread made
  other threads' calls raise). The engine lock is taken with the GIL
  released. Scope and `copy_many` site decisions are saved in profiles and
  loaded with them; with `explore=False` sites are not sampled.

- Lazy copies whose source may share bytes with their destination (block
  copies within one pool, `ce.copy_(ce.lazy(pool)[d], ce.lazy(pool)[s])`)
  give torch's `pool[d] = pool[s]` result: two passes through a temporary
  where the sides may overlap, one where they cannot. Overlap is judged by
  storage and byte ranges with stride-preserving footprints (column slices of
  one pool stay one pass); host indices are checked with their values, in
  step plans before every replay. GPU-index gathers and scatters that may
  overlap are refused, as torch refuses them (before: a race).

- torch's own copy that converts, gathers or copies pageable memory on the CPU
  now waits for GPU work on that host memory (a stale read about one run in
  twenty, found by the fuzzer); a queue flushed because work arrives on another
  stream orders the two streams; CPU baselines no longer read indices still
  on their way from the GPU.

- Frozen plans keep the index tensors they read (a temporary index tensor's
  memory could be reused before a replay); step plans notice a tensor
  argument that now holds other memory (`.data =`, `set_`, `resize_`); a step
  plan swapping through a temporary (`tmp = a; a = b; b = tmp`) replays
  right; replays run on the stream of their tensors' GPU; plan keys cover the
  destination's allocation.

- README: quick start and a table of situations first.

- Host memory registered with `cudaHostRegister` (pinned, but possibly at
  another device address, as under WSL2) is no longer handed to kernels or
  batched DMA at its host address (an illegal-address fault): copyengine asks
  the driver whether host memory is pageable, pinned or mapped for every
  frontend, copies registered memory with plain DMA or staging, and refuses
  GPU-index copies that only a kernel could do there.

- Fixed a misaligned-address fault (which poisoned the process's CUDA
  context) in the statement kernel for destinations whose rows are not
  aligned like their base (16-byte rows 24 bytes apart, say).

- Index tables are checked against the rows they select in every frontend
  (the compiled quantize op could read past a shorter one); copies with
  run-time indices no longer share a plan group with segments or index-list
  copies (the flush failed); batched DMA is admitted per direction; a 0-d host
  index works in step plans.

- No result depends on which side of a measurement won: `copy_many` and
  compiled regions offer torch's copy as the baseline only for dtype pairs
  whose bits equal copyengine's, and compiled baselines are checked on
  samples that include NaN, infinities, saturation and rounding edges.

- An engine destroyed while its last copies failed says so on stderr
  instead of discarding the error; a CUDA failure during cleanup no longer
  terminates the process. The explicit indexed-copy op checks `dim`.

- Dependencies between copies use each op's exact footprint on every tensor
  it touches (an index the host does not know spanning its bound) instead of
  its hull: copies and quantizations into disjoint column slices of one pool
  (a block-major host pool `[blocks, layer bytes]`, say) run together instead
  of one after another; overlapping ones stay ordered.

- Copies that keep arriving at new addresses (views of a pool picked per
  call, `copy_many([pool[i] for i in ids], ...)`) settle: their plans are
  tuned in buckets of their own with each strategy's setup counted (before,
  setup runs were no samples, so such copies explored and synchronized
  forever; `stats()["churn_flushes"]` counts them). `copy_many` is measured
  against `torch._foreach_copy_` per call layout and keeps the faster
  (`stats()["copy_many"]`); `freeze()` after a call torch ran refuses
  instead of freezing an older flush. A step plan over the pool and the
  indices plans such copies once.

- Step plans replay copies with host indices into and out of pageable host
  memory (before: refused), and accept tensors the step names instead of
  taking them as arguments (closure variables, globals, attributes of a bound
  method's object), checked to be the same views before each replay.

- Lazy views index like tensors: `ce.lazy(t)[:, a:b]`, integers, `...`,
  `None`, one index array; and `narrow`.

- Quantization kernels launch as many warps as have work: a warp handles
  several small groups (MX, NVFP4, the 128-value formats), and the grid used
  to count one warp per group, most of them idle.

- The structures the CUDA executor passes to the kernels are defined once,
  for both compilers, with their sizes and offsets checked; the kernels are
  admitted only if their layout matches the executor's (a `CUtensorMap`'s
  alignment differs between CUDA headers).

- `ce.rearrange` / `ce.repeat`: einops' patterns on lazy views, also inside
  torch.compile regions, where einops' own functions may be graph nodes a
  lazy view cannot enter.

- Inside torch.compile regions copyengine is measured against what
  torch.compile makes of the same operation: per call kind and layout, the
  torch equivalent compiled by Inductor (admitted only with the same bits)
  and copyengine are timed when first traced, and the graph gets
  copyengine's op only where it is faster. `ce.compiled_stats()` reports the
  engine compiled regions use and these decisions; `CE_COMPILE_BASELINE=0`
  skips the measurement.

- Step plans replay steps that stage copies through tensors they allocate
  and drop: copies into and out of such a temporary compose into direct
  copies, and it is not written (before, the step ran plainly).

- einops on lazy views: `einops.rearrange(ce.lazy(x), ...)` and
  `einops.repeat` compose into the view (no materialization), so a chain of
  rearrangements, gathers, conversions and the write into a destination is
  one planned copy; also inside torch.compile regions. Lazy reshapes no torch
  view expresses (a merge after a transpose or a gather) compose too.

- `tma2d` (sm90 and later): pure device copies whose rows form 2-D blocks
  (narrow rows under a permute, say) move as TMA tensor-map boxes, many rows
  per operation; admitted by a verified test copy and chosen by measurement
  (`stats()["cuda"][i]["tma2d"]`).

- Block quantization: `ce.quantize(w, q, scale, fmt="fp8_e4m3", block=(128, 128))`
  (FP8 E4M3 / E5M2, INT8) uses one float32 scale per block of the last two
  dimensions, as DeepSeek-V3 stores weights; the same formula as the group
  formats over a box's values, recipes in `copyengine/_quant.py`. On the host
  and the GPU, with indices, pinned host memory, step plans and torch.compile.

- A new engine's first copy on a GPU no longer loads the kernels and runs the
  admission test copies again: that happens once per device and process, and
  engines share the module (each engine used to load, and keep, its own).

- The explicit API inside torch.compile regions is one graph node per call
  (`copy_`, `copy_many`, lazy gathers and scatters with GPU indices,
  `gather`, `scatter_`, `quantize`, `dequantize`), also under CUDA graphs;
  forms no op takes break the graph explicitly and run eagerly. Module-level
  calls there use one process-wide engine.

- Explicit copies with GPU indices may have their other side in pinned host
  memory (paged offload by GPU block ids: `ce.copy_(host, ce.lazy(kv).index_select(0, gpu_ids))`,
  and the load back): the statement kernel reads the indices and reaches the
  host side directly; the indices never leave the GPU.

## 0.4.2

- Fix: memory copyengine took from torch's allocator (copy tables, staging,
  upload buffers) was first written on copyengine's own stream without
  waiting for torch's stream, where a freed tensor's kernels could still be
  writing the same block: the table could be overwritten and its copies read
  garbage (wrong data, or an illegal memory access), mostly while exploring.
  The stream that first uses such memory now waits for its allocation stream.

## 0.4.1

- Fix: torch 2.13 converts float -> float8_e4m3fn beyond the format
  (|x| > 464, infinities) to +-448 instead of NaN, on the CPU and the GPU.
  copyengine now follows the installed torch (detected at import), so copies
  into float8_e4m3fn keep matching it bit for bit; with 0.4.0 they returned
  NaN there under torch 2.13. Quantization is unaffected (its recipes clamp
  first).
- Prebuilt interception shims for torch 2.12.0, 2.12.1 and 2.13.0 (with
  2.9.1, 2.10.0 and 2.11.0): those install without a C++ compiler. CI tests
  torch 2.12 and 2.13, and the explicit API's ops built against 2.11 on 2.13.
- Faster converting copies with narrow rows (a head of 128 values under a
  permute, say): lanes now span rows with 16-byte accesses instead of moving
  one element each.
- `bench/bench_compile.py`: copyengine against torch eager and torch.compile
  on the same copies. The adopt Skill says when each fits.

## 0.4.0

- `bulk` copies through shared memory for pure copies on one GPU: TMA bulk
  copies on sm90 and later, `cp.async` on sm80 to sm89, admitted by a test
  copy and chosen by measurement (`stats()["cuda"][i]["bulk"]`).
- Quantization between a GPU and pinned host memory runs either in place
  (zero-copy) or through device staging (compress then move, move then
  decompress), chosen by measurement; with indices too (paged offload: GPU
  pages by `x_index`, host pool slots by `index`, the slots moved by batched
  DMA). Staging runs in chunks, so DMA overlaps the kernels, and within the
  staging budget.
- Faster quantization kernels, most where few SMs run them (`max_sms`).
- Fixes: explicit quantize calls with host indices planned anew on every
  call (new index values missed the plan cache), so they were never tuned;
  device staging of several tile ops in one group could write descriptors
  through dangling pointers.
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
