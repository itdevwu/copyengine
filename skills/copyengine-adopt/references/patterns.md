# Integration patterns (copyengine 0.3.x)

Each pattern shows the original, the adopted code, and how to check it.

## Loop of copy_ calls (no code change)

```python
# original
for l in range(L):
    for i, b in enumerate(block_ids):
        kv_cache[l, b].copy_(host_slot[l, i])

# adopted
import copyengine as ce
with ce.scope():
    for l in range(L):
        for i, b in enumerate(block_ids):
            kv_cache[l, b].copy_(host_slot[l, i])
```

The copies are queued and merged at scope exit; `non_blocking` keeps its
meaning. Results are torch's.

## Gather + layout change + cast in one pass

```python
# original: three passes, two temporaries
out.copy_(cache[ids].permute(0, 2, 1, 3).to(torch.float16))

# adopted
ce.copy_(out, ce.lazy(cache).index_select(0, ids).permute(0, 2, 1, 3))
```

`ce.lazy` views support `index_select`, `slice`, `select`, `permute`,
`transpose`, `flip`, `unsqueeze`, `expand`, `reshape`. `ids` should be a host
list or CPU tensor. Writing through a lazy destination scatters:
`ce.copy_(ce.lazy(cache).index_select(0, ids), new_rows)`.

## Quantize / dequantize while copying

```python
lim = torch.finfo(torch.float8_e4m3fn).max
ce.copy_(q, x, fn=ce.Elementwise(scale=1 / s, clamp=(-lim, lim)))   # saturating FP8 quantize
ce.copy_(y, q, fn=ce.Elementwise(scale=s))                          # dequantize
ce.copy_(q8, x, fn=ce.Elementwise(scale=1 / s, round=True, clamp=(-127, 127)))  # int8
```

The function is `clamp(round(x * scale + bias))` in float32 without FMA, so it
equals `(x.float() * scale + bias)` with the same rounding and clamping in
torch. Without `clamp`, FP8 e4m3fn overflow becomes NaN, as in torch.

## Ragged segments

```python
ce.segments(src, dst, src_off, dst_off, length)   # element offsets into contiguous tensors
```

## Same copies every step: freeze

```python
def refresh(e):                       # the step's copies, in one scope
    with e.scope():
        for l in range(L):
            e.copy_(graph_inputs[l], staging[l])

eng = ce.Engine()
eng.warmup(refresh)                   # until its decisions are locked
refresh(eng)
plan = eng.freeze()                   # the last flush, with the locked strategies
plan.run()                            # each step; also inside torch.cuda.graph
```

A frozen plan replays the same buffers, layouts and indices: re-freeze when
they change.

## Overlap with compute

```python
with ce.scope(stream="copy") as h:    # or stream=<torch.cuda.Stream>
    for l in range(L):
        next_kv[l].copy_(host_kv[l, ids], non_blocking=True)
y = model_layer(x)                    # runs concurrently on the current stream
h.wait()                              # GPU-side: later work sees next_kv
```

The scope starts after work already queued on the current stream; the host
never waits. Inside a CUDA graph capture it is a fork and `h.wait()` the join;
create the stream before capturing (`eng.copy_stream()`).

## Leave SMs to compute

```python
eng = ce.Engine(max_sms=8)            # or CE_MAX_SMS=8
print(eng.stats()["cuda"][0]["sm_limit"])  # "hard: green context ..." or "soft: ..."
```

DMA copies use copy engines, not SMs; memory bandwidth is still shared.


## Quantize while copying (0.4)

```python
# Offload KV pages to a pinned FP8 cache, scales per group of 128 values.
ce.quantize(kv, q_cache, scales, fmt="fp8_e4m3", group=128, x_index=pages, index=slots)
# Load them back.
ce.dequantize(q_cache, scales, kv, fmt="fp8_e4m3", group=128, index=slots, x_index=pages)
# Per channel over tokens (a V cache): quantize the transposed view.
ce.quantize(v.transpose(-1, -2), qv.transpose(-1, -2), sv, fmt="int8", group=16)
```

The cache may be in pinned host memory: the kernel then writes it directly
(zero-copy) or through device staging in chunks (DMA overlapping the
kernels), whichever measures faster; host `index=` values select pool slots.

Only quantize tokens that hold data (pass their rows with `index=`); keep an
incomplete group in full precision until it fills. Results follow the
recipes in `copyengine/_quant.py` on every device; torch code computing the
same formula another way may differ in the last bit.
