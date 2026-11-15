"""ce.quantize / ce.dequantize against the torch recipes (copyengine._quant),
bit for bit, through copyengine's own strategy and through the torch baseline."""

import pytest

import copyengine as ce
from copyengine import _quant

torch = pytest.importorskip("torch")
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

FORMATS = [("fp8_e4m3", 128), ("fp8_e5m2", 64), ("int8", 32), ("int4", 32), ("uint4", 64),
           ("mxfp8_e4m3", 32), ("mxfp8_e5m2", 32), ("nvfp4", 16)]
Q_DTYPE = {"fp8_e4m3": torch.float8_e4m3fn, "mxfp8_e4m3": torch.float8_e4m3fn, "fp8_e5m2": torch.float8_e5m2,
           "mxfp8_e5m2": torch.float8_e5m2, "int8": torch.int8}
S_DTYPE = {"mxfp8_e4m3": torch.float8_e8m0fnu, "mxfp8_e5m2": torch.float8_e8m0fnu, "nvfp4": torch.float8_e4m3fn}


def outputs(fmt, lead, D, g, dev):
    packed = fmt in _quant.PACKED
    q = torch.zeros(*lead, D // 2 if packed else D, dtype=Q_DTYPE.get(fmt, torch.uint8), device=dev)
    scale = torch.zeros(*lead, D // g, dtype=S_DTYPE.get(fmt, torch.float32), device=dev)
    zero = torch.zeros(*lead, D // g, dtype=torch.uint8, device=dev) if fmt == "uint4" else None
    return q, scale, zero


def data(lead, D, dtype, dev, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*lead, D, generator=g) * torch.exp(torch.randn(*lead, 1, generator=g) * 4)
    x.view(-1)[:3] = torch.tensor([float("nan"), float("inf"), 0.0])
    return x.to(dtype).to(dev)


def rows(t, idx):
    """t[idx] through a byte view: old torch cannot index fp8 tensors on the CPU."""
    if t.element_size() == 1:
        return t.view(torch.uint8)[idx.to(t.device)].view(t.dtype)
    return t[idx.to(t.device)]


def engines():
    # The torch baseline only, and ce's strategies while exploring (the
    # recipe's own bits either way).
    return [ce.Engine(disable=True), ce.Engine(tune_trials=1, tune_topk=4)]


def check(fmt, g, x, q, scale, zero, gs=1.0):
    codes, s, bits = _quant.quantize(x.cpu(), g, fmt, gs=gs)
    got = q.cpu().view(torch.uint8).reshape(-1)
    if fmt in Q_DTYPE and fmt != "int8":  # fp8 NaN patterns differ between torch's CPU and CUDA casts: positions
        nan = got.view(Q_DTYPE[fmt]).float().isnan()
        assert torch.equal(nan, codes.view(Q_DTYPE[fmt]).float().isnan())
        assert torch.equal(got[~nan], codes[~nan])
    else:
        assert torch.equal(got, codes)
    if scale.dtype == torch.float32:
        a, b = scale.cpu().reshape(-1), s
        nan = a.isnan()
        assert torch.equal(nan, b.isnan()) and torch.equal(a[~nan], b[~nan])
    else:
        assert torch.equal(scale.cpu().view(torch.uint8).reshape(-1), bits)
    if zero is not None:
        assert torch.equal(zero.cpu().reshape(-1), bits)


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("fmt,g", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quantize(dev, fmt, g, dtype):
    lead, D = (3, 5), 4 * g
    x = data(lead, D, dtype, dev)
    gs = 0.02 if fmt == "nvfp4" else 1.0
    for e in engines():
        for _ in range(6):
            q, scale, zero = outputs(fmt, lead, D, g, dev)
            e.quantize(x, q, scale, zero, fmt=fmt, group=g, global_scale=gs)
            check(fmt, g, x, q, scale, zero, gs)


@pytest.mark.parametrize("fmt,g", FORMATS)
def test_strided_rows_and_dequantize(fmt, g):
    D = 2 * g
    x = data((6, 4), D, torch.float32, "cpu").transpose(0, 1)  # strided rows
    gs = 0.02 if fmt == "nvfp4" else 1.0
    e = ce.Engine()
    q, scale, zero = outputs(fmt, (4, 6), D, g, "cpu")
    e.quantize(x, q, scale, zero, fmt=fmt, group=g, global_scale=gs)
    check(fmt, g, x, q, scale, zero, gs)
    out = torch.empty(4, 6, D, dtype=torch.bfloat16)
    e.dequantize(q, scale, out, zero, fmt=fmt, group=g, global_scale=gs)
    codes, s, bits = _quant.quantize(x, g, fmt, gs=gs)
    want = _quant.dequantize(codes, s, bits, g, fmt, gs).view(out.shape).to(out.dtype)
    nan = out.isnan()
    assert torch.equal(nan, want.isnan()) and torch.equal(out[~nan], want[~nan])


def test_layers_quantized_in_one_scope_are_one_group():
    e = ce.Engine(prior_only=True)
    xs = [data((8,), 256, torch.float32, "cpu", seed=k) for k in range(5)]
    outs = [outputs("fp8_e4m3", (8,), 256, 128, "cpu") for _ in xs]
    with e.scope():
        for x, (q, s, _) in zip(xs, outs):
            e.quantize(x, q, s, fmt="fp8_e4m3", group=128)
    last = e.stats()["last"]
    assert last["ops"] == 5 and last["groups"] == 1 and last["strategies"] == ["tile"]
    for x, (q, s, z) in zip(xs, outs):
        check("fp8_e4m3", 128, x, q, s, z)


def test_bad_shapes_are_rejected():
    x = torch.randn(4, 100)
    q, s, _ = outputs("fp8_e4m3", (4,), 128, 128, "cpu")
    with pytest.raises(ValueError):
        ce.quantize(x, q, s, fmt="fp8_e4m3", group=128)
    with pytest.raises(ValueError):
        ce.quantize(torch.randn(4, 64), *outputs("mxfp8_e4m3", (4,), 64, 32, "cpu")[:2], fmt="mxfp8_e4m3", group=16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fmt,g", FORMATS + [("fp8_e4m3", 4096), ("uint4", 1024)])  # long units: many values per lane
def test_gpu_strided_and_dequantize(fmt, g):
    D = 2 * g
    x = data((6, 4), D, torch.float16, "cuda").transpose(0, 1)
    gs = 0.02 if fmt == "nvfp4" else 1.0
    e = ce.Engine(prior_only=True)
    q, scale, zero = outputs(fmt, (4, 6), D, g, "cuda")
    e.quantize(x, q, scale, zero, fmt=fmt, group=g, global_scale=gs)
    assert e.stats()["last"]["strategies"] == ["tile"]
    torch.cuda.synchronize()
    check(fmt, g, x, q, scale, zero, gs)
    out = torch.empty(4, 6, D, dtype=torch.float32, device="cuda")
    e.dequantize(q, scale, out, zero, fmt=fmt, group=g, global_scale=gs)
    torch.cuda.synchronize()
    codes, s, bits = _quant.quantize(x.cpu(), g, fmt, gs=gs)
    want = _quant.dequantize(codes, s, bits, g, fmt, gs).view(out.shape)
    got = out.cpu()
    nan = got.isnan()
    assert torch.equal(nan, want.isnan()) and torch.equal(got[~nan], want[~nan])


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("uint4", 64), ("nvfp4", 16)])
@pytest.mark.parametrize("gpu_index", [False, True])
def test_gather_then_quantize_into_cache_blocks(dev, fmt, g, gpu_index):
    # Pages gathered from a KV cache, quantized, written to other blocks of a
    # quantized cache: x_index and index, with host or GPU indices.
    if gpu_index and dev == "cpu":
        pytest.skip("GPU indices need GPU tensors")
    blocks, D = 32, 2 * g
    kv = data((blocks, 4), D, torch.bfloat16, dev, seed=5)          # [blocks, tokens, D]
    q, scale, zero = outputs(fmt, (blocks, 4), D, g, dev)
    gs = 0.02 if fmt == "nvfp4" else 1.0
    for e in engines():
        for k in range(6):
            src = torch.randperm(blocks)[:8]
            dst = torch.randperm(blocks)[:8]
            si, di = (src.to(dev), dst.to(dev)) if gpu_index else (src.numpy(), dst.tolist())
            e.quantize(kv, q, scale, zero, fmt=fmt, group=g, global_scale=gs, x_index=si, index=di)
            if dev == "cuda":
                torch.cuda.synchronize()
            check(fmt, g, kv[src], rows(q, dst), rows(scale, dst), None if zero is None else zero[dst], gs)
            back = torch.zeros(blocks, 4, D, dtype=torch.float32, device=dev)
            e.dequantize(q, scale, back, zero, fmt=fmt, group=g, global_scale=gs, x_index=si, index=di)
            if dev == "cuda":
                torch.cuda.synchronize()
            codes, s, bits = _quant.quantize(kv[src].cpu(), g, fmt, gs=gs)
            want = _quant.dequantize(codes, s, bits, g, fmt, gs).view(8, 4, D)
            got = back[src].cpu()
            nan = got.isnan()
            assert torch.equal(nan, want.isnan()) and torch.equal(got[~nan], want[~nan])


def test_out_of_range_index_raises():
    x = torch.randn(4, 4, 128)
    q, scale, _ = outputs("fp8_e4m3", (4, 4), 128, 128, "cpu")
    with pytest.raises(ValueError):
        ce.quantize(x, q, scale, fmt="fp8_e4m3", group=128, index=[0, 1, 9, 2])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_out_of_range_gpu_index_raises_at_synchronize():
    x = torch.randn(4, 4, 128, device="cuda")
    q, scale, _ = outputs("fp8_e4m3", (4, 4), 128, 128, "cuda")
    e = ce.Engine(prior_only=True)
    e.quantize(x, q, scale, fmt="fp8_e4m3", group=128, index=torch.tensor([0, 1, 9, 2], device="cuda"))
    with pytest.raises(ValueError, match="out of range"):
        e.synchronize()
    e.quantize(x, q, scale, fmt="fp8_e4m3", group=128, index=torch.tensor([0, 1, 3, 2], device="cuda"))
    e.synchronize()  # cleared


@pytest.mark.parametrize("dev", DEVS)
@pytest.mark.parametrize("fmt,g", FORMATS)
def test_static_scale(dev, fmt, g):
    # Scales from a calibration (here: a dynamic pass), reused for new data;
    # and one scale broadcast to every unit.
    lead, D = (4, 3), 2 * g
    gs = 0.02 if fmt == "nvfp4" else 1.0
    x0, x1 = data(lead, D, torch.float32, dev, seed=1), data(lead, D, torch.float32, dev, seed=2)
    q, scale, zero = outputs(fmt, lead, D, g, dev)
    ce.quantize(x0, q, scale, zero, fmt=fmt, group=g, global_scale=gs)
    for e in engines():
        q1 = torch.zeros_like(q)
        e.quantize(x1, q1, scale, zero, fmt=fmt, group=g, global_scale=gs, static=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        sb = scale.cpu().view(torch.uint8) if scale.element_size() == 1 else None
        bits = zero.cpu() if fmt == "uint4" else sb
        want = _quant.quantize_static(x1.cpu(), g, fmt, scale.cpu() if sb is None else None, bits, gs)
        got = q1.cpu().view(torch.uint8).reshape(-1)
        if fmt in Q_DTYPE and fmt != "int8":
            nan = got.view(Q_DTYPE[fmt]).float().isnan()
            assert torch.equal(nan, want.view(Q_DTYPE[fmt]).float().isnan()) and torch.equal(got[~nan], want[~nan])
        else:
            assert torch.equal(got, want)
    if fmt in ("fp8_e4m3", "int8"):  # one scale for everything: a broadcast (stride 0) scale tensor
        one = torch.full((1, 1), 0.05, device=dev).expand(*lead, D // g)
        q2 = torch.zeros_like(q)
        ce.quantize(x1, q2, one, fmt=fmt, group=g, static=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        want = _quant.quantize_static(x1.cpu(), g, fmt, torch.full((x1.numel() // g,), 0.05))
        got = q2.cpu().view(torch.uint8).reshape(-1)
        nan = got.view(torch.uint8) == 0x7F if fmt == "fp8_e4m3" else torch.zeros_like(got, dtype=torch.bool)
        assert torch.equal(got[~nan], want[~nan])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("nvfp4", 16), ("uint4", 64)])
def test_across_gpu_and_pinned_host(fmt, g):
    # Offload: quantize GPU pages straight into pinned host buffers, and load
    # them back (dequantize from pinned host into the GPU), in one pass each.
    blocks, D = 16, 2 * g
    gs = 0.02 if fmt == "nvfp4" else 1.0
    kv = data((blocks, 4), D, torch.bfloat16, "cuda", seed=9)
    q, scale, zero = (None if t is None else t.pin_memory() for t in outputs(fmt, (blocks, 4), D, g, "cpu"))
    seen = set()
    for e in engines():
        for _ in range(8):  # explore: zero-copy kernel, device staging, torch's recipe
            for t in (q, scale, zero):
                if t is not None:
                    t.zero_()
            e.quantize(kv, q, scale, zero, fmt=fmt, group=g, global_scale=gs)
            e.synchronize()
            seen.update(e.stats()["last"]["strategies"])
            check(fmt, g, kv, q, scale, zero, gs)
            back = torch.zeros(blocks, 4, D, device="cuda")
            e.dequantize(q, scale, back, zero, fmt=fmt, group=g, global_scale=gs)
            torch.cuda.synchronize()
            seen.update(e.stats()["last"]["strategies"])
            codes, s, bits = _quant.quantize(kv.cpu(), g, fmt, gs=gs)
            want = _quant.dequantize(codes, s, bits, g, fmt, gs).view(back.shape)
            got = back.cpu()
            nan = got.isnan()
            assert torch.equal(nan, want.isnan()) and torch.equal(got[~nan], want[~nan])
    assert "tile_staged" in seen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("uint4", 64)])
def test_many_layers_across_gpu_and_pinned_host_in_one_scope(fmt, g):
    # Many tile ops in one group (one per layer), each staged through the
    # device: every layer's staged addresses must land in its own descriptor.
    layers, blocks, D = 24, 4, 2 * g
    kvs = [data((blocks, 4), D, torch.bfloat16, "cuda", seed=k) for k in range(layers)]
    outs = [tuple(None if t is None else t.pin_memory() for t in outputs(fmt, (blocks, 4), D, g, "cpu"))
            for _ in kvs]
    e = ce.Engine(tune_trials=1, tune_topk=4)
    seen = set()
    for _ in range(8):
        for o in outs:
            for t in o:
                if t is not None:
                    t.zero_()
        with e.scope():
            for kv, (q, s, z) in zip(kvs, outs):
                e.quantize(kv, q, s, z, fmt=fmt, group=g)
        e.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        for kv, (q, s, z) in zip(kvs, outs):
            check(fmt, g, kv, q, s, z)
        backs = [torch.zeros(blocks, 4, D, device="cuda") for _ in kvs]
        with e.scope():
            for b, (q, s, z) in zip(backs, outs):
                e.dequantize(q, s, b, z, fmt=fmt, group=g)
        torch.cuda.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        for kv, b in zip(kvs, backs):
            codes, s, bits = _quant.quantize(kv.cpu(), g, fmt)
            want = _quant.dequantize(codes, s, bits, g, fmt).view(b.shape)
            got = b.cpu()
            nan = got.isnan()
            assert torch.equal(nan, want.isnan()) and torch.equal(got[~nan], want[~nan])
    assert "tile_staged" in seen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("nvfp4", 16), ("uint4", 64)])
@pytest.mark.parametrize("gpu_pages", [False, True])
def test_paged_offload_through_device_staging(fmt, g, gpu_pages):
    # Pages of a GPU KV cache (x_index: GPU or host ids) quantized into the
    # slots of a pinned host pool (index: host ids), and loaded back: device
    # staging gathers / scatters the pool's rows by DMA.
    blocks, slots_n, n, D = 32, 48, 8, 2 * g
    gs = 0.02 if fmt == "nvfp4" else 1.0
    kv = data((blocks, 4), D, torch.bfloat16, "cuda", seed=11)
    q, scale, zero = (None if t is None else t.pin_memory() for t in outputs(fmt, (slots_n, 4), D, g, "cpu"))
    seen = set()
    e = ce.Engine(tune_trials=1, tune_topk=4)
    for k in range(10):
        pages, slots = torch.randperm(blocks)[:n], torch.randperm(slots_n)[:n]
        xi = pages.cuda() if gpu_pages else pages.tolist()
        e.quantize(kv, q, scale, zero, fmt=fmt, group=g, global_scale=gs, x_index=xi, index=slots.numpy())
        e.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        check(fmt, g, kv[pages], rows(q, slots), rows(scale, slots), None if zero is None else zero[slots], gs)
        back = torch.full((blocks, 4, D), 7.0, device="cuda")
        e.dequantize(q, scale, back, zero, fmt=fmt, group=g, global_scale=gs, x_index=xi, index=slots.tolist())
        torch.cuda.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        codes, s, bits = _quant.quantize(kv[pages].cpu(), g, fmt, gs=gs)
        want = torch.full((blocks, 4, D), 7.0)
        want[pages] = _quant.dequantize(codes, s, bits, g, fmt, gs).view(n, 4, D)
        got = back.cpu()
        nan = got.isnan()
        assert torch.equal(nan, want.isnan()) and torch.equal(got[~nan], want[~nan])
    assert "tile_staged" in seen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("fmt,g", [("fp8_e4m3", 128), ("nvfp4", 16), ("uint4", 64)])
@pytest.mark.parametrize("slots", [False, True])
def test_staging_in_chunks(fmt, g, slots):
    # Larger than one staging round (4 MB budget: chunks of up to 2 MB in two
    # halves): DMA of one chunk overlaps the kernel of another.
    blocks, tokens, D = 64, 64, 1024  # host side: 2-4 MB of codes
    gs = 0.02 if fmt == "nvfp4" else 1.0
    kv = data((blocks, tokens), D, torch.bfloat16, "cuda", seed=13)
    pool = blocks + 8 if slots else blocks
    q, scale, zero = (None if t is None else t.pin_memory() for t in outputs(fmt, (pool, tokens), D, g, "cpu"))
    idx = torch.randperm(pool)[:blocks] if slots else None
    ix = None if idx is None else idx.tolist()
    e = ce.Engine(tune_trials=1, tune_topk=4, staging_budget_mb=4)
    seen, chunked = set(), 0
    for _ in range(6):
        e.quantize(kv, q, scale, zero, fmt=fmt, group=g, global_scale=gs, index=ix)
        e.synchronize()
        last = e.stats()["last"]
        seen.update(last["strategies"])
        if last["strategies"] == ["tile_staged"]:
            chunked = max(chunked, last["calls"])
        sel = idx if idx is not None else torch.arange(blocks)
        check(fmt, g, kv, rows(q, sel), rows(scale, sel), None if zero is None else zero[sel], gs)
        back = torch.zeros(blocks, tokens, D, device="cuda")
        e.dequantize(q, scale, back, zero, fmt=fmt, group=g, global_scale=gs, index=ix)
        torch.cuda.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        codes, s, bits = _quant.quantize(kv.cpu(), g, fmt, gs=gs)
        want = _quant.dequantize(codes, s, bits, g, fmt, gs).view(back.shape)
        got = back.cpu()
        nan = got.isnan()
        assert torch.equal(nan, want.isnan()) and torch.equal(got[~nan], want[~nan])
    assert "tile_staged" in seen and chunked >= 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_staging_interleaved_host_rows():
    # Host tensors viewed transposed (rows of the first dimension interleave):
    # staged as one piece, results unchanged.
    g, T, C = 64, 128, 512
    v = data((T,), C, torch.bfloat16, "cuda", seed=14)                     # [tokens, channels]
    q = torch.zeros(C, T, dtype=torch.float8_e4m3fn).pin_memory()        # per channel over tokens
    scale = torch.zeros(T // g, C).pin_memory()
    e = ce.Engine(tune_trials=1, tune_topk=4)
    seen = set()
    for _ in range(6):
        e.quantize(v.transpose(0, 1), q, scale.transpose(0, 1), fmt="fp8_e4m3", group=g)
        e.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        check("fp8_e4m3", g, v.transpose(0, 1).contiguous(), q, scale.transpose(0, 1).contiguous(), None)
    assert seen <= {"tile", "tile_staged", "native"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_staging_leaves_other_host_rows_alone():
    # Quantizing into some slots of a pinned pool through staging writes
    # those slots only.
    g, n = 128, 4
    kv = data((n, 4), 256, torch.float32, "cuda", seed=12)
    q, scale = (t.pin_memory() for t in outputs("fp8_e4m3", (16, 4), 256, g, "cpu")[:2])
    e = ce.Engine(tune_trials=1, tune_topk=4)
    slots = [3, 9, 0, 14]
    seen = set()
    for _ in range(8):
        q.view(torch.uint8).fill_(0xAB)
        scale.fill_(-1.0)
        e.quantize(kv, q, scale, fmt="fp8_e4m3", group=g, index=slots)
        e.synchronize()
        seen.update(e.stats()["last"]["strategies"])
        others = [i for i in range(16) if i not in slots]
        assert bool((q.view(torch.uint8)[others] == 0xAB).all()) and bool((scale[others] == -1.0).all())
        check("fp8_e4m3", g, kv, rows(q, torch.tensor(slots)), scale[slots], None)
    assert "tile_staged" in seen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_step_plan_quantizes_with_new_gpu_indices():
    blocks, D, g = 32, 256, 128
    kv = data((blocks, 4), D, torch.float32, "cuda", seed=3)
    q, scale, _ = outputs("fp8_e4m3", (blocks, 4), D, g, "cuda")

    def offload(kv, q, scale, pages, slots):
        ce.quantize(kv, q, scale, fmt="fp8_e4m3", group=g, x_index=pages, index=slots)

    p = ce.Engine().plan(offload)
    replayed = False
    for k in range(200):
        pages, slots = torch.randperm(blocks, device="cuda")[:8], torch.randperm(blocks, device="cuda")[:8]
        p(kv, q, scale, pages, slots)
        torch.cuda.synchronize()
        check("fp8_e4m3", g, kv[pages], rows(q, slots), rows(scale, slots), None)
        tr = next(iter(p._layouts.values()))["traces"]
        if tr and tr[-1].site is not None and tr[-1].site.decided is not None:
            replayed = True
            if k > 80:
                break
    assert replayed and tr[-1].plain is None


@pytest.mark.parametrize("dev", DEVS)
def test_step_plan_quantizes_with_new_host_indices(dev):
    # Host block ids (a list, new each step) are plan parameters: no new trace.
    blocks, D, g = 32, 256, 128
    kv = data((blocks, 4), D, torch.float32, dev, seed=4)
    q, scale, _ = outputs("fp8_e4m3", (blocks, 4), D, g, dev)

    def offload(kv, q, scale, pages, slots):
        ce.quantize(kv, q, scale, fmt="fp8_e4m3", group=g, x_index=pages, index=slots)

    p = ce.Engine().plan(offload)
    for k in range(150):
        pages, slots = torch.randperm(blocks)[:8].tolist(), torch.randperm(blocks)[:8].numpy()
        p(kv, q, scale, pages, slots)
        if dev == "cuda":
            torch.cuda.synchronize()
        check("fp8_e4m3", g, kv[torch.tensor(pages)], rows(q, torch.from_numpy(slots)),
              rows(scale, torch.from_numpy(slots)), None)
    lay = next(iter(p._layouts.values()))
    assert len(p._layouts) == 1 and len(lay["traces"]) == 1, "new host indices traced again"
    tr = lay["traces"][0]
    assert tr.plain is None and len(tr.params) == 2 and not tr.vguard
