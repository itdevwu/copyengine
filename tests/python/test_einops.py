"""einops on lazy views: rearrangements compose into ce.lazy (no
materialization) and the whole chain is one planned copy, bit for bit as the
same einops pattern on the torch tensor."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
einops = pytest.importorskip("einops")
DEVS = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("dev", DEVS)
def test_rearrange_split_permute_gather(dev):
    kv = torch.randn(8 * 16, 4, 32, device=dev)  # [(blocks tok), heads, d]
    pages = [5, 1, 7]
    want = einops.rearrange(kv, "(blk tok) h d -> blk h tok d", tok=16)[pages]
    out = torch.empty_like(want)
    e = ce.Engine(prior_only=True)
    src = einops.rearrange(ce.lazy(kv), "(blk tok) h d -> blk h tok d", tok=16)
    assert isinstance(src, ce.Lazy) and tuple(src.shape) == (8, 4, 16, 32)
    e.copy_(out, src.index_select(0, pages))
    if dev == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(out, want)
    assert e.stats()["last"]["ops"] == 1


@pytest.mark.parametrize("dev", DEVS)
def test_rearrange_into_a_destination_and_cast(dev):
    x = torch.randn(2, 6, 8, 16, device=dev)
    out = torch.empty(2, 8, 6, 16, device=dev, dtype=torch.float16)
    ce.copy_(out, einops.rearrange(ce.lazy(x), "b s h d -> b h s d"))
    if dev == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(out, einops.rearrange(x, "b s h d -> b h s d").to(torch.float16))
    # The destination side may be rearranged too: write through a lazy view of it.
    back = torch.zeros(2, 6, 8, 16, device=dev)
    ce.copy_(einops.rearrange(ce.lazy(back), "b s h d -> b h s d"), out)
    if dev == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(back, x.to(torch.float16).float())


@pytest.mark.parametrize("dev", DEVS)
def test_repeat_and_what_is_not_a_view_raises(dev):
    x = torch.randn(4, 8, device=dev)
    out = torch.empty(4, 3, 8, device=dev)
    ce.copy_(out, einops.repeat(ce.lazy(x), "a d -> a r d", r=3))
    if dev == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(out, einops.repeat(x, "a d -> a r d", r=3))
    with pytest.raises(NotImplementedError):
        einops.reduce(ce.lazy(x), "a d -> a", "max")
    # Repeating into an existing axis reads each source value twice: a gather.
    out2 = torch.empty(8, 8, device=dev)
    ce.copy_(out2, einops.repeat(ce.lazy(x), "a d -> (a r) d", r=2))
    if dev == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(out2, einops.repeat(x, "a d -> (a r) d", r=2))


@pytest.mark.parametrize("dev", DEVS)
def test_merges_that_are_not_views(dev):
    # "(t h)" after a transpose, and a merge after a gather by host indices:
    # no torch view expresses them; the merged dimension is a table of offsets.
    x = torch.randn(2, 8, 6, 16, device=dev)  # b h t d
    out = torch.empty(2, 48, 16, device=dev)
    ce.copy_(out, einops.rearrange(ce.lazy(x), "b h t d -> b (t h) d"))
    if dev == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(out, einops.rearrange(x, "b h t d -> b (t h) d"))
    kv = torch.randn(10, 4, 8, device=dev)
    pages = [9, 2, 4]
    out2 = torch.empty(12, 8, device=dev)
    ce.copy_(out2, einops.rearrange(ce.lazy(kv).index_select(0, pages), "p h d -> (p h) d"))
    if dev == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(out2, kv[pages].reshape(12, 8))


@pytest.mark.skipif(not torch.cuda.is_available() or not ce._compiled.available(), reason="needs CUDA and torch ops")
def test_rearrange_in_a_compiled_region(monkeypatch):
    monkeypatch.setenv("CE_COMPILE_BASELINE", "0")
    kv = torch.randn(8 * 16, 4, 32, device="cuda")
    out = torch.empty(3, 4, 16, 32, device="cuda")

    def f(idx):
        src = einops.rearrange(ce.lazy(kv), "(blk tok) h d -> blk h tok d", tok=16)
        ce.copy_(out, src.index_select(0, idx))
        return idx

    idx = torch.tensor([5, 1, 7], device="cuda")
    torch._dynamo.reset()
    torch.compile(f)(idx)
    torch.cuda.synchronize()
    assert torch.equal(out, einops.rearrange(kv, "(blk tok) h d -> blk h tok d", tok=16)[idx])


@pytest.mark.parametrize("dev", DEVS)
def test_ce_rearrange_and_repeat_match_einops(dev):
    kv = torch.randn(8 * 16, 4, 32, device=dev)
    out = torch.empty(8, 4, 16, 32, device=dev)
    ce.copy_(out, ce.rearrange(ce.lazy(kv), "(blk tok) h d -> blk h tok d", tok=16))
    out2 = torch.empty(8 * 16, 2, 4, 32, device=dev)
    ce.copy_(out2, ce.repeat(ce.lazy(kv), "n h d -> n r h d", r=2))
    if dev == "cuda":
        torch.cuda.synchronize()
    assert torch.equal(out, einops.rearrange(kv, "(blk tok) h d -> blk h tok d", tok=16))
    assert torch.equal(out2, einops.repeat(kv, "n h d -> n r h d", r=2))


@pytest.mark.skipif(not torch.cuda.is_available() or not ce._compiled.available(), reason="needs CUDA and torch ops")
def test_ce_rearrange_in_a_compiled_region_is_one_graph(monkeypatch):
    # einops' own functions may be graph nodes (allow_in_graph) that a lazy
    # view cannot enter; ce.rearrange computes the view ops as a constant.
    monkeypatch.setenv("CE_COMPILE_BASELINE", "0")
    kv = torch.randn(8 * 16, 4, 32, device="cuda")
    out = torch.empty(3, 4, 16, 32, device="cuda")

    def f(idx):
        ce.copy_(out, ce.rearrange(ce.lazy(kv), "(blk tok) h d -> blk h tok d", tok=16).index_select(0, idx))
        return idx

    idx = torch.tensor([5, 1, 7], device="cuda")
    torch._dynamo.reset()
    ex = torch._dynamo.explain(f)(idx)
    assert ex.graph_count == 1 and ex.graph_break_count == 0
    torch._dynamo.reset()
    torch.compile(f)(idx)
    torch.cuda.synchronize()
    assert torch.equal(out, einops.rearrange(kv, "(blk tok) h d -> blk h tok d", tok=16)[idx])
