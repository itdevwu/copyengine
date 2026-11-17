"""Vector widths come from every row start's alignment (base and outer strides),
not the base alone (fuzzer seed 18: 16-byte rows 24 bytes apart misaligned
every other row, a CUDA fault). Every strategy forced on such layouts."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"

# (destination base shape, view of it, source) with row strides that are not multiples of 16 bytes
CASES = [
    ((4, 6), lambda t: t[:, 0:4], "pinned"),        # 16-byte rows 24 bytes apart (the seed)
    ((5, 6, 10), lambda t: t[:, 1:3, 2:6], "cuda"),  # two outer strides, offset base
    ((8, 3, 4), lambda t: t[1:, :, :], "cuda"),     # 12-byte middle stride
    ((6, 10), lambda t: t[:, 2:10], "pinned"),      # 32-byte rows 40 bytes apart, base offset 8
]


@pytest.mark.parametrize("strategy", ["fused", "kernel", "staged", "runs", "batch"])
@pytest.mark.parametrize("case", range(len(CASES)))
def test_rows_at_unaligned_strides(strategy, case):
    shape, view, kind = CASES[case]
    dst = torch.zeros(shape, device=DEV)
    v = view(dst)
    src = torch.randn(v.shape)
    src = src.pin_memory() if kind == "pinned" else src.to(DEV)
    e = ce.Engine(_force=strategy)
    e.copy_(v, src)
    torch.cuda.synchronize()
    want = torch.zeros(shape)
    view(want).copy_(src.cpu())
    assert torch.equal(dst.cpu(), want)
