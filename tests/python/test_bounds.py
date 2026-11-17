"""Bounds and gates found by the 2026-11-17 review: index lengths the kernels
read, opaque copies grouped with run-described ones, dtype gates on baselines,
zero-dimensional host indices."""

import numpy as np
import pytest

import copyengine as ce

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
DEV = "cuda"


def test_quantize_op_checks_index_lengths():
    x = torch.randn(16, 128, device=DEV)
    q = torch.zeros(16, 128, dtype=torch.int8, device=DEV)
    s = torch.zeros(16, 1, device=DEV)
    xi = torch.arange(8, device=DEV)
    ci = torch.arange(4, device=DEV)  # fewer than x_index: the kernel would read past it
    op = torch.ops.copyengine.quantize_
    with pytest.raises((RuntimeError, ValueError), match="indices, not rows"):
        op(x, q, s, None, 2, 128, 1, 1e-12, 1.0, False, xi, ci, 0)  # fmt 2: int8


def test_opaque_and_run_copies_in_one_scope():
    # A gather with GPU indices and a segments copy of the same sides: they used to share a
    # group whose forms were empty, and the flush failed for copies that work alone.
    src = torch.randint(0, 255, (64, 32), dtype=torch.uint8, device=DEV)
    out = torch.zeros(8, 32, dtype=torch.uint8, device=DEV)
    a = torch.randint(0, 255, (4096,), dtype=torch.uint8, device=DEV)
    b = torch.zeros(4096, dtype=torch.uint8, device=DEV)
    idx = torch.randperm(64, device=DEV)[:8]
    e = ce.Engine()
    with e.scope():
        e.gather(src, idx, 0, out=out)
        e.segments(a, b, np.array([0, 1024]), np.array([2048, 0]), np.array([512, 512]))
    torch.cuda.synchronize()
    assert torch.equal(out, src[idx])
    assert torch.equal(b[2048:2560], a[0:512]) and torch.equal(b[0:512], a[1024:1536])


def test_copy_many_never_takes_torch_bits_for_other_dtype_pairs():
    src = [torch.tensor([float("nan"), 3.7e9, -1.5, 2.5, 1e30], device=DEV) for _ in range(4)]
    e = ce.Engine()
    for k in range(60):
        dst = [torch.zeros(5, dtype=torch.int32, device=DEV) for _ in range(4)]
        e.copy_many(dst, src)
    torch.cuda.synchronize()
    sites = e.stats()["copy_many"]
    assert all(s["copyengine"] is not False for s in sites), sites  # never decided for torch's copies


def test_zero_dimensional_host_index_in_a_step_plan():
    src = torch.randn(16, 8, device=DEV)
    out = torch.zeros(1, 8, device=DEV)
    e = ce.Engine()

    def step(out, src, i):
        e.copy_(ce.lazy(out), ce.lazy(src).index_select(0, i))

    p = e.plan(step)
    for k in range(40):
        i = np.array(k % 16)  # 0-d
        p(out, src, i)
        torch.cuda.synchronize()
        assert torch.equal(out[0], src[k % 16]), k
