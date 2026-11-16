"""`ce.lazy(t)[key]` and `narrow`: torch's basic indexing on lazy views, on the
host (no GPU needed), checked against torch."""

import pytest

import copyengine as ce

torch = pytest.importorskip("torch")

KEYS = [
    (slice(None), slice(2, 5)),
    (1,),
    (-1, Ellipsis, 3),
    (Ellipsis, slice(None, None, 2)),
    (None, 2, slice(1, -1)),
    (slice(-3, None), None, Ellipsis),
    ([3, 0, 2],),
    (slice(None), [4, 1]),
    (2, Ellipsis, torch.tensor([0, 5, 5, 1])),
]


@pytest.mark.parametrize("key", KEYS, ids=[str(k) for k in KEYS])
def test_getitem_matches_torch(key):
    x = torch.arange(4 * 6 * 7, dtype=torch.float32).reshape(4, 6, 7)
    want = x[key]
    out = torch.empty_like(want)
    ce.copy_(out, ce.lazy(x)[key])
    assert torch.equal(out, want)


def test_narrow_matches_torch():
    x = torch.randn(5, 9)
    out = torch.empty(5, 4)
    ce.copy_(out, ce.lazy(x).narrow(1, 3, 4))
    assert torch.equal(out, x.narrow(1, 3, 4))


@pytest.mark.parametrize("key, err", [
    ((torch.tensor([True, False, True, False]),), TypeError),
    (([0, 1], [1, 2]), TypeError),
    ((slice(None, None, -1),), ValueError),
    ((0, 0, 0, 0), IndexError),
    ((Ellipsis, Ellipsis), IndexError),
])
def test_unsupported_keys_refuse(key, err):
    with pytest.raises(err):
        ce.lazy(torch.zeros(4, 6, 7))[key]
