"""Turn numpy arrays and torch tensors into (Endpoint, Access) pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _core

try:  # torch is optional
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

_NP_DTYPES = {
    "float64": "float64", "float32": "float32", "float16": "float16", "bfloat16": "bfloat16",
    "int64": "int64", "int32": "int32", "int16": "int16", "int8": "int8", "uint8": "uint8", "bool": "bool",
}

if torch is not None:
    _TORCH_DTYPES = {
        torch.float64: "float64", torch.float32: "float32", torch.float16: "float16",
        torch.bfloat16: "bfloat16", torch.int64: "int64", torch.int32: "int32", torch.int16: "int16",
        torch.int8: "int8", torch.uint8: "uint8", torch.bool: "bool",
    }


@dataclass
class View:
    """A tensor as the engine sees it."""

    obj: Any                 # the array / tensor (kept alive while queued)
    endpoint: _core.Endpoint  # the whole allocation
    access: _core.Access     # this view's elements, relative to the allocation
    is_cuda: bool
    device_index: int


def is_tensor(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def view(x, *, writable: bool = False) -> View:
    if is_tensor(x):
        return _torch_view(x)
    if isinstance(x, np.ndarray):
        return _numpy_view(x, writable)
    raise TypeError(f"expected a numpy array or torch tensor, got {type(x).__name__}")


def _torch_view(t) -> View:
    if t.layout != torch.strided:
        raise _core.Unsupported(f"only strided tensors are supported, got {t.layout}")
    if t.dtype not in _TORCH_DTYPES:
        raise _core.Unsupported(f"unsupported dtype {t.dtype}")
    st = t.untyped_storage()
    if t.device.type == "cuda":
        kind, index, pinned = "cuda", t.device.index if t.device.index is not None else 0, False
    elif t.device.type == "cpu":
        kind, index, pinned = "host", 0, bool(t.is_pinned())
    else:
        raise _core.Unsupported(f"unsupported device {t.device}")
    ep = _core.Endpoint(kind, index, st.data_ptr(), st.nbytes(), _TORCH_DTYPES[t.dtype], pinned)
    acc = _core.Access.strided(list(t.shape), list(t.stride()), t.storage_offset())
    return View(t, ep, acc, kind == "cuda", index)


def _numpy_view(a: np.ndarray, writable: bool) -> View:
    name = _NP_DTYPES.get(a.dtype.name)
    if name is None:
        raise _core.Unsupported(f"unsupported dtype {a.dtype}")
    if writable and not a.flags.writeable:
        raise ValueError("destination array is read-only")
    it = a.itemsize
    ptr = a.__array_interface__["data"][0]
    # The owning array bounds every view of it; fall back to the view's own span.
    root = a
    while isinstance(root.base, np.ndarray):
        root = root.base
    if root.flags.c_contiguous or root.flags.f_contiguous:
        base, nbytes = root.__array_interface__["data"][0], root.nbytes
    else:
        lo = ptr + sum(min(0, (n - 1) * s) for n, s in zip(a.shape, a.strides) if n > 0)
        hi = ptr + sum(max(0, (n - 1) * s) for n, s in zip(a.shape, a.strides) if n > 0) + it
        base, nbytes = lo, (hi - lo) if a.size else 0
    if base % it or (ptr - base) % it or any(s % it for s in a.strides):
        raise _core.Unsupported("array is not aligned to its itemsize")
    ep = _core.Endpoint("host", 0, base, nbytes, name, False)
    acc = _core.Access.strided(list(a.shape), [s // it for s in a.strides], (ptr - base) // it)
    return View(a, ep, acc, False, 0)


def flat_endpoint(x, *, writable: bool = False) -> tuple[_core.Endpoint, int, Any]:
    """Endpoint for flat element offsets into a contiguous tensor/array.

    Returns (endpoint, element offset of x[0] within the endpoint, keepalive).
    """
    v = view(x, writable=writable)
    contiguous = x.is_contiguous() if is_tensor(x) else x.flags.c_contiguous
    if not contiguous:
        raise ValueError("segment / index copies need a contiguous tensor; use copy_ for strided views")
    return v.endpoint, v.access.base, v


def broadcast_to(acc: _core.Access, shape: list[int]) -> _core.Access:
    src = list(acc.shape)
    if len(src) > len(shape):
        raise ValueError(f"cannot broadcast shape {src} to {shape}")
    for _ in range(len(shape) - len(src)):
        acc = acc.unsqueeze(0)
    for d, (s, n) in enumerate(zip(acc.shape, shape)):
        if s == n:
            continue
        if s != 1:
            raise ValueError(f"cannot broadcast shape {src} to {shape}")
        acc = acc.expand(d, n)
    return acc


def current_stream(*views: View) -> int:
    """The torch current stream of the CUDA device involved, or 0."""
    for v in views:
        if v.is_cuda:
            return torch.cuda.current_stream(v.device_index).cuda_stream
    return 0


def index_array(index) -> np.ndarray:
    """Host int64 copy of an index (a CUDA index tensor costs a device sync)."""
    if is_tensor(index):
        index = index.detach().to("cpu", torch.int64).numpy()
    return np.ascontiguousarray(index, dtype=np.int64).reshape(-1)
