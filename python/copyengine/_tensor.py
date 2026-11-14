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
        torch.float8_e4m3fn: "float8_e4m3fn", torch.float8_e5m2: "float8_e5m2",
        torch.float8_e8m0fnu: "float8_e8m0fnu", torch.float4_e2m1fn_x2: "float4_e2m1fn_x2",
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


class Lazy:
    """A tensor plus data-movement ops recorded as Access algebra, not executed.

    ``ce.lazy(cache).index_select(0, ids).permute(0, 2, 1, 3)`` describes a view
    torch cannot express without materializing; the engine copies it in one pass.
    The ops are also kept, so the same copy can be replayed with torch ops: that
    is the baseline copyengine must beat (see :func:`torch_replay`).
    """

    def __init__(self, base: View, access: _core.Access, ops: tuple = ()):
        self._base, self._access, self._ops = base, access, ops

    def _with(self, access, op) -> "Lazy":
        return Lazy(self._base, access, self._ops + (op,))

    @property
    def shape(self) -> tuple:
        return tuple(self._access.shape)

    def permute(self, *dims):
        dims = dims[0] if len(dims) == 1 and isinstance(dims[0], (list, tuple)) else dims
        return self._with(self._access.permute(list(dims)), ("permute", tuple(dims)))

    def transpose(self, a: int, b: int):
        p = list(range(len(self.shape)))
        p[a], p[b] = p[b], p[a]
        return self.permute(p)

    def index_select(self, dim: int, index):
        idx = index_array(index)
        return self._with(self._access.index_select(dim, idx), ("index_select", dim, idx))

    def slice(self, dim: int, start=None, stop=None, step: int = 1):
        return self._with(self._access.slice(dim, start, stop, step), ("slice", dim, start, stop, step))

    def select(self, dim: int, index: int):
        return self._with(self._access.select(dim, index), ("select", dim, index))

    def flip(self, dim: int):
        return self._with(self._access.flip(dim), ("flip", dim))

    def unsqueeze(self, dim: int):
        return self._with(self._access.unsqueeze(dim), ("unsqueeze", dim))

    def expand(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return self._with(broadcast_to(self._access, list(shape)), ("expand", tuple(shape)))

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (list, tuple)) else shape
        return self._with(self._access.reshape(list(shape)), ("reshape", tuple(shape)))

    def as_view(self) -> View:
        b = self._base
        return View(b.obj, b.endpoint, self._access, b.is_cuda, b.device_index)


def device_index(idx: np.ndarray, device) -> "torch.Tensor":
    """`idx` as an int64 tensor on `device`, sent from pinned memory without
    making the host wait (what a torch user would do with host indices)."""
    t = torch.from_numpy(idx)
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t


_COPYING = ("index_select", "flip")  # torch materializes these: not views


def torch_replay(x: "Lazy"):
    """The lazy view rebuilt with torch ops on its base tensor, or None if the
    base is not a tensor or torch cannot express an op (a negative step)."""
    t = x._base.obj
    if not is_tensor(t):
        return None
    for op in x._ops:
        name = op[0]
        if name == "permute":
            t = t.permute(*op[1])
        elif name == "index_select":
            t = t.index_select(op[1], device_index(op[2], t.device))
        elif name == "slice":
            _, dim, start, stop, step = op
            if step is not None and step < 0:
                return None
            t = t[(slice(None),) * dim + (slice(start, stop, step),)]
        elif name == "select":
            t = t.select(op[1], op[2])
        elif name == "flip":
            t = t.flip(op[1])
        elif name == "unsqueeze":
            t = t.unsqueeze(op[1])
        elif name == "expand":
            t = t.expand(*op[1])
        elif name == "reshape":
            t = t.reshape(*op[1])
        else:
            return None
    return t


def torch_dst(x: "Lazy"):
    """How torch would write into the lazy destination `x`: ("view", v) for a
    view, ("index_copy", base, dim, index) for exactly one index_select, else None."""
    t = x._base.obj
    if not is_tensor(t):
        return None
    ops = x._ops
    if len(ops) == 1 and ops[0][0] == "index_select":
        return ("index_copy", t, ops[0][1], ops[0][2])
    if any(op[0] in _COPYING for op in ops):
        return None
    v = torch_replay(x)
    return None if v is None else ("view", v)


def lazy(x) -> Lazy:
    if isinstance(x, Lazy):
        return x
    v = view(x)
    return Lazy(v, v.access)


def view(x, *, writable: bool = False) -> View:
    if isinstance(x, Lazy):
        if writable and not is_tensor(x._base.obj) and not x._base.obj.flags.writeable:
            raise ValueError("destination array is read-only")
        return x.as_view()
    if is_tensor(x):
        return _torch_view(x)
    if isinstance(x, np.ndarray):
        return _numpy_view(x, writable)
    raise TypeError(f"expected a numpy array or torch tensor, got {type(x).__name__}")


def torch_fast(t) -> bool:
    """Plain strided CPU/CUDA tensor of a supported dtype (the copy_ fast path)."""
    return (torch is not None and type(t) is torch.Tensor and t.layout == torch.strided
            and t.dtype in _TORCH_DTYPES and t.device.type in ("cpu", "cuda"))


def torch_parts(t):
    """(endpoint, shape, strides, storage offset, is_cuda) without building Python Access objects."""
    st = t.untyped_storage()
    dev = t.device
    if dev.type == "cuda":
        ep = _core.Endpoint("cuda", dev.index or 0, st.data_ptr(), st.nbytes(), _TORCH_DTYPES[t.dtype], False)
        return ep, list(t.shape), list(t.stride()), t.storage_offset(), True
    ep = _core.Endpoint("host", 0, st.data_ptr(), st.nbytes(), _TORCH_DTYPES[t.dtype], t.is_pinned())
    return ep, list(t.shape), list(t.stride()), t.storage_offset(), False


def raw_stream(t) -> int:
    return torch._C._cuda_getCurrentRawStream(t.device.index or 0)


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
            return torch._C._cuda_getCurrentRawStream(v.device_index)
    return 0


def index_array(index) -> np.ndarray:
    """Host int64 copy of an index (a CUDA index tensor costs a device sync)."""
    if is_tensor(index):
        index = index.detach().to("cpu", torch.int64).numpy()
    return np.ascontiguousarray(index, dtype=np.int64).reshape(-1)
