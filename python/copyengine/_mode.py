"""Scope interception in Python: the fallback when the C++ shim is unavailable.

Same semantics as the shim (src/torch/shim.cpp), only slower (~9 us per op):
inside a scope, ``Tensor.copy_`` is queued in the engine when ce can give
torch's exact result, and every other op is ordered after the queued copies
whose memory it touches. Views and metadata queries pass through untouched.
"""

from __future__ import annotations

import torch
from torch.utils._python_dispatch import TorchDispatchMode, _disable_current_modes

from . import _core
from ._tensor import torch_parts, raw_stream

_COPY = torch.ops.aten.copy_.default
_PASS_THROUGH = {torch.ops.aten.is_pinned.default}

_FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64,
           torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu)
_DTYPES = {torch.float64, torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int16,
           torch.int8, torch.uint8, torch.bool, torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu,
           torch.float4_e2m1fn_x2}

_schemas: dict = {}  # op -> None (pass through) or indices of written arguments


def _writes(func):
    """None if `func` touches no memory (views, metadata, copyengine's own ops), else its written args."""
    try:
        return _schemas[func]
    except KeyError:
        pass
    s = func._schema
    writes = tuple(i for i, a in enumerate(s.arguments) if a.alias_info is not None and a.alias_info.is_write)
    view = not writes and s.returns and all(r.alias_info is not None for r in s.returns)
    info = None if (view or func in _PASS_THROUGH or s.name.startswith("copyengine::")) else writes
    _schemas[func] = info
    return info


def _usable(t) -> bool:
    return (t.layout == torch.strided and t.dtype in _DTYPES and t.device.type in ("cpu", "cuda")
            and not t.is_conj() and not t.is_neg())


def eligible(dst, src) -> bool:
    """Whether ce may run ``dst.copy_(src)``: it must give torch's exact result. Mirrors the shim."""
    if not (isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor) and _usable(dst) and _usable(src)):
        return False
    if dst.numel() == 0 or not (dst.dtype == src.dtype or (dst.dtype in _FLOATS and src.dtype != torch.bool)):
        return False
    return not (torch.is_grad_enabled() and (dst.requires_grad or src.requires_grad))


def _range(t):
    st = t.untyped_storage()
    lo = st.data_ptr()
    return ("cuda" if t.is_cuda else "host", t.device.index or 0, lo, lo + st.nbytes())


def _tensors(x, out):
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            if isinstance(y, torch.Tensor):
                out.append(y)


def _fence(engine, func, args, kwargs, writes) -> None:
    reads_, writes_ = [], []
    names = [a.name for a in func._schema.arguments]
    for i, name in enumerate(names):
        v = args[i] if i < len(args) else kwargs.get(name)
        ts = []
        _tensors(v, ts)
        for t in ts:
            if t.device.type in ("cpu", "cuda") and t.layout == torch.strided and t.untyped_storage().data_ptr():
                (writes_ if i in writes else reads_).append(_range(t))
    engine._e.fence(reads_, writes_)


def native_copy(dst, src, non_blocking) -> None:
    """torch's own copy_, never intercepted (the baseline)."""
    with _disable_current_modes():
        dst.copy_(src, non_blocking=non_blocking)


def intercepted_copy(engine, dst, src, non_blocking=False):
    if eligible(dst, src):
        dep, ds, dst_, doff, dcuda = torch_parts(dst)
        sep, ss, sst, soff, scuda = torch_parts(src)
        stream = raw_stream(dst if dcuda else src) if (dcuda or scuda) else 0
        try:
            engine._e.copy_strided(sep, dep, ss, sst, soff, ds, dst_, doff, _core.Elementwise(), stream,
                                   (native_copy, dst, src, non_blocking), host_async=non_blocking)
            return dst
        except _core.CopyEngineError:
            pass  # torch decides (and raises) as it would
    if engine._e.queued:
        engine._e.fence([_range(src)] if isinstance(src, torch.Tensor) else [], [_range(dst)])
    native_copy(dst, src, non_blocking)
    return dst


class Intercept(TorchDispatchMode):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is _COPY:
            return intercepted_copy(self.engine, *args, **kwargs)
        writes = _writes(func)
        if writes is not None and self.engine._e.queued:
            _fence(self.engine, func, args, kwargs, writes)
        return func(*args, **kwargs)
