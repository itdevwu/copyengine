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
from ._tensor import raw_stream, same_bits_dtypes, torch_parts

_COPY = torch.ops.aten.copy_.default
_PASS_THROUGH = {torch.ops.aten.is_pinned.default}

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


def decline_reason(dst, src):
    """None if ce may run ``dst.copy_(src)`` (it gives torch's exact result), else why not. Mirrors the shim."""
    if not (isinstance(dst, torch.Tensor) and isinstance(src, torch.Tensor) and _usable(dst) and _usable(src)):
        return "layout, device or dtype copyengine does not handle"
    if not same_bits_dtypes(dst.dtype, src.dtype):
        return "dtype pair where torch's result differs"
    if torch.is_grad_enabled() and (dst.requires_grad or src.requires_grad):
        return "autograd"
    return None


_REFUSALS = {"UnsafeCopy": "overlapping source and destination", "NeedsMaterialize": "view copyengine cannot describe",
             "Unsupported": "unsupported by copyengine"}


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
    empty = isinstance(dst, torch.Tensor) and dst.numel() == 0
    why = None if empty else decline_reason(dst, src)
    if why is None and not empty:
        dep, ds, dst_, doff, dcuda = torch_parts(dst)
        sep, ss, sst, soff, scuda = torch_parts(src)
        stream = raw_stream(dst if dcuda else src) if (dcuda or scuda) else 0
        try:
            # In an async scope the host never waits: torch's copy runs non_blocking too.
            nb = non_blocking or engine._async_depth > 0
            engine._e.copy_strided(sep, dep, ss, sst, soff, ds, dst_, doff, _core.Elementwise(), stream,
                                   (native_copy, dst, src, nb), host_async=non_blocking)
            return dst
        except _core.CopyEngineError as e:  # torch decides (and raises) as it would
            why = _REFUSALS.get(type(e).__name__, "refused by copyengine")
    if why is not None:
        engine._e.decline(why)
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
        if _core._framework_copy_running():  # copyengine runs torch's own copy (a native): let it through
            return func(*args, **kwargs)
        if func is _COPY:
            return intercepted_copy(self.engine, *args, **kwargs)
        writes = _writes(func)
        if writes is not None and self.engine._e.queued:
            _fence(self.engine, func, args, kwargs, writes)
        return func(*args, **kwargs)
