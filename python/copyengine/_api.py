"""User-facing engine API."""

from __future__ import annotations

import contextlib
import json
import threading
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from . import _core
from ._tensor import (
    Lazy,
    View,
    lazy,
    broadcast_to,
    current_stream,
    flat_endpoint,
    index_array,
    is_tensor,
    raw_stream,
    torch,
    torch_fast,
    torch_parts,
    view,
)

Elementwise = _core.Elementwise
_IDENTITY = _core.Elementwise()


class Engine:
    """A lazy copy queue.

    Copies issued inside ``with engine.scope():`` are queued; at scope exit they
    are planned together (dependency waves, cross-copy merging, 2-D recovery),
    issued with the best measured strategy on the current stream. Outside a scope
    every copy is issued immediately. Like torch, device-to-device copies are not
    waited for; copies touching host memory are complete when the call returns.

    Buffers touched inside a scope must not be read or written by other code
    until the scope exits (or :meth:`barrier` is called).
    """

    def __init__(self, *, staging_budget_mb: int = 256, tune: bool = True, tune_trials: int = 3,
                 tune_topk: int = 3, allow_async: bool = False, host_threads: int = 0,
                 disable: bool = False):
        o = _core.EngineOptions()
        o.staging_budget = int(staging_budget_mb) << 20
        o.tune = tune
        o.tune_trials = tune_trials
        o.tune_topk = tune_topk
        o.allow_async = allow_async
        o.host_threads = host_threads
        o.disable = disable
        self._opts = _core.EngineOptions.from_env(o)  # CE_* environment variables win
        self._e = _core.Engine(self._opts)
        self._held: list[Any] = []
        self._lock = threading.RLock()

    # ---- scope ----------------------------------------------------------------

    @contextlib.contextmanager
    def scope(self) -> Iterator["Engine"]:
        with self._lock:
            self._e.begin_scope()
        try:
            yield self
        finally:
            with self._lock:
                self._e.end_scope()
                if self._e.depth == 0 and not self._opts.allow_async:
                    self._held.clear()

    def barrier(self) -> None:
        """Issue everything queued and wait for it."""
        with self._lock:
            self._e.flush()
            self._e.synchronize()
            if self._e.depth == 0:
                self._held.clear()

    def synchronize(self) -> None:
        with self._lock:
            self._e.synchronize()
            if self._e.depth == 0 and self._e.queued == 0:
                self._held.clear()

    def _after(self, *objs) -> None:
        if self._e.depth > 0 or self._opts.allow_async:
            self._held.extend(objs)

    # ---- copies -----------------------------------------------------------------

    def copy_(self, dst, src, *, fn: Optional[Elementwise] = None, ctx: str = "") -> Any:
        """``dst[...] = fn(src)`` for any strided views; ``src`` broadcasts to ``dst``."""
        if not ctx and torch_fast(dst) and torch_fast(src):
            dep, ds, dst_, doff, dcuda = torch_parts(dst)
            sep, ss, sst, soff, scuda = torch_parts(src)
            stream = raw_stream(dst if dcuda else src) if (dcuda or scuda) else 0
            with self._lock:
                self._e.copy_strided(sep, dep, ss, sst, soff, ds, dst_, doff, fn or _IDENTITY, stream)
                self._after(src, dst)
            return dst
        d, s = view(dst, writable=True), view(src)
        sacc = broadcast_to(s.access, list(d.access.shape))
        stmt = _core.Stmt(d.access, sacc)
        with self._lock:
            self._e.copy_stmt(s.endpoint, d.endpoint, stmt, fn or _IDENTITY, ctx=ctx, stream=current_stream(d, s))
            self._after(s.obj, d.obj)
        return dst

    def segments(self, src, dst, src_off, dst_off, length, *, fn: Optional[Elementwise] = None,
                 valid_src: Optional[tuple] = None, ctx: str = "") -> None:
        """Copy runs ``dst.flat[d:d+n] = fn(src.flat[s:s+n])`` (element offsets).

        ``valid_src=(starts, lengths)`` marks the only source ranges holding data;
        the rest is skipped.
        """
        sep, sbase, sv = flat_endpoint(src)
        dep, dbase, dv = flat_endpoint(dst, writable=True)
        so = np.asarray(src_off, dtype=np.int64).reshape(-1) + sbase
        do = np.asarray(dst_off, dtype=np.int64).reshape(-1) + dbase
        vs = vl = None
        if valid_src is not None:
            vs = np.asarray(valid_src[0], dtype=np.int64).reshape(-1) + sbase
            vl = np.asarray(valid_src[1], dtype=np.int64).reshape(-1)
        with self._lock:
            self._e.copy_runs(sep, dep, so, do, length, fn or _IDENTITY, vs, vl, ctx=ctx,
                              stream=current_stream(dv, sv))
            self._after(sv.obj, dv.obj)

    def indices(self, src, dst, src_idx, dst_idx, *, fn: Optional[Elementwise] = None, ctx: str = "") -> None:
        """Element-wise ``dst.flat[dst_idx] = fn(src.flat[src_idx])``; structure is recovered."""
        sep, sbase, sv = flat_endpoint(src)
        dep, dbase, dv = flat_endpoint(dst, writable=True)
        si = index_array(src_idx) + sbase
        di = index_array(dst_idx) + dbase
        with self._lock:
            self._e.copy_indices(sep, dep, si, di, fn or _IDENTITY, ctx=ctx, stream=current_stream(dv, sv))
            self._after(sv.obj, dv.obj)

    def gather(self, src, index, dim: int = 0, *, out=None, fn: Optional[Elementwise] = None,
               dtype=None, ctx: str = ""):
        """``out = fn(src.index_select(dim, index))``; allocates ``out`` like ``src`` if omitted."""
        s = view(src)
        sacc = s.access.index_select(dim, index_array(index))
        if out is None:
            shape = list(sacc.shape)
            if is_tensor(src):
                out = torch.empty(shape, dtype=dtype or src.dtype, device=src.device)
            else:
                out = np.empty(shape, dtype=dtype or src.dtype)
        d = view(out, writable=True)
        stmt = _core.Stmt(d.access, sacc)
        with self._lock:
            self._e.copy_stmt(s.endpoint, d.endpoint, stmt, fn or _IDENTITY, ctx=ctx, stream=current_stream(d, s))
            self._after(s.obj, d.obj)
        return out

    def scatter_(self, dst, index, src, dim: int = 0, *, fn: Optional[Elementwise] = None, ctx: str = ""):
        """``dst.index_copy_(dim, index, fn(src))``. Duplicate indices raise UnsafeCopy."""
        d, s = view(dst, writable=True), view(src)
        dacc = d.access.index_select(dim, index_array(index))
        stmt = _core.Stmt(dacc, broadcast_to(s.access, list(dacc.shape)))
        with self._lock:
            self._e.copy_stmt(s.endpoint, d.endpoint, stmt, fn or _IDENTITY, ctx=ctx, stream=current_stream(d, s))
            self._after(s.obj, d.obj)
        return dst

    # ---- plans, tuning, stats -------------------------------------------------------

    def freeze(self) -> "FrozenPlan":
        """Replayable copy of the last flush (no planning, no checks on replay)."""
        with self._lock:
            return FrozenPlan(self, self._e.freeze(), list(self._held))

    def stats(self) -> dict:
        return json.loads(self._e.stats_json())

    def dump_profile(self) -> str:
        return self._e.dump_profile()

    def load_profile(self, profile: str) -> None:
        self._e.load_profile(profile)

    def save_profile(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.dump_profile())

    def warmup(self, fn, iters: int = 4) -> None:
        """Run ``fn(engine)`` a few times so every bucket it hits gets tuned."""
        for _ in range(iters):
            fn(self)
            self.barrier()

    @property
    def options(self) -> _core.EngineOptions:
        return self._opts


class FrozenPlan:
    """Replays a flushed plan. Buffers, layouts and index tables must not change."""

    def __init__(self, engine: Engine, frozen, keep: Sequence[Any]):
        self._engine, self._frozen, self._keep = engine, frozen, list(keep)

    def run(self, stream: Optional[int] = None) -> None:
        if stream is None:
            stream = 0
            if torch is not None and torch.cuda.is_available():
                stream = torch.cuda.current_stream().cuda_stream
        self._frozen.run(stream)


# ---- module-level default engine ------------------------------------------------------

_default: Optional[Engine] = None
_default_lock = threading.Lock()


def default_engine() -> Engine:
    global _default
    with _default_lock:
        if _default is None:
            _default = Engine()
        return _default


def scope():
    return default_engine().scope()


def copy_(dst, src, **kw):
    return default_engine().copy_(dst, src, **kw)


def segments(src, dst, src_off, dst_off, length, **kw):
    return default_engine().segments(src, dst, src_off, dst_off, length, **kw)


def indices(src, dst, src_idx, dst_idx, **kw):
    return default_engine().indices(src, dst, src_idx, dst_idx, **kw)


def gather(src, index, dim: int = 0, **kw):
    return default_engine().gather(src, index, dim, **kw)


def scatter_(dst, index, src, dim: int = 0, **kw):
    return default_engine().scatter_(dst, index, src, dim, **kw)


def barrier():
    default_engine().barrier()


def stats() -> dict:
    return default_engine().stats()
