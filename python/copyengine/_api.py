"""User-facing engine API."""

from __future__ import annotations

import collections
import json
import statistics
import sys
import threading
import time
import warnings
from typing import Any, Optional, Sequence

import numpy as np

from . import _core, _torch_ops, _shim  # the shim lets copyengine:: ops through: load them first
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
_Tensor = torch.Tensor if torch is not None else None
_STRIDED = torch.strided if torch is not None else None

if torch is not None:
    _FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _same_bits(dst, src, fn) -> bool:
    """Whether the framework's own copy gives the same bits as ce.

    The engine then measures that copy as the baseline and runs it when ce is not
    significantly faster.
    """
    if fn is not None and fn is not _IDENTITY:
        return False
    if is_tensor(dst) and is_tensor(src):
        return dst.dtype == src.dtype or (dst.dtype in _FLOATS and src.dtype != torch.bool)
    return isinstance(dst, np.ndarray) and isinstance(src, np.ndarray) and dst.dtype == src.dtype


def native_copy(dst, src) -> None:
    """The framework's own copy: the baseline ce must beat."""
    if is_tensor(dst):
        dst.copy_(src, non_blocking=True)
    else:
        np.copyto(dst, src)


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

    def __init__(self, *, staging_budget_mb: int = 256, explore: bool = True, tune_trials: int = 5,
                 tune_topk: int = 3, ab_pairs: int = 20, ab_max_pairs: int = 50, online_every: int = 64,
                 allow_async: bool = False,
                 host_threads: int = 0, disable: bool = False, prior_only: bool = False,
                 _intercept: Optional[str] = None):
        """``explore=False`` uses only a loaded profile (uncovered buckets take the
        baseline). ``prior_only`` is for tests: always the prior's first strategy,
        no measurement and no no-regression guarantee."""
        o = _core.EngineOptions()
        o.staging_budget = int(staging_budget_mb) << 20
        o.explore = explore
        o.prior_only = prior_only
        o.tune_trials = tune_trials
        o.tune_topk = tune_topk
        o.ab_pairs = ab_pairs
        o.ab_max_pairs = ab_max_pairs
        o.online_every = online_every
        o.allow_async = allow_async
        o.host_threads = host_threads
        o.disable = disable
        self._opts = _core.EngineOptions.from_env(o)  # CE_* environment variables win
        self._e = _core.Engine(self._opts)
        self._handle = self._e.handle
        self._native = native_copy  # replaceable in tests
        self._held: list[Any] = []         # inside a scope: buffers of queued copies
        self._inflight: collections.deque = collections.deque()  # allow_async: (token, buffers) until done
        self._lock = threading.RLock()
        self._force = _intercept  # tests: "shim" or "mode" regardless of what is installed
        self._set_depth(0)        # scope depth as seen from Python: whether queued copies need references held
        self._sites: dict = {}    # call site -> _Site (Python-mode on/off decision)

    # ---- scope ----------------------------------------------------------------

    def scope(self, _site=None) -> "_Scope":
        """Queue copies until exit, then plan and issue them together.

        Inside a scope ``Tensor.copy_`` is intercepted too, and every other op is
        ordered after the queued copies it touches. Results are always torch's.
        """
        if _site is None:
            f = sys._getframe(1)
            _site = (f.f_code, f.f_lineno)
        return _Scope(self, _site)

    def _interception(self, site):
        """'shim', 'mode' or None for one scope entry; also whether to time it."""
        if self._force is not None:
            return self._force, False
        if torch is None or self._opts.disable:
            return None, False
        if _shim.lib is not None:
            return "shim", False
        st = self._sites.get(site)
        if st is None:
            st = self._sites[site] = _Site()
        return st.choose()

    def barrier(self) -> None:
        """Issue everything queued and wait for it."""
        with self._lock:
            self._e.flush()
            self._e.synchronize()
            if self._e.depth == 0:
                self._held.clear()
                self._inflight.clear()

    def synchronize(self) -> None:
        with self._lock:
            self._e.synchronize()
            if self._e.depth == 0 and self._e.queued == 0:
                self._held.clear()
                self._inflight.clear()

    def _after(self, *objs) -> None:
        if self._depth > 0:
            self._held.extend(objs)
        elif self._opts.allow_async:
            self._retire(objs)

    def _retire(self, objs) -> None:
        """allow_async, outside a scope: keep `objs` until the work issued so far completes."""
        inflight = self._inflight
        while inflight and self._e.done(inflight[0][0]):
            inflight.popleft()
        token = self._e.mark_pending()
        if token >= 0 and objs:
            inflight.append((token, objs))

    def _set_depth(self, d: int) -> None:
        self._depth = d
        self._hold = d > 0 or self._opts.allow_async

    # ---- copies -----------------------------------------------------------------

    def copy_(self, dst, src, *, fn: Optional[Elementwise] = None, ctx: str = "") -> Any:
        """``dst[...] = fn(src)`` for any strided views; ``src`` broadcasts to ``dst``."""
        op = _torch_ops.copy_
        # Strided only: the stable ABI the op targets cannot tell it a tensor's layout.
        if (op is not None and not ctx and type(dst) is _Tensor and type(src) is _Tensor
                and dst.layout is _STRIDED and src.layout is _STRIDED):
            # Descriptors, checks and the native baseline in C++; the engine locks itself.
            try:
                if fn is None or fn is _IDENTITY:
                    op(dst, src, self._handle)
                else:
                    lo, hi = fn.clamp or (0.0, 0.0)
                    _torch_ops.copy_fn_(dst, src, self._handle, fn.scale, fn.bias, fn.round,
                                        fn.clamp is not None, lo, hi)
            except RuntimeError as e:
                err = _torch_ops.translate(e)
                if not isinstance(err, _core.Unsupported):
                    raise err from None
                # dtype / device / layout the op does not take: the general path decides
            else:
                if self._hold:
                    self._after(src, dst)
                return dst
        if not ctx and torch_fast(dst) and torch_fast(src):
            dep, ds, dst_, doff, dcuda = torch_parts(dst)
            sep, ss, sst, soff, scuda = torch_parts(src)
            stream = raw_stream(dst if dcuda else src) if (dcuda or scuda) else 0
            with self._lock:
                self._e.copy_strided(sep, dep, ss, sst, soff, ds, dst_, doff, fn or _IDENTITY, stream,
                                     (self._native, dst, src) if _same_bits(dst, src, fn) else None)
                self._after(src, dst)
            return dst
        d, s = view(dst, writable=True), view(src)
        sacc = broadcast_to(s.access, list(d.access.shape))
        stmt = _core.Stmt(d.access, sacc)
        with self._lock:
            self._e.copy_stmt(s.endpoint, d.endpoint, stmt, fn or _IDENTITY, ctx=ctx, stream=current_stream(d, s),
                              native=(self._native, dst, src) if _same_bits(dst, src, fn) else None)
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


class _Site:
    """Python-mode interception costs ~9 us per op, paid at scope entry. One
    call site alternates entries with and without it; the mode stays only if
    it wins clearly (the same rule as the engine's A/B), else the site runs
    torch natively from then on."""

    PAIRS = 10

    def __init__(self):
        self.on: list[float] = []
        self.off: list[float] = []
        self.decided: Optional[bool] = None

    def choose(self):
        if self.decided is not None:
            return ("mode" if self.decided else None), False
        use = len(self.on) <= len(self.off)
        return ("mode" if use else None), True

    def report(self, used: bool, seconds: float) -> None:
        (self.on if used else self.off).append(seconds)
        n = min(len(self.on), len(self.off))
        if n < self.PAIRS:
            return
        on, off = self.on[:n], self.off[:n]
        wins = sum(a < b for a, b in zip(on, off))
        self.decided = statistics.median(on) <= 0.95 * statistics.median(off) and wins >= _sign_test(n)


def _sign_test(n: int) -> int:
    """Smallest k with P(Binomial(n, 1/2) >= k) <= 0.05."""
    from math import comb
    tail = 0.0
    for k in range(n, -1, -1):
        tail += comb(n, k) / 2 ** n
        if tail > 0.05:
            return k + 1
    return n + 1


class _Scope:
    __slots__ = ("engine", "site", "kind", "mode", "timed", "t0")

    def __init__(self, engine: Engine, site):
        self.engine, self.site = engine, site
        self.kind = self.mode = None
        self.timed, self.t0 = False, 0.0

    def __enter__(self) -> Engine:
        e = self.engine
        with e._lock:
            e._e.begin_scope()
            e._set_depth(e._depth + 1)
        self.kind, self.timed = e._interception(self.site)
        if self.timed:
            self.t0 = time.perf_counter()
        if self.kind == "shim":
            if _shim.lib is None:
                raise RuntimeError(f"the interception shim is unavailable: {_shim.error}")
            _shim.lib.copyengine_shim_enter(e._handle)
        elif self.kind == "mode":
            from . import _mode
            self.mode = _mode.Intercept(e)
            self.mode.__enter__()
        return e

    def __exit__(self, *exc) -> None:
        e = self.engine
        try:
            if self.kind == "shim":
                _shim.lib.copyengine_shim_exit()
            elif self.mode is not None:
                self.mode.__exit__(*exc)
                self.mode = None
        finally:
            with e._lock:
                e._set_depth(e._depth - 1)
                e._e.end_scope()
                if e._depth == 0:
                    held, e._held = e._held, []
                    if e._opts.allow_async:
                        e._retire(held)
        if self.timed:
            e._sites[self.site].report(self.kind == "mode", time.perf_counter() - self.t0)


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

# One per thread: a scope queues the copies of the engine it belongs to, so a
# shared default engine would pull other threads' copies into it.
_default = threading.local()


if torch is not None and _shim.lib is None and "CE_SHIM=0" not in _shim.error:
    warnings.warn(f"copyengine: the C++ interception shim is unavailable ({_shim.error.splitlines()[0]}); "
                  "scopes intercept copies in Python instead: same results, slower", RuntimeWarning, stacklevel=2)


def default_engine() -> Engine:
    """This thread's engine (created on first use)."""
    e = getattr(_default, "engine", None)
    if e is None:
        e = _default.engine = Engine()
    return e


def scope():
    f = sys._getframe(1)
    return default_engine().scope(_site=(f.f_code, f.f_lineno))


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
