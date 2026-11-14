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
    device_index,
    flat_endpoint,
    index_array,
    is_tensor,
    raw_stream,
    torch,
    torch_dst,
    torch_fast,
    torch_parts,
    torch_replay,
    view,
)

Elementwise = _core.Elementwise
_IDENTITY = _core.Elementwise()
_Tensor = torch.Tensor if torch is not None else None
_STRIDED = torch.strided if torch is not None else None
# dtypes torch's stable ABI cannot pass to a custom op yet: these take the Python path.
_NOT_STABLE = (torch.float8_e8m0fnu, torch.float4_e2m1fn_x2) if torch is not None else ()

if torch is not None:
    _FLOATS = (torch.float16, torch.bfloat16, torch.float32, torch.float64,
               torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu)


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


# ---- torch equivalents of the explicit API (the baselines; built only when run) ------------


def _dtype_of(x):
    if isinstance(x, Lazy):
        x = x._base.obj
    return x.dtype if is_tensor(x) else None


def _lazy_payload(dst, src, fn):
    """Payload replaying ce.copy_ with lazy views as torch ops, or None."""
    if fn is not None and fn is not _IDENTITY:
        return None
    sd, dd = _dtype_of(src), _dtype_of(dst)
    if sd is None or dd is None or not (dd == sd or (dd in _FLOATS and sd != torch.bool)):
        return None
    if isinstance(src, Lazy) and any(op[0] == "slice" and (op[4] or 1) < 0 for op in src._ops):
        return None
    if isinstance(dst, Lazy) and torch_dst(dst) is None:
        return None
    return (_native_lazy, dst, src)


def _native_lazy(dst, src) -> None:
    s = torch_replay(src) if isinstance(src, Lazy) else src
    if not isinstance(dst, Lazy):
        dst.copy_(s, non_blocking=True)
        return
    kind = torch_dst(dst)
    if kind[0] == "view":
        kind[1].copy_(s, non_blocking=True)
    else:
        _, base, dim, idx = kind
        shape = list(base.shape)
        shape[dim] = len(idx)
        base.index_copy_(dim, device_index(idx, base.device), s.expand(shape).to(base.dtype))


def _native_gather(out, src, dim, idx) -> None:
    out.copy_(src.index_select(dim, device_index(idx, src.device)), non_blocking=True)


def _native_scatter(dst, src, dim, idx) -> None:
    shape = list(dst.shape)
    shape[dim] = len(idx)
    dst.index_copy_(dim, device_index(idx, dst.device), src.expand(shape).to(dst.dtype))


def _native_flat(dst, src, si, di) -> None:
    """dst.flat[di] = src.flat[si] (element indices into contiguous tensors)."""
    dst.view(-1).index_copy_(0, device_index(di, dst.device),
                             src.reshape(-1).index_select(0, device_index(si, src.device)).to(dst.dtype))


def _native_segments(dst, src, so, do, n) -> None:
    if len(n) <= 64:  # few segments: slices, as a torch user would write them
        sf, df = src.reshape(-1), dst.view(-1)
        for a, b, k in zip(so.tolist(), do.tolist(), n.tolist()):
            df[b:b + k].copy_(sf[a:a + k], non_blocking=True)
        return
    rep = np.repeat(np.arange(len(n)), n)
    within = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
    _native_flat(dst, src, so[rep] + within, do[rep] + within)


def _flat_payload(dst, src, fn):
    """Whether a flat-index torch equivalent gives ce's bits for these tensors."""
    if fn is not None and fn is not _IDENTITY or not (is_tensor(dst) and is_tensor(src)):
        return False
    return dst.dtype == src.dtype or (dst.dtype in _FLOATS and src.dtype != torch.bool)


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
                 host_threads: int = 0, disable: bool = False, prior_only: bool = False, max_sms: int = 0,
                 _intercept: Optional[str] = None):
        """``explore=False`` uses only a loaded profile (uncovered buckets take the
        baseline). ``max_sms`` bounds the SMs copy kernels may occupy per GPU (see
        ``stats()["cuda"]`` for whether it is enforced by a green context or by
        capping thread blocks). ``prior_only`` is for tests: always the prior's
        first strategy, no measurement and no no-regression guarantee."""
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
        o.max_sms = max_sms
        self._opts = _core.EngineOptions.from_env(o)  # CE_* environment variables win
        self._e = _core.Engine(self._opts)
        self._handle = self._e.handle
        self._native = native_copy  # replaceable in tests
        self._held: list[Any] = []         # inside a scope: buffers of queued copies
        self._inflight: collections.deque = collections.deque()  # allow_async: (token, buffers) until done
        self._lock = threading.RLock()
        self._force = _intercept  # tests: "shim" or "mode" regardless of what is installed
        self._copy_streams: dict = {}  # device index -> torch.cuda.Stream (scope(stream="copy"))
        self._async_depth = 0          # open scope(stream=...)s
        self._set_depth(0)        # scope depth as seen from Python: whether queued copies need references held
        self._sites: dict = {}    # outermost scope call site -> _Site (intercept or not)
        self._scope_kind = None   # the outermost open scope's interception (inner scopes share it)

    # ---- scope ----------------------------------------------------------------

    def scope(self, stream=None, _site=None) -> "_Scope":
        """Queue copies until exit, then plan and issue them together.

        Inside a scope ``Tensor.copy_`` is intercepted too, and every other op is
        ordered after the queued copies it touches. Results are always torch's.

        ``stream="copy"`` (this engine's own stream for the current device) or a
        ``torch.cuda.Stream`` runs the scope's work on that stream, after what is
        already queued on the current one, and never makes the host wait:
        ``with engine.scope(stream="copy") as h:`` gives a :class:`Pending`, and
        ``h.wait()`` orders the current stream after the copies.
        """
        if _site is None:
            f = sys._getframe(1)
            _site = (f.f_code, f.f_lineno)
        return _Scope(self, _site, stream)

    def copy_stream(self, device=None):
        """This engine's copy stream for `device` (created on first use)."""
        dev = torch.device("cuda", torch.cuda.current_device() if device is None else torch.device(device).index or 0)
        st = self._copy_streams.get(dev.index)
        if st is None:
            if torch.cuda.is_current_stream_capturing():
                raise _core.Unsupported("the copy stream is created on first use; use scope(stream='copy') once "
                                        "before capturing a CUDA graph")
            st = self._copy_streams[dev.index] = torch.cuda.Stream(dev)
        return st

    def _interception(self, site):
        """'shim', 'mode' or None for one scope entry; also whether to time it."""
        if self._force is not None:
            return self._force, False
        if torch is None or self._opts.disable:
            return None, False
        on = "shim" if _shim.lib is not None else "mode"
        if self._depth > 1:  # an inner scope: the outermost one decided (this one is already counted in depth)
            return self._scope_kind, False
        st = self._sites.get(site)
        if st is None:
            st = self._sites[site] = _Site(on)
        kind, timed = st.choose()
        if timed and torch is not None and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return (kind if st.decided is not None else on), False  # nothing runs during a capture: no samples
        return kind, timed

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
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return  # recorded into a CUDA graph: its owner keeps the tensors alive
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
                and dst.layout is _STRIDED and src.layout is _STRIDED
                and dst.dtype not in _NOT_STABLE and src.dtype not in _NOT_STABLE):
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
            if isinstance(dst, Lazy) or isinstance(src, Lazy):
                native = _lazy_payload(dst, src, fn)
            else:
                native = (self._native, dst, src) if _same_bits(dst, src, fn) else None
            self._e.copy_stmt(s.endpoint, d.endpoint, stmt, fn or _IDENTITY, ctx=ctx, stream=current_stream(d, s),
                              native=native)
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
        n = np.asarray(length, dtype=np.int64).reshape(-1)
        native = None
        if valid_src is None and _flat_payload(dst, src, fn):
            native = (_native_segments, dst, src, so - sbase, do - dbase, n)
        with self._lock:
            self._e.copy_runs(sep, dep, so, do, length, fn or _IDENTITY, vs, vl, ctx=ctx,
                              stream=current_stream(dv, sv), native=native)
            self._after(sv.obj, dv.obj)

    def indices(self, src, dst, src_idx, dst_idx, *, fn: Optional[Elementwise] = None, ctx: str = "") -> None:
        """Element-wise ``dst.flat[dst_idx] = fn(src.flat[src_idx])``; structure is recovered."""
        sep, sbase, sv = flat_endpoint(src)
        dep, dbase, dv = flat_endpoint(dst, writable=True)
        si = index_array(src_idx) + sbase
        di = index_array(dst_idx) + dbase
        native = (_native_flat, dst, src, si - sbase, di - dbase) if _flat_payload(dst, src, fn) else None
        with self._lock:
            self._e.copy_indices(sep, dep, si, di, fn or _IDENTITY, ctx=ctx, stream=current_stream(dv, sv),
                                 native=native)
            self._after(sv.obj, dv.obj)

    def gather(self, src, index, dim: int = 0, *, out=None, fn: Optional[Elementwise] = None,
               dtype=None, ctx: str = ""):
        """``out = fn(src.index_select(dim, index))``; allocates ``out`` like ``src`` if omitted."""
        s = view(src)
        idx = index_array(index)
        sacc = s.access.index_select(dim, idx)
        if out is None:
            shape = list(sacc.shape)
            if is_tensor(src):
                out = torch.empty(shape, dtype=dtype or src.dtype, device=src.device)
            else:
                out = np.empty(shape, dtype=dtype or src.dtype)
        d = view(out, writable=True)
        stmt = _core.Stmt(d.access, sacc)
        native = (_native_gather, out, src, dim, idx) if _flat_payload(out, src, fn) else None
        with self._lock:
            self._e.copy_stmt(s.endpoint, d.endpoint, stmt, fn or _IDENTITY, ctx=ctx, stream=current_stream(d, s),
                              native=native)
            self._after(s.obj, d.obj)
        return out

    def scatter_(self, dst, index, src, dim: int = 0, *, fn: Optional[Elementwise] = None, ctx: str = ""):
        """``dst.index_copy_(dim, index, fn(src))``. Duplicate indices raise UnsafeCopy."""
        d, s = view(dst, writable=True), view(src)
        idx = index_array(index)
        dacc = d.access.index_select(dim, idx)
        stmt = _core.Stmt(dacc, broadcast_to(s.access, list(dacc.shape)))
        native = (_native_scatter, dst, src, dim, idx) if _flat_payload(dst, src, fn) else None
        with self._lock:
            self._e.copy_stmt(s.endpoint, d.endpoint, stmt, fn or _IDENTITY, ctx=ctx, stream=current_stream(d, s),
                              native=native)
            self._after(s.obj, d.obj)
        return dst

    # ---- plans, tuning, stats -------------------------------------------------------

    def freeze(self) -> "FrozenPlan":
        """Replayable copy of the last flush (no planning, no checks on replay)."""
        with self._lock:
            return FrozenPlan(self, self._e.freeze(), list(self._held))

    def stats(self) -> dict:
        out = json.loads(self._e.stats_json())
        # Per outermost scope call site: whether it intercepts (None: still deciding).
        out["scopes"] = [{"site": f"{code.co_filename}:{line}", "intercept": st.decided}
                         for (code, line), st in self._sites.items()]
        return out

    def dump_profile(self) -> str:
        return self._e.dump_profile()

    def load_profile(self, profile: str) -> None:
        self._e.load_profile(profile)

    def save_profile(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self.dump_profile())

    def warmup(self, fn, iters: Optional[int] = None, max_iters: int = 500) -> int:
        """Run ``fn(engine)`` until every copy pattern it hits has locked its
        decision (or `iters` times exactly), and return how many runs it took.

        Run it with representative inputs before serving; then
        ``save_profile`` lets later processes skip it.
        """
        n = 0
        while True:
            fn(self)
            self.barrier()
            n += 1
            if iters is not None:
                if n >= iters:
                    return n
                continue
            buckets = json.loads(self.dump_profile())["buckets"].values()
            if all(b["locked"] for b in buckets) or n >= max_iters:
                return n

    @property
    def options(self) -> _core.EngineOptions:
        return self._opts


class _Site:
    """Whether one outermost scope call site intercepts copies.

    Interception costs something even when ce wins every copy it plans: each op
    in the scope is dispatched through it, and copies left to torch are queued
    and replayed. The baseline is the scope not intercepting at all. A site
    first runs intercepted until the engine is settled there (``SETTLE``
    entries in a row without tuning runs), then alternates entries with and
    without interception, each timed end to end (host time, and GPU time from
    the scope's entry to its last work, read later without waiting).
    Interception stays only if it wins clearly (the engine's A/B rule); ties
    keep torch. Entries that ran tuning measurements are not samples.
    """

    PAIRS = 10
    SETTLE = 3
    MAX_WARM = 256  # an engine that never settles: compare anyway

    def __init__(self, on_kind: str = "mode"):
        self.on_kind = on_kind
        self.on: list[float] = []
        self.off: list[float] = []
        self.decided: Optional[bool] = None
        self.settled = 0
        self.warm = 0
        self.waiting: list = []  # (used, host seconds, start event, end event) not yet complete

    def choose(self):
        """(kind, timed) for the next entry: kind is the interception or None."""
        if self.decided is not None:
            return (self.on_kind if self.decided else None), False
        self._poll()
        if self.settled < self.SETTLE and self.warm < self.MAX_WARM:
            return self.on_kind, "warm"
        if self.decided is not None:
            return (self.on_kind if self.decided else None), False
        use = len(self.on) + sum(w[0] for w in self.waiting) <= len(self.off) + sum(not w[0] for w in self.waiting)
        return (self.on_kind if use else None), True

    def warmed(self, quiet: bool) -> None:
        """A warm-up entry ended; `quiet`: it ran no tuning measurements."""
        self.warm += 1
        self.settled = self.settled + 1 if quiet else 0

    def sample(self, used: bool, seconds: float, start=None, end=None) -> None:
        if start is None:
            self.report(used, seconds)
        else:
            self.waiting.append((used, seconds, start, end))

    def _poll(self) -> None:
        while self.waiting and self.waiting[0][3].query():
            used, host, start, end = self.waiting.pop(0)
            self.report(used, max(host, start.elapsed_time(end) / 1e3))

    def report(self, used: bool, seconds: float) -> None:
        (self.on if used else self.off).append(seconds)
        n = min(len(self.on), len(self.off))
        if n < self.PAIRS:
            return
        on, off = self.on[:n], self.off[:n]
        wins = sum(a < b for a, b in zip(on, off))
        self.decided = statistics.median(on) <= 0.95 * statistics.median(off) and wins >= _sign_test(n)
        self.waiting.clear()


def _sign_test(n: int) -> int:
    """Smallest k with P(Binomial(n, 1/2) >= k) <= 0.05."""
    from math import comb
    tail = 0.0
    for k in range(n, -1, -1):
        tail += comb(n, k) / 2 ** n
        if tail > 0.05:
            return k + 1
    return n + 1


class Pending:
    """The copies of one ``scope(stream=...)``, in flight on that stream.

    ``wait()`` makes a stream (default: the current one) wait for them on the
    GPU; ``synchronize()`` makes the host wait; ``done()`` asks without
    waiting. ``event`` is the ``torch.cuda.Event`` behind all three. The
    copies' buffers stay alive until they are done. Inside a CUDA graph
    capture, ``wait()`` is the join of the stream the scope forked.
    """

    __slots__ = ("event", "stream", "_keep")

    def __init__(self, event, stream, keep):
        self.event, self.stream, self._keep = event, stream, keep

    def wait(self, stream=None) -> None:
        (stream if stream is not None else torch.cuda.current_stream()).wait_event(self.event)

    def synchronize(self) -> None:
        self.event.synchronize()
        self._keep = None

    def done(self) -> bool:
        if self.event.query():
            self._keep = None
            return True
        return False


class _Scope:
    __slots__ = ("engine", "site", "kind", "mode", "timed", "t0", "stream", "stream_ctx", "outer_held", "pending",
                 "runs0", "ev0")

    def __init__(self, engine: Engine, site, stream=None):
        self.engine, self.site = engine, site
        self.kind = self.mode = None
        self.timed, self.t0 = False, 0.0
        self.runs0, self.ev0 = 0, None
        self.stream, self.stream_ctx, self.outer_held, self.pending = stream, None, None, None

    def __enter__(self):
        e = self.engine
        st = self.stream
        if st is not None:
            if torch is None or not torch.cuda.is_available():
                raise ValueError("scope(stream=...) needs CUDA")
            if isinstance(st, str):
                if st != "copy":
                    raise ValueError(f"stream must be 'copy' or a torch.cuda.Stream, got {st!r}")
                st = self.stream = e.copy_stream()
            st.wait_stream(torch.cuda.current_stream())  # sees what was queued before the scope
            self.stream_ctx = torch.cuda.stream(st)
            self.stream_ctx.__enter__()
        with e._lock:
            e._e.begin_scope(host_async=st is not None)
            e._set_depth(e._depth + 1)
            e._async_depth += st is not None
            if st is not None:
                self.outer_held, e._held = e._held, []
        self.kind, self.timed = e._interception(self.site)
        if e._depth == 1:
            e._scope_kind = self.kind
        if self.timed:
            self.runs0 = e._e.tuning_runs
            if self.timed is True and torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
                self.ev0 = torch.cuda.Event(enable_timing=True)
                self.ev0.record()
            self.t0 = time.perf_counter()
        if self.kind == "shim":
            if _shim.lib is None:
                raise RuntimeError(f"the interception shim is unavailable: {_shim.error}")
            _shim.lib.copyengine_shim_enter(e._handle)
        elif self.kind == "mode":
            from . import _mode
            self.mode = _mode.Intercept(e)
            self.mode.__enter__()
        return e if self.stream is None else _PendingSlot(self)

    def __exit__(self, *exc) -> None:
        e = self.engine
        try:
            if self.kind == "shim":
                _shim.lib.copyengine_shim_exit()
            elif self.mode is not None:
                self.mode.__exit__(*exc)
                self.mode = None
        finally:
            try:
                with e._lock:
                    e._set_depth(e._depth - 1)
                    e._async_depth -= self.stream is not None
                    e._e.end_scope()
                    if self.stream is not None:
                        e._e.flush()  # issue now, also inside an outer scope: the event follows the copies
                        held, e._held = e._held, self.outer_held
                        event = torch.cuda.Event()
                        event.record(self.stream)
                        e._retire(held)
                        self.pending = Pending(event, self.stream, held)
                    elif e._depth == 0:
                        held, e._held = e._held, []
                        if e._opts.allow_async:
                            e._retire(held)
            finally:
                if self.stream_ctx is not None:
                    self.stream_ctx.__exit__(*exc)
                    self.stream_ctx = None
        if self.timed:
            host = time.perf_counter() - self.t0
            site = e._sites[self.site]
            quiet = e._e.tuning_runs == self.runs0
            if self.timed == "warm":
                site.warmed(quiet)
            elif quiet:  # an entry that ran tuning measurements is not a sample
                end = None
                if self.ev0 is not None:
                    end = torch.cuda.Event(enable_timing=True)
                    end.record(self.stream if self.stream is not None else torch.cuda.current_stream())
                site.sample(self.kind is not None, host, self.ev0, end)
            self.ev0 = None


class _PendingSlot:
    """What ``with scope(stream=...) as h`` binds: the scope's Pending, once
    the scope has exited (its copies are issued then)."""

    __slots__ = ("_scope",)

    def __init__(self, scope):
        self._scope = scope

    def _get(self) -> Pending:
        p = self._scope.pending
        if p is None:
            raise RuntimeError("the scope is still open: its copies are issued when it exits")
        return p

    @property
    def event(self):
        return self._get().event

    @property
    def stream(self):
        return self._scope.stream

    def wait(self, stream=None) -> None:
        self._get().wait(stream)

    def synchronize(self) -> None:
        self._get().synchronize()

    def done(self) -> bool:
        return self._get().done()


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


def scope(stream=None):
    f = sys._getframe(1)
    return default_engine().scope(stream, _site=(f.f_code, f.f_lineno))


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
