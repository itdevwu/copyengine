"""Step plans: plan a step of copies once, replay it with new index tensors.

``plan = eng.plan(step)``; ``plan(*args)`` runs ``step(*args)``. The first
calls for a given argument layout run the step with interception on until the
engine has settled there; then its copies are frozen and later calls replay
them without running the step's Python or the planner. Index arguments are
parameters (a replay reads the new ones): CUDA int64 tensors directly, host
indices (numpy arrays, lists of ints, CPU integer tensors) through a pinned
buffer the plan refills each call, when the step passes them straight to a ce
index API. Every other tensor the copies touch must be the same buffer (else
the step is traced again): an argument, or a tensor the step reaches by name
(a closure variable, a global, an attribute of a bound method's object; one
level of lists), which is checked before each replay to be the same view
(address, dtype, shape, strides), else the step is traced again. A replay is
kept only if it beats running the step as plain code (paired, end to end).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any, Callable

import numpy as np

from . import _core

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

_tracing = threading.local()  # engine the module-level functions use during a trace


def tracing_engine():
    return getattr(_tracing, "engine", None)


def param_for(obj):
    """While a step traces: the pinned buffer standing for host index argument `obj`."""
    params = getattr(_tracing, "params", None)
    if not params:
        return None
    buf = params.get(id(obj))
    if buf is not None:
        _tracing.used.add(id(obj))
    return buf


def overlap_checked_per_call() -> bool:
    """While a step plan traces with host index parameters: a copy whose sides may share
    bytes only through those indices is traced as one pass, and the plan checks the
    values exactly before each replay (Frozen.host_overlap); where they do overlap,
    that call runs the step plainly (two passes, decided with the values)."""
    if getattr(_tracing, "params", None) is None:
        return False
    _tracing.overlaps += 1
    return True


def note_values() -> None:
    """While a step traces: its copies took index values (not parameters)."""
    if getattr(_tracing, "params", None) is not None:
        _tracing.values += 1


def _dead_temporaries(allocs):
    """GPU allocations the step's copies used that no tensor holds now (the step
    allocated and dropped them: torch's allocator has them free), as
    (device, lo, hi); None if any is anything else (host memory, alive)."""
    if not allocs or torch is None or not torch.cuda.is_available():
        return None
    if any(kind != 1 for kind, _, _, _ in allocs):
        return None
    try:
        segments = torch.cuda.memory._snapshot()["segments"]
    except Exception:  # pragma: no cover - allocator without snapshots
        return None
    free = {}
    for seg in segments:
        for b in seg["blocks"]:
            free.setdefault(seg["device"], []).append((b["address"], b["address"] + b["size"], b["state"]))
    dead = []
    for _, index, lo, hi in allocs:
        blocks = free.get(index, [])
        inside_free = any(a <= lo and hi <= b and st == "inactive" for a, b, st in blocks)
        overlaps_live = any(a < hi and lo < b and st != "inactive" for a, b, st in blocks)
        if not inside_free or overlaps_live:
            return None
        dead.append((index, lo, hi))
    return dead


def _is_host_index(x) -> bool:
    if isinstance(x, np.ndarray):
        return x.ndim <= 1 and x.dtype.kind in "iu"
    if isinstance(x, (list, tuple)):
        return bool(x) and all(type(v) is int for v in x)
    return _is_tensor(x) and not x.is_cuda and x.dim() <= 1 and not x.is_floating_point() and x.dtype is not torch.bool


def _values(x) -> bytes:
    if _is_tensor(x):
        return x.detach().reshape(-1).to(torch.int64).numpy().tobytes()
    return np.asarray(x, dtype=np.int64).reshape(-1).tobytes()


def _fill(buf, x) -> None:
    if _is_tensor(x):
        buf.copy_(x.detach().reshape(-1))
    else:
        buf.numpy()[:] = np.asarray(x).reshape(-1)


def _buffer(x):
    n = x.numel() if _is_tensor(x) else int(np.asarray(x).size)  # 0-d arrays too
    pin = torch.cuda.is_available()
    buf = torch.empty(n, dtype=torch.int64, pin_memory=pin)
    _fill(buf, x)
    return buf


def _flatten(args, kw) -> list:
    leaves = []
    for a in list(args) + [kw[k] for k in sorted(kw)]:
        if isinstance(a, (list, tuple)) and a and all(_is_tensor(x) for x in a):
            leaves.extend(a)  # TensorList: one level; the container itself is a leaf too
            leaves.append(_TensorList(a))
        else:
            leaves.append(a)
    return leaves


class _TensorList:
    """A sequence argument of tensors: its length selects the plan."""

    __slots__ = ("seq", "n")

    def __init__(self, seq):
        self.seq, self.n = seq, len(seq)

    def __eq__(self, other):
        return isinstance(other, _TensorList) and other.seq is self.seq and other.n == self.n

    __hash__ = None


def _is_tensor(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _describe(x):
    """The part of a leaf that selects a plan (not tensor addresses)."""
    if _is_tensor(x):
        return ("t", x.dtype, x.device, tuple(x.shape), x.stride(), x.storage_offset(),
                x.untyped_storage().nbytes(), x.data_ptr() % 16)
    if isinstance(x, _TensorList):
        return ("L", x.n)
    if isinstance(x, np.ndarray):
        if _is_host_index(x):
            return ("h", x.dtype.str, x.shape)  # a host index: values are parameters or guarded
        return ("a", x.dtype.str, x.shape, x.tobytes())
    if isinstance(x, (list, tuple)):
        return ("hl", len(x)) if _is_host_index(x) else ("l", tuple(x))
    if isinstance(x, (int, float, bool, str, type(None))) or isinstance(x, tuple):
        return ("v", x)
    raise TypeError(f"plan arguments must be tensors, lists of tensors, numpy arrays or scalars, "
                    f"got {type(x).__name__}")


class _Trace:
    """One frozen layout: identities it needs, index tensors it rebinds."""

    __slots__ = ("frozen", "ident", "tables", "params", "vguard", "consts", "site", "plain", "device", "checks")

    def __init__(self):
        self.frozen = None
        self.ident: list = []   # (leaf index, storage pointer) that must not change
        self.tables: list = []  # (leaf index, traced pointer, bytes): device index parameters
        self.params: list = []  # (leaf index, pinned buffer): host index parameters, refilled each call
        self.vguard: list = []  # (leaf index, values): host indices whose values the plan depends on
        self.consts: list = []  # (getter, view signature): tensors reached by name, not as arguments
        self.device = None      # the GPU of its CUDA tensors: replays run on that device's current stream
        self.checks = False     # copies that may overlap through host indices: frozen.host_overlap() per call
        self.site = None        # replay-or-plain decision (a _Site)
        self.plain = None       # why this layout always runs plainly (None: it replays)


class StepPlan:
    MAX_TRACES = 8   # per argument layout (e.g. double-buffered arguments)
    MAX_LAYOUTS = 64

    def __init__(self, engine, step: Callable):
        self._engine, self._step = engine, step
        self._layouts: OrderedDict = OrderedDict()  # layout key -> {"warm": int, "traces": [_Trace]}
        self._last = None                           # (leaves, trace) of the previous call
        self._last_memory = None                    # its tensors' memory signature, see _fast
        self._mem_now = None
        self._mu = threading.RLock()

    # ---- calls -------------------------------------------------------------------------

    def __call__(self, *args, **kw) -> None:
        with self._mu:  # threads sharing a plan take turns (its traces and last call are one state)
            self._call(*args, **kw)

    def _call(self, *args, **kw) -> None:
        leaves = _flatten(args, kw)
        self._mem_now = None
        tr = self._fast(leaves)
        if tr is None:
            key = tuple(_describe(x) for x in leaves)
            lay = self._layouts.get(key)
            if lay is None:
                lay = self._layouts[key] = {"warm": 0, "traces": []}
                while len(self._layouts) > self.MAX_LAYOUTS:
                    self._layouts.popitem(last=False)
            self._layouts.move_to_end(key)
            tr = next((t for t in lay["traces"] if self._matches(t, leaves)), None)
            if tr is None:
                self._warm(lay, args, kw, leaves)
                return
        if tr.consts and not all(_sig_now(get) == sig for get, sig in tr.consts):
            # A tensor the step reaches by name is another one now: trace again.
            for lay in self._layouts.values():
                if tr in lay["traces"]:
                    lay["traces"].remove(tr)
            self._last = None
            self(*args, **kw)
            return
        self._last = (leaves, tr)
        self._last_memory = self._mem_now if self._mem_now is not None else _memory(leaves)
        if tr.plain is not None:
            self._step(*args, **kw)
            return
        self._run(tr, args, kw, leaves)

    def _fast(self, leaves):
        """The previous call's trace when every leaf is the same object over the same memory."""
        if self._last is None:
            return None
        prev, tr = self._last
        # Same objects only for tensors, tensor lists and immutable scalars: a
        # numpy array or list may hold new host indices in place. The same tensor
        # object may hold other memory now (`.data =`, set_, resize_): the
        # addresses and shapes are checked too (the object is a proxy for them).
        if len(prev) == len(leaves) and all(map(_same_leaf, prev, leaves)):
            self._mem_now = _memory(leaves)
            if self._mem_now == self._last_memory:
                return tr
        return None

    @staticmethod
    def _matches(tr: _Trace, leaves) -> bool:
        return (all(leaves[i].untyped_storage().data_ptr() == p for i, p in tr.ident)
                and all(_values(leaves[i]) == v for i, v in tr.vguard))

    def _run(self, tr: _Trace, args, kw, leaves) -> None:
        from ._api import _Site  # noqa: F401  (decision logic shared with scope sites)
        e = self._engine
        site = tr.site
        capturing = torch is not None and torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        kind, timed = site.choose()
        if capturing:
            kind, timed = ("replay" if site.decided is not False else None), False
        if timed:
            ev0, t0 = _start()
        if kind == "replay":
            for i, buf in tr.params:  # read on the host when the copies are issued: refilled at once
                _fill(buf, leaves[i])
            if tr.checks and tr.frozen.host_overlap():
                # This call's indices make a copy read what it writes: two passes, plainly
                # (no A/B sample: it is not the replay's alternative).
                kind, timed = None, False
        if kind == "replay":
            stream = 0
            if torch is not None and torch.cuda.is_available():
                stream = torch.cuda.current_stream(tr.device).cuda_stream
            rebinds = [(p, p + n, leaves[i].data_ptr()) for i, p, n in tr.tables]
            if rebinds and tr.frozen.rebind_aliases(rebinds):
                kind, timed = None, False  # this call's index tensors lie where the copies write: plainly
        if kind == "replay":
            with e._lock:
                if e._depth:
                    raise RuntimeError("a step plan replays outside scopes (it would reorder the queued copies)")
                tr.frozen.run_rebound(stream, rebinds)
                e._after(*leaves)
        if kind != "replay":
            self._step(*args, **kw)
        if timed:
            site.sample(kind == "replay", *_stop(ev0, t0))

    # ---- tracing ---------------------------------------------------------------------

    def _warm(self, lay, args, kw, leaves) -> None:
        """Run the step intercepted until the engine settled, then freeze it."""
        from ._api import _Site
        from . import _shim
        e = self._engine
        f0, r0 = e._e.flushes, e._e.tuning_runs
        prev_force, prev_tracing = e._force, tracing_engine()
        e._force = e._force or ("shim" if _shim.lib is not None else "mode")
        cands = [i for i, x in enumerate(leaves) if _is_host_index(x)]
        params = {id(leaves[i]): _buffer(leaves[i]) for i in cands} if torch is not None else {}
        _tracing.engine = e
        e._e.set_recording(True)
        try:
            # The step runs once: a copy that cannot read a host index parameter when it
            # runs takes the values instead (Engine.copy_), so nothing is done twice.
            _tracing.params, _tracing.used, _tracing.values, _tracing.overlaps = params, set(), 0, 0
            with e.scope():
                self._step(*args, **kw)
            used, values, overlaps = _tracing.used, _tracing.values, _tracing.overlaps
        finally:
            _tracing.params = None
            e._e.set_recording(False)
            e._force, _tracing.engine = prev_force, prev_tracing
        flushes, quiet = e._e.flushes - f0, e._e.tuning_runs == r0
        tr = _Trace()
        if flushes != 1:
            tr.plain = ("the step copies nothing" if flushes == 0 else
                        "other work in the step reads or writes its copies' memory (they ran in several flushes)")
        else:
            lay["warm"] = lay["warm"] + 1 if quiet else 0
            if lay["warm"] < _Site.SETTLE:
                return  # the engine is still tuning this step
            tr.frozen = e._e.freeze(rebindable=True)
            caps = _captured(self._step)
            if self._unbound(tr, leaves, caps):
                # Copies through tensors the step allocated and dropped: compose them away.
                # The engine's own references (torch's copies, its baseline) go first.
                tr.frozen = None
                e._e.release_payloads()
                dead = _dead_temporaries(self._unbound_of(e._e.freeze(rebindable=True).allocations, leaves, caps))
                composed = e._e.freeze_composed(dead, rebindable=True) if dead else None
                tr.frozen = composed if composed is not None else e._e.freeze(rebindable=True)
            for i in cands:
                if id(leaves[i]) in used and not values:
                    tr.params.append((i, params[id(leaves[i])]))
                else:  # not a parameter (or the step derived indices from values): guarded by value
                    tr.vguard.append((i, _values(leaves[i])))
            tr.plain = self._bind(tr, leaves, caps)
            tr.device = next((x.device for x in leaves if _is_tensor(x) and x.is_cuda), None)
            tr.checks = overlaps > 0
            tr.site = _Site("replay")
            tr.site.settled = _Site.SETTLE  # the engine already settled while tracing
        lay["warm"] = 0
        lay["traces"].append(tr)
        if len(lay["traces"]) > self.MAX_TRACES:
            lay["traces"].pop(0)
        self._last, self._last_memory = (leaves, tr), _memory(leaves)

    @staticmethod
    def _storages(leaves):
        storages = {}
        for i, x in enumerate(leaves):
            if _is_tensor(x):
                s = x.untyped_storage()
                storages.setdefault((x.device.type == "cuda", x.device.index or 0, s.data_ptr()), (i, s.nbytes()))
        return storages

    @staticmethod
    def _unbound(tr: _Trace, leaves, caps=()):
        """The frozen copies' allocations that are no argument nor reached by name: (kind, index, lo, hi)."""
        return StepPlan._unbound_of(tr.frozen.allocations, leaves, caps)

    @staticmethod
    def _unbound_of(allocations, leaves, caps=()):
        storages = StepPlan._storages(leaves)
        named = _cap_storages(caps)
        return [(kind, index, lo, hi) for kind, index, lo, hi in allocations
                if storages.get((kind == 1, index if kind == 1 else 0, lo)) is None
                and (kind == 1, index if kind == 1 else 0, lo) not in named]

    @staticmethod
    def _bind(tr: _Trace, leaves, caps=()):
        """Maps the frozen copies' memory to the arguments; why not replayable, or None."""
        storages = {}
        for i, x in enumerate(leaves):
            if _is_tensor(x):
                s = x.untyped_storage()
                storages.setdefault((x.device.type == "cuda", x.device.index or 0, s.data_ptr()), (i, s.nbytes()))
        tables = set(tr.frozen.tables)
        for i, x in enumerate(leaves):
            if _is_tensor(x) and x.is_cuda and x.data_ptr() in tables and not x.is_floating_point():
                if x.dtype is not torch.int64 or not x.is_contiguous():
                    return "a CUDA index argument is not a contiguous int64 tensor"
                tr.tables.append((i, x.data_ptr(), x.numel() * 8))
        table_ranges = [(p, p + n) for _, p, n in tr.tables]
        table_ranges += [(b.data_ptr(), b.data_ptr() + b.numel() * 8) for _, b in tr.params]
        for p in tables:
            if not any(lo <= p < hi for lo, hi in table_ranges):
                return "an index tensor the copies read is not an argument (or was converted inside the step)"
        named = _cap_storages(caps)
        for kind, index, lo, hi in tr.frozen.allocations:
            where = (kind == 1, index if kind == 1 else 0, lo)
            hit = storages.get(where)
            if hit is None and where in named:
                for get, t in named[where]:  # reached by name: the same view at each replay
                    tr.consts.append((get, _sig(t)))
                continue
            if hit is None:
                return ("the copies touch memory that is neither an argument nor a tensor the step names (a "
                        "tensor allocated inside the step, or reached through other objects): pass it as an "
                        "argument")
            if (hit[0], lo) not in tr.ident:
                tr.ident.append((hit[0], lo))
        return None


def _captured(step):
    """Tensors the step reaches by name: (getter, tensor) for its closure
    variables, the globals and (for a bound method) its object's attributes it
    names, and the tensors of lists or tuples held there."""
    out = []
    fn = getattr(step, "__func__", step)
    code = getattr(fn, "__code__", None)
    if code is None:
        return out

    def add(get):
        try:
            v = get()
        except Exception:
            return
        if _is_tensor(v):
            out.append((get, v))
        elif isinstance(v, (list, tuple)):
            for j, x in enumerate(v):
                if _is_tensor(x):
                    out.append((lambda g=get, j=j: g()[j], x))

    for cell in fn.__closure__ or ():
        add(lambda c=cell: c.cell_contents)
    names = set(code.co_names)
    glb = getattr(fn, "__globals__", {})
    for n in names & glb.keys():
        add(lambda n=n: glb[n])
    obj = getattr(step, "__self__", None)
    attrs = getattr(obj, "__dict__", None)
    if isinstance(attrs, dict):
        for n in names & attrs.keys():
            add(lambda n=n: attrs[n])
    return out


def _cap_storages(caps):
    named = {}
    for get, t in caps:
        named.setdefault((t.device.type == "cuda", t.device.index or 0, t.untyped_storage().data_ptr()), []).append((get, t))
    return named


def _sig(t):
    return (t.data_ptr(), t.dtype, tuple(t.shape), t.stride(), t.device)


def _sig_now(get):
    try:
        t = get()
    except Exception:
        return None
    return _sig(t) if _is_tensor(t) else None


def _memory(leaves):
    """(data_ptr, shape, strides, storage offset) of every tensor leaf: what a replay's
    addresses depend on."""
    return [(x.data_ptr(), x.shape, x.stride(), x.storage_offset()) for x in leaves if _is_tensor(x)]


def _same_leaf(a, b) -> bool:
    if isinstance(a, _TensorList):
        return a == b
    return a is b and not isinstance(a, (np.ndarray, list)) and not _is_host_index(a)


def _start():
    import time
    ev = None
    if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
    return ev, time.perf_counter()


def _stop(ev0, t0):
    import time
    host = time.perf_counter() - t0
    if ev0 is None:
        return host, None, None
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    return host, ev0, end
