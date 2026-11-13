"""copyengine: a software copy planner for ML infra (not the GPU's DMA hardware)."""

import json as _json

from . import _core
from ._api import (
    Elementwise,
    Engine,
    FrozenPlan,
    barrier,
    copy_,
    default_engine,
    gather,
    indices,
    scatter_,
    scope,
    segments,
    stats,
)
from ._core import (
    ABI_VERSION,
    Access,
    BackendError,
    CopyEngineError,
    NeedsMaterialize,
    Rects,
    Runs,
    Stmt,
    Unsupported,
    UnsafeCopy,
    __version__,
    hardware_fingerprint,
    lower,
)

__all__ = [
    "ABI_VERSION", "Access", "BackendError", "CopyEngineError", "Elementwise", "Engine", "FrozenPlan",
    "NeedsMaterialize", "Rects", "Runs", "Stmt", "Unsupported", "UnsafeCopy", "__version__", "barrier",
    "copy_", "default_engine", "gather", "hardware_fingerprint", "indices", "lower", "probe", "scatter_",
    "scope", "segments", "stats",
]


def probe(bandwidth: bool = False) -> dict:
    """Probe the platform: which copy mechanisms work here, verified bit-exact.

    With ``bandwidth=True`` also runs short microbenchmarks (about a second per GPU).
    """
    return _json.loads(_core.probe_json(bandwidth))
