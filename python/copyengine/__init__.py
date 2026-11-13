"""copyengine: a software copy planner for ML infra (not the GPU's DMA hardware)."""

import json as _json

from . import _core
from ._core import ABI_VERSION, __version__

__all__ = ["ABI_VERSION", "__version__", "probe"]


def probe(bandwidth: bool = False) -> dict:
    """Probe the platform: which copy mechanisms work here, verified bit-exact.

    With ``bandwidth=True`` also runs short microbenchmarks (about a second per GPU).
    """
    return _json.loads(_core.probe_json(bandwidth))
