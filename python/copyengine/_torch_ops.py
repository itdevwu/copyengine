"""Loads copyengine._torch_stable, the explicit copy API as torch custom ops.

The library is built on torch's stable ABI (2.10+). When it is missing, or torch
is older, the Python layer issues copies itself; results are the same.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from . import _core

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

_ERRORS = {
    0: ValueError, 1: _core.NeedsMaterialize, 2: _core.UnsafeCopy, 3: _core.Unsupported, 4: _core.BackendError,
}

copy_ = copy_fn_ = None
error = ""  # why the ops are unavailable


def _load() -> None:
    global copy_, copy_fn_, error
    if torch is None:
        error = "torch is not installed"
        return
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 10):
        error = f"torch {torch.__version__} predates the stable ABI the torch ops use (2.10)"
        return
    path = Path(__file__).with_name("_torch_stable.so")
    if not path.exists():
        error = "copyengine was built without the torch extension"
        return
    try:
        torch.ops.load_library(str(path))
        if ctypes.CDLL(str(path)).copyengine_torch_bind(ctypes.c_size_t(_core._frontend_api())) != 0:
            error = "torch extension and copyengine._core come from different builds"
            return
    except (OSError, RuntimeError) as e:  # e.g. torch older than the stable ABI it needs
        error = f"cannot load the torch extension: {e}"
        return
    copy_ = torch.ops.copyengine.copy_.default
    copy_fn_ = torch.ops.copyengine.copy_fn_.default


def translate(e: RuntimeError) -> Exception:
    """The copyengine exception a custom op's error stands for."""
    msg = str(e)
    at = msg.find("[copyengine:")
    if at < 0:
        return e
    end = msg.find("]", at)
    code = int(msg[at + len("[copyengine:"):end])
    return _ERRORS.get(code, _core.CopyEngineError)(msg[end + 1:].strip())


_load()
