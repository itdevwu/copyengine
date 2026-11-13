"""The C++ interception shim behind ``scope()``.

It is compiled against the exact torch in use (torch's unstable dispatcher
API), so the wheel carries one prebuilt for the torch it was built with and the
sources for any other: on import, a missing or stale build is compiled once
(a C++ compiler and a few seconds) and cached per torch build. If that fails,
scopes use the Python interception mode instead: same semantics, slower.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import _core
from ._shim_key import key as _key

try:
    import torch
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None

_HERE = Path(__file__).parent
_SRC = _HERE / "_shim_src"

lib = None    # ctypes handle once loaded and registered
error = ""    # why the shim is unavailable


def cache_dir() -> Path:
    base = os.environ.get("CE_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "copyengine" / "shim"


def _compiler() -> str:
    for c in (os.environ.get("CXX"), "c++", "g++", "clang++"):
        if c and shutil.which(c):
            return c
    raise RuntimeError("no C++ compiler found (set CXX)")


def build(out: Path) -> None:
    """Compiles the shim for the running torch into `out` (atomically)."""
    from torch.utils import cpp_extension

    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".so", dir=out.parent)
    os.close(fd)
    cmd = [_compiler(), "-shared", "-fPIC", "-O2", "-std=c++20", "-fvisibility=hidden",
           f"-D_GLIBCXX_USE_CXX11_ABI={abi}", f"-I{_SRC / 'include'}",
           *[f"-isystem{p}" for p in cpp_extension.include_paths()],
           str(_SRC / "shim.cpp"), "-o", tmp,
           *[f"-L{p}" for p in cpp_extension.library_paths()], "-lc10", "-ltorch_cpu", "-ltorch", "-ldl"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"compiling the shim failed:\n{r.stderr[-4000:]}")
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def shim_key() -> str:
    return _key(_SRC, _core.__version__)


def _locate() -> Path:
    k = shim_key()
    pre = _HERE / "shims" / k / "libcopyengine_shim.so"  # prebuilt ones ship in the wheel
    if pre.exists():
        return pre
    out = cache_dir() / k / "libcopyengine_shim.so"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.parent / ".lock", "w") as lock:  # one build per machine, other processes wait
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not out.exists():
            build(out)
    return out


def _load() -> None:
    global lib, error
    if torch is None:
        error = "torch is not installed"
        return
    if os.environ.get("CE_SHIM", "1") in ("0", "off", "false"):
        error = "disabled by CE_SHIM=0"
        return
    try:
        path = _locate()
        handle = ctypes.CDLL(str(path))
        handle.copyengine_shim_enter.argtypes = [ctypes.c_size_t]
        handle.copyengine_shim_exit.argtypes = []
        rc = handle.copyengine_shim_init(ctypes.c_size_t(_core._frontend_api()))
    except Exception as e:  # noqa: BLE001 - any failure means: use the Python mode
        error = str(e)
        return
    if rc != 0:
        error = {1: "shim and copyengine._core come from different builds",
                 2: "another library already intercepts the DeferredInit dispatch key (torchdistx?)",
                 3: "registering the shim with torch's dispatcher failed"}.get(rc, f"init returned {rc}")
        return
    if not _verify(handle):
        handle.copyengine_shim_disable()
        error = "the shim did not pass its verification copy"
        return
    lib = handle


def _verify(handle) -> bool:
    """Admission: one copy through the interception, checked."""
    o = _core.EngineOptions()
    o.prior_only = True
    e = _core.Engine(o)
    dst, src = torch.zeros(16), torch.arange(16.0)
    e.begin_scope()
    handle.copyengine_shim_enter(e.handle)
    try:
        dst.copy_(src)
        queued = e.queued
    finally:
        handle.copyengine_shim_exit()
        e.end_scope()
    return queued == 1 and torch.equal(dst, src)


_load()
