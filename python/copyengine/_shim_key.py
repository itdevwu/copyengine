"""Identity of an interception shim build: torch build x shim sources.

Standalone (no copyengine imports) so the CMake build can stamp the shim it
prebuilds with the same key the runtime checks:

    python _shim_key.py <dir with shim.cpp and include/> <copyengine version>
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HEADERS = ("frontend.hpp", "engine.hpp", "ir.hpp", "types.hpp")


def sources(src_dir: Path) -> list[Path]:
    return [src_dir / "shim.cpp"] + [src_dir / "include" / "copyengine" / h for h in HEADERS]


def key(src_dir: Path, ce_version: str) -> str:
    import torch

    h = hashlib.sha256()
    for p in sources(src_dir):
        h.update(p.read_bytes())
    git = (getattr(torch.version, "git_version", None) or "nogit")[:12]
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    return f"torch{torch.__version__}-{git}-abi{abi}-ce{ce_version}-{h.hexdigest()[:12]}"


if __name__ == "__main__":
    print(key(Path(sys.argv[1]), sys.argv[2]), end="")
