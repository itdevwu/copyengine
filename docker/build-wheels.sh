#!/bin/bash
# Release wheels and sdist, built inside quay.io/pypa/manylinux_2_28_x86_64 (CI
# runs exactly this):
#
#   docker run --rm -e HOST_UID=$(id -u) -v "$PWD":/work -w /work \
#     quay.io/pypa/manylinux_2_28_x86_64 docker/build-wheels.sh
#
# Python 3.12+ gets one abi3 wheel; older versions one wheel each. Every wheel
# carries the CUDA kernels for all architectures, the stable-ABI torch ops, and
# a prebuilt interception shim per supported torch release (other releases
# compile it on import). Nothing links libcuda; libtorch is the user's own.
set -euo pipefail

CUDA=${CE_WHEEL_CUDA:-12-9}                              # toolkit for nvcc and headers only
PYTHONS=${CE_WHEEL_PYTHONS:-"cp310 cp311 cp312"}
TORCH_BUILD=${CE_WHEEL_TORCH:-2.11.0}                    # built against: stable ops, its own shim
TORCH_SHIMS=${CE_WHEEL_SHIM_TORCH:-"2.9.1 2.10.0"}       # further releases with a prebuilt shim
INDEX=https://download.pytorch.org/whl/cpu
OUT=$(realpath -m "${CE_WHEEL_OUT:-dist}")
WORK=$(mktemp -d)
# Build from a snapshot of the tracked sources, in a private build directory:
# never from (or into) a checkout that local builds configure differently.
SRC="$WORK/src"
mkdir -p "$SRC"
git config --global --add safe.directory "$PWD" 2>/dev/null || true
if git rev-parse --git-dir >/dev/null 2>&1; then
  git ls-files -z | xargs -0 cp --parents -t "$SRC"
else
  cp -r . "$SRC"
fi

step() { echo "== $*"; }

step "CUDA $CUDA toolkit (nvcc, headers)"
dnf -y -q install dnf-plugins-core
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf -y -q install "cuda-nvcc-$CUDA" "cuda-cudart-devel-$CUDA" "cuda-driver-devel-$CUDA" "cuda-nvtx-$CUDA"
export PATH=/usr/local/cuda/bin:$PATH

venv() {  # venv <python tag> <dir>
  "/opt/python/$1-$1/bin/python" -m venv "$2"
  "$2/bin/pip" install -q -U pip
}

for py in $PYTHONS; do
  step "wheel $py (torch $TORCH_BUILD)"
  venv "$py" "$WORK/v-$py"
  "$WORK/v-$py/bin/pip" install -q scikit-build-core nanobind numpy cmake ninja wheel
  "$WORK/v-$py/bin/pip" install -q "torch==$TORCH_BUILD" --index-url "$INDEX"
  "$WORK/v-$py/bin/pip" wheel -q --no-build-isolation --no-deps "$SRC" -w "$WORK/raw" \
    -C build-dir="$WORK/build/{wheel_tag}"
done

# A shim works with any Python: compile one per further torch release (by
# importing a built wheel with that torch) and add it to every wheel.
first=$(ls "$WORK"/raw/*cp312*.whl | head -1)
for t in $TORCH_SHIMS; do
  step "shim for torch $t"
  venv cp312 "$WORK/s-$t"
  "$WORK/s-$t/bin/pip" install -q numpy "torch==$t" --index-url "$INDEX" --extra-index-url https://pypi.org/simple
  "$WORK/s-$t/bin/pip" install -q --no-deps "$first"
  (cd "$WORK" && CE_CACHE_DIR="$WORK/shims-cache" "$WORK/s-$t/bin/python" -I -c '
import copyengine._shim as s
assert s.lib is not None, s.error
print(s.shim_key())')
done

step "add shims, repair (libtorch and libc10 stay the user's), sdist"
W="$WORK/v-cp312/bin"
"$W/pip" install -q auditwheel build
mkdir -p "$OUT" "$WORK/packed"
for whl in "$WORK"/raw/*.whl; do
  rm -rf "$WORK/unpacked"
  "$W/python" -m wheel unpack -d "$WORK/unpacked" "$whl"
  pkg=$(echo "$WORK"/unpacked/*/copyengine)
  cp -r "$WORK"/shims-cache/copyengine/shim/* "$pkg/shims/"
  "$W/python" -m wheel pack -d "$WORK/packed" "$WORK"/unpacked/*
done
"$W/auditwheel" repair -w "$OUT" --plat manylinux_2_28_x86_64 \
  --exclude 'libtorch*.so*' --exclude 'libc10*.so*' "$WORK"/packed/*.whl
"$W/python" -m build --sdist --no-isolation -o "$OUT" "$SRC"

step "smoke test: each wheel in a fresh environment"
for py in $PYTHONS; do
  venv "$py" "$WORK/t-$py"
  "$WORK/t-$py/bin/pip" install -q numpy pytest "torch==$TORCH_BUILD" --index-url "$INDEX" --extra-index-url https://pypi.org/simple
  "$WORK/t-$py/bin/pip" install -q --no-deps --find-links "$OUT" copyengine
  (cd "$WORK" && CE_CACHE_DIR="$WORK/empty-cache" "$WORK/t-$py/bin/python" -I -c '
import copyengine as ce, copyengine._shim as s, os
assert ce._torch_ops.copy_ is not None, ce._torch_ops.error
assert s.lib is not None and "/shims/" in s.lib._name, (s.error, getattr(s.lib, "_name", None))
assert not os.path.exists(os.environ["CE_CACHE_DIR"]), "the prebuilt shim was not used"
print("ok", ce.__version__)')
done

ls -l "$OUT"
[ -n "${HOST_UID:-}" ] && chown -R "$HOST_UID" "$OUT"
rm -rf "$WORK"
