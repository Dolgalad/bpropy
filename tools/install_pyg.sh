#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Config (env vars)
# -------------------------
TORCH_VERSION="${TORCH_VERSION:-}"      # e.g. 2.6.0 (optional)
CUDA="${CUDA:-auto}"                   # auto | cpu | cu118 | cu121 | cu124 ...
PYG_PKGS="${PYG_PKGS:-torch-scatter}"  # space-separated: "torch-scatter torch-sparse ..."
INSTALL_GEOMETRIC="${INSTALL_GEOMETRIC:-1}"  # 1=yes, 0=no

# -------------------------
# Helper: detect GPU presence
# -------------------------
has_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

# -------------------------
# Choose torch index-url
# -------------------------
if [[ "${CUDA}" == "auto" ]]; then
  if has_gpu; then
    # Default CUDA tag if GPU exists. Change to your preferred default.
    CUDA="cu121"
  else
    CUDA="cpu"
  fi
fi

if [[ "${CUDA}" == "cpu" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
else
  TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA}"
fi

echo "CUDA mode: ${CUDA}"
echo "Torch index-url: ${TORCH_INDEX_URL}"

python -m pip install -U pip

# -------------------------
# Install torch (pinned if requested)
# -------------------------
if [[ -n "${TORCH_VERSION}" ]]; then
  echo "Installing torch==${TORCH_VERSION} from ${TORCH_INDEX_URL}"
  python -m pip install --upgrade --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" "torchvision"
else
  echo "Installing torch (unversioned) from ${TORCH_INDEX_URL}"
  python -m pip install --upgrade --index-url "${TORCH_INDEX_URL}" torch torchvision
fi

# -------------------------
# Read installed torch version/build tag
# -------------------------
TORCH_FULL_VERSION="$(python - <<'PY'
import torch
print(torch.__version__)
PY
)"

BASE_VERSION="${TORCH_FULL_VERSION%%+*}"
BUILD_TAG="${TORCH_FULL_VERSION#*+}"
if [[ "${BASE_VERSION}" == "${BUILD_TAG}" ]]; then
  BUILD_TAG="cpu"
fi

echo "Installed torch: ${TORCH_FULL_VERSION}"
echo "Resolved for PyG: torch-${BASE_VERSION}+${BUILD_TAG}"

# -------------------------
# Check PyG wheel index exists
# -------------------------
PYG_WHL_URL="https://data.pyg.org/whl/torch-${BASE_VERSION}+${BUILD_TAG}.html"
echo "PyG wheel index: ${PYG_WHL_URL}"

python - <<PY
import sys, urllib.request
url = "${PYG_WHL_URL}"
try:
    with urllib.request.urlopen(url, timeout=15) as r:
        if r.status != 200:
            raise RuntimeError(f"HTTP {r.status}")
except Exception as e:
    print("ERROR: PyG wheels not available for this torch build yet.")
    print(f"  Tried: {url}")
    print("Suggested fixes:")
    print("  1) Set TORCH_VERSION to a supported version (e.g. TORCH_VERSION=2.5.1)")
    print("  2) Or choose a different CUDA tag (CUDA=cpu/cu121/...)")
    print("  3) Or build torch-scatter from source for this torch version")
    sys.exit(1)
PY

# -------------------------
# Install PyG binary deps (must come from the wheel index)
# -------------------------
echo "Installing PyG binary packages: ${PYG_PKGS}"
python -m pip install ${PYG_PKGS} -f "${PYG_WHL_URL}"

# -------------------------
# Install torch-geometric (pure python)
# -------------------------
if [[ "${INSTALL_GEOMETRIC}" == "1" ]]; then
  python -m pip install torch-geometric
fi

echo "PyG installation completed successfully."

