#!/bin/bash
# setup_sherlock_env.sh -- build the hirise2ctx venv on Sherlock for the regional/global
# map inference (PLAN_RegionalMap.md §4a). GPU build (unlike the CPU runbook): the Fang-ViT
# embedding runs on the `gpu` partition. No conda (discouraged on Sherlock). Idempotent.
# Run INSIDE a GPU session (sh_dev -p gpu -G 1) so torch sees CUDA during the smoke test.
set -euo pipefail

# Load a Python module that provides `venv`. Sherlock uses Lmod; `ml` can exit 0 even when a
# version is missing, and some python modules expose `python3` but not bare `python` -- either
# way the system /bin/python (no venv) sneaks in. So load, then VERIFY python3 has venv, and
# fail loudly with the fix rather than the cryptic "No module named venv".
# Override the version without editing this file:  PYMODULE=python/3.12.4 bash setup_sherlock_env.sh
PYMODULE="${PYMODULE:-python/3.12.1}"     # `ml spider python` to list real versions (need >=3.10)
ml "$PYMODULE" 2>/dev/null || true
PYBIN="$(command -v python3 || true)"
if [ -z "$PYBIN" ] || ! "$PYBIN" -c "import venv, ensurepip" >/dev/null 2>&1; then
    echo "ERROR: module '$PYMODULE' did not put a venv-capable python3 on PATH (got: ${PYBIN:-none})." >&2
    echo "  Fix: run 'ml spider python' (or 'module avail python') to find the exact name, then:" >&2
    echo "       PYMODULE=python/<version> bash setup_sherlock_env.sh" >&2
    exit 1
fi
echo "using $PYMODULE -> $PYBIN ($($PYBIN --version))"

ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/hirise2ctx}"
if [ ! -d "$ENV_DIR" ]; then
    "$PYBIN" -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# CUDA PyTorch (default index = CUDA wheel, bundles its own CUDA runtime; no system CUDA
# module needed). The embedder auto-uses cuda + fp16 autocast when a GPU is visible.
pip install torch

# Project core + modeling (lightgbm/sklearn/scipy/pyarrow) + scikit-image (feature stage;
# not declared in pyproject). geopandas/shapely/pyproj come via the core deps (Linux wheels).
pip install -e ".[dev,modeling]" scikit-image

# Fang-ViT checkpoint (341 MB) -- NOT auto-downloaded; fetch from Zenodo 18180801 once.
CKPT="models/pretrained/mars-mae-dino-vit-base-v1.pth"
if [ ! -f "$CKPT" ]; then
    mkdir -p models/pretrained
    echo "Downloading Fang checkpoint from Zenodo 18180801 ..."
    curl -L -o "$CKPT" \
      "https://zenodo.org/records/18180801/files/mars-mae-dino-vit-base-v1.pth?download=1"
fi

# Smoke test: torch sees CUDA, and the embedder + head load.
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
PY

echo "OK -- activate later with:  ml $PYMODULE && source $ENV_DIR/bin/activate"
