#!/bin/bash
# setup_sherlock_env.sh -- build the hirise2ctx venv on Sherlock for the regional/global
# map inference (PLAN_RegionalMap.md §4a). GPU build (unlike the CPU runbook): the Fang-ViT
# embedding runs on the `gpu` partition. No conda (discouraged on Sherlock). Idempotent.
# Run INSIDE a GPU session (sh_dev -p gpu -G 1) so torch sees CUDA during the smoke test.
set -euo pipefail

ml python/3.12.1                 # `ml spider python` for versions (need >=3.10)

ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/hirise2ctx}"
if [ ! -d "$ENV_DIR" ]; then
    python -m venv "$ENV_DIR"
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

echo "OK -- activate later with:  ml python/3.12.1 && source $ENV_DIR/bin/activate"
