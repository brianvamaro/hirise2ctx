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
# Reject a free-threaded / too-new interpreter: scipy/scikit-learn/rasterio don't ship wheels
# for it, so everything falls back to source builds that fail (no OpenBLAS/GDAL on Sherlock).
if [ "$("$PYBIN" -c 'import sysconfig;print(sysconfig.get_config_var("Py_GIL_DISABLED") or 0)')" = "1" ]; then
    echo "ERROR: $PYBIN is a FREE-THREADED Python -- the sci stack has no wheels for it." >&2
    echo "  Use a standard CPython 3.11/3.12 module:  ml spider python" >&2
    echo "  then:  rm -rf '$ENV_DIR'; PYMODULE=python/<ver> bash setup_sherlock_env.sh" >&2
    exit 1
fi

ENV_DIR="${ENV_DIR:-/home/groups/mlapotre/bamaro/envs/hirise2ctx}"
if [ ! -d "$ENV_DIR" ]; then
    "$PYBIN" -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel

# CUDA PyTorch (default index = CUDA wheel, bundles its own CUDA runtime; no system CUDA
# module needed). The embedder auto-uses cuda + fp16 autocast when a GPU is visible.
pip install torch

# Pre-install EVERY compiled dependency the inference path needs, WHEELS ONLY. The Sherlock
# gotcha: its base OS has an older glibc (~2.17) and no system GDAL/OpenBLAS, so (a) the newest
# scipy/scikit-learn/numpy/pandas wheels (built for manylinux_2_28) won't load and (b) a source
# fallback can't compile (no gdal-config / no OpenBLAS). `--only-binary` forbids sdists, so pip
# picks the newest version whose wheel is *compatible with this glibc* (a slightly older one)
# instead of trying to build. These + torch (above) cover map_region.py + parity_check.py.
pip install --only-binary=:all: \
    numpy pandas matplotlib pyyaml \
    "rasterio>=1.3" "pyproj>=3.6" "shapely>=2.0" pyogrio fiona scikit-image \
    "scipy>=1.11" "scikit-learn>=1.4"
# If this errors "Could not find a version that satisfies ... (--only-binary)", no compatible
# wheel exists for this Python -> try another module: ml spider python ;
#   rm -rf "$ENV_DIR"; PYMODULE=python/<ver> bash setup_sherlock_env.sh

# The project itself (editable). CORE deps only -- the geo/sci stack above is already satisfied,
# so this won't rebuild anything; remaining core deps (geopandas) are pure-python.
pip install -e .

# Training/notebook extras, BEST-EFFORT. lightgbm/pyarrow are training-only and may lack an
# old-glibc wheel; jupyter/pytest are for local dev. None are needed for the inference run, so
# don't let them fail the setup.
pip install --only-binary=:all: lightgbm pyarrow pytest jupyter nbconvert \
    || echo "NOTE: skipped some optional extras (no compatible wheel); the inference path is unaffected."

# Fang-ViT checkpoint (~341.7 MB) from Zenodo 18180801. A bare `curl -L` silently saves an
# error/redirect page on a stalled transfer (seen: a 92-byte file), which then blows up at
# FangEmbedder.load(). So: -f (fail on HTTP error, don't save the page), --retry/-C (resume),
# and a size check that rejects a too-small file and tells you how to recover.
CKPT="models/pretrained/mars-mae-dino-vit-base-v1.pth"
CKPT_URL="https://zenodo.org/records/18180801/files/mars-mae-dino-vit-base-v1.pth?download=1"
CKPT_MIN_BYTES=$((300 * 1024 * 1024))   # expect ~341 MB
ckpt_size() { stat -c%s "$CKPT" 2>/dev/null || echo 0; }
if [ ! -f "$CKPT" ] || [ "$(ckpt_size)" -lt "$CKPT_MIN_BYTES" ]; then
    mkdir -p models/pretrained
    [ -f "$CKPT" ] && rm -f "$CKPT"     # drop any truncated/garbage partial first
    echo "Downloading Fang checkpoint (~341 MB) from Zenodo 18180801 ..."
    curl -fL --retry 5 --retry-delay 5 -C - -o "$CKPT" "$CKPT_URL" || true
    if [ "$(ckpt_size)" -lt "$CKPT_MIN_BYTES" ]; then
        echo "ERROR: checkpoint download incomplete ($(ckpt_size) bytes; expected ~341 MB)." >&2
        echo "  Retry on the data-transfer node, or just upload it from the laptop (you have it):" >&2
        echo "    scp models/pretrained/mars-mae-dino-vit-base-v1.pth \\" >&2
        echo "        bamaro@dtn.sherlock.stanford.edu:hirise2ctx/models/pretrained/" >&2
        rm -f "$CKPT"
        exit 1
    fi
    echo "checkpoint OK ($(ckpt_size) bytes)"
fi

# Smoke test: torch sees CUDA, and the whole inference import chain resolves (so a missing
# dep shows up here, not mid-run). Run from the repo root so `import src` works.
python - <<'PY'
import torch, rasterio, scipy, sklearn, numpy, pandas
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device", (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
print("rasterio", rasterio.__version__, "| scipy", scipy.__version__,
      "| sklearn", sklearn.__version__, "| numpy", numpy.__version__)
import sys; sys.path.insert(0, ".")
import src.modeling  # OpenMP/DLL bootstrap; must precede the heavy imports
from src.mapping import predict_window          # rasterio path
from src.calibration import CalibrationLayer    # scipy + sklearn path
from src.fm_embeddings import FangEmbedder       # torch path
from src.modeling.mlp_head import DeployableHead
print("inference imports OK -> ready for parity_check + map_region")
PY

echo "OK -- activate later with:  ml $PYMODULE && source $ENV_DIR/bin/activate"
