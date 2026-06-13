"""PLAN_FM §2.2 productization parity: src.fm_embeddings must reproduce the
cached embedding store the frozen recipe was measured on. For one image, slice
the 96-px (S=32 3x3) context boxes from the CTX window via the productized path
and compare GeM vectors to dataset_v2/fang_embeddings/{obs}_P96.npz.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np

from src.fm_embeddings import FangEmbedder, slice_context_boxes

DATASET = REPO_ROOT / "dataset_v2"


def main() -> int:
    import rasterio

    obs = sys.argv[1] if len(sys.argv) > 1 else "ESP_046328_2180"
    z = np.load(DATASET / "fang_embeddings" / f"{obs}_P96.npz")
    ti, tj, valid, gem_cached = z["ti"], z["tj"], z["valid"].astype(bool), z["gem"]

    side = json.loads((DATASET / "labels" / f"{obs}.json").read_text(encoding="utf-8"))
    with rasterio.open(side["ctx_window_tif"]) as src:
        window = src.read(1).astype(np.uint8, copy=False)
    row0, col0 = int(side["mosaic_row_origin"]), int(side["mosaic_col_origin"])

    boxes, valid_new = slice_context_boxes(window, ti, tj, tile_px=32, row0=row0, col0=col0)
    assert np.array_equal(valid_new, valid), "validity mask diverged from the cached store"

    emb = FangEmbedder.load()  # GPU if available -- matches the cached extraction
    gem_new = emb.embed_patches(boxes, pool="gem")

    a = gem_cached[valid]
    diff = np.abs(a - gem_new)
    rel = diff / (np.abs(a) + 1e-6)
    print(f"obs={obs}  device={emb.device}  n_valid={int(valid.sum())}/{valid.size}")
    print(f"  max abs diff = {diff.max():.3e}   mean abs diff = {diff.mean():.3e}")
    print(f"  max rel diff = {rel.max():.3e}   mean rel diff = {rel.mean():.3e}")
    cos = (a * gem_new).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(gem_new, axis=1) + 1e-9)
    print(f"  per-row cosine: min={cos.min():.6f}  mean={cos.mean():.6f}")
    ok = cos.min() > 0.999
    print("PARITY OK" if ok else "PARITY MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
