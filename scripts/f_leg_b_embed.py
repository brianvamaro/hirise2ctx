"""F pilot leg B — laptop: embed calibrated-frame crops -> fang_embeddings_f NPZ files.

For each training obs_id, composites the I/F crops transferred from Sherlock onto the
CTX mosaic pixel grid, applies per-frame robust normalization (matching the 'perframe'
mapping from leg A: median -> 125 DN, IQR -> 27.7 DN), and embeds with the frozen
Fang-ViT at S=32 / P=96 / GeM pooling — the recipe used in fang_embeddings/.

Output matches the existing embedding format: dataset_v2/fang_embeddings_f/{obs}_P96.npz
with arrays (ti, tj, valid, gem).  The LOIO gate script (f_leg_b_loio.py) reads from this
store and compares skill against the baseline fang_embeddings/ store.

Run (laptop GPU, after transferring obs_crops from Sherlock):
  # expected layout: reports/f_leg_b/obs_crops/{obs_id}_{pid}_ifcrop.tif
  conda run --no-capture-output -n geospatial python -u scripts/f_leg_b_embed.py
  conda run --no-capture-output -n geospatial python -u scripts/f_leg_b_embed.py --smoke  # 2-img test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

from src.fm_embeddings import FangEmbedder
from src.striping import A1_REF_MEDIAN, A1_REF_IQR

CROPS_DIR = REPO / "reports" / "f_leg_b" / "obs_crops"
LABELS_DIR = REPO / "dataset_v2" / "labels"
FEATURES_DIR = REPO / "dataset_v2" / "features"
OUT_DIR = REPO / "dataset_v2" / "fang_embeddings_f"
TILE_PX = 32      # S=32; context patch = 3*32 = 96 px
BATCH = 96
PX_M = 5.0        # native CTX resolution


# ------------------------------------------------------------------ normalization

def to_uint8_perframe(arr: np.ndarray) -> np.ndarray:
    """Per-frame robust normalization: I/F median -> A1_REF_MEDIAN, IQR -> A1_REF_IQR."""
    fin = np.isfinite(arr)
    if fin.sum() < 50:
        return np.zeros(arr.shape, dtype=np.uint8)
    v = arr[fin]
    med = float(np.median(v))
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(max(q75 - q25, 1e-6))
    out = np.zeros(arr.shape, dtype=np.uint8)
    out[fin] = np.clip(
        (v - med) / iqr * A1_REF_IQR + A1_REF_MEDIAN, 1, 255
    ).astype(np.uint8)
    return out


# ------------------------------------------------------------------ composite

def composite_crops(obs_id: str, row0: int, col0: int, H: int, W: int) -> np.ndarray:
    """Composite all I/F crops for obs_id onto the mosaic pixel grid (H×W uint8).

    Each crop is normalized per-frame before compositing.  Where multiple frames
    overlap, the last one written wins (crops are sorted by filename so the order
    is deterministic; seams between frames normalized to the same target should be
    small).
    """
    canvas = np.zeros((H, W), dtype=np.float32)
    covered = np.zeros((H, W), dtype=bool)

    crops = sorted(CROPS_DIR.glob(f"{obs_id}_*_ifcrop.tif"))
    if not crops:
        return np.zeros((H, W), dtype=np.uint8)

    # Destination transform: top-left of the mosaic window at native 5 m/px
    # row0/col0 are mosaic pixel coordinates of the window's top-left corner
    # x = col0 * 5 m + mosaic_x_origin — but we work in relative pixel coords:
    # tile (ti, tj) covers rows [ti*32, ti*32+32) globally, so window rows
    # start at row0.  We reconstruct the world coords from the first crop's CRS.
    dst_transform = None

    for crop_path in crops:
        with rasterio.open(crop_path) as src:
            arr = src.read(1).astype(np.float32)
            src_crs = src.crs
            src_transform = src.transform
            if dst_transform is None:
                # Use this crop's CRS to fix the destination transform
                dst_transform = src_transform  # will be recomputed below
                # Reconstruct from row0/col0 and the crop's CRS transform:
                # The mosaic window top-left in world coords:
                from rasterio.transform import xy as _xy
                # We need the world coords of mosaic pixel (row0, col0).
                # They are stored in the ctx_window_tif which has the same
                # transform as the mosaic tile. Read from the crop itself:
                # crop covers the exact obs bounds, so its top-left IS (row0,col0).
                dst_transform = src_transform  # same grid as the crop

        # Reproject onto the destination H×W grid
        normed_if = arr.copy()
        fin = np.isfinite(normed_if) & (normed_if > 0)
        if fin.sum() < 50:
            continue
        normed_if[~fin] = np.nan

        dst_if = np.full((H, W), np.nan, dtype=np.float32)
        reproject(source=normed_if, destination=dst_if,
                  src_transform=src_transform, src_crs=src_crs,
                  dst_transform=dst_transform, dst_crs=src_crs,
                  src_nodata=np.nan, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)

        new = np.isfinite(dst_if)
        canvas[new] = dst_if[new]
        covered |= new

    # Apply per-frame normalization to the composite (as if it were one frame)
    result = np.zeros((H, W), dtype=np.uint8)
    if covered.any():
        result = to_uint8_perframe(canvas)
    return result


# ------------------------------------------------------------------ embed one image

def embed_one(obs_id: str, embedder: FangEmbedder) -> bool:
    """Embed a single training image from its I/F crops.  Returns True on success."""
    out_path = OUT_DIR / f"{obs_id}_P96.npz"
    if out_path.exists():
        print(f"  {obs_id}: cached", flush=True)
        return True

    sidecar_path = LABELS_DIR / f"{obs_id}.json"
    if not sidecar_path.exists():
        print(f"  {obs_id}: no sidecar JSON; skipping", flush=True)
        return False

    sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    row0 = int(sc["mosaic_row_origin"])
    col0 = int(sc["mosaic_col_origin"])

    # Get window dimensions from the existing ctx_window_tif
    ctx_tif = Path(sc["ctx_window_tif"])
    if not ctx_tif.exists():
        print(f"  {obs_id}: ctx_window_tif missing; skipping", flush=True)
        return False
    with rasterio.open(ctx_tif) as ds:
        H, W = ds.height, ds.width

    # Build composite uint8 window
    window8 = composite_crops(obs_id, row0, col0, H, W)
    if not window8.any():
        print(f"  {obs_id}: all-zero composite (no crops found?); skipping", flush=True)
        return False

    # Get tile grid from features parquet
    feat_path = FEATURES_DIR / f"{obs_id}.parquet"
    if not feat_path.exists():
        print(f"  {obs_id}: no features parquet; skipping", flush=True)
        return False

    import pandas as pd
    feats = pd.read_parquet(feat_path)
    # Filter to scale_idx=2 (S=32) and unique (ti, tj)
    f32 = feats[feats["scale_idx"] == 2][["ti", "tj"]].drop_duplicates()
    ti = f32["ti"].to_numpy(np.int64)
    tj = f32["tj"].to_numpy(np.int64)

    emb, valid = embedder.embed_window(
        window8, ti, tj, tile_px=TILE_PX, row0=row0, col0=col0, pool="gem",
        batch=BATCH
    )

    n_valid = int(valid.sum())
    print(f"  {obs_id}: {len(ti)} tiles, {n_valid} valid "
          f"({n_valid / max(len(ti), 1):.0%})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, ti=ti, tj=tj, valid=valid, gem=emb.astype(np.float32))
    return True


# ------------------------------------------------------------------ main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="embed only the first 2 obs_ids")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--obs", nargs="+", help="embed specific obs_ids only")
    args = ap.parse_args()

    crops = sorted({p.name.split("_")[0] + "_" + p.name.split("_")[1]
                    for p in CROPS_DIR.glob("*_ifcrop.tif")})
    if not crops:
        print(f"No I/F crops found in {CROPS_DIR}\n"
              "Transfer obs_crops from Sherlock first:\n"
              "  tar cf obs_crops.tar -C $SCRATCH/hirise2ctx/f_leg_b obs_crops\n"
              "  scp obs_crops.tar laptop:~/hirise2ctx/reports/f_leg_b/\n"
              "  cd reports/f_leg_b && tar xf obs_crops.tar")
        sys.exit(1)

    # Reconstruct obs_ids from crop filenames: {obs_id}_{pid}_ifcrop.tif
    # obs_id is "ESP_XXXXXX_XXXX" (3 parts joined by _)
    obs_ids: list[str] = []
    seen: set[str] = set()
    for p in sorted(CROPS_DIR.glob("*_ifcrop.tif")):
        parts = p.name.replace("_ifcrop.tif", "").split("_")
        # obs_id = first 3 parts: "ESP", "XXXXXX", "XXXX"
        obs_id = "_".join(parts[:3])
        if obs_id not in seen:
            seen.add(obs_id)
            obs_ids.append(obs_id)

    if args.obs:
        obs_ids = [o for o in obs_ids if o in set(args.obs)]
    if args.smoke:
        obs_ids = obs_ids[:2]

    print(f"{len(obs_ids)} obs_ids to embed", flush=True)
    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)

    ok = fail = 0
    for obs_id in obs_ids:
        if embed_one(obs_id, embedder):
            ok += 1
        else:
            fail += 1

    print(f"\nembedded: {ok}  failed/skipped: {fail}")
    print(f"store: {OUT_DIR}")
    print(f"\nnext: conda run -n geospatial python scripts/f_leg_b_loio.py")


if __name__ == "__main__":
    main()
