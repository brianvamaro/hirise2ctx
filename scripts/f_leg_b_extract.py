"""F pilot leg B — Sherlock-side: extract I/F crops for each training image.

Runs in the MAP venv on Sherlock after all ISIS tasks have completed.  For each
(obs_id, PRODUCT_ID) pair in obs_frame_map.csv, reads the projected .map.cub and
extracts the obs_id's CTX window area as a small compressed float32 I/F GeoTIFF
(the same area the mosaic ctx_window_tif covers, so the laptop embed step can read
it at native 5 m/px resolution).

Output: $WORK/obs_crops/{obs_id}_{pid}_ifcrop.tif  (one TIFF per frame × obs_id)

After this script:
    tar cf obs_crops.tar -C $SCRATCH/hirise2ctx/f_leg_b obs_crops
    scp obs_crops.tar <laptop>:~/repos/hirise2ctx/reports/f_leg_b/
    # on laptop: cd reports/f_leg_b && tar xf obs_crops.tar

Run (Sherlock MAP venv):
    ml python/3.12.1
    source /home/groups/mlapotre/$USER/envs/hirise2ctx/bin/activate
    cd $HOME/hirise2ctx
    python scripts/f_leg_b_extract.py
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("SCRATCH", "/tmp")) / "hirise2ctx" / "f_leg_b"
OBS_MAP = REPO / "reports" / "f_leg_b" / "obs_frame_map.csv"
BOUNDS_CSV = REPO / "reports" / "f_leg_b" / "cohort_obs_bounds.csv"
OUT = WORK / "obs_crops"


def load_bounds(csv_path: Path) -> dict:
    """obs_id -> dict(minx, miny, maxx, maxy, row0, col0)."""
    bd = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            bd[r["obs_id"]] = {k: float(r[k]) if k != "obs_id" else r[k]
                                for k in r}
    return bd


def extract_crop(cub_path: Path, minx: float, miny: float,
                 maxx: float, maxy: float, out_path: Path) -> bool:
    """Extract the given bounds from a projected ISIS cube into a GeoTIFF.

    Reprojects if needed (same CRS assumed for the matched mosaic projection).
    Returns True on success.
    """
    with rasterio.open(cub_path) as src:
        win = from_bounds(minx, miny, maxx, maxy, src.transform)
        # clip to the cube's extent (frame may only partially cover the obs window)
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        if win.width <= 0 or win.height <= 0:
            return False  # no overlap
        win = win.round_offsets().round_lengths()
        arr = src.read(1, window=win, masked=True).astype("float32")
        src_transform = src.window_transform(win)
        src_crs = src.crs

    # Reproject to the exact 5 m/px mosaic grid (aligns pixel boundaries)
    from rasterio.transform import from_origin
    from math import ceil
    # Compute destination grid anchored at the obs window minx/maxy
    px = 5.0
    H = ceil((maxy - miny) / px)
    W = ceil((maxx - minx) / px)
    dst_transform = from_origin(minx, maxy, px, px)
    dst = np.full((H, W), np.nan, dtype=np.float32)
    reproject(source=arr, destination=dst,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=src_crs,
              src_nodata=arr.fill_value if hasattr(arr, "fill_value") else None,
              dst_nodata=np.nan, resampling=Resampling.bilinear)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "GTiff", "dtype": "float32", "count": 1,
            "crs": src_crs, "transform": dst_transform,
            "width": W, "height": H, "compress": "lzw",
            "nodata": np.nan, "tiled": True}
    with rasterio.open(out_path, "w", **prof) as dst_ds:
        dst_ds.write(dst, 1)

    v = dst[np.isfinite(dst)]
    if v.size == 0:
        out_path.unlink(missing_ok=True)
        return False

    print(f"  {out_path.name}: {W}×{H} px, valid {v.size / dst.size:.0%}, "
          f"I/F [{v.min():.4f}, {np.median(v):.4f}, {v.max():.4f}], "
          f"{out_path.stat().st_size / 1e6:.1f} MB", flush=True)
    return True


def main() -> None:
    bounds = load_bounds(BOUNDS_CSV)
    print(f"{len(bounds)} obs_id bounds loaded")

    # Read the (obs_id, PRODUCT_ID) pairs
    pairs: list[tuple[str, str]] = []
    with open(OBS_MAP, newline="") as f:
        for r in csv.DictReader(f):
            pairs.append((r["obs_id"], r["PRODUCT_ID"]))
    print(f"{len(pairs)} (obs_id, frame) pairs")

    ok = skip = fail = 0
    for obs_id, pid in pairs:
        cub = WORK / f"{pid}.map.cub"
        if not cub.exists():
            print(f"  {obs_id} / {pid}: cube missing at {cub}", flush=True)
            fail += 1
            continue
        bd = bounds.get(obs_id)
        if bd is None:
            print(f"  {obs_id}: no bounds entry; skipping", flush=True)
            skip += 1
            continue
        out = OUT / f"{obs_id}_{pid}_ifcrop.tif"
        if out.exists():
            print(f"  {obs_id}_{pid}: already done — skipped", flush=True)
            skip += 1
            continue
        if extract_crop(cub, bd["minx"], bd["miny"], bd["maxx"], bd["maxy"], out):
            ok += 1
        else:
            print(f"  {obs_id} / {pid}: no valid overlap in cube", flush=True)
            fail += 1

    print(f"\nextracted: {ok}  skipped: {skip}  failed/no-overlap: {fail}")
    print(f"obs_crops dir: {OUT}")
    print(f"\nnext: tar cf obs_crops.tar -C {WORK} obs_crops && scp back to laptop")


if __name__ == "__main__":
    main()
