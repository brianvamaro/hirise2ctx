"""F pilot, Sherlock-side extract: window the projected per-frame cubes to the E8_N44 crop.

Runs in the MAP venv on Sherlock (rasterio reads ISIS .cub via GDAL's ISIS3 driver) after the
KEEP_CUBES=1 timing run has left {PRODUCT_ID}.map.cub files on scratch. Writes one compressed
float32 I/F GeoTIFF per crop frame (~tens of MB each vs ~4.8 GB cubes) so the pilot analysis
can run on the laptop GPU.

    ml python/3.12.1 && source /home/groups/mlapotre/$USER/envs/hirise2ctx/bin/activate
    cd ~/hirise2ctx
    python scripts/f_pilot_extract_crop.py
    tar cf pilot_crops.tar -C $SCRATCH/hirise2ctx/f_timing pilot_crops
    # then scp pilot_crops.tar to the laptop repo root and untar into reports/f_timing/

If rasterio errors "not recognized as a supported file format" the wheel lacks the ISIS3
driver -- fall back to the isis micromamba env's gdal_translate:
    gdal_translate -projwin <minx> <maxy> <maxx> <miny> in.map.cub out.tif \
        -co COMPRESS=LZW -a_nodata nan
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("SCRATCH", "/tmp")) / "hirise2ctx" / "f_timing"
OUT = WORK / "pilot_crops"
LIST = REPO / "reports" / "f_timing" / "frame_list.csv"

# E8_N44 A1-payoff crop (native px 1504,8992 +15008) in the Mars_2015 clon_0 CRS,
# +1024 m buffer (scripts/probes/_f_pilot_bounds.py)
BUF = 1024.0
MINX, MINY, MAXX, MAXY = 519317.3 - BUF, 2762465.9 - BUF, 594357.0 + BUF, 2837505.5 + BUF


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(LIST, newline="") as f:
        pids = [r["PRODUCT_ID"] for r in csv.DictReader(f) if r["source"] == "E8_N44_a1_crop"]
    print(f"{len(pids)} crop frames")
    for pid in pids:
        cub = WORK / f"{pid}.map.cub"
        if not cub.exists():
            print(f"  {pid}: MISSING {cub} (need the KEEP_CUBES=1 run)")
            continue
        with rasterio.open(cub) as ds:
            w = from_bounds(MINX, MINY, MAXX, MAXY, ds.transform)
            # clip to the cube's own extent (frames only partially cover the crop)
            w = w.intersection(Window(0, 0, ds.width, ds.height))
            if w.width <= 0 or w.height <= 0:
                print(f"  {pid}: no overlap with crop?!")
                continue
            w = w.round_offsets().round_lengths()
            arr = ds.read(1, window=w, masked=True).astype("float32")
            transform = ds.window_transform(w)
            prof = {"driver": "GTiff", "dtype": "float32", "count": 1, "crs": ds.crs,
                    "transform": transform, "width": arr.shape[1], "height": arr.shape[0],
                    "compress": "lzw", "nodata": np.nan, "tiled": True}
        out = OUT / f"{pid}_ifcrop.tif"
        with rasterio.open(out, "w", **prof) as dst:
            dst.write(arr.filled(np.nan), 1)
        v = arr.compressed()
        print(f"  {pid}: {arr.shape[1]}x{arr.shape[0]} px, valid {v.size/arr.size:.0%}, "
              f"I/F [{v.min():.4f}, {np.median(v):.4f}, {v.max():.4f}], "
              f"{out.stat().st_size/1e6:.0f} MB")
    print(f"\nwrote to {OUT}\nnext: tar cf pilot_crops.tar -C {WORK} pilot_crops  && scp home")


if __name__ == "__main__":
    main()
