"""Preview the representative E8_N44 crop for the A1 payoff test — confirm it spans multiple CTX
source frames (so eta^2 is meaningful) before spending inference compute.

Shows: CTX brightness crop + frame outlines | baseline abundance crop + frame outlines, with the
crop box on the full tile for context. Prints how many distinct source frames intersect the crop.
Run: conda run -n geospatial python scripts/striping_a1_crop_preview.py [--r0 R --c0 C --size N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.striping import CTX_ZIP_DIR, MAP_DIR, _inner_tif_name, load_frames, load_raster

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
TILE = "E8_N44"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r0", type=int, default=16000)
    ap.add_argument("--c0", type=int, default=14000)
    ap.add_argument("--size", type=int, default=12000)   # native px (~60 km)
    args = ap.parse_args()
    r0, c0, size = args.r0, args.c0, args.size

    zip_path = CTX_ZIP_DIR / f"{TILE}.zip"
    inner = _inner_tif_name(zip_path)
    with rasterio.open(f"/vsizip/{zip_path.as_posix()}/{inner}") as src:
        tr, H, W = src.transform, src.height, src.width
        win = Window(c0, r0, size, size)
        crop = src.read(1, window=win, out_shape=(1200, 1200)).astype(float)
        cb = rasterio.windows.bounds(win, tr)            # (left, bottom, right, top) metres
    crop[crop == 0] = np.nan
    px_m = tr.a
    print(f"tile {TILE}: {H}x{W} native px; crop r0={r0} c0={c0} size={size} "
          f"(~{size*px_m/1000:.0f} km); coarse pred tiles ~{(size//32)**2//1000}k")

    frames = load_frames(TILE)
    cbox = box(*cb)
    inter = frames[frames.intersects(cbox)]
    print(f"distinct CTX source frames intersecting the crop: {len(inter)} of {len(frames)}")

    ab = load_raster(MAP_DIR / f"{TILE}_abundance.tif")
    with rasterio.open(MAP_DIR / f"{TILE}_abundance.tif") as ds:
        abtr = ds.transform
    # crop in coarse (160 m) pixel coords
    ac0 = int((cb[0] - abtr.c) / abtr.a); ar0 = int((cb[3] - abtr.f) / abtr.e)
    asz = int(size * px_m / abtr.a)
    ab_crop = ab[ar0:ar0 + asz, ac0:ac0 + asz]

    ext = [cb[0], cb[2], cb[1], cb[3]]
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    # context: full tile abundance + crop box
    with rasterio.open(MAP_DIR / f"{TILE}_abundance.tif") as ds:
        fb = ds.bounds
    ax[0].imshow(ab, cmap="magma", vmax=np.nanpercentile(ab, 99),
                 extent=[fb.left, fb.right, fb.bottom, fb.top], origin="upper")
    ax[0].add_patch(Rectangle((cb[0], cb[1]), cb[2] - cb[0], cb[3] - cb[1], fill=False,
                              edgecolor="cyan", lw=2))
    ax[0].set_title(f"{TILE} full-tile abundance + crop box")
    # crop CTX + frames
    ax[1].imshow(crop, cmap="gray", extent=ext, origin="upper",
                 vmin=np.nanpercentile(crop, 2), vmax=np.nanpercentile(crop, 98))
    inter.boundary.plot(ax=ax[1], edgecolor="lime", linewidth=0.8)
    ax[1].set_xlim(ext[0], ext[1]); ax[1].set_ylim(ext[2], ext[3])
    ax[1].set_title(f"crop CTX brightness + {len(inter)} source-frame outlines")
    # crop baseline abundance + frames
    ax[2].imshow(ab_crop, cmap="magma", extent=ext, origin="upper",
                 vmax=np.nanpercentile(ab_crop, 99))
    inter.boundary.plot(ax=ax[2], edgecolor="cyan", linewidth=0.8)
    ax[2].set_xlim(ext[0], ext[1]); ax[2].set_ylim(ext[2], ext[3])
    ax[2].set_title("crop baseline abundance + frames\n(blocks should hug frame outlines)")
    fig.tight_layout()
    out = FIG / "striping_a1_crop_preview.png"
    fig.savefig(out, dpi=110)
    print("wrote", out)


if __name__ == "__main__":
    main()
