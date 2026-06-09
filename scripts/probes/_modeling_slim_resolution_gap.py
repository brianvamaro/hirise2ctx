"""Build the HiRISE / CTX resolution-gap figure for docs/modeling_slim.md.

Side-by-side two-panel comparison of the same physical 200 m x 200 m
patch on ESP_053989_2260: CTX raster (~5 m/px) on the left and HiRISE
RED panchromatic (~0.5 m/px native on this image) on the right.
HiRISE is reprojected into the CTX coordinate system via WarpedVRT
(with the Stage-1 SP1-corrected source CRS) so both panels show the
exact same physical patch.

The patch is centred on the highest-boulder-count S=64 tile in the
image (ti=483, tj=221, boulder_count=902 in a 320 m tile -- a dense
boulder field).

Output: reports/figures/modeling_slim_resolution_gap.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import colour  # noqa: E402 -- only for the corrected_source_crs helper
CTX_PATH = ROOT / "cache_v2" / "ctx_windows" / "ESP_053989_2260.tif"
HIRISE_PATH = ROOT / "cache" / "hirise_jp2" / "ESP_053989_2260_RED.JP2"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

OBS_ID = "ESP_053989_2260"
# Centre of the densest S=64 tile (ti=483, tj=221) in CTX CRS
PATCH_HALF_M = 100.0   # 200 m x 200 m window


def main():
    ctx = rasterio.open(CTX_PATH)
    print(f"CTX: shape={ctx.shape}, crs={ctx.crs}, bounds={ctx.bounds}")

    # Densest S=64 tile centre, from dataset_v2/labels parquet
    labels = pd.read_parquet(ROOT / "dataset_v2" / "labels" / f"{OBS_ID}.parquet")
    s64 = labels[labels["scale_idx"] == 3].sort_values("boulder_count", ascending=False)
    top = s64.iloc[0]
    cx = (top["xmin"] + top["xmax"]) / 2
    cy = (top["ymin"] + top["ymax"]) / 2
    print(f"Patch centre (CTX CRS): ({cx:.1f}, {cy:.1f})")
    print(f"Top tile: ti={int(top['ti'])} tj={int(top['tj'])} "
          f"boulder_count={int(top['boulder_count'])} "
          f"fractional_area={top['fractional_area']:.4f}")

    # Window bounds in CTX CRS
    win_bounds = (cx - PATCH_HALF_M, cy - PATCH_HALF_M,
                  cx + PATCH_HALF_M, cy + PATCH_HALF_M)
    win = from_bounds(*win_bounds, transform=ctx.transform)
    ctx_data = ctx.read(1, window=win)
    print(f"CTX patch shape: {ctx_data.shape}  "
          f"(extent = {2*PATCH_HALF_M:.0f} m at {abs(ctx.transform.a):.1f} m/px)")

    # HiRISE RED.JP2 has the SP1 bug -- its bounds + transform are correct
    # for the SP1-corrected CRS (the Stage 1 sidecar value), but the file's
    # own CRS metadata is wrong. WarpedVRT lets us open HiRISE with the
    # corrected source CRS applied and the destination in CTX CRS, so we
    # can read HiRISE pixels for the exact 200 m x 200 m physical patch
    # the CTX panel shows (both panels in CTX CRS, both physically square).
    corrected_crs = colour.corrected_source_crs(OBS_ID, ROOT / "cache_v2")
    if corrected_crs is None:
        raise RuntimeError(f"No SP1-corrected source CRS for {OBS_ID} -- "
                           f"run Stage 1 first.")

    hirise_src = rasterio.open(HIRISE_PATH)
    target_res = abs(hirise_src.transform.a)   # 0.5 m/px native
    print(f"HiRISE: shape={hirise_src.shape}, native resolution="
          f"{target_res:.2f} m/px")
    with WarpedVRT(hirise_src, src_crs=corrected_crs, crs=ctx.crs,
                   resampling=Resampling.bilinear) as vrt:
        print(f"  HiRISE reprojected into CTX CRS via WarpedVRT: "
              f"shape={vrt.shape}, bounds={vrt.bounds}")
        # Read HiRISE for the same window the CTX panel uses
        vrt_win = from_bounds(*win_bounds, transform=vrt.transform)
        hirise_data = vrt.read(1, window=vrt_win,
                               out_shape=(int(round(2*PATCH_HALF_M / target_res)),
                                          int(round(2*PATCH_HALF_M / target_res))),
                               resampling=Resampling.bilinear)
    hirise_src.close()
    print(f"  HiRISE patch shape: {hirise_data.shape}")
    hirise_extent = (win_bounds[0], win_bounds[2],
                     win_bounds[1], win_bounds[3])

    # Stretch both rasters for display: percentile clip (tighter for
    # HiRISE so individual boulders pop against the surrounding
    # regolith brightness)
    def stretch(arr, lo_p=2, hi_p=98):
        valid = arr[(arr > 0) & np.isfinite(arr)]
        if not valid.size:
            return arr
        lo, hi = np.percentile(valid, (lo_p, hi_p))
        return np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)

    ctx_disp = stretch(ctx_data.astype(np.float32), 2, 98)
    # Use a tighter percentile range on HiRISE so the boulder shadows pop
    hirise_disp = stretch(hirise_data.astype(np.float32), 5, 95)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

    panels = [
        (axes[0], ctx_disp,
         (win_bounds[0], win_bounds[2], win_bounds[1], win_bounds[3]),
         "CTX (5 m / pixel)"),
        (axes[1], hirise_disp, hirise_extent,
         f"HiRISE RED ({target_res:.2g} m / pixel)"),
    ]

    for ax, raster, extent, title in panels:
        ax.imshow(raster, cmap="gray", extent=extent, origin="upper",
                  interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        # 50 m scale bar -- 10 m from the left + 10 m from the bottom of
        # the visible extent
        x0, x1, y0, y1 = extent
        sb_x = x0 + 10
        sb_y = y0 + 10
        ax.plot([sb_x, sb_x + 50], [sb_y, sb_y], color="cyan", linewidth=3,
                solid_capstyle="butt")
        ax.text(sb_x + 25, sb_y + 6, "50 m", color="cyan", fontsize=9,
                ha="center", va="bottom", weight="bold")

    # No suptitle -- everything contextual lives in the markdown caption.
    fig.tight_layout()
    out = FIG / "modeling_slim_resolution_gap.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nWrote {out}")
    ctx.close()


if __name__ == "__main__":
    main()
