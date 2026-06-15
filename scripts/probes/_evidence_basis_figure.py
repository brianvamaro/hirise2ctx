"""Basis figure for docs/model_evidence.md: CTX carries the boulder signal.

Two S=32 (160 m) tiles from the SAME image (ESP_053989_2260, so illumination /
season / CTX source are identical) -- one boulder-RICH, one boulder-POOR -- each
shown as HiRISE (0.5 m/px, individual boulders + their shadows resolved, with the
BoulderNet ground-truth polygons outlined) beside the co-located CTX (5 m/px, what
the model actually sees). The rich tile's CTX is visibly rougher / brighter-speckled
than the poor tile's smooth regolith: the per-tile texture difference the embedding
reads.

Output: reports/figures/model_evidence_basis_hirise_ctx.png
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
from src import colour, detections  # noqa: E402

OBS = "ESP_053989_2260"
CTX_PATH = ROOT / "cache_v2" / "ctx_windows" / f"{OBS}.tif"
HIRISE_PATH = ROOT / "cache" / "hirise_jp2" / f"{OBS}_RED.JP2"
FIG = ROOT / "reports" / "figures"

# tiles chosen by _evidence_select_exemplars.py
RICH = dict(ti=975, tj=431, bounds=(-2776225.8, 2689026.3, -2776065.8, 2689186.3),
            bc=250, fa=0.1304)
POOR = dict(ti=962, tj=414, bounds=(-2778945.8, 2691106.3, -2778785.8, 2691266.3),
            bc=0, fa=0.0)


def stretch(arr, lo_p, hi_p):
    valid = arr[(arr > 0) & np.isfinite(arr)]
    if not valid.size:
        return arr
    lo, hi = np.percentile(valid, (lo_p, hi_p))
    return np.clip((arr - lo) / max(hi - lo, 1e-9), 0, 1)


def raw_stats(gdf, cx, cy, xmin, ymin, xmax, ymax):
    """Boulders whose centroid lies in the tile, from the RAW reprojected
    detections (the same frame as the HiRISE shown). The label parquet grid
    carries the co-registration shift, so for a self-consistent figure we count
    and outline the raw detections directly rather than trust the label cell."""
    cand = gdf.cx[xmin:xmax, ymin:ymax]
    c = cand.geometry.centroid
    sub = cand[(c.x >= xmin) & (c.x < xmax) & (c.y >= ymin) & (c.y < ymax)]
    area_frac = float(sub.geometry.area.sum()) / ((xmax - xmin) * (ymax - ymin))
    return sub, len(sub), area_frac


def main():
    ctx = rasterio.open(CTX_PATH)
    corrected_crs = colour.corrected_source_crs(OBS, ROOT / "cache_v2")
    gdf = detections.load_reprojected(OBS, ROOT / "cache_v2")
    print(f"CTX crs={ctx.crs}; {len(gdf)} boulder polygons loaded")

    # Reselect a genuinely raw-empty POOR tile with full CTX coverage, near the
    # rich tile, by scanning the S=32 label grid for min raw centroid count.
    lab = pd.read_parquet(ROOT / "dataset_v2" / "labels" / f"{OBS}.parquet")
    s32 = lab[lab["scale_idx"] == 2]
    cx_g = gdf.geometry.centroid.x.to_numpy()
    cy_g = gdf.geometry.centroid.y.to_numpy()
    best = None
    for _, t in s32.iterrows():
        xa, ya, xb, yb = t["xmin"], t["ymin"], t["xmax"], t["ymax"]
        nraw = int(((cx_g >= xa) & (cx_g < xb) & (cy_g >= ya) & (cy_g < yb)).sum())
        if nraw > 0:
            continue
        cwin = from_bounds(xa, ya, xb, yb, transform=ctx.transform)
        cdata = ctx.read(1, window=cwin)
        if cdata.size == 0 or (cdata <= 0).mean() > 0.02:   # require full coverage
            continue
        # prefer a tile close to the rich one (same neighbourhood, same lighting)
        d = abs((xa + xb) / 2 - (RICH["bounds"][0] + RICH["bounds"][2]) / 2) \
            + abs((ya + yb) / 2 - (RICH["bounds"][1] + RICH["bounds"][3]) / 2)
        if best is None or d < best[0]:
            best = (d, (xa, ya, xb, yb))
    if best is not None:
        POOR["bounds"] = best[1]
        print(f"POOR tile reselected (raw-empty): {best[1]}")

    hirise_src = rasterio.open(HIRISE_PATH)
    hres = abs(hirise_src.transform.a)

    fig, axes = plt.subplots(2, 2, figsize=(8.4, 9.0))
    row_specs = [("BOULDER-RICH tile", RICH, axes[0]),
                 ("BOULDER-POOR tile", POOR, axes[1])]

    with WarpedVRT(hirise_src, src_crs=corrected_crs, crs=ctx.crs,
                   resampling=Resampling.bilinear) as vrt:
        for label, spec, (ax_h, ax_c) in row_specs:
            xmin, ymin, xmax, ymax = spec["bounds"]
            extent = (xmin, xmax, ymin, ymax)
            side = xmax - xmin  # 160 m

            # HiRISE window for the exact tile footprint
            n = int(round(side / hres))
            hwin = from_bounds(xmin, ymin, xmax, ymax, transform=vrt.transform)
            hdata = vrt.read(1, window=hwin, out_shape=(n, n),
                             resampling=Resampling.bilinear)
            # CTX window for the same footprint
            cwin = from_bounds(xmin, ymin, xmax, ymax, transform=ctx.transform)
            cdata = ctx.read(1, window=cwin)

            ax_h.imshow(stretch(hdata.astype(np.float32), 4, 96), cmap="gray",
                        extent=extent, origin="upper", interpolation="nearest")
            # ground-truth boulders (raw reprojected detections, centroid-in-tile)
            sub, n_b, fa = raw_stats(gdf, None, None, xmin, ymin, xmax, ymax)
            for geom in sub.geometry:
                if geom.is_empty:
                    continue
                gs = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
                for poly in gs:
                    x, y = poly.exterior.xy
                    ax_h.plot(x, y, color="#39ff14", lw=0.6, alpha=0.9)
            ax_h.set_title(f"{label} — HiRISE 0.5 m/px\n{n_b} boulders "
                           f"(BoulderNet); area fraction {fa*100:.1f}%",
                           fontsize=9.5)

            ax_c.imshow(stretch(cdata.astype(np.float32), 2, 98), cmap="gray",
                        extent=extent, origin="upper", interpolation="nearest")
            ax_c.set_title(f"{label} — CTX 5 m/px\n(the model's input)", fontsize=9.5)

            for ax in (ax_h, ax_c):
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
                # 50 m scale bar
                ax.plot([xmin + 12, xmin + 62], [ymin + 12, ymin + 12],
                        color="cyan", lw=3, solid_capstyle="butt")
                ax.text(xmin + 37, ymin + 17, "50 m", color="cyan", fontsize=8,
                        ha="center", va="bottom", weight="bold")

    hirise_src.close(); ctx.close()
    fig.suptitle("CTX carries the boulder signal: same 160 m tile, two resolutions\n"
                 f"({OBS}; identical illumination & CTX source for both tiles)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIG / "model_evidence_basis_hirise_ctx.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
