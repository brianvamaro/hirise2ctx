"""Side panel: BoulderNet polygons overlaid on the CTX window for the
same two ObsIds used in the binary rich/poor figure.

Each ObsId has 10^5 - 10^6 boulder polygons; we plot their centroids as a
density of tiny semi-transparent dots so that the overall spatial pattern
of boulder presence is visible against the CTX background. Where many
boulders cluster the colour saturates; sparse regions stay close to CTX
grey.

Output: reports/figures/modeling_slim_boulders_on_ctx.png
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
CACHE_CTX = ROOT / "cache_v2" / "ctx_windows"
CACHE_POLYS = ROOT / "cache_v2" / "reprojected_detections"
FIG = ROOT / "reports" / "figures"

GOOD_OBS = "ESP_053989_2260"
BAD_OBS = "ESP_046328_2180"
BOULDER_COLOR = "#e63946"  # red dots


def render(ax, obs_id: str, label_prefix: str):
    src = rasterio.open(CACHE_CTX / f"{obs_id}.tif")
    ctx = src.read(1).astype(np.float32)
    p2, p98 = np.percentile(ctx[ctx > 0], (2, 98))
    extent = (src.bounds.left, src.bounds.right,
              src.bounds.bottom, src.bounds.top)
    src.close()

    polys = gpd.read_file(CACHE_POLYS / f"{obs_id}.gpkg")
    cents = polys.geometry.centroid
    xs = cents.x.to_numpy()
    ys = cents.y.to_numpy()
    n_polys = len(polys)

    # Clip to CTX extent
    mask = ((xs >= extent[0]) & (xs <= extent[1])
            & (ys >= extent[2]) & (ys <= extent[3]))
    xs = xs[mask]
    ys = ys[mask]

    ax.imshow(ctx, cmap="gray", vmin=p2, vmax=p98, extent=extent,
              origin="upper", interpolation="nearest")
    ax.scatter(xs, ys, s=0.4, c=BOULDER_COLOR, alpha=0.18, linewidths=0,
               rasterized=True)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_title(f"{label_prefix}: {obs_id}\n"
                 f"{n_polys:,} BoulderNet polygons on CTX",
                 fontsize=10)
    ax.set_xlabel("Eastings (m)")
    ax.set_ylabel("Northings (m)")
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.tick_params(labelsize=7)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    render(axes[0], GOOD_OBS, "GOOD")
    render(axes[1], BAD_OBS, "ANTI-SIGNAL")
    fig.suptitle(
        "BoulderNet detections overlaid on CTX for the two exemplar images\n"
        "Each red dot is the centroid of one detected boulder polygon "
        "(0.4 px @ alpha 0.18); dense clusters appear saturated red.",
        y=1.02)
    fig.tight_layout()
    out = FIG / "modeling_slim_boulders_on_ctx.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
