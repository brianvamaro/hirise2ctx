"""W1 Rung 1c — direct measurement of label-vs-CTX displacement.

Model-free test of the coreg sign-error hypothesis (coregister.py:383 misses
the row->world-y flip). For each test image:

1. Rasterize the reprojected boulder polygons onto the CTX window grid at
   (a) nominal positions and (b) as-applied positions (translate by cached
   shift_m, replicating labeling._apply_coreg_shift).
2. Build comparable "roughness" maps: smoothed boulder density vs CTX local
   texture energy (boulder fields read as rough/mottled texture at 5 m/px).
3. Phase-correlate (reference=CTX texture, moving=boulder density) inside the
   densest 512-px window fully covered by HiRISE.

Expected measured row-correction for the boulder field:
  - nominal:    ~ cached dy_px      (labels share HiRISE's nominal georef)
  - as-applied: ~ 2 x cached dy_px  if the y sign was inverted (the bug)
                ~ 0                 if the applied correction was right

Also writes a 3-panel overlay figure (nominal / applied / sign-corrected) to
reports/figures/ for the notebook-18 record.
"""
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio import features as rfeatures
from scipy.ndimage import gaussian_filter, uniform_filter

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.coregister import phase_correlate_translation

CACHE = Path("cache_v2")
FIGDIR = Path("reports/figures")
WIN = 512
TEST_IMAGES = ["ESP_042964_2160", "ESP_066634_2210", "ESP_017355_2260", "ESP_069763_2235"]


def texture_energy(arr, inner=2.0, outer=4.0):
    """Local high-frequency energy: |x - lowpass(x)| smoothed — a roughness map."""
    hp = arr - gaussian_filter(arr, inner)
    return gaussian_filter(np.abs(hp), outer)


def densest_window(density, valid, win):
    """Top-left corner of the win x win box with max density, fully valid."""
    ok = uniform_filter(valid.astype(np.float32), win) > 0.999
    score = uniform_filter(density, win)
    score[~ok] = -np.inf
    # uniform_filter is centered; convert center index to corner
    r_c, c_c = np.unravel_index(np.argmax(score), score.shape)
    r0 = max(0, min(density.shape[0] - win, r_c - win // 2))
    c0 = max(0, min(density.shape[1] - win, c_c - win // 2))
    return r0, c0


results = []
for obs in TEST_IMAGES:
    shift = json.loads((CACHE / "coregistration" / f"{obs}.json").read_text())
    dy_m, dx_m = shift["shift_m"]["dy"], shift["shift_m"]["dx"]
    dy_px_cached, dx_px_cached = shift["shift_px"]["dy"], shift["shift_px"]["dx"]

    with rasterio.open(CACHE / "ctx_windows" / f"{obs}.tif") as src:
        ctx = src.read(1).astype(np.float32)
        transform = src.transform
    with rasterio.open(CACHE / "ctx_windows" / f"{obs}_hirise_mask.tif") as src:
        cover = src.read(1).astype(bool)

    gdf = gpd.read_file(CACHE / "reprojected_detections" / f"{obs}.gpkg")
    variants = {
        "nominal": gdf.geometry,
        "applied": gdf.geometry.translate(xoff=dx_m, yoff=dy_m),
        "corrected": gdf.geometry.translate(xoff=dx_m, yoff=-dy_m),
    }

    dens = {}
    for name, geoms in variants.items():
        ras = rfeatures.rasterize(
            ((g, 1) for g in geoms), out_shape=ctx.shape, transform=transform,
            all_touched=True, dtype="uint8",
        )
        dens[name] = gaussian_filter(ras.astype(np.float32), 4.0)

    ctx_tex = texture_energy(ctx)
    valid = cover & (ctx > 0)
    r0, c0 = densest_window(dens["nominal"], valid, WIN)
    sl = (slice(r0, r0 + WIN), slice(c0, c0 + WIN))

    row = {"obs_id": obs, "cached_dy_px": dy_px_cached, "cached_dx_px": dx_px_cached}
    for name in ("nominal", "applied"):
        dy, dx, peak = phase_correlate_translation(ctx_tex[sl], dens[name][sl])
        row[f"{name}_dy"] = dy
        row[f"{name}_dx"] = dx
        row[f"{name}_peak"] = peak
    results.append(row)
    print(
        f"{obs}: cached shift_px (dy={dy_px_cached:+.1f}, dx={dx_px_cached:+.1f}) | "
        f"measured nominal (dy={row['nominal_dy']:+.1f}, dx={row['nominal_dx']:+.1f}, "
        f"peak {row['nominal_peak']:.3f}) | applied (dy={row['applied_dy']:+.1f}, "
        f"dx={row['applied_dx']:+.1f}, peak {row['applied_peak']:.3f})"
    )

    # 3-panel overlay figure on a 256-px sub-crop for visibility
    sub = (slice(r0 + WIN // 4, r0 + WIN // 4 + 256), slice(c0 + WIN // 4, c0 + WIN // 4 + 256))
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    v = ctx[sub]
    for ax, name in zip(axes, ("nominal", "applied", "corrected")):
        ax.imshow(v, cmap="gray", vmin=np.percentile(v, 2), vmax=np.percentile(v, 98))
        d = dens[name][sub]
        ax.contour(d, levels=[np.percentile(dens[name][dens[name] > 0], 75)],
                   colors="red", linewidths=0.8)
        ax.set_title(f"{name} (yoff={'0' if name == 'nominal' else f'{dy_m:+.0f}' if name == 'applied' else f'{-dy_m:+.0f}'} m)")
        ax.set_axis_off()
    fig.suptitle(f"{obs} — boulder-density contours over CTX (rung 1c)")
    out = FIGDIR / f"w1_rung1c_{obs}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"  figure -> {out}")

import pandas as pd

df = pd.DataFrame(results)
df.to_csv("scripts/probes/_w1_label_ctx_displacement.csv", index=False)
print("\nSummary (sign-error predicts applied_dy ~ 2 x cached_dy_px; correct predicts ~0):")
print(df.to_string(float_format=lambda x: f"{x:+.2f}", index=False))
