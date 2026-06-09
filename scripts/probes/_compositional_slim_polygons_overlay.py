"""Build the 'show the data' figure for docs/compositional_slim.md.

Shows one exemplar HiRISE COLOR.JP2 swath as an IRB false-colour composite,
with BoulderNet detection polygon centroids overlaid as small dots so the
spatial distribution of detected boulders is visible against the colour
imagery.

Exemplar: ESP_046959_2225 -- a "Deposit!"-flagged mesas image in the
composition_residual attribution category. Mid-cohort latitude (42 N), well
within the colour-cohort norm.

Output: reports/figures/compositional_slim_polygons_on_color.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.warp import Resampling

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src import colour  # noqa: E402

CACHE = REPO_ROOT / "cache_v2"
POLYS = CACHE / "reprojected_detections"
FIG = REPO_ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

OBS_ID = "ESP_046959_2225"   # Deposit! anchor, composition_residual exemplar
DECIMATION = 4    # 0.5 m/px native -> ~2 m/px display
BOULDER_COLOR = "#00ffff"   # cyan -- best contrast on Mars dusty red


def read_color_window(obs_id: str, crop_frac: tuple[float, float] = (0.45, 0.55)):
    """Read a CROPPED window of the COLOR.JP2 IRB bands at the decimated scale.

    crop_frac picks an along-track fraction of the swath (defaults to the
    middle 10%, ~1 km along-track) so the figure focuses on a region where
    individual boulders are resolvable rather than the full kilometres-long
    swath.

    Returns (bands_iof, bounds, crs, lbl) where bands_iof is shape (3, H, W)
    in I/F units, ordered IR/RED/BG.
    """
    jp2 = colour.color_jp2_path(CACHE, obs_id)
    lbl = colour.parse_color_lbl(colour.color_lbl_path(CACHE, obs_id))
    crs = colour.corrected_source_crs(obs_id, CACHE)
    with rasterio.open(jp2) as src:
        # Along-track crop in line direction (= y direction in image space)
        nrows = src.height
        ncols = src.width
        row0 = int(crop_frac[0] * nrows)
        row1 = int(crop_frac[1] * nrows)
        window = rasterio.windows.Window(0, row0, ncols, row1 - row0)
        H = window.height // DECIMATION
        W = window.width // DECIMATION
        bands = src.read(
            [colour.BAND_IR, colour.BAND_RED, colour.BAND_BG],
            window=window,
            out_shape=(3, H, W),
            resampling=Resampling.average,
        )
        win_transform = src.window_transform(window)
        sf = win_transform.a * DECIMATION
        left = win_transform.c
        top = win_transform.f
        right = left + W * sf
        bottom = top + H * win_transform.e * DECIMATION
        bounds = (left, right, bottom, top)
    # Valid-pixel mask computed on the RAW bands, before scaling+offset
    # (the offset shifts true-zero nodata up into a positive value, which
    # makes a post-scaling mask useless for distinguishing on- from off-
    # swath pixels).
    valid_mask = np.any(bands > 0, axis=0)
    bands = bands.astype(np.float32) * lbl.scaling_factor + lbl.offset
    cos_i = lbl.cos_incidence
    bands = bands / cos_i
    return bands, bounds, crs, lbl, valid_mask


def stretch_for_display(bands: np.ndarray) -> np.ndarray:
    """Per-band percentile stretch to [0, 1] for display. HiRISE Mars colour is
    naturally low-contrast; we use a moderate clip and no gamma so the
    composite has realistic dim tones rather than the washed-out look that
    aggressive gamma introduces.
    """
    out = np.empty_like(bands, dtype=np.float32)
    for i in range(bands.shape[0]):
        b = bands[i]
        valid = b[(b > 0) & np.isfinite(b)]
        if valid.size == 0:
            out[i] = 0
            continue
        lo, hi = np.percentile(valid, (5, 99.5))
        if hi <= lo:
            out[i] = 0
            continue
        out[i] = np.clip((b - lo) / (hi - lo), 0, 1)
    rgb = np.transpose(out, (1, 2, 0))
    return rgb


def main():
    print(f"Loading {OBS_ID} COLOR.JP2 at decimation {DECIMATION}x, middle 10% swath ...")
    bands, bounds, crs, lbl, valid_mask = read_color_window(OBS_ID, crop_frac=(0.45, 0.55))
    print(f"  bands shape = {bands.shape}, "
          f"map_scale_native = {lbl.map_scale_mpp} m/px -> "
          f"display ~{lbl.map_scale_mpp * DECIMATION:.2f} m/px")
    print(f"  bounds (l, r, b, t) = {bounds}")
    print(f"  IR/RED/BG I/F medians = "
          f"{np.nanmedian(bands[0]):.3f} / {np.nanmedian(bands[1]):.3f} / "
          f"{np.nanmedian(bands[2]):.3f}")

    rgb = stretch_for_display(bands)
    print(f"  display rgb shape = {rgb.shape}")

    # Use the raw-band valid mask for transparency too, so nodata regions
    # render transparent rather than black.
    rgba = np.dstack([rgb, valid_mask.astype(np.float32)])

    print(f"Loading polygons from {POLYS / (OBS_ID + '.gpkg')} ...")
    polys = gpd.read_file(POLYS / f"{OBS_ID}.gpkg")
    print(f"  polygon CRS: {polys.crs.to_string() if polys.crs else 'unknown'}")
    print(f"  colour CRS:  {crs.to_string() if crs else 'unknown'}")
    # The polygon GPKG and the SP1-corrected colour CRS use different
    # equirectangular projections (different central meridian / standard
    # parallel). Reproject the polygons into the colour CRS before
    # overlaying.
    polys_in_color_crs = polys.to_crs(crs)
    cents = polys_in_color_crs.geometry.centroid
    xs = cents.x.to_numpy()
    ys = cents.y.to_numpy()
    print(f"  {len(polys):,} polygons; centroid x range {xs.min():.1f} - "
          f"{xs.max():.1f}, y range {ys.min():.1f} - {ys.max():.1f}")

    # Restrict centroids to pixels with actual colour data. The polygon
    # GPKG covers the full HiRISE panchromatic footprint, which is wider
    # than the COLOR.JP2 swath; off-swath polygons land in nodata pixels
    # that contribute nothing to the analysis, so we drop them here.
    left, right, bottom, top = bounds
    H_mask, W_mask = valid_mask.shape
    cols_idx = ((xs - left) / (right - left) * W_mask).astype(int)
    rows_idx = ((top - ys) / (top - bottom) * H_mask).astype(int)
    in_bounds = (
        (rows_idx >= 0) & (rows_idx < H_mask)
        & (cols_idx >= 0) & (cols_idx < W_mask)
    )
    in_swath = np.zeros_like(in_bounds, dtype=bool)
    in_swath[in_bounds] = valid_mask[rows_idx[in_bounds], cols_idx[in_bounds]]
    xs = xs[in_swath]
    ys = ys[in_swath]
    print(f"  polygons within colour swath: {in_swath.sum():,}")

    # Figure: full colour swath as IRB false colour, polygon centroids
    # scattered on top.
    H, W = rgb.shape[:2]
    # Pick an aspect-preserving figure size
    aspect = W / max(H, 1)
    fig_h = 6.5
    fig_w = max(8.0, fig_h * aspect)
    # Tighten display extent to the valid-pixel column range so the figure
    # zooms to just the colour swath rather than the wider raster bbox.
    valid_cols = np.where(valid_mask.any(axis=0))[0]
    if valid_cols.size:
        col_left, col_right = valid_cols.min(), valid_cols.max() + 1
        x_left = bounds[0] + col_left * (bounds[1] - bounds[0]) / valid_mask.shape[1]
        x_right = bounds[0] + col_right * (bounds[1] - bounds[0]) / valid_mask.shape[1]
    else:
        x_left, x_right = bounds[0], bounds[1]

    valid_rows = np.where(valid_mask.any(axis=1))[0]
    if valid_rows.size:
        row_top, row_bot = valid_rows.min(), valid_rows.max() + 1
        y_top = bounds[3] - row_top * (bounds[3] - bounds[2]) / valid_mask.shape[0]
        y_bot = bounds[3] - row_bot * (bounds[3] - bounds[2]) / valid_mask.shape[0]
    else:
        y_top, y_bot = bounds[3], bounds[2]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(rgba, extent=bounds, origin="upper", interpolation="nearest")
    ax.scatter(xs, ys, s=18, c=BOULDER_COLOR, alpha=0.9,
               linewidths=0.4, edgecolors="black", rasterized=True)
    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bot, y_top)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out = FIG / "compositional_slim_polygons_on_color.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
