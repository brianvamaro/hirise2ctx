"""Check 2 — model-free geometry audit on ALL 38 images (closes rung 1 fully).

For every image: rasterize the boulder polygons at their as-applied positions
(translate by the migrated coreg shift, replicating Stage 4), build roughness
maps (smoothed boulder density vs CTX texture energy), phase-correlate inside
the densest fully-covered 512-px window. Post-fix expectation: residual ~0 px.
Images without correlation lock (low peak) are reported as no-lock, not as
geometry failures.

Writes scripts/probes/_w1_geometry_audit_all38.csv.
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features as rfeatures
from scipy.ndimage import gaussian_filter, uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.coregister import phase_correlate_translation

CACHE = Path("cache_v2")
WIN = 512
OUT = Path("scripts/probes/_w1_geometry_audit_all38.csv")


def texture_energy(arr, inner=2.0, outer=4.0):
    hp = arr - gaussian_filter(arr, inner)
    return gaussian_filter(np.abs(hp), outer)


def densest_window(density, valid, win):
    ok = uniform_filter(valid.astype(np.float32), win) > 0.999
    score = uniform_filter(density, win)
    score[~ok] = -np.inf
    if not np.isfinite(score).any():
        return None
    r_c, c_c = np.unravel_index(np.argmax(score), score.shape)
    r0 = max(0, min(density.shape[0] - win, r_c - win // 2))
    c0 = max(0, min(density.shape[1] - win, c_c - win // 2))
    return r0, c0


rows = []
for f in sorted((CACHE / "reprojected_detections").glob("*.gpkg")):
    obs = f.stem
    coreg_f = CACHE / "coregistration" / f"{obs}.json"
    ctx_f = CACHE / "ctx_windows" / f"{obs}.tif"
    mask_f = CACHE / "ctx_windows" / f"{obs}_hirise_mask.tif"
    if not (coreg_f.exists() and ctx_f.exists() and mask_f.exists()):
        continue
    shift = json.loads(coreg_f.read_text())
    assert shift.get("y_sign_fix_applied"), f"{obs}: cache not migrated?!"
    dy_m, dx_m = shift["shift_m"]["dy"], shift["shift_m"]["dx"]

    with rasterio.open(ctx_f) as src:
        ctx = src.read(1).astype(np.float32)
        transform = src.transform
    with rasterio.open(mask_f) as src:
        cover = src.read(1).astype(bool)

    gdf = gpd.read_file(f)
    geoms = gdf.geometry.translate(xoff=dx_m, yoff=dy_m)  # as Stage 4 applies it
    ras = rfeatures.rasterize(((g, 1) for g in geoms), out_shape=ctx.shape,
                              transform=transform, all_touched=True, dtype="uint8")
    dens = gaussian_filter(ras.astype(np.float32), 4.0)
    valid = cover & (ctx > 0)
    loc = densest_window(dens, valid, WIN)
    if loc is None:
        rows.append(dict(obs_id=obs, resid_dy_px=np.nan, resid_dx_px=np.nan,
                         peak=np.nan, note="no fully-covered window"))
        print(f"{obs}: no window")
        continue
    r0, c0 = loc
    sl = (slice(r0, r0 + WIN), slice(c0, c0 + WIN))
    dy, dx, peak = phase_correlate_translation(texture_energy(ctx)[sl], dens[sl])
    lock = peak >= 0.15
    rows.append(dict(obs_id=obs, resid_dy_px=dy, resid_dx_px=dx, peak=peak,
                     note="" if lock else "no-lock"))
    print(f"{obs}: residual (dy={dy:+.1f}, dx={dx:+.1f}) px, peak {peak:.3f}"
          f"{'' if lock else '  [no-lock: unreliable]'}")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
locked = df[df.peak >= 0.15]
print(f"\n{len(locked)}/{len(df)} images with lock (peak>=0.15); "
      f"|residual| median dy={locked.resid_dy_px.abs().median():.2f} px, "
      f"dx={locked.resid_dx_px.abs().median():.2f} px; "
      f"max |dy|={locked.resid_dy_px.abs().max():.2f} px")
print(f"wrote {OUT}")
