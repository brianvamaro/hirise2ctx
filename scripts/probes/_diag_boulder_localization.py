"""Verify boulders are correctly located: a zoomed visual overlay (polygons on the HiRISE
they were detected on, and on CTX after the co-registration shift, exactly as Stage 4
rasterizes them) + a fast centroid gate across all images.

Fast centroid = area-weighted... actually just the mean of per-polygon centroids
(vectorised; no union_all, which is far too slow on 100k-700k-polygon images). Compared in
target-CRS metres against the manifest centre projected into the same CRS. A coarse gate
that rules out CRS / local-radius gross errors; fine placement is the overlay.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import CRS, Transformer

from src import manifest as M
from src.config import load_config
from src.coregister import load_shift, warp_hirise_to_ctx_grid
from src.ctx_retrieve import CTX_WINDOWS_SUBDIR
from src.detections import load_reprojected

cfg = load_config(REPO_ROOT / "config_v2.yaml")
mdf = M.load_manifest(cfg.manifest_path)
EXCLUDED = {"ESP_046803_2325"}
ZOOM_OBS = sys.argv[1] if len(sys.argv) > 1 else "ESP_069669_2220"


def norm(a):
    a = a.astype(np.float32); m = a > 0
    lo, hi = np.percentile(a[m], (2, 98)) if m.any() else (0.0, 1.0)
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


# ---- 1. Zoomed overlay FIRST (the convincing artifact) -------------------------------
obs = ZOOM_OBS
ctx_tif = cfg.cache_dir / CTX_WINDOWS_SUBDIR / f"{obs}.tif"
with rasterio.open(ctx_tif) as ds_:
    ctx = ds_.read(1).astype(np.float32)
    T = ds_.transform
row = mdf.set_index("ObsId").loc[obs]
hi, _, _ = warp_hirise_to_ctx_grid(obs, jp2_url=str(row["JP2_URL"]),
                                   cache_dir=cfg.cache_dir, ctx_window_tif=ctx_tif)
gdf = load_reprojected(obs, cfg.cache_dir)
coreg = load_shift(obs, cfg.cache_dir)
dx_m, dy_m = coreg["shift_m"]["dx"], coreg["shift_m"]["dy"]

a, c, e, f = T.a, T.c, T.e, T.f
to_px = lambda g: g.affine_transform([1.0 / a, 0.0, 0.0, 1.0 / e, -c / a, -f / e])
gdf_px_unshift = to_px(gdf.geometry)                                  # aligns with HiRISE-on-CTX
gdf_px_shift = to_px(gdf.geometry.translate(xoff=dx_m, yoff=dy_m))    # aligns with CTX (Stage-4)

cen = gdf_px_unshift.centroid
cc = np.clip(cen.x.to_numpy().astype(int), 0, ctx.shape[1] - 1)
rr = np.clip(cen.y.to_numpy().astype(int), 0, ctx.shape[0] - 1)
Z = 300
H2d, _, _ = np.histogram2d(rr, cc, bins=[max(1, ctx.shape[0] // Z), max(1, ctx.shape[1] // Z)])
bi, bj = np.unravel_index(np.argmax(H2d), H2d.shape)
r0 = min(bi * Z, max(0, ctx.shape[0] - Z)); c0 = min(bj * Z, max(0, ctx.shape[1] - Z))
r1, c1 = r0 + Z, c0 + Z

fig, ax = plt.subplots(1, 2, figsize=(15, 7.6))
ax[0].imshow(norm(hi[r0:r1, c0:c1]), cmap="gray", extent=[c0, c1, r1, r0])
gdf_px_unshift.cx[c0:c1, r0:r1].plot(ax=ax[0], facecolor="none", edgecolor="red", linewidth=0.6)
ax[0].set_title(f"{obs}: UNSHIFTED polygons on HiRISE (detected here)\nshould enclose the boulder features")
ax[1].imshow(norm(ctx[r0:r1, c0:c1]), cmap="gray", extent=[c0, c1, r1, r0])
gdf_px_shift.cx[c0:c1, r0:r1].plot(ax=ax[1], facecolor="none", edgecolor="red", linewidth=0.6)
ax[1].set_title(f"SHIFTED polygons on CTX (Stage-4 placement, |shift|={coreg['shift_m']['magnitude']:.0f} m)\nshould track the CTX texture")
for a_ in ax:
    a_.set_xlim(c0, c1); a_.set_ylim(r1, r0); a_.set_xticks([]); a_.set_yticks([])
fig.suptitle(f"Boulder localization check — {obs}  ({Z}px = {Z*abs(a):.0f} m zoom on the densest region)", fontsize=12)
fig.tight_layout()
outfig = REPO_ROOT / "reports" / "figures" / f"05_boulder_localization_{obs}.png"
fig.savefig(outfig, dpi=130)
print(f"zoom rows[{r0}:{r1}] cols[{c0}:{c1}]  polygons in view: {int(gdf_px_unshift.cx[c0:c1, r0:r1].shape[0])}")
print(f"wrote {outfig}")


# ---- 2. Fast centroid gate across all retained images --------------------------------
print("\n=== centroid residual (mean polygon centroid vs manifest centre, target-CRS metres) ===")
dists = []
for _, r in mdf.iterrows():
    o = str(r["ObsId"])
    if o in EXCLUDED:
        continue
    g = load_reprojected(o, cfg.cache_dir)
    if len(g) == 0:
        continue
    tcrs = CRS.from_user_input(g.crs)
    cn = g.geometry.centroid
    cx_m, cy_m = float(cn.x.mean()), float(cn.y.mean())
    fwd = Transformer.from_crs(tcrs.geodetic_crs, tcrs, always_xy=True)
    mx, my = fwd.transform(float(r["CenterLon_180"]), float(r["CenterLat"]))
    dists.append((o, math.hypot(cx_m - mx, cy_m - my) / 1000.0))
d = np.array([x for _, x in dists])
gate = cfg["sanity"]["centroid_max_km"]
worst = sorted(dists, key=lambda t: -t[1])[:3]
print(f"  {len(dists)} images: min={d.min():.1f} median={np.median(d):.1f} max={d.max():.1f} km  (gate {gate} km)")
print(f"  over gate: {[o for o, x in dists if x > gate] or 'none'}")
print(f"  largest 3: {[(o, round(x, 1)) for o, x in worst]}")
