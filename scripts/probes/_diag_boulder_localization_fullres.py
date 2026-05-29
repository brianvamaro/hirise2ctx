"""Definitive boulder-localization check: overlay the BoulderNet polygons on the
FULL-RESOLUTION HiRISE (native ~0.25-0.5 m/px) at a tight zoom, where individual
boulders are clearly visible. Uses a trusted-prj image so the source shapefile CRS ==
the native-window CRS (no SP1 reprojection in play) -- the cleanest possible test that the
polygons sit on the actual boulders.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from src import manifest as M
from src.config import load_config
from src.detections import drop_null_geometries
from src.hirise_imagery import ensure_jp2_local, read_native_window

cfg = load_config(REPO_ROOT / "config_v2.yaml")
mdf = M.load_manifest(cfg.manifest_path).set_index("ObsId")
OBS = sys.argv[1] if len(sys.argv) > 1 else "ESP_069669_2220"
WIN_M = 120.0  # tight zoom side, metres


def norm(a):
    a = a.astype(np.float32); m = a > 0
    lo, hi = np.percentile(a[m], (2, 98)) if m.any() else (0.0, 1.0)
    return np.clip((a - lo) / max(hi - lo, 1e-9), 0, 1)


shp = M.find_shapefile(OBS, cfg.detections_root)
g = gpd.read_file(shp)
g, _ = drop_null_geometries(g)            # source HiRISE CRS (trusted-prj here)
jp2 = str(mdf.loc[OBS, "JP2_URL"])
ensure_jp2_local(OBS, jp2, cfg.cache_dir)

# Densest ~WIN_M cell from polygon centroids (in source-CRS metres).
cen = g.geometry.centroid
cx = cen.x.to_numpy(); cy = cen.y.to_numpy()
xed = np.arange(cx.min(), cx.max() + WIN_M, WIN_M)
yed = np.arange(cy.min(), cy.max() + WIN_M, WIN_M)
Hh, _, _ = np.histogram2d(cx, cy, bins=[xed, yed])
ix, iy = np.unravel_index(np.argmax(Hh), Hh.shape)
x0, y0 = xed[ix], yed[iy]
bounds = (x0, y0, x0 + WIN_M, y0 + WIN_M)   # left, bottom, right, top (source CRS)

arr, T, crs = read_native_window(OBS, jp2, bounds, cfg.cache_dir, "loc_check")
mpp = abs(T.a)
# world -> pixel for this native window
a, c, e, f = T.a, T.c, T.e, T.f
g_px = g.geometry.affine_transform([1.0 / a, 0.0, 0.0, 1.0 / e, -c / a, -f / e])
H, W = arr.shape
inwin = g_px.cx[0:W, 0:H]

fig, ax = plt.subplots(1, 2, figsize=(15, 7.6))
ax[0].imshow(norm(arr), cmap="gray"); ax[0].set_title(f"{OBS}: native HiRISE ({mpp:.2f} m/px), {WIN_M:.0f} m zoom")
ax[1].imshow(norm(arr), cmap="gray")
inwin.plot(ax=ax[1], facecolor="none", edgecolor="red", linewidth=0.8)
ax[1].set_title(f"+ BoulderNet polygons ({inwin.shape[0]} in view) — should sit ON the boulders")
for a_ in ax:
    a_.set_xlim(0, W); a_.set_ylim(H, 0); a_.set_xticks([]); a_.set_yticks([])
fig.suptitle(f"Definitive boulder-localization check (full-res HiRISE) — {OBS}", fontsize=12)
fig.tight_layout()
out = REPO_ROOT / "reports" / "figures" / f"05_boulder_localization_fullres_{OBS}.png"
fig.savefig(out, dpi=140)
print(f"native window {arr.shape} @ {mpp:.3f} m/px  polygons in view: {inwin.shape[0]}")
print(f"wrote {out}")
