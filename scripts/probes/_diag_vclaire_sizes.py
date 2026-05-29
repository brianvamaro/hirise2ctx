"""Reprojected boulder size distribution for the vClaire set -> the min_size_m decision.

Equivalent-circle diameter = 2*sqrt(area/pi), area in target-CRS metres (the .gpkg is
already reprojected to the Mars-2000 sphere). Reports per-image percentiles + the
fraction below the current 1.4105 m floor, on a sample spanning the density range, plus
a pooled summary. Also reports score percentiles (the min_confidence lever).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geopandas as gpd
import numpy as np

GPKG_DIR = REPO_ROOT / "cache_v2" / "reprojected_detections"
FLOOR_M = 1.4105
SAMPLE = [
    "ESP_017355_2260",  # 1.1M (densest)
    "ESP_068483_2280",  # 1.06M
    "ESP_045139_2270",  # 244k (mid)
    "ESP_069669_2220",  # 35k  (overlap with v1)
    "ESP_055978_2270",  # 9.6k (sparsest)
]

pooled_diam = []
pooled_score = []
print(f"{'ObsId':<18} {'n':>9}  diam_m pctiles [5,25,50,75,95,99]            <floor   score[50/90]")
print("-" * 100)
for obs in SAMPLE:
    p = GPKG_DIR / f"{obs}.gpkg"
    if not p.exists():
        print(f"{obs:<18} (no gpkg)")
        continue
    g = gpd.read_file(p, layer="detections")
    diam = 2.0 * np.sqrt(g.geometry.area.to_numpy() / np.pi)
    diam = diam[np.isfinite(diam)]
    pct = np.percentile(diam, [5, 25, 50, 75, 95, 99])
    below = float((diam < FLOOR_M).mean())
    sc = g["score"].to_numpy() if "score" in g.columns else np.array([np.nan])
    s50, s90 = (np.nanpercentile(sc, [50, 90]) if np.isfinite(sc).any() else (np.nan, np.nan))
    print(f"{obs:<18} {len(diam):>9}  {np.round(pct, 2).tolist()}   {below:5.1%}   "
          f"[{s50:.2f}/{s90:.2f}]")
    pooled_diam.append(diam)
    if np.isfinite(sc).any():
        pooled_score.append(sc)

alld = np.concatenate(pooled_diam)
alls = np.concatenate(pooled_score) if pooled_score else np.array([np.nan])
print("-" * 100)
print(f"POOLED n={len(alld):,}  diam_m pctiles [5,25,50,75,95,99] = "
      f"{np.round(np.percentile(alld, [5,25,50,75,95,99]), 2).tolist()}")
print(f"  fraction < {FLOOR_M} m floor: {(alld < FLOOR_M).mean():.1%}")
for thr in (0.5, 1.0, 1.4105, 2.0, 3.0):
    print(f"  diam >= {thr:>6} m : {(alld >= thr).mean():6.1%} kept")
print(f"  score pctiles [10,25,50,75,90] = {np.round(np.nanpercentile(alls, [10,25,50,75,90]), 3).tolist()}")
for thr in (0.2, 0.3, 0.5):
    print(f"  score >= {thr} : {(alls >= thr).mean():6.1%} kept")
