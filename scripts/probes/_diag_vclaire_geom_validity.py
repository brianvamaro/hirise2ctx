"""Why do the densest vClaire GPKGs have far fewer finite-area polygons than rows?

Checks geometry validity / emptiness / area for the two densest images.
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

for obs in ["ESP_017355_2260", "ESP_068483_2280", "ESP_069669_2220"]:
    g = gpd.read_file(GPKG_DIR / f"{obs}.gpkg", layer="detections")
    geom = g.geometry
    area = geom.area.to_numpy()
    print(f"\n=== {obs} ===")
    print(f"  rows:            {len(g)}")
    print(f"  crs:             {g.crs.name if g.crs else None}")
    print(f"  geom types:      {geom.geom_type.value_counts().to_dict()}")
    print(f"  null geometry:   {int(geom.isna().sum())}")
    print(f"  empty geometry:  {int(geom.is_empty.sum())}")
    print(f"  invalid (~valid):{int((~geom.is_valid).sum())}")
    print(f"  area==0:         {int((area == 0).sum())}")
    print(f"  area is NaN:     {int(np.isnan(area).sum())}")
    print(f"  area>0 finite:   {int((np.isfinite(area) & (area > 0)).sum())}")
