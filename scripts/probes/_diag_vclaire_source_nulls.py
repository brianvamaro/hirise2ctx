"""Are the null geometries present in the SOURCE shapefile, or introduced by reproject?"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geopandas as gpd

from src import manifest as M

ROOT = Path(r"C:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise_40_vClaire")

for obs in ["ESP_017355_2260", "ESP_068483_2280"]:
    shp = M.find_shapefile(obs, ROOT)
    g = gpd.read_file(shp)  # raw, native CRS, no reprojection
    geom = g.geometry
    print(f"\n=== {obs} (SOURCE shapefile, native CRS) ===")
    print(f"  rows:           {len(g)}")
    print(f"  null geometry:  {int(geom.isna().sum())}")
    print(f"  empty geometry: {int(geom.is_empty.sum())}")
    # is_at_edge / isin_slice breakdown for the null vs non-null split (diagnostic)
    if "is_at_edge" in g.columns:
        null_mask = geom.isna()
        print(f"  is_at_edge among null:    {int(g.loc[null_mask, 'is_at_edge'].sum())}/{int(null_mask.sum())}")
        print(f"  is_at_edge among nonnull: {int(g.loc[~null_mask, 'is_at_edge'].sum())}/{int((~null_mask).sum())}")
