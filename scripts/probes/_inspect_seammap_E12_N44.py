"""Confirm the Murray Lab SeamMap.shp embeds per-source illumination angles directly.

If true, we don't need the PDS CUMINDEX at all for Stage 6b — the SeamMap is
sufficient. (Notebook 13 said the seam files "don't carry illumination angles"; the
column listing for E012_N44 suggests otherwise.)
"""
from __future__ import annotations
from pathlib import Path
import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parents[2]
SHP = REPO_ROOT / "cache_v2" / "ctx_tiles" / "_seammap_E12_N44" / "MurrayLab_CTX_V01_E012_N44_SeamMap.shp"

sm = gpd.read_file(SHP)
print(f"Loaded {len(sm)} rows.")
print()
print("First 8 rows (key cols):")
key = ["PRODUCT_ID", "OG_PROD_ID", "EMISSION", "INCIDENCE", "PHASE",
       "SB_SLR_AZ", "CLAT", "CLONG", "PDS_IMG"]
print(sm[key].head(8).to_string())
print()
print("Numeric summary:")
for c in ["EMISSION", "INCIDENCE", "PHASE", "SB_SLR_AZ"]:
    s = sm[c]
    print(f"  {c:12s}  n={s.notna().sum():5d}  "
          f"min={s.min():7.2f}  med={s.median():7.2f}  "
          f"max={s.max():7.2f}  std={s.std():6.2f}")
print()
print(f"PRODUCT_ID uniqueness: {sm['PRODUCT_ID'].nunique()} unique / {len(sm)} rows")
print(f"Top 3 PRODUCT_ID by row count (multi-polygon sources):")
print(sm["PRODUCT_ID"].value_counts().head(3))
print()
print("SeamMap CRS:")
print(sm.crs)
