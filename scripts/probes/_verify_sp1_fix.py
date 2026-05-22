"""Force a re-read of the ESP_047976_2020 decimated cache through the new override path
and dump the resulting CRS SP1 to confirm the fix kicked in."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rasterio

from src.hirise_imagery import _sp1_literal, read_full_footprint_decimated
from src import manifest as M

df = M.load_manifest(REPO_ROOT / "hirise_priority10.csv")
row = df.set_index("ObsId").loc["ESP_047976_2020"]

before_path = REPO_ROOT / "cache" / "hirise_decimated" / "ESP_047976_2020_5mpp_full.tif"
print("BEFORE re-read:")
with rasterio.open(before_path) as ds:
    print("  cache SP1 =", _sp1_literal(ds.crs))

print("Triggering read_full_footprint_decimated (should rebuild if stale)...")
arr, tr, crs = read_full_footprint_decimated(
    "ESP_047976_2020", str(row["JP2_URL"]),
    REPO_ROOT / "cache", target_mpp=5.0,
)
print("AFTER re-read:")
with rasterio.open(before_path) as ds:
    print("  cache SP1 =", _sp1_literal(ds.crs))
print("  returned-CRS SP1 =", _sp1_literal(crs))
