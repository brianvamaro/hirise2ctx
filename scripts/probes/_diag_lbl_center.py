"""Compare CenterLat/Lon sources for a few vClaire images: projection_origin (WRONG for
center) vs image_footprint midpoint (authoritative) vs spreadsheet corner1."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from src import pds_labels

CACHE = REPO_ROOT / "cache"
XLSX = Path(r"C:\Users\brian\Downloads\Mapping_Images_33_36.xlsx")
SAMPLE = ["ESP_017355_2260", "ESP_076499_1160", "ESP_069669_2220", "ESP_047976_2020"]

# spreadsheet corner lookup
lut = {}
xl = pd.ExcelFile(XLSX)
for s in xl.sheet_names:
    df = xl.parse(s)
    if "Image Name" not in df.columns:
        continue
    for _, r in df.iterrows():
        o = str(r["Image Name"]).strip()
        if o.startswith("ESP_") and o not in lut and pd.notna(r.get("corner1_latitude")):
            lut[o] = (float(r["corner1_latitude"]), float(r["corner1_longitude"]))

for obs in SAMPLE:
    print(f"\n=== {obs} ===")
    try:
        po = pds_labels.projection_origin(obs, CACHE)
        print(f"  projection_origin: lat={po['center_lat_deg']}  lon360={po['center_lon_deg']}")
    except Exception as e:  # noqa: BLE001
        print(f"  projection_origin ERR: {e}")
    try:
        fp = pds_labels.image_footprint(obs, CACHE)
        mid_lat = (fp["max_lat_deg"] + fp["min_lat_deg"]) / 2
        print(f"  footprint: lat[{fp['min_lat_deg']}, {fp['max_lat_deg']}] -> mid {mid_lat:.4f}")
        print(f"  footprint: lon E={fp['east_lon_deg']} W={fp['west_lon_deg']}")
    except Exception as e:  # noqa: BLE001
        print(f"  image_footprint ERR: {e}")
    if obs in lut:
        print(f"  spreadsheet corner1: lat={lut[obs][0]}  lon={lut[obs][1]}")
