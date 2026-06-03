"""Join the user's terrain spreadsheet onto the v2 manifest.

Filters the Mapping_Images_33_36 sheet to just the v2 cohort ObsIds and
prints the available Notes / Quality columns. Anything missing is then
the gap I need to classify from browse images.
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
XL_PATH = Path("C:/Users/brian/Downloads/Mapping_Images_33_36.xlsx")

manifest = pd.read_csv(ROOT / "hirise_40_vclaire.csv")
v2_obs = set(manifest["ObsId"])

xl_lon = pd.read_excel(XL_PATH, sheet_name="Sorted_Lon")
xl_lon["ObsId"] = xl_lon["Image Name"].astype(str).str.strip()

hits = xl_lon[xl_lon["ObsId"].isin(v2_obs)].copy()
print(f"v2 ObsIds in manifest: {len(v2_obs)}")
print(f"v2 ObsIds found in spreadsheet: {len(hits)}")
print(f"v2 ObsIds NOT in spreadsheet: {len(v2_obs - set(hits['ObsId']))}")
print()
print("Missing (need browse-image classification):")
for o in sorted(v2_obs - set(hits["ObsId"])):
    print(f"  {o}")
print()
print("Found in spreadsheet (with available terrain notes):")
disp_cols = ["ObsId", "Quality of boulders", "Overall... ", "Notes "]
disp_cols = [c for c in disp_cols if c in hits.columns]
print(hits[disp_cols].to_string(index=False, max_colwidth=120))
