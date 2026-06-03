"""Build a structured terrain_class table from Brian's free-text notes.

Inputs:
  - hirise_40_vclaire.csv (v2 manifest, 39 rows)
  - C:/Users/brian/Downloads/Mapping_Images_33_36.xlsx (Sorted_Lon sheet,
    Brian's mapping notes; 37 of 39 ObsIds covered)

Outputs (printed):
  - For each v2 ObsId: parsed terrain features {plains, mesas, channels,
    crater, hills, deposit_flag, streamlined_flag}, dominant terrain
    category, and the raw note.
  - Cross-tab vs the Stage 7d attribution table at T=0.10 / P4_area.

The 2 missing ObsIds (ESP_017355_2260, ESP_076499_1160) are flagged as
needing browse-image inspection.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
XL_PATH = Path("C:/Users/brian/Downloads/Mapping_Images_33_36.xlsx")

manifest = pd.read_csv(ROOT / "hirise_40_vclaire.csv")
xl = pd.read_excel(XL_PATH, sheet_name="Sorted_Lon")
xl["ObsId"] = xl["Image Name"].astype(str).str.strip()
notes_col = "Notes "  # trailing space is in the actual header

KEYWORDS = {
    "plains": r"\bplain",
    "mesas": r"\bmesa",
    "channels": r"\bchannel",
    "crater": r"\bcrater",
    "hills": r"\bhill",
    "deposit_flag": r"\bdeposit\!",
    "streamlined_flag": r"streamlined",
    "eroded_flag": r"erod",
    "hiview_flag": r"\bHIVIEW",
    "buried_flag": r"\bburied",
}


def parse_notes(note: str) -> dict:
    if not isinstance(note, str):
        return {k: False for k in KEYWORDS}
    note_l = note.lower()
    return {k: bool(re.search(pat, note_l, flags=re.IGNORECASE))
            for k, pat in KEYWORDS.items()}


def dominant_terrain(flags: dict) -> str:
    """Pick a single dominant category for stratification (priority order)."""
    if flags["channels"]:
        return "channels"
    if flags["mesas"]:
        return "mesas"
    if flags["crater"] and flags["plains"]:
        return "plains_with_crater"
    if flags["crater"]:
        return "crater_dominated"
    if flags["hills"]:
        return "hills"
    if flags["plains"]:
        return "plains"
    return "unclassified"


rows = []
for _, m in manifest.iterrows():
    obs = m["ObsId"]
    xl_row = xl[xl["ObsId"] == obs]
    note = xl_row[notes_col].iloc[0] if len(xl_row) else None
    flags = parse_notes(note) if note is not None else {k: False for k in KEYWORDS}
    rows.append({
        "obs_id": obs, "BoulderLabel": m["BoulderLabel"],
        "CenterLat": m["CenterLat"], "CenterLon_180": m["CenterLon_180"],
        "in_spreadsheet": note is not None,
        "note": note,
        **flags,
        "terrain_category": dominant_terrain(flags) if note is not None else "MISSING",
    })

terrain = pd.DataFrame(rows)

# Save the structured terrain table
out_path = ROOT / "dataset_v2" / "terrain_classification_v2.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
terrain.to_parquet(out_path, index=False)
print(f"Wrote {out_path}  ({len(terrain)} rows)")
print()

print("Per-ObsId structured terrain classification:")
disp_cols = ["obs_id", "BoulderLabel", "terrain_category",
             "deposit_flag", "streamlined_flag", "note"]
print(terrain[disp_cols].to_string(index=False, max_colwidth=100))
print()

print(f"terrain_category counts: {dict(terrain['terrain_category'].value_counts())}")
print(f"deposit_flag count: {int(terrain['deposit_flag'].sum())} of {len(terrain)}")
print(f"streamlined_flag count: {int(terrain['streamlined_flag'].sum())} of {len(terrain)}")
print()

print("=== Cross-tab vs Stage 7d attribution (T=0.10 / P4_area) ===")
attr = pd.read_parquet(ROOT / "dataset_v2" / "stage7d_attribution_shadow_0.10.parquet")
attr_p4 = attr.query("partition_rule == 'P4_area'").copy()
joined = attr_p4.merge(terrain, on="obs_id", how="left")
print(f"Joined: {len(joined)} eligible images")
print()
print("Attribution x terrain_category:")
ct = pd.crosstab(joined["attribution"], joined["terrain_category"], margins=True)
print(ct)
print()
print("Attribution x deposit_flag:")
ct2 = pd.crosstab(joined["attribution"], joined["deposit_flag"], margins=True)
print(ct2)
print()
print("Per-image attribution + terrain:")
disp = joined[["obs_id", "attribution", "terrain_category",
               "deposit_flag", "streamlined_flag", "note"]] \
    .sort_values(["attribution", "obs_id"])
print(disp.to_string(index=False, max_colwidth=80))
