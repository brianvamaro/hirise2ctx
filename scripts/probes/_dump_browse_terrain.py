"""Dump the v2 manifest's ObsId, BoulderLabel, lat/lon, TerrainNote, BrowseURL."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
df = pd.read_csv(ROOT / "hirise_40_vclaire.csv")
cols = ["ObsId", "BoulderLabel", "CenterLat", "CenterLon_180",
        "TerrainNote", "BrowseURL"]
print(df[cols].to_string(index=False, max_colwidth=120))
print()
print(f"Total: {len(df)} ObsIds")
print(f"BoulderLabel counts: {dict(df['BoulderLabel'].value_counts())}")
print(f"Existing TerrainNote (non-null): {df['TerrainNote'].notna().sum()}")
print()
print("TerrainNote samples (non-null):")
for _, row in df[df["TerrainNote"].notna()].iterrows():
    print(f"  {row['ObsId']}: {row['TerrainNote']}")
