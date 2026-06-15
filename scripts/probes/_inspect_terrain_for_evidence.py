"""Inspect terrain classes + per-image AUC + base rates to plan evidence figures."""
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 60)

terr = pd.read_parquet(REPO / "dataset_v2" / "terrain_classification_v2.parquet")
print("=== terrain_classification_v2 columns ===")
print(terr.columns.tolist())
print(terr.head(3).to_string())
# value counts of any terrain-type column
for c in terr.columns:
    if terr[c].dtype == object or str(terr[c].dtype).startswith("category"):
        print(f"\n[{c}] value counts:")
        print(terr[c].value_counts().to_string())

dos = pd.read_parquet(REPO / "dataset_v2" / "w1_dossier.parquet")
print("\n\n=== w1_dossier columns ===")
print(dos.columns.tolist())
print(f"rows={len(dos)}")
