"""Inspect dataset_v2 labels parquet schema for ESP_042964_2160 (Stage 7.0 Test B prep)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

df = pd.read_parquet("dataset_v2/labels/ESP_042964_2160.parquet")
print("Columns:", list(df.columns))
print("Dtypes:")
print(df.dtypes)
print("\nUnique scale_S values:", sorted(df["scale_S"].unique()) if "scale_S" in df else "no scale_S col")
if "scale_idx" in df:
    print("Unique scale_idx:", sorted(df["scale_idx"].unique()))
print(f"\nRow count: {len(df)}")
print("\nSample row:")
print(df.head(3).T)
