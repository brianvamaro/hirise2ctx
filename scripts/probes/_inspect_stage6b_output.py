"""Sanity-check the Stage 6b output on ESP_055978_2270.

Verifies the 7 new columns are populated with plausible per-image values and
prints per-scale distribution summaries.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
P = REPO_ROOT / "dataset_v2_dev" / "features_ctx_illum" / "ESP_055978_2270.parquet"
df = pd.read_parquet(P)
print(f"{len(df):,} tiles, {len(df.columns)} columns")
print()
print("Stage 6b columns:")
for c in df.columns:
    if c.startswith("ctx_"):
        s = df[c]
        print(f"  {c:<32s}  n_finite={s.notna().sum():>7,}  "
              f"min={s.min():7.3f}  med={s.median():7.3f}  "
              f"max={s.max():7.3f}  std={s.std():6.3f}")
print()
print("Per-scale CTX_INCIDENCE_MEAN stats:")
for S, sub in df.groupby("tile_size_px"):
    s = sub["ctx_incidence_mean"]
    print(f"  S={S:>3d}  n_tiles={len(sub):>6,}  finite={s.notna().sum():>6,}  "
          f"mean={s.mean():.2f}  std={s.std():.2f}  "
          f"n_sources avg={sub['ctx_n_sources'].mean():.2f}")
print()
print("Sample (S=64):")
sample = df[df["tile_size_px"] == 64].iloc[:5]
cols = ["obs_id", "scale_idx", "tile_size_px", "ti", "tj",
        "ctx_incidence_mean", "ctx_emission_mean", "ctx_phase_mean",
        "ctx_n_sources", "ctx_dominant_source_fraction"]
print(sample[cols].to_string(index=False))
