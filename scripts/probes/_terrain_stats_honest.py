"""Recompute the terrain × attribution Fisher's exact test with proper
handling of images missing terrain annotations -- exclude them rather
than imputing transport_indicator = False (which was the full
compositional.md original approach).

The exclusion is what `compositional_slim.md` should report.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
terrain = pd.read_parquet(ROOT / "dataset_v2" / "terrain_classification_v2.parquet")
attr = pd.read_parquet(ROOT / "dataset_v2" / "stage7d_attribution_shadow_0.10.parquet")

for rule in ("P4_area", "P2_count"):
    print(f"\n{'='*70}")
    print(f"PARTITION: {rule}")
    print('='*70)
    sub = attr[attr["partition_rule"] == rule].merge(
        terrain, on="obs_id", how="left")

    print("\n-- A. Original approach: impute missing terrain as transport_indicator=False --")
    sub_impute = sub.copy()
    sub_impute["transport_indicator"] = (
        sub_impute["deposit_flag"].fillna(False)
        | sub_impute["streamlined_flag"].fillna(False))
    sub_impute["is_comp_resid"] = (sub_impute["attribution"] == "composition_residual")
    ct = pd.crosstab(sub_impute["transport_indicator"], sub_impute["is_comp_resid"])
    print(ct)
    if ct.shape == (2, 2):
        odds, p = stats.fisher_exact(ct.values, alternative="two-sided")
        print(f"n = {len(sub_impute)}, OR = {odds:.2f}, p = {p:.4f}")

    print("\n-- B. Honest approach: exclude images missing terrain annotations --")
    sub_honest = sub[sub["in_spreadsheet"].fillna(False)].copy()
    sub_honest["transport_indicator"] = (
        sub_honest["deposit_flag"] | sub_honest["streamlined_flag"])
    sub_honest["is_comp_resid"] = (sub_honest["attribution"] == "composition_residual")
    ct2 = pd.crosstab(sub_honest["transport_indicator"], sub_honest["is_comp_resid"])
    print(ct2)
    if ct2.shape == (2, 2):
        odds, p = stats.fisher_exact(ct2.values, alternative="two-sided")
        print(f"n = {len(sub_honest)}, OR = {odds:.2f}, p = {p:.4f}")

    # Identify which images get dropped
    excluded = sub[~sub["in_spreadsheet"].fillna(False)]
    print(f"\nExcluded under honest approach: {len(excluded)} images")
    for _, row in excluded.iterrows():
        print(f"  {row['obs_id']}  (attribution: {row['attribution']})")
