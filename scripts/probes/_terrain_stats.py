"""Statistical test of terrain class vs Stage 7d attribution.

Tests:
  1. deposit_flag (transport indicator) x has-any-signal — Fisher's exact + OR
  2. deposit_flag x composition_residual specifically
  3. terrain_category x attribution — chi-squared
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent.parent
terrain = pd.read_parquet(ROOT / "dataset_v2" / "terrain_classification_v2.parquet")
attr = pd.read_parquet(ROOT / "dataset_v2" / "stage7d_attribution_shadow_0.10.parquet")

for rule in ("P4_area", "P2_count"):
    print(f"\n{'='*70}")
    print(f"PARTITION: {rule}")
    print('='*70)
    sub = attr[attr["partition_rule"] == rule].merge(terrain, on="obs_id", how="left")
    print(f"Eligible images: {len(sub)}")

    # ---------------- deposit_flag x has-any-signal ----------------
    sub["has_signal"] = sub["attribution"].isin(
        ["composition_residual", "dust_attributable"])
    sub["transport_indicator"] = (sub["deposit_flag"] | sub["streamlined_flag"])

    ct = pd.crosstab(sub["transport_indicator"], sub["has_signal"], margins=False)
    print("\ndeposit_or_streamlined x has_signal:")
    print(ct)

    if ct.shape == (2, 2):
        odds, p = stats.fisher_exact(ct.values, alternative="two-sided")
        print(f"Fisher's exact two-sided: odds ratio = {odds:.2f}, p = {p:.4f}")
        # Also compute as % signal in each group
        n_trans_signal = int(ct.loc[True, True]) if True in ct.index else 0
        n_trans_total = int(ct.loc[True].sum()) if True in ct.index else 0
        n_other_signal = int(ct.loc[False, True])
        n_other_total = int(ct.loc[False].sum())
        if n_trans_total:
            print(f"  Signal rate, transport-indicator images:    "
                  f"{n_trans_signal}/{n_trans_total} = {100*n_trans_signal/n_trans_total:.0f}%")
        if n_other_total:
            print(f"  Signal rate, non-transport-indicator images: "
                  f"{n_other_signal}/{n_other_total} = {100*n_other_signal/n_other_total:.0f}%")

    # ---------------- deposit_flag x composition_residual ----------------
    sub["is_comp"] = (sub["attribution"] == "composition_residual")
    ct2 = pd.crosstab(sub["transport_indicator"], sub["is_comp"], margins=False)
    print("\ndeposit_or_streamlined x is_composition_residual:")
    print(ct2)
    if ct2.shape == (2, 2):
        odds, p = stats.fisher_exact(ct2.values, alternative="two-sided")
        print(f"Fisher's exact two-sided: odds ratio = {odds:.2f}, p = {p:.4f}")

    # ---------------- terrain_category overview ----------------
    print("\nterrain_category x attribution:")
    ct3 = pd.crosstab(sub["terrain_category"], sub["attribution"], margins=True)
    print(ct3)

print("\n" + "="*70)
print("Summary: composition_residual images at T=0.10 / P4_area")
print("="*70)
sub_p4 = attr.query("partition_rule == 'P4_area' and attribution == 'composition_residual'") \
    .merge(terrain, on="obs_id", how="left")
print(sub_p4[["obs_id", "terrain_category", "deposit_flag", "streamlined_flag",
              "note"]].to_string(index=False, max_colwidth=80))
