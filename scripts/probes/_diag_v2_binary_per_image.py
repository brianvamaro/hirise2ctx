"""Per-image breakdown of the v2 binary sweep at fa_gt_1e-2 — does individual-image
performance materially exceed the cross-image AUC mean?
"""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401
import numpy as np
import pandas as pd

sf = pd.read_parquet(REPO / "models/_sweep_binary/20260529T075754Z/summary.parquet")
print(f"per-fold rows: {len(sf):,}")

target = "fa_gt_1e-2"
scale_idx = 3  # S=64
sub = sf[(sf["target_id"] == target) & (sf["scale_idx"] == scale_idx) & ~sf["is_specificity_only"].astype(bool)].copy()
sub = sub.dropna(subset=["auc"])
print(f"\n{target} S=64 per-fold AUC distribution over {len(sub)} folds:")
print(sub["auc"].describe().to_string())
print()
print("top 10 folds by AUC:")
print(sub.sort_values("auc", ascending=False).head(10)[
    ["held_out_obs_id", "n_tiles", "n_positive", "n_negative", "auc",
     "lift_at_top_k", "base_rate", "ece"]
].to_string(index=False))
print()
print("bottom 5 by AUC:")
print(sub.sort_values("auc").head(5)[
    ["held_out_obs_id", "n_tiles", "n_positive", "n_negative", "auc",
     "lift_at_top_k", "base_rate", "ece"]
].to_string(index=False))

# Compare to bc_ge_1 at S=64
sub_bc = sf[(sf["target_id"] == "bc_ge_1") & (sf["scale_idx"] == 3) & ~sf["is_specificity_only"].astype(bool)].copy()
sub_bc = sub_bc.dropna(subset=["auc"])

merged = pd.merge(
    sub[["held_out_obs_id", "auc", "lift_at_top_k", "base_rate"]].rename(
        columns={"auc": "auc_rich", "lift_at_top_k": "lift_rich", "base_rate": "br_rich"}),
    sub_bc[["held_out_obs_id", "auc", "lift_at_top_k", "base_rate"]].rename(
        columns={"auc": "auc_any", "lift_at_top_k": "lift_any", "base_rate": "br_any"}),
    on="held_out_obs_id",
)
merged["delta_auc"] = merged["auc_rich"] - merged["auc_any"]
merged["delta_lift"] = merged["lift_rich"] - merged["lift_any"]
print("\nPer-image: fa_gt_1e-2 vs bc_ge_1 at S=64")
print(merged.sort_values("delta_lift", ascending=False).head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print()
print(f"mean delta_auc: {merged['delta_auc'].mean():+.3f}")
print(f"mean delta_lift: {merged['delta_lift'].mean():+.3f}")
print(f"images where boulder-rich beats any-boulder on AUC: "
      f"{(merged['delta_auc'] > 0).sum()}/{len(merged)}")
print(f"images where boulder-rich beats any-boulder on lift: "
      f"{(merged['delta_lift'] > 0).sum()}/{len(merged)}")
