"""Per-image meaningful-AUC for a banked recipe cell, sorted ascending, with
anti-signal (<0.50) flags and the n_neg validity column (W1 lesson: per-image
AUC on near-saturated images is statistically meaningless).

Usage: python _w1_antisignal_list.py [sweep_dir] [variant] [target]
"""
import sys
from pathlib import Path

import pandas as pd

sweep_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "models/_sweep_w0/20260610T221932Z")
variant = sys.argv[2] if len(sys.argv) > 2 else "lightgbm_two_stage_balanced"
target = sys.argv[3] if len(sys.argv) > 3 else "boulder_count"

agg = pd.read_parquet(sweep_dir / "aggregate.parquet")
art = agg.loc[(agg.variant == variant) & (agg.target_col == target), "artifact_dir"].iloc[0]
print(f"artifact_dir: {art}")

summ = pd.read_parquet(sweep_dir / "summary.parquet")
rec = summ[(summ.variant == variant) & (summ.target_col == target)].copy()
rec["n_neg"] = rec.n_tiles - rec.n_meaningful_positive
rec = rec.sort_values("meaningful_auc")
cols = ["held_out_obs_id", "n_tiles", "n_meaningful_positive", "n_neg",
        "meaningful_auc", "pr_auc", "spearman_rho"]
print(rec[cols].to_string(index=False))
auc = rec.meaningful_auc
print(f"\nmedian AUC {auc.median():.3f} | >0.70: {(auc > 0.7).mean():.1%} | <0.50: {(auc < 0.5).mean():.1%}")
anti = rec[rec.meaningful_auc < 0.5]
print(f"anti-signal: {len(anti)} of {len(rec)}: {list(anti.held_out_obs_id)}")
valid = rec[(rec.n_neg >= 50) & (rec.n_meaningful_positive >= 50)]
print(f"\nwith >=50 pos and >=50 neg (n={len(valid)}): median {valid.meaningful_auc.median():.3f} "
      f"| <0.50: {(valid.meaningful_auc < 0.5).mean():.1%}")
