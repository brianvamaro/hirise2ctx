"""Build the per-image AUC figure for docs/modeling_slim.md.

Reads dataset_v2/modeling_slim_summary.parquet.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SUM = pd.read_parquet(ROOT / "dataset_v2" / "modeling_slim_summary.parquet")
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

per_fold = SUM[SUM["fold_idx"] >= 0].copy()
pooled = SUM[SUM["fold_idx"] == -1].iloc[0]

# Per-image AUC distribution at fa_gt_1e-2
fig, ax = plt.subplots(figsize=(8.5, 4.5))
vals = per_fold["auc_fa_rich"].dropna().sort_values().to_numpy()
colors_bars = ["#e76f51" if v < 0.5 else ("#bdbdbd" if v < 0.7 else "#2a9d8f")
               for v in vals]
ax.bar(np.arange(len(vals)), vals, color=colors_bars, edgecolor="black", linewidth=0.4)
ax.axhline(0.5, color="k", linewidth=0.7, linestyle="--", label="chance (0.5)")
ax.axhline(0.7, color="darkgreen", linewidth=0.7, linestyle=":", label="usable (0.7)")
ax.set_xlabel("per-image rank (sorted by AUC)")
ax.set_ylabel("per-image AUC at fa_gt_1e-2")
ax.set_title(
    f"Per-image AUC distribution at the boulder-rich threshold (fa >= 1%, 320 m tiles)\n"
    f"median={np.median(vals):.3f}, max={vals.max():.3f}, min={vals.min():.3f}, "
    f"frac>=0.70 = {(vals>=0.70).mean():.0%}, frac<0.50 = {(vals<0.50).mean():.0%}; "
    f"n={len(vals)} folds with both classes")
ax.grid(True, axis="y", linestyle=":", alpha=0.4)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "modeling_slim_per_image_auc.png", dpi=140, bbox_inches="tight")
plt.show()
print(f"Wrote {FIG / 'modeling_slim_per_image_auc.png'}")

# Drop the older slim-vs-full comparison figure if present (no longer used)
old = FIG / "modeling_slim_vs_full_rho.png"
if old.exists():
    old.unlink()
    print(f"Removed obsolete {old}")
