"""H1+H2 results figure for notebook 12.

Two panels:
  A) Operational metrics bar chart -- PR-AUC, normalised_lift, precision@5%, recall@5%
     by target, S=64.  Shows the +22-27% relative lift from boulder_count without an
     AUC change (the H1 framework prediction).
  B) Ranking vs. detection scatter: Spearman x meaningful_AUC by target,
     S=32 + S=64.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = REPO_ROOT / "models" / "_sweep_target_reformulation"
runs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
df = pd.read_parquet(runs[-1] / "aggregate.parquet")
print(f"Loaded {len(df)} rows from {runs[-1].name}")

TARGET_LABEL = {
    "fractional_area": "fractional_area\n(baseline)",
    "boulder_count": "boulder_count\n(alias-robust)",
    "log_boulder_count": "log_boulder_count\n(explicit log)",
}
TARGET_COLOR = {
    "fractional_area": "C0",
    "boulder_count": "C2",
    "log_boulder_count": "C1",
}

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# --- Panel A: operational metrics at S=64 ---
ax = axes[0]
s64 = df[df["tile_size_px"] == 64].set_index("target_col")
metrics = [
    ("presence_auc_mean",          "ROC-AUC\n(presence)"),
    ("meaningful_auc_mean",        "ROC-AUC\n(meaningful)"),
    ("pr_auc_mean",                "PR-AUC"),
    ("normalised_lift_meaningful_mean", "Normalised\nlift@top-K"),
    ("precision_at_top_5pct_mean", "Precision\n@top-5%"),
    ("recall_at_top_5pct_mean",    "Recall\n@top-5%"),
]
x = np.arange(len(metrics))
w = 0.27
order = ["fractional_area", "boulder_count", "log_boulder_count"]
for i, target in enumerate(order):
    if target not in s64.index:
        continue
    row = s64.loc[target]
    vals = [row[m[0]] for m in metrics]
    bars = ax.bar(x + (i - 1) * w, vals, w,
                  label=TARGET_LABEL[target], color=TARGET_COLOR[target], alpha=0.85)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
ax.set_ylabel("metric value (mean across folds)")
ax.set_title("H1+H2 result at S=64: ROC-AUC unchanged, PR-AUC + lift jump ~25 %\n"
             "(H1 prediction confirmed: AUC hides the operational gain)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3, axis="y")
ax.axhline(0, color="k", lw=0.5)

# --- Panel B: ranking vs detection scatter, S=32 and S=64 ---
ax = axes[1]
markers = {32: "o", 64: "s"}
for target in order:
    sub = df[df["target_col"] == target]
    for _, row in sub.iterrows():
        S = int(row["tile_size_px"])
        x = row["spearman_rho_mean"]
        y = row["pr_auc_mean"]
        ax.scatter(x, y, s=150, color=TARGET_COLOR[target], marker=markers[S],
                   edgecolors="k", linewidths=0.7)
        ax.annotate(f"{TARGET_LABEL[target].split(chr(10))[0]}\nS={S}",
                    (x, y), xytext=(8, 5), textcoords="offset points", fontsize=8)
ax.set_xlabel("Spearman ρ  (ranking quality, rank-invariant across targets)")
ax.set_ylabel("PR-AUC  (operational detection at fa>1e-2 equivalent)")
ax.set_title("Ranking is the same across targets (rank-invariant)\n"
             "but PR-AUC jumps with boulder_count -- compression was in the loss-target match")
ax.grid(alpha=0.3)

plt.suptitle(f"H2 result: target reformulation lifts operational metrics "
             f"at fixed ranking quality\n({runs[-1].name}, "
             f"v2-dev within-image 20 folds, variant=lightgbm_two_stage_balanced)",
             fontsize=11)
plt.tight_layout()
out = REPO_ROOT / "reports" / "figures" / "12_target_reformulation.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Figure -> {out}")
