"""Render the headline compression-fix comparison figure for notebook 12.

Two panels:
  A) Per-truth-bin mean_pred for each variant at S=64, overlaid on the truth identity.
  B) Headline scatter: Spearman vs (1 - |1 - high_bin_ratio|) — i.e., ranking vs
     tail calibration trade-off; the closer to the upper-right corner, the better
     both ranking AND tail recovery.
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

ROOT = REPO_ROOT / "models" / "_sweep_compression_fixes"
runs = sorted([p for p in ROOT.iterdir() if p.is_dir()])
df = pd.read_parquet(runs[-1] / "aggregate.parquet")
print(f"Loaded {len(df)} rows from {runs[-1].name}")

BINS = ("zero", "0_to_1e-4", "1e-4_to_1e-3", "1e-3_to_1e-2", "1e-2_to_max")
TRUTH_PER_BIN = {
    b: float(df[f"{b}__mean_true"].iloc[0]) for b in BINS
}

VARIANT_SHORT = {
    "lightgbm_two_stage": "baseline",
    "lightgbm_two_stage_balanced": "balanced (presence fix)",
    "lightgbm_two_stage_weighted": "weighted (mag y-weight)",
    "lightgbm_two_stage_gamma": "gamma (mag loss)",
    "lightgbm_two_stage_combined": "combined (all 3)",
}
COLORS = {
    "lightgbm_two_stage": "C0",
    "lightgbm_two_stage_balanced": "C2",
    "lightgbm_two_stage_weighted": "C3",
    "lightgbm_two_stage_gamma": "C1",
    "lightgbm_two_stage_combined": "C4",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# --- Panel A: per-bin mean_pred at S=64 ---
ax = axes[0]
s64 = df[df["tile_size_px"] == 64]
x = np.arange(len(BINS))
truth_y = [TRUTH_PER_BIN[b] for b in BINS]
n_v = len(s64)
w = 0.8 / n_v
for i, (_, row) in enumerate(s64.iterrows()):
    means = [row[f"{b}__mean_pred"] for b in BINS]
    label = VARIANT_SHORT[row["variant"]]
    ax.bar(x + (i - (n_v - 1) / 2) * w, means, w,
           label=f"{label}\n(ρ={row['spearman_rho_mean']:.3f}, AUC={row['presence_auc_mean']:.3f})",
           color=COLORS[row["variant"]], alpha=0.85)
ax.plot(x, truth_y, "k^-", lw=2, ms=8, label="mean_true (identity)")
ax.set_xticks(x); ax.set_xticklabels(BINS, rotation=20)
ax.set_ylabel("mean prediction (linear scale)")
ax.set_title("Per-truth-bin mean prediction — v2-dev within-image S=64\n"
             "(closer to black ▲ = less compression)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)

# --- Panel B: ranking vs tail-calibration trade-off ---
ax = axes[1]
markers = {32: "o", 64: "s"}
for variant in VARIANT_SHORT:
    sub = df[df["variant"] == variant]
    for _, row in sub.iterrows():
        S = int(row["tile_size_px"])
        x = row["spearman_rho_mean"]
        # y = how close to 1.0 the high-bin ratio is (1.0 = perfect tail recovery)
        y = row["1e-2_to_max__ratio"]
        ax.scatter(x, y, s=140, color=COLORS[variant], marker=markers[S],
                   edgecolors="k", linewidths=0.7,
                   label=f"{VARIANT_SHORT[variant]} S{S}")
        ax.annotate(VARIANT_SHORT[variant].split()[0] + f" S{S}",
                    (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8)
ax.axhline(1.0, color="k", ls=":", lw=1, alpha=0.6, label="perfect tail calibration")
ax.axvline(df[df["variant"] == "lightgbm_two_stage"]["spearman_rho_mean"].max(),
           color="C0", ls=":", lw=1, alpha=0.6, label="baseline best ρ")
ax.set_xlabel("Spearman ρ  (higher = better ranking)")
ax.set_ylabel("high-bin ratio  mean_pred / mean_true  (1.0 = perfect)")
ax.set_title("Ranking vs tail-calibration trade-off\n(upper-right = both good; weighted+combined trade ρ for tail)")
ax.grid(alpha=0.3)

plt.suptitle("Compression-fix sweep — v2-dev within-image (20 folds)", fontsize=12)
plt.tight_layout()
out = REPO_ROOT / "reports" / "figures" / "12_compression_fix_sweep.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"Figure -> {out}")
