"""Compute the headline within-image vs LOIO deltas for docs/modeling_results.md.

For each (variant, scale_idx) cell:
  1. Aggregate the 4 within-image quadrant folds per image -> 1 AUC per image.
  2. Pair with the corresponding LOIO fold AUC for the same image.
  3. Report mean delta, bootstrap 95% CI, Wilcoxon signed-rank p-value.

For two_stage (regression), AUC = presence_auc.
For classification, AUC = auc.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
from scipy import stats

from src.modeling.sweep_select import pick_sweep

MODELS_ROOT = REPO_ROOT / "models"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--dataset-dir", default="dataset",
                help="Dataset version to select sweeps for (dataset = v1, dataset_v2 = vClaire).")
args = ap.parse_args()

WITHIN_DIR = pick_sweep("within_image", args.dataset_dir)
LOIO_DIR = pick_sweep("regression", args.dataset_dir)
BIN_DIR = pick_sweep("binary", args.dataset_dir)

print(f"dataset: {args.dataset_dir}")
print(f"within: {WITHIN_DIR.name}")
print(f"loio:   {LOIO_DIR.name}")
print(f"bin:    {BIN_DIR.name}")

within_summary = pd.read_parquet(WITHIN_DIR / "summary.parquet")
loio_summary = pd.read_parquet(LOIO_DIR / "summary.parquet")
loio_bin = pd.read_parquet(BIN_DIR / "summary.parquet")

# AUC column per variant.
def loio_for(variant: str, scale_idx: int) -> pd.DataFrame:
    if variant == "lightgbm_two_stage":
        sub = loio_summary[(loio_summary["variant"] == "lightgbm_two_stage") & (loio_summary["scale_idx"] == scale_idx)]
        sub = sub[~sub["is_specificity_only"].astype(bool)]
        return sub[["held_out_obs_id", "presence_auc"]].rename(columns={"presence_auc": "loio_auc"})
    if variant == "lightgbm_classification":
        sub = loio_bin[(loio_bin["target_id"] == "bc_ge_1") & (loio_bin["scale_idx"] == scale_idx)]
        sub = sub[~sub["is_specificity_only"].astype(bool)]
        return sub[["held_out_obs_id", "auc"]].rename(columns={"auc": "loio_auc"})
    raise ValueError(variant)

def within_per_image(variant: str, scale_idx: int) -> pd.DataFrame:
    sub = within_summary[(within_summary["variant"] == variant) & (within_summary["scale_idx"] == scale_idx)]
    sub = sub[~sub["is_specificity_only"].astype(bool)]
    col = "presence_auc" if variant == "lightgbm_two_stage" else "auc"
    out = sub.groupby("held_out_obs_id").agg(within_auc=(col, "mean"), n_real_folds=("fold_idx", "count")).reset_index()
    return out


variants = ["lightgbm_two_stage", "lightgbm_classification"]
scales = [0, 1, 2, 3]
rng = np.random.default_rng(0)
rows = []
print("\nvariant                    S   n  mean_delta   CI95             Wilcoxon p   within_AUC   LOIO_AUC")
print("-" * 110)
for v in variants:
    for s in scales:
        w = within_per_image(v, s)
        lo = loio_for(v, s)
        paired = w.merge(lo, on="held_out_obs_id", how="inner")
        paired["delta"] = paired["within_auc"] - paired["loio_auc"]
        # Drop images whose AUC is undefined on either side (all-positive folds at coarse
        # scales -- v2 is ~93% positive at S=64, so some images saturate). Pair only where
        # both sides are defined.
        paired = paired.dropna(subset=["delta"]).reset_index(drop=True)
        n = len(paired)
        delta = paired["delta"].to_numpy()
        mean_delta = float(delta.mean()) if n else float("nan")
        if n >= 2:
            boots = rng.choice(delta, size=(10_000, n), replace=True).mean(axis=1)
            ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            wp = float(stats.wilcoxon(delta, alternative="two-sided").pvalue) if (delta != 0).any() else float("nan")
        else:
            ci_lo = ci_hi = wp = float("nan")
        rows.append({
            "variant": v, "scale_idx": s, "tile_size_px": int(2 ** (3 + s)),
            "n_paired": n, "mean_delta": mean_delta, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "wilcoxon_p": wp,
            "within_auc_mean": float(paired["within_auc"].mean()),
            "loio_auc_mean": float(paired["loio_auc"].mean()),
        })
        print(f"{v:<25s}  S={int(2 ** (3 + s)):<2d}  n={n}  {mean_delta:+.4f}    "
              f"[{ci_lo:+.4f}, {ci_hi:+.4f}]   p={wp:.4f}     {paired['within_auc'].mean():.4f}    {paired['loio_auc'].mean():.4f}")

df = pd.DataFrame(rows)
df.to_parquet(WITHIN_DIR / "delta_vs_loio.parquet", index=False)
print(f"\nWrote {WITHIN_DIR / 'delta_vs_loio.parquet'}")
