"""Companion to _summarize_modeling_results.py for the Stage 5b binary sweep.

Computes the diagnostics needed to write the 'Binary reframing' section of
docs/modeling_results.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.config import load_config
import src.manifest as M


def main() -> int:
    cfg = load_config("config.yaml")
    manifest = M.load_manifest(cfg.manifest_path)
    obs_to_label = dict(zip(manifest["ObsId"], manifest["BoulderLabel"]))

    sweep_root = REPO_ROOT / "models" / "_sweep_binary"
    sweep_dirs = sorted(sweep_root.glob("*"))
    sweep_dir = sweep_dirs[-1]
    print(f"# Binary sweep dir: {sweep_dir.name}\n")

    agg = pd.read_parquet(sweep_dir / "aggregate.parquet").sort_values(
        ["scale_idx", "target_id"]
    )
    summary = pd.read_parquet(sweep_dir / "summary.parquet")
    summary["boulder_label"] = summary["held_out_obs_id"].map(obs_to_label).fillna("empty")

    # ---------- 1. Aggregate table ----------
    print("## 1. Binary aggregate (12 target x scale, mean ± std over real folds)\n")
    print(agg[["target_id", "scale_idx", "tile_size_px",
               "auc_mean", "auc_std", "brier_mean", "ece_mean",
               "lift_at_top_k_mean", "n_real_folds"]
              ].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---------- 2. Sign tests ----------
    auc_means = agg["auc_mean"].to_numpy()
    n_above_chance = int((auc_means > 0.5).sum())
    p_auc = binomtest(n_above_chance, len(auc_means), 0.5, alternative="greater").pvalue
    print(f"\n## 2. Sign-tests across the 12 (target, scale) combos\n")
    print(f"AUC above 0.5 in {n_above_chance}/{len(auc_means)} combos. Sign-test p = {p_auc:.4f}")
    print(f"Mean AUC across all 12: {auc_means.mean():+.4f}")
    print(f"Median AUC: {np.median(auc_means):+.4f}")

    real_summary = summary[~summary["is_specificity_only"]].copy()
    fold_auc = real_summary["auc"].dropna().to_numpy()
    n_pos_fold = int((fold_auc > 0.5).sum())
    p_fold = binomtest(n_pos_fold, len(fold_auc), 0.5, alternative="greater").pvalue
    print(f"Per-fold AUC > 0.5: {n_pos_fold}/{len(fold_auc)}  Sign-test p = {p_fold:.4f}")

    # ---------- 3. Per-target rollup ----------
    print(f"\n## 3. Per-target headline (averaged over the 4 scales)\n")
    by_t = agg.groupby("target_id").agg(
        auc_mean=("auc_mean", "mean"),
        auc_std=("auc_mean", "std"),
        lift_mean=("lift_at_top_k_mean", "mean"),
        brier_mean=("brier_mean", "mean"),
        ece_mean=("ece_mean", "mean"),
    ).reset_index()
    print(by_t.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---------- 4. Per-class fold breakdown ----------
    print(f"\n## 4. Per-fold AUC by held-out BoulderLabel (pooled over all targets x scales)\n")
    by_label = real_summary.groupby("boulder_label")["auc"].agg(["count", "mean", "std", "min", "max"])
    print(by_label.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ---------- 5. Best (target, scale) per target ----------
    print(f"\n## 5. Best (target, scale) per target by AUC mean\n")
    for t in sorted(agg["target_id"].unique()):
        sub = agg[agg["target_id"] == t].sort_values("auc_mean", ascending=False)
        b = sub.iloc[0]
        w = sub.iloc[-1]
        print(f"  {t:<11s}: best=S={int(b['tile_size_px']):>2d}  auc={b['auc_mean']:+.4f} +/- {b['auc_std']:.4f}   "
              f"worst=S={int(w['tile_size_px']):>2d}  auc={w['auc_mean']:+.4f}")

    # ---------- 6. Head-to-head: binary classifier vs regression two-stage presence head ----------
    print(f"\n## 6. Head-to-head: bc_ge_1 classifier vs lightgbm_two_stage presence AUC\n")
    # Need to pull from the existing regression sweep
    reg_dirs = sorted((REPO_ROOT / "models" / "_sweep").glob("*"))
    for rd in reversed(reg_dirs):
        ragg = pd.read_parquet(rd / "aggregate.parquet")
        if len(ragg) >= 12:
            reg_sweep_dir = rd
            break
    ragg = pd.read_parquet(reg_sweep_dir / "aggregate.parquet")
    two_stage = ragg[ragg["variant"] == "lightgbm_two_stage"][
        ["scale_idx", "tile_size_px", "presence_auc_mean", "presence_auc_std"]
    ].rename(columns={"presence_auc_mean": "two_stage_presence_auc", "presence_auc_std": "ts_std"})
    classifier = agg[agg["target_id"] == "bc_ge_1"][
        ["scale_idx", "tile_size_px", "auc_mean", "auc_std"]
    ].rename(columns={"auc_mean": "classifier_auc", "auc_std": "clf_std"})
    h2h = classifier.merge(two_stage, on=["scale_idx", "tile_size_px"]).sort_values("scale_idx")
    h2h["delta"] = h2h["classifier_auc"] - h2h["two_stage_presence_auc"]
    print(h2h.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    print(f"\nMean delta (classifier - two_stage_presence): {h2h['delta'].mean():+.4f}")
    print(f"  classifier wins in {(h2h['delta'] > 0).sum()}/{len(h2h)} matched scales")

    # ---------- 7. Two-stage embedded presence head: pull its per-fold AUC table too ----------
    print(f"\n## 7. Sign-test on the binary classifier vs regression-presence head (paired)\n")
    # Use summary parquets at the per-fold level for a Wilcoxon-style sign test
    # of paired (fold, scale) comparisons.
    rs = pd.read_parquet(reg_sweep_dir / "summary.parquet")
    rs_ts = rs[(rs["variant"] == "lightgbm_two_stage") & (~rs["is_specificity_only"])][
        ["scale_idx", "fold_idx", "held_out_obs_id", "presence_auc"]
    ].rename(columns={"presence_auc": "two_stage_presence_auc"})
    bs_bc = real_summary[real_summary["target_id"] == "bc_ge_1"][
        ["scale_idx", "fold_idx", "held_out_obs_id", "auc"]
    ].rename(columns={"auc": "classifier_auc"})
    paired = rs_ts.merge(bs_bc, on=["scale_idx", "fold_idx", "held_out_obs_id"])
    paired["delta"] = paired["classifier_auc"] - paired["two_stage_presence_auc"]
    n_wins = int((paired["delta"] > 0).sum())
    n_total = len(paired)
    p_wins = binomtest(n_wins, n_total, 0.5, alternative="two-sided").pvalue
    print(f"Paired folds: classifier > two_stage in {n_wins}/{n_total}")
    print(f"Mean paired delta: {paired['delta'].mean():+.4f} (std {paired['delta'].std():+.4f})")
    print(f"Two-sided sign test p = {p_wins:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
