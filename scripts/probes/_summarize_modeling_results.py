"""Synthesis probe: compute the diagnostics that justify modeling-results.md.

Produces a single stdout blob that the writer can transcribe into prose:
  - GBM headline table (mean +/- std per variant x scale)
  - Sign-test on Spearman rho (12 variant x scale combos)
  - Sign-test on presence AUC > 0.5
  - Best-fold-by-BoulderLabel breakdown
  - Per-bin RMSE: shape across scales
  - Feature importance: top-N across variants
  - CNN headline + suspected collapse evidence

Reads from the most-recent sweep + the per-(variant, scale) metrics.json files
that scripts/sweep.py (post-2026-05-26 fix) and scripts/train_gbm.py both write.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401 -- DLL bootstrap on Windows

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import load_config
import src.manifest as M


def main() -> int:
    cfg = load_config("config.yaml")
    manifest = M.load_manifest(cfg.manifest_path)
    obs_to_label = dict(zip(manifest["ObsId"], manifest["BoulderLabel"]))

    sweep_root = REPO_ROOT / "models" / "_sweep"
    # Pick the most-recent FULL (12-row) sweep, not the smoke-test partial.
    sweep_dirs = sorted(sweep_root.glob("*"))
    for sd in reversed(sweep_dirs):
        agg = pd.read_parquet(sd / "aggregate.parquet")
        if len(agg) >= 12:
            sweep_dir = sd
            break
    else:
        raise RuntimeError("No full (12-row) sweep found")

    print(f"# Sweep dir: {sweep_dir.name}\n")

    agg = pd.read_parquet(sweep_dir / "aggregate.parquet").sort_values(
        ["scale_idx", "variant"]
    )
    summary = pd.read_parquet(sweep_dir / "summary.parquet")
    summary["boulder_label"] = summary["held_out_obs_id"].map(obs_to_label).fillna("empty")

    # ---------- 1. GBM headline table ----------
    print("## 1. GBM headline (12 variant x scale, mean ± std over 8 real folds)\n")
    print(agg[["variant", "scale_idx", "tile_size_px",
               "spearman_rho_mean", "spearman_rho_std",
               "presence_auc_mean", "presence_auc_std", "rmse_log1p_mean"]
               ].to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

    # ---------- 2. Sign-test diagnostics ----------
    rho_means = agg["spearman_rho_mean"].to_numpy()
    auc_means = agg["presence_auc_mean"].to_numpy()
    n_pos_rho = int((rho_means > 0).sum())
    n_pos_auc = int((auc_means > 0.5).sum())
    n = len(agg)

    from scipy.stats import binomtest
    p_rho = binomtest(n_pos_rho, n, 0.5, alternative="greater").pvalue
    p_auc = binomtest(n_pos_auc, n, 0.5, alternative="greater").pvalue
    print(f"\n## 2. Sign-tests (12 variant x scale combos)\n")
    print(f"Spearman rho > 0 in {n_pos_rho}/{n} combos. Sign-test p = {p_rho:.4f}")
    print(f"Presence AUC > 0.5 in {n_pos_auc}/{n} combos. Sign-test p = {p_auc:.4f}")
    print(f"Mean Spearman across all 12: {rho_means.mean():+.4f}")
    print(f"Mean AUC across all 12: {auc_means.mean():+.4f}")

    # Per-fold sign-test: across the 12*8 = 96 (variant, scale, fold) cells,
    # how many fold-level rho > 0?
    real_summary = summary[summary["boulder_label"] != "empty"].copy()
    fold_rho = real_summary["spearman_rho"].dropna().to_numpy()
    n_pos_fold = int((fold_rho > 0).sum())
    n_fold = len(fold_rho)
    p_fold = binomtest(n_pos_fold, n_fold, 0.5, alternative="greater").pvalue
    print(f"\nPer-fold rho > 0: {n_pos_fold}/{n_fold} (over all variants, scales, real folds)")
    print(f"  Sign-test p = {p_fold:.4f}")

    # ---------- 3. Per-fold breakdown by BoulderLabel ----------
    print(f"\n## 3. Per-fold rho by held-out BoulderLabel (all 12 variant x scale combos pooled)\n")
    by_label = real_summary.groupby("boulder_label")["spearman_rho"].agg(["count", "mean", "std", "min", "max"])
    print(by_label.to_string(float_format=lambda x: f"{x:+.4f}"))

    print(f"\n## 3b. Per-fold AUC by held-out BoulderLabel (including empty/specificity fold)\n")
    by_label_auc = summary.groupby("boulder_label")["presence_auc"].agg(["count", "mean", "std", "min", "max"])
    print(by_label_auc.to_string(float_format=lambda x: f"{x:+.4f}"))

    # ---------- 4. Best variant per scale + worst ----------
    print(f"\n## 4. Best vs worst variant per scale (by Spearman mean)\n")
    for s in sorted(agg["scale_idx"].unique()):
        sub = agg[agg["scale_idx"] == s].sort_values("spearman_rho_mean", ascending=False)
        best = sub.iloc[0]
        worst = sub.iloc[-1]
        tile = int(best["tile_size_px"])
        print(f"  S={tile:>2d}: best={best['variant']:<22s} {best['spearman_rho_mean']:+.4f} +/- {best['spearman_rho_std']:.4f}   "
              f"worst={worst['variant']:<22s} {worst['spearman_rho_mean']:+.4f}")

    # ---------- 5. Per-bin RMSE shape ----------
    print(f"\n## 5. Per-bin RMSE shape (across scales for the headline variant: two_stage)\n")
    for scale in (0, 1, 2, 3):
        tile = 2 ** (3 + scale)
        ddirs = sorted((REPO_ROOT / "models" / "lightgbm_two_stage").glob(f"*/scale_S{tile}/metrics.json"),
                        key=lambda p: p.stat().st_mtime)
        if not ddirs:
            print(f"  S={tile}: no two_stage metrics.json")
            continue
        m = json.loads(ddirs[-1].read_text())
        # Mean RMSE across folds for each bin
        bins = {}
        for f in m["per_fold"]:
            for b in f.get("per_bin_rmse", []):
                bins.setdefault(b["bin"], []).append(b["rmse"])
        bin_order = ["zero", "0_to_1e-4", "1e-4_to_1e-3", "1e-3_to_1e-2", "1e-2_to_max"]
        print(f"  S={tile} two_stage  per-bin mean RMSE:")
        for b in bin_order:
            vals = bins.get(b, [])
            if vals:
                print(f"    {b:<15s}  {np.mean(vals):.4e}  (n_folds={len(vals)})")
            else:
                print(f"    {b:<15s}  -")

    # ---------- 6. Feature importance ----------
    print(f"\n## 6. Top-10 features by mean split-gain (Tweedie @ each scale)\n")
    for scale in (0, 1, 2, 3):
        tile = 2 ** (3 + scale)
        scale_dirs = sorted((REPO_ROOT / "models" / "lightgbm_tweedie").glob(f"*/scale_S{tile}"),
                             key=lambda p: p.stat().st_mtime)
        if not scale_dirs:
            print(f"  S={tile}: no tweedie boosters")
            continue
        boosters = sorted(scale_dirs[-1].glob("fold_*/booster.txt"))
        if not boosters:
            print(f"  S={tile}: tweedie scale dir has no booster files")
            continue
        imps = []
        for bp in boosters:
            b = lgb.Booster(model_str=bp.read_text(encoding="utf-8"))
            imp = pd.Series(b.feature_importance(importance_type="gain"),
                            index=b.feature_name())
            imps.append(imp)
        mean_imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
        total = mean_imp.sum()
        if total == 0:
            print(f"  S={tile} tweedie: zero gain across all features (no useful splits)")
            continue
        print(f"  S={tile} tweedie (top 10 of {len(mean_imp)} features, % of total gain):")
        for name, val in mean_imp.head(10).items():
            print(f"    {name:<35s}  {100*val/total:>5.1f} %")

    # ---------- 7. CNN ----------
    print(f"\n## 7. CNN headline + collapse diagnosis\n")
    for patch in (32, 64):
        cnn_root = REPO_ROOT / "models" / f"cnn_log1p_huber_S{patch}"
        runs = sorted(cnn_root.glob("*"))
        if not runs:
            print(f"  S={patch}: no CNN runs")
            continue
        scale_dirs = sorted(runs[-1].glob(f"scale_S{patch}_P{patch}"))
        if not scale_dirs:
            continue
        m = json.loads((scale_dirs[0] / "metrics.json").read_text())
        agg_m = m["aggregate"]
        print(f"  S={patch}:")
        print(f"    Spearman rho (mean +/- std) = {agg_m['spearman_rho_mean']:+.4f} +/- {agg_m['spearman_rho_std']:.4f}")
        print(f"    Presence AUC (mean +/- std) = {agg_m['presence_auc_mean']:+.4f} +/- {agg_m['presence_auc_std']:.4f}")
        # Predictions parquet contains y_pred per tile -- check how concentrated
        pred_path = scale_dirs[0] / "predictions.parquet"
        if pred_path.exists():
            pred = pd.read_parquet(pred_path)
            yp = pred["y_pred"].to_numpy()
            yt = pred["y_true"].to_numpy()
            zero_truth_pct = float((yt == 0).mean() * 100)
            zero_pred_pct = float((yp < 1e-6).mean() * 100)
            print(f"    y_true: {zero_truth_pct:.1f}% are exact zero")
            print(f"    y_pred: {zero_pred_pct:.1f}% predicted < 1e-6 (effectively zero)")
            print(f"    y_pred range: [{yp.min():.2e}, {yp.max():.2e}]   median = {np.median(yp):.2e}")
            print(f"    y_true range: [{yt.min():.2e}, {yt.max():.2e}]   median = {np.median(yt):.2e}")

    # ---------- 8. Dataset ceiling ----------
    print(f"\n## 8. Target distribution sanity (the 0.27 ceiling)\n")
    # Pool all summary fold rows to get aggregate truth from predictions
    sample_paths = list((REPO_ROOT / "models" / "lightgbm_tweedie").glob("*/scale_S64/predictions.parquet"))
    if sample_paths:
        pred = pd.read_parquet(sorted(sample_paths)[-1])
        yt = pred["y_true"].to_numpy()
        print(f"  S=64 truth distribution across all 9 folds (n={len(yt)}):")
        print(f"    fraction == 0:           {(yt == 0).mean()*100:.2f} %")
        print(f"    fraction in (0, 0.001]:  {((yt > 0) & (yt <= 0.001)).mean()*100:.2f} %")
        print(f"    fraction in (0.001, 0.01]: {((yt > 0.001) & (yt <= 0.01)).mean()*100:.2f} %")
        print(f"    fraction > 0.01:         {(yt > 0.01).mean()*100:.2f} %")
        print(f"    max value:               {yt.max():.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
