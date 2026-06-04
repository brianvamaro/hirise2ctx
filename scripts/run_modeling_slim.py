"""Slim modeling runner -- 5-feature LightGBM on the 36-image v2 cohort.

A deliberately simple LightGBM model for predicting per-tile meter-scale boulder
count from CTX texture features, used as the reportable modelling variant.

Feature set (5):
  - shadow_fraction         : fraction of in-tile CTX pixels below the shadow-DN threshold
  - shadow_fraction_strict  : tighter shadow threshold
  - bright_cap_fraction     : fraction of in-tile pixels above the bright threshold
  - grad_mag_std            : Sobel gradient-magnitude standard deviation (texture)
  - intensity_std           : per-tile pixel-value standard deviation (texture)

Cohort: 36 of 38 v2 manifest images (drops ESP_017355_2260 and ESP_076499_1160,
which are manifest "unknown"-BoulderLabel diversity picks rather than part of
the boulder-rich/poor cohort).

Target: ``boulder_count`` (number of detected boulder polygons per tile).
Variant: ``lightgbm_two_stage_balanced`` with default LightGBM hyperparameters.
Scale: S=64 (320 m tile size).
CV: leave-image-out on the 36-image cohort.

Outputs:
  dataset_v2/modeling_slim_predictions.parquet    : per-tile predictions
  dataset_v2/modeling_slim_summary.parquet        : per-fold + pooled metrics

Run via:
    conda run --no-capture-output -n geospatial python -u scripts/run_modeling_slim.py
"""
from __future__ import annotations

import functools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src import modeling  # noqa: E402 -- Windows OpenMP fix
from src.modeling.gbm import LGBMParams, LightGBMTwoStageBalanced  # noqa: E402
from src.modeling.loaders import load_fold, load_metadata  # noqa: E402

print = functools.partial(print, flush=True)

SLIM_FEATURES = [
    "shadow_fraction",
    "shadow_fraction_strict",
    "bright_cap_fraction",
    "grad_mag_std",
    "intensity_std",
]
EXCLUDE_OBS = {"ESP_017355_2260", "ESP_076499_1160"}  # unknown BoulderLabel
SCHEME = "loio_nfold"
SCALE_IDX = 3  # S=64
TARGET_COL = "boulder_count"
FA_RICH_THRESHOLD = 1e-2  # the operationally meaningful "boulder-rich" cut
DATASET_DIR = REPO_ROOT / "dataset_v2"

OUT_PRED = DATASET_DIR / "modeling_slim_predictions.parquet"
OUT_SUMMARY = DATASET_DIR / "modeling_slim_summary.parquet"


def make_variant() -> LightGBMTwoStageBalanced:
    """Fresh variant instance with project-default LightGBM hyperparameters."""
    params = LGBMParams(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=64,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        early_stopping_rounds=50,
        seed=0,
    )
    return LightGBMTwoStageBalanced(params=params)


def per_image_auc_fa_rich(y_truth_fa: np.ndarray, pred: np.ndarray) -> float:
    """ROC-AUC at the fa_gt_1e-2 threshold; NaN if single-class."""
    y_binary = (y_truth_fa >= FA_RICH_THRESHOLD).astype(int)
    if y_binary.sum() == 0 or y_binary.sum() == len(y_binary):
        return float("nan")
    return float(roc_auc_score(y_binary, pred))


def main() -> int:
    t0 = time.time()
    meta = load_metadata(SCHEME, dataset_dir=DATASET_DIR)
    n_folds = len(meta["per_fold"])
    print(f"[slim] scheme = {SCHEME}, n_folds = {n_folds}, scale_idx = {SCALE_IDX}")
    print(f"[slim] features (slim) = {SLIM_FEATURES}")
    print(f"[slim] target = {TARGET_COL}; variant = lightgbm_two_stage_balanced")
    print(f"[slim] excluding from train + eval: {sorted(EXCLUDE_OBS)}")

    # We will collect predictions across folds for both slim + full
    pred_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for fold_idx in range(n_folds):
        fold_meta = meta["per_fold"][fold_idx]
        held_out = fold_meta["test_obs_ids"]
        if any(o in EXCLUDE_OBS for o in held_out):
            print(f"[slim] fold {fold_idx} held-out = {held_out} -> SKIP (excluded)")
            continue

        fold = load_fold(SCHEME, fold_idx, scale_idx=SCALE_IDX,
                        dataset_dir=DATASET_DIR)

        # Drop excluded ObsIds from the training set
        obs_to_int = fold.obs_to_int
        excluded_codes = {obs_to_int[o] for o in EXCLUDE_OBS if o in obs_to_int}
        train_keep = ~np.isin(fold.groups_train, list(excluded_codes))

        X_train_full = fold.X_train[train_keep]
        y_train_full = fold.y_train.loc[train_keep].reset_index(drop=True)
        n_train_kept = int(train_keep.sum())
        n_train_dropped = int((~train_keep).sum())

        # Subset to slim feature columns
        feat_idx = [fold.feature_names.index(f) for f in SLIM_FEATURES]
        X_train = X_train_full[:, feat_idx]
        X_test = fold.X_test[:, feat_idx]

        # Target -- two-stage takes y as a vector
        y_train = y_train_full[TARGET_COL].to_numpy(dtype=np.float64)
        y_test = fold.y_test[TARGET_COL].to_numpy(dtype=np.float64)
        y_test_fa = fold.y_test["fractional_area"].to_numpy(dtype=np.float64)

        # Train + predict
        model = make_variant()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        # Per-fold metrics
        rho, _ = stats.spearmanr(pred, y_test)
        auc = per_image_auc_fa_rich(y_test_fa, pred)

        held_str = ",".join(held_out)
        n_test = len(pred)
        n_rich = int((y_test_fa >= FA_RICH_THRESHOLD).sum())
        print(f"[slim] fold {fold_idx:2d} {held_str}: n_test={n_test:6d} n_rich={n_rich:5d} "
              f"n_train(kept/dropped)={n_train_kept}/{n_train_dropped}  "
              f"rho={rho:+.4f}  AUC@fa_gt_1e-2={auc:.3f}")

        summary_rows.append({
            "fold_idx": fold_idx,
            "held_out_obs_id": held_str,
            "n_test_tiles": n_test,
            "n_test_rich_fa_ge_1e-2": n_rich,
            "n_train_kept": n_train_kept,
            "n_train_dropped_excluded": n_train_dropped,
            "rho": float(rho),
            "auc_fa_rich": auc,
        })

        # Save predictions
        pred_rows.append(pd.DataFrame({
            "obs_id": held_str,
            "fold_idx": fold_idx,
            "ti": fold.keys_test["ti"].to_numpy(),
            "tj": fold.keys_test["tj"].to_numpy(),
            "fractional_area_truth": y_test_fa,
            "boulder_count_truth": y_test,
            "pred": pred,
        }))

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)

    print(f"\n[slim] aggregated across {len(summary)} folds")

    # Pooled Spearman across all test tiles
    pooled_rho, pooled_p = stats.spearmanr(
        predictions["pred"], predictions["boulder_count_truth"])
    print(f"[slim] POOLED Spearman rho (all {len(predictions)} tiles across folds): "
          f"{pooled_rho:+.4f}  (p = {pooled_p:.2e})")

    valid = summary["rho"].dropna()
    print(f"[slim] PER-FOLD rho (n={len(valid)}):")
    print(f"         mean = {valid.mean():+.4f}, std = {valid.std():.4f}, "
          f"median = {valid.median():+.4f}")

    valid_auc = summary["auc_fa_rich"].dropna()
    print(f"[slim] PER-IMAGE AUC at fa_gt_1e-2 "
          f"(n={len(valid_auc)} folds with both classes):")
    print(f"         median = {valid_auc.median():.3f}, "
          f"min = {valid_auc.min():.3f}, max = {valid_auc.max():.3f}")
    print(f"         frac >= 0.70 = {(valid_auc >= 0.70).mean():.2f}, "
          f"frac < 0.50 = {(valid_auc < 0.50).mean():.2f}")

    # Add pooled row to summary
    pooled_row = {
        "fold_idx": -1, "held_out_obs_id": "POOLED",
        "n_test_tiles": int(predictions.shape[0]),
        "n_test_rich_fa_ge_1e-2": int(
            (predictions["fractional_area_truth"] >= FA_RICH_THRESHOLD).sum()),
        "n_train_kept": -1, "n_train_dropped_excluded": -1,
        "rho": float(pooled_rho),
        "auc_fa_rich": float("nan"),
    }
    summary_with_pooled = pd.concat(
        [summary, pd.DataFrame([pooled_row])], ignore_index=True)

    print(f"\n[slim] writing {OUT_PRED} ({len(predictions)} rows)")
    predictions.to_parquet(OUT_PRED, index=False)
    print(f"[slim] writing {OUT_SUMMARY} ({len(summary_with_pooled)} rows)")
    summary_with_pooled.to_parquet(OUT_SUMMARY, index=False)
    print(f"\n[slim] DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
