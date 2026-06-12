"""W2 follow-up (free test): "CNN ranks, GBM scales" score fusion.

Diagnosis (_w2_midgrid_diag.py): cell A's CNN ranks tiles well WITHIN images
(per-image AUC median 0.694, beats Tier 1 paired p=0.016) but its image-level
score barely tracks the image's true base rate (rank-corr +0.22 vs Tier 1's
+0.41), so the POOLED ranking interleaves images badly (PR-AUC 0.510 vs
Tier 1's 0.565). Hypothesis: replacing the CNN's cross-image leveling with
Tier 1's, while keeping the CNN's within-image ordering, recovers pooled
PR-AUC. Inference-compatible: both models run on any CTX window.

Variants (all preserve or blend within-image order):
  F1 cnn_rank * t1_image_mean   -- CNN within-image quantile scaled by the
                                   image's mean Tier-1 probability (pure test:
                                   within-image AUC identical to CNN's)
  F2 sqrt(cnn_prob * t1_prob)   -- tile-level geometric blend
  F3 0.5*(pooled_rank(cnn) + pooled_rank(t1)) -- rank-average ensemble

Usage: python _w2_fusion.py [cnn_predictions_parquet]
       (default: the cell-A seed-0 artifact)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

CNN_PREDS = (Path(sys.argv[1]) if len(sys.argv) > 1 else
             REPO_ROOT / ("models/cnn_bce_S64/40d843617a09e3c7/"
                          "scale_S64_tfa_gt_1e-2_aug_none/predictions.parquet"))
T1_PREDS = REPO_ROOT / ("models/lightgbm_classification/99de85c1ad2a72e6/"
                        "scale_S64_tfa_gt_1e-2/predictions.parquet")


def pooled_metrics(y: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    k = max(1, int(0.05 * y.size))
    top = np.argsort(-s)[:k]
    return float(average_precision_score(y, s)), float(y[top].mean())


def per_image_auc_median(df: pd.DataFrame, col: str) -> float:
    aucs = []
    for _, g in df.groupby("obs_id"):
        y = g["y_true"].to_numpy()
        if 0 < y.sum() < y.size:
            aucs.append(roc_auc_score(y, g[col].to_numpy()))
    return float(np.median(aucs))


def main() -> int:
    cnn = pd.read_parquet(CNN_PREDS, columns=["obs_id", "ti", "tj", "y_true", "y_pred"])
    t1 = pd.read_parquet(T1_PREDS, columns=["obs_id", "ti", "tj", "y_pred"])
    t1 = t1.rename(columns={"y_pred": "t1_prob"})
    df = cnn.merge(t1, on=["obs_id", "ti", "tj"], how="inner", validate="one_to_one")
    assert len(df) == len(cnn), f"join loss: {len(cnn)} -> {len(df)}"
    y = df["y_true"].to_numpy().astype(int)

    # Within-image quantile of the CNN score.
    df["cnn_rank"] = df.groupby("obs_id")["y_pred"].transform(
        lambda s: rankdata(s) / len(s))
    df["t1_image_mean"] = df.groupby("obs_id")["t1_prob"].transform("mean")

    scores = {
        "cnn_raw (cell A)": df["y_pred"].to_numpy(),
        "tier1 (ref)": df["t1_prob"].to_numpy(),
        "F1 cnn_rank * t1_image_mean": (df["cnn_rank"] * df["t1_image_mean"]).to_numpy(),
        "F2 sqrt(cnn*t1) tile blend": np.sqrt(df["y_pred"].to_numpy()
                                              * df["t1_prob"].to_numpy()),
        "F3 pooled-rank average": 0.5 * (rankdata(df["y_pred"]) + rankdata(df["t1_prob"]))
                                  / len(df),
    }
    print(f"n tiles pooled: {len(df)}  base rate: {y.mean():.4f}\n")
    print(f"{'variant':<32s} {'pooled_pr_auc':>13s} {'prec@5%':>8s} {'med per-img AUC':>16s}")
    for name, s in scores.items():
        df["_s"] = s
        pr, p5 = pooled_metrics(y, s)
        med = per_image_auc_median(df, "_s")
        print(f"{name:<32s} {pr:>13.4f} {p5:>8.4f} {med:>16.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
