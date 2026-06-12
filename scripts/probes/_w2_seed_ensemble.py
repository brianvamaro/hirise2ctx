"""W2 follow-up: 3-seed ensemble of cell A, raw and fused with Tier 1.

Motivation: the pre-declared gate (median paired per-image dAUC >= +0.05 vs
Tier 1, Wilcoxon p < 0.05) passed on seed 0 but FAILED on seeds 1 and 2
(+0.038 p=0.059; +0.005 p=0.66), and pooled PR-AUC swings 0.49-0.56 across
seeds. Seed-averaging is the standard variance reduction (and the lit-review
Tier-2 plan calls for a small ensemble anyway, canopy-height style). This
probe asks: does mean-of-3-seeds + score fusion give a SEED-FREE promotable
recipe?

Variants:
  ens_mean        mean of the 3 seeds' probabilities
  ens_rank        mean of the 3 seeds' pooled ranks
  F1(ens)         within-image quantile of ens_mean x Tier-1 image mean
  F3(ens)         0.5*(pooled_rank(ens_mean) + pooled_rank(t1))

Gate stats are computed exactly as in _w2_cnn_verdict.py: paired per-image
AUC deltas vs the Tier 1 summary on dossier validity-passing images.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

SEED_PREDS = {
    0: "models/cnn_bce_S64/40d843617a09e3c7/scale_S64_tfa_gt_1e-2_aug_none/predictions.parquet",
    1: "models/cnn_bce_S64/73edf3e7abbbb363/scale_S64_tfa_gt_1e-2_aug_none/predictions.parquet",
    2: "models/cnn_bce_S64/a596167d2dfbae2d/scale_S64_tfa_gt_1e-2_aug_none/predictions.parquet",
}
T1_PREDS = REPO_ROOT / ("models/lightgbm_classification/99de85c1ad2a72e6/"
                        "scale_S64_tfa_gt_1e-2/predictions.parquet")
T1_SUMMARY = REPO_ROOT / "models/_sweep_binary/20260611T214042Z/summary.parquet"
DOSSIER = REPO_ROOT / "dataset_v2/w1_dossier.parquet"
T1_POOLED = 0.5651  # tier1_pooled_pr_auc(), computed by _w2_cnn_verdict.py


def per_image_auc(df: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for obs, g in df.groupby("obs_id"):
        y = g["y_true"].to_numpy()
        out[obs] = roc_auc_score(y, g[col].to_numpy()) if 0 < y.sum() < y.size else np.nan
    return pd.Series(out)


def main() -> int:
    base = None
    for seed, rel in SEED_PREDS.items():
        p = pd.read_parquet(REPO_ROOT / rel, columns=["obs_id", "ti", "tj", "y_true", "y_pred"])
        p = p.rename(columns={"y_pred": f"p{seed}"})
        base = p if base is None else base.merge(
            p.drop(columns="y_true"), on=["obs_id", "ti", "tj"], validate="one_to_one")
    t1 = pd.read_parquet(T1_PREDS, columns=["obs_id", "ti", "tj", "y_pred"])
    t1 = t1.rename(columns={"y_pred": "t1_prob"})
    df = base.merge(t1, on=["obs_id", "ti", "tj"], how="inner", validate="one_to_one")
    assert len(df) == len(base), f"join loss: {len(base)} -> {len(df)}"
    y = df["y_true"].to_numpy().astype(int)

    seeds = [f"p{s}" for s in SEED_PREDS]
    df["ens_mean"] = df[seeds].mean(axis=1)
    df["ens_rank"] = np.mean([rankdata(df[c]) for c in seeds], axis=0) / len(df)
    df["ens_q"] = df.groupby("obs_id")["ens_mean"].transform(lambda s: rankdata(s) / len(s))
    df["t1_image_mean"] = df.groupby("obs_id")["t1_prob"].transform("mean")
    df["F1_ens"] = df["ens_q"] * df["t1_image_mean"]
    df["F3_ens"] = 0.5 * (rankdata(df["ens_mean"]) + rankdata(df["t1_prob"])) / len(df)

    dossier = pd.read_parquet(DOSSIER)
    vok = set(dossier[dossier.validity_ok].index)
    t1_auc = pd.read_parquet(T1_SUMMARY).set_index("held_out_obs_id")["auc"]

    k = max(1, int(0.05 * y.size))
    print(f"n tiles pooled: {len(df)}  base rate: {y.mean():.4f}  validity-passing imgs: {len(vok)}\n")
    print(f"{'variant':<22s} {'pooled_pr':>9s} {'prec@5%':>8s} {'med_auc':>8s} "
          f"{'dAUC_med(v)':>11s} {'win':>5s} {'p':>8s}  gate")
    for col in ["p0", "p1", "p2", "ens_mean", "ens_rank", "F1_ens", "F3_ens", "t1_prob"]:
        s = df[col].to_numpy()
        pr = average_precision_score(y, s)
        p5 = float(y[np.argsort(-s)[:k]].mean())
        aucs = per_image_auc(df, col)
        d = (aucs - t1_auc).dropna()
        d_v = d[[o in vok for o in d.index]]
        med = float(aucs.median())
        if col == "t1_prob":
            print(f"{col:<22s} {pr:>9.4f} {p5:>8.4f} {med:>8.4f} {'--':>11s} {'--':>5s} {'--':>8s}")
            continue
        try:
            pval = stats.wilcoxon(d_v, zero_method="wilcox").pvalue
        except ValueError:
            pval = float("nan")
        gate_auc = (d_v.median() >= 0.05) and (pval < 0.05)
        gate_pr = (pr - T1_POOLED) >= 0.03
        verdict = ("PASS(auc)" if gate_auc else "") + ("+PASS(pr)" if gate_pr else "")
        print(f"{col:<22s} {pr:>9.4f} {p5:>8.4f} {med:>8.4f} {d_v.median():>+11.4f} "
              f"{(d_v > 0).mean():>5.2f} {pval:>8.4f}  {verdict or 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
