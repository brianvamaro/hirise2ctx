"""W2 S=32 HELD-OUT CONFIRMATION of the seed-ensemble + fusion recipe.

The recipe (3-seed cell-A ensemble for within-image ranking, fused with the
Tier-1 LightGBM for image-level scale) was assembled after seeing the S=64
per-seed results, so S=32 is its held-out test. Pre-declared read
(DECISIONS.md 2026-06-11 3-seed entry, BEFORE the S=32 runs finished):

  CONFIRMED iff (a) the 3-seed ensemble passes the per-image gate vs the
  S=32 Tier-1 baseline (median paired dAUC >= +0.05, Wilcoxon p < 0.05,
  dossier validity-passing images) AND (b) fusion recovers pooled PR-AUC
  >= that baseline. F1 if pooled PR-AUC is binding, F3 if per-image is.

Finds the three seed runs by glob (each seed gets its own config-hash dir,
per the sweep-artifact-layout gotcha).
"""
import json
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

CNN_GLOB = "models/cnn_bce_S32/*/scale_S32_tfa_gt_1e-2_aug_none"
T1_PREDS = REPO_ROOT / ("models/lightgbm_classification/2d046f48c722f0a5/"
                        "scale_S32_tfa_gt_1e-2/predictions.parquet")
T1_SUMMARY = REPO_ROOT / "models/_sweep_binary/20260612T062412Z/summary.parquet"
DOSSIER = REPO_ROOT / "dataset_v2/w1_dossier.parquet"
SCALE_IDX = 2


def per_image_auc(df: pd.DataFrame, col: str) -> pd.Series:
    out = {}
    for obs, g in df.groupby("obs_id"):
        y = g["y_true"].to_numpy()
        out[obs] = roc_auc_score(y, g[col].to_numpy()) if 0 < y.sum() < y.size else np.nan
    return pd.Series(out)


def main() -> int:
    cell_dirs = sorted(REPO_ROOT.glob(CNN_GLOB))
    seeds = {}
    for d in cell_dirs:
        snap = json.loads((d / "snapshot.json").read_text())
        seeds[snap["model"]["params"]["seed"]] = d / "predictions.parquet"
    print(f"found seeds: {sorted(seeds)}  ({len(seeds)} runs)")
    assert len(seeds) == 3, "expected exactly 3 seed runs at S=32"

    base = None
    for seed in sorted(seeds):
        p = pd.read_parquet(seeds[seed], columns=["obs_id", "ti", "tj", "y_true", "y_pred"])
        p = p.rename(columns={"y_pred": f"p{seed}"})
        base = p if base is None else base.merge(
            p.drop(columns="y_true"), on=["obs_id", "ti", "tj"], validate="one_to_one")
    t1 = pd.read_parquet(T1_PREDS, columns=["obs_id", "ti", "tj", "y_pred"])
    t1 = t1.rename(columns={"y_pred": "t1_prob"})
    df = base.merge(t1, on=["obs_id", "ti", "tj"], how="inner", validate="one_to_one")
    assert len(df) == len(base), f"join loss: {len(base)} -> {len(df)}"
    y = df["y_true"].to_numpy().astype(int)

    cols = [f"p{s}" for s in sorted(seeds)]
    df["ens_mean"] = df[cols].mean(axis=1)
    df["ens_q"] = df.groupby("obs_id")["ens_mean"].transform(lambda s: rankdata(s) / len(s))
    df["t1_image_mean"] = df.groupby("obs_id")["t1_prob"].transform("mean")
    df["F1_ens"] = df["ens_q"] * df["t1_image_mean"]
    df["F3_ens"] = 0.5 * (rankdata(df["ens_mean"]) + rankdata(df["t1_prob"])) / len(df)

    dossier = pd.read_parquet(DOSSIER)
    vok = set(dossier[dossier.validity_ok].index)
    t1_auc = per_image_auc(df.assign(_s=df["t1_prob"]), "_s")
    t1_pooled = float(average_precision_score(y, df["t1_prob"]))
    t1_sum = pd.read_parquet(T1_SUMMARY)
    print(f"S=32 Tier-1 baseline: pooled PR-AUC={t1_pooled:.4f}  "
          f"per-img AUC median={t1_auc.median():.4f} "
          f"(sweep summary mean {t1_sum['auc'].mean():.4f})")

    k = max(1, int(0.05 * y.size))
    print(f"n tiles pooled: {len(df)}  base rate: {y.mean():.4f}  "
          f"validity-passing imgs: {len(vok)}\n")
    print(f"{'variant':<12s} {'pooled_pr':>9s} {'prec@5%':>8s} {'med_auc':>8s} "
          f"{'dAUC_med(v)':>11s} {'win':>5s} {'p':>8s}")
    results = {}
    for col in cols + ["ens_mean", "F1_ens", "F3_ens", "t1_prob"]:
        s = df[col].to_numpy()
        pr = float(average_precision_score(y, s))
        p5 = float(y[np.argsort(-s)[:k]].mean())
        aucs = per_image_auc(df, col)
        med = float(aucs.median())
        if col == "t1_prob":
            print(f"{col:<12s} {pr:>9.4f} {p5:>8.4f} {med:>8.4f} {'--':>11s} {'--':>5s} {'--':>8s}")
            continue
        d = (aucs - t1_auc).dropna()
        d_v = d[[o in vok for o in d.index]]
        try:
            pval = float(stats.wilcoxon(d_v, zero_method="wilcox").pvalue)
        except ValueError:
            pval = float("nan")
        results[col] = dict(pooled=pr, d_med=float(d_v.median()), p=pval)
        print(f"{col:<12s} {pr:>9.4f} {p5:>8.4f} {med:>8.4f} {d_v.median():>+11.4f} "
              f"{(d_v > 0).mean():>5.2f} {pval:>8.4f}")

    gate_a = (results["ens_mean"]["d_med"] >= 0.05) and (results["ens_mean"]["p"] < 0.05)
    gate_b = max(results["F1_ens"]["pooled"], results["F3_ens"]["pooled"]) >= t1_pooled
    print(f"\nPRE-DECLARED READ: (a) ensemble per-image gate "
          f"{'PASS' if gate_a else 'FAIL'} "
          f"(d_med={results['ens_mean']['d_med']:+.4f}, p={results['ens_mean']['p']:.4f}); "
          f"(b) fusion pooled >= Tier1 {'PASS' if gate_b else 'FAIL'} "
          f"(best fusion {max(results['F1_ens']['pooled'], results['F3_ens']['pooled']):.4f} "
          f"vs {t1_pooled:.4f})")
    print(f"=> recipe {'CONFIRMED at S=32' if (gate_a and gate_b) else 'NOT confirmed at S=32'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
