"""PLAN_FM 2.1c: paired per-image statistics BETWEEN bake-off heads.

The pooled PR-AUC gaps between the top heads (0.74-0.79) are within the
per-image fold-ripple, so the honest winner call is paired: per-image AUC
deltas between each head pair on dossier validity-passing images, Wilcoxon +
median + win rate. All from cached predictions (heads_* dirs, emb matrix);
mlp_ens3 is rebuilt from the three seed prediction parquets.

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/probes/_w2_fang_head_pairs.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

PROBE = REPO_ROOT / "models" / "fang_probe"
DOSSIER = REPO_ROOT / "dataset_v2/w1_dossier.parquet"
HEADS = ["lgbm", "logreg", "knn50", "mlp_ens3"]


def load_preds(head: str) -> pd.DataFrame:
    if head == "mlp_ens3":
        base = None
        for s in (0, 1, 2):
            p = pd.read_parquet(
                sorted(PROBE.glob(f"heads_mlp_seed{s}/*/predictions.parquet"))[0],
                columns=["obs_id", "ti", "tj", "y_true", "y_pred"])
            p = p.rename(columns={"y_pred": f"p{s}"})
            base = p if base is None else base.merge(
                p.drop(columns="y_true"), on=["obs_id", "ti", "tj"], validate="one_to_one")
        base["y_pred"] = base[["p0", "p1", "p2"]].mean(axis=1)
        return base[["obs_id", "ti", "tj", "y_true", "y_pred"]]
    return pd.read_parquet(sorted(PROBE.glob(f"heads_{head}/*/predictions.parquet"))[0],
                           columns=["obs_id", "ti", "tj", "y_true", "y_pred"])


def per_image_auc(df: pd.DataFrame) -> pd.Series:
    out = {}
    for obs, g in df.groupby("obs_id"):
        y = g["y_true"].to_numpy()
        out[obs] = roc_auc_score(y, g["y_pred"].to_numpy()) if 0 < y.sum() < y.size else np.nan
    return pd.Series(out)


def main() -> int:
    dossier = pd.read_parquet(DOSSIER)
    vok = set(dossier[dossier.validity_ok].index)
    aucs = {h: per_image_auc(load_preds(h)) for h in HEADS}

    print(f"{'pair':<22s} {'d_med(v)':>9s} {'win':>5s} {'p':>8s}   n(v)")
    results = {}
    for a, b in itertools.combinations(HEADS, 2):
        d = (aucs[b] - aucs[a]).dropna()
        d_v = d[[o in vok for o in d.index]]
        try:
            p = float(stats.wilcoxon(d_v, zero_method="wilcox").pvalue)
        except ValueError:
            p = float("nan")
        results[f"{b}-vs-{a}"] = {"median": float(d_v.median()),
                                  "win": float((d_v > 0).mean()), "p": p,
                                  "n": int(len(d_v))}
        print(f"{b + ' - ' + a:<22s} {d_v.median():>+9.4f} {(d_v > 0).mean():>5.2f} "
              f"{p:>8.4f}   {len(d_v)}")

    print("\nmedian per-image AUC (all images with finite AUC):")
    for h in HEADS:
        print(f"  {h:<10s} {aucs[h].median():.4f}")

    out = PROBE / "head_pairs.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
