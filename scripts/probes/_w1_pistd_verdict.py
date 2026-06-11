"""Verdict on the per-image standardization sweep vs the banked baseline.

Promotion criteria (declared in advance): paired Wilcoxon over 38 folds vs
the banked cell (two_stage_balanced x boulder_count @ S=64, sweep
20260611T054855Z) must show median delta(meaningful_auc) > 0 with p < 0.05
AND pooled PR-AUC delta >= -0.01. Mechanistic check: the dossier's
distribution_shift images should be among the gainers.

Usage: python _w1_pistd_verdict.py <pistd_sweep_dir>
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

BASELINE = Path("models/_sweep_w0/20260611T054855Z/summary.parquet")
DOSSIER = Path("dataset_v2/w1_dossier.parquet")
METRICS = ["meaningful_auc", "pr_auc", "spearman_rho", "precision_at_top_5pct"]

sweep_dir = Path(sys.argv[1])
ours = pd.read_parquet(sweep_dir / "summary.parquet")

base = pd.read_parquet(BASELINE)
base = base[(base.variant == "lightgbm_two_stage_balanced") & (base.target_col == "boulder_count")]
base = base.set_index("held_out_obs_id")

dossier = pd.read_parquet(DOSSIER)
shift_imgs = list(dossier[dossier.attributed_cause == "distribution_shift"].index)

for method, g in ours.groupby("pistd"):
    g = g.set_index("held_out_obs_id")
    common = g.index.intersection(base.index)
    print(f"\n== pistd={method} vs raw baseline (n={len(common)} paired folds) ==")
    for metric in METRICS:
        d = (g.loc[common, metric] - base.loc[common, metric]).dropna()
        try:
            p = stats.wilcoxon(d, zero_method="wilcox").pvalue
        except ValueError:
            p = float("nan")
        print(f"  {metric:<24s} mean={d.mean():+.4f} median={d.median():+.4f} "
              f"win={(d > 0).mean():.2f} wilcoxon_p={p:.4f}")
    d_auc = g.loc[common, "meaningful_auc"] - base.loc[common, "meaningful_auc"]
    print(f"  per-image median AUC: {base.loc[common, 'meaningful_auc'].median():.3f} -> "
          f"{g.loc[common, 'meaningful_auc'].median():.3f}")
    print("  distribution_shift images (mechanistic check):")
    for obs in shift_imgs:
        if obs in d_auc.index:
            print(f"    {obs}: {base.loc[obs, 'meaningful_auc']:.3f} -> "
                  f"{g.loc[obs, 'meaningful_auc']:.3f} ({d_auc.loc[obs]:+.3f})")
