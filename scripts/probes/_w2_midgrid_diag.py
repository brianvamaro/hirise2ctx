"""Mid-grid diagnostic on the W2 CNN cells finished so far (PLAN_CNN.md §4).

Reads the incremental summary.parquet, joins the W1 dossier + both tabular
baselines, and answers four structural questions while cell D runs:
  1. Where does the CNN win/lose vs the GBM per image -- same failure images
     or complementary ones?
  2. Is the pooled-PR-AUC deficit explained by per-image score shift
     (mean_pred_prob vs base_rate misalignment across images)?
  3. What did geometric augmentation break (B vs A per image)?
  4. Early mechanism read: distribution_shift / texture_decorrelated classes
     under each cell.

Usage: python _w2_midgrid_diag.py models/_sweep_cnn/<TS>
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

sweep_dir = Path(sys.argv[1])
cnn = pd.read_parquet(sweep_dir / "summary.parquet")
dossier = pd.read_parquet(REPO_ROOT / "dataset_v2/w1_dossier.parquet")
gbm = pd.read_parquet(REPO_ROOT / "models/_sweep_w0/20260611T054855Z/summary.parquet")
gbm = gbm[(gbm.variant == "lightgbm_two_stage_balanced")
          & (gbm.target_col == "boulder_count")].set_index("held_out_obs_id")
t1 = pd.read_parquet(REPO_ROOT / "models/_sweep_binary/20260611T214042Z/summary.parquet"
                     ).set_index("held_out_obs_id")

cells = list(dict.fromkeys(cnn["aug_cell"]))
print(f"cells on disk: {cells}\n")
A = cnn[cnn.aug_cell == "none"].set_index("held_out_obs_id")

# --- 1. complementarity: CNN-A vs baselines, per image ---------------------
common = A.index.intersection(gbm.index)
ok = A.loc[common, "auc"].notna()
r_gbm = stats.spearmanr(A.loc[common, "auc"][ok], gbm.loc[common, "meaningful_auc"][ok])
r_t1 = stats.spearmanr(A.loc[common, "auc"][ok], t1.loc[common, "auc"][ok])
print("1) Per-image AUC correlation, CNN cell A vs baselines:")
print(f"   vs GBM banked:  rho={r_gbm.correlation:+.3f} (p={r_gbm.pvalue:.3f})")
print(f"   vs Tier 1:      rho={r_t1.correlation:+.3f} (p={r_t1.pvalue:.3f})")
d = (A.loc[common, "auc"] - t1.loc[common, "auc"]).dropna().sort_values()
print(f"   CNN-A minus Tier1 per image: median {d.median():+.3f}; "
      f"biggest losses: {[(o, round(v, 3)) for o, v in d.head(3).items()]}")
print(f"   biggest wins: {[(o, round(v, 3)) for o, v in d.tail(3).items()]}\n")

# --- 2. score-shift structure ----------------------------------------------
print("2) Per-image score shift (mean_pred_prob - base_rate), by cell:")
for cell in cells:
    g = cnn[cnn.aug_cell == cell]
    shift = g["mean_pred_prob"] - g["base_rate"]
    r = stats.spearmanr(g["mean_pred_prob"], g["base_rate"])
    print(f"   {cell:<16s} |shift| median={shift.abs().median():.3f} max={shift.abs().max():.3f}  "
          f"rank-corr(pred, base_rate)={r.correlation:+.3f}")
shift_t1 = (t1["mean_pred_prob"] - t1["base_rate"])
r = stats.spearmanr(t1["mean_pred_prob"], t1["base_rate"])
print(f"   {'Tier1 (ref)':<16s} |shift| median={shift_t1.abs().median():.3f} "
      f"max={shift_t1.abs().max():.3f}  rank-corr={r.correlation:+.3f}\n")

# --- 3. what geometric aug broke -------------------------------------------
if "geometric" in cells:
    B = cnn[cnn.aug_cell == "geometric"].set_index("held_out_obs_id")
    common_ab = A.index.intersection(B.index)
    d_ba = (B.loc[common_ab, "auc"] - A.loc[common_ab, "auc"]).dropna().sort_values()
    print("3) Cell B minus A per image:")
    print(f"   median {d_ba.median():+.3f}, win rate {(d_ba > 0).mean():.2f}, "
          f"degraded on {(d_ba < -0.02).sum()}/{len(d_ba)} images")
    print(f"   worst: {[(o, round(v, 3)) for o, v in d_ba.head(4).items()]}\n")

# --- 4. dossier classes under each cell -------------------------------------
print("4) Dossier classes (AUC per cell; baseline GBM in last col):")
for cls in ("distribution_shift", "texture_decorrelated"):
    imgs = sorted(dossier[dossier.attributed_cause == cls].index)
    print(f"   {cls}:")
    for obs in imgs:
        vals = []
        for cell in cells:
            g = cnn[cnn.aug_cell == cell].set_index("held_out_obs_id")
            vals.append(f"{cell[:5]}={g.loc[obs, 'auc']:.3f}" if obs in g.index else f"{cell[:5]}=--")
        base = f"gbm={gbm.loc[obs, 'meaningful_auc']:.3f}" if obs in gbm.index else ""
        print(f"     {obs}: {'  '.join(vals)}  {base}")
