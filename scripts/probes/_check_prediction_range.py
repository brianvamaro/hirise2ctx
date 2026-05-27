"""Confirm 'model predicts near-constant near-zero' diagnosis.

For each (variant, scale), look at the distribution of model predictions across
all 9 folds: range, quantiles, fraction of predictions above each abundance bin
threshold.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]

EDGES = [1e-4, 1e-3, 1e-2, 1e-1]
EDGE_LABELS = [">1e-4", ">1e-3", ">1e-2", ">1e-1"]

print("variant              scale  n_tiles  median(yp)  max(yp)    max(yt)    " +
      "  ".join(f"yp_{l}" for l in EDGE_LABELS) + "   |   " +
      "  ".join(f"yt_{l}" for l in EDGE_LABELS))
print("-" * 165)

for variant in ("lightgbm_tweedie", "lightgbm_log1p_huber", "lightgbm_two_stage"):
    for s in (8, 16, 32, 64):
        scale_dirs = sorted((REPO / "models" / variant).glob(f"*/scale_S{s}/predictions.parquet"),
                            key=lambda p: p.stat().st_mtime)
        if not scale_dirs:
            continue
        pred = pd.read_parquet(scale_dirs[-1])
        yp = pred["y_pred"].to_numpy()
        yt = pred["y_true"].to_numpy()
        yp_above = [f"{(yp > e).mean()*100:>6.3f}%" for e in EDGES]
        yt_above = [f"{(yt > e).mean()*100:>6.3f}%" for e in EDGES]
        print(f"{variant:<20s} S={s:<3d} {len(yp):>7d}  {np.median(yp):.3e}   {yp.max():.3e}  {yt.max():.3e}    " +
              "   ".join(yp_above) + "   |   " +
              "   ".join(yt_above))
