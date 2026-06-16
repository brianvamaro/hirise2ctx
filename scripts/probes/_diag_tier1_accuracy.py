"""Tier-1 accuracy + companions (banked LOIO classifier preds, threshold 0.5)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)

REPO = Path(__file__).resolve().parents[2]
t1 = pd.read_parquet(REPO / "models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet")
y = t1.y_true.to_numpy().astype(int)
p = t1.y_pred.to_numpy()
yhat = (p >= 0.5).astype(int)
base = y.mean()

print(f"n={len(y):,}  positive (rich) rate = {base:.1%}")
print(f"  majority-class baseline accuracy (always 'poor') = {1-base:.3f}")
print(f"  accuracy @0.5            = {accuracy_score(y, yhat):.3f}")
print(f"  balanced accuracy @0.5   = {balanced_accuracy_score(y, yhat):.3f}")
print(f"  precision / recall @0.5  = {precision_score(y, yhat):.3f} / {recall_score(y, yhat):.3f}")
print(f"  F1 @0.5                  = {f1_score(y, yhat):.3f}")

# accuracy at the base-rate-matched threshold (predict the top `base` fraction as rich)
thr = np.quantile(p, 1 - base)
yhat2 = (p >= thr).astype(int)
print(f"\n  at base-rate threshold ({thr:.3f}): accuracy {accuracy_score(y, yhat2):.3f}  "
      f"bal-acc {balanced_accuracy_score(y, yhat2):.3f}  F1 {f1_score(y, yhat2):.3f}")
