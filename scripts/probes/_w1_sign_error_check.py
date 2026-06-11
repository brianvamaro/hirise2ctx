"""W1 Rung 1b clincher — does the rescore optimum track the predicted sign-error residual?

Hypothesis (code reading, coregister.py:383): dy_m lacks the row->world-y sign
flip, so the applied y-shift has inverted sign and the post-"correction" label
displacement is 2*dy_m south. Prediction: per-image optimal label-read offset
di* ~= 2*|dy_m| / 320 m (and ~0 for images with tiny |dy_m|), while dj* ~= 0
(dx is converted correctly).

Tests:
1. Spearman(best_di, predicted_di) and exact/within-1 match counts.
2. Same for a continuous di estimate: AUC-argmax along dj in {-1,0,1} averaged.
3. Control: same regression for dj vs predicted_dj = 0 (should be null).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

TILE_M = 320.0
grid = pd.read_parquet("scripts/probes/_w1_shift_rescore.parquet")

rows = []
for f in sorted(Path("cache_v2/coregistration").glob("*.json")):
    d = json.loads(f.read_text())
    rows.append(dict(obs_id=d["obs_id"], dy_m=d["shift_m"]["dy"], dx_m=d["shift_m"]["dx"]))
coreg = pd.DataFrame(rows).set_index("obs_id")

best = grid.loc[grid.groupby("obs_id")["auc"].idxmax()].set_index("obs_id")
tab = coreg.join(best[["di", "dj"]].rename(columns={"di": "best_di", "dj": "best_dj"}), how="inner")

# predicted optimum under the sign-error hypothesis (residual = 2*dy_m, south)
tab["pred_di_cont"] = 2.0 * (-tab.dy_m) / TILE_M   # dy_m < 0 -> positive di
tab["pred_di"] = tab.pred_di_cont.round().clip(-2, 2).astype(int)

# smoother empirical di*: AUC-weighted argmax along di, averaging dj in {-1,0,1}
sub = grid[grid.dj.abs() <= 1]
mean_by_di = sub.groupby(["obs_id", "di"])["auc"].mean().reset_index()
emp = mean_by_di.loc[mean_by_di.groupby("obs_id")["auc"].idxmax()].set_index("obs_id")["di"]
tab["emp_di"] = emp

for est in ["best_di", "emp_di"]:
    rho, p = spearmanr(tab[est], tab.pred_di_cont)
    exact = (tab[est] == tab.pred_di).mean()
    within1 = ((tab[est] - tab.pred_di).abs() <= 1).mean()
    print(f"{est}: Spearman vs pred {rho:+.3f} (p={p:.5f}); exact match {exact:.0%}; within +-1 {within1:.0%}")

# control: dj should center on 0 regardless of dx_m (dx applied correctly)
rho_j, p_j = spearmanr(tab.best_dj, 2.0 * (-tab.dx_m) / TILE_M)
print(f"control best_dj vs sign-error-predicted dj: rho {rho_j:+.3f} (p={p_j:.4f})")
print(f"mean best_dj {tab.best_dj.mean():+.2f} vs mean best_di {tab.best_di.mean():+.2f}")

print(tab.sort_values("pred_di_cont")[["dy_m", "pred_di_cont", "pred_di", "best_di", "emp_di", "best_dj"]]
      .to_string(float_format=lambda v: f"{v:.2f}"))
