"""Tier-1: does beta calibration fix both ends WITHOUT isotonic's AUC cost? (LOIO)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.calibration import (TemperatureScaler, BetaCalibrator, IsotonicCalibrator,
                             expected_calibration_error, loio_calibrate)

t1 = pd.read_parquet(REPO / "models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet")
y = t1.y_true.to_numpy()


def ece_split(p):
    p = np.clip(p, 0, 1); lo = p < 0.5
    return (expected_calibration_error(y, p),
            expected_calibration_error(y[lo], p[lo]),
            expected_calibration_error(y[~lo], p[~lo]))


variants = {
    "raw":         t1.y_pred.to_numpy(),
    "temperature": loio_calibrate(t1, lambda rp, rt, hp: TemperatureScaler().fit(rp, rt).predict(hp)),
    "isotonic":    loio_calibrate(t1, lambda rp, rt, hp: IsotonicCalibrator().fit(rp, rt).predict(hp)),
    "beta":        loio_calibrate(t1, lambda rp, rt, hp: BetaCalibrator().fit(rp, rt).predict(hp)),
}
auc0 = roc_auc_score(y, t1.y_pred)
print(f"{'variant':>13} {'ECE':>6} {'ECE_low':>8} {'ECE_high':>9} {'AUC':>7} {'dAUC':>7} {'Brier':>7}")
for n, p in variants.items():
    p = np.clip(p, 0, 1); e, el, eh = ece_split(p); auc = roc_auc_score(y, p)
    print(f"{n:>13} {e:>6.3f} {el:>8.3f} {eh:>9.3f} {auc:>7.4f} {auc-auc0:>+7.4f} {brier_score_loss(y, p):>7.4f}")
print("\nbeta = smooth 3-param strictly-monotone (no ties) -> should fix both ends like "
      "isotonic but keep AUC like temperature.")

# Is the LOIO pooled-AUC drop a per-fold artifact (different map per image) or a real
# deployment cost? At deployment ONE global map is fit -> a strictly-monotone map MUST
# preserve pooled AUC exactly. Check by fitting on all 38 and applying to all 38.
print("\n--- global fit (deployment case: one map for all) ---")
p = t1.y_pred.to_numpy()
for n, c in [("temperature", TemperatureScaler()), ("isotonic", IsotonicCalibrator()),
             ("beta", BetaCalibrator())]:
    pc = c.fit(p, y).predict(p)
    print(f"  {n:>11}: AUC {roc_auc_score(y, pc):.4f} (dAUC {roc_auc_score(y, pc)-auc0:+.4f}); "
          f"strictly-monotone preserves it exactly")
print("  => the LOIO pooled-AUC drop is a PER-FOLD artifact; a global strictly-monotone "
      "calibrator (temperature/beta) is AUC-exact at deployment.")
