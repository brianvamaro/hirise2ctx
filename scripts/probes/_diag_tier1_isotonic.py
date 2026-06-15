"""Tier-1: does a flexible monotone calibrator beat single-parameter temperature?

Temperature is one global squeeze -> can't fix the over-confident high end AND the
low end independently. Isotonic is a free monotone map -> can. Compare LOIO, split
ECE into low/high halves to see the two-ended effect Brian spotted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.calibration import (TemperatureScaler, IsotonicCalibrator,
                             expected_calibration_error, loio_calibrate)

t1 = pd.read_parquet(REPO / "models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet")
y = t1.y_true.to_numpy()


def ece_split(yt, p):
    lo = p < 0.5
    return (expected_calibration_error(yt, p),
            expected_calibration_error(yt[lo], p[lo]) if lo.any() else np.nan,
            expected_calibration_error(yt[~lo], p[~lo]) if (~lo).any() else np.nan)


def iso_prob(rp, rt, hp):
    return IsotonicCalibrator().fit(rp, rt).predict(hp)


variants = {
    "raw":              t1.y_pred.to_numpy(),
    "temperature":      loio_calibrate(t1, lambda rp, rt, hp: TemperatureScaler().fit(rp, rt).predict(hp)),
    "isotonic":         loio_calibrate(t1, iso_prob),
    "temp->isotonic":   None,  # filled below
}
# temp then isotonic, both fit LOIO on the same ref split
tmp = pd.DataFrame({"obs_id": t1.obs_id, "y_true": t1.y_true,
                    "y_pred": variants["temperature"]})
variants["temp->isotonic"] = loio_calibrate(tmp, iso_prob)

print(f"{'variant':>15} {'ECE':>6} {'ECE_low':>8} {'ECE_high':>9} {'AUC':>7} {'Brier':>7}")
for name, p in variants.items():
    p = np.clip(p, 0, 1)
    e, el, eh = ece_split(y, p)
    print(f"{name:>15} {e:>6.3f} {el:>8.3f} {eh:>9.3f} "
          f"{roc_auc_score(y, p):>7.4f} {brier_score_loss(y, p):>7.4f}")
print("\nECE_low = calibration error on predictions <0.5; ECE_high on >=0.5")
