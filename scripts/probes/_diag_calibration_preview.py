"""Preview the post-hoc calibrators on the banked Tier-1 + Tier-2 predictions (LOIO).

Confirms (for PLAN_Calibration.md): temperature scaling fixes Tier-1 ECE without
touching AUC; isotonic / quantile-matching de-compress Tier-2 without touching
Spearman. All LOIO-honest (fit on the other 37 images, apply to the held-out).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.calibration import (  # noqa: E402
    TemperatureScaler, IsotonicCalibrator, quantile_match,
    expected_calibration_error, compression_metrics, loio_calibrate,
)

T1 = REPO / "models" / "fang_probe" / "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2" / "predictions.parquet"
T2 = (REPO / "models" / "fang_tier2" / "tier2_mlp_reg_emb_fractional_area_S32"
      / "1e01ad8b17447599" / "predictions.parquet")


def line(d, keys):
    return "  ".join(f"{k}={d[k]:.3f}" if abs(d[k]) < 100 else f"{k}={d[k]:.0f}" for k in keys)


print("=== TIER-1 probability calibration (temperature scaling, LOIO) ===")
t1 = pd.read_parquet(T1)
auc0 = roc_auc_score(t1["y_true"], t1["y_pred"])
ece0 = expected_calibration_error(t1["y_true"].to_numpy(), t1["y_pred"].to_numpy())
cal = loio_calibrate(t1, lambda rp, rt, hp: TemperatureScaler().fit(rp, rt).predict(hp))
auc1 = roc_auc_score(t1["y_true"], cal)
ece1 = expected_calibration_error(t1["y_true"].to_numpy(), cal)
# fit one T on all (just to report the direction)
Tall = TemperatureScaler().fit(t1["y_pred"].to_numpy(), t1["y_true"].to_numpy()).T
print(f"  before: ECE={ece0:.3f}  AUC={auc0:.4f}  pred mean={t1['y_pred'].mean():.3f} "
      f"std={t1['y_pred'].std():.3f}")
print(f"  after : ECE={ece1:.3f}  AUC={auc1:.4f}  pred mean={cal.mean():.3f} "
      f"std={cal.std():.3f}   (global T={Tall:.2f})")
print(f"  -> ECE {ece0:.3f}->{ece1:.3f}, AUC change {auc1-auc0:+.4f} (ranking preserved)")

print("\n=== TIER-2 abundance de-compression (LOIO) ===")
t2 = pd.read_parquet(T2)
base = compression_metrics(t2["y_true"].to_numpy(), t2["y_pred"].to_numpy())
iso = loio_calibrate(t2, lambda rp, rt, hp: IsotonicCalibrator().fit(rp, rt).predict(hp))
qm = loio_calibrate(t2, lambda rp, rt, hp: quantile_match(hp, rp, rt))
m_iso = compression_metrics(t2["y_true"].to_numpy(), iso)
m_qm = compression_metrics(t2["y_true"].to_numpy(), qm)
keys = ["spearman", "top_ratio", "low_over", "near_zero_pred", "marginal_l1"]
print(f"  truth near-zero share = {base['near_zero_true']:.1%}")
print(f"  raw mlp_reg     : {line(base, keys)}")
print(f"  + isotonic      : {line(m_iso, keys)}")
print(f"  + quantile-match: {line(m_qm, keys)}")
print("\n  (goal: top_ratio->1, low_over down, near_zero_pred->~0.18, marginal_l1->0, "
      "spearman unchanged)")
