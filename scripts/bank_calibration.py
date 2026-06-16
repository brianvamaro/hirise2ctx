"""Bank the Stage-1 CalibrationLayer (PLAN_Calibration §5 Stage 1).

Fits the deployment calibration on the POOLED LOIO predictions of all 38 images
(deployment-honest: out-of-fold preds, no further holdout) and saves it next to the
DeployableHead. **One-model default:** Tier-1 isotonic on `P(rich)` + Tier-2 *global*
quantile-match of the SAME `P(rich)` onto the `fractional_area` marginal (no separate
Tier-2 head). The in-cohort metrics printed (ECE, top_ratio) are the conservative bound
the deployed layer inherits — off-HiRISE terrain has no truth.

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/bank_calibration.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.calibration import (CalibrationLayer, IsotonicCalibrator, quantile_match,
                             loio_calibrate, expected_calibration_error, compression_metrics)

T1_PREDS = REPO / "models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet"
OUT = REPO / "models/deployable/calibration.npz"


def main() -> int:
    t1 = pd.read_parquet(T1_PREDS).rename(columns={"y_true": "y_binary", "y_pred": "p_rich"})
    parts = []
    for p in (REPO / "dataset_v2/labels").glob("*.parquet"):
        d = pd.read_parquet(p)
        parts.append(d[d.tile_size_px == 32][["obs_id", "ti", "tj", "fractional_area"]])
    lab = pd.concat(parts, ignore_index=True)
    df = t1.merge(lab, on=["obs_id", "ti", "tj"], how="inner")
    print(f"fit on {len(df)} tiles / {df.obs_id.nunique()} images (pooled LOIO, one-model)", flush=True)

    layer = CalibrationLayer.from_loio_predictions(
        df, meta={"recipe": "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2", "scale": "S32",
                  "mode": "one_model", "fit": "pooled_loio_38"})
    layer.save(OUT)

    yb = df.y_binary.to_numpy(); pr = df.p_rich.to_numpy(); fa = df.fractional_area.to_numpy()

    # In-sample (the banked global map vs its own fit data) — a SANITY check that the
    # fit reproduces the training marginal, NOT a deployment estimate (isotonic fits its
    # training ECE to ~0 by construction; qmatch matches its training marginal exactly).
    ece_in = expected_calibration_error(yb, layer.calibrate_prob(pr))
    m_in = compression_metrics(fa, layer.calibrate_abundance(pr))
    print(f"  [in-sample sanity] Tier-1 ECE {expected_calibration_error(yb, pr):.3f} -> {ece_in:.3f}; "
          f"Tier-2 top_ratio {m_in['top_ratio']:.2f}, marginal_L1 {m_in['marginal_l1']:.4f}", flush=True)

    # LOIO deployment bound (honest): fit the map on the other 37 images, apply to the
    # held-out one — what off-cohort-like terrain inherits. This is the number to trust.
    iso_loio = loio_calibrate(df.rename(columns={"p_rich": "y_pred", "y_binary": "y_true"}),
                              lambda rp, rt, hp: IsotonicCalibrator().fit(rp, rt).predict(hp))
    ab_loio = loio_calibrate(df.rename(columns={"p_rich": "y_pred", "fractional_area": "y_true"}),
                             lambda rp, rt, hp: quantile_match(hp, rp, rt))
    ece_loio = expected_calibration_error(yb, iso_loio)
    m_loio = compression_metrics(fa, ab_loio)
    print(f"  [LOIO bound] Tier-1 ECE {ece_loio:.3f}  (gate <=0.05: "
          f"{'PASS' if ece_loio <= 0.05 else 'FAIL'})", flush=True)
    print(f"  [LOIO bound] Tier-2 top_ratio {m_loio['top_ratio']:.2f}  near0 {m_loio['near_zero_pred']:.1%} "
          f"(true {m_loio['near_zero_true']:.1%})  marginal_L1 {m_loio['marginal_l1']:.4f}  "
          f"spearman {m_loio['spearman']:.3f}  (gate top in [0.8,1.2]: "
          f"{'PASS' if 0.8 <= m_loio['top_ratio'] <= 1.2 else 'FAIL'})", flush=True)

    back = CalibrationLayer.load(OUT)
    d = max(float(np.abs(back.calibrate_prob(pr[:4096]) - layer.calibrate_prob(pr[:4096])).max()),
            float(np.abs(back.calibrate_abundance(pr[:4096]) - layer.calibrate_abundance(pr[:4096])).max()))
    print(f"  save/load round-trip max |d| = {d:.2e} ({'OK' if d < 1e-9 else 'MISMATCH'}) "
          f"-> {OUT.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
