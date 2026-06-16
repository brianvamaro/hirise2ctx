"""Unit tests for src.calibration (PLAN_Calibration.md).

Synthetic data only. Asserts the invariants the de-compression layer relies on:
the calibrators are monotone (rank/AUC preserving), quantile-matching recovers the
target marginal, ECE is sane, and the LOIO helper never leaks the held-out image.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.calibration import (
    reliability_curve, expected_calibration_error,
    TemperatureScaler, BetaCalibrator, IsotonicCalibrator, quantile_match,
    compression_metrics, loio_calibrate,
)


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(int)   # P(y=1) == p exactly
    assert expected_calibration_error(y, p, n_bins=10) < 0.02


def test_temperature_preserves_ranking_and_auc():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(int)
    cal = TemperatureScaler().fit(p, y).predict(p)
    # strictly monotone in p -> identical ordering -> identical AUC
    assert np.all(np.diff(cal[np.argsort(p)]) >= -1e-12)
    assert roc_auc_score(y, cal) == pytest.approx(roc_auc_score(y, p), abs=1e-9)


def test_temperature_fixes_overconfidence():
    # over-confident probs (pushed toward 0/1) -> ECE should drop after scaling
    rng = np.random.default_rng(2)
    z = rng.normal(0, 1, 20000)
    ptrue = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, 20000) < ptrue).astype(int)
    p_over = 1 / (1 + np.exp(-z * 2.5))            # too sharp
    ece_before = expected_calibration_error(y, p_over)
    cal = TemperatureScaler().fit(p_over, y).predict(p_over)
    assert expected_calibration_error(y, cal) < ece_before


def test_beta_is_strictly_monotone_and_auc_exact():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(11)
    z = rng.normal(0, 1, 20000)
    ptrue = 1 / (1 + np.exp(-z))
    y = (rng.uniform(0, 1, 20000) < ptrue).astype(int)
    p_over = 1 / (1 + np.exp(-z * 2.5))               # over-confident at both ends
    beta = BetaCalibrator().fit(p_over, y)
    grid = np.linspace(0.01, 0.99, 500)
    out = beta.predict(grid)
    assert np.all(np.diff(out) > 0)                    # strictly increasing -> no ties
    # strictly monotone -> ranking (AUC) preserved exactly
    assert roc_auc_score(y, beta.predict(p_over)) == pytest.approx(roc_auc_score(y, p_over), abs=1e-9)
    # and it improves calibration
    assert expected_calibration_error(y, beta.predict(p_over)) < expected_calibration_error(y, p_over)


def test_isotonic_is_monotone():
    rng = np.random.default_rng(3)
    pred = rng.uniform(0, 1, 2000)
    true = pred ** 2 + rng.normal(0, 0.05, 2000)
    cal = IsotonicCalibrator().fit(pred, true)
    grid = np.linspace(0, 1, 200)
    out = cal.predict(grid)
    assert np.all(np.diff(out) >= -1e-9)


def test_quantile_match_recovers_target_marginal():
    rng = np.random.default_rng(4)
    ref_pred = rng.uniform(0.2, 0.5, 5000)          # compressed predictions
    ref_true = rng.exponential(0.05, 5000)          # skewed truth with a tail
    out = quantile_match(ref_pred, ref_pred, ref_true)
    # calibrated marginal quantiles should match the truth's
    for q in (0.1, 0.5, 0.9, 0.99):
        assert np.quantile(out, q) == pytest.approx(np.quantile(ref_true, q), rel=0.1, abs=0.01)
    # monotone -> ranking preserved
    assert spearmanr(ref_pred, out).correlation == pytest.approx(1.0, abs=1e-6)


def test_quantile_match_decompresses_top_and_bottom():
    rng = np.random.default_rng(5)
    true = np.concatenate([np.zeros(1800), rng.exponential(0.05, 8200)])
    # a compressed predictor: correct ranking, squashed range
    order = np.argsort(true) + rng.normal(0, 200, len(true))
    pred = 0.01 + 0.02 * (order - order.min()) / (order.max() - order.min())
    base = compression_metrics(true, pred)
    cal = quantile_match(pred, pred, true)
    m = compression_metrics(true, cal)
    assert m["marginal_l1"] < base["marginal_l1"]          # marginal matched
    assert m["near_zero_pred"] > base["near_zero_pred"]     # lows reach zero
    assert abs(m["top_ratio"] - 1) < abs(base["top_ratio"] - 1)  # tail recovered


def test_loio_calibrate_no_leak():
    # held-out image's own rows must not influence its calibration
    rng = np.random.default_rng(6)
    rows = []
    for g in range(4):
        p = rng.uniform(0, 1, 50)
        rows.append(pd.DataFrame({"obs_id": f"img{g}", "y_pred": p, "y_true": (p > 0.5).astype(float)}))
    df = pd.concat(rows, ignore_index=True)
    seen = {}

    def fit_apply(rp, rt, hp):
        seen["ref_n"] = len(rp)
        return hp  # identity

    out = loio_calibrate(df, fit_apply)
    assert out.shape == (len(df),)
    assert seen["ref_n"] == len(df) - 50          # fit saw exactly the other 3 images
    assert not np.isnan(out).any()


def test_compression_metrics_keys():
    yt = np.array([0.0, 0.0, 0.02, 0.08])
    yp = np.array([0.005, 0.006, 0.02, 0.04])
    m = compression_metrics(yt, yp)
    assert set(m) == {"spearman", "top_ratio", "low_over", "near_zero_pred",
                      "near_zero_true", "marginal_l1"}
    assert m["near_zero_true"] == 0.5
