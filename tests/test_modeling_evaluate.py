"""Unit tests for src.modeling.evaluate metric helpers + aggregation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modeling.evaluate import (
    ABUNDANCE_BIN_LABELS,
    EMPTY_TRUTH_OBS_ID,
    aggregate_fold_metrics,
    per_bin_rmse,
    per_fold_metrics,
    presence_auc,
    rmse,
    rmse_log1p,
    spearman_safe,
)


# ============================================================================
# Spearman / RMSE helpers
# ============================================================================


def test_spearman_safe_returns_nan_for_constant_input():
    assert np.isnan(spearman_safe(np.zeros(10), np.arange(10).astype(float)))
    assert np.isnan(spearman_safe(np.arange(10).astype(float), np.zeros(10)))


def test_spearman_safe_returns_one_for_monotonic_input():
    y_true = np.arange(20).astype(float)
    y_pred = y_true * 3.0 + 5.0
    assert spearman_safe(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_zero_on_perfect_predictions():
    y = np.array([0.1, 0.2, 0.3])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_log1p_clips_negatives():
    y_true = np.array([0.0, 0.001])
    y_pred = np.array([-0.5, 0.001])  # negative pred gets clipped to 0
    val = rmse_log1p(y_true, y_pred)
    assert val == pytest.approx(0.0, abs=1e-9)


def test_presence_auc_perfect_separation():
    y_true_pos = np.array([False, False, True, True])
    y_pred = np.array([0.1, 0.2, 0.8, 0.9])
    assert presence_auc(y_true_pos, y_pred) == pytest.approx(1.0)


def test_presence_auc_nan_for_one_class_missing():
    y_true_pos = np.array([True, True])
    assert np.isnan(presence_auc(y_true_pos, np.array([0.5, 0.6])))


# ============================================================================
# Per-abundance-bin RMSE
# ============================================================================


def test_per_bin_rmse_shape_and_zero_bin():
    y_true = np.array([0.0, 0.0, 0.0, 5e-5, 5e-4, 5e-3, 5e-2])
    y_pred = np.array([0.01, 0.02, 0.0, 0.0, 0.001, 0.01, 0.1])
    df = per_bin_rmse(y_true, y_pred)
    assert list(df["bin"]) == list(ABUNDANCE_BIN_LABELS)
    # zero bin: 3 tiles with y_true == 0
    zero_row = df[df["bin"] == "zero"].iloc[0]
    assert zero_row["n_tiles"] == 3
    assert zero_row["rmse"] == pytest.approx(np.sqrt((0.01**2 + 0.02**2 + 0.0**2) / 3))


def test_per_bin_rmse_handles_empty_bins():
    y_true = np.array([0.0, 0.0])   # only zero bin populated
    y_pred = np.array([0.0, 0.0])
    df = per_bin_rmse(y_true, y_pred)
    nonzero_bins = df[df["bin"] != "zero"]
    assert (nonzero_bins["n_tiles"] == 0).all()
    assert nonzero_bins["rmse"].isna().all()


# ============================================================================
# per_fold_metrics + specificity flag
# ============================================================================


def test_per_fold_metrics_flags_empty_truth_image():
    y_true = np.zeros(50)
    y_pred = np.full(50, 1e-5)
    m = per_fold_metrics(y_true, y_pred, held_out_obs_ids=[EMPTY_TRUTH_OBS_ID])
    assert m["is_specificity_only"] is True
    assert np.isnan(m["spearman_rho"])
    assert np.isnan(m["presence_auc"])
    # The pred_above_1e-4 stat must be computed on specificity folds
    assert "pred_above_1e-4" in m


def test_per_fold_metrics_real_fold_returns_real_spearman():
    rng = np.random.default_rng(0)
    n = 1000
    y_true = rng.exponential(1e-4, n)
    y_pred = y_true + rng.normal(0, 1e-5, n)
    m = per_fold_metrics(y_true, y_pred, held_out_obs_ids=["OBS_X"])
    assert m["is_specificity_only"] is False
    assert m["spearman_rho"] > 0.8  # strong signal
    assert 0.0 < m["presence_auc"] <= 1.0 or np.isnan(m["presence_auc"])


# ============================================================================
# Aggregation
# ============================================================================


def test_aggregate_fold_metrics_separates_specificity_folds():
    per_fold = [
        {"spearman_rho": 0.5, "rmse_log1p": 0.01, "rmse_raw": 0.02, "presence_auc": 0.8,
         "is_specificity_only": False, "held_out_obs_ids": ["A"]},
        {"spearman_rho": 0.3, "rmse_log1p": 0.02, "rmse_raw": 0.03, "presence_auc": 0.7,
         "is_specificity_only": False, "held_out_obs_ids": ["B"]},
        {"spearman_rho": float("nan"), "rmse_log1p": 0.001, "rmse_raw": 0.001,
         "presence_auc": float("nan"), "is_specificity_only": True,
         "held_out_obs_ids": [EMPTY_TRUTH_OBS_ID]},
    ]
    agg = aggregate_fold_metrics(per_fold)
    assert agg["n_real_folds"] == 2
    assert agg["n_specificity_folds"] == 1
    assert agg["spearman_rho_mean"] == pytest.approx(0.4)
    # rmse_raw is computed across ALL folds (including specificity)
    assert agg["rmse_raw_mean"] == pytest.approx((0.02 + 0.03 + 0.001) / 3)
