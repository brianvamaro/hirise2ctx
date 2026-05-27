"""Unit tests for the Stage 5b classification additions in src.modeling.evaluate."""
from __future__ import annotations

import numpy as np
import pytest

from src.modeling.evaluate import (
    EMPTY_TRUTH_OBS_ID,
    aggregate_fold_metrics_classification,
    brier_score,
    calibration_deciles,
    expected_calibration_error,
    lift_at_top_k,
    per_fold_metrics_classification,
    run_loio,
)


# ============================================================================
# Scalar metric helpers
# ============================================================================


def test_brier_score_zero_on_perfect_predictions():
    y_true = np.array([0, 1, 0, 1], dtype=np.int8)
    y_pred = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier_score(y_true, y_pred) == pytest.approx(0.0)


def test_brier_score_quarter_on_constant_half():
    """Predicting 0.5 on a balanced binary set -> MSE = 0.25."""
    y_true = np.array([0, 1, 0, 1], dtype=np.int8)
    y_pred = np.full(4, 0.5)
    assert brier_score(y_true, y_pred) == pytest.approx(0.25)


def test_brier_score_nan_on_empty():
    assert np.isnan(brier_score(np.array([], dtype=np.int8), np.array([])))


def test_lift_at_top_k_perfect_classifier_equals_inverse_base_rate():
    """A perfectly ranked classifier puts every positive in the top-k -> lift = 1/base_rate."""
    n = 1000
    n_pos = 50
    y_true = np.zeros(n, dtype=np.int8)
    y_true[:n_pos] = 1
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    y_true = y_true[perm]
    # Perfect predictor: probability == truth (with tiny noise so argpartition is unique)
    y_pred = y_true.astype(np.float64) + 1e-9 * rng.standard_normal(n)
    base_rate = n_pos / n
    assert lift_at_top_k(y_true, y_pred) == pytest.approx(1.0 / base_rate, rel=1e-6)


def test_lift_at_top_k_random_classifier_near_one():
    """A truly random ranking gives precision@k ~ base_rate, i.e. lift ~ 1."""
    rng = np.random.default_rng(0)
    n = 5000
    n_pos = 250
    y_true = np.zeros(n, dtype=np.int8)
    y_true[rng.choice(n, n_pos, replace=False)] = 1
    y_pred = rng.uniform(size=n)
    # Expected lift = 1.0; allow 0.7-1.5 for finite-sample variation.
    lift = lift_at_top_k(y_true, y_pred)
    assert 0.7 <= lift <= 1.5


def test_lift_at_top_k_nan_when_no_positives_or_all_positives():
    n = 100
    y_pred = np.random.default_rng(0).uniform(size=n)
    assert np.isnan(lift_at_top_k(np.zeros(n, dtype=np.int8), y_pred))
    assert np.isnan(lift_at_top_k(np.ones(n, dtype=np.int8), y_pred))


def test_expected_calibration_error_zero_on_calibrated_predictions():
    """If predicted prob == empirical positive rate within every bin, ECE == 0."""
    # Build a dataset where 30% of tiles get pred 0.3 and 30% are actually positive,
    # and 60% get pred 0.9 and 90% are actually positive within that bin.
    rng = np.random.default_rng(0)
    y_pred = np.concatenate([np.full(300, 0.3), np.full(700, 0.9)])
    # Within first 300: 30% positive
    y_true_low = np.zeros(300, dtype=np.int8); y_true_low[:90] = 1
    rng.shuffle(y_true_low)
    # Within last 700: 90% positive
    y_true_high = np.zeros(700, dtype=np.int8); y_true_high[:630] = 1
    rng.shuffle(y_true_high)
    y_true = np.concatenate([y_true_low, y_true_high])
    ece = expected_calibration_error(y_true, y_pred, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_large_on_anti_calibrated_predictions():
    """A model that predicts 0.95 for all negatives + 0.05 for all positives is maximally
    miscalibrated; ECE should be ~0.9."""
    y_true = np.array([0] * 100 + [1] * 100, dtype=np.int8)
    y_pred = np.array([0.95] * 100 + [0.05] * 100)
    ece = expected_calibration_error(y_true, y_pred, n_bins=10)
    assert ece > 0.8


def test_calibration_deciles_returns_n_bins_rows():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    y_pred = np.linspace(0.05, 0.95, 10)
    rows = calibration_deciles(y_true, y_pred, n_bins=5)
    assert len(rows) == 5
    # Bins are evenly spaced in [0, 1]
    assert rows[0]["lo"] == 0.0 and rows[0]["hi"] == pytest.approx(0.2)
    assert rows[-1]["lo"] == pytest.approx(0.8) and rows[-1]["hi"] == 1.0


# ============================================================================
# Per-fold + aggregate
# ============================================================================


def test_per_fold_metrics_classification_returns_full_metric_set_on_real_fold():
    y_true = np.array([0] * 80 + [1] * 20, dtype=np.int8)
    rng = np.random.default_rng(0)
    # Predictions roughly tracking truth + noise
    y_pred = np.clip(y_true * 0.4 + 0.1 + 0.1 * rng.standard_normal(100), 0, 1)
    m = per_fold_metrics_classification(y_true, y_pred, held_out_obs_ids=["ESP_039820_1750"])
    assert m["n_tiles"] == 100
    assert m["n_positive"] == 20
    assert m["n_negative"] == 80
    assert m["base_rate"] == pytest.approx(0.2)
    assert m["is_specificity_only"] is False
    assert not np.isnan(m["auc"])
    assert not np.isnan(m["brier"])
    assert not np.isnan(m["ece"])
    assert not np.isnan(m["lift_at_top_k"])
    assert len(m["calibration_deciles"]) == 10


def test_per_fold_metrics_classification_flags_specificity_on_empty_truth_fold():
    """The empty-truth ObsId is automatically tagged is_specificity_only with no AUC/lift."""
    y_true = np.zeros(50, dtype=np.int8)
    y_pred = np.random.default_rng(0).uniform(size=50) * 0.1
    m = per_fold_metrics_classification(y_true, y_pred, held_out_obs_ids=[EMPTY_TRUTH_OBS_ID])
    assert m["is_specificity_only"] is True
    assert np.isnan(m["auc"])
    assert np.isnan(m["lift_at_top_k"])
    # Brier is still defined (MSE on all-zero truth)
    assert not np.isnan(m["brier"])
    # FP rate at default threshold = fraction with y_pred >= 0.5
    assert "false_positive_rate_at_threshold" in m


def test_per_fold_metrics_classification_flags_specificity_on_single_class_fold():
    """A non-empty-truth fold that happens to be single-class is still specificity-only."""
    y_true = np.zeros(50, dtype=np.int8)
    y_pred = np.random.default_rng(0).uniform(size=50)
    m = per_fold_metrics_classification(y_true, y_pred, held_out_obs_ids=["ESP_039820_1750"])
    assert m["is_specificity_only"] is True


def test_aggregate_fold_metrics_classification_separates_specificity_folds():
    real_folds = [
        {"is_specificity_only": False, "auc": 0.7, "brier": 0.1, "ece": 0.05, "lift_at_top_k": 2.0},
        {"is_specificity_only": False, "auc": 0.6, "brier": 0.15, "ece": 0.08, "lift_at_top_k": 1.5},
    ]
    spec_folds = [
        {"is_specificity_only": True, "auc": float("nan"), "brier": 0.02, "ece": float("nan"), "lift_at_top_k": float("nan")},
    ]
    agg = aggregate_fold_metrics_classification(real_folds + spec_folds)
    assert agg["n_real_folds"] == 2
    assert agg["n_specificity_folds"] == 1
    assert agg["auc_mean"] == pytest.approx(0.65)
    assert agg["lift_at_top_k_mean"] == pytest.approx(1.75)
    # Brier is averaged across ALL folds, including the specificity-only one
    assert agg["brier_mean"] == pytest.approx((0.1 + 0.15 + 0.02) / 3)


# ============================================================================
# run_loio in classification mode
# ============================================================================


def test_run_loio_classification_requires_binarize_callable():
    with pytest.raises(ValueError, match="requires a binarize callable"):
        run_loio(lambda: None, task="classification")


@pytest.mark.slow
def test_run_loio_classification_end_to_end_on_real_fold():
    """Smoke test: classification mode against the real packaged loio_9fold dataset.

    Uses LightGBMClassification + BinaryTarget('bc_ge_1') at scale_idx=3 (S=64,
    cheapest). Asserts that the aggregate dict has classification-shaped keys
    and that AUC is computed on >= 1 real fold.
    """
    from src.modeling.binary_target import get_target
    from src.modeling.gbm import LGBMParams, make_factory

    target = get_target("bc_ge_1")
    factory = make_factory("lightgbm_classification", LGBMParams(n_estimators=50, early_stopping_rounds=10))
    result = run_loio(
        factory,
        binarize=target.binarize,
        task="classification",
        scheme="loio_9fold",
        scale_idx=3,
        verbose=False,
    )
    assert "auc_mean" in result.aggregate
    assert "brier_mean" in result.aggregate
    assert "ece_mean" in result.aggregate
    assert "lift_at_top_k_mean" in result.aggregate
    assert result.aggregate["n_real_folds"] >= 1
    assert result.aggregate["n_specificity_folds"] >= 1  # ESP_065711_1545 must be flagged
    # Snapshot carries the task tag for downstream parquet readers
    assert result.snapshot["task"] == "classification"
