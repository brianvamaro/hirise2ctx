"""Unit tests for src.modeling.gbm: fit/predict/save/load on synthetic data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.modeling.gbm import (
    CLASSIFICATION_VARIANTS,
    LGBMParams,
    LightGBMClassification,
    LightGBMLog1pHuber,
    LightGBMTweedie,
    LightGBMTwoStage,
    VARIANT_CONSTRUCTORS,
    make_factory,
    snapshot_params,
)

REGRESSION_VARIANTS = [v for v in VARIANT_CONSTRUCTORS if v not in CLASSIFICATION_VARIANTS]


def _synth(n: int = 600, n_features: int = 6, pos_frac: float = 0.1, seed: int = 0):
    """Synthetic zero-inflated regression task."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    # Linear positive intensity + heavy zero mass
    intensity = (X[:, 0] + X[:, 1]) ** 2 * 0.05
    pos_mask = rng.random(n) < pos_frac
    y = np.where(pos_mask, np.clip(intensity, 1e-6, 0.5), 0.0).astype(np.float64)
    return X, y


def test_variant_constructors_complete():
    assert set(VARIANT_CONSTRUCTORS) == {
        "lightgbm_tweedie", "lightgbm_log1p_huber", "lightgbm_two_stage",
        "lightgbm_two_stage_balanced", "lightgbm_two_stage_weighted",
        "lightgbm_two_stage_gamma", "lightgbm_two_stage_combined",
        "lightgbm_classification",
    }


@pytest.mark.parametrize("variant", REGRESSION_VARIANTS)
def test_fit_predict_basic(variant):
    X, y = _synth(n=800, pos_frac=0.15)
    factory = make_factory(variant, LGBMParams(n_estimators=50, early_stopping_rounds=0))
    model = factory()
    model.fit(X[:600], y[:600], eval_set=(X[600:], y[600:]))
    preds = model.predict(X[600:])
    # Predictions on original scale, non-negative
    assert preds.shape == (200,)
    assert (preds >= 0).all()
    assert preds.dtype == np.float64 or preds.dtype == np.float32


def test_two_stage_predict_presence_prob_in_unit_interval():
    X, y = _synth(n=800, pos_frac=0.15)
    model = LightGBMTwoStage(params=LGBMParams(n_estimators=50, early_stopping_rounds=0))
    model.fit(X[:600], y[:600])
    p = model.predict_presence_prob(X[600:])
    assert p is not None
    assert p.shape == (200,)
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_single_stage_presence_prob_returns_none():
    X, y = _synth()
    model = LightGBMTweedie(params=LGBMParams(n_estimators=20, early_stopping_rounds=0))
    model.fit(X[:400], y[:400])
    assert model.predict_presence_prob(X[400:]) is None


@pytest.mark.parametrize("variant", REGRESSION_VARIANTS)
def test_save_load_roundtrip(variant, tmp_path):
    X, y = _synth(n=400, pos_frac=0.2)
    factory = make_factory(variant, LGBMParams(n_estimators=30, early_stopping_rounds=0))
    m1 = factory()
    m1.fit(X[:300], y[:300])
    if variant == "lightgbm_two_stage":
        save_path = tmp_path / "two_stage"
    else:
        save_path = tmp_path / "booster.txt"
    m1.save(save_path)
    h1 = m1.model_hash()
    m2 = factory()
    m2.load(save_path)
    h2 = m2.model_hash()
    assert h1 == h2
    np.testing.assert_allclose(m1.predict(X[300:]), m2.predict(X[300:]), rtol=0, atol=0)


# ============================================================================
# LightGBMClassification (Stage 5b)
# ============================================================================


def _synth_binary(n: int = 600, n_features: int = 6, pos_frac: float = 0.1, seed: int = 0):
    """Synthetic binary classification task with controllable class imbalance."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    # Signal: positives concentrated where X[:, 0] + X[:, 1] is large
    score = X[:, 0] + X[:, 1] + 0.3 * rng.standard_normal(n)
    cutoff = np.quantile(score, 1.0 - pos_frac)
    y = (score >= cutoff).astype(np.int8)
    return X, y


def test_classification_fit_predict_returns_probabilities_in_unit_interval():
    X, y = _synth_binary(n=600, pos_frac=0.2)
    model = LightGBMClassification(params=LGBMParams(n_estimators=50, early_stopping_rounds=0))
    model.fit(X[:450], y[:450], eval_set=(X[450:], y[450:]))
    p = model.predict(X[450:])
    assert p.shape == (150,)
    assert (p >= 0.0).all() and (p <= 1.0).all()


def test_classification_predict_presence_prob_equals_predict():
    """For this variant, both methods return P(y=1)."""
    X, y = _synth_binary(n=400, pos_frac=0.25)
    model = LightGBMClassification(params=LGBMParams(n_estimators=30, early_stopping_rounds=0))
    model.fit(X[:300], y[:300])
    np.testing.assert_array_equal(model.predict(X[300:]), model.predict_presence_prob(X[300:]))


def test_classification_save_load_roundtrip(tmp_path):
    X, y = _synth_binary(n=400, pos_frac=0.2)
    m1 = LightGBMClassification(params=LGBMParams(n_estimators=30, early_stopping_rounds=0))
    m1.fit(X[:300], y[:300])
    save_path = tmp_path / "classifier.txt"
    m1.save(save_path)
    h1 = m1.model_hash()
    m2 = LightGBMClassification()
    m2.load(save_path)
    h2 = m2.model_hash()
    assert h1 == h2
    np.testing.assert_allclose(m1.predict(X[300:]), m2.predict(X[300:]), rtol=0, atol=0)


def test_classification_auto_scale_pos_weight_on_imbalanced_synthetic():
    """With ~1% positives, the auto scale_pos_weight should pull predictions
    away from the all-zero collapse. Without weighting, LightGBM on 99%-negative
    data converges to predicting probabilities ~ base rate; with weighting,
    predictions for true positives should be discernibly higher than for true
    negatives."""
    X, y = _synth_binary(n=2000, pos_frac=0.01, seed=1)
    model = LightGBMClassification(params=LGBMParams(n_estimators=100, early_stopping_rounds=0))
    model.fit(X, y)
    # The auto scale_pos_weight should be roughly neg/pos ~ 99
    assert model._scale_pos_weight is not None
    assert 50 < model._scale_pos_weight < 200
    # On the training set, mean predicted prob on positives should exceed mean
    # on negatives by a clear margin (this is a sanity floor, not a tight bound).
    p = model.predict(X)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.1


def test_classification_no_scale_pos_weight_when_y_is_all_one_class():
    """Degenerate case: y is all-zero. We must not divide by zero."""
    X = np.random.default_rng(0).standard_normal((200, 4)).astype(np.float32)
    y = np.zeros(200, dtype=np.int8)
    model = LightGBMClassification(params=LGBMParams(n_estimators=20, early_stopping_rounds=0))
    model.fit(X, y)  # should not raise
    assert model._scale_pos_weight is None


def test_classification_raises_on_non_binary_y():
    X = np.random.default_rng(0).standard_normal((100, 4)).astype(np.float32)
    y = np.array([0, 1, 2] * 33 + [0])  # 0, 1, 2 -- not binary
    model = LightGBMClassification(params=LGBMParams(n_estimators=10))
    with pytest.raises(ValueError, match="binary 0/1"):
        model.fit(X, y)


def test_model_hash_is_stable():
    """Two models trained with the same seed on identical data have identical hashes."""
    X, y = _synth(n=400)
    p = LGBMParams(n_estimators=30, early_stopping_rounds=0, seed=42)
    m1 = LightGBMTweedie(params=p); m1.fit(X[:300], y[:300])
    m2 = LightGBMTweedie(params=p); m2.fit(X[:300], y[:300])
    assert m1.model_hash() == m2.model_hash()


def test_snapshot_params_captures_variant_and_params():
    snap = snapshot_params("lightgbm_tweedie", LGBMParams(n_estimators=123, tweedie_variance_power=1.7))
    assert snap["variant"] == "lightgbm_tweedie"
    assert snap["params"]["n_estimators"] == 123
    assert snap["params"]["tweedie_variance_power"] == 1.7
    assert "positive_rule_eps" in snap


def test_two_stage_handles_few_positives_gracefully():
    """When fewer than 10 positives, the magnitude head is skipped, not raised."""
    X = np.random.default_rng(0).standard_normal((100, 4)).astype(np.float32)
    y = np.zeros(100)  # ALL zero
    model = LightGBMTwoStage(params=LGBMParams(n_estimators=20, early_stopping_rounds=0))
    model.fit(X, y)  # should not raise
    preds = model.predict(X)
    # All zero predictions because magnitude head was skipped
    np.testing.assert_allclose(preds, np.zeros(100))
