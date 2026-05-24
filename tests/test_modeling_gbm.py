"""Unit tests for src.modeling.gbm: fit/predict/save/load on synthetic data."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.modeling.gbm import (
    LGBMParams,
    LightGBMLog1pHuber,
    LightGBMTweedie,
    LightGBMTwoStage,
    VARIANT_CONSTRUCTORS,
    make_factory,
    snapshot_params,
)


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
    }


@pytest.mark.parametrize("variant", list(VARIANT_CONSTRUCTORS))
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


@pytest.mark.parametrize("variant", list(VARIANT_CONSTRUCTORS))
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
