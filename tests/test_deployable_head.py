"""Unit tests for the productized MLP head + deployable ensemble (PLAN_FM §2.6.A).

Synthetic data only (CPU, no checkpoint / dataset). A linearly-separable-ish
embedding cloud lets the tiny MLP learn something so the contracts (shape,
probability range, scaler parity, save/load round-trip, NaN-row imputation) are
exercised on a non-degenerate fit -- without asserting any accuracy gate (that
is the LOIO harness's job, and the recipe is frozen).
"""
import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy/torch

import numpy as np
import pytest

from src.modeling.mlp_head import (
    DeployableHead, FeatureScaler, MLPClassifierHead, build_mlp,
)

EMBED_DIM = 32  # small stand-in for 768 to keep the tests fast


def _synthetic(n=600, d=EMBED_DIM, n_groups=4, seed=0):
    """Embedding cloud + a label correlated with a fixed direction + group codes."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d)
    score = X @ w
    y = (score > np.quantile(score, 0.65)).astype(np.int8)  # ~35% positive
    groups = rng.integers(0, n_groups, size=n).astype(np.int32)
    return X, y, groups


# ----------------------------------------------------------------------------
# FeatureScaler
# ----------------------------------------------------------------------------


def test_scaler_standardizes_and_imputes_nan():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((50, 5)).astype(np.float32)
    X[3, 2] = np.nan  # one NaN cell -> imputed to the column median
    sc = FeatureScaler()
    Xs = sc.fit(X)
    assert np.isfinite(Xs).all()
    # Standardized: per-column mean ~0, std ~1 (on the imputed matrix).
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-5)


def test_scaler_all_nan_column_maps_to_constant():
    X = np.ones((10, 3), dtype=np.float32)
    X[:, 1] = np.nan  # all-NaN column -> median 0, std 0 -> 1, output 0
    sc = FeatureScaler()
    Xs = sc.fit(X)
    assert np.all(Xs[:, 1] == 0.0) and np.isfinite(Xs).all()


def test_scaler_roundtrip_arrays():
    X = np.random.default_rng(2).standard_normal((20, 6)).astype(np.float32)
    sc = FeatureScaler()
    sc.fit(X)
    sc2 = FeatureScaler.from_arrays(sc.to_arrays())
    np.testing.assert_array_equal(sc.apply(X), sc2.apply(X))


# ----------------------------------------------------------------------------
# MLPClassifierHead
# ----------------------------------------------------------------------------


def test_build_mlp_output_is_single_logit():
    import torch

    net = build_mlp(EMBED_DIM)
    out = net(torch.zeros(7, EMBED_DIM))
    assert out.shape == (7, 1)


def test_head_fit_predict_probability_range():
    X, y, groups = _synthetic()
    head = MLPClassifierHead(seed=0, batch=256, epochs=15)
    val = groups == 0
    head.fit(X[~val], y[~val], eval_set=(X[val], y[val]))
    p = head.predict(X)
    assert p.shape == (X.shape[0],)
    assert p.min() >= 0.0 and p.max() <= 1.0


def test_head_predict_imputes_nan_rows():
    X, y, groups = _synthetic(n=300)
    head = MLPClassifierHead(seed=0, batch=128, epochs=10)
    head.fit(X, y, eval_set=(X[:64], y[:64]))
    Xn = X.copy()
    Xn[0] = np.nan  # fully-NaN row -> scaler imputes -> finite prediction
    p = head.predict(Xn)
    assert np.isfinite(p).all()


def test_head_save_load_roundtrip(tmp_path):
    X, y, _ = _synthetic(n=300)
    head = MLPClassifierHead(seed=1, batch=128, epochs=10)
    head.fit(X, y, eval_set=(X[:64], y[:64]))
    p = head.predict(X)
    head.save(tmp_path / "h")
    reloaded = MLPClassifierHead()
    reloaded.load(tmp_path / "h")
    np.testing.assert_allclose(reloaded.predict(X), p, rtol=0, atol=1e-6)
    assert reloaded.model_hash() == head.model_hash()


# ----------------------------------------------------------------------------
# DeployableHead
# ----------------------------------------------------------------------------


def test_deployable_fit_predict_and_ensemble_mean():
    X, y, groups = _synthetic(n=800, n_groups=5)
    head = DeployableHead(seeds=(0, 1, 2), batch=256, epochs=12)
    head.fit(X, y, groups=groups, verbose=False)
    assert len(head._members) == 3
    p = head.predict(X)
    assert p.shape == (X.shape[0],)
    assert p.min() >= 0.0 and p.max() <= 1.0
    # Ensemble prediction is exactly the mean of the member predictions.
    members = np.mean([m.predict(X) for m in head._members], axis=0)
    np.testing.assert_allclose(p, members, rtol=0, atol=1e-9)


def test_deployable_requires_multiple_groups():
    X, y, _ = _synthetic(n=100)
    head = DeployableHead(seeds=(0,), batch=64, epochs=3)
    with pytest.raises(ValueError):
        head.fit(X, y, groups=np.zeros(len(y), dtype=np.int32), verbose=False)


def test_deployable_save_load_roundtrip(tmp_path):
    X, y, groups = _synthetic(n=600, n_groups=4)
    obs_to_int = {f"IMG_{i}": i for i in range(4)}
    head = DeployableHead(seeds=(0, 1), batch=256, epochs=10)
    head.fit(X, y, groups=groups, obs_to_int=obs_to_int, verbose=False)
    p = head.predict(X)
    out = tmp_path / "deploy"
    head.save(out)
    assert (out / "recipe.json").exists()

    reloaded = DeployableHead.load(out)
    np.testing.assert_allclose(reloaded.predict(X), p, rtol=0, atol=1e-6)
    assert reloaded.recipe_hash() == head.recipe_hash()
    assert reloaded.model_hash() == head.model_hash()
    assert set(reloaded._train_obs_ids) == set(obs_to_int)


def test_recipe_hash_is_config_only():
    # Same recipe config -> same recipe_hash regardless of trained weights.
    a = DeployableHead(seeds=(0, 1, 2))
    b = DeployableHead(seeds=(0, 1, 2))
    assert a.recipe_hash() == b.recipe_hash()
    c = DeployableHead(seeds=(0, 1, 2), dropout=0.5)
    assert c.recipe_hash() != a.recipe_hash()
