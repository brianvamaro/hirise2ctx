"""Unit tests for embedding-space novelty (PLAN_FM §2.7).

Synthetic Gaussian clouds only -- no embeddings or dataset on disk. Asserts the
deployment contracts the reliability overlay relies on: far-from-training tiles
score higher than near ones, margin (NaN) tiles stay NaN and don't leak into the
aggregate, and scoring is deterministic.
"""
import numpy as np
import pytest

from src.reliability import (
    MahalanobisNovelty, KNNNovelty, valid_rows, aggregate_per_image,
)


def _clouds(seed=0, d=64):
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(2000, d)).astype("float32")
    near = rng.normal(size=(40, d)).astype("float32")
    far = (rng.normal(size=(40, d)) + 12.0).astype("float32")   # shifted far away
    return train, near, far


def test_valid_rows_flags_nan():
    X = np.ones((3, 4), dtype="float32")
    X[1, 2] = np.nan
    np.testing.assert_array_equal(valid_rows(X), [True, False, True])


@pytest.mark.parametrize("factory", [
    lambda: MahalanobisNovelty(n_components=32),
    lambda: KNNNovelty(k=10, metric="euclidean", max_reference=1000),
])
def test_far_scores_higher_than_near(factory):
    train, near, far = _clouds()
    scorer = factory().fit(train)
    near_s = scorer.score(near)
    far_s = scorer.score(far)
    assert np.nanmedian(far_s) > np.nanmedian(near_s)


@pytest.mark.parametrize("factory", [
    lambda: MahalanobisNovelty(n_components=32),
    lambda: KNNNovelty(k=10, max_reference=1000),
])
def test_nan_rows_score_nan(factory):
    train, near, _ = _clouds()
    X = near.copy()
    X[0] = np.nan
    s = factory().fit(train).score(X)
    assert np.isnan(s[0])
    assert np.isfinite(s[1:]).all()


def test_mahalanobis_deterministic():
    train, near, _ = _clouds()
    s1 = MahalanobisNovelty(n_components=32).fit(train).score(near)
    s2 = MahalanobisNovelty(n_components=32).fit(train).score(near)
    np.testing.assert_allclose(s1, s2)


def test_knn_reference_subsample_deterministic():
    train, near, _ = _clouds()
    s1 = KNNNovelty(k=10, max_reference=500, seed=7).fit(train).score(near)
    s2 = KNNNovelty(k=10, max_reference=500, seed=7).fit(train).score(near)
    np.testing.assert_allclose(s1, s2)


def test_aggregate_ignores_nan_and_empty():
    obs = np.array(["a", "a", "b", "b", "c"])
    scores = np.array([1.0, 3.0, np.nan, np.nan, 5.0])
    agg = aggregate_per_image(obs, scores, how="median")
    assert agg["a"] == 2.0          # median of 1,3
    assert "b" not in agg           # all-NaN image dropped
    assert agg["c"] == 5.0


def test_fit_raises_when_too_few_tiles():
    with pytest.raises(ValueError):
        MahalanobisNovelty(n_components=256).fit(np.zeros((10, 64), dtype="float32"))
    with pytest.raises(ValueError):
        KNNNovelty(k=50).fit(np.zeros((10, 64), dtype="float32"))


def test_score_before_fit_raises():
    with pytest.raises(RuntimeError):
        MahalanobisNovelty().score(np.zeros((3, 64), dtype="float32"))
    with pytest.raises(RuntimeError):
        KNNNovelty().score(np.zeros((3, 64), dtype="float32"))
