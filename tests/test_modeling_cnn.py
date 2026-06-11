"""Light smoke tests for the CNN models (no data on disk).

`gather_patches` is monkeypatched to return synthetic uint8 patches, so these exercise
the SmallCNN backbone + the regressor/classifier fit/predict contracts without needing
`dataset*/context_patches/`. The full LOIO behaviour is covered by the integration runs.
"""
import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy/torch

import numpy as np
import pandas as pd
import pytest
import torch

from src.modeling import cnn as cnn_mod
from src.modeling.cnn import (
    AUG_CELLS, CNNParams, SmallCNN, SmallCNNClassifier, SmallCNNRegressor, _PatchDataset,
)


def test_smallcnn_forward_shape():
    net = SmallCNN(patch_size_px=32)
    out = net(torch.zeros(4, 1, 32, 32))
    assert out.shape == (4,)


def _fake_gather(monkeypatch, n: int, S: int):
    rng = np.random.default_rng(0)
    patches = rng.integers(0, 256, size=(n, S, S), dtype=np.uint8)

    def fake(keys, patch_size_px, *, dataset_dir=None):
        k = len(keys)
        return patches[:k], np.arange(k)

    monkeypatch.setattr(cnn_mod, "gather_patches", fake)


def test_classifier_predict_is_probability(monkeypatch):
    n, S = 64, 32
    _fake_gather(monkeypatch, n, S)
    keys = pd.DataFrame({"obs_id": ["x"] * n})
    y = (np.arange(n) % 2).astype(float)  # balanced 0/1
    m = SmallCNNClassifier(params=CNNParams(patch_size_px=S, epochs=2, batch_size=16))
    m.bind_train_data(keys, y)
    m.bind_val_data(keys.iloc[:16], y[:16])
    m.fit(np.empty((n, 0), dtype=np.float32), y)
    m.bind_predict_data(keys)
    p = m.predict(np.empty((n, 0), dtype=np.float32))
    assert p.shape == (n,)
    assert np.all((p >= 0.0) & (p <= 1.0))
    pp = m.predict_presence_prob(np.empty((n, 0), dtype=np.float32))
    assert pp is not None and np.all((pp >= 0.0) & (pp <= 1.0))


def test_regressor_predict_is_nonnegative(monkeypatch):
    n, S = 64, 32
    _fake_gather(monkeypatch, n, S)
    keys = pd.DataFrame({"obs_id": ["x"] * n})
    y = np.abs(np.random.default_rng(1).normal(0, 0.01, n))
    m = SmallCNNRegressor(params=CNNParams(patch_size_px=S, epochs=2, batch_size=16))
    m.bind_train_data(keys, y)
    m.bind_val_data(keys.iloc[:16], y[:16])
    m.fit(np.empty((n, 0), dtype=np.float32), y)
    m.bind_predict_data(keys)
    p = m.predict(np.empty((n, 0), dtype=np.float32))
    assert p.shape == (n,)
    assert np.all(p >= 0.0)


# ---------------------------------------------------------------------------
# W2 augmentation cells (PLAN_CNN.md §4.2)
# ---------------------------------------------------------------------------


def test_unknown_aug_cell_rejected():
    with pytest.raises(ValueError, match="aug_cell"):
        CNNParams(aug_cell="nonsense")


def test_cell_none_is_identity_div255():
    """Cell A: no augmentation stages fire; input is the plain /255 cast."""
    rng = np.random.default_rng(2)
    patches = rng.integers(0, 256, size=(4, 16, 16), dtype=np.uint8)
    ds = _PatchDataset(patches, np.zeros(4, dtype=np.float32), augment=True, rng_seed=0,
                       **AUG_CELLS["none"])
    x, _ = ds[1]
    np.testing.assert_allclose(x.numpy()[0], patches[1].astype(np.float32) / 255.0)


def test_cell_geometric_preserves_pixel_multiset():
    """Cell B: flips/rots only -- pixel values are rearranged, never changed."""
    rng = np.random.default_rng(3)
    patches = rng.integers(0, 256, size=(4, 16, 16), dtype=np.uint8)
    ds = _PatchDataset(patches, np.zeros(4, dtype=np.float32), augment=True, rng_seed=0,
                       **AUG_CELLS["geometric"])
    x, _ = ds[0]
    got = np.sort((x.numpy()[0] * 255.0).round().astype(np.uint8), axis=None)
    want = np.sort(patches[0], axis=None)
    np.testing.assert_array_equal(got, want)


def test_cell_d_per_patch_std_applies_without_augment():
    """Cell D's standardization is a normalization choice: it must apply at eval time
    (augment=False) too, and be finite on a constant DN<=1 clip patch (std floor)."""
    patches = np.ones((2, 16, 16), dtype=np.uint8)  # DN=1 clip-like, std=0
    ds = _PatchDataset(patches, np.zeros(2, dtype=np.float32), augment=False, rng_seed=0,
                       per_patch_std=True)
    x, _ = ds[0]
    arr = x.numpy()
    assert np.isfinite(arr).all()
    np.testing.assert_allclose(arr, 0.0, atol=1e-6)  # (1 - 1)/max(0, 1) = 0


def test_classifier_runs_under_each_cell(monkeypatch):
    n, S = 32, 16
    _fake_gather(monkeypatch, n, S)
    keys = pd.DataFrame({"obs_id": ["x"] * n})
    y = (np.arange(n) % 2).astype(float)
    for cell in AUG_CELLS:
        m = SmallCNNClassifier(params=CNNParams(patch_size_px=S, epochs=1, batch_size=16,
                                                aug_cell=cell))
        m.bind_train_data(keys, y)
        m.fit(np.empty((n, 0), dtype=np.float32), y)
        m.bind_predict_data(keys)
        p = m.predict(np.empty((n, 0), dtype=np.float32))
        assert np.all((p >= 0.0) & (p <= 1.0)), cell
