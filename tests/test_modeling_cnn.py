"""Light smoke tests for the CNN models (no data on disk).

`gather_patches` is monkeypatched to return synthetic uint8 patches, so these exercise
the SmallCNN backbone + the regressor/classifier fit/predict contracts without needing
`dataset*/context_patches/`. The full LOIO behaviour is covered by the integration runs.
"""
import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy/torch

import numpy as np
import pandas as pd
import torch

from src.modeling import cnn as cnn_mod
from src.modeling.cnn import CNNParams, SmallCNN, SmallCNNClassifier, SmallCNNRegressor


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
