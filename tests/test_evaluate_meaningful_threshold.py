"""run_loio must thread a target-appropriate `meaningful_threshold` into the
regression metric pack (the rich/poor cut). Regression test for the count-target
bug: the default 1e-2 applied to raw boulder counts collapses to count > 0.01 ==
presence (count >= 1), the degenerate metric we don't use (Brian, 2026-06-12).

Synthetic LOIO fold + a dummy model that predicts the first feature; no torch,
no data on disk.
"""
import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd

from src.modeling.evaluate import run_loio
from src.modeling.loaders import Fold


class _DummyReg:
    name = "dummy_reg"

    def fit(self, X, y, *, groups=None, eval_set=None):
        pass

    def predict(self, X):
        return np.asarray(X)[:, 0]  # perfect ranking: X[:,0] == y_true below

    def predict_presence_prob(self, X):
        return None

    def save(self, path):
        pass

    def load(self, path):
        pass

    def model_hash(self):
        return "dummy"


def _fold() -> Fold:
    # 2 train images (groups 0,1) + 1 held-out test image (group 2).
    y_tr = np.linspace(0.0, 100.0, 10)
    g_tr = np.array([0] * 5 + [1] * 5, dtype=np.int32)
    y_te = np.array([0.0, 10.0, 40.0, 60.0, 90.0, 120.0])  # counts; >50 -> 3 pos, >0.01 -> 5 pos
    g_te = np.array([2] * 6, dtype=np.int32)
    keys_tr = pd.DataFrame({"obs_id": ["A"] * 5 + ["B"] * 5, "ti": range(10), "tj": [0] * 10})
    keys_te = pd.DataFrame({"obs_id": ["C"] * 6, "ti": range(6), "tj": [0] * 6})
    return Fold(
        fold_idx=0, scheme="t", scale_idx=2,
        X_train=y_tr.reshape(-1, 1).astype(np.float32), y_train=pd.DataFrame({"boulder_count": y_tr}),
        groups_train=g_tr, keys_train=keys_tr,
        X_test=y_te.reshape(-1, 1).astype(np.float32), y_test=pd.DataFrame({"boulder_count": y_te}),
        groups_test=g_te, keys_test=keys_te,
        feature_names=["f"], obs_to_int={"A": 0, "B": 1, "C": 2}, held_out_obs_ids=["C"],
    )


def _run(**kw):
    return run_loio(_DummyReg, target_col="boulder_count", task="regression",
                    fold_iter=lambda: iter([_fold()]), verbose=False, **kw)


def test_threshold_is_threaded_to_metrics():
    m = _run(meaningful_threshold=50.0).per_fold_metrics[0]
    assert m["meaningful_threshold"] == 50.0
    assert m["n_meaningful_positive"] == 3        # {60, 90, 120} > 50
    assert not np.isnan(m["meaningful_auc"])      # both classes present -> defined


def test_default_threshold_is_presence_on_counts():
    # Documents the trap the fix guards against: default 1e-2 on raw counts is presence.
    m = _run().per_fold_metrics[0]
    assert m["meaningful_threshold"] == 1e-2
    assert m["n_meaningful_positive"] == 5        # everything > 0.01 except the zero tile
