"""Abstract Model interface shared by all baselines.

The LOIO runner in `src.modeling.evaluate` doesn't know whether it's training a GBM,
a two-stage hurdle, or a CNN -- it just calls `fit / predict / save / load /
model_hash`. New baselines plug in by implementing this Protocol.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Model(Protocol):
    """A trainable model with a uniform fit/predict/persist surface.

    `name` is the human-readable model family ('lightgbm_tweedie', 'cnn_log1p_huber',
    ...). `predict` returns a vector of `fractional_area` predictions on the original
    target scale (NOT log space) -- baselines that train on a transformed target
    must back-transform internally so all evaluation code can treat predictions
    uniformly.

    For two-stage models, `predict_presence_prob` returns the binary stage's
    probability output (or None for single-stage models). The evaluator persists it
    to the prediction parquet so two-stage diagnostics are reproducible.
    """

    name: str

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def predict_presence_prob(self, X: np.ndarray) -> np.ndarray | None: ...

    def save(self, path: str | Path) -> None: ...

    def load(self, path: str | Path) -> None: ...

    def model_hash(self) -> str: ...


def hash_bytes(blob: bytes) -> str:
    """SHA256 hex digest of a byte string -- the canonical model_hash format."""
    return hashlib.sha256(blob).hexdigest()
