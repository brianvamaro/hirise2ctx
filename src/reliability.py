"""Embedding-space novelty as a per-tile reliability signal (PLAN_FM §2.7).

The frozen recipe predicts a rich/poor probability for every CTX tile, on or off
HiRISE coverage. The §2.3 confirmation certifies the recipe generalizes *on
average*; this module answers the complementary deployment question — *where* on
the map to trust it — with a label-free per-tile score: "is this CTX texture like
what the model trained on?"

Two novelty methods over the frozen 768-dim GeM embeddings (no GPU, no labels):

- ``MahalanobisNovelty`` — distance to the training embedding cloud in a
  PCA-whitened subspace (top-k components tame the 768² covariance; truncation is
  the regularizer). Parametric, single Gaussian assumption.
- ``KNNNovelty`` — mean cosine distance to the k nearest *training* tiles
  (non-parametric; handles a multimodal training distribution). Reference set is
  subsampled for tractability — the novelty estimate is robust to it.

Higher score = more novel = less trustworthy. Both return NaN for window-margin
tiles (all-NaN embedding rows) so the reliability raster inherits the same mask
as the probability raster. Validation (does novelty flag where the frozen recipe
*itself* underperforms?) and the map overlay live in
``scripts/probes/_fm_reliability_validation.py`` and ``src.mapping``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MahalanobisNovelty", "KNNNovelty", "valid_rows", "aggregate_per_image"]


def valid_rows(X: np.ndarray) -> np.ndarray:
    """Boolean mask of embedding rows that are usable (no NaN component).

    Window-margin tiles whose 3×3 context box spilled the read edge are stored as
    all-NaN rows (loaders/embed convention); they carry no texture to score.
    """
    return ~np.isnan(np.asarray(X, dtype=np.float64)).any(axis=1)


# ============================================================================
# Mahalanobis distance to the training cloud (PCA-whitened subspace)
# ============================================================================


@dataclass
class MahalanobisNovelty:
    """Mahalanobis distance to the training embedding mean in a whitened subspace.

    Fit centers the training embeddings and fits a whitening PCA on the top
    ``n_components`` directions; ``score`` returns the L2 norm of the whitened
    projection — exactly the Mahalanobis distance under the rank-``n_components``
    covariance, with truncation acting as shrinkage so tiny-variance directions
    can't blow the score up. ``eps`` floors the per-component variance.
    """

    n_components: int = 256
    eps: float = 1e-6
    _mean: np.ndarray | None = None
    _components: np.ndarray | None = None   # (k, d) principal axes
    _inv_std: np.ndarray | None = None      # (k,) 1/sqrt(eigenvalue) per axis

    def fit(self, X: np.ndarray) -> "MahalanobisNovelty":
        from sklearn.decomposition import PCA

        X = np.asarray(X, dtype=np.float64)
        X = X[valid_rows(X)]
        if X.shape[0] <= self.n_components:
            raise ValueError(
                f"need > n_components={self.n_components} valid tiles to fit, got {X.shape[0]}")
        k = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self._mean = X.mean(axis=0)
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
        pca.fit(X - self._mean)
        self._components = pca.components_                       # (k, d)
        self._inv_std = 1.0 / np.sqrt(pca.explained_variance_ + self.eps)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise RuntimeError("MahalanobisNovelty.score before fit")
        X = np.asarray(X, dtype=np.float64)
        out = np.full(X.shape[0], np.nan, dtype=np.float64)
        m = valid_rows(X)
        if m.any():
            proj = (X[m] - self._mean) @ self._components.T     # (n, k)
            whitened = proj * self._inv_std                     # unit-variance axes
            out[m] = np.sqrt((whitened ** 2).sum(axis=1))
        return out


# ============================================================================
# kNN distance to the training cloud (non-parametric)
# ============================================================================


@dataclass
class KNNNovelty:
    """Mean distance to the k nearest *training* tiles (subsampled reference).

    ``metric="cosine"`` is natural for ViT embeddings (angle, not magnitude).
    The reference is randomly subsampled to ``max_reference`` rows — kNN novelty
    is robust to reference size, and brute cosine over the full ~150k×768 cloud
    per LOIO fold is needlessly slow. NaN rows score NaN.
    """

    k: int = 50
    metric: str = "cosine"
    max_reference: int = 20000
    seed: int = 0
    _nn: object = None

    def fit(self, X: np.ndarray) -> "KNNNovelty":
        from sklearn.neighbors import NearestNeighbors

        X = np.asarray(X, dtype=np.float32)
        X = X[valid_rows(X)]
        if X.shape[0] <= self.k:
            raise ValueError(f"need > k={self.k} valid tiles to fit, got {X.shape[0]}")
        if X.shape[0] > self.max_reference:
            rng = np.random.default_rng(self.seed)
            X = X[rng.choice(X.shape[0], self.max_reference, replace=False)]
        self._nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric,
                                    algorithm="brute").fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._nn is None:
            raise RuntimeError("KNNNovelty.score before fit")
        X = np.asarray(X, dtype=np.float32)
        out = np.full(X.shape[0], np.nan, dtype=np.float64)
        m = valid_rows(X)
        if m.any():
            dist, _ = self._nn.kneighbors(X[m], return_distance=True)
            out[m] = dist.mean(axis=1)
        return out


# ============================================================================
# Per-image aggregation (for the taxonomy-validation step)
# ============================================================================


def aggregate_per_image(obs_ids: np.ndarray, scores: np.ndarray,
                        *, how: str = "median") -> dict[str, float]:
    """Collapse per-tile novelty to one value per image (NaN tiles ignored).

    ``how`` is "median" (robust, the map-overlay quantity) or "mean". Images with
    no valid tile are omitted.
    """
    obs_ids = np.asarray(obs_ids)
    scores = np.asarray(scores, dtype=np.float64)
    reducer = np.nanmedian if how == "median" else np.nanmean
    out: dict[str, float] = {}
    for obs in np.unique(obs_ids):
        vals = scores[obs_ids == obs]
        if np.isfinite(vals).any():
            out[str(obs)] = float(reducer(vals))
    return out
