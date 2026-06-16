"""Post-hoc calibration & de-compression of the boulder-abundance outputs.

Both products compress toward the middle (PLAN_Calibration.md): the Tier-1 rich/poor
probability is mis-calibrated, and the Tier-2 abundance regressor is two-sided
regression-to-the-mean (over-predicts the low end / floors above true zero,
under-predicts the high tail). This module holds the *post-hoc, ranking-preserving*
calibrators that sit AFTER the frozen recipe — so they de-compress the outputs
without reopening the freeze.

All calibrators are monotone (rank-preserving): they change the *values*, never the
ordering, so AUC / Spearman / NDCG are invariant by construction. Fit them
LOIO-honestly (`loio_calibrate`): for each held-out image, fit on the OTHER images
and apply to the held-out one — never fit on the data you score.

Primitives
----------
- ``reliability_curve`` / ``expected_calibration_error`` — Tier-1 calibration diagnosis.
- ``TemperatureScaler`` — 1-parameter probability calibration; AUC-exact.
- ``IsotonicCalibrator`` — monotone pred→E[true] fit (de-compresses toward the mean).
- ``quantile_match`` — map the prediction distribution onto the truth distribution
  (histogram/quantile transfer); recovers the full marginal incl. the tail.
- ``compression_metrics`` — the scorecard (top/low ratios, near-zero share, marginal L1).
- ``loio_calibrate`` — the honest fit-on-others / apply-to-held-out protocol.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "reliability_curve", "expected_calibration_error",
    "TemperatureScaler", "BetaCalibrator", "IsotonicCalibrator",
    "QuantileMatcher", "quantile_match", "CalibrationLayer",
    "compression_metrics", "loio_calibrate",
]


# ============================================================================
# Tier-1 probability calibration diagnosis
# ============================================================================


def reliability_curve(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10):
    """Binned (mean confidence, empirical accuracy, count) for a reliability diagram.

    `y_true` is 0/1; `p_pred` in [0, 1]. Equal-width bins on the probability axis.
    Empty bins are dropped.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(p_pred, dtype=np.float64), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf.append(p[m].mean()); acc.append(y_true[m].mean()); cnt.append(int(m.sum()))
    return np.array(conf), np.array(acc), np.array(cnt)


def expected_calibration_error(y_true: np.ndarray, p_pred: np.ndarray, n_bins: int = 10) -> float:
    """ECE: count-weighted mean |accuracy - confidence| over probability bins."""
    conf, acc, cnt = reliability_curve(y_true, p_pred, n_bins)
    if cnt.sum() == 0:
        return float("nan")
    return float(np.sum(cnt * np.abs(acc - conf)) / cnt.sum())


# ============================================================================
# Monotone calibrators (rank-preserving by construction)
# ============================================================================

_EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1 - _EPS)
    return np.log(p / (1 - p))


class TemperatureScaler:
    """One-parameter probability calibration: p' = sigmoid(logit(p) / T).

    Fit T>0 by minimizing binary cross-entropy. T>1 softens (spreads toward 0.5),
    T<1 sharpens. Strictly monotone in p, so ROC-AUC / ranking are unchanged — it
    only fixes the *confidence*, the canonical safe Tier-1 calibrator.
    """

    def __init__(self) -> None:
        self.T: float = 1.0

    def fit(self, p_pred: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        from scipy.optimize import minimize_scalar

        z = _logit(p_pred)
        y = np.asarray(y_true, dtype=np.float64)

        def nll(logT):
            t = np.exp(logT)
            q = 1.0 / (1.0 + np.exp(-z / t))
            q = np.clip(q, _EPS, 1 - _EPS)
            return -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))

        res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def predict(self, p_pred: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-_logit(p_pred) / self.T))


class BetaCalibrator:
    """Beta calibration (Kull, Silva Filho & Flach 2017): a 3-parameter *smooth*
    strictly-monotone probability map `p' = sigmoid(a·ln p − b·ln(1−p) + c)`.

    Fit by logistic regression of `y` on the two features `[ln p, ln(1−p)]`; the
    coefficients give `(a, −b)` and the intercept `c`. Flexible enough to bend the
    two ends independently (fixes the over-confident highs AND the lows that one-knob
    temperature trades off), yet — unlike isotonic — strictly monotone with no flat
    steps, so ROC-AUC / ranking are preserved. Monotonicity needs a≥0, b≥0; if the
    unconstrained fit violates it (rare here) we drop the offending feature and refit
    (Kull's fallback), guaranteeing a monotone map.
    """

    def __init__(self) -> None:
        self.a = 1.0
        self.b = 1.0
        self.c = 0.0

    def fit(self, p_pred: np.ndarray, y_true: np.ndarray) -> "BetaCalibrator":
        from sklearn.linear_model import LogisticRegression

        p = np.clip(np.asarray(p_pred, dtype=np.float64), _EPS, 1 - _EPS)
        y = np.asarray(y_true, dtype=np.float64)
        s1, s2 = np.log(p), np.log(1 - p)

        def _fit(cols):
            lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
            lr.fit(np.column_stack(cols), y)
            return lr

        lr = _fit([s1, s2])
        a, nb = lr.coef_[0]            # coefficients on [ln p, ln(1-p)] = (a, -b)
        b = -nb
        if a < 0 or b < 0:            # enforce monotonicity by dropping the bad feature
            if a < 0:                 # drop ln p
                lr = _fit([s2]); a = 0.0; b = -lr.coef_[0][0]
            else:                     # drop ln(1-p)
                lr = _fit([s1]); a = lr.coef_[0][0]; b = 0.0
        self.a, self.b, self.c = float(max(a, 0.0)), float(max(b, 0.0)), float(lr.intercept_[0])
        return self

    def predict(self, p_pred: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p_pred, dtype=np.float64), _EPS, 1 - _EPS)
        z = self.a * np.log(p) - self.b * np.log(1 - p) + self.c
        return 1.0 / (1.0 + np.exp(-z))


class IsotonicCalibrator:
    """Monotone pred -> E[true] fit (sklearn isotonic, clipped at the train range).

    Minimizes squared error to the truth under a monotonicity constraint, so it
    de-compresses toward the conditional mean and preserves ranking. (Fits the
    *mean* at each level, so it recovers the tail less aggressively than
    ``quantile_match`` but adds no variance.)
    """

    def __init__(self) -> None:
        self._iso = None

    def fit(self, pred: np.ndarray, true: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        self._iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        self._iso.fit(np.asarray(pred, dtype=np.float64), np.asarray(true, dtype=np.float64))
        return self

    def predict(self, pred: np.ndarray) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator.predict before fit")
        return self._iso.predict(np.asarray(pred, dtype=np.float64))

    def knots(self) -> tuple[np.ndarray, np.ndarray]:
        """The fitted piecewise-linear knots `(x, y)`. `np.interp(v, x, y)` reproduces
        ``predict`` exactly (sklearn isotonic with ``out_of_bounds='clip'`` clamps `v`
        to `[x[0], x[-1]]` then linearly interpolates) — used to serialize the map
        without pickling the sklearn estimator."""
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator.knots before fit")
        return (np.asarray(self._iso.X_thresholds_, dtype=np.float64),
                np.asarray(self._iso.y_thresholds_, dtype=np.float64))


class QuantileMatcher:
    """Deployable quantile-matching: the *fixed* monotone map carrying a reference
    prediction distribution onto a reference truth distribution.

    The runtime form of :func:`quantile_match` — fit once on ``(ref_pred, ref_true)``,
    storing the paired sorted quantiles, then apply pointwise. Rank-preserving (so
    Spearman/AUC are invariant) and distribution-matching (so the calibrated marginal
    equals the truth marginal — recovering the high tail + true-zero mass). Because the
    map is a fixed function of the value, it does NOT re-rank per window: a tile gets
    the same calibrated value regardless of its neighbours.
    """

    def __init__(self, n_quantiles: int = 4000) -> None:
        self.n_quantiles = n_quantiles
        self._xp: np.ndarray | None = None   # sorted reference-prediction quantiles
        self._fp: np.ndarray | None = None   # sorted reference-truth quantiles

    def fit(self, ref_pred: np.ndarray, ref_true: np.ndarray) -> "QuantileMatcher":
        sp = np.sort(np.asarray(ref_pred, dtype=np.float64))
        st = np.sort(np.asarray(ref_true, dtype=np.float64))
        q = np.linspace(0, 1, min(len(sp), self.n_quantiles))
        self._xp = np.quantile(sp, q)
        self._fp = np.quantile(st, q)
        return self

    def predict(self, pred: np.ndarray) -> np.ndarray:
        if self._xp is None:
            raise RuntimeError("QuantileMatcher.predict before fit")
        return np.interp(np.asarray(pred, dtype=np.float64), self._xp, self._fp)

    def knots(self) -> tuple[np.ndarray, np.ndarray]:
        if self._xp is None:
            raise RuntimeError("QuantileMatcher.knots before fit")
        return self._xp, self._fp


def quantile_match(pred: np.ndarray, ref_pred: np.ndarray, ref_true: np.ndarray) -> np.ndarray:
    """Map `pred` through the monotone function that carries the ref-prediction
    distribution onto the ref-truth distribution (histogram / quantile transfer).

    By construction the calibrated marginal matches the truth marginal, so the high
    tail and the true-zero mass are both recovered — directly attacking compression
    — while monotonicity preserves ranking. Thin functional wrapper over
    :class:`QuantileMatcher` (fit-and-apply in one call).
    """
    return QuantileMatcher().fit(ref_pred, ref_true).predict(pred)


# ============================================================================
# Compression scorecard
# ============================================================================


def compression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                        *, rich_threshold: float = 1e-2) -> dict:
    """The de-compression scorecard for an abundance prediction.

    - ``spearman``       : rank skill (must-not-regress under calibration)
    - ``top_ratio``      : mean_pred/mean_true for true > rich_threshold (1.0 = calibrated; <1 = high tail squashed)
    - ``low_over``       : mean_pred/mean_true in the truly-zero bin (>>1 = low-end over-prediction)
    - ``near_zero_pred`` : share of predictions < 1e-4 (compare to the true exact-zero share)
    - ``near_zero_true`` : share of truth exactly zero
    - ``marginal_l1``    : mean |quantile(true) - quantile(pred)| over the CDF (0 = matched marginal)
    """
    from scipy.stats import spearmanr

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.clip(np.asarray(y_pred, dtype=np.float64), 0, None)
    top = yt > rich_threshold
    zero = yt <= 0
    q = np.linspace(0, 1, 101)
    return {
        "spearman": float(spearmanr(yt, yp).correlation),
        "top_ratio": float(yp[top].mean() / yt[top].mean()) if top.any() else float("nan"),
        "low_over": float(yp[zero].mean() / max(yt[zero].mean(), 1e-9)) if zero.any() else float("nan"),
        "near_zero_pred": float(np.mean(yp < 1e-4)),
        "near_zero_true": float(np.mean(zero)),
        "marginal_l1": float(np.mean(np.abs(np.quantile(yt, q) - np.quantile(yp, q)))),
    }


# ============================================================================
# LOIO-honest application
# ============================================================================


def loio_calibrate(df: pd.DataFrame, fit_apply, *, group_col: str = "obs_id",
                   pred_col: str = "y_pred", true_col: str = "y_true") -> np.ndarray:
    """Apply a calibrator leave-one-image-out: fit on the other images, score the held-out.

    `fit_apply(ref_pred, ref_true, held_pred) -> held_pred_calibrated`. Returns the
    calibrated prediction aligned to `df`'s row order. This is the deployment-honest
    protocol — the calibrator never sees the image it is scoring.
    """
    out = np.full(len(df), np.nan, dtype=np.float64)
    groups = df[group_col].to_numpy()
    pred = df[pred_col].to_numpy(dtype=np.float64)
    true = df[true_col].to_numpy(dtype=np.float64)
    for g in np.unique(groups):
        held = groups == g
        ref = ~held
        out[held] = fit_apply(pred[ref], true[ref], pred[held])
    return out


# ============================================================================
# Deployment calibration layer (Stage 1)
# ============================================================================


class CalibrationLayer:
    """The deployed, rank-preserving calibration that sits AFTER the frozen Tier-1 head.

    Bundles two fitted monotone maps (PLAN_Calibration Stage 1):

    - **Tier-1** ``calibrate_prob``: isotonic ``P(rich) → calibrated probability``
      (ECE 0.060 → 0.014; AUC-exact at deployment).
    - **Tier-2** ``calibrate_abundance``: **global** quantile-match
      ``input → fractional_area`` marginal. In the **one-model** default the input is
      the *same* ``P(rich)`` (no separate Tier-2 head); for a two-model deployment it is
      a dedicated regressor's output.

    Both maps are monotone (ranking invariant) and **global** (a fixed pointwise
    function), so they never reopen the freeze and only mis-scale where the head itself
    is fooled by out-of-distribution texture — an off-cohort / global-map concern handled
    later by the (deferred) novelty hook, not here. Fit **deployment-honest** on the
    pooled LOIO predictions of all labelled images (``from_loio_predictions``); the LOIO
    scorecard is the conservative bound the deployed layer inherits, since off-HiRISE
    terrain has no truth. Serialized as a single ``.npz`` of interpolation knots (no
    pickle): ``np.interp`` on the knots reproduces either map exactly.
    """

    def __init__(self, t1_knots=None, t2_knots=None, meta: dict | None = None) -> None:
        self._t1 = t1_knots   # (x, y) for isotonic Tier-1; None until fit/load
        self._t2 = t2_knots   # (x, y) for quantile-match Tier-2
        self.meta = meta or {}

    @classmethod
    def fit(cls, p_rich, y_binary, abundance_input, y_fractional_area,
            *, meta: dict | None = None) -> "CalibrationLayer":
        """Fit both maps. ``abundance_input`` is ``p_rich`` for the one-model default,
        or a dedicated regressor's prediction for a two-model deployment."""
        t1 = IsotonicCalibrator().fit(p_rich, y_binary).knots()
        t2 = QuantileMatcher().fit(abundance_input, y_fractional_area).knots()
        return cls(t1, t2, {"n": int(np.size(p_rich)), **(meta or {})})

    @classmethod
    def from_loio_predictions(cls, df: pd.DataFrame, *, p_rich_col="p_rich",
                              y_binary_col="y_binary", fa_col="fractional_area",
                              abundance_col: str | None = None,
                              meta: dict | None = None) -> "CalibrationLayer":
        """Fit from a pooled LOIO predictions table. One-model unless ``abundance_col``
        (a dedicated regressor's prediction) is given."""
        ab = df[p_rich_col] if abundance_col is None else df[abundance_col]
        return cls.fit(df[p_rich_col].to_numpy(), df[y_binary_col].to_numpy(),
                       ab.to_numpy(), df[fa_col].to_numpy(),
                       meta={"abundance_source": abundance_col or p_rich_col, **(meta or {})})

    def _require_fit(self):
        if self._t1 is None or self._t2 is None:
            raise RuntimeError("CalibrationLayer used before fit/load")

    def calibrate_prob(self, p_rich) -> np.ndarray:
        """Tier-1 rich/poor product: raw ``P(rich)`` → calibrated probability."""
        self._require_fit()
        return np.interp(np.asarray(p_rich, dtype=np.float64), self._t1[0], self._t1[1])

    def calibrate_abundance(self, abundance_input) -> np.ndarray:
        """Tier-2 abundance product: input → ``fractional_area``. Pass raw ``P(rich)``
        in the one-model default (the same scores feeding ``calibrate_prob``)."""
        self._require_fit()
        return np.interp(np.asarray(abundance_input, dtype=np.float64), self._t2[0], self._t2[1])

    def save(self, path: str | Path) -> None:
        self._require_fit()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, t1_x=self._t1[0], t1_y=self._t1[1], t2_x=self._t2[0], t2_y=self._t2[1],
                 meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationLayer":
        path = Path(path)
        if path.suffix != ".npz" and not path.exists():
            path = path.with_suffix(".npz")
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        return cls((d["t1_x"], d["t1_y"]), (d["t2_x"], d["t2_y"]), meta)
