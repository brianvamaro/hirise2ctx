"""LightGBM baselines for the zero-inflated rock-abundance target.

Three variants ship in parallel (PLAN_modeling.md §2, decision locked
AskUserQuestion 2026-05-27):

  * LightGBMTweedie       -- single-stage GBM with `objective='tweedie'`.
                             Compound Poisson-Gamma likelihood, the canonical
                             zero-inflated continuous loss. tweedie_variance_power
                             in (1, 2); default 1.5.
  * LightGBMLog1pHuber    -- single-stage GBM on log1p-stabilised target with
                             Huber loss (`objective='huber'`). Variance-stabilising
                             baseline that should track Tweedie roughly.
  * LightGBMTwoStage      -- composes a presence classifier with `fractional_area
                             > 0` as the positive rule (probe 2026-05-27) and a
                             magnitude regressor (log1p+Huber, positives-only).
                             Prediction: P(positive) * E[mag | positive].

All three implement the `Model` Protocol from `src.modeling.base`. `predict` always
returns predictions on the original `fractional_area` scale; transforms are applied
internally.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.modeling.base import Model, hash_bytes

# Strict-presence positive rule per probe 2026-05-27.
# Used by LightGBMTwoStage to decompose the target into classification + regression.
POSITIVE_RULE_EPS = 0.0  # `fractional_area > POSITIVE_RULE_EPS`


# ============================================================================
# Hyperparameter container
# ============================================================================


@dataclass
class LGBMParams:
    """Hyperparameters shared by every LightGBM variant.

    Defaults are PLAN_modeling.md §2 "small coarse grid" starting point; tuning
    happens later via the sweep harness.
    """

    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 64
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 5
    early_stopping_rounds: int = 50
    seed: int = 0
    verbose: int = -1

    # Variant-specific knobs
    tweedie_variance_power: float = 1.5  # for objective='tweedie'
    huber_alpha: float = 0.9             # for objective='huber'
    class_weight_negative: float = 1.0   # for two-stage classifier (positives boost)

    def to_lgb_kwargs(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "seed": self.seed,
            "bagging_seed": self.seed,
            "feature_fraction_seed": self.seed,
            "data_random_seed": self.seed,
            "deterministic": True,
            "verbose": self.verbose,
        }


def _booster_hash(booster: lgb.Booster | None) -> str:
    if booster is None:
        return ""
    txt = booster.model_to_string()
    return hash_bytes(txt.encode("utf-8"))


# ============================================================================
# Single-stage variants
# ============================================================================


@dataclass
class LightGBMTweedie:
    """Single-stage LightGBM with Tweedie objective."""

    params: LGBMParams = field(default_factory=LGBMParams)
    name: str = "lightgbm_tweedie"
    _booster: lgb.Booster | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,  # not used; kept for Protocol uniformity
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        kw = self.params.to_lgb_kwargs()
        kw["objective"] = "tweedie"
        kw["tweedie_variance_power"] = self.params.tweedie_variance_power
        # Tweedie requires y >= 0
        y_pos = np.clip(y, 0.0, None)
        train_set = lgb.Dataset(X, label=y_pos, free_raw_data=False)
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = []
        if eval_set is not None:
            Xv, yv = eval_set
            yv_pos = np.clip(yv, 0.0, None)
            valid_sets.append(lgb.Dataset(Xv, label=yv_pos, reference=train_set, free_raw_data=False))
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self.params.early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(0))
        self._booster = lgb.train(
            kw,
            train_set,
            num_boost_round=self.params.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._booster is not None, "fit() before predict()"
        # Tweedie objective with log link -> booster.predict returns mean of the response
        # on the original scale already.
        return np.asarray(self._booster.predict(X, num_iteration=self._booster.best_iteration))

    def predict_presence_prob(self, X: np.ndarray) -> np.ndarray | None:
        return None

    def save(self, path: str | Path) -> None:
        assert self._booster is not None
        Path(path).write_text(self._booster.model_to_string(), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self._booster = lgb.Booster(model_str=Path(path).read_text(encoding="utf-8"))

    def model_hash(self) -> str:
        return _booster_hash(self._booster)


@dataclass
class LightGBMLog1pHuber:
    """Single-stage LightGBM, trained on log1p(target) with Huber loss."""

    params: LGBMParams = field(default_factory=LGBMParams)
    name: str = "lightgbm_log1p_huber"
    _booster: lgb.Booster | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        kw = self.params.to_lgb_kwargs()
        kw["objective"] = "huber"
        kw["alpha"] = self.params.huber_alpha
        y_log = np.log1p(np.clip(y, 0.0, None))
        train_set = lgb.Dataset(X, label=y_log, free_raw_data=False)
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = []
        if eval_set is not None:
            Xv, yv = eval_set
            yv_log = np.log1p(np.clip(yv, 0.0, None))
            valid_sets.append(lgb.Dataset(Xv, label=yv_log, reference=train_set, free_raw_data=False))
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self.params.early_stopping_rounds, verbose=False))
        callbacks.append(lgb.log_evaluation(0))
        self._booster = lgb.train(
            kw,
            train_set,
            num_boost_round=self.params.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._booster is not None
        log_pred = np.asarray(self._booster.predict(X, num_iteration=self._booster.best_iteration))
        # Back-transform log1p -> linear; clip negatives to zero (Huber can go below 0).
        return np.clip(np.expm1(log_pred), 0.0, None)

    def predict_presence_prob(self, X: np.ndarray) -> np.ndarray | None:
        return None

    def save(self, path: str | Path) -> None:
        assert self._booster is not None
        Path(path).write_text(self._booster.model_to_string(), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        self._booster = lgb.Booster(model_str=Path(path).read_text(encoding="utf-8"))

    def model_hash(self) -> str:
        return _booster_hash(self._booster)


# ============================================================================
# Two-stage (hurdle) variant
# ============================================================================


@dataclass
class LightGBMTwoStage:
    """Two-stage hurdle: presence classifier x magnitude regressor.

    Positive rule: `fractional_area > 0` (probe 2026-05-27 -- the rule with the
    highest cross-image consistency between area- and count-based definitions).
    Magnitude head is `log1p+huber` trained ONLY on positive tiles.

    Prediction: `P(positive) * E[mag | positive]`. The presence probability is
    exposed via `predict_presence_prob` so the evaluator can persist it.
    """

    params: LGBMParams = field(default_factory=LGBMParams)
    name: str = "lightgbm_two_stage"
    _presence: lgb.Booster | None = field(default=None, init=False, repr=False)
    _magnitude: lgb.Booster | None = field(default=None, init=False, repr=False)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> None:
        # ---- Presence head: binary classification on (y > 0) ----
        y_bin = (y > POSITIVE_RULE_EPS).astype(np.int8)
        kw_cls = self.params.to_lgb_kwargs()
        kw_cls["objective"] = "binary"
        # Use is_unbalance so we don't have to hand-tune scale_pos_weight; LightGBM
        # internally weights to match positive prior. Cheap, well-tested.
        kw_cls["is_unbalance"] = True
        train_cls = lgb.Dataset(X, label=y_bin, free_raw_data=False)
        valid_sets = [train_cls]
        valid_names = ["train"]
        callbacks = [lgb.log_evaluation(0)]
        if eval_set is not None:
            Xv, yv = eval_set
            yv_bin = (yv > POSITIVE_RULE_EPS).astype(np.int8)
            # Only add early-stopping if test set actually has both classes (the
            # empty-truth fold sometimes hands us all-zero test sets).
            if np.unique(yv_bin).size > 1:
                valid_sets.append(lgb.Dataset(Xv, label=yv_bin, reference=train_cls, free_raw_data=False))
                valid_names.append("valid")
                callbacks.insert(0, lgb.early_stopping(self.params.early_stopping_rounds, verbose=False))
        self._presence = lgb.train(
            kw_cls,
            train_cls,
            num_boost_round=self.params.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        # ---- Magnitude head: log1p+Huber on POSITIVES only ----
        pos_mask = y_bin.astype(bool)
        X_pos = X[pos_mask]
        y_pos = y[pos_mask]
        if y_pos.size < 10:
            # Pathological: too few positives to fit a magnitude model. Skip; predict
            # falls back to mean-positive constant.
            self._magnitude = None
            return
        y_log = np.log1p(y_pos)
        kw_mag = self.params.to_lgb_kwargs()
        kw_mag["objective"] = "huber"
        kw_mag["alpha"] = self.params.huber_alpha
        train_mag = lgb.Dataset(X_pos, label=y_log, free_raw_data=False)
        valid_sets_m = [train_mag]
        valid_names_m = ["train"]
        callbacks_m = [lgb.log_evaluation(0)]
        if eval_set is not None:
            Xv, yv = eval_set
            yv_pos_mask = yv > POSITIVE_RULE_EPS
            if yv_pos_mask.sum() >= 10:
                valid_sets_m.append(
                    lgb.Dataset(
                        Xv[yv_pos_mask], label=np.log1p(yv[yv_pos_mask]),
                        reference=train_mag, free_raw_data=False,
                    )
                )
                valid_names_m.append("valid")
                callbacks_m.insert(0, lgb.early_stopping(self.params.early_stopping_rounds, verbose=False))
        self._magnitude = lgb.train(
            kw_mag,
            train_mag,
            num_boost_round=self.params.n_estimators,
            valid_sets=valid_sets_m,
            valid_names=valid_names_m,
            callbacks=callbacks_m,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self._presence is not None, "fit() before predict()"
        p_pos = np.asarray(self._presence.predict(X, num_iteration=self._presence.best_iteration))
        if self._magnitude is not None:
            mag_log = np.asarray(self._magnitude.predict(X, num_iteration=self._magnitude.best_iteration))
            mag = np.clip(np.expm1(mag_log), 0.0, None)
        else:
            mag = np.zeros_like(p_pos)
        return p_pos * mag

    def predict_presence_prob(self, X: np.ndarray) -> np.ndarray | None:
        assert self._presence is not None
        return np.asarray(self._presence.predict(X, num_iteration=self._presence.best_iteration))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        assert self._presence is not None
        (path / "presence.txt").write_text(self._presence.model_to_string(), encoding="utf-8")
        if self._magnitude is not None:
            (path / "magnitude.txt").write_text(self._magnitude.model_to_string(), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        self._presence = lgb.Booster(model_str=(path / "presence.txt").read_text(encoding="utf-8"))
        mag_path = path / "magnitude.txt"
        if mag_path.exists():
            self._magnitude = lgb.Booster(model_str=mag_path.read_text(encoding="utf-8"))
        else:
            self._magnitude = None

    def model_hash(self) -> str:
        h_pres = _booster_hash(self._presence)
        h_mag = _booster_hash(self._magnitude)
        return hashlib.sha256(f"{h_pres}|{h_mag}".encode("utf-8")).hexdigest()


# ============================================================================
# Factory helpers used by the training scripts
# ============================================================================


VARIANT_CONSTRUCTORS = {
    "lightgbm_tweedie": LightGBMTweedie,
    "lightgbm_log1p_huber": LightGBMLog1pHuber,
    "lightgbm_two_stage": LightGBMTwoStage,
}


def make_factory(variant: str, params: LGBMParams | None = None):
    cls = VARIANT_CONSTRUCTORS[variant]
    p = params or LGBMParams()

    def _f():
        return cls(params=p)

    return _f


def snapshot_params(variant: str, params: LGBMParams) -> dict:
    """Provenance dict for snapshot.json -- captures the model class + hyperparameters."""
    return {
        "variant": variant,
        "params": asdict(params),
        "positive_rule_eps": POSITIVE_RULE_EPS,
    }
