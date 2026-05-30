"""Smoke-test the new two-stage compression-fix variants on small synthetic data.

Verifies each variant constructs, fits, and predicts. Catches sklearn-API /
LightGBM-option typos before the dev sweep wastes compute.
"""

from __future__ import annotations

import numpy as np

import src.modeling  # noqa: F401 (Windows DLL bootstrap)

from src.modeling.gbm import (
    LGBMParams,
    LightGBMTwoStage,
    LightGBMTwoStageBalanced,
    LightGBMTwoStageCombined,
    LightGBMTwoStageGamma,
    LightGBMTwoStageWeighted,
    VARIANT_CONSTRUCTORS,
)


def synth(n: int = 600, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 8))
    # Zero-inflated heavy-tail target: ~70% zeros, rest log-normal with a tail
    z = (rng.random(n) < 0.7).astype(float)
    base = np.exp(rng.standard_normal(n) - 6.0)          # log-normal centred ~ exp(-6) ~ 2.5e-3
    y = np.where(z == 1, 0.0, base + 1e-4 * (X[:, 0] ** 2))
    return X.astype(np.float64), y.astype(np.float64)


def main() -> None:
    X, y = synth(800, seed=0)
    params = LGBMParams(n_estimators=50, learning_rate=0.1, early_stopping_rounds=20, verbose=-1)
    Xv, yv = synth(200, seed=1)

    for name in (
        "lightgbm_two_stage",
        "lightgbm_two_stage_balanced",
        "lightgbm_two_stage_weighted",
        "lightgbm_two_stage_gamma",
        "lightgbm_two_stage_combined",
    ):
        cls = VARIANT_CONSTRUCTORS[name]
        m = cls(params=params)
        m.fit(X, y, eval_set=(Xv, yv))
        p = m.predict(Xv)
        ppos = m.predict_presence_prob(Xv)
        print(f"  {name:<35s}  pred[min..max]={p.min():.5f}..{p.max():.5f}  "
              f"p_pos[mean]={ppos.mean():.3f}  ok")

    print("All variants OK.")


if __name__ == "__main__":
    main()
