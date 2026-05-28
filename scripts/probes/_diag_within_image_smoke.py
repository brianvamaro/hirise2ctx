"""One-off smoke test: run lightgbm_classification at scale_idx=3 on within_image_4fold.

Confirms the existing run_loio harness accepts the new scheme without any harness-side
changes. Prints per-fold AUC for the first 4 folds (one image's 4 quadrants), then exits.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401 -- DLL bootstrap; must precede numpy

from src.modeling.binary_target import get_target
from src.modeling.evaluate import run_loio
from src.modeling.gbm import LGBMParams, LightGBMClassification


def main() -> int:
    target = get_target("bc_ge_1")
    params = LGBMParams(n_estimators=400, learning_rate=0.05, early_stopping_rounds=40)

    def factory() -> LightGBMClassification:
        return LightGBMClassification(params=params)

    print("Smoke: lightgbm_classification @ scale_idx=3 (S=64) on within_image_4fold ...", flush=True)
    result = run_loio(
        factory,
        binarize=target.binarize,
        task="classification",
        scheme="within_image_4fold",
        scale_idx=3,
        snapshot={"smoke": True, "variant": "lightgbm_classification", "target": "bc_ge_1"},
        verbose=True,
    )
    agg = result.aggregate
    print(
        f"\nSmoke OK. auc={agg['auc_mean']:+.4f} +/- {agg['auc_std']:.4f}  "
        f"n_real_folds={agg['n_real_folds']}  n_spec={agg['n_specificity_folds']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
