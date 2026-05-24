"""Run the full 9-fold LOIO loop for one GBM variant to verify the harness end-to-end."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.evaluate import run_loio
from src.modeling.gbm import LGBMParams, make_factory


def main() -> int:
    factory = make_factory(
        "lightgbm_tweedie",
        params=LGBMParams(n_estimators=300, learning_rate=0.05, early_stopping_rounds=30),
    )
    print("Running LOIO 9-fold sweep, scale_idx=0, lightgbm_tweedie ...\n")
    result = run_loio(factory, target_col="fractional_area", scheme="loio_9fold", scale_idx=0)
    print("\nAggregate:")
    for k, v in result.aggregate.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
