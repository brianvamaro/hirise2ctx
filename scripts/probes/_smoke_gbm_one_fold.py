"""Smoke-test one fold x one GBM variant end-to-end via the LOIO harness."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.evaluate import per_fold_metrics
from src.modeling.gbm import LGBMParams, LightGBMTweedie, LightGBMLog1pHuber, LightGBMTwoStage
from src.modeling.loaders import load_fold


def smoke(model_cls, label: str) -> None:
    f = load_fold("loio_9fold", 0, scale_idx=0)
    y_train = f.y_train["fractional_area"].to_numpy()
    y_test = f.y_test["fractional_area"].to_numpy()
    print(f"\n=== {label} on fold 0 (test={f.held_out_obs_ids}, scale_idx=0) ===")
    print(f"  X_train: {f.X_train.shape}  pos_frac_train={float((y_train > 0).mean()):.4f}")
    print(f"  X_test : {f.X_test.shape}   pos_frac_test ={float((y_test > 0).mean()):.4f}")

    model = model_cls(params=LGBMParams(n_estimators=200, learning_rate=0.05, early_stopping_rounds=30))
    model.fit(f.X_train, y_train, eval_set=(f.X_test, y_test))
    y_pred = model.predict(f.X_test)
    print(f"  y_pred range: [{y_pred.min():.3g}, {y_pred.max():.3g}]  mean={y_pred.mean():.3g}")
    m = per_fold_metrics(y_test, y_pred, held_out_obs_ids=f.held_out_obs_ids)
    print(f"  spearman={m['spearman_rho']:+.4f}  rmse_log1p={m['rmse_log1p']:.4g}  auc={m['presence_auc']:.3f}")
    print(f"  model_hash={model.model_hash()[:16]}...")


def main() -> int:
    smoke(LightGBMTweedie, "Tweedie")
    smoke(LightGBMLog1pHuber, "log1p+Huber")
    smoke(LightGBMTwoStage, "Two-stage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
