"""W1 next-bet 1 sweep: per-image feature standardization on the banked recipe.

Runs lightgbm_two_stage_balanced x boulder_count @ S=64, full-v2 LOIO, with
X standardized per image (loaders.standardize_fold_per_image) for each method
in {rank, zscore, robust}. The raw-feature baseline is the banked sweep
(models/_sweep_w0/20260611T054855Z) -- not re-run here.

Promotion criteria (declared in advance, DECISIONS.md pending entry):
paired Wilcoxon over 38 folds vs the banked cell must show median
delta(meaningful_auc) > 0 with p < 0.05 AND pooled PR-AUC delta >= -0.01.
Mechanistic check: the distribution_shift dossier images should gain.

Usage:
    conda run --no-capture-output -n geospatial python -u \
        scripts/probes/_sweep_perimage_std.py [--methods rank zscore robust]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy

import pandas as pd

from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, VARIANT_CONSTRUCTORS, snapshot_params
from src.modeling.loaders import (
    augment_fold_with_per_image,
    iter_loio_folds,
    standardize_fold_per_image,
)

VARIANT = "lightgbm_two_stage_balanced"
TARGET = "boulder_count"
SCALE_IDX = 3
TILE_PX = 64
SCHEME = "loio_nfold"
DATASET_DIR = "dataset_v2"
MEANINGFUL_THRESHOLD = 50.0
MODELS_ROOT = REPO_ROOT / "models"


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_one(method: str, params: LGBMParams) -> tuple[list[dict], dict, Path]:
    snapshot = {
        "variant": VARIANT,
        "task": "regression",
        "target_col": TARGET,
        "scheme": SCHEME,
        "dataset_dir": DATASET_DIR,
        "scale_idx": SCALE_IDX,
        "tile_size_px": TILE_PX,
        "model": snapshot_params(VARIANT, params),
        "meaningful_threshold": MEANINGFUL_THRESHOLD,
        "per_image_standardization": method,
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash
    out_dir = MODELS_ROOT / VARIANT / cfg_hash / f"scale_S{TILE_PX}_target_{TARGET}_pistd_{method}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls = VARIANT_CONSTRUCTORS[VARIANT]

    def factory():
        return cls(params=params)

    def fold_iter():
        for fold in iter_loio_folds(SCHEME, scale_idx=SCALE_IDX, dataset_dir=DATASET_DIR):
            if method.startswith("aug_"):
                # raw + standardized concatenated (width doubles); GBM picks per split
                yield augment_fold_with_per_image(fold, method.removeprefix("aug_"))
            else:
                yield standardize_fold_per_image(fold, method)

    import src.modeling.evaluate as _ev
    orig = _ev.per_fold_metrics

    def _patched(y_true, y_pred, *, held_out_obs_ids, meaningful_threshold=MEANINGFUL_THRESHOLD):
        return orig(y_true, y_pred, held_out_obs_ids=held_out_obs_ids,
                    meaningful_threshold=meaningful_threshold)

    _ev.per_fold_metrics = _patched
    try:
        result = run_loio(
            factory,
            target_col=TARGET,
            task="regression",
            scheme=SCHEME,
            scale_idx=SCALE_IDX,
            dataset_dir=DATASET_DIR,
            fold_iter=fold_iter,
            snapshot=snapshot,
            verbose=False,
        )
    finally:
        _ev.per_fold_metrics = orig
    write_run_artifacts(result, out_dir)
    return result.per_fold_metrics, result.aggregate, out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["rank", "zscore", "robust"])
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    args = ap.parse_args()

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_dir = MODELS_ROOT / "_sweep_perimage_std" / ts
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"per-image standardization sweep: {args.methods} on {VARIANT} x {TARGET} @ S={TILE_PX}")
    print(f"Output: {sweep_dir}")

    fold_rows, agg_rows = [], []
    for i, method in enumerate(args.methods, 1):
        print(f"[{i}/{len(args.methods)}] pistd={method} ...", flush=True)
        per_fold, aggregate, art = run_one(method, params)
        for f in per_fold:
            row = {k: v for k, v in f.items() if not isinstance(v, (list, dict))}
            row["held_out_obs_id"] = f["held_out_obs_ids"][0]
            row["pistd"] = method
            fold_rows.append(row)
        agg_rows.append({"pistd": method, **aggregate, "artifact_dir": str(art.relative_to(REPO_ROOT))})
        print(f"        rho={aggregate.get('spearman_rho_mean', float('nan')):+.4f}  "
              f"meaningful_auc={aggregate.get('meaningful_auc_mean', float('nan')):.4f}  "
              f"PR-AUC={aggregate.get('pr_auc_mean', float('nan')):.4f}  "
              f"prec@5%={aggregate.get('precision_at_top_5pct_mean', float('nan')):.4f}", flush=True)

    pd.DataFrame(fold_rows).to_parquet(sweep_dir / "summary.parquet", index=False)
    pd.DataFrame(agg_rows).to_parquet(sweep_dir / "aggregate.parquet", index=False)
    (sweep_dir / "sweep_meta.json").write_text(json.dumps({
        "timestamp": ts, "methods": args.methods, "variant": VARIANT, "target": TARGET,
        "scale_idx": SCALE_IDX, "scheme": SCHEME, "dataset_dir": DATASET_DIR,
        "baseline_sweep": "models/_sweep_w0/20260611T054855Z",
    }, indent=2))
    print(f"wrote {sweep_dir}\\summary.parquet + aggregate.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
