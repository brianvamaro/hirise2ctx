"""Train one binary classifier on one (target, scale) via the LOIO harness.

Usage:
    python scripts/train_binary.py {bc_ge_1|fa_gt_1e-3|fa_gt_1e-2} \\
        --scale-idx 0 \\
        [--scheme loio_9fold] [--n-estimators 500] [--learning-rate 0.05] ...

Writes:
    models/lightgbm_classification/{config_hash}/scale_S{tile_size}_t{target_id}/
      predictions.parquet, metrics.json, snapshot.json
      fold_{obs_id}/classifier.txt

Same `config_hash` mechanism as scripts/train_gbm.py / scripts/sweep.py -- same
params + target + scale produce the same hash, so train_binary and sweep_binary
write to identical paths and are interchangeable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy

import numpy as np

from src.modeling.binary_target import BINARY_TARGETS_BY_ID, get_target
from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, LightGBMClassification, snapshot_params
from src.modeling.loaders import iter_loio_folds

MODELS_ROOT = REPO_ROOT / "models"
TILE_SIZE_FOR_SCALE = {0: 8, 1: 16, 2: 32, 3: 64}


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target_id", choices=list(BINARY_TARGETS_BY_ID),
                    help="Binary target spec id (see src.modeling.binary_target)")
    ap.add_argument("--scale-idx", type=int, required=True,
                    help="0/1/2/3 for tile_size 8/16/32/64 px (40/80/160/320 m)")
    ap.add_argument("--scheme", default="loio_9fold")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--min-data-in-leaf", type=int, default=64)
    ap.add_argument("--feature-fraction", type=float, default=0.9)
    ap.add_argument("--bagging-fraction", type=float, default=0.9)
    ap.add_argument("--early-stopping-rounds", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    target = get_target(args.target_id)
    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
    )

    def factory() -> LightGBMClassification:
        return LightGBMClassification(params=params)

    tile_size = TILE_SIZE_FOR_SCALE[args.scale_idx]
    snapshot = {
        "variant": "lightgbm_classification",
        "task": "classification",
        "target_id": target.id,
        "target_source_col": target.source_col,
        "target_threshold": target.threshold,
        "target_comparison": target.comparison,
        "scheme": args.scheme,
        "scale_idx": args.scale_idx,
        "tile_size_px": tile_size,
        "model": snapshot_params("lightgbm_classification", params),
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash

    out_dir = (
        MODELS_ROOT / "lightgbm_classification" / cfg_hash
        / f"scale_S{tile_size}_t{target.id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir = {out_dir}")

    print(f"\n[1/2] LOIO eval pass: classification target={target.id} scale_idx={args.scale_idx}")
    result = run_loio(
        factory,
        binarize=target.binarize,
        task="classification",
        scheme=args.scheme,
        scale_idx=args.scale_idx,
        snapshot=snapshot,
    )
    paths = write_run_artifacts(result, out_dir)
    print(f"\n  wrote: predictions.parquet={paths['predictions']}")
    print(f"         metrics.json={paths['metrics']}")
    print(f"         snapshot.json={paths['snapshot']}")

    print(f"\n[2/2] Persist per-fold classifier artifacts")
    # Refit per fold with the same inner-validation rotation so saved boosters
    # match the predictions in predictions.parquet.
    for fold in iter_loio_folds(args.scheme, scale_idx=args.scale_idx):
        train_codes = fold.groups_train
        unique_train = np.unique(train_codes)
        inner_val_code = int(unique_train[fold.fold_idx % unique_train.size])
        inner_val_mask = train_codes == inner_val_code
        inner_train_mask = ~inner_val_mask
        y_train_full = target.binarize(fold.y_train)
        model = factory()
        model.fit(
            fold.X_train[inner_train_mask],
            y_train_full[inner_train_mask],
            groups=train_codes[inner_train_mask],
            eval_set=(fold.X_train[inner_val_mask], y_train_full[inner_val_mask]),
        )
        held = fold.held_out_obs_ids[0] if fold.held_out_obs_ids else f"fold{fold.fold_idx}"
        fold_out = out_dir / f"fold_{held}"
        fold_out.mkdir(parents=True, exist_ok=True)
        model.save(fold_out / "classifier.txt")
        print(f"  saved {fold_out}")

    print(f"\nDONE: target={target.id} scale_idx={args.scale_idx}")
    print(
        f"  AUC (mean +/- std):    {result.aggregate['auc_mean']:+.4f} +/- "
        f"{result.aggregate['auc_std']:.4f}  (n={result.aggregate['auc_n']} real folds)"
    )
    print(f"  Brier (mean):         {result.aggregate['brier_mean']:.4g}")
    print(f"  ECE (mean):           {result.aggregate['ece_mean']:.4f}")
    print(f"  Lift@top-k (mean):    {result.aggregate['lift_at_top_k_mean']:.3f}")
    print(f"  out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
