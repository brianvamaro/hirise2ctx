"""Train one GBM variant on one scale via the LOIO harness.

Usage:
    python scripts/train_gbm.py {lightgbm_tweedie|lightgbm_log1p_huber|lightgbm_two_stage} \\
        --scale-idx 0 \\
        [--scheme loio_9fold] [--n-estimators 500] [--learning-rate 0.05] ...

Writes:
    models/{variant}/{config_hash}/scale_S{tile_size}/
      predictions.parquet, metrics.json, snapshot.json
      fold_{obs_id}/booster.txt   (or presence.txt + magnitude.txt for two-stage)

The default hyperparameters live in `LGBMParams`; CLI flags override individual fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, VARIANT_CONSTRUCTORS, make_factory, snapshot_params
from src.modeling.loaders import iter_loio_folds

MODELS_ROOT = REPO_ROOT / "models"
TILE_SIZE_FOR_SCALE = {0: 8, 1: 16, 2: 32, 3: 64}


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("variant", choices=list(VARIANT_CONSTRUCTORS))
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
    ap.add_argument("--tweedie-variance-power", type=float, default=1.5)
    ap.add_argument("--huber-alpha", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-col", default="fractional_area")
    args = ap.parse_args()

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        min_data_in_leaf=args.min_data_in_leaf,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        early_stopping_rounds=args.early_stopping_rounds,
        seed=args.seed,
        tweedie_variance_power=args.tweedie_variance_power,
        huber_alpha=args.huber_alpha,
    )
    factory = make_factory(args.variant, params)

    snapshot = {
        "variant": args.variant,
        "target_col": args.target_col,
        "scheme": args.scheme,
        "scale_idx": args.scale_idx,
        "tile_size_px": TILE_SIZE_FOR_SCALE.get(args.scale_idx),
        "model": snapshot_params(args.variant, params),
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash

    out_dir = MODELS_ROOT / args.variant / cfg_hash / f"scale_S{TILE_SIZE_FOR_SCALE[args.scale_idx]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  out_dir = {out_dir}")

    # We re-iterate inside `run_loio`; persisting per-fold model artifacts requires
    # an extra pass with the same factory. Cheap (LightGBM fits are seconds) and the
    # alternative -- making `run_loio` aware of disk layout -- couples it tightly.
    # Run the eval pass first; persist booster files in a follow-up pass.
    print(f"\n[1/2] LOIO eval pass: {args.variant} @ scale_idx={args.scale_idx}")
    result = run_loio(
        factory,
        target_col=args.target_col,
        scheme=args.scheme,
        scale_idx=args.scale_idx,
        snapshot=snapshot,
    )
    paths = write_run_artifacts(result, out_dir)
    print(f"\n  wrote: predictions.parquet={paths['predictions']}")
    print(f"         metrics.json={paths['metrics']}")
    print(f"         snapshot.json={paths['snapshot']}")

    print(f"\n[2/2] Persist per-fold booster artifacts")
    # Persist booster artifacts. Refit per fold; cheap and gives us deterministic
    # save_paths that map 1:1 to the prediction parquet's `fold_held_out_obs_id`.
    for fold in iter_loio_folds(args.scheme, scale_idx=args.scale_idx):
        # Same inner-validation rotation as the harness so models are identical.
        import numpy as np

        train_codes = fold.groups_train
        inner_val_code = int(np.unique(train_codes)[fold.fold_idx % np.unique(train_codes).size])
        inner_val_mask = train_codes == inner_val_code
        inner_train_mask = ~inner_val_mask
        y_train_full = fold.y_train[args.target_col].to_numpy(dtype=np.float64)
        model = factory()
        model.fit(
            fold.X_train[inner_train_mask],
            y_train_full[inner_train_mask],
            groups=train_codes[inner_train_mask],
            eval_set=(
                fold.X_train[inner_val_mask],
                y_train_full[inner_val_mask],
            ),
        )
        held = fold.held_out_obs_ids[0] if fold.held_out_obs_ids else f"fold{fold.fold_idx}"
        fold_out = out_dir / f"fold_{held}"
        fold_out.mkdir(parents=True, exist_ok=True)
        # Single-booster models save to a file; two-stage saves to a directory.
        if args.variant == "lightgbm_two_stage":
            model.save(fold_out)
        else:
            model.save(fold_out / "booster.txt")
        print(f"  saved {fold_out}")

    print(f"\nDONE: scale_idx={args.scale_idx} variant={args.variant}")
    print(f"  Spearman rho (mean +/- std): {result.aggregate['spearman_rho_mean']:+.4f} +/- {result.aggregate['spearman_rho_std']:.4f}  (n={result.aggregate['spearman_n']} real folds)")
    print(f"  out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
