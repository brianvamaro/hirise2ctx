"""Train the small CNN baseline on Stage 4b context patches via LOIO CV.

Usage:
    python scripts/train_cnn.py --patch-size-px 32 --scale-idx 2 --epochs 30
    python scripts/train_cnn.py --patch-size-px 64 --scale-idx 3 --epochs 30

The CNN consumes per-tile patches (S=32 or S=64) and a regression target. The
patch size and tile scale are usually matched (S=32 -> scale_idx=2, S=64 ->
scale_idx=3) so the CNN sees the same spatial extent the GBM does at the same
scale. Other (patch_size, scale_idx) combinations are valid but cover an asymmetric
window.

Writes:
    models/{name}/{config_hash}/scale_S{tile_size}_P{patch}/
        predictions.parquet, metrics.json, snapshot.json,
        fold_{obs_id}/state_dict.pt
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CRITICAL: import src.modeling BEFORE numpy/pandas. Its __init__ runs the Windows DLL
# bootstrap (KMP env + add_dll_directory + shm.dll preload) so torch loads cleanly
# alongside the geospatial env's MKL OpenMP. See DECISIONS.md 2026-05-27.
import src.modeling  # noqa: F401

import numpy as np
import pandas as pd

from src.modeling.cnn import CNNParams, SmallCNNRegressor, SmallCNNClassifier
from src.modeling.binary_target import get_target
from src.modeling.evaluate import (
    per_fold_metrics, aggregate_fold_metrics,
    per_fold_metrics_classification, aggregate_fold_metrics_classification,
)
from src.modeling.loaders import iter_loio_folds

MODELS_ROOT = REPO_ROOT / "models"
TILE_SIZE_FOR_SCALE = {0: 8, 1: 16, 2: 32, 3: 64, 4: 128}


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _split_inner_val(fold, y_train_full: np.ndarray):
    """Same rotation rule as src.modeling.evaluate.run_loio; y is precomputed (regression
    target or binarised label) so this serves both tasks."""
    train_codes = fold.groups_train
    unique_train = np.unique(train_codes)
    inner_val_code = int(unique_train[fold.fold_idx % unique_train.size])
    inner_val_mask = train_codes == inner_val_code
    inner_train_mask = ~inner_val_mask
    return (
        fold.keys_train[inner_train_mask].reset_index(drop=True), y_train_full[inner_train_mask],
        fold.keys_train[inner_val_mask].reset_index(drop=True), y_train_full[inner_val_mask],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch-size-px", type=int, choices=[32, 64, 128], required=True)
    ap.add_argument("--scale-idx", type=int, required=True, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--scheme", default="loio_9fold")
    ap.add_argument("--dataset-dir", default=None,
                    help="Packaged dataset root (default: ./dataset = v1). Use dataset_v2[_dev].")
    ap.add_argument("--task", choices=["regression", "classification"], default="regression")
    ap.add_argument("--target-id", default="bc_ge_1", help="Binary target id when --task classification")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--early-stopping-patience", type=int, default=6)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-col", default="fractional_area")
    args = ap.parse_args()

    params = CNNParams(
        patch_size_px=args.patch_size_px,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
    )

    is_cls = args.task == "classification"
    target = get_target(args.target_id) if is_cls else None
    tile_size = TILE_SIZE_FOR_SCALE[args.scale_idx]
    variant = (f"cnn_bce_S{args.patch_size_px}" if is_cls
               else f"cnn_log1p_huber_S{args.patch_size_px}")
    snapshot = {
        "variant": variant,
        "task": args.task,
        "target_col": args.target_col if not is_cls else None,
        "target_id": args.target_id if is_cls else None,
        "scheme": args.scheme,
        "dataset_dir": args.dataset_dir or "dataset",
        "scale_idx": args.scale_idx,
        "tile_size_px": tile_size,
        "model": {"variant": variant, "params": asdict(params)},
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash

    suffix = f"scale_S{tile_size}_P{args.patch_size_px}" + (f"_t{args.target_id}" if is_cls else "")
    out_dir = MODELS_ROOT / variant / cfg_hash / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  task={args.task}  dataset={args.dataset_dir or 'dataset'}  out_dir = {out_dir}")

    per_fold_records: list[dict] = []
    pred_rows: list[pd.DataFrame] = []

    for fold in iter_loio_folds(args.scheme, scale_idx=args.scale_idx, dataset_dir=args.dataset_dir):
        held = fold.held_out_obs_ids[0] if fold.held_out_obs_ids else f"fold{fold.fold_idx}"
        print(f"\nfold {fold.fold_idx}  held_out={held}  n_train={len(fold.keys_train)}  "
              f"n_test={len(fold.keys_test)}")

        if is_cls:
            y_train_full = target.binarize(fold.y_train).astype(np.float64)
            y_test = target.binarize(fold.y_test).astype(np.float64)
            model = SmallCNNClassifier(params=params)
        else:
            y_train_full = fold.y_train[args.target_col].to_numpy(dtype=np.float64)
            y_test = fold.y_test[args.target_col].to_numpy(dtype=np.float64)
            model = SmallCNNRegressor(params=params)

        keys_tr, y_tr, keys_vl, y_vl = _split_inner_val(fold, y_train_full)
        model.bind_train_data(keys_tr, y_tr)
        model.bind_val_data(keys_vl, y_vl)
        # X passed to fit/predict is ignored by the CNN; pass a dummy shape consistent with row count
        model.fit(np.empty((len(keys_tr), 0), dtype=np.float32), y_tr, groups=None)

        model.bind_predict_data(fold.keys_test)
        y_pred = model.predict(np.empty((len(fold.keys_test), 0), dtype=np.float32))

        if is_cls:
            m = per_fold_metrics_classification(
                y_test.astype(np.int8), y_pred, held_out_obs_ids=fold.held_out_obs_ids)
        else:
            m = per_fold_metrics(y_test, y_pred, held_out_obs_ids=fold.held_out_obs_ids)
        m["fold_idx"] = fold.fold_idx
        m["scale_idx"] = args.scale_idx
        m["model_name"] = model.name
        m["model_hash"] = model.model_hash()
        per_fold_records.append(m)

        block = fold.keys_test.copy()
        block["fold_held_out_obs_id"] = held
        block["fold_idx"] = fold.fold_idx
        block["y_true"] = y_test
        block["y_pred"] = y_pred
        block["y_pred_presence_prob"] = y_pred if is_cls else np.nan
        block["model_hash"] = model.model_hash()
        pred_rows.append(block)

        # Persist per-fold state_dict
        fold_out = out_dir / f"fold_{held}"
        fold_out.mkdir(parents=True, exist_ok=True)
        model.save(fold_out / "state_dict.pt")

        if m["is_specificity_only"]:
            print("  spec")
        elif is_cls:
            print(f"  auc={m['auc']:+.4f}  brier={m['brier']:.4g}  lift={m['lift_at_top_k']:.3f}")
        else:
            print(f"  rho={m['spearman_rho']:+.4f}  rmse_log1p={m['rmse_log1p']:.4g}  auc={m['presence_auc']:.3f}")

    aggregate = (aggregate_fold_metrics_classification(per_fold_records) if is_cls
                 else aggregate_fold_metrics(per_fold_records))
    pred_df = pd.concat(pred_rows, ignore_index=True)
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)

    (out_dir / "metrics.json").write_text(
        json.dumps({"per_fold": per_fold_records, "aggregate": aggregate}, indent=2, default=float),
        encoding="utf-8",
    )
    snapshot["written_at_iso"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    (out_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")

    print(f"\nDONE: {snapshot['variant']} scale_idx={args.scale_idx} task={args.task}")
    if is_cls:
        print(f"  AUC (mean +/- std): {aggregate['auc_mean']:+.4f} +/- {aggregate['auc_std']:.4f}  "
              f"(n={aggregate['auc_n']} real folds)")
    else:
        print(f"  Spearman rho (mean +/- std): {aggregate['spearman_rho_mean']:+.4f} +/- {aggregate['spearman_rho_std']:.4f}")
    print(f"  out: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
