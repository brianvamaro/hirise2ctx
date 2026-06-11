"""W2 Phase 1: fan the binary CNN out over augmentation cells (PLAN_CNN.md §4).

One run = one (aug_cell, patch_size/scale) over every LOIO fold with a group-aware
inner-validation pool (whole images, never tile-random) for early stopping. Mirrors
sweep_binary.py's artifact layout so the paired-probe tooling works unchanged.

Usage:
    # The 4-cell S=64 grid (one cell at a time is fine too via --cells):
    python scripts/sweep_cnn.py --dataset-dir dataset_v2 --scheme loio_nfold
    python scripts/sweep_cnn.py --dataset-dir dataset_v2 --scheme loio_nfold --cells none

    # S=32 replication (PLAN_CNN.md §4.2b -- cell A + the S=64 winner only):
    python scripts/sweep_cnn.py --dataset-dir dataset_v2 --scheme loio_nfold \
        --patch-size-px 32 --scale-idx 2 --cells none photometric

Writes:
    models/_sweep_cnn/{timestamp}/summary.parquet    # one row per (cell, fold)
    models/_sweep_cnn/{timestamp}/aggregate.parquet  # one row per cell
    models/cnn_bce_S{P}/{config_hash}/scale_S{n}_t{target}_aug_{cell}/
        predictions.parquet, metrics.json, snapshot.json
        fold_{obs_id}/state_dict.pt
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# CRITICAL: import src.modeling BEFORE numpy/pandas (Windows DLL bootstrap).
import src.modeling  # noqa: F401

import numpy as np
import pandas as pd

from src.modeling.binary_target import get_target
from src.modeling.cnn import AUG_CELLS, CNNParams, SmallCNNClassifier
from src.modeling.evaluate import (
    aggregate_fold_metrics_classification,
    per_fold_metrics_classification,
)
from src.modeling.loaders import iter_loio_folds

MODELS_ROOT = REPO_ROOT / "models"
TILE_SIZE_FOR_SCALE = {0: 8, 1: 16, 2: 32, 3: 64, 4: 128}
DEFAULT_CELLS = ("none", "geometric", "photometric", "photometric_std")


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _split_inner_val_groups(fold, y_train_full: np.ndarray, n_val_images: int):
    """Hold out `n_val_images` whole images from the training pool for early stopping.

    Deterministic rotation over the sorted unique group codes so each fold sees a
    different (but reproducible) validation pool; never tile-random (PLAN_CNN.md §4.1).
    """
    train_codes = fold.groups_train
    unique_train = np.unique(train_codes)
    n = unique_train.size
    k = min(n_val_images, max(1, n - 1))
    val_codes = {int(unique_train[(fold.fold_idx * k + i) % n]) for i in range(k)}
    inner_val_mask = np.isin(train_codes, sorted(val_codes))
    inner_train_mask = ~inner_val_mask
    return (
        fold.keys_train[inner_train_mask].reset_index(drop=True), y_train_full[inner_train_mask],
        fold.keys_train[inner_val_mask].reset_index(drop=True), y_train_full[inner_val_mask],
    )


def run_one_cell(
    cell: str,
    *,
    target_id: str,
    scale_idx: int,
    patch_size_px: int,
    scheme: str,
    dataset_dir: str | None,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    early_stopping_patience: int,
    seed: int,
    inner_val_images: int,
) -> tuple[list[dict], dict, Path]:
    target = get_target(target_id)
    tile_size = TILE_SIZE_FOR_SCALE[scale_idx]
    params = CNNParams(
        patch_size_px=patch_size_px,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        seed=seed,
        dataset_dir=dataset_dir,
        aug_cell=cell,
    )
    variant = f"cnn_bce_S{patch_size_px}"
    snapshot = {
        "variant": variant,
        "task": "classification",
        "target_id": target.id,
        "scheme": scheme,
        "dataset_dir": dataset_dir or "dataset",
        "scale_idx": scale_idx,
        "tile_size_px": tile_size,
        "aug_cell": cell,
        "inner_val_images": inner_val_images,
        "model": {"variant": variant, "params": asdict(params)},
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash
    out_dir = MODELS_ROOT / variant / cfg_hash / f"scale_S{tile_size}_t{target.id}_aug_{cell}"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_records: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    for fold in iter_loio_folds(scheme, scale_idx=scale_idx, dataset_dir=dataset_dir):
        held = fold.held_out_obs_ids[0] if fold.held_out_obs_ids else f"fold{fold.fold_idx}"
        t0 = time.monotonic()
        y_train_full = target.binarize(fold.y_train).astype(np.float64)
        y_test = target.binarize(fold.y_test).astype(np.float64)

        keys_tr, y_tr, keys_vl, y_vl = _split_inner_val_groups(fold, y_train_full, inner_val_images)
        model = SmallCNNClassifier(params=params)
        model.bind_train_data(keys_tr, y_tr)
        model.bind_val_data(keys_vl, y_vl)
        model.fit(np.empty((len(keys_tr), 0), dtype=np.float32), y_tr)

        model.bind_predict_data(fold.keys_test)
        y_pred = model.predict(np.empty((len(fold.keys_test), 0), dtype=np.float32))
        assert np.isfinite(y_pred).all(), f"non-finite predictions on fold {held}"

        m = per_fold_metrics_classification(
            y_test.astype(np.int8), y_pred, held_out_obs_ids=fold.held_out_obs_ids)
        m["fold_idx"] = fold.fold_idx
        m["aug_cell"] = cell
        m["model_hash"] = model.model_hash()
        m["fit_seconds"] = time.monotonic() - t0
        per_fold_records.append(m)

        block = fold.keys_test.copy()
        block["fold_held_out_obs_id"] = held
        block["fold_idx"] = fold.fold_idx
        block["y_true"] = y_test
        block["y_pred"] = y_pred
        pred_rows.append(block)

        fold_out = out_dir / f"fold_{held}"
        fold_out.mkdir(parents=True, exist_ok=True)
        model.save(fold_out / "state_dict.pt")

        auc_str = "spec-only" if m["is_specificity_only"] else f"auc={m['auc']:+.4f}"
        print(f"    fold {fold.fold_idx:>2d} {held}: {auc_str}  "
              f"n_pos={m['n_positive']} n_neg={m['n_negative']}  {m['fit_seconds']:.0f}s", flush=True)

    aggregate = aggregate_fold_metrics_classification(per_fold_records)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    # Protocol metrics (PLAN_CNN.md §4.1): median per-image AUC over validity-passing
    # folds (under LOIO each fold's AUC IS a per-image AUC), pooled PR-AUC +
    # precision@top-5% over all held-out tiles concatenated (matches the banked
    # GBM tooling's pooled_global_pr_auc definition).
    from sklearn.metrics import average_precision_score
    real_aucs = [f["auc"] for f in per_fold_records
                 if not f["is_specificity_only"] and np.isfinite(f["auc"])]
    aggregate["auc_median"] = float(np.median(real_aucs)) if real_aucs else float("nan")
    aggregate["pr_auc_mean"] = float(np.nanmean(
        [f["pr_auc"] for f in per_fold_records if not f["is_specificity_only"]]))
    yt = pred_df["y_true"].to_numpy().astype(int)
    yp = pred_df["y_pred"].to_numpy()
    aggregate["pooled_pr_auc"] = float(average_precision_score(yt, yp))
    k = max(1, int(0.05 * len(yt)))
    top = np.argsort(-yp)[:k]
    aggregate["pooled_precision_at_top_5pct"] = float(yt[top].mean())
    pred_df.to_parquet(out_dir / "predictions.parquet", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps({"per_fold": per_fold_records, "aggregate": aggregate}, indent=2, default=float),
        encoding="utf-8",
    )
    snapshot["written_at_iso"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    (out_dir / "snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str), encoding="utf-8")
    return per_fold_records, aggregate, out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS),
                    choices=sorted(AUG_CELLS))
    ap.add_argument("--target", default="fa_gt_1e-2")
    ap.add_argument("--patch-size-px", type=int, default=64, choices=[32, 64])
    ap.add_argument("--scale-idx", type=int, default=3, choices=[0, 1, 2, 3])
    ap.add_argument("--scheme", default="loio_nfold")
    ap.add_argument("--dataset-dir", default="dataset_v2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--early-stopping-patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-val-images", type=int, default=4)
    args = ap.parse_args()

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "models" / "_sweep_cnn" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_meta.json").write_text(json.dumps({
        "kind": "cnn_aug_grid",
        "cells": args.cells,
        "target_id": args.target,
        "patch_size_px": args.patch_size_px,
        "scale_idx": args.scale_idx,
        "dataset_dir": args.dataset_dir,
        "scheme": args.scheme,
        "seed": args.seed,
        "inner_val_images": args.inner_val_images,
        "timestamp": timestamp,
        "script": "sweep_cnn.py",
    }, indent=2), encoding="utf-8")

    print(f"CNN augmentation grid: cells={args.cells}  target={args.target}  "
          f"S={TILE_SIZE_FOR_SCALE[args.scale_idx]} P={args.patch_size_px}  seed={args.seed}")
    print(f"Output: {out_dir}\n")

    summary_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for i, cell in enumerate(args.cells, 1):
        print(f"[{i}/{len(args.cells)}] aug_cell={cell}", flush=True)
        t0 = time.monotonic()
        per_fold, aggregate, artifact_dir = run_one_cell(
            cell,
            target_id=args.target,
            scale_idx=args.scale_idx,
            patch_size_px=args.patch_size_px,
            scheme=args.scheme,
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            inner_val_images=args.inner_val_images,
        )
        for f in per_fold:
            row = {
                "aug_cell": cell,
                "patch_size_px": args.patch_size_px,
                "scale_idx": args.scale_idx,
                "seed": args.seed,
                **{k: v for k, v in f.items() if k not in ("calibration_deciles",)},
                "held_out_obs_id": f["held_out_obs_ids"][0] if f["held_out_obs_ids"] else "",
            }
            row.pop("held_out_obs_ids", None)
            summary_rows.append(row)
        aggregate_rows.append({
            "aug_cell": cell,
            "patch_size_px": args.patch_size_px,
            "scale_idx": args.scale_idx,
            "seed": args.seed,
            **aggregate,
        })
        # Incremental write so a crashed grid still leaves the finished cells readable.
        pd.DataFrame(summary_rows).to_parquet(out_dir / "summary.parquet", index=False)
        pd.DataFrame(aggregate_rows).to_parquet(out_dir / "aggregate.parquet", index=False)
        dt = time.monotonic() - t0
        print(f"  cell {cell}: auc mean={aggregate['auc_mean']:+.4f} "
              f"median={aggregate['auc_median']:+.4f}  pooled_pr_auc={aggregate['pooled_pr_auc']:.4f}  "
              f"{dt/60:.1f} min  ({artifact_dir.relative_to(REPO_ROOT)})\n", flush=True)

    agg_df = pd.DataFrame(aggregate_rows)
    print("\n=== Aggregate ===")
    cols = [c for c in ("aug_cell", "auc_mean", "auc_std", "auc_median", "pooled_pr_auc",
                        "pooled_precision_at_top_5pct", "brier_mean") if c in agg_df.columns]
    print(agg_df[cols].to_string(index=False))
    print(f"\nWrote: {out_dir / 'summary.parquet'}")
    print(f"       {out_dir / 'aggregate.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
