"""Fan out the binary-classification variant over (target_id, scale_idx).

Per PLAN_Stage5b.md §6: ship all 3 binary targets x all 4 scales = 12 runs.
Mirrors scripts/sweep.py's structure and produces both the cross-config
sweep parquets AND the per-(target, scale) artifacts the notebook consumes.

Usage:
    python scripts/sweep_binary.py                            # all 3 targets, all 4 scales
    python scripts/sweep_binary.py --targets bc_ge_1 --scales 3
    python scripts/sweep_binary.py --skip-fa-gt-1e-2-s8       # skip the infeasible cell

Writes:
    models/_sweep_binary/{timestamp}/summary.parquet     # one row per (target, scale_idx, fold)
    models/_sweep_binary/{timestamp}/aggregate.parquet   # one row per (target, scale_idx)
    models/lightgbm_classification/{config_hash}/scale_S{n}_t{target_id}/
        predictions.parquet, metrics.json, snapshot.json
        fold_{obs_id}/classifier.txt
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd

from src.modeling.binary_target import BINARY_TARGETS, BINARY_TARGETS_BY_ID, get_target
from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, LightGBMClassification, snapshot_params
from src.modeling.loaders import iter_loio_folds

SCALE_TILE_PX = {0: 8, 1: 16, 2: 32, 3: 64}
ALL_TARGET_IDS = [t.id for t in BINARY_TARGETS]
DEFAULT_SCALES = (0, 1, 2, 3)
MODELS_ROOT = REPO_ROOT / "models"
DEFAULT_SCHEME = "loio_9fold"  # v1; v2 (dataset_v2) uses loio_nfold -- pass --scheme.


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_one(
    target_id: str,
    scale_idx: int,
    params: LGBMParams,
    *,
    scheme: str = DEFAULT_SCHEME,
    dataset_dir: str | None = None,
) -> tuple[list[dict], dict, Path]:
    """LOIO eval + per-fold classifier persistence for one (target, scale)."""
    target = get_target(target_id)
    tile_size = SCALE_TILE_PX[scale_idx]
    snapshot = {
        "variant": "lightgbm_classification",
        "task": "classification",
        "target_id": target.id,
        "target_source_col": target.source_col,
        "target_threshold": target.threshold,
        "target_comparison": target.comparison,
        "scheme": scheme,
        "dataset_dir": dataset_dir or "dataset",
        "scale_idx": scale_idx,
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

    def factory() -> LightGBMClassification:
        return LightGBMClassification(params=params)

    result = run_loio(
        factory,
        binarize=target.binarize,
        task="classification",
        scheme=scheme,
        scale_idx=scale_idx,
        dataset_dir=dataset_dir,
        snapshot=snapshot,
        verbose=False,
    )
    write_run_artifacts(result, out_dir)

    # Persist per-fold classifier artifacts (mirrors train_gbm/sweep.py pass 2).
    for fold in iter_loio_folds(scheme, scale_idx=scale_idx, dataset_dir=dataset_dir):
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

    return result.per_fold_metrics, result.aggregate, out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=ALL_TARGET_IDS, choices=ALL_TARGET_IDS)
    ap.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--skip-fa-gt-1e-2-s8", action="store_true",
                    help="Skip fa_gt_1e-2 at S=8 (only ~73 positives in the whole "
                         "dataset; PLAN_Stage5b.md §11 q4 flags as essentially infeasible)")
    ap.add_argument("--dataset-dir", default=None,
                    help="Packaged dataset root (default: ./dataset = v1). Use dataset_v2 for the vClaire A/B.")
    ap.add_argument("--scheme", default=DEFAULT_SCHEME,
                    help=f"LOIO scheme name (default: {DEFAULT_SCHEME} for v1; use loio_nfold for dataset_v2).")
    args = ap.parse_args()

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "models" / "_sweep_binary" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_meta.json").write_text(json.dumps({
        "kind": "binary",
        "dataset_dir": args.dataset_dir or "dataset",
        "scheme": args.scheme,
        "timestamp": timestamp,
        "script": "sweep_binary.py",
    }, indent=2), encoding="utf-8")

    summary_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    runs = [
        (t, s) for t in args.targets for s in args.scales
        if not (args.skip_fa_gt_1e_2_s8 and t == "fa_gt_1e-2" and s == 0)
    ]
    print(f"Binary sweep: {len(runs)} runs across {len(args.targets)} targets x {len(args.scales)} scales")
    print(f"Dataset: {args.dataset_dir or 'dataset'}  scheme: {args.scheme}")
    print(f"Output: {out_dir}\n")

    for i, (target_id, scale_idx) in enumerate(runs, 1):
        tile_size = SCALE_TILE_PX[scale_idx]
        print(f"[{i:>2d}/{len(runs)}] target={target_id:<11s} scale_idx={scale_idx} (S={tile_size:>2d}) ...", flush=True)
        per_fold, aggregate, artifact_dir = run_one(
            target_id, scale_idx, params, scheme=args.scheme, dataset_dir=args.dataset_dir,
        )
        for f in per_fold:
            row = {
                "target_id": target_id,
                "scale_idx": scale_idx,
                "tile_size_px": tile_size,
                # Drop the per-fold calibration_deciles list-of-dicts (ugly in parquet;
                # available in the per-fold metrics.json for analysis).
                **{k: v for k, v in f.items() if k not in ("calibration_deciles",)},
                "held_out_obs_id": f["held_out_obs_ids"][0] if f["held_out_obs_ids"] else "",
            }
            row.pop("held_out_obs_ids", None)
            summary_rows.append(row)
        aggregate_rows.append({
            "target_id": target_id,
            "scale_idx": scale_idx,
            "tile_size_px": tile_size,
            **aggregate,
        })
        auc_m = aggregate["auc_mean"]
        auc_s = aggregate["auc_std"]
        lift_m = aggregate["lift_at_top_k_mean"]
        brier_m = aggregate["brier_mean"]
        print(f"        auc={auc_m:+.4f} +/- {auc_s:.4f}  lift={lift_m:.3f}  brier={brier_m:.4g}  "
              f"({artifact_dir.relative_to(REPO_ROOT)})")

    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_df.to_parquet(out_dir / "summary.parquet", index=False)
    aggregate_df.to_parquet(out_dir / "aggregate.parquet", index=False)

    print(f"\n=== Aggregate ===")
    print(aggregate_df[
        ["target_id", "scale_idx", "tile_size_px",
         "auc_mean", "auc_std",
         "brier_mean", "ece_mean", "lift_at_top_k_mean"]
    ].to_string(index=False))
    print(f"\nWrote: {out_dir / 'summary.parquet'}")
    print(f"       {out_dir / 'aggregate.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
