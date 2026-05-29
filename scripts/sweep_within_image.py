"""Fan out the within-image diagnostic CV sweep over (variant, scale).

Per PLAN_Stage5c.md §5: run the two model variants for which we have the strongest LOIO
evidence -- `lightgbm_two_stage` (best regression Spearman at S=64) and
`lightgbm_classification` at `bc_ge_1` (best binary AUC at S=32/S=64) -- across all four
scales = 8 (variant x scale) configurations on the `within_image_4fold` packaged scheme.
Each config evaluates 32 folds (8 non-empty images x 4 quadrants).

Diagnostic question (PLAN_Stage5c.md §1): does within-image AUC reach >> the LOIO baseline?
Yes -> per-image generalisation is the binding constraint, more HiRISE images is the unlock.
No  -> the 5 m/px texture signal floor is the binding constraint.

Usage:
    python scripts/sweep_within_image.py                            # both variants, all 4 scales
    python scripts/sweep_within_image.py --variants lightgbm_classification --scales 3
    python scripts/sweep_within_image.py --variants lightgbm_two_stage      # regression only

Writes:
    models/_sweep_within_image/{timestamp}/summary.parquet       # one row per (variant, scale, fold)
    models/_sweep_within_image/{timestamp}/aggregate.parquet     # one row per (variant, scale)
    models/_sweep_within_image/{timestamp}/per_image.parquet     # one row per (variant, scale, ObsId)
    models/<variant>/<config_hash>/scale_S{n}_within[/_t{target_id}]/
        predictions.parquet, metrics.json, snapshot.json
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

from src.modeling.binary_target import get_target
from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import (
    LGBMParams,
    LightGBMClassification,
    LightGBMTwoStage,
    snapshot_params,
)

SCALE_TILE_PX = {0: 8, 1: 16, 2: 32, 3: 64}
ALL_VARIANTS = ("lightgbm_two_stage", "lightgbm_classification")
DEFAULT_SCALES = (0, 1, 2, 3)
MODELS_ROOT = REPO_ROOT / "models"
SCHEME = "within_image_4fold"
# PLAN_Stage5c.md §5: the binary variant runs at bc_ge_1 (best Stage 5b cell).
BINARY_TARGET_ID = "bc_ge_1"


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _run_two_stage(
    scale_idx: int, params: LGBMParams, *, dataset_dir: str | None = None,
) -> tuple[list[dict], dict, Path]:
    """Regression: lightgbm_two_stage on `fractional_area`."""
    tile_size = SCALE_TILE_PX[scale_idx]
    snapshot = {
        "variant": "lightgbm_two_stage",
        "task": "regression",
        "scheme": SCHEME,
        "dataset_dir": dataset_dir or "dataset",
        "scale_idx": scale_idx,
        "tile_size_px": tile_size,
        "model": snapshot_params("lightgbm_two_stage", params),
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash
    out_dir = MODELS_ROOT / "lightgbm_two_stage" / cfg_hash / f"scale_S{tile_size}_within"
    out_dir.mkdir(parents=True, exist_ok=True)

    def factory() -> LightGBMTwoStage:
        return LightGBMTwoStage(params=params)

    result = run_loio(
        factory,
        target_col="fractional_area",
        task="regression",
        scheme=SCHEME,
        scale_idx=scale_idx,
        dataset_dir=dataset_dir,
        snapshot=snapshot,
        verbose=False,
    )
    write_run_artifacts(result, out_dir)
    return result.per_fold_metrics, result.aggregate, out_dir


def _run_classification(
    scale_idx: int, params: LGBMParams, *, dataset_dir: str | None = None,
) -> tuple[list[dict], dict, Path]:
    """Binary: lightgbm_classification at bc_ge_1."""
    target = get_target(BINARY_TARGET_ID)
    tile_size = SCALE_TILE_PX[scale_idx]
    snapshot = {
        "variant": "lightgbm_classification",
        "task": "classification",
        "target_id": target.id,
        "target_source_col": target.source_col,
        "target_threshold": target.threshold,
        "target_comparison": target.comparison,
        "scheme": SCHEME,
        "dataset_dir": dataset_dir or "dataset",
        "scale_idx": scale_idx,
        "tile_size_px": tile_size,
        "model": snapshot_params("lightgbm_classification", params),
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash
    out_dir = (
        MODELS_ROOT / "lightgbm_classification" / cfg_hash
        / f"scale_S{tile_size}_within_t{target.id}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    def factory() -> LightGBMClassification:
        return LightGBMClassification(params=params)

    result = run_loio(
        factory,
        binarize=target.binarize,
        task="classification",
        scheme=SCHEME,
        scale_idx=scale_idx,
        dataset_dir=dataset_dir,
        snapshot=snapshot,
        verbose=False,
    )
    write_run_artifacts(result, out_dir)
    return result.per_fold_metrics, result.aggregate, out_dir


def _flatten_fold_row(per_fold: dict, *, variant: str, scale_idx: int, tile_size: int) -> dict:
    """Strip out parquet-unfriendly list-of-dict columns; flatten held_out_obs_ids."""
    drop_keys = {"calibration_deciles", "per_bin_rmse"}
    row = {
        "variant": variant,
        "scale_idx": scale_idx,
        "tile_size_px": tile_size,
        **{k: v for k, v in per_fold.items() if k not in drop_keys},
        "held_out_obs_id": per_fold["held_out_obs_ids"][0] if per_fold["held_out_obs_ids"] else "",
    }
    row.pop("held_out_obs_ids", None)
    return row


def _per_image_aggregate(summary_df: pd.DataFrame, task_metric: dict[str, str]) -> pd.DataFrame:
    """Average the 4 within-image folds for one image -> per-(variant, scale, ObsId) row.

    `task_metric` maps variant -> primary metric column ('auc' or 'spearman_rho'). Specificity-only
    folds are excluded from the metric average; their count is reported per row.
    """
    rows: list[dict] = []
    for (variant, scale_idx, obs_id), grp in summary_df.groupby(["variant", "scale_idx", "held_out_obs_id"]):
        metric_col = task_metric[variant]
        real = grp[~grp["is_specificity_only"].astype(bool)]
        rows.append({
            "variant": variant,
            "scale_idx": int(scale_idx),
            "tile_size_px": int(grp["tile_size_px"].iloc[0]),
            "held_out_obs_id": obs_id,
            "n_folds": int(len(grp)),
            "n_real_folds": int(len(real)),
            f"{metric_col}_mean": float(real[metric_col].mean()) if len(real) > 0 else float("nan"),
            f"{metric_col}_std": float(real[metric_col].std(ddof=0)) if len(real) > 0 else float("nan"),
            "n_test_tiles_total": int(grp["n_tiles"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["variant", "scale_idx", "held_out_obs_id"]).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(ALL_VARIANTS), choices=list(ALL_VARIANTS))
    ap.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--dataset-dir", default=None,
                    help="Packaged dataset root (default: ./dataset = v1). Use dataset_v2 for the vClaire A/B. "
                         "Scheme is within_image_4fold in both (the fold count differs, not the name).")
    args = ap.parse_args()

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "models" / "_sweep_within_image" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_meta.json").write_text(json.dumps({
        "kind": "within_image",
        "dataset_dir": args.dataset_dir or "dataset",
        "scheme": SCHEME,
        "timestamp": timestamp,
        "script": "sweep_within_image.py",
    }, indent=2), encoding="utf-8")

    runs = [(v, s) for v in args.variants for s in args.scales]
    print(f"Within-image sweep on scheme={SCHEME}: {len(runs)} runs "
          f"({len(args.variants)} variants x {len(args.scales)} scales)")
    print(f"Dataset: {args.dataset_dir or 'dataset'}")
    print(f"Output: {out_dir}\n", flush=True)

    summary_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for i, (variant, scale_idx) in enumerate(runs, 1):
        tile_size = SCALE_TILE_PX[scale_idx]
        print(f"[{i:>2d}/{len(runs)}] variant={variant:<25s} scale_idx={scale_idx} (S={tile_size:>2d}) ...",
              flush=True)
        if variant == "lightgbm_two_stage":
            per_fold, aggregate, artifact_dir = _run_two_stage(
                scale_idx, params, dataset_dir=args.dataset_dir)
            primary = ("spearman_rho_mean", "spearman_rho_std")
        elif variant == "lightgbm_classification":
            per_fold, aggregate, artifact_dir = _run_classification(
                scale_idx, params, dataset_dir=args.dataset_dir)
            primary = ("auc_mean", "auc_std")
        else:
            raise AssertionError(f"unexpected variant {variant!r}")

        for f in per_fold:
            summary_rows.append(_flatten_fold_row(
                f, variant=variant, scale_idx=scale_idx, tile_size=tile_size,
            ))
        aggregate_rows.append({
            "variant": variant,
            "scale_idx": scale_idx,
            "tile_size_px": tile_size,
            **aggregate,
        })
        m_mean = aggregate[primary[0]]
        m_std = aggregate[primary[1]]
        print(f"        {primary[0].replace('_mean','')}={m_mean:+.4f} +/- {m_std:.4f}  "
              f"({artifact_dir.relative_to(REPO_ROOT)})", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_df.to_parquet(out_dir / "summary.parquet", index=False)
    aggregate_df.to_parquet(out_dir / "aggregate.parquet", index=False)

    # Per-image aggregate: average the 4 quadrant-folds per image to get one number per
    # (variant, scale, ObsId). This is the PLAN_Stage5c.md §6 quantity used in the LOIO
    # comparison.
    task_metric = {
        "lightgbm_two_stage": "spearman_rho",
        "lightgbm_classification": "auc",
    }
    # Only include variants that produced summary rows (in case the user filtered).
    used_task_metric = {v: m for v, m in task_metric.items() if v in summary_df["variant"].unique().tolist()}
    per_image_df = _per_image_aggregate(summary_df, used_task_metric)
    per_image_df.to_parquet(out_dir / "per_image.parquet", index=False)

    print(f"\n=== Aggregate ===")
    cols = ["variant", "scale_idx", "tile_size_px"]
    for k in ("auc_mean", "auc_std", "spearman_rho_mean", "spearman_rho_std",
              "presence_auc_mean", "presence_auc_std", "brier_mean"):
        if k in aggregate_df.columns:
            cols.append(k)
    print(aggregate_df[cols].to_string(index=False))
    print(f"\nWrote: {out_dir / 'summary.parquet'}")
    print(f"       {out_dir / 'aggregate.parquet'}")
    print(f"       {out_dir / 'per_image.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
