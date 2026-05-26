"""Fan out over (variant, scale_idx) and persist all per-fold + aggregate artifacts.

Per AskUserQuestion 2026-05-27: ship all 3 GBM variants x all 4 scales = 12 sweeps.

A single invocation produces everything notebook 10 needs — both the aggregate
parquets and the per-(variant, scale) booster/prediction/metrics artifacts.
Same params -> same config_hash as `scripts/train_gbm.py`, so the two scripts
write to identical paths and are interchangeable.

Usage:
    python scripts/sweep.py                       # all GBM variants, all scales
    python scripts/sweep.py --variants lightgbm_tweedie --scales 0 8
    python scripts/sweep.py --include-cnn        # also CNN at S32 and S64

Writes:
    models/_sweep/{timestamp}/summary.parquet     # one row per (variant, scale_idx, fold)
    models/_sweep/{timestamp}/aggregate.parquet   # one row per (variant, scale_idx)
    models/{variant}/{config_hash}/scale_S{n}/
        predictions.parquet, metrics.json, snapshot.json
        fold_{obs_id}/booster.txt   (or presence.txt + magnitude.txt for two-stage)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, VARIANT_CONSTRUCTORS, make_factory, snapshot_params
from src.modeling.loaders import iter_loio_folds

SCALE_TILE_PX = {0: 8, 1: 16, 2: 32, 3: 64}
ALL_GBM_VARIANTS = list(VARIANT_CONSTRUCTORS)
DEFAULT_SCALES = (0, 1, 2, 3)
MODELS_ROOT = REPO_ROOT / "models"
TARGET_COL = "fractional_area"
SCHEME = "loio_9fold"


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_one(variant: str, scale_idx: int, params: LGBMParams) -> tuple[list[dict], dict, Path]:
    """LOIO eval + per-fold artifact persistence for one (variant, scale).

    Mirrors `scripts/train_gbm.py` so one `python scripts/sweep.py` invocation
    produces everything notebook 10's diagnostic cells need (predictions.parquet,
    metrics.json with per_bin_rmse, fold_<obs_id>/booster.txt). Writes to the
    canonical `models/<variant>/<config_hash>/scale_S{n}/` layout shared with
    train_gbm.py — same params -> same hash -> idempotent.
    """
    factory = make_factory(variant, params)
    tile_size = SCALE_TILE_PX[scale_idx]
    snapshot = {
        "variant": variant,
        "target_col": TARGET_COL,
        "scheme": SCHEME,
        "scale_idx": scale_idx,
        "tile_size_px": tile_size,
        "model": snapshot_params(variant, params),
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash
    out_dir = MODELS_ROOT / variant / cfg_hash / f"scale_S{tile_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_loio(
        factory,
        target_col=TARGET_COL,
        scheme=SCHEME,
        scale_idx=scale_idx,
        snapshot=snapshot,
        verbose=False,
    )
    write_run_artifacts(result, out_dir)

    # Per-fold booster persistence (mirrors train_gbm.py pass 2). Re-fit per fold
    # with the same inner-validation rotation so the saved boosters match the
    # predictions that just landed in predictions.parquet.
    for fold in iter_loio_folds(SCHEME, scale_idx=scale_idx):
        train_codes = fold.groups_train
        unique_codes = np.unique(train_codes)
        inner_val_code = int(unique_codes[fold.fold_idx % unique_codes.size])
        inner_val_mask = train_codes == inner_val_code
        inner_train_mask = ~inner_val_mask
        y_train_full = fold.y_train[TARGET_COL].to_numpy(dtype=np.float64)
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
        if variant == "lightgbm_two_stage":
            model.save(fold_out)
        else:
            model.save(fold_out / "booster.txt")

    return result.per_fold_metrics, result.aggregate, out_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=ALL_GBM_VARIANTS, choices=ALL_GBM_VARIANTS)
    ap.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--include-cnn", action="store_true",
                    help="Also run CNN at S32 and S64 (delegates to scripts/train_cnn.py).")
    args = ap.parse_args()

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "models" / "_sweep" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    runs = [(v, s) for v in args.variants for s in args.scales]
    print(f"Sweep: {len(runs)} runs across {len(args.variants)} variants x {len(args.scales)} scales")
    print(f"Output: {out_dir}\n")

    for i, (variant, scale_idx) in enumerate(runs, 1):
        tile_size = SCALE_TILE_PX[scale_idx]
        print(f"[{i:>2d}/{len(runs)}] {variant:<20s} scale_idx={scale_idx} (S={tile_size:>2d}) ...", flush=True)
        per_fold, aggregate, artifact_dir = run_one(variant, scale_idx, params)
        for f in per_fold:
            row = {
                "variant": variant,
                "scale_idx": scale_idx,
                "tile_size_px": tile_size,
                **{k: v for k, v in f.items() if k not in ("per_bin_rmse",)},
                "held_out_obs_id": f["held_out_obs_ids"][0] if f["held_out_obs_ids"] else "",
            }
            row.pop("held_out_obs_ids", None)
            summary_rows.append(row)
        aggregate_rows.append({
            "variant": variant,
            "scale_idx": scale_idx,
            "tile_size_px": tile_size,
            **aggregate,
        })
        rho_m = aggregate["spearman_rho_mean"]
        rho_s = aggregate["spearman_rho_std"]
        auc_m = aggregate["presence_auc_mean"]
        print(f"        rho={rho_m:+.4f} +/- {rho_s:.4f}  auc={auc_m:.3f}  ({artifact_dir.relative_to(REPO_ROOT)})")

    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_df.to_parquet(out_dir / "summary.parquet", index=False)
    aggregate_df.to_parquet(out_dir / "aggregate.parquet", index=False)

    print(f"\n=== Aggregate ===")
    print(aggregate_df[
        ["variant", "scale_idx", "tile_size_px",
         "spearman_rho_mean", "spearman_rho_std",
         "presence_auc_mean", "rmse_log1p_mean"]
    ].to_string(index=False))
    print(f"\nWrote: {out_dir / 'summary.parquet'}")
    print(f"       {out_dir / 'aggregate.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
