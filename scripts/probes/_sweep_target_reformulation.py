"""H2 dev sweep: does target reformulation move the headline metrics?

Compares `lightgbm_two_stage_balanced` (the §5 winner) on three targets:

  - `fractional_area`              -- baseline (continuous, pixel-aliasing-noisy at low end)
  - `boulder_count`                -- discrete, alias-robust at the low end
  - `log_boulder_count`            -- explicit log transform of the above (rank-equivalent
                                      under monotone transform, but with the same explicit-log
                                      semantics as our log1p+huber magnitude head)

The variants log1p-transform internally on their magnitude head; passing
`target=boulder_count` causes the model to fit `log1p(boulder_count)` on positives, which
is the natural Poisson-ish scale.  Spearman is rank-invariant under monotone transforms,
but the loss interacts differently with each target's distribution.

Composite metrics (Phase A2 / H1 additions in `src/modeling/evaluate.py`):
  - Spearman ρ                -- ranking (rank-invariant: should be similar across targets)
  - presence AUC              -- detection power at the trivial `y > 0` threshold
  - meaningful AUC            -- detection at the operational threshold (default fa > 0.01)
  - PR-AUC                    -- precision-recall AUC (more informative than ROC-AUC when
                                 base rate is high)
  - lift@top-K + normalised   -- operational top-of-ranking quality
  - precision/recall @top-{1,5,10}%
  - compression_score, zero_pred, high_ratio  (the §5 per-bin diagnostic)

Usage:
    conda run -n geospatial python scripts/probes/_sweep_target_reformulation.py
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

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd

from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, VARIANT_CONSTRUCTORS, snapshot_params
from src.modeling.loaders import Fold, iter_loio_folds

SCALE_TILE_PX = {0: 8, 1: 16, 2: 32, 3: 64, 4: 128}
SCHEME = "within_image_4fold"
VARIANT = "lightgbm_two_stage_balanced"
DEFAULT_TARGETS = ("fractional_area", "boulder_count", "log_boulder_count")
DEFAULT_SCALES = (2, 3)
MODELS_ROOT = REPO_ROOT / "models"

# When the target is not the canonical fractional_area, the "operational
# meaningful threshold" needs to be re-mapped.  These map roughly to "boulder-rich tile"
# (the v1 fa_gt_1e-2 definition translated into the new units).
MEANINGFUL_THRESHOLDS = {
    "fractional_area": 1e-2,        # >1% area
    "boulder_count": 50.0,          # ~50 boulders/tile (matches fa~1e-2 at typical density)
    "log_boulder_count": np.log1p(50.0),  # log1p(50) ~ 3.93
}


def _add_log_target(fold: Fold) -> None:
    """Mutate fold.y_train / fold.y_test in-place to add log_boulder_count column."""
    for df in (fold.y_train, fold.y_test):
        if "log_boulder_count" not in df.columns:
            df["log_boulder_count"] = np.log1p(df["boulder_count"].to_numpy(dtype=np.float64))


def _wrapped_fold_iter(scheme: str, scale_idx: int, dataset_dir: str):
    """Yield folds with log_boulder_count derived on the fly."""
    def _it():
        for f in iter_loio_folds(scheme, scale_idx=scale_idx, dataset_dir=dataset_dir):
            _add_log_target(f)
            yield f
    return _it


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_one(
    target_col: str, scale_idx: int, params: LGBMParams, *, dataset_dir: str,
) -> tuple[list[dict], dict, Path]:
    tile_size = SCALE_TILE_PX[scale_idx]
    snapshot = {
        "variant": VARIANT,
        "task": "regression",
        "target_col": target_col,
        "scheme": SCHEME,
        "dataset_dir": dataset_dir,
        "scale_idx": scale_idx,
        "tile_size_px": tile_size,
        "model": snapshot_params(VARIANT, params),
        "meaningful_threshold": MEANINGFUL_THRESHOLDS[target_col],
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash
    out_dir = MODELS_ROOT / VARIANT / cfg_hash / f"scale_S{tile_size}_target_{target_col}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls = VARIANT_CONSTRUCTORS[VARIANT]

    def factory():
        return cls(params=params)

    # Need to plumb meaningful_threshold to per_fold_metrics.  We do this by
    # patching run_loio's internal per_fold_metrics call -- simplest path is a
    # monkeypatch in the calling module's namespace.
    import src.modeling.evaluate as _ev
    orig_per_fold_metrics = _ev.per_fold_metrics
    mt = MEANINGFUL_THRESHOLDS[target_col]

    def _patched(y_true, y_pred, *, held_out_obs_ids, meaningful_threshold=mt):
        return orig_per_fold_metrics(y_true, y_pred, held_out_obs_ids=held_out_obs_ids,
                                     meaningful_threshold=meaningful_threshold)

    _ev.per_fold_metrics = _patched
    try:
        result = run_loio(
            factory,
            target_col=target_col,
            task="regression",
            scheme=SCHEME,
            scale_idx=scale_idx,
            dataset_dir=dataset_dir,
            fold_iter=_wrapped_fold_iter(SCHEME, scale_idx, dataset_dir),
            snapshot=snapshot,
            verbose=False,
        )
    finally:
        _ev.per_fold_metrics = orig_per_fold_metrics
    write_run_artifacts(result, out_dir)
    return result.per_fold_metrics, result.aggregate, out_dir


def _per_target_compression(out_dir: Path, target_col: str) -> dict:
    """Read predictions parquet; compute per-bin compression stats on *fractional_area*.

    Even when the model predicts boulder_count or log_boulder_count, we report compression
    relative to the canonical fractional_area target so the numbers are comparable to §5.
    For non-fractional_area targets, we read fractional_area from the same predictions
    parquet (it's a column on the labels dataframe, persisted by write_run_artifacts).
    """
    pred_path = out_dir / "predictions.parquet"
    if not pred_path.exists():
        return {}
    df = pd.read_parquet(pred_path)
    # write_run_artifacts persists y_true under whatever target_col we asked for; for
    # non-fractional_area targets we can't recover fractional_area from the parquet alone.
    # Just report the compression on the OWN target scale -- bins shifted accordingly.
    if target_col == "fractional_area":
        edges = [(-1e-12, 0.0, "zero"),
                 (0.0, 1e-4, "0_to_1e-4"),
                 (1e-4, 1e-3, "1e-4_to_1e-3"),
                 (1e-3, 1e-2, "1e-3_to_1e-2"),
                 (1e-2, 1.0, "1e-2_to_max")]
    elif target_col == "boulder_count":
        # Bins chosen to roughly match the fractional_area decades on v2 at typical density
        edges = [(-1e-12, 0.0, "zero"),
                 (0.0, 1.0, "1_to_10"),
                 (1.0, 10.0, "10_to_100"),
                 (10.0, 100.0, "100_to_1k"),
                 (100.0, 1e9, "1k_plus")]
    else:  # log_boulder_count
        edges = [(-1e-12, 0.0, "zero"),
                 (0.0, np.log1p(1.0), "0_to_log1p_1"),
                 (np.log1p(1.0), np.log1p(10.0), "log1p_1_to_10"),
                 (np.log1p(10.0), np.log1p(100.0), "log1p_10_to_100"),
                 (np.log1p(100.0), 100.0, "log1p_100_plus")]
    rows = {}
    for lo, hi, name in edges:
        if name == "zero":
            mask = df["y_true"] <= 0.0
        else:
            mask = (df["y_true"] > lo) & (df["y_true"] <= hi)
        sub = df[mask]
        n = int(len(sub))
        mt = float(sub["y_true"].mean()) if n else float("nan")
        mp = float(sub["y_pred"].mean()) if n else float("nan")
        rows[f"{name}__n"] = n
        rows[f"{name}__mean_true"] = mt
        rows[f"{name}__mean_pred"] = mp
        rows[f"{name}__ratio"] = mp / mt if mt > 0 else float("nan")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=list(DEFAULT_TARGETS))
    ap.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--dataset-dir", default="dataset_v2_dev")
    args = ap.parse_args()

    for t in args.targets:
        if t not in MEANINGFUL_THRESHOLDS:
            raise SystemExit(f"unknown target_col: {t!r}")

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "models" / "_sweep_target_reformulation" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_meta.json").write_text(json.dumps({
        "kind": "target_reformulation",
        "dataset_dir": args.dataset_dir,
        "scheme": SCHEME,
        "variant": VARIANT,
        "targets": list(args.targets),
        "scales": list(args.scales),
        "timestamp": timestamp,
        "script": "_sweep_target_reformulation.py",
    }, indent=2), encoding="utf-8")

    runs = [(t, s) for t in args.targets for s in args.scales]
    print(f"Target-reformulation sweep on {SCHEME} dataset_dir={args.dataset_dir}")
    print(f"variant={VARIANT}  {len(runs)} runs across {len(args.targets)} targets x {len(args.scales)} scales")
    print(f"Output: {out_dir}\n", flush=True)

    summary_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for i, (target_col, scale_idx) in enumerate(runs, 1):
        tile_size = SCALE_TILE_PX[scale_idx]
        print(f"[{i:>2d}/{len(runs)}] target={target_col:<22s} S={tile_size:>2d} ...", flush=True)
        per_fold, aggregate, artifact_dir = run_one(
            target_col, scale_idx, params, dataset_dir=args.dataset_dir,
        )
        for f in per_fold:
            drop = {"calibration_deciles", "per_bin_rmse"}
            row = {
                "target_col": target_col,
                "scale_idx": scale_idx,
                "tile_size_px": tile_size,
                **{k: v for k, v in f.items() if k not in drop},
                "held_out_obs_id": f["held_out_obs_ids"][0] if f["held_out_obs_ids"] else "",
            }
            row.pop("held_out_obs_ids", None)
            summary_rows.append(row)
        comp = _per_target_compression(artifact_dir, target_col)
        agg_row = {
            "target_col": target_col,
            "scale_idx": scale_idx,
            "tile_size_px": tile_size,
            **aggregate,
            **comp,
            "artifact_dir": str(artifact_dir.relative_to(REPO_ROOT)),
        }
        aggregate_rows.append(agg_row)
        print(f"        rho={aggregate.get('spearman_rho_mean', float('nan')):+.4f}  "
              f"presence_auc={aggregate.get('presence_auc_mean', float('nan')):.3f}  "
              f"meaningful_auc={aggregate.get('meaningful_auc_mean', float('nan')):.3f}  "
              f"PR-AUC={aggregate.get('pr_auc_mean', float('nan')):.3f}  "
              f"lift_norm={aggregate.get('normalised_lift_meaningful_mean', float('nan')):.3f}",
              flush=True)

    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_df.to_parquet(out_dir / "summary.parquet", index=False)
    aggregate_df.to_parquet(out_dir / "aggregate.parquet", index=False)

    print(f"\n=== Composite metric table (higher = better on all but compression_score) ===")
    cols = [
        "target_col", "tile_size_px",
        "spearman_rho_mean", "presence_auc_mean", "meaningful_auc_mean",
        "pr_auc_mean",
        "lift_at_top_k_meaningful_mean", "normalised_lift_meaningful_mean",
        "precision_at_top_5pct_mean", "recall_at_top_5pct_mean",
    ]
    cols = [c for c in cols if c in aggregate_df.columns]
    print(aggregate_df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote: {out_dir / 'aggregate.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
