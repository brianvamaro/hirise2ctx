"""W0 "bank the wins" sweep: (variant x target x scheme x scale) on full-v2 LOIO.

PLAN_ModelUsability.md W0. One probe covers the whole regression-side matrix:

  - P1+P2 promotion:   lightgbm_two_stage_balanced x {fractional_area, boulder_count}
  - single-stage test: lightgbm_log1p_huber        x {fractional_area, boulder_count}
  - historical anchor: lightgbm_two_stage          x fractional_area
  - Stage 6a confirm:  lightgbm_two_stage_balanced x boulder_count at S=32 on
                       --scheme loio_nfold (baseline) vs loio_nfold_nbr_s5 (5x5 stencil)

Generalises scripts/probes/_sweep_target_reformulation.py (which hardcodes the
within_image_4fold scheme + the two-stage-balanced variant) to arbitrary
(variant, scheme). Same artifact layout + meaningful-threshold conventions:
fa > 1e-2 / boulder_count > 50 at every scale (the flat-50 convention matches
_sweep_stage6a.py so S=32 numbers stay comparable to the dev sweep).

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/probes/_sweep_w0.py \
        --variants lightgbm_two_stage_balanced lightgbm_log1p_huber \
        --targets fractional_area boulder_count --scales 3

Writes:
    models/_sweep_w0/{timestamp}/summary.parquet    # one row per (variant, target, scale, fold)
    models/_sweep_w0/{timestamp}/aggregate.parquet  # one row per (variant, target, scale)
    models/_sweep_w0/{timestamp}/result.md          # cross-cell comparison table
    models/{variant}/{config_hash}/scale_S{n}_target_{target}/   # canonical artifacts
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

import numpy as np
import pandas as pd

from src.modeling.evaluate import run_loio, write_run_artifacts
from src.modeling.gbm import LGBMParams, VARIANT_CONSTRUCTORS, snapshot_params

SCALE_TILE_PX = {0: 8, 1: 16, 2: 32, 3: 64, 4: 128}
REGRESSION_VARIANTS = [v for v in VARIANT_CONSTRUCTORS if not v.startswith("lightgbm_classification")]
SUPPORTED_TARGETS = ("fractional_area", "boulder_count")
MODELS_ROOT = REPO_ROOT / "models"


def _meaningful_threshold(target_col: str) -> float:
    """Operational boulder-rich cut. Flat across scales by convention
    (matches _sweep_stage6a.py; the fa=0.01-equivalent count at S=32 would be
    12.5, but cross-sweep comparability wins over per-scale remapping)."""
    if target_col == "fractional_area":
        return 1e-2
    if target_col == "boulder_count":
        return 50.0
    raise ValueError(f"unsupported target_col {target_col!r}")


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_one(
    variant: str,
    target_col: str,
    scale_idx: int,
    params: LGBMParams,
    *,
    scheme: str,
    dataset_dir: str,
) -> tuple[list[dict], dict, Path]:
    tile_size = SCALE_TILE_PX[scale_idx]
    snapshot = {
        "variant": variant,
        "task": "regression",
        "target_col": target_col,
        "scheme": scheme,
        "dataset_dir": dataset_dir,
        "scale_idx": scale_idx,
        "tile_size_px": tile_size,
        "model": snapshot_params(variant, params),
        "meaningful_threshold": _meaningful_threshold(target_col),
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash
    out_dir = MODELS_ROOT / variant / cfg_hash / f"scale_S{tile_size}_target_{target_col}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls = VARIANT_CONSTRUCTORS[variant]

    def factory():
        return cls(params=params)

    # Plumb meaningful_threshold into per_fold_metrics (same monkeypatch route
    # as _sweep_target_reformulation.py -- run_loio has no threshold kwarg).
    import src.modeling.evaluate as _ev
    orig_per_fold_metrics = _ev.per_fold_metrics
    mt = _meaningful_threshold(target_col)

    def _patched(y_true, y_pred, *, held_out_obs_ids, meaningful_threshold=mt):
        return orig_per_fold_metrics(y_true, y_pred, held_out_obs_ids=held_out_obs_ids,
                                     meaningful_threshold=meaningful_threshold)

    _ev.per_fold_metrics = _patched
    try:
        result = run_loio(
            factory,
            target_col=target_col,
            task="regression",
            scheme=scheme,
            scale_idx=scale_idx,
            dataset_dir=dataset_dir,
            snapshot=snapshot,
            verbose=False,
        )
    finally:
        _ev.per_fold_metrics = orig_per_fold_metrics
    write_run_artifacts(result, out_dir)
    return result.per_fold_metrics, result.aggregate, out_dir


# presence_auc deliberately omitted: deprecated 2026-06-10 (Brian) -- ">=1 boulder
# anywhere in a 320 m tile" is unobservable at 5 m/px and undefined (single-class)
# on ~1/4 of the cohort's images. meaningful_auc is the discrimination metric.
_MD_ROWS = (
    ("Spearman rho", "spearman_rho_mean"),
    ("meaningful AUC (ROC)", "meaningful_auc_mean"),
    ("PR-AUC", "pr_auc_mean"),
    ("normalised lift @top-K", "normalised_lift_meaningful_mean"),
    ("precision @top-5%", "precision_at_top_5pct_mean"),
    ("recall @top-5%", "recall_at_top_5pct_mean"),
)


def _write_result_md(out_dir: Path, aggregate_df: pd.DataFrame, meta: dict) -> None:
    lines = [
        f"# W0 sweep -- {meta['timestamp']}",
        "",
        f"Dataset: `{meta['dataset_dir']}`  | Scheme: `{meta['scheme']}`  | "
        f"params: n_estimators={meta['n_estimators']} lr={meta['learning_rate']}",
        "",
    ]
    cells = [
        f"{r.variant} / {r.target_col} / S={r.tile_size_px}"
        for r in aggregate_df.itertuples()
    ]
    lines.append("| metric | " + " | ".join(cells) + " |")
    lines.append("|---" * (len(cells) + 1) + "|")
    for label, key in _MD_ROWS:
        if key not in aggregate_df.columns:
            continue
        vals = " | ".join(f"{v:+.4f}" for v in aggregate_df[key])
        lines.append(f"| {label} | {vals} |")
    (out_dir / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", required=True, choices=REGRESSION_VARIANTS)
    ap.add_argument("--targets", nargs="+", default=list(SUPPORTED_TARGETS),
                    choices=list(SUPPORTED_TARGETS))
    ap.add_argument("--scales", nargs="+", type=int, default=[3])
    ap.add_argument("--scheme", default="loio_nfold")
    ap.add_argument("--dataset-dir", default="dataset_v2")
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    args = ap.parse_args()

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "models" / "_sweep_w0" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "kind": "w0_bank_the_wins",
        "dataset_dir": args.dataset_dir,
        "scheme": args.scheme,
        "variants": list(args.variants),
        "targets": list(args.targets),
        "scales": list(args.scales),
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "timestamp": timestamp,
        "script": "_sweep_w0.py",
    }
    (out_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    runs = [(v, t, s) for v in args.variants for t in args.targets for s in args.scales]
    print(f"W0 sweep on scheme={args.scheme} dataset_dir={args.dataset_dir}")
    print(f"{len(runs)} cells: variants={args.variants} x targets={args.targets} x scales={args.scales}")
    print(f"Output: {out_dir}\n", flush=True)

    summary_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for i, (variant, target_col, scale_idx) in enumerate(runs, 1):
        tile_size = SCALE_TILE_PX[scale_idx]
        print(f"[{i:>2d}/{len(runs)}] {variant:<28s} target={target_col:<16s} S={tile_size:>3d} ...",
              flush=True)
        per_fold, aggregate, artifact_dir = run_one(
            variant, target_col, scale_idx, params,
            scheme=args.scheme, dataset_dir=args.dataset_dir,
        )
        for f in per_fold:
            drop = {"calibration_deciles", "per_bin_rmse"}
            row = {
                "variant": variant,
                "target_col": target_col,
                "scale_idx": scale_idx,
                "tile_size_px": tile_size,
                **{k: v for k, v in f.items() if k not in drop},
                "held_out_obs_id": f["held_out_obs_ids"][0] if f["held_out_obs_ids"] else "",
            }
            row.pop("held_out_obs_ids", None)
            summary_rows.append(row)
        aggregate_rows.append({
            "variant": variant,
            "target_col": target_col,
            "scale_idx": scale_idx,
            "tile_size_px": tile_size,
            **aggregate,
            "artifact_dir": str(artifact_dir.relative_to(REPO_ROOT)),
        })
        print(f"        rho={aggregate.get('spearman_rho_mean', float('nan')):+.4f}  "
              f"meaningful_auc={aggregate.get('meaningful_auc_mean', float('nan')):.3f}  "
              f"PR-AUC={aggregate.get('pr_auc_mean', float('nan')):.3f}  "
              f"lift_norm={aggregate.get('normalised_lift_meaningful_mean', float('nan')):.3f}  "
              f"prec@5%={aggregate.get('precision_at_top_5pct_mean', float('nan')):.3f}",
              flush=True)

    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_df.to_parquet(out_dir / "summary.parquet", index=False)
    aggregate_df.to_parquet(out_dir / "aggregate.parquet", index=False)
    _write_result_md(out_dir, aggregate_df, meta)

    print("\n=== Aggregate ===")
    cols = ["variant", "target_col", "tile_size_px",
            "spearman_rho_mean", "presence_auc_mean", "meaningful_auc_mean",
            "pr_auc_mean", "normalised_lift_meaningful_mean",
            "precision_at_top_5pct_mean", "recall_at_top_5pct_mean"]
    cols = [c for c in cols if c in aggregate_df.columns]
    print(aggregate_df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote: {out_dir / 'result.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
