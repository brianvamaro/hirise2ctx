"""Run the compression-fix variants on the v2-dev within-image scheme.

Mirrors `scripts/sweep_within_image.py`'s two-stage path but fans out across the
4 new two-stage subvariants (balanced / weighted / gamma / combined) plus the
baseline, at S=32 and S=64 by default.

Per the handoff prompt and the AskUserQuestion answer (2026-05-29):
  - dev harness only (dataset_v2_dev / within_image_4fold = 20 folds)
  - composite metric: per-truth-bin mean_pred/mean_true ratio + Spearman + AUC
  - the bin-ratio numbers are computed by reading each variant's per-fold
    predictions.parquet after the sweep finishes (see _summarise_compression()
    below); the in-sweep `metrics.json` already carries per_bin_rmse with
    `mean_true` / `mean_pred`, so this is a parquet aggregation, not a refit.

Usage (from repo root):
    conda run -n geospatial python scripts/probes/_sweep_compression_fixes.py
    conda run -n geospatial python scripts/probes/_sweep_compression_fixes.py \\
        --variants lightgbm_two_stage_combined --scales 3

Writes:
    models/_sweep_compression_fixes/{timestamp}/{summary,aggregate}.parquet
    models/<variant>/<config_hash>/scale_S{n}_within/{predictions,metrics}*
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
from src.modeling.gbm import (
    LGBMParams,
    VARIANT_CONSTRUCTORS,
    snapshot_params,
)

SCALE_TILE_PX = {0: 8, 1: 16, 2: 32, 3: 64, 4: 128}
SCHEME = "within_image_4fold"
TARGET_COL = "fractional_area"
DEFAULT_VARIANTS = (
    "lightgbm_two_stage",            # baseline
    "lightgbm_two_stage_balanced",
    "lightgbm_two_stage_weighted",
    "lightgbm_two_stage_gamma",
    "lightgbm_two_stage_combined",
)
DEFAULT_SCALES = (2, 3)   # S=32 and S=64 -- where the compression bites hardest
MODELS_ROOT = REPO_ROOT / "models"


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_one(
    variant: str, scale_idx: int, params: LGBMParams, *, dataset_dir: str,
) -> tuple[list[dict], dict, Path]:
    tile_size = SCALE_TILE_PX[scale_idx]
    snapshot = {
        "variant": variant,
        "task": "regression",
        "scheme": SCHEME,
        "dataset_dir": dataset_dir,
        "scale_idx": scale_idx,
        "tile_size_px": tile_size,
        "model": snapshot_params(variant, params),
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash
    out_dir = MODELS_ROOT / variant / cfg_hash / f"scale_S{tile_size}_within"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls = VARIANT_CONSTRUCTORS[variant]

    def factory():
        return cls(params=params)

    result = run_loio(
        factory,
        target_col=TARGET_COL,
        task="regression",
        scheme=SCHEME,
        scale_idx=scale_idx,
        dataset_dir=dataset_dir,
        snapshot=snapshot,
        verbose=False,
    )
    write_run_artifacts(result, out_dir)
    return result.per_fold_metrics, result.aggregate, out_dir


def _per_variant_compression(out_dir: Path) -> dict:
    """Read predictions.parquet, return per-bin {mean_true, mean_pred, ratio}.

    Computed pooled across folds (consistent with the scripts/probes/_diag_compression_mechanism.py
    diagnostic).
    """
    pred_path = out_dir / "predictions.parquet"
    if not pred_path.exists():
        return {}
    df = pd.read_parquet(pred_path)
    # Use the same bin edges as src/modeling/evaluate.py per_bin_rmse
    edges = [(-1e-12, 0.0, "zero"),
             (0.0, 1e-4, "0_to_1e-4"),
             (1e-4, 1e-3, "1e-4_to_1e-3"),
             (1e-3, 1e-2, "1e-3_to_1e-2"),
             (1e-2, 1.0, "1e-2_to_max")]
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
        rows[f"{name}__ratio"] = mp / mt if (mt > 0) else float("nan")
        rows[f"{name}__log10_ratio"] = float(np.log10(mp / mt)) if (mp > 0 and mt > 0) else float("nan")
    # Compression score: mean |log10(ratio)| across the 4 nonzero bins where it's defined.
    nz_log_abs = [
        abs(rows[f"{n}__log10_ratio"])
        for n in ("0_to_1e-4", "1e-4_to_1e-3", "1e-3_to_1e-2", "1e-2_to_max")
        if np.isfinite(rows.get(f"{n}__log10_ratio", np.nan))
    ]
    rows["compression_score"] = float(np.mean(nz_log_abs)) if nz_log_abs else float("nan")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    ap.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--dataset-dir", default="dataset_v2_dev")
    args = ap.parse_args()

    for v in args.variants:
        if v not in VARIANT_CONSTRUCTORS:
            raise SystemExit(f"unknown variant: {v!r}")

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "models" / "_sweep_compression_fixes" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sweep_meta.json").write_text(json.dumps({
        "kind": "compression_fixes",
        "dataset_dir": args.dataset_dir,
        "scheme": SCHEME,
        "timestamp": timestamp,
        "script": "_sweep_compression_fixes.py",
        "variants": list(args.variants),
        "scales": list(args.scales),
    }, indent=2), encoding="utf-8")

    runs = [(v, s) for v in args.variants for s in args.scales]
    print(f"Compression-fix sweep on {SCHEME} dataset_dir={args.dataset_dir}")
    print(f"{len(runs)} runs across {len(args.variants)} variants x {len(args.scales)} scales")
    print(f"Output: {out_dir}\n", flush=True)

    summary_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for i, (variant, scale_idx) in enumerate(runs, 1):
        tile_size = SCALE_TILE_PX[scale_idx]
        print(f"[{i:>2d}/{len(runs)}] {variant:<35s} S={tile_size:>2d} ...", flush=True)
        per_fold, aggregate, artifact_dir = run_one(
            variant, scale_idx, params, dataset_dir=args.dataset_dir,
        )
        for f in per_fold:
            drop = {"calibration_deciles", "per_bin_rmse"}
            row = {
                "variant": variant,
                "scale_idx": scale_idx,
                "tile_size_px": tile_size,
                **{k: v for k, v in f.items() if k not in drop},
                "held_out_obs_id": f["held_out_obs_ids"][0] if f["held_out_obs_ids"] else "",
            }
            row.pop("held_out_obs_ids", None)
            summary_rows.append(row)
        comp = _per_variant_compression(artifact_dir)
        agg_row = {
            "variant": variant,
            "scale_idx": scale_idx,
            "tile_size_px": tile_size,
            **aggregate,
            **comp,
            "artifact_dir": str(artifact_dir.relative_to(REPO_ROOT)),
        }
        aggregate_rows.append(agg_row)
        print(f"        rho={aggregate.get('spearman_rho_mean', float('nan')):+.4f}  "
              f"auc={aggregate.get('presence_auc_mean', float('nan')):.3f}  "
              f"compression={comp.get('compression_score', float('nan')):.3f}", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_df.to_parquet(out_dir / "summary.parquet", index=False)
    aggregate_df.to_parquet(out_dir / "aggregate.parquet", index=False)

    print(f"\n=== Aggregate (composite metric: lower compression = less squash) ===")
    cols = [
        "variant", "scale_idx", "tile_size_px",
        "spearman_rho_mean", "presence_auc_mean",
        "compression_score",
        "zero__mean_pred", "1e-2_to_max__mean_true", "1e-2_to_max__mean_pred",
    ]
    cols = [c for c in cols if c in aggregate_df.columns]
    print(aggregate_df[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWrote: {out_dir / 'aggregate.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
