"""Stage 6a dev sweep: spatial-context neighbour features on top of P1+P2 baseline.

Runs `lightgbm_two_stage_balanced` (P1) with `target_col=boulder_count` (P2) on the
neighbour-augmented scheme `within_image_4fold_nbr` and compares against the
existing P1+P2 baseline (same variant + target, no neighbour features) on
`within_image_4fold`.

Both runs use the dev within-image 4-fold scheme (5 images x 4 quadrants = 20 folds)
on `dataset_v2_dev`.  The fold definitions are byte-identical between the two
packaged dirs by construction (see `scripts/run_stage6a_repackage.py`), so the
only difference is the X-matrix column set.

Acceptance criterion (PROMOTION_QUEUE.md Stage 6a):
  - Spearman rho lift +>= 0.05 over P1+P2 baseline, AND
  - PR-AUC lift +>= 0.03 over P1+P2 baseline.
Both must clear for the dev result to be considered a pass.

Usage:
    conda run -n geospatial python scripts/probes/_sweep_stage6a.py

Output:
    models/_sweep_stage6a/{timestamp}/aggregate.parquet
    models/_sweep_stage6a/{timestamp}/summary.parquet
    models/_sweep_stage6a/{timestamp}/sweep_meta.json
    models/_sweep_stage6a/{timestamp}/result.md   # baseline-vs-nbr table + verdict
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.modeling.evaluate import run_loio, write_run_artifacts  # noqa: E402
from src.modeling.gbm import LGBMParams, VARIANT_CONSTRUCTORS, snapshot_params  # noqa: E402
from src.modeling.loaders import Fold, iter_loio_folds  # noqa: E402

SCALE_TILE_PX = {0: 8, 1: 16, 2: 32, 3: 64, 4: 128}
VARIANT = "lightgbm_two_stage_balanced"
TARGET = "boulder_count"
DEFAULT_SCALES = (3,)  # S=64
BASELINE_SCHEME = "within_image_4fold"
NBR_SCHEME = "within_image_4fold_nbr"
MODELS_ROOT = REPO_ROOT / "models"
CTX_M = 5.0


def _meaningful_threshold(scale_idx: int) -> float:
    """At-inference boulder-rich threshold for boulder_count.

    Mirrors `_sweep_target_reformulation._meaningful_threshold(boulder_count, ...)`:
    50 boulders per tile is the operational `boulder-rich` cut.
    """
    return 50.0


def _add_log_target(fold: Fold) -> None:
    for df in (fold.y_train, fold.y_test):
        if "log_boulder_count" not in df.columns:
            df["log_boulder_count"] = np.log1p(
                df["boulder_count"].to_numpy(dtype=np.float64),
            )


def _wrapped_fold_iter(scheme: str, scale_idx: int, dataset_dir: str):
    def _it():
        for f in iter_loio_folds(scheme, scale_idx=scale_idx, dataset_dir=dataset_dir):
            _add_log_target(f)
            yield f
    return _it


def _config_hash(snapshot: dict) -> str:
    blob = json.dumps(snapshot, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def run_one(
    *, scheme: str, scale_idx: int, params: LGBMParams, dataset_dir: str,
    out_root: Path,
) -> tuple[list[dict], dict, Path]:
    tile_size = SCALE_TILE_PX[scale_idx]
    snapshot = {
        "variant": VARIANT,
        "task": "regression",
        "target_col": TARGET,
        "scheme": scheme,
        "dataset_dir": dataset_dir,
        "scale_idx": scale_idx,
        "tile_size_px": tile_size,
        "model": snapshot_params(VARIANT, params),
        "meaningful_threshold": _meaningful_threshold(scale_idx),
    }
    cfg_hash = _config_hash(snapshot)
    snapshot["config_hash"] = cfg_hash
    out_dir = out_root / scheme / cfg_hash / f"scale_S{tile_size}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cls = VARIANT_CONSTRUCTORS[VARIANT]

    def factory():
        return cls(params=params)

    import src.modeling.evaluate as _ev
    orig = _ev.per_fold_metrics
    mt = _meaningful_threshold(scale_idx)

    def _patched(y_true, y_pred, *, held_out_obs_ids, meaningful_threshold=mt):
        return orig(y_true, y_pred, held_out_obs_ids=held_out_obs_ids,
                    meaningful_threshold=meaningful_threshold)

    _ev.per_fold_metrics = _patched
    try:
        result = run_loio(
            factory,
            target_col=TARGET,
            task="regression",
            scheme=scheme,
            scale_idx=scale_idx,
            dataset_dir=dataset_dir,
            fold_iter=_wrapped_fold_iter(scheme, scale_idx, dataset_dir),
            snapshot=snapshot,
            verbose=False,
        )
    finally:
        _ev.per_fold_metrics = orig
    write_run_artifacts(result, out_dir)
    return result.per_fold_metrics, result.aggregate, out_dir


def _format_diff_row(metric: str, base: float, nbr: float, fmt: str = "+.4f") -> str:
    delta = nbr - base
    return (
        f"| {metric:<32s} | {base:{fmt}} | {nbr:{fmt}} | {delta:+.4f} |"
    )


def _build_result_md(
    timestamp: str, dataset_dir: str, scales: list[int],
    rows: list[dict],
) -> str:
    """Produce a comparison markdown summary.  One row per scale x scheme.

    `rows` carries dicts with keys: scheme, scale_idx, tile_size_px,
    spearman_rho_mean, presence_auc_mean, pr_auc_mean,
    normalised_lift_meaningful_mean, precision_at_top_5pct_mean.
    """
    buf = io.StringIO()
    print(f"# Stage 6a dev sweep -- {timestamp}\n", file=buf)
    print(
        f"Dataset: `{dataset_dir}`  | Scheme: `{BASELINE_SCHEME}` vs `{NBR_SCHEME}`  | "
        f"Variant: `{VARIANT}`  | Target: `{TARGET}`\n",
        file=buf,
    )
    by_scale: dict[int, dict[str, dict]] = {}
    for r in rows:
        by_scale.setdefault(int(r["scale_idx"]), {})[r["scheme"]] = r
    for scale_idx in sorted(by_scale):
        tile_size = SCALE_TILE_PX[scale_idx]
        print(f"## S = {tile_size} (scale_idx={scale_idx})\n", file=buf)
        base = by_scale[scale_idx].get(BASELINE_SCHEME)
        nbr = by_scale[scale_idx].get(NBR_SCHEME)
        if base is None or nbr is None:
            print(
                f"_(missing scheme: base={base is None}, nbr={nbr is None})_\n", file=buf,
            )
            continue
        print(
            "| metric                           | P1+P2 base | P1+P2 + Stage 6a | "
            "Delta |",
            file=buf,
        )
        print(
            "|----------------------------------|-----------:|-----------------:|"
            "------:|",
            file=buf,
        )
        for label, key in [
            ("Spearman rho", "spearman_rho_mean"),
            ("presence AUC (ROC)", "presence_auc_mean"),
            ("meaningful AUC (ROC)", "meaningful_auc_mean"),
            ("PR-AUC", "pr_auc_mean"),
            ("normalised lift @top-K", "normalised_lift_meaningful_mean"),
            ("precision @top-5%", "precision_at_top_5pct_mean"),
            ("recall @top-5%", "recall_at_top_5pct_mean"),
        ]:
            if key in base and key in nbr:
                print(_format_diff_row(label, float(base[key]), float(nbr[key])),
                      file=buf)
        # Acceptance verdict
        rho_delta = float(nbr["spearman_rho_mean"]) - float(base["spearman_rho_mean"])
        pr_delta = float(nbr["pr_auc_mean"]) - float(base["pr_auc_mean"])
        rho_pass = rho_delta >= 0.05
        pr_pass = pr_delta >= 0.03
        verdict = "PASS" if (rho_pass and pr_pass) else "FAIL"
        print(f"\n**Acceptance** (Stage 6a):  "
              f"Spearman delta = {rho_delta:+.4f} ({'>=' if rho_pass else '<'} +0.05)  AND  "
              f"PR-AUC delta = {pr_delta:+.4f} ({'>=' if pr_pass else '<'} +0.03)  -->  "
              f"**{verdict}**\n",
              file=buf)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--dataset-dir", default="dataset_v2_dev")
    ap.add_argument(
        "--schemes", nargs="+", default=[BASELINE_SCHEME, NBR_SCHEME],
        help="Which schemes to run (default: both -- baseline then Stage 6a)",
    )
    args = ap.parse_args()

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = MODELS_ROOT / "_sweep_stage6a" / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sweep_meta.json").write_text(json.dumps({
        "kind": "stage6a_dev_sweep",
        "dataset_dir": args.dataset_dir,
        "variant": VARIANT,
        "target": TARGET,
        "scales": list(args.scales),
        "schemes": list(args.schemes),
        "timestamp": timestamp,
        "script": "_sweep_stage6a.py",
    }, indent=2), encoding="utf-8")

    runs = [(scheme, scale_idx) for scheme in args.schemes for scale_idx in args.scales]
    print(f"Stage 6a dev sweep on {args.dataset_dir}  "
          f"variant={VARIANT}  target={TARGET}")
    print(f"  schemes:  {args.schemes}")
    print(f"  scales:   {args.scales}")
    print(f"  output:   {out_root}\n", flush=True)

    summary_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for i, (scheme, scale_idx) in enumerate(runs, 1):
        tile_size = SCALE_TILE_PX[scale_idx]
        print(f"[{i:>2d}/{len(runs)}] scheme={scheme:<28s} S={tile_size:>3d} ...",
              flush=True)
        per_fold, aggregate, artifact_dir = run_one(
            scheme=scheme, scale_idx=scale_idx, params=params,
            dataset_dir=args.dataset_dir, out_root=out_root,
        )
        for f in per_fold:
            drop = {"calibration_deciles", "per_bin_rmse"}
            row = {
                "scheme": scheme, "scale_idx": scale_idx, "tile_size_px": tile_size,
                **{k: v for k, v in f.items() if k not in drop},
                "held_out_obs_id": f["held_out_obs_ids"][0]
                if f["held_out_obs_ids"] else "",
            }
            row.pop("held_out_obs_ids", None)
            summary_rows.append(row)
        agg_row = {
            "scheme": scheme, "scale_idx": scale_idx, "tile_size_px": tile_size,
            **aggregate,
            "artifact_dir": str(artifact_dir.relative_to(REPO_ROOT)),
        }
        aggregate_rows.append(agg_row)
        print(f"        rho={aggregate.get('spearman_rho_mean', float('nan')):+.4f}  "
              f"presence_auc={aggregate.get('presence_auc_mean', float('nan')):.3f}  "
              f"PR-AUC={aggregate.get('pr_auc_mean', float('nan')):.3f}  "
              f"lift_norm={aggregate.get('normalised_lift_meaningful_mean', float('nan')):.3f}  "
              f"prec@5%={aggregate.get('precision_at_top_5pct_mean', float('nan')):.3f}",
              flush=True)

    summary_df = pd.DataFrame(summary_rows)
    aggregate_df = pd.DataFrame(aggregate_rows)
    summary_df.to_parquet(out_root / "summary.parquet", index=False)
    aggregate_df.to_parquet(out_root / "aggregate.parquet", index=False)

    md = _build_result_md(timestamp, args.dataset_dir, list(args.scales), aggregate_rows)
    (out_root / "result.md").write_text(md, encoding="utf-8")
    print("\n" + md, flush=True)
    print(f"Wrote: {out_root / 'aggregate.parquet'}")
    print(f"Wrote: {out_root / 'result.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
