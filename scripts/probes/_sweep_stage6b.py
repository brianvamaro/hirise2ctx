"""Stage 6b dev sweep: CTX-source illumination features on top of P1+P2 baseline.

Runs `lightgbm_two_stage_balanced` (P1) with `target_col=boulder_count` (P2) on
the Stage-6b-augmented packaged dir (`{scheme}_ctx_illum`) and compares against
the existing P1+P2 baseline on the unaugmented scheme.

Two schemes are tested by default:

  * `within_image_4fold` vs `within_image_4fold_ctx_illum`
      Mirrors the Stage 6a comparison. Predicted to show ~no lift since the
      illumination features are near-constant within an image (most tiles sit in
      one CTX source).
  * `loio_nfold` vs `loio_nfold_ctx_illum`
      Structurally-correct test for cross-image features on dev. 5 folds (one
      held-out image each), so the model can learn "high CTX incidence -> bad"
      from 4 training images and predict on the 5th. Statistically thin (n=5)
      relative to the full-v2 LOIO (n=38); H3 mechanism check defers to full v2.

Acceptance criterion (PROMOTION_QUEUE.md Stage 6b):
  - PR-AUC lift +>= 0.03 over P1+P2 baseline (on full-v2 LOIO, not dev), AND
  - per-image AUC <-> tile-mean CTX_IncidenceAngle correlation significantly
    negative (rho < -0.30, p < 0.05) across the 38 v2 images.
On the 5-image dev set, the latter check is statistically meaningless; we
report Spearman + PR-AUC lift only.

Usage:
    conda run -n geospatial python scripts/probes/_sweep_stage6b.py

Output:
    models/_sweep_stage6b/{timestamp}/aggregate.parquet
    models/_sweep_stage6b/{timestamp}/summary.parquet
    models/_sweep_stage6b/{timestamp}/sweep_meta.json
    models/_sweep_stage6b/{timestamp}/result.md   # comparison table
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
DEFAULT_SCALES = (3,)  # S=64 canonical
DEFAULT_SCHEME_PAIRS = (
    ("within_image_4fold", "within_image_4fold_ctx_illum"),
    ("loio_nfold", "loio_nfold_ctx_illum"),
)
MODELS_ROOT = REPO_ROOT / "models"


def _meaningful_threshold(scale_idx: int) -> float:
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


def _format_diff_row(metric: str, base: float, ill: float, fmt: str = "+.4f") -> str:
    delta = ill - base
    return (
        f"| {metric:<32s} | {base:{fmt}} | {ill:{fmt}} | {delta:+.4f} |"
    )


def _build_result_md(
    timestamp: str, dataset_dir: str, scale_idx: int, tile_size: int,
    by_pair: dict[tuple[str, str], dict[str, dict]],
) -> str:
    buf = io.StringIO()
    print(f"# Stage 6b dev sweep -- {timestamp}\n", file=buf)
    print(
        f"Dataset: `{dataset_dir}`  | Variant: `{VARIANT}`  | Target: `{TARGET}`  "
        f"| Scale: S={tile_size} (scale_idx={scale_idx})\n",
        file=buf,
    )
    for (baseline, illum), schemes_data in by_pair.items():
        base = schemes_data.get(baseline)
        ill = schemes_data.get(illum)
        if base is None or ill is None:
            print(
                f"## {baseline} vs {illum}\n_(missing: "
                f"base={base is None}, illum={ill is None})_\n",
                file=buf,
            )
            continue
        print(f"## `{baseline}` vs `{illum}`\n", file=buf)
        print(
            "| metric                           | P1+P2 base | P1+P2 + Stage 6b | "
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
            if key in base and key in ill:
                print(_format_diff_row(label, float(base[key]), float(ill[key])),
                      file=buf)
        rho_delta = float(ill["spearman_rho_mean"]) - float(base["spearman_rho_mean"])
        pr_delta = float(ill["pr_auc_mean"]) - float(base["pr_auc_mean"])
        rho_pass = rho_delta >= 0.05
        pr_pass = pr_delta >= 0.03
        verdict = "PASS" if (rho_pass and pr_pass) else "FAIL"
        print(f"\n**Acceptance** (Stage 6b proxy; full-v2 H3 mechanism check "
              f"defers): Spearman delta = {rho_delta:+.4f} "
              f"({'>=' if rho_pass else '<'} +0.05)  AND  "
              f"PR-AUC delta = {pr_delta:+.4f} "
              f"({'>=' if pr_pass else '<'} +0.03)  -->  **{verdict}**\n",
              file=buf)
    print(
        "_Caveat: dev has 5 images; LOIO has only n=5 folds. The H3 mechanism "
        "check (per-image AUC <-> CTX_IncidenceAngle correlation) requires the "
        "full-v2 38-image LOIO and is not run here._\n",
        file=buf,
    )
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scales", nargs="+", type=int, default=list(DEFAULT_SCALES))
    ap.add_argument("--n-estimators", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--early-stopping-rounds", type=int, default=40)
    ap.add_argument("--dataset-dir", default="dataset_v2_dev")
    ap.add_argument(
        "--scheme-pairs", nargs="+", default=None,
        help="Optional: override baseline,illum pairs (comma-separated, repeatable). "
             "Default: within_image_4fold,within_image_4fold_ctx_illum and "
             "loio_nfold,loio_nfold_ctx_illum.",
    )
    args = ap.parse_args()

    if args.scheme_pairs:
        scheme_pairs = tuple(tuple(p.split(",")) for p in args.scheme_pairs)
    else:
        scheme_pairs = DEFAULT_SCHEME_PAIRS

    params = LGBMParams(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_root = MODELS_ROOT / "_sweep_stage6b" / timestamp
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "sweep_meta.json").write_text(json.dumps({
        "kind": "stage6b_dev_sweep",
        "dataset_dir": args.dataset_dir,
        "variant": VARIANT,
        "target": TARGET,
        "scales": list(args.scales),
        "scheme_pairs": [list(p) for p in scheme_pairs],
        "timestamp": timestamp,
        "script": "_sweep_stage6b.py",
    }, indent=2), encoding="utf-8")

    all_schemes = []
    for baseline, illum in scheme_pairs:
        all_schemes.append(baseline)
        all_schemes.append(illum)

    runs = [(scheme, scale_idx) for scheme in all_schemes for scale_idx in args.scales]
    print(f"Stage 6b dev sweep on {args.dataset_dir}  "
          f"variant={VARIANT}  target={TARGET}")
    print(f"  scheme_pairs:  {list(scheme_pairs)}")
    print(f"  scales:        {args.scales}")
    print(f"  output:        {out_root}\n", flush=True)

    summary_rows: list[dict] = []
    aggregate_rows: list[dict] = []
    for i, (scheme, scale_idx) in enumerate(runs, 1):
        tile_size = SCALE_TILE_PX[scale_idx]
        print(f"[{i:>2d}/{len(runs)}] scheme={scheme:<36s} S={tile_size:>3d} ...",
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

    # Build comparison MD per scheme-pair, per scale
    for scale_idx in args.scales:
        tile_size = SCALE_TILE_PX[scale_idx]
        by_pair: dict[tuple[str, str], dict[str, dict]] = {}
        for baseline, illum in scheme_pairs:
            pair_key = (baseline, illum)
            by_pair[pair_key] = {}
            for r in aggregate_rows:
                if r["scale_idx"] != scale_idx:
                    continue
                if r["scheme"] == baseline or r["scheme"] == illum:
                    by_pair[pair_key][r["scheme"]] = r
        md = _build_result_md(timestamp, args.dataset_dir, scale_idx, tile_size, by_pair)
        out_md = out_root / f"result_S{tile_size}.md"
        out_md.write_text(md, encoding="utf-8")
        print("\n" + md, flush=True)
        print(f"Wrote: {out_md}")
    print(f"Wrote: {out_root / 'aggregate.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
