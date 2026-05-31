"""H3 mechanism check for Stage 6b: per-image baseline AUC vs mean CTX_INCIDENCE.

H3 hypothesis (PROMOTION_QUEUE.md Stage 6b): on per-image anti-signal failure cases,
oblique CTX-source illumination causes ``shadow_fraction`` to mis-read
ripple-field / crater-rim shadows as boulders, dragging per-image AUC the wrong way.
The prediction: across the 38 v2 images, Spearman rho(per_image_AUC,
mean_CTX_INCIDENCE) should be **significantly negative** (rho < -0.30, p < 0.05).

This probe also reports:
  * Per-image Stage 6b deltas (PR-AUC, Spearman, presence AUC) -- which images
    benefited or regressed.
  * Top-line aggregate delta on full v2 LOIO.

Output written as a markdown table to
``scripts/probes/_diag_stage6b_h3_check.md``.

Usage:
    conda run -n geospatial python scripts/probes/_diag_stage6b_h3_check.py \
        --sweep-dir models/_sweep_stage6b/{timestamp}
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

BASELINE_SCHEME = "loio_nfold"
ILLUM_SCHEME = "loio_nfold_ctx_illum"
DEFAULT_SCALE = 3  # S=64


def _per_image_features(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted((dataset_dir / "features_ctx_illum").glob("*.parquet")):
        obs = p.stem
        df = pd.read_parquet(p, columns=["tile_size_px", "ctx_incidence_mean",
                                          "ctx_n_sources",
                                          "ctx_dominant_source_fraction"])
        df64 = df[df["tile_size_px"] == 64]
        if len(df64) == 0:
            continue
        rows.append({
            "ObsId": obs,
            "mean_ctx_incidence": float(df64["ctx_incidence_mean"].mean()),
            "median_ctx_incidence": float(df64["ctx_incidence_mean"].median()),
            "std_ctx_incidence": float(df64["ctx_incidence_mean"].std(ddof=0)),
            "mean_n_sources": float(df64["ctx_n_sources"].mean()),
            "dominant_source_frac_mean": float(
                df64["ctx_dominant_source_fraction"].mean()
            ),
        })
    return pd.DataFrame(rows)


def _per_image_metrics(sweep_dir: Path, scheme: str, scale_idx: int) -> pd.DataFrame:
    summary = pd.read_parquet(sweep_dir / "summary.parquet")
    sub = summary[(summary["scheme"] == scheme) & (summary["scale_idx"] == scale_idx)].copy()
    if "held_out_obs_id" not in sub.columns:
        raise RuntimeError(
            f"summary.parquet for {scheme} missing held_out_obs_id; cols={sub.columns.tolist()}"
        )
    keep_cols = [c for c in sub.columns if c not in (
        "scheme", "scale_idx", "tile_size_px", "calibration_deciles", "per_bin_rmse",
    )]
    sub = sub[keep_cols].rename(columns={"held_out_obs_id": "ObsId"})
    return sub


def _spearman_block(df: pd.DataFrame, features: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for feat in features:
        for met in metrics:
            sub = df[[feat, met]].dropna()
            if len(sub) < 5:
                rows.append({"feature": feat, "metric": met, "n": len(sub),
                             "rho": float("nan"), "p_value": float("nan")})
                continue
            rho, pval = stats.spearmanr(sub[feat], sub[met])
            rows.append({"feature": feat, "metric": met, "n": len(sub),
                         "rho": float(rho), "p_value": float(pval)})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep-dir", required=True,
                    help="models/_sweep_stage6b/{timestamp} containing summary.parquet")
    ap.add_argument("--dataset-dir", default="dataset_v2",
                    help="Dataset root with features_ctx_illum/ (default: dataset_v2)")
    ap.add_argument("--scale-idx", type=int, default=DEFAULT_SCALE)
    args = ap.parse_args()

    sweep_dir = Path(args.sweep_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()

    feats = _per_image_features(dataset_dir)
    print(f"Per-image features: {len(feats)} images.")
    print(feats.head().to_string(index=False))
    print()

    base_metrics = _per_image_metrics(sweep_dir, BASELINE_SCHEME, args.scale_idx)
    illum_metrics = _per_image_metrics(sweep_dir, ILLUM_SCHEME, args.scale_idx)
    print(f"Baseline per-image: {len(base_metrics)} folds.")
    print(f"Illum    per-image: {len(illum_metrics)} folds.")

    # Identify which numeric metric columns exist in both -- for delta + correlation.
    # Skip booleans (e.g. is_specificity_only) -- subtraction is undefined.
    common_metrics = [
        c for c in base_metrics.columns
        if c in illum_metrics.columns and c != "ObsId"
        and pd.api.types.is_numeric_dtype(base_metrics[c])
        and not pd.api.types.is_bool_dtype(base_metrics[c])
    ]
    print(f"Common metric columns: {len(common_metrics)}")

    # H3 correlation: baseline AUC vs mean CTX_INCIDENCE across images
    join = feats.merge(base_metrics[["ObsId"] + common_metrics], on="ObsId", how="inner")
    print(f"\nJoined ({len(join)} images):")
    print(join[["ObsId", "mean_ctx_incidence", "mean_n_sources",
                "presence_auc", "pr_auc", "spearman_rho"]].head(6).to_string(index=False))

    h3_metrics = [m for m in (
        "presence_auc", "pr_auc", "spearman_rho",
        "normalised_lift_meaningful", "precision_at_top_5pct",
    ) if m in join.columns]
    corr = _spearman_block(
        join,
        features=["mean_ctx_incidence", "std_ctx_incidence", "mean_n_sources",
                  "dominant_source_frac_mean"],
        metrics=h3_metrics,
    )
    print("\nH3 correlations (Spearman rho of feature vs baseline metric across images):")
    pivot = corr.pivot(index="feature", columns="metric", values="rho")
    print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"))

    print("\nSignificant correlations (p < 0.05):")
    sig = corr[corr["p_value"] < 0.05].sort_values("p_value")
    if len(sig) > 0:
        print(sig.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        print("  (none)")

    # Per-image deltas
    delta = base_metrics[["ObsId"]].copy()
    for m in common_metrics:
        delta[f"delta_{m}"] = illum_metrics.set_index("ObsId").loc[
            base_metrics["ObsId"], m
        ].to_numpy() - base_metrics[m].to_numpy()
    delta = delta.merge(feats[["ObsId", "mean_ctx_incidence", "mean_n_sources"]],
                        on="ObsId", how="left")
    print("\nPer-image Stage 6b deltas (top winners by delta_pr_auc):")
    delta_sorted = delta.sort_values("delta_pr_auc", ascending=False) \
        if "delta_pr_auc" in delta.columns else delta
    cols = ["ObsId", "mean_ctx_incidence", "mean_n_sources",
            "delta_pr_auc", "delta_spearman_rho",
            "delta_presence_auc", "delta_precision_at_top_5pct"]
    print(delta_sorted[[c for c in cols if c in delta_sorted.columns]]
          .head(8).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\n... and the bottom (regressions):")
    print(delta_sorted[[c for c in cols if c in delta_sorted.columns]]
          .tail(8).to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    # Markdown writeup
    md = io.StringIO()
    print("# Stage 6b H3 mechanism check -- full-v2 LOIO\n", file=md)
    print(f"Sweep: `{sweep_dir.relative_to(REPO_ROOT)}`  | Dataset: `{dataset_dir.name}`  | "
          f"Scale: S=64\n", file=md)

    print("## Acceptance summary\n", file=md)
    pr_auc_base = base_metrics["pr_auc"].mean() if "pr_auc" in base_metrics else float("nan")
    pr_auc_illum = illum_metrics["pr_auc"].mean() if "pr_auc" in illum_metrics else float("nan")
    rho_base = base_metrics["spearman_rho"].mean() if "spearman_rho" in base_metrics else float("nan")
    rho_illum = illum_metrics["spearman_rho"].mean() if "spearman_rho" in illum_metrics else float("nan")
    delta_pr = pr_auc_illum - pr_auc_base
    delta_rho = rho_illum - rho_base
    print(f"- PR-AUC mean: baseline {pr_auc_base:.4f} -> +Stage 6b {pr_auc_illum:.4f}  "
          f"(delta {delta_pr:+.4f}; pass = +>= 0.03)", file=md)
    print(f"- Spearman mean: baseline {rho_base:+.4f} -> +Stage 6b {rho_illum:+.4f}  "
          f"(delta {delta_rho:+.4f})", file=md)
    inc_pres = corr[(corr["feature"] == "mean_ctx_incidence")
                    & (corr["metric"] == "presence_auc")]
    inc_prauc = corr[(corr["feature"] == "mean_ctx_incidence")
                    & (corr["metric"] == "pr_auc")]
    inc_rho = corr[(corr["feature"] == "mean_ctx_incidence")
                    & (corr["metric"] == "spearman_rho")]
    for label, row in [("presence_auc", inc_pres), ("pr_auc", inc_prauc),
                       ("spearman_rho", inc_rho)]:
        if len(row):
            r = row.iloc[0]
            sig_mark = "**" if r["p_value"] < 0.05 else ""
            print(f"- mean_ctx_incidence vs per-image {label}: rho = {sig_mark}"
                  f"{r['rho']:+.3f}{sig_mark} (p={r['p_value']:.3f}, n={int(r['n'])})  "
                  f"-- H3 prediction: rho < -0.30",
                  file=md)
    print("", file=md)

    print("## H3 correlation table\n", file=md)
    print("Spearman rho of per-image feature vs per-image **baseline** metric across "
          f"{len(join)} images. ** marks p < 0.05.\n", file=md)
    print("```", file=md)
    print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"), file=md)
    print("```\n", file=md)

    print("## Per-image deltas (top winners)\n", file=md)
    print("```", file=md)
    print(delta_sorted[[c for c in cols if c in delta_sorted.columns]]
          .head(10).to_string(index=False, float_format=lambda v: f"{v:+.3f}"),
          file=md)
    print("```\n", file=md)
    print("## Per-image deltas (largest regressions)\n", file=md)
    print("```", file=md)
    print(delta_sorted[[c for c in cols if c in delta_sorted.columns]]
          .tail(10).to_string(index=False, float_format=lambda v: f"{v:+.3f}"),
          file=md)
    print("```\n", file=md)

    out_md = Path(__file__).with_suffix(".md")
    out_md.write_text(md.getvalue(), encoding="utf-8")
    print(f"\nMarkdown -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
