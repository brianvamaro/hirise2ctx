"""Paired per-fold delta analysis for the W0 S=64 matrix (PLAN_ModelUsability W0).

Two judgment calls need per-fold pairing, not aggregate means:
  1. P2 promotion: does boulder_count's small aggregate Spearman dip vs
     fractional_area exceed fold noise? (acceptance says "should not regress")
  2. Single-stage vs two-stage on boulder_count: is the hurdle's Spearman edge
     real, given operational metrics tie?

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/probes/_w0_paired_deltas.py <sweep_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ("spearman_rho", "pr_auc", "normalised_lift", "precision_at_top_5pct")


def paired(df: pd.DataFrame, cell_a: dict, cell_b: dict, label: str) -> None:
    """Per-fold paired deltas (B - A) with Wilcoxon signed-rank p."""
    def cell(d):
        m = pd.Series(True, index=df.index)
        for k, v in d.items():
            m &= df[k] == v
        return df[m].set_index("held_out_obs_id")

    a, b = cell(cell_a), cell(cell_b)
    common = a.index.intersection(b.index)
    print(f"\n== {label}  (n={len(common)} paired folds) ==")
    for metric in METRICS:
        if metric not in df.columns:
            continue
        da, db = a.loc[common, metric], b.loc[common, metric]
        ok = da.notna() & db.notna()
        delta = (db[ok] - da[ok]).to_numpy()
        if delta.size < 5:
            print(f"  {metric:<24s} insufficient folds")
            continue
        try:
            w_p = stats.wilcoxon(delta, zero_method="wilcox").pvalue
        except ValueError:
            w_p = float("nan")
        print(f"  {metric:<24s} mean_delta={delta.mean():+.4f}"
              f"  median={np.median(delta):+.4f}  win_rate={(delta > 0).mean():.2f}"
              f"  wilcoxon_p={w_p:.4f}")


def main() -> int:
    sweep_dir = Path(sys.argv[1])
    df = pd.read_parquet(sweep_dir / "summary.parquet")

    bal, van, sgl = ("lightgbm_two_stage_balanced", "lightgbm_two_stage", "lightgbm_log1p_huber")
    paired(df, {"variant": bal, "target_col": "fractional_area"},
           {"variant": bal, "target_col": "boulder_count"},
           "P2: fractional_area -> boulder_count (balanced)")
    paired(df, {"variant": van, "target_col": "boulder_count"},
           {"variant": bal, "target_col": "boulder_count"},
           "P1: vanilla -> balanced (boulder_count)")
    paired(df, {"variant": bal, "target_col": "boulder_count"},
           {"variant": sgl, "target_col": "boulder_count"},
           "Hurdle test: two_stage_balanced -> log1p_huber (boulder_count)")
    paired(df, {"variant": bal, "target_col": "fractional_area"},
           {"variant": sgl, "target_col": "fractional_area"},
           "Hurdle test: two_stage_balanced -> log1p_huber (fractional_area)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
