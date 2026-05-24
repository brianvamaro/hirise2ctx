"""Recommend two-stage binary thresholds.

The CLAUDE.md placeholders (binary_area_threshold=0.005, binary_count_threshold=5) are
miscalibrated against each other: per DECISIONS.md 2026-05-23, they yield 5,504 area-only
positives vs 2 count-only positives at the finest scale. The Week 3 two-stage hurdle (per
PLAN_modeling.md §2 / §10.6) needs ONE definition of "positive tile" that is internally
consistent.

This probe loads all dataset/labels/{ObsId}.parquet, reports the joint distribution of
fractional_area and boulder_count at every scale, quantifies the placeholder disagreement,
and recommends matched (area_threshold, count_threshold) pairs at three plausible
positive-class fractions plus the strict "any boulder" rule.

Standalone — no project imports beyond pandas/numpy. Runs in a few seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = REPO_ROOT / "dataset" / "labels"


def load_all_labels() -> pd.DataFrame:
    parts = []
    for p in sorted(LABELS_DIR.glob("*.parquet")):
        df = pd.read_parquet(
            p, columns=["obs_id", "scale_idx", "tile_size_px", "fractional_area", "boulder_count", "boulder_area", "tile_area"]
        )
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def per_scale_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scale_idx, tile_size_px), g in df.groupby(["scale_idx", "tile_size_px"]):
        n = len(g)
        fa = g["fractional_area"].to_numpy()
        bc = g["boulder_count"].to_numpy()
        rows.append({
            "scale_idx": scale_idx,
            "tile_size_px": tile_size_px,
            "n_tiles": n,
            "n_pos_strict_fa_gt_0": int((fa > 0).sum()),
            "frac_pos_strict": float((fa > 0).mean()),
            "n_pos_area_0.005": int((fa >= 0.005).sum()),
            "frac_pos_area_0.005": float((fa >= 0.005).mean()),
            "n_pos_count_5": int((bc >= 5).sum()),
            "frac_pos_count_5": float((bc >= 5).mean()),
            "fa_max": float(fa.max()),
            "bc_max": int(bc.max()),
        })
    return pd.DataFrame(rows).sort_values("scale_idx").reset_index(drop=True)


def placeholder_disagreement(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scale_idx, tile_size_px), g in df.groupby(["scale_idx", "tile_size_px"]):
        a = (g["fractional_area"] >= 0.005).to_numpy()
        c = (g["boulder_count"] >= 5).to_numpy()
        both = int((a & c).sum())
        area_only = int((a & ~c).sum())
        count_only = int((~a & c).sum())
        neither = int((~a & ~c).sum())
        union = both + area_only + count_only
        jaccard = (both / union) if union > 0 else float("nan")
        rows.append({
            "scale_idx": scale_idx,
            "tile_size_px": tile_size_px,
            "both_positive": both,
            "area_only": area_only,
            "count_only": count_only,
            "neither": neither,
            "jaccard_at_placeholders": jaccard,
        })
    return pd.DataFrame(rows).sort_values("scale_idx").reset_index(drop=True)


def match_count_threshold_to_target(bc: np.ndarray, target_frac: float) -> tuple[int, float]:
    """Smallest k such that frac(bc >= k) <= target_frac. Returns (k, achieved_frac)."""
    if target_frac >= 1.0:
        return 0, 1.0
    # Sweep candidate ks; bc is small-valued so linear sweep is fine.
    ks = np.arange(0, int(bc.max()) + 2)
    fracs = np.array([(bc >= k).mean() for k in ks])
    # The smallest k whose positive fraction <= target.
    feasible = np.where(fracs <= target_frac)[0]
    if len(feasible) == 0:
        return int(ks[-1]), float(fracs[-1])
    k = int(ks[feasible[0]])
    return k, float(fracs[feasible[0]])


def match_area_threshold_to_target(fa: np.ndarray, target_frac: float) -> tuple[float, float]:
    """Smallest threshold t such that frac(fa >= t) <= target_frac. Returns (t, achieved_frac)."""
    if target_frac >= 1.0:
        return 0.0, 1.0
    fa_sorted = np.sort(fa)
    n = len(fa_sorted)
    # Take the (1 - target_frac) quantile of fa, then nudge up to the next distinct value.
    q_idx = int(np.ceil((1 - target_frac) * n))
    q_idx = min(max(q_idx, 0), n - 1)
    t = float(fa_sorted[q_idx])
    achieved = float((fa >= t).mean())
    return t, achieved


def recommend_matched_thresholds(df: pd.DataFrame, scale_idx: int, target_fracs: list[float]) -> pd.DataFrame:
    g = df[df["scale_idx"] == scale_idx]
    fa = g["fractional_area"].to_numpy()
    bc = g["boulder_count"].to_numpy()
    rows = []
    for tf in target_fracs:
        t_area, frac_area = match_area_threshold_to_target(fa, tf)
        k_count, frac_count = match_count_threshold_to_target(bc, tf)
        a_pos = fa >= t_area
        c_pos = bc >= k_count
        both = int((a_pos & c_pos).sum())
        union = int((a_pos | c_pos).sum())
        jaccard = (both / union) if union > 0 else float("nan")
        rows.append({
            "target_pos_frac": tf,
            "area_threshold": t_area,
            "area_pos_frac_achieved": frac_area,
            "count_threshold": k_count,
            "count_pos_frac_achieved": frac_count,
            "jaccard_after_matching": jaccard,
        })
    return pd.DataFrame(rows)


def fa_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scale_idx, tile_size_px), g in df.groupby(["scale_idx", "tile_size_px"]):
        fa = g["fractional_area"].to_numpy()
        fa_nonzero = fa[fa > 0]
        rows.append({
            "scale_idx": scale_idx,
            "tile_size_px": tile_size_px,
            "n_tiles": len(fa),
            "n_nonzero": len(fa_nonzero),
            "fa_mean_all": float(fa.mean()),
            "fa_nonzero_p50": float(np.median(fa_nonzero)) if len(fa_nonzero) else float("nan"),
            "fa_nonzero_p90": float(np.quantile(fa_nonzero, 0.90)) if len(fa_nonzero) else float("nan"),
            "fa_nonzero_p99": float(np.quantile(fa_nonzero, 0.99)) if len(fa_nonzero) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("scale_idx").reset_index(drop=True)


def main() -> int:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:.4g}")

    print("Loading dataset/labels/*.parquet ...")
    df = load_all_labels()
    print(f"  {len(df):,} rows across {df['obs_id'].nunique()} ObsIds, {df['scale_idx'].nunique()} scales\n")

    print("=== Per-scale positive-class summary ===")
    print(per_scale_summary(df).to_string(index=False))
    print()

    print("=== Placeholder threshold disagreement (area>=0.005 vs count>=5) ===")
    print(placeholder_disagreement(df).to_string(index=False))
    print()

    print("=== fractional_area quantiles (nonzero tail) ===")
    print(fa_quantiles(df).to_string(index=False))
    print()

    for scale_idx in sorted(df["scale_idx"].unique()):
        tile_size = int(df.loc[df["scale_idx"] == scale_idx, "tile_size_px"].iloc[0])
        print(f"=== Matched-threshold candidates at scale_idx={scale_idx} (tile_size_px={tile_size}) ===")
        rec = recommend_matched_thresholds(df, scale_idx, [0.02, 0.01, 0.005, 0.002, 0.001])
        print(rec.to_string(index=False))
        print()

    print("=== Strict 'any boulder' positive (fractional_area > 0) cross-check ===")
    for scale_idx in sorted(df["scale_idx"].unique()):
        g = df[df["scale_idx"] == scale_idx]
        bc = g["boulder_count"].to_numpy()
        fa = g["fractional_area"].to_numpy()
        # fa > 0 IFF boulder_area > 0 IFF at least one polygon-fragment touched the tile.
        # boulder_count >= 1 is the "centroid inside" rule; the two can differ at borders.
        n_fa = int((fa > 0).sum())
        n_bc = int((bc >= 1).sum())
        union = int(((fa > 0) | (bc >= 1)).sum())
        both = int(((fa > 0) & (bc >= 1)).sum())
        jaccard = (both / union) if union > 0 else float("nan")
        print(
            f"  scale_idx={scale_idx} tile_size_px={int(g['tile_size_px'].iloc[0]):>3d}  "
            f"fa>0={n_fa:>6d}  bc>=1={n_bc:>6d}  jaccard={jaccard:.3f}  "
            f"agreement={both/max(union,1):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
