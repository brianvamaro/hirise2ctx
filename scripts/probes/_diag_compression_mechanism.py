"""Diagnose the dynamic-range compression of the v2 two_stage S=64 regressor.

Three questions, answered from the cached OOF predictions:

  1. Mechanism — is the compression coming from the presence head (p_pos doesn't
     drop on true zeros) or the magnitude head (E[mag|pos] is squashed near the
     log-positive median) or both?  Decompose pred = p_pos * mag and look at
     mean_p_pos / mean_mag per truth bin.
  2. How much of the compression does a *post-hoc isotonic recalibration* on
     LOIO-OOF predictions recover?  This is the cheapest possible intervention
     and gives an upper bound on what monotone calibration buys.  Fit one
     isotonic map per held-out fold using the OTHER folds' OOF data as the
     calibration set (proper LOIO recalibration).
  3. How much does just clipping p_pos to a hard 0/1 (or using p_pos as a soft
     gate) change things?

Writes a 2x3 figure to reports/figures/12_compression_diagnostic.png plus a
small markdown summary table to scripts/probes/_diag_compression_mechanism.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import src.modeling  # noqa: F401 (Windows DLL bootstrap)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
PREDS = REPO / "models/lightgbm_two_stage/629276139c22da68/scale_S64/predictions.parquet"
METRICS = REPO / "models/lightgbm_two_stage/629276139c22da68/scale_S64/metrics.json"
FIG = REPO / "reports/figures/12_compression_diagnostic.png"
OUT_MD = Path(__file__).with_suffix(".md")

# Truth bin edges shared with evaluate.py per_bin_rmse
BIN_EDGES = [(-1e-12, 0.0, "zero"),
             (0.0, 1e-4, "0_to_1e-4"),
             (1e-4, 1e-3, "1e-4_to_1e-3"),
             (1e-3, 1e-2, "1e-3_to_1e-2"),
             (1e-2, 1.0, "1e-2_to_max")]


def bin_label(y: float) -> str:
    if y <= 0:
        return "zero"
    for lo, hi, name in BIN_EDGES[1:]:
        if lo < y <= hi:
            return name
    return "1e-2_to_max"


def isotonic_oof_recalibrate(df: pd.DataFrame, pred_col: str, target_col: str = "y_true") -> np.ndarray:
    """LOIO-correct isotonic recalibration: for each held-out fold, fit the iso
    map on the OOF preds of every OTHER fold, then apply it to this fold."""
    from sklearn.isotonic import IsotonicRegression
    folds = df["fold_idx"].unique()
    cal = np.empty_like(df[pred_col].to_numpy(), dtype=np.float64)
    cal[:] = np.nan
    for f in folds:
        train_mask = df["fold_idx"] != f
        test_mask = df["fold_idx"] == f
        iso = IsotonicRegression(out_of_bounds="clip", increasing=True)
        iso.fit(df.loc[train_mask, pred_col].to_numpy(),
                df.loc[train_mask, target_col].to_numpy())
        cal[test_mask.to_numpy()] = iso.predict(df.loc[test_mask, pred_col].to_numpy())
    return cal


def per_bin_means(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    g = df.groupby("truth_bin", observed=True).agg(
        n_tiles=(pred_col, "size"),
        mean_true=("y_true", "mean"),
        mean_pred=(pred_col, "mean"),
        median_pred=(pred_col, "median"),
    )
    g["ratio_pred_over_true"] = g["mean_pred"] / g["mean_true"].where(g["mean_true"] > 0, np.nan)
    return g


def main() -> None:
    df = pd.read_parquet(PREDS)
    print(f"loaded {len(df):,} rows, cols: {list(df.columns)}")
    print(df.head(3))
    # Standardise column names: parquet from evaluator should have y_true, y_pred,
    # and either a p_pos column or presence_prob.
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in {"y_true", "fractional_area"}:
            rename[c] = "y_true"
        elif cl in {"y_pred", "pred"}:
            rename[c] = "y_pred"
        elif cl in {"p_pos", "presence_prob", "presence", "p_presence", "y_pred_presence_prob"}:
            rename[c] = "p_pos"
        elif cl in {"fold_idx", "fold"}:
            rename[c] = "fold_idx"
        elif cl in {"obs_id", "obsid", "image_id"}:
            rename[c] = "obs_id"
    df = df.rename(columns=rename)
    needed = {"y_true", "y_pred", "fold_idx"}
    miss = needed - set(df.columns)
    if miss:
        print(f"WARNING: missing expected columns {miss}; have {list(df.columns)}")
    # Filter to non-specificity folds (folds where the held-out image has any boulders)
    if "obs_id" in df.columns:
        spec_mask = df.groupby("fold_idx")["y_true"].transform("max") == 0
        df = df[~spec_mask].copy()
    # Assign truth bins
    bin_names = [b[2] for b in BIN_EDGES]
    df["truth_bin"] = pd.Categorical(df["y_true"].apply(bin_label),
                                     categories=bin_names, ordered=True)

    print("\n=== Question 1: where is the compression?  (raw pred decomposition) ===")
    base = per_bin_means(df, "y_pred")
    print(base.to_string())

    have_ppos = "p_pos" in df.columns
    have_mag = "mag" in df.columns
    if have_ppos and not have_mag:
        # mag = pred / p_pos where p_pos > 0
        df["mag"] = np.where(df["p_pos"] > 1e-9, df["y_pred"] / df["p_pos"], np.nan)
        have_mag = True

    if have_ppos:
        print("\n--- p_pos by truth bin ---")
        print(df.groupby("truth_bin", observed=True)["p_pos"].agg(["mean", "median", "std"]).to_string())
    if have_mag:
        print("\n--- mag (=pred/p_pos) by truth bin ---")
        print(df.groupby("truth_bin", observed=True)["mag"].agg(["mean", "median", "std"]).to_string())

    print("\n=== Question 2: how much does LOIO-isotonic recalibration recover? ===")
    df["y_pred_iso"] = isotonic_oof_recalibrate(df, "y_pred")
    iso_table = per_bin_means(df, "y_pred_iso")
    print(iso_table.to_string())

    # Compare scalar metrics
    def spearman(y, p):
        from scipy.stats import spearmanr
        if np.unique(y).size < 2 or np.unique(p).size < 2:
            return np.nan
        return spearmanr(y, p).statistic

    def auc(y, p):
        from sklearn.metrics import roc_auc_score
        yb = (y > 0).astype(int)
        if yb.sum() in (0, len(yb)):
            return np.nan
        return roc_auc_score(yb, p)

    # Per-fold spearman & auc (raw vs iso)
    rows = []
    for f, sub in df.groupby("fold_idx"):
        rows.append({
            "fold_idx": f,
            "n": len(sub),
            "spearman_raw": spearman(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()),
            "spearman_iso": spearman(sub["y_true"].to_numpy(), sub["y_pred_iso"].to_numpy()),
            "auc_raw": auc(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()),
            "auc_iso": auc(sub["y_true"].to_numpy(), sub["y_pred_iso"].to_numpy()),
            "mean_true": sub["y_true"].mean(),
            "mean_raw": sub["y_pred"].mean(),
            "mean_iso": sub["y_pred_iso"].mean(),
        })
    foldwise = pd.DataFrame(rows)
    print("\nPer-fold (LOIO) headline: spearman + AUC, raw vs iso")
    print(foldwise.describe().to_string())

    # Figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Panel A: per-bin mean_pred bars: raw vs iso vs identity
    ax = axes[0, 0]
    bin_labels = [b[2] for b in BIN_EDGES]
    base_filt = base.reindex(bin_labels).dropna()
    iso_filt = iso_table.reindex(bin_labels).dropna()
    x = np.arange(len(base_filt))
    w = 0.35
    ax.bar(x - w/2, base_filt["mean_pred"], w, label="raw mean_pred", color="C0")
    ax.bar(x + w/2, iso_filt["mean_pred"], w, label="iso mean_pred", color="C2")
    ax.plot(x, base_filt["mean_true"], "ko-", label="mean_true (identity)", lw=2)
    ax.set_xticks(x); ax.set_xticklabels(base_filt.index, rotation=30)
    ax.set_ylabel("mean (linear scale)")
    ax.set_title("Mean prediction per truth bin\n(closer to identity = less compression)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel B: same on log scale
    ax = axes[0, 1]
    ax.semilogy(x, base_filt["mean_pred"].clip(1e-7), "o-", label="raw mean_pred", color="C0")
    ax.semilogy(x, iso_filt["mean_pred"].clip(1e-7), "s-", label="iso mean_pred", color="C2")
    ax.semilogy(x, base_filt["mean_true"].clip(1e-7), "k^-", label="mean_true", lw=2)
    ax.set_xticks(x); ax.set_xticklabels(base_filt.index, rotation=30)
    ax.set_ylabel("mean (log scale)")
    ax.set_title("Same but log scale — shows over-prediction\non low bins, under-prediction on high")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel C: ratio pred/true (closer to 1 = better)
    ax = axes[0, 2]
    ax.bar(x - w/2, base_filt["ratio_pred_over_true"], w, label="raw", color="C0")
    ax.bar(x + w/2, iso_filt["ratio_pred_over_true"], w, label="iso", color="C2")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="perfect calibration")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(base_filt.index, rotation=30)
    ax.set_ylabel("mean_pred / mean_true (log)")
    ax.set_title("Bias ratio per truth bin\nlow bin >>1 (overpred), high bin <<1 (underpred)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel D: p_pos distribution by truth bin (boxplot)
    ax = axes[1, 0]
    if have_ppos:
        boxdata = [df.loc[df["truth_bin"] == b, "p_pos"].to_numpy() for b in bin_labels]
        ax.boxplot(boxdata, tick_labels=bin_labels, showfliers=False)
        ax.set_ylabel("p_pos (presence-head probability)")
        ax.set_title("Presence-head p_pos by truth bin\n(if zero-bin p_pos is high, presence head over-confidence drives the overpred floor)")
    else:
        ax.text(0.5, 0.5, "p_pos not in predictions.parquet", ha="center", va="center")
    ax.grid(alpha=0.3)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)

    # Panel E: mag distribution by truth bin (boxplot)
    ax = axes[1, 1]
    if have_mag:
        boxdata = [df.loc[df["truth_bin"] == b, "mag"].dropna().to_numpy() for b in bin_labels]
        ax.boxplot(boxdata, tick_labels=bin_labels, showfliers=False)
        ax.set_yscale("log")
        ax.set_ylabel("mag = pred/p_pos (log)")
        ax.set_title("Magnitude head E[mag|pos] by truth bin\n(if mag is flat across bins, mag head is squashed)")
    else:
        ax.text(0.5, 0.5, "no magnitude column derivable", ha="center", va="center")
    ax.grid(alpha=0.3)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)

    # Panel F: pooled pred-vs-true scatter with iso overlay
    ax = axes[1, 2]
    sub = df.sample(min(50_000, len(df)), random_state=0)
    ax.loglog(np.clip(sub["y_true"], 1e-7, None), np.clip(sub["y_pred"], 1e-7, None),
              ".", ms=1.5, alpha=0.15, color="C0", label="raw")
    ax.loglog(np.clip(sub["y_true"], 1e-7, None), np.clip(sub["y_pred_iso"], 1e-7, None),
              ".", ms=1.5, alpha=0.3, color="C2", label="iso")
    lo, hi = 1e-7, 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    ax.set_xlabel("true fractional_area")
    ax.set_ylabel("predicted")
    ax.set_xlim(1e-7, 1.0); ax.set_ylim(1e-7, 1.0)
    ax.set_title("Pooled pred-vs-true scatter\n(iso lifts the high-tail; doesn't change rank)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")

    plt.suptitle("Compression diagnostic — v2 LOIO `lightgbm_two_stage` S=64\n"
                 "(black = truth, blue = raw, green = iso-recalibrated)",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG, dpi=120, bbox_inches="tight")
    print(f"\nFigure -> {FIG}")

    # Markdown summary
    lines = ["# Compression diagnostic — v2 LOIO `lightgbm_two_stage` S=64", ""]
    lines.append(f"Source: {PREDS.relative_to(REPO)}")
    lines.append(f"Figure: {FIG.relative_to(REPO)}")
    lines.append("")
    lines.append("## Per-bin mean prediction (raw vs LOIO-isotonic recalibration)")
    lines.append("")
    table = base[["n_tiles", "mean_true", "mean_pred", "ratio_pred_over_true"]].copy()
    table.columns = ["n_tiles", "mean_true", "mean_pred_raw", "ratio_raw"]
    table["mean_pred_iso"] = iso_table["mean_pred"]
    table["ratio_iso"] = iso_table["ratio_pred_over_true"]
    table = table.reindex(bin_labels)
    lines.append(table.to_string(float_format=lambda v: f"{v:.4f}"))
    lines.append("")
    lines.append("## Per-fold (LOIO) headline")
    lines.append("")
    desc = foldwise[["spearman_raw", "spearman_iso", "auc_raw", "auc_iso"]].describe().loc[["mean", "std"]]
    lines.append(desc.to_string(float_format=lambda v: f"{v:.4f}"))
    lines.append("")
    lines.append("**Interpretation cheatsheet:**")
    lines.append("- Panel A/B/C: how much does iso-recalibration close the bin-mean gap?")
    lines.append("- Panel D: is p_pos collapsing toward 0 on true-zero tiles, or is it spreading mass everywhere?")
    lines.append("- Panel E: does the magnitude head produce a flat distribution across truth bins (= the squash)?")
    lines.append("- Panel F: do iso preds reach the high-bin diagonal that raw preds miss?")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary -> {OUT_MD}")


if __name__ == "__main__":
    main()
