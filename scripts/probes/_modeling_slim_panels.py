"""Build the good-vs-bad AUC binary-overlay figure for docs/modeling_slim.md.

Each panel shows a binary boulder-rich/poor decision overlaid on the CTX
window for one held-out image:
  - Truth: rich = fractional_area >= 1e-2 (the AUC threshold)
  - Prediction: rich = top-K tiles by predicted count, where K is set to the
    truth-rich count in that image, so both panels show the same number of
    "rich" tiles by construction. The visual question is then: are they the
    same tiles?

Top row -- ESP_053989_2260 (good case, AUC 0.880).
Bottom row -- ESP_046328_2180 (anti-signal case, AUC 0.344).

Output: reports/figures/modeling_slim_good_vs_bad.png  (4-panel)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "dataset_v2"
CACHE = ROOT / "cache_v2" / "ctx_windows"
FIG = ROOT / "reports" / "figures"

SCALE_IDX = 3  # S=64
GOOD_OBS = "ESP_053989_2260"
BAD_OBS = "ESP_046328_2180"
FA_RICH_THRESHOLD = 1e-2

RICH_COLOR = "#2a9d8f"     # green = rich
POOR_COLOR = "#9b59b6"     # purple = poor (used in the "both" variant only)


def stamp_grid(tile_df: pd.DataFrame, value_col: str,
               transform, raster_shape) -> np.ndarray:
    """Build a 2D pixel-sized grid of tile values, NaN outside the
    HiRISE-eligible footprint, ready for imshow over the CTX raster.
    """
    grid = np.full(raster_shape, np.nan, dtype=np.float64)
    side_m = float(tile_df["tile_size_m"].iloc[0])
    px = abs(transform[0])
    side_px = int(round(side_m / px))
    left = transform[2]
    top = transform[5]
    for _, row in tile_df.iterrows():
        col_start = int(round((row["xmin"] - left) / px))
        row_start = int(round((top - row["ymax"]) / px))
        r0 = max(0, row_start)
        c0 = max(0, col_start)
        r1 = min(raster_shape[0], row_start + side_px)
        c1 = min(raster_shape[1], col_start + side_px)
        if r1 > r0 and c1 > c0:
            grid[r0:r1, c0:c1] = row[value_col]
    return grid


def render_obs(ax_truth, ax_pred, obs_id: str, preds_all: pd.DataFrame,
               summary: pd.DataFrame, label_prefix: str,
               variant: str = "both"):
    """variant = 'both' shows rich (green) + poor (purple);
       variant = 'rich_only' shows only rich tiles (green), poor transparent."""
    src = rasterio.open(CACHE / f"{obs_id}.tif")
    ctx = src.read(1).astype(np.float32)
    p2, p98 = np.percentile(ctx[ctx > 0], (2, 98))
    transform = src.transform
    extent = (src.bounds.left, src.bounds.right,
              src.bounds.bottom, src.bounds.top)

    labels = pd.read_parquet(DATASET / "labels" / f"{obs_id}.parquet")
    truth = labels[labels["scale_idx"] == SCALE_IDX].copy()
    pred = preds_all[preds_all["obs_id"] == obs_id].copy()

    # Restrict truth to eval-eligible tiles (the same set the AUC sees).
    pred_keyed = pred[["ti", "tj", "pred"]]
    truth_eval = truth.merge(pred_keyed, on=["ti", "tj"], how="inner")

    n_rich_truth = int((truth_eval["fractional_area"] >= FA_RICH_THRESHOLD).sum())
    truth_eval["is_rich_truth"] = (truth_eval["fractional_area"] >= FA_RICH_THRESHOLD).astype(int)

    # Prediction binary: top-K by predicted count where K = n_rich_truth, so
    # both panels show the same count of "rich" tiles by construction.
    truth_eval = truth_eval.sort_values("pred", ascending=False).reset_index(drop=True)
    truth_eval["is_rich_pred"] = 0
    if n_rich_truth > 0:
        truth_eval.loc[:n_rich_truth - 1, "is_rich_pred"] = 1

    if variant == "rich_only":
        # Stamp NaN everywhere except rich tiles; rich tiles get value 1.
        truth_eval["truth_overlay"] = np.where(
            truth_eval["is_rich_truth"] == 1, 1.0, np.nan)
        truth_eval["pred_overlay"] = np.where(
            truth_eval["is_rich_pred"] == 1, 1.0, np.nan)
        grid_truth = stamp_grid(truth_eval, "truth_overlay", transform, ctx.shape)
        grid_pred = stamp_grid(truth_eval, "pred_overlay", transform, ctx.shape)
        cmap = mcolors.ListedColormap([RICH_COLOR])
        cmap.set_bad(alpha=0.0)
        norm = mcolors.BoundaryNorm([0.5, 1.5], cmap.N)
    else:
        # Stamp grids: rich = 1, poor = 0
        grid_truth = stamp_grid(truth_eval, "is_rich_truth", transform, ctx.shape)
        grid_pred = stamp_grid(truth_eval, "is_rich_pred", transform, ctx.shape)
        cmap = mcolors.ListedColormap([POOR_COLOR, RICH_COLOR])
        cmap.set_bad(alpha=0.0)
        norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    auc = summary.loc[summary["held_out_obs_id"] == obs_id, "auc_fa_rich"].iloc[0]
    rho = summary.loc[summary["held_out_obs_id"] == obs_id, "rho"].iloc[0]
    n_total = int(summary.loc[summary["held_out_obs_id"] == obs_id,
                             "n_test_tiles"].iloc[0])

    # Agreement count: tiles called rich by BOTH truth and prediction
    n_agree_rich = int(((truth_eval["is_rich_truth"] == 1) &
                        (truth_eval["is_rich_pred"] == 1)).sum())

    for ax, grid, title in [
        (ax_truth, grid_truth, "truth: rich = fractional_area >= 1%"),
        (ax_pred, grid_pred,
         f"slim model: rich = top {n_rich_truth} tiles by predicted count"),
    ]:
        ax.imshow(ctx, cmap="gray", vmin=p2, vmax=p98, extent=extent,
                  origin="upper", interpolation="nearest")
        ax.imshow(grid, cmap=cmap, norm=norm, extent=extent,
                  origin="upper", interpolation="nearest", alpha=0.6)
        ax.set_title(
            f"{label_prefix}: {obs_id}\n{title}\n"
            f"per-image AUC={auc:.3f}  rho={rho:+.3f}  "
            f"n_rich={n_rich_truth}/{n_total}  agree={n_agree_rich}",
            fontsize=10)
        ax.set_xlabel("Eastings (m)")
        ax.set_ylabel("Northings (m)")
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.tick_params(labelsize=7)
        # Custom legend per panel
        if variant == "rich_only":
            legend_handles = [
                mpatches.Patch(color=RICH_COLOR, alpha=0.6, label="boulder-rich"),
            ]
        else:
            legend_handles = [
                mpatches.Patch(color=RICH_COLOR, alpha=0.6, label="boulder-rich"),
                mpatches.Patch(color=POOR_COLOR, alpha=0.6, label="boulder-poor"),
            ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=8,
                  framealpha=0.85)
    src.close()


def build(variant: str, out_name: str, suptitle: str) -> Path:
    preds_all = pd.read_parquet(DATASET / "modeling_slim_predictions.parquet")
    summary = pd.read_parquet(DATASET / "modeling_slim_summary.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    render_obs(axes[0, 0], axes[0, 1], GOOD_OBS, preds_all, summary, "GOOD",
               variant=variant)
    render_obs(axes[1, 0], axes[1, 1], BAD_OBS, preds_all, summary, "ANTI-SIGNAL",
               variant=variant)
    fig.suptitle(suptitle, y=1.005)
    fig.tight_layout()
    out = FIG / out_name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    # The canonical headline figure for docs/modeling_slim.md uses the
    # rich-only variant -- poor tiles are kept transparent so the CTX
    # surface context (craters, terrain) is visible underneath.
    out = build(
        variant="rich_only",
        out_name="modeling_slim_good_vs_bad.png",
        suptitle=(
            "Per-image truth vs slim-model binary rich/poor calls at S=64\n"
            "Both panels show the same number of 'rich' tiles for "
            "each image; the question is whether they pick the same ones."),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
