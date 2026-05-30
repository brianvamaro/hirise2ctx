"""Confusion-matrix-on-a-map: render TP/FP/FN colored overlays on the CTX windows
for best / median / worst-performing v2 images at the boulder-rich threshold.

Motivation: the continuous-fractional_area heatmap in notebook 13 §5 makes even the best
image (ESP_042964_2160, AUC 0.91) look washed out because the v2 baseline regressor
compresses its prediction range. The binary view at fa_gt_1e-2 is what AUC actually
measures, and rendering top-K predicted tiles colored by their truth gives a vivid sense
of why AUC 0.91 is qualitatively different from AUC 0.40.

Visualization recipe per image:
  - Left panel:  CTX window, grayscale.
  - Right panel: same CTX, with each tile overlaid by its confusion class at top-K
    (K = number of true positives in the image, which is the convention lift@K uses):
      TP (boulder-rich AND in top-K predicted)  -> green
      FP (in top-K predicted but NOT boulder-rich) -> red
      FN (boulder-rich but NOT in top-K predicted) -> orange
      TN (rest) -> no overlay (CTX shows through)
    Visual scoreboard: more green = better AUC.

Output: reports/figures/13_topk_confusion_map.png + per-tile counts in markdown.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Full-v2 lightgbm_classification at fa_gt_1e-2, S=64
BIN_PRED = REPO / "models/lightgbm_classification/99de85c1ad2a72e6/scale_S64_tfa_gt_1e-2/predictions.parquet"
DATASET_V2 = REPO / "dataset_v2"
CTX_WINDOWS = REPO / "cache_v2" / "ctx_windows"
OUT_FIG = REPO / "reports" / "figures" / "13_topk_confusion_map.png"
OUT_MD = Path(__file__).with_suffix(".md")

# Threshold-on-truth: fa_gt_1e-2 = boulder-rich
TRUTH_THRESHOLD = 0.01

# Three pin images selected to span the AUC range (from the per-image breakdown):
#   best   : ESP_042964_2160  (AUC 0.91, lift 5.4)
#   median : ESP_046959_2225  (AUC 0.60, lift 1.5)  -- roughly median of v2
#   worst  : ESP_054000_2255  (AUC 0.40, lift 0.29) -- anti-signal
PIN_IMAGES = [
    ("ESP_042964_2160", "BEST  | AUC 0.91, lift 5.4x"),
    ("ESP_046959_2225", "TYP   | AUC 0.60, lift 1.5x"),
    ("ESP_054000_2255", "WORST | AUC 0.40, lift 0.29x (anti-signal)"),
]


def load_obs(obs_id: str, scale_idx: int = 3):
    lab = pd.read_parquet(DATASET_V2 / "labels" / f"{obs_id}.parquet")
    lab = lab[lab["scale_idx"] == scale_idx].copy()
    with rasterio.open(CTX_WINDOWS / f"{obs_id}.tif") as r:
        ctx = r.read(1)
        ctx_ext = (r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top)
    return lab, ctx, ctx_ext


def confusion_grid(truth_binary: np.ndarray, pred_top_k: np.ndarray) -> np.ndarray:
    """Return per-tile confusion class:
       0 = TN, 1 = FN, 2 = FP, 3 = TP."""
    out = np.zeros_like(truth_binary, dtype=np.int8)
    out[(truth_binary == 1) & (pred_top_k == 0)] = 1   # FN (missed)
    out[(truth_binary == 0) & (pred_top_k == 1)] = 2   # FP (wrong alert)
    out[(truth_binary == 1) & (pred_top_k == 1)] = 3   # TP (caught)
    return out


def confusion_to_grid(df: pd.DataFrame, ti_min: int, ti_max: int, tj_min: int, tj_max: int) -> np.ndarray:
    """Place per-tile confusion class onto the (ti, tj) grid as a 2-D array. NaN = TN
    (rendered transparent so the CTX shows through)."""
    g = np.full((ti_max - ti_min + 1, tj_max - tj_min + 1), np.nan)
    g[df["ti"].to_numpy() - ti_min, df["tj"].to_numpy() - tj_min] = df["confusion"].to_numpy()
    # Replace 0 (TN) with NaN so it doesn't paint
    g[g == 0] = np.nan
    return g


def main() -> int:
    print(f"Loading binary predictions: {BIN_PRED}")
    preds = pd.read_parquet(BIN_PRED)
    print(f"  {len(preds):,} rows  cols={preds.columns.tolist()}")

    # Truth at fa > 1e-2 from each image's labels parquet
    confusion_cmap = ListedColormap(["#f9a23566", "#e7212199", "#2ca02c"])  # FN orange, FP red, TP green
    confusion_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], confusion_cmap.N)

    fig, axes = plt.subplots(len(PIN_IMAGES), 2, figsize=(13, 5.0 * len(PIN_IMAGES)), squeeze=False)
    stats_rows = []
    for i, (obs, title) in enumerate(PIN_IMAGES):
        lab, ctx, ctx_ext = load_obs(obs, scale_idx=3)
        sub = lab[["ti", "tj", "xmin", "ymin", "xmax", "ymax", "fractional_area"]].copy()
        sub["truth_binary"] = (sub["fractional_area"] > TRUTH_THRESHOLD).astype(np.int8)
        n_pos = int(sub["truth_binary"].sum())
        n = len(sub)
        base_rate = n_pos / max(n, 1)

        # Join with classifier predictions for this image
        rp = preds[preds["obs_id"] == obs].merge(
            sub[["ti", "tj", "truth_binary", "xmin", "xmax", "ymin", "ymax"]],
            on=["ti", "tj"], how="inner",
        )
        if n_pos == 0 or len(rp) == 0:
            print(f"  {obs}: skipping (no positives or no preds)")
            continue
        # Top-K by predicted probability where K = n_pos (lift@K convention)
        rp = rp.sort_values("y_pred", ascending=False).reset_index(drop=True)
        rp["pred_topk"] = 0
        rp.loc[:n_pos - 1, "pred_topk"] = 1

        # Per-tile confusion class
        rp["confusion"] = confusion_grid(rp["truth_binary"].to_numpy(), rp["pred_topk"].to_numpy())
        tp = int((rp["confusion"] == 3).sum())
        fp = int((rp["confusion"] == 2).sum())
        fn = int((rp["confusion"] == 1).sum())
        precision_at_k = tp / max(n_pos, 1)
        lift = precision_at_k / max(base_rate, 1e-12)
        stats_rows.append({
            "ObsId": obs, "n_tiles": n, "n_pos": n_pos, "base_rate": base_rate,
            "TP": tp, "FP": fp, "FN": fn, "precision@K": precision_at_k, "lift@K": lift,
        })
        print(f"  {obs}: n={n}, n_pos={n_pos}, TP={tp}, FP={fp}, FN={fn}, "
              f"precision@K={precision_at_k:.3f}, lift@K={lift:.2f}")

        ti_min, ti_max = int(sub["ti"].min()), int(sub["ti"].max())
        tj_min, tj_max = int(sub["tj"].min()), int(sub["tj"].max())
        ext = (float(sub["xmin"].min()), float(sub["xmax"].max()),
               float(sub["ymin"].min()), float(sub["ymax"].max()))
        conf_grid_2d = confusion_to_grid(rp, ti_min, ti_max, tj_min, tj_max)

        # Render CTX baseline contrast
        p1, p99 = (np.percentile(ctx[ctx > 0], [1, 99]) if (ctx > 0).any() else (0, 255))

        ax_ctx = axes[i][0]
        ax_ctx.imshow(ctx, extent=ctx_ext, cmap="gray", vmin=p1, vmax=p99,
                      origin="upper", aspect="equal")
        ax_ctx.set_xlim(ctx_ext[0], ctx_ext[1]); ax_ctx.set_ylim(ctx_ext[2], ctx_ext[3])
        ax_ctx.set_title(f"{obs}\nCTX window", fontsize=10)
        ax_ctx.set_xticks([]); ax_ctx.set_yticks([])

        ax_conf = axes[i][1]
        ax_conf.imshow(ctx, extent=ctx_ext, cmap="gray", vmin=p1, vmax=p99,
                       origin="upper", aspect="equal")
        ax_conf.imshow(conf_grid_2d, extent=ext, cmap=confusion_cmap, norm=confusion_norm,
                       alpha=0.7, origin="upper", aspect="equal")
        ax_conf.set_xlim(ctx_ext[0], ctx_ext[1]); ax_conf.set_ylim(ctx_ext[2], ctx_ext[3])
        ax_conf.set_title(f"{title}\nTP={tp}  FP={fp}  FN={fn}  base={base_rate:.2%}",
                          fontsize=10)
        ax_conf.set_xticks([]); ax_conf.set_yticks([])

        if i == 0:
            legend = [
                Patch(facecolor="#2ca02c", alpha=0.7, label="TP (boulder-rich AND in top-K predicted)"),
                Patch(facecolor="#e72121", alpha=0.7, label="FP (in top-K predicted, NOT boulder-rich)"),
                Patch(facecolor="#f9a235", alpha=0.7, label="FN (boulder-rich, NOT in top-K)"),
                Patch(facecolor="lightgray", alpha=0.7, label="TN (rest, CTX shown through)"),
            ]
            ax_conf.legend(handles=legend, loc="upper center",
                           bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=8)

    plt.suptitle("Top-K confusion overlay at fa_gt_1e-2 boulder-rich threshold (S=64)\n"
                 "K = n_positives per image (the lift@K convention) — green wins, red/orange lose",
                 fontsize=12, y=1.00)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=120, bbox_inches="tight")
    print(f"\nFigure -> {OUT_FIG}")

    # Markdown summary
    stats_df = pd.DataFrame(stats_rows)
    lines = ["# Top-K confusion-map probe", "", f"Figure: {OUT_FIG.relative_to(REPO)}", "",
             "Per-image confusion counts at K = n_pos:", "",
             stats_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
             "",
             "**Reading the figure:**",
             "- Green tiles dominate ⇒ the model correctly identified boulder-rich tiles.",
             "- A green-dominant cluster ⇒ AUC is high and that's *operationally* meaningful.",
             "- Red dominant ⇒ the model's top-K confidently flagged tiles that aren't boulder-rich.",
             "- Orange dominant ⇒ the model missed many true boulder-rich tiles.",
             "- Anti-signal cases (AUC < 0.5) have nearly all red/orange and almost no green."]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown -> {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
