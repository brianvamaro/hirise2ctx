"""Prediction gallery for docs/model_evidence.md (PLAN_FM §2.5).

Six held-out images spanning terrain + outcome regimes (hybrid axis, Brian
2026-06-14b). Each panel: the CTX window in grayscale, the frozen recipe's
per-tile P(rich) as a heatmap overlay, and the true boulder-rich tiles outlined
in lime. All predictions are group-aware LOIO (the image was held out), read from
the banked frozen `predictions.parquet`. Per-image AUC + base rate annotated.

Output: reports/figures/model_evidence_prediction_gallery.png
"""
from __future__ import annotations

import sys
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling as RIOResampling
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mapping import tiles_to_raster  # noqa: E402

PRED = ROOT / "models" / "fang_probe" / "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2" / "predictions.parquet"
FIG = ROOT / "reports" / "figures"

# (obs_id, terrain label, regime caption)
PANELS = [
    ("ESP_045139_2270", "Plains", "dense boulder-rich plains"),
    ("ESP_046959_2225", "Mesas", "mesa terrain"),
    ("ESP_071699_2260", "Crater-dominated", "crater terrain"),
    ("ESP_068402_2240", "Channels", "channelled terrain"),
    ("ESP_046328_2180", "Plains + crater", "rescued anti-signal (Part-1 AUC 0.34)"),
    ("ESP_076499_1160", "Southern hemisphere", "region/azimuth outlier (rescued)"),
]


def main():
    pred = pd.read_parquet(PRED)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.2))

    for ax, (obs, terrain, regime) in zip(axes.ravel(), PANELS):
        g = pred[pred["obs_id"] == obs]
        lab = pd.read_parquet(ROOT / "dataset_v2" / "labels" / f"{obs}.parquet")
        lab = lab[lab["scale_idx"] == 2][["ti", "tj", "xmin", "ymin", "xmax", "ymax"]]
        d = g.merge(lab, on=["ti", "tj"], how="left", validate="one_to_one")

        ti = d["ti"].to_numpy(); tj = d["tj"].to_numpy()
        prob, _, _ = tiles_to_raster(ti, tj, d["y_pred"].to_numpy())
        truth, _, _ = tiles_to_raster(ti, tj, d["y_true"].to_numpy().astype(float))
        left, right = d["xmin"].min(), d["xmax"].max()
        bottom, top = d["ymin"].min(), d["ymax"].max()
        extent = (left, right, bottom, top)

        # CTX background, decimated
        with rasterio.open(ROOT / "cache_v2" / "ctx_windows" / f"{obs}.tif") as ctx:
            sc = max(1, int(max(ctx.shape) / 700))
            cdata = ctx.read(1, out_shape=(ctx.height // sc, ctx.width // sc),
                             resampling=RIOResampling.average).astype(np.float32)
            cb = ctx.bounds
        v = cdata[(cdata > 0) & np.isfinite(cdata)]
        lo, hi = np.percentile(v, (2, 98)) if v.size else (0, 1)
        cdisp = np.clip((cdata - lo) / max(hi - lo, 1e-9), 0, 1)
        ax.imshow(cdisp, cmap="gray", extent=(cb.left, cb.right, cb.bottom, cb.top),
                  origin="upper", interpolation="nearest")

        # P(rich) heatmap overlay (transparent where no tile)
        cmap = copy(plt.cm.turbo); cmap.set_bad(alpha=0.0)
        im = ax.imshow(np.ma.masked_invalid(prob), cmap=cmap, vmin=0, vmax=1,
                       alpha=0.58, extent=extent, origin="upper",
                       interpolation="nearest")
        # true boulder-rich tiles outlined (white reads on both hot & cold heatmap)
        ax.contour(np.where(np.isnan(truth), 0, truth), levels=[0.5],
                   colors="white", linewidths=0.8,
                   extent=extent, origin="upper")

        ax.set_xlim(left, right); ax.set_ylim(bottom, top)
        ax.set_xticks([]); ax.set_yticks([])
        auc = roc_auc_score(g["y_true"], g["y_pred"]) if g["y_true"].nunique() > 1 else float("nan")
        base = g["y_true"].mean()
        ax.set_title(f"{obs}  —  {terrain}\n{regime}\n"
                     f"per-image AUC {auc:.3f}  ·  base rate {base:.0%}",
                     fontsize=10)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01,
                        location="right")
    cbar.set_label("model P(boulder-rich)")
    fig.suptitle("Held-out predictions across terrain & outcome regimes "
                 "(white outline = true boulder-rich tiles; group-aware LOIO at 160 m)",
                 fontsize=12, y=0.98)
    out = FIG / "model_evidence_prediction_gallery.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
