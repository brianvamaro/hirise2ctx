"""DRAFT: the model_evidence Tier-2 abundance map WITH the calibration fix applied.

Same image as `model_evidence_tier2_map.png` (ESP_053989_2260), now three panels:
TRUE | raw mlp_reg | quantile-matched (de-compressed). The calibration is LOIO-honest
(quantile-match fit on the other 37 images, applied to this one). Does NOT touch the
original figure -- this is a separate draft to preview the fix on a real map.

Output: reports/figures/model_evidence_tier2_map_calibrated.png
"""
from __future__ import annotations

import sys
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.mapping import tiles_to_raster  # noqa: E402
from src.calibration import quantile_match, loio_calibrate, compression_metrics  # noqa: E402

PRED = (REPO / "models" / "fang_tier2" / "tier2_mlp_reg_emb_fractional_area_S32"
        / "1e01ad8b17447599" / "predictions.parquet")
FIG = REPO / "reports" / "figures"
OBS = "ESP_053989_2260"


def main():
    pred = pd.read_parquet(PRED)
    # LOIO quantile-matching: fit on the other 37 images, apply to each held-out one
    pred = pred.copy()
    pred["y_cal"] = loio_calibrate(pred, lambda rp, rt, hp: quantile_match(hp, rp, rt))

    g = pred[pred["obs_id"] == OBS]
    lab = pd.read_parquet(REPO / "dataset_v2" / "labels" / f"{OBS}.parquet")
    lab = lab[lab["scale_idx"] == 2][["ti", "tj", "xmin", "ymin", "xmax", "ymax"]]
    d = g.merge(lab, on=["ti", "tj"], how="left", validate="one_to_one")
    ti, tj = d["ti"].to_numpy(), d["tj"].to_numpy()
    truth, _, _ = tiles_to_raster(ti, tj, d["y_true"].to_numpy())
    raw, _, _ = tiles_to_raster(ti, tj, np.clip(d["y_pred"].to_numpy(), 0, None))
    cal, _, _ = tiles_to_raster(ti, tj, np.clip(d["y_cal"].to_numpy(), 0, None))
    extent = (d["xmin"].min(), d["xmax"].max(), d["ymin"].min(), d["ymax"].max())

    m_raw = compression_metrics(d["y_true"].to_numpy(), d["y_pred"].to_numpy())
    m_cal = compression_metrics(d["y_true"].to_numpy(), d["y_cal"].to_numpy())
    vmax = np.nanpercentile(d["y_true"][d["y_true"] > 0], 99)
    norm = LogNorm(vmin=max(vmax * 1e-3, 1e-4), vmax=vmax)
    cmap = copy(plt.cm.inferno); cmap.set_bad(alpha=0.0)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    panels = [(truth, "TRUE area-fraction (HiRISE)", None),
              (raw, f"PRED raw mlp_reg\n(top-bin {m_raw['top_ratio']:.2f})", None),
              (cal, f"PRED + quantile-match (de-compressed)\n(top-bin {m_cal['top_ratio']:.2f})", None)]
    for ax, (r, title, _) in zip(axes, panels):
        rr = np.ma.masked_invalid(r); rr = np.ma.masked_where(rr <= 0, rr)
        im = ax.imshow(rr, cmap=cmap, norm=norm, extent=extent, origin="upper",
                       interpolation="nearest")
        ax.set_title(title, fontsize=10.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, location="right")
    cbar.set_label("boulder area-fraction (log)")
    fig.suptitle(f"DRAFT — Tier-2 abundance with calibration ({OBS}). Quantile-matching "
                 f"recovers the high tail + true-zero lows;\nper-image Spearman "
                 f"{spearmanr(d['y_true'], d['y_pred']).correlation:.2f} unchanged "
                 "(ranking-preserving). Calibration fit LOIO on the other 37 images.",
                 fontsize=11, y=0.99)
    out = FIG / "model_evidence_tier2_map_calibrated.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}")
    print(f"  raw top-bin {m_raw['top_ratio']:.3f} near0 {m_raw['near_zero_pred']:.1%} "
          f"-> cal top-bin {m_cal['top_ratio']:.3f} near0 {m_cal['near_zero_pred']:.1%}")


if __name__ == "__main__":
    main()
