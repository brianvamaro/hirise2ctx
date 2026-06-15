"""Tier-2 abundance-map figure for docs/model_evidence.md (PLAN_FM §2.5 / §8).

Held-out true vs predicted boulder area-fraction for one image, both on the same
log colour scale -- shows the Tier-2 regressor reproduces the spatial *ordering*
of abundance (where it is denser vs sparser), the rank-faithful product, while the
absolute high tail is compressed (§8 caveat). From the banked single-stage mlp_reg
predictions (group-aware LOIO).

Output: reports/figures/model_evidence_tier2_map.png
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.mapping import tiles_to_raster  # noqa: E402

PRED = (ROOT / "models" / "fang_tier2" / "tier2_mlp_reg_emb_fractional_area_S32"
        / "1e01ad8b17447599" / "predictions.parquet")
FIG = ROOT / "reports" / "figures"
OBS = "ESP_053989_2260"   # rich plains, good dynamic range; same image as basis/product


def main():
    pred = pd.read_parquet(PRED)
    g = pred[pred["obs_id"] == OBS]
    lab = pd.read_parquet(ROOT / "dataset_v2" / "labels" / f"{OBS}.parquet")
    lab = lab[lab["scale_idx"] == 2][["ti", "tj", "xmin", "ymin", "xmax", "ymax"]]
    d = g.merge(lab, on=["ti", "tj"], how="left", validate="one_to_one")
    ti, tj = d["ti"].to_numpy(), d["tj"].to_numpy()
    truth, _, _ = tiles_to_raster(ti, tj, d["y_true"].to_numpy())
    predr, _, _ = tiles_to_raster(ti, tj, np.clip(d["y_pred"].to_numpy(), 0, None))
    extent = (d["xmin"].min(), d["xmax"].max(), d["ymin"].min(), d["ymax"].max())

    rho = spearmanr(d["y_true"], d["y_pred"]).correlation
    vmax = np.nanpercentile(d["y_true"][d["y_true"] > 0], 99)
    norm = LogNorm(vmin=max(vmax * 1e-3, 1e-4), vmax=vmax)
    cmap = copy(plt.cm.inferno); cmap.set_bad(alpha=0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.6))
    for ax, raster, title in [
        (axes[0], truth, "TRUE boulder area-fraction (HiRISE)"),
        (axes[1], predr, "PREDICTED area-fraction (CTX only, held-out)")]:
        r = np.ma.masked_invalid(raster)
        r = np.ma.masked_where(r <= 0, r)
        im = ax.imshow(r, cmap=cmap, norm=norm, extent=extent, origin="upper",
                       interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02, location="right")
    cbar.set_label("boulder area-fraction (log)")
    fig.suptitle(f"Tier-2 calibrated abundance — {OBS}  ·  per-image Spearman "
                 f"{rho:.2f}  ·  ordering faithful, high tail ~30% compressed",
                 fontsize=12, y=0.98)
    out = FIG / "model_evidence_tier2_map.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"Wrote {out}  (rho={rho:.3f}, vmax={vmax:.4f})")


if __name__ == "__main__":
    main()
