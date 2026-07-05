"""F leg B diagnostic figures: what did the F embedder actually see?

Fig 1 (f_leg_b_diag_scatter.png): per-image ΔAUC sorted bar chart + ΔAUC vs
pre-normalization I/F IQR (log x, colored by I/F median) — the bimodal pattern
and its contrast/illumination correlates.

Fig 2 (f_leg_b_diag_gallery.png): for the 3 worst collapsed images + 3 best
improvers, the baseline mosaic window vs the F composite uint8 (identical 0-255
gray scale), full window + 512-px native-res zoom — texture differences are
directly visible.

Run: conda run --no-capture-output -n geospatial python -u scripts/probes/_f_leg_b_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

import f_leg_b_embed as fe  # composite_crops + normalization (laptop embed code path)

FIG = REPO / "reports" / "figures"
LABELS_DIR = REPO / "dataset_v2" / "labels"
DIAG = REPO / "reports" / "f_leg_b" / "diag_per_image.csv"

COLLAPSED = ["ESP_045550_2180", "ESP_046328_2180", "ESP_069763_2235"]
IMPROVERS = ["ESP_055978_2270", "ESP_042964_2160", "ESP_046959_2225"]
ZOOM = 512  # native-res zoom half-window (px)


def fig_scatter(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    d = df.sort_values("d_auc")
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in d["d_auc"]]
    axes[0].barh(range(len(d)), d["d_auc"], color=colors)
    axes[0].set_yticks(range(len(d)))
    axes[0].set_yticklabels([o.replace("ESP_", "") for o in d["obs_id"]], fontsize=7)
    axes[0].axvline(0, color="k", lw=0.8)
    axes[0].axvline(-0.02, color="orange", ls="--", lw=1, label="gate −0.02 (on median)")
    axes[0].set_xlabel("Δ per-image AUC (F − baseline)")
    axes[0].set_title("Leg B per-image ΔAUC — bimodal, not uniform")
    axes[0].legend(fontsize=8)

    sc = axes[1].scatter(d["if_median"], d["d_auc"], c=d["if_iqr"], cmap="viridis",
                         s=60, edgecolor="k", lw=0.5)
    axes[1].axhline(0, color="k", lw=0.8)
    for _, r in d.iterrows():
        if abs(r["d_auc"]) > 0.12:
            axes[1].annotate(r["obs_id"].replace("ESP_", ""),
                             (r["if_median"], r["d_auc"]), fontsize=6,
                             xytext=(4, 3), textcoords="offset points")
    axes[1].set_xlabel("composite I/F median before normalization (illumination proxy)")
    axes[1].set_ylabel("Δ per-image AUC")
    axes[1].set_title("DIM scenes collapse (ρ=+0.35) — illumination is the live correlate;\n"
                      "post-norm uint8 contrast is pinned at IQR≈27.7 for all (ratio ρ=+0.09, null)")
    plt.colorbar(sc, ax=axes[1], label="I/F IQR (concat crops)")

    fig.tight_layout()
    out = FIG / "f_leg_b_diag_scatter.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def load_pair(obs_id: str) -> tuple[np.ndarray, np.ndarray]:
    """(mosaic uint8, F composite uint8) on the same window grid."""
    sc = json.loads((LABELS_DIR / f"{obs_id}.json").read_text(encoding="utf-8"))
    with rasterio.open(sc["ctx_window_tif"]) as ds:
        mosaic = ds.read(1)
        H, W = ds.height, ds.width
    comp = fe.composite_crops(obs_id, int(sc["mosaic_row_origin"]),
                              int(sc["mosaic_col_origin"]), H, W)
    return mosaic.astype(np.uint8), comp


def fig_gallery(df: pd.DataFrame) -> None:
    picks = COLLAPSED + IMPROVERS
    aucs = df.set_index("obs_id")
    n = len(picks)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.6 * n))

    for i, obs in enumerate(picks):
        mosaic, comp = load_pair(obs)
        H, W = mosaic.shape
        r0, c0 = H // 2 - ZOOM // 2, W // 2 - ZOOM // 2
        zm = mosaic[r0:r0 + ZOOM, c0:c0 + ZOOM]
        zc = comp[r0:r0 + ZOOM, c0:c0 + ZOOM]
        k = max(1, min(H, W) // 900)  # decimate full views for a sane figure size

        row = aucs.loc[obs]
        kind = "COLLAPSED" if obs in COLLAPSED else "IMPROVER"
        panels = [
            (mosaic[::k, ::k], f"{obs}  [{kind}]\nmosaic (baseline)  AUC {row['auc_base']:.3f}"),
            (comp[::k, ::k], f"F composite (perframe)  AUC {row['auc_f']:.3f}"
                             f"  Δ {row['d_auc']:+.3f}\nI/F IQR {row['if_iqr']:.3f}"),
            (zm, "mosaic zoom (512 px native)"),
            (zc, "F zoom (512 px native)"),
        ]
        for j, (img, title) in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        print(f"  {obs}: rendered", flush=True)

    fig.suptitle("Leg B: what the embedder saw — baseline mosaic vs F perframe composite "
                 "(identical 0-255 gray scale)", fontsize=12, y=1.0)
    fig.tight_layout()
    out = FIG / "f_leg_b_diag_gallery.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> None:
    df = pd.read_csv(DIAG)
    fig_scatter(df)
    fig_gallery(df)


if __name__ == "__main__":
    main()
