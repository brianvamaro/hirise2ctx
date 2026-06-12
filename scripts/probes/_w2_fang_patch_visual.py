"""Visual check for Fang-ViT embedding input alignment (W2 Phase 2, PLAN_CNN.md 5.1).

For a few S=64 tiles per image: draw the CTX window with the tile (64 px, solid) and its
192-px context box (dashed) outlined, and next to it the two actual ViT inputs -- the
cached S64 patch and the 192-px window slice with its center 64x64 outlined. The center
of the 192-px slice is re-verified bit-identical to the cached patch (the same assert
the extractor samples), and the verdict is printed on the figure.

Tile picks per image: (a) max fractional_area (boulder-rich), (b) the tile whose 192-px
box sits closest to the window boundary (stresses the slice arithmetic), (c) a median-
label tile near the window center.

Figures -> reports/figures/19_w2_fang_patch_alignment_{obs}.png

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/probes/_w2_fang_patch_visual.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

DATASET_DIR = REPO_ROOT / "dataset_v2"
FIG_DIR = REPO_ROOT / "reports" / "figures"
OBS_IDS = ["ESP_042964_2160", "ESP_076499_1160"]
TILE_PX = 64
CONTEXT_PX = 192
COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]


def _load_window(obs_id: str) -> tuple[np.ndarray, int, int]:
    import rasterio

    sidecar = json.loads((DATASET_DIR / "labels" / f"{obs_id}.json").read_text(encoding="utf-8"))
    with rasterio.open(sidecar["ctx_window_tif"]) as src:
        arr = src.read(1).astype(np.uint8, copy=False)
    return arr, int(sidecar["mosaic_row_origin"]), int(sidecar["mosaic_col_origin"])


def _pick_tiles(df: pd.DataFrame, H: int, W: int) -> pd.DataFrame:
    """max-label, nearest-to-window-edge, and median-label tile (distinct)."""
    df = df.copy()
    r0 = df["r_win"] - TILE_PX
    c0 = df["c_win"] - TILE_PX
    df["edge_dist"] = np.minimum.reduce([
        r0.to_numpy(), c0.to_numpy(),
        (H - (r0 + CONTEXT_PX)).to_numpy(), (W - (c0 + CONTEXT_PX)).to_numpy(),
    ])
    picks = [df["fractional_area"].idxmax(), df["edge_dist"].idxmin()]
    med = df.loc[~df.index.isin(picks), "fractional_area"]
    picks.append((med - med.median()).abs().idxmin())
    return df.loc[picks]


def figure_one(obs_id: str) -> None:
    from src.modeling.loaders import load_context_patch_stack

    arr, row0, col0 = _load_window(obs_id)
    H, W = arr.shape

    feats = pd.read_parquet(DATASET_DIR / "features" / f"{obs_id}.parquet",
                            columns=["tile_size_px", "ti", "tj", "patch_idx_S64"])
    feats = feats[feats["tile_size_px"] == TILE_PX]
    labels = pd.read_parquet(DATASET_DIR / "labels" / f"{obs_id}.parquet",
                             columns=["tile_size_px", "ti", "tj", "fractional_area"])
    labels = labels[labels["tile_size_px"] == TILE_PX]
    df = feats.merge(labels, on=["tile_size_px", "ti", "tj"], validate="one_to_one")
    df["r_win"] = df["ti"] * TILE_PX - row0
    df["c_win"] = df["tj"] * TILE_PX - col0
    picks = _pick_tiles(df, H, W)

    stack = load_context_patch_stack(obs_id, TILE_PX, dataset_dir=DATASET_DIR)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(3, 4, width_ratios=[2.2, 1, 1, 0.06], wspace=0.15, hspace=0.25)
    ax_w = fig.add_subplot(gs[:, 0])
    lo, hi = np.percentile(arr[arr > 0], [2, 98])
    ax_w.imshow(arr, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    ax_w.set_title(f"{obs_id} -- cached CTX window ({H}x{W} px)\n"
                   "solid = S=64 tile, dashed = 192-px ViT context input", fontsize=10)
    ax_w.set_xticks([]); ax_w.set_yticks([])

    for k, (_, t) in enumerate(picks.iterrows()):
        col = COLORS[k]
        r, c = int(t.r_win), int(t.c_win)
        ax_w.add_patch(Rectangle((c, r), TILE_PX, TILE_PX, fill=False, ec=col, lw=1.4))
        ax_w.add_patch(Rectangle((c - TILE_PX, r - TILE_PX), CONTEXT_PX, CONTEXT_PX,
                                 fill=False, ec=col, lw=1.2, ls="--"))

        own = stack[int(t.patch_idx_S64)]
        big = arr[r - TILE_PX: r - TILE_PX + CONTEXT_PX, c - TILE_PX: c - TILE_PX + CONTEXT_PX]
        center_ok = np.array_equal(big[TILE_PX: 2 * TILE_PX, TILE_PX: 2 * TILE_PX], own)

        ax_p = fig.add_subplot(gs[k, 1])
        ax_p.imshow(own, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax_p.set_title(f"cached S64 patch  ti={int(t.ti)} tj={int(t.tj)}\n"
                       f"fa={t.fractional_area:.4f}", fontsize=8, color=col)
        ax_c = fig.add_subplot(gs[k, 2])
        ax_c.imshow(big, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        ax_c.add_patch(Rectangle((TILE_PX, TILE_PX), TILE_PX, TILE_PX, fill=False, ec=col, lw=1.4))
        ax_c.set_title(f"192-px ViT input\ncenter == cached patch: {center_ok}", fontsize=8,
                       color=("#222222" if center_ok else "red"))
        for ax in (ax_p, ax_c):
            ax.set_xticks([]); ax.set_yticks([])
        if not center_ok:
            raise AssertionError(f"{obs_id}: center mismatch at ti={t.ti} tj={t.tj}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"19_w2_fang_patch_alignment_{obs_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


def main() -> int:
    for obs_id in OBS_IDS:
        figure_one(obs_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
