"""A1 payoff test on an E8_N44 crop — does per-frame A1 normalization REDUCE the frame-block
artifact (eta^2) in the model output?

Runs inference twice on the SAME ~75 km / 8-frame crop, on the raw P(rich) (no calibrator, so we
isolate the model's frame-sensitivity):
  * baseline: raw CTX + baseline head (`models/deployable`)
  * A1:       per-frame robust offset+gain CTX (SeamMap partition) + A1 head (`models/deployable_a1`)
Then compares eta^2 (variance of prediction explained by source frame) and renders a before/after
frame-mean choropleth. If A1 << baseline eta^2, the mitigation works (pairs with the −0.024 LOIO cost
to judge the trade).

Run: conda run -n geospatial python scripts/striping_a1_infer_crop.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; precede numpy

import numpy as np
import rasterio
from rasterio.features import rasterize
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.fm_embeddings import FangEmbedder
from src.mapping import predict_window, read_tile_window
from src.modeling.mlp_head import DeployableHead
from src.striping import (A1_REF_IQR, A1_REF_MEDIAN, CTX_ZIP_DIR, MAP_DIR, _inner_tif_name,
                          a1_apply, a1_stats, eta2, frame_label_map, load_frames,
                          read_ctx_on_grid)

FIG = REPO / "reports" / "figures"
TILE = "E8_N44"
R0, C0, SIZE = 1504, 8992, 15008   # multiples of 32 (≈ the previewed 8-frame crop)
BASE_DIR = REPO / "models" / "deployable" / "86c51a5dca220f63"
A1_DIR = REPO / "models" / "deployable_a1" / "86c51a5dca220f63"


def frame_mean_choropleth(raster, labels, n):
    out = np.full(raster.shape, np.nan)
    fin = np.isfinite(raster) & (labels >= 0)
    for i in range(n):
        sel = fin & (labels == i)
        if sel.sum() >= 30:
            out[labels == i] = raster[sel].mean()
    return out


def main():
    # per-frame robust (median, IQR) at 160 m, indexed to load_frames order
    ctx160 = read_ctx_on_grid(TILE, MAP_DIR / f"{TILE}_abundance.tif")
    frames = load_frames(TILE)
    L160 = frame_label_map(TILE, frames)
    fstats = {}
    for i in range(len(frames)):
        sel = (L160 == i) & np.isfinite(ctx160)
        if sel.sum() >= 50:
            fstats[i] = a1_stats(np.where(sel, ctx160, 0))
    print(f"{TILE}: {len(fstats)} frames with stats", flush=True)

    zip_path = CTX_ZIP_DIR / f"{TILE}.zip"
    inner = _inner_tif_name(zip_path)
    window = read_tile_window(zip_path, inner, R0, C0, SIZE)
    print(f"read crop {window.data.shape} at ({R0},{C0})", flush=True)

    embedder = FangEmbedder.load()
    base_head = DeployableHead.load(BASE_DIR)
    a1_head = DeployableHead.load(A1_DIR)

    # ---- baseline inference (raw prob) ----
    pred_b = predict_window(window, embedder, base_head, tile_px=32, batch=256, calibrator=None)
    print(f"baseline predicted: {np.isfinite(pred_b.raster).sum()} tiles", flush=True)

    # ---- A1: per-frame normalize the native crop, then infer with the A1 head ----
    Lnat = rasterize(((g, i) for i, g in enumerate(frames.geometry)),
                     out_shape=window.data.shape, transform=window.transform,
                     fill=-1, dtype="int16", all_touched=False)
    # R38: call `a1_apply` rather than re-inlining the stretch. This copy had drifted to a
    # `[0, 255]` clip, which wrote legitimately dark terrain as the mosaic nodata sentinel; the
    # shared definition floors valid pixels at `A1_VALID_FLOOR` so DN 0 means only "no data".
    arr = window.data.copy()
    nodata_mask = window.data == 0                 # from the RAW DN, before normalization
    for i, (med, iqr) in fstats.items():
        sel = (Lnat == i) & (window.data > 0)
        if sel.any():
            arr[sel] = a1_apply(window.data, med, iqr)[sel]
    arr[nodata_mask] = 0
    window_a1 = replace(window, data=arr.astype(np.uint8))
    pred_a = predict_window(window_a1, embedder, a1_head, tile_px=32, batch=256, calibrator=None,
                            nodata_mask=nodata_mask)
    print(f"A1 predicted: {np.isfinite(pred_a.raster).sum()} tiles", flush=True)

    # ---- eta^2 on the SAME coarse grid (rasters share grid/shape) ----
    Lc = rasterize(((g, i) for i, g in enumerate(frames.geometry)),
                   out_shape=pred_b.raster.shape, transform=pred_b.transform,
                   fill=-1, dtype="int16", all_touched=False)
    fb = np.isfinite(pred_b.raster) & (Lc >= 0)
    fa = np.isfinite(pred_a.raster) & (Lc >= 0)
    e_b = eta2(pred_b.raster, Lc, fb)
    e_a = eta2(pred_a.raster, Lc, fa)
    n_fr = len(np.unique(Lc[Lc >= 0]))
    print("\n=== A1 PAYOFF (raw P(rich) frame-coherence on the crop) ===")
    print(f"frames in crop: {n_fr}")
    print(f"eta^2 baseline = {e_b:.4f}")
    print(f"eta^2 A1       = {e_a:.4f}")
    print(f"reduction      = {(1 - e_a / e_b) * 100:.0f}%  ({'A1 reduces the artifact' if e_a < e_b else 'no reduction'})")

    # ---- before/after choropleth ----
    chor_b = frame_mean_choropleth(pred_b.raster, Lc, len(frames))
    chor_a = frame_mean_choropleth(pred_a.raster, Lc, len(frames))
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    vmax = np.nanpercentile(pred_b.raster, 99)
    for a, r, t in [(ax[0, 0], pred_b.raster, f"baseline raw P(rich)  (eta²={e_b:.3f})"),
                    (ax[0, 1], pred_a.raster, f"A1 raw P(rich)  (eta²={e_a:.3f})")]:
        im = a.imshow(r, cmap="magma", vmax=vmax); a.set_title(t); plt.colorbar(im, ax=a, fraction=0.046)
    for a, r, t in [(ax[1, 0], chor_b, "baseline frame-mean (blocks)"),
                    (ax[1, 1], chor_a, "A1 frame-mean (flatter = artifact removed)")]:
        im = a.imshow(r, cmap="magma", vmax=vmax); a.set_title(t); plt.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle(f"A1 payoff on {TILE} crop ({n_fr} frames): "
                 f"eta² {e_b:.3f} → {e_a:.3f} ({(1-e_a/e_b)*100:.0f}% reduction)", fontsize=13)
    fig.tight_layout()
    out = FIG / "striping_a1_payoff.png"
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
