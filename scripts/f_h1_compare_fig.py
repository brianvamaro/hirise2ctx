"""H1 before/after figure: the frame-block artifact, log-minnaert F vs H1 (minnaert_center).

Builds a 2x2 comparison from the CACHED pilot predictions (no GPU) on the 7 E8_N44 frames:
  row 1  median composite (deploy-style map; value = P(boulder-rich), fa > 1e-2 —
          the raw classifier probability, NOT the CalibrationLayer-adjusted abundance)
  row 2  per-frame-mean choropleth (each frame painted its mean prediction — the artifact,
          isolated: flat within a frame, jumps at seams == the striping)
  col A  F log-minnaert  (leg B, eta2 median 0.179)
  col B  H1 minnaert_center  (eta2 median 0.081)

Reuses f_pilot_crop's frame_labels / eta2 / geometry so the composites match the pilot exactly.

Run:
  conda run -n geospatial python scripts/f_h1_compare_fig.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import numpy as np
from rasterio.transform import Affine
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.f_pilot_crop import frame_labels, preds_path
from src.striping import eta2

FIG = REPO / "reports" / "figures"
HEAD = "f_wl"                       # both stores were run with --head-name f_wl (default)
PANELS = [("minnaert_log", "F log-minnaert (leg B)"),
          ("minnaert_center", "H1 minnaert_center")]


def composites(mapping: str):
    """Return (median_composite, frame_mean_choropleth, labels, eta2_median)."""
    z = np.load(preds_path(mapping), allow_pickle=False)
    pids = [str(p) for p in z["pids"]]
    transform = Affine(*z["transform"])
    stack = z[HEAD]                                    # (n_frames, H, W)
    shape = stack.shape[1:]
    labels = frame_labels(pids, shape, transform)
    valid = np.isfinite(stack)

    # deploy-style median composite over overlapping frames
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(stack, axis=0).astype(np.float32)

    # SeamMap partition (each cell from its selected frame) -> per-frame-mean choropleth
    part = np.full(shape, np.nan, dtype=np.float32)
    for i in range(len(pids)):
        sel = (labels == i) & valid[i]
        part[sel] = stack[i][sel]
    chor = np.full(shape, np.nan, dtype=np.float32)
    for i in range(len(pids)):
        sel = (labels == i) & np.isfinite(part)
        if sel.sum() >= 30:
            chor[labels == i] = float(np.nanmean(part[sel]))

    fin = np.isfinite(med) & (labels >= 0)
    e = float(eta2(med, labels, fin))
    return med, chor, labels, e


def main() -> None:
    data = {m: composites(m) for m, _ in PANELS}
    # shared color scale from the median composites (robust top percentile)
    vmax = np.nanpercentile(np.concatenate(
        [data[m][0][np.isfinite(data[m][0])] for m, _ in PANELS]), 99)

    fig, ax = plt.subplots(2, 2, figsize=(11, 10.5))
    for c, (m, title) in enumerate(PANELS):
        med, chor, _, e = data[m]
        im0 = ax[0, c].imshow(med, cmap="magma", vmax=vmax)
        ax[0, c].set_title(f"{title}\nmedian composite", fontsize=11)
        plt.colorbar(im0, ax=ax[0, c], fraction=0.046, label="P(boulder-rich)  [fa > 1e-2]")
        im1 = ax[1, c].imshow(chor, cmap="magma", vmax=vmax)
        ax[1, c].set_title(f"frame-mean choropleth — η² (median comp.) = {e:.3f}",
                           fontsize=11)
        plt.colorbar(im1, ax=ax[1, c], fraction=0.046,
                     label="frame-mean P(boulder-rich)")
        for a in (ax[0, c], ax[1, c]):
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Striping artifact on the E8_N44 crop — H1 (per-frame log-median centering) "
                 "halves the frame blocks\n"
                 "bottom row isolates the artifact: flat within a frame, jumps at seams. "
                 "(baselines on this crop: mosaic raw η²=0.196, A1 η²=0.141)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "f_h1_before_after_choropleth.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")
    for m, _ in PANELS:
        print(f"  {m}: eta2 median composite = {data[m][3]:.4f}")


if __name__ == "__main__":
    main()
