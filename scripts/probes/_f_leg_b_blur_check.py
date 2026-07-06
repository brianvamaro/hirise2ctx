"""Is the F path softer than the mosaic?  High-frequency content, F crop vs mosaic.

The F pipeline resamples twice with bilinear (cam2map -> 5 m grid, extract reproject ->
mosaic-aligned 5 m grid) where the Murray mosaic had its own single projection + blend.
If F windows are measurably blurrier, texture-based embeddings lose signal — a
mapping-independent skill floor consistent with the ≈ −0.034 convergence.

Metric (stretch-invariant): high-frequency energy fraction on the SAME 1024² native
patch — HF = var(Laplacian) / var(signal) and gradient-energy fraction
GE = var(∇) / var(signal).  Linear maps (all our stretches) cancel in the ratio, so
mosaic uint8 and F float I/F are directly comparable.  Ratio F/mosaic < 1 = F softer.

Run: conda run --no-capture-output -n geospatial python -u scripts/probes/_f_leg_b_blur_check.py
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

CROPS_DIR = REPO / "reports" / "f_leg_b" / "obs_crops"
LABELS_DIR = REPO / "dataset_v2" / "labels"
PATCH = 1024


def hf_stats(a: np.ndarray) -> tuple[float, float]:
    """(laplacian HF fraction, gradient energy fraction) over finite pixels."""
    a = a.astype(np.float64)
    lap = (a[1:-1, 2:] + a[1:-1, :-2] + a[2:, 1:-1] + a[:-2, 1:-1] - 4 * a[1:-1, 1:-1])
    gx = np.diff(a, axis=1)[:-1, :]
    gy = np.diff(a, axis=0)[:, :-1]
    v = np.nanvar(a)
    if v <= 0:
        return np.nan, np.nan
    return float(np.nanvar(lap) / v), float((np.nanvar(gx) + np.nanvar(gy)) / v)


def main() -> None:
    rows = []
    for crop_path in sorted(CROPS_DIR.glob("*_ifcrop.tif")):
        parts = crop_path.name.replace("_ifcrop.tif", "").split("_")
        obs_id = "_".join(parts[:3])
        sidecar = LABELS_DIR / f"{obs_id}.json"
        if not sidecar.exists():
            continue
        sc = json.loads(sidecar.read_text(encoding="utf-8"))
        # only need one crop per obs — skip if we already have this obs
        if any(r["obs_id"] == obs_id for r in rows):
            continue

        with rasterio.open(crop_path) as src:
            if src.height < PATCH + 200 or src.width < PATCH + 200:
                continue
            r0 = (src.height - PATCH) // 2
            c0 = (src.width - PATCH) // 2
            fpatch = src.read(1, window=Window(c0, r0, PATCH, PATCH))
        fin = np.isfinite(fpatch) & (fpatch > 0)
        if fin.mean() < 0.98:   # need a fully-covered patch for a fair spectrum
            continue

        # the crop grid is anchored at the window bounds, so patch pixel coords
        # map 1:1 onto the ctx_window_tif grid
        with rasterio.open(sc["ctx_window_tif"]) as ds:
            if ds.height != fpatch.shape[0] and (ds.height < r0 + PATCH or ds.width < c0 + PATCH):
                continue
            mpatch = ds.read(1, window=Window(c0, r0, PATCH, PATCH)).astype(np.float32)
        if (mpatch > 0).mean() < 0.98:
            continue

        f_hf, f_ge = hf_stats(np.where(fin, fpatch, np.nan))
        m_hf, m_ge = hf_stats(np.where(mpatch > 0, mpatch, np.nan))
        rows.append(dict(obs_id=obs_id, f_hf=f_hf, m_hf=m_hf, hf_ratio=f_hf / m_hf,
                         f_ge=f_ge, m_ge=m_ge, ge_ratio=f_ge / m_ge))
        print(f"  {obs_id}: HF F/mosaic = {f_hf / m_hf:.3f}   "
              f"gradE F/mosaic = {f_ge / m_ge:.3f}", flush=True)

    df = pd.DataFrame(rows)
    print(f"\n{len(df)} images with clean 1024² patches")
    print(f"HF ratio (F/mosaic):    median {df.hf_ratio.median():.3f}  "
          f"IQR {df.hf_ratio.quantile(.25):.3f}–{df.hf_ratio.quantile(.75):.3f}")
    print(f"gradE ratio (F/mosaic): median {df.ge_ratio.median():.3f}  "
          f"IQR {df.ge_ratio.quantile(.25):.3f}–{df.ge_ratio.quantile(.75):.3f}")
    print("\nratio < 1 = F is softer (less high-frequency texture) than the mosaic")
    out = REPO / "reports" / "f_leg_b" / "blur_check.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
