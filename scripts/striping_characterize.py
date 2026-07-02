"""PLAN_StripingArtifact §1a — characterize the regional-map striping artifact (geometry).

Cheap, no re-inference. Operates only on the per-tile GeoTIFFs in ``reports/map_region/``
(abundance = calibrated; prob_raw = raw model output before qmatch). For each tile we estimate,
via a windowed 2-D FFT of the detrended raster, the dominant orientation of any periodic
structure and its anisotropy, for BOTH abundance and prob_raw (to tell whether striping is in
the raw output or introduced by qmatch). We also report the directional banding metric.

Finding (DECISIONS 2026-06-18c): the structure is APERIODIC -> FFT anisotropy is weak (~1.3)
and orientation unreliable; the robust facts are (a) it is identical in prob_raw & abundance
(raw, not qmatch) and (b) banding is weak and not strongly vertical. The decisive test is
``scripts/striping_seam_test.py`` (§1b/§2). Analysis logic lives in ``src/striping.py``.

Outputs: reports/figures/striping_fft_<tile>.png, striping_orientation_summary.png,
striping_characterize_summary.csv.   Run: conda run -n geospatial python scripts/striping_characterize.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.striping import (MAP_DIR, all_map_tiles, angular_radial_power, banding_indices,
                          detrend, load_raster)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DETAIL = {"E4_N44", "E8_N44", "E0_N40", "E4_N40", "E-8_N32"}


def analyse(raw):
    field, finite = detrend(raw)
    field = np.where(finite, field, 0.0)
    power, centers, apow = angular_radial_power(field)
    dom = float(centers[np.argmax(apow)])
    aniso = float(apow.max() / (apow.mean() + 1e-30))
    vi, hi = banding_indices(np.where(finite, field, np.nan), finite)
    disp = np.where(finite, field, np.nan)
    return dict(power=power, centers=centers, apow=apow, dom=dom, aniso=aniso,
                vi=vi, hi=hi, disp=disp)


def per_tile_figure(tile, ab, pr):
    fig, ax = plt.subplots(2, 2, figsize=(12, 11))
    im0 = ax[0, 0].imshow(ab["disp"], cmap="magma", vmin=np.nanpercentile(ab["disp"], 2),
                          vmax=np.nanpercentile(ab["disp"], 98))
    ax[0, 0].set_title(f"{tile} abundance (detrended)"); plt.colorbar(im0, ax=ax[0, 0], fraction=0.046)
    lp = np.log10(ab["power"] + 1e-12)
    ax[0, 1].imshow(lp, cmap="viridis", vmin=np.percentile(lp, 60), vmax=np.percentile(lp, 99.5))
    ax[0, 1].axhline(lp.shape[0] / 2, color="w", lw=0.4, alpha=0.5)
    ax[0, 1].axvline(lp.shape[1] / 2, color="w", lw=0.4, alpha=0.5)
    ax[0, 1].set_title("abundance 2-D log-power (center=DC)")
    im2 = ax[1, 0].imshow(pr["disp"], cmap="magma", vmin=np.nanpercentile(pr["disp"], 2),
                          vmax=np.nanpercentile(pr["disp"], 98))
    ax[1, 0].set_title(f"{tile} prob_raw (detrended)"); plt.colorbar(im2, ax=ax[1, 0], fraction=0.046)
    ax[1, 1].plot(ab["centers"], ab["apow"], "-o", ms=3, label="abundance")
    ax[1, 1].plot(pr["centers"], pr["apow"], "-s", ms=3, label="prob_raw")
    ax[1, 1].axvline(0, color="k", ls=":", lw=0.8); ax[1, 1].axvline(90, color="grey", ls=":", lw=0.8)
    ax[1, 1].set_xlabel("wavevector orientation (deg from East)\n0=vertical stripes (N-S), 90=horizontal")
    ax[1, 1].set_ylabel("normalized angular power")
    ax[1, 1].set_title(f"abund vs raw  (aniso {ab['aniso']:.2f}; weak => aperiodic)")
    ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / f"striping_fft_{tile}.png", dpi=110); plt.close(fig)


def main():
    tiles = all_map_tiles()
    print(f"{len(tiles)} tiles")
    rows = []
    for t in tiles:
        ab = analyse(load_raster(MAP_DIR / f"{t}_abundance.tif"))
        pr = analyse(load_raster(MAP_DIR / f"{t}_prob_raw.tif"))
        rows.append(dict(tile=t, ab_orient=ab["dom"], ab_aniso=ab["aniso"], ab_vert=ab["vi"],
                         ab_horiz=ab["hi"], pr_orient=pr["dom"], pr_aniso=pr["aniso"]))
        print(f"  {t:9s} abund orient={ab['dom']:5.0f}deg aniso={ab['aniso']:4.2f} "
              f"bandV/H={ab['vi']:.3f}/{ab['hi']:.3f} | raw orient={pr['dom']:5.0f}deg", flush=True)
        if t in DETAIL:
            per_tile_figure(t, ab, pr)

    with open(FIG_DIR / "striping_characterize_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    angs = np.array([r["ab_orient"] for r in rows]); anis = np.array([r["ab_aniso"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(angs, bins=np.linspace(0, 180, 19), color="steelblue", edgecolor="k")
    ax[0].axvline(0, color="r", ls="--", label="vertical (N-S track)")
    ax[0].axvline(90, color="g", ls="--", label="horizontal")
    ax[0].set_xlabel("dominant wavevector orientation (deg from East)"); ax[0].set_ylabel("# tiles")
    ax[0].set_title("Abundance dominant orientation"); ax[0].legend()
    sc = ax[1].scatter(angs, anis, c=anis, cmap="plasma")
    for r in rows:
        ax[1].annotate(r["tile"], (r["ab_orient"], r["ab_aniso"]), fontsize=6, alpha=0.7)
    ax[1].set_xlabel("dominant orientation (deg from East)"); ax[1].set_ylabel("anisotropy (peak/mean)")
    ax[1].set_title("Low anisotropy (~1.3) => no strong periodic stripe"); plt.colorbar(sc, ax=ax[1])
    fig.tight_layout(); fig.savefig(FIG_DIR / "striping_orientation_summary.png", dpi=120); plt.close(fig)

    vert = np.mean((angs < 20) | (angs > 160)); horiz = np.mean((angs > 70) & (angs < 110))
    print(f"\nSUMMARY: {len(rows)} tiles | vertical-dom {100*vert:.0f}% | horizontal-dom "
          f"{100*horiz:.0f}% | median aniso {np.median(anis):.2f}")


if __name__ == "__main__":
    main()
