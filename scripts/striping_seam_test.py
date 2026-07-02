"""PLAN_StripingArtifact §1b + §2 — the decisive tests of the CTX-stitching hypothesis.

For the tiles that have an abundance raster + cached Murray CTX zip + cached SeamMap
(``src.striping.equipped_tiles()``) we:

  1. **Directional banding metric** (replaces FFT, which is blind to aperiodic stripes):
     for abundance, prob_raw, and coarsened CTX brightness, measure variance organised into
     vertical bands (col = N-S orbital-track direction) vs horizontal bands (row).
  2. **§1b edge coincidence:** Sobel |grad| of coarsened CTX vs |grad| of abundance; Spearman
     correlation vs a row-shuffled null.
  3. **§2 gold-standard seam test:** rasterize the per-frame footprint boundaries from the
     SeamMap onto the abundance grid; test whether |grad(abundance)| is elevated within a few
     pixels of a seam vs far (permutation test). NOTE underpowered — see DECISIONS 2026-06-18c.

All analysis logic lives in ``src/striping.py``. Outputs: ``reports/figures/striping_seam_*.png``
and ``striping_seam_test_summary.csv``.

Run:  conda run -n geospatial python scripts/striping_seam_test.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.striping import (MAP_DIR, PX_M, banding_indices, detrend, equipped_tiles,
                          grad_mag, load_raster, read_ctx_on_grid, seam_line_mask)

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def main():
    tiles = equipped_tiles()
    print(f"{len(tiles)} fully-equipped tiles: {tiles}")
    rng = np.random.default_rng(0)
    rows = []
    for t in tiles:
        ab_path = MAP_DIR / f"{t}_abundance.tif"
        ab = load_raster(ab_path)
        pr = load_raster(MAP_DIR / f"{t}_prob_raw.tif")
        ctx = read_ctx_on_grid(t, ab_path)

        ab_d, abf = detrend(ab)
        pr_d, prf = detrend(pr)
        ctx_d, ctxf = detrend(ctx)
        ab_vi, ab_hi = banding_indices(ab_d, abf)
        pr_vi, pr_hi = banding_indices(pr_d, prf)
        ctx_vi, ctx_hi = banding_indices(ctx_d, ctxf)

        gab, gctx = grad_mag(ab_d), grad_mag(ctx_d)
        valid = np.isfinite(gab) & np.isfinite(gctx)
        rho_edge = spearmanr(gab[valid], gctx[valid]).statistic if valid.sum() > 100 else np.nan
        gctx_sh = gctx[rng.permutation(gctx.shape[0]), :]
        v2 = np.isfinite(gab) & np.isfinite(gctx_sh)
        rho_null = spearmanr(gab[v2], gctx_sh[v2]).statistic if v2.sum() > 100 else np.nan

        seam, _ = seam_line_mask(t, ab_path)
        dist = distance_transform_edt(~seam)
        near = (dist <= 1) & np.isfinite(gab)
        far = (dist >= 5) & np.isfinite(gab)
        g_near = np.nanmean(gab[near]) if near.sum() else np.nan
        g_far = np.nanmean(gab[far]) if far.sum() else np.nan
        seam_ratio = g_near / g_far if g_far and np.isfinite(g_far) and g_far > 0 else np.nan
        finite_idx = np.flatnonzero(np.isfinite(gab))
        gabf = gab.ravel()
        nnear = int(near.sum())
        null_ratios = np.array([np.nanmean(gabf[rng.choice(finite_idx, nnear, replace=False)]) / g_far
                                for _ in range(200)])
        p_seam = float(np.mean(null_ratios >= seam_ratio)) if np.isfinite(seam_ratio) else np.nan

        rows.append(dict(tile=t, ab_vert=ab_vi, ab_horiz=ab_hi, pr_vert=pr_vi, pr_horiz=pr_hi,
                         ctx_vert=ctx_vi, ctx_horiz=ctx_hi, rho_edge=rho_edge, rho_null=rho_null,
                         seam_grad_ratio=seam_ratio, seam_p=p_seam))
        print(f"  {t:9s} | band V/H abund {ab_vi:.3f}/{ab_hi:.3f}  ctx {ctx_vi:.3f}/{ctx_hi:.3f}"
              f" | edge rho {rho_edge:+.3f} (null {rho_null:+.3f})"
              f" | seam grad x{seam_ratio:.2f} p={p_seam:.3f}", flush=True)

        fig, ax = plt.subplots(2, 3, figsize=(16, 10))
        c = ax[0, 0].imshow(ctx, cmap="gray"); ax[0, 0].set_title(f"{t} CTX brightness + seam lines")
        ax[0, 0].imshow(np.ma.masked_where(~seam, seam), cmap="autumn", alpha=0.5)
        plt.colorbar(c, ax=ax[0, 0], fraction=0.046)
        a1 = ax[0, 1].imshow(ab_d, cmap="magma", vmin=np.nanpercentile(ab_d, 2),
                             vmax=np.nanpercentile(ab_d, 98))
        ax[0, 1].imshow(np.ma.masked_where(~seam, seam), cmap="cool", alpha=0.35)
        ax[0, 1].set_title("abundance (detrended) + seams"); plt.colorbar(a1, ax=ax[0, 1], fraction=0.046)
        a2 = ax[0, 2].imshow(ctx_d, cmap="RdBu_r", vmin=np.nanpercentile(ctx_d, 2),
                             vmax=np.nanpercentile(ctx_d, 98))
        ax[0, 2].set_title("CTX brightness (detrended)"); plt.colorbar(a2, ax=ax[0, 2], fraction=0.046)
        ax[1, 0].imshow(gctx, cmap="viridis", vmax=np.nanpercentile(gctx, 98)); ax[1, 0].set_title("|grad CTX|")
        ax[1, 1].imshow(gab, cmap="viridis", vmax=np.nanpercentile(gab, 98)); ax[1, 1].set_title("|grad abundance|")
        dbins = np.arange(0, 12)
        prof = [np.nanmean(gab[(dist >= d) & (dist < d + 1)]) for d in dbins]
        ax[1, 2].plot(dbins * PX_M / 1000, prof, "-o")
        ax[1, 2].set_xlabel("distance to nearest CTX seam (km)"); ax[1, 2].set_ylabel("mean |grad abundance|")
        ax[1, 2].set_title(f"seam grad ratio x{seam_ratio:.2f} (p={p_seam:.3f})\n"
                           f"edge rho {rho_edge:+.3f} vs null {rho_null:+.3f}")
        fig.tight_layout(); fig.savefig(FIG_DIR / f"striping_seam_{t}.png", dpi=110); plt.close(fig)

    with open(FIG_DIR / "striping_seam_test_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    g = lambda k: np.array([r[k] for r in rows])
    print("\n=== VERDICT (medians across tiles) ===")
    print(f"abundance vertical/horizontal banding: {np.nanmedian(g('ab_vert')):.3f} / {np.nanmedian(g('ab_horiz')):.3f}")
    print(f"CTX vertical banding: {np.nanmedian(g('ctx_vert')):.3f}")
    print(f"edge |grad| corr abundance~CTX: rho {np.nanmedian(g('rho_edge')):+.3f} vs null {np.nanmedian(g('rho_null')):+.3f}")
    print(f"seam |grad| ratio: x{np.nanmedian(g('seam_grad_ratio')):.2f} ({np.mean(g('seam_p') < 0.05)*100:.0f}% tiles p<0.05)")
    print(f"Figures + CSV -> {FIG_DIR}")


if __name__ == "__main__":
    main()
