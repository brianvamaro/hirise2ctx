"""PLAN_StripingArtifact (rev. 2026-06-18d) — the rectangular blocks are CTX SOURCE-FRAME
radiometric offsets. Quantify it.

The Murray Lab SeamMap is a *partition* (one source CTX frame per pixel; the polygons are
fragments of ~dozens of source images, recovered by dissolving on PRODUCT_ID). The regional
abundance map shows high-amplitude rectangular blocks that align with these source frames:
whole frames read systematically high/low abundance because the per-patch model keys on each
frame's radiometry (the Fang embedder applies a fixed /255 scaling, no per-frame normalization).

Three tests on the tiles with a cached CTX tile + SeamMap (``src.striping.equipped_tiles``):
  1. **eta^2** — variance of *detrended* abundance explained by source frame, vs a rotation null.
  2. **per-frame CTX brightness -> abundance** (pooled Spearman; weak == effect is texture not DN).
  3. **near-boundary step test** (geology continuous across a seam) — does the abundance step
     track the CTX-brightness step?

All analysis logic lives in ``src/striping.py``. Outputs:
reports/figures/26_frameblocks_{region,choropleth,scatter}.png and
striping_frameblocks_{perframe,summary}.csv.  Run: conda run -n geospatial python scripts/striping_frame_blocks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.striping import (MAP_DIR, all_map_tiles, boundary_steps, detrend, equipped_tiles,
                          eta2, eta2_rotation_null, frame_label_map, load_frames, load_raster,
                          per_frame_stats, read_ctx_on_grid)

FIG = Path(__file__).resolve().parents[1] / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def main():
    tiles = equipped_tiles()
    print(f"quantifying {len(tiles)} CTX+SeamMap tiles: {tiles}")
    perframe, summ, bnd_all = [], [], []
    for t in tiles:
        ab = load_raster(MAP_DIR / f"{t}_abundance.tif")
        resid, finite = detrend(ab)
        ctx = read_ctx_on_grid(t, MAP_DIR / f"{t}_abundance.tif")
        frames = load_frames(t)
        L = frame_label_map(t, frames)
        fin = finite & np.isfinite(ctx) & (L >= 0)

        e2 = eta2(resid, L, fin)
        e2n, e2n95 = eta2_rotation_null(resid, L, fin)
        pf = per_frame_stats(t, frames, resid, ctx, ab, fin, L)
        perframe.append(pf)
        bnd = boundary_steps(np.where(fin, resid, np.nan), np.where(fin, ctx, np.nan),
                             np.where(fin, L, -1))
        bnd["tile"] = t
        bnd_all.append(bnd)
        rho_b = spearmanr(bnd["dResid"], bnd["dCtx"]).statistic if len(bnd) > 5 else np.nan
        summ.append(dict(tile=t, n_frames=len(pf), eta2=e2, eta2_null=e2n, eta2_null95=e2n95,
                         n_bnd_pairs=len(bnd), boundary_rho=rho_b))
        print(f"  {t:9s} eta2={e2:.3f} (null {e2n:.3f}/95p {e2n95:.3f}) | {len(pf)} frames | "
              f"{len(bnd)} adj-pairs boundary rho(dAbund,dCTX)={rho_b:+.3f}", flush=True)

    pf = pd.concat(perframe, ignore_index=True)
    pf.to_csv(FIG / "striping_frameblocks_perframe.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(FIG / "striping_frameblocks_summary.csv", index=False)
    bnd_all = pd.concat(bnd_all, ignore_index=True)

    rho_frame = spearmanr(pf["mean_ctx"], pf["mean_resid"]).statistic
    rho_bnd = spearmanr(bnd_all["dResid"], bnd_all["dCtx"]).statistic
    print("\n=== VERDICT ===")
    print(f"frame eta^2 (variance explained): median {s['eta2'].median():.3f} vs null "
          f"{s['eta2_null'].median():.3f}  ({(s['eta2'] > s['eta2_null95']).mean()*100:.0f}% tiles > null95)")
    print(f"per-frame CTX brightness -> abundance: pooled Spearman = {rho_frame:+.3f} (n={len(pf)})")
    print(f"near-boundary step (geology-controlled): Spearman(dAbund,dCTX) = {rho_bnd:+.3f} "
          f"(n={len(bnd_all)})")

    # --- figure 1: region abundance + frame outlines ---
    from rasterio.merge import merge
    srcs = [rasterio.open(MAP_DIR / f"{t}_abundance.tif") for t in all_map_tiles()]
    arr, tr = merge(srcs); crs = srcs[0].crs; nd = srcs[0].nodata
    b = rasterio.transform.array_bounds(arr.shape[1], arr.shape[2], tr)
    for s_ in srcs:
        s_.close()
    A = arr[0].astype(float); A[A == nd] = np.nan
    ext = [b[0], b[2], b[1], b[3]]
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))
    for a in ax:
        a.imshow(A, cmap="magma", vmax=np.nanpercentile(A, 99), extent=ext, origin="upper")
    ax[0].set_title("Regional abundance (26 tiles)")
    n = 0
    for t in all_map_tiles():
        try:
            load_frames(t).boundary.plot(ax=ax[1], edgecolor="cyan", linewidth=0.25); n += 1
        except Exception as e:
            print(f"   frames {t}: {e}", flush=True)
    ax[1].set_xlim(ext[0], ext[1]); ax[1].set_ylim(ext[2], ext[3])
    ax[1].set_title(f"+ CTX source-frame outlines ({n} tiles)")
    fig.tight_layout(); fig.savefig(FIG / "26_frameblocks_region.png", dpi=120); plt.close(fig)

    # --- figure 2: E8_N36 lead choropleth (raw+frames | frame-mean detrended abundance) ---
    # E8_N36 (Brian's example) has no cached CTX zip; the CTX<->abundance link is in figure 3.
    LEAD = "E8_N36"
    abL = load_raster(MAP_DIR / f"{LEAD}_abundance.tif")
    residL, finL0 = detrend(abL)
    fr = load_frames(LEAD)
    LL = frame_label_map(LEAD, fr)
    finL = finL0 & (LL >= 0)
    chor_r = np.full(abL.shape, np.nan)   # frame-mean detrended abundance (artifact, geology removed)
    for i in range(len(fr)):
        sel = finL & (LL == i)
        if sel.sum() >= 50:
            chor_r[LL == i] = residL[sel].mean()
    with rasterio.open(MAP_DIR / f"{LEAD}_abundance.tif") as ds:
        bb = ds.bounds
    e = [bb.left, bb.right, bb.bottom, bb.top]
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    i0 = ax[0].imshow(abL, cmap="magma", vmax=np.nanpercentile(abL, 99), extent=e, origin="upper")
    fr.boundary.plot(ax=ax[0], edgecolor="cyan", linewidth=0.4)
    ax[0].set_title(f"{LEAD} raw abundance + frame outlines"); plt.colorbar(i0, ax=ax[0], fraction=0.046)
    vlo, vhi = np.nanpercentile(chor_r, [2, 98])
    i1 = ax[1].imshow(chor_r, cmap="RdBu_r", vmin=vlo, vmax=vhi, extent=e, origin="upper")
    ax[1].set_title("frame-mean DETRENDED abundance\n(geology removed -> frame-coherent residual = artifact)")
    plt.colorbar(i1, ax=ax[1], fraction=0.046)
    fig.tight_layout(); fig.savefig(FIG / "26_frameblocks_choropleth.png", dpi=120); plt.close(fig)

    # --- figure 3: the two quantitative scatters ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for t, sub in pf.groupby("tile"):
        ax[0].scatter(sub["mean_ctx"], sub["mean_resid"], s=12, label=t)
    ax[0].set_xlabel("per-frame mean CTX brightness (DN)")
    ax[0].set_ylabel("per-frame mean detrended abundance")
    ax[0].set_title(f"Brighter frames vs abundance (weak == texture-driven, not DN)\n"
                    f"pooled Spearman = {rho_frame:+.2f}")
    ax[0].legend(fontsize=6, ncol=2)
    ax[1].scatter(bnd_all["dCtx"], bnd_all["dResid"], s=8, alpha=0.4)
    ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(0, color="k", lw=0.5)
    ax[1].set_xlabel("CTX-brightness step across seam (DN)")
    ax[1].set_ylabel("abundance step across seam")
    ax[1].set_title(f"Geology-controlled seam step (rho = {rho_bnd:+.2f}, n={len(bnd_all)})")
    fig.tight_layout(); fig.savefig(FIG / "26_frameblocks_scatter.png", dpi=120); plt.close(fig)
    print(f"Figures + CSVs -> {FIG}")


if __name__ == "__main__":
    main()
