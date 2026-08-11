"""Regional gap-fill map for docs/model_evidence.md §0 (replaces the confounded
'validated vs deployed' two-panel — Brian 2026-06-14b).

ONE continuous scene over a region that contains a training HiRISE footprint: the
deployable head predicts P(boulder-rich) over a large window centred on the
footprint, and the HiRISE truth (true boulder-rich tiles) is outlined inside the
footprint as the validation anchor. Same terrain throughout, so there is no
rich-vs-poor confound -- the figure shows the product the project delivers: train
on the scattered HiRISE footprints, predict the CTX between them (regional
gap-fill, Serrano et al. 2010 framing).

ONE GPU inference run (~1 min). No download (tile zip cached).

Output: reports/figures/model_evidence_gapfill_map.png  (+ GeoTIFF/JSON sidecar)
"""
from __future__ import annotations

import json
import sys
import time
from copy import copy
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd

from src.mapping import predict_window, read_tile_window, write_geotiff

OBS = "ESP_045139_2270"      # dense rich plains, 46.4N 13.4E, tile E12_N44 (cached)
WIN = 5000                   # window side in CTX px (5 m/px -> 25 km)
TILE_PX = 32
FIG = REPO / "reports" / "figures"
OUT = REPO / "reports" / "map_pilot"


def footprint_pixel_box(actual_bounds, inner_transform):
    xmin, ymin, xmax, ymax = actual_bounds
    a, _, c, _, e, f = (inner_transform[i] for i in range(6))
    px_x, px_y = abs(a), abs(e)
    col_min = int(round((xmin - c) / px_x)); col_max = int(round((xmax - c) / px_x))
    row_min = int(round((f - ymax) / px_y)); row_max = int(round((f - ymin) / px_y))
    return row_min, row_max, col_min, col_max


def main():
    side = json.loads((REPO / "cache_v2" / "ctx_windows" / f"{OBS}.json").read_text())
    murray = side["source_murray_tile"]
    bounds = side["actual_bounds_target_crs"]
    tinfo = json.loads((REPO / "cache_v2" / "ctx_tiles" / f"{murray}.json").read_text())
    inner_tif, itf = tinfo["inner_tif"], tinfo["inner_transform"]
    th, tw = tinfo["inner_shape"]
    zip_path = REPO / "cache_v2" / "ctx_tiles" / f"{murray}.zip"

    r0, r1, c0, c1 = footprint_pixel_box(bounds, itf)
    rc, cc = (r0 + r1) // 2, (c0 + c1) // 2
    row_off = max(0, min(rc - WIN // 2, th - WIN))
    col_off = max(0, min(cc - WIN // 2, tw - WIN))
    print(f"tile={murray} {th}x{tw}  footprint rows[{r0}:{r1}] cols[{c0}:{c1}]  "
          f"window off=({row_off},{col_off}) win={WIN}", flush=True)

    window = read_tile_window(zip_path, inner_tif, row_off, col_off, WIN)
    zero_frac = float((window.data == 0).mean())
    print(f"window zero_frac={zero_frac:.3f}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tile_stem = f"gapfill_{murray}_{OBS}"      # inference identity: GeoTIFF + sidecar
    tif_path = OUT / f"{tile_stem}.tif"

    import rasterio
    if "--force" not in sys.argv and tif_path.exists():
        # render-only: reuse the banked prediction raster (no GPU)
        with rasterio.open(tif_path) as ds:
            raster = ds.read(1); transform = tuple(ds.transform)[:6]; crs_wkt = ds.crs.to_wkt()
        print(f"render-only: reusing banked {tif_path.name}", flush=True)
    else:
        from src.fm_embeddings import FangEmbedder
        from src.modeling.mlp_head import DeployableHead
        model_dir = sorted(p for p in (REPO / "models" / "deployable").glob("*")
                           if (p / "recipe.json").exists())[-1]
        t0 = time.monotonic()
        embedder = FangEmbedder.load()
        head = DeployableHead.load(model_dir)
        pred = predict_window(window, embedder, head, tile_px=TILE_PX, max_zero_fraction=0.3,
                              max_context_zero_fraction=0.0)   # R13: production thresholds
        finite = np.isfinite(pred.prob)
        pp = pred.prob[finite]
        print(f"embed+predict {time.monotonic()-t0:.0f}s  tiles={pred.ti.size}  "
              f"predicted={int(finite.sum())}  mean P={pp.mean():.3f}  "
              f">=0.5 share={(pp>=0.5).mean():.3f}", flush=True)
        write_geotiff(tif_path, pred.raster, pred.transform, pred.crs_wkt)
        raster, transform, crs_wkt = pred.raster, pred.transform, pred.crs_wkt

    # --- geo extents ---
    a, _, c, _, e, f = (transform[i] for i in range(6))
    H, W = raster.shape
    pred_extent = (c, c + W * a, f + H * e, f)   # (left,right,bottom,top), e<0

    # CTX backdrop extent (the read window), for the plain-CTX panel
    wt = window.transform
    wa, wc, we, wf = wt[0], wt[2], wt[4], wt[5]
    WH, WW = window.data.shape
    ctx_extent = (wc, wc + WW * wa, wf + WH * we, wf)

    # HiRISE truth from labels: per-tile boulder area-fraction (rock abundance) +
    # the boulder-rich tiles + the footprint coverage
    lab = pd.read_parquet(REPO / "dataset_v2" / "labels" / f"{OBS}.parquet")
    lab = lab[lab["scale_idx"] == 2]
    from src.mapping import tiles_to_raster
    ti_l, tj_l = lab["ti"].to_numpy(), lab["tj"].to_numpy()
    ab = lab["fractional_area"].to_numpy()
    rich = (ab > 1e-2).astype(float)
    traster, _, _ = tiles_to_raster(ti_l, tj_l, rich)
    abr, _, _ = tiles_to_raster(ti_l, tj_l, ab)
    t_extent = (lab["xmin"].min(), lab["xmax"].max(), lab["ymin"].min(), lab["ymax"].max())

    # footprint coverage mask (same grid as abr) -- for hole-filling the truth panel
    foot, _, _ = tiles_to_raster(ti_l, tj_l, np.ones_like(ab))
    from scipy.ndimage import binary_fill_holes, binary_closing, distance_transform_edt
    footbin = binary_fill_holes(binary_closing(~np.isnan(foot), iterations=2))

    # footprint outline = convex hull of the labelled tiles, lightly simplified.
    # (HiRISE strips are sheared parallelograms; a minimum-rotated *rectangle* forces
    # 90deg corners and overshoots the acute ones -- the hull follows the true shape.)
    from shapely.geometry import MultiPoint
    cxs = np.concatenate([lab["xmin"], lab["xmax"], lab["xmax"], lab["xmin"]])
    cys = np.concatenate([lab["ymin"], lab["ymin"], lab["ymax"], lab["ymax"]])
    tile_m = float(lab["tile_size_m"].iloc[0])
    quad = MultiPoint(np.column_stack([cxs, cys])).convex_hull.simplify(tile_m * 2)
    qx, qy = quad.exterior.xy

    def draw_footprint(ax):
        ax.plot(qx, qy, color="cyan", lw=1.8)

    # optional: apply the Tier-1 calibrator (PLAN_Calibration L3) to the off-HiRISE
    # P(rich) -- a DRAFT preview. Fit on the 38 labelled images (deployment-honest:
    # the calibrator learns "what P=x means" from labelled data, applied where there
    # is none), write a separate _calibrated file, leave the original untouched.
    fig_stem, suffix = "model_evidence_gapfill_map", ""   # figure output name (per variant)
    if "--calibrate" in sys.argv:
        from src.calibration import IsotonicCalibrator
        clf = pd.read_parquet(REPO / "models/fang_probe"
                              / "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet")
        calr = IsotonicCalibrator().fit(clf.y_pred.to_numpy(), clf.y_true.to_numpy())
        fin = np.isfinite(raster)
        raster = raster.copy()
        raster[fin] = calr.predict(raster[fin])
        fig_stem += "_calibrated"
        suffix += "  [DRAFT: P(rich) isotonic-calibrated on the 38 labelled images]"
        print(f"calibrated: mean P {float(np.nanmean(raster[fin])):.3f}  "
              f">=0.5 share {float(np.mean(raster[fin] >= 0.5)):.3f}", flush=True)
    use_inferno = "--inferno" in sys.argv   # right (model) panel: inferno vs turbo
    if use_inferno:
        fig_stem += "_inferno"

    # --- render: plain CTX | HiRISE ground truth | model prediction ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    inferno = copy(plt.cm.inferno); inferno.set_bad(alpha=0.0)
    turbo = copy(plt.cm.turbo); turbo.set_bad(alpha=0.0)
    ctx_disp = window.data[::5, ::5].astype(np.float32)
    v = ctx_disp[(ctx_disp > 0) & np.isfinite(ctx_disp)]
    lo, hi = np.percentile(v, (2, 98))
    ctx_norm = np.clip((ctx_disp - lo) / max(hi - lo, 1e-9), 0, 1)

    def ctx_bg(ax):
        ax.imshow(ctx_norm, cmap="gray", extent=ctx_extent, origin="upper",
                  interpolation="nearest")

    fig, axes = plt.subplots(1, 3, figsize=(21, 8.4))

    # (1) plain CTX
    ctx_bg(axes[0]); draw_footprint(axes[0])
    axes[0].set_title("Plain CTX (5 m/px) — the model's only input", fontsize=11)

    # (2) HiRISE ground truth: rock abundance over the footprint
    vmax = float(np.percentile(abr[np.isfinite(abr) & (abr > 0)], 99))
    floor = vmax * 1e-2
    abr_disp = abr.copy()
    fin = np.isfinite(abr_disp)
    abr_disp[fin & (abr_disp < floor)] = floor   # zeros/near-zero -> darkest abundance
    # fill interior missing-label holes (dropped tiles) with nearest valid abundance
    # for a clean continuous field, then keep everything OUTSIDE the footprint blank
    holes = np.isnan(abr_disp)
    if holes.any():
        idx = distance_transform_edt(holes, return_distances=False, return_indices=True)
        abr_disp = abr_disp[tuple(idx)]
    abr_disp = np.where(footbin, abr_disp, np.nan)
    ctx_bg(axes[1])
    imt = axes[1].imshow(np.ma.masked_invalid(abr_disp), cmap=inferno,
                         norm=LogNorm(vmin=floor, vmax=vmax, clip=True),
                         extent=t_extent, origin="upper", interpolation="nearest")
    draw_footprint(axes[1])
    axes[1].set_title("HiRISE ground truth — rock abundance (area fraction)", fontsize=11)

    # (3) model prediction: P(rich) over the whole scene. Turbo (blue->red) gives
    # honest variation across the 0-1 range; inferno made the high-but-uniform P
    # look over-saturated and overstated the compression.
    im = axes[2].imshow(np.ma.masked_invalid(raster), cmap=(inferno if use_inferno else turbo),
                        vmin=0, vmax=1, extent=pred_extent, origin="upper", interpolation="nearest")
    draw_footprint(axes[2])
    axes[2].set_title("Model P(boulder-rich) @ 160 m", fontsize=11)
    axes[2].text(0.015, 0.985, "cyan = HiRISE footprint (truth available, in training)",
                 transform=axes[2].transAxes, fontsize=8.5, va="top", ha="left", color="white",
                 bbox=dict(boxstyle="round", fc="black", ec="none", alpha=0.55))

    # equal-size panels: append a same-width colourbar slot to every axis (panel 0's
    # is hidden) so the colourbars on 1-2 don't shrink those panels below panel 0.
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    for ax in axes:
        ax.set_xlim(pred_extent[0], pred_extent[1]); ax.set_ylim(pred_extent[2], pred_extent[3])
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for ax, mappable, label in [(axes[0], None, None),
                                (axes[1], imt, "boulder area-fraction (log)"),
                                (axes[2], im, "model P(boulder-rich)")]:
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
        if mappable is None:
            cax.axis("off")
        else:
            fig.colorbar(mappable, cax=cax).set_label(label)

    km = (pred_extent[1] - pred_extent[0]) / 1000
    pcells = raster[np.isfinite(raster)]
    fig.suptitle(f"Regional gap-fill — one continuous CTX scene, {km:.0f} km across "
                 f"({murray}). HiRISE truth exists only inside the cyan footprint; the model "
                 "predicts boulder-rich\nprobability everywhere from CTX alone — reproducing the "
                 f"truth inside the footprint and filling the gap outside.{suffix}", fontsize=11.5)
    out = FIG / f"{fig_stem}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    (OUT / f"{tile_stem}.json").write_text(json.dumps({
        "obs_id": OBS, "murray_tile": murray, "win_px": WIN,
        "window_offset_rowcol": [window.row_off, window.col_off],
        "n_predicted": int(pcells.size), "mean_p": float(pcells.mean()),
        "rich_share_at_0p5": float((pcells >= 0.5).mean()),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {out}  (mean P={pcells.mean():.3f}, rich share={float((pcells>=0.5).mean()):.3f})",
          flush=True)


if __name__ == "__main__":
    main()
