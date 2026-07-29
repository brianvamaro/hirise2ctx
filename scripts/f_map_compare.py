"""PLAN_FBuild §5.1 — head-to-head scorecard: F-build vs the mosaic-path map vs the A1 fallback.

Mandated by Brian 2026-07-23: the build is only worth its cost if it beats the cheap A1 fallback, and
this table is the evidence the ship-vs-fallback call rests on (§0.1 guard 2 moved the whole go/no-go
here plus the Stage-C/D gates).

Two rulings shape it (Brian 2026-07-28):
  * **Footprint = the 9 CTX-equipped tiles.** There is NO A1 raster on disk at any extent, and A1
    normalises raw CTX DN before the frozen ViT, so an A1 row can only exist where a local Murray
    mosaic zip does — that is 9 of the 26 block tiles. Every row is scored on that same 9-tile
    footprint (the mosaic and F rows are also reported over all 26 for context).
  * **η² is windowed and floor-relative.** All the numbers previously on record mix three
    incomparable scales (pilot-crop raw P, regional detrended abundance, block-scale raw P); this
    re-scores every row on ONE grid and ONE quantity — raw P(rich), partition composite, ~75 km
    windows, each window against its own rotation null.

Run (laptop, CPU, minutes):
  conda run --no-capture-output -n geospatial python -u scripts/f_map_compare.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/pandas

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import fcompose as fc
from src import fgates as fg

FIG = REPO / "reports" / "figures"
MAP_MOSAIC = REPO / "reports" / "map_region"
MAP_A1 = REPO / "reports" / "map_a1"
MAP_F = REPO / "reports" / "map_fbuild"
THEMIS = REPO / "cache_v2" / "validation" / "themis_night_ir_region.tif"

# The 9 tiles with a locally cached Murray CTX mosaic zip — the only tiles an A1 row can cover
# without ~30 GB of downloads (src.striping.equipped_tiles(), verified 2026-07-28).
EQUIPPED_FALLBACK = ["E-12_N36", "E-8_N32", "E0_N40", "E4_N40", "E4_N44",
                     "E8_N40", "E8_N44", "E12_N44", "E16_N44"]

# Run-cost ledger. Sources are recorded per row because they are NOT all measured: the mosaic figure
# is the run_region_array.sbatch header estimate for the 19 expansion tiles (region_manifest.json is
# stale — it records only the last 4-tile array task), while the F-build Stage A/B numbers are
# probe-measured (V1, DECISIONS 2026-07-24).
COST_LEDGER = [
    {"row": "mosaic", "cpu_h": 0.0, "gpu_h_l40s": 16.0, "wall_h": 2.5, "tiles": 26,
     "frames": 0, "source": "run_region_array.sbatch header (13-19 GPU-h); PLANNING figure"},
    {"row": "A1", "cpu_h": 0.0, "gpu_h_l40s": 16.0, "wall_h": 2.5, "tiles": 26,
     "frames": 0, "source": "same as mosaic + a per-frame DN renorm; A1 has no post-hoc path "
                            "(the ~14 min on record is the TRAINING re-embed, not a map)"},
    {"row": "F-build", "cpu_h": 265.0, "gpu_h_l40s": 33.0, "wall_h": 17.0, "tiles": 26,
     "frames": 907, "source": "Stage A 200-330 CPU-h + Stage B 33 L40S-h, probe-measured "
                              "(V1 DECISIONS 2026-07-24); + ~10 min laptop for Stages C+D"},
]


def equipped_tiles() -> list[str]:
    try:
        from src.striping import equipped_tiles as et
        got = et()
        return got or EQUIPPED_FALLBACK
    except Exception:                                        # noqa: BLE001
        return EQUIPPED_FALLBACK


def row_layers(map_f: Path, map_mosaic: Path, map_a1: Path, tile: str) -> dict[str, dict]:
    """Which raster stands for each comparison row, for η² (partition) and for ρ (shipped map).

    The mosaic and A1 maps have exactly ONE value per pixel (the SeamMap partition IS the map), so
    their partition raster and their shipped raster are the same file. The F build has n_frames values
    per pixel, so its partition raster is a separate, deliberately single-owner product.
    """
    rows = {
        "mosaic": {"eta2": map_mosaic / f"{tile}_prob_raw.tif",
                   "rho": map_mosaic / f"{tile}_prob_raw.tif"},
        "A1": {"eta2": map_a1 / f"{tile}_prob_raw.tif", "rho": map_a1 / f"{tile}_prob_raw.tif"},
    }
    for v in ("h1only", "full", "resid"):
        rows[f"F_{v}"] = {"eta2": map_f / f"{tile}_{v}_prob_partition.tif",
                          "rho": map_f / f"{tile}_{v}_prob_raw.tif"}
    return rows


def quality_table(tiles, map_f, map_mosaic, map_a1, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-tile η² (windowed + tile-scale, floor-relative) and THEMIS ρ, all on ONE common mask."""
    from src.striping import load_frames

    fl = FIG / "region_frame_list.csv"
    lut_path = map_f / "frame_lut.csv"
    lut = (list(pd.read_csv(lut_path).sort_values("frame_idx").PRODUCT_ID) if lut_path.exists()
           else (sorted(pd.read_csv(fl).PRODUCT_ID) if fl.exists() else []))
    themis_path = Path(args.themis)
    win_rows, tile_rows = [], []
    for tile in tiles:
        ref = map_mosaic / f"{tile}_prob_raw.tif"
        if not ref.exists():
            print(f"  ⚠ {tile}: no mosaic reference -> skipped", flush=True)
            continue
        grid = fc.tile_grid_from_raster(ref, tile)
        labels = fc.frame_labels_on_grid(grid, load_frames(tile), lut)
        layers = row_layers(map_f, map_mosaic, map_a1, tile)
        present = {k: v for k, v in layers.items() if v["eta2"].exists()}
        if not present:
            continue
        eta_arrays = {k: fg.read_layer(v["eta2"]) for k, v in present.items()}
        rho_arrays = {k: fg.read_layer(v["rho"]) for k, v in present.items() if v["rho"].exists()}
        themis = fg.themis_on_grid(grid, themis_path) if themis_path.exists() else None
        # ONE footprint for every row (a coverage difference must never read as a metric difference)
        mask = fg.common_finite(*eta_arrays.values())
        if themis is not None:
            mask &= np.isfinite(themis)
        t0 = time.monotonic()
        for name, arr in eta_arrays.items():
            masked = np.where(mask, arr, np.nan)
            ws = fg.window_eta2(masked, labels, tile, win_px=args.window_px,
                                n_draws=args.null_draws, seed=args.seed)
            for s in ws:
                win_rows.append({"row": name, "tile": tile, "r0": s.r0, "c0": s.c0,
                                 "eta2": s.eta2, "null_mean": s.null_mean, "null_p95": s.null_p95,
                                 "excess": s.excess, "ratio": s.ratio, "n_frames": s.n_frames})
            e, nm, n95, nc, nf = fg.eta2_with_null(masked, labels, n_draws=args.null_draws,
                                                   seed=args.seed)
            rho, nrho = ((np.nan, 0) if themis is None or name not in rho_arrays
                         else fg.spearman_rho(np.where(mask, rho_arrays[name], np.nan), themis))
            tile_rows.append({"row": name, "tile": tile, "n_cells": nc, "n_frames": nf,
                              "eta2": e, "null_mean": nm, "null_p95": n95,
                              "excess": e - nm if np.isfinite(e) else np.nan,
                              "ratio": e / n95 if np.isfinite(n95) and n95 > 0 else np.nan,
                              "themis_rho": rho, "n_rho": nrho,
                              "common_mask_frac": float(mask.mean())})
        print(f"  {tile}: {len(present)} rows on a common mask covering {mask.mean():.1%} "
              f"({time.monotonic() - t0:.0f}s)", flush=True)
    return pd.DataFrame(win_rows), pd.DataFrame(tile_rows)


def skill_column() -> pd.DataFrame:
    """Pooled pr_auc@1e-2 / prec@5% per row, from the LOIO prediction tables already on disk.

    These are OBS-level LOIO numbers, not map-footprint numbers — the honest out-of-sample skill of
    each input path's head. The map-footprint deltas live in gate 5 (`fbuild_gate5_skill.csv`), which
    is a different (in-sample-head, delta-scored) instrument; both are reported, never mixed.
    """
    rows = []
    legb = FIG / "f_h4_legb_summary.csv"
    if legb.exists():
        m = {"baseline (mosaic)": "mosaic", "H1 (F, unleveled)": "F_h1only",
             "H1+H4 (F, leveled)": "F_full"}
        for r in pd.read_csv(legb).itertuples():
            if r.pipeline in m:
                rows.append({"row": m[r.pipeline], "pooled_pr_auc": r.pooled_pr_auc,
                             "precision@5%": getattr(r, "_3", np.nan),
                             "n_img": r.n_img, "source": "f_h4_legb_summary.csv (36-img LOIO)"})
    a1 = FIG / "striping_a1_loio_summary_36.csv"
    a1_legacy = FIG / "striping_a1_loio_summary.csv"
    if a1.exists():
        for r in pd.read_csv(a1).itertuples():
            if "a1" in str(r.store):
                rows.append({"row": "A1", "pooled_pr_auc": r.pooled_pr_auc,
                             "precision@5%": np.nan, "n_img": r.n_img,
                             "source": "striping_a1_loio_summary_36.csv (re-run on the 36)"})
    elif a1_legacy.exists():
        for r in pd.read_csv(a1_legacy).itertuples():
            if "a1" in str(r.store):
                rows.append({"row": "A1", "pooled_pr_auc": r.pooled_pr_auc,
                             "precision@5%": np.nan, "n_img": r.n_img,
                             "source": "striping_a1_loio_summary.csv — ⚠ 38-IMG FOLDS, not "
                                       "comparable; re-run restricted to the 36"})
    return pd.DataFrame(rows)


def edge_cv_column() -> pd.DataFrame:
    p = FIG / "fbuild_gate2_edgecv.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)[["row", "unleveled_dp", "insample_dp", "heldout_cv_dp"]]


def figure(tile_df: pd.DataFrame, win_df: pd.DataFrame, out: Path) -> None:
    if tile_df.empty:
        return
    order = [r for r in ("mosaic", "A1", "F_h1only", "F_full", "F_resid")
             if r in set(tile_df.row)]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    med = win_df.groupby("row")[["eta2", "null_p95"]].median().reindex(order)
    x = np.arange(len(order))
    ax[0].bar(x - 0.18, med.eta2, 0.36, label="partition η² (median window)")
    ax[0].bar(x + 0.18, med.null_p95, 0.36, label="rotation-null p95", color="0.7")
    ax[0].axhline(fg.ETA2_BAR, color="tab:red", ls="--", lw=1, label=f"bar {fg.ETA2_BAR}")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    ax[0].set_title("gate 1 — artifact η² vs its own geological floor", fontsize=9)
    ax[0].legend(fontsize=7)
    rho = tile_df.groupby("row")["themis_rho"].median().reindex(order)
    ax[1].bar(x, rho, 0.6, color="tab:green")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    ax[1].set_title("THEMIS night-IR ρ (median tile)", fontsize=9)
    for r in order:
        g = win_df[win_df.row == r]
        if len(g):
            ax[2].scatter(g.null_p95, g.eta2, s=14, label=r, alpha=0.75)
    lim = float(np.nanpercentile(win_df[["eta2", "null_p95"]].to_numpy(), 99))
    ax[2].plot([0, lim], [0, lim], color="0.6", lw=0.8)
    ax[2].set_xlabel("rotation-null p95 (geological floor)")
    ax[2].set_ylabel("partition η²")
    ax[2].set_title("per window: above the diagonal = real frame structure", fontsize=9)
    ax[2].legend(fontsize=7)
    fig.suptitle("PLAN_FBuild §5.1 — F-build vs mosaic vs A1 on one common footprint")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-f", default=str(MAP_F))
    ap.add_argument("--map-mosaic", default=str(MAP_MOSAIC))
    ap.add_argument("--map-a1", default=str(MAP_A1))
    ap.add_argument("--themis", default=str(THEMIS))
    ap.add_argument("--tiles", nargs="*", default=None,
                    help="default = the 9 CTX-equipped tiles (Brian 2026-07-28)")
    ap.add_argument("--window-px", type=int, default=fg.WINDOW_PX)
    ap.add_argument("--null-draws", type=int, default=fg.NULL_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tiles = args.tiles or equipped_tiles()
    print(f"common footprint: {len(tiles)} tiles {tiles}", flush=True)
    map_a1 = Path(args.map_a1)
    if not any((map_a1 / f"{t}_prob_raw.tif").exists() for t in tiles):
        print(f"⚠ NO A1 map in {map_a1} — the A1 quality row will be BLANK. A1 renormalises raw CTX "
              f"DN before the frozen ViT, so there is no post-hoc path from the existing rasters: "
              f"run scripts/striping_a1_map.py over these tiles first (needs the cached CTX zips).",
              flush=True)

    print("\n=== quality table ===", flush=True)
    win_df, tile_df = quality_table(tiles, Path(args.map_f), Path(args.map_mosaic), map_a1, args)
    if tile_df.empty:
        raise SystemExit("no rows could be scored — is reports/map_region/ on disk?")
    FIG.mkdir(parents=True, exist_ok=True)
    win_df.to_csv(FIG / "fbuild_compare_windows.csv", index=False)
    tile_df.to_csv(FIG / "fbuild_compare_tiles.csv", index=False)

    summ = []
    for row, g in win_df.groupby("row"):
        t = tile_df[tile_df.row == row]
        summ.append({"row": row, "n_tiles": int(t.tile.nunique()), "n_windows": int(len(g)),
                     "eta2_window_median": float(g.eta2.median()),
                     "null_p95_window_median": float(g.null_p95.median()),
                     "excess_window_median": float(g.excess.median()),
                     "ratio_window_median": float(g.ratio.median()),
                     "frac_windows_below_bar": float((g.eta2 <= fg.ETA2_BAR).mean()),
                     "passes_eta2_bar": bool(g.eta2.median() <= fg.ETA2_BAR),
                     "eta2_tile_median": float(t.eta2.median()),
                     "excess_tile_median": float(t.excess.median()),
                     "themis_rho_median": float(t.themis_rho.median())})
    sdf = pd.DataFrame(summ).set_index("row")
    skill = skill_column()
    if not skill.empty:
        sdf = sdf.join(skill.set_index("row")[["pooled_pr_auc", "precision@5%"]], how="left")
    ecv = edge_cv_column()
    if not ecv.empty:
        sdf = sdf.join(ecv.set_index("row")[["heldout_cv_dp"]], how="left")
    order = [r for r in ("mosaic", "A1", "F_h1only", "F_full", "F_resid") if r in sdf.index]
    sdf = sdf.loc[order]
    sdf.to_csv(FIG / "fbuild_vs_mosaic_vs_a1.csv")

    print("\n=== §5.1 SCORECARD (one grid, one quantity: raw P(rich), partition composite) ===")
    show = ["n_tiles", "n_windows", "eta2_window_median", "null_p95_window_median",
            "excess_window_median", "passes_eta2_bar", "themis_rho_median"]
    show += [c for c in ("pooled_pr_auc", "precision@5%", "heldout_cv_dp") if c in sdf.columns]
    print(sdf[show].to_string())

    cost = pd.DataFrame(COST_LEDGER)
    cost.to_csv(FIG / "fbuild_cost_ledger.csv", index=False)
    print("\n=== run-cost ledger ===")
    print(cost[["row", "cpu_h", "gpu_h_l40s", "wall_h", "tiles", "frames"]].to_string(index=False))
    print("  (sources per row in fbuild_cost_ledger.csv — the mosaic/A1 GPU figures are PLANNING "
          "estimates, the F-build ones are probe-measured)")

    figure(tile_df, win_df, FIG / "fbuild_vs_mosaic_vs_a1.png")
    (FIG / "fbuild_compare_summary.json").write_text(
        json.dumps({"footprint_tiles": tiles, "scorecard": sdf.to_dict(orient="index"),
                    "cost_ledger": COST_LEDGER}, indent=2, default=str), encoding="utf-8")
    print("\nDecision framing (§5.1): the F build must materially beat A1 on artifact η² AND not "
          "lose on THEMIS-ρ / pooled skill to justify ~265 CPU-h + 33 GPU-h against A1's ~16 GPU-h.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
