"""PLAN_FBuild §5 — score the six pre-declared acceptance gates on the Stage-D map.

Gate 1 is scored the way Brian ruled on 2026-07-28 (the literal "partition η² ≤ 0.05 on the full
block" is not interpretable at 907-frame scale — see `src/fgates.py` for the measured evidence):
**headline = partition η² on ~75 km windows, each against its own rotation null, bar applied to the
median window**, with the per-tile/block numbers reported floor-relative alongside.

  1  partition η² (windowed, floor-relative)     vs the 0.05 bar + the mosaic row
  2  held-out-edge |Δp|                          recomputed for the offsets the map actually uses
  3  THEMIS night-IR ρ                           not degraded vs the mosaic map (Δρ ≥ −0.02)
  4  visual                                      frame-mean choropleth, mosaic vs H1-only vs H1+H4
  5  pooled skill Δ(H1+H4 − H1)                  ≥ −0.02 on the in-region cohort obs (head cancels)
  6  calibrated-abundance fidelity               top_ratio / marginal-L1 / per-bin RMSE, BOTH layers

Run (laptop, CPU, minutes; needs Stage D's output dir):
  conda run --no-capture-output -n geospatial python -u scripts/f_region_gates.py
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

from scripts.map_region import BLOCK_TILES
from src import fcompose as fc
from src import fgates as fg
from src import leveling as lv

FIG = REPO / "reports" / "figures"
MAP_F = REPO / "reports" / "map_fbuild"
MAP_MOSAIC = REPO / "reports" / "map_region"
THEMIS = REPO / "cache_v2" / "validation" / "themis_night_ir_region.tif"
EDGE_CACHE = REPO / "reports" / "f_stagec"
OFFSETS_CSV = FIG / "fbuild_stagec_offsets.csv"
GUARD_CSV = FIG / "fbuild_trend_guard.csv"
COHORT_BOUNDS = REPO / "reports" / "f_leg_b" / "cohort_obs_bounds.csv"
LABELS_DIR = REPO / "dataset_v2" / "labels"
VARIANTS = ("h1only", "full", "resid")


# --------------------------------------------------------------------------- gate 1
def gate1(tiles, map_f: Path, map_mosaic: Path, lut: list[str], args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Windowed partition η² + rotation nulls for every map row, on one grid and one quantity."""
    win_rows, tile_rows = [], []
    for tile in tiles:
        ref = map_mosaic / f"{tile}_prob_raw.tif"
        if not ref.exists():
            continue
        grid = fc.tile_grid_from_raster(ref, tile)
        try:
            from src.striping import load_frames
            labels = fc.frame_labels_on_grid(grid, load_frames(tile), lut)
        except Exception as exc:                       # noqa: BLE001
            print(f"  ⚠ {tile}: no SeamMap labels ({exc}) -> skipped", flush=True)
            continue
        rows = {"mosaic": fg.read_layer(ref)}
        for v in VARIANTS:
            p = map_f / f"{tile}_{v}_prob_partition.tif"
            if p.exists():
                rows[f"F_{v}"] = fg.read_layer(p)
        t0 = time.monotonic()
        for name, arr in rows.items():
            ws = fg.window_eta2(arr, labels, tile, win_px=args.window_px,
                                n_draws=args.null_draws, seed=args.seed)
            for s in ws:
                win_rows.append({"row": name, "tile": tile, "r0": s.r0, "c0": s.c0,
                                 "n_cells": s.n_cells, "n_frames": s.n_frames,
                                 "eta2": round(s.eta2, 5), "null_mean": round(s.null_mean, 5),
                                 "null_p95": round(s.null_p95, 5),
                                 "excess": round(s.excess, 5), "ratio": round(s.ratio, 4)})
            e, nm, n95, nc, nf = fg.eta2_with_null(arr, labels, n_draws=args.null_draws,
                                                   seed=args.seed)
            tile_rows.append({"row": name, "tile": tile, "scope": "tile", "n_cells": nc,
                              "n_frames": nf, "eta2": e, "null_mean": nm, "null_p95": n95,
                              "excess": e - nm if np.isfinite(e) else np.nan,
                              "ratio": e / n95 if np.isfinite(n95) and n95 > 0 else np.nan})
        print(f"  {tile}: {len(rows)} rows scored ({time.monotonic() - t0:.0f}s)", flush=True)
    return pd.DataFrame(win_rows), pd.DataFrame(tile_rows)


# --------------------------------------------------------------------------- gate 2
def gate2(offsets: pd.DataFrame, guard: dict, args) -> pd.DataFrame:
    cache = sorted(EDGE_CACHE.glob("stagec_edges_min*.npz"))
    if not cache:
        print("  ⚠ no Stage-C edge cache -> gate 2 skipped", flush=True)
        return pd.DataFrame()
    es_all = lv.EdgeSet.load(cache[-1])
    es = es_all.filter(es_all.w >= args.min_tiles)
    n = len(es.pids)
    lam = float(guard.get("lambda_star") or 0.0)
    rows = []
    for v in VARIANTS:
        col = {"h1only": None, "full": "offset_logit", "resid": "offset_residual_only"}[v]
        o = np.zeros(n) if col is None else np.array(
            [float(offsets.loc[p, col]) if p in offsets.index else 0.0 for p in es.pids])
        r = fg.edge_cv_for_offsets(es, o, n, lam, frac=args.cv_frac, repeats=args.cv_repeats,
                                  seed=args.seed)
        rows.append({"row": f"F_{v}", "n_edges": es.n_edges, "lambda": lam, **r})
        print(f"  F_{v}: unleveled |Δp| {r['unleveled_dp']:.4f} -> in-sample {r['insample_dp']:.4f}, "
              f"held-out {r['heldout_cv_dp']:.4f}", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- gate 3
def gate3(tiles, map_f: Path, map_mosaic: Path, args) -> pd.DataFrame:
    if not Path(args.themis).exists():
        print(f"  ⚠ {args.themis} missing -> gate 3 skipped", flush=True)
        return pd.DataFrame()
    rows = []
    for tile in tiles:
        ref = map_mosaic / f"{tile}_prob_raw.tif"
        if not ref.exists():
            continue
        grid = fc.tile_grid_from_raster(ref, tile)
        th = fg.themis_on_grid(grid, Path(args.themis))
        cand = {"mosaic": ref}
        cand.update({f"F_{v}": map_f / f"{tile}_{v}_prob_raw.tif" for v in VARIANTS})
        arrays = {k: fg.read_layer(p) for k, p in cand.items() if p.exists()}
        if not arrays:
            continue
        mask = fg.common_finite(th, *arrays.values())       # one footprint for every row
        for name, arr in arrays.items():
            r, nn = fg.spearman_rho(arr[mask], th[mask])
            rows.append({"row": name, "tile": tile, "rho": r, "n": nn})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    piv = df.pivot_table(index="tile", columns="row", values="rho")
    for col in [c for c in piv.columns if c != "mosaic"]:
        piv[f"delta_{col}"] = piv[col] - piv["mosaic"]
    return piv.reset_index()


# --------------------------------------------------------------------------- gate 5 / 6 cohort join
def cohort_table(map_f: Path, map_mosaic: Path, tiles, args) -> pd.DataFrame:
    """Labelled cohort tiles joined to the F map through the global lattice (gates 5 + 6)."""
    if not COHORT_BOUNDS.exists():
        print(f"  ⚠ {COHORT_BOUNDS} missing -> gates 5/6 skipped", flush=True)
        return pd.DataFrame()
    bounds = pd.read_csv(COHORT_BOUNDS)
    labs = []
    for p in sorted(LABELS_DIR.glob("*.parquet")):
        d = pd.read_parquet(p, columns=["obs_id", "tile_size_px", "ti", "tj", "fractional_area"])
        labs.append(d[d.tile_size_px == 32])
    if not labs:
        return pd.DataFrame()
    lab = pd.concat(labs, ignore_index=True)
    joined = fg.cohort_tiles_to_global(bounds, lab)
    if joined.empty:
        return joined
    out = []
    for tile in tiles:
        ref = map_mosaic / f"{tile}_prob_raw.tif"
        if not ref.exists():
            continue
        grid = fc.tile_grid_from_raster(ref, tile)
        ti0, ti1 = grid.TI_range()
        tj0, tj1 = grid.TJ_range()
        sel = joined[(joined.TI >= ti0) & (joined.TI <= ti1) &
                     (joined.TJ >= tj0) & (joined.TJ <= tj1)].copy()
        if sel.empty:
            continue
        rows = grid.rows_of_TI(sel.TI.to_numpy())
        cols = grid.cols_of_TJ(sel.TJ.to_numpy())
        sel["murray_tile"] = tile
        for v in VARIANTS:
            for layer, key in (("prob_raw", f"p_{v}"), ("abundance", f"ab_{v}"),
                               ("abundance_moscal", f"abmos_{v}")):
                p = map_f / f"{tile}_{v}_{layer}.tif"
                sel[key] = fg.read_layer(p)[rows, cols] if p.exists() else np.nan
        out.append(sel)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def gate5(coh: pd.DataFrame) -> pd.DataFrame:
    """Δ(H1+H4 − H1) pooled skill; absolutes reported but NOT gated (the head is in-sample)."""
    if coh.empty:
        return pd.DataFrame()
    y = (coh.fractional_area.to_numpy() > 1e-2).astype(int)
    rows = []
    for v in VARIANTS:
        col = f"p_{v}"
        if col not in coh or not np.isfinite(coh[col]).any():
            continue
        m = np.isfinite(coh[col])
        rows.append({"row": f"F_{v}", "n_obs": int(coh.loc[m, "obs_id"].nunique()),
                     **fg.pooled_skill(y[m.to_numpy()], coh.loc[m, col].to_numpy())})
    df = pd.DataFrame(rows)
    if {"F_h1only", "F_full"} <= set(df.row):
        h1 = df.loc[df.row == "F_h1only"].iloc[0]
        for v in ("full", "resid"):
            if f"F_{v}" in set(df.row):
                r = df.loc[df.row == f"F_{v}"].iloc[0]
                df.loc[df.row == f"F_{v}", "delta_pr_auc_vs_h1"] = (r["pooled_pr_auc"]
                                                                    - h1["pooled_pr_auc"])
                df.loc[df.row == f"F_{v}", "delta_prec5_vs_h1"] = (r["precision@5%"]
                                                                   - h1["precision@5%"])
    if "delta_pr_auc_vs_h1" in df:
        df["passes"] = df["delta_pr_auc_vs_h1"] >= fg.SKILL_TOL
    return df


def gate6(coh: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if coh.empty:
        return pd.DataFrame(), pd.DataFrame()
    fa = coh.fractional_area.to_numpy(float)
    rows, bins = [], []
    for v in VARIANTS:
        for layer, tag in ((f"ab_{v}", "rebanked_f"), (f"abmos_{v}", "reused_mosaic")):
            if layer not in coh or not np.isfinite(coh[layer]).any():
                continue
            res = fg.abundance_fidelity(fa, coh[layer].to_numpy(float))
            per_bin = res.pop("per_bin", None)
            rows.append({"row": f"F_{v}", "calibrator": tag, **res})
            if per_bin is not None:
                bins.append(per_bin.assign(row=f"F_{v}", calibrator=tag))
    return pd.DataFrame(rows), (pd.concat(bins, ignore_index=True) if bins else pd.DataFrame())


# --------------------------------------------------------------------------- gate 4 (visual)
def gate4(tiles, map_f: Path, map_mosaic: Path, lut: list[str], out_png: Path) -> None:
    """Frame-mean choropleth: the original success criterion (blocks present -> gone)."""
    from src.striping import load_frames

    panels, titles = [], []
    tile = next((t for t in tiles if (map_f / f"{t}_full_prob_partition.tif").exists()), None)
    if tile is None:
        print("  ⚠ no F partition raster -> gate 4 figure skipped", flush=True)
        return
    grid = fc.tile_grid_from_raster(map_mosaic / f"{tile}_prob_raw.tif", tile)
    labels = fc.frame_labels_on_grid(grid, load_frames(tile), lut)
    for name, path in (("mosaic", map_mosaic / f"{tile}_prob_raw.tif"),
                       ("F H1-only", map_f / f"{tile}_h1only_prob_partition.tif"),
                       ("F H1+H4", map_f / f"{tile}_full_prob_partition.tif")):
        if not path.exists():
            continue
        arr = fg.read_layer(path)
        cho = np.full(arr.shape, np.nan)
        for fi in np.unique(labels[labels >= 0]):
            sel = (labels == fi) & np.isfinite(arr)
            if sel.sum() >= 30:
                cho[labels == fi] = arr[sel].mean()
        panels.append((arr, cho))
        titles.append(name)
    if not panels:
        return
    vmax = float(np.nanpercentile(panels[0][0], 99))
    fig, ax = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 8), squeeze=False)
    for k, ((arr, cho), t) in enumerate(zip(panels, titles)):
        for r, (img, lab) in enumerate(((arr, "P(rich)"), (cho, "frame-mean"))):
            im = ax[r, k].imshow(img, cmap="magma", vmin=0, vmax=vmax)
            ax[r, k].set_title(f"{t} — {lab}", fontsize=9)
            ax[r, k].set_xticks([])
            ax[r, k].set_yticks([])
            plt.colorbar(im, ax=ax[r, k], fraction=0.046)
    fig.suptitle(f"PLAN_FBuild gate 4 — frame blocks, {tile} (partition composite)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    print(f"  wrote {out_png}", flush=True)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map-f", default=str(MAP_F))
    ap.add_argument("--map-mosaic", default=str(MAP_MOSAIC))
    ap.add_argument("--offsets", default=str(OFFSETS_CSV))
    ap.add_argument("--guard", default=str(GUARD_CSV))
    ap.add_argument("--themis", default=str(THEMIS))
    ap.add_argument("--tiles", nargs="*", default=None)
    ap.add_argument("--window-px", type=int, default=fg.WINDOW_PX)
    ap.add_argument("--null-draws", type=int, default=fg.NULL_DRAWS)
    ap.add_argument("--min-tiles", type=int, default=200)
    ap.add_argument("--cv-frac", type=float, default=0.05)
    ap.add_argument("--cv-repeats", type=int, default=4)
    ap.add_argument("--gates", nargs="*", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    map_f, map_mosaic = Path(args.map_f), Path(args.map_mosaic)
    tiles = args.tiles or list(BLOCK_TILES)
    FIG.mkdir(parents=True, exist_ok=True)
    lut_path = map_f / "frame_lut.csv"
    if lut_path.exists():
        lut = list(pd.read_csv(lut_path).sort_values("frame_idx").PRODUCT_ID)
    else:
        # Fall back to the planned frame list so the MOSAIC row (and gate 1's whole path) can be
        # scored before Stage B/C land — the label vocabulary only has to be consistent within a run.
        fl = FIG / "region_frame_list.csv"
        lut = sorted(pd.read_csv(fl).PRODUCT_ID) if fl.exists() else []
        print(f"⚠ {lut_path} missing — using {fl.name} for the frame vocabulary "
              f"({len(lut)} frames); F rows will be absent until Stage D runs", flush=True)
    offsets = (pd.read_csv(args.offsets).set_index("PRODUCT_ID")
               if Path(args.offsets).exists() else pd.DataFrame())
    guard = (pd.read_csv(args.guard).iloc[0].to_dict() if Path(args.guard).exists() else {})
    verdict = {"verdict": guard.get("verdict", "UNKNOWN"), "apply": guard.get("apply", "?"),
               "needs_ruling": bool(guard.get("needs_ruling", True)),
               "lambda_star": guard.get("lambda_star")}
    scored = {"verdict": verdict}

    if 1 in args.gates:
        print("\n=== gate 1: partition η², windowed with per-window rotation nulls ===", flush=True)
        win, per_tile = gate1(tiles, map_f, map_mosaic, lut, args)
        if not win.empty:
            win.to_csv(FIG / "fbuild_gate1_windows.csv", index=False)
            per_tile.to_csv(FIG / "fbuild_gate1_tiles.csv", index=False)
            summ = []
            for row, g in win.groupby("row"):
                ws = [fg.WindowScore(t, r0, c0, nc, nf, e, nm, n95) for t, r0, c0, nc, nf, e, nm, n95
                      in zip(g.tile, g.r0, g.c0, g.n_cells, g.n_frames, g.eta2, g.null_mean, g.null_p95)]
                s = fg.summarize_windows(ws)
                pt = per_tile[per_tile.row == row]
                s.update({"row": row,
                          "tile_eta2_median": float(pt.eta2.median()) if len(pt) else np.nan,
                          "tile_excess_median": float(pt.excess.median()) if len(pt) else np.nan,
                          "tile_ratio_median": float(pt.ratio.median()) if len(pt) else np.nan})
                summ.append(s)
            sdf = pd.DataFrame(summ).set_index("row")
            sdf.to_csv(FIG / "fbuild_gate1_summary.csv")
            print(sdf[["n_windows", "eta2_median", "null_p95_median", "excess_median",
                       "ratio_median", "frac_windows_below_bar", "passes_bar",
                       "tile_eta2_median", "tile_excess_median"]].to_string())
            print(f"\n  bar = median-window partition η² <= {fg.ETA2_BAR} "
                  f"(window {args.window_px} px ~ {args.window_px * 160 / 1000:.0f} km, the scale the "
                  f"bar was calibrated on)")
            scored["gate1"] = sdf.to_dict(orient="index")

    if 2 in args.gates and len(offsets):
        print("\n=== gate 2: held-out-edge |Δp| for the offsets the map uses ===", flush=True)
        g2 = gate2(offsets, verdict, args)
        if not g2.empty:
            g2.to_csv(FIG / "fbuild_gate2_edgecv.csv", index=False)
            scored["gate2"] = g2.to_dict(orient="records")

    if 3 in args.gates:
        print("\n=== gate 3: THEMIS night-IR ρ (not degraded vs the mosaic map) ===", flush=True)
        g3 = gate3(tiles, map_f, map_mosaic, args)
        if not g3.empty:
            g3.to_csv(FIG / "fbuild_gate3_themis.csv", index=False)
            cols = [c for c in g3.columns if c.startswith("delta_")]
            med = g3[cols].median()
            print(g3.to_string(index=False))
            print(f"\n  median Δρ vs mosaic: "
                  + ", ".join(f"{c[6:]} {med[c]:+.4f} "
                              f"({'PASS' if med[c] >= -fg.THEMIS_TOL else 'FAIL'})" for c in cols))
            scored["gate3"] = {c: float(med[c]) for c in cols}

    if 4 in args.gates and lut:
        print("\n=== gate 4: visual (frame-mean choropleth) ===", flush=True)
        gate4(tiles, map_f, map_mosaic, lut, FIG / "fbuild_gate4_choropleth.png")

    coh = pd.DataFrame()
    if (5 in args.gates) or (6 in args.gates):
        print("\n=== cohort join (labelled tiles -> global lattice -> the F map) ===", flush=True)
        coh = cohort_table(map_f, map_mosaic, tiles, args)
        if not coh.empty:
            coh.to_parquet(FIG / "fbuild_cohort_join.parquet", index=False)
            print(f"  {len(coh):,} labelled tiles in {coh.obs_id.nunique()} obs land inside the "
                  f"block (of 36 cohort images — only the in-region ones are scorable)", flush=True)

    if 5 in args.gates and not coh.empty:
        print("\n=== gate 5: pooled skill Δ(leveled − unleveled) ===", flush=True)
        g5 = gate5(coh)
        if not g5.empty:
            g5.to_csv(FIG / "fbuild_gate5_skill.csv", index=False)
            print(g5.to_string(index=False))
            print(f"\n  gate = Δ pooled pr_auc vs H1-only >= {fg.SKILL_TOL} (the head is IN-SAMPLE on "
                  f"these obs, so absolutes are inflated and are reported, not gated)")
            scored["gate5"] = g5.to_dict(orient="records")

    if 6 in args.gates and not coh.empty:
        print("\n=== gate 6: calibrated-abundance fidelity (both calibrators) ===", flush=True)
        g6, bins = gate6(coh)
        if not g6.empty:
            g6.drop(columns=[c for c in ("per_bin",) if c in g6]).to_csv(
                FIG / "fbuild_gate6_abundance.csv", index=False)
            if not bins.empty:
                bins.to_csv(FIG / "fbuild_gate6_perbin.csv", index=False)
            print(g6[["row", "calibrator", "n", "spearman", "top_ratio", "marginal_l1",
                      "rich_bin_rmse", "passes_top_ratio"]].to_string(index=False))
            print(f"\n  band: top_ratio in {fg.TOP_RATIO_BAND}; monotone calibrators preserve "
                  f"ranking, so only absolute abundance moves")
            scored["gate6"] = g6.drop(columns=[c for c in ("per_bin",) if c in g6]
                                      ).to_dict(orient="records")

    (FIG / "fbuild_gates.json").write_text(json.dumps(scored, indent=2, default=str),
                                           encoding="utf-8")
    print(f"\nwrote {FIG / 'fbuild_gates.json'}")
    if verdict["needs_ruling"]:
        print("⚠ the trend-guard verdict is AMBIGUOUS — the gate table scores all three variants; "
              "which one ships is Brian's call (PLAN_FBuild §7 Q3).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
