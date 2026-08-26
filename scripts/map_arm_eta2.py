"""PLAN_Rebuild step 12, §6's first item -- re-derive the source-frame artifact η² for BOTH
map arms on the corrected basis.

η² is the fraction of map variance organised **between CTX source frames**: the striping
artifact, measured against a rotation null that preserves the field's own spatial
autocorrelation and the frame geometry but breaks their alignment (so the null is a
*geological* floor, not a white-noise one).

**Why this has to be re-derived rather than quoted.** The numbers on record are mosaic
**0.196** / A1 **0.141** (28 % reduction), and they are not usable as they stand:

1. They came from **different A1 definitions** -- the banked pair predates R07, which replaced
   the cheap whole-window ``a1_stats(arr)`` with the per-frame **native** statistic
   (``A1_ARM = a1_native_perframe_tilesupport_v2``). PLAN_Rebuild §6 rules them
   "already non-comparable -- re-derive end to end".
2. They were measured on the **pre-R01 lattice**, where every tile sat on its own sub-cell
   phase (25 of 26 tiles displaced, median 140 m).
3. They are **prevalence-conditioned**, and R74+R29 moved rich prevalence 0.3598 -> 0.373272.
4. ⚠ **A1 had never actually been rendered as a map before step 11** (R06). The 0.141 is a
   *pilot-crop* number from ``scripts/striping_a1_infer_crop.py``, not a map number.

So do **not** pair the new -0.0024 skill cost with the old 0.141.

Three scales are reported, and they are not interchangeable:

``window``
    η² per ~75 km window (469 coarse cells = the E8_N44 pilot crop's own size), each against
    its own null. This is gate 1's headline scale and the primary number.
``tile``
    η² over a whole Murray tile. Larger footprint, so more real geology enters the between-frame
    term; systematically different from the window scale, never comparable to it.
``pilot_crop``
    η² on the **exact world extent** of the historical E8_N44 crop that produced 0.196 / 0.141.
    This is the only cell-for-cell like-for-like successor to those two numbers, and it exists
    precisely so the comparison does not have to be made by eye across three scales.

Both arms are scored on **one common finite mask** per tile so that a coverage difference can
never read as a metric difference, and on one frame-label vocabulary pooled over all 26 tiles
so a frame is never silently dropped for being absent from a reference list.

Needs no network and no CTX zips: the SeamMap frame footprints come from the cached
``cache_v2/ctx_tiles/_frames_{tile}.gpkg`` (all 26 present).

Run (laptop, CPU, ~20-40 min for both arms over 26 tiles):

    C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run --no-capture-output -n geospatial \\
        python -u scripts/map_arm_eta2.py
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

import src.modeling  # noqa: F401,E402  OpenMP bootstrap; must precede numpy/pandas

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402
import rasterio                                                             # noqa: E402

from src import fcompose as fc                                              # noqa: E402
from src import fgates as fg                                                # noqa: E402
from src.striping import eta2, eta2_rotation_null, load_frames              # noqa: E402

ARMS = {"baseline": REPO / "reports" / "map_region", "a1": REPO / "reports" / "map_a1"}
FIG = REPO / "reports" / "figures"
THEMIS = REPO / "cache_v2" / "validation" / "themis_night_ir_region.tif"

# The historical crop that produced mosaic 0.196 / A1 0.141 (scripts/f_pilot_crop.py:58-61):
# tile E8_N44, world UL (519317.3, 2837505.5), 15008 native px at 5 m/px = 469 coarse cells.
PILOT = {"tile": "E8_N44", "ul_x": 519317.3, "ul_y": 2837505.5, "coarse_px": 469}

# The banked pre-rebuild pair, carried here only so the report can state what it is NOT
# comparable to. See the module docstring.
BANKED = {"baseline": 0.196, "a1": 0.141, "basis": "pilot crop, pre-R01 lattice, pre-R07 A1 "
                                                   "definition, prevalence 0.3598"}


def common_tiles(dirs: dict[str, Path], layer: str) -> list[str]:
    sets = []
    for d in dirs.values():
        sets.append({p.name[: -len(f"_{layer}.tif")] for p in d.glob(f"*_{layer}.tif")
                     if not p.name.startswith("regional_")})
    common = sorted(set.intersection(*sets))
    extra = {a: sorted(s - set(common)) for a, s in zip(dirs, sets) if s - set(common)}
    if extra:
        print(f"  ⚠ REDUCED FOOTPRINT: scoring the {len(common)} tiles both arms carry; "
              f"arm-only tiles dropped: {extra}", flush=True)
    return common


def frame_vocabulary(tiles: list[str]) -> tuple[list[str], dict[str, object]]:
    """One PRODUCT_ID vocabulary pooled over every tile, so no frame is ever dropped.

    ``fc.frame_labels_on_grid`` filters shapes to those whose PRODUCT_ID is in the vocabulary,
    so a vocabulary read off a stale reference list would silently unlabel real frames (and
    unlabelled cells leave the η² denominator entirely). Building it from the frames themselves
    makes that impossible; the count is cross-checked against
    ``reports/figures/region_frame_list.csv`` for provenance only.
    """
    frames, pids = {}, set()
    for t in tiles:
        g = load_frames(t)
        frames[t] = g
        pids |= set(g["PRODUCT_ID"].astype(str))
    lut = sorted(pids)
    ref = FIG / "region_frame_list.csv"
    if ref.exists():
        n_ref = pd.read_csv(ref).PRODUCT_ID.nunique()
        print(f"  frame vocabulary: {len(lut)} PRODUCT_IDs pooled over {len(tiles)} tiles "
              f"({n_ref} in region_frame_list.csv -- provenance cross-check only)", flush=True)
    else:
        print(f"  frame vocabulary: {len(lut)} PRODUCT_IDs pooled over {len(tiles)} tiles",
              flush=True)
    return lut, frames


def read_layer(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        a = ds.read(1).astype(np.float64)
        nd = ds.nodata
    if nd is not None and np.isfinite(nd):
        a[a == nd] = np.nan
    return a


def pilot_window(grid) -> tuple[int, int, int] | None:
    """(row, col, size) of the historical pilot crop on this tile's grid, or None if off-tile.

    Derived from world coordinates, not from a remembered pixel offset: the crop is defined by
    its world UL, and the g2 lattice moved +100 m E / -140 m S relative to the pre-R01 product
    the original offsets were expressed in. Re-using row_off/col_off would silently score a
    different patch of ground and call it the same crop.
    """
    a, _, c, _, e, f = grid.transform
    col = (PILOT["ul_x"] - c) / a
    row = (PILOT["ul_y"] - f) / e
    r0, c0 = int(round(row)), int(round(col))
    n = PILOT["coarse_px"]
    if r0 < 0 or c0 < 0 or r0 + n > grid.height or c0 + n > grid.width:
        return None
    return r0, c0, n


def figure(win_df: pd.DataFrame, tile_df: pd.DataFrame, pilot_df: pd.DataFrame,
           out: Path) -> None:
    """Three panels: the gate-1 headline, the per-window null comparison, the like-for-like crop.

    Panel 2 plots each window against **its own** rotation null rather than a shared floor,
    because that is the whole point of the null: a window whose η² sits on the diagonal has
    no frame structure beyond what its own geology reproduces under rotation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in ("baseline", "a1") if a in set(win_df.arm)]
    colors = {"baseline": "tab:blue", "a1": "tab:orange"}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

    x = np.arange(len(arms))
    med = win_df.groupby("arm").eta2.median().reindex(arms)
    nul = win_df.groupby("arm").null_p95.median().reindex(arms)
    ax[0].bar(x - 0.18, med, 0.36, label="partition η² (median window)",
              color=[colors[a] for a in arms])
    ax[0].bar(x + 0.18, nul, 0.36, label="rotation-null p95", color="0.75")
    ax[0].axhline(fg.ETA2_BAR, color="tab:red", ls="--", lw=1,
                  label=f"F-reopening bar {fg.ETA2_BAR}")
    ax[0].set_xticks(x, arms)
    ax[0].set_ylabel("η²")
    ax[0].set_title(f"artifact η² vs its own geological floor\n"
                    f"{len(win_df) // max(len(arms), 1)} windows x {tile_df.tile.nunique()} tiles",
                    fontsize=9)
    ax[0].legend(fontsize=7)

    for a in arms:
        g = win_df[win_df.arm == a]
        ax[1].scatter(g.null_p95, g.eta2, s=14, alpha=0.7, label=a, color=colors[a])
    lim = float(np.nanpercentile(win_df[["eta2", "null_p95"]].to_numpy(), 99.5))
    ax[1].plot([0, lim], [0, lim], color="0.6", lw=0.8)
    ax[1].set_xlabel("rotation-null p95 (this window's geological floor)")
    ax[1].set_ylabel("partition η²")
    ax[1].set_title("per window: DISTANCE FROM the diagonal is the real question", fontsize=9)
    ax[1].legend(fontsize=7, loc="lower right")
    # The caveat belongs ON the panel a reader actually looks at. A1's cloud moves toward the
    # ORIGIN (compression: both η² and its own null shrink), not toward the DIAGONAL
    # (de-striping: η² falls to its geological floor). The ratio medians say which happened.
    if len(arms) == 2:
        rb = win_df[win_df.arm == arms[0]].ratio.median()
        ra = win_df[win_df.arm == arms[1]].ratio.median()
        ax[1].text(0.03, 0.97,
                   f"median η²/null_p95:  {arms[0]} {rb:.2f}  →  {arms[1]} {ra:.2f}\n"
                   f"A1 moves toward the ORIGIN, not the diagonal:\n"
                   f"it narrows the field's bulk, so the floor drops too.",
                   transform=ax[1].transAxes, va="top", ha="left", fontsize=7,
                   bbox=dict(boxstyle="round", fc="#fff6e0", ec="0.6", lw=0.6))

    if not pilot_df.empty:
        lab, val, col = [], [], []
        for a in arms:
            p = pilot_df[pilot_df.arm == a]
            if len(p):
                lab += [f"{a}\nrebuilt", f"{a}\nbanked"]
                val += [float(p.eta2.iloc[0]), float(p.banked_pre_rebuild.iloc[0])]
                col += [colors[a], "0.75"]
        ax[2].bar(np.arange(len(val)), val, 0.6, color=col)
        ax[2].set_xticks(np.arange(len(lab)), lab, fontsize=7)
        ax[2].set_ylabel("η²")
        ax[2].set_title(f"the E8_N44 pilot crop, like for like\n"
                        f"(grey = banked, different lattice/A1 definition/prevalence)",
                        fontsize=9)
    fig.suptitle("PLAN_Rebuild step 12 §6 — source-frame artifact η², both arms, "
                 "corrected basis")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ARMS["baseline"]))
    ap.add_argument("--a1", default=str(ARMS["a1"]))
    ap.add_argument("--layer", default="prob_raw",
                    help="raw P(rich) is the quantity every on-record η² was computed on")
    ap.add_argument("--tiles", nargs="*", default=None)
    ap.add_argument("--window-px", type=int, default=fg.WINDOW_PX)
    ap.add_argument("--null-draws", type=int, default=fg.NULL_DRAWS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--themis", default=str(THEMIS))
    ap.add_argument("--out-prefix", default="step12_eta2")
    args = ap.parse_args()

    dirs = {"baseline": Path(args.baseline), "a1": Path(args.a1)}
    tiles = args.tiles or common_tiles(dirs, args.layer)
    print(f"=== η² on {len(tiles)} tiles x {len(dirs)} arms, layer '{args.layer}', "
          f"window {args.window_px} px, {args.null_draws} null draws ===", flush=True)
    lut, frames = frame_vocabulary(tiles)
    themis_path = Path(args.themis)
    if not themis_path.exists():
        print(f"  ⚠ no THEMIS raster at {themis_path} -- the ρ column will be blank", flush=True)

    win_rows, tile_rows, pilot_rows = [], [], []
    for tile in tiles:
        t0 = time.monotonic()
        grid = fc.tile_grid_from_raster(dirs["baseline"] / f"{tile}_{args.layer}.tif", tile)
        labels = fc.frame_labels_on_grid(grid, frames[tile], lut)
        arrs = {a: read_layer(d / f"{tile}_{args.layer}.tif") for a, d in dirs.items()}
        # ONE footprint for both arms: a coverage difference must never read as a metric one
        mask = np.ones(grid.shape, dtype=bool)
        for a in arrs.values():
            mask &= np.isfinite(a)
        themis = fg.themis_on_grid(grid, themis_path) if themis_path.exists() else None
        if themis is not None:
            mask &= np.isfinite(themis)
        pw = pilot_window(grid)

        for arm, arr in arrs.items():
            masked = np.where(mask, arr, np.nan)
            for s in fg.window_eta2(masked, labels, tile, win_px=args.window_px,
                                    n_draws=args.null_draws, seed=args.seed):
                win_rows.append({"arm": arm, "tile": tile, "r0": s.r0, "c0": s.c0,
                                 "n_cells": s.n_cells, "n_frames": s.n_frames,
                                 "eta2": s.eta2, "null_mean": s.null_mean,
                                 "null_p95": s.null_p95, "excess": s.excess,
                                 "ratio": s.ratio})
            e, nm, n95, nc, nf = fg.eta2_with_null(masked, labels, n_draws=args.null_draws,
                                                   seed=args.seed)
            rho, nrho = ((np.nan, 0) if themis is None
                         else fg.spearman_rho(masked, np.where(mask, themis, np.nan)))
            tile_rows.append({"arm": arm, "tile": tile, "n_cells": nc, "n_frames": nf,
                              "eta2": e, "null_mean": nm, "null_p95": n95,
                              "excess": e - nm if np.isfinite(e) else np.nan,
                              "ratio": e / n95 if np.isfinite(n95) and n95 > 0 else np.nan,
                              "themis_rho": rho, "n_rho": nrho,
                              "common_mask_frac": float(mask.mean())})
            if pw is not None:
                r0, c0, n = pw
                v = masked[r0:r0 + n, c0:c0 + n]
                lab = labels[r0:r0 + n, c0:c0 + n]
                fin = np.isfinite(v) & (lab >= 0)
                pe = eta2(v, lab, fin)
                pnm, pn95 = eta2_rotation_null(v, lab, fin, n=args.null_draws, seed=args.seed)
                pilot_rows.append({"arm": arm, "tile": tile, "r0": r0, "c0": c0, "size": n,
                                   "n_cells": int(fin.sum()),
                                   "n_frames": int(np.unique(lab[fin]).size),
                                   "eta2": float(pe), "null_mean": pnm, "null_p95": pn95,
                                   "banked_pre_rebuild": BANKED[arm]})
        print(f"  {tile:10s} mask {mask.mean():6.1%}  frames {tile_rows[-1]['n_frames']:3d}  "
              f"({time.monotonic() - t0:.0f}s)", flush=True)

    FIG.mkdir(parents=True, exist_ok=True)
    win_df = pd.DataFrame(win_rows)
    tile_df = pd.DataFrame(tile_rows)
    pilot_df = pd.DataFrame(pilot_rows)
    win_df.to_csv(FIG / f"{args.out_prefix}_windows.csv", index=False)
    tile_df.to_csv(FIG / f"{args.out_prefix}_tiles.csv", index=False)
    if not pilot_df.empty:
        pilot_df.to_csv(FIG / f"{args.out_prefix}_pilotcrop.csv", index=False)

    summary = {}
    for arm in dirs:
        w = win_df[win_df.arm == arm]
        t = tile_df[tile_df.arm == arm]
        summary[arm] = {
            "n_tiles": int(t.tile.nunique()), "n_windows": int(len(w)),
            "window_eta2_median": float(w.eta2.median()),
            "window_eta2_p90": float(w.eta2.quantile(0.90)),
            "window_null_mean_median": float(w.null_mean.median()),
            "window_null_p95_median": float(w.null_p95.median()),
            "window_excess_median": float(w.excess.median()),
            "window_ratio_median": float(w.ratio.median()),
            "frac_windows_below_bar": float((w.eta2 <= fg.ETA2_BAR).mean()),
            "passes_eta2_bar": bool(w.eta2.median() <= fg.ETA2_BAR),
            "tile_eta2_median": float(t.eta2.median()),
            "tile_null_mean_median": float(t.null_mean.median()),
            "tile_excess_median": float(t.excess.median()),
            "themis_rho_median": float(t.themis_rho.median()),
        }
        if not pilot_df.empty:
            p = pilot_df[pilot_df.arm == arm]
            if len(p):
                summary[arm]["pilot_crop_eta2"] = float(p.eta2.iloc[0])
                summary[arm]["pilot_crop_null_mean"] = float(p.null_mean.iloc[0])

    base, a1 = summary["baseline"], summary["a1"]
    deltas = {}
    for k in ("window_eta2_median", "window_null_p95_median", "window_excess_median",
              "window_ratio_median", "tile_eta2_median", "tile_excess_median",
              "pilot_crop_eta2", "themis_rho_median"):
        if k in base and k in a1:
            deltas[k] = {"baseline": base[k], "a1": a1[k], "delta": a1[k] - base[k],
                         "relative": ((a1[k] - base[k]) / base[k]) if base[k] else float("nan")}

    # ⚠ The medians above are NOT the whole story, and the difference is load-bearing.
    # A1 renormalises per frame, which compresses the WHOLE field -- so it lowers the
    # rotation null as well as the between-frame term. Paired, per-unit sign censuses
    # separate "A1 removed frame structure" from "A1 shrank everything":
    #   * raw eta2       -- the quantity the historical 0.196/0.141 pair used
    #   * excess         -- eta2 minus its OWN null mean
    #   * ratio          -- eta2 over its OWN null p95, i.e. artifact RELATIVE to geology
    # If raw improves but ratio does not, A1 is compressing, not selectively de-striping.
    paired = {}
    for name, df, keys in (("window", win_df, ["tile", "r0", "c0"]),
                           ("tile", tile_df, ["tile"])):
        p = df.pivot_table(index=keys, columns="arm",
                           values=["eta2", "null_mean", "null_p95", "ratio"])
        d_raw = p["eta2"]["a1"] - p["eta2"]["baseline"]
        d_exc = ((p["eta2"]["a1"] - p["null_mean"]["a1"])
                 - (p["eta2"]["baseline"] - p["null_mean"]["baseline"]))
        d_rat = p["ratio"]["a1"] - p["ratio"]["baseline"]
        paired[name] = {
            "n": int(len(p)),
            "a1_better_raw_eta2": int((d_raw < 0).sum()),
            "a1_better_excess": int((d_exc < 0).sum()),
            "a1_better_ratio": int((d_rat < 0).sum()),
            "raw_delta_median": float(d_raw.median()),
            "raw_delta_min": float(d_raw.min()), "raw_delta_max": float(d_raw.max()),
            "ratio_median_baseline": float(p["ratio"]["baseline"].median()),
            "ratio_median_a1": float(p["ratio"]["a1"].median()),
            "null_p95_median_baseline": float(p["null_p95"]["baseline"].median()),
            "null_p95_median_a1": float(p["null_p95"]["a1"].median()),
        }

    print("\n=== η² SCORECARD (raw P(rich), SeamMap partition, one common mask per tile) ===")
    keys = ["n_tiles", "n_windows", "window_eta2_median", "window_null_p95_median",
            "window_excess_median", "frac_windows_below_bar", "passes_eta2_bar",
            "tile_eta2_median", "pilot_crop_eta2", "themis_rho_median"]
    print(pd.DataFrame(summary).reindex(keys).to_string())
    print("\n=== A1 vs baseline (medians) ===")
    for k, v in deltas.items():
        print(f"  {k:26s} {v['baseline']:.4f} -> {v['a1']:.4f}   "
              f"Δ {v['delta']:+.4f} ({v['relative']:+.1%})")

    print("\n=== ⚠ PAIRED sign census — A1 narrows the field's bulk, so it lowers the NULL too ===")
    for name, p in paired.items():
        n = p["n"]
        print(f"  {name} scale (n={n}):")
        print(f"    raw η²  : A1 better on {p['a1_better_raw_eta2']}/{n} "
              f"({p['a1_better_raw_eta2'] / n:.0%}), median Δ {p['raw_delta_median']:+.4f}, "
              f"range {p['raw_delta_min']:+.4f}..{p['raw_delta_max']:+.4f}")
        print(f"    excess  : A1 better on {p['a1_better_excess']}/{n} "
              f"({p['a1_better_excess'] / n:.0%})")
        print(f"    RATIO   : A1 better on {p['a1_better_ratio']}/{n} "
              f"({p['a1_better_ratio'] / n:.0%})  — median ratio "
              f"{p['ratio_median_baseline']:.3f} -> {p['ratio_median_a1']:.3f}")
        print(f"    null p95: {p['null_p95_median_baseline']:.4f} -> "
              f"{p['null_p95_median_a1']:.4f} (A1 lowers the geological floor as well)")
    print("  READ THIS WITH THE HEADLINE: A1's raw-η² reduction is real and is the quantity the\n"
          "  banked 0.196/0.141 pair measured, but η² RELATIVE TO ITS OWN ROTATION NULL barely\n"
          "  moves — so A1 works substantially by narrowing the BULK of the field, not only by\n"
          "  removing frame structure. ⚠ Not a uniform compression: on prob_raw the IQR falls\n"
          "  ~15% while the sd RISES ~3%, i.e. the tails widen slightly (notebook 29 §2b).\n"
          "  It is a PARTIAL mitigation, and this is the measurement that says so quantitatively.")
    print(f"\n⚠ NOT comparable to the banked pre-rebuild pair "
          f"(baseline {BANKED['baseline']} / A1 {BANKED['a1']}), whose basis was: "
          f"{BANKED['basis']}. The `pilot_crop_eta2` row is the closest successor -- same world "
          f"extent, but on the corrected lattice, the R07 A1 definition and prevalence 0.373272.")

    figure(win_df, tile_df, pilot_df, FIG / f"{args.out_prefix}.png")

    out = FIG / f"{args.out_prefix}_summary.json"
    out.write_text(json.dumps({
        "layer": args.layer, "tiles": tiles, "n_frames_vocabulary": len(lut),
        "window_px": args.window_px, "null_draws": args.null_draws, "seed": args.seed,
        "eta2_bar": fg.ETA2_BAR, "themis": str(themis_path),
        "pilot_crop_definition": PILOT, "banked_pre_rebuild_NOT_comparable": BANKED,
        "summary": summary, "a1_vs_baseline": deltas, "paired_sign_census": paired,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out.relative_to(REPO)} (+ _windows/_tiles/_pilotcrop.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
