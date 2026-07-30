"""PLAN_FBuild Stage D — composite the H4-leveled per-frame logits into the regional map.

Consumes Stage B's per-frame npz logit tiles + Stage C's per-frame offset table and writes, per
Murray tile, on the EXACT grid of the existing mosaic-path map:

  {tile}_{variant}_prob_raw.tif      composite P(rich), uncalibrated   (the gate-scoring layer)
  {tile}_{variant}_prob.tif          isotonic-calibrated P(rich)       (Tier 1)
  {tile}_{variant}_abundance.tif     qmatch fractional_area            (Tier 2, F-path calibrator)
  {tile}_{variant}_abundance_moscal.tif   same under the REUSED mosaic-path calibrator (§5.1 column)
  {tile}_{variant}_overlap_dp.tif    max co-located |Δp| after leveling (H6 overlap QA)
  {tile}_n_frames.tif / _primary_frame.tif / _incidence.tif / _offset_source.tif   (H6 provenance)
  {tile}.json                        sidecar (map_region's keys + Stage-D provenance)

for variant in {h1only (o=0), full (offset_logit), resid (offset_residual_only),
pfree (offset_logit_pfree)} — PLAN §1 deliverable 5 requires all of them from ONE Stage-B run, which
is exactly why Stage C emits only the offset table. `pfree` was added 2026-07-30 (Brian) and is the
SHIPPED variant: the free solve returns a −22.7-logit east–west ramp that rails 51.8% of co-located
tiles, because a small patchy per-step bias gets integrated over ~100 chain steps. `full`/`resid`
are retained as the pre-declared audit trail. See `lv.solve_offsets_planefree` for the justification.

Composite rule (PLAN §5): p = sigmoid(mean_f[logit(prob_f) + o_f]) — mean in LOGIT space over the
frames covering a tile, one sigmoid at the end. Calibration is applied ONCE to the composited
probability (both maps consume the same raw P, as src/mapping.predict_window does) — never per frame,
because calibrate_abundance is nonlinear and mean-then-calibrate != calibrate-then-mean.

The headline (plain-named `{tile}_prob.tif` / `_abundance.tif` / `_prob_raw.tif`) copies exist so
notebook 24 and the `src.striping` helpers work unchanged against `--out-dir`. They are written ONLY
when the trend-guard verdict names a variant: an AMBIGUOUS verdict (`full_pending_ruling`) leaves the
map deliberately unshipped until Brian rules (§0.1 guard 1, §7 Q3) — pass --headline to override.

Run (laptop, CPU, ~minutes):
  conda run --no-capture-output -n geospatial python -u scripts/f_region_staged.py
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

from scripts.map_region import BLOCK_TILES
from src import fcompose as fc
from src import leveling as lv
from src.calibration import CalibrationLayer
from src.mapping import write_geotiff

FIG = REPO / "reports" / "figures"
DEFAULT_LOGITS = REPO / "reports" / "f_region_logits"
DEFAULT_OUT = REPO / "reports" / "map_fbuild"
DEFAULT_MAP = REPO / "reports" / "map_region"
OFFSETS_CSV = FIG / "fbuild_stagec_offsets.csv"
GUARD_CSV = FIG / "fbuild_trend_guard.csv"
CAL_F = REPO / "models" / "deployable_f_center" / "calibration.npz"
CAL_MOSAIC = REPO / "models" / "deployable" / "calibration.npz"

VARIANTS = {"h1only": None,                      # o_f == 0 (the un-leveled counterpart, deliverable 5)
            "full": "offset_logit",
            "resid": "offset_residual_only",
            # `pfree` — the SHIPPED solve (Brian, 2026-07-30): the region-wide plane is constrained
            # out of the solve because the measured per-step term is patchy, not a constant gradient.
            # `full`/`resid` stay on disk as the pre-declared audit trail. See lv.solve_offsets_planefree.
            "pfree": "offset_logit_pfree"}
APPLY_TO_VARIANT = {"full": "full", "residual": "resid", "full_pending_ruling": None}
SHARED_LAYERS = ("n_frames", "primary_frame", "incidence", "offset_source")


# --------------------------------------------------------------------------- inputs
def load_offsets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing {path} — run scripts/f_region_stagec.py first")
    df = pd.read_csv(path).set_index("PRODUCT_ID")
    for col in ("offset_logit", "offset_residual_only"):
        if col not in df.columns:
            raise SystemExit(f"{path} has no column {col!r} — is it a Stage-C offsets table?")
    if "offset_logit_pfree" not in df.columns:                # pre-2026-07-30 Stage-C table
        print("⚠ no `offset_logit_pfree` column — Stage C predates the plane-free solve; "
              "the `pfree` variant is unavailable (re-run Stage C to get it)", flush=True)
    if "offset_source" not in df.columns:
        df["offset_source"] = "solved"
    return df


def load_verdict(path: Path) -> dict:
    if not path.exists():
        print(f"⚠ {path} missing — no trend-guard verdict; headline map will NOT be written "
              f"unless --headline is given", flush=True)
        return {"verdict": "UNKNOWN", "apply": "full_pending_ruling", "needs_ruling": True}
    row = pd.read_csv(path).iloc[0].to_dict()
    return {"verdict": str(row.get("verdict", "UNKNOWN")),
            "apply": str(row.get("apply", "full_pending_ruling")),
            "needs_ruling": bool(row.get("needs_ruling", True)),
            "why": str(row.get("why", "")), "lambda_star": row.get("lambda_star")}


def frame_index(logits_dir: Path, cache: Path, rebuild: bool = False) -> pd.DataFrame:
    """Per-frame global-tile bbox, so each Murray tile only opens the npzs that can reach it.

    frame_tile_map.csv is NOT used as the authority here: it is derived from the SeamMap PARTITION
    (one winning frame per mosaic pixel), while the F build renders each frame's full cam2map
    footprint, which spills into tiles that frame never won. The npz keys are the ground truth.
    """
    npzs = sorted(logits_dir.glob("*.npz"))
    if not npzs:
        raise SystemExit(f"no *.npz in {logits_dir} — Stage B must run (and be tar'd home) first")
    if cache.exists() and not rebuild:
        cached = pd.read_csv(cache)
        # Auto-invalidate: a cache built during a PARTIAL Stage B was otherwise baked in permanently
        # and invisibly — the print lived inside the build branch, so a stale cached run emitted no
        # frame count at all, and even `--overwrite` reused it (review 2026-07-29).
        newest = max(p.stat().st_mtime for p in npzs)
        stamp = cache.with_suffix(".stamp.json")
        prev = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else {}
        if len(cached) == len(npzs) and prev.get("newest_mtime", -1) >= newest - 1e-6:
            print(f"frame index: {len(cached)} frames (cached)", flush=True)
            return cached
        print(f"frame index: STALE ({len(cached)} cached vs {len(npzs)} npz on disk"
              f"{', newer files present' if prev.get('newest_mtime', -1) < newest else ''}) "
              f"-> rebuilding", flush=True)
    rows = []
    t0 = time.monotonic()
    for p in npzs:
        z = np.load(p)
        ti, tj = z["TI"], z["TJ"]
        ti0, ti1, tj0, tj1 = fc.frame_bbox(ti, tj)
        rows.append({"PRODUCT_ID": p.stem, "n_tiles": int(ti.size),
                     "TI_min": ti0, "TI_max": ti1, "TJ_min": tj0, "TJ_max": tj1})
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    cache.with_suffix(".stamp.json").write_text(
        json.dumps({"n_npz": len(npzs),
                    "newest_mtime": max(p.stat().st_mtime for p in npzs)}), encoding="utf-8")
    print(f"frame index: {len(df)} frames, {df.n_tiles.sum():,} global tiles "
          f"({time.monotonic() - t0:.0f}s) -> {cache}", flush=True)
    return df


# --------------------------------------------------------------------------- one tile
def seam_labels(tile: str, grid: fc.TileGrid, frame_lut: dict[str, int]) -> np.ndarray | None:
    """SeamMap single-owner labels on this grid, valued by GLOBAL frame index (gate 1's labelling).

    Returns None (with a warning) if the dissolved SeamMap gpkg or its CRS reference is unavailable,
    so the composite still ships without the partition scoring layer.
    """
    try:
        from src.striping import load_frames
        g = load_frames(tile)
        return fc.frame_labels_on_grid(grid, g, list(frame_lut))
    except Exception as exc:                     # noqa: BLE001 - any I/O/CRS failure is non-fatal here
        print(f"    ⚠ {tile}: no SeamMap labels ({type(exc).__name__}: {exc}) -> skipping the "
              f"partition layer (gate 1 needs it; the mean composite is unaffected)", flush=True)
        return None


def compose_tile(tile: str, grid: fc.TileGrid, frames: pd.DataFrame, offsets: pd.DataFrame,
                 logits_dir: Path, frame_lut: dict[str, int], variants: list[str],
                 labels: np.ndarray | None = None):
    """One pass over the tile's contributing frames, accumulating every variant simultaneously.

    When `labels` is given, the PARTITION composite is accumulated alongside the mean one: every
    pixel takes the value of its SeamMap-designated owner frame. That is the quantity every on-record
    η² number was computed on (`scripts/f_h2_eta2.score`), and the mosaic/A1 maps have exactly one
    value per pixel by construction, so it is what makes gate 1 label- and row-comparable.
    """
    accums = {v: fc.TileAccum.zeros(grid.shape) for v in variants}
    part = ({v: np.full(grid.shape, np.nan, dtype=np.float32) for v in variants}
            if labels is not None else None)
    used, missing_offset, n_px = [], [], 0
    for r in frames.itertuples():
        pid = r.PRODUCT_ID
        z = np.load(logits_dir / f"{pid}.npz")
        TI, TJ, prob = z["TI"], z["TJ"], z["prob"]
        if TI.size == 0:
            continue
        rows, cols = fc.frame_rows_cols(grid, TI, TJ)
        inb = (rows >= 0) & (rows < grid.height) & (cols >= 0) & (cols < grid.width)
        if not inb.any():
            continue
        rows, cols = rows[inb], cols[inb]
        base_logit = lv.logit(prob[inb])
        if pid in offsets.index:
            orow = offsets.loc[pid]
            inc = float(orow.get("incidence", np.nan))
            src = fc.OFFSET_SOURCE_CODE.get(str(orow.get("offset_source", "solved")), 0)
        else:
            missing_offset.append(pid)
            orow, inc, src = None, np.nan, fc.OFFSET_SOURCE_CODE["none"]
        fidx = frame_lut[pid]
        own = labels[rows, cols] == fidx if labels is not None else None
        for v in variants:
            col = VARIANTS[v]
            if col is None or orow is None:
                off = 0.0
            else:
                off = float(orow[col])
                if not np.isfinite(off):
                    off = 0.0
            n = accums[v].add_frame(rows, cols, base_logit + off,
                                    frame_idx=fidx, incidence=inc, src_code=src)
            if part is not None and own is not None and own.any():
                part[v][rows[own], cols[own]] = lv.sigmoid(base_logit[own] + off).astype(np.float32)
            if v == variants[0]:
                n_px += n
        used.append(pid)
    return accums, part, used, missing_offset, n_px


def write_tile(tile: str, grid: fc.TileGrid, accums: dict, part: dict | None, out_dir: Path,
               cal_f, cal_mos, variants: list[str], headline: str | None, meta_extra: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    shared_written = False
    for v in variants:
        res = accums[v].finish()
        p = res["prob_raw"]
        fin = np.isfinite(p)
        layers = {f"{v}_prob_raw": p, f"{v}_overlap_dp": res["overlap_dp"]}
        if part is not None:
            layers[f"{v}_prob_partition"] = part[v]
        if cal_f is not None:
            layers[f"{v}_prob"] = _apply(cal_f.calibrate_prob, p, fin)
            layers[f"{v}_abundance"] = _apply(cal_f.calibrate_abundance, p, fin)
        if cal_mos is not None:
            layers[f"{v}_abundance_moscal"] = _apply(cal_mos.calibrate_abundance, p, fin)
        if not shared_written:                      # variant-independent H6 layers
            for name in SHARED_LAYERS:
                layers[name] = res[name]
            shared_written = True
        for name, arr in layers.items():
            write_geotiff(out_dir / f"{tile}_{name}.tif", arr, _affine(grid), grid.crs_wkt)
        ab = layers.get(f"{v}_abundance")
        stats[v] = {
            "n_finite": int(fin.sum()), "coverage": float(fin.mean()),
            "prob_mean": float(np.nanmean(p)) if fin.any() else None,
            "rich_share_at_0p5": float(np.nanmean(p[fin] > 0.5)) if fin.any() else None,
            "abundance_mean": float(np.nanmean(ab)) if ab is not None and fin.any() else None,
            "abundance_saturated_frac": (float(np.nanmean(ab[fin] >= cal_f._t2[1][-1] - 1e-12))
                                         if ab is not None and cal_f is not None and fin.any() else None),
            "overlap_dp_median": (float(np.nanmedian(res["overlap_dp"]))
                                  if np.isfinite(res["overlap_dp"]).any() else None),
            "partition_coverage": (float(np.isfinite(part[v]).mean()) if part is not None else None),
            "mean_n_frames": float(np.nanmean(res["n_frames"])) if fin.any() else None,
            "max_n_frames": int(np.nanmax(res["n_frames"])) if fin.any() else 0,
        }
    if headline is not None:                        # notebook-24-compatible plain names
        res = accums[headline].finish()
        p = res["prob_raw"]
        fin = np.isfinite(p)
        write_geotiff(out_dir / f"{tile}_prob_raw.tif", p, _affine(grid), grid.crs_wkt)
        if cal_f is not None:
            write_geotiff(out_dir / f"{tile}_prob.tif", _apply(cal_f.calibrate_prob, p, fin),
                          _affine(grid), grid.crs_wkt)
            write_geotiff(out_dir / f"{tile}_abundance.tif",
                          _apply(cal_f.calibrate_abundance, p, fin), _affine(grid), grid.crs_wkt)
    side = {"murray_tile": tile, "tile_px": 32, "raster_shape": list(grid.shape),
            "ti_min": 1, "tj_min": 1,
            "n_predicted_tiles": stats[variants[0]]["n_finite"],
            "calibrated": cal_f is not None, "isotonic": cal_f is not None,
            "prob_mean": stats.get(headline or variants[0], {}).get("prob_mean"),
            "rich_share_at_0p5": stats.get(headline or variants[0], {}).get("rich_share_at_0p5"),
            "abundance_mean": stats.get(headline or variants[0], {}).get("abundance_mean"),
            "stage": "D", "headline_variant": headline, "per_variant": stats,
            "global_lattice": {"Kj": grid.Kj, "Ki": grid.Ki, "dx_m": round(grid.dx_m, 3),
                               "dy_m": round(grid.dy_m, 3),
                               "tie_margin_m": round(grid.tie_margin_m, 4)},
            **meta_extra}
    (out_dir / f"{tile}.json").write_text(json.dumps(side, indent=2), encoding="utf-8")
    return stats


def _affine(grid: fc.TileGrid):
    from rasterio.transform import Affine
    return Affine(*grid.transform)


def _apply(fn, p: np.ndarray, fin: np.ndarray) -> np.ndarray:
    """Calibrate only the finite cells (np.interp would turn NaN into NaN anyway, but the calibrators
    are not NaN-documented and this keeps nodata exactly NaN)."""
    out = np.full(p.shape, np.nan, dtype=np.float64)
    if fin.any():
        out[fin] = fn(p[fin])
    return out


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits-dir", default=str(DEFAULT_LOGITS))
    ap.add_argument("--offsets", default=str(OFFSETS_CSV))
    ap.add_argument("--guard", default=str(GUARD_CSV))
    ap.add_argument("--map-dir", default=str(DEFAULT_MAP), help="reference grid (the mosaic-path map)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--tiles", nargs="*", default=None, help="default = the 26-tile block")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--calibration-f", default=str(CAL_F))
    ap.add_argument("--calibration-mosaic", default=str(CAL_MOSAIC))
    ap.add_argument("--headline", choices=list(VARIANTS), default=None,
                    help="override the trend-guard verdict's choice of shipped variant")
    ap.add_argument("--raw", action="store_true", help="skip calibration entirely")
    ap.add_argument("--no-partition", action="store_true",
                    help="skip the SeamMap partition-composite layer (gate 1 needs it)")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write a headline map even though Stage B is short of the planned frames")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    logits_dir, out_dir, map_dir = Path(args.logits_dir), Path(args.out_dir), Path(args.map_dir)
    tiles = args.tiles or list(BLOCK_TILES)
    variants = [v for v in VARIANTS if v in args.variants]     # keep the canonical order
    offsets = load_offsets(Path(args.offsets))
    guard = load_verdict(Path(args.guard))
    dropped = [v for v in variants if VARIANTS[v] and VARIANTS[v] not in offsets.columns]
    if dropped:                                                # never silently skip coverage
        print(f"⚠ variants {dropped} have no offset column in {args.offsets} -> SKIPPED", flush=True)
        variants = [v for v in variants if v not in dropped]

    headline = args.headline or APPLY_TO_VARIANT.get(guard["apply"])
    print(f"trend-guard verdict: {guard['verdict']} -> apply '{guard['apply']}'", flush=True)
    if headline is None:
        print("⚠ NO headline map will be written: the verdict does not name a variant "
              "(§0.1 guard 1 — an AMBIGUOUS attribution must not silently ship full offsets).\n"
              f"  All {len(variants)} variants ARE written under explicit names; "
              f"pass --headline to ship one.", flush=True)
    else:
        print(f"headline (plain-named) variant = {headline}"
              f"{' [--headline override]' if args.headline else ''}", flush=True)

    cal_f = cal_mos = None
    if not args.raw:
        cf, cm = Path(args.calibration_f), Path(args.calibration_mosaic)
        if not cf.exists():
            raise SystemExit(f"missing {cf} — run scripts/bank_calibration_f.py (Brian 2026-07-28: "
                             f"the F path gets its own calibrator; --raw to skip)")
        cal_f = CalibrationLayer.load(cf)
        cal_mos = CalibrationLayer.load(cm) if cm.exists() else None
        print(f"calibration: F={cf.name} (ceiling {cal_f._t2[1][-1]:.6f})"
              f"{'; mosaic-reused column ON' if cal_mos is not None else ''}", flush=True)

    idx = frame_index(logits_dir, out_dir / "frame_index.csv", args.rebuild_index)
    # Census against the plan and against Stage C, so a partial Stage B cannot pass unremarked
    # (Stage C is explicitly partial-safe; Stage D was not — review 2026-07-29).
    planned = FIG / "region_frame_list.csv"
    n_planned = len(pd.read_csv(planned)) if planned.exists() else 0
    n_matched = int(idx.PRODUCT_ID.isin(offsets.index).sum())
    print(f"census: {len(idx)} frames with logits"
          + (f" of {n_planned} planned" if n_planned else "")
          + f"; {n_matched} matched to a Stage-C offset row", flush=True)
    if n_planned and len(idx) < n_planned:
        msg = (f"⚠ Stage B is short by {n_planned - len(idx)} frame(s) — the composite will have "
               f"holes to patch with the mosaic + flag in H6 (PLAN §2)")
        if headline is not None and not args.allow_partial:
            raise SystemExit(msg + "\n  Refusing to write a HEADLINE (shippable) map from a partial "
                                   "Stage B. Pass --allow-partial to proceed, or --headline to "
                                   "override deliberately; the per-variant maps are unaffected.")
        print(msg, flush=True)
    frame_lut = {pid: i for i, pid in enumerate(sorted(idx.PRODUCT_ID))}
    pd.DataFrame({"frame_idx": list(frame_lut.values()), "PRODUCT_ID": list(frame_lut)}
                 ).sort_values("frame_idx").to_csv(out_dir / "frame_lut.csv", index=False)

    reg, tile_rows, all_missing = [], [], set()
    for tile in tiles:
        ref = map_dir / f"{tile}_prob_raw.tif"
        if not ref.exists():
            ref = map_dir / f"{tile}_abundance.tif"
        if not ref.exists():
            print(f"  ⚠ {tile}: no reference raster in {map_dir} -> skip (Stage D writes on the "
                  f"mosaic-path grid; that map must be on disk)", flush=True)
            continue
        # The done-check must include the HEADLINE products, not just the per-variant ones: keying
        # only on `{tile}_{variant}_prob_raw.tif` meant a second run with --headline (or after the
        # verdict changed from AMBIGUOUS) skipped the tile and silently never wrote the plain-named
        # map, while still printing "headline variant = ..." (review 2026-07-29).
        need = [out_dir / f"{tile}_{v}_prob_raw.tif" for v in variants]
        if headline is not None:
            need.append(out_dir / f"{tile}_prob_raw.tif")
        done = all(p.exists() for p in need)
        if done and not args.overwrite:
            print(f"  {tile}: outputs exist -> skip (--overwrite to redo)", flush=True)
            continue
        grid = fc.tile_grid_from_raster(ref, tile)
        sel = idx[[fc.bbox_intersects_tile((r.TI_min, r.TI_max, r.TJ_min, r.TJ_max), grid)
                   for r in idx.itertuples()]]
        if sel.empty:
            print(f"  ⚠ {tile}: 0 frames reach this tile -> skip", flush=True)
            continue
        t0 = time.monotonic()
        labels = None if args.no_partition else seam_labels(tile, grid, frame_lut)
        accums, part, used, missing, n_px = compose_tile(tile, grid, sel, offsets, logits_dir,
                                                        frame_lut, variants, labels)
        all_missing.update(missing)
        stats = write_tile(tile, grid, accums, part, out_dir, cal_f, cal_mos, variants, headline,
                           meta_extra={"offsets_csv": str(Path(args.offsets).name),
                                       "trend_guard": guard, "n_frames_used": len(used),
                                       "frames_without_offset": missing})
        h = stats[headline or variants[0]]
        print(f"  {tile}: {len(used)} frames, coverage {h['coverage']:.1%}, "
              f"mean n_frames {h['mean_n_frames']:.2f} (max {h['max_n_frames']}), "
              f"P(rich) mean {h['prob_mean']:.4f}, overlap|Δp| med "
              f"{h['overlap_dp_median'] if h['overlap_dp_median'] is None else round(h['overlap_dp_median'], 4)}"
              f"  {time.monotonic() - t0:.0f}s", flush=True)
        reg.append({"tile": tile, "Kj": grid.Kj, "Ki": grid.Ki, "dx_m": round(grid.dx_m, 3),
                    "dy_m": round(grid.dy_m, 3), "tie_margin_m": round(grid.tie_margin_m, 4),
                    "height": grid.height, "width": grid.width, "n_frames": len(used)})
        for v, s in stats.items():
            tile_rows.append({"tile": tile, "variant": v, **s})

    if reg:
        rdf = pd.DataFrame(reg)
        rdf.to_csv(FIG / "fbuild_staged_registration.csv", index=False)
        pd.DataFrame(tile_rows).to_csv(FIG / "fbuild_staged_tiles.csv", index=False)
        print(f"\n=== registration vs the global 160 m lattice (exact integer shift per tile) ===")
        print(rdf[["tile", "Kj", "Ki", "dx_m", "dy_m", "tie_margin_m", "n_frames"]].to_string(index=False))
        print(f"\nworst sub-pixel translation: dx {rdf.dx_m.max():.1f} m, dy {rdf.dy_m.max():.1f} m "
              f"(cell 160 m; project registration budget is O(200 m) per CLAUDE.md)")
        tight = rdf[rdf.tie_margin_m < 1.0]
        if len(tight):
            print(f"⚠ {len(tight)} tile(s) sit <1 m from a half-cell rounding tie "
                  f"({', '.join(tight.tile)}) — their global-lattice column is a coin flip in the "
                  f"last float bits; the shift is applied as ONE deterministic integer per tile, so "
                  f"the map is self-consistent, but it may be one 160 m column off the mosaic map.")
    if all_missing:
        print(f"\n⚠ {len(all_missing)} frames had logits but NO Stage-C offset row -> composited "
              f"with o=0 and flagged offset_source='none' in the H6 layer: "
              f"{sorted(all_missing)[:8]}", flush=True)
    print(f"\nStage D composite done -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
