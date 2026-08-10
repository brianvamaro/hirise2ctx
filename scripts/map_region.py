"""Regional (→ global) rock-abundance inference driver (PLAN_RegionalMap §4 / §4a).

Scales the validated one-window path (`scripts/map_pilot.py`) out to **whole Murray
Lab CTX tiles**, tile-list-driven and **resumable**, so it runs unchanged on a Sherlock
`gpu` node for the 7-tile circum-Chryse block now and the full Murray index later.

    for each Murray tile:
        sweep overlapping read windows across the 47420² px tile
            window -> FangEmbedder.embed_window -> DeployableHead.predict
            -> CalibrationLayer (isotonic P(rich) + qmatch abundance)
            -> append finite tiles to a per-window partial (.npz)        [checkpoint]
        assemble all partials -> per-tile GeoTIFFs (prob / abundance / prob_raw)

Resumability is at the **(tile, read-window) granularity**: each finished window writes
`partials/<tile>/<row>_<col>.npz`; a re-run skips windows whose partial exists and skips
tiles whose final GeoTIFF exists. A Slurm wall-clock limit or pre-emption therefore
resumes mid-tile with at most one window re-done.

Read windows overlap by `3*tile_px` because the embedder drops any tile whose 96-px (3×32)
context box spills the window edge; the overlap lets a neighbouring window supply that tile
with full context. The outermost one-tile ring of each Murray tile has no context and is
legitimately left nodata (a ~160 m seam).

**R01 — the coarse grid is global, not per-tile.** A Murray tile is 47,420 native px wide and
`gcd(47420, 32) = 4`, so anchoring the 32-px coarse lattice to each tile's own pixel origin
put every tile on its own sub-cell phase (8 distinct x-phases over the 26-tile footprint,
adjacent tiles 20 m apart). `rasterio.merge` floors each fractional offset, so that phase
became a whole-cell displacement in the shipped mosaic — 25 of 26 tiles, median 140 m.
`(ti, tj)` are therefore **global** cell indices on one planet-wide lattice anchored at
lon 0 / lat 0, and each tile's sweep is shifted by its own phase to land on it. Two
consequences worth knowing before reading any output:
  * every raster this driver has ever written is on the *old* lattice and must be re-rendered;
    products carry `grid_id` so the two can never be silently compared or merged;
  * map cells no longer coincide with Stage-4 *label* cells, which stay tile-anchored
    deliberately (re-anchoring them would force a relabel + retrain for no modelling gain),
    so any map↔label comparison must resample rather than index-match.

Output per Murray tile (single-band float32, 160 m/px, NaN = nodata/masked):
    <tile>_prob.tif       calibrated P(boulder-rich) in [0, 1]   (raw if --raw)
    <tile>_abundance.tif  fractional_area (qmatch)               (omitted with --raw)
    <tile>_prob_raw.tif   uncalibrated P(rich), QA               (omitted with --raw)

Usage (Sherlock gpu node, venv active):
    python scripts/map_region.py --all
    python scripts/map_region.py --tiles E4_N44 E8_N44
    python scripts/map_region.py --tiles E4_N44 --limit-windows 4   # throughput probe
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- OpenMP/DLL bootstrap; must precede numpy

import numpy as np

from src.mapping import (COARSE_GRID_ID, artifact_digest, assert_shared_lattice,
                         predict_window, read_tile_window, tile_global_grid, uncovered_cells,
                         window_offsets, write_geotiff)

CTX_TILES = REPO_ROOT / "cache_v2" / "ctx_tiles"
DEFAULT_MODEL_PARENT = REPO_ROOT / "models" / "deployable"
DEFAULT_CALIBRATION = DEFAULT_MODEL_PARENT / "calibration.npz"
DEFAULT_OUT = REPO_ROOT / "reports" / "map_region"
TILE_PX = 32  # frozen S=32 (160 m at 5 m/px)

# The circum-Chryse regional map (PLAN_RegionalMap §10 decision #5, box lon[-10,10] lat[32,46]
# snapped to whole 4-deg Murray tiles = the 24 box tiles, PLUS the 2 already-run tiles NE of the
# box (E12_N44, E16_N44) so the original block stays in-map). 26 tiles total. The first 7 were
# run 2026-06-17; the other 19 are the expansion. `--tiles` skips any whose final GeoTIFF exists,
# so re-running --all only computes what's missing.
BLOCK_TILES = [
    "E-12_N32", "E-12_N36", "E-12_N40", "E-12_N44",
    "E-8_N32", "E-8_N36", "E-8_N40", "E-8_N44",
    "E-4_N32", "E-4_N36", "E-4_N40", "E-4_N44",
    "E0_N32", "E0_N36", "E0_N40", "E0_N44",
    "E4_N32", "E4_N36", "E4_N40", "E4_N44",
    "E8_N32", "E8_N36", "E8_N40", "E8_N44",
    "E12_N44", "E16_N44",
]
# The 19 net-new tiles (everything in the box minus the 5 already-run box tiles + the 2 kept NE
# tiles); pass these to --tiles for the incremental expansion run so done tiles aren't recomputed.
EXPANSION_TILES = [
    "E-12_N32", "E-12_N36", "E-12_N40", "E-12_N44",
    "E-8_N32", "E-8_N36", "E-8_N40", "E-8_N44",
    "E-4_N32", "E-4_N36", "E-4_N40", "E-4_N44",
    "E0_N32", "E0_N36", "E0_N44",
    "E4_N32", "E4_N36", "E8_N32", "E8_N36",
]


def resolve_model_dir(arg: str | None, model_parent: str | Path | None = None) -> Path:
    """Resolve the deployable head: an explicit path, else the lexicographically last one.

    NOTE (audit, "Product semantics"): picking `hits[-1]` is choosing a head by *name*, not
    by compatibility with the calibrator or the preprocessing arm. That is a separate open
    finding; this function only makes the search root parameterizable so a scratch rebuild
    can resolve against its own `models/` tree.
    """
    if arg:
        return Path(arg)
    parent = Path(model_parent) if model_parent is not None else DEFAULT_MODEL_PARENT
    hits = sorted(p for p in parent.glob("*") if (p / "recipe.json").exists())
    if not hits:
        raise SystemExit(f"no deployable head under {parent}; "
                         "run scripts/train_deployable_head.py")
    return hits[-1]


def load_tile_sidecar(murray_tile: str, ctx_tiles: str | Path | None = None) -> dict:
    """Read a Murray tile's cached sidecar + zip.

    `ctx_tiles` is an argument so a scratch rebuild can point at an isolated tile cache
    (audit isolation criterion 4). It defaults to the live `cache_v2/ctx_tiles` because
    that directory is a read-only source archive here -- nothing in the map path writes it.
    """
    ctx_tiles = Path(ctx_tiles) if ctx_tiles is not None else CTX_TILES
    side_path = ctx_tiles / f"{murray_tile}.json"
    zip_path = ctx_tiles / f"{murray_tile}.zip"
    if not side_path.exists():
        raise SystemExit(f"tile sidecar missing: {side_path} "
                         "(re-fetch via ctx_retrieve.ensure_tile_cached)")
    if not zip_path.exists():
        raise SystemExit(f"tile zip missing: {zip_path} "
                         "(re-fetch via ctx_retrieve.ensure_tile_cached)")
    info = json.loads(side_path.read_text(encoding="utf-8"))
    info["_zip_path"] = zip_path
    return info


def partial_grid_id(path: str | Path) -> str | None:
    """`grid_id` recorded in a per-window partial, or None for a pre-R01 one.

    A MISSING key is a mismatch, not an error: every partial written before R01 part 2 lacks
    it, and those are exactly the ones that must not be mixed into a corrected product.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            if "grid_id" not in z.files:
                return None
            return str(z["grid_id"])
    except Exception:                                    # noqa: BLE001
        # Unreadable counts as foreign, never as an exception escaping the gate.
        # `np.savez_compressed` writes the zip in place with no tmp+rename, so a job killed
        # mid-save leaves a truncated `.npz`; raising `BadZipFile` here would make even
        # `--force` unable to clear it, which is strictly worse than the pre-R01 behaviour.
        return None


def reject_foreign_partials(partial_dir: Path, args) -> None:
    """Refuse to resume onto partials from another lattice — BEFORE any GPU time is spent.

    A resumed Sherlock run is the realistic way a corrected product silently reacquires the
    old lattice: `$SCRATCH` keeps per-window `.npz` files across jobs and the old ones carry
    tile-anchored `(ti, tj)`; assembling a mix scatters two lattices into one raster with no
    visible error. Note that R01 *also* moved the window offsets (`step` went 4032 → 4000
    once `overlap` became `3*tile_px`), so only the `000000_000000.npz` filename actually
    collides — 1 of 144. That narrows the exposure but does not remove it, and the surviving
    143 stale files would otherwise sit in the directory and defeat the
    `len(present) == len(grid)` completeness check at assembly.
    """
    stale = [p for p in sorted(partial_dir.glob("*.npz"))
             if partial_grid_id(p) != COARSE_GRID_ID]
    if not stale:
        return
    if args.force:
        print(f"  ⚠ --force: discarding {len(stale)} partial(s) from another lattice in "
              f"{partial_dir}", flush=True)
        for p in stale:
            p.unlink()
        return
    raise SystemExit(
        f"{len(stale)} of {len(list(partial_dir.glob('*.npz')))} partials in {partial_dir} "
        f"were written on a different coarse lattice (grid_id != {COARSE_GRID_ID}; pre-R01 "
        f"partials carry none). Assembling them would mix lattices silently.\n"
        f"  Re-run with --force to discard and recompute them, or delete the directory."
    )


def existing_product_off_lattice(prob_tif: Path) -> str | None:
    """Why an already-written per-tile raster is not on the current grid, else None.

    Checks the raster's own affine (the thing that is actually wrong on a pre-R01 product)
    and, when a sidecar is present, that its `grid_id` agrees. A missing sidecar or a missing
    `grid_id` is treated as pre-R01, because absence must never read as "checked and clean".
    """
    import rasterio

    try:
        with rasterio.open(prob_tif) as ds:
            transform = ds.transform
    except Exception as exc:                             # noqa: BLE001
        return f"unreadable ({type(exc).__name__})"
    try:
        assert_shared_lattice([transform], tile_px=TILE_PX)
    except ValueError as exc:
        return str(exc)
    side = prob_tif.parent / f"{prob_tif.name.replace('_prob.tif', '')}.json"
    if not side.exists():
        return "no sidecar, so its grid cannot be identified (pre-R01)"
    try:
        claim = json.loads(side.read_text(encoding="utf-8")).get("grid_id")
    except (ValueError, OSError) as exc:
        return f"unreadable sidecar ({type(exc).__name__})"
    if claim != COARSE_GRID_ID:
        return f"sidecar grid_id is {claim!r}, not {COARSE_GRID_ID!r}"
    return None


def as_int32_cells(v: np.ndarray, name: str, tile: str) -> np.ndarray:
    """Narrow global cell indices to int32, refusing to wrap silently."""
    if v.size and (int(v.min()) < np.iinfo(np.int32).min
                   or int(v.max()) > np.iinfo(np.int32).max):
        raise SystemExit(f"[{tile}] global {name} out of int32 range "
                         f"[{int(v.min())}, {int(v.max())}]")
    return v.astype(np.int32)


def map_one_tile(murray_tile: str, embedder, head, calibrator, *, args) -> dict:
    """Sweep one Murray tile and write its GeoTIFFs. Returns a status dict."""
    info = load_tile_sidecar(murray_tile, getattr(args, "ctx_tiles", None))
    zip_path = info["_zip_path"]
    inner_tif = info["inner_tif"]
    inner_transform = tuple(info["inner_transform"])
    crs_wkt = info.get("inner_crs_wkt", "")
    H, W = info["inner_shape"]

    out_dir = Path(args.out_dir)
    prob_tif = out_dir / f"{murray_tile}_prob.tif"
    if prob_tif.exists() and not args.force:
        # R01: existence is not completeness once the lattice has changed. Every tile of the
        # pre-R01 product is present on disk, so a bare existence check would skip all 26,
        # write a manifest stamped with the NEW grid_id, and report "26/26 tiles complete" --
        # a rebuild that rendered nothing and then certified itself. Check the lattice.
        why = existing_product_off_lattice(prob_tif)
        if why:
            raise SystemExit(
                f"[{murray_tile}] {prob_tif} exists but is NOT on {COARSE_GRID_ID}: {why}.\n"
                f"  This is the pre-R01 product. Re-run with --force to re-render it, or "
                f"point --out-dir at a fresh directory. Refusing to skip it as 'done'."
            )
        print(f"[{murray_tile}] final GeoTIFF exists -> skip (use --force to redo)", flush=True)
        return {"tile": murray_tile, "status": "skipped_done"}

    # R01: place this tile on the ONE global coarse lattice. Constructing the grid is what
    # verifies it -- the sphere radius is parsed from this tile's own CRS and the origin is
    # checked against the native lattice, so nothing downstream can stamp COARSE_GRID_ID on
    # a product that was not actually rendered on it.
    grid_geom = tile_global_grid(inner_transform, crs_wkt, TILE_PX)

    partial_dir = out_dir / "partials" / murray_tile
    partial_dir.mkdir(parents=True, exist_ok=True)
    reject_foreign_partials(partial_dir, args)

    # overlap = 3*TILE_PX and a non-tile-aligned final offset are BOTH required once the
    # lattice has a phase; either alone still leaves holes (see `window_offsets`). Free:
    # 12 offsets per axis in every variant, so the window count is unchanged.
    win, overlap = args.win_px, 3 * TILE_PX
    row_offs = window_offsets(H, win, overlap, TILE_PX, tile_aligned=False)
    col_offs = window_offsets(W, win, overlap, TILE_PX, tile_aligned=False)
    miss_r = uncovered_cells(row_offs, H, win, TILE_PX, phase=grid_geom.phase_c)
    miss_c = uncovered_cells(col_offs, W, win, TILE_PX, phase=grid_geom.phase_r)
    if miss_r or miss_c:
        raise SystemExit(
            f"[{murray_tile}] sweep would leave {len(miss_r)} row / {len(miss_c)} col cells "
            f"uncomputable at phase ({grid_geom.phase_r}, {grid_geom.phase_c}); "
            f"first row hole at px {miss_r[:1]}, col {miss_c[:1]}. Refusing to render a "
            f"product with holes in it."
        )
    grid = [(r, c) for r in row_offs for c in col_offs]
    print(f"[{murray_tile}] {H}x{W}px  win={win} overlap={overlap}  "
          f"{len(row_offs)}x{len(col_offs)}={len(grid)} windows  "
          f"phase=({grid_geom.phase_r},{grid_geom.phase_c}) "
          f"cell0=({grid_geom.cell_row0},{grid_geom.cell_col0})", flush=True)

    done_tiles = 0
    t_tile = time.monotonic()
    for k, (row_off, col_off) in enumerate(grid):
        part_path = partial_dir / f"{row_off:06d}_{col_off:06d}.npz"
        if part_path.exists() and not args.force:
            continue
        if args.limit_windows is not None and done_tiles >= args.limit_windows:
            print(f"[{murray_tile}] --limit-windows {args.limit_windows} reached", flush=True)
            break

        t0 = time.monotonic()
        window = read_tile_window(zip_path, inner_tif, row_off, col_off, win)
        pred = predict_window(window, embedder, head, tile_px=TILE_PX,
                              batch=args.batch, max_zero_fraction=args.max_zero_fraction,
                              calibrator=calibrator, apply_isotonic=not args.no_isotonic,
                              global_grid=grid_geom.as_tuple)
        keep = np.isfinite(pred.prob)
        cols = {
            # ti/tj are GLOBAL cell indices now. int32 is ample -- at S=32 the whole planet
            # spans row +-90*11855/32 = +-33,342 and col +-180*11855/32 = +-66,684 cells --
            # but the cast is where a future tile_px or ppd change would silently wrap, so
            # range-check rather than assume.
            "ti": as_int32_cells(pred.ti[keep], "ti", murray_tile),
            "tj": as_int32_cells(pred.tj[keep], "tj", murray_tile),
            "prob": pred.prob[keep].astype(np.float32),
            "grid_id": np.array(COARSE_GRID_ID),
        }
        if calibrator is not None:
            cols["prob_raw"] = pred.prob_raw[keep].astype(np.float32)
            cols["abundance"] = pred.abundance[keep].astype(np.float32)
        np.savez_compressed(part_path, **cols)
        done_tiles += 1
        dt = time.monotonic() - t0
        rate = int(keep.sum() / dt) if dt > 0 else 0
        print(f"[{murray_tile}] win {k + 1}/{len(grid)} off=({row_off},{col_off}) "
              f"kept={int(keep.sum())} {dt:.1f}s ~{rate} tiles/s", flush=True)

    # Assemble: need every window's partial present before writing the final raster.
    present = sorted(partial_dir.glob("*.npz"))
    if len(present) < len(grid) and args.limit_windows is None:
        print(f"[{murray_tile}] {len(present)}/{len(grid)} windows done -> "
              "not yet assembling (re-run to finish)", flush=True)
        return {"tile": murray_tile, "status": "partial",
                "windows_done": len(present), "windows_total": len(grid)}
    if args.limit_windows is not None:
        print(f"[{murray_tile}] benchmark mode (--limit-windows) -> skip assembly", flush=True)
        return {"tile": murray_tile, "status": "benchmark", "windows_done": done_tiles,
                "elapsed_s": round(time.monotonic() - t_tile, 1)}

    write_tile_geotiffs(murray_tile, present, grid_geom, crs_wkt, calibrator, args)
    if args.clean_partials:
        for p in present:
            p.unlink()
        try:
            partial_dir.rmdir()
        except OSError:
            pass
    print(f"[{murray_tile}] DONE in {time.monotonic() - t_tile:.0f}s", flush=True)
    return {"tile": murray_tile, "status": "done", "windows": len(grid)}


def write_tile_geotiffs(murray_tile, partials, grid_geom, crs_wkt, calibrator, args):
    """Scatter all per-window partials into the per-tile rasters and write GeoTIFFs.

    `(ti, tj)` are GLOBAL coarse-cell indices (R01), so the affine comes from the global
    lattice rather than from this tile's own origin. Those two go together: keeping the
    parent-tile affine while the indices are global multiplies a ~-16,000 index against the
    tile origin and lands the raster ~2,600 km away.
    """
    foreign = [str(p) for p in partials if partial_grid_id(p) != COARSE_GRID_ID]
    if foreign:
        raise SystemExit(
            f"[{murray_tile}] refusing to assemble {len(foreign)} partial(s) from another "
            f"lattice: {foreign[:3]}{' ...' if len(foreign) > 3 else ''}")
    ti = np.concatenate([np.load(p)["ti"] for p in partials]).astype(np.int64)
    tj = np.concatenate([np.load(p)["tj"] for p in partials]).astype(np.int64)
    prob = np.concatenate([np.load(p)["prob"] for p in partials]).astype(np.float64)
    has_cal = calibrator is not None
    prob_raw = (np.concatenate([np.load(p)["prob_raw"] for p in partials]).astype(np.float64)
                if has_cal else None)
    abundance = (np.concatenate([np.load(p)["abundance"] for p in partials]).astype(np.float64)
                 if has_cal else None)

    ti_min, ti_max = int(ti.min()), int(ti.max())
    tj_min, tj_max = int(tj.min()), int(tj.max())
    shape = (ti_max - ti_min + 1, tj_max - tj_min + 1)
    transform = grid_geom.transform(ti_min, tj_min)
    out_dir = Path(args.out_dir)

    def scatter(values):
        r = np.full(shape, np.nan, dtype=np.float64)
        r[ti - ti_min, tj - tj_min] = values  # overlap re-writes identical values
        return r

    write_geotiff(out_dir / f"{murray_tile}_prob.tif", scatter(prob), transform, crs_wkt)
    if has_cal:
        write_geotiff(out_dir / f"{murray_tile}_abundance.tif", scatter(abundance), transform, crs_wkt)
        write_geotiff(out_dir / f"{murray_tile}_prob_raw.tif", scatter(prob_raw), transform, crs_wkt)
    n_tiles = ti.size
    (out_dir / f"{murray_tile}.json").write_text(json.dumps({
        "murray_tile": murray_tile, "tile_px": TILE_PX, "raster_shape": list(shape),
        # R01: ti_min/tj_min are GLOBAL cell indices, not tile-local. Two products are
        # provably co-registered iff grid_id and grid_cell_m match and their (ti_min, tj_min)
        # differ by integers -- which is checkable, unlike the old bare tile-local pair.
        **grid_geom.provenance(),
        "ti_min": ti_min, "tj_min": tj_min, "n_predicted_tiles": int(n_tiles),
        "calibrated": has_cal, "isotonic": has_cal and not args.no_isotonic,
        "prob_mean": float(np.nanmean(prob)),
        "rich_share_at_0p5": float((prob >= 0.5).mean()),
        "abundance_mean": float(np.nanmean(abundance)) if has_cal else None,
        # Which head and calibrator produced this raster. The baseline tile sidecar used to
        # record neither -- only `calibrated: true/false` -- so a tile could not be traced
        # to the artifacts that made it, and two tiles rendered from different heads were
        # indistinguishable. Digests rather than paths, because a path can be overwritten
        # in place (audit, "Product semantics and provenance").
        "head": str(getattr(args, "_model_dir", "")) or None,
        "head_digest": artifact_digest(getattr(args, "_model_dir", "")) if getattr(args, "_model_dir", None) else None,
        "calibration": str(args.calibration) if has_cal else None,
        "calibration_digest": artifact_digest(args.calibration) if has_cal else None,
    }, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tiles", nargs="+", help="Murray tile ids, e.g. E4_N44 E8_N44")
    g.add_argument("--all", action="store_true",
                   help="the full 26-tile circum-Chryse regional map (BLOCK_TILES)")
    g.add_argument("--expansion", action="store_true",
                   help="only the 19 net-new expansion tiles (EXPANSION_TILES); skips the "
                        "7 already-run tiles regardless of $SCRATCH state")
    ap.add_argument("--win-px", type=int, default=4096, help="read-window side in CTX px")
    ap.add_argument("--batch", type=int, default=96,
                    help="embedder batch size. Default 96 matches the parity reference. The ViT is "
                         "per-sample so larger batches (e.g. 256) better saturate an L40S/A100 and "
                         "are ~parity-safe; if you bump it, re-emit the parity reference at the same "
                         "--batch (fp16 GEMM kernels can vary slightly by batch).")
    ap.add_argument("--max-zero-fraction", type=float, default=0.3,
                    help="mask a tile whose own CTX is more than this share mosaic nodata")
    ap.add_argument("--model", default=None, help="deployable head dir (default: latest)")
    # Isolation criterion 4: every artifact root the driver reads or searches is a flag, so
    # a scratch rebuild never has to touch the live tree.
    ap.add_argument("--ctx-tiles", default=str(CTX_TILES),
                    help="directory of Murray tile zips + sidecars")
    ap.add_argument("--model-parent", default=str(DEFAULT_MODEL_PARENT),
                    help="where --model is searched when it is not given explicitly")
    ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION),
                    help="banked CalibrationLayer .npz")
    ap.add_argument("--raw", action="store_true",
                    help="render RAW P(rich) only (skip the Stage-1 CalibrationLayer)")
    ap.add_argument("--no-isotonic", action="store_true",
                    help="skip the Tier-1 isotonic polish (abundance qmatch still applied)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                    help="output dir (point at $SCRATCH on Sherlock)")
    ap.add_argument("--limit-windows", type=int, default=None,
                    help="process at most N windows per tile then stop (throughput probe)")
    ap.add_argument("--force", action="store_true", help="recompute existing windows/tiles")
    ap.add_argument("--clean-partials", action="store_true",
                    help="delete per-window .npz after a tile's GeoTIFFs are written")
    args = ap.parse_args()

    tiles = BLOCK_TILES if args.all else (EXPANSION_TILES if args.expansion else args.tiles)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    model_dir = resolve_model_dir(args.model, args.model_parent)
    # Threaded onto args so `map_one_tile` can record it in each tile sidecar without
    # another parameter through the call chain.
    args._model_dir = model_dir
    card = json.loads((model_dir / "recipe.json").read_text(encoding="utf-8"))
    print(f"=== map_region: {len(tiles)} tile(s)  model={model_dir.name}  "
          f"recipe={card['recipe'].get('cell')}  out={args.out_dir} ===", flush=True)

    from src.fm_embeddings import FangEmbedder
    from src.modeling.mlp_head import DeployableHead

    calibrator = None
    if not args.raw:
        from src.calibration import CalibrationLayer
        calibrator = CalibrationLayer.load(args.calibration)
        print(f"  calibration={Path(args.calibration).name}  "
              f"isotonic={'off' if args.no_isotonic else 'on'}  abundance=qmatch(P(rich))",
              flush=True)

    embedder = FangEmbedder.load()
    head = DeployableHead.load(model_dir)
    dev = getattr(getattr(embedder, "device", None), "type", "?")
    print(f"  embedder device={dev}  head seeds={card.get('n_seeds', '?')}", flush=True)

    results = []
    t0 = time.monotonic()
    for tile in tiles:
        results.append(map_one_tile(tile, embedder, head, calibrator, args=args))

    manifest = Path(args.out_dir) / "region_manifest.json"
    manifest.write_text(json.dumps({
        # `relative_to(REPO_ROOT)` raised for any head outside the repo, which is exactly
        # what a scratch rebuild uses. Record the absolute path plus a content digest.
        "tiles": tiles, "model_dir": str(model_dir),
        "head_digest": artifact_digest(model_dir),
        "calibration": str(args.calibration) if calibrator is not None else None,
        "calibration_digest": (
            artifact_digest(args.calibration) if calibrator is not None else None),
        "ctx_tiles": str(args.ctx_tiles),
        "grid_id": COARSE_GRID_ID,
        "recipe_hash": card.get("recipe_hash"), "win_px": args.win_px,
        "calibrated": calibrator is not None, "raw": args.raw,
        "elapsed_s": round(time.monotonic() - t0, 1), "results": results,
    }, indent=2), encoding="utf-8")
    n_done = sum(r["status"] == "done" for r in results)
    print(f"=== {n_done}/{len(tiles)} tiles complete  "
          f"{time.monotonic() - t0:.0f}s  manifest -> {manifest} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
