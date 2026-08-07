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

Read windows overlap by `2*tile_px` (one context ring on each side) because the embedder
drops any tile whose 96-px (3×32) context box spills the window edge; the overlap lets a
neighbouring window supply that tile with full context. Tiles are anchored to the parent
tile's pixel origin (CLAUDE.md Stage 4), so global `(ti, tj)` are consistent across
windows and overlap just re-writes identical values. The outermost one-tile ring of each
Murray tile has no context and is legitimately left nodata (a ~160 m seam; cross-tile
stitching is a laptop-side validation concern, not this driver's).

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

from src.mapping import (artifact_digest, coarsened_transform, predict_window,
                         read_tile_window, write_geotiff)

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


def window_offsets(extent: int, win: int, overlap: int, tile_px: int) -> list[int]:
    """Tile-aligned read-window start offsets covering [0, extent) with `overlap`.

    Offsets are multiples of `tile_px` (so global tile indices stay aligned). `step =
    win - overlap` keeps consecutive gaps <= step, so every interior tile sits with its
    full 3x3 context box inside at least one window. The final offset is the tile-aligned
    `extent - win` (its window reaches within one tile of the tile edge; the outermost
    context-less ring is left nodata by design).
    """
    win = min(win, extent)
    step = max(tile_px, win - overlap)
    last = ((extent - win) // tile_px) * tile_px  # tile-aligned final start
    offs: list[int] = []
    o = 0
    while o < last:
        offs.append(o)
        o += step
    if not offs or offs[-1] != last:
        offs.append(last)
    return offs


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
        print(f"[{murray_tile}] final GeoTIFF exists -> skip (use --force to redo)", flush=True)
        return {"tile": murray_tile, "status": "skipped_done"}

    partial_dir = out_dir / "partials" / murray_tile
    partial_dir.mkdir(parents=True, exist_ok=True)

    win, overlap = args.win_px, 2 * TILE_PX
    row_offs = window_offsets(H, win, overlap, TILE_PX)
    col_offs = window_offsets(W, win, overlap, TILE_PX)
    grid = [(r, c) for r in row_offs for c in col_offs]
    print(f"[{murray_tile}] {H}x{W}px  win={win} overlap={overlap}  "
          f"{len(row_offs)}x{len(col_offs)}={len(grid)} windows", flush=True)

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
                              calibrator=calibrator, apply_isotonic=not args.no_isotonic)
        keep = np.isfinite(pred.prob)
        cols = {
            "ti": pred.ti[keep].astype(np.int32),
            "tj": pred.tj[keep].astype(np.int32),
            "prob": pred.prob[keep].astype(np.float32),
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

    write_tile_geotiffs(murray_tile, present, inner_transform, crs_wkt, calibrator, args)
    if args.clean_partials:
        for p in present:
            p.unlink()
        try:
            partial_dir.rmdir()
        except OSError:
            pass
    print(f"[{murray_tile}] DONE in {time.monotonic() - t_tile:.0f}s", flush=True)
    return {"tile": murray_tile, "status": "done", "windows": len(grid)}


def write_tile_geotiffs(murray_tile, partials, inner_transform, crs_wkt, calibrator, args):
    """Scatter all per-window partials into the per-tile rasters and write GeoTIFFs."""
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
    transform = coarsened_transform(inner_transform, ti_min, tj_min, TILE_PX)
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
