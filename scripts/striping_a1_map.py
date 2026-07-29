"""Generate the A1-fallback regional map — §5.1's missing comparison row.

PLAN_FBuild §5.1 compares the F build against the mosaic-path map AND the A1 fallback, but **no A1
raster exists on disk at any extent**: `scripts/striping_a1_infer_crop.py` computes A1 predictions in
memory for one E8_N44 crop and saves only a PNG. A1 renormalises raw CTX **DN** before the frozen
ViT, so there is no post-hoc path from the existing probability rasters — the A1 row has to be
inferred from scratch.

Footprint (Brian 2026-07-28): the **9 tiles with a locally cached Murray CTX mosaic zip**. A1 needs
raw DN, so those are the only tiles it can cover without ~30 GB of extra downloads, and §5.1's
"one common footprint" rule then makes them the footprint for every row.

This is `scripts/map_region.py`'s window sweep with two changes, both taken verbatim from the
reference A1 path (`scripts/striping_a1_infer_crop.py`):
  1. per-frame robust (median, IQR) computed at **160 m** from `read_ctx_on_grid`, keyed by the
     SeamMap partition labels — NOT from the native array (`src.striping.a1_normalize_per_frame`
     derives them differently, and the head was trained against the 160 m statistics);
  2. the native window DN is remapped per frame to (A1_REF_MEDIAN, A1_REF_IQR) = (125.0, 27.7),
     nodata (DN == 0) preserved, then inferred with the **A1 head** `models/deployable_a1`.

Everything else — grid, window offsets, tile_px, GeoTIFF profile, sidecar keys — is map_region's, so
the output is byte-grid-identical to `reports/map_region/` and drops straight into
`scripts/f_map_compare.py`.

Cost: a full map_region-equivalent GPU pass over the chosen tiles (~0.6 GPU-h/tile on an L40S at
batch 256; ~5-7 GPU-h for the 9). Resumable per (tile, window).

Run (GPU; laptop RTX 5070 or a Sherlock gpu node):
  conda run --no-capture-output -n geospatial python -u scripts/striping_a1_map.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import numpy as np
from rasterio.features import rasterize

from scripts.map_region import load_tile_sidecar, window_offsets
from src.calibration import CalibrationLayer
from src.fm_embeddings import FangEmbedder
from src.mapping import coarsened_transform, predict_window, read_tile_window, write_geotiff
from src.modeling.mlp_head import DeployableHead
from src.striping import (A1_REF_IQR, A1_REF_MEDIAN, CTX_ZIP_DIR, MAP_DIR, _inner_tif_name,
                          a1_stats, frame_label_map, load_frames, read_ctx_on_grid)

TILE_PX = 32
A1_HEAD = REPO / "models" / "deployable_a1" / "86c51a5dca220f63"
DEFAULT_OUT = REPO / "reports" / "map_a1"
EQUIPPED_FALLBACK = ["E-12_N36", "E-8_N32", "E0_N40", "E4_N40", "E4_N44",
                     "E8_N40", "E8_N44", "E12_N44", "E16_N44"]


def frame_stats_160(tile: str) -> tuple[dict, int]:
    """Per-frame robust (median, IQR) of the 160 m CTX brightness, indexed by load_frames order.

    This is the statistic the A1 head was trained against (striping_a1_infer_crop.py:56-64); deriving
    it from the native 5 m array instead gives different numbers and invalidates models/deployable_a1.
    """
    ctx160 = read_ctx_on_grid(tile, MAP_DIR / f"{tile}_abundance.tif")
    frames = load_frames(tile)
    labels160 = frame_label_map(tile, frames)
    stats = {}
    for i in range(len(frames)):
        sel = (labels160 == i) & np.isfinite(ctx160)
        if sel.sum() >= 50:
            med, iqr = a1_stats(np.where(sel, ctx160, 0))
            if np.isfinite(med) and np.isfinite(iqr) and iqr > 0:
                stats[i] = (med, iqr)
    return stats, len(frames)


def a1_window(window, frames, stats: dict):
    """Per-frame A1 remap of one native window; DN == 0 (mosaic nodata) preserved as 0."""
    labels_nat = rasterize(((g, i) for i, g in enumerate(frames.geometry)),
                           out_shape=window.data.shape, transform=window.transform,
                           fill=-1, dtype="int16", all_touched=False)
    arr = window.data.astype(np.float32)
    n_norm = 0
    for i, (med, iqr) in stats.items():
        sel = (labels_nat == i) & (window.data > 0)
        if sel.any():
            arr[sel] = np.clip((arr[sel] - med) / iqr * A1_REF_IQR + A1_REF_MEDIAN, 0, 255)
            n_norm += int(sel.sum())
    arr[window.data == 0] = 0
    return replace(window, data=arr.astype(np.uint8)), n_norm


def process_tile(tile: str, embedder, head, calibrator, args) -> dict:
    out_dir = Path(args.out_dir)
    if (out_dir / f"{tile}_prob_raw.tif").exists() and not args.force:
        print(f"[{tile}] exists -> skip (--force to redo)", flush=True)
        return {"tile": tile, "status": "skipped_done"}
    zip_path = CTX_ZIP_DIR / f"{tile}.zip"
    if not zip_path.exists():
        print(f"[{tile}] ⚠ no cached CTX zip ({zip_path}) — A1 needs raw DN, so this tile cannot be "
              f"rendered without fetching it (~1.7 GB)", flush=True)
        return {"tile": tile, "status": "no_ctx_zip"}
    side = load_tile_sidecar(tile)
    inner_transform, (H, W) = side["inner_transform"], side["inner_shape"]
    crs_wkt, inner_tif = side["inner_crs_wkt"], _inner_tif_name(zip_path)

    stats, n_frames = frame_stats_160(tile)
    frames = load_frames(tile)
    print(f"[{tile}] {H}x{W}px, {len(stats)}/{n_frames} frames with A1 stats", flush=True)
    if not stats:
        return {"tile": tile, "status": "no_frame_stats"}

    partial_dir = out_dir / "partials" / tile
    partial_dir.mkdir(parents=True, exist_ok=True)
    win, overlap = args.win_px, 2 * TILE_PX
    grid = [(r, c) for r in window_offsets(H, win, overlap, TILE_PX)
            for c in window_offsets(W, win, overlap, TILE_PX)]
    t_tile = time.monotonic()
    for k, (row_off, col_off) in enumerate(grid):
        part = partial_dir / f"{row_off:06d}_{col_off:06d}.npz"
        if part.exists() and not args.force:
            continue
        t0 = time.monotonic()
        window = read_tile_window(zip_path, inner_tif, row_off, col_off, win)
        w_a1, n_norm = a1_window(window, frames, stats)
        pred = predict_window(w_a1, embedder, head, tile_px=TILE_PX, batch=args.batch,
                              max_zero_fraction=args.max_zero_fraction, calibrator=calibrator,
                              apply_isotonic=not args.no_isotonic)
        keep = np.isfinite(pred.prob)
        cols = {"ti": pred.ti[keep].astype(np.int32), "tj": pred.tj[keep].astype(np.int32),
                "prob": pred.prob[keep].astype(np.float32)}
        if calibrator is not None:
            cols["prob_raw"] = pred.prob_raw[keep].astype(np.float32)
            cols["abundance"] = pred.abundance[keep].astype(np.float32)
        else:
            cols["prob_raw"] = pred.prob[keep].astype(np.float32)
        np.savez_compressed(part, **cols)
        if k % 10 == 0 or k == len(grid) - 1:
            print(f"[{tile}] win {k + 1}/{len(grid)} kept={int(keep.sum())} "
                  f"a1px={n_norm:,} {time.monotonic() - t0:.1f}s", flush=True)

    present = sorted(partial_dir.glob("*.npz"))
    if len(present) < len(grid):
        print(f"[{tile}] {len(present)}/{len(grid)} windows -> re-run to finish", flush=True)
        return {"tile": tile, "status": "partial", "windows_done": len(present),
                "windows_total": len(grid)}
    write_tile(tile, present, inner_transform, crs_wkt, calibrator, args)
    if args.clean_partials:
        for p in present:
            p.unlink()
    print(f"[{tile}] DONE in {time.monotonic() - t_tile:.0f}s", flush=True)
    return {"tile": tile, "status": "done", "windows": len(grid)}


def write_tile(tile, partials, inner_transform, crs_wkt, calibrator, args) -> None:
    """Assemble the per-window partials into map_region-shaped GeoTIFFs (same grid, same profile)."""
    def cat(key):
        return np.concatenate([np.load(p)[key] for p in partials])

    ti, tj = cat("ti").astype(np.int64), cat("tj").astype(np.int64)
    prob, prob_raw = cat("prob").astype(np.float64), cat("prob_raw").astype(np.float64)
    ab = cat("abundance").astype(np.float64) if calibrator is not None else None
    ti_min, tj_min = int(ti.min()), int(tj.min())
    shape = (int(ti.max()) - ti_min + 1, int(tj.max()) - tj_min + 1)
    transform = coarsened_transform(inner_transform, ti_min, tj_min, TILE_PX)
    out_dir = Path(args.out_dir)

    def scatter(v):
        r = np.full(shape, np.nan, dtype=np.float64)
        r[ti - ti_min, tj - tj_min] = v
        return r

    write_geotiff(out_dir / f"{tile}_prob_raw.tif", scatter(prob_raw), transform, crs_wkt)
    write_geotiff(out_dir / f"{tile}_prob.tif", scatter(prob), transform, crs_wkt)
    if ab is not None:
        write_geotiff(out_dir / f"{tile}_abundance.tif", scatter(ab), transform, crs_wkt)
    (out_dir / f"{tile}.json").write_text(json.dumps({
        "murray_tile": tile, "tile_px": TILE_PX, "raster_shape": list(shape),
        "ti_min": ti_min, "tj_min": tj_min, "n_predicted_tiles": int(ti.size),
        "calibrated": calibrator is not None,
        "isotonic": calibrator is not None and not args.no_isotonic,
        "prob_mean": float(np.nanmean(prob)), "rich_share_at_0p5": float((prob >= 0.5).mean()),
        "abundance_mean": float(np.nanmean(ab)) if ab is not None else None,
        "variant": "A1", "head": str(A1_HEAD.relative_to(REPO)),
        "a1_ref": {"median": A1_REF_MEDIAN, "iqr": A1_REF_IQR},
        "a1_stats_source": "read_ctx_on_grid at 160 m, SeamMap partition labels "
                           "(striping_a1_infer_crop.py convention)",
    }, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=None,
                    help="default = the 9 CTX-equipped block tiles (§5.1's common footprint)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--head", default=str(A1_HEAD))
    ap.add_argument("--calibration", default=None,
                    help="CalibrationLayer npz; omit for raw P(rich) only (η² is scored on raw P)")
    ap.add_argument("--win-px", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-zero-fraction", type=float, default=0.3)
    ap.add_argument("--no-isotonic", action="store_true")
    ap.add_argument("--clean-partials", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    tiles = args.tiles
    if not tiles:
        try:
            from src.striping import equipped_tiles
            tiles = equipped_tiles() or EQUIPPED_FALLBACK
        except Exception:                                    # noqa: BLE001
            tiles = EQUIPPED_FALLBACK
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    calibrator = CalibrationLayer.load(args.calibration) if args.calibration else None
    embedder = FangEmbedder.load(device="cpu" if args.cpu else None)
    head = DeployableHead.load(Path(args.head))
    print(f"A1 map: {len(tiles)} tiles {tiles}\n  head={Path(args.head).parent.name}, "
          f"A1 ref (median, IQR) = ({A1_REF_MEDIAN}, {A1_REF_IQR}), "
          f"calibration={'on' if calibrator else 'raw only'}", flush=True)

    results = []
    for tile in tiles:
        results.append(process_tile(tile, embedder, head, calibrator, args))
    (Path(args.out_dir) / "a1_manifest.json").write_text(
        json.dumps({"tiles": results, "head": str(args.head), "win_px": args.win_px,
                    "batch": args.batch, "a1_ref_median": A1_REF_MEDIAN,
                    "a1_ref_iqr": A1_REF_IQR}, indent=2), encoding="utf-8")
    done = sum(1 for r in results if r["status"] == "done")
    print(f"\nA1 map: {done}/{len(tiles)} tiles complete -> {args.out_dir}")
    missing = [r["tile"] for r in results if r["status"] == "no_ctx_zip"]
    if missing:
        print(f"⚠ {len(missing)} tile(s) had no cached CTX zip and were skipped: {missing}\n"
              f"  §5.1's common footprint is the tiles that DID render — record it on the table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
