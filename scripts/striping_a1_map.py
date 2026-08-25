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
  1. per-frame robust (median, IQR) of the **native 5 m** CTX DN, over each frame's extent in the
     whole Murray tile, keyed by the SeamMap partition labels — the SAME statistic training uses
     (R07; see `frame_stats_native`). This corrects the previous 160 m statistic, and the two
     docstrings that asserted the exact inverse;
  2. the native window DN is remapped per frame to (A1_REF_MEDIAN, A1_REF_IQR) = (125.0, 27.7),
     nodata (DN == 0) preserved, then inferred with the **A1 head** `models/deployable_a1`.

**R38 — DN 0 in an A1 array now means nodata and nothing else.** The remap used to clip to
`[0, 255]`, so terrain darker than about `med - 4.51*iqr` was written as the mosaic nodata
sentinel and thereafter counted as a data gap: 0.64 % of training tiles and 6.7 % of deploy-sim
tiles carried at least one such false-black pixel while still passing the mask, and whole tiles
went black in low-IQR frames. Three things changed together, and the third is the one that lasts:
valid pixels floor at `src.striping.A1_VALID_FLOOR`; this driver hands `predict_window` an
**explicit nodata mask taken from the raw DN** rather than letting it infer one from the
normalised array; and the texture the clip destroys is recorded separately as `a1_clip_*` in each
tile sidecar (`--warn-clip-fraction`). Moving the floor alone would have been worse than useless —
DN 0 and DN 1 damage the frozen embedding identically to three decimals, so the pixels would have
become invisible rather than safe. Brian's call 2026-08-10: record the loss, do **not** retune the
transfer function. That is also what lets `--max-context-zero-fraction` default to 0.0 here, as it
does on the baseline arm.

Everything else — grid, window offsets, tile_px, GeoTIFF profile, sidecar keys — is map_region's, so
the output is byte-grid-identical to `reports/map_region/` and drops straight into
`scripts/f_map_compare.py`.

**R01 — one lattice.** Both drivers render onto the globally anchored coarse lattice
(`src.mapping.COARSE_GRID_ID`), wired in the same commit precisely so A1 can never land on a
different lattice than the baseline it is compared against.

**R01's ordering constraint is GONE, because R07 removed it.** It existed only because the A1
statistic used to be read off `reports/map_region/{tile}_abundance.tif`'s grid, which forced the
corrected baseline to be rendered first and made A1's normalisation sensitive to the re-anchoring
(measured: >1 DN on 11 of 74 frames, up to 9 DN on the smallest). `frame_stats_native` derives the
statistic from the native tile instead, so A1 depends on the baseline product for nothing but the
CRS that `load_frames` reads, and the two rows can be built in either order.

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
import traceback
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy

import numpy as np
from rasterio.features import rasterize

from scripts.map_region import (DEFAULT_SIZE_FLOOR_BASIS, as_int32_cells, check_overlap,
                                compute_device_name, gate_cols, gate_summary,
                                load_tile_sidecar, project_tile_cost, require_device,
                                partial_grid_id, partial_name, read_partial,
                                reject_foreign_partials, size_floor_tags, sweep_manifest,
                                run_tile_isolated, sweep_mismatch, tile_is_reusable,
                                window_offsets,
                                write_json_atomic)
from src.calibration import CalibrationLayer
from src.fm_embeddings import FangEmbedder
from src.mapping import (COARSE_GRID_ID, artifact_digest, assert_shared_lattice, file_sha256,
                         open_tile, predict_window, read_tile_window, tile_global_grid,
                         uncovered_cells,
                         write_geotiff)
from src.modeling.mlp_head import DeployableHead, require_norm_arm
from src.striping import (A1_ARM, A1_MIN_FRAME_PX, A1_REF_IQR, A1_REF_MEDIAN, A1_VALID_FLOOR,
                          CTX_ZIP_DIR, _inner_tif_name, a1_normalize_native,
                          a1_stats_native_tile, find_seam_shp, frame_bounds, frame_labels_on,
                          load_frames)

TILE_PX = 32
A1_HEAD = REPO / "models" / "deployable_a1" / "86c51a5dca220f63"
DEFAULT_OUT = REPO / "reports" / "map_a1"
EQUIPPED_FALLBACK = ["E-12_N36", "E-8_N32", "E0_N40", "E4_N40", "E4_N44",
                     "E8_N40", "E8_N44", "E12_N44", "E16_N44"]


def frame_stats_native(tile: str, frames) -> tuple[dict, tuple, dict]:
    """**R07.** Per-frame robust (median, IQR) of the **native 5 m** CTX DN, over each frame's
    extent in the whole Murray tile — the one statistic training now uses too.

    Replaces `frame_stats_160`, which derived the statistic from CTX area-averaged to 160 m and
    then applied that gain to native DN. Measured over all 39 Stage-2 windows, that inflated the
    gain by a median **1.35x** (p95 1.83x, max 2.15x): training pins the input IQR to exactly
    27.7, and the 160 m path delivered a median of 37.3 and a max of 59.6 while clipping ~10x
    more pixels. See DECISIONS 2026-08-09a.

    Two consequences worth knowing. (1) This costs one streamed pass over the native tile
    (~2.2 Gpx, I/O-bound) instead of one 160 m resample — cheap next to the ~0.6 GPU-h/tile of
    inference it precedes, and exact, because uint8 percentiles come from a 256-bin histogram.
    (2) **The R01 ordering constraint is gone**: the statistic no longer comes off
    `reports/map_region/{tile}_abundance.tif`, so A1 no longer has to follow the corrected
    baseline. `load_frames` still opens that raster, but only to read its CRS.
    """
    return a1_stats_native_tile(tile, frames)


def a1_window(window, frames, stats: dict, fallback: tuple[float, float], *,
              frames_bounds=None):
    """Per-frame A1 remap of one native window. Returns `(window', nodata_mask, clip_report)`.

    **R07/R08.** This used to leave any pixel outside a qualifying frame at **raw DN**, mixing
    two radiometric scales in one array and handing the mixture to a frozen embedder that
    cannot tell them apart. `a1_normalize_native` normalizes those by the tile-wide native
    statistic instead, and refuses rather than falling back to raw.

    **R38.** The nodata mask is taken from the **raw** window, before normalization, and handed
    to `predict_window` explicitly. Deriving it from the A1 output was the defect: the clip
    floored legitimately dark terrain onto the sentinel, so those pixels were counted as a data
    gap. Valid pixels now floor at `A1_VALID_FLOOR = 1`, which makes DN 0 unambiguous — but the
    mask is still passed rather than re-inferred, because "the output happens to be safe to
    infer from" is a property that quietly stops holding the next time the transfer function
    changes.

    What the clip *destroys* is counted elsewhere and once: `a1_stats_native_tile` derives it
    exactly from the per-frame DN histogram it already builds. Accumulating it here instead
    would have been both approximate and resume-dependent — read windows overlap by 96 px, so
    per-window pixel counts double-count the seams, and a resumed run only sees the windows it
    recomputed.
    """
    labels_nat = frame_labels_on(window.transform, window.data.shape, frames, dtype="int32",
                                bounds=frames_bounds)
    nodata_mask = window.data == 0                # from the RAW DN, before any normalization
    out = a1_normalize_native(window.data, labels_nat, stats, fallback)
    return replace(window, data=out), nodata_mask


def seammap_digest(tile: str) -> str | None:
    """Content digest of the SeamMap that defines this tile's A1 frames, or None if absent.

    Every sibling of the `.shp` is hashed, not just the `.shp` itself — the **`.prj` is
    load-bearing** here. The frames are reprojected into the tile CRS before rasterization, so a
    changed projection silently changes which pixels belong to which frame, and therefore every
    per-frame statistic, without touching a single coordinate in the `.shp`.
    """
    shp = find_seam_shp(tile)
    if shp is None or not Path(shp).exists():
        return None
    import hashlib

    h = hashlib.sha256()
    shp = Path(shp)
    for p in sorted(shp.parent.glob(shp.stem + ".*")):
        h.update(p.name.encode("utf-8"))
        h.update(file_sha256(p).encode("ascii"))
    return h.hexdigest()


def a1_sweep_manifest(grid_geom, row_offs, col_offs, args, *, extent, tile,
                      head_digest, calibration_digest, device=None) -> dict:
    """The identity of an A1 sweep — `map_region`'s, plus what makes A1 *A1*.

    **R14 on the A1 arm, which never had it.** This driver deleted a `partials/<tile>/_sweep.json`
    in its `--clean-partials` path but never wrote one and never called `sweep_manifest`, so
    R14's protection covered `map_region.py` only: a resumed A1 run could mix partials from a
    different `--win-px`, head, calibrator or masking threshold behind nothing but the `grid_id`
    lattice check. Every field below is a resume-match field, and a mismatch names itself.

    The A1-specific fields matter as much as the shared ones, because A1's input is *derived*:
    two runs can agree on window geometry and head and still have normalised the DN differently.
    `norm_arm` pins R07's statistic definition, `a1_ref` / `a1_clip_floor` pin the transfer
    function (R38 moved the floor, which changes pixels), `a1_min_frame_px` pins R08's ratified
    fallback boundary, and `a1_seammap_digest` pins the frame partition itself.

    Deliberately digests the *inputs* rather than the resulting per-frame statistics: those cost
    a ~3 min streaming pass per tile, and hashing them would force that pass before the driver
    could decide whether a tile is already done.
    """
    return {
        **sweep_manifest(grid_geom, row_offs, col_offs, args, extent=extent,
                         head_digest=head_digest, calibration_digest=calibration_digest,
                         device=device),
        "variant": "A1",
        "norm_arm": A1_ARM,
        "a1_ref": [A1_REF_MEDIAN, A1_REF_IQR],
        "a1_clip_floor": A1_VALID_FLOOR,
        "a1_min_frame_px": A1_MIN_FRAME_PX,
        "a1_seammap_digest": seammap_digest(tile),
    }


def process_tile(tile: str, embedder, head, calibrator, args) -> dict:
    out_dir = Path(args.out_dir)
    zip_path = CTX_ZIP_DIR / f"{tile}.zip"
    if not zip_path.exists():
        print(f"[{tile}] ⚠ no cached CTX zip ({zip_path}) — A1 needs raw DN, so this tile cannot be "
              f"rendered without fetching it (~1.7 GB)", flush=True)
        return {"tile": tile, "status": "no_ctx_zip"}
    side = load_tile_sidecar(tile)
    inner_transform, (H, W) = side["inner_transform"], side["inner_shape"]
    crs_wkt, inner_tif = side["inner_crs_wkt"], _inner_tif_name(zip_path)

    # R01: identical grid vocabulary to map_region -- same lattice, same phase, same
    # window sweep. If these two drivers ever diverge, A1 stops being comparable to the
    # baseline, which is the entire point of the row.
    grid_geom = tile_global_grid(inner_transform, crs_wkt, TILE_PX)
    win, overlap = args.win_px, 3 * TILE_PX
    row_offs = window_offsets(H, win, overlap, TILE_PX, tile_aligned=False)
    col_offs = window_offsets(W, win, overlap, TILE_PX, tile_aligned=False)
    miss_r = uncovered_cells(row_offs, H, win, TILE_PX, phase=grid_geom.phase_r)
    miss_c = uncovered_cells(col_offs, W, win, TILE_PX, phase=grid_geom.phase_c)
    if miss_r or miss_c:
        raise SystemExit(f"[{tile}] sweep would leave {len(miss_r)} row / {len(miss_c)} col "
                         f"cells uncomputable at phase "
                         f"({grid_geom.phase_r}, {grid_geom.phase_c})")

    # R14: pin the sweep BEFORE any partial is written, and before the ~3 min statistics pass --
    # everything here is cheap metadata, so an already-committed tile is skipped without paying
    # for a streaming read it does not need. (`a1_sweep_manifest` digests A1's *inputs* rather
    # than the derived per-frame statistics precisely so this ordering is possible.)
    want_sweep = a1_sweep_manifest(
        grid_geom, row_offs, col_offs, args, extent=(H, W), tile=tile,
        head_digest=artifact_digest(args.head),
        calibration_digest=artifact_digest(args.calibration) if args.calibration else None,
        device=compute_device_name(embedder))

    # R14: resume on the SIDECAR, not on the first artifact written. This driver's sentinel was
    # `{tile}_prob_raw.tif`, which `write_tile` emits FIRST -- the identical defect map_region
    # had, on the one product that has never been generated, so fixing it here costs nothing.
    # `want_sweep` (not None) is new: content alone cannot see a raster that is structurally
    # perfect and was normalised by a different A1 arm.
    if not args.force:
        why = tile_is_reusable(out_dir, tile, want_sweep,
                               verify=getattr(args, "verify_existing", False),
                               trust_existing=getattr(args, "trust_existing", False))
        if why is None and (out_dir / f"{tile}_prob_raw.tif").exists():
            print(f"[{tile}] committed and verified -> skip (--force to redo)", flush=True)
            return {"tile": tile, "status": "skipped_done"}
        if (out_dir / f"{tile}_prob_raw.tif").exists():
            raise SystemExit(f"[{tile}] {out_dir / f'{tile}_prob_raw.tif'} exists but cannot "
                             f"be reused: {why}\n  Re-run with --force, or use a fresh "
                             f"--out-dir.")

    frames = load_frames(tile)
    print(f"[{tile}] streaming the native tile for per-frame A1 statistics (R07) ...",
          flush=True)
    t_stats = time.monotonic()
    stats, fallback, a1_prov = frame_stats_native(tile, frames)
    t_stats = time.monotonic() - t_stats
    print(f"[{tile}] {H}x{W}px, {a1_prov['n_frames_with_stats']}/{a1_prov['n_frames']} frames "
          f"with A1 stats ({a1_prov['n_frames_too_small']} too small), fallback covers "
          f"{a1_prov['fallback_pixel_fraction']:.3%} of valid px, "
          f"phase=({grid_geom.phase_r},{grid_geom.phase_c})", flush=True)
    # R38: say out loud how much texture the clip destroys. It is not a data gap and no nodata
    # count will ever show it, so if it is not printed and recorded here it is invisible.
    cf = a1_prov.get("clip_fraction")
    if cf:
        worst = a1_prov.get("clip_worst_frames") or []
        line = (f"[{tile}] A1 clip: {cf:.4%} of valid px hit the [{a1_prov['clip_floor']},255] "
                f"bounds ({a1_prov['clip_n_floored_px']:,} floored / "
                f"{a1_prov['clip_n_ceiled_px']:,} ceiled) across "
                f"{a1_prov['clip_n_frames_affected']} frame(s)")
        if worst:
            line += f"; worst frame {worst[0]['frame']} at {worst[0]['clipped_fraction']:.3%}"
        print(("  ⚠ " + line if cf >= args.warn_clip_fraction else "  " + line), flush=True)
    if not stats:
        return {"tile": tile, "status": "no_frame_stats"}

    partial_dir = out_dir / "partials" / tile
    partial_dir.mkdir(parents=True, exist_ok=True)
    reject_foreign_partials(partial_dir, args)

    # R14 on the A1 arm. This block did not exist: the driver deleted a `_sweep.json` it never
    # wrote, so `grid_id` was the only thing standing between a resume and a two-run raster.
    # `grid_id` is a *lattice* check -- two runs with different heads, window sizes, masking
    # thresholds or A1 statistics all share it, and their partials have colliding filenames.
    grid_path = partial_dir / "_sweep.json"
    if grid_path.exists():
        try:
            have = json.loads(grid_path.read_text(encoding="utf-8"))
        except ValueError:
            have = {}
        why = sweep_mismatch(have, want_sweep)
        if why and not args.force:
            raise SystemExit(
                f"[{tile}] {partial_dir} holds partials from a different sweep — {why}.\n"
                f"  Assembling them would mix two runs into one raster. Re-run with --force to "
                f"discard and recompute, or delete the directory.")
        if why:
            print(f"  ⚠ --force: discarding partials from a different sweep ({why})", flush=True)
            for p in partial_dir.glob("*.npz"):
                p.unlink()
    write_json_atomic(grid_path, want_sweep)

    grid = [(r, c) for r in row_offs for c in col_offs]
    # Hoisted out of the 144-window loop: the bboxes are a property of `frames`, not of any
    # window, and `frame_labels_on` prefilters against them (see its docstring).
    frames_bounds = frame_bounds(frames)
    t_tile = time.monotonic()
    # Where the per-window time goes. `cost["stats"]` is the once-per-tile streaming pass;
    # the rest accumulate over the sweep. Printed with DONE and banked in the sidecar.
    cost = {"stats": t_stats, "read": 0.0, "a1": 0.0, "predict": 0.0, "write": 0.0}
    n_computed = 0                  # windows this process actually computed (not resumed)
    # DECISIONS 2026-08-18c, same fix as `map_region.py`: one open for the whole sweep instead
    # of 144. `rasterio.open` on a Murray vsizip costs 7.95 s against 1.4 s for the read.
    # Lazy, so an already-partialled tile still costs nothing; `finally`, so a raise mid-sweep
    # does not leak the handle across the rest of a multi-tile run.
    tile_src = None
    try:
        for k, (row_off, col_off) in enumerate(grid):
            part = partial_dir / partial_name(row_off, col_off)
            if part.exists() and not args.force:
                try:                                         # R14: "exists" is not "usable"
                    read_partial(part)
                    continue
                except Exception as exc:                     # noqa: BLE001
                    print(f"[{tile}] partial {part.name} unreadable ({type(exc).__name__}) "
                          f"-> recomputing", flush=True)
                    part.unlink(missing_ok=True)
            t0 = time.monotonic()
            if tile_src is None:
                tile_src = open_tile(zip_path, inner_tif)
            window = read_tile_window(zip_path, inner_tif, row_off, col_off, win,
                                      dataset=tile_src)
            t_read = time.monotonic()
            cost["read"] += t_read - t0
            w_a1, nodata_mask = a1_window(window, frames, stats, fallback,
                                          frames_bounds=frames_bounds)
            n_norm = int((w_a1.data > 0).sum())
            t_a1 = time.monotonic()
            cost["a1"] += t_a1 - t_read
            pred = predict_window(w_a1, embedder, head, tile_px=TILE_PX, batch=args.batch,
                                  max_zero_fraction=args.max_zero_fraction,
                                  max_context_zero_fraction=args.max_context_zero_fraction,
                                  calibrator=calibrator,
                                  apply_isotonic=not args.no_isotonic,
                                  global_grid=grid_geom.as_tuple,
                                  nodata_mask=nodata_mask)
            t_pred = time.monotonic()
            cost["predict"] += t_pred - t_a1
            keep = np.isfinite(pred.prob)
            cols = {"ti": as_int32_cells(pred.ti[keep], "ti", tile),
                    "tj": as_int32_cells(pred.tj[keep], "tj", tile),
                    "prob": pred.prob[keep].astype(np.float32),
                    "grid_id": np.array(COARSE_GRID_ID),
                    **gate_cols(pred, tile)}
            if calibrator is not None:
                cols["prob_raw"] = pred.prob_raw[keep].astype(np.float32)
                cols["abundance"] = pred.abundance[keep].astype(np.float32)
            else:
                cols["prob_raw"] = pred.prob[keep].astype(np.float32)
            tmp = part.with_name(part.name + ".tmp")         # R14: stage, CRC-check, rename
            try:
                with open(tmp, "wb") as fh:
                    np.savez_compressed(fh, **cols)
                read_partial(tmp)
                tmp.replace(part)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
            cost["write"] += time.monotonic() - t_pred
            dt = time.monotonic() - t0
            if k % 10 == 0 or k == len(grid) - 1:
                print(f"[{tile}] win {k + 1}/{len(grid)} kept={int(keep.sum())} "
                      f"a1px={n_norm:,} {dt:.1f}s", flush=True)
            n_computed += 1
            if n_computed == 1:
                print(project_tile_cost(tile, dt, len(grid), k), flush=True)
    finally:
        if tile_src is not None:
            tile_src.close()

    # R14: set equality, not a count -- a superset satisfies a count gate.
    want_names = {partial_name(r, c) for r, c in grid}
    have_names = {p.name for p in partial_dir.glob("*.npz")}
    missing, extra = want_names - have_names, sorted(have_names - want_names)
    if missing:
        print(f"[{tile}] {len(want_names) - len(missing)}/{len(want_names)} windows -> "
              f"re-run to finish", flush=True)
        return {"tile": tile, "status": "partial",
                "windows_done": len(want_names) - len(missing), "windows_total": len(want_names)}
    if extra:
        raise SystemExit(f"[{tile}] {len(extra)} partial(s) are not part of this sweep "
                         f"(e.g. {extra[:3]}); assembling the union would mix two runs.")
    present = sorted(partial_dir / n for n in sorted(want_names))
    write_tile(tile, present, grid_geom, crs_wkt, calibrator, args, a1_prov,
               sweep_prov=want_sweep)
    if args.clean_partials:
        for p in present:
            p.unlink()
        grid_path.unlink(missing_ok=True)   # R14: else an orphan partials/<tile>/ survives
        # 2026-08-25: this `rmdir` was in `map_region.py` and NOT here, so a completed A1 arm
        # left 26 empty `partials/<tile>/` directories behind -- harmless in itself, but they
        # make `ls -d partials/*/ | wc -l` read as "26 tiles did not finish assembling", which
        # is the one signal you want trustworthy when deciding whether scratch is safe to
        # clear. Cosmetic asymmetries in a completion marker are not cosmetic.
        try:
            partial_dir.rmdir()
        except OSError:                     # not empty -- leave it and let the count show it
            pass
    elapsed = time.monotonic() - t_tile
    print(f"[{tile}] DONE in {elapsed:.0f}s  |  "
          + "  ".join(f"{k} {v:.0f}s" for k, v in cost.items())
          + f"  (unaccounted {elapsed - sum(cost.values()) + cost['stats']:.0f}s)", flush=True)
    return {"tile": tile, "status": "done", "windows": len(grid),
            "elapsed_s": round(elapsed, 1),
            # R06: the breakdown the last A1 array could not supply. `a1` is
            # frame_labels_on + a1_normalize_native; `stats` is the streaming pass.
            "cost_s": {k: round(v, 1) for k, v in cost.items()}}


def write_tile(tile, partials, grid_geom, crs_wkt, calibrator, args, a1_prov=None,
               sweep_prov=None) -> None:
    """Assemble the per-window partials into map_region-shaped GeoTIFFs (same grid, same profile).

    `(ti, tj)` are GLOBAL coarse-cell indices on `COARSE_GRID_ID`, so the affine comes from
    the global lattice — byte-identical construction to `map_region.write_tile_geotiffs`,
    which is what keeps the A1 row co-registered with the baseline row cell for cell.
    """
    loaded = [read_partial(p) for p in partials]

    def cat(key):
        return np.concatenate([z[key] for z in loaded])

    foreign = [str(p) for p in partials if partial_grid_id(p) != COARSE_GRID_ID]
    if foreign:
        raise SystemExit(f"[{tile}] refusing to assemble {len(foreign)} partial(s) from "
                         f"another lattice: {foreign[:3]}")
    ti, tj = cat("ti").astype(np.int64), cat("tj").astype(np.int64)
    prob, prob_raw = cat("prob").astype(np.float64), cat("prob_raw").astype(np.float64)
    ab = cat("abundance").astype(np.float64) if calibrator is not None else None
    ti_min, tj_min = int(ti.min()), int(tj.min())
    shape = (int(ti.max()) - ti_min + 1, int(tj.max()) - tj_min + 1)
    transform = grid_geom.transform(ti_min, tj_min)
    out_dir = Path(args.out_dir)

    def scatter(v):
        r = np.full(shape, np.nan, dtype=np.float64)
        r[ti - ti_min, tj - tj_min] = v
        return r

    # R14: the fraction of duplicated cells that disagree, on `prob_raw`. Not a magnitude on
    # `prob` -- that test stalled two fully-rendered baseline tiles on fp16 noise. See
    # `check_overlap`. The A1 arm was expected to hit the same lottery on ~2 of its 26 tiles.
    overlap = check_overlap(tile, ti, tj,
                            {"prob_raw": prob_raw, "prob": prob, "abundance": ab})

    # R14: commit as a SET, sidecar LAST. The old sentinel was `_prob_raw.tif`, written first.
    emitted = [("prob_raw", scatter(prob_raw)), ("prob", scatter(prob))]
    if ab is not None:
        emitted.append(("abundance", scatter(ab)))
    rasters = []
    tags = size_floor_tags(args)          # R84: identical basis to the baseline arm, by design --
    for kind, arr in emitted:             # the two rows are compared, so they must count the same
        p = write_geotiff(out_dir / f"{tile}_{kind}.tif", arr, transform, crs_wkt, tags=tags)
        rasters.append({"name": p.name, "kind": kind, "bytes": p.stat().st_size,
                        "sha256": file_sha256(p), "shape": list(shape),
                        "n_finite": int(np.isfinite(arr.astype(np.float32)).sum())})
    write_json_atomic(out_dir / f"{tile}.json", {
        "murray_tile": tile, "tile_px": TILE_PX, "raster_shape": list(shape),
        "rasters": rasters,
        "overlap": overlap,                 # R14: full per-layer record; see `check_overlap`
        "overlap_disagreements": overlap[overlap["gate_layer"]]["n_significant"],
        # R14: the commit record. `rasters` lets a resume verify content without re-deriving it;
        # `run` lets it verify the content was made the way this run would -- including the A1
        # arm, reference and SeamMap digest, which no content check can see.
        "run": sweep_prov,
        # R13, and note the threshold this row ships with (see `--max-context-zero-fraction`).
        "nodata_gate": gate_summary(
            loaded, max_zero_fraction=args.max_zero_fraction,
            max_context_zero_fraction=args.max_context_zero_fraction),
        # R01: same keys, same values as the baseline sidecar, so "is the A1 row on the
        # baseline's lattice?" is answerable from the two JSONs without opening a raster.
        **grid_geom.provenance(),
        "ti_min": ti_min, "tj_min": tj_min, "n_predicted_tiles": int(ti.size),
        "calibrated": calibrator is not None,
        "isotonic": calibrator is not None and not args.no_isotonic,
        "prob_mean": float(np.nanmean(prob)), "rich_share_at_0p5": float((prob >= 0.5).mean()),
        "abundance_mean": float(np.nanmean(ab)) if ab is not None else None,
        # Provenance must describe the run, not the default. Until 2026-08-06 this line
        # recorded the A1_HEAD *constant* while the region manifest recorded `args.head`,
        # so a `--head <other>` run produced two provenance records that contradicted each
        # other and neither was flagged. The digest is what actually pins the artifact: a
        # path can be overwritten in place, and the head directory's name is a hash of the
        # training recipe, not of the weights it produced.
        "variant": "A1",
        "head": str(args.head),
        "head_digest": artifact_digest(args.head),
        "calibration": str(args.calibration) if args.calibration else None,
        "calibration_digest": artifact_digest(args.calibration) if args.calibration else None,
        "a1_ref": {"median": A1_REF_MEDIAN, "iqr": A1_REF_IQR},
        # R07: record the arm and the measured statistic, not a prose description of it.
        **{f"a1_{k}": v for k, v in (a1_prov or {}).items()},
    })


def build_parser() -> argparse.ArgumentParser:
    """The CLI, separated from `main` so the shipped DEFAULTS are assertable in a test —
    specifically that this arm's R13 context gate stays disabled until R38 lands."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="*", default=None,
                    help="default = the 9 CTX-equipped block tiles (§5.1's common footprint)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--head", default=str(A1_HEAD))
    ap.add_argument("--calibration", default=None,
                    help="CalibrationLayer npz; omit for raw P(rich) only (η² is scored on raw P)")
    ap.add_argument("--win-px", type=int, default=4096)
    ap.add_argument("--require-device", action="append", metavar="SUBSTRING",
                    help="refuse to render unless the GPU name contains this (repeatable). "
                         "See scripts/map_region.py:require_device.")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-zero-fraction", type=float, default=0.3)
    # R13 x R38 — ENABLED (0.0) as of 2026-08-10, and here is why it was not before. A1 used to
    # clip to [0, 255], so a legal dark pixel was written as the nodata sentinel: measured on the
    # 38 training windows as a deploy proxy, the share of own-tile-passing cells carrying >=1
    # "nodata" context pixel went 0.00 % on the raw mosaic -> 2.67 % under the native A1
    # statistic -> ~13 % under the 160 m one. A zero-tolerance gate would have deleted a large
    # slice of the map for a RADIOMETRIC reason dressed as a data gap.
    #
    # R38 fixed that at the root and not by moving the floor. Flooring valid pixels at 1 alone
    # would have made those pixels invisible to this gate while leaving the embedding damage
    # intact (DN 0 and DN 1 move the prediction identically to three decimals -- the damage is
    # blackness, not the sentinel). So the driver now passes `predict_window` an EXPLICIT nodata
    # mask taken from the raw DN, the clip floor moved to A1_VALID_FLOOR so DN 0 is unambiguous,
    # and the destroyed-texture count is recorded separately as `a1_clip_*`. Only then does 0.0
    # mean here what it means on the baseline arm.
    ap.add_argument("--max-context-zero-fraction", type=float, default=0.0,
                    help="R13 context-nodata gate. 0.0 = not one nodata pixel in the 96-px "
                         "context box, matching scripts/map_region.py. Safe on this arm only "
                         "since R38: the mask is explicit, not inferred from A1's output.")
    ap.add_argument("--size-floor-basis", default=str(DEFAULT_SIZE_FLOOR_BASIS),
                    help="R84: banked size-floor basis JSON. Must be the SAME basis the baseline "
                         "arm used -- the two rows are compared cell for cell, so a raster that "
                         "counts different boulders is not comparable however well it aligns.")
    ap.add_argument("--warn-clip-fraction", type=float, default=0.01,
                    help="R38: warn loudly when this share of a tile's valid pixels hits the A1 "
                         "clip bounds. Warn, not refuse -- a clipped pixel is a radiometric "
                         "loss, not a data gap, and punching holes in the map over it needs its "
                         "own justification. The number is recorded in the sidecar either way.")
    ap.add_argument("--no-isotonic", action="store_true")
    ap.add_argument("--clean-partials", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()

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
    # R07: strict here. This path feeds A1-normalised DN, so an unverifiable head must not
    # pass as correct -- and the A1 head has to be retrained for R07 regardless, which is
    # exactly when it will start declaring the arm.
    require_norm_arm(head, A1_ARM, where=str(args.head), strict=True)
    require_device(embedder, args.require_device)
    print(f"A1 map: {len(tiles)} tiles {tiles}\n  head={Path(args.head).parent.name}, "
          f"A1 ref (median, IQR) = ({A1_REF_MEDIAN}, {A1_REF_IQR}), "
          f"calibration={'on' if calibrator else 'raw only'}", flush=True)

    results = []
    for tile in tiles:
        # same isolation as the baseline arm: one bad tile must not forfeit the stride
        results.append(run_tile_isolated(
            tile, lambda t=tile: process_tile(t, embedder, head, calibrator, args)))
    (Path(args.out_dir) / "a1_manifest.json").write_text(
        json.dumps({"tiles": results, "head": str(args.head),
                    "head_digest": artifact_digest(args.head),
                    "calibration": str(args.calibration) if args.calibration else None,
                    "calibration_digest": (
                        artifact_digest(args.calibration) if args.calibration else None),
                    "win_px": args.win_px, "grid_id": COARSE_GRID_ID,
                    "batch": args.batch, "a1_ref_median": A1_REF_MEDIAN,
                    "a1_ref_iqr": A1_REF_IQR}, indent=2), encoding="utf-8")
    done = sum(1 for r in results if r["status"] == "done")
    print(f"\nA1 map: {done}/{len(tiles)} tiles complete -> {args.out_dir}")
    missing = [r["tile"] for r in results if r["status"] == "no_ctx_zip"]
    if missing:
        print(f"⚠ {len(missing)} tile(s) had no cached CTX zip and were skipped: {missing}\n"
              f"  §5.1's common footprint is the tiles that DID render — record it on the table.")
    failed = [r["tile"] for r in results if r["status"] == "failed"]
    if failed:
        print(f"⚠ {len(failed)}/{len(tiles)} tile(s) FAILED: {failed}")
        for r in results:
            if r["status"] == "failed":
                print(f"    {r['tile']}: {r['error']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
