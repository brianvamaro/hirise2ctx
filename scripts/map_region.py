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

from src.mapping import (COARSE_GRID_ID, artifact_digest, assert_shared_lattice, file_sha256,
                         predict_window, read_tile_window, tile_global_grid, uncovered_cells,
                         verify_geotiff, window_offsets, write_geotiff)

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


def partial_status(path) -> str:
    """`"ok"` | `"damaged"` | `"foreign"` for a per-window partial.

    **R14/R01.** These need different responses and used to get the same one. A *damaged*
    partial (truncated by a wall-clock kill) is just work to redo — deleting and recomputing it
    is always right. A *foreign* partial carries a different `grid_id`: recomputing silently
    would be wrong, because it means the operator is pointing this run at another lattice's
    output. Collapsing the two made a corrupt file demand `--force`, which also discards every
    good partial in the directory.
    """
    try:
        with np.load(path, allow_pickle=False) as z:
            if "grid_id" not in z.files:
                return "foreign"                 # pre-R01 partials carry none
            gid = str(z["grid_id"])
            for k in z.files:                    # force the CRC check on every member
                _ = z[k]
    except Exception:                            # noqa: BLE001
        return "damaged"
    return "ok" if gid == COARSE_GRID_ID else "foreign"


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
    status = {p: partial_status(p) for p in sorted(partial_dir.glob("*.npz"))}
    damaged = [p for p, s in status.items() if s == "damaged"]
    if damaged:
        # Not a lattice question — just work to redo. Delete and let the sweep recompute;
        # requiring --force here would also throw away every good partial beside it.
        print(f"  ⚠ {len(damaged)} partial(s) are unreadable (truncated by a kill?) -> "
              f"deleting so they are recomputed: {[p.name for p in damaged[:3]]}", flush=True)
        for p in damaged:
            p.unlink(missing_ok=True)
    stale = [p for p, s in status.items() if s == "foreign"]
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


def tile_is_reusable(out_dir: Path, tile: str, want_run: dict | None, *,
                     verify: bool = False, trust_existing: bool = False) -> str | None:
    """Why an already-rendered tile may NOT be reused, or None if it may be.

    **R14.** Resume was `{tile}_prob.tif.exists()` — no size, no read, no provenance, and
    keyed on the FIRST of four artifacts. Content and provenance catch disjoint failures and
    both are needed: provenance alone cannot see post-commit transport damage (all 78 shipped
    rasters share one mtime — they were bulk-copied off Sherlock, which is exactly the vector
    that produced R23's truncated shapefiles), and content alone cannot see a raster that is
    structurally perfect and made with the wrong head.

    Cheap by default: the exhaustive decode happens once at commit, so reuse checks size +
    sha256 (a few MB per tile). `verify=True` re-decodes.
    """
    side = out_dir / f"{tile}.json"
    if not side.exists():
        return "no sidecar (the commit marker) — the tile was never committed"
    try:
        rec = json.loads(side.read_text(encoding="utf-8"))
    except ValueError as exc:
        return f"sidecar is unreadable ({type(exc).__name__})"
    rasters = rec.get("rasters")
    if not rasters:
        # the 26 shipped tiles predate this field entirely
        return (None if trust_existing else
                "sidecar predates R14 provenance, so its contents cannot be verified "
                "(pass --trust-existing to accept it anyway)")
    for r in rasters:
        p = out_dir / r["name"]
        why = verify_geotiff(
            p, expect_bytes=r.get("bytes"), expect_sha256=r.get("sha256"),
            expect_shape=tuple(r["shape"]) if verify and r.get("shape") else None,
            expect_finite=r.get("n_finite") if verify else None,
            expect_count=1 if verify else None, expect_dtype="float32" if verify else None)
        if why:
            return f"{r['name']}: {why}"
    if want_run is not None:
        why = sweep_mismatch(rec.get("run") or {}, want_run)
        if why:
            return f"rendered by a different run — {why}"
    return None


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


def write_json_atomic(path: Path, obj) -> Path:
    """Write JSON via a `.tmp` sibling + rename, so a reader never sees half a record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def partial_name(row_off: int, col_off: int) -> str:
    return f"{row_off:06d}_{col_off:06d}.npz"


def read_partial(path) -> dict:
    """Load a per-window partial, forcing every member through zipfile's CRC check.

    **R14.** `np.load` alone builds a `ZipFile` and reads the central directory, so it raises
    on a *truncated* `.npz` but happily accepts one whose directory survived while a deflate
    stream is corrupt. Only touching each member validates the CRC.
    """
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sweep_manifest(grid_geom, row_offs, col_offs, args, *, extent, head_digest,
                   calibration_digest) -> dict:
    """The identity of the sweep that a set of partials belongs to.

    **R14.** Resume used to be `path.exists()`, which cannot tell a partial written by a
    different window size, head or masking threshold from one written by this run. Every field
    here is a resume-match field; a mismatch names itself.
    """
    return {
        "grid_id": COARSE_GRID_ID,
        "cell_row0": grid_geom.cell_row0, "cell_col0": grid_geom.cell_col0,
        "phase": [grid_geom.phase_r, grid_geom.phase_c],
        "tile_px": TILE_PX, "win_px": int(args.win_px), "overlap": 3 * TILE_PX,
        "extent": [int(extent[0]), int(extent[1])],
        "n_windows": len(row_offs) * len(col_offs),
        "row_offsets": [int(o) for o in row_offs], "col_offsets": [int(o) for o in col_offs],
        "max_zero_fraction": float(args.max_zero_fraction),
        "isotonic": (not args.no_isotonic),
        "calibrated": calibration_digest is not None,
        "head_digest": head_digest, "calibration_digest": calibration_digest,
    }


def sweep_mismatch(have: dict, want: dict) -> str | None:
    """First field on which two sweep manifests differ, or None. Missing counts as differing."""
    for k, v in want.items():
        if k not in have:
            return f"{k} absent (partials predate this field)"
        if have[k] != v:
            return f"{k}: partials have {have[k]!r}, this run wants {v!r}"
    return None


def expected_cells_per_axis(extent: int, phase: int, tile_px: int = TILE_PX) -> int:
    """How many coarse cells this tile can support on one axis, given its R01 phase.

    Deliberately derived rather than taken from the register's
    `n_windows * (win_px/tile_px - 2)**2`, which R01 invalidated: with a phase the per-window
    yield is `(win/S - 3)**2` for windows whose shifted origin is not a multiple of `S`, so
    that formula now fails on correct output.
    """
    return len(range(tile_px + phase, extent - 2 * tile_px + 1, tile_px))


def overlap_disagreement(ti, tj, values) -> tuple[int, float]:
    """`(n_cells, max_abs_diff)` where overlapping windows wrote DIFFERENT values for a cell.

    **R14, and it is a CROSS-RUN detector specifically.** Windows overlap by 3·tile_px in
    *pixels* so every cell has full context, but the cell assignment is a **partition**:
    measured on the sweep this driver uses, 900 cells over 36 windows with 0 computed twice.
    So within one run this returns `(0, 0.0)` by construction and can never false-positive.

    It fires when partials from two runs are assembled together — both runs compute the same
    cell set on the same lattice, so every cell collides, and a cell's value depends only on
    the cell (its context box is fixed by the cell, not by the window). Measured on a real
    stale-partial mixture: 63.1 % of finite pixels came from the stale run while every file
    was structurally perfect and the raster came out the right shape. Neither a shape check
    nor a decode can see that; this can.
    """
    key = np.asarray(ti, dtype=np.int64) * (2 ** 21) + np.asarray(tj, dtype=np.int64)
    o = np.argsort(key, kind="stable")
    k, v = key[o], np.asarray(values, dtype=np.float64)[o]
    dup = k[1:] == k[:-1]
    if not dup.any():
        return 0, 0.0
    a, b = v[:-1][dup], v[1:][dup]
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    diff = np.nan_to_num(diff, nan=np.inf)
    return int((diff > 0).sum()), float(diff.max())


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

    # R01: place this tile on the ONE global coarse lattice. Constructing the grid is what
    # verifies it -- the sphere radius is parsed from this tile's own CRS and the origin is
    # checked against the native lattice, so nothing downstream can stamp COARSE_GRID_ID on
    # a product that was not actually rendered on it.
    grid_geom = tile_global_grid(inner_transform, crs_wkt, TILE_PX)

    # overlap = 3*TILE_PX and a non-tile-aligned final offset are BOTH required once the
    # lattice has a phase; either alone still leaves holes (see `window_offsets`). Free:
    # 12 offsets per axis in every variant, so the window count is unchanged.
    win, overlap = args.win_px, 3 * TILE_PX
    row_offs = window_offsets(H, win, overlap, TILE_PX, tile_aligned=False)
    col_offs = window_offsets(W, win, overlap, TILE_PX, tile_aligned=False)
    # the ROW axis takes the row phase. These were transposed when R01 part 2 landed; inert in
    # the shipped configuration (coverage is complete at every phase, so both orderings return
    # empty) but the guard would have checked the wrong axis the moment anything else moved.
    miss_r = uncovered_cells(row_offs, H, win, TILE_PX, phase=grid_geom.phase_r)
    miss_c = uncovered_cells(col_offs, W, win, TILE_PX, phase=grid_geom.phase_c)
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

    # R14: pin the sweep these partials belong to, BEFORE any of them is written. Without it
    # a resume cannot tell a partial from a different --win-px, head or masking threshold from
    # one of its own -- and the old count gate let 719 files satisfy a 144-window grid.
    want_sweep = sweep_manifest(grid_geom, row_offs, col_offs, args, extent=(H, W),
                                head_digest=artifact_digest(getattr(args, "_model_dir", ""))
                                if getattr(args, "_model_dir", None) else None,
                                calibration_digest=(artifact_digest(args.calibration)
                                                    if calibrator is not None else None))

    # R14: resume on the SIDECAR (the commit marker), with content AND provenance. The old
    # sentinel was `{tile}_prob.tif.exists()` -- the FIRST of four artifacts, checked for
    # nothing but existence -- so a kill between artifacts 1 and 4 left a tile permanently
    # "done" with no abundance raster, and a --raw run made a later calibrated run report
    # skipped_done while the region shipped a raw tile inside a map claiming calibrated:true.
    if not args.force:
        why = tile_is_reusable(out_dir, murray_tile, want_sweep,
                               verify=getattr(args, "verify_existing", False),
                               trust_existing=getattr(args, "trust_existing", False))
        if why is None and prob_tif.exists():
            print(f"[{murray_tile}] committed and verified -> skip "
                  f"(--force to redo)", flush=True)
            return {"tile": murray_tile, "status": "skipped_done"}
        if prob_tif.exists():
            # R01's case is one instance of this: every pre-R01 tile is on disk, so a bare
            # existence check would have skipped all 26 and certified a rebuild that rendered
            # nothing. Refuse rather than silently re-render over a shipped product.
            lat = existing_product_off_lattice(prob_tif)
            raise SystemExit(
                f"[{murray_tile}] {prob_tif} exists but cannot be reused: {why}"
                + (f"\n  Also: not on {COARSE_GRID_ID} ({lat})." if lat else "")
                + f"\n  Re-run with --force to re-render, --trust-existing to accept a "
                  f"pre-R14 sidecar, or point --out-dir at a fresh directory."
            )

    partial_dir = out_dir / "partials" / murray_tile
    partial_dir.mkdir(parents=True, exist_ok=True)
    reject_foreign_partials(partial_dir, args)
    grid_path = partial_dir / "_sweep.json"
    if grid_path.exists():
        try:
            have = json.loads(grid_path.read_text(encoding="utf-8"))
        except ValueError:
            have = {}
        why = sweep_mismatch(have, want_sweep)
        if why and not args.force:
            raise SystemExit(
                f"[{murray_tile}] {partial_dir} holds partials from a different sweep — {why}.\n"
                f"  Assembling them would mix two runs into one raster. Re-run with --force to "
                f"discard and recompute, or delete the directory.")
        if why:
            print(f"  ⚠ --force: discarding partials from a different sweep ({why})", flush=True)
            for p in partial_dir.glob("*.npz"):
                p.unlink()
    write_json_atomic(grid_path, want_sweep)

    done_tiles = 0
    t_tile = time.monotonic()
    for k, (row_off, col_off) in enumerate(grid):
        part_path = partial_dir / partial_name(row_off, col_off)
        if part_path.exists() and not args.force:
            # R14: "exists" is not "usable". A partial truncated by a wall-clock kill was
            # skipped here and then blew up at assembly, so every re-run crashed at the same
            # point forever. Validate it (CRC included) and recompute it if it is damaged.
            try:
                read_partial(part_path)
                continue
            except Exception as exc:                     # noqa: BLE001
                print(f"[{murray_tile}] partial {part_path.name} is unreadable "
                      f"({type(exc).__name__}) -> recomputing", flush=True)
                part_path.unlink(missing_ok=True)
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
        # R14: stage then rename, so the final name never exists in a half-written state --
        # `np.savez_compressed` writes the zip in place with no tmp+rename of its own.
        # Write through a HANDLE: `np.savez_compressed` appends ".npz" to a path that does not
        # already end in it, so a `.npz.tmp` target silently becomes `.npz.tmp.npz`. Keeping
        # the ".tmp" suffix also keeps these out of the `*.npz` glob the set gate uses.
        tmp_part = part_path.with_name(part_path.name + ".tmp")
        try:
            with open(tmp_part, "wb") as fh:
                np.savez_compressed(fh, **cols)
            read_partial(tmp_part)                       # CRC round-trip before it counts
            tmp_part.replace(part_path)
        except BaseException:
            tmp_part.unlink(missing_ok=True)
            raise
        done_tiles += 1
        dt = time.monotonic() - t0
        rate = int(keep.sum() / dt) if dt > 0 else 0
        print(f"[{murray_tile}] win {k + 1}/{len(grid)} off=({row_off},{col_off}) "
              f"kept={int(keep.sum())} {dt:.1f}s ~{rate} tiles/s", flush=True)

    # Assemble. R14: this gate was `len(present) < len(grid)` — a COUNT over a glob, which a
    # superset satisfies. Measured on the reachable stale state (a completed --win-px 2048
    # sweep, then --win-px 4096 --force): 719 partials on disk against 144 expected, the count
    # gate passes, and the emitted raster is the right shape with 63.1 % of its finite pixels
    # from the stale run. Set equality is what fails it (extras 575, missing 0).
    if args.limit_windows is not None:
        print(f"[{murray_tile}] benchmark mode (--limit-windows) -> skip assembly", flush=True)
        return {"tile": murray_tile, "status": "benchmark", "windows_done": done_tiles,
                "elapsed_s": round(time.monotonic() - t_tile, 1)}
    want_names = {partial_name(r, c) for r, c in grid}
    have_names = {p.name for p in partial_dir.glob("*.npz")}
    missing, extra = sorted(want_names - have_names), sorted(have_names - want_names)
    if missing:
        print(f"[{murray_tile}] {len(have_names & want_names)}/{len(want_names)} windows done "
              f"-> not yet assembling (re-run to finish)", flush=True)
        return {"tile": murray_tile, "status": "partial",
                "windows_done": len(have_names & want_names), "windows_total": len(want_names)}
    if extra:
        raise SystemExit(
            f"[{murray_tile}] {len(extra)} partial(s) in {partial_dir} are not part of this "
            f"sweep (e.g. {extra[:3]}). Assembling the union would mix two runs into one "
            f"raster. Delete them, or re-run with --force.")

    present = sorted(partial_dir / n for n in sorted(want_names))
    write_tile_geotiffs(murray_tile, present, grid_geom, crs_wkt, calibrator, args,
                        expected_cells=(expected_cells_per_axis(H, grid_geom.phase_r),
                                        expected_cells_per_axis(W, grid_geom.phase_c)),
                        sweep_prov=want_sweep)
    if args.clean_partials:
        for p in present:
            p.unlink()
        grid_path.unlink(missing_ok=True)   # R14: else an orphan partials/<tile>/ survives
        try:
            partial_dir.rmdir()
        except OSError:
            pass
    print(f"[{murray_tile}] DONE in {time.monotonic() - t_tile:.0f}s", flush=True)
    return {"tile": murray_tile, "status": "done", "windows": len(grid)}


def write_tile_geotiffs(murray_tile, partials, grid_geom, crs_wkt, calibrator, args,
                        expected_cells=None, sweep_prov=None):
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
    loaded = [read_partial(p) for p in partials]
    ti = np.concatenate([z["ti"] for z in loaded]).astype(np.int64)
    tj = np.concatenate([z["tj"] for z in loaded]).astype(np.int64)
    prob = np.concatenate([z["prob"] for z in loaded]).astype(np.float64)
    has_cal = calibrator is not None
    prob_raw = (np.concatenate([z["prob_raw"] for z in loaded]).astype(np.float64)
                if has_cal else None)
    abundance = (np.concatenate([z["abundance"] for z in loaded]).astype(np.float64)
                 if has_cal else None)

    # R14: overlapping windows must agree. A cell's context box is fixed by the cell, not by
    # the window that computed it, so two windows covering it produce the same prediction.
    # Disagreement means two runs are being assembled into one raster -- the failure that is
    # invisible to every structural check because each file is perfect and the shape is right.
    n_dis, max_dis = overlap_disagreement(ti, tj, prob)
    if n_dis and max_dis > 1e-6:
        raise SystemExit(
            f"[{murray_tile}] {n_dis} cells were written twice with DIFFERENT values "
            f"(max |Δ| = {max_dis:.4g}). Overlapping windows of one run agree by construction, "
            f"so these partials come from more than one run. Refusing to assemble.")

    ti_min, ti_max = int(ti.min()), int(ti.max())
    tj_min, tj_max = int(tj.min()), int(tj.max())
    shape = (ti_max - ti_min + 1, tj_max - tj_min + 1)
    n_unique = int(np.unique(ti.astype(np.int64) * (2 ** 21) + tj).size)
    if expected_cells is not None:
        want = (int(expected_cells[0]), int(expected_cells[1]))
        if shape > want:            # never larger than the tile can support
            raise SystemExit(f"[{murray_tile}] assembled shape {shape} exceeds the {want} "
                             f"cells this tile can support -- partials from another sweep?")
    transform = grid_geom.transform(ti_min, tj_min)
    out_dir = Path(args.out_dir)

    def scatter(values):
        r = np.full(shape, np.nan, dtype=np.float64)
        r[ti - ti_min, tj - tj_min] = values  # overlap re-writes identical values
        return r

    # R14: commit the tile as a SET, sidecar LAST. Every raster is staged and verified by
    # `write_geotiff`, but per-file atomicity is not enough on its own: the old resume sentinel
    # was `_prob.tif`, the FIRST of four artifacts, so a kill between artifacts 1 and 4 left a
    # tile permanently marked done with no abundance raster -- and abundance is the deliverable.
    # The sidecar is now the only completion marker, and it is written last.
    emitted = [("prob", scatter(prob))]
    if has_cal:
        emitted += [("abundance", scatter(abundance)), ("prob_raw", scatter(prob_raw))]
    rasters = []
    for kind, arr in emitted:
        p = write_geotiff(out_dir / f"{murray_tile}_{kind}.tif", arr, transform, crs_wkt)
        rasters.append({"name": p.name, "kind": kind, "bytes": p.stat().st_size,
                        "sha256": file_sha256(p), "shape": list(shape),
                        "n_finite": int(np.isfinite(arr.astype(np.float32)).sum())})
    n_tiles = ti.size
    write_json_atomic(out_dir / f"{murray_tile}.json", {
        "murray_tile": murray_tile, "tile_px": TILE_PX, "raster_shape": list(shape),
        # R14: the commit record. `rasters` lets a resume verify content without re-deriving
        # it, and `run` lets it verify that the content was made the way this run would.
        "rasters": rasters,
        "run": sweep_prov,
        "n_unique_cells": n_unique,
        "overlap_disagreements": n_dis,
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
    })


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
    # R14: the 26 shipped tiles carry pre-R14 sidecars with no `rasters` block, so their
    # contents cannot be verified. That is correctly "not reusable"; these are the escapes.
    ap.add_argument("--trust-existing", action="store_true",
                    help="accept a pre-R14 sidecar (no per-raster bytes/sha256) as complete")
    ap.add_argument("--verify-existing", action="store_true",
                    help="fully re-decode every existing raster on resume, not just size+sha256")
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
    from src.modeling.mlp_head import NO_NORM_ARM, DeployableHead, require_norm_arm

    calibrator = None
    if not args.raw:
        from src.calibration import CalibrationLayer
        calibrator = CalibrationLayer.load(args.calibration)
        print(f"  calibration={Path(args.calibration).name}  "
              f"isotonic={'off' if args.no_isotonic else 'on'}  abundance=qmatch(P(rich))",
              flush=True)

    embedder = FangEmbedder.load()
    head = DeployableHead.load(model_dir)
    # R07: this driver feeds RAW Murray DN. Refuse a head that declares an A1 arm; only warn
    # on an unversioned one, since unversioned + raw is exactly the pre-R07 status quo and
    # blocking the baseline re-render on a provenance field would buy no safety.
    require_norm_arm(head, NO_NORM_ARM, where=str(model_dir), strict=False)
    dev = getattr(getattr(embedder, "device", None), "type", "?")
    print(f"  embedder device={dev}  head seeds={card.get('n_seeds', '?')}", flush=True)

    results = []
    t0 = time.monotonic()
    for tile in tiles:
        results.append(map_one_tile(tile, embedder, head, calibrator, args=args))

    # R14: MERGE, do not clobber. The shipped manifest lists 4 tiles while 26 tiles' rasters
    # are on disk, because every run overwrote it -- so 22 of 26 shipped tiles have no manifest
    # record at all and `win_px` (recorded only here) is unknown for them. The per-tile `run`
    # block in each sidecar is the authority for how a tile was made; this file is an index of
    # runs, so the per-run scalars are appended as a list rather than kept as one lying scalar.
    manifest = Path(args.out_dir) / "region_manifest.json"
    prev = {}
    if manifest.exists():
        try:
            prev = json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError:
            prev = {}
    by_tile = {r["tile"]: r for r in (prev.get("results") or []) if isinstance(r, dict)}
    by_tile.update({r["tile"]: r for r in results})
    runs = list(prev.get("runs") or [])
    if not runs and prev.get("model_dir"):          # fold a pre-R14 manifest in as run #0
        runs.append({k: prev.get(k) for k in
                     ("model_dir", "head_digest", "calibration", "calibration_digest",
                      "ctx_tiles", "recipe_hash", "win_px", "calibrated", "raw")})
    runs.append({
        "model_dir": str(model_dir), "head_digest": artifact_digest(model_dir),
        "calibration": str(args.calibration) if calibrator is not None else None,
        "calibration_digest": (
            artifact_digest(args.calibration) if calibrator is not None else None),
        "ctx_tiles": str(args.ctx_tiles), "recipe_hash": card.get("recipe_hash"),
        "win_px": args.win_px, "calibrated": calibrator is not None, "raw": args.raw,
        "tiles": tiles, "elapsed_s": round(time.monotonic() - t0, 1),
    })
    write_json_atomic(manifest, {
        "grid_id": COARSE_GRID_ID,
        "tiles": sorted(by_tile), "runs": runs,
        "results": [by_tile[t] for t in sorted(by_tile)],
    })
    n_done = sum(r["status"] == "done" for r in results)
    print(f"=== {n_done}/{len(tiles)} tiles complete  "
          f"{time.monotonic() - t0:.0f}s  manifest -> {manifest} ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
