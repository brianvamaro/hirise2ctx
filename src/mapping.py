"""Off-HiRISE map inference helpers (PLAN_FM §2.6 B-D).

The frozen recipe's deployable head (`src.modeling.mlp_head.DeployableHead`)
predicts rich/poor per CTX tile from a single GeM embedding. To paint a map
beyond HiRISE coverage we (1) window a region of a Murray Lab CTX tile,
(2) enumerate the S=32 tiles whose full 3x3-context box fits, (3) embed each box
with `src.fm_embeddings.FangEmbedder`, (4) predict, and (5) place the per-tile
probabilities into a coarse (160 m/px) raster georeferenced in the tile's CRS.

This module owns the torch-free geometry/raster glue (window read, own-tile
validity, (ti,tj)->raster placement, the 32x-coarsened affine). The embedding
and the head are passed in by the caller (`scripts/map_pilot.py`), so this stays
a thin, testable seam.

Grid convention — **R01, and it changed.** The coarse cell lattice is now anchored
**globally**, at projected (0, 0) = lon 0 / lat 0, not to each parent Murray tile's
pixel origin. A tile is 47,420 native px wide and `gcd(47420, 32) = 4`, so tile
anchoring put every tile on its own sub-cell phase and `rasterio.merge` floored that
phase into a whole-cell displacement. `(ti, tj)` from `predict_window(global_grid=...)`
are therefore **global cell indices**, unique across the planet, and a cross-tile
combine no longer needs the Murray-tile id to disambiguate them.

Two things that did **not** change, deliberately: the legacy `global_grid=None` path
still anchors to `row0=row_off, col0=col_off` (so `scripts/map_pilot.py` is untouched),
and the Stage-4 **label** grid stays tile-anchored (`src.labeling._compute_grid_alignment`,
CLAUDE.md Stage 4) — re-anchoring it would force a relabel + retrain of the frozen recipe
for no modelling gain. Consequence: map cells no longer coincide with label cells, so a
map↔label comparison must resample rather than index-match.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ============================================================================
# R01 — the globally anchored coarse grid
# ============================================================================
#
# A Murray Lab tile is 47,420 native px wide and adjacent tile origins are exactly
# 47,420 px apart. But `gcd(47420, 32) = 4`, so `47420 % 32 = 28 == -4 (mod 32)`: each
# tile's coarse 32-px lattice starts at a *different* sub-cell phase, walking 4 native px
# (20 m) per 4-degree step. Measured over the 26-tile footprint: 8 distinct x-phases,
# 4 distinct y-phases, and every adjacent tile pair offset by exactly 20 m.
#
# Anchoring each tile's predictions to its own parent-tile origin therefore puts every
# tile on its own lattice. `rasterio.merge` then *floors* each fractional destination
# offset, converting the sub-cell phase into a whole-cell placement error: measured on the
# shipped mosaic, 25 of 26 tiles are displaced, median 140 m, max 198 m, with 21 of 26
# beyond half a cell. Correcting only the integer part already lifts the THEMIS validation
# correlation from |rho| 0.0741 to 0.0821 (n=26 tiles).
#
# The fix is one grid for the whole planet, anchored at projected (0, 0) = lon 0 / lat 0.
# Every Murray tile origin is exactly integral on that native lattice (verified: 24/24
# cached sidecars, worst residual 2.3e-9 m), so a tile maps onto it with an integer offset
# and an integer phase.

MURRAY_RADIUS_M = 3396190.0        # Mars_2015 sphere, present in every tile's inner_crs_wkt
MURRAY_PPD = 11855                 # 47420 px / 4 deg, Murray Lab CTX mosaic V01
MURRAY_NATIVE_M = math.pi * MURRAY_RADIUS_M / 180.0 / MURRAY_PPD   # 4.999974485306303
COARSE_GRID_ID = "murray_v01_clon0_R3396190_ppd11855_S32_anchor_lonlat0"

# Why a canonical constant rather than each tile's own `a`: the cached sidecars carry FOUR
# distinct pixel sizes (4.999974485306304 x14, ...035 x8, ...295 x1, ...302 x1) and **none**
# equals the exact value. Building per-tile transforms from `a` re-imports that ULP spread
# and makes the merge offsets non-integral; the canonical constant makes `a`, `c` and `f`
# bit-identical across tiles, which is what lets `rasterio.merge` place them exactly.

_SPHEROID_RE = re.compile(r'SPHEROID\s*\[\s*"[^"]*"\s*,\s*([0-9.eE+-]+)')


def assert_murray_sphere(crs_wkt: str | None, *, tol_m: float = 1.0) -> float:
    """Read the sphere radius out of a tile's CRS WKT and check it, rather than assume it.

    `COARSE_GRID_ID` asserts `R3396190`; without this the radius half of that identity
    would be an assertion nothing measures — the failure mode caught twice already this
    week. Returns the parsed radius.
    """
    if not crs_wkt:
        raise ValueError("cannot verify the Murray sphere: no CRS WKT supplied")
    m = _SPHEROID_RE.search(crs_wkt)
    if not m:
        raise ValueError("cannot verify the Murray sphere: no SPHEROID in the CRS WKT")
    radius = float(m.group(1))
    if abs(radius - MURRAY_RADIUS_M) > tol_m:
        raise ValueError(
            f"tile sphere radius {radius} != the grid's {MURRAY_RADIUS_M}; "
            f"{COARSE_GRID_ID} does not describe this product."
        )
    return radius


def global_native_origin(inner_transform, *, native_m: float = MURRAY_NATIVE_M,
                         tol_m: float = 1e-3) -> tuple[int, int]:
    """`(row, col)` of a Murray tile's origin in GLOBAL native px, anchored at lon0/lat0.

    VERIFY AT RUNTIME: raises when the origin is not integral on that lattice, which is the
    standing tripwire for "this raster is not a Murray V01 tile".
    """
    a, b, c, d, e, f = (float(v) for v in list(inner_transform)[:6])
    if b or d:
        raise ValueError("rotated tile transform; the global lattice assumes north-up")
    gc, gr = c / native_m, -f / native_m
    rc, rr = round(gc), round(gr)
    if abs(gc - rc) * native_m > tol_m or abs(gr - rr) * native_m > tol_m:
        raise ValueError(
            f"tile origin ({c}, {f}) is off the Murray global native lattice by "
            f"({abs(gc - rc) * native_m:.3e}, {abs(gr - rr) * native_m:.3e}) m"
        )
    return int(rr), int(rc)


def tile_grid_phase(inner_transform, tile_px: int = 32, **kw) -> tuple[int, int]:
    """Local `(row, col)` pixel at which this tile's first GLOBAL coarse cell begins.

    **Convention, pinned deliberately** — this returns `(-global_origin) % tile_px`, i.e.
    how far into the tile you must step to reach a global cell boundary. Over the footprint
    that is `{16, 20, 24, 28}` in row for `{N44, N40, N36, N32}`. The complementary
    quantity `global_origin % tile_px` gives `{16, 12, 8, 4}` and is **not** what the
    callers want; cross-wiring the two is a real and tested-for mutant, and it is invisible
    on an N44 tile where both are 16.
    """
    gr, gc = global_native_origin(inner_transform, **kw)
    return (-gr) % tile_px, (-gc) % tile_px


def global_cell_transform(cell_row: int, cell_col: int, tile_px: int = 32, *,
                          native_m: float = MURRAY_NATIVE_M):
    """Affine of a coarse raster whose top-left cell is the global cell `(row, col)`."""
    from rasterio.transform import Affine

    cell = tile_px * native_m
    return Affine(cell, 0.0, cell_col * cell, 0.0, -cell, -cell_row * cell)


def assert_shared_lattice(transforms, *, tile_px: int = 32,
                          native_m: float = MURRAY_NATIVE_M, tol_cell: float = 1e-6) -> None:
    """Every transform must sit on the one global coarse lattice. Raises otherwise.

    This is the acceptance gate for a merged product, and it is pure geometry — it needs no
    inference, so it can be run before spending GPU-hours. Against the currently shipped
    26 tiles it fails 26/26, which is the defect stated executably.
    """
    cell = tile_px * native_m
    bad = []
    for i, t in enumerate(transforms):
        a, b, c, d, e, f = (float(v) for v in list(t)[:6])
        if b or d:
            bad.append((i, "rotated", None, None))
            continue
        if abs(abs(a) - cell) > tol_cell * cell or abs(abs(e) - cell) > tol_cell * cell:
            bad.append((i, "cell size", a, e))
            continue
        rj, ri = c / cell, -f / cell
        if abs(rj - round(rj)) > tol_cell or abs(ri - round(ri)) > tol_cell:
            bad.append((i, "phase", rj - round(rj), ri - round(ri)))
    if bad:
        detail = "; ".join(f"[{i}] {why} {x} {y}" for i, why, x, y in bad[:8])
        raise ValueError(
            f"{len(bad)} of {len(transforms)} rasters are not on {COARSE_GRID_ID}: {detail}"
            + ("" if len(bad) <= 8 else f" ... and {len(bad) - 8} more")
        )


def assert_coregistered(transform_a, transform_b, *, shape_a=None, shape_b=None,
                        name_a: str = "a", name_b: str = "b", tol_m: float = 1e-3) -> None:
    """Two rasters must be cell-for-cell aligned before they are compared **by index**.

    R01 made this necessary rather than pedantic. The corrected regional mosaic keeps the
    shipped shape (5925 x 11852) but its origin moves +100 m E / -80 m S, while
    `cache_v2/validation/themis_night_ir_region.tif` was fetched `--match-mosaic` against the
    *old* transform. Same dtype, same shape, different ground position: `ab[good]` vs
    `ti[good]` would keep running and quietly correlate cells 0.625 of a cell apart. Equal
    shapes are the trap, not the reassurance.
    """
    a = [float(v) for v in list(transform_a)[:6]]
    b = [float(v) for v in list(transform_b)[:6]]
    bad = []
    if (a[0], a[4]) != (b[0], b[4]):
        bad.append(f"pixel size {(a[0], a[4])} vs {(b[0], b[4])}")
    if abs(a[2] - b[2]) > tol_m or abs(a[5] - b[5]) > tol_m:
        bad.append(f"origin ({a[2]:.4f}, {a[5]:.4f}) vs ({b[2]:.4f}, {b[5]:.4f}) "
                   f"-- offset ({b[2] - a[2]:+.1f}, {b[5] - a[5]:+.1f}) m")
    if shape_a is not None and shape_b is not None and tuple(shape_a) != tuple(shape_b):
        bad.append(f"shape {tuple(shape_a)} vs {tuple(shape_b)}")
    if bad:
        raise ValueError(
            f"{name_a} and {name_b} are not co-registered, so comparing them by array index "
            f"is meaningless: " + "; ".join(bad) + ".\n"
            f"  Re-fetch or reproject one onto the other's grid before correlating."
        )


@dataclass(frozen=True)
class TileGlobalGrid:
    """One Murray tile's placement on the global coarse lattice — all fields **measured**.

    The only way to obtain one is `tile_global_grid()`, which reads the sphere radius out of
    the tile's own CRS and checks the origin against the native lattice before returning.
    That is deliberate: `COARSE_GRID_ID` asserts both `R3396190` and `ppd11855`, and a
    provenance field that asserts rather than measures has been caught four times on this
    project. `provenance()` is unreachable without those checks having passed.
    """

    cell_row0: int      # global cell index of this tile's first whole cell (row)
    cell_col0: int      # ... and column
    phase_r: int        # tile-local pixel at which that first global cell begins
    phase_c: int
    tile_px: int
    radius_m: float     # parsed out of the tile CRS, not assumed

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        """The `global_grid=` argument of `predict_window` — one tuple, deliberately."""
        return (self.cell_row0, self.cell_col0, self.phase_r, self.phase_c)

    def transform(self, cell_row: int, cell_col: int):
        """Affine of a raster whose top-left cell is the GLOBAL cell `(row, col)`."""
        return global_cell_transform(cell_row, cell_col, self.tile_px)

    def provenance(self) -> dict:
        """Grid identity for a product sidecar. Two products are on one lattice iff their
        `grid_id` and `grid_cell_m` match and their `cell_row0`/`cell_col0` differ by
        integers — which is checkable after the fact, unlike a bare boolean."""
        return {
            "grid_id": COARSE_GRID_ID,
            "grid_anchor": "lonlat0",
            "grid_ppd": MURRAY_PPD,
            "grid_radius_m": self.radius_m,
            "grid_native_m": MURRAY_NATIVE_M,
            "grid_cell_m": self.tile_px * MURRAY_NATIVE_M,
            "grid_tile_px": self.tile_px,
            "cell_row0": self.cell_row0,
            "cell_col0": self.cell_col0,
            "grid_phase_px": [self.phase_r, self.phase_c],
        }


def tile_global_grid(inner_transform, crs_wkt: str | None, tile_px: int = 32,
                     **kw) -> TileGlobalGrid:
    """Place a Murray tile on the global coarse lattice, verifying every step.

    `phase` is where the tile's first *global* cell boundary falls in tile-local pixels;
    `cell_*0` is that cell's global index, so a tile-local cell index `t` (as enumerated by
    `tile_grid_for_window` from the phase-shifted origin) is global cell `cell_*0 + t`.
    """
    radius = assert_murray_sphere(crs_wkt)
    gr, gc = global_native_origin(inner_transform, **kw)
    phase_r, phase_c = (-gr) % tile_px, (-gc) % tile_px
    # Exact division or nothing: `(origin + phase)` is a whole number of cells by
    # construction, and asserting it here is what makes the flipped-sign phase (mutant M6)
    # fail loudly at N32/N36/N40 instead of silently mis-centring every context box.
    if (gr + phase_r) % tile_px or (gc + phase_c) % tile_px:
        raise ValueError(
            f"phase ({phase_r}, {phase_c}) does not land origin ({gr}, {gc}) on a cell "
            f"boundary; the phase convention is (-origin) % tile_px, not its complement."
        )
    return TileGlobalGrid(cell_row0=(gr + phase_r) // tile_px,
                          cell_col0=(gc + phase_c) // tile_px,
                          phase_r=phase_r, phase_c=phase_c, tile_px=tile_px,
                          radius_m=radius)


# ============================================================================
# Read-window sweep
# ============================================================================


def window_offsets(extent: int, win: int, overlap: int, tile_px: int, *,
                   tile_aligned: bool = True) -> list[int]:
    """Read-window start offsets covering `[0, extent)` with `overlap`.

    `step = win - overlap`; the final offset closes the run at the far edge.

    **`tile_aligned` (R01).** The legacy contract was that every offset is a multiple of
    `tile_px`, which mattered when the coarse grid was anchored to the window sweep. On the
    globally anchored grid the cell lattice has its own phase, and a tile-aligned final
    offset then leaves holes: measured over `extent=47420, win=4096, tile_px=32`, the
    shipped configuration loses **11 computable cells per axis at all seven non-zero
    phases**, and each half of the fix alone still loses some —

        tile_aligned=True,  overlap=2*tile_px   ->  11 lost/axis   (shipped)
        tile_aligned=True,  overlap=3*tile_px   ->   1 lost/axis
        tile_aligned=False, overlap=2*tile_px   ->  10 lost/axis
        tile_aligned=False, overlap=3*tile_px   ->   0 lost/axis

    — all four at 12 offsets per axis, so the fix is free (144 windows either way). Phase 0
    loses nothing in every configuration, which is why this never bit before.

    `tile_aligned=True` stays the default so `scripts/f_region_stageb.py`, which sweeps ISIS
    frame cubes on their own phase-0 grid, keeps its existing window set and per-window
    partial filenames. Pass `tile_aligned=False` on the global-grid map path, and check the
    result with `uncovered_cells` rather than trusting the arithmetic.
    """
    win = min(win, extent)
    step = max(tile_px, win - overlap)
    last = ((extent - win) // tile_px) * tile_px if tile_aligned else extent - win
    offs: list[int] = []
    o = 0
    while o < last:
        offs.append(o)
        o += step
    if not offs or offs[-1] != last:
        offs.append(last)
    return offs


def uncovered_cells(offsets, extent: int, win: int, tile_px: int, *,
                    phase: int = 0) -> list[int]:
    """Coarse cells that no window in `offsets` can compute. Measured, not assumed.

    Returns the tile-local start pixels of every cell that *could* be computed from the full
    tile (its 3x3 context box fits: `u >= tile_px` and `u + 2*tile_px <= extent`) but whose
    context box fits inside no single window. Cell starts sit at `u = phase (mod tile_px)`.

    This is the executable form of the coverage contract: the drivers call it after building
    their sweep and refuse to run if it is non-empty, so choosing the wrong `overlap` or
    `tile_aligned` fails before any GPU time is spent rather than punching a hole in the
    product. Pure arithmetic — microseconds.
    """
    phase = int(phase) % tile_px
    want = np.arange(tile_px + phase, extent - 2 * tile_px + 1, tile_px, dtype=np.int64)
    covered = np.zeros(want.size, dtype=bool)
    for o in offsets:
        o = int(o)
        h = min(win, extent - o)
        covered |= (want >= o + tile_px) & (want + 2 * tile_px <= o + h)
    return want[~covered].tolist()


# ============================================================================
# Windowed CTX read
# ============================================================================


@dataclass(frozen=True)
class CtxWindow:
    """A CTX sub-window plus everything needed to grid + georeference it.

    `data` is (H, W) uint8; `row_off`/`col_off` are its top-left pixel offset in
    the parent Murray tile (= the tile-anchored grid origin row0/col0);
    `transform` is the window's affine (6-tuple a,b,c,d,e,f); `crs_wkt` the tile CRS.
    """

    data: np.ndarray
    row_off: int
    col_off: int
    transform: tuple[float, ...]
    crs_wkt: str


def artifact_digest(path: str | Path) -> str | None:
    """Content digest of a model/calibration artifact: a file, or a whole directory.

    A `DeployableHead` is a directory and a `CalibrationLayer` is one `.npz`, so map
    provenance needs both shapes. Directories hash the sorted `(relative posix path,
    file sha256)` listing, so a renamed or added file changes the digest.

    Recording the *path* is not enough: the audit's requirement is that a tile sidecar and
    a region manifest identify the head and calibration by content, because a path can be
    overwritten in place and a directory name is only a recipe hash of the training
    configuration, not of the weights that came out of it. Returns None for a missing path
    so a raw-probability run (no calibrator) records `null` rather than crashing.
    """
    import hashlib

    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    if p.is_file():
        files = [p]
    else:
        files = sorted(q for q in p.rglob("*") if q.is_file())
    for q in files:
        if p.is_dir():
            h.update(q.relative_to(p).as_posix().encode("utf-8"))
        with open(q, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def read_tile_window(zip_path: str | Path, inner_tif: str, row_off: int, col_off: int,
                     size: int) -> CtxWindow:
    """Window-read a `size x size` uint8 block from `/vsizip/{zip}/{inner_tif}`.

    No full-tile materialization: rasterio reads only the requested window via the
    zip's internal tiling. Returns a `CtxWindow` carrying the read offset (grid
    origin), the window's affine, and the tile CRS.
    """
    import rasterio
    from rasterio.windows import Window

    vsizip = f"/vsizip/{Path(zip_path).as_posix()}/{inner_tif}"
    with rasterio.open(vsizip) as src:
        window = Window(col_off=int(col_off), row_off=int(row_off), width=int(size), height=int(size))
        data = src.read(1, window=window).astype(np.uint8, copy=False)
        wt = src.window_transform(window)
        crs_wkt = src.crs.to_wkt() if src.crs else ""
    return CtxWindow(data=data, row_off=int(row_off), col_off=int(col_off),
                     transform=tuple(wt)[:6], crs_wkt=crs_wkt)


# ============================================================================
# Own-tile validity (mask CTX nodata before trusting a prediction)
# ============================================================================


def as_nodata_mask(window: np.ndarray, nodata: np.ndarray | None = None) -> np.ndarray:
    """Boolean "this pixel is missing data" for a CTX array — supplied, or inferred as `== 0`.

    **R38.** Inferring nodata from the pixel VALUE is only safe while nothing downstream can
    synthesize that value. A1 could: it clipped to `[0, 255]`, so a legitimately dark pixel was
    written as the sentinel and thereafter counted as a data gap. A1 now floors valid pixels at
    `src.striping.A1_VALID_FLOOR`, but the deeper fix is that a caller which *knows* the true
    mask can hand it over instead of having it guessed from a transformed array.

    `nodata=None` keeps the inference, which is correct and exact for the raw Murray mosaic
    (`scripts/map_region.py`): its GeoTIFF declares `nodata=0` and the minimum valid DN is 1 in
    every tile strip sampled, because Murray bottom-clips valid data to 1.
    """
    if nodata is None:
        return window == 0
    nodata = np.asarray(nodata, dtype=bool)
    if nodata.shape != window.shape:
        raise ValueError(f"nodata mask {nodata.shape} does not match the window {window.shape}")
    return nodata


def own_tile_zero_fraction(window: np.ndarray, ti: np.ndarray, tj: np.ndarray, *,
                           tile_px: int, row0: int, col0: int,
                           nodata: np.ndarray | None = None) -> np.ndarray:
    """Per-tile fraction of own-tile CTX pixels that are nodata (Murray mosaic gap).

    A tile sitting in a mosaic data gap embeds black pixels and yields a
    meaningless prediction; the caller masks tiles whose zero-fraction is high.
    `ti, tj` are global tile indices; the own tile occupies window rows
    [ti*tile_px - row0, +tile_px) (CLAUDE.md grid anchor).

    `nodata` (R38) is the explicit mask; omitted, it is inferred as `window == 0`.
    """
    ti = np.asarray(ti, dtype=np.int64)
    tj = np.asarray(tj, dtype=np.int64)
    nd = as_nodata_mask(window, nodata)
    H, W = window.shape
    out = np.ones(ti.size, dtype=np.float32)
    r = ti * tile_px - row0
    c = tj * tile_px - col0
    for i in range(ti.size):
        if r[i] < 0 or c[i] < 0 or r[i] + tile_px > H or c[i] + tile_px > W:
            continue  # own tile outside window (shouldn't happen for enumerated grid)
        box = nd[r[i]: r[i] + tile_px, c[i]: c[i] + tile_px]
        out[i] = float(box.mean())
    return out


# Thresholds the per-tile context-nodata fraction is histogrammed against, so a future
# session can re-tune `max_context_zero_fraction` from the committed sidecars instead of
# re-running a ~0.6 GPU-h/tile pass. Counts are "cells with ctx_frac > edge".
CONTEXT_ZERO_HIST_EDGES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)


def context_zero_fraction(window: np.ndarray, ti: np.ndarray, tj: np.ndarray, *,
                          tile_px: int, row0: int, col0: int,
                          nodata: np.ndarray | None = None) -> np.ndarray:
    """Fraction of each tile's `3*tile_px` CONTEXT box that is CTX nodata (DN 0).

    **R13.** `own_tile_zero_fraction` tests the centre `tile_px²` only — 1024 of 9216 px at
    the frozen S=32, so **88.9 % of what the embedder actually sees is never checked**. The
    box here is exactly the one `src.fm_embeddings.slice_context_boxes` slices, and the
    validity rule is bit-identical to its `valid` (a box that spills the window returns 1.0,
    which the caller's `valid` mask has already dropped anyway).

    Measured impact of admitting a dirty context, real frozen ViT + real shipped head against
    the shipped E4_N44 IQR of 0.152: one whole blackened 32-block in the ring moves p90 |ΔP|
    by 0.45 (≈3× IQR), and 92 *scattered* black pixels by 0.70 (≈4.6×). Shape matters more
    than count. Cost on the shipped map is 290 of 19,685,689 measured cells (1.5e-05).

    **Lattice-block form, deliberately.** Every enumerated cell start is congruent to
    `-row0 (mod tile_px)`, so the window crops to a whole number of cells and a
    reshape-sum gives a small per-cell zero-count grid; the 3×3 context sum is then an
    integral image over *that*. Benchmarked on the production 4096² / 15,876-cell geometry:
    **0.016 s and +18 MB**, against 0.44 s / +419 MB for a full-resolution int64 integral
    image (an earlier draft of this fix, which would have been 1.4× *slower* than the
    own-tile loop it was claimed to subsume).
    """
    ti = np.asarray(ti, dtype=np.int64)
    tj = np.asarray(tj, dtype=np.int64)
    out = np.ones(ti.size, dtype=np.float32)
    if ti.size == 0:
        return out
    H, W = window.shape
    row0, col0, side = int(row0), int(col0), 3 * tile_px
    # First cell boundary inside the window, and how many whole cells fit after it.
    pr, pc = (-row0) % tile_px, (-col0) % tile_px
    n_br, n_bc = (H - pr) // tile_px, (W - pc) // tile_px
    if n_br < 3 or n_bc < 3:
        return out                       # no cell in this window can carry a full context box
    nd = as_nodata_mask(window, nodata)
    z = np.ascontiguousarray(nd[pr: pr + n_br * tile_px, pc: pc + n_bc * tile_px])
    counts = z.reshape(n_br, tile_px, n_bc, tile_px).sum(axis=(1, 3), dtype=np.int64)
    ii = np.zeros((n_br + 1, n_bc + 1), dtype=np.int64)
    ii[1:, 1:] = counts.cumsum(axis=0).cumsum(axis=1)
    br = (ti * tile_px - row0 - pr) // tile_px
    bc = (tj * tile_px - col0 - pc) // tile_px
    ok = (br >= 1) & (bc >= 1) & (br + 2 <= n_br) & (bc + 2 <= n_bc)
    r, c = br[ok] - 1, bc[ok] - 1
    out[ok] = ((ii[r + 3, c + 3] - ii[r, c + 3] - ii[r + 3, c] + ii[r, c])
               / float(side * side))
    return out


def context_zero_histogram(ctx_frac: np.ndarray, valid: np.ndarray,
                           edges=CONTEXT_ZERO_HIST_EDGES) -> np.ndarray:
    """Counts of `ctx_frac > edge` among `valid` cells, one per `CONTEXT_ZERO_HIST_EDGES`."""
    ctx_frac = np.asarray(ctx_frac)
    valid = np.asarray(valid, dtype=bool)
    return np.array([int((valid & (ctx_frac > float(e))).sum()) for e in edges],
                    dtype=np.int64)


# ============================================================================
# (ti, tj) -> raster placement + the 32x-coarsened affine
# ============================================================================


def tiles_to_raster(ti: np.ndarray, tj: np.ndarray, values: np.ndarray,
                    *, fill: float = np.nan) -> tuple[np.ndarray, int, int]:
    """Scatter per-tile `values` into a dense (n_ti, n_tj) raster.

    Returns `(raster, ti_min, tj_min)`. Rows index `ti` (north-south), cols index
    `tj` (east-west); `(ti_min, tj_min)` anchor the raster to the tile grid so the
    affine can be reconstructed. Tiles not present stay `fill` (the enumerated grid
    is gap-free, so this only matters when the caller passes a subset).
    """
    ti = np.asarray(ti, dtype=np.int64)
    tj = np.asarray(tj, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    ti_min, ti_max = int(ti.min()), int(ti.max())
    tj_min, tj_max = int(tj.min()), int(tj.max())
    raster = np.full((ti_max - ti_min + 1, tj_max - tj_min + 1), fill, dtype=np.float64)
    raster[ti - ti_min, tj - tj_min] = values
    return raster, ti_min, tj_min


def tile_origin_transform(window_transform: tuple[float, ...], row_off: int,
                          col_off: int) -> tuple[float, ...]:
    """Reconstruct the PARENT TILE affine from a window's affine + its read offset.

    A window read at pixel (row_off, col_off) has origin
    `c_win = c_tile + col_off*a + row_off*b`, `f_win = f_tile + col_off*d + row_off*e`.
    The tile-anchored `(ti, tj)` grid needs the tile origin, so invert that. Without
    this the window offset is double-counted (the window affine already carries it,
    and `coarsened_transform` adds `tj_min*tile_px` on top).
    """
    a, b, c, d, e, f = (window_transform[i] for i in range(6))
    c_tile = c - col_off * a - row_off * b
    f_tile = f - col_off * d - row_off * e
    return (a, b, c_tile, d, e, f_tile)


def coarsened_transform(tile_transform: tuple[float, ...], ti_min: int, tj_min: int,
                        tile_px: int):
    """Affine for the per-tile raster: tile_px-coarsened, origin at (ti_min, tj_min).

    `tile_transform` is the PARENT TILE's affine (a,b,c,d,e,f). Output pixel (0,0)
    is the top-left of tile (ti_min, tj_min) -> mosaic pixel (ti_min*tile_px,
    tj_min*tile_px), and each output pixel spans tile_px source pixels (160 m at
    tile_px=32, 5 m/px). Returns a `rasterio.Affine`.
    """
    from rasterio.transform import Affine

    a, b, c, d, e, f = (tile_transform[i] for i in range(6))
    x0 = c + (tj_min * tile_px) * a + (ti_min * tile_px) * b
    y0 = f + (tj_min * tile_px) * d + (ti_min * tile_px) * e
    return Affine(a * tile_px, b, x0, d, e * tile_px, y0)


def mosaic_geotiffs(paths, out_path: str | Path | None = None, *,
                    tile_px: int = 32, require_shared_lattice: bool = True):
    """Merge same-CRS single-band GeoTIFFs into one raster (Stage: regional mosaic).

    The per-tile `map_region` outputs all share the Murray global equirectangular CRS
    (`clon_0`), so a straight merge stitches them — no reprojection. Returns
    `(array2d, transform, crs_wkt)`; NaN fills any uncovered gap (the block is an L-shape,
    so two corners are nodata). Writes `out_path` if given.

    **R01.** The old docstring said the inputs differ "only in extent". They do not: each
    tile carried its own sub-cell phase, and `rasterio.merge` floors each fractional
    destination offset, so the phase became a whole-cell displacement — 25 of 26 shipped
    tiles, median 140 m. `require_shared_lattice` therefore refuses to merge rasters that
    are not on the one global lattice. It **fails loudly on the currently shipped tiles by
    design**; pass `require_shared_lattice=False` to reproduce a pre-R01 product knowingly.
    """
    import rasterio
    from rasterio.merge import merge

    paths = [str(p) for p in paths]
    srcs = [rasterio.open(p) for p in paths]
    try:
        if require_shared_lattice:
            try:
                assert_shared_lattice([s.transform for s in srcs], tile_px=tile_px)
            except ValueError as exc:
                raise ValueError(
                    f"{exc}\nMerging these would bake each tile's own sub-cell phase into a "
                    "whole-cell displacement. Re-render on the global grid, or pass "
                    "require_shared_lattice=False to reproduce the pre-R01 product."
                ) from None
        arr, transform = merge(srcs, nodata=np.nan)
        crs_wkt = srcs[0].crs.to_wkt() if srcs[0].crs else ""
    finally:
        for s in srcs:
            s.close()
    arr = arr[0]  # single band
    if out_path is not None:
        write_geotiff(out_path, arr, transform, crs_wkt)
    return arr, transform, crs_wkt


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_geotiff(path: str | Path, *, expect_shape=None, expect_count: int = 1,
                   expect_dtype: str = "float32", expect_finite: int | None = None,
                   expect_bytes: int | None = None,
                   expect_sha256: str | None = None) -> str | None:
    """Is this GeoTIFF actually complete? Returns None if acceptable, else the reason.

    **R14.** A killed write leaves three distinct signatures, and the obvious checks catch
    only some of them — all measured on real `reports/map_region` rasters:

    1. **Truncated file.** `rasterio.open` SUCCEEDS at 10/50/90/99/99.99 % truncation and
       reports the correct shape and dtype, because the first IFD sits at byte 8. A full
       decode raises; so does reading the last block.
    2. **Valid but all-nodata.** A closed 100 %-NaN raster opens, decodes cleanly, and reads
       its last block fine. Nothing structural distinguishes it from a legitimately-masked
       tile — NaN *is* this product's nodata. Only a finite-count check sees it.
    3. **Half the blocks written.** Opens, decodes, last block OK, finite fraction 0.5193 with
       the first all-nodata row at 768. Again only the finite count catches it.

    So `expect_finite` is not decoration: it is the only test that sees signatures 2 and 3,
    and those are exactly what a Slurm wall-clock kill produces. Checks run cheapest-first.
    The decode is **blockwise** — `mosaic_geotiffs` already holds a 281 MB array in memory and
    a naive `read(1)` here would double it.
    """
    import rasterio

    path = Path(path)
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if expect_bytes is not None and size != expect_bytes:
        return f"size {size} != expected {expect_bytes}"
    if expect_sha256 is not None:
        got = file_sha256(path)
        if got != expect_sha256:
            return f"sha256 {got[:12]}… != expected {expect_sha256[:12]}…"
    try:
        with rasterio.open(path) as src:
            if expect_count is not None and src.count != expect_count:
                return f"band count {src.count} != {expect_count}"
            if expect_dtype is not None and src.dtypes[0] != expect_dtype:
                return f"dtype {src.dtypes[0]} != {expect_dtype}"
            if expect_shape is not None and (src.height, src.width) != tuple(expect_shape):
                return f"shape {(src.height, src.width)} != {tuple(expect_shape)}"
            n_finite = 0
            for _, win in src.block_windows(1):
                n_finite += int(np.isfinite(src.read(1, window=win)).sum())
    except Exception as exc:                             # noqa: BLE001
        return f"unreadable ({type(exc).__name__}: {exc})"
    if expect_finite is not None and n_finite != expect_finite:
        return f"{n_finite} finite pixels != expected {expect_finite}"
    return None


def write_geotiff(path: str | Path, raster: np.ndarray, transform, crs_wkt: str,
                  *, nodata: float = np.nan, atomic: bool = True,
                  verify: bool = True) -> Path:
    """Write a single-band float32 GeoTIFF (the abundance/probability raster).

    **R14 — atomic.** `rasterio.open(path, "w")` occupies the *destination* for the whole
    write: measured, the file existed at 7,799,350 of 7,854,955 bytes before `close()`, and a
    hard-killed 12000² write left 154 MB sitting at the final path. Worse, it **truncates an
    existing destination immediately**, so a re-run that dies mid-write destroys the good tile
    it was going to replace. The raster is therefore staged as a `.tmp` *sibling* (same volume
    — `Path.replace` is only atomic within a volume), verified, and renamed. On any failure the
    `.tmp` is removed and the destination is left exactly as it was.

    `expect_finite` is computed on the **float32 cast**, not the caller's array: the cast can
    turn a finite float64 into `inf` (`1e300` → `inf`), which would make this reject its own
    correct output. `atomic=False` restores the old behaviour for callers that need it.
    """
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = raster.astype(np.float32)
    n_finite = int(np.isfinite(data).sum())
    dest = path if not atomic else path.with_name(path.name + ".tmp")
    try:
        with rasterio.open(
            dest, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
            count=1, dtype="float32", crs=crs_wkt or None, transform=transform,
            nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        ) as dst:
            dst.write(data, 1)
        if verify:
            why = verify_geotiff(dest, expect_shape=data.shape, expect_finite=n_finite)
            if why is not None:
                raise OSError(f"{path.name} failed post-write verification: {why}")
        if atomic:
            Path(dest).replace(path)
    except BaseException:
        if atomic:
            Path(dest).unlink(missing_ok=True)
        raise
    return path


# ============================================================================
# High-level one-window inference (geometry only; model/embedder injected)
# ============================================================================


@dataclass
class WindowPrediction:
    """Result of predicting one CTX window: tile keys + probability raster.

    With a Stage-1 ``calibrator`` (see ``predict_window``), ``prob``/``raster`` carry the
    **calibrated** rich/poor probability, ``prob_raw`` keeps the uncalibrated value for
    QA, and ``abundance``/``abundance_raster`` carry the de-compressed Tier-2 abundance
    (one-model quantile-match of the raw ``P(rich)``). Without a calibrator the
    abundance fields are ``None`` and ``prob`` is raw — backward-compatible.
    """

    ti: np.ndarray
    tj: np.ndarray
    prob: np.ndarray            # per-tile probability (calibrated if calibrator given), NaN where masked
    raster: np.ndarray          # (n_ti, n_tj) prob, NaN nodata
    ti_min: int
    tj_min: int
    transform: object           # rasterio Affine for `raster`
    crs_wkt: str
    n_valid: int
    n_masked_nodata: int
    calibrated: bool = False
    prob_raw: np.ndarray | None = None          # uncalibrated P(rich), kept for QA
    abundance: np.ndarray | None = None         # per-tile fractional_area (one-model)
    abundance_raster: np.ndarray | None = None  # (n_ti, n_tj) abundance, NaN nodata
    # --- R13: the context gate. Separate from the own-tile counter ON PURPOSE ---
    # Folding both drops into `n_masked_nodata` would leave the sidecar under-reporting
    # exactly as it does today, which is the half of the finding the register calls "Record".
    n_masked_context_nodata: int = 0
    # The thresholds that produced the two counters, so a product can be read without
    # guessing which gate made it. `None` = a caller that predates the field.
    max_zero_fraction: float | None = None
    max_context_zero_fraction: float | None = None
    # (n, 2) GLOBAL cell indices of what each gate dropped. Stored rather than counted
    # because read windows can share a cell at phase 0 (one per axis seam), so summing
    # per-window counters over a tile is not exact; a de-duplicated cell SET is.
    masked_own_cells: np.ndarray | None = None
    masked_context_cells: np.ndarray | None = None
    # counts of ctx_frac > CONTEXT_ZERO_HIST_EDGES among `valid` cells
    context_zero_hist: np.ndarray | None = None


def predict_window(window: CtxWindow, embedder, head, *, tile_px: int = 32,
                   pool: str = "gem", batch: int = 96,
                   max_zero_fraction: float = 0.3,
                   max_context_zero_fraction: float = 0.0, calibrator=None,
                   apply_isotonic: bool = True,
                   global_grid: tuple[int, int, int, int] | None = None,
                   nodata_mask: np.ndarray | None = None) -> WindowPrediction:
    """Embed -> predict -> (optionally calibrate) -> rasterize one CTX window.

    `embedder` is a `src.fm_embeddings.FangEmbedder`; `head` exposes
    `predict(emb)->prob` (`DeployableHead`). Tiles whose context box spills the
    window edge (embed returns NaN) or whose own-tile CTX is >`max_zero_fraction`
    nodata are masked (prob NaN). Returns the dense raster + its affine.

    **R13 — `max_context_zero_fraction`, and the own-tile default moved to 0.3.** The
    own-tile gate tests 1024 of the 9216 pixels the embedder consumes; a tile whose own
    32² is spotless can sit against a mosaic gap and still be embedded almost entirely
    black. `max_context_zero_fraction` gates the full `3*tile_px` box.

    The default is **0.0** — compared with `<=`, so "not one nodata pixel in the context".
    Two measured legs, and only two: (i) the frozen head's training set contains **0**
    nodata pixels in 161,005 context boxes, so 0.0 is the only value that reproduces the
    distribution it was fitted on; (ii) on the shipped 26-tile map it costs 290 of
    19,685,689 measured cells (1.5e-05; hard ceiling 1,167 map-wide), so false rejection is
    free. It is *conservative rather than forced*: at exactly one nodata pixel there is no
    measurable sentinel-specific signal — DN 0 and the perfectly legal DN 1 move the
    prediction identically to three decimals, because the damage is caused by blackness,
    not by the sentinel. That also means **this gate cannot see a radiometrically blackened
    pixel whose value is not 0** (see R38 on A1's `[0,255]` clip), and a driver that
    normalises DN before inference must not assume it can.

    `max_zero_fraction`'s signature default was **0.5** while every production driver passed
    0.3; `scripts/parity_check.py` took the signature default, so the one cross-machine gate
    exercised a threshold nothing shipped with. It is now 0.3.

    **R38 — `nodata_mask`.** Both gates ask "is this pixel missing data?", and until now that
    was answered by testing the pixel's VALUE against 0 on whatever array arrived. That is exact
    for the raw Murray mosaic, whose GeoTIFF declares `nodata=0` and whose minimum valid DN is 1
    (Murray bottom-clips valid data). It was **not** exact for the A1 path, which clipped to
    `[0, 255]` and so manufactured the sentinel out of legitimately dark terrain. A caller that
    transforms the DN should pass the mask it computed from the *untransformed* array; `None`
    keeps the inference. Note the mask answers coverage only — a pixel blackened by A1's clip is
    a *radiometric* problem and is deliberately not representable here (see
    `src.striping.a1_clip_counts`, which counts it separately).

    `calibrator` is an optional Stage-1 `src.calibration.CalibrationLayer`. When given,
    an **abundance** raster `calibrate_abundance(raw P(rich))` (the one-model
    quantile-match — the de-compression win) is added, and the rich/poor raster is
    isotonic-calibrated unless `apply_isotonic=False` (the isotonic ECE polish is a
    rank-safe gate-clear, not a per-image-significant win, so it is toggleable). The
    raw probability is always kept in `prob_raw`. `calibrator=None` (default) renders
    raw, unchanged — the raw/calibrated toggle.

    **R01 — `global_grid`.** `(cell_row0, cell_col0, phase_r, phase_c)` puts this window's
    tiles on the one globally anchored coarse lattice instead of the parent tile's own.
    `phase_*` shifts the grid origin so cell boundaries land on the global lattice;
    `cell_*0` converts the resulting tile-local `(ti, tj)` into global cell indices, and
    the output transform is built from the global cell rather than the parent-tile origin.

    Those two halves are **inseparable and passed as one argument on purpose**: making
    `(ti, tj)` global while still deriving the transform from the parent-tile origin
    multiplies a ~-16,000 index against that origin and lands the raster ~2,600 km away.
    A sentinel like `cell_offset != (0, 0)` would make that a data-dependent coupling; one
    optional tuple makes it structural. `None` (default) keeps the legacy tile-anchored
    behaviour, so `map_pilot.py` and the existing tests are untouched.
    """
    from src.fm_embeddings import tile_grid_for_window

    arr = window.data
    row0, col0 = window.row_off, window.col_off
    cell_row0 = cell_col0 = 0
    if global_grid is not None:
        cell_row0, cell_col0, phase_r, phase_c = global_grid
        # Shift the grid origin back to the previous global cell boundary. Everything
        # downstream that indexes the window (`embed_window`, `own_tile_zero_fraction`)
        # takes this shifted, still-LOCAL row0/col0 -- see the note before the += below.
        row0, col0 = row0 - phase_r, col0 - phase_c
    ti, tj = tile_grid_for_window(arr.shape, row0, col0, tile_px)
    emb, valid = embedder.embed_window(arr, ti, tj, tile_px=tile_px, row0=row0,
                                       col0=col0, pool=pool, batch=batch)

    # R38: one mask, computed once, used by both gates — supplied by a caller that transformed
    # the DN (the A1 path), inferred as `arr == 0` for the raw mosaic.
    nd = as_nodata_mask(arr, nodata_mask)
    zero_frac = own_tile_zero_fraction(arr, ti, tj, tile_px=tile_px, row0=row0, col0=col0,
                                       nodata=nd)
    ctx_frac = context_zero_fraction(arr, ti, tj, tile_px=tile_px, row0=row0, col0=col0,
                                     nodata=nd)
    own_ok = zero_frac <= max_zero_fraction
    ctx_ok = ctx_frac <= max_context_zero_fraction
    usable = valid & own_ok & ctx_ok
    # R13: attribute each drop to the gate that made it, and keep the histogram of what the
    # context gate saw. Computed here because `predict_window` is the only place that has
    # both fractions; thrown away here is how the shipped sidecars ended up with neither.
    ctx_hist = context_zero_histogram(ctx_frac, valid)

    # R01: only NOW convert to global cell indices. Everything above indexes the window as
    # `ti*tile_px - row0` with a LOCAL row0, so promoting ti early (to ~-16,300 against a
    # local row0 of 0..47420) drives the slice origin to ~-521,600: `valid` goes all-False,
    # every prob is NaN, the partial is empty, and assembly dies on `ti.min()` of an empty
    # array. This ordering is load-bearing and is covered by a test.
    if global_grid is not None:
        ti = ti + cell_row0
        tj = tj + cell_col0

    prob = np.full(ti.size, np.nan, dtype=np.float64)
    if usable.any():
        prob[usable] = head.predict(emb[usable])

    prob_raw = abundance = abundance_raster = None
    if calibrator is not None:
        prob_raw = prob.copy()                       # keep the uncalibrated value for QA
        abundance = np.full(ti.size, np.nan, dtype=np.float64)
        cal = np.full(ti.size, np.nan, dtype=np.float64)
        if usable.any():
            # both maps consume the RAW P(rich) (one-model): isotonic -> calibrated prob,
            # qmatch -> abundance. Compute before overwriting `prob`.
            cal[usable] = calibrator.calibrate_prob(prob_raw[usable])
            abundance[usable] = calibrator.calibrate_abundance(prob_raw[usable])
        # abundance (qmatch) is always applied; isotonic on the rich/poor map is optional
        prob = cal if apply_isotonic else prob_raw.copy()

    raster, ti_min, tj_min = tiles_to_raster(ti, tj, prob, fill=np.nan)
    if calibrator is not None:
        abundance_raster, _, _ = tiles_to_raster(ti, tj, abundance, fill=np.nan)
    if global_grid is not None:
        # ti_min/tj_min are already GLOBAL cell indices, so the affine comes straight from
        # the global lattice. Deriving it from the parent tile here instead is the ~2,600 km
        # error described in the docstring.
        transform = global_cell_transform(ti_min, tj_min, tile_px)
    else:
        # Legacy: (ti, tj) are anchored to the parent Murray tile; rebuild the tile origin
        # so the window offset isn't double-counted (it already lives in window.transform).
        tile_transform = tile_origin_transform(window.transform, row0, col0)
        transform = coarsened_transform(tile_transform, ti_min, tj_min, tile_px)
    dropped_own = valid & ~own_ok
    dropped_ctx = valid & own_ok & ~ctx_ok
    return WindowPrediction(
        ti=ti, tj=tj, prob=prob, raster=raster, ti_min=ti_min, tj_min=tj_min,
        transform=transform, crs_wkt=window.crs_wkt,
        n_valid=int(valid.sum()),
        # R13: `(valid & ~usable)` would now absorb the context drops into the own-tile
        # counter, so the sidecar would keep under-reporting exactly as it does today.
        n_masked_nodata=int(dropped_own.sum()),
        n_masked_context_nodata=int(dropped_ctx.sum()),
        max_zero_fraction=float(max_zero_fraction),
        max_context_zero_fraction=float(max_context_zero_fraction),
        masked_own_cells=np.stack([ti[dropped_own], tj[dropped_own]], axis=1),
        masked_context_cells=np.stack([ti[dropped_ctx], tj[dropped_ctx]], axis=1),
        context_zero_hist=ctx_hist,
        calibrated=calibrator is not None, prob_raw=prob_raw,
        abundance=abundance, abundance_raster=abundance_raster,
    )
