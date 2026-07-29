"""PLAN_FBuild Stage D core — composite per-frame leveled logits onto the map grid.

Stage B emits per-frame sparse tile lists keyed to an exact-160.0 m GLOBAL lattice anchored at the
CRS origin; the existing mosaic-path map lives on a per-Murray-tile grid whose pitch is
**159.9991835298017 m** (= 32 x the Murray native 4.9999744853063 m/px) with per-tile origins that
are not multiples of 160. Those are two different lattices, and PLAN_FBuild §1 requires the F map to
ship on the mosaic one so notebook 24 and the validation harness work unchanged.

Measured 2026-07-28 (`tile_index_map`, asserted at runtime for every tile): within any one Murray
tile the relation is an EXACT CONSTANT INTEGER SHIFT,

    TJ = col + Kj        TI = Ki - row          (TI increases NORTHWARD; row does not)

so placing global tiles needs no interpolation at all. What it does carry is a fixed sub-pixel
TRANSLATION between the two lattices — the offset of a map pixel centre from the global node it maps
to runs 6.0-80.0 m in x and 7.9-50.3 m in y depending on the tile, and the E0 lon column sits
1.2 mm from a half-cell tie. That translation is real and is reported per tile (`dx_m`/`dy_m`) rather
than hidden; it is well inside the project's own O(200 m) HiRISE-CTX registration budget (CLAUDE.md).

Composite rule (PLAN_FBuild §5, reference implementation `scripts/f_h4_legb_perframe.composites`):

    p = sigmoid( mean_f [ logit(prob_f) + o_f ] )

mean over the frames covering that tile, in LOGIT space, with a single sigmoid at the end — NOT a
mean of probabilities and NOT a median. `src.leveling.logit/sigmoid` (EPS = 1e-4) are used verbatim
so the composite composes exactly with the offsets Stage C solved.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src import leveling as lv

REPO = Path(__file__).resolve().parents[1]

# H6 offset-provenance severity codes (worst contributor wins per pixel; PLAN §1 deliverable 2)
OFFSET_SOURCE_CODE = {"solved": 0, "component_gauged": 1, "interpolated": 2, "none": 3}
OFFSET_SOURCE_NAME = {v: k for k, v in OFFSET_SOURCE_CODE.items()}


# --------------------------------------------------------------------------- the grid
@dataclass(frozen=True)
class TileGrid:
    """A Murray tile's coarse (160 m-ish) prediction grid + its exact map to the global lattice."""
    tile: str
    transform: tuple            # 6-tuple (a, b, c, d, e, f) of the coarse affine
    height: int
    width: int
    crs_wkt: str
    Kj: int                     # TJ = col + Kj
    Ki: int                     # TI = Ki - row
    dx_m: float                 # max |x_centre(col) - TJ*160| over the tile (sub-pixel translation)
    dy_m: float                 # max |y_centre(row) - TI*160|
    tie_margin_m: float         # distance of the worst pixel from a half-cell (80 m) rounding tie

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def cols_of_TJ(self, TJ: np.ndarray) -> np.ndarray:
        return np.asarray(TJ, dtype=np.int64) - self.Kj

    def rows_of_TI(self, TI: np.ndarray) -> np.ndarray:
        return self.Ki - np.asarray(TI, dtype=np.int64)

    def TJ_range(self) -> tuple[int, int]:
        return (self.Kj, self.Kj + self.width - 1)

    def TI_range(self) -> tuple[int, int]:
        return (self.Ki - (self.height - 1), self.Ki)


def tile_index_map(transform, height: int, width: int, tile: str = "?") -> tuple[int, int, float, float, float]:
    """(Kj, Ki, dx_m, dy_m, tie_margin_m) for one coarse grid; raises if the shift is not constant.

    Failing loudly here is deliberate: a non-constant shift would mean the two lattices differ by
    more than a translation, and every downstream raster would be silently warped.
    """
    a, _b, c, _d, e, f = tuple(transform)[:6]
    col = np.arange(width)
    row = np.arange(height)
    xc = c + (col + 0.5) * a
    yc = f + (row + 0.5) * e
    TJ = np.round(xc / lv.TILE_M).astype(np.int64)
    TI = np.round(yc / lv.TILE_M).astype(np.int64)
    Kj, Ki = TJ - col, TI + row
    if not ((Kj == Kj[0]).all() and (Ki == Ki[0]).all()):
        raise ValueError(f"{tile}: global-lattice shift is not constant across the tile "
                         f"(Kj spans {Kj.min()}..{Kj.max()}, Ki spans {Ki.min()}..{Ki.max()}) — "
                         f"the grids differ by more than a translation; do not composite blindly")
    dx = float(np.abs(xc - TJ * lv.TILE_M).max())
    dy = float(np.abs(yc - TI * lv.TILE_M).max())
    return int(Kj[0]), int(Ki[0]), dx, dy, float(lv.TILE_M / 2 - dx)


def tile_grid_from_raster(path: str | Path, tile: str) -> TileGrid:
    """Read a reference map raster (the mosaic-path tif) and derive the Stage-D grid from it.

    Reading the existing product rather than re-deriving from the Murray sidecar is deliberate: only
    9 of the 26 block tiles have a cached `cache_v2/ctx_tiles/{tile}.json`, but all 26 have a map tif,
    and byte-compatibility with THAT file is what PLAN §1 actually asks for.
    """
    import rasterio

    with rasterio.open(path) as ds:
        transform = tuple(ds.transform)[:6]
        h, w = ds.height, ds.width
        crs_wkt = ds.crs.to_wkt() if ds.crs else ""
    Kj, Ki, dx, dy, tie = tile_index_map(transform, h, w, tile)
    return TileGrid(tile=tile, transform=transform, height=h, width=w, crs_wkt=crs_wkt,
                    Kj=Kj, Ki=Ki, dx_m=dx, dy_m=dy, tie_margin_m=tie)


# --------------------------------------------------------------------------- accumulators
@dataclass
class TileAccum:
    """Streaming per-pixel accumulators for one Murray tile (one pass per contributing frame).

    Only quantities that are exactly computable in one pass are kept. In particular the H6
    overlap-QA layer wants "max co-located |Δp| after leveling", and

        max over frame PAIRS |p_i - p_j|  ==  max_f p_f - min_f p_f

    exactly, so an O(k) running min/max is not an approximation of the O(k²) pairwise max — it IS it.
    """
    shape: tuple[int, int]
    sum_logit: np.ndarray       # float64 running sum of leveled logits
    n_frames: np.ndarray        # int16 count of contributing frames
    p_min: np.ndarray           # float32 running min of leveled probability
    p_max: np.ndarray           # float32 running max
    best_inc: np.ndarray        # float32 incidence of the best-illuminated contributor so far
    primary: np.ndarray         # int32 index (into the frame table) of that contributor
    src_code: np.ndarray        # int8 worst offset-provenance severity among contributors

    @classmethod
    def zeros(cls, shape) -> "TileAccum":
        return cls(shape=shape,
                   sum_logit=np.zeros(shape, dtype=np.float64),
                   n_frames=np.zeros(shape, dtype=np.int16),
                   p_min=np.full(shape, np.inf, dtype=np.float32),
                   p_max=np.full(shape, -np.inf, dtype=np.float32),
                   best_inc=np.full(shape, np.inf, dtype=np.float32),
                   primary=np.full(shape, -1, dtype=np.int32),
                   src_code=np.zeros(shape, dtype=np.int8))

    def add_frame(self, rows: np.ndarray, cols: np.ndarray, leveled_logit: np.ndarray,
                  *, frame_idx: int, incidence: float, src_code: int) -> int:
        """Accumulate one frame's in-bounds tiles. Returns the number of pixels touched."""
        h, w = self.shape
        ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w) & np.isfinite(leveled_logit)
        if not ok.any():
            return 0
        r, c, lg = rows[ok], cols[ok], np.asarray(leveled_logit, dtype=np.float64)[ok]
        p = lv.sigmoid(lg).astype(np.float32)
        self.sum_logit[r, c] += lg
        self.n_frames[r, c] += 1
        np.minimum.at(self.p_min, (r, c), p)
        np.maximum.at(self.p_max, (r, c), p)
        # primary frame = the best-illuminated (lowest-incidence) contributor. With a mean composite
        # no single frame "owns" a pixel, so this is provenance-of-record, not the value's source.
        inc = np.float32(incidence if np.isfinite(incidence) else np.inf)
        better = inc < self.best_inc[r, c]
        if better.any():
            rb, cb = r[better], c[better]
            self.best_inc[rb, cb] = inc
            self.primary[rb, cb] = frame_idx
        np.maximum.at(self.src_code, (r, c), np.int8(src_code))
        return int(ok.sum())

    def finish(self) -> dict[str, np.ndarray]:
        """Turn the accumulators into the shippable rasters (nodata = NaN, as the map path uses)."""
        covered = self.n_frames > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_logit = np.where(covered, self.sum_logit / np.maximum(self.n_frames, 1), np.nan)
        prob = np.where(covered, lv.sigmoid(mean_logit), np.nan).astype(np.float32)
        dp = np.where(covered & (self.n_frames > 1), self.p_max - self.p_min, np.nan).astype(np.float32)
        return {
            "prob_raw": prob,
            "mean_logit": mean_logit.astype(np.float32),
            "n_frames": np.where(covered, self.n_frames, np.nan).astype(np.float32),
            "overlap_dp": dp,
            "primary_frame": np.where(covered, self.primary, np.nan).astype(np.float32),
            "incidence": np.where(covered & np.isfinite(self.best_inc), self.best_inc,
                                  np.nan).astype(np.float32),
            "offset_source": np.where(covered, self.src_code, np.nan).astype(np.float32),
        }


# --------------------------------------------------------------------------- one frame -> one tile
def frame_rows_cols(grid: TileGrid, TI: np.ndarray, TJ: np.ndarray):
    """Map a frame's global tile keys onto (row, col) of this tile's grid (exact integer shift)."""
    return grid.rows_of_TI(TI), grid.cols_of_TJ(TJ)


def frame_bbox(TI: np.ndarray, TJ: np.ndarray) -> tuple[int, int, int, int]:
    """(TI_min, TI_max, TJ_min, TJ_max) of one frame's global tiles; used to prescreen tiles."""
    if TI.size == 0:
        return (0, -1, 0, -1)
    return (int(TI.min()), int(TI.max()), int(TJ.min()), int(TJ.max()))


def bbox_intersects_tile(bbox, grid: TileGrid) -> bool:
    ti0, ti1, tj0, tj1 = bbox
    if ti1 < ti0 or tj1 < tj0:
        return False
    gi0, gi1 = grid.TI_range()
    gj0, gj1 = grid.TJ_range()
    return not (ti1 < gi0 or ti0 > gi1 or tj1 < gj0 or tj0 > gj1)


# --------------------------------------------------------------------------- scoring support
def frame_labels_on_grid(grid: TileGrid, frames, pid_order: list[str]) -> np.ndarray:
    """SeamMap single-owner frame labels on this tile's grid, valued by index into `pid_order`.

    `src.striping.frame_label_map` does the same rasterize but HARD-WIRES the grid to
    `reports/map_region/{tile}_abundance.tif` and labels by GeoDataFrame row order, so it cannot
    label an F-build raster consistently with a frame table. These are the same six lines with the
    grid and the label vocabulary passed in (the pattern of `scripts/f_pilot_crop.frame_labels`).
    """
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    idx = {pid: i for i, pid in enumerate(pid_order)}
    shapes = [(geom, idx[pid]) for geom, pid in zip(frames.geometry, frames["PRODUCT_ID"])
              if pid in idx and geom is not None and not geom.is_empty]
    if not shapes:
        return np.full(grid.shape, -1, dtype=np.int32)
    return rasterize(shapes, out_shape=grid.shape, transform=Affine(*grid.transform),
                     fill=-1, dtype="int32", all_touched=False)


def partition_composite(per_frame_prob: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
                        labels: np.ndarray) -> np.ndarray:
    """The gate-1 scoring composite: every pixel takes the value of ITS SeamMap owner frame.

    Mirrors `scripts/f_h2_eta2.score`'s partition branch, which is what every on-record η² number
    (mosaic 0.196 / A1 0.141 / H1 0.128 / H1+H4 0.0505) was computed on. Distinct from the SHIPPED
    mean composite: it uses one frame per pixel, so it deliberately discards the overlap information
    the mean uses — that is the price of being label-comparable to the mosaic map, which has exactly
    one value per pixel by construction.
    """
    out = np.full(labels.shape, np.nan, dtype=np.float32)
    for fi, (rows, cols, prob) in per_frame_prob.items():
        if rows.size == 0:
            continue
        own = labels[rows, cols] == fi
        if own.any():
            out[rows[own], cols[own]] = prob[own]
    return out


def windows_over_grid(grid: TileGrid, win_px: int, min_frac: float = 0.5):
    """Non-overlapping pilot-scale windows over a tile (gate 1's headline scoring scale).

    The 0.05 η² bar was calibrated on a ~75 km / 7-frame crop, not on a 4° tile or the whole block,
    and η² has no group-count correction so it grows mechanically with frame count. Scoring in
    pilot-sized windows is what keeps the pre-declared bar meaningful (Brian 2026-07-28).
    """
    h, w = grid.shape
    for r0 in range(0, h, win_px):
        for c0 in range(0, w, win_px):
            r1, c1 = min(r0 + win_px, h), min(c0 + win_px, w)
            if (r1 - r0) * (c1 - c0) < min_frac * win_px * win_px:
                continue
            yield (r0, r1, c0, c1)
