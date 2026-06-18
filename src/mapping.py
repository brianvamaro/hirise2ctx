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

Grid convention (CLAUDE.md Stage 4 / `src.labeling._compute_grid_alignment`):
tiles are anchored to the **parent Murray tile's pixel origin**, so a window read
at pixel offset (row_off, col_off) has grid origin `row0=row_off, col0=col_off`.
`(ti, tj)` are therefore unique within a tile; cross-tile combine additionally
keys on the Murray-tile id (the scale-out step, not this pilot).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


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


def own_tile_zero_fraction(window: np.ndarray, ti: np.ndarray, tj: np.ndarray, *,
                           tile_px: int, row0: int, col0: int) -> np.ndarray:
    """Per-tile fraction of own-tile CTX pixels that are 0 (Murray mosaic nodata).

    A tile sitting in a mosaic data gap embeds black pixels and yields a
    meaningless prediction; the caller masks tiles whose zero-fraction is high.
    `ti, tj` are global tile indices; the own tile occupies window rows
    [ti*tile_px - row0, +tile_px) (CLAUDE.md grid anchor).
    """
    ti = np.asarray(ti, dtype=np.int64)
    tj = np.asarray(tj, dtype=np.int64)
    H, W = window.shape
    out = np.ones(ti.size, dtype=np.float32)
    r = ti * tile_px - row0
    c = tj * tile_px - col0
    for i in range(ti.size):
        if r[i] < 0 or c[i] < 0 or r[i] + tile_px > H or c[i] + tile_px > W:
            continue  # own tile outside window (shouldn't happen for enumerated grid)
        box = window[r[i]: r[i] + tile_px, c[i]: c[i] + tile_px]
        out[i] = float((box == 0).mean())
    return out


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


def mosaic_geotiffs(paths, out_path: str | Path | None = None):
    """Merge same-CRS single-band GeoTIFFs into one raster (Stage: regional mosaic).

    The per-tile `map_region` outputs all share the Murray global equirectangular CRS
    (`clon_0`), differing only in extent, so a straight merge stitches them — no
    reprojection. Returns `(array2d, transform, crs_wkt)`; NaN fills any uncovered gap
    (the block is an L-shape, so two corners are nodata). Writes `out_path` if given.
    """
    import rasterio
    from rasterio.merge import merge

    paths = [str(p) for p in paths]
    srcs = [rasterio.open(p) for p in paths]
    try:
        arr, transform = merge(srcs, nodata=np.nan)
        crs_wkt = srcs[0].crs.to_wkt() if srcs[0].crs else ""
    finally:
        for s in srcs:
            s.close()
    arr = arr[0]  # single band
    if out_path is not None:
        write_geotiff(out_path, arr, transform, crs_wkt)
    return arr, transform, crs_wkt


def write_geotiff(path: str | Path, raster: np.ndarray, transform, crs_wkt: str,
                  *, nodata: float = np.nan) -> Path:
    """Write a single-band float32 GeoTIFF (the abundance/probability raster)."""
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=raster.shape[0], width=raster.shape[1],
        count=1, dtype="float32", crs=crs_wkt or None, transform=transform,
        nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(raster.astype(np.float32), 1)
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


def predict_window(window: CtxWindow, embedder, head, *, tile_px: int = 32,
                   pool: str = "gem", batch: int = 96,
                   max_zero_fraction: float = 0.5, calibrator=None,
                   apply_isotonic: bool = True) -> WindowPrediction:
    """Embed -> predict -> (optionally calibrate) -> rasterize one CTX window.

    `embedder` is a `src.fm_embeddings.FangEmbedder`; `head` exposes
    `predict(emb)->prob` (`DeployableHead`). Tiles whose context box spills the
    window edge (embed returns NaN) or whose own-tile CTX is >`max_zero_fraction`
    nodata are masked (prob NaN). Returns the dense raster + its affine.

    `calibrator` is an optional Stage-1 `src.calibration.CalibrationLayer`. When given,
    an **abundance** raster `calibrate_abundance(raw P(rich))` (the one-model
    quantile-match — the de-compression win) is added, and the rich/poor raster is
    isotonic-calibrated unless `apply_isotonic=False` (the isotonic ECE polish is a
    rank-safe gate-clear, not a per-image-significant win, so it is toggleable). The
    raw probability is always kept in `prob_raw`. `calibrator=None` (default) renders
    raw, unchanged — the raw/calibrated toggle.
    """
    from src.fm_embeddings import tile_grid_for_window

    arr = window.data
    row0, col0 = window.row_off, window.col_off
    ti, tj = tile_grid_for_window(arr.shape, row0, col0, tile_px)
    emb, valid = embedder.embed_window(arr, ti, tj, tile_px=tile_px, row0=row0,
                                       col0=col0, pool=pool, batch=batch)

    zero_frac = own_tile_zero_fraction(arr, ti, tj, tile_px=tile_px, row0=row0, col0=col0)
    usable = valid & (zero_frac <= max_zero_fraction)

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
    # (ti, tj) are tile-anchored (global); rebuild the tile origin so the window
    # offset isn't double-counted (it already lives in window.transform).
    tile_transform = tile_origin_transform(window.transform, row0, col0)
    transform = coarsened_transform(tile_transform, ti_min, tj_min, tile_px)
    return WindowPrediction(
        ti=ti, tj=tj, prob=prob, raster=raster, ti_min=ti_min, tj_min=tj_min,
        transform=transform, crs_wkt=window.crs_wkt,
        n_valid=int(valid.sum()), n_masked_nodata=int((valid & ~usable).sum()),
        calibrated=calibrator is not None, prob_raw=prob_raw,
        abundance=abundance, abundance_raster=abundance_raster,
    )
