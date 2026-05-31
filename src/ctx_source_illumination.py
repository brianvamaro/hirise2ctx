"""Stage 6b -- CTX-source illumination features.

For each tile, look up the contributing CTX source images (from the Murray Lab
``SeamMap.shp``) and compute area-weighted aggregates of their INCIDENCE,
EMISSION, PHASE, and SUB_SOLAR_AZIMUTH angles.  Output columns appended to a
Stage 4b feature parquet:

    ctx_incidence_mean      (deg)  area-weighted across pixels in tile
    ctx_incidence_std       (deg)  std across pixels in tile (variance signal)
    ctx_emission_mean       (deg)
    ctx_phase_mean          (deg)
    ctx_subsolar_az_mean    (deg)  linear mean -- see caveat below
    ctx_n_sources           (int)  number of distinct CTX sources in tile
    ctx_dominant_source_fraction   share of tile area covered by dominant source

Tests the H3 anti-signal hypothesis (PROMOTION_QUEUE.md Stage 6b): at very oblique
CTX-source illumination, ``shadow_fraction`` mis-reads ripple-field / crater-rim
shadows as boulders, dragging per-image AUC the wrong way on those images.
``ctx_incidence_mean`` and ``ctx_n_sources`` are the candidate H3 signals; the
``_std`` and ``_dominant_source_fraction`` columns probe within-tile geometry
mixing (mosaic-seam-like effects, partial overlap with Stage 6e).

The Murray Lab SeamMap turns out to embed per-source illumination angles directly
(``INCIDENCE``, ``EMISSION``, ``PHASE``, ``SB_SLR_AZ`` columns) -- the
PDS CTX CUMINDEX is therefore not required for Stage 6b. Notebook 13 said the seam
files "don't carry illumination angles"; this module verified otherwise on
``E12_N44`` (56 unique sources, INCIDENCE range 40-81 deg, std 8.9 deg).

CRS: the SeamMap is in ``Mars_2015_Ocentric_Equirectangular_clon_0`` -- same CRS
the Murray Lab mosaic GeoTIFFs use, which is what Stage 2 cached the CTX windows
in.  No reprojection required for the spatial join.

Sub-solar-azimuth caveat: ``SB_SLR_AZ`` is a directional angle in [0, 360).  For
the v2 latitude band the values cluster around 130-180 deg (no wrap), so linear
mean and std are correct.  For coverage that crosses 0/360 the linear mean would
be wrong; the module emits a warning if min/max span > 180 deg within any window.
"""
from __future__ import annotations

import shutil
import warnings
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Angle columns we surface from the SeamMap, in (output_col_suffix, seam_col) order.
ANGLE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("incidence", "INCIDENCE"),
    ("emission", "EMISSION"),
    ("phase", "PHASE"),
    ("subsolar_az", "SB_SLR_AZ"),
)
# The full output column set Stage 6b emits.
OUTPUT_COLUMNS: tuple[str, ...] = tuple(
    f"ctx_{suffix}_mean" for suffix, _ in ANGLE_COLUMNS
) + (
    "ctx_incidence_std",
    "ctx_n_sources",
    "ctx_dominant_source_fraction",
)


# ============================================================================
# SeamMap loader
# ============================================================================

def _extract_seam_map_dir(tile_zip: Path) -> Path:
    """Extract the SeamMap shapefile out of a Murray tile zip into a sibling dir.

    Idempotent: re-uses an existing extraction dir if all 4 component files
    (.shp/.dbf/.shx/.prj) are present.
    """
    extract_dir = tile_zip.parent / f"_seammap_{tile_zip.stem}"
    expected = {".shp", ".dbf", ".shx", ".prj"}
    if extract_dir.exists():
        have = {p.suffix.lower() for p in extract_dir.iterdir() if "SeamMap" in p.name}
        if expected.issubset(have):
            return extract_dir
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(tile_zip) as zf:
        for name in zf.namelist():
            if "SeamMap" not in name:
                continue
            target = extract_dir / Path(name).name
            with zf.open(name) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
    return extract_dir


def load_seam_map(murray_tile: str, cache_dir: Path) -> "geopandas.GeoDataFrame":
    """Load the Murray Lab SeamMap for ``murray_tile`` from ``cache_dir/ctx_tiles``.

    Extracts the SeamMap.shp from the cached Murray tile zip if it has not been
    extracted yet. Returns a GeoDataFrame with at least the angle columns named
    in :data:`ANGLE_COLUMNS` plus ``PRODUCT_ID`` and ``geometry``.

    Args:
        murray_tile: Tile name in the cache form (e.g. ``"E12_N44"``).  Use
            ``src.ctx_tiles.manifest_to_murray`` to translate from manifest form.
        cache_dir: Project cache root (the dir containing ``ctx_tiles/``).
    """
    import geopandas as gpd

    tile_zip = Path(cache_dir) / "ctx_tiles" / f"{murray_tile}.zip"
    if not tile_zip.exists():
        raise FileNotFoundError(f"no Murray tile zip at {tile_zip}")
    extract_dir = _extract_seam_map_dir(tile_zip)
    shp = next(extract_dir.glob("*SeamMap.shp"), None)
    if shp is None:
        raise FileNotFoundError(f"no SeamMap.shp inside {extract_dir}")
    gdf = gpd.read_file(shp)
    needed = {"INCIDENCE", "EMISSION", "PHASE", "SB_SLR_AZ", "PRODUCT_ID", "geometry"}
    missing = needed - set(gdf.columns)
    if missing:
        raise RuntimeError(
            f"SeamMap {shp} missing required columns: {sorted(missing)}; "
            f"present: {sorted(gdf.columns)}"
        )
    return gdf


# ============================================================================
# Per-window rasterization
# ============================================================================

def rasterize_seam_map_window(
    seam_gdf,
    window_transform,
    window_h: int,
    window_w: int,
) -> dict[str, np.ndarray]:
    """Rasterize SeamMap angle fields onto the CTX window's pixel grid.

    Returns a dict with float32 arrays for each of the four angle channels
    (``INCIDENCE``, ``EMISSION``, ``PHASE``, ``SB_SLR_AZ``) plus a ``SOURCE_ID``
    uint16 array.  The arrays have shape ``(window_h, window_w)``; pixels not
    covered by any polygon get NaN in the angle arrays and 0 in ``SOURCE_ID``.

    The SeamMap is restricted to polygons intersecting the window bbox before
    rasterization, which is much cheaper than rasterizing across the full
    4 deg x 4 deg Murray tile (typically 56 sources reduce to 5-15 inside one
    HiRISE-sized window).

    For overlapping polygons, the last polygon in the GDF iteration order wins
    -- this matches GDAL rasterize default behaviour and is consistent with
    treating the mosaic value at that pixel as coming from the most-recently
    listed source.  Overlap is rare in practice (SeamMap is the "winning"
    selection per region).
    """
    from rasterio.features import rasterize
    from shapely.geometry import box

    if len(seam_gdf) == 0:
        empty = np.full((window_h, window_w), np.nan, dtype=np.float32)
        return {
            "INCIDENCE": empty.copy(),
            "EMISSION": empty.copy(),
            "PHASE": empty.copy(),
            "SB_SLR_AZ": empty.copy(),
            "SOURCE_ID": np.zeros((window_h, window_w), dtype=np.uint16),
        }

    xmin = window_transform.c
    ymax = window_transform.f
    # window_transform.a > 0 (E-pos), window_transform.e < 0 (N-pos), so:
    xmax = xmin + window_w * window_transform.a
    ymin = ymax + window_h * window_transform.e
    bbox = box(xmin, ymin, xmax, ymax)

    subset = seam_gdf[seam_gdf.intersects(bbox)].reset_index(drop=True)
    if len(subset) == 0:
        empty = np.full((window_h, window_w), np.nan, dtype=np.float32)
        return {
            "INCIDENCE": empty.copy(),
            "EMISSION": empty.copy(),
            "PHASE": empty.copy(),
            "SB_SLR_AZ": empty.copy(),
            "SOURCE_ID": np.zeros((window_h, window_w), dtype=np.uint16),
        }

    out: dict[str, np.ndarray] = {}
    for _suffix, col in ANGLE_COLUMNS:
        shapes = [
            (g, float(v))
            for g, v in zip(subset.geometry, subset[col])
            if g is not None and not g.is_empty and np.isfinite(v)
        ]
        arr = rasterize(
            shapes=shapes,
            out_shape=(window_h, window_w),
            transform=window_transform,
            fill=np.nan,
            dtype=np.float32,
            all_touched=False,
        )
        out[col] = arr

    src_codes = subset["PRODUCT_ID"].astype("category").cat.codes.to_numpy() + 1
    if src_codes.max() > 65535:
        raise RuntimeError(
            f"too many distinct sources ({src_codes.max()}) for uint16 SOURCE_ID"
        )
    src_shapes = [
        (g, int(i))
        for g, i in zip(subset.geometry, src_codes)
        if g is not None and not g.is_empty
    ]
    out["SOURCE_ID"] = rasterize(
        shapes=src_shapes,
        out_shape=(window_h, window_w),
        transform=window_transform,
        fill=0,
        dtype=np.uint16,
        all_touched=False,
    )

    saz = out["SB_SLR_AZ"]
    finite = np.isfinite(saz)
    if finite.any():
        spread = float(saz[finite].max() - saz[finite].min())
        if spread > 180.0:
            warnings.warn(
                f"ctx_source_illumination: SB_SLR_AZ spread {spread:.1f} deg > 180 "
                "within a window; linear mean may be wrong (wrap not handled).",
                RuntimeWarning,
            )
    return out


# ============================================================================
# Per-tile aggregation
# ============================================================================

def _aggregate_per_tile(
    df: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    mosaic_row_origin: int,
    mosaic_col_origin: int,
) -> dict[str, np.ndarray]:
    """Vectorized for mean / std; tight loop for n_sources / dominant_fraction.

    Tile (ti, tj) at scale S covers window-rows ``[ti*S - mosaic_row_origin,
    (ti+1)*S - mosaic_row_origin)`` and likewise for cols.  Tiles whose pixel
    block lies fully outside the window get NaN.
    """
    n = len(df)
    if n == 0:
        return {c: np.zeros(0, dtype=np.float32) for c in OUTPUT_COLUMNS}

    out = {
        "ctx_incidence_mean": np.full(n, np.nan, dtype=np.float32),
        "ctx_incidence_std": np.full(n, np.nan, dtype=np.float32),
        "ctx_emission_mean": np.full(n, np.nan, dtype=np.float32),
        "ctx_phase_mean": np.full(n, np.nan, dtype=np.float32),
        "ctx_subsolar_az_mean": np.full(n, np.nan, dtype=np.float32),
        "ctx_n_sources": np.zeros(n, dtype=np.int32),
        "ctx_dominant_source_fraction": np.full(n, np.nan, dtype=np.float32),
    }

    ti = df["ti"].to_numpy(dtype=np.int64)
    tj = df["tj"].to_numpy(dtype=np.int64)
    sz = df["tile_size_px"].to_numpy(dtype=np.int64)

    inc = arrays["INCIDENCE"]
    emi = arrays["EMISSION"]
    pha = arrays["PHASE"]
    saz = arrays["SB_SLR_AZ"]
    sid = arrays["SOURCE_ID"]
    H, W = inc.shape

    r0 = ti * sz - mosaic_row_origin
    c0 = tj * sz - mosaic_col_origin
    r1 = r0 + sz
    c1 = c0 + sz

    in_bounds = (r0 >= 0) & (c0 >= 0) & (r1 <= H) & (c1 <= W)

    for i in np.flatnonzero(in_bounds):
        rr0 = int(r0[i]); cc0 = int(c0[i])
        rr1 = int(r1[i]); cc1 = int(c1[i])
        inc_blk = inc[rr0:rr1, cc0:cc1]
        valid = np.isfinite(inc_blk)
        n_valid = int(valid.sum())
        if n_valid == 0:
            continue
        out["ctx_incidence_mean"][i] = float(inc_blk[valid].mean())
        if n_valid >= 2:
            out["ctx_incidence_std"][i] = float(inc_blk[valid].std(ddof=0))
        out["ctx_emission_mean"][i] = float(emi[rr0:rr1, cc0:cc1][valid].mean())
        out["ctx_phase_mean"][i] = float(pha[rr0:rr1, cc0:cc1][valid].mean())
        out["ctx_subsolar_az_mean"][i] = float(saz[rr0:rr1, cc0:cc1][valid].mean())
        sid_blk = sid[rr0:rr1, cc0:cc1]
        sid_nonzero = sid_blk[sid_blk > 0]
        if sid_nonzero.size > 0:
            uniq, counts = np.unique(sid_nonzero, return_counts=True)
            out["ctx_n_sources"][i] = int(uniq.size)
            tile_area_px = int(sz[i]) * int(sz[i])
            out["ctx_dominant_source_fraction"][i] = float(counts.max()) / tile_area_px
    return out


def add_ctx_source_illumination_features(
    df: pd.DataFrame,
    *,
    seam_gdf,
    window_transform,
    window_h: int,
    window_w: int,
    mosaic_row_origin: int,
    mosaic_col_origin: int,
) -> pd.DataFrame:
    """Return a copy of ``df`` with Stage 6b illumination columns appended.

    All rows must share a single ``obs_id`` (this function operates on one image's
    feature parquet at a time -- the driver script loops over images).

    Args:
        df: Stage 4b feature frame for one ObsId.  Must contain columns ``obs_id``,
            ``scale_idx``, ``tile_size_px``, ``ti``, ``tj``.
        seam_gdf: SeamMap GeoDataFrame for the Murray tile that produced the CTX
            window (see :func:`load_seam_map`).
        window_transform: rasterio Affine for the cached CTX window TIFF.
        window_h, window_w: pixel shape of the cached CTX window TIFF.
        mosaic_row_origin, mosaic_col_origin: absolute mosaic-pixel indices of
            the window's (0, 0); see ``src.labeling._compute_grid_alignment``.

    Returns:
        New dataframe with the original columns followed by the columns in
        :data:`OUTPUT_COLUMNS`.
    """
    arrays = rasterize_seam_map_window(seam_gdf, window_transform, window_h, window_w)
    cols = _aggregate_per_tile(df, arrays, mosaic_row_origin, mosaic_col_origin)
    augmented = df.copy()
    for c, v in cols.items():
        augmented[c] = v
    return augmented


# ============================================================================
# Window geometry helpers (one-stop shop for the driver script)
# ============================================================================

def load_window_metadata(ctx_window_tif: Path) -> dict[str, Any]:
    """Read transform + pixel shape from a Stage 2 ``ctx_windows/{ObsId}.tif``."""
    import rasterio

    with rasterio.open(ctx_window_tif) as src:
        return {
            "window_transform": src.transform,
            "window_h": int(src.height),
            "window_w": int(src.width),
            "window_crs_wkt": src.crs.to_wkt() if src.crs else None,
        }


def mosaic_origin_pixels(
    window_transform, mosaic_transform: list[float],
) -> tuple[int, int]:
    """Compute (mosaic_row_origin, mosaic_col_origin) for a window.

    Mirrors the calculation in ``src.labeling._compute_grid_alignment`` so this
    module doesn't depend on Stage 4 internals.
    """
    px_x = abs(window_transform.a)
    px_y = abs(window_transform.e)
    mx_origin_x = mosaic_transform[2]
    mx_origin_y = mosaic_transform[5]
    mosaic_col_origin = int(round((window_transform.c - mx_origin_x) / px_x))
    mosaic_row_origin = int(round((mx_origin_y - window_transform.f) / px_y))
    return mosaic_row_origin, mosaic_col_origin
