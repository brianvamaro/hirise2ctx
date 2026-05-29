"""Stage 4 - label generation on a nested x2 grid anchored to the CTX mosaic pixel origin.

Per CLAUDE.md Section 4:

- Grid anchor is the **CTX mosaic native pixel origin**, not the HiRISE footprint, so every
  label tile is an integer block of CTX pixels and grids are reproducible across runs and
  exactly nested across scales. Stage 2 already snaps each ctx_windows/{ObsId}.tif origin
  to the mosaic pixel grid, so the CTX window's (row=0, col=0) lives at some integer mosaic
  pixel offset and aligning to multiples of `tile_size_px` is exact integer arithmetic.

- Tile sizes form an ascending x2 ladder (config `labeling.tile_sizes_px`, e.g. [8, 16, 32,
  64]). Base per-tile stats (`boulder_area`, `boulder_count`, `tile_area`) are computed once
  on the finest grid and **summed upward**: a coarse tile at scale 2*S contains exactly 4
  sub-tiles at scale S, so per-tile sums are 2x2 reductions. The same is true of
  eligibility -- a coarse tile is eligible iff every sub-tile is eligible -- so summing
  cleanly drops partial-coverage coarse tiles.

- A tile is **eligible** iff every HiRISE coverage-mask pixel inside it is 1 (see
  DECISIONS.md 2026-05-21 "Labels-only-on-HiRISE-coverage constraint"). The strict
  `coverage == 1.0` rule was chosen 2026-05-23 in preference to a relaxed `>= 0.95` because
  fractional_area is biased low under partial coverage (numerator scales with covered area,
  denominator stays full tile area).

- The Stage 3 (dx_m, dy_m) shift is **applied to the polygons** before rasterization (2026-
  05-23 decision). The grid itself stays anchored to the CTX pixel origin (no resampling).
  This aligns HiRISE-derived boulder positions with the CTX texture features in the same
  tile, eliminating the systematic ~200 m HiRISE-vs-CTX-feature offset that Stage 3
  measured. ESP_057469_2215 is dropped before Stage 4 (its Stage 2 window only covers
  0.1% of the HiRISE swath, see DECISIONS.md 2026-05-22 tile-straddle entry).

- All derived label transforms are emitted regardless of `labeling.label_type`:
  `fractional_area`, `binary_by_area`, `binary_by_count`, `count`, `count_density`,
  optional `categorical` if `categorical_bins` is non-empty. The cheap/idempotent
  re-runnability requirement (CLAUDE.md acceptance #4) is satisfied by writing the base
  stats (boulder_area, boulder_count, tile_area) into every row so a downstream config
  change to thresholds re-derives labels from the cached parquet in milliseconds.

- **Texture features (GLCM, intensity stats, gradient, shadow-fraction) are a separable
  second pass**, not in this module. Stage 4 here owns label generation; a follow-on
  features module will read the same per-tile bounds and emit CTX-derived inputs.

Boulder area is computed by rasterizing polygons at a sub-pixel oversample (default 5x =>
1 m sub-pixels at 5 m/px CTX) and counting boulder sub-pixels per finest tile. This avoids
expensive per-tile shapely intersection (O(n_tiles x n_polygons)) and gives ~1 m^2
granularity, which is below the ~3.7 m^2 median boulder area (DECISIONS.md 2026-05-20).

Boulder count is computed by binning polygon centroids into finest-grid cells -- this is
unambiguous at tile borders (each boulder counts exactly once in the tile owning its
centroid), unlike "any intersection" which double-counts boulders that span a boundary.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABELS_SUBDIR = "labels"
DEFAULT_SUBPIXEL_FACTOR = 5

# All tiles are required to have HiRISE mask coverage == 1.0 (every pixel covered).
# Chosen 2026-05-23 over a relaxed >= 0.95 -- see module docstring.
ELIGIBILITY_RULE = "coverage_equals_one"


def _load_ctx_window(ctx_window_tif: Path) -> tuple[int, int, Any, Any]:
    """Return (height, width, transform, crs) of a Stage 2 CTX window GeoTIFF."""
    import rasterio

    with rasterio.open(ctx_window_tif) as src:
        return int(src.height), int(src.width), src.transform, src.crs


def _load_mosaic_transform(cache_dir: Path, murray_tile: str) -> list[float]:
    """Read the parent CTX tile's affine transform from its Stage 2 sidecar."""
    sidecar = cache_dir / "ctx_tiles" / f"{murray_tile}.json"
    info = json.loads(sidecar.read_text(encoding="utf-8"))
    return info["inner_transform"]


def _apply_coreg_shift(gdf, shift: dict | None):
    """Translate every polygon by (dx_m, dy_m). No-op when shift is None or gdf empty."""
    if shift is None or len(gdf) == 0:
        return gdf
    dx = float(shift["shift_m"]["dx"])
    dy = float(shift["shift_m"]["dy"])
    out = gdf.copy()
    out.geometry = out.geometry.translate(xoff=dx, yoff=dy)
    return out


def _apply_detection_filters(gdf, filters: dict | None):
    """Drop polygons failing `min_confidence` (DBF `score`) or `min_size_m` (derived diameter).

    `min_size_m` is interpreted as a minimum equivalent-circle diameter,
    `2*sqrt(area/pi)`. Returns a (possibly identical) GeoDataFrame.
    """
    if filters is None or len(gdf) == 0:
        return gdf
    keep = np.ones(len(gdf), dtype=bool)
    min_conf = filters.get("min_confidence")
    if min_conf is not None and "score" in gdf.columns:
        keep &= gdf["score"].to_numpy() >= float(min_conf)
    min_size_m = filters.get("min_size_m")
    if min_size_m is not None:
        diam = 2.0 * np.sqrt(gdf.geometry.area.to_numpy() / np.pi)
        keep &= diam >= float(min_size_m)
    if keep.all():
        return gdf
    return gdf.iloc[keep].reset_index(drop=True)


def _compute_grid_alignment(
    window_transform,
    mosaic_transform: list[float],
    window_h: int,
    window_w: int,
    tile_sizes_px: list[int],
) -> dict[str, int]:
    """Return the absolute mosaic-pixel offsets + the coarsest-scale-aligned finest-grid range.

    The returned dict has:
      mosaic_row_origin, mosaic_col_origin -- window (0,0) at these mosaic-pixel indices
      j_min_row, j_max_row, j_min_col, j_max_col -- finest-grid cell index range, inclusive
      r0_win, r1_win, c0_win, c1_win -- window-pixel slice of the working region

    Raises ValueError if the window is too small to contain a single coarsest tile.
    """
    px_x = abs(window_transform.a)
    px_y = abs(window_transform.e)
    mx_origin_x = mosaic_transform[2]
    mx_origin_y = mosaic_transform[5]

    mosaic_col_origin = int(round((window_transform.c - mx_origin_x) / px_x))
    mosaic_row_origin = int(round((mx_origin_y - window_transform.f) / px_y))

    S_min = int(tile_sizes_px[0])
    S_max = int(tile_sizes_px[-1])

    # Coarsest-scale cell range: cell k is fully inside window iff
    #   k * S_max >= mosaic_row_origin  AND  (k+1) * S_max <= mosaic_row_origin + window_h
    K_min_row = math.ceil(mosaic_row_origin / S_max)
    K_max_row = (mosaic_row_origin + window_h) // S_max - 1
    K_min_col = math.ceil(mosaic_col_origin / S_max)
    K_max_col = (mosaic_col_origin + window_w) // S_max - 1

    if K_min_row > K_max_row or K_min_col > K_max_col:
        raise ValueError(
            f"window ({window_h}x{window_w} px, mosaic-origin "
            f"({mosaic_row_origin},{mosaic_col_origin})) cannot fit a single "
            f"{S_max}-px coarsest tile -- no labels emittable."
        )

    ratio = S_max // S_min
    j_min_row = K_min_row * ratio
    j_max_row = (K_max_row + 1) * ratio - 1
    j_min_col = K_min_col * ratio
    j_max_col = (K_max_col + 1) * ratio - 1

    r0_win = j_min_row * S_min - mosaic_row_origin
    r1_win = (j_max_row + 1) * S_min - mosaic_row_origin
    c0_win = j_min_col * S_min - mosaic_col_origin
    c1_win = (j_max_col + 1) * S_min - mosaic_col_origin

    assert 0 <= r0_win < r1_win <= window_h, (r0_win, r1_win, window_h)
    assert 0 <= c0_win < c1_win <= window_w, (c0_win, c1_win, window_w)

    return {
        "mosaic_row_origin": mosaic_row_origin,
        "mosaic_col_origin": mosaic_col_origin,
        "j_min_row": j_min_row, "j_max_row": j_max_row,
        "j_min_col": j_min_col, "j_max_col": j_max_col,
        "r0_win": r0_win, "r1_win": r1_win,
        "c0_win": c0_win, "c1_win": c1_win,
    }


def _rasterize_boulders_subpixel(
    gdf,
    window_transform,
    align: dict[str, int],
    subpixel_factor: int,
) -> np.ndarray:
    """Rasterize polygons at `subpixel_factor` x CTX resolution within the working region.

    Returns a uint8 array of shape `((r1-r0)*subpixel_factor, (c1-c0)*subpixel_factor)`,
    1 where any polygon covers the sub-pixel center, 0 elsewhere. Empty gdf -> all-zero.
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import Affine

    r0, r1, c0, c1 = align["r0_win"], align["r1_win"], align["c0_win"], align["c1_win"]
    sub_h = (r1 - r0) * subpixel_factor
    sub_w = (c1 - c0) * subpixel_factor

    sub_origin_x = window_transform.c + c0 * window_transform.a
    sub_origin_y = window_transform.f + r0 * window_transform.e
    sub_transform = Affine(
        window_transform.a / subpixel_factor, 0.0, sub_origin_x,
        0.0, window_transform.e / subpixel_factor, sub_origin_y,
    )

    if len(gdf) == 0:
        return np.zeros((sub_h, sub_w), dtype=np.uint8)
    shapes = ((geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    raster = rasterize(
        shapes=shapes,
        out_shape=(sub_h, sub_w),
        transform=sub_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    )
    return raster


def _count_centroids_per_finest_cell(
    gdf,
    mosaic_transform: list[float],
    align: dict[str, int],
    finest_size_px: int,
    px_x: float,
    px_y: float,
) -> np.ndarray:
    """Return an (n_jr, n_jc) int32 array of centroid counts per finest-grid cell."""
    n_jr = align["j_max_row"] - align["j_min_row"] + 1
    n_jc = align["j_max_col"] - align["j_min_col"] + 1
    counts = np.zeros((n_jr, n_jc), dtype=np.int32)
    if len(gdf) == 0:
        return counts

    centroids = gdf.geometry.centroid
    cx = centroids.x.to_numpy()
    cy = centroids.y.to_numpy()
    mosaic_col_frac = (cx - mosaic_transform[2]) / px_x
    mosaic_row_frac = (mosaic_transform[5] - cy) / px_y
    cell_row = np.floor(mosaic_row_frac / finest_size_px).astype(np.int64)
    cell_col = np.floor(mosaic_col_frac / finest_size_px).astype(np.int64)
    in_range = (
        (cell_row >= align["j_min_row"]) & (cell_row <= align["j_max_row"])
        & (cell_col >= align["j_min_col"]) & (cell_col <= align["j_max_col"])
    )
    cr = cell_row[in_range] - align["j_min_row"]
    cc = cell_col[in_range] - align["j_min_col"]
    np.add.at(counts, (cr.astype(np.intp), cc.astype(np.intp)), 1)
    return counts


def _build_finest_stats(
    boulder_sub: np.ndarray,
    mask: np.ndarray,
    align: dict[str, int],
    finest_size_px: int,
    subpixel_factor: int,
    sub_pixel_area_m2: float,
    centroid_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reduce the sub-pixel boulder raster + mask into finest-grid per-tile arrays."""
    r0, r1, c0, c1 = align["r0_win"], align["r1_win"], align["c0_win"], align["c1_win"]
    n_jr = align["j_max_row"] - align["j_min_row"] + 1
    n_jc = align["j_max_col"] - align["j_min_col"] + 1
    F = finest_size_px
    SF = F * subpixel_factor

    # Boulder area via sub-pixel block sums.
    boulder_pixel_count = (
        boulder_sub.reshape(n_jr, SF, n_jc, SF).sum(axis=(1, 3)).astype(np.int64)
    )
    boulder_area = boulder_pixel_count.astype(np.float64) * sub_pixel_area_m2

    # Mask eligibility: every CTX-pixel mask value in the tile must be 1.
    mask_crop = mask[r0:r1, c0:c1]
    mask_min = mask_crop.reshape(n_jr, F, n_jc, F).min(axis=(1, 3))
    eligible = (mask_min == 1)

    return {
        "boulder_area": boulder_area,
        "boulder_count": centroid_counts.astype(np.int64),
        "eligible": eligible,
    }


def _sum_up_ladder(
    finest: dict[str, np.ndarray],
    tile_sizes_px: list[int],
    px_x: float,
    px_y: float,
    j_min_row: int,
    j_min_col: int,
) -> list[dict[str, Any]]:
    """Walk the x2 ladder upward, halving cells per axis each step.

    Returns one dict per scale with arrays + absolute index offsets.
    """
    S_min = int(tile_sizes_px[0])
    scales: list[dict[str, Any]] = []
    scales.append({
        "scale_idx": 0,
        "tile_size_px": S_min,
        "tile_area_m2": S_min * S_min * px_x * px_y,
        "j_min_row": j_min_row,
        "j_min_col": j_min_col,
        "boulder_area": finest["boulder_area"],
        "boulder_count": finest["boulder_count"],
        "eligible": finest["eligible"],
    })

    for k in range(1, len(tile_sizes_px)):
        S = int(tile_sizes_px[k])
        prev = scales[-1]
        ny_prev, nx_prev = prev["boulder_area"].shape
        if ny_prev % 2 or nx_prev % 2:
            raise RuntimeError(
                f"sum-up ladder broke at scale {S}: prev shape ({ny_prev}, {nx_prev}) "
                "not divisible by 2 -- alignment math is wrong"
            )
        ny, nx = ny_prev // 2, nx_prev // 2
        area_k = prev["boulder_area"].reshape(ny, 2, nx, 2).sum(axis=(1, 3))
        count_k = prev["boulder_count"].reshape(ny, 2, nx, 2).sum(axis=(1, 3))
        eligible_k = prev["eligible"].reshape(ny, 2, nx, 2).all(axis=(1, 3))
        scales.append({
            "scale_idx": k,
            "tile_size_px": S,
            "tile_area_m2": S * S * px_x * px_y,
            "j_min_row": j_min_row // (S // S_min),
            "j_min_col": j_min_col // (S // S_min),
            "boulder_area": area_k,
            "boulder_count": count_k,
            "eligible": eligible_k,
        })
    return scales


def _flatten_to_dataframe(
    scales: list[dict[str, Any]],
    *,
    obs_id: str,
    mosaic_transform: list[float],
    px_x: float,
    px_y: float,
    labeling_cfg: dict,
) -> pd.DataFrame:
    """Build the tidy per-tile DataFrame from per-scale arrays. Drops ineligible tiles."""
    frames = []
    binary_area_threshold = float(labeling_cfg.get("binary_area_threshold", 0.0))
    binary_count_threshold = int(labeling_cfg.get("binary_count_threshold", 0))
    categorical_bins = labeling_cfg.get("categorical_bins") or []

    mx_origin_x = mosaic_transform[2]
    mx_origin_y = mosaic_transform[5]

    for sc in scales:
        eligible = sc["eligible"]
        if not eligible.any():
            continue
        ny, nx = eligible.shape
        li, lj = np.where(eligible)
        ti_abs = sc["j_min_row"] + li.astype(np.int64)
        tj_abs = sc["j_min_col"] + lj.astype(np.int64)
        S = sc["tile_size_px"]

        xmin = mx_origin_x + tj_abs * S * px_x
        xmax = mx_origin_x + (tj_abs + 1) * S * px_x
        ymax = mx_origin_y - ti_abs * S * px_y
        ymin = mx_origin_y - (ti_abs + 1) * S * px_y

        ba = sc["boulder_area"][li, lj].astype(np.float64)
        bc = sc["boulder_count"][li, lj].astype(np.int64)
        tile_area = float(sc["tile_area_m2"])
        frac = ba / tile_area
        density = bc / tile_area

        df = pd.DataFrame({
            "obs_id": obs_id,
            "scale_idx": int(sc["scale_idx"]),
            "tile_size_px": int(S),
            "tile_size_m": float(S * px_x),
            "ti": ti_abs,
            "tj": tj_abs,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "boulder_area": ba,
            "boulder_count": bc,
            "tile_area": tile_area,
            "fractional_area": frac,
            "binary_by_area": frac >= binary_area_threshold,
            "binary_by_count": bc >= binary_count_threshold,
            "count_density": density,
        })
        if categorical_bins:
            df["categorical"] = pd.cut(
                df["fractional_area"], bins=categorical_bins,
                labels=False, include_lowest=True,
            ).astype("Int64")
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "obs_id", "scale_idx", "tile_size_px", "tile_size_m",
                "ti", "tj", "xmin", "ymin", "xmax", "ymax",
                "boulder_area", "boulder_count", "tile_area",
                "fractional_area", "binary_by_area", "binary_by_count", "count_density",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def stage4_one_image(
    obs_id: str,
    *,
    cache_dir: str | Path,
    output_dir: str | Path,
    manifest_row,
    target_crs: str,
    labeling_cfg: dict,
    config_hash: str,
    subpixel_factor: int = DEFAULT_SUBPIXEL_FACTOR,
    apply_coreg_shift: bool = True,
) -> dict:
    """Generate per-tile labels for one ObsId and cache them to `output_dir/labels/`.

    Returns the provenance dict written alongside the parquet.

    Requires the Stage 1 / Stage 2 caches; reads the Stage 3 shift if available and
    `apply_coreg_shift=True`.
    """
    from . import ctx_tiles
    from . import coregister
    from . import detections as det
    from .ctx_retrieve import CTX_WINDOWS_SUBDIR

    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    labels_dir = output_dir / LABELS_SUBDIR
    labels_dir.mkdir(parents=True, exist_ok=True)

    ctx_window_tif = cache_dir / CTX_WINDOWS_SUBDIR / f"{obs_id}.tif"
    mask_tif = cache_dir / CTX_WINDOWS_SUBDIR / f"{obs_id}_hirise_mask.tif"
    if not ctx_window_tif.exists():
        raise FileNotFoundError(
            f"Stage 4 requires Stage 2 output {ctx_window_tif}. "
            f"Run scripts/run_stage2.py {obs_id} first."
        )
    if not mask_tif.exists():
        raise FileNotFoundError(
            f"Stage 4 requires HiRISE coverage mask {mask_tif}. "
            f"Re-run Stage 2 for {obs_id}."
        )

    import rasterio

    # ---- inputs ----
    window_h, window_w, window_transform, window_crs = _load_ctx_window(ctx_window_tif)

    gdf = det.load_reprojected(obs_id, cache_dir)
    # Reproject detections into the CTX window's CRS — the Murray Lab oblate frame the tile
    # grid is anchored to — so rasterization and the metre co-reg shift are exact and
    # correct-by-construction. NB the sphere (Mars_2000) and oblate (Mars_2015) equirectangular
    # definitions are numerically identical at our coordinates (PROJ's eqc uses the shared
    # semi-major radius → verified 0.000 m displacement, DECISIONS.md 2026-05-28), so this
    # changes no numbers today; it makes the consistency explicit + future-proofs a CTX
    # source whose CRS genuinely differs.
    if len(gdf) and window_crs is not None:
        gdf = gdf.to_crs(window_crs)
    gdf_pre_filter_n = len(gdf)
    gdf = _apply_detection_filters(gdf, labeling_cfg.get("detection_filters"))
    n_after_filter = len(gdf)

    shift = coregister.load_shift(obs_id, cache_dir) if apply_coreg_shift else None
    gdf = _apply_coreg_shift(gdf, shift)

    with rasterio.open(mask_tif) as src:
        mask = src.read(1)

    murray_tile = ctx_tiles.murray_tile_for_manifest_row(manifest_row)
    mosaic_transform = _load_mosaic_transform(cache_dir, murray_tile)

    px_x = abs(window_transform.a)
    px_y = abs(window_transform.e)
    # Sanity: the window's pixel size must match the mosaic's. If not, the integer-pixel
    # alignment claim is wrong and the grid wouldn't be nested cleanly.
    if not (
        abs(px_x - abs(mosaic_transform[0])) < 1e-6
        and abs(px_y - abs(mosaic_transform[4])) < 1e-6
    ):
        raise RuntimeError(
            f"{obs_id}: window pixel size ({px_x}, {px_y}) does not match parent mosaic "
            f"({abs(mosaic_transform[0])}, {abs(mosaic_transform[4])}). Stage 2 should have "
            "preserved this -- investigate cache/ctx_tiles/{murray_tile}.json."
        )

    tile_sizes_px = list(labeling_cfg["tile_sizes_px"])
    if labeling_cfg.get("grid_anchor") != "ctx_pixel_origin":
        raise ValueError(
            f"{obs_id}: labeling.grid_anchor must be 'ctx_pixel_origin' "
            f"(got {labeling_cfg.get('grid_anchor')!r}); other anchors are not implemented."
        )

    align = _compute_grid_alignment(
        window_transform, mosaic_transform, window_h, window_w, tile_sizes_px,
    )

    # ---- finest-grid base stats ----
    boulder_sub = _rasterize_boulders_subpixel(
        gdf, window_transform, align, subpixel_factor,
    )
    sub_pixel_area_m2 = (px_x / subpixel_factor) * (px_y / subpixel_factor)
    centroid_counts = _count_centroids_per_finest_cell(
        gdf, mosaic_transform, align, int(tile_sizes_px[0]), px_x, px_y,
    )
    finest = _build_finest_stats(
        boulder_sub, mask, align,
        finest_size_px=int(tile_sizes_px[0]),
        subpixel_factor=subpixel_factor,
        sub_pixel_area_m2=sub_pixel_area_m2,
        centroid_counts=centroid_counts,
    )

    # ---- sum up the x2 ladder ----
    scales = _sum_up_ladder(
        finest, tile_sizes_px, px_x, px_y,
        j_min_row=align["j_min_row"], j_min_col=align["j_min_col"],
    )

    # ---- flatten + apply derived label columns ----
    df = _flatten_to_dataframe(
        scales,
        obs_id=obs_id,
        mosaic_transform=mosaic_transform,
        px_x=px_x, px_y=px_y,
        labeling_cfg=labeling_cfg,
    )
    df["config_hash"] = config_hash

    # ---- write parquet + provenance sidecar ----
    parquet_path = labels_dir / f"{obs_id}.parquet"
    sidecar_path = labels_dir / f"{obs_id}.json"
    df.to_parquet(parquet_path, index=False)

    eligible_counts_per_scale = {
        int(sc["tile_size_px"]): int(sc["eligible"].sum()) for sc in scales
    }
    total_tiles_per_scale = {
        int(sc["tile_size_px"]): int(sc["eligible"].size) for sc in scales
    }

    provenance = {
        "obs_id": obs_id,
        "n_polygons_stage1": int(gdf_pre_filter_n),
        "n_polygons_after_filter": int(n_after_filter),
        "detection_filters": labeling_cfg.get("detection_filters") or {
            "min_confidence": None, "min_size_m": None,
        },
        "coreg_shift_applied": bool(shift is not None),
        "coreg_shift_m": (
            {
                "dx": float(shift["shift_m"]["dx"]),
                "dy": float(shift["shift_m"]["dy"]),
                "magnitude": float(shift["shift_m"]["magnitude"]),
            } if shift is not None else None
        ),
        "coreg_peak_correlation": (
            float(shift["peak_correlation"]) if shift is not None else None
        ),
        "tile_sizes_px": [int(s) for s in tile_sizes_px],
        "tile_sizes_m": [float(int(s) * px_x) for s in tile_sizes_px],
        "grid_anchor": "ctx_pixel_origin",
        "mosaic_row_origin": int(align["mosaic_row_origin"]),
        "mosaic_col_origin": int(align["mosaic_col_origin"]),
        "finest_grid_cells": list(finest["eligible"].shape),
        "eligibility_rule": ELIGIBILITY_RULE,
        "eligible_tiles_per_scale": eligible_counts_per_scale,
        "total_candidate_tiles_per_scale": total_tiles_per_scale,
        "subpixel_factor": int(subpixel_factor),
        "subpixel_area_m2": float(sub_pixel_area_m2),
        "binary_area_threshold": float(labeling_cfg.get("binary_area_threshold", 0.0)),
        "binary_count_threshold": int(labeling_cfg.get("binary_count_threshold", 0)),
        "categorical_bins": list(labeling_cfg.get("categorical_bins") or []),
        "label_type_primary": labeling_cfg.get("label_type", "fractional_area"),
        "ctx_window_tif": str(ctx_window_tif),
        "hirise_mask_tif": str(mask_tif),
        "parquet_path": str(parquet_path),
        "config_hash": config_hash,
        "written_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    sidecar_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance


def load_labels(obs_id: str, output_dir: str | Path) -> pd.DataFrame:
    """Load a Stage 4 per-tile label table."""
    parquet = Path(output_dir) / LABELS_SUBDIR / f"{obs_id}.parquet"
    return pd.read_parquet(parquet)


def load_provenance(obs_id: str, output_dir: str | Path) -> dict:
    """Load a Stage 4 provenance sidecar."""
    sidecar = Path(output_dir) / LABELS_SUBDIR / f"{obs_id}.json"
    return json.loads(sidecar.read_text(encoding="utf-8"))
