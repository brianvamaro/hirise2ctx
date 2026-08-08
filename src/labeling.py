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

# R68. Tolerances for the window <-> parent-mosaic grid check. NB these two 1e-6 literals
# are in DIFFERENT UNITS: pixel size in METRES, phase in mosaic PIXELS. Do not "unify" them.
# Phase: measured over all 49 cached windows (cache/ + cache_v2/), the worst residual is
# 1.38e-10 px -- not bit-zero, so an `== 0` test would break on real data. The only source
# of a non-zero residual on the producing path is float rounding in the affine arithmetic
# plus the GeoTIFF header round-trip, bounded by ~1e-9 px at Murray Lab coordinate
# magnitudes. 1e-6 px = 5 um of ground: ~3 orders above the noise, ~5 orders below the
# smallest consequential break (one sub-pixel of the 5x rasteriser = 0.2 px).
GRID_PHASE_TOL_PX = 1e-6
GRID_PIXEL_SIZE_TOL_M = 1e-6


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


def _apply_detection_filters(gdf, filters: dict | None, *, diagnostics: dict | None = None):
    """Drop polygons failing `min_confidence` (DBF `score`) or `min_size_m` (derived diameter).

    `min_size_m` is interpreted as a minimum equivalent-circle **diameter**,
    `2*sqrt(area/pi)`, measured in the CRS the GeoDataFrame is currently in — which by the
    time Stage 4 calls this is the projected CTX frame, **not** the image's own source
    frame. That distinction is R03/R80's mechanism and is recorded, not corrected; see
    `_describe_realised_size_basis`.

    Returns a (possibly identical) GeoDataFrame. When `diagnostics` is given it is filled
    in place with per-floor counts and areas — an out-parameter rather than a changed
    return type so the sole caller (`stage4_one_image`) keeps its signature, and so the
    pre- and post-filter frames are never both held (727k polygons on `ESP_068483_2280`).

    Only keys that were actually **measured** are written. A numeric key is absent rather
    than zero when the corresponding computation did not run — a seeded `area_total_m2: 0.0`
    on a non-empty image would be a positive false claim, not a missing measurement.
    """
    _f = filters or {}
    if diagnostics is not None:
        diagnostics.update(
            n_in=int(len(gdf)),
            n_dropped_by_size=0,
            n_dropped_by_confidence=0,
            # Configured-ness is known before anything runs, and must not be confused with
            # applied-ness: a configured confidence floor is silently skipped when there is
            # no `score` column, and an empty image applies neither.
            size_floor_configured=_f.get("min_size_m") is not None,
            confidence_floor_configured=_f.get("min_confidence") is not None,
            size_floor_applied=False,
            confidence_floor_applied=False,
        )
    if filters is None or len(gdf) == 0:
        if diagnostics is not None and len(gdf) == 0 and _f:
            diagnostics["note"] = "no polygons to filter, so neither floor was applied"
        return gdf
    keep = np.ones(len(gdf), dtype=bool)

    min_conf = filters.get("min_confidence")
    conf_applied = min_conf is not None and "score" in gdf.columns
    if conf_applied:
        fail_conf = ~(gdf["score"].to_numpy() >= float(min_conf))
        keep &= ~fail_conf
    else:
        fail_conf = np.zeros(len(gdf), dtype=bool)

    min_size_m = filters.get("min_size_m")
    size_applied = min_size_m is not None
    area = gdf.geometry.area.to_numpy() if (size_applied or diagnostics is not None) else None
    if size_applied:
        diam = 2.0 * np.sqrt(area / np.pi)
        fail_size = ~(diam >= float(min_size_m))
        keep &= ~fail_size
    else:
        fail_size = np.zeros(len(gdf), dtype=bool)

    if diagnostics is not None:
        # Attribute each floor independently, so a polygon failing BOTH is not counted
        # twice and "configured but silently not applied" (no `score` column) is
        # distinguishable from "applied and dropped nothing".
        diagnostics.update(
            n_dropped_by_size=int(fail_size.sum()),
            n_dropped_by_confidence=int(fail_conf.sum()),
            size_floor_applied=bool(size_applied),
            confidence_floor_applied=bool(conf_applied),
            size_floor_was_binding=bool(fail_size.any()),
        )
        if area is not None:
            diagnostics["area_total_m2"] = float(area.sum())
            diagnostics["area_dropped_by_size_m2"] = float(area[fail_size].sum())
        if size_applied and (~fail_size).any():
            # Survivors of the SIZE floor alone -- mixing in the confidence floor would
            # make this describe a population the size basis is not about.
            kept_diam = 2.0 * np.sqrt(area[~fail_size] / np.pi)
            diagnostics.update(
                min_surviving_diameter_ctx_frame_m=float(kept_diam.min()),
                diameter_p1_ctx_frame_m=float(np.percentile(kept_diam, 1)),
                diameter_median_ctx_frame_m=float(np.percentile(kept_diam, 50)),
            )
        if min_conf is not None and not conf_applied:
            diagnostics["confidence_floor_note"] = (
                "min_confidence is configured but no `score` column is present, so the "
                "confidence floor was NOT applied"
            )

    if keep.all():
        return gdf
    return gdf.iloc[keep].reset_index(drop=True)


_COREG_MASK_SHIFT_VERSION = 1


def _shift_coverage_mask(mask, shift, px_x: float, px_y: float) -> tuple:
    """Translate the HiRISE coverage mask by the same Stage-3 (dx, dy) as the polygons.

    R29/R75. Stage 4 translates every detection polygon by the Stage-3 shift but used to
    gate eligibility with a coverage mask reprojected from the **unshifted** HiRISE
    product. The shift is a whole-product geolocation offset, so the mask and the polygons
    must move together; leaving the mask still opens an L-shaped strip along the receding
    edges (dy>0 in 38/38 images, dx>0 in 30/38) that stays `eligible` while no detection
    can possibly land in it. Measured over 38 images / 161,005 S=32 tiles: **340 tiles
    (0.21 %) are zero by construction** and a further **5,862 (3.85 % in total) have a
    partially depressed `fractional_area`**, always at the same edges and the same sign,
    which also biases any edge-vs-interior diagnostic.

    Vacated area is filled with 0 (**not** eligible): we have no HiRISE coverage evidence
    there, and eligibility already requires every CTX pixel in a tile to be covered, so a
    tile overlapping the vacated strip correctly drops out rather than reporting a
    depressed abundance.

    Rounding note: the register asserts the shifts are "already quantised to CTX pixels".
    **They are not** -- measured 2026-08-06, 0 of 39 are integer-pixel; they are quantised
    to 1/20 px by the phase-correlation upsampling. A raster mask can only move by whole
    pixels without resampling, so we round to nearest. The residual is <= 0.5 px (2.5 m)
    against a median shift of 194.7 m -- a ~78x reduction, and far below the 160 m tile.

    Returns `(shifted_mask, provenance)`.
    """
    prov: dict = {"method": "integer_pixel_roll", "version": _COREG_MASK_SHIFT_VERSION}
    if shift is None:
        prov.update(applied=False, reason="no Stage-3 shift for this image")
        return mask, prov

    dx = float(shift["shift_m"]["dx"])
    dy = float(shift["shift_m"]["dy"])
    # +dx is east -> +columns. +dy is north -> DECREASING row index (north-up raster).
    dcol = int(round(dx / px_x))
    drow = int(round(-dy / px_y))
    prov.update(
        shift_m={"dx": dx, "dy": dy},
        shift_px={"drow": drow, "dcol": dcol},
        residual_m={
            "dx": float(dx - dcol * px_x),
            "dy": float(dy + drow * px_y),
        },
        n_eligible_px_before=int((mask == 1).sum()),
    )
    if drow == 0 and dcol == 0:
        prov.update(applied=False, reason="shift rounds to zero pixels")
        prov["n_eligible_px_after"] = prov["n_eligible_px_before"]
        return mask, prov

    h, w = mask.shape
    out = np.zeros_like(mask)
    dr0, dr1 = max(0, drow), min(h, h + drow)
    sr0, sr1 = max(0, -drow), min(h, h - drow)
    dc0, dc1 = max(0, dcol), min(w, w + dcol)
    sc0, sc1 = max(0, -dcol), min(w, w - dcol)
    if dr1 > dr0 and dc1 > dc0:
        out[dr0:dr1, dc0:dc1] = mask[sr0:sr1, sc0:sc1]
    prov.update(applied=True, n_eligible_px_after=int((out == 1).sum()))
    return out, prov


def _describe_realised_label_basis(gdf, cache_dir, obs_id: str, detection_filters=None) -> dict:
    """The confidence floor these labels were ACTUALLY built at, per image.

    `detection_filters` records the *configured* floor, which is byte-identical across all
    38 vClaire sidecars (`min_confidence: null`) and therefore cannot distinguish an image
    labelled at score >= 0.10 from one labelled at score >= 0.617. The realised floor --
    the minimum `score` actually surviving into the labels -- can, and that difference is
    R23. Brian's 2026-08-06 decision is to RETAIN the resulting mixed floor and DOCUMENT
    it (a temporary measure pending the v3 re-detection), which makes this record the
    thing that carries the decision downstream. See DECISIONS 2026-08-06o.

    Mirrors the shape of the mixed *size*-floor convention (R03/R83/R84): per-image basis
    persisted, product-level mixture describable by aggregating over images.
    """
    out: dict = {
        "convention": "mixed_per_image_confidence_floor",
        "temporary_pending": "v3 re-detection",
        "decision": "DECISIONS 2026-08-06o (retain + document; do not silently harmonise)",
    }
    if len(gdf) and "score" in gdf.columns:
        s = pd.to_numeric(gdf["score"], errors="coerce").to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if s.size:
            out["realised_score_floor"] = float(s.min())
            out["score_max"] = float(s.max())
            out["score_p1"] = float(np.percentile(s, 1))
            out["score_median"] = float(np.percentile(s, 50))
    else:
        out["note"] = "no `score` column on the cached detections; floor not derivable"

    # `level_claims_unsafe` must NOT depend solely on the Stage-1 sidecar: every sidecar
    # banked before 2026-08-06 predates `source_integrity`, so keying off it alone would
    # leave the flag absent on exactly the affected images, and absence would be
    # indistinguishable from "checked and clean". Derive it from the realised floor first
    # -- that is measured right here, from the labels themselves.
    #
    # The comparison floor is the CONFIGURED cut when one is set, else BoulderNet's own
    # detector floor, measured 2026-08-06 as exactly 0.100000 in all 39 readable v2 .dbf
    # files. A realised floor materially above that means detections were lost upstream.
    detector_floor = 0.100000
    configured = (detection_filters or {}).get("min_confidence")
    expected_floor = float(configured) if configured is not None else detector_floor
    realised = out.get("realised_score_floor")
    if realised is not None and realised > expected_floor + 1e-6:
        out["level_claims_unsafe"] = True
        out["realised_floor_exceeds_expected_by"] = float(realised - expected_floor)

    # Corroborate from Stage 1 where available; where the sidecar predates the field, say
    # so explicitly and re-derive from the source path it does record, so a Stage-4-only
    # re-run still gets the finding without a Stage-1 rebuild.
    from . import detections as _det

    s1: dict = {}
    try:
        s1 = json.loads(
            (Path(cache_dir) / _det.CACHE_SUBDIR / f"{obs_id}.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        out["stage1_provenance"] = "unavailable (Stage-1 sidecar missing or unreadable)"

    integrity = s1.get("source_integrity")
    if integrity is None and s1.get("source_path"):
        integrity = _det.inspect_shapefile_integrity(s1["source_path"])
        out["stage1_provenance"] = (
            "re-derived from source_path (sidecar predates 2026-08-06 `source_integrity`)"
        )
    integrity = integrity or {}
    basis = s1.get("null_geometry_basis") or {}

    if integrity.get("status") == "truncated":
        out["source_truncated"] = True
        out["source_missing_bytes"] = integrity.get("missing_bytes")
    elif integrity.get("status") == "complete":
        out["source_truncated"] = False
    else:
        out["source_truncated"] = None  # unknown -- never read absence as safety
    if basis:
        out["stage1_rank_truncation"] = bool(basis.get("is_rank_truncation"))
        out["stage1_dropped_fraction"] = basis.get("dropped_fraction")

    if out.get("source_truncated") or out.get("stage1_rank_truncation"):
        out["level_claims_unsafe"] = True
    if out.get("level_claims_unsafe"):
        out["level_claims_note"] = (
            "This image's labels are a high-confidence subset of its detections, so its "
            "per-image abundance LEVEL is biased low. Safe for rank-only statistics; "
            "exclude from per-image level claims (calibration pool, mean(pred)/mean(true), "
            "thermal comparisons) unless the bias is corrected."
        )
    return out


def _crs_name_and_projected(crs) -> tuple[str | None, bool | None]:
    """`(short name, is_projected)` for a rasterio CRS, a pyproj CRS, a WKT string or None.

    `rasterio.crs.CRS` has **no** `.name` attribute, so a naive `getattr` chain falls
    through to `to_string()` and records a 400+ character WKT blob in every sidecar.
    Normalising through pyproj gives the same short name for every input type, which also
    means the tests exercise the branch production takes.
    """
    if crs is None:
        return None, None
    try:
        from pyproj import CRS as _CRS

        parsed = _CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
        return str(parsed.name), bool(parsed.is_projected)
    except Exception:  # noqa: BLE001 -- provenance must never break label generation
        try:
            return str(crs)[:120], bool(crs.is_projected)
        except Exception:  # noqa: BLE001
            return None, None


def _equirect_params(crs) -> tuple[float, float] | None:
    """`(semi_major_m, standard_parallel_deg)` for an equirectangular CRS, else None.

    Read via `coordinate_operation.params` rather than `to_dict()`/`to_proj4()`, both of
    which emit `UserWarning: You will likely lose important projection information` — a new
    warning inside a producer would be noise in exactly the suite whose acceptance signal
    is that a warning disappeared.
    """
    if crs is None:
        return None
    try:
        from pyproj import CRS as _CRS

        parsed = _CRS.from_user_input(crs.to_wkt() if hasattr(crs, "to_wkt") else crs)
        radius = float(parsed.ellipsoid.semi_major_metre)
        lat_ts = 0.0
        op = parsed.coordinate_operation
        if op is not None:
            for p in op.params:
                if "standard parallel" in p.name.lower():
                    lat_ts = float(p.value)
                    break
        return radius, lat_ts
    except Exception:  # noqa: BLE001
        return None


def _source_to_target_diameter_scale(source_crs_wkt, window_crs) -> float | None:
    """How much an equivalent-circle diameter grows from the source frame to the CTX frame.

    Equirectangular maps easting by `R*cos(lat_ts)` and northing by `R`, so area scales by
    `(R_t/R_s)**2 * cos(lat_ts_t)/cos(lat_ts_s)` and an equivalent-circle **diameter** by
    the square root of that. Returns None when either frame is unavailable or not
    equirectangular — the caller must then record "unknown", never "equal".
    """
    src = _equirect_params(source_crs_wkt)
    tgt = _equirect_params(window_crs)
    if src is None or tgt is None:
        return None
    r_s, lat_s = src
    r_t, lat_t = tgt
    cos_s, cos_t = math.cos(math.radians(lat_s)), math.cos(math.radians(lat_t))
    if not (r_s > 0 and cos_s > 0 and cos_t > 0):
        return None
    return math.sqrt((r_t / r_s) ** 2 * (cos_t / cos_s))


def _describe_realised_size_basis(
    diagnostics: dict, window_crs, detection_filters, source_crs_wkt=None,
) -> dict:
    """The physical size floor these labels were ACTUALLY built at, per image.

    The size-floor analogue of `_describe_realised_label_basis`, and for the same reason:
    `detection_filters` records the *configured* `min_size_m` and is byte-identical across
    all 38 v2 sidecars, so it cannot express the mixture R03/R83/R84 found. Brian's
    2026-08-06 decision is to **retain and document** that mixture, which makes this the
    field that carries it downstream. See DECISIONS 2026-08-06u.

    Two things are deliberately recorded rather than corrected:

    * **The floor is applied in the projected CTX frame.** Polygons reach Stage 4 already
      reprojected into the clon_0 target CRS, where easting is stretched relative to each
      image's own source standard parallel, so the realised *physical* floor is looser than
      the configured metres and differs image to image (measured over the 39 v2 images:
      0.993-1.367 m realised against a configured 1.4105 m). Moving the filter earlier, or
      dividing by the scale here, would delete a further ~0.4-3 % of each fine-cohort
      image's polygons — i.e. redefine the target. That needs its own decision.
    * **The realised floor is measured, not assumed.** `realised_diameter_floor_m` is the
      smallest surviving equivalent-circle diameter, exactly as `realised_score_floor` is
      the smallest surviving score.

    Deliberately NOT recorded: a `detector_min_size_px` / "binding floor" pair. It was
    drafted and then refuted by measurement — the detections do not obey a 5-pixel floor,
    so publishing one as provenance would assert something false.
    """
    filters = detection_filters or {}
    configured = filters.get("min_size_m")
    out: dict = {
        "convention": "mixed_per_image_size_floor",
        "temporary_pending": "v3 re-detection / common-floor decision",
        "decision": "DECISIONS 2026-08-06u (retain + document; do not silently harmonise)",
        "size_metric": "equivalent_circle_diameter_2sqrt_area_over_pi",
        "configured_min_size_m": (float(configured) if configured is not None else None),
        "configured_min_area_m2": (
            float(math.pi * (float(configured) / 2.0) ** 2) if configured is not None else None
        ),
    }
    out["measured_in_crs"], out["measured_in_crs_is_projected"] = _crs_name_and_projected(
        window_crs
    )
    out["measured_in_frame"] = (
        f"areas measured in {out['measured_in_crs']}"
        if out["measured_in_crs"] else "unknown -- window CRS unavailable"
    )

    for k in ("n_in", "n_dropped_by_size", "n_dropped_by_confidence",
              "size_floor_configured", "confidence_floor_configured",
              "size_floor_applied", "confidence_floor_applied", "size_floor_was_binding",
              "area_total_m2", "area_dropped_by_size_m2",
              "min_surviving_diameter_ctx_frame_m", "diameter_p1_ctx_frame_m",
              "diameter_median_ctx_frame_m", "confidence_floor_note", "note"):
        if k in diagnostics:
            out[k] = diagnostics[k]

    # --- the per-image number this block exists to carry ---------------------------------
    # The floor is enforced on areas measured in the CTX frame, but the *physical* floor it
    # corresponds to depends on each image's own source projection. Emit the scale and the
    # resulting physical floor, so a product-level mixture statement can be aggregated from
    # the sidecars. Do NOT assert a direction: for an image whose source frame already
    # equals the CTX frame the scale is exactly 1.0 and nothing is loosened (measured on
    # v1's ESP_039820_1750: source lat_ts=0, R=3396190, scale 1.000000000000).
    scale = _source_to_target_diameter_scale(source_crs_wkt, window_crs)
    out["source_crs_available"] = scale is not None
    if scale is not None:
        out["source_to_target_diameter_scale"] = float(scale)
        if configured is not None:
            physical = float(configured) / float(scale)
            out["realised_physical_min_size_m"] = physical
            out["realised_physical_min_area_m2"] = float(math.pi * (physical / 2.0) ** 2)
            out["realised_floor_is_looser_than_configured"] = bool(scale > 1.0 + 1e-9)
            out["realised_floor_note"] = (
                "min_size_m is applied after reprojection into the CTX frame. This image's "
                f"source->target equivalent-circle-diameter scale is {scale:.6f}, so the "
                f"physical floor actually enforced is {physical:.4f} m against a configured "
                f"{float(configured):.4f} m. Retained and documented, not corrected -- "
                "see DECISIONS 2026-08-06u."
            )
    elif configured is not None:
        # Never let absence read as "checked and equal".
        out["realised_floor_is_looser_than_configured"] = None
        out["realised_floor_note"] = (
            "the image's source CRS is unavailable, so the physical floor this configured "
            "min_size_m corresponds to could not be derived. Absence is unknown, not equal."
        )
    n_in = diagnostics.get("n_in") or 0
    if n_in:
        out["dropped_by_size_fraction"] = float(diagnostics.get("n_dropped_by_size", 0) / n_in)
    total_area = diagnostics.get("area_total_m2") or 0.0
    if total_area:
        out["dropped_by_size_area_fraction"] = float(
            diagnostics.get("area_dropped_by_size_m2", 0.0) / total_area
        )
    return out


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

    # ---- R68: check the property this rounding silently assumes -------------------------
    # Stage 4's x2 tile ladder is anchored on ABSOLUTE mosaic-pixel indices, so the window's
    # upper-left must sit at an INTEGER mosaic-pixel offset from the parent tile's origin.
    # `int(round(...))` below discards any fractional phase without complaint. Nothing else
    # checks it: Stage 4's old "runtime pixel-size guard" compared the window's pixel size
    # against the parent mosaic's, but Stage 2 cuts the window from the SAME /vsizip/ handle
    # whose transform it wrote into the tile sidecar, so `a`/`e` agree bit-identically and
    # the comparison was a tautology that could not fire on any pipeline-reachable input.
    # A fractional phase matters because the two halves of Stage 4 are anchored differently:
    # `_rasterize_boulders_subpixel` and the eligibility crop are WINDOW-anchored (r0_win /
    # c0_win) while `_count_centroids_per_finest_cell` and the emitted bbox are
    # MOSAIC-anchored, so a half-pixel phase slides them apart -- measured on a synthetic
    # +0.5 px window, a 2x2 m boulder reports boulder_count = 1 while boulder_area is 0.0.
    # See DECISIONS 2026-08-06r.
    mx_px, my_px = abs(mosaic_transform[0]), abs(mosaic_transform[4])
    if not (abs(px_x - mx_px) < GRID_PIXEL_SIZE_TOL_M
            and abs(px_y - my_px) < GRID_PIXEL_SIZE_TOL_M):
        raise RuntimeError(
            f"CTX window pixel size ({px_x}, {px_y}) m does not match its parent mosaic "
            f"({mx_px}, {my_px}) m -- the tile sidecar and the window were written from "
            "different products. Re-run Stage 2 for this image."
        )

    col_f = (window_transform.c - mx_origin_x) / px_x
    # NB the row quotient's sign is the OPPOSITE of the column's: e < 0, so the mosaic row
    # index grows as f decreases. Copying the column form onto the row axis is a real trap.
    row_f = (mx_origin_y - window_transform.f) / px_y
    mosaic_col_origin = int(round(col_f))
    mosaic_row_origin = int(round(row_f))
    d_col = abs(col_f - mosaic_col_origin)
    d_row = abs(row_f - mosaic_row_origin)
    if d_row > GRID_PHASE_TOL_PX or d_col > GRID_PHASE_TOL_PX:
        raise RuntimeError(
            f"CTX window origin is NOT on its parent mosaic's pixel lattice. Phase residual "
            f"(row, col) = ({d_row:.6g}, {d_col:.6g}) px = ({d_row * px_y:.4g}, "
            f"{d_col * px_x:.4g}) m, tolerance {GRID_PHASE_TOL_PX:g} px. The x2 label ladder "
            "is anchored on absolute mosaic-pixel indices, so a fractional phase displaces "
            "the window-anchored boulder raster from the mosaic-anchored (ti, tj) bbox."
        )
    if mosaic_row_origin < 0 or mosaic_col_origin < 0:
        # Defence in depth for R31: a window whose origin is outside its parent tile was
        # written with the requested (un-cropped) transform while rasterio clipped the read,
        # so it is misregistered by exactly the overhang. R31 now refuses to write one, but
        # an already-cached window can still carry it -- v1's ESP_057469_2215 is off by
        # 1,924 px (9.6 km) west today. Its phase is still integer, so only this clause
        # catches it.
        raise RuntimeError(
            f"CTX window origin ({mosaic_row_origin}, {mosaic_col_origin}) in parent-mosaic "
            "pixels is negative, i.e. outside the parent tile. The window overhangs its "
            "Murray Lab tile and its georeferencing cannot be trusted (R31). Re-run Stage 2."
        )

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


def _upstream_identity(
    cache_dir: Path, obs_id: str, ctx_window_tif: Path, mask_tif: Path, shift: dict | None,
) -> dict:
    """Content identity of the Stage 2 / Stage 3 artifacts these labels were built from.

    **R74.** Recording pathnames cannot distinguish generations, and a YAML config hash is
    identical either side of an algorithm change. This records the digests of the CTX
    window and coverage mask actually read, the coverage-mask algorithm identity copied
    from the Stage 2 sidecar, and the Stage 3 solve's `shift_id`.
    """
    from . import ctx_retrieve
    from .ctx_retrieve import CTX_WINDOWS_SUBDIR

    stage2 = cache_dir / CTX_WINDOWS_SUBDIR / f"{obs_id}.json"
    mask_prov = None
    if stage2.exists():
        mask_prov = json.loads(stage2.read_text(encoding="utf-8")).get("hirise_mask")
    return {
        "ctx_window_sha256": ctx_retrieve.file_sha256(ctx_window_tif),
        "hirise_mask_sha256": ctx_retrieve.file_sha256(mask_tif),
        "coverage_mask": mask_prov,
        "coreg_shift_id": (shift or {}).get("shift_id"),
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
    shift_coverage_mask: bool = True,
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
    size_diag: dict = {}
    gdf = _apply_detection_filters(
        gdf, labeling_cfg.get("detection_filters"), diagnostics=size_diag,
    )
    n_after_filter = len(gdf)
    realised_basis = _describe_realised_label_basis(
        gdf, cache_dir, obs_id, labeling_cfg.get("detection_filters"),
    )
    # The image's own source CRS, for the per-image physical-floor scale. Same Stage-1
    # sidecar `_describe_realised_label_basis` reads; absent on a pre-Stage-1 cache, in
    # which case the size basis records "unknown" rather than assuming equality.
    _s1_wkt = None
    try:
        from . import detections as _det_mod

        _s1_wkt = json.loads(
            (Path(cache_dir) / _det_mod.CACHE_SUBDIR / f"{obs_id}.json").read_text(
                encoding="utf-8"
            )
        ).get("source_crs_wkt")
    except (OSError, ValueError):
        pass
    realised_size_basis = _describe_realised_size_basis(
        size_diag, window_crs, labeling_cfg.get("detection_filters"), _s1_wkt,
    )

    shift = coregister.load_shift(obs_id, cache_dir) if apply_coreg_shift else None
    gdf = _apply_coreg_shift(gdf, shift)

    with rasterio.open(mask_tif) as src:
        mask = src.read(1)

    murray_tile = ctx_tiles.murray_tile_for_manifest_row(manifest_row)
    mosaic_transform = _load_mosaic_transform(cache_dir, murray_tile)

    px_x = abs(window_transform.a)
    px_y = abs(window_transform.e)
    # R68: the pixel-size "sanity" check that used to live here was a tautology -- Stage 2
    # cuts the window from the same /vsizip/ handle whose transform it wrote into the tile
    # sidecar, so the two agree bit-identically and it could not fire. (Its own error
    # message gave it away: a non-f-string containing a literal `{murray_tile}` placeholder
    # and a hardcoded `cache/`, i.e. it had never once been rendered.) The real property --
    # that the window origin sits on the parent mosaic's integer pixel lattice -- is now
    # checked inside `_compute_grid_alignment`, where the rounding that assumes it happens.

    # R29/R75: the coverage mask must move with the polygons, or an L-shaped strip along
    # the receding edges stays eligible while no detection can land in it.
    if shift_coverage_mask:
        mask, coreg_mask_shift = _shift_coverage_mask(mask, shift, px_x, px_y)
    else:
        coreg_mask_shift = {
            "method": "integer_pixel_roll", "version": _COREG_MASK_SHIFT_VERSION,
            "applied": False, "reason": "disabled by caller (shift_coverage_mask=False)",
        }

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
        # The CONFIGURED filter above is identical across every image and so cannot
        # express the mixed basis R23 found; this is the REALISED one. See DECISIONS
        # 2026-08-06o.
        "realised_label_basis": realised_basis,
        # The realised SIZE floor, the analogue of the above for R03/R83/R84's mixed
        # physical-size convention. See DECISIONS 2026-08-06u.
        "realised_size_basis": realised_size_basis,
        "coreg_shift_applied": bool(shift is not None),
        # R29/R75: whether the coverage mask was translated with the polygons. Pre- and
        # post-fix labels are otherwise indistinguishable (the Pattern-D failure), so this
        # block is the generation marker. See DECISIONS 2026-08-06p.
        "coreg_mask_shift": coreg_mask_shift,
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
        # R74: which *generation* of the upstream artifacts these labels were built from.
        # Stage 4's `mask_min == 1` eligibility rule means the coverage mask decides which
        # tiles exist at all, so a pre-R74 and a post-R74 label table differ in row set
        # while sharing a config hash. Pathnames cannot distinguish them; these can.
        "inputs": _upstream_identity(cache_dir, obs_id, ctx_window_tif, mask_tif, shift),
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
