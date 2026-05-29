"""Stage 1 — ingest BoulderNet detections, reproject to a common CTX CRS, cache.

The reprojection is intentionally boring: geopandas + pyproj read each shapefile's own
`.prj` (which carries the per-image local-Mars-radius equirectangular CRS) and project
to the target CTX CRS. The whole point of this stage is to NEVER bypass the source CRS
or hardcode a sphere radius — see CLAUDE.md §3.3.

Exception: 4 of 10 BoulderNet `.prj` files in the priority10 manifest are mis-labelled
with `Standard_Parallel_1=0` (datum `D_unnamed`) even though the geometry was actually
generated with the PDS-declared projection latitude. We detect that case and override
SP1 with the authoritative value from the HiRISE `.LBL` (CENTER_LATITUDE). See
`DECISIONS.md` 2026-05-20 entries.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

import geopandas as gpd
from pyproj import CRS

from . import manifest as manifest_mod
from . import pds_labels

CACHE_SUBDIR = "reprojected_detections"

# Matches ESRI-WKT1-style `PARAMETER["Standard_Parallel_1",<num>]`
_SP1_PATTERN = re.compile(r'PARAMETER\["Standard_Parallel_1",([-\d.eE]+)\]')
# Bug fingerprint: BoulderNet's mis-labelled exports use the `D_unnamed` / `unnamed`
# placeholder strings instead of the canonical `D_MARS` / `MARS_localRadius`.
_BAD_DATUM_FINGERPRINT = re.compile(r'DATUM\["D_unnamed"', re.IGNORECASE)
# Tolerance: if the .prj's SP1 is within this many degrees of the manifest CenterLat,
# trust the .prj. The buggy files are off by tens of degrees; the good files are within
# ~5°. 15° is a generous margin that cleanly separates the two regimes.
_SP1_TOLERANCE_DEG = 15.0


def _suspect_sp1(prj_text: str, image_lat_deg: float) -> tuple[bool, float | None]:
    """Return `(is_buggy, current_sp1)`. `is_buggy` is True iff the .prj looks like a
    BoulderNet mis-labelled export AND its SP1 disagrees with `image_lat_deg` by more
    than `_SP1_TOLERANCE_DEG`.
    """
    sp1_match = _SP1_PATTERN.search(prj_text)
    if not sp1_match:
        return False, None
    current_sp1 = float(sp1_match.group(1))
    has_bad_datum = bool(_BAD_DATUM_FINGERPRINT.search(prj_text))
    far_from_image = abs(current_sp1 - image_lat_deg) > _SP1_TOLERANCE_DEG
    return (has_bad_datum and far_from_image), current_sp1


def _override_sp1(prj_text: str, new_sp1_deg: float) -> str:
    """Return `prj_text` with `Standard_Parallel_1` set to `new_sp1_deg`."""
    return _SP1_PATTERN.sub(f'PARAMETER["Standard_Parallel_1",{new_sp1_deg}]', prj_text)


def read_detection_shapefile(
    obs_id: str,
    detections_root: str | Path,
    *,
    manifest_row=None,
    cache_dir: str | Path | None = None,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Read the BoulderNet shapefile for `obs_id`, returning the GeoDataFrame in its
    native CRS (after correction, if needed) plus a small `correction` dict for provenance.

    When `manifest_row` and `cache_dir` are given, the function checks for the
    BoulderNet `.prj` SP1 mis-labelling bug and, if found, fetches the PDS `.LBL` for
    `obs_id` and overrides SP1 with the authoritative `CENTER_LATITUDE` from the label.
    """
    shp = manifest_mod.find_shapefile(obs_id, detections_root)
    prj_path = shp.with_suffix(".prj")
    original_prj = prj_path.read_text(encoding="latin-1")
    correction: dict = {"status": "trusted_prj"}

    if manifest_row is not None and cache_dir is not None:
        image_lat = float(manifest_row["CenterLat"])
        is_buggy, current_sp1 = _suspect_sp1(original_prj, image_lat)
        if is_buggy:
            pds_labels.fetch_label(obs_id, manifest_row["LabelURL"], cache_dir)
            origin = pds_labels.projection_origin(obs_id, cache_dir)
            corrected_prj = _override_sp1(original_prj, origin["center_lat_deg"])
            gdf = gpd.read_file(shp)
            gdf = gdf.set_crs(corrected_prj, allow_override=True)
            correction = {
                "status": "sp1_corrected_from_pds_label",
                "original_sp1_deg": current_sp1,
                "corrected_sp1_deg": float(origin["center_lat_deg"]),
                "pds_center_lat_deg": float(origin["center_lat_deg"]),
                "pds_center_lon_deg": float(origin["center_lon_deg"]),
                "pds_a_axis_km": float(origin["a_axis_km"]),
            }
            return gdf, correction

    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        raise RuntimeError(
            f"{shp}: shapefile has no CRS (.prj missing or unreadable). "
            "Reprojection requires a known source CRS — refusing to guess."
        )
    return gdf, correction


def reproject_to_target(gdf: gpd.GeoDataFrame, target_crs: str | CRS) -> gpd.GeoDataFrame:
    """Reproject `gdf` to `target_crs`. Source CRS comes from the GeoDataFrame, not args."""
    target = CRS.from_user_input(target_crs)
    return gdf.to_crs(target)


def drop_null_geometries(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """Drop rows with null or empty geometry. Returns (cleaned_gdf, n_dropped).

    BoulderNet `*-mask-nms` shapefiles can carry many records that have a DBF row (score,
    id, ...) but no polygon geometry -- e.g. the dense vClaire exports, where up to ~67%
    of rows are null-geometry (verified 2026-05-28). They cannot be rasterized or
    centroid-counted, so Stage 4 would error/miscount. We drop them at ingest so the
    cached GPKG and its `n_polygons` reflect only real boulder outlines. No-op on the
    priority10 set (0 nulls)."""
    if len(gdf) == 0:
        return gdf, 0
    valid = ~(gdf.geometry.isna() | gdf.geometry.is_empty)
    n_dropped = int((~valid).sum())
    if n_dropped == 0:
        return gdf, 0
    return gdf.loc[valid].reset_index(drop=True), n_dropped


def cache_reprojected(
    gdf: gpd.GeoDataFrame,
    obs_id: str,
    cache_dir: str | Path,
    *,
    source_wkt: str,
    target_wkt: str,
    config_hash: str,
    source_path: str | Path,
    correction: dict | None = None,
    n_polygons_raw: int | None = None,
    n_dropped_null: int = 0,
) -> Path:
    """Write reprojected GeoDataFrame to `cache_dir/reprojected_detections/{obs_id}.gpkg`
    plus a sidecar `{obs_id}.json` provenance record. Returns the GPKG path.
    """
    out_dir = Path(cache_dir) / CACHE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / f"{obs_id}.gpkg"
    sidecar = out_dir / f"{obs_id}.json"

    gdf.to_file(gpkg, driver="GPKG", layer="detections")

    sidecar.write_text(
        json.dumps(
            {
                "obs_id": obs_id,
                "n_polygons": int(len(gdf)),
                "n_polygons_raw": int(n_polygons_raw) if n_polygons_raw is not None else int(len(gdf)),
                "n_dropped_null_geometry": int(n_dropped_null),
                "source_path": str(source_path),
                "source_mtime_iso": _dt.datetime.fromtimestamp(
                    Path(source_path).stat().st_mtime, tz=_dt.timezone.utc
                ).isoformat(),
                "source_crs_wkt": source_wkt,
                "target_crs_wkt": target_wkt,
                "config_hash": config_hash,
                "correction": correction or {"status": "trusted_prj"},
                "written_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return gpkg


def load_reprojected(obs_id: str, cache_dir: str | Path) -> gpd.GeoDataFrame:
    """Load a previously cached reprojected GeoDataFrame."""
    gpkg = Path(cache_dir) / CACHE_SUBDIR / f"{obs_id}.gpkg"
    return gpd.read_file(gpkg, layer="detections")


def stage1_one_image(
    obs_id: str,
    *,
    detections_root: str | Path,
    target_crs: str,
    cache_dir: str | Path,
    config_hash: str,
    manifest_row=None,
) -> tuple[gpd.GeoDataFrame, Path, dict]:
    """Run Stage 1 end-to-end for one ObsId: read, reproject (correcting buggy .prj
    if `manifest_row` is provided), cache. Returns the reprojected GeoDataFrame, the
    cache GPKG path, and the `correction` provenance dict.
    """
    shp = manifest_mod.find_shapefile(obs_id, detections_root)
    gdf, correction = read_detection_shapefile(
        obs_id, detections_root, manifest_row=manifest_row, cache_dir=cache_dir,
    )
    source_wkt = gdf.crs.to_wkt()
    n_raw = len(gdf)
    gdf, n_dropped = drop_null_geometries(gdf)
    gdf_t = reproject_to_target(gdf, target_crs)
    target_wkt = gdf_t.crs.to_wkt()
    gpkg = cache_reprojected(
        gdf_t,
        obs_id,
        cache_dir,
        source_wkt=source_wkt,
        target_wkt=target_wkt,
        config_hash=config_hash,
        source_path=shp,
        correction=correction,
        n_polygons_raw=n_raw,
        n_dropped_null=n_dropped,
    )
    return gdf_t, gpkg, correction
