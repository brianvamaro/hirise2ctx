"""Shared sanity checks for the pipeline. Used by tests and QA notebooks alike.

The Stage 0–1 sanity check (`assert_centroid_consistent`) catches the most common CRS
bug — accidentally swapping the per-image local Mars radius for the IAU2000 radius — by
comparing the reprojected polygon-footprint centroid (back-projected to lat/lon on the
target sphere) against the manifest's published center. A multi-km mismatch is diagnostic
of the wrong-sphere class of error.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from pyproj import CRS, Geod, Transformer


@dataclass
class CentroidCheck:
    """Result of `assert_centroid_consistent` — exposed for logging/QA."""
    obs_id: str
    manifest_lat_deg: float
    manifest_lon_deg: float
    centroid_lat_deg: float
    centroid_lon_deg: float
    distance_m: float
    threshold_m: float
    target_sphere_m: float

    def as_dict(self) -> dict:
        return self.__dict__


def _sphere_radius_m(crs: CRS) -> float:
    """Return the semi-major axis of `crs`'s ellipsoid in metres.

    For Mars equirectangular CRSes (CTX mosaic, HiRISE per-image), the ellipsoid is a
    sphere and `semi_major_metre == semi_minor_metre`.
    """
    ell = crs.ellipsoid
    if ell.semi_major_metre is None:
        raise ValueError(f"CRS has no ellipsoid semi-major axis: {crs.name}")
    return float(ell.semi_major_metre)


def assert_centroid_consistent(
    gdf_target: gpd.GeoDataFrame,
    *,
    obs_id: str,
    manifest_lat_deg: float,
    manifest_lon_deg: float,
    max_km: float,
    source_crs_wkt: str | None = None,
) -> CentroidCheck | None:
    """Verify that the reprojected polygon centroid is within `max_km` of the manifest center.

    Fails with a `RuntimeError` (loudly, with diagnostic context) if the distance exceeds
    the threshold — the typical signature of having mishandled the per-image local Mars
    radius during reprojection.

    Returns `None` (no centroid computable) when `gdf_target` is empty — e.g. BoulderNet
    found zero boulders in this image. Empty inputs are a valid case for rock-abundance
    regression (all tiles get `fractional_area=0` in Stage 4), so the absence of a
    centroid is not a failure condition.
    """
    if gdf_target.crs is None:
        raise ValueError(f"{obs_id}: gdf_target has no CRS; cannot inverse-project centroid")
    if len(gdf_target) == 0:
        return None
    target_crs = CRS.from_user_input(gdf_target.crs)

    # Footprint centroid in target CRS (metres). `union_all()` is the modern replacement
    # for the deprecated `unary_union` accessor.
    union_geom = gdf_target.geometry.union_all()
    cx_m, cy_m = float(union_geom.centroid.x), float(union_geom.centroid.y)

    # Inverse-project to lat/lon on the target CRS's geographic base (same sphere).
    geographic = target_crs.geodetic_crs
    to_lonlat = Transformer.from_crs(target_crs, geographic, always_xy=True)
    centroid_lon, centroid_lat = to_lonlat.transform(cx_m, cy_m)

    # Great-circle distance on the target sphere.
    R = _sphere_radius_m(target_crs)
    geod = Geod(a=R, b=R)
    # Normalize the manifest longitude into the range the geographic CRS uses. pyproj
    # Geod accepts any longitude convention as long as both ends share it, but staying
    # in [-180, 180] avoids surprises.
    m_lon = ((manifest_lon_deg + 180.0) % 360.0) - 180.0
    c_lon = ((centroid_lon + 180.0) % 360.0) - 180.0
    _, _, dist_m = geod.inv(m_lon, manifest_lat_deg, c_lon, centroid_lat)

    threshold_m = max_km * 1000.0
    result = CentroidCheck(
        obs_id=obs_id,
        manifest_lat_deg=float(manifest_lat_deg),
        manifest_lon_deg=float(m_lon),
        centroid_lat_deg=float(centroid_lat),
        centroid_lon_deg=float(c_lon),
        distance_m=float(dist_m),
        threshold_m=float(threshold_m),
        target_sphere_m=R,
    )

    if dist_m > threshold_m:
        source_note = f"\n  source CRS WKT (first 200 chars): {source_crs_wkt[:200]!r}" if source_crs_wkt else ""
        raise RuntimeError(
            f"CRS sanity check failed for {obs_id}: reprojected centroid is "
            f"{dist_m/1000:.3f} km from the manifest center (> {max_km} km threshold).\n"
            f"  manifest center: lat={manifest_lat_deg:.6f}, lon={m_lon:.6f}\n"
            f"  reprojected centroid: lat={centroid_lat:.6f}, lon={c_lon:.6f}\n"
            f"  target sphere radius: {R:.3f} m  ({target_crs.name})"
            f"{source_note}\n"
            "  This usually means the per-image .prj local Mars radius was bypassed "
            "during reprojection. Verify that geopandas read the source .prj (the "
            "GeoDataFrame's source CRS must NOT have been overridden)."
        )
    return result
