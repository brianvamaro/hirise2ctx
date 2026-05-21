"""Stage 1 reprojection unit tests (synthetic — no shapefile I/O, no network)."""
from __future__ import annotations

import geopandas as gpd
import pytest
from pyproj import CRS, Transformer
from shapely.geometry import Point

from src.detections import reproject_to_target


# Source CRS = the exact local-radius equirectangular from ESP_047976_2020's .prj
HIRISE_LOCAL_WKT = (
    'PROJCS["Equirectangular_MARS",GEOGCS["GCS_MARS",DATUM["D_unnamed",'
    'SPHEROID["unnamed",3393833.2607584,0.0]],PRIMEM["Reference_meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]],PROJECTION["Equidistant_Cylindrical"],'
    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",180.0],PARAMETER["Standard_Parallel_1",0.0],'
    'UNIT["Meter",1.0]]'
)

# Target CRS = standard IAU2000 Mars equirectangular (sphere 3,396,190 m, cm 0). Stand-in
# for the CTX mosaic CRS for synthetic tests; the real one is read at runtime in Stage 0.5.
CTX_MARS_WKT = (
    'PROJCS["Mars_2000_Equidistant_Cylindrical",GEOGCS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",SPHEROID["Mars_2000_IAU_IAG",3396190.0,0.0]],'
    'PRIMEM["Reference_Meridian",0.0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Equidistant_Cylindrical"],PARAMETER["False_Easting",0.0],'
    'PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",0.0],'
    'PARAMETER["Standard_Parallel_1",0.0],UNIT["Meter",1.0]]'
)


def _lonlat_to_hirise_xy(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """Forward-project a (lon, lat) on the HiRISE source sphere into its projected coords.

    Used to build a synthetic point at a known geographic position WITHOUT round-tripping
    through the same machinery we're testing.
    """
    geo = CRS.from_user_input(HIRISE_LOCAL_WKT).geodetic_crs
    fwd = Transformer.from_crs(geo, HIRISE_LOCAL_WKT, always_xy=True)
    return fwd.transform(lon_deg, lat_deg)


def test_reproject_preserves_geographic_position_on_known_point():
    """A point at (lon=20E, lat=46N) on the HiRISE local sphere should reproject to the
    SAME geographic coordinates on the CTX sphere — i.e. the lat/lon, not the metres,
    are what we trust across CRSes. Round-trip via inverse projection on each side.
    """
    lon_deg, lat_deg = 20.0, 46.0
    x, y = _lonlat_to_hirise_xy(lon_deg, lat_deg)

    src = gpd.GeoDataFrame(geometry=[Point(x, y)], crs=HIRISE_LOCAL_WKT)
    dst = reproject_to_target(src, CTX_MARS_WKT)

    # Inverse-project the destination point back to its geographic coords on the CTX sphere.
    inv = Transformer.from_crs(dst.crs, CRS.from_user_input(dst.crs).geodetic_crs, always_xy=True)
    dst_lon, dst_lat = inv.transform(dst.geometry.iloc[0].x, dst.geometry.iloc[0].y)

    # Same physical point on slightly different sphere radii -> sub-degree-second agreement
    # in lat/lon, despite the metres being numerically different.
    assert dst_lon == pytest.approx(lon_deg, abs=1e-6)
    assert dst_lat == pytest.approx(lat_deg, abs=1e-6)


def test_reproject_changes_metric_coordinates_as_expected():
    """The HiRISE source has central meridian 180 and a smaller sphere than the CTX target
    (central meridian 0, larger sphere). A point at lon=20E should end up at different
    projected x coordinates in the two CRSes — confirming we're actually reprojecting and
    not silently identity-mapping.
    """
    lon_deg, lat_deg = 20.0, 46.0
    x, y = _lonlat_to_hirise_xy(lon_deg, lat_deg)

    src = gpd.GeoDataFrame(geometry=[Point(x, y)], crs=HIRISE_LOCAL_WKT)
    dst = reproject_to_target(src, CTX_MARS_WKT)
    dx, dy = dst.geometry.iloc[0].x, dst.geometry.iloc[0].y
    # Numerically must differ (different cm, different radius)
    assert abs(dx - x) > 1.0
    assert abs(dy - y) > 0.0  # different radii alter the y as well


def test_reproject_does_not_clobber_source_crs():
    """The reprojection helper must NOT mutate the input GeoDataFrame's CRS."""
    src = gpd.GeoDataFrame(geometry=[Point(0, 0)], crs=HIRISE_LOCAL_WKT)
    src_crs_before = CRS.from_user_input(src.crs)
    _ = reproject_to_target(src, CTX_MARS_WKT)
    src_crs_after = CRS.from_user_input(src.crs)
    assert src_crs_before == src_crs_after
