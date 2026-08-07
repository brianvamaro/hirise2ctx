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


# ----------------------------------------------------------------------------
# R23 provenance: byte-truncated .shp detection + rank-truncation characterisation
# (DECISIONS 2026-08-06b). These are the two checks whose absence let a score-rank
# truncation be recorded as "benign density hygiene" for two months.
# ----------------------------------------------------------------------------
import struct
import warnings

from shapely.geometry import box

from src.detections import (
    describe_null_geometry_drop,
    drop_null_geometries,
    inspect_shapefile_integrity,
)


def _write_shp(path, *, declared_bytes: int, actual_bytes: int, n_records: int = 0):
    """Minimal .shp with a valid header whose declared length may exceed the file size."""
    header = bytearray(100)
    header[0:4] = (9994).to_bytes(4, "big")
    header[24:28] = (declared_bytes // 2).to_bytes(4, "big")
    struct.pack_into("<i", header, 28, 1000)  # version
    struct.pack_into("<i", header, 32, 5)     # shape type: polygon
    path.write_bytes(bytes(header) + b"\0" * max(0, actual_bytes - 100))
    if n_records:
        # .shx: 100-byte header then (offset, length) pairs in 16-bit words.
        rec = 48  # bytes per record, arbitrary but constant
        shx = bytearray(b"\0" * 100)
        for i in range(n_records):
            off = 100 + i * rec
            shx += struct.pack(">ii", off // 2, (rec - 8) // 2)
        path.with_suffix(".shx").write_bytes(bytes(shx))


def test_integrity_flags_a_byte_truncated_shapefile(tmp_path):
    shp = tmp_path / "trunc.shp"
    _write_shp(shp, declared_bytes=1000, actual_bytes=400)
    out = inspect_shapefile_integrity(shp)
    assert out["status"] == "truncated"
    assert out["declared_bytes"] == 1000
    assert out["actual_bytes"] == 400
    assert out["missing_bytes"] == 600


def test_integrity_passes_a_complete_shapefile(tmp_path):
    shp = tmp_path / "ok.shp"
    _write_shp(shp, declared_bytes=400, actual_bytes=400)
    out = inspect_shapefile_integrity(shp)
    assert out["status"] == "complete"
    assert out["missing_bytes"] == 0


def test_integrity_counts_surviving_records_via_the_shx(tmp_path):
    # 10 records of 48 bytes from offset 100 -> record i ends at 148 + 48*i.
    # Truncating at 340 bytes keeps every record ending <= 340, i.e. records 0..4
    # (record 4 ends at exactly 340); record 5 would end at 388.
    shp = tmp_path / "part.shp"
    _write_shp(shp, declared_bytes=580, actual_bytes=340, n_records=10)
    out = inspect_shapefile_integrity(shp)
    assert out["status"] == "truncated"
    assert out["n_records_index"] == 10
    assert out["n_records_present"] == 5
    # One byte short of record 4's end drops it.
    shp2 = tmp_path / "part2.shp"
    _write_shp(shp2, declared_bytes=580, actual_bytes=339, n_records=10)
    assert inspect_shapefile_integrity(shp2)["n_records_present"] == 4


def test_integrity_never_raises_on_a_missing_or_bogus_file(tmp_path):
    assert inspect_shapefile_integrity(tmp_path / "nope.shp")["status"] == "unreadable"
    bogus = tmp_path / "bogus.shp"
    bogus.write_bytes(b"not a shapefile" * 10)
    assert inspect_shapefile_integrity(bogus)["status"] == "unreadable"


def _gdf(scores, geoms):
    return gpd.GeoDataFrame({"score": scores}, geometry=geoms, crs=CRS.from_epsg(4326))


def test_describe_flags_a_score_rank_truncation():
    """The R23 signature: every dropped row scores at or below every kept row.

    Populations are >= `_MIN_DROPPED_FOR_RANK_VERDICT` so the verdict is actually issued;
    the real cases drop 291k-875k rows.
    """
    n_keep, n_drop = 150, 120
    scores = [0.9 - 0.001 * i for i in range(n_keep)] + [0.4 - 0.001 * i for i in range(n_drop)]
    geoms = [box(i, 0, i + 1, 1) for i in range(n_keep)] + [None] * n_drop
    out = describe_null_geometry_drop(_gdf(scores, geoms))
    assert out["is_rank_truncation"] is True
    assert out["n_dropped"] == n_drop
    assert out["dropped_fraction"] == pytest.approx(n_drop / (n_keep + n_drop))
    assert out["realised_score_floor"] == pytest.approx(scores[n_keep - 1])
    assert out["dropped_score_max"] == pytest.approx(0.4)
    assert out["kept_minus_dropped_gap"] > 0


def test_describe_does_not_flag_genuinely_sparse_nulls():
    """Nulls interleaved through the score range are export noise, not a truncation."""
    scores, geoms = [], []
    for i in range(300):                       # 150 nulls, spread across the whole range
        scores.append(0.9 - 0.002 * i)
        geoms.append(None if i % 2 else box(i, 0, i + 1, 1))
    out = describe_null_geometry_drop(_gdf(scores, geoms))
    assert out["is_rank_truncation"] is False
    assert out["n_dropped"] == 150
    assert out["kept_minus_dropped_gap"] < 0   # the populations interleave


def test_describe_is_none_when_nothing_is_dropped():
    g = _gdf([0.5, 0.6], [box(0, 0, 1, 1), box(2, 0, 3, 1)])
    assert describe_null_geometry_drop(g) is None


def test_describe_survives_a_missing_score_column():
    g = gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1), None], crs=CRS.from_epsg(4326))
    out = describe_null_geometry_drop(g)
    assert out["n_dropped"] == 1
    assert "is_rank_truncation" not in out
    assert "note" in out


def test_describe_agrees_with_what_drop_null_geometries_removes():
    """The characterisation must describe exactly the rows the dropper drops."""
    geoms = [box(i, 0, i + 1, 1) for i in range(4)] + [None, None]
    g = _gdf([0.9, 0.8, 0.7, 0.65, 0.2, 0.1], geoms)
    out = describe_null_geometry_drop(g)
    kept, n_dropped = drop_null_geometries(g)
    assert out["n_dropped"] == n_dropped
    assert out["n_kept"] == len(kept)
    assert out["realised_score_floor"] == pytest.approx(kept["score"].min())


def test_rank_verdict_withheld_on_a_tiny_dropped_population():
    """All-equal scores satisfy max(dropped) <= min(kept) by construction."""
    geoms = [box(0, 0, 1, 1), box(2, 0, 3, 1), None]
    out = describe_null_geometry_drop(_gdf([0.5, 0.5, 0.5], geoms))
    assert out["is_rank_truncation"] is None      # withheld, not True
    assert out["kept_minus_dropped_gap"] == pytest.approx(0.0)
    assert "too few" in out["note"]


def test_describe_never_raises_on_a_non_numeric_score_column():
    """Stage 1 must still ingest a manifest row whose export stores `score` as text."""
    g = gpd.GeoDataFrame(
        {"score": ["high", "low"]},
        geometry=[box(0, 0, 1, 1), None],
        crs=CRS.from_epsg(4326),
    )
    out = describe_null_geometry_drop(g)          # must not raise
    assert out["n_dropped"] == 1
    assert "not numeric" in out["note"]
    assert "is_rank_truncation" not in out


def test_integrity_flags_a_stale_or_oversized_header(tmp_path):
    """`actual > declared` and a sub-header declared length must not read as complete."""
    over = tmp_path / "over.shp"
    _write_shp(over, declared_bytes=200, actual_bytes=400)
    assert inspect_shapefile_integrity(over)["status"] == "length_mismatch"
    zero = tmp_path / "zero.shp"
    _write_shp(zero, declared_bytes=0, actual_bytes=400)
    assert inspect_shapefile_integrity(zero)["status"] == "suspect_header"


def test_integrity_marks_the_record_count_a_lower_bound_when_the_shx_is_short(tmp_path):
    shp = tmp_path / "bothshort.shp"
    _write_shp(shp, declared_bytes=580, actual_bytes=340, n_records=10)
    shx = shp.with_suffix(".shx")
    shx.write_bytes(shx.read_bytes()[: 100 + 8 * 4])   # keep only 4 index entries
    out = inspect_shapefile_integrity(shp)
    assert out["shx_status"] == "truncated"
    assert out["n_records_present_is_lower_bound"] is True


# --- wiring: the integration, not just the two pure functions --------------------------

def _make_real_shapefile(path, scores, *, keep_fraction: float = 1.0):
    """Write a genuine shapefile (score-descending) and optionally truncate its .shp.

    Truncating the payload while leaving the 100-byte header intact reproduces the real
    R23 failure mode exactly: an interrupted copy, whose header still declares the full
    length and whose .shx/.dbf are complete.
    """
    g = gpd.GeoDataFrame(
        {"score": scores},
        geometry=[box(i, 0, i + 0.5, 0.5) for i in range(len(scores))],
        crs=CRS.from_user_input(HIRISE_LOCAL_WKT),
    )
    g.to_file(path, driver="ESRI Shapefile")
    if keep_fraction < 1.0:
        raw = path.read_bytes()
        path.write_bytes(raw[: 100 + int((len(raw) - 100) * keep_fraction)])
    return path


def test_stage1_warns_and_persists_provenance_on_a_truncated_source(tmp_path):
    """The whole point of the R23 fix: Stage 1 must SAY SO and RECORD it."""
    from src import detections as det

    det_root = tmp_path / "dets" / "OBS_TEST"
    det_root.mkdir(parents=True)
    n = 400
    shp = _make_real_shapefile(
        det_root / "OBS_TEST_RED-mask-nms.shp",
        [0.9 - 0.001 * i for i in range(n)],
        keep_fraction=0.5,
    )
    integ = det.inspect_shapefile_integrity(shp)
    assert integ["status"] == "truncated"
    # enough records lost to clear the rank-verdict floor
    assert n - integ["n_records_present"] >= 100

    cache = tmp_path / "cache"
    with pytest.warns(RuntimeWarning, match="BYTE-TRUNCATED"):
        det.stage1_one_image(
            "OBS_TEST",
            detections_root=tmp_path / "dets",
            target_crs=CTX_MARS_WKT,
            cache_dir=cache,
            config_hash="test",
        )

    import json as _json
    sidecar = _json.loads(
        (cache / det.CACHE_SUBDIR / "OBS_TEST.json").read_text(encoding="utf-8")
    )
    assert sidecar["source_integrity"]["status"] == "truncated"
    assert sidecar["source_integrity"]["missing_bytes"] > 0
    assert sidecar["null_geometry_basis"]["is_rank_truncation"] is True
    # score-descending source -> the survivors are the top scores, so the realised floor
    # sits well above the lowest score actually present in the .dbf
    assert sidecar["null_geometry_basis"]["realised_score_floor"] > 0.5


def test_stage1_is_silent_and_records_complete_on_an_intact_source(tmp_path):
    from src import detections as det

    det_root = tmp_path / "dets" / "OBS_OK"
    det_root.mkdir(parents=True)
    _make_real_shapefile(det_root / "OBS_OK_RED-mask-nms.shp", [0.9 - 0.001 * i for i in range(400)])

    cache = tmp_path / "cache"
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # any warning fails this test
        det.stage1_one_image(
            "OBS_OK",
            detections_root=tmp_path / "dets",
            target_crs=CTX_MARS_WKT,
            cache_dir=cache,
            config_hash="test",
        )
    import json as _json
    sidecar = _json.loads((cache / det.CACHE_SUBDIR / "OBS_OK.json").read_text(encoding="utf-8"))
    assert sidecar["source_integrity"]["status"] == "complete"
    assert sidecar["null_geometry_basis"] is None
    assert sidecar["n_dropped_null_geometry"] == 0
