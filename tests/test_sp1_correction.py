"""Unit tests for the buggy-.prj SP1 detector and override in `src/detections.py`."""
from __future__ import annotations

from src.detections import _suspect_sp1, _override_sp1

# Authentic ESP_047976_2020 .prj text (the bad one — SP1=0, datum D_unnamed)
BAD_PRJ = (
    'PROJCS["Equirectangular_MARS",GEOGCS["GCS_MARS",DATUM["D_unnamed",'
    'SPHEROID["unnamed",3393833.2607584,0.0]],PRIMEM["Reference_meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]],PROJECTION["Equidistant_Cylindrical"],'
    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",180.0],PARAMETER["Standard_Parallel_1",0.0],'
    'UNIT["Meter",1.0]]'
)

# Authentic ESP_069669_2220 .prj text (the good one — SP1=40, datum D_MARS)
GOOD_PRJ = (
    'PROJCS["Equirectangular_MARS",GEOGCS["GCS_MARS",DATUM["D_MARS",'
    'SPHEROID["MARS_localRadius",3387887.658234,0.0]],PRIMEM["Reference_Meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]],PROJECTION["Equidistant_Cylindrical"],'
    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",0.0],PARAMETER["Standard_Parallel_1",40.0],'
    'UNIT["Meter",1.0]]'
)


def test_suspect_sp1_flags_d_unnamed_with_zero_sp1_far_from_image_lat():
    """The 4 bad BoulderNet exports: D_unnamed + SP1=0 + image not at equator."""
    is_buggy, current_sp1 = _suspect_sp1(BAD_PRJ, image_lat_deg=21.64)
    assert is_buggy
    assert current_sp1 == 0.0


def test_suspect_sp1_trusts_d_mars_prj():
    """A correctly-labelled .prj must NOT be flagged, even if SP1 differs slightly from image_lat."""
    is_buggy, current_sp1 = _suspect_sp1(GOOD_PRJ, image_lat_deg=41.69)
    assert not is_buggy
    assert current_sp1 == 40.0


def test_suspect_sp1_does_not_flag_image_at_equator():
    """SP1=0 is plausible if the image is at the equator (where any SP1 is acceptable);
    only flag when SP1 disagrees with image_lat by more than the tolerance."""
    is_buggy, _ = _suspect_sp1(BAD_PRJ, image_lat_deg=0.5)
    assert not is_buggy  # within 15° tolerance even though datum is buggy


def test_override_sp1_replaces_only_target_parameter():
    """Override should only touch Standard_Parallel_1, leaving everything else byte-identical."""
    fixed = _override_sp1(BAD_PRJ, new_sp1_deg=20.0)
    assert 'PARAMETER["Standard_Parallel_1",20.0]' in fixed
    assert 'PARAMETER["Standard_Parallel_1",0.0]' not in fixed
    # Other params unchanged
    assert 'PARAMETER["Central_Meridian",180.0]' in fixed
    assert "3393833.2607584" in fixed
    assert 'DATUM["D_unnamed"' in fixed  # we don't normalize the datum; LBL drives correctness


def test_override_sp1_is_idempotent():
    """Applying the same override twice yields the same result."""
    first = _override_sp1(BAD_PRJ, new_sp1_deg=20.0)
    second = _override_sp1(first, new_sp1_deg=20.0)
    assert first == second
