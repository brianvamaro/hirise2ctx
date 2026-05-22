"""Debug the SP1 regex match on real pyproj output."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyproj

from src.hirise_imagery import _SP1_LITERAL_PATTERN, _sp1_literal, _crs_equal

wkt_template = (
    'PROJCS["Equirectangular_MARS",'
    'GEOGCS["GCS_MARS",DATUM["D_MARS",'
    'SPHEROID["MARS_localRadius",3393833.2607584,0.0]],'
    'PRIMEM["Reference_Meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Equidistant_Cylindrical"],'
    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",180.0],'
    'PARAMETER["Standard_Parallel_1",{sp1}],'
    'UNIT["Meter",1.0]]'
)
for sp1 in (0.0, 20.0):
    crs = pyproj.CRS.from_user_input(wkt_template.format(sp1=sp1))
    print(f"--- input SP1={sp1} ---")
    out = crs.to_wkt()
    print("wkt:", out[:200], "...")
    m = _SP1_LITERAL_PATTERN.search(out)
    print("regex match:", m, "value:", m.group(1) if m else None)
    print("literal:", _sp1_literal(crs))
    print()

a = pyproj.CRS.from_user_input(wkt_template.format(sp1=20.0))
b = pyproj.CRS.from_user_input(wkt_template.format(sp1=0.0))
print("_crs_equal(SP1=20, SP1=0):", _crs_equal(a, b))
