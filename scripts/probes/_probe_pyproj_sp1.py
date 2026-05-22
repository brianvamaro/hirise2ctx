"""Inspect what pyproj does to an Equirectangular WKT with SP1=20 vs SP1=0."""
from __future__ import annotations

import pyproj

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
    print(crs.to_wkt())
    print()
