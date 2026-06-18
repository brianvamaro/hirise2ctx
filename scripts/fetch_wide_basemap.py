"""Fetch the wide circum-Chryse MOLA basemap used by notebook 24 §1a (coverage planning).

Downloads the 2 GB global MOLA DEM once (cached under cache_v2/validation/_src/) and writes a
coarse (2 km/px) reprojection over lon[-60,26] lat[16,56] in the CTX clon_0 CRS. Separate from
the configured `validation_rasters` region (which is the tight 26-tile map) because the planning
figure needs the whole cohort spread. Idempotent; re-run is a no-op once cached.

    python scripts/fetch_wide_basemap.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.modeling  # noqa: F401,E402  OpenMP guard
import rasterio  # noqa: E402

from src import validation_retrieve as vr  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MOLA_URL = "https://planetarymaps.usgs.gov/mosaic/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif"
OUT = REPO / "cache_v2" / "validation" / "mola_dem_wide.tif"


def main() -> None:
    ref = sorted((REPO / "reports" / "map_region").glob("*_abundance.tif"))
    if not ref:
        raise SystemExit("need a mapped tile in reports/map_region/ to read the CTX CRS")
    ctx_crs = rasterio.open(ref[0]).crs.to_wkt()
    prov = vr.fetch_region_raster(
        "mola_dem_wide", source_url=MOLA_URL, bounds_lonlat=(-60.0, 16.0, 26.0, 56.0),
        dst_crs_wkt=ctx_crs, out_path=OUT, cache_dir=REPO / "cache_v2",
        dst_res_m=2000.0, src_lon_domain="180", read_mode="download", buffer_deg=1.0,
    )
    print("wrote", OUT.relative_to(REPO), "shape", prov["dst_shape"],
          "valid_frac %.3f" % prov["valid_fraction"])


if __name__ == "__main__":
    main()
