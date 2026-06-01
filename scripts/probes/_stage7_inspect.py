"""Quick smoke test: verify src/colour.py LBL parser + check COLOR.JP2 raster metadata.

Reports per-image:
  - Parsed LBL: incidence, scaling, map_scale, lines x samples
  - JP2 CRS + bounds (in JP2's own CRS metres)
  - Pixel dtype + nodata + a sample of values
  - Coverage overlap with the original detection shapefile (how many polygons fall
    inside the colour swath)
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import colour  # noqa: E402

CACHE = Path("cache_v2")
DETECTIONS_ROOT = Path("C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise_40_vClaire")

TRIO = ["ESP_042964_2160", "ESP_054000_2255", "ESP_055253_2245"]


def _detection_shp(obs_id: str) -> Path:
    candidates = list((DETECTIONS_ROOT / obs_id).glob("*-mask-nms.shp"))
    if not candidates:
        raise FileNotFoundError(f"no detection shp for {obs_id}")
    return candidates[0]


def inspect(obs_id: str) -> None:
    print(f"\n=== {obs_id} ===")
    lbl = colour.parse_color_lbl(colour.color_lbl_path(CACHE, obs_id))
    print(f"  LBL: incidence={lbl.incidence_deg:.2f}deg "
          f"cos(i)={lbl.cos_incidence:.4f}  emission={lbl.emission_deg:.2f}")
    print(f"       scaling={lbl.scaling_factor:.4e}  offset={lbl.offset:.6f}  "
          f"map_scale={lbl.map_scale_mpp:.4f} m/px")
    print(f"       size={lbl.lines} x {lbl.line_samples} (bands={lbl.bands})")

    # SP1-corrected CRS from Stage 1 sidecar -- the JP2's embedded CRS is buggy.
    corrected_crs = colour.corrected_source_crs(obs_id, CACHE)
    print(f"  Stage1 corrected CRS present: {corrected_crs is not None}")

    jp2 = colour.color_jp2_path(CACHE, obs_id)
    with rasterio.open(jp2) as ds:
        print(f"  JP2: dtype={ds.dtypes[0]}  bands={ds.count}  shape=({ds.height}, {ds.width})")
        print(f"       res={ds.res}  bounds (raw): {tuple(round(b, 2) for b in ds.bounds)}")
        cy, cx = ds.height // 2, ds.width // 2
        w = rasterio.windows.Window(cx - 128, cy - 128, 256, 256)
        sample = ds.read(window=w)
        nz = sample[sample > 0]
        if nz.size:
            iof_min = lbl.scaling_factor * int(nz.min()) + lbl.offset
            iof_max = lbl.scaling_factor * int(nz.max()) + lbl.offset
            print(f"       DN range center 256x256: {int(nz.min())}..{int(nz.max())}  "
                  f"-> I/F {iof_min:.4f}..{iof_max:.4f}")
        jp2_bounds = box(*ds.bounds)

    shp_path = _detection_shp(obs_id)
    gdf = gpd.read_file(shp_path)
    # Override the buggy SP1=0 .prj with the Stage 1 corrected CRS (same correction as the JP2).
    gdf = gdf.set_crs(corrected_crs, allow_override=True)
    inside = gdf[gdf.intersects(jp2_bounds)]
    print(f"  polygons: {len(gdf)} total; {len(inside)} inside colour swath "
          f"({100*len(inside)/max(1,len(gdf)):.1f}%)")
    if len(inside):
        print(f"  polygon-area range (m^2): {inside.area.min():.3f} .. {inside.area.max():.3f} "
              f"(median {inside.area.median():.3f})")


def main() -> int:
    for obs in TRIO:
        try:
            inspect(obs)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
