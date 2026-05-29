"""One-off: inspect the new vClaire detection set vs the pipeline's expectations.

Checks the glob match, DBF schema, source CRS (+ SP1-bug fingerprint), polygon
count, and equivalent-circle diameter distribution (drives the min_size_m call).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import geopandas as gpd
import numpy as np

from src import manifest as M

NEW_ROOT = Path(r"C:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise_40_vClaire")
OLD_ROOT = Path(r"C:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise_priority10_detections")
MIN_SIZE_M = 1.4105  # current config floor (equiv-circle diameter)


def inspect(root: Path, obs_id: str, target_crs_for_area: str | None = None) -> None:
    print(f"\n=== {obs_id}  ({root.name}) ===")
    try:
        shp = M.find_shapefile(obs_id, root)
    except Exception as e:  # noqa: BLE001
        print(f"  find_shapefile FAILED: {e}")
        return
    print(f"  glob match: {shp.name}")
    gdf = gpd.read_file(shp)
    print(f"  n_polygons: {len(gdf)}")
    print(f"  columns:    {gdf.columns.tolist()}")
    if "score" in gdf.columns:
        s = gdf["score"]
        print(f"  score:      min={s.min():.3f} max={s.max():.3f} mean={s.mean():.3f}")
    # Source CRS + SP1 fingerprint
    prj_text = shp.with_suffix(".prj").read_text(encoding="latin-1")
    has_d_unnamed = 'D_unnamed' in prj_text or 'd_unnamed' in prj_text.lower()
    import re
    m = re.search(r'Standard_Parallel_1",([-\d.eE]+)', prj_text)
    sp1 = float(m.group(1)) if m else None
    print(f"  prj: D_unnamed={has_d_unnamed}  Standard_Parallel_1={sp1}")
    print(f"  crs name:   {gdf.crs.name if gdf.crs else None}")
    # Diameter distribution in the SOURCE CRS metres (good enough for a first look;
    # the pipeline applies min_size_m after reprojecting to the target sphere, sub-metre diff).
    if len(gdf) and gdf.crs is not None and gdf.crs.axis_info and gdf.crs.axis_info[0].unit_name in ("metre", "meter", "m"):
        diam = 2.0 * np.sqrt(gdf.geometry.area.to_numpy() / np.pi)
        pct = np.percentile(diam, [5, 25, 50, 75, 95, 99])
        print(f"  diam_m pctiles [5,25,50,75,95,99]: {np.round(pct, 2).tolist()}")
        below = int((diam < MIN_SIZE_M).sum())
        print(f"  diam < {MIN_SIZE_M} m (current floor): {below}/{len(gdf)} = {below/len(gdf):.1%} would be DROPPED")
    else:
        print("  (source CRS not in metres or empty; skipping diameter percentiles)")


def main() -> int:
    print("NEW vClaire set:")
    for d in sorted(NEW_ROOT.iterdir()):
        if d.is_dir():
            inspect(NEW_ROOT, d.name)
    # One old image for a side-by-side on schema + size floor.
    print("\n--- reference: one OLD priority10 image ---")
    inspect(OLD_ROOT, "ESP_069669_2220")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
