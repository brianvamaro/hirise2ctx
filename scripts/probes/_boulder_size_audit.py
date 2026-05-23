"""Audit per-image polygon-area distributions against the BoulderNet 5x5-px design floor.

Per Amaro et al. 2026 (the lunar rock-abundance paper), BoulderNet predictions are most
accurate for boulders covering > 5 x 5 = 25 source-image pixels in area, regardless of
pixel scale; detections below that threshold are filtered in post-processing.

This probe loads each priority10 Stage 1 cached GeoPackage (polygons reprojected into
the CTX target CRS, in metres) and reports the per-image polygon-area distribution.
It also reads the matching HiRISE PDS .LBL to recover the per-image pixel scale and
computes the 5x5-px threshold area for that image. The output tells us whether the
shapefiles we ingested have the post-processing filter applied or whether
sub-threshold detections survived.
"""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import manifest as M
from src.config import load_config
from src.pds_labels import read_label, _strip_units


def hirise_pixel_size_m(obs_id: str, cache_dir: Path) -> float | None:
    """Return the HiRISE pixel ground sample distance in metres from the cached .LBL.

    The PDS3 HiRISE RDR label exposes the projected pixel size as `MAP_SCALE`
    (in m/pixel) -- this is the value to use for the 5x5-px design floor since
    BoulderNet operates on the projected RDR.
    """
    try:
        kw = read_label(obs_id, cache_dir)
    except FileNotFoundError:
        return None
    val = kw.get("MAP_SCALE")
    if val is None:
        return None
    try:
        return _strip_units(val)
    except ValueError:
        return None


def main() -> int:
    cfg = load_config("config.yaml")
    manifest = M.load_manifest(cfg.manifest_path)
    rows = []
    for obs in manifest["ObsId"]:
        gpkg = cfg.cache_dir / "reprojected_detections" / f"{obs}.gpkg"
        if not gpkg.exists():
            continue
        gdf = gpd.read_file(gpkg, layer="detections")
        if len(gdf) == 0:
            rows.append({"ObsId": obs, "n": 0})
            continue
        areas = gdf.geometry.area.to_numpy()  # m^2 in CTX CRS (close to true area in this lat band)
        diameters = 2.0 * np.sqrt(areas / np.pi)
        px_m = hirise_pixel_size_m(obs, cfg.cache_dir)
        if px_m is not None:
            threshold_m2 = (5 * px_m) ** 2
            n_below = int((areas < threshold_m2).sum())
        else:
            threshold_m2 = float("nan")
            n_below = -1
        rows.append({
            "ObsId": obs,
            "n_polys": int(len(gdf)),
            "px_m": px_m,
            "5x5_threshold_m2": round(threshold_m2, 3) if not np.isnan(threshold_m2) else None,
            "area_min": round(float(areas.min()), 3),
            "area_p05": round(float(np.percentile(areas, 5)), 3),
            "area_p50": round(float(np.percentile(areas, 50)), 3),
            "area_p95": round(float(np.percentile(areas, 95)), 3),
            "area_max": round(float(areas.max()), 3),
            "diam_min_m": round(float(diameters.min()), 3),
            "diam_p50_m": round(float(np.percentile(diameters, 50)), 3),
            "n_below_threshold": n_below,
            "pct_below_threshold": round(n_below / len(gdf) * 100, 2) if n_below >= 0 else None,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print()
    if (df["pct_below_threshold"].fillna(-1) > 0).any():
        print("==> Sub-threshold polygons are present in the cached shapefiles.")
        print("    Per Amaro et al. 2026, these should have been filtered in BoulderNet's")
        print("    post-processing. Either (a) the post-filter was not applied to our copy,")
        print("    or (b) the priority10 shapefiles predate that post-processing step.")
    else:
        print("==> No sub-threshold polygons; post-filter was applied to our copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
