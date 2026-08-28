"""Export a single GeoPackage of *where the project's data lives*, for QGIS/ArcGIS.

Not a pipeline producer: it reads shipped artifacts and writes one self-contained
`.gpkg` outside the artifact roots. Layers:

* `hirise_footprints` -- the 39-image vClaire cohort as PDS-declared extent rectangles
* `hirise_centers`    -- the same 39 as points (labelling / point-symbol friendly)
* `map_tiles`         -- Murray CTX tiles, tagged by which product covers them
                         (shipped `map_region`/`map_a1`, planned `map_extended`)

Everything is written in the CTX mosaic CRS (Mars_2015 equirectangular clon_0,
sphere 3396190 m) *and* carries plain lon/lat columns, so the layers drop straight
onto the shipped mosaics without reprojection.

`--format` picks the container. ArcGIS Pro **silently refuses** the GeoPackage: the
mosaic's own WKT tags its spheroid `AUTHORITY["IAU","49901"]` (which is *Ographic*)
while naming itself *Ocentric*, and GDAL records that CRS under `organization=NONE`.
GDAL tolerates the mismatch; Pro validates authority codes when it enumerates GPKG
layers and drops them without an error dialog. `--format filegdb` writes Pro's own
native container instead and is the recommended route for Pro.

Usage:
    conda run -n geospatial python scripts/export_context_gpkg.py            # gpkg
    conda run -n geospatial python scripts/export_context_gpkg.py -f filegdb # ArcGIS Pro
    conda run -n geospatial python scripts/export_context_gpkg.py -f shapefile
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

REPO_ROOT = Path(__file__).resolve().parents[1]

# The CTX mosaic CRS, as carried by every cached tile sidecar's `inner_crs_wkt`
# (Mars_2015_Ocentric_Equirectangular_clon_0).
CTX_WKT = (
    'PROJCS["Mars_2015_Ocentric_Equirectangular_clon_0",'
    'GEOGCS["GCS_Mars_2015_Ocentric",DATUM["Mars (2015)",'
    'SPHEROID["Mars_2015",3396190,169.894447223612,AUTHORITY["IAU","49901"]]],'
    'PRIMEM["Reference Meridian",0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],PARAMETER["standard_parallel_1",0],'
    'PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],'
    'AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
)
CTX_RADIUS_M = 3396190.0

# A Murray tile is named by its LOWER-LEFT corner and spans 4 deg x 4 deg
# (DECISIONS 2026-08-28b; verified here against each cached tile's own transform).
MURRAY_TILE_DEG = 4.0


def _lon_to_x(lon_deg: float) -> float:
    return math.radians(lon_deg) * CTX_RADIUS_M


def _lat_to_y(lat_deg: float) -> float:
    return math.radians(lat_deg) * CTX_RADIUS_M


def _wrap180(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _read_lbl_footprint(lbl_path: Path) -> dict[str, float]:
    """Pull the PDS-declared extent from a HiRISE RDR .LBL.

    Deliberately standalone (rather than `src.pds_labels.image_footprint`) so the
    export runs off a plain cache directory with no Config resolution.
    """
    keys: dict[str, float | None] = {
        "MAXIMUM_LATITUDE": None,
        "MINIMUM_LATITUDE": None,
        "EASTERNMOST_LONGITUDE": None,
        "WESTERNMOST_LONGITUDE": None,
    }
    for raw in lbl_path.read_text(errors="replace").splitlines():
        if "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        key = key.strip()
        if key in keys and keys[key] is None:
            keys[key] = float(value.split("<")[0].strip())
    missing = [k for k, v in keys.items() if v is None]
    if missing:
        raise ValueError(f"{lbl_path.name}: missing {missing}")
    return {
        "max_lat_deg": keys["MAXIMUM_LATITUDE"],
        "min_lat_deg": keys["MINIMUM_LATITUDE"],
        "east_lon_deg": keys["EASTERNMOST_LONGITUDE"],
        "west_lon_deg": keys["WESTERNMOST_LONGITUDE"],
    }


def _tile_bounds_deg(tile: str) -> tuple[float, float, float, float]:
    """`E-24_N28` -> (lon_min, lat_min, lon_max, lat_max) in degrees."""
    lon_s, _, lat_s = tile.partition("_")
    lon = float(lon_s.lstrip("E"))
    lat = float(lat_s.lstrip("N"))
    return lon, lat, lon + MURRAY_TILE_DEG, lat + MURRAY_TILE_DEG


def _verify_tile_convention(cache_tiles: Path) -> int:
    """Cross-check `_tile_bounds_deg` against every cached tile's own transform.

    VERIFY-AT-RUNTIME: a silently wrong corner convention yields a valid-looking
    layer covering the wrong ground, so this fails loudly instead.
    """
    checked = 0
    for sidecar in sorted(cache_tiles.glob("*.json")):
        if sidecar.name.startswith("_"):
            continue
        meta = json.loads(sidecar.read_text())
        transform, shape = meta.get("inner_transform"), meta.get("inner_shape")
        if not transform or not shape:
            continue
        a, _, c, _, e, f = transform[:6]
        height, width = shape
        got = (
            math.degrees(c / CTX_RADIUS_M),
            math.degrees((f + e * height) / CTX_RADIUS_M),
            math.degrees((c + a * width) / CTX_RADIUS_M),
            math.degrees(f / CTX_RADIUS_M),
        )
        want = _tile_bounds_deg(meta["murray_tile"])
        if max(abs(g - x) for g, x in zip(got, want)) > 1e-3:
            raise SystemExit(
                f"tile-name convention disagrees with {sidecar.name}: "
                f"name implies {want}, raster says {tuple(round(v, 4) for v in got)}"
            )
        checked += 1
    return checked


def build_hirise_layers(
    manifest_csv: Path, cache_root: Path
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    manifest = pd.read_csv(manifest_csv)
    rows = []
    for _, row in manifest.iterrows():
        obs = row["ObsId"]
        fp = _read_lbl_footprint(cache_root / "pds_labels" / f"{obs}.LBL")
        west, east = _wrap180(fp["west_lon_deg"]), _wrap180(fp["east_lon_deg"])
        if west > east:  # antimeridian-spanning: none in this cohort, but be loud
            raise SystemExit(f"{obs}: footprint spans the antimeridian ({west}, {east})")

        # n_detections: what the pipeline actually kept, from the reprojection
        # sidecar -- NOT the manifest's raw NPolygons, since some source shapefiles
        # are truncated upstream and the two disagree by a lot.
        det = cache_root / "reprojected_detections" / f"{obs}.json"
        n_kept = n_raw = None
        if det.exists():
            meta = json.loads(det.read_text())
            n_kept, n_raw = meta.get("n_polygons"), meta.get("n_polygons_raw")

        rows.append(
            {
                "obs_id": obs,
                "boulder_label": row.get("BoulderLabel"),
                "terrain_note": row.get("TerrainNote"),
                "quality_note": row.get("QualityNote"),
                "center_lat": row.get("CenterLat"),
                "center_lon_180": row.get("CenterLon_180"),
                "ctx_tile": row.get("CTX_TileName"),
                "map_pixel_mpp": row.get("MapPixel_mpp"),
                "n_detections_kept": n_kept,
                "n_detections_raw": n_raw,
                "truncated_source": (
                    None if n_kept is None else bool(n_raw and n_kept != n_raw)
                ),
                "hirise_url": row.get("BrowseURL"),
                "jp2_url": row.get("JP2_URL"),
                "west_lon_180": west,
                "east_lon_180": east,
                "min_lat": fp["min_lat_deg"],
                "max_lat": fp["max_lat_deg"],
            }
        )

    df = pd.DataFrame(rows)
    footprints = gpd.GeoDataFrame(
        df,
        geometry=[
            box(
                _lon_to_x(r.west_lon_180),
                _lat_to_y(r.min_lat),
                _lon_to_x(r.east_lon_180),
                _lat_to_y(r.max_lat),
            )
            for r in df.itertuples()
        ],
        crs=CTX_WKT,
    )
    centers = gpd.GeoDataFrame(
        df.copy(),
        geometry=[
            Point(_lon_to_x(_wrap180(r.center_lon_180)), _lat_to_y(r.center_lat))
            for r in df.itertuples()
        ],
        crs=CTX_WKT,
    )
    return footprints, centers


def build_map_tiles_layer(reports_root: Path) -> gpd.GeoDataFrame:
    """Murray tiles covered by each map product, one row per tile."""
    sources: dict[str, list[str]] = {}

    for name, manifest_path in [
        ("map_region (shipped baseline)", reports_root / "map_region" / "region_manifest.json"),
        ("map_a1 (sensitivity arm)", reports_root / "map_a1" / "a1_manifest.json"),
    ]:
        if manifest_path.exists():
            for tile in json.loads(manifest_path.read_text()).get("tiles", []):
                sources.setdefault(tile, []).append(name)

    plan_path = reports_root / "map_extended" / "plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        rendered = set(plan.get("already_rendered", {}))
        for tile in plan.get("tiles", []):
            label = (
                "map_extended (planned, adopted)"
                if tile in rendered
                else "map_extended (planned, to render)"
            )
            sources.setdefault(tile, []).append(label)

    rows = []
    for tile, srcs in sorted(sources.items()):
        lon0, lat0, lon1, lat1 = _tile_bounds_deg(tile)
        rows.append(
            {
                "tile": tile,
                "products": "; ".join(srcs),
                "shipped": any(s.startswith(("map_region", "map_a1")) for s in srcs),
                "planned_extension": any(s.startswith("map_extended") for s in srcs),
                "lon_min": lon0,
                "lat_min": lat0,
                "lon_max": lon1,
                "lat_max": lat1,
                "geometry": box(
                    _lon_to_x(lon0), _lat_to_y(lat0), _lon_to_x(lon1), _lat_to_y(lat1)
                ),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=CTX_WKT)


# Shapefile truncates field names to 10 chars, silently colliding e.g.
# `n_detections_kept` / `n_detections_raw`. Rename up front so the .dbf is readable.
SHAPEFILE_FIELD_NAMES = {
    "boulder_label": "blabel",
    "terrain_note": "terrain",
    "quality_note": "quality",
    "center_lat": "ctr_lat",
    "center_lon_180": "ctr_lon",
    "map_pixel_mpp": "mpp",
    "n_detections_kept": "n_det_keep",
    "n_detections_raw": "n_det_raw",
    "truncated_source": "trunc_src",
    "planned_extension": "planned",
    "west_lon_180": "lon_w",
    "east_lon_180": "lon_e",
}


def _write_layers(
    layers: dict[str, gpd.GeoDataFrame], out: Path, fmt: str
) -> list[Path]:
    """Write `layers` in `fmt`, returning the paths actually created."""
    if fmt == "gpkg":
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        for name, gdf in layers.items():
            gdf.to_file(out, layer=name, driver="GPKG")
        return [out]

    if fmt == "filegdb":
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            shutil.rmtree(out)
        for name, gdf in layers.items():
            # Without TARGET_ARCGIS_VERSION the driver demotes int64 detection
            # counts to Float64, so they surface in Pro as 359933.0.
            gdf.to_file(
                out,
                layer=name,
                driver="OpenFileGDB",
                TARGET_ARCGIS_VERSION="ARCGIS_PRO_3_2_OR_LATER",
            )
        return [out]

    if fmt == "shapefile":
        out.mkdir(parents=True, exist_ok=True)
        written = []
        for name, gdf in layers.items():
            path = out / f"{name}.shp"
            gdf.rename(columns=SHAPEFILE_FIELD_NAMES).to_file(
                path, driver="ESRI Shapefile"
            )
            written.append(path)
        return written

    raise SystemExit(f"unknown --format {fmt!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-f",
        "--format",
        choices=("gpkg", "filegdb", "shapefile"),
        default="gpkg",
        help="gpkg (QGIS), filegdb (ArcGIS Pro native), shapefile (universal fallback)",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output path; defaults to reports/context/hirise2ctx_context.<ext>",
    )
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "hirise_40_vclaire.csv")
    ap.add_argument("--cache-root", type=Path, default=REPO_ROOT / "cache_v2")
    ap.add_argument("--reports-root", type=Path, default=REPO_ROOT / "reports")
    args = ap.parse_args()

    if args.out is None:
        suffix = {"gpkg": ".gpkg", "filegdb": ".gdb", "shapefile": "_shp"}[args.format]
        args.out = REPO_ROOT / "reports" / "context" / f"hirise2ctx_context{suffix}"

    n_checked = _verify_tile_convention(args.cache_root / "ctx_tiles")
    print(f"tile-name convention verified against {n_checked} cached tile rasters")

    footprints, centers = build_hirise_layers(args.manifest, args.cache_root)
    tiles = build_map_tiles_layer(args.reports_root)

    written = _write_layers(
        {
            "hirise_footprints": footprints,
            "hirise_centers": centers,
            "map_tiles": tiles,
        },
        args.out,
        args.format,
    )

    for path in written:
        print(f"wrote {path}")
    print(f"  hirise_footprints  {len(footprints):4d}")
    print(f"  hirise_centers     {len(centers):4d}")
    print(
        f"  map_tiles          {len(tiles):4d} "
        f"({int(tiles.shipped.sum())} shipped, "
        f"{int(tiles.planned_extension.sum())} in planned extension)"
    )
    print(
        f"  lon {footprints.west_lon_180.min():.2f}..{footprints.east_lon_180.max():.2f}  "
        f"lat {footprints.min_lat.min():.2f}..{footprints.max_lat.max():.2f}"
    )


if __name__ == "__main__":
    main()
