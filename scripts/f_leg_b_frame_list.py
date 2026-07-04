"""F pilot leg B — build the cohort frame list for Sherlock ISIS processing.

For each of the ~38 training HiRISE images, queries the cached SeamMap GeoPackage (or
builds one from the cached seammap shapefile) to find the CTX source frames whose footprints
intersect the training image's CTX window.  Deduplicates across all images to produce the
master frame list for ISIS processing, plus per-obs metadata consumed by the Sherlock-side
extract and laptop-side embed steps.

Outputs (all in reports/f_leg_b/):
  cohort_frame_list.csv   -- one row per unique CTX EDR  (PRODUCT_ID, VOLUME_ID, edr_url)
  obs_frame_map.csv       -- one row per (obs_id, PRODUCT_ID) pair
  cohort_obs_bounds.csv   -- one row per obs_id with CTX-CRS bounds + mosaic origin

Run (laptop):
  conda run -n geospatial python scripts/f_leg_b_frame_list.py [--verify]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import geopandas as gpd
import pandas as pd
import rasterio
from shapely.geometry import box

from src.ctx_edr import edr_url

MANIFEST = REPO / "hirise_40_vclaire.csv"
LABELS_DIR = REPO / "dataset_v2" / "labels"
SEAM_DIR = REPO / "cache" / "ctx_tiles"
OUT_DIR = REPO / "reports" / "f_leg_b"

# Mars_2015 equirectangular clon_0 (sphere 3396190 m) — the CTX mosaic CRS
_CTX_CRS_WKT = (
    'PROJCS["Mars_2015_Sphere_GCS_Equirectangular",'
    'GEOGCS["GCS_Mars_2015_Sphere",DATUM["D_Mars_2015_Sphere",'
    'SPHEROID["Mars_2015_Sphere_IAU_IAG",3396190.0,0.0]],'
    'PRIMEM["Reference_Meridian",0.0],UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Equirectangular"],'
    'PARAMETER["False_Easting",0.0],'
    'PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",0.0],'
    'PARAMETER["Standard_Parallel_1",0.0],'
    'UNIT["Meter",1.0]]'
)


def _murray_to_internal(name: str) -> str:
    """E016_N44 -> E16_N44, W008_N32 -> E-8_N32, E020_S64 -> E20_N-64."""
    m = re.fullmatch(r"([EW])(\d+)_([NS])(\d+)", name)
    if not m:
        raise ValueError(f"unrecognised Murray tile name: {name!r}")
    lon = (-1 if m.group(1) == "W" else 1) * int(m.group(2))
    lat = (-1 if m.group(3) == "S" else 1) * int(m.group(4))
    lat_str = f"N-{abs(lat)}" if lat < 0 else f"N{lat}"
    return f"E{lon}_{lat_str}"


def _load_frame_gpkg(tile: str) -> gpd.GeoDataFrame | None:
    """Return dissolved frame footprints for `tile`, building the GeoPackage if needed."""
    cache_gpkg = SEAM_DIR / f"_frames_{tile}.gpkg"
    if cache_gpkg.exists():
        return gpd.read_file(cache_gpkg)
    # Try to build from the local seammap shapefile cache
    seam_path = SEAM_DIR / f"_seammap_{tile}"
    if seam_path.is_dir():
        shps = list(seam_path.glob("*SeamMap.shp"))
        if shps:
            print(f"  building frame gpkg for {tile} from seammap …")
            g = gpd.read_file(shps[0])
            if g.crs is None:
                g = g.set_crs(_CTX_CRS_WKT)
            else:
                from rasterio.crs import CRS as RCRS
                g = g.to_crs(_CTX_CRS_WKT)
            g = g.dissolve(by="PRODUCT_ID", as_index=False)
            g.to_file(cache_gpkg, driver="GPKG")
            print(f"    cached {cache_gpkg.name} ({len(g)} frames)")
            return g
    return None


def _obs_bounds(obs_id: str) -> dict | None:
    """Read CTX window bounds + mosaic origin from the labels sidecar JSON."""
    sidecar_path = LABELS_DIR / f"{obs_id}.json"
    if not sidecar_path.exists():
        return None
    sc = json.loads(sidecar_path.read_text(encoding="utf-8"))
    tif = sc.get("ctx_window_tif")
    if not tif:
        return None
    tif = Path(tif)
    if not tif.exists():
        print(f"  {obs_id}: ctx_window_tif missing at {tif}; skipping")
        return None
    with rasterio.open(tif) as ds:
        b = ds.bounds
    return dict(obs_id=obs_id,
                minx=b.left, miny=b.bottom, maxx=b.right, maxy=b.top,
                row0=sc["mosaic_row_origin"], col0=sc["mosaic_col_origin"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="ranged-GET every EDR URL to confirm it is live (slow)")
    args = ap.parse_args()

    mf = pd.read_csv(MANIFEST)
    print(f"manifest: {len(mf)} images")

    frame_rows, obs_frame_rows, bounds_rows = [], [], []
    seen_pids: set[str] = set()

    for _, row in mf.iterrows():
        obs_id = row["ObsId"]
        murray_tile = str(row["CTX_TileName"])
        try:
            tile = _murray_to_internal(murray_tile)
        except ValueError as e:
            print(f"  {obs_id}: {e}; skipping")
            continue

        bd = _obs_bounds(obs_id)
        if bd is None:
            print(f"  {obs_id}: no bounds; skipping")
            continue
        bounds_rows.append(bd)

        footprint = box(bd["minx"], bd["miny"], bd["maxx"], bd["maxy"])

        frames = _load_frame_gpkg(tile)
        if frames is None:
            print(f"  {obs_id}: no frame data for tile {tile}; skipping")
            continue

        hits = frames[frames.geometry.intersects(footprint)]
        if len(hits) == 0:
            print(f"  {obs_id}: 0 frames in tile {tile} intersect footprint — "
                  f"footprint bounds: {bd['minx']:.0f},{bd['miny']:.0f},"
                  f"{bd['maxx']:.0f},{bd['maxy']:.0f}")
            continue

        for _, fr in hits.iterrows():
            pid = str(fr["PRODUCT_ID"])
            vid = str(fr["VOLUME_ID"])
            url = edr_url(vid, pid)
            obs_frame_rows.append({"obs_id": obs_id, "PRODUCT_ID": pid,
                                   "tile": tile, "VOLUME_ID": vid})
            if pid not in seen_pids:
                seen_pids.add(pid)
                frame_rows.append({"PRODUCT_ID": pid, "VOLUME_ID": vid, "edr_url": url})

        print(f"  {obs_id} ({tile}): {len(hits)} frame(s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fl = pd.DataFrame(frame_rows)
    om = pd.DataFrame(obs_frame_rows)
    ob = pd.DataFrame(bounds_rows)

    if args.verify:
        import urllib.request
        import truststore
        truststore.inject_into_ssl()
        statuses = []
        for url in fl["edr_url"]:
            req = urllib.request.Request(
                url, headers={"User-Agent": "hirise2ctx-research", "Range": "bytes=0-399"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    total = (r.headers.get("Content-Range", "") or "/").split("/")[-1]
                    statuses.append((r.status, int(total) / 1e6 if total.isdigit() else None))
            except Exception as e:
                statuses.append((getattr(e, "code", str(type(e).__name__)), None))
        fl["http"] = [s[0] for s in statuses]
        fl["size_mb"] = [round(s[1], 1) if s[1] else None for s in statuses]
        n_ok = sum(1 for s in statuses if s[0] == 206)
        print(f"verify: {n_ok}/{len(fl)} live EDRs")

    fl.to_csv(OUT_DIR / "cohort_frame_list.csv", index=False)
    om.to_csv(OUT_DIR / "obs_frame_map.csv", index=False)
    ob.to_csv(OUT_DIR / "cohort_obs_bounds.csv", index=False)

    print(f"\n=== SUMMARY ===")
    print(f"unique CTX frames : {len(fl)}")
    print(f"obs_id × frame    : {len(om)}")
    print(f"obs_ids with data : {ob['obs_id'].nunique()}")
    n_tasks = max(1, len(fl) // 3)   # rough target: ~3 frames/task → ~1h on 24-task array
    print(f"\nSherlock wall-clock estimate: {len(fl) * 22 / 60:.0f} CPU-h  "
          f"≈ {len(fl) * 22 / n_tasks:.0f} min on {n_tasks} tasks "
          f"(set N_TASKS in run_f_leg_b.sbatch)")
    print(f"\nOutput: {OUT_DIR}")


if __name__ == "__main__":
    main()
