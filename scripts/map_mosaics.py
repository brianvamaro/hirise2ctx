"""PLAN_Rebuild step 12 -- assemble the regional mosaics for both map arms, and QA them.

Merges the 26 per-tile GeoTIFFs of each arm into one regional raster per layer
(``abundance`` / ``prob`` / ``prob_raw``), writes an A1-minus-baseline difference mosaic for
the shipped ``abundance`` layer, and emits a footprint/nodata/parity QA report.

Three things make this more than a `gdal_merge` wrapper:

* **``mosaic_geotiffs(require_shared_lattice=True)`` is itself the R01 gate.** It refuses to
  merge rasters that are not on the one global lattice, and it *fails by design* on the
  pre-R01 product now archived at ``reports/map_region_g1/``. A clean merge here is positive
  evidence that step 11 rendered on ``COARSE_GRID_ID``.
* **Nodata is not one thing.** The 26 tiles form an L, and the tile pitch (~1481.9 cells)
  exceeds each tile raster's 1479 cells, so thin seams run between adjacent tiles.
  ``src.map_qa.seam_widths`` separates those two expected populations from a real hole.
* **The difference mosaic is only meaningful because the arms are cell-for-cell
  co-registered** (``scripts/verify_arm_parity.py``). Run that first; this asserts shape.

Run (laptop, CPU, a few minutes; ~281 MB per mosaic held one at a time):

    C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run --no-capture-output -n geospatial \\
        python -u scripts/map_mosaics.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                                        # noqa: E402
import rasterio                                                           # noqa: E402

from src import map_qa                                                    # noqa: E402
from src.mapping import (COARSE_GRID_ID, assert_murray_sphere, mosaic_geotiffs,  # noqa: E402
                         verify_geotiff, write_geotiff)

ARMS = {"baseline": REPO / "reports" / "map_region",
        "a1": REPO / "reports" / "map_a1"}
FIG = REPO / "reports" / "figures"


def _rel(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise.

    A bare ``relative_to(REPO)`` raises on any out-of-repo ``--baseline``, which is
    exactly the explicit-scratch-root invocation CLAUDE.md asks scripts to be run with
    when they must not touch a live artifact.
    """
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def arm_tiles(map_dir: Path, layer: str) -> list[str]:
    return sorted(p.name[: -len(f"_{layer}.tif")]
                  for p in map_dir.glob(f"*_{layer}.tif"))


def scan_tiles(map_dir: Path, tiles: list[str], layer: str, arm: str) -> tuple[dict, dict]:
    """One pass over the tile rasters: size-floor tags to carry, plus the nodata census.

    R84 put the size-floor mixture on every per-tile raster because the layer's units are
    meaningless without it. A mosaic that dropped those tags would be a *less* self-describing
    product than its own inputs, so they are copied forward -- but only after checking that
    every tile agrees, since a mixed basis across the mosaic would make one tag set a lie.

    The nodata census exists to make the mosaic's footprint a **closed account** rather than a
    plausible-looking percentage. Six of the 26 tiles carry genuine intra-tile nodata where the
    CTX mosaic has no coverage (102-2,781 cells, 0.005-0.127 %), so
    ``n_finite == n_tiles * 1479**2 - sum(per-tile nodata)`` exactly. Without this term the
    leftover would have to be waved through as "some nodata is normal", which is how a real
    hole survives QA.
    """
    tagsets, per_tile_nodata = [], {}
    for t in tiles:
        with rasterio.open(map_dir / f"{t}_{layer}.tif") as ds:
            tagsets.append({k: v for k, v in ds.tags().items()
                            if k.startswith("SIZE_FLOOR_")})
            a = ds.read(1)
            miss = int(a.size - np.isfinite(a).sum())
        if miss:
            per_tile_nodata[t] = miss
    distinct = {json.dumps(ts, sort_keys=True) for ts in tagsets}
    if len(distinct) != 1:
        raise SystemExit(f"{arm}/{layer}: {len(distinct)} distinct SIZE_FLOOR_* tag sets across "
                         f"{len(tiles)} tiles -- the mosaic cannot carry one honest basis")
    tags = dict(tagsets[0])
    tags.update(MOSAIC_ARM=arm, MOSAIC_LAYER=layer, MOSAIC_N_TILES=len(tiles),
                MOSAIC_TILES=",".join(tiles), MOSAIC_GRID_ID=COARSE_GRID_ID,
                MOSAIC_SOURCE_DIR=map_dir.name,
                MOSAIC_BUILT_BY="scripts/map_mosaics.py (PLAN_Rebuild step 12)")
    return tags, per_tile_nodata


def build(map_dir: Path, arm: str, layer: str, *, seams: bool) -> dict:
    tiles = arm_tiles(map_dir, layer)
    if not tiles:
        raise SystemExit(f"no *_{layer}.tif under {map_dir}")
    paths = [map_dir / f"{t}_{layer}.tif" for t in tiles]
    out = map_dir / f"regional_{layer}_mosaic.tif"
    t0 = time.monotonic()
    tags, per_tile_nodata = scan_tiles(map_dir, tiles, layer, arm)
    arr, transform, crs_wkt = mosaic_geotiffs(paths, require_shared_lattice=True)
    radius = assert_murray_sphere(crs_wkt)
    write_geotiff(out, arr, transform, crs_wkt, tags=tags)
    n_finite = int(np.isfinite(arr.astype(np.float32)).sum())
    why = verify_geotiff(out, expect_shape=arr.shape, expect_finite=n_finite)
    if why is not None:
        raise SystemExit(f"{out.name} failed verification: {why}")
    # the closed account: every finite mosaic cell is a tile cell, and every missing tile
    # cell is a named tile's named nodata. A residual here is an unexplained hole.
    tile_cells = 0
    for t in tiles:
        with rasterio.open(map_dir / f"{t}_{layer}.tif") as ds:
            tile_cells += ds.height * ds.width
    expected = tile_cells - sum(per_tile_nodata.values())
    if n_finite != expected:
        raise SystemExit(
            f"{arm}/{layer}: footprint does not close -- mosaic has {n_finite} finite cells but "
            f"{len(tiles)} tiles supply {tile_cells} minus {sum(per_tile_nodata.values())} "
            f"intra-tile nodata = {expected} (residual {n_finite - expected}). Either tiles "
            "overlap on the lattice or a cell was lost in the merge.")
    rec = {"arm": arm, "layer": layer, "path": _rel(out),
           "n_tiles": len(tiles), "sphere_radius_m": radius,
           "transform": list(transform)[:6], "seconds": round(time.monotonic() - t0, 1),
           "tile_cells_total": tile_cells, "intra_tile_nodata": per_tile_nodata,
           "intra_tile_nodata_total": sum(per_tile_nodata.values()),
           "footprint_closes": True,
           **map_qa.mosaic_footprint(arr)}
    if seams:
        rec["interior_nan_runs_by_width"] = map_qa.seam_widths(arr)
    print(f"  {arm:9s} {layer:10s} {rec['shape']}  finite {rec['finite_fraction']:.4%}  "
          f"[{rec.get('value_min', float('nan')):.4g}, {rec.get('value_max', float('nan')):.4g}] "
          f"mean {rec.get('value_mean', float('nan')):.4g}  ({rec['seconds']}s)", flush=True)
    print(f"            footprint CLOSES: {n_finite} = {len(tiles)}x{tile_cells // len(tiles)} "
          f"- {rec['intra_tile_nodata_total']} intra-tile nodata on "
          f"{len(per_tile_nodata)} tile(s)", flush=True)
    if seams:
        print(f"            interior NaN runs by width: {rec['interior_nan_runs_by_width']}",
              flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ARMS["baseline"]))
    ap.add_argument("--a1", default=str(ARMS["a1"]),
                    help="the A1 sensitivity arm; pass 'none' for a SINGLE-ARM product "
                         "(e.g. reports/map_extended), which also skips the difference "
                         "mosaic")
    ap.add_argument("--layers", nargs="*", default=list(map_qa.LAYERS))
    ap.add_argument("--diff-layer", default="abundance",
                    help="layer for the A1-minus-baseline difference mosaic")
    ap.add_argument("--no-seams", action="store_true",
                    help="skip the per-row interior NaN run census (the slow part)")
    args = ap.parse_args()

    # A1 is OPTIONAL. This script was written for the two-arm 26-tile rebuild, where both
    # arms always existed. Pointing --baseline at a single-arm product (map_extended) left
    # --a1 at its default, which would have (1) REWRITTEN the shipped reports/map_a1 mosaics
    # as a side effect and (2) then died differencing a 35-tile mosaic against a 26-tile one.
    # Neither is anything the caller asked for, so "no second arm" is now a first-class case.
    dirs = {"baseline": Path(args.baseline)}
    if args.a1 and args.a1.lower() != "none":
        dirs["a1"] = Path(args.a1)
    print("=== regional mosaics ===", flush=True)
    records = []
    for arm, d in dirs.items():
        for layer in args.layers:
            records.append(build(d, arm, layer, seams=not args.no_seams))

    if "a1" not in dirs:
        FIG.mkdir(parents=True, exist_ok=True)
        out = FIG / f"mosaic_qa_{Path(args.baseline).name}.json"
        out.write_text(json.dumps({"grid_id": COARSE_GRID_ID, "mosaics": records,
                                   "difference": None}, indent=2), encoding="utf-8")
        print(f"\nsingle arm ({Path(args.baseline).name}) -- no difference mosaic",
              flush=True)
        print(f"wrote {_rel(out)}", flush=True)
        return 0

    print("\n=== A1 - baseline difference mosaic ===", flush=True)
    lay = args.diff_layer
    with rasterio.open(dirs["baseline"] / f"regional_{lay}_mosaic.tif") as ds:
        base = ds.read(1).astype(np.float64)
        transform, crs_wkt = ds.transform, ds.crs.to_wkt()
    with rasterio.open(dirs["a1"] / f"regional_{lay}_mosaic.tif") as ds:
        a1 = ds.read(1).astype(np.float64)
        if tuple(ds.transform)[:6] != tuple(transform)[:6]:
            raise SystemExit("the two mosaics do not share a transform -- not differenceable")
    diff = map_qa.difference_stats(base, a1)
    dpath = dirs["a1"] / f"regional_{lay}_minus_baseline.tif"
    write_geotiff(dpath, a1 - base, transform, crs_wkt,
                  tags={"MOSAIC_LAYER": f"{lay}_a1_minus_baseline",
                        "MOSAIC_GRID_ID": COARSE_GRID_ID,
                        "MOSAIC_BUILT_BY": "scripts/map_mosaics.py (PLAN_Rebuild step 12)",
                        "MOSAIC_NOTE": "A1 minus baseline; legitimate only because "
                                       "verify_arm_parity.py established cell-for-cell "
                                       "co-registration on one lattice"})
    print(f"  wrote {_rel(dpath)}", flush=True)
    for k, v in diff.items():
        print(f"    {k:14s} {v}", flush=True)

    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "step12_mosaic_qa.json"
    out.write_text(json.dumps({"grid_id": COARSE_GRID_ID, "mosaics": records,
                               "difference": {"layer": lay, **diff}},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {_rel(out)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
