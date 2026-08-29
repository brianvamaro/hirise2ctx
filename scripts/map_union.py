r"""PLAN_MapValidation step 1 -- assemble the **read-only union** of every shipped map arm.

The five validation notebooks (30-34) must cover *all* mapped ground and must keep being
correct as the map grows. There are now two products on the same global R01 lattice --
``reports/map_region`` (26 tiles) and ``reports/map_extended`` (35 tiles) -- and they
**overlap in 8 tiles**::

    E-12_N32  E-12_N36  E-12_N40  E-12_N44  E-8_N32  E-8_N36  E-8_N40  E-8_N44

so the union is **53** tiles (26 + 35 - 8, measured -- PLAN_MapValidation's "54" was an
arithmetic slip) and a notebook that naively pools the two per-arm mosaics **double-counts 8
tiles = 15% of the footprint**, non-uniformly, biasing every pooled statistic toward that
block's terrain. This script produces one deduplicated mosaic per layer under
``reports/map_union`` and the notebooks read only that (via
``src.map_validation.load_union``).

Four things make it more than a merge over a longer file list:

* **Dedup asserts byte-equality, it does not choose.** ``map_extended`` *adopted* those 8
  tiles from ``map_region`` (``scripts/adopt_map_tiles.py``), so the two copies are the same
  bytes -- verified 8/8 x 3 layers. A genuine mismatch would mean two different heads
  rendered the same ground, which is not something a merge policy should paper over, so it is
  a **hard failure**.
* **``mosaic_geotiffs(require_shared_lattice=True)`` is the R01 gate**, exactly as in
  ``scripts/map_mosaics.py``. Both arms are on ``COARSE_GRID_ID``, so it must pass; if it ever
  fails, the union is refusing to bake a sub-cell phase into a whole-cell displacement.
* **The size-floor basis must be one basis.** ``abundance`` is meaningless without it, and a
  union spanning two bases would make one tag set a lie. Verified: all 61 tile rasters carry
  the single ``v2_mixed_floor_2`` basis off ``models/deployable_g2``.
* **The footprint closes**, on the deduplicated tile set: ``n_finite == n_tiles * 1479**2 -
  intra-tile nodata``. A residual means tiles overlapped on the lattice or a cell was lost.

**This script never writes into a source arm.** It refuses if ``--out`` is one of them, and
the per-arm mosaics stay frozen so their footprint gate, 12-gate sidecar QA and cell-for-cell
arm parity keep passing.

**Growing the map edits no code**: a round-2 product joins by adding one ``--source``.

Seams default OFF here. ``map_qa.seam_widths`` is a pure-Python per-row run census; the union
is ~169M cells (11 x 7 tile slots), ~2.4x the 26-tile mosaic, so it is minutes per layer. The
per-arm mosaics already passed it on the same rasters. Pass ``--seams`` to run it.

Run (laptop, CPU, a few minutes; ~680 MB per mosaic held one at a time)::

    conda run --no-capture-output -n geospatial python -u scripts/map_union.py
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

import numpy as np                                                          # noqa: E402
import rasterio                                                             # noqa: E402

from src import map_qa                                                      # noqa: E402
from src.map_manifest import file_sha256, write_json_atomic                 # noqa: E402
from src.mapping import (COARSE_GRID_ID, assert_murray_sphere,              # noqa: E402
                         mosaic_geotiffs, verify_geotiff, write_geotiff)

#: The arms the union is made of, in **precedence order** -- which only decides whose
#: (byte-identical) copy of a shared tile is read, never what the values are.
DEFAULT_SOURCES = (REPO / "reports" / "map_region", REPO / "reports" / "map_extended")
DEFAULT_OUT = REPO / "reports" / "map_union"


def _rel(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (out-of-repo scratch roots)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def arm_tiles(map_dir: Path, layer: str) -> list[str]:
    """Tile ids with a ``{tile}_{layer}.tif`` in this arm. Skips the arm's own mosaic."""
    suffix = f"_{layer}.tif"
    return sorted(p.name[: -len(suffix)] for p in Path(map_dir).glob(f"*{suffix}")
                  if not p.name.startswith("regional_"))


def resolve_union(sources, layer: str):
    """Deduplicate the tile rasters across arms, asserting byte-equality on any overlap.

    Returns ``(tile -> chosen raster path, tile -> every arm that has it)``. The first source
    that carries a tile wins, which is a no-op given the equality assertion -- the ordering
    exists only so the choice is deterministic and reportable, not so it can decide anything.

    A sha256 mismatch raises ``SystemExit``: two arms rendered the same ground differently,
    and there is no honest way to merge that. It is a fact about the products to resolve, not
    a parameter of the union.
    """
    chosen: dict[str, Path] = {}
    origin: dict[str, list[str]] = {}
    for d in sources:
        d = Path(d)
        for t in arm_tiles(d, layer):
            p = d / f"{t}_{layer}.tif"
            origin.setdefault(t, []).append(d.name)
            if t not in chosen:
                chosen[t] = p
                continue
            a, b = file_sha256(chosen[t]), file_sha256(p)
            if a != b:
                raise SystemExit(
                    f"{layer}: tile {t} differs between {_rel(chosen[t].parent)} ({a[:16]}) "
                    f"and {_rel(d)} ({b[:16]}). Two arms rendered the same ground with "
                    "different bytes -- that is a product inconsistency to resolve, not an "
                    "overlap to merge. Refusing to build a union that hides it.")
    return chosen, origin


def scan_tiles(chosen, layer: str):
    """One pass over the deduplicated tile rasters: tags to carry, plus the nodata census.

    Mirrors ``scripts/map_mosaics.scan_tiles`` -- same reasons (R84 put the size-floor
    mixture on every per-tile raster because the layer's units are meaningless without it,
    and the nodata census is what makes the footprint a closed account rather than a
    plausible-looking percentage) -- but reads paths that may come from different arms, so
    the "every tile agrees" check now also spans arms.
    """
    tagsets, per_tile_nodata = [], {}
    for t in sorted(chosen):
        with rasterio.open(chosen[t]) as ds:
            tagsets.append({k: v for k, v in ds.tags().items()
                            if k.startswith("SIZE_FLOOR_")})
            a = ds.read(1)
            miss = int(a.size - np.isfinite(a).sum())
        if miss:
            per_tile_nodata[t] = miss
    distinct = {json.dumps(ts, sort_keys=True) for ts in tagsets}
    if len(distinct) != 1:
        raise SystemExit(
            f"{layer}: {len(distinct)} distinct SIZE_FLOOR_* tag sets across "
            f"{len(chosen)} tiles spanning {len({p.parent.name for p in chosen.values()})} "
            "arm(s) -- the union cannot carry one honest size-floor basis, and `abundance` "
            "has no meaning without one. Do not report a mixed-basis union.")
    return dict(tagsets[0]), per_tile_nodata


def union_tags(chosen, origin, out_dir: Path, layer: str, base_tags: dict) -> dict:
    """The provenance a union mosaic must carry beyond its inherited size-floor basis."""
    tiles = sorted(chosen)
    adopted = sorted(t for t in tiles if len(origin[t]) > 1)
    tags = dict(base_tags)
    tags.update(
        MOSAIC_ARM="union", MOSAIC_LAYER=layer, MOSAIC_N_TILES=len(tiles),
        MOSAIC_TILES=",".join(tiles), MOSAIC_GRID_ID=COARSE_GRID_ID,
        MOSAIC_SOURCE_DIR=Path(out_dir).name,
        MOSAIC_BUILT_BY="scripts/map_union.py (PLAN_MapValidation step 1)",
        UNION_SOURCE_DIRS=",".join(dict.fromkeys(chosen[t].parent.name for t in tiles)),
        UNION_TILES=",".join(tiles), UNION_N_TILES=len(tiles),
        UNION_TILE_ORIGIN=json.dumps({t: chosen[t].parent.name for t in tiles},
                                     sort_keys=True),
        UNION_ADOPTED_TILES=",".join(adopted), UNION_N_ADOPTED=len(adopted),
        UNION_NOTE="deduplicated read surface over the shipped arms; dedup asserted "
                   "sha256-identical, never merged. Read via "
                   "src.map_validation.load_union; this script is the sole producer.")
    return tags


def build(chosen, origin, out_dir: Path, layer: str, *, seams: bool) -> dict:
    """Merge one layer into ``out_dir/regional_{layer}_mosaic.tif`` and gate its footprint."""
    tiles = sorted(chosen)
    out = Path(out_dir) / f"regional_{layer}_mosaic.tif"
    t0 = time.monotonic()
    base_tags, per_tile_nodata = scan_tiles(chosen, layer)
    tags = union_tags(chosen, origin, out_dir, layer, base_tags)
    arr, transform, crs_wkt = mosaic_geotiffs([chosen[t] for t in tiles],
                                              require_shared_lattice=True)
    radius = assert_murray_sphere(crs_wkt)
    write_geotiff(out, arr, transform, crs_wkt, tags=tags)
    n_finite = int(np.isfinite(arr).sum())
    why = verify_geotiff(out, expect_shape=arr.shape, expect_finite=n_finite)
    if why is not None:
        raise SystemExit(f"{out.name} failed verification: {why}")

    # the closed account: every finite union cell is a tile cell, and every missing tile
    # cell is a named tile's named nodata. A residual is an unexplained hole -- or, here,
    # the specific failure the dedup exists to prevent: two copies of one tile merged.
    tile_cells = 0
    for t in tiles:
        with rasterio.open(chosen[t]) as ds:
            tile_cells += ds.height * ds.width
    expected = tile_cells - sum(per_tile_nodata.values())
    if n_finite != expected:
        raise SystemExit(
            f"{layer}: footprint does not close -- union has {n_finite} finite cells but "
            f"{len(tiles)} deduplicated tiles supply {tile_cells} minus "
            f"{sum(per_tile_nodata.values())} intra-tile nodata = {expected} (residual "
            f"{n_finite - expected}). Either two tiles overlap on the lattice or a cell was "
            "lost in the merge.")
    adopted = sorted(t for t in tiles if len(origin[t]) > 1)
    rec = {"arm": "union", "layer": layer, "path": _rel(out),
           "n_tiles": len(tiles), "n_adopted_tiles": len(adopted),
           "sphere_radius_m": radius, "transform": list(transform)[:6],
           "seconds": round(time.monotonic() - t0, 1),
           "tile_cells_total": tile_cells, "intra_tile_nodata": per_tile_nodata,
           "intra_tile_nodata_total": sum(per_tile_nodata.values()),
           "footprint_closes": True,
           **map_qa.mosaic_footprint(arr)}
    if seams:
        rec["interior_nan_runs_by_width"] = map_qa.seam_widths(arr)
    print(f"  union {layer:10s} {rec['shape']}  finite {rec['finite_fraction']:.4%}  "
          f"[{rec.get('value_min', float('nan')):.4g}, "
          f"{rec.get('value_max', float('nan')):.4g}] "
          f"mean {rec.get('value_mean', float('nan')):.4g}  ({rec['seconds']}s)", flush=True)
    print(f"        footprint CLOSES: {n_finite} = {len(tiles)}x{tile_cells // len(tiles)} "
          f"- {rec['intra_tile_nodata_total']} intra-tile nodata on "
          f"{len(per_tile_nodata)} tile(s)", flush=True)
    if seams:
        print(f"        interior NaN runs by width: {rec['interior_nan_runs_by_width']}",
              flush=True)
    del arr
    return rec


def check_out_dir(out_dir: Path, sources) -> None:
    """Refuse the two ways this script could damage a frozen product."""
    sources = [Path(s) for s in sources]
    if out_dir in sources:
        raise SystemExit(
            f"--out {_rel(out_dir)} is also a --source. The union must be its own product: "
            "writing it into an arm would replace that arm's tagged mosaic with a "
            "wider-footprint look-alike, and the arms stay frozen so their footprint gate, "
            "sidecar QA and arm parity keep passing.")
    if len(set(sources)) != len(sources):
        raise SystemExit("--source lists the same directory twice; dedup is per tile, not "
                         "per path, so this would only mask a real overlap")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", nargs="*", default=[str(d) for d in DEFAULT_SOURCES],
                    help="map-arm directories to union, in precedence order. A new product "
                         "(e.g. a round-2 extension) joins by adding one path -- no code "
                         "change, per the plan-driven convention.")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="union output dir. Must NOT be one of --source: the per-arm "
                         "products stay frozen.")
    ap.add_argument("--layers", nargs="*", default=list(map_qa.LAYERS))
    ap.add_argument("--seams", action="store_true",
                    help="run the per-row interior NaN run census (minutes per layer at "
                         "union size; the per-arm mosaics already passed it)")
    args = ap.parse_args(argv)

    sources = [Path(s).resolve() for s in args.source]
    out_dir = Path(args.out).resolve()
    missing = [s for s in sources if not s.is_dir()]
    if missing:
        raise SystemExit("no such source dir(s): " + ", ".join(_rel(m) for m in missing))
    check_out_dir(out_dir, sources)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== map union ===", flush=True)
    print(f"sources: {', '.join(_rel(s) for s in sources)}", flush=True)
    print(f"out:     {_rel(out_dir)}", flush=True)
    records, manifest_layers = [], {}
    for layer in args.layers:
        chosen, origin = resolve_union(sources, layer)
        if not chosen:
            raise SystemExit(f"no *_{layer}.tif under any of: "
                             + ", ".join(_rel(s) for s in sources))
        adopted = sorted(t for t in chosen if len(origin[t]) > 1)
        n_rasters = sum(len(arm_tiles(s, layer)) for s in sources)
        print(f"  {layer:10s} {len(chosen)} tiles from {n_rasters} rasters "
              f"({len(adopted)} shared, sha256-identical)", flush=True)
        records.append(build(chosen, origin, out_dir, layer, seams=args.seams))
        manifest_layers[layer] = {
            "n_tiles": len(chosen),
            "tiles": sorted(chosen),
            "tile_origin": {t: chosen[t].parent.name for t in sorted(chosen)},
            "shared_tiles": adopted,
        }

    tile_sets = {layer: frozenset(v["tiles"]) for layer, v in manifest_layers.items()}
    if len(set(tile_sets.values())) != 1:
        raise SystemExit(
            "the layers do not cover the same tiles: "
            + "; ".join(f"{k}={len(v)}" for k, v in tile_sets.items())
            + ". The three targets of PLAN_MapValidation ruling 3 must describe the SAME "
              "cells, so a layer-dependent footprint is a defect, not a caveat.")

    manifest = {"grid_id": COARSE_GRID_ID,
                "built_by": "scripts/map_union.py (PLAN_MapValidation step 1)",
                "sources": [_rel(s) for s in sources],
                "layers": manifest_layers,
                "mosaics": records}
    # The manifest is the ONLY thing written, and it lands inside --out. `map_mosaics.py`
    # additionally copies its QA record to reports/figures/, which is fine for a script with
    # no --out but wrong here: a caller who points --out at a scratch root has said where the
    # output goes, and a second copy into a live tree is precisely the leak CLAUDE.md asks
    # scripts to avoid (the test-side write guard duly caught it). The manifest carries the
    # full per-layer QA, so there is nothing a second file would add but a way to disagree.
    mpath = write_json_atomic(out_dir / "union_manifest.json", manifest)
    print(f"\nwrote {_rel(mpath)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
