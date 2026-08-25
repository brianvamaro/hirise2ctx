#!/usr/bin/env python3
"""Check that two map arms can actually be COMPARED, not just that each is complete.

`verify_map_download.py` proves one directory is internally complete and uncorrupted. It says
nothing about whether two arms may be differenced -- which is the entire point of shipping two
rows (section 5.1's one-common-footprint rule), and the thing a step-12 comparison silently
assumes. This checks the assumption:

  * identical tile sets, and the expected count;
  * one `grid_id` across BOTH arms -- R01: products on two lattices must never be compared or
    merged, and the failure is invisible because each raster is individually perfect;
  * identical `(ti_min, tj_min)`, `raster_shape`, `tile_px` and `grid_cell_m` per tile, so the
    rows line up CELL FOR CELL rather than merely both being "26 tiles of Chryse";
  * one size-floor basis across both arms -- the rows are differenced, so they must count the
    same boulders (R84);
  * no leftover `partials/`, which would mean a tile never assembled.

Read-only. Exits non-zero, so it can gate a promotion step.

Usage (needs python3 >= 3.10; on Sherlock `ml python/3.12.1` first, because its bare `python`
is 2.7 and `python3` is 3.6):
    python3 scripts/verify_arm_parity.py reports/map_region_g2 reports/map_a1_g2
"""
import sys

# See scripts/rebuild_map_manifest.py for why this block is where it is and what it cannot do:
# nothing above it may use syntax an old interpreter cannot compile, or the guard never runs.
if sys.version_info < (3, 10):
    sys.stderr.write(
        "\nERROR: this needs Python >= 3.10; it is running under %s (%s).\n"
        "On a Sherlock login node the default `python` is 2.7 and `python3` is 3.6; do:\n"
        "    ml python/3.12.1\n"
        "    python %s --help\n\n"
        % (sys.version.split()[0], sys.executable, sys.argv[0]))
    sys.exit(2)

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.map_manifest import tile_sidecars

#: Fields that must match tile-for-tile for the two rows to be differenced.
ALIGN_FIELDS = ("ti_min", "tj_min", "raster_shape", "tile_px", "grid_cell_m")
MANIFEST_OF = {"region_manifest.json", "a1_manifest.json"}


def load_arm(d: Path) -> dict:
    out = {}
    for tile, p in tile_sidecars(d).items():
        out[tile] = json.loads(p.read_text(encoding="utf-8"))
    return out


def check(dirs: list[Path], expect_tiles: int) -> list[str]:
    problems = []
    arms = {d.name: load_arm(d) for d in dirs}
    for name, recs in arms.items():
        print(f"{name}: {len(recs)} tiles")
        if expect_tiles and len(recs) != expect_tiles:
            problems.append(f"{name}: {len(recs)} tiles, expected {expect_tiles}")

    names = list(arms)
    ref = names[0]
    for other in names[1:]:
        only_ref = sorted(set(arms[ref]) - set(arms[other]))
        only_oth = sorted(set(arms[other]) - set(arms[ref]))
        if only_ref or only_oth:
            problems.append(f"tile sets differ: {ref}-only {only_ref}, "
                            f"{other}-only {only_oth}")

    all_recs = [r for recs in arms.values() for r in recs.values()]
    grids = {r.get("grid_id") for r in all_recs}
    print(f"grid_id across all arms: {grids if len(grids) != 1 else next(iter(grids))}")
    if len(grids) != 1:
        problems.append(f"arms span {len(grids)} lattices {grids} -- R01 forbids comparing them")

    shared = sorted(set.intersection(*(set(r) for r in arms.values())))
    aligned = 0
    for t in shared:
        bad = [f for f in ALIGN_FIELDS
               if len({json.dumps(arms[n][t].get(f), sort_keys=True) for n in names}) != 1]
        if bad:
            problems.append(f"{t}: not co-registered, {bad} differ between arms")
        else:
            aligned += 1
    print(f"cell-for-cell co-registered: {aligned}/{len(shared)} tiles")

    digests = {r.get("size_floor_basis_digest") for r in all_recs}
    print(f"size-floor basis digests: {len(digests)}")
    if len(digests) > 1:
        problems.append(f"{len(digests)} size-floor bases across the arms -- the rows are "
                        f"differenced, so they must count the same boulders (R84)")

    for d in dirs:
        leftover = list(d.glob("partials/*"))
        if leftover:
            problems.append(f"{d.name}: {len(leftover)} leftover partials/ entries -- a tile "
                            f"may never have assembled")
        for m in MANIFEST_OF:
            if (d / m).exists():
                doc = json.loads((d / m).read_text(encoding="utf-8"))
                n = len(doc.get("tiles") or [])
                print(f"{d.name}/{m}: {n} tiles indexed, "
                      f"{len(doc.get('runs') or [])} run record(s)")
                if expect_tiles and n != expect_tiles:
                    problems.append(f"{d.name}/{m}: indexes {n} tiles, expected {expect_tiles} "
                                    f"-- run scripts/rebuild_map_manifest.py")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--expect-tiles", type=int, default=26,
                    help="0 to skip the count check (default: 26 circum-Chryse tiles)")
    args = ap.parse_args()
    for d in args.dirs:
        if not d.is_dir():
            raise SystemExit(f"{d}: not a directory")
    problems = check(args.dirs, args.expect_tiles)
    print()
    if problems:
        print(f"=== {len(problems)} PROBLEM(S) ===")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"=== ARM PARITY: PASS -- {len(args.dirs)} arms complete, aligned, comparable ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
