#!/usr/bin/env python3
"""Rebuild a map-output manifest from the tile sidecars on disk.

**Why this exists (2026-08-25).** Step 11 shipped 26/26 tiles on both arms with the *index*
damaged: `region_manifest.json` listed **21 of 26** and `a1_manifest.json` **1 of 26**. Three
distinct causes, none of which touched a raster:

  * a task killed mid-stride (wall clock, or the dead GPU on `sh03-12n13`) never reached its
    manifest write at all  --  worth 4 tiles;
  * `write_json_atomic` staged to a **fixed** `<path>.tmp`, so two tasks finishing in the same
    second collided and the loser died renaming a file the winner had already moved  --  worth
    `E0_N36`, and the crash came *after* the tile was committed;
  * `striping_a1_map.py` did not merge at all, it clobbered with a bare `write_text`, so the
    manifest recorded whichever single tile finished last.

The drivers now derive `results` from the sidecars at every write, so the index is self-healing
and any later run repairs it. This script does the same repair **without running a driver**  -- 
no GPU, no re-render  --  which is what you want for an already-shipped generation.

The sidecar is the authority for how a tile was made (R14 writes it last, as the completion
marker, and it carries the full `run` block including `win_px`), so everything needed is on disk.
What CANNOT be reconstructed is `runs[]`: the per-run history of model paths, `ctx_tiles`,
size-floor digests and elapsed times. Runs already recorded are preserved; runs whose task died
before writing are simply gone, and `--note` lets you say so in the rebuilt file rather than
leaving a silent hole.

Usage (needs python3 >= 3.10; on Sherlock `ml python/3.12.1` first, because
its bare `python` is 2.7 and `python3` is 3.6):
    python3 scripts/rebuild_map_manifest.py reports/map_region_g2 reports/map_a1_g2
    python3 scripts/rebuild_map_manifest.py --dry-run reports/map_a1_g2
"""
import sys

# NOTE ON THE GUARD BELOW, and what it cannot do. Python 2 cannot be caught this way at all:
# function annotations and f-strings are Python-3-only SYNTAX, so under Sherlock's bare
# `python` (2.7) this file fails to compile at the first `def`, and no runtime check can run.
# Three rounds were lost to that class of mistake (DECISIONS 2026-08-25c/d/e). The guard covers
# old Python **3** -- 3.6 on a Sherlock login node -- which is the reachable case. For 2.7 the
# shebang above is the defence: `./scripts/<tool>.py` picks python3, never python2.

# WARNING: everything above this block must parse under a VERY old interpreter, or the guard
# can never run. Two rounds were lost to exactly that (DECISIONS 2026-08-25c/d):
#   * a non-ASCII docstring gave "SyntaxError: Non-ASCII character ... no encoding declared"
#     under Sherlock's default `python`, which is 2.7;
#   * a `__future__` import of `annotations` gave "SyntaxError: future feature annotations is
#     not defined" under its `python3`, which is 3.6. A __future__ import is COMPILE-time, so
#     no runtime check can be placed before it.
# Hence: ASCII-only source, no __future__ import, no walrus, and this block before every other
# import. The annotations further down use PEP 604/585 syntax, which merely *parses* on old
# versions and is never evaluated, because sys.exit() runs first.
if sys.version_info < (3, 10):
    sys.stderr.write(
        "\nERROR: this needs Python >= 3.10; it is running under %s (%s).\n"
        "The tool is deliberately standard-library only so it works when the full project\n"
        "environment does not -- but it still needs a modern interpreter.\n"
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

# STANDARD LIBRARY ONLY, deliberately. This used to load `map_region.py`, which does
# `import src.modeling` (the torch/OpenMP bootstrap) -- so repairing a JSON index required
# CUDA-capable torch, and on a Sherlock login node without the module loaded it did not even
# get that far: the system `python` is 2.7 and it died on a non-ASCII SyntaxError. A recovery
# tool must not share the heavy dependencies of the thing it recovers.
from src.map_manifest import merge_manifest, tile_result_rows, tile_sidecars

def manifest_path(d: Path) -> Path:
    """Which manifest a directory carries. Prefers one that already exists."""
    for name in ("region_manifest.json", "a1_manifest.json"):
        if (d / name).exists():
            return d / name
    # nothing there yet -- guess from the directory name
    return d / ("a1_manifest.json" if "a1" in d.name else "region_manifest.json")


def rebuild(d: Path, *, dry_run: bool, note: str | None) -> tuple[int, int]:
    """Returns (tiles_before, tiles_after)."""
    path = manifest_path(d)
    prev = {}
    if path.exists():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            print(f"  WARNING: {path.name} is not valid JSON  --  rebuilding from scratch")
    before = len(prev.get("tiles") or [])
    rows = tile_result_rows(d)
    grid_ids = set()
    for t in tile_sidecars(d).values():
        try:
            grid_ids.add(json.loads(t.read_text(encoding="utf-8")).get("grid_id"))
        except ValueError:
            pass
    if len(grid_ids) > 1:
        raise SystemExit(f"{d}: sidecars span {len(grid_ids)} grid_ids {grid_ids}  --  R01 says "
                         f"products on two lattices must never be indexed as one product")
    grid_id = grid_ids.pop() if grid_ids else prev.get("grid_id")

    print(f"=== {d} -> {path.name} ===")
    print(f"  tiles indexed: {before} -> {len(rows)}")
    recovered = sorted({r["tile"] for r in rows} - set(prev.get("tiles") or []))
    if recovered:
        print(f"  recovered {len(recovered)}: {recovered}")
    print(f"  runs preserved: {len(prev.get('runs') or [])}")
    if dry_run:
        print("  (--dry-run: nothing written)")
        return before, len(rows)

    run_record = None
    if note:
        # Not a real run record -- it is a marker saying the index was repaired out of band, so
        # that `runs[]` never implies a rendering pass that did not happen.
        run_record = {"rebuilt_from_sidecars": True, "note": note,
                      "tiles": [r["tile"] for r in rows]}
    doc = merge_manifest(path, out_dir=d, grid_id=grid_id,
                         run_record=run_record)
    print(f"  wrote {path}  --  {len(doc['tiles'])} tiles, {len(doc['runs'])} run record(s)")
    return before, len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note", default=None,
                    help="append a `rebuilt_from_sidecars` marker to runs[] with this note, so "
                         "the repair is visible in the provenance instead of silent")
    args = ap.parse_args()
    for d in args.dirs:
        if not d.is_dir():
            raise SystemExit(f"{d}: not a directory")
        rebuild(d, dry_run=args.dry_run, note=args.note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
