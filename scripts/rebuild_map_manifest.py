#!/usr/bin/env python
"""Rebuild a map-output manifest from the tile sidecars on disk.

**Why this exists (2026-08-25).** Step 11 shipped 26/26 tiles on both arms with the *index*
damaged: `region_manifest.json` listed **21 of 26** and `a1_manifest.json` **1 of 26**. Three
distinct causes, none of which touched a raster:

  * a task killed mid-stride (wall clock, or the dead GPU on `sh03-12n13`) never reached its
    manifest write at all — worth 4 tiles;
  * `write_json_atomic` staged to a **fixed** `<path>.tmp`, so two tasks finishing in the same
    second collided and the loser died renaming a file the winner had already moved — worth
    `E0_N36`, and the crash came *after* the tile was committed;
  * `striping_a1_map.py` did not merge at all, it clobbered with a bare `write_text`, so the
    manifest recorded whichever single tile finished last.

The drivers now derive `results` from the sidecars at every write, so the index is self-healing
and any later run repairs it. This script does the same repair **without running a driver** —
no GPU, no re-render — which is what you want for an already-shipped generation.

The sidecar is the authority for how a tile was made (R14 writes it last, as the completion
marker, and it carries the full `run` block including `win_px`), so everything needed is on disk.
What CANNOT be reconstructed is `runs[]`: the per-run history of model paths, `ctx_tiles`,
size-floor digests and elapsed times. Runs already recorded are preserved; runs whose task died
before writing are simply gone, and `--note` lets you say so in the rebuilt file rather than
leaving a silent hole.

Usage:
    python scripts/rebuild_map_manifest.py reports/map_region_g2 reports/map_a1_g2
    python scripts/rebuild_map_manifest.py --dry-run reports/map_a1_g2
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- OpenMP/DLL bootstrap; must precede numpy

# `scripts.map_region` pulls in torch at import; loading it by spec keeps this runnable on a
# machine with no CUDA, which is the point of a repair tool.
_spec = importlib.util.spec_from_file_location(
    "_map_region_for_rebuild", Path(__file__).with_name("map_region.py"))
_mr = importlib.util.module_from_spec(_spec)
sys.modules["_map_region_for_rebuild"] = _mr
_spec.loader.exec_module(_mr)

MANIFEST_FOR = {"map_a1": "a1_manifest.json"}      # default is region_manifest.json


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
            print(f"  ⚠ {path.name} is not valid JSON — rebuilding from scratch")
    before = len(prev.get("tiles") or [])
    rows = _mr.tile_result_rows(d)
    grid_ids = set()
    for t in _mr.tile_sidecars(d).values():
        try:
            grid_ids.add(json.loads(t.read_text(encoding="utf-8")).get("grid_id"))
        except ValueError:
            pass
    if len(grid_ids) > 1:
        raise SystemExit(f"{d}: sidecars span {len(grid_ids)} grid_ids {grid_ids} — R01 says "
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
    doc = _mr.merge_manifest(path, out_dir=d, grid_id=grid_id, run_record=run_record,
                             results=None)
    print(f"  wrote {path} — {len(doc['tiles'])} tiles, {len(doc['runs'])} run record(s)")
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
