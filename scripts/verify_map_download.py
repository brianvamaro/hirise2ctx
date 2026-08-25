#!/usr/bin/env python
"""Verify a downloaded map-output directory against its own sidecars. **Read-only.**

R14 made every tile sidecar carry a `rasters[]` commit record — name, bytes, sha256, shape,
n_finite — precisely so that content can be checked without re-deriving it. Nothing used that
until now, which meant a transfer off Sherlock was trusted on file count alone. A truncated or
silently-corrupted GeoTIFF has the right name and very nearly the right size.

Checks, per directory:
  * every expected tile has a sidecar, and no sidecar is for an unexpected tile;
  * every raster named in a sidecar exists, with matching **bytes** and **sha256**;
  * all tiles agree on `grid_id` (products on two lattices must never be merged -- R01);
  * the overlap record is summarised, tolerating BOTH field sets: tiles rendered before
    2026-08-24f have `fraction` = the raw fraction and no `n_significant`, later ones have
    `n_significant` / `fraction_raw` and `fraction` = the gate quantity. Conflating them is a
    real trap, so this prints which schema each tile uses rather than averaging over both.

Usage:
    python scripts/verify_map_download.py reports/map_region_g2 reports/map_a1_g2
    python scripts/verify_map_download.py --quick reports/map_a1_g2      # sizes only, no hashing

Exits non-zero if anything fails, so it can gate a promotion step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- OpenMP/DLL bootstrap; must precede numpy

from src.mapping import file_sha256

# scripts.map_region imports torch via DeployableHead; the tile list is all we need, and this
# script must stay runnable on a machine with no CUDA, so it is duplicated deliberately. Kept
# honest by test_verify_map_download_tile_list_matches_the_driver.
BLOCK_TILES = [
    "E-12_N32", "E-12_N36", "E-12_N40", "E-12_N44",
    "E-8_N32", "E-8_N36", "E-8_N40", "E-8_N44",
    "E-4_N32", "E-4_N36", "E-4_N40", "E-4_N44",
    "E0_N32", "E0_N36", "E0_N40", "E0_N44",
    "E4_N32", "E4_N36", "E4_N40", "E4_N44",
    "E8_N32", "E8_N36", "E8_N40", "E8_N44",
    "E12_N44", "E16_N44",
]


def overlap_line(rec: dict) -> str:
    """One-line summary of a tile's overlap record, naming which schema it uses."""
    ov = rec.get("overlap")
    if not isinstance(ov, dict):
        n = rec.get("overlap_disagreements")
        return f"pre-2026-08-24d scalar only (overlap_disagreements={n})"
    gate = ov.get("gate_layer", "?")
    g = ov.get(gate, {})
    if "n_significant" in g:
        return (f"[floor] {gate}: {g['n_significant']}/{g['n_dup']} significant "
                f"({100 * g.get('fraction', 0):.4f} %), {g.get('n_disagree')} at any magnitude "
                f"({100 * g.get('fraction_raw', 0):.4f} %), max |d| {g.get('max_abs')}")
    return (f"[pre-floor] {gate}: {g.get('n_disagree')}/{g.get('n_dup')} disagree at ANY "
            f"magnitude ({100 * g.get('fraction', 0):.4f} % -- RAW fraction, not the gate "
            f"quantity), max |d| {g.get('max_abs')}")


def verify_dir(d: Path, *, expect: list[str], quick: bool) -> list[str]:
    """Verify one map-output directory. Returns a list of problem strings (empty = clean)."""
    problems: list[str] = []
    print(f"\n=== {d} ===", flush=True)
    if not d.is_dir():
        return [f"{d}: not a directory"]

    sidecars = {p.stem: p for p in sorted(d.glob("*.json"))
                if p.stem not in ("region_manifest", "a1_manifest")}
    missing = [t for t in expect if t not in sidecars]
    extra = [t for t in sidecars if t not in expect]
    print(f"sidecars: {len(sidecars)}/{len(expect)}")
    if missing:
        problems.append(f"{d}: {len(missing)} tile(s) have NO sidecar: {missing}")
    if extra:
        problems.append(f"{d}: {len(extra)} unexpected sidecar(s): {extra}")

    grid_ids, n_rasters, total_bytes = set(), 0, 0
    for tile in sorted(sidecars):
        rec = json.loads(sidecars[tile].read_text(encoding="utf-8"))
        grid_ids.add(rec.get("grid_id"))
        rasters = rec.get("rasters") or []
        if not rasters:
            problems.append(f"{tile}: sidecar carries no `rasters` commit record")
        bad = []
        for r in rasters:
            p = d / r["name"]
            if not p.exists():
                bad.append(f"{r['name']} MISSING")
                continue
            size = p.stat().st_size
            total_bytes += size
            n_rasters += 1
            if size != r["bytes"]:
                bad.append(f"{r['name']} size {size} != {r['bytes']}")
                continue
            if not quick:
                got = file_sha256(p)
                if got != r["sha256"]:
                    bad.append(f"{r['name']} sha256 {got[:12]} != {r['sha256'][:12]}")
        status = "OK " if not bad else "BAD"
        print(f"  {status} {tile:10s} {len(rasters)} raster(s)  {overlap_line(rec)}", flush=True)
        for b in bad:
            problems.append(f"{tile}: {b}")

    if len(grid_ids) > 1:
        problems.append(f"{d}: tiles span {len(grid_ids)} grid_ids {grid_ids} -- R01 says "
                        f"products on two lattices must never be compared or merged")
    print(f"grid_id: {grid_ids.pop() if len(grid_ids) == 1 else grid_ids}")
    print(f"{n_rasters} raster(s), {total_bytes / 2**30:.2f} GiB, "
          f"{'sizes only (--quick)' if quick else 'sha256 verified'}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--quick", action="store_true",
                    help="check sizes only, skip hashing (fast, much weaker)")
    ap.add_argument("--tiles", nargs="*", default=None,
                    help="expected tile list; default is the 26 circum-Chryse BLOCK_TILES")
    args = ap.parse_args()

    expect = args.tiles if args.tiles else BLOCK_TILES
    problems: list[str] = []
    for d in args.dirs:
        problems += verify_dir(d, expect=expect, quick=args.quick)

    print()
    if problems:
        print(f"=== {len(problems)} PROBLEM(S) ===")
        for p in problems:
            print(f"  {p}")
        return 1
    print(f"=== ALL CLEAN: {len(args.dirs)} directory/ies, {len(expect)} tile(s) each ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
