#!/usr/bin/env python
"""Verify a downloaded map-output directory against its own sidecars. **Read-only.**

R14 made every tile sidecar carry a `rasters[]` commit record  --  name, bytes, sha256, shape,
n_finite  --  precisely so that content can be checked without re-deriving it. Nothing used that
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
import sys

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

# No `import src.modeling` here on purpose: this script does not touch torch, and a verifier
# that needs the training environment cannot check a transfer on a bare login node.
from src.map_manifest import file_sha256, tile_sidecars

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

    sidecars = tile_sidecars(d)          # excludes MANIFEST_NAMES
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
