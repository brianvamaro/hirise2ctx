#!/usr/bin/env python3
"""Copy already-rendered map tiles into a second product directory, verifying as it goes.

When the map grows into a box that overlaps an existing product, the overlapping tiles have
already been rendered -- same head, same calibrator, same global R01 lattice, same CTX source.
Re-rendering them would cost GPU-hours to reproduce bytes we already have, and would risk
producing *different* bytes (fp16 GEMM kernel choice varies with `--batch`, so a re-render at a
different batch is not bit-identical). Copying is therefore both cheaper and more faithful.

What makes this a script rather than a `cp`:

* **It verifies against each sidecar's own `rasters[]` commit record** -- bytes and sha256 --
  on the source before copying and on the destination after, so a silently-truncated source or
  a half-finished copy fails here instead of surfacing as a strange map three steps later. This
  is the same check `scripts/verify_map_download.py` applies to a Sherlock transfer.
* **It refuses to cross lattices.** A tile's `grid_id` must match the destination's other
  tiles. Merging a pre-R01 raster into a post-R01 product is the one error the whole R01 fix
  exists to prevent, and file copying is exactly where it would sneak in.
* **It never overwrites.** A destination tile that already exists is left alone and reported;
  `--force` is required to replace one, and even then the source is verified first.

The adopted tile keeps its original sidecar verbatim, so its provenance still names the run that
actually produced it -- the destination product does not get to claim it rendered these.

    python scripts/adopt_map_tiles.py --from reports/map_region --to reports/map_extended \\
        --tiles E-12_N32 E-12_N36 E-12_N40 E-12_N44 E-8_N32 E-8_N36 E-8_N40 E-8_N44
    python scripts/adopt_map_tiles.py --from reports/map_region --to reports/map_extended \\
        --plan reports/map_extended/plan.json        # adopt everything the planner found
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.map_manifest import file_sha256, tile_sidecars                   # noqa: E402


def verify_tile(map_dir: Path, tile: str) -> tuple[list[Path], dict]:
    """Sidecar + rasters for `tile`, checked against the sidecar's own commit record.

    Returns `(paths_to_copy, sidecar_dict)`. Raises `SystemExit` with the specific failure --
    a missing sidecar, a missing raster, a byte-count mismatch and a hash mismatch are four
    different problems and collapsing them into "verification failed" wastes the diagnosis.
    """
    side = map_dir / f"{tile}.json"
    if not side.exists():
        raise SystemExit(f"{tile}: no sidecar at {side}")
    rec = json.loads(side.read_text(encoding="utf-8"))
    rasters = rec.get("rasters")
    if not rasters:
        raise SystemExit(f"{tile}: sidecar has no rasters[] commit record -- cannot verify. "
                         "This tile predates R14; copy it by hand if you accept that.")
    paths = [side]
    for r in rasters:
        p = map_dir / r["name"]
        if not p.exists():
            raise SystemExit(f"{tile}: sidecar names {r['name']} but it is not in {map_dir}")
        size = p.stat().st_size
        if size != r["bytes"]:
            raise SystemExit(f"{tile}: {r['name']} is {size} bytes, sidecar says {r['bytes']}")
        got = file_sha256(p)
        if got != r["sha256"]:
            raise SystemExit(f"{tile}: {r['name']} sha256 {got[:16]}... != sidecar "
                             f"{r['sha256'][:16]}...")
        paths.append(p)
    return paths, rec


def destination_grid_id(dst: Path) -> str | None:
    """The `grid_id` the destination product is already on, or None if it is empty.

    Uses `tile_sidecars` rather than its own idea of which JSONs are tiles, so this script and
    `verify_map_download.py` can never disagree about what is in a product directory.
    """
    for _tile, side in sorted(tile_sidecars(dst).items()):
        gid = json.loads(side.read_text(encoding="utf-8")).get("grid_id")
        if gid:
            return gid
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", required=True)
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument("--plan", default=None,
                    help="a plan_map_extent.py --json file; adopts its already_rendered tiles")
    ap.add_argument("--force", action="store_true", help="replace an existing destination tile")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if not src.is_dir():
        raise SystemExit(f"source {src} is not a directory")
    tiles = list(args.tiles)
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        from_plan = [t for t, dirs in plan.get("already_rendered", {}).items()
                     if src.name in dirs]
        print(f"plan {args.plan}: {len(from_plan)} tile(s) already rendered in {src.name}")
        tiles += [t for t in from_plan if t not in tiles]
    if not tiles:
        raise SystemExit("no tiles to adopt (pass --tiles and/or --plan)")

    dst_gid = destination_grid_id(dst) if dst.exists() else None
    print(f"=== adopt {len(tiles)} tile(s): {src} -> {dst} ===")
    print(f"    destination lattice: {dst_gid or '(empty destination)'}")

    planned: list[tuple[str, list[Path]]] = []
    for t in sorted(tiles):
        paths, rec = verify_tile(src, t)
        gid = rec.get("grid_id")
        if dst_gid and gid != dst_gid:
            raise SystemExit(f"{t}: grid_id {gid} != destination's {dst_gid} -- refusing to "
                             "mix lattices in one product (R01)")
        dst_gid = dst_gid or gid
        existing = [p for p in paths if (dst / p.name).exists()]
        if existing and not args.force:
            print(f"  {t:12s} SKIP -- already in {dst.name} ({len(existing)} file(s)); "
                  "--force to replace")
            continue
        planned.append((t, paths))
        mb = sum(p.stat().st_size for p in paths) / 1e6
        print(f"  {t:12s} OK   {len(paths)} file(s), {mb:6.1f} MB verified")

    if not planned:
        print("\nnothing to do")
        return 0
    if args.dry_run:
        print(f"\ndry run: would copy {sum(len(p) for _, p in planned)} files")
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for t, paths in planned:
        for p in paths:
            out = dst / p.name
            shutil.copy2(p, out)
            if out.suffix == ".tif":
                # re-verify at the destination: a copy is exactly where a half-written file
                # comes from, and the sidecar we just copied is the reference for the check.
                if file_sha256(out) != file_sha256(p):
                    raise SystemExit(f"{t}: copy of {p.name} does not match its source")
            n += 1
        print(f"  copied {t}")
    print(f"\ncopied {n} files into {dst}")
    print(f"verify:  python3 scripts/verify_map_download.py {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
