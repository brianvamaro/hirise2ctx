"""Fetch the Murray Lab CTX tile zips a map run needs, from a tile list or a plan file.

Every previous expansion did this with a heredoc pasted into SHERLOCK_RUN.md holding a
hardcoded tile list. That works exactly once: the next region needs the list edited in a
markdown file, which is the one place nothing checks it. This reads `plan_map_extent.py`'s
JSON instead, so growing the map is a re-plan, not a doc edit.

Three things it adds over a bare `ensure_tile_cached` loop:

* **It reports the disk it is about to consume before consuming it**, from the plan's own
  URL check when one was run (true `Content-Length` per tile) and from the ~1.8 GB average
  otherwise. 27 tiles is ~48 GB, which is a number worth seeing before a login-node fetch.
* **It is resumable and says so.** `ensure_tile_cached` is a no-op on a cached tile, so a
  re-run after a dropped connection costs one `stat` per finished tile.
* **A tile that fails does not abort the rest.** A transient PDS/Caltech hiccup on tile 9 of
  27 should not throw away tiles 10-27; failures are collected and re-listed at the end with
  a non-zero exit, so the retry is a re-run of the same command.

    python scripts/fetch_ctx_tiles.py --plan reports/map_extended/plan.json
    python scripts/fetch_ctx_tiles.py --tiles E-24_N44 E-20_N44
    python scripts/fetch_ctx_tiles.py --plan ... --dry-run     # just the disk budget

On Sherlock, `export HIRISE2CTX_INSECURE_TLS=1` first -- Murray Lab serves an incomplete
certificate chain and Linux has no fallback (SHERLOCK_RUN.md, the TLS note).
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

NOMINAL_ZIP_GB = 1.81


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default=None, help="plan_map_extent.py --json output")
    ap.add_argument("--tiles", nargs="*", default=[])
    ap.add_argument("--config", default="config_v2.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from src.config import load_config
    from src.ctx_retrieve import ensure_tile_cached

    tiles, sizes = list(args.tiles), {}
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        tiles += [t for t in plan["to_render"] if t not in tiles]
        for t, rec in (plan.get("url_check") or {}).items():
            if rec.get("bytes"):
                sizes[t] = rec["bytes"]
    if not tiles:
        raise SystemExit("no tiles requested (pass --tiles and/or --plan)")

    cfg = load_config(args.config)
    tmpl, cache = cfg["ctx_mosaic"]["url_template"], Path(cfg["cache_dir"])
    tiles_dir = cache / "ctx_tiles"

    have = [t for t in tiles if (tiles_dir / f"{t}.zip").exists()]
    need = [t for t in tiles if t not in have]
    budget = sum(sizes.get(t, NOMINAL_ZIP_GB * 1e9) for t in need) / 1e9
    measured = sum(1 for t in need if t in sizes)
    print(f"=== CTX tiles for {len(tiles)} map tiles -> {tiles_dir} ===")
    print(f"    already cached: {len(have)}")
    print(f"    to fetch:       {len(need)}  ~{budget:.0f} GB "
          f"({measured} sizes measured, {len(need) - measured} at the "
          f"{NOMINAL_ZIP_GB} GB average)")
    if args.dry_run:
        print("    dry run -- fetching nothing")
        return 0
    if not need:
        return 0

    failed = {}
    t0 = time.monotonic()
    for i, t in enumerate(need, 1):
        try:
            zip_path, inner = ensure_tile_cached(t, url_template=tmpl, cache_dir=cache)
            gb = zip_path.stat().st_size / 1e9
            print(f"  [{i}/{len(need)}] {t:12s} {gb:.2f} GB  {inner}", flush=True)
        except Exception as exc:                                        # noqa: BLE001
            failed[t] = f"{type(exc).__name__}: {exc}"
            print(f"  [{i}/{len(need)}] {t:12s} FAILED  {failed[t]}", flush=True)

    mins = (time.monotonic() - t0) / 60
    print(f"\nfetched {len(need) - len(failed)}/{len(need)} in {mins:.1f} min")
    if failed:
        print("FAILED (re-run this same command to retry only these):")
        for t, why in failed.items():
            print(f"  {t:12s} {why}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
