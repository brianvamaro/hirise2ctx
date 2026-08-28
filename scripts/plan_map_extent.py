"""Plan a map-inference run over a lat/lon box: tile list, GPU-hours, download bytes.

`scripts/map_region.py` is tile-list-driven, so extending the map to a new region is a
planning problem before it is a compute problem. This script answers the three questions
that decide whether a region is worth submitting, **from measured artifacts rather than
from a guess**:

* **Which Murray tiles?** The box is snapped out to whole 4-degree tiles, and the snap is
  reported explicitly -- a box whose edges are not multiples of 4 always maps to a slightly
  larger footprint, and quoting the requested box as if it were the delivered one is how a
  map ends up described wrong in a caption.
* **How long?** Every tile is the same 47,420^2 px and sweeps the same 144 read windows, so
  wall-clock scales by tile count. The per-window seconds come from `region_manifest.json`'s
  own `runs[].elapsed_s / (n_tiles * 144)` -- the real spread across the shipped run, not a
  single number. It is **GPU-conditional**: the shipped baseline ran on a 2080 Ti at ~17.6
  s/window and the A1 arm's Pascal cards took ~10x that (project_state_2026-08-24b), which is
  what turned a comfortable 6 h wall-clock into a timeout.
* **How much download?** ~1.8 GB per CTX tile zip, fetched on the cluster before any GPU time.
  Tiles already cached locally are excluded from the estimate, but the cluster keeps its own
  cache that this script cannot see -- so it says so rather than implying otherwise.

`--verify-urls` issues one ranged request per tile against the Murray Lab mosaic to confirm
the tile is actually published (the mosaic is not a complete planetary grid) and to read its
true `Content-Length` instead of assuming the average. Off by default so the planning
arithmetic stays offline.

    python scripts/plan_map_extent.py --lat 20 35 --lon -25 -5
    python scripts/plan_map_extent.py --lat 20 35 --lon -25 -5 --verify-urls --json plan.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TILE_DEG = 4
WINDOWS_PER_TILE = 144        # 12 x 12 read windows over a 47,420^2 px tile at win_px=4096
NOMINAL_ZIP_GB = 1.81         # mean of the 24 locally cached Murray zips
PRODUCT_MB_PER_TILE = 11.6    # prob + abundance + prob_raw, measured on the shipped tiles
DEFAULT_MANIFEST = REPO / "reports" / "map_region" / "region_manifest.json"
DEFAULT_MAP_DIRS = [REPO / "reports" / "map_region"]
DEFAULT_CTX_CACHE = REPO / "cache_v2" / "ctx_tiles"


def _rel(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise (`--manifest` may point anywhere)."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def snap_box(lo0: float, lo1: float, la0: float, la1: float) -> tuple[int, int, int, int]:
    """Smallest whole-tile box containing the request. Returns (lon0, lon1, lat0, lat1)."""
    lo0, lo1 = sorted((lo0, lo1))
    la0, la1 = sorted((la0, la1))
    return (int(math.floor(lo0 / TILE_DEG) * TILE_DEG),
            int(math.ceil(lo1 / TILE_DEG) * TILE_DEG),
            int(math.floor(la0 / TILE_DEG) * TILE_DEG),
            int(math.ceil(la1 / TILE_DEG) * TILE_DEG))


def tiles_in_box(lon0: int, lon1: int, lat0: int, lat1: int) -> list[str]:
    """Murray tile ids covering a snapped box, north-west first.

    Murray names a tile by its **lower-left** corner (`E-12_N32` spans lon [-12,-8],
    lat [32,36]), so the last row starts at `lat1 - TILE_DEG`, not at `lat1`.
    """
    return [f"E{lo}_N{la}"
            for la in range(lat1 - TILE_DEG, lat0 - TILE_DEG, -TILE_DEG)
            for lo in range(lon0, lon1, TILE_DEG)]


def expected_digests(map_dirs, tiles) -> dict:
    """The head + calibration digests the already-rendered tiles were made with.

    Recorded INTO the plan, at plan time, on the machine where those tiles actually live. The
    Sherlock job's preflight then compares against the plan rather than against a product
    directory -- which on the cluster is a symlink to `$SCRATCH`, and was empty, so the check
    found no basis and *skipped itself* on the first real submission. A gate that silently
    becomes a no-op when its reference is absent is the "absence of measurement reported as a
    pass" failure this project keeps paying for. The plan travels with the repo, so the
    reference cannot go missing.

    Returns `{}` when nothing is already rendered (a genuinely fresh product has nothing to
    match), and raises when the existing tiles disagree with each other -- that is a mixed
    product, and no single expectation would be honest.
    """
    from src.map_manifest import tile_sidecars

    heads, calibs, seen = set(), set(), set()   # a set: an adopted tile appears in BOTH dirs
    for d in map_dirs:
        sides = tile_sidecars(Path(d))
        for t in tiles:
            if t not in sides:
                continue
            rec = json.loads(sides[t].read_text(encoding="utf-8"))
            if "head_digest" in rec:
                heads.add(rec["head_digest"])
                calibs.add(rec.get("calibration_digest"))
                seen.add(t)
    if not heads:
        return {}
    if len(heads) != 1 or len(calibs) != 1:
        raise SystemExit(f"the {len(seen)} already-rendered tiles are MIXED: {len(heads)} head "
                         f"digest(s), {len(calibs)} calibration digest(s) -- no single "
                         "expectation can be recorded")
    return {"head_digest": heads.pop(), "calibration_digest": calibs.pop(),
            "measured_from": sorted(seen)}


def window_seconds(manifest_path: Path) -> dict:
    """Measured seconds-per-window from a shipped run's own manifest.

    Each `runs[]` entry timed a whole sbatch task, so `elapsed_s / (n_tiles * 144)` is that
    task's per-window rate. Entries with no `elapsed_s` are dropped, and so are ones under
    1 s/window -- those are resumed tasks that found every tile already done and are a
    measurement of the skip path, not of inference.
    """
    if not manifest_path.exists():
        return {"source": "none", "median": None, "min": None, "max": None, "n_runs": 0}
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    rates = []
    for r in man.get("runs", []):
        elapsed, n = r.get("elapsed_s"), len(r.get("tiles", []))
        if not elapsed or not n:
            continue
        rate = elapsed / (n * WINDOWS_PER_TILE)
        if rate < 1.0:
            continue
        rates.append(rate)
    if not rates:
        return {"source": str(manifest_path), "median": None, "min": None, "max": None,
                "n_runs": 0}
    return {"source": _rel(manifest_path), "median": statistics.median(rates),
            "min": min(rates), "max": max(rates), "n_runs": len(rates)}


def verify_tile_urls(tiles: list[str], url_template: str) -> dict[str, dict]:
    """One ranged GET per tile: is it published, and how big is it really?

    A ranged GET rather than a HEAD: the Murray Lab host sits behind a CDN that answers the
    two differently, which is the same class of behaviour that made the USGS fetch look like
    a corrupt source in rebuild step 12. `Content-Range` carries the true total size.

    **It must try both name forms, exactly as `ensure_tile_cached` does.** Murray Lab's live
    filenames are signed-prefix ZERO-PADDED (`E-024_N28`), not the bare signed-int form
    (`E-24_N28`) this project uses as its tile id. Checking only the bare form returns 404 on
    every western tile and reports a fully-published region as entirely missing -- which is
    what the first run of this script did.
    """
    import truststore
    truststore.inject_into_ssl()
    import urllib.error
    import urllib.request

    from src.ctx_retrieve import _padded_manifest_form

    def probe(url):
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            crange = resp.headers.get("Content-Range", "")
            total = int(crange.rsplit("/", 1)[-1]) if "/" in crange else None
            return {"published": True, "status": resp.status, "bytes": total, "url": url}

    out = {}
    for t in tiles:
        names = [t] + [n for n in (_padded_manifest_form(t),) if n]
        for i, name in enumerate(names):
            url = url_template.format(tile_name=name)
            try:
                out[t] = {**probe(url), "resolved_tile_name": name}
                break
            except urllib.error.HTTPError as exc:
                out[t] = {"published": False, "status": exc.code, "bytes": None, "url": url,
                          "resolved_tile_name": None}
                if exc.code != 404 or i == len(names) - 1:
                    break
            except Exception as exc:                                    # noqa: BLE001
                out[t] = {"published": None, "status": f"{type(exc).__name__}: {exc}",
                          "bytes": None, "url": url, "resolved_tile_name": None}
                break
        st = out[t]
        flag = "OK  " if st["published"] else "MISS"
        print(f"  {t:12s} -> {str(st['resolved_tile_name'] or '?'):12s} {flag} {st['status']}  "
              f"{(st['bytes'] or 0) / 1e9:.2f} GB", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", nargs=2, type=float, required=True, metavar=("LAT0", "LAT1"))
    ap.add_argument("--lon", nargs=2, type=float, required=True, metavar=("LON0", "LON1"),
                    help="east longitude, negative for west (25W = -25)")
    ap.add_argument("--map-dirs", nargs="*", default=[str(p) for p in DEFAULT_MAP_DIRS],
                    help="existing product dirs; tiles already rendered there are excluded")
    ap.add_argument("--ctx-cache", default=str(DEFAULT_CTX_CACHE))
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--gpus", type=int, default=6, help="concurrent Slurm array tasks")
    ap.add_argument("--verify-urls", action="store_true")
    ap.add_argument("--json", default=None, help="write the plan to this path")
    args = ap.parse_args()

    lon0, lon1, lat0, lat1 = snap_box(args.lon[0], args.lon[1], args.lat[0], args.lat[1])
    tiles = tiles_in_box(lon0, lon1, lat0, lat1)
    n_lon, n_lat = (lon1 - lon0) // TILE_DEG, (lat1 - lat0) // TILE_DEG
    req = (min(args.lon), max(args.lon), min(args.lat), max(args.lat))

    print(f"=== requested  lon[{req[0]:g}, {req[1]:g}]  lat[{req[2]:g}, {req[3]:g}] ===")
    print(f"    snapped to lon[{lon0}, {lon1}]  lat[{lat0}, {lat1}]  "
          f"= {n_lon} x {n_lat} = {len(tiles)} Murray tiles of {TILE_DEG} deg")
    if (lon0, lon1, lat0, lat1) != req:
        print("    the delivered footprint is LARGER than the request -- quote the snapped box")

    done: dict[str, list[str]] = {}
    for dname in args.map_dirs:
        d = Path(dname)
        for t in tiles:
            if (d / f"{t}_abundance.tif").exists():
                done.setdefault(t, []).append(d.name)
    todo = [t for t in tiles if t not in done]
    print(f"\n    already rendered: {len(done)}  {sorted(done) or ''}")
    print(f"    to render:        {len(todo)}")

    cache = Path(args.ctx_cache)
    cached = [t for t in todo if (cache / f"{t}.zip").exists()]
    to_fetch = [t for t in todo if t not in cached]
    print(f"\n    CTX zips cached LOCALLY: {len(cached)} {sorted(cached) or ''}")
    print(f"    CTX zips to fetch:       {len(to_fetch)}  "
          f"~{len(to_fetch) * NOMINAL_ZIP_GB:.0f} GB at ~{NOMINAL_ZIP_GB} GB/tile")
    print("    (that is the LAPTOP cache; the cluster keeps its own and this cannot see it)")

    rates = window_seconds(Path(args.manifest))
    if rates["median"] is None:
        print("\n    no measured per-window rate available; cannot estimate wall-clock")
    else:
        print(f"\n=== timing, from {rates['source']} ({rates['n_runs']} timed runs) ===")
        print(f"    measured {rates['min']:.1f}-{rates['max']:.1f} s/window "
              f"(median {rates['median']:.1f}) on a 2080 Ti; {WINDOWS_PER_TILE} windows/tile")
        for label, rate in (("fastest", rates["min"]), ("median", rates["median"]),
                            ("slowest", rates["max"])):
            gpu_h = len(todo) * WINDOWS_PER_TILE * rate / 3600
            print(f"      {label:8s} {rate:5.1f} s/win -> "
                  f"{rate * WINDOWS_PER_TILE / 60:5.1f} min/tile  {gpu_h:6.1f} GPU-h  -> "
                  f"{gpu_h / max(args.gpus, 1):5.1f} h wall-clock on {args.gpus} GPUs")
        print("    GPU-conditional: the A1 arm's Pascal cards ran ~10x slower and timed out.")

    print(f"\n    products: {len(todo)} x ~{PRODUCT_MB_PER_TILE:.1f} MB = "
          f"~{len(todo) * PRODUCT_MB_PER_TILE / 1000:.2f} GB to bring home")

    exp = expected_digests(args.map_dirs, tiles)
    if exp:
        print(f"\n    the new tiles MUST be rendered by this head "
              f"({len(exp['measured_from'])} already-rendered tile(s) say so):")
        print(f"      head_digest        {exp['head_digest']}")
        print(f"      calibration_digest {exp['calibration_digest']}")
        print("    recorded in the plan; the Sherlock job refuses to start without a match")
    else:
        print("\n    nothing already rendered -- the plan records NO head expectation, so the "
              "job's preflight will have nothing to check")

    plan = {"requested": {"lon": [req[0], req[1]], "lat": [req[2], req[3]]},
            "snapped": {"lon": [lon0, lon1], "lat": [lat0, lat1]},
            "n_tiles": len(tiles), "tiles": tiles,
            "already_rendered": done, "to_render": todo,
            "expect_digests": expected_digests(args.map_dirs, tiles),
            "ctx_cached_locally": cached, "ctx_to_fetch": to_fetch,
            "seconds_per_window": rates, "windows_per_tile": WINDOWS_PER_TILE}

    if args.verify_urls:
        from src.config import load_config
        tmpl = load_config("config_v2.yaml")["ctx_mosaic"]["url_template"]
        print(f"\n=== verifying {len(todo)} tile URLs at the Murray Lab mosaic ===")
        plan["url_check"] = verify_tile_urls(todo, tmpl)
        missing = [t for t, v in plan["url_check"].items() if v["published"] is not True]
        real = [v["bytes"] for v in plan["url_check"].values() if v.get("bytes")]
        if real:
            print(f"    true total: {sum(real) / 1e9:.1f} GB over {len(real)} published tiles "
                  f"(mean {statistics.mean(real) / 1e9:.2f} GB)")
        print(f"    NOT PUBLISHED: {missing}" if missing else "    all tiles published")
        plan["n_not_published"] = len(missing)

    if args.json:
        Path(args.json).write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
