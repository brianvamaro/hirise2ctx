"""Fetch the regional-map validation rasters: MOLA topography + THEMIS/TES thermal (PLAN §3, phase 1).

Config-driven (config_v2.yaml `validation_rasters:` block). For each requested product,
windowed-reads the global mosaic around the circum-Chryse block and reprojects it onto the CTX
clon_0 CRS (read from the regional abundance mosaic so the output co-registers with
reports/map_region/). Caches to `cache_v2/validation/<product>_region.tif` (+ sidecar); idempotent.

Usage:
    python scripts/fetch_validation_data.py --product mola_dem        # one product
    python scripts/fetch_validation_data.py --all                     # all configured products
    python scripts/fetch_validation_data.py --all --res-m 463         # override target resolution
    python scripts/fetch_validation_data.py --product themis_night_ir --match-mosaic   # land on the 160 m abundance grid

THEMIS is a 15 GB global mosaic -> windowed `/vsicurl/` only (its config `read_mode`).
TLS: export HIRISE2CTX_INSECURE_TLS=1 if the USGS/ASU hosts present an incomplete cert chain.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.modeling  # noqa: F401,E402  OpenMP guard before numpy-heavy imports

import rasterio  # noqa: E402

from src import validation_retrieve as vr  # noqa: E402
from src.config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MAP_DIR = REPO / "reports" / "map_region"


def _ctx_crs_wkt() -> str:
    """The CTX clon_0 CRS, read from a regional abundance GeoTIFF (the inference output)."""
    tifs = sorted(MAP_DIR.glob("*_abundance.tif"))
    if not tifs:
        raise SystemExit(
            f"no abundance GeoTIFFs under {MAP_DIR} to read the CTX CRS from; "
            "run scripts/map_region.py first (or point --ctx-crs-from at one)."
        )
    with rasterio.open(tifs[0]) as src:
        return src.crs.to_wkt()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_v2.yaml")
    ap.add_argument("--product", action="append", default=[],
                    help="product key from config validation_rasters.products (repeatable)")
    ap.add_argument("--all", action="store_true", help="fetch every configured product")
    ap.add_argument("--res-m", type=float, default=None,
                    help="target pixel size (m); default = product native_res_m")
    ap.add_argument("--match-mosaic", action="store_true",
                    help="land on the regional_abundance_mosaic grid (exact co-registration)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    val = cfg.get("validation_rasters")
    if not val:
        raise SystemExit(f"{args.config} has no `validation_rasters:` block")
    products = val["products"]
    bounds = tuple(val["region_bounds_lonlat"])
    out_dir = cfg.cache_dir / val.get("cache_subdir", "validation")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = list(products) if args.all else args.product
    if not names:
        raise SystemExit("specify --product <key> (repeatable) or --all")
    unknown = [n for n in names if n not in products]
    if unknown:
        raise SystemExit(f"unknown product(s) {unknown}; configured: {sorted(products)}")

    ctx_crs = _ctx_crs_wkt()
    grid_kwargs = {}
    if args.match_mosaic:
        ref = MAP_DIR / "regional_abundance_mosaic.tif"
        if not ref.exists():
            raise SystemExit(f"--match-mosaic needs {ref}; run notebook 24 §2 to write it first")
        _, transform, shape = vr.reference_grid(ref)
        grid_kwargs = {"dst_transform": transform, "dst_shape": shape}

    for name in names:
        p = products[name]
        res_m = args.res_m or float(p["native_res_m"])
        out_path = out_dir / f"{name}_region.tif"
        print(f"[{name}] -> {out_path.relative_to(REPO)}  "
              f"({'mosaic grid' if args.match_mosaic else f'{res_m:g} m/px'})", flush=True)
        prov = vr.fetch_region_raster(
            name,
            source_url=p["url"],
            bounds_lonlat=bounds,
            dst_crs_wkt=ctx_crs,
            out_path=out_path,
            cache_dir=cfg.cache_dir,
            dst_res_m=None if args.match_mosaic else res_m,
            src_lon_domain=str(p.get("lon_domain", "180")),
            read_mode=p.get("read_mode", "vsicurl"),
            resampling=p.get("resampling", "bilinear"),
            overwrite=args.overwrite,
            **grid_kwargs,
        )
        print(f"    valid_frac={prov['valid_fraction']:.3f}  shape={prov['dst_shape']}", flush=True)


if __name__ == "__main__":
    main()
