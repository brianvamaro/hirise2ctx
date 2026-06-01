"""Stage 7c -- per-tile colour features for the colour-covered fraction of the v2 cohort.

Per [`PLAN_Compositional.md`](../PLAN_Compositional.md) §3 (with Stage 7b folded in
2026-05-31 night). For each S=64 tile in each colour-covered image, reproject the
tile bounds CTX -> source-CRS at read time, do a windowed read of the COLOR.JP2,
compute the per-band mean I/F, apply the per-image Lambertian correction, and
emit the band ratios + a `dust_index`.

Output: `dataset_v2/features_colour.parquet` with one row per (obs_id, scale_idx,
ti, tj) tile that has >= MIN_TILE_PIXELS valid colour pixels. Joinable on
(obs_id, scale_idx, ti, tj) against `dataset_v2/labels/{ObsId}.parquet`.

Columns:
  obs_id              :  HiRISE observation id
  scale_idx           :  3 (S=64; tile_size_px=64; tile_size_m ~= 320)
  ti, tj              :  tile grid indices in CTX pixel space
  n_color_pixels      :  number of valid colour pixels contributing to the mean
  IR_iof, RED_iof, BG_iof : Lambertian-corrected per-band mean I/F
                          (I/F_observed / cos(incidence_deg))
  IR_over_RED, IR_over_BG, dust_index_RED_over_BG : band ratios
                          (Lambertian-invariant since cos(i) cancels)
  cos_incidence       :  per-image cos(incidence_deg) carried for reproducibility

Tiles that fall entirely outside the colour swath, or have fewer than
MIN_TILE_PIXELS valid pixels (i.e. mostly off-swath or saturated), are omitted
rather than emitted as NaN -- downstream (Stage 7d) does an inner-join on
(obs_id, scale_idx, ti, tj) and the absence of a row IS the "no colour" signal.

Architectural note: Stage 7b was *eliminated* (DECISIONS.md 2026-05-31 night) in
favour of this stay-in-source-CRS pattern. No per-image colour raster cache is
ever built; the windowed read replaces the would-be reprojected raster.

Run via:
    conda run --no-capture-output -n geospatial python -u scripts/run_stage7c_features.py
Optional flags:
    --only ESP_XXXXXX_XXXX [...]  : process only these obs_ids (sanity-run subset)
    --scale-idx N                 : process scale (default 3 = S=64)
    --out PATH                    : override output parquet path

Typical full-cohort runtime: ~30-90 min for 37 images, dominated by JP2 windowed reads.
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.crs import CRS as RioCRS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import colour  # noqa: E402

print = functools.partial(print, flush=True)  # noqa: A001

# ----------------------------------------------------------------------- config
CACHE = Path("cache_v2")
LABELS_DIR = Path("dataset_v2/labels")
DEFAULT_OUT = Path("dataset_v2/features_colour.parquet")
COVERAGE_PARQUET = CACHE / "hirise_color" / "coverage.parquet"
LBL_META_PARQUET = CACHE / "hirise_color" / "lbl_metadata.parquet"

# Tile filter -- minimum valid colour pixels per tile to emit a row.
# At map_scale=0.5 m/px this is 16 m^2 of valid swath inside a 320x320 m tile (0.016%),
# at map_scale=0.25 m/px it's 4 m^2. Effectively just "tile barely overlaps swath".
MIN_TILE_PIXELS = 64

# Default scale: 3 = S=64 = tile_size_px=64 (~320 m at 5 m/px CTX). Matches
# Stage 7.0 Test B and the P4 fa_gt_1e-2 binary partition.
DEFAULT_SCALE_IDX = 3

# CTX target CRS (Mars_2000 IAU sphere @ 3396190 m, from Stage 1 sidecar target_crs_wkt).
CTX_CRS = RioCRS.from_wkt(
    'PROJCRS["Mars_2000_Equidistant_Cylindrical",BASEGEOGCRS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",ELLIPSOID["Mars_2000_IAU_IAG",3396190,0,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Reference_Meridian",0,ANGLEUNIT["Degree",0.0174532925199433]]],'
    'CONVERSION["Equidistant Cylindrical (Spherical)",'
    'METHOD["Equidistant Cylindrical (Spherical)",ID["EPSG",1029]],'
    'PARAMETER["Latitude of 1st standard parallel",0,'
    'ANGLEUNIT["Degree",0.0174532925199433]],'
    'PARAMETER["Longitude of natural origin",0,'
    'ANGLEUNIT["Degree",0.0174532925199433]],'
    'PARAMETER["False easting",0,LENGTHUNIT["metre",1]],'
    'PARAMETER["False northing",0,LENGTHUNIT["metre",1]]],'
    'CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1]],'
    'AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1]]]'
)

PROGRESS_EVERY = 500


# ----------------------------------------------------------------------- core
def process_image(
    obs_id: str,
    *,
    cache_dir: Path,
    labels_dir: Path,
    scale_idx: int,
    min_tile_pixels: int,
) -> pd.DataFrame:
    """Extract Lambertian-corrected per-tile colour features for one image.

    Returns an empty DataFrame (with the expected columns) if the image has no
    colour, no Stage 1 sidecar, or no tiles intersect the swath. Never raises on
    missing-data cases; raises only on truly unexpected conditions (e.g. label
    parquet missing a required column).
    """
    out_cols = [
        "obs_id", "scale_idx", "ti", "tj", "n_color_pixels",
        "IR_iof", "RED_iof", "BG_iof",
        "IR_over_RED", "IR_over_BG", "dust_index_RED_over_BG",
        "cos_incidence",
    ]
    empty = pd.DataFrame({c: pd.Series(dtype=_dtype_for(c)) for c in out_cols})

    jp2_path = colour.color_jp2_path(cache_dir, obs_id)
    lbl_path = colour.color_lbl_path(cache_dir, obs_id)
    if not jp2_path.exists() or not lbl_path.exists():
        print(f"  SKIP: no COLOR.JP2 or .LBL for {obs_id}")
        return empty

    corrected_crs = colour.corrected_source_crs(obs_id, cache_dir)
    if corrected_crs is None:
        print(f"  SKIP: no Stage 1 corrected CRS sidecar for {obs_id}")
        return empty

    label_path = labels_dir / f"{obs_id}.parquet"
    if not label_path.exists():
        print(f"  SKIP: no labels parquet for {obs_id}")
        return empty

    lbl = colour.parse_color_lbl(lbl_path)
    cos_i = lbl.cos_incidence
    if cos_i <= 0:
        print(f"  SKIP: non-illuminated geometry for {obs_id} (incidence={lbl.incidence_deg})")
        return empty

    labels = pd.read_parquet(label_path)
    tiles = labels[labels["scale_idx"] == scale_idx].reset_index(drop=True)
    if tiles.empty:
        print(f"  SKIP: no tiles at scale_idx={scale_idx} for {obs_id}")
        return empty

    transformer = pyproj.Transformer.from_crs(CTX_CRS, corrected_crs, always_xy=True)

    print(f"  {len(tiles)} tiles @ scale_idx={scale_idx}, cos(i)={cos_i:.4f}, "
          f"map_scale={lbl.map_scale_mpp} m/px")

    records: list[dict] = []
    n_outside = n_pixel_skip = 0
    t0 = time.time()

    # Cache ones-masks by (H, W). Tile windows have a small number of distinct shapes
    # since tiles are uniform-sized in CTX CRS and the reprojection only varies by a
    # pixel or two in the source CRS.
    ones_cache: dict[tuple[int, int], np.ndarray] = {}

    with rasterio.open(jp2_path) as ds:
        jp2_bounds = tuple(ds.bounds)
        for tile_idx, row in tiles.iterrows():
            if tile_idx and tile_idx % PROGRESS_EVERY == 0:
                rate = tile_idx / max(0.01, time.time() - t0)
                print(f"    [{obs_id}] {tile_idx}/{len(tiles)} tiles "
                      f"({rate:.1f}/s, kept={len(records)})")
            tile_bounds = (row.xmin, row.ymin, row.xmax, row.ymax)
            arr, _ = colour.windowed_colour_read(
                ds, tile_bounds, transformer=transformer, jp2_bounds=jp2_bounds
            )
            if arr is None:
                n_outside += 1
                continue
            shape = arr.shape[1:]
            ones = ones_cache.get(shape)
            if ones is None:
                ones = np.ones(shape, dtype=bool)
                ones_cache[shape] = ones
            means = colour.region_means(arr, ones, min_pixels=min_tile_pixels)
            if means is None:
                n_pixel_skip += 1
                continue
            # COLOR.JP2 stores raw uint16 DN. Convert mean DN -> I/F via the
            # COLOR.LBL's per-image scaling: I/F = DN * scaling_factor + offset.
            # Without this, cross-image pooling (Stage 7d) is meaningless because
            # scaling_factor varies ~5x across the cohort.
            ir_iof = means["IR"] * lbl.scaling_factor + lbl.offset
            red_iof = means["RED"] * lbl.scaling_factor + lbl.offset
            bg_iof = means["BG"] * lbl.scaling_factor + lbl.offset
            # Lambertian per band (multiplicative scalar 1/cos(i)) on top of I/F.
            ir = ir_iof / cos_i
            red = red_iof / cos_i
            bg = bg_iof / cos_i
            records.append({
                "obs_id": obs_id,
                "scale_idx": int(scale_idx),
                "ti": int(row.ti),
                "tj": int(row.tj),
                "n_color_pixels": int(means["n_pixels"]),
                "IR_iof": float(ir),
                "RED_iof": float(red),
                "BG_iof": float(bg),
                "IR_over_RED": float(ir / red) if red > 0 else float("nan"),
                "IR_over_BG": float(ir / bg) if bg > 0 else float("nan"),
                "dust_index_RED_over_BG": float(red / bg) if bg > 0 else float("nan"),
                "cos_incidence": float(cos_i),
            })
    dt = time.time() - t0
    print(f"  kept {len(records)}/{len(tiles)} tiles "
          f"(outside-swath {n_outside}, pixel-skip {n_pixel_skip}) in {dt:.1f} s")

    if not records:
        return empty
    return pd.DataFrame(records, columns=out_cols)


def _dtype_for(col: str) -> str:
    if col == "obs_id":
        return "object"
    if col in ("scale_idx", "ti", "tj", "n_color_pixels"):
        return "int64"
    return "float64"


# ----------------------------------------------------------------------- driver
def list_available_obs_ids() -> list[str]:
    """Return v2 ObsIds that have a cached COLOR.JP2 according to the coverage parquet."""
    if not COVERAGE_PARQUET.exists():
        raise FileNotFoundError(
            f"{COVERAGE_PARQUET} missing -- run scripts/run_stage7a_audit.py first"
        )
    cov = pd.read_parquet(COVERAGE_PARQUET)
    return cov[cov["has_color"]]["obs_id"].tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", nargs="*", default=None,
                    help="Process only these obs_ids (sanity-run subset).")
    ap.add_argument("--scale-idx", type=int, default=DEFAULT_SCALE_IDX,
                    help=f"Tile scale to extract (default {DEFAULT_SCALE_IDX} = S=64).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output parquet path (default {DEFAULT_OUT}).")
    ap.add_argument("--min-tile-pixels", type=int, default=MIN_TILE_PIXELS,
                    help=f"Min valid colour pixels per tile (default {MIN_TILE_PIXELS}).")
    args = ap.parse_args()

    if args.only:
        obs_ids = list(args.only)
        print(f"Sanity-subset run: {len(obs_ids)} images -- {obs_ids}")
    else:
        obs_ids = list_available_obs_ids()
        print(f"Full-cohort run: {len(obs_ids)} colour-covered images")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    per_image: list[pd.DataFrame] = []
    t_total = time.time()
    for i, obs_id in enumerate(obs_ids, 1):
        print(f"\n[{i}/{len(obs_ids)}] {obs_id}")
        df = process_image(
            obs_id,
            cache_dir=CACHE,
            labels_dir=LABELS_DIR,
            scale_idx=args.scale_idx,
            min_tile_pixels=args.min_tile_pixels,
        )
        if not df.empty:
            per_image.append(df)

    if not per_image:
        print("\nNo tiles produced for any image; output not written.")
        return 1

    all_df = pd.concat(per_image, ignore_index=True)
    all_df.to_parquet(args.out)
    dt = time.time() - t_total
    print(f"\nWrote {len(all_df):,} rows ({all_df['obs_id'].nunique()} images) "
          f"-> {args.out} in {dt/60:.1f} min")
    print(f"  IR  I/F: median={all_df['IR_iof'].median():.4f}  "
          f"p5={all_df['IR_iof'].quantile(0.05):.4f}  "
          f"p95={all_df['IR_iof'].quantile(0.95):.4f}")
    print(f"  RED I/F: median={all_df['RED_iof'].median():.4f}  "
          f"p5={all_df['RED_iof'].quantile(0.05):.4f}  "
          f"p95={all_df['RED_iof'].quantile(0.95):.4f}")
    print(f"  BG  I/F: median={all_df['BG_iof'].median():.4f}  "
          f"p5={all_df['BG_iof'].quantile(0.05):.4f}  "
          f"p95={all_df['BG_iof'].quantile(0.95):.4f}")
    print(f"  dust_index (RED/BG): median={all_df['dust_index_RED_over_BG'].median():.4f}  "
          f"p5={all_df['dust_index_RED_over_BG'].quantile(0.05):.4f}  "
          f"p95={all_df['dust_index_RED_over_BG'].quantile(0.95):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
