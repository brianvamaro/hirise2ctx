"""Measure the deployed abundance layer's size-floor basis (R03 / R83 / R84, leg 4).

`fractional_area` is the area share of boulders *large enough to have been detected in that HiRISE
image*, and the qualifier varies ~3.6x in area across the cohort. The deployed raster is
quantile-matched onto a pool that mixes those conventions, and nothing on disk recorded it. This
driver measures the basis and banks it as JSON so `scripts/map_region.py` and
`scripts/striping_a1_map.py` can stamp it onto every raster they write (`src.size_floor`).

What it reads, all read-only:
  * `cache_v2/pds_labels/{obs}.LBL`             -> MAP_SCALE, the authoritative pixel scale
  * `cache_v2/reprojected_detections/{obs}.gpkg` -> each image's natural detection floor
  * `dataset_v2/labels/{obs}.parquet`            -> the S=32 pool and its per-image tile counts

Runtime is dominated by the ~7 M detection polygons: ~6-7 min for the 38-image v2 cohort. Nothing
else in the pipeline needs to pay that, which is the point of banking the result.

    conda run --no-capture-output -n geospatial python -u scripts/measure_size_floor.py \
        --out models/deployable/size_floor_basis.json

**Why the pixel scale comes from the `.LBL` and not the manifest.** `MapPixel_mpp` is sourced from
the label spreadsheet by `scripts/build_vclaire_manifest.py`, so the two `LabelSource: none` rows
are blank. Both are 0.5 m/px and always were; the `.LBL`s are already cached. Reading the `.LBL`
makes a blank impossible rather than fixing two blanks.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.size_floor import (DEFAULT_MIN_SIZE_M, SizeFloorBasis, map_scale_from_pds_label,
                            natural_floor_from_detections)

DEFAULT_LABELS = REPO_ROOT / "dataset_v2" / "labels"
DEFAULT_DETECTIONS = REPO_ROOT / "cache_v2" / "reprojected_detections"
DEFAULT_PDS = REPO_ROOT / "cache_v2" / "pds_labels"
DEFAULT_OUT = REPO_ROOT / "models" / "deployable" / "size_floor_basis.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Isolation criterion 4: every root is a flag, so a scratch measurement never touches the
    # live tree -- and `--out` in particular, because the default writes into `models/`.
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--detections", default=str(DEFAULT_DETECTIONS))
    ap.add_argument("--pds-labels", default=str(DEFAULT_PDS))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--tile-px", type=int, default=32, help="the frozen recipe's S")
    ap.add_argument("--min-size-m", type=float, default=DEFAULT_MIN_SIZE_M,
                    help="the Stage-4 global filter actually in force (config detection_filters)")
    ap.add_argument("--dry-run", action="store_true",
                    help="measure and print, write nothing")
    args = ap.parse_args()

    labels, dets, pds = Path(args.labels), Path(args.detections), Path(args.pds_labels)
    obs_ids = sorted(p.stem for p in labels.glob("*.parquet"))
    if not obs_ids:
        raise SystemExit(f"no label parquets under {labels}")
    print(f"=== size-floor basis: {len(obs_ids)} images, S={args.tile_px}, "
          f"min_size_m={args.min_size_m} ===", flush=True)

    t0 = time.monotonic()
    per_image, tile_counts, missing_scale, missing_det = [], {}, [], []
    for obs in obs_ids:
        # pool tiles at the frozen S. `tile_size_px` is the selector: `tile_size_m` is
        # 159.9991835, so an equality test against 160.0 silently matches nothing.
        t = pd.read_parquet(labels / f"{obs}.parquet", columns=["tile_size_px"])
        tile_counts[obs] = int((t.tile_size_px == args.tile_px).sum())

        mpp = map_scale_from_pds_label(pds / f"{obs}.LBL")
        floor = natural_floor_from_detections(dets / f"{obs}.gpkg")
        if mpp is None:
            missing_scale.append(obs)
        if floor is None:
            missing_det.append(obs)
            continue
        per_image.append({"obs_id": obs, "map_scale_mpp": mpp, **floor})
        print(f"  {obs}  {mpp} m/px  natural {floor['natural_floor_m2']:.3f} m2  "
              f"n_poly {floor['n_polygons']:,}  pool tiles {tile_counts[obs]:,}", flush=True)

    if missing_scale:
        raise SystemExit(
            f"no MAP_SCALE for {len(missing_scale)} image(s): {missing_scale}\n"
            f"  The basis states a mixture BY pixel scale; measuring it with unknown members "
            f"would silently under-count one cohort. Fetch the .LBL(s) first.")
    if missing_det:
        print(f"  ⚠ {len(missing_det)} image(s) have no Stage-1 detections and are excluded "
              f"from the basis: {missing_det}", flush=True)

    basis = SizeFloorBasis.from_records(per_image, tile_counts, tile_px=args.tile_px,
                                        min_size_m=args.min_size_m)
    print(f"\n{basis.summary()}\n")
    print(f"  pool                 {basis.n_tiles:,} tiles / {basis.n_images} images")
    print(f"  effective floor      {basis.floor_min_m2:.4f} - {basis.floor_max_m2:.4f} m2 "
          f"({basis.n_distinct_floors} distinct)")
    print(f"  tile-weighted mean   {basis.floor_tile_weighted_mean_m2:.4f} m2")
    print(f"  tile share by mpp    {basis.tile_share_by_scale}")
    print(f"  image share by mpp   {basis.image_share_by_scale}   <- a DIFFERENT quantity")
    print(f"  measured in {time.monotonic() - t0:.0f}s", flush=True)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    out = basis.to_json(args.out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
