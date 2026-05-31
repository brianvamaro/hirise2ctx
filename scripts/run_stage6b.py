"""Run Stage 6b (CTX-source illumination features) for one ObsId or all.

Reads existing Stage 4b feature parquets, augments them with per-tile CTX-source
illumination columns (area-weighted aggregates of contributing CTX images'
INCIDENCE / EMISSION / PHASE / SUB_SOLAR_AZIMUTH angles + source-diversity stats),
and writes the augmented parquets to a parallel ``features_ctx_illum/`` directory.
The original Stage 4b cache is NOT modified.

Per-tile data flow:
  Stage 4b parquet (obs_id, scale_idx, tile_size_px, ti, tj, ...features)
  + Murray SeamMap.shp for the ObsId's CTX tile (cached in ctx_tiles/{tile}.zip)
  + cached CTX window TIFF (for window_transform, window_h, window_w)
  + Murray tile sidecar JSON (for mosaic transform -> mosaic_row_origin/col_origin)
  -> augmented parquet with 7 extra columns (see src.ctx_source_illumination.OUTPUT_COLUMNS)

Usage:
    # Single image:
    conda run -n geospatial python scripts/run_stage6b.py ESP_069669_2220 \
        --dataset-dir dataset_v2_dev --config config_v2_dev.yaml

    # All Stage-4b-ready ObsIds in a dataset:
    conda run -n geospatial python scripts/run_stage6b.py --all \
        --dataset-dir dataset_v2_dev --config config_v2_dev.yaml

Outputs:
    dataset_v2_dev/features_ctx_illum/{ObsId}.parquet
    dataset_v2_dev/features_ctx_illum/{ObsId}.json     # provenance

Stage 6b spec: PROMOTION_QUEUE.md "Stage 6b -- CTX-source illumination angles".
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src import manifest as M  # noqa: E402
from src.config import load_config  # noqa: E402
from src.ctx_retrieve import CTX_WINDOWS_SUBDIR  # noqa: E402
from src.ctx_source_illumination import (  # noqa: E402
    OUTPUT_COLUMNS,
    add_ctx_source_illumination_features,
    load_seam_map,
    load_window_metadata,
    mosaic_origin_pixels,
)
from src.ctx_tiles import murray_tile_for_manifest_row  # noqa: E402
from src.features import EXCLUDED_FROM_SWEEP, FEATURES_SUBDIR  # noqa: E402

CTX_ILLUM_SUBDIR = "features_ctx_illum"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _load_mosaic_transform(cache_dir: Path, murray_tile: str) -> list[float]:
    sidecar = cache_dir / "ctx_tiles" / f"{murray_tile}.json"
    if not sidecar.exists():
        raise FileNotFoundError(f"missing Murray tile sidecar {sidecar}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    return list(meta["inner_transform"])


def _augment_one(
    obs_id: str,
    *,
    dataset_dir: Path,
    cache_dir: Path,
    manifest_row,
    output_subdir: str = CTX_ILLUM_SUBDIR,
) -> dict | None:
    in_parquet = dataset_dir / FEATURES_SUBDIR / f"{obs_id}.parquet"
    if not in_parquet.exists():
        print(f"  {obs_id}: SKIP (Stage 4b output missing at {in_parquet})", flush=True)
        return None
    ctx_tif = cache_dir / CTX_WINDOWS_SUBDIR / f"{obs_id}.tif"
    if not ctx_tif.exists():
        print(f"  {obs_id}: SKIP (CTX window TIFF missing at {ctx_tif})", flush=True)
        return None
    murray_tile = murray_tile_for_manifest_row(manifest_row)

    out_dir = dataset_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_parquet = out_dir / f"{obs_id}.parquet"
    out_sidecar = out_dir / f"{obs_id}.json"

    t0 = time.monotonic()
    df = pd.read_parquet(in_parquet)
    seam_gdf = load_seam_map(murray_tile, cache_dir)
    window_meta = load_window_metadata(ctx_tif)
    mosaic_transform = _load_mosaic_transform(cache_dir, murray_tile)
    mosaic_row_origin, mosaic_col_origin = mosaic_origin_pixels(
        window_meta["window_transform"], mosaic_transform,
    )

    augmented = add_ctx_source_illumination_features(
        df,
        seam_gdf=seam_gdf,
        window_transform=window_meta["window_transform"],
        window_h=window_meta["window_h"],
        window_w=window_meta["window_w"],
        mosaic_row_origin=mosaic_row_origin,
        mosaic_col_origin=mosaic_col_origin,
    )
    augmented.to_parquet(out_parquet, index=False)
    dt = time.monotonic() - t0

    new_cols = [c for c in augmented.columns if c.startswith("ctx_")]
    per_scale = {
        int(S): int((df["tile_size_px"] == S).sum())
        for S in sorted({int(s) for s in df["tile_size_px"]})
    }
    finite_per_col = {
        c: int(augmented[c].notna().sum()) for c in OUTPUT_COLUMNS if c in augmented
    }
    provenance = {
        "obs_id": obs_id,
        "murray_tile": murray_tile,
        "source_features_parquet": str(in_parquet),
        "source_sha256_short": _file_sha256(in_parquet),
        "ctx_window_tif": str(ctx_tif),
        "window_h": window_meta["window_h"],
        "window_w": window_meta["window_w"],
        "mosaic_row_origin": mosaic_row_origin,
        "mosaic_col_origin": mosaic_col_origin,
        "stage6b": {
            "n_seam_sources_total": int(len(seam_gdf)),
            "n_seam_sources_window": int(
                len(seam_gdf[seam_gdf.intersects(_bbox_from_window(window_meta))])
            ),
            "n_new_columns": len(new_cols),
            "new_columns": list(OUTPUT_COLUMNS),
            "finite_counts_per_column": finite_per_col,
        },
        "n_tiles_total": int(len(augmented)),
        "per_scale_tile_counts": per_scale,
        "elapsed_seconds": round(dt, 3),
        "written_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "parquet_path": str(out_parquet),
    }
    out_sidecar.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(
        f"  {obs_id}: tile={murray_tile:>8s}  n_tiles={len(augmented):6d}  "
        f"n_sources_window={provenance['stage6b']['n_seam_sources_window']:3d}  "
        f"new_cols={len(new_cols)}  elapsed={dt:.2f}s",
        flush=True,
    )
    return provenance


def _bbox_from_window(window_meta: dict):
    from shapely.geometry import box
    t = window_meta["window_transform"]
    xmin = t.c
    ymax = t.f
    xmax = xmin + window_meta["window_w"] * t.a
    ymin = ymax + window_meta["window_h"] * t.e
    return box(xmin, ymin, xmax, ymax)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 6b CTX-source-illumination driver")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("obs_id", nargs="?", default=None, help="HiRISE Observation ID")
    g.add_argument("--all", action="store_true",
                   help="Process every ObsId with a Stage 4b parquet in --dataset-dir")
    parser.add_argument(
        "--dataset-dir", default="dataset_v2_dev",
        help="Dataset root (default: dataset_v2_dev)",
    )
    parser.add_argument(
        "--config", default="config_v2_dev.yaml",
        help="Pipeline config YAML (drives manifest + cache_dir)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    df_manifest = M.load_manifest(cfg.manifest_path)
    dataset_dir = Path(args.dataset_dir).resolve()
    cache_dir = cfg.cache_dir

    if not (dataset_dir / FEATURES_SUBDIR).exists():
        print(f"No Stage 4b output dir at {dataset_dir / FEATURES_SUBDIR}", flush=True)
        return 1

    df_manifest_indexed = df_manifest.set_index("ObsId")

    if args.all:
        obs_ids = sorted(
            p.stem for p in (dataset_dir / FEATURES_SUBDIR).glob("*.parquet")
            if p.stem not in EXCLUDED_FROM_SWEEP
        )
        print(
            f"Stage 6b :: {len(obs_ids)} ObsIds in {dataset_dir / FEATURES_SUBDIR} "
            f"(excluding {sorted(EXCLUDED_FROM_SWEEP)})",
            flush=True,
        )
        t_all = time.monotonic()
        results = []
        for obs in obs_ids:
            if obs not in df_manifest_indexed.index:
                print(f"  {obs}: SKIP (not in manifest {cfg.manifest_path.name})", flush=True)
                results.append((obs, None))
                continue
            row = df_manifest_indexed.loc[obs]
            results.append(
                (obs, _augment_one(
                    obs, dataset_dir=dataset_dir, cache_dir=cache_dir,
                    manifest_row=row,
                )),
            )
        dt_all = time.monotonic() - t_all
        solved = [(o, p) for o, p in results if p is not None]
        skipped = [o for o, p in results if p is None]
        print(f"\nSolved {len(solved)} / {len(obs_ids)} in {dt_all:.1f}s", flush=True)
        if skipped:
            print(f"  Skipped: {', '.join(skipped)}", flush=True)
        return 0

    obs = args.obs_id
    if obs not in df_manifest_indexed.index:
        print(f"ObsId {obs!r} not in manifest {cfg.manifest_path.name}", flush=True)
        return 2
    row = df_manifest_indexed.loc[obs]
    in_dir = dataset_dir / FEATURES_SUBDIR
    if not (in_dir / f"{obs}.parquet").exists():
        print(f"{obs}: no Stage 4b parquet at {in_dir / f'{obs}.parquet'}", flush=True)
        return 2
    print(f"Stage 6b :: {obs}", flush=True)
    prov = _augment_one(
        obs, dataset_dir=dataset_dir, cache_dir=cache_dir, manifest_row=row,
    )
    if prov is None:
        return 1
    print(f"  wrote {prov['parquet_path']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
