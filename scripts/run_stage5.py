"""Run Stage 5 (split construction + dataset packaging) for one or all named schemes.

Usage:
    # Build + package all schemes from config.yaml:
    conda run -n geospatial python scripts/run_stage5.py --all

    # Build + package a single named scheme:
    conda run -n geospatial python scripts/run_stage5.py loio_9fold

    # Just build the split metadata, don't materialise per-fold parquets:
    conda run -n geospatial python scripts/run_stage5.py --all --no-package

Stage 5 reads `dataset/labels/{ObsId}.parquet` + `dataset/features/{ObsId}.parquet` and
writes:

    dataset/splits/{name}.json                       # split metadata + per-fold summary
    dataset/packaged/{name}/X_train_fold{k}.parquet  # features only
    dataset/packaged/{name}/y_train_fold{k}.parquet  # labels only
    dataset/packaged/{name}/X_test_fold{k}.parquet
    dataset/packaged/{name}/y_test_fold{k}.parquet
    dataset/packaged/{name}/groups_train_fold{k}.npy # obs_id-as-int
    dataset/packaged/{name}/groups_test_fold{k}.npy
    dataset/packaged/{name}/all.parquet              # consolidated view (if emit_all_parquet=true)
    dataset/packaged/{name}/metadata.json            # packaging provenance
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import manifest as M
from src.config import load_config
from src.dataset import (
    FEATURES_SUBDIR, LABELS_SUBDIR, build_image_inventory, build_split,
    discover_obs_ids, package_split, write_split_metadata,
)


def _run_one(cfg, scheme_name: str, scheme: dict, inventory, do_package: bool) -> dict | None:
    labels_dir = cfg.output_dir / LABELS_SUBDIR
    features_dir = cfg.output_dir / FEATURES_SUBDIR
    t0 = time.monotonic()
    # within_image schemes need extra kwargs sourced from the per-scheme config block.
    extra_kwargs: dict = {}
    if str(scheme["stratification"]) == "within_image":
        extra_kwargs = {
            "labels_dir": labels_dir,
            "n_folds_per_image": int(scheme.get("n_folds_per_image", 4)),
            "buffer_tiles": int(scheme.get("buffer_tiles", 0)),
            "excluded_obs_ids": list(scheme.get("excluded_obs_ids", []) or []),
        }
    try:
        meta = build_split(
            name=scheme_name,
            n_folds=int(scheme["n_folds"]),
            stratification=str(scheme["stratification"]),
            seed=int(scheme["seed"]),
            inventory=inventory,
            config_hash=cfg.hash,
            **extra_kwargs,
        )
        path = write_split_metadata(meta, cfg.output_dir)
    except ValueError as e:
        print(f"  {scheme_name}: FAILED to build ({e})", flush=True)
        return None
    print(f"  {scheme_name}: built {len(meta['folds'])} folds in {time.monotonic()-t0:.2f}s "
          f"-> {path.relative_to(cfg.output_dir.parent)}", flush=True)
    for fold in meta["folds"]:
        ts = fold["test_summary"]
        labels_str = ", ".join(f"{k}={v}" for k, v in sorted(ts["boulder_labels"].items()))
        print(f"    fold {fold['fold_idx']}: test={fold['test_obs_ids']}  "
              f"({ts['n_images']} img, {ts['n_tiles_total']:,} tiles, {labels_str})",
              flush=True)
    if not do_package:
        return meta
    t1 = time.monotonic()
    pkg = package_split(
        meta, labels_dir=labels_dir, features_dir=features_dir,
        output_dir=cfg.output_dir,
        scale_filter=cfg["splits"].get("scale_filter"),
        emit_all_parquet=bool(cfg["splits"]["emit_all_parquet"]),
        config_hash=cfg.hash,
    )
    total_train = sum(f["n_train_tiles"] for f in pkg["per_fold"])
    total_test = sum(f["n_test_tiles"] for f in pkg["per_fold"])
    print(f"  {scheme_name}: packaged in {time.monotonic()-t1:.1f}s  "
          f"train_rows_sum={total_train:,}  test_rows_sum={total_test:,}  "
          f"X_cols={pkg['per_fold'][0]['n_train_x_cols']}  "
          f"y_cols={pkg['per_fold'][0]['n_y_cols']}",
          flush=True)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 split + packaging driver")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("scheme", nargs="?", default=None, help="Named scheme from config.yaml splits.schemes")
    g.add_argument("--all", action="store_true", help="Run every named scheme")
    parser.add_argument("--no-package", action="store_true",
                        help="Only build split metadata; skip per-fold parquet materialisation")
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    if "splits" not in cfg.raw:
        print("config.yaml has no `splits:` block; nothing to do.", flush=True)
        return 2
    manifest = M.load_manifest(cfg.manifest_path)
    labels_dir = cfg.output_dir / LABELS_SUBDIR
    obs_ids = discover_obs_ids(labels_dir)
    if not obs_ids:
        print(f"No Stage 4 outputs found in {labels_dir}; run scripts/run_stage4.py first.",
              flush=True)
        return 1
    print(f"Stage 5 :: {len(obs_ids)} ObsIds with Stage 4 outputs", flush=True)
    inv = build_image_inventory(obs_ids, manifest, labels_dir)
    print(f"  labels:    {labels_dir}", flush=True)
    print(f"  features:  {cfg.output_dir / FEATURES_SUBDIR}", flush=True)
    print(f"  emit_all:  {cfg['splits']['emit_all_parquet']}", flush=True)
    print(f"  scale_filter: {cfg['splits'].get('scale_filter')}", flush=True)

    schemes = cfg["splits"]["schemes"]
    if args.all:
        targets = list(schemes.items())
    else:
        if args.scheme not in schemes:
            print(f"scheme {args.scheme!r} not in config (have {sorted(schemes)})", flush=True)
            return 2
        targets = [(args.scheme, schemes[args.scheme])]

    for scheme_name, scheme in targets:
        print(f"\n=== {scheme_name} ===", flush=True)
        _run_one(cfg, scheme_name, scheme, inv, do_package=not args.no_package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
