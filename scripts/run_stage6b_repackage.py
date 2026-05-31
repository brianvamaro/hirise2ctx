"""Repackage an existing within_image / LOIO split using Stage 6b augmented features.

Mirrors ``scripts/run_stage6a_repackage.py`` but reads from
``dataset/features_ctx_illum/`` (Stage 6b output) and writes
``dataset/packaged/{scheme}_ctx_illum/``. The fold definitions are byte-identical
to the source scheme by construction; only the X-matrix columns differ.

Usage:
    # Repackage within_image_4fold + loio_nfold on the dev dataset:
    conda run -n geospatial python scripts/run_stage6b_repackage.py \
        within_image_4fold --dataset-dir dataset_v2_dev
    conda run -n geospatial python scripts/run_stage6b_repackage.py \
        loio_nfold --dataset-dir dataset_v2_dev

Outputs:
    dataset_v2_dev/splits/{scheme}_ctx_illum.json
    dataset_v2_dev/packaged/{scheme}_ctx_illum/      # per-fold X/y/groups parquets

This script does NOT re-build splits. Run ``scripts/run_stage5.py`` first if the
split JSON doesn't exist. Stage 6b feature parquets must already exist under
``{dataset_dir}/features_ctx_illum/`` (run ``scripts/run_stage6b.py --all`` first).
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

from src.dataset import (  # noqa: E402
    LABELS_SUBDIR, SPLITS_SUBDIR, load_split_metadata, package_split,
    write_split_metadata,
)

CTX_ILLUM_FEATURES_SUBDIR = "features_ctx_illum"
CTX_ILLUM_SCHEME_SUFFIX = "_ctx_illum"


def _split_hash(meta: dict) -> str:
    keys = (
        "name", "kind", "n_folds", "stratification", "seed", "manifest_obs_ids", "folds",
        "n_folds_per_image", "buffer_tiles", "excluded_obs_ids",
    )
    canonical = json.dumps(
        {k: meta[k] for k in keys if k in meta},
        sort_keys=True, default=str, separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 6b repackaging driver")
    parser.add_argument(
        "scheme", help="Name of an existing split scheme (e.g. loio_nfold)",
    )
    parser.add_argument(
        "--dataset-dir", default="dataset_v2_dev",
        help="Dataset root (default: dataset_v2_dev)",
    )
    parser.add_argument(
        "--scale-filter", nargs="*", type=int, default=None,
        help="Optional subset of tile_size_px values to package (default: all)",
    )
    parser.add_argument(
        "--no-emit-all", action="store_true",
        help="Skip the consolidated all.parquet output (saves disk on dev runs)",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    labels_dir = dataset_dir / LABELS_SUBDIR
    features_dir = dataset_dir / CTX_ILLUM_FEATURES_SUBDIR

    if not labels_dir.exists():
        print(f"No labels dir at {labels_dir}; run Stage 4 first.", flush=True)
        return 1
    if not features_dir.exists() or not any(features_dir.glob("*.parquet")):
        print(
            f"No augmented features at {features_dir}; run "
            f"scripts/run_stage6b.py --all --dataset-dir {args.dataset_dir} first.",
            flush=True,
        )
        return 1

    try:
        meta = load_split_metadata(args.scheme, dataset_dir)
    except FileNotFoundError:
        print(
            f"No split metadata at {dataset_dir / SPLITS_SUBDIR / f'{args.scheme}.json'}; "
            f"run scripts/run_stage5.py {args.scheme} first.",
            flush=True,
        )
        return 1

    new_name = f"{args.scheme}{CTX_ILLUM_SCHEME_SUFFIX}"
    new_meta = dict(meta)
    new_meta["name"] = new_name
    new_meta.pop("split_hash", None)
    new_meta.pop("written_at_iso", None)
    new_meta["split_hash"] = _split_hash(new_meta)
    new_meta["written_at_iso"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    new_meta["repackaged_from_scheme"] = args.scheme
    new_meta["features_subdir"] = CTX_ILLUM_FEATURES_SUBDIR

    new_split_path = write_split_metadata(new_meta, dataset_dir)
    print(f"Stage 6b repackage :: {args.scheme} -> {new_name}", flush=True)
    print(f"  split JSON: {new_split_path}", flush=True)
    print(f"  features:   {features_dir}", flush=True)
    print(f"  labels:     {labels_dir}", flush=True)
    print(f"  scale_filter: {args.scale_filter}", flush=True)

    t0 = time.monotonic()
    pkg = package_split(
        new_meta,
        labels_dir=labels_dir,
        features_dir=features_dir,
        output_dir=dataset_dir,
        scale_filter=args.scale_filter,
        emit_all_parquet=not args.no_emit_all,
        config_hash=new_meta.get("config_hash", ""),
    )
    dt = time.monotonic() - t0
    total_train = sum(f["n_train_tiles"] for f in pkg["per_fold"])
    total_test = sum(f["n_test_tiles"] for f in pkg["per_fold"])
    print(
        f"  packaged in {dt:.1f}s  train_rows_sum={total_train:,}  test_rows_sum={total_test:,}  "
        f"X_cols={pkg['per_fold'][0]['n_train_x_cols']}  y_cols={pkg['per_fold'][0]['n_y_cols']}",
        flush=True,
    )
    print(f"  output: {dataset_dir / 'packaged' / new_name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
