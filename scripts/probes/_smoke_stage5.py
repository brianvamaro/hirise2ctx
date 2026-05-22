"""One-off smoke test: build both split schemes against the real Stage 4 caches."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import manifest as M
from src.config import load_config
from src.dataset import (
    FEATURES_SUBDIR, LABELS_SUBDIR, build_image_inventory, build_split,
    discover_obs_ids, package_split, write_split_metadata,
)


def main() -> int:
    cfg = load_config("config.yaml")
    labels_dir = cfg.output_dir / LABELS_SUBDIR
    features_dir = cfg.output_dir / FEATURES_SUBDIR
    manifest = M.load_manifest(cfg.manifest_path)

    obs_ids = discover_obs_ids(labels_dir)
    print(f"Discovered {len(obs_ids)} ObsIds with Stage 4 outputs: {obs_ids}")

    inv = build_image_inventory(obs_ids, manifest, labels_dir)
    print("\nPer-image inventory:")
    print(inv.to_string())

    for scheme_name, scheme in cfg["splits"]["schemes"].items():
        print(f"\n=== Building scheme {scheme_name} (n_folds={scheme['n_folds']}, "
              f"stratification={scheme['stratification']}) ===")
        t0 = time.monotonic()
        meta = build_split(
            name=scheme_name,
            n_folds=int(scheme["n_folds"]),
            stratification=str(scheme["stratification"]),
            seed=int(scheme["seed"]),
            inventory=inv,
            config_hash=cfg.hash,
        )
        path = write_split_metadata(meta, cfg.output_dir)
        print(f"wrote {path}  ({time.monotonic()-t0:.2f}s)")
        for fold in meta["folds"]:
            ts = fold["test_summary"]
            labels_str = ", ".join(f"{k}={v}" for k, v in sorted(ts["boulder_labels"].items()))
            print(f"  fold {fold['fold_idx']}: test={fold['test_obs_ids']} "
                  f"({ts['n_images']} imgs, {ts['n_tiles_total']:,} tiles, {labels_str})")

        # Sanity: no leakage.
        for fold in meta["folds"]:
            shared = set(fold["test_obs_ids"]) & set(fold["train_obs_ids"])
            assert not shared, f"leak in fold {fold['fold_idx']}: {shared}"

        # Package.
        t1 = time.monotonic()
        pkg_meta = package_split(
            meta, labels_dir=labels_dir, features_dir=features_dir,
            output_dir=cfg.output_dir,
            scale_filter=cfg["splits"].get("scale_filter"),
            emit_all_parquet=bool(cfg["splits"]["emit_all_parquet"]),
            config_hash=cfg.hash,
        )
        print(f"  packaged in {time.monotonic()-t1:.1f}s")
        print(f"  X cols={pkg_meta['per_fold'][0]['n_train_x_cols']}  "
              f"y cols={pkg_meta['per_fold'][0]['n_y_cols']}")
        total_train = sum(f["n_train_tiles"] for f in pkg_meta["per_fold"])
        total_test = sum(f["n_test_tiles"] for f in pkg_meta["per_fold"])
        print(f"  total train rows across folds: {total_train:,}  "
              f"total test rows across folds: {total_test:,}")
        if pkg_meta["all_parquet_path"]:
            import os
            size_mb = os.path.getsize(pkg_meta["all_parquet_path"]) / 1e6
            print(f"  all.parquet: {size_mb:.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
