"""Build the 5-image v2 dev harness (PLAN_ModelImprovement.md Phase 0).

Reuses everything already computed for the 5 dev images:
  - writes hirise_5_dev.csv (subset of hirise_40_vclaire.csv),
  - copies the 5 images' dataset_v2/{labels,features}/{obs}.parquet into dataset_v2_dev/,
  - junctions cache_v2_dev -> cache_v2 (so Stage 4/4b dev read the shared per-image caches).

Idempotent: re-running overwrites the dev manifest + parquet copies and leaves the
junction in place. Stage 5 (and, for Phase B/C, Stage 4/4b) then run against config_v2_dev.yaml.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Density-span + v1 overlap (Stage-1 valid-polygon counts), PLAN_ModelImprovement.md Phase 0.
DEV_OBS = [
    "ESP_055978_2270",  # 9.6k  sparsest
    "ESP_069669_2220",  # 35k   v1-overlap, low-mid
    "ESP_064510_2260",  # 81k   mid
    "ESP_071093_2210",  # 245k  v1-overlap, high-mid
    "ESP_068483_2280",  # 727k  densest
]

SRC_MANIFEST = REPO_ROOT / "hirise_40_vclaire.csv"
DEV_MANIFEST = REPO_ROOT / "hirise_5_dev.csv"
SRC_DATASET = REPO_ROOT / "dataset_v2"
DEV_DATASET = REPO_ROOT / "dataset_v2_dev"
SRC_CACHE = REPO_ROOT / "cache_v2"
DEV_CACHE = REPO_ROOT / "cache_v2_dev"


def main() -> int:
    # 1. dev manifest
    man = pd.read_csv(SRC_MANIFEST)
    dev = man[man["ObsId"].isin(DEV_OBS)].reset_index(drop=True)
    missing = set(DEV_OBS) - set(dev["ObsId"])
    if missing:
        print(f"ERROR: dev ObsIds missing from manifest: {sorted(missing)}")
        return 1
    dev.to_csv(DEV_MANIFEST, index=False)
    print(f"wrote {DEV_MANIFEST.name}  ({len(dev)} rows)")

    # 2. copy the 5 images' label + feature parquets AND their .json sidecars
    #    (build_image_inventory / packaging read the per-image sidecar provenance).
    for sub in ("labels", "features"):
        dst = DEV_DATASET / sub
        dst.mkdir(parents=True, exist_ok=True)
        for obs in DEV_OBS:
            src_p = SRC_DATASET / sub / f"{obs}.parquet"
            if not src_p.exists():
                print(f"ERROR: missing {src_p}")
                return 1
            shutil.copy2(src_p, dst / f"{obs}.parquet")
            src_j = SRC_DATASET / sub / f"{obs}.json"
            if src_j.exists():
                shutil.copy2(src_j, dst / f"{obs}.json")
        print(f"copied {len(DEV_OBS)} {sub} parquet+json -> {dst.relative_to(REPO_ROOT)}")

    # 3. cache_v2_dev -> cache_v2 junction (no admin needed). Stage 4/4b dev read it.
    if not DEV_CACHE.exists():
        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(DEV_CACHE), str(SRC_CACHE)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"WARN: junction not created ({r.stderr.strip()}); "
                  f"only needed for Phase B/C Stage 4/4b.")
        else:
            print(f"junction {DEV_CACHE.name} -> {SRC_CACHE.name}")
    else:
        print(f"junction {DEV_CACHE.name} already exists")

    print("\nDev harness ready. Next: run_stage5.py --all --config config_v2_dev.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
