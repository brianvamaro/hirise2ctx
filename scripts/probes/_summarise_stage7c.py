"""Summarise the Stage 7c full-cohort output -- numbers for DECISIONS.md entry.

Reads `dataset_v2/features_colour.parquet` and reports:
  - rows + image count + per-image tile retention rate
  - per-image I/F band ranges (sanity-check Lambertian correction works)
  - cross-image dust_index distribution
  - which v2 ObsIds got 0 rows (e.g. no Stage 1 sidecar, no labels parquet,
    everything off-swath) -- these are colour-eligible per coverage.parquet but
    failed Stage 7c for one of those reasons.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path("cache_v2")
LABELS_DIR = Path("dataset_v2/labels")
FEATURES = Path("dataset_v2/features_colour.parquet")


def main() -> int:
    if not FEATURES.exists():
        print(f"{FEATURES} not found")
        return 1
    fc = pd.read_parquet(FEATURES)
    print(f"\nStage 7c features: {len(fc):,} rows across {fc['obs_id'].nunique()} images")
    print(f"  scale_idx unique: {sorted(fc['scale_idx'].unique())}")
    print(f"  cohort I/F medians: IR={fc.IR_iof.median():.4f}  "
          f"RED={fc.RED_iof.median():.4f}  BG={fc.BG_iof.median():.4f}")
    print(f"  cohort dust_index (RED/BG): "
          f"p5={fc.dust_index_RED_over_BG.quantile(0.05):.3f}  "
          f"p50={fc.dust_index_RED_over_BG.median():.3f}  "
          f"p95={fc.dust_index_RED_over_BG.quantile(0.95):.3f}")
    print(f"  per-image tile counts: min={fc.groupby('obs_id').size().min()}  "
          f"median={int(fc.groupby('obs_id').size().median())}  "
          f"max={fc.groupby('obs_id').size().max()}")

    # Retention rate per image: kept / total S=64 tiles in labels parquet.
    print(f"\nPer-image retention (kept / total S=64 tiles):")
    rows = []
    for obs_id, sub in fc.groupby("obs_id"):
        lbl_path = LABELS_DIR / f"{obs_id}.parquet"
        if not lbl_path.exists():
            continue
        lbl = pd.read_parquet(lbl_path)
        total = (lbl.scale_idx == 3).sum()
        kept = len(sub)
        rows.append((obs_id, kept, total, kept / total if total else float("nan")))
    rows.sort(key=lambda r: -r[3])
    print(f"  best 3 (highest swath coverage):")
    for r in rows[:3]:
        print(f"    {r[0]}: {r[1]:>4} / {r[2]:>5} = {r[3]:.1%}")
    print(f"  worst 3:")
    for r in rows[-3:]:
        print(f"    {r[0]}: {r[1]:>4} / {r[2]:>5} = {r[3]:.1%}")

    # Cross-image cos_incidence range (sanity for Lambertian).
    per_img = fc.groupby("obs_id").agg(
        cos_i=("cos_incidence", "first"),
        IR_med=("IR_iof", "median"),
        RED_med=("RED_iof", "median"),
        BG_med=("BG_iof", "median"),
        dust_med=("dust_index_RED_over_BG", "median"),
    )
    print(f"\nCross-image cos(i) range: "
          f"{per_img.cos_i.min():.3f} - {per_img.cos_i.max():.3f}")
    print(f"Per-image IR_iof median range: "
          f"{per_img.IR_med.min():.3f} - {per_img.IR_med.max():.3f}")
    print(f"Per-image dust_index median range: "
          f"{per_img.dust_med.min():.3f} - {per_img.dust_med.max():.3f}")

    # Coverage parquet sanity: are all colour-covered images represented?
    cov = pd.read_parquet(CACHE / "hirise_color" / "coverage.parquet")
    available = set(cov[cov["has_color"]]["obs_id"])
    represented = set(fc["obs_id"].unique())
    missing = sorted(available - represented)
    print(f"\nColour-eligible images with 0 Stage 7c rows: "
          f"{len(missing)} / {len(available)}")
    if missing:
        for obs_id in missing:
            print(f"  - {obs_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
