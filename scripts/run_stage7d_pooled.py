"""Stage 7d -- pooled cross-image boulder-rich vs boulder-poor colour test.

Per [`PLAN_Compositional.md`](../PLAN_Compositional.md) §4.2 + §4.3 + §5.2.
Joins `dataset_v2/features_colour.parquet` (Stage 7c output) to per-image
`dataset_v2/labels/{ObsId}.parquet` at S=64, runs:

  * pooled MW + Cohen's d under raw / per-image-standardised / partial-dust
    transforms (per `partition_rule` ∈ {P4_area, P2_count});
  * per-image MW + Cohen's d on the raw features (per-image heterogeneity check);
  * Spearman rho between each colour feature and `boulder_count` (continuous
    target check from §4.3) -- pooled standardised, pooled partial-dust, per-image.

Output: `dataset_v2/stage7d_pooled.parquet`.

Run via:
    conda run --no-capture-output -n geospatial python -u scripts/run_stage7d_pooled.py
Optional flags:
    --features PATH       : override input features parquet
    --labels-dir PATH     : override input labels directory
    --out PATH            : override output parquet path
    --min-per-class N     : per-image min rich AND min poor tiles (default 5)

Typical runtime: ~10-30 s on the v2 cohort (9 860 rows / 36 images).
"""
from __future__ import annotations

import argparse
import functools
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import stage7d_pooled as s7d  # noqa: E402

print = functools.partial(print, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path,
                        default=REPO_ROOT / "dataset_v2" / "features_colour.parquet")
    parser.add_argument("--labels-dir", type=Path,
                        default=REPO_ROOT / "dataset_v2" / "labels")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "dataset_v2" / "stage7d_pooled.parquet")
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--scale-idx", type=int, default=s7d.SCALE_IDX_S64)
    args = parser.parse_args()

    t0 = time.time()
    print(f"[stage7d] features = {args.features}")
    print(f"[stage7d] labels-dir = {args.labels_dir}")
    print(f"[stage7d] out      = {args.out}")
    print(f"[stage7d] min-per-class = {args.min_per_class}, scale_idx = {args.scale_idx}")

    df = s7d.load_joined(args.features, args.labels_dir, scale_idx=args.scale_idx)
    print(f"[stage7d] joined rows: {len(df)} across {df['obs_id'].nunique()} images")

    df = s7d.add_partitions(df)
    n_rich_p4 = int(df["is_rich_P4"].sum())
    n_rich_p2 = int(df["is_rich_P2"].sum())
    print(f"[stage7d] P4 rich (fa>=1e-2): {n_rich_p4} / {len(df)} "
          f"({100*n_rich_p4/len(df):.1f}%)")
    print(f"[stage7d] P2 rich (bc>50)   : {n_rich_p2} / {len(df)} "
          f"({100*n_rich_p2/len(df):.1f}%)")

    for rule in ("P4_area", "P2_count"):
        keep = s7d.eligible_images(df, rule, min_per_class=args.min_per_class)
        print(f"[stage7d] eligible images for {rule}: {len(keep)} / "
              f"{df['obs_id'].nunique()}")

    print("[stage7d] running pooled + per-image binary tests + Spearman ...")
    results = s7d.run_all(df, min_per_class=args.min_per_class)
    print(f"[stage7d] result rows: {len(results)}")

    print(f"[stage7d] writing {args.out} ...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.out, index=False)

    # Headline print: pooled standardised, P4 partition, sorted by |effect_size|
    pooled_p4 = results[(results.level == "pooled")
                       & (results.partition_rule == "P4_area")
                       & (results.test_type == "mann_whitney_standardised")]
    print("\n[stage7d] HEADLINE -- pooled standardised, P4 partition:")
    cols = ["feature", "n_rich", "n_poor", "effect_size", "p_value"]
    print(pooled_p4[cols].to_string(index=False))

    pooled_p4_dust = results[(results.level == "pooled")
                            & (results.partition_rule == "P4_area")
                            & (results.test_type == "mann_whitney_partial_dust")]
    print("\n[stage7d] DUST DISCRIMINATOR -- pooled, P4 partition, residualised on dust:")
    print(pooled_p4_dust[cols].to_string(index=False))

    print(f"\n[stage7d] DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
