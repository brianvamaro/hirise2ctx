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
    parser.add_argument("--ctx-features-dir", type=Path,
                        default=REPO_ROOT / "dataset_v2" / "features",
                        help="Stage 4b per-image feature parquets (for shadow_fraction).")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "dataset_v2" / "stage7d_pooled.parquet")
    parser.add_argument("--attribution-out", type=Path,
                        default=REPO_ROOT / "dataset_v2" / "stage7d_per_image_attribution.parquet")
    parser.add_argument("--min-per-class", type=int, default=5)
    parser.add_argument("--scale-idx", type=int, default=s7d.SCALE_IDX_S64)
    parser.add_argument("--shadow-threshold", type=float, default=None,
                        help="If set, drop tiles where shadow_fraction > T (e.g. 0.10).")
    parser.add_argument("--no-attribution", action="store_true",
                        help="Skip the per-image attribution table emission.")
    args = parser.parse_args()

    t0 = time.time()
    print(f"[stage7d] features         = {args.features}")
    print(f"[stage7d] labels-dir       = {args.labels_dir}")
    print(f"[stage7d] ctx-features-dir = {args.ctx_features_dir}")
    print(f"[stage7d] out              = {args.out}")
    print(f"[stage7d] attribution-out  = {args.attribution_out}")
    print(f"[stage7d] min-per-class    = {args.min_per_class}, scale_idx = {args.scale_idx}")
    print(f"[stage7d] shadow-threshold = {args.shadow_threshold}")

    df = s7d.load_joined(args.features, args.labels_dir, scale_idx=args.scale_idx)
    print(f"[stage7d] joined rows: {len(df)} across {df['obs_id'].nunique()} images")

    df = s7d.attach_shadow_fraction(df, args.ctx_features_dir, scale_idx=args.scale_idx)
    print(f"[stage7d] after shadow-fraction join: {len(df)} rows "
          f"(shadow_fraction median = {df['shadow_fraction'].median():.4f})")

    n_before = len(df)
    df = s7d.filter_shadow(df, args.shadow_threshold)
    if args.shadow_threshold is not None:
        print(f"[stage7d] shadow-masked rows: {len(df)} / {n_before} "
              f"({100*len(df)/n_before:.1f}% kept at threshold {args.shadow_threshold})")

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

    include_pi_partial = not args.no_attribution
    print(f"[stage7d] running pooled + per-image binary + Spearman "
          f"(per_image_partial_dust={include_pi_partial}) ...")
    results = s7d.run_all(df, min_per_class=args.min_per_class,
                          include_per_image_partial_dust=include_pi_partial)
    print(f"[stage7d] result rows: {len(results)}")

    print(f"[stage7d] writing {args.out} ...")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(args.out, index=False)

    if not args.no_attribution:
        attr_frames = []
        for rule in ("P4_area", "P2_count"):
            attr = s7d.build_attribution_table(results, results, rule)
            attr_frames.append(attr)
        attribution = pd.concat(attr_frames, ignore_index=True)
        attribution.to_parquet(args.attribution_out, index=False)
        print(f"[stage7d] wrote attribution table: {args.attribution_out}")
        for rule in ("P4_area", "P2_count"):
            sub = attribution[attribution["partition_rule"] == rule]
            counts = sub["attribution"].value_counts()
            print(f"[stage7d] {rule} attribution counts: {counts.to_dict()}")

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
