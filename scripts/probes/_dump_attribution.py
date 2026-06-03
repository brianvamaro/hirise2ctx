"""Dump per-image attribution table (T=0.10, P4_area) for the writeup."""
from __future__ import annotations

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
attr = pd.read_parquet(ROOT / "dataset_v2" / "stage7d_attribution_shadow_0.10.parquet")
sub = attr.query("partition_rule == 'P4_area'").copy()
sub = sub.sort_values(["attribution", "obs_id"])

print(f"Total eligible images (P4_area, T=0.10): {len(sub)}")
print(f"Counts: {dict(sub['attribution'].value_counts())}")
print()
print(f"{'obs_id':<20} {'attribution':<24} {'n_rich':>7} {'n_poor':>7} "
      f"{'IRBG_raw_d':>11} {'IRBG_par_d':>11} {'IRRED_par_d':>12}")
print("-" * 100)
for _, row in sub.iterrows():
    irbg_raw = row.get("IR_over_BG_raw_d", float("nan"))
    irbg_par = row.get("IR_over_BG_partial_d", float("nan"))
    irred_par = row.get("IR_over_RED_partial_d", float("nan"))
    print(f"{row['obs_id']:<20} {row['attribution']:<24} "
          f"{int(row['n_rich']) if row['n_rich']==row['n_rich'] else '-':>7} "
          f"{int(row['n_poor']) if row['n_poor']==row['n_poor'] else '-':>7} "
          f"{irbg_raw:>+11.3f} {irbg_par:>+11.3f} {irred_par:>+12.3f}")
