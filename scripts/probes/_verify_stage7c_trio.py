"""Cross-check Stage 7c trio output against Stage 7.0 Test B findings.

Computes within-image boulder-rich vs boulder-poor effect sizes from
`dataset_v2/features_colour_trio.parquet` and compares to the Stage 7.0
verdict table in DECISIONS.md 2026-05-31:

  ESP_042964_2160: dramatic + dust-attributable (d>0.7 in ratios)
  ESP_054000_2255: no per-polygon signal (Test A null)
  ESP_055253_2245: blueish, IR/BG signal survives dust control (partial r=0.16, p=0.037)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def cohens_d(rich: np.ndarray, poor: np.ndarray) -> float:
    rich = rich[~np.isnan(rich)]
    poor = poor[~np.isnan(poor)]
    if len(rich) < 2 or len(poor) < 2:
        return float("nan")
    pooled = np.sqrt(
        ((len(rich) - 1) * rich.var(ddof=1) + (len(poor) - 1) * poor.var(ddof=1))
        / (len(rich) + len(poor) - 2)
    )
    if pooled == 0:
        return float("nan")
    return float((rich.mean() - poor.mean()) / pooled)


def main() -> int:
    fc = pd.read_parquet("dataset_v2/features_colour_trio.parquet")
    print(f"Stage 7c trio: {len(fc)} rows across {fc['obs_id'].nunique()} images")
    print(f"  scale_idx: {sorted(fc['scale_idx'].unique())}")
    print(f"  IR_iof range:  {fc['IR_iof'].min():.4f} - {fc['IR_iof'].max():.4f}")
    print(f"  RED_iof range: {fc['RED_iof'].min():.4f} - {fc['RED_iof'].max():.4f}")
    print(f"  BG_iof range:  {fc['BG_iof'].min():.4f} - {fc['BG_iof'].max():.4f}")
    print()

    for obs_id in ["ESP_042964_2160", "ESP_054000_2255", "ESP_055253_2245"]:
        lbl = pd.read_parquet(f"dataset_v2/labels/{obs_id}.parquet")
        lbl = lbl[lbl.scale_idx == 3]
        merged = fc[fc.obs_id == obs_id].merge(
            lbl[["obs_id", "scale_idx", "ti", "tj", "fractional_area", "boulder_count"]],
            on=["obs_id", "scale_idx", "ti", "tj"],
        )
        rich = merged[merged.fractional_area >= 1e-2]
        poor = merged[merged.fractional_area < 1e-2]
        print(f"{obs_id}: merged rows={len(merged)}, n_rich={len(rich)}, n_poor={len(poor)}")
        for col in [
            "IR_iof", "RED_iof", "BG_iof",
            "IR_over_RED", "IR_over_BG", "dust_index_RED_over_BG",
        ]:
            d = cohens_d(rich[col].to_numpy(), poor[col].to_numpy())
            print(f"  {col:25s} rich={rich[col].mean():.4f} poor={poor[col].mean():.4f} d={d:+.3f}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
