"""Verify the DN-clip fix: shadow features must be alive (non-constant) on the
two affected images and the within-image shadow-label correlation computable."""
from pathlib import Path

import pandas as pd

for obs in ["ESP_046328_2180", "ESP_064510_2260"]:
    d = pd.read_parquet(f"dataset_v2/features/{obs}.parquet")
    d = d[d.scale_idx == 3]
    lab = pd.read_parquet(f"dataset_v2/labels/{obs}.parquet")
    lab = lab[lab.scale_idx == 3][["ti", "tj", "boulder_count"]]
    m = lab.merge(d, on=["ti", "tj"])
    print(f"{obs}:")
    for c in ["shadow_fraction", "shadow_fraction_strict", "lacunarity_shadow_b2"]:
        rho = m[c].corr(m.boulder_count, method="spearman")
        print(f"  {c}: nunique={d[c].nunique()}, mean={d[c].mean():.4f}, "
              f"rho(label)={rho:+.3f}" if pd.notna(rho) else f"  {c}: still constant!")
