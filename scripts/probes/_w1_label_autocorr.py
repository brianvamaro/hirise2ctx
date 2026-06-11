"""How much does a 1-tile label shift actually corrupt the labels?

Per image at S=64: Spearman correlation and binary (bc>50) agreement between
y_true(ti, tj) and y_true(ti+1, tj). High values mean the boulder field is
spatially smooth at 320 m, so the pre-fix ~1.1-tile misalignment behaved as
mild label noise, not full scrambling — bounding the achievable AUC gain.
"""
from pathlib import Path

import numpy as np
import pandas as pd

rows = []
for f in sorted(Path("dataset_v2/labels").glob("*.parquet")):
    d = pd.read_parquet(f)
    d = d[d.scale_idx == 3].set_index(["ti", "tj"])["boulder_count"]
    shifted = d.copy()
    shifted.index = pd.MultiIndex.from_arrays([d.index.get_level_values(0) + 1,
                                               d.index.get_level_values(1)])
    j = pd.DataFrame({"a": d, "b": shifted}).dropna()
    if len(j) < 50:
        continue
    rows.append(dict(
        obs_id=f.stem,
        rho=j.a.corr(j.b, method="spearman"),
        binary_agree=float(((j.a > 50) == (j.b > 50)).mean()),
    ))
df = pd.DataFrame(rows)
print(df.describe().loc[["mean", "50%", "min", "max"]].to_string(float_format=lambda v: f"{v:.3f}"))
print(f"\ncohort: median label autocorr at 1 tile = {df.rho.median():.3f}; "
      f"median binary (bc>50) agreement = {df.binary_agree.median():.1%}")
