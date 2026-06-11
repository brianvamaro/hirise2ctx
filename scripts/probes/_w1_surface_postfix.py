"""Cohort mean-AUC surface on the post-fix rescore grid (validation of the
coreg sign fix: should peak at (0,0) with symmetric falloff)."""
import pandas as pd

grid = pd.read_parquet("scripts/probes/_w1_shift_rescore_postfix.parquet")
coh = grid.pivot_table(index="di", columns="dj", values="auc", aggfunc="mean")
print(coh.to_string(float_format=lambda v: f"{v:.3f}"))
center = grid[(grid.di == 0) & (grid.dj == 0)].set_index("obs_id")["auc"]
for di, dj, label in [(1, 0, "(+1,0)"), (-1, 0, "(-1,0)"), (0, 1, "(0,+1)"), (0, -1, "(0,-1)")]:
    off = grid[(grid.di == di) & (grid.dj == dj)].set_index("obs_id")["auc"]
    d = (off - center).dropna()
    print(f"mean delta {label}: {d.mean():+.4f} (median {d.median():+.4f})")
