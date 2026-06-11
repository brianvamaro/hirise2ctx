"""Checks 4 + 1 from the post-W1 bug-hunt list.

Check 4 — presence-AUC coincidence: compare PER-FOLD presence AUC between the
pre-fix sweep (20260610T221932Z) and post-fix sweep (20260611T013810Z) for the
banked cell. If folds differ but means coincide at 4 dp, it's a real
coincidence AND proof the retrain consumed the new labels.

Check 1 — degenerate features per image: for each image, which of the S=64
feature columns have zero variance? Focus on shadow features for the two
anti-signal images that produced constant-input warnings.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PRE = Path("models/_sweep_w0/20260610T221932Z/summary.parquet")
POST = Path("models/_sweep_w0/20260611T013810Z/summary.parquet")
FEATURES = Path("dataset_v2/features")

def cell(p):
    s = pd.read_parquet(p)
    s = s[(s.variant == "lightgbm_two_stage_balanced") & (s.target_col == "boulder_count")]
    return s.set_index("held_out_obs_id")

pre, post = cell(PRE), cell(POST)
cmp = pd.DataFrame({"pre": pre.presence_auc, "post": post.presence_auc})
cmp["delta"] = cmp.post - cmp.pre
n_diff = int((cmp.delta.abs() > 1e-6).sum())
print("== Check 4: per-fold presence AUC pre vs post ==")
print(cmp.sort_values("delta").head(5).to_string(float_format=lambda v: f"{v:.4f}"))
print("  ...")
print(cmp.sort_values("delta").tail(5).to_string(float_format=lambda v: f"{v:.4f}"))
print(f"folds that changed: {n_diff}/{len(cmp)}; mean pre {cmp.pre.mean():.6f} vs post {cmp.post.mean():.6f}")
print(f"mean |delta| = {cmp.delta.abs().mean():.4f}, max |delta| = {cmp.delta.abs().max():.4f}")

print("\n== Check 1: degenerate (zero-variance) features per image at S=64 ==")
KEY = {"obs_id", "scale_idx", "tile_size_px", "ti", "tj", "config_hash"}
rows = []
for f in sorted(FEATURES.glob("*.parquet")):
    d = pd.read_parquet(f)
    d = d[d.scale_idx == 3]
    feats = [c for c in d.columns if c not in KEY]
    dead = [c for c in feats if d[c].nunique(dropna=False) <= 1]
    rows.append(dict(obs_id=f.stem, n_dead=len(dead), dead=",".join(dead)))
df = pd.DataFrame(rows)
print(df[df.n_dead > 0].to_string(index=False))
print(f"\nimages with >=1 dead feature: {(df.n_dead > 0).sum()}/{len(df)}")

print("\nshadow feature detail for the two suspects:")
for obs in ["ESP_046328_2180", "ESP_064510_2260"]:
    d = pd.read_parquet(FEATURES / f"{obs}.parquet")
    d = d[d.scale_idx == 3]
    cols = [c for c in d.columns if "shadow" in c or "lacunarity" in c]
    print(f"  {obs}:")
    for c in cols:
        print(f"    {c}: min={d[c].min():.6f} max={d[c].max():.6f} mean={d[c].mean():.6f} nunique={d[c].nunique()}")
df.to_csv("scripts/probes/_w1_dead_features.csv", index=False)
