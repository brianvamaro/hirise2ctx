"""W1 synthesis — assemble the 38-row per-image dossier.

Joins: post-fix banked-recipe per-image metrics + validity counts, coreg solve
stats, post-fix rescore best offset, detection stats, CTX source stats,
within-image feature-sign class, per-image terrain class (where available),
and an attributed cause per the W1 ladder.

Attribution rules (mundane -> fundamental, evidence in DECISIONS.md):
  geometry_fixed       anti-signal pre-fix, recovered post-fix (rung 1)
  validity_limited     n_neg < 50 or n_pos < 50 (per-image AUC unreliable)
  texture_decorrelated anti + within-image texture-label correlation flat or
                       inverted (class B; rung 5)
  distribution_shift   anti + strong cohort-consistent within-image signal
                       (class C; model misses it at LOIO; rung 5)
  ok                   AUC >= 0.5

Writes dataset_v2/w1_dossier.parquet + scripts/probes/_w1_dossier.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SWEEP = Path(sys.argv[1] if len(sys.argv) > 1 else "models/_sweep_w0/20260611T013810Z/summary.parquet")
PRE_GRID = Path("scripts/probes/_w1_shift_rescore.parquet")        # pre-fix
POST_GRID = Path("scripts/probes/_w1_shift_rescore_postfix.parquet")
COREG = Path("cache_v2/coregistration")
ILLUM = Path("dataset_v2/features_ctx_illum")
TERRAIN = Path("dataset_v2/terrain_classification_v2.parquet")
SIGN_MD_FEATS = ["shadow_fraction", "grad_mag_mean", "glcm_contrast_d1", "edge_density"]

summ = pd.read_parquet(SWEEP)
rec = summ[(summ.variant == "lightgbm_two_stage_balanced") & (summ.target_col == "boulder_count")]
d = rec.set_index("held_out_obs_id")[
    ["n_tiles", "n_meaningful_positive", "meaningful_auc", "pr_auc", "spearman_rho"]
].copy()
d["n_neg"] = d.n_tiles - d.n_meaningful_positive
d["validity_ok"] = (d.n_neg >= 50) & (d.n_meaningful_positive >= 50)

pre = pd.read_parquet(PRE_GRID)
pre_center = pre[(pre.di == 0) & (pre.dj == 0)].set_index("obs_id")["auc"]
d["auc_prefix"] = pre_center

post = pd.read_parquet(POST_GRID)
best = post.loc[post.groupby("obs_id")["auc"].idxmax()].set_index("obs_id")
d["best_offset"] = best.apply(lambda r: f"({int(r.di)},{int(r.dj)})", axis=1)
d["rescore_gain"] = best["auc"] - post[(post.di == 0) & (post.dj == 0)].set_index("obs_id")["auc"]

for obs in d.index:
    j = json.loads((COREG / f"{obs}.json").read_text())
    d.loc[obs, "coreg_dy_m"] = j["shift_m"]["dy"]
    d.loc[obs, "coreg_peak"] = j["peak_correlation"]
    il = pd.read_parquet(ILLUM / f"{obs}.parquet")
    il = il[il.scale_idx == 3]
    d.loc[obs, "mean_n_sources"] = il.ctx_n_sources.mean()
    d.loc[obs, "dom_frac"] = il.ctx_dominant_source_fraction.mean()

# within-image feature-sign summary (recompute, cheap)
LAB, FEA = Path("dataset_v2/labels"), Path("dataset_v2/features")
for obs in d.index:
    lab = pd.read_parquet(LAB / f"{obs}.parquet")
    lab = lab[lab.scale_idx == 3][["ti", "tj", "boulder_count"]]
    fea = pd.read_parquet(FEA / f"{obs}.parquet")
    fea = fea[fea.scale_idx == 3]
    m = lab.merge(fea, on=["ti", "tj"])
    rhos = [m[c].corr(m.boulder_count, method="spearman") for c in SIGN_MD_FEATS]
    d.loc[obs, "texture_rho_med"] = float(np.nanmedian(rhos))

if TERRAIN.exists():
    t = pd.read_parquet(TERRAIN)
    key = [c for c in t.columns if c.lower() in ("obs_id", "obsid")][0]
    cls = [c for c in t.columns if "class" in c.lower() or "terrain" in c.lower()]
    if cls:
        d = d.join(t.set_index(key)[cls[0]].rename("terrain"), how="left")

def attribute(r):
    if r.meaningful_auc >= 0.5:
        if not r.validity_ok:
            return "ok_validity_limited"
        if r.auc_prefix < 0.5:
            return "ok_geometry_fixed"
        return "ok"
    if not r.validity_ok:
        return "validity_limited"
    if r.texture_rho_med < 0.15:
        return "texture_decorrelated"
    return "distribution_shift"

d["attributed_cause"] = d.apply(attribute, axis=1)
# The DN-clip shadow fix (DECISIONS.md 2026-06-10 round 2) recovered these two,
# not the geometry fix -- the generic rule can't distinguish, so override.
for obs in ("ESP_046328_2180", "ESP_064510_2260"):
    if obs in d.index and d.loc[obs, "meaningful_auc"] >= 0.5:
        d.loc[obs, "attributed_cause"] = "ok_shadowfeat_fixed"
d = d.sort_values("meaningful_auc")
d.to_parquet("dataset_v2/w1_dossier.parquet")

cols = ["meaningful_auc", "auc_prefix", "n_meaningful_positive", "n_neg", "validity_ok",
        "best_offset", "rescore_gain", "coreg_dy_m", "coreg_peak", "mean_n_sources",
        "dom_frac", "texture_rho_med", "attributed_cause"]
if "terrain" in d.columns:
    cols.append("terrain")
body = d[cols].to_string(float_format=lambda v: f"{v:.3f}")
Path("scripts/probes/_w1_dossier.md").write_text(
    "# W1 per-image dossier (38 rows, post-fix banked recipe @ S=64)\n\n"
    "```\n" + body + "\n```\n\n## Cause counts\n"
    + d.attributed_cause.value_counts().to_string() + "\n",
    encoding="utf-8",
)
print(body)
print()
print(d.attributed_cause.value_counts().to_string())
print("\nwrote dataset_v2/w1_dossier.parquet + scripts/probes/_w1_dossier.md")
