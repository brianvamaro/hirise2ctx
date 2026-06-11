"""W1 — inference-computable reliability proxies vs per-image AUC.

Candidates (all computable from CTX alone at inference):
  - between-tile dispersion of texture features within the window (uniform
    speckle -> nothing to discriminate -> class-B failure)
  - CTX source stats (mean_n_sources, dominant_source_fraction)
  - feature-distribution shift vs the training cohort (median |z| of the
    image's feature medians under cohort statistics) -> class-C failure

Reports Spearman vs the post-fix per-image meaningful AUC and a simple
2-of-3 flag's confusion against anti-signal status.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FEATURES = Path("dataset_v2/features")
ILLUM = Path("dataset_v2/features_ctx_illum")
SWEEP = Path("models/_sweep_w0/20260611T013810Z/summary.parquet")
FEATS = ["glcm_contrast_d1", "grad_mag_mean", "edge_density", "shadow_fraction", "intensity_std"]

summ = pd.read_parquet(SWEEP)
rec = summ[(summ.variant == "lightgbm_two_stage_balanced") & (summ.target_col == "boulder_count")]
auc = rec.set_index("held_out_obs_id")["meaningful_auc"]

per_img = {}
for f in sorted(FEATURES.glob("*.parquet")):
    obs = f.stem
    d = pd.read_parquet(f)
    d = d[d.scale_idx == 3]
    il = pd.read_parquet(ILLUM / f"{obs}.parquet")
    il = il[il.scale_idx == 3]
    per_img[obs] = dict(
        # dispersion: robust CV of each feature across tiles, averaged
        dispersion=float(np.nanmean([
            (d[c].quantile(0.9) - d[c].quantile(0.1)) / (abs(d[c].median()) + 1e-9)
            for c in FEATS
        ])),
        med=d[FEATS].median(),
        mean_n_sources=float(il.ctx_n_sources.mean()),
        dom_frac=float(il.ctx_dominant_source_fraction.mean()),
    )

obs_ids = list(per_img)
med_mat = pd.DataFrame({o: per_img[o]["med"] for o in obs_ids}).T
coh_med = med_mat.median()
coh_iqr = med_mat.quantile(0.75) - med_mat.quantile(0.25)
shift = ((med_mat - coh_med).abs() / (coh_iqr + 1e-9)).median(axis=1)

df = pd.DataFrame(
    dict(
        auc=auc,
        dispersion=pd.Series({o: per_img[o]["dispersion"] for o in obs_ids}),
        feat_shift=shift,
        mean_n_sources=pd.Series({o: per_img[o]["mean_n_sources"] for o in obs_ids}),
        dom_frac=pd.Series({o: per_img[o]["dom_frac"] for o in obs_ids}),
    )
).dropna(subset=["auc"])
df["anti"] = df.auc < 0.5

for c in ["dispersion", "feat_shift", "mean_n_sources", "dom_frac"]:
    rho, p = spearmanr(df[c], df.auc)
    print(f"{c:>16s} vs AUC: rho={rho:+.3f} p={p:.4f}")

print()
print(df.sort_values("auc").to_string(float_format=lambda v: f"{v:.3f}"))
df.to_csv("scripts/probes/_w1_reliability_proxy.csv")
print("wrote scripts/probes/_w1_reliability_proxy.csv")
