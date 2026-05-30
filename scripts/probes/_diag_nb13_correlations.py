"""Replicate the §4 correlation analysis from notebook 13 and dump to a markdown table."""
import sys, re, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import src.modeling  # noqa: F401
import pandas as pd, numpy as np
from scipy import stats

# Reproduce the per-image join from notebook 13 setup-data
MODELS = REPO / "models"
LBL_DIR = REPO / "cache" / "pds_labels"

manifest = pd.read_csv(REPO / "hirise_40_vclaire.csv")
reg = pd.read_parquet(MODELS / "_sweep" / "20260529T061553Z" / "summary.parquet")
reg = reg[(reg["variant"] == "lightgbm_two_stage") & (reg["scale_idx"] == 3)
          & ~reg["is_specificity_only"].astype(bool)].dropna(subset=["spearman_rho"]).copy()
reg = reg.rename(columns={"held_out_obs_id": "ObsId",
                          "spearman_rho": "reg_spearman",
                          "presence_auc": "reg_presence_auc",
                          "mean_true": "reg_mean_true_fa"})

binsf = pd.read_parquet(MODELS / "_sweep_binary" / "20260529T075754Z" / "summary.parquet")
bin_rich = binsf[(binsf["target_id"] == "fa_gt_1e-2") & (binsf["scale_idx"] == 3)
                 & ~binsf["is_specificity_only"].astype(bool)].dropna(subset=["auc"]).copy()
bin_rich = bin_rich.rename(columns={"held_out_obs_id": "ObsId",
                                    "auc": "bin_rich_auc",
                                    "lift_at_top_k": "bin_rich_lift",
                                    "base_rate": "bin_rich_base_rate",
                                    "ece": "bin_rich_ece"})

df = (manifest[["ObsId", "BoulderLabel", "CenterLat", "NPolygons"]]
      .merge(reg[["ObsId", "reg_spearman", "reg_presence_auc", "reg_mean_true_fa"]], on="ObsId", how="inner")
      .merge(bin_rich[["ObsId", "bin_rich_auc", "bin_rich_lift", "bin_rich_base_rate", "bin_rich_ece"]], on="ObsId", how="left"))

# LBL augmentation
def parse_lbl(p):
    text = p.read_text(errors="ignore")
    return {
        n: float(m.group(1)) if (m := re.search(rf"{n}\s*=\s*([\d.+-]+)", text)) else None
        for n in ["INCIDENCE_ANGLE", "EMISSION_ANGLE", "PHASE_ANGLE", "SUB_SOLAR_AZIMUTH"]
    }

lbl = pd.DataFrame([
    {"ObsId": o, **(parse_lbl(LBL_DIR / f"{o}.LBL") if (LBL_DIR / f"{o}.LBL").exists() else {})}
    for o in df["ObsId"]
])
lbl = lbl.rename(columns={
    "INCIDENCE_ANGLE": "IncidenceAngle",
    "EMISSION_ANGLE": "EmissionAngle",
    "PHASE_ANGLE": "PhaseAngle",
    "SUB_SOLAR_AZIMUTH": "SubSolarAzimuth",
})
df = df.merge(lbl, on="ObsId", how="left")

# Correlation table
feat_cols = ["CenterLat", "IncidenceAngle", "EmissionAngle", "PhaseAngle", "SubSolarAzimuth",
             "NPolygons", "bin_rich_base_rate", "reg_mean_true_fa"]
perf_cols = ["bin_rich_auc", "bin_rich_lift", "bin_rich_ece", "reg_spearman"]

rows = []
for f in feat_cols:
    for p in perf_cols:
        sub = df[[f, p]].dropna()
        if len(sub) < 5:
            rows.append({"feature": f, "metric": p, "n": len(sub), "rho": float("nan"), "p": float("nan")})
        else:
            rho, pval = stats.spearmanr(sub[f], sub[p])
            rows.append({"feature": f, "metric": p, "n": len(sub), "rho": rho, "p": pval})
corr = pd.DataFrame(rows)
piv_rho = corr.pivot(index="feature", columns="metric", values="rho").round(3)
piv_p = corr.pivot(index="feature", columns="metric", values="p").round(3)

# Mark cells where p<0.05
out_lines = ["# Notebook 13 §4 correlation table", "",
             "## Spearman rho (per-image features vs performance metrics)", "",
             piv_rho.to_string(),
             "",
             "## p-values",
             "",
             piv_p.to_string(),
             "",
             "## Significant (p < 0.05) only:",
             "",
             corr[corr["p"] < 0.05].sort_values("p").round(3).to_string(index=False)]
out = Path(__file__).with_suffix(".md")
out.write_text("\n".join(out_lines), encoding="utf-8")
print(f"wrote {out}", flush=True)
