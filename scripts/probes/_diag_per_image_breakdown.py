"""Per-image breakdown: which v2 images worked, which didn't, and why?

Joins per-fold metrics from the full-v2 regression + binary sweeps with the
manifest fields (BoulderLabel, CenterLat/Lon, IncidenceAngle, EmissionAngle,
NPolygons, TerrainNote).  Asks: what predicts per-image model performance?

Outputs:
  - scripts/probes/_diag_per_image_breakdown.md  (full ranking + correlations)
  - reports/figures/13_per_image_performance.png  (4-panel diagnostic)
"""

from __future__ import annotations
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# 1. Manifest
manifest = pd.read_csv(REPO_ROOT / "hirise_40_vclaire.csv")
print(f"Manifest: {len(manifest)} rows, columns: {manifest.columns.tolist()}")
print()
print("Manifest field summary (which fields are populated for all images?):")
for c in ["BoulderLabel", "CenterLat", "IncidenceAngle", "EmissionAngle",
          "NPolygons", "TerrainNote", "QualityNote"]:
    n_nonnull = manifest[c].notna().sum()
    print(f"  {c:<18s}: {n_nonnull}/{len(manifest)} non-null")

# 2. Per-fold from regression sweep (lightgbm_two_stage S=64)
reg_path = REPO_ROOT / "models/_sweep/20260529T061553Z/summary.parquet"
reg = pd.read_parquet(reg_path)
reg = reg[(reg["variant"] == "lightgbm_two_stage") & (reg["scale_idx"] == 3)
          & ~reg["is_specificity_only"].astype(bool)].copy()
reg = reg.dropna(subset=["spearman_rho"])
reg = reg.rename(columns={"held_out_obs_id": "ObsId",
                          "spearman_rho": "reg_spearman",
                          "presence_auc": "reg_presence_auc",
                          "mean_true": "reg_mean_true_fa",
                          "mean_pred": "reg_mean_pred_fa",
                          "n_tiles": "reg_n_tiles"})
print(f"\nRegression sweep: {len(reg)} held-out folds at S=64")

# 3. Per-fold from binary sweep at fa_gt_1e-2 S=64
bin_path = REPO_ROOT / "models/_sweep_binary/20260529T075754Z/summary.parquet"
binsf = pd.read_parquet(bin_path)
bin_rich = binsf[(binsf["target_id"] == "fa_gt_1e-2") & (binsf["scale_idx"] == 3)
                 & ~binsf["is_specificity_only"].astype(bool)].dropna(subset=["auc"])
bin_rich = bin_rich.rename(columns={"held_out_obs_id": "ObsId",
                                    "auc": "bin_rich_auc",
                                    "lift_at_top_k": "bin_rich_lift",
                                    "base_rate": "bin_rich_base_rate",
                                    "n_positive": "n_boulder_rich_tiles",
                                    "n_tiles": "bin_rich_n_tiles"})
# Also fa_gt_1e-3 and bc_ge_1 for comparison
bin_any = binsf[(binsf["target_id"] == "bc_ge_1") & (binsf["scale_idx"] == 3)
                & ~binsf["is_specificity_only"].astype(bool)].dropna(subset=["auc"])
bin_any = bin_any.rename(columns={"held_out_obs_id": "ObsId",
                                  "auc": "bin_any_auc",
                                  "lift_at_top_k": "bin_any_lift"})
print(f"Binary sweep (fa_gt_1e-2 S=64): {len(bin_rich)} held-out folds")
print(f"Binary sweep (bc_ge_1 S=64): {len(bin_any)} held-out folds")

# 4. Merge
keep_manifest = ["ObsId", "BoulderLabel", "CenterLat", "CenterLon_180",
                 "IncidenceAngle", "EmissionAngle", "NPolygons", "TerrainNote"]
df = (manifest[keep_manifest]
      .merge(reg[["ObsId", "reg_spearman", "reg_presence_auc", "reg_mean_true_fa",
                  "reg_mean_pred_fa", "reg_n_tiles"]], on="ObsId", how="inner")
      .merge(bin_rich[["ObsId", "bin_rich_auc", "bin_rich_lift", "bin_rich_base_rate",
                       "n_boulder_rich_tiles"]], on="ObsId", how="left")
      .merge(bin_any[["ObsId", "bin_any_auc", "bin_any_lift"]], on="ObsId", how="left"))
print(f"\nMerged: {len(df)} images with full metric set")

# 5. Print the ranking
print()
print("=" * 110)
print("PER-IMAGE RANKING — fa_gt_1e-2 boulder-rich classifier at S=64")
print("=" * 110)

cols_to_show = ["ObsId", "BoulderLabel", "CenterLat", "IncidenceAngle",
                "NPolygons", "bin_rich_base_rate",
                "bin_rich_auc", "bin_rich_lift",
                "reg_spearman", "reg_presence_auc"]
top = df.dropna(subset=["bin_rich_auc"]).sort_values("bin_rich_lift", ascending=False)
print("\nTOP 10 by boulder-rich lift:")
print(top.head(10)[cols_to_show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\nBOTTOM 10 by boulder-rich lift:")
print(top.tail(10)[cols_to_show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# 6. Correlations between metadata and performance
print()
print("=" * 110)
print("Correlations: per-image features vs per-image performance")
print("=" * 110)
print("(Spearman rank correlation; |rho| > 0.3 with p < 0.05 is suggestive)")
print()

perf_cols = ["bin_rich_auc", "bin_rich_lift", "reg_spearman", "reg_presence_auc"]
feat_cols = ["CenterLat", "IncidenceAngle", "EmissionAngle", "NPolygons",
             "bin_rich_base_rate", "reg_mean_true_fa"]
rows = []
for f in feat_cols:
    for p in perf_cols:
        sub = df[[f, p]].dropna()
        if len(sub) < 5:
            rows.append({"feature": f, "metric": p, "n": len(sub),
                         "rho": float("nan"), "p_value": float("nan")})
            continue
        rho, pval = stats.spearmanr(sub[f], sub[p])
        rows.append({"feature": f, "metric": p, "n": len(sub),
                     "rho": rho, "p_value": pval})
corr = pd.DataFrame(rows)
pivot = corr.pivot(index="feature", columns="metric", values="rho")
print(pivot.to_string(float_format=lambda v: f"{v:+.3f}"))

# Print significant correlations
print("\nSignificant correlations (p < 0.05):")
sig = corr[corr["p_value"] < 0.05].sort_values("p_value")
if len(sig) > 0:
    print(sig.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
else:
    print("  (none)")

# 7. Group by BoulderLabel
print()
print("Performance by BoulderLabel (manifest tag):")
group = df.groupby("BoulderLabel", dropna=False)[perf_cols].agg(["mean", "median", "count"])
print(group.round(3).to_string())

# 8. Figure: 4-panel diagnostic
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: per-image AUC histogram, colored by BoulderLabel
ax = axes[0, 0]
for label, color in [("Boulder rich", "C2"), ("Boulder poor", "C3"), ("unknown", "C0")]:
    sub = df[df["BoulderLabel"] == label].dropna(subset=["bin_rich_auc"])
    ax.hist(sub["bin_rich_auc"], bins=12, alpha=0.6, color=color,
            label=f"{label} (n={len(sub)})")
ax.axvline(0.5, color="red", linestyle="--", label="chance")
ax.axvline(df["bin_rich_auc"].mean(), color="black", linestyle="-",
           lw=2, label=f"mean={df['bin_rich_auc'].mean():.2f}")
ax.set_xlabel("per-image AUC (fa_gt_1e-2, S=64)")
ax.set_ylabel("# images")
ax.set_title("A: Per-image AUC by manifest BoulderLabel\n"
             "(do label categories cluster on performance?)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel B: AUC vs CenterLat
ax = axes[0, 1]
for label, color in [("Boulder rich", "C2"), ("Boulder poor", "C3"), ("unknown", "C0")]:
    sub = df[df["BoulderLabel"] == label].dropna(subset=["bin_rich_auc"])
    ax.scatter(sub["CenterLat"], sub["bin_rich_auc"], s=80, alpha=0.7,
               color=color, label=label, edgecolors="k", linewidths=0.5)
ax.axhline(0.5, color="red", linestyle="--", alpha=0.6)
ax.set_xlabel("CenterLat (deg)")
ax.set_ylabel("per-image AUC (fa_gt_1e-2, S=64)")
ax.set_title("B: AUC vs latitude\n"
             "(does H3 illumination story show through latitude?)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Panel C: AUC vs base rate (boulder-rich fraction)
ax = axes[1, 0]
sub = df.dropna(subset=["bin_rich_auc", "bin_rich_base_rate"])
ax.scatter(sub["bin_rich_base_rate"], sub["bin_rich_auc"], s=80, alpha=0.7,
           c=sub["NPolygons"].apply(lambda x: np.log10(x) if x > 0 else 0),
           cmap="viridis", edgecolors="k", linewidths=0.5)
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label("log10(N polygons)")
ax.axhline(0.5, color="red", linestyle="--", alpha=0.6)
ax.set_xlabel("base rate (P boulder-rich tile in held-out image)")
ax.set_ylabel("per-image AUC")
ax.set_title("C: AUC vs base rate, colored by total polygon count\n"
             "(does easy = high boulder density?)")
ax.grid(alpha=0.3)

# Panel D: AUC vs IncidenceAngle (only for images that have it)
ax = axes[1, 1]
sub = df.dropna(subset=["IncidenceAngle", "bin_rich_auc"])
ax.scatter(sub["IncidenceAngle"], sub["bin_rich_auc"], s=80, alpha=0.7,
           edgecolors="k", linewidths=0.5)
ax.axhline(0.5, color="red", linestyle="--", alpha=0.6)
if len(sub) >= 5:
    rho, pval = stats.spearmanr(sub["IncidenceAngle"], sub["bin_rich_auc"])
    ax.set_title(f"D: AUC vs IncidenceAngle (n={len(sub)} images)\n"
                 f"Spearman ρ={rho:+.3f} p={pval:.3f}\n"
                 f"(H3 shadow_fraction prediction: high IncidenceAngle ⇒ worse model)")
else:
    ax.set_title(f"D: AUC vs IncidenceAngle (n={len(sub)} only — manifest data sparse)")
ax.set_xlabel("IncidenceAngle (deg)")
ax.set_ylabel("per-image AUC")
ax.grid(alpha=0.3)

plt.tight_layout()
out = REPO_ROOT / "reports" / "figures" / "13_per_image_performance.png"
plt.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nFigure -> {out}")

# 9. Markdown summary
out_md = Path(__file__).with_suffix(".md")
lines = [
    "# Per-image breakdown — which v2 images worked, which didn't",
    "",
    f"Source data: full-v2 regression sweep `models/_sweep/20260529T061553Z/` and binary sweep ",
    f"`models/_sweep_binary/20260529T075754Z/` at fa_gt_1e-2 S=64.  Manifest: hirise_40_vclaire.csv.",
    "",
    f"Total images joined: **{len(df)}**",
    "",
    "## Top 10 by boulder-rich lift@top-K",
    "",
    top.head(10)[cols_to_show].to_string(index=False, float_format=lambda v: f"{v:.3f}"),
    "",
    "## Bottom 10 by boulder-rich lift@top-K",
    "",
    top.tail(10)[cols_to_show].to_string(index=False, float_format=lambda v: f"{v:.3f}"),
    "",
    "## Spearman correlations: per-image features vs performance",
    "",
    "```",
    pivot.to_string(float_format=lambda v: f"{v:+.3f}"),
    "```",
    "",
    "## Performance by manifest BoulderLabel",
    "",
    "```",
    group.round(3).to_string(),
    "```",
]
out_md.write_text("\n".join(lines), encoding="utf-8")
print(f"Markdown -> {out_md}")
