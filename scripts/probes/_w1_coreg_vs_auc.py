"""W1 Rung 1b — co-registration shift/quality vs per-image AUC.

Joins cache_v2/coregistration/*.json (block-median solves, applied to the
polygons in Stage 4) against the banked-recipe per-image meaningful AUC and
the rung-1a best-shift results. Questions:

1. Does coreg quality (peak correlation, n_confident_blocks, block MAD)
   predict per-image AUC? (bad solve -> bad geometry -> bad AUC)
2. Does the applied shift magnitude/direction predict the rung-1a best
   offset? (sign error or residual misalignment would show as a systematic
   relationship; tile = 320 m at S=64, row di ~ -dy_m/320 if labels lag)

Writes scripts/probes/_w1_coreg_vs_auc.md.
"""
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

COREG_DIR = Path("cache_v2/coregistration")
SUMMARY = Path("models/_sweep_w0/20260610T221932Z/summary.parquet")
SHIFT_GRID = Path("scripts/probes/_w1_shift_rescore.parquet")
OUT_MD = Path("scripts/probes/_w1_coreg_vs_auc.md")
TILE_M = 320.0

rows = []
for f in sorted(COREG_DIR.glob("*.json")):
    d = json.loads(f.read_text())
    bf = d.get("block_field") or {}
    mad = bf.get("block_mad_px") or {}
    rows.append(
        dict(
            obs_id=d["obs_id"],
            dx_m=d["shift_m"]["dx"],
            dy_m=d["shift_m"]["dy"],
            mag_m=d["shift_m"]["magnitude"],
            peak=d["peak_correlation"],
            method=d.get("method", "single_window"),
            n_blocks=bf.get("n_blocks"),
            n_confident=bf.get("n_confident_blocks"),
            mad_dy_px=mad.get("dy"),
            mad_dx_px=mad.get("dx"),
        )
    )
coreg = pd.DataFrame(rows).set_index("obs_id")

summ = pd.read_parquet(SUMMARY)
rec = summ[
    (summ.variant == "lightgbm_two_stage_balanced") & (summ.target_col == "boulder_count")
].set_index("held_out_obs_id")

grid = pd.read_parquet(SHIFT_GRID)
center = grid[(grid.di == 0) & (grid.dj == 0)].set_index("obs_id")["auc"]
best = grid.loc[grid.groupby("obs_id")["auc"].idxmax()].set_index("obs_id")

tab = coreg.join(rec[["meaningful_auc", "meaningful_base_rate", "n_tiles"]], how="inner")
tab = tab.join(best[["di", "dj", "auc"]].rename(columns={"di": "best_di", "dj": "best_dj", "auc": "auc_best"}))
tab["gain"] = tab.auc_best - tab.meaningful_auc
tab["confident_frac"] = tab.n_confident / tab.n_blocks
tab["anti_signal"] = tab.meaningful_auc < 0.5
# applied shift expressed in S=64 tile units (row axis: +row = south = -y)
tab["dy_tiles"] = tab.dy_m / TILE_M
tab["dx_tiles"] = tab.dx_m / TILE_M
tab = tab.sort_values("meaningful_auc")

print(f"images with coreg solve: {len(coreg)}; joined to recipe AUC: {len(tab)}")
missing = set(rec.index) - set(coreg.index)
print("recipe images missing coreg:", sorted(missing))

corr_lines = []
for col in ["peak", "mag_m", "confident_frac", "mad_dy_px", "mad_dx_px", "n_confident"]:
    sub = tab[[col, "meaningful_auc"]].dropna()
    rho, p = spearmanr(sub[col], sub["meaningful_auc"])
    corr_lines.append(f"- `{col}` vs meaningful_auc: Spearman rho={rho:+.3f} p={p:.4f} (n={len(sub)})")
    print(corr_lines[-1])

# does the applied shift direction predict the best rescore offset?
for sign in (+1, -1):
    sub = tab.dropna(subset=["dy_tiles", "best_di"])
    rho_i, p_i = spearmanr(sign * sub.dy_tiles, sub.best_di)
    rho_j, p_j = spearmanr(sign * sub.dx_tiles, sub.best_dj)
    line = (f"- sign {sign:+d}: dy_tiles vs best_di rho={rho_i:+.3f} p={p_i:.4f}; "
            f"dx_tiles vs best_dj rho={rho_j:+.3f} p={p_j:.4f}")
    corr_lines.append(line)
    print(line)

cols = ["meaningful_auc", "anti_signal", "best_di", "best_dj", "gain", "dy_m", "dx_m",
        "mag_m", "peak", "confident_frac", "mad_dy_px", "mad_dx_px", "method"]
body = tab[cols].to_string(float_format=lambda v: f"{v:.3f}")
print(body)

OUT_MD.write_text(
    "\n".join(
        [
            "# W1 Rung 1b — coreg shift/quality vs per-image AUC",
            "",
            f"Coreg solves: `{COREG_DIR}` (block-median, applied to polygons in Stage 4).",
            f"Recipe: two_stage_balanced × boulder_count @ S=64. Tile = {TILE_M:g} m.",
            "",
            "## Correlations",
            *corr_lines,
            "",
            "## Per-image table (sorted by AUC)",
            "```",
            body,
            "```",
        ]
    ),
    encoding="utf-8",
)
print(f"\nwrote {OUT_MD}")
