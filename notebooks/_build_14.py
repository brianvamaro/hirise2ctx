"""Build notebooks/14_compositional_feasibility.ipynb from Python source.

Stage 7.0 feasibility analysis -- companion to scripts/probes/_stage7_feasibility.py.
Reads the parquet outputs of the probe and renders the per-image and pooled stats,
the dust-confound discriminator, and the go/no-go decision per PLAN_Compositional.md
section 3.1.

Sections:
  1. Question + design (recap of PLAN section 3.1; what changed during implementation)
  2. Per-image overview (LBL metadata, swath coverage, polygon counts)
  3. Test A -- per-polygon paired spectra (interior vs 2-10 m outward ring)
  4. Test B -- per-tile S=64 spectra (boulder-rich vs boulder-poor truth partition)
  5. Dust-confound discriminator (dust_index distributions + partial correlation)
  6. Go / no-go decision against the PLAN section 3.1 pass conditions
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "14_compositional_feasibility.ipynb"


def md(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

# ---------------------------------------------------------------------------
# 1. Intro
# ---------------------------------------------------------------------------
cells.append(md(
    """# 14 -- Compositional feasibility (Stage 7.0)

Gate for the [PLAN_Compositional.md](../PLAN_Compositional.md) Stage 7a-7e investment.
Tests whether **boulder-rich HiRISE pixels differ spectrally from their immediate
surroundings** at the per-polygon (Test A) and per-tile (Test B) scales, using
*truth* BoulderNet polygons (not model predictions) on a 3-image trio.

**Trio (as locked-in 2026-05-31)**:
- `ESP_042964_2160` -- high-density positive (model AUC 0.91)
- `ESP_054000_2255` -- anti-signal #1 (model AUC 0.40, anti-correlated)
- `ESP_055253_2245` -- anti-signal #2 (model AUC 0.42), substituted for the
  original `ESP_055978_2270` which has no PDS `COLOR.JP2`

This notebook RENDERS the probe outputs at
[`cache_v2/stage7/`](../cache_v2/stage7) -- it does not re-extract spectra.
Re-run the probe via:

```powershell
& "C:/Users/brian/anaconda3/Scripts/conda.exe" run --no-capture-output -n geospatial `
    python -u scripts/probes/_stage7_feasibility.py
```

Probe runtime is ~3-10 min on cached COLOR.JP2s.
""",
    cell_id="intro",
))

# ---------------------------------------------------------------------------
# 2. Setup + per-image overview
# ---------------------------------------------------------------------------
cells.append(md(
    """## 1. Setup + per-image overview

Cache layout (under `cache_v2/`):
- `hirise_color/{ObsId}_COLOR.JP2` + `.LBL` -- the colour rasters and labels
  (downloaded by `scripts/probes/_fetch_color.py`)
- `reprojected_detections/{ObsId}.json` -- Stage 1 SP1-corrected source CRS,
  reused to override the buggy CRS the COLOR.JP2 reports
- `stage7/test_a_per_polygon_{ObsId}.parquet` -- per-polygon paired spectra
- `stage7/test_b_per_tile_{ObsId}.parquet` -- per-tile spectra + truth labels
- `stage7/{test_a,test_b}_summary.parquet` -- per-image, per-feature statistics
""",
    cell_id="setup-md",
))

cells.append(code(
    """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sst

ROOT = Path("..").resolve()
sys.path.insert(0, str(ROOT))
from src import colour  # noqa: E402

CACHE = ROOT / "cache_v2"
STAGE7 = CACHE / "stage7"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TRIO = ["ESP_042964_2160", "ESP_054000_2255", "ESP_055253_2245"]
ROLES = {
    "ESP_042964_2160": "high-density positive (AUC 0.91)",
    "ESP_054000_2255": "anti-signal #1 (AUC 0.40)",
    "ESP_055253_2245": "anti-signal #2 (AUC 0.42)",
}

# Load per-image LBL metadata
overview = []
for obs in TRIO:
    lbl = colour.parse_color_lbl(colour.color_lbl_path(CACHE, obs))
    overview.append({
        "obs_id": obs, "role": ROLES[obs],
        "incidence_deg": lbl.incidence_deg, "cos_i": lbl.cos_incidence,
        "emission_deg": lbl.emission_deg, "phase_deg": lbl.phase_deg,
        "solar_longitude_deg": lbl.solar_longitude_deg,
        "map_scale_mpp": lbl.map_scale_mpp,
        "swath_size_px": f"{lbl.lines} x {lbl.line_samples}",
        "swath_width_m": round(lbl.line_samples * lbl.map_scale_mpp),
    })
overview_df = pd.DataFrame(overview)
overview_df
""",
    cell_id="setup-code",
))

# ---------------------------------------------------------------------------
# 3. Test A
# ---------------------------------------------------------------------------
cells.append(md(
    """## 2. Test A -- per-polygon paired spectra

For each boulder polygon (within the colour swath), we extract the per-band mean
I/F **inside the polygon** and in a **2-10 m outward buffer ring**, excluding the
polygon interior. Then we test whether `interior_band` differs from `ring_band`
using a paired Wilcoxon signed-rank test, and report Cohen's d (paired) on the
diff distribution.

**Lambertian correction note** (per PLAN section 5.3, 2026-05-31 update):
`I/F_corrected = I/F_obs / cos(i)` is a multiplicative constant per image. It
cancels in (interior - ring) differences AND in all band ratios -- so the Test A
results below are Lambertian-invariant within each image, even though they are
raw I/F values.

The PLAN section 3.1 pass condition (per Test A) is:
- significant difference in **at least one** of {IR, RED, BG, IR/BG, IR/RED,
  dust_index = RED/BG} in **at least one** image (Wilcoxon `p < 0.05`,
  `|d| > 0.3`).
""",
    cell_id="test-a-md",
))

cells.append(code(
    """# Load per-image Test A spectra + summary
per_polygon = pd.concat(
    [pd.read_parquet(STAGE7 / f"test_a_per_polygon_{obs}.parquet") for obs in TRIO],
    ignore_index=True,
)
summary_a = pd.read_parquet(STAGE7 / "test_a_summary.parquet")
summary_a_display = (
    summary_a
    .pivot_table(index="feature",
                 columns="obs_id",
                 values=["mean_diff", "cohens_d_paired", "wilcoxon_p"],
                 aggfunc="first")
)
print(f"Total polygons across the trio: {len(per_polygon)}")
print(f"Per-image:")
print(per_polygon["obs_id"].value_counts())
summary_a_display
""",
    cell_id="test-a-code",
))

cells.append(code(
    """# Visualise the paired diff distribution per band x ObsId
fig, axes = plt.subplots(3, 6, figsize=(18, 9), sharex=False)
bands = ["IR", "RED", "BG", "IR/BG", "IR/RED", "RED/BG"]
for r, obs in enumerate(TRIO):
    sub = per_polygon[per_polygon["obs_id"] == obs]
    if sub.empty:
        for c, _ in enumerate(bands):
            axes[r, c].text(0.5, 0.5, "no data", transform=axes[r, c].transAxes, ha="center")
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        continue
    for c, band in enumerate(bands):
        if "/" in band:
            num, den = band.split("/")
            diffs = (sub[f"{num}_in"] / sub[f"{den}_in"]
                     - sub[f"{num}_ring"] / sub[f"{den}_ring"]).dropna()
        else:
            diffs = (sub[f"{band}_in"] - sub[f"{band}_ring"]).dropna()
        ax = axes[r, c]
        ax.hist(diffs, bins=40, color="steelblue", alpha=0.75)
        ax.axvline(0, color="k", linewidth=0.7)
        ax.axvline(diffs.mean(), color="crimson", linewidth=1.2,
                   label=f"mean={diffs.mean():+.4f}")
        try:
            _, p = sst.wilcoxon(diffs)
        except ValueError:
            p = float("nan")
        d = diffs.mean() / (diffs.std(ddof=1) or 1)
        ax.set_title(f"{obs}\\n{band}: d={d:+.2f}, p={p:.1e}", fontsize=8)
        if r == len(TRIO) - 1:
            ax.set_xlabel("interior - ring")
fig.suptitle("Test A -- per-polygon paired diffs (interior - ring) per band / ratio",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG / "stage7_test_a_paired_diffs.png", dpi=120, bbox_inches="tight")
plt.show()
""",
    cell_id="test-a-fig",
))

# ---------------------------------------------------------------------------
# 4. Test B
# ---------------------------------------------------------------------------
cells.append(md(
    """## 3. Test B -- per-tile @ S=64, boulder-rich vs boulder-poor truth partition

For each 320 m (`S=64` CTX-pixel) tile, we extract the mean COLOR.JP2 I/F per band
within the tile bounds (after reprojection from CTX CRS to the SP1-corrected source
CRS). Tiles are partitioned by the truth label `fa_gt_1e-2` -- i.e. tiles whose
HiRISE-derived `fractional_area >= 1e-2` are "boulder-rich".

Two-sample Mann-Whitney U + Cohen's d (unpaired) per band/ratio.
""",
    cell_id="test-b-md",
))

cells.append(code(
    """per_tile = pd.concat(
    [pd.read_parquet(STAGE7 / f"test_b_per_tile_{obs}.parquet") for obs in TRIO],
    ignore_index=True,
)
summary_b = pd.read_parquet(STAGE7 / "test_b_summary.parquet")
summary_b_display = (
    summary_b
    .pivot_table(index="feature",
                 columns="obs_id",
                 values=["mean_rich", "mean_poor", "cohens_d", "mannwhitney_p"],
                 aggfunc="first")
)
print(f"Per-image S=64 tile counts (in swath):")
print(per_tile.groupby("obs_id").size())
print(f"\\nRich vs poor counts (fa >= 1e-2):")
print(per_tile.assign(rich=per_tile["fractional_area"] >= 1e-2)
      .groupby(["obs_id", "rich"]).size().unstack(fill_value=0))
summary_b_display
""",
    cell_id="test-b-code",
))

cells.append(code(
    """# Boulder-rich vs boulder-poor I/F distributions per band x ObsId
fig, axes = plt.subplots(3, 6, figsize=(18, 9))
features = [
    ("IR", lambda d: d["IR"]),
    ("RED", lambda d: d["RED"]),
    ("BG", lambda d: d["BG"]),
    ("IR/BG", lambda d: d["IR"] / d["BG"]),
    ("IR/RED", lambda d: d["IR"] / d["RED"]),
    ("RED/BG (dust)", lambda d: d["RED"] / d["BG"]),
]
for r, obs in enumerate(TRIO):
    sub = per_tile[per_tile["obs_id"] == obs]
    rich = sub[sub["fractional_area"] >= 1e-2]
    poor = sub[sub["fractional_area"] < 1e-2]
    for c, (name, fn) in enumerate(features):
        ax = axes[r, c]
        try:
            r_vals = fn(rich).dropna()
            p_vals = fn(poor).dropna()
            if len(r_vals) and len(p_vals):
                lo = min(r_vals.min(), p_vals.min())
                hi = max(r_vals.max(), p_vals.max())
                bins = np.linspace(lo, hi, 30)
                ax.hist(p_vals, bins=bins, alpha=0.5, label=f"poor n={len(p_vals)}",
                        color="grey", density=True)
                ax.hist(r_vals, bins=bins, alpha=0.6, label=f"rich n={len(r_vals)}",
                        color="crimson", density=True)
                # Stats
                if len(r_vals) >= 10 and len(p_vals) >= 10:
                    u, p_val = sst.mannwhitneyu(r_vals, p_vals)
                    na, nb = len(r_vals), len(p_vals)
                    pooled = np.sqrt(((na-1)*r_vals.var(ddof=1)
                                      + (nb-1)*p_vals.var(ddof=1)) / (na+nb-2))
                    d = (r_vals.mean() - p_vals.mean()) / (pooled or 1)
                else:
                    p_val = float("nan"); d = float("nan")
                ax.set_title(f"{obs}\\n{name}: d={d:+.2f}, p={p_val:.1e}", fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"err: {e}", transform=ax.transAxes, fontsize=6)
        if r == 0 and c == 0:
            ax.legend(fontsize=6)
fig.suptitle("Test B -- per-tile I/F distributions, boulder-rich vs boulder-poor (truth)",
             fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG / "stage7_test_b_rich_vs_poor.png", dpi=120, bbox_inches="tight")
plt.show()
""",
    cell_id="test-b-fig",
))

# ---------------------------------------------------------------------------
# 5. Dust-confound
# ---------------------------------------------------------------------------
cells.append(md(
    """## 4. Dust-confound discriminator (PLAN section 5)

A spectral difference between boulder-rich and boulder-poor regions has TWO competing
explanations: (i) **composition** (different mineralogy of the boulders vs the
surrounding regolith) or (ii) **dust** (boulder-rich areas have less dust accumulation,
shifting them bluer in IRB).

The PLAN section 5.1 dust proxy is `dust_index = RED / BG` (higher = more dust). The
PLAN section 5.2 procedure: (a) test boulder-rich vs boulder-poor on `dust_index`,
(b) test on `IR/BG` and `IR/RED`, (c) **partial-correlate** the IR/BG difference with
`dust_index` to see if it survives.

We do the partial-correlation analysis on the Test B per-tile data (more direct
sample-size match to the boulder-rich/boulder-poor partition). The Test A polygon
data gives a similar within-image analysis for free.
""",
    cell_id="dust-md",
))

cells.append(code(
    """def partial_corr(x, y, z):
    \"\"\"Partial correlation of x and y controlling for z (Pearson on residuals).\"\"\"
    x = np.asarray(x); y = np.asarray(y); z = np.asarray(z)
    m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[m], y[m], z[m]
    if len(x) < 3:
        return float("nan"), float("nan")
    # OLS residuals
    bx = np.polyfit(z, x, 1); rx = x - np.polyval(bx, z)
    by = np.polyfit(z, y, 1); ry = y - np.polyval(by, z)
    r, p = sst.pearsonr(rx, ry)
    return float(r), float(p)


rows = []
for obs in TRIO:
    sub = per_tile[per_tile["obs_id"] == obs].copy()
    if sub.empty:
        continue
    sub["rich"] = (sub["fractional_area"] >= 1e-2).astype(int)
    sub["dust_index"] = sub["RED"] / sub["BG"]
    sub["IR_over_BG"] = sub["IR"] / sub["BG"]
    sub["IR_over_RED"] = sub["IR"] / sub["RED"]

    # (a) rich vs poor on dust_index
    rich_dust = sub.loc[sub["rich"] == 1, "dust_index"].dropna()
    poor_dust = sub.loc[sub["rich"] == 0, "dust_index"].dropna()
    if len(rich_dust) >= 10 and len(poor_dust) >= 10:
        _, p_dust = sst.mannwhitneyu(rich_dust, poor_dust)
        pooled = np.sqrt(((len(rich_dust)-1) * rich_dust.var(ddof=1)
                          + (len(poor_dust)-1) * poor_dust.var(ddof=1))
                         / (len(rich_dust) + len(poor_dust) - 2))
        d_dust = (rich_dust.mean() - poor_dust.mean()) / (pooled or 1)
    else:
        p_dust = d_dust = float("nan")

    # (b) marginal correlation of `rich` (1/0) with IR/BG and IR/RED
    r_irbg, p_irbg = sst.pointbiserialr(sub["rich"], sub["IR_over_BG"])
    r_irred, p_irred = sst.pointbiserialr(sub["rich"], sub["IR_over_RED"])

    # (c) partial correlation of `rich` with IR/BG (and IR/RED) controlling for dust_index
    pr_irbg, pp_irbg = partial_corr(sub["rich"], sub["IR_over_BG"], sub["dust_index"])
    pr_irred, pp_irred = partial_corr(sub["rich"], sub["IR_over_RED"], sub["dust_index"])
    rows.append({
        "obs_id": obs,
        "dust_d_rich_vs_poor": round(d_dust, 3), "dust_p": p_dust,
        "marginal_r_IRoverBG": round(r_irbg, 3), "marg_p_IRoverBG": p_irbg,
        "partial_r_IRoverBG|dust": round(pr_irbg, 3), "partial_p_IRoverBG|dust": pp_irbg,
        "marginal_r_IRoverRED": round(r_irred, 3), "marg_p_IRoverRED": p_irred,
        "partial_r_IRoverRED|dust": round(pr_irred, 3), "partial_p_IRoverRED|dust": pp_irred,
    })
dust_df = pd.DataFrame(rows)
dust_df.to_parquet(STAGE7 / "dust_summary.parquet")
dust_df
""",
    cell_id="dust-code",
))

# ---------------------------------------------------------------------------
# 6. Go / no-go decision
# ---------------------------------------------------------------------------
cells.append(md(
    """## 5. Go / no-go decision against PLAN section 3.1 pass conditions

Pass conditions:

- **(a) Pass**: at least one image shows a statistically significant boulder-vs-
  surroundings difference (`p < 0.05`, `|d| > 0.3`) in **at least one** of
  IR/RED/BG or a band ratio, in **at least one of Test A or Test B**, AND the
  dust-confound test returns an interpretable result.
- **(b) Conditional pass**: significant differences but fully attributable to
  dust (partial correlation washes out the IR/BG signal).
- **(c) Fail**: no significant difference on truth labels.

The cell below computes the pass status from the loaded summaries and prints the
verdict. **Final write-up of the decision goes into [DECISIONS.md](../DECISIONS.md)**
once Brian signs off.
""",
    cell_id="verdict-md",
))

cells.append(code(
    """# Pass criterion: p < 0.05 AND |d| > 0.3 anywhere in Test A or Test B
THRESH_P = 0.05
THRESH_D = 0.3

def passes(df, p_col, d_col):
    if df.empty:
        return df
    return df[(df[p_col] < THRESH_P) & (df[d_col].abs() > THRESH_D)]

passing_a = passes(summary_a, "wilcoxon_p", "cohens_d_paired")
passing_b = passes(summary_b, "mannwhitney_p", "cohens_d")

print("=== Test A passing (feature, ObsId): ===")
if not passing_a.empty:
    print(passing_a[["obs_id", "feature", "n_pairs", "mean_diff",
                     "cohens_d_paired", "wilcoxon_p"]].to_string(index=False))
else:
    print("  none")

print("\\n=== Test B passing (feature, ObsId): ===")
if not passing_b.empty:
    print(passing_b[["obs_id", "feature", "n_rich", "n_poor",
                     "mean_rich", "mean_poor",
                     "cohens_d", "mannwhitney_p"]].to_string(index=False))
else:
    print("  none")

# Now interpret with dust confound
any_pass = (not passing_a.empty) or (not passing_b.empty)
print(f"\\nPLAN 3.1 (a) pass condition met (p<{THRESH_P}, |d|>{THRESH_D}, "
      f"any feature, any test, any image): {any_pass}")

# Dust-confound result: did `partial_r_IRoverBG|dust` lose significance?
if "partial_p_IRoverBG|dust" in dust_df.columns:
    survived = dust_df[(dust_df["partial_p_IRoverBG|dust"] < 0.05)
                       & (dust_df["partial_r_IRoverBG|dust"].abs() > 0.1)]
    print(f"\\nDust-controlled IR/BG signal survives in {len(survived)}/{len(dust_df)} images:")
    print(dust_df[["obs_id", "marginal_r_IRoverBG", "partial_r_IRoverBG|dust",
                   "partial_p_IRoverBG|dust"]].to_string(index=False))

# Final verdict string
if any_pass:
    if "partial_p_IRoverBG|dust" in dust_df.columns and not survived.empty:
        verdict = "PASS (a) -- composition signal detected (dust-controlled)"
    elif "partial_p_IRoverBG|dust" in dust_df.columns:
        verdict = "CONDITIONAL PASS (b) -- signal fully dust-attributable (relative age, not composition)"
    else:
        verdict = "PASS (a) -- signal detected; dust analysis inconclusive"
else:
    verdict = "FAIL (c) -- no significant boulder-vs-surroundings signal"
print(f"\\n>>> VERDICT: {verdict}")
""",
    cell_id="verdict-code",
))

# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "geospatial", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH}")
