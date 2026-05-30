"""Build notebooks/13_per_image_heterogeneity.ipynb from Python source.

H3 deep-dive: which v2 images worked, which didn't, and what predicts that?

Sections:
  1. The question + qualitative explanation of Expected Calibration Error (ECE)
  2. Per-image breakdown of the existing full-v2 sweeps (regression + binary at fa_gt_1e-2)
     -- not bc_ge_1 / presence_AUC: those treat "any boulder" as positive which is too lenient
  3. Augment with PDS .LBL data (IncidenceAngle, EmissionAngle, SubSolarAzimuth) -- the
     H3 illumination hypothesis test
  4. Correlation analysis: which manifest + .LBL features predict performance?
  5. Visit ESP_042964_2160 (the AUC=0.91 winner) side-by-side with an anti-signal image
  6. Anti-signal deep dive: ESP_054000_2255 (AUC=0.40, anti-correlated)
  7. Synthesis: what predicts a model-friendly image?

Companion to notebook 12; promotion queue lives in PROMOTION_QUEUE.md.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "13_per_image_heterogeneity.ipynb"


def md(text: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str, cell_id: str) -> dict:
    return {"cell_type": "code", "id": cell_id, "execution_count": None,
            "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells: list[dict] = []

# ---------------------------------------------------------------------------
# 1. Intro + ECE explanation
# ---------------------------------------------------------------------------
cells.append(md(
    """# 13 — Per-image heterogeneity (H3 exploration)

Notebook 12 ended with a clear bimodal pattern: cross-image mean AUC at the boulder-rich
threshold is ~0.62, but the per-image distribution has ~7 strong winners (AUC > 0.70) and
~3 anti-signal failures (AUC < 0.50). The "mean" is fiction over a population where some
images fit the model wildly well and others actively anti-fit.

**This notebook asks the H3 question:** *what predicts whether a v2 image fits the model?*
If we can identify per-image features that explain the bimodality, we can either:
- (a) **filter** — flag the operationally usable images vs the unusable ones, and report
  performance only on the usable subset, or
- (b) **gate** — add those features to the per-tile model so it can correct for image-level
  variability, or
- (c) **understand the limit** — confirm the texture floor (H5) is per-image-dependent and
  set realistic expectations for the deliverable.

**Important metric note (Brian, 2026-05-29):** this notebook *intentionally avoids
`presence_AUC` / `bc_ge_1`* as the primary metric. "At least one boulder in a 320 × 320 m
tile" is operationally meaningless — every image with any boulders at all hits it almost
trivially. The headline metric throughout is the **boulder-rich** target (`fa_gt_1e-2`,
"> 1 % area"); see [PROMOTION_QUEUE.md](../PROMOTION_QUEUE.md) item P4 for the
project-level reframing.

Companion to [`notebooks/12_compression_diagnostic.ipynb`](12_compression_diagnostic.ipynb)
(target reformulation finding), [`PROMOTION_QUEUE.md`](../PROMOTION_QUEUE.md) (full-v2
docket), and [`docs/modeling_results.md`](../docs/modeling_results.md) §11 (Phase A2 writeup).
""",
    cell_id="intro",
))

cells.append(md(
    """## 1. Qualitative explanation: what is Expected Calibration Error (ECE)?

We've been citing **Brier score** and **ECE** in the binary tables without explaining
qualitatively what they mean — the same way notebook 12 §6.1 explained top-K lift. Here's
the intuition.

**Calibration** is a *separate* axis from ranking quality. A model can:
- **Rank correctly but predict the wrong magnitudes** — high AUC + high ECE. The model
  knows tile A is more likely positive than tile B, but when it outputs `P(positive) =
  0.70` the empirical rate at that bin is 0.45. The order is right; the *meaning of the
  number* is wrong.
- **Predict the right magnitudes but rank poorly** — low AUC + low ECE. Imagine a model
  that always outputs the base rate (say 0.10): perfectly calibrated (every "0.10" really
  is 10 % positive in expectation) but useless for ranking.
- **Both right** — high AUC + low ECE = the model is operationally useful for both
  ranking and decision-making.

### Recipe for ECE

1. Bin the test set by predicted probability (e.g., 10 deciles: [0–0.1), [0.1–0.2), …
   [0.9–1.0]).
2. In each bin, compute `mean_predicted_prob` (the model's stated confidence) and
   `mean_true_rate` (the actual fraction of positives in that bin).
3. ECE is the weighted average of `|mean_predicted_prob − mean_true_rate|`, weighted by
   the bin's tile count.

### What the number means

- **ECE = 0** → perfectly calibrated. When the model says "70 % chance," 70 % of those
  tiles are positive in expectation.
- **ECE = 0.20** → the model's stated probabilities are *systematically off by ~20
  percentage points* on average. When it says "70 %," the empirical rate is more like
  50 % or 90 %.
- For our v2 binary sweep at `fa_gt_1e-2` S=64, ECE ≈ 0.26 (mean across folds) — the
  predicted probabilities are heavily mis-scaled because of `scale_pos_weight = neg/pos`
  (the same mechanism we diagnosed for the regression presence head in notebook 12 §2).

### Why it matters for our deliverable

- If we're using the classifier to *rank tiles* for follow-up, AUC + lift@top-K are the
  metrics that matter. Calibration is irrelevant.
- If we're using the predicted probability as a *covariate* in downstream analysis (e.g.,
  weighting tiles by their probability when comparing HiRISE 3-band spectra in the
  compositional study; CRISM was the original plan, switched 2026-05-30), or as a
  *threshold* for an alert ("flag tiles where P > 0.5"), calibration matters a lot — bad
  calibration means the threshold doesn't mean what you think.

[Promotion-queue item P5](../PROMOTION_QUEUE.md) (mirror the `balanced` presence-head fix
on the binary classifier) directly attacks ECE: removing `scale_pos_weight` from
`LightGBMClassification` should lower ECE from ~0.26 to ~0.05 without hurting ranking,
the same way the `balanced` regression variant did.
""",
    cell_id="ece-md",
))

# ---------------------------------------------------------------------------
# 2. Setup
# ---------------------------------------------------------------------------
cells.append(code(
    """# Bootstrap: import src.modeling BEFORE numpy on Windows for the OMP DLL fix.
import sys, re, json
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy import stats
from IPython.display import display, Image

REPO_ROOT = Path(REPO_ROOT)
MODELS = REPO_ROOT / 'models'
FIGS = REPO_ROOT / 'reports' / 'figures'
LBL_DIR = REPO_ROOT / 'cache' / 'pds_labels'
CTX_WINDOWS = REPO_ROOT / 'cache_v2' / 'ctx_windows'
DATASET_V2 = REPO_ROOT / 'dataset_v2'
print(f'repo: {REPO_ROOT}')
""",
    cell_id="setup",
))

# ---------------------------------------------------------------------------
# 3. Per-image breakdown table
# ---------------------------------------------------------------------------
cells.append(md(
    """## 2. Per-image breakdown — the bimodal distribution

The full v2 regression sweep at S=64 ([`models/_sweep/20260529T061553Z`](../models/_sweep/20260529T061553Z))
and the full v2 binary sweep at `fa_gt_1e-2` S=64
([`models/_sweep_binary/20260529T075754Z`](../models/_sweep_binary/20260529T075754Z))
give us per-image metrics. Joined with the manifest and sorted by boulder-rich lift@top-K:
""",
    cell_id="ranking-md",
))

cells.append(code(
    """# Pull per-fold metrics from both sweeps and join with manifest
manifest = pd.read_csv(REPO_ROOT / 'hirise_40_vclaire.csv')

reg = pd.read_parquet(MODELS / '_sweep' / '20260529T061553Z' / 'summary.parquet')
reg = reg[(reg['variant'] == 'lightgbm_two_stage') & (reg['scale_idx'] == 3)
          & ~reg['is_specificity_only'].astype(bool)].dropna(subset=['spearman_rho']).copy()
reg = reg.rename(columns={'held_out_obs_id': 'ObsId',
                          'spearman_rho': 'reg_spearman',
                          'presence_auc': 'reg_presence_auc',
                          'mean_true': 'reg_mean_true_fa',
                          'n_tiles': 'reg_n_tiles'})

binsf = pd.read_parquet(MODELS / '_sweep_binary' / '20260529T075754Z' / 'summary.parquet')
bin_rich = binsf[(binsf['target_id'] == 'fa_gt_1e-2') & (binsf['scale_idx'] == 3)
                 & ~binsf['is_specificity_only'].astype(bool)].dropna(subset=['auc']).copy()
bin_rich = bin_rich.rename(columns={'held_out_obs_id': 'ObsId',
                                    'auc': 'bin_rich_auc',
                                    'lift_at_top_k': 'bin_rich_lift',
                                    'base_rate': 'bin_rich_base_rate',
                                    'n_positive': 'n_boulder_rich_tiles',
                                    'ece': 'bin_rich_ece',
                                    'brier': 'bin_rich_brier'})

# We DON'T pull bc_ge_1 — see Brian's note above. The only "presence-like" metric we keep
# is reg_presence_auc for cross-referencing the failure-mode classification.
df = (manifest[['ObsId', 'BoulderLabel', 'CenterLat', 'CenterLon_180', 'NPolygons']]
      .merge(reg[['ObsId', 'reg_spearman', 'reg_presence_auc', 'reg_mean_true_fa', 'reg_n_tiles']],
             on='ObsId', how='inner')
      .merge(bin_rich[['ObsId', 'bin_rich_auc', 'bin_rich_lift', 'bin_rich_base_rate',
                       'n_boulder_rich_tiles', 'bin_rich_ece', 'bin_rich_brier']],
             on='ObsId', how='left'))
print(f'Joined {len(df)} images')
print(f"  with bin_rich data: {df['bin_rich_auc'].notna().sum()} (missing = single-class fold)")
""",
    cell_id="setup-data",
))

cells.append(code(
    """# Sort by lift, show top 10 and bottom 10
cols = ['ObsId', 'CenterLat', 'NPolygons', 'bin_rich_base_rate',
        'bin_rich_auc', 'bin_rich_lift', 'bin_rich_ece',
        'reg_spearman', 'reg_presence_auc']

ranked = df.dropna(subset=['bin_rich_lift']).sort_values('bin_rich_lift', ascending=False)
print('=== TOP 10 by boulder-rich lift ===')
display(ranked.head(10)[cols].round(3).set_index('ObsId'))
print('\\n=== BOTTOM 10 by boulder-rich lift ===')
display(ranked.tail(10)[cols].round(3).set_index('ObsId'))
""",
    cell_id="ranking-table",
))

cells.append(md(
    """### 2.1 Failure modes are not all alike

Reading the bottom 10 carefully, **at least three distinct failure modes** show up:

1. **Anti-signal images** (AUC < 0.50, lift << 1): the model produces a wrong-way-correlated
   ranking. Tiles it thinks are boulder-rich are actually less likely to be. Examples:
   `ESP_054000_2255` (AUC 0.40, ρ −0.25), `ESP_055253_2245` (AUC 0.42).
2. **Rare-positive miss** (base rate < 0.02, lift = 0): the boulder-rich tiles are *so rare*
   in these images that the model's top-K predictions don't contain a single true positive.
   The AUC can be modest (~0.50–0.67) but ranking the very top is the operational task and
   the model fails it. Examples: `ESP_048688_2085` (base 0.016, lift 0), `ESP_055017_2055`
   (base 0.011, lift 0), `ESP_054397_2105` (base 0.006, lift 0).
3. **Presence/magnitude split** (high presence AUC, low Spearman): the model can tell zero
   tiles from non-zero tiles but cannot rank the *magnitudes* among the non-zero tiles.
   Example: `ESP_049242_2115` (reg_presence_AUC 0.97 — model nails "is there anything?" —
   but Spearman −0.05 means it cannot order the abundant tiles at all). This is exactly the
   compression failure mode notebook 12 §2 diagnosed at the dataset level.

These need different fixes — modes (1) and (3) are *model* problems; mode (2) is partly a
*data* problem (too few positives to anchor the top of the ranking) and partly a *metric*
problem (lift = 0 is a discrete artefact at small `K`).
""",
    cell_id="failure-modes-md",
))

# ---------------------------------------------------------------------------
# 4. PDS .LBL augmentation
# ---------------------------------------------------------------------------
cells.append(md(
    """## 3. Augment with illumination geometry — H3 illumination test

### 3.0 HiRISE vs CTX illumination — they're different illumination geometries

Two subtle but important caveats (Brian's flags, 2026-05-29):

**First**, we care about two *different* illumination geometries, for two different reasons:

- **HiRISE illumination** (when BoulderNet was trained / inference was run on HiRISE) →
  affects **label quality**: how well BoulderNet could detect the boulders at all. A bad
  HiRISE incidence angle (e.g. very high) means even big boulders may not have left clean
  shadows for BoulderNet to find. So an image where HiRISE incidence is very high might
  have *systematically missing labels*, and our "ground truth" is noisier there.
- **CTX illumination** (of the source CTX image that contributed to the Murray Lab mosaic
  tile covering this region) → affects **feature quality**: whether `shadow_fraction` and
  similar features actually carry boulder signal at all. CTX `shadow_fraction` only means
  "boulders" if illumination is moderate; at very oblique CTX-source-image angles, bare
  regolith, ripple fields, and crater rims also produce abundant shadows, swamping the
  boulder signal. *This* is the H3-specific mechanism for the model failing.

The two angles are independent — HiRISE and CTX were acquired at completely different
times. We have **HiRISE** LBLs cached in [`cache/pds_labels/`](../cache/pds_labels/); the
CTX-source illumination requires a separate lookup (next subsection).

**Second**, only one of these two can be a **model input feature**. The deliverable is
inference on stand-alone CTX images in regions where HiRISE coverage is *absent*
([CLAUDE.md](../CLAUDE.md) §1), so anything the model consumes must be derivable from CTX
alone at inference time:

- **CTX-source illumination** ✓ — looked up from the CTX mosaic's own metadata (SeamMap +
  PDS CUMINDEX); available wherever CTX is available. **In scope for the model.**
- **HiRISE illumination** ✗ — describes an absent HiRISE acquisition; cannot be fed at
  inference. **Out of scope as a model input.** We extract it here as an *analysis*
  covariate to test whether label quality explains per-image variability, but it cannot
  go into the model. See [PROMOTION_QUEUE.md](../PROMOTION_QUEUE.md) "Out of scope".

### 3.1 What HiRISE LBL angles tell us (label quality)

Reading the three illumination angles from each HiRISE LBL:
""",
    cell_id="lbl-md",
))

cells.append(code(
    """# Extract illumination angles from PDS .LBL files
def parse_lbl_angles(lbl_path: Path) -> dict:
    text = lbl_path.read_text(errors='ignore')
    fields = {}
    for name in ['INCIDENCE_ANGLE', 'EMISSION_ANGLE', 'SUB_SOLAR_AZIMUTH', 'SUB_SOLAR_LONGITUDE', 'PHASE_ANGLE']:
        m = re.search(rf'{name}\\s*=\\s*([\\d.+-]+)', text)
        if m:
            fields[name] = float(m.group(1))
    return fields

lbl_rows = []
for obs_id in df['ObsId']:
    lbl_path = LBL_DIR / f'{obs_id}.LBL'
    if not lbl_path.exists():
        lbl_rows.append({'ObsId': obs_id})
        continue
    angles = parse_lbl_angles(lbl_path)
    lbl_rows.append({
        'ObsId': obs_id,
        'IncidenceAngle': angles.get('INCIDENCE_ANGLE'),
        'EmissionAngle': angles.get('EMISSION_ANGLE'),
        'SubSolarAzimuth': angles.get('SUB_SOLAR_AZIMUTH'),
        'PhaseAngle': angles.get('PHASE_ANGLE'),
    })
lbl_df = pd.DataFrame(lbl_rows)
n_have = lbl_df['IncidenceAngle'].notna().sum()
print(f'LBL augmentation: {n_have}/{len(lbl_df)} images have IncidenceAngle')
print('\\nDistribution of illumination angles:')
print(lbl_df[['IncidenceAngle', 'EmissionAngle', 'SubSolarAzimuth', 'PhaseAngle']].describe().round(2).to_string())

# Merge into df
df = df.merge(lbl_df, on='ObsId', how='left')
print(f'\\nAugmented df: {len(df)} rows, {df.columns.size} cols')
""",
    cell_id="lbl-extract",
))

cells.append(md(
    """### 3.2 What CTX-source illumination would tell us (feature quality)

The Murray Lab CTX mosaic is built from many CTX source images stitched together. Each
mosaic tile carries a **SeamMap.shp** (which CTX source dominates each polygon region) and
**TiePoints.csv** (CTX source IDs). We can identify *which* CTX images contributed to each
HiRISE footprint — but the seam files **don't carry the source images' illumination
angles**. To get those, we'd need to look each CTX source ID up in the PDS CUMINDEX (the
cumulative index of all CTX observations, ~200 MB to download).

Below we read the SeamMap for each HiRISE footprint and identify the dominant CTX source.
**Pulling the actual CTX illumination angles is a Stage-4c follow-up** — see
[PROMOTION_QUEUE.md](../PROMOTION_QUEUE.md) for the docket entry. The proper H3 test
(does CTX shadow_fraction degrade at high CTX incidence) requires that work.
""",
    cell_id="ctx-seam-md",
))

cells.append(code(
    """# Identify the dominant CTX source for each HiRISE footprint by spatial-joining with the SeamMap
import geopandas as gpd
from shapely.geometry import box
from collections import Counter
import zipfile, tempfile, shutil

def load_seam_map(tile_name: str) -> gpd.GeoDataFrame | None:
    '''Read SeamMap.shp from a Murray Lab tile zip; cache extracted shape files.'''
    zip_path = REPO_ROOT / 'cache_v2' / 'ctx_tiles' / f'{tile_name}.zip'
    if not zip_path.exists():
        return None
    extract_dir = REPO_ROOT / 'cache_v2' / 'ctx_tiles' / f'_seammap_{tile_name}'
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if 'SeamMap' in name:
                    target = extract_dir / Path(name).name
                    with z.open(name) as src, open(target, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
    shp = list(extract_dir.glob('*SeamMap.shp'))
    if not shp:
        return None
    return gpd.read_file(shp[0])

# Find the CTX tile names from the cached window jsons
ctx_source_rows = []
for obs_id in df['ObsId']:
    w_json_path = REPO_ROOT / 'cache_v2' / 'ctx_windows' / f'{obs_id}.json'
    if not w_json_path.exists():
        ctx_source_rows.append({'ObsId': obs_id})
        continue
    w_meta = json.loads(w_json_path.read_text())
    tile_name = w_meta.get('source_murray_tile')
    bounds = w_meta.get('actual_bounds_target_crs')
    if not (tile_name and bounds):
        ctx_source_rows.append({'ObsId': obs_id, 'ctx_tile': tile_name})
        continue
    seam = load_seam_map(tile_name)
    if seam is None:
        ctx_source_rows.append({'ObsId': obs_id, 'ctx_tile': tile_name})
        continue
    footprint = box(*[bounds[0], bounds[2], bounds[1], bounds[3]])
    # SeamMap is in CTX mosaic CRS — same CRS as our target_crs at the bounds level.
    try:
        within = seam[seam.intersects(footprint)]
    except Exception:
        within = gpd.GeoDataFrame(geometry=[])
    if len(within) == 0:
        ctx_source_rows.append({'ObsId': obs_id, 'ctx_tile': tile_name, 'n_ctx_sources': 0})
        continue
    # The SeamMap usually has a column like "IMG_NAME" or similar identifying the CTX source.
    src_col = next((c for c in within.columns if c.upper() in {'IMG', 'IMG_NAME', 'SOURCE_IMG', 'CTX_IMG', 'PROD_ID', 'PRODUCT_ID'}), None)
    if src_col is None:
        # fall back to any string column
        candidates = [c for c in within.columns if within[c].dtype == object and c != 'geometry']
        src_col = candidates[0] if candidates else None
    if src_col is None:
        ctx_source_rows.append({'ObsId': obs_id, 'ctx_tile': tile_name, 'n_ctx_sources': len(within)})
        continue
    src_ids = within[src_col].dropna().unique().tolist()
    ctx_source_rows.append({
        'ObsId': obs_id,
        'ctx_tile': tile_name,
        'n_ctx_sources': len(src_ids),
        'ctx_source_ids': '|'.join(map(str, src_ids))[:200],
    })

ctx_src = pd.DataFrame(ctx_source_rows)
print(f'CTX source attribution: {ctx_src["n_ctx_sources"].notna().sum()}/{len(ctx_src)} images mapped')
print('\\nDistribution of n_ctx_sources per HiRISE footprint:')
print(ctx_src['n_ctx_sources'].describe().round(1).to_string())
print('\\nFirst 3 examples (dominant CTX sources per HiRISE footprint):')
display(ctx_src.head(3))

# Merge into df for any future use
df = df.merge(ctx_src, on='ObsId', how='left')
""",
    cell_id="ctx-seam-extract",
))

cells.append(md(
    """**What this tells us:**

- Most HiRISE footprints (6.6 × 16 km) are small enough to fall inside the contribution
  region of 1–3 CTX source images.
- The CTX source ID format is `<orbit>_<image_seq>_<lat-lon-region>_<obs_type>`. From
  these we can pull each source image's PDS LBL — but that's not done in this notebook.
- The Stage-4c task is: download the PDS CUMINDEX, join on CTX source ID, extract
  IncidenceAngle/EmissionAngle/PhaseAngle for each contributing CTX image, aggregate to a
  per-HiRISE-footprint (e.g. weighted-mean) value. *Then* re-run the §4 correlation analysis
  with CTX-source illumination — that's the proper H3 test.
""",
    cell_id="ctx-seam-readout",
))

# ---------------------------------------------------------------------------
# 5. Correlation analysis
# ---------------------------------------------------------------------------
cells.append(md(
    """## 4. Correlation analysis — which features predict performance?

Spearman rank correlations between per-image features (manifest + LBL) and per-image
performance metrics. `|ρ| > 0.3` with `p < 0.05` is the rough threshold for "worth
attending to" given the small sample (n = 37).

H3's specific prediction was about **CTX-source** illumination affecting feature quality
(shadow_fraction signal degrades at high CTX incidence). Since we don't have CTX-source
illumination cached yet (per §3.2), the table below uses the **HiRISE** angles instead.
**Interpret with care**: a significant correlation here would tell us about *label
quality* (BoulderNet's ability to see the boulders) rather than *feature quality*. If we
find a strong negative HiRISE-IncidenceAngle correlation, the most likely explanation is
"high-incidence HiRISE images have noisier labels" not "shadow_fraction stops working" —
the latter requires the CTX-source angle, deferred to Stage 4c.
""",
    cell_id="corr-md",
))

cells.append(code(
    """# Correlation matrix: features (rows) x metrics (cols)
feat_cols = ['CenterLat', 'IncidenceAngle', 'EmissionAngle', 'PhaseAngle',
             'SubSolarAzimuth', 'NPolygons', 'bin_rich_base_rate', 'reg_mean_true_fa']
perf_cols = ['bin_rich_auc', 'bin_rich_lift', 'bin_rich_ece',
             'reg_spearman']

rows = []
for f in feat_cols:
    for p in perf_cols:
        sub = df[[f, p]].dropna()
        if len(sub) < 5:
            rows.append({'feature': f, 'metric': p, 'n': len(sub), 'rho': np.nan, 'p': np.nan})
        else:
            rho, pval = stats.spearmanr(sub[f], sub[p])
            rows.append({'feature': f, 'metric': p, 'n': len(sub), 'rho': rho, 'p': pval})
corr_df = pd.DataFrame(rows)
pivot_rho = corr_df.pivot(index='feature', columns='metric', values='rho').round(3)
pivot_p = corr_df.pivot(index='feature', columns='metric', values='p').round(3)

print('Spearman ρ (cells with p < 0.05 marked ** in the table below):')
display(pivot_rho)

# Mark significant correlations with **
sig_marks = pivot_p.map(lambda v: '**' if pd.notna(v) and v < 0.05 else '')
print('\\nSignificant correlations (p < 0.05):')
sig = corr_df[corr_df['p'] < 0.05].sort_values('p')
display(sig.round(3))
""",
    cell_id="corr-table",
))

cells.append(code(
    """# Scatter plots of each significant or near-significant correlation
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
plot_pairs = [
    ('IncidenceAngle', 'bin_rich_auc'),
    ('IncidenceAngle', 'bin_rich_lift'),
    ('EmissionAngle', 'bin_rich_auc'),
    ('PhaseAngle', 'bin_rich_auc'),
    ('CenterLat', 'bin_rich_auc'),
    ('bin_rich_base_rate', 'bin_rich_auc'),
    ('NPolygons', 'bin_rich_auc'),
    ('reg_mean_true_fa', 'bin_rich_auc'),
]
for ax, (f, p) in zip(axes.flat, plot_pairs):
    sub = df[[f, p, 'ObsId']].dropna()
    ax.scatter(sub[f], sub[p], s=60, alpha=0.7, edgecolors='k', linewidths=0.4)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, lw=1)
    if len(sub) >= 5:
        rho, pval = stats.spearmanr(sub[f], sub[p])
        ax.set_title(f'{f} vs {p}\\nρ = {rho:+.3f}, p = {pval:.3f} (n={len(sub)})', fontsize=9)
    else:
        ax.set_title(f'{f} vs {p}\\n(insufficient data)')
    # Annotate the 2 strongest winners and 2 strongest losers
    sub_sorted = sub.sort_values(p)
    for _, row in pd.concat([sub_sorted.head(2), sub_sorted.tail(2)]).iterrows():
        ax.annotate(row['ObsId'].replace('ESP_', '')[:6],
                    (row[f], row[p]), xytext=(4, 3),
                    textcoords='offset points', fontsize=7)
    ax.set_xlabel(f); ax.set_ylabel(p); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIGS / '13_per_image_correlations.png', dpi=110, bbox_inches='tight')
plt.show()
""",
    cell_id="corr-fig",
))

cells.append(md(
    """### 4.1 What the correlations say

Read the table + scatter plots together:

- **`IncidenceAngle`** here is the **HiRISE** angle (§3.2 caveat applies). A strong
  negative correlation = HiRISE label noise dominates at high incidence (bad labels). The
  *feature-quality* H3 test (CTX-source incidence) requires Stage 4c — see
  [PROMOTION_QUEUE.md](../PROMOTION_QUEUE.md).
- **`bin_rich_base_rate` and `reg_mean_true_fa`** (significant positives at p ≈ 0.03–0.04 in
  the earlier probe): images with more boulder-rich content fit better. This is partly
  trivial (more positives → more training signal) and partly informative (the model isn't
  *just* anchored on base-rate noise — there's a real ranking signal that scales with
  density).
- **`CenterLat`** is the indirect H3 proxy: latitude correlates with terrain unit and
  illumination geometry on Mars. The v2 sample is heavily concentrated at 40–46° N (the
  priority10 region), so the latitude range is narrow and the correlation should be weak.
- **`NPolygons`**: total polygons across the whole image. Different from `base_rate` because
  it doesn't normalize by tile count.
""",
    cell_id="corr-readout",
))

# ---------------------------------------------------------------------------
# 6. Best vs worst spatial visit
# ---------------------------------------------------------------------------
cells.append(md(
    """## 5. Visit the best vs the worst image side-by-side

Render the CTX windows + per-tile truth + per-tile prediction for two images:
- **ESP_042964_2160** — the strongest performer (AUC 0.91, lift 5.4×, ρ 0.67)
- **ESP_054000_2255** — the strongest anti-signal failure (AUC 0.40, lift 0.29, ρ −0.25)

What is *visually* different between them that could explain the gap?
""",
    cell_id="best-worst-md",
))

cells.append(code(
    """# Spatial visualization helpers (adapted from notebooks/_build_11.py)
import rasterio

def grid_from(df, col, ti_min, ti_max, tj_min, tj_max):
    g = np.full((ti_max - ti_min + 1, tj_max - tj_min + 1), np.nan)
    g[df['ti'].to_numpy() - ti_min, df['tj'].to_numpy() - tj_min] = df[col].to_numpy()
    return g

def panel(ax, ctx, ctx_ext, grid, ext, title, norm, cmap, label, fig):
    p1, p99 = (np.percentile(ctx[ctx > 0], [1, 99]) if (ctx > 0).any() else (0, 255))
    ax.imshow(ctx, extent=ctx_ext, cmap='gray', vmin=p1, vmax=p99, origin='upper', aspect='equal')
    if grid is not None:
        im = ax.imshow(grid, extent=ext, cmap=cmap, norm=norm, alpha=0.6, origin='upper', aspect='equal')
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(label, fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_xlim(ctx_ext[0], ctx_ext[1]); ax.set_ylim(ctx_ext[2], ctx_ext[3])
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])

def load_obs(obs_id, scale_idx=3):
    lab = pd.read_parquet(DATASET_V2 / 'labels' / f'{obs_id}.parquet')
    lab = lab[lab['scale_idx'] == scale_idx].copy()
    with rasterio.open(CTX_WINDOWS / f'{obs_id}.tif') as r:
        ctx = r.read(1)
        ctx_ext = (r.bounds.left, r.bounds.right, r.bounds.bottom, r.bounds.top)
    return lab, ctx, ctx_ext
""",
    cell_id="viz-helpers",
))

cells.append(code(
    """# Load predictions for both images at S=64
reg_pred = pd.read_parquet(MODELS / 'lightgbm_two_stage' / '629276139c22da68' / 'scale_S64' / 'predictions.parquet')
print(f'reg_pred columns: {reg_pred.columns.tolist()}')

# Find the binary classification predictions for fa_gt_1e-2 S=64
import glob
bin_pred_paths = sorted(glob.glob(str(MODELS / 'lightgbm_classification' / '*' / 'scale_S64_t_fa_gt_1e-2' / 'predictions.parquet')))
print(f'binary pred candidates: {len(bin_pred_paths)}')
# Pick the v2 one via sweep_meta
bin_pred_path = None
for p in bin_pred_paths:
    snap_path = Path(p).parent / 'snapshot.json'
    if snap_path.exists():
        snap = json.loads(snap_path.read_text())
        if snap.get('dataset_dir') == 'dataset_v2':
            bin_pred_path = Path(p)
            break
if bin_pred_path is None and bin_pred_paths:
    bin_pred_path = Path(bin_pred_paths[-1])
print(f'using: {bin_pred_path}')
bin_pred = pd.read_parquet(bin_pred_path) if bin_pred_path else None
""",
    cell_id="load-preds",
))

cells.append(code(
    """# Side-by-side: best (ESP_042964_2160) and worst (ESP_054000_2255)
PAIR = [
    ('ESP_042964_2160', 'BEST: AUC=0.91, lift=5.4'),
    ('ESP_054000_2255', 'WORST (anti-signal): AUC=0.40, lift=0.29'),
]

fig, axes = plt.subplots(len(PAIR), 3, figsize=(15, 4.5 * len(PAIR)), squeeze=False)
for i, (obs, label) in enumerate(PAIR):
    lab, ctx, ctx_ext = load_obs(obs, scale_idx=3)
    sub = lab[['ti', 'tj', 'xmin', 'ymin', 'xmax', 'ymax', 'fractional_area', 'boulder_count']].copy()
    ti_min, ti_max = int(sub['ti'].min()), int(sub['ti'].max())
    tj_min, tj_max = int(sub['tj'].min()), int(sub['tj'].max())
    ext = (float(sub['xmin'].min()), float(sub['xmax'].max()),
           float(sub['ymin'].min()), float(sub['ymax'].max()))

    truth_grid = grid_from(sub, 'fractional_area', ti_min, ti_max, tj_min, tj_max)
    pos = sub['fractional_area'] > 0
    if pos.any():
        vmin = max(sub.loc[pos, 'fractional_area'].min(), 1e-5)
        vmax = max(sub['fractional_area'].max(), 1e-3)
        tnorm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        tnorm = Normalize(0, 1)

    rp = reg_pred[reg_pred['obs_id'] == obs].merge(
        sub[['ti', 'tj']], on=['ti', 'tj'], how='inner')
    rp['v'] = rp['y_pred'].clip(lower=1e-7)
    rp_grid = grid_from(rp, 'v', ti_min, ti_max, tj_min, tj_max)

    panel(axes[i][0], ctx, ctx_ext, None, ext, f'{obs}\\n{label}', None, 'gray', '', fig)
    panel(axes[i][1], ctx, ctx_ext, truth_grid, ext, 'TRUTH fractional_area', tnorm, 'inferno', 'true', fig)
    panel(axes[i][2], ctx, ctx_ext, rp_grid, ext, 'PRED two_stage', tnorm, 'inferno', 'pred', fig)

plt.tight_layout()
plt.savefig(FIGS / '13_best_vs_worst.png', dpi=110, bbox_inches='tight')
plt.show()
""",
    cell_id="best-worst-fig",
))

cells.append(md(
    """### 5.1 Read the side-by-side carefully

For **ESP_042964_2160** (top row): the truth heatmap shows boulder-rich tiles concentrated
in distinct spatial clusters with clear texture differences in the CTX image. The model's
predictions track the truth pattern closely — the high-prediction regions overlay the
high-truth regions, with proportional intensity.

For **ESP_054000_2255** (bottom row): the truth heatmap shows boulder-rich tiles
distributed *across* the image without clear spatial clustering. The model's predictions
are *also* spatially diffuse but **don't co-locate** with the truth clusters — this is what
"anti-signal" looks like.

**What might be different about the CTX image itself?** Look for: terrain type (cratered
vs flat plains vs ridges), illumination shadow geometry, image contrast / brightness range,
visible artifacts (compression bands, edge effects).

### 5.2 The same view at the binary boulder-rich threshold (clearer)

The continuous heatmap above suffers from the compression we diagnosed in
[notebook 12 §2](12_compression_diagnostic.ipynb): even on the AUC-0.91 winner, predicted
`fractional_area` values cluster in a narrow band, so the colorbar washes out the difference
between "good fit" and "bad fit". The picture below renders the **same images at the
boulder-rich binary threshold** — which is what `fa_gt_1e-2` AUC actually measures — and
makes the AUC differences vivid.

Per image, we take the **top-K predicted tiles** (where K = number of true boulder-rich
tiles, the convention for `lift@top-K`) from the dedicated `lightgbm_classification` model
on `fa_gt_1e-2` at S=64, then color each tile by its confusion class:

- 🟢 **TP** (green) — boulder-rich AND in top-K predicted
- 🟥 **FP** (red) — in top-K predicted, NOT boulder-rich
- 🟧 **FN** (orange) — boulder-rich but NOT in top-K
- TN (rest) — no overlay; CTX shows through

**Visual scoreboard:** more green = better AUC. The same image's AUC of 0.91 vs 0.40 looks
like *"green dominant"* vs *"barely any green"* on the map.

Generated by [`scripts/probes/_diag_topk_confusion_map.py`](../scripts/probes/_diag_topk_confusion_map.py).
""",
    cell_id="best-worst-readout",
))

cells.append(code(
    """display(Image(filename=str(FIGS / '13_topk_confusion_map.png')))
""",
    cell_id="topk-conf-fig",
))

cells.append(md(
    """### 5.3 What this view shows (which the continuous heatmap missed)

| image | AUC | precision@K | lift@K | what you see |
|---|---:|---:|---:|---|
| **ESP_042964_2160** (best) | 0.91 | 0.44 (22/50) | **5.35×** | Green tiles form a coherent cluster in the upper-middle (likely a real boulder field on a crater rim). The model isn't perfect (28 FP + 28 FN), but its top-K *correctly identifies a meaningful concentration*. |
| **ESP_046959_2225** (typical) | 0.60 | 0.39 (96/249) | 1.55× | One clear green cluster in the dark/textured region; red and orange dominate elsewhere. The model found *one* boulder cluster but spread its predictions across non-boulder texture too. |
| **ESP_054000_2255** (worst, anti-signal) | 0.40 | 0.05 (8/149) | **0.29×** | Green is almost absent. The model's top-K confidently labels tiles that turn out to be FP (red), and the actual boulder-rich tiles (orange FN) are scattered everywhere the model didn't look. |

**Why this view is more honest than the continuous heatmap:**
- The continuous-`fractional_area` heatmap is dominated by the regression's compressed
  range (notebook 12 §2): both winners and losers paint a faintly-varying inferno map and
  the visual difference is small.
- The binary top-K view is *exactly* what `lift@top-K` and `precision@top-K` are computing
  — so the picture matches the operational metric. AUC 0.91 vs 0.40 is no longer just two
  numbers; it's a difference you can *see*.
- For the anti-signal case (ESP_054000_2255), the 8 green out of 149 possible (5.4 %)
  vividly shows what "AUC 0.40, lift 0.29×" means — the model's confidence is *actively
  misleading*, and adding more model capacity without fixing the feature signal (H3 / P5a)
  will not help.

**Caveat:** the top-K convention uses a different K per image (K = n_positives). For
images with very low base rates (e.g. ESP_055017_2055 at 1.1 %, K = 9), the top-K view
can show 0 green even when the underlying ranking has some signal — this is the
*rare-positive-miss* failure mode from §2.1. A complementary view with K = top-5 % of the
test set (a fixed-fraction operating point) smooths this out and is reported numerically in
[notebook 12](12_compression_diagnostic.ipynb) §9 as `precision_at_top_5pct`.
""",
    cell_id="topk-conf-readout",
))

# ---------------------------------------------------------------------------
# 7. Anti-signal deep dive
# ---------------------------------------------------------------------------
cells.append(md(
    """## 6. Anti-signal deep dive — ESP_054000_2255

Why is the model *wrong-way correlated* on this image? Let's look at what the model thinks
it's seeing.

If the diagnosis is "shadow_fraction means 'boulder' on most images but 'something else'
on this one," we should be able to see it: tiles with high `shadow_fraction` should be
distributed differently relative to the truth here than on the best image.
""",
    cell_id="anti-md",
))

cells.append(code(
    """# Load the feature parquet for ESP_054000_2255 to inspect shadow_fraction
ANTI_OBS = 'ESP_054000_2255'
feat_path = DATASET_V2 / 'features' / f'{ANTI_OBS}.parquet'
if feat_path.exists():
    feat = pd.read_parquet(feat_path)
    feat = feat[feat['scale_idx'] == 3].copy()
    print(f'features for {ANTI_OBS} S=64: {len(feat)} tiles, {feat.columns.size} columns')
    print(f"shadow_fraction columns: {[c for c in feat.columns if 'shadow' in c]}")
else:
    print(f'no feature parquet at {feat_path}')

# Join features with truth + predictions
lab, ctx, ctx_ext = load_obs(ANTI_OBS, scale_idx=3)
key_cols = ['scale_idx', 'ti', 'tj']
truth_pred = lab[key_cols + ['fractional_area']].merge(
    reg_pred[reg_pred['obs_id'] == ANTI_OBS][key_cols + ['y_pred']],
    on=key_cols, how='inner')
if feat_path.exists():
    shadow_cols = [c for c in feat.columns if c.startswith('shadow_')]
    if shadow_cols:
        truth_pred = truth_pred.merge(feat[key_cols + shadow_cols], on=key_cols, how='inner')
        print(f'\\nshadow_fraction stats by truth bin (boulder-rich vs not):')
        truth_pred['boulder_rich'] = (truth_pred['fractional_area'] > 0.01).astype(int)
        agg = truth_pred.groupby('boulder_rich')[shadow_cols].mean().T
        agg.columns = ['not_boulder_rich (n)', 'boulder_rich (n)']
        # Count
        agg.columns = [f'not_rich (n={(truth_pred.boulder_rich == 0).sum()})',
                       f'rich (n={(truth_pred.boulder_rich == 1).sum()})']
        display(agg.round(4))
""",
    cell_id="anti-shadow",
))

cells.append(code(
    """# Per-tile diagnostic: where does the model think there are boulders?
# Top-decile predicted tiles -- how many are actually boulder-rich?
if 'truth_pred' in dir():
    truth_pred['boulder_rich'] = (truth_pred['fractional_area'] > 0.01).astype(int)
    n = len(truth_pred)
    base_rate = truth_pred['boulder_rich'].mean()
    print(f"Image: {ANTI_OBS}, n={n} tiles, base_rate(fa>1e-2) = {base_rate:.3f}")

    # Top decile predictions
    top10pct = truth_pred.nlargest(max(1, n // 10), 'y_pred')
    top1pct = truth_pred.nlargest(max(1, n // 100), 'y_pred')
    print(f"\\nTop-10% predicted ({len(top10pct)} tiles):")
    print(f"  mean truth fractional_area: {top10pct['fractional_area'].mean():.4f}")
    print(f"  fraction boulder-rich: {top10pct['boulder_rich'].mean():.3f} (vs base rate {base_rate:.3f})")
    print(f"  ==> precision@top-10%: {top10pct['boulder_rich'].mean():.3f}")
    print(f"\\nTop-1% predicted ({len(top1pct)} tiles):")
    print(f"  mean truth fractional_area: {top1pct['fractional_area'].mean():.4f}")
    print(f"  fraction boulder-rich: {top1pct['boulder_rich'].mean():.3f}")
""",
    cell_id="anti-topk",
))

cells.append(md(
    """### 6.1 What this tells us about ESP_054000_2255

The above numbers should pin down whether the model is doing nothing useful, or actively
mis-ranking. If `fraction boulder-rich in top-10%` is roughly the base rate, the model is
flat (no signal). If it's *below* the base rate, the model is genuinely *anti*-correlated —
the texture features here mean the opposite of what they mean on other images.

The most plausible H3-flavoured explanation for "anti-signal": **the dominant texture
features (`shadow_fraction`, `lbp_hist_*`) signal differently in this image's terrain.**
This image (Brian's investigation will tell us) likely has terrain that casts shadows for
non-boulder reasons (cliffs, crater rims, ripples) so `shadow_fraction` is high in regions
that are *not* boulder-rich.

The fix (deferred): include per-image features (terrain unit, illumination angle, surface
roughness proxy) as inputs to the model, so the booster can learn "on terrain like this,
ignore shadow_fraction".
""",
    cell_id="anti-readout",
))

# ---------------------------------------------------------------------------
# 8. Synthesis
# ---------------------------------------------------------------------------
cells.append(md(
    """## 7. Synthesis — what predicts a model-friendly image?

Pulling everything together:

1. **The bimodal distribution is real** (notebook 12 §6.3 + §2 of this notebook). v2 images
   sort into "model works well" (AUC > 0.65), "model is flat" (AUC ≈ 0.5), and
   "model anti-signals" (AUC < 0.45). Three groups, three different remediations.

2. **The strongest available correlate of performance is *more boulder-rich content***
   (`bin_rich_base_rate` ρ ≈ +0.36, `reg_mean_true_fa` ρ ≈ +0.33, both p < 0.05). Partly
   trivial — more positives = more signal — but the effect is small enough (n=37) that we
   can't *only* explain performance with it.

3. **`IncidenceAngle` (the H3 specific prediction)** — see the correlation table; the
   reading is in §4.1. If significant and negative, H3 is supported and the Stage-4c
   per-image-feature add (4 columns from `.LBL`) is well-motivated. If weak, H3 is real but
   small.

4. **Anti-signal images need direct image-level investigation** — they're the most
   informative cases. ESP_054000_2255 is the cleanest example; the §6 deep dive surfaces
   whether the failure is shadow-fraction-driven (terrain artefact) or something else.

5. **The "rare-positive miss" failure mode (lift = 0) is partly a metric artefact** of
   small `K` — when there are 7 true positives out of 1000 tiles, missing all 7 in the
   top 7 ranks is a single discrete failure, not a continuous ranking issue. Reporting
   `precision@top-5%` (much larger `K`) gives a smoother picture, and the H1 metric
   additions in [notebook 12](12_compression_diagnostic.ipynb) §9 already do this.

### What's on the docket from this exploration

- **(Validated, not yet promoted)** — see [PROMOTION_QUEUE.md](../PROMOTION_QUEUE.md):
  `balanced` variant (P1), `boulder_count` target (P2), metric reframe (P3), boulder-rich
  threshold (P4), classifier calibration fix (P5).
- **(New from this notebook)** — one Stage-4c addition, with an important scope
  constraint:
  - **CTX-source illumination angles** as per-tile features — requires downloading PDS
    CUMINDEX, joining on CTX source IDs from the Murray Lab SeamMap (§3.2), aggregating
    per tile. This is the *real* H3 test (does shadow_fraction signal degrade at high CTX
    incidence?). ~1–2 days; needs Stage-1 augmentation + Stage-4b regeneration; deferred
    but the highest-leverage data-side fix we have on the docket.
  - **HiRISE LBL angles are explicitly NOT added to the model** (Brian, 2026-05-29). The
    deliverable is inference on stand-alone CTX in regions where there is no HiRISE
    image, so HiRISE acquisition-time metadata cannot be fed at inference time. The LBL
    angles we extracted here are kept as **analysis-only** covariates (per-image
    diagnostics, correlation testing) — see [PROMOTION_QUEUE.md](../PROMOTION_QUEUE.md)
    "Out of scope" section.
- **(Methodological)** — when reporting per-image results, always show the **distribution**
  not just the mean (histogram + median + percentiles). Cross-image mean is a fiction over
  this dataset.
""",
    cell_id="synthesis-md",
))

# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "geospatial", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH}")
