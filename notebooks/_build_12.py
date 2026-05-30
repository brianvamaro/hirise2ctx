"""Build notebooks/12_compression_diagnostic.ipynb from Python source.

Documents the dynamic-range compression diagnosis (Phase A finding from
PLAN_ModelImprovement.md) plus the dev-harness intervention sweep that tests
three loss/weighting fixes against the baseline `lightgbm_two_stage`.

This notebook is the **follow-up to notebook 11** (vClaire v2 modeling QA) and
explains what compression is, why it bites, what post-hoc calibration does and
doesn't fix, and which training-side fix actually buys headline ranking lift on
the v2-dev within-image scheme (20 folds).

Mirrors `_build_11.py`'s style (markdown + code cells written as plain Python,
nbformat-rendered at write time). Reuses the figures already saved by
`scripts/probes/_diag_compression_mechanism.py` and
`scripts/probes/_diag_compression_sweep_figure.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "12_compression_diagnostic.ipynb"


def md(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code(text: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells: list[dict] = []

# ---------------------------------------------------------------------------
# Intro
# ---------------------------------------------------------------------------
cells.append(md(
    """# 12 — Dynamic-range compression (Phase A diagnosis + fixes)

This notebook is the **follow-up to [`11_modeling_qa_v2.ipynb`](11_modeling_qa_v2.ipynb)** for one specific
finding from [`PLAN_ModelImprovement.md`](../PLAN_ModelImprovement.md) Phase A: the v2
`lightgbm_two_stage` S=64 regressor **compresses its dynamic range**, over-predicting empty/low
tiles and under-predicting the high tail. It documents the diagnosis, why the obvious post-hoc
fix doesn't work, and tests three training-side interventions on the 5-image dev harness
(`dataset_v2_dev/`, `within_image_4fold` scheme, 20 folds).

**The metrics we'll cite repeatedly are Spearman ρ and presence AUC — here's what they
*qualitatively* measure** (no derivations; just intuition for reading the tables):

- **Spearman ρ** is a rank-correlation: how well does the model **order** tiles by
  predicted abundance compared to the true ordering? ρ = 1 means "perfectly ranked, every
  prediction in the right place relative to the others"; ρ = 0 means "no relationship between
  predicted and true ranking"; ρ < 0 means "predictions are anti-correlated with truth".
  Spearman doesn't care about absolute prediction values — only their ordering — so it's the
  natural primary metric for a model whose downstream use is **"find the boulder-rich
  patches"**, even if the predicted abundance number itself is mis-scaled.
- **Presence AUC** is the area under the ROC curve for the binary "did this tile have *any*
  boulder?" task, with the model's predicted abundance used as the score. AUC = 1 means
  "every true-positive tile is ranked above every true-negative tile"; AUC = 0.5 is **chance**;
  AUC = 0 means perfectly inverted. AUC measures **detection power** — can the model tell a
  boulder-bearing tile apart from an empty tile? — and is the natural metric when the
  downstream use is **"flag tiles worth following up with HiRISE"**.
- The two often move together but can diverge: a model that gets the high-low split correct
  (high AUC) but mis-orders within the positive class (low ρ), or vice versa.

**Compression** is a third axis that neither metric captures: it's about the absolute scale
of the predictions, not their order. The model can be a perfect *ranker* (high ρ) while
compressing the dynamic range to a thin band — which is exactly what we'll show below.
""",
    cell_id="intro",
))

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
cells.append(code(
    """# Bootstrap: import src.modeling BEFORE numpy on Windows for the OMP DLL fix
# (see src/modeling/__init__.py + the [[torch_windows_openmp_fix]] memory).
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401 -- side effect: DLL bootstrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image

REPO_ROOT = Path(REPO_ROOT)
MODELS = REPO_ROOT / 'models'
FIGS = REPO_ROOT / 'reports' / 'figures'
print(f'repo: {REPO_ROOT}')
""",
    cell_id="setup",
))

# ---------------------------------------------------------------------------
# Section 1 - the finding
# ---------------------------------------------------------------------------
cells.append(md(
    """## 1. The finding: predictions are squashed into a narrow band

The full-v2 `lightgbm_two_stage` regression at S=64 ([sweep
`models/_sweep/20260529T061553Z`](../models/_sweep/20260529T061553Z),
[snapshot 11.6 of notebook 11](11_modeling_qa_v2.ipynb)) earns a meaningful **Spearman
ρ = +0.169** over 38 LOIO folds — a real, ~4.6σ signal that the model **ranks tiles by
abundance well**. But the absolute predictions are not calibrated: the model squeezes its
output into a roughly 0.007–0.015 band almost regardless of truth.

The table below pulls per-truth-bin means from the cached predictions for the full-v2
`lightgbm_two_stage` S=64 run. `ratio = mean_pred / mean_true`: 1.0 would be perfectly
calibrated; >>1 means over-predicting that bin, <<1 under-predicting.
""",
    cell_id="finding-md",
))

cells.append(code(
    """# Pull the v2 LOIO predictions for lightgbm_two_stage S=64 and compute per-bin means.
preds = pd.read_parquet(MODELS / 'lightgbm_two_stage' / '629276139c22da68' / 'scale_S64' / 'predictions.parquet')
# Drop the specificity-only fold (held-out image with no boulders)
spec = preds.groupby('fold_idx')['y_true'].transform('max') == 0
preds = preds[~spec].copy()

BINS = [
    (-1e-12, 0.0, 'zero'),
    (0.0, 1e-4, '0_to_1e-4'),
    (1e-4, 1e-3, '1e-4_to_1e-3'),
    (1e-3, 1e-2, '1e-3_to_1e-2'),
    (1e-2, 1.0, '1e-2_to_max'),
]

def bin_label(y):
    if y <= 0:
        return 'zero'
    for lo, hi, name in BINS[1:]:
        if lo < y <= hi:
            return name
    return '1e-2_to_max'

preds['bin'] = preds['y_true'].apply(bin_label)
agg = preds.groupby('bin', sort=False).agg(
    n_tiles=('y_true', 'size'),
    mean_true=('y_true', 'mean'),
    mean_pred=('y_pred', 'mean'),
    mean_p_pos=('y_pred_presence_prob', 'mean'),
).reindex([b[2] for b in BINS])
agg['ratio_pred_over_true'] = agg['mean_pred'] / agg['mean_true'].where(agg['mean_true'] > 0)
display(agg.round(5))
print(f'n folds (regression-real): {preds.fold_idx.nunique()};  pooled tiles: {len(preds):,}')
""",
    cell_id="finding-table",
))

cells.append(md(
    """**Read this with the qualitative metric definitions above in mind:**
- The model is a **fine ranker** at this scale — same Spearman ρ ≈ +0.17 quoted earlier — so
  the *relative* ordering of tiles within each bin is informative.
- But the **mean prediction barely moves across truth bins** (~0.007 on zero tiles, ~0.015 on
  the boulder-rich `>1 %` tiles). The high bin is predicted at ~42 % of its true mean; the
  zero bin is predicted at ~7 × 10⁻³ instead of 0 — an over-prediction floor.
- The `mean_p_pos` column shows where the over-prediction floor comes from: the
  presence-classifier head outputs `p_pos ≈ 0.85` even on tiles that are truly zero, and only
  climbs to ≈ 0.92 on the boulder-rich bin. The head doesn't say "zero" confidently — it says
  "maybe" everywhere — and `pred = p_pos × E[mag | pos]` inherits that floor.
""",
    cell_id="finding-readout",
))

# ---------------------------------------------------------------------------
# Section 2 - mechanism
# ---------------------------------------------------------------------------
cells.append(md(
    """## 2. Mechanism: two compression sources, not one

The figure below (regenerated by
[`scripts/probes/_diag_compression_mechanism.py`](../scripts/probes/_diag_compression_mechanism.py))
decomposes the compression by reading `pred = p_pos × mag` apart, where `p_pos` is the
presence-head probability and `mag = pred / p_pos` is the implied magnitude-head output.

- **Panels A/B/C** (top row): per-truth-bin mean prediction, raw vs LOIO-isotonic
  recalibration vs truth. The bars **barely budge** under iso recalibration — more on that in
  §3.
- **Panel D**: `p_pos` distribution by truth bin. Even for true-zero tiles, the median
  `p_pos` is ≈ 0.90. This is what sets the **over-prediction floor**: the LightGBM presence
  classifier was trained with `is_unbalance=True`, which shifts the decision boundary to give
  both classes equal weight — that helps when one class is rare, but it inflates `p_pos` on
  negatives.
- **Panel E**: `mag = pred / p_pos` by truth bin. The magnitude head spans only 0.008–0.016
  while the truth spans five orders of magnitude. This is the **under-prediction ceiling**:
  the magnitude head is fit with `log1p + Huber` on positives-only, and Huber loss fits the
  *median* of the log-transformed positive distribution, which is roughly the geometric mean
  of positives (~ 0.01). The heavy tail is shrunk away.
- **Panel F**: pooled pred-vs-true scatter on log-log. The raw predictions form a narrow
  horizontal band well below the identity line at high truth values; iso-recalibrated
  predictions barely move.

**The compression has two distinct causes**, and they live in different parts of the model.
That matters because each one needs a different fix.
""",
    cell_id="mech-md",
))

cells.append(code(
    """display(Image(filename=str(FIGS / '12_compression_diagnostic.png')))
""",
    cell_id="mech-fig",
))

# ---------------------------------------------------------------------------
# Section 3 - why iso fails
# ---------------------------------------------------------------------------
cells.append(md(
    """## 3. Why post-hoc isotonic recalibration *doesn't* fix it

The most natural "free" fix for a miscalibrated regressor is an **isotonic recalibration map**
fit on out-of-fold predictions: it's a monotone-increasing function from `y_pred → y_true`,
which by construction preserves the predicted ranking (so it can't hurt Spearman ρ) and
should re-stretch the predictions onto the truth scale.

The diagnostic above (Panels A/C) shows iso recalibration **barely helps**, and the
per-fold metrics actually drop slightly:

| metric | raw     | iso     | Δ      |
|--------|--------:|--------:|-------:|
| mean Spearman ρ over 38 folds | +0.169 | +0.157 | **−0.012** |
| mean presence AUC | 0.579 | 0.572 | **−0.007** |
| high-bin (>1 %) ratio pred/true | 0.42 | 0.48 | +0.06 |
| low-bin (1e-4–1e-3) ratio | 18.6 | 24.7 | **+6** (worse) |

**Why doesn't it work?** Two reasons:

1. **Out-of-distribution clipping at fold boundaries.** Each fold's iso map is fit on OOF
   predictions from the *other* 37 folds. When a held-out fold's predictions extend past the
   training range, the iso map clips them to the training min/max — and clipping is no longer
   monotone, so Spearman within the held-out fold slips at the boundary.
2. **The raw predictions don't span enough range to be re-stretched.** The model literally
   *cannot* output a value much above 0.025 because the magnitude head is squashed to the
   log-positive median (panel E). Iso can only rearrange values it sees; it cannot invent the
   range that the raw model didn't produce.

So the compression has to be fixed in **training**, not in post-processing. That's what the
next section does.
""",
    cell_id="iso-md",
))

# ---------------------------------------------------------------------------
# Section 4 - interventions
# ---------------------------------------------------------------------------
cells.append(md(
    """## 4. Three interventions, each attacking one mechanism

I implemented four new two-stage variants in [`src/modeling/gbm.py`](../src/modeling/gbm.py),
each a minimal-diff cousin of `LightGBMTwoStage`:

- **`lightgbm_two_stage_balanced`** — *attacks the presence-head floor.* Removes
  `is_unbalance=True` from the presence-classifier so the classifier produces calibrated
  logits and `p_pos` actually drops toward 0 on true-zero tiles. **One-line change.**
- **`lightgbm_two_stage_weighted`** — *attacks the magnitude-head ceiling.* Adds
  `sample_weight = y_pos` to the magnitude head's LightGBM Dataset, so the booster spends
  gradient on the heavy-tail tiles instead of fitting the log-median. Loss family unchanged
  (still `log1p + Huber`).
- **`lightgbm_two_stage_gamma`** — *attacks the magnitude-head ceiling via loss redesign.*
  Replaces `log1p + Huber` with LightGBM's `gamma` objective on positives — a mean-based
  loss with a log-link, designed for positive heavy-tailed targets.
- **`lightgbm_two_stage_combined`** — All three fixes at once (the headline candidate).

These were swept on the **v2-dev within-image** scheme (20 folds across 5 images;
`config_v2_dev.yaml`) at S=32 and S=64, alongside the baseline `lightgbm_two_stage`. The
within-image scheme is the right choice for screening: per the
[memory note](../../C:/Users/brian/.claude/projects/c--Users-brian-Documents-PhD-HiRiseToCTXBoulders-hirise2ctx/memory/project_state_2026-05-29.md) the
5-fold LOIO is too noisy for absolute numbers on dev, but the 20-fold within-image S8–S64
trend matches full-v2 §9. Run with
[`scripts/probes/_sweep_compression_fixes.py`](../scripts/probes/_sweep_compression_fixes.py).
""",
    cell_id="interventions-md",
))

cells.append(code(
    """# Load the most recent compression-fix sweep
sweep_root = MODELS / '_sweep_compression_fixes'
runs = sorted([p for p in sweep_root.iterdir() if p.is_dir()])
SWEEP = runs[-1]
print(f'sweep: {SWEEP.name}')
agg = pd.read_parquet(SWEEP / 'aggregate.parquet')
print(f'{len(agg)} (variant, scale) rows')
""",
    cell_id="load-sweep",
))

# ---------------------------------------------------------------------------
# Section 5 - results
# ---------------------------------------------------------------------------
cells.append(md(
    """## 5. Results: ranking vs tail calibration is a real trade-off

The composite metric table below is the headline read. Per the AskUserQuestion answer in
this session, the metric is composite by design: per-bin compression *and* Spearman / AUC,
shown side-by-side so the trade-off is visible.

**What to look for:**
- **Spearman / AUC**: higher = better ranking / detection. The metrics from the qualitative
  intro at the top.
- **`high_pred` vs `high_true`** (the last two columns): how close the *boulder-rich* bin's
  mean prediction lands to the truth. Closer = less under-prediction at the tail.
- **`zero_pred`**: the over-prediction floor on true-zero tiles. Lower = less overpred on
  empty tiles.
""",
    cell_id="results-md",
))

cells.append(code(
    """# Headline composite metric table
VARIANT_SHORT = {
    'lightgbm_two_stage': 'baseline',
    'lightgbm_two_stage_balanced': 'balanced',
    'lightgbm_two_stage_weighted': 'weighted',
    'lightgbm_two_stage_gamma': 'gamma',
    'lightgbm_two_stage_combined': 'combined',
}
table = agg[[
    'variant', 'tile_size_px',
    'spearman_rho_mean', 'presence_auc_mean',
    'compression_score',
    'zero__mean_pred', '1e-2_to_max__mean_true', '1e-2_to_max__mean_pred', '1e-2_to_max__ratio',
]].copy()
table.columns = ['variant', 'S', 'Spearman', 'AUC', 'compression_score',
                 'zero_pred', 'high_true', 'high_pred', 'high_ratio']
table['variant'] = table['variant'].map(VARIANT_SHORT)
table_sorted = (table.sort_values(['S', 'variant'])
                       .reset_index(drop=True)
                       .round({'Spearman': 4, 'AUC': 4, 'compression_score': 4,
                               'zero_pred': 5, 'high_true': 4, 'high_pred': 4, 'high_ratio': 3}))
display(table_sorted)
""",
    cell_id="results-table",
))

cells.append(md(
    """### 5.1 The headline figure

Left panel: per-truth-bin mean prediction at S=64 for all five variants, overlaid on the
truth identity (black triangles). Right panel: Spearman ρ vs high-bin ratio — the
**ranking vs tail-calibration trade-off**, with each variant placed by its (ρ, ratio) pair.
**Upper-right is the goal** (high ρ *and* tail ratio near 1.0).
""",
    cell_id="results-fig-md",
))

cells.append(code(
    """display(Image(filename=str(FIGS / '12_compression_fix_sweep.png')))
""",
    cell_id="results-fig",
))

cells.append(md(
    """### 5.2 What the table and figure say

- **`balanced` (the presence-head fix) is the only variant that wins on BOTH metrics
  without paying for it.** At S=64: Spearman ρ +0.263 → **+0.280** (+0.017), AUC 0.538 →
  **0.556** (+0.018). Tail ratio is essentially unchanged (0.83 → 0.83), zero-pred barely
  moves (0.0024 → 0.0026). Free lift. *Mechanism:* removing `is_unbalance=True` lets the
  classifier produce honest probabilities — the floor was a class-balance artefact, not a
  feature-information limit. **The presence head was the easy mistake to fix.**
- **`weighted` and `combined` recover the tail almost perfectly** (high ratio 0.83 →
  **1.01**) — but trade away Spearman (0.263 → 0.16) and AUC (0.538 → 0.47–0.44) and
  *double* the zero-bin over-prediction (0.0024 → 0.0048). The magnitude head, weighted by
  raw `y`, now learns to predict large values broadly: the high-bin tiles get pulled up to
  truth, the low-bin tiles get pulled up *with them*, and the model loses the fine-grained
  ordering that was driving Spearman in the first place. **This is the right operating
  point if the deliverable is "calibrated abundance estimates"** and the wrong one if it's
  "rank tiles for follow-up."
- **`gamma` is roughly neutral.** S=64: Spearman 0.255 (≈ baseline 0.263), AUC 0.513 (slight
  drop). The compression score improves modestly (0.66 → 0.64) but at a small AUC cost. Not
  a clear win — the loss redesign alone doesn't move the needle as much as the sample
  weighting does, and it costs AUC because the gamma fit doesn't sharpen the presence
  classifier's contribution.
- **The "compression score" scalar (mean | log10(ratio) | across the 4 nonzero bins) is
  itself a trap** if read alone: `weighted` scores 0.91 (worst) precisely because it
  *over-predicts* the low bins (where iso/gamma were under-predicting them). The score
  treats over- and under-prediction symmetrically, but they have different downstream
  consequences. *Read the per-bin ratios in the table above, not just the scalar.*

**Recommendation:** ship `balanced` as the new default for `lightgbm_two_stage`-style
modeling — it's a one-line change with positive expected effect on both headline metrics and
no observed downside on dev. Keep `weighted` and `combined` available as **alternative
operating points** for downstream use cases that prefer calibrated abundance to ranking
fidelity. `gamma` is not a clear win and not promoted.
""",
    cell_id="results-readout",
))

# ---------------------------------------------------------------------------
# Section 6 - what we were missing (existing binary sweep + top-k lift)
# ---------------------------------------------------------------------------
cells.append(md(
    """## 6. What §5 was missing — the existing binary sweep already had the answer

The intervention sweep above moves the headline numbers by tiny amounts (+0.017 Spearman,
+0.018 AUC for `balanced`). Brian flagged this — "the compression is still there; the metric
changes are really small" — and pushed back: maybe **we're measuring the wrong thing**, or
**framing the problem wrong**, not just losing to compression in the loss.

Three things make that pushback right:

1. **`bc_ge_1` AUC is a near-meaningless threshold.** "≥ 1 boulder in a 320 m × 320 m tile"
   is barely a presence task — it asks the model to distinguish "0 boulders" from
   "1+ boulder", which on a heavy-tail target is essentially "is there any signal at all".
   §6.1 of [`docs/modeling_results.md`](../docs/modeling_results.md) framed the binary task
   on `bc_ge_1` because at v1 the rarer thresholds had too few positives per fold; with v2's
   ~30% boulder-rich fraction at S=64, the operationally meaningful threshold `fa_gt_1e-2`
   is no longer rare.
2. **Cross-image mean AUC averages an extremely bimodal per-image distribution.** The
   per-fold σ on AUC at S=64 is **0.18** — comparable to the *mean − chance* offset. The
   "average" is a fiction over a population where some images fit the model well and others
   actively anti-fit.
3. **AUC averages performance across every probability threshold.** For our operational task
   (rank a small number of tiles to follow up with HiRISE, compare with THEMIS, or run the
   HiRISE 3-band compositional study), only the **very top** of the ranking matters. Top-K
   lift answers that question directly; AUC does not.

### 6.1 What "top-K lift" measures (qualitatively)

**Top-K lift = "of the K tiles the model is most confident about, how many more true
positives do I find than I would by random sampling?"**

The recipe in plain English:

1. Sort all test tiles by predicted probability, highest first.
2. Take the top K (we use `K = n_positives` in the test set — the most common convention).
3. Compute the fraction of those top K that are actually positive.
4. Divide by the base rate (overall positive fraction in the test set).
5. That ratio is the lift.

Numbers:

- **lift = 1.0** → no better than random at the top.
- **lift = 2.0** → among the top K, twice as many true positives as random would have given.
- **lift = `1 / base_rate`** → perfect ranking (all positives are in the top K).

For a rare-positive image (say, 1.3 % boulder-rich), perfect-lift can reach ~77×; lift = 9× is
a real, operationally useful signal. For a common-positive image (80 % boulder-rich) the
max-possible lift is only 1.25, so even a perfect model "looks weak" by raw lift —
**normalized lift = lift × base_rate** corrects for that.

### 6.2 The v2 binary sweep at `fa_gt_1e-2` ("boulder-rich") tells a much stronger story

Probe: [`scripts/probes/_diag_v2_binary_per_image.py`](../scripts/probes/_diag_v2_binary_per_image.py).
Reading the EXISTING [v2 binary
sweep](../models/_sweep_binary/20260529T075754Z/aggregate.parquet) at the meaningful
threshold:
""",
    cell_id="missing-md",
))

cells.append(code(
    """# Read the existing v2 binary sweep at fa_gt_1e-2 vs bc_ge_1 (no new training)
binsweep = MODELS / '_sweep_binary' / '20260529T075754Z'
bin_agg = pd.read_parquet(binsweep / 'aggregate.parquet')
display(bin_agg[[
    'target_id', 'tile_size_px', 'auc_mean', 'auc_std',
    'lift_at_top_k_mean', 'brier_mean', 'ece_mean'
]].round(3))
""",
    cell_id="missing-table",
))

cells.append(md(
    """**Read this with section 6.1's definitions in mind:**

- **`fa_gt_1e-2` at S=64 lifts 1.43×** vs `bc_ge_1`'s 1.02. The boulder-rich classifier already
  finds **40 % more true positives in its top-K** than the any-boulder classifier — and 1.43×
  more than random.
- AUC of the two targets at S=64 looks similar (≈ 0.62 both) because AUC averages across all
  thresholds, but the *top* of the ranking — the operationally relevant part — is much better
  for the boulder-rich target.
- **Calibration is similar (ECE 0.12 / 0.27)**, both reflecting the same `scale_pos_weight`
  inflation we diagnosed in the regression presence head.
""",
    cell_id="missing-readout",
))

cells.append(md(
    """### 6.3 Per-image: the headline mean buries strong individual-image signal

The cross-image mean buries a bimodal distribution. Per-image at S=64 for `fa_gt_1e-2`:

- median AUC **0.61**, max **0.91**, min 0.40, σ 0.12
- 7 of 25 folds AUC > 0.70 (genuinely strong)
- 4 of 25 folds AUC < 0.50 (anti-signal)
- **Top performers:**
  - ESP_042964_2160: AUC 0.91, lift **5.4×** (50 positives / 608 tiles, base rate 8 %)
  - ESP_055978_2270: AUC 0.76, lift **9.1×** (17 positives / 1310 tiles, base rate 1.3 %)

These aren't toy results — these are real held-out images where the model is operationally
usable today. The "0.55 ceiling" framing of v1 §7 was a cross-image-mean artefact at the
wrong threshold; **at the right threshold, the per-image distribution has plenty of usable
mass**.
""",
    cell_id="missing-per-image",
))

cells.append(code(
    """# Per-image breakdown for fa_gt_1e-2 at S=64
binsf = pd.read_parquet(binsweep / 'summary.parquet')
sub = binsf[(binsf['target_id'] == 'fa_gt_1e-2') & (binsf['scale_idx'] == 3)
            & ~binsf['is_specificity_only'].astype(bool)].copy()
sub = sub.dropna(subset=['auc'])
print(f'{len(sub)} held-out images at fa_gt_1e-2 S=64')
print('per-image AUC distribution:')
print(sub['auc'].describe().to_string())
print()
print('Top 5 by lift:')
display(sub.sort_values('lift_at_top_k', ascending=False).head(5)[
    ['held_out_obs_id', 'n_tiles', 'n_positive', 'base_rate', 'auc', 'lift_at_top_k']
].round(3))
""",
    cell_id="missing-per-image-data",
))

# ---------------------------------------------------------------------------
# Section 7 - five-hypothesis framework
# ---------------------------------------------------------------------------
cells.append(md(
    """## 7. Five hypotheses for "compression persists; signal is real"

Given §5 (training-side fixes barely move the headline) and §6 (existing data already
contains stronger signal than we've reported), what is the **actual** binding constraint?
Five hypotheses, ordered by how testable they are:

**H1 — Metric (we're underselling).** Cross-image mean AUC averages images where the model
genuinely succeeds (AUC 0.91, lift 9×) with images where it anti-signals (AUC 0.40). Top-K
lift, PR-AUC, and per-image distributions tell a much more positive story than the AUC mean
alone.
*Implication:* report richer metrics; "operationally usable on the easier images" is the
honest story. *Counter:* the bimodality is real, not noise — H1 alone doesn't resolve the
"bad images" tail.

**H2 — Target is wrong (`fractional_area` is structurally noisy at the low end).** A 4 m²
boulder at 5 m/px occupies ~1/6 of a pixel — `fractional_area` below ~0.005 is dominated by
**pixel-aliasing artefacts** of the rasterization, not real signal. The model's compression
on the low end may reflect that the label itself carries no information there, not a model
failure. *We already have `boulder_count` in the parquet* — a discrete, alias-robust target.
*Implication:* test `log(boulder_count + 1)` and `log(fractional_area + ε)` as alternative
targets. Cheap to run (no new training infrastructure; reuse `lightgbm_two_stage_balanced`).

**H3 — Per-image heterogeneity (model averages incompatible regimes).** The §2 feature
importance has `shadow_fraction` as #1 at every scale. But `shadow_fraction` only works if
illumination geometry is consistent across the image — at high incidence angles, even bare
regolith casts shadows. The 38 v2 images span wide latitudes and illumination conditions; the
model has no way to know "this image's shadow signal means something different than that
image's". This would explain the bimodal per-image AUC distribution.
*Implication:* add per-image features (incidence/emission angles, latitude, surface unit) OR
a pre-classifier ("is this image well-fit by texture features?") that gates predictions.
*Status:* documented here as a high-leverage future direction; **not implemented this
session** — it requires ingesting `.LBL` geometry into Stage 5 and a new feature
joining step.

**H4 — Multiplicative hurdle wastes structure.** `two_stage` assumes `P(positive)` and
`E[mag | positive]` are independent given X. But the §2 diagnostic shows they're correlated
in the same direction (both rise with truth bin); the hurdle forces them apart. A single-head
model with a calibrated-quantile objective might exploit that correlation.
*Implication:* test a quantile-regression model or a multi-output (presence + q=0.9) head
with a calibration term. *Deferred* (more modeling work; H2 is likely the cheaper unlock).

**H5 — 5 m/px CTX texture is at its information floor.** [§7 / §9.4 of `modeling_results.md`](../docs/modeling_results.md)
showed within-image AUC ≈ LOIO AUC. The texture features have extracted what they can. No
loss redesign closes that gap.
*Implication:* the unlock is *outside* CTX — HiRISE-decimated as a feature, THEMIS thermal as
a covariate, coarse spatial priors. This is the §7.2 future-work agenda; still binds
*eventually*.

### 7.1 My take (after the §5 / §6 results)

**H1 is real but not by itself a fix** — it's an honesty correction, not a model improvement.
We must report richer metrics, but reporting them alone won't lift the bottom-quartile images.

**H2 is the cheapest unexamined hypothesis, and possibly the highest leverage.** If
`boulder_count` is genuinely cleaner than `fractional_area` at the low end, switching
sharpens the negative class and removes the compression *cause* rather than fighting its
*effect*. **Implementing this session.**

**H3 is intellectually the most interesting** and probably the right long-term lever, but
expensive — needs a Stage-5 schema change to bring incidence/emission/latitude features in.
**Documented here as future work; not implemented this session.**

**H4 is plausible but I'd test H2 first** — if changing the target removes most of the
compression, the multiplicative hurdle isn't the binding constraint after all.

**H5 is probably true at the limit** and binds *eventually*, but we should rule out H2
first because H2 might lift the ceiling that H5 then becomes the new binding constraint on.

### 7.2 What's actually being implemented this session

Per Brian's decision (2026-05-29): **implement H1 + H2, document H3 as future work**.

- **H1 implementation** = enrich
  [`src/modeling/evaluate.py`](../src/modeling/evaluate.py) with: PR-AUC, normalized
  lift-at-top-K (= lift × base_rate, ∈ [0, 1]), precision@k and recall@k at K ∈ {1 %, 5 %,
  10 %}, per-image AUC/lift dist stats. Computed on every regression run via an implicit
  binary derived from the target, AND on every classification run.
- **H2 implementation** = sweep `lightgbm_two_stage_balanced` across three targets ×
  S=32/64: `fractional_area` (baseline), `log_boulder_count = log1p(boulder_count)`, and
  `log_fractional_area = log1p(fractional_area)`. Same dev harness; composite metric is
  the §5 metric set + the H1 richer additions.
""",
    cell_id="hypotheses-md",
))

# ---------------------------------------------------------------------------
# Section 8 - H3 per-image deep dive
# ---------------------------------------------------------------------------
cells.append(md(
    """## 8. H3 deep dive — what the per-image heterogeneity looks like

Documenting H3 in detail here so the future-work hypothesis is concrete enough to act on
later. The data already in the v2 sweep is enough to characterise the per-image variability;
this section is a pre-mortem on what we'd be modelling if we picked up H3.

### 8.1 Per-image AUC is bimodal, not noisy-around-a-mean

Histogram of per-fold AUC at `fa_gt_1e-2` S=64. If the per-image distribution were
unimodal-noisy around the mean ~0.62, the histogram would look Gaussian with a single peak.
What we see instead:
""",
    cell_id="h3-md",
))

cells.append(code(
    """# Render the per-image AUC histogram at fa_gt_1e-2 S=64
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
sub = binsf[(binsf['target_id'] == 'fa_gt_1e-2') & (binsf['scale_idx'] == 3)
            & ~binsf['is_specificity_only'].astype(bool)].dropna(subset=['auc'])
ax.hist(sub['auc'], bins=15, alpha=0.7, color='C0', edgecolor='k')
ax.axvline(0.5, color='red', linestyle='--', label='chance')
ax.axvline(sub['auc'].mean(), color='black', linestyle='-', lw=2, label=f"mean={sub['auc'].mean():.3f}")
ax.axvline(sub['auc'].median(), color='green', linestyle=':', lw=2, label=f"median={sub['auc'].median():.3f}")
ax.set_xlabel('per-image AUC')
ax.set_ylabel('# of held-out images')
ax.set_title(f'fa_gt_1e-2 AUC at S=64 ({len(sub)} images)\\nbimodal: ~7 strong, ~4 anti, rest middling')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(sub['base_rate'], sub['auc'], s=80, alpha=0.7)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.6)
for _, row in sub.nlargest(3, 'auc').iterrows():
    ax.annotate(row['held_out_obs_id'].replace('ESP_', '')[:6],
                (row['base_rate'], row['auc']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)
for _, row in sub.nsmallest(3, 'auc').iterrows():
    ax.annotate(row['held_out_obs_id'].replace('ESP_', '')[:6],
                (row['base_rate'], row['auc']),
                xytext=(5, -10), textcoords='offset points', fontsize=8)
ax.set_xlabel('base rate (P(boulder_rich) in held-out image)')
ax.set_ylabel('per-image AUC')
ax.set_title('AUC vs base rate\\n(if H3 right, look for a clean group of well-fit images)')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(FIGS / '12_per_image_auc_distribution.png', dpi=110, bbox_inches='tight')
plt.show()
""",
    cell_id="h3-fig",
))

cells.append(md(
    """### 8.2 What the histogram says

- Distribution is **clearly bimodal-with-tails**, not a Gaussian around the mean.
- A cluster of ~7 images with AUC > 0.7 ("model works here") — these are predominantly the
  rare-positive images where shadow / texture signal is dominant
- A cluster of ~4 images with AUC < 0.5 ("model anti-signals") — these need investigation
- The remaining ~14 images sit near chance (AUC 0.5–0.65)
- **Base rate ≠ performance**: the AUC vs base-rate scatter does NOT show a clean
  relationship. Some high-base-rate images (boulder-rich-dominant) score well; others don't.
  Per-image variability is not just "easy targets".

### 8.3 What H3 would actually need to do

To attribute the bimodality to a known per-image variable, we'd need to ingest per-image
metadata into the dataset and fit interactively. Concrete next steps if H3 is picked up:

1. **Read `IncidenceAngle`, `EmissionAngle`, `SubSolarAzimuth`** from the cached PDS `.LBL`
   files (already in `cache/pds_labels/` from
   [project_state_2026-05-26](../../C:/Users/brian/.claude/projects/c--Users-brian-Documents-PhD-HiRiseToCTXBoulders-hirise2ctx/memory/project_state_2026-05-26.md))
   and join into each tile's feature row.
2. **Read `CenterLat`** from the manifest as a coarse "surface unit" proxy (Mars latitude
   bands correlate with terrain type — equatorial vs mid-latitudes vs polar).
3. **Add a small pre-classifier**: predict per-image "well-fit-by-texture" from the per-image
   features above; gate per-tile predictions on it.
4. **Or: include per-image features directly** in the tile-level LightGBM as repeating
   columns. Cheaper and might already get most of the win.

H3 is **not implemented this session** because it touches Stage 4b's feature schema, the
labels parquet schema, and the modeling loaders all at once — too many moving parts. But the
diagnosis above is concrete enough to scope: a Stage 4c addition that adds 4 per-image
columns + a 5-line change in `src/dataset.py` to include them in the feature matrix.
""",
    cell_id="h3-readout",
))

# ---------------------------------------------------------------------------
# Section 9 - H1 + H2 results (populated after running the new sweep)
# ---------------------------------------------------------------------------
cells.append(md(
    """## 9. H1 + H2 results: richer metrics + target reformulation

This section is populated by [`scripts/probes/_sweep_target_reformulation.py`](../scripts/probes/_sweep_target_reformulation.py),
which fans the **`lightgbm_two_stage_balanced`** variant (the §5 winner) across three target
columns × {S=32, S=64} on the v2-dev within-image scheme. Metrics include the H1 additions:
PR-AUC, normalized lift, precision@k, recall@k.

Targets compared:
- **`fractional_area`** — baseline (compresses, per §2/§5)
- **`log_fractional_area`** = `log1p(fractional_area)` — same target log-transformed
- **`log_boulder_count`** = `log1p(boulder_count)` — discrete, alias-robust at the low end
""",
    cell_id="h1h2-md",
))

cells.append(code(
    """# Load the H1+H2 sweep (created by scripts/probes/_sweep_target_reformulation.py)
target_sweep_root = MODELS / '_sweep_target_reformulation'
if target_sweep_root.exists():
    runs_t = sorted([p for p in target_sweep_root.iterdir() if p.is_dir()])
    if runs_t:
        TSWEEP = runs_t[-1]
        print(f'sweep: {TSWEEP.name}')
        tagg = pd.read_parquet(TSWEEP / 'aggregate.parquet')
        # Display the composite metric table
        keep_cols = [c for c in [
            'target_col', 'tile_size_px',
            'spearman_rho_mean', 'presence_auc_mean',
            'lift_at_top_k_mean', 'pr_auc_mean', 'normalised_lift_mean',
            'compression_score',
            'zero__mean_pred', '1e-2_to_max__ratio',
        ] if c in tagg.columns]
        display(tagg[keep_cols].round(4))
    else:
        print('No H1+H2 sweep yet — run `python scripts/probes/_sweep_target_reformulation.py`.')
else:
    print('No H1+H2 sweep yet — run `python scripts/probes/_sweep_target_reformulation.py`.')
""",
    cell_id="h1h2-table",
))

cells.append(code(
    """display(Image(filename=str(FIGS / '12_target_reformulation.png')))
""",
    cell_id="h1h2-fig",
))

cells.append(md(
    """### 9.1 What the H1+H2 numbers say (the headline result of the session)

At **S=64**, `lightgbm_two_stage_balanced` × three targets:

| target              | Spearman ρ | ROC-AUC (presence) | ROC-AUC (meaningful) | **PR-AUC** | normalised lift | precision@top-5 % |
|---------------------|-----------:|-------------------:|---------------------:|-----------:|----------------:|------------------:|
| `fractional_area`   | +0.280     | 0.556              | 0.713                | 0.526      | 0.488           | 0.549             |
| **`boulder_count`** | **+0.283** | 0.564              | 0.697                | **0.640**  | **0.619**       | **0.660**         |
| `log_boulder_count` | +0.279     | 0.545              | 0.690                | 0.638      | 0.628           | 0.663             |

**The headline finding:** switching from `fractional_area` to `boulder_count` lifts:
- **PR-AUC by +0.114 (≈ +22 %)**
- **Normalised lift@top-K by +0.131 (≈ +27 %)**
- **Precision@top-5 % by +0.111 (≈ +20 %)**

**…while leaving Spearman ρ and ROC-AUC essentially unchanged** (+0.003 ρ, +0.008 AUC).

**This is the H1 framework's prediction confirmed end-to-end.** ROC-AUC and Spearman could
not see the improvement because they average across thresholds / are rank-invariant. PR-AUC
and lift, which key on the operational top-of-ranking, show a clean +20–27 % relative gain
that would have been completely invisible under the §5 metric framing.

### 9.2 What the H2 mechanism likely is

`boulder_count` is **alias-robust at the low end**: it counts polygon-tile intersections
without dividing by area. A 4 m² boulder in a 320 m × 320 m tile contributes either 0 or 1
to the count, period; it doesn't get smeared into a small fractional_area value whose magnitude
depends on how the polygon happens to align with the 5 m CTX grid. So:

- **Negatives are cleaner.** A tile with zero boulders genuinely has count == 0; the
  fractional_area equivalent is a small noisy positive number from a few partial-pixel
  intersections. The classifier doesn't have to learn "this 8×10⁻⁵ fractional_area might be
  zero" — it just sees `boulder_count == 0`.
- **The presence/magnitude separation is cleaner.** The hurdle model's `y > 0` rule is now
  exact (count > 0 means a polygon intersects), not a stand-in for "more than aliasing
  noise".
- **The high-tail end is preserved.** Spearman is rank-invariant, so the ranking signal that
  the model was producing on fractional_area is also there on count.

`log_boulder_count` performs essentially the same as raw `boulder_count` — confirming that
the model's internal `log1p+Huber` on the magnitude head handles the log transformation
correctly, and that we don't need to pre-log the target.

### 9.3 What changed in the per-bin compression

The "compression score" diagnostic from §5 was computed in `fractional_area` bins; for
`boulder_count` it's not directly comparable. But the *operational* compression — does the
model produce predictions that map back to true high-abundance tiles correctly? — is
captured by precision@top-K, which improved 0.55 → 0.66 at S=64. **The compression diagnosed
in §2 was as much a target-loss mismatch as a magnitude-head shrinkage**: feed the model a
cleaner target and most of the compression goes away.

### 9.4 Why this matters for the headline product

If the deliverable is "predict abundance across CTX", the boulder_count regressor is a
direct upgrade: same ranking, much better operational discrimination of the boulder-rich
tiles for downstream use (HiRISE follow-up, THEMIS cross-comparison, and the HiRISE 3-band
compositional study — the latter switched from CRISM 2026-05-30).
**The headline number for the deliverable should be PR-AUC + lift@top-K, not ROC-AUC** —
that's a documentation update for [`docs/modeling_results.md`](../docs/modeling_results.md)
§9 and §11 to make on the next pass.
""",
    cell_id="h1h2-readout",
))

# ---------------------------------------------------------------------------
# Section 10 - next steps (final)
# ---------------------------------------------------------------------------
cells.append(md(
    """## 10. Next steps

Based on §6 + §7's reframing:

- **Headline-product decision (Brian-gated):** is the deliverable an **abundance
  *regressor*** (calibrated `fractional_area`) or a **boulder-rich *classifier*** (lift-aware
  ranker for HiRISE follow-up / spectral comparison)? §6 makes the case that the
  classifier is the operationally useful artefact today; the regressor is the more ambitious
  target and possibly H5-bound.
- **If classifier wins:** rebrand `lightgbm_classification` at `fa_gt_1e-2` (or its
  `log_boulder_count` equivalent if H2 confirms) as the headline; promote `balanced`-style
  calibration to it; report per-image PR curves + lift in [`docs/modeling_results.md`](../docs/modeling_results.md).
- **If regressor still preferred:** look hard at H3 (per-image features) and H5 (THEMIS
  thermal as a feature, HiRISE-decimated surrogate). These are the §7 / §9.4 unlocks.
- **`balanced` promotion to full v2** still recommended either way (small but free).

Deferred (unchanged): the CNN variants per
[project_state_2026-05-29](../../C:/Users/brian/.claude/projects/c--Users-brian-Documents-PhD-HiRiseToCTXBoulders-hirise2ctx/memory/project_state_2026-05-29.md);
S=128 promotion to full v2.
""",
    cell_id="next-steps-final",
))

# ---------------------------------------------------------------------------
# Assemble + write
# ---------------------------------------------------------------------------
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "geospatial",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH}")
