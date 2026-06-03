# Handoff prompt — next session

**Last updated 2026-06-03 — PROJECT WRAPPED at a v1-reportable state.
Stage 7d closed with shadow masking + per-image attribution. Paper-Methods
style writeups landed at `docs/compositional.md` + `docs/modeling.md`.
Unfinished items (Stage 7e refinement, provenance disambiguation, ESP_046803_2325
backfill, Path A model bank) are recorded as future work, not blockers.**

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`
(see memory [[conda_run_no_capture_output]]).

## Read in this order before starting

1. **Memory** [`project_state_2026-06-03.md`](../memory/...) (CURRENT) —
   wrap-up summary; what landed + what was deferred.
2. [`docs/compositional.md`](docs/compositional.md) — the headline science
   deliverable. Bottom line at top, full Methods + Results + Discussion +
   Limitations + Future-work + References.
3. [`docs/modeling.md`](docs/modeling.md) — Methods companion to
   [`docs/modeling_results.md`](docs/modeling_results.md).
4. [`DECISIONS.md`](DECISIONS.md) 2026-06-03 entry — shadow sweep numbers,
   attribution counts, the two direction-reversal images, full code-change
   inventory.
5. [`HANDOFF_NEXT_SESSION.md`](HANDOFF_NEXT_SESSION.md) (this file) —
   priorities for the next session.

## Where we are

**Project wrap state**: Stage 7 thread has a publishable + properly-bounded
conclusion. Boulder-rich vs boulder-poor colour difference is real
(|d| 0.21 – 0.37, all 6 features, all p ≤ 1e-26), ~50–80 % dust-attributable
and ~20–50 % composition-attributable. Composition residual survives both
per-image dust control AND tile-level shadow filtering (T=0.10 partial-dust
|d| 0.13 – 0.18 across all 5 non-dust features, p ≤ 1e-8). Strongest
composition signal in `IR/BG` and `IR/RED` (the ferric/ferrous-indexing
ratios). Signal direction is "boulders less ferric-altered than surrounding
regolith" — consistent with either locally-sourced-with-maturity OR
transported (e.g. megatsunami) provenance; Stage 7d alone cannot
disambiguate. See [`docs/compositional.md`](docs/compositional.md) for the
full write-up.

**Modeling state**: Stage 6c soft PASS (+0.056 pooled-global PR-AUC via
Strategy B down-weighting on the v1 ridge gate); Stage 6e mechanism
(CTX-source heterogeneity) empirically validated. Per-tile presence-AUC
ceiling ~0.55 – 0.62 is a real signal floor at 5 m/px CTX texture (within ≈
LOIO across all variants). v2 dense labels lifted regression Spearman
+0.10 over v1 (signal floor was *partly* a missed-boulder artefact, not
fully). Per-image bimodality remains: 7 images AUC > 0.70, 4 anti-signal
(AUC < 0.50), rest near chance. See
[`docs/modeling_results.md`](docs/modeling_results.md) + new
[`docs/modeling.md`](docs/modeling.md).

**Tests + figures**: 281 pytest pass. Stage 7 figures all in
`reports/figures/stage7*` (8 figures). Notebooks 14, 15, 16 all executed.

## Goal of this session (Brian to decide at start)

The project is in a publishable shape. The most likely next moves, in
order of "tight science follow-up to wrap better":

- **D1. Provenance disambiguation (the original instructor goal).** Stage 7d
  cannot distinguish locally-sourced-with-maturity from transported. Three
  tiers in [`PLAN_Compositional.md §11`](PLAN_Compositional.md):
  - **Quick (½ day)**: manually classify the 36 ObsIds by terrain context
    using HiRISE browse images, into {crater-ejecta-dominated / plains /
    mass-wasting / mixed / candidate-tsunami-deposit}. Re-stratify Stage 7d
    per-image effect sizes. Tests whether the composition_residual images
    cluster on candidate-tsunami terrain or distribute uniformly.
  - **Cleaner (1 – 2 days)**: cross-reference against [Robbins & Hynek 2012](https://doi.org/10.1029/2011JE003966)
    crater catalog; flag tiles within N crater radii; check whether the
    composition residual concentrates in crater-distal tiles (the
    transported prediction).
  - **Most rigorous (multi-day)**: compare composition signature of
    candidate-tsunami boulder fields against inferred southern-highlands
    source-unit colour (CRISM or HiRISE colour of plausible upstream
    units). The headline science answer the project was originally set
    up to give.
- **D2. Stage 7e formal dust analysis.** Two refinements with cheap upside:
  - [Atwood-Stone & McEwen 2013](https://doi.org/10.1029/2013GL058355)
    dust index (absolute reflectance + band-shape, not RED/BG ratio).
  - Pixel-level shadow masking on the COLOR.JP2 itself rather than the
    coarse CTX-tile filter currently used. Stage 7d's shadow sweep already
    showed partial-dust effects grow under tile filtering; pixel masking
    should grow them further.
  ~1 day. Sharpens the composition residual; doesn't change the
  qualitative conclusion.
- **D3. ESP_046803_2325 Stage 4 backfill.** ~half hour. Adds one image
  (currently has COLOR.JP2 + LBL on disk but no labels parquet). Lifts
  cohort 36→37 colour-eligible / 30→31 P4-eligible.
- **D4. Path A model bank.** P1+P2 full-v2 LOIO sweep promotion (currently
  dev-validated only). Strengthens the modelling story for the report.
  ~2 – 3 hr.
- **D5. Submission prep.** If this is "good enough to submit," next step
  is paper formatting / abstract / committee delivery rather than more
  analysis. Brian's call.

## Critical gotchas (carry forward)

- **`conda run` swallows subprocess stdout** unless invoked with
  `--no-capture-output`. Combine with `python -u` + `print=functools.partial(print, flush=True)`
  in every long-running probe. Memory [[conda_run_no_capture_output]].
- **`conda run python -c "..."` doesn't accept multi-line strings** on
  Windows — write a tempfile and run it. Hit twice; fix in
  `scripts/probes/_inspect_nb15_outputs.py` + `_dump_attribution.py`.
- **HiRISE PDS SP1 bug also poisons `COLOR.JP2`** — use
  `src.colour.corrected_source_crs` as CRS override at every COLOR.JP2
  read.
- **COLOR.JP2 stores raw uint16 DN** — convert via per-image
  `I/F = DN * scaling_factor + offset` from the LBL. Fixed in
  `scripts/run_stage7c_features.py`.
- **Lambertian correction cancels in within-image diffs and band ratios**
  but is REQUIRED for cross-image pooling (Stage 7d standardised tests).
- **Inference-time scope (carry-over)**: model features must be derivable
  from CTX alone. Colour features are *analysis-only* per
  `PLAN_Compositional.md §10` — they cannot be fed to the rock-abundance
  model at inference time over CTX-only regions.
- **Tile-level shadow filtering uses CTX-derived `shadow_fraction`**, not
  pixel-level HiRISE-side masking. The Stage 7d shadow sweep is therefore
  a *coarse* shadow control; Stage 7e refinement should pixel-mask the
  COLOR.JP2 itself.
- **Per-image attribution is conservative.** 16 of 26 eligible images
  land at `no_signal` largely because of small per-image n_rich or
  n_poor — the test is underpowered, not the signal absent. The pooled
  cohort evidence is the stronger evidence.
- **AskUserQuestion before**: expensive sweeps, git commits, destructive
  ops on cached artefacts.
- **281 pytest pass baseline (Stage 7d wrap-up era)** — Stage 7d tests
  are `tests/test_stage7d.py` (28 total). Run `pytest tests/ -q` before
  any promotion.

## Stage 7 artefacts (reference)

- **Module**: [`src/stage7d_pooled.py`](src/stage7d_pooled.py),
  [`src/colour.py`](src/colour.py).
- **Runners**: [`scripts/run_stage7a_audit.py`](scripts/run_stage7a_audit.py),
  [`scripts/run_stage7a_fetch.py`](scripts/run_stage7a_fetch.py),
  [`scripts/run_stage7c_features.py`](scripts/run_stage7c_features.py),
  [`scripts/run_stage7d_pooled.py`](scripts/run_stage7d_pooled.py).
- **Probes**: [`scripts/probes/_stage7_feasibility.py`](scripts/probes/_stage7_feasibility.py),
  `_verify_stage7c_trio.py`, `_summarise_stage7c.py`,
  `_inspect_nb15_outputs.py`, `_dump_attribution.py`.
- **Tests**: [`tests/test_colour.py`](tests/test_colour.py) (8),
  [`tests/test_stage7d.py`](tests/test_stage7d.py) (28).
- **Notebooks** (all executed):
  [`notebooks/14_compositional_feasibility.ipynb`](notebooks/14_compositional_feasibility.ipynb),
  [`notebooks/15_stage7d_pooled.ipynb`](notebooks/15_stage7d_pooled.ipynb),
  [`notebooks/16_stage7d_shadow_attribution.ipynb`](notebooks/16_stage7d_shadow_attribution.ipynb).
- **Figures** (committed): `reports/figures/stage7*.png` — 8 figures.
- **Docs**: [`docs/compositional.md`](docs/compositional.md),
  [`docs/modeling.md`](docs/modeling.md).
- **Outputs** (gitignored): `dataset_v2/features_colour.parquet`
  (9 860 rows), `dataset_v2/stage7d_pooled*.parquet` (4 variants),
  `dataset_v2/stage7d_per_image_attribution.parquet` + the 3 shadow-
  threshold variants.

## Reporting protocol (carry forward)

1. **`PROMOTION_QUEUE.md`**: nothing new to move from this session.
2. **`DECISIONS.md`**: one entry per Stage 7 stage with numbers; 2026-06-03
   added the wrap-up entry.
3. **Memory**: `project_state_2026-06-03.md` is CURRENT.
4. **`HANDOFF_NEXT_SESSION.md`**: this file — rewrite based on what
   actually lands next.
5. **AskUserQuestion before `git commit`** of any of the above.
