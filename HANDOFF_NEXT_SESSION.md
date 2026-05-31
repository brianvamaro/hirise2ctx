# Handoff prompt — next session

**Last updated 2026-05-31 (late) after Stage 6c full evaluation (v1 + v2 push).**

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run -n geospatial python …` (never the
env's `python.exe` directly — see memory note [[conda_location]]).

## Read in this order before starting

1. **Memory** `project_state_2026-05-31-late.md` (CURRENT) — Stage 6c outcome
   + the two remaining docket paths.
2. **[`docs/modeling_results.md`](docs/modeling_results.md) §14** — Stage 6c
   writeup (strict-FAIL evidence, soft-PASS deliverable, structural ceiling
   argument).
3. **[`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md)** — Problem 3 status now closed
   as "mechanism identified, residual gap needs more LOIO images or HiRISE
   priors"; Stage 6c entry marked ◐ DEV-PARTIAL.
4. **[`scripts/probes/_stage6c_gate.md`](scripts/probes/_stage6c_gate.md)** +
   **[`_stage6c_gate_v2.md`](scripts/probes/_stage6c_gate_v2.md)** — full
   per-gate / per-strategy tables.

## Where we are

**Stage 6c is settled as ◐ DEV-PARTIAL.** Two passes ran end-to-end:

- **v1** (3 features × 3 models): rule_n_sources_gt_median is the best gate
  (ROC-AUC 0.606); ridge_then_logistic gives the best pooled-global Strategy B
  delivering **+0.056 PR-AUC** (no tiles dropped).
- **v2** (6 features × 5 models × 4 bad-image cutoffs = 20 combos): LightGBM
  and L1 logreg underperform L2 logreg, consistent with n=38 limiting
  non-linear/sparse models. **0 / 20 combos clear the strict bar**.

Strict acceptance (retained PR-AUC ≥ 0.65 AND tile_kept_frac ≥ 70 % AND lift
≥ +0.10) **cannot be satisfied** — the bad-image set carries disproportionately
many tiles, and per-fold PR-AUC is rank-invariant within a held-out image (a
mid-implementation realisation that wasn't in the spec) so Strategies B and C
only move the *pooled-global* metric, not the per-fold mean targeted by the
strict criterion.

**Problem 3 (per-image anti-signal) status**: mechanism quantified, residual
gap not closeable from CTX provenance alone at this dataset scale. Stage 6e
(distance-to-seam, the one remaining unimplemented per-tile feature) is low
priority — the Stage 6b bimodality pattern likely persists for any per-tile
SeamMap use.

## Goal of this session (Brian to decide at start)

Two paths remain on offer (Stage 6c is closed; Stage 6a/6b moved to Banked-but-
DEV-PARTIAL):

- **A. Bank wins — P1+P2 full-v2 promotion + P3+P4 doc reframe (Recommended).**
  ~2-3 hr total. Bank the validated +22 % PR-AUC dev gain at full v2; promote
  PR-AUC + lift@top-K as headline metrics. Brian-gated full-v2 sweeps.
- **B. Pivot to Stage 7 — compositional HiRISE 3-band feasibility test.**
  1-2 days. [`PLAN_Compositional.md`](PLAN_Compositional.md) §7.0. De-risks a
  separate research thread that's not bound by the 5 m/px CTX texture floor.

## Path A detail — P1+P2 full-v2 + P3+P4 reframe

Unchanged from prior handoff. P1 (presence-head fix) + P2 (boulder_count
target) are validated +22 % PR-AUC on dev; need full-v2 LOIO sweep to bank.

```powershell
$conda = "C:\Users\brian\anaconda3\Scripts\conda.exe"

# P1 (presence-head fix) — full-v2 LOIO
& $conda run -n geospatial python scripts/sweep.py `
    --variants lightgbm_two_stage_balanced `
    --dataset-dir dataset_v2 --scheme loio_nfold

# P2 (boulder_count target) — via the probe until --target-col lands on sweep.py
& $conda run -n geospatial python scripts/probes/_sweep_target_reformulation.py `
    --targets boulder_count --scales 3 `
    --dataset-dir dataset_v2
```

P3 / P4 are doc-only — update
[`docs/modeling_results.md`](docs/modeling_results.md) §9 headline metrics +
change default binary in
[`src/modeling/binary_target.py`](src/modeling/binary_target.py) from `bc_ge_1`
to `fa_gt_1e-2`. **AskUserQuestion before each expensive sweep.**

## Path B detail — Stage 7.0 feasibility

[`PLAN_Compositional.md`](PLAN_Compositional.md) §7.0 unchanged. 1-2 day
feasibility test on 2-3 images using actual BoulderNet labels (not model
predictions) to test whether HiRISE 3-band spectra of boulder-rich areas differ
from surroundings ([Delamere 2010](https://doi.org/10.1016/j.icarus.2009.03.012)).

## Stage 6c artefacts (reference, no re-run needed)

- Probe v1: [`scripts/probes/_stage6c_gate.py`](scripts/probes/_stage6c_gate.py)
  + [`_stage6c_gate.md`](scripts/probes/_stage6c_gate.md) (tracked)
- Probe v2: [`scripts/probes/_stage6c_gate_v2.py`](scripts/probes/_stage6c_gate_v2.py)
  + [`_stage6c_gate_v2.md`](scripts/probes/_stage6c_gate_v2.md) (tracked)
- Predictor table: `cache/stage6c/predictor_table.parquet` (gitignored)
- LOIO CV out-of-fold gate predictions: `cache/stage6c/gate_cv.parquet`
- v2 summary: `cache/stage6c/v2_gate_summary.parquet`

Both probes complete in ~30 s on the cached inputs. Re-run with:

```powershell
$conda = "C:\Users\brian\anaconda3\Scripts\conda.exe"
& $conda run -n geospatial python scripts/probes/_stage6c_gate.py
& $conda run -n geospatial python scripts/probes/_stage6c_gate_v2.py
```

## Critical gotchas (carry forward)

- **Inference-time scope**: model features must be derivable from CTX alone.
  HiRISE LBL angles are diagnostic-only. See PROMOTION_QUEUE.md "Inference-time
  scope". The Stage 6b CTX-source features ARE inference-compatible (SeamMap is
  public).
- **`conda run python -c` rejects multiline strings** — write probes to files
  under `scripts/probes/_*.py`.
- **CUMINDEX is downloaded but unused**: `cache/pds_ctx_cumindex.{lbl,tab}`
  (91 MB total). The Murray Lab SeamMap embeds illumination angles directly.
  Kept on disk in case Stage 6e seam-distance is ever taken.
- **`models/*` and `dataset_v2*/*` are gitignored** — sweep artefacts and
  augmented feature parquets don't persist across machines. Persistent record
  lives in `scripts/probes/_diag_*.md`, `scripts/probes/_stage6c_*.md`, and
  `docs/modeling_results.md`.
- **230 pytest pass baseline.** Run `pytest tests/ -q` before any promotion.
- **Stage 6c-specific realisation**: per-fold PR-AUC is rank-invariant within a
  held-out image, so Strategies B (down-weighting) and C (normalisation) only
  move pooled-global metrics, not the per-fold mean targeted by Strategy A's
  strict acceptance. If future per-image gate work is considered, the
  acceptance bar should be set on pooled-global, not per-fold mean.
- **AskUserQuestion before**: full-v2 sweeps (Brian-gated; expensive), `git
  commit`, destructive operations on cached artefacts.

## What we know vs what we suspect (after Stage 6c)

Per [`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md) Problem catalog:

- ✓ **Problem 1 (target distribution noise)**: solved by P2 (`boulder_count`),
  dev +22 % PR-AUC. Awaiting full-v2 promotion.
- ◐ **Problem 2 (compression)**: P1 fixes presence-head only; magnitude-head
  shrinkage remains. Ship as ranker, not calibrated regressor.
- ◐ **Problem 3 (per-image anti-signal)**: mechanism identified (Stage 6b H3
  check) and quantified (Stage 6c). Residual gap not closeable from CTX
  provenance at n=38 — needs more LOIO images or HiRISE-side priors.
- ◐ **Problem 4 (no surrounding spatial context)**: Stage 6a DEV-PARTIAL —
  5 × 5 @ S=32 PASSES strict criteria. Full-v2 promotion deferred.
- ✓ **Problem 5 (metric framing)**: P3+P4 doc reframes (queued).
- ✗ **Problem 6 (5 m/px CTX texture floor)**: unresolved. Stage 6a + Stage 6b
  + Stage 6c each produced small lifts — collectively supporting the
  texture-floor reading. Per-tile signal is *near a ceiling* that further
  feature engineering on CTX alone is unlikely to break.

## How this session should report progress

Same as previous handoff:
1. **`PROMOTION_QUEUE.md`**: move passed items to "Promoted"; failed items to
   "Tried, didn't work".
2. **`DECISIONS.md`**: one entry per promoted item with full-v2 numbers.
3. **`docs/modeling_results.md`**: §9 update if P3/P4 land; §15 if a new probe
   ships.
4. **Memory**: `project_state_2026-XX-XX.md` with day's outcomes; mark previous
   superseded.
5. **`HANDOFF_NEXT_SESSION.md`**: rewrite priority order based on what landed.
6. **AskUserQuestion before `git commit`** of any of the above.
