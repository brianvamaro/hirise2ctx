# Handoff prompt — next session

**Last updated 2026-05-31 after Stage 6b full-v2 LOIO sweep + H3 mechanism check.**

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run -n geospatial python …` (never the
env's `python.exe` directly — see memory note [[conda_location]]).

## Read in this order before starting

1. **Memory** `project_state_2026-05-31.md` (CURRENT) — today's outcome + Stage 6c
   plan in one place.
2. **[`docs/modeling_results.md`](docs/modeling_results.md) §13** — full Stage 6b
   writeup (table, H3 falsification, per-image bimodality, decision).
3. **[`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md)** — Problem 3 status (H3 falsified +
   Stage 6e validated); Stage 6c entry (now the prioritised next bet with concrete
   features + acceptance criteria); Stage 6b moved to ◐ DEV-PARTIAL with full result.
4. **[`scripts/probes/_diag_stage6b_h3_check.md`](scripts/probes/_diag_stage6b_h3_check.md)**
   — H3 correlation table + per-image deltas (the table of significant correlations
   is the substantive finding).

## Where we are

**Stage 6b is implemented end-to-end and dev-validated as ◐ DEV-PARTIAL.** Strict
criteria FAIL on full-v2 LOIO (PR-AUC Δ +0.017, need +0.03; Spearman Δ +0.008,
need +0.05). BUT the **mechanism check on n=38 settled both H3 and Stage 6e**:

- **H3 (CTX-source illumination angle) is FALSIFIED**: `mean_ctx_incidence` ↔
  per-image AUC ρ = −0.213 (p > 0.05). Not a driver.
- **Stage 6e mechanism (CTX-source heterogeneity / mosaic stitching) is
  VALIDATED**: `mean_n_sources` ↔ Spearman ρ = **−0.405 (p = 0.012)**;
  `std_ctx_incidence` ↔ PR-AUC = **−0.370 (p = 0.022)**;
  `dominant_source_frac_mean` ↔ Spearman ρ = **+0.394 (p = 0.014)**.

**Per-image bimodality is the actionable insight**: the two canonical anti-signal
images (ESP_054000_2255, ESP_064510_2260) BOTH improve substantially (ΔPR-AUC
+0.055 and +0.207 respectively); but other images regress (ESP_055690_2200
ΔSpearman −0.780). Net flat on average. Using the features as per-tile inputs leaks
the signal both helpfully and harmfully.

**The data points to Stage 6c (per-image anti-signal gate)** as the next bet —
use the validated features as inputs to an *image-level* classifier, not as per-tile
model inputs. Predictor table is already in `dataset_v2/features_ctx_illum/`; the
per-image baseline AUC training labels are in
`models/_sweep_stage6b/20260531T020308Z/summary.parquet`.

## Goal of this session (Brian to decide at start)

**Recommended path: Stage 6c — anti-signal image gate.** Concrete plan +
acceptance in [PROMOTION_QUEUE.md "Stage 6c"](PROMOTION_QUEUE.md). Three
sub-options to discuss:

- **A. Stage 6c — anti-signal gate (Recommended)**: ~1 day. Implement and dev-test a
  per-image gate using the now-validated features. Strict criterion: gated PR-AUC on
  retained "good" images >= 0.65 mean (vs 0.54 full-set baseline) AND retained-tile
  fraction >= 70 % AND gated normalised lift >= +0.10 over un-gated baseline on
  retained set. Brian-gated for the full-v2 dev sweep.

- **B. Bank wins — P1+P2 full-v2 promotion + P3+P4 doc reframe**: ~2-3 hr total.
  Bank the validated +22 % PR-AUC dev gain at full v2; promote PR-AUC + lift@top-K
  as headline metrics. Defers Stage 6c. Useful if you want the deliverable
  numbers anchored before the next research bet.

- **C. Pivot to Stage 7 — compositional HiRISE 3-band feasibility test**: 1-2 days.
  [`PLAN_Compositional.md`](PLAN_Compositional.md) §7.0. De-risks a separate
  research thread that's not bound by the 5 m/px CTX texture floor.

## Before doing anything

**Working tree status (start-of-session):** clean. Last commit will be (when
this session commits):

- HEAD (this session): Stage 6b implementation + sweep + H3 finding + doc updates

**AskUserQuestion before doing**: full-v2 sweeps (Brian-gated; expensive), `git
commit`, destructive operations on cached artifacts, downloading new external data.

## Path A detail — Stage 6c (anti-signal gate)

Full spec in [`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md) "Stage 6c — anti-signal
image gate". Implementation outline:

1. Build a per-image predictor table from
   `dataset_v2/features_ctx_illum/*.parquet`: for each ObsId, aggregate at S=64 →
   `mean_n_sources`, `std_ctx_incidence`, `mean_dominant_source_fraction`. Total:
   38 rows × 3+ features.
2. Per-image baseline AUC labels: read
   `models/_sweep_stage6b/20260531T020308Z/summary.parquet`, filter to
   `scheme=loio_nfold`, `scale_idx=3`, take per-fold (= per-held-out-image) `pr_auc`,
   `spearman_rho`, `presence_auc`. Three labels to test.
3. Train + cross-validate a small classifier/regressor (logreg or LightGBM):
   leave-one-image-out, predict each label. Report cross-validated ROC-AUC + threshold
   sweep.
4. Build the gate strategies (see PROMOTION_QUEUE.md "Gate strategies" A/B/C):
   headline exclusion, prediction down-weighting, per-image normalisation. Test on
   the existing LOIO sweep summary.
5. Acceptance check + writeup. AskUserQuestion before any expensive sweep.

Cost: ~1 day. The features and labels already exist on disk; the gate model is small
(38 rows, 3 features). The expensive part is interpreting + documenting the result.

## Path B detail — P1+P2 full-v2 + P3+P4 reframe

Per the original handoff:

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

P3 / P4 are doc-only — update [`docs/modeling_results.md`](docs/modeling_results.md)
§9 headline metrics + change default binary in
[`src/modeling/binary_target.py`](src/modeling/binary_target.py) from `bc_ge_1` to
`fa_gt_1e-2`. **AskUserQuestion before each expensive sweep.**

## Path C detail — Stage 7.0 feasibility

[`PLAN_Compositional.md`](PLAN_Compositional.md) §7.0 unchanged from previous
handoff. 1-2 day feasibility test on 2-3 images using actual BoulderNet labels
(not model predictions) to test whether HiRISE 3-band spectra of boulder-rich
areas differ from surroundings ([Delamere 2010](https://doi.org/10.1016/j.icarus.2009.03.012)).

## Critical gotchas (carry forward)

- **Inference-time scope**: model features must be derivable from CTX alone. HiRISE
  LBL angles are diagnostic-only. See PROMOTION_QUEUE.md "Inference-time scope". The
  Stage 6b CTX-source features ARE inference-compatible (SeamMap is public).
- **`conda run python -c` rejects multiline strings** — write probes to files under
  `scripts/probes/_*.py`. Hit again 2026-05-31 with the SeamMap-inspect probe.
- **CUMINDEX is downloaded but unused**: `cache/pds_ctx_cumindex.{lbl,tab}`
  (91 MB total). The Murray Lab SeamMap embeds illumination angles directly; CUMINDEX
  fall-back was unnecessary. Kept on disk in case Stage 6e / Stage 6c needs it.
- **`models/*` and `dataset_v2*/*` are gitignored** — sweep artifacts and augmented
  feature parquets don't persist across machines. The readable tables in
  `scripts/probes/_diag_*.md` and the writeups in `docs/modeling_results.md` are the
  persistent record.
- **230 pytest pass baseline** (was 220 + 10 new for Stage 6b). Run `pytest tests/ -q`
  before any promotion.
- **AskUserQuestion before**: full-v2 sweeps (Brian-gated; expensive), `git commit`,
  destructive operations on cached artifacts.
- **Stage 6b augmented features + repackaged dirs already exist** on disk:
  `dataset_v2/features_ctx_illum/` (38 parquets, ~3.6 M tiles), `dataset_v2_dev/`
  equivalents, plus packaged versions for both LOIO and within_image schemes. No
  re-augmentation needed for Stage 6c — it reads the same parquets.

## What we know vs what we suspect (honest status after Stage 6b)

Per [`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md) Problem catalog:

- ✓ **Problem 1 (target distribution noise)**: solved by P2 (`boulder_count`),
  dev +22 % PR-AUC. Awaiting full-v2 promotion.
- ◐ **Problem 2 (compression)**: P1 fixes presence-head only; magnitude-head
  shrinkage remains. Ship as ranker, not calibrated regressor.
- ◐ **Problem 3 (per-image anti-signal)**: **mechanism narrowed by Stage 6b.** H3
  (illumination) FALSIFIED. Stage 6e mechanism (CTX heterogeneity) VALIDATED.
  Stage 6c is the data-pointed-to fix.
- ◐ **Problem 4 (no surrounding spatial context)**: Stage 6a DEV-PARTIAL —
  5 × 5 @ S=32 PASSES strict criteria. Full-v2 promotion deferred.
- ✓ **Problem 5 (metric framing)**: P3+P4 doc reframes (queued).
- ✗ **Problem 6 (5 m/px CTX texture floor)**: unresolved. Stage 6a + Stage 6b
  both produced small lifts in operational metrics, supporting the texture-floor
  reading; the per-tile signal is *near a ceiling* that Stage 6c (gating, not
  feature addition) is a different angle on.

## How this session should report progress

Same as previous handoff:
1. **`PROMOTION_QUEUE.md`**: move passed items to "Promoted"; failed items to
   "Tried, didn't work".
2. **`DECISIONS.md`**: one entry per promoted item with full-v2 numbers.
3. **`docs/modeling_results.md`**: §14 for Stage 6c result (or §9 update if P3/P4
   land).
4. **Memory**: `project_state_2026-XX-XX.md` with day's outcomes; mark previous
   superseded.
5. **`HANDOFF_NEXT_SESSION.md`**: rewrite priority order based on what landed.
6. **AskUserQuestion before `git commit`** of any of the above.
