# Pending rebuild — code fixed, artifacts not yet regenerated

> **Critical update (2026-08-06):** do not execute this rebuild or an unfiltered/slow test suite until
> the isolation gates in [CODE_REVIEW_AUDIT_2026-08-06.md](CODE_REVIEW_AUDIT_2026-08-06.md) are closed.
> That audit supersedes this file's older claims that the full suite cannot write live data, records
> the complete Stage 2 → Stage 3 → Stage 4/4b → Stage 5 → model/calibration → baseline+A1 map chain,
> and excludes superseded v1 from the rebuild.

**Purpose.** Brian's policy (2026-08-04): as review findings are fixed, **apply the code fix but defer
the re-run**, batching every rebuild-requiring change into one pass once the review is complete — so
the expensive stages are re-run once, not once per fix.

**The cost of that policy is deliberate artifact drift.** Between now and the rebuild, the current
on-disk artifacts are *not* what the current code would produce. That is exactly the Pattern-D failure
mode the review named, so it is recorded here loudly rather than left implicit. **Anything in this table is a
known, accepted divergence — do not re-file it as a finding.**

> **Before the rebuild:** follow the safety gates and complete dependency DAG in the 2026-08-06 audit;
> this table alone is not an execution plan. Then re-derive every number the "invalidates" column names
> and update the docs that quote them.
> **After the rebuild:** empty this table and say so in `DECISIONS.md`.

## Fixes applied, rebuild outstanding

| # | Finding | Fix applied | Stages to re-run | What it invalidates |
|---|---|---|---|---|
| 1 | **R74** — the HiRISE coverage mask calls deep-shadow pixels "no coverage" | `src/ctx_retrieve.py` — new `_fill_interior_shadow_holes`, called from `build_hirise_coverage_mask`; new kwarg `max_interior_hole_px=16` (`0` restores the old behaviour). Commit: see below. | Isolated Stage 2 → Stage 3 → Stage 4/4b → Stage 5 → fresh embeddings and forced LOIO predictions → deployable head(s) → calibration(s) → fresh baseline and A1 regional mosaics | **The label basis itself.** The isolated R74 counterfactual under existing Stage-3 shifts and current R23/R29 behavior recovers ~3,236 S=32 tiles (1.97 %), 93 % rich, holding 7.70 % of detected boulder area, and moves rich prevalence 0.3598 → **0.3733**. These are not guaranteed final-rebuild counts because Stage 3 and R23/R29 can change alignment, eligibility, and targets. Every prevalence-dependent statistic (`pr_auc@1e-2`, `precision@5%`), the frozen recipe's headline numbers, calibrator, and deployed maps must be regenerated. |

### R74 — read-only validation done at fix time (tests/provenance still required before rebuild)

Run read-only against the 138 **cached decimated** 5 m/px arrays (never a JP2, never the producer):

- Every re-marked pixel has **DN exactly 0** across all 12 images sampled — they are the shadow zeros,
  nothing else.
- `ESP_017355_2260` re-marks **1,185 px**, reproducing the reviewer's independently measured interior-zero
  count exactly.
- Asserted and passing: the fix **only ever adds** coverage, **never alters the swath border** (so the
  rotated-rectangle geometry and any edge-connected missing scan are untouched), and
  `max_interior_hole_px=0` is an **exact no-op**.
- Scale check: 0.0048 % of *valid pixels* are re-marked, but 1.97 % of *tiles* are recovered — the
  amplification is Stage 4's unanimous `mask_min == 1` rule, which is the point of the finding.
- Historical fix-time evidence only: `pytest -m "not slow"` → **490 passed, 21 deselected**, identical
  to the review baseline. This is not a current safety endorsement; marker assignments must be checked
  before any later run.

## Incidental: three v1 cache files carry the R74 fix while their labels do not

On **2026-08-05 09:16**, a full-suite run made *before* the R77 redirect landed regenerated
`cache/ctx_windows/ESP_069669_2220.{tif,json}` and **`ESP_069669_2220_hirise_mask.tif`** — via
`test_stage2_one_image.py`, one of the three producer tests the review had not found. Because the
**R74 fix was already applied**, that mask now has its interior shadow holes filled, while the v1
labels for the same image (restored 2026-08-04 15:21) were built against the *pre*-R74 mask.

So for this one v1 image the mask and labels are one generation apart. The whole v1 tree is already a
stale generation (**R81**) and is now explicitly superseded, so this historical drift is accepted and
v1 will not be rebuilt. It remains recorded so historical results are interpreted against the correct
tree. **Do not repeat the full-suite run:** the Stage 2/3 fixture's hard-linked mutable derived TIFF
leaves a conditional write-through path, documented in the 2026-08-06 audit.

## Stale artifacts NOT caused by a deferred fix

### Frozen historical artifacts — documentation only, no rebuild

| Artifact | Finding | Note |
|---|---|---|
| The whole v1 `dataset/` label tree | **R81** | Superseded historical generation; no rebuild planned. Pre-2026-06-10 y-sign fix; Stage 4 was re-run for v2 only. Every v1 label sits 236–493 m south of its CTX texture. |
| v1 `dataset/splits/within_image_4fold.json` | **R92 / R97** | 543 of 27,307 S=32 tiles (1.99 %) are in a different quadrant than today's splitter assigns. Preserve as historical drift; R45 is a separate matched-quadrant scoring defect. |

### Active v2 artifacts — address in the rebuild or retire explicitly

| Artifact | Finding | Note |
|---|---|---|
| `packaged/loio_nfold_ctx_illum`, `packaged/loio_nfold_nbr_s5` | **R82** | Pre-sign-fix targets: 65.4 % / 88.3 % of `fractional_area` values differ; 19.4 % / 13.6 % of tiles flip the frozen class. Provenance is *inverted* — the stale ones' `config_hash` matches, the current ones' does not. |
| `dataset_v2/features/**` derived caches | `features-deep` | Two generations stale (the sidecars themselves are clean). |

## Not yet fixed — will add rebuild cost when they are

- **R23** (blocker) — two cohort images' labels are a score-rank truncation. Its natural fix is blocked
  by **R56**, whose comparison must be re-run with the target held fixed first.
- **R38** — A1's clip floor collides with the nodata sentinel. A1 is now an active planned parallel
  regional product, so land the explicit-nodata-mask fix before generating it. `[1,255]` may be used
  diagnostically but does not satisfy the product gate because it moves information loss to DN 1.
- **R03 / R83 / R84** — the pixel-scale size floor. R83/R84 estimate a roughly 78/22 mixture in the
  calibration tile pool (not the image pool; not independently verified), so per-image
  `map_scale_mpp` alone is **not sufficient** for PLAN_RegionalMap leg 4. Decision 2026-08-06: retain
  and explicitly document the mixed-floor primary product. A separately identified common-floor
  target remains optional and must not reuse primary-product labels, model, calibration, or claims
  implicitly.
