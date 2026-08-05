# Pending rebuild — code fixed, artifacts not yet regenerated

**Purpose.** Brian's policy (2026-08-04): as review findings are fixed, **apply the code fix but defer
the re-run**, batching every rebuild-requiring change into one pass once the review is complete — so
the expensive stages are re-run once, not once per fix.

**The cost of that policy is deliberate artifact drift.** Between now and the rebuild, the committed
artifacts are *not* what the current code would produce. That is exactly the Pattern-D failure mode the
review named, so it is recorded here loudly rather than left implicit. **Anything in this table is a
known, accepted divergence — do not re-file it as a finding.**

> **Before the rebuild:** work top-down through this table, re-run the listed stages once, then re-derive
> every number the "invalidates" column names and update the docs that quote them.
> **After the rebuild:** empty this table and say so in `DECISIONS.md`.

## Fixes applied, rebuild outstanding

| # | Finding | Fix applied | Stages to re-run | What it invalidates |
|---|---|---|---|---|
| 1 | **R74** — the HiRISE coverage mask calls deep-shadow pixels "no coverage" | `src/ctx_retrieve.py` — new `_fill_interior_shadow_holes`, called from `build_hirise_coverage_mask`; new kwarg `max_interior_hole_px=16` (`0` restores the old behaviour). Commit: see below. | Stage 2 (coverage masks) → Stage 4 (labels) → features/embeddings for the recovered tiles → head retrain → recalibrate → regional map | **The label basis itself.** Recovers ~3,236 S=32 tiles (1.97 %), 93 % of them rich, holding 7.70 % of all detected boulder area. Moves the cohort's rich prevalence 0.3598 → **0.3733**, so every prevalence-dependent statistic (`pr_auc@1e-2`, `precision@5%`), the frozen recipe's headline numbers, the calibrator's quantile grid, and the deployed map's upper range all shift. |

### R74 — validation done at fix time (no rebuild required to trust the fix)

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
- `pytest -m "not slow"` → **490 passed, 21 deselected**, identical to the review baseline.

## Incidental: three v1 cache files carry the R74 fix while their labels do not

On **2026-08-05 09:16**, a full-suite run made *before* the R77 redirect landed regenerated
`cache/ctx_windows/ESP_069669_2220.{tif,json}` and **`ESP_069669_2220_hirise_mask.tif`** — via
`test_stage2_one_image.py`, one of the three producer tests the review had not found. Because the
**R74 fix was already applied**, that mask now has its interior shadow holes filled, while the v1
labels for the same image (restored 2026-08-04 15:21) were built against the *pre*-R74 mask.

So for this one v1 image the mask and the labels are one generation apart. Low stakes — the whole v1
tree is already a stale generation (**R81**) and nothing live reads it — but it is real drift and it is
recorded here rather than left to be rediscovered. The batched rebuild resolves it. **No further writes
are possible:** the full suite is now verified non-mutating.

## Known-stale artifacts NOT caused by a deferred fix

Listed so the rebuild pass can sweep them at the same time; each has its own finding.

| Artifact | Finding | Note |
|---|---|---|
| The whole v1 `dataset/` label tree | **R81** | Pre-2026-06-10 y-sign fix; Stage 4 was re-run for v2 only. Every v1 label sits 236–493 m south of its CTX texture. |
| `packaged/loio_nfold_ctx_illum`, `packaged/loio_nfold_nbr_s5` | **R82** | Pre-sign-fix targets: 65.4 % / 88.3 % of `fractional_area` values differ; 19.4 % / 13.6 % of tiles flip the frozen class. Provenance is *inverted* — the stale ones' `config_hash` matches, the current ones' does not. |
| v1 `dataset/splits/within_image_4fold.json` | R45 / `labeling-deep-artifact` | 543 of 27,307 S=32 tiles (1.99 %) in a different quadrant than today's splitter assigns. v2 LOIO splits are sound and structurally cannot drift. |
| `dataset_v2/features/**` derived caches | `features-deep` | Two generations stale (the sidecars themselves are clean). |

## Not yet fixed — will add rebuild cost when they are

- **R23** (blocker) — two cohort images' labels are a score-rank truncation. Its natural fix is blocked
  by **R56**, whose comparison must be re-run with the target held fixed first.
- **R38** — A1's clip floor collides with the nodata sentinel. Cheap, and nothing shipped carries it
  (`reports/map_a1/` does not exist), so it only needs to land before any A1 map is built.
- **R03 / R83 / R84** — the pixel-scale size floor. R84 establishes that emitting `map_scale_mpp` per
  image is **not sufficient** for PLAN_RegionalMap leg 4, because the deployed layer's floor is a 78/22
  *mixture*; that needs a product-level attribute, decided before the rebuild.
