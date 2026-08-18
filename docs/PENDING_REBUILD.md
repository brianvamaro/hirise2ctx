# Pending rebuild — code fixed, artifacts not yet regenerated

> **Update (2026-08-18): the isolation gate is CLOSED — the rebuild is unblocked.** All five criteria
> in [CODE_REVIEW_AUDIT_2026-08-06.md](CODE_REVIEW_AUDIT_2026-08-06.md) are met; criterion 5's snapshot
> is `D:\HiRISE2CTX Backup` (11,260 files / 125.55 GB on an independent USB device, verified 8/8 roots
> at 0 missing / 0 extra / 0 size mismatch — `scripts/backup_artifacts.ps1`, DECISIONS 2026-08-18).
> **Two things this does NOT license.** The runtime write guard is still **test-only**, so hand-run
> producers and notebooks remain governed by the absolute-scratch-root discipline — a mistake is now
> *recoverable*, not *harmless*. And the snapshot is point-in-time: once the rebuild starts writing it
> becomes the only record of the pre-rebuild state, so do not refresh it mid-rebuild.
>
> *Superseded, retained for the record —* **Critical update (2026-08-06):** do not execute this rebuild or an unfiltered/slow test suite until
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
| 2 | **R27** — `lacunarity_shadow_b*` emitted `0.0`, an out-of-range sentinel, for shadow-free tiles | `src/features.py::_lacunarity_per_tile` — a tile with `M1 == 0` now stays NaN (the array is NaN-prefilled) instead of being written `0.0`. `dataset/DATA_DICTIONARY.md` documents the case. Verified read-only before the change: **42,015 / 198,320 = 21.2 %** of S ≥ 32 rows in `dataset_v2/features/` were exactly `0.0`, **every one** with `shadow_fraction == 0`, smallest non-zero value exactly `1.0`, **zero** rows in `(0, 1)`. | Stage 4b features for all 38 v2 images, then every Stage-6a `features_nbr_*` derived from them, then any model/metric trained on those columns | **Two columns directly, six downstream.** `lacunarity_shadow_b2/_b4` change from `0.0` to NaN on 21.2 % of S ≥ 32 rows. The real damage is Stage 6a: its neighbour aggregation is NaN-aware (`np.isfinite`) but not sentinel-aware, so it averaged the sentinel with real measurements — **2.16 %** of `nbr_mean_lacunarity_*` rows pooled, and **16.7 %** for `ESP_068402_2240`, currently sit in the impossible interval `(0, 1)`. `nbr_{mean,max,std}_lacunarity_shadow_b{2,4}` all change. LightGBM can split away a `0.0` in the base column; nothing can undo an average. |
| 3 | **R28** — Canny thresholds were absolute, and the config asserted the opposite | `src/features.py::_compute_canny_window` gained a config-driven `use_quantiles` (hard error if enabled without explicit percentile thresholds), and **the default is now `use_quantiles: true`, `0.80 / 0.90`** (Brian's choice, 2026-08-06) in `DEFAULT_FEATURES_CFG`, `config.yaml` and `config_v2.yaml`. `dataset/DATA_DICTIONARY.md` corrected. A test asserts the two YAMLs agree with the default, since a YAML `features:` block overrides key-by-key. | Stage 4b for all 38 images → every Stage-6a `features_nbr_*` → every GBM / W1 number and error atlas computed from an `edge_*` column | **Both `edge_*` columns for every tile, plus six `nbr_*` derivatives.** Verified before the change: per-image `edge_density` tracked per-image `intensity_std` at Spearman **ρ = 0.965** across the 38 images with a **12.2×** spread, and **33.8 %** of `ESP_068402_2240`'s S = 64 tiles had zero Canny edge pixels. On a synthetic scene a ~3× DN-spread cut collapsed edge density **×0.01** under absolute thresholds and **×1.00** under quantiles. Expect the cohort spread in `edge_density` to shrink sharply; that is the point. Trap: an explicit `0.1` is **not** the old behaviour — skimage divides explicit *absolute* thresholds by `dtype_max`, i.e. 0.1/255 on a uint8 window. |

### R74 — tests and provenance landed 2026-08-06

The audit's two pre-rebuild conditions are met, so R74 can now serve as a rebuild boundary:

- **Ten direct synthetic tests** (`tests/test_coverage_mask_shadow_fill.py`): small enclosed hole,
  hole above the threshold, inclusive threshold boundary, edge-connected invalid region, the mixed
  enclosed-plus-edge-connected case, `max_interior_hole_px <= 0` as an exact no-op, add-only over
  random fields, all-valid and all-nodata.
- **Machine-readable identity.** `ctx_retrieve.max_interior_hole_px` is a config key wired through
  `scripts/run_stage2.py`. `build_hirise_coverage_mask` returns `(path, fraction, provenance)` with
  `method`, `version` (1 = pre-R74, 2 = post-R74), threshold, filled-pixel count and the mask's
  SHA-256. Stage 2 persists that plus `ctx_window_sha256`; Stage 3 records both input digests, the
  mask identity, and a `shift_id` digest over its own shift + inputs; Stage 4 records
  `inputs.{ctx_window_sha256, hirise_mask_sha256, coverage_mask, coreg_shift_id}`.
- A regression test flips one mask pixel and asserts the Stage 4 sidecar changes while the config
  hash does not — the Pattern-D gap in one assertion.

**Every existing sidecar predates these fields**, so absence of `inputs` / `hirise_mask` is itself the
marker of a pre-2026-08-06 generation.

### R74 — read-only validation done at fix time

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

> **2026-08-06:** the whole v1 `dataset/` tree is also **expendable** — Brian's decision; it is not
> backed up and losing it is accepted. Everything below about v1 records a *conclusion* that survives
> in git; what would be lost is the ability to re-derive it. See
> [ARTIFACT_RECOVERY.md](ARTIFACT_RECOVERY.md).

| Artifact | Finding | Note |
|---|---|---|
| The whole v1 `dataset/` label tree | **R81** | Superseded historical generation; no rebuild planned. Pre-2026-06-10 y-sign fix; Stage 4 was re-run for v2 only. Every v1 label sits 236–493 m south of its CTX texture. |
| v1 `dataset/splits/within_image_4fold.json` | **R92 / R97** | **Not drifted — it was right and the splitter was wrong.** Measured 2026-08-06: v1's persisted `quadrant_definitions` match a step-8 recompute **8 of 8** images, and the earlier "543 of 27,307 S=32 tiles (1.99 %) disagree with today's splitter" was disagreement with the R97-inflated step-16 splitter. With R97 fixed, v1 agrees again. Still not rebuilt (v1 is superseded for other reasons — R81). R45 is a separate matched-quadrant scoring defect. |

### Active v2 artifacts — address in the rebuild or retire explicitly

| Artifact | Finding | Note |
|---|---|---|
| `packaged/loio_nfold_ctx_illum`, `packaged/loio_nfold_nbr_s5` | **R82** | Pre-sign-fix targets: 65.4 % / 88.3 % of `fractional_area` values differ; 19.4 % / 13.6 % of tiles flip the frozen class. Provenance is *inverted* — the stale ones' `config_hash` matches, the current ones' does not. |
| `dataset_v2/features/**` derived caches | `features-deep` | Two generations stale (the sidecars themselves are clean). |
| `dataset_v2/splits/within_image_4fold.json` + `packaged/within_image_*` | **R97** | **Now stale.** The persisted cuts match the R97-inflated step-16 splitter **38 of 38** images; with the snap step derived from the scales actually present, the cut moves for **29 of 38**. Regenerate with Stage 5 (within-image only — the LOIO split and the regional product are unaffected). |

| 4 | **R29/R75** — Stage 4 shifted the polygons but not the coverage mask, leaving an L-shaped strip along the receding edges eligible where no detection could land | `src/labeling._shift_coverage_mask` translates the mask by the same Stage-3 `(dx, dy)` before eligibility gating, vacated area filled as **ineligible**; opt-out `shift_coverage_mask=False`; sidecar block `coreg_mask_shift` is the generation marker. Shifts are **not** integer-pixel (0/39 measured), so it rounds to nearest whole pixel — residual ≤ 2.5 m vs a 194.7 m median shift. | Stage 4 labels for all 38 v2 images → Stage 4b → Stage 5 → embeddings/LOIO → head → calibration → maps | **The eligible tile set is re-registered.** At S=32, **6,202 tiles lose eligibility** (exactly R75's measured overlap population) on the receding edges and **6,255 gain it** on the advancing edges, net 161,005 → **161,058**. Of the 6,202 lost, 4,915 currently carry `fa > 0` (partially depressed) and 1,287 carry `fa == 0`. The newly eligible tiles have **no labels on disk at all**, so they only materialise on the Stage-4 re-run. Every prevalence-dependent statistic moves; combines with R74, which also moves eligibility. |

| 5 | **R65** — Stage 3's `peak_correlation` is a conditional median, bounded below by the floor it is screened against, and on the block path it does not score the applied shift | `_robust_shift_from_field` now also emits `all_block_peak` (unconditional min/p25/median/p75/max), `confident_fraction`, `median_block_peak_is_conditional` / `block_mad_px_is_conditional`, and `quality_version: 2`; the sidecar gains `peak_correlation_kind`. Deliberately not a new composite — an unconditional median conflates registration quality with scene texture. | Stage 3 for all 39 v2 images (fold into the batched rebuild; the per-block field is not persisted so it cannot be back-computed) | **Nothing numeric.** No shift, label or downstream value changes; `peak_correlation` itself is unchanged. **0 of 39** sidecars carry the new fields today, so the quality figure stays uninterpretable until Stage 3 re-runs. |

| 6 | **R01** — the coarse 32-px grid was anchored to each Murray tile's own pixel origin, so every tile sat on its own sub-cell phase and `rasterio.merge` floored that phase into a whole-cell placement error | Part 1 (`519227c`) added the global grid vocabulary + a merge guard; **part 2** threads `global_grid` through `scripts/map_region.py` **and** `scripts/striping_a1_map.py` in one commit, fixes the window sweep (`tile_aligned=False` + `overlap = 3*tile_px`; either alone still loses cells), adds an executable coverage guard, and records `grid_id` in partials, sidecars and manifests. | **Baseline map for all 26 tiles, then A1 for the 9 CTX-equipped tiles — strictly in that order.** ~16 GPU-h + ~5–7 GPU-h on an L40S; resumable per (tile, window). Then `regional_abundance_mosaic.tif` / `regional_prob_mosaic.tif`, notebook 24 §2/2b/3 and `reports/figures/24_*.png`. | **Every regional output.** Per-tile shape is unchanged (1479×1479 at every phase) and so is the mosaic shape (5925×11852), but the origin moves **+100.0 m E / −80.0 m S** and no resample recovers the sub-cell part (≤0.875 cell = 140 m). 25 of 26 shipped tiles are displaced today, median 140 m, max 198 m. THEMIS leg 1 improves from \|ρ\| 0.0741 → ≥0.0821 (n=26, lower bound). **Ordering is no longer binding** (it was, until R07 on 2026-08-09 moved the A1 statistic off the baseline's grid; the two rows can now be built in either order). **`cache_v2/validation/themis_night_ir_region.tif` must be re-fetched or reprojected** (it was built `--match-mosaic` against the old transform; the 15 GB source is *not* cached, so this is the one genuine network item). After the rebuild it is the *same shape* as the corrected mosaic but 0.625 of a cell out — notebook 24 leg 1 compares the two **by array index**, so `assert_coregistered` now guards it instead of silently correlating displaced cells. **MOLA does not** need re-fetching — it is bounds-derived at 463 m, not mosaic-derived. **Run `map_region.py` with `--force` or a fresh `--out-dir`**: every pre-R01 tile exists, and the driver now refuses to skip an off-lattice product rather than reporting it done. Outstanding before A1 renders: **R38** (clip floor vs nodata sentinel) and **R08** — re-anchoring perturbs A1's per-frame statistics by >1 DN on 11 of 74 measured frames, concentrated in small frames (108–2,068 cells) where the robust IQR is unstable, which is R08's exact population. |

| 7 | **R07** — A1's train/deploy preprocessing statistic was mismatched on **two** axes (training: one *native 5 m* statistic per Stage-2 *window*; deploy: one *160 m area-averaged* statistic per *SeamMap frame*), and the preprocessing arm was recorded nowhere | Both sides now call one definition (`src.striping.A1_ARM`: native 5 m, per dissolved SeamMap frame, over the frame's extent in the parent Murray tile, no pixel left at raw DN). `norm_arm` is stamped into `recipe.json` at train time and folded into `recipe_hash` when declared; `require_norm_arm` refuses a mismatched or (on the A1 path) unverifiable head. `load_frames` no longer needs a rendered abundance raster. | **Re-embed the A1 arm** (`_w2_fang_embed.py --norm a1` over the 38 v2 images) → **retrain the A1 head** → A1 regional map. The statistic itself costs ~3.4 min/tile: ~69 min for the 20 training tiles, ~31 min for the 9 A1 tiles, **no downloads** (all 39 windows' parent tiles are cached). The baseline arm is unaffected numerically but its head should be retrained too, purely to stamp `norm_arm`. | **Every A1 artifact, and the comparability of the two banked A1 numbers.** The η² payoff (0.196 → 0.141) was measured under the 160 m-per-frame definition while the −0.024 AUC skill cost came from the native-per-window store: **they are not comparable and never were**. Both must be re-derived under the single definition before the A1 row is quoted. Deployed A1 inputs previously carried a realised IQR of 37.3 median / 59.6 max against the 27.7 the head was trained on (gain error 1.35× median, 2.15× max) and clipped ~10× more pixels, so predictions change materially. Nothing on the **baseline** map path changes numerically — `map_region.py` applies no A1. |

| 8 | **R13** — the nodata gate tested only the central 32² px of the 96² box the embedder actually consumes (1,024 of 9,216 = 88.9 % unchecked), and neither the threshold nor the masked count was recorded anywhere | `src.mapping.context_zero_fraction` (lattice-block form, 0.016 s/window) + `predict_window(max_context_zero_fraction=0.0)` with **two** counters; both thresholds land in the sweep manifest (closing the R14 resume coupling), the tile sidecar's `nodata_gate` block and the run record, with de-duplicated masked-cell counts and a re-tunable histogram. A1's arm keeps the gate **disabled (1.0) until R38**. DECISIONS 2026-08-10b. | **Fold into row 6/7's pass — it is free there.** Baseline map for all 26 tiles; A1 for the 9 when its own blockers clear. A standalone regeneration would cost ~0.6 GPU-h/tile × 26 for a 1e-5 change and is not worth it. | **Output bytes only; no published statistic moves.** ~770 of 56,870,060 shipped cells turn NaN — measured exactly as **290 of 19,685,689** on the nine tiles with exact block arithmetic, hard ceiling 1,167 map-wide. `prob_mean`, `rich_share_at_0p5`, the sd(log₁₀ pred/label) level table and the 26-tile mosaic are all unaffected at three decimals. **No rebuild is forced by R13 alone.** What *does* change on re-render is the sidecar contract: pre-R13 tiles carry no `nodata_gate` block, so absence marks a pre-2026-08-10 generation. |

| 9 | **R38** — A1 clipped to `[0,255]`, so terrain darker than about `med − 4.51·iqr` was written as the mosaic **nodata sentinel** and thereafter counted as a data gap (6.7 % of deploy-sim tiles carried a false-black pixel; whole tiles went black in low-IQR frames) | Three parts, because moving the floor alone would have made the damage *invisible* rather than absent (R13: DN 0 and DN 1 move the frozen prediction identically to 3 dp). (1) valid pixels floor at `src.striping.A1_VALID_FLOOR = 1`; (2) `predict_window` takes an explicit `nodata_mask` and both A1 drivers derive it from the **raw** DN, so coverage is never re-inferred from a transformed array; (3) the destroyed texture is counted exactly off the existing per-frame DN histogram and recorded per tile as `a1_clip_*` (`--warn-clip-fraction`). Sibling fix: `a1_stats`/`a1_stats_from_hist` returned a fabricated `iqr = 1.0` that **defeated** `a1_stats_native_tile`'s `iqr > 0` guard and gave that frame a 27.7× gain — now NaN. `striping_a1_infer_crop.py` no longer re-inlines the clip. DECISIONS 2026-08-10c. | **Fold into row 7 — it is free there.** R38 changes `a1_apply`, which BOTH the training path (`_w2_fang_embed.py --norm a1`) and the deploy path call, so it rides along on R07's A1 re-embed + retrain. No separate pass. | **Every A1 artifact, and only A1 artifacts.** `dataset_v2/fang_embeddings_a1` and `models/deployable_a1` were baked under floor 0 and must be re-made — which row 7 already requires. Nothing on the record moves: no shipped raster calls `a1_apply` (`reports/map_a1/` does not exist = R06; `reports/map_region/` never imports `src.striping`), and the banked −0.024 LOIO cost and 28 % η² reduction are unaffected. Measured cost of the floor itself under R07's native statistic, streaming whole cached tiles: **0.0015 %–0.0118 %** of valid pixels (`E4_N44` / `E8_N44` / `E-8_N32`), i.e. 3–27× *below* the 0.04–0.41 % the finding was filed with, because the native per-frame IQR (median 30–36) exceeds `A1_REF_IQR = 27.7` so the typical gain is a shrink. New sidecar keys `a1_clip_*`; their absence marks a pre-2026-08-10 generation. |

- All Stage-1 sidecars (`cache_v2/reprojected_detections/*.json`) and all Stage-4 label sidecars
  (`dataset_v2/labels/*.json`) | **R23** | **Provenance-stale, not numerically stale.** The
  2026-08-06 fix adds `source_integrity` + `null_geometry_basis` (Stage 1) and `realised_label_basis`
  (Stage 4); **zero banked sidecars carry any of them**, so the mixed confidence floor is currently
  undocumented on disk. Re-run Stage 1 for all 39, then Stage 4. **Invalidates nothing numeric** — no
  label value, polygon set, or downstream metric changes (verified: `src.dataset.source_digests` is
  bit-identical with and without the new key, so the R04 content digests and the 7 live packages are
  untouched). Until it runs, absence of these keys means *unknown*, not *clean*.

## Not yet fixed — will add rebuild cost when they are

- ~~**R23** (blocker) — two cohort images' labels are a score-rank truncation. Its natural fix is
  blocked by **R56**~~ **R56 CLOSED and R23's remedy DECIDED 2026-08-06** (Brian: retain the mixed
  confidence floor + document it). Root cause is **three byte-truncated `.shp` files** (four across
  the 40 exports), not a BoulderNet export artefact. **Recovery is CLOSED as of 2026-08-09**: a
  filesystem sweep found 6 copies and 0 complete ones, and the three copies of `ESP_017355_2260` are
  bit-identical, so the truncation predates every local copy and the missing 1.17 GB was never on
  this machine. **Brian's ruling: the fix is a v3 re-detection dataset he will supply; v2 proceeds
  as-is and other findings keep being fixed against it.** Retain-and-document is therefore v2's
  *final* disposition, not a holding position. The provenance/documentation half is built; see the
  row above for its rebuild cost, **DECISIONS 2026-08-06o** for the pricing of all four remedies and
  **2026-08-06z** for the recovery measurement.
- ~~**R38** — A1's clip floor collides with the nodata sentinel.~~ **FIXED 2026-08-10; see row 9.**
  The explicit nodata mask was indeed the only acceptable remedy: measured with the real frozen ViT,
  DN 0 and DN 1 damage the embedding **identically to three decimals**, so moving the floor alone
  would have made clip-blackened pixels *invisible* to R13's context gate while leaving the damage
  intact. All three parts landed together, and `--max-context-zero-fraction` is now 0.0 on both arms.
- **R03 / R83 / R84** — the pixel-scale size floor. **R84 CLOSED 2026-08-11b** (product-level half):
  the mixture is measured, verified and recorded. `src/size_floor.py` +
  `scripts/measure_size_floor.py` derive it; `write_geotiff(tags=...)` — which wrote **no metadata at
  all** — now stamps `SIZE_FLOOR_*` on every raster from both map drivers. The 78/22 estimate is
  **independently verified**: 78.3914 % / 21.6086 % of 161,005 S=32 pool tiles, with
  `calibration.npz` `t2_y` max == pool max `fa` == 0.293242 confirming which pool. Image share is a
  *different* number (68.4 / 31.6) and is carried separately. R83's correction also confirms: the
  effective floor is `max(global filter, natural floor)`, so the **fine** cohort is uniform at
  1.5626 m² and the **coarse** cohort is the heterogeneous one (2.9652–5.5719 m², 26 distinct values).
  Decision 2026-08-06 stands: retain and explicitly document the mixed-floor primary product; a
  common-floor target remains optional and must not reuse primary-product labels, model, calibration
  or claims implicitly.
  **Still open, and both need the rebuild:** (i) **R03 item (d)** — persisting per-image
  `map_scale_mpp` + the measured floor into the Stage-1 and Stage-4 sidecars (a producer change);
  (ii) **re-measuring the basis after Stage 4 re-runs** — it is a property of the label pool, so the
  banked JSON goes stale the moment the pool changes. Add `scripts/measure_size_floor.py` to the tail
  of the rebuild DAG, after Stage 4 and before the map drivers.
