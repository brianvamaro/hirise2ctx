# Review area: probes-w1-geospatial

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-02
- **Verification:** self-refuted (single-agent pass; not independently verified)

> **Headline first.** The W1 rung-1 coregistration-sign programme — the one place in this area where a
> real bug was caught and a real fix shipped — **reproduces exactly**. The committed
> `scripts/probes/_w1_shift_rescore{,_postfix}.parquet` give cohort-mean AUC `0.5983` at (0,0) and
> `0.6157` at (+1,0) pre-fix, and `0.6243` peaking at (0,0) post-fix, matching `DECISIONS.md:2551-2585`
> to 3 dp on a **fully balanced 38-image panel** (38/38 images defined at all 25 offsets). The
> migration `_w1_migrate_coreg_sign.py` is complete and non-idempotent-safe, and `_w1_latitude_distortion.py`'s
> arithmetic is right. **The defects below are all in the *ingest-side* probes — the ones that
> characterised the detection population before labelling — and in two aggregate statistics that mix
> populations.**

## Findings

### probes-w1-geospatial-1 — The v2 `detection_filters` decision is recorded from a score distribution over 2.45 M rows, 44 % of which the pipeline deletes before labelling — and it is the exact statistic that would have exposed R23
- **Severity:** medium
- **Liveness:** live-shipped (the `min_confidence: null` / `min_size_m: 1.4105` pair in `config_v2.yaml:104-105` is the shipped label basis)
- **Confidence:** high (reproduced both ways to 3 sig figs)
- **Where:** `scripts/probes/_diag_vclaire_sizes.py:40-41`, `:44`, `:48-50`, `:60-62`; recorded at
  `DECISIONS.md:1203-1208`

`_diag_vclaire_sizes.py` builds one summary table from one GeoDataFrame per image, but filters the two
halves of that table differently: diameters are restricted to rows with a finite area (`:41`, i.e.
polygon-bearing rows) while `score` is taken from every row (`:44`, `:50`), including the
745 k / 330 k **null-geometry** rows that `src/detections.drop_null_geometries` deletes at Stage 1. The
pooled block at `:60-62` therefore reports size over 1,375,638 rows and score over 2,451,145 rows in
adjacent lines. `DECISIONS.md:1203-1208` reads them as one population and turns them into one decision.

I reproduced both numbers. The recorded score line (`100 % ≥ 0.2, 89 % ≥ 0.3, 52 % ≥ 0.5`) is the
**pre-drop** distribution (measured: 100.0 / 88.7 / 52.3 over the 5-image SAMPLE's 2,451,145 source
rows). The same statistic on the rows that actually reach Stage 4 is **100.0 / 97.4 / 77.1**. The
recorded size line (`pooled median 3.4 m, p5 ≈ 1.9 m, ~0 % below the floor`) is the post-drop set
(measured: 3.43 / 1.94 / 0.02 %), and **79.0 % of those rows come from the two R23-truncated images**.

This materially extends **R23**. R23's account of how the truncation escaped notice is "the probe
[`_diag_vclaire_source_nulls.py`] never looked at `score` … one extra line would have shown the rank
truncation immediately." In fact `score` *was* looked at, in the same session, at the very decision it
mattered for — but on the pre-drop rows, where the truncation is invisible **by construction**: every
image reads `min = 0.1000, p50 ≈ 0.51`, uniform across the cohort. Had `sc` been filtered the same way
as `diam`, the probe's own per-image `score[50/90]` column (`:32`, `:46-47`) would have printed:

| ObsId | truncated? | kept rows | score min | p50 | p90 | % ≥ 0.5 |
|---|---|---|---|---|---|---|
| ESP_017355_2260 | **yes** | 359,933 | **0.6173** | 0.698 | 0.764 | **100.0** |
| ESP_068483_2280 | **yes** | 727,160 | **0.4067** | 0.599 | 0.739 | **76.2** |
| ESP_045139_2270 | no | 243,679 | 0.1000 | 0.506 | 0.716 | 51.0 |
| ESP_069669_2220 | no | 35,238 | 0.1000 | 0.519 | 0.708 | 53.2 |
| ESP_055978_2270 | no | 9,628 | 0.1000 | 0.424 | 0.678 | 35.9 |

— i.e. the R23 signature (two images with a hard score floor two to six times the cohort's), on the
same screen as the decision.

- **Failure scenario:** a maintainer re-opening the `min_confidence` question (`DECISIONS.md:3916`
  lists a `min_confidence` label-noise sweep as greenlit work) reads `DECISIONS.md:1205-1207` —
  "52 % ≥ 0.5 … the denser set is *more* boulders, not *smaller*" — and reasons about a cohort whose
  labelled half is actually 77 % ≥ 0.5 with two images at a hard 0.41/0.62 floor. The specific
  inference recorded ("the denser set is *more* boulders, not *smaller*") is the one the truncation
  manufactures: the dense images look "not smaller" precisely because their small, low-confidence
  detections were deleted upstream. Any `min_confidence` value chosen from that distribution is
  calibrated on a population 44 % of which never reaches Stage 4.
- **Evidence:**
  ```python
  # scripts/probes/_diag_vclaire_sizes.py:40-50 -- two populations, one table
  diam = 2.0 * np.sqrt(g.geometry.area.to_numpy() / np.pi)
  diam = diam[np.isfinite(diam)]                    # <- null-geometry rows removed
  ...
  sc = g["score"].to_numpy() if "score" in g.columns else np.array([np.nan])
  s50, s90 = (np.nanpercentile(sc, [50, 90]) if np.isfinite(sc).any() else (np.nan, np.nan))
  ...
  pooled_diam.append(diam)
  if np.isfinite(sc).any():
      pooled_score.append(sc)                       # <- ALL rows, incl. null-geometry
  ```
  ```
  DECISIONS.md:1203-1207
  **Filter decision (`detection_filters`).** Reprojected equivalent-circle diameters are
  large (pooled median 3.4 m, p5 ~ 1.9 m) -> **~0% below the `min_size_m=1.4105` floor**, so
  that filter is a no-op (kept, consistent with v1). Scores: 100% >= 0.2, 89% >= 0.3,
  52% >= 0.5 -- `min_confidence` kept `null`. The denser set is *more* boulders, not
  *smaller*.
  ```
  Reproduced (5-image SAMPLE, `cache_v2/reprojected_detections/*.gpkg` + the source shapefiles):

  | statistic | population | ≥0.2 | ≥0.3 | ≥0.5 | n |
  |---|---|---|---|---|---|
  | as recorded at `:1205-1206` | **all source rows (pre-drop)** | 100.0 % | **88.7 %** | **52.3 %** | 2,451,145 |
  | same probe, post-drop GPKG | labelled polygons only | 100.0 % | 97.4 % | **77.1 %** | 1,375,638 |
  | post-drop, untruncated images only | 3 of 5 images | 100.0 % | 87.8 % | 50.8 % | 288,545 |
- **Self-refutation attempted:** (a) *Is this just R23 restated?* No — R23 is about the labels of two
  images; this is about a **different probe**, a **different recorded number**, and it **corrects
  R23's stated causal account** of how the truncation was missed. The register invites exactly this
  ("only mention one if your own reading materially corrects or extends it"). (b) *Did the probe
  perhaps run post-drop and I am reading history wrong?* Then the score line would read 97/77, not
  89/52; the recorded 89/52 matches the pre-drop set to 0.3 pp and the post-drop set to 25 pp, so the
  attribution is not in doubt. (c) *Does it change a label?* No — `min_confidence: null` means no
  filter fired, so no tile value moves. The damage is to the **record** and to any future re-decision.
  (d) *Is the 79 % pooling weight already noted anywhere?* `grep`ed `DECISIONS.md`, `PLAN_NewDetections.md`
  and `docs/` for the SAMPLE list and for "pooled" near this entry: no.
- **Fix:** in the probe, filter `sc` with the same `np.isfinite(area)` mask as `diam` and print
  `n_score` beside `n_diam`; in `DECISIONS.md:1203-1208`, annotate the score line as pre-drop and add
  the post-drop cohort figure plus the two per-image floors, cross-referencing R23.

### probes-w1-geospatial-2 — The published 5×5-pixel boulder-size audit compares plate-carrée-projected areas to a source-pixel threshold, so `docs/methods.md`'s table under-counts sub-threshold detections by 2.5× overall and 11× on two images
- **Severity:** medium
- **Liveness:** live-shipped (the table is in `docs/methods.md`, the reader-facing writeup CLAUDE.md points non-coders to)
- **Confidence:** high (published table reproduced exactly, then re-derived)
- **Where:** `scripts/probes/_boulder_size_audit.py:64`, `:68-69`; published at `docs/methods.md:178-201`
  and `DECISIONS.md:894-905`

The probe reads the Stage-1 cached GeoPackages, whose CRS is the pipeline `target_crs`
(`config.yaml:15-16`, `Mars_2000_Equidistant_Cylindrical`, verified `"Latitude of 1st standard
parallel" = 0` in every sidecar's `target_crs_wkt`). In that projection E–W lengths at latitude φ are
inflated by `1/cos φ`, so polygon **area** is inflated by `1/cos φ` — 1.41× at the cohort's 45 °N. The
threshold it compares against, `(5 · MAP_SCALE)²`, is in **HiRISE RDR projected metres** (a pixel of
the raster BoulderNet actually ran on). The comment at `:64` asserts the opposite:

```python
# scripts/probes/_boulder_size_audit.py:64
areas = gdf.geometry.area.to_numpy()  # m^2 in CTX CRS (close to true area in this lat band)
```

It is not close: at SP1 = 45° the discrepancy is 41 %. Converting each image's areas back into its own
`.prj` frame (multiply by `cos(SP1)`, read from the sidecar's `source_crs_wkt`) reproduces the
published `n < threshold` column exactly in the uncorrected case, and changes it as follows:

| ObsId | SP1 | px m | n | published n<thr / % | corrected n<thr / % |
|---|---:|---:|---:|---:|---:|
| ESP_055714_2270 | 45 | 0.50 | 1,974 | 7 / 0.35 % | **78 / 3.95 %** |
| ESP_054857_2270 | 45 | 0.25 | 6,462 | 0 / 0.00 % | 2 / 0.03 % |
| ESP_069669_2220 | 40 | 0.25 | 1,462 | 1 / 0.07 % | 3 / 0.21 % |
| ESP_057469_2215 | 40 | 0.50 | 940 | 2 / 0.21 % | **22 / 2.34 %** |
| ESP_071093_2210 | 40 | 0.25 | 961 | 1 / 0.10 % | 1 / 0.10 % |
| ESP_047976_2020 | 20 | 0.25 | 1,346 | 22 / 1.63 % | 25 / 1.86 % |
| ESP_056165_2200 | 35 | 0.50 | 26 | 21 / 80.77 % | 21 / 80.77 % |
| ESP_075577_2105 | 30 | 0.25 | 624 | 9 / 1.44 % | 11 / 1.76 % |
| ESP_039820_1750 | 0 | 0.25 | 497 | 3 / 0.60 % | 3 / 0.60 % |
| **total** | | | 14,292 | **66** | **166 (2.52×)** |

- **Failure scenario:** a reader of `docs/methods.md:193-196` — "small numbers (**0-2 %**) of
  sub-threshold polygons survived in eight of the nine audited images" — takes away that the
  BoulderNet post-filter leak is negligible. For `ESP_055714_2270` it is 3.95 % (78 polygons), outside
  the stated range, and the same sentence is the basis for `:237`'s downstream caveat. The knock-on is
  `DECISIONS.md:892`'s claim that `min_size_m = 1.4105` "**Matches** the Amaro et al. 2026 BoulderNet
  design floor for 0.25 m/px HiRISE binning **exactly**": applied to projected areas, the filter's
  effective floor is 25·cos(SP1) source-pixels — 17.7 px² at 45 °N, 25 px² only at the equator, and
  different for every image.
- **Evidence:**
  ```python
  # scripts/probes/_boulder_size_audit.py:66-69  -- ground-metre threshold vs projected-metre areas
  px_m = hirise_pixel_size_m(obs, cfg.cache_dir)     # PDS MAP_SCALE, HiRISE RDR projected m/px
  if px_m is not None:
      threshold_m2 = (5 * px_m) ** 2
      n_below = int((areas < threshold_m2).sum())    # `areas` is in the SP1=0 target CRS
  ```
  ```
  docs/methods.md:193-196
  filter described above appears to **not have been applied consistently** to
  the priority10 shapefile copies -- small numbers (0-2 %) of sub-threshold
  polygons survived in eight of the nine audited images, and the majority of
  polygons in ESP_056165_2200 are sub-threshold.
  ```
- **Self-refutation attempted:** (a) *Already filed?* `labeling.md`'s *Refuted* section covers
  "`min_size_m` is applied to latitude-inflated projected areas" and rules it "already measured and
  accepted" via `_w1_latitude_distortion.py` / `DECISIONS.md:2741-2751`. That entry is dated 2026-06-10
  and concerns the **v2** cohort's effective floor; it never touches the **v1** audit table, was never
  propagated back to `docs/methods.md`, and does not correct the `n < threshold` column, which is a
  different published number. R03 is a distinct axis (pixel scale, not projection). (b) *Does the
  conclusion flip?* No — correcting makes the leak **larger**, so "the post-filter was not applied
  consistently" is strengthened. That is why this is medium, not high. (c) *Is `MAP_SCALE` a
  ground or a projected scale?* Projected, at the RDR's standard parallel — which is exactly the frame
  `cos(SP1)` maps back into, so the correction is the right one and needs no true-ground assumption.
  (d) *Does the same slip infect the v2 numbers?* Yes, `_diag_vclaire_sizes.py:40` and
  `_w1_rung3_detection_stats.py:47-49` use the same projected diameters, but their consumers report
  operational (filter-effect) fractions, where projected areas are the correct frame.
- **Fix:** multiply `areas` by `cos(SP1)` (read from the Stage-1 sidecar's `source_crs_wkt`, which the
  probe already has on disk) before comparing to `threshold_m2`; regenerate the `docs/methods.md` §2.2
  and `DECISIONS.md:894` tables and change "0-2 %" to "0.03-4 %".

### probes-w1-geospatial-3 — The "presence-AUC coincidence" is an unpaired comparison of means over 26 and 25 different folds, and "23/38 folds changed" is really 23 of 23 comparable folds
- **Severity:** low
- **Liveness:** dead-closed (the metric was retired in the same entry) — but the record's reasoning is wrong
- **Confidence:** high (reproduced exactly)
- **Where:** `scripts/probes/_w1_check4_presence_and_check1_deadfeat.py:27-35`; recorded at
  `DECISIONS.md:2750-2759`

The probe builds `cmp = DataFrame({"pre": pre.presence_auc, "post": post.presence_auc})` and reports
`cmp.pre.mean()` vs `cmp.post.mean()` and `(cmp.delta.abs() > 1e-6).sum()`. `presence_auc` is NaN on
single-class folds, and pandas' `.mean()` skips NaN, so the two means are over **different image
sets**; the NaN sets also differ between the two runs because the label shift changed which images are
single-class. Measured: `pre` is defined on 26 images, `post` on 25, only **23** in common.

- 26-image mean 0.614934 vs 25-image mean 0.614904 → the recorded "collision at 4 dp".
- On the **23 paired** folds: 0.585317 vs 0.581580, a difference of **−0.0037** — no collision.
- `cmp.delta` is NaN wherever either side is NaN, so those 15 rows fail `abs() > 1e-6` and are counted
  as *unchanged*. The honest statement is "**23 of the 23 comparable folds changed**", i.e. **all** of
  them, not 60 % of them.

- **Failure scenario:** `DECISIONS.md:2750-2755` records the collision as "a genuine coincidence, and
  **proof the re-bank consumed the new labels**". The evidentiary weight is misplaced: the proof is the
  paired result (23/23 folds moved, mean |Δ| 0.091), while the "coincidence" is an artifact of
  averaging two different populations, so a future reader looking for a similar consistency check
  copies a method that cannot support the claim. The same NaN-dropping mechanism is **R24**, here
  appearing in a probe rather than in `evaluate.py`'s `mean_std`.
- **Evidence:**
  ```python
  # scripts/probes/_w1_check4_presence_and_check1_deadfeat.py:27-35
  cmp = pd.DataFrame({"pre": pre.presence_auc, "post": post.presence_auc})
  cmp["delta"] = cmp.post - cmp.pre
  n_diff = int((cmp.delta.abs() > 1e-6).sum())          # NaN rows counted as "unchanged"
  ...
  print(f"folds that changed: {n_diff}/{len(cmp)}; mean pre {cmp.pre.mean():.6f} vs post {cmp.post.mean():.6f}")
  ```
  ```
  rows 38; pre non-null 26; post non-null 25; both non-null 23
  skipna mean : pre 0.614934  post 0.614904   (matches DECISIONS.md:2753 exactly)
  paired mean : pre 0.585317  post 0.581580   diff -0.003737
  folds changed >1e-6: 23/38  ==  23/23 comparable
  NaN-pre-only: ESP_045139_2270, ESP_048688_2085   NaN-post-only: ESP_049242_2115, ESP_054134_2265, ESP_066634_2210
  ```
- **Self-refutation attempted:** (a) *Is this R02?* No — R02 is `evaluate.py` computing and printing
  `presence_auc`; this is a probe's *comparison* of two runs and the DECISIONS sentence it produced.
  (b) *Does it matter, given the metric was retired?* Only to the record — hence `low`. (c) *Is the
  NaN pattern maybe stable so the populations coincide?* No: five images flip NaN status between the
  two runs (listed above). (d) *Does the same defect touch `meaningful_auc`?* Checked — it is defined
  on all 38 folds in both runs, so the headline `0.598 → 0.624` is a clean paired comparison.
- **Fix:** compare on `cmp.dropna()` and report `n` beside every mean; count NaN folds in a separate
  column rather than letting them fall into the "unchanged" bucket.

### probes-w1-geospatial-4 — "89.8 % agreement after a 1-tile shift" is a base-rate artifact: chance agreement at this prevalence is 76.9 % (κ = 0.44)
- **Severity:** low
- **Liveness:** dead-closed (explanatory narrative, no gate depends on it)
- **Confidence:** high (both recorded numbers reproduced exactly)
- **Where:** `scripts/probes/_w1_label_autocorr.py:26`, `:30-31`; recorded at `DECISIONS.md:2760-2768`

The probe reports raw agreement between a tile's `boulder_count > 50` label and its neighbour's, and
`DECISIONS.md:2761-2765` converts the pair (ρ = 0.72, 89.8 %) into "the pre-fix 1.1-tile offset
therefore acted as **~10-28 % label noise**, not scrambling". The 10 % end is `1 − 0.898`, and raw
binary agreement is prevalence-dependent: the median per-image base rate of `bc > 50` is **0.384**, so
two *statistically independent* label fields would already agree **76.9 %** of the time. Cohen's κ on
the same data is **0.438** (mean 0.452) — moderate, not "mild noise". Measured over all 38 images:
observed 0.898, chance 0.769, κ 0.438, ρ 0.720 (`DECISIONS` quotes 89.8 % and 0.72 — exact match).

- **Failure scenario:** the recorded floor "~10 % label noise" understates the damage a one-tile shift
  does to the binary target by roughly a factor of five relative to the achievable range (the shift
  destroys 56 % of the above-chance agreement, not 10 % of the labels). The sentence exists to explain
  why the sign fix only bought +0.026 AUC; on the corrected reading the small gain is *less* explained
  by "the labels were still 90 % right", so anyone re-opening "was the sign fix fully effective?" is
  starting from a wrong premise. Same prevalence-dependence family as R26 / `notebooks-5` /
  `probes-stage6-1`.
- **Evidence:**
  ```python
  # scripts/probes/_w1_label_autocorr.py:26,30-31 -- raw agreement, no chance correction
  binary_agree=float(((j.a > 50) == (j.b > 50)).mean()),
  ...
  print(f"\ncohort: median label autocorr at 1 tile = {df.rho.median():.3f}; "
        f"median binary (bc>50) agreement = {df.binary_agree.median():.1%}")
  ```
  ```
  38 images, S=64, dataset_v2/labels:
    median base rate (bc>50)  = 0.384
    median observed agreement = 0.898   <- DECISIONS.md:2763
    median chance   agreement = 0.769
    median Cohen kappa        = 0.438
    median Spearman rho       = 0.720   <- DECISIONS.md:2762
    images where observed <= chance: 1/38
  ```
- **Self-refutation attempted:** (a) *Is the conclusion wrong?* Not necessarily — the ρ = 0.72 half of
  the sentence is prevalence-free and supports "smooth, not scrambled", and the narrative was
  independently corroborated by the rescore surface (+0.018 predicted vs +0.026 delivered). Only the
  10 % end of the "10-28 %" range is an artifact, which is why this is `low`. (b) *Is κ the right
  correction?* Any chance-corrected measure gives the same picture; I report both the raw chance level
  and κ so the reader can pick. (c) *Does anything downstream consume `binary_agree`?* `grep`ed: no
  code reads it; it appears only in the DECISIONS prose.
- **Fix:** in the DECISIONS entry, replace "89.8 % in agreement" with "89.8 % raw agreement against a
  76.9 % chance level (κ = 0.44)" and drop the "~10 %" end of the noise range.

### probes-w1-geospatial-5 — "Seam-tile masking does nothing (improved 29 % of images)" is scored on a denominator 37 % of which cannot move: 14 of 38 images have zero seam tiles
- **Severity:** low
- **Liveness:** dead-closed (the Tier-1 reliability flag was deferred at PLAN_FM §2.7)
- **Confidence:** high
- **Where:** `scripts/probes/_w1_rung4_seam_error.py:50`, `:67-69`, `:74`, `:84-86`; recorded at
  `DECISIONS.md:2664-2668` and as a decision at `:2696`

`delta_single = auc_single − auc_all` is computed for every image and aggregated with
`d = df.delta_single.dropna(); (d > 0).mean()`. For an image whose tiles are all single-source the two
AUCs are computed on the *identical* rows, so the delta is exactly `0.000` — a structural zero, not an
observation. Measured over `dataset_v2/features_ctx_illum` at S=64: **14 of 38 images have zero
multi-source tiles**, pooled seam tiles are **1,168 of 37,315 = 3.13 %**, median per-image seam
fraction 2.96 %, max 11.8 %. The committed `scripts/probes/_w1_rung4_seam_error.md` shows the exact
`0.000` deltas (e.g. `ESP_047976_2020`, `ESP_054622_2240`, `ESP_068483_2280`, `ESP_045550_2180`).

`improved in 29 % of 38` is therefore 11/38 where 14 of the 38 are structurally incapable of being
positive; on the 24 images that actually carry seam tiles it is **11/24 = 46 %** — indistinguishable
from a coin flip, which is the *stronger* form of the same null.

- **Failure scenario:** `DECISIONS.md:2696` records "Seam-tile masking rejected (rung 4)" as a Tier-1
  design decision. A reader takes "improved 29 % of images" as evidence that masking is actively
  harmful in 71 % of cases; the honest reading is that it is untestable on 37 % of the cohort and a
  coin flip on the rest. If the seam-mitigation idea is ever revisited (the CTX source-frame
  radiometry programme did come back, twice), the recorded number understates how little was
  actually measured.
- **Evidence:**
  ```python
  # scripts/probes/_w1_rung4_seam_error.py:50,67-69,84-86
  single = m.ctx_n_sources == 1
  ...
  auc_all=safe_auc(np.ones(len(m), bool)),
  auc_single=safe_auc(single.to_numpy()),
  ...
  d = df.delta_single.dropna()
  lines.append(f"- single-source-only AUC delta: mean {d.mean():+.4f}, median {d.median():+.4f}, "
               f"improved in {(d > 0).mean():.0%} of {len(d)} images")
  ```
  ```
  dataset_v2/features_ctx_illum, scale_idx==3, 38 images:
    seam_frac: median 0.0296  mean 0.0360  min 0.0000  max 0.1181
    images with ZERO seam tiles (delta_single == 0 by construction): 14/38
    pooled seam tiles 1,168 / 37,315 = 3.13%
  ```
- **Self-refutation attempted:** (a) *Is the limitation already disclosed?* Partly —
  `DECISIONS.md:2667` says "within-image seam fractions only 0-12 %", which is honest about the *size*
  of the manipulation but not about the 14 structural zeros inside the reported denominator. That
  disclosure is why this is `low` and not medium. (b) *Does correcting it flip the verdict?* No — 11/24
  is still a null; the correction makes the null cleaner. (c) *Is there a real seam effect being
  missed?* The image-level correlations do replicate (`mean_n_sources` ρ = −0.378 p = 0.019,
  `dom_frac` +0.376 p = 0.020, recomputed from the committed `_w1_reliability_proxy.csv`), and
  `DECISIONS.md:2668` already attributes the mechanism to between-image rather than within-image
  structure — consistent.
- **Fix:** exclude `n_seam == 0` images from the aggregate and report `improved in 11/24 testable
  images (14 of 38 have no seam tiles)`.

## Load-bearing map

| probe | cited by | number it produced | verdict |
|---|---|---|---|
| `_diag_vclaire_sizes.py` | `DECISIONS.md:1203-1208` (unattributed but the exact producer) | v2 filter decision: "pooled median 3.4 m, p5 1.9 m, ~0 % below floor; 100/89/52 % ≥ 0.2/0.3/0.5" | **WRONG population** — finding **-1** |
| `_boulder_size_audit.py` | `docs/methods.md:178-201`, `DECISIONS.md:794-820, 894-905` | per-image 5×5-px sub-threshold counts (9 images, total 66) | **WRONG by 2.52×** — finding **-2** |
| `_w1_check4_presence_and_check1_deadfeat.py` | `DECISIONS.md:2714, 2750-2759`; committed `_w1_dead_features.csv` | "presence AUC 0.614934 vs 0.614904, coincidence; 23/38 folds changed"; dead-feature list | means over 26 vs 25 folds — finding **-3**; dead-feature half correct (found the 2 shadow-dead images; canny gap is R28) |
| `_w1_label_autocorr.py` | `DECISIONS.md:2763` | "ρ 0.72 / 89.8 % agreement → ~10-28 % label noise" | both numbers reproduce; 89.8 % is prevalence-inflated — finding **-4** |
| `_w1_rung4_seam_error.py` | `DECISIONS.md:2648, 2664-2668, 2696`; `scripts/probes/_w1_rung4_seam_error.md`; `reports/figures/w1_rung4_errmap_*.png` | "seam masking does NOTHING; improved 29 % of images"; ρ(mean_n_sources, AUC) = −0.378 | ρ reproduces; the 29 % denominator is 37 % structural zeros — finding **-5** |
| `_w1_shift_rescore.py` | `DECISIONS.md:2551-2556`; committed `_w1_shift_rescore{,_postfix}.parquet` + `.md` | pre-fix cohort surface: 0.598 centre → 0.616 at (+1,0) | **VERIFIED** — 0.5983 / 0.6157, balanced 38/38 panel at every offset |
| `_w1_surface_postfix.py` | `DECISIONS.md:2582-2585` | post-fix surface peaks at (0,0), 0.624 | **VERIFIED** — 0.6243, argmax (0,0), all 24 neighbours lower |
| `_w1_migrate_coreg_sign.py` | `DECISIONS.md:2573-2576`; rewrote 48 cached JSONs | `dy_m = −dy_px·px_y` + `y_sign_fix_applied` marker | **VERIFIED complete** — `magnitude` is sign-invariant, `single_window.dy_m` migrated, assert blocks a double-flip |
| `_w1_sign_error_check.py` | `scripts/probes/_w1_rung1_findings.md:4,54` | best_di vs predicted 2·|dy|/320 correlation | not re-derived (needs the pre-fix grid semantics); logic reads correct |
| `_w1_label_ctx_displacement.py` | `DECISIONS.md:2557-2562`; committed csv; `reports/figures/w1_rung1c_*.png` | applied-position displacement = 2×cached dy | consistent with the confirmed code bug; not re-run |
| `_w1_latitude_distortion.py` | `DECISIONS.md:2716, 2741-2751`; committed csv | "true min-size floor 0.94 m vs 1.16-1.36; GLCM E-W 2.22 vs 3.4-4.6 m/px" | **VERIFIED** — `1.4105·√cos φ` is the correct transform; csv row for ESP_076499_1160 = 0.9389 |
| `_w1_geometry_audit_all38.py` | `DECISIONS.md:2715`; committed csv | "23/39 lock; median residual \|dy\| 0.65, \|dx\| 1.80 px" | disclosed lock rate; guards on the migration marker; not re-derived |
| `_w1_reliability_proxy.py` | `DECISIONS.md:2690-2698`; committed csv | "dispersion −0.15 / feat_shift −0.03 / source stats ~0.38" | **VERIFIED** (−0.150, −0.034, −0.378, +0.376) — but 4 uncorrected tests, neither survives Bonferroni |
| `_w1_rung5_feature_sign.py` | `DECISIONS.md:2648`; `_w1_rung5_feature_sign.md` | per-image feature↔label signs → `texture_decorrelated` class | see *Refuted* (near-circular but the class definition itself is already filed as `probes-fm-recipe-1`) |
| `_w1_rung2_join_audit.py` | `DECISIONS.md:2646, 2652-2655` | "CLEAN: unique keys, zero join loss, exact nesting" | logic checked and sound (outer-join indicator + exact nested sum) |
| `_w1_rung3_detection_stats.py` | `DECISIONS.md:2647, 2656-2661` | "anti vs cohort indistinguishable, all MWU p>0.35" | diameters are projected (same slip as **-2**) but the comparison is within-cohort, so the null holds |
| `_w1_rung3_fullres_visual.py` | `DECISIONS.md:2648`; `reports/figures/w1_rung3_*.png` | visual detection-quality panels | not audited beyond the CRS-consistency claim in its docstring (correct: JP2 and `.prj` share the same uncorrected SP1) |
| `_w1_coreg_vs_auc.py` | `DECISIONS.md:2565-2567`; `_w1_coreg_vs_auc.md` | "coreg quality uncorrelated with per-image AUC" | paired correctly; conclusion is a null and survives |
| `_w1_build_dossier.py` | `dataset_v2/w1_dossier.parquet`, `_w1_dossier.md`, `DECISIONS.md:2649`, and every FM gate | `validity_ok` (27/38), `attributed_cause` | **already filed** as `probes-fm-recipe-1` / `-2`; not re-reported |
| `_w1_pistd_verdict.py` | `DECISIONS.md:2815-2827`; `PLAN_CNN.md:182` | per-image-standardization promotion verdict (all FAIL) | paired Wilcoxon on a common index — correct; see *Refuted* for the pooled-vs-per-fold PR-AUC wording |
| `_w1_shadow_threshold_diag.py` / `_w1_dn_clip_extent.py` / `_w1_verify_shadow_fix.py` / `_w1_shadowfix_compare.py` | `DECISIONS.md:2715, 2718-2727` + the DN-clip fix entry; committed `_w1_shadow_threshold_diag.csv` | dead shadow channel on 2 images; +0.249 / +0.127 AUC after the fix | mechanism chain is coherent (`mode` lands on the DN=1 clip spike); the "bit-identical for every other image" claim is sound because `arr > 0` already excluded DN=0 |
| `_w1_shift_surface.py` | `_w1_shift_surface.md` | per-image 5×5 surfaces + n_neg | correct; the `n_neg` column is what exposed the near-saturated-image caveat |
| `_crater_distance.py` | `DECISIONS.md:2339`; `notebooks/_build_17.py:13,99`; `dataset_v2/crater_distance_v2.parquet`; `reports/figures/stage7_tier2_crater_distance.png` | per-image nearest-crater-rim distances (Tier-2 null) | see *Refuted* — nearest-by-centre ≠ nearest-by-rim, but Spearman(probe, true) = 0.992 at D≥5 km |
| `_diag_vclaire_source_nulls.py` | `docs/CODE_REVIEW_2026-07-31.md:886-899` (R23) | null geometries are upstream, not a reprojection bug | correct as far as it went; R23's account of *why* the truncation was missed is corrected by finding **-1** |
| `_diag_vclaire_detections.py` | `PLAN_NewDetections.md:107` | v2 ingest inspection, SP1 fingerprint, diameter percentiles | the `:52` comment "sub-metre diff" between source and target CRS diameters understates a 19 % scale factor at 45 °N (same family as **-2**) |
| `_diag_tocrs_displacement.py` | `DECISIONS.md:1280-1285` | "`to_crs` is a 0.000 m change at our coordinates" | plausible and mechanistically right (PROJ `eqc` uses the shared semi-major axis; both CRSs are ocentric) but measured on **2 of 39** images |
| `_diag_boulder_localization_fullres.py` / `_diag_boulder_localization.py` | `DECISIONS.md:1289-1298`; `reports/figures/01_localization_fullres_*.png`; notebook 01 | "polygons sit on individual boulders"; centroid gate 0.2-5.0 km | not re-derived (needs imagery); the centroid gate is the `qa.assert_centroid_consistent` family already covered by R30 |
| `_probe_jp2_crs.py` | `DECISIONS.md:334-341` | JP2 ships the same SP1=0 bug as the `.prj` | correct; drove the `hirise_imagery` override |
| `_verify_sp1_fix.py` | `DECISIONS.md:364-366` | ESP_047976_2020 cache SP1 0.0 → 20.0 | correct |
| `_probe_murray_url_variants.py` | `DECISIONS.md:374-386`; live in `src/ctx_retrieve._padded_manifest_form` | the `E<±3digit>_N<±2digit>` convention | live-shipped and covered by `tests/test_murray_url_padding.py` |
| `_check_decimated_sp1.py` | `scripts/probes/README.md:9` | per-TIFF SP1 vs sidecar expected | see *Refuted* — `None == None` reports `OK` |
| `_probe_pyproj_sp1.py`, `_probe_sp1_regex.py`, `_diag_crs_names.py`, `_diag_lbl_center.py`, `_diag_block_shift_field.py`, `_diag_vclaire_geom_validity.py`, `_w1_antisignal_list.py` | not cited outside `scripts/probes/README.md` | — | low priority; read, nothing load-bearing |

## Refuted by my own check

- **The rescore-surface cohort means are averages over different image sets per offset cell** (the
  `pivot_table(..., aggfunc="mean")` NaN-skip pattern that produced finding **-3**). They are not: both
  committed parquets are **fully balanced** — 38 of 38 images have a defined AUC at all 25 offsets, so
  the balanced-panel surface is bit-identical to the reported one and the argmax is unchanged
  ((+1,0) pre-fix, (0,0) post-fix). The load-bearing rung-1 evidence is clean.
- **`_w1_migrate_coreg_sign.py` left derived y-quantities un-migrated.** Checked a cached JSON: the
  only y-derived fields are `shift_m.dy`, `shift_m.magnitude`, `single_window.dy_m` and
  `single_window.magnitude_m`; both magnitudes are sign-invariant and both `dy` fields are rewritten
  (`:26`, `:30`). `block_field` stores only px-space MADs. The migration is complete.
- **`_crater_distance.py:84-88` takes the crater nearest by *centre* and subtracts *that* crater's
  radius, so a large distant crater with a nearer rim is missed.** Real, but immaterial: recomputing
  the true minimum over `d − D/2` for all craters gives probe == true on **34/39** images at
  D ≥ 5 km, mean overstatement 0.54 km, and Spearman(probe, true) = **+0.9919** — the Kruskal-Wallis
  and Mann-Whitney tests behind the Tier-2 null are rank-based, so the verdict at
  `DECISIONS.md:2300-2320` is unaffected. (The `D ≥ 1 km` column is the worst: median 4.4 vs 3.1 km,
  ρ = 0.936. Worth a one-line caveat in `docs/compositional.md` §4.7, not a finding.)
- **`_crater_distance.py:29` hardcodes `MARS_RADIUS_KM = 3389.5` against the invariant-1 "never
  hardcode a radius" rule.** It is a great-circle distance on an auxiliary sphere, not a CRS
  definition; the 0.2 % difference from 3396.19 km is far below the image-centre-as-proxy error the
  notebook already declares.
- **`_w1_rung5_feature_sign.py` is circular: images selected for LOIO AUC < 0.5 must carry
  cohort-inverted within-image feature correlations.** Largely true, and it does license the
  `texture_decorrelated` class — but the class definition itself
  (`_w1_build_dossier.py:78-89`, an AUC threshold feeding a downstream ΔAUC read) is already filed as
  `probes-fm-recipe-1`, and rung 5 adds a genuinely non-guaranteed piece (*which* features flip, and
  that `ESP_064510_2260`'s flip replicates in the independent phase-correlation audit). Not re-filed.
- **`_w1_pistd_verdict.py` scores the pre-declared "pooled PR-AUC delta ≥ −0.01" criterion on per-fold
  PR-AUC from `summary.parquet`, not on the pooled value from `aggregate.parquet`.** True, but moot:
  every arm failed the *first* criterion (paired Wilcoxon on `meaningful_auc`), so the second was never
  binding, and `DECISIONS.md:2820-2825` quotes pooled PR-AUC from the sweep aggregate, not from this
  probe.
- **`_check_decimated_sp1.py:41` compares `cache_sp1 == expected_sp1` where both can be `None`, so a
  WKT with no SP1 parameter at all reports `OK`.** A test-that-cannot-fail shape, but it is a
  throwaway console diagnostic that was superseded by `_verify_sp1_fix.py` (which prints the literal
  value) and by `tests/test_hirise_imagery_sp1_override.py`. Zero severity.
- **`_w1_shift_rescore.py`'s "healthy images give the null for max-over-25-offsets inflation" is an
  invalid null** because healthy images are selected for high centre AUC and so have less headroom.
  Real, and the probe already carries the caveat at `:116-118`; more importantly the rung-1 conclusion
  was confirmed by two independent routes (the direct displacement measurement and the source-code
  bug itself), so no recorded verdict rests on it.
- **`_w1_geometry_audit_all38.py` calls the cohort "clean" from the 23 of 39 images that locked.** The
  probe prints the lock count and `DECISIONS.md:2733-2740` quotes "23/39 achieve lock" verbatim, so the
  population is disclosed, not silent.

## Verified clean

- The committed `scripts/probes/_w1_shift_rescore.parquet` / `_w1_shift_rescore_postfix.parquet`
  reproduce `DECISIONS.md`'s 0.598 / 0.616 / 0.624 to 4 dp, on a balanced 38-image panel, with
  symmetric `n_overlap` shrinkage (37,315 at centre → 30,744 at the corners) that is identical between
  the two runs.
- `_w1_latitude_distortion.py`'s three transforms are all correct for an SP1 = 0 equirectangular target:
  true min-size floor `1.4105·√cos φ`, tile true area `(320 m)²·cos φ`, E–W ground scale `5·cos φ`. The
  committed csv's ESP_076499_1160 row (cos 0.4431, floor 0.9389 m, ×2.257 density, 2.216 m/px) is
  arithmetically exact.
- `_w1_migrate_coreg_sign.py` is safe to re-run: the marker guard plus the
  `assert abs(new + old) < 1e-6` at `:27` fails loudly on an already-correct file (independently
  confirms `labeling.md`'s refutation).
- `_w1_pistd_verdict.py` is the only probe in this area that uses torch-dependent code, and it
  correctly does `import src.modeling` before numpy/pandas (invariant 9).
- `_w1_rung2_join_audit.py`'s integrity checks are genuinely capable of failing: an outer-join
  indicator for both-sided loss, `duplicated(subset=key)`, and an exact `S=32 → S=64` nested-sum
  identity — the last is the only real guard in the repo against a corrupted tile grid.
- `_w1_reliability_proxy.py`'s four correlations reproduce exactly from its committed csv
  (−0.150 / −0.034 / −0.378 / +0.376), matching `DECISIONS.md:2690-2698`.
- `_w1_geometry_audit_all38.py:57` asserts the coreg cache carries `y_sign_fix_applied` before using
  `shift_m`, so it cannot silently re-measure the pre-fix state.
- The `_boulder_size_audit.py` and `_diag_vclaire_sizes.py` numbers, whatever their frame, are
  faithfully transcribed into `DECISIONS.md` and `docs/methods.md` — I found no transcription errors
  anywhere in this area, only frame/population errors.

## Coverage note

**Read in full (42 files, ~2,580 LOC — the whole assigned list):** all 24 `_w1_*.py`,
`_boulder_size_audit.py`, `_crater_distance.py`, `_diag_block_shift_field.py`,
`_diag_tocrs_displacement.py`, `_diag_crs_names.py`, `_diag_lbl_center.py`, `_check_decimated_sp1.py`,
`_probe_pyproj_sp1.py`, `_probe_sp1_regex.py`, `_verify_sp1_fix.py`, and the four `_diag_vclaire_*.py`.
**Read partially (docstring + the load-bearing block only):** `_diag_boulder_localization.py`,
`_diag_boulder_localization_fullres.py`, `_probe_jp2_crs.py`, `_probe_murray_url_variants.py`,
`_w1_rung3_fullres_visual.py` — all are figure/URL probes whose outputs I could not regenerate.

**Reproduced numerically** (conda `geospatial`, read-only, from cached/committed artifacts):
the 9-image boulder-size audit (exactly, then re-derived in the HiRISE frame); the v2 pooled
size/score table both pre- and post-null-drop; the 38-image label-autocorrelation table with base rates
and κ; both rescore-surface parquets (means, per-cell n, balanced panel, paired deltas); the
presence-AUC pre/post comparison; the four reliability-proxy correlations; per-image seam fractions
from `features_ctx_illum`; and the full crater nearest-rim recomputation over the Robbins catalog.

**Could NOT check:** anything requiring imagery or the network — the full-res localization overlays,
the rung-3 visual panels, the rung-4 error maps, the Murray URL probe, and `_w1_geometry_audit_all38.py`'s
phase-correlation residuals (rasterizing 39 images' polygons against CTX windows was out of scope for a
read-only pass). `_w1_sign_error_check.py`'s and `_w1_label_ctx_displacement.py`'s numbers are pre-fix
measurements whose inputs (the pre-fix coreg caches) no longer exist, so they are unverifiable by
construction; both are consistent with the confirmed `src/coregister.py` bug and with the post-fix
surface, which is the strongest available check. I also did not re-audit `_w1_build_dossier.py`, whose
two defects are already filed as `probes-fm-recipe-1` / `-2`.

**Deliberately not re-filed:** R02 (presence AUC on the reported surface), R03 (pixel-scale label
confound), R23 (the score-rank truncation itself), R24 (`mean_std` NaN dropping), R26/R28/R30, and
`labeling.md`'s refutation of the `min_size_m` latitude effect. Findings **-1** and **-3** are new
*instances* in the probe layer of the R23 and R24 mechanisms respectively, and **-1** materially
corrects R23's stated account of how the truncation escaped detection.
