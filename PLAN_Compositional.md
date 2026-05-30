# PLAN — Compositional study of boulders (HiRISE 3 bands)

**Status:** drafted 2026-05-30. **Future work** — not yet started, not yet scheduled.
Follows [CLAUDE.md §10](CLAUDE.md) "instructor's extra goal" (updated 2026-05-30 to use
HiRISE 3 bands instead of CRISM).

## 1. Goal (one paragraph)

After the rock-abundance map (Stages 0–5 + Stage 6 model improvements) is in a defensible
state, test whether the **boulders are locally sourced or transported** by determining
whether the **spectral signatures of boulder-rich areas are distinct from their
surroundings**, using the three HiRISE colour bandpasses ([Delamere et al. 2010, *Icarus*](https://doi.org/10.1016/j.icarus.2009.03.012)):
**BG ≈ 502 nm**, **RED ≈ 686 nm**, **IR ≈ 874 nm**. This study was originally planned with
CRISM; the switch to HiRISE bands removes the 18 m/px scale mismatch with our boulder
detections but introduces the dust-confound (§5).

The science question reduces to two hypotheses:

- **H_local**: boulder-rich tile spectra match boulder-poor tile spectra in the same image
  (boulders are weathered from in-place bedrock with the same composition as the
  surrounding regolith fines).
- **H_transported**: boulder-rich tile spectra differ from boulder-poor tile spectra (the
  boulders were transported from elsewhere — fluvial, impact-ejecta, mass-wasted from a
  cliff — and retain a different composition than the local regolith).

A finding of "no significant difference" supports H_local; a significant difference
supports H_transported, **conditional on ruling out dust as the explanation** (§5).

---

## 2. Inputs

### 2.1 HiRISE colour products (NEW — not currently cached)

Per [HiRISE color documentation](https://www.uahirise.org/pdf/color-products.pdf):

- **3 bandpasses**: BG (502 ± 157 nm), RED (686 ± 267 nm), IR (874 ± 143 nm).
- **Color CCDs**: 14 total (RED0–RED9 + IR10, IR11, BG12, BG13). Only the central 6 CCDs
  (RED4, RED5, IR10, IR11, BG12, BG13) overlap to produce true colour. The IR + BG bands
  are *missing* from the lateral RED swath.
- **Coverage**: colour images have a central swath of ~1.0–1.3 km wide; full HiRISE
  observation is ~6 km wide. So **colour is on the central ~20 % of each HiRISE image only**.
- **Product types**: PDS publishes RGB (natural) and IRB (false-colour) JP2s separately
  from the panchromatic RED used in Stages 1–4b. URLs differ from `JP2_URL` in the
  manifest — need to discover them per ObsId.

**Operational implication**: we cannot do a per-tile colour comparison for ALL tiles in
our v2 dataset. We can only compare colour-covered tiles. The 24-CTX-sources-per-footprint
finding from notebook 13 §3.2 means a HiRISE footprint's "central swath" maps to roughly
20 % of the footprint area, which after coverage masking gives ~10–15 % of v2 tiles eligible
for compositional analysis.

### 2.2 Predicted rock-abundance map

The output of Stages 0–5 + Stage 6, per
[`PROMOTION_QUEUE.md`](PROMOTION_QUEUE.md). Operationally we need either:

- **Continuous predicted `boulder_count`** (per the P2 promotion path) — use as a
  continuous covariate when comparing colour distributions.
- **Binary boulder-rich label** at `fa_gt_1e-2` (per P4 promotion) — partition tiles into
  two populations.

Either works; the **binary partition is the cleaner statistical test** (two-sample
comparison), so this plan assumes the binary deliverable.

### 2.3 Per-image manifest metadata

- `IncidenceAngle`, `EmissionAngle` from cached PDS `.LBL`
  (`cache/pds_labels/{ObsId}.LBL`; verified in notebook 13 §3.1, all 38/38 v2 images have
  these). Used for **photometric correction** of the colour bands.

---

## 3. Stage architecture

This work sits **after** Stage 6 model promotion. Naming: **Stage 7 — Compositional
analysis** (a new pipeline layer for the science deliverable, distinct from the model
improvements of Stage 6).

| Sub-stage | What | Notes |
|---|---|---|
| **7.0** | **Feasibility test on 2–3 images using actual BoulderNet labels, NOT predictions** | **Gate for Stages 7a–7e.** De-risks the methodology end-to-end before building the full pipeline. Details in §3.1 below. |
| **7a** | Discover + cache HiRISE colour JP2s for each manifest ObsId | New Stage 1-like fetch. ~200–500 MB per ObsId for the IRB + RGB pair. |
| **7b** | Per-image radiometric correction + reprojection of colour bands onto the CTX grid | Mirrors Stage 1/2 reprojection logic but on colour bands; needs Lambertian / photometric correction for incidence angle (`I/F = cos(i) * a_0` first-order). |
| **7c** | Per-tile colour features: mean BG, mean RED, mean IR, plus band ratios IR/RED, IR/BG, BG/RED, and a dust index (§5.1) | Joinable on `(scale_idx, ti, tj)` to the existing feature parquet. |
| **7d** | Statistical comparison: boulder-rich vs boulder-poor tile spectra, per image and pooled | The hypothesis test (§4). |
| **7e** | Dust-confound analysis: does the colour difference look more like dust or like composition? | (§5). |

Stages 7a / 7b are the data-engineering chunks; 7c / 7d / 7e are the analysis chunks.
**Stage 7.0 is a separate prerequisite that uses a hand-cut subset of the data** and is the
gating decision for whether to invest in the full pipeline.

### 3.1 Stage 7.0 — Feasibility test (the gate before Stage 7a)

**Why this first**: the full Stage 7a–7e pipeline is ~5–7 days of work and depends on
several things we *haven't verified*:

1. That PDS HiRISE colour products exist for our specific ObsIds.
2. That photometric correction and CTX-grid reprojection produce sensible colour values.
3. That **a real spectral difference between boulder areas and surroundings is detectable
   at all** with a simple two-sample test — and not lost in within-image noise, dust
   variation, or scale mismatch.
4. That the dust-confound discrimination (§5) produces an interpretable result.

If any of these fail, we want to know **before** investing the 5–7 days in the full
pipeline. Stage 7.0 is the small-scale end-to-end de-risking test.

**Critical choice: use actual BoulderNet labels, NOT model predictions.** The full Stage 7
pipeline uses the rock-abundance model's predicted boulder-rich tiles. The feasibility
test should use the **ground-truth BoulderNet polygons** instead:

- This removes prediction noise from the test. If we can't see a colour signal with
  ground-truth boulder locations, we definitely won't see it with noisier predictions.
- The truth set is what the science question is fundamentally about — "do boulders differ
  spectrally from their surroundings?" doesn't depend on our rock-abundance model
  succeeding.
- Falsifies the methodology cleanly: a negative result on truth means "the spectral signal
  isn't detectable at this scale / through this dust mantle / with this correction
  pipeline" — independent of how well the abundance model works.

**Image selection (pick 2–3)**:

Criteria, in priority order:

1. **HiRISE colour coverage confirmed** — verify the IRB/RGB JP2s exist in PDS before
   selecting. (This is itself a sub-check of the §8 open question #1.)
2. **High boulder density** — gives more boulder pixels for statistics. Use the notebook
   13 §2 ranking: `ESP_042964_2160` (34k polygons, AUC 0.91 in our binary classifier),
   `ESP_055978_2270` (9.6k polygons, lift 9.1× — rare-positive but each tile has many
   detections), `ESP_064510_2260` (81k polygons, mid-density).
3. **Confirmed manifest BoulderLabel = "Boulder rich"** — we want the test case to have
   abundant truth signal.
4. **Different latitudes or terrains** if 3+ images are tested — gives a cross-image
   sanity check.

Suggested initial trio (verify colour coverage first):
- `ESP_042964_2160` — high density, model favourite (AUC 0.91)
- `ESP_055978_2270` — rare-positive but dense detections per tile
- `ESP_054000_2255` — the *anti-signal* image (AUC 0.40). If the feasibility test works
  on this image (where the model fails!), that's strong evidence the spectral test
  surfaces something the rock-abundance model can't.

**Methodology for 7.0**:

Two complementary tests at different scales, since the right granularity is an open
question (see §8 q4):

**Test A — per-polygon spectra (the finer-grained test)**:

1. For each boulder polygon (from BoulderNet `.shp` after Stage 1 reprojection), extract
   the HiRISE colour pixels *inside* the polygon (the boulder itself).
2. Define a "surroundings buffer" — e.g. a ring 2–10 m outward from the polygon, excluding
   neighbouring boulder polygons.
3. Compare mean BG/RED/IR + band ratios between "inside boulder" and "surroundings".
4. Per-image two-sample test + effect size.

This is the methodology from [Stanley & Hughes 2020](https://www.sciencedirect.com/science/article/pii/S0019103520302773),
the published precedent.

**Test B — per-tile spectra at S=64 (the coarser-grained test, matches the full pipeline)**:

1. Partition the image's S=64 tiles at `fa_gt_1e-2` using the **truth** `fractional_area`
   from Stage 4 labels (NOT model prediction).
2. Extract mean BG/RED/IR per tile.
3. Compare boulder-rich vs boulder-poor tile populations.
4. Two-sample test + effect size.

This is the methodology from Stage 7d but with truth instead of prediction.

**Why both**: Test A is more sensitive (boulder-pixel directly) but more involved data
work (per-polygon pixel extraction). Test B matches the eventual Stage 7d test. If A and
B agree, we're confident the methodology generalises. If they disagree, the scale matters
and we need to choose deliberately.

**Pass conditions (gate for committing to Stage 7a–7e)**:

- **(a) Pass**: at least one image shows a statistically significant boulder-vs-
  surroundings difference (`p < 0.05`, effect size `|d| > 0.3`) in **at least one** of
  BG/RED/IR or a band ratio, in **at least one of Test A or Test B**, AND the dust-
  confound test (§5) returns an interpretable result (either "dust-attributable" or
  "composition-attributable" with reasonable confidence).
- **(b) Conditional pass**: significant differences found but only attributable to dust.
  This is still scientifically interesting (relative deposit age) and worth pursuing; flag
  the deliverable framing as "relative age, not composition".
- **(c) Fail**: no statistically significant difference on truth labels across any of the
  test images. This kills the Stage 7a–7e investment and triggers a methodological
  rethink (different scale? per-boulder spectra at native HiRISE resolution? different
  bands or band combinations? confounds we haven't enumerated?).

**Implementation effort for 7.0**: ~1–2 days. Fetch colour JP2s for the chosen 2–3 ObsIds,
do quick photometric correction (Lambertian on the central swath only — no need to
reproject to CTX grid for 7.0), run the tests, write a 2–3 page summary in
`notebooks/14_compositional_feasibility.ipynb` (or similar).

**Output**: a go / no-go / conditional-go decision + a writeup that informs §8 open
questions before Stage 7a starts.

---

## 4. Methodology (Stage 7d)

### 4.1 Per-image two-sample test

For each ObsId with colour coverage:

1. **Partition tiles** at the binary boulder-rich threshold (`fa_gt_1e-2` per the P4
   promotion, or `boulder_count > 50` per the P2 path). Drop tiles with no colour
   coverage. Drop tiles with too little RED or BG signal (clouds, shadows).
2. **Compute summary statistics** per tile in each band: mean, median, p5, p95.
3. **Compute band ratios**: IR/RED, IR/BG, BG/RED.
4. **Two-sample test**: Welch's t-test or Mann-Whitney U on each feature, comparing
   boulder-rich vs boulder-poor populations.
5. **Effect size**: Cohen's d on the means; report alongside p-value to distinguish
   "statistically significant but tiny" from "real".

### 4.2 Pooled (cross-image) test

Same machinery as 4.1 but pool tiles across all images. **Adjust for per-image effects**
via a mixed-effects model or by per-image standardisation first:

- Per-image standardise each colour feature using the image's mean and std → tile-level
  feature is "how anomalous is this tile relative to its image's distribution?"
- Then compare standardised boulder-rich vs boulder-poor tiles pooled across images.

The cross-image pooled result is the headline test; per-image tests are the per-image
heterogeneity check.

### 4.3 Continuous-target check

Independent of binary thresholds: compute the Spearman correlation between predicted
`boulder_count` and each colour feature (per image and pooled). A monotone relationship
(positive or negative) supports the hypothesis that the colour signal scales with boulder
density.

---

## 5. The dust confound — the central methodological challenge

Per [HiRISE color documentation](https://www.uahirise.org/pdf/color-products.pdf): **dust
is the reddest material on Mars** (red in RGB, yellow in IRB). Coarser-grained materials
(sand, rocks) are **bluer / cyan-violet in IRB**.

So a finding "boulder-rich tiles are bluer than surroundings" has TWO competing
explanations:

1. **Composition**: the boulders are bluer because they expose primary igneous minerals
   (pyroxene, olivine) and ferrous iron, whereas the surroundings are weathered
   alteration products.
2. **Dust**: the boulders are bluer because **the surroundings have more dust** (i.e.
   boulder-rich areas have less dust accumulation because rocks shed dust off, or because
   recent emplacement hasn't given dust time to accumulate).

Hypothesis 2 still tells us *something useful* — differential dust accumulation indicates
**relative age of the boulder deposit**. Older deposits accumulate dust; younger deposits
have boulders still proud of the dust mantle. But it's NOT a compositional finding.

### 5.1 Dust index

A simple Mars-surface dust proxy: `dust_index = mean_RED / mean_BG`. Higher dust → more
red-shift → higher ratio. Use this as a per-tile feature.

If the boulder-rich vs boulder-poor difference is **fully explained by `dust_index`**,
attribute to dust (= relative age). If the boulder-rich vs boulder-poor difference
**persists after controlling for `dust_index`** (e.g. via partial correlation or residual
analysis), attribute to composition.

### 5.2 Discrimination procedure

Per image and pooled:

1. Compute `dust_index` per tile. Test: do boulder-rich tiles have lower `dust_index`
   than boulder-poor tiles?
2. Compute IR/BG and IR/RED ratios (sensitive to ferric vs ferrous iron per the [HiRISE
   color products documentation](https://www.uahirise.org/pdf/color-products.pdf) — higher
   ratios indicate altered ferric materials).
3. **Partial correlation**: control for `dust_index` and re-test boulder-rich vs
   boulder-poor on the IR/BG and IR/RED ratios. If the difference survives, this is a
   compositional signal independent of dust.

### 5.3 Other confounds to flag

- **Photometric (illumination geometry)**: incidence/emission angles vary across images
  and within an image (HiRISE swath is ~6 km — large enough for incidence-angle gradients
  on a sloped surface). Mitigate via simple Lambertian correction at Stage 7b:
  `I/F_corrected = I/F_observed / cos(incidence_angle)`. More sophisticated corrections
  (Hapke, Minnaert) deferred unless first-pass is dominated by photometric variation.
- **Atmospheric scattering**: scattered light is redder than direct sunlight, so the
  apparent `dust_index` of any surface depends on atmospheric opacity at acquisition
  time. The HiRISE PDS LBL includes the optical depth (`OPTICAL_DEPTH`) which we can use
  as a per-image covariate or filter (drop high-tau acquisitions).
- **Shadow contamination**: boulder shadows are darker (lower signal in all bands) and
  also shift colour. Mask shadow pixels (use the same `shadow_fraction` machinery from
  Stage 4b) before computing per-tile colour means.
- **Seasonal frost**: drop polar / high-latitude images with frost (CenterLat |lat| > 50°
  AND the LBL season suggests frost season).

---

## 6. Acceptance criteria

**Stage 7.0 gate** (see §3.1 for full details): pass / conditional pass / fail decision on
2–3 images using truth labels. **Failure here halts Stage 7a–7e**; conditional pass
reframes the deliverable as "relative age" rather than composition.

**Full Stage 7 deliverable** (assuming 7.0 passes): **a per-image report** that answers,
for each ObsId with colour coverage:

1. Is the boulder-rich vs boulder-poor colour difference statistically significant?
   (Welch's t, Mann-Whitney U, p-value, effect size.)
2. If significant, is it dust-attributable or composition-attributable? (Partial
   correlation result.)
3. A confidence categorisation per image: {locally sourced | transported | dust-age
   difference | inconclusive}.

**Pooled headline finding**: across the colour-covered fraction of the v2 cohort,
do the images cluster on one of these four categories? Does the category correlate with
manifest BoulderLabel or with geographic context (latitude, terrain type)?

---

## 7. Implementation cost estimate

| Sub-stage | Effort | Dependencies |
|---|---|---|
| **7.0 — Feasibility test (2–3 images on truth labels)** | **1–2 days** | **PDS colour JP2 fetch for 2–3 ObsIds only; Lambertian correction; per-polygon AND per-tile tests on the central swath. Gates 7a–7e.** |
| 7a — colour JP2 fetch (full v2 cohort) | 1 day | PDS URL discovery per ObsId; ~10 GB cache |
| 7b — reprojection + photometric correction | 1–2 days | Stage 1/2 logic; Lambertian I/F correction |
| 7c — per-tile colour features | 0.5 day | Stage 4b machinery |
| 7d — statistical comparison | 1 day | scipy, statsmodels |
| 7e — dust-confound analysis | 1 day | Partial correlation + per-image discrimination |
| Writeup + figures | 1–2 days | Notebook 14 + a docs/compositional.md paper-Methods style writeup |

**Total**: ~6–9 days end to end (1–2 for 7.0 gate + 5–7 for the full pipeline if 7.0
passes). Cheap relative to the full Stage 6 work, and **the 7.0 gate caps the downside
risk at ~2 days** if the methodology turns out not to work.

---

## 8. Open questions for runtime (to surface via AskUserQuestion when execution starts)

1. **Colour coverage per ObsId**: how many of the 38 v2 images actually have IRB +
   RGB products in PDS? (Suspect ~60–80 % based on standard HiRISE acquisition modes,
   but needs PDS query at Stage 7a.) **VERIFY AT RUNTIME.**
2. **Photometric correction sophistication**: first-pass Lambertian only, or invest in
   Hapke / Minnaert? Probably Lambertian is fine for this first study; flag for revision
   if the per-image effects dominate the per-tile signal.
3. **Atmospheric opacity filter**: include all images, or filter to tau < 0.5? Probably
   include all + use as covariate in the statistical model; flag if filtering is needed.
4. **Sample matching for "boulder-rich vs surroundings"**: at the tile level (binary
   partition above some abundance threshold) OR at the boulder-polygon level (per-boulder
   spectra vs surrounding regolith pixels)? The tile-level test is methodologically
   simpler and is what this plan assumes; the per-polygon-level test is more sensitive
   but requires aligning HiRISE polygon coordinates with HiRISE colour pixels (not CTX
   grid) — a more involved data pipeline. Decide at runtime based on signal strength of
   the tile-level test.
5. **Output target audience**: a chapter in the thesis? A paper for *Icarus* or *JGR
   Planets*? Affects how the writeup is structured (length, depth of methodology
   section).

---

## 9. Related literature

- [Delamere et al. 2010, *Icarus*](https://doi.org/10.1016/j.icarus.2009.03.012) — HiRISE
  colour imaging methodology; canonical citation.
- [McEwen et al. 2007, *JGR Planets*](https://doi.org/10.1029/2005JE002605) — HiRISE
  instrument overview.
- [Stanley & Hughes 2020, *Icarus*](https://doi.org/10.1016/j.icarus.2020.113890) — origin
  and composition of three heterolithic boulder- and cobble-bearing deposits overlying the
  Murray and Stimson formations, Gale Crater. Direct methodological precedent for our
  approach (boulder-by-boulder composition analysis).
- Boulder halo / clast attribute work — [LPSC 2020 abstract on clast attributes of
  Martian boulder halos](https://www.hou.usra.edu/meetings/lpsc2020/pdf/1458.pdf).
- For the dust confound: [Atwood-Stone & McEwen 2013, *Icarus*](https://doi.org/10.1016/j.icarus.2013.09.026)
  on dust and HiRISE colour interpretation (more general; cited as a dust-mantle
  reference).

---

## 10. Scope notes

- **Not in scope for this plan**: the boulder-detection-side colour analysis (analysing
  the colour of individual detected boulder polygons rather than tile means). That would
  be a follow-up if the tile-level test surfaces a real signal.
- **Not in scope**: extending the analysis to non-v2 images. The v2 cohort is the
  go-forward dataset.
- **Not in scope**: integrating colour features into the rock-abundance model itself.
  That would be Stage 6h or similar and would need to confirm that colour is
  inference-time-compatible (it is, for CTX-only regions where HiRISE colour is sometimes
  available — but the cases where HiRISE color is available are exactly the cases where
  we have HiRISE detections, so colour as a model feature is *training-only* useful,
  not inference-useful in CTX-only regions).

This last point is important and parallel to the HiRISE-LBL "out of scope" finding in
[PROMOTION_QUEUE.md](PROMOTION_QUEUE.md): **HiRISE colour features are out of scope as
model inputs for the same reason**. They're an *analysis* layer on top of the model
output, not a model input.
