# PLAN — Making the rock-abundance model usable

**Status:** planned 2026-06-10 (Brian + Claude session). Supersedes the
report-writing thread as the active program. The slimmer writeups
([docs/classification_slimmer.md](docs/classification_slimmer.md),
[docs/compositional_slimmer.md](docs/compositional_slimmer.md)) are submitted;
this plan is **not** about improving the writeup — it is about turning the
model into something that can actually be run over CTX and trusted.

**Brian's scoping decisions (2026-06-10):**

1. **"Usable" = two product tiers.** Tier 1 (first): a **binary
   boulder-rich / boulder-poor tile map** with a per-region reliability flag.
   Tier 2 (after): a **calibrated rock-abundance map** comparable to THEMIS.
2. **Cohort is fixed at the 38 v2 (vClaire) images for now.** More BoulderNet
   detections are a *maybe later* unlock — plan around 38, document where
   n=38 binds.
3. **CUDA torch approved** for the `geospatial` env (GPU present; current
   torch is CPU-only).
4. **First session = W0 (bank the wins).**

**Out of scope for this program:** all compositional work (dust index, Tier 3
source-unit comparison, anything in PLAN_Compositional.md §11), report
formatting, data expansion (W5 documents it as the known unlock only).

---

## 1. Why the model is weak — root-cause synthesis

Evidence-ranked synthesis of [docs/modeling_results.md](docs/modeling_results.md)
§9–14 + [PROMOTION_QUEUE.md](PROMOTION_QUEUE.md):

| # | Cause | Evidence | Status |
|---|-------|----------|--------|
| 1 | **CTX-source heterogeneity** (mosaic stitching) | `mean_n_sources` ↔ Spearman ρ=−0.405 p=0.012; `std_ctx_incidence` ↔ PR-AUC ρ=−0.370 p=0.022; `dominant_source_frac` ρ=+0.394 (n=38). [Dickson 2024](https://doi.org/10.1029/2024EA003555) seam-artefact prediction confirmed in our model. | **Validated mechanism**, no working fix yet (per-tile features net-flat, gate ceiling 0.606 at n=38) |
| 2 | **n=38 images, geographically clustered** | Every per-image-level fix (Stage 6c gate, Stage 6b selectivity) failed in the way small-n predicts; gate models with >3 params underperform L2 logreg. | Binding; fixed for now by decision |
| 3 | **Known-best recipe never promoted** | P2 `boulder_count` +22 % PR-AUC, P1 `balanced` +0.017 ρ (dev); Stage 6a 5×5@S=32 strict PASS (dev); P5 calibration fix unrun. All still dev-only. | **Cheapest fix available** — W0 |
| 4 | **5 m/px texture floor — partly real, partly features** | Within-image ≈ LOIO AUC 0.55–0.62 across framings; BUT v2 dense labels lifted ρ +0.10, so the "floor" moved once before when an input improved. Hand-crafted features are the untested half. | CNN (W2) is the honest test |
| 5 | **Hurdle architecture + compression** | Magnitude head shrinks the high tail (high-bin ratio 0.83); presence head can't physically detect 1 boulder/tile at 5 m/px; two-stage lift over single-stage only +0.017 ρ dev. Model is a ranker, not a calibrated regressor. | W0 single-stage test; W3 loss redesign for Tier 2 |

Reading: causes 3 and 5 are cheap to act on now; cause 1 needs tile-level
mechanism work (W1) and illumination-invariance (W2); cause 4 is what the CNN
tests; cause 2 is accepted and documented.

**Cause 0 — data-integrity artifacts (Brian directive, 2026-06-10):** before
any of the above is blamed — and especially before concluding "CTX isn't
enough" — the mundane failure modes must be excluded per image: misprojected
labels, bad co-registration, grid/join bugs, BoulderNet false positives.
None of these has been systematically audited per image. W1 is restructured
as a differential diagnosis that works from mundane to fundamental; the
sensor-floor conclusion is reached only by exclusion. Note that **anti-signal
(AUC < 0.5) is itself evidence for cause 0**: a genuinely uninformative
sensor yields AUC ≈ 0.5; *systematically inverted* predictions need a
mechanism, and spatially shifted labels over a coherent boulder field (or a
garbled tile join) is exactly the kind of mechanism that produces inversion.

### 1.1 Prior art — Serrano et al. (2010), the direct ancestor of this project

[Serrano, McGuire, Mayer, Huertas & Arvidson, "Predicting HiRISE-equivalent
Rock Density on Mars Using CTX Image Features," AIAA Infotech@Aerospace](https://www-robotics.jpl.nasa.gov/media/documents/Infotech_Paper.pdf)
([NTRS record](https://ntrs.nasa.gov/citations/20100039411)). Bayesian
Network inferring HiRISE shadow-detector rock density (rocks > 1.5 m per
100 m hectare) from CTX GLCM + intensity features, Phoenix landing area,
10 image pairs (8 train / 2 test). Reviewed 2026-06-10; takeaways:

1. **Geomorphic unit as a mediating variable.** Their BN is
   `density → geomorphic unit → features`: the texture↔density mapping is
   *conditional on terrain class*, and their observed failure modes were
   terrain misclassifications (Ld↔Lb under-prediction, Ce↔Ld
   over-prediction). This is our per-image heterogeneity problem stated
   30× smaller. Actionable for us: a **terrain-unit covariate is
   inference-compatible** — global geologic maps (e.g. USGS Tanaka et al.
   2014 global geologic map) exist everywhere CTX exists. Add terrain unit
   per tile to the W1 dossier; if the error atlas shows terrain-conditioned
   failure, a terrain-categorical feature (or per-unit models) is a
   legitimate Stage-6-style candidate.
2. **They used concurrent native HiRISE–CTX pairs, not a mosaic.** HiRISE
   and CTX acquire simultaneously, so their CTX had identical illumination,
   season, and atmosphere to the HiRISE ground truth — by construction zero
   source-heterogeneity (our validated cause 1). A **native-CTX variant of
   our pipeline** (train on each image's concurrent CTX acquisition; infer
   on native CTX catalog images rather than the Murray mosaic) would remove
   cause 1 from training entirely, at the cost of per-image photometric
   normalization at inference (illumination known from CTX metadata — we
   already parse this via SeamMap/CUMINDEX). **Candidate pivot, decided
   after W1**: if the error atlas confirms seam/source-driven failure
   dominates, this is the structural fix.
3. **Regional "fill the gaps" deployment framing.** They explicitly
   disclaim cross-region transfer ("not intended to be trained on one area
   and tested in an entirely different area") — the product is: train on
   scattered HiRISE within a region, predict the CTX between them. Our
   LOIO across geographically scattered images attempts something strictly
   harder. Tier 1's honest first deployment claim should likely be
   regional gap-fill (W4 demo should be designed this way), with global
   transfer as the stretch goal.
4. **Hazard-class product format.** Their output is a rock-density class
   map (0–3 / 4–8 / 9–19 / >19 rocks per hectare) plus a continuous
   expected-safety map (probability-weighted class values). Classes are
   defined by the *application* (lander safety), not by statistics — and
   counts-per-area aligns exactly with our `boulder_count` target. Good
   presentation precedent for Tier 1 (class map) and Tier 2 (expected-value
   map with calibrated bin probabilities).
5. **Feature-family validation, and its age.** Their features are
   literally ours (GLCM contrast/energy/homogeneity/correlation + window
   mean/std, ~hectare scale) and visibly correlated with rock density —
   independent confirmation the feature family carries signal. But these
   are 2010-era hand-crafted features; that they remain our feature set is
   itself an argument for W2's CNN test.

---

## 2. Product definitions (what "done" looks like)

### Tier 1 — binary boulder-rich map (first)

- **Unit:** S=64 tile (320 m), binary label `fa_gt_1e-2` (fractional_area > 1 %)
  — the operationally meaningful threshold (P4, already primary).
- **Output:** per-tile probability + binary flag, plus a **per-region
  reliability layer** derived from SeamMap stats (`n_sources`,
  `dominant_source_fraction` — inference-compatible everywhere CTX exists).
- **Honest acceptance target** (calibrated against current numbers: per-image
  median AUC 0.61, pooled PR-AUC 0.54 @ P1+P2): per-image **median AUC ≥ 0.70
  on reliability-passing images**, pooled PR-AUC ≥ 0.60, ECE ≤ 0.05, with the
  reliability layer flagging the anti-signal failure mode *before* prediction.
- The classifier itself should be a genuine binary model
  (`LightGBMClassificationBalanced` and/or CNN head), not a count regressor
  read at a threshold — decided in W0.

### Tier 2 — calibrated abundance map (after Tier 1)

- **Unit:** same grid; target = `boulder_count` (model space) with documented
  post-hoc conversion to area-fraction
  (`count × mean_boulder_area / tile_area × population_scaling`) for THEMIS
  comparability (PROMOTION_QUEUE.md P2 section; Brian's lean = rank
  correlation suffices, calibrated values are the stretch goal).
- **Honest acceptance target:** high-bin ratio (mean_pred/mean_true at
  fa > 1e-2) in [0.8, 1.2] (currently 0.83 dev, worse on LOIO); LOIO Spearman
  ≥ 0.25 pooled (currently 0.14–0.15); positive rank correlation vs THEMIS on
  overlap regions.

---

## 3. Workstreams

### W0 — Bank the wins ✅ DONE 2026-06-10 (see DECISIONS.md 2026-06-10 entry)

**Outcome**: P2 promoted (PR-AUC +0.162, p<1e-4); P1 and P5 null at LOIO
(dev wins didn't replicate — both consistent with per-image distribution
shift as the binding constraint); two-stage hurdle retained on per-image
meaningful-AUC evidence (+0.022, p=0.008); Stage 6a S=32 strict FAIL with
partial carry (Δρ +0.072 PASS / ΔPR-AUC +0.017 FAIL), S=64 stays the
operating scale. **Promoted baseline recipe: `lightgbm_two_stage_balanced`
× `boulder_count` @ S=64** (ρ +0.1431 / PR-AUC 0.5431 / prec@5% 0.5679;
per-image meaningful-AUC median 0.594 / max 0.979 / 29% < 0.50). Tier 1
reference classifier unchanged. Original plan below for reference.

Establish the true best baseline. Everything later is measured against this.

1. **P1+P2 full-v2 LOIO promotion.** The P1+P2 numbers already exist as the
   Stage 6b sweep baseline (Spearman 0.1431, PR-AUC 0.5431 @ S=64,
   `models/_sweep_stage6b/20260531T020308Z/`); what's missing is the formal
   delta vs the `fractional_area` baseline + DECISIONS.md promotion entry.
   May need one baseline re-run for a clean apples-to-apples.
2. **Single-stage vs two-stage** (memory `modeling-single-stage-future`):
   `LightGBMLog1pHuber`-style single-stage vs `LightGBMTwoStageBalanced`,
   same LOIO. If Δ within noise → drop the hurdle, simplify everything
   downstream (CNN heads, calibration, docs).
3. **Genuine binary classifier + P5 calibration fix.**
   Add `LightGBMClassificationBalanced` (drop `scale_pos_weight`), sweep on
   `fa_gt_1e-2` LOIO; report per-image AUC/PR-AUC + ECE (expect ECE
   0.16–0.27 → ~0.05). This becomes the Tier 1 reference model.
4. **Stage 6a 5×5 @ S=32 full-v2 confirmation** (the only strict dev PASS in
   the Stage 6 family). If it carries, it joins the recipe; also test the
   stencil at S=64 on top of the binary classifier.

**Deliverable:** a single "recipe table" in DECISIONS.md naming the promoted
baseline (variant, target, features, scale) + per-image metric distribution.
All later W-items compare against this.

### W1 — Error atlas as differential diagnosis (~3–4 days)

> **Status 2026-06-10: W1 COMPLETE (one session).** Rung 1 found the
> headline bug — the coreg y-shift was applied with inverted sign to every
> v2 label (fixed; labels regenerated; baseline re-banked: ρ +0.1878 /
> PR-AUC 0.5616 / median per-image AUC 0.603 / anti-signal 11→8; all W0
> verdicts re-verified). Rungs 2–5: join integrity CLEAN; BoulderNet content
> clean (failures carry small 2–4 m boulders in uniform speckle); seam-tile
> masking does nothing (source effect is regional, ρ≈0.38 replicated);
> anti-signal splits into texture_decorrelated (3, sensor floor) +
> distribution_shift (2, real signal missed — fixable) + validity_limited
> (3). Deliverables: notebook 18, `dataset_v2/w1_dossier.parquet`,
> DECISIONS.md two entries. **Decisions: reliability flag = graded
> region-level confidence (seam masking rejected); native-CTX pivot NO-GO
> for now; next bets = per-image feature standardization, then W2 CNN with
> photometric augmentation, then terrain covariate.**

Turn "anti-signal" from a label into an actionable mechanism. The per-image
diagnosis exists (notebook 13); the **per-tile** level was never done.
**Structure (Brian, 2026-06-10): work the ladder from mundane to
fundamental, and treat every rung as live until excluded.** "CTX isn't
enough" is the conclusion of exclusion at the bottom — not a hypothesis to
reach for when an upper rung looks hard. Remember the cause-0 argument:
AUC < 0.5 favours artifact explanations, because absent signal gives ≈ 0.5
while inversion needs a mechanism.

**Rung 1 — label geometry (misprojection / co-registration).** Per-image
audit, prioritizing the anti-signal images: re-render reprojected polygons
over the decimated HiRISE *and* the CTX window (the notebook-02/03 QA
visuals, never done for v2's worst images); pull each image's block-median
co-registration shift + correlation-peak quality and correlate against
per-image AUC. Scale context: the ~200 m mosaic registration error is
1.25 tiles at S=32 and 0.6 tiles at S=64 — residual misalignment alone can
destroy or invert per-tile signal. **Decisive cheap test: re-score a bad
image with its labels shifted by ±1 tile in each direction; if AUC recovers
at some offset, it was geometry all along.**

**Rung 2 — pipeline / join integrity.** (ti, tj) ↔ CTX-pixel-block mapping
spot-checks; label-parquet ↔ feature-parquet join audit per image (row
counts, key uniqueness, scale_idx consistency); NaN / nodata fractions;
duplicated or dropped tiles. A garbled join on one image is indistinguishable
from "model fails on that image" until checked.

**Rung 3 — label content (BoulderNet quality).** Visual sampling of
detections on full-res HiRISE for the anti-signal images specifically: are
"boulders" there actually ripple crests, dune brinks, or crater-wall
texture? Per-image detection score/size distributions vs the cohort. (v2
ingest dropped null geometries, but detection *quality* per image was never
audited against performance.)

**Rung 4 — feature / CTX content.** Per-image feature-distribution drift;
clipped or saturated CTX windows; **tile-level error maps with SeamMap seam
polygons overlaid** and per-tile `n_sources` / `dominant_source_fraction` —
the direct tile-level test of the validated image-level CTX-source
correlation. Plus terrain unit per tile (global geologic map join, per
Serrano takeaway 1) to test terrain-conditioned failure.

**Rung 5 — genuine signal limits.** Only what survives rungs 1–4 gets
attributed to heterogeneity (cause 1, fix = native CTX / invariance) or the
texture floor (cause 4, fix = CNN or nothing).

**Synthesis outputs:**
- **Per-image dossier table** (38 rows): banked-baseline metrics, rung-1
  geometry audit results, CTX-source stats, terrain class, base rate, label
  density, dominant failure mode (notebook-13 taxonomy recomputed on the W0
  recipe), and an *attributed cause* per problem image.
- **Decision memo**: (a) Tier 1 **reliability flag** definition (if
  seam-local masking explains anti-signal, the gate problem changes from
  "predict bad images" at n=38 to "mask bad tiles" at n≈38k — not
  small-n-bound); (b) **go/no-go on the native-CTX pivot** (prior-art
  takeaway 2) based on how much failure is seam/source-attributed.

**Deliverable:** notebook 18 + figures + dossier + reliability-flag
definition with measured precision/recall against known-bad images.

### W2 — CNN on context patches (~1–2 weeks, parallelizable after W0)

> **Status 2026-06-11: expanded into [PLAN_CNN.md](PLAN_CNN.md)** (executable
> spec: setup tasks, 4-cell augmentation grid, pre-declared gates, budget).
> Motivating evidence updated post-W1: the v1 "CNN dead-end" judgment
> predates the coreg+shadow fixes and the bet-1 zscore result showing the
> distribution-shift class is treatable. Section below kept for history.

The honest test of cause 4 (feature ceiling) and the natural attack on
cause 1 (learned illumination invariance).

1. **Setup** (½ day): install CUDA torch into `geospatial`
   (check driver CUDA version via `nvidia-smi` first; respect the
   `src/modeling/__init__.py` OpenMP import-order gotcha). Enable
   `context_patch_px` in Stage 4 config (32 and 64 px; CTX windows are
   cached, so this re-runs Stage 4 only — minutes, no downloads).
2. **Phase 1 — binary CNN** (matches Tier 1): small CNN (e.g. ResNet-18-class,
   single-band input) on 64 px patches predicting `fa_gt_1e-2`, honest LOIO
   (group-aware splits, never tile-random). Compare per-image AUC / pooled
   PR-AUC against the W0 binary baseline.
   - **Photometric augmentation is the key experiment**, not an afterthought:
     brightness/contrast/gamma jitter simulates exactly the CTX-source
     heterogeneity that breaks the tabular model. Run with/without to
     measure whether learned invariance closes the anti-signal gap.
3. **Phase 2 — squeeze 38 images** (only if Phase 1 shows signal):
   - **Self-supervised pretraining** on unlabeled CTX patches (no BoulderNet
     needed — sample the mosaic freely; SimCLR/MAE-style), fine-tune on the
     38 labeled images. This is the label-free way around the n=38 bind.
   - Regression head for Tier 2; multi-scale patch input; per-image
     standardization layers.
4. **Decision gate:** if the CNN with augmentation does NOT beat the W0
   tabular baseline (pooled PR-AUC +0.03 or per-image median AUC +0.05),
   that is a *real result*: the 5 m/px floor is sensor-bound, and usability
   work shifts entirely to reliability-flagging + scaffold honesty (W1/W4).

### W3 — Tier 2 calibration path (after Tier 1 is solid)

> **Operationalized 2026-06-14 as [PLAN_Calibration.md](PLAN_Calibration.md).**
> The compression Brian flagged on the model-evidence figures is now measured and
> Stage 0 (diagnose + preview) is DONE (`src/calibration.py`, notebook 23). Key
> finding that refines W3: **Tier-1 is already well-calibrated** (ECE 0.06); the
> compression is essentially **Tier-2**, and it is *two-sided*
> (regression-to-the-mean). **Quantile-matching** (not the per-image isotonic
> guessed below — isotonic fits the compressed mean and does NOT help) recovers the
> true value distribution while preserving ranking. See that plan for the staged
> approach; W3's THEMIS validation (item 3) consumes its calibrated output. Original
> sketch kept:

1. Attack compression directly: **Stage 6f Zero-Inflated Tweedie**
   (docketed, untested; [Chen et al. 2024](https://arxiv.org/abs/2406.16206)),
   quantile heads, or single-stage + post-hoc per-image isotonic — pick based
   on what W0's single-stage test showed.
2. Calibration protocol: LOIO-held-out isotonic/Platt on probabilities
   (Tier 1) and on abundance (Tier 2); report ECE + high-bin ratio.
3. THEMIS rank-correlation validation on overlap regions (this is
   *validation*, not the compositional thread — in scope).

### W4 — Inference scaffold (buildable in parallel, any time after W0)

What makes any of this "usable" in practice:

1. **`scripts/infer_ctx_region.py`**: lon/lat box → CTX window fetch
   (machinery exists in `src/ctx_retrieve.py`) → Stage-4b features → banked
   model → GeoTIFF / parquet map (probability + binary + abundance) **+
   reliability layer** (SeamMap-derived, per W1's definition).
2. Constraint check at the door: every feature consumed must be CTX-derivable
   (the PROMOTION_QUEUE.md inference-time scope rule).
3. **Demo region**: design as **regional gap-fill** (prior-art takeaway 3) —
   a region containing several training HiRISE footprints, predicting the
   CTX between them (prediction-continuity sanity check at the footprint
   edges), plus one THEMIS-overlap region. This is the deployment scenario
   the evidence actually supports; global transfer is the stretch claim.
4. QA notebook rendering the demo maps.

**Deliverable:** one command that produces a map a third party could look at,
with the reliability layer making the model's limits visible rather than
hidden.

### W5 — Data expansion (deferred, documented as the unlock)

Not now, by decision. When it reopens: target selection should prioritize
(a) geographic/terrain diversity beyond the 40–46°N cluster, (b) regions
where the CTX mosaic is **single-source / high dominant_source_fraction**
(cleanly separates cause 1 from cause 4 in training data), (c) genuine
boulder-poor images (vClaire is 37/40 boulder-rich — the binary classifier
has almost no true-negative *images*).

---

## 4. Sequencing

```
W0 (1 day) ──► W1 (3–4 days) ──► decision point ──► W2 Phase 1 (CNN binary, ~1 wk)
   │                │                 │                  │
   │                │                 ├─► native-CTX     ├─► W2 Phase 2 (SSL / regression)
   │                │                 │   pivot (if      │
   │                │                 │   seam-driven)   │
   └────► W4 scaffold (parallel) ◄────┴──────────────────┘
                                                         └─► W3 Tier 2 calibration
```

- W0 first (Brian decision). W1 next — its reliability-flag output is needed
  by both W4 and the Tier 1 product definition, and its **attributed-cause
  outcome decides what W2 even is**: artifact fixes (rungs 1–3), the
  native-CTX pivot (seam-driven), terrain conditioning (terrain-driven), or
  the CNN as planned (feature-driven).
- W2 and W4 can interleave; W4 doesn't depend on the CNN (it ships with the
  best banked model and upgrades later).
- W3 starts once Tier 1 is accepted.

## 5. Standing constraints (carry into every workstream)

- **Inference features must be CTX-derivable** (no HiRISE metadata at
  inference) — PROMOTION_QUEUE.md scope rule.
- **Group-aware (leave-image-out) evaluation always**; never tile-random
  splits, including for the CNN.
- **AskUserQuestion before expensive sweeps and git commits** (standing
  protocol).
- **Promotion discipline**: dev → full-v2 LOIO → DECISIONS.md entry; strict
  acceptance thresholds declared before each sweep, soft results documented
  honestly.
- `conda run --no-capture-output -n geospatial python -u …` for all
  long-running launches; `import src.modeling` before numpy/pandas in any
  torch-adjacent script (OpenMP gotcha).
