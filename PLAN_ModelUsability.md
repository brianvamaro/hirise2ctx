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

### W0 — Bank the wins (~1 day) ← START HERE

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

### W1 — Error atlas (~2–3 days)

Turn "anti-signal" from a label into an actionable mechanism. The per-image
diagnosis exists (notebook 13); the **per-tile** level was never done.

1. **Tile-level error maps**: for each of the 38 images, map LOIO
   residual / rank-error per tile over the CTX window, with **SeamMap seam
   polygons overlaid** and per-tile `n_sources` / `dominant_source_fraction`.
   Direct test: do errors concentrate near seams / in multi-source tiles
   *within* images (the tile-level version of the validated image-level
   correlation)?
2. **Per-image dossier table** (38 rows): banked-baseline metrics, CTX-source
   stats, terrain class (from the Tier-1 spreadsheet), base rate, label
   density, dominant failure mode (anti-signal / rare-positive-miss /
   compression — the notebook 13 taxonomy, recomputed on the W0 recipe).
3. **Decision memo**: define the Tier 1 **reliability flag** from what the
   atlas shows (e.g. exclude tiles with `n_sources > k`, or per-region
   `dominant_source_fraction` cutoff). If seam-local masking explains the
   anti-signal images, the gate problem changes from "predict bad images"
   (n=38-bound) to "mask bad tiles" (n≈38k — not small-n-bound).

**Deliverable:** notebook 18 + figures + reliability-flag definition with
measured precision/recall of the flag against known-bad images.

### W2 — CNN on context patches (~1–2 weeks, parallelizable after W0)

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
3. **Demo region**: a CTX area adjacent to a known good-performing image
   (prediction continuity sanity check), plus one THEMIS-overlap region.
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
W0 (1 day) ──► W1 (2–3 days) ──► W2 Phase 1 (CNN binary, ~1 wk)
   │                                   │
   │                                   ├─► W2 Phase 2 (SSL / regression)
   │                                   │
   └────► W4 scaffold (parallel) ◄─────┘
                                       └─► W3 Tier 2 calibration
```

- W0 first (Brian decision). W1 next — its reliability-flag output is needed
  by both W4 and the Tier 1 product definition.
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
