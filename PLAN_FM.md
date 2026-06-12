# PLAN_FM — the post-foundation-model program

**Created 2026-06-12 (Brian-approved direction), after the Fang-ViT probe
passed both gates at both scales by the program's largest margin**
(DECISIONS.md 2026-06-12 ×2; notebook 20). Supersedes
[PLAN_CNN.md](PLAN_CNN.md) §5 as the active plan — the SmallCNN line and
its support machinery are closed (§4 below). Parent program remains
[PLAN_ModelUsability.md](PLAN_ModelUsability.md): Tier-1 binary rich/poor
map, then Tier-2 calibrated abundance.

## 1. Where we stand

Frozen GeM(p=3) embeddings from the Fang et al. 2026 ViT-B/16
(MAE+DINO on 3.9M Murray-mosaic crops, Zenodo 18180801) appended to the
Tier-1 features → LightGBM, standard LOIO over the 38 v2 images,
`fa_gt_1e-2`:

| recipe | S | pooled PR-AUC | prec@5% | med AUC | dAUC med (v) |
|---|---|---|---|---|---|
| t1_gem192 | 64 | **0.7637** | **0.977** | 0.770 | +0.0746 |
| t1_gem64_gem192 | 64 | 0.7549 | 0.884 | **0.7777** | **+0.0918** (win 0.93) |
| t1_gem96 | 32 | 0.7639 | 0.966 | 0.729 | +0.0818 |
| Tier-1 (ref) | 64 | 0.5651 | 0.771 | 0.681 | — |

Candidate recipes (pooled-binding / per-image-binding): t1_gem192 /
t1_gem64_gem192. Standing caveats carried with every claim: transductive
pretraining (disclosure + deployment-matching argument, DECISIONS.md) and
post-hoc assembly (→ §3 confirmation).

**The binding constraint moved.** W0–W2 fought representation (feature-set
floor — now proven: emb_only ≈ fused). The remaining error candidates, in
estimated order: (a) the **head** — LightGBM is a tree reader of a dense
768-dim embedding; the FM literature standard is linear/MLP/kNN probes;
(b) **label quality** — BoulderNet noise + the untested
min_confidence filter; (c) **task formulation** — Tier-2 calibrated
abundance is the product, and regression on this feature set is untested;
(d) **spatial context at the embedding level** (3×3 embedding field vs the
smoothing control).

## 2. Queue (evidence order)

1. **Head bake-off** (probe-tier, current 38 images, cached embeddings —
   `scripts/probes/_w2_fang_heads.py`): logistic probe / small MLP (3
   seeds) / kNN (cosine, k=50) / LightGBM, all on the identical gem192
   matrix in the identical LOIO harness. Fixed hyperparameters (C=1.0,
   one MLP arch) — this is a head-class read, not a tuning exercise.
   If a non-tree head wins, re-run it on t1+gem192 (median-impute the
   T1 columns). Output: one winner-or-tie verdict → freezes the recipe.
   1b. **Target-definition re-read** (Brian, 2026-06-12): re-test
   count-based vs area-based targets on the new features, inside the same
   freeze window. The W0 finding "boulder_count beats fractional_area"
   was established under handcrafted features whose area signal is
   dominated by large-polygon/shadow-merge noise; FM embeddings may shift
   that balance. Protocol honesty: cross-target metrics are NOT directly
   comparable (different positive sets — standing gotcha), so each target
   (`bc_ge_1`, `fa_gt_1e-3`, `fa_gt_1e-2`) is compared against its OWN
   Tier-1 baseline; the claim tested is "the FM advantage transfers
   across target definitions", and the map's target choice remains a
   scientific decision (Brian's) that these numbers inform. Tier-2's
   regression-target version of the same question (log1p count vs
   fractional_area) lives in item 4, where it interacts with the
   single-stage-vs-hurdle retest.
2. **Productize extraction into `src/`** (e.g. `src/fm_embeddings.py`):
   embed arbitrary CTX windows (inference path), wire `fang_*` columns as
   an optional feature source for the packaged-dataset loaders; pytest
   coverage (probe-tier currently has none); README/DATA_DICTIONARY
   entries.
3. **Pre-declared confirmation** (the promotion vehicle): freeze ONE
   recipe (from 1), then write a dedicated DECISIONS.md declaration —
   gates, baseline, test protocol — **before any expansion-image number
   exists**. Shape: new images are pure held-out (train on the 38, predict
   each new image; Tier-1 trained identically as the paired baseline);
   gates to be finalized at declaration time (default: the standard pair,
   pooled ΔPR-AUC ≥ +0.03 / per-image ΔAUC median ≥ +0.05, Wilcoxon
   p < 0.05). Inputs: `cohort_expansion_candidates.csv` (23 verified
   ObsIds incl. 4 ground-truthed lander sites); BoulderNet runs are
   Brian's side.
4. **Tier-2 on the new feature set**: regression head (log1p
   boulder_count / fractional_area) + calibration reporting (W3-style
   compression/high-bin metrics); retest single-stage vs hurdle
   ([[modeling_single_stage_future]] — the hurdle may be unnecessary with
   stronger features).
5. **Model-evidence report** (Brian, 2026-06-12; must land BEFORE the map
   pilot): a standalone persuasion-grade document (docs/, slimmer-doc
   register) whose explicit job is to convince a skeptical reader
   (advisor / committee member) that **the model works and the project is
   worth pursuing to completion**. Required contents:
   - example-prediction galleries — truth-vs-model CTX maps and top-k
     tile strips, covering good images AND the formerly-failing classes
     (the old anti-signal exemplar ESP_046328_2180: slim 0.344 → FM ~0.79;
     the azimuth outlier ESP_076499_1160);
   - a plain-language **metric interpretation guide**: what pooled PR-AUC
     means against the base rate, what prec@5% buys operationally (top
     map tiles are ~98% correct), per-image AUC with its ±0.1–0.2
     fold-ripple error bars, why group-aware LOIO is the honest protocol
     and dev-set numbers are not;
   - the improvement trajectory (slim 5-feature model → Tier-1 → CNN/
     fusion → FM recipe) with what each step ruled out;
   - the honest-caveats section (transductive disclosure, confirmation
     status, label-noise limits) — credibility comes from stating them;
   - "what a map user gets": the operational framing for the Tier-1 map.
   Written after §3 confirmation so the headline numbers carry the
   held-out stamp.
6. **Map pilot**: one Murray tile beyond HiRISE coverage, end-to-end
   (window → embed → predict → map PNG + reliability overlay). The
   usability demo PLAN_ModelUsability exists for; also the first real
   exercise of the §2-productized inference path.
7. **Reliability via embedding-space novelty**: per-tile/per-image
   Mahalanobis or kNN distance to the training distribution in embedding
   space as the label-free warning signal (replaces the AdaBN-disagreement
   idea). Evaluate against the W1 failure taxonomy.
8. **Optional / gated**: MOMO disjoint-corpus probe (bounds the
   transductive caveat; candidate ensemble partner); emb_only @ S=32
   overnight completeness read; ViT fine-tune go/no-go EXPLICITLY decided
   after §3 lands (LoRA/last-block on the 8 GB card; costs determinism,
   risks 38-image overfit; head bake-off may capture the headroom free).

## 3. Discipline

- **Freeze-then-confirm**: recipe shopping (head bake-off, pooling, scale
  mix) happens ONLY on the current 38 images; one recipe is frozen before
  the confirmation declaration; no re-shopping after expansion numbers
  exist. Misses are recorded as declared (house rule since the S=32 read).
- **3-seed rule applies only to stochastic cells** (MLP head, any
  fine-tune). The frozen-embedding + deterministic-head path needs no
  seed protocol — that simplicity is part of its value; don't give it
  away casually.
- Group-aware LOIO always; inner-val rotation unchanged; inference
  features must be CTX-derivable (embeddings are, mosaic-global).
- Every claim ships with the transductive-pretraining disclosure until/
  unless the MOMO bound retires it.

## 4. Retired by the FM result (recorded; do not resurrect silently)

- PLAN_CNN §5.0 conditional-leveler fusion productization — embeddings
  beat the CNN ensemble on both axes at both scales with no fusion;
  formally retire after §3 confirms.
- §5.2 augmentation refinements (FDA/RHM, azimuth-canonical, illumination
  conditioning) — built to protect SmallCNN; the protected images are now
  the biggest FM winners (ESP_076499_1160 +0.458).
- §5.3 AdaBN-disagreement flag — superseded by §2.7 embedding-space
  novelty.
- §5.4 capacity scaling / stride-1 no-pool variant — SmallCNN line closed.
- SmallCNN itself remains in `src/modeling/cnn.py` as the W2 record; not
  deleted, not developed.
