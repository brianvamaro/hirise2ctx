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

1. **Freeze-window evidence runs — CLOSED 2026-06-12; RECIPE FROZEN**
   (DECISIONS.md "Freeze window CLOSED" entry). All probe-tier, on the
   current 38 images, cached embeddings. Runner
   `scripts/probes/_fm_freeze_window.py`; cells under
   `models/fang_probe/fw_*`. **FROZEN RECIPE (Brian sign-off):
   `mlp_ens3` (3-seed MLP 768-256-64-1, dropout 0.2) on the S=32 96-px
   3×3-context GeM(p=3) 768-dim embedding, emb-only (no handcrafted
   features), target `fa_gt_1e-2`. Banked
   `fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2`: pooled 0.7832 / prec@5% 0.948 /
   med per-image AUC 0.7865 / dAUC(v) +0.120 / win 0.96, both gates PASS.**
   Sub-items, evidence order:
   - **1a. Head bake-off — DONE 2026-06-12**
     (`scripts/probes/_w2_fang_heads.py`): on the identical gem192-only
     matrix / identical LOIO harness, every non-tree head beat LightGBM:
     MLP-ens3 0.7852 pooled / med AUC 0.8035 / dAUC +0.1374; kNN(cos,50)
     0.7709; logreg 0.7385; LightGBM 0.7146. Class question settled:
     trees are the wrong reader of dense embeddings. MLP pooled
     calibration is seed-wobbly (0.745–0.787; per-image skill stable) —
     the 3-seed ensemble is the promotable form, exactly the SmallCNN
     lesson.
   - **1b. Target-definition re-read — DONE 2026-06-12**: the FM advantage
     transfers to EVERY non-degenerate target (each vs its OWN Tier-1):
     fa_gt_1e-2 0.8040 / fa_gt_1e-3 0.9183 / bc_ge_50 0.8260 /
     bc_ge_100 0.7312 all pass both gates. **`bc_ge_1` was the wrong count
     target** (Brian): saturated at S=64 (0.93 positive = presence), gates
     fail — replaced with data-grounded thresholds bc_ge_50/bc_ge_100
     (`_fm_count_dist.py`, registered in `binary_target.py`). At matched
     base rate (bc_ge_100 vs fa_gt_1e-2) area edges pooled (0.804 vs 0.731)
     but per-image AUC ties (0.828 vs 0.821) → advantage is
     target-definition-robust; reverses the W0 "count beats area" finding
     (that held only under handcrafted features). Map target choice remains
     Brian's scientific call; **frozen = fa_gt_1e-2** (continuity). (Tier-2's
     regression version lives in item 4 with the hurdle retest.)
   - **1c. Head-vs-head paired statistics — DONE 2026-06-12**: mlp_ens3
     beats every other head with clean paired significance (vs lgbm
     +0.0595 p~0; vs logreg +0.0292 p=0.0006; vs knn50 +0.0499 p=0.0032);
     the other three are tied among themselves. Winner: **MLP 3-seed
     ensemble** (`models/fang_probe/head_pairs.json`).
   - **1d. Pool × head interaction — DONE 2026-06-12**: under mlp_ens3
     (t1ctx, S=64, fa_gt_1e-2) GeM(p=3) 0.8040 > mean 0.8015 > cls 0.7900
     pooled; GeM also best win-rate. **GeM confirmed** under the MLP.
   - **1e. Winner micro-sweep + ensembling + calibration — DONE
     2026-06-12; all three add-ons REJECTED**: (a) arch sweep (3 widths ×
     2 dropouts, gem192/S=64) spread 0.799–0.811, incumbent 256×64/d0.2
     mid-pack, none separable at n=38 → **default kept**; (b) calibration
     layer net-harmful — per-image rank collapses pooled to 0.5056, blend
     0.7352, both leave med_auc unchanged (the 3-seed mean already IS the
     wobble fix); (c) cross-head ensemble (mlp+logreg rank-mean) 0.7995 <
     mlp_ens3 alone (logreg dilutes; kNN can't join t1ctx — no
     standardization in KNNHead). Plain `mlp_ens3` is simplest and best.
     **(Brian, 2026-06-12: exhaustive architecture search/tuning becomes
     worth it once the cohort expands** — revisit as a declared dev phase
     after the §3 confirmation images are absorbed into training.)
   - **1f. Handcrafted-feature elimination check — RUN 2026-06-12,
     decision pending** (Brian: *ideally eliminate*): mlp_ens3 on
     t1+gem192 = pooled 0.8040 / med AUC 0.8284 / win 0.96 vs 0.7852 /
     0.8035 / 0.85 on gem192-only — the 52 handcrafted columns still add
     ~+0.02 (and narrow the MLP seed spread), so elimination is NOT free.
     Dropping them buys embed-and-predict inference (no GLCM/gradient/
     shadow at map time; prec@5% slightly better emb-only). Simplicity
     vs ~2 points = Brian's freeze-time call; both variants stay
     candidates until then.
   - **1g. Operating-scale decision — DONE 2026-06-12; S=32 chosen**
     (Brian sign-off): mlp_ens3/GeM/fa_gt_1e-2 at S=32 holds skill — both
     matrices pass both gates (emb-only 0.7832 / med 0.7865 / dAUC +0.120 /
     win 0.96; t1ctx 0.7764). Cost vs S=64 is modest (~−0.03 pooled /
     −0.04 med) but the map is 4× finer (160 m) and **prec@5% is higher**
     (0.948). Crucially, **feature elimination is FREE at S=32** (emb-only
     ties t1ctx; the 1f +2-pt gap dissolves), so the frozen recipe is
     emb-only. S=16 probe not pursued.
   - **1h. (optional) 320-px (5×5) context probe — SKIPPED**: the S=32
     finer-map decision made the larger-context question moot for the
     operating recipe; revisit only if cohort expansion reopens
     scale/context tuning.
2. **Productize extraction into `src/` — DONE 2026-06-12 (commit 032fa75)**:
   `src/fm_embeddings.py` = ViT-B/16 encoder + GeM(p=3) + 3×3-context slicing +
   `FangEmbedder.embed_window` (arbitrary-CTX-window inference path for the map
   pilot). `src/modeling/loaders.py` cached-store join
   (`load_fang_store`/`fang_columns_for_keys`/`augment_fold_with_fang`,
   torch-free; `replace=True` = emb-only). 15 pytest (full fast suite 292
   green); README + DATA_DICTIONARY. **Bit-exact parity** with the cached store
   the frozen 0.7832 was measured on (`scripts/probes/_fm_parity_check.py`). The
   MLP *head* is now productized too — DONE 2026-06-14 in
   `src/modeling/mlp_head.py` (item 6.A).
3. **Pre-declared confirmation** (the promotion vehicle): write a dedicated
   DECISIONS.md declaration — gates, baseline, test protocol — **before any
   expansion-image number exists**. Recipe is frozen (from 1). Inputs:
   `cohort_expansion_candidates.csv` (23 verified ObsIds incl. 4 ground-truthed
   lander sites); BoulderNet runs are Brian's side.
   - **Protocol = confirm-then-absorb (TENTATIVE, Brian 2026-06-12; RE-CHECK
     when the expanded cohort actually lands)**: (a) train the frozen recipe on
     the current 38, predict each expansion image as pure held-out, record the
     gate verdict; (b) THEN fold all expansion images into training — the
     deployed model + the next (cohort-expanded) dev phase train on all ~61.
     The confirmation certifies the *recipe* generalizes (selection-bias-free);
     the deployed all-data model inherits that as a CONSERVATIVE estimate (more
     data ≥ less). **Open decision deferred to execution time**: keep a SUBSET
     as a permanent holdout (e.g. the 4 ground-truthed lander sites, which carry
     external truth) vs full absorb — decide on cohort size/composition BEFORE
     computing any predictions (so it stays pre-registered). Discussed
     alternatives: permanent holdout (costly in small data), rolling LOIO over
     the growing cohort (the living estimate; keep it as a complement). The
     deployment-time generalization safeguard is the §2.7 embedding-novelty
     reliability overlay (per-tile "trust here?"), not a held-out number for the
     all-data model (which cannot exist by construction).
   - **Baseline (proposed)**: `mlp_ens3` on the 52 handcrafted features — same
     head/scale/target, only features differ → isolates the FM contribution as a
     clean causal claim (NOT the old LightGBM Tier-1, which collapses at S=32).
   - **Gates (proposed, standard pair, at the rich/poor threshold `fa_gt_1e-2`
     — NOT presence, [[feedback_no_presence_auc]])**: pooled ΔPR-AUC ≥ +0.03 /
     per-image ΔAUC median ≥ +0.05 / Wilcoxon p < 0.05; win-rate reported.
     Finalize at declaration time. Misses recorded as declared (house rule).
4. **Tier-2 on the new feature set — DONE 2026-06-13** (DECISIONS.md
   "Tier-2 regression"): calibrated-abundance regression on the frozen emb-only
   S=32 features, 3 heads × 2 targets × {emb, t1}.
   `scripts/probes/_fm_tier2_regression.py` (+ `_fm_tier2_ceiling.py`).
   **Verdicts**: (a) `mlp_reg` (NEW 3-seed MLP regressor) wins regression too
   (Spearman 0.431 fa / 0.386 count); (b) FM ~2× the handcrafted baseline
   (mlp fa 0.431 vs t1 0.223); (c) **single-stage beats the hurdle** — `mlp_reg`
   ≫ two-stage (0.431 vs 0.329); hurdle DROPPED ([[modeling_single_stage_future]]
   confirmed); (d) regression matches the classifier on rich/poor
   (meaningful_auc 0.78 ≈ classifier 0.7865) — calibrated magnitude ~free.
   **Ceiling tested** (Brian): zero-inflation is NOT the limiter (16% exact-zeros
   at S=32; removing them moves ρ by ~0.01) — the wall is intrinsic
   magnitude-ranking difficulty; NDCG@5% 0.50 vs t1 0.35. **Compression
   quantified**: under-predicts the high tail ~30% (top-bin pred/true 0.71; FM
   compresses less than handcrafted 0.55) → a calibration layer is future work.
   Metrics bug found+fixed in review (count `meaningful_threshold` was presence;
   threaded through `run_loio`, [[feedback_no_presence_auc]]). The single-stage
   `mlp_reg` is the Tier-2 candidate; freeze/productize + the calibration layer
   come with the map pilot (2.6).
5. **Model-evidence report — DRAFTED 2026-06-14b** (`docs/model_evidence.md`
   prose complete; held-out headline row + the §3 schematic figure pending):
   a standalone persuasion-grade document (docs/, slimmer-doc
   register) whose explicit job is to convince a skeptical reader
   (advisor / committee member) that **the model works and the project is
   worth pursuing to completion**. Uses the §2.6 map-pilot figure as its
   predict-beyond-coverage example and honestly reports the §2.7 reliability
   deferral (novelty = OOD flag, not accuracy predictor). Required contents:
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
6. **Deployable head + map pilot — DONE 2026-06-14** (DECISIONS.md
   "Deployable head + map pilot"): `src/modeling/mlp_head.py`
   (`DeployableHead`, banked `models/deployable/86c51a5dca220f63/`, 38 imgs,
   in-sample sanity AUC 0.966, save/load round-trips) + `src/mapping.py` +
   `scripts/{train_deployable_head,map_pilot}.py`. First off-HiRISE map rendered
   (E4_N44 beyond ESP_055253_2245's footprint, no download — tile zips cached;
   8281 tiles, 21 s; `reports/figures/map_pilot_E4_N44_*.png` + GeoTIFF). A
   parent-tile-anchor georef double-count was found by a post-render check and
   fixed (`tile_origin_transform`, regression-tested). +18 tests, fast suite 312.
   The usability demo PLAN_ModelUsability exists for; first real exercise of the
   §2-productized inference path. Original spec (kept for the scale-out TODO):
   - **A. Deployable head (PREREQUISITE, not yet built)**: today the frozen
     `mlp_ens3` exists only INSIDE the LOIO harness (re-trained per fold). A map
     needs ONE model trained on ALL 38. Productize the MLP head into `src/`
     (currently probe-tier `_w2_fang_heads`/`_fm_freeze_window`): `fit` on the
     full emb-only S=32 matrix, persist the 3 seed state-dicts + feature scaler
     + recipe hash, `load`/`predict(emb)→prob`. (Apply the MLP perf fix here:
     batch 4096 + tensors-on-device-once.)
   - **B. Per-tile inference (one Murray tile)**: window+buffer (reuse
     `ctx_retrieve`) → `tile_grid_for_window` (owned tiles) →
     `FangEmbedder.embed_window` → `DeployableHead.predict` → §2.7 reliability
     score → rows keyed by GLOBAL `(ti,tj)` + geo-bounds.
   - **C. Combine across Murray tiles is TRIVIAL by design**: Murray tiles
     partition the globe (4°×4°, abut, NO overlap), and the tile grid is
     anchored to the GLOBAL mosaic pixel origin (CLAUDE.md Stage 4) → `(ti,tj) =
     (mosaic_row//32, mosaic_col//32)` is globally unique regardless of which
     window computed it. Combine = PLACEMENT into the global raster, no blending.
     Each map tile produced exactly once. The ONLY cross-tile care: the 3×3
     context at a tile's edge needs neighbor pixels → read each Murray tile with
     a ≥96 px BUFFER and OWN-BY-CENTER (tile T predicts global tiles whose center
     falls in T's box; buffer guarantees full context; discard buffer-region
     predictions). VERIFY-AT-RUNTIME: Murray-tile pixel dims (a 32-px tile may
     straddle a boundary — center-ownership + buffer handle it) and the buffer
     size, on the first real tile.
   - **D. Render**: place by `(ti,tj)` into the mosaic-CRS raster → GeoTIFF +
     PNG rich/poor (or abundance) map at 160 m + reliability overlay.
   - **E. Pilot scope**: ONE Murray tile beyond HiRISE coverage end-to-end —
     proves the inference path AND the combine pattern before any scale-out.
7. **Reliability via embedding-space novelty — VALIDATED 2026-06-14; overlay
   DEFERRED to post-expansion** (DECISIONS.md "2026-06-14b"). Built
   `src/reliability.py` (Mahalanobis + kNN, +10 tests) and the LOIO validation
   (`scripts/probes/_fm_reliability_validation.py`). **Pre-registered bar NOT
   cleared at n=38**: per-image novelty vs the frozen recipe's OWN per-image AUC
   = Mahalanobis rho −0.108 (p=0.52) / kNN-cos50 rho −0.141 (p=0.40), bottom-5
   flag prec 0.00. Right direction, insignificant. **Cause = the FM decoupled
   novelty from skill**: it already absorbed the covariate-shift class, so the
   most-novel image (ESP_076499_1160, rank 1/38) is a FM *winner* (AUC 0.868)
   while the weakest (ESP_045983_2270, AUC 0.564) is texturally ordinary
   (intrinsic difficulty, invisible to a novelty detector). Novelty IS a valid
   OOD/extrapolation flag but NOT an accuracy predictor → **deferred** rather
   than ship a weakly-justified trust layer; re-run this same validation when the
   §3 expansion images land (n=38 underpowered). Map stays trust-layer-less.
   Original design (kept for the rerun):
   a label-free per-tile
   "is this CTX texture like what I trained on?" score — the deployment-time
   answer to *where* on the map to trust the prediction (the confirmation says
   it generalizes on average; this says where). Replaces the retired
   AdaBN-disagreement idea (no BatchNorm in the frozen-embedding path).
   - **Methods (compare two)**: (a) Mahalanobis distance to the training
     embedding cloud — μ, shrinkage/PCA-whitened Σ on training tiles (tame the
     768² covariance); (b) kNN distance — mean cosine/euclidean distance to the
     k nearest *training* tiles (non-parametric, handles a multimodal training
     distribution; 147k×768 trivial for sklearn/faiss).
   - **Granularity**: per-tile for the map overlay; aggregated (median per-tile
     or image-mean-embedding distance) per-image for taxonomy validation.
   - **Validation subtlety (important)**: the W1 failure taxonomy was defined on
     *Tier-1* failures, and the FM RESCUED most of those (distribution_shift
     +0.23–0.31). So validate novelty against where the **frozen recipe ITSELF**
     underperforms — does per-image novelty rank-correlate with the frozen
     recipe's per-image AUC (the freeze-window numbers)? Does it flag the
     low-AUC images? If novelty predicts the FM's *own* weak spots, it's a valid
     trust signal.
   - **LOIO-honest by construction**: for each held-out image, fit the novelty
     model on the other 37, score the held-out one — exactly the deployment case.
   - **Deliverable**: per-tile `reliability` column banked beside predictions +
     per-image aggregate + a novelty-vs-frozen-AUC validation figure + the map
     overlay. Prototype is CPU-only — defer launch when GPU chains are CPU-bound
     (overhead-bound MLP cells consume CPU), otherwise free to run.
8. **Optional / gated**: MOMO disjoint-corpus probe (bounds the
   transductive caveat; candidate ensemble partner); emb_only @ S=32
   overnight completeness read; ViT fine-tune go/no-go EXPLICITLY decided
   after §3 lands (LoRA/last-block on the 8 GB card; costs determinism,
   risks 38-image overfit; head bake-off may capture the headroom free);
   per-image-standardized embeddings (Brian, 2026-06-12: explicitly
   deferred / low priority — the FM already rescued the shift class;
   revisit only if the confirmation read surfaces residual shift
   failures).

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
