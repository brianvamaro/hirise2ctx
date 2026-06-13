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
   MLP *head* is not yet productized (still `_w2_fang_heads.py`) — fold into
   item 4 / map pilot.
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
