# Review area: probes-fm-recipe

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-02
- **Verification:** self-refuted (single-agent pass; not independently verified)

> **Headline first, because it is the most useful result here:** the three frozen-recipe numbers
> **reproduce exactly** from the committed artifacts. Recomputing
> `models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet` against
> `models/lightgbm_classification/2d046f48c722f0a5/scale_S32_tfa_gt_1e-2/predictions.parquet` gives
> `pooled_pr_auc = 0.783213`, `prec@5% = 0.947578`, `median per-image AUC = 0.786509`, `n = 161,005`,
> `pos_rate = 0.359827`, one join, no dropped rows, one tie at the top-5 % cutoff. Every cell in the
> five DECISIONS tables (1a/1b/1d/1e/1g and the 2026-06-11/-12 probe tables) matches its banked
> `verdict.json` to the digit. **The defects below are in the *comparisons and captions* around those
> numbers, not in the numbers.**

## Findings

### probes-fm-recipe-1 — "the FM differentially rescues the W1 failure classes" is a conditioning artifact: those images are the FM's *worst*, in absolute terms
- **Severity:** medium
- **Liveness:** live-shipped (the claim is in the reader-facing `docs/model_evidence.md`)
- **Confidence:** high
- **Where:** [scripts/probes/_w2_fang_probe.py:191-192](../../scripts/probes/_w2_fang_probe.py#L191-L192)
  (producer) · [scripts/probes/_w1_build_dossier.py:78-89](../../scripts/probes/_w1_build_dossier.py#L78-L89)
  (the class definition) · consumed at `DECISIONS.md:3202-3206`, `notebooks/_build_20.py:152-158`,
  `docs/model_evidence.md:170-173`

`verdict()` groups the **per-image ΔAUC (FM − Tier-1)** by `w1_dossier.attributed_cause`, and
`attribute()` assigns that cause from **a per-image AUC threshold on a baseline model**
(`meaningful_auc >= 0.5` splits `ok` from the two failure classes; `texture_rho_med < 0.15` then
splits them). Conditioning a *difference* on the subtrahend being small is the textbook
regression-to-the-mean setup: the class means of ΔAUC are ordered almost exactly inversely to the
class means of the baseline AUC, so **any** replacement model with roughly uniform skill produces the
reported pattern. Measured: Spearman(ΔAUC, baseline Tier-1 per-image AUC) = **−0.678** (p = 4e−06) at
S=64 and **−0.663** (p = 6e−06) at the frozen S=32 operating point. The absolute numbers tell the
opposite story: at S=32 the FM's mean per-image AUC on the 6 `distribution_shift` +
`texture_decorrelated` images is **0.686**, versus **0.815** on the 18 `ok` images, and **5 of those
6 are still in the bottom third of the 38-image AUC ranking** (only `ESP_076499_1160` is genuinely
rescued, to 0.868). Group sizes are also n = 3 / 3 / 2 / 1, reported as bare means with no dispersion.

- **Failure scenario:** a reader of `docs/model_evidence.md` Figure 4 ("largest exactly on the classes
  Part 1 failed") or of DECISIONS finding 2 ("The W1 failure classes are differentially rescued … The
  per-image shift problem that killed every W0–W2 dev win at LOIO is exactly where the FM helps")
  concludes the FM specifically solves covariate shift, and plans the expansion-cohort confirmation
  or a deployment risk statement on that basis. In fact the distribution-shift and
  texture-decorrelated images remain the cohort's least reliable tiles under the shipped recipe — the
  regional map is *least* trustworthy exactly where the record says the model was fixed.
- **Evidence:**
  ```python
  # scripts/probes/_w2_fang_probe.py:191-192
  cause = dossier["attributed_cause"].reindex(d.index).fillna("unclassified")
  row["dauc_by_cause"] = d.groupby(cause).mean().round(4).to_dict()

  # scripts/probes/_w1_build_dossier.py:78-89  -- the class IS a threshold on the baseline AUC
  def attribute(r):
      if r.meaningful_auc >= 0.5:
          ...
          return "ok"
      if not r.validity_ok:
          return "validity_limited"
      if r.texture_rho_med < 0.15:
          return "texture_decorrelated"
      return "distribution_shift"
  ```
  Frozen S=32 cell, grouped by `attributed_cause` (`t1` = Tier-1 S=32 per-image AUC, `fm` = frozen
  recipe per-image AUC):

  | cause | n | mean t1 AUC | mean FM AUC | mean ΔAUC |
  |---|---|---|---|---|
  | validity_limited | 2 | 0.757 | 0.780 | +0.023 |
  | ok_validity_limited | 9 | 0.658 | 0.751 | +0.093 |
  | ok | 18 | 0.711 | **0.815** | +0.105 |
  | texture_decorrelated | 3 | 0.490 | **0.645** | +0.155 |
  | ok_shadowfeat_fixed | 2 | 0.576 | 0.753 | +0.176 |
  | distribution_shift | 3 | 0.542 | **0.726** | +0.184 |
  | ok_geometry_fixed | 1 | 0.594 | 0.818 | +0.225 |

  The ΔAUC column is monotone in the (inverse of the) `t1` column; the `fm` column is not.
- **Self-refutation attempted:** (a) *Is it literally circular?* No — the class-defining statistic is
  `meaningful_auc` from a **different** run (`models/_sweep_w0/20260611T013810Z`,
  `lightgbm_two_stage_balanced` on `boulder_count`), whereas the ΔAUC baseline is the Tier-1
  *classifier* on `fa_gt_1e-2`. I measured their correlation: Spearman = **+0.632** (S=64) / **+0.567**
  (S=32). So the selector is strongly correlated with, not identical to, the subtrahend — the
  regression-to-the-mean is attenuated, not eliminated, which is why I file this as medium and not
  high. (b) *Are the improvements real?* Yes, in absolute terms `distribution_shift` goes 0.542 → 0.726
  at S=32 — the *existence* of an improvement survives; only the **comparative** claim ("most", "exactly
  where") fails. (c) *Is it already filed?* `stats-fallacies` covers circularity in the F/striping
  programme (R11, R36, R40) and R43; nothing in §4–§4e or §5 touches the dossier-class ΔAUC read.
  (d) *Does it move a gate?* No — both frozen gates are on the pooled ΔPR-AUC and the median ΔAUC,
  neither of which uses the classes.
- **Fix:** report the **absolute** per-image AUC by class beside the delta (one extra column in
  `dauc_by_cause`), state the n per class, and reword the three consuming claims to "the failure
  classes improve the most *relative to a baseline that was near chance on them*, but remain the
  cohort's weakest images under the FM".

### probes-fm-recipe-2 — the frozen recipe's per-image gate is scored on 27 of 38 images by a validity rule computed at the wrong scale on the wrong target, and it excludes both of the recipe's two largest losses
- **Severity:** medium
- **Liveness:** live-shipped (`win 0.96` / `dAUC(v) +0.120` are the frozen recipe's banked gate numbers)
- **Confidence:** high
- **Where:** [scripts/probes/_w1_build_dossier.py:34-40](../../scripts/probes/_w1_build_dossier.py#L34-L40)
  (producer) · [scripts/probes/_fm_freeze_window.py:134-135](../../scripts/probes/_fm_freeze_window.py#L134-L135),
  [:146-157](../../scripts/probes/_fm_freeze_window.py#L146-L157) and
  [scripts/probes/_w2_fang_probe.py:166-167](../../scripts/probes/_w2_fang_probe.py#L166-L167),
  [:186-188](../../scripts/probes/_w2_fang_probe.py#L186-L188) (consumers) · quoted at
  `DECISIONS.md:3430,3457-3458`

`validity_ok` is `(n_neg >= 50) & (n_meaningful_positive >= 50)` computed on the
**`lightgbm_two_stage_balanced` / `boulder_count > 50` / scale_idx 3 (S=64)** rows of
`models/_sweep_w0/20260611T013810Z/summary.parquet`
([`_sweep_w0.py:55-63`](../../scripts/probes/_sweep_w0.py#L55-L63) sets the count threshold to 50).
Every probe in this area then applies that 27-image mask verbatim as the population for the per-image
gate — including the frozen recipe, which is `fa_gt_1e-2` at **S=32**, where each image has ~4× the
tiles. Applying the same ≥50/≥50 rule to the *actual* gated target and scale (from
`models/_sweep_binary/20260612T062412Z/summary.parquet`), **36 of 38** images qualify, not 27. The
mask both over-excludes (10 images valid at S=32 are dropped) and under-excludes
(`ESP_048688_2085` has 35 positives at S=32 and is kept).

- **Failure scenario:** the banked verdict reports `dauc_win_v = 0.9630` (26/27). The frozen recipe
  has **three** negative per-image ΔAUCs across the cohort, and the two largest —
  `ESP_068483_2280` (**−0.256**, the single worst image in the cohort) and `ESP_055978_2270`
  (**−0.078**) — are both outside `vok`. On the S=32-honest population the same statistic is
  median +0.1182, **win 0.917** (33/36), p = 3.8e−07; on all 38 it is +0.1182 / **0.921** / 1.4e−07.
  A reader of `DECISIONS.md:3458` ("dAUC(v) +0.120 / win 0.96") takes away "the FM beats Tier-1 on 96 %
  of images" when the cohort-wide figure is 92 %, and never learns that the worst case is a 0.26-AUC
  regression on a near-shadowless image.
- **Evidence:**
  ```python
  # scripts/probes/_w1_build_dossier.py:34-40  -- S=64, boulder_count>50, GBM regression run
  summ = pd.read_parquet(SWEEP)   # default models/_sweep_w0/20260611T013810Z/summary.parquet
  rec = summ[(summ.variant == "lightgbm_two_stage_balanced") & (summ.target_col == "boulder_count")]
  d = rec.set_index("held_out_obs_id")[["n_tiles", "n_meaningful_positive", ...]].copy()
  d["n_neg"] = d.n_tiles - d.n_meaningful_positive
  d["validity_ok"] = (d.n_neg >= 50) & (d.n_meaningful_positive >= 50)

  # scripts/probes/_fm_freeze_window.py:135,147,153-156  -- reused unchanged for S=32 / fa_gt_1e-2
  vok = set(dossier[dossier.validity_ok].index)
  d_v = d[[o in vok for o in d.index]]
  "dauc_median_v": float(d_v.median()),
  "dauc_win_v":    float((d_v > 0).mean()),
  "gate_per_image": bool(d_v.median() >= 0.05 and pval < 0.05),
  ```
  ```
  dossier validity_ok            : 27 / 38
  same rule, S=32 fa_gt_1e-2     : 36 / 38  (fails: ESP_047976_2020, ESP_048688_2085)
  excluded-but-valid-at-S=32 ΔAUC: 068483 -0.2562  055978 -0.0779  054622 +0.0928  063429 +0.0930
                                   069763 +0.0945  045550 +0.1165  059686 +0.1247  055017 +0.1598
                                   054134 +0.1999  071093 +0.2138
  ```
- **Self-refutation attempted:** (a) *Does the gate flip?* **No** — median ΔAUC stays +0.118 ≥ 0.05 and
  p stays ≪ 0.05 on all three populations, so the frozen recipe's PASS is safe. That is what keeps this
  at medium rather than high. (b) *Is it deliberate and declared?* The label "(v)" and the phrase
  "dossier validity-passing images" appear in DECISIONS, but nowhere does any doc say the mask is a
  S=64 `boulder_count` criterion or that it is stale at the operating scale; `_w2_fang_probe.py:16`
  merely says "dossier validity-passing images". (c) *Is the ≥50 rule itself wrong?* No — it is a
  sensible per-image AUC reliability floor; the defect is applying an S=64 count-target instance of it
  to an S=32 area-target gate. (d) *Already filed?* R24 is the sibling pattern (a mean over 5 of 20
  folds reported as 20) but a different probe and statistic; nothing in the register covers
  `w1_dossier.validity_ok`.
- **Fix:** compute `validity_ok` per (scale, target) from the run's own `n_positive`/`n_negative`
  (both columns are already in every `summary.parquet`), or at minimum emit `n_v` and the excluded
  obs_ids into `verdict.json` and record the all-images median/win beside the `(v)` pair.

### probes-fm-recipe-3 — the per-image ΔAUC figure in the reader-facing evidence writeup is the S=64 LightGBM probe cell, captioned as the frozen FM recipe
- **Severity:** medium
- **Liveness:** live-shipped (published writeup)
- **Confidence:** high
- **Where:** `docs/model_evidence.md:168-173` (consumer) vs
  [notebooks/_build_20.py:170-186](../../notebooks/_build_20.py#L170-L186) (producer)

`docs/model_evidence.md` embeds `reports/figures/20_fang_perimage_dauc.png` with the caption
"*Per-image AUC change of **the FM recipe** over the handcrafted baseline, one bar per held-out
image*". The figure is built in notebook 20 from `models/fang_probe/t1_gem192/*/verdict.json` — the
**S=64, Tier-1-features-plus-GeM192, LightGBM** probe cell — and its own axis label says so
("per-image dAUC (t1_gem192 − Tier-1, S=64)"). That is a different tile scale (320 m vs the recipe's
160 m), a different feature matrix (52 handcrafted + 768 emb vs emb-only), and a different head
(LightGBM vs `mlp_ens3`) from the recipe every other number in that document describes.

- **Failure scenario:** the document's §0 table, §3 bullets and §8 all describe the S=32 emb-only
  `mlp_ens3` recipe; a reader takes Figure 4 as that recipe's per-image evidence. It is not. The two
  differ materially: the figure's median ΔAUC is **+0.0729 over 37 images** (the 38th,
  `ESP_047976_2020`, has no positives at S=64 and is silently absent, so the figure has 37 bars for a
  cohort the text calls 38), while the frozen recipe's is **+0.1182 over 38**. The per-image identities
  differ too — `ESP_068483_2280` is +0.026 in the figure but **−0.256** under the shipped recipe, i.e.
  the figure hides the recipe's single worst image. The caption's specific claim
  "the azimuth outlier ESP_076499_1160 (the cohort's single biggest win)" happens to hold for both
  cells, but by coincidence.
- **Evidence:**
  ```python
  # notebooks/_build_20.py:170-186 -- the producer, explicitly the S=64 t1_gem192 cell
  vj = sorted(PROBE.glob("t1_gem192/*/verdict.json"))[0]
  d = pd.Series(v["t1_gem192"]["per_image_dauc"], dtype=float).sort_values()
  ax.set_xlabel("per-image dAUC (t1_gem192 - Tier-1, S=64)")
  fig.savefig(FIG / "20_fang_perimage_dauc.png", ...)
  ```
  ```
  docs/model_evidence.md:168  ![Per-image deltas over the handcrafted baseline](../reports/figures/20_fang_perimage_dauc.png)
  docs/model_evidence.md:170  *Figure 4. Per-image AUC change of the FM recipe over the handcrafted baseline...
  ```
  Banked medians: `t1_gem192` (the figure) +0.0729, n=37 · `fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2`
  (the recipe) +0.1182, n=38.
- **Self-refutation attempted:** (a) *Is "the FM recipe" loose shorthand for "the FM programme"?* The
  same document's §0 table and §4 define "the FM recipe" precisely as
  `fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2` (§9 cites that exact directory), so the caption is a
  mis-attribution, not shorthand. (b) *Is the difference immaterial?* No — it is 0.045 of median ΔAUC
  and it reverses the sign on the recipe's worst image. (c) *Already filed?* `docs-consistency`
  audited `model_evidence.md`'s **numbers** (its §5 explicitly clears lines 22/26-28/29/107/294) but
  did not trace its **figures** to their producers; `notebooks-1` is the analogous defect in notebook
  10 (most-recently-modified model dirs) — different mechanism, different file.
- **Fix:** regenerate the figure from
  `models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/verdict.json` (the data is banked; no rerun
  needed), or retitle the caption "the S=64 `t1_gem192` probe cell (not the frozen recipe)" and note
  the 37-vs-38 image count.

### probes-fm-recipe-4 — `DECISIONS.md:3204` quotes a per-image ΔAUC that appears in none of the 90 banked verdicts, and the next entry silently contradicts it 5×
- **Severity:** low
- **Liveness:** dead-closed (a caveat sentence in a superseded entry)
- **Confidence:** high
- **Where:** `DECISIONS.md:3202-3205` vs `DECISIONS.md:3294-3298` and
  `models/fang_probe/t1_gem192/ed6b211643a2148e/verdict.json`

The 2026-06-11 entry closes with "*the illumination-geometry caveat remains untested (no
azimuth-conditioned read yet; ESP_076499_1160 dAUC **+0.087** under t1_gem192, so the azimuth outlier
is not failing)*". The banked `t1_gem192` verdict records **+0.4576** for that image — the cohort's
largest value in that cell — and the next day's entry, reading the same artifact via
`_w2_fang_azimuth.py`, quotes "+0.458". I searched all 90 `verdict.json` files under
`models/fang_probe/`: `ESP_076499_1160`'s ΔAUC ranges 0.148–0.632 across every cell and **no cell
yields 0.087**.

- **Failure scenario:** a future session grepping DECISIONS for the azimuth-outlier evidence finds two
  contradictory numbers a day apart with no retraction, and cannot tell which cell either came from.
  (The error is conservative — the true value supports the same conclusion more strongly — which is
  why this is low.)
- **Evidence:**
  ```
  DECISIONS.md:3204  ... ESP_076499_1160 dAUC +0.087 under t1_gem192, so the azimuth outlier is not failing).
  DECISIONS.md:3297  - **ESP_076499_1160 (azimuth outlier, 228.6 deg) is the cohort's biggest
  DECISIONS.md:3298    winner: dAUC +0.458**
  models/fang_probe/t1_gem192/ed6b211643a2148e/verdict.json  per_image_dauc.ESP_076499_1160 = 0.4576
  models/fang_probe/t1_gem192/ed6b211643a2148e/azimuth_read.json  outliers.ESP_076499_1160.dauc = 0.4576
  ```
- **Self-refutation attempted:** looked for 0.087 as (i) another image's value in the same cell
  (`ESP_068483_2280` is +0.0255, nothing is near 0.087), (ii) the same image in another cell (min
  across all banked cells is 0.148), (iii) the cell median (0.0729) or `dauc_median_v` (0.0746) —
  neither is 0.087 either. It is not a mislabelled but real statistic.
- **Fix:** correct `DECISIONS.md:3204` to +0.458 with an inline note that it was mis-transcribed, so
  the two entries stop disagreeing.

## Refuted by my own check

- **The `0.7832 / 0.948 / 0.7865` headline itself.** Reproduced exactly from
  `predictions.parquet` (see banner). The join is 161,005 → 161,005 with `validate="one_to_one"` and a
  hard `assert len(df) == len(preds)`; no fold is dropped; all 38 images have both classes at S=32 so
  no per-image AUC is NaN; exactly one tie sits at the top-5 % cutoff, so `np.argsort`'s
  non-stable tie-break moves at most 1 of 8,050 tiles.
- **"Pooling mixes 38 per-fold models' output scales."** True mechanically, but the project measured
  the consequence directly: the 1e per-image-rank ablation collapses pooled PR-AUC 0.7832 → 0.5056
  while leaving `med_auc` at 0.8284, and `DECISIONS.md:3409-3415` reads it correctly (cross-image level
  is information pooled PR-AUC rewards, and rank is monotone within image so per-image AUC is
  invariant). No defect.
- **`k = max(1, int(0.05 * y.size))` (floor) vs `src/modeling/evaluate.py`'s `round`.** Two
  implementations of precision@5 % (the R25 pattern), but at n = 161,005 they differ by at most one
  element and the recomputation matches the banked value to 6 d.p. Not filed.
- **NaN-imputed context embeddings inflating or depressing the headline.** `_w2_fang_embed.py:217`
  marks tiles whose 3×3 box overruns the CTX window invalid, and `EmbeddingBank.__init__:96` NaN-fills
  them, which `_StandardizedHead._impute` would median-fill. I read the `valid` array of all 152
  `.npz` files: **0 invalid rows** at every input size (P32 0/161,005, P96 0/161,005, P64 0/37,315,
  P192 0/37,315). DECISIONS' "100 % context coverage" is exact; no tile was scored on an imputed
  embedding.
- **Train/deploy embedder parity on the plain (non-A1) path** (the R07 sibling question).
  `_w2_fang_embed.embed_batches:176-181` and `src/fm_embeddings.FangEmbedder.preprocess/embed_patches`
  agree line for line: `uint8 → float/255 → (x−0.5)/0.5 → bicubic 224 (align_corners=False) →
  autocast fp16 on CUDA → tokens.float() → GeM(clamp 1e-6, p=3)`, same `BATCH = 96`. The box geometry
  also agrees (`r0 = ti*tile_px − row0 − tile_px` in both). No inversion on this path.
- **The CNN arm was unfairly compared (`_w2_cnn_verdict.py`).** The confound I expected — the FM gets a
  192-px 3×3 context while the CNN only sees the 64-px own tile — does not survive: the *own-tile*
  FM cell `t1_gem64` scores pooled 0.7531 against Tier-1's 0.5651, while the CNN's best cell A scores
  0.5095. At matched input size the FM adds +0.19 and the CNN −0.06, so "PLAN_CNN superseded" holds.
  (Both sweeps are the same target `fa_gt_1e-2` at the same scale_idx 3.)
- **`_w2_fang_head_pairs.py` runs 6 uncorrected pairwise Wilcoxon tests.** The two that carry the
  "mlp_ens3 beats every other head" verdict are p = 0.0006 and p = 0.0032, both inside Bonferroni
  0.05/6 = 0.0083, so the conclusion survives correction. Also checked whether the verdict conflates
  "MLP" with "3-seed ensembling": it does not — each single MLP seed already beats logreg and kNN on
  ΔAUC (+0.12…+0.14 vs +0.10 / +0.09).
- **The cross-scale S=64-vs-S=32 comparison in freeze-window 1g** (a prevalence-dependence risk, cf.
  R26). Base rates are 0.3543 (S=64) and 0.3598 (S=32) — matched to 0.006 — and both prec@5 % values
  are far below their `min(1, base_rate/0.05)` ceiling of 1.0. "prec@5 % is actually HIGHER at S=32" is
  a legitimate read. Same for 1b's "matched base rate" claim (`bc_ge_100` 0.3518 vs `fa_gt_1e-2`
  0.3543).
- **`resolve_t1_baseline`'s docstring claim that the sweep summary's `auc` is "identical" to a
  recomputation from the predictions.** Verified: max |difference| over 38 images = 1.1e−16 at S=32.
  The baseline injected into the ΔAUC and the baseline shown in the `tier1_ref` row are the same
  quantity.
- **`sorted(glob(...))[0]` in `_fm_freeze_window.resolve_t1_baseline:112-117` and `load_label:280-284`**
  (a stale-artifact hazard: silently takes the lexicographically first `config_hash` dir). Checked
  every cell under `models/fang_probe/`: **no cell has more than one hash dir today**. Latent only.
- **`docs/model_evidence.md:28`'s S=64 Tier-1 row used as the baseline for an S=32 FM number.** Already
  cleared by `docs-consistency` (the row is labelled "320 m (S=64)"), and the direction is
  conservative: the same-scale S=32 Tier-1 is 0.4840/0.607/0.6631, which would make the FM's margin
  larger, not smaller.
- **`_w2_azimuth_spread.py` takes a linear mean of a circular quantity.** Real, but 36 of 38 images sit
  in 142–186°, so no wrap is crossed; `_w2_fang_azimuth.py` uses a proper `circ_dist_deg` for the
  statistic that matters. No consequence for the recorded claim.
- **`_w2_fang_azimuth.py:89` standardizes the 38 image-mean embeddings over all images before the LOO
  ridge** (a mild transductive leak into a held-out-r statistic, and `pearsonr` on LOO predictions has
  correlated residuals so its p is anti-conservative). The claim it supports — "illumination direction
  IS in the embeddings" — is a *caveat against* the project's own result, so the bias direction is
  self-penalising. Not filed.
- **`_w2_s32_confirm.py:111`'s gate (b) takes `max(F1_ens, F3_ens)`** where the pre-declaration says
  "F1 if pooled is binding, F3 if per-image is" — a best-of-2 read of a pre-declared single choice.
  The whole CNN-fusion recipe was superseded by the FM two days later and no number from it survives
  in ROADMAP/README/docs. Low enough to leave as a note.

## Verified clean

- `run_loio`'s inner-validation rotation (`src/modeling/evaluate.py:610-636`): every probe in this area
  goes through it, `inner_val_code` is drawn from `unique(groups_train)` with an assert that it never
  collides with the held-out code, so no MLP/LightGBM cell in the freeze window early-stops on the test
  image.
- `EmbeddingBank.lookup` (`_w2_fang_probe.py:108-114`): left-merge on `(obs_id, ti, tj)` with
  `validate="one_to_one"` **and** an `assert not np.isnan(rows).any()`, so a missing or duplicated
  embedding row fails loudly instead of silently mis-joining. The two-scale index merge preserves the
  positional `row{px}` pointers correctly.
- `_StandardizedHead._fit_scaler` / `_apply` (`_w2_fang_heads.py:70-86`): median and z-score are fit on
  the fold's training matrix only and re-used unchanged on inner-val and test; the all-NaN-column
  guard and the `sd == 0 → 1` guard are both present.
- The `fa_gt_1e-2` binarisation is invariant-8-compliant everywhere in this area; the one presence-like
  target (`bc_ge_1`, pos_rate 0.93) is explicitly labelled "presence", fails both gates, and is the
  reason `_fm_count_dist.py` was written and `bc_ge_50`/`bc_ge_100` registered.
- The freeze-window arch sweep is honestly read: 7 cells spanning 0.799–0.811 pooled, the incumbent
  mid-pack, and `DECISIONS.md:3404-3408` declines the +0.007 winner as forking-paths overfit. All seven
  banked verdicts match the quoted values.
- Every number in the five DECISIONS tables for this area (2026-06-11 probe, 2026-06-12 S=32/pool/
  combined, 1a/1c/1f bake-off, 1b/1d/1e/1g freeze window) matches its `verdict.json` to 4 d.p.,
  including the two per-image claims in `docs/model_evidence.md:180-186` (`ESP_046328_2180` 0.7481 ≈
  "0.748"; `ESP_076499_1160` 0.8679 ≈ "0.868", and it *is* the cohort's largest ΔAUC at +0.337).

## Load-bearing map

| probe | cited by | number it produced | verdict |
|---|---|---|---|
| `_fm_freeze_window.py` | `DECISIONS.md:3366-3465`, `PLAN_FM.md:45,217`, `HANDOFF_NEXT_SESSION.md:125`, `_fm_fw_chain{1,2_count,3_s32}.sh`; banks 50 `fw_*` cells | **the frozen recipe: pooled 0.7832 / prec@5 % 0.948 / med AUC 0.7865 / ΔAUC(v) +0.120 / win 0.96**, plus 1b/1d/1e/1g | numbers **reproduce exactly**; gate population wrong (**-2**) |
| `_w2_fang_probe.py` | `DECISIONS.md:3176-3247`, imported by `_w2_fang_heads` + `_fm_freeze_window` | `verdict()` itself (the metric definitions); `t1_gem{64,192,32,96}`, `emb_only`, pool ablation | `verdict()` sound; `dauc_by_cause` is a conditioning artifact (**-1**) |
| `_w2_fang_embed.py` | `DECISIONS.md:3160-3175`, `README.md:270-271,321`, `DATA_DICTIONARY.md`, `PLAN_StripingArtifact.md`, `src/fm_embeddings.py:3` | the whole `dataset_v2/fang_embeddings/` store (161,005 × 4 inputs); "100 % context coverage" | verified: 0 invalid rows; deploy parity holds on the plain path (A1 path = R07) |
| `_w2_fang_heads.py` | `DECISIONS.md:3308-3357`, `PLAN_FM.md:54,217`, `src/modeling/mlp_head.py:4` | 1a head bake-off (`mlp_ens3` 0.7852/0.8035), 1f (0.8040/0.8284) | reproduces; head-class conclusion survives Bonferroni + de-ensembling |
| `_w1_build_dossier.py` *(out-of-area producer, in-area consumer)* | every gate in this area, `notebooks/_build_19.py:123` | `validity_ok` (27/38) and `attributed_cause` | both are the root of **-1** and **-2** |
| `_w2_cnn_verdict.py` | `DECISIONS.md:2924,2934-2960` | the W2 Phase-1 CNN verdict (H-B REFUTED; no cell promotable) | comparison is fair (same target/scale); verdict stands |
| `_w2_fang_azimuth.py` | `DECISIONS.md:3287-3300`, `notebooks/_build_20.py`; writes `reports/figures/19_w2_fang_azimuth_read.png` + `azimuth_read.json` | ρ(ΔAUC, incidence) −0.058; LOO ridge sin(az) r=+0.588; ESP_076499_1160 +0.458 | +0.458 correct; `DECISIONS:3204`'s +0.087 is not (**-4**) |
| `_w2_seed_ensemble.py` | `DECISIONS.md`, `PLAN_CNN.md`, `docs/w2_litreview.md`, `notebooks/_build_19.py` | the 3-seed CNN ensemble + F1/F3 fusion numbers | superseded by the FM; consistent with `_w2_cnn_verdict` |
| `_fm_count_dist.py` | `DECISIONS.md:3390-3394`, `PLAN_FM.md`, `src/modeling/binary_target.py` | the `bc_ge_50` / `bc_ge_100` thresholds (0.48 / 0.35 pos rate) | correct; base-rate matching to `fa_gt_1e-2` verified |
| `_fm_parity_check.py` | `README.md:271`, `PLAN_FM.md`, `DATA_DICTIONARY.md` | probe↔`src` embedding parity (cosine > 0.999) | compares only `valid` rows; sibling of the register's `fm-embeddings-2` |
| `_w2_fang_head_pairs.py` | `DECISIONS.md:3315,3336-3341`; writes `models/fang_probe/head_pairs.json` | pairwise head deltas (mlp_ens3 vs lgbm/logreg/knn) | 6 uncorrected tests, but the verdict survives correction |
| `_w2_s32_confirm.py` | `DECISIONS.md`, `notebooks/_build_19.py` | S=32 CNN-recipe confirmation read | best-of-2 gate (b); superseded, no surviving number |
| `_w2_midgrid_diag.py` | `DECISIONS.md:2924`, `docs/w2_litreview.md`, `_w2_fusion.py` | "CNN ranks within, GBM scales across" (rank-corr +0.22 vs +0.41) | diagnostic; cross-target caveat is declared |
| `_w2_azimuth_spread.py` | `DECISIONS.md:2951-2960` | "cohort azimuth 142–186° for 36/38"; the two outliers | linear mean of a circular quantity, no wrap crossed |
| `_w2_adabn.py` | `docs/w2_litreview.md`, `notebooks/_build_19.py`; writes `models/_sweep_cnn/_adabn_cellA.parquet` | AdaBN rescue read on cell A | closed CNN arm; not re-derived |
| `_w2_photonly_read.py` | `DECISIONS.md` | cell-E photometric-only de-confound | closed CNN arm |
| `_w2_fusion.py` | `docs/w2_litreview.md` | F1/F2/F3 fusion table | superseded by `_w2_seed_ensemble` |
| `_w2_fang_patch_visual.py` | — (writes `reports/figures/19_w2_fang_patch_alignment_*.png`, embedded in notebook 20 §3) | the input-alignment eyeball check | figure only |
| `_w2_fang_inspect.py`, `_w2_fang_ckpt_keys.py` | — | pre-flight prints (dossier counts, 3×3 coverage 71 %, checkpoint key layout) | cited nowhere; no artifact |

## Coverage note

**Read in full:** `_w2_fang_probe.py`, `_fm_freeze_window.py`, `_w2_fang_heads.py`,
`_w2_fang_embed.py`, `_w2_cnn_verdict.py`, `_w2_seed_ensemble.py`, `_w2_fang_head_pairs.py`,
`_w2_fang_azimuth.py`, `_w2_s32_confirm.py`, `_w2_fusion.py`, `_w2_midgrid_diag.py`, `_w2_adabn.py`,
`_w2_azimuth_spread.py`, `_w2_photonly_read.py`, `_fm_count_dist.py`, `_fm_parity_check.py`,
`_w2_fang_inspect.py`, plus the out-of-area producer `scripts/probes/_w1_build_dossier.py` and
`scripts/probes/_sweep_w0.py:50-63` (the source of `validity_ok`). Also read `src/fm_embeddings.py`
in full and `src/modeling/evaluate.py:555-645` (`run_loio`), `src/modeling/mlp_head.py:1-50,130-145`.

**Grepped only:** `_w2_fang_ckpt_keys.py`, `_w2_fang_patch_visual.py`, `_fm_fw_chain*.sh`,
`notebooks/_build_19.py` (read only the `VOK` usage), `notebooks/_build_20.py` (read §1–§3 cells).

**Reproduced numerically from committed artifacts:** the frozen recipe's three headline metrics and
its `tier1_ref` row; the Tier-1 summary-vs-predictions AUC identity at S=32 (max Δ 1.1e−16); all 90
`verdict.json` files enumerated and cross-checked against the five DECISIONS tables; per-image AUC and
ΔAUC for all 38 images at S=32 and 37 at S=64; the `validity_ok` recomputation at both scales; the
`valid` masks of all 152 embedding `.npz` files; class-wise means and Spearman correlations for
finding 1.

**Could not check (needs execution or data outside the rules):** anything requiring a rerun — whether
the `batch=512 → 4096` change between the probe `MLPHead` and the productized
`MLPClassifierHead` (`src/modeling/mlp_head.py:22-30`, declared "no material effect on the numbers",
never measured) actually leaves 0.7832 intact; whether `_fm_parity_check.py` still passes on today's
`src/fm_embeddings.py` (needs the 341 MB checkpoint and a GPU); whether the 3 MLP seeds produce
genuinely distinct networks (only the `torch.manual_seed(self.seed)` call was read); the CTX window
nodata content behind R13's "training context boxes came from gap-free CTX windows" premise — the
probe applies **no** zero-fraction test to the 3×3 box at extraction time
(`_w2_fang_embed.py:217-221`), so that premise is unverified in this direction too. Also **not** in
scope here and still open: the Tier-1 reference's own early-stopping defect (**R32**), which every
gate in this area is scored against.
