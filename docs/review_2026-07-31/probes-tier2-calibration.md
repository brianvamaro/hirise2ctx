# Review area: probes-tier2-calibration

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-02
- **Verification:** self-refuted (single-agent pass; not independently verified). Every number below
  was **recomputed from the on-disk artifacts the docs name**; the artifact paths and the
  aggregation are quoted so a verifier can re-run the same snippets. No probe was executed except
  `_diag_tier1_accuracy.py`, which is a pure pandas readout of a banked parquet (see *Coverage note*).

Scope: the 17 probes listed for this area in `_prompts_probes.md` §2, plus `src/calibration.py` (the
module every one of them calls) and the parts of `notebooks/_build_23.py` that are their declared
consumer/producer. Triage was by citation (`DECISIONS.md`, `PLAN_Calibration.md`, `PLAN_FM.md`,
`docs/`, `HANDOFF_NEXT_SESSION.md`, `notebooks/_build_23.py`, `models/fang_tier2/**`), then a
statistic-level audit of the load-bearing ones.

---

## Findings

### probes-tier2-calibration-1 — The shipped abundance layer's calibration is reported **pooled only** (`top_ratio 0.86`, gate PASS); the per-image figure is **0.566 median / 0.168 p10, with only 11 of 37 images inside the declared [0.8, 1.2] band** — and the project measured exactly this harsher statistic for the *F* layer and never for the shipped one

- **Severity:** high
- **Liveness:** live-shipped — `models/deployable/calibration.npz` is the Stage-1 layer applied by
  `src/mapping.py:272` to produce the regional abundance raster, the project's headline product
- **Confidence:** high (recomputed from the banked LOIO predictions + `dataset_v2/labels`; the pooled
  value reproduces the record to 4 dp)
- **Where:** `src/calibration.py:251-276` (`compression_metrics` — pooled by construction),
  `scripts/probes/_diag_calibration_preview.py:45-56` (the Stage-0 producer of "0.71→0.87"),
  `scripts/probes/_diag_tier2_l1_bakeoff.py:306-317` (`score_point`, same pooled call, the Stage-2
  scorecard), `scripts/bank_calibration.py:54-67` (the shipped gate). Consumers:
  `DECISIONS.md:3794-3796`, `PLAN_Calibration.md:218`, `:311`, `:356`, `DECISIONS.md:5044`

`compression_metrics` computes `top_ratio = mean(pred | true > 1e-2) / mean(true | true > 1e-2)` over
**all tiles of all 38 held-out images pooled**. That is a true-mass-weighted average of the per-image
ratios, so it is dominated by the handful of high-abundance images. Every Stage-0/1/2 probe reports it
pooled, no doc states the aggregation, and `PLAN_Calibration.md:356` gates on
`top-bin ratio ∈ [0.8,1.2]` without saying which. Recomputed on the shipped one-model layer
(LOIO-honest, `quantile_match` fit on the other 37 images per held-out image):

| aggregation | value |
|---|---|
| **pooled** (what is reported / gated) | **0.8573** |
| per-image median | **0.5656** |
| per-image mean | 0.7476 |
| per-image p10 / p90 | **0.1683** / 1.3604 |
| per-image min / max | 0.0092 / 2.8625 |
| images inside the declared band [0.8, 1.2] | **11 / 37** |

The spread is not small-sample noise: the extremes include the largest images
(`ESP_076499_1160`, 2928 rich tiles, ratio **0.009**; `ESP_066634_2210`, 1458 rich tiles, **2.86**;
`ESP_045139_2270`, 4559 rich tiles, **2.16**; `ESP_068483_2280`, 4275 rich tiles, **0.263**). The
dedicated Tier-2 regressor behaves the same way: pooled 0.8742 / per-image median 0.5361 / p10 0.1497 /
6 of 37 in band.

- **Failure scenario:** a reader of `PLAN_Calibration.md:311` — "(Tier-1 ECE 0.014; Tier-2 top_ratio
  **0.86**, rank-preserved) are the **conservative bound**" — or of `DECISIONS.md:3794` ("Tier-2
  quantile-matching is the post-hoc win: top-bin ratio 0.71→**0.87** … marginal-L1 0.0057→**0.000**")
  concludes the abundance product is level-calibrated to ~15 %. On the typical held-out *place* the
  rich-tile mean is under-valued by ~1.8×, and across places the level error spans two orders of
  magnitude. That is exactly the use-case `docs/model_evidence.md:334-340` gates on
  ("THEMIS-comparable absolute abundance") and exactly the quantity PLAN_RegionalMap's thermal legs
  will compare against. The pooled figure is the *optimistic* aggregation, so calling it a
  "conservative bound" inverts the direction of the caveat.
- **Evidence:**
  ```python
  # src/calibration.py:266-271 -- one pooled ratio over the concatenated 38-image vector
  top = yt > rich_threshold
  ...
  "top_ratio": float(yp[top].mean() / yt[top].mean()) if top.any() else float("nan"),
  ```
  ```python
  # scripts/bank_calibration.py:62-67 -- the only gate the shipped layer ever saw, pooled
  m_loio = compression_metrics(fa, ab_loio)
  print(f"  [LOIO bound] Tier-2 top_ratio {m_loio['top_ratio']:.2f} ... "
        f"(gate top in [0.8,1.2]: {'PASS' if 0.8 <= m_loio['top_ratio'] <= 1.2 else 'FAIL'})")
  ```
  Reproduction (pandas only, no probe run): join
  `models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/predictions.parquet` to
  `dataset_v2/labels/*.parquet` on `(obs_id, ti, tj)` at `tile_size_px == 32`, apply
  `loio_calibrate(..., quantile_match)`, then compute `top_ratio` pooled and per `obs_id`. Pooled
  reproduces `0.8573` exactly (`DECISIONS.md:5044` "the mosaic layer's number of record is 0.8573"),
  Spearman 0.6248, near-zero 18.57 %, marginal-L1 0.0002.
- **The project already knows the per-image number is far harsher — for the other layer.**
  `DECISIONS.md:5049-5053`, during the F build: *"I first reported Tier-2 `top_ratio` as a **median
  over per-image ratios** (0.5925) … but the 0.8573 on record is a **pooled** statistic, and a
  per-image ratio is far harsher … Both are now reported; the pooled one is the gate. The per-image
  spread is itself informative: median 0.5925, p10 0.0696."* The F layer's 0.5925/0.0696 and the
  shipped layer's 0.5656/0.1683 are the same phenomenon, so this is a property of the pooled-qmatch
  design, not of the F path — but the correction was applied only to the F gate and never
  back-propagated to the shipped layer or to `bank_calibration.py`.
- **It also measures the risk the plan deferred.** `PLAN_Calibration.md:265-266` lists
  "**Marginal-match assumes in-cohort** — mitigate via per-region fit or novelty gating (documented
  risk)", and `:373-376` scopes it out because the regional map is "in-distribution by construction".
  The 0.009–2.86 spread is *within* the cohort the map was fit on, so the deferral's premise is
  already violated at the only scale where truth exists.
- **Self-refutation attempted:**
  (i) *"The per-image ratio is harsher by construction because the calibrator is LOIO."* No — both
  numbers are computed from the **same** LOIO-calibrated vector; only the aggregation differs
  (mass-weighted vs unweighted). (ii) *"Is per-image top_ratio < 1 simply forced by imperfect
  ranking?"* Partly — with per-image ρ ≈ 0.44, `E[qm(pred) | true rich] < E[true | true rich]` is
  expected. But that is the point: the pooled aggregation *hides* it, and the band [0.8, 1.2] was
  never calibrated against what a ranking-limited model can achieve per image. The
  finding is that the reported level is an artefact of aggregation, not that qmatch is broken.
  (iii) *"Already filed?"* No. `calibration-5` (in `calibration.md`) covers the 0.2932 abundance
  **ceiling**; `calibration-4` covers the gates the banking script never evaluates and explicitly
  re-derived only the Spearman and AUC constraints; **R33** is the F `full` arm's clamping. Grepping
  `docs/review_2026-07-31/*.md` for `0.857` / per-image `top_ratio` returns only the F-layer
  disclosure, which that reviewer called "honest" — correctly, because the F script *does* report
  both. (iv) *"Does it change the map's ranking or the abort?"* No — qmatch is monotone, so Spearman /
  AUC / the striping and abort verdicts are untouched. Only the abundance *values* are affected.
- **Fix:** report both aggregations everywhere `top_ratio` appears (as `bank_calibration_f.py:121-128`
  already does), and make `bank_calibration.py` emit the per-image median/p10 and the
  fraction-of-images-in-band beside the pooled gate. Amend `PLAN_Calibration.md:311` (drop
  "conservative bound" or attach the per-image figure), `:218`, `:356` (state *pooled*), and
  `DECISIONS.md:3794-3796`. If the per-image level matters for the THEMIS leg, the documented
  mitigation (per-region / per-image qmatch, `PLAN_Calibration.md:265`) is no longer deferrable.

---

### probes-tier2-calibration-2 — "`min_confidence` filtering is HARMFUL, ruled out — monotonically degrades ranking" is a **two-factor** comparison (the model *and* the target changed); with the target held fixed, `conf ≥ 0.5` is a **null** (paired Δ −0.003, p = 0.43) and the arm's own banked scorecard shows per-image ρ going **up**

- **Severity:** high
- **Liveness:** dead-closed programme, but it is the recorded justification for the **live**
  `min_confidence: null` in both `config.yaml` and `config_v2.yaml`, it closed a CLAUDE.md §11 open
  item, and it is one of the "five ways" behind the Stage-2 closure (see finding 3)
- **Confidence:** high (the probe's own paired numbers reproduce to 4 dp from the banked per-tile
  parquets; the decomposition uses the same three artifacts)
- **Where:** `scripts/probes/_diag_tier2_minconf_sweep.py:98-111` (`run_threshold` — retrains **and**
  re-scores on the filtered target), `:149-161` (scoring + the paired Wilcoxon), `:168-169`.
  Consumers: `DECISIONS.md:3916-3923`, `PLAN_Calibration.md:179-187`, `:274`,
  `notebooks/_build_23.py:504-518` (published notebook 23 §8), `:568-570` (§9 verdict)

Each arm trains `mlp_reg` on labels regenerated at `score ≥ t` **and** scores it against those same
regenerated labels. The paired per-image Wilcoxon therefore compares `ρ(pred_t, y_t)` against
`ρ(pred_none, y_none)` — two predictors *and* two targets. Decomposing it on the banked per-tile
parquets (`models/fang_tier2/l1_bakeoff/preds_minconf_{none,conf050,conf070}.parquet`, keys aligned
row-for-row, 161,005 tiles × 38 images):

| comparison | conf ≥ 0.5 | conf ≥ 0.7 |
|---|---|---|
| **as the probe measured it** (both factors) | **−0.0210, 11/38 wins, p = 0.0100** | **−0.0703, 7/38, p = 0.0001** |
| target factor only (same `pred_none`, filtered target) | −0.0172, p = 0.0002 | −0.0557, p < 1e-4 |
| **training-label factor only** (filtered model, common unfiltered target) | **−0.0034, 17/38, p = 0.4294** | −0.0213, 13/38, p = 0.0608 |

The first row is what `DECISIONS.md:3921` and `PLAN_Calibration.md:183` quote. Essentially all of it
is the *target* factor: the filtered `fractional_area` is intrinsically harder to rank from CTX,
which a fixed model shows on its own. The factor the decision is actually about — *does training on
higher-confidence labels help or hurt the product?* — is **null** at conf ≥ 0.5 (pooled ρ even rises,
0.6478 → 0.6506) and not significant at conf ≥ 0.7.

- **Failure scenario:** `PLAN_Calibration.md:179-187` records "**HARMFUL, ruled out** … monotonically
  degrades both ranking and dynamic range … Low-confidence detections are **real boulders**, not
  removable noise", and notebook 23 §8 (a published notebook) restates it. A future session reads the
  label-confidence question as **settled against filtering**. It is not settled: on the honest
  one-factor test the answer is "no measurable effect at 0.5", i.e. the cohort could be harmonised at
  a confidence floor with no measured ranking cost — which is exactly the remedy **R23** proposes for
  the two cohort images whose labels are already truncated at 0.407 / 0.617.
- **"Monotonically degrades ranking" is contradicted by the probe's own banked scorecard.**
  `models/fang_tier2/l1_bakeoff/minconf_scorecard.csv` (and notebook 23 cell 19's printed table):

  | label | rich_share | raw_top | **raw_perimg_rho** | **raw_pooled_rho** |
  |---|---|---|---|---|
  | none | 0.360 | 0.664 | **0.4333** | **0.6478** |
  | conf050 | 0.270 | 0.578 | **0.4563** | **0.6471** |
  | conf070 | 0.112 | 0.314 | 0.3044 | 0.5634 |

  Both ranking columns are flat-to-**up** at conf ≥ 0.5. Only the paired median of differences is
  negative, and that is the two-factor statistic above.
- **"Dynamic range monotonically degrades" is a population artefact (the R26 pattern).** `top_ratio`
  is taken at a **fixed absolute** `fa > 1e-2` while the filter changes the rich share 36 % → 27 % →
  11 %, so the "top" set becomes a rarer, more extreme subset in each arm and regression-to-the-mean
  compression is worse there by construction. At a **matched population fraction** (each arm's own
  top 36.0 % of tiles) the sequence is **0.664 → 0.623 → 0.519**, not 0.664 → 0.578 → 0.314: roughly
  half the conf050 loss and two-thirds of the conf070 loss is the population change.
- **The treatment was not applied uniformly — an independent confirmation of R23.** Per-image label
  mass retained at conf ≥ 0.5, computed from the three parquets:
  `ESP_017355_2260` = **1.000** (literally zero treatment — its label set already contains no
  detection below score 0.617), `ESP_068483_2280` = 0.803, cohort median ≈ 0.54, minimum 0.403. So the
  paired n = 38 includes one completely untreated unit — and it is the largest observation in the
  cohort (13,457 tiles at S=32). At conf ≥ 0.7 its retention is 0.577 against a next-highest 0.312.
  This is R23's score-rank truncation observed from a completely different artifact, and it means the
  probe's premise (`:5-7` "the pipeline kept `min_confidence: null`") is false for 2 of the 38 images.
- **Self-refutation attempted:**
  (i) *"The plan already names the mechanism."* `PLAN_Calibration.md:186` does say "filtering **thins**
  the target rather than cleaning it" — but it still records the verdict as HARMFUL/monotone and still
  quotes the two-factor Δ as the evidence. Naming a confound while quoting the confounded number as
  the verdict is not a mitigation. (ii) *"Is scoring on a common target the right question?"* It is the
  deployment question: the product predicts `fractional_area` as the project defines it
  (`min_confidence: null`), and the shipped calibrator is fit to that marginal. I also ran the cleanest
  one-factor test in the other direction (fix the model, vary the target) and report it above, so both
  single-factor decompositions are given. (iii) *"Does the config decision change?"* No — nothing shows
  filtering **helps** either, so keeping `min_confidence: null` is defensible. The defect is the
  recorded evidence and the causal claim, not the setting. (iv) *"Already filed?"* No hit for
  `minconf` / `min_confidence` in any `docs/review_2026-07-31/*.md` finding.
- **Fix:** rescore every arm against **one** target (the unfiltered `fractional_area`) by joining on
  `(obs_id, ti, tj)` — the parquets already support it — and report the fixed-model/varying-target
  delta separately; use a matched-quantile cut for `top_ratio`; record the per-image retention so an
  untreated unit is visible. Then amend `DECISIONS.md:3916-3923`, `PLAN_Calibration.md:179-187` and
  `:274`, and notebook 23 §8/§9 to "**no measurable effect at conf ≥ 0.5; harmful only at 0.7, and
  mostly because the target thins**", and note the R23 interaction.

---

### probes-tier2-calibration-3 — "The ~0.43 per-image ceiling is the **5 m/px CTX magnitude floor**, confirmed five ways" — all five ways hold the frozen Fang-ViT/GeM-96/S=32 embedding and the same MLP trunk fixed, and the "two different model families" that agree are two readouts of one representation (their outputs correlate at pooled ρ = **0.92**)

- **Severity:** high
- **Liveness:** dead-closed plan stage, but the conclusion is the project's standing strategic premise
  — it is the recorded reason "**the path forward is not a better model**", it is what keeps the ViT
  fine-tune **out**, and it is the memory note's one-line summary of Stage 2
- **Confidence:** high (structural, plus a direct measurement of the two heads' agreement)
- **Where:** `notebooks/_build_23.py:402-412` (§7 claim), `:561-575` (§9 verdict);
  `PLAN_Calibration.md:158-170`, `:285-292`, `:392`; `DECISIONS.md:3890-3899`, `:3926-3932`. The five
  "ways" are produced by `_diag_tier2_objectives.py`, `_diag_tier2_l1_bakeoff.py`,
  `_diag_tier2_reweight.py`, `_diag_tier2_minconf_sweep.py`, `_diag_tier2_scale_sweep.py`

Every Stage-2 probe builds its folds the same way —
`augment_fold_with_fang(f, px=96, replace=True, store=load_fang_store(96, pool="gem"))` — and every
head is the same `768→256→64→1`, dropout 0.2, 3-seed trunk
(`_fm_tier2_regression.py:143-149`, `_diag_tier2_l1_bakeoff.py:57-61`,
`_diag_tier2_reweight.py:99-100`, `_diag_tier2_objectives.py:72-73`). The five levers vary the
**loss**, the **target scale**, the **sample weights** and the **label filter**. None varies the
representation. Such experiments can only bound *"is the head extracting everything the frozen
embedding contains"*; they carry no information about whether the **5 m/px CTX imagery** is the limit.
The record nonetheless names the imagery.

The "independent proof" is weaker still. `PLAN_Calibration.md:161-165` and notebook 23 §7 call
Tier-1 `P(rich)` and Tier-2 `mlp_reg` "**two different model families**" that "never saw the continuous
target". They are the same architecture on the same 768-d embeddings over the same 38 LOIO folds, and
the classifier's label `fa > 1e-2` is a deterministic threshold of the regressor's label. Measured
directly on the two banked prediction sets:

```
per-image Spearman(P_rich, mlp_reg point): median 0.8685  (min 0.6497, max 0.9646)
pooled  Spearman(P_rich, mlp_reg point):   0.9221
```

The two "families" are near-duplicate rankers. Their agreeing at ρ ≈ 0.43 against `fractional_area` is
a property of the shared representation, not evidence about CTX.

- **Failure scenario — the conclusion vetoes the only experiment that could test it.**
  `PLAN_Calibration.md` "Decisions — RESOLVED 2026-06-16", item 5: *"**ViT fine-tune** → still **out**
  — the ceiling is the data, not the representation; nothing in Stage 2 motivates it (ties to the
  PLAN_FM fine-tune go/no-go)."* But `PLAN_FM.md:283-284` says that go/no-go is "EXPLICITLY decided
  after §3 lands" — the expansion-cohort confirmation, which never landed. So a representation lever
  was closed on the strength of five experiments that all held the representation fixed, and the
  independent decision point that would have re-opened it was deferred to work that never happened.
- **The plan's own text contradicts the "exhausted" claim.** `PLAN_Calibration.md:189-194` lists three
  **untested** L2 representation levers — "(a) Multi-scale embedding fusion (S=16/32/64) … (b) a small
  **spatial head** over the 3×3 embedding field … (c) **ViT fine-tune** (LoRA/last block) — … if the
  frozen embedding undersells the tail it is the representation-level fix" — plus untried FDS
  feature-smoothing (`:207`) and untried noise-robust losses (`:187`). Yet `:291-292` concludes
  "**no in-cohort retraining lever moves the per-image ranking ceiling** — it is the 5 m/px CTX
  magnitude floor, confirmed five ways" and "**Only S=128** … remains untested".
- **The one lever that did change the input points the other way.** The scale sweep (S=32 → S=64, the
  only manipulation of what the embedder sees) produced the **largest single effect in the whole
  Stage-2 table**: per-image ρ 0.4333 → 0.4816, paired Δmed **+0.025**, 25/38 images — and it was
  dismissed as "p = 0.19, not significant at n = 38", while a −0.014 (p = 0.018) reweighting delta of
  half that size was accepted as a "significant ranking cost". The evidence base treats a positive
  representation effect as noise and a negative loss-shape effect as signal.
- **Self-refutation attempted:**
  (i) *"Isn't 'the data' loose shorthand for 'the frozen inputs'?"* It is not read that way: it is used
  to veto a *representation* change (item 5) and to declare "the §2.3 expansion cohort is the only
  thing that can raise the ranking ceiling" (`:293-295`). Those are substantive, and the honest
  statement — "the ceiling is the frozen Fang-ViT GeM-96 S=32 embedding at this cohort size" — implies
  a different next step. (ii) *"Was the representation varied downward?"* Yes — the `t1` handcrafted
  arm (mlp_reg ρ 0.223 vs emb 0.431) shows the embedding beats handcrafted features by ~2×. That
  demonstrates the representation **matters a lot**, which argues against, not for, the "the head can't
  be the problem" reading. (iii) *"Does the striping record agree?"* No — `DECISIONS` 2026-07-05d (the
  amended F verdict that opened the H1–H6 docket) concluded the **embedder** is the real floor, the
  opposite attribution, and neither entry cites the other. (iv) *"Is this just `docs-consistency-8`?"*
  No — that finding is about "per-image AUC ≈ 0.43" being a Spearman ρ mislabelled as an AUC. This is
  about what the 0.43 is *attributed to*. (v) *"Same fallacy elsewhere?"* Yes —
  `docs/modeling_results.md:805-817` runs the v1-era version ("Three independent target framings … at
  the same ceiling is the strongest evidence"), and **R45** flags the pairing defect in one of those
  three. Recorded as a recurring pattern, not double-filed.
- **Fix:** restate the Stage-2 conclusion as "no *loss-, weighting- or label-side* lever moves the
  ranking ceiling **on the frozen embedding**"; strike "confirmed five ways" or list the five and note
  they share one representation; remove "Only S=128 remains untested" (three representation levers and
  FDS are also untested); and re-open decision item 5 as *undecided*, deferring properly to PLAN_FM's
  own go/no-go rather than pre-empting it. Drop "two different model families" in
  `PLAN_Calibration.md:164` and notebook 23 §7 and quote the ρ = 0.92 agreement instead.

---

### probes-tier2-calibration-4 — `DECISIONS.md`'s L1 bake-off deltas do not match the committed producer's table, and the `hlgauss.mode` paired test silently drops a fold

- **Severity:** low
- **Liveness:** dead-closed; the verdict ("all a wash") is unaffected
- **Confidence:** high (both tables reproduced)
- **Where:** `DECISIONS.md:3870-3877` vs `notebooks/_build_23.py:310-334` (the declared producer) and
  its executed output in `notebooks/23_calibration_diagnostic.ipynb`;
  `scripts/probes/_diag_tier2_l1_bakeoff.py:300-303` (`per_image_spearman`)

`DECISIONS.md:3873` records "*HL-Gauss.mean Δ **−0.017** (p≈0.08), ziln Δ **−0.019/−0.025**"*. The
committed notebook cell — the only code in the repo that computes these paired deltas, and which I
reproduced independently from `models/fang_tier2/l1_bakeoff/preds_*.parquet` — gives:

| readout | raw_top | perimg_rho | paired_d | p |
|---|---|---|---|---|
| hlgauss.mean | 0.72 | 0.433 | **−0.011** | 0.077 |
| hlgauss.mode | 0.60 | 0.343 | −0.065 | 0.000 |
| hlgauss.p90 | 1.13 | 0.412 | −0.024 | 0.004 |
| pinball.median | 0.62 | 0.462 | −0.002 | 0.482 |
| pinball.p90 | 0.98 | 0.460 | +0.000 | 0.388 |
| ziln.mean | 0.66 | 0.435 | **−0.012** | 0.051 |
| ziln.median | 0.55 | 0.427 | **−0.007** | 0.280 |
| ziln.p90 | 1.17 | 0.413 | −0.021 | 0.005 |

`pinball.median −0.002 / 18-of-38 / p = 0.48` matches the record exactly, so the producer is right;
the HL-Gauss and ziln deltas in `DECISIONS` match nothing in it (`−0.019/−0.025` is not any ziln row).

Separately, `hlgauss.mode` is **constant on 1 of the 38 images** (a bin-centre readout can collapse).
`_diag_tier2_l1_bakeoff.py:303` takes `np.nanmedian` and reports no `n`; notebook 23's `per_img_rho`
(`:315-317`) drops the image via `g[col].nunique() > 1`, so that row's paired test is over 37 images
while every other row is over 38, with nothing in the output saying so — the same silent-n pattern as
**R24**.

- **Failure scenario:** low. The bake-off verdict ("wash on ranking, keepers = pinball.P90 and
  ziln.median") is unchanged at either set of numbers; only the magnitudes quoted in the log are wrong,
  and one row's `n` is over-stated by 1.
- **Fix:** re-quote `DECISIONS.md:3873` from the notebook table; have both the probe and the notebook
  emit `n` beside every median/paired statistic.

---

### probes-tier2-calibration-5 — `_fm_tier2_ceiling.py` is the declared producer of a "zero-inflation ceiling" but computes no ceiling; the arithmetic one is 0.9997, so the hypothesis it "empirically refuted" was never arithmetically possible

- **Severity:** low
- **Liveness:** dead-closed (the conclusion recorded is correct)
- **Confidence:** high
- **Where:** `scripts/probes/_fm_tier2_ceiling.py:1-14` (docstring), `:47-62` (`per_image`);
  consumer `DECISIONS.md:3519-3539`

The probe's headline instrument is `zero drag = rho_among_pos − rho_overall`, i.e. the difference
between two Spearman coefficients computed over **two different populations** (all tiles vs `y > 0`).
That is a range-restriction delta, not a bound, and `DECISIONS.md:3531-3534` reads its −0.011 value as
*"the zeros are the easy part … The earlier '0.43 is capped by zero-inflation' framing is empirically
refuted"*. The actual ceiling is one line of arithmetic: with the per-image exact-zero share averaging
0.163, the tie structure of `y_true` caps per-image Spearman at a **median 0.9997** (worst image
0.8649) against an observed 0.4446. Zero-inflation was never within two decades of being the binding
constraint, so nothing needed refuting — and the delta the probe reports would not have shown it if it
had been.

The one genuinely ceiling-normalised statistic in the probe is NDCG (`:39-44`, IDCG = the ideal
ordering), and it is reported (0.502 @5 %, 0.851 full) — but the DECISIONS table leads with the drag.

- **Failure scenario:** low — the recorded conclusion is right. The cost is a mis-named instrument
  carried into `DECISIONS.md` as an "evidence" column, which a future session could reuse to "test a
  ceiling" it cannot test.
- **Fix:** report the tie-imposed maximum attainable Spearman (a two-line computation on `y_true`
  alone) as the ceiling, keep NDCG, and re-label the `rho_among_pos` row as a range-restriction
  diagnostic rather than a ceiling.

---

## Refuted by my own check

- **"Isotonic is AUC-exact at deployment" was demonstrated in-sample.** True — `_diag_tier1_beta.py:42-50`
  fits on all 38 and scores all 38, and isotonic is *not* strictly monotone, so `+0.0003` is an
  in-sample number that a monotone map with ties cannot be assumed to reproduce out of sample. But the
  honest measurement kills it: per-image AUC under the **LOIO** isotonic map costs a median
  **−0.00046** (worst image −0.0059, 31/38 worse), well inside the declared ±0.005 gate, and applying
  the actually-shipped global map (`models/deployable/calibration.npz`) to the LOIO predictions gives
  0.84837 → **0.84867**. The pooled LOIO drop 0.848 → 0.833 really is the per-fold-map artefact the
  record says it is. Claim survives; recorded so it is not re-opened.
- **"Regression matches the classifier on rich/poor: 0.784 ≈ 0.7865" compares a mean to a median.**
  It does — `_fm_tier2_regression.py:276` prints `meaningful_auc_mean` while `0.7865` is the frozen
  recipe's *median* per-image AUC. Like-for-like the claim still holds and slightly understates:
  mean-vs-mean 0.7843 vs **0.7747**, median-vs-median 0.7922 vs **0.7865**. Not a defect.
- **The Tier-2 headline table compares prevalence-sensitive metrics across the `fa` and `count`
  targets** (base rates differ ~3×, cf. R26). It does not — `DECISIONS.md:3493-3499` compares only
  `meaningful_auc` (prevalence-invariant); the prevalence-sensitive `pr_auc_mean` (0.526 vs 0.412) and
  `precision_at_top_5pct_mean` (0.591 vs 0.441) are computed but never compared cross-target in any doc.
- **"count-Poisson is worse *because* the count→area conversion discards size info."** The conversion
  (`_diag_tier2_objectives.py:162`, a single global constant `mean_indiv / tile_area`) is
  rank-preserving, so it cannot affect the ρ 0.425-vs-0.433 comparison at all — only `top_ratio` and
  `marginal_l1`. The sentence is loose, but the substantive point (count is a lossy proxy for area) is
  right and the ρ difference is a wash anyway. Not worth filing.
- **LDS reweighting "DOMINATED" is confounded like finding 2.** It is not — `_diag_tier2_reweight.py:141-145`
  holds `fractional_area` fixed across all three arms, so only the model changes. Paired deltas
  reproduce exactly (`lds_sqrt` −0.0138, 11/38, p = 0.0176; `lds_inv` −0.0322, 13/38, p = 0.0148).
- **The reliability leg's "LOIO-NEGATIVE at n=38" is too weak to support a deferral.** It is adequately
  hedged: `DECISIONS.md:3670-3684` states the bar was not cleared, the direction is right, and
  "n=38 is underpowered — direction right, CI wide". I reproduced both ρ from
  `reports/reliability/per_image_novelty.csv` (mahalanobis −0.1078 p = 0.520; knn_cos50 −0.1413
  p = 0.398) and the study can only detect |ρ| ≳ **0.33** at n = 38 (Fisher-z), so "defer and re-run
  post-expansion" is the correct call. One weak spot worth a line if it is ever re-run: the
  "bottom-5-AUC flag precision **0.00**" quoted as corroboration is uninformative — under the null,
  P(zero overlap between two random 5-of-38 sets) = **0.47**.
- **`_fm_tier2_ceiling.py:79` / `_diag_tier2_variant_compression.py:20` resolve the cell by the
  *first* glob hit** (`sorted(...)[0]` / `list(...)[0]`) rather than the intended `config_hash` — the
  `notebooks-1` pattern. Latent only: every `models/fang_tier2/tier2_*` label has exactly one
  config-hash directory on disk today, and `--force` re-runs into the same hash. Worth one line in the
  probes.
- **Two banked "mlp_reg baseline" incarnations are quoted interchangeably.** The `run_loio` harness cell
  (`.../1e01ad8b17447599/predictions.parquet`, batch 512/60 epochs/rotated inner-val image) gives
  pooled ρ 0.6507 and `top_ratio` 0.7139 — the "0.71→0.87" of `DECISIONS.md:3794` — while the Stage-2
  baseline (`l1_bakeoff/preds_mlp_reg.parquet`, batch 4096/random-10 %-rows inner val) gives 0.6478 and
  0.6637 — the "0.66→0.87" of `DECISIONS.md:3852`. Both appear in `PLAN_Calibration.md` as *the*
  mlp_reg raw top-bin ratio (`:31` 0.71, `:151` 0.66). Real but immaterial; noted so it is not chased.
- **`_diag_tier2_reweight.py`'s `none` arm looked like a ±0.025 reproducibility hole** (median per-image
  ρ 0.4582 vs the bake-off's 0.4333, from a device-side `randperm` and 50-vs-60 epoch cap). Killed: the
  **paired** difference between the two baselines is only −0.0014 (15/38, p = 0.29). What the exercise
  does show is that the banked `raw_perimg_rho` column (a median of 38) is unstable at ±0.025 between
  runs whose paired difference is ~0.001 — which is why the paired tests, not the medians, must be
  quoted (the record already does this, and notebook 23 says so explicitly).
- **The `+qmatch` column of the L1 bake-off is pinned by construction.** All nine readouts land at
  `qm_top_ratio` 0.840–0.870 and `qm_marginal_l1` 0.0002–0.0005 regardless of raw behaviour — including
  `hlgauss.mode`, whose per-image ρ is 0.34. The column discriminates nothing. Not filed because the
  record reads it correctly ("all qmatch can't fix", `DECISIONS.md:3876`) and uses the *raw* column for
  the keepers.

## Verified clean

- **Every headline number in this area reproduces from the on-disk artifacts**, to the precision
  quoted: Tier-1 LOIO ECE raw 0.0604 / temperature 0.0490 / **isotonic 0.0137** / beta 0.0395 and
  pooled AUC 0.8484 / 0.8439 / **0.8331** / 0.8354 (`DECISIONS.md:3816`, `:3843-3848`); the split-ECE
  low/high pairs (raw 0.043/0.096, temp 0.063/0.021, iso 0.014/0.014); Tier-1 accuracy 0.800 @0.5 vs
  0.640 majority, bal-acc 0.775, F1 0.712, prec/rec 0.737/0.689 (`DECISIONS.md:3859-3860`); Tier-2
  Stage-0 raw ρ 0.6507 / top 0.7139 / near-zero 1.79 % / L1 0.0057 → qmatch 0.6443 / **0.8742** /
  18.59 % / 0.0003, and isotonic's non-help (top 0.7043, L1 **0.0077**, i.e. worse than raw)
  (`DECISIONS.md:3794-3796`); the whole L1 bake-off scorecard, the scale sweep (0.4333 → 0.4816, paired
  +0.025, p = 0.194), the reweight scorecard and the minconf scorecard; and both reliability ρ.
  `notebooks/23_calibration_diagnostic.ipynb`'s executed outputs match the on-disk artifacts exactly —
  the notebook is not stale.
- **LOIO honesty of every probe in the area.** `loio_calibrate` (`src/calibration.py:284-300`) is used
  correctly everywhere (fit on the other images' `(pred, true)`, apply to the held out).
  `inner_val()` in `_diag_tier2_{l1_bakeoff,reweight,scale_sweep,minconf_sweep,objectives}.py` draws a
  random 10 % of **training-fold rows only** — it never touches the held-out image, so early stopping
  is clean (it differs from the frozen recipe's rotated-inner-val-image rule, which is a fidelity note,
  not leakage). `_fm_reliability_validation.py:69-81` refits novelty per held-out image on the other 37.
- **Invariant 8 (no presence AUC) is respected throughout this area.**
  `_fm_tier2_regression.py:64-73` and `:262-270` thread a per-target `meaningful_threshold`
  (fa > 1e-2, count ≥ 50) precisely to avoid the `bc_ge_1` degeneracy, with an explanatory comment; the
  per-fold `meaningful_threshold` in every banked `metrics.json` is 0.01 / 50.0 as claimed.
  `_fm_reliability_validation.py:50-60` scores against the frozen cell's binary rich label, i.e. the
  meaningful AUC, not presence. No probe in this area computes `y_true > 0`.
- **No silent fold drop in the Tier-2 aggregates** (the R24 channel): all three `mlp_reg` cells report
  `n_real_folds = 38`, `spearman_n = 38`, and all 38 per-fold `meaningful_auc` values are finite.
- **`_diag_tier2_minconf_sweep.py`'s label-regeneration harness** is sound where it claims to be: the
  `none` arm reproduces `dataset_v2` labels exactly (0 key-misses, confirmed by the identical
  `y_true` vectors in `preds_minconf_none.parquet` and the bake-off baseline), the `(obs_id, ti, tj)`
  keys are row-aligned across all three arms, and `remap()` falls back to the fold's own `fa` so a miss
  cannot fabricate a zero.
- **`_fm_reliability_smoke.py`** is a synthetic near/far smoke test that emits no reported number, and
  **`_fm_tier2_collect.py`** is a pure metrics-table printer.

## Coverage note

**Read in full (all 17 probes in scope):** `_diag_tier1_accuracy.py`, `_diag_tier1_beta.py`,
`_diag_tier1_isotonic.py`, `_diag_tier2_compression_direction.py`, `_diag_tier2_l1_bakeoff.py`,
`_diag_tier2_minconf_sweep.py`, `_diag_tier2_objectives.py`, `_diag_tier2_reweight.py`,
`_diag_tier2_scale_sweep.py`, `_diag_tier2_variant_compression.py`, `_fm_tier2_ceiling.py`,
`_fm_tier2_collect.py`, `_fm_tier2_regression.py`, `_fm_reliability_inspect.py`,
`_fm_reliability_smoke.py`, `_fm_reliability_validation.py`, `_diag_calibration_preview.py`. Also read
in full: `src/calibration.py`; and the Stage-2/§7/§8/§9 cells of `notebooks/_build_23.py` plus the
executed outputs of `notebooks/23_calibration_diagnostic.ipynb`. Grepped and read the cited passages of
`DECISIONS.md` (2026-06-13 → 2026-06-16 and the 2026-07 F-gate ruling at :5040-5060),
`PLAN_Calibration.md` (in full), `PLAN_FM.md` §2.4/§2.7, `scripts/bank_calibration.py`,
`docs/model_evidence.md` §8, `HANDOFF_NEXT_SESSION.md`.

**Reproduced numerically** (pandas/scipy over on-disk artifacts): the Tier-1 calibration table and its
in-sample-vs-LOIO decomposition; per-image AUC under each LOIO calibrator; the shipped
`models/deployable/calibration.npz` applied to the LOIO predictions; the Tier-2 Stage-0 compression
table; the full L1 bake-off paired-Wilcoxon table; the scale, reweight and minconf scorecards and their
paired tests; the two-factor decomposition of the minconf result on a common target; matched-quantile
`top_ratio`; per-image `top_ratio` for both the shipped one-model layer and the dedicated regressor;
per-image label retention under each confidence filter; the tie-imposed Spearman ceiling; and
Spearman(P_rich, mlp_reg).

**Executed exactly one probe:** `_diag_tier1_accuracy.py` (28 lines) — a pure pandas readout of a
banked `predictions.parquet` with no imagery, network, GPU or writes. Every other check was run from
my own read-only snippets. No GPU/torch probe, no label regeneration, no notebook and no map build
was run.

**Not checked, and why:**
- **Whether re-running any lever with a *different representation* would move the ceiling** — the
  substantive question behind finding 3. It requires a fresh embedding pass (S=16/S=128 stores, or a
  LoRA fine-tune), which is out of scope for a read-only review.
- **The interval/L4 leg beyond the banked `coverage.json`** (pinball 58.6 %, ziln 58.8 % vs nominal
  80 %). The under-dispersion is real and honestly recorded; I did not audit the two quantile
  constructions (`_diag_tier2_l1_bakeoff.py:224-226`, `:253-271`) for whether part of the shortfall is
  the `np.sort`/clip post-processing rather than the model.
- **`_diag_tier2_scale_sweep.py`'s "easier-target" caveat** — I confirmed the true-zero share changes
  18 % → 6.9 %, but did not construct a matched-difficulty S=64 comparison (e.g. scoring both scales
  against a common coarse-aggregated target), which is what would settle whether the +0.025 is real.
- **`_diag_tier2_variant_compression.py`** — reads the same banked cells as
  `_diag_tier2_compression_direction.py`; I verified its statistics are the standard pooled ones and
  its glob resolution (see *Refuted*) but did not reproduce its printed table, since no doc cites it.
- **The upstream cause of the R23 truncation** (BoulderNet's export) — out of this repo, per §6 of the
  register.

## Load-bearing map

| probe | cited by | number it produced | verdict |
|---|---|---|---|
| `_diag_tier2_minconf_sweep.py` | `DECISIONS.md:3916-3923`; `PLAN_Calibration.md:179-187, 274`; notebook 23 §8/§9; `models/fang_tier2/l1_bakeoff/minconf_scorecard.csv` + 3 parquets | "**HARMFUL, ruled out** — monotonically degrades ranking and dynamic range; conf≥0.5 paired Δ −0.021 p=0.010; conf≥0.7 Δ −0.070 top_ratio 0.66→0.31" | numbers **REPRODUCE**; **verdict WRONG — finding 2** (two-factor; label factor null at conf≥0.5, p=0.43; "monotone" contradicted by its own scorecard; `top_ratio` population-confounded) |
| `_diag_calibration_preview.py` | `DECISIONS.md:3791-3796`; `PLAN_Calibration.md:31, 218, 311`; notebook 23 §2/§4 | "Tier-1 ECE 0.060, AUC 0.848; Tier-2 top-bin **0.71→0.87**, near-zero 1.8 %→18.6 %, marginal-L1 0.0057→0.000, Spearman 0.651→0.644; isotonic does NOT help" | **REPRODUCES exactly**; but the top-bin ratio is **pooled-only — finding 1** (per-image median 0.566) |
| `_diag_tier2_l1_bakeoff.py` | `DECISIONS.md:3870-3881`; `PLAN_Calibration.md:106-118, 273`; notebook 23 §5; `l1_bakeoff/{scorecard.csv,coverage.json}` + 4 parquets | "all a WASH on ranking (best pinball.median p=0.48); keepers pinball.P90 top_ratio 0.98, ziln.median near-zero 9.9 %; intervals 58 % vs 80 %" | scorecard **REPRODUCES**; verdict sound. Two defects: **DECISIONS' HL-Gauss/ziln deltas don't match the producer — finding 4**; the `+qmatch` column is pinned (0.84–0.87 for every readout) |
| `_diag_tier2_reweight.py` | `DECISIONS.md:3911-3915`; `PLAN_Calibration.md:196-206, 275`; notebook 23 §8; `reweight_scorecard.csv` + 3 parquets | "**DOMINATED** — top_ratio 0.67→0.77→0.88 but paired ranking cost −0.014 p=0.018 / −0.032 p=0.015" | **REPRODUCES; comparison is honest** (target held fixed across arms) |
| `_diag_tier2_scale_sweep.py` | `DECISIONS.md:3882-3889`; `PLAN_Calibration.md:148-157, 274`; notebook 23 §6; `scale_sweep.csv`, `preds_mlp_reg_S64.parquet` | "S=32→64 directional: per-image ρ 0.433→0.482, paired +0.025, **p=0.19 n.s.**; top_ratio 0.66→0.72" | **REPRODUCES**; the caveat is honestly recorded — but it is the **only** input-side lever and its dismissal is load-bearing for **finding 3** |
| `_diag_tier2_objectives.py` | `DECISIONS.md:3851-3858`; `PLAN_Calibration.md:93, 170-178`; notebook 23 §5 | "log1p a WASH (top 0.66→0.67, ρ 0.433→0.445); count-Poisson WORSE (ρ 0.425, top 0.54, +qmatch 0.78)" | sound; the stated *mechanism* ("the count→area conversion discards size info") cannot explain the ρ — the conversion is a global constant, hence rank-preserving |
| `_fm_tier2_regression.py` | `DECISIONS.md:3474-3569, 3587`; `PLAN_FM.md:157`; `src/modeling/mlp_head.py:22`; notebook 22; 8 banked `models/fang_tier2/tier2_*` cells | the whole Tier-2 table: mlp_reg emb ρ **0.431**, meaningful_auc **0.784**; "MLP wins; FM ~2× lift; single-stage beats the hurdle" | **REPRODUCES** (0.4307 / 0.7843, `n_real_folds=38`, no NaN drop). One loose comparison: "0.784 ≈ classifier 0.7865" is mean-vs-median; like-for-like it still holds |
| `_fm_tier2_ceiling.py` | `DECISIONS.md:3519-3539`; `PLAN_FM.md:157` | "zero-inflation ceiling TESTED and set aside — zero_frac 0.163, drag −0.011, NDCG@5 % 0.502 / full 0.851" | conclusion correct, **instrument is not a ceiling — finding 5** (true tie-imposed cap = 0.9997) |
| `_fm_reliability_validation.py` | `DECISIONS.md:3658-3684`; `PLAN_FM.md:244`; `HANDOFF_NEXT_SESSION.md:49,140`; `src/reliability.py:22`; `reports/reliability/per_image_novelty.csv`; `reports/figures/27_reliability_validation.png`; `docs/model_evidence.md` §5 | "bar NOT cleared: Maha ρ=−0.108 p=0.52, kNN ρ=−0.141 p=0.40, bottom-5 prec 0.00 → overlay DEFERRED" | **REPRODUCES — clean and correctly hedged.** Power: n=38 detects only \|ρ\|≳0.33. The `prec@5 = 0.00` corroboration is uninformative (P=0.47 under the null) |
| `_diag_tier1_isotonic.py` | `DECISIONS.md:3813-3820`; `PLAN_Calibration.md` Tier-1 section; notebook 23 §3 | "temperature trades the ends (ECE_low 0.043→0.063, high 0.096→0.021); isotonic fixes both (0.060→0.014) but LOIO AUC 0.848→0.833" | **REPRODUCES exactly** |
| `_diag_tier1_beta.py` | `DECISIONS.md:3841-3850`; notebook 23 §3 | "isotonic ECE 0.014 beats beta 0.040; the LOIO AUC drop is a per-fold artifact — global fit isotonic +0.0003, beta +0.0000" | **REPRODUCES**; the global check is in-sample, but the honest per-image LOIO cost is −0.0005 median → claim survives (see *Refuted*) |
| `_diag_tier1_accuracy.py` | `DECISIONS.md:3859-3860` | "0.800 @0.5 vs 0.640 majority; bal-acc 0.775; F1 0.712; prec/rec 0.737/0.689" | **REPRODUCED exactly** (re-ran the probe; pure pandas) |
| `_diag_tier2_compression_direction.py` | `DECISIONS.md:3767-3775`; `docs/model_evidence.md` §8 prose + Figs 8/9 captions | "compression is two-sided: lows floor ~0.005 (1.8 % near-zero preds vs 18 % true zeros), top fixed bin pred/true 0.71, top decile 0.53, crossover ≈ 0.015" | **REPRODUCES**; pinned to an explicit `config_hash`, no glob ambiguity |
| `_fm_reliability_inspect.py` | `HANDOFF_NEXT_SESSION.md:51-52` | the outlier confound: ESP_076499_1160 novelty rank 1/38 at AUC 0.868; drop-it ρ −0.174 / −0.210 | consistent with the banked CSV; supports the (correct) negative |
| `_diag_tier2_variant_compression.py` | — (uncited) | per-variant ρ / low_over / top_ratio / distL1 readout | not cited, writes nothing; resolves the cell by first glob hit (latent) |
| `_fm_tier2_collect.py` | — (uncited) | prints the `models/fang_tier2/*/metrics.json` table | not cited, writes nothing |
| `_fm_reliability_smoke.py` | `HANDOFF_NEXT_SESSION.md:52` | synthetic near/far smoke check on the two novelty scorers | no reported number |
