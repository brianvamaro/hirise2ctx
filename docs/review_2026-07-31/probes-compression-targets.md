# Review area: probes-compression-targets

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-02
- **Verification:** self-refuted (single-agent pass; not independently verified). Every number below
  was **recomputed from the committed/on-disk artifacts the docs name** — the artifact paths are
  quoted so a verifier can re-run the same three snippets.

Scope: the 17 probes listed for this area in `_prompts_probes.md` §2. Triage was by citation first
(`DECISIONS.md`, `PLAN_*.md`, `docs/`, `PROMOTION_QUEUE.md`, `README.md`, `reports/figures/`,
`notebooks/_build_*.py`), then a statistic-level audit of the load-bearing ones.

---

## Findings

### probes-compression-targets-1 — The `boulder_count` target win (+22 % dev PR-AUC, and the W0 **P2 promotion** at +0.146/+0.162) is a change in the *positive-class definition*, not in model skill: rescored on one common positive class the effect is **+0.004 (p = 0.58) on dev and −0.013 (p = 0.11) on full-v2 LOIO**

- **Severity:** high
- **Liveness:** dead-closed for the shipped map (the frozen FM recipe reverted to `fa_gt_1e-2` @ S=32,
  `models/deployable/86c51a5dca220f63/recipe.json`) — but **live for the record**: the wrong number is
  the *promoted W0 baseline* `DECISIONS.md:2470` explicitly calls "the W0 baseline all later work
  compares against", it is `README.md:68`'s shipped item, and it drives a published mechanism claim.
- **Confidence:** high (reproduced from four independent artifacts, including a within-record control
  the project itself ran and mis-read)
- **Where:** `scripts/probes/_sweep_target_reformulation.py:66-79` (`_meaningful_threshold`),
  `scripts/probes/_sweep_w0.py:55-63` (same convention), `scripts/probes/_sweep_w0.py:133-140`
  (`_MD_ROWS`, the reported metric set); consumers `DECISIONS.md:1464-1483`, `DECISIONS.md:2465-2506`,
  `DECISIONS.md:2607-2612`, `PROMOTION_QUEUE.md:306-345`, `docs/modeling_results.md:1213-1223`,
  `notebooks/_build_12.py:790-820` (§9.1, "the headline result of the session"),
  `reports/figures/12_target_reformulation.png` (committed) via
  `scripts/probes/_diag_target_reformulation_figure.py:1-9`.

Both sweeps score the `fractional_area` arm against `fa > 1e-2` and the `boulder_count` arm against a
hardcoded `boulder_count > 50`. Those are **different positive classes with different prevalence**, and
every metric the comparison is decided on (`pr_auc`, `normalised_lift_meaningful`,
`precision_at_top_5pct`) has its no-skill floor *at* the base rate — so a prevalence change alone moves
all three. The one base-rate-invariant metric in the table, meaningful ROC-AUC, is flat or negative in
every run. `src/modeling/evaluate.py:361-366` already emits `meaningful_base_rate` per fold, so the
confound was measurable from the artifacts the whole time.

- **Failure scenario:** `DECISIONS.md:2497-2506` records **"P2 (target = boulder_count): PROMOTED …
  PR-AUC +0.162 (win rate 89 %, p < 1e-4), precision@top-5 % +0.182 … The +22 % dev win carried and
  grew"**, and `DECISIONS.md:2468-2472` then makes `lightgbm_two_stage_balanced × boulder_count @ S=64`
  the banked comparison baseline for W1/W2 and all of `PROMOTION_QUEUE.md` Part B. Rescoring both arms
  of the **banked** baseline (`models/_sweep_w0/20260611T054855Z`) on the identical `fa > 1e-2` class
  gives PR-AUC **−0.0129** (win rate 0.32, Wilcoxon p = 0.107). Every Stage-6 acceptance criterion
  phrased as "+≥0.02 PR-AUC over the P1+P2 baseline" (e.g. `PROMOTION_QUEUE.md:940`) is therefore
  measured against a baseline that is ~0.14 PR-AUC too high for prevalence reasons, and
  `docs/modeling_results.md:1220-1223` states a *mechanism* — "CTX texture features respond to *count*
  of texture events more than to *total area*" — that the corrected numbers do not support.
- **Evidence:**
  ```python
  # scripts/probes/_sweep_target_reformulation.py:66-79
  def _meaningful_threshold(target_col: str, scale_idx: int) -> float:
      tile_size_px = SCALE_TILE_PX[scale_idx]
      tile_area = (CTX_M * tile_size_px) ** 2
      if target_col == "fractional_area":
          return 1e-2
      if target_col == "boulder_count":
          return 50.0                            # <- flat, scale-independent
      ...
      if target_col == "boulder_area":
          return 0.01 * tile_area                # fa=0.01 equivalent in m^2  (EXACT)
  ```
  ```python
  # scripts/probes/_sweep_w0.py:55-63  -- same convention, carried to full-v2 LOIO
  """Operational boulder-rich cut. Flat across scales by convention
  (matches _sweep_stage6a.py; the fa=0.01-equivalent count at S=32 would be
  12.5, but cross-sweep comparability wins over per-scale remapping)."""
  ```

  **(a) The project's own control, mis-read.** In `models/_sweep_target_reformulation/20260530T154730Z`
  (the 2026-05-30 follow-up quoted at `docs/modeling_results.md:1218`), `boulder_area` and
  `log_boulder_area` use `0.01 × tile_area`, which is `fa > 0.01` **exactly**. Measured base rates and
  positives at S=64 (`summary.parquet`):

  | target | threshold | mean base rate | positives | PR-AUC |
  |---|---:|---:|---:|---:|
  | `fractional_area` | 0.01 | 0.3718 | 1,937 | 0.5263 |
  | `boulder_area` | 1024.0 m² | **0.3718** | **1,937** | 0.5306 |
  | `log_boulder_area` | log1p(1024) | **0.3718** | **1,937** | 0.5251 |
  | `boulder_count` | **50** | **0.4984** | **2,592** | **0.6396** |

  The three targets that share a positive set tie within ±0.005. The only target that "wins" is the
  only one whose positive set changed. The doc reads this as "the win is *specific to count of
  distinct detection events*".

  **(b) Direction tracks prevalence, including the sign flip at S=32** (dev sweep
  `20260529T221912Z`, paired by `fold_idx`):

  | | Δ base rate | Δ PR-AUC | Δ norm. lift | Δ prec@5 % | Δ **meaningful ROC-AUC** | Δ Spearman |
  |---|---:|---:|---:|---:|---:|---:|
  | S=64 (n=17) | **+0.149** | +0.113 | +0.130 | +0.111 | **−0.016** | +0.001 |
  | S=32 (n=15) | **−0.062** | −0.064 | −0.063 | −0.054 | +0.014 | +0.016 |

  Measured as **excess over the random-ranker floor** (= base rate) the S=64 deltas become −0.036 /
  −0.019 / −0.038, with `boulder_count` winning only 35 % / 41 % / 24 % of folds.

  **(c) Full-v2 LOIO, S=64 — the promotion itself.** I first reproduced the published deltas exactly
  from `models/_sweep_w0/20260610T221932Z` (PR-AUC +0.1616, win 0.89, p = 3.5e-8; prec@5 % +0.1822,
  win 0.81, p = 1.5e-5 — matching `DECISIONS.md:2498-2500`), alongside Δ base rate **+0.1714**
  (0.3389 → 0.5103; 13,183 → 17,797 positives of 37,315 tiles, win 0.97, p = 7.3e-11) and Δ meaningful
  ROC-AUC **−0.0002 (p = 0.70)**. Then I joined both arms' `predictions.parquet` to
  `dataset_v2/packaged/loio_nfold/all.parquet` on `(obs_id, scale_idx, ti, tj)` and rescored both on the
  **same** `fa > 1e-2` class (base rate 0.3397, 37 paired folds):

  | metric, common positive class | `fractional_area` arm | `boulder_count` arm | paired Δ | win | Wilcoxon p |
  |---|---:|---:|---:|---:|---:|
  | PR-AUC | 0.4240 | 0.4111 | **−0.0129** | 0.32 | 0.107 |
  | ROC-AUC | 0.6638 | 0.6242 | −0.0396 | 0.38 | 0.177 |
  | precision@5 % | 0.4441 | 0.4287 | −0.0154 | 0.22 | 0.107 |

  (The on-disk `models/lightgbm_two_stage_balanced/{96b5c61ca48a7edb,8c7523615964f5cb}/scale_S64_target_*`
  artifacts are the post-coreg-fix **banked** run `20260611T054855Z`, per `DECISIONS.md:2587`; its
  published deltas are +0.146/+0.147, `DECISIONS.md:2607-2610`.) Same test on the dev sweep at S=64:
  PR-AUC **+0.0036 (p = 0.58)**, prec@5 % +0.0125 (win 0.12, p = 1.0) — i.e. **3 % of the reported
  +0.113**.

  **(d) The tell was quoted as the confirmation.** `notebooks/_build_12.py:813` and
  `DECISIONS.md:1478-1481` present "PR-AUC/lift/precision up 20–27 % *while Spearman and ROC-AUC are
  essentially unchanged*" as "the H1 framework's prediction confirmed end-to-end". Rank-invariance plus
  prevalence-sensitivity moving together is the signature of a base-rate change, not of a metric
  finally seeing a real gain. The "+0.008 AUC" quoted there is the **presence** AUC (0.556 → 0.564);
  the mandated meaningful AUC moved −0.016.
- **Self-refutation attempted:**
  (i) *"DECISIONS already caveats it."* `DECISIONS.md:2501-2506` does say the thresholds were
  "designed equivalent (50 boulders ~ 1 % area at S=64) but not identical positive sets". That records
  the *convention* but asserts the equivalence holds; measured, it is 35 % more positives, and the
  caveat does not stop the verdict "PROMOTED … the +22 % dev win carried and grew" from standing, nor
  the mechanism claim in `docs/modeling_results.md:1220-1223`. A caveat that says "not identical" while
  the non-identity is 100 % of the effect is not a mitigation.
  (ii) *"Maybe `boulder_count` really is a better training target and my rescoring is unfair."* The
  rescoring is exactly the deployment question (rank tiles for boulder-richness) and both arms are
  ranked on their own raw output, which is scale-free for AUC/PR-AUC/precision@k; the fold sets and
  tile sets are identical. It comes out null-to-negative on both cohorts.
  (iii) *"Is `normalised_lift` not already the fix?"* No — `src/modeling/evaluate.py:271-285` normalises
  the *ceiling* (`1/base_rate`); its floor is still the base rate, so it rises with prevalence. Its
  excess-over-floor is −0.019 at S=64.
  (iv) *"Is this just `notebooks-5` again?"* No. `notebooks-5` is the raw `lift@top-K` comparison of
  `bc_ge_1` vs `fa_gt_1e-2` **binary classifier labels** in `models/_sweep_binary/20260529T075754Z`
  (notebook 12 §6.1). This is the **regression target-reformulation** sweep, a different probe, a
  different artifact, a different claim, and it reaches a promotion decision. Same failure class, and
  the two together make it systemic.
  (v) *"Does it change the shipped map?"* No — the frozen recipe is `fa_gt_1e-2` @ S=32, and the FM
  freeze cells each compare against a `tier1_ref` at the **same** `pos_rate` (verified in
  `models/fang_probe/*/verdict.json`), so the freeze does not inherit the confound. That is why this is
  `high` and not `blocker`.
- **Fix:** (1) In both probes, score every target arm on **one** positive class — the canonical
  `fa > 1e-2` — by joining the labels parquet, exactly as done above; keep the own-target threshold only
  as a secondary diagnostic. (2) Emit `meaningful_base_rate` into the printed/`result.md` table so any
  future cross-arm comparison shows the prevalence beside the metric (this closes R26 for these probes
  too). (3) Correct the record: `DECISIONS.md:2497-2506` (P2 verdict), `DECISIONS.md:2607-2612`
  (re-check), `DECISIONS.md:1470-1483`, `PROMOTION_QUEUE.md` P2, `docs/modeling_results.md:1218-1223`
  (retract the count-vs-area mechanism claim), notebook 12 §9.1, and re-state any Stage-6 acceptance
  delta that was measured against the `boulder_count` baseline.

---

### probes-compression-targets-2 — The "Bottom line" of the reader-facing `docs/modeling_results.md` rests on a sign test that counts 12 correlated re-analyses of the *same 8 images* as 12 independent observations; at the honest unit the same data gives 5/8 (p = 0.36) and 4/8 (p = 0.64)

- **Severity:** high (record correctness on the most externally-visible modelling document)
- **Liveness:** live-shipped document — `README.md:45-46` and `docs/index.md:34` route readers here;
  `docs/modeling.md:530-534` restates it in "10.1 Principal findings"
- **Confidence:** high (all four published p-values reproduced to 4 dp from the named artifact)
- **Where:** `scripts/probes/_summarize_modeling_results.py:66-90` (the sign tests),
  `:60-64` and `:96-99` (the presence-AUC surfaces); consumers `docs/modeling_results.md:57-62`
  (Bottom line), `:161-181` (§2.1 table), `docs/modeling.md:442-449`, `docs/modeling.md:530-534`

The probe takes the 12 `(variant × scale)` rows of a single sweep's `aggregate.parquet` and runs
`binomtest(n_above, 12, 0.5)` on them. The 12 rows are 3 LightGBM variants × 4 **nested** tile scales,
all fit on the *same 8 LOIO folds of the same 9 images*, so they are not 12 draws from anything — they
are one sample re-analysed 12 times. Measured on the exact artifact the doc used
(`models/_sweep/20260524T071830Z/summary.parquet`), the mean pairwise correlation of the per-fold
values across the 12 configurations is **0.721** for presence AUC and **0.566** for Spearman
(design effect 8.9 and 7.2 → n_eff ≈ **1.3** and **1.7**, not 12).

Compounding it, the quantity carrying the headline p-value is **presence AUC** (invariant 8) — here not
merely reported but escalated into the document's lead statistical claim.

- **Failure scenario:** `docs/modeling_results.md:57-62` opens the document with *"every model has a
  presence-AUC above 0.5 (sign-test p = 0.0002) and ten of twelve have a positive Spearman ρ (sign-test
  p = 0.019)"*, and `:176-181` calls the configuration-level tests **"decisive"** while dismissing the
  more-nearly-independent fold-level test (56/96, p = 0.063) as "on the edge". `docs/modeling.md:530-534`
  then hardens it into a Principal Finding: *"confirmed by an **independent** sign test across 12 v1
  configurations (p = 0.019 on ρ > 0)"*. At the unit LOIO exists to respect — the held-out image —
  the same 12 configurations averaged per image give:

  | quantity | published (config-level) | image-level (n = 8) | image-level sign test | Wilcoxon |
  |---|---|---|---|---|
  | presence AUC > 0.5 | 12/12, **p = 0.0002** | **5/8** | p = 0.363 | p = 0.230 |
  | Spearman ρ > 0 | 10/12, **p = 0.019** | **4/8** | p = 0.637 | p = 0.320 |

  Per-image mean ρ across the 12 configs: `[-0.019, -0.009, 0.128, 0.016, 0.028, -0.036, -0.006, 0.026]`
  — one image (ESP_056165_2200, mean AUC 0.661) carries the result. A reader is told "it is very
  unlikely that all twelve independent models would land above chance by random fluctuation alone"
  when the honest restatement is a coin flip.
- **Evidence:**
  ```python
  # scripts/probes/_summarize_modeling_results.py:66-80
  rho_means = agg["spearman_rho_mean"].to_numpy()
  auc_means = agg["presence_auc_mean"].to_numpy()
  n_pos_rho = int((rho_means > 0).sum())
  n_pos_auc = int((auc_means > 0.5).sum())
  n = len(agg)
  p_rho = binomtest(n_pos_rho, n, 0.5, alternative="greater").pvalue
  p_auc = binomtest(n_pos_auc, n, 0.5, alternative="greater").pvalue
  ```
  Reproduction from `models/_sweep/20260524T071830Z` (8 real folds, all 12 cells): 10/12 → p = 0.0193
  (doc: 0.019); 12/12 → p = 0.000244 (doc: 0.0002); 56/96 → p = 0.0627 (doc: 0.063); mean ρ = +0.0161
  (doc: +0.016); mean AUC = +0.5261 (doc: +0.526). Exact match — the probe is unambiguously the producer.
- **Self-refutation attempted:**
  (i) *"The doc discloses the assumption."* `docs/modeling_results.md:161-164` does say "Treating each
  of the 12 … configurations as one independent observation". But three lines later it asserts they
  *are* independent and calls the result decisive, and `docs/modeling.md:532` drops the hedge entirely.
  Disclosure of an assumption that is measurably false by a factor of ~9 in effective n is not a fix.
  (ii) *"`docs/modeling.md:443-449` already downgrades the presence-AUC version."* It does — but it
  keeps the ρ version as "the load-bearing" one, and that version is 4/8 (p = 0.64) at the image level.
  The Bottom line of `modeling_results.md`, which has no such hedge, leads with the presence-AUC one.
  (iii) *"Is the conclusion actually wrong?"* Probably not — v2 and the FM recipe later established
  real skill on much better evidence. The defect is that the *stated* evidence does not support the
  claim, in a document written for a general scientific reader and pointed at by the README.
  (iv) *"Is this already covered by `stats-fallacies`?"* No — that area audited `evaluate.py`,
  `fgates.py`, `leveling.py`, `stage7d_pooled.py` and the F/striping PLANs. Grepping
  `docs/review_2026-07-31/*.md` for "sign test"/"12 config" returns only Stage-7d and F-gate items.
- **Fix:** Replace the configuration-level sign test with a test at the LOIO image level (sign test or
  Wilcoxon over the 8 per-image means, as above), report `n_eff` or the pairwise correlation alongside
  if the configuration view is kept at all, and drop the presence-AUC version entirely per invariant 8.
  Update `docs/modeling_results.md:57-62`, `:161-181` and `docs/modeling.md:442-449`, `:530-534`.

---

### probes-compression-targets-3 — "Isotonic recalibration drops Spearman 0.169 → 0.157 and AUC 0.579 → 0.572" is a mathematically pinned outcome (a monotone map cannot raise a rank metric) and the "AUC" is presence AUC computed inside the probe

- **Severity:** medium
- **Liveness:** dead-closed decision (it motivated the four two-stage variants, later found NULL at
  LOIO), but the numbers are still in `DECISIONS.md` and in a reader-facing doc section, and the
  project **later shipped isotonic calibration** (`PLAN_Calibration.md` Stage 0/1) without reconciling
  this entry
- **Confidence:** high
- **Where:** `scripts/probes/_diag_compression_mechanism.py:58-72` (`isotonic_oof_recalibrate`),
  `:149-154` (`auc()` = presence AUC), `:156-172` (the per-fold raw-vs-iso comparison),
  `scripts/probes/_diag_compression_mechanism.md:19-22` (the banked output); consumers
  `DECISIONS.md:1396-1400`, `docs/modeling_results.md:1105-1135`, committed figure
  `reports/figures/12_compression_diagnostic.png`, notebook 12 §3

`IsotonicRegression(out_of_bounds="clip", increasing=True)` is a non-decreasing step function applied
**within** each fold, and the comparison metrics are computed **per fold**. Any non-decreasing map can
only merge distinct scores into ties; AUC is a rank statistic, so under such a coarsening it can only
move toward 0.5 — it is arithmetically impossible for `auc_iso > auc_raw` when `auc_raw > 0.5`. The
same holds for Spearman up to pathological re-orderings of the merged groups. So the two numbers
offered as evidence *against* isotonic recalibration were guaranteed non-positive before the probe ran;
they measure how much resolution the step function destroyed, not whether recalibration helps. This is
the register's most productive pattern (R36 / R11 / R43 / `leakage-3`) in mirror image: a comparison
whose sign was fixed by construction.

- **Failure scenario:** `DECISIONS.md:1396-1400` records *"Post-hoc isotonic recalibration does NOT fix
  it … and **drops** mean Spearman 0.169 → 0.157 and AUC 0.579 → 0.572 — the raw predictions don't span
  enough range to be re-stretched, and out-of-range clipping at fold boundaries breaks ranking.
  **Compression must be fixed in training.**"* The causal explanation attributes a tie-generation
  artifact to a substantive property of the predictions, and the decision it licensed — build four new
  training-time variants — produced `balanced`, which `DECISIONS.md:2504` later found **NULL at LOIO**.
  The only non-pinned number in the entry is the high-bin ratio (0.4197 → 0.4814), which is a genuine
  measurement and does support "isotonic does not fix the level".
- **Evidence:**
  ```python
  # scripts/probes/_diag_compression_mechanism.py:68-71
  iso = IsotonicRegression(out_of_bounds="clip", increasing=True)   # monotone by construction
  iso.fit(df.loc[train_mask, pred_col].to_numpy(),
          df.loc[train_mask, target_col].to_numpy())
  cal[test_mask.to_numpy()] = iso.predict(df.loc[test_mask, pred_col].to_numpy())

  # :149-154 -- presence AUC under another name; this is the "AUC" quoted in DECISIONS
  def auc(y, p):
      from sklearn.metrics import roc_auc_score
      yb = (y > 0).astype(int)
      ...
      return roc_auc_score(yb, p)
  ```
  ```
  # scripts/probes/_diag_compression_mechanism.md (banked output, per-fold LOIO headline)
        spearman_raw  spearman_iso  auc_raw  auc_iso
  mean        0.1689        0.1568   0.5792   0.5716
  ```
- **Self-refutation attempted:** (i) *Is the map really monotone per fold?* Yes — one
  `IsotonicRegression(increasing=True)` is fit per held-out fold and applied only to that fold's rows
  (`:65-71`), and the metrics are computed per fold (`:158-169`), so no cross-fold non-monotonicity can
  enter. (ii) *Could out-of-bounds clipping break monotonicity?* No — `"clip"` is still non-decreasing;
  it just adds ties at the extremes, which is precisely the mechanism. (iii) *Is the probe's LOIO
  protocol dishonest?* No — it fits on the other folds' OOF predictions and applies to the held-out
  fold, which is correct and matches the protocol `calibration` verified clean. (iv) *Is the AUC
  finding just R02?* R02 is `src/modeling/evaluate.py`'s `presence_auc` surface. This is a *probe* that
  reimplements it and whose value is quoted verbatim in `DECISIONS.md` and `docs/modeling_results.md`
  as "AUC" — the "presence AUC under another name" pattern the brief asks for more of.
- **Fix:** Delete the Spearman/AUC row from the isotonic comparison (or state explicitly that a
  monotone recalibration cannot improve a rank metric and that the delta bounds the tie-loss), keep the
  per-bin ratio as the real evidence, and amend `DECISIONS.md:1396-1400` accordingly — noting that the
  later `PLAN_Calibration` Stage-0/1 decision to ship isotonic for Tier-1 is not in fact contradicted
  by this entry. Rename the probe's `auc()` to `presence_auc_diagnostic` or replace it with the
  meaningful-threshold AUC.

---

## Refuted by my own check

- **`_pick_binary_thresholds.py` chose a threshold on the test fold (the leakage channel the brief
  flags).** No. It pools **all** labels (`:28-35`) with no fold structure, but the rule the decision
  adopted is `fractional_area > 0` / `boulder_count >= 1` — data-independent presence rules. The
  data-derived `match_*_to_target` candidates at `:83-109` were explicitly *rejected*
  (`DECISIONS.md:951`). No tuned threshold reaches any model.
- **`_w0_paired_deltas.py` deltas are not paired on the same folds.** They are — `paired()` joins on
  `held_out_obs_id` and intersects (`:36-39`), and on `loio_nfold` that key is unique per fold. Latent
  only: on `within_image_4fold` the key repeats 4×, so `set_index` + `.loc[common]` would misalign; the
  probe is never invoked that way (`DECISIONS.md:2457`, `:2607`).
- **`_summarize_modeling_results.py` reads `presence_auc_mean` (the brief's explicit question).**
  It does (`:63, :68, :78, :80, :98, :178`) — but that is R02's surface, so it is folded into finding 2
  rather than re-filed as its own item.
- **`_summarize_modeling_results.py`'s artifact resolution has produced a wrong published number
  (the `notebooks-1` glob bug).** Not currently. `:40-49` picks the most-recent sweep with `len(agg)
  >= 12` and `:113-120`/`:137-146` use `st_mtime` globs, so re-running today would silently select the
  **v2** 16-row sweep `20260529T061553Z` (whose 4 `lightgbm_classification` rows have
  `presence_auc = 0.500` exactly, giving 12/16 and p = 0.038, not 12/12 p = 0.0002), and `:195-197`
  picks `sorted(...)[-1]` by *config_hash* while its header hardcodes "across all 9 folds". But the
  published §2.2 feature-importance shares (16.1 % at S=8) are the v1 values per `notebooks-1`, so no
  live number is wrong — this is a reproducibility hazard, not a defect. Worth one line in the probe.
- **`_sweep_target_reformulation.py`'s boulder-count bin labels are off by a decade** — `:181-187`
  labels the `(0, 1]` bin `"1_to_10"`, `(1, 10]` `"10_to_100"`, etc. Real, but grepping
  `docs/`, `DECISIONS.md`, `PROMOTION_QUEUE.md` and `notebooks/_build_12.py` for `1_to_10` /
  `100_to_1k` / `1k_plus` returns **zero** hits: no doc quotes a count-bin ratio. Low, uncited.
- **`_sweep_compression_fixes.py` carries the same cross-target confound as finding 1.** It does not —
  it holds `TARGET_COL = "fractional_area"` fixed (`:52`) and varies only the variant, so its
  comparison is prevalence-matched. Its "+0.018 AUC" is presence AUC (R02's key), and the win was
  later found NULL at LOIO (`DECISIONS.md:2504`), but the probe's comparison is sound.

## Verified clean

- **`_pick_binary_thresholds.py`** — recomputed from `dataset/labels/*.parquet` (643,910 rows, 9 files):
  Jaccard(`fa > 0`, `bc >= 1`) = **0.903 / 0.960 / 0.984 / 0.993** at S = 8/16/32/64, matching
  `DECISIONS.md:951`'s "90–99 % Jaccard agreement". The per-scale counts in `DECISIONS.md:958-963`
  drift by ≤ 26 tiles (10,331 vs 10,305 at S=8, etc.) from a later label regeneration — immaterial to
  the decision, which was `fa > 0`. `config.yaml:76`'s comment is accurate.
- **`_diag_target_dist_v1v2.py`** — the table published at `docs/modeling_results.md:906-914` reproduces
  exactly from `dataset/packaged/loio_9fold/all.parquet` and `dataset_v2/packaged/loio_nfold/all.parquet`:
  zero-frac 0.9789/0.9428/0.8692/0.7202 (v1) and 0.5020/0.3282/0.1801/0.0691 (v2); frac > 1e-2
  0.0050/0.0034/0.0025/0.0017 and 0.3277/0.3522/0.3598/0.3543; 3.56 M vs 0.64 M tiles. Both parquets
  have **zero** duplicate `(obs_id, scale_idx, ti, tj)` keys, so the doc's "each tile appears once" is
  correct. Only drift: the v2 S=8 max is 0.4362, published as 0.429.
- **The FM freeze does not inherit finding 1's confound** — every
  `models/fang_probe/fw_emb_mlp_ens3_*/verdict.json` compares the FM cell against a `tier1_ref` at the
  **identical** `pos_rate` (e.g. 0.3598 for `fa_gt_1e-2 @ S=32`, 0.4828 for `bc_ge_50 @ S=64`), and the
  frozen cell is *not* the pooled-PR-AUC maximum across cells (0.7832 vs `bc_ge_1`'s 0.9393), so the
  freeze was not selected on the prevalence-inflated axis. Left to `probes-fm-recipe` to confirm.
- **`_diag_compression_mechanism.py`'s source pinning** — `:36` hardcodes
  `models/lightgbm_two_stage/629276139c22da68/scale_S64`, whose `snapshot.json` is
  `dataset_v2 / loio_nfold / S=64 / fractional_area`, i.e. exactly the "v2 two_stage S=64" the docs
  claim. No mtime glob, no version ambiguity. Its LOIO isotonic protocol (fit on other folds' OOF,
  apply to the held-out fold) is honest.
- **`_sweep_target_reformulation.py` / `_sweep_w0.py` monkeypatch of `per_fold_metrics`** — restored in
  a `finally` (`:154-155` / `:124-125`), so no threshold leaks into a later cell of the same process.
- **`_diag_compression_variants_smoke.py`** — synthetic-data constructor/fit/predict smoke test; emits
  no reported number.

## Coverage note

**Read in full:** `_pick_binary_thresholds.py`, `_w0_paired_deltas.py`, `_sweep_target_reformulation.py`,
`_sweep_w0.py`, `_diag_compression_mechanism.py`, `_summarize_modeling_results.py`,
`_diag_target_dist_v1v2.py`, plus `src/modeling/evaluate.py:240-430` (the metric definitions all these
probes call). **Read the first 60–120 lines / headers of:** `_sweep_compression_fixes.py`,
`_diag_compression_sweep_table.py`, `_diag_compression_sweep_figure.py`,
`_diag_target_reformulation_figure.py`, `_diag_compression_variants_smoke.py`,
`_diag_v2_binary_per_image.py`, `_diag_v2_binary_thresholds.py`, `_summarize_binary_results.py`,
`_diag_per_image_breakdown.py`, `_diag_topk_confusion_map.py`.

**Reproduced numerically** (pandas over on-disk artifacts, no probe executed): the H2 dev sweep
(`models/_sweep_target_reformulation/{20260529T221912Z,20260530T154730Z}`), the W0 matrix
(`models/_sweep_w0/{20260610T221932Z,20260611T054855Z}`) including a full re-scoring of both target arms
against a common positive class via `dataset_v2/packaged/loio_nfold/all.parquet` and
`dataset_v2_dev/packaged/within_image_4fold/all.parquet`, the v1 sign tests
(`models/_sweep/20260524T071830Z`), the v1/v2 target distributions, and the v1 binary-threshold Jaccards.

**Not checked, and why:**
- **`_summarize_binary_results.py`** (135 lines) — writes the "Binary reframing" section of
  `docs/modeling_results.md` §6 (mean paired Δ +0.003 AUC, p = 1.00 on 32 paired folds; "7 of 12 cells
  above AUC 0.5"). Those are the *same* 12-cell sign-test construction as finding 2 applied to the
  binary sweep, so the same objection presumably applies, but I did not reproduce them. **This is the
  single largest gap in this area.**
- **`_diag_per_image_breakdown.py`** (240 lines) — writes `reports/figures/13_per_image_performance.png`
  and a banked `.md`, and feeds notebook 13's "which images worked" catalogue and the anti-signal
  taxonomy quoted at `docs/modeling_results.md:1235-1245`. Cited in `PROMOTION_QUEUE.md:934` only as a
  capability reference ("we already extracted SeamMap.shp"). Its per-image correlations are per-image
  aggregates of spatially autocorrelated data and would be worth a `stats-fallacies`-style pass.
- **`_diag_topk_confusion_map.py`** — renders a qualitative TP/FP/FN overlay for notebook 13; touches
  CTX imagery, so it was not run and its figure is not committed. No number.
- I did **not** re-derive `mean boulder area` to test whether `boulder_count > 50` was ever the
  fa = 0.01 equivalent at S=64 in principle; the empirical base rates settle the question either way.
- Per the rules of engagement I did not execute any probe, touch imagery or the network, or run
  training. All artifacts I read under `models/` are **untracked** (present locally, not committed), so
  a verifier on a fresh clone cannot reproduce findings 1–2 without regenerating the sweeps — itself
  worth noting: the numbers in `DECISIONS.md`, `PROMOTION_QUEUE.md` and `docs/modeling_results.md` have
  no committed provenance (compare **R12**).

## Load-bearing map

| probe | cited by | number it produced | verdict |
|---|---|---|---|
| `_sweep_target_reformulation.py` | `DECISIONS.md:1464-1483`; `PROMOTION_QUEUE.md:306,327,342`; `docs/modeling_results.md:1214-1223`; notebook 12 §9; `reports/figures/12_target_reformulation.png` | "+22 % PR-AUC / +27 % lift / +20 % prec@5 % from `boulder_count`" @ S=64 | **WRONG — finding 1** (+0.004, p = 0.58 on a common positive class) |
| `_sweep_w0.py` | `DECISIONS.md:2454-2530, 2587, 2607-2612, 2757, 2781, 2817, 2879, 2927`; `PROMOTION_QUEUE.md:11` | W0 baseline recipe; **"P2 PROMOTED"** at PR-AUC +0.162 / re-check +0.146, p < 1e-4 | **WRONG — finding 1** (−0.013, p = 0.11 on a common positive class) |
| `_w0_paired_deltas.py` | `DECISIONS.md:2457, 2607` | the paired Wilcoxon p's behind P1 / P2 / hurdle | pairing itself correct; **P2's inputs are confounded** (finding 1) |
| `_summarize_modeling_results.py` | (unattributed) `docs/modeling_results.md:57-62, 161-181`; `docs/modeling.md:442-449, 530-534` | "12/12 presence AUC > 0.5, p = 0.0002"; "10/12 ρ > 0, p = 0.019"; 56/96, p = 0.063; mean ρ +0.016 | reproduces exactly; **statistically invalid — finding 2** |
| `_diag_compression_mechanism.py` | `DECISIONS.md:1391-1400`; `docs/modeling_results.md:1105-1135`; `reports/figures/12_compression_diagnostic.png`; notebook 12 §3 | p_pos ≈ 0.85 on true zeros; high-bin ratio 0.42 → 0.48; "Spearman 0.169 → 0.157, AUC 0.579 → 0.572" | mechanism + bin ratios sound; **rank-metric evidence pinned + presence AUC — finding 3** |
| `_sweep_compression_fixes.py` | `DECISIONS.md:1409-1425`; `PROMOTION_QUEUE.md:272`; `docs/modeling_results.md:1145`; notebook 12 §5 | "`balanced` +0.017 ρ / +0.018 AUC at S=64"; the 5-variant bin-ratio table | prevalence-matched (target held fixed) — sound; the "+0.018 AUC" is presence AUC; later NULL at LOIO |
| `_diag_target_reformulation_figure.py` | committed `reports/figures/12_target_reformulation.png`, notebook 12 §9 | the published bar chart of PR-AUC / lift / prec@5 % by target | **renders finding 1's confound**; its docstring states the tell ("without an AUC change") as the claim |
| `_diag_compression_sweep_figure.py` / `_diag_compression_sweep_table.py` | committed `reports/figures/12_compression_fix_sweep.png`, notebook 12 §5 | per-bin `mean_pred` / ratio table; figure legends label presence AUC as "AUC" | content sound; both resolve `runs[-1]` (latest sweep) — reproducibility hazard only |
| `_pick_binary_thresholds.py` | `DECISIONS.md:951, 953-1021`; `config.yaml:76` | Jaccard 0.903/0.960/0.984/0.993 ("90–99 %"); per-scale positive counts; `binary_count_threshold 5 → 1` | **REPRODUCED — clean** (counts drift ≤ 26 tiles from label regeneration) |
| `_diag_target_dist_v1v2.py` | `docs/modeling_results.md:903-915`; `PLAN_ModelImprovement.md:165` | v1-vs-v2 zero-fraction and frac > 1e-2 table | **REPRODUCED — clean** (only v2 S=8 max drifts, 0.436 vs 0.429) |
| `_diag_per_image_breakdown.py` | `PROMOTION_QUEUE.md:934`; notebook 13 §3.2; writes `reports/figures/13_per_image_performance.png` + a banked `.md` | per-image performance ranking + metadata correlations | **not audited** (see coverage) |
| `_diag_v2_binary_per_image.py` | `notebooks/_build_12.py` §6 | per-image AUC bimodality at `fa_gt_1e-2` (median 0.61 / max 0.91 / lift 9.1× at 1.3 % base rate) | source sweep is **pinned** (`_sweep_binary/20260529T075754Z`); the cross-target reading is `notebooks-5` |
| `_diag_topk_confusion_map.py` | `notebooks/_build_13.py` | qualitative TP/FP/FN map overlay | no number; figure not committed |
| `_summarize_binary_results.py` | (unattributed) `docs/modeling_results.md` §6 | "+0.003 mean paired Δ AUC, p = 1.00 on 32 folds"; "7 of 12 cells above AUC 0.5, p = 0.39" | **not audited** — same 12-cell construction as finding 2; largest remaining gap |
| `_diag_v2_binary_thresholds.py` | — | threshold-comparison readout off the pinned binary sweep | not cited, writes nothing |
| `_diag_compression_variants_smoke.py` | — | synthetic constructor/fit/predict smoke test | not cited, writes nothing |
</content>
</invoke>
