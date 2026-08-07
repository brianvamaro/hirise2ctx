# PLAN_Calibration — de-compressing the boulder-abundance outputs

**Created 2026-06-14 (Brian-approved direction); expanded same day to a full
solution-space treatment at Brian's request** ("a full consideration of how to fix
this including new ways of training or anything else — this is an important problem
for us to tackle"). Operationalizes the calibration half of
[PLAN_ModelUsability.md](PLAN_ModelUsability.md) **W3** and the "calibration layer
is future work" line of [PLAN_FM.md](PLAN_FM.md) item 4. Standalone because the
solution space is large.

Stage 0 (diagnose + post-hoc preview) is **DONE** — `src/calibration.py`,
`tests/test_calibration.py` (8 pass), [notebook 23](notebooks/23_calibration_diagnostic.ipynb).
This plan governs everything past it.

---

## 1. The problem, measured (Stage 0 — DONE)

Group-aware LOIO, banked predictions, `src.calibration` (notebook 23):

**Tier-1 (rich/poor probability) is already well-calibrated** — ECE **0.060**,
probabilities well-spread (std 0.36), reliability near-diagonal, AUC 0.848;
temperature scaling (T≈1.70) trims ECE to 0.049 with AUC unchanged. The "mostly
rich" maps over boulder-rich regions are largely *correct*. **Tier-1 is a minor
refinement, not the problem.**

**Tier-2 (abundance) compresses, two-sided and intrinsically.** The single-stage
`mlp_reg` (identity target + MSE) is textbook regression-to-the-mean:

| | Spearman | top-bin ratio | near-zero pred | marginal-L1 |
|---|---|---|---|---|
| raw `mlp_reg` (emb) | 0.651 | 0.71 | 1.8 % | 0.0057 |
| LightGBM Tweedie | 0.539 | 0.55 | 0.0 % | 0.0088 |
| LightGBM two-stage | 0.589 | 0.59 | 0.3 % | 0.0091 |

(true exact-zero share **18 %**; top-bin ratio = mean_pred/mean_true for true
fa>1e-2; near-zero pred should match 18 %; marginal-L1 = mean |Δquantile|, 0=matched.)
It **over-predicts the low end** (floors ~0.005, never true zero) and
**under-predicts the high tail (~30 %)**, crossing the diagonal near the rich/poor
threshold. The banked structural alternatives compress **more**, not less.

## 2. Why MSE compresses — the theory that organizes every fix

The squared-error-optimal predictor is the conditional mean `E[y|x]`. When the
input cannot determine the target — here, 5 m/px CTX texture only weakly constrains
a meter-scale, label-noisy abundance — the conditional distribution `p(y|x)` is
**wide**, so its mean sits far from the extremes: high-abundance inputs share their
texture with medium ones, so `E[y|x]` for a true-high tile is pulled down; true-zero
tiles share texture with low-positive ones, so their mean is pulled up. Compression
is therefore **not a bug to patch but the Bayes-optimal behaviour of any
mean-seeking point estimator under aleatoric uncertainty on a skewed target.**

That single fact implies the **four — and only four — levers** to fix it. Every
method below is one of these:

> **L1. Stop predicting the mean.** Change the objective so its optimum is a
> quantile, an expectile, or a full distribution — then report a non-mean summary
> (a high quantile, the mode, or the distribution itself), which is not compressed.
>
> **L2. Shrink `p(y|x)`.** Reduce the aleatoric uncertainty — better features,
> coarser tiles, less label noise, more data, a cleaner target — so even the mean
> is less compressed.
>
> **L3. Fix the marginal post-hoc.** Monotonically remap the point predictions so
> their *distribution* matches the truth (quantile-matching). Cheap, ranking-safe,
> fixes aggregate/area statistics — but not per-tile placement.
>
> **L4. Report the distribution, not a point.** Deliver a calibrated interval /
> quantiles per tile. The honest answer to an irreducible floor; not a per-tile
> accuracy claim.

Goals map onto levers: a distributionally-correct **area-integrated** map needs only
L3 (done); a less-compressed **point** map needs L1 and/or L2; an honest **per-tile**
product is L4. L2 is the only lever that raises the ranking ceiling itself.

## 3. The solution space (full catalog)

Ordered within each lever by expected impact-per-cost on *our* problem. Citations
hyperlinked.

### L1 — objectives whose optimum isn't the mean (retraining; Tier-2 is not yet frozen)

L1 is "**what functional of `p(y|x)` does the loss target?**" The arithmetic mean
(plain MSE) is compressed; change the loss — or, equivalently, the target's scale —
to target a less tail-shy functional. **This is where "predict log" and "change the
error function" live** (both are L1; they alter the targeted functional, not the
information in `x`):

- **Target transform** (the lightweight entry, ~one line): train MSE on
  `log1p(y)` / `sqrt(y)` / a Box-Cox–Yeo-Johnson fit, then back-transform. MSE in
  log-space targets ≈ the conditional *median / geometric mean*, which the heavy
  right tail pulls far less than the arithmetic mean. **TESTED 2026-06-15
  (`_diag_tier2_objectives.py`): log1p is a WASH** — raw top-bin 0.66→0.67, per-image
  ρ 0.433→0.445 (within noise); after quantile-matching both reach 0.87. The
  compression here is intrinsic (aleatoric floor), not a target-scale artifact, so
  the cheap transform doesn't move it. Deprioritized; the heavier L1 below is the
  real lever.
- **Regression-as-classification / histogram loss (HL-Gauss).** Bin the target,
  put a (soft, Gaussian-smoothed) label over bins, train cross-entropy, report the
  distribution's mean *or a high quantile or the mode*. Repeatedly beats direct MSE
  for deep regression precisely because it doesn't collapse to a point and is robust
  to heavy tails ([Imani & White 2018](https://arxiv.org/abs/1806.04613);
  [Farebrother et al. 2024, "Stop Regressing"](https://arxiv.org/abs/2403.03950)).
  Was the **top candidate** — directly attacks the compression cause, gives a full
  predictive distribution (feeds L4), small change to the MLP head
  (768→256→64→K-bin softmax). **TESTED 2026-06-15 (`_diag_tier2_l1_bakeoff.py`):
  WASH on ranking.** Its mean readout ties `mlp_reg` per-image (paired Δ −0.017,
  Wilcoxon p≈0.08, i.e. marginally *worse*); mode/P90 readouts rank worse still. The
  P90 does de-compress the tail (raw top_ratio 1.13) but at a ranking cost. Ruled out
  as a ranking lever, alongside the cheap swaps.
- **Quantile / pinball regression.** Predict P10/P50/P90 with the pinball loss
  ([Koenker & Bassett 1978](https://doi.org/10.2307/1913643)). The median is a
  robust, less-compressed point; P90 captures the tail; the spread *is* L4's
  interval. Cheap (multi-output head + pinball). **TESTED 2026-06-15 (bake-off): the
  best of the L1 heads but still a WASH on ranking** — median ties `mlp_reg`
  per-image (paired Δ −0.002, 18/38 wins, Wilcoxon p=0.48). The **one keeper**: its
  **P90 has raw top_ratio 0.98** — a tail-calibrated point *without* the L3 layer, at
  no ranking cost (a useful optional readout). Intervals: [P10,P90] covered only
  **58.6 %** vs nominal 80 % → the head under-estimates its own spread; L4 needs
  interval recalibration before it is honest.
- **Expectile regression** — the asymmetric-MSE analogue of quantiles; a tunable
  knob between mean and tail. Cheaper to optimize than pinball (smooth), worth a
  sweep of the asymmetry parameter.
- **Heteroscedastic / distributional NLL.** Predict a *distribution* per tile and
  train negative log-likelihood:
  - Gaussian (mean, variance) — [Kendall & Gal 2017](https://arxiv.org/abs/1703.04977);
    gives aleatoric uncertainty directly (L4) but a symmetric Gaussian fits a skewed
    zero-inflated target poorly.
  - **Zero-inflated log-normal / Tweedie-NLL** — matches the data-generating process
    (a zero spike + a positive right tail). The LightGBM Tweedie under-performed, but
    a *neural* zero-inflated head on the FM embedding is untried and principled.
    **TESTED 2026-06-15 (bake-off): neural ZILN is a WASH** — mean/median tie or
    slightly trail `mlp_reg` per-image (paired Δ −0.019/−0.025). Its **median is the
    only readout that recovers near-zero mass** (near0 9.9 % vs truth 18 %), but that
    doesn't help ranking; [P10,P90] covered 58.8 % (same under-dispersion as pinball).
  - **Mixture density network** ([Bishop 1994](https://publications.aston.ac.uk/id/eprint/373/))
    — a flexible multi-modal `p(y|x)`; heavier, keep as a fallback.
- **Ordinal regression** (CORAL/CORN, [Cao et al. 2020](https://arxiv.org/abs/2111.08851)).
  Rank-consistent ordered-bin classification; report the expected value or a
  quantile. Less mean-shrinkage than continuous regression and *is* the hazard-class
  product (L→ Serrano). Overlaps HL-Gauss; pick one.

### L2 — shrink the aleatoric uncertainty (raise the ranking ceiling)

This is the only lever that improves the *ranking* (the post-L3 residual), so it is
strategically important even though each item is more work.

- **Coarser operating scale for Tier-2.** Per-tile abundance at S=64 / S=128
  averages over more area → higher SNR, less label-noise per tile → `p(y|x)`
  narrows → less compression. **TESTED 2026-06-15 (`_diag_tier2_scale_sweep.py`,
  S=32→64):** raw top_ratio 0.66→**0.72** (less compressed), pooled rho 0.648→**0.695**,
  per-image rho paired Δmed **+0.025** (25/38 images) — directionally the right way
  but **Wilcoxon p=0.19, not significant at n=38**, and partly an easier-target
  artefact (true zero share 18 %→6.9 % at the coarser tile). So coarsening *probably*
  helps and the Tier-2 *map* can run coarser than the Tier-1 rich/poor map, but a
  confident ranking gain needs the §2.3 expansion. S=128 untested (needs a P384
  embedding pass + a 128-px label grid — out of the cheap scope).
- **Ceiling diagnostic — the wall is the data, not the head** (`_diag_tier1_vs_tier2_ranking`,
  2026-06-15). Tier-1 `P(rich)` — a *classifier* that never saw the continuous target
  — ranks `fractional_area` per-image at **0.437**, statistically identical to the
  dedicated Tier-2 regressor's **0.433** (and counts: 0.436). Within the rich class it
  falls to **0.34** (texture barely resolves *how* rich). Two different model families
  hitting the same ~0.43 wall ⇒ the magnitude signal in 5 m/px CTX ≈ the rich/poor
  signal, with little extra. **Direct test (qmatch both onto the fa marginal):** the
  abundance maps share an identical marginal, but the dedicated regressor keeps a small,
  borderline edge on ranking (pooled 0.642 vs 0.625; paired per-image wins 24/38,
  Wilcoxon p≈0.05). So the **one-model simplification** (drop the Tier-2 head, use a
  quantile-matched `P(rich)`) is **viable at a ~0.02 ranking cost, not free** — a Stage
  1/4 option. Either way, L1/representation tweaks can't beat a ceiling in the inputs.
- **Target choice: counts, with a count likelihood.** Predict `boulder_count` under
  a **Poisson / negative-binomial NLL** (the natural count model). **TESTED
  2026-06-15 (`_diag_tier2_objectives.py`): Poisson-count → area is WORSE for the
  area target** — per-image ρ 0.425 vs 0.433, raw top-bin 0.54, +qmatch only 0.78
  (vs 0.87). The count→area conversion (× mean boulder size) discards the per-tile
  size information that area-fraction needs, so area can't be recovered from counts
  alone. Count-Poisson would only make sense **if the product itself is count-density**
  (Serrano hazard classes), not area-fraction.
- **Reduce label noise.** The `min_confidence` BoulderNet `score` filter
  ([CLAUDE.md §11]). **TESTED 2026-06-16 (`_diag_tier2_minconf_sweep.py`)** →
  **AMENDED 2026-08-06 (R56): the original "HARMFUL, monotonically degrading" verdict was a
  two-factor comparison and does not survive.** The probe trained on labels regenerated at
  `score ≥ t` *and* scored against those same regenerated labels, so its paired Wilcoxon varied
  **both the predictor and the target**. Re-scored on the banked per-tile parquets against **one
  common target** (the unfiltered `fractional_area`; 161,005 tiles × 38 images, keys row-aligned):
  - **conf ≥ 0.5 is a null on every project metric** (paired per-image, n=38): `meaningful_auc`
    Δ −0.0028 (15/38, p=0.31), `pr_auc@1e-2` −0.0007 (17/38, p=0.44), `precision@5%` 0.0000
    (14/38, p=0.41), Spearman −0.0034 (17/38, p=0.43). Min p over nine metrics = 0.178.
    Of the recorded −0.021 Spearman loss, **−0.017 (82 %) is the target moving**, not the model.
    Precisely: a **bounded** null under the project's group-aware equal-per-image weighting
    (95 % CI on mean Δ `meaningful_auc` [−0.0116, +0.0029]) — not a claim of exact zero. The
    **pooled** `precision@5%` does fall −0.0143, which an image-clustered bootstrap distinguishes
    from zero (p ≈ 0.03–0.04); the per-image basis is the one the project reports on.
  - **conf ≥ 0.7 is directionally harmful** — `meaningful_auc` −0.0104 (p=0.014), `pr_auc`
    −0.0068 (p=0.015), `precision@5%` −0.0096 (p=0.018) — but Spearman falls to p=0.061 and
    **no metric survives Holm correction** across the nine (all adj. p ≈ 0.12).
  - **"Degrades dynamic range" is a population artefact plus a missing calibrator.** `top_ratio`
    was read at a fixed `fa > 1e-2` while the filter moved the rich share 36 %→27 %→11 %; at a
    **matched 36 % top fraction** the sequence is 0.664 → 0.623 → 0.519, not 0.664 → 0.578 → 0.314
    (53–59 % of the collapse is the population change). After the **shipped** quantile-match layer,
    against the common target, `top_ratio` is **0.870 → 0.859 → 0.829** — essentially flat.
  - **Not monotone even in the probe's own scorecard:** per-image ρ 0.4333 → **0.4563** → 0.3044.

  What survives: filtering **thins** the target rather than cleaning it (the target factor is real
  and significant on its own), so there is still no evidence filtering *helps*, and keeping
  `min_confidence: null` remains defensible. What does **not** survive: "monotonically degrades",
  "harmful at conf ≥ 0.5", and the causal claim that this proves low-confidence detections are real
  boulders. **This ruling no longer forbids harmonising a confidence floor** — see R23. Note the
  re-score covers 0.5 and 0.7 only; **0.6173 was never measured** and interpolates to 56–70 % of the
  conf070 harm. (Noise-robust losses untried.)
- **Better representation.** (a) Multi-scale embedding fusion (S=16/32/64) for more
  context to disambiguate high tiles; (b) a small **spatial head** over the 3×3
  embedding field (already partly present); (c) **ViT fine-tune** (LoRA/last block)
  — PLAN_FM defers this, but if the frozen embedding undersells the tail it is the
  representation-level fix. Decide after the cheaper L2 items.
- **More data** — the §2.3 expansion cohort adds tail examples and shrinks shrinkage;
  out of this plan's scope but the standing unlock.

### L1+L2 imbalanced-regression machinery (re-weighting, applies to any objective)

- **Label/feature distribution smoothing (LDS/FDS)** — the canonical deep
  imbalanced-regression method ([Yang et al. 2021, "Delving into Deep Imbalanced
  Regression"](https://arxiv.org/abs/2102.09554)): smooth the empirical label density
  and re-weight the loss by its inverse, and smooth features across neighbouring
  target values. **TESTED 2026-06-16 (`_diag_tier2_reweight.py`, LDS label-density
  weights on mlp_reg): DOMINATED, ruled out.** It de-compresses the *raw* marginal
  (top_ratio 0.67→0.77 sqrt → 0.88 inv) but at a **significant paired ranking cost**
  (Δ −0.014 p=0.018 sqrt; −0.032 p=0.015 inv). Quantile-match (L3) recovers the same
  marginal for free with **zero** ranking cost, so reweighting is strictly worse for
  our goal. (FDS feature-smoothing untried; unmotivated given the label-weighting
  result.)
- **Density-based / cost-sensitive weighting** ([Steininger et al. 2021](https://doi.org/10.1007/s10994-021-06023-5)),
  tail oversampling, and SMOTER-style synthesis. Cheaper than LDS, same intent
  (don't let the 18 % zeros + bulk drown the rare tail). A `sample_weight ∝
  1/density(y)` is a one-line first probe.

### L3 — post-hoc marginal calibration (DONE-as-preview; the cheap product win)

Ranking-preserving monotone remaps fit LOIO (`src.calibration`):
- **Quantile-matching** (= histogram transfer): map the prediction distribution onto
  the truth distribution. **Recovers the marginal by construction** — preview LOIO:
  top-bin ratio **0.71→0.87**, near-zero **1.8 %→18.6 %**, marginal-L1 **0.0057→0.000**,
  Spearman **0.651→0.644** (preserved).
- **Isotonic** — *does not help* (fits the compressed conditional mean).
- Open refinements: **global vs per-region vs covariate-conditioned** mapping
  (condition on predicted-mean or the §2.7 novelty score) to handle a genuinely
  boulder-poor image whose lows should *not* be lifted.
- **Tier-1 — use a *flexible* monotone calibrator; ISOTONIC wins** (TESTED
  2026-06-15, `_diag_tier1_{isotonic,beta}.py`). The raw classifier is over-dispersed
  (under-confident lows, over-confident highs); temperature is one global knob, so it
  fixes the high end *at the cost of* the low end (split-ECE low 0.043→0.063, high
  0.096→0.021; net ECE 0.060→0.049). **Isotonic** bends both ends → ECE **0.060→0.014**
  (low/high both 0.014). **Beta** (smooth 3-param, `BetaCalibrator`) lands at 0.040 —
  its 3 parameters underfit the reliability curve. The feared ranking cost is **not
  real at deployment**: the LOIO pooled-AUC drop (isotonic 0.848→0.833) is a per-fold
  artifact; a single *global* calibrator is AUC-exact (isotonic +0.0003, beta +0.0000),
  and isotonic's ties are harmless at n=161k. **Recommendation: isotonic** as the Tier-1
  `CalibrationLayer`; beta is a smooth fallback if step artifacts ever matter. Gate
  ECE ≤ 0.05 (both pass) and global-fit AUC within ±0.005 (both pass).

### L4 — report the distribution (the honest product)

Deliver Tier-2 as **(median, [P10,P90]) per tile** + rich/poor probability, the
natural output of L1's quantile/distributional heads. Validate **interval coverage**
LOIO (does [P10,P90] cover truth ≈80 %?). The map renders the median with an
uncertainty overlay — the partner to the §2.7 reliability layer, and the only honest
answer to the texture floor. (Closes the loop: the floor that caps the point estimate
becomes a *reported* quantity rather than a hidden error.)

### Ranking-first (cross-cutting)

Because the L3 residual is *ranking*, an objective that optimizes ranking directly —
a differentiable Spearman/soft-rank surrogate or a listwise learning-to-rank loss —
then hands off to L3 for magnitude, is a principled combination. Speculative; park
behind HL-Gauss and the L2 items unless they stall.

## 4. Discipline (every stage)

- **Post-hoc calibration does NOT reopen the freeze.** Tier-1 classifier stays
  frozen (PLAN_FM §2.1); the calibration layer is rank-preserving and sits after it.
  The Tier-2 `mlp_reg` is **not yet frozen**, so L1/L2 retraining is allowed there.
- **LOIO-honest always** (`loio_calibrate`); the deployed calibrator/head is fit on
  all 38 and inherits the LOIO number as a conservative bound.
- **Must-not-regress ranking.** Report Spearman/NDCG/AUC; reject magnitude gains
  bought with ranking loss.
- **Marginal-match assumes in-cohort** — mitigate via per-region fit or novelty
  gating (documented risk).
- **Time-box the bake-off.** If nothing beats `mlp_reg`+quantile-match on the
  scorecard without hurting ranking, ship L3 and stop.

## 5. Staged execution

| stage | lever(s) | content | cost |
|---|---|---|---|
| **0 DONE** | diagnose | `src/calibration.py`, notebook 23, scorecard | — |
| **1 (next)** | L3 (+Tier-1) | bank a `CalibrationLayer` (isotonic Tier-1 + **global** quantile-match Tier-2), wire into `predict_window`/map as an **optional** step (raw toggle); ship for the **regional map**. Design resolved 2026-06-16 — see below. | low, no GPU |
| **2 DONE** | L1 | **HL-Gauss + pinball + neural-ZILN bake-off** (`_diag_tier2_l1_bakeoff.py`): all a **WASH on ranking** vs `mlp_reg` (best = pinball.median, paired p=0.48). Keepers: pinball.P90 = raw tail-calibrated point (top_ratio 0.98); intervals under-dispersed (58 % vs 80 %). **L1 ruled out as a ranking lever.** | head re-train, ~10 min/head GPU |
| **2b DONE** | L2 | **scale sweep** (`_diag_tier2_scale_sweep.py`) S=32→64: directional-up (paired +0.025, **p=0.19 n.s.**), partly easier-target. **`min_confidence` sweep** (`_diag_tier2_minconf_sweep.py`): ~~**HARMFUL** — monotonically worse~~ **AMENDED 2026-08-06 (R56): two-factor comparison; on a common target conf≥0.5 is a NULL (min p=0.178, n=38) and conf≥0.7 is directionally harmful but does not survive Holm.** S=128 still untested (needs P384 embed + 128-px grid). | cheap re-runs |
| **2c DONE** | L1+L2 | **LDS density reweighting** (`_diag_tier2_reweight.py`): **DOMINATED** — de-compresses raw (top 0.67→0.88) but costs ranking (paired p≈0.015); qmatch gives the marginal free. | small |
| **3** | L4 | uncertainty product: intervals + coverage validation + map overlay | small |
| **4** | — | freeze the Tier-2 head + calibrator into the deployable path; re-render docs §8; hand to THEMIS (W3) | — |

Order rationale (updated 2026-06-15 after the Stage-2 bake-off): **L1 is now fully
ruled out as a ranking lever** — the cheap swaps (log1p, count-Poisson) *and* the
distributional heads (HL-Gauss, pinball, neural-ZILN) all wash out vs `mlp_reg`,
because compression is the intrinsic aleatoric floor, not a loss-shape artefact. A
second, independent proof: **Tier-1 `P(rich)` ranks `fractional_area` as well as the
dedicated Tier-2 regressor** (per-image 0.437 vs 0.433; within-rich only 0.34) — a
*classifier* hits the same ~0.43 wall, so the wall is the data, not the head. **The L2/2c levers are now also exhausted (2026-06-16) and none beats `mlp_reg`:**
coarser scale (S=64) is directional-only (paired +0.025, p=0.19); LDS reweighting is
**dominated** (de-compresses raw, costs ranking p≈0.015); `min_confidence` filtering is
~~**harmful** (monotonically worse, conf≥0.7 p<0.001)~~ **a null at conf≥0.5 on a common target
(R56, amended 2026-08-06) — it neither helps nor hurts, so it is not evidence about the ceiling
at all.** So **no in-cohort retraining lever moves the per-image ranking ceiling** — it is the
5 m/px CTX magnitude floor, ~~confirmed five ways~~ **on the frozen Fang-ViT/GeM-96/S=32
embedding; the "five ways" all hold that representation fixed (R55) and one of them (this row)
has now been withdrawn as evidence.** **The path forward is not a better model:** ship **Stage 1** (productize
qmatch + isotonic into the map — the marginal *is* fixable, the visible win) and pursue
the **§2.3 expansion cohort** (the only thing that can raise the ranking ceiling — more
data, more tail). L3 stays the product win (Tier-1 = isotonic, Tier-2 = quantile-match);
L4 is the honest endpoint (intervals need recalibration, 58 % coverage); the one-model
simplification (qmatch `P(rich)`) is viable at a small cost (§3 ceiling diagnostic).
Only S=128 (needs a fresh embedding pass + label grid) remains untested.

### Stage 1 design — RESOLVED 2026-06-16 (Brian); IMPLEMENTED same day

Built: `CalibrationLayer` + `QuantileMatcher` in `src/calibration.py` (+8 tests);
optional `calibrator`/`apply_isotonic` wiring in `mapping.predict_window` (+abundance
raster, +3 tests); `scripts/bank_calibration.py` banks the layer to
`models/deployable/calibration.npz`; `scripts/map_pilot.py` gains `--raw` / `--no-isotonic`
+ an abundance panel. The deployment is **staged**, and the granularity question follows
the stage:

1. **`CalibrationLayer`** (new, in `src/calibration.py`) bundles two fitted, monotone
   maps with a `DeployableHead`-style `save`/`load`: **isotonic** (`P(rich)` →
   calibrated probability) + **global quantile-match** (raw abundance → `fractional_area`
   marginal). Fit **deployment-honest on all 38 images** (not LOIO); the LOIO numbers
   (Tier-1 ECE 0.014; Tier-2 top_ratio 0.86, rank-preserved) are the conservative bound
   it inherits, since off-HiRISE terrain has no truth. **Asymmetry between the two maps
   (verified 2026-06-16):** the **Tier-2 qmatch is the substantive, robust win** (top
   0.66→0.86, marginal-L1→0, by construction) — the abundance product. The **Tier-1
   isotonic ECE win is NOT statistically robust** — pooled 0.060→0.014 but per-image
   paired Wilcoxon p=0.11 (13/38) and the image-bootstrap CI on the reduction crosses
   zero ([−0.054, +0.007]); it is a rank-safe, gate-clearing *aggregate polish*, not a
   per-image improvement. Hence isotonic is **optional** (`apply_isotonic`, default-on
   for the in-distribution regional map) and must be described as a polish, not touted.
2. **Calibration is OPTIONAL — raw must stay a one-flag toggle.** `predict_window`
   (and the map scripts) take an optional `calibrator`; `None` ⇒ render raw, exactly as
   today. The map scripts get a `--calibrate/--raw` switch so raw and de-compressed maps
   are both trivially producible (and comparable side-by-side, the QA pattern).
3. **One-model simplification is the default abundance path.** No separate Tier-2 head:
   the abundance map is `quantile_match(P(rich) → fa)` (isotonic gives the rich/poor map
   off the *same* `P(rich)`). Costs ~0.02 ranking vs a dedicated head (§3) but deletes a
   whole trained artifact. Banking a dedicated all-data Tier-2 regression head stays a
   documented option if abundance ranking is later promoted to a first-class deliverable.
4. **Granularity is GLOBAL for the regional map.** The first deliverable maps the cohort
   cluster (≈40–46°N, 0–20°E), so deployment terrain ≈ the training cohort **by
   construction** — global qmatch's assumption holds, no gating needed. Global qmatch is
   also a *fixed pointwise map* (not per-window re-ranking), so it only mis-scales where
   the **classifier itself** is fooled by out-of-distribution texture.
5. **OOD handling is DEFERRED to the global-map phase** — but the layer is built
   **extensible** (an optional novelty/flag hook) so adding it later is a small change,
   not a re-architecture. The dusty-plains / OOD problem (where global qmatch can amplify
   a fooled `P(rich)` into fake abundance) is a *global-map* concern; it is materially
   shrunk first by the **§2.3 expansion cohort** (broader reference) and checked by
   **THEMIS (W3)** (independent coarse abundance over exactly the un-HiRISE'd regions).
   When built, the OOD layer **flags** ("classifier is extrapolating here"), it does not
   fabricate a fix; covariate-conditioned qmatch is the eventual refinement once the
   data to validate it exists. (Note: the §2.7 novelty score validated *negative as a
   skill predictor* at n=38 — it is usable as a definitional OOD/extrapolation flag, not
   as an accuracy estimate.)
6. **L4 uncertainty is OUT of Stage 1** — its intervals are under-covered (58 % vs 80 %)
   and need recalibration first (Stage 3).

Deliverables: `CalibrationLayer` + tests (round-trip, monotonicity, the §6 gates);
optional wiring in `predict_window` + a `--calibrate/--raw` map flag; bank the fitted
layer; re-render the regional/headline maps raw-vs-calibrated; update `docs/model_evidence.md`
§8 + `DATA_DICTIONARY` if output columns change.

## 6. Metrics (declared)

- **Tier-1:** ECE ≤ 0.05; AUC within ±0.005; reliability diagonal; Brier.
- **Tier-2 point:** top-bin ratio ∈ [0.8,1.2]; near-zero pred within ±3 pts of
  truth; marginal-L1 ↓ ≥ 50 %; **Spearman & NDCG@5 % within ±0.01** of the
  uncalibrated recipe (the hard constraint).
- **Tier-2 distribution (L4):** interval coverage within ±5 pts of nominal; sharpness
  (mean interval width) reported.
- **External:** positive THEMIS rank-correlation on overlap (W3 consumes Stage-4).

## 7. Sequencing, risks, open decisions

```
PLAN_FM 2.6 deployable head + map (DONE)
   └─ Stage 2 + 2b/2c (L1/L2 bake-off) DONE → all levers exhausted, ceiling = data
        └─ Stage 1 (L3 CalibrationLayer, global qmatch, NEXT) → regional map shipped
             └─ Stage 3 (L4 intervals) → Stage 4 freeze+wire → global-map OOD handling
PLAN_ModelUsability W3 (operationalized here) ── THEMIS validation consumes Stage 4
§2.3 expansion cohort ── the only ranking-ceiling lever; feeds the global map
```

Independent of the §2.3 expansion cohort (works on the 38). **Risks:** covariate
shift breaks global marginal-matching on OOD terrain — **scoped out of the regional
map (in-distribution by construction); a global-map concern, deferred** (§Stage 1
design pt 5); the floor is the floor (L1/L2 raise it, L4 reports it — no per-tile
miracle); per-fold pooling adds a small Spearman wobble (report per-image).

**Decisions — RESOLVED 2026-06-16 (Brian):**
1. **Headline Tier-2 product** → distributionally-correct **point map** (Stage 1,
   global qmatch); the L4 uncertainty-interval map is deferred (Stage 3, intervals
   need recalibration).
2. **How far to push L2** → **done** — all in-cohort L2/2c levers tested and exhausted
   (§5); not pursued further. The ranking ceiling is the data; §2.3 expansion is the
   lever, not a coarser/cleaner in-cohort target.
3. **Calibration granularity** → **global** for the regional map (in-distribution);
   OOD flag + covariate-conditioning **deferred to the global-map phase**, layer built
   extensible (§Stage 1 design pts 4–5). Calibration is an **optional** step (raw toggle).
4. **One-model simplification** → **default** abundance path (qmatch `P(rich)`, no
   separate Tier-2 head); dedicated head remains a documented option.
5. **ViT fine-tune** → still **out** — the ceiling is the data, not the representation;
   nothing in Stage 2 motivates it (ties to the PLAN_FM fine-tune go/no-go).
