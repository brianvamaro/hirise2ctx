# Review area: stats-fallacies

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-01
- **Verification:** self-refuted (two independent single-agent passes at the same commit; not
  cross-verified by a human)

> **Two passes.** Findings **1–5** are pass 1, already folded into the register as **R36 / R39 / R40 /
> R41 / R42**. Findings **6–7** are a **second, independent agent** re-running this area at the same
> commit; it read pass 1's file first and deliberately worked the gaps pass 1 listed as unchecked, so
> it adds only what pass 1 did not file. Pass 2 also **corrects one sub-claim inside finding 4** — see
> "Pass-2 corrections" below. Nothing from pass 1 was removed.

## Findings

### stats-fallacies-1 — The §4.3 trend guard adjudicates metadata-vs-geology on *raw* R², throws away the null it computed for each side, and the two sides' nulls differ 2.2×
- **Severity:** high (record correctness; the guard is explicitly retained as reusable code)
- **Liveness:** dead-closed programme, but `lv.trend_verdict` is listed under "Retained deliverables — general, stays in the codebase" (`DECISIONS.md:5578-5581`)
- **Confidence:** high
- **Where:** [src/leveling.py:723](../../src/leveling.py#L723), [:726](../../src/leveling.py#L726) (the rule);
  [:645](../../src/leveling.py#L645), [:667](../../src/leveling.py#L667) (`null_p95_r2` computed and returned);
  [scripts/f_region_stagec.py:555-558](../../scripts/f_region_stagec.py#L555-L558) (banked without the nulls);
  `PLAN_FBuild.md:248-254`; `tests/test_leveling.py:323-333`

`attribution` and `group_r2` exist *because* raw correlation is meaningless here — the module's own
docstring says so: "*Both metadata … and geology proxies … are themselves spatially smooth, so a naive
correlation p-value is meaningless here — every axis 'wins'*" ([src/leveling.py:624-627](../../src/leveling.py#L624-L627)).
Both functions therefore return `null_p95_r2` beside `r2`. But `trend_verdict` then decides the case on
**`m_r2` vs `g_r2 + margin`** — bare, uncalibrated R² — and the Stage-C driver never banks the nulls, so
the floor-relative numbers are computed and discarded (the same pattern as R11's dropped `tr2`).

On the banked 906-frame run the two sides' floors differ by 2.2× and the ranking **inverts** under
calibration: metadata's dominant axis clears its null (`ln_frame_median` R² 0.0847 vs its own null p95
0.0685, ratio **1.24**, p = 0.029) while geology's dominant axis does **not** (`mola_elev` R² 0.1415 vs
null p95 **0.1504**, ratio **0.94**, p = 0.056). Raw, geology "leads" 0.1415 vs 0.1083; floor-relative,
geology is *below its own spatial null* and metadata is above it. The group p-values already encode this
(meta 0.0190 significant, geo 0.0579 not) and the rule uses them only as an on/off gate before comparing
the uncalibrated magnitudes.

Second limb: **the rule has no absolute-explanatory-power floor.** The winner explains ~11 % of the
smooth field; 86–89 % of a **21.3-logit region-wide ramp** is unexplained by *any* axis, yet
`trend_verdict` will return `FULL` ("smooth field is artifact-side → apply the full offsets") on that
evidence, and under the pre-2026-07-30 rule it did.

- **Failure scenario:** any region where the geology proxy is smoother than the metadata proxies (the
  normal case — MOLA/THEMIS are smooth by construction) mechanically inflates `g_r2`, so `RESIDUAL_ONLY`
  becomes easy to reach and `FULL` nearly unreachable, independent of what the offsets actually track.
  Concretely, on the 906-frame run the verdict is `AMBIGUOUS` — escalated to Brian, and `pfree` (which
  deletes the region-wide plane outright) was chosen — where the null-calibrated reading says
  metadata-dominant, i.e. the plane is artifact and should have been *applied*, not deleted. The claim
  now on record in `PLAN_FBuild.md:250-252` and `DECISIONS.md`, "geology R² **0.142** > metadata R²
  **0.108** … geology R² > metadata R² in 20/20 seeds", is 20/20 repetitions of an uncalibrated
  comparison against an axis that never cleared its own null.
- **Evidence:**
  ```
  src/leveling.py:723-729
      if g_sig and g_r2 > m_r2 + margin:
          return {"verdict": "RESIDUAL_ONLY", ...}
      if m_sig and m_r2 > g_r2 + margin:
          return {"verdict": "FULL", ...}
      return ambiguous

  src/leveling.py:667   (group_r2 — the calibration it returns and nobody reads)
      "axes": used, "null_p95_r2": float(np.percentile(null, 95))}

  scripts/f_region_stagec.py:555-558   (what is actually banked)
      "meta_group_r2": ..., "meta_group_p": gmeta["p_value"], "meta_axes": ...,
      "geo_group_r2": ...,  "geo_group_p": ggeo["p_value"], "geo_axes": ...,
      # no meta_group_null_p95 / geo_group_null_p95 anywhere in fbuild_trend_guard.csv (47 columns)

  reports/figures/fbuild_stagec_attribution.csv
      ln_frame_median, r2=0.08472, p=0.0290, null_p95=0.06854   -> ratio 1.24
      mola_elev,       r2=0.14152, p=0.0559, null_p95=0.15040   -> ratio 0.94
  ```
- **Self-refutation attempted:** (a) I checked whether the raw comparison is fair on degrees of freedom —
  metadata has 4 axes vs geology's 2, so `m_r2` is the *inflated* side, which would push the other way;
  at n = 906 the adjusted-R² correction is ~0.004 vs ~0.002, immaterial, so DoF is not the story.
  (b) `tests/test_leveling.py:323-333` pins exactly this behaviour with the 906-frame numbers verbatim —
  but it pins the *implementation* of the raw-R² margin, and its docstring's justification ("a side could
  win holding the LOWER R²") is the claim I am challenging: under a permutation design, winning while
  holding the lower raw R² is the *expected* outcome when your axis is less spatially smooth.
  (c) The 2026-07-30 change was motivated by a real problem — the `g_sig` flag flips across seeds
  ("8 draws in 1000") — so reverting to a pure significance rule is not the fix either; the fix is to
  compare on the scale the nulls already provide. (d) Verdict impact: `full` was not shipped and the
  build was aborted for other reasons, so no shipped number changes — hence "record correctness", not
  blocker. It survives because the guard is retained as general-purpose code and the mis-stated evidence
  is quoted in three documents.
- **Fix:** compare `m_r2 − meta_null_median` vs `g_r2 − geo_null_median` (or the ratios to `null_p95`)
  with the margin applied on that scale; bank `meta_group_null_p95` / `geo_group_null_p95` /
  `*_null_median` into `fbuild_trend_guard.csv`; add a floor branch that returns `AMBIGUOUS` (never
  `FULL`) when neither side's excess-over-null exceeds some minimum share of the smooth field; and amend
  `PLAN_FBuild.md:250-252` to quote the floor-relative pair.

---

### stats-fallacies-2 — The H4 leg-B skill gate could not have failed: the λ-regularised solve pins the between-observation offset to *exactly zero* on a disconnected graph, and 17 of 28 observations received one identical constant
- **Severity:** high (this PASS is one of the two gates in the reopening rule that authorised ~265 CPU-h + 33 GPU-h)
- **Liveness:** dead-closed, but quoted as live evidence in `ROADMAP.md:19`, `PLAN_StripingArtifact.md:238,267`, `PLAN_H4_Leveling.md:59-67`, `DECISIONS.md:4534-4538`
- **Confidence:** high (mechanism proved algebraically and reproduced numerically; the numbers re-derive from committed artifacts)
- **Where:** [scripts/f_h4_legb.py:145](../../scripts/f_h4_legb.py#L145), [:149-154](../../scripts/f_h4_legb.py#L149-L154),
  [:162](../../scripts/f_h4_legb.py#L162); [scripts/f_h4_level.py:90-106](../../scripts/f_h4_level.py#L90-L106);
  `reports/figures/f_h4_legb_offsets.csv`

`PLAN_H4_Leveling.md:48-57` correctly rules out per-image AUC as blind to H4 and pre-declares **pooled**
PR-AUC / prec@5% as the instruments that "DO see cross-frame level changes". But on the leg-B graph
(58 frames / 47 edges / **21 components**) the pooled instrument sees nothing either, and for a provable
reason: for any connected component *c*, the all-ones vector `1_c` lies in the null space of the graph
Laplacian **and** `Aᵀ W b` has exactly zero projection on it (each edge contributes `+w·δ̄` to *i* and
`−w·δ̄` to *j*, both inside *c*). So `(L + λI) o = AᵀWb` forces `mean(o_c) = 0` **exactly, for every
component and every λ**. The between-component level — the *only* thing a pooled cross-image metric can
respond to — is therefore identically zero by construction, not estimated. Because the graph is "mostly
within-obs", most observations *are* a whole component, so `obs_off = mean(frame offsets) = 0` before the
global `o − median(o)` gauge and `= −0.0753` after it.

Measured on the banked artifact: **17 of 28** observations carry the identical value `−0.0753`; the
applied offsets have **interquartile range exactly 0** (both quartiles at `−0.0753`) and sd **0.308**
logits — against the build's own solved offsets at sd **1.46** (`resid`) / **1.78** (`pfree`) / **6.45**
(`full`). Only **9 of the 36 scored images** received a shift differing from that constant by ≥ 0.05.
The gate then reported `Δ pooled PR-AUC = −0.0104` and PASS.

- **Failure scenario:** the reopening rule was "η² ≲ 0.05 **at skill ≥ −0.02**"
  (`PLAN_StripingArtifact.md:242`). The skill half was cleared by an instrument that applies a
  near-constant to the predictions, so it would have returned ≈ 0 no matter how damaging real leveling
  is. When the same statistic was finally computed on real, one-component offsets at build scale, it read
  **−0.089 (full) / −0.030 (resid) / −0.186 (pfree)** (`reports/figures/fbuild_gate5_skill.csv`) — 3–18×
  the tolerance. The four documents that say "PASS — leveling preserves skill on real LOIO predictions"
  describe a measurement of nothing.
- **Evidence:**
  ```
  scripts/f_h4_level.py:105-106
      o = np.linalg.lstsq(ata, atb, rcond=None)[0]
      return o - np.median(o)          # ONE global gauge across 21 components

  scripts/f_h4_legb.py:149-154
      for obs_id, pids in obs_frames_all.items():
          vals = [offset[p] for p in pids if p in offset]
          if vals:
              obs_off[obs_id] = float(np.mean(vals))     # = 0 pre-gauge for a whole component

  reports/figures/f_h4_legb_offsets.csv  (17 of 28 rows)
      ESP_017355_2260,-0.0753   ESP_042964_2160,-0.0753   ESP_046328_2180,-0.0753   ...
  ```
  Numerical confirmation of the mean-zero property on a synthetic 2-component graph through
  `f_h4_level.solve_offsets` (λ = 0, 1e-6, 300): both components' post-gauge means are identical to 6 dp
  at every λ, i.e. zero before the gauge.
- **Self-refutation attempted:** (a) I re-ran the whole gate from `f_leg_b_loio_preds_minnaert_center.csv`
  and reproduced −0.0104 exactly, so the banked number is right — the defect is in what it measures.
  (b) I tested whether the arbitrary gauge constant itself contaminates the comparison (28 obs get
  `−0.0753`, the other 8 get `fillna(0.0)`): recentring so no-edge obs get 0 moves the result to −0.0103,
  and mean-centring to −0.0103 — **immaterial**, so that sub-claim is dead. (c) The fragmentation caveat
  *is* recorded ("⚠ graph fragmented … 21 components ⇒ mostly within-obs"), so this is not undocumented —
  but "mostly within-obs" understates it: the correct statement is that the cross-image component of the
  correction is mathematically zero, and the caveat sits beside a bolded **PASS** in every document.
  (d) The abort went the other way, so no shipped number is wrong; the damage is to the reopening
  decision and the record.
- **Fix:** on a disconnected graph, refuse to score a cross-image level instrument at all (assert
  `n_components == 1`, or report the between-component shift as "unidentified") and state the applied
  shift's spread beside every skill delta. Retro-annotate the four documents: the leg-B row is a
  *within-obs* null result, not a cross-image skill PASS.

---

### stats-fallacies-3 — Every acceptance tolerance in the striping/F programme is ±0.02, and not one of the statistics it gates has ever been given a sampling uncertainty; the tolerance is at or below the noise
- **Severity:** medium (methodology; affects how every PASS/FAIL in the programme should be read)
- **Liveness:** live methodology (`src/fgates.py` constants are current; the same pattern would be reused by PLAN_RegionalMap's parked validation legs)
- **Confidence:** high
- **Where:** [src/fgates.py:38-39](../../src/fgates.py#L38-L39) (`THEMIS_TOL = 0.02`, `SKILL_TOL = -0.02`);
  [scripts/f_h4_legb.py:48](../../scripts/f_h4_legb.py#L48) (`GATE = -0.02`);
  `PLAN_StripingArtifact.md:214`; `PLAN_FBuild.md:370`;
  contrast [scripts/probes/_diag_within_image_deltas.py:89-90](../../scripts/probes/_diag_within_image_deltas.py#L89-L90)

`grep -rniE "bootstrap|confidence.interval|\bCI\b|std.?err"` over all 23 `scripts/f_*.py`, `src/fgates.py`
and `src/striping.py` returns **only** the `# OpenMP bootstrap` import comment — there is no resampling,
no standard error and no interval anywhere in the F/striping evidence chain. (Permutation nulls exist,
but only for η² and the trend surface; nothing for any skill or ρ delta.) The modeling programme, by
contrast, bootstraps exactly this class of statistic at image level and reports a CI + Wilcoxon p
(`docs/modeling_results.md:970-980`, produced by `_diag_within_image_deltas.py`). The F programme's
numbers are single point estimates compared against a bare ±0.02.

I ran the missing 36-image cluster bootstrap over the committed
`reports/figures/f_leg_b_loio_preds_minnaert_center.csv` (2 000 → 400 draws, resampling whole
observations, recomputing pooled PR-AUC on each draw):

| claim on record | point | cluster-boot 95 % CI | reading |
|---|---|---|---|
| H1 − mosaic baseline (`DECISIONS.md:5195` "+0.0296"; ROADMAP/PLAN "+0.019 above mosaic") | +0.0296 | **[−0.0218, +0.0863]**, P(Δ<0) = 0.16 | indistinguishable from zero |
| H4 − H1 (`GATE = −0.02`, declared PASS) | −0.0104 | **[−0.0272, +0.0018]**, P(Δ < −0.02) = 0.10 | PASS not established |

The H1 − mosaic standard error is ≈ 0.028 — **1.4× the tolerance the gate uses**. And `minnaert_center`
was itself the argmax over 7 candidate input mappings scored on this same statistic
(`f_leg_b_loio_summary_{,global,minnaert,minnaert_center,minnaert_cubic,minnaert_w,minnaert_wl}.csv`:
pooled PR-AUC 0.626 / 0.721 / 0.727 / **0.796** / 0.726 / 0.743 / 0.784), so the point estimate is
selection-inflated on top of being inside the noise.

- **Failure scenario:** a gate whose tolerance is smaller than the statistic's standard error decides by
  coin flip. `H1 PASS → H2/H3/H4 → reopening call → 907-frame build` is a chain of such decisions; the
  chain would read identically if H1's true effect were 0 or −0.02. Same structure for gate 3
  (`THEMIS_TOL = 0.02` on a median of 26 per-tile Spearman ρ, each computed over ~10⁵ spatially
  autocorrelated pixels with no effective-n correction) and gate 5.
- **Evidence:**
  ```
  src/fgates.py:38-39
      THEMIS_TOL = 0.02        # gate 3 "not degraded" (mirrors scripts/f_h4_themis.GATE_TOL)
      SKILL_TOL = -0.02        # gate 5 delta tolerance

  scripts/f_h4_legb.py:192-193
      print("\nGATE (H4 does not degrade skill vs H1):",
            "PASS" if d_pr >= GATE else "FAIL")        # one number, no interval
  ```
- **Self-refutation attempted:** (a) A cluster bootstrap over images is itself approximate for a *pooled*
  metric, so I also checked the paired form — the H4−H1 delta is paired within observation and comes out
  much tighter (SE ≈ 0.007) than the unpaired H1−mosaic delta (SE ≈ 0.028), which is the correct
  behaviour and means only the *unpaired* comparison is badly underpowered; I have narrowed the claim
  accordingly. (b) Gate 3's improvements do survive an image-level sign test (see "Refuted"), so not every
  ±0.02 verdict is noise. (c) The project does report permutation nulls for η², so it is not blanket
  innumerate — the gap is specific to skill/ρ deltas. It survives because the ±0.02 constants are still in
  `src/fgates.py` and PLAN_RegionalMap's parked legs 2–5 inherit the same harness.
- **Fix:** add one `cluster_bootstrap_delta(obs_id, y, p_a, p_b, n=2000)` helper (the code already exists
  in `_diag_within_image_deltas.py`) and emit `delta`, `ci_lo`, `ci_hi`, `n_obs` for every gate-3/gate-5
  row and for `f_h4_legb`; require the CI, not the point estimate, to clear the tolerance.

---

### stats-fallacies-4 — Stage 7d's pooled tests treat spatially autocorrelated tiles as independent; the headline p-values overstate the evidence by ~12 orders of magnitude and the strongest feature fails an image-level test
- **Severity:** medium (closed programme, but `docs/compositional.md` is a writeup intended for external readers)
- **Liveness:** dead-closed (PLAN_Compositional), live document
- **Confidence:** medium-high (re-derived from the committed `stage7d_pooled_shadow_0.10.parquet`)
- **Where:** [src/stage7d_pooled.py:174](../../src/stage7d_pooled.py#L174) (`stats.mannwhitneyu` on pooled tiles),
  [:187](../../src/stage7d_pooled.py#L187) (`stats.spearmanr` likewise), [:286-304](../../src/stage7d_pooled.py#L286-L304)
  (the pooled rich/poor vectors); `docs/compositional.md:405-412`, `:896-903`

`run_pooled_binary_tests` concatenates every eligible image's tiles into one rich vector and one poor
vector and runs a single Mann-Whitney U. CLAUDE.md's own invariant 6 states the reason this is invalid:
"*tiles within an image are spatially correlated*" — which is why splits are group-aware. Per-image
z-scoring ([:191-209](../../src/stage7d_pooled.py#L191-L209)) removes the per-image *mean*, not the
within-image dependence, and nothing in the module aggregates at image level. `docs/compositional.md:899`
attributes the tiny p-values to "the large pooled sample size", which is the *n*-inflation half of the
problem but not the pseudoreplication half, and no cluster-level statistic is emitted anywhere.

Re-analysing the committed artifact at image level (the per-image rows the module already computes,
`level == "per_image"`, `test_type == "mann_whitney_partial_dust"`, P4_area, 26 images):

| feature | pooled p (as published) | image-level sign test | Wilcoxon on the 26 per-image d |
|---|---|---|---|
| `IR_iof` | **2.8e-14** (\|d\| 0.183, the largest partial-dust effect) | 18/26 negative, **p = 0.076** | p = 0.013 |
| `IR_over_RED` | 3.4e-09 | 19/26, p = 0.029 | p = 0.022 |
| `IR_over_BG` | 2.0e-09 | 19/26, p = 0.029 | p = 0.025 |

- **Failure scenario:** a reader of `docs/compositional.md:405-412` (or of the DECISIONS 2026-06-02 entry)
  takes `p = 1.7e-73` / `9.3e-33` as the strength of evidence for a cross-image compositional signal. The
  honest cluster-level figure is `p ≈ 0.02–0.08` on 26 images — still supportive for the two ratio
  features that carry the argument, but one significance star, not seventy, and `IR_iof` does not survive
  at all. There is also no multiplicity control across 6 features × 2 partition rules × 3 transforms
  (36 pooled tests) plus 2 features × 26 images in the attribution classifier, whose thresholds
  (`ATTRIBUTION_RAW_P = 1e-3`, `ATTRIBUTION_PARTIAL_P = 0.05`, [:422-424](../../src/stage7d_pooled.py#L422-L424))
  are uncorrected — at α = 0.05 on 26 images, ~1.3 `composition_residual` classifications are expected by
  chance.
- **Evidence:**
  ```
  src/stage7d_pooled.py:286-288
      rich = sub.loc[sub[rich_col], value_col].to_numpy()      # all images concatenated
      poor = sub.loc[~sub[rich_col], value_col].to_numpy()
      r = mann_whitney_with_effect(rich, poor)

  src/stage7d_pooled.py:174
      U, p = stats.mannwhitneyu(rich, poor, alternative="two-sided")   # iid assumption

  docs/compositional.md:899-900
      "The headline-significant p-values reflect the large pooled sample size (n ≈ 4 450 tiles at
       T=0.10), not large per-tile separability."
  ```
- **Self-refutation attempted:** (a) The Limitations section *does* discount the p-values, and the effect
  sizes (the numbers the argument actually leans on) are unaffected by clustering — so this is not a
  hidden claim. It survives because the published tables give p-values to 2 significant figures at 1e-73
  and the module offers no cluster-level alternative, while the per-image rows needed to compute one are
  already in the artifact. (b) The headline conclusion **survives** the correction for the two ratio
  features (p 0.029/0.022), so the science is not overturned — which is why this is medium, not high.
  (c) `residualise_per_image` fits a *linear* dust control and then a *rank* correlation is taken on the
  residuals — a mismatch, but a conventional one, and I could not show it changes a sign.
- **Fix:** add an image-level aggregate (sign test / Wilcoxon over the per-image Cohen's d already
  produced by `run_per_image_binary_tests`) as the headline row, keep the pooled p only as a descriptive
  companion, and replace `docs/compositional.md`'s p-value column with the cluster-level figure.

---

### stats-fallacies-5 — The abort has exactly one cross-arm instrument: gate 5 is F-vs-F by construction and the cohort join never reads the mosaic raster, so the only mosaic-vs-F *skill* number ever measured points the opposite way to the verdict
- **Severity:** medium (record completeness on the project's most consequential decision)
- **Liveness:** dead-closed
- **Confidence:** medium
- **Where:** [scripts/f_region_gates.py:250-254](../../scripts/f_region_gates.py#L250-L254) (`cohort_table` reads only `map_f`),
  [:259-284](../../scripts/f_region_gates.py#L259-L284) (`gate5` iterates `VARIANTS` only);
  [scripts/f_map_compare.py:182-190](../../scripts/f_map_compare.py#L182-L190) (§5.1's skill column);
  `DECISIONS.md:5522-5552`

The abort verdict is "F trades the artifact it was built to remove for a worse one". The evidence that
compares F to the incumbent is a **single** table — per-observation `mean(calibrated abundance) /
mean(label fa)` — on the calibrated scale, mediated by the calibrator issues already filed as R33/R34.
Every other cross-arm instrument is missing or points the other way:

* **Gate 5 has no mosaic row and structurally cannot have one.** `cohort_table` builds `p_*`/`ab_*`/
  `abmos_*` from `map_f` only; the join carries no mosaic column (verified: the 24 columns of
  `fbuild_cohort_join.parquet` contain no mosaic field). So no same-footprint, same-scale pooled-skill
  comparison of mosaic vs F exists anywhere in the build — even though the abort's own level table did
  exactly this join for the mosaic's calibrated abundance, so the raster read was one line away.
* **The only mosaic-vs-F skill number on record says F wins**: leg-B pooled PR-AUC, 36 LOIO images, raw
  P(rich) — mosaic 0.7668 vs F/H1 0.7964 (+0.0296). §5.1's scorecard sources its F rows straight from
  that table (`f_map_compare.py:182-190`), so the ship-vs-fallback comparison would have carried
  `F_full = 0.7860` — the *inert* leg-B number of finding 2 — rather than the build's own 0.6547.
* The two cross-arm readings are never reconciled in `DECISIONS.md:5522-5552`, which quotes only the
  level table.

- **Failure scenario:** a reader (or a future revival of F) concludes the build was scored head-to-head
  against the incumbent on skill and lost. It was not scored head-to-head on skill at all; the one time
  it was, on a different footprint, scale and cohort, it won. The gap matters because R10 already shows
  the level table changes head, training set and input radiometry simultaneously, so the single surviving
  cross-arm instrument is also the confounded one.
- **Evidence:**
  ```
  scripts/f_region_gates.py:250-254
      for v in VARIANTS:                                   # h1only/full/resid/pfree — no mosaic
          for layer, key in (("prob_raw", f"p_{v}"), ("abundance", f"ab_{v}"),
                             ("abundance_moscal", f"abmos_{v}")):
              p = map_f / f"{tile}_{v}_{layer}.tif"

  scripts/f_map_compare.py:184-185
      m = {"baseline (mosaic)": "mosaic", "H1 (F, unleveled)": "F_h1only",
           "H1+H4 (F, leveled)": "F_full"}                 # from f_h4_legb_summary.csv
  ```
- **Self-refutation attempted:** (a) Ruling 4 (`DECISIONS.md:5054-5058`) deliberately scopes gate 5 to a
  **delta** because the F head is in-sample on all 21 obs — adding a mosaic row would compare two
  differently-in-sample heads, so the omission is defensible *as a gate*. It survives because that is an
  argument for not *gating* on it, not for never *measuring* it: the level table has exactly the same
  in-sample symmetry and is reported. (b) §5.1 never produced its artifact (R06 — no A1 raster), so the
  mislabelled `F_full` skill row is latent, not shipped; that is why this is medium and ranked last.
  (c) The abort verdict itself is about *level*, for which the level table is the right instrument — I am
  not claiming the verdict is wrong, only that it rests on one confounded arm with no skill counterpart.
- **Fix:** add `reports/map_region/{tile}_prob_raw.tif` and `{tile}_abundance.tif` to `cohort_table` as a
  `mosaic` row, report gate 5 with the mosaic absolute beside the F deltas (labelled in-sample for both),
  and have §5.1 take the F skill column from `fbuild_gate5_skill.csv` rather than `f_h4_legb_summary.csv`.

---

### stats-fallacies-6 — The η² reopening bar is an absolute constant that sits *below* the geological floor of the very crop it was calibrated on, and not one pilot-scale η² was ever given the rotation null that already existed in the module the pilot scripts import
- **Severity:** high (record correctness; this is the gate that sequenced H1→H2→H3→H4 and, together with
  the leg-B skill gate of finding 2, authorised the 907-frame build)
- **Liveness:** dead-closed programme, but the raw numbers are quoted as verdicts in `ROADMAP.md:19`,
  `PLAN_StripingArtifact.md:219-220`/`:231-256`, `PLAN_FBuild.md:46`, and notebook 28
- **Confidence:** high (chronology from `git log`; artifact headers dumped; the arithmetic is on numbers
  already on record)
- **Where:** [scripts/f_pilot_crop.py:56](../../scripts/f_pilot_crop.py#L56),
  [:270](../../scripts/f_pilot_crop.py#L270); [scripts/f_h2_eta2.py:47](../../scripts/f_h2_eta2.py#L47),
  [:115](../../scripts/f_h2_eta2.py#L115); [src/striping.py:339](../../src/striping.py#L339);
  `PLAN_StripingArtifact.md:219-220`, `:242`; `DECISIONS.md:5013-5016`, `:5019-5030`;
  `reports/figures/f_pilot_eta2_summary*.csv`, `f_h2_eta2_summary*.csv`, `f_h4_leveling_summary.csv`

The reopening rule is an **absolute** value of a statistic the module's own docstring says has no
group-count correction, so it has no fixed interpretable scale — which is exactly why
`src.striping.eta2_rotation_null` (the "rotation-null geological floor") exists. That helper was
committed on **2026-07-02** (`830a39b`), *one day before* `f_pilot_crop.py` was created (`f3a7c50`,
2026-07-03) and three days before PHASE 2 opened; `scripts/striping_frame_blocks.py:56` was already
calling it. Both pilot η² scripts import from that same module — `from src.striping import eta2,
load_frames` — and **never call the null**. Every number the docket turned on is therefore a bare η²
against a constant: F 0.179/0.277, H1 **0.1281**/0.0809, H2 0.1104–0.1492, H3 0.035–0.126, H4 **0.0505**.
None of `f_pilot_eta2_summary*.csv`, `f_h2_eta2_summary*.csv` or `f_h4_leveling_summary.csv` has a null
column (headers dumped; they are `eta2` / `partition,median,pred_overlap` / `partition_eta2,…`).

When the floor was finally measured — at build scale, 2026-07-28 — the project reached the general
conclusion itself: the absolute form is uninterpretable ("nothing can pass" at block scale, "nothing can
fail" detrended) and **"the 0.05 bar was calibrated on a ~75 km / 7-frame crop where the mosaic scores
0.1948 against a null of 0.083–0.117"** (`DECISIONS.md:5015-5016`). The bar is thus **40–57 % below the
only floor ever measured on the pilot footprint**. That ruling was applied *forward* to gate 1 only; the
pilot rows were never re-read.

- **Failure scenario:** read floor-relative against that one measured floor, **H1 alone removes 60–86 %
  of the mosaic's excess-over-null** (mosaic 0.1948 − [0.083, 0.117] = +0.078…+0.112; H1 0.1281 − same =
  +0.011…+0.045), while both points that "cross the bar" — H3 λ=100 (0.035) and H4 full (0.0505) — sit
  *below* that floor, i.e. reachable only by removing variance the roll-null attributes to geology. That
  reading inverts three recorded readings: H1 goes from "halves the artifact, but η² 0.081 does **not**
  yet clear the bar" (`DECISIONS.md:5211-5212`) to "essentially at the floor"; H3's skill collapse
  becomes the *expected price of chasing a sub-floor target* rather than evidence that "artifact removal
  and skill lie on **one monotone axis**" (a mechanism claim about the frozen ViT entangling radiometry
  with texture, `DECISIONS.md:4450-4455`); and H4's headline PASS becomes over-flattening. H2's
  "even k=64 leaves partition η² 0.131 ≈ H1's 0.128" is likewise a comparison of two null-free numbers
  0.0027 apart. The ~265 CPU-h + 33 GPU-h build was authorised on the un-floor-relative reading.
- **Evidence:**
  ```
  PLAN_StripingArtifact.md:219-220
      Baselines: mosaic raw **0.196** / A1 **0.141** / current F **0.179** / target **≲ 0.03–0.05**.
  PLAN_StripingArtifact.md:242
      **Decision rule:** if H1–H4 (or their combination) reach **η² ≲ 0.05 at skill ≥ −0.02**, the
      907-frame regional F build is back on the table

  DECISIONS.md:5013-5016
      So the literal "partition η² ≤ 0.05 on the full block" sits **below the geological floor**
      (nothing can pass) while the detrended reading is **already passed by the un-mitigated map**
      (nothing can fail). The 0.05 bar was calibrated on a ~75 km / 7-frame crop where the mosaic
      scores 0.1948 against a null of 0.083–0.117.

  scripts/f_pilot_crop.py:56    from src.striping import eta2, load_frames      # not eta2_rotation_null
  scripts/f_pilot_crop.py:270           e = eta2(comp, labels, fin)             # bare η², no null
  scripts/f_h2_eta2.py:115              rows[name] = round(float(eta2(comp, labels, fin)), 4)

  $ head -1 reports/figures/f_pilot_eta2_summary_minnaert_center.csv
  mapping,head,composite,eta2,n_cells,n_frames                 # no null_mean / null_p95 / excess
  $ cat reports/figures/f_h2_eta2_summary.csv | head -2
  label,k,partition,median,pred_overlap
  center,0,0.1281,0.0809,0.0738
  ```
- **Self-refutation attempted:** (a) **The obvious kill** — the null depends on the field's own
  autocorrelation, so the mosaic's 0.083–0.117 is not H1's or H4's null; `DECISIONS.md:5027-5030` says
  precisely this, and **pass 1 of this review retired the narrow version of the claim on that ground**
  (see the refuted list below). It does not kill this finding, because the claim here is *not*
  "0.0505 < 0.083 ⇒ over-flattened"; it is that **no pilot field's own null was ever computed** although
  the helper was one import away, so none of the pilot η² numbers has a scale at all — and the one floor
  that *was* measured on that footprint is ~2× the bar those numbers were judged against. (b) *Was the
  tool unavailable then?* No — `git log -S` puts `eta2_rotation_null` in `830a39b` (2026-07-02), before
  both pilot scripts existed. (c) *Was the bar pre-registered, so it cannot be outcome-driven?* The 0.05
  constant, yes (2026-07-05d / 07-09b), and the 2026-07-28 re-scoping is explicitly "encoded before any F
  number existed" — a genuine defence **for gate 1**. The defect is that the same logic was never carried
  back. (d) *Does it change the abort?* No: the abort turned on level coherence, and `ROADMAP.md:19`
  already states "Absolute η²≤0.05 met by nothing incl. the mosaic" — hence high/record-correctness, not
  blocker. That acknowledgement sits on the PLAN_FBuild row (`ROADMAP.md:18`); the raw pilot verdicts sit
  on the PLAN_StripingArtifact row (`:19`), which is where the annotation is needed. (e) *Is the 0.083–0.117
  band itself trustworthy?* It has **no committed artifact** (it appears
  only in `DECISIONS.md:5015-5016` and `:5025`, from read-only probes). Substituting gate 1's independently
  computed windowed null (p95 0.0676–0.0700, `fbuild_gate1_summary.csv`) leaves the bar still below the
  floor, by less (0.05/0.068 ≈ 0.74) — so the direction is robust to which measured floor you take.
- **Fix:** have `f_pilot_crop.py` / `f_h2_eta2.py` call `eta2_rotation_null` (both already import the
  module) and emit `null_mean` / `null_p95` / `excess` / `ratio` beside every η²; then either restate the
  PHASE-2 table floor-relative or annotate `PLAN_StripingArtifact.md:219-256` and `ROADMAP.md:19` that the
  pilot η² column is **not** floor-relative and that the 0.05 constant is below the only floor measured on
  that footprint — in particular that H3's "one monotone axis" and H4's "crosses the bar" both need that
  caveat.

---

### stats-fallacies-7 — The "MOLA leg" credited in the only ACTIVE plan's validation ledger is a self-fulfilling site-selection check with no model output in it, and the MOLA leg that the design actually specifies (leg 3) was never run
- **Severity:** low (record correctness, but on the only ACTIVE plan and on the shipped deliverable's
  validation ledger)
- **Liveness:** live-active-plan
- **Confidence:** high
- **Where:** `PLAN_RegionalMap.md:64` (leg 3 as designed), `:276`, `:284`, `:316-317` (the roll-up);
  `ROADMAP.md:12`; propagated into the `project_state_2026-06-17` memory note

The five-leg design's only MOLA leg is **leg 3 — "abundance vs distance from the −3795 m contour:
boundary peak, distal decay"**. What was actually executed is *"block median elevation = **−3794 m**,
i.e. the lHl1 contour bisects the block"* — a statistic computed from MOLA alone over the cohort
footprint, containing **no predicted abundance**. The footprint was selected *because* it straddles the
highland–lowland boundary (`:18`; `:284` "we are imaging exactly the boundary"), so the outcome is a
restatement of the site-selection criterion: there is no result it could have returned that would have
disconfirmed anything, and it is not a function of the quantity being validated. The plan body labels it
correctly — "**Correctness check:** block median elevation = −3794 m" (`:284`) — but the status roll-up
promotes it into the evidence ledger ("Validation so far: **MOLA leg shipped**", `:316-317`) and
`ROADMAP.md:12`, the file `CLAUDE.md` designates as the authority for current phase, records
"**MOLA leg done**".

- **Failure scenario:** the ledger for the shipped map reads "MOLA leg done; THEMIS leg-1 done but weak
  (ρ ≈ +0.07)", i.e. **one** of five legs produced a (weak) result and a non-test is counted as a second.
  A reader — or the eventual write-up, whose figure list at `:184` reuses the same contour — credits
  geometric corroboration of the abundance band against the paleoshoreline that has never been measured.
  Leg 3 is also the cheapest outstanding leg: `cache_v2/thermal/mola_dem_region.tif` and the 26-tile
  abundance mosaic are both already on disk, so it needs no fetch and no GPU.
- **Evidence:**
  ```
  PLAN_RegionalMap.md:64
      | **3. Shoreline-distance profile** | abundance vs distance from the −3795 m contour:
        boundary peak, distal decay | geometry independent |

  PLAN_RegionalMap.md:284
      Correctness check: block median elevation = **−3794 m**, i.e. the lHl1 contour bisects the
      block — we are imaging exactly the boundary.

  PLAN_RegionalMap.md:316-317
      Validation so far: **MOLA leg shipped** (block median −3794 m ≈ the lHl1 shoreline — the
      contour bisects the block); **leg 1 (THEMIS night-IR co-location) DONE but WEAK** ...

  ROADMAP.md:12
      **ACTIVE** — map shipped (26 tiles, Sherlock); MOLA leg done; THEMIS night-IR leg-1 done but
      weak (ρ ≈ +0.07); ...
  ```
- **Self-refutation attempted:** (a) *Just loose wording?* `:284` does say "Correctness check", so the
  plan body knows what it is — but the roll-up and ROADMAP do not, and ROADMAP is the index a new session
  reads first; the phrasing also travelled into the memory notes, which is how it becomes durable.
  (b) *Is leg 3 subsumed by leg 1?* No — leg 1 is THEMIS co-location scored by ρ, leg 3 is a
  distance-profile shape test; the design lists them as separate independent axes (`:62-64`) and
  `:68` groups 1–3 as jointly establishing "works at regional scale". (c) *Is the elevation check
  worthless?* No — it is a genuine and valuable CRS/reprojection sanity check (invariants 1–2), which is
  why the fix is relabelling, not deletion. (d) *Already filed?* `docs-consistency-2` quotes
  `ROADMAP.md:12` but only to show the ACTIVE row is stale about F-map blocking; neither it nor any other
  area file observes that the credited leg is not a test. (e) *Severity* — no shipped number is wrong, so
  low; it is filed because the only ACTIVE plan's next actions are chosen from this ledger.
- **Fix:** in `PLAN_RegionalMap.md:316-317` and `ROADMAP.md:12` replace "MOLA leg shipped/done" with
  "MOLA **retrieval + CRS check** done (block median −3794 m = the site-selection criterion, not a test);
  **leg 3 (shoreline-distance profile) NOT RUN**", and list leg 3 as the cheapest unblocked leg.

## Pass-2 corrections to pass-1 findings

- **`stats-fallacies-4` (registered as R42) over-states the chance rate of `composition_residual` by
  ~500×.** Its failure-scenario paragraph says "at α = 0.05 on 26 images, ~1.3 `composition_residual`
  classifications are expected by chance". But `classify_image`
  ([src/stage7d_pooled.py:459-467](../../src/stage7d_pooled.py#L459-L467)) requires a **conjunction**:
  the raw gate `|d| ≥ ATTRIBUTION_RAW_D = 0.20` **and** `p ≤ ATTRIBUTION_RAW_P = 1e-3`
  ([:421-422](../../src/stage7d_pooled.py#L421-L422)) must pass first,
  and only then is the partial-dust gate (`0.10` / `0.05`) consulted. The nominal chance rate is
  therefore governed by the 1e-3 gate (order 26 × 2 × 1e-3 × 0.05 ≈ **0.003** expected false
  classifications), not by α = 0.05. The *conclusion* that the attribution labels are noisier than they
  look survives, but for the reason finding 4 establishes in its own first half — the per-image p-values
  are computed on spatially autocorrelated tiles, so they are anti-conservative and the *effective*
  false-positive rate is unknown rather than nominally 1e-3. The Fisher's-exact Tier-1 test
  (`docs/compositional.md:561-563`, OR 23.0, p = 0.018) inherits that: it conditions on 4 attribution
  labels as if they were observed data, propagating none of their uncertainty, and reports OR to 3
  significant figures from 4 events with no interval (its exact 95 % CI spans roughly 1.6–350). Both
  corrections point the same way — cite the autocorrelation channel, not a multiplicity arithmetic that
  does not hold.

## Refuted by my own check

- **Gate 1's rotation null draws could be near-identity.** `eta2_rotation_null`
  ([src/striping.py:339-347](../../src/striping.py#L339-L347)) draws `rng.integers(H//8, H)`, whose upper
  end is a *circular* shift of −1 px, so the `H//8` floor does not exclude near-identity rolls. Checked
  with seed 0 at the gate-1 window size (469 px): 4/20 draws are closer than the intended floor in one
  axis but **0/20 in both**, so no draw reproduces the observed alignment. Immaterial.
- **The metadata side of the trend guard has 4 predictors vs geology's 2, inflating its R².** True but
  immaterial at n = 906 (adjusted-R² correction ~0.004 vs ~0.002), and it pushes *against* the finding.
- **Reusing `seed=0` for every gate-1 window and every map row biases the comparison.** It does the
  opposite: identical roll offsets across rows are common random numbers, which *reduces* the variance of
  the mosaic-vs-F contrast. (Separately, `np.percentile(out, 95)` on 20 draws is really ≈ the 91st
  percentile, so `null_p95` is mildly understated and `ratio_median` mildly overstated — but shared across
  all rows, so no comparison moves. Low value, recorded here rather than filed.)
- **The gauge constant leaks into the leg-B skill comparison** (28 obs get `−0.0753`, 8 get
  `fillna(0.0) = 0`). Recomputed three ways from the committed preds: as-run −0.0104, no-edge-obs-at-zero
  −0.0103, mean-centred −0.0103. Immaterial.
- **The abort's "per-obs discrimination is untouched: median Δap 0.0000"** (`DECISIONS.md:5519`) is partly
  structural — 6 of 21 observations are single-frame, so an additive offset cannot change their AP and
  `d_ap` is exactly 0 (`sd_off_pfree` ≈ 1e-8 for those rows), which is the identity
  `PLAN_H4_Leveling.md:48-57` said must not be cited as evidence of harmlessness. **But** restricting to
  the 15 observations whose offset actually varies within the obs leaves median Δap = +0.000066 (resid
  +0.000043, full −0.000061), so the conclusion stands and I am not filing it.
- **Gate 3's Δρ criterion is sign-agnostic**, so on the 7 of 26 tiles where the mosaic's THEMIS ρ is
  negative a merely noisier map scores an "improvement" (median Δ|ρ| there is −0.020 for `h1only`,
  −0.023 for `full`). But the improvement also holds on the 19 positive-ρ tiles (median Δρ +0.024 to
  +0.066) and an image-level sign test is significant for all four variants (p 0.0001–0.0094), so the
  gate-3 PASS is not an artifact of attenuation.
- **The F arm might be out-of-distribution relative to its own head** (H1 centres per-CROP at train time,
  per-FRAME at deploy — `PLAN_H4_Leveling.md:123-129`), which would make the abort's level comparison
  three-factor rather than R10's two. Checked: `reports/figures/f_h4_buildprep_median_stability.csv`
  measures within-frame ln-median drift at 0.007–0.056 against a between-frame spread of ~0.22
  (`DECISIONS.md:4618-4622`), a 4–30× margin, and the branch was resolved to per-frame centring on that
  evidence. Dead.
- **Pilot post-H4 η² 0.0505 sits below the pilot crop's rotation-null band (0.083–0.117), i.e. the
  leveling over-flattened past the geological floor.** The project already states the correct caveat —
  that comparing one field's η² to another field's null is meaningless because the null depends on the
  field's own autocorrelation (`DECISIONS.md:5027-5030`). Not re-filed.
  *(Pass 2: correct as far as it goes, but it retires only the narrow version. The surviving claim —
  that **no** pilot field's own null was ever computed, though `eta2_rotation_null` shipped a day before
  the pilot script existed, and that the bar itself is a constant below the only floor measured on that
  footprint — is now filed as `stats-fallacies-6`.)*

### Refuted in pass 2

- **The H4 pilot's λ sweep is inert, so "held-out edge-CV FLAT across λ ⇒ offsets generalize, not
  memorize" is a non-inference.** The flatness limb really is vacuous:
  `reports/figures/f_h4_leveling_summary.csv` shows `max_abs_offset` moving only **1.724 → 1.703** across
  λ = 0 → 1000 (1.2 %), so the "sweep" spans essentially one estimator and flatness cannot distinguish
  generalization from memorization; picking `λ* = 300` off that curve is arbitrary. **But** the load-bearing
  limb is the *held-out* comparison itself (0.0357 vs the unleveled 0.0738 at λ ≈ 0), which is legitimate,
  and pass 1 already covers the λ-selection-then-report circularity at build scale via `leakage-3`.
  Not filed.
- **Leave-one-edge-out CV on a 7-node/15-edge graph is nearly in-sample, so "the decisive non-circular
  gate" could not fail** (in-sample |Δp| 0.0338 vs held-out 0.0357 — a 5.6 % gap, i.e. dropping 1 of 15
  edges barely moves the solve). **The project found this itself**: `scripts/f_h4_lofo.py`'s docstring
  opens *"the honest generalization instrument the pilot lacked … the 2026-07-15 adversarial review flagged
  that PLAN_H4_Leveling §3.2 pre-declared ONLY leave-one-EDGE-out CV, which on the over-determined
  7-frame graph is nearly an in-sample check"*, LOFO was run in response, and the reopening call
  (2026-07-23) post-dates it. Dead.
- **The Tier-1 Fisher test's "honest exclusion" of the two un-annotated ObsIds is a result-driven analysis
  choice** — `docs/compositional.md:557-559` justifies it partly by its effect ("whose imputed-False value
  would inappropriately weaken the association"), which is a motivated-analysis red flag. It does not
  survive: complete-case deletion of a missing *covariate* is the statistically correct default, and the
  doc **discloses both** results (`:571-572`: impute-as-False gives P2 p = 0.034, P4 p = 0.10), so the
  forking path is visible and does not flip the P2 verdict. Only the uncertainty-propagation half is
  filed, under "Pass-2 corrections".
- **The H1/H2/H3 skill gate (Δ median per-image AUC ≥ −0.02 vs mosaic 0.786) has a tolerance at its own
  noise level** — `docs/model_evidence.md:115` describes that very metric as carrying "±0.1–0.2 fold-ripple
  error bars", and `DECISIONS.md:4629` records individual images moving −0.235/−0.185 while the median moved
  0.014, giving an SE-of-the-median of order 0.02. Real, but this is precisely the thesis of
  `stats-fallacies-3`/R41 applied to a second statistic, not a new defect; recorded here so the extension
  is not lost.
- **`docs/model_evidence.md` quotes prec@5 % = 0.948 without a prevalence baseline**, which would make it
  look far stronger than it is. It does not survive: `:107` states the base rate (0.36) and the no-skill
  line for PR-AUC in the same section, and R26 already covers the base-rate cap on the metric itself.
- **The H2/H3 → "not separable by ANY data-driven invariance instrument" generalisation
  (`PLAN_StripingArtifact.md:256-257`) is a universal claim from two negatives.** It is hedged in place —
  the sentence continues "*by any data-driven **invariance** instrument — but H4 (a post-hoc leveling
  instrument, orthogonal axis) succeeds where they failed*", so the plan states its own scope limit and
  immediate counter-example. The confound with a sub-floor target is filed as part of
  `stats-fallacies-6`(b) instead.

## Verified clean

- `src/striping.py:320-336` (`eta2`) — NaN-safe, between/total is computed correctly, and the docstring's
  admission that η² has no group-count correction is honest and is exactly what the rotation null exists
  to handle.
- `src/leveling.py:588-603` (`block_permute`) — moving whole ~4° blocks rather than plain permutation is
  the right null for a large-scale-trend test, and the docstring's justification is correct.
- `src/leveling.py:614` and `:644` and `:666` — the `(1 + #{null ≥ obs}) / (1 + n_draws)` permutation
  p-value is the standard finite-sample-valid form (never reports 0).
- `src/leveling.py:553-564` (`design_matrix`) — the claim that R² of a least-squares surface is invariant
  to affine rescaling of (x, y) is correct, so degrees-vs-metres genuinely does not matter here.
- `src/fgates.py:311-317` (`common_finite`) + `f_region_gates.py:78-84` + `f_map_compare.py:107-130` —
  the one-footprint discipline is real and is applied at both the row level *and* (in §5.1) the tile
  level, with the reason stated.
- `src/fgates.py:122-183` (`edge_cv_for_offsets`) — the per-variant fold logic is right: `resid` refits
  its plane inside the fold and `pfree` re-applies the constraint inside the fold, both of which would
  leak if done outside. (R19's mislabel of the fallback branch stands separately.)
- `src/fgates.py:247-266` (`pooled_skill`) — one `average_precision_score` over the pooled vector,
  `k = max(1, round(0.05·N))`, `argsort` (matching every number of record), no presence AUC.
- `gate5`'s decision to score Δ rather than absolutes, and to say so on the table
  (`f_region_gates.py:453-454`), is the correct handling of an in-sample head.
- `src/stage7d_pooled.py:135-149` (`cohen_d`) — pooled-variance denominator with `ddof=1`, sign
  convention documented and correct; `eligible_images` genuinely requires both classes.
- `docs/modeling_results.md:955-980` (§9.4) — paired image-level bootstrap CI + Wilcoxon, correctly
  interpreted ("every CI brackets zero"); this is the standard the F programme should have met.

### Verified clean in pass 2

- `src/stage7d_pooled.py:426-468` (`classify_image`) — the four-way category logic is exhaustive and the
  raw/partial gates are a genuine conjunction with the stringent gate first (`1e-3`, `|d| ≥ 0.20`), which
  is the conservative ordering; `no_signal` / `dust_attributable` / `composition_residual` are mutually
  exclusive and the docstring matches the code. (R15's separate point — that `inconclusive` is
  unreachable — stands and is unaffected.)
- `scripts/f_h4_themis.py:1-24` — the P3 THEMIS guard is correctly framed: pre-declared as *not degraded*
  rather than as a positive test, the decision quantity is Δρ (before/after on the same footprint and
  grid), the small-|ρ| expectation is stated up front ("the footprint is one ~75 km crop where the regional
  leg-1 signal was already weak"), Spearman is used because the THEMIS mosaic stores scaled brightness-temp
  DN not K, and the harness is cross-validated against the independent regional leg-1 value
  (unleveled 0.068 vs +0.07). The missing effective-*n* correction is R41's point, not a new one.
- `scripts/f_h4_lofo.py:1-24` — the docstring states the pre-declared LOFO gate *before* the numbers, names
  the review that demanded it, explains why edge-CV is nearly in-sample, and reports the cruder
  "drop-frame regret" as a cross-check against an earlier probe. This is the best-specified measurement in
  the F programme.
- `DECISIONS.md:5006-5030` (gate 1's re-scoping) — the three-scope table with each scope's **own** rotation
  null, the explicit "nothing can pass / nothing can fail" diagnosis, and the ruling that each row is scored
  against its own null are all correct, and the note that "comparing one field's η² to another field's null
  is meaningless" is the right caveat. The defect filed as `stats-fallacies-6` is that this reasoning was
  never applied backwards, not that it is wrong.
- `docs/model_evidence.md:105-118` — each headline metric is quoted with the thing it must be read against
  (PR-AUC vs the 0.36 base rate, prec@5 % as a top-5 %-of-ranked-tiles statement, median per-image AUC with
  a stated ±0.1–0.2 fold ripple). Prevalence dependence is disclosed where the numbers are published.
- `docs/compositional.md:553-577` — the Tier-1 cross-tabulation states the independence argument for the
  annotations, both partitions (significant and marginal), the exclusion rule, *and* the superseded
  impute-as-False numbers. The forking path is fully visible; only the OR's precision and the
  uncertainty in the attribution labels are unaddressed.

## Coverage note

Read in full: `src/fgates.py`, `src/stage7d_pooled.py`, `src/leveling.py:553-729` (the statistics half)
and `:214-320`, `src/striping.py:320-397`, `scripts/f_region_gates.py`, `scripts/f_h4_legb.py`,
`scripts/f_h4_level.py:90-120`, `scripts/f_map_compare.py`, `scripts/f_region_stagec.py:278-332` and
`:490-572`, `PLAN_FBuild.md` §4–§5.1, `PLAN_H4_Leveling.md`, `PLAN_StripingArtifact.md` PHASE 2,
`tests/test_leveling.py:280-345`. Grepped and spot-read `DECISIONS.md` around 4480–4760, 5000–5060,
5180–5200 and 5490–5600; `docs/compositional.md` §4/§7; `docs/modeling_results.md` §9.3–§9.5.

Re-derived numerically from committed artifacts: the leg-B pooled PR-AUC deltas and their 36-image
cluster bootstrap; the leg-B offset distribution vs the Stage-C offset distributions; the per-obs Δap
table split by within-obs offset variance; the gate-3 Δρ split by the sign of the mosaic ρ; the Stage-7d
image-level sign/Wilcoxon tests; the gate-1 rotation-null shift draws; and the per-component mean-zero
property of `f_h4_level.solve_offsets` on a synthetic 2-component graph.

**Not checked.** (a) Anything requiring the map rasters — I did not open `reports/map_fbuild/` or
`reports/map_region/`, so gate 1's η² and its nulls were audited from code and from the banked window
CSVs, never recomputed. (b) `scripts/probes/` (229 files) — the origin of many DECISIONS numbers,
including the H1/H2/H3 verdicts' underlying probes; I read only `_diag_within_image_deltas.py`. (c) The
H2/H3 FAIL verdicts' internal statistics (`f_h2_eta2.py`, `f_h3_pareto.py`) beyond what the plan tables
state — a FAIL is the conservative direction, so I deprioritised them, but the "H2+H3 ⇒ the artifact is
not separable from geology by *any* data-driven invariance instrument" generalisation
(`PLAN_StripingArtifact.md:256-257`) is a universal claim drawn from two negatives and was not audited.
(d) `docs/modeling.md`, `docs/model_evidence.md`, `docs/methods.md` and the Tier-1/Tier-2 compositional
provenance argument (Fisher's exact OR = 23.0) — the Fisher test itself is on 38 image-level annotations
and looks like the right unit of analysis, but I did not re-derive it. (e) The 2026-07-05d "embedder is a
5–20× amplifier" inference, which underpins the whole PHASE-2 docket.

### Pass-2 coverage note

Pass 2 read pass 1's file first and worked only its declared gaps, so it is **not** an independent
re-derivation of findings 1–5 — treat those as single-agent still.

Read in full: `scripts/f_pilot_crop.py` (the η²/overlap half), `scripts/f_h2_eta2.py`,
`scripts/f_h4_lofo.py:1-90`, `scripts/f_h4_themis.py:1-60`, `src/striping.py:320-352`,
`src/stage7d_pooled.py:390-468`, `PLAN_RegionalMap.md` §"5 legs" + the 2026-06-17b/2026-06-18 roll-ups,
`PLAN_StripingArtifact.md:195-275` (PHASE 2 in full), `ROADMAP.md:8-22`,
`docs/compositional.md:530-585`, `docs/model_evidence.md:100-120`. Grepped and spot-read
`DECISIONS.md` 4363-4470 (H2/H3 verdicts), 4600-4660 (P3 + LOFO), 4995-5060 (gate-1 re-scoping),
5184-5230 (H1). Verified against `git log`: `eta2_rotation_null` in `830a39b` (2026-07-02) vs
`f_pilot_crop.py` in `f3a7c50` (2026-07-03) and `f_h2_eta2.py` in `1f37fac` (2026-07-09).
Dumped headers/rows of `f_pilot_eta2_summary{,_minnaert_center}.csv`, `f_h2_eta2_summary.csv`,
`f_h4_leveling_summary.csv` to confirm no null column and the inert λ sweep.

Cross-checked for duplication against `docs/CODE_REVIEW_2026-07-31.md` §4/§4b/§4c/§4d/§5 (R01–R42) and
against the sibling area files — `notebooks.md:250-255` and `docs-consistency.md:125-155` both *pointed at*
the 0.05-bar question and explicitly deferred it to this file, which is why finding 6 was still open.

**Still not checked by either pass.** (a) The H2 nuisance basis's premise that a co-located embedding
difference is "pure frame-nuisance, zero geology" — co-registration error injects a geology gradient into
`d`, which would bias the PCA; not quantified (note R35's `fm-embeddings-1` covers a different defect in
the same pool). (b) The 2026-07-05d "embedder is a 5–20× amplifier" inference and the
`review_overlap_residual.csv` behind it. (c) `scripts/probes/` (229 files). (d) Legs 2/4/5 of
PLAN_RegionalMap, none of which has run. (e) Anything requiring the map rasters.
