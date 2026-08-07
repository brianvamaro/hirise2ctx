# Full-codebase review — 2026-07-31

> **Current-state correction (2026-08-06):** before using this register as a fixing queue, read
> [CODE_REVIEW_AUDIT_2026-08-06.md](CODE_REVIEW_AUDIT_2026-08-06.md). It records stale statuses,
> rejected fix alternatives, the unresolved test-mutation hazard, current product decisions, and the
> complete v2 rebuild dependency chain.

**Purpose.** A resumable, actionable defect register for the whole `hirise2ctx` repo. Written so a
future session can (a) pick up the *review* where it stopped, and (b) fix the findings without
re-deriving them.

**Provenance.** Multi-agent review at commit `da884c7` (clean tree). **R01–R22** (§4) were each
adversarially verified by an independent agent that read the cited code and tried to refute it — 8 of 20
candidates died that way, and several severities were corrected downward; the severities in §4 are
post-verification. **R23–R42** (§4b–§4d) come from per-area reviewers that self-refuted their own
candidates but were **not** independently verified — except **R23**, which I confirmed directly. Their
measurements are reproducible from committed artifacts; verify before acting.
**Update 2026-08-04:** the **15 high-severity live-path findings** have since been independently
verified — **0 died, 7 were downgraded** (§7). Any finding whose `Verified:` line carries a
`verify/<Rxx>.md` link is post-verification and can be acted on; the rest still cannot.
Baseline confirmed at review time: `pytest -m "not slow"` → **490 passed, 21 deselected** (~50 s).

**Scale.** 20 reviewers over ~46k LOC (`src/` 12.6k, `scripts/` 13.3k, `notebooks/_build_*` 11.8k,
`tests/` 8.3k) plus the docs; ~11M subagent tokens across four passes, three of which were partly killed
by session limits. §5 records what was investigated and **refuted**, so it is not re-filed.

> **Nothing in this document has been fixed.** It is a register, not a changelog. When you fix an
> item, change its `Status:` line to `FIXED <commit>` and record the change in `DECISIONS.md` as usual.

---

## 1. How to resume the review (idempotent protocol)

The review is split into **areas**. Each completed area has a file in
[docs/review_2026-07-31/](review_2026-07-31/). **An area is done iff its file exists.**

```bash
# 1. What is already done?  (ground truth — never trust an agent's return value)
ls docs/review_2026-07-31/

# 2. What is left?  Prints the exact list to hand to the reviewers:
for a in geo-crs labeling features evaluate modeling-heads calibration fm-embeddings \
         leakage numerics invariants tests stats-fallacies other-scripts \
         docs-consistency notebooks \
         probes-fm-recipe probes-w1-geospatial probes-stage6 probes-compression-targets \
         probes-tier2-calibration probes-stage7 probes-fbuild probes-utility \
         geo-crs-deep features-deep; do
  [ -f "docs/review_2026-07-31/$a.md" ] || echo "$a"
done

# 2b. Which high-severity live findings are still UNVERIFIED?
#     (protocol + scope list: docs/review_2026-07-31/_prompts_verify.md)
for r in R60 R61 R54 R32 R56 R24 R31 R36 R03 R48 R51 R44 R45 R37 R38; do
  [ -f "docs/review_2026-07-31/verify/$r.md" ] || echo "$r unverified"
done

# 3. Run those areas, 3-4 at a time. Give each subagent this instruction:
#      "Read docs/review_2026-07-31/_prompts.md. Follow section 1 (shared brief) and the
#       section 2 entry for area <area>. Write your findings to
#       docs/review_2026-07-31/<area>.md using the section 3 template, as your FINAL action."
#    Each reviewer persists its own file, so a session-limit kill loses only the in-flight
#    agents and never completed work. Re-running is safe: skip areas whose file exists.

# 4. Fold any new findings into section 4 of this file, and update the section 2 status table.
```

Batch in **3–4**, not 15: PASS 1 lost every in-flight agent to one limit hit. Small batches bound the
loss.

The per-area reviewer briefs are in [docs/review_2026-07-31/_prompts.md](review_2026-07-31/_prompts.md)
(the 15 `src/`-and-docs areas), [_prompts_probes.md](review_2026-07-31/_prompts_probes.md) (the 8
`scripts/probes/` areas) and [_prompts_verify.md](review_2026-07-31/_prompts_verify.md) (the
adversarial-verification protocol + its scope list) — all self-contained, so no session state is needed
to re-run them.

**Two kinds of work remain, and they are tracked separately.** *Discovery* (a new area → a
`<area>.md` file) and *verification* (an existing finding → a `verify/<Rxx>.md` verdict file).
**Both are now complete for their triaged scope** — all 31 areas, and all 15 high-severity live-path
findings (§7). What is left is verification of the ~58 findings that were *not* triaged as
decision-changing; see §6. Note the base rates differ sharply: the pass-1 sweep of unrefereed
candidates killed 8 of 20, whereas §7's sweep of author-self-refuted findings killed **0 of 15**.

**Why it is built this way:** PASS 1 lost 15 of 20 reviewers to Anthropic session limits, twice, and
because results only existed in the workflow's return value, ~4.8M tokens of review work evaporated.
Agents now persist their own output as their final action.

---

## 2. Area status

| Area | Scope | Status |
|---|---|---|
| `dataset-splits` | `src/dataset.py`, `src/config.py`, `src/manifest.py`, `scripts/run_stage5.py`, split integrity | **DONE** (verified) |
| `f-scripts` | `scripts/f_region_stage{a,b,c,d}.py`, `f_region_gates.py`, `f_map_compare.py`, sbatch | **DONE** (verified) |
| `mapping-striping` | `src/mapping.py`, `src/striping.py`, `src/validation_retrieve.py`, `map_*.py`, `striping_*.py` | **DONE** (verified) |
| `fcompose-fgates` | `src/fcompose.py`, `src/fgates.py` + their tests | **DONE** (verified) |
| `leveling` | `src/leveling.py`, `tests/test_leveling.py`, Stage-C consumers | **DONE** (verified) |
| `completeness` | files/areas no other reviewer opened | **DONE** (unverified) |
| `labeling` | `src/labeling.py`, `src/qa.py`, target construction | **DONE** (self-refuted) → [labeling.md](review_2026-07-31/labeling.md) |
| `features` | `src/features.py`, `spatial_features.py`, `colour.py`, `ctx_source_illumination.py` | **DONE** (self-refuted) → [features.md](review_2026-07-31/features.md) |
| `evaluate` | `src/modeling/evaluate.py`, `src/modeling/loaders.py` | **DONE** (self-refuted) → [evaluate.md](review_2026-07-31/evaluate.md) |
| `geo-crs` | `src/coregister.py`, `ctx_retrieve.py`, `ctx_tiles.py`, `hirise_imagery.py`, `pds_labels.py`, `ctx_edr.py`, `detections.py` | **DONE** (self-refuted) → [geo-crs.md](review_2026-07-31/geo-crs.md) |
| `fm-embeddings` | `src/fm_embeddings.py`, `src/modeling/mlp_head.py`, train/deploy parity | **DONE** (self-refuted) → [fm-embeddings.md](review_2026-07-31/fm-embeddings.md) |
| `modeling-heads` | `src/modeling/gbm.py`, `cnn.py`, `binary_target.py`, `inference.py`, `sweep_select.py` | **DONE** (self-refuted) → [modeling-heads.md](review_2026-07-31/modeling-heads.md) |
| `calibration` | `src/calibration.py`, `src/reliability.py`, `bank_calibration*.py` | **DONE** (self-refuted) → [calibration.md](review_2026-07-31/calibration.md) |
| `leakage` | cross-cutting LOIO-protocol integrity audit | **DONE** (self-refuted) → [leakage.md](review_2026-07-31/leakage.md) |
| `other-scripts` | `scripts/run_stage*.py`, `train_*.py`, `sweep*.py`, `parity_check.py` | **DONE** (self-refuted) → [other-scripts.md](review_2026-07-31/other-scripts.md) |
| `stats-fallacies` | cross-cutting inferential/logical validity audit | **DONE** (self-refuted) → [stats-fallacies.md](review_2026-07-31/stats-fallacies.md) |
| `invariants` | cross-cutting CLAUDE.md invariant compliance sweep | **DONE** (self-refuted) → [invariants.md](review_2026-07-31/invariants.md) |
| `numerics` | cross-cutting silent-failure / numerical-hazard sweep | **DONE** (self-refuted) → [numerics.md](review_2026-07-31/numerics.md) |
| `docs-consistency` | README/SHERLOCK_RUN command validity, DATA_DICTIONARY drift, doc-vs-code contradictions | **DONE** (self-refuted) → [docs-consistency.md](review_2026-07-31/docs-consistency.md) |
| `notebooks` | `notebooks/_build_*.py` logic-in-notebook, artifact drift, reproducibility | **DONE** (self-refuted) → [notebooks.md](review_2026-07-31/notebooks.md) |
| `tests` | test-suite integrity + coverage gaps | **DONE** (direct pass, commands quoted) → [tests.md](review_2026-07-31/tests.md) |

| `geo-crs-deep` | second pass: `coregister.py`, `ctx_retrieve.py`, `hirise_imagery.py`, `ctx_tiles.py`, `ctx_edr.py`, `pds_labels.py` | **DONE** → [geo-crs-deep.md](review_2026-07-31/geo-crs-deep.md) |
| `features-deep` | second pass: `features.py`, `spatial_features.py`, `colour.py`, `ctx_source_illumination.py` | **DONE** → [features-deep.md](review_2026-07-31/features-deep.md) |
| `labeling-deep-footprint` | second pass: is the detector footprint == the image footprint? (false zeros) | **DONE** — the footprint question is **REFUTED** → [labeling-deep-footprint.md](review_2026-07-31/labeling-deep-footprint.md) |
| `labeling-deep-artifact` | second pass: is the label artifact on disk what today's code produces? | **DONE** → [labeling-deep-artifact.md](review_2026-07-31/labeling-deep-artifact.md) |
| `labeling-deep-semantics` | second pass: what does the labeller *publish*, and can that statistic move? | **DONE** → [labeling-deep-semantics.md](review_2026-07-31/labeling-deep-semantics.md) |
| `labeling-deep-tests` | second pass: does `test_labeling.py` (668 lines) pin wrong science? | **DONE** — no, but it pins far less than it appears to (mutation-tested) → [labeling-deep-tests.md](review_2026-07-31/labeling-deep-tests.md) |
| `tests-deep-splits` | mutation-test `tests/test_splits.py` (399) — invariant 6 | **DONE** → [tests-deep-splits.md](review_2026-07-31/tests-deep-splits.md) |
| `tests-deep-features` | mutation-test `tests/test_features.py` (533) | **DONE** → [tests-deep-features.md](review_2026-07-31/tests-deep-features.md) |
| `tests-deep-within-image` | mutation-test `tests/test_within_image_split.py` (445) | **DONE** → [tests-deep-within-image.md](review_2026-07-31/tests-deep-within-image.md) |
| `tests-deep-region-staged` | mutation-test `tests/test_region_staged.py` (409) | **DONE** — the one suite WITHOUT the (0,0)-origin fixture defect → [tests-deep-region-staged.md](review_2026-07-31/tests-deep-region-staged.md) |

**All 15 `src/`-and-docs areas are complete.** Re-running any is wasted work unless the code changed.

### `scripts/probes/` — 184 files, 17.7k LOC

Reviewed separately because the whole directory was the first pass's biggest blind spot, and because
the question there is different: *not* "is this throwaway script clean" but **"did a number it computed
reach the record, and is that number right?"** Briefs: [_prompts_probes.md](review_2026-07-31/_prompts_probes.md).
Each area file carries a **Load-bearing map** table listing which of its probes are cited anywhere.

| Area | Scope | Status |
|---|---|---|
| `probes-fm-recipe` | `_w2_fang_*`, `_fm_freeze_window`, `_w2_cnn_verdict`, `_fm_parity_check` — **the frozen recipe's origin, incl. the `0.7832` headline computed in `_w2_fang_probe.verdict()`** | **DONE** → [probes-fm-recipe.md](review_2026-07-31/probes-fm-recipe.md) |
| `probes-w1-geospatial` | all 24 `_w1_*` + the CRS/SP1/vClaire diagnostics (~45 files) | **DONE** → [probes-w1-geospatial.md](review_2026-07-31/probes-w1-geospatial.md) |
| `probes-stage6` | `_sweep_stage6a/6b`, `_diag_stage6b_h3_check`, `_stage6c_gate{,_v2}` — the most-cited probes in the directory | **DONE** → [probes-stage6.md](review_2026-07-31/probes-stage6.md) |
| `probes-compression-targets` | compression diagnosis, target reformulation, W0 promotion, binary thresholds | **DONE** → [probes-compression-targets.md](review_2026-07-31/probes-compression-targets.md) |
| `probes-tier2-calibration` | `_diag_tier1/2_*`, `_fm_tier2_*`, `_fm_reliability_*` | **DONE** → [probes-tier2-calibration.md](review_2026-07-31/probes-tier2-calibration.md) |
| `probes-stage7` | Stage-7 compositional + terrain (feeds the reader-facing `docs/compositional.md`) | **DONE** → [probes-stage7.md](review_2026-07-31/probes-stage7.md) |
| `probes-fbuild` | `_f_leg_b_*`, `_f_review_overlap_residual` (which opened the H1–H6 docket), `_f02_diagnose` | **DONE** → [probes-fbuild.md](review_2026-07-31/probes-fbuild.md) |
| `probes-utility` | `_evidence_*`, `_modeling_slim_*` (published figures), smoke tests, fetchers | **DONE** (highest live-shipped yield) → [probes-utility.md](review_2026-07-31/probes-utility.md) |

Partial coverage of the PENDING areas exists from a direct (non-agent) pass — see §6 for exactly what
was and was not checked, so it is not re-done or wrongly assumed.

---

## 3. Priority order for fixing

> **This ordering predates the §7 verification pass (2026-08-04) and has not been re-sorted.** Seven of
> its entries were downgraded — **R38 / R36 / R03 / R37 / R44 / R32 high → medium**, R61 high → medium —
> while **R24 / R31 / R45 / R48 / R51 hold at high**. In particular **R38** (listed at 8 below) is no
> longer a "fix before building any A1 map" blocker: its blast radius largely evaporated and the η²
> confound it warned of never happened, though the one-line clip change is still right.
> Re-sort against §7 before working this list top-down.

**Externally visible first** — in a submitted PDF or a reader-facing writeup, so these are the only
findings with an audience outside the project:

0a. **R60** every number in the submitted `classification_slimmer.pdf` / `docs/modeling_slim.md` is on
    the **pre-sign-fix labels** (correcting them moves "usable" 14 % → 26 %; direction is favourable)
0b. **R61** the same PDF's ">90 % agreement" is ~10 points above random at that image's base rate
0c. **R62** the "exact same physical patch" figure is offset by the project's own measured 116.3 m shift
0d. **R51** `docs/modeling_results.md`'s "Bottom line" sign test counts 12 correlated re-analyses of 8
    images as independent · **R63** three more published-figure defects · **R59** the `docs/methods.md`
    size-audit table under-counts sub-threshold detections by 2.5–11×

**Then — they change numbers or decisions:**

1. **R23** two cohort images' labels are a score-rank truncation of the detection set (11.6 % of tiles) — **independently confirmed**. *Read **R56** first: the ruling that currently forbids the natural fix is itself a confounded comparison.*
2. **R54** the shipped abundance layer's calibration PASS is pooled; per image only **11 of 37** images are inside the declared band
3. **R32** the Tier-1 reference classifier early-stops on AUC and ships 1-tree boosters on 11 of 38 folds — *the FM-vs-Tier-1 margin rests on it*
4. **R55** the "5 m/px CTX ceiling, confirmed five ways" shares one embedding and trunk across all five — *this premise closed an improvement avenue*
5. **R36** the H4 leg-B skill gate could not have failed — it authorised ~265 CPU-h + 33 GPU-h
6. **R24** the S=128 Spearman that justified opening Stage 6a is a mean over 5 of 20 folds
7. **R03** HiRISE pixel-scale label confound — 15.8 % of the mosaic's level-error variance ·
   **R48**/**R49**/**R50** the other prevalence confounds · **R57** the Stage-7 GO numbers are DN, not I/F

**Then — live-path correctness:**

6. **R31** `extract_ctx_window` georeferences a cropped read with the un-cropped transform (live invariant-7 hazard)
7. **R01** mosaic tile-phase misregistration — shipped raster is wrong by 20–140 m per tile
8. **R38** A1's clip floor collides with the nodata sentinel (fix before building any A1 map)
9. **R02** `presence_auc` on the reported surface · **R25** mandated metrics computed then discarded
10. **R04** stale packaged splits are undetectable (+ `other-scripts-1`: the split hash has already drifted)

**Then — the record (cheap, and it is what the next session will read):**

11. **R37** README + SHERLOCK_RUN still instruct the next session to run the aborted F build
12. **R10 + R34** `DECISIONS` retraction #3 rests on a false premise — found twice, independently
13. **R33** the abort's `full` row measures calibrator clamping, not level · **R39** gate 5 is F-vs-F
14. **R12** the abort's decisive table has no committed producer
15. everything else, roughly in the order listed

**Cross-cutting theme worth its own decision.** Four separate findings are the same failure mode — a
gate that could not fail: **R36** (leg-B skill: near-constant offsets), **R11** (trend guard:
tautological on `pfree`), **`leakage-3`** (λ chosen as the argmin of the statistic gate 2 then tests),
**R40** (`trend_verdict` on raw R² while discarding its own nulls). Add **R41** (every tolerance is
±0.02 and no gated statistic has a sampling spread) and the programme's gate methodology, not just its
individual gates, needs a rule: *state what would falsify this before measuring, and report the
treatment's magnitude beside the metric delta.*

### Two systemic patterns — the most useful thing in this review

Individually these are 50-odd findings. Collectively, two failure modes recur often enough to deserve a
standing rule rather than a dozen separate fixes.

**Pattern A — a gate that could not fail.** The statistic is mathematically pinned by the very
construction it is meant to test, so the PASS carries no information. Six instances; three authorised
real spend or a whole programme:

| # | Gate | Why it could not fail |
|---|---|---|
| **R36** ⚠ | H4 leg-B skill gate | `(L+λI)o = AᵀWb` forces `mean(o_c) = 0` per component, so on a 21-component graph the between-obs level is zero *by construction*; 17 of 28 obs got one identical constant. — **CORRECTED by verification ([verify/R36.md](review_2026-07-31/verify/R36.md)): the gate is *monotone* in the applied differential and the same offsets ×2 give −0.0274 = FAIL. It was handed a ~5× attenuated treatment, not rendered inert. "Authorised ~265 CPU-h" also over-attributes — that figure is a probe-extrapolated midpoint, not measured spend.** |
| **R11** | Stage-C trend guard on `pfree` | `solve_offsets_planefree` zeroes span{1,lon,lat} exactly and the verdict reads only the order-1 surface → `NO_TREND` always |
| **R43** | Gate 1's absolute η² ≤ 0.05 bar | sits *below* the geological floor of the crop it was calibrated on → nothing can pass |
| **R49** | Stage 6c acceptance | the "bad image" label *is* the per-image base rate, so a prevalence oracle passes |
| **R52** | "isotonic drops Spearman/AUC" | a monotone map cannot raise a rank metric |
| `leakage-3` | Gate 2 λ selection | λ is chosen as the argmin of the statistic gate 2 then reports and tests |
| **R94** ⭐ | the two tests pinning "the verdict ships the right variant" | the `biases=[0.5,-0.5]` fixture makes `h1only`/`full`/`resid` **bit-identical** (measured max diff `0.000e+00`), so a mutant taking the headline from `variants[0]` survives. **Seventh instance — and the first found inside the test suite rather than in a gate**, which extends the pattern's reach: a *test* can be pinned by its fixture exactly as a *gate* is pinned by its construction. |

**Rule to adopt:** before declaring a gate, state what a FAIL would look like and verify that outcome is
*reachable* under the construction being tested — ideally by running the gate on a null/shuffled input
and confirming it fails. The project already does this correctly elsewhere (`st.eta2_rotation_null`,
`lv.block_permute`), so the machinery exists; it was just never pointed at the gates themselves.

**Pattern B — prevalence wearing another name.** A "signal" or "improvement" that is really the
per-image positive base rate. Five instances:

- **R48** — the CTX-source-heterogeneity "validated mechanism": `Spearman(pr_auc, base_rate) = +0.983`,
  and partialling out the base rate kills 10 of the 12 significant cells.
- **R49** — Stage 6c's reliability label *is* the base rate.
- **R50** — the `boulder_count` "+22 %" win is a change in the positive-class definition.
- **R26** — `precision@5%` is capped at `base_rate / 0.05`, then averaged unweighted over folds whose
  base rates span 0.0015–0.97.
- `notebooks-5` — the same comparison made across populations with different base rates.

**Rule to adopt:** report the base rate next to any cross-image or cross-target metric, and report any
claimed correlation with a per-image quality measure as a **partial** correlation controlling for
prevalence. `normalised_lift_at_top_k` already does exactly this and its docstring explains why — the
correction simply was not propagated to `precision@k` or to the reliability work.

**Pattern C — the gate is computed at the one aggregation level where the failure cancels.** Four
instances, and one is on the live shipped product:

- **R54** (live-shipped) — the abundance layer's `top_ratio 0.86` PASS is pooled; per image it is
  0.566 median / 0.168 p10, with **only 11 of 37 images inside the declared band**.
- **R33** — gate 6's `full` row is a pooled, rich-truth-conditioned ratio dominated by calibrator
  clamping; 45 % of its scored population is a constant.
- **R26** — `precision@5%` averaged unweighted over folds whose base rates span 0.0015–0.97.
- **R39** — gate 5 is F-vs-F by construction, so the abort's skill evidence has no cross-arm term.

**Rule to adopt:** any gate on a *per-place* quantity must be reported per place, with the pooled value
beside it, and the ruling on which one binds must be explicit. `DECISIONS.md:5049-5053` does make that
ruling for `top_ratio` (pooled binds) — the problem is that the per-image companion was then never
reported, so nobody saw how far outside the band the per-image figure was.

**Pattern D — the review (and the project) audited the *computation* and never the *artifact*.**
Added 2026-08-03 after two independent second-pass reviews reached this conclusion separately, which is
what makes it a pattern rather than a coincidence:

- `geo-crs-deep`: *"pass 1 audited what the module **computes** and never audited what the module
  **publishes**."* All 1,450 lines of affine/window/decimation arithmetic are sound — but
  `peak_correlation`, the only per-image quality number Stage 3 emits, is bounded below by the threshold
  it is screened against **and** is a fit statistic for a different model than the one applied. A cohort
  screening decision and a published figure rest on it.
- `features-deep`: *"under-reviewed at the artifact and semantics level, not the code level."* Pass 1's
  method — read each function, check the arithmetic against the docstring, count suspicious constants —
  finds R27 and R28 and **cannot** find "is the artifact on disk the one this code would produce today?".
  It checked the `features/*.json` sidecars for the post-fix thresholds, found all 38 clean, and stopped
  one directory short of the *derived* caches, which are two generations stale.

**Rule to adopt:** review three things, not one — (i) the computation, (ii) the statistic it publishes
and whether that statistic can move, and (iii) the artifact on disk versus what the current code would
produce. Items (ii) and (iii) need no code reading and were skipped by every first-pass reviewer.

A fifth, smaller theme: **R41** — every acceptance tolerance in the striping/F programme is ±0.02 and no
gated statistic was ever given a sampling uncertainty, so several PASS/FAIL calls sit inside the noise.

**One dependency worth flagging:** **R56** shows the "`min_confidence` filtering is harmful" ruling was
a two-factor comparison. That ruling is what currently forbids the natural fix for **R23** (harmonising
the cohort's confidence floor). Re-run that comparison with the target held fixed *before* deciding how
to fix R23 — otherwise the top-priority finding gets fixed against a confounded constraint.

---

R23 has since been **independently confirmed** (see its entry). R24–R53 live in
`docs/review_2026-07-31/*.md` with full measurements and self-refutation notes but are **single-agent
findings**; their numbers are reproducible from committed artifacts, so verify before acting.

---

## 4. Findings

Severity: `blocker` = invalidates a shipped number or verdict / crashes the live path · `high` = wrong
results in a plausible scenario · `medium` = wrong results in a narrow scenario, or a real protocol
defect with bounded impact · `low` = hygiene with teeth.

Liveness: `live-shipped` = the A1/mosaic regional map + frozen recipe · `live-active-plan` =
PLAN_RegionalMap · `dead-closed` = the aborted F build / closed striping work.

---

### R01 — Merged regional mosaic is misregistered tile-by-tile
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped · **Verified:** yes
- **Where:** [src/mapping.py:170](../src/mapping.py#L170) (`mosaic_geotiffs`), docstring
  [:159-162](../src/mapping.py#L159-L162)

A Murray tile is 47420 px wide and adjacent tile origins are exactly 47420 px apart, but
`47420 % 32 = 28`, so the **tile-anchored** 32 px lattices of adjacent Murray tiles are offset by
28 CTX px = 140 m = 0.875 of a 160 m coarse pixel. `rasterio.merge` floors each source's destination
offset (`merge.py` `win_align`: `math.floor(off + 0.1)`), so the displacement is one-sided
(west/north) and per-tile constant.

Measured on the shipped rasters: mosaic `c=-711136.371096145`, `res=159.9991835298017`;
`E4_N44 dcol=5927.500`, `E8_N44 dcol=7409.375`, `E16_N44 dcol=10373.125`, `E-12_N32 drow=4445.625`,
`E0_N40 dcol=4445.625 drow=1481.875`. For E-12_N44/E4_N44/E8_N44/E4_N32 the first 12×3 block of the
per-tile tif matches the mosaic at `floor(dcol+0.1)`, not `ceil`. **25 of 26 tiles' data are displaced
by 20–140 m.**

- **Failure scenario:** `regional_abundance_mosaic.tif` / `regional_prob_mosaic.tif` are the published
  deliverable *and* the `--match-mosaic` reference grid for
  [scripts/fetch_validation_data.py:79-84](../scripts/fetch_validation_data.py#L79-L84), so the THEMIS
  leg-1 Spearman ρ ([notebooks/_build_24.py:551-570](../notebooks/_build_24.py#L551-L570)) is computed
  against a per-tile-misregistered abundance field.
- **Not a problem, contrary to first report:** pixel *values* are untouched, and tiles never abut
  (1479 px tiles on a 1481.875 px stride leave an intentional 2 px nodata seam,
  [scripts/map_region.py:23-25](../scripts/map_region.py#L23-L25)), so `Resampling.nearest` never fires
  and there is **no** NN duplication/drop seam in the figures.
- **Fix:** anchor the coarse grid **globally** rather than per-parent-tile — derive `ti/tj` from the
  tile's global mosaic pixel origin (`round(c_tile/a)` is an exact multiple of 47420) so all Murray
  tiles share one 32 px lattice and `merge` becomes exact. No relabeling can fix it otherwise, since
  `47420 % 32 = 28`. **No-recompute stopgap:** in `mosaic_geotiffs`, assert
  `((src.transform.c - dst_w) / res) % 1 == 0` for every source and fail loudly (or `reproject` onto
  the target grid explicitly). Correct the docstring at :159-162 either way.
- **Note:** the displacement (≤140 m) sits inside the project's O(200 m) registration budget
  (CLAUDE.md invariant 2), and `DECISIONS.md:4985-4995` already verified per-tile sub-cell translations
  of 6.0–80.0 m in x / 7.9–50.3 m in y — which is the same phenomenon, recorded but not diagnosed.

---

### R02 — `presence_auc` is computed, aggregated and printed, against the project's own rule
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped · **Verified:** direct read
- **Where:** [src/modeling/evaluate.py:343](../src/modeling/evaluate.py#L343),
  [:399](../src/modeling/evaluate.py#L399), [:412-413](../src/modeling/evaluate.py#L412-L413),
  [:681](../src/modeling/evaluate.py#L681)

CLAUDE.md's reporting standard is *never report presence AUC* (`y_true > 0`). But
`per_fold_metrics` sets `out["presence_auc"] = presence_auc(y_true > 0, y_pred)`,
`aggregate_fold_metrics` emits `presence_auc_mean` / `presence_auc_std` into the standard summary
dict, and the per-fold log line prints `auc={m['presence_auc']:.3f}` — i.e. the number a human reads
off the console as "AUC" is the forbidden one, not `meaningful_auc`.

- **Failure scenario:** any consumer (notebook, doc, future session) that reads `presence_auc_mean`
  from a run's artifact JSON, or quotes the `auc=` figure from a training log, reports presence AUC in
  violation of the rule — silently, because the key is right there beside the legitimate metrics.
- **Fix:** keep the Mann-Whitney helper (it is correctly reused for `meaningful_auc` at
  [:366](../src/modeling/evaluate.py#L366)) but rename it (e.g. `_auc_mw`), drop `presence_auc` from
  `per_fold_metrics`/`aggregate_fold_metrics` output, and change the log line to print
  `meaningful_auc`. If the diagnostic is genuinely wanted, prefix the key
  `diagnostic_only_presence_auc` and exclude it from `write_run_artifacts`.
- **Follow-up owned by the `evaluate` area (PENDING):** grep for any *other* place presence AUC leaks
  into a reported artifact.

---

### R03 — HiRISE pixel scale is an unaccounted label confound, and it is a material part of the mosaic's level floor
- **Status:** OPEN · **Severity:** ~~high (if confirmed)~~ → **medium** · **Liveness:** live-shipped (the live object is the label basis in `dataset_v2/labels/`, *not* the F-abort level table) · **Verified:** [CONFIRMED-BUT-MIS-STATED](review_2026-07-31/verify/R03.md) — mechanism is *larger* than stated, but the 15.8 % headline fails (CI [0.003, 0.472]; does not replicate on the shipped head)
- **Where:** [config_v2.yaml:105](../config_v2.yaml#L105) (`min_size_m: 1.4105`), manifest column
  `MapPixel_mpp` in [hirise_40_vclaire.csv](../hirise_40_vclaire.csv)

`MapPixel_mpp` spans **0.25 m/px (13 images) / 0.50 m/px (24) / blank (2)** — the blanks being
`ESP_017355_2260` (the largest observation in the abort table, 13,457 tiles) and `ESP_076499_1160`.
**No code in `src/` or `scripts/` reads it**: the only writer is
[scripts/build_vclaire_manifest.py:304](../scripts/build_vclaire_manifest.py#L304), and
`src/colour.py` / `run_stage7a_fetch.py` use the PDS `.LBL`'s own `MAP_SCALE` for unrelated Stage-7
work. `min_size_m: 1.4105` is a single global floor = the 0.25 m/px design floor, so a 1.41 m boulder
is ~5.6 px across at 0.25 m/px but only ~2.8 px at 0.50 m/px, and BoulderNet's completeness at that
physical size is not the same in the two regimes.

The asymmetry itself **is documented and was deliberately deferred** — `DECISIONS.md:891` ("The
0.50 m/px images' own 5×5-px floor (6.25 m²) is **not** enforced under this global filter … a true
per-image filter … is deferred") and `DECISIONS.md:1355-1362` ("deferred until/if the
ESP_056165_2200 surviving-sub-threshold polygons turn into a modeling problem"). What was never
checked is whether that trigger has since been met. Measured on the 20 abort observations that have a
recorded pixel scale:

| arm | sd(log10 pred/label) | after removing pixel-scale group means | variance explained |
|---|---|---|---|
| **mosaic** | 0.1711 | 0.1570 | **15.8 %** |
| h1only | 0.3296 | 0.3292 | 0.2 % |
| resid | 0.3780 | 0.3690 | 4.7 % |
| pfree | 0.5421 | 0.4990 | 15.3 % |

Group gap (0.25 m/px vs 0.50 m/px) on the mosaic arm = **−0.148 dex**, two-sided permutation
p = 0.0825 (20,000 draws, n=20, null sd 0.0859); `Spearman(MapPixel_mpp, mosaic_ratio)` = +0.397
(p=0.083). Raw label level does **not** differ significantly by pixel scale (Mann-Whitney p=0.72), so
this is about the *model's* level calibration, not a crude label-magnitude shift.

- **Why it matters:** 0.170 is exactly the number `DECISIONS.md:5534` and `ROADMAP.md:18` cite as the
  incumbent mosaic map being "well calibrated against truth" and as the benchmark the F build failed
  to match. If ~16% of its variance is HiRISE pixel-scale label heterogeneity, part of that "floor" is
  label noise, not model error.
- **Important negative:** pixel scale explains only **0.2%** of `h1only`'s variance, so this does
  **not** rescue the F build or weaken the abort. It is a finding about the incumbent's label basis.
- **Fix / next step (cheap):** (a) recompute the level table with pixel scale as a covariate or as a
  stratum and report both; (b) fill the two blank `MapPixel_mpp` values from the PDS `.LBL`
  `MAP_SCALE` (`src/pds_labels.py` already fetches labels); (c) decide explicitly whether to enforce a
  per-image `min_size_m` (the deferred `_apply_detection_filters` extension in `DECISIONS.md:1355`) or
  to record in DECISIONS that the mixed-floor label basis is accepted, with this variance share as the
  quantified cost. Note `ESP_056165_2200`, the case the deferral was pinned to, is **not** in the v2
  38-image cohort, so the original trigger went moot without the underlying 2:1 scale mix going away.
- **Reproduce:** the measurement script is trivial — merge `hirise_40_vclaire.csv[MapPixel_mpp]` onto
  `reports/figures/fbuild_abort_level_vs_labels.csv` by `obs_id` and compute the group-mean-removed sd
  of `log10(<arm>_ratio)`.

---

### R04 — Stage-5 split failure is swallowed, and stale packaged splits are undetectable downstream
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped · **Verified:** yes
- **Where:** [scripts/run_stage5.py:58-71](../scripts/run_stage5.py#L58-L71),
  [:137-140](../scripts/run_stage5.py#L137-L140); [src/modeling/loaders.py:74-80](../src/modeling/loaders.py#L74-L80),
  [:112-168](../src/modeling/loaders.py#L112-L168)

`_run_one` wraps `build_split` + `write_split_metadata` in `try/except ValueError`, prints
`"{scheme}: FAILED to build ({e})"`, and returns `None`; `main` discards the return and
`return 0` unconditionally, so `raise SystemExit(main())` exits **0** on failure. Both guards that can
raise are live-data-driven: [src/dataset.py:469-472](../src/dataset.py#L469-L472)
(`stratification='none'` requires `n_folds == n_images`) and
[:442-447](../src/dataset.py#L442-L447) (within-image fold count), where the expectation comes from
`discover_obs_ids(labels_dir)` but `n_folds` comes from static config.

The consequential half is not the exit code (no in-repo caller reads it; there is no Makefile or
`.github/`) but that **nothing detects staleness**: `package_split` keys output on scheme name only
([src/dataset.py:640](../src/dataset.py#L640)), and `load_metadata`/`load_fold` read
`packaged/{scheme}/metadata.json` and its parquets without ever comparing `split_hash` /
`config_hash` / `obs_to_int` against `dataset*/splits/{scheme}.json` or the current inventory.

- **Failure scenario:** a cohort expansion adds images but the operator does not hand-edit
  `config_v2.yaml`'s `n_folds` (flagged as a manual post-Stage-4 edit at `DECISIONS.md:1211` and
  `:1266`, and it has already needed one correction). Stage 5 prints `FAILED to build`, exits 0, and
  leaves the previous cohort's `packaged/{name}/` in place; the next sweep trains and reports on the
  **old** folds with no warning.
- **Impact today:** nil — `dataset_v2/splits/loio_nfold.json` has `n_folds: 38` and
  `dataset_v2/labels/*.parquet` counts 38.
- **Fix:** have `main` collect `_run_one` results and `return 0 if all built else 1` (matching
  [scripts/run_stage1.py:75,84](../scripts/run_stage1.py#L75)); add a consumer-side guard in
  `loaders.load_metadata`/`load_fold` comparing the packaged `split_hash` to
  `dataset*/splits/{scheme}.json`'s and raising on mismatch. Optionally allow `n_folds: auto` for
  `stratification: none` / `within_image` so the count derives from the inventory.

---

### R05 — `sweep.py` and `run_modeling_slim.py` violate the torch/OpenMP import-order invariant
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped · **Verified:** direct read + observed warning
- **Where:** [scripts/sweep.py:31-32](../scripts/sweep.py#L31) vs [:38](../scripts/sweep.py#L38);
  [scripts/run_modeling_slim.py:36-39](../scripts/run_modeling_slim.py#L36-L39) vs
  [:44](../scripts/run_modeling_slim.py#L44)

CLAUDE.md invariant 9: any script using torch must `import src.modeling` **before** numpy/pandas, so
`src/modeling/__init__.py` can set `KMP_DUPLICATE_LIB_OK` and add torch's DLL directory before any
OpenMP-bearing DLL is loaded. `sweep.py` imports `numpy`/`pandas` at 31-32 and only reaches
`src.modeling.*` at 38-40; `run_modeling_slim.py` imports numpy/pandas/scipy/sklearn at 36-39 and
`from src import modeling` at 44 — where its own comment claims it *is* the "Windows OpenMP fix".
`src/modeling/gbm.py` pulls in LightGBM, which bundles its own OpenMP runtime alongside numpy's MKL.

Confirmed live: the fast test run emitted
`Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at the same time … can cause random
crashes or deadlocks`.

- **Fix:** move `import src.modeling  # noqa: F401` above the numpy/pandas block in both files, as
  ~40 other scripts already do. `scripts/train_gbm.py` is fine (its first third-party import is
  `from src.modeling.evaluate import …`, which triggers the package init).
- **Also check (`other-scripts` area, PENDING):** `scripts/bank_calibration.py` imports numpy/pandas
  at 19-20 with no bootstrap; it appears not to touch torch, so it is probably benign — confirm.

---

### R06 — "A1 is the shipped mitigation" is not backed by any artifact
- **Status:** OPEN · **Severity:** medium (documentation / deliverable definition) · **Liveness:** live-shipped · **Verified:** yes
- **Where:** `ROADMAP.md:19`, `README.md:101`; [scripts/striping_a1_map.py:59](../scripts/striping_a1_map.py#L59)
  (`DEFAULT_OUT = REPO / "reports" / "map_a1"`)

ROADMAP says PLAN_StripingArtifact closed with "**A1 is the shipped mitigation**" and "A1 stands as
the mitigation"; README:101 speaks of "shipping the A1 map + caveat". But **`reports/map_a1/` does
not exist**, and [scripts/f_map_compare.py:9-12](../scripts/f_map_compare.py#L9-L12) states outright
"There is **NO** A1 raster on disk at any extent", with A1 scoreable on only 9 of 26 tiles.
`DECISIONS.md:5102` confirms "**Not yet run:** … The A1 map and the A1 LOIO re-run are GPU steps."
What is on disk and published is `reports/map_region/` — the **un-mitigated** mosaic map (80 tifs).

- **Why it matters:** the closing narrative of a major programme names a deliverable that was never
  built, and A1's own measured cost is −0.024 AUC for a 28% η² reduction — so "ship A1" is a real
  trade someone has to actually choose.
- **Fix:** decide and record — either build A1 at the 26-tile extent
  (`scripts/striping_a1_map.py --all`, a GPU step) and re-point the deliverable, or amend
  ROADMAP:19 + README:101 to say the **mosaic** map ships with the striping artifact as a documented
  caveat and A1 remains an unbuilt option. See also **R07**, which must be fixed before any A1 map is
  generated.

---

### R07 — A1's train/deploy preprocessing statistic is inverted, and the docstrings assert the opposite
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed (blocks R06 if A1 is revived) · **Verified:** yes
- **Where:** [scripts/striping_a1_map.py:65-70](../scripts/striping_a1_map.py#L65-L70),
  [:93](../scripts/striping_a1_map.py#L93); [scripts/striping_a1_infer_crop.py:57-64](../scripts/striping_a1_infer_crop.py#L57-L64),
  [:88](../scripts/striping_a1_infer_crop.py#L88); training path
  `scripts/probes/_w2_fang_embed.py:202,209-211`

The A1 head (`models/deployable_a1`) was trained on embeddings whose (median, IQR) came from the
**native 5 m/px** Stage-2 CTX window (`_med, _iqr = a1_stats(arr)` where `arr` is `_load_ctx_window`'s
output). Both A1 inference paths instead derive (median, IQR) from `read_ctx_on_grid`, i.e. the CTX
**area-averaged to 160 m** ([src/striping.py:66-83](../src/striping.py#L66-L83),
`Resampling.average`), and then apply that gain to the *native* window DN. Since `a1_apply` maps
`(x-med)/iqr*27.7+125` clipped to [0,255], deployed windows carry native IQR
`27.7 * IQR_native/IQR_160m` rather than the 27.7 training pinned by construction.

The docstrings state the exact inverse: :16-17 "the head was trained against the 160 m statistics" and
:65-68 "deriving it from the native 5 m array instead gives different numbers and invalidates
models/deployable_a1" — and they disparage [src/striping.py:262-271](../src/striping.py#L262-L271)
`a1_normalize_per_frame` ("A1 at deploy"), which is the **train-consistent** one and is what
`PLAN_StripingArtifact.md:81` and `DECISIONS.md:4108-4109` document as the A1 deploy implementation.
(The `125.0/27.7` reference target *did* come from the 160 m CSV — median-of-medians 124.95, median
IQR 27.65 — but a shared reference *target* is not a shared source *statistic*.)

- **Impact:** biases the single banked A1 payoff number (η² 0.196→0.141, "28% reduction") and makes it
  non-comparable with the A1 skill number (−0.024 AUC) quoted beside it, which came from the
  native-stat store. Does **not** touch the shipped map (`scripts/map_region.py` +
  `models/deployable` apply no A1) and does **not** touch the F abort (`striping_a1_map.py` was never
  run; there is no A1 row in the §5.1 or abort tables).
- **Fix:** make both A1 inference paths use native-resolution per-frame statistics — call
  `src.striping.a1_normalize_per_frame(window.data, labels_nat)`, or accumulate per-frame native
  median/IQR by streaming the tile once — and correct the two docstrings. If the 160 m statistic is
  kept for cost reasons, record it in DECISIONS as a deliberate approximation and re-annotate the
  banked η² 0.141 as measured under a different A1 definition than the −0.024 skill cost.

---

### R08 — `a1_normalize_per_frame` leaves un-labelled and small frames at raw DN
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed (blocks R06) · **Verified:** direct read
- **Where:** [src/striping.py:262-271](../src/striping.py#L262-L271)

```python
out = arr.copy()
for f in np.unique(labels[labels >= 0]):
    sel = (labels == f) & (arr > 0)
    if sel.sum() < 50:
        continue
    ...
```

`out` starts as a copy of the **raw** array, and only pixels belonging to a frame with ≥50 valid
pixels are overwritten. So pixels with `labels == -1` (outside any SeamMap frame) and pixels in
frames under the 50-pixel floor keep their raw DN, and the returned array is a **mixture of
A1-normalized and un-normalized pixels** — re-introducing exactly the per-frame level offset A1
exists to remove, precisely at frame edges and mosaic gaps where the artifact is most visible.

- **Fix:** decide the contract explicitly. Either mask un-covered pixels to nodata (0) so they cannot
  contribute an un-normalized embedding, or fall back to the global reference `(A1_REF_MEDIAN,
  A1_REF_IQR)` for them, and record which. Also note the loop is O(n · n_frames) because
  `a1_apply(arr, …)` remaps the whole array once per frame before indexing — compute per-frame on the
  masked subset instead.

---

### R09 — `recipe_hash` collides across two different models, so the F head misreports its own metrics
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped (provenance) · **Verified:** direct read
- **Where:** `models/deployable/86c51a5dca220f63/recipe.json` vs
  `models/deployable_f_center/86c51a5dca220f63/recipe.json`

Both cards carry `recipe_hash: "86c51a5dca220f63"` and a **byte-identical `recipe` dict**, including
`loio_pooled_pr_auc: 0.7832` and `loio_med_per_image_auc: 0.7865`. But they are different models:
`n_train_images` 38 vs **36** (the F head lacks `ESP_066634_2210` and `ESP_071093_2210`), trained
2026-06-14 vs 2026-07-07, on different inputs (mosaic DN vs H1-centered log-minnaert), and only
`model_hash` differs. The F head's true pooled PR-AUC on the F path is **0.7438**
(`reports/figures/fbuild_gates.json`, gate 5 `F_h1only`), not 0.7832. `models/deployable` also has a
`parity_ref.npz`; `models/deployable_f_center` has **none**, so the F head was never parity-checked.

- **Failure scenario:** the artifact layout keys on `recipe_hash`, so the two heads are distinguished
  only by their parent directory name. Anyone reading `deployable_f_center`'s card attributes the
  frozen recipe's validated skill to a model that never achieved it; and pointing a map run at the
  wrong parent produces no complaint (same hash, same recipe block).
- **Fix:** include the embedding store name / input mapping and the sorted train-obs list in the
  recipe hash, or at minimum stop copying `FROZEN_RECIPE`'s measured metrics into a derived head's
  card — write that head's own measured numbers, or `null`. Add a `parity_ref.npz` for any head that
  is used to produce a map.

---

### R10 — The abort's mosaic-vs-F comparison is two-factor, and its causal attribution overreaches
- **Status:** OPEN · **Severity:** medium (record correctness; does **not** overturn the abort) · **Liveness:** dead-closed · **Verified:** direct measurement
- **Where:** `ROADMAP.md:18`; `DECISIONS.md:5524-5550`, retraction #3 at `DECISIONS.md:5560-5561`

The mosaic arm of the decisive level table uses `models/deployable` (38 train images); the four F arms
use `models/deployable_f_center` (36). So the comparison changes the **head**, its **training set**,
*and* the **input radiometry** simultaneously. Decomposing the published numbers:

| step | sd(log10 pred/label) | Δ | what changed |
|---|---|---|---|
| mosaic | 0.170 | — | — |
| h1only | **0.328** | **+0.158** | F input path + different head, **no leveling** |
| resid | 0.371 | +0.043 | leveling only |

**79 % of the mosaic→resid gap predates any leveling.** `ROADMAP.md:18` quotes only "mosaic **0.170**
vs resid 0.371 / pfree 0.532" and attributes root cause to "the … within-frame ramp, materialised" —
a leveling-specific mechanism that cannot explain the dominant term. `DECISIONS.md:5535` **does**
carry the h1only 0.328 row and states the 1.13×/1.62×/1.26×-vs-unleveled decomposition correctly, so
the defect is in ROADMAP's summary, not the log.

Separately, `DECISIONS.md` "Corrections to earlier readings" #3 **retracts** the hypothesis that the
calibrator/head structurally favoured h1only, on the stated ground: "*There are no metadata JSONs in
`models/deployable{,_f_center}/`; unsubstantiated.*" The `recipe.json` cards **are** there, one
directory down, and they document the 38-vs-36 training-set difference — so the retraction rests on a
false premise, and the alternative it dismissed is partly substantiated.

- **What this does NOT do:** overturn the abort. Gate 5 (pooled PR-AUC Δ −0.030 resid / −0.186 pfree)
  and gate 6 (`full` top_ratio 8.74) are independent of the level table, and R03 shows pixel-scale
  heterogeneity explains 0.2% of h1only's variance, so it is not a hidden confound either.
- **Also worth recording:** all 21 observations are in **both** heads' training sets, so mosaic 0.170
  is an **in-sample** level-coherence figure. Symmetric between arms, so the comparison is fair, but
  it is not a held-out number and should not be read as deployment accuracy.
- **Fix:** (a) amend `ROADMAP.md:18` to quote the h1only row and split the gap into
  input-path+head (+0.158) vs leveling (+0.043); (b) correct retraction #3, citing the two
  `recipe.json` cards; (c) if F is ever revisited, the cheap disambiguating control is to score
  `deployable_f_center` on **mosaic** inputs (or `deployable` on H1-centered inputs) — it is absent
  from the artifacts; (d) label 0.170 as in-sample wherever it is quoted.

---

### R11 — Two §0.1 abort guards were scored on the wrong solve, and both would have argued for ABORT
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed · **Verified:** yes, both re-derived
- **Where:** [scripts/f_region_stagec.py:452](../scripts/f_region_stagec.py#L452),
  [:432](../scripts/f_region_stagec.py#L432), [:476-489](../scripts/f_region_stagec.py#L476-L489),
  [:510-511](../scripts/f_region_stagec.py#L510-L511); `src/leveling.py:699` (`trend_verdict`)

**(a) The trend guard is tautological on `pfree`.** `solve_offsets_planefree`
([src/leveling.py:289-292](../src/leveling.py#L289-L292)) constrains span{1, lon, lat} to zero
exactly, and `trend_verdict` reads only the **order-1** surface, so it can only ever return
`NO_TREND` for pfree. Confirmed: unweighted order-1 R² is exactly 0.000000 (the banked 0.0078 is pure
degree-weighting slop), and `tests/test_leveling.py:470` pins the tautology. The **order-2**
companion `tr2` is computed at :302, printed, and **dropped** — `fbuild_trend_guard.csv` carries
`quad_r2`/`quad_p` only for the free solve. Re-running it on the same banked offset vector
(`reports/figures/fbuild_stagec_offsets.csv`, w=degree, cell_deg 4.0, 1000 block-permutation draws,
seed 0) gives **R² = 0.3858, p = 0.0030 vs null p95 = 0.2477** — a significant smooth region-wide
field of ~1.1 logits sd. `DECISIONS.md:5481` records "trend verdict | NO_TREND" for pfree as an
evidence column with no note that it is true by construction.

**(b) Guards 3 and 4 are computed only on `o_star`, the free solve**, and emitted under
*unsuffixed* column names beside the suffixed `offset_logit_pfree`. Guard 4 is not merely uncomputed
but **mis-reported for the shipped variant**: banked
`corr(offset, frame-mean P(rich)) = +0.0497 / +0.0713` reads as a clean pass against the pilot's −0.94,
whereas the same statistic on `offset_logit_pfree` is **−0.4309 / −0.4290** (and −0.3879 / −0.4920 on
`resid`), with `n_big_offset_normal_radiometry` = **565** on pfree vs the banked 756 for the free
solve. The free solve's 22.7-logit ramp was swamping the mean-flattening signature.

- **Direction:** both corrections push *toward* ABORT (a truthful trend guard would have flagged
  pfree; a truthful guard 4 would have flagged it too), so the verdict is safe. No automated consumer
  is misled — `f_region_staged.py:66-72` reads only the three offset columns + `offset_source`, and
  `f_region_gates.py:378-383` reads only verdict/apply/needs_ruling/lambda_star. The misled consumers
  are human readers of `fbuild_stagec_offsets.csv` / `fbuild_trend_guard.csv` and the DECISIONS record.
- **Fix:** score constrained solves on a surface the constraint did not delete (pass `tr2` for
  constrained variants, or take the more significant of tr1/tr2, and use that surface's `fitted` as
  the `smooth` field fed to attribution); emit `pfree_quad_r2`/`pfree_quad_p`/`pfree_null_p95_r2` and
  annotate `pfree_linear_*` as zero-by-construction. Move the guard-3/4 block inside a per-variant
  loop and suffix the columns. Guard 3 for pfree additionally needs `lv.lofo_offsets` to accept a
  solver argument, since it re-solves with the free `solve_offsets` at
  [src/leveling.py:481](../src/leveling.py#L481).

---

### R12 — The abort's decisive evidence has no committed producer, and mixes footprints
- **Status:** OPEN · **Severity:** low (hygiene; numerically immaterial) · **Liveness:** dead-closed · **Verified:** yes, impact bounded
- **Where:** `reports/figures/fbuild_abort_level_vs_labels.csv`, `_per_obs_skill.csv`,
  `_level_per_tile.csv` + two PNGs; referenced only as prose at
  [notebooks/_build_28.py:61](../notebooks/_build_28.py#L61)

`grep -rn fbuild_abort` over the repo hits only `_build_28.py` and `28_f_verdict.ipynb`, and neither
has a code cell that reads them (contrast §1-§9, which all `pd.read_csv` their inputs). Commit
`41a6f26` modified six `.py` files, none of which write these paths. The tables that closed three
plans and ~265 CPU-h + 33 GPU-h of work were generated by uncommitted ad-hoc code — a CLAUDE.md
invariant-10 violation on the project's most consequential decision.

It also mixes footprints: the mosaic arm is scored on all **95,606** labelled cohort tiles while the
four F arms are scored on the **89,145** where the F map is finite — the entire deficit being
`ESP_017355_2260` (6,996 of 13,457, because the F rasters are NaN outside Stage-D coverage). This is
the same one-footprint rule `scripts/f_region_gates.py:78-84` enforces for gate 1.

- **Impact: immaterial.** Matching footprints moves sd(log10 pred/label) by ≤0.0005 (mosaic
  0.1744→0.1746, pfree 0.5449→0.5446), because the mosaic numerator shrinks alongside the label
  denominator (0.030062→0.029421 vs 0.022253→0.021636). Every published 3-decimal value is unchanged.
  The analysis **is** fully auditable from committed artifacts: all five headline sd values re-derive
  exactly as the population sd of `log10(ratio)` over the committed per-obs CSV, and the mosaic column
  re-derives to 16 digits from `reports/map_region/*_abundance.tif` + `fbuild_cohort_join.parquet`.
- **Related, same table:** the `full` row silently drops **2 of 21** observations —
  `ESP_042964_2160` and `ESP_059421_2170` have `full_pred` exactly 0.0, so `log10` is undefined. The
  table is captioned "95,606 tiles, 21 obs" for every row, but `full`'s 0.412 / 81.3× are over 19.
  mosaic/h1only/resid/pfree reproduce over 21; `full` only over 19. `full` was never the shipping
  candidate, so no verdict changes — but the caption is wrong for that row.
- **Fix:** commit the ~40-line generator as `scripts/f_abort_level.py` (aggregation helper in
  `src/fgates.py`) so the table regenerates from `fbuild_cohort_join.parquet` +
  `reports/map_region/*_abundance.tif`; apply `fg.common_finite` across the mosaic and all four F
  columns before taking per-obs means; emit `n_matched`/`coverage` columns so a future coverage hole is
  visible in the output; and emit `n_used` per arm so a dropped-observation row cannot hide.

---

### R13 — Context-box nodata is never checked; only the own 32 px tile is
- **Status:** OPEN · **Severity:** low (real, but ~4 orders rarer than first claimed) · **Liveness:** live-shipped · **Verified:** yes
- **Where:** [src/mapping.py:76-96](../src/mapping.py#L76-L96) (`own_tile_zero_fraction`),
  [:256-257](../src/mapping.py#L256-L257); [src/fm_embeddings.py:195-199](../src/fm_embeddings.py#L195-L199)

`predict_window` masks on the own 32×32 block's zero fraction, but the embedding is taken from the
3×3 (96 px) context box, which is sliced with **no** zero test. A tile whose own block is fully valid
can sit at the edge of a Murray mosaic gap with most of its context box at DN 0; those pixels dominate
the bicubic-resized 224 px input and the GeM pooling, and no training patch looks like that (training
context boxes came from gap-free CTX windows around HiRISE footprints). Neither
`scripts/map_region.py` nor `scripts/striping_a1_map.py` adds a context-level check, so the sidecar's
`n_masked_nodata` under-reports untrustworthy tiles.

- **Fix:** compute the zero fraction over the 96 px context box (or return it from
  `slice_context_boxes`) and mask on both, with the context threshold recorded in the sidecar. Also
  reconsider `max_zero_fraction=0.5` — a tile half nodata currently still gets a prediction.

---

### R14 — Regional-map resume trusts file existence, and GeoTIFFs are written non-atomically
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped · **Verified:** no (completeness pass)
- **Where:** [scripts/map_region.py:141-143](../scripts/map_region.py#L141-L143),
  [:157-159](../scripts/map_region.py#L157-L159), [src/mapping.py:181-194](../src/mapping.py#L181-L194)

`if prob_tif.exists() and not args.force: skip` — a truncated `_prob.tif` from a killed or OOM'd job
is indistinguishable from a complete one, and `write_geotiff` writes straight to the final path with
no temp-file rename. The same pattern applies to per-window partials (`part_path.exists()`), though a
truncated `.npz` at least fails loudly in `np.load`.

- **Fix:** write to `*.tif.tmp` and `os.replace` on success; or validate on resume (open the raster,
  check `height`/`width` against the expected shape and that the last block reads). Same for the
  per-window `.npz` partials.

---

### R15 — Stage-7d `classify_image` can never return `inconclusive`, and two docs report that as a result
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped module, PARKED programme · **Verified:** direct read (proof below)
- **Where:** [src/stage7d_pooled.py:462-468](../src/stage7d_pooled.py#L462-L468)

```python
if not raw_pass:                      return "no_signal"
if partial_pass:                      return "composition_residual"
if raw_pass and not partial_pass:     return "dust_attributable"
return "inconclusive"                 # <- unreachable
```

After line 462 `raw_pass` is truthy; after 464 `partial_pass` is falsy; so at 466 the condition is
`True and True` and always returns. Line 468 is dead, and no test covers it. The docstring at
[:449](../src/stage7d_pooled.py#L449) documents `inconclusive` as "ambiguous (only some features
pass)" — a category the function cannot emit — and `docs/compositional.md:692` ("**No images were
labelled `inconclusive`**; 16 of 26 fall into `no_signal`…") plus `DECISIONS.md:2132` ("0
`inconclusive`") report a structural impossibility as an empirical finding.

The substantive part: the documented 4-way taxonomy
(`docs/compositional.md:375`, `PLAN_Compositional.md:382`) is really **3-way with a hidden
any-one-feature-suffices OR rule** — an image where only one of several features passes raw is called
`dust_attributable` or `composition_residual` with no ambiguity flag.

- **Fix:** either implement the intended ambiguity rule (e.g. require a majority of `features` to pass,
  and return `inconclusive` when some-but-not-most do) and re-run the attribution table, or delete the
  category from the code, the docstring, `docs/compositional.md:375`, `PLAN_Compositional.md:382`, and
  the notebook's `ordered_cats`/colour map ([notebooks/_build_16.py:291-293](../notebooks/_build_16.py#L291-L293)),
  and strike the "no images were labelled inconclusive" sentences as vacuous. Parked programme, so the
  substantive Stage-7 conclusions are unaffected either way — but the claim as written is false.

---

### R16 — Stage B's per-frame uint8 clip fraction is never measured
- **Status:** OPEN · **Severity:** low (a missing diagnostic, not a bug) · **Liveness:** dead-closed · **Verified:** direct read
- **Where:** [scripts/f_region_stageb.py:52](../scripts/f_region_stageb.py#L52),
  [:66-80](../scripts/f_region_stageb.py#L66-L80) (`map_uint8`), sidecar at [:244-247](../scripts/f_region_stageb.py#L244-L247)

The I/F→uint8 stretch bounds are **globally fixed** (`STRETCH_LO, STRETCH_HI = 0.8400, 1.1170`) with
no per-frame or per-window percentile stretch — this is the correct design, and the only per-frame term
is the deliberate H1 median division. **Recorded here so it is not re-flagged.**

But the fixed ln-range is only `ln(1.1170/0.8400) = 0.285`, so any pixel whose centered ratio falls
outside ±~12 % is hard-clipped to 1 or 255, and the **fraction clipped is frame-dependent** — a
second-order per-frame nuisance that a DC (median) centering cannot remove. Stage C measures *logit*
saturation (`lv.edge_saturated_frac`) and Stage D measures *abundance* saturation, but nobody measured
the **input** clip. It is a plausible unexamined contributor to the F arm's behaviour and a one-line
addition.

- **Fix:** in `map_uint8`, accumulate `n_clipped_low`/`n_clipped_high`/`n_valid` and write them into
  the per-frame sidecar JSON; then check whether the clipped fraction correlates with the per-frame
  offsets or with the per-obs level ratios.

---

### R17 — Retained `src/leveling.py` items (all low, but the module is explicitly kept as "generally useful")
- **Status:** OPEN · **Severity:** low · **Liveness:** dead-closed, retained for reuse · **Verified:** yes

- **R17a — `frame_level_spread` computes `logit(mean p)`, not `mean logit(p)`.**
  [src/leveling.py:424-431](../src/leveling.py#L424-L431) is handed Stage-B's `prob_mean`
  ([scripts/f_region_stagec.py:456](../scripts/f_region_stagec.py#L456) ← `f_region_stageb.py:246`
  `float(prob.mean())`). Measured on 150 real Stage-B npzs: p5..p95 of `logit(mean p)` = 3.35 vs
  `mean logit` = 4.35 — a **1.31× Jensen compression** (not an `EPS` artifact; zero tiles fall below
  1e-4). So the banked ×frame-spread ratios (6.58 / 2.26 / 1.52) are ~30 % inflated and
  `n_over_frame_spread` (525 / 32 / 37) up to ~6×. Harmless now — one scalar divides all three
  variants so every comparative reading is invariant, nothing reads these fields, and the "23.6-logit
  ramp is physically impossible" argument survives at 4.35 (5.4× instead of 7.3×). **Fix:** Stage C
  already holds the per-frame tile logits, so pass `np.array([float(l.mean()) for l in logits])`, or
  have `frame_level_spread` accept already-logit input; then re-emit
  `fbuild_stagec_lean_guards.csv` (Stage C reruns in ~2 min from the edge cache) and correct
  `DECISIONS.md:5477` + `PLAN_FBuild.md:298`.
- **R17b — `solve_offsets_planefree`'s "the other n−3 are fit exactly as before" is provably false.**
  [src/leveling.py:280-281](../src/leveling.py#L280-L281). It would require `Zᵀ·AᵀWA·(I − ZZᵀ) = 0`,
  which a weighted graph Laplacian does not satisfy — verified numerically on a random 8-node weighted
  graph built with the same `normal_equations` assembly (max |ZᵀMP| = 36.8, n−3 coefficients move 15 %
  relative) — and it is self-contradicted two lines later and by
  [src/fgates.py:169](../src/fgates.py#L169) ("re-solving free and detrending afterwards would score
  `resid`, not `pfree` … SSR 4.65e7 vs 5.83e7"). **The same false claim is propagated into
  `DECISIONS.md:5472` and `PLAN_FBuild.md:297`** ("903 of 906 directions still fit exactly"), which
  matters more than the docstring. Also [src/leveling.py:305](../src/leveling.py#L305) still calls
  pfree "what Stage C ships as of 2026-07-30" with no mention of the abort or retraction #4.
  **Fix:** state that the plane is a data-determined direction the solve declines to estimate (only the
  per-component constant is a true gauge) and that constraining it also changes the remaining n−3
  coefficients — which is exactly why pfree ≠ resid; fix the two docs; drop the stale "ships" line.
- **R17c — `block_permute`'s docstring overclaims.** It says it "keeps within-block structure" while
  `rng.choice(pool, replace=False)` reorders values inside the destination block, mildly biasing the
  null R². Documentation only.

---

### R18 — Stage B resume can skip frames and race two array tasks onto one output
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed · **Verified:** mechanism yes, consequences unproven
- **Where:** [scripts/f_region_stageb.py:205-206](../scripts/f_region_stageb.py#L205-L206)

The balanced-resume (commit `3585826`) snapshots the `undone` list **per task** and then strides it.
With staggered array-task starts the residue classes shift, so the six per-task lists are not a
partition: two tasks can pick the same frame (racing onto one output path) and a frame can fall
through every task's stride. Completion is detected by **file existence** only.

- **Unresolved:** whether the 906/907 hole came from this race or from a genuine Stage-A failure. The
  Slurm submission history needed to tell is not in the repo.
- **Fix:** compute the partition from a **deterministic** key (`hash(frame) % n_tasks == task_id`, or
  a pre-written assignment manifest) rather than from a live `undone` snapshot; and gate completion on
  a sidecar marker written after a successful close, not on the output file's existence.

---

### R19 — `edge_cv_for_offsets` would label full-model numbers as `resid`/`pfree`
- **Status:** OPEN · **Severity:** low (unreachable via the pipeline) · **Liveness:** dead-closed · **Verified:** yes
- **Where:** [src/fgates.py:148](../src/fgates.py#L148) vs [:152-155](../src/fgates.py#L152-L155),
  [:182-183](../src/fgates.py#L182-L183)

`out` snapshots `variant` **before** the `lon is None or lat is None` fallback rebinds it to
`"full"`, and the fallback branch sets no `note` (only the `h1only` branch does), so the persisted
CSV/JSON would attribute a full-offset number and its PASS/FAIL to a variant that was never scored —
exactly the failure the docstring at :126-132 says it exists to prevent. Unreachable in practice: the
only producer of the Stage-C table writes lon/lat unconditionally
([scripts/f_region_stagec.py:506](../scripts/f_region_stagec.py#L506)). And
`tests/test_fgates.py:162-174` is *named* `test_edge_cv_resid_refits_its_plane_per_fold` and its
docstring claims it checks the fallback "must SAY it fell back" — but it passes `lon=`/`lat=`, so the
branch is never exercised.

- **Fix:** set `out["variant"]` after the fallback (or add `note="fell back to full: no lon/lat"`), and
  add a test that calls it with `lon=None, lat=None` and asserts the returned label/note.

---

### R20 — Stage D keeps a frame's provenance code when it zeroes a non-finite offset, and maps unknown codes to "solved"
- **Status:** OPEN · **Severity:** low · **Liveness:** dead-closed · **Verified:** partially (downgraded to a hygiene nit)
- **Where:** [scripts/f_region_staged.py:194](../scripts/f_region_staged.py#L194),
  [:210](../scripts/f_region_staged.py#L210)

A NaN offset is silently coerced to 0.0 (the frame is composited unleveled) while `src_code` stays
whatever Stage C wrote — so the H6 `offset_source` raster, whose purpose is per-pixel offset
provenance, reports "solved" for pixels that received no offset. Second path:
`fc.OFFSET_SOURCE_CODE.get(str(...), 0)` maps any unrecognised provenance string to **0 = the best
severity** rather than to "none" (3) or an error, so a renamed label in a future Stage C reads as
fully solved. Unreachable with the repo's only writer (`lv.patch_graph_holes` emits exactly
solved/component_gauged/interpolated).

- **Fix:** default the `.get` to the worst code (or raise), and set a distinct provenance code when an
  offset is coerced to 0.

---

### R21 — Stage B never verifies the per-frame CRS it records, while hardcoding the Mars radius
- **Status:** OPEN · **Severity:** low · **Liveness:** dead-closed · **Verified:** no
- **Where:** [scripts/f_region_stageb.py:233](../scripts/f_region_stageb.py#L233)

Stage B reads each frame's CRS into the sidecar but never checks it against the expected CTX
equirectangular clon_0 / 3396190 m, while separately hardcoding the radius — CLAUDE.md invariants 1-2
say unknowns must be verified at runtime and a CRS mismatch must fail loudly.

- **Fix:** assert the frame CRS matches the expected WKT (or that the residual is O(200 m)) and fail
  loudly; take the radius from the dataset rather than a literal.

---

### R22 — Latent: within-image streaming iterators lack the kind-dispatch `package_split` has
- **Status:** OPEN · **Severity:** low (dormant API, zero callers) · **Liveness:** unused · **Verified:** yes, downgraded from "high leakage"
- **Where:** [src/dataset.py:547-583](../src/dataset.py#L547-L583) vs
  [:632](../src/dataset.py#L632)

`_assign_within_image_kfold` writes `test_obs_ids == train_obs_ids == [obs_id]` (deliberate, and pinned
by `tests/test_within_image_split.py:239-244` per `PLAN_Stage5c.md:262`). `package_split` dispatches on
`metadata["kind"] == "within-image"` and partitions by quadrant; `iter_train_batches` /
`iter_test_batches` do **not**, and call `_join_one_image(obs)` with no quadrant filter — so a
within-image fold would yield byte-identical train and test frames.

**Not leakage in any reported result:** repo-wide grep finds **zero** call sites in `src/` or
`scripts/`; `DECISIONS.md:759-761` and `PLAN_Stage5.md:10-11` record the streaming path as "wired in
but unused" with a ~50-**image** switch trigger (the cohort is 38); every within-image number came from
`scripts/sweep_within_image.py` reading the packaged path.

- **Fix:** raise `NotImplementedError` at the top of both iterators when
  `metadata.get("kind") == "within-image"`, or implement the quadrant mask properly from
  `fold["quadrant_definitions"]` + `fold["test_quadrant"]`; add a test asserting a within-image fold's
  train and test frames are disjoint.

---

## 4b. Findings from the per-area reviews (PASS 2)

> **Verification caveat.** R01–R22 above were adversarially verified by an independent agent. R23–R30
> below come from single-agent passes that **self-refuted** their own candidates but were not
> independently checked. The measurements they quote are reproducible from committed artifacts — verify
> before acting, especially R23. Full detail, evidence, self-refutation notes and coverage limits live
> in the linked area file; only the summary is duplicated here.

### R23 — Two cohort images' labels are a score-rank truncation of the detection set, documented as benign density hygiene
- **Status:** OPEN · **Severity:** high · **Liveness:** live-shipped · **Verified:** ✅ **INDEPENDENTLY
  CONFIRMED 2026-07-31** — I re-derived it from the source `.dbf`s and the cached GPKGs. The split is
  exact in all three images: **zero** kept rows score below the dropped maximum, and
  `min(kept) − max(dropped)` = +1e-6:

  | ObsId | src rows | null-geom | kept | dropped score max | kept score min | kept rows below dropped max |
  |---|---|---|---|---|---|---|
  | ESP_017355_2260 | 1,105,447 | 745,514 | 359,933 | 0.6173 | 0.6173 | **0** |
  | ESP_046803_2325 | 658,290 | 291,150 | 367,140 | 0.4734 | 0.4734 | **0** |
  | ESP_068483_2280 | 1,057,153 | 329,993 | 727,160 | 0.4067 | 0.4067 | **0** |

  All three dropped sets start at score 0.1000 — the same floor every *unaffected* image reaches
  (checked 6 of them: score min 0.1000, p1 ≈ 0.253). So the null-geometry rows are not sparse export
  noise; they are the entire low-score tail, and these three images are labelled at a
  0.41–0.62 confidence floor while the other 36 are at 0.10.
- **Where:** [src/detections.py:112-127](../src/detections.py#L112-L127) (`drop_null_geometries`), called
  at [:202](../src/detections.py#L202); consumed blind at
  [src/labeling.py:460](../src/labeling.py#L460); documented as benign at `DECISIONS.md:1194-1201`
- **Detail:** [labeling.md](review_2026-07-31/labeling.md) finding `labeling-1`

Three of the 39 vClaire exports carry DBF rows with no polygon, dropped at Stage 1. `DECISIONS.md:1194`
records this as "BoulderNet emits many null-geometry records **at this density**" — a scale artifact
whose removal is hygiene. The reviewer's measurement says otherwise: the null rows are **exactly the
lowest-scoring detections**, separated by a sharp per-image score threshold (dropped-max == kept-min to
3 dp in all three images), so the surviving polygon set is a high-confidence subset at a *different*
cut per image:

| ObsId | raw rows | kept | dropped | kept `score` min |
|---|---|---|---|---|
| ESP_017355_2260 | 1,105,447 | 359,933 | **67.4 %** | 0.617 |
| ESP_046803_2325 | 658,290 | 367,140 | 44.2 % | 0.473 |
| ESP_068483_2280 | 1,057,153 | 727,160 | 31.2 % | 0.407 |
| other 36 images | — | — | **0.00 %** | ~0.10 |

`min_confidence: null` means no confidence filter is applied, so 36 images are labelled at
`score ≥ 0.10` while two of the 38-image cohort sit at `≥ 0.407` and `≥ 0.617`
(`ESP_046803_2325` is already excluded from the sweep). Estimated from retention curves on six
unaffected images, `ESP_017355_2260`'s `fractional_area` is **2.5–4.5× too low** and
`ESP_068483_2280`'s ≈1.4–1.7× too low relative to the cohort basis — **11.6 % of the 161,005 S=32
tiles**, including the largest observation (13,457 tiles).

- **Why it matters:** `pr_auc@1e-2` and `precision@5%` are prevalence-dependent, and the per-image
  *level* of two images is multiplicatively wrong — the exact quantity the striping/F programme spent
  months measuring. Nothing in the code, the sidecars or the docs distinguishes those images.
- **Does NOT change the abort verdict** (the reviewer checked): `sd(log10 mosaic_ratio)` over the 21
  abort observations is 0.1744, 0.1755 without `ESP_017355_2260`, and 0.1791 under a ×2.5 label
  correction. R10 and R03 stand.
- **How the "benign" framing got into the record — the diagnostic stopped one column short.**
  `scripts/probes/_diag_vclaire_source_nulls.py` asked exactly the right question — its docstring is
  *"Are the null geometries present in the SOURCE shapefile, or introduced by reproject?"* — and ran it
  on exactly the two affected cohort images (`ESP_017355_2260`, `ESP_068483_2280`). It printed row
  counts, null/empty counts, and an `is_at_edge` breakdown split by null-vs-non-null. It **never looked
  at `score`**:

  ```python
  # scripts/probes/_diag_vclaire_source_nulls.py — the only per-column breakdown it computes
  if "is_at_edge" in g.columns:
      null_mask = geom.isna()
      print(f"  is_at_edge among null:    ...")
      print(f"  is_at_edge among nonnull: ...")
  ```

  So the probe correctly established "the nulls are upstream, not a reprojection bug", that was read as
  "upstream export artifact, safe to drop", and nobody asked whether the dropped rows were a *biased*
  subset. One extra line — `g.loc[null_mask, "score"].describe()` — would have shown the rank
  truncation immediately. This is the specific mechanism by which `DECISIONS.md:1194` came to call it
  density hygiene, and it is worth recording as a lesson: *when a filter drops a third to two-thirds of
  the rows, characterise the dropped population on every available column, not just the one you
  suspect.*
- **Verify like this:** join each source `.dbf` to its cached GPKG on `id` for the three ObsIds and
  compare `max(score)` of dropped rows against `min(score)` of kept rows. If they coincide, the
  truncation is real.
- **Fix:** in `drop_null_geometries`, when `n_dropped > 0` record the dropped-vs-kept score
  distributions in the Stage-1 sidecar and **fail loudly** (or set a `label_basis_truncated` flag) when
  `max(score_dropped) <= min(score_kept)` — the signature of a rank truncation rather than sparse
  nulls. Then either re-export those folders or harmonise the cohort by setting `min_confidence` to the
  max of the per-image kept-score minima (0.617) and regenerating Stage 4. Record the choice in
  DECISIONS either way.

### R24 — The S=128 Spearman that justified Stage 6a is a mean over 5 of 20 folds
- **Status:** OPEN · **Severity:** high (unchanged) · **Liveness:** live aggregator, closed number · **Verified:** [CONFIRMED](review_2026-07-31/verify/R24.md) — every number reproduces exactly; blast radius bounded to 5 cells across all 236 committed `metrics.json`, only S=128 material. **The register's headline fix would not have prevented the error** — see the verdict file.
- **Where:** [src/modeling/evaluate.py:390-394](../src/modeling/evaluate.py#L390-L394) (`mean_std`),
  [:397-399](../src/modeling/evaluate.py#L397-L399), [:423-426](../src/modeling/evaluate.py#L423-L426),
  [:65](../src/modeling/evaluate.py#L65) (`spearman_safe` → NaN on constant `y_pred`)
- **Detail:** [evaluate.md](review_2026-07-31/evaluate.md) finding `evaluate-1`

`mean_std` drops NaN fold values and returns the surviving count, but **only the Spearman call site
keeps it** (`spearman_n`); every other key discards it, while the dict advertises `n_real_folds`
alongside — so a reader reads `<metric>_mean` as a mean over `n_real_folds`. `spearman_safe` returns
NaN when the *model's* prediction is constant, so the dropped folds are exactly the ones where the
model had zero ranking skill. Dropping them is directly optimistic.

At S=128 the within-image quadrants have 41–101 test tiles and the two-stage hurdle emitted a single
constant prediction in **15 of 20 folds**. `metrics.json` says `spearman_n = 5`, `n_real_folds = 20`.
Scoring those 15 at ρ = 0 gives **0.101** — *below* S=8's 0.118, collapsing the monotone ladder that
`docs/modeling_results.md:1077` reports as 0.406 and `PROMOTION_QUEUE.md:191-192` cites as the
"**indirect evidence**" for opening Stage 6a.

- **Blast radius (checked):** the 38-fold `loio_nfold` runs have `meaningful_auc`/`pr_auc` defined on
  37–38/38, so the full-v2 LOIO comparisons survive. The damage is concentrated in the small-fold
  within-image sweeps.
- **Fix:** emit `f"{key}_n"` for every aggregated key, and count constant-prediction folds separately
  (`n_degenerate_pred`) rather than sharing the NaN channel with genuinely undefined folds. Then correct
  `docs/modeling_results.md` §10.2 and retract the "0.26 → 0.41" claim and its restatements in
  `PROMOTION_QUEUE.md` and `DECISIONS.md:1605`.

### R25 — The classification aggregator computes the mandated metrics and throws them away, leaving ROC-AUC (= presence AUC on `bc_ge_1`) as the only aggregate
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped · **Verified:** no
- **Where:** [src/modeling/evaluate.py:496-524](../src/modeling/evaluate.py#L496-L524) vs the regression
  aggregator at [:415-426](../src/modeling/evaluate.py#L415-L426); per-fold values computed and
  discarded at [:469-474](../src/modeling/evaluate.py#L469-L474)
- **Detail:** [evaluate.md](review_2026-07-31/evaluate.md) finding `evaluate-2`

`per_fold_metrics_classification` computes `pr_auc`, `normalised_lift` and
`precision/recall_at_top_{1,5,10}pct` per fold, but `aggregate_fold_metrics_classification` has no
counterpart to the regression H1 loop, so every classification artifact — including the frozen
recipe's task — surfaces ROC-AUC as its sole discrimination number and CLAUDE.md's mandated
`pr_auc@1e-2` / `precision@5%` are absent. With `BINARY_TARGET_ID = "bc_ge_1"`
([scripts/sweep_within_image.py:58](../scripts/sweep_within_image.py#L58)) that ROC-AUC **is** presence
AUC under another name — which `src/modeling/binary_target.py:61-63` says in a comment. This is why the
frozen headline `pooled_pr_auc 0.7832` had to be recomputed by a bespoke `verdict()` in
`scripts/probes/_w2_fang_probe.py` rather than read from the artifact: two independent implementations
of the same headline metric. Related to **R02** but a different function, key and caller set.

- **Fix:** add the H1 aggregation loop (with an `_n` per key, per R24), and either retire `bc_ge_1` from
  `sweep_within_image.py` or rename its aggregate key so the presence semantics are unmissable.

### R26 — `precision@5%` is hard-capped by the fold's base rate, and the unweighted fold-mean is reported as a quality level
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped · **Verified:** no
- **Where:** [src/modeling/evaluate.py:288-311](../src/modeling/evaluate.py#L288-L311); contrast
  [:271-285](../src/modeling/evaluate.py#L271-L285) (`normalised_lift_at_top_k`, which fixes exactly
  this for the sibling metric) · **Detail:** [evaluate.md](review_2026-07-31/evaluate.md) `evaluate-3`

`k = round(0.05·n)` and the metric returns `tp / k`, so when `n_pos < k` it is capped at
`base_rate / 0.05` — a perfect ranker on a 1.3 %-base-rate image scores 0.26. The lift metric's own
docstring diagnoses the mirror-image problem and divides it out; precision@k has no such correction,
and the aggregator averages raw values with equal fold weight. On the frozen scale/target the rich-tile
base rate spans 0.0015–0.97 and **10 of 38 folds sit below 5 %**: one tier-2 cell reports
`precision_at_top_5pct_mean = 0.5906` against a mean attainable ceiling of **0.8711**, so ~30 % of the
shortfall from 1.0 is base rate, not model quality. Arm-vs-arm *deltas* at the same scale/target share
the ceiling and are unaffected; the pooled headline `prec@5% 0.948` is computed over the concatenated
vector at ~0.35 base rate and is also safe. The defect is in the reported *level* and in any
cross-scale/cross-target/cross-cohort comparison.

- **Fix:** emit `precision_at_top_5pct_normalised = precision / min(1, base_rate / k_frac)` beside the
  raw value, or at minimum emit `meaningful_base_rate` into the aggregate so the ceiling is visible.

### R27 — `lacunarity_shadow_b*` emits `0.0`, an out-of-range sentinel, on 21.2 % of S≥32 tiles, and Stage 6a averages it as a real value
- **Status:** **CODE FIXED 2026-08-06 — REBUILD PENDING.** Reproduced exactly as filed before fixing: 42,015 / 198,320 = 21.2 % of S ≥ 32 rows in `dataset_v2/features/` are exactly `0.0`, every one with `shadow_fraction == 0`, smallest non-zero value exactly 1.0, nothing in (0, 1). Downstream measured at 2.16 % of `nbr_mean_lacunarity_*` rows pooled in the impossible interval (0, 1), worst image `ESP_068402_2240` at 16.7 % (the register cited 12.6 % for one file; `ESP_076499_1160` is 13.2 %). `_lacunarity_per_tile` now leaves the NaN prefill in place; `dataset/DATA_DICTIONARY.md` documents it; two regression tests added. Artifact impact recorded in [PENDING_REBUILD.md](PENDING_REBUILD.md) row 2. · **Severity:** medium · **Liveness:** dead-closed for the shipped map; live for every GBM/W1 number off `dataset_v2/features/` · **Verified:** **yes — reproduced read-only against `dataset_v2` before the fix**
- **Where:** [src/features.py:422](../src/features.py#L422) (producer),
  [src/spatial_features.py:100-105](../src/spatial_features.py#L100-L105) (consumer),
  `dataset/DATA_DICTIONARY.md:278-284` · **Detail:** [features.md](review_2026-07-31/features.md) `features-1`

Gliding-box lacunarity is `≥ 1` by Cauchy–Schwarz, but a tile with no shadow pixels returns `0.0`
instead of `NaN` — while everywhere else in Stage 4b "not computable" is `NaN`, and Stage 6a's
neighbour aggregation is NaN-aware but **not** sentinel-aware (`np.isfinite`). Measured: 42,015 /
198,320 = **21.2 %** of S≥32 rows are exactly `0.0`, every one with `shadow_fraction == 0`, minimum
non-zero exactly `1.0`, and **no** row in `(0, 1)` — proving it is a sentinel. Downstream,
12.6 % of one `features_nbr` file's rows have `nbr_mean_lacunarity_shadow_b2` in the impossible
interval `(0, 1)`. LightGBM can split away a `0.0` in the base column, but no split can undo an
average of a sentinel and a measurement.

- **Fix:** return `np.nan` (the array is already NaN-prefilled, so the `else` branch can just be
  dropped) and document the case; regenerating Stage 6a is required for the `nbr_*_lacunarity_*` columns.

### R28 — Canny thresholds are a fixed fraction of the dtype range, not adaptive; the config asserts the opposite
- **Status:** **CODE FIXED 2026-08-06 — REBUILD PENDING.** Landed: `use_quantiles` plumbed through `_compute_canny_window` (hard error if enabled without explicit percentile thresholds), the false config comment replaced in both `config.yaml` and `config_v2.yaml`, `dataset/DATA_DICTIONARY.md` corrected, five regression tests added. **Default switched to `use_quantiles: true`, `0.80 / 0.90`** (Brian, 2026-08-06) in `DEFAULT_FEATURES_CFG` and both YAMLs, with a test asserting they agree (a YAML `features:` block overrides key-by-key). Artifact impact: [PENDING_REBUILD.md](PENDING_REBUILD.md) row 3 -- every `edge_*` value and its six `nbr_*` derivatives change at the batched rebuild. Measured, all read-only: Spearman **ρ = 0.965** (register said 0.894) between per-image `edge_density` and `intensity_std` over 38 images, **12.2×** spread, **33.8 %** of `ESP_068402_2240`'s S=64 tiles with zero edges; on a synthetic scene a ~3× DN-spread cut collapses edge density **×0.01** while quantile thresholds hold at **×1.00**. New trap found while fixing: an explicit `low_threshold=0.1` is **not** equivalent to `None` — skimage divides explicit thresholds by `dtype_max`, so on a uint8 window it becomes 0.1/255 and passes nearly every gradient. · **Severity:** medium · **Liveness:** dead-closed for the shipped map; live for the GBM matrix and the W1 error atlas · **Verified:** **yes — mechanism reproduced synthetically and in the cohort**
- **Where:** [src/features.py:199-208](../src/features.py#L199-L208), `config.yaml:149-150`,
  `dataset/DATA_DICTIONARY.md:297` · **Detail:** [features.md](review_2026-07-31/features.md) `features-2`

`low_threshold=None` passes through to `skimage.feature.canny`, which with `use_quantiles=False` maps
`None` to the **constants 0.1 / 0.2** on the `img_as_float` image — an absolute gradient threshold in DN
units, not a distribution-derived one. `config.yaml:149`'s comment says "None -> skimage chooses from
gradient magnitude", the opposite. So `edge_density` / `edge_orientation_entropy` partly measure how
much radiometric contrast the CTX frame happens to have: per-image `edge_density` correlates with
per-image `intensity_std` at Spearman **ρ = 0.894** across the 38 images, a 12× cohort spread, with
**33.9 %** of one low-gain image's S=64 tiles having zero Canny edge pixels. That is the same
"dead feature across a whole image" failure the project found and fixed for `shadow_fraction`
(DECISIONS 2026-06-10, worth +0.249/+0.127 meaningful AUC) — never checked for the canny family, and it
is the per-frame-radiometry-into-features mechanism the striping programme spent months on.

- **Fix:** set `use_quantiles=True` with explicit quantile thresholds (or derive them per tile), correct
  the config comment and the data dictionary, and re-run the dead-feature audit for the canny columns.

### R29 — Stage-3 co-registration shifts the polygons but not the coverage mask, so a ~1-tile strip inside every swath edge is zero by construction
- **Status:** OPEN · **Severity:** low · **Liveness:** live-shipped · **Verified:** no (magnitude analytic, not measured)
- **Where:** [src/labeling.py:474-478](../src/labeling.py#L474-L478),
  [:85-93](../src/labeling.py#L85-L93); mask producer
  [src/ctx_retrieve.py:459-531](../src/ctx_retrieve.py#L459-L531) ·
  **Detail:** [labeling.md](review_2026-07-31/labeling.md) `labeling-2`

Stage 4 translates every detection polygon by the Stage-3 `(dx, dy)` and then gates eligibility with a
coverage mask reprojected from the **unshifted** HiRISE product. The shift is a whole-product
geolocation offset, so a strip of width `|shift|` on the receding side of each swath stays
`eligible = True` while no detection can land in it. Measured shifts over 39 images: median 194.7 m
(max 327.3 m), `dy` northward in 38 of 39 — i.e. ~1–2 rows of 160 m tiles along the southern boundary
get `fractional_area = 0` over ordinary terrain. Estimated ~2 % of tiles per image, always at the edge
and always the same sign, so it also biases any edge-vs-interior diagnostic.

- **Fix:** translate the mask by the same `(dx, dy)` before gating (shifts are already quantised to CTX
  pixels), or — cheaper and strictly conservative — erode the eligible mask by `ceil(|shift| / px)`.

### R30 — The invariant-2 CRS gate has no production caller; 38 of 39 v2 images were ingested without it
- **Status:** OPEN · **Severity:** low · **Liveness:** live-shipped · **Verified:** no
- **Where:** [src/qa.py:45-116](../src/qa.py#L45-L116) (`assert_centroid_consistent`); no `src/` or
  `scripts/` caller — only two tests and one notebook with a hardcoded ObsId ·
  **Detail:** [labeling.md](review_2026-07-31/labeling.md) `labeling-3`

CLAUDE.md invariant 2 says the residual HiRISE↔CTX offset "must fail loudly" if it comes out in km, and
`PLAN_NewDetections.md:463-467` lists `qa.assert_centroid_consistent` as acceptance check #1 for **each**
new image of the vClaire cohort. It was never wired into `scripts/run_stage1.py`, the driver added for
v2 — which calls `stage1_one_image` and prints the SP1 status, nothing more. The de-facto backstop is
the Stage-3 block-median correlation, which did lock on 38 of 39 images (the one failure,
`ESP_046803_2325`, was correctly excluded) — which is why this is low. It survives because CLAUDE.md and
a plan's acceptance list assert the guard runs, and it does not.

- **Fix:** call it inside `det.stage1_one_image` (it already has the manifest row and the reprojected
  gdf) or in `run_stage1.py::_reproject_one`, and record `distance_m` in the Stage-1 sidecar.

**Lower-severity items in the area files, not promoted here:** `features-3` (the degenerate-window
fallback in `_compute_dn_thresholds` lacks the main path's protections and can still return
`shadow = 0`), `features-4` (Stage 6b joins the SeamMap with no CRS check and discards the CRS it
reads), `features-5` (`inference.py`'s "the seam is genuinely clean" claim about Stage 4b is false in
both halves), `evaluate-4` (`per_bin_rmse`'s top bin is labelled `1e-2_to_max` but hard-capped at 1.0,
so 40 committed count-target runs silently drop 66–88 % of their tiles with no partition assertion).

---

## 4c. Findings from the per-area reviews (PASS 3)

> Same verification caveat as §4b: single-agent, self-refuted, **not** independently checked. Full
> evidence in the linked area file.

### R31 — `extract_ctx_window` stamps a silently-cropped read with the *un-cropped* transform, so a window overhanging its Murray tile is georeferenced kilometres off
- **Status:** OPEN · **Severity:** high (unchanged) · **Liveness:** live-shipped (the only Stage-2 window path) — but **dormant**: no active plan re-runs Stage 2 · **Verified:** [CONFIRMED](review_2026-07-31/verify/R31.md) — proved by synthetic-raster experiment (1500 m west overhang → transform 1500 m too far west). **The proposed alternative fix does not work** — see the verdict file.
- **Where:** [src/ctx_retrieve.py:433-434](../src/ctx_retrieve.py#L433-L434) (+ `:425-432`, `:442-447`),
  caller [:585](../src/ctx_retrieve.py#L585) · **Detail:** [geo-crs.md](review_2026-07-31/geo-crs.md) `geo-crs-1`

`src.read(window=…)` with `boundless=False` **crops** the window to the dataset, but
`src.window_transform(window)` is called on the *original* window. When the requested window overhangs
the tile's north or west edge, the crop moves the data's start while the transform keeps the negative
offset — so the output GeoTIFF holds real CTX pixels from *inside* the tile stamped with the overhang's
coordinates. Nothing compares `actual_bounds` to `requested_bounds`, so it is recorded as provenance
rather than raised. (East/south overhang is silently truncated instead — lesser, still unflagged.)

Measured on a real instance, `ESP_057469_2215`: `from_bounds` gives `col_off = -1924, width = 2128`;
the crop reads tile columns 0…204 while the transform writes `c = -9619.95`. The cached window is
**100 % real imagery** (mean DN 88.6, range 25–159) georeferenced **9,620 m too far west**, and its
HiRISE mask marks 917 pixels covered — so Stage 4 would have emitted tiles whose CTX texture comes from
9.6 km east of their labels.

- **Why it matters despite zero current impact:** that ObsId is excluded, and 48 of 49 cached windows
  are fully inside their tile, so no shipped v2 number is wrong. But it is a live **invariant-7**
  hazard: adding a manifest row whose footprint crosses a Murray tile edge silently yields wrong data
  rather than an error, and a ~50/50 straddle keeps a normal-looking `hirise_coverage_fraction` (~0.5)
  and passes every existing check while every tile is misregistered by kilometres.
- **`DECISIONS.md:397-419` records the incident with the wrong mechanism** — "the strip is entirely
  WEST of x=0 … **reads as zero pixels**". The pixels are not zero. Because the entry concluded the
  output was empty, the deferral ("not fixing this in Stage 2 now") was taken against a benign failure
  mode that does not exist.
- **Fix:** derive the transform from the **cropped** window (`src.window_transform(window.crop(...))`,
  or re-read `actual_bounds` from the returned array shape), and assert
  `actual_bounds ≈ requested_bounds` — failing loudly otherwise, per invariant 2. Correct the
  DECISIONS entry.

### R32 — The Tier-1 reference classifier early-stops on AUC against its own docstring and plan, shipping 1-tree boosters on 11 of 38 LOIO folds
- **Status:** OPEN · **Severity:** ~~high~~ → **medium** · **Liveness:** live-shipped (default classification head) — but nothing in the deployed map/head depends on it · **Verified:** [CONFIRMED-BUT-MIS-STATED](review_2026-07-31/verify/R32.md)
- **Where:** [src/modeling/gbm.py:419](../src/modeling/gbm.py#L419) (`metric` list),
  [:436](../src/modeling/gbm.py#L436) (`lgb.early_stopping` with default `first_metric_only=False`),
  contradicting [:381-384](../src/modeling/gbm.py#L381-L384) and `PLAN_Stage5b.md:136-138` ·
  **Detail:** [modeling-heads.md](review_2026-07-31/modeling-heads.md) `modeling-heads-1`

`fit` sets `metric = ["binary_logloss", "auc"]` and attaches `lgb.early_stopping(rounds)` with
`first_metric_only` defaulting to **False**. LightGBM's callback loops over every (dataset, metric)
pair and raises on the **first** that stalls, so valid-set AUC co-governs both the stop and the selected
`best_iteration` — and `model_to_string()` truncates to `best_iteration`. The class docstring and the
plan both state the opposite: "Early-stopping metric is `binary_logloss`, **not AUC** — AUC is
non-decomposable and noisier … on small inner-validation sets."

Measured on the two banked `fa_gt_1e-2` Tier-1 runs (`n_estimators=400`, `early_stopping_rounds=40`),
counting `^Tree=` in each `fold_*/classifier.txt`:

| run | trees min / p25 / median / max | folds ≤5 trees | folds with exactly 1 tree | mean AUC ≤10 trees vs >10 |
|---|---|---|---|---|
| `lightgbm_classification/99de85c1…/scale_S64_tfa_gt_1e-2` | 1 / 1 / 13 / 400 | 42 % | **11** | 0.649 vs 0.661 |
| `…/2d046f48…/scale_S32_tfa_gt_1e-2` | 1 / 1 / 7 / 249 | 42 % | **13** | 0.643 vs **0.683** |

A 1-tree booster at `num_leaves=63` is effectively a constant predictor.

- **Why it matters:** this is the Tier-1 reference head the **entire FM/Fang programme was benchmarked
  against** (`_w2_fang_heads.py:293`, `_w2_fang_probe.py:236`, `_fm_freeze_window.py:255`, plus
  `sweep_binary.py` / `train_binary.py` / `sweep_within_image.py`). The baseline that the frozen
  recipe's 0.7865 median AUC was declared to beat is an average over a cohort in which ~40 % of folds
  were truncated to a handful of trees by a metric the plan explicitly told the code not to use. The
  frozen recipe may still win — but the margin is not currently trustworthy.
- **Fix:** either pass `first_metric_only=True` to `lgb.early_stopping` or drop `"auc"` from `metric`
  (keep it as a *reported* metric, not a monitored one), then re-run the Tier-1 reference and re-state
  the FM-vs-Tier-1 margin wherever it is quoted.

### R33 — The abort table's `full` row measures calibrator clamping, not abundance level, and drops its 2 worst observations
- **Status:** OPEN · **Severity:** high (record correctness) · **Liveness:** dead-closed · **Verified:** no
- **Where:** [src/calibration.py:365-369](../src/calibration.py#L365-L369) (`calibrate_abundance` is a
  clamped `np.interp`), [scripts/f_region_staged.py:232](../scripts/f_region_staged.py#L232),
  `DECISIONS.md:5532-5541`, `reports/map_fbuild/README.md:14-20` ·
  **Detail:** [calibration.md](review_2026-07-31/calibration.md) `calibration-1`

`calibrate_abundance` is `np.interp` over the qmatch knots, so any probability above
`t2_x[-1] = 0.9999163` returns the constant `t2_y[-1] = 0.29324219`, and anything at/below
`t2_x[752] = 0.064311` returns exactly 0. The `full` variant rails (51.8 % of co-located tiles,
|o|max 21.3 logits vs the model's ±9.21 range), so **37.07 % of `full`'s 89,145 scored tiles land above
the calibrator's entire reference range** and get the clamp constant, plus 8.02 % at the zero floor —
**45 % of the scored population is a constant**. For 6 of 21 observations ≥73.9 % of finite tiles are
clamped (up to 100 %), so their published "over-prediction ratio" is just `ceiling / mean(label)`:
ESP_055978_2270 published 380.28 vs `ceiling/label_mean` = 380.47 (98.8 % clamped); ESP_045983_2270
60.21 vs 60.24 (99.1 %); ESP_017355_2260 published exactly the ceiling to 8 s.f. (100 % clamped). Two
further observations sit at the zero floor (ratio 0) and are **silently excluded**, so `full` is
reported over 19 observations while every other row is over 21 — the same drop noted in **R12**, now
with its mechanism.

- **Why it matters:** the row is bounded *above* by the calibrator, so it also **understates** how badly
  `full` fails. A future reader comparing `full` 0.412 to `pfree` 0.532 would conclude `full` is the
  more level-stable variant; it is the least *measurable* one.
- **Fix:** report `full` on the logit/probability scale, or report the clamped fraction beside every
  row and mark rows where it exceeds a few percent as unmeasurable; emit `n_used` per arm (per R12).

### R34 — The F Tier-2 calibrator is fitted on the un-levelled path and reused unchanged for all four variants — and `DECISIONS` retracts exactly this on a false factual ground
- **Status:** OPEN · **Severity:** high (record correctness) · **Liveness:** dead-closed · **Verified:** no
- **Detail:** [calibration.md](review_2026-07-31/calibration.md) `calibration-2`

The banked F calibrator is fitted on the **un-levelled** per-frame predictions and then applied
unchanged to `h1only` / `full` / `resid` / `pfree`, so the three levelled variants are scored through a
quantile map calibrated for a different prediction distribution. `DECISIONS.md` "Corrections" #3
retracts the hypothesis that the calibrator structurally favoured `h1only` — on the ground that "there
are no metadata JSONs in `models/deployable{,_f_center}/`". **This is the same false premise as R10**:
the `recipe.json` cards are there one directory down, and the calibration provenance is in
`calibration.npz`'s own metadata. Two independent reviewers reached this conclusion from different
directions, which is the strongest signal in the review that retraction #3 needs revisiting.
- **Fix:** re-fit the calibrator per variant (or state explicitly that the comparison holds calibration
  fixed and what that costs), and correct retraction #3 together with R10's half of it.

### R35 — Lower-severity findings from PASS 3
All in the linked area files; listed so they are not lost.

- **`geo-crs-2`** (medium, live-shipped) — invariant 2's only automated km-scale guard **cannot fail**:
  the phase-correlation solve is bounded to ±640 m by construction, and the solved shift is applied to
  every polygon with no band check. All 39 v2 label sets were produced this way. Pairs with **R30**.
- **`leakage-1`** (medium, code path still default) — Stage-6a neighbour features are computed *across*
  the within-image quadrant cut, so the **treatment arm's** test tiles carry training-fold feature
  values and only that arm does. The surviving Stage-6a decision was re-made at LOIO, so no live number
  is wrong, but the dev-PASS that promoted it is confounded.
- **`leakage-2`** (medium, live-shipped) — the one cohort image whose CTX is featureless is excluded
  from labels *and* features, so it is absent from every reported per-image metric — on a criterion that
  includes the CTX↔abundance relation being scored. Conditions the frozen recipe's headline per-image
  distribution.
- **`fm-embeddings-1`** (medium, dead-closed but quoted in DECISIONS) — the H2 nuisance basis and H3
  consistency pairs are estimated on a pool containing **every LOIO fold's held-out image**, so both
  "skill Δ" columns are transductive, not deployable.
- **`modeling-heads-2`** (medium, live) — the two classification variants can be trained as *regression*
  with no error: `fit` validates `y` **after** `astype(np.int8)`, so continuous `fractional_area`
  truncates to all-zeros and passes the "y must be binary 0/1" check. Already happened — 4 junk artifact
  dirs and 4 rows are banked in the v2 regression sweep, and `sweep.py`'s **default** `--variants`
  includes both classifiers.
- **`modeling-heads-3`** (medium, latent regression) — commit `61184fd` silently neutralised the
  `meaningful_threshold` monkeypatch in five count-target sweep probes; re-running any of them scores
  counts as **presence** (invariant 8) while `snapshot.json` still claims the right threshold.
- **`calibration-3`** (medium, live-shipped) — a banked calibrator is bound to **no** head: `load()`
  ignores the provenance it stores, `--model` and `--calibration` are independent flags, and the map
  records neither. Compare **R09**.
- **`calibration-4`** (medium, live-shipped) — `bank_calibration.py` writes the shipped calibrator
  *before* computing any gate, exits 0 whatever the result, and never evaluates 4 of the 6 declared §6
  metrics.
- **`leakage-3`** (low-medium, dead-closed) — Stage C picks λ as the argmin of the held-out-edge CV and
  gate 2 then reports and tests that same statistic at that λ; the code also selects on the metric its
  own docstring says must not select λ. Direction is toward PASS, so the ABORT is safe.
- **`leakage-4`** (low, dead probe) — the one GBM `eval_set` site that is not the rotated inner-val image
  early-stops on the **held-out test fold** and prints per-fold Spearman/AUC beside it.
- **`geo-crs-3`** (low, latent) — the SP1 correction is silently skipped for low-latitude images, and the
  tolerance is in degrees of latitude while the resulting ground error scales with longitude distance
  from the 180° central meridian.
- **`geo-crs-4`**, **`fm-embeddings-2..5`**, **`modeling-heads-4..6`** (low) — unreachable
  out-of-extent diagnostic; the parity gate exercises zero masked tiles and a different threshold than
  production; embedding stores carry no build provenance and resume on filename existence;
  `DeployableHead.load` verifies nothing though a recomputable `model_hash` exists; the H3 penalty runs
  in `train()` mode so dropout contaminates its objective; the four two-stage cousins write artifacts
  into a directory literally named `booster.txt`; CNN `load()` leaves the net on CPU while `predict()`
  moves inputs to CUDA; the CNN's brightness jitter is ±15 % of the full 0–255 DN range rather than of
  the per-tile range (2.1× the intended magnitude) — a confounder on the "photometric augmentation
  REFUTED" verdict.

---

## 4d. Findings from the per-area reviews (PASS 4 — the cross-cutting sweeps)

> Same verification caveat as §4b/§4c: single-agent, self-refuted, not independently checked.

### R36 — The H4 leg-B skill gate could not have failed: the offsets it applied were a near-constant
- **Status:** OPEN · **Severity:** ~~high~~ → **medium** · **Liveness:** dead-closed programme, but quoted as live evidence in `ROADMAP.md:19`, `PLAN_StripingArtifact.md:238,267`, `PLAN_H4_Leveling.md:59-67`, `DECISIONS.md:4534-4538` · **Verified:** [CONFIRMED-BUT-MIS-STATED](review_2026-07-31/verify/R36.md) — ⚠ **"could not have failed" is FALSE**: the gate is monotone in the applied differential and the same offsets ×2 give −0.0274 = FAIL. It was handed a ~5× attenuated treatment, not rendered inert. "Authorised ~265 CPU-h" over-attributes (probe-extrapolated midpoint, not measured spend). The mean-zero algebra itself was reproduced *harder* — the 17-obs set was predicted from graph topology alone, with set equality against the banked CSV.
- **Where:** [scripts/f_h4_legb.py:145](../scripts/f_h4_legb.py#L145),
  [:149-154](../scripts/f_h4_legb.py#L149-L154); [scripts/f_h4_level.py:90-106](../scripts/f_h4_level.py#L90-L106);
  `reports/figures/f_h4_legb_offsets.csv` · **Detail:** [stats-fallacies.md](review_2026-07-31/stats-fallacies.md) `stats-fallacies-2`

`PLAN_H4_Leveling.md:48-57` correctly rules out per-image AUC as blind to H4 and pre-declares **pooled**
PR-AUC / prec@5% as the instruments that "DO see cross-frame level changes". On the leg-B graph
(58 frames / 47 edges / **21 components**) the pooled instrument sees nothing either, for a provable
reason: for any connected component *c*, `1_c` lies in the null space of the graph Laplacian **and**
`AᵀWb` has exactly zero projection on it (each edge contributes `+w·δ̄` to *i* and `−w·δ̄` to *j*, both
inside *c*). So `(L + λI) o = AᵀWb` forces `mean(o_c) = 0` **exactly, for every component and every λ**.
The between-component level — the only thing a pooled cross-image metric can respond to — is identically
zero *by construction*, not estimated. Because the graph is "mostly within-obs", most observations are a
whole component, so `obs_off = 0` before the global `o − median(o)` gauge and `= −0.0753` after it.

Measured on the banked artifact: **17 of 28** observations carry the identical value `−0.0753`; the
applied offsets have **interquartile range exactly 0** and sd 0.308 logits — against the build's own
solved offsets at sd 1.46 (`resid`) / 1.78 (`pfree`) / 6.45 (`full`). Only 9 of 36 scored images received
a shift differing from that constant by ≥0.05. The gate then reported `Δ pooled PR-AUC = −0.0104` → PASS.

- **Why it matters:** the reopening rule was "η² ≲ 0.05 **at skill ≥ −0.02**". The skill half was cleared
  by an instrument that applies a near-constant, so it would have returned ≈0 however damaging real
  leveling is. When the same statistic was finally computed on real one-component offsets at build scale
  it read **−0.089 (full) / −0.030 (resid) / −0.186 (pfree)** — 3–18× the tolerance. Four documents
  saying "PASS — leveling preserves skill on real LOIO predictions" describe a measurement of nothing.
  This is the cleanest instance in the review of a **gate that could not fail** authorising real spend
  (~265 CPU-h + 33 GPU-h). Compare **R11** (the tautological trend guard) and **`leakage-3`**.
- **Fix:** record the retraction in `DECISIONS.md` and amend the four documents. Any future
  overlap-leveling gate must (a) assert the frame graph is one component before claiming a cross-image
  instrument means anything, and (b) report the sd/IQR of the offsets it actually applied beside the
  metric delta, so a near-constant treatment is visible.

### R37 — README and SHERLOCK_RUN still instruct the next session to run the aborted F build
- **Status:** OPEN · **Severity:** ~~high (operational)~~ → **medium** · **Liveness:** live-shipped (sub-claim (iii) is live-active-plan) · **Verified:** [CONFIRMED — all three sub-claims](review_2026-07-31/verify/R37.md). Gate-1 drift re-derived exactly: docs cite η² 0.1222 / p95 0.0676 / ratio 1.65, file holds **0.120535 / 0.07001 / 1.5276** (cause: the common-footprint mask added by `41a6f26`). It is **four docs / five sites**, not three — this entry misses **`PLAN_FBuild.md:337`**. Downgraded because `ROADMAP.md:18` (bold **HARD ABORT**) and the CURRENT memory note are both read before README.
- **Detail:** [docs-consistency.md](review_2026-07-31/docs-consistency.md) `docs-consistency-1`, `-2`, `-3`

`README.md`'s Status + "Next priorities" sections and `SHERLOCK_RUN.md` Part J are **entirely
pre-abort**: they tell the reader to execute the 907-frame F build that was hard-aborted on 2026-07-30.
The project's own convention ("when reality diverges from a doc, update it in the same change") was not
applied to the two docs a new session reads first. Related: the only ACTIVE plan
(`PLAN_RegionalMap.md`) is still documented as **blocked on the F map**, contradicting the commit message
of the very change that made it the only active plan; and gate 1's "banked mosaic baseline" was silently
overwritten in place, so three docs cite numbers the file they name no longer contains.
- **Fix:** rewrite README Status / Next priorities and SHERLOCK_RUN Part J against the post-abort
  reality; unblock PLAN_RegionalMap's thermal legs explicitly (the abort is what unblocked them); and
  either restore or re-cite the overwritten gate-1 baseline.

### R38 — A1's uint8 clip floor is `0`, which is the mosaic nodata sentinel, so dark valid CTX pixels become "nodata"
- **Status:** OPEN · **Severity:** ~~high~~ → **medium** · **Liveness:** ~~live-shipped~~ → **live-active-plan** (nothing shipped carries it: `reports/map_a1/` does not exist, and the shipped `reports/map_region/` never imports `src.striping`) · **Verified:** [CONFIRMED-BUT-MIS-STATED](review_2026-07-31/verify/R38.md) — **verified twice independently; the second pass merged both into one file.** Mechanism confirmed at code level and reproduced on a synthetic array: there is **no separate nodata mask**, `predict_window` infers nodata from `arr == 0` on the normalised array it is handed. Not deliberate (`git log -S`: born in `830a39b`, never touched, no DECISIONS entry), not tested, and three sibling stretches do use `[1,255]` with a written rationale — the Stage-B comparison checks out.
  **Blast radius dies:** **0.041 %** of valid native pixels on the training path (so the banked −0.024 LOIO cost is uncontaminated), **0.41 %** on a deploy-statistic simulation, region-wide bracketed **0.04–0.41 %**; tiles actually dropped 0.044 % train / 0.375 % deploy-sim, and **0** in the A1-payoff crop. ⚠ **The η² confound never happened** — numerics-1's load-bearing consequence is refuted: `DECISIONS.md:4133`'s 218,089 = 467² exactly = the complete interior grid of the 15008-px crop (469² = 219,961), so zero tiles were masked in *either* arm and the 28 % reduction is already on a common footprint.
  **What survives:** 180/380 frames have a positive threshold; the damage concentrates in **low-IQR frames** — exactly where a robust gain is largest — where whole tiles can reach `own_tile_zero_fraction = 1.00`. R07's aggravation is real and quantified: the 160 m IQR is 1.50× narrower → `thr` +31 DN → **10× more pixels clipped**.
  **Fix correction:** `[1,255]` is right and test-safe but only moves the collapse onto DN 1 — itself the Murray bottom-clip value that already bit this project (`DECISIONS.md:2725-2732`, where DN 1 being modal silently killed all four shadow features in two images). The clean fix is an explicit nodata mask. Drop the "re-score η² on a common mask" sub-fix as unnecessary.
- **Where:** [src/striping.py:251-253](../src/striping.py#L251-L253) ·
  **Detail:** [numerics.md](review_2026-07-31/numerics.md) `numerics-1`

`a1_apply` clips to `[0, 255]` and then re-zeroes `arr == 0`. But `0` is the Murray mosaic's **nodata
sentinel** (that is exactly what `own_tile_zero_fraction` tests for, and what `a1_stats` excludes via
`arr > 0`). Any genuinely dark valid pixel that the robust rescale pushes to ≤0 becomes indistinguishable
from a data gap — so it is counted as nodata by the masking logic downstream and contributes a hard black
pixel to the embedding. Compounds **R07** (the gain is already inflated by the 160 m-vs-native statistic
mismatch, which pushes more pixels past the floor) and **R08** (mixed normalized/raw output).
- **Fix:** clip to `[1, 255]` and reserve 0 for nodata — the same convention Stage B already uses
  deliberately ([scripts/f_region_stageb.py:76-77](../scripts/f_region_stageb.py#L76-L77) clips to
  `1, 255`). Fix before building any A1 map (**R06**).

### R39 — The abort has exactly one cross-arm skill instrument, and it is F-vs-F by construction
- **Status:** OPEN · **Severity:** medium (record completeness on the most consequential decision) · **Liveness:** dead-closed · **Verified:** no
- **Detail:** [stats-fallacies.md](review_2026-07-31/stats-fallacies.md) `stats-fallacies-5`

Gate 5 compares F variants against `F_h1only`, not against the mosaic — it is F-vs-F by construction —
and the cohort join never reads the mosaic raster at all. So the abort's *skill* evidence contains no
mosaic-vs-F comparison, and the only mosaic-vs-F skill number ever measured elsewhere points the
opposite way. Combined with **R10** (the level comparison is two-factor) this means the decisive
comparison rests on the level table alone. It does not overturn the abort — gate 6's `full` top_ratio
8.74 and the level spread are independent — but the record overstates how many independent instruments
agreed.
- **Fix:** state plainly in `DECISIONS.md` which gates are F-vs-F and which are cross-arm, and note that
  gate 5 is the former.

### R40 — Retained `trend_verdict` adjudicates on raw R² while discarding the per-side nulls it computed, and the two sides' nulls differ 2.2×
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed programme, but `lv.trend_verdict` is listed under "Retained deliverables — general, stays in the codebase" (`DECISIONS.md:5578-5581`) · **Verified:** no
- **Detail:** [stats-fallacies.md](review_2026-07-31/stats-fallacies.md) `stats-fallacies-1`

The §4.3 guard compares metadata-vs-geology on **raw** R², throws away the block-permutation null it
computed for each side, and the two sides' nulls differ by 2.2× — so the comparison is not
floor-relative and the side with the looser null is favoured. This is the same class of error as the
already-fixed "verdict-margin" bug and it sits in code the project deliberately kept for reuse.
- **Fix:** adjudicate on excess (`R² − null_mean`) or on `R²/null_p95`, not raw R², and keep the margin
  rule.

### R41 — Every acceptance tolerance in the striping/F programme is ±0.02, and none of the gated statistics has a sampling uncertainty
- **Status:** OPEN · **Severity:** medium (methodology) · **Liveness:** live methodology — `src/fgates.py` constants are current and PLAN_RegionalMap's parked legs would reuse the pattern · **Verified:** no
- **Detail:** [stats-fallacies.md](review_2026-07-31/stats-fallacies.md) `stats-fallacies-3`

`SKILL_TOL = -0.02`, `THEMIS_TOL = 0.02`, the η² bar, the reopening rule's `≥ −0.02` — all ±0.02, and not
one of the gated statistics has ever been given a CI or a resampling spread. So a PASS at −0.014 and a
FAIL at −0.030 may be indistinguishable. This is the frame in which every PASS/FAIL in the programme
should be read, and it interacts with **R36**, **R11** and **`leakage-3`**: several gates were decided
inside the noise band.
- **Fix:** attach a bootstrap or block-permutation spread to each gated statistic (the machinery already
  exists — `lv.block_permute`, `st.eta2_rotation_null`) and require the margin to exceed it.

### R42 — Lower-severity findings from PASS 4
- **`numerics-3`** (medium, live code path) — a two-stage hurdle that cannot fit its magnitude head
  silently predicts **all zeros**, and the docstring promises a different fallback.
- **`numerics-2`** (medium, closed programme, quoted in docs) — Stage-7d's "per-image standardised"
  Spearman standardises the *feature* but not the *target*, understating the reported ρ by ~1.4–1.6×
  (conservative direction).
- **`stats-fallacies-4`** (medium, live document) — Stage 7d's pooled tests treat spatially
  autocorrelated tiles as independent; the headline p-values overstate the evidence by ~12 orders of
  magnitude, and the strongest feature **fails** an image-level test. `docs/compositional.md` is written
  for external readers.
- **`docs-consistency-4/-5/-6`** (medium/low) — `docs/modeling.md` §11 "Reproducibility" gives a command
  that cannot run and names a directory that does not exist; `DATA_DICTIONARY` says the detection filters
  are null "the current default" when both configs set `min_size_m: 1.4105` and it demonstrably drops
  polygons (compare **R23**); `docs/index.md` omits 4 of 11 docs including `model_evidence.md`.
- **`other-scripts-1/-2/-3`** (medium) — the two repackage drivers' copy of the split hash has **drifted**
  from `src/dataset.py`, so 7 committed split JSONs carry a `split_hash` the canonical function cannot
  reproduce (compare **R04**); `sweep.py --include-cnn` is accepted, documented and **never read**, so the
  CNN arm silently does not run and notebook 10 reports a stale CNN row; `train_gbm.py` / `train_binary.py`
  cannot select a dataset root, omit it from provenance, and write into the same `*/scale_S{n}` namespace
  the sweeps use while claiming interchangeability.
- **`invariants-1..5`** (medium→low) — notebook 17's committed verdict cell reports the **retracted**
  Fisher's exact result, contradicting its own executed output and its `_build` source; notebook 20's
  SUPERSEDED banner exists only in the `.ipynb`, so running the documented regeneration command silently
  deletes it; notebooks 12 and 13 are committed with **zero executed cells**; `hirise_decimation_mpp` is
  a required, provenance-hashed config key that **no code reads** (every call site hardcodes 5.0);
  CLAUDE.md's "notebooks are generated" is false for 7 of 28 notebooks.
- **`notebooks-1/-2`** (medium) — notebook 10 is pinned to the v1 sweep for its tables but resolves model
  artifacts by **most-recently-modified** directory, so three figures published as the v1 baseline in
  `docs/modeling_results.md` are actually v2; the Stage-7.0 GO statistic exists **only** inside notebook
  14, its declared producer never computes it, and the published writeup calls it a Spearman correlation
  when the code computes something else.
- **`numerics-4/-5`, `other-scripts-4/-5/-6`, `notebooks-3`** (low) — a missing Stage-4b features parquet
  is silently swallowed into an all-NaN feature block; `precision@5%` is read by **positional**
  `itertuples` field so a column insertion swaps in a different metric; `--target-col` on
  `train_gbm.py`/`train_cnn.py` silently degrades the mandated rich/poor metrics to presence metrics;
  `build_vclaire_manifest.py`'s antimeridian guard is a no-op in the only case it exists for;
  `run_stage7c_features.py --only` overwrites the full-cohort colour features with the sanity subset and
  four `--all` drivers exit 0 after failing images; `.gitignore` excludes an 18 MB LOIO prediction dump by
  name while tracking 267 MB of identical-class F dumps and an 86 MB notebook.

---

## 4e. Findings from the second passes (PASS 5)

Three areas (`stats-fallacies`, `docs-consistency`, `notebooks`) were re-dispatched after their files
already existed; rather than clobber them, those agents ran an **independent second pass** over the gaps
their own PASS-1 coverage notes declared unchecked, and extended the files. Same verification caveat.

### R43 — The η² reopening bar sits *below* the geological floor of the crop it was calibrated on
- **Status:** OPEN · **Severity:** high (record correctness) · **Liveness:** dead-closed programme, but the raw numbers are quoted as verdicts in `ROADMAP.md:19` · **Verified:** no
- **Detail:** [stats-fallacies.md](review_2026-07-31/stats-fallacies.md) `stats-fallacies-6`

The reopening rule's `η² ≲ 0.05` is an **absolute** constant, but `src/fgates.py`'s own header already
records that on the pilot crop the mosaic scores 0.1948 against a rotation null of 0.083–0.117 — i.e. the
bar sits below the geological floor of the very crop it was calibrated on, so **nothing can pass it** on
its own terms. Read floor-relative against that measured floor instead, **H1 alone removes 60–86 % of the
excess**. This is the gate that sequenced H1→H2→H3→H4, and (with **R36**) it is what authorised the
907-frame build. The abort's own gate-1 table says as much — "no variant clears the absolute η² bar, and
neither does the mosaic (0.121)" — but that was discovered *after* the spend, not before.
- **Fix:** record in `DECISIONS.md` that the H1–H4 sequencing rested on an absolute bar known to be below
  the floor, and that floor-relative reading changes the ordering. Any future artifact bar must be stated
  as excess-over-null or ratio-to-null, never as an absolute η².

### R44 — `docs/methods.md`, the document the README sends external readers to, is half-migrated to v2
- **Status:** OPEN · **Severity:** ~~high~~ → **medium** · **Liveness:** live-shipped — `README.md:8-10` and `docs/index.md:32` route reviewers, collaborators and the advising committee here for "how the dataset was produced" (routing verified verbatim) · **Verified:** [CONFIRMED-BUT-MIS-STATED](review_2026-07-31/verify/R44.md) — half-migration real (`479688d` migrated only §5), but the area file's "wrong by 2×–70×" is **wrong**: every v1 number reproduces *exactly* against `dataset/` (643,910 tiles to the unit; all 13 §7.4 ρ to 3 dp). The charge is stale **scope**, not bad arithmetic → **the fix is relabel, not recompute**. Four further defects found, incl. that **§5 — the one migrated section — is itself stale** (its `dy` column predates the 2026-06-10 y-sign fix), so "relabel §§6–8 and leave §5" is insufficient.
- **Detail:** [docs-consistency.md](review_2026-07-31/docs-consistency.md) `docs-consistency-7`

The reader-facing Methods document was only partly updated for the vClaire v2 cohort: it pins itself to
one cohort in §1 and then reports another's numbers downstream. This is the single most externally-visible
document in the repo, and the audience is a committee.
- **Fix:** reconcile `docs/methods.md` against the v2 cohort end to end, or state explicitly at the top
  which cohort each section describes.

### R45 — The within-image-vs-LOIO diagnostic pairs a *quadrant* AUC against a *whole-image* AUC
- **Status:** OPEN · **Severity:** high (unchanged; one verifier considered `blocker`) · **Liveness:** live-shipped — the live damage runs through **`scripts/probes/_diag_within_image_deltas.py` → `docs/modeling_results.md` §9.4** as well as the cited `_build_10.py` copy · **Verified:** [CONFIRMED **twice, independently**](review_2026-07-31/verify/R45.md) — two agents worked it without seeing each other and agreed on verdict, severity and liveness; the second pass is appended below the first in the verdict file. Every banked per-fold AUC reproduces exactly (max diff 0.0); all 8 published cells reproduce digit-for-digit.
  ⚠ **The size null is ~zero** (pass 1: −0.0000/+0.0000/+0.0007/−0.0001; pass 2, 200 draws/image: −0.0000/+0.0004/−0.0002/+0.0004), so **the proposed fix's second branch would report zero and entrench the error**. Pass 2 adds the mechanism: base rates are *identical* whole vs quadrant (0.5061/0.5071 at S=8), so **this entry's "compare R26" analogy is wrong** — the driver is truncated dynamic range, not size and not prevalence. Matched pairing moves **4 of 8 cells to p<0.05 and 5 of 8 CIs off zero**; on the mandated `meaningful_auc` it fails at all four scales (p ≤ 0.0011). The bias depresses the within arm, i.e. it **manufactures the false null** rather than inventing an effect.
  H5's substantive reading survives either way (the pre-declared `within ≥ 0.7` bar is unmet; within AUC 0.54–0.68), so "the only quantitative instrument behind the H5 conclusion" is an overstatement — what dies is the significance claim the docs print. Line ref correction: the "every CI brackets zero" sentence spans `docs/modeling_results.md:980-985`, not `:981-983`.
- **Where:** [notebooks/_build_10.py:789-823](../notebooks/_build_10.py#L789-L823)
  (`per_image_within_minus_loio`) · **Detail:** [notebooks.md](review_2026-07-31/notebooks.md) `notebooks-4`

The comparison handicaps the within-image arm by construction (a quadrant has fewer tiles and a narrower
abundance range than a whole image), yet `docs/modeling_results.md:981-983` reads the result as
**"every CI brackets zero"** — a null conclusion drawn from a mismatched pairing. Compare **R26**: the
same class of error (comparing a ranking metric across populations with different sizes/base rates).
- **Fix:** pair like with like — score the LOIO arm on the same quadrants, or bootstrap the quadrant-size
  effect and report it as the null — then re-read the H5 conclusion.

### R46 — Lower-severity findings from PASS 5
- **`docs-consistency-8`** (medium, live-active-plan) — "per-image AUC ≈ 0.43" is the **Tier-2 abundance
  Spearman ρ mislabelled as an AUC**, and it is the stated skill baseline for leg 4 of PLAN_RegionalMap
  (the only ACTIVE plan) and the record of the striping-mitigation adjudication rule. The session that
  runs leg 4 would compare an AUC against a ρ.
- **`docs-consistency-9`** (medium, live-shipped) — `docs/model_evidence.md` §8 still lists the
  calibration layer and the abundance product as **future work**; both shipped 2026-06.
- **`notebooks-5`** (medium, live document + live reporting rule) — notebook 12's target-reframing
  evidence compares raw `lift@top-K` across two targets whose base rates differ **2.7×**, so most of the
  apparent gain is the base rate (the same defect **R26** fixes for `precision@k`, and the same one
  `normalised_lift_at_top_k` exists to solve).
- **`stats-fallacies-7`** (low, live-active-plan) — the "MOLA leg" credited in the only ACTIVE plan's
  validation ledger is a **self-fulfilling site-selection check with no model prediction in it**. The
  ledger reads "MOLA leg done" as if it were independent corroboration of the shipped map.
- **`docs-consistency-10`** (low, live-active-plan) — PLAN_RegionalMap's final deliverable is named two
  different things, neither exists, and its declared figure prefix does not match what the code writes.
- **`notebooks-6`** (low, dead leg but restated live) — notebook 12's per-image summary hard-codes
  "7 of 25 / 4 of 25 folds"; the artifact it names says **8 of 37 / 6 of 37**, overstating the cohort's
  usable fraction.

### R47 — No test covers the v2 splits, and both load-bearing invariant guards are `slow`
- **Status:** OPEN · **Severity:** high · **Liveness:** live-shipped · **Verified:** ✅ direct pass, commands quoted
- **Where:** [tests/test_modeling_group_leak.py:21](../tests/test_modeling_group_leak.py#L21),
  `:23-26`, `:29,34,47,60,72,83`; [tests/test_sanity_residual_one_image.py:24](../tests/test_sanity_residual_one_image.py#L24) ·
  **Detail:** [tests.md](review_2026-07-31/tests.md)

The group-leak suite — the only automated enforcement of **invariant 6** — is pinned to the **v1**
`dataset/packaged/loio_9fold`. A repo-wide grep of `tests/` for `dataset_v2` or `loio_nfold` returns
**two hits**, both reading v2 *labels*, never the packaged splits. So `loio_nfold` (the 38-fold scheme
behind `pooled_pr_auc 0.7832`, the frozen recipe, the deployable head and the shipped map),
`loio_nfold_ctx_illum`, `loio_nfold_nbr_s5` and v2's `within_image_4fold` are covered by **no test at
all**.

Compounding it: all 6 group-leak tests are `@pytest.mark.slow`, as is
`test_stage1_centroid_residual_under_threshold` (**invariant 2**, the O(200 m) CRS residual CLAUDE.md
says "must fail loudly"). Both invariants the operating manual calls load-bearing are outside the
`pytest -m "not slow"` loop the manual tells you to run. Invariant 2 is doubly exposed — **R30** shows
it has no production caller either. And every integration test degrades to a silent pass when its
cache is absent, so a fresh clone reports green having evaluated none of them.

This closes a loop with **R04** (a failed Stage-5 rebuild leaves stale packaged splits, undetectable
downstream) and **`other-scripts-1`** (the repackage drivers' split hash has already drifted, so 7
committed split JSONs — including two v2 schemes — carry a hash the canonical function cannot
reproduce): three independent findings on one uncovered surface.
- **Fix:** parameterise `test_modeling_group_leak.py` over `(dataset_dir, scheme)` and add the four v2
  schemes; split each assertion into a **metadata-only** half (fast — reads
  `dataset*/splits/{scheme}.json`, needs no parquet) and a parquet-backed half (stays `slow`); add a
  `--strict-integration` mode where the data-missing `skip`s become failures.
- **Also from this area (low):** the per-image-local-radius invariant is exercised with only **one**
  local radius, and the assertion meant to prove the radius matters is
  `assert abs(dy - y) > 0.0` — true for any float difference.

---

## 4f. Findings from `scripts/probes/` (PASS 6)

> Single-agent, self-refuted, not independently verified. Detail in the linked area file. Severity here
> is about **the record**, not the probe: these are throwaway scripts, and only the numbers that
> escaped into `DECISIONS.md` / `docs/` / a `PLAN` matter.

### R48 — The "CTX-source-heterogeneity mechanism is EMPIRICALLY VALIDATED" verdict is a per-image prevalence confound
- **Status:** OPEN · **Severity:** high (unchanged) · **Liveness:** dead-closed programme, quoted as a *live* validated mechanism in `docs/modeling.md:490-491` and `PLAN_ModelUsability.md:35` · **Verified:** [CONFIRMED](review_2026-07-31/verify/R48.md) — ρ = **+0.9834** reproduces (claimed +0.983); **11 of 12** cells die at the correct dof = n−3 (10/12 under the area file's dof = n−2). Strengthened three ways: none of the 12 survives *any* multiplicity correction even before partialling; base-rate-free instruments give p ≥ 0.10 for all four features; and the two "independent" heterogeneity features are ρ = −0.979 with each other. **Blast radius corrected: the later striping diagnosis does *not* rest on this** (no hits in any PLAN), so it is a record fix, not a science re-do.
- **Where:** [scripts/probes/_diag_stage6b_h3_check.py:133-145](../scripts/probes/_diag_stage6b_h3_check.py#L133-L145),
  `:6-7`; [scripts/probes/_sweep_stage6b.py:19-24](../scripts/probes/_sweep_stage6b.py#L19-L24) ·
  **Detail:** [probes-stage6.md](review_2026-07-31/probes-stage6.md) `probes-stage6-1`

The **pre-declared** test — `rho(per-image AUC, mean_ctx_incidence) < −0.30, p < 0.05` — **failed**
(+0.050, p = 0.765 on `pr_auc`). The record then substitutes a post-hoc 4-feature × 5-metric grid the
same probe prints, harvests the 12 cells at `p < 0.05`, and declares the mechanism validated. But four
of the five metrics in that grid are the image's **positive base rate re-expressed**: over the 38
images, `Spearman(pr_auc, meaningful_base_rate) = +0.983`, `(normalised_lift, base_rate) = +0.981`,
`(precision@5%, base_rate) = +0.918` — and the CTX features themselves correlate with the base rate
(`std_ctx_incidence` −0.371, `mean_n_sources` −0.299, `dominant_source_frac` +0.338). Partial out the
base rate and **10 of the 12 significant cells die**.

- **Fix:** re-state the Stage-6b/6e conclusion as base-rate-conditional and report the partial
  correlations, or withdraw "empirically validated" from `docs/modeling.md` and `PLAN_ModelUsability.md`.

### R49 — Stage 6c's acceptance test is passed by a prevalence oracle carrying no anti-signal information
- **Status:** OPEN · **Severity:** high · **Liveness:** dead-closed (never shipped) but quoted as a verdict in `ROADMAP.md:30` · **Verified:** no
- **Detail:** [probes-stage6.md](review_2026-07-31/probes-stage6.md) `probes-stage6-2`

Stage 6c's "bad image" label **is** the per-image base rate, so its strict acceptance test can be passed
by a predictor that knows only prevalence and nothing about anti-signal — the thing the stage exists to
detect. The implemented criterion is also not the pre-declared one. Same family as **R48**: the
"reliability" signal is prevalence wearing a different name.
- **Also from this area:** the `+0.056` "Strategy B" soft-PASS that `ROADMAP.md:30` records as the one
  positive Stage-6 result sits at the **95th percentile of a permutation null nobody computed**, and
  v2's headline is a **max over 20 configurations** with no selection correction
  (`probes-stage6-3`, medium). `std_ctx_incidence` is two different statistics in the two probes while
  the record says 6c reuses the 6b-validated feature (`probes-stage6-4`, medium). Stage 6a's single
  dev-PASS clears its bar by **0.13 of a standard error**, and the fold-variance probe that exists to
  answer "is this noise?" is hardcoded to the *other* sweep (`probes-stage6-6`, low) — it was later
  re-tested at full-v2 LOIO and STRICT-FAILED, so the record self-corrected.

### R50 — The `boulder_count` target win is a change in the positive-class definition, not in model skill
- **Status:** OPEN · **Severity:** high · **Liveness:** dead-closed for the shipped map (the frozen recipe reverted to `fa_gt_1e-2` @ S=32) · **Verified:** no
- **Detail:** [probes-compression-targets.md](review_2026-07-31/probes-compression-targets.md) `probes-compression-targets-1`

The `boulder_count` target's "+22 % dev PR-AUC" and the **W0 P2 promotion** (+0.146 / +0.162) are a
change in what counts as a positive, not an improvement in ranking. Rescored on one common positive
definition the advantage largely disappears. Same prevalence family as **R48**, **R26** and
`notebooks-5`. The frozen recipe reverted to `fa_gt_1e-2` independently, so no shipped number depends
on it — but the promotion decision and its restatements do.

### R51 — The reader-facing `docs/modeling_results.md` "Bottom line" rests on a sign test over 12 correlated re-analyses of the same 8 images
- **Status:** OPEN · **Severity:** high (unchanged) · **Liveness:** live-shipped document — `README.md:45-46` and `docs/index.md:34` route readers to it · **Verified:** [CONFIRMED](review_2026-07-31/verify/R51.md) — all four published p-values reproduce to 4 dp. The decisive new number: under the exact image-level sign-flip null (2⁸ = 256 flips) the published test's **true rejection rate at nominal α = 0.05 is 0.324** — it rejects a true null one time in three; exact clustered p-values are inflated ~17× and ~740×. At n=8 **0 of 12** configs reach p<0.05, so "statistically real" is unsupported. **Blast radius corrected downward:** the project's scientific verdict survives on independent n=38 image-level evidence (p = 2.8e-5) — R51 removes a corroboration, not the finding. ⚠ Its source sweep dir `models/_sweep/20260524T071830Z` is **untracked in git**.
- **Detail:** [probes-compression-targets.md](review_2026-07-31/probes-compression-targets.md) `probes-compression-targets-2`

The document's headline conclusion is supported by a sign test that counts **12 correlated
re-analyses of the same 8 images** as 12 independent observations. Same defect class as
`stats-fallacies-4` (Stage 7d's pooled tests treating autocorrelated tiles as independent) — and again
in a document written for readers outside the project.
- **Fix:** recompute with the unit of analysis being the image (n = 8), or drop the significance claim.

### R52 — "Isotonic recalibration drops Spearman and AUC" was a structurally guaranteed result
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed decision (it motivated the four two-stage variants, later found NULL) · **Verified:** no
- **Detail:** [probes-compression-targets.md](review_2026-07-31/probes-compression-targets.md) `probes-compression-targets-3`

"Isotonic recalibration drops Spearman 0.169 → 0.157 and AUC 0.579 → 0.572" is mathematically pinned: a
**monotone map cannot raise a rank metric**, so the only possible outcomes were "unchanged" or "down by
tie-handling". The "AUC" quoted is also presence AUC (invariant 8). **Sixth instance** of the
could-not-fail pattern.

### R53 — The frozen recipe's evidence framing has three defects, though the `0.7832` headline itself survives
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped (`docs/model_evidence.md` is reader-facing; the gate numbers are banked) · **Verified:** no
- **Detail:** [probes-fm-recipe.md](review_2026-07-31/probes-fm-recipe.md)

**The good news first:** the reviewer audited `_w2_fang_probe.verdict()` — which computes the headline
`pooled_pr_auc 0.7832` / `precision@5% 0.948` / `median per-image AUC 0.7865` baked into every
`recipe.json`, `ROADMAP.md` and `PLAN_FM.md` — and did **not** find a defect in the statistic itself.
The problems are in the framing around it:
- **`probes-fm-recipe-1`** (medium, live doc): "the FM differentially rescues the W1 failure classes"
  is a conditioning artifact — those images are the FM's *worst* in absolute terms.
- **`probes-fm-recipe-2`** (medium): the per-image gate (`win 0.96`, `dAUC(v) +0.120`) is scored on
  **27 of 38 images** by a validity rule computed at the wrong scale on the wrong target, and it
  excludes **both of the recipe's two largest losses**.
- **`probes-fm-recipe-3`** (medium, published writeup): the per-image ΔAUC figure in
  `docs/model_evidence.md` is the **S=64 LightGBM probe cell**, captioned as the frozen FM recipe.
- **`probes-fm-recipe-4`** (low): `DECISIONS.md:3204` quotes a per-image ΔAUC appearing in none of the
  90 banked verdicts, and the next entry silently contradicts it 5×.

---

## 4g. Findings from `scripts/probes/` (PASS 7)

### R54 — The shipped abundance layer's calibration is reported pooled only; per-image, only 11 of 37 images are inside the declared band
- **Status:** OPEN · **Severity:** high · **Liveness:** **live-shipped** — `models/deployable/calibration.npz` is the Stage-1 layer applied by the regional map · **Verified:** no
- **Detail:** [probes-tier2-calibration.md](review_2026-07-31/probes-tier2-calibration.md) `probes-tier2-calibration-1`

The Tier-2 gate is recorded as `top_ratio 0.86` → **PASS** against the declared `[0.8, 1.2]` band. That
is a **pooled** statistic. Per image it is **0.566 median / 0.168 p10, with only 11 of 37 images inside
the band**. So the calibrated abundance layer that the circum-Chryse map publishes is, for two-thirds of
the cohort, outside the tolerance the project set for it — and the gate that was supposed to catch that
is computed at the one aggregation level where the failure cancels.

This is the **fourth** instance of pooled-vs-per-image hiding a per-place failure (**R33** gate 6,
**R26** precision@5%, `fgates` gate 6's rich-truth conditioning). Unlike those, this one is on the
**live shipped product**, not the aborted F build.
- **Fix:** report `top_ratio` per image alongside the pooled value wherever the gate is quoted — the
  per-obs block already exists in `scripts/bank_calibration_f.py:104-119` — and decide explicitly
  whether the pooled PASS is sufficient to ship. `DECISIONS.md:5049-5053` rules the pooled one *is* the
  gate, so this needs a ruling, not a silent fix.

### R55 — "The ~0.43 per-image ceiling is the 5 m/px CTX floor, confirmed five ways" — all five ways share the same embedding and trunk
- **Status:** OPEN · **Severity:** high · **Liveness:** dead-closed plan stage, but **the conclusion is the project's standing strategic premise** · **Verified:** no
- **Detail:** [probes-tier2-calibration.md](review_2026-07-31/probes-tier2-calibration.md) `probes-tier2-calibration-3`

This conclusion is why PLAN_Calibration Stage 2 was **closed** ("retraining ceiling = the 5 m/px CTX
floor") and why the resolution floor is treated as fundamental rather than as a modelling limitation.
But all five "independent" confirmations hold the frozen Fang-ViT / GeM-96 / S=32 embedding **and** the
same MLP trunk fixed, and the "two different architectures" claim does not survive inspection. Five
variations of the head on one frozen representation cannot distinguish "the imagery has no more
information" from "this representation extracts no more information".
- **Why it matters:** it is the premise behind closing an improvement avenue. If the ceiling is
  representational rather than physical, Stage 2 was closed early.
- **Fix:** re-state the ceiling as *conditional on the frozen embedding*, and note what would actually
  test the physical claim (a different backbone, or a native-resolution upper bound).

### R56 — "`min_confidence` filtering is HARMFUL, ruled out" is a two-factor comparison — and this is what blocks R23's fix
- **Status:** OPEN · **Severity:** high · **Liveness:** dead-closed programme, but it is the recorded justification for the **live** `min_confidence: null` · **Verified:** no
- **Detail:** [probes-tier2-calibration.md](review_2026-07-31/probes-tier2-calibration.md) `probes-tier2-calibration-2`

The record rules out confidence filtering as "monotonically degrading ranking". The comparison changed
**both** the model and the target; with the target held fixed, `conf ≥ 0.5` is not harmful.

**This matters because it is the obstacle to fixing R23.** R23 (independently confirmed) is that two
cohort images are labelled at a 0.41/0.62 confidence floor while 36 are at 0.10, and the natural remedy
is to harmonise the cohort by setting `min_confidence` to the max of the per-image kept-score minima.
The record currently forbids that on the strength of a confounded comparison. **Re-run the
`min_confidence` comparison with the target held fixed before deciding how to fix R23.**

### R57 — The Stage-7.0 GO numbers are band ratios on raw uint16 DN, not I/F
- **Status:** OPEN · **Severity:** high · **Liveness:** dead-closed (Stage 7 PARKED), but these are the recorded **GO** numbers for the whole Stage-7 build · **Verified:** no
- **Where:** [scripts/probes/_stage7_feasibility.py](../scripts/probes/_stage7_feasibility.py) ·
  **Detail:** [probes-stage7.md](review_2026-07-31/probes-stage7.md) `probes-stage7-1`

The feasibility probe computes every band ratio on **raw uint16 DN** while its own docstring and the
reader-facing `docs/compositional.md` both say I/F. Applying the PDS conversion
(`I/F = DN·SCALING_FACTOR + OFFSET`) changes the ratios, because the three bands have different scaling
factors — so a DN ratio is not proportional to the I/F ratio. Related, same area:
`probes-stage7-4` (low) — the cohort "I/F medians 0.169 / 0.165 / 0.077, inside the expected 0.05–0.30
range for Mars" are **Lambert albedos**, not I/F; the true I/F medians are 0.086 / 0.087 / 0.048.
- **Also from this area:** the published Tier-1 provenance result (**Fisher's exact OR = 23.0,
  p = 0.018**) is the single significant cell of **twelve** analysis choices the two terrain probes
  compute, and it is not the doc's own declared test (`probes-stage7-2`, medium) — it is the **only
  positive empirical evidence** in the compositional writeup. And the Stage-7.0 verdict table compares
  three images whose surviving boulder populations differ **4.4× in median area**, because the
  0.25/0.50 m/px cohort split (**R03**) plus a fixed 8-pixel floor cuts each image at a different
  physical size (`probes-stage7-3`, low).

### R58 — A second probe had R23's data and did not look at the score column
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped (the `min_confidence: null` / `min_size_m: 1.4105` pair is the shipped label basis) · **Verified:** no
- **Detail:** [probes-w1-geospatial.md](review_2026-07-31/probes-w1-geospatial.md) `probes-w1-geospatial-1`

The v2 `detection_filters` decision (`DECISIONS.md:1204`, "~0 % below the floor, so the filter is a
no-op") is recorded from a score distribution computed over **2.45 M rows, 44 % of which the pipeline
deletes before labelling** — i.e. over the pre-drop population. It is the exact statistic that would have
exposed **R23**, computed on the wrong population.

Together with `_diag_vclaire_source_nulls.py` (which examined the two affected images but only broke the
nulls down by `is_at_edge`, never by `score` — see R23), that is **two independent probes that had the
data and stopped one column short**. The "benign density hygiene" framing in the record is the product
of two near-misses, not of an absent investigation.
- **Fix:** recompute the filter-decision statistic on the post-drop population and record it; fold into
  the R23 fix.

### R59 — The published boulder-size audit compares projected areas to a source-pixel threshold
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped — the table is in `docs/methods.md`, the writeup CLAUDE.md points non-coders to · **Verified:** no
- **Detail:** [probes-w1-geospatial.md](review_2026-07-31/probes-w1-geospatial.md) `probes-w1-geospatial-2`

The 5×5-pixel audit compares **plate-carrée-projected** areas against a **source-pixel** threshold. Since
the projection inflates area with latitude, the comparison is not like-for-like: `docs/methods.md`
under-counts sub-threshold detections by **2.5× overall and 11× for one group**. Same root cause as the
already-accepted latitude-distortion systematic (`DECISIONS.md:2741-2751`), but here it lands in a
reader-facing table rather than a caveat.
- **Also from this area (low):** the "presence-AUC coincidence" is an unpaired comparison of means over
  26 vs 25 *different* folds, and "23/38 folds changed" is really 23 of 23 comparable folds
  (`probes-w1-geospatial-3`); "89.8 % agreement after a 1-tile shift" is a base-rate artifact — chance
  agreement at that prevalence is 76.9 %, κ = 0.44 (`probes-w1-geospatial-4`); "seam-tile masking does
  nothing (improved 29 % of images)" is scored on a denominator 37 % of which **cannot move**, because
  14 of 38 images have zero seam tiles (`probes-w1-geospatial-5`).

**Low-severity, dead-closed, recorded for completeness:** `probes-tier2-calibration-4` — `DECISIONS.md`'s
L1 bake-off deltas do not match the committed producer's table and the `hlgauss.mode` paired test
silently drops a fold (verdict "all a wash" unaffected); `probes-tier2-calibration-5` —
`_fm_tier2_ceiling.py` is the declared producer of a "zero-inflation ceiling" but computes no ceiling
(the arithmetic one is 0.9997, so the hypothesis it "empirically refuted" was never arithmetically
in play).

---

## 4h. Findings from `scripts/probes/` (PASS 8) — the published-figure and submitted-PDF findings

> This area (`probes-utility`) was briefed as "expect a thin yield". It produced the most consequential
> live-shipped findings in the whole probe sweep, because it is where **published figures and a
> submitted PDF** are generated. Worth remembering as a lesson about where to look.

> **⚠ REMEDY FOR ALL DOC/FIGURE FINDINGS (Brian, 2026-08-03).** Do **not** retro-fix the numbers in the
> existing writeups — new versions will be written. Instead **date each affected document and add a
> caveat header at the top** listing its known errors and the context. That applies to R60, R61, R62,
> R63 and R51 here, and equally to **R57**/`probes-stage7-4` (`docs/compositional.md`), **R59**
> (`docs/methods.md`), **R48** (`docs/modeling.md`) and **R55**. The findings stay OPEN as *records* —
> what closes them is the header, not a recomputation. Suggested header shape:
>
> ```markdown
> > **⚠ SUPERSEDED / KNOWN ERRORS — as of 2026-08-03.** Written <date>; numbers computed against
> > <label vintage / recipe / cohort>. Known defects, from docs/CODE_REVIEW_2026-07-31.md:
> > * R60 — all results use the pre-sign-fix labels; corrected, "usable" is 26 %, not 14 %.
> > * R61 — the ">90 % agreement" claim is ~10 points above chance at this image's base rate.
> > A revised version is planned; prefer it over this document.
> ```
>
> The point of dating them is that several of these documents are *individually* fine for their vintage
> and only wrong relative to later corrections (R60 is exactly this) — a date plus a defect list makes
> that legible, where a silent edit would not.

### R60 — Every result in the slim writeup and the submitted PDF is computed on the pre-sign-fix labels
- **Status:** OPEN · **Severity:** high · **Liveness:** **live-shipped and externally submitted** — `docs/modeling_slim.md` is linked from `docs/index.md:12-18`; `docs/classification_slimmer.pdf` was submitted for a report (commit `19aa19b`) · **Verified:** no
- **Detail:** [probes-utility.md](review_2026-07-31/probes-utility.md) `probes-utility-1`

The coregistration **sign error** that put v2 labels ~360 m south was found and fixed (DECISIONS
2026-06-10c; the memory note records "pre-fix v2 numbers stale"). The slim writeup's numbers were never
recomputed. Rescoring the *same predictions* against the *corrected* labels moves the headline "usable"
fraction from **14 % to 26 %**.

Note the direction: the correction is **favourable** — the submitted document *understates* the model.
That makes it less urgent than a defect that overstates, but it is still a submitted document whose
numbers do not correspond to the current label set.
- **Fix:** re-run the slim pipeline against the corrected labels, update `docs/modeling_slim.md` and
  regenerate `classification_slimmer.pdf`; note the correction and its direction in DECISIONS. If the
  PDF cannot be re-submitted, record the erratum.

### R61 — The submitted PDF quotes ">90 % agreement" on an image where the worst possible ranking scores 74.6 %
- **Status:** OPEN · **Severity:** high · **Liveness:** live-shipped and submitted · **Verified:** no
- **Detail:** [probes-utility.md](review_2026-07-31/probes-utility.md) `probes-utility-2`

The Conclusions section claims the model "agrees on over 90 % of its rich-tile calls" for an image whose
base rate is high enough that the **worst possible** ranking scores **74.6 %** and a **random** one
**79.7 %**. So ">90 %" is ~10 points above random, not near-perfect agreement. **Pattern B** again
(prevalence wearing another name), and this instance is in the externally submitted document.
- **Fix:** quote the agreement against its chance baseline (or use a chance-corrected statistic such as
  κ, which the project already uses elsewhere — `probes-w1-geospatial-4` computes exactly this
  correction for a different claim).

### R62 — Two published figures claim HiRISE and CTX show "the exact same physical patch"; they are offset by the project's own measured shift
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped — `docs/model_evidence.md` Figure 2 is described in-document as "the scientific basis the whole project rests on"; the same figure is Figure 2 of the submitted PDF · **Verified:** no
- **Detail:** [probes-utility.md](review_2026-07-31/probes-utility.md) `probes-utility-3`

The side-by-side HiRISE/CTX panels are cut from the same nominal bounds without applying the Stage-3
coregistration shift, so they are offset by exactly the amount the project measured for that image —
**116.3 m on the exemplar**. The caption's claim of pixel-for-pixel correspondence is therefore false by
about two-thirds of a 160 m tile, in the figure the document leans on hardest.
- **Fix:** apply the per-image `(dx, dy)` before cutting the panels (`coregister.load_shift` is right
  there), or soften the caption and state the residual.

### R63 — Three more published-figure defects in `docs/model_evidence.md`
- **Status:** OPEN · **Severity:** medium (two) / low (one) · **Liveness:** live-shipped · **Verified:** no
- **Detail:** [probes-utility.md](review_2026-07-31/probes-utility.md) `probes-utility-4/-5/-6`

- **Figure 8** — the burned-in title calls **raw** predictions "calibrated" and repeats a compression
  claim the project **retracted the same day**; the markdown caption beneath it says the opposite. A
  reader gets contradictory statements from the image and its caption.
- **Figure 3, the flagship "visual proof" gallery** — captioned "six images spanning the cohort" but
  contains **no image from the bottom 14 of 38**, and its two nominal "hard cases" rank 15th and 34th.
  It is a favourable sample presented as a spanning one.
- **Figure 1, the document's headline figure** — the "HiRISE ground truth" panel silently
  nearest-neighbour-fills **5.2 %** of its cells and floors the zeros, so the panel is smoother and
  less zero-inflated than the actual label field.

### R64 — Six record-correctness defects in the F leg-B probes, including a key collision
- **Status:** OPEN · **Severity:** high (one) / medium (three) / low (two) · **Liveness:** dead-closed programme; several are quoted in `ROADMAP.md:19` and `PLAN_StripingArtifact.md` · **Verified:** no
- **Detail:** [probes-fbuild.md](review_2026-07-31/probes-fbuild.md)

- **`probes-fbuild-1`** (high) — leg B's "surviving correlate = illumination, exactly A0's cos-i axis"
  is a **brightness/incidence confound**: the ΔAUC signal is the part of scene brightness that
  `cos^k(i)` *cannot* touch, and it was harvested from six unlabelled scenes. The claim is stated as
  mechanism in a committed, executed notebook plus two committed figures, and it selected the next
  experiment.
- **`probes-fbuild-2`** (medium) — a **truncated 3-character pair key collides two frames**, so the
  amended verdict's "prediction disagreement anti-correlates with Δincidence (ρ = −0.33)" is really
  **ρ = −0.03**. This is in the entry that **opened the H1–H6 docket**.
- **`probes-fbuild-4`** (medium) — the "post-minnaert **4.0 %**" figure that opened that docket is
  algebraically a per-pair *constant* rescale, so it cannot distinguish "photometrically correctable"
  from "information-level" — which is the distinction it was used to make. H5 (stronger physics) was
  de-prioritised on this basis.
- **`probes-fbuild-3`** (medium) — the record's diagnosis of the ESP_053989 minnaert inversion is
  arithmetically wrong and refuted by the same probe's own committed table (20 of 36 images get a
  *larger* `cos^k` step and none inverts). The proposed fix targets a non-cause; P4 passed anyway so it
  was never applied.
- **`probes-fbuild-5`** (low) — "over-stretch REFUTED (ρ = +0.09)" could not have produced anything
  else: `f_iqr` is a two-valued constant, so the statistic is the mosaic baseline's own contrast,
  inverted. **Seventh** instance of Pattern A, though here the record does disclose the pinning and the
  conclusion is right.
- **`probes-fbuild-6`** (low) — `DECISIONS` credits DOI verification to two probes that resolve no
  DOIs; the script that does verify checks 4 of ~13 hyperlinked citations, is cited nowhere, and banks
  no log. Against the standing CLAUDE.md rule to hyperlink every citation to its canonical DOI.

---

## 4i. Findings from the second-pass deep re-reviews (PASS 9)

> Commissioned because `geo-crs` and `features` yielded only **one** top-level finding each in pass 1,
> across ~1,450 and ~1,725 LOC — suspiciously thin for the project's self-declared #1 gotcha and for 872
> lines of texture math. **Both reviewers concluded the code is genuinely sound and the *artifacts and
> published statistics* were under-reviewed** — see Pattern D in §3. Single-agent, self-refuted.

### R65 — `peak_correlation`, Stage 3's only quality number, is truncated at its own screening threshold and scores the wrong model
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped — it accompanies every v2 label set · **Verified:** no
- **Detail:** [geo-crs-deep.md](review_2026-07-31/geo-crs-deep.md) `geo-crs-deep-1`

Two defects in one statistic: it is **bounded below by the threshold it is screened against** (so its
distribution is censored and cannot indicate a marginal image), and it is a fit statistic for the
**per-block** shift model rather than for the **global** shift that Stage 4 actually applies. A cohort
screening decision (`DECISIONS.md:2376-2396`) and a published figure (`docs/methods.md` Figure 3) rest on
it. This is **Pattern A one level up** from `geo-crs-2`: pass 1 noticed a `peak_correlation` floor was
*missing* without noticing that adding one there would be vacuous.

### R66 — `ensure_jp2_local` can commit a truncated HiRISE JP2 to the permanent cache
- **Status:** OPEN · **Severity:** medium · **Liveness:** live — every Stage 2 / Stage 3 run and every new manifest row (invariants 4 and 7) · **Verified:** no
- **Detail:** [geo-crs-deep.md](review_2026-07-31/geo-crs-deep.md) `geo-crs-deep-2`

`HTTPResponse.read(amt)` **provably does not raise on premature EOF** — the reviewer cites an explicit
comment in the CPython source, which contradicts pass 1's refutation of this path (pass 1 reasoned from
library semantics rather than from the library source). The only integrity check is a **1 MB floor on
files of 149 MB – 1.31 GB**, so a truncated download is cached permanently and silently.

### R67 — `nominal_footprint_bounds` spends its width in projected metres, clipping the swath by cos(lat)
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped code, live invariant-7 hazard · **Verified:** no
- **Detail:** [geo-crs-deep.md](review_2026-07-31/geo-crs-deep.md) `geo-crs-deep-3`

`nominal_hirise_width_m` is spent in *projected* metres of an equirectangular clon_0 CRS, so the window
covers only `width_m·cos(lat)` of ground. The one current user (the empty-shapefile fallback) is
measurably clipped. Pass 1 gave this path a clean bill — it checked the **CRS** and never the **units**,
which the image's own cached PDS label settles in one line. The same near-miss appears in
`labeling.md`'s refuted list.

### R68 — Stage 4's "runtime pixel-size guard" cannot fire
- **Status:** OPEN · **Severity:** low · **Liveness:** live-shipped · **Verified:** no
- **Detail:** [geo-crs-deep.md](review_2026-07-31/geo-crs-deep.md) `geo-crs-deep-4`

It cannot fire, and it does not test the property it is cited as guaranteeing (the integer-nesting
precondition). **Pattern A**, eighth instance.

### R69 — The hand-crafted features' per-frame nuisance variance is 2–2.5× the embedding η² that launched the striping programme, and was never measured
- **Status:** OPEN · **Severity:** high · **Liveness:** dead-closed for the shipped map (the frozen recipe is emb-only) · **live** for the Tier-1 reference classifier the FM headline is measured against, for every W1 attribution, and for the "per-image heterogeneity" claims · **Verified:** no
- **Detail:** [features-deep.md](review_2026-07-31/features-deep.md) `features-deep-1`

The entire striping/F programme — PLAN_StripingArtifact, the H1–H6 docket, the 907-frame build, ~265
CPU-h + 33 GPU-h — was opened because the *embeddings* carried per-source-frame nuisance variance. The
**hand-crafted features were never measured on the same axis.** They are **2–2.5× worse**, and the
nuisance is concentrated exactly in the families the attribution work leaned on. Nobody looked, because
the striping programme was framed as an embedder problem from the start (`regional_map_rectangular_artifact`
memory: "per-frame radiometry × fixed-`/255` embedder").
- **Why it matters:** the Tier-1 reference classifier that the FM's advantage is measured against runs on
  these features. **R32** already shows that baseline was crippled by an AUC early-stop; this is a second,
  independent reason the FM-vs-Tier-1 margin is not currently trustworthy.

### R70 — Two derived feature caches and their packaged splits are two generations stale, and the staleness detector misses it
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-on-disk hazard on a path `DECISIONS.md:2514` explicitly keeps open ("Artifacts kept") · **Verified:** no
- **Detail:** [features-deep.md](review_2026-07-31/features-deep.md) `features-deep-2`

`features_nbr_s5` and the Stage-6 caches carry **pre-DN-clip-fix dead shadow features** and
**pre-coreg-sign-fix labels**. The built-in staleness detector checks the `features/*.json` sidecars —
which are all clean — and never the derived caches one directory over. Compare **R04** and
`other-scripts-1`: this is the third independent finding that stale derived artifacts are undetectable
here.

### R71 — The shadow detector's zero point is per-image, but the DN zero point moves up to 47 DN *between source frames inside one window*
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed for the shipped map; live for every GBM/Tier-1/W1 number off `dataset*/features/` · **Verified:** no
- **Detail:** [features-deep.md](review_2026-07-31/features-deep.md) `features-deep-3`

The offset used is 20 DN; the actual between-frame movement inside a single window is up to **47 DN,
2.4× larger**, and the docstring claims otherwise. So `shadow_fraction` partly measures which source
frame a tile came from — the same mechanism as **R28** (Canny) and **R69**, and the same mechanism the
striping programme existed to fight.

### R72 — Stage 6b propagated a physically impossible illumination geometry from one corrupt SeamMap row to all 115,878 tiles of an image
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed (Stage 6b / W2 closed) but the columns are on disk and one queued work item is motivated by the error · **Verified:** no
- **Detail:** [features-deep.md](review_2026-07-31/features-deep.md) `features-deep-4`

No illumination angle is range-checked anywhere, so one corrupt SeamMap row poisoned every tile of
`ESP_068483_2280`. This **answers the question `probes-stage6-4` punted on** (why `std_ctx_incidence` is
two different statistics in two probes). Note the image is also one of R23's two truncated-label images —
worth checking whether the two defects interact.

### R73 — `docs/methods.md` §7.4 reads five GLCM entries as converging evidence; three are one statistic
- **Status:** OPEN · **Severity:** low · **Liveness:** live, reader-facing (cf. **R44**) · **Verified:** no
- **Detail:** [features-deep.md](review_2026-07-31/features-deep.md) `features-deep-5`

ρ = 1.0000 between `energy` and `ASM`; −0.9998 between `homogeneity` and `dissimilarity`. Five
"independent" GLCM signals are really about three. Same family as **R55** ("confirmed five ways" sharing
one embedding) and **R51** (12 correlated re-analyses counted as 12 observations): **counting correlated
evidence as independent** now appears in three separate documents.

---

## 4j. Findings from the `labeling-deep` second pass (PASS 10, 2026-08-04)

Four sub-areas — [labeling-deep-footprint](review_2026-07-31/labeling-deep-footprint.md),
[-artifact](review_2026-07-31/labeling-deep-artifact.md),
[-semantics](review_2026-07-31/labeling-deep-semantics.md),
[-tests](review_2026-07-31/labeling-deep-tests.md). Brief:
[_prompts_labeling_deep.md](review_2026-07-31/_prompts_labeling_deep.md).
Opened because Pattern D said pass 1 audited the *computation* and never the *artifact*, and because
the label basis carries the register's #1 (**R23**) and a confirmed **R03**.

**The headline is a refutation.** The area's own top question — *does BoulderNet's inference footprint
have interior gaps, so that some zero labels are false zeros?* — is **REFUTED** on four independent
tests: no crop (HiRISE-valid ground extends 21–29 m median beyond the extreme detection on all four
sides, 38/38 images), no margin (unshifted detection density at 40 m from the boundary is **1.506 ±
0.112×** the image mean — *enriched*, and the shifted control does show a deficit, so the test has
demonstrated power), no detector grid (spectral amplitude at the 512 m CCD pitch and the SAHI 256 m /
204.8 m tiling is *below* median), and no geometric holes (150 enclosed zero-components, rectangularity
max 0.859 — none is a rectangle). **Do not re-open this.**

### R74 — The HiRISE coverage mask calls deep-shadow pixels "no coverage", silently deleting 1.97 % of S=32 tiles that are 93 % rich
- **Status:** **CODE FIXED 2026-08-04, REBUILD DEFERRED** — see [PENDING_REBUILD.md](PENDING_REBUILD.md) #1. `src/ctx_retrieve.py` gained `_fill_interior_shadow_holes`, called from `build_hirise_coverage_mask` (`max_interior_hole_px=16`; `0` restores the old behaviour). Validated read-only on the 138 cached decimated arrays: every re-marked pixel has DN exactly 0; `ESP_017355_2260` re-marks 1,185 px, reproducing the measurement below exactly; the fix only ever *adds* coverage and never alters the swath border; `pytest -m "not slow"` unchanged at 490 passed. **The artifacts are deliberately not regenerated yet** — per policy, all rebuild-requiring fixes are batched into one re-run once the review is complete. **2026-08-06 — the audit's two pre-rebuild conditions are now met.** (a) *Tests:* ten direct synthetic tests (`tests/test_coverage_mask_shadow_fill.py`) covering a small enclosed hole, a hole above the threshold, the inclusive threshold boundary, an edge-connected invalid region, the mixed enclosed-plus-edge-connected case, `max_interior_hole_px <= 0` as an exact no-op, add-only-never-remove over random fields, and the all-valid / all-nodata degenerate cases. (b) *Provenance:* the threshold is now a config key (`ctx_retrieve.max_interior_hole_px`) wired through `scripts/run_stage2.py`, and `build_hirise_coverage_mask` returns `(path, fraction, provenance)` carrying `method`, `version` (2 = post-R74), threshold, filled-pixel count and the output mask's SHA-256. Stage 2 persists that plus the CTX window's digest; Stage 3 records both input digests and the mask identity and emits a `shift_id` over its shift + inputs; Stage 4 records `inputs.{ctx_window_sha256, hirise_mask_sha256, coverage_mask, coreg_shift_id}`. A test flips one mask pixel and asserts the Stage 4 sidecar changes while the config hash does not · **Severity:** high · **Liveness:** live-shipped (`dataset_v2/labels` is the basis of the frozen recipe, the deployed head, the banked calibrator and the shipped map) · **Verified:** no (single-agent, but measured on all 38 images)
- **Where:** [src/ctx_retrieve.py:507](../src/ctx_retrieve.py#L507) · **Detail:** [labeling-deep-footprint.md](review_2026-07-31/labeling-deep-footprint.md) `-1`

Coverage is defined as `hi_arr > 0` on a **nearest-neighbour** 5 m decimation. HiRISE DN is continuous
through 0 (DN = 1, 2, 3 … all populated), so deep-shadow pixels are classified "not observed" — and
because eligibility is `all(mask == 1)`, **one 5 m pixel deletes a 160 m tile**. Measured: **3,236 S=32
tiles (1.97 %) dropped, 93.0 % of them rich against a 36.0 % base rate, holding 7.70 % of all detected
boulder area**, mean `fa` 4.15× the kept tiles. Interior zeros are 99 % isolated single pixels with
systematically darker neighbourhoods (5/5 images checked) — not the "missing scans" the docstring
claims. The refutation-of-the-refutation: **BoulderNet detected boulders inside those tiles at 3×
density**, so the data is plainly there. This is a *shadow-biased* deletion of exactly the rock-rich
tiles the model is trained to find. Unpinned by tests (every fixture uses a synthetic `np.full` mask);
nothing in `DECISIONS.md` anticipates it.
- **Fix:** one line at the producer — define coverage from the nodata mask / validity band rather than
  `> 0`, or decimate with a min-filter over the block. Then quantify what the 3,236 recovered tiles do
  to the frozen recipe's metrics.

### R75 — `labeling-2` measured on the cached masks: the two reviewers counted different populations, and both were right
- **Status:** **RESOLVED 2026-08-04** by a third, independent measurement · **Severity:** low for the strict defect (as pass 1 had it); **medium** for the partial-depression effect neither reviewer named · **Liveness:** live-shipped · **Verified:** **yes — three independent measurements now agree**
- **Detail:** [labeling-deep-footprint.md](review_2026-07-31/labeling-deep-footprint.md) `-2` **and** [labeling-deep-artifact.md](review_2026-07-31/labeling-deep-artifact.md) `-3`

Pass 1's swath-edge zero strip was analytic (~2 % of tiles, ~60/image). The two second-pass reviewers
measured it on the cached `*_hirise_mask.tif` and appeared to contradict each other — 3.89 % vs 0.21 %.
**They were measuring different populations.** Reconstructing the vacated region directly (mask ∧
¬shift(mask), then per-tile coverage via an integral image over all 38 images, 161,005 eligible S=32
tiles):

| population | measured | share | reviewer it matches |
|---|---|---|---|
| tiles **overlapping** the vacated strip | **6,202** | **3.85 %** | `-footprint`'s 3.89 % |
| tiles **fully inside** the vacated strip | **340** | **0.21 %** | `-artifact`'s 337 / 0.21 % |
| …of which `fa == 0` | **340** | **100 %** | `-artifact`'s self-check (0 of its 337 had `fa > 0`) |
| share of the whole zero class | — | **1.17 %** | `-artifact`'s 1.16 % |

**Which number is the finding.** Only the **340 tiles (0.21 %)** are labelled zero *by construction* —
a partially-vacated tile can still legitimately hold detections in its remaining area, so its zero (if
it is zero) is not forced. Pass 1's "low" was the right severity for the strict claim, and `-artifact`'s
"~7× too high" correction to pass 1's analytic estimate stands.
**But `-footprint` found something real that neither filed as its own effect:** the other **5,862 tiles
have a *partially* depressed `fa`** — part of their area cannot contain a polygon — which is a milder
bias over an 18× larger population, and it is not captured by counting zeros. That is the part worth
carrying forward. `-footprint`'s other two claims are unaffected and stand: the strip is an **L along
the southern *and* western** edges (dy>0 in 38/38, dx>0 in 30/38), and the shift pushes **82,210
detections (1.39 %)** out of the labelled area.
- **One number does not reconcile:** `-footprint` states both "3.89 %" and "2,502 tiles", but 2,502 is
  not 3.89 % of 161,005 (that is 6,263) nor of its own 164,273 interior grid (1.52 %). The *percentage*
  reproduces; the count does not correspond to any denominator in play. Treat the count as suspect.
- **Fix:** report the partial-depression population (3.85 % of tiles, area-weighted) alongside the
  by-construction count (0.21 %), and shift the mask with the polygons in Stage 3 so neither arises.

### R76 — R23's score truncation moves ~2,200 tiles into the wrong class of the frozen recipe's actual target
- **Status:** OPEN · **Severity:** medium (extends **R23**, filed blocker) · **Liveness:** live-shipped · **Verified:** no
- **Detail:** [labeling-deep-footprint.md](review_2026-07-31/labeling-deep-footprint.md) `-3`

R23 is filed in terms of depressed `fa`. Its consequence on the *actual* target `fa_gt_1e-2` is a
prevalence-matched flip rate of **41.3 %**, i.e. **≈2,200 of `ESP_017355_2260`'s tiles sit in the wrong
class**. Also narrows pass 1's "2.5–4.5×" `fa` depression to **≈2.6×**. The reviewer killed its own
first version: the pooled 51.8 % rate is prevalence-dependent (ρ = −0.757) and implies an impossible
>100 % true rich share.

### R77 — **Six** `slow` tests write into the live `dataset/` and `cache/` trees
- **Status:** **FIXED 2026-08-06** (redirects 2026-08-05; residual staging/isolation hole closed 2026-08-06). Mutable derived artifacts are copied rather than hard-linked, a session-wide runtime guard refuses any write under `cache*/`, `dataset*/`, `models/`, `reports/`, and a static AST scan fails even when a producer test skips. See "residual hole" below for the corrected mechanism. · **Severity:** high · **Liveness:** live-shipped · **Verified:** **yes — reproduced by accident, which is how it was found**
- ⚠ **The review undercounted by half.** It named three; fixing them and re-running the *full* suite exposed **three more**, in files no reviewer had flagged: `tests/test_stage2_one_image.py` (×2 — `stage2_one_image`, which writes `ctx_windows/` **including the HiRISE coverage mask**), `tests/test_coregister.py` (×2 — `stage3_one_image` → `coregistration/`), and `tests/test_sanity_residual_one_image.py` (`stage1_one_image` → `reprojected_detections/`; **DECISIONS already recorded this one silently rewriting a cache file on 2026-06-10**, but nobody connected it to the pattern). The lesson: *the only reliable detector for this defect class is running the suite and diffing the tree*, which is now the standing check.
- **Fix as applied (2026-08-05):** the three `dataset/`-writing tests take `tmp_path` (the features one stages its labels into the tmp tree, since Stage 4b reads labels from `output_dir`). The three `cache/`-writing tests use a new `read_only_cache` factory fixture in `tests/conftest.py`, which **hard-linked** the read-side subdirs into a throwaway cache.
- **Residual hole and its closure (2026-08-06):** the hard link was justified by "each producer's read and write subdirs are disjoint", and `hirise_decimated/` broke that invariant — it is staged for reading *and* rewritten by `read_full_footprint_decimated` when the cached CRS is stale. **The 2026-08-06 audit's stated mechanism does not reproduce**: measured on rasterio 1.5.0 / GDAL 3.11.4 / NTFS, `rasterio.open(path, "w")` deletes-then-creates, which breaks the link and leaves the source intact. In-place writers *do* write through the link — `open(p,"wb")`, `rasterio.open(p,"r+")`, `Path.write_text`, `shutil.copy2` all clobbered the source in a controlled probe — so the hazard is a real design defect that current library behaviour happens to mask, not a live data-loss path anyone has demonstrated. The fix removes the dependency on that library detail: mutable derived artifacts are **copied** and only `{tile}.zip` / `{obs}_RED.JP2` are linked (sidecars beside them, including GDAL `.aux.xml`, are copied). Added: a session-wide runtime write guard (`tests/live_artifact_guard.py`), a static AST scan that fails even when the producer test skips, an end-to-end stale-CRS regression on two temporary roots that exercises the invalidation branch the 511-pass run never entered, and a teardown assertion that every linked source is unchanged.
- **Where:** `tests/test_labeling.py::test_stage4_runs_on_ESP_069669_2220` (passes `output_dir=cfg.output_dir`), `tests/test_empty_shapefile.py` (passes `cache_dir=cfg.cache_dir`), and — found later by `tests-deep-features` — **`tests/test_features.py:485-495::test_features_align_with_labels_row_for_row`**, which calls `stage4b_one_image(..., output_dir=repo_root/"dataset")` and overwrites the features parquet, its sidecar and **both context-patch `.npy` stacks** · **Detail:** [labeling-deep-tests.md](review_2026-07-31/labeling-deep-tests.md) `-1`, [tests-deep-features.md](review_2026-07-31/tests-deep-features.md) `-1`

**The third instance is the worst of the three, because it buys nothing.** Its assertion
(`label_keys == feature_keys`) **cannot fail**: `stage4b_one_image` emits one row per label row by
iterating the labels groupby, so row-for-row alignment is true by construction. It is a producer
against the live tree in exchange for a tautology. It would also now *launder* the 2026-08-04 labelling
incident forward into the features tree.
**All three are `slow`-marked** — verified — so `pytest -m "not slow"` cannot reach them, which is why
the fast suite is safe to run and was confirmed clean after the R74 fix. (2026-08-06: `slow` is *not*
the guarantee. Re-auditing markers found **20 non-slow tests that call a producer**, all writing to
`tmp_path`. The static scan and runtime guard, not the marker, are what make the fast loop safe.)

Running the labelling suite **silently overwrites the provenance an audit reads**, and the rewrite is
**not value-preserving**: against the untouched 2026-05-23 packaged vintage, `max|Δfa|` = 0.115 (S=8)
and `max|Δcount|` = 115 (S=64), because the v1 labels predate the 2026-06-10 y-sign fix — so the test
migrates one of nine v1 images across a **correctness boundary**. `dataset_v2`/`cache_v2` are untouched
(verified). **This has happened before**: `cache/reprojected_detections/ESP_069669_2220.json` was
rewritten 2026-06-10 by `test_sanity_residual_one_image.py`. The underlying hazard is broader than the
tests — the producers write to config-derived live paths with **no dry-run mode**
([labeling.py:543](../src/labeling.py#L543), [:591](../src/labeling.py#L591),
[detections.py:151](../src/detections.py#L151), [coregister.py:436](../src/coregister.py#L436)) — so
*any* audit that calls one mutates the dataset. `load_shift` is a pure read (verified).
- **Occurred 2026-08-04, and was restored:** a review agent ran the suite and overwrote
  `dataset/labels/ESP_069669_2220.{parquet,json}` + `cache/reprojected_detections/ESP_065711_1545.{gpkg,json}`.
  Recoverable only by luck — the original label values survived in
  `dataset/packaged/loio_9fold/y_test_fold6.parquet`. All 96,354 rows were restored to an exact match
  (0 differing values across all 7 label columns) and the sidecar carries a `restored_from` block.
  The reviewer brief ([_prompts.md](review_2026-07-31/_prompts.md) §1) now carries an explicit
  no-producer rule.
- **Fix:** point both tests at `tmp_path`. Then consider a `dry_run=` on the producers, since the tests
  are only the most visible caller of a general hazard.

### R78 — Every end-to-end fixture pins the mosaic grid phase to zero — a configuration no production image has. **Now confirmed in three suites.**
- **Status:** **FIXED 2026-08-06, both mutants killed.** All three suites carry the real phase `(894, 12645)` read off `dataset_v2/labels/ESP_042964_2160.json`. Mutant verification, run on two independent scratchpad copies (the working-tree `src/` was never modified): **(a)** dropping the mosaic origin from the labels bounds arithmetic (`src/labeling.py:367-370`) now fails `test_tile_bounds_align_with_mosaic_pixel_grid` — the very assertion whose docstring names the failure — plus `test_label_transforms_emit_expected_columns`; **(b)** flipping the origin sign in the features window arithmetic (`src/features.py:653-654`, `- origin` → `+ origin`) now fails four Stage-4b tests on the bounds guard (`RuntimeError: some Stage 4 tiles fall outside the cached CTX window`). Both left the suite green before. The last `(0,0)` call site (`test_ctx_source_illumination.py::test_add_features_end_to_end`) was re-based on the real phase; because `894 % 4 = 2` and `12645 % 4 = 1`, its tiles now start at window row 2 and column 3, so a row/col swap fails too. **`test_alignment_aligned_window` keeps its zero origin deliberately** — its docstring makes the aligned case the point and `test_alignment_offset_window` supplies the `(3,5)` complement. · **Severity:** high · **Liveness:** live-shipped · **Verified:** **yes — by mutation, in both suites**
- **Detail:** [labeling-deep-tests.md](review_2026-07-31/labeling-deep-tests.md) `-2` **and** [tests-deep-features.md](review_2026-07-31/tests-deep-features.md) `-3`

`test_tile_bounds_align_with_mosaic_pixel_grid` **cannot detect the failure its own docstring names**:
dropping the mosaic origin from the bounds displaces real `ymin` by **2,608 km** and the suite stays
green. `tests-deep-features` then found the same shape in `test_features.py` — every fixture there
pins the origin to `(0, 0)`, which **0 of 52 production sidecars** has (real ranges 894–43,790 and
183–41,945) — and it is *why* that suite's origin-sign-flip and bounds-guard mutants both survive.

**This is the third instance, and the first two are not hypothetical:** `src/fgates.py:211-231`
already records this exact fixture defect as the cause of the **~100 km gate mis-key**
(*"the old unit test … pinned `row0=col0=0`"*). A defect that has now been found in three separate
suites, having already caused one real error, is a systemic fixture-design problem rather than three
bugs.
- **Fix:** parameterise the shared fixtures over a non-zero `(row0, col0)` drawn from a real sidecar —
  one change, three suites. This is the highest-leverage single test fix in the register.

### R79 — `boulder_count` can be identically zero on every tile of every image and the labelling suite stays green
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped (`boulder_count` is a packaged target, `src/dataset.py:61`) · **Verified:** no (measured by mutation: 5,646 → 0 on the real image, 20 passed)
- **Detail:** [labeling-deep-tests.md](review_2026-07-31/labeling-deep-tests.md) `-3`

Part of a broader result from **mutation testing** (`src/` copied to a scratchpad, 25 seeded defects):
**16 of 20 survive `pytest -m "not slow"`** — CLAUDE.md's documented dev loop — and **12 of 20 survive
the full suite**. The tests do not pin wrong science; they pin much less than they appear to.
- **Fix:** the area file lists which mutants each surviving assertion would have caught, so the gaps are
  addressable individually rather than by a rewrite.

### R80 — The size-floor filter, the mechanism behind R03, is the least-tested code in the module
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped (`min_size_m: 1.4105` is set in both configs) · **Verified:** no
- **Detail:** [labeling-deep-tests.md](review_2026-07-31/labeling-deep-tests.md) `-4`

The fixture **cannot distinguish diameter from radius**, is exercised in EPSG:4326 (degrees², geopandas
warns), and **no end-to-end test wires a non-`None` filter at all**. Given R03 is confirmed and the
floor is the mechanism, this is the untested code with the highest known consequence.

### R81 — The entire v1 `dataset/` label tree is a pre-y-sign-fix generation that today's code cannot reproduce
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped — the v1 baseline is `docs/modeling_results.md` §§1–8, which `README.md` / `docs/index.md` route external readers to; the go-forward recipe and shipped map are v2 and unaffected · **Verified:** no
- **Detail:** [labeling-deep-artifact.md](review_2026-07-31/labeling-deep-artifact.md) `-1`

`_w1_migrate_coreg_sign.py` migrated `cache/coregistration/` (all 9 now `+dy`) but Stage 4 was re-run
for **v2 only** (`DECISIONS.md:2577`), so every v1 sidecar records the **sign-inverted** `dy` against a
Stage-3 cache that carries the corrected sign. Every v1 label field sits **236–493 m south** of its CTX
texture. Related: **R44**'s verifier found `docs/methods.md` §5's `dy` column has the same vintage.
- **Fix:** either re-run Stage 4 for v1 and re-derive the v1 numbers in `modeling_results.md`, or label
  the v1 sections as a superseded generation and stop citing them as current. **Note R44's verifier
  established every v1 number reproduces exactly against the tree as it stands** — so this is a
  correctness-of-basis issue, not an arithmetic one.

### R82 — Two of the four packaged v2 label artifacts carry pre-sign-fix labels, and the provenance field a consumer would check is inverted
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped (artifacts and their consuming scripts are on disk and runnable; the verdicts they produced are dead-closed) · **Verified:** no
- **Detail:** [labeling-deep-artifact.md](review_2026-07-31/labeling-deep-artifact.md) `-2`

`packaged/loio_nfold_ctx_illum` and `loio_nfold_nbr_s5` carry pre-fix targets: **65.4 % / 88.3 %** of
`fractional_area` values differ from the labels and **19.4 % / 13.6 %** of tiles flip the frozen
`fa > 1e-2` class. The provenance is **inverted** — the two *stale* packages' `config_hash` equals the
labels' own, while the two *correct* ones differ. `_sweep_stage6b.py` would today compare a post-fix
baseline against a pre-fix arm. (Contrast the clean result below: for v2 `dataset_v2/labels`, all
3,564,767 rows are self-consistent with their sidecars and both *live* packaged schemes match the
labels bit-for-bit.)

### R83 — The `fa > 1e-2` rich/poor class is cohort-dependent, and it is a re-ranking no level correction can absorb
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped (the target of the frozen recipe + shipped map) **and** live-active-plan · **Verified:** no
- **Detail:** [labeling-deep-semantics.md](review_2026-07-31/labeling-deep-semantics.md) `-1`

Restated on one common size floor, the 0.25 m/px cohort's rich prevalence **halves** (0.326 → 0.164)
while the 0.50 m/px cohort's does not move (0.369 → 0.366); up to **64 %** of one image's tiles flip.
Critically it is a **re-ranking, not a rescale** (within-image Spearman 0.60–0.98 fine vs 0.96–0.9998
coarse), so A1 / H1 / H4 / calibration **cannot** absorb it. Holding the shipped predictions fixed, the
committed `striping_a1_loio_summary.csv` moves: `median_auc` 0.79038 → **0.7968**, `pooled_pr_auc`
0.77729 → **0.7728**; individual fine-image AUC by −0.18 to +0.43.
**Corrects R03** (twice): the *post-filter* floors are 1.411–1.427 m vs 1.943–2.664 m diameter — a
**1.9–3.6×** area gap, not 3–4× — and it is the **coarse** cohort that is internally heterogeneous.
**Corrects R03's verifier:** its "93–99 % area recovery" used a whole-image denominator; on the same
eligible tiles the 5× rasteriser recovers **99.7–100.2 %** with no cohort difference, so
`methods.md`'s unbiasedness claim stands.
**Strongest self-refutation, and why this is medium not high:** `Spearman(sub-floor area share,
per-image AUC) = −0.468, p = 0.003` survives every control — but **also survives inside the 0.50 m/px
cohort alone** (−0.467, p = 0.016, n = 26), so it is mostly small-boulder terrain, not pixel scale.

### R84 — The deployed abundance layer publishes a physical quantity whose size convention is an unrecorded 78/22 mixture of two floors
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped (`*_abundance.tif` is the deliverable) **and** live-active-plan · **Verified:** no
- **Detail:** [labeling-deep-semantics.md](review_2026-07-31/labeling-deep-semantics.md) `-2`

Proved from `calibration.npz`: `t2_y` max = **0.293242** = exactly the max `fa` of the 161,005-tile
pool, so the abundance layer is quantile-matched onto a **78.4 % coarse / 21.6 % fine** mixture of two
floors. `write_geotiff` writes **no tags at all**. **R03's recommended remedy (emit `map_scale_mpp` +
the measured floor per image) is necessary but not sufficient** — fine for PLAN_RegionalMap legs 1–3
(within-map rank statistics), insufficient for **leg 4** and for any external comparison, because the
deployed layer's floor is a *mixture* that no per-image sidecar can state.
- **Fix:** compute and record the mixture floor as a product-level attribute on the GeoTIFF.

### R85 — `boulder_count` / `count_density` are 5–10× more cohort-distorted than `fa`
- **Status:** OPEN · **Severity:** medium · **Liveness:** mixed — `count_density` is live-published in every label parquet; the Stage-7d partition is dead-closed · **Verified:** no
- **Detail:** [labeling-deep-semantics.md](review_2026-07-31/labeling-deep-semantics.md) `-3`

**~90 %** of a fine image's counted boulders lie below the coarse floor, against 1.5 % of its *area*.
The published Stage-7d `boulder_count > 50` partition moves **0.621 → 0.244** on the fine cohort
(0.439 → 0.429 coarse). Distinct from **R50** (which is about the positive-class definition changing).

### R86 — Four lower-severity label-record defects
- **`semantics-4`** (low, live-shipped) — the only published number describing the size floor,
  `n_polygons_after_filter`, equals `n_polygons_stage1` for **26/26** coarse images and **0/12** fine:
  it reports "nothing was filtered" precisely where the floor is most wrong. `detection_filters` is
  byte-identical across all 38 sidecars. Same shape as `geo-crs-deep`'s `peak_correlation`. **Includes
  the clean negative the brief asked for: no image beyond R23's two has a non-uniform confidence basis**
  (36/38 have score min exactly 0.100000) — but `DATA_DICTIONARY.md:19`'s "0.10–0.83" is wrong in both
  directions (measured **0.100–0.956**).
- **`semantics-5`** (low) — all four documents defining the target omit the size floor;
  `methods.md:753` states a `binary_by_count` default of 5 against a shipped 1.
- **`artifact-4`** (low) — `config_hash` cannot detect label staleness **in either direction** and is
  **read by zero call sites**: it moved when `validation_rasters` changed (irrelevant) and stayed fixed
  across the y-sign fix (code, not config), so all 38 v2 sidecars now mismatch the current config.
- **`artifact-5`** (low) — `DATA_DICTIONARY.md:134` still defines `shift_m.dy` with the pre-fix,
  sign-inverted formula (distinct from R44's `methods.md` instance).
- **`tests-5`/`-6`** (low) — `test_stage4_runs_on_ESP_069669_2220` is a runs-not-right test (six of
  seven assertions cannot fail on a wrong labeller); `test_empty_shapefile.py:32`'s
  `assert crs is not None` pins nothing about the target CRS.

### Verified clean by this pass — do not re-file
- **BoulderNet's inference footprint equals the HiRISE image footprint** (four independent tests, above).
- **`dataset_v2/labels` is internally sound**: all 3,564,767 rows self-consistent with their sidecars
  and `_flatten_to_dataframe`; both *live* packaged schemes match the labels bit-for-bit (0 differing
  values); both live split JSONs rebuild to an identical `split_hash`.
- **The v2 LOIO splits do not have `within_image_4fold`'s vintage problem and structurally cannot** —
  fold *i* is `sorted(obs_ids)[i]`, content-independent; verified identical across all 3 v2 LOIO splits
  and 3 packaged metadata. R45's drift is specific to `_compute_quadrant_definitions`, which medians
  over label rows. (The **v1** `within_image_4fold.json` *does* have it: 543 of 27,307 S=32 tiles,
  1.99 %, in a different quadrant than today's splitter assigns.)
- **The labelling tests pin no wrong science** — no assertion defends a known defect. R23,
  `labeling-2` and R03 are pinned *nowhere*. What the tests genuinely do pin is enumerated in the area
  file (eligibility semantics, sub-pixel area arithmetic, the ×2 ladder, centroid ownership, shift sign,
  idempotency, the column contract), each named by the mutant that it killed.
- **`load_shift` is a pure read** — not the cause of the 2026-08-04 artifact writes.

---

## 4k. Findings from the `tests-deep` pass (PASS 11, 2026-08-04) — **COMPLETE, 4 of 4 areas**

Mutation testing of the four large test bodies §6 named as the highest-yield code-reading left (the
fifth, `test_labeling.py`, was done in PASS 10). Brief:
[_prompts_tests_deep.md](review_2026-07-31/_prompts_tests_deep.md). Areas:
[splits](review_2026-07-31/tests-deep-splits.md) · [features](review_2026-07-31/tests-deep-features.md)
· [within-image](review_2026-07-31/tests-deep-within-image.md) ·
[region-staged](review_2026-07-31/tests-deep-region-staged.md).

**Across ~100 seeded defects in five suites, not one assertion was found defending a known defect.**
The register's hypothesis for this work — "assertions that pin wrong science" — is **refuted**. The real
answer is uniform: *the suites pin far less than they appear to*, and roughly **half of all seeded
defects survive**.

**Structural finding: the fast-vs-full survival gap is exactly zero in every suite**, for four different
reasons — `splits`' `slow` tests only call the metadata loaders and can never reach
`build_split`/`package_split`; `features`' one `slow` test asserts something true by construction;
`region-staged` and `within-image` have no reachable `slow` coverage of the code under test. So
`pytest -m "not slow"`, CLAUDE.md's documented dev loop, is **not weaker** than the full suite — the
full suite simply is not stronger. That is a useful licence: the fast loop is the right one.

| suite | survived (fast) | survived (full) | honest rate after discarding equivalent mutants |
|---|---|---|---|
| `test_labeling.py` (PASS 10 → R77–R80) | 16/20 | 12/20 | — |
| `test_splits.py` (R87, R88, R90) | **10/16** | **10/16** | **8/14 = 57 %** |
| `test_features.py` (R89, R90) | **12/22** | **12/22** | — |
| `test_within_image_split.py` (R91, R92, R96) | **10/15** | **10/15** | **9/14 = 64 %** |
| `test_region_staged.py` (R93–R96) | **15/25** | 11/25 *(with the unit suites)* | **60 % / 44 %** |

**The dominant cause is fixture degeneracy, not missing assertions** — and that reframes the fix.
`within-image` proved it quantitatively: its suite already *has* the assertion that catches the worst
mutant, but every fixture image is the same symmetric 64×64 square, so the mutant moves 0 tiles instead
of collapsing 8 of 9 real images into one quadrant. Fix the fixtures and much of the existing coverage
starts working. See **R78** (the (0,0) mosaic origin, now in three suites) and **R91**.

### R87 — A `package_split` fallback to a random per-tile split leaves the test suite fully green
- **Status:** **FIXED 2026-08-06 (guard added, three mutants killed).** `test_packaged_folds_contain_exactly_the_split_obs_ids` asserts per-fold `obs_id` set membership in all four packaged parquets *and* in `groups_*.npy`, plus train/test disjointness and hold-out-exactly-once across the scheme. `test_within_image_packaged_folds_contain_exactly_the_expected_tiles` does the same for the (image, quadrant) arm on exact tile-key sets. Killed on scratch copies: **(i)** LOIO random per-tile re-split with counts preserved, **(ii)** within-image random per-tile re-split, **(iii)** within-image train rows taken from the *wrong image*. Each left every pre-existing assertion green, which is exactly the review's point. · **Severity:** high · **Liveness:** live-shipped (the guard was absent; the splitter itself is correct today) · **Verified:** **yes — by mutation**
- **Detail:** [tests-deep-splits.md](review_2026-07-31/tests-deep-splits.md) `-1`

This is the exact **invariant-6** violation — the one whose occurrence would invalidate every number the
project reports — and nothing would catch it. Every packaging assertion is a **row count or a length**;
nothing checks *which* `obs_id`s land in `X_test_fold{k}.parquet`. Demonstrated: all 4 fixture images
placed in both train and test while `n_test/n_train` stayed exactly 10/30, green. It also survives
`test_within_image_split.py` + `test_modeling_loaders.py` (41 passed).
**Calibration:** `labeling-deep-artifact` established the v2 LOIO splits *are* correct and structurally
cannot drift. This finding is about the **absence of a regression guard**, not a live wrong number.
- **Fix:** assert set membership of `obs_id` per fold in the packaged parquets, not just row counts.

### R88 — The X/y column split is unpinned, so a label column entering the feature matrix would be silent
- **Status:** **FIXED 2026-08-06 (both halves of the fix, mutant killed).** Stage-5 side: `test_packaged_x_columns_are_exactly_the_expected_feature_set` pins the emitted X and y column sets exactly (not counted), for train and test, on every fold; `test_within_image_packaged_x_never_carries_a_label_column` covers the shared `_split_columns` on the within-image arm. Loader side: `src/modeling/loaders.py::_feature_columns` now **raises** on any `FORBIDDEN_X_COLUMNS` member, checked on both the train and test parquet — it raises rather than filtering, because a target in X means the package is corrupt and silently dropping it would hide that. Dropping `label_cols` from `_split_columns`' exclusion set on a scratch copy now fails two tests; splicing a label into a packaged X now fails `load_fold`. Verified read-only over **all 620 packaged X parquets** in `dataset/` and `dataset_v2/`: none carries a label column, so this is a regression guard, not a migration. · **Severity:** high · **Liveness:** live-shipped (guard was absent) · **Verified:** **yes — by mutation**
- **Detail:** [tests-deep-splits.md](review_2026-07-31/tests-deep-splits.md) `-2`

Dropping `label_cols` from `_split_columns`' exclusion set puts `fractional_area` into the feature
matrix and the suite stays green. [loaders.py:91-95](../src/modeling/loaders.py#L91-L95) has **no second
filter**, so the failure mode is a silent perfect-score leak — the single most misleading result this
codebase could produce.
- **Fix:** assert the emitted feature columns against an explicit expected set, and add a
  target-absence assertion in `loaders`.

### R89 — 12 of 22 seeded feature defects survive, including the entire labels→window registration arithmetic
- **Status:** OPEN · **Severity:** medium · **Liveness:** live-shipped · **Verified:** no (demonstrated by mutation)
- **Detail:** [tests-deep-features.md](review_2026-07-31/tests-deep-features.md) `-2`, `-4`, `-5`, `-6`

Survivors include a `_stack_tiles` row/col transpose, an origin sign flip, deletion of the bounds
guard, collapsing the GLCM distance→column mapping, inverting `grad_dir_circvar`, halving
`edge_density`, and an off-by-one in the gliding-box loop. Two amplifiers: the HiRISE mask in every
fixture is **all ones**, so deleting `arr[mask == 1]` and hardcoding `valid_pixel_fraction = 1.0` both
pass — exactly the train/deploy seam `features-5` flags; and Stage 4b runs end-to-end at **S=16 only**
(measured: 58 columns, zero `lacunarity_*`), a scale the frozen recipe never uses, so lacunarity is
never exercised end-to-end at all.
- **Useful negative, and the cross-check this pass was sent to do:** **no known feature defect is
  pinned as intended.** The register's own fixes for **R27** (lacunarity → NaN instead of the 0.0
  sentinel) and **R28** (canny quantile thresholds) were executed *as mutants* and left the file green.
  **Both can be applied without touching a test.**

### R90 — Lower-severity test-coverage gaps from PASS 11
- **`splits-3`** (medium) — `groups_*.npy` is checked for **length only**; collapsing every obs code to
  0 passes. The within-image arm has exactly this assertion (`unique_train == 3`); the LOIO arm never
  got it.
- **`splits-4`** (medium) — **fold identity is unpinned.** All LOIO assertions are set/length, so
  reversing fold order passes. `labeling-deep-artifact`'s "fold *i* = `sorted(obs)[i]`, structurally
  cannot drift" is true of today's code but undefended by any test.
- **`splits-5`** (low) — `split_hash` is only ever compared to itself; removing `"folds"` from the
  hashed keys passes, and the mutated hash gives an identical digest for two different partitions.
- **`splits-6`** (low) — three fixture-blind or vacuous tests, incl. one whose own docstring admits its
  fixture has a single scale, and `_fold_summary`, which is entirely unasserted. No fixture reaching
  `build_image_inventory` has more than one `BoulderLabel`, though production is 5 rich / 2 poor /
  2 unknown and that column drives all stratification.
- **`features-6`** (low) — three assertions cannot fail: the context-patch test compares three outputs
  of the *same* call (patch centring can be removed entirely), `valid_pixel_fraction == 1.0` asserts a
  constant, and `test_stack_tiles` uses symmetric offsets.

### R91 — Every within-image fixture is the same symmetric 64×64 square, which disarms the suite's own strongest assertion
- **Status:** OPEN · **Severity:** high · **Liveness:** live-shipped (guard absent) · **Verified:** no (demonstrated by mutation, and quantified against real labels)
- **Detail:** [tests-deep-within-image.md](review_2026-07-31/tests-deep-within-image.md) `-1`

Three "surviving" mutants are **literal no-ops on the fixture** — 0 tiles move — while on real labels the
same mutations move **75–78 %** (`ti`/`tj` transpose), 27–32 % (pooled median) and 0–4 % (median→mean)
of tiles. The sharpest case: **M13 (the quadrant cut computed once and reused for every image) collapses
8 of 9 real images into a single quadrant**, and the file's *strongest* assertion — `len(unique_train)
== 3` at `:406`, which does kill M16 and M17 — **would have caught it on a real footprint**. So the
suite has the right assertion and a fixture that disarms it. That is a more precise diagnosis than
"missing coverage", and it makes **R78** the fix: this is the same fixture-shape defect, third instance.
- **Fix:** give the fixtures asymmetric, differently-shaped real footprints (one non-square, one
  ragged, and two that differ from each other). The assertions already exist.

### R92 — ~~The **v2** within-image split has drifted from its own labels~~ → **REFUTED as filed. It is v1 that drifted, and the cause is now known.**
- **Status:** **CORRECTED 2026-08-05.** The original claim (v2 drifted in 29 of 38 images, 3.53 % of tiles) is **wrong — the cohorts are inverted.** · **Severity:** low (v1 is a superseded generation) · **Liveness:** dead artifact (v1) · **Verified:** **yes — measured directly against the only durable artifact**
- **Detail:** [tests-deep-within-image.md](review_2026-07-31/tests-deep-within-image.md) `-2`; correction found while fixing **R91**

**Why both original measurements were unsound: quadrant definitions are never persisted.** They are
computed inside `build_split` (`src/dataset.py:295`) and written to **neither** the split JSON **nor**
the packaged metadata — verified by reading the keys of both. So no reviewer could have compared a
"stored cut"; whatever they compared, it was reconstructed, and the reconstruction is where the error
entered.

**The measurement that settles it** compares the *packaged fold membership* — the only durable trace —
against what today's code computes from today's labels, using the production
`_compute_quadrant_definitions` + `_quadrant_array_for_image`:

| cohort | images differing | tiles differing |
|---|---|---|
| **`dataset_v2`** | **0 / 38** | **0 / 3,564,767 (0.00 %)** |
| `dataset` (v1) | **5 / 8** | **13,969 / 610,586 (2.29 %)** |

So **the v2 split is exactly reproducible from today's code and labels** — the live cohort is clean —
and the drift is confined to the superseded v1 tree.

**The cause, which the original finding called "genuinely unexplained":** the quadrant cut snaps to
`max(SCALE_TO_FACTOR_FROM_FINEST.values())`, and commit **`29b0adb`** ("Model-improvement experiments
… CNN + S128 HELD as dev-only") added the entry `128: 16`, **doubling the production snap step from 8
to 16**. The v1 split predates that commit and sits on the old 8-tile lattice. The internal tell that
the review's probe used the stale factor: its quoted `ESP_017355_2260 STORED 688 → RECOMPUTED 696`
cannot both be right under a 16-snap, because **696 is not a multiple of 16**.
- **What survives, and is now the real finding — see R97:** a change flagged "dev-only" silently moved
  a production splitting constant.
- **What still stands from the original entry:** nothing pins the cut's value or its stability. The one
  test that opens a real split JSON reads only `kind`/`n_folds`/exclusions and is hardcoded to the v1
  cohort; a mutant that re-derives the cut at packaging time survives. R91's fix kills M13/M04 but not
  M05, precisely because the cut's *value* is unpinned.
- **Not addressed by this measurement:** R45's verifier compared the split artifact against **the
  sweep's** fold assignment, which is a *different* comparison from "against today's labels". That
  claim is untouched either way and R45 stands.

### R97 — A change marked "dev-only" silently doubled a production splitting constant
- **Status:** **CODE FIXED 2026-08-06 — WITHIN-IMAGE REBUILD PENDING.** `_compute_quadrant_definitions` now derives the snap step from the scales present in the image's own labels intersected with the factor map, so a table entry for an absent scale is inert; it also raises instead of returning `{}` when no present scale is known. Measured read-only before the change: the inflated step moves the cut for **29 of 38** v2 images, v2's persisted `quadrant_definitions` match step 16 **38/38**, and **v1 matches step 8 8/8** — so **v1 was correct all along and the splitter was the drifted party**, which inverts the old R92/R97 note in [PENDING_REBUILD.md](PENDING_REBUILD.md). Three tests added (production ladder, mixed set containing a real S=128, all-unknown-scale rejection); reverting the one line to `max(scale_to_factor.values())` on a scratch copy fails the first of them. Artifact impact: `dataset_v2/splits/within_image_4fold.json` + `packaged/within_image_*` are now stale; the LOIO split and the regional product are unaffected. · **Severity:** medium · **Liveness:** live (the constant is current) · **Verified:** yes — measured
- **Where:** `SCALE_TO_FACTOR_FROM_FINEST` in [src/dataset.py](../src/dataset.py); introduced by `29b0adb`

The within-image quadrant cut snaps to `max(SCALE_TO_FACTOR_FROM_FINEST.values())`. Commit `29b0adb`
added `128: 16` for an **S=128 modelling experiment that was explicitly HELD as dev-only and that no
shipped config emits** — and thereby changed the snap step for *every* within-image split from 8 to 16.
It is the reason the v1 split no longer reproduces (**R92**), and it cost two separate agents a wrong
answer, because both assumed the step was 8.
- **Fix:** derive the snap step from the *scales actually present in the labels*, not from the global
  table — or gate the `128` entry behind the dev config that needs it. Either way, add a test that pins
  the snap step against a fixture whose scale set is the production one.

### R93 — `pfree`, the shipped variant the HARD ABORT verdict was pronounced on, is never composited by any test
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed programme, but it is the *record* of the most consequential decision · **Verified:** no
- **Detail:** [tests-deep-region-staged.md](review_2026-07-31/tests-deep-region-staged.md) `-1`

The Stage-D fixture omits the `offset_logit_pfree` column, so all 18 tests silently drop that variant —
and `:365` asserts the variant set is exactly the pre-`pfree` three, so **fixing the fixture breaks the
suite**. The reviewer judged this honestly in the other direction too, which is the valuable part:
it **verified the wiring is correct** (`f_region_stagec.py:498` ↔ `VARIANTS["pfree"]`, composite exact
to 1e-5 in a direct probe), so **the abort verdict is not impugned.** Coverage gap, not a wrong number.

### R94 — The two tests that pin "the verdict ships the right variant" cannot fail
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed · **Verified:** no (measured)
- **Detail:** [tests-deep-region-staged.md](review_2026-07-31/tests-deep-region-staged.md) `-2`

Their `biases=[0.5, -0.5]` fixture makes `h1only`, `full` and `resid` **bit-identical** (measured max
difference **0.000e+00**), so a mutant taking the headline from `variants[0]` survives. Guard 1's
*existence* is pinned; its *identity* is not. **Seventh instance of Pattern A** — a check that could not
fail — and the first one found inside the test suite rather than in a gate.

### R95 — No assertion checks the georeferencing of any Stage-D output raster
- **Status:** OPEN · **Severity:** medium · **Liveness:** dead-closed · **Verified:** no (demonstrated by mutation)
- **Detail:** [tests-deep-region-staged.md](review_2026-07-31/tests-deep-region-staged.md) `-3`

A GDAL-order affine, an empty CRS and a row/col transpose all survive — although "ship on the mosaic's
exact grid" is Stage D's entire purpose, and **R01** (tile-phase misregistration) is a *shipped* raster
defect of exactly this class. Compounded by fixture degeneracy: every frame covers the whole tile
identically and the offsets CSV is already in sorted-key order, so a **positional** offsets join
(the ~100 km Blocker-1 class) is indistinguishable from a keyed one.

### R96 — Lower-severity items from the last two PASS-11 areas
- **`within-image-3`** (medium) — `buffer_tiles` is one-sidedly asserted and never exercised through
  packaging; a mutant hardcoding `buffer_tiles = 0` in packaging survives.
- **`within-image-4`** (low) — `_within_image_fold_summary`'s finest-scale stats are unasserted and
  feed a printed table in `notebooks/_build_09.py:161`.
- **`region-staged-4`** (low) — all three `(review 2026-07-29)` fixes in the Stage-D driver are
  unprotected; `458168f` added tests to `test_fgates.py` / `test_leveling.py` but never touched this
  file, which has **one commit ever**.
- **`region-staged-6`** (low) — assertions that read their expected value from the code under test,
  plus `after < 0.02*before or after < 1e-6`, whose second disjunct accepts **any negative value**.

### Verified clean by PASS 11 — do not re-file
- **`test_splits.py` genuinely pins**, each named by its killer mutant: metadata group-leak, image-level
  train/test sizes in the parquets, streaming-iterator membership (the one place membership *is*
  asserted), the label↔feature join key, the `stratification='none'` guard including its message, and
  that `seed` really reaches the RNG.
- **`test_features.py` genuinely pins** 10 items, most valuably that the **2026-06-10 DN-clip fix is a
  real, working regression guard** (`test_dn_threshold_survives_clip_spike`); also
  `test_subtile_variance_positive_on_split_tile`, `test_lacunarity_on_clumped_…`,
  `test_gradient_on_step_function` and `test_intensity_stats_ramp_…`.
- **`test_region_staged.py` reaches no producer** — verified before any run: no `cfg.output_dir`,
  `cfg.cache_dir` or `slow` marker; every path is under `tmp_path`; the one module global pointing at a
  live tree (`sd.FIG`) is monkeypatched by the `staged` fixture. Its two live-tree accesses are
  harmless reads.
- **The Stage-D abort wiring is correct.** `pfree` is untested but correctly wired
  (`f_region_stagec.py:498` ↔ `VARIANTS["pfree"]`; composite exact to 1e-5 in a direct probe), so
  **R93 does not impugn the abort verdict.**
- **`test_region_staged.py` is the one suite WITHOUT the (0,0)-origin fixture defect** — it uses
  E-12_N32's real origin and pitch, and `TILE_M` 160→80 kills all 18 tests. **Copy this fixture's
  shape when fixing R78/R91.**
- **`test_region_staged.py` genuinely pins** 10 things, each named by its killing mutant: lattice pitch,
  the mean-of-logits rule, offset sign, prescreen axes, guard-1 routing, worst-contributor provenance,
  missing-offset flagging, Tier-1/Tier-2 separation, calibrate-once, and the leveled partition
  composite.
- **`test_within_image_split.py` genuinely pins** the cross-scale coherence of the quadrant cuts, the
  buffer-band drop, and — its strongest assertion — that each fold's training set spans exactly 3
  quadrant codes (which killed the self-leak and the group-content mutants).
- **R11 and R19 are neither pinned nor re-filed here** — R11's guard is computed in Stage C (the
  nearest assertion, `test_region_stagec.py:110`, is a 4-way membership check) and R19 lives in
  `fgates`, unreachable from this file.
- **A consistent non-logistic link survives all three suites** that touch it — they pin that `sigmoid`
  and `logit` invert each other, not *which* link they are. Benign today; worth knowing.

---

## 5. Refuted / verified-clean — do not re-litigate

**Refuted by an independent verifier** (the claim was wrong, unreachable, or a documented deliberate
choice). Each was investigated with line-level evidence; if you rediscover one, read the reasoning
before re-filing.

| Claim | Why it fell |
|---|---|
| Within-image streaming = live leakage | dormant API, zero callers → downgraded to **R22** (low) |
| labels↔features join: two `config_hash` columns never compared | `config_hash` hashes the whole config, not a stage; on-disk data contradicts the scenario |
| `buffer_tiles: 0` invalidates within-image CV | closed dev-only work; the "not hypothetical" evidence was wrong |
| §5.1 scorecard "omits the shipped `pfree`" | pfree is **not** shipped (`DECISIONS.md:5562-5565` retracts it); the variant list predates the abort |
| Gate 6 pooling is what admitted `pfree` | mechanism is wrong — pooled ratio puts pfree at 1.52, outside the band. The real mechanism is rich-truth conditioning (`src/calibration.py:266-271`), and pooled-vs-per-obs is a pre-registered ruling (`DECISIONS.md:5049-5053`) |
| Gate 2's bare `<` violates the spec | `PLAN_FBuild.md:362` specifies "materially below **the unleveled value**", a graph-wide property; no paired test was ever specified, and `DECISIONS.md:5397-5399` re-declared the gate |
| Gate 3 scoring F on the mean composite is a defect | the η²-on-partition / ρ-on-shipped split is explicitly documented in the sibling harness |
| `_build_24`'s Spearman p-value | load-bearing nowhere, and the project already ruled leg-1 underpowered |

**Checked and found correct** (recorded so effort is not repeated): the calibration LOIO protocol and
its explicit in-sample-vs-LOIO labelling discipline (`scripts/bank_calibration.py`); `loio_calibrate`;
GeM clamping in both the numpy and torch paths; embedder train/deploy normalization parity (bicubic
interpolation is linear, so `/255 → (x−0.5)/0.5 → resize` and `resize → normalize` agree); strict
checkpoint loading; the MLP `FeatureScaler` is fit on train rows only; `DeployableHead`'s
early-stopping inner-val image is drawn from the training set, never the test fold
(`src/modeling/evaluate.py:631`); per-image local-radius CRS handling and the "refusing to guess"
guard in `src/detections.py`; the weighted normal equations (`sqrt(w)` consistent on both sides), λ
scale-correction (`λ = frac·median(W)`), the per-component gauge, and the sign convention in
`src/leveling.py`; `pack/unpack_key`, `intersect_sorted`, `candidate_pairs`; `edge_dlogit`'s
saturation-immunity; the mean-of-logits composite arithmetic and its nodata handling; gate PASS/FAIL
inequality directions (all correct); gate 1's common-footprint fix (all 234 windows have identical
`n_cells`/`n_frames` across all 5 rows — no differential filtering between the mosaic and F arms);
the cohort-join bbox route and its zero `(obs_id, TI, TJ)` duplicates; Stage B's globally-fixed uint8
stretch, its per-row physical incidence formula and the `cos(i) ≤ 0` clip at 89.5°;
`window_offsets` tiling is gap-free; `coarsened_transform` has no half-pixel shift *within* a tile.

**One latent nit not filed above:** `coarsened_transform`
([src/mapping.py:153](../src/mapping.py#L153)) scales `a` and `e` by `tile_px` but leaves the rotation
terms `b`/`d` unscaled. Harmless for the north-up CTX mosaic (`b = d = 0`), wrong for any rotated
transform.

---

## 6. What remains unchecked

**All 31 areas are reviewed** — the 6 from the first pass, the 15 `src/`-and-docs areas, the 8
`scripts/probes/` areas (184 files), and the 2 second-pass deep re-reviews. What is still uncovered:

- **Independent verification — the triaged scope is now DONE (2026-08-04); the tail is not.**
  Verified by someone other than their author: R01–R22 (dedicated adversarial verifier, killed 8 of 20),
  R23 / R47 / R46 (confirmed directly with quoted commands), and the **15 high-severity live-path
  findings of §7** (killed 0 of 15, downgraded 7). That is **~28 of 73**. The remaining **~45 are still
  single-agent, self-refuted** — but they are the ones triage judged *not* decision-changing, so the
  expected value of verifying them is much lower than it was for §7's scope. Their measurements are
  reproducible from committed artifacts, so it stays cheap if wanted.
  **Read §7's closing notes before doing more of it:** the 0-of-15 kill rate says the residual risk in
  this register is in *severity and blast radius*, not in whether the defects exist.
- ~~**Assertions that pin wrong science.**~~ **DONE — and the hypothesis is REFUTED.** All five large
  test bodies have now been read line-by-line *and* mutation-tested: `test_labeling.py` (668 →
  R77–R80), `test_splits.py` (399 → R87, R88, R90), `test_features.py` (533 → R89, R90, + it extended
  R77 and R78), `test_within_image_split.py` (445 → R91, R92, R96), `test_region_staged.py` (409 →
  R93–R96). **Across ~100 seeded defects, not one assertion was found defending a known defect** —
  what the register expected to find is not there. What *is* there: roughly half of all seeded defects
  survive, the fast/full gap is zero everywhere, and the dominant cause is **fixture degeneracy rather
  than missing assertions** (§4k). Two method notes worth carrying to any future test audit:
  **mutation testing produced all of this and reading alone would not have**; and the honest survival
  rate requires discarding equivalent mutants, which two areas did explicitly (57 % and 64 % after
  discarding, vs 63 % and 67 % raw).
- **Anything requiring execution.** No reviewer ran a notebook, sweep, training run, map build, ISIS
  step, GDAL/CTX read, or network fetch. Open as a result: the Slurm history that would settle whether
  the 906/907 hole came from **R18**'s resume race; whether fixing **R32** changes the FM-vs-Tier-1
  margin; whether re-running the `min_confidence` comparison with the target fixed (**R56**) unblocks
  **R23**'s fix; whether BoulderNet's inference footprint equals the HiRISE image footprint (an interior
  detector gap would still be labelled zero); the exact tile count affected by **R29** (analytic
  estimate only); and the current pass/fail state of the 21 slow tests (only their collection was
  verified).
- **Upstream of this repo.** The cause of **R23**'s score-ordered geometry loss is in BoulderNet's
  export, not here. (Its *inference footprint*, however, is no longer an open question — `REFUTED`,
  §4j.)
- ~~**One open contradiction.** R75's 3.89 % vs 0.21 %.~~ **RESOLVED 2026-08-04** — they counted
  different populations (overlapping vs fully-inside the vacated strip) and a third measurement
  reproduces both. See R75.
- **Small residue.** `setup_sherlock_env.sh`, `f_timing_test.sh`, `config_v2_dev.yaml`, and the
  `cache_v2_dev` symlink.

### A note on three "findings" that are containers, not findings

**R35**, **R42** and **R46** are roll-ups — each collects several lower-severity items from one pass
rather than describing a single defect, so they carry no single Status/Severity/Liveness triple. Any
count of "64 findings" therefore slightly understates the item count (the three containers hold ~30
sub-items between them) while overstating the count of distinct *top-level* defects. Treat 64 as the
count of register **entries**, not of defects.

Each area file's own *Coverage note* records what that reviewer read in full, only grepped, and could
not check — consult it before assuming a specific function was examined.

---

## 7. Verification status (added 2026-08-03)

Protocol + scope list: [_prompts_verify.md](review_2026-07-31/_prompts_verify.md). Verdicts:
`docs/review_2026-07-31/verify/<Rxx>.md`. **A finding is verified iff its verdict file exists.**

| finding | verdict | severity change | note |
|---|---|---|---|
| **R60** | CONFIRMED | high (unchanged) | numbers reproduce; the PDF itself is gitignored — cite `docs/classification_slimmer.md` |
| **R61** | CONFIRMED, strengthened | **high → medium** | the ">90 %" sentence lives only in the submitted variant; producing probe is off the live pipeline |
| **R54** | CONFIRMED-BUT-MIS-STATED | high (unchanged) | the pooled stat is **true-mass-weighted**, and liveness *broadens* — PLAN_RegionalMap's THEMIS leg consumes exactly this quantity |
| **R56** | CONFIRMED, every number reproduces | **high → medium** | liveness corrected to **dead-closed**: the defect object is the *record* of a closed stage. It still blocks R23's fix. |
| **R32** | CONFIRMED-BUT-MIS-STATED | **high → medium** | live-shipped as the default classification head, but nothing in the deployed map/head depends on it |
| **R24** | CONFIRMED | high (unchanged) | every number reproduces exactly. Liveness splits: the *aggregator* is live and unfixed, the *number* is dead-closed but still asserted in three live docs |
| **R31** | CONFIRMED | high (unchanged) | proven by synthetic-raster experiment; liveness refined to live-shipped **but dormant** (no active plan re-runs Stage 2) |
| **R36** | CONFIRMED-BUT-MIS-STATED | **high → medium** | **"could not have failed" is false** — the gate is monotone; ×2 offsets → −0.0274 = FAIL. It was handed a ~5× too-small treatment. Liveness **dead-closed** |
| **R03** | CONFIRMED-BUT-MIS-STATED | **high → medium** | mechanism is *larger* than stated (cohort floors disjoint; global `min_size_m` removes 0 of 3.1 M coarse polygons) but the **15.8 % headline fails** — CI [0.003, 0.472], does not replicate on the shipped head |
| **R48** | CONFIRMED | high (unchanged) | ρ = +0.9834 reproduces; 11 of 12 cells die at correct dof. Programme **dead-closed**, but the claim sits in reader-routed `docs/modeling.md` |
| **R51** | CONFIRMED | high (unchanged) | image-level sign-flip null: the published test's true rejection rate at α=0.05 is **0.324**. Conclusion flips at n=8. Project's scientific verdict survives on independent n=38 evidence |
| **R44** | CONFIRMED-BUT-MIS-STATED | **high → medium** | half-migration real, but "wrong by 2×–70×" is wrong — every v1 number reproduces exactly. Stale *scope*, not bad arithmetic → fix is **relabel, not recompute** |
| **R45** | CONFIRMED — **twice, independently** | high (unchanged) | two agents verified it without seeing each other's work and agreed on verdict, severity and liveness. **The size null is ~zero** (both passes) — the handicap is *not* sample size and *not* base rate (identical whole vs quadrant), but **truncated dynamic range**, so §4e's "compare R26" analogy is wrong. Matched pairing moves **4 of 8 cells to p<0.05** |
| **R37** | CONFIRMED (all 3 sub-claims) | **high → medium** | gate-1 drift re-derived exactly (0.1222→0.120535, 1.65→1.5276); **4 docs / 5 sites**, not 3. Downgraded because ROADMAP + memory both carry the abort and are read first |
| **R38** | CONFIRMED-BUT-MIS-STATED — **verified twice** (files merged) | **high → medium** | mechanism real and never deliberate, but blast radius dies: **0.04–0.41 %** of valid native pixels, no shipped raster affected (`reports/map_a1/` does not exist), and ⚠ **the η² confound did not happen** — `DECISIONS.md:4133`'s 218,089 = 467² exactly, the complete interior grid, so zero tiles were masked in either arm. Liveness → **live-active-plan** |

**Final tally: 15 of 15 verified — 0 refutations, 15 CONFIRMED (7 of them mis-stated), 7 severity
downgrades, all downward.** Not one finding died, which is a materially different rate from pass 1's
~40 % kill. The reason looks structural: these findings had already been self-refuted by their authors,
whereas the pass-1 candidates were unrefereed first drafts. **So the register's remaining 58 unverified
findings should be assumed roughly true in mechanism — and roughly one in two wrong about severity or
blast radius.**

**What verification actually bought, since it was not refutations.** Every downgrade came from judging
*blast radius* independently of mechanism, exactly as the protocol asks:

- **R38** — measurable after all. The review called the native-pixel count unmeasurable without imagery;
  `dataset_v2/features/*.parquet` holds per-tile intensity percentiles for 161 k committed tiles, giving
  0.055 % region-wide and 0 tiles dropped. Critically, `DECISIONS.md:4133`'s "218,089 of 219,961" is
  exactly 467²/469², so the 28 % η² reduction is on a common footprint — **the confound never happened**.
- **R36** — the could-not-fail claim is itself falsifiable and false. §3's Pattern A table lists R36 as its
  lead example; **that row now needs correcting** to "handed a ~5× attenuated treatment", not "inert".
  Pattern A stands on its other five instances.
- **R03**, **R44**, **R51**, **R45** — in each, one half of the finding survived and the other half did not:
  mechanism without the number (R03), scope without the arithmetic (R44), the statistic without the
  scientific verdict (R51), the defect at a *different* call site than the one cited (R45).

**Four corrections to the register's own proposed fixes**, each found by a verifier who tried to apply them:

1. **R24** — "emit `_n` for every key" **would not have prevented the error**: `spearman_n = 5` already sat
   one column from `n_real_folds = 20` in the same parquet row. The fix with teeth is the second clause.
2. **R31** — the alternative "re-read `actual_bounds` from the returned array shape" **does not work** for
   west/north overhang; only `Window.crop` + transform-from-clipped fixes it.
3. **R45** — "bootstrap the quadrant-size effect and report it as the null" would **report zero and
   entrench the error**, because the size null *is* zero.
4. **R03** — option (c) "enforce a per-image `min_size_m`" would delete **~67 %** of the fine images'
   labelled boulder area: a target redefinition plus full Stage-4 rebuild, not a tidy-up.

**Two things found that were not in any finding.** `docs/modeling_results.md`'s source sweep directory
`models/_sweep/20260524T071830Z` is **untracked in git**, so a reader-facing document's numbers are
reproducible only from an uncommitted local dir. And `normalised_lift_at_top_k` — which §3's Pattern-B
"Rule to adopt" cites as *already* solving prevalence — is R-precision floored at the base rate and
correlates ρ = +0.98 with it, so **that rule statement and R26's exemption are both wrong**.

**Verification is now complete for the high-severity live-path scope.** The remaining backlog is the 58
findings outside it (§6), which were never triaged as decision-changing.
