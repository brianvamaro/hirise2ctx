# PLAN_H4_Leveling — overlap-constrained per-frame prediction leveling (PHASE 2 H4)

> **STATUS: ACTIVE (2026-07-09b — H3 verdict folded into §6; cleared to execute on the H1 head).**
> This is the H4 row of [PLAN_StripingArtifact.md](PLAN_StripingArtifact.md) §PHASE 2 expanded into
> a self-contained executable plan. H3 landed **FAIL to reopen** (`a0cdcf1`), and Brian's ruling that
> **combined levers count toward the reopening bar** resolves §6's open question. H1 stays the
> operating baseline; H4 is the last F-mode lever, run on the H1 head. See §6 for the folded verdict.

## 1. Why H4 is next regardless of H3's outcome

H3 (consistency-regularized head) targets the *diffuse* embedder-amplification component. H4
targets the *level-offset* component — and we have direct evidence the level component survives
everything upstream: **F02** is −2.23σ darker than incidence predicts, resists Minnaert + H1
centering, and over-predicts P(rich) 0.222 vs ≤0.07 on the same ground it overlaps (DECISIONS
2026-07-07, `_f02_diagnose.py`). No input mapping or training-time regularizer *guarantees* a
frame-level fix; post-hoc leveling of co-located prediction disagreement removes it **by
construction** (same ground, two frames → the disagreement is artifact; no geology assumption,
so D's circularity does not apply).

- If H3 **passes** (η² ≤ 0.05): H4 is the approved polish for the 907-frame build (decision rule).
- If H3 lands **gray** (η² 0.05–0.08 at OK skill): H1(+H3)+H4 may clear the bar together — see §6.
- If H3 **fails**: H4 is the last F-mode lever before falling back to 2026-07-05c (ship A1 + caveat).

**Scope limit (unchanged):** H4 needs per-frame predictions *with overlaps* — the F deployment
only. On the mosaic map (a partition, no overlaps) leveling degenerates to the ruled-out option D.

## 2. Method

Per-frame additive offset **in logit domain**. Let ℓ_f(t) = logit(p) for tile t predicted from
frame f. Solve for offsets o_f minimizing

    Σ_{edges (i,j)} Σ_{co-located t} w_t · [ (ℓ_i(t) + o_i) − (ℓ_j(t) + o_j) ]²  +  λ · Σ_f o_f²

- **Gauge:** the objective is invariant to a global constant → fix median(o_f) = 0.
- **Regularization λ:** small Tikhonov keeps disconnected/weakly-connected frames from blowing up
  and biases toward "no correction" where evidence is thin. Sweep λ over decades; pick by
  leave-one-edge-out CV (§3.2), not by η².
- **Weights w_t:** downweight tiles near valid-data edges / low-coverage; start uniform.
- **Trend guard:** leveling must not launder a real regional gradient into offsets. Guard = fit a
  low-order (linear or quadratic in lon/lat) field to the o_f vs frame-center positions after
  solving; if the smooth component is significant, report it and add it back (only the *residual*
  offsets are applied). This is the standard mosaicking/gravity-survey leveling decomposition.
- **No-overlap frames:** interpolate offsets from graph neighbors (inverse-distance on frame
  centers); flag interpolated frames in the H6 provenance layer.

## 3. Measurement design — two corrections to the PHASE-2 row as written

### 3.1 The per-image AUC skill gate is (mostly) blind to H4 — by construction

A per-frame **additive logit offset does not change within-frame ranking**. For any obs whose
leg-B composite comes from a single frame, per-image AUC is *provably unchanged* by H4; only
multi-frame composite obs (the 28 multi-crop training obs; 47 pairs) can move at all. So "skill
Δ median per-image AUC ≥ −0.02" would pass H4 near-vacuously and must not be cited as evidence
of harmlessness. **Pre-declared H4 skill instruments instead:**
- pooled `pr_auc@1e-2`, pooled Spearman ρ, `precision@5%` on the leg-B common-36 LOIO predictions
  (these DO see cross-frame level changes) — no presence AUC, per project rule;
- per-image AUC reported only as a sanity row (expected ≈ unchanged).

> **DONE 2026-07-09b — PASS** (`scripts/f_h4_legb.py`, DECISIONS 2026-07-09b(H4-legB), notebook 28
> §9). Offsets solved on the 28-training-obs graph (58 frames / 47 edges / **21 components** — the
> graph is fragmented at this scale, so leveling is mostly within-obs), applied at obs level to the
> composite LOIO preds: **Δ pooled PR-AUC (H1+H4 − H1) = −0.0104** (within the −0.02 gate), H1+H4
> stays **+0.019 above the mosaic baseline**, and **Δ per-image AUC = exactly 0.0000** — confirming
> the structural claim on real data. prec@5% 0.972→0.968. (Spearman/continuous-target metrics skipped
> — the LOIO preds carry only the binarized fa>1e-2 target; pooled PR-AUC + prec@5% both key off it.)
> Caveat: obs-level application is exact for single-frame obs, mean-of-frames for composites; the
> deploy-faithful per-frame-inference LOIO is a build-scale rebuild, deferred.

### 3.2 η² is trivially optimizable by leveling — held-out validation is mandatory

H4 *directly minimizes* co-located disagreement, and partition η² is its close cousin — quoting
post-H4 η² alone would be circular in exactly the way that killed option D. **Pre-declared
held-out checks:**
1. **Leave-one-edge-out CV on the overlap graph:** solve offsets with edge (i,j) removed; report
   held-out co-located |Δp| on that edge, aggregated over all 15 pilot edges. Success = held-out
   |Δp| drops materially below the unleveled 0.073 (H1 baseline); failure mode (offsets that only
   memorize their own edges) shows up here immediately.
2. **THEMIS night-IR leg ρ not degraded** (rerun the leg-1 harness on the leveled pilot map).
3. **Visual:** choropleth blocks gone (the original success criterion).
4. **Trend-guard report** (§2): the smooth component of the fitted offsets, so a real regional
   gradient can't silently vanish.

## 4. Pilot recipe (concrete; ~minutes of CPU once per-frame predictions exist)

Everything needed is already on disk:
- **Frames:** 7 aligned E8_N44 crops `reports/f_timing/pilot_work/aligned/*.npy` (15 overlap edges).
- **Per-frame predictions:** `scripts/f_pilot_crop.py --mappings minnaert_log --minnaert-k 0.580
  --stretch-lohi 0.0965 0.2374 --head-dir models/deployable_f_center/86c51a5dca220f63` (H1 head;
  swap in the H3 head dir if H3 is adopted). Per-frame embedding cache inside `scripts/f_h2_eta2.py`
  is reused — predictions are a head-forward pass.
- **Co-located tile indexing:** the pair cache in `f_h2_eta2.py` (ti/tj per edge) and/or
  `reports/f_leg_b/h3_consistency_pairs.npz` from the H3 machinery.
- **Solver:** closed-form weighted least squares (7 unknowns, 15 edge blocks) — `scripts/f_h4_level.py`
  (to write): builds the system, sweeps λ, runs leave-one-edge-out, emits
  `reports/figures/f_h4_leveling_summary.csv` + before/after choropleth + offset-vs-incidence and
  offset-vs-epoch scatter (F02 should be the outlier it is).
- **Leg-B side (skill instruments, §3.1):** apply the *training-obs* frame offsets to the LOIO
  per-tile predictions (`f_leg_b_loio_preds_minnaert_center*.csv`) via `obs_frame_map.csv`, recompute
  pooled metrics. Note: leg-B frames ≠ pilot frames — offsets there come from the 47 training-obs
  overlap pairs (same solver, bigger graph).

Order: pilot choropleth + edge-CV first (cheap, decisive); leg-B pooled metrics second; THEMIS ρ
last (only if 1–2 pass).

## 5. Scale-up sketch — the 907-frame build contingency (pre-planning only)

> **2026-07-13: promoted to an executable plan — [PLAN_FBuild.md](PLAN_FBuild.md)** (stages A–D,
> pre-declared trend-guard method, acceptance gates, verify items, open questions). This section
> stays as the original sketch; the build-prep verify items below are mirrored in PLAN_FBuild §0.

If the reopening bar is met, the regional F build parameters (all previously verified, DECISIONS
2026-07-02/03): 907 frames / 26 tiles (1,371 footprint polygons), ISIS ≈ 22 min/frame ⇒ ≈333 CPU-h
serial, embarrassingly parallel ≈10.4 h on Sherlock arrays; ~3.2 TB scratch if all kept; EDR
resolver `src/ctx_edr.py` (12/12 + 10/10 verified); env kits `setup_isis_env.sh` + `run_f_timing.sbatch`.
H4 additions to build prep:
- **Overlap graph at scale — ✅ VERIFIED 2026-07-11** (`scripts/f_h4_buildprep.py`,
  `reports/f_leg_b/h4_buildprep_graph.log`, DECISIONS 2026-07-11): **907 unique frames from 1,371
  per-tile footprints** (plan's counts reproduced from the SeamMaps), and the adjacency graph is a
  **single connected component at buffer 0** — 3,584 edges, 907/907 frames, 0 isolated, median
  degree 7. **One gauge for the whole region; no per-component flagging needed.** (Ops note: raw
  dissolved SeamMap polygons are pixel-resolution — the script now `simplify(50 m)`s before
  buffering; the first run stalled 5.7 CPU-h on the moot buffer-sensitivity sweeps.)
- **H1 deploy-time statistic check (⚠ PENDING — waiting on free CPU 2026-07-11):** H1 centers by
  per-*CROP* median; at deploy the statistic becomes a per-*FRAME* median over a much larger, more
  heterogeneous support. Local probe built as part B of `f_h4_buildprep.py` (runs first now):
  B1 = 3×3 sub-window ln-median drift across each 75 km pilot aligned array; B2 = ln-median range
  across independent crop windows of the same frame (multi-obs frames + pilot overlap). Yardsticks:
  between-frame spread ~0.22, H1 stretch width 0.285. If drift ≪ between-frame spread → per-frame
  centering is safe; else center per-window or per-frame-with-latitude-band.
- **H6 rides along regardless:** per-frame id + incidence + overlap-QA + interpolated-offset flag
  rasters ship with any final map.

## 6. Decision after H3 (verdict folded in 2026-07-09b)

**H3 outcome = FAIL to reopen** (DECISIONS 2026-07-09b; committed `a0cdcf1`). The consistency
penalty *does* move η² monotonically with λ — λ=100 crosses the bar (partition η² 0.035,
pred-overlap 0.031 < input I/F 0.102, amplification killed) — but **only by collapsing the head's
global dynamic range**, so skill degrades on the *same monotone axis* (pooled PR 0.796→0.621; skill
gate crossed λ3→λ10, η² bar only at λ100). **No Pareto point clears both gates.** H2+H3 together ⇒
per-frame prediction-block variance is not separable from geology by any *data-driven invariance*
instrument. **H1 (log-median centering, η² 0.081) stays the operating baseline; the H1 head is what
H4 levels.**

This lands H3 squarely in the FAIL row of the (now-collapsed) tree:

| situation | action |
|---|---|
| **H3 FAILED (actual)** | H1 stays baseline; run the §4 pilot on the **H1 head** (`models/deployable_f_center/86c51a5dca220f63`) as the last F-mode lever. If H4 also misses η²≲0.05 at skill ≥ −0.02 (held-out, §3.2) → 2026-07-05c fallback (ship A1 mosaic map + caveat + H6 provenance layer). |

**Open decision — RESOLVED by Brian (2026-07-09b).** The question was whether a *combined*
H1(+…)+H4 result that clears η² ≤ 0.05 with the §3.2 held-out checks passing counts toward reopening
the 907-frame build, given the PHASE-2 rule keyed reopening on H1–H3 alone. **Brian ruled: combined
levers count toward the reopening bar** (recorded DECISIONS 2026-07-09b + PLAN_StripingArtifact
decision rule). So H4 is no longer mere "polish": if **H1+H4** clears η² ≲ 0.05 at skill ≥ −0.02
*and* passes the non-circular held-out validation (leave-one-edge-out edge-CV drop below the 0.073
baseline + THEMIS ρ not degraded + trend-guard clean + blocks visually gone), the 907-frame build
reopens. No further AskUserQuestion is needed to start — the pilot is cleared to run on the H1 head.
