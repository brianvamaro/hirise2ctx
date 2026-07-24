# PLAN_FBuild — the 907-frame regional F build (per-frame inference + H1 centering + H4 leveling)

> **STATUS: APPROVED — EXECUTING (2026-07-23).** Brian's reopening call = **reopen-with-guards**,
> with an added standing requirement: the build must produce a **head-to-head comparison of the
> F-build vs the existing mosaic-path map and the A1 fallback, on both quality and run-cost** (§5.1).
> §0 P1–P5 are all cleared. First executable step = frame-list build + the V1/V5 sizing probe.
>
> _(Original)_ **DRAFT 2026-07-13 — staged ahead of the reopening call.** This promotes
> [PLAN_H4_Leveling.md](PLAN_H4_Leveling.md) §5 ("pre-planning only") into an executable plan so the
> build can start the day the call lands. **Execution is gated on §0.** Written while build-prep
> part B waits on free CPU; nothing here depends on its answer except the §3 centering statistic
> (two variants pre-declared).

## 0. Preconditions — the reopening call checklist

The PHASE-2 decision rule ([PLAN_StripingArtifact.md](PLAN_StripingArtifact.md) "PHASE 2"):
**η² ≲ 0.05 at skill ≥ −0.02 reopens the build; combined levers count (Brian 2026-07-09b).**
H1+H4 already clears the numeric bar on the pilot (partition η² 0.0505, held-out edge-CV |Δp|
0.074→0.035, leg-B skill PASS). Still outstanding before the call:

| # | item | status | where |
|---|---|---|---|
| P1 | Build-prep part A — 907-frame graph connectivity | ✅ **VERIFIED 2026-07-11**: ONE component at buffer 0, 3,584 edges, median degree 7 → one gauge | `f_h4_buildprep.py`, DECISIONS 2026-07-11 |
| P2 | Build-prep part B — H1 centering-statistic stability (per-crop vs per-frame ln-median: B1 sub-window drift, B2 cross-window range vs yardsticks 0.22 / 0.285) | ✅ **STABLE 2026-07-14**: within-frame drift worst 0.056 (22% of the 0.256 between-frame spread) → **per-frame median** centering for the build | `f_h4_buildprep.py` part B, DECISIONS 2026-07-14 |
| P3 | THEMIS-ρ leg on the leveled pilot map — last pre-declared §3.2 check (ρ **not degraded** vs the unleveled H1 pilot; leg-1 harness rerun) | ✅ **PASS 2026-07-14**: median-composite ρ 0.068→0.137 (Δ +0.069), not degraded (strengthened); unleveled 0.068 matches regional leg-1 +0.07 | `scripts/f_h4_themis.py`, DECISIONS 2026-07-14 |
| P4 | **ESP_053989 recheck under `minnaert_center`** — the minnaert-inversion fix was declared moot 2026-07-05c *because F was closed*; F reopening voids that. Verify its per-image AUC in `f_leg_b_loio_preds_minnaert_center.csv` recovered from the ~0.2 inversion (H1's centering plausibly fixed the stretch-floor clipping; unverified). If still inverted → diagnose before the build (candidate fixes in DECISIONS 2026-07-05b caveat 1) | ✅ **RECOVERED 2026-07-14**: per-image AUC 0.884 (mosaic 0.873); inversion gone → no separate fix needed | DECISIONS 2026-07-14 |
| P5 | Brian makes the reopening call on P1–P4 | ✅ **REOPEN 2026-07-23** (Brian) — reopen-with-guards; added requirement: head-to-head comparison vs the mosaic-path map + the A1 fallback on quality AND run-cost (§5.1) | — |

If a HARD-ABORT guard trips mid-build (§0.1) → fall back to 2026-07-05c (ship the A1 mosaic map +
caveat + H6 provenance); §5.1 is the instrument that makes that ship-vs-fallback call on evidence.

### 0.1 Adversarial review + pre-spend probes (2026-07-15)

Before committing 1–2 Sherlock days, the H1+H4 reopening case was adversarially stress-tested
(workflow `fbuild-reopening-adversarial-review`; DECISIONS 2026-07-15) → **YELLOW: reopen with
guards.** No blocker; the four "serious" lenses share one root cause — at n=7 the pilot cannot
separate "removed an artifact" from "absorbed a real regional gradient" (η²=0.0505 PASS holds only
under full offsets; residual-only 0.0595 fails). Two cheap pre-spend probes were run in response,
**both green**:

- **Leave-one-FRAME-out CV** (`scripts/f_h4_lofo.py`) — the honest generalization instrument (edge-CV
  is near in-sample on the over-determined graph): held-out |Δp| generalizes (median 0.0365 vs
  unleveled 0.0738), η² marginal (median 0.049, worst-frame 0.063), fragility isolated to the two
  least-pinned large offsets (J02, B03) — expected to improve on the median-degree-7 build graph.
- **Deploy-faithful per-frame skill** (`scripts/f_h4_legb_perframe.py`) — the build's mean-of-leveled-
  logits composite preserves skill (Δ_deploy −0.0007) and leg-B's obs-level approximation was faithful
  (approx_err +0.0000). Gate #5's skill concern is retired pre-build.

**Guards promoted to HARD ABORTS for the build (from review + probes):**
1. **Stage-C attribution must BIND** (§4.3): if the smooth offset field correlates with geology
   proxies (MOLA/THEMIS) rather than epoch/incidence metadata → **mandated fallback to residual-only**;
   do NOT let an "ambiguous" verdict silently default to full offsets.
2. **Early stopping rule:** at the first ~50–100 leveled frames, recompute leave-one-FRAME-out η² and
   **abort to the A1 fallback if > ~0.06** (Stage A/B are checkpointed, so this is free insurance).
3. **Physically vet the two large under-pinned offsets** J02 (+1.71, on a radiometrically-normal
   frame, LOFO pred-err 0.49) and P22_009549 (−1.39) before trusting the η² reduction — 85% of the
   pilot's η² drop rides on 3 frames and only F02 has a documented physical story.
4. **Watch the mean-flattening signature:** report corr(offset, frame-mean P(rich)) at 907 scale
   (pilot −0.94); if it stays strongly negative with |o|>1 on radiometrically-normal frames, that is
   wholesale contrast-erasure → trigger a magnitude cap / per-frame anomaly gate.
5. Keep §5 gates 1–5 pre-declared; make gate 3 (THEMIS-ρ) per-frame abundance-fidelity where feasible.

## 1. Product definition — what the build ships

**The final regional map = per-frame inference on ISIS-calibrated CTX frames, H1-centered,
H4-leveled, composited over the 26-tile circum-Chryse block**, replacing (not overwriting) the
mosaic-path map. Deliverables:

1. **P(rich) GeoTIFF** per tile + stitched block (same grid/CRS as the mosaic map so notebook 24 and
   the validation harness work unchanged), plus the calibrated-abundance layer via `CalibrationLayer`.
2. **H6 provenance layers** (ship regardless, Dickson-style): per-pixel **frame id** raster,
   **incidence** raster, **overlap-QA** raster (n_frames covering each tile + max co-located |Δp|
   after leveling), **offset-provenance** flag (solved vs interpolated — expected all-solved given P1).
3. **Offset table** `o_f` for all 907 frames + the trend-guard decomposition report (§4).
4. The **mosaic map stays on disk** as the comparison object (the before/after figure is a headline
   deliverable of `docs/striping_artifact.md`, queued in docs/index.md).
5. **H1-only (pre-leveling) composite GeoTIFF** — the un-leveled counterpart of deliverable 1, saved
   as a first-class artifact. It falls out for free (Stage C emits only the offset *table*; leveling
   is applied at composite time, so re-running Stage D with `o_f = 0` yields the pre-H4 map) and is
   *required* anyway — gate 2 scores against "the unleveled value," the before/after choropleth
   (gate 4) is H1-only vs H1+H4, and keeping it on disk makes the trend-guard call (§4.3) reversible
   without a Sherlock re-run. Also emit the **residual-only** composite (offsets minus the smooth
   plane) as the §4.3 conservative variant.
6. **Comparison scorecard (§5.1)** — a head-to-head table + figure of the F-build vs the mosaic-path
   map and the A1 fallback, on quality (artifact η², THEMIS-ρ, pooled skill, visual) **and** run-cost,
   all recomputed on a common footprint. Mandated by Brian 2026-07-23; it is the evidence the
   ship-vs-fallback call rests on.

**Non-goals:** no global build (26 tiles only); no re-freeze of the recipe (head stays
`models/deployable_f_center/86c51a5dca220f63` unless a §7 open question rules otherwise); the
far-western cohort cluster stays out of scope (PLAN_RegionalMap §10 #5).

## 2. Stage A — ISIS processing of 907 frames (Sherlock, CPU arrays)

All parameters previously verified (DECISIONS 2026-07-02/03; SHERLOCK_RUN Parts E–F):

- **Frame list:** from the 26 SeamMap gpkgs — reuse the dissolve in `f_h4_buildprep.py`; emit
  `region_frame_list.csv` (PRODUCT_ID, VOLUME_ID → EDR URL via `src/ctx_edr.py`, resolver verified
  12/12 + 10/10) and `frame_tile_map.csv` (frame → tiles it must be rendered into).
- **Pipeline per frame** (as Part F): EDR fetch → `mroctx2isis` → `spiceinit` (local kernels; web
  service resolved names cached from the timing run) → `ctxcal` → `ctxevenodd` → `cam2map` onto the
  CTX CRS at 5 m/px. **≈22 min/frame ⇒ ≈333 CPU-h serial; embarrassingly parallel ≈10.4 h on a
  32-task array.**
- **Stage-A robustness (sizing-probe findings 2026-07-24 — fold both into `f_leg_b_process.sh`
  before the 907 array):** (1) **Kernel completeness** — `spiceinit web=no` needs each frame's
  weekly reconstructed CK/SPK locally, and the July mirror is INCOMPLETE for 2018+ dates (2/5 probe
  frames failed on missing `mro_sc_psp_18xxxx.bc`). Fetch the full kernel set for all 907 dates
  (`f_fetch_kernels.sh` harvests names from the failed-run log, or a full `downloadIsisData mro`);
  treat a residual missing-kernel `spiceinit_fail` as a hole to patch + H6-flag, not a silent drop.
  (2) **Conditional `ctxevenodd`** — summed frames (`SpatialSumming > 1`) make ctxevenodd error
  (benign; they have no even/odd artifact), so skip it for those and project the calibrated cube
  (fixed in `f_timing_test.sh`; mirror into the build worker).
- **Scratch (~3.2 TB if everything kept):** keep only the `.map.cub` (or convert to windowed
  GeoTIFF) + delete EDR/intermediate cubes per-frame as each finishes (the timing kit already
  measures per-step sizes). Retention target ≤ 1.5 TB peak.
- **Resumability:** per-frame `status_*.csv` sentinel exactly as Part F; rerunning the array skips
  done frames. Failure budget: leg B saw 1/81 frames fail (K04_054963) — expect a handful at 907;
  the graph's median degree 7 means isolated failures cost coverage, not the gauge. Report failures;
  patch holes with the mosaic + flag in the H6 provenance layer rather than blocking.

## 3. Stage B — input mapping, embedding, head inference (per frame; GPU)

Train/deploy-matched H1 recipe (`minnaert_center`, DECISIONS 2026-07-07):

- **Mapping per frame:** I/F ÷ cos^k(i), **k = 0.580**, incidence per frame from the PDS volume
  indexes (`frame_incidence.csv` machinery + the `OVERRIDES` table — the P20_008839 decimal-shift
  class of SeamMap bug is why we do NOT trust SeamMap incidence); then ÷ per-frame ln-median
  (centering); then the FIXED centered-pool log stretch (ratio 0.84–1.12) → uint8.
- **Within-frame incidence ramp (build-only risk; audit 2026-07-23, `genuine-risk`).** The pilot
  crop is a ~1.3° window so incidence is ~constant, but full 907 frames span 3–4° latitude → a real
  ~2% top-to-bottom I/F ramp that a single per-frame `cos^k(i)` scalar does **not** remove — and that
  **neither H1 nor H4 can touch** (both are per-frame *DC* operators; η² sees only *between*-frame
  variance, so a within-frame ramp would render as a smooth abundance gradient inside each frame block
  and never register in the reopening metric). **Fix:** a **per-row `cos^k(i(lat))` divisor** from
  each frame's N/S incidence endpoints (incidence ~linear in latitude; the ISIS `phocube`/`caminfo`
  incidence band is available post-`cam2map`). Decide via §6 V5 — if the residual post-H1 ramp on a
  few full frames is <~0.5% the scalar is fine and this closes as expected-by-design.
- **Minnaert k (hygiene, audit 2026-07-23).** k = 0.580 is the frozen training-cohort fit; the
  pilot's own 7 frames fit 0.694. This is train/deploy-correct **and** first-order harmless — a
  global-k error is a per-frame constant that per-frame median centering removes *exactly*
  (`d/median(d)` cancels the `cos(i_f)^(k*−k)` factor). Still, re-fit k on the 907-frame cohort and
  report η² sensitivity over k ∈ [0.55, 0.70] (§6 V6) to close it on paper.
- **Centering statistic — pre-declared on part B's answer:** if B1/B2 drift ≪ the between-frame
  spread (0.22) → **per-frame** median (one number per frame, metadata-like, cleanest). Else →
  **per-latitude-band-within-frame** median (CTX frames are long in latitude; a banded median absorbs
  along-track drift while staying texture-blind). Do not invent a third variant mid-build.
- **Embedding + head:** frozen Fang recipe (GeM p=3, S=32) + `DeployableHead`
  `models/deployable_f_center/86c51a5dca220f63` → per-frame P(rich) in **logit** domain (H4 operates
  on logits; sigmoid only at composite time).
- **Sizing (to verify with a probe, §6 V1):** the 26-tile mosaic was ~57M tiles; per-frame inference
  re-embeds every overlap, and with median degree 7 the overlap multiplicity is plausibly ~2–3× →
  **~120–170M tile embeddings ≈ 25–40 L40S-h**, fanned by the existing `run_region_array.sbatch`
  pattern (per-frame outputs are race-free). Checkpoint per frame.
- **Storage:** per-frame logit rasters at S=32 tile resolution (160 m) are small (~kB–MB each);
  keep all 907 — they are the input to Stage C and the H6 QA layers.

## 4. Stage C — the H4 solve on the full graph + the pre-declared trend-guard method

**Solver:** identical to the pilot (`f_h4_level.py` generalized): per-edge sufficient statistics
(δ̄_ij, W_ij) over co-located S=32 tiles → 907-unknown weighted least squares + λ·Σo² Tikhonov,
gauge median(o)=0. 3,584 edge blocks is still a trivial sparse solve (seconds). **λ by
leave-one-edge-out CV** as pre-registered — at build scale, hold out a random 5% edge sample
(~180 edges) rather than all-edges-loop if runtime bites.

**Trend guard — method pre-declared HERE, before any offsets are seen (avoids the D-style
circularity argument):**

1. Fit linear + quadratic lon/lat surfaces to the solved o_f (weighted by frame degree); report
   R² of each.
2. **Significance by spatial permutation:** re-assign offsets among frames while preserving the
   spatial autocorrelation scale (block-permutation over ~4° cells), 1,000 draws → null R²
   distribution. The pilot's 58% was ≈ chance for 7 frames; at 907 frames chance R² for a 3-param
   plane is ~0.3%, so this test has real power. (The pilot `trend_guard`'s `frac > 0.5` flag + the
   "SIGNIFICANT" print is **not** a significance test — R² ~ Beta(1,2) at n=7 fires 25% under pure
   noise, and the observed 0.58 has p ≈ 0.17; it is a non-load-bearing reporting flag, superseded
   here by the permutation p-value. Audit 2026-07-23.)
3. **Attribution, not deletion:** if the smooth component is significant, do NOT silently add it
   back. Test it against *independent* axes: epoch/instrument metadata (acquisition year, Ls,
   incidence — artifact-side) vs geology proxies (MOLA elevation, THEMIS night-IR — geology-side).
   - Correlates with metadata, not geology → treat as artifact, **apply full offsets** (the pilot's
     F02 lesson: the plane mis-attributed a genuine radiometric offset).
   - Correlates with geology proxies → apply **residual offsets only** and report the smooth field
     as a candidate real gradient (this is the outcome that would *matter scientifically* — it feeds
     the Rodriguez-2016 story).
   - Ambiguous → apply full offsets (Brian's standing 2026-07-09b ruling) but ship the smooth field
     as an H6 diagnostic layer and say so in `docs/striping_artifact.md`.
4. Everything in 1–3 lands in `reports/figures/fbuild_trend_guard.{csv,png}` + a DECISIONS entry.

**No-overlap frames:** none expected (P1: 0 isolated). If Stage-A failures disconnect a frame,
interpolate its offset from graph neighbors (inverse-distance on frame centers) + flag in H6.

## 5. Stage D — compositing + final map + validation

- **Composite rule:** per tile, combine overlapping frames' *leveled* logits — default **mean of
  leveled logits** (post-H4 co-located disagreement is ~0.035, so the rule barely matters; mean is
  smooth across seams where last-write-wins would reintroduce edges). Record n_frames + max |Δp|
  per tile → H6 overlap-QA raster.
- **Acceptance gates on the final map (pre-declared):**
  1. **Partition η² over frame footprints ≤ ~0.05** on the full block (the original bar; now
     non-circular company: gate 2) — and it is **partition, not median-composite** that is scored
     against the bar (the median blends across seams and is scored against single-owner labels, so it
     deflates; audit 2026-07-23). Report it against the **rotation-null geological floor**
     (`eta2_rotation_null`, already in `src/striping.py`): at 907 frames the roll is no longer
     confounded by a few dominant blocks, so the bar reads as floor-plus-margin, not a bare constant.
  2. **Held-out edge-CV |Δp|** (the 5% held-out edge sample) materially below the unleveled value —
     the non-circular instrument that carried the pilot.
  3. **THEMIS night-IR ρ not degraded** vs the mosaic map on the same block (leg-1 harness).
  4. **Visual:** rectangular blocks gone in the notebook-24-style choropleth (the original
     success criterion).
  5. **Deploy-faithful LOIO spot-check:** the leg-B skill numbers used an obs-level approximation;
     on the build, per-frame inference over the cohort footprint is available for free → recompute
     pooled pr_auc@1e-2 / prec@5% (no presence AUC) on the true per-frame path for the 36 common
     images. Gate: within −0.02 of the H1 leg-B values. NB "skill safe by construction" is a
     *within-image* identity (per-image AUC Δ = 0); cross-image/pooled skill is this empirical gate.
  6. **Calibrated-abundance fidelity (audit 2026-07-23):** the additive logit offset *does* move the
     abundance **values** through the nonlinear `CalibrationLayer` (isotonic + quantile-match), which
     the raw-P gates never see. Where H4 composes with the CalibrationLayer, check per-bin RMSE /
     marginal-L1 on calibrated abundance (`compression_metrics` in `src/calibration.py`). Monotone
     calibrators preserve ranking, so only absolute values move — the cross-frame level correction we
     want; the gate just confirms it is not distorting the abundance scale.
- Then **hand off to PLAN_RegionalMap** — the parked validation legs 2–5 resume on THIS map
  (see the 2026-07-13 refresh note there).

### 5.1 Comparison scorecard vs mosaic + A1 (mandated by Brian 2026-07-23)

The build is only worth its cost if it beats the cheap A1 fallback; the mosaic-path map is the
un-mitigated reference. Ship an `f_map_compare` harness (reuses the η² scorer
`f_h2_eta2`/`src/striping`, the THEMIS leg-1 harness `f_h4_themis.py`, the notebook-24 choropleth,
and the pooled-skill instruments) producing:

- **Quality table — every metric recomputed on the SAME common footprint/grid.** *(Critical: the
  numbers on record today mix a pilot-crop scale [mosaic 0.196 / A1 0.141 / H1+H4 0.0505] with a
  regional detrended-residual scale [mosaic frame-block η² ~0.011]; the comparison must re-score all
  maps on one grid so the columns are apples-to-apples.)* Rows = {mosaic-path, A1, F-build H1-only,
  F-build H1+H4, F-build residual-only}. Columns:
  - **partition η²** over frame footprints (artifact),
  - **THEMIS night-IR ρ** on the common block (geology fidelity — A1's ρ is *not on record*, a known
    gap this closes),
  - **pooled pr_auc@1e-2 / prec@5%** on the common validation footprint (no presence AUC),
  - **held-out edge-CV |Δp|** where defined (F-build variants),
  - **visual** choropleth panel (blocks present → gone).
- **Run-cost ledger** per map (GPU-h, CPU-h, wall-clock, tiles/frames). Starting numbers on record:
  mosaic ~13–19 L40S-h / ~2–3 h wall (26 tiles); A1 ≈ a ~14-min post-hoc re-embed + re-bake (no
  re-inference); F-build ~333 CPU-h ISIS + ~25–40 GPU-h (§8). The ledger prices the quality gain.
- **Decision framing:** the F-build must materially beat A1 on artifact η² **and** not lose on
  THEMIS-ρ / pooled skill to justify its cost; if not, A1 remains the shipped product. Wire a
  *preliminary* three-way read into the §0.1 early-stop checkpoint (first 50–100 leveled frames) so a
  weak F-build is caught before the full spend. Lands in
  `reports/figures/fbuild_vs_mosaic_vs_a1.{csv,png}` + `docs/striping_artifact.md` + a DECISIONS entry.

## 6. Verify-at-runtime items (beyond §0)

| # | item | when |
|---|---|---|
| V1 | **Sizing probe:** embed 5 representative frames end-to-end (Stage B) on one GPU → measure tiles/frame + s/frame → size the array before submitting 907. **Kit built 2026-07-23**: `f_build_sizing_frames.py` (selects 5) → `run_f_build_probe.sbatch` (Stage A, KEEP_CUBES) → `f_build_sizing_probe.py` (Stage B); runbook SHERLOCK_RUN Part G | first Sherlock session |
| V2 | **Incidence table completeness:** PDS volume-index incidence resolved for all 907 (the P20_008839 typo class); fail loudly on gaps | frame-list build |
| V3 | **Parity gate:** one pilot frame (E8_N44) through the full build path must reproduce the pilot's per-tile logits (the map_region parity-check pattern) | before the array |
| V4 | **Stage-A failure census** vs the graph: recompute components with failed frames removed; if >1 component, per-component gauge + H6 flags | after Stage A |
| V5 | **Within-frame incidence-ramp check** (audit 2026-07-23 `genuine-risk`): on 3–5 full 3–4° frames, measure the residual top-vs-bottom I/F trend after the H1 mapping. <~0.5% → per-frame `cos^k(i)` scalar OK (close as expected-by-design); ≥~1% → switch Stage B to the per-row `cos^k(i(lat))` divisor **before** the array | Stage B sizing probe (with V1); kit built 2026-07-23 — SHERLOCK_RUN Part G |
| V6 | **Minnaert-k re-fit + sensitivity** (audit 2026-07-23): re-fit k on the 907-frame cohort (log-median vs log-cos i); report final-map partition η² sensitivity over k ∈ [0.55, 0.70]. Expected flat (centering cancels the per-frame constant); fail loudly if not | frame-list build |

## 7. Open questions (Brian — surface via AskUserQuestion at execution, not pre-decided)

1. **Where does Stage B run?** Sherlock L40S arrays (fits the existing pattern, data already on
   scratch) vs bringing cubes home (3+ TB — almost certainly not). Default: Sherlock.
2. **Scratch retention** after the build: keep 907 `.map.cub` (~1.5–3 TB, enables reruns/H5) or
   keep only per-frame logit rasters + final maps (GB-scale)? Default: keep logits, drop cubes
   after V3/V4 pass.
3. **Trend-guard ambiguous branch** (§4.3): confirm "full offsets + diagnostic layer" is still the
   ruling when the dense-graph evidence is in hand (the standing ruling was made on pilot evidence).
4. **Does the final map re-run the far-south specificity tiles first** (cheap early read on
   highlands behavior under the F path) or straight through all 26? Default: straight through
   (resumable anyway).

## 8. Cost summary

| stage | compute | wall-clock (parallel) |
|---|---|---|
| A — ISIS 907 frames | ≈333 CPU-h | ≈10.4 h @ 32 tasks |
| B — embed + infer | ~25–40 GPU-h (V1 refines) | ~4–7 h @ 6 GPUs |
| C — H4 solve + trend guard | minutes, laptop | — |
| D — composite + gates + figures | hours, laptop CPU | — |

Total: **one to two Sherlock days plus a laptop day** — cheap relative to the two months the
artifact has cost; the expensive part was proving it would work.
