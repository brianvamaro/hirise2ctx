# PLAN_FBuild — the 907-frame regional F build (per-frame inference + H1 centering + H4 leveling)

> **STATUS: DRAFT 2026-07-13 — staged ahead of the reopening call.** This promotes
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
| P2 | Build-prep part B — H1 centering-statistic stability (per-crop vs per-frame ln-median: B1 sub-window drift, B2 cross-window range vs yardsticks 0.22 / 0.285) | ⚠ **PENDING** — minutes of local CPU, blocked on the BoulderNet run | `f_h4_buildprep.py` part B |
| P3 | THEMIS-ρ leg on the leveled pilot map — last pre-declared §3.2 check (ρ **not degraded** vs the unleveled H1 pilot; leg-1 harness rerun) | ⚠ **PENDING** | PLAN_H4_Leveling §3.2 #2 |
| P4 | **ESP_053989 recheck under `minnaert_center`** — the minnaert-inversion fix was declared moot 2026-07-05c *because F was closed*; F reopening voids that. Verify its per-image AUC in `f_leg_b_loio_preds_minnaert_center.csv` recovered from the ~0.2 inversion (H1's centering plausibly fixed the stretch-floor clipping; unverified). If still inverted → diagnose before the build (candidate fixes in DECISIONS 2026-07-05b caveat 1) | ⚠ **PENDING** — two-minute pandas check | added 2026-07-13 |
| P5 | Brian makes the reopening call on P1–P4 | ⚠ **PENDING** | AskUserQuestion |

If the call is NO → fall back to 2026-07-05c (ship the A1 mosaic map + caveat + H6 provenance).

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
   plane is ~0.3%, so this test has real power.
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
     non-circular company: gate 2).
  2. **Held-out edge-CV |Δp|** (the 5% held-out edge sample) materially below the unleveled value —
     the non-circular instrument that carried the pilot.
  3. **THEMIS night-IR ρ not degraded** vs the mosaic map on the same block (leg-1 harness).
  4. **Visual:** rectangular blocks gone in the notebook-24-style choropleth (the original
     success criterion).
  5. **Deploy-faithful LOIO spot-check:** the leg-B skill numbers used an obs-level approximation;
     on the build, per-frame inference over the cohort footprint is available for free → recompute
     pooled pr_auc@1e-2 / prec@5% (no presence AUC) on the true per-frame path for the 36 common
     images. Gate: within −0.02 of the H1 leg-B values.
- Then **hand off to PLAN_RegionalMap** — the parked validation legs 2–5 resume on THIS map
  (see the 2026-07-13 refresh note there).

## 6. Verify-at-runtime items (beyond §0)

| # | item | when |
|---|---|---|
| V1 | **Sizing probe:** embed 5 representative frames end-to-end (Stage B) on one GPU → measure tiles/frame + s/frame → size the array before submitting 907 | first Sherlock session |
| V2 | **Incidence table completeness:** PDS volume-index incidence resolved for all 907 (the P20_008839 typo class); fail loudly on gaps | frame-list build |
| V3 | **Parity gate:** one pilot frame (E8_N44) through the full build path must reproduce the pilot's per-tile logits (the map_region parity-check pattern) | before the array |
| V4 | **Stage-A failure census** vs the graph: recompute components with failed frames removed; if >1 component, per-component gauge + H6 flags | after Stage A |

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
