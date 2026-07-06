# PLAN_StripingArtifact — investigate the rectangular/striping artifact in the regional map

**Created 2026-06-18 (Brian-flagged; next-session first step).** The 26-tile circum-Chryse
abundance map (notebook 24 §2) shows **rectangular / vertical-striping** patterns. This plan is the
focused investigation to characterize the cause, decide a mitigation, and re-test. Context +
ruled-out causes live in [[regional_map_rectangular_artifact]] (memory) and DECISIONS 2026-06-18.

## ✅ SOLVED (2026-06-18d) — cause = CTX SOURCE-FRAME radiometry
Brian corrected the target: the artifact is **high-amplitude rectangular BLOCKS** (visible raw,
tilted ~275°, not vertical) that **align with CTX source frames**. Each Murray tile is a patchwork of
~46–63 distinct CTX source images (SeamMap = a partition); the per-patch model + **fixed `/255`
embedder with no per-frame normalization** maps each frame's radiometry to a different abundance level
→ filled rectangular blocks (filled, not seam lines, because per-frame radiometry is footprint-uniform).
**Evidence:** frames explain eta² 0.011 vs 0.002 null (89% tiles > null-95p); frame-mean choropleth
reproduces the blocks after geology removed; effect is texture/contrast-driven (per-frame mean-DN
Spearman only +0.14). **Why invisible before:** training windows ≈8 km inside one ~28 km frame + LOIO
scores per-image=per-frame. **Mitigation (NOT done):** per-frame radiometric normalization before
embedding; adjudicate by LOIO skill preserved + thermal ρ up. Full record: DECISIONS 2026-06-18d,
notebook 25 (rewritten), notebook 24 §2d, `src/striping.py`, `scripts/striping_frame_blocks.py`,
[[regional_map_rectangular_artifact]]. **The §1–§6 below (vertical-stripe / seam-line plan) is
SUPERSEDED — wrong feature; kept for method history only.**

## MITIGATION — options, the A-vs-C distinction, and the A1 prototype (2026-06-19)

**Pipeline framing.** Every prediction is:
`CTX pixels → [normalize?] → frozen Fang ViT embedder → 768-d embedding → MLP head → abundance`.
The **ViT is frozen** (pretrained, not ours); the **head (`mlp_ens3`) is the only trainable part**.
Embeddings are a deterministic function of (input pixels, frozen ViT) and are *cached after computing*
— so changing the input means re-running the ViT to get new embeddings (a cache miss), then re-fitting
the head. At deploy (`map_region`) the embedder runs **live** per read-window (no cache).

**The three candidate fixes:**

- **A — per-frame normalization (SeamMap-aware), input-side.** Remap each CTX source frame's DN
  distribution to a common reference *before* the ViT. A monotonic per-pixel remap → removes
  between-frame level/contrast, preserves within-frame texture, **no seam ringing** (no spatial
  kernel). Variants: **A1** robust offset+gain (`(x−median)/IQR·s0+m0`), **A2** full histogram match,
  **A3** cross-seam least-squares offset solve (geology-agnostic).
- **B — frame-agnostic local contrast normalization, input-side.** Local high-pass / divisive
  normalization; no SeamMap needed, but **smears the DC step at seams** (it can't locate them) and
  amplifies noise in bland terrain.
- **C — radiometric augmentation, model-side.** Leave the input raw; during training generate **many
  augmented copies** of each window (random offset/gain/contrast/gamma), run the ViT over **all** of
  them, and train the head on all those embeddings with the *same* label, so the head learns to ignore
  radiometric directions.

**A vs C — the crux (both retrain the head, both run the frozen ViT):**
the difference is *what you feed the ViT and how many times*.
- **A1 embeds each window ONCE, cleaned** → variation removed *before* the ViT → invariance lives in
  the **input**. Embedding cost = N windows × 1 pass.
- **C embeds each window MANY times, deliberately perturbed** → invariance lives in the **head**.
  Embedding cost = N × K passes (K× more).
- **Ceiling:** A acts *before* the frozen ViT, so it can **fully remove** the radiometric signal from
  the embeddings; C acts *after* the frozen ViT (only the head changes), so it can only **down-weight**
  whatever the head can separate — **partial**, because we cannot fine-tune the ViT. → **A has the
  higher ceiling; that is why we try A1 first.**
- **Deploy:** A needs the SeamMap per-frame at inference; C needs nothing special (invariance baked
  into head weights).

| | A (per-frame) | B (local) | C (augment) |
|---|---|---|---|
| ViT fed | each window once, cleaned | once, locally normed | each window ×K, perturbed |
| invariance in | input | input | head |
| removes artifact | fully | smears seams | partial (frozen-ViT ceiling) |
| deploy needs SeamMap | yes | no | no |
| embedding cost | N×1 | N×1 | N×K |

**Validation protocol (identical for any input-side fix).** Because the head learned the *un-normalized*
embedding statistics, an input change makes the old head mismatched. So: apply the fix to the **training**
windows too → re-embed (frozen ViT) → **re-bake the head** → adjudicate by **(1) LOIO per-image AUC held**
(median ≈ 0.79 for `mlp_ens3`; the *skill gate*), **(2) eta² down** on a real tile (artifact gone), **(3)
THEMIS/TES thermal ρ ideally up** (external check). "Looks cleaner" alone is insufficient.

**A1 prototype status (2026-06-19):**
- Diagnostic (`scripts/striping_frame_radiometry.py`): between-frame **level spread ≈20 DN, scale
  CV ≈0.43**; robust offset+gain collapses the per-frame histograms → A1 is the right first cut. Set
  reference **m0=125 DN, s0=27.7** (global median-of-frame-medians / median-of-frame-IQRs).
- A1 implemented in `src/striping.py` (`a1_apply` / `a1_normalize_window` / `a1_normalize_per_frame`) +
  7 unit tests (pass). Wired into the embedder (`_w2_fang_embed.py --norm a1 --out-suffix _a1`).
- Re-embedded the 38-image cohort with A1 → `dataset_v2/fang_embeddings_a1/` (done). Loaders got a
  `store_name` param (full suite 351 passed — no regression).
- **Skill gate DONE** (`scripts/striping_a1_loio.py`): baseline median per-image AUC **0.790** / pooled
  PR 0.777; A1 **0.766** / 0.771. **Δ median AUC = −0.024** (marginal FAIL of −0.02), Δ pooled PR
  −0.007. ⇒ the model used absolute radiometry as a within-image cue; A1 removes it at a small cost.
- **Payoff DONE** (`scripts/striping_a1_infer_crop.py`, E8_N44 8-frame crop, raw P(rich)):
  **eta² 0.196 → 0.141 = 28% artifact reduction** (`striping_a1_payoff.png`). A1 **partially** flattens
  the per-frame blocks — they are reduced but still visible (residual = shape/contrast/noise-character
  differences offset+gain can't capture + the frozen-ViT ceiling).
- **NET: A1 = partial 28% reduction for −0.024 LOIO — real but not decisive. NO DECISION TAKEN.** Full
  option space + pros/cons below.
- Note: m0/s0 measured on the *region* frames (deploy side); same constant used train+deploy so it
  doesn't bias the A/B — minor refinement = recompute over train∪region if A1 is adopted.

## ✅ F DE-RISK STEP 1 DONE (2026-07-02) — EDR resolver solved; timing kit ready

**Brian chose "de-risk F first"** (the recommended path from the 2026-06-22 framing below).
- **Resolver SOLVED without `planetarypy`:** the stale `PDS_IMG` 404s were just a renamed path
  segment — the live URL is fully determined by the SeamMap's own fields:
  `planetarydata.jpl.nasa.gov/img/data/mro/ctx/{VOLUME_ID.lower()}/data/{PRODUCT_ID}.IMG`.
  Verified 12/12 mission-spanning + 10/10 on the timing list; ODE REST = documented fallback.
  Code: `src/ctx_edr.py` + tests; check: `scripts/probes/_f_edr_url_verify.py`. DECISIONS 2026-07-02.
- **Step 2 kit built, waiting on Brian:** `sbatch run_f_timing.sbatch` on Sherlock (after one-time
  `setup_isis_env.sh`) times EDR→`mroctx2isis`→`spiceinit web=yes`→`ctxcal`→`ctxevenodd`→`cam2map`
  on 10 frames (`reports/f_timing/frame_list.csv` = the 7 E8_N44 A1-crop frames + 3 era extremes)
  and prints the ×907 regional / ×86,571 global extrapolation. SHERLOCK_RUN.md **Part E**.
- **Then:** timing.csv prices F's pipeline cost → make the F-vs-E call (embedding/head-retrain
  cost is already understood from the A1 cycle; the ISIS leg was the unknown).
- **✅ TIMING DONE (2026-07-03, after a Sherlock env gauntlet — DECISIONS 2026-07-02b/c/d): 10/10
  frames end-to-end, zero failures.** Mean **22 min/frame, 96.6% = cam2map**; regional 907 frames
  ≈ **333 CPU-h** (≈10 h on a 32-task array, CPU-only); global ≈ 31,800 CPU-h. Regional storage
  ~3.2 TB of projected cubes if kept (stream/16-bit/crop levers exist). **F's ISIS leg is proven
  and affordable at regional scale → decision is now purely Brian's F-vs-E call** (DECISIONS
  2026-07-03).

### DECISION (Brian, 2026-07-03): **small F pilot first**, then commit
One more gate before the full 907-frame build — prove on the **7 already-timed E8_N44 crop
frames** that per-frame inference actually kills the blocks:
- **Leg A (deploy-side, no retrain):** extract the crop windows from the projected I/F cubes
  (`scripts/f_pilot_extract_crop.py`, Sherlock) → laptop: embed + predict with the existing
  mosaic-trained heads under 2–3 **I/F→ViT-input mappings** — (a) global affine (pooled 2–98% →
  0–255; the "calibrated frames need no per-frame norm" bet), (b) **Lambert cos(incidence)**
  correction from SeamMap metadata then affine (calibrated I/F *exposes* real illumination
  differences the mosaic's per-frame stretch used to hide — the A-meta insight resurfacing
  inside F), (c) per-frame robust A1-style (reference). **Metrics:** frame eta² vs the mosaic
  baseline **0.196** / A1 **0.141** (target ≲ ~0.03 ≈ block-free), frame-mean choropleth, and
  **overlap-pair pixel agreement** (the Walter ±2% claim checked on our own frames — F's
  built-in internal validation). Caveat: mosaic-trained head on F inputs is train/deploy
  mismatched — eta² (between-frame structure) is still the right readout; absolute calibration
  is NOT scored here.
- **Leg B (skill-side, real test — decided 2026-07-04):** project the source frames under the
  38-image cohort, re-embed training windows with perframe normalization, re-bake head, **LOIO gate**
  → then the full 907-frame regional build.  Scripts: `f_leg_b_frame_list.py` (laptop, builds frame
  list + bounds CSVs), `run_f_leg_b.sbatch` + `f_leg_b_process.sh` (Sherlock ~1h wall / 24-task
  array), `f_leg_b_extract.py` (Sherlock MAP venv, extracts I/F crops), `f_leg_b_embed.py` (laptop
  GPU, embeds → `fang_embeddings_f/`), `f_leg_b_loio.py` (LOIO gate, same Δ ≥ −0.02 threshold as
  A1 cycle).  See SHERLOCK_RUN.md Part F for the full step-by-step.
- **Step 1 is a rerun of the timing job with cubes kept:** `KEEP_CUBES=1 sbatch
  run_f_timing.sbatch` (the first run deleted its cubes by default; ~3.7 h, ~36 GB scratch).
- **✅ Leg A0 DONE (2026-07-03b, CPU):** crops extracted + aligned; calibrated frames' 24.9% level
  spread is **real illumination** (median↔cos i r=+0.83; same-incidence pairs agree 1–3% ≈ the
  Walter claim; Lambert overcorrects; empirical **Minnaert k≈0.66**) → F needs an input-side
  illumination layer; pilot now tests **4 mappings** (affine / lambert / **minnaert** / perframe).
  `f_pilot_ifcheck.png` + CSVs; DECISIONS 2026-07-03b.
- **✅ Leg A eta² DONE (2026-07-04, GPU): FAIL — all 4 mappings worse than raw mosaic baseline.**
  Best: perframe eta² 0.233 (partition 0.257); target ≲ 0.03. Choropleth blocks clearly
  visible in all mappings. **Cause = train/deploy mismatch** (mosaic-trained head is
  out-of-distribution on calibrated-frame embeddings); NOT a fundamental F failure.
  Perframe best (most like mosaic stretch); lambert worst (overcorrects cos i).
  Full record: DECISIONS 2026-07-04, `reports/figures/f_pilot_eta2_summary.csv`,
  `f_pilot_{affine,lambert,minnaert,perframe}.png`. **Decision on leg B deferred to Brian.**
- **✅ Leg B LOIO gate DONE (2026-07-04b): FAIL — Δ median AUC −0.0499 (gate −0.02) — but
  strongly BIMODAL, and the perframe uint8 mapping (not F) is the suspect.** 81 cohort frames
  ISIS-processed on Sherlock (32-task array); 36/38 obs_ids re-embedded from calibrated crops
  (2 missing = one failed K04 frame). Median AUC 0.786→0.736, pooled PR-AUC 0.767→0.626 —
  yet 11 images IMPROVE (up to +0.155, incl. project-best 0.951) while 8 drop below 0.5.
  Diagnostics (`_f_leg_b_diag.py` / `_f_leg_b_uint8_contrast.py` / `_f_leg_b_crop_stats.py`,
  `reports/f_leg_b/diag_*.csv`, figures `f_leg_b_diag_{scatter,gallery}.png`): composite
  mechanics null (coverage/overlap/n_crops; frame-mismatch anti-correlates); over-stretch
  hypothesis REFUTED (F uint8 contrast pinned at IQR≈27.7 by construction, ratio ρ +0.09);
  live correlate = composite I/F median ρ +0.35 → **DIM scenes collapse = illumination, A0's
  cos-i axis again**. Cheap iteration available WITHOUT Sherlock: re-embed with minnaert
  (best-motivated) and/or global-affine mapping (~1 h GPU each), re-gate.
  Full record: DECISIONS 2026-07-04b.
- **✅ Leg B mapping iteration DONE (2026-07-05): global −0.0387 / minnaert −0.0341 — both
  FAIL; the mapping family has CONVERGED ≈ −0.034, short of the −0.02 bar.** Fixed stretches
  cure all 4 perframe collapses; minnaert beats global on dim scenes as predicted. Found +
  fixed a SeamMap metadata bug (P20_008839 incidence 4.2759 = decimal-shift of true 42.76,
  verified vs PDS index; OVERRIDES in `_f_leg_b_incidence_check.py`). Residual floor suspects:
  double-resampling blur / dim-scene stretch clipping (1 image) / calibrated noise character.
  **Key unmeasured: retrained-head eta² on the E8_N44 pilot frames** — if block-free, −0.03
  skill vs FULL artifact removal may beat A1's −0.024 for 28% reduction. DECISIONS 2026-07-05.
- **✅✅ Leg B GATE PASSED (2026-07-05b): minnaert + LOG stretch = Δ median +0.0067** (first PASS;
  F now EXCEEDS mosaic 0.786→0.793, pooled PR-AUC +0.017). Log domain (ln I/F = level-independent
  texture DN) is the lever — biggest improvers in the cohort (ESP_068483 +0.235, ESP_069763 +0.119).
  **Cubic resampling REFUTED** (−0.027, worse than bilinear): the HF-texture deficit is not the cap.
  Caveats: mean still < baseline (ESP_053989 minnaert-specific inversion 0.167, diagnose before
  regional); **eta² with retrained head STILL unmeasured** (F's actual purpose). DECISIONS 2026-07-05b.
  **Next-step decision (confirm eta² / fix ESP_053989 / head-rebuild + 907-frame regional) → Brian.**

## NEXT SESSION — decision setup (collected; no decision taken 2026-06-20)

**Where we are:** cause = CTX source-frame radiometry (proven). A1 (per-frame offset+gain) is built
and measured: it **partially** mitigates (28% eta² down) at a small skill cost (−0.024). The artifact
is still visibly present. The question for next session is **which path (if any) to invest in** to get
a clean map without paying unacceptable skill.

**⚠️ UPDATE 2026-06-22 — thermal-ρ RETIRED as a mitigation adjudicator; D RULED OUT.** The original plan
(below) was to run THEMIS thermal ρ FIRST to adjudicate the A1 −0.024 LOIO cost and the de-block (D).
Two findings collapsed that:
- **Thermal is underpowered for this.** Baseline abundance↔TI ρ ≈ +0.07 (leg-1), and A1 removes only 28%
  of an already-modest artifact (eta² 0.011 vs 0.002 null) → expected Δρ is inside bootstrap noise.
  Thermal-on-A1 would almost certainly read "ambiguous" and tell us nothing the eye doesn't.
- **D is RULED OUT (Brian) as circular** — see table. D was thermal's one remaining legitimate
  referee job (the zero-skill-cost option LOIO can't see). With D gone, thermal has no
  mitigation-refereeing role left.
**Thermal ρ survives only in its ORIGINAL role: an independent validation leg for the *final* map**
(does abundance corroborate thermal inertia as external science support — PLAN_RegionalMap §5), which
is a different question from "which mitigation." **The mitigation decision no longer routes through
thermal.** It is now a direct judgement: *how clean a map does the science need?* →
- **good enough** → cheapest geology-agnostic input fix (A1 / A1-λ / A2 / A3 / A-meta) or **E** (accept);
- **must be right for regional discovery** → **F** (source-frame, full removal at the ±2% floor).
Because the circularity that kills D *also* argues against leaning on the capped A-fixes for the final
discovery map, the live decision is essentially **F vs E** for the science map, with the cheap A-fixes
as a partial-credit middle option. **Still NO decision taken.**

*(Superseded reasoning kept for context:)* ~~Run the adjudicator FIRST: THEMIS/TES thermal ρ. LOIO is a
*within-frame* metric and **blind to the cross-frame artifact**, so the −0.024 "cost" may be the model
losing a spurious within-image cue that's harmful across frames. Thermal (external) tells whether
removing the blocks removes noise (ρ up) or signal (ρ down) and adjudicates D.~~ — retired (underpowered;
D ruled out).

### The option space (pros / cons)

| Option | What | Pros | Cons | Cost |
|---|---|---|---|---|
| **A2** — per-frame histogram match | match each frame's full DN CDF (not just median/IQR) to a reference | catches the shape residual A1 misses → likely > 28% reduction; still seam-clean | more skill cost likely (removes more radiometric info); unstable on small frames; still input-DN-only → can't fix per-frame **noise/MTF/compression** character the ViT keys on | re-embed + re-bake + gate + crop (~1 h) |
| **A3** — cross-seam destripe (input) | solve per-frame offset/gain to minimise cross-seam discontinuity | geology-agnostic, principled; no external reference | only offset/gain (~A1 ceiling); more code | medium |
| **A1-λ** — gentler A1 | shrink toward reference by λ<1 (partial normalize) | tunable skill-vs-artifact knob; could find a sweet spot | still partial; adds a hyperparameter to tune | cheap (re-bake sweep) |
| **B2** — local contrast norm (frame-agnostic) | divisive local normalization; or **A1+B2 combo** | attacks the contrast/texture lever A1 leaves; no SeamMap at deploy; combo addresses offset(A1)+texture(B2) | smears seams; **amplifies noise in bland plains → false positives**; bigger input change → bigger skill risk | re-embed + re-bake (~1 h) |
| **C** — radiometric augmentation | re-bake head on augmented embeddings (random offset/gain/contrast/gamma) | teaches invariance to the part A1 can't normalize away; **deploy unchanged (no SeamMap)**; generalises to other radiometric nuisances | **frozen-ViT ceiling → only partial** (head can't undo what the ViT entangled); N×K embedding cost; **prior W2 augmentation was harmful** (caution) | heavy (N×K re-embed) |
| **D** — post-hoc de-block (output) ~~subtract each frame's mean detrended-abundance via the SeamMap~~ | **❌ RULED OUT (Brian, 2026-06-22)** | — | **CIRCULAR**: separating a frame's artifact offset from real geology requires a model of the abundance field — the very unknown the map exists to discover. Would assume regional structure, subtract frame-scale deviations from it, then present the result as a *discovery* of that structure. Especially poisonous for circum-Chryse megatsunami deposits (real coherent between-frame variation D could erase/manufacture). Poisson form doesn't escape it (assumes seam-steps are artifact; real geology may step at a seam). | — |
| **E** — accept + disclose | document the residual as a partially-mitigable 5 m/px frame-radiometry limitation | honest; no further work; lean on coarse-scale + thermal results robust to it | the map still shows blocks; we *know* it's partly mitigable so "irreducible floor" would be over-claiming — frame it as "residual after A1/de-block" | none |
| **A-meta** — illumination-metadata norm | normalize each frame from SeamMap **INCIDENCE / sub-solar geometry / local-time** (no pixel pass) | cheapest physically-grounded lever (≈0 marginal cost); metadata already cached for all 907 frames; scales free to global | only removes the illumination-correlated component; needs an abundance↔incidence model | very cheap |
| **F** — per-source-frame inference | run the model on individual **`ctxcal`-calibrated CTX frames** (not the mosaic) + composite back (per Bickel 2025) | **highest ceiling** — removes the artifact *at source* (frames consistent to ±2%, Walter 2024); recovers low-freq radiometry + 12-bit; published deployment precedent | needs EDR→ISIS recalibration + **retrain head on source embeddings** (train/deploy parity); heaviest at **global** scale (~86k frames) | trivial (907 frames regional) → very heavy (global) |

### Recommended decision sequence (REVISED 2026-06-22 — thermal/D dropped)
The decision is now a direct cost-vs-need judgement, not a thermal-gated sequence:
1. **Decide the bar:** does the *science* (regional circum-Chryse abundance structure, megatsunami
   deposits) need a **clean, trustworthy** between-frame map, or is a **partially-mitigated** map +
   honest disclosure acceptable? This is the real fork.
2. **If "must be right"** → pursue **F** (per-source-frame inference). De-risk first with the 10-frame
   Sherlock timing test (also resolves the stale-`PDS_IMG` / `planetarypy` question). F removes the
   artifact at source with no assumption about the abundance field.
3. **If "good enough"** → take the best cheap geology-agnostic input fix (**A1** as-built, or **A1-λ**
   for a gentler skill/artifact point, or **A2** for more reduction, or **A3** for the principled
   cross-seam solve) and **E**-style disclose the residual. Re-run the 26-tile region.
4. **C** stays in reserve as the deploy-simplest option (no SeamMap at inference) if the SeamMap
   dependency becomes operationally annoying at global scale — accept its partial ceiling.

*Note:* thermal ρ is **not** in this sequence anymore (retired as referee; see UPDATE above). It returns
later only as an independent check on whatever final map is produced.

**Artifacts in place for resumption:** `src/striping.py` (A1 + analysis), `scripts/striping_a1_*`,
`models/deployable_a1/`, `dataset_v2/fang_embeddings_a1/`, figs `striping_a1_*`/`26_frameblocks_*`,
notebook 25, DECISIONS 2026-06-18d + 2026-06-19/20.

## LITERATURE & DATA-ROUTE FINDINGS (2026-06-20) — research complete, NO decision

Comprehensive review; Brian provided 5 paywalled papers, **all read in full**. Five independent
sources converge on the same diagnosis and sharpen the menu — especially **F** and a new low-cost
entry **A-meta**. Compute is now framed for **global** inference (full mosaic = 3,960 tiles ≈
**86,571** source frames), not just the 26-tile box.

### What the papers establish
- **Dickson et al. 2024 ([10.1029/2024EA003555](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024EA003555)) — the exact mechanism.**
  Each CTX frame is independently **contrast-stretched to min/max = mean ± 8σ + a uniform non-linear
  tone stretch**, *per image before blending*, then per-frame **feathering** across seams. "Contrast
  differences rendered in the mosaic are exaggerated compared to reality"; the mosaic "does not
  preserve radiometry within an image" and "should not be used for radiometric statistics." → our
  artifact is the per-frame **contrast** rescale (not a brightness offset) — explains texture/contrast
  dominance (mean-DN Spearman only +0.14). **A1's 28% ceiling follows**: the nonlinear stretch is
  **non-invertible** from mosaic pixels.
- **Walter et al. 2024 ([10.1029/2023EA003491](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023EA003491)) — F's ceiling + independent corroboration.**
  CTX flat-field is **stable to ±2% over the whole ~20-yr mission** → a uniformly `ctxcal`'d EDR set
  is frame-to-frame consistent to ~2%; **no per-frame radiometric drift at source.** Independently
  critiques Murray's per-image `cubenorm`: "homogenization of natural reflectance," a "**high-pass
  filter effect**" — our mechanism, from a calibration expert who never saw our model. Use the new
  **v0003 flat-field**. EDR→ctxcal is HPC-scriptable (planetarypy + ISIS) and recovers **12-bit**
  (mosaic is 8-bit).
- **Fang et al. 2026 ([10.1029/2025JH000827](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2025JH000827)) — the embedder's blind spot.**
  Our ViT's only radiometry statement: the mosaic "ensures **consistent radiometric calibration**" —
  contradicted by Dickson. The frozen ViT was never trained for per-frame invariance; the authors
  themselves note embeddings whose "similarity is **radiometric rather than geological**." Confirms
  **C's frozen-ViT ceiling** (DINO photometric aug already in pretraining; the signal survives).
- **Bickel & Valantinas 2025 ([10.1038/s41467-025-59395-w](https://doi.org/10.1038/s41467-025-59395-w)) — the per-frame deployment precedent.**
  The closest analog (global CNN over the same 86k corpus) **ran on individual source frames off the
  ASU stream, NOT the mosaic** — structurally avoiding cross-frame steps; deduped ~6% overlap
  duplicates; verified no CTX-imaging radiometric bias. This is **F's template**. Their task (local
  detection) tolerated per-frame radiometry; ours (cross-frame regression) does not → F-for-us =
  per-frame source + per-frame normalization.
- **Zhang 2020 (IEEE GRSL 9242244) + Lin/Pang 2024 ([ISPRS, S0924271624003277](https://www.sciencedirect.com/science/article/abs/pii/S0924271624003277)) — canonical RRN = our A3.**
  Solve per-frame **(mean, std)** compensation by global least-squares over **image overlaps** + local
  refinement. **Catch:** needs overlapping pixels the mosaic discarded (it's a partition) → a *proper*
  A3 needs source frames = the **F** route.

### Data-route facts (from our cached SeamMap, verified)
- The 26-tile map spans **907 distinct source frames** (1,371 footprint-polygons). Global = ~86,571.
- Each SeamMap polygon carries **`PDS_IMG`** + **`SESE_LINK`** + 50 metadata fields incl.
  INCIDENCE/EMISSION/PHASE/sub-solar-az/local-time/image-time.
- **⚠️ `PDS_IMG` URLs are STALE — verified all-404 (10/10 sampled, 2026-06-20).** The PDS Imaging Node
  reorganized its tree since the 2024 mosaic release (Dickson hedged the links were "valid at time of
  publication"). The frames themselves are real (CTX 5056-px labels) and the EDRs remain in the
  **permanent PDS archive** (live index pages exist on the USGS mirror `pdsimage2.wr.usgs.gov/archive/
  mro-m-ctx-2-edr-l0-v1.0/mrox_NNNN/` and JPL), but they must be retrieved via a **product-id →
  current-URL resolver** (`planetarypy` — what Walter used — or a verified USGS template), **NOT** the
  cached `PDS_IMG`. So F gains a one-library resolver dependency; `planetarypy` is **not** in the env yet.
  *(RESOLVED 2026-07-02 — no library needed after all: the live URL is a pure template of the SeamMap's
  own `VOLUME_ID`+`PRODUCT_ID` (`src/ctx_edr.py`); the USGS mirror route is dead. See top banner.)*
- **`SESE_LINK` / ASU stream = 8-bit display-stretched browse** (ASU `planetview`; resolves HTTP 200) →
  reintroduces a per-frame stretch → **NOT clean for regression.** The radiometry-preserving F input is
  **EDR (resolved via planetarypy) → uniform `ctxcal` (v0003) → project**.
- **Robbins "Fully Controlled" CTX mosaic ([10.1029/2022EA002443](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022EA002443)) ruled out** as a basemap
  swap: equatorial **±30°N only** (our region is 32–46°N) and "**cosmetically** corrected" (display
  equalization, not radiometric). Strength is geometry, not radiometry.

### F in detail (replaces the old one-line "ensemble over EDRs")
Instead of running on the blended mosaic, run on the **individual CTX source frames** beneath it, then
composite predictions back to a map. Each frame is internally radiometrically consistent (one
acquisition, one illumination) and after uniform `ctxcal` v0003 frames are mutually consistent to ±2%
(Walter) — so the cross-frame artifact is removed *at source*, not patched. **Pipeline:** resolve
PRODUCT_ID → current EDR URL (**`planetarypy` / USGS mirror — the SeamMap `PDS_IMG` is stale/404**) →
download EDR → `mroctx2isis` → `spiceinit` → `ctxcal` (v0003) → `ctxevenodd` → `cam2map`
(calibrated, projected, **12-bit I/F**) → per-frame normalize → frozen ViT embed → head → composite
overlapping frames (median/feather) + dedup (à la Bickel). *(Resolver step SOLVED 2026-07-02 without
`planetarypy` — `src/ctx_edr.py`; see top banner.)* **Requires retraining:** the head learned
mosaic-pixel statistics, so the HiRISE-co-located training windows must be re-embedded from *source*
frames too (train/deploy parity) — this is F's real "rebuild" line-item. **Cost:** regional 907 frames
(trivial); global ~86,571 frames = recalibrate+project+embed the archive (Bickel did detection-only on
86k in ~2 months / 1 GPU; our ViT embed is heavier but Sherlock-parallel). **Highest ceiling, heaviest
lift.**

### New option A-meta (illumination-metadata normalization)
Normalize each frame's radiometry from the **SeamMap metadata we already have** (INCIDENCE / sub-solar
geometry / local time) rather than from pixels — model abundance-vs-illumination and remove the
per-frame illumination component. Cheapest physically-grounded lever (≈0 marginal cost, no pixel pass);
supported by Bickel tracking per-frame illumination and preserving it (no left-right flip).

### Global-compute reframing (the new constraint Brian raised)
Marginal cost *on top of* the (already ~150×-larger) global embedding job:

| Option | Marginal global cost | Why |
|---|---|---|
| A-meta | ≈0 | metadata rescale folded into the embed pass |
| A1 / A2 / A3 (input, on mosaic) | low | one per-frame stats pass; embed pass unchanged |
| D (Poisson de-block output) | ≈0, post-hoc | operates on the final raster — no re-embed |
| F (EDR rebuild) | highest | recalibrate+project+embed ~86k EDRs (+overlap redundancy) + retrain head |

→ Global inference **pushes toward cheap input-side (A-meta / A1) + post-hoc D**, away from F-rebuild —
**unless** F's quality gain (~2% floor, recovers low-freq radiometry + 12-bit) justifies re-deriving
the global embedding. That cost/benefit is the central decision question.

### Papers Brian should read himself
1. **Dickson 2024 §3.2** — the mechanism (essential).
2. **Walter 2024 Intro + §3.3–3.4 + §4** — corroboration + the ±2% ceiling (co-essential).
3. **Bickel & Valantinas 2025 Methods** — the per-frame deployment precedent.
4. (opt) **Lin/Pang 2024 §3** — the principled A3.

~~**Still NO decision.** The thermal-ρ adjudicator remains step 1; the literature changes the *menu and
its cost axis*, not the decision rule.~~ *(Superseded 2026-06-22: thermal-ρ retired as adjudicator, D
ruled out — see the UPDATE in "NEXT SESSION" above; decision reframed to F vs E, and 2026-07-02 Brian
chose "de-risk F first".)*

## INTERIM VERDICT (2026-06-18c — SUPERSEDED by 18d above — §1+§2 run + notebook 25; wrong feature)
- **Recorded with visuals in `notebooks/25_striping_artifact.ipynb`** (incl. zoomed edge panels);
  logic in `src/striping.py`.
- **In raw output, not qmatch.** The structure is identical in `prob_raw` and `abundance`.
- **The model is genuinely sensitive to CTX brightness/texture** (§1b ρ≈+0.14, all 9 tiles > null).
- **Murray-tile assembly seams re-verified CLEAN** (notebook §3 zoom, quantified): full-seam abundance
  step ~1e-4 = 16th pctile of the interior-column geology null; the visible "rectangles" are cosmetic
  4° gridlines + geology + the weak CTX-brightness sensitivity, NOT inference discontinuities. (A
  single cross-seam transect can mislead — must average the full seam.)
- **The specific "CTX frame-stitching seam" hypothesis is NOT confirmed.** The gold-standard seam
  test (§2) is ≈null but underpowered (frame footprints tile too densely to get near/far contrast),
  AND the visible CTX brightness structure looks **dominated by geology**, not linear track seams.
  Directional banding is weak and not vertical.
- **Consequence (Brian's caution applies):** we have NOT established a "CTX-floor limitation" — we've
  if anything weakened the seam mechanism. Item stays **OPEN**.
- **Recommended NEXT (still pre-mitigation):** §3 synthetic brightness gain/offset-step susceptibility
  (clean, no seam-density confound) to quantify *how much* a radiometric step actually moves predicted
  abundance; and a better-powered §2 using the per-pixel *selected*-frame / per-frame incidence map
  (the metadata is in the SeamMap) rather than all candidate footprints. Only after a cause is
  confirmed do we touch §4.

## 0. What we already know (don't re-do)
- **NOT** map_region read-window seams (4096px boundaries don't align with the structure).
- **NOT** per-tile radiometric offsets (per-tile mean abundance is a smooth N→S geology gradient).
- **NOT** HiRISE footprints (overlaid; abundance doesn't trace them — Brian agrees).
- Abundance has a **weak non-monotonic dependence on CTX brightness** (peaks mid-DN, r≈0.07).
- **Leading hypothesis (Brian): CTX mosaic stitching** — the Murray global CTX mosaic is a patchwork
  of individual CTX frames / pushbroom orbital tracks with imperfect radiometric matching; those
  ~rectangular **frame/track seams** are brightness steps the Fang ViT + head key on.

## 1. Characterize the artifact (cheap, no re-inference)  — DONE 2026-06-18c
1a. **Geometry/scale** — FFT / directional autocorrelation of one tile's abundance (and prob_raw):
   dominant orientation (vertical = N–S = CTX orbital tracks?) and wavelength (km). Compare prob_raw
   vs calibrated abundance (is it in the raw model output or introduced by qmatch?). → expect raw.
   **RESULT (`scripts/striping_characterize.py`, `striping_fft_*` / `striping_orientation_summary`):**
   the structure is **aperiodic**, so FFT shows only weak anisotropy (~1.3) and an unreliable
   orientation — FFT is the wrong tool. Confirmed finding: the structure is **identical in `prob_raw`
   and `abundance`** → it is in the **raw model output, NOT introduced by qmatch.** ✓ (matches the
   §1a expectation). A directional banding metric (col-vs-row variance) shows banding is **weak and
   NOT strongly vertical** (vertical≈horizontal ≈ 0.005–0.006), so there is no strong organised
   N–S periodic stripe.
1b. **Co-locate with CTX brightness discontinuities** — edge-detect the coarsened CTX mosaic
   (Sobel/Canny) and the abundance; quantify whether abundance edges coincide with CTX brightness
   edges (vs random). The decisive test of the stitching hypothesis.
   **RESULT (`scripts/striping_seam_test.py`, `striping_seam_*`): POSITIVE and robust.** |∇abundance|
   correlates with |∇CTX brightness| at Spearman ρ ≈ **+0.08…+0.27 (median +0.14), all 9 tiles
   positive**, well above the row-shuffled null (~0.00). So the model **is** sensitive to CTX
   brightness/texture structure — the general mechanism is real. BUT the visible CTX brightness
   structure is **dominated by geology** (valley networks, ridges, craters — see E12_N44), not linear
   track seams.

## 2. Source the CTX frame provenance (the key unknown)  — DATA FOUND; test UNDERPOWERED 2026-06-18c
- Does **Murray Lab** publish a per-pixel **source-frame index / CTX EDR footprint layer** for the
  V01 mosaic? **YES** — the mosaic ships a per-tile **SeamMap shapefile** (one polygon per source CTX
  frame, with PRODUCT_ID / INCIDENCE / EMISSION / IMAGE_TIME metadata), already cached for 9 of the
  26 map tiles (`cache/ctx_tiles/_seammap_<tile>/`).
- **RESULT: the frame-seam test is ≈NULL but UNDERPOWERED, so it neither confirms nor refutes.**
  Rasterizing frame-footprint boundaries and comparing |∇abundance| near vs far from a seam gives a
  ratio of **~1.02 (range 0.86–1.14)** — no meaningful elevation at seams. **Caveat that blocks a
  clean conclusion:** the SeamMap lists ~758 heavily-overlapping *candidate* frames per tile, so
  footprints tile the whole 237 km tile densely → *every pixel is within ~1 km of a boundary* (the
  seam-distance profile maxes at 1.75 km). There is essentially no "far from any seam" reference, and
  the boundaries rasterized are all candidate footprints, **not** the mosaic's actual per-pixel
  selected-frame seams. So this test is confounded; a proper test needs the per-pixel *selected*
  frame (or a per-frame illumination/incidence map) — see §3 / next steps.

## 3. Quantify the model's susceptibility (synthetic, one tile)
- Take CTX patches; apply a **brightness gain/offset step** (simulating a frame seam) and measure
  how much predicted abundance shifts. Establishes how strongly the embedder/head respond to a
  radiometric seam vs true texture. (Ties to the r≈0.07 finding — is it gain, offset, or contrast?)

## 4. Prototype mitigations (on a test tile; only re-run inference there)
**GATE (Brian, 2026-06-18): do NOT start any mitigation until §1–§3 have positively confirmed the
cause.** A mitigation is only meaningful once we know *what* we're mitigating; building a destriper /
contrast-norm before §1b shows abundance edges actually coincide with CTX brightness edges would be
premature. Confirm the issue first, then pick the mitigation that targets the confirmed mechanism.

Pick based on §1–§3. Candidates, cheapest first:
- **Local contrast normalization / high-pass** on CTX before the Fang embedding (removes large-scale
  brightness/frame steps, keeps boulder-scale texture). Re-embed + predict one tile; does striping
  drop?
- **Per-frame radiometric normalization (destriping)** if §2 gives frame provenance.
- **Per-image/per-track standardization** (the deferred W1 bet) at inference time.
- **Post-hoc notch/high-pass on the output abundance** (cosmetic, least principled — last resort).

## 5. Validate any mitigation against the things that matter (don't just make it look nicer)
A mitigation is only worth adopting if it does **not** cost skill and ideally helps the real tests:
- **LOIO skill preserved/improved** — re-score the held-out CV with the normalized CTX input
  (per-image AUC ≈ 0.43 baseline must not drop).
- **Thermal correlation improves** — if the stripes are noise, removing them should *raise* the
  leg-1/leg-2 Spearman ρ (currently leg-1 ≈ +0.07). This is the objective adjudicator.
- Re-render the regional map; visually confirm the striping is gone without washing out real signal.

## 6. Decision
- If a mitigation clears §5 → adopt it, **re-run the 26-tile regional inference on Sherlock** (same
  job-array kit) and regenerate the validation figures.
- If nothing helps without costing skill → **leave the artifact OPEN for further investigation.**
  **(Brian, 2026-06-18): do NOT prematurely write this up as a "5 m/px CTX-mosaic-floor limitation."**
  Failing to fix it with *one* mitigation attempt does not prove it is an irreducible data floor — it
  only proves that mitigation didn't work. Calling it a CTX-floor limitation is a positive claim that
  needs its own evidence (e.g. §1b/§2 actually demonstrating the stripes ARE CTX frame seams the model
  can't escape). If a mitigation fails, record the negative result, keep the item open, and try the
  next hypothesis/mitigation in a later pass rather than closing it. Only conclude "CTX-floor" if §1–§3
  affirmatively establish that mechanism AND multiple reasonable mitigations all fail.

## 7. Deliverable
`reports/figures/striping_*` (the characterization + edge-coincidence + before/after) and a
DECISIONS entry with the verdict. Update notebook 24 / the regional map if a mitigation is adopted.
