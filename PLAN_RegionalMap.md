# PLAN_RegionalMap — regional rock-abundance map + thermal validation (circum-Chryse)

**Created 2026-06-16 (Brian-approved direction).** First real regional deployment of the
frozen head + Stage-1 `CalibrationLayer` ([PLAN_Calibration.md](PLAN_Calibration.md)).
Goal: **convincing, rigorous figures that the CTX-texture abundance method works at
regional scale**, by testing it against an independent, published geological prediction
— the boulder-rich tsunami deposits of [Rodriguez et al. 2016,
*Sci. Rep.* 6:25106](https://doi.org/10.1038/srep25106) — corroborated by independent
**THEMIS / TES thermal inertia**. Ties to Brian's tsunami-transport framing
([[compositional_tsunami_context]]).

---

## 0. The realization that makes this work

Our 38-image cohort **densely samples the exact region** the paper studies: ~21
boulder-rich images sit on the **eastern circum-Chryse / NW Arabia Terra
highland–lowland boundary** (≈0–17°E, 40–47°N), which is the paper's **lHl1 boulder-
deposit zone**. The paper's Fig 2C HiRISE image, **ESP_017355_2260** (16.5°E, 45.8°N),
*is one of our cohort images*. The CTX Murray tiles for that whole boundary block
(`E0_N40, E4_N40, E4_N44, E8_N40, E8_N44, E12_N44, E16_N44` — a contiguous ~20°×8°
strip) are **already cached** (`cache_v2/ctx_tiles/`). So:

- The boundary region is **in-distribution** → global qmatch ([Stage 1](PLAN_Calibration.md))
  is valid there; we lead with where the method is most trustworthy.
- The **distal dusty Chryse plains** are the explicit **out-of-distribution stress-test**
  (the "dusty plains, no boulders" case) — validated/flagged with THEMIS + the deferred
  novelty layer.
- **No new CTX downloads** for the in-distribution core.

---

## 1. The falsifiable predictions (from the paper)

[Rodriguez et al. 2016](https://doi.org/10.1038/srep25106) maps two Late Hesperian
lowland units as tsunami deposits:

1. **lHl1 (older tsunami) = boulder-rich lithic lobes** (meter- to ~10 m boulders),
   concentrated at the **highland–lowland boundary** where overflowing waves dropped
   below the ~1 m/s boulder-transport threshold → **predicted abundance peaks in a band
   along the boundary run-up zone.**
2. **Distal Chryse plains = finer-grained, sorted, less bouldery** → **abundance declines
   lowland-ward.**
3. **lHl1 boulder surfaces are thermally BRIGHT in THEMIS night-IR** (high thermal
   inertia = rocky), with an abrupt upland-ward transition to thermally DARK (dust/fines)
   → **predicted abundance co-locates with the THEMIS thermal-bright zone + the mapped
   contact.**
4. Paleoshoreline elevations (the paper's own values, p.5 + Fig 4 caption):
   **lHl1 ≈ −3795 m, lHl2 ≈ −4100 m** (MOLA) — used only as reproducible *context*
   contours, NOT as the test (see §6; the paper itself calls equipotentiality
   "impossible to rigorously test").

---

## 2. Validation design — five independent legs

No single leg is decisive; together they are convincing. The boulder-location *target*
is the **THEMIS thermal-bright deposit** (independent of CTX), NOT the elevation contour.

| leg | tests | independent of our model? |
|---|---|---|
| **1. Spatial co-location** | predicted-abundance band ↔ THEMIS thermal-bright ↔ mapped contact | THEMIS fully independent |
| **2. Thermal-inertia correlation** | rank-corr(predicted abundance, THEMIS/TES TI) over the region | independent rockiness measure |
| **3. Shoreline-distance profile** | abundance vs distance from the −3795 m contour: boundary peak, distal decay | geometry independent |
| **4. Cohort-truth anchor** | at HiRISE sites, predictions match the actual BoulderNet detections | the ground truth |
| **5. Generalization** | does the band continue along boundary segments with **no** cohort image? | rebuts circularity |

Legs 1–3 = "works at regional scale"; leg 4 anchors to truth; leg 5 rebuts the
train-on-this-region circularity.

---

## 3. Data (CTX already cached; fetch thermal + topo)

- **CTX** — Murray Lab tiles for the boundary block: **already cached**. Windowed reads
  as in the existing pipeline (per-tile oblate-equirectangular CRS gotcha applies —
  [[murray_ctx_conventions]]).
- **THEMIS night-time IR mosaic (100 m/px)** — the paper's exact proxy; for the
  high-res spatial co-location figure. Plus **THEMIS-derived quantitative thermal inertia**
  ([Fergason et al. 2006](https://doi.org/10.1029/2006JE002735)).
- **TES thermal inertia (~3 km/px)** — calibrated absolute TI
  ([Putzig & Mellon 2007](https://doi.org/10.1016/j.icarus.2007.05.013)); carries a
  **dust-cover index** for confound control. **Both thermal sources** (Brian, 2026-06-16):
  THEMIS for the visual, TES for the calibrated correlation + dust mask.
- **MOLA MEGDM (463 m/px)** — to draw the −3795 m / −4100 m context contours.
- All reproject to the CTX mosaic CRS; co-registration across CTX (~200 m offset) /
  THEMIS / MOLA quantified at the boundary (reuse cohort co-reg shifts where available).
- **THEMIS retrieval is net-new** (CLAUDE.md §3.4 had it as future work) → a small
  `src/thermal_retrieve.py` (windowed mosaic reads + reproject), mirroring `ctx_retrieve`.

---

## 4. Compute & scale — FULL-RES ENTIRE BLOCK via CLOUD GPU (Brian, 2026-06-16)

**Decision:** rather than subsample, run the **entire 7-tile block at full S=32 (160 m)
on a cloud GPU.** Embedding is the only cost (the Fang-ViT; head + calibration are free).
Full block ≈ **15 M tiles**; on a cloud A100/H100 in fp16 that is ~**1.5–3 h ≈ $3–15** —
cost is trivial, so the work is **porting to a Linux box + trusting the output**, not the
GPU time. (Laptop hybrid — subsampled overview + full-res zoom windows, ~few GPU-h — is
the FALLBACK if cloud setup stalls.)

**Portability verdict (checked 2026-06-16): favorable.**
- No env spec yet → author `environment.yml` (+ optional CUDA-torch Dockerfile).
- OpenMP bootstrap (`src/modeling/__init__.py`) is **Linux-safe** (`KMP_DUPLICATE_LIB_OK`
  `setdefault`; the libiomp5md reference is a comment, not a dependency).
- Fang ViT checkpoint re-downloadable from **Zenodo 18180801**
  (`mars-mae-dino-vit-base-v1.pth`, 341 MB) — no upload.
- **Re-fetch the 7 CTX Murray tiles on the box** (fast net to Murray Lab) — don't upload GBs.

**Cloud-run scope is narrow:** CTX + embedder + head + calibrator → regional prediction
GeoTIFFs (rich/poor + abundance). Validation (THEMIS/MOLA, the 5 legs) stays on the laptop.

**Cloud setup checklist — DETAILS NEXT SESSION (Brian to provide provider/GPU):**
1. Linux GPU box (A100/H100/L40S; RunPod/Lambda/Vast ~$1–3/h).
2. Reproducible env (`environment.yml` or Dockerfile).
3. git clone + upload small artifacts (`models/deployable/<hash>` head + `calibration.npz`);
   re-download Fang from Zenodo; re-fetch CTX tiles.
4. **`scripts/map_region.py`** — checkpointed/resumable block driver (tile → embed →
   predict → **calibrate** (Stage-1 layer) → mosaic; partial GeoTIFFs + a done-manifest so
   spot-instance interruptions resume).
5. **fp16 inference path** in `FangEmbedder` (halves $).
6. **Numerical-parity check FIRST** — reproduce the laptop `map_pilot` E4_N44 window on the
   box and assert predictions match within tolerance (catches torch/CUDA/fp16 drift). The
   key de-risk before the multi-hour run.
7. Download the (small) prediction GeoTIFFs; discard/keep the ~46 GB embedding cache.

**Laptop-side prep that makes the box turnkey (buildable now, no GPU):** the
`map_region.py` driver, the fp16 path + a throughput benchmark, `environment.yml`, a cloud
setup script + the parity-test script.

### 4a. Target chosen: Stanford Sherlock HPC (Brian, 2026-06-16)

The "cloud GPU" of §4 is **Sherlock** (`sherlock.stanford.edu`). Strategy = **prepare the
turnkey kit here, Brian runs it there, downloads the prediction GeoTIFFs back** — *not*
running Claude Code on Sherlock. The `sherlock_hirise2ctx_runbook.md` doc describes porting
the *whole CPU pipeline*; we **ignore that** — the cohort dataset already exists locally and
is not rebuilt. Our scope is only the §4 inference path.

**Confirmed Sherlock parameters:**
- Group **`mlapotre`**; home `/home/groups/mlapotre/bamaro`. Venv at
  `/home/groups/mlapotre/bamaro/envs/hirise2ctx` (backed-up `$GROUP_HOME`); heavy I/O
  (`cache/`, prediction outputs) symlinked to `$SCRATCH`.
- **GPU partition `gpu`** (so **CUDA torch**, *not* the runbook's CPU-only wheel). The fp16
  path is **already in `FangEmbedder`** (auto-CUDA + `autocast(fp16)`, `src/fm_embeddings.py`),
  so §4 item 5 is effectively done.
- **Scope now = regional (the 7-tile block); architecture must scale to global later**
  (Brian: "eventually we will be doing global inference there"). → `map_region.py` is
  **tile-list-driven** (regional = 7 tiles; global = the full Murray index, no rewrite) and
  **resumable at the (tile, read-window) granularity** so a Slurm wall-clock/pre-emption
  resumes mid-tile.

**Upload artifacts** (exist only on the laptop, ~2.7 MB total, one `scp`):
`models/deployable/86c51a5dca220f63/` (the DeployableHead: `recipe.json` + `seed{0,1,2}`)
and `models/deployable/calibration.npz` (the Stage-1 CalibrationLayer). Fang (341 MB) and the
CTX tiles (GBs) are re-fetched on the box, not uploaded.

**Output GeoTIFFs** (per Murray tile; single-band float32, 160 m/px, `NaN`=nodata/masked):
`*_prob.tif` (calibrated P(rich)∈[0,1]), `*_abundance.tif` (fractional_area qmatch, ≥0),
`*_prob_raw.tif` (uncalibrated P(rich), QA). All validation/figures stay on the laptop.

**The turnkey kit (built here, no GPU):**
1. `scripts/map_region.py` — resumable tile-list block driver (per-window partials + a
   done-manifest → per-tile GeoTIFFs).
2. `scripts/parity_check.py` — reproduce a fixed `map_pilot` E4_N44 window, assert match vs a
   laptop-generated reference (catches torch/CUDA/fp16 drift). **Run this first on Sherlock.**
3. `environment.yml` / `setup_sherlock_env.sh` (CUDA torch) + `run_region.sbatch` (gpu
   partition, resumable) + `SHERLOCK_RUN.md` (exact commands).
4. Throughput micro-benchmark → sizes `--time` and gives the global-cost extrapolation.

---

## 5. The convincing figures (the deliverable)

1. **Regional context** — MOLA shaded relief + cohort footprints + the −3795 m contour +
   the mapped region outline (the paper's Fig 1A, *our* assets).
2. **Money panel** — co-registered region/swath: CTX | **calibrated abundance** | THEMIS
   night-IR | overlaid shoreline contour. Visual: high abundance ↔ thermal-bright ↔
   run-up band.
3. **Quantitative** — (a) abundance vs TI binned scatter + rank-ρ (dust-masked);
   (b) abundance vs distance-from-shoreline profile with the boundary peak.
4. **Truth anchor** — HiRISE image + BoulderNet detections + predicted abundance at a
   cohort site (they match).
5. **OOD honesty** — a distal dusty plain where the model over-predicts but THEMIS says
   low TI / high dust → flagged by the novelty score. Turns a weakness into a rigor demo.

Output: `docs/regional_validation.md` + `reports/figures/regional_*` (committed PNGs).

---

## 6. Shoreline & thermal references (the honest framing)

- **Primary boulder target = THEMIS thermal-bright zone** (independent, ~the deposit).
  This is what legs 1–2 validate against.
- **Paleoshoreline = MOLA −3795 m / −4100 m contours**, drawn by us from the paper's
  stated elevations — *context only*, with the explicit caveat (the paper's own) that the
  shorelines are obscured/resurfaced and **not rigorously equipotential**, so the contour
  is an idealized proxy that diverges from their hand-mapped lobe margins.
- Digitizing the paper's exact red/black lines = optional low-priority context overlay.
- **Claim discipline:** we do **not** claim to prove tsunamis. We claim *our independent
  CTX-texture abundance map recovers the boulder geography the hypothesis predicts,
  corroborated by independent thermal data.*

---

## 7. Honest caveats — and how the design pre-empts each

- **Circularity** (trained on boulder-rich images here) → rebutted by leg 5
  (un-imaged boundary segments) + the *spatial pattern* being finer than "rich" + THEMIS
  independence.
- **TI is indirect** (rockiness ≠ meter-boulders; bedrock/crust/sand also raise TI) →
  report rank-corr not causation; **mask high-dust pixels** (TES dust index), since dust
  is the dominant TI confound.
- **OOD distal plains** (global qmatch can amplify a fooled `P(rich)`) → *show* it
  (figure 5) + THEMIS as the external check + first deployment of the deferred **novelty
  flag** (the Stage-1 layer is built extensible for exactly this).
- **Co-registration** — the ~200 m CTX offset matters at a sharp boundary; quantify
  residuals; use cohort co-reg shifts.
- **Cohort is small (38)** — generalizing to a ~1000 km region is the leap; figures 4–5
  bound it honestly.

---

## 8. OOD tie-in — the novelty flag's first real job

This regional map is the **global-map phase** flagged in [PLAN_Calibration §Stage 1
design pt 5](PLAN_Calibration.md): in-distribution at the boundary, OOD in the distal
plains. The deferred **novelty/OOD flag** (the §2.7 reliability score, usable as a
definitional OOD/extrapolation flag — not a skill predictor) gets wired here as the
optional hook on the `CalibrationLayer`, and **THEMIS is the external ground for whether
the flag is right** (does flagged terrain coincide with thermal-dust/low-TI?).

---

## 9. Phased execution

| phase | content | cost |
|---|---|---|
| **0** | finalize region/swath + the cohort-anchored zoom sites; confirm tile coverage | cheap, no GPU |
| **1** | `src/thermal_retrieve.py`: fetch+reproject THEMIS night-IR + TES TI + MOLA to CTX CRS; draw the −3795/−4100 contours | downloads, CPU |
| **2** | `scripts/map_region.py`: scale-out mosaic inference driver (checkpointed) | code |
| **3** | regional subsampled overview + full-res zoom windows (calibrated) | few GPU h |
| **4** | thermal co-registration + the 3 quantitative legs (corr, profile, anchor); dust mask | CPU |
| **5** | wire the novelty/OOD flag; OOD figure | small |
| **6** | the 5 figures + `docs/regional_validation.md` | — |

---

## 10. Status & open decisions

**SESSION WRAP 2026-06-16:** plan drafted; **extent RESOLVED → full-res entire block via
cloud GPU** (§4). **NEXT SESSION:** Brian provides cloud provider/GPU details → plan the
actual run (env spec, `map_region.py` driver, fp16, parity check). The laptop-side prep
(§4 last paragraph) is the critical path and is buildable without the GPU.

Remaining open decisions (Brian):
1. ~~Overview resolution~~ → **moot** (full-res entire block, §4).
2. **Parity/anchor site** — confirmed default: the E12/E16_N44 dense-cohort segment
   (anchored on ESP_017355_2260 / ESP_055978_2270) for the parity check + the truth-anchor
   zoom (leg 4). Generalization (leg 5) is automatic once the whole block is mapped.
3. **Novelty flag** — wire it in this plan (its natural debut) or defer and ship the
   in-distribution validation first?
4. **Cloud specifics** — provider, GPU, Docker-vs-conda env, spot-vs-on-demand (NEXT SESSION).
