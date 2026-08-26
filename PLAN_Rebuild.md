# PLAN_Rebuild.md — the batched v2 rebuild (a WAYPOINT, not a destination)

> **Status 2026-08-25: THE REBUILD IS COMPLETE. §0 gates closed; steps 1 → 12 EXECUTED and
> verified.** Step 11 shipped both map arms at **26/26 tiles**, 156/156 rasters sha256-verified
> against their own sidecars, **26/26 cell-for-cell co-registered**, one `grid_id` and one
> size-floor basis (`1c1cb55`). **Step 12 promoted them to `reports/map_region` /
> `reports/map_a1`** (the pre-R01 product is archived at `reports/map_region_g1`), built the six
> regional mosaics on a footprint that closes exactly, re-derived every §6 number, and passed
> 12/12 sidecar gates.
>
> **Headline: rich prevalence 0.373272** (predicted 0.3733), pool 161,005 → **164,644**, CRS gate
> 194.7 m median, and **the frozen recipe transfers to the corrected label basis unchanged**
> (median per-image AUC −0.0087, inside one SE). **A1's skill cost collapsed to −0.0024** (banked
> −0.024) with no THEMIS-ρ cost, and its raw artifact η² on the like-for-like pilot crop fell
> **0.2327 → 0.1298 (−44%**, vs the banked −28%).
>
> ⚠ **But η² measured RELATIVE to its own rotation null did not improve at all** (median ratio
> 1.599 → 1.639; A1 better on only 106 of 234 windows). A1 renormalises per frame and so
> narrows the *bulk* of the field — `prob_raw` IQR ratio 0.85, while its sd **rises** 3%, so it is
> **not** a uniform compression — which lowers the geological floor along with the artifact. **The raw
> reduction is real and is the quantity the banked pair measured, but it is not evidence that the
> artifact shrank relative to geology.** A1 remains a *partial* mitigation shipping as a documented
> caveat — see §6.
>
> Step 11's own history is worth reading before quoting anything from it — the A1
> array's 10 h timeout was the **GPU allocation** (RTX 2080 Ti 17.6 s/window vs TITAN Xp
> ~202), not the arm; R14's overlap guard rested on a false premise and is now a fraction
> gate on `prob_raw` with a 1e-6 per-cell floor; and the manifest *index* was damaged three
> ways while every raster stayed perfect. DECISIONS 2026-08-24d → 2026-08-25h.
> Supersedes nothing; it *executes* the "Complete v2 rebuild DAG" in
> [docs/CODE_REVIEW_AUDIT_2026-08-06.md](docs/CODE_REVIEW_AUDIT_2026-08-06.md) §"Complete v2 rebuild
> DAG" (12 steps). [docs/PENDING_REBUILD.md](docs/PENDING_REBUILD.md) lists *what* is stale; the audit
> lists the *dependency order*; this file is the *execution plan* — commands, roots, verification
> gates, abort conditions and the split between this laptop and Sherlock.

**Why this rebuild exists, and why it is deliberately modest.** Every code task in the 2026-07-31
review register is discharged, but the artifacts on disk are not what the current code produces. This
pass regenerates them. It is a **waypoint**: v3 (BoulderNet retrained, applied to more diverse
locations) is already in progress and will force another full rerun, so this pass answers only
*"what does the current dataset support under corrected code?"* — it does not re-open any modelling
question.

## Decisions already ruled — do NOT relitigate

| # | Decision | Where |
|---|---|---|
| 1 | **Retrain the frozen recipe AS-IS.** No bake-off, no re-selection. Pre-registration is the point. | DECISIONS 2026-08-18b |
| 2 | **Size floor deferred to v3** as a design requirement. A 1 m diameter floor (0.785 m²) sits *below* the current 1.4105 m filter and would remove nothing; unification needs *raising* to 2.664 m at ~67% of fine-cohort labelled area. Arithmetic is banked — do not redo it. | DECISIONS 2026-08-18b |
| 3 | **Build IN PLACE into `dataset_v2`**, gated on the SHA-256 backup pass completing first. `D:` is a genuine full copy (78.34 GB of `dataset_v2` inside 125.55 GB); restore is ~25 min at the drive's measured 53.5 MB/s. ✅ Gate closed 2026-08-19: the hash pass returned `VERIFIED`. | 2026-08-19, this plan §0 |
| 4 | **FM path only.** R27/R28 change `features/`, but the frozen recipe is embeddings-only, so no GBM sweep and no W1 error atlas. PENDING_REBUILD rows 2–3 stay open with an explicit note. | 2026-08-19 |
| 5 | **Land the `rasterio.open` hoist before step 11.** Measured 23% of map wall-clock; output-identical. | DECISIONS 2026-08-18c, 2026-08-19 |
| 6 | **Job ARRAY on Sherlock for the maps**, not one long job. R14 makes pre-emption safe. | DECISIONS 2026-08-18b |

## §0 — Gates that must close before step 1

| # | Gate | How |
|---|---|---|
| 0a | ✅ **CLOSED 2026-08-19 — `VERIFIED`.** 8/8 roots, 11,260 files / 125.55 GB, 0 hash mismatches, ≈42 min, verdict JSON at `D:\HiRISE2CTX Backup\_backup_meta\backup_20260819_103028.json`. No bus drop — the lid-closed diagnosis is confirmed. ~~**SHA-256 backup pass completes.**~~ This is the one argument that survives building in place: right now two independent copies exist, so a bit flip is detectable by comparison; after an in-place overwrite `D:` is the only record and its byte-integrity is unproven. | Machine kept awake (Brian is handling this — no `powercfg` command needed). Attach `D:`, run `scripts/backup_artifacts.ps1 -SkipCopy -Hash` to completion, confirm a verdict JSON is written. |
| 0b | **Detach `D:` and leave it detached for the whole rebuild.** An unmounted snapshot cannot be clobbered by a stray `--out-dir`. Do **not** refresh it mid-rebuild — it is the point-in-time record of the pre-rebuild state. | physical |
| 0c | ✅ **CLOSED 2026-08-19 — landed and bit-neutral.** Steady state 33.4 → 22.7 s/window (**−32 %**, better than the −28 % projected); ≈16 GPU-h across 52 tile-renders. Verified three ways (§4). Leg 2 passed against **pristine HEAD** (bit-identical). It also surfaced a cross-run non-determinism anomaly, since characterised as 1 run in 15 and NOT caused by the hoist — §4a. | §4 |
| 0d | ✅ **CLOSED 2026-08-19** — `pytest -m "not slow"` = **800 passed, 1 skipped, 21 deselected**, identical to the pre-change baseline, after the hoist + the 7 fake / 8 stub test edits. | local |
| 0e | ✅ **CLOSED 2026-08-19** — committed and pushed as **`bdc1d19`** (11 files, +901/−116). Tree clean, `HEAD == origin/fm-deployable-head-and-map-pilot`. **This is the code provenance point the rebuild runs from.** | `git status` |

## §1 — Ground rules that hold for every step

1. **The write guard is TEST-ONLY.** It does not cover scripts or notebooks. Producers have **no
   dry-run mode**. Every hand-run command below is written out in full for that reason — do not
   improvise a root.
2. **Building in place is deliberate, and it means a mid-stage abort leaves a mixed generation.**
   Post-2026-08-06 sidecars carry `inputs.{ctx_window_sha256, hirise_mask_sha256, coverage_mask,
   coreg_shift_id}`, so a mixed state is *detectable* — but **verify each step before starting the
   next** (§3's "verify" column is not optional).
3. **Never run two CTX-heavy jobs at once.**
4. **DO NOT regenerate the seven F stores**:
   `dataset_v2/fang_embeddings_f{,_global,_minnaert,_minnaert_w,_minnaert_wl,_minnaert_cubic,_minnaert_center}`
   (~2.7 GB). F was **hard-aborted 2026-07-30**. They are backed up and must survive untouched. Any
   step-6 command that would write them is wrong.
5. **`dataset/` (v1) is excluded** from this rebuild entirely.
6. `import src.modeling` **before** numpy/pandas in any new torch script (OpenMP/MKL).
7. Invoke as `C:\Users\brian\anaconda3\Scripts\conda.exe run -n geospatial --no-capture-output python -u ...`.
   ⚠ `conda run ... python -c` **rejects newlines in the argument** — put multi-line probes in a file.

## §2 — Where each step runs, and why the upload list shrank

The 2026-08-18b ruling was "GPU on Sherlock, Stages 1–5 local, ~17.5 GB upload (`context_patches`
17.0 + ckpt 0.32 + `ctx_windows` 0.19 + labels/splits 0.01)". **The profiling on 2026-08-18c changes
this, and the plan reflects the corrected version:**

- **Step 6 (embeddings) does not need Sherlock.** The training pool is **161,005 tiles**, i.e. ~161 k
  ViT forwards. At the measured **730 img/s** that is **≈3.7 min of GPU** (≈8 min if both the P32 and
  P96 inputs are banked). The laptop does this over lunch. So **`context_patches` (17.0 GB) never
  moves**, and the upload shrinks from ~17.5 GB to the head + calibration artifacts (a few hundred MB).
- **The A1 embedding arm cannot move to Sherlock cheaply anyway.** `--norm a1` calls
  `src.striping.a1_stats_native_tile(tile, frames)`, which **streams the parent Murray tile** and reads
  the cached SeamMaps — `cache/ctx_tiles/` (24 zips, 19 `_seammap_*`, 36 `_frames_*.gpkg`). None of
  that is in the 17.5 GB list. Locally the inputs are already on disk; the cost is the streaming pass,
  ~3 min × 24 parent tiles ≈ **1.2 h**, CPU/IO-bound.
- **Only step 11 is genuinely GPU-heavy**: 26 tiles × 2 arms = 52 tile-renders. At the banked
  ≈0.6 GPU-h/tile that is ≈31 GPU-h, ≈**23 GPU-h after the open-hoist**; ~4–5 h wall on a 6-way array.
  The 26 Murray zips (41 GB) are fetched **on** Sherlock, as before.

**Net split: everything local except step 11.** That is simpler and smaller than the 2026-08-18b
sketch, and it is a consequence of measurement, not a change of mind about Sherlock — the *reason* for
Sherlock (laptop sleep during a long unattended GPU job) applies to step 11 and only step 11.

## §3 — The 12 steps

Timings are **reconstructed from the original v2 build's artifact mtimes on this same machine** (§3a). Two are genuinely unknown and marked **(unknown)**.

⚠ **They are unreliable in BOTH directions — treat §3a as an order-of-magnitude sanity check, not a schedule.** Step 1 ran **3.5× slower** than its mtimes implied (R23 added a byte-integrity scan + score-rank analysis the original never did); step 2 ran **5× faster** (its original 58 min *included downloading the Murray zips*, now cached). Measured so far: steps 1–2 in **~20 min** against a combined ~61 min estimate. The 1.5–2 day total still holds.

| # | DAG step | Where | Command | Verify before proceeding |
|---|---|---|---|---|
| 1 | Stage 1 — reproject detections. **Required**: the R23 drop-null/filtering provenance fix landed. | ✅ **DONE 2026-08-20, ~9 min** | `python -u scripts/run_stage1.py --config config_v2.yaml --all` | ✅ **39/39, 0 failed** (the failure mode is a per-image `FAILED` + `None`, **not** a nonzero exit — the count is the real gate). ✅ SP1: 32 corrected + 7 `trusted_prj`, matching the manifest's `{True: 32, False: 7}`. ✅ **7 distinct per-image radii, 3384416.50–3393833.26 m, none the standard 3396190** — the #1 gotcha, confirmed live. ✅ Mixed-floor contract present 39/39: **`source_integrity` + `null_geometry_basis`** (⚠ `realised_label_basis` is a **Stage 4** field, `src/labeling.py:1070` — corrected 2026-08-20). ✅ R23's 3 truncated sources reproduce DECISIONS 2026-08-06o exactly (ESP_017355_2260 kept **359,933**, Δ0). Total kept polygons 6,278,986. *(The O(200 m) residual check moved to step 3 — Stage 1 only reprojects.)* |
| 2 | Stage 2 — CTX windows + coverage masks | ✅ **DONE 2026-08-20, 11 min** | ⚠ **NO `--all` flag** — `run_stage2.py` is ONE positional ObsId per invocation, so this step is a **loop** over the 39 manifest rows: `for o in $(tail -n +2 hirise_40_vclaire.csv | cut -d, -f1); do python -u scripts/run_stage2.py $o --config config_v2.yaml; done` (continue past failures and tally — a per-image failure is not a nonzero exit) | ✅ **39/39, 0 failed**, no network (all 39 resolved to cached zips). ✅ `ctx_window_sha256` 39/39. ✅ coverage-mask **`version: 2` on all 39** (`version: 1` anywhere ⇒ R74 did not take). ✅ method `decimated_nonzero_with_interior_shadow_fill`, `max_interior_hole_px` 16, **6,514 shadow px re-marked across 39/39**. ✅ `footprint_source` **`polygon_bbox` ×39, 0 fallbacks** — confirms R67 latent on live output. Coverage fraction 0.4666–0.6313, median 0.5450. |
| 3 | Stage 3 — co-registration + QA | ✅ **DONE 2026-08-20, ~40 s** | `python -u scripts/run_stage3.py --config config_v2.yaml --all` | ✅ **CRS SANITY GATE PASSED: |shift| min 79.9 / median 194.7 / max 327.3 m** — O(200 m), not km. This is also the end-to-end proof step 1's seven per-image radii were applied right. ✅ 39/39 solved, 0 skipped; `block_median` 38 + `single_window_fallback` 1 (`ESP_046803_2325`, 3/44 blocks confident). ✅ R65 components under `block_field` (`quality_version: 2`, `confident_fraction`, `all_block_peak`, `*_is_conditional`) with `peak_correlation_kind` labelling 38 conditional-median vs 1 post-shift-Pearson. ✅ `shift_id` 39/39 and **39/39 bind the Stage-2 `ctx_window_sha256` + mask `version: 2`**. ⚠ `coregistration.enabled: false` in the config is a **DEAD KEY** — nothing reads it; do not read it as "v2 has no coreg". Binds to the exact Stage-2 digests; emits `shift_id` and the R65 *components* (`all_block_peak`, `confident_fraction`, `*_is_conditional`, `quality_version`) rather than a conflated `peak_correlation`. Sanity: centroid offsets under `sanity.centroid_max_km: 15.0`. |
| 4 | Stage 4 — labels | ✅ **DONE 2026-08-20, ~7 min** | `python -u scripts/run_stage4.py --config config_v2.yaml --all` | R29: coverage mask shifts with the polygons (`coreg_mask_shift` present). R80: `realised_size_basis.realised_physical_min_size_m` emitted per image. **Record the new rich prevalence** — the R74 counterfactual predicts 0.3598 → **≈0.3733**; a materially different number is a finding, not a nuisance. Confirm which images survive and **finalize `splits.schemes.*.n_folds`** before step 5 (currently 38 / 152). |
| 4b | Stage 4b — patches + features | ✅ **DONE 2026-08-20, 16 min** | `python -u scripts/run_stage4b.py --config config_v2.yaml --all` | R27: `lacunarity_shadow_b{2,4}` is **NaN**, never `0.0`, on shadow-free tiles (21.2% of S≥32 rows previously `0.0`). R28: `edge_*` from **quantile** thresholds 0.80/0.90 — expect the per-image `edge_density` spread to collapse from 12.2×; that is the intended effect. `context_patches` regenerate here and feed step 6. |
| 4c | **`measure_size_floor.py`** **(unknown, ~10–30 min)** — must run **after Stage 4** and **before any map driver**; `models/deployable/size_floor_basis.json` does not exist yet (only `--dry-run` has run) and it **goes stale whenever Stage 4 re-runs**. | local | `python -u scripts/measure_size_floor.py --labels dataset_v2/labels --detections "C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise_40_vClaire" --pds-labels cache_v2/pds_labels --tile-px 32 --min-size-m 1.4105 --out models/deployable/size_floor_basis.json` | File exists and `is_file()` (⚠ `Path("")` is `.` and `Path(".").exists()` is True — the gotcha this guards). Pixel scale must come from `cache_v2/pds_labels/{obs}.LBL`, **never** the manifest (blank for two `LabelSource: none` rows). Expect ~27 distinct floors, tile-weighted mean ≈3.37 m²; both drivers stamp `SIZE_FLOOR_*` on every raster including `_prob.tif`. |
| 5 | Stage 5 — splits + packaged datasets | ✅ **DONE 2026-08-20, ~2 min** | `python -u scripts/run_stage5.py --config config_v2.yaml --all` | R04 propagates failure as a **nonzero exit** — check it. Package metadata binds per-obs label/feature content digests plus each label sidecar's R74 `inputs`; `loaders.verify_package_freshness` must pass from `load_fold`. R97: within-image splits land on the correct snap step. |
| 6 | Fresh **baseline** + **A1** embeddings. **Two arms only.** | local **~1 h baseline + ~1.3 h A1** | baseline: `python -u scripts/probes/_w2_fang_embed.py --tile-px 32 --norm none`  ·  A1: `python -u scripts/probes/_w2_fang_embed.py --tile-px 32 --norm a1 --out-suffix _a1` | Writes `dataset_v2/fang_embeddings` and `..._a1` **only**. ⚠ **Confirm the seven `_f*` stores' mtimes are unchanged afterwards** — that is the concrete check on ground rule §1.4. R07: the A1 statistic is the per-frame **native** one (`src.striping.A1_ARM`), identical on both sides; R38's `A1_VALID_FLOOR = 1` rides along so DN 0 means nodata only. |
| 7 | Forced frozen-recipe **LOIO predictions**, all seeds + ensemble, per arm | ✅ **DONE 2026-08-21/23** (baseline ~28 min via `_fm_freeze_window ... --no-verdict`; A1 ~40 min via `striping_a1_loio.py`) | `python -u scripts/probes/_fm_freeze_window.py ... --force` per arm | **`--force` is mandatory** — the script returns a cached `predictions.parquet` if one exists, and a stale reuse here silently poisons steps 9–12. Every prediction artifact must retain **`obs_id, ti, tj`** (the A1 LOIO CSV historically dropped `ti,tj`). ⚠ `OUT_ROOT` in `scripts/probes/_w2_fang_probe.py` is **hardcoded**, not a flag — probe-tier scripts were outside isolation criterion 4. Read it before running. |
| 8 | Train the all-data **deployable head**, per arm | ✅ **DONE 2026-08-23/24, ~100 s/arm** → `models/deployable_g2/a5ffca2dcc536855` + `models/deployable_a1_g2/**66ec8b755b9c0b20**` ⚠ **OMIT `--norm-arm`** — let `infer_norm_arm(store)` derive it. Passing the literal `a1` records an arm the A1 map driver refuses (`A1_ARM` is the versioned `a1_native_perframe_tilesupport_v2`); that killed all 6 array tasks 2026-08-24 | `python -u scripts/train_deployable_head.py --dataset-dir dataset_v2 --store-name fang_embeddings --target fa_gt_1e-2 --norm-arm none --out models/deployable_g2` (and `--store-name fang_embeddings_a1 --out models/deployable_a1_g2`, **NO `--norm-arm`**) | R07: `norm_arm` is **part of the recipe hash** — the two arms must produce *different* hashes. Eleven heads previously shared `86c51a5dca220f63` with the arm recorded nowhere; if the new baseline and A1 hashes match, the fix is not in effect. |
| 9 | Fit the **calibration layer**, per arm | ✅ **DONE 2026-08-23** (baseline gates PASS; A1 forced, ECE 0.0523 vs 0.05) | `python -u scripts/bank_calibration.py --predictions <arm preds.parquet> --labels-dir dataset_v2/labels --out <arm>/calibration.npz --scale-px 32` | Completeness + anti-join assertions must pass (recovered/missing keys cannot vanish silently). Emit pooled results **and** R54's per-image `mean(pred)/mean(true)` distribution + count, and record which aggregation level governs promotion. Must **fail before write** on rejection. |
| 10 | Materialize the versioned globally anchored regional grid | Sherlock | — (`COARSE_GRID_ID = murray_v01_clon0_R3396190_ppd11855_S32_anchor_lonlat0`) | ⚠ **The DAG text here is STALE**: "render baseline tiles first" no longer applies — R07 removed A1's dependency on the baseline raster, so **either order works**. Both drivers must consume the one grid spec; `assert_shared_lattice` and `assert_murray_sphere` are the runtime tripwires. |
| 11 | Render **26 tiles × 2 arms** to new generation paths | 🟡 **IN PROGRESS** — baseline **21/26** (2 partial at 144/144 windows, 3 not started); A1 array running. Sherlock allocated **RTX 2080 Ti**, not L40S, so the 8 h/10 h `--time` limits bite; resubmit resumes | `sbatch run_region_array.sbatch` (baseline) then the A1 equivalent via `scripts/striping_a1_map.py`. Match the parity reference exactly with `BATCH=96`. | **Any missing input or output is a failure, not a reduced footprint.** Do not resume from an existing final TIFF unless complete upstream provenance matches — `tile_is_reusable` + the sweep manifest enforce this, and the A1 path now builds its own `a1_sweep_manifest` (incl. `a1_seammap_digest` over every shapefile sibling, `.prj` included). R06 closes here: **A1 has never actually been generated.** |
| 12 | Mosaics, QA, promotion, docs | ✅ **DONE 2026-08-25** | `scripts/map_mosaics.py` · `scripts/map_arm_eta2.py` · `scripts/map_sidecar_qa.py` · `bank_calibration.py --report-only` · doc sweep | ✅ **Promotion:** old `reports/map_region` archived to `map_region_g1`; `map_region_g2` → **`map_region`**, `map_a1_g2` → **`map_a1`**; both step-11 gates re-run clean *after* the rename, so zero code churn (`src.striping.MAP_DIR` etc. now read the corrected product). ✅ **6 mosaics** (3 layers × 2 arms) at **5925×11852**, `require_shared_lattice=True` — which *is* the R01 gate and fails by design on the archived pre-R01 arm. **Footprint CLOSES exactly**: 56,865,526 finite = 26×1479² − 7,940 intra-tile nodata on 6 tiles; both arms identical (`only_a = only_b = 0`), so the arms are differenceable and the A1−baseline mosaic is written. ✅ **Sidecar QA 12/12 gates PASS** (`step12_sidecar_qa.csv`), generation-aware. ✅ **THEMIS re-fetched** onto the corrected grid, `assert_coregistered` dx=dy=**0.000 m** (the archived one is confirmed −100/+80 m out). ✅ **§6 all re-derived.** ✅ PENDING_REBUILD emptied except rows 2–3. |

### §3a — Where these timings come from, and the total

Measured 2026-08-19 by reconstructing the **original v2 build's** per-stage wall clock from artifact
mtimes. Same machine, same cohort, so they transfer directly — this is evidence, not estimation.
⚠ Stage 1's raw mtime span reads 392 min; **clustered, the real run is 2.6 min** (74 files in one
burst, plus 4 stragglers touched 6.5 h later). Do not quote the span.

| stage | original v2 run | evidence |
|---|---|---|
| Stage 1 | **2.6 min** (39 img) | `cache_v2/reprojected_detections`, 74 files, 05-28 12:27→12:30 |
| Stage 2 | **58.4 min** (~90 s/img) | `cache_v2/ctx_windows`, 05-28 12:36→13:34 |
| Stage 3 | **unrecoverable** | all 39 `cache_v2/coregistration` sidecars share one mtime — bulk-rewritten later, so mtimes say nothing. v1 was ~5 s for 10 images but with `enabled: false`; the block-median FFT solve on 38 is heavier. **Time the first image and record it.** |
| Stage 4 | **7.0 min** | `dataset_v2/labels`, 06-10 18:19→18:26 |
| Stage 4b | **11.4 min** | `dataset_v2/features` + `context_patches`, 06-11 12:03→12:15 |
| Stage 5 | **unrecoverable** (12-day span = repeated repackaging); 48.9 GB written, I/O-bound | `dataset_v2/packaged`, 1,603 files |
| Step 6 baseline | **123.9 min** for **four** stores (S=32 *and* S=64) | `dataset_v2/fang_embeddings`, 152 npz, 06-12 11:59→14:03. S=32 only ⇒ **~1 h**. Note this is **13× slower than the GPU term** (~161 k forwards ≈ 3.7 min at 730 img/s) — the step is dominated by window reads, slicing and npz writes, **not** the ViT. |
| Step 6 A1 | 13.0 min historically — **but that number is obsolete** | R07 replaced the cheap whole-window `a1_stats(arr)` with the **per-frame native** statistic, which streams each parent Murray tile: 24 tiles × 2.58 min ≈ **62 min**, so budget **~1.3 h**. |
| Step 7 LOIO | **16.4 min** per arm | `models/fang_probe/fw_emb_mlp_*_gem96_S32_fa_gt_1e-2`, 06-12 21:54→22:10 |

**Total, if nothing fails:**

| | |
|---|---|
| Steps 1–9, local | **≈5–6 h** of compute, dominated by Stage 2 (~1 h), step 6 (~2.3 h) and Stage 5 |
| + per-step verification (§3's "verify" column) | **+2–3 h** of judgement, not compute |
| Step 11, Sherlock | **≈28 GPU-h** (23 baseline post-hoist + 5 A1) ⇒ **~5 h wall** on a 6-way array, **plus queue wait** |
| Step 12, local | **~1–2 h** incl. the doc sweep |
| **Realistic end to end** | **~1.5–2 working days**, and it is *attention*-bound, not compute-bound |

**What could blow this up, in order of likelihood:** (1) Stage 3 and `measure_size_floor` are genuinely
unmeasured; (2) Sherlock queue wait is unbounded and not included; (3) an in-place mid-stage failure
costs recovery judgement, not just a re-run; (4) a verification gate actually *failing* — Stage 4's
prevalence landing far from ≈0.3733 would be a finding to investigate, not a step to repeat.

## §4 — Step 0c: the open-hoist, in detail

**Measured** (DECISIONS 2026-08-18c): `scripts/map_region.py:630` calls `read_tile_window(zip_path, …)`
inside the window loop, and `src.mapping.read_tile_window` opens `/vsizip/…` on **every** call —
**144 opens/tile at 7.95 s each = 0.318 h/tile, 23% of map wall-clock**. With one open held, a 4096²
window read costs **1.4 s** and a full sequential pass over the whole 47,420² tile costs **16.6 s**.

Cause is verified, not assumed: the inner TIFF is `compression=None`, `tiled=False`, `blockysize=1`
inside a DEFLATE zip member, so GDAL inflates ≈2.25 GB to reach the strip-offset table on every open.

**Change.** Add a keyword-only `dataset=None` to `read_tile_window`; when given, skip the open and
window-read the supplied handle. Open once per tile inside `map_one_tile` under a `with`, so a 26-tile
run holds one handle at a time, not 26.

**Blast radius — wider than "one call site", and it is the tests.** `read_tile_window` has **six
production callers** (`map_region.py:630`, `striping_a1_map.py:312`, `map_pilot.py:152`,
`parity_check.py:93`, `probes/_evidence_gapfill_map.py:67`, `striping_a1_infer_crop.py:70`). Only the
two loop drivers need the hoist — but **seven tests monkeypatch `read_tile_window`** with a fake whose
signature is exactly `fake_read(zip_path, inner_tif, row_off, col_off, size)`
(`test_map_region_global_grid.py` ×5, `test_mapping_context_nodata.py` ×2, `test_map_region_resume.py`).
Passing a new argument breaks all of them unless each fake gains `**kw`. That is mechanical, but it is
the actual work: budget the edit across seven fakes, not one function.

**⚠ `striping_a1_map.py` needs the same hoist** or the A1 arm keeps paying the full 0.318 h/tile.

**Acceptance — two legs, because parity_check alone is NOT sufficient.** `parity_check.py` calls
`read_tile_window` **once, directly** — it never exercises the driver loop, which is the only thing
the hoist changes. So:

1. **`scripts/parity_check.py --rtol 0 --atol 0`** must pass. This proves `read_tile_window` still
   returns the identical window. ✅ **The yardstick is verified valid (2026-08-19):** the banked
   `models/deployable/parity_ref.npz` (emitted 2026-06-16) still reproduces at **`max|d| = 0.00e+00`
   on `prob_raw`, `abundance` *and* `prob(cal)`** on current code, torch 2.12.0+cu130, this GPU. So
   exact equality is an achievable bar, not an aspirational one.
2. **A driver-loop comparison.** Run `map_region.py --limit-windows N` into an **absolute scratch
   `--out-dir`** before and after the change and assert the partial `.npz`s are array-identical. This
   is the leg that actually covers the hoist.

If either leg is not exact, the hoist does not land and step 11 runs as-is.

**Two caveats on the reference itself, inherited not introduced.** It **predates the R13 gate record**,
so it does not pin the masking policy and `parity_check` warns and falls back to the CLI thresholds.
And its window masks **0 tiles**, so it exercises neither nodata gate. Neither weakens leg 1 for *this*
change — the hoist alters how the array is fetched, not what the gates do to it — but the module
docstring's advice stands: emit a second, gap-bearing reference on `E-8_N32` before relying on
`parity_check` to catch a gate regression.

**Explicitly NOT in scope** (measured and rejected, DECISIONS 2026-08-18c): band reads (a further
~5% but it reorders the loop and R14's partial bookkeeping would need re-checking); gate-before-embed
(worth ≈0 here — the circum-Chryse mosaic is fully populated, `n_usable == n_valid` in the profiled
## §4a — RESOLVED: a single unexplained run in 15; the guard stays at 1e-6

**Found by leg 2 of gate 0c, and it is not caused by the hoist** (running the *unmodified* code twice
reproduces it; original-run-2 == hoisted-run-1 == hoisted-run-2, bit-identical). Full evidence:
DECISIONS 2026-08-19b.

Two facts that are individually harmless and jointly dangerous:

1. **The production sweep duplicates cells.** Measured on `E-12_N36`, 144 windows: **2,250,000 cells
   emitted, 2,187,441 unique, 62,559 duplicated = 2.78 %.**
   ⚠ This **contradicts `overlap_disagreement`'s docstring**, which claims "measured on the sweep this
   driver uses, 900 cells over 36 windows with **0 computed twice** … within one run this returns
   `(0, 0.0)` **by construction**". That measurement came from a 36-window *test* sweep. The
   conclusion (agreement within one run) still holds — a cell's value depends only on the cell — but
   via the docstring's *other* argument, not the partition claim, which is false at production scale.
   *(It also corrects my own 2026-08-19 entry, which quoted 4.51 % from pre-R01 sidecars.)*
2. **Identical code, run twice, can disagree by ~1.5e-4 on `prob_raw`** — ordinary fp16/cuBLAS
   reduction-order variation. **The isotonic calibrator amplifies that to 0.13 on the shipped `prob`
   raster** by stepping across a knot. Observed once in four run-pairs; cause not identified
   (plausibly cuBLAS algorithm selection under different free VRAM on an 8 GB card).

**The consequence.** `scripts/map_region.py:755` does
`if n_dis and max_dis > 1e-6: raise SystemExit("… Refusing to assemble")`. A tile whose partials span
**two runs — i.e. a Slurm pre-emption resume** — has 62,559 colliding cells, any of which may differ
by ~1e-4. **The assembly can refuse.** That is exactly the scenario decision 6 depends on ("R14 is
what makes pre-emption safe — resubmit and it resumes"), and step 11 is a 26-tile × 2-arm array job.

**Characterised 2026-08-19 (Brian's call), and it resolved the question.** 10 further independent
runs: **45 of 45 pairwise comparisons bit-identical**, 0 would trip the guard. Decisive control:
`src/mapping.py` and `scripts/map_region.py` restored to **pristine `HEAD`** from git produce output
**bit-identical to the hoisted code** — so the hoist is exonerated against the committed baseline, not
just against itself. **Leg 2 PASSES.**

**The anomaly is singular: 1 run in 15.** `hoist_before`, the session's first `map_region` invocation,
disagrees with all fourteen later runs (including pristine HEAD), with identical deltas against every
one of them. **Cause not identified**; candidates are a transient cuBLAS algorithm choice under
different free VRAM, or an uncorrected bit error on a consumer GPU with **no ECC**. Ruled out: the
hoist, the refactor, code drift, run order.

**RULING: leave the guard at 1e-6.** It is **fail-safe** — it refuses to assemble, it does not ship a wrong raster, so a trip costs one `--force` re-render (~0.9 GPU-h), not a corrupted deliverable. At ~1 in 15 runs the expected cost is small, and if the mechanism is an uncorrected memory error then a tripped guard is a *real error signal* that a 1e-3 threshold would suppress. Weakening a detector built after a 63.1 %-wrong raster, on one unexplained event, is the wrong trade.

**Triage if step 11 ever dies with `cells were written twice with DIFFERENT values`:** `max|Δ|` of order **1e-4 or below** = this phenomenon → re-run that tile with `--force`. `max|Δ|` of order **0.1–1** = the stale-partial case R14 exists for → **investigate, do not force.**

**Options considered and NOT taken:**
- **(a) Raise the guard** above the observed noise. 1e-3 still catches what it was built for — the
  stale-partial case showed **63.1 %** of pixels from the wrong run, orders of magnitude above it.
- **(b) Force deterministic inference** (`torch.use_deterministic_algorithms`, TF32 off, pinned cuBLAS
  workspace). Makes the guard sound as written and makes maps reproducible, which has provenance value
  beyond this bug; costs unmeasured throughput and may not cover every kernel.
- **(c) Deduplicate cells** so windows truly partition. Removes the failure mode *and* recovers the
  2.78 % — but deletes the cross-run detector's only signal, which is the R14 defect that took a
  63.1 %-wrong raster to find. **This revises the "bad trade" call in DECISIONS 2026-08-19**: it is a
  better trade than stated there, because the duplication is not merely wasted compute, it is the
  mechanism that makes the guard trip.

**The hold on step 11 is LIFTED.** Residual risk accepted and bounded: a resumed tile can refuse assembly; recovery is one `--force` re-render.

## §4b — Sherlock hand-off for steps 10–11 (prepared 2026-08-23)

Steps 1–9 are done locally. Only step 11 is GPU-heavy, so only step 11 goes to Sherlock — the
*reason* being laptop sleep on a long unattended job, demonstrated 2026-08-18, not throughput.

### Upload: 347 MB, nine items, all digested

| what | path | sha256[:16] |
|---|---|---|
| head, baseline (`norm_arm=none`) | `models/deployable_g2/a5ffca2dcc536855` | `29e833be74e5cc15` |
| calibration, baseline (gates PASS) | `models/deployable_g2/calibration.npz` | `290a86614f190ced` |
| size-floor basis | `models/deployable_g2/size_floor_basis.json` | `4e22a85aa1f02135` |
| head, A1 (`norm_arm=a1_native_perframe_tilesupport_v2`) | `models/deployable_a1_g2/66ec8b755b9c0b20` | *(re-digest before upload)* |
| calibration, A1 (**forced**, ECE 0.0523) | `models/deployable_a1_g2/calibration.npz` | `6f2d7a77b5e70a0c` |
| size-floor basis (A1 copy) | `models/deployable_a1_g2/size_floor_basis.json` | `4e22a85aa1f02135` |
| Fang ViT checkpoint | `models/pretrained/mars-mae-dino-vit-base-v1.pth` | `bdaacc1b930559ba` |
| sbatch, baseline arm | `run_rebuild_map_array.sbatch` | `a6206817f5eaf328` |
| sbatch, A1 arm | `run_rebuild_a1_array.sbatch` | `f01c11d67e638628` |

The two basis copies share a digest deliberately: the size-floor basis is a property of the **label
pool**, not of CTX preprocessing, so both arms take the same file. One copy sits beside each head so
each arm's provenance is self-contained.

**Fetched ON Sherlock — do not upload:** the 26 Murray zips (~41 GB); the SeamMap shapefiles
(`load_frames` pulls them out of the zips via `/vsizip/vsicurl/` range requests and caches them).
**Stays local:** `context_patches` (18.3 GB — embeddings were computed here) and `packaged` (50 GB,
tabular/GBM only).

```bash
# from the repo root, after `git push`
rsync -avP models/deployable_g2 models/deployable_a1_g2 \
      <sunet>@dtn.sherlock.stanford.edu:hirise2ctx/models/
rsync -avP models/pretrained/mars-mae-dino-vit-base-v1.pth \
      <sunet>@dtn.sherlock.stanford.edu:hirise2ctx/models/pretrained/
# on Sherlock: git pull   (brings both sbatch files + the open-hoist + all step 1-9 code)
```

### ⚠ `git pull` on Sherlock is not optional

The **open-hoist** (`bdc1d19`) is what makes step 11 cost ~23 GPU-h instead of ~31. Without it every
tile pays 144 × 7.95 s of redundant `rasterio.open`. The same pull carries the `--no-verdict` flag,
the embedding staleness check, and the A1 tile-key fix.

### Two new sbatch scripts, and why the old one is not reused

`run_region_array.sbatch` is left **untouched** as the record of how the shipped map was made. The
rebuild uses `run_rebuild_map_array.sbatch` + `run_rebuild_a1_array.sbatch`, which differ in four
ways that each matter:

1. **26 tiles, not 19.** The old script covers `EXPANSION_TILES`. The rebuild invalidates all 26 —
   rendering only the expansion would silently ship 7 pre-rebuild tiles.
2. **`--model-parent models/deployable_g2`.** `resolve_model_dir` picks `hits[-1]` **by name**; with
   the legacy `86c51a5dca220f63` also present the default is a coin flip.
3. **`--size-floor-basis`.** Absent it, `size_floor_tags` warns and emits **no** `SIZE_FLOOR_*` tags,
   so 52 rasters would ship unable to state which boulders they count.
4. **`BATCH=96`, not 256.** 96 is the parity reference's batch, and 256 buys nothing measurable:
   32/96/256/512 → 766/723/730/731 img/s. The larger batch was a guess that costs parity
   comparability for no throughput.

The A1 script passes `--head` directly (that driver has no `--model-parent`) and R07 makes it
**refuse** a head it cannot verify as the `a1` arm — so the path must be the armed
`7bbd8a8e1d377f6e`, never the legacy `models/deployable_a1/86c51a5dca220f63`.

### Order, cost, and what closes

Either arm may go first — **the DAG's "render baseline tiles first" is stale**, R07 removed A1's
dependency on the baseline raster. Baseline ≈23 GPU-h; A1 ≈23 + ~5 GPU-h (its per-tile overhead is
+0.193 h, measured). ~4–6 h wall each on a 6-way array. **R06 closes with the A1 arm — A1 has never
been generated at region scale.**

### If assembly fails

`N cells were written twice with DIFFERENT values`: `max|Δ| ≲ 1e-4` is the cross-run fp
non-determinism (1 run in 15, cause unidentified, no ECC) hitting the 2.78 % of cells the sweep
duplicates → re-run that tile with `--force`. `max|Δ| ~ 0.1–1` is the stale-partial case R14 exists
for → **investigate, do not force.**

window); window-overlap dedup (**2.78 % measured on the R01 sweep — the 4.51 % first quoted was from pre-R01 sidecars, see §4a**; overlap is what R14's cross-run detector runs
on); larger `--win-px` (**worse** — 8192 gives ~5.3% duplication vs 4096's 4.51%); `--batch` changes
(flat 32→512); fp16 weights and `channels_last` (both 0.99×); SDPA (already in use).

## §5 — What this rebuild deliberately does NOT do

- **No GBM sweep, no W1 error atlas** (decision 4). Stage 4b still runs — `context_patches` feed
  step 6 — but the tabular downstream is not re-derived. PENDING_REBUILD rows 2–3 stay open with the
  note *"features regenerated; downstream tabular numbers not re-derived."*
- **No size-floor unification** (decision 2). `measure_size_floor.py` *measures and stamps*; it does
  not impose.
- **No F-arm anything.** Seven stores stay frozen.
- **No recipe change.** Native-96 (measured **5.64–6.09×**), dense/convolutional inference for the 9×
  stride redundancy, and distillation are all v3.
- **No `dataset/` v1.**

## §6 — Numbers that must be re-derived afterwards ✅ ALL RE-DERIVED (step 12, 2026-08-25)

Every one of these is prevalence-conditioned, and R74+R29 moved rich prevalence 0.3598 →
**0.373272** (~6% of the pool changed status). **They were stale the moment step 4 completed;
the right-hand column is the rebuilt value and is what may now be quoted.**

| Quantity | Banked value | **Rebuilt value** | Source |
|---|---|---|---|
| pooled PR-AUC @ `fa > 1e-2` | 0.7832 | **0.7826** (−0.0006) | DECISIONS 2026-08-21 |
| median per-image AUC | 0.7865 | **0.7778** (−0.0087, inside 1 SE ≈ 0.0144) | same |
| precision@5% | 0.948 | **0.9638** (+0.0158, partly mechanical — prevalence rose) | same |
| `meaningful_auc` | — | **0.8342** | same |
| Spearman(pred, `fractional_area`) | — | **0.6050** | same |
| rich prevalence | 0.3598 | **0.373272** | DECISIONS 2026-08-20d |
| **A1's η² (raw)** | 0.196 → 0.141 (−28%) | **regional window median 0.1444 → 0.1145 (−20.7%)**; tile scale 0.2105 → 0.1817 (−13.7%); **like-for-like on the E8_N44 pilot crop 0.2327 → 0.1298 (−44.2%)** | `scripts/map_arm_eta2.py`, DECISIONS 2026-08-25h |
| **A1's η² RELATIVE to its own rotation null** | never measured | **1.599 → 1.639 (+2.5%) — NO improvement**, better on only 106/234 windows. Read this beside the raw row, never instead of it | same |
| A1's ΔAUC | −0.024 | **−0.0024** (Δ pooled PR-AUC **+0.0082**) | DECISIONS 2026-08-23 |
| A1's THEMIS ρ cost | — | **none**: per-tile median ρ 0.0653 → 0.0654 | `scripts/map_arm_eta2.py` |
| calibration `t2_y` max / pool max `fa` | 0.293242 | **0.293242 — unchanged**, and that is the R84 invariant holding, not a stale copy | DECISIONS 2026-08-23c |
| size-floor mixture | 78.39 / 21.61% of 161,005 tiles; 27 floors; mean 3.3687 m² | **78.73 / 21.27% of 164,644 tiles; 20 floors; mean 3.3914 m²** (range 1.5626–5.5719, 38 images) | `SIZE_FLOOR_*` tags on every shipped raster |
| R54 per-image level | not emitted | **pooled 1.0220 / 1.0278 (baseline / A1) but only 8/38 and 7/38 images within ±20%**, range 0.013–6.5× | now emitted by `bank_calibration.py`; DECISIONS 2026-08-23c |

⚠ **The η² rows are the only ones where banked and rebuilt are not on one basis.** The banked pair
came from a single **pilot crop** on the **pre-R01 lattice** under a **pre-R07 A1 definition**, and
A1 had never been rendered as a map at all (R06). The `pilot_crop` figure is the closest
like-for-like successor — same world extent, corrected lattice, `A1_ARM =
a1_native_perframe_tilesupport_v2` — and on that basis A1's raw-η² reduction is **larger** than
banked (−44% vs −28%) at **a tenth of the skill cost**.

⚠⚠ **But the raw-η² headline overstates what A1 does, and step 12 measured by how much.** A1
renormalises per source frame, which compresses the **whole** field — so it lowers the rotation
null as well as the between-frame term (window-scale null p95 median 0.0771 → 0.0622). Three
paired, per-unit views of the same 234 windows:

| view | what it asks | baseline → A1 | A1 better on |
|---|---|---|---|
| raw η² | the banked quantity | 0.1444 → 0.1145 (**−21%**) | 144/234 (62%) |
| excess (η² − own null mean) | artifact above this window's own geology | 0.0887 → 0.0690 (**−22%**) | 134/234 (57%) |
| **ratio (η² ÷ own null p95)** | **artifact RELATIVE to geology** | **1.599 → 1.639 (+2.5%)** | **106/234 (45%)** |

**On the null-relative metric A1 is a coin flip and very slightly worse.** Per-window raw Δη²
spans −0.41 to **+0.44**, and 9 of 26 tiles get *worse* on raw η² (worst: `E-12_N32` 0.207 →
0.371). So A1 works substantially **by compressing the field, not only by removing frame
structure**. Neither arm reaches the 0.05 F-reopening bar — that bar belonged to the aborted F
build — and this is the quantitative statement of why A1 is a **partial** mitigation shipping as a
documented caveat. Quote the raw reduction only alongside the ratio.

Report on the project's standard metrics only — `meaningful_auc` / `pr_auc@1e-2` / `precision@5%` +
Spearman ρ + per-bin RMSE at the `fa > 1e-2` rich/poor threshold. **Never presence AUC.**

**Docs swept (step 12):** `PLAN_Rebuild.md` §6 (this table), `ROADMAP.md`,
`docs/model_evidence.md`, `docs/PENDING_REBUILD.md`, `DECISIONS.md`. Checked and found to carry
**none** of these quantities, so untouched: `README.md`, `docs/index.md`, `docs/methods.md`,
`dataset/DATA_DICTIONARY.md`, `SHERLOCK_RUN.md`. `docs/modeling.md` and
`docs/modeling_results.md` carry only **GBM-path** numbers, which decision 4 deliberately does
not re-derive — PENDING_REBUILD rows 2–3 stay open for exactly that. ⚠ §6's original claim that
`docs/modeling_results.md` and `docs/index.md` quote the headline FM numbers was **wrong**; they
never did. This is the fifth §-level gate row this rebuild has found overstated.

## §7 — Open, and NOT to be pre-decided by execution

1. **Leg 2's TI product** — physical TES TI + a DCI dust mask, or THEMIS TI alone with a dust caveat?
   Open since 2026-07-13. Not a rebuild dependency.
2. **THEMIS night-IR re-fetch/reproject** — the **only genuine network item**; R01 moved the mosaic
   transform and the 15 GB source is not cached. MOLA does not need it. Parallel track, not on the
   critical path.
3. **Leg 4's LOIO re-run** with the size floor as a covariate. R83 measured
   `Spearman(sub-floor area share, per-image AUC) = −0.468, p 0.003`, surviving inside the coarse
   cohort alone (−0.467, p 0.016, n 26) — so mostly small-boulder terrain, not pixel scale. Runs after
   step 9.
4. **`torch.compile` / TensorRT on Sherlock** — untestable on Windows (no Triton). At ~13% of the
   card's fp16 peak there is theoretical headroom. TensorRT would be ~1.5–2× but is **not
   bit-identical**, so it needs an explicit parity-tolerance ruling before it could be used. Worth one
   hour of probe; not a plan dependency.
5. **Whether a mid-rebuild abort should roll back or resume.** Building in place makes this a live
   question the first time a stage fails. Default assumed here: fix forward and re-run the failed
   stage, because every stage is idempotent given its inputs — but confirm at the time.
