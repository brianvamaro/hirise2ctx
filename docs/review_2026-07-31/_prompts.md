# Reviewer briefs — code review 2026-07-31

Self-contained specs for re-running any review area. **An area is done iff `<area>.md` exists in this
directory.** Give a subagent §1 (shared brief) + §2's entry for the area, and require it to write
`docs/review_2026-07-31/<area>.md` in the §3 template **as its final action**. That write is what makes
the review resumable — never rely on an agent's return value surviving a session-limit kill.

Remaining areas: `geo-crs`, `labeling`, `features`, `fm-embeddings`, `modeling-heads`, `evaluate`,
`calibration`, `other-scripts`, `leakage`, `stats-fallacies`, `invariants`, `numerics`, `tests`,
`docs-consistency`, `notebooks`.

Run **3–4 at a time**, not all at once: with 15 in flight a single limit hit kills everything, whereas
small batches lose at most the in-flight ones and keep every completed file.

---

## 1. Shared brief

You are reviewing a scientific Python research codebase: the **hirise2ctx** pipeline.
Repo root: `c:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise2ctx`

**What it does.** Learns 5 m/px CTX orbital-image texture/shadow → per-tile **rock abundance**, trained
on BoulderNet HiRISE (25–50 cm/px) boulder detections, then maps abundance across the CTX mosaic where
HiRISE is absent. Layers: manifest → reproject HiRISE detections → retrieve CTX → co-register → label
tiles → features / foundation-model (Fang-ViT) embeddings → heads (LightGBM / MLP ensemble / CNN) →
evaluation → calibration → regional map. A separate programme fought a "CTX source-frame striping
artifact" (mitigation A1, and an aborted "F build" = per-frame ISIS photometric calibration + H1
log-median centering + H4 overlap-constrained leveling).

**Current state (2026-07-30).** The **mosaic-based regional map is the shipped deliverable** (A1 is
documented as the mitigation but was never built — see finding R06). The 907-frame F build was
HARD-ABORTED (commit `41a6f26`). Frozen recipe `fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2`; deployed head
`models/deployable/86c51a5dca220f63`. Only `PLAN_RegionalMap.md` is ACTIVE; PLAN_FBuild /
PLAN_StripingArtifact / PLAN_H4_Leveling are CLOSED. Docs: `CLAUDE.md` (invariants), `ROADMAP.md`,
`DECISIONS.md` (386 KB log, authoritative for numbers — grep it, never read it linearly), `README.md`,
`SHERLOCK_RUN.md`, `docs/build_spec.md`, `dataset/DATA_DICTIONARY.md`.

**Load-bearing invariants (CLAUDE.md) — violations are high severity.**
1. **Per-image local-radius CRS.** HiRISE detection shapefiles are equirectangular on a sphere whose
   radius is the *local Mars radius at that image's centre latitude* (e.g. 3393833.26 m), differing
   image-to-image. Code must read each shapefile's own `.prj` and reproject per-image into the common
   CTX CRS. Never hardcode a radius or assume a shared datum. CTX mosaic CRS = Mars_2015 equirect
   clon_0, sphere 3396190 m.
2. **CRS sanity check.** After correct reprojection the residual HiRISE↔CTX offset is O(200 m), not km.
   If km, CRS handling is wrong → must fail loudly.
3. **HiRISE PDS SP1 bug.** The upstream `Standard_Parallel_1=0` bug is auto-corrected via the PDS
   `.LBL`. The override is intentional — do **not** flag it as wrong, but do check it is applied
   everywhere needed.
4. **CTX tiles are GB-scale → windowed reads only. HiRISE RED → read decimated (~5 m/px), never
   full-res.** Prefer `/vsicurl/` range requests.
5. **Target is heavily zero-inflated / right-skewed** — raw base stats must be preserved.
6. **Splits are group-aware leave-image-out (LOIO), never random tiles.** Within-image k-fold is a
   separate deliberate variant.
7. **Manifest-driven.** Adding a manifest row + detection folder must flow end-to-end with no code
   change. Shapefiles discovered by glob `{ObsId}/*-mask-nms.shp`. No hardcoded image lists.
8. **Never report presence AUC** (`y_true > 0`). The meaningful threshold is **fa > 1e-2**:
   `meaningful_auc` / `pr_auc@1e-2` / `precision@5%` + Spearman ρ + per-bin RMSE.
9. **Environment.** Windows + conda env `geospatial`, invoked as
   `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`
   (conda is **not** on PATH; `conda run -c` cannot take newlines — write a temp script instead). Any
   script using torch must `import src.modeling` **before** numpy/pandas. stdlib `urllib` needs
   `truststore.inject_into_ssl()`.
10. **Notebooks are generated.** Source of truth is `notebooks/_build_NN.py`. Logic must live in
    importable `src/` modules; notebooks and scripts only *call* it.

**Already-found — do NOT re-report** (see `docs/CODE_REVIEW_2026-07-31.md` for the full register; only
mention one if your own reading materially corrects or extends it):
R01 `mosaic_geotiffs` tile-phase misregistration · R02 `evaluate.py` `presence_auc` on the reported
surface · R03 HiRISE `MapPixel_mpp` label confound (0.25 vs 0.50 m/px, read by nothing) · R04
`run_stage5.py` swallowed failure + undetectable stale packaged splits · R05 `sweep.py` /
`run_modeling_slim.py` OpenMP import order · R06 `reports/map_a1/` does not exist · R07 A1 160 m-vs-
native statistic mismatch · R08 `a1_normalize_per_frame` leaves un-labelled pixels raw · R09
`recipe_hash` collision + copied metrics on `deployable_f_center` · R10 two-factor abort comparison ·
R11 tautological trend guard + guards 3/4 on the wrong solve · R12 `fbuild_abort_*` has no producer ·
R13 context-box nodata unchecked · R14 `map_region.py` resume + non-atomic writes · R15 Stage-7d
`inconclusive` unreachable · R16 Stage-B uint8 clip fraction unmeasured (the stretch itself is
correctly global and fixed) · R17 `frame_level_spread` Jensen gap + `planefree` docstring false · R18
Stage-B resume race · R19 `edge_cv_for_offsets` fallback mislabel · R20 Stage-D provenance code · R21
Stage-B unverified CRS · R22 within-image streaming iterators.

Also **verified clean** — do not re-file: the calibration LOIO protocol and its in-sample/LOIO
labelling; GeM clamping; embedder train/deploy normalization parity; the MLP `FeatureScaler` is
train-only; `DeployableHead`'s early-stopping inner-val image comes from the training set; per-image
CRS handling in `src/detections.py`; `src/leveling.py`'s weighted normal equations, λ scale-correction,
per-component gauge and sign convention; gate PASS/FAIL inequality directions; gate 1's common
footprint.

**Your job.** Find real defects: bugs, silent-failure paths, statistical/logical fallacies, invariant
violations, discrepancies between code and its documentation/claims, and material misses. A defect that
would change a reported number or a scientific verdict is the most valuable thing you can find.

**Rules of engagement.**
- **Read-only except your own output file.** Do not edit or create anything other than
  `docs/review_2026-07-31/<area>.md`. Do not run notebooks, sweeps, training, or map builds. You may
  read, grep, run `git log`/`git show`, and run small python snippets over **committed report
  artifacts** (`reports/figures/*.csv|json`) to check a numerical claim. Do not touch CTX/HiRISE
  imagery or the network.
- ⚠ **NEVER CALL A PRODUCER FUNCTION, AND DO NOT RUN THE SLOW TESTS.** This is not theoretical: on
  2026-08-04 a reviewer ran `pytest tests/test_labeling.py` and silently overwrote four **gitignored**
  v1 artifacts (`dataset/labels/ESP_069669_2220.{parquet,json}`,
  `cache/reprojected_detections/ESP_065711_1545.{gpkg,json}`), migrating one of nine v1 images across
  the 2026-06-10 y-sign-fix correctness boundary. It was recoverable only by luck — the original label
  values happened to survive inside `dataset/packaged/loio_9fold/y_test_fold6.parquet`.
  **The mechanism:** the producers write straight into the live artifact tree using config-derived
  paths and have **no dry-run mode** — `src/labeling.py:543` (`df.to_parquet`), `:591` (sidecar),
  `src/detections.py:151` (`gdf.to_file`), `src/coregister.py:436`. And **three** tests pass
  `output_dir` / `cache_dir` pointing at the **real** `dataset/` and `cache/` trees rather than a tmp
  fixture: `test_labeling.py::test_stage4_runs_on_ESP_069669_2220`, `test_empty_shapefile.py`, and
  `test_features.py:485::test_features_align_with_labels_row_for_row` (which also overwrites both
  context-patch `.npy` stacks, in exchange for an assertion that cannot fail).
  So *merely calling a producer, or running those tests, mutates the dataset.* `git` cannot restore any
  of it — these paths are gitignored. **All three are `slow`-marked**, so `-m "not slow"` is safe.
  Concretely: (a) never invoke a Stage-1/3/4 entry point or anything that reaches those write sites;
  (b) if you must run pytest, use `-m "not slow"` and never the two tests named above; (c) to answer
  "would today's code produce this artifact?", read the code and reason, or copy inputs to your
  scratchpad and redirect the output path **explicitly**; (d) `load_shift` is a pure read and is safe.
- **Read the actual code before claiming anything.** Cite `path:line`; quote the offending lines. A
  finding with no line-level evidence is worthless.
- **Self-refute before reporting.** For each candidate, spend real effort trying to kill it: is a
  caller already guarding it? is it unreachable? does a test pin the behaviour as intended? does
  `DECISIONS.md` or the relevant `PLAN_*.md` record it as a deliberate choice (grep for the term)? If
  it survives, say what you tried. If it dies, put it in the "Refuted by my own check" section — that
  is valuable output, not waste.
- **No style nits**, no type-hint/formatting/"add more tests" suggestions. Only defects with a concrete
  consequence.
- Distinguish **live** code (shipped mosaic map, frozen recipe, PLAN_RegionalMap) from **dead/closed**
  code (aborted F build, closed striping/compositional work). Both are in scope; severity depends on
  it. Exception: a defect in the F-build/abort machinery is high severity if it could mean the **abort
  verdict itself** was wrong.
- Report at most 6 findings, ranked most-severe first. 3 real defects beat 6 maybes.
- **Write your output file as your final action, even if you found nothing** (an empty findings list
  plus an honest coverage note is a valid, useful result — it marks the area done).

---

## 2. Per-area briefs

### `geo-crs`
Deep-read `src/coregister.py` (560), `src/ctx_retrieve.py` (627), `src/ctx_tiles.py`,
`src/hirise_imagery.py`, `src/pds_labels.py`, `src/ctx_edr.py`. (`src/detections.py`'s SP1/CRS path is
already verified clean, but note `stage1_one_image(manifest_row=None)` silently skips the SP1
correction — check whether any path can hit that.)
Hunt for: hardcoded Mars radii / shared-datum assumptions / a per-image CRS read once and reused;
`set_crs` (assign) where `to_crs` (reproject) was meant; transform/affine errors (row/col vs x/y swap,
y-sign, off-by-one window bounds, pixel-centre vs corner, `rasterio.windows` rounding, composition
order); degrees-vs-metres mixups; equirectangular cos(lat) scale errors; clon_0 vs clon_180 and
longitude wrapping at the seam; whether the O(200 m) residual check really fails loudly and whether
its threshold is right; whether HiRISE reads truly decimate and whether the decimation factor and the
**returned transform** are scaled consistently (classic: decimate the array, forget the transform);
`/vsicurl/` URL construction, retries, and error handling that could silently yield a wrong or empty
array.
Cross-check `tests/test_coregister.py`, `test_ctx_window_geometry.py`, `test_ctx_tiles.py`,
`test_murray_url_padding.py`, `test_sp1_correction.py`, `test_hirise_imagery_sp1_override.py`,
`test_sanity_residual_one_image.py`.

### `labeling`
Deep-read `src/labeling.py` (604) plus `src/qa.py`, `src/manifest.py`, and the label-relevant parts of
`src/config.py`.
Hunt for: is `fa` (fractional area) computed in a CRS where polygon area is latitude-distorted, and is
that corrected? Nested ×2 grid construction, tile bounds, partial tiles at image edges, and whether
tiles outside the detection footprint are dropped or silently counted as zero (conflating "no
coverage" with "zero boulders" poisons the target). Double counting: overlapping polygons, boulders
straddling tile boundaries (clip vs centroid vs whole-polygon), NMS duplicates, multipart geoms,
invalid/self-intersecting geometries. Anything that silently drops zero-abundance tiles or clips small
values (invariant 5). Consistency of `boulder_count` / `fa` / `boulder_area` and whether the
min/max boulder-size filters apply identically to all three. (Note R03: `min_size_m: 1.4105` is one
global floor across a 0.25/0.50 m/px cohort — that specific confound is filed; look for *other*
consequences in the labeling code.)
Cross-check `tests/test_labeling.py` (668), `test_stage2_one_image.py`, `test_empty_shapefile.py` — and
flag any behaviour the tests assert that looks scientifically wrong.

### `features`
Deep-read `src/features.py` (872), `src/spatial_features.py`, `src/colour.py`,
`src/ctx_source_illumination.py`.
Hunt for: texture math (GLCM/LBP/gradient/FFT/lacunarity/canny) — window sizes, normalization,
quantization levels, boundary handling, and whether nodata pixels contaminate statistics (a nodata 0
or −32768 entering a mean/std/GLCM). dtype issues: overflow, wraparound, premature clipping, integer
division, `/255` applied twice, `astype` truncation. Degenerate tiles (all-nodata, constant, 1-px) →
div-by-zero, 0/0, `np.std` of an empty slice, NaN into the model matrix. Whether any spatial/context
feature pools information from outside the tile in a way that leaks across a split boundary.
Illumination: incidence/emission/phase math, subsolar geometry, `cos(i) ≤ 0` in a power,
degrees-vs-radians. **Feature column-order stability**: is the matrix column order deterministic and
identical between train and inference (a dict-ordering dependency silently permutes features at
inference)?
Cross-check `tests/test_features.py` (533), `test_spatial_features.py`, `test_colour.py`,
`test_ctx_source_illumination.py`, and `dataset/DATA_DICTIONARY.md` for column-definition drift.

### `fm-embeddings`
Deep-read `src/fm_embeddings.py`, `src/modeling/mlp_head.py` (475), `src/modeling/loaders.py` (433),
and the training-time extraction probe `scripts/probes/_w2_fang_embed.py`.
(Verified clean already: GeM clamping, `preprocess` normalize-then-resize parity, strict checkpoint
loading, the train-only `FeatureScaler`, the inner-val rotation. Go deeper.)
Hunt for: whether the 3 ensemble members are genuinely distinct (different init/seed) and whether
averaging is in probability or logit space, consistent with the calibration layer downstream; whether
`DeployableHead.save/load` captures **everything** needed for inference (scaler, feature order,
pooling params, nuisance basis, dtype/device) and whether version skew fails loudly or silently
mispredicts; what `scripts/parity_check.py` + `models/deployable/parity_ref.npz` actually pin, and what
they do **not** (R09 notes `deployable_f_center` has no parity ref); the H2 `nuisance_basis` projection
applied identically in fit and predict; any embedding standardization/whitening/PCA fit over all
images rather than train folds.
Cross-check `tests/test_fm_embeddings.py`, `tests/test_deployable_head.py`,
`scripts/probes/_fm_parity_check.py`.

### `modeling-heads`
Deep-read `src/modeling/gbm.py` (722), `src/modeling/cnn.py` (578), `src/modeling/binary_target.py`,
`base.py`, `inference.py`, `sweep_select.py`, `__init__.py`.
Hunt for: two-stage/hurdle composition (`P(nonzero) × E[y|y>0]`) correctness; `balanced` class
reweighting applied at fit but not undone at predict (silently miscalibrated output); log1p / Tweedie /
Huber inverse transforms applied exactly once and in the right place (`expm1(mean(log1p(y)))` is a
biased mean estimator — is that acknowledged?); target-vs-objective consistency across
`fa` / `boulder_count` / `boulder_area` / binary; **`eval_set` provenance at all 7 GBM call sites** (is
early stopping ever on the held-out image?); seeding (torch, numpy, python, cudnn); train/eval mode and
`no_grad`; **DataLoader `shuffle` on an eval loader** (would desync preds from labels); masked-loss
reduction; device mismatches. `sweep_select.py`: is the winning variant chosen on the same held-out
metric later reported as performance (optimistic-selection bias), and is that acknowledged?
`inference.py`: does it re-derive feature order/preprocessing from the artifact or recompute it?
Cross-check `tests/test_modeling_gbm.py`, `test_modeling_cnn.py`, `test_modeling_binary_target.py`,
`test_modeling_loaders.py`.

### `evaluate`
Deep-read `src/modeling/evaluate.py` (753) and `src/modeling/loaders.py` (433). (R02 covers the
`presence_auc` surface at :343/:399/:412-413/:681 — do not re-report it, but **do** report any *other*
place presence AUC leaks into a reported artifact.)
Hunt for: `meaningful_auc` / `pr_auc@1e-2` / `precision@5%` definitions — is precision@5% "top 5 % of
predictions" (correct), and is `k = round(0.05·n)` off-by-one or tie-broken nondeterministically?
Pooled vs per-image aggregation: is "pooled" a concatenation of all folds' predictions or a mean of
per-fold metrics, are the two ever compared as if equal, and does pooling mix per-fold models with
different output scales? Degenerate folds: single-class → `roc_auc_score` raises or NaN; dropped
(biasing the median up) or counted? NaN silently skipped in a mean, changing `n`. `per_bin_rmse`: bin
edges, empty bins, log vs linear space vs how it is reported. Spearman: ties, NaN pairs,
per-image-then-averaged vs pooled. Any metric computed on a transformed target but reported as raw.
Audit `write_run_artifacts` — what lands on disk and whether it can go stale or collide.
Cross-check `tests/test_modeling_evaluate.py`, `test_modeling_evaluate_classification.py`,
`test_evaluate_meaningful_threshold.py`.

### `calibration`
Deep-read `src/calibration.py` (385), `src/reliability.py`, `scripts/bank_calibration.py`,
`scripts/bank_calibration_f.py`. (The LOIO-honesty protocol and the in-sample-vs-LOIO labelling are
verified clean — confirm independently, then go deeper.)
Hunt for: isotonic `out_of_bounds` / extrapolation / clipping / ties / single-distinct-value behaviour;
quantile matching — grids, extrapolation beyond the extremes, sensitivity to `n_quantiles`, and whether
qmatch is **monotone** (if not it reorders predictions and invalidates any ranking metric computed
after it); how the zero atom of the zero-inflated target flows through qmatch (where "de-compression"
can invent abundance); the space (log10 / log1p / linear) the calibrators are fit in and whether the
inverse is applied consistently so reported "abundance" matches the labels' units; whether a banked
calibrator is keyed to the exact head that produced it or can be silently loaded for a different head;
`CalibrationLayer.save/load` round-trip fidelity and behaviour on a missing key.
Cross-check `tests/test_calibration.py`, `tests/test_reliability.py`; grep `DECISIONS.md` for
`qmatch` / `isotonic` / "Stage 2".

### `other-scripts`
Review `scripts/run_stage1..7*.py`, `train_gbm.py`, `train_cnn.py`, `train_binary.py`,
`train_deployable_head.py`, `sweep.py`, `sweep_binary.py`, `sweep_cnn.py`, `sweep_stage2.py`,
`sweep_within_image.py`, `run_modeling_slim.py`, `build_vclaire_manifest.py`, `parity_check.py`.
(R05 covers the `sweep.py` / `run_modeling_slim.py` import order — do not re-report, but do report any
**other** script importing torch-dependent code before `src.modeling`, or doing a network fetch
without `truststore.inject_into_ssl()`. Confirm whether `bank_calibration.py` touches torch.)
Hunt for: **artifact path / config-hash collisions** — the `*/scale_S{n}` + `config_hash` layout: can
two different configurations collide onto one directory (silently overwriting, or worse silently
*reusing* another run's results)? Can a stale artifact be picked up by a glob matching more than
intended? Argparse: flags documented in README/SHERLOCK_RUN that do not exist, defaults differing from
the frozen recipe, unenforced mutually-exclusive options. Copy-paste logic duplicated between a script
and the `src/` module it should call, where the two have since **drifted** (invariant 10) — report the
drift, not the duplication. Bare/broad `except` that swallows a real failure and lets the pipeline emit
an empty or partial artifact (e.g. `run_stage1.py`'s except around `stage1_one_image`), and whether the
runner still exits 0.

### `leakage`
One question: **is any reported number optimistically biased by information flowing from held-out data
into the thing evaluated?** Trace the LOIO protocol end to end across `src/dataset.py`,
`src/modeling/*`, `src/calibration.py`, `src/reliability.py`, `src/fm_embeddings.py`, and the
sweep/train scripts. (Verified clean: the MLP `FeatureScaler` is train-only; `DeployableHead`'s
early-stopping inner-val image comes from the training set. Verify independently, then check the rest.)
Channels to check and report the real ones: (1) feature/embedding normalization or whitening fit over
all images; (2) calibration fit including the evaluation fold; (3) early stopping on the held-out image
— **check GBM's 7 `eval_set` sites, not just the MLP**; (4) hyperparameter/variant selection using the
same held-out metric later reported, and whether the selection bias is acknowledged ("dev wins die at
LOIO"); (5) any decision threshold / top-k cutoff / minconf tuned on the test fold; (6) nested-grid
parent and child tiles in different folds; (7) within-image k-fold spatial adjacency with
`buffer_tiles: 0`; (8) leveling offsets solved on overlaps that include the evaluation frames — what
H4's "skill preserved by construction" actually means in code; (9) group-key correctness — can two
manifest rows share an observation so LOIO leaves in a sibling?
For each real channel give the likely direction and rough magnitude of bias, and note whether
`DECISIONS.md` already acknowledges it.

### `stats-fallacies`
Audit **inferential and logical validity** — the reasoning, not just the code. Read the metric
implementations in `src/modeling/evaluate.py`, `src/fgates.py`, `src/leveling.py`,
`src/stage7d_pooled.py`, and the argument structure in `PLAN_FBuild.md` §5,
`PLAN_StripingArtifact.md` PHASE 2, `PLAN_H4_Leveling.md`, `docs/modeling_results.md`,
`docs/compositional.md`, and the `DECISIONS.md` entries for 2026-07-30 / -30b / 2026-07-09b /
2026-07-05d / 2026-06-02 (grep by date).
(Known: R10 the two-factor abort comparison; R11 the tautological trend guard; R03 the pixel-scale
confound. Build on them, do not re-report.)
Hunt for: **circularity** — metrics mathematically guaranteed to move as claimed (the project caught
"post-H4 η² is circular" and "per-image AUC is blind to leveling"; are there **others** still used as
evidence?). **Underpowered nulls treated as refutations** (H4's 7-frame pilot; leg-B's 21 components).
**Multiple comparisons / forking paths**: how many variants, mappings, λs and gate formulations were
tried before any PASS was declared, and does the implemented design match the "pre-declared" one?
**Spatial autocorrelation** ignored in any correlation/CI (THEMIS ρ +0.07, the MOLA leg, the
trend-guard plane fit, η² significance, the Stage-7d compositional test). **Simpson's paradox** /
pooled-vs-per-image disagreement with the favourable one quoted. **Prevalence dependence** of PR-AUC
and precision@5% compared across image sets with different rich-tile base rates. Any **correlation
described causally**, or a mechanism claim asserted from consistency-only evidence.
Cite the exact claim (doc + line) and the exact code that computes it.

### `invariants`
Systematically audit the repo against each of the 10 invariants in §1. Grep the **whole** repo
(`src/`, `scripts/`, `scripts/probes/`, `notebooks/_build_*.py`, `tests/`), then read the hits.
Suggested greps: hardcoded radii/datums (`3396190`, `3393`, `+R=`, `sphere`, `from_epsg`, `CRS.from_`,
`set_crs`, `to_crs`); full-res HiRISE reads (`out_shape`, `decimat`, `overview`, a `.read()` with no
window/out_shape on a HiRISE JP2 path); CTX `.read()` without `window=`; presence-AUC-shaped code
(`y_true *> *0`, `(y *> *0)`, `any_boulder`, `presence`); hardcoded observation IDs (`ESP_`, `PSP_`) in
`src/`, anywhere the manifest is bypassed, or where `{ObsId}/*-mask-nms.shp` is not the discovery path;
`urllib`/`requests`/`urlopen` without `truststore`; notebook drift (for each `_build_NN.py` is there a
matching `.ipynb`, and does git show the `.ipynb` modified without its `_build` source, or a `_build`
edited but never re-executed so the committed notebook shows stale numbers?).
(Known: R02, R05, and `src/striping.py:118`'s radius default. Do not re-report.)
Do **not** report the deliberate SP1 override as a bug — but do report anywhere the SP1 correction is
**missing** on a path that needs it. Report concrete violations with `path:line`.

### `numerics`
Sweep for **silent wrongness** — code that returns a plausible number instead of an error. Grep-then-
read across `src/` and top-level `scripts/` (probes only if they feed a reported number).
Patterns: bare `except:` / `except Exception` / `: pass` / `continue` / a fallback default that lets a
partial or empty artifact be written — list the ones that can mask a real failure. Empty-array
reductions (`np.mean/std/median/max/min/percentile` on possibly-empty input → NaN into a reported
metric, or `nan_to_num`'d to 0). Every `np.nan_to_num` / `fillna(0)` / `dropna()` — which change a
reported number or silently shrink a denominator? Division by a possibly-zero/NaN denominator;
`errstate` suppression. `np.log/log10/log1p` of a value that can be 0 or negative (predictions, ratios,
areas) — `log10(pred/label)` with either term 0 is the exact statistic the abort used. `x ** (1/p)`
with `x < 0`; `sqrt` of a negative variance. dtype: `astype(np.uint8)` without clipping (wraparound),
`astype(int)` truncation or NaN→INT_MIN, float32 accumulation over millions of pixels, int overflow in
a pixel-count product. Raster nodata sentinels (0, −9999, −32768, NaN) entering arithmetic unmasked.
Float equality; `np.isclose` defaults across very different magnitudes. Mutable default args; in-place
mutation of a caller's DataFrame/array (`inplace=True`, `arr[:] =`, SettingWithCopy, pandas chained
assignment that silently no-ops).
Rank by "could this have changed a number the project reported?"

### `tests`
Audit `tests/` (44 files, 8.3k lines) for **false assurance** and for load-bearing code with no test.
Confirmed baseline: `pytest -m "not slow"` → **490 passed, 21 deselected** in ~50 s.
Hunt for: **tests that cannot fail** — assertions on constants, assertions on a mock the test itself
configured, `pytest.approx` tolerances so wide any implementation passes, `assert x > 0` on an
always-positive quantity, `try/except: pytest.skip()` that skips on the very error it should catch,
tests whose only assertion is "it did not raise". **Over-mocking of the invariants**: per-image CRS,
the O(200 m) residual check, the SP1 override, the decimated read, the windowed read — if these are
tested only against fixtures hardcoding **one** radius/datum, the tests would pass even with the
per-image-CRS invariant broken; check whether **any** test exercises two different local radii.
**The 21 deselected slow tests**: enumerate them and say whether anything load-bearing is covered only
there. **Coverage gaps**: which `src/` modules have no dedicated test file, and which functions in
`src/dataset.py` (841), `src/features.py` (872), `src/modeling/evaluate.py` (753),
`src/modeling/gbm.py` (722) are untested — prioritize by "this computes a reported number".
**Tests asserting wrong science** — a test pinning behaviour this review would call a bug: the most
valuable find. Read `tests/conftest.py` (634 bytes for 44 files) for fixture infrastructure that makes
tests trivially pass. You may run `pytest --collect-only -q`; do **not** run the slow suite.
Note R19 as a worked example: `tests/test_fgates.py:162-174` is named for a fallback branch it never
exercises. Look for more of exactly that.

### `docs-consistency`
Audit consistency between documented claims and code. Docs: `CLAUDE.md`, `ROADMAP.md`, `README.md`
(37 KB), `SHERLOCK_RUN.md` (39 KB), `docs/*.md`, `dataset/DATA_DICTIONARY.md`,
`HANDOFF_NEXT_SESSION.md`, the `PLAN_*.md` files, `DECISIONS.md` (grep, do not read linearly).
**Do this first and be exhaustive:** enumerate every `python scripts/X.py --flag` command in
`README.md` and `SHERLOCK_RUN.md` and check the script exists and accepts that flag (read the
argparse). List the broken ones with doc line + script line.
Then: **DATA_DICTIONARY vs actually-emitted columns** (cross-check `src/features.py` /
`src/dataset.py` column construction) — columns listed but no longer emitted, or emitted but
undocumented. **Numbers that contradict** between ROADMAP / DECISIONS / a PLAN /
`docs/modeling_results.md` (e.g. pooled 0.7832, prec@5% 0.948, median AUC 0.7865; H1 η² 0.081 / 0.128)
— genuine contradictions, not restatements at different precision. **Stale status**: anything besides
`HANDOFF_NEXT_SESSION.md` still asserting a status the 2026-07-30 abort invalidated. **Dead code from
closed plans**: verify by grepping for callers (ROADMAP says `lv.solve_offsets*` is unused) — flag
anything both dead **and** dangerous, and anything documented as available that is actually dead.
**Docs that are themselves wrong**: does `CLAUDE.md` or `docs/build_spec.md` state something the code
contradicts? (Known: R06 `reports/map_a1` missing; R09 the copied recipe metrics; R10 ROADMAP's abort
summary; R15 the `inconclusive` claim; R17b the "903 of 906 directions" claim. Do not re-report.)

### `notebooks`
Audit `notebooks/_build_07..28.py` (11.8k lines, source of truth) and the generated `.ipynb`, plus
`reports/` artifacts and `.gitignore`.
Hunt for: **logic living only in a notebook** (invariant 10) — metric definitions, leveling/compositing
math, gate logic, label/feature construction in a `_build_NN.py` that is not a call into `src/`; rank
by "does a reported figure/number depend on it?" The big ones are `_build_10` (1294), `_build_08`
(1130), `_build_13` (923), `_build_12` (888), `_build_28` (707, the F verdict).
**Duplicated-and-drifted logic**: the same metric/η²/threshold computed both in a notebook and in
`src/` with different code — which produced the reported number, and do they disagree numerically?
**Generated-artifact drift**: for each `_build_NN.py`, is there a matching `.ipynb`, and does
`git log --format=%ad --name-only` show an `.ipynb` modified without its `_build` source, or a `_build`
edited but never re-executed (so the committed notebook shows stale numbers)?
**Reproducibility**: seeds (torch/numpy/random/LightGBM/cudnn); any reported number depending on an
unseeded RNG; hardcoded absolute paths (`C:/Users/brian`, `/scratch/`, `/home/`) in tracked files;
environment assumptions that break a fresh clone.
**Committed artifacts vs their producer**: for `reports/figures/*.csv|json` and `reports/map_*/`,
compare the artifact's commit against `git log -p` on the producing script — a committed CSV whose
producer has since changed is a stale-result hazard. `.gitignore` (touched in `41a6f26`): does it
exclude something needed for reproducibility, or fail to exclude something huge/derived?
(Known: R12 `fbuild_abort_*` has no producer; R14 `map_region.py` resume. Do not re-report.)

---

## 3. Output template — write this to `docs/review_2026-07-31/<area>.md`

```markdown
# Review area: <area>

- **Reviewed at commit:** <git rev-parse --short HEAD>
- **Date:** <YYYY-MM-DD>
- **Verification:** self-refuted (single-agent pass; not independently verified)

## Findings

### <area>-1 — <one-line claim>
- **Severity:** blocker | high | medium | low
- **Liveness:** live-shipped | live-active-plan | dead-closed | unclear
- **Confidence:** high | medium | low
- **Where:** `path:line` (+ related sites)

<What is wrong and why, 2-5 sentences.>

- **Failure scenario:** <concrete inputs/state -> wrong output, crash, or wrong scientific conclusion.>
- **Evidence:**
  ```
  <quoted offending lines, with path:line>
  ```
- **Self-refutation attempted:** <what you tried in order to kill it, and why it survived.>
- **Fix:** <the minimal correct change.>

### <area>-2 — ...

## Refuted by my own check
<Candidates that looked like defects and did not survive. One line each + why. This prevents a future
session re-filing them.>

## Verified clean
<Specific things you checked and found correct, so the effort is not repeated.>

## Coverage note
<What you read in full, what you only grepped, and what you could NOT check and why.>
```
