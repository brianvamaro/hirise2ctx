# hirise2ctx

HiRISE boulder detections → CTX rock-abundance paired dataset.
See [CLAUDE.md](CLAUDE.md) for the full build spec; see [DECISIONS.md](DECISIONS.md)
for runtime-verified facts and deviations; see [ROADMAP.md](ROADMAP.md) for the
phase-by-phase index of planning + status.

For a narrative paper-Methods-style description of the pipeline aimed at readers who
won't touch the code (collaborators, reviewers, committee members) see
[docs/methods.md](docs/methods.md).

## Status

**Current phase (2026-07): regional deployment + striping-artifact Phase 2 (invariance & leveling).** See
[ROADMAP.md](ROADMAP.md) for the full plan index and the live `project_state_*` memory notes
for session state. The arc since the v1-reportable wrap:
- **Foundation-model recipe frozen** ([PLAN_FM.md](PLAN_FM.md)): Fang-ViT embeddings + 3-seed MLP
  ensemble (`fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2`) productized to one all-data `DeployableHead`.
- **Calibration shipped** ([PLAN_Calibration.md](PLAN_Calibration.md)): `CalibrationLayer`
  (isotonic P(rich) + quantile-match abundance); Stage-2 retraining ceiling = 5 m/px CTX floor.
- **Regional map** ([PLAN_RegionalMap.md](PLAN_RegionalMap.md)): 26-tile circum-Chryse abundance
  map on Sherlock GPUs (`scripts/map_region.py`), validated vs MOLA shoreline + THEMIS/TES thermal
  (legs in progress). Notebook 24.
- **Striping artifact SOLVED + mitigation in Phase 2** ([PLAN_StripingArtifact.md](PLAN_StripingArtifact.md)):
  the regional-map rectangular blocks are **CTX source-frame radiometry** (notebook 25). A1
  (per-frame normalization) = partial (28% eta² ↓ at −0.024 skill). The **F campaign** (inference on
  ISIS-calibrated source frames, notebooks 26–28) closed the input-mapping leg (skill PASS, eta² FAIL)
  → the Brian-approved **Phase-2 docket H1–H6**: **H1** per-frame log-median centering **PASS**
  (eta² 0.179 → 0.081, embedder amplification killed), **H2** linear nuisance-subspace removal
  **FAIL/refuted**, **H3** consistency-regularized head **FAIL** (eta² and skill collapse on one
  axis), **H4** overlap leveling **PASS** on the pilot + leg-B
  ([PLAN_H4_Leveling.md](PLAN_H4_Leveling.md)): partition eta² 0.128 → 0.0505, held-out edge-CV
  disagreement halved, within-image skill preserved by construction (pooled Δ PR-AUC −0.0104, inside
  the −0.02 gate). **H1+H4 is the first stack to reach the
  reopening bar** (eta² ≲ 0.05 at skill ≥ −0.02); the 907-frame build is planned in
  [PLAN_FBuild.md](PLAN_FBuild.md), gated on its §0 checklist.

**v1-reportable wrap (2026-06-03):** Stage 7 compositional analysis landed at "modest empirical
support for transported provenance over crater-ejecta-locally-sourced; surface-maturity alternative
needs Tier 3." Paper-Methods writeups in [docs/](docs/).

| Deliverable | Where |
|---|---|
| **Headline compositional thread writeup** | [docs/compositional.md](docs/compositional.md) |
| **Modeling thread methods writeup** | [docs/modeling.md](docs/modeling.md) |
| **Modeling results (deep dive)** | [docs/modeling_results.md](docs/modeling_results.md) |
| **Data-pipeline methods writeup** | [docs/methods.md](docs/methods.md) |
| **Docs index + style guide** | [docs/index.md](docs/index.md) |

**Pipeline (build stages):**

| Stage | What | Status |
|---|---|---|
| 0 | Load manifest + config | ✓ |
| 1 | Per-image detection ingest + reproject to common CTX CRS (auto-corrects upstream HiRISE PDS `Standard_Parallel_1=0` bug, polygon side) | ✓ both manifests (v1 priority10: 10 rows; v2 vClaire: 39 rows) |
| 2 | Download Murray Lab CTX tile + window around HiRISE footprint + HiRISE coverage mask (auto-corrects same SP1 bug on JP2 side) | ✓ full sweep on both manifests |
| 3 | Co-registration (sub-pixel phase-correlation translation; v2 uses robust block-median) | ✓ |
| 4 | Label generation on nested ×2 grid (8/16/32/64 CTX px) | ✓ |
| 4b | Per-tile CTX texture features (9 families) + bundled context patches | ✓ |
| 5 | Leave-image-out splits + dataset packaging | ✓ schemes `loio_9fold` (v1), `loio_nfold` (v2 38-fold), `within_image_4fold` |
| 5b / 5c | Binary-classification reframing + within-image diagnostic CV | ✓ (shipped 2026-05-27) |

**Modeling (Stage 6):**

| Item | Status | See |
|---|---|---|
| v2 LOIO modeling A/B (denser labels vs v1 ceiling) | ✓ shipped 2026-05-29 | [modeling_results.md §9](docs/modeling_results.md) |
| Phase A2 compression diagnosis + 4 hurdle variants + `boulder_count` target | ✓ shipped 2026-05-29 (`a003d33`) | [modeling_results.md §11](docs/modeling_results.md) |
| Stage 6a spatial-context neighbour features | ✓ dev-PASS at 5×5/S=32 (promotion deferred) | [modeling_results.md §12](docs/modeling_results.md) |
| Stage 6b CTX-source illumination + H3 mechanism check | ✓ strict-FAIL net flat; H3 falsified, Stage 6e mechanism empirically validated | [modeling_results.md §13](docs/modeling_results.md) |
| Stage 6c image-level reliability gate | ✓ soft PASS at +0.056 pooled-global PR-AUC via Strategy B | [modeling_results.md §14](docs/modeling_results.md) |
| Path A bank (P1+P2 full-v2 LOIO promotion) | open | [PROMOTION_QUEUE.md](PROMOTION_QUEUE.md) |

**Compositional analysis (Stage 7):**

| Item | Status | See |
|---|---|---|
| Stage 7.0 feasibility gate (3-image trio on truth labels) | ✓ PASS 2026-05-31 | [notebook 14](notebooks/14_compositional_feasibility.ipynb) |
| Stage 7a HiRISE COLOR.JP2 fetch | ✓ 37 of 39 v2 ObsIds (94.9 %) | [DECISIONS.md](DECISIONS.md) 2026-05-31 night |
| Stage 7b reprojection cache | ✗ skipped (folded into 7c via "stay in source CRS") | [DECISIONS.md](DECISIONS.md) 2026-05-31 night |
| Stage 7c per-tile colour features | ✓ 9 860 rows / 36 images, 2026-06-01 | `dataset_v2/features_colour.parquet` |
| Stage 7d pooled rich-vs-poor + shadow refinement + per-image attribution | ✓ PASS 2026-06-02/03 | [docs/compositional.md §4](docs/compositional.md), [notebooks 15+16](notebooks/) |
| Provenance disambiguation Tier 1 (terrain context) | ✓ Fisher's exact OR=23 p=0.018 (P2, honest exclusion) | [docs/compositional.md §4.7](docs/compositional.md), [notebook 17](notebooks/17_provenance_disambiguation.ipynb) |
| Provenance disambiguation Tier 2 (Robbins 2012 crater catalog) | ✓ Kruskal-Wallis null (p>0.7); disfavours crater-ejecta-locally-sourced | [docs/compositional.md §4.7](docs/compositional.md), [notebook 17](notebooks/17_provenance_disambiguation.ipynb) |
| Provenance disambiguation Tier 3 (CRISM/HiRISE upstream source comparison) | open — decisive transport-vs-maturity test | [docs/compositional.md §8](docs/compositional.md) |
| Stage 7e (Atwood-Stone & McEwen 2013 dust index + pixel-level shadow masking) | open | [docs/compositional.md §8](docs/compositional.md) |

**366 pytest pass** (fast suite; +A1/striping + EDR-resolver + nuisance-basis tests). ESP_057469_2215 is excluded from Stage 4 / 4b / 5 sweeps
because its polygon bbox straddles a Murray Lab tile boundary (see
[DECISIONS.md](DECISIONS.md) 2026-05-22 entry). ESP_046803_2325 is in the v2
cohort with COLOR.JP2 + LBL on disk but never had Stage 4 run; it is excluded
from Stage 7 (cohort 36 of 37 colour-eligible).

**Next priorities:** clear the **reopening-call checklist** ([PLAN_FBuild.md](PLAN_FBuild.md) §0):
build-prep part B (H1 centering-statistic stability, `f_h4_buildprep.py`, waiting on free CPU) +
the ESP_053989 recheck under `minnaert_center` + the THEMIS-ρ leg on the leveled pilot map → Brian's
reopening call. If YES → execute **PLAN_FBuild** (the 907-frame regional F build: ≈333 CPU-h ISIS +
~25–40 GPU-h, one to two Sherlock days); if NO → fall back to shipping the A1 map + caveat + H6
provenance. Then the parked validation legs resume on the final map
([PLAN_RegionalMap.md](PLAN_RegionalMap.md), 2026-07-13 refresh note). Parked: Stage-7 Tier 3,
Path A model bank. (Live session state = the `project_state_*` memory notes;
`HANDOFF_NEXT_SESSION.md` is stale.)

## Setup

```powershell
# uses the existing `geospatial` conda env (GDAL, rasterio, geopandas, pyproj, shapely,
# scikit-image, scikit-learn, pyarrow)
& "C:\Users\brian\anaconda3\Scripts\conda.exe" run -n geospatial pip install -e .
```

`conda` is not on PATH in fresh shells on this machine — invoke `conda.exe` by absolute
path or use the snippet above. Direct invocation of the env's `python.exe` fails with
exit code 127 (env DLLs not on PATH without activation); always go through `conda run`.

## Run

```powershell
$conda = "C:\Users\brian\anaconda3\Scripts\conda.exe"

# All tests (fast unit + slow integration; ~70 s total).
# Slow tests auto-skip when their cache prerequisites are missing.
& $conda run -n geospatial pytest tests/ -v

# Fast unit tests only.
& $conda run -n geospatial pytest tests/ -m "not slow" -v

# ---- Stages 2-5 drivers (each one is independently re-runnable; later stages
#      auto-skip ObsIds whose earlier-stage caches are missing) ----

# Stage 2: CTX windowed retrieval. First call per Murray Lab tile downloads ~1.5 GB;
# first call per ObsId downloads the HiRISE JP2 (~200-500 MB). Subsequent calls reuse.
& $conda run -n geospatial python scripts/run_stage2.py ESP_069669_2220
& $conda run -n geospatial python scripts/sweep_stage2.py            # full manifest

# Stage 3: co-registration. Needs Stage 2 caches.
& $conda run -n geospatial python scripts/run_stage3.py ESP_069669_2220
& $conda run -n geospatial python scripts/run_stage3.py --all

# Stage 4: label generation on the nested x2 grid. Needs Stage 2 + 3. Cheap (~3s/ObsId).
& $conda run -n geospatial python scripts/run_stage4.py ESP_069669_2220
& $conda run -n geospatial python scripts/run_stage4.py --all

# Stage 4b: per-tile CTX texture features + bundled context patches. Needs Stage 4.
# ~20-35s per ObsId (GLCM dominates); ~3 min for the 9-image sweep.
& $conda run -n geospatial python scripts/run_stage4b.py ESP_069669_2220
& $conda run -n geospatial python scripts/run_stage4b.py --all

# Stage 5: build named split schemes + materialise per-fold parquets. Needs Stages 4 + 4b.
# ~23s for both schemes on the 9-image manifest.
& $conda run -n geospatial python scripts/run_stage5.py loio_9fold
& $conda run -n geospatial python scripts/run_stage5.py --all

# ---- QA notebooks (one per major stage; all import from src/) ----
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/01_detections_qa.ipynb              # Stage 1 overlay
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/04_ctx_retrieval_qa.ipynb           # Stage 2: window + mask + zooms
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/05_coregistration_qa.ipynb          # Stage 3: shift distribution + before/after
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/06_labeling_qa.ipynb                # Stage 4: heatmaps, target distribution, nested consistency
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/07_features_qa.ipynb                # Stage 4b: per-image heatmaps, correlation matrix, timing
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/08_features_explained.ipynb         # Stage 4b: per-family math/physics/why + stratified patch viewer
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/09_splits_qa.ipynb                  # Stage 5: fold composition + train-vs-test target dist + group-leak check

# ---- Modeling notebooks (10 = v1, 11 = v2, 12 = compression diagnosis,
#                          13 = per-image heterogeneity) ----
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/10_modeling_qa.ipynb                # v1 priority10 modeling QA (frozen baseline)
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/11_modeling_qa_v2.ipynb             # v2 vClaire modeling QA
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/12_compression_diagnostic.ipynb     # compression diagnosis + 4 hurdle variants + boulder_count target
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/13_per_image_heterogeneity.ipynb    # H3 deep dive + top-K confusion overlay

# ---- Foundation-model notebooks (20 = Fang-ViT probe, 21 = deployable head + map pilot) ----
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/20_fang_vit_probe.ipynb             # frozen-embedding probe verdicts + per-image dAUC
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/21_map_pilot.ipynb                  # deployable head + off-HiRISE map (rebuild via notebooks/_build_21.py)
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/22_freeze_and_tier2.ipynb           # head bake-off -> freeze -> Tier-2 (rebuild via notebooks/_build_22.py)
& $conda run -n geospatial jupyter nbconvert --to notebook --execute --inplace `
    notebooks/23_calibration_diagnostic.ipynb     # compression diagnosis + de-compression preview (rebuild via notebooks/_build_23.py; see PLAN_Calibration.md)
```

QA notebooks are generated from `notebooks/_build_NN.py` (re-run the builder to
regenerate the `.ipynb`, then `nbconvert --execute` to render figures).

**Don't run two `nbconvert --execute` against the same notebook concurrently** — caused
~14 min hangs in earlier sessions when overlapping kernels contended for caches.

## Running a parameter sweep

The cheap stages (4 and 4b) are deliberately separated from the expensive ones (2 and
3) so a config sweep doesn't re-run any download or co-registration. Per CLAUDE.md
acceptance #4:

- Changing `labeling.binary_area_threshold` / `binary_count_threshold` / `categorical_bins`:
  Stage 4 re-derives derived label columns from cached base stats in **milliseconds per ObsId**.
  Re-run `scripts/run_stage4.py --all` (~3 s/ObsId).
- Changing `features.glcm.*` / `features.shadow_fraction.*` / `features.enabled`:
  Stage 4b re-extracts features from cached CTX windows. Re-run
  `scripts/run_stage4b.py --all` (~3 min for the full sweep).
- Changing `splits.*`: Stage 5 re-builds split metadata + per-fold parquets. Re-run
  `scripts/run_stage5.py --all` (~25 s for both schemes).

Adding a new manifest row + its BoulderNet detections folder requires no code changes;
re-run Stage 1 → 5 in order (each stage's `--all` skips already-cached ObsIds).

## Modeling sweeps

The three sweep drivers fan out the GBM variants / binary targets / within-image
diagnostic over all four tile scales and write per-fold + aggregate artifacts under
`models/`. They default to the v1 `dataset/` and the `loio_9fold` scheme; pass
`--dataset-dir` / `--scheme` to run the A/B on the versioned `dataset_v2/`:

```powershell
# v1 (priority10):
& $conda run -n geospatial python scripts/sweep.py
& $conda run -n geospatial python scripts/sweep_binary.py
& $conda run -n geospatial python scripts/sweep_within_image.py

# v2 (vClaire): LOIO scheme is loio_nfold; within-image scheme name is unchanged.
& $conda run -n geospatial python scripts/sweep.py             --dataset-dir dataset_v2 --scheme loio_nfold
& $conda run -n geospatial python scripts/sweep_binary.py      --dataset-dir dataset_v2 --scheme loio_nfold
& $conda run -n geospatial python scripts/sweep_within_image.py --dataset-dir dataset_v2
```

The `scheme` + `dataset_dir` enter each run's `config_hash`, so v1 and v2 artifacts land
in distinct `models/<variant>/<hash>/` dirs and never clobber each other.

## Foundation-model embeddings (PLAN_FM)

The frozen recipe (DECISIONS.md 2026-06-12 "Freeze window CLOSED") replaces the
handcrafted features with **frozen Fang-ViT embeddings**: `mlp_ens3` (3-seed MLP) on
the **S=32 96-px 3×3-context GeM(p=3)** 768-dim embedding, emb-only, target
`fa_gt_1e-2`. `src/fm_embeddings.py` is the productized extraction + inference path
(the torch half); `src/modeling/loaders.py` carries the numpy-only cached-store join.

The model is the Fang et al. 2026 ViT-B/16 (MAE+DINO on the Murray Lab CTX mosaic,
[Zenodo 18180801](https://doi.org/10.5281/zenodo.18180801)); the checkpoint
`models/pretrained/mars-mae-dino-vit-base-v1.pth` (341 MB) is **untracked** —
re-download from Zenodo if absent.

```python
import numpy as np
from src.fm_embeddings import FangEmbedder, tile_grid_for_window

# --- inference on an arbitrary CTX window (the map-pilot path) ---
emb = FangEmbedder.load()                       # strict checkpoint load; GPU if present
ti, tj = tile_grid_for_window(window.shape, row0, col0, tile_px=32)
vecs, valid = emb.embed_window(window, ti, tj, tile_px=32, row0=row0, col0=col0)
# vecs: (n_tiles, 768) GeM, NaN rows where the 96-px context spilled past the edge

# --- training join: rebuild a packaged fold's X from the cached store ---
from src.modeling.loaders import load_fold, augment_fold_with_fang
fold = load_fold("loio_nfold", 0, scale_idx=2, dataset_dir="dataset_v2")  # S=32
fold = augment_fold_with_fang(fold, px=96, dataset_dir="dataset_v2", replace=True)  # emb-only
```

The probe-tier extraction over the 38 v2 images (writes
`dataset_v2/fang_embeddings/{obs}_P{32,96}.npz`) still lives at
`scripts/probes/_w2_fang_embed.py --tile-px 32`; `scripts/probes/_fm_parity_check.py`
asserts the productized `src/` path reproduces that cached store bit-for-bit.

### Deployable head + off-HiRISE map (PLAN_FM §2.6)

The frozen recipe is validated under LOIO (a fresh head per fold); a *map* needs ONE
head trained on all images. `src/modeling/mlp_head.py` productizes it: `DeployableHead`
(the 3-seed `mlp_ens3`) trains on the full cohort, persists 3 seed state-dicts + feature
scalers + a recipe card, and exposes `load`/`predict(emb)→prob`. `src/mapping.py` carries
the off-HiRISE inference glue (windowed CTX read → tile grid → predict → 160 m GeoTIFF).

```powershell
# Train the deployable head on ALL images -> models/deployable/<recipe_hash>/
& $conda run -n geospatial python scripts/train_deployable_head.py

# Map pilot: predict rich/poor on a CTX region BEYOND HiRISE coverage. Windows a cohort
# tile away from its footprint (reuses a cached tile zip; no download). Writes a GeoTIFF
# (160 m, Mars CRS) + a 3-panel PNG to reports/.
& $conda run -n geospatial python scripts/map_pilot.py --obs-id ESP_055253_2245 --win-px 3000
```

```python
from src.fm_embeddings import FangEmbedder
from src.mapping import predict_window, read_tile_window
from src.modeling.mlp_head import DeployableHead

head = DeployableHead.load("models/deployable/<recipe_hash>")   # recipe card self-describes
window = read_tile_window(zip_path, inner_tif, row_off, col_off, size=3000)
pred = predict_window(window, FangEmbedder.load(), head, tile_px=32)  # .raster, .transform
```

Grid note: tiles are anchored to the **parent Murray-tile** pixel origin, so `(ti, tj)`
are unique within a tile; `tile_origin_transform` rebuilds that origin for georeferencing
(passing the window affine directly double-counts the read offset). Cross-tile scale-out
additionally keys placement on the Murray-tile id.

### Regional map (PLAN_RegionalMap) + striping artifact (PLAN_StripingArtifact)

```powershell
# Regional map: sweep whole Murray tiles -> per-tile {prob,abundance,prob_raw}.tif (160 m).
# Resumable at (tile, read-window) granularity; built for the Sherlock job array (SHERLOCK_RUN.md).
& $conda run -n geospatial python scripts/map_region.py --tiles E8_N44   # one tile
& $conda run -n geospatial python scripts/map_region.py --all            # the 26-tile block
# Notebook 24 stitches the mosaic + validation legs (MOLA / THEMIS); notebook 25 = striping analysis.

# Striping artifact = CTX SOURCE-FRAME radiometry (the rectangular blocks). Analysis (no inference):
& $conda run -n geospatial python scripts/striping_frame_blocks.py       # eta^2 + choropleth proof
& $conda run -n geospatial python scripts/striping_frame_radiometry.py   # per-frame DN decomposition

# A1 mitigation prototype: per-frame robust offset+gain CTX normalization before the embedder.
& $conda run -n geospatial python scripts/probes/_w2_fang_embed.py --tile-px 32 --norm a1 --out-suffix _a1
& $conda run -n geospatial python scripts/train_deployable_head.py --store-name fang_embeddings_a1 --out models/deployable_a1
& $conda run -n geospatial python scripts/striping_a1_loio.py            # skill gate (baseline vs A1)
& $conda run -n geospatial python scripts/striping_a1_infer_crop.py      # eta^2 payoff on a crop

# F de-risk (per-source-frame inference): build + URL-verify the 10-frame timing list (laptop),
# then run the ISIS timing test on Sherlock (SHERLOCK_RUN.md Part E; setup_isis_env.sh once).
& $conda run -n geospatial python scripts/f_edr_frame_list.py --verify   # -> reports/f_timing/frame_list.csv

# F pilot (after a KEEP_CUBES=1 timing run + f_pilot_extract_crop.py on Sherlock brought the
# 7 calibrated I/F crops home to reports/f_timing/pilot_crops/):
& $conda run -n geospatial python scripts/f_pilot_ifcheck.py             # A0: I/F consistency (CPU)
& $conda run -n geospatial python scripts/f_pilot_crop.py                # leg A: eta^2, 4 mappings (GPU)
```

The Murray Lab **SeamMap** (per-pixel source-frame partition) is pulled from the remote tile zip via
`/vsizip/vsicurl/` range requests (no GB download) and cached as `cache/ctx_tiles/_frames_<tile>.gpkg`
by `src.striping.load_frames`. All striping/A1 logic lives in `src/striping.py`. CTX **EDR**
URLs resolve from SeamMap `VOLUME_ID`+`PRODUCT_ID` alone via `src/ctx_edr.py` (the cached
`PDS_IMG` links are stale; DECISIONS 2026-07-02).

## Layout

```
src/
  config.py          # load/validate YAML; SHA256 config hash for provenance
  manifest.py        # read hirise_priority10.csv; resolve per-ObsId shapefile
  pds_labels.py      # fetch + cache + parse HiRISE .LBL (authoritative metadata)
  detections.py      # Stage 1: read shapefile (auto-corrects buggy `D_unnamed` .prj via
                     # PDS LBL), reproject, cache GPKG
  ctx_tiles.py       # manifest <-> Murray Lab tile-name translator
  ctx_retrieve.py    # Stage 2: download tile zip, window + write CTX GeoTIFF,
                     # warp HiRISE -> CTX grid to build coverage mask
  hirise_imagery.py  # JP2 cache + decimated read helpers (auto-applies the SP1
                     # corrected CRS from Stage 1 sidecars)
  coregister.py      # Stage 3: warp HiRISE onto CTX grid, pick a power-of-2 FFT
                     # window, sub-pixel phase-correlate, cache (dx, dy) per ObsId
  labeling.py        # Stage 4: nested x2 grid anchored to CTX mosaic pixel origin;
                     # 5x sub-pixel polygon rasterisation; sum-up x2 ladder; all
                     # label transforms emitted regardless of `labeling.label_type`
  features.py        # Stage 4b: 9 feature families per-tile (intensity_stats, glcm,
                     # gradient, shadow_fraction, lbp, lacunarity, subtile_variance,
                     # canny_edges) + bundled context patches per (ObsId, patch_size)
  dataset.py         # Stage 5: leave-image-out split construction + in-memory
                     # package_split + streaming iter_train_batches/iter_test_batches
  fm_embeddings.py   # PLAN_FM: frozen Fang-ViT extraction + CTX-window inference path
                     # (ViT-B/16 encoder, GeM(p=3) pool, 3x3-context slicing) -- torch half
  qa.py              # shared sanity-check helpers
scripts/
  run_stage2.py        # headless per-ObsId Stage 2 driver
  sweep_stage2.py      # full-manifest Stage 2 sweep
  run_stage3.py        # Stage 3 driver (single ObsId or --all)
  run_stage4.py        # Stage 4 driver (single ObsId or --all; excludes ESP_057469_2215)
  run_stage4b.py       # Stage 4b driver (single ObsId or --all)
  run_stage5.py        # Stage 5 driver (named scheme or --all)
  probes/              # throwaway debug scripts (see probes/README.md)
tests/                 # 125 tests; integration tests skip until caches exist
notebooks/
  01_detections_qa.ipynb                  # Stage 1 overlay
  02_investigate_misplaced_detections.ipynb  # the SP1 bug, before the fix
  03_hirise_overlay.ipynb                 # decimated HiRISE imagery overlay
  04_ctx_retrieval_qa.ipynb               # Stage 2: window + mask + zooms
  05_coregistration_qa.ipynb              # Stage 3: shift distribution + before/after
  06_labeling_qa.ipynb                    # Stage 4: heatmaps + nested consistency
  07_features_qa.ipynb                    # Stage 4b: cross-image QA + correlation matrix
  08_features_explained.ipynb             # Stage 4b: per-family math/physics + stratified patch viewer
  09_splits_qa.ipynb                      # Stage 5: fold composition + group-leak check
  _build_*.py                             # source-of-truth Python builders for notebooks 07/08/09
cache/                # (gitignored) regenerable artifacts
  pds_labels/                  # PDS .LBL text files (~10-20 KB each)
  reprojected_detections/      # Stage 1: per-ObsId GPKG + provenance JSON
  ctx_tiles/                   # Stage 2: Murray Lab zipped tiles + JSON sidecar
  ctx_windows/                 # Stage 2: per-ObsId CTX GeoTIFF + HiRISE coverage mask
  hirise_jp2/                  # cached HiRISE JP2s (~200-500 MB each)
  hirise_decimated/            # 5 mpp HiRISE GeoTIFFs for co-registration
  coregistration/              # Stage 3: per-ObsId shift JSON
dataset/              # (gitignored except DATA_DICTIONARY.md)
  DATA_DICTIONARY.md           # schema reference for every artifact below
  labels/                      # Stage 4: per-ObsId parquet + JSON sidecar
  features/                    # Stage 4b: per-ObsId parquet + JSON sidecar
  context_patches/             # Stage 4b: bundled (n, P, P) uint8 stacks per (ObsId, P)
  splits/                      # Stage 5: per-scheme split metadata JSON
  packaged/                    # Stage 5: per-(scheme, fold) X/y parquets + groups.npy + all.parquet
reports/figures/     # PNGs from QA notebooks (committed so visuals persist)
config.yaml          # single source of truth for pipeline parameters
CLAUDE.md            # spec (authoritative)
DECISIONS.md         # runtime-verified facts and deviations
ROADMAP.md           # phase-by-phase index of planning + status
PLAN_Stage4b.md      # Stage 4b architecture plan (shipped)
PLAN_Stage5.md       # Stage 5 architecture plan (shipped)
PLAN_modeling.md     # Week 3 modeling plan
```

## How to grow the dataset

Adding a new image is two steps and zero code changes:

1. Add a row to `hirise_priority10.csv` with at minimum `ObsId`, `ProductId`,
   `BoulderLabel`, `CenterLat`, `CenterLon_180`, `CenterLon_360`, `CTX_TileName`,
   `JP2_URL`, `LabelURL`, and the other URL columns.
2. Drop the BoulderNet detections folder under `detections_root/{ObsId}/` containing
   a `*-mask-nms.shp` (with sidecar `.prj`, `.dbf`, `.shx`).

Then run the pipeline in order. **Start at Stage 1** — a genuinely new image has no
`reprojected_detections` cache, and Stage 2 reads it (`load_reprojected`); skipping
Stage 1 fails there:

```powershell
& $conda run -n geospatial python scripts/run_stage1.py {new ObsId}   # reproject + SP1 fix
& $conda run -n geospatial python scripts/run_stage2.py {new ObsId}
& $conda run -n geospatial python scripts/run_stage3.py {new ObsId}
& $conda run -n geospatial python scripts/run_stage4.py {new ObsId}
& $conda run -n geospatial python scripts/run_stage4b.py {new ObsId}
& $conda run -n geospatial python scripts/run_stage5.py --all   # re-build all schemes
```

Each `--all` driver skips ObsIds whose caches already exist (Stage 2/3/4/4b) and
re-derives split assignments deterministically (Stage 5).

### A/B on a second detection set (the vClaire v2 dataset)

To build a parallel dataset on a different BoulderNet run without touching the v1
`dataset/`, point every stage at a second config (`config_v2.yaml`, which sets its own
`manifest` / `detections_root` / `cache_dir` / `output_dir`). Every stage driver takes
`--config`:

```powershell
& $conda run -n geospatial python scripts/run_stage1.py  --all --config config_v2.yaml
& $conda run -n geospatial python scripts/sweep_stage2.py      --config config_v2.yaml   # run_stage2.py is single-ObsId
& $conda run -n geospatial python scripts/run_stage3.py  --all --config config_v2.yaml
& $conda run -n geospatial python scripts/run_stage4.py  --all --config config_v2.yaml
& $conda run -n geospatial python scripts/run_stage4b.py --all --config config_v2.yaml
& $conda run -n geospatial python scripts/run_stage5.py  --all --config config_v2.yaml
```

The imagery caches (`ctx_tiles`, `hirise_jp2`, `hirise_decimated`, `pds_labels`) are
shared via Windows junctions so they aren't re-downloaded; the detection-derived caches
(`reprojected_detections`, `ctx_windows`, `coregistration`) stay separate. Then model on
the versioned dataset by passing `--dataset-dir`/`--scheme` to the sweep drivers (see
"Running a parameter sweep" below). See [PLAN_NewDetections.md](PLAN_NewDetections.md)
for the full A/B design.

## Gotchas to read first

These have all bitten us once already and are worth knowing before you touch the code:

- **HiRISE `.prj` / JP2 SP1 bug** ([DECISIONS.md](DECISIONS.md) 2026-05-20, 2026-05-22):
  4 of 10 BoulderNet shapefiles ship with `Standard_Parallel_1 = 0` (datum
  `D_unnamed`) even though their geometry was generated with the PDS-declared
  projection latitude. The matching JP2s inherit the same buggy metadata.
  `src/detections.py` corrects the shapefile side; `src/hirise_imagery.py` applies
  the symmetric correction at JP2 read time, replacing the JP2's embedded CRS with
  the Stage 1 sidecar's corrected CRS. Caches built before the JP2-side fix are
  detected via a literal-SP1 comparison and rebuilt automatically (pyproj's
  `.equals()` canonicalizes spherical Equirectangular without SP1, so the literal
  parse is necessary). Don't "fix" either override thinking it's wrong — both are
  auto-correcting the same upstream issue.
- **Murray Lab tile URL form** ([DECISIONS.md](DECISIONS.md) 2026-05-21/22): the URL
  uses a signed-prefix zero-padded form universally. Positive longitudes get
  `E<abs:03d>` (e.g. `E000`, `E012`, `E160`); negative longitudes get `E-<abs:03d>`
  (e.g. `E-040`, NOT `W040`); latitudes follow the same rule at 2 digits (`N20`,
  `N-08`). `ctx_retrieve.ensure_tile_cached` tries the bare `manifest_to_murray`
  output first, then falls back to this canonical form on 404.
- **CTX mosaic CRS surprise**: the actual Murray Lab CRS is
  `Mars_2015_Ocentric_Equirectangular` with inverse flattening 169.894 (oblate), not the
  IAU 2000 sphere we configured for `target_crs`. Sub-pixel discrepancy at 5 m/px so not
  blocking; revisit if Stage 3 phase correlation shows a systematic equator-to-pole bias
  (it doesn't, as of the 2026-05-22 sweep).
- **HiRISE-coverage constraint** ([DECISIONS.md](DECISIONS.md) 2026-05-21): Stage 4 drops
  any tile where the cached `{ObsId}_hirise_mask.tif` isn't fully 1. The polygon-bbox
  CTX window contains ~40% HiRISE-unobserved area on average; counting those as "no
  boulders" would inflate zero-tile counts (it's "unobserved", not "absent"). The strict
  `coverage == 1.0` eligibility rule was chosen 2026-05-23 over a relaxed `>= 0.95`
  because partial coverage biases `fractional_area` low.
- **`/vsicurl/` is 140× slower than bulk download** for HiRISE JP2s on this network.
  Stage 2 uses `download_then_window` mode; don't switch to `/vsicurl/` without
  re-benchmarking. Disk usage of cached tiles + JP2s (~12 GB if all 10 ObsIds get
  fetched) is acceptable.
- **Source TIFF block size**: Murray Lab tiles have a full-raster internal block
  (47420×47420). Naively copying `src.profile` into a small windowed output triggers
  `_TIFFVSetField: Bad value 47420 for "TileWidth"`. Drop `blockxsize`/`blockysize`
  from the profile before writing.
- **Stage 4b context-patch layout deviates from PLAN_Stage4b.md §6**: patches are
  bundled per (ObsId, patch_size) into a single `.npy` stack, not per-tile
  `{ti}_{tj}.npy` files. The literal per-tile layout would produce ~1.3M small files
  (NTFS-hostile). 18 bundled files mmap-load fine for the CNN DataLoader path;
  `patch_idx_S{32,64}` columns in the feature parquet point into the stack.
- **Stage 5 splits are over images, never tiles** ([CLAUDE.md](CLAUDE.md) §4 acceptance
  #5). Tiles within an image share illumination, surface composition, and BoulderNet
  detector behaviour, so a random per-tile split leaks the per-image background into
  the test fold. Notebook 09 has an explicit group-leak assertion cell that fails the
  notebook execution if any `obs_id` appears in both train and test for any fold.
