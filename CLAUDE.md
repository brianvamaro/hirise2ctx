# HiRISE → CTX Rock Abundance Pipeline — Claude Code Handoff

**Scope of this document:** Weeks 1–2 of the project — build the data pipeline that turns
BoulderNet detections (already produced) + retrieved CTX imagery into a **regression-ready,
config-driven, re-runnable paired dataset**. Model training, validation, and compositional
analysis are **out of scope here** and captured as future-work notes at the end so they aren't
forgotten.

Treat this file as the build spec. Where it says **VERIFY AT RUNTIME**, do not hardcode — read
the value from the data or confirm against the source, then record what you found.

---

## 1. Goal (one paragraph)

BoulderNet has already detected meter-scale boulders on ~50 cm/px HiRISE images and saved them as
georeferenced polygon shapefiles. We want to learn a mapping from 5 m/px **CTX** texture/shadow
signatures → a per-tile **rock abundance** label (continuous fractional boulder area, or a
configurable alternative), so the model can later predict abundance across the near-global CTX
mosaic where HiRISE coverage is absent. This handoff builds everything up to and including the
paired training dataset. It does **not** train the model.

---

## 2. Environment

- OS: **Windows**. Claude Code runs locally on the same machine, with network access.
- Use the existing conda env **`geospatial`** (expected: GDAL, rasterio, geopandas, shapely,
  pyproj, scikit-image, scikit-learn, numpy, pandas; add anything missing into that env).
- Assume nothing else is local: **only the boulder-prediction shapefiles exist on disk.** HiRISE
  imagery, CTX tiles, and THEMIS data must be fetched.

---

## 3. Inputs

### 3.1 Manifest (drives everything)
`hirise_priority10.csv` is the **manifest**. The pipeline must be manifest-driven: adding rows +
their detection folders is the only thing required to grow the dataset. Do not hardcode the 10
images anywhere. Relevant columns: `ObsId`, `ProductId` (= `ObsId` + `_RED`), `BoulderLabel`
(`Boulder rich` / `Boulder poor` / `unknown` — a per-IMAGE qualitative tag, not a per-tile label),
`CenterLat`, `CenterLon_360`, `CenterLon_180`, `IncidenceAngle`/`EmissionAngle` (present only for
the diversity picks — backfill the rest from the HiRISE `.LBL` if illumination geometry is used as
a feature), `CTX_TileName` (4°×4° Murray Lab tile; note **`E000_N40` covers 3 images**, so dedupe
retrieval by tile), and the `BrowseURL` / `JP2_URL` / `LabelURL` for retrieval.

Current set: 6 boulder-rich (tightly clustered ~40–46°N, 0–20°E), 2 boulder-poor, 2 unknown
diversity picks. Expect the set to grow and to become geographically broader.

### 3.2 Boulder detections (local)
Per-image folders, e.g.:
```
{DETECTIONS_ROOT}\{ObsId}\{ProductId}-predictions-ct-010-ss-512-is-1024-ov-020-mask-nms.shp
```
- `DETECTIONS_ROOT` is a config value (the example machine uses
  `C:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise_priority10_detections`).
- **Discover the shapefile by glob** (`{ObsId}/*-mask-nms.shp`); the BoulderNet param suffix
  (`ct-010-ss-512-...`) may differ on future re-runs.
- Each shapefile has a sidecar `.prj`, `.dbf`, `.shx`. **Inspect the `.dbf` columns at runtime**
  and report them — if confidence and/or size attributes exist, expose them as label-gen filters.

### 3.3 CRS — READ THIS, it is the main gotcha
The detections are georeferenced (units = meters), but in a HiRISE-derived **equirectangular**
projection (`Equidistant_Cylindrical`, central meridian 180°) **on a sphere whose radius is the
local Mars radius at that image's center latitude** (the example image's `.prj` gives
`3393833.26 m`, not the standard equatorial `3396190 m`). Consequences:

- The **sphere radius almost certainly differs from image to image**, and differs from the CTX
  mosaic's CRS. **Read each shapefile's own `.prj`** and reproject per-image into a single common
  target CRS. **Never assume a shared datum and never hardcode a radius.**
- Target CRS = the CTX mosaic's CRS, **VERIFY AT RUNTIME** by reading it from a downloaded CTX tile
  (GDAL exposes it). Reproject both sides into that.
- This CRS mismatch is **separate** from the ~200 m CTX mosaic registration error. Built-in sanity
  check: after correct reprojection, the residual HiRISE↔CTX offset should be **O(200 m)**. If it
  comes out in kilometers, the CRS handling is wrong — fail loudly.

### 3.4 To be retrieved
- **HiRISE RED** (`JP2_URL`) + **`.LBL`** (`LabelURL`): needed only for co-registration. Read
  **decimated** (~5 m/px) via GDAL overviews / windowed reads — do **not** materialize full res.
- **Murray Lab Global CTX Mosaic** 4°×4° tiles: huge (~GB each). Prefer **windowed reads around
  each HiRISE footprint + buffer** (try `/vsicurl/` range requests first; fall back to
  download-then-window). A HiRISE footprint is small (~6 km wide), so never load a whole tile.
- **THEMIS** rock-abundance / bedrock map: **future work only**, do not fetch in this phase.

---

## 4. Architecture — staged, cached, config-driven

Per-image stages are **independent and parallelizable**, keyed by `ObsId`. Expensive stages cache
their outputs so the cheap label step can be re-run repeatedly with different parameters.

```
Stage 0  Load manifest + config
Stage 1  Per-image: ingest detections, read each .prj, reproject to common CTX CRS   [cache]
Stage 2  Per-image: retrieve CTX window around footprint (+buffer)                   [cache]
Stage 3  Per-image: (OPTIONAL) co-registration → (dx, dy) shift                      [cache]
Stage 4  Label generation (CHEAP, RE-RUNNABLE) → paired dataset
Stage 5  Package dataset + split metadata
```

### Stage 3 — co-registration (optional refinement, off by default)
Build the dataset on **nominal geolocation first**. Co-registration is a separable, optional
refinement that only adjusts the per-image grid anchor:
- Decimate HiRISE RED to ~CTX scale, **sub-pixel phase-correlate** against the co-located CTX
  window (power-of-2 FFT window), solve a **rigid translation `(dx, dy)`** per image. Full warp is
  a documented fallback, not built now.
- Cache the per-image shift. Acceptance: |shift| ~ O(200 m); flag outliers (e.g. bland plains with
  weak correlation) rather than trusting them silently.

### Stage 4 — label generation (the part that gets re-run a lot)
This must be **fast, idempotent, and fully parameterized** so we can sweep settings without
re-downloading or re-co-registering.
- **Grid:** anchored to the **CTX mosaic native pixel origin** (not the HiRISE footprint) so each
  label tile is an integer block of CTX pixels — no resampling, reproducible across runs.
- **Tile sizes:** specified in **CTX pixels on a ×2 ladder** (e.g. `[8, 16, 32, 64]` px =
  40/80/160/320 m; report the meter equivalent). Because sizes are nested, **compute base stats on
  the finest grid once and sum upward** to get all coarser scales for free and exactly nested.
- **Base per-tile stats (computed once):** `boulder_area` (Σ polygon area within tile),
  `boulder_count`, `tile_area`. Optionally size-distribution moments if `.dbf` carries size.
- **`label_type` transform (applied last, cheap, swappable):**
  - `fractional_area` → `boulder_area / tile_area` (continuous regression target) — primary.
  - `binary` → boulder-rich vs boulder-poor per tile. **Keep both rules available:** emit
    `binary_by_area` (`fractional_area >= area_threshold`) and `binary_by_count`
    (`boulder_count >= count_threshold`) as separate columns, each with its own configurable
    threshold, so neither rule has to be committed to up front.
  - `count` / `density` → counts or count/area.
  - `categorical` → binned `fractional_area` (configurable bin edges).
- **Detection filters (if attributes exist):** min confidence, min boulder size — config values.
- **CTX features/inputs per tile:** intensity stats, GLCM texture, local gradient, a
  shadow-fraction proxy. Optionally also save a **power-of-2 context patch** (the raw grid of CTX
  pixels around the tile, e.g. 32 or 64 px) for later CNN use — **off by default**
  (`context_patch_px: null`); enabling it later re-runs only Stage 4, since the CTX windows are cached.
- **Output:** a tidy table (one row per tile per scale) carrying `ObsId`, grid indices, CTX pixel
  block reference, the **base stats**, the **derived label**, the feature columns, and provenance
  (config hash). Keep base stats in the table so labels can be re-derived without recomputation.
  Optionally emit context-patch arrays alongside.

### Stage 5 — packaging + splits
- Produce **group-aware, leave-image-out** split metadata (never random tile splits — tiles within
  an image are spatially correlated). Stratify so high-abundance images and the 2 boulder-poor
  images are spread across folds. Make the splitter generic for a growing set.

---

## 5. Config (single source of truth, e.g. YAML)
```yaml
manifest: hirise_priority10.csv
detections_root: "C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise_priority10_detections"
cache_dir: "./cache"
output_dir: "./dataset"

target_crs: from_ctx_tile      # VERIFY AT RUNTIME, do not hardcode
ctx_read: vsicurl_window       # fallback: download_then_window
hirise_decimation_mpp: 5

coregistration:
  enabled: false               # nominal geolocation first; flip on later
  method: phase_correlation_translation
  search_radius_m: 400
  fft_window_px: 256

labeling:
  grid_anchor: ctx_pixel_origin
  tile_sizes_px: [8, 16, 32, 64]   # ×2 ladder, nested
  label_type: fractional_area      # fractional_area | binary | count | categorical
  binary_area_threshold: 0.005     # used for binary_by_area  (placeholder; tune after seeing data)
  binary_count_threshold: 5        # used for binary_by_count (placeholder; tune after seeing data)
  categorical_bins: []             # set when label_type=categorical
  detection_filters: {min_confidence: null, min_size_m: null}  # null until .dbf confirmed
  context_patch_px: null           # OFF by default; set to 32/64 later to also save raw CTX chips
  features: [intensity_stats, glcm, gradient, shadow_fraction]
```

---

## 6. Acceptance criteria / sanity checks
1. **CRS:** post-reprojection residual HiRISE↔CTX offset is O(200 m), not km. Fail loudly otherwise.
2. **Nested grids:** summing the finest grid up the ×2 ladder reproduces the directly-computed
   coarse-grid stats (consistency test).
3. **Idempotent / reproducible:** re-running label gen with the same config reproduces outputs
   bit-for-bit; config hash recorded in provenance.
4. **Re-runnable sweep:** changing `tile_sizes_px`, `label_type`, or thresholds re-runs Stage 4
   only (no re-download, no re-co-registration), in seconds–minutes.
5. **Extensible:** adding a new manifest row + its detection folder flows end-to-end with no code
   changes.
6. **Per-image independence:** stages 1–3 can run per `ObsId` in isolation and in parallel.
7. **Tests pass:** the `pytest` suite (Section 7) is green, including the nested-grid consistency
   and CRS sanity tests.
8. **QA notebooks render:** each stage's QA notebook runs top-to-bottom and writes its figures to
   `reports/figures/`.

---

## 7. Documentation, QA notebooks & tests

Documentation, notebooks, and tests are **required deliverables, not optional extras** — build them
alongside the code, not at the end. Keep all real logic in importable `src/` modules; notebooks and
tests *call* that code rather than re-implementing it, so nothing important ever lives only in a
notebook.

### Documentation
- **`README.md`** — how to set up the env, run each stage, and run a parameter sweep (exact
  commands). Treat it as the entry point for a new person (or a future you).
- **`DECISIONS.md`** — a running log that records every **VERIFY-AT-RUNTIME** answer as it's found:
  the actual CTX mosaic CRS, the `.dbf` attribute schema, per-image co-registration shifts, chosen
  thresholds, and any deviation from this spec (with the reason and date). This is how the runtime
  unknowns in Section 11 get pinned down permanently.
- **`dataset/DATA_DICTIONARY.md`** — every output column with its meaning, units, and how it was
  derived (especially base stats vs. the derived label, plus the config/provenance fields).
- **Docstrings** on every public function and a one-line purpose header on each module. Reserve
  inline comments for non-obvious rationale (e.g. the per-image local-radius CRS gotcha), not
  narration of obvious code.
- Keep `CLAUDE.md` (this spec) authoritative; when reality diverges, update `DECISIONS.md` and note
  it — don't silently drift.

### QA notebooks (visual verification)
One notebook per major step under `notebooks/`, each importing from `src/`, runnable top-to-bottom,
and saving figures to `reports/figures/` so results persist without re-running. **The `.ipynb` files
are generated — each has a source-of-truth builder `notebooks/_build_NN.py`. Edit the builder, not
the `.ipynb`; regenerate with `python notebooks/_build_NN.py` then `nbconvert --execute --inplace`
(see README "QA notebooks").** Minimum visuals:
- **Detection ingest:** boulder polygons overlaid on the decimated HiRISE image for one ObsId.
- **Reprojection / CRS:** reprojected polygons overlaid on the CTX window, with the measured
  HiRISE↔CTX residual offset annotated (the visual form of acceptance check #1).
- **Co-registration (when enabled):** before/after overlay + the solved `(dx, dy)` shift.
- **Labeling:** the tile grid drawn on the CTX window with per-tile `fractional_area` as a heatmap,
  at two or three tile sizes from the ladder.
- **Target distribution:** histogram of `fractional_area` across all tiles (shows the
  zero-inflation/skew), plus a boulder-rich vs boulder-poor comparison.

Notebooks are for inspection; the pipeline itself must run headless from `src/` without them. (The
VS Code extension can execute notebook cells with a confirmation step, handy for stepping through
these.)

### Tests (`pytest` under `tests/`)
Split fast unit tests (synthetic data, no downloads) from a slower integration test (one real image):
- **Reprojection** — a known point transforms to the expected CTX coordinate; round-trip is stable.
- **Nested-grid consistency** — summing the finest grid up the ×2 ladder equals the directly
  computed coarse-grid stats (synthetic boulders).
- **Label transforms** — `fractional_area` in [0, 1]; `boulder_area`/`boulder_count` ≥ 0; each
  `label_type` yields the expected columns/values on a hand-built tile; `binary_by_area` and
  `binary_by_count` both present.
- **Grid alignment** — tile bounds land on integer CTX pixel blocks anchored to the mosaic origin.
- **Idempotency** — same config in → identical output and identical config hash.
- **Integration (one ObsId, marked slow):** runs Stages 0–1 end to end and asserts the residual
  offset is O(200 m), not km.

---

## 8. Suggested repo layout
```
src/
  config.py          # load/validate config, config hashing
  manifest.py        # read CSV manifest
  detections.py      # glob shapefile, read per-image .prj, reproject (Stage 1)
  ctx_retrieve.py    # windowed CTX reads (Stage 2)
  coregister.py      # phase-correlation translation (Stage 3, optional)
  labeling.py        # nested grid, base stats, label_type transforms (Stage 4)
  features.py        # CTX feature extraction + context patches
  dataset.py         # packaging + leave-image-out splits (Stage 5)
  qa.py              # shared sanity-check helpers used by both tests and notebooks
tests/               # pytest suite (Section 7); unit tests + one slow integration test
notebooks/           # one QA notebook per stage; import from src/, save figures to reports/
reports/figures/     # PNGs written by the QA notebooks (committed so visuals persist)
cache/               # reprojected detections, CTX windows, coreg shifts
dataset/             # paired dataset + DATA_DICTIONARY.md
config.yaml
README.md            # setup + how to run each stage + how to run a sweep
DECISIONS.md         # running log of runtime-verified facts and any spec deviations
CLAUDE.md            # this spec (project memory, auto-read by Claude Code)
```

---

## 9. Known gotchas (carry these into the code)
- Per-image local-radius equirectangular `.prj` — reproject per image, never hardcode (Section 3.3).
- CTX tiles are GB-scale — windowed reads only.
- HiRISE RED must be read decimated, never full-res.
- Target is **heavily zero-inflated and right-skewed**: even "boulder rich" scenes are mostly
  sparse, so low/zero tiles dominate and the high-abundance tail is rare and clustered. This is a
  modeling-stage concern but the labeling output should preserve raw base stats so the modeling
  stage can choose log1p / two-stage / stratified handling later.
- Splits must be by image (group-aware), never random tiles.

---

## 10. Future work — NOT in scope now, documented so it isn't lost

**Model training (Week 3).** Baseline = CTX patch features (intensity/GLCM/gradient/shadow) +
gradient boosting / random forest; optional small CNN on the power-of-2 context patch. Target is
zero-inflated/skewed → consider `log1p`, or a two-stage presence→magnitude model. Evaluate with
**leave-image-out CV** and **stratified metrics across abundance bins** (not a single RMSE
dominated by near-zero tiles).

**Validation against THEMIS.** Fetch the THEMIS rock-abundance / bedrock map (~100 m/px) and
compare predicted abundance where footprints overlap, as an independent coarse-scale check.

**Compositional analysis (instructor's extra goal — updated 2026-05-30).** After a
rock-abundance / binary boulder-rich vs boulder-poor map exists, determine whether boulders
are locally sourced or transported by testing whether the **spectral signatures of
boulder-rich areas differ from their surroundings**, using **the three HiRISE bands**
(BLUE-GREEN, RED, NEAR-IR; [Delamere et al. 2010, *Icarus*](https://doi.org/10.1016/j.icarus.2009.03.012)).
*Originally planned to use CRISM; switched to HiRISE bands 2026-05-30.*

**Limitation:** the signal may be affected by dust. Two cases to disentangle:
1. Dust uniformly obscures any compositional signal → spectra look similar regardless of
   underlying composition.
2. The detected spectral difference itself comes from *differential dust presence* between
   boulder areas and surroundings — in which case it would indicate the **relative age of
   the boulder deposit** (older deposits accumulate more dust), not a compositional
   distinction.

So a finding of "boulders differ from surroundings" requires care to attribute to
composition vs dust loading.

---

## 11. Open items for Claude Code to resolve at runtime
- Exact CTX mosaic CRS (read from a downloaded tile) and whether Murray Lab GeoTIFFs support
  `/vsicurl/` range requests.
- The `.dbf` attribute schema of the BoulderNet shapefiles (confidence? size?) → enables filters.
- HiRISE `.LBL` fields for backfilling incidence/emission angles on the non-diversity images.
