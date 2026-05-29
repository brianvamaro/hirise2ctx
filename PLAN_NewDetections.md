# PLAN — Building the dataset on the new 40-image "vClaire" detection set

**Status:** planned, not yet implemented (updated 2026-05-28). Drafted after
reading CLAUDE.md, README.md, DATA_DICTIONARY.md, DECISIONS.md, and the
Stage 1–5 source; updated after inspecting the first 2 of 40 incoming
detection folders (§1.1).

A new BoulderNet run ("vClaire") covers **40 HiRISE images** and captures
**far more** of the boulders actually present than the priority10 set the
current dataset was built on. The first inspected image carries **1.1 million
polygons** vs ~1–5k per image in the old set. This plan builds a parallel
paired dataset on the vClaire detections, keeping the old 9-image dataset
intact for comparison, and downloading HiRISE JP2s from PDS as before (no
local GeoTIFFs).

The scientific payoff (§9): the old dataset hit an AUC ≈ 0.55 ceiling under
three independent framings (modeling_results.md §1/§6/§7) and the Stage 5c
within-image diagnostic attributed it to the 5 m/px CTX signal floor *given
the labels then available*. A vastly denser, more-complete label set is the
direct test of whether that ceiling is a true signal floor or was partly an
artifact of missed boulders. If the within-image AUC on vClaire still sits at
~0.55, the floor is real; if it lifts, label completeness was a confound.

---

## 0. Execution status (2026-05-28)

All 40 detection folders are present in `hirise_40_vClaire`. Ran
`scripts/build_vclaire_manifest.py` → integrity pre-flight on all 40 + built
`hirise_40_vclaire.csv` (40 rows, all `REQUIRED_COLUMNS`).

**Done:**
- **Manifest built:** `hirise_40_vclaire.csv`. URLs templated from the PDS RDR
  convention (orbit folder `ORB_{orbit//100*100}_{+99}`); all 40 `LabelURL`s
  resolved (LBLs cached). `CTX_TileName` derived by a floor-to-4° rule that was
  first validated to reproduce all 10 existing-manifest tiles, and the 3
  overlap images reproduce v1 (`ESP_047976_2020`→`W040_N20`,
  `ESP_069669_2220`/`ESP_071093_2210`→`E000_N40`). Tiles span ~15 unique
  Murray Lab tiles (≈22 GB of CTX to download).
- **Center coords:** taken from the PDS **footprint midpoint**
  (MIN/MAX_LAT, E/W_LON), cross-checked against the spreadsheet corners
  (all agree < 1°). **Gotcha (record in DECISIONS.md):** `pds_labels.projection_origin`
  returns the map-projection central meridian / standard parallel (rounded,
  e.g. lon 180, lat 45) — correct for the SP1 `.prj` fix but **wrong** for the
  manifest center. Use `pds_labels.image_footprint` midpoints instead.
- **BoulderLabel:** from the spreadsheet "Overall…" column → **37 `Boulder rich`,
  3 `unknown`** (the 3 absent from the spreadsheet). No `Boulder poor` (the
  vClaire set looks curated to boulder-rich targets — confirm).
- **SP1 bug:** 33 of 40 `.prj` carry `D_unnamed`/SP1=0 → Stage 1 will correct
  them via the now-cached LBLs. The 3 overlap images + 4 others are clean.
- **Polygon counts** range 9.6k → **1.1M** (`ESP_017355_2260`); ~10 images
  exceed 200k. Confirms the §3.4 scale concern.

**Blockers / needs your input (see §13):**
- ⚠ **`ESP_028537_2270`** — `.dbf`/`.shp` truncated (read fails); re-copy.
- ⚠ **`ESP_045878_2235`** — the only file present is `…-downscaled-**bbox**-nms.shp`,
  not `…-**mask**-nms.shp`. The glob (correctly) doesn't match it. Need the
  mask-nms version, or a decision to special-case bbox.
- **3 images have no spreadsheet row** (`ESP_017355_2260`, `ESP_028537_2270`,
  `ESP_076499_1160`) → `BoulderLabel='unknown'`; supply labels if you have them.
  Note `ESP_076499_1160` is a **southern −63.7°** image — far outside the
  northern cluster, geographically a strong diversity pick.

**Pipeline progress (2026-05-28):**
- ✅ Code gaps closed: `scripts/run_stage1.py` added; `--config` added to
  run_stage1/2/3/4/4b/5 + sweep_stage2.
- ✅ `config_v2.yaml` (manifest=`hirise_40_vclaire.csv`, cache_v2/dataset_v2,
  `context_patch.enabled=false`). `cache_v2` junctions the shared imagery
  caches (ctx_tiles, hirise_jp2, hirise_decimated, pds_labels).
- ✅ **Stage 1 done — 39/39 reprojected, 0 failures, 32 SP1-corrected.**
  1.1M-row reproject runs in ~16 s; no scale problem.
- 🔎 **Null-geometry finding (DECISIONS.md-worthy):** BoulderNet's dense
  vClaire `*-mask-nms` shapefiles carry many rows with a DBF record but NO
  polygon (745k of 1.1M for ESP_017355_2260; 0 for the priority10 set).
  Hardened `src/detections.py` to drop null/empty geometries at ingest
  (no-op on v1) and record `n_polygons_raw` / `n_dropped_null_geometry` in the
  Stage-1 sidecar. **True boulder counts (non-null)** range 9.6k → 727k.
- 🔎 **Filter decision (§7.1) resolved by data:** reprojected equivalent-circle
  diameters are large (pooled median 3.4 m, p5 ≈ 1.9 m) — **~0% below the
  1.4105 m floor**, so `min_size_m` stays a no-op (keep it). Scores: 100% ≥ 0.2,
  89% ≥ 0.3, 52% ≥ 0.5 — recommend `min_confidence=null` (the boulders are
  real-sized; confidence filtering isn't needed for quality). The "more
  boulders" are *more*, not *smaller*.
- ⏭ **Next = Stage 2** (~15 unique Murray Lab tiles ≈ 22 GB + 39 HiRISE JP2s
  ≈ 10–20 GB). Large/slow download — checkpoint with the user before launching.

The remaining sections are the forward plan (unchanged in shape; the manifest
sub-task §5.2 + the filter decision §7.1 are now done).

---

## 1. Decisions pinned (AskUserQuestion 2026-05-27, updated 2026-05-28)

| Question | Decision | Consequence for this plan |
|---|---|---|
| Image scope | **40-image vClaire set** — the 2 inspected ObsIds (`ESP_017355_2260`, `ESP_028537_2270`) are **not** in the current manifest, so this is a (largely or entirely) **new cohort**, not a denser re-run of the old 10. | Dominant work is now the **full pipeline on ~40 new images** + building a 40-row manifest (§5). The "same-image label-only refresh" sub-path only applies to any ObsIds that overlap the old 10 (TBD — open question §13.1). |
| Detection format | **Same BoulderNet shapefile set**, verified at runtime (§1.1): `*-mask-nms.shp` + `.prj`/`.dbf`/`.shx`/`.cpg`, identical DBF schema, same per-image local-radius equirectangular CRS. Filename suffix differs (`ss-256` + `-downscaled`) but still matches the `*-mask-nms.shp` glob. | No ingest-adapter or glob change. `src/detections.py` + `manifest.find_shapefile` work unchanged. |
| Old vs new | **Keep both for A/B comparison** | Build the vClaire dataset under a versioned namespace (`dataset_v2/`, `cache_v2/`) so the existing `dataset/` is untouched (§4). Because the cohorts differ, the A/B is cohort-level (old 9 vs new 40); the *within-image* diagnostic stays per-image-comparable (§9). |
| HiRISE imagery | **Download JP2s from PDS as before; no GeoTIFFs provided.** | The JP2-URL path (`hirise_imagery.py` + Stage 2 mask) is the only path. The local-GeoTIFF extension is dropped from scope. `JP2_URL` **and** `LabelURL` are required per image (LabelURL also drives the SP1 fix, which fires on this set — §1.1). |

### 1.1 What the directory inspection found (first 2 of 40)

Inspected `C:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise_40_vClaire`
via `scripts/probes/_diag_vclaire_detections.py`:

- **Layout matches** the expected `{root}/{ObsId}/*-mask-nms.shp` + sidecars.
  Folder names are the ObsIds (`ESP_017355_2260`, `ESP_028537_2270`).
- **Glob OK:** filenames end `...-downscaled-mask-nms.shp`; `*-mask-nms.shp`
  still matches. The `ss-256` / `-downscaled` tokens indicate a different
  BoulderNet inference config (256-px slices on downscaled imagery) — record
  this as provenance (§4.3).
- **DBF schema identical:** `score, cat_id, cat_name, isin_slice, is_at_edge,
  id`. `min_confidence` (DBF `score`) is therefore available as a filter; on
  `ESP_017355_2260` score ranges 0.10–0.91, mean 0.51.
- **Polygon counts are enormous:** `ESP_017355_2260` = **1,105,447** polygons.
  `ESP_028537_2270`'s `.shx` implies ~950k. ~750× the old per-image scale —
  see the new §3.5 (scale & performance) and the elevated filter decision (§7.1).
- **SP1 bug present:** `.prj` has `DATUM["D_unnamed"]`, `Standard_Parallel_1=0`,
  `Equirectangular_MARS`. Stage 1's correction will fire on this set, so each
  image **must** have a working `LabelURL` (PDS `.LBL` → `CENTER_LATITUDE`).
- **⚠ `ESP_028537_2270` is truncated/corrupt:** its `.dbf` (59 MB) and `.shp`
  (58 MB) are far too small for the ~950k records its `.shx` (7.6 MB) implies
  (the schema is ~159 bytes/DBF-record, so ~950k records need a ~151 MB `.dbf`);
  `geopandas.read_file` fails with `fread(159) failed on DBF file`. **This
  folder needs to be re-copied** before it can be ingested (open question §13.2).

---

## 2. The one fact that drives the whole plan: what depends on detections

Stages cache their outputs keyed by `ObsId`; **`config_hash` is computed from
`config.yaml` only and does NOT include detection content** (verified in
`src/config.py` + `src/detections.py`). So changing the detection files does
*not* auto-invalidate any cache. We must drive invalidation by hand. The
table below is the authoritative "what must re-run" map.

| Stage | Cache / output | Depends on detections? | Re-run on new detections? |
|---|---|---|---|
| 1 — reproject detections | `cache/reprojected_detections/{ObsId}.gpkg` | **Yes — directly** | **Always.** `stage1_one_image` overwrites unconditionally. |
| 2 — CTX window + HiRISE mask | `cache/ctx_windows/{ObsId}.{tif,_hirise_mask.tif}` | **Window bbox only** (= polygon footprint + buffer; see `ctx_retrieve.compute_window_bounds`). Mask is imagery-only. | **Same images:** reuse — denser detections sit inside the existing HiRISE swath, so the cached window already covers them (verify, §8.3). **New images:** run (may download a new ~1.5 GB Murray Lab tile). |
| 3 — co-registration | `cache/coregistration/{ObsId}.json` | **No** (imagery-only; off by default) | Reuse for same images. Optional for new images. |
| 4 — labels | `dataset/labels/{ObsId}.{parquet,json}` | **Yes — directly** (boulder_area/count from polygons) | **Always.** `run_stage4.py` overwrites; eligibility (mask coverage == 1) is imagery-only so the *eligible tile set is unchanged* for same images — only label *values* change. |
| 4b — features | `dataset/features/{ObsId}.parquet` + context patches | **No** — CTX-texture only; iterates the Stage-4 eligible-tile set | Re-run only if the window/eligible-tile set changed (i.e. new images, or a re-cut window). For same images with a reused window, features are **bit-identical** — but we still regenerate into the versioned output dir (§4) so v2 is self-contained and the A/B is "labels-only differ, features held constant." |
| 5 — splits + packaging | `dataset/splits/*`, `dataset/packaged/*` | Indirectly (labels) | **Always** (cheap). |
| modeling | `models/_sweep*/…` | Indirectly (packaged labels) | **Re-sweep** regression + binary + within-image (§9). |

Net: the minimal correct refresh for the **same images** is
**Stage 1 → 4 → 5 → models**, holding Stage 2/3/4b constant; for **new
images** it is the **full Stage 1 → 5 → models**.

---

## 3. Code gaps to close before any re-run

Three gaps surfaced while tracing the headless path. Fix these first; they are
small and each is independently testable.

### 3.1 No headless Stage 1 driver (blocking)
`stage1_one_image` is only invoked by notebooks 01/02/03 and the test suite.
Both Stage 2 (`ctx_retrieve.py:559`) and Stage 4 (`labeling.py:458`) *consume*
`cache/reprojected_detections/` via `load_reprojected` but never build it. The
original 10 images were reprojected by running notebook 01 by hand.

**Action:** add `scripts/run_stage1.py` (single ObsId + `--all`), mirroring
`scripts/run_stage4.py`. It must pass `manifest_row` into `stage1_one_image`
so the SP1-bug correction (`D_unnamed` datum → PDS-LBL `CENTER_LATITUDE`) and
the PDS `.LBL` fetch happen automatically. Print per-image `n_polygons` and the
`correction.status` so the SP1 outcomes are visible. ~50 LOC.

### 3.2 README "grow the dataset" recipe is missing Stage 1 (latent bug)
README §"How to grow the dataset" lists Stage 2 → 3 → 4 → 4b → 5 for a new
image. A genuinely new image has no `reprojected_detections` cache, so Stage 2
would fail at `load_reprojected`. The recipe only ever worked because the
existing images were reprojected via notebook 01.

**Action:** prepend `scripts/run_stage1.py {ObsId}` to the README recipe (and
the §6b sequence here). Same fix in the in-repo `ROADMAP.md` if it repeats the
recipe.

### 3.3 Modeling sweeps hardcode `dataset/` (blocking for A/B)
`src/modeling/loaders.py::DEFAULT_DATASET_DIR = REPO_ROOT/"dataset"`; the
sweep drivers (`sweep.py`, `sweep_binary.py`, `sweep_within_image.py`) call
`iter_loio_folds(...)` without a `dataset_dir`, so they always read `dataset/`.
To model on the versioned `dataset_v2/` we need an override.

**Action:** add a `--dataset-dir` argument to the three sweep scripts that
threads through to `iter_loio_folds(..., dataset_dir=...)` /
`load_fold(..., dataset_dir=...)` (the loader functions already accept it). The
artifact output root (`models/`) can stay shared — the per-run timestamp dir
already disambiguates, and we tag each sweep's `snapshot` with the dataset
version (§4). ~10 LOC per script.

### 3.4 Scale: ~1M polygons/image × ~40 images is a new performance regime
The old set was ~1–5k polygons/image; vClaire's first image is **1.1M**. Across
40 images this is plausibly **20–40M polygons**. This was never exercised and
needs validation, not assumption, before committing to the full sweep:

- **Stage 1 (reproject + GPKG write):** `gdf.to_crs` is vectorised (fine), but
  `gdf.to_file(..., driver="GPKG")` on 1M+ geometries per image may be slow and
  produces large GPKGs (the source `.shp` alone is ~215 MB). Validate wall-time
  + on-disk size on `ESP_017355_2260` first; if GPKG write dominates, consider a
  more compact cache (e.g. FlatGeobuf/Parquet-WKB) — a `src/detections.py`
  touch, only if measured to matter.
- **Stage 4 (rasterise + centroid count):** `_rasterize_boulders_subpixel`
  rasterises at 5× CTX resolution (C-backed; scales with raster area, not
  polygon count — fine). But `_count_centroids_per_finest_cell` computes a
  shapely centroid per polygon — 1M centroids/image is the likely hotspot.
  Profile Stage 4 on one image before the `--all` run; vectorise the centroid
  step (`gdf.geometry.centroid` is already array-wise) if needed.
- **`boulder_count` semantics at this density:** with ~1M detections in a
  ~6×12 km image, a 40 m (S=8) tile may contain hundreds of polygons. The base
  stats still hold, but `fractional_area` may approach or exceed sane bounds if
  polygons overlap heavily (the rasteriser is union-correct, so `boulder_area`
  is overlap-safe; `boulder_count` is not de-overlapped). Sanity-check the new
  `fractional_area` range in [0, 1] (§8).
- **Run one image end-to-end (Stage 1→4) and measure** before launching the
  40-image sweep. Extrapolate disk + wall-time; revisit the filter (§7.1),
  which is now the main lever on polygon volume.

### 3.5 Disk + download budget for 40 images
The 9-image set used ~10 GB CTX tiles + ~3 GB JP2s + ~3.3 GB patches + ~1.3 GB
packaged. For ~40 geographically spread images:

- **CTX tiles:** each unique 4°×4° Murray Lab tile is ~1.5 GB. 40 images could
  touch 15–40 unique tiles → **~25–60 GB**. (Northern-clustered images share
  tiles, e.g. the `_22xx` band; estimate unique tiles from the manifest's
  `CTX_TileName` before downloading.)
- **JP2s:** ~200–500 MB each × 40 → **~10–20 GB**.
- **Context patches:** scales ~linearly with eligible tiles → **~12–15 GB**.
- **Detection GPKGs:** 1M+ polygons/image → potentially **~5–10 GB**.

Budget **~80–120 GB** for the vClaire dataset on top of the existing v1. Check
free space first; consider whether context patches (CNN-only, currently a
documented dead-end per modeling_results.md §3.3) are worth generating for v2 —
disabling `features.context_patch.enabled` for v2 saves the largest chunk and
is reversible (re-runs only Stage 4b).

---

## 4. A/B isolation design (keep the old dataset intact)

Goal: regenerate the dataset on the new detections without touching the
existing `dataset/` or the existing detection cache, and keep the expensive
**imagery** caches shared so we don't re-download ~12 GB.

Two paths; **start with the zero-code path**, graduate to the config-key
refactor only if A/B becomes recurring.

### 4.1 Zero-code path (recommended to start): second config + shared imagery
1. New detections live in their own root, e.g.
   `…/hirise_priority10_detections_v2/{ObsId}/*-mask-nms.shp`. (Same images get
   their new shapefile here; new images get a folder here too.)
2. Copy `config.yaml` → `config_v2.yaml` and change only:
   - `manifest:` → a v2 manifest CSV (existing rows unchanged + new-image rows; §6b),
   - `detections_root:` → the v2 detections root,
   - `cache_dir:` → `./cache_v2`,
   - `output_dir:` → `./dataset_v2`.
3. Pre-seed `cache_v2/` so the **detection-independent imagery** is shared, not
   re-downloaded. On Windows, directory junctions need no admin:
   ```
   mklink /J cache_v2\ctx_tiles        cache\ctx_tiles
   mklink /J cache_v2\hirise_jp2       cache\hirise_jp2
   mklink /J cache_v2\hirise_decimated cache\hirise_decimated
   mklink /J cache_v2\pds_labels       cache\pds_labels
   ```
   Leave `cache_v2\reprojected_detections`, `cache_v2\ctx_windows`,
   `cache_v2\coregistration` as **real** (fresh) dirs.
   - For the **same images**, also `mklink /J cache_v2\ctx_windows cache\ctx_windows`
     and `…\coregistration` so we reuse the exact v1 windows (guarantees
     features are held constant — the cleanest A/B). **If you junction
     `ctx_windows`, do NOT run Stage 2 under v2 for those images** — it would
     overwrite the shared window with a v2-footprint window and perturb v1.
   - New images can't junction (no v1 counterpart); their Stage 2 writes fresh
     windows into the (real or shared) dir.
4. Every script already takes `config.yaml` as a literal path argument inside
   `load_config("config.yaml")`. **Caveat:** the stage drivers currently
   hardcode `load_config("config.yaml")` rather than taking a `--config` flag.
   So either (a) add a `--config` arg to `run_stage1/2/3/4/4b/5.py` (~3 LOC
   each, recommended), or (b) temporarily swap `config.yaml` for the v2 copy
   during the v2 run. Option (a) is cleaner and avoids a footgun.

### 4.2 Cleaner long-term: a `dataset_version` config key
Add `dataset_version: "v2"` to `config.yaml`; `src/config.py` suffixes the
**detection-derived** cache subdirs (`reprojected_detections`, `ctx_windows`,
`coregistration`) and `output_dir` with the version, while leaving the imagery
caches (`ctx_tiles`, `hirise_jp2`, `hirise_decimated`, `pds_labels`)
unversioned/shared. This removes the junction dance and the `--config` swap.
Larger change (touches every `cache_dir / SUBDIR` construction); do it if we
expect to A/B detection sets routinely.

### 4.3 Provenance (close the config_hash gap)
Because `config_hash` doesn't see detection content, record the new set's
identity explicitly:
- The Stage 1 sidecar already stores `source_path` + `source_mtime_iso` per
  image — that is the per-image anchor.
- Add a one-line `detection_set` identifier to `config_v2.yaml` (e.g.
  `detection_set: "vclaire_40img_ct010_ss256_downscaled_2026-05-28"`) and a
  DECISIONS.md entry recording the BoulderNet run config (the `ss-256` /
  `-downscaled` inference params seen in §1.1), the date received, the 40-image
  ObsId list, and per-image polygon counts.

---

## 5. Inputs needed from you + the 40-row manifest

### 5.1 Detection folders
The 40 detection folders under a single root (the `hirise_40_vClaire` dir is
exactly the right shape), one per ObsId, each with one `*-mask-nms.shp` +
`.prj`/`.dbf`/`.shx`. Re-copy any truncated folder first (`ESP_028537_2270`,
§1.1).

### 5.2 The manifest is the gating sub-task (40 rows)
Every stage is manifest-driven; a 40-image run needs a 40-row CSV with
`manifest.py::REQUIRED_COLUMNS`: `ObsId`, `ProductId` (= `ObsId` + `_RED`),
`BoulderLabel`, `CenterLat`, `CenterLon_360`, `CenterLon_180`, `CTX_TileName`,
`BrowseURL`, `JP2_URL`, `LabelURL`. What can be auto-derived vs what you need
to supply:

| Field | Source | Auto-derivable? |
|---|---|---|
| `ObsId` | the 40 detection folder names | ✅ from the directory |
| `ProductId` | `{ObsId}_RED` | ✅ trivial |
| `CenterLat`, `CenterLon_180/360` | PDS `.LBL` `CENTER_LATITUDE/LONGITUDE` (we fetch the `.LBL` anyway for the SP1 fix) | ✅ once `LabelURL` resolves |
| `CTX_TileName` | `src/ctx_tiles.py` translator from lon/lat | ✅ from CenterLat/Lon |
| `JP2_URL`, `LabelURL`, `BrowseURL` | PDS HiRISE RDR URL convention, templatable from ObsId+orbit | ⚠ templatable but **must be verified to resolve** — see §13.3 |
| `BoulderLabel` | per-image qualitative tag (`Boulder rich`/`poor`/`unknown`) | ❌ **needs your input** (or default all to `unknown`; only affects Stage 5 stratification, not labels) |

**Proposed approach (pending your confirmation, §13.3):** I generate the 40-row
manifest programmatically from the folder list — template the PDS URLs from
each ObsId, fetch each `.LBL` to fill CenterLat/Lon, derive `CTX_TileName`, and
default `BoulderLabel='unknown'`. You then (a) correct any `BoulderLabel`s you
care about and (b) confirm the templated URLs resolve. Alternatively, if you
already have a CSV/spreadsheet with these columns for the 40, hand it over and
I skip the templating.

### 5.3 Imagery
JP2s download from PDS via `JP2_URL` as in v1 — no GeoTIFFs needed. Budget the
download/disk per §3.5.

---

## 6. Stage-by-stage sequence

The 40 vClaire images are a (largely) new cohort → the **full pipeline** per
image. Assumes the A/B namespace (§4.1) with `config_v2.yaml`, the new
`scripts/run_stage1.py`, and `--config`/`--dataset-dir` flags (§3).

### 6.0 Bring up before the full sweep (validate on ONE image first)
Do not launch 40 images blind given the new scale (§3.4). Bring up
`ESP_017355_2260` (the known-good 1.1M-polygon image) end-to-end, measure, then
decide filters:
```powershell
$conda = "C:\Users\brian\anaconda3\Scripts\conda.exe"
# 0. config_v2.yaml (manifest_v2.csv, detections_root=hirise_40_vClaire,
#    cache_dir=./cache_v2, output_dir=./dataset_v2). 40-row manifest per §5.2.
& $conda run -n geospatial python scripts/run_stage1.py ESP_017355_2260 --config config_v2.yaml  # time + GPKG size
& $conda run -n geospatial python scripts/run_stage2.py ESP_017355_2260 --config config_v2.yaml  # CTX tile download
& $conda run -n geospatial python scripts/run_stage4.py ESP_017355_2260 --config config_v2.yaml  # PROFILE (centroid step)
# -> inspect reprojected diameter + score distributions; set detection_filters (§7.1)
# -> confirm fractional_area in [0,1], wall-time + disk extrapolate acceptably (§3.4/§3.5)
```

### 6.1 Full cohort sweep (after the one-image bring-up validates)
```powershell
& $conda run -n geospatial python scripts/run_stage1.py  --all --config config_v2.yaml
& $conda run -n geospatial python scripts/run_stage2.py  --all --config config_v2.yaml   # or sweep_stage2.py; downloads unique CTX tiles
& $conda run -n geospatial python scripts/run_stage3.py  --all --config config_v2.yaml   # optional; off by default
& $conda run -n geospatial python scripts/run_stage4.py  --all --config config_v2.yaml
& $conda run -n geospatial python scripts/run_stage4b.py --all --config config_v2.yaml   # consider context_patch.enabled=false for v2 (§3.5)
& $conda run -n geospatial python scripts/run_stage5.py  --all --config config_v2.yaml
```
Notes:
- `run_stage2.py` currently takes a single ObsId; either add an `--all` mode
  (mirror run_stage4) or use the existing `sweep_stage2.py`. Each unique Murray
  Lab tile downloads ~1.5 GB once and is shared across images on that tile.
- Per-ObsId exclusions are data-driven, not hardcoded for vClaire: an image
  whose `hirise_coverage_fraction` is near-zero (tile-straddle, the
  ESP_057469_2215 failure mode) or whose truth is empty should be flagged by
  the §8 checks and excluded from the relevant schemes, exactly as in v1.

### 6.2 Overlap with the old 10 (if any — open question §13.1)
If some of the 40 ObsIds equal old priority10 ObsIds and you want the cleanest
"labels-only differ" A/B for *those* images, junction their v1 `ctx_windows` +
`coregistration` into `cache_v2` and skip Stage 2/3 for them (so the window and
features are held constant and only the labels change). For the disjoint
majority this doesn't apply — they get fresh everything.

### 6.3 Modeling re-sweep
```powershell
& $conda run -n geospatial python scripts/sweep.py             --dataset-dir dataset_v2
& $conda run -n geospatial python scripts/sweep_binary.py      --dataset-dir dataset_v2
& $conda run -n geospatial python scripts/sweep_within_image.py --dataset-dir dataset_v2
```
Tag each run's snapshot with `dataset_version: v2`. Note the LOIO scheme is now
~40-fold (one per image) — the per-fold standard error shrinks substantially
vs the 9-image v1, which is itself a partial answer to modeling_results.md §5
experiment 2 ("more HiRISE images").

---

## 7. Decision points to resolve at runtime

### 7.1 `detection_filters` — now the central modeling decision, not a footnote
With ~1M raw detections/image at mean `score` 0.51, the filter choice
(`min_confidence` on DBF `score` **and** `min_size_m`) is the main lever on
both polygon volume (perf, §3.4) and label semantics:

- **`min_size_m`** is pinned at `1.4105` (≡ 1.5625 m² ≡ the BoulderNet 0.25 m/px
  design floor; DECISIONS.md 2026-05-26). The vClaire run is on **downscaled**
  imagery (`-downscaled`, `ss-256`), so its native detection scale differs from
  the old `ss-512` run — the size floor must be re-evaluated against the
  *reprojected* (target-CRS metres) diameter distribution, not assumed
  transferable. The source-CRS area came back NaN in the probe (degenerate
  `D_unnamed` ellipsoid), so the size distribution must be measured **after**
  Stage 1 reprojects to the target sphere.
- **`min_confidence`** was `null` in v1 (the old set was already sparse). At 1M
  detections/image with mean score 0.51, a confidence floor may now be the
  right way to suppress low-quality detections and cut volume. v1 left it null;
  v2 should reconsider.

**Decision (recommend AskUserQuestion once measured):** after Stage 1 on one
image, plot the reprojected diameter + score distributions and choose the two
thresholds. Both are config values that re-run only Stage 4 (seconds/image at
the old scale; validate timing at the new scale). The choice directly shapes
the §9 ceiling comparison, so it should be deliberate, not defaulted.

### 7.2 `.dbf` schema verification
"Same BoulderNet format" still warrants a check: confirm the new `.dbf` carries
`score` (drives `min_confidence`, currently null) and the same `cat_id`/
`cat_name`/`is_at_edge` columns the DATA_DICTIONARY documents. If `score`'s
range shifted or new size columns appear, surface them as filter options. One
`gpd.read_file(...).columns` + `.describe()` per image.

### 7.3 `find_shapefile` glob + uniqueness
`manifest.find_shapefile` fails loudly on **0 or >1** `*-mask-nms.shp` matches.
Verify each v2 folder has exactly one. If the new run's filename suffix differs
(not `…-mask-nms.shp`), update `manifest.SHAPEFILE_GLOB` (and add a test). Do
**not** drop new files beside old ones in the same folder — that trips the
">1 match" guard (which is why the v2 detections live in a separate root, §4.1).

### 7.4 SP1 bug per new `.prj`
Stage 1 auto-corrects the `D_unnamed` + far-from-CenterLat fingerprint via the
PDS `.LBL`. Which of the new `.prj` files trip it is data-dependent; the
`run_stage1.py` output (§3.1) prints `correction.status` per image. Record the
new corrections in DECISIONS.md. (No action unless a new `.prj` has a CRS
pathology the heuristic misses — then widen the fingerprint, with a test.)

### 7.5 DBF integrity (per the truncated-folder finding)
Before ingest, verify every folder's `.shp`/`.dbf`/`.shx` are internally
consistent: `geopandas.read_file` succeeds, and `len(gdf)` matches the `.shx`
record count `((shx_bytes - 100) / 8)`. `ESP_028537_2270` failed this (§1.1) —
its `.dbf` is ~1/3 the size its record count implies. Add this to the §6.0
bring-up as a pre-flight on all 40 folders so a corrupt transfer is caught
before a multi-hour sweep, not during it. (No local-GeoTIFF path is needed —
imagery comes from `JP2_URL` as in v1.)

---

## 8. Sanity checks / acceptance (per new/changed image)

1. **CRS residual O(200 m), not km** — `qa.assert_centroid_consistent` on the
   reprojected polygons vs manifest CenterLat/CenterLon (CLAUDE.md acceptance
   #1; `sanity.centroid_max_km = 15`). Fails loudly if the new `.prj` handling
   is wrong.
2. **Nested-grid consistency** — summing the finest grid up the ×2 ladder
   reproduces the coarse-grid stats (the existing Stage 4 test, run on a new
   image).
3. **`fractional_area` in [0, 1]** — at ~1M overlapping polygons/image the
   rasteriser keeps `boulder_area` union-correct, but confirm no tile exceeds
   `tile_area` (would indicate a rasterisation or units bug). Check the new
   target distribution (notebook 06) — far less zero-inflated than v1.
4. **`hirise_coverage_fraction` sane** per image — a near-zero value flags a
   tile-straddle (ESP_057469_2215 failure mode) → exclude that image or handle
   with multi-tile windowing.
5. **Polygon counts logged** — `n_polygons` (Stage 1 sidecar) and
   `n_polygons_after_filter` (Stage 4 sidecar) per image. A 0 or read-error is a
   corrupt/truncated folder (§7.5); a count far below the `.shx`-implied record
   count is the same.
6. **Idempotency / provenance** — re-running Stage 4 with the same v2 config
   reproduces the labels bit-for-bit and records the v2 `config_hash` +
   `detection_set`.
7. **Group-leak + within-image fold count** — Stage 5 / 5c invariants
   (notebook 09 group-leak assertion; `within_image_4fold` = (n_nonempty × 4)
   folds — now ~ (40 − excluded) × 4) hold on the v2 dataset.

---

## 9. The scientific payoff: does the denser 40-image set lift the AUC ceiling?

The reason to do the work. The vClaire set changes **two** things at once vs v1
— far denser labels **and** ~4× more images — so attribute carefully:

1. **Target-distribution shift** — recompute the `fractional_area`
   zero-inflation/skew on v2 vs v1 (notebook 06 distribution cells). 1M+
   detections/image should sharply reduce the zero-tile fraction and lengthen
   the positive tail; this alone changes what the regression/classification
   targets even look like.
2. **LOIO + binary re-sweep at 40 images** — mean Spearman / presence-AUC /
   bc_ge_1 AUC per (variant, scale). The 40-fold LOIO also directly tests
   modeling_results.md §5 experiment 2 ("more HiRISE images" → smaller per-fold
   SE), so a change here is a *mix* of label density and image count. Note the
   confound explicitly in the write-up.
3. **Within-image diagnostic (Stage 5c) is the cleanest comparison** — it is
   per-image and density-sensitive but image-count-independent, so it isolates
   the label-completeness effect from the cohort-size effect. v1 concluded
   "signal floor, not data quantity" on the sparse labels. If v2 within-image
   AUC is still ≈0.55, the floor is robust to label completeness (strong claim).
   If it lifts, the v1 floor was partly an artifact of missed boulders.
4. **Write-up** — a new `docs/modeling_results.md` section (v1-vs-v2 table +
   within-image deltas, with the density-vs-count confound called out) and a
   DECISIONS.md entry. Reuse `scripts/probes/_diag_within_image_deltas.py`
   pointed at the v2 sweep dir.

Because the cohorts differ, the v1↔v2 LOIO comparison is cohort-level (not
same-image). The "keep both" decision (§1) preserves v1 so this comparison is
reproducible; for any overlapping ObsIds (§6.2) a true same-image A/B is also
available.

---

## 10. Tests

- **`run_stage1.py` smoke** — building a gpkg for one ObsId produces a sidecar
  with the expected `correction.status`; integrates with the existing
  `test_sanity_residual_one_image.py` pattern.
- **Glob/uniqueness** — `find_shapefile` raises on 0 and on >1 matches (extend
  `test_manifest.py` if not already covered).
- **Filter sensitivity** — `_apply_detection_filters` with a lowered
  `min_size_m` keeps more polygons (synthetic; `test_labeling.py`).
- **Versioned-path integration (slow, opt-in)** — once v2 caches exist, a slow
  test asserting the v2 within-image scheme yields (n_nonempty × 4) folds and
  no group leak. Skips until `dataset_v2/` is present (same pattern as the
  existing slow tests).
- The bulk of the suite is synthetic/data-agnostic and stays green; the
  ObsId-specific slow tests (ESP_069669_2220 etc.) continue to validate v1.

Target: +4–6 tests; keep them light.

---

## 11. Documentation to update

- **README.md** — fix the grow-the-dataset recipe to start at Stage 1 (§3.2);
  add the A/B versioning recipe (§4.1).
- **DECISIONS.md** — new entry: the new detection set's provenance (date,
  BoulderNet params/version), the `min_size_m` decision (§7.1), per-image
  old→new polygon deltas, any new SP1 corrections, and the modeling A/B result
  (§9).
- **DATA_DICTIONARY.md** — only if the `.dbf` schema or any output column
  changes (e.g. a new size attribute exposed as a filter/feature).
- **docs/modeling_results.md** — the v1-vs-v2 comparison section (§9.4).
- **ROADMAP.md** — index this plan and its status.
- **config_v2.yaml** — the `detection_set` identifier (§4.3).

---

## 12. Time estimate

| Task | Est. |
|---|---|
| `scripts/run_stage1.py` + `--config`/`--dataset-dir`/`--all` (run_stage2) flags (§3) + tests | 75 min |
| Build the 40-row manifest (template URLs, fetch LBLs, derive tiles) + integrity pre-flight (§5.2/§7.5) | 60 min |
| `config_v2.yaml` + cache_v2 setup (§4.1) | 20 min |
| One-image bring-up + profile + filter decision (§6.0, §7.1) | 45 min |
| Stage 2 downloads (40 images, ~25–60 GB CTX + ~10–20 GB JP2) | **download-bound**, hours of wall-clock |
| Stage 1/4/4b/5 `--all` on 40 images (compute, after validating one) | ~1–2 h (Stage 4 at 1M polys/image is the unknown — §3.4) |
| Modeling re-sweep ×3 at 40 images (§6.3) | ~30–45 min |
| A/B comparison + write-up (§9) | 60–90 min |
| Doc updates (§11) | 30 min |
| **Total compute/dev (excl. download wait)** | **~6–8 h** + the download wait |

The dominant unknowns are the CTX download wall-time (§3.5) and Stage 4 at 1M
polygons/image (§3.4) — both resolved by the §6.0 one-image bring-up before
committing.

---

## 13. Open questions / risks

1. **Do the 40 vClaire ObsIds overlap the old priority10 set, or are they
   disjoint?** The 2 inspected are new. If disjoint, there is no shared imagery
   and no same-image A/B — it's purely a new 40-image dataset (§6.2). Need the
   full ObsId list to confirm.
2. **`ESP_028537_2270` is truncated/corrupt (§1.1).** Its `.dbf`/`.shp` are far
   smaller than its `.shx` record count implies and fail to read. Needs
   re-copying before ingest. Worth re-verifying the integrity of all 40 folders
   on arrival (§7.5 pre-flight) — a partial copy mid-sweep is expensive.
3. **How do I get the 40 manifest rows?** Proposed: auto-generate from the
   folder list + PDS URL templates + `.LBL` fetch, defaulting
   `BoulderLabel='unknown'` (§5.2). Confirm that's acceptable, or hand over a
   CSV. The PDS URL template must be verified to resolve for these ObsIds (HiRISE
   RDR path encodes the orbit range; templating from ObsId alone may need the
   orbit folder, which the `.LBL`/`BrowseURL` provides).
4. **Detection volume is ~1M polygons/image — is that expected?** That density
   (≈1 boulder / 70 m² if all real) suggests a very permissive detector or many
   sub-resolution/spurious detections on the downscaled imagery. This makes the
   filter decision (§7.1) central and the perf validation (§3.4) mandatory.
   Confirm the intended interpretation before treating all 1M as boulders.
5. **`min_size_m` / `min_confidence`** (§7.1) — highest-leverage knobs; decide
   from the *reprojected* distributions after the §6.0 bring-up. AskUserQuestion
   at that point.
6. **Disk budget ~80–120 GB for v2** (§3.5) — confirm free space; consider
   disabling `context_patch` for v2 (CNN is a documented dead-end) to save the
   largest chunk.
7. **Manifest naming** — use a new `manifest_v2.csv` (or `hirise_40_vclaire.csv`)
   referenced by `config_v2.yaml` so v1's `hirise_priority10.csv` stays
   reproducible.
8. **`models/` artifact sharing** — v1 and v2 sweeps both write under `models/`;
   rely on the per-run timestamp dir + a `dataset_version` snapshot tag. Version
   `models/` too if it gets confusing.
