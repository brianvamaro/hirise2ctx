# PLAN_MapValidation — validating the regional boulder-abundance map against independent data

**Opened 2026-08-28.** Five experiments that ask, from five independent directions, whether the
shipped abundance map is measuring **Mars** rather than measuring **CTX**. Each becomes one
notebook. All five read **one** growable product so that adding map tiles never edits an analysis.

> **Scope note.** This plan is *validation and interpretation of an existing product*. It renders no
> new map tiles, retrains nothing, and writes nothing into `reports/map_region`, `reports/map_a1` or
> `reports/map_extended`. The one new artifact it produces is the read-only union mosaic (§1).

---

## 0. The rulings this plan was built on (Brian, 2026-08-28)

| # | Decision | Ruling |
|---|---|---|
| 1 | How the notebooks see "all mapped areas" | **New `map_union` derived product** (§1). Notebooks read only this |
| 2 | Striping-artifact confound control | **Report raw contrasts + caveat in text.** No rotation nulls, no A1 arm. Quantifying and correcting the artifact is a **separate investigation** — Brian is not confident in how it has been done so far. Notebook 32 (§4) is the entry point to that separate work, not a control for the others |
| 3 | Target layers | **`abundance`** (calibrated) + **`prob_raw`** (uncalibrated) + **rich-tile fraction** |
| 4 | Rich-cell cutoff | **calibrated `prob >= 0.5`** — notebook 24's existing binary convention |
| 5 | Analysis unit | **native 160 m cell**, with significance never taken from pixel count (§2.3) |
| 6 | Crater catalog | **Robbins 2012 now** (already on disk), Liu 2024 as a drop-in second backend if access arrives |
| 7 | Degradation axis | **all four**: `DEGRADATION_STATE`, d/D ratio, `MORPHOLOGY_EJECTA_1`, `MORPHOLOGY_CRATER_1` (as control) |
| 8 | Crater metric | **radial profiles in crater radii**, stratified by degradation |
| 9 | Geologic-unit pooling | **pooled + per-polygon breakdown** |
| 10 | Thermal products | **all four**: Fergason THEMIS TI, TES TI, TES/IRTM rock abundance, THEMIS night IR |
| 11 | Validation-raster bounds | **derived from the union footprint**, not the config literal |

**On ruling 2 — what "caveat in text" commits us to.** Every notebook carries the same standing
caveat box: the map ships with the CTX source-frame artifact unmitigated (window-median η² **0.1444**,
ratio **1.599** over its own rotation null; DECISIONS 2026-08-25k), the artifact has power at the same
10s-of-km scale as the geologic and crater features being tested, and therefore **no contrast reported
in these notebooks is corrected for it**. Contrasts are to be read as *upper bounds* on the geologic
signal. This is a deliberate, logged choice to keep the five experiments independent of an artifact
treatment that is itself unsettled.

---

## 1. Prerequisite — `reports/map_union` (the growable product)

### 1.1 Why it must exist

`reports/map_region` (26 tiles, lon −12→20, lat 32→48) and `reports/map_extended` (104 tiles,
lon −56→−4, lat 16→48, both rounds shipped) **overlap in 8 tiles**:

```
E-12_N32  E-12_N36  E-12_N40  E-12_N44  E-8_N32  E-8_N36  E-8_N40  E-8_N44
```

Union = **122 tiles** as of 2026-08-29, when **round 2 rendered**: `map_region` 26 +
`map_extended` 104 − **8 shared**. It was **53** on the first build (26 + 35 − 8) — and the plan
as written said **54**, an arithmetic slip corrected against the products on disk (DECISIONS
2026-08-29a). Both numbers are historical now; the notebooks read the count from the product. Any analysis that naively pools the
two mosaics **double-counts 8 tiles = 15% of the current footprint** — silently, and worst of all
*non-uniformly*, biasing every pooled statistic toward that block's terrain.

**Verified 2026-08-28: all 8 overlapping tiles are byte-identical** (sha256 match on
`{tile}_abundance.tif`, 8/8) because `map_extended` *adopted* them from `map_region` via
`scripts/adopt_map_tiles.py`. So dedup is unambiguous — there is no "which copy wins" policy to
decide, and the union can assert byte-equality rather than choose.

### 1.2 What to build

`scripts/map_union.py` — modelled directly on `scripts/map_mosaics.py`, which stays the sole producer
of the per-arm mosaics.

- **Input:** a list of source arm dirs (default `map_region`, `map_extended`; a round-2 dir joins by
  adding one argument — **no code change**, per the plan-driven convention).
- **Dedup:** group tile rasters by tile id. Where a tile appears in more than one arm,
  **assert sha256 equality and take one**; a genuine mismatch is a **hard failure**, not a merge —
  it would mean two different heads rendered the same ground.
- **Lattice:** reuse `mapping.mosaic_geotiffs(..., require_shared_lattice=True)`. Both arms are on
  the R01 global lattice (`murray_v01_clon0_R3396190_ppd11855_S32_anchor_lonlat0`), so this must pass;
  if it ever fails, the union is refusing to bake a sub-cell phase into a displacement — the exact
  failure R01 exists to prevent.
- **Provenance tags:** carry `SIZE_FLOOR_*` forward (identical across arms — both are
  `v2_mixed_floor_2` off `deployable_g2`; **verify, don't assume**), and add
  `UNION_SOURCE_DIRS`, `UNION_TILES`, `UNION_N_TILES`, `UNION_TILE_ORIGIN` (a tile→arm map),
  `UNION_ADOPTED_TILES`.
- **Footprint gate:** the same closed-account check the per-arm mosaics pass —
  `finite_cells == n_tiles × 1479² − intra-tile nodata`, reported and asserted.
- **Output:** `reports/map_union/regional_{abundance,prob,prob_raw}_mosaic.tif` +
  `union_manifest.json`.

⚠ **`MANIFEST_NAMES`.** Anything new written into a map-output directory must join
`src.map_manifest.MANIFEST_NAMES` — `tile_sidecars` is a denylist *on purpose*, and an unlisted JSON
reads as a corrupt tile on a second lattice (CLAUDE.md). `union_manifest.json` must be added there.

**Expected geometry.** The union is an L/T-shaped block, not a rectangle: bbox lon −24→20,
lat 16→48 / lon −56→20 after round 2, i.e. 19 × 8 = 152 tile slots for 122 actual tiles.
**20.3% of the bbox is nodata** (measured) and every
notebook must handle NaN rather than assume a filled rectangle.

### 1.3 Shared analysis module — `src/map_validation.py`

Logic lives in `src/`, notebooks call it (CLAUDE.md). One module serves all five notebooks:

| Function | Purpose |
|---|---|
| `load_union(layer, ...)` | thin wrapper over `mapping.load_regional_mosaic` pointed at `map_union`; **read-only**, returns `(arr, transform, crs_wkt, meta)` |
| `three_targets(...)` | returns the ruling-3 triple: `abundance`, `prob_raw`, and `rich = prob >= 0.5` on one shared finite mask, so all three describe the *same* cells |
| `zonal_cells(geom, arr, transform)` | the cell values inside a geometry — returns the **distribution**, not a summary (rulings 5 + 9) |
| `radial_annuli(cx, cy, radius_m, edges_R, ...)` | cell values per annulus in crater radii (§3) |
| `frame_effective_n(...)` | independent-sample count from CTX source-frame membership (§2.3) |
| `CAVEAT_MD` | the single standing artifact caveat string, so all five notebooks quote it identically |

---

## 2. Cross-cutting method

### 2.1 The three targets (ruling 3)

Every experiment reports all three, in this order:

1. **`abundance`** — calibrated areal fraction. ⚠ Carries the size-floor caveat: it is the area share
   of boulders above a **per-image detection floor of 1.563–5.572 m²** (1.41–2.66 m diameter) mixed
   over 20 floors / 38 images. It is **NOT size-independent rock abundance** — which matters enormously
   for the thermal notebook, where TES/IRTM rock abundance *is* a different quantity (§5.3).
2. **`prob_raw`** — uncalibrated head output. The layer every striping/THEMIS diagnostic in the project
   used, so it is the comparable one.
3. **rich-tile fraction** — share of finite cells with calibrated `prob >= 0.5` (ruling 4).

A result that holds on all three is robust; one that appears only in `abundance` is likely a
calibration-curve artifact and must be reported as such.

### 2.2 Reporting standard (project rule, CLAUDE.md)

**Never presence AUC.** Where a skill-like number is wanted use the rich/poor threshold
`fa > 1e-2` family: `meaningful_auc` / `pr_auc@1e-2` / `precision@5%`, plus **Spearman ρ** and
**per-bin RMSE**. Hyperlink every citation to its canonical DOI.

### 2.3 Effective n — the one inferential guard that stays

Ruling 2 removed the rotation nulls. It did **not** license quoting p-values off ~57 million cells.
The 160 m cells are massively spatially autocorrelated, so:

- **Never** report a significance test whose n is the pixel count.
- Report **n at every level**: cells, polygons/craters, and **CTX source frames**.
- For any contrast, the headline uncertainty comes from a **bootstrap over the coarsest sensible
  unit** — polygons for geology, craters for §3, source frames for §4.
- `frame_effective_n` uses `striping.load_frames` (which pulls each tile's SeamMap over
  `/vsizip/vsicurl/` range requests — **no 1.8 GB tile download**) to count distinct contributing
  frames.

This is honest error-bar accounting, not artifact correction — it stands independent of ruling 2.

### 2.4 Standing caveats every notebook carries

1. **The striping artifact is present and uncorrected** (ruling 2 text above).
2. **`abundance` is size-floor-referenced**, not absolute rock abundance.
3. **Truth coverage thins fast outside circum-Chryse** — 23 of the 39-image cohort sit inside the
   shipped 26-tile block, only 1 in the new southern block. Any claim about `map_extended` terrain is
   extrapolation. Say so in captions.
4. **Map cells ≠ label cells.** The map grid is globally anchored (R01); the Stage-4 label grid stays
   tile-anchored. A map↔label comparison must resample, never index-match.

---

## 3. Notebook 30 — abundance by geologic unit

**Data (verified 2026-08-28, on disk):** `C:\Users\brian\Downloads\sim3292_database.zip` —
Tanaka et al. 2014 global geologic map (SIM3292), 1311 polygons, 44 units.
DOI [10.3133/sim3292](https://doi.org/10.3133/sim3292).

**Reads via** `/vsizip/C:/Users/brian/Downloads/sim3292_database.zip/SIM3292_MarsGlobalGeologicGIS_20M/SIM3292_geodatabase.gdb`,
layer `SIM3292_Global_Geology`, engine `pyogrio` (⚠ **`fiona` is not installed** in `geospatial`;
use `pyogrio`. ⚠ The `/vsizip/` path needs **forward slashes and a single leading slash** — the
backslash and `//` forms both fail).

**⚠ CRS — this is the #1 gotcha for this notebook.** SIM3292 is in
**`Robinson_clon0_Mars_2000_Sphere`** (sphere 3396190), *not* equirectangular. It must be reprojected
into the map's `clon_0` equirectangular CRS. Robinson is not equal-area and not conformal, so **any
area computed in the source CRS is wrong** — use the supplied `SphArea_km` field or recompute after
reprojection. Follow the project rule: read the CRS from the source, never hardcode.

**⚠ Invalid geometries.** `.intersects()` on the raw layer raises
`RuntimeWarning: invalid value encountered in intersects`. Run `make_valid` / `buffer(0)` before any
spatial predicate and report how many polygons were repaired.

**Measured coverage in the union bbox: 67 polygons, 16 units.**

| Unit | polys | Unit | polys |
|---|---|---|---|
| `AHi` Amazonian–Hesperian impact | 17 | `HNt` Hesperian–Noachian transition | 5 |
| `lNh` Late Noachian highland | 10 | `lAv` Late Amazonian volcanic | 3 |
| `eNh` Early Noachian highland | 8 | `eHt` Early Hesperian transition | 3 |
| `mNh` Middle Noachian highland | 8 | `mAl` Middle Amazonian lowland | 3 |

(plus `AHtu`, `lHl`, `lApc`, `Apu` and 4 more at 1–2 polygons.) Round 2 would take this to
**81 polygons / 19 units**.

**Sections**

- **§1** Load union + geology, reproject, repair, clip. Report polygon/unit/cell counts per unit and
  **flag every unit below a stated minimum cell count** as not analysable.
- **§2 Pooled distributions (ruling 9a).** Per unit, per target: full distribution — violin or ECDF,
  not just a median — because the target is heavily **zero-inflated and right-skewed** (CLAUDE.md), so
  a mean is close to meaningless. Rank units by median abundance with polygon-bootstrap CIs.
- **§3 Per-polygon breakdown (ruling 9b).** For each unit, the spread of *per-polygon* medians. A unit
  whose polygons disagree wildly is flagged explicitly — this is the section that answers whether
  "unit" is even the right explanatory variable, or whether within-unit regional variation dominates.
  Report a variance decomposition: between-unit vs within-unit-between-polygon vs within-polygon.
- **§4 Stratigraphic-age view.** Units carry epoch in their names (`eN`/`mN`/`lN`/`eH`/`lH`/`A`).
  Abundance vs relative age is the geologically interesting axis — older surfaces should be more
  boulder-poor if boulders break down over time, *unless* exhumation dominates.
- **§5 The honest limits.** The caveat box; which units are confounded with which map block; the fact
  that `AHi` (impact) overlaps notebook 31's craters by construction.

**Expected figure set:** `reports/figures/30_geology_{coverage,pooled,perpolygon,byage}.png`.

---

## 4. Notebook 31 — abundance vs crater degradation state

**Data (verified 2026-08-28, ALREADY ON DISK — no download):**
`cache_v2/craters/RobbinsCraters_20121016.tsv`, 58 MB, **384,345 craters, 70 columns**.
Robbins & Hynek 2012, DOI [10.1029/2011JE003966](https://doi.org/10.1029/2011JE003966).

**⚠ Encoding: the file is NOT UTF-8.** `pd.read_csv(..., sep='\t')` dies with
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe0 at position 54969`. Use
**`encoding='latin-1'`** (and `low_memory=False`).

**⚠ Liu et al. 2024 is NOT obtainable right now.** DOI
[10.1016/j.jag.2024.103952](https://doi.org/10.1016/j.jag.2024.103952); data at
[zenodo.org/records/10401940](https://zenodo.org/records/10401940) — the record is **published but
Restricted**: the API lists **zero files** and `/files` returns **HTTP 403**. It needs an access
request. Per ruling 6 the catalog loader takes a **backend argument** so a Liu ingest drops in later
without touching the analysis; the notebook runs on Robbins today.

### 4.1 Why Robbins is sufficient — the coverage check that settles it

The worry was that Robbins' `DEGRADATION_STATE` is sparse (only 2,507 of 14,526 craters ≥1 km in the
union bbox). **Measured: the NaNs are almost entirely craters too small to resolve anyway.**

| D ≥ | craters | with `DEGRADATION_STATE` | cells across radius @160 m |
|---|---|---|---|
| 1 km | 14,526 | 2,507 (17%) | 3.1 |
| 2 km | 4,717 | 2,507 (53%) | 6.2 |
| **3 km** | **2,785** | **2,497 (90%)** | **9.4** |
| 5 km | 1,661 | 1,609 (97%) | 15.6 |
| 10 km | 871 | 860 (99%) | 31.2 |

**Cut at D ≥ 3 km.** 2,497 craters, 90% degradation-labelled, 9.4 cells across the radius — enough to
resolve interior / rim / near / far ejecta annuli. All four states populated:
**1: 1046, 2: 534, 3: 757, 4: 160**. The D ≥ 3 km cut is a *resolution* requirement, not a
convenience: below it the radial profile has fewer cells than annuli.

### 4.2 The four degradation axes (ruling 7)

| Axis | Column(s) | Coverage @ D≥3 km | Role |
|---|---|---|---|
| Expert state | `DEGRADATION_STATE` (1–4) | 2,497 | **primary**, categorical |
| Depth/diameter | `DEPTH_RIMFLOOR_TOPOG` / `DIAM_CIRCLE_IMAGE` | 2,392 | **continuous** cross-check — regress, don't bin |
| Ejecta preservation | `MORPHOLOGY_EJECTA_1` | (report) | most direct: are ejecta *boulders* expected at all |
| Crater morphology | `MORPHOLOGY_CRATER_1` | 2,047 | **control** — simple vs complex emplace ejecta differently; never mix |

⚠ Read Robbins' own definition of `DEGRADATION_STATE` from the paper before interpreting the
direction of the scale (1 = fresh or 1 = degraded). **VERIFY AT RUNTIME**, log in DECISIONS.

### 4.3 Geometry (ruling 8)

Robbins gives **circular** outlines: `LATITUDE_CIRCLE_IMAGE`, `LONGITUDE_CIRCLE_IMAGE` (0–360 →
convert to ±180), `DIAM_CIRCLE_IMAGE`. This is a real limitation vs Liu's mapped boundaries and must
be stated: ejecta are *not* radially symmetric, so annuli average over real azimuthal structure.

- Annuli in crater radii **R**: `[0, 0.5]`, `[0.5, 1]` (interior/rim), `[1, 1.5]`, `[1.5, 2]`,
  `[2, 3]`, `[3, 4]` (ejecta), and a **background annulus `[5, 7] R`** for self-normalisation.
- Each crater is reported **as a ratio to its own local background** — this is what makes the
  comparison robust to regional level differences (and, incidentally, to a slowly-varying part of the
  artifact, though ruling 2 means we claim no artifact correction).
- **⚠ Overlap handling:** at D ≥ 3 km with a 7R background, craters *will* overlap each other. Mask
  cells belonging to another catalogued crater's interior out of any background annulus, and report
  how many craters lose their background entirely.
- **⚠ Projection:** compute annuli in **projected metres on the map's CRS**, not degrees — a degree of
  longitude at lat 48 is 0.67 of one at lat 20.

**Sections:** §1 load + filter + spatial join to union; §2 the radial-profile machinery + one worked
example crater; §3 mean profiles stratified by `DEGRADATION_STATE`, with crater-bootstrap CIs;
§4 the continuous d/D regression; §5 ejecta-morphology and simple/complex control splits; §6 the
decay-length summary — does ejecta boulder excess persist further from fresh craters; §7 caveats.

**Expected figures:** `reports/figures/31_craters_{coverage,example,profiles_by_state,dD_regression,controls}.png`.

---

## 5. Notebook 32 — abundance vs CTX source illumination *(= quantifying the artifact)*

**Framing (ruling 10 / question 3): this notebook does not pretend to be independent validation.**
It measures the shipped map against the *known physical cause* of its known artifact, at regional
scale over all 122 tiles. Deliverable: **how much apparent abundance variation is attributable to CTX
source-image illumination geometry** — i.e. an error bar to attach to the map, and the natural entry
point for the separate artifact investigation Brian flagged in ruling 2.

**Data — already solved, and cheap.** The Murray Lab **SeamMap** embeds per-source illumination
angles directly: `INCIDENCE`, `EMISSION`, `PHASE`, `SB_SLR_AZ`, per `PRODUCT_ID`
(verified on `E12_N44`: 56 sources, incidence 40–81°, sd 8.9°). **The PDS CTX CUMINDEX is not
required.**

**Existing machinery to reuse — do not rewrite:**

| Piece | Where | What it gives |
|---|---|---|
| `striping.load_frames(tile)` | `src/striping.py:192` | per-source-frame polygons, dissolved by `PRODUCT_ID`; pulls the SeamMap out of the remote zip over **`/vsizip/vsicurl/` range requests** and caches a GeoPackage — **no 1.8 GB download** |
| `ctx_source_illumination.rasterize_seam_map_window` | `src/ctx_source_illumination.py:129` | rasterizes all four angle fields **onto an arbitrary transform/shape** — i.e. straight onto the 160 m union grid |
| `striping.frame_labels_on`, `.eta2` | `src/striping.py` | per-frame label raster + η² |

**⚠ Cache-root trap.** Cached SeamMaps currently live under **`cache/ctx_tiles/_seammap_*`** (the *v1*
cache) while `striping.SEAM_DIR` points at **`cache_v2/ctx_tiles`** — the module's own comment flags
exactly this two-roots confusion. Only a handful are cached; the rest come over vsicurl. Resolve the
root explicitly and log which tiles came from cache vs network.

**⚠ Azimuth wrap.** `SB_SLR_AZ` is directional in [0,360). The existing module warns when a window
spans > 180° and its linear mean is then wrong. Over 122 tiles this **will** trigger — use circular
statistics for azimuth, not a linear mean.

**⚠ Zero-padded Murray tile ids** (`E-024_N28`, not `E-24_N28`) — every western tile 404s on the bare
form. `_padded_tile` handles it; anything new must too.

**Sections**

- **§1** Build the per-cell illumination stack on the union grid: incidence, emission, phase, azimuth,
  `SOURCE_ID`, `n_sources`, dominant-source fraction.
- **§2** Abundance vs incidence — the headline. Binned medians + Spearman ρ, **bootstrapped over
  source frames** (§2.3), for all three targets. Report the fraction of total abundance variance
  attributable to incidence.
- **§3** The other angles: emission, phase, and (circularly) sub-solar azimuth. Azimuth matters most —
  a shadow-driven artifact should track illumination *direction*, and this is the sharpest
  discriminator between "shadows read as boulders" and a radiometric level effect.
- **§4** Per-frame η² over the union (extending the 26-tile 0.1444 to all 122), and the frame-boundary
  step statistic — how big is the discontinuity across a seam where illumination changes by Δi.
- **§5** **The deliverable table**: apparent abundance change per 10° of incidence, in calibrated
  units, with the honest statement of what that means for reading the map. Plus: which mapped regions
  are most affected (low-contrast terrain), so the caveat can be made *spatial* rather than global.
- **§6** What this does **not** do: it does not correct anything, and it does not separate
  illumination from the geology that co-varies with it. Hand-off notes for the separate artifact
  investigation, including the untried **gain-capped A1** (logged as v3).

**Expected figures:** `reports/figures/32_illum_{stack,vs_incidence,vs_angles,eta2_union,deliverable}.png`.

---

## 6. Notebook 33 — abundance vs thermal-derived products (TES + THEMIS)

This is PLAN_RegionalMap's **unblocked thermal legs**, done properly and over the full union.
Notebook 24 §3.1 did leg 1 (THEMIS night IR, pooled, co-registered) on the 26-tile block only:
ρ = **+0.052** pixel / **+0.066** at ~1.3 km / **+0.063** at ~10 km. Leg 2 (quantitative TI) was
**never done**.

### 6.1 The four products — all verified reachable 2026-08-28

| # | Product | Endpoint | Verified | Format / grid |
|---|---|---|---|---|
| 1 | **Fergason THEMIS quantitative TI**, 100 m | `https://asc-astropedia.s3.us-west-2.amazonaws.com/Mars/Odyssey/THEMIS-Global-Thermal-Inertia-Mosaic/Quantitative-32-Bit/THEMIS_TI_Mosaic_Quant_{tile}_100mpp.cub` | **HTTP 206** on all 4 needed tiles; **opened over `/vsicurl/`** | ISIS3 `.cub`, **float32**, 2.53 GB/tile, 17783×35565, nodata −3.4e38 |
| 2 | **TES TI (Putzig 2007)**, 20 ppd | `https://pds-geosciences.wustl.edu/mgs/mgs-m-tes-5-timap-v1/mgst_9001/data/global_ti_{night,day}_2007.{img,lbl}` | **HTTP 206**, label read | PDS3, 7200×3600, MSB 16-bit, 51.8 MB |
| 3 | **IRTM rock abundance (Christensen 1986)** | WMS `http://ms-mars.mars.asu.edu/VI_blocks_numeric?...&LAYERS=VI_blocks_numeric&FORMAT=image/vicar` | **HTTP 200, real float raster returned** for the union bbox | VICAR REAL, values **0.98–25.1 %**, mean 9.2 |
| 4 | **THEMIS night IR**, 100 m | already in `config_v2.yaml`; cached at `cache_v2/validation/themis_night_ir_region.tif` | in use | GeoTIFF |

**The Fergason find is the important one.** Its CRS is
`SimpleCylindrical Mars, sphere 3396190, central_meridian 0` — **the same equirectangular clon_0
sphere as the CTX mosaic**. It drops straight into the existing `fetch_validation_data.py` /
`fetch_region_raster` path with no special handling. Tiles needed for the union bbox
(lon −24→20 = 336–20 °E, lat 20→48): **`30N000E`, `30N300E`, `00N000E`, `00N300E`** (tiles are
30° lat × 60° lon, named by lower-right corner — **VERIFY the corner convention at runtime** before
trusting the tile choice).

⚠ **Rejected:** `https://www.mars.asu.edu/data/tes_putzigti/nighttime2005/nmap2003.tif` is an
**8-bit RGB picture with no geotransform** — a figure, not data. Do not use it. Product 2 (PDS) is the
real TES TI.

⚠ **Product 3 is quantised.** The WMS returned only **25 distinct values** over the union bbox, so it
is a stretched 8-bit derivative of the underlying ~1 ppd map, not full-precision. Fine for
**Spearman rank** correlation; **not** fine for regression slopes or RMSE. Report it as ordinal.

### 6.2 The bounds fix (ruling 11)

`config_v2.yaml` `validation_rasters.region_bounds_lonlat` is hardcoded to **`[-12, 32, 20, 48]`** —
the 26-tile block. It **does not cover `map_extended` at all**. Add
`--bounds-from-union` to `scripts/fetch_validation_data.py`, deriving bounds from the union mosaic
footprint, with the config literal as fallback default. Growing the map then widens the thermal fetch
automatically. Add the two new products to the config `products:` block.

⚠ `--match-mosaic` co-registration must now reference **`reports/map_union`**. A validation raster
built `--match-mosaic` is only index-comparable to the generation of map it was matched against —
which arm it points at is part of the product.

### 6.3 The science — and the one trap that matters

**§5.3 the quantity mismatch, stated up front.** Our `abundance` is **size-floor-referenced**
(boulders above ~1.4–2.7 m diameter). TES/IRTM rock abundance is the areal fraction of material with
TI ≥ 1250 (bedrock, boulders, indurated sediment) at ~1° resolution. **These are different physical
quantities at ~50× different resolution.** A modest correlation is the *expected* result; a high one
would be surprising. Frame the comparison as **rank agreement on where rocky terrain is**, not as
validation of absolute values. Anything else over-claims.

**Sections:** §1 fetch + co-register all four onto the union grid (`assert_coregistered`, expect
dx=dy=0); §2 leg 1 re-run over the full union — does the weak +0.05 hold on 122 tiles; §3 **leg 2, the
new one** — abundance vs quantitative Fergason TI, at native, ~1.3 km and ~10 km block scales;
§4 TES TI cross-check; §5 **abundance vs IRTM rock abundance** (Spearman only, at ~8 km blocks — the
closest thing to an independent estimate of the same variable); §6 multi-scale summary: ρ vs
aggregation scale for all four products, which is the single most informative plot for "at what scale
is this map trustworthy"; §7 caveats.

**Expected figures:** `reports/figures/33_thermal_{coverage,leg1_union,ti_scatter,tes_ti,rock_abundance,rho_vs_scale}.png`.

---

## 7. Notebook 34 — abundance along known boulder deposits (Rodriguez 2016) *(later stage)*

Rodriguez et al. 2016, *Sci Rep* 6:25106, DOI [10.1038/srep25106](https://doi.org/10.1038/srep25106)
— open access. Late-Hesperian **megatsunami** deposits in circum-Chryse; the boulder-rich cohort sites
sit on the highland–lowland boundary this paper maps, and `lHl` (Late Hesperian lowland) is one of the
16 units present in the union footprint. This connects directly to Brian's working hypothesis that
many v2 boulder fields are megatsunami-transported.

**Blocked on a manual step, by design.** The deposit outlines exist only as **figures**, so they must
be digitised and georeferenced before any analysis. Sequence:

1. **Georeference the figure** (Fig. 2 / the deposit maps) against MOLA or THEMIS using identifiable
   craters — the accuracy of everything downstream is set here, so report control-point residuals.
2. **Digitise** deposit polygons (older vs younger lobe where the paper distinguishes them).
3. Store as a versioned GeoPackage under `cache_v2/validation/` with a **provenance sidecar**
   recording source figure, control points, residuals, and who digitised it.
4. Then: abundance inside vs outside deposits, along-flow profiles, older vs younger lobe contrast —
   and the honest confound statement, since deposit boundaries partly follow topography that the map
   may respond to independently.

**Open question deferred to execution:** whether digitising from figures is accurate enough to be
worth it, or whether the better instrument is a *predicted-deposit* test — does the map's
high-abundance terrain independently reproduce the mapped deposit outline. That is arguably the
stronger claim and needs no perfect digitisation.

---

## 8. Build order

| Step | What | Depends on | Notes |
|---|---|---|---|
| 1 | `scripts/map_union.py` + `src/map_validation.py` + tests | — | ✅ **DONE 2026-08-29** (DECISIONS 2026-08-29a). Rebuilt at **122 tiles** after round 2; footprint closes |
| 2 | Notebook 30 — geology | 1 | ✅ **DONE 2026-08-29** (DECISIONS 2026-08-29b). 75 polygons / 14 units; partition + cell accounts both close |
| 3 | Notebook 31 — craters | 1 | data on disk; no network |
| 4 | Notebook 33 — thermal | 1, config + fetch changes | biggest download |
| 5 | Notebook 32 — illumination | 1 | 122 SeamMap fetches over vsicurl |
| 6 | Notebook 34 — Rodriguez | 1 + manual digitisation | last |

Notebooks 30 and 31 are the fastest to a real result — both datasets are already local.

**Convention reminders.** Notebooks are **generated**: edit `notebooks/_build_NN.py`, regenerate, then
`nbconvert --execute --inplace`. **Never run two notebooks (or two CTX-heavy jobs) at once.** Scripts
and notebooks are **not** covered by the test-side write guard — give producers explicit absolute
scratch roots and never let a notebook write into a shipped map directory.

---

## 9. Open questions for execution (do **not** pre-decide)

1. **Minimum cell count** for a geologic unit / crater stratum to be reportable. Needs the actual
   distributions to set sensibly.
2. **Robbins `DEGRADATION_STATE` direction** — 1 = fresh or 1 = degraded. VERIFY from the paper, log
   in DECISIONS.
3. **Fergason tile corner convention** — confirm `30N000E` covers lat 30–60 N before trusting the
   4-tile selection.
4. Whether the union should also carry an **A1 union** for future differencing. Deferred: A1 covers
   only 26 tiles, and ruling 2 keeps A1 out of these notebooks.
5. Whether notebook 34 should use digitised outlines or the predicted-deposit test (§7).

---

## 10. What would make this a negative result

Stated in advance, so the programme cannot be graded after the fact:

- If **notebook 32** shows abundance-vs-incidence explains a large share of regional variance, then
  the geologic and crater contrasts in 30/31 are correspondingly weaker than they appear, and the
  honest headline becomes "the map is substantially an illumination map." Ruling 2 means these
  notebooks will not *correct* for that — but 32's number is what tells us how loudly to caveat.
- If **notebook 30** finds within-unit-between-polygon variance dominating between-unit variance,
  geologic unit is not a useful predictor of boulder abundance at this scale, and that is a
  publishable negative.
- If **notebook 31** finds no ejecta excess even for the freshest craters, the map is not resolving a
  signal that is known to exist — a strong argument that the 5 m/px CTX floor is binding.
- If **notebook 33** finds ρ with quantitative TI no better than the weak +0.05 already seen with
  night IR, the map is not tracking the best independent measure of surface rockiness available.
