# PLAN — Stage 4b: per-tile CTX texture features

**Status:** **shipped** in commit `014f645` (2026-05-23). Results + decisions logged in
[DECISIONS.md](DECISIONS.md) under the 2026-05-23 Stage 4b entry. 9 feature families
(the 4 from CLAUDE.md §4 plus 5 from §3.5 below) totalling 60 columns; 643,910 feature
rows across 9 ObsIds; 3.3 GB of bundled context patches. Notebook 07 (cross-image QA) +
notebook 08 (per-feature walkthrough + stratified patch viewer) are the visual outputs.

This plan stays checked in as the architecture-level reference; minor deviations are
recorded in the DECISIONS.md entry (most notably the context-patch storage layout,
§6 below — patches are bundled per (ObsId, patch_size) into single `.npy` stacks
rather than the per-tile `{ti}_{tj}.npy` files the plan originally prescribed).

Reads from existing Stage 2 + Stage 4 caches; emits one parquet per ObsId alongside `dataset/labels/{ObsId}.parquet`.

**Why a separate stage** — CLAUDE.md acceptance #4 requires that adding or changing features doesn't re-run Stages 1-3 or even Stage 4. Splitting feature extraction out of Stage 4 satisfies that: a config change to `labeling.features` (or to a future `features.*` block) re-runs Stage 4b only, in seconds-to-minutes.

**Scope already pinned in CLAUDE.md** — §4 lists the four feature families (`intensity_stats`, `glcm`, `gradient`, `shadow_fraction`) and the power-of-2 context patch. This plan fills in the *how*: which specific stats, which GLCM parameterisation, which shadow detector, where the artifacts land, what the output schema is. It also adds five complementary feature families surfaced by 2026-05-23 literature pass (§3.5).

**Resolution-preservation principle (cross-ref PLAN_modeling.md §0):** CTX pixels stay at native 5 m/px throughout — boulders are already sub-pixel to a few-pixel at this resolution, so any spatial downsampling forfeits signal we can't recover. Intensity quantization (GLCM `levels`) is a related-but-separate concern; defaults raised in §3.2 to preserve more intensity bins.

**Context patches: ON by default** for Week 3, because PLAN_modeling.md §4 makes the CNN baseline non-optional. Stage 4b emits them in the same pass as the tabular features.

---

## 1. Inputs

| Input | Path | Used for |
|---|---|---|
| CTX window (uint8) | `cache/ctx_windows/{ObsId}.tif` | All features |
| HiRISE coverage mask | `cache/ctx_windows/{ObsId}_hirise_mask.tif` | Eligibility (must match Stage 4's set exactly) |
| Per-tile bounds + indices | `dataset/labels/{ObsId}.parquet` | Iteration grid — guarantees Stage 4b emits exactly the rows Stage 4 emitted |
| Stage 4 provenance | `dataset/labels/{ObsId}.json` | `tile_sizes_px`, `subpixel_factor`, `mosaic_row/col_origin` |
| Stage 3 shift (optional) | `cache/coregistration/{ObsId}.json` | Not used — CTX pixels are fixed; features are about CTX texture, not boulder placement |

**Key invariant:** Stage 4b iterates the rows of the existing label parquet, not the eligibility math from scratch. This guarantees the join `(obs_id, scale_idx, ti, tj)` is exact across labels + features and avoids re-deriving the alignment.

## 2. Output

`dataset/features/{ObsId}.parquet`, one row per (scale_idx, ti, tj), join key matches the label parquet exactly.

`dataset/features/{ObsId}.json` sidecar with the same provenance pattern as Stage 4: source paths, feature-config hash, per-feature parameters, written-at timestamp.

Optional `dataset/context_patches/{ObsId}/S{tile_size_px}/{ti}_{tj}.npy` directory layout for raw CTX patches when `context_patch_px` is non-null. Off by default; see §6.

## 3. Feature families

All features computed on the CTX window's native pixel grid (no resampling). For each tile, slice the CTX array at `[r_win : r_win+S, c_win : c_win+S]` where `(r_win, c_win)` is the tile's window-pixel origin (recoverable from `ti, tj` + `mosaic_row/col_origin` + `tile_size_px`).

### 3.1 `intensity_stats` (cheap, always on)

Per-tile reductions over the CTX uint8 array:

| Column | Reduction |
|---|---|
| `intensity_mean` | mean |
| `intensity_std` | population stddev |
| `intensity_min` | min |
| `intensity_max` | max |
| `intensity_p10` | 10th percentile |
| `intensity_p50` | median |
| `intensity_p90` | 90th percentile |
| `intensity_iqr` | p75 - p25 |

Trivially vectorisable. ~8 columns; ~minutes for the whole dataset.

### 3.2 `glcm` (gray-level co-occurrence matrix texture)

Use `skimage.feature.graycomatrix` + `graycoprops`. Architectural decisions (revised 2026-05-23 to preserve more intensity information per the resolution-preservation principle, citing Clausi 2002 "An analysis of co-occurrence texture statistics as a function of grey level quantization", *Canadian J. Remote Sensing*):

- **Levels (scale-dependent):** 8 at the finest scale (8 CTX px tile has only 64 samples — high quantization avoids degenerate co-occurrence matrices), **16 at scales 16-32 px**, **32 at scale 64 px** (4096 samples can fill a 32×32 matrix meaningfully). Recorded per scale in `features.glcm_levels_per_scale`.
- **Distances:** `[1]` at the finest scale (8 px); `[1, 2, 3]` at coarser scales. Adding `d=3` matches the typical 1-3 px boulder-shadow separation at CTX 5 m/px under realistic sun angles.
- **Angles:** `[0, π/4, π/2, 3π/4]`. Average over angles for rotational invariance — call this single-value-per-property GLCM. Per-angle values explode the column count without much modeling lift on small datasets.
- **Properties:** `contrast`, `dissimilarity`, `homogeneity`, `energy`, `correlation`, `ASM`. 6 per (distance, [averaged-over-angle]).

Output column naming: `glcm_{property}_d{distance}` (rotational average baked in). At finest scale: 6 columns. At coarser: 12 columns. Pad missing distances with NaN at finest so the schema is stable across scales.

**Performance:** `graycomatrix` is the bottleneck — Python loop over tiles. For ~500k tiles total this is the slowest feature family. Two mitigations:
- Process per-(scale, image) in chunks; reuse the quantised image across all tiles at that scale.
- If still too slow, drop `correlation` (most expensive) or rewrite the GLCM inner loop in numpy. Punt this optimisation until measured.

### 3.3 `gradient`

Sobel gradients via `scipy.ndimage.sobel` on the full window once per scale, then per-tile reductions:

| Column | Reduction |
|---|---|
| `grad_mag_mean` | mean magnitude `sqrt(gx² + gy²)` |
| `grad_mag_std` | stddev of magnitude |
| `grad_mag_p90` | 90th percentile of magnitude |
| `grad_mag_p99` | 99th percentile of magnitude (added 2026-05-23 — boulder edges are rare bright outliers; p90 saturates in busy tiles) |
| `grad_dir_circvar` | circular variance of gradient direction `atan2(gy, gx)` over pixels with non-trivial magnitude |

Cheap. Whole-window gradient is computed once; per-tile slicing is a view.

### 3.4 `shadow_fraction` + `bright_cap_fraction`

The simplest reliable shadow detector at CTX scale is brightness thresholding derived from the per-image dark-tail distribution.

Recommendation: **per-image absolute DN threshold derived once from the dark-tail mode** (more stable across tiles than a local percentile, which moves with the tile's own distribution). Compute the threshold during a one-pass per-image pass over the masked CTX, then apply per-tile.

Emit three companion columns:
- `shadow_fraction` — fraction of tile pixels below the per-image dark threshold.
- `shadow_fraction_p05` — stricter variant (p5 instead of p10 of the dark-tail distribution); disentangles "dark terrain" from true shadows.
- `bright_cap_fraction` — fraction of tile pixels **above** the per-image bright-tail mode. Sunlit boulder tops pair with adjacent shadows; the bright/shadow asymmetry is a stronger boulder signal than either alone. Cheap and supported by the shape-from-shading intuition used in HiRISE DTM photoclinometry (Kirk et al. 2008).

Tradeoffs surfaced for AskUserQuestion at execution time:
- DN-mode threshold (recommended) vs image-percentile threshold (simpler, drifts with image overall brightness).
- p5 vs p10 vs p1 for the strict variant.

### 3.5 Additional feature families (research-informed, 2026-05-23 web pass)

The four families above (intensity/GLCM/gradient/shadow) match CLAUDE.md §4. A literature pass surfaced five more families that complement them without overlap. Add all five; each is cheap and operates at native CTX resolution.

#### 3.5.1 Higher-order intensity moments
Per-tile `intensity_skewness`, `intensity_kurtosis`. Boulder shadows produce a left-skewed, heavy-tailed intensity distribution that std alone misses. Trivially cheap; works down to 8 px. Precedent: Bandeira et al. 2007 (Mars dune-field detection on HiRISE-class data).

#### 3.5.2 Local Binary Patterns (LBP)
Use `skimage.feature.local_binary_pattern` with `method='uniform'`, `P=8`, `R=1`. Emit the 10-bin uniform-LBP histogram per tile (`lbp_hist_0` ... `lbp_hist_9`). Illumination-robust; complements GLCM by capturing micro-pattern frequencies. Works at 8-16 px and up. Mars precedent: Palafox et al. 2017 (geological landform detection); lunar precedent: Vijayan et al. 2013 (crater detection).

#### 3.5.3 Lacunarity (gliding-box on shadow mask)
For the two largest scales only (32-, 64-px tiles). Measures gappiness/clustering of dark pixels — directly relevant to "are boulders evenly scattered or clustered?". Degenerate below 16 px (not enough box positions). Emit `lacunarity_shadow_b2` and `lacunarity_shadow_b4` (gliding-box sizes 2 and 4). Planetary precedent: Plesko et al. 2009/2010 (lunar surface roughness via lacunarity).

#### 3.5.4 Multi-scale variance (variance-of-sub-tile-means)
**Essentially free given the nested ×2 ladder.** For a 32-px tile, compute the mean of each 8-px sub-block (16 sub-blocks) and take their variance. Captures internal heterogeneity in a way single-tile std cannot. Emit `intensity_subtile_var` at scales 16/32/64 (no value at the finest scale — no sub-blocks). Standard planetary roughness rationale: Shepard et al. 2001 ("The roughness of natural terrain", JGR).

#### 3.5.5 Edge density + edge-orientation entropy
Canny edges via `skimage.feature.canny`. Emit:
- `edge_density` — Canny pixel count / tile pixel count.
- `edge_orientation_entropy` — Shannon entropy of the orientation histogram of Canny pixels.

Captures "structured vs isotropic" texture independent of GLCM contrast magnitude. Works at 16 px and up. Precedent: Stepinski & Vilalta 2005 (automated Mars terrain classification).

### 3.6 Considered but skipped (with reasons)

- **HOG** — overlaps with gradient features; cell/block geometry awkward at 8-16 px.
- **Gabor banks** — multi-scale info already covered by the nested grid + GLCM `d ∈ {1, 2, 3}`.
- **Wavelet packet energies** — same multi-scale story as the nested variance, more expensive, harder to interpret.
- **Semivariogram / Moran's I / Geary's C** — unstable on 8-16 px tiles (too few lag pairs); revisit if a ≥64 px-only feature pass is wanted.
- **TPI / TRI / slope-from-intensity** — without a DEM these collapse into the gradient family.
- **Fractal box-counting dimension** — needs ≥64 px + a binarization choice; lacunarity gives most of the same signal more stably.

### 3.7 Operational column `valid_pixel_fraction`

Add a `valid_pixel_fraction` column to the features parquet — share of pixels inside the tile that are within the HiRISE coverage mask. By Stage 4 eligibility this is always 1.0 for emitted tiles, but recording it as an explicit column lets future relaxed-eligibility configs (e.g. coverage >= 0.95) be filtered downstream without re-running.

## 4. Module + file layout

```
src/features.py              # new — Stage 4b implementation
scripts/run_stage4b.py       # new — driver mirroring run_stage4.py
tests/test_features.py       # new — unit + slow integration
notebooks/07_features_qa.ipynb  # new (or rename — depends on Stage 5 ordering)
```

`src/features.py` interface (mirrors `src/labeling.py`):

```python
def stage4b_one_image(
    obs_id,
    *,
    cache_dir,
    output_dir,
    features_cfg,
    config_hash,
) -> dict:
    """Read the Stage 4 label parquet for obs_id and emit features per tile."""
```

Per-feature-family functions are small + tested in isolation:

- `_compute_intensity_stats(arr, tile_slices) -> dict[str, np.ndarray]`
- `_compute_glcm(arr, tile_slices, *, levels, distances, properties) -> dict[str, np.ndarray]`
- `_compute_gradient(arr, tile_slices) -> dict[str, np.ndarray]`
- `_compute_shadow_fraction(arr, mask, tile_slices, *, percentile) -> np.ndarray`

`tile_slices` is a list of `(r0, c0, S)` triples derivable from the label parquet's `(ti, tj, tile_size_px)` + `mosaic_row/col_origin` from provenance.

## 5. Config

Add a `features` block to `config.yaml` parallel to `labeling`:

```yaml
features:
  enabled: [intensity_stats, glcm, gradient, shadow_fraction]
  glcm:
    levels: 8
    distances_px: [1, 2]     # auto-clipped to tile_size_px - 1 at runtime
    properties: [contrast, dissimilarity, homogeneity, energy, correlation, ASM]
    angle_average: true
  gradient:
    sigma: 1.0               # Gaussian smoothing before Sobel
  shadow_fraction:
    threshold_method: image_percentile   # | local_window
    image_percentile: 10
  context_patch:
    enabled: false
    sizes_px: [32, 64]       # powers of 2; one patch per (tile, size) when enabled
```

`labeling.features` and `labeling.context_patch_px` from the current config are deprecated in favor of the above; document the move in a `DECISIONS.md` entry at execution time. Don't delete them silently — config validation should still accept the old keys with a warning.

## 6. Context patches (optional)

When `features.context_patch.enabled: true`, for each label tile and each `sizes_px[k]`, save a power-of-2 patch of CTX pixels centered on the tile. Patch size is a CTX-pixel power of 2 (32 or 64 px = 160 or 320 m), so the largest is the same as the coarsest label-tile size — they're literally the same pixels for a coarsest-scale tile.

Storage: `dataset/context_patches/{ObsId}/S{patch_size_px}/{ti}_{tj}.npy` (uint8 numpy arrays). Reference from the feature parquet by adding columns `patch_path_S32`, `patch_path_S64` (relative paths or null).

Cost: a 32×32 uint8 patch is ~1 KB. ~500k tiles × 2 patches = ~1 GB on disk. Acceptable if the user wants the CNN baseline; skip-by-default keeps the dataset lean.

## 7. Tests

Mirrors the Stage 4 test split (fast unit + slow integration):

- `test_intensity_stats_simple` — known uniform tile, known gradient tile.
- `test_glcm_uniform_image_has_zero_contrast` — sanity.
- `test_gradient_on_step_function` — recovers expected magnitude/direction.
- `test_shadow_fraction_image_percentile` — synthetic bimodal image.
- `test_features_align_with_labels_row_for_row` — slow integration on ESP_069669_2220; assert the labels parquet and features parquet have identical (scale_idx, ti, tj) sets.
- `test_stage4b_is_idempotent` — same config in → identical output.

Target: +10-15 tests; pytest 88 → ~100.

## 8. QA notebook

Patterns from notebook 06:
- Per-image heatmap of `intensity_mean`, `glcm_contrast`, `grad_mag_mean`, `shadow_fraction` at finest scale.
- Scatter `glcm_contrast` vs `fractional_area` colored by image (sanity — boulders should drive contrast up).
- Per-image distribution of `shadow_fraction` (sanity — should peak at the configured percentile by construction; tails carry the signal).
- Pair-plot or correlation matrix of the feature families (look for redundancy before modeling).

## 9. Key decisions to surface via AskUserQuestion at execution time

1. **GLCM levels** — 8 (recommended) vs 16 vs 32. Higher = more detail, slower.
2. **Shadow detector method** — image-percentile (recommended, simple, stable) vs local-window (more robust to illumination gradient, more compute).
3. **Context patches default** — off (recommended, save disk, enable later when CNN is on the table) vs on (1 GB for both sizes; have them ready).
4. **GLCM angle averaging** — average over 4 angles (recommended, rotation-invariant, smaller schema) vs per-angle columns (more info, may help modeling, 4x columns).
5. **Whether to deprecate `labeling.features` / `labeling.context_patch_px` keys** — move to a `features.*` block (recommended) vs keep them in `labeling.*` for backwards compatibility.

## 10. Sequencing relative to Stage 5

Stage 5 (splitter) doesn't need features — it operates on `(obs_id, scale_idx, ti, tj)` group structure only. So the order can be either:

- **4b then 5** (recommended) — features land before the splitter is written, so when Stage 5 emits a sample split for sanity-checking, it can show the actual feature distributions per fold.
- **5 then 4b** — splitter ships first; features get added as a separate concern. Marginally faster path to a baseline but loses the cross-validation feature-stability sanity check.

Either way, both must land before Week 3 modeling.

## 11. Open questions (carry forward to PLAN_modeling.md)

- Do we want to compute features at **all 4 scales** or only the scales the modeler cares about? Current default is all 4 — let the modeler decide which to use.
- Do we want the feature parquet to embed the label columns too (denormalised) for easier ad-hoc analysis, or stay strict per-stage and require joins? Recommendation: stay strict; the join is one line in pandas.
