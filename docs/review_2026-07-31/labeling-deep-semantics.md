# Review area: labeling-deep-semantics

- **Reviewed at commit:** 7bfedb8
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified)
- **Relation to prior work:** second pass over the label basis, Pattern D item (ii) — *what does the
  labeller publish, and can that statistic move?* Nothing here duplicates `labeling-1..4`, R03, R23,
  R44, R50 or R56. Three places where I **extend or correct** an existing finding are flagged inline
  with **EXTENDS R03** / **CORRECTS R03's verifier** / **EXTENDS R23**.

---

## Summary in one line

`fractional_area` is not an areal boulder fraction; it is *the areal fraction of boulders larger than
an undeclared, per-image minimum size* — **1.411–1.427 m equivalent-circle diameter for 12 images and
1.943–2.664 m for the other 26** — and every statistic the project publishes on top of it (the
`fa > 1e-2` rich/poor class, the Stage-7d `boulder_count > 50` partition, the LOIO per-image AUC, the
deployed `*_abundance.tif` values) inherits that undeclared convention. Holding the shipped model
predictions **completely fixed** and changing only the label's size convention moves committed
artifact values.

---

## Findings

### semantics-1 — The `fa > 1e-2` rich/poor class is cohort-dependent: restated on one common size floor the 0.25 m/px cohort's rich prevalence halves while the 0.50 m/px cohort's does not move, and the change is a *re-ranking*, not a rescale, so no per-image level correction can absorb it
- **Severity:** medium
- **Liveness:** live-shipped (the target of the frozen recipe + the shipped mosaic map) **and**
  live-active-plan (`PLAN_RegionalMap.md` leg 4)
- **Confidence:** high (the counterfactual machinery reproduces the shipped `boulder_count` *exactly*
  and the shipped rich share to 3e-4 — see Verified clean)
- **Where:** target defined at `src/labeling.py:375, 389`; floor applied at `src/labeling.py:96-114`;
  published without a floor at `dataset/DATA_DICTIONARY.md:161,164`, `docs/methods.md:697-700,750`;
  consumed as the frozen positive class in `models/deployable/86c51a5dca220f63/recipe.json`
  (`"target_id": "fa_gt_1e-2"`); the moved artifacts are
  `reports/figures/striping_a1_loio_summary.csv` and `reports/figures/striping_a1_loio_preds.csv`

R03 establishes that the two pixel-scale cohorts have disjoint detection floors. This finding follows
that through to the **published target**. The shipped label basis's *effective* floors, measured after
the shipped `min_size_m` filter on all 5,892,089 surviving polygons of the 38 label images, are:

| cohort | n images | floor area (m²) | floor diameter (m) | median boulder |
|---|---:|---|---|---|
| 0.25 m/px | 12 | **1.5628 – 1.6004** (the configured floor binds) | **1.411 – 1.427** | 3.10 m² / 1.99 m |
| 0.50 m/px | 26 | **2.9652 – 5.5719** (the configured floor never binds) | **1.943 – 2.664** | 11.06 m² / 3.75 m |

**CORRECTS R03 (a precision amendment, not a reversal).** R03's verifier quotes the *pre-filter*
minima (0.83–1.37 vs 2.97–4.45 m²) as the shipped asymmetry. Post-filter — which is what actually
enters `boulder_area` — the fine cohort is equalised at exactly 1.5625 m² by `min_size_m`, so the
shipped asymmetry is **1.9–3.6× in area / 1.4–1.9× in diameter**, not 3–4× in area. The mechanism and
the direction are unchanged; the magnitude is somewhat smaller than the register states. Symmetrically,
the *coarse* cohort is the heterogeneous one (floor spans 1.9× within itself), not the fine one.

Restating the target on one common floor (the config's own stated 0.50 m/px design floor, 6.25 m²
= 2.82 m) and re-deriving the class:

| | rich share `fa>1e-2` @ S=32 | on the common floor | ratio |
|---|---:|---:|---:|
| pooled (161,005 tiles) | **0.3598** | 0.3226 | 0.897 |
| 0.50 m/px (126,214 tiles, 78.4 %) | 0.3692 | 0.3663 | **0.992** |
| 0.25 m/px (34,791 tiles, 21.6 %) | 0.3258 | **0.1642** | **0.504** |

Per image the fine cohort's ratio has median **0.191** (range 0.029–0.869) against 0.996
(0.726–1.010) for the coarse cohort (Mann–Whitney p = 1.4e-6). Up to **64.3 %** of one image's tiles
flip rich→poor (`ESP_045550_2180`: 0.677 → 0.034); coarse-cohort flip rates are ≤ 4.1 %, median 0.27 %.

Crucially this is **not a per-image multiplicative offset**. The within-image Spearman between the
shipped `fa` and the same tiles' `fa` on the common floor is **0.597–0.979 (median 0.860)** for the
fine cohort against **0.962–0.9998 (median 0.9976)** for the coarse. The size floor changes the *rank
order of tiles inside an image*, so none of the per-image level machinery the project built
(A1 per-frame normalisation, H1 log-median centering, H4 leveling, the `CalibrationLayer`) can absorb it —
they are all monotone per-image maps.

- **Failure scenario:** two adjacent CTX tiles of identical terrain, one under a 0.25 m/px HiRISE
  observation and one under a 0.50 m/px one. The first is labelled rich, the second poor. The head
  sees only CTX, which carries no information about the HiRISE binning, so the discrepancy is
  irreducible label noise entering exactly as a per-image level error — the quantity the
  striping/F programme spent months measuring against a 0.170 dex floor. Downstream, `pr_auc@1e-2`
  and `precision@5%` are prevalence-dependent (invariant 8 names them), and the prevalence of the
  positive class is 2.0× cohort-dependent.
- **Evidence:** holding the **shipped predictions fixed** (`striping_a1_loio_preds.csv`, whose row
  order I verified is bit-identical to `(fractional_area > 1e-2)` in parquet order for 38/38 images)
  and changing only the label's size convention:

  ```
  common floor (m²)   0.0(shipped)  1.5625   2.50    3.50    4.50    6.25
  equiv. diameter (m)      —         1.41    1.78    2.11    2.39    2.82
  ------------------------------------------------------------------------
  prevalence             0.3601     0.3601  0.3534  0.3436  0.3358  0.3226
  pooled meaningful AUC  0.8413     0.8413  0.8448  0.8484  0.8502  0.8523
  pooled PR-AUC@1e-2     0.7770     0.7770  0.7774  0.7776  0.7769  0.7728
  precision@5%           0.9365     0.9365  0.9365  0.9364  0.9363  0.9330
  median per-image AUC   0.7921     0.7921  0.7897  0.7934  0.7934  0.7968
    0.25 m/px median     0.7268     0.7268  0.7388  0.7398  0.7258  0.7702
    0.50 m/px median     0.7990     0.7990  0.7990  0.7990  0.8012  0.8005
  zero share (fa == 0)   0.1836     0.1836  0.1905  0.2045  0.2202  0.2501
  ```

  The committed values in `reports/figures/striping_a1_loio_summary.csv` are
  `median_auc = 0.7903835812655231` and `pooled_pr_auc = 0.7772901307405782`; my shipped-label
  recomputation gives `0.79038` and `0.77729`, i.e. exact. Those two committed numbers become
  **0.7968** and **0.7728** under a 2.82 m common floor. Per-image AUC for an individual fine-cohort
  image moves by **−0.182 (`ESP_045550_2180`) to +0.429 (`ESP_045983_2270`)**.

  ```
  src/labeling.py:374-375   # the published target — no size term anywhere
      tile_area = float(sc["tile_area_m2"])
      frac = ba / tile_area

  src/labeling.py:108-111   # one global scalar; binds for 12 images, inert for 26
      min_size_m = filters.get("min_size_m")
      if min_size_m is not None:
          diam = 2.0 * np.sqrt(gdf.geometry.area.to_numpy() / np.pi)
          keep &= diam >= float(min_size_m)

  dataset/DATA_DICTIONARY.md:164
      | `fractional_area` | float | Derived: `boulder_area / tile_area`. Primary regression
        target. Heavily zero-inflated; ... |        # no minimum boulder size
  ```
- **Self-refutation attempted:**
  (a) **Is my counterfactual an artifact of centroid-assignment rather than rasterisation?** No — the
  same centroid binning reproduces the shipped `boulder_count` with a **per-tile exact-match rate of
  1.0000 on all 38 images at S=64**, and at `floor = 0` it reproduces the shipped rich share to
  0.0003 (0.3601 vs 0.3598). (b) **Is 6.25 m² a strawman?** The sweep above shows the effect is
  smooth and monotone in the floor, and 6.25 m² is the config's own stated 0.50 m/px design floor
  (`config.yaml:83-88`); at the *measured* coarse floor (~3.5 m²) prevalence still moves 0.360→0.344.
  (c) **Is the aggregate headline actually threatened?** No — pooled AUC moves +0.011 and pooled
  PR-AUC −0.004; the fine-cohort per-image deltas have median −0.019, mean +0.031, Wilcoxon p = 0.91,
  i.e. **no systematic direction**. That is why this is medium, not high: the shipped headline is
  robust; the *meaning* of the number is not. (d) **Is it recorded as a known deferral?** The
  asymmetry is (`DECISIONS.md:891`, `:1355-1362`) and R03 says so; what is undocumented is that the
  target's *class definition* — not just its scale — is cohort-dependent, and that it re-ranks.
  (e) **Does a test pin it?** No test in `tests/test_labeling.py` uses two pixel scales.
- **Fix:** treat the size floor as part of the target's definition, not as a filter setting.
  (i) add `size_floor_m2` / `size_floor_diameter_m` (measured, post-filter) and `map_scale_mpp` to
  `dataset_v2/labels/{ObsId}.json`; (ii) state in `DATA_DICTIONARY.md:164` and `docs/methods.md:697`
  that `fractional_area` is "areal fraction of boulders with equivalent-circle diameter ≥ D, D
  per-image, 1.41 m (12 images) / 1.94–2.66 m (26 images)"; (iii) when a cross-image or cross-place
  number is reported, report it *also* on a common floor as a sensitivity, using the table above as
  the template. Do **not** globally re-floor the labels — see semantics-2's fix discussion.

---

### semantics-2 — The deployed abundance layer publishes a physical quantity with no declared minimum boulder size, and the size convention it actually carries is a 78/22 mixture of two different floors that nothing in the product chain records
- **Severity:** medium
- **Liveness:** live-shipped (`*_abundance.tif` is the deliverable) **and** live-active-plan
  (`PLAN_RegionalMap.md` legs 2–4 are the remaining ACTIVE work)
- **Confidence:** high
- **Where:** `src/calibration.py:199-226` (`QuantileMatcher`), `src/mapping.py:263-288`
  (`calibrate_abundance` → `abundance_raster`), `src/mapping.py:181-195` (`write_geotiff`),
  `models/deployable/calibration.npz`, `PLAN_RegionalMap.md:60-66, 203-212, 216-236`

`calibrate_abundance` is a `QuantileMatcher` fitted on `(ref_pred, ref_true)` where `ref_true` is the
cohort's own `fractional_area` distribution. The banked artifact proves it: `calibration.npz` carries
`t2_y` with `max = 0.293242`, which is **exactly** the maximum `fractional_area` over the 161,005
shipped S=32 tiles (`ESP_054622_2240`), and `meta = {"n": 161005, "abundance_source": "p_rich",
"fit": "pooled_loio_38", ...}`. So every value in `reports/map_*/**_abundance.tif` is, by
construction, a quantile of the 38-image label pool — a pool that is **78.4 % tiles whose boulders are
≥ 1.94–2.66 m and 21.6 % tiles whose boulders are ≥ 1.41 m**.

Nothing in the chain states this. `write_geotiff` (`src/mapping.py:181-195`) writes no GDAL tags at
all — no units, no floor, no provenance. `calibration.npz`'s `meta` has recipe/scale/mode/fit and no
floor. `DATA_DICTIONARY.md`'s `fractional_area` row has no floor.
`PLAN_RegionalMap.md:216-236`'s explicit "honest caveats" list — which covers circularity, TI
indirectness, OOD distal plains, co-registration, small cohort and the striping artifact — does not
mention it either.

**Assessment of R03's recommended remedy (d), as the brief asks.** It is *necessary but not
sufficient* for what `PLAN_RegionalMap`'s legs need:
- **Sufficient for legs 1–3** (co-location, TI rank-correlation, shoreline profile). These are
  *within-map* rank statistics on a single deployed head, so the size convention is a constant and
  cancels. R03's sidecar fields would let a reader audit it; nothing more is required.
- **Not sufficient for leg 4** (the LOIO truth anchor, `PLAN_RegionalMap.md:65, 71-77, 191-193`).
  That leg compares held-out predicted abundance against BoulderNet detections **at cohort sites**,
  i.e. explicitly across the two floors. semantics-1 measures what that does: the fine cohort's
  median per-image AUC is 0.7268 vs 0.7990 coarse, and an individual image's number moves by up to
  0.43 under a change of convention. A per-image sidecar field does not fix the leg; the leg needs
  the common-floor sensitivity reported alongside.
- **Not sufficient for the external comparison the plan is built on.** `PLAN_RegionalMap.md:35-46,
  203-212` compares the map to Rodriguez 2016 and to THEMIS/TES thermal inertia. TES-derived rock
  abundance is an areal fraction with its *own* (thermal, ~10 cm) size convention. A per-image label
  sidecar cannot state the deployed layer's floor, because the deployed layer has no image — its
  floor is the *mixture*. What is missing from R03's list is a **product-level** declaration:
  one number (or two, honestly bimodal) attached to the abundance raster itself.
- **Silent on `boulder_count` / `count_density`**, which are 5–10× more distorted (semantics-3).

- **Failure scenario:** `docs/regional_validation.md` reports "predicted rock abundance in the
  boulder band is 0.05–0.10, consistent with TES rock abundance of 0.08 in the same block". The two
  numbers are areal fractions over different minimum sizes and the comparison is not like-for-like;
  neither the GeoTIFF nor any sidecar lets a reader detect this. Because the deployed floor is a
  *mixture*, no consumer can even reconstruct it from the per-image provenance R03 proposes.
- **Evidence:**
  ```
  src/calibration.py:215-221      # ref_true == the cohort's fractional_area distribution
      def fit(self, ref_pred, ref_true):
          q = np.linspace(0, 1, min(len(sp), self.n_quantiles))
          self._xp = np.quantile(sp, q);  self._fp = np.quantile(st, q)

  models/deployable/calibration.npz
      t2_y : shape (4000,)  min 0  max 0.293242      # == max fa over the 161,005 shipped S=32 tiles
      meta : {"n": 161005, "abundance_source": "p_rich",
              "recipe": "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2", "fit": "pooled_loio_38"}

  src/mapping.py:186-194          # no dst.update_tags(...) anywhere: units/floor unrecorded
      with rasterio.open(path, "w", driver="GTiff", ..., nodata=nodata,
                         compress="deflate", tiled=True, ...) as dst:
          dst.write(raster.astype(np.float32), 1)

  measured cohort mix behind that reference distribution:
      0.50 m/px  26 images  126,214 tiles  78.4 %   floor 1.943-2.664 m
      0.25 m/px  12 images   34,791 tiles  21.6 %   floor 1.411-1.427 m
  ```
- **Self-refutation attempted:** (a) checked whether the map writer records anything elsewhere —
  `scripts/map_region.py` writes `*_prob.tif` / `*_abundance.tif` / `*_prob_raw.tif` and no metadata
  sidecar carrying a size convention; (b) checked whether the plan already carries the caveat —
  §7's caveat list is thorough on five other axes and silent on this one; (c) checked whether legs
  1–3 are actually threatened — they are not (rank statistics on one constant convention), which is
  why this is medium and why the fix is a declaration rather than a rebuild; (d) checked whether
  R03's item (d) already covers it — it covers the *label* artifacts (`{ObsId}.json` ×2 and a
  `DATA_DICTIONARY` row) and explicitly motivates them by the cross-place legs, but stops at the
  per-image level and never reaches the deployed product; that is the gap I am filing.
- **Fix:** three cheap additions, none of which re-runs anything. (i) `write_geotiff` →
  `dst.update_tags(quantity="areal boulder fraction", min_boulder_diameter_m="1.41 (22% of training
  tiles) / 1.94-2.66 (78%)", reference="pooled_loio_38 fractional_area quantiles",
  recipe=<recipe_hash>)`. (ii) add the same two keys to `calibration.npz`'s `meta`. (iii) add the
  bullet to `PLAN_RegionalMap.md:216-236`'s caveat list and require `docs/regional_validation.md` to
  state the floor beside every absolute abundance number. **Do not** globally re-floor the labels:
  enforcing 6.25 m² would delete ~72 % of the fine cohort's labelled area (measured: median 0.724,
  range 0.398–0.897) and is a target redefinition, not a tidy-up — R03's verifier reaches the same
  conclusion and I confirm it.

---

### semantics-3 — `boulder_count` / `count_density` are 5–10× more cohort-distorted than `fa`, and the published Stage-7d `boulder_count > 50` rich/poor partition moves 2.5× on the fine cohort
- **Severity:** medium
- **Liveness:** mixed — `count_density` is live-published in every label parquet; the Stage-7d
  partition is live in a reader-facing document (`docs/compositional.md`); `boulder_count` as a
  *modeling target* is dead-closed (the frozen recipe reverted to `fa_gt_1e-2`)
- **Confidence:** high (the centroid binning reproduces the shipped `boulder_count` exactly)
- **Where:** `src/labeling.py:222-251, 376, 391` (`boulder_count`, `count_density`,
  `binary_by_count`); `src/stage7d_pooled.py:36, 92` (`P2_COUNT_THRESHOLD = 50`,
  `is_rich_P2 = boulder_count > 50` @ `SCALE_IDX_S64`); `src/modeling/binary_target.py:52-82`
  (`bc_ge_1`, `bc_ge_50`, `bc_ge_100`); `dataset/DATA_DICTIONARY.md:162, 167`

The size floor bites a *count* far harder than an *area*, because the sub-floor population is
numerous and individually small. Measured over all 5.89 M post-filter polygons:

| | share of counted polygons below 6.25 m² | share of labelled *area* below 6.25 m² |
|---|---|---|
| 0.50 m/px | median **0.015** (range 0.000–0.281) | median 0.006 (0.0000–0.167) |
| 0.25 m/px | median **0.898** (range 0.600–0.970) | median 0.724 (0.398–0.897) |

So for a typical fine-cohort image **~90 % of the boulders that `boulder_count` counts are boulders
the coarse cohort structurally cannot see**, against ~1.5 % the other way. R03 measured the *area*
channel only; this is the same mechanism ~50× larger on the count channel.

That propagates straight into a published partition. `src/stage7d_pooled.py:92` defines the
compositional work's P2 rich class as `boulder_count > 50` at S=64. Recomputed with the same
counting rule on a common 6.25 m² floor:

```
pooled P2 rich share  0.4792 -> 0.3883   (ratio 0.810)
  0.50 m/px           0.4393 -> 0.4290   (ratio 0.977)
  0.25 m/px           0.6208 -> 0.2436   (ratio 0.392)
extremes: ESP_048688_2085 0.794 -> 0.000 | ESP_046328_2180 0.877 -> 0.025
          ESP_045983_2270 0.736 -> 0.014 | ESP_047976_2020 0.043 -> 0.000
```

- **Failure scenario:** any Stage-7-family statement of the form "boulder-rich tiles differ
  compositionally from boulder-poor tiles" is partly a statement about which HiRISE binning observed
  the tile: three fine-cohort images move from ~80 % rich to ~0 % rich under a size convention that
  nobody declared. The same applies to `bc_ge_50` / `bc_ge_100` in `binary_target.py` and to any
  future use of `count_density` as a physical density.
- **Evidence:**
  ```
  src/stage7d_pooled.py:36,92
      P2_COUNT_THRESHOLD = 50
      out["is_rich_P2"] = out["boulder_count"] > P2_COUNT_THRESHOLD

  src/labeling.py:376,391       # published, with no size qualifier
      density = bc / tile_area
      "binary_by_count": bc >= binary_count_threshold,

  dataset/DATA_DICTIONARY.md:162
      | `boulder_count` | int64 | Base stat: number of polygons whose centroid lies inside
        the tile. Unambiguous at borders (each boulder counted once). |
  ```
  Validation of the recomputation: per-tile exact-match against the shipped `boulder_count` at S=64
  is **1.0000 on every one of the 38 images**.
- **Self-refutation attempted:** (a) **is this just R50?** No — R50 is "the `boulder_count` target's
  +22 % win is a change in the positive-class definition *versus `fa`*". This is a different axis:
  the *same* count rule means different things in different images. I do not re-file R50 and this
  does not depend on it. (b) **is it dead?** The modeling target is, but `count_density` ships in
  every parquet, `stage7d_pooled.py` is live `src/` code, and `docs/compositional.md` is reader-facing
  — hence medium rather than low. (c) **is the count rule already known to be scale-fragile?**
  `DECISIONS.md:2745` records a *latitude* distortion on `bc>=50` ("true-density threshold 2.26x") —
  a different mechanism, and it does not mention pixel scale. (d) **could the P2 threshold have been
  chosen with this in mind?** `PLAN_Compositional`/`DECISIONS` record it as a distribution-based cut,
  with no cohort term.
- **Fix:** same provenance as semantics-1/2, plus one line in `docs/compositional.md` §on P2 stating
  that the rule's prevalence is 2.5× cohort-dependent, and — if Stage 7 is ever re-run — apply the
  count rule to polygons above a common floor.

---

### semantics-4 — The only published number that describes the size floor is pinned to "no filtering happened" for exactly the 26 images whose true floor is furthest from the configured one
- **Severity:** low
- **Liveness:** live-shipped (the sidecars ship with the label basis)
- **Confidence:** high
- **Where:** `src/labeling.py:554-558` (`n_polygons_stage1`, `n_polygons_after_filter`,
  `detection_filters`); `dataset/DATA_DICTIONARY.md:175-177`; the Stage-1 sidecar
  `cache_v2/reprojected_detections/{ObsId}.json`; `dataset/DATA_DICTIONARY.md:19`

This is the `peak_correlation` shape (`geo-crs-deep-1`) in the label sidecar. The one place a
consumer could learn anything about the size basis is
`dataset_v2/labels/{ObsId}.json`'s `detection_filters` + the `n_polygons_stage1` /
`n_polygons_after_filter` pair. Both are structurally uninformative exactly where the heterogeneity
is:

- `detection_filters` is a snapshot of the *requested* config, and it is **byte-identical
  (`{"min_confidence": null, "min_size_m": 1.4105}`) across all 38 sidecars** — so a consumer diffing
  sidecars concludes the cohort has a uniform size basis.
- `min_size_m = 1.4105` **is** the 0.25 m/px cohort's own design floor, so the filter binds only for
  the 12 fine images and is inert for the 26 coarse ones. Measured:
  `n_polygons_after_filter == n_polygons_stage1` for **26/26** coarse images and **0/12** fine
  (drop fraction 0.0004–0.0826, median 0.0075). The provenance field therefore reports "nothing was
  filtered" precisely for the images whose actual floor (1.94–2.66 m) is 1.4–1.9× the configured one.
- The *realised* floor is recorded nowhere. The Stage-1 sidecar's full key set is
  `obs_id, n_polygons, n_polygons_raw, n_dropped_null_geometry, source_path, source_mtime_iso,
  source_crs_wkt, target_crs_wkt, config_hash, correction, written_at_iso` — no map scale, no size
  statistic, no score statistic. The Stage-4 sidecar's only scale-ish keys are `tile_sizes_px` /
  `tile_sizes_m` (CTX tile sizes, not boulder sizes).

**EXTENDS R23 (the confidence-basis half of the brief's question, not re-filed).** I checked the
realised `score` minimum of every cached Stage-1 GPKG. **No image other than R23's has a non-uniform
confidence basis**: 36 of the 38 shipped-label images have `score` min exactly `0.100000`; the two
exceptions are R23's (`ESP_017355_2260` 0.617257, `ESP_068483_2280` 0.406699), plus the already-excluded
`ESP_046803_2325` 0.473420. That is a clean negative result. But the floor is recorded nowhere a
consumer could read it, and the one document that appears to record it is **wrong in both
directions**: `dataset/DATA_DICTIONARY.md:19` states `score | float | BoulderNet confidence,
0.10–0.83 in our manifest`, whereas the measured v2 range is **0.100–0.956** and three images'
minima are 0.407 / 0.473 / 0.617. A reader consulting the schema is actively told the basis is
uniform at 0.10.

- **Failure scenario:** a downstream user (or a future session) audits the label basis by diffing
  the 38 sidecars, sees one identical `detection_filters` block and 26 clean `n1 == n2` rows, and
  concludes the cohort is homogeneous — which is the inference `DECISIONS.md:891`'s "leaves the
  0.50 m/px images untouched" invites, and which R03 shows is exactly backwards.
- **Evidence:**
  ```
  src/labeling.py:554-558
      "n_polygons_stage1": int(gdf_pre_filter_n),
      "n_polygons_after_filter": int(n_after_filter),
      "detection_filters": labeling_cfg.get("detection_filters") or {...},

  measured over dataset_v2/labels/*.json (38 sidecars):
      n1 == n2 :  26/26 coarse   0/12 fine
      distinct detection_filters snapshots : 1
      sidecar keys matching /scale|mpp|floor|size/ :
          ['tile_sizes_px','tile_sizes_m','eligible_tiles_per_scale',
           'total_candidate_tiles_per_scale']      # all CTX tile sizes, none boulder size

  measured over cache_v2/reprojected_detections/*.gpkg (39, 7.0 M polygons):
      score min == 0.100000 exactly in 36 of the 38 shipped-label images
  ```
- **Self-refutation attempted:** (a) checked whether `n_polygons_after_filter` is merely redundant
  rather than misleading — `DATA_DICTIONARY.md:176` explicitly glosses it as "equal to
  `n_polygons_stage1` when both are null, the current default", i.e. the doc reads equality as
  "no filter configured", which is *not* the case here (a filter is configured; it just cannot bind);
  (b) checked whether any other artifact carries the scale — `MapPixel_mpp` exists in
  `hirie_40_vclaire.csv` but R03 already established it has no reader; (c) checked whether this is
  just R03 restated — R03 is about the confound; this is about the *provenance field being
  self-fulfilling*, which is a different (and cheaply fixable) defect; (d) kept it **low** because
  no number is wrong today, only unauditable.
- **Fix:** record the measured floor, not the requested one — add `min_polygon_area_m2`,
  `min_polygon_diameter_m`, `median_polygon_area_m2`, `score_min`, `score_max` and `map_scale_mpp`
  to the Stage-1 sidecar (all six are one pass over the GPKG that Stage 1 already has in memory) and
  carry them into `dataset_v2/labels/{ObsId}.json`. Correct `DATA_DICTIONARY.md:19`'s score range.

---

### semantics-5 — Every document that defines the target omits the size floor, and `docs/methods.md` states a `binary_by_count` threshold the shipped configs do not implement
- **Severity:** low
- **Liveness:** live-shipped documents (`README.md` routes external readers to `docs/methods.md`)
- **Confidence:** high
- **Where:** `docs/methods.md:697-700, 750-755`; `dataset/DATA_DICTIONARY.md:161-167`;
  `docs/build_spec.md:127-133, 250`; `CLAUDE.md:7-8`; vs `config.yaml:70-77` / `config_v2.yaml:96-98`

Four documents define the target; **none** of them qualifies it by a minimum boulder size:

| doc | text | size floor stated? |
|---|---|---|
| `docs/methods.md:697` | "the fraction of each tile's area covered by detected boulder polygons" | no |
| `dataset/DATA_DICTIONARY.md:164` | "Derived: `boulder_area / tile_area`. Primary regression target." | no |
| `docs/build_spec.md:127` | "`fractional_area` → `boulder_area / tile_area` (continuous regression target)" | no |
| `CLAUDE.md:7-8` | "per-tile **rock abundance**" | no |

`docs/methods.md` *does* discuss the floor at `:206-230` — but in the Stage-1 audit section, for the
**v1 priority10** cohort ("drops 36 polygons out of 13,352 across the sweep"), 470 lines away from
§6.5 where the target is defined, and it never says the resulting target has a per-image floor.
(R44 covers methods.md's v1/v2 half-migration; this is the separate point that the *target
semantics* section carries no floor at all.)

Separately, `docs/methods.md:753` states `binary_by_count ≡ boulder_count ≥ binary_count_threshold
(default 5, ...)`. Both shipped configs set **1**, and `config.yaml:71-77` documents the 2026-05-27
change from 5 → 1 with its rationale. `docs-consistency.md` verified the `DATA_DICTIONARY` rows
against the code and found them correct; it did not check `methods.md`'s restatement, which is stale.

- **Failure scenario:** an external reader (methods.md is the document `README.md:45-46` sends them
  to) takes "the fraction of each tile's area covered by detected boulder polygons" at face value and
  compares the published abundance to a rock-abundance product with a stated size convention; or
  reproduces `binary_by_count` at threshold 5 and gets a different positive class from every
  artifact in `dataset_v2/`.
- **Evidence:**
  ```
  docs/methods.md:753
      - `binary_by_count` ≡ `boulder_count ≥ binary_count_threshold` (default 5, also
        a placeholder).
  config.yaml:71-77
      # 2026-05-27: lowered 5 -> 1 (binary_by_count now means "any boulder by centroid rule").
      binary_count_threshold: 1
  config_v2.yaml:98
      binary_count_threshold: 1
  ```
- **Self-refutation attempted:** (a) checked whether `binary_by_count` has a consumer that could be
  affected — it does not; it is carried in `src/dataset.py:62` and `src/stage7d_pooled.py:74` column
  lists and never thresholded downstream, which caps this at low; (b) checked whether
  `docs-consistency.md` already filed it — it checked `DATA_DICTIONARY.md:165-166` (correct) and did
  not reach `methods.md:753`; (c) checked whether R44 subsumes it — R44 is about cohort scope
  (v1 tables presented as v2), not about a threshold value.
- **Fix:** one word in `docs/methods.md:753` (5 → 1), and one clause added to the `fractional_area`
  definition in all four documents naming the per-image floor.

---

## Refuted by my own check

- **`fa` saturates against its 1.0 ceiling in dense terrain, censoring the richest tiles (which are
  disproportionately fine-cohort).** No. Over all 161,005 shipped S=32 tiles the maximum
  `fractional_area` is **0.29324** (`ESP_054622_2240`); **zero** tiles reach 0.5. The union
  rasterisation does bound `fa ≤ 1`, but the bound is never approached, so there is no ceiling
  effect and no censoring.
- **The 5× sub-pixel rasteriser systematically loses sub-1 m² boulders, adding a second,
  cohort-dependent downward bias to `fa` and falsifying `docs/methods.md:721-730`'s "zero mean"
  claim.** Refuted, and this **CORRECTS R03's verifier**. Compared on the *same eligible tile set*,
  the rasterised `boulder_area` recovers **99.7 %–100.2 %** of the centroid-assigned polygon-area sum
  (median 0.9989 coarse / 0.9990 fine — **no cohort difference**). R03's verifier's "93–99 %" used a
  whole-image denominator that includes polygons falling in ineligible (mask-gated) tiles; against
  the correct denominator the rasteriser is unbiased to 0.1 % and `methods.md`'s claim stands.
- **The 1 m² sub-pixel quantum pins or perturbs the `fa > 1e-2` threshold.** No: at S=32 the quantum
  is `subpixel_area / tile_area = 1/25600 = 3.9e-5`, i.e. 256 quanta below the threshold. (At S=8 it
  is 16 quanta — worth knowing, but the frozen recipe is S=32.)
- **Some image other than R23's two has a non-uniform confidence basis.** No — 36 of the 38
  shipped-label images have `score` min exactly `0.100000`. Clean negative (see semantics-4).
- **`binary_by_area`'s 0.005 threshold is somewhere mistaken for the frozen `fa > 1e-2`.** No
  consumer: `binary_by_area` appears only in column lists (`src/dataset.py:62`,
  `src/stage7d_pooled.py:74`) and is never used as a target. `src/modeling/binary_target.py`
  re-derives every binary target from `fractional_area` / `boulder_count` at fit time.
- **The per-image-AUC ↔ sub-floor-area correlation is a pixel-scale artifact, i.e. the cohort split
  itself costs skill.** *This is the most important refutation in this pass, and it is why
  semantics-1 is framed as "the number's meaning is undeclared" rather than "the cohort split biases
  the metric".* The correlation is strong and survives every control —
  `Spearman(share of labelled area below 6.25 m², per-image meaningful AUC) = −0.468, p = 0.0030`
  (n = 38); partial-Spearman controlling prevalence −0.474 (p 0.0030), controlling `log10 n_tiles`
  −0.440 (p 0.0064), controlling label level −0.486 (p 0.0023), all three together **−0.492
  (p 0.0027)** — **but it also survives inside the 0.50 m/px cohort alone** (ρ **−0.467, p 0.016**,
  n = 26, over a sub-floor range of only 0–0.167, where pixel scale is constant). So the covariate
  is mostly *small-boulder terrain that 5 m/px CTX cannot resolve*, of which pixel scale is only a
  partial determinant. This independently reproduces R03's verifier's second caveat on a different
  statistic (per-image AUC rather than the F-abort level ratio), at n = 38 rather than n = 21.
- **Re-flooring the labels would systematically improve the fine cohort's scores.** No. Fine-cohort
  per-image AUC deltas span −0.182 to +0.429 with median −0.019, mean +0.031, Wilcoxon p = 0.91.
  Only the pooled/median statistics move, and modestly. Several of the large positive deltas
  (`ESP_045983_2270` 0.558 → 0.988) are low-prevalence artifacts — its positive class shrinks from
  326 tiles to 17.
- **`min_size_m` is compared against latitude-inflated projected areas so the physical floor varies
  with latitude.** True but already on the record (`DECISIONS.md:2741-2751`,
  `scripts/probes/_w1_latitude_distortion.py`) and already in `labeling.md`'s refuted list. Note it
  compounds semantics-1: the fine cohort's nominally-uniform 1.4105 m floor is really ~1.17–1.41 m
  depending on latitude. Not re-filed.
- **`binary_by_area`/`binary_by_count` degenerate under their code defaults (0.0 / 0).** Already
  refuted in `labeling.md`; both configs set 0.005 / 1. Confirmed independently.

## Verified clean

- **The counterfactual machinery itself** (this is what makes every number above trustworthy):
  centroid-binning the cached Stage-1 polygons, applying the shipped `min_size_m` filter and the
  cached Stage-3 shift, reproduces the shipped `boulder_count` with a **per-tile exact-match rate of
  1.0000 on all 38 images at S=64**, and reproduces the shipped pooled rich share to 3e-4
  (0.3601 vs 0.3598) at S=32.
- **`reports/figures/striping_a1_loio_preds.csv` row alignment.** For **38/38** images the `y` column
  is bit-identical to `(fractional_area > 1e-2)` in `dataset_v2/labels/{ObsId}.parquet` row order at
  S=32, and my recomputation reproduces the committed `striping_a1_loio_summary.csv`
  `median_auc = 0.7903835812655231` and `pooled_pr_auc = 0.7772901307405782` exactly. The artifact
  and its summary are mutually consistent and correctly ordered.
- **`fa` is bounded [0, 1] by construction and the bound is honest.** `_rasterize_boulders_subpixel`
  produces a union mask, so overlapping polygons are counted once and `boulder_area ≤ tile_area`;
  `boulder_count` (a sum) is not union-reduced, which is the correct asymmetry given the two rules'
  different definitions.
- **The filters apply identically to `fa`, `boulder_count` and `boulder_area`** — all three derive
  from the single filtered/shifted `gdf` (`src/labeling.py:471-522`). Confirmed independently of
  pass 1.
- **The cohort mix is what the recipe says it is.** `models/deployable/86c51a5dca220f63/recipe.json`
  lists 38 `train_obs_ids`; those are exactly the 38 parquets in `dataset_v2/labels/`, 26 at
  0.50 m/px and 12 at 0.25 m/px, 161,005 S=32 tiles (126,214 / 34,791). Both R03 blanks resolve to
  0.5 m/px from the cached `.LBL`s, as R03's verifier found.
- **The asymmetry is a documented, explicit deferral.** `DECISIONS.md:891` states the choice, its
  rationale and that the 0.50 m/px 5×5 floor is "**not** enforced"; `:1355-1362` records the
  per-image extension as deferred. Nothing here is an undocumented decision — the gap is entirely in
  what the artifacts and reader-facing docs *publish*.

## Coverage note

**Read in full:** `src/labeling.py` (604), `src/stage7d_pooled.py` (label-side), the
`labeling`/`detection_filters` blocks of `config.yaml` and `config_v2.yaml`,
`dataset/DATA_DICTIONARY.md` (all 471 lines), `docs/review_2026-07-31/verify/R03.md`,
`docs/review_2026-07-31/labeling.md`, `docs/review_2026-07-31/geo-crs-deep.md`, and
`PLAN_RegionalMap.md` §2/§5/§6/§7. **Read in part:** `docs/methods.md` §4 (the Stage-1 size audit,
`:180-235`) and §6.3–6.5 (`:690-760`); `docs/build_spec.md` §labels; `src/calibration.py`
(`QuantileMatcher`, `:199-231`); `src/mapping.py` (`:175-290`); `src/modeling/binary_target.py`;
`src/modeling/evaluate.py` (metric surface only); `docs/compositional.md` (P2/S=64 framing);
`docs/CODE_REVIEW_2026-07-31.md` R50/R51/R53 and the R03/R23/R44/R56 index rows;
`docs/review_2026-07-31/docs-consistency.md` (the DATA_DICTIONARY cross-checks).
**Grepped by term:** `DECISIONS.md` for `min_size_m`, `1.4105`, `5x5`, `design floor`, `size floor`,
`binning`, `MapPixel`, `0.25 m/px`; the repo for `boulder_count`, `count_density`, `binary_by_area`,
`fractional_area` consumers.

**Measurements (all read-only; local caches only, no network, no notebooks, no imagery pixels).**
Scripts in the session scratchpad, quoted inline above.
(1) Per-polygon area + centroid + score for **all 39** `cache_v2/reprojected_detections/*.gpkg`
(7.0 M polygons; 5.89 M post-filter across the 38 label images) via `pyogrio` + vectorised `shapely`.
(2) Per-image post-filter size floors, medians and sub-floor area/count shares at 1.5625 / 2.5 / 3.5 /
4.5 / 6.25 m². (3) Per-tile counterfactual labels at S=32 and S=64 by centroid binning against each
parquet's own recovered mosaic origin, with the cached Stage-3 shift applied — validated to an exact
`boulder_count` match. (4) Pooled/median/per-image `meaningful_auc`, `pr_auc@1e-2`, `precision@5%`,
prevalence and zero share on `striping_a1_loio_preds.csv` (both stores, 322,010 rows) under six size
conventions. (5) Partial Spearman (rank-residual) controls and within-cohort splits.
(6) `models/deployable/calibration.npz` key/array inspection and its `t2_y` max against the label
pool's max `fa`. (7) All 38 `dataset_v2/labels/*.json` sidecars and one
`cache_v2/reprojected_detections/*.json` key set. (8) `MapPixel_mpp` from `hirise_40_vclaire.csv`
with the two blanks resolved from `cache_v2/pds_labels/*.LBL` (both 0.5, matching R03's verifier).

**Could NOT check:** (a) the *true* per-image detection floor as BoulderNet defines it — I use the
measured minimum surviving polygon area as a proxy, and the measured coarse minima (1.94–2.66 m
diameter) sit **below** the config's stated 5×5-px design floor (2.5 m at 0.50 m/px), so
"5×5 source pixels" is evidently not a hard polygon-area criterion upstream; the nominal 6.25 m²
common floor is therefore an upper bound on the true convention gap, which is why I report the whole
1.56 → 6.25 m² sweep rather than one number. (b) Whether the head could in principle recover the
cohort split from CTX texture alone (would need a training run — forbidden); I argue it cannot on
physical grounds only. (c) The counterfactual on the *frozen recipe's own* LOIO predictions — no
committed per-tile prediction artifact exists for `fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2` other than
the two A1-comparison stores, so semantics-1's numbers are for those two stores (whose `median_auc`
0.7904 / 0.7664 bracket the recipe's reported 0.7865). (d) Whether `reports/map_*/**_abundance.tif`
carries any GDAL tag written by something other than `mapping.write_geotiff` — I read the writer, not
the rasters.
