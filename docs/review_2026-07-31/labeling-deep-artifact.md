# Review area: labeling-deep-artifact

- **Reviewed at commit:** 7bfedb8
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified)

> **One-line answer to the area question.** For the **v2 cohort that the frozen recipe and the shipped
> mosaic map are built on, YES** — `dataset_v2/labels/*.parquet`, the `loio_nfold` and
> `within_image_4fold` split JSONs, their packaged folds, and the abort artifact's label column are all
> **bit-identical** to what today's code produces from today's inputs (proved below, 3,564,767 rows,
> 0 differing). The staleness is in the **neighbouring** artifacts: the whole v1 `dataset/` tree is a
> pre-y-sign-fix generation, and two of the four v2 packaged schemes carry pre-fix labels while their
> provenance fields claim they are in sync.

> **⚠ Concurrency note, read this first.** At 2026-08-04 14:26:35 PDT — *during* this review, and not by
> me (I am read-only) — another process regenerated `dataset/labels/ESP_069669_2220.{json,parquet}` and
> `cache/reprojected_detections/ESP_065711_1545.{gpkg,json}`. I captured the *pre*-regeneration state at
> the start of this session and re-captured it afterwards; both snapshots are quoted in
> `labeling-deep-artifact-1`. The regeneration is not a pipeline defect — but it is what turns
> finding 1 from "stale" into "**stale and now internally mixed**", and it is an accidental
> proof-by-execution that today's code produces the opposite `dy` sign from what was on disk.

## Findings

### labeling-deep-artifact-1 — The entire v1 `dataset/` label tree is a pre-2026-06-10 generation: every sidecar records the **sign-inverted** `dy`, while the Stage-3 cache it points at was migrated to the corrected sign — so today's code cannot reproduce these labels, and one image has now silently been replaced with a post-fix one
- **Severity:** medium
- **Liveness:** live-shipped (the v1 baseline is `docs/modeling_results.md` §§1–8, which `README.md`/`docs/index.md` route external readers to; the go-forward recipe and the shipped map are v2 and unaffected)
- **Confidence:** high (measured from the artifacts twice, before and after the mid-session regeneration)
- **Where:** `dataset/labels/*.json` (9 files) vs `cache/coregistration/*.json` (9 files);
  migration `scripts/probes/_w1_migrate_coreg_sign.py:11` (`DIRS` **includes** `cache/coregistration`);
  fix `src/coregister.py:281-298`; consumer `src/labeling.py:474-475`;
  the record `DECISIONS.md:2577` ("Stage 4 re-run, all 38 **v2** images") — silent on v1;
  reproducibility claim `docs/modeling_results.md:862-867`

The 2026-06-10 W1 rung-1 migration rewrote **all 48** cached coregistration JSONs — `cache/`, `cache_v2/`
and `cache_v2_dev/` — flipping `shift_m.dy` to `-dy_px * px_y`. Stage 4 was then re-run for **v2 only**.
So `cache/coregistration/` now holds the corrected shifts while `dataset/labels/` still holds labels
built from the *old* ones: every v1 label field sits `2·|dy|` = **236 – 493 m** (1.5 – 3.1 tiles at
S=32) south of its CTX texture, which is exactly the defect the fix removed from v2 and which DECISIONS
measured as worth cohort-mean meaningful-AUC 0.598 → **0.624** there. On v2 the label mtimes are
`18:19–18:26` on 2026-06-10, one minute after the migration at `18:18:31`; on v1 eight files are from
2026-05-23 `15:37` and the ninth was written 2026-06-10 `17:32` — **46 minutes before** the migration.

- **Failure scenario:** anyone re-running `run_stage4.py --config config.yaml` (or `run_stage5.py`, which
  repackages from whatever `dataset/labels/` currently holds) gets a materially different v1 dataset with
  no warning, and `docs/modeling_results.md:862-867`'s claim that re-running "reproduces the numbers in
  this document" quietly stops holding. That has now *begun*: after the 14:26 regeneration the v1 label
  tree is **mixed** — 8 pre-fix images + 1 post-fix image — so a repackage today would emit a dataset in
  which one image's labels are registered correctly and eight are not. Substantively, the v1 "5 m/px CTX
  signal floor" verdict (AUC ≈ 0.55) that motivated the whole vClaire v2 rebuild was measured on
  mis-registered labels and, unlike the v2 W0 verdicts, was never re-measured after the fix.
- **Evidence:**
  ```
  # BEFORE the mid-session regeneration (my first read of dataset/labels/*.json):
  ESP_039820_1750  2026-05-23T22:37:24Z  dy=-147.0   ESP_055714_2270  2026-05-23T22:37:22Z  dy=-239.0
  ESP_047976_2020  2026-05-23T22:37:23Z  dy=-125.2   ESP_056165_2200  2026-05-23T22:37:24Z  dy=-152.0
  ESP_054857_2270  2026-05-23T22:37:23Z  dy=-118.0   ESP_065711_1545  2026-05-23T22:37:24Z  dy=-220.0
  ESP_071093_2210  2026-05-23T22:37:23Z  dy=-246.5   ESP_075577_2105  2026-05-23T22:37:24Z  dy=-157.5
  ESP_069669_2220  2026-06-11T00:32:10Z  dy=-239.75           <-- 17:32 PDT, 46 min pre-migration

  # the Stage-3 cache those labels point at, migrated 2026-06-10 18:18:31 PDT (all 9):
  cache/coregistration/ESP_069669_2220.json  dy_m=+239.75  dy_px=-48.0  y_sign_fix_applied=2026-06-10
  cache/coregistration/ESP_039820_1750.json  dy_m=+147.00  dy_px=-29.4  y_sign_fix_applied=2026-06-10   (... 9/9)

  # AFTER the mid-session regeneration (same file, mtime 2026-08-04 14:26:35 PDT):
  dataset/labels/ESP_069669_2220.json:  "dy": 239.74878038511508,   "written_at_iso": "2026-08-04T21:26:35Z"
  #  -> today's code, same cache, produces the OPPOSITE SIGN. The other 8 are untouched.

  scripts/probes/_w1_migrate_coreg_sign.py:11
      DIRS = [Path("cache/coregistration"), Path("cache_v2/coregistration"), ...]   # v1 WAS migrated
  DECISIONS.md:2577
      "- Stage 4 re-run, all 38 v2 images (apply_coreg_shift=True, now-correct shifts);"
      #  ^ v1 is never mentioned, before or after
  ```
  Measured impact of the fix on the one image that has now been redone (packaged 2026-05-23 y vs the
  labels regenerated today): S=8 `fa` differs on 1,484/72,821 rows (2.0 %), S=32 on **776/4,428 (17.5 %)**,
  S=64 on 457/1,062 (43.0 %); rich/poor flips 0.43 % / 0.23 % / 0 %. That image is the *sparsest* in v1
  (1,462 polygons, mean `fa` ≈ 1e-4), so it is a **lower bound** on the eight remaining ones.
- **Self-refutation attempted:** (a) *"v1 is retired, so it does not matter."* It is not retired in the
  reader-facing docs — `docs/modeling_results.md` §§1–8 are v1, `docs/methods.md` is priority10
  throughout, and `README.md:55/175/226` still lists the v1 path. (b) *"The published v1 numbers are
  wrong then."* No — and I checked: `dataset/packaged/{loio_9fold, loio_3fold_balanced,
  within_image_4fold}` were all written 2026-05-23/27 from the *same* stale labels, so they are
  internally self-consistent and every published v1 number reproduces from them. This is a
  reproducibility/coherence defect, not an arithmetic error (same conclusion R44's verifier reached for
  methods.md). (c) *"Maybe the negative dy was correct for v1."* No: `src/coregister.py:298` returns
  `-dy_px * px_y`, every v1 `dy_px` is negative, and the migrated cache says `+`. (d) *"`dataset/` is
  gitignored, so it is not really an artifact."* `.gitignore:15` does exclude it, which caps the blast
  radius to this machine — but it is the only copy of the dataset the docs' reproducibility claim points
  at. (e) I grepped `DECISIONS.md` for `y_sign`, `sign error`, `rung-1` and read `2529-2640` in full
  looking for a deliberate "leave v1 alone" decision. There is none — the entry simply says v2.
- **Fix:** decide and *record* one of two things. Either (i) re-run Stage 4 + Stage 5 for the 9 v1 images
  so `dataset/` is coherent, re-derive the §§1–8 numbers, and note the delta; or (ii) declare v1 frozen
  at the pre-fix generation, stamp that in `dataset/labels/*.json` (a `label_generation` /
  `y_sign_fix_applied` field) and in `docs/modeling_results.md` §§1–8, and revert the one image that was
  regenerated today so the tree stops being mixed. Doing neither leaves a tree that changes under a
  re-run. Independently: `stage4_one_image` should copy the shift record's `y_sign_fix_applied` marker
  into the label sidecar so the generation is legible from the artifact.

---

### labeling-deep-artifact-2 — Two of the four packaged v2 label artifacts carry pre-sign-fix labels (65 % / 88 % of `fractional_area` values differ; 19.4 % / 13.6 % of tiles flip the frozen `fa > 1e-2` class) — and the two provenance fields a consumer would check say they are *in sync*, while the two that are genuinely current say they are *not*
- **Severity:** medium
- **Liveness:** live-shipped (the artifacts and the scripts that consume them are on disk and runnable; the *verdicts* they produced are dead-closed)
- **Confidence:** high (measured row-by-row against the current labels)
- **Where:** `dataset_v2/packaged/loio_nfold_ctx_illum/`, `dataset_v2/packaged/loio_nfold_nbr_s5/`
  (+ their `metadata.json` and `dataset_v2/splits/loio_nfold_{ctx_illum,nbr_s5}.json`);
  consumers `scripts/probes/_sweep_stage6b.py:63-64`, `scripts/probes/_sweep_w0.py:9`;
  the partial record `DECISIONS.md:2870-2873`; the documented-but-useless detector
  `dataset/DATA_DICTIONARY.md:464`

`loio_nfold` and `within_image_4fold` were repackaged after the y-sign fix (2026-06-11 14:24/14:28) and
are perfect. The two Stage-6 side schemes were not, and their `y_*_fold*.parquet` still hold the
2026-05-30 / 2026-06-10-15:21 pre-fix targets. The provenance is worse than absent, it is **inverted**:
both stale packages record `config_hash = 343f0624…`, which is *exactly the hash the label sidecars
carry*, and a `split_hash` that matches their split JSON — i.e. both documented consistency checks pass.
The two **correct** packages record `config_hash = f0d2e71d…`, which *differs* from the labels'. A
consumer applying the rule in `DATA_DICTIONARY.md:464` ("mismatch indicates the package and split
metadata are out of sync") gets precisely the wrong answer on all four.

- **Failure scenario:** `scripts/probes/_sweep_stage6b.py` compares `loio_nfold` (post-fix) against
  `loio_nfold_ctx_illum` (pre-fix); `scripts/probes/_sweep_w0.py --scheme loio_nfold_nbr_s5` does the
  same for the neighbour-stencil arm. Both comparisons were *valid when run* (in 2026-05/06 both arms
  were pre-fix, so the delta sat on a common basis) — but re-running either **today** silently pits a
  correctly-registered baseline against a 480-m-displaced treatment arm and attributes the difference to
  the feature family. Related: the Stage-6a `nbr_s5` verdict at `DECISIONS.md:2506-2515` is the one item
  in the 2026-06-10 W0 entry that was *not* re-checked on corrected labels (the rung-1 entry at
  `DECISIONS.md:2607-2612` re-checks P1/P2/hurdle/P5 and stops there).
- **Evidence:**
  ```
  # y_test_fold*.parquet joined to dataset_v2/labels on (obs_id, tile_size_px, ti, tj):
  loio_nfold            3,564,767 rows | fa differs        0 (0.000 %)  fa>1e-2 flips      0   cfg=f0d2e71d
  within_image_4fold    3,564,767 rows | fa differs        0 (0.000 %)  fa>1e-2 flips      0   cfg=f0d2e71d
  loio_nfold_ctx_illum  3,564,767 rows | fa differs 2,332,350 (65.428 %) flips 690,971 (19.383 %) cfg=343f0624
  loio_nfold_nbr_s5       161,005 rows | fa differs   142,090 (88.252 %) flips  21,894 (13.598 %) cfg=343f0624
  #                                                                     max |Δfa| 0.396 / 0.200
  # dataset_v2/labels/*.json all record: config_hash = 343f0624f126ead9…   <-- matches the STALE two

  DECISIONS.md:2870-2873
      "- Stage 5 `--all` repackaged `loio_nfold` + `within_image_4fold` ...
         NOTE: the stage-6 side schemes `loio_nfold_ctx_illum` / `loio_nfold_nbr_s5`
         were NOT refreshed (built by their own repackage scripts; not in the recipe)."

  scripts/probes/_sweep_stage6b.py:63-64
      ("within_image_4fold", "within_image_4fold_ctx_illum"),
      ("loio_nfold",         "loio_nfold_ctx_illum"),        # post-fix vs pre-fix, today
  ```
- **Self-refutation attempted:** (a) *Is it already documented?* Partly — `DECISIONS.md:2872` says the
  two schemes "were NOT refreshed", which is why this is **medium** and not high. But that note is about
  the 2026-06-11 `patch_idx` repackage; nothing anywhere states that their **targets** are the pre-fix
  generation, and nothing quantifies it. The 19.4 % class-flip rate at the project's own reporting
  threshold is the number a reader needs and does not have. (b) *Does it invalidate the Stage-6a/6b
  verdicts?* No — I checked the run timestamps: `models/_sweep_w0/20260610T223114Z` (baseline) and
  `…223410Z` (nbr_s5) are both 2026-06-10 **22:31/22:34 UTC = 15:31/15:34 PDT**, i.e. both before the
  18:18 PDT migration, and `models/_sweep_stage6b/20260531T*` predates it entirely. The comparisons were
  apples-to-apples at the time. The hazard is prospective, not retrospective. (c) *Are the folds
  themselves wrong?* No — the LOIO partition is identical across all three v2 LOIO split JSONs and all
  three packaged `metadata.json` (verified). Only the `y` values are stale. (d) *Is this just R04?*
  R04 is about `run_stage5.py` swallowing a build failure and about staleness being *undetectable*; this
  is a measured instance plus the specific finding that the detector is **anti**-correlated with truth.
- **Fix:** two lines of provenance plus a decision. Add the label-generation identity to
  `package_split`'s metadata — the simplest sufficient key is a hash over the source label parquets'
  `written_at_iso` (or over `y`'s content), not `config_hash` — and have `_sweep_stage6b.py` /
  `_sweep_w0.py` refuse to compare two schemes whose label-generation keys differ. Then either
  repackage the two side schemes or delete them.

---

### labeling-deep-artifact-3 — `labeling-2` measured on the cached masks: the swath-edge strip is **337 of 161,005** S=32 tiles (0.21 %), **1.16 %** of the zero class — pass 1's analytic estimate (~2 % of tiles, ~60 tiles per image) is ~7× too high
- **Severity:** low (this is a **correction to pass 1's `labeling-2`**, filed here because the brief asked for it; it *strengthens* the existing "low" rating rather than adding a defect)
- **Liveness:** live-shipped (the mechanism is real and in the shipped labels; only the magnitude changes)
- **Confidence:** high — the measurement is self-validating (see below)
- **Where:** `src/labeling.py:474-478` (polygons shifted, mask not), measured against
  `cache_v2/ctx_windows/*_hirise_mask.tif` (38 rasters) + `dataset_v2/labels/*.parquet`

A tile can contain a detection only if it lies inside the HiRISE support region **translated by the same
`(dx, dy)` the polygons got**. I rebuilt that translated support from each cached mask and intersected it
with the eligible S=32 tile set taken straight from the label parquet. Tiles fully outside it are zero by
construction. The measurement validates itself three ways: (i) for all 161,005 eligible tiles the
*un*-translated mask is 1 everywhere inside the tile — my grid arithmetic reproduces the labeller's
eligibility rule exactly, 0 mismatches; (ii) **0 of the 337** structurally-zero tiles has `fa > 0`;
(iii) **0 of the 337** has `boulder_count > 0`. Any error in the geometry would break (ii) or (iii).

- **Corrected numbers (S=32, the frozen recipe's scale, all 38 images):**
  | quantity | measured |
  |---|---|
  | eligible tiles | 161,005 |
  | zero-`fa` tiles | 28,991 (18.01 %) |
  | **zero by construction** (fully outside the translated support) | **337 (0.209 % of tiles; 1.162 % of the zero class)** |
  | per image | mean 8.9, median **2**, max 74 (`ESP_076499_1160`); **17 of 38 images have none** |
  | partially outside (depressed, not necessarily zero) | 5,865 (3.64 %); 950 of them are zero |
  | zero class attributable to the shift at all (337 + 950) | 4.44 % |
  | shift magnitude | min 79.9 m, median 200.1 m, max 327.3 m = 0.03 / 1.16 / 1.78 S=32 tile rows |

  Per-image share of that image's own zero class ranges 0 – 53.3 %; the high ratios are all
  boulder-*rich* images with a tiny zero class (`ESP_071093_2210`: 24 of 45 zeros; `ESP_076565_2215`:
  28 of 124), not images with many spurious zeros.
- **Why pass 1's estimate was high:** it reasoned from swath width × 1–2 tile rows. Two effects shrink it
  an order of magnitude — the `coverage == 1.0` eligibility rule has *already* eroded the ragged mask
  boundary before the shift is considered, and the shift is only 1.16 tile rows at the median, so a
  32-px tile clears the translated boundary completely only when the phase happens to line up.
- **Evidence:**
  ```python
  # scratchpad/a4_mask_strip.py  (read-only)
  a = int(round(dy / py)); b = int(round(dx / px))          # mask_shifted[r,c] = mask[r+a, c-b]
  msh[r_lo:r_hi, c_lo:c_hi] = mask[r_lo+a:r_hi+a, c_lo-b:c_hi-b]
  r0 = ti*32 - prov["mosaic_row_origin"];  c0 = tj*32 - prov["mosaic_col_origin"]
  orig_min[k] = mask[r0:r0+32, c0:c0+32].min()   # == 1 for all 161,005 eligible tiles  (validation)
  sh_max[k]   = msh [r0:r0+32, c0:c0+32].max()   # == 0  -> zero by construction
  # totals: struct_zero 337 | struct_zero with fa>0: 0 | with boulder_count>0: 0 | orig_min!=1: 0
  ```
- **Self-refutation attempted:** (a) *Is the "translated support" the right object?* Detections exist only
  where HiRISE observed; `_apply_coreg_shift` translates them rigidly; so the post-shift detection support
  is exactly `mask ⊕ (dx, dy)`. (b) *Sub-pixel shifts.* `dy_px` is a block median and is fractional; I
  round to whole pixels, a ±2.5 m error against a 160 m tile. (c) *Am I over-counting by including the
  partial tiles?* I report them separately and headline only the fully-outside count. (d) Does this change
  `labeling-2`'s severity? It stays **low** — more confidently so.
- **Fix:** unchanged from pass 1 (erode the eligible mask by `ceil(|shift|/px)`, or translate the mask).
  With 337 tiles at stake this is a hygiene fix, not a priority. **Update `labeling.md`'s magnitude
  paragraph and its "could not check" note.**

---

### labeling-deep-artifact-4 — `config_hash` cannot detect label staleness in either direction, and nothing reads it: it changes when nothing relevant changed (all 38 v2 sidecars now mismatch the current config) and stays fixed across the one change that mattered (the y-sign fix is code, not config)
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `src/config.py:234-241` (`config_hash` over the **whole** raw dict);
  written at `src/labeling.py:538, 588`, `src/dataset.py:459, 502, 712, 830`,
  `src/features.py:786, 828`, `src/coregister.py:430`, `src/ctx_retrieve.py:621`,
  `src/detections.py:166` — **read for comparison nowhere**

`config_hash` is documented as provenance "so downstream stages can detect when their inputs were
generated under a different config" (`src/config.py:238-239`). It cannot serve that purpose here.
It hashes the entire config, so the `validation_rasters:` block added on 2026-06-17/18 (commits
`79b6431`, `5ece95c`, `0c0f12c`) and the `features.context_patch` toggle of 2026-06-11 (`b295bc4`)
each moved it, even though `git diff 3c2117e HEAD -- config_v2.yaml` shows **zero** changes inside the
`labeling:` block. Result: all 38 v2 label sidecars record `343f0624…` against a current
`ed28b9dd…` — a guaranteed, meaningless mismatch that trains any reader to ignore the field. In the
other direction it is blind to exactly the drift that mattered: the y-sign fix changed
`src/coregister.py`, not the YAML, so the pre- and post-fix v2 labels would have carried the *same*
hash. A repo-wide grep finds 20 write sites and **no** site that compares a stored hash to a current one.

- **Failure scenario:** a future session (or the next reviewer) opens `dataset_v2/labels/*.json`, sees
  `config_hash` ≠ current, and either regenerates 3.5 M rows for nothing or — the likelier outcome, and
  the one that already happened in finding 2 — concludes the field is noise and stops checking it, at
  which point the one artifact pair whose hashes genuinely agree is the stale one.
- **Evidence:**
  ```
  current  config.yaml    -> 958fdc25e828feb9…      current  config_v2.yaml -> ed28b9ddccb741ff…
  dataset_v2/labels/*.json  (38/38)  -> 343f0624f126ead9…      # spurious mismatch
  dataset/labels/*.json     ( 8/ 9)  -> e9962e9418a759e9…  (1/9 -> 958fdc25…, regenerated today)
  git diff 3c2117e HEAD -- config_v2.yaml   ->  only the `validation_rasters:` block  (labeling: unchanged)

  src/config.py:240
      canonical = json.dumps(cfg, sort_keys=True, default=str, separators=(",",":"))  # WHOLE config
  # grep -rn 'config_hash' src/ scripts/  -> 20 write sites, 0 comparison sites
  ```
- **Self-refutation attempted:** (a) *Maybe a notebook or test compares it.* Grepped `src/`, `scripts/`,
  `scripts/probes/`; every hit is a write, a column-exclusion set (`src/dataset.py:599`,
  `src/spatial_features.py:37,47`, `src/modeling/loaders.py:91`) or a docstring. (b) *Maybe a stage-scoped
  hash would be over-engineering.* It would not — `labeling.stage4_one_image` already receives
  `labeling_cfg` and snapshots six of its fields individually into the sidecar; hashing that same dict
  is a one-line change. (c) *Is this just cosmetic?* On its own, yes — which is why it is **low**. It is
  filed because it is the mechanism behind finding 2's inverted provenance.
- **Fix:** hash the *stage-relevant* subtree (`config_hash(cfg["labeling"])` for Stage 4) instead of the
  whole config, and add a code-identity component — the cheapest sufficient one is the git rev, or a hash
  of the producing module's source — so a code-only fix like the y-sign migration invalidates downstream
  artifacts. Then have one consumer actually compare it and warn.

---

### labeling-deep-artifact-5 — `dataset/DATA_DICTIONARY.md` still defines `shift_m.dy` with the **pre-fix, sign-inverted formula**, and its Stage-5 section predates the within-image scheme it ships
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `dataset/DATA_DICTIONARY.md:134` vs `src/coregister.py:298`; also
  `dataset/DATA_DICTIONARY.md:400,402` vs the shipped `dataset_v2/splits/within_image_4fold.json`

Line 134 reads ``| `shift_m.dy` | float | `dy_px * abs(ctx_transform.e)` — translation in metres |``.
That is verbatim the W1 rung-1 bug. `src/coregister.py:298` returns `float(-dy_px * px_y)`, and the
function's own docstring (`:292-296`) warns that getting the sign wrong "inverts the y-correction and
doubles the misalignment". So the repo's schema reference for the coregistration artifact documents the
formula the code was fixed to stop using — and it is the *only* place a reader would go to interpret the
`dy` in the label sidecars that findings 1 and 2 turn on. R44's verifier found the same vintage problem
in `docs/methods.md` §5 (a *table of dy values*); this is distinct and worse in kind, because it defines
the field. Secondary, same file: line 400 says split `kind` is "Always `"leave-image-out"` today" and
line 402's `stratification` row lists only `none` / `boulder_label_size_balanced`, but
`dataset_v2/splits/within_image_4fold.json` ships `kind: "within-image"`,
`stratification: "within_image"`, and `src/dataset.py:448-460` emits four fields
(`n_folds_per_image`, `buffer_tiles`, `excluded_obs_ids`, plus the per-fold `quadrant_definitions` /
`n_{test,train}_tiles_per_scale`) that the table does not mention.

- **Failure scenario:** someone reading the dictionary to interpret or re-derive a cached shift
  reintroduces the sign error, or reads `dy = +183.0` in `dataset_v2/labels/ESP_017355_2260.json` as
  "southward 183 m" and concludes the v2 labels are the mis-registered ones.
- **Evidence:**
  ```
  dataset/DATA_DICTIONARY.md:134
      | `shift_m.dy` | float | `dy_px * abs(ctx_transform.e)` — translation in metres |
  src/coregister.py:292-298
      "...rows increase as world y decreases ... so the row component must flip sign ...
         Getting this wrong inverts the y-correction and doubles the misalignment"
      return float(dx_px * px_x), float(-dy_px * px_y)
  dataset/DATA_DICTIONARY.md:400   | `kind` | str | Always `"leave-image-out"` today |
  dataset_v2/splits/within_image_4fold.json   "kind": "within-image", n_folds 152
  ```
- **Self-refutation attempted:** (a) *Is this R44?* No — R44 is `docs/methods.md`; its verifier's note
  ("§5's `dy` column predates the fix") is about a values table in a different document. I checked
  whether `DATA_DICTIONARY.md` was touched by the fix commit: it was not. (b) *Is the dictionary dead?*
  `CLAUDE.md`'s Pointers section lists it as **the** "Output column dictionary", so it is the canonical
  reference. (c) *Is `shift_m.dx` also wrong?* No — line 135 (`dx_px * abs(a)`) matches the code.
- **Fix:** one character plus a note on line 134 (`-dy_px * abs(ctx_transform.e)`, with a pointer to
  DECISIONS 2026-06-10 and the `y_sign_fix_applied` marker), and update lines 400/402 plus the
  within-image fields in the Stage-5 table.

## Refuted by my own check

- **"The LOIO splits have `within_image_4fold`'s vintage problem (R45)."** They do **not**, and
  structurally cannot. `_assign_loio_9fold` (`src/dataset.py:321-328`) returns `[[obs] for obs in
  sorted(inventory.index)]` — fold *i* is `sorted(obs_ids)[i]`, with no dependence on label content, so
  the only drift vector is a change in the image set. Verified: fold → `test_obs_ids` is byte-identical
  across all three v2 LOIO split JSONs (`loio_nfold`, `loio_nfold_ctx_illum`, `loio_nfold_nbr_s5`) and
  all three packaged `metadata.json`, and equals `[(sorted(obs)[i],) for i]`. The quadrant-cut drift R45
  found is specific to `_compute_quadrant_definitions`, which takes a **median over the label rows** and
  therefore moves whenever the labels move.
- **"`dataset_v2/splits/within_image_4fold.json` is itself stale."** No. Rebuilding it with today's
  splitter on today's labels reproduces `split_hash = 5a03892c2439beae…` exactly; quadrant definitions
  differ in **0 of 152** folds and **0 of 161,005** S=32 tiles change quadrant. Same for `loio_nfold`
  (`split_hash = 9df749096c1a09c6…`, 0 of 38 folds differ). R45's finding is that the *sweep* consumed an
  earlier vintage — the file on disk today is current.
- **"The 38 v2 label sidecars' `config_hash` mismatch means the labels are stale w.r.t. `config_v2.yaml`."**
  They are not: `git diff 3c2117e HEAD -- config_v2.yaml` is entirely the `validation_rasters:` block;
  every field of the `labeling:` block (`grid_anchor`, `tile_sizes_px [8,16,32,64]`, `label_type`,
  `binary_area_threshold 0.005`, `binary_count_threshold 1`, `categorical_bins []`,
  `detection_filters {null, 1.4105}`) is unchanged and is echoed identically in all 38 sidecars. See
  finding 4 for why the hash moved anyway.
- **"The Stage-6a `nbr_s5` and Stage-6b `ctx_illum` verdicts are invalid because their packages hold
  pre-fix labels."** Refuted for the verdicts as *recorded*: both arms of each comparison were packaged
  before the fix (`_sweep_w0/20260610T223114Z` baseline and `…223410Z` nbr_s5 are 15:31/15:34 PDT vs the
  18:18 PDT migration; `_sweep_stage6b/20260531T*` is earlier still), so each delta sits on a common
  basis. The live hazard is prospective only — see finding 2.
- **"`reports/figures/fbuild_abort_level_vs_labels.csv` is built on a stale label generation, so the
  abort verdict is suspect."** It is not. Its `label_mean` reproduces today's S=32 mean `fa` for **all
  21** images to ≤ 7.3e-14 relative, and its `n` matches the eligible S=32 tile count exactly for all 21.
  The abort comparison's label side is current. (R12's "no producer" point stands; the *content* checks
  out.)
- **"`dataset/labels/ESP_069669_2220.*` having today's mtime is a pipeline bug."** No — it is a
  concurrent review session running Stage 4 (alongside `cache/reprojected_detections/ESP_065711_1545.*`
  and ten `docs/review_2026-07-31/verify/R*.md` files, all written 2026-08-04 14:19–14:26). Reported in
  finding 1 as evidence, not as a defect.
- **"A derived cache somewhere holds a stale copy of the target."** I enumerated **every** parquet under
  `dataset/` and `dataset_v2/` (excluding `labels/`) that carries any of `fractional_area`,
  `boulder_count`, `boulder_area`, `tile_area`, `binary_by_*`, `count_density`. The complete list is the
  seven `packaged/*/` directories — `features/`, `features_colour*`, `features_ctx_illum/`,
  `features_nbr_s5/`, `fang_embeddings*/`, `context_patches/`, `crater_distance_v2`,
  `terrain_classification_v2`, `w1_dossier`, `stage7d_*`, `modeling_slim_*` carry **no** label copy, so
  they cannot go stale in this way. Findings 1 and 2 cover the seven that can.
- **"A v2 label is older than one of its own inputs."** None is: for all 38 images the label parquet is
  newer than its `reprojected_detections/*.gpkg`, `ctx_windows/*.tif`, `*_hirise_mask.tif` and
  `coregistration/*.json`. (For v1, 8 of 9 fail this test — finding 1.)

## Verified clean

- **`dataset_v2/labels/*.parquet` is exactly what today's code emits.** All 3,564,767 rows across 38
  images are self-consistent with their own sidecar and with `_flatten_to_dataframe`:
  `fractional_area == boulder_area/tile_area` (0 mismatches, exact float equality),
  `binary_by_area == fa >= 0.005`, `binary_by_count == boulder_count >= 1`,
  `count_density == boulder_count/tile_area`, `tile_area == tile_size_px² · px²` (≤1e-6),
  `0 ≤ fa ≤ 1`, every row's `config_hash` equals the sidecar's, the scale set equals
  `tile_sizes_px`, and the per-scale row counts equal `eligible_tiles_per_scale` exactly. Same check on
  all 643,910 v1 rows: also 0 defects (v1's problem is the *shift*, not the derivation).
- **The label parquet schema matches `dataset/DATA_DICTIONARY.md` exactly** — 18 columns, none
  documented-but-missing, none emitted-but-undocumented, and the dtypes are as documented (`ti`/`tj`
  `int64`, `boulder_count` `int64`, `binary_by_*` `bool`, the rest `double`/`large_string`).
  `categorical` is correctly absent (`categorical_bins: []`).
- **`dataset_v2/packaged/loio_nfold` and `within_image_4fold`** reproduce the current labels
  **bit-for-bit**: 0 of 3,564,767 `fractional_area` values and 0 `boulder_count` values differ, 0 keys
  missing from the labels, 0 rich/poor class flips.
- **All 38 v2 sidecars record identical, current filter provenance:** `min_confidence: null`,
  `min_size_m: 1.4105`, `binary_area_threshold: 0.005`, `binary_count_threshold: 1`,
  `tile_sizes_px [8,16,32,64]`, `grid_anchor: ctx_pixel_origin`,
  `eligibility_rule: coverage_equals_one`, `subpixel_factor: 5`, `coreg_shift_applied: true` — matching
  `config_v2.yaml:93-107` field for field. No image has a divergent threshold.
- **All 38 v2 `coreg_shift_m.dy` are positive (northward) and equal their migrated cache entries**;
  all 39 `cache_v2/coregistration/*.json` carry `y_sign_fix_applied: 2026-06-10`. The v2 label tree is
  post-fix throughout.
- **The eligibility rule I reconstructed from the sidecar's `mosaic_row_origin`/`mosaic_col_origin` and
  the parquet's `(ti, tj)` reproduces the labeller's eligible set exactly** on all 38 masks — 161,005 of
  161,005 S=32 tiles have `mask == 1` at every one of their 1,024 pixels, 0 exceptions. This
  independently confirms the grid-alignment arithmetic (`src/labeling.py:117-179`) against the artifacts.
- **`reports/figures/fbuild_abort_level_vs_labels.csv`** — `label_mean` and `n` match today's S=32
  labels for all 21 images (max relative difference 7.3e-14; 0 tile-count mismatches).
- **`dataset_v2/features/`, `context_patches/` (2026-06-11 12:03–12:15) and `fang_embeddings*/`
  (2026-06-12)** are all newer than the post-fix labels (2026-06-10 18:19–18:26), so no v2 feature or
  embedding cache mirrors a pre-fix tile set.

## Coverage note

**Read in full:** `src/labeling.py` (605), `src/dataset.py:70-523` (inventory, splitting, quadrant
definitions, `_split_metadata_hash`, `build_split`), `src/config.py` (242),
`dataset/DATA_DICTIONARY.md` (472), `config_v2.yaml`, `scripts/probes/_w1_migrate_coreg_sign.py`,
`docs/review_2026-07-31/labeling.md`. **Read in part:** `src/coregister.py:278-300, 395-432`,
`DECISIONS.md:2449-2530, 2529-2640, 2855-2895` (grepped by term for `y_sign`, `rung-1`, `2026-06-10`,
`nbr_s5`, `ctx_illum`), `docs/modeling_results.md:830-900`, `docs/CODE_REVIEW_2026-07-31.md` (R04, R23,
R31, R44, R45, R56), `scripts/probes/_sweep_stage6b.py`, `scripts/probes/_fetch_missing_labels.py`,
`README.md`. **Git archaeology:** `git log`/`git diff` on `config.yaml` and `config_v2.yaml` across all
their commits.

**Measurements I ran** (all read-only; scratch scripts in the session scratchpad, none in the repo):
1. current `config_hash` for both configs vs all 47 label sidecars;
2. per-image sidecar table (written_at, hash, filters, thresholds, `coreg_shift_m`) for both cohorts,
   captured **twice** (before and after the mid-session regeneration of one v1 image);
3. all 48 coregistration caches — `dy_m`, `dy_px`, `y_sign_fix_applied`, mtime;
4. **the `labeling-2` mask measurement** — 38 `*_hirise_mask.tif` rasters, translated support region,
   per-tile block min/max at S=32 over all 161,005 eligible tiles, with three built-in validations;
5. split rebuild — `build_split` re-run in memory from today's labels for all 7 split JSONs in both
   cohorts, comparing `split_hash`, `manifest_obs_ids`, per-fold summaries and (for within-image)
   quadrant definitions and per-scale tile counts;
6. packaged-vs-labels row join over all 7 packaged directories (~8.6 M rows) on
   `(obs_id, tile_size_px, ti, tj)`, reporting differing-`fa` counts, `boulder_count` diffs, max |Δ| and
   `fa > 1e-2` class flips;
7. a label-column-carrier census over every parquet under `dataset/` and `dataset_v2/`;
8. label-vs-input mtime ordering for all 47 images against 4 upstream caches each;
9. derived-column self-consistency over all 4,208,677 label rows in both cohorts;
10. `fbuild_abort_level_vs_labels.csv` vs today's S=32 labels;
11. LOIO partition equality across 3 split JSONs + 3 packaged metadata files;
12. parquet schema/dtype dump vs the DATA_DICTIONARY Stage-4 table.

**Could NOT check:** whether the two stale side-scheme packages' *feature* columns are also a different
vintage (that is `features-deep`'s derived-cache thread; I only compared the `y` side); whether any
`models/**` metrics were produced from the stale packages after 2026-06-10 18:18 PDT (I dated the three
candidate sweep runs and all three predate it, but I did not audit every timestamped run directory);
what today's Stage 4 would emit for the eight un-regenerated v1 images (that requires *writing* labels,
which is out of scope — I used the one image another process regenerated as the empirical proxy and
flagged it as a lower bound); and the `labeling-2` count at S=8/16/64 (measured at S=32 only, the frozen
recipe's scale). Note also that a `.gpkg` read can bump the file's mtime under GDAL, so mtime-based
staleness reasoning on `cache*/reprojected_detections/` is unreliable — I relied on
`written_at_iso`/content wherever it mattered.
