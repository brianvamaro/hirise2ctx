# Review area: tests-deep-features

- **Reviewed at commit:** bd19da8
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified)
- **Method:** `tests/test_features.py` (533 lines) read line by line, then **mutation-tested**.
  `src/` was copied to the scratchpad (`…/scratchpad/featmut/src`), `tests/conftest.py` +
  `tests/test_features.py` were copied **verbatim**, and pytest was run from the scratchpad root.
  The repo's `src/` and `tests/` were never modified; no producer was run against the live tree.
  25 single-point mutants were seeded (22 realistic defects + 3 "does the suite pin a *known*
  defect as intended?" probes). Full per-mutant table in the coverage note.

## Headline result

```
baseline   pytest tests/test_features.py -q -m "not slow"  ->  20 passed,  2 deselected  (1.4 s)
baseline   pytest tests/test_features.py -q                ->  20 passed,  2 skipped     (1.4 s)

22 defect mutants:  12 SURVIVED  -m "not slow"   (54.5 %)
                    12 SURVIVED  the full file   (54.5 %)   <- identical set, zero gap
 3 known-defect fix probes (R27, R28, the features-6 guard): 3/3 SURVIVED
```

**The fast-vs-full gap is exactly zero, and that is itself the finding.** The two
`@pytest.mark.slow` tests are the only difference, and they cannot be run safely: the first is a
**producer call on the live gitignored `dataset/` tree** (finding 1). In the scratchpad they skip;
in the repo they overwrite four artifacts. So CLAUDE.md's documented dev loop
(`pytest -m "not slow"`) and its documented "full suite" are, for this file, the same test set
plus one unrunnable pair.

For calibration against the sibling area: `labeling-deep-tests` measured 16/20 surviving fast and
12/20 surviving full. This file is slightly tighter on the fast loop and has **no** extra
protection at all in the full loop.

## Findings

### tests-deep-features-1 — The one test that touches real data is a producer writing into the live `dataset/` tree, and the assertion it pays that price for is true by construction
- **Severity:** high
- **Liveness:** live-shipped (CLAUDE.md documents "full suite incl. slow integration" as a thing to run)
- **Confidence:** high
- **Where:** `tests/test_features.py:479-507` (`test_features_align_with_labels_row_for_row`),
  `:510-533` (`test_features_sanity_on_real_data`); producer write sites
  `src/features.py:760` (`np.save`), `:793` (`to_parquet`), `:831` (sidecar)

`test_features_align_with_labels_row_for_row` calls `stage4b_one_image(obs,
cache_dir=repo_root/"cache", output_dir=repo_root/"dataset", …)` — the **real** trees. It
therefore overwrites `dataset/features/ESP_069669_2220.parquet` (96,354 rows),
`dataset/features/ESP_069669_2220.json`, and `dataset/context_patches/ESP_069669_2220_S{32,64}.npy`.
`.gitignore:15` is `dataset/*`, so **git cannot restore any of them**. This is a *third* live-tree
test beyond the two `_prompts.md` names (`test_stage4_runs_on_ESP_069669_2220`,
`test_empty_shapefile.py`), and it is in the file whose brief says "prefer `-m "not slow"`".

What it buys: `assert len(features_df) == len(labels_df)` and `assert label_keys == feature_keys`.
`stage4b_one_image` builds its output by iterating `labels_df.groupby("tile_size_px")` and copying
that group's `ti`/`tj` arrays verbatim (`src/features.py:647-674`), so **one feature row per label
row with the identical key tuple is guaranteed by the loop**, not verified by the test. The only
way it can fail is (a) `stage4b_one_image` raising, or (b) a `tile_size_px` group containing two
different `scale_idx` values (`scale_idx` is broadcast from `.iloc[0]` at `:649`) — measured on
this image: `{8: 1, 16: 1, 32: 1, 64: 1}`, so (b) is impossible. Critically, the test compares
*keys*, never *pixels*, so it cannot detect the misregistration hazard of **features-6** (the
mosaic origin copied from the labels sidecar and never cross-checked) — which is the one thing a
real-data alignment test could have been for.

- **Failure scenario:** a developer follows CLAUDE.md ("full suite incl. slow integration") or a
  reviewer runs `pytest tests/test_features.py`. Four gitignored artifacts are silently
  regenerated. Right now that is not a no-op: `dataset/labels/ESP_069669_2220.json` was rewritten
  **today** (`written_at_iso 2026-08-04T21:26:35`) with `config_hash e9962e9418a759e9…`, while
  `config.yaml` hashes to `958fdc25e828feb9…` and the surviving June features sidecar carries
  `958fdc25…`. So the slow test would rebuild the features from the post-incident labels and stamp
  them `config_hash = cfg.hash` (`:494`) — producing an artifact that *claims* config consistency
  with a labels set built under a different hash, i.e. it would launder the 2026-08-04 labelling
  incident into the features tree and destroy the only surviving evidence of the pre-incident state.
- **Evidence:**
  ```
  tests/test_features.py:485-495
      cache_dir = repo_root / "cache"
      output_dir = repo_root / "dataset"
      ...
      stage4b_one_image(
          obs, cache_dir=cache_dir, output_dir=output_dir,
          features_cfg=cfg["features"], config_hash=cfg.hash,
      )

  tests/test_features.py:499-507
      assert len(features_df) == len(labels_df)
      label_keys   = set(zip(labels_df[...]))
      feature_keys = set(zip(features_df[...]))
      assert label_keys == feature_keys        # guaranteed by src/features.py:647-674

  measured read-only: labels rows 96354 == features rows 96354, key sets equal,
      labels config_hash e9962e9418a759e9  vs  features config_hash 958fdc25e828feb9
  git check-ignore -v dataset/features/ESP_069669_2220.parquet -> .gitignore:15:dataset/*
  ```
- **Self-refutation attempted:** (a) *Is it marked `slow`, so nobody runs it?* It is — but
  `test_labeling.py`'s equivalent was also marked slow and was run anyway on 2026-08-04; and
  CLAUDE.md explicitly names the full suite as a mode. (b) *Does the second slow test compensate?*
  No: `test_features_sanity_on_real_data` only range-checks (`intensity_mean ∈ [0,255]`, fractions
  ∈ [0,1], LBP sums, `glcm_contrast ≥ 0`). I checked all 12 fast-survivors against those
  assertions by hand: **11 of 12 pass them** (a halved `edge_density` is still in [0,1], a
  transposed `_stack_tiles` still yields valid DN statistics, a collapsed GLCM distance mapping is
  still ≥ 0). Only M02 (origin sign flip) would be caught, and then only as a `RuntimeError` from
  the bounds check at `:656`, reported as a cache mismatch. (c) *Could the key-set assertion ever
  fail on some other image?* Only via the `scale_idx` pathology above; every packaged v2 image has
  a 1:1 `tile_size_px ↔ scale_idx` map. It survives.
- **Fix:** give the test a `tmp_path` output tree (copy/symlink the two labels files in, or add a
  `dry_run`/`output_dir` override), and replace the key-set assertion with one that can fail —
  e.g. assert the tile's own `xmin/ymin` from the labels parquet equals
  `window_transform * (c_win, r_win)` for one tile per scale, which is exactly the check
  **features-6** asks for and would make the real-data test worth its cost.

---

### tests-deep-features-2 — 12 of 22 seeded defects survive both loops; the whole labels→window registration arithmetic is among them
- **Severity:** medium
- **Liveness:** live (Stage 4b is the documented "re-run features only" seam and produced every `dataset*/features/` cache)
- **Confidence:** high (executed)
- **Where:** `tests/test_features.py` as a whole; the unprotected code is
  `src/features.py:215-229` (`_stack_tiles`), `:653-663` (origin arithmetic + the bounds guard),
  `:268-297` (gradient), `:334-363` (canny), `:381-423` (lacunarity), `:479-482` (GLCM
  distance→column), `:490-538` (context patches)

Twelve realistic single-point defects pass all 20 fast tests **and** the full file. The most
consequential cluster is registration: `_stack_tiles` reading `arr[c:c+S, r:r+S]` instead of
`arr[r:r+S, c:c+S]` (**M01**), `r_win = ti*S + mosaic_row_origin` instead of `−` (**M02**), and
deleting the tiles-inside-window guard entirely (**M25**) are all invisible. Beyond registration:
the GLCM distance→column mapping can be collapsed so every `glcm_*_d{1,2,3}` carries d=1's value
(**M09**); `grad_dir_circvar` can be inverted from `1−R` to `R` (**M20**); `edge_density` can be
halved (**M17**); the gliding-box lacunarity loop can drop its last box row/column (**M15**);
`intensity_iqr` can become `p90−p10` (**M11**); the shadow cut can flip from `<` to `<=` (**M07**);
the low-mode fallback trigger can be off by one (**M06**); the context patch can stop being centred
on the tile (**M19**).

- **Failure scenario:** any of these introduced by a refactor ships silently. `_stack_tiles` is the
  single funnel every per-tile family goes through (intensity, gradient, shadow, LBP, canny, mask),
  so **M01** alone would corrupt every feature column of every non-square-symmetric window with no
  test, no exception and no NaN — the parquet would look complete.
- **Evidence:**
  ```
  M01 [SURVIVED fast | SURVIVED full] _stack_tiles: row/col transposed slice
  M02 [SURVIVED fast | SURVIVED full] stage4b: mosaic_row_origin sign flipped (- -> +)
  M09 [SURVIVED fast | SURVIVED full] _glcm_per_tile: every distance column gets distances[0]'s value
  M20 [SURVIVED fast | SURVIVED full] _gradient_stats: circular variance 1-R -> R
  M25 [SURVIVED fast | SURVIVED full] the tiles-inside-window bounds check never fires
  ```
  Per-survivor, the assertion that *should* have caught it:
  | mutant | test that should have caught it | why it didn't |
  |---|---|---|
  | M01 transpose | `test_stack_tiles_preserves_pixel_values:375-383` | `r_win = c_win = [0, 8]` on a square array, so the transposed slice is bit-identical |
  | M02 origin sign | the three `stage4b_*` end-to-end tests | fixture sidecar sets `mosaic_row_origin = mosaic_col_origin = 0` (finding 3) |
  | M03 mask dropped | every `_compute_dn_thresholds` test | all pass `mask = np.ones_like(arr)` (finding 4) |
  | M06 `<=`→`<` fallback trigger | `test_dn_threshold_percentile_fallback_when_mode_is_dark:279` | mode ≈ 15 ⇒ `shadow = max(0, −5) = 0`, which satisfies both predicates; needs mode = 21 |
  | M07 `<`→`<=` shadow cut | `test_shadow_fraction_on_synthetic_bimodal_image:294` | tile DNs are {30, 200} vs threshold 100 — no pixel sits on the boundary |
  | M09 GLCM d-mapping | `test_glcm_uniform_image_has_zero_contrast:175` | constant image ⇒ d1 = d2 = d3 = 0; the padding test passes `distances=[1]` only |
  | M11 IQR redefined | `test_intensity_stats_constant_tile:134` | asserts `iqr == 0` on a constant tile — true for any percentile difference; the ramp test never checks IQR (it would discriminate: 31.5 vs 50.4) |
  | M15 gliding-box off-by-one | both `test_lacunarity_*` | both assertions are qualitative (`== 1.0` on all-ones, `> 1.0` on clumped); dropping the last box preserves both |
  | M17 `edge_density` halved | nothing — `_canny_per_tile` has **no** unit test | the end-to-end test only checks the column *exists* |
  | M19 patch not centred | `test_stage4b_context_patches_bundle_indices:451` | it compares `(patch_idx ≥ 0).sum()` with `prov["patch_counts"]` and `patches.shape` — all three come from the same `_build_context_patches` return (finding 6) |
  | M20 circvar inverted | nothing — no assertion on `grad_dir_circvar`'s value anywhere | only the column name is asserted |
  | M22 `valid_pixel_fraction` hardcoded | `assert (df["valid_pixel_fraction"] == 1.0).all():416` | the fixture's mask is all ones, so the constant satisfies it (finding 4/6) |
- **Self-refutation attempted:** (a) *Are the mutants unrealistically exotic?* Every one is a
  documented mutation-testing archetype the brief names (transpose, sign flip, off-by-one, wrong
  column, dropped filter, wrong axis) and each is a plausible refactor slip; three of them
  (M01/M02/M25) target code `features.md` already flags as under-verified. (b) *Do the other test
  files cover the same functions?* No — `tests/test_spatial_features.py`, `test_colour.py` and
  `test_ctx_source_illumination.py` import from different modules; `grep -l "src.features"
  tests/*.py` returns only `test_features.py`. (c) *Is the killed set trivial?* No — the 10 kills
  are substantive and are enumerated under **Verified clean** below.
- **Fix:** the four highest-value additions, in order: (i) make `test_stack_tiles` use asymmetric
  offsets (`r_win=[0,8], c_win=[8,0]`) on a rectangular array; (ii) give one end-to-end fixture a
  non-zero `mosaic_row_origin`/`mosaic_col_origin` and assert an emitted `intensity_mean` equals
  the value computed directly from the window slice; (iii) a `_glcm_per_tile` test on a *textured*
  tile asserting `contrast_d1 != contrast_d3`; (iv) a `_canny_per_tile` unit test asserting
  `edge_density == n_edge_px / S²` on a hand-built edge map.

---

### tests-deep-features-3 — Every end-to-end fixture pins the mosaic grid origin to (0, 0), which 0 of 52 production images has
- **Severity:** medium
- **Liveness:** live
- **Confidence:** high (measured over every labels sidecar on disk)
- **Where:** `tests/test_features.py:116-121` (the sidecar the fixture writes);
  consumed at `src/features.py:578-579`, used at `:653-654`

The fixture writes `"mosaic_row_origin": 0, "mosaic_col_origin": 0`. I read every labels sidecar in
the repo — `dataset/labels` (9), `dataset_v2/labels` (38), `dataset_v2_dev/labels` (5): **0 of 52
have either origin equal to 0**; `row_origin` spans 894…43,790 and `col_origin` 183…41,945. With
both origins zero, `ti*S − origin`, `ti*S + origin` and `ti*S` are the same expression, so the sign,
the row/col pairing and the subtraction itself are all untested — and so is the bounds guard at
`:656-663` that is the *only* protection **features-6** credits against a Stage-2/Stage-4 origin
drift (deleting it, M25, changes nothing).

This is precisely the shape `labeling-deep-tests` generalised from (`R77-R80`: every labelling
fixture pinned the mosaic grid phase to zero, which 0 of 47 production images has) and that
`src/fgates.py:211-231` records as having already caused the ~100 km gate mis-key. **It has now
bitten this project three times.**

- **Failure scenario:** a refactor of the origin arithmetic (e.g. adopting features-6's fix, which
  *recomputes* the origin from the window affine) inverts a sign or swaps row/col. The whole
  feature matrix is read from the wrong part of the window, silently misregistered from its labels
  by up to ~1 km (the `ctx_retrieve.buffer_m: 1000` margin keeps interior tiles in bounds), the
  parquet looks complete, and the suite is green. The symptom is an unexplained metric drop.
- **Evidence:**
  ```
  tests/test_features.py:116-121
      sidecar = {
          "obs_id": obs_id,
          "tile_sizes_px": [tile_size_px],
          "mosaic_row_origin": 0,
          "mosaic_col_origin": 0,

  measured (read-only, all *.json under */labels):
      dataset/labels:        n=9  both-origins-zero=0  row 3886..36943   col 1530..41342
      dataset_v2/labels:     n=38 both-origins-zero=0  row  894..43790   col  183..41945
      dataset_v2_dev/labels: n=5  both-origins-zero=0  row 4563..36922   col 7683..26146

  M02 [SURVIVED] r_win = (ti * S + mosaic_row_origin)
  M25 [SURVIVED] bounds guard replaced by `if False:`
  ```
- **Self-refutation attempted:** (a) *Is (0,0) legitimate because the fixture's window is the whole
  mosaic tile?* It is internally consistent — but that is the defect: the fixture chooses the one
  configuration in which the arithmetic is degenerate. (b) *Does the slow real-data test cover it?*
  It would, but only by crashing (M02 → `RuntimeError` "Stage 2/4 cache mismatch"), and it may not
  be run (finding 1). (c) *Does another test file exercise a non-zero origin?* `grep -rn
  "mosaic_row_origin" tests/` hits only `test_features.py` and `test_labeling.py`, and
  `labeling-deep-tests` already established the labelling fixtures are zero-phase too.
- **Fix:** one line — set `mosaic_row_origin`/`mosaic_col_origin` to a non-zero, non-equal pair
  (e.g. 37 / 19) in `_write_synthetic_stage4_cache` and shift the labels' `ti`/`tj` accordingly, so
  the tiles still land inside the window. That single change kills M02 and M25 and makes the
  row/col pairing observable.

---

### tests-deep-features-4 — The HiRISE coverage mask is all-ones in every fixture, so the mask-support dependence — the train/deploy seam of `features-5` — is entirely unpinned
- **Severity:** medium
- **Liveness:** live for the documented off-HiRISE deployment contract (`src/modeling/inference.py:32-35`)
- **Confidence:** high (executed)
- **Where:** `tests/test_features.py:83` (`mask = np.full(..., mask_fill=1)`), `:248`, `:269`, `:284`
  (`mask = np.ones_like(arr)`); code `src/features.py:141`, `:680-683`

Every fixture that supplies a HiRISE mask supplies an all-ones one. Consequently
`covered = arr[mask == 1]` (`:141`) can be replaced by `arr.ravel()` — deleting the entire
mask restriction — and the suite is green (**M03**); and `valid_pixel_fraction` can be hardcoded to
`np.ones(...)` instead of measured from the mask and the suite is green (**M22**), because the only
assertion is `assert (df["valid_pixel_fraction"] == 1.0).all()` (`:416`), whose own comment says
"must be 1.0 by construction" — an assertion on the fixture, not on the code.

The mask support is not incidental: `features-5` identifies it as the reason the documented
off-HiRISE extractor contract is false ("the mode is computed over `arr[mask == 1]`, the HiRISE
coverage mask, which by definition does not exist off-HiRISE"), and every shadow-family column plus
both `lacunarity_shadow_b*` columns hang off that one number. The one behaviour a test could pin —
that the DN mode is a function of the *covered* pixels only — is exactly what is not pinned.

- **Failure scenario:** someone implements the off-HiRISE extractor (or simplifies `:141` during a
  refactor, since it looks like a no-op under test) and the DN mode silently moves from the
  HiRISE-covered support to the whole window. Every shadow/bright/lacunarity column shifts, the
  train/deploy mismatch is invisible, and this is the failure class that killed F pilot leg A
  (DECISIONS 2026-07-04).
- **Evidence:**
  ```
  tests/test_features.py:83   mask = np.full((height, width), mask_fill, dtype=np.uint8)   # mask_fill=1
  tests/test_features.py:248  mask = np.ones_like(arr)
  tests/test_features.py:416  assert (df["valid_pixel_fraction"] == 1.0).all()

  M03 [SURVIVED fast | SURVIVED full] covered = arr[mask == 1]  ->  covered = arr.ravel()
  M22 [SURVIVED fast | SURVIVED full] valid_pixel_fraction = np.ones(r_win.size)
  ```
- **Self-refutation attempted:** (a) *Is a partial mask untestable here?* No — `mask_fill` is
  already a fixture parameter (`:54`), it is simply never given a value other than 1, and no test
  builds a spatially varying mask. (b) *Does the slow test cover it?* Only by range-checking
  `valid_pixel_fraction ∈ [0,1]` (`:523-526`), which a hardcoded 1.0 satisfies. (c) *Is the mask
  genuinely all-ones in production?* For *eligible* Stage-4 tiles yes (that is the eligibility
  rule), which is why this is medium and not high — but `_compute_dn_thresholds` reads the mask
  over the **whole window**, where it is emphatically not all ones, and that is the path M03 breaks.
- **Fix:** give one fixture a half-covered mask (`mask[:, :W//2] = 0`) plus a distinctly different
  DN population under the uncovered half, and assert (i) the modal DN follows the *covered* half,
  and (ii) `valid_pixel_fraction` is 0.5 for tiles straddling the boundary.

---

### tests-deep-features-5 — Stage 4b is only ever run end-to-end at S=16, a scale the frozen recipe never uses; lacunarity is therefore never exercised end-to-end at all
- **Severity:** low
- **Liveness:** live
- **Confidence:** high (measured by running the fixture in the scratchpad)
- **Where:** `tests/test_features.py:47-56` (`tile_size_px: int = 16`), used by all three
  `test_stage4b_*` tests; gating at `src/features.py:705`, `:710`, `:719`

I ran the fixture through `stage4b_one_image` in the scratchpad and enumerated the output: the
emitted scales are `[16]` and the parquet has **58 columns with zero `lacunarity_*` columns**,
because `cfg["lacunarity"]["min_tile_size_px"] = 32` gates the family off at S=16
(`src/features.py:719`). The frozen recipe is `…_S32` and the v2 caches carry S=8/16/32/64, so the
end-to-end path is exercised at exactly one scale, and not one the pipeline reports on. Nothing
tests the multi-scale concat, the per-scale GLCM `levels_per_scale[S]` lookup (a direct `[]` index
that `features.md` notes would `KeyError` on an unlisted scale), the NaN-padding schema *across*
scales, or the `patch_idx` column's per-scale ordering with more than one scale block.

- **Failure scenario:** a config adds or renames a scale, or the per-scale gating thresholds change;
  the end-to-end suite cannot see it. Concretely, the `glcm_cfg["levels_per_scale"][S]` /
  `distances_per_scale[S]` lookups at `:730-731` would raise `KeyError` for a scale present in
  `labeling.tile_sizes_px` but absent from the GLCM config — a crash in the production stage that
  no test can reach, because the fixture emits a single scale that is always present.
- **Evidence:**
  ```
  probe (scratchpad, tmp output dir):
      fixture emitted scales: [16]
      n columns: 58
      lacunarity cols: []
      constant columns: ['config_hash', 'obs_id', 'scale_idx', 'tile_size_px', 'valid_pixel_fraction']
  src/features.py:719
      if "lacunarity" in enabled and S >= int(cfg["lacunarity"]["min_tile_size_px"]) \
  ```
- **Self-refutation attempted:** (a) *Do the two lacunarity unit tests compensate?* They cover the
  arithmetic (and killed M14) but not the gating, the NaN pre-fill at `:392`, or the `b > S`
  continue at `:403`. (b) *Would a second scale be expensive?* No — the fixture is a 128×128 array;
  adding an S=32 scale group costs milliseconds. (c) *Is S=16 deliberate?* Nothing in the file or in
  `DECISIONS.md` says so; it reads as the smallest convenient value.
- **Fix:** parameterise `_write_synthetic_stage4_cache` over two scales (16 and 32) in at least the
  `test_stage4b_synthetic_emits_one_row_per_label` case, and assert that `lacunarity_shadow_b2` is
  NaN at S=16 and finite at S=32 — which pins the gating *and* the NaN-padding convention.

---

### tests-deep-features-6 — Three assertions cannot fail: one compares two outputs of the same call, one asserts a constant, one is symmetric under the defect it targets
- **Severity:** low
- **Liveness:** live
- **Confidence:** high (executed)
- **Where:** `tests/test_features.py:465-471`, `:416`, `:375-383`

Three of the file's assertions are structurally incapable of failing:

1. `test_stage4b_context_patches_bundle_indices:465-471` asserts
   `(df["patch_idx_S32"] >= 0).sum() == prov["context_patch"]["patch_counts"][32]` and
   `patches.shape == (n_valid, 32, 32)`. All three quantities are produced by the same
   `_build_context_patches` call, so they agree under *any* centring rule — removing the
   `− P//2` centring offset entirely (**M19**) leaves the test green. Nothing compares patch
   *content* to `arr[r_win:r_win+S, c_win:c_win+S]`. (The FM recipe's own extraction probe
   `scripts/probes/_w2_fang_embed.py:228-233` does pin this — outside the test suite.)
2. `test_stage4b_synthetic_emits_one_row_per_label:416` — `valid_pixel_fraction == 1.0` on an
   all-ones-mask fixture; satisfied by hardcoding the column (**M22**).
3. `test_stack_tiles_preserves_pixel_values:375-383` uses `r_win = c_win = [0, 8]`, so the
   row/col transpose it exists to prevent is a no-op (**M01**).

- **Failure scenario:** the file reads as covering context patches, mask validity and tile slicing;
  a future session (or this review's `tests` area, which surveyed assertion *shapes*) counts them as
  covered and does not add the real check. Meanwhile the context-patch geometry is the input to the
  frozen FM recipe.
- **Evidence:**
  ```
  tests/test_features.py:465-471
      n_valid = int((df["patch_idx_S32"] >= 0).sum())
      assert n_valid == prov["context_patch"]["patch_counts"][32]
      patches = np.load(patches_path)
      assert patches.shape == (n_valid, 32, 32)

  M19 [SURVIVED fast | SURVIVED full] r0 = rc - half  ->  r0 = rc   (patch no longer centred)
  ```
- **Self-refutation attempted:** (a) *Is (1) worth anything?* Yes, a little: it checks the index
  array and the stack length stay in sync, which is a real desync class — I do not claim it is
  worthless, only that it cannot see geometry. (b) *Is patch content checked anywhere in `tests/`?*
  `grep -rn "context_patch" tests/` returns only this file. (c) *Is `P == S` the only case that
  matters?* No — `DEFAULT_FEATURES_CFG` ships `sizes_px: [32, 64]` against S ∈ {8,16,32,64}, so the
  centring matters for every P ≠ S pair, none of which is content-checked.
- **Fix:** in the same test, assert `np.array_equal(patches[idx], arr[r0:r0+32, c0:c0+32])` for one
  known tile with hand-computed `r0`/`c0` — three lines, and it kills M19.

## Refuted by my own check

- **"The suite pins a known feature defect as intended" (the brief's cross-check question).** It
  does **not**, and I tested this directly by applying the register's own proposed fixes as mutants:
  **M23** (R27's fix: lacunarity returns `np.nan` instead of the `0.0` sentinel when a tile has no
  shadow pixels) and **M24** (R28's fix: canny switched to `use_quantiles=True`) both leave the file
  **green**. So neither defect is defended by an assertion, and both fixes can be applied without
  touching a test. `features.md` inferred this from reading; it is now executed.
- **"The idempotence test is vacuous."** It is not. `test_stage4b_is_idempotent` compares two full
  runs with `pd.testing.assert_frame_equal`, which would catch dict-ordering column permutation,
  RNG leakage or a mutable-state bug across runs. It is a weak-but-real check; no mutant I seeded
  exercised it, but it is not a tautology (the two frames come from two independent invocations).
- **"`test_glcm_padding_with_nan_for_missing_distances` is testing skimage, not us."** No — the
  NaN-padded per-scale schema is the project's own convention (`src/features.py:454-457`) and the
  test pins it. It also killed nothing I seeded only because I did not seed a schema mutant there;
  M09's collapse writes into the same columns, which the padding test cannot see (`distances=[1]`).
- **"The fast/full gap is hidden because the slow tests were skipped rather than run."** True but
  unavoidable and correctly handled: I reasoned about the slow pair's assertions by hand
  (finding 1, self-refutation (b)) and only one of the twelve survivors (M02) would change verdict,
  as a crash. Running them is forbidden and would have destroyed live artifacts.
- **"`test_dn_threshold_survives_clip_spike` is another test that appears to exercise a defect but
  does not" (the R24 shape).** It genuinely does: dropping the `covered > _DN_CLIP_FLOOR` filter
  (**M04**) fails it. This is a real regression guard for the 2026-06-10 finding worth +0.249 /
  +0.127 meaningful AUC on two images.
- **`test_features_sanity_on_real_data` reaching a real tree.** It only *reads*
  `dataset/features/{obs}.parquet` and skips if absent — it is not itself a producer. The hazard is
  entirely in the test above it, which it depends on (its skip message says "run
  test_features_align... first").

## Verified clean — what this suite genuinely DOES pin (each named by the mutant that killed it)

1. **The 2026-06-10 DN-clip fix.** Excluding `DN <= 1` from the modal histogram is a live
   regression guard — **M04** (`covered[covered > _DN_CLIP_FLOOR]` → no filter) is killed by
   `test_dn_threshold_survives_clip_spike`. This is the highest-value assertion in the file.
2. **The identity of the three DN offsets.** Deriving the shadow cut from `strict_offset_dn`
   instead of `shadow_offset_dn` (**M05**) is killed by `test_dn_mode_threshold_finds_modal_peak`
   *and* the clip-spike test.
3. **Shadow-family column identity.** Computing `bright_cap_fraction` from the strict-shadow mask
   (**M08**) is killed by `test_shadow_fraction_on_synthetic_bimodal_image`, which pins all three
   fractions to exact values (16/64, 0.0, 16/64) on a hand-counted tile.
4. **GLCM quantisation bucket boundaries.** `bin_width = 256 // levels` with the exact 0/31/32/255
   boundaries (**M10**, 256→255) is killed by `test_glcm_quantize_levels`.
5. **Intensity percentile identity.** Swapping p10 and p90 (**M12**) is killed by
   `test_intensity_stats_ramp_tile_p10_p90`, which pins numpy's default `linear` interpolation
   (6.3 / 56.7) — so a numpy percentile-method change would also be caught.
6. **The sub-tile block decomposition.** `blocks.mean(axis=(2, 4))` really is the four (S/2)²
   *spatial* block means: taking axes (1,3) instead (**M13**) is killed by
   `test_subtile_variance_positive_on_split_tile`. That test is doing real work.
7. **Lacunarity is `E[M²]/E[M]²` and actually discriminates.** Replacing `E[M²]` with `E[M]²`
   (**M14**, which makes L ≡ 1) is killed by `test_lacunarity_on_clumped_shadow_mask_above_one`.
   The uniform-mask test alone would not have caught it — the clumped one is load-bearing.
8. **`grad_mag_p99` is the 99th percentile, not the 90th.** Swapping them (**M16**) is killed by
   `test_gradient_on_step_function` (p90 is 0 on a 6 %-nonzero gradient field, so `p99 > mean`
   fails). The one gradient assertion in the file is stronger than it looks.
9. **The LBP schema is P+2 = 10 bins.** `n_lbp_bins = P + 1` (**M18**) is killed by the column-list
   assertion in `test_stage4b_synthetic_emits_one_row_per_label` (`lbp_hist_9`). That column list
   is the file's main schema guard — 24 named columns across 8 families.
10. **The LBP histogram normaliser is exactly the tile pixel count.** An off-by-one denominator
    (**M21**) is killed by `test_lbp_hist_sums_to_one` at `pytest.approx`'s default 1e-6.

Also pinned, though no mutant of mine tested them: run-to-run determinism
(`test_stage4b_is_idempotent`), the `context_patch.enabled = false` switch (no patch files, no
`patch_idx_*` columns, provenance flag), the per-scale GLCM NaN-padding convention, and
`prov["n_tiles_total"]` equalling the row count.

## Coverage note

**Read in full:** `tests/test_features.py` (533), `src/features.py` (872),
`docs/review_2026-07-31/features.md`, `features-deep.md`, `_prompts.md` §1/§3,
`_prompts_tests_deep.md`, `tests/conftest.py`, `pyproject.toml`'s pytest config.

**Executed:** 25 mutants × 2 pytest invocations each (fast + full), all against a scratchpad copy of
`src/`. Driver + raw results at
`…/scratchpad/featmut/{driver.py,results_features.json}`. Full table:

| id | mutation | fast | full | killer |
|---|---|---|---|---|
| M01 | `_stack_tiles` row/col transposed | **SURVIVED** | **SURVIVED** | — |
| M02 | `mosaic_row_origin` sign flipped | **SURVIVED** | **SURVIVED** | — |
| M03 | HiRISE-mask restriction dropped | **SURVIVED** | **SURVIVED** | — |
| M04 | DN≤1 clip filter dropped | killed | killed | `test_dn_threshold_survives_clip_spike` |
| M05 | shadow cut uses `strict_offset_dn` | killed | killed | `test_dn_mode_threshold_finds_modal_peak` (+1) |
| M06 | low-mode fallback trigger `<=`→`<` | **SURVIVED** | **SURVIVED** | — |
| M07 | shadow test `<`→`<=` | **SURVIVED** | **SURVIVED** | — |
| M08 | `bright_cap` from strict mask | killed | killed | `test_shadow_fraction_on_synthetic_bimodal_image` |
| M09 | GLCM: all distances get d1's value | **SURVIVED** | **SURVIVED** | — |
| M10 | quantise `bin_width` 256→255 | killed | killed | `test_glcm_quantize_levels` |
| M11 | IQR = p90−p10 | **SURVIVED** | **SURVIVED** | — |
| M12 | intensity p10/p90 swapped | killed | killed | `test_intensity_stats_ramp_tile_p10_p90` |
| M13 | subtile block axes (2,4)→(1,3) | killed | killed | `test_subtile_variance_positive_on_split_tile` |
| M14 | lacunarity E[M²]→E[M]² | killed | killed | `test_lacunarity_on_clumped_shadow_mask_above_one` |
| M15 | gliding-box range off-by-one | **SURVIVED** | **SURVIVED** | — |
| M16 | grad p90/p99 swapped | killed | killed | `test_gradient_on_step_function` |
| M17 | `edge_density` halved | **SURVIVED** | **SURVIVED** | — |
| M18 | `n_lbp_bins` P+2→P+1 | killed | killed | `test_stage4b_synthetic_emits_one_row_per_label` |
| M19 | context patch not centred | **SURVIVED** | **SURVIVED** | — |
| M20 | `grad_dir_circvar` 1−R→R | **SURVIVED** | **SURVIVED** | — |
| M21 | LBP denominator off-by-one | killed | killed | `test_lbp_hist_sums_to_one` |
| M22 | `valid_pixel_fraction` hardcoded 1.0 | **SURVIVED** | **SURVIVED** | — |
| M23 | *R27 fix* (lacunarity NaN not 0.0) | SURVIVED | SURVIVED | — (fix probe, not a defect) |
| M24 | *R28 fix* (canny quantile thresholds) | SURVIVED | SURVIVED | — (fix probe) |
| M25 | features-6 bounds guard removed | SURVIVED | SURVIVED | — (guard probe) |

**Read-only measurements against the live repo:** every `*/labels/*.json` sidecar's
`mosaic_row_origin`/`mosaic_col_origin` (52 files); the `written_at_iso`/`config_hash` of
`dataset/features/ESP_069669_2220.json` and `dataset/labels/ESP_069669_2220.json`;
`load_config("config.yaml").hash`; row counts and key sets of that image's labels vs features
parquets; `git check-ignore` on the three artifacts the slow test would overwrite. No producer was
called, no imagery decoded, no network access.

**Could NOT check:** (1) the two `@pytest.mark.slow` tests — running them writes into the live
gitignored `dataset/` tree (finding 1), so their behaviour under each mutant is *reasoned from
their assertions, not executed*, and is labelled as such; (2) whether any *reported* number moves
under any surviving mutant (needs a re-sweep, out of scope); (3) `src/spatial_features.py`,
`src/colour.py` and `src/ctx_source_illumination.py` — **the brief lists these as covered by
`tests/test_features.py`, but they are not**: this file imports only from `src.features` (plus
`LABELS_SUBDIR` from `src.labeling`), and `grep -l "src\.features" tests/*.py` returns only
`tests/test_features.py`. Those three modules are covered by `tests/test_spatial_features.py` (268),
`test_colour.py` (148) and `test_ctx_source_illumination.py` (273), which are **outside this
sub-area's target file and were not mutation-tested** — if the register wants them mutation-tested,
that is a separate sub-area.
