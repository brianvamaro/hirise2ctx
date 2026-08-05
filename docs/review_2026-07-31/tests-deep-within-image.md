# Review area: tests-deep-within-image

- **Reviewed at commit:** 577277f
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified)

Target: `tests/test_within_image_split.py` (445 lines, 16 tests, 1 `slow`), covering the quadrant
splitter in `src/dataset.py` (`_compute_quadrant_definitions` :121-177, `_quadrant_array_for_image`
:180-218, `_within_image_fold_summary` :221-263, `_assign_within_image_kfold` :266-318,
`_package_within_image_split` :735-841).

## Headline

**The suite's assertions are strong; its *fixture* is not.** Every synthetic image the file builds is
the same perfectly square, complete, uniformly-populated 64×64 grid (`_write_multiscale_image` :42-102,
always called with defaults), and every image in a multi-image fixture is **byte-identical in
geometry**. That single property makes four independent mutations of the cut computation into literal
no-ops — including "give every image the same quadrant cut", which on the real label parquets collapses
**8 of 9 images into a single quadrant** and *would* trip the file's own strongest assertion
(`len(unique_train) == 3`, `:406`) if the fixture could express it. This is the third instance in this
project of the shape `src/fgates.py:211-231` records: **a fixture configuration no production input
has**.

Separately, and directly on the R45 thread: **nothing anywhere pins the quadrant cut's *value*, or
checks a stored split against the labels it was derived from.** The live
`dataset_v2/splits/within_image_4fold.json` is measurably stale — 29 of 38 images carry a cut that
today's labels do not reproduce, moving **3.53 % of tiles** into a different quadrant — and the whole
test tree is blind to it.

## Baselines (scratchpad `mutroot/`, pristine `src/`)

- `-m "not slow"` → **15 passed, 1 deselected** in 3.5 s
- full file → **16 passed** in 3.5 s (the slow test *executes*, does not skip: `dataset/labels/` +
  `dataset/splits/` copied read-only into the scratchpad)

## Mutation results

| survival | `-m "not slow"` | full file |
|---|---|---|
| 15 seeded (first pass) | **10 survived (67 %)** | **10 survived (67 %)** |
| 16 seeded (incl. M20, added for the R45 probe) | 11 survived (69 %) | 11 survived (69 %) |
| 14 after discarding 2 I proved equivalent/benign | **9 survived (64 %)** | **9 survived (64 %)** |

**The fast/full gap is exactly zero.** Structural, not luck: the only `slow` test (`:430-445`) never
calls the splitter — it reads the *stored* `dataset/splits/within_image_4fold.json` and asserts on that
dict, plus `discover_obs_ids`. CLAUDE.md's documented dev loop (`-m "not slow"`) loses nothing here.

| id | mutation | verdict |
|---|---|---|
| M01 | `_compute_quadrant_definitions`: median → **mean** | **SURVIVED** |
| M03 | floor-snap → **ceil-snap** (cut moves 24 → 32) | **SURVIVED** |
| M04 | `ti_mid` / `tj_mid` **transposed** in the returned dict | **SURVIVED** |
| M05 | median over **all scales pooled**, not the finest | **SURVIVED** |
| M07 | quadrant code weights swapped (`2*ti+tj` → `ti+2*tj`) | *benign* (see Refuted) |
| M08 | buffer band `<` → `<=` (drops 3 rows/cols, not 1) | **SURVIVED** |
| M11 | `n_train_tiles_per_scale` ignores the buffer keep-mask | *equivalent* (see Refuted) |
| M12 | `finest_px` `min` → `max` in the fold summary | **SURVIVED** |
| M13 | quadrant defs computed **once** and reused for every image | **SURVIVED** |
| M19 | packaging hardcodes `buffer_tiles = 0` | **SURVIVED** |
| M20 | packaging **re-derives** the cut from today's labels instead of using the fold's declared `quadrant_definitions` (added this pass) | **SURVIVED** |
| M02 | floor-snap to the coarsest factor removed | killed (`test_quadrant_cuts_are_strictly_coherent_across_scales`) |
| M06 | predicate `>=` → `>` | killed (same test) |
| M09 | buffer `OR` → `AND` (only the corner tile dropped) | killed (`test_within_image_buffer_drops_boundary_tiles`) |
| M16 | packaging train rows include the **test quadrant** (self-leak) | killed (`test_within_image_groups_have_3_unique_train_codes_per_fold`) |
| M17 | `groups_*.npy` store the obs code, not the quadrant index | killed (same test) |

---

## Findings

### tests-deep-within-image-1 — Every fixture image is the same symmetric square, so four cut defects are literal no-ops — including one that destroys the partition on real data
- **Severity:** high
- **Liveness:** live-shipped (the within-image arm is the instrument behind `docs/modeling_results.md`
  §9.4 / §7.1 and the H5 conclusion; `PROMOTION_QUEUE.md:222-229` still cites it as UNRESOLVED)
- **Confidence:** high
- **Where:** `tests/test_within_image_split.py:42-102` (`_write_multiscale_image`), `:177-201`
  (`_build_within_image_meta`, which calls it with defaults for **every** image); code under test
  `src/dataset.py:158-176`, `:294-295`

`_write_multiscale_image` always builds `ti ∈ [0,64) × tj ∈ [0,64)`, complete and uniformly populated,
and `_build_within_image_meta` calls it with **no per-image variation** — so `OBS_000`, `OBS_001`, …
are geometrically indistinguishable. Three consequences, each of which kills a mutant's visibility:
(a) `mean == median == 31.5` exactly, so median→mean is a no-op; (b) the `ti` and `tj` distributions
are identical, so transposing `ti_mid`/`tj_mid` is a no-op; (c) all images share one cut, so computing
the cut once and reusing it is a no-op. Real images share none of these properties — the function's own
docstring example (`src/dataset.py:140`) is `ti_mid 1352, tj_mid 5184`, a 3.8× asymmetry.

- **Failure scenario:** any refactor of `_compute_quadrant_definitions` that hoists the call out of the
  per-image loop (`src/dataset.py:294-295` — an obvious "why recompute this 38 times?" optimisation, and
  the function *is* pure and cacheable-looking) silently gives every image image-0's cut. Most images
  then have **one** non-empty quadrant, three of the four folds per image are empty, and the within-image
  AUC is computed on degenerate folds. The suite stays green.
- **Evidence** — pristine vs mutated, run side by side on the fixture and on the 9 real v1 label
  parquets (read-only copies), measuring the **fraction of tiles that change quadrant**:

  ```
  === TEST FIXTURE (64x64 synthetic — what EVERY test in the file uses)
      raw median ti/tj = 31.5/31.5     pristine S8 defs = {ti_mid: 24, tj_mid: 24}
      M01 median->mean        defs_identical=True    tiles_moved =     0 ( 0.00 %)
      M04 ti/tj transpose     defs_identical=True    tiles_moved =     0 ( 0.00 %)
      M05 pool all scales     defs_identical=True    tiles_moved =     0 ( 0.00 %)

  === ESP_039820_1750 (real)   raw median ti/tj = 1352.0/5188.0
      M01 median->mean        defs_identical=False   tiles_moved =  1307 ( 2.01 %)
      M04 ti/tj transpose     defs_identical=False   tiles_moved = 50595 (77.74 %)
      M05 pool all scales     defs_identical=False   tiles_moved = 17678 (27.16 %)
  (across all 9 real images: M01 0.00-4.08 %, M04 74.98-78.30 %, M05 26.68-32.37 %)
  ```

  And M13, the reuse mutant, on real footprints — reusing `ESP_039820_1750`'s cut
  (`ti_mid=1352, tj_mid=5184`):

  ```
  image                own-defs quadrant sizes        M13 shared-defs quadrant sizes   min unique train codes
  ESP_039820_1750      [18077, 14486, 12013, 20505]   [18077, 14486, 12013, 20505]     3
  ESP_047976_2020      [19762, 14801, 14989, 21897]   [    0,     0,     0, 71449]     0
  ESP_054857_2270      [11714, 10851, 12692, 13618]   [    0,     0, 48875,     0]     0
  ESP_056165_2200      [24842, 21503, 18321, 30737]   [95403,     0,     0,     0]     0
  ESP_069669_2220      [25485, 21374, 20909, 28586]   [    0,     0, 96354,     0]     0
  ... 8 of 9 images collapse to a single non-empty quadrant
  ```
- **Which assertion should have caught it, and why it did not:**
  `test_within_image_groups_have_3_unique_train_codes_per_fold` (`:385-408`) asserts exactly the right
  thing — `len(unique_train) == 3`, `len(unique_test) == 1`, no collision — and it is *not* a weak test:
  it killed M16 (self-leak) and M17 (wrong group codes). Under M13 on real footprints it would see **0**
  distinct train codes and fail loudly. It cannot fire only because the fixture's images are identical.
  For M01/M04/M05 the responsible assertions are
  `test_quadrant_cuts_are_strictly_coherent_across_scales` (`:143-144`) and
  `test_within_image_metadata_records_quadrant_definitions` (`:272-279`); both are *relational*
  (`ti_mid_S8 == 2*ti_mid_S16 == …`, `0 <= ti_mid <= 64/factor`) and hold for any cut, and the square
  fixture additionally makes the `ti`/`tj` halves of both assertions redundant with each other.
- **Self-refutation attempted:** (a) is any *other* test file heterogeneous? No — `grep -rl within_image
  tests/` returns only this file, and `tests/test_splits.py`'s fixtures are single-scale, so nothing
  else reaches this code. (b) Is M13 unreachable in practice — is the cut actually image-invariant?
  No: the 9 real v1 images yield **9 distinct** `quadrant_definitions` (`ti_mid` spans 736 → 4768,
  `tj_mid` 264 → 5312). (c) Is M04 an equivalent mutant? No: 75–78 % of real tiles move. (d) Does
  `PLAN_Stage5c.md` or `DECISIONS.md` record the identical-fixture choice as deliberate? No mention.
- **Fix:** give `_build_within_image_meta` per-image offsets/extents (e.g.
  `_write_multiscale_image(..., ti_lo_s8=16*i, ti_hi_s8=64+16*i, tj_hi_s8=48)`), which makes the fixture
  rectangular *and* image-varying and reinstates all four mutants; and add one absolute assertion,
  `assert defs["8"] == {"ti_mid": <computed>, "tj_mid": <computed>}` for a hand-checked footprint.

### tests-deep-within-image-2 — Nothing pins the cut's *value* or its stability against the labels, and the live v2 split is already stale by 3.53 % of tiles
- **Severity:** medium
- **Liveness:** live-shipped (`dataset_v2/splits/within_image_4fold.json` → the
  `models/_sweep_within_image/20260529T142227Z` numbers published as §9.4)
- **Confidence:** high
- **Where:** `tests/test_within_image_split.py:163-170`, `:267-279`, `:430-445`; code at
  `src/dataset.py:158-164`, `:773`, `:390-401`

Two assertions touch the cut and neither constrains it: `:169-170` asserts `ti_mid % 8 == 0` (true of
0, 8, 16, 24, 32 …) and `:278-279` asserts `0 <= ti_mid <= 64/factor` (true of every index in the
grid). So the cut can move anywhere on the snap lattice unnoticed — M03 (floor-snap → ceil-snap) moves
it from 24 to 32, relocating **23.4 %** of the fixture's tiles, and passes both. Worse, no test ever
checks a *stored* split against the labels it was derived from, and the only test that opens a real
split JSON (`:430-445`) is hardcoded to the **v1** 8-image cohort (`n_folds == 32`) and reads only
`kind`, `n_folds`, `excluded_obs_ids`, `manifest_obs_ids` — it never touches `quadrant_definitions`,
and it never opens the v2 file at all.

- **Failure scenario:** the labels are regenerated (a coregistration fix, a filter change) and the split
  JSON is not. Every downstream artifact keyed to `fold_idx` now scores a quadrant that is not the
  quadrant the JSON declares, and the only symptom is a few per-cent of tiles quietly changing sides in
  a comparison whose headline claim is a **null**.
- **Evidence** — recomputing `_compute_quadrant_definitions` at this commit from the labels on disk and
  diffing against the stored `quadrant_definitions`:
  ```
  v1  dataset/      : 0 / 8  images drifted;      0 /   610,586 tiles (0.00 %)
  v2  dataset_v2/   : 29 / 38 images drifted; 125,830 / 3,564,767 tiles (3.53 %)
      ESP_017355_2260: STORED S8 ti/tj=3232/688   RECOMPUTED=3232/696   3.72 % moved
      ESP_042964_2160: STORED S8 ti/tj= 256/1696  RECOMPUTED= 264/1704  7.87 % moved
      ESP_049242_2115: STORED S8 ti/tj=1072/1136  RECOMPUTED=1080/1144  8.14 % moved
      ... every drift is exactly +1 snap step (+8 finest tiles = 320 m), never negative
  what tests/test_within_image_split.py:430-445 checks on the v2 file:
      kind == 'within-image'   -> True
      n_folds == 32            -> False   (v2 has 152; the test only ever reads dataset/, not dataset_v2/)
      quadrant_definitions     -> NOT ASSERTED ANYWHERE
  ```
  M20, the same defect seeded as code — `_package_within_image_split:773` re-deriving the cut from
  today's labels instead of using `fold["quadrant_definitions"]` — also **survives** (15 passed / 16
  passed), so the suite could not distinguish a packaging step that silently used a different vintage
  from the declared split.
- **Self-refutation attempted, and three candidate causes killed:** (a) *did the cut code change since
  the split was built?* No — `git log -L 149,165:src/dataset.py` shows one commit, `5ba0a07`, the
  introduction. (b) *was the split built with a different recipe?* I scored 9 candidate recipes (mean,
  round-instead-of-truncate, ceil-snap, extent midpoint, pooled-scale median, S=64 median×8, positive-
  `fa`-only median, no-snap) against the stored values: the **current** recipe is the best match at
  9/38 and no alternative beats it. (c) *did the tile inventory change?* No — summing each split's own
  `n_test_tiles_per_scale` over its 4 quadrants gives exactly today's S=8 row count for **all** 38
  images. So the JSON encodes label *positions* that are not the ones on disk; the cause belongs to the
  labeling/artifact areas, not here. (d) *does `split_hash` protect against it?* No: `_split_metadata_hash`
  (`:390-401`) does hash `"folds"`, so it covers the cuts — but it is computed **from the folds it just
  built**, so it is self-consistent by construction and cannot detect that those folds disagree with
  today's labels. (e) *is this R45 or R04?* Neither. This **extends R45's incidental** (which measured
  3.5 % between the JSON and the *sweep run's* fold membership) by an independent route — JSON vs
  recomputation — arriving at the same 3.53 %, which localises the discrepancy to the **split artifact**
  rather than the sweep, and adds that no test could ever catch it. R04 is a *packaged*-vs-*split*
  staleness; this is *split*-vs-*labels*. Not re-filing either.
- **Fix:** in the slow test, additionally assert
  `fold["quadrant_definitions"] == _compute_quadrant_definitions(fold["test_obs_id"], labels_dir)` for
  every fold — 3 lines, and it fails today on v2 — and parameterise the test over
  `{dataset, dataset_v2}` instead of hardcoding `n_folds == 32`. In `_package_within_image_split`, raise
  if the declared and re-derived defs disagree rather than trusting the JSON silently.

### tests-deep-within-image-3 — `buffer_tiles` is never exercised through packaging, and the one buffer assertion is one-sided
- **Severity:** medium
- **Liveness:** unclear — dormant under the shipped config (`config.yaml:190`, `config_v2.yaml:170`
  both `buffer_tiles: 0`), but R45's fix assessment and invariant 7 both propose turning it on
- **Confidence:** high
- **Where:** `tests/test_within_image_split.py:282-297`, `:341-423`; code at `src/dataset.py:212-217`,
  `:766`

`test_within_image_buffer_drops_boundary_tiles` is the only test that sets `buffer_tiles=1`, and its
assertions are `(sub["ti"] != qd["ti_mid"]).all()` / same for `tj` — **one-sided**. A mutant that drops
*more* than intended satisfies them: M08 widens the band from `< buffer_tiles` to `<= buffer_tiles`,
dropping 3 rows and 3 columns per scale instead of 1, and passes. (The complementary direction *is*
pinned — M09, narrowing the band from `OR` to `AND`, is killed by this test.) Separately, **every
packaging test uses the default `buffer_tiles=0` fixture**, so `_package_within_image_split:766` is
never run with a non-zero buffer: M19, hardcoding `buffer_tiles = 0` there, survives.

- **Failure scenario:** someone acts on R45's "re-run `within_image_4fold` with `buffer_tiles ≥ 1`"
  recommendation. The split JSON records `buffer_tiles: 1` and the fold summaries exclude the cut-line
  tiles, but under a defect at `:766` the packaged `X_train_fold{k}.parquet` still contains them — the
  declared split and the packaged data disagree, the adjacency control the buffer exists to provide is
  silently absent, and every test passes. Nothing downstream re-checks: `package_meta["buffer_tiles"]`
  (`:835`) is written from the same local variable that was used to build the rows.
- **Evidence:**
  ```
  src/dataset.py:213   in_buf = (np.abs(ti_sub - ti_mid) < buffer_tiles) | (np.abs(tj_sub - tj_mid) < buffer_tiles)
  M08  ->  ... <= buffer_tiles ...                                    15 passed, 1 deselected / 16 passed
  src/dataset.py:766   buffer_tiles = int(metadata.get("buffer_tiles", 0))
  M19  ->  buffer_tiles = 0                                           15 passed, 1 deselected / 16 passed
  ```
- **Self-refutation attempted:** I checked whether the *count* assertions would catch M19 —
  `test_within_image_packaged_test_tile_counts_match_metadata` (`:411-423`) compares `len(x_test)`
  against `fold_pkg["n_test_tiles"]`, but both come from the same mutated packaging pass, so it is
  reflexive; the authoritative counts (`n_test_tiles_per_scale`, computed by the *splitter* with the
  correct buffer) are never compared against the packaged rows. I also confirmed M08 is a genuine
  behaviour change, not equivalent: at `buffer_tiles=1` on the fixture it drops 3× the rows.
- **Fix:** make the buffer assertion two-sided — assert the kept row count equals the expected
  `total − (cut row + cut column − overlap)` — and add one packaging test built with
  `buffer_tiles=1` asserting `set(zip(x_train.ti, x_train.tj))` excludes the cut lines and that the
  packaged counts equal the splitter's `n_train_tiles_per_scale`.

### tests-deep-within-image-4 — `_within_image_fold_summary`'s "finest scale" statistics are unasserted and reach a printed table
- **Severity:** low
- **Liveness:** live-shipped (notebook 09's fold-composition table)
- **Confidence:** high
- **Where:** `src/dataset.py:244-258`; consumer `notebooks/_build_09.py:151-166`; no assertion anywhere
  in `tests/`

`finest_px = min(int(s) for s in quadrant_definitions.keys())` (`:244`) selects the scale whose tile
count and mean `fractional_area` are written into every fold's `test_summary` as `n_tiles_finest` /
`frac_mean_finest_avg`. Changing `min` to `max` (M12) makes those fields describe S=64 instead of S=8 —
a ~64× smaller tile count and a different mean — and the suite passes. `notebooks/_build_09.py:161`
reads `ts['frac_mean_finest_avg']` straight into a printed fold-composition table, so the wrong number
is displayed under the right label. This is the within-image twin of the sibling area's
`tests-deep-splits-6(3)` (`_fold_summary` entirely unasserted); filing it because the consumer here is
a reader-facing table rather than dead metadata.

- **Failure scenario:** the scale map gains an entry, or `min`/`max` is edited during a refactor; the
  notebook then reports S=64 statistics as "finest", inflating apparent per-fold abundance because
  coarse tiles are far more likely to contain a boulder (base rate 0.51 at S=8 vs 0.94 at S=64, per
  R45's population descriptives).
- **Evidence:** `M12 fast=SURVIVED full=SURVIVED | 15 passed, 1 deselected | 16 passed`.
- **Self-refutation attempted:** I checked whether `test_within_image_packaged_test_tile_counts_match_metadata`
  covers it — it compares against the *packaging* counts (`per_fold[...]["n_test_tiles"]`), which are
  computed independently in `_package_within_image_split:796-797` and never against the splitter's
  `test_summary`. The two count paths are never reconciled by any test.
- **Fix:** one assertion in `test_within_image_packaged_test_tile_counts_match_metadata`:
  `assert fold_pkg["n_test_tiles"] == sum(meta_fold["n_test_tiles_per_scale"].values())`, which ties the
  splitter's summary to the packaged rows and kills M12 and M11 together.

---

## Refuted by my own check

- **"M07 — swapping the quadrant code weights (`2*ti+tj` → `ti+2*tj`) is a coverage gap."** Discarded
  as **benign**. Measured on the fixture *and* on all 9 real images: the partition is byte-identical
  and the mutation is a pure relabelling that exchanges codes 1 and 2 (`relabelling={0:[0], 1:[2],
  2:[1], 3:[3]}`, `partition_identical=True` everywhere). Nothing in `src/` reads a quadrant index
  semantically — the group arrays only need *distinct* codes (all three assertions at `:406-408` hold
  identically), and the published within-image number is a per-image mean over all four quadrants
  (`notebooks/_build_10.py:799-802`, `groupby('held_out_obs_id').agg(mean)`), which is invariant under
  relabelling. Unlike the LOIO fold-order case (`tests-deep-splits-4`), a quadrant index has no external
  referent — no image name, no banked per-fold artifact is keyed to it — so there is nothing for the
  relabelling to break. Not filed.
- **"M11 — `n_train_tiles_per_scale` ignoring the keep-mask is a coverage gap."** Discarded as an
  **equivalent mutant under every shipped configuration**. When `buffer_tiles == 0` (both `config.yaml`
  and `config_v2.yaml`), `keep` is `True` exactly when the row's scale is in `quadrant_definitions`,
  which is exactly the condition `q_arr >= 0` already in the expression — so
  `(q_arr != q) & (q_arr >= 0) & keep` and `(q_arr != q) & (q_arr >= 0)` are identically equal, by
  construction, not by fixture accident. Independently, `n_train_tiles_per_scale` has **no consumer**:
  grepping `src/`, `scripts/`, `scripts/probes/` and `notebooks/` finds it written by
  `src/dataset.py:306` and read nowhere. Two independent reasons; not filed.
- **"The slow test is unsafe to run."** It is not. `test_within_image_4fold_on_priority10_yields_32_folds`
  (`:430-445`) calls only `json.loads` and `discover_obs_ids` (a glob) — no producer, no `cfg.output_dir`,
  no write. I grepped the file for `cfg.`/`output_dir`/`cache_dir` before running anything and ran
  everything against a read-only scratchpad copy of `dataset/` regardless. (Same conclusion the
  `tests-deep-splits` sibling reached about its own slow tests.)
- **"The fast/full gap hides coverage."** It does not — it is exactly zero, and structurally so: the
  single slow test never invokes the splitter, so no mutation of `src/dataset.py`'s quadrant code can
  reach it. Every mutant scored identically under `-m "not slow"` and under the full file.
- **"The stale v2 split is R45, already filed."** Related but distinct, and I nearly discarded it on
  that basis. R45's incidental measures the JSON against the *sweep run's* fold membership and concludes
  (in the independent second pass) that the published conclusion is insensitive to it. My measurement is
  JSON vs *recomputation from the labels*, which is a different comparison, reaches the same 3.53 %, and
  yields a claim R45 does not make: **no test in the tree could detect it, and the only test that reads
  a real split JSON stops one field short.** Filed as finding 2, explicitly as an extension.
- **"The floor-snap is a knife-edge, so the v2 drift is rounding jitter."** Refuted by measurement: the
  median distance from the raw median to the next snap boundary is 5.0 of 8 finest tiles (only 7 of 76
  axes are within 1 tile of a boundary), so a whole-step drift in 29 of 38 images is a genuine movement
  of the label grid, not quantisation noise.
- **"The `unique_train == 3` assertion is weak."** False — it is the best assertion in the file and does
  real work (it alone killed both M16 and M17, i.e. a train/test self-leak *and* wrong group codes). Its
  problem is exclusively that the fixture cannot present it with an image whose quadrants are unequal.

## Verified clean

Named by the mutant that killed each — these are the things the suite genuinely does pin:

- **Cross-scale quadrant coherence** — every S=8 tile lands in the same quadrant as its S=16/32/64
  parent. Killed **M02** (removing the floor-snap to the coarsest factor) and **M06** (predicate
  `>=` → `>`), both via `test_quadrant_cuts_are_strictly_coherent_across_scales` (`:131-160`). This test
  is doing real work: it checks both the `defs` divisibility chain *and* per-tile parent agreement via a
  merge, so it catches both an arithmetic and a predicate error.
- **The buffer is a union, not an intersection** — dropping the cut row **and** the cut column, not just
  their intersection. Killed **M09** via `test_within_image_buffer_drops_boundary_tiles` (`:282-297`).
  (Only this direction; see finding 3 for the other.)
- **No train/test self-leak in the packaged rows, and group codes really are quadrant indices.** Killed
  **M16** (train rows including the test quadrant) and **M17** (group arrays storing the obs code) via
  `test_within_image_groups_have_3_unique_train_codes_per_fold` (`:385-408`). Note the contrast the
  sibling area drew: this is precisely the content-level group assertion the **LOIO** arm lacks
  (`tests-deep-splits-3`). The within-image arm got it right.
- **`_split_metadata_hash` covers the quadrant cuts.** `src/dataset.py:392-396` hashes `"folds"`, and
  `quadrant_definitions` lives inside each fold dict, so two splits with different cuts do get different
  hashes. (What it cannot do is detect that the folds disagree with today's labels — finding 2.)
- **`_package_within_image_split` uses the *declared* cuts, not re-derived ones** (`:773`), so the
  packaged parquets are internally consistent with the split JSON. The vintage problem is upstream of
  packaging, not inside it.
- **Also genuinely pinned** (not separately mutated, but the assertions are non-vacuous): each image
  appears as `test_obs_id` in exactly 4 folds with quadrants `[0,1,2,3]` (`:204-213`); the 4 test sets
  are disjoint and their union is the whole image (`:216-236`); `train_obs_ids == [test_obs_id]`, i.e.
  training never leaves the image (`:239-244`); `EMPTY_TRUTH_OBS_ID` (`ESP_065711_1545`) is excluded and
  `n_folds` reflects the exclusion (`:247-254`); split determinism (`:257-264`); the
  `n_folds != n_images * n_folds_per_image` guard **including its message** (`:300-318`); the
  `labels_dir`-required guard including its message (`:321-334`); package metadata round-trip and
  `split_hash` preservation (`:341-360`).

## Coverage note

- **Read in full:** `tests/test_within_image_split.py` (445 lines, line by line, before mutating);
  `src/dataset.py:121-318` and `:735-841` (the within-image code) plus `:390-401` and `:404-505`
  (`_split_metadata_hash`, `build_split`).
  **Read in part:** `notebooks/_build_09.py:140-180` (to confirm `frac_mean_finest_avg` is displayed);
  `docs/review_2026-07-31/verify/R45.md` and `tests-deep-splits.md` (to extend rather than duplicate);
  `DECISIONS.md:2529-2552` (the 2026-06-10 coreg y-sign fix, checked as a candidate cause of the v2
  drift and *not* confirmed — see below).
- **Method:** `src/`, `tests/`, `pyproject.toml`, `config.yaml` and a **read-only** copy of
  `dataset/{labels,splits}` in a scratchpad `mutroot/`; pytest run with `cwd=mutroot` so `import src`
  resolves to the mutated copy. **The repo's `src/` and `tests/` were never modified, no producer was
  called, and no test touching a live tree was run.** All repo-tree access outside `mutroot` was pandas /
  `json.loads` reads of `dataset/` and `dataset_v2/` parquets and JSON. Baseline re-confirmed pristine
  after the last mutant: 16 passed.
- **16 mutants seeded, each run twice** (`-m "not slow"` and full). Raw results in
  `<scratchpad>/results.jsonl`; drivers `mutants.py`, `run_mut.py`; equivalence and R45 probes
  `wi_equiv.py`, `wi_m13.py`, `wi_r45.py`, `wi_snap.py`, `wi_forensic.py`, `wi_inv.py`.
- **Every survivor was checked for equivalence** by executing pristine and mutated logic side by side on
  the tests' own fixture *and* on the 9 real v1 label parquets, and by grepping for a consumer of each
  affected field. Two (M07, M11) failed that check and were discarded rather than counted, giving the
  honest **9 of 14 (64 %)**.
- **Could not check / left open:** *why* `dataset_v2/splits/within_image_4fold.json` disagrees with
  `dataset_v2/labels/`. I eliminated three explanations (the cut code has not changed since `5ba0a07`;
  no alternative cut recipe reproduces the stored values better than the current one; the per-scale tile
  inventory is unchanged) and one attractive one (the labels' mtimes, 2026-06-10 18:19–18:26, *predate*
  the split's `written_at_iso` of 2026-06-11T21:28Z, so a simple "split built before the y-sign fix"
  story does not hold). The residual explanation is that the JSON was produced against a labels tree
  that is not the one now on disk. That belongs to the labeling / artifact areas, not to a test review;
  what belongs here is that **no test can see it**. I also did not run the full 490-test suite, and did
  not mutate `_join_one_image` / `_split_columns` / `package_split`'s dispatch, which the
  `tests-deep-splits` sibling already covered.
