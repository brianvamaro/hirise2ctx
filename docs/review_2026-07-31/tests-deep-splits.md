# Review area: tests-deep-splits

- **Reviewed at commit:** bd19da8
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified)

Target: `tests/test_splits.py` (399 lines), covering `src/dataset.py` (842) — **invariant 6**
(group-aware leave-image-out splitting).

## Headline

`tests/test_splits.py` **pins the split *metadata* and does not pin the *packaged data***.
Every assertion about `package_split` is a **row-count** or a **length**; not one assertion
anywhere in the file (or in the two neighbouring files that touch the same code) checks *which
`obs_id`s* are in `X_train_fold{k}.parquet` / `X_test_fold{k}.parquet`, or *which columns* are on
the X side. Consequently a regression that makes the packaging step **fall back to a random
per-tile split** — the exact violation of invariant 6, the one that would invalidate every number
this project reports — leaves the suite **fully green**. So does a regression that puts
`fractional_area` (the target) into the feature matrix.

The abstract group-leak property *is* well tested — but only in the JSON the splitter returns,
which is not what any model reads.

### Mutation results

16 single-point defects seeded into a scratchpad copy of `src/dataset.py`; the real (unmodified)
`tests/test_splits.py` run against each.

| survival | `-m "not slow"` | full file |
|---|---|---|
| all 16 seeded | 10 survived (62 %) | 10 survived (62 %) |
| 14 after discarding 2 I proved equivalent/benign | **8 survived (57 %)** | **8 survived (57 %)** |

**The fast/full gap is exactly zero: the two `slow` tests killed nothing.** That is not an
accident — `test_priority10_loio_9fold_matches_sweep` and
`test_priority10_all_parquet_row_count_matches_sum_of_test_folds` only call
`load_split_metadata` / `load_package_metadata` / `discover_obs_ids` / `pd.read_parquet`. They
audit the **artifact already on disk**; they never invoke `build_split` or `package_split`, so no
mutation of the splitter can reach them. CLAUDE.md's documented dev loop (`-m "not slow"`) loses
nothing here.

| id | mutation | verdict |
|---|---|---|
| M01 | `package_split` silently falls back to a **random tile split** (row counts preserved) | **SURVIVED** |
| M02 | `_assign_loio_9fold` fold order reversed (fold *i* ≠ `sorted(obs)[i]`) | **SURVIVED** |
| M06 | `_join_one_image` ignores `scale_filter` | **SURVIVED** |
| M08 | `_split_columns` stops excluding `LABEL_COLUMNS` from X (target → features) | **SURVIVED** |
| M10 | `_split_metadata_hash` drops `"folds"` from the hashed keys | **SURVIVED** |
| M12 | `package_split` writes all-zero `obs` codes to `groups_*.npy` | **SURVIVED** |
| M13 | `_fold_summary` reports `n_tiles_finest` as `n_tiles_total` | **SURVIVED** |
| M15 | `build_image_inventory` takes `BoulderLabel` from the first manifest row for every image | **SURVIVED** |
| M03 | `build_split`: `train_obs_ids` = all images (leak in the metadata) | killed |
| M04 | `package_split`: train rows from all images (leak in the parquets) | killed |
| M05 | `iter_train_batches` yields the **test** images | killed |
| M07 | label↔feature merge key truncated to `obs_id` (cartesian join) | killed |
| M09 | `stratification='none'` ⇒ `n_folds == n_images` guard removed | killed |
| M11 | `_assign_size_balanced_kfold` ignores `seed` | killed |
| M14 | label-balance tie-break dropped from the greedy k-fold | *equivalent* (see Refuted) |
| M16 | `discover_obs_ids` returns unsorted glob order | *benign* (see Refuted) |

Cross-check: M01, M02, M08 and M12 also survive `test_splits.py` **+** `test_within_image_split.py`
**+** `test_modeling_loaders.py` run together (41 passed, 3 deselected, all green).

---

## Findings

### tests-deep-splits-1 — A `package_split` fallback to a random per-tile split passes the whole suite
- **Severity:** high
- **Liveness:** live-shipped (`dataset_v2/packaged/loio_nfold` trains the frozen recipe and the deployed head)
- **Confidence:** high
- **Where:** `tests/test_splits.py:245-267` (`test_package_split_round_trip`), `:304-320`
  (`test_package_groups_npy_aligns_with_x_rows`), `:269-287`; code under test `src/dataset.py:656-685`

Invariant 6 is enforced in `src/dataset.py` in two independent places: `build_split` computes
`train = sorted(all_obs_set - set(test))` (`:485`), and `package_split` materialises rows by
`per_image[o] for o in train_obs` (`:660-661`). The suite tests **only the first**.
`test_no_obs_id_in_both_train_and_test_in_any_fold` (`:208-217`) intersects two *lists of strings in
the metadata dict* — it never opens a parquet. The packaging tests assert only
`n_test_tiles == 10` / `n_train_tiles == 30` and `len(x_train) == len(groups)`. A mutation that keeps
those counts exact while drawing the rows at random from all images is invisible.

- **Failure scenario:** someone refactors `package_split` (e.g. to concat once and slice, or to add
  a shuffle for a "balanced batch" experiment) and the per-image indexing is lost. Every fold's
  `X_test_fold{k}.parquet` then contains tiles from all 9/38 images, `src/modeling/loaders.py:123-128`
  loads it unchanged, and every reported LOIO number becomes an inflated random-tile number. Suite green.
- **Evidence** — mutant M01 applied to `src/dataset.py:660-661`, then packaged the same 4-image
  fixture `test_package_split_round_trip` uses:
  ```
  fold0 declared test_obs_ids  : ['OBS_000']
  fold0 X_test  obs_ids on disk: ['OBS_000','OBS_001','OBS_002','OBS_003']  n_rows = 10
  fold0 X_train obs_ids on disk: ['OBS_000','OBS_001','OBS_002','OBS_003']  n_rows = 30
  n_test/n_train reported      : 10 30      (the test asserts exactly 10 / 30 -> passes)
  IMAGES PRESENT IN BOTH TRAIN AND TEST: ['OBS_000','OBS_001','OBS_002','OBS_003']
  pytest tests/test_splits.py -q            -> 17 passed
  ```
- **Self-refutation attempted:** (a) does another file cover it? `tests/test_within_image_split.py`
  reads `X_test_fold{k}.parquet` at `:381` and `:422` — but only through
  `_package_within_image_split`, a *different* function; running M01 against
  `test_splits.py + test_within_image_split.py + test_modeling_loaders.py` gives 41 passed.
  (b) does a downstream consumer guard it? `src/modeling/loaders.py:151` takes `held_out_obs_ids`
  from `metadata["per_fold"][k]["test_obs_ids"]` — i.e. from the *metadata*, so it would report the
  intended held-out image while scoring contaminated rows. No guard. (c) `DECISIONS.md` records no
  deliberate choice to test counts only. It survives.
- **Fix:** one line in `test_package_split_round_trip`:
  `assert set(pd.read_parquet(out_dir/PACKAGED_SUBDIR/"loio_4fold"/f"X_test_fold{k}.parquet")["obs_id"]) == set(fold["test_obs_ids"])`
  and the complementary `isdisjoint` for the train side.

### tests-deep-splits-2 — Nothing pins the X/y column split, so a target-into-features leak is untested
- **Severity:** high
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `src/dataset.py:590-601` (`_split_columns`); no test in `tests/` asserts on `x_cols`/`y_cols`

`_split_columns` is the only thing keeping `fractional_area`, `boulder_area`, `boulder_count`,
`count_density`, `binary_by_*` out of `X_*_fold{k}.parquet`. `package_split` records
`n_train_x_cols` / `n_y_cols` in the metadata, but **no test asserts either number**, and no test
reads a `y_*` parquet at all. Downstream there is no second line of defence:
`src/modeling/loaders.py:91-95` drops only tile keys, `config_hash_feat` and `patch_idx_S*` — every
other column becomes a model feature.

- **Failure scenario:** the `set(label_cols)` term is dropped from `excluded` (a plausible edit when
  someone adds a new label column and rewrites the set expression). The target itself is then a
  feature; every head reports a near-perfect AUC; nothing in the suite objects.
- **Evidence** — mutant M08 (`src/dataset.py:599`, dropping `| set(label_cols)`):
  ```
  pristine X cols: ['intensity_mean', 'shadow_fraction', 'config_hash_feat']
  mutated  X cols: ['boulder_area', 'boulder_count', 'tile_area', 'fractional_area',
                    'binary_by_area', 'binary_by_count', 'count_density',
                    'intensity_mean', 'shadow_fraction', 'config_hash_feat']
  pytest tests/test_splits.py -q  -> 17 passed
  (also 41 passed with test_within_image_split.py + test_modeling_loaders.py)
  ```
- **Self-refutation attempted:** `tests/test_within_image_split.py:363-382` looked promising — it is
  named "…have_expected_columns" — but it only asserts that `TILE_KEY_COLUMNS` are *present* on both
  sides, never that label columns are *absent* from X. `tests/test_modeling_loaders.py` builds its
  own synthetic parquets rather than calling `package_split`, so it cannot see this either.
- **Fix:** assert `set(LABEL_COLUMNS) & set(pd.read_parquet(X_train_fold0).columns) == set()` in
  `test_package_split_round_trip`.

### tests-deep-splits-3 — `groups_*.npy` is checked for length only, never for content
- **Severity:** medium
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `tests/test_splits.py:304-320`; code at `src/dataset.py:652`, `:673-676`

`test_package_groups_npy_aligns_with_x_rows` asserts `len(x_train) == len(groups)` and nothing else.
Collapsing every ObsId to the same integer code passes. Those arrays are the group key for
`_standardize_matrix_per_group` (`src/modeling/loaders.py:189`) and for the inner-validation rotation
in `run_loio`, so degenerate codes would silently turn per-image standardisation into a single global
standardisation and make any inner GroupKFold non-grouped.

- **Failure scenario:** `obs_to_int` is rebuilt from a source that no longer varies per image (e.g.
  keyed on a constant, or `enumerate` over a 1-element list); groups become all-zero; the per-image
  transforms in `loaders.py` silently become global; no test fires.
- **Evidence** — mutant M12 (`src/dataset.py:652`, `{obs: i ...}` → `{obs: 0 ...}`): 17 passed / 41 passed.
- **Self-refutation attempted:** the analogous assertion **does** exist for the within-image arm —
  `test_within_image_groups_have_3_unique_train_codes_per_fold`
  (`tests/test_within_image_split.py:385-408`) checks `len(unique_train) == 3`,
  `len(unique_test) == 1` and no code collision. The LOIO arm simply never got the equivalent check,
  which also confirms the check is considered worth having by the authors.
- **Fix:** in `test_package_groups_npy_aligns_with_x_rows`, additionally assert
  `set(np.unique(groups_train)) == {obs_to_int[o] for o in fold["train_obs_ids"]}` and disjointness
  from the test codes.

### tests-deep-splits-4 — Fold *identity* (fold *i* ↔ which ObsId) is not pinned anywhere
- **Severity:** medium
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `tests/test_splits.py:139-151`, `:208-217`, `:220-228`; code at `src/dataset.py:321-328`

Every LOIO assertion is a **set/length** assertion: `set(flat) == set(obs_labels)`,
`len(flat) == 9`, `len(set(flat)) == 9`, `sorted(flat) == sorted(obs_labels)`. None asserts that
fold 0's test image is `sorted(obs_ids)[0]`. `labeling-deep-artifact.md:319-326` leans on exactly
that property ("fold *i* is `sorted(obs_ids)[i]`, content-independent") to conclude the v2 LOIO
splits structurally cannot drift. That conclusion is correct about *today's code* but is **not
defended by any test**.

- **Failure scenario:** `_assign_loio_9fold` is changed to iterate `inventory.index` unsorted, or a
  future stratified LOIO variant reorders folds. Every per-fold artifact keyed by fold index
  (`X_*_fold{k}`, per-fold banked calibrators, `reports/figures/*fold*`) then refers to a different
  image than the one it did before, while all cross-run comparisons still line up by index.
- **Evidence** — mutant M02 (`sorted(inventory.index)` → `sorted(..., reverse=True)`):
  ```
  pristine fold->test: [['OBS_000'], ['OBS_001'], ['OBS_002'], ['OBS_003']]
  mutated  fold->test: [['OBS_003'], ['OBS_002'], ['OBS_001'], ['OBS_000']]
  split_hash differs: True        pytest -> 17 passed  (41 passed across 3 files)
  ```
- **Self-refutation attempted:** I first tried the weaker mutation (drop `sorted()` entirely) and it
  is a **no-op on these fixtures** — the synthetic inventory is constructed from `sorted(obs_labels.items())`
  (`tests/test_splits.py:42`), so the index is already sorted. That is the fixture-blindness shape
  from `labeling-deep-tests`; I therefore used the strictly stronger reversed variant, which is an
  unambiguous behaviour change, and it still survived. The `split_hash` *does* change — but no test
  compares a hash against a stored constant (see finding 5).
- **Fix:** one line in `test_loio_9fold_uses_each_image_exactly_once_in_test`:
  `assert [f["test_obs_ids"] for f in meta["folds"]] == [[o] for o in sorted(obs_labels)]`.

### tests-deep-splits-5 — `split_hash` is only ever compared to itself, so it is not pinned as a partition fingerprint
- **Severity:** medium
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `tests/test_splits.py:180-192` (`test_split_reproducibility_with_seed`), `:266`;
  code at `src/dataset.py:390-401`

The only two `split_hash` assertions are `m1["split_hash"] == m2["split_hash"]` (same inputs, same
code) and `loaded["split_hash"] == meta["split_hash"]` (round-trip of the same object). Both are
reflexive: they verify the hash is *deterministic*, never that it is *discriminative*. Removing
`"folds"` from the hashed key list — which is precisely what would make the hash stop detecting a
changed partition — passes.

- **Failure scenario:** the hash silently stops covering the partition; `DATA_DICTIONARY.md:464`'s
  documented consistency rule ("mismatch indicates the package and split have diverged") then
  reports a match for two genuinely different partitions, and the stale-package check that R04 and
  `other-scripts-1` both rely on becomes a no-op.
- **Evidence** — mutant M10 (`src/dataset.py:393`), hashing two *different* partitions of the same 4
  images:
  ```
  M10 hash of forward partition : 995dffe1f573d8aa
  M10 hash of REVERSED partition: 995dffe1f573d8aa
  distinguishes the two partitions: False        pytest -> 17 passed
  ```
- **Self-refutation attempted:** I checked whether the slow tests compare the on-disk hash to
  anything — they do not (`test_priority10_*` never reads `split_hash`). And `R04` /
  `other-scripts-1` are about the *artifact* being stale; this is the distinct claim that the
  *hash function's discriminative power* has no test, which is why those two findings could not be
  caught by a regression test either. Not re-filing R04.
- **Fix:** add `assert build_split(..., inventory=inv_a)["split_hash"] != build_split(..., inventory=inv_b)["split_hash"]`
  for two inventories that differ only in the partition.

### tests-deep-splits-6 — Three tests that cannot fail, or whose fixture cannot express the defect
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `tests/test_splits.py:323-337`, `:117-125`, `:372-387`

Three separate instances of the two shapes the brief flagged:

1. **`test_scale_filter_restricts_emitted_rows` (`:323-337`) cannot fail.** Its own docstring says it:
   *"scale_filter=[8] on synthetic parquets that only have S=8 leaves rows unchanged."* The fixture
   emits a single scale, so the assertion `pkg["per_fold"][0]["n_test_tiles"] == 4` is the *unfiltered*
   count. Deleting the filter body in `_join_one_image` (`src/dataset.py:542-543`) passes (mutant M06).
   The real `loio_nfold` package is built with a scale filter, and `sweep`/`load_fold(scale_idx=…)`
   depend on the emitted scales being right.
2. **No fixture reaching `build_image_inventory` has more than one `BoulderLabel`.** Every call site
   (`:122`, `:252`, `:275`, `:295`, `:311`, `:330`, `:350`) passes an all-`"Boulder rich"` manifest,
   and `test_build_image_inventory_round_trip` asserts `(inv["BoulderLabel"] == "Boulder rich").all()`.
   Replacing the per-image lookup `manifest.loc[obs, "BoulderLabel"]` (`src/dataset.py:108`) with the
   *first row's* label passes (mutant M15) — yet the production manifest is 5 rich / 2 poor /
   2 unknown, and that column is the sole input to `loio_3fold_balanced`'s stratification. The
   stratification tests dodge this by building the inventory by hand
   (`_synthetic_inventory`, `:39-51`) instead of through the production function.
3. **`_fold_summary` (`src/dataset.py:372-387`) is entirely unasserted.** Its output is written into
   every split JSON as `test_summary`/`train_summary`; swapping `n_tiles_total` for `n_tiles_finest`
   passes (mutant M13).

- **Self-refutation attempted:** for (1) I checked whether any *other* test supplies a multi-scale
  fixture to `package_split` — `test_within_image_split.py` does build multi-scale parquets, but it
  never passes `scale_filter`, so the filter is unexercised across the whole test tree. For (2) I
  confirmed M15 is invisible only because of the fixture, not because the code path is unreachable:
  the real manifest is heterogeneous, so the mutation is a genuine production behaviour change.
- **Fix:** give `_write_synthetic_image_parquets` a second scale and assert the filtered count
  differs from the unfiltered one; give one packaging fixture a mixed manifest.

---

## Refuted by my own check

- **"The slow tests are unsafe to run."** They are not — `test_priority10_loio_9fold_matches_sweep`
  (`:373-387`) and `test_priority10_all_parquet_row_count_matches_sum_of_test_folds` (`:390-399`)
  only call `load_split_metadata`, `load_package_metadata`, `discover_obs_ids` and `pd.read_parquet`.
  No producer, no `cfg.output_dir`, no write. I grepped the file for `cfg.`/`output_dir`/`cache_dir`
  before running anything, and ran everything against a scratchpad copy of `dataset/` regardless.
- **"M14 — dropping the label-balance tie-break in `_assign_size_balanced_kfold` survives."**
  Discarded: it is an **equivalent mutant** on every distribution I could construct.
  `key=(label_counts[label][k], len(folds[k]), k)` and `key=(len(folds[k]), k)` produce byte-identical
  fold assignments for 5R/2P/2U at seeds 0 and 7, 6R/3P at seeds 0 and 3, and 8R/4P k=4 seed 1.
  (Processing label groups largest-first keeps the per-fold totals within 1, which makes the two keys
  order folds identically.) So `test_stratified_3fold_balances_image_count` is not *failing* to pin
  the tie-break — the tie-break is arguably redundant. Not filed.
- **"M16 — `discover_obs_ids` losing its `sorted()` survives."** Discarded as benign: the mutation is
  a no-op on Windows/NTFS (glob already returns alphabetical order, so
  `test_discover_obs_ids_finds_parquets` passes vacuously on the dev platform), and even on Linux it
  has no downstream consequence, because `build_split` re-sorts (`src/dataset.py:429`) and
  `_assign_loio_9fold` sorts again (`:328`). Worth knowing, not worth filing.
- **"`test_within_image_split.py` covers the LOIO packaging gaps."** It does not: `package_split`
  dispatches on `metadata["kind"]` (`src/dataset.py:632`), so every within-image test exercises
  `_package_within_image_split`, a physically different function body.
- **"R04 / `other-scripts-1` already cover the `split_hash` finding."** Related but distinct — those
  are about a stale artifact on disk; finding 5 is about the hash function's discriminative power
  having no test, which is *why* neither of those could have been caught by a regression test. Not
  re-filed.
- **"The LOIO group-leak property is untested."** False — it is the best-tested thing in the file
  (see below). The gap is that it is tested only in the metadata, not in the packaged rows.

## Verified clean

Named by the mutant that killed each — these are the things the suite genuinely does pin:

- **Group leakage in the split metadata.** `train_obs_ids` must exclude `test_obs_ids` in every fold.
  Killed M03 (`train = sorted(all_obs_set)`) via `test_no_obs_id_in_both_train_and_test_in_any_fold`
  **and** `test_package_split_round_trip`.
- **Image-level train/test *sizes* in the packaged parquets.** Killed M04 (`train_obs =
  manifest_obs_ids`) via `test_package_split_round_trip`'s `n_train_tiles == 30`. This catches a
  whole-image leak; it does **not** catch a per-tile leak (finding 1).
- **The streaming iterators' membership.** `iter_train_batches` / `iter_test_batches` must yield one
  frame per ObsId of the right side, and every row's `obs_id` must be in that side. Killed M05
  (train iterator yielding test ids) via `test_streaming_iterator_yields_one_dataframe_per_obs`
  (`:344-366`) — this is the one place membership *is* asserted. (Cross-ref R22, filed separately,
  concerns the within-image iterators, not these.)
- **The label↔feature join key.** Killed M07 (`on=TILE_KEY_COLUMNS[:1]`, a cartesian join) via three
  tests at once: `test_package_split_round_trip`, `test_package_emits_all_parquet_when_enabled`,
  `test_scale_filter_restricts_emitted_rows`.
- **The `stratification='none'` misconfiguration guard.** Killed M09 via
  `test_split_none_stratification_requires_n_folds_equals_n_images` — including the error-message
  match, so the guard cannot be silently weakened.
- **`seed` is actually plumbed into the balanced k-fold RNG.** Killed M11
  (`default_rng(seed)` → `default_rng(0)`) via `test_split_different_seed_can_change_assignment`.
  This test is doing real work.
- **Also genuinely pinned** (not separately mutated, but the assertions are non-vacuous):
  each image appears in exactly one test fold (`:139-151`); 3/3/3 fold sizes and `[1,2,2]` rich
  distribution for the 9-image manifest (`:154-177`); split determinism under a fixed seed
  (`:180-192`); the splitter is not hardcoded to 9 images (`:220-228`); `all.parquet` contains each
  tile exactly once tagged with its test fold (`:269-287`); `emit_all_parquet=False` really suppresses
  it (`:289-301`).

## Coverage note

- **Read in full:** `tests/test_splits.py` (399 lines, line by line, before mutating);
  `src/dataset.py` (842); `tests/conftest.py`; `pyproject.toml`.
  **Read in part:** `src/modeling/loaders.py:60-190` (to judge whether any downstream consumer
  re-derives the group codes or filters the feature columns — it does not);
  `tests/test_within_image_split.py:336-445`; `docs/review_2026-07-31/labeling-deep-artifact.md`
  (splits verdict, not redone).
- **Method:** `src/`, `tests/{conftest,test_splits,test_within_image_split,test_modeling_loaders}.py`,
  `pyproject.toml`, `config.yaml` and a read-only copy of
  `dataset/{splits/loio_9fold.json, labels/*.parquet, packaged/loio_9fold/{metadata.json,all.parquet}}`
  copied to a scratchpad `mutroot/`; pytest run with `cwd=mutroot` so `import src` resolves to the
  mutated copy. **The repo's `src/` and `tests/` were never modified and no producer was called.**
  Baselines in `mutroot`: `-m "not slow"` → 15 passed / 2 deselected in 1.8 s; full → 17 passed in
  2.7 s (the two slow tests execute for real against the copied artifacts, they do not skip).
- **16 mutants seeded, each run twice** (`-m "not slow"` and full). Raw results in
  `<scratchpad>/results_splits.jsonl`; drivers `mutate_splits.py`, `verify_splits.py`,
  `crosscheck.py`, `m14b.py`.
- **Every survivor was verified to be a genuine behaviour change**, not an equivalent mutant, by
  executing the pristine and mutated functions side by side on the tests' own fixtures
  (`verify_splits.py` output is quoted in the findings). Two mutants (M14, M16) failed that check and
  were discarded rather than counted as survivors.
- **Could not check:** whether the *within-image* `package_split` branch has the same random-split
  blind spot — out of scope here, it is `tests-deep-within-image`'s target (though I did confirm its
  group-code assertion at `test_within_image_split.py:385-408` is the check the LOIO arm lacks).
  I also did not attempt mutants in `_compute_quadrant_definitions` / `_quadrant_array_for_image` /
  `_assign_within_image_kfold` for the same reason. I did not run the full 490-test suite.
