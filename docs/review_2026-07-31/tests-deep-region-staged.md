# Review area: tests-deep-region-staged

- **Reviewed at commit:** 577277f (the four target files are byte-identical to bd19da8, where the
  baseline below was first taken)
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified)

Target: `tests/test_region_staged.py` (409 lines), the end-to-end suite for the F-build **Stage D**
composite driver `scripts/f_region_staged.py` (442) and its core `src/fcompose.py` (269), plus the
`src/leveling.py` helpers it reaches (`sigmoid`, `logit`, `TILE_M`, `EPS`).

## Headline

The suite **pins the composite arithmetic well and the shipping decision badly.** The composite rule
(mean of leveled logits, one sigmoid, offset sign, lattice pitch, provenance severity) is genuinely
defended — seven mutants die there. But the file's own docstring says its most important job is
*"the §0.1 guard-1 rule that an AMBIGUOUS trend-guard verdict must NOT silently ship a headline map"*,
and of that rule it pins only **whether** a plain-named map exists, never **which variant** it
contains: the two tests that assert the headline map equals a named variant use a 2-frame symmetric
bias fixture in which **all three variants are bit-identical**, so they cannot fail.

Separately, **`pfree` — the variant Brian declared SHIPPED on 2026-07-30 and the one the HARD ABORT
verdict was pronounced on — is never composited by any test**, and one assertion actively pins the
pre-`pfree` variant list, so a fixture that exercised it would *break* the suite.

### Baseline (carried forward, re-confirmed at 577277f)

```
pytest tests/test_region_staged.py -q -m "not slow"  ->  18 passed in 4.15 s
pytest tests/test_region_staged.py -q                ->  18 passed in 4.09 s
--collect-only                                       ->  18 tests collected
```

**This file contains zero `slow`-marked tests, so the fast/full gap is structurally zero** — not
because the slow tests are vacuous (the `-features`/`-splits` siblings' reason) but because there are
none. CLAUDE.md's documented dev loop loses nothing here.

### Safety pre-check (carried forward; it came back clean, and I re-verified)

No `cfg.output_dir`, `cfg.cache_dir`, `config.` path attribute or `@pytest.mark.slow` appears in the
file. Every path is under `tmp_path`, and the one module global pointing at a live tree
(`sd.FIG = reports/figures`) is monkeypatched to `tmp_path/figures` by the `staged` fixture
(`tests/test_region_staged.py:91-98`). Two live-tree **reads** remain and are harmless: `seam_labels`
→ `src.striping.load_frames("T00_N00")` fails on the missing `reports/map_region/T00_N00_abundance.tif`
and is swallowed by the intended `except Exception` (`scripts/f_region_staged.py:159`); and
`test_calibration_is_applied_once_to_the_composite_not_per_frame` lets `--calibration-mosaic` default
to `models/deployable/calibration.npz` (read-only). **No producer is reachable from this file.**

### Mutation results

27 single-point defects + 1 two-point probe, seeded into a scratchpad copy of `src/` and `scripts/`;
the real, unmodified `tests/test_region_staged.py` run against each.

| survival | vs `tests/test_region_staged.py` | + the two sibling module suites |
|---|---|---|
| all 27 seeded | 17 survived (63 %) | 13 survived (48 %) |
| 25 after discarding 2 I proved equivalent/benign | **15 survived (60 %)** | **11 survived (44 %)** |

The second column adds `tests/test_fcompose.py` (234 lines, a real unit suite for the same module)
and `tests/test_leveling.py`, which I ran every `src/`-level survivor against. **All 8 survivors that
live in `scripts/f_region_staged.py` are repo-wide survivors**: `grep -l f_region_staged tests/*.py`
returns only this file, so nothing else can catch them.

| id | file | mutation | this file | sibling suites |
|---|---|---|---|---|
| M01 | fcompose | `frame_rows_cols` row/col transposed | **SURVIVED** | **SURVIVED** |
| M03 | fcompose | `OFFSET_SOURCE_CODE` severity order scrambled (`none` no longer worst) | **SURVIVED** | **SURVIVED** |
| M04 | fcompose | `overlap_dp` sign flipped (`p_min - p_max`) | **SURVIVED** | killed |
| M05 | fcompose | `dx_m` reports the MIN sub-pixel translation, not the max | **SURVIVED** | killed¹ |
| M06 | script | output rasters georeferenced with a GDAL-order affine | **SURVIVED** | (none) |
| M09 | fcompose | frame with UNKNOWN incidence becomes the best-illuminated primary | **SURVIVED** | **SURVIVED** |
| M14 | leveling | logit clip `EPS` 1e-4 → 1e-6 | **SURVIVED** | killed |
| M16 | script | partial-Stage-B headline refusal disabled | **SURVIVED** | (none) |
| M17 | script | done-check drops the HEADLINE product (the 2026-07-29 bug, restored) | **SURVIVED** | (none) |
| M18 | script | frame-index staleness guard dropped (the 2026-07-29 bug, restored) | **SURVIVED** | (none) |
| M19 | script | headline raster taken from `variants[0]`, not the named variant | **SURVIVED** | (none) |
| M20 | fcompose | `overlap_dp` emitted as 0.0 for SINGLE-frame pixels | **SURVIVED** | killed |
| M26 | script | every per-variant output raster written with an EMPTY CRS | **SURVIVED** | (none) |
| M27 | script | offsets joined POSITIONALLY instead of by `PRODUCT_ID` | **SURVIVED** | (none) |
| M28 | script | global frame LUT built in REVERSE order | **SURVIVED** | (none) |
| M02 | leveling | `TILE_M` 160 → 80 m | killed (18/18) | — |
| M07 | fcompose | composite is the SUM of leveled logits, not the mean | killed (4) | — |
| M08 | script | `full_pending_ruling` → `"full"` (guard 1 broken) | killed (1) | — |
| M10 | script | leveling offset sign flipped | killed (4) | — |
| M11 | script | Tier-2 abundance written from the Tier-1 calibrator | killed (2) | — |
| M12 | fcompose | `frame_bbox` returns TJ in the TI slots | killed (16) | — |
| M13 | fcompose | provenance keeps the BEST contributor, not the worst | killed (2) | — |
| M15 | leveling | `sigmoid` sign flipped | killed (5) | — |
| M23 | script | PARTITION composite ignores the leveling offset | killed (1) | — |
| M24 | script | `pfree` wired to `offset_residual_only` | killed (1)² | — |
| M21 | fcompose | `add_frame` in-bounds screen dropped | SURVIVED | *equivalent* (see Refuted) |
| M22 | fcompose | `frame_bbox` empty-frame sentinel neutered | SURVIVED | *benign* (see Refuted) |
| P1 | leveling | *probe*: `sigmoid`/`logit` → a consistent NON-logistic inverse pair | SURVIVED | SURVIVED |

¹ M05's only killer is `test_fcompose.py::test_the_e0_column_is_flagged_as_a_rounding_tie`, which
`pytest.skip`s when `reports/map_region/E0_N44_prob_raw.tif` is absent — a **gitignored** path. It
kills on this machine (26 tifs on disk); on a fresh clone or CI, M05 survives repo-wide.
² M24 is killed *only* by the stale variant-set assertion, not by any check of `pfree` — see finding 1.

---

## Findings

### tests-deep-region-staged-1 — `pfree`, the SHIPPED variant the abort verdict was pronounced on, is never composited by any test, and one assertion pins the pre-`pfree` variant list
- **Severity:** medium
- **Liveness:** dead-closed (F build hard-aborted `41a6f26`), but it is the coverage gap sitting closest to the abort verdict
- **Confidence:** high (executed)
- **Where:** `tests/test_region_staged.py:66-78` (`_write_stagec`, which never writes an
  `offset_logit_pfree` column), `:365` (`assert set(tiles.variant) == {"h1only","full","resid"}`);
  code at `scripts/f_region_staged.py:66-72` (`VARIANTS`), `:323-326` (the drop-a-variant path)

`scripts/f_region_staged.py:16-19` states plainly that `pfree` "was added 2026-07-30 (Brian) and is
the **SHIPPED** variant"; `full`/`resid` are "retained as the pre-declared audit trail". The fixture
`_write_stagec` emits `offset_logit`, `offset_residual_only`, `offset_source`, `incidence`,
`component`, `degree` — and no `offset_logit_pfree`. So **every one of the 18 tests runs down the
`dropped` branch and silently composites only three variants**, printing two warnings nobody asserts on.
Worse, `test_registration_report_records_the_subpixel_translation` asserts the emitted variant set is
*exactly* `{"h1only","full","resid"}` — so a fixture that supplied the column, which is the obvious
fix, makes the suite red. The test file has not been touched since `afe6fce`; `pfree` arrived in
`41a6f26`, the abort commit itself, with no test change.

- **Failure scenario:** the abort verdict (`sd(log10)` mosaic 0.170 vs resid 0.371 vs **pfree 0.532**)
  was computed downstream of Stage D's `{tile}_pfree_*` rasters. A mis-plumbed `pfree` column — the
  single most likely Stage-D slip, since it is the one variant added after the tests were written —
  would have made the decisive number an artifact, and the suite would have been green throughout.
- **Evidence** (my own probe, not a test edit — I appended the column and ran the driver directly):
  ```
  baseline stdout, EVERY test:
    ⚠ no `offset_logit_pfree` column — Stage C predates the plane-free solve; the `pfree` variant is unavailable
    ⚠ variants ['pfree'] have no offset column in .../fbuild_stagec_offsets.csv -> SKIPPED

  with the column supplied (biases [0.8,-0.3,0.1], pfree offsets = 0.25*off):
    pfree raster written : True
    pfree == expected    : True                       <- the code path is CORRECT
    variants in fbuild_staged_tiles.csv: ['full','h1only','pfree','resid']
    -> tests/test_region_staged.py:365 asserts this set == {'h1only','full','resid'}  => would FAIL

  M24 (VARIANTS["pfree"] -> "offset_residual_only"): killed by exactly ONE test,
       test_registration_report_records_the_subpixel_translation, i.e. by the stale
       variant-set assertion — not by anything that looks at pfree's values.
  ```
- **Self-refutation attempted:** (a) *Is `pfree` actually mis-wired, i.e. is the abort verdict wrong?*
  **No — I checked and it is correct.** `scripts/f_region_stagec.py:498` writes
  `"offset_logit_pfree": np.round(o_pfree, 4)` and `VARIANTS["pfree"]` reads exactly that string; my
  probe confirms the composite equals `sigmoid(mean(logit(p) + o_pfree))` to 1e-5. So this is a
  coverage gap, **not** a wrong number, and I am explicitly *not* claiming the abort should be
  reopened. That is why this is medium and not high. (b) *Does another test file cover `pfree`?*
  `grep -rn "pfree" tests/` returns `test_leveling.py` (which pins `solve_offsets_planefree`, the
  *solver*) and `test_fgates.py` — nothing exercises the Stage-D *plumbing* of the column. (c) *Is
  leaving it untested deliberate?* Nothing in `DECISIONS.md` or `PLAN_FBuild.md` says so; the test
  file simply predates the variant by two commits.
- **Fix:** two lines — add `offset_logit_pfree` to `_write_stagec`'s frame (e.g. `off * 0.25`) and
  change `:365` to `{"h1only","full","resid","pfree"}`. That alone would kill M24 for the right reason.

### tests-deep-region-staged-2 — The two tests that pin "the verdict ships the right variant" cannot fail: their fixture makes all three variants bit-identical
- **Severity:** medium
- **Liveness:** dead-closed
- **Confidence:** high (executed)
- **Where:** `tests/test_region_staged.py:286-296` (`test_headline_override_ships_an_explicitly_named_variant`),
  `:299-306` (`test_residual_verdict_ships_the_residual_variant`); code at
  `scripts/f_region_staged.py:255-264`

Both tests build the fixture with `biases = [0.5, -0.5]` and then assert
`allclose(headline_map, resid_map)`. But `_write_stagec` sets `off = -(b - median(b))`, so for a
symmetric two-frame bias set `median(b) = mean(b) = 0` and the composited mean logit is
`mean(b_f + α·o_f) = (1-α)·mean(b_f) = 0` **for every α**. h1only (α=0), resid (α=0.5) and full (α=1)
therefore produce numerically identical rasters, and the assertion is satisfied by *any* variant the
code chooses to ship. Five of the 18 tests use this bias set.

- **Failure scenario:** a refactor of the headline block (`:255-264`) — e.g. hoisting `res` out of the
  per-variant loop, or reading `variants[0]` instead of `headline` — ships the **un-leveled** h1only
  map under the plain `{tile}_prob.tif` / `_abundance.tif` names that notebook 24 and every
  `src.striping` helper read. The suite stays green; the shipped map is the one the whole exercise
  was meant to correct.
- **Evidence** — mutant **M19** (`res = accums[headline].finish()` → `accums[variants[0]].finish()`),
  plus the pristine composite run at two bias sets:
  ```
  === biases [0.5, -0.5]  (used by BOTH headline tests) ===
    max |h1only - full|  = 0.000e+00   identical=True
    max |h1only - resid| = 0.000e+00   identical=True
    max |full   - resid| = 0.000e+00   identical=True
    headline == h1only : True          headline == resid : True

  === biases [0.8, -0.3, 0.1]  (the composite-correctness tests) ===
    max |h1only - full|  = 2.499e-02   max |full - resid| = 1.250e-02   identical=False

  M19  [SURVIVED]  18 passed
  ```
- **Self-refutation attempted:** (a) *Does `test_ambiguous_verdict_writes_no_headline_map` compensate?*
  Partly, and it is a real test — it killed **M08** (`APPLY_TO_VARIANT["full_pending_ruling"] = "full"`).
  But it asserts only `not (…_prob_raw.tif).exists()`, i.e. **existence**, never content. Existence is
  pinned; identity is not. (b) *Is the degeneracy avoidable?* Yes, trivially — the
  composite-correctness tests at `:120-162` already use `[0.8,-0.3,0.1]`, and `test_resid_variant…`
  even carries an explicit `assert not np.allclose(p, full…)` clause with the comment *"else the test
  proves nothing"*. The author understood the hazard in one section and did not carry it into the
  next. (c) *Would `--headline` at least be checked?* `test_headline_override…` passes
  `--headline resid` — into the same degenerate fixture.
- **Fix:** change `[0.5, -0.5]` to `[0.8, -0.3, 0.1]` in those two tests (and add the
  `assert not allclose(headline, h1only_map)` companion). That one edit kills M19.

### tests-deep-region-staged-3 — Nothing asserts the georeferencing of any output raster, though "ship on the EXACT grid of the mosaic map" is Stage D's entire purpose
- **Severity:** medium
- **Liveness:** dead-closed
- **Confidence:** high (executed)
- **Where:** `tests/test_region_staged.py:113-116` (`_read`, which returns `ds.read(1)` and nothing
  else); code at `scripts/f_region_staged.py:240`, `:259-264`, `:281-283`; `src/fcompose.py:193-195`

`grep -n "transform\|crs\|bounds\|nodata\|dtype" tests/test_region_staged.py` hits **only lines 29-40**
— inside `_write_ref`, the fixture that *writes* the reference raster. No assertion anywhere opens an
**output** raster's `transform`, `crs`, `nodata` or `dtype`. Three mutants exploit that: writing every
output with a GDAL-order affine (**M06**), with an empty CRS (**M26**), and transposing the
frame→pixel map itself (**M01**) all pass 18/18. M01 also passes `tests/test_fcompose.py`, so it is a
repo-wide survivor.

- **Failure scenario:** the F map is written on a grid that does not match `reports/map_region/`.
  Every downstream comparison — gate 1's partition scoring, `f_map_compare.py`, notebook 24's overlay,
  and the abort's mosaic-vs-F level comparison — is then a comparison of misaligned rasters. This is
  the same defect class `DECISIONS.md:5111` records as **Blocker 1** of the 2026-07-29 Stage-D review
  ("the gate-5/6 cohort join was off by ~100 km"), and the same shape `src/fgates.py:211-231` and the
  `-features`/`labeling-deep-tests` areas record. It has now bitten this project four times.
- **Evidence:**
  ```
  M01 [SURVIVED here | SURVIVED tests/test_fcompose.py]  frame_rows_cols -> (cols, rows)
  M06 [SURVIVED]  _affine: Affine(*t) -> Affine.from_gdal(*t)   (a valid, wrong affine; no exception)
  M26 [SURVIVED]  write_geotiff(..., grid.crs_wkt) -> write_geotiff(..., "")

  pristine output transform = (159.999184, 0.0, -711136.371096, 0.0, -159.999184, 2133729.111655)
  pristine output CRS empty = False        <- correct, and never asserted
  ```
- **Self-refutation attempted:** (a) *Is M01 unobservable for a benign reason?* Two, and both are the
  brief's named shapes: the fixture grid is **square** (`SIDE = 24`) and every frame covers it
  entirely, so a transpose is a bijection onto the same pixel set; and the assertions index the
  output with `fc.frame_rows_cols(grid, TI, TJ)` — **the mutated function itself** — so writing and
  reading agree by construction. In production, frames are sub-rectangles and the transpose is a
  catastrophic misplacement. (b) *Does `test_fcompose.py` cover it?* It pins the axis convention
  directly at `test_ti_increases_northward_so_row_zero_is_the_HIGHEST_ti` (`g.rows_of_TI`, not
  `frame_rows_cols`), so the *convention* is safe; but its own `_add` helper also routes through
  `frame_rows_cols`, so the wrapper is unpinned there too — confirmed by running it. (c) *Would a
  wrong affine crash?* No: `Affine.from_gdal` on a rasterio-order 6-tuple yields a perfectly valid
  affine and the run completes.
- **Fix:** three lines in `_read` — return `ds.transform`/`ds.crs` too, and in
  `test_registration_report_records_the_subpixel_translation` assert the output transform and CRS
  equal the reference raster's. That kills M06 and M26. For M01, index one assertion with
  hand-computed `Ki - TI` / `TJ - Kj` instead of calling `frame_rows_cols`.

### tests-deep-region-staged-4 — All three fixes the 2026-07-29 Stage-D review made to this driver are unprotected, and the review commit added tests to two *other* files
- **Severity:** low
- **Liveness:** dead-closed
- **Confidence:** high (executed)
- **Where:** `scripts/f_region_staged.py:117-128` (frame-index auto-invalidation), `:359-366`
  (partial-Stage-B headline refusal), `:380-387` (done-check must include the headline product) —
  each carrying an explicit `(review 2026-07-29)` comment; no test reaches any of them

The three defects the review found are each documented *in the code* with a comment explaining the
bug. None is pinned:

1. **M18** — dropping the mtime staleness check from the `frame_index.csv` cache reintroduces
   "a cache built during a PARTIAL Stage B was baked in permanently and invisibly". No test ever
   changes an npz between two runs, so the branch is never exercised.
2. **M16** — disabling the `--allow-partial` refusal lets a headline (shippable) map be written from
   an incomplete Stage B. `FIG` is monkeypatched to `tmp_path/figures`, so `region_frame_list.csv`
   never exists, `n_planned = 0`, and the entire census branch is dead in tests.
3. **M17** — removing `need.append(out_dir / f"{tile}_prob_raw.tif")` restores the exact bug the
   comment describes ("a second run with `--headline` skipped the tile and silently never wrote the
   plain-named map"). `test_rerun_skips_completed_tiles_unless_overwritten` runs three times but
   always with the same verdict, so it never crosses the no-headline → headline transition.

`git log --oneline -- tests/test_region_staged.py` returns **one commit, `afe6fce`**. The review-fix
commit `458168f` modified `scripts/f_region_staged.py` and added tests to `tests/test_fgates.py` and
`tests/test_leveling.py` — but not to the driver's own test file.

- **Failure scenario:** the exact regressions the review already caught once, recurring silently.
  (2) is the most consequential: it is the only thing stopping a partial Stage B from being shipped
  as a headline map, and it is the guard whose absence the review called out.
- **Evidence:**
  ```
  M16 [SURVIVED]  if headline is not None and not args.allow_partial:  ->  if False:
  M17 [SURVIVED]  if headline is not None:  ->  if False:    (done-check drops the headline product)
  M18 [SURVIVED]  ... and prev.get("newest_mtime",-1) >= newest - 1e-6  -> dropped

  probe: FIG/region_frame_list.csv exists = False  -> n_planned = 0  -> census branch never entered
  git log --oneline -- tests/test_region_staged.py   ->  afe6fce   (one commit, never revised)
  git show --name-only 458168f | grep ^tests/        ->  test_fgates.py, test_leveling.py
  ```
- **Self-refutation attempted:** (a) *Are these paths worth testing given F is dead?* Marginal — hence
  **low**. I file it because the *shape* generalises: three fixes, each carefully commented with the
  bug it prevents, none converted into an assertion, in a commit that did add tests elsewhere. That is
  a process signal, not a Stage-D one. (b) *Is (2) reachable at all?* Yes — one line in `_run` (write
  a two-row `region_frame_list.csv` into `fig`) makes it fire.
- **Fix:** one test for each; the cheapest is (2) — write a 3-row `region_frame_list.csv` with only 2
  frames on disk and assert `SystemExit`, then assert `--allow-partial` proceeds.

### tests-deep-region-staged-5 — The fixture gives every frame identical, whole-tile coverage and an offsets table already in sorted key order — so the offset↔frame join, the frame LUT, and every partial-coverage path are unpinned
- **Severity:** low
- **Liveness:** dead-closed
- **Confidence:** high (executed)
- **Where:** `tests/test_region_staged.py:45-63` (`_write_stage_b`, which meshgrids the *whole*
  `TI_range()` × `TJ_range()` for every frame), `:66-78` (`_write_stagec`); code at
  `scripts/f_region_staged.py:191-198`, `:367`

Every synthetic frame covers exactly the same 576 pixels. Measured on the 3-frame fixture: the
`n_frames` histogram is `{3.0: 576}` with **0 uncovered pixels, 0 single-frame pixels and 0
out-of-tile tiles**. Every `incidence` is finite. And the offsets CSV rows are written in the same
order as `sorted(PRODUCT_ID)`, so a **positional** join is indistinguishable from a keyed one. Four
mutants live in that gap:

- **M27** — `offsets.loc[pid]` → `offsets.iloc[frame_lut[pid] % len(offsets)]`. This is the
  offset↔frame join key, the same class as `DECISIONS.md:5111`'s Blocker 1 and as
  `tests-deep-splits`' M07. Real Stage-C tables are written in solve order, not PID order.
- **M28** — the global frame LUT built in reverse. It keys the `primary_frame` H6 layer and the
  partition-composite ownership; `test_partition_layer_takes_each_pixels_owner_frame` even *comments*
  "global lut order == sorted pids". It cannot see the change because under full offsets the two
  frames' leveled probabilities are equal to 0.000e+00, so which frame owns a pixel is unobservable.
- **M09** — a frame with unknown incidence becoming the *best*-illuminated primary (`np.inf` → `-inf`).
  Repo-wide survivor: `test_fcompose.py::test_primary_frame_is_the_best_illuminated_contributor` also
  passes only finite incidences.
- **M20** — `overlap_dp` emitted as 0.0 rather than NaN for single-frame pixels, i.e. "no data"
  rendered as "perfect agreement" in the H6 QA layer. (Killed by `test_fcompose.py`, not by this file.)

- **Failure scenario:** M27 is the one that matters. A positional offsets join scrambles which frame
  gets which correction; the composite still looks plausible, coverage and `n_frames` are unchanged,
  and every η²/level number computed on it is meaningless. Nothing in this file could tell.
- **Evidence:**
  ```
  n_frames histogram over the tile: {3.0: 576}   uncovered pixels = 0 / 576
  incidence column = [40.0, 45.0, 50.0]          all finite = True
  csv row order    == sorted(PRODUCT_ID) : True  -> positional join == keyed join
  max |frame0 - frame1| after full offsets = 0.000e+00  -> partition ownership unobservable

  M27 [SURVIVED]  M28 [SURVIVED]  M09 [SURVIVED both suites]  M20 [SURVIVED here, killed by test_fcompose]
  ```
- **Self-refutation attempted:** (a) *Is this the (0,0)-origin shape the brief warned about?* **No —
  and that is worth recording.** See Refuted below: this fixture uses a real tile origin and the real
  pitch. The degenerate-fixture shape recurs, but in *coverage geometry*, not grid phase. (b) *Is
  partial coverage hard to synthesise?* No — dropping a slice of `TI`/`TJ` for one frame in
  `_write_stage_b` is one line and would exercise M20, M09's tie-break and the in-bounds screens at
  once. (c) *Is M27 realistic given `.loc` is idiomatic?* The 2026-07-29 review found a ~100 km
  mis-key in the sibling `fgates` join, so yes.
- **Fix:** shuffle `_write_stagec`'s rows (`df.sample(frac=1, random_state=0)`) — one line, kills M27 —
  and give one frame a partial footprint plus one a NaN incidence.

### tests-deep-region-staged-6 — Three assertions that read their expected value out of the code under test, and one whose second disjunct accepts anything
- **Severity:** low
- **Liveness:** dead-closed
- **Confidence:** high (executed)
- **Where:** `tests/test_region_staged.py:188`, `:220`, `:132`/`:144`/`:156`, `:207`

1. **`assert np.nanmax(src) == fc.OFFSET_SOURCE_CODE["interpolated"]`** (`:188`) and
   **`== fc.OFFSET_SOURCE_CODE["none"]`** (`:220`) look up the expected value in the very dict the
   production code uses. **M03** permutes the severity ranking so `none` (a frame with *no* offset at
   all) is no longer the worst code — and both assertions still pass, because both sides moved
   together. Repo-wide survivor: `test_fcompose.py:140-146` does the same lookup.
2. **The composite assertions index the output with `fc.frame_rows_cols`** (`:129`, `:144`, `:156`),
   the function they would need to be independent of — see finding 3 / M01.
3. **`assert after < 0.02 * before or after < 1e-6`** (`:207`) — the second disjunct is satisfied by
   *any* negative number, so the overlap-QA layer's sign is unpinned. **M04** (`p_min - p_max`) makes
   every reported `overlap_dp` negative and the test passes via that clause. Measured:
   pristine `before=0.223382, after=3.7e-09`; flipped `before=-0.223382, after=-3.7e-09` →
   `after < 0.02*before` is **False**, `after < 1e-6` is **True**.
4. Two more that cannot fail, not separately mutated:
   `test_frame_index_and_lut_are_cached:394`'s `assert list(lut.frame_idx) == sorted(lut.frame_idx)`
   is a tautology — `scripts/f_region_staged.py:368-369` calls `.sort_values("frame_idx")` immediately
   before writing; and `test_h6_layers_are_written_and_variant_independent` never compares a layer
   across variants, so its "variant_independent" claim is structurally true (the shared layers are
   written once, from `variants[0]`).
- **Evidence:** `M03 [SURVIVED both suites]`, `M04 [SURVIVED here, killed by test_fcompose]`,
  `M01 [SURVIVED both suites]`, plus the four numbers above.
- **Self-refutation attempted:** (a) *Is the `< 1e-6` disjunct wrong to have?* No — it is there because
  the leveled composite agrees to ~1e-9, so a pure ratio test would be numerically fragile. The fix is
  `abs(after)`, not deleting it. (b) *Is looking up a code by name bad practice?* Usually it is good
  practice; here the dict *is* the specification (the severity **ordering** is the H6 contract), so
  the test must hard-code `3` or assert the ordering explicitly.
- **Fix:** one line each — `assert fc.OFFSET_SOURCE_CODE["none"] == max(fc.OFFSET_SOURCE_CODE.values())`
  in the H6 test; `0 <= after < max(0.02 * before, 1e-6)` at `:207`.

---

## Refuted by my own check

- **"This suite has the zero-origin fixture defect found in three sibling areas."** It does **not**,
  and this is the first of the four to get it right. `_write_ref` (`:20-42`) uses
  `PITCH = 159.9991835298017` and `ORIGIN = (-711136.371096145, 2133729.111655494)` with the comment
  *"E-12_N32's real origin"*, producing `Kj/Ki = -4444/13335` — exactly the shift
  `tests/test_fcompose.py:208` regression-pins for that real tile. The grid arithmetic is therefore
  exercised at a genuine, non-degenerate phase, and **M02** (`TILE_M` 160 → 80) kills all 18 tests.
  The degenerate-fixture shape does recur here (finding 5) but in coverage geometry, not grid phase,
  so `R77-R80`'s exact recommendation does not apply.
- **"M21 — dropping `add_frame`'s in-bounds screen is a coverage gap."** Discarded as **equivalent
  through the only path this file exercises**. `compose_tile` filters `rows`/`cols` to in-bounds at
  `scripts/f_region_staged.py:186-189` *before* calling `add_frame`, and `add_frame` has no other
  caller in the driver — so the screen inside it is a redundant second guard and cannot be observed
  end-to-end. It is not dead code: `tests/test_fcompose.py::test_out_of_bounds_tiles_are_dropped_not_wrapped`
  calls `add_frame` directly with out-of-tile indices and kills the mutant. Correctly tested, just not
  from here. Not counted as a survivor.
- **"M22 — neutering `frame_bbox`'s empty-frame sentinel is a coverage gap."** Discarded as **benign**.
  `(0, -1, 0, -1)` is returned only when `TI.size == 0`, and `compose_tile` hits
  `if TI.size == 0: continue` (`:183`) before the bbox value can affect anything; the sole consequence
  of `(0, 0, 0, 0)` is that one empty npz gets opened and skipped. `test_fcompose.py:69` does pin the
  sentinel's *meaning* by passing the literal tuple to `bbox_intersects_tile`. Not counted.
- **"R11 (the tautological Stage-C trend guard) is pinned as intended here."** It is not, and it
  cannot be: this file **writes** `fbuild_trend_guard.csv` by hand (`:75-77`) and only consumes the
  verdict string, so it can pin the *routing* of a verdict (it does — M08) and nothing about how the
  verdict is computed. Repo-wide the nearest thing is `tests/test_region_stagec.py:110`,
  `assert guard["verdict"] in {"NO_TREND","FULL","RESIDUAL_ONLY","AMBIGUOUS"}` — a membership check
  satisfied by all four values, so R11 is not pinned there either. **Not re-filed.**
- **"R19 (`edge_cv_for_offsets` fallback mislabel) is pinned as intended here."** Out of reach:
  `edge_cv_for_offsets` lives in `src/fgates.py` and is exercised by `tests/test_fgates.py:125-170`;
  nothing in `tests/test_region_staged.py` or `scripts/f_region_staged.py` imports `fgates`.
  **Not re-filed.**
- **"The abort verdict could be wrong because `pfree` is untested."** Checked directly and **no**:
  `scripts/f_region_stagec.py:498` emits `offset_logit_pfree`, `VARIANTS["pfree"]` consumes exactly
  that name, and my probe confirms the composite is correct to 1e-5 when the column is supplied.
  Finding 1 is a coverage gap only; I am not impugning `41a6f26`.
- **"`test_overlap_dp_shrinks_when_the_offsets_are_applied` is vacuous."** No — it killed **M10**
  (offset sign flip). Only its second disjunct is weak (finding 6).
- **"The suite pins the logistic link, not just self-consistency."** Refuted by probe **P1**: replacing
  `sigmoid`/`logit` with a *consistent* non-logistic inverse pair (`σ(x/2)` and `2·logit`) leaves this
  file **and** `test_fcompose.py` + `test_leveling.py` green (70 passed). The fixture generates its
  probabilities with `lv.sigmoid`, so the suites pin that the two functions invert each other — which
  is the property that matters for composing with Stage C — but not which link they are. Recorded as a
  probe, not filed as a defect: any single-point change to either function *is* caught (M15).
- **"The two `EPS`/precision survivors are real gaps."** M14 (`EPS` 1e-4 → 1e-6) survives this file
  only because the fixture's probabilities span `[0.016, 0.946]` and never approach the clip; it is
  killed by `tests/test_leveling.py::test_offset_magnitude_report_uses_the_measured_yardstick`. The
  `EPS = 1e-4` contract is defended where it lives.

## Verified clean — what this suite genuinely DOES pin (each named by the mutant that killed it)

1. **The global 160 m lattice and its exact-integer relation to the 159.9991835 m mosaic grid.**
   **M02** (`TILE_M` 160 → 80) fails **all 18** tests — via `tile_index_map`'s own
   `"shift is not constant"` raise, i.e. the module's deliberate fail-loudly design working as
   documented (`src/fcompose.py:77-81`). This is the strongest thing in the file.
2. **The frame-prescreen axes.** **M12** (`frame_bbox` returning TJ in the TI slots) kills 16 of 18.
3. **The composite rule is the MEAN of leveled logits with one sigmoid at the end.** **M07**
   (sum instead of mean) kills `test_full_variant_cancels_the_planted_biases`,
   `test_h1only_variant_is_the_unleveled_composite`, `test_resid_variant_uses_the_residual_only_column`
   and `test_partition_layer_takes_each_pixels_owner_frame`.
4. **The offset sign convention (`logit(p) + o`).** **M10** kills `test_full_variant…`,
   `test_resid_variant…`, `test_partition_layer…` and `test_overlap_dp_shrinks_when_the_offsets_are_applied`
   — so the H4 claim "co-located disagreement falls after leveling" is a real, directional check.
   (It correctly leaves `test_h1only_variant…` passing: that variant applies no offset.)
5. **The direction of the link function.** **M15** (sigmoid sign flip) kills 5 tests.
6. **§0.1 guard 1 — an AMBIGUOUS verdict must not produce a plain-named map.** **M08**
   (`APPLY_TO_VARIANT["full_pending_ruling"] = "full"`) is killed by
   `test_ambiguous_verdict_writes_no_headline_map`, which also checks the sidecar's
   `headline_variant is None` and `needs_ruling is True`. The file's declared headline purpose is
   defended — for *existence* (contrast finding 2, which is about *identity*).
7. **H6 provenance is "worst contributor wins", and a frame with no Stage-C offset row is composited
   with `o = 0` and flagged rather than dropped.** **M13** (`np.maximum.at` → `np.minimum.at`) is
   killed by `test_frames_without_an_offset_row_are_flagged_not_dropped` *and*
   `test_h6_layers_are_written_and_variant_independent`; the first also pins `n_frames == 3` with only
   2 offset rows and the sidecar's `frames_without_offset` list.
8. **Tier-1 and Tier-2 calibrators are not interchangeable, and calibration is applied ONCE to the
   composited probability.** **M11** (abundance written from `calibrate_prob`) kills both calibration
   tests. `test_calibration_is_applied_once_to_the_composite_not_per_frame` uses a deliberately convex
   `x³` Tier-2 map and explicitly excludes the calibrate-then-mean alternative — it is doing real work.
9. **The PARTITION composite (gate 1's scoring layer) is built from *leveled* per-frame values.**
   **M23** (partition drops the offset) is killed by
   `test_partition_layer_takes_each_pixels_owner_frame`, the single most load-bearing test in the file.
10. **The `resid` variant really reads `offset_residual_only` and really differs from `full`.**
    Not separately mutated, but non-vacuous: `test_resid_variant_uses_the_residual_only_column` uses
    the asymmetric `[0.8,-0.3,0.1]` set (measured `|full − resid| = 1.25e-02` against its `atol=1e-3`)
    and carries the comment *"else the test proves nothing"*.

Also genuinely pinned, though no mutant of mine targeted them: `--no-partition` really suppresses the
partition layer while leaving `prob_raw`; a missing reference raster is skipped with the documented
`"no reference raster"` message rather than crashing; a completed tile is skipped on re-run and
reproduces byte-identically under `--overwrite`; `frame_index.csv` and `frame_lut.csv` are written and
contain every PID.

## Coverage note

- **This file has no `slow` tests**, so the `-m "not slow"` / full-file gap is **structurally zero**
  (18 passed either way). Both baselines re-confirmed at 577277f.
- **Read in full:** `tests/test_region_staged.py` (409, line by line before mutating),
  `scripts/f_region_staged.py` (442), `src/fcompose.py` (269), `tests/test_fcompose.py` (234),
  `tests/conftest.py`; `src/leveling.py:29-57` (the four names this file reaches — `EPS`, `TILE_M`,
  `sigmoid`, `logit`; nothing else in that 500+-line module is exercised, so the brief's "leveling
  retains general machinery" caveat mostly does not bite here).
  **Read in part:** `tests/test_leveling.py` (safety-grepped in full — no `cfg.`, no `output_dir`, no
  writes, no `slow`), `tests/test_region_stagec.py:100-112`, `scripts/f_region_stagec.py:490-505`,
  `DECISIONS.md:5085-5140`. Docs: `_prompts_tests_deep.md`, `verify/R36.md`, `tests-deep-splits.md`,
  `tests-deep-features.md`.
- **Method:** `src/`, `scripts/`, `tests/conftest.py`, `tests/test_region_staged.py`,
  `pyproject.toml`, `config.yaml` and `models/deployable/calibration.npz` copied to
  `<scratchpad>/rsmut/`, with pristine reference copies at `src_pristine`/`scripts_pristine`; pytest
  run with `cwd=rsmut` so `import src` / `import scripts` resolve to the mutated copy. For the
  cross-check I additionally copied `tests/test_fcompose.py`, `tests/test_leveling.py` and four
  **read-only** `reports/map_region/*_prob_raw.tif` (so `test_fcompose`'s real-map tests execute
  instead of skipping); combined baseline in the scratchpad = **88 passed**.
  **The repo's `src/`, `scripts/` and `tests/` were never modified and no producer was called.**
- **Executed:** 28 mutant runs against the target file (drivers `rsmut_driver.py`, `rsmut_driver2.py`;
  raw results `rsmut_results.json`, `rsmut_results2.json`), 10 cross-check runs against the sibling
  suites (`rsmut_driver3.py` → `rsmut_crosscheck.json`), and two genuineness probe scripts
  (`rsmut_verify.py`, `rsmut_pfree.py`) whose output is quoted verbatim in the findings.
- **Every survivor was verified to be a genuine behaviour change** by executing the pristine code on
  the tests' own fixtures and showing *why* the change is unobservable (quoted numbers in each
  finding). Two mutants (M21, M22) failed that check and were discarded rather than counted — matching
  `tests-deep-splits`' 2-of-16 discard rate.
- **Could NOT check:** (1) whether any *reported* F number moves under a surviving mutant — that needs
  the 906 real Stage-B npzs, which are not on this machine at build scale, and the F programme is
  closed; (2) `pytest-cov` is not installed in `geospatial`, so there is no line-coverage figure to
  accompany the mutation score; (3) `src/fcompose.py`'s other consumers — `scripts/f_region_gates.py`
  and `scripts/f_map_compare.py` both call `fc.frame_labels_on_grid`, which **this suite never
  executes** (`seam_labels` is monkeypatched in one test and swallows a `RasterioIOError` in the other
  17), and `fc.partition_composite` / `fc.windows_over_grid` are not called by the Stage-D driver at
  all — all three are covered by `tests/test_fcompose.py` and belong to a different sub-area;
  (4) I did not run the full 490-test suite.
