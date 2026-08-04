# Review area: tests

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-01
- **Verification:** direct pass by the orchestrating session (three subagent attempts died on
  connection errors after ~330k tokens each; this was done hands-on instead). Every claim below is
  reproduced from a command whose output is quoted.

Baseline: `pytest -m "not slow"` → **490 passed, 21 deselected, 6 warnings** in ~50 s.
`pytest -m slow --collect-only` → **21 collected, 490 deselected**. Note the fast run reports **no
skips**, so on this machine every data-dependent guard in the fast suite did execute.

## Findings

### tests-1 — No test anywhere covers the v2 splits, which are the basis of every reported number
- **Severity:** high
- **Liveness:** live-shipped (the frozen recipe, every LOIO number, and the shipped map all rest on `dataset_v2/packaged/loio_nfold`)
- **Confidence:** high (exhaustive grep of `tests/`)
- **Where:** `tests/test_modeling_group_leak.py:21`; absence across all of `tests/`

The group-leak integration suite — the only automated enforcement of CLAUDE.md **invariant 6**
("splits are group-aware leave-image-out, never random tiles") — is pinned to the **v1** dataset:

```
tests/test_modeling_group_leak.py:21
REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGED_LOIO_9 = REPO_ROOT / "dataset" / "packaged" / "loio_9fold"
```

But the cohort in use is v2. On disk:

```
dataset/packaged/      -> loio_3fold_balanced, loio_9fold, within_image_4fold        (v1, 9 images)
dataset_v2/packaged/   -> loio_nfold, loio_nfold_ctx_illum, loio_nfold_nbr_s5,
                          within_image_4fold                                          (v2, 38 images)
```

and a repo-wide grep of `tests/` for `dataset_v2` or `loio_nfold` returns **exactly two hits**, both in
`tests/test_fgates.py:241,244`, and both reading `dataset_v2/labels/*.parquet` — never the *packaged
splits*. So `loio_nfold` (the 38-fold LOIO scheme behind `pooled_pr_auc 0.7832`, the frozen recipe, the
deployable head and the regional map), `loio_nfold_ctx_illum`, `loio_nfold_nbr_s5` and v2's
`within_image_4fold` are verified by **no test at all** — not for group leakage, not for fold count,
not for train/test disjointness, not for the `patch_idx` exclusion.

- **Failure scenario:** any defect introduced into v2 split construction or packaging — a group key
  regression, a fold-assignment change, a repackage driver drift — passes the whole suite. This is not
  hypothetical: **R04** shows a failed Stage-5 rebuild leaves the previous cohort's packaged directory
  in place with nothing downstream detecting the staleness, and `other-scripts-1` shows the two
  repackage drivers' copy of the split hash has **already drifted** from `src/dataset.py`, so 7
  committed split JSONs — including `loio_nfold_ctx_illum` and `loio_nfold_nbr_s5` — carry a
  `split_hash` the canonical function cannot reproduce. Three independent findings, one uncovered
  surface.
- **Evidence:**
  ```
  $ grep -rn "dataset_v2\|loio_nfold" tests/*.py
  tests/test_fgates.py:241:    paths = sorted(glob.glob(str(repo_root / "dataset_v2" / "labels" / "*.parquet")))
  tests/test_fgates.py:244:        pytest.skip("dataset_v2/labels or cohort_obs_bounds.csv not on disk")
  ```
- **Self-refutation attempted:** (a) Checked whether the v1 scheme is a faithful proxy — it is not:
  9 images vs 38, a different split algorithm branch (`loio_9fold` vs `stratification: none` →
  `loio_nfold`), and v2 adds two feature-augmented schemes with no v1 counterpart. (b) Checked whether
  `tests/test_splits.py` / `test_within_image_split.py` cover v2 — both are pinned to `priority10`
  (v1) by name (`test_priority10_loio_9fold_matches_sweep`,
  `test_within_image_4fold_on_priority10_yields_32_folds`). (c) Checked whether the notebooks cover it
  — `notebooks/09_splits_qa.ipynb` is a QA notebook, not a test, and `test_modeling_group_leak.py:36`
  says it "mirrors" that notebook, i.e. the notebook is the v1 original. (d) Checked
  `DECISIONS.md`/`PLAN_Stage5.md` for a deliberate decision to leave v2 uncovered — nothing.
- **Fix:** parameterise `test_modeling_group_leak.py` over `(dataset_dir, scheme)` and add the four v2
  schemes, or at minimum add `dataset_v2` + `loio_nfold`. Keep the file `slow` if the parquet reads are
  expensive, but see **tests-2** — a metadata-only variant of the group-leak assertion (read
  `dataset_v2/splits/loio_nfold.json`, check `set(train_obs) ∩ set(test_obs) == ∅` per fold) needs no
  parquet at all and belongs in the fast suite.

### tests-2 — Both invariant guards that CLAUDE.md calls load-bearing are `slow`, so neither runs in the routine suite
- **Severity:** medium
- **Liveness:** live-shipped
- **Confidence:** high (reproduced from `--collect-only`)
- **Where:** `tests/test_modeling_group_leak.py:29,34,47,60,72,83`;
  `tests/test_sanity_residual_one_image.py:24`

Of the 21 tests deselected by the default `-m "not slow"` run, **6 are the entire group-leak file**
(invariant 6) and **1 is `test_stage1_centroid_residual_under_threshold`** (invariant 2, the O(200 m)
CRS residual that CLAUDE.md says "must fail loudly"). So the two invariants the operating manual
singles out as load-bearing are both outside the suite anyone actually runs.

Invariant 2 is doubly exposed: **R30** established that `qa.assert_centroid_consistent` has no
production caller either, so the O(200 m) residual is checked neither at ingest nor in the routine
tests — only by a slow test, on one v1 image, if someone remembers to run `-m slow`.

- **Failure scenario:** the documented development loop is `pytest -m "not slow"` (CLAUDE.md: "Tests:
  `pytest -m "not slow"` (fast) / full suite incl. slow integration"). A change that reintroduces group
  leakage or breaks per-image CRS handling passes 490/490 green.
- **Evidence:**
  ```
  $ pytest -m slow --collect-only -q
  tests/test_modeling_group_leak.py::test_loio_9fold_has_expected_number_of_folds
  tests/test_modeling_group_leak.py::test_no_obs_id_appears_in_both_train_and_test_of_any_fold
  tests/test_modeling_group_leak.py::test_each_obs_id_appears_as_test_in_exactly_one_fold
  tests/test_modeling_group_leak.py::test_scale_subset_preserves_train_test_disjointness
  tests/test_modeling_group_leak.py::test_x_train_excludes_patch_idx_columns
  tests/test_modeling_group_leak.py::test_per_fold_test_count_matches_metadata
  tests/test_sanity_residual_one_image.py::test_stage1_centroid_residual_under_threshold
  ...
  21/511 tests collected (490 deselected)
  ```
  The file's own docstring explains the marker: *"Marked `slow` because they read 9 real X/y parquets
  totalling ~500 MB"* — a justified reason for the parquet-reading tests, but the *split-metadata*
  assertions (fold count, obs-id disjointness, per-fold counts) need only the split JSON.
- **Self-refutation attempted:** the `slow` marker is deliberate and its rationale is written down, so
  this is not an oversight in itself. What survives is that the **cheap half** of these assertions was
  never split out, leaving the invariant unguarded in the loop that is actually run.
- **Fix:** split each group-leak test into a metadata-only assertion (fast, reads
  `dataset*/splits/{scheme}.json`) and a parquet-backed one (stays `slow`). Do the same for the CRS
  residual: a fast synthetic case with two *different* local radii (see **tests-4**) alongside the slow
  real-image check.

### tests-3 — Every slow integration test degrades to a silent pass when its data is absent
- **Severity:** medium
- **Liveness:** live-shipped (this is the CI/fresh-clone behaviour)
- **Confidence:** high
- **Where:** `tests/test_modeling_group_leak.py:23-26` (module-level `skipif`);
  `tests/test_splits.py:379,395`; `tests/test_within_image_split.py:437`;
  `tests/test_labeling.py:579,631,635`; `tests/test_coregister.py:225,270`;
  `tests/test_stage2_one_image.py:34,107`; `tests/test_features.py:489,517`;
  `tests/test_fm_embeddings.py:221`; `tests/test_fcompose.py:218,230`;
  `tests/test_fgates.py:244`; `tests/test_ctx_edr.py:37`

Every integration test is guarded by "skip if the cache/artifact is not on disk". Combined with
**tests-2**, the failure mode is: run the fast suite → the invariant tests are deselected; run the slow
suite on a machine without the ~500 MB of caches → they are skipped. Either way the result is green and
the invariant was never evaluated. `pytest` reports skips, but a skip is not a failure and nothing in
the repo asserts a minimum number of executed integration tests.

Note the fast suite currently reports **no** skips on this machine, so `test_fgates`, `test_fcompose`
and `test_ctx_edr`'s data-dependent guards do run here — but they would silently vanish on a fresh
clone, and `test_fgates.py:244` guards precisely the cohort-join logic whose ~100 km mis-key was the
blocker fixed in `458168f`.

- **Failure scenario:** a fresh clone or a CI runner without `dataset*/packaged/` and `cache_v2/`
  reports "all tests pass" while having executed zero group-leak, zero CRS-residual, zero Stage-2/3/4
  integration and zero cohort-join coverage.
- **Fix:** add a `--strict-integration` opt-in (or an env var) under which these `skip`s become
  failures, and run the slow suite that way whenever Stage 4/5 packaging changes — the condition the
  group-leak docstring already names ("rerun whenever Stage 5 packaging changes").

### tests-4 — The per-image-local-radius invariant is exercised with only one radius, and the test that would catch a shared-datum bug asserts `> 0.0`
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `tests/test_detections_reprojection.py:14-20`, `:80`

CLAUDE.md **invariant 1** is that the local Mars radius *differs image-to-image* and must be read per
image. The reprojection tests use exactly one local radius — `3393833.2607584` (ESP_047976_2020's) — so
a regression that reads one image's CRS and reuses it for the whole cohort would pass every test. The
only other local radius in the suite, `3387887.658234`, is in
`tests/test_sp1_correction.py:19`, which tests the SP1 regex, not reprojection. Counted across
`tests/*.py`: `3393833.2607584` ×4, `3387887.658234` ×1, `3396190` ×4.

Separately, the assertion meant to prove the radius matters is:

```
tests/test_detections_reprojection.py:78-80
    # Numerically must differ (different cm, different radius)
    assert abs(dx - x) > 1.0
    assert abs(dy - y) > 0.0  # different radii alter the y as well
```

`> 0.0` passes on any floating-point difference whatsoever, so it does not establish what its comment
claims. The `dx` assertion is dominated by the 180° central-meridian change, not the radius.

- **Failure scenario:** `src/detections.py` starts caching the first image's CRS (or a future refactor
  hoists `CRS.from_user_input` out of a loop). Every existing test still passes; the resulting
  mislocation is a few hundred metres — inside the O(200 m) band the project treats as normal, and
  **R30**/**tests-2** mean the residual check is not running either.
- **Self-refutation attempted:** (a) Checked whether Stage 3 would catch it — partially, but
  `geo-crs-2` shows the phase-correlation solve is bounded to ±640 m by construction, so a
  radius-induced error of that scale is absorbed rather than flagged. (b) Checked
  `tests/test_hirise_imagery_sp1_override.py` and `test_sp1_correction.py` for multi-radius coverage —
  both use a single WKT each.
- **Fix:** add a test that reprojects the *same* lon/lat through two different local-radius source
  CRSes and asserts the projected metres differ by the amount the radius ratio predicts
  (`Δy/y ≈ ΔR/R`), and tighten `:80` from `> 0.0` to that predicted magnitude.

## Refuted by my own check

- **"The fast suite is hiding skips."** It is not: `490 passed, 21 deselected, 6 warnings` — no
  `skipped` count, so every data-dependent guard in the fast suite executed on this machine. The
  fresh-clone exposure in tests-3 is real but is a portability claim, not a claim about the current run.
- **Wide `pytest.approx` tolerances letting any implementation pass.** Checked every `approx` with a
  non-trivial tolerance. They are physically motivated and tight where it matters: `abs=1e-9` on the
  isotonic/beta AUC-preservation checks, `abs=1e-6` on the reprojection lat/lon and on
  `spearmanr == 1.0`, `rel=1e-5` on the illumination angles, `abs=0.15` px on the synthetic
  co-registration shift (sub-pixel, on a 5 m grid), `abs=1.0` m on the ±30–50 m shift-direction test.
  The loosest — `rel=0.1, abs=0.01` on quantile matching and `abs=0.03` on the near-zero mass — are on
  deliberately stochastic fixtures. No finding.
- **Trivially-true assertions.** 13 hits for `assert X is not None` / `len(...) > 0` shapes. All are
  *preconditions* inside tests whose substantive assertions follow (e.g.
  `test_modeling_gbm.py:63 assert p is not None` precedes a probability-range check;
  `test_labeling.py:553 assert len(df) > 0` precedes the nested-ladder sum check). None is a test whose
  *only* assertion is trivial. No finding.
- **`conftest.py` fixture infrastructure making tests trivially pass.** It is 23 lines: the
  `src.modeling` OpenMP bootstrap plus two session fixtures (`repo_root`, `cfg`). No autouse fixture,
  no mocking, no monkeypatching, nothing that could neuter an assertion.
- **Over-mocking of the invariants.** The suite is notably *un*-mocked — the CRS, SP1, windowed-read and
  decimation tests use real WKTs, real `.LBL` text and real rasterio objects rather than fakes. The
  weakness in this area is coverage breadth (tests-4), not mocking.

## Verified clean

- `tests/test_detections_reprojection.py` is well constructed for what it covers: it forward-projects
  through `pyproj` independently rather than round-tripping the code under test, checks lat/lon
  preservation to 1e-6°, explicitly guards against a silent identity-map, and asserts the helper does
  not mutate the caller's CRS.
- The `slow` marker is registered in `pyproject.toml` (`markers = ["slow: integration tests that touch
  real detection files or the network"]`), so `-m` selection is not silently matching nothing.
- `testpaths = ["tests"]` is set, so a bare `pytest` cannot accidentally collect from `scripts/probes/`.
- Test count reconciles exactly: 490 fast + 21 slow = 511 collected.
- The group-leak assertions themselves are correct where they run — `train_obs & test_obs` per fold,
  each obs as test exactly once, disjointness preserved under scale subsetting, `patch_idx` columns
  excluded from `X_train`.

## Coverage note

Read in full: `tests/conftest.py`, `tests/test_modeling_group_leak.py`,
`tests/test_detections_reprojection.py`, and the marker/skip surface of all 44 test files. Ran
(read-only): `pytest -m "not slow" -q` (the 490/21 baseline), `pytest -m slow --collect-only -q` (the
21-test list), and greps for `pytest.mark.slow` / `pytest.skip` / `skipif` / `xfail`, weak-assertion
shapes, `approx` tolerances, Mars radii, and `dataset_v2`/`loio_nfold` references.

**Not** checked, and worth a follow-up pass: the *semantic* correctness of the larger test bodies —
`test_labeling.py` (668 lines), `test_features.py` (533), `test_within_image_split.py` (445),
`test_region_staged.py` (409), `test_splits.py` (399) were surveyed for markers and assertion shapes
but not read line-by-line for assertions that pin wrong science. The three known instances of that
class were found by *other* areas, not by this one, and are already in the register: **R19**
(`test_fgates.py:162-174` is named for a fallback branch it never exercises), **R24**
(`test_modeling_evaluate.py` has no test pinning `spearman_n` against `n_real_folds`, which is why the
S=128 defect was invisible), **R11** (`test_leveling.py:470` pins a tautology as intended behaviour).
That pattern — a test whose name or docstring claims more than its body checks — is the highest-yield
thing left in this area.

I did not run the slow suite (it needs ~500 MB of caches and touches the network), so the 21 slow tests'
current pass/fail state is unverified; only their collection is.
