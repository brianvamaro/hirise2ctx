# Review area: tests-deep-region-staged

- **Reviewed at commit:** bd19da8
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified)
- **Status: WORK IN PROGRESS** — this file is written early and overwritten after each mutant batch,
  per the brief's "a partial file on disk beats a perfect answer you never got to write".

Target: `tests/test_region_staged.py` (409 lines), covering the F-build **Stage D** composite driver
`scripts/f_region_staged.py` (442) and its core `src/fcompose.py` (269), plus `src/leveling.py`'s
`sigmoid`/`logit`/`TILE_M`.

## Baseline (scratchpad copy, established)

```
pytest tests/test_region_staged.py -q -m "not slow"  ->  18 passed in 4.15 s
pytest tests/test_region_staged.py -q                ->  18 passed in 3.90 s
--collect-only                                       ->  18 tests collected
```

**This file contains zero `slow`-marked tests, so the fast/full gap is structurally zero** — the two
sibling areas found a gap of zero for a different reason (their slow tests were vacuous or
unrunnable); here there is nothing to differ.

**Safety pre-check (done before running anything):** no `cfg.output_dir`, `cfg.cache_dir`,
`config.` path attribute or `@pytest.mark.slow` appears in the file. Every path is under `tmp_path`,
and the one module global that points at a live tree (`sd.FIG = reports/figures`) is monkeypatched to
`tmp_path/figures` by the `staged` fixture (`tests/test_region_staged.py:91-98`). Two live-tree
*reads* remain and are harmless: `seam_labels` → `src.striping.load_frames("T00_N00")` fails on the
missing `reports/map_region/T00_N00_abundance.tif` and is swallowed by the intended
`except Exception` (`scripts/f_region_staged.py:159`); and
`test_calibration_is_applied_once_to_the_composite_not_per_frame` lets `--calibration-mosaic` default
to `models/deployable/calibration.npz` (read-only). **No producer is reachable from this file.**

## Method

`src/`, `scripts/`, `tests/conftest.py`, `tests/test_region_staged.py`, `pyproject.toml`,
`config.yaml` and `models/deployable/calibration.npz` copied to
`<scratchpad>/rsmut/`; pytest run with `cwd=rsmut` so `import src` / `import scripts` resolve to the
mutated copy. **The repo's `src/`, `scripts/` and `tests/` were never modified.**

## Mutants — results so far

(pending; table filled in as the batches complete)

## Coverage note (in progress)

Read in full: `tests/test_region_staged.py`, `scripts/f_region_staged.py`, `src/fcompose.py`,
`docs/review_2026-07-31/verify/R36.md`, the two sibling deep-test area files.
