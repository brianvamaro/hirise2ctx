# Review area: tests-deep-within-image

- **Reviewed at commit:** bd19da8
- **Date:** 2026-08-04
- **Verification:** self-refuted (single-agent pass; not independently verified)
- **STATUS: WORK IN PROGRESS** — all 15 mutants run; equivalence-verification of survivors in
  progress. If this banner is still here the session was cut short; the mutation table below is final.

Target: `tests/test_within_image_split.py` (445 lines, 16 tests, 1 `slow`), covering the quadrant
splitter in `src/dataset.py`.

## Baselines (scratchpad `mutroot/`, pristine `src/`)

- `-m "not slow"` → **15 passed, 1 deselected** in 3.5 s
- full file → **16 passed** in 3.5 s (the slow test *executes*, does not skip: I copied
  `dataset/labels/` + `dataset/splits/` read-only into the scratchpad)

## Mutation results — 15 seeded defects

| survival | `-m "not slow"` | full file |
|---|---|---|
| 15 seeded | **10 survived (67 %)** | **10 survived (67 %)** |

**The fast/full gap is exactly zero.** Structural, not luck: the only `slow` test
(`:430-445`) never calls the splitter — it reads the *stored*
`dataset/splits/within_image_4fold.json` and asserts on that dict, plus `discover_obs_ids`.

| id | mutation | verdict |
|---|---|---|
| M01 | `_compute_quadrant_definitions`: median → **mean** | **SURVIVED** |
| M03 | floor-snap → **ceil-snap** (cut moves 24 → 32) | **SURVIVED** |
| M04 | `ti_mid` / `tj_mid` **transposed** in the returned dict | **SURVIVED** |
| M05 | median over **all scales pooled**, not the finest | **SURVIVED** |
| M07 | quadrant code weights swapped (`2*ti+tj` → `ti+2*tj`) | **SURVIVED** |
| M08 | buffer band `<` → `<=` (drops 3 rows/cols, not 1) | **SURVIVED** |
| M11 | `n_train_tiles_per_scale` ignores the buffer keep-mask | **SURVIVED** |
| M12 | `finest_px` `min` → `max` in the fold summary | **SURVIVED** |
| M13 | quadrant defs computed **once** and reused for every image | **SURVIVED** |
| M19 | packaging hardcodes `buffer_tiles = 0` | **SURVIVED** |
| M02 | floor-snap to the coarsest factor removed | killed (`test_quadrant_cuts_are_strictly_coherent_across_scales`) |
| M06 | predicate `>=` → `>` | killed (same test) |
| M09 | buffer `OR` → `AND` (only the corner tile dropped) | killed (`test_within_image_buffer_drops_boundary_tiles`) |
| M16 | packaging train rows include the **test quadrant** (self-leak) | killed (`test_within_image_groups_have_3_unique_train_codes_per_fold`) |
| M17 | `groups_*.npy` store the obs code, not the quadrant index | killed (same test) |

Findings, equivalence checks and the R45 quadrant-stability probe to follow.
