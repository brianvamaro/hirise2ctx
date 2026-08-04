# Deep-pass briefs — `tests-deep-*` (opened 2026-08-04)

Line-by-line + **mutation testing** of the four large test bodies that pass 1's `tests` area never read.
One agent each, each writing its own file. **An area is done iff its file exists.**

| sub-area | file | lines |
|---|---|---|
| `tests-deep-features` | `tests/test_features.py` | 533 |
| `tests-deep-within-image` | `tests/test_within_image_split.py` | 445 |
| `tests-deep-region-staged` | `tests/test_region_staged.py` | 409 |
| `tests-deep-splits` | `tests/test_splits.py` | 399 |

## Why, and what the fifth body already told us

§6 of the register names these as *"the highest-yield code-reading work left"*: pass 1's `tests` area
surveyed markers, skips and assertion *shapes* but never read the bodies, and all three known instances
of the "a test pins wrong science" class (**R19**, **R24**, **R11**) were found by *other* areas,
accidentally.

The fifth body, `test_labeling.py` (668 lines), was done on 2026-08-04
([labeling-deep-tests.md](labeling-deep-tests.md), findings **R77–R80**) and the result should
**calibrate your expectations**:

> **No assertion defended a known defect.** But the suite pins far less than it appears to: of 25
> defects seeded into a scratchpad copy of `src/`, **16 of 20 survived `pytest -m "not slow"`** —
> CLAUDE.md's documented dev loop — and **12 of 20 survived the full suite.**

So the likely yield is **"pins less than it appears to"**, not "pins wrong science". Do not manufacture
the latter. Two shapes from that pass generalise and are worth hunting explicitly:
- **A fixture whose configuration no production input has.** Every end-to-end labelling fixture pinned
  the mosaic grid phase to **zero**, which 0 of 47 production images has — so the alignment test could
  not detect a 2,608 km displacement. This is the *same* fixture defect `src/fgates.py:211-231` records
  as having caused the ~100 km gate mis-key, so it has bitten this project twice already.
- **A test that appears to exercise a defect but does not.** R24's aggregation test NaNs only a
  *specificity* fold, which is filtered out before `mean_std` ever runs.

**Mutation testing is what produced the evidence; reading alone would not have.** Use it.

## Method

1. **Read the body line by line first**, and write down what each test *claims* to pin.
2. **Then mutate.** Copy `src/` into your scratchpad, seed ~15–25 realistic single-point defects into the
   functions the suite covers (flip a comparison, drop a term, swap an axis, off-by-one an index, skip a
   filter, return the input unchanged, transpose, use the wrong column), and run the real test file
   against the mutated copy. **Report the survival rate for `-m "not slow"` and for the full file
   separately** — the gap between them is itself a finding, because the documented dev loop is the
   fast one.
   Make the copy importable without touching the repo: copy `src/` to the scratchpad, run pytest from
   there with the scratchpad ahead on `sys.path`/`PYTHONPATH`, or point `conftest`-level imports at it.
   **Never mutate `src/` in the repo, and never modify anything under `tests/`.**
3. For each surviving mutant, say which assertion *should* have caught it and why it did not.
4. **Also report what the suite genuinely DOES pin**, each item named by the mutant it killed. That is
   the output that lets a future session stop re-reading these files, and it is as valuable as a defect.

## Ground rules

- **READ-ONLY apart from your own area file and the scratchpad.** Follow §1 of
  [_prompts.md](_prompts.md) (shared context, the 10 invariants, the do-not-re-file list R01–R22, the
  §3 output template you must use). Do not re-file **R77–R80**.
- ⚠ **NEVER call a producer function, and NEVER run the slow real-data tests.** See the expanded warning
  in §1 of `_prompts.md`. On 2026-08-04 a reviewer ran `pytest tests/test_labeling.py` and silently
  overwrote four gitignored v1 artifacts, because `test_stage4_runs_on_ESP_069669_2220` passes
  `output_dir=cfg.output_dir` — the **live** `dataset/` tree. `git` cannot restore those paths.
  Producers write to config-derived live paths with no dry-run mode: `src/labeling.py:543`, `:591`,
  `src/detections.py:151`, `src/coregister.py:436`. **Before running any test in your file, grep it for
  `cfg.output_dir`, `cfg.cache_dir`, `config.` path attributes and the `slow` marker; if a test reaches
  a real tree, do not run it — read it, and file that as a finding.**
- Prefer `-m "not slow"`. If your file has no slow tests, say so explicitly in the coverage note.
- Environment: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u -m pytest ...`
  (conda is **not** on PATH). Any script importing torch must `import src.modeling` before numpy/pandas.
- Self-refute before filing; judge **severity** and **liveness** independently; report at most 6
  findings ranked most-severe first. An honest "this suite is sound, here is what it pins" is a valid
  and useful result.
- **Write your file as your FINAL action** — return values are discarded on a session kill.

## Sub-area notes

- **`tests-deep-features`** — covers `src/features.py`, `spatial_features.py`, `colour.py`,
  `ctx_source_illumination.py`. Cross-check against **R27/R28** and the `features-deep` area file: is
  any known feature defect *pinned as intended*? Pattern D applies — `features-deep` found the code
  sound but the derived caches two generations stale.
- **`tests-deep-within-image`** — covers the quadrant splitter. **R45 is confirmed and directly
  relevant**: the within-image arm is scored per quadrant against a whole-image LOIO arm, and
  `_compute_quadrant_definitions` medians over label rows, so its output drifts with the labels
  (`dataset_v2/splits/within_image_4fold.json` is a different vintage from the sweep that used it —
  3.5 % of tiles in a different quadrant). Does any test pin the quadrant definition, or its stability?
- **`tests-deep-region-staged`** — covers the F-build Stage A–D machinery. The F build is
  **dead-closed** (hard-aborted `41a6f26`), so cap severity accordingly — **except** where a defect
  could mean the **abort verdict itself** was wrong, which is high. Relevant: **R11** (tautological
  trend guard) and **R19** (`edge_cv_for_offsets` fallback mislabel) both live here and are already
  filed — check whether either is *pinned as intended*.
- **`tests-deep-splits`** — covers LOIO group-aware splitting, i.e. **invariant 6**, the one whose
  violation would invalidate every reported number. `labeling-deep-artifact` established the v2 LOIO
  splits are sound and structurally cannot drift (fold *i* is `sorted(obs_ids)[i]`, content-independent)
  — so the question here is whether the *tests* would catch a regression that broke that, e.g. a silent
  fallback to random tile splits. Also **R04** (stale packaged splits undetectable) and
  `other-scripts-1` (the split hash has already drifted).
