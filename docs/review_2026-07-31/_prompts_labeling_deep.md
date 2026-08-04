# Deep-pass briefs — `labeling-deep-*` (opened 2026-08-04)

Second pass over the label basis, in the mould of `geo-crs-deep` and `features-deep`. Four sub-areas,
one agent each, each writing its own file. **An area is done iff its file exists.**

## Why this area, and why a second pass

Pass 1 (`labeling.md`) was thorough **at the computation level** and found four real defects. But the
two second-pass deep reviews already run both reached the same conclusion independently, which is why
it is **Pattern D** in the register:

> *"Pass 1 audited what the module **computes** and never audited what the module **publishes**."*
> *"Under-reviewed at the artifact and semantics level, not the code level."*

**Review three things, not one:** (i) the computation, (ii) the statistic it publishes and whether that
statistic can move, and (iii) **the artifact on disk versus what the current code would produce.**
Items (ii) and (iii) need almost no code reading and were skipped by every first-pass reviewer.

The label basis is also where the register's damage concentrates. Both **R23** (confirmed; the
register's #1 fix priority — two cohort images' labels are a score-rank truncation of the detection
set, 11.6 % of tiles) and **R03** (confirmed 2026-08-04, and the verifier found the mechanism *larger*
than filed: the 0.25 and 0.50 m/px cohorts' detection floors are **fully disjoint**, and the global
`min_size_m: 1.4105` removes **0 of 3,109,321** coarse-cohort polygons, so it is structurally
incapable of equalising them) are label-basis defects. **R56** exists only because it blocks R23's fix.
Nobody has done a deep pass here.

## Shared brief

Read **§1 of [_prompts.md](_prompts.md)** for the pipeline description, the 10 load-bearing invariants,
the already-found list (do **not** re-file R01–R22) and the verified-clean list. Use the **§3 output
template** in that same file. Additionally do not re-file the four pass-1 findings — read
[labeling.md](labeling.md) first — unless your own reading **materially corrects or extends** one, in
which case say so explicitly and give the correction.

**What pass 1 already measured** (so you do not redo it): a `.shx`-vs-`.shp` integrity scan of all 46
detection folders, the null-shape-record census, `.dbf` header parse + seeked `score` reads at the
truncation boundaries, a per-image window/coverage audit, score→count/area retention curves, and
per-image S=32 tile counts / mean `fa` / rich share / zero share over all 38 images (161,005 tiles,
pooled rich share 0.3598).

**What pass 1 explicitly could NOT check** — this is the seed for sub-areas A and B, and both are now
reachable from *local cached* files:
- whether BoulderNet's inference footprint equals the HiRISE image footprint (*"the coverage mask is
  built from image validity, not detector coverage, so an interior inference gap would still be
  labelled zero"*);
- the exact tile count affected by `labeling-2` (the ~1-tile strip inside every swath edge labelled
  zero by construction) — pass 1's estimate is **analytic only**; the cached `*_hirise_mask.tif`
  rasters would settle it.

## Ground rules

- **READ-ONLY** apart from your own area file. No notebooks, sweeps, training, map builds, ISIS, GPU.
- ⚠ **NEVER call a producer function, and never run the two slow real-data tests.** See the expanded
  warning in §1 of [_prompts.md](_prompts.md). An earlier run of *this very brief* overwrote four
  gitignored v1 artifacts by running `pytest tests/test_labeling.py`, because
  `test_stage4_runs_on_ESP_069669_2220` passes `output_dir=cfg.output_dir` — the live `dataset/` tree.
  It was restored on 2026-08-04 from `dataset/packaged/loio_9fold/y_test_fold6.parquet`; the sidecar
  carries a `restored_from` block recording it. **Do not reproduce that.**
- **No network.** Reading **local cached** rasters (`cache_v2/**`, `*_hirise_mask.tif`), vector caches
  (`.gpkg`, `.shp`/`.dbf`/`.shx`), `dataset*/`, `reports/`, `models/**/metrics.json` and PDS `.LBL`
  caches is expected and encouraged. If you read a HiRISE RED product, **decimated only** (invariant 4).
- Re-derive every number you assert with a small read-only pandas/numpy/rasterio snippet, and quote the
  snippet. Scratch scripts go in the session scratchpad, not the repo.
- Environment: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u <script>`
  (conda is **not** on PATH). Any script importing torch must `import src.modeling` before numpy/pandas.
- **Self-refute before filing.** For each candidate ask: is it factually wrong about the code, already
  guarded by a caller/config default, deliberate and documented (grep `DECISIONS.md` and the
  `PLAN_*.md` by term *before* concluding this), or pinned by a passing test? Say what you tried.
- Judge **liveness** and **blast radius** independently of how alarming the mechanism looks. On the 15
  findings verified so far, correcting the blast radius was the single most useful output, and two
  findings' headline consequences turned out never to have happened. Give a **severity**
  (blocker/high/medium/low) and a **liveness** (live-shipped / live-active-plan / dead-closed) per
  finding, and justify both.
- **Write your file as your FINAL action.** Return values are discarded on a session kill; the file is
  the deliverable.

---

## `labeling-deep-footprint` → `labeling-deep-footprint.md`

**The question pass 1 could not answer, and the only one here that could be blocker-class:
are some zero labels false zeros?**

The coverage mask is built from *image validity*, not *detector coverage*. If BoulderNet's inference
footprint is smaller than the HiRISE image footprint — or has interior gaps (tiles it skipped, a
stride/padding artifact, a failed chip, a margin it never scored) — then tiles inside those gaps are
labelled `fa = 0` and enter training and evaluation as genuine rock-free ground truth.

- Establish the **detection footprint empirically** per image from the cached reprojected detections
  (`cache_v2/reprojected_detections/*.gpkg`) — extent, convex hull, and more importantly the *interior*
  structure: rasterise detection presence onto the S=32 tile grid and look for holes that are
  implausible given neighbouring density. Compare against the image/coverage mask the labeller used.
- A real detector gap should look **geometric** (rectangular, striped, grid-aligned, tile-boundary
  aligned) rather than geological. Test that: are zero-runs axis-aligned in *detector* space? Do they
  align with a chip/stride size? Do they align with HiRISE CCD boundaries?
- Guard against the obvious refutation: **genuinely rock-free terrain is common** — the target is
  heavily zero-inflated by design (invariant 5), and the pooled rich share is only 0.3598. A zero
  region is not evidence of a gap. Quantify how you distinguish them and give your false-positive
  reasoning. **Default to REFUTED if you cannot positively demonstrate a non-geological gap.**
- If you do find gaps, quantify: how many tiles, in how many images, what share of the zero class, and
  what it would do to the shipped metrics (the frozen recipe is `fa_gt_1e-2` @ S=32).
- Also check the converse: detections falling **outside** the coverage mask (dropped silently?), and
  whether `drop_null_geometries` or any filter removes detections in a spatially structured way.

## `labeling-deep-artifact` → `labeling-deep-artifact.md`

**Pattern D item (iii): is the artifact on disk the one this code would produce today?**

`features-deep` found the `features/*.json` sidecars clean but *"stopped one directory short of the
derived caches, which are two generations stale."* Do the label-side equivalent.

- Inventory every committed/cached label artifact — `dataset_v2/labels/*.parquet`, `dataset/`,
  `cache_v2/**` derived caches, sidecars, split JSONs — and for each ask: what code version produced
  it, and would today's code produce the same bytes? Check recorded provenance (config hashes, git
  revs, thresholds, timestamps) against current `config.yaml` / `config_v2.yaml` and current code.
- Known drift signals to chase: the **2026-06-10 coregistration y-sign fix** (R44's verifier found
  `docs/methods.md` §5's `dy` column predates it and the cache has all 39 `dy` positive); the
  `min_size_m` / confidence-floor values; any threshold that moved after the artifact was written.
- Settle **`labeling-2` exactly** — read the cached `*_hirise_mask.tif` rasters and count the tiles in
  the swath-edge strip labelled zero by construction. Pass 1's number is analytic; replace it with a
  measured one, per image and in total, and say what share of the zero class it is.
- Check the split artifacts for the same staleness class: R45's verifier found
  `dataset_v2/splits/within_image_4fold.json` is a **different vintage** from the sweep that used it
  (3.5 % of tiles in a different quadrant). Is that true of the LOIO splits too? Compare split files
  against what the current splitter emits from the current labels.
- Cross-check `dataset/DATA_DICTIONARY.md` against the actual parquet columns and dtypes.

## `labeling-deep-semantics` → `labeling-deep-semantics.md`

**Pattern D item (ii): what does the labeller *publish*, and can that statistic move?**

- Take each published target column (`fractional_area` / `fa`, `boulder_count`, and any derived
  rich/poor flag) and ask what it is *bounded by*. `geo-crs-deep`'s finding was that
  `peak_correlation` is bounded below by the very threshold it is screened against — look for that
  shape here. Is any label statistic floored, capped, or pinned by the filter that produced it?
- **Cross-cohort comparability is the live thread.** R03 established the two pixel-scale cohorts have
  disjoint detection floors and that ~67 % (range 40–89 %) of a fine image's labelled boulder area lies
  below the coarse cohort's floor. Follow that through to the *published* target: what does `fa` mean
  when the size floor differs 3–4× between images, and what does that do to a cross-image metric, a
  LOIO fold, and the deployed abundance layer's physical interpretation? R03's verifier recommends
  emitting `map_scale_mpp` + the measured floor into the sidecars and documenting the size floor under
  `fractional_area` — assess whether that is sufficient and what `PLAN_RegionalMap`'s cross-place
  validation legs actually need.
- Check the target definition against every place it is *described*: `dataset/DATA_DICTIONARY.md`,
  `docs/methods.md`, `docs/build_spec.md`, `CLAUDE.md`. Flag any doc that states a definition the code
  does not implement. (R44 covers `methods.md`'s cohort scope — do not re-file that; this is about the
  *target semantics*.)
- Confidence floors and score thresholds: R23 is confirmed, so do not re-file it, but do check whether
  any *other* image has a non-uniform confidence basis, and whether the floor is recorded anywhere a
  consumer could read it.

## `labeling-deep-tests` → `labeling-deep-tests.md`

**Do the labelling tests pin wrong science?** `tests/test_labeling.py` is 668 lines and the largest of
the five test bodies §6 of the register names as *"the highest-yield code-reading work left"*. Pass 1's
`tests` area surveyed markers, skips and assertion *shapes* — it never read these line-by-line. All
three known instances of this defect class (**R19**, **R24**, **R11**) were found by *other* areas,
accidentally, which is what suggests more remain.

> ⚠ **Do NOT execute `test_stage4_runs_on_ESP_069669_2220` or `test_empty_shapefile.py`.** They write
> into the live `dataset/` and `cache/` trees (that is finding `labeling-deep-tests-1`, already filed —
> do not re-file it). Read them; do not run them. If you run pytest at all, use `-m "not slow"`.
> **Mutation-testing against a scratchpad copy of `src/` is the right technique here** and is safe.

- Read `tests/test_labeling.py` and `tests/test_empty_shapefile.py` **line by line**. For each
  assertion ask: does this pin *correct* behaviour, or does it pin the current behaviour whatever it is?
  A test asserting a defect is intended is worse than no test — it actively defends the defect.
- Specific shapes to hunt: assertions on a value the test itself computed with the same code path
  (tautological); tolerances wide enough to pass under the bug; fixtures constructed so the failure mode
  cannot occur (e.g. no null geometries, one cohort only, no swath edge, a single pixel scale);
  `pytest.approx` with a default tolerance on a quantity that should be exact; tests that assert a
  function *runs* rather than that it is *right*; and skipped/xfailed tests hiding a known break.
- Cross-check against the confirmed findings: is `labeling-1`/R23's truncation, `labeling-2`'s edge
  strip, or R03's pixel-scale floor **pinned as intended** anywhere? R24's verifier found the analogous
  aggregation test NaNs only a *specificity* fold, which is filtered before `mean_std` — i.e. the test
  does not actually exercise the defect. Look for that pattern.
- If the tests are sound, **say so plainly and list what they genuinely pin** — that is a useful result
  and it is what lets the next session stop re-reading them.
