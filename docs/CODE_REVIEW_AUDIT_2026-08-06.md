# Code-review audit and fixing-stage handoff — 2026-08-06

## Purpose and authority

This is the current-state correction layer for the 2026-07-31 full-codebase review. Read it before
using [CODE_REVIEW_2026-07-31.md](CODE_REVIEW_2026-07-31.md) as a fixing queue or following
[PENDING_REBUILD.md](PENDING_REBUILD.md). The original review remains the evidence register; this
document records status drift, rejected fix alternatives, newly identified safety gaps, the complete
rebuild dependency chain, and Brian's 2026-08-06 product decisions.

The audit was performed read-only at commit `b987f83` on branch
`fm-deployable-head-and-map-pilot`. The worktree was clean and 13 commits ahead of the locally known
tracking ref before this document was written. No tests, producers, imports with pipeline side effects,
or artifact regeneration were run. The audit covered the review, its verifier files, the 2026-08-05
project-state and mutation-hazard memory notes, the pending-rebuild note, current source and tests,
and relevant git history.

The review baseline was `da884c7`. Eight commits follow that baseline at the audited HEAD. Of those,
only the R74 change in `src/ctx_retrieve.py` changed production pipeline behavior; the remaining
executable changes were tests and fixtures. Therefore, findings should be presumed open unless a
specific current-state correction below says otherwise.

## Decisions recorded on 2026-08-06

1. **The current deliverables are a corrected-grid baseline v2 regional mosaic and a matched
   A1-normalized version over the same planned 26-tile footprint.** Each arm should carry the regional
   rich-probability and calibrated-abundance layers expected by the map pipeline. A1 is an active
   planned product, not a dead experiment and not yet a shipped artifact. Generate the baseline and
   A1 products as a comparable pair and fail closed rather than shrinking either footprint.
2. **Retain and document the current resolution-dependent physical-size floors for the primary
   product.** Do not silently relabel this as one size-independent abundance target. A parallel
   common-floor product remains an explicit option and may eventually be produced, but it is a
   different scientific target definition rather than a correction or rescaling.
3. **`dataset/` v1 is superseded and will not be rebuilt.** Exclude it from the rebuild DAG and do not
   route current-product readers to it. Numbers derived from it describe that historical tree, not
   outputs reproducible under current code.
   **Amended later on 2026-08-06: it is also EXPENDABLE — not backed up, and losing it is accepted.**
   It is non-reproducible, so the cost is that every v1 measurement (R81's label offset, R92/R97's
   step-8 agreement, the 2026-08-04 incident deltas) becomes unre-verifiable. Those conclusions survive
   in git-tracked `DECISIONS.md` and the review register; only the ability to re-derive them dies.

These choices imply two product axes that must not be conflated:

| Preprocessing arm (both on the corrected shared grid) | Primary label convention | Optional alternative label convention |
|---|---|---|
| Baseline/raw DN | mixed resolution-dependent floor | common floor, only if separately approved and defined |
| A1-normalized | mixed resolution-dependent floor | common floor, only if separately approved and defined |

Until the common-floor option receives a numeric diameter/area rule, only the two mixed-floor products
in the middle column are committed scope. If the option is approved, it needs a distinct label-basis
identity, metrics, head, calibration, and raster provenance; it cannot reuse the primary product's
scientific claims by implication.

> **Update 2026-08-06 (fixing session). The rule immediately below is SUPERSEDED for the test suite.**
> Isolation criteria **1–4 are closed**, and the mechanism claim in "Why the R77 safety claim is not
> structurally true" is **partly refuted** —
> see [the measurement](#measured-2026-08-06-which-writers-actually-write-through-a-hard-link).
> **The full suite has been run and verified non-mutating**: `pytest` → **581 passed** (560 fast +
> 21 slow), with an 11,218-file path/size/mtime manifest over all six artifact roots bit-identical
> before and after. That is the first time it is safe *by construction* rather than by luck.
> Only criterion 5 — an independent backup of the ignored trees, ≈110 GB — remains, and it gates the
> **rebuild**, not the suite. Producers pointed at repository roots by hand, and notebooks, are still
> uncovered: the runtime guard is test-only.

## Immediate operational safety rule (superseded — see the update above)

> **Do not run an unfiltered/full pytest suite, any slow producer-calling test, or a pipeline producer
> against the repository's dataset/cache roots until the isolation gate below is closed.**

`pytest -m "not slow"` was safe under the audited marker assignments, but marker drift must be checked
if new tests have landed. A specifically inspected test that uses synthetic data and unique temporary
roots is also acceptable. Do not rely on a before/after checksum as the primary guard: a checksum
detects damage after the write and does not provide rollback.

### Why the R77 safety claim is not structurally true

The R77 redirects landed, but the claim that the full suite is now non-mutating is conditional on the
current cache contents:

- `tests/conftest.py::read_only_cache` hard-links `hirise_decimated` into a temporary cache. A hard link
  is another name for the live inode, not a read-only copy.
- Stage 2 and Stage 3 tests consume that hard-linked file through
  `read_full_footprint_decimated`. If the cached CRS differs from the current corrected CRS,
  `src/hirise_imagery.py` rebuilds the cache path using `rasterio.open(..., "w")`. Truncating the
  temporary hard link would therefore truncate/rewrite the live derived TIFF.
- The 511-pass checksum run did not encounter the invalidation branch because the current CRS already
  matched. That run is useful evidence about that one cache state, not proof of isolation under a stale
  cache, a pre-SP1 cache, or a future invalidation rule.
- The review's phrase "six tests" means six test files/redirections, but there are eight
  producer-calling test functions across those files. Stage 1, 4, and 4b outputs have been redirected
  safely; the Stage 2/3 read/write separation remains conditional because of the hard link.

Two related escape paths also remain:

- `config_v2_dev.yaml` points at `cache_v2_dev`, which is a junction to the live `cache_v2`; it is not
  an isolated development cache.
- `Config` resolves relative paths against the repository root, not against the copied config file's
  directory. Copying a YAML file into a temporary directory while leaving relative paths unchanged
  still targets repository data.

The existing mutation-hazard memory note correctly says that producer safety remains unresolved. Its
older categorical frontmatter and the review's/PENDING_REBUILD's stronger "full suite safe" wording
must not override the code-level analysis above. `CLAUDE.md` and `.claude/settings.json` also need a
warning/permission review because they currently make full pytest and producer commands too easy to
invoke against live roots.

### Measured 2026-08-06: which writers actually write through a hard link

The analysis above was right that the fixture's own stated invariant was broken, and wrong about the
consequence. Controlled probe, wholly inside a temp directory, rasterio 1.5.0 / GDAL 3.11.4 / NTFS:

| write API | source reached through the link? |
|---|---|
| `rasterio.open(p, "w")` | **no** — deletes-then-creates; the link breaks and the source survives |
| `rasterio.open(p, "r+")` | **yes** |
| `open(p, "wb")` | **yes** |
| `Path(p).write_text(...)` | **yes** |
| `shutil.copy2(other, p)` | **yes** |
| `Path(new).replace(p)` | no — swaps a directory entry |

`read_full_footprint_decimated` rebuilds via `rasterio.open(cache, "w")`, so **the specific truncation
this document predicted does not fire on the current stack**. The defect is therefore a latent design
error — a staged directory that is also a write target — that current library behaviour masks, not a
demonstrated data-loss path. Nothing rests on that masking any more: the fix copies instead.

Also corrected: `slow` was never the control. Re-auditing markers found **20 non-slow tests that call
a producer** (all of them writing to `tmp_path`). The static scan and runtime guard are the control.

Two other checks came back clean and are worth not re-deriving: the SP1=20 Mars equirectangular CRS
**does** survive a GeoTIFF round trip, so `_crs_equal` converges and a corrected cache is not rewritten
on every call; and `pytest -m "not slow"` (512 passed, 21 deselected) left an 11,218-file
path/size/mtime manifest of all six artifact roots bit-identical with the guard installed.

### Isolation gate acceptance criteria

Before running slow tests or a rebuild:

1. ✅ **CLOSED 2026-08-06.** Mutable derived inputs such as `hirise_decimated` must be copied/rebuilt in
   a scratch cache, never hard-linked. Large immutable source archives may be linked only where the
   called code has no write, replace, or invalidation path to those names. — `read_only_cache` now
   copies everything except `{tile}.zip` / `{obs}_RED.JP2`; sidecars beside an archive (including GDAL
   PAM `.aux.xml`, which GDAL rewrites in place) are copied. Every copy asserts a distinct inode, and
   teardown asserts each linked source is unchanged in size and mtime.
2. ✅ **CLOSED 2026-08-06.** Every producer test must receive independent absolute cache and output
   roots. Those roots must not be junctions, symlinks, or hard links to a live mutable tree. — each
   `read_only_cache(...)` call gets its own root under `tmp_path`; the static scan below enforces it.
3. ✅ **CLOSED 2026-08-06.** Add a guard that rejects producer output/cache roots under the live
   repository artifact roots during tests. Reproduce stale-CRS invalidation only with two entirely
   temporary roots, and assert the synthetic source remains unchanged. Separately prove through a
   static/runtime write guard that repository artifact roots were never opened for write. —
   `tests/live_artifact_guard.py` (autouse, session-scoped) refuses `open`/`os.open`/`os.replace`/
   `os.remove`/`os.link`/`shutil.*`/`numpy.save*`/`rasterio.open(mode!="r")`/`to_parquet`/`to_file`
   under `cache*`, `dataset*`, `models`, `reports`, resolving the `cache_v2_dev` junction; a static AST
   scan in `tests/test_artifact_isolation.py` fails even when a producer test skips; the stale-CRS
   regression there runs on two temporary roots and exercises the invalidation branch the 511-pass run
   never entered.
4. 🟡 **PARTIAL 2026-08-06.** Parameterize scripts that currently hard-code `dataset_v2`, embedding,
   model, calibration, or map paths. A scratch rebuild must be able to run without writing any live
   ignored tree.
   - ✅ **Stages 1–5** are already config-driven, and the recipe is now pinned by test:
     `Config.resolve` is `(root / raw).resolve()` and `Path.__truediv__` discards the left operand
     when the right is absolute, so **absolute** `cache_dir` / `output_dir` values in a copied YAML
     genuinely redirect. Relative ones do not, wherever the file lives. Both directions are asserted
     in `tests/test_artifact_isolation.py`.
   - ✅ **`scripts/bank_calibration.py`** takes `--predictions`, `--labels-dir`, `--out`; it also no
     longer saves before evaluating its gates, and no longer fails open.
   - ⬜ **Still hard-coded** (exact lines, so the next session need not re-survey):

     | script | hard-coded roots |
     |---|---|
     | `map_region.py:55-58` | `cache_v2/ctx_tiles`, `models/deployable`, `reports/map_region` |
     | `map_pilot.py:38-43` | `dataset_v2`, `cache_v2/ctx_windows`, `cache_v2/ctx_tiles`, `models/deployable`, `reports/figures`, `reports/map_pilot` |
     | `train_deployable_head.py:43` | `dataset_v2` |
     | `parity_check.py:36-37` | `cache_v2/ctx_tiles`, `models/deployable` |
     | `striping_a1_map.py:58-59` | `models/deployable_a1/86c51a5dca220f63`, `reports/map_a1` |
     | `striping_a1_loio.py:44-46` | `dataset_v2`, `reports/figures` |

     `map_region.py` and `map_pilot.py` already expose `--out-dir` / `--model` / `--calibration`, so
     the missing pieces are the *input* roots. The A1 pair needs the same treatment plus the
     arm-specific head/calibration parameterization the "Product semantics" section requires.
   - ⬜ The runtime write guard is **test-only**. Scripts and notebooks are not covered by it, so the
     absolute-scratch-root discipline is the only control there.
5. ⬜ Snapshot ignored caches, datasets, models, and reports separately before regeneration.
   **Deferred 2026-08-06** pending an external drive; the interim reconstruction plan is
   [ARTIFACT_RECOVERY.md](ARTIFACT_RECOVERY.md), which also records the ~30 GB that **no** plan can
   reconstruct -- including the 4.18 GB of BoulderNet detections that live outside the repo and were
   outside every artifact manifest taken during this review. Pushing git
   commits does not back up those artifacts, and hard links are not independent backups.
   Scale, measured 2026-08-06 (`Get-ChildItem -Recurse` does not follow the reparse points, so
   `cache_v2`'s figure excludes the archive subdirs it shares with `cache` by junction):

   | root | size | in the backup? |
   |---|---|---|
   | `dataset_v2` | 78.3 GB | **yes** — irreplaceable |
   | `reports` | 21.2 GB | **yes** |
   | `dataset` (v1, frozen) | 5.0 GB | **yes** — superseded but not reproducible under current code |
   | `models` | 1.4 GB | **yes** |
   | derived cache subdirs (`ctx_windows`, `coregistration`, `reprojected_detections`, `hirise_decimated`, both trees) | 4.3 GB | **yes** — expensive to recompute |
   | `cache` + `cache_v2` remainder (CTX tile zips, HiRISE JP2s) | ≈61 GB, shared via junction | no — re-downloadable |

   So the irreplaceable set is **≈110 GB**, not the ≈180 GB the roots total. Note the junction: a
   naive `robocopy cache cache_backup /E` would follow it and duplicate 61 GB of re-downloadable
   archives. Exclude `ctx_tiles` and `hirise_jp2` explicitly.

Criteria 4–5 gate the **rebuild**, not the test suite. What is still unproven for the test suite: the
four reworked slow producer tests have not been executed since the fixture changed, so the `only=`
staging filter is verified only by a read-only listing of the files each producer needs.

## Corrected current state of the review

The register should be normalized before it is used top-down. Keep separate notions of finding state,
evidence strength, affected object, and the exact stage/product blocked.

| Finding | Correct current interpretation |
|---|---|
| **R01** | OPEN and required for both regional products. Define one globally anchored coarse grid before per-tile inference. Reprojecting already local-phase prediction TIFFs only at merge time is a stopgap, not the fresh-build fix. |
| **R04** | **FIXED 2026-08-06.** Stage 5 returns 1 and names the schemes whose packaged output is now stale; packages record per-obs labels/features SHA-256 plus each label sidecar's R74 `inputs` block; `loaders.verify_package_freshness` runs from `load_metadata`/`load_fold` and raises on split-hash drift, cohort drift, or content drift. Packages predating the field warn instead of failing — all seven live packages pass with that warning. |
| **R06** | OPEN for the active A1 deliverable. `reports/map_a1/` does not exist, so A1 is planned, not shipped. |
| **R07** | OPEN and verified, but the old fix wording is incomplete. Training used one native-resolution statistic per Stage-2 observation window, while deployment partitions by SeamMap source frame and currently derives statistics from 160 m data. Choose and version one statistical unit, re-embed/retrain or prove equivalence, and make training and deployment identical; resolution alone is not parity. |
| **R08** | OPEN. Define and test how A1 handles unlabelled/small frames; do not emit a mixture of raw and normalized DN. |
| **R13** | OPEN for both maps. Record and test a context-window nodata threshold; checking only the central 32-pixel tile admits embeddings whose surrounding 96-pixel context is mostly nodata. |
| **R14** | OPEN. A requested new map generation must not silently accept an existing final TIFF as complete. |
| **R29** | ✅ **CODE FIXED 2026-08-06 — rebuild pending.** `src/labeling._shift_coverage_mask` translates the coverage mask by the same Stage-3 `(dx, dy)` as the polygons before eligibility gating, filling vacated area as ineligible; recorded in the label sidecar as `coreg_mask_shift`. Validated read-only against R75 by an independent route: **6,202 tiles lose eligibility — R75's number exactly** — while **6,255 gain it on the advancing edges**, so the eligible set is *re-registered*, not shrunk (161,005 → 161,058 at S=32). **Register correction:** its fix bullet says the shifts are "already quantised to CTX pixels"; measured, **0 of 39** are — they are 1/20-px quantised, so the mask rounds to the nearest whole pixel (residual ≤ 2.5 m vs a 194.7 m median shift). See DECISIONS 2026-08-06p. |
| **R31** | OPEN and now active because Stage 2 is in the rebuild. Crop the raster `Window` and derive the transform from the clipped window. Reconstructing bounds from array shape cannot repair west/north overhang. |
| **R38** | OPEN, medium, and required for A1. The nodata collision is real, but the alleged A1-payoff footprint confound was refuted. The committed-product gate requires an explicit nodata mask. `[1,255]` is useful only as a temporary sentinel-preserving diagnostic because it moves information loss to DN 1. |
| **R74** | CODE FIXED — REBUILD PENDING. **Tests and provenance landed 2026-08-06**, so it is now usable as a rebuild boundary: ten synthetic hole/threshold/topology tests, the threshold promoted to `ctx_retrieve.max_interior_hole_px`, and a `method`/`version`/threshold/filled-count/SHA-256 record persisted in Stage 2 and propagated through Stage 3 (`shift_id` + input digests) into Stage 4 (`inputs.*`). |
| **R77** | **FIXED 2026-08-06.** Direct test-output redirects landed 2026-08-05; the residual hard-link staging hole is now closed by copy-staging plus static and runtime write guards. The predicted `rasterio "w"` truncation mechanism was measured and does not fire on this stack; the fix no longer depends on that. |
| **R78** | **FIXED 2026-08-06.** The last `(0,0)` illumination fixture is re-based on the real phase, and both outstanding mutants — drop the mosaic origin from the labels bounds, flip the origin sign in the features window arithmetic — are now killed. Run on independent scratchpad copies; the working-tree `src/` was never modified. `test_alignment_aligned_window`'s zero origin is deliberate and retained. |
| **R87/R88** | **FIXED 2026-08-06.** Guards added and mutation-verified (four mutants, each previously green). Confirmed non-corrupt as filed: production splitting is group-aware, and all 620 packaged X parquets under `dataset/` and `dataset_v2/` are free of label columns. R88 also added a *production* second filter in `src/modeling/loaders.py`, which raises rather than silently dropping. |
| **R91** | FIXED by differently shaped and origin-asymmetric rectangular within-image extents; the main register's OPEN line is stale. |
| **R92** | REFUTED AS FILED and corrected to historical v1 drift. With v1 superseded, no rebuild action remains. |
| **R97** | **CODE FIXED 2026-08-06 — within-image rebuild pending.** The step now comes from the scales present in the labels. Measured: the cut moves for 29 of 38 v2 images; v2's persisted cuts match the inflated step 38/38 and **v1 matches the correct step 8/8**, so v1's within-image split was never drifted — the splitter was. `dataset_v2/splits/within_image_4fold.json` and `packaged/within_image_*` are now the stale artifacts. |

### Corrections to proposed actions and summaries

- **R03:** the mechanism survives but the original number does not. Enforcing one common physical
  floor would remove roughly 67% of the fine cohort's labelled boulder area; that is an optional target
  redefinition, not housekeeping. The primary decision is now to retain and document mixed floors.
- **R24:** emitting another `_n` field alone would not catch the error. Constant-prediction folds must
  contribute zero rank skill and be separately counted/reported.
- **R36:** replace "could not have failed" with "received an approximately fivefold attenuated
  treatment". The gate is monotone and can fail.
- **R38:** retain the mechanism and corrected remediation above; drop the instruction to re-score the
  A1 payoff on a common mask because the compared footprint was already common.
- **R44:** retain the verified scope, not the original arithmetic.
- **R45:** use matched quadrant-to-quadrant scoring. Do not bootstrap the near-zero quadrant-size effect
  as the null; that would entrench the error.
- **R51:** keep the verified statistic separate from the unsupported scientific verdict.
- **R54:** the original 11/37 `top_ratio` statement is wrong. `top_ratio` is 12/38; the appropriate
  per-place level comparison is 11/38. Report `mean(pred)/mean(true)` for the level instrument rather
  than promoting per-image `top_ratio` into an impossible new gate.
- **R56:** ✅ **CLOSED 2026-08-06.** The finding was verified but belongs at medium severity and on a
  dead-closed comparison. The fixed-target re-score has now been run (DECISIONS 2026-08-06o): the
  "HARMFUL / monotonically degrading" verdict is withdrawn, `conf ≥ 0.5` is a null on the common
  target, and the ruling no longer blocks R23 — but it does not license R23's 0.6173 floor either,
  which was never measured.
- **R60:** Brian's later caveat-header directive supersedes the proposed rerun of the historical PDF.
- **R61:** verified/confirmed but medium, not high.
- The old finding-area totals, pass counts, statements that no reviewer ran GDAL/tests, and the old
  priority ordering are internally inconsistent with later review passes. Regenerate them rather than
  carrying any existing total forward.
- The review says unverified findings cannot be acted on, while the prior handoff asks the fixing
  session to work on several of them. The operating rule is: reproduce/verify the defect while fixing
  it, record the evidence, then update the finding state. Do not apply an unverified raw `Fix:` bullet
  merely because it appears in the register.

## R74 tests and provenance required before rebuild

The small enclosed-shadow-hole fill is implemented, but its current boundary is implicit and largely
human-readable:

- Add synthetic tests for a small enclosed zero hole, a hole larger than the threshold, an
  edge-connected invalid region, and `max_interior_hole_px=0` as an exact no-op.
- Make the algorithm/method and threshold explicit configuration or an explicit versioned producer
  parameter rather than relying only on the default value 16.
- Persist the method/version, threshold, filled-pixel count, and output mask digest in Stage 2
  provenance. A config hash over YAML alone cannot distinguish pre- and post-R74 masks.
- Bind Stage 3 provenance/freshness to the input CTX-window and coverage-mask digests/version, then
  propagate the Stage-3 shift artifact digest and coverage-mask identity into Stage 4. Recording only
  pathnames cannot distinguish generations.

The reported 3,236 recovered tiles and prevalence shift to 0.3733 are an isolated R74 counterfactual
using the existing Stage-3 shifts and current R23/R29 behavior. They are not guaranteed final-rebuild
counts because the required Stage-3 rerun and R23/R29 fixes can change alignment, eligibility, and
targets.

Without those fields, pre-R74 and post-R74 masks/labels can have indistinguishable sidecars, which is
the Pattern-D provenance failure the rebuild note is meant to control.

## Product semantics and provenance

The primary mixed-floor maps must name their target honestly. Persist the minimum included-label
polygon size/filtering convention for each HiRISE training image and an aggregate product-level
description of the calibration pool's mixture. Keep this distinct from detector completeness: a
regional output pixel does not itself inherit a HiRISE-specific detection floor. R83/R84's
78.4%/21.6% estimate refers to tile share in the calibration pool, not image share, and has not been
independently verified; label it accordingly if quoted.

Do not call the result size-independent "physical rock abundance." Reader-facing prose and raster/model
metadata must state which boulders are included by the training-label convention. If the common-floor
option is later approved, specify its numeric minimum-included diameter/area rule first and keep all
target-dependent lineage separate. Within one preprocessing arm, embeddings may be shared only when
the image keys and pixel-input hashes are identical; LOIO predictions, heads, calibrations, metrics,
and claims remain separate.

For the baseline-versus-A1 comparison:

- Use the same corrected global grid, footprint, nodata semantics, and comparison mask.
- Define A1's estimator scope explicitly. Current training computed one native-resolution statistic
  per Stage-2 observation window, while deployment partitions by source frame and can see different
  fragments in overlapping read windows. The preferred contract is a fixed native-resolution
  statistic computed once per declared SeamMap source-frame footprint, with the same pixel-to-frame
  assignment used when rebuilding training embeddings. Do not use sliding-window-local statistics.
- Enforce that full A1 contract in embeddings, LOIO predictions, deployable head, calibration, and
  regional inference. Resolution agreement alone is insufficient.
- Give each radiometric arm its own hashes and provenance. Do not apply an A1 transform only at map
  time to a head/calibrator trained on baseline inputs.
- Fetch and verify all tile-level inputs needed for the planned 26 CTX tiles. A missing zip, required
  source-frame metadata/statistics artifact, output tile, head, or A1 calibration must produce a
  nonzero failure, not a reduced "common" footprint. Pixel-level unlabelled or small-frame cases inside
  an otherwise complete tile follow the explicit tested R08 mask/fallback contract. Raw A1 probability
  is a QA output, not a substitute for the required calibrated abundance layer.
- Make both drivers consume one versioned, globally anchored grid specification. The current A1 path
  derives its grid from an existing baseline abundance raster; preferably decouple that dependency.
  Until then, render the corrected baseline first and never mistake parity with the existing
  R01-affected grid for the desired corrected-grid parity.
- Record the actual selected head and calibration hashes in the manifest and tile sidecars. The
  current A1 path can report the global default head even when a different command-line head is used;
  fix that before accepting provenance.
- Parameterize LOIO and calibration banking by preprocessing arm and output generation. The current A1
  LOIO CSV drops `ti,tj`, while the baseline banker expects `obs_id,ti,tj`; every arm's prediction
  artifact must retain unique tile keys. The banker must accept explicit input/output paths, enforce
  completeness, pass all gates before writing, and fail nonzero on rejection.
- Include the preprocessing arm, embedding-store identity/digest, and target definition in the head
  recipe/card/hash. At map time, verify head/calibrator compatibility rather than independently choosing
  a lexicographically "latest" head and a fixed calibrator, and record both content digests in tile and
  region manifests.
- Do not write "A1 shipped" until the files exist, pass numerical-parity and footprint checks, and are
  identified as the selected deliverable in current documentation.

## Pre-rebuild gates

| Gate | Must be true before regeneration starts |
|---|---|
| Safety | Isolation criteria 1–4 ✅ **CLOSED 2026-08-06** (copy-staging, runtime write guard, static AST scan, every rebuild-DAG script root now a flag), and the full suite verified non-mutating at 581 passed. Still open: criterion 5 — an independent ≈110 GB backup of the ignored trees. The guard is test-only, so notebooks and hand-run producers still rely on the absolute-scratch-root discipline. |
| Review state | Stale statuses and rejected fix alternatives are normalized; R91 is closed, R78 is marked partial, and R97 is in the live queue. |
| Label semantics | R56 ✅ **CLOSED 2026-08-06** — re-scored with the target held fixed; verdict withdrawn. R23 remedy ✅ **DECIDED** (Brian: retain + document, temporary pending v3). Mixed-floor metadata contract ✅ **BUILT** (`source_integrity` / `null_geometry_basis` / `realised_label_basis`) but ⬜ **not yet emitted** — zero banked sidecars carry it, so Stage 1 + Stage 4 must re-run. Common-floor target ✅ numerically specified (**0.617257475852966**) and priced, and explicitly **deferred**. All: DECISIONS 2026-08-06o. |
| Stage 1 | Run it if source geometry/filtering, CRS logic, or Stage-1 provenance changes. It is required if the R23 filtering/provenance fix lands. |
| Stage 2 | R74 tests and provenance ✅ **landed 2026-08-06**. Still open: R31 and R67; resolve R66 before using any download/cold-cache path. |
| Stage 3 | Provenance binding ✅ **landed 2026-08-06** (input digests + coverage-mask identity + a `shift_id` propagated into Stage 4). Still open: R65. |
| Stage 4 | R29 ✅ **fixed 2026-08-06** (coverage mask now shifts with the polygons; `coreg_mask_shift` provenance). R68 still open before labels are generated. R80 is reproduced and covered end to end with projected units, a non-`None` size filter, and a fixture that distinguishes diameter from radius; retained mixed-floor semantics make this path load-bearing. |
| Stage 4b | R27 and R28 ✅ **landed 2026-08-06**, so an ordinary Stage-4b regeneration is now the right move — no patch-only path is needed. Patches are regenerated from current labels alongside the corrected `lacunarity_*` (NaN, not the `0.0` sentinel) and `edge_*` (quantile thresholds 0.80/0.90). |
| Stage 5 | ✅ **CLOSED 2026-08-06.** R04 propagates failure as a nonzero exit; package metadata binds per-obs label/feature content digests plus each label sidecar's R74 `inputs`, and `loaders.verify_package_freshness` enforces them from `load_fold`. R87/R88 guards exist and are mutation-verified. R97 is fixed, so within-image splits regenerate onto the correct snap step. |
| Modeling/calibration | `fm-embeddings-3`/`fm-embeddings-4` are fixed so arm/suffix, store provenance, existence-only resume, persisted model hash, and nuisance-basis consistency are enforced. LOIO artifacts retain unique tile keys; calibration asserts a complete one-to-one join. R09 plus `calibration-3`/`calibration-4` compatibility and fail-before-write defects are fixed. R54's corrected per-image `mean(pred)/mean(true)` level instrument is emitted beside pooled results, and the pooled-vs-per-place shipping rule is explicit. |
| Mapping | R01 creates the globally anchored grid before inference; R13 records/enforces context nodata; R14 prevents false resume; R84 product metadata is emitted. |
| A1 | R07's statistical unit and R08 pixel-level fallback contract are fixed/tested; R38 uses an explicit nodata mask; all 26 tile-level inputs/outputs and A1-specific head/calibration artifacts are required; parity passes end to end. |

R87/R88 are included as catastrophic-regression guards even though their corresponding production
paths are correct today. This distinction should remain visible in the register.

## Complete v2 rebuild DAG

The current `PENDING_REBUILD.md` chain omits data dependencies. The batched v2 rebuild must be:

1. Stage 1 if source geometry/filtering, CRS logic, or Stage-1 provenance changes. In particular, run it
   if the R23 drop-null/filtering provenance fix lands, even if the source GPKG geometry is unchanged.
2. Stage 2 CTX windows and coverage masks in isolated roots.
3. Stage 3 co-registration and coregistration QA. Stage 3 consumes the Stage 2 coverage mask when it
   selects its FFT window and computes its block field, so R74 requires this stage and its provenance
   must bind to the exact CTX-window/mask digests.
4. Stage 4 labels, then Stage 4b patches/features.
5. Stage 5 splits and packaged datasets with input content digests/generation IDs. Recovered R74 keys
   cannot enter downstream training until the packages are rebuilt and loaders verify their lineage.
6. Fresh baseline and A1 embeddings in new stores or under an explicit overwrite operation. Build A1
   embeddings using the approved fixed statistic scope and pixel-to-source-frame assignment. A
   common-floor target may share an arm's embeddings only when keys and pixel hashes match.
7. Forced frozen-recipe LOIO predictions for all seeds and the ensemble for each active preprocessing
   arm/target. Preserve `obs_id,ti,tj` and do not reuse an existing predictions artifact merely because
   it exists.
8. Train the all-data deployable head for each active preprocessing/label arm, with arm/store/target
   identity in its recipe hash and card.
9. Fit a compatible calibration layer for each corresponding arm, with completeness and anti-join
   assertions so recovered/missing keys cannot disappear silently. Emit the pooled calibration results
   and R54's per-image `mean(pred)/mean(true)` distribution/count in band, record which aggregation
   level governs promotion, and pass all gates before writing.
10. Materialize one versioned globally anchored regional grid. Prefer making both map drivers consume
    it independently; if the current A1 baseline-raster dependency remains, render baseline tiles
    first.
11. Render all 26 corrected-grid baseline and A1 regional outputs to new generation paths with the
    required arm-specific head and calibration. Any missing input/output is a failure. Do not resume
    from an existing final TIFF unless its complete upstream provenance matches.
12. Build the mosaics, run numerical-parity/footprint/nodata/scientific QA, promote validated outputs,
    and update every reader-facing metric and document.

`dataset/` v1 is explicitly excluded. The rebuild concerns `dataset_v2` and new versioned downstream
artifacts only.

The required lineage should identify, at minimum: source and coverage-mask algorithm/digest; Stage-3
input and shift digests; label/feature content generation; label convention; split identity;
preprocessing arm and A1 statistic scope; embedding store/model; LOIO prediction generation;
deployable head; calibration; and raster grid/footprint. A git commit or YAML-only config hash is not
sufficient when cached derived inputs can change independently.

## Recommended fixing-stage order

> **Progress 2026-08-06 (fixing session — CLOSED).** Steps **1 and 3 are DONE**: R77 (test-side
> isolation), R78, R87, R88, R27, R28, R97, R74 tests+provenance and R04 are all closed, with mutation
> verification wherever the finding was about absent coverage. Step 1's script half is done too
> (isolation criterion 4); only the *notebook* half and criterion 5's backup remain, and the runtime
> guard is still test-only. Step 2 is partly done (register statuses, PENDING_REBUILD,
> DATA_DICTIONARY, CLAUDE.md, index); **README/ROADMAP/SHERLOCK were not touched**. Steps 5–6
> untouched. Full suite: **581 passed**, artifact manifest bit-identical across every run.
>
> ### Step 4 — R56 **DONE 2026-08-06**; R23 root cause **FOUND**, remedy **awaiting Brian**
>
> Full measurements and corrected numbers: **DECISIONS 2026-08-06o**. Both passes were read-only on
> banked artifacts — no producer ran and no live artifact changed.
>
> **R56 — RESOLVED, verdict withdrawn.** Re-scored the three banked arms against **one common
> target** (161,005 tiles × 38 images, keys row-aligned; the `none` arm's `y_true` is bit-identical to
> the shipped `dataset_v2` S=32 labels). The reviewer's decomposition reproduces to 4 dp: of the
> recorded −0.021 per-image Spearman loss at `conf ≥ 0.5`, **−0.0172 (82 %) is the target moving** and
> only −0.0034 (17/38, p=0.43) is the model. On the project's standard metrics `conf ≥ 0.5` is a
> **null** (`meaningful_auc` −0.0028 p=0.31; `pr_auc@1e-2` −0.0007 p=0.44; `precision@5%` 0.0000
> p=0.41; min p over nine metrics 0.178). `conf ≥ 0.7` is directionally harmful but survives no Holm
> correction. "Monotonically" is false in the probe's own scorecard (per-image ρ 0.4333 → **0.4563** →
> 0.3044), and the dynamic-range collapse is 53–59 % population artefact — after the **shipped**
> quantile-match layer `top_ratio` is 0.870 → 0.859 → 0.829, essentially flat.
> **The R56 ruling no longer forbids harmonising a confidence floor.** It does **not**, however,
> license 0.6173: no arm was measured there, and it interpolates to 56–70 % of the `conf070` harm.
> Amended in `PLAN_Calibration.md` (§L2 lever, stage table, order rationale), `notebooks/_build_23.py`
> §8/§9 and its verdict table, and `DECISIONS.md:3916-3925`.
> ⚠ `notebooks/23_calibration_diagnostic.ipynb` is now **stale** against its `_build_23.py` source and
> needs regeneration; it was deliberately not executed (notebooks are uncovered by the runtime guard).
>
> **R23 — the register's framing of the root cause is wrong, and it changes the menu.** This is not a
> BoulderNet export artefact. The three `.shp` files are **byte-truncated**: each one's own header
> declares a length far larger than what is on disk, while `.dbf` and `.shx` are complete. Records are
> stored **score-descending**, so the records whose bytes physically fit are exactly the highest-scoring
> prefix — and that count equals the pipeline's kept count with **zero** delta in all three images.
>
> | ObsId | header declares | on disk | missing | records that fit | pipeline kept | Δ |
> |---|---|---|---|---|---|---|
> | ESP_017355_2260 | 569,266,636 B | 214,884,317 B | 354.4 MB | 359,933 | 359,933 | **0** |
> | ESP_046803_2325 | 323,962,020 B | 192,091,266 B | 131.9 MB | 367,140 | 367,140 | **0** |
> | ESP_068483_2280 | 616,023,244 B | 443,015,777 B | 173.0 MB | 727,160 | 727,160 | **0** |
>
> Three control images are byte-exact complete. `drop_null_geometries` was faithfully reporting a
> **truncated file**. So there is a **fourth remedy the register does not list — re-copy the 659 MB of
> missing bytes and restore the data** rather than working around its absence. Whether that is possible
> depends on a complete source copy existing off this disk; nothing in the repo can establish that, and
> criterion 5 already flags these 4.18 GB as un-backed-up and outside every artifact manifest.
>
> The three original remedies, priced read-only (adversarially verified; several first-pass numbers
> were wrong and are not carried forward):
> 1. **harmonise upward** to the true max-of-minima **0.617257** (the register's rounded 0.6173 is
>    4.25e-05 *above* it and would drop 86 polygons from `ESP_017355_2260` itself) — the 36 unaffected
>    images retain a median **33.1 %** of kept detection area, **53 %** of cohort label mass is
>    discarded, ~24,000 of 161,005 tiles change rich/poor class;
> 2. **exclude the truncated images** — costs **2** images, not 3 (`ESP_046803_2325` has no labels and
>    was dropped for a coregistration failure, `DECISIONS.md:1258`): 18,754 tiles (11.65 %), 21.9 % of
>    rich tiles, 19.9 % of labelled boulder mass, LOIO 38→36 folds, and it deletes the cohort's largest
>    observation (13,457 S=32 tiles, 1.84× the runner-up);
> 3. **retain and document a mixed confidence floor** — keeps ~2,000–2,250 mis-classed tiles (1.2–1.4 %)
>    and leaves all 18,754 tiles in those two folds carrying a level-biased target (low by ≈2.4× and
>    ≈1.35× on own-anchored estimates).
>
> **Correction to this document's own earlier wording:** confidence filtering is a **Stage 4**
> operation (`src/labeling.py:96` `_apply_detection_filters`, called at `:496` inside
> `stage4_one_image`) — `DECISIONS.md:813` already said so. Only the *provenance* half of R23's fix
> (recording dropped-vs-kept score distributions, flagging rank truncation in `drop_null_geometries`)
> is Stage 1. The Stage-1 gate still fires for that half under every remedy.
>
> **REMEDY DECIDED (Brian, 2026-08-06): retain and document, as a temporary measure pending the v3
> re-detection.** Not adopted as a target definition; harmonisation, exclusion and byte-range
> recovery all remain open. The provenance/documentation half is **built** —
> `inspect_shapefile_integrity` + `describe_null_geometry_drop` in Stage 1 (persisted as
> `source_integrity` / `null_geometry_basis`, both warning loudly), `realised_label_basis` in the
> Stage-4 sidecar, `DATA_DICTIONARY` entries, 9 tests, fast suite 569 passed. Full detail:
> **DECISIONS 2026-08-06o**.
>
> **A fourth truncated export, and a gate that checked the wrong file.** Scanning all 40 vClaire
> exports with the new check finds **`ESP_028537_2270`** truncated too (513.4 MB of 571.9 MB
> missing). That one was already known and excluded, and `scripts/build_vclaire_manifest.py` carries
> an integrity gate written *for* it — but the gate validates `.dbf` self-consistency, `.dbf`↔`.shx`
> agreement and pyogrio's feature count, **all of which pass** when only the `.shp` is short.
> `ESP_028537_2270` was caught because its `.dbf` was truncated as well. The gate now also checks the
> `.shp`. This is a Pattern-D instance worth carrying into the register: the failure mode was
> diagnosed, named, and guarded, and three further instances still reached the shipped label basis.

1. Close the mutation/isolation hole and parameterize artifact roots. Update the operating manual and
   command permissions before anyone runs a broad test command.
2. Normalize the review register and stale README/ROADMAP/SHERLOCK/PENDING_REBUILD claims so the next
   session cannot accidentally resume the aborted F path or claim A1 already exists.
3. Finish R78; add R87/R88 guards; fix R97 and R04; add R74 tests/provenance.
4. Resolve R56/R23 and write the mixed-floor metadata contract. Keep the optional common-floor product
   separately named and deferred unless its target is approved.
5. Fix all code that affects Stage 2 through Stage 5 outputs, then the map/provenance blockers and A1
   preprocessing defects listed in the gates above.
6. Execute the complete isolated v2 rebuild once, validate both baseline and A1 products, and only then
   promote artifacts and regenerate claims.

Thermal-validation/map-dependent science legs should remain paused until the R74-dependent head,
calibrator, baseline mosaic, and A1 mosaic have been rebuilt and validated.
