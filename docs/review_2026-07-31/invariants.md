# Review area: invariants

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-01
- **Verification:** self-refuted (single-agent pass; not independently verified)

Scope note: §6 of `docs/CODE_REVIEW_2026-07-31.md` records that the earlier direct pass swept
hardcoded radii, `set_crs`-vs-`to_crs`, presence-AUC greps and torch import order (→ R02, R05) but
**not** full-res HiRISE reads, CTX reads without `window=`, hardcoded observation IDs in `src/`,
missing `truststore`, or notebook `.ipynb`-vs-`_build` drift. I re-ran the swept greps to confirm
nothing new, and concentrated on the five unswept items. Four of the five came back clean (see
*Verified clean*); the notebook-drift item did not, and it is where all the findings below sit.

**Method for the drift check (reproducible).** I copied every `notebooks/_build_NN.py` into a
scratch directory and ran it there (`Path(__file__).parent` makes it write the `.ipynb` beside
itself, so nothing in the repo was touched), then compared the regenerated cell sources
cell-by-cell against the committed `.ipynb`, and separately counted executed code cells. Result:
**21/21** `_build` scripts regenerate; **19** produce byte-identical cell sources; **2** (17, 20)
drift; **2** committed notebooks (12, 13) have zero outputs.

## Findings

### invariants-1 — Notebook 17's committed verdict cell reports the *retracted* Fisher's exact result, contradicting both its own executed output and its `_build` source

- **Severity:** medium
- **Liveness:** dead-closed (compositional programme) — but a committed, reportable QA artifact
- **Confidence:** high
- **Where:** `notebooks/17_provenance_disambiguation.ipynb:881`, `:892` (§4 "Combined verdict")
  vs `notebooks/_build_17.py:334`, `:346` and vs the notebook's own cell output at `:157`

Commit `486af93` ("Tier 1 methodology correction (honest exclusion)") replaced the imputed
`OR = 12.0, p = 0.034` with the honest-exclusion `OR = 23.0, p = 0.018` and its message claims it
propagated to "`notebooks/_build_17.py` + the executed `notebooks/17_provenance_disambiguation.ipynb`".
It updated `_build_17.py` in **both** places (§3 and §4) but the `.ipynb` in only **one** (§3). The
committed notebook therefore self-contradicts: cell output line 157 prints `OR = 23.00, p = 0.0181`,
the §3 markdown at 666-668 says `OR = 23.0, p = 0.018`, and the §4 **Combined verdict** — the cell a
reader stops at — still says `OR = 12.0, p = 0.034`. `DECISIONS.md:2310` explicitly records
`p = 0.034` as superseded.

- **Failure scenario:** a future session (or a co-author) opens `notebooks/17_…ipynb`, reads §4
  "Combined verdict", and quotes `OR = 12.0, p = 0.034` — the number the project retracted — as the
  Tier-1 provenance headline. It is also the only place in the repo where the retracted pair still
  appears without a "superseded" qualifier (`docs/compositional.md:572` and `DECISIONS.md:2241` both
  carry correction notes).
- **Evidence:**
  ```
  notebooks/_build_17.py:334      on transport-indicator terrain (Fisher's exact OR = 23.0, p = 0.018 under
  notebooks/_build_17.py:335      P2_count, honest-exclusion handling of missing terrain data). Tier 2 finds
  notebooks/_build_17.py:346         correlation). We see it at p = 0.018.

  notebooks/17_provenance_disambiguation.ipynb:157   "Fisher's exact two-sided: OR = 23.00, p = 0.0181\n",   <- executed output
  notebooks/17_provenance_disambiguation.ipynb:666   "classified `composition_residual` (Fisher's exact OR = 23.0,\n",
  notebooks/17_provenance_disambiguation.ipynb:881   "on transport-indicator terrain (Fisher's exact OR = 12.0, p = 0.034 under\n",
  notebooks/17_provenance_disambiguation.ipynb:892   "   correlation). We see it at p = 0.034.\n",
  ```
- **Self-refutation attempted:** (a) I checked whether the `.ipynb` cell is a deliberately preserved
  historical record — it is not; there is no "superseded"/"as originally reported" qualifier
  anywhere in that cell, and every other statement in it was updated. (b) I checked whether
  `_build_17.py` is the stale side — it is not: the corrected numbers match the notebook's *own*
  executed output and `DECISIONS.md:2310` / `PLAN_Compositional.md:508`. (c) I checked whether any
  live doc repeats the stale pair uncaveated — `docs/compositional.md:572` and `DECISIONS.md:2241`
  both flag it as the earlier, under-stating value, so the notebook is the only uncaveated site.
  (d) I confirmed the drift is real and not a comparison artifact by regenerating from `_build_17.py`
  and diffing: exactly one cell (index 12) differs, and it is this one.
- **Fix:** regenerate (`python notebooks/_build_17.py`) and re-execute the notebook, or hand-patch
  the two lines in the `.ipynb` to the corrected values so the artifact matches its source.

### invariants-2 — Notebook 20's SUPERSEDED banner exists only in the `.ipynb`; running the documented regeneration command silently deletes it and restores a superseded disposition

- **Severity:** medium
- **Liveness:** live-shipped (the provenance chain behind the frozen `mlp_ens3` recipe)
- **Confidence:** high
- **Where:** `notebooks/20_fang_vit_probe.ipynb:800-812` (banner at `:802`) vs
  `notebooks/_build_20.py:343` (§7 Disposition, no banner); producing commit `177f731`

Commit `177f731` added a 13-line "**SUPERSEDED 2026-06-12**" banner to §7 of the *executed notebook
only* — its own message says "Markdown-only edit (no re-execution)" — and never touched
`notebooks/_build_20.py`. That inverts invariant 10: the `.ipynb` now carries information its
source-of-truth does not. Because `_build_20.py` ends with an unconditional
`NB_PATH.write_text(json.dumps(nb, indent=1), …)`, the documented regeneration step
(`python notebooks/_build_NN.py` then `nbconvert --execute --inplace`, CLAUDE.md) **overwrites the
banner away**, leaving §7 asserting the probe-phase pick (LightGBM on `t1_gem192`) as the standing
disposition — a recipe the project superseded the same day with `mlp_ens3` on `t1+gem192` /
emb-only S=32.

- **Failure scenario:** anyone touching notebook 20 (e.g. to refresh a figure) follows the CLAUDE.md
  protocol, regenerates, and the committed notebook reverts to recommending the wrong head. The
  reverted text reads as current, and the only signal it was ever corrected is a commit six weeks
  back in the log.
- **Evidence:**
  ```
  # committed .ipynb has it:
  notebooks/20_fang_vit_probe.ipynb:800  "## 7. Disposition\n",
  notebooks/20_fang_vit_probe.ipynb:802  "> **SUPERSEDED 2026-06-12 (read as a dated probe-phase record).** This\n",

  # the source of truth does not:
  notebooks/_build_20.py:343      """## 7. Disposition
  notebooks/_build_20.py:345  The probe phase is **closed**. The candidate Tier-1 replacement at both
  notebooks/_build_20.py:346  scales is *Tier-1 features + GeM context-input Fang-ViT embeddings →
  notebooks/_build_20.py:347  LightGBM* (t1_gem192 if pooled PR-AUC is binding; ...)

  $ git show --stat 177f731 | tail -4
   notebooks/20_fang_vit_probe.ipynb    |  13 +++++++++++++
   reports/figures/20_fang_verdicts.png | Bin 77988 -> 87089 bytes
  ```
  (`grep -c "SUPERSEDED 2026-06-12"` → `.ipynb` 1, `_build_20.py` 0. Regeneration confirmed: the
  regenerated notebook differs from the committed one in exactly one cell, index 14, and the
  difference is the missing banner.)
- **Self-refutation attempted:** (a) I looked for a "do not regenerate / banner added manually"
  guard in `_build_20.py`'s docstring — there is none. (b) I checked whether the write is
  conditional or merge-like — it is a flat `write_text`, so regeneration is destructive by
  construction. (c) I checked whether the banner is redundant with an in-repo pointer — the freeze
  is recorded in `DECISIONS.md` and the memory notes, but nothing inside notebook 20's source marks
  §7 as dated, which is exactly what the banner was added to fix. (d) I confirmed this is the only
  instance of an `.ipynb` edited without its `_build` across all 21 pairs.
- **Fix:** move the banner text into `notebooks/_build_20.py`'s §7 cell (one string edit) so the
  source of truth and the artifact agree.

### invariants-3 — Notebooks 12 and 13 are committed with zero executed cells; notebook 12 silently lost the outputs it previously had

- **Severity:** low-medium
- **Liveness:** dead-closed (Phase-A2 compression / W0 heterogeneity) — but the numbers they narrate
  are cited forward
- **Confidence:** high
- **Where:** `notebooks/12_compression_diagnostic.ipynb` (11 code cells, **0** with outputs; first
  `"execution_count": null` at `:44`), `notebooks/13_per_image_heterogeneity.ipynb` (13 code cells,
  **0** with outputs; first at `:97`); regressing commit `6e3b9f1`

CLAUDE.md's protocol is "regenerate with `python notebooks/_build_NN.py` **then**
`nbconvert --execute --inplace`". These two are the only committed notebooks that skipped the second
half. Notebook 12 was fully executed at commit `a003d33` (0 cells with `execution_count: null`);
commit `6e3b9f1` edited `_build_12.py` by 9 lines, regenerated, and committed without re-executing —
the diff shows the `.ipynb` shrinking by ~1,000 lines as its outputs were wiped. Notebook 13 was
born unexecuted in the same commit. Notebook 13 is also the one that computes its statistics inline
(`stats.spearmanr` at `_build_13.py:488` and `:524`) rather than calling `src/`, so with no outputs
there is nothing on disk showing what those cells produced.

- **Failure scenario:** a reader opening `13_per_image_heterogeneity.ipynb` — the artifact behind the
  "bimodal per-image AUC, three failure modes" framing that opened the whole Stage-6 docket — sees
  narrative markdown asserting numbers above thirteen empty code cells, with no way to tell whether
  the assertions match what the code would produce today. This is the same class of exposure as R24
  (a notebook-era Spearman that turned out to be a mean over 5 of 20 folds).
- **Evidence:**
  ```
  $ git show a003d33:notebooks/12_compression_diagnostic.ipynb | grep -c '"execution_count": null'
  0
  $ grep -c '"execution_count": null' notebooks/12_compression_diagnostic.ipynb
  11
  $ git show --stat 6e3b9f1 | grep notebooks
   notebooks/12_compression_diagnostic.ipynb     | 1209 ++++---------------------
   notebooks/13_per_image_heterogeneity.ipynb    |  946 +++++++++++++++++++
   notebooks/_build_12.py                        |    9 +-
   notebooks/_build_13.py                        |  923 +++++++++++++++++++
  ```
- **Self-refutation attempted:** (a) I checked whether the outputs are stripped repo-wide by an
  nbstripout filter or `.gitignore` rule — they are not: 26 of 28 notebooks carry full outputs
  (`10_modeling_qa` 24/25, `28_f_verdict` 13/13). (b) I checked whether the narrative numbers are
  orphaned — they are not entirely: `6e3b9f1` also committed
  `scripts/probes/_diag_nb13_correlations.{py,md}`, `_diag_per_image_breakdown.{py,md}` and
  `_diag_extract_nb13_results.{py,md}`, which hold the same measurements. That is what keeps this
  below medium. (c) I checked commit-date ordering across all 21 pairs — no `_build` is newer than
  its `.ipynb`, so this is the only staleness signature present.
- **Fix:** re-execute both (`nbconvert --execute --inplace`), or add a one-line banner in each
  `_build` stating the notebook is committed unexecuted and naming the `_diag_*` probe that holds
  the numbers.

### invariants-4 — `hirise_decimation_mpp` is a required, provenance-hashed config key that no code reads; every call site hardcodes 5.0

- **Severity:** low-medium
- **Liveness:** live-shipped (Stage 2 / Stage 3 run through these call sites)
- **Confidence:** high
- **Where:** `src/config.py:23` (required key) vs `src/ctx_retrieve.py:499`,
  `src/coregister.py:72`, `src/hirise_imagery.py:164`; declared at `config.yaml:42`,
  `config_v2.yaml:72`, `config_v2_dev.yaml:37`, and in the build spec at `docs/build_spec.md:164`

`REQUIRED_TOP_LEVEL` makes `hirise_decimation_mpp` mandatory and `config_hash` hashes the entire raw
config (`src/config.py:240`), so the key's value is stamped into every Stage-1/2/4 sidecar and
dataset row. But a full-repo grep finds no reader: all three production call sites pass the literal
`target_mpp=5.0`. The knob is inert while looking authoritative — and it is exactly the knob one
would reach for after R03 (the 0.25-vs-0.50 m/px HiRISE cohort confound).

- **Failure scenario:** someone sets `hirise_decimation_mpp: 2.5` in `config_v2.yaml` to test whether
  the label confound is a decimation artifact. Validation passes, `config_hash` changes, every cache
  and artifact directory is invalidated and re-derived — and the HiRISE coverage mask, the
  coregistration reference and the decimated cache are all still built at 5.0 m/px. The provenance
  record says 2.5; the data is 5.0. Nothing warns.
- **Evidence:**
  ```
  src/config.py:23        "hirise_decimation_mpp",          # <- required
  src/config.py:240       canonical = json.dumps(cfg, sort_keys=True, ...)   # <- whole config hashed

  src/ctx_retrieve.py:498     hi_arr, hi_transform, hi_crs = hirise_imagery.read_full_footprint_decimated(
  src/ctx_retrieve.py:499         obs_id, jp2_url, cache_dir, target_mpp=5.0,
  src/coregister.py:72            obs_id, jp2_url, cache_dir, target_mpp=5.0,

  $ grep -rn "hirise_decimation_mpp" --include=*.py .
  ./src/config.py:23:    "hirise_decimation_mpp",       # (the only hit outside the YAMLs)
  ```
- **Self-refutation attempted:** (a) I checked for an indirect read via `cfg.get(...)` with a
  computed key or a `**cfg` splat into the reader — none; `read_full_footprint_decimated` is only
  ever called positionally with an explicit `target_mpp`. (b) I checked whether it is deliberately
  vestigial like `ctx_read` — `ctx_read` carries an in-file `# DEPRECATED 2026-05-22` comment
  (`config.yaml:39`); `hirise_decimation_mpp` carries none, and `docs/build_spec.md:164` still
  presents it as live config. (c) I checked the sibling sentinel pattern — `target_crs:
  from_ctx_tile` **is** honoured (`src/ctx_retrieve.py:145-152`), which shows the config-driven
  pattern is real elsewhere and this key is the exception. (d) `DECISIONS.md` grep for
  `hirise_decimation_mpp` / "decimation" returns nothing recording it as intentionally frozen.
- **Fix:** thread the value through (`cfg["hirise_decimation_mpp"]` → `target_mpp` at the three call
  sites, and fix `read_full_footprint_decimated`'s `f"{int(target_mpp)}mpp_full"` cache key, which
  collides for non-integer values — noted as unreachable in `geo-crs.md`, but this change makes it
  reachable), **or** delete it from `REQUIRED_TOP_LEVEL` and the YAMLs and hardcode 5.0 with a
  comment.

### invariants-5 — CLAUDE.md's "notebooks are generated" is false for 7 of 28 notebooks, including one added long after the convention

- **Severity:** low
- **Liveness:** live (the invariant is asserted every session)
- **Confidence:** high
- **Where:** `notebooks/18_w1_error_atlas.ipynb` (no `_build_18.py`, never existed in any commit),
  `notebooks/01..06_*.ipynb` (no `_build_0N.py`); claim at `CLAUDE.md:63-66`

`git log --all -- "notebooks/_build_18*"` is empty, so notebook 18 has never had a source. It was
added at commit `478293c` (2026-06-10), three weeks after `_build_07` established the convention
(`014f645`, 2026-05-22). It is not a trivial notebook: cell 2 defines the W1 cause→colour mapping and
writes `reports/figures/w1_synthesis_causes.png`, and cell 3 carries the W1 **decision memo**
("Seam-tile masking is REJECTED"; the reliability layer ships as a graded covariate) — the write-up
that PLAN_ModelUsability's W1 closure rests on. Editing it means hand-editing an `.ipynb`, which is
exactly the failure mode that produced invariants-1 and invariants-2.

- **Failure scenario:** a future session reads CLAUDE.md, assumes every notebook has a `_build`,
  tries `python notebooks/_build_18.py`, gets `No such file`, and either hand-edits the `.ipynb`
  (re-creating the invariants-2 pattern) or concludes the notebook is unmaintainable.
- **Evidence:**
  ```
  CLAUDE.md:63  - **Notebooks are generated:** edit the source-of-truth `notebooks/_build_NN.py`, not the `.ipynb`;
  CLAUDE.md:64    regenerate with `python notebooks/_build_NN.py` then `nbconvert --execute --inplace`.

  $ git log --oneline --all -- "notebooks/_build_18*"        # (empty)
  $ git log --oneline -- notebooks/18_w1_error_atlas.ipynb
  478293c W1 rungs 2-5 + synthesis: ladder complete, causes attributed
  ```
- **Self-refutation attempted:** (a) I checked whether `_build_18.py` was written then deleted —
  `git log --all` finds no such path in history. (b) I checked whether notebooks 01–06 pre-date the
  convention — they do, so they are grandfathered and I am not counting them as the defect, only as
  evidence the blanket claim is wrong. (c) I checked whether notebook 18 is a thin viewer that needs
  no source — it is not; it holds the W1 decision memo and a figure producer.
- **Fix:** either add `notebooks/_build_18.py` (the notebook is 4 cells) or qualify CLAUDE.md to
  "notebooks 07+ are generated; 01–06 and 18 are hand-written".

## Refuted by my own check

- **SP1 correction missing on the HiRISE-coverage-mask path.** `build_hirise_coverage_mask`
  (`src/ctx_retrieve.py:499`) goes through `read_full_footprint_decimated`, which applies
  `_corrected_source_crs` from the Stage-1 sidecar and *rebuilds* a cached GeoTIFF whose embedded CRS
  disagrees (`src/hirise_imagery.py:181-186`). The "sidecar missing → `None` → buggy JP2 CRS used
  silently" path is unreachable in production because `stage2_one_image` calls
  `det.load_reprojected(obs_id, cache_dir)` first (`src/ctx_retrieve.py:558`), which requires Stage 1
  to have run.
- **SP1 missing on the colour / Stage-7c path.** `src/colour.py:73-87` has its own
  `corrected_source_crs`, `read_color_window`'s docstring explicitly forbids passing `ds.crs`-derived
  bounds (`:139-142`), and `scripts/run_stage7c_features.py:127,149` transforms CTX→corrected CRS.
  Correct.
- **`urllib`/`requests` without `truststore`.** Every network-touching module either imports
  `truststore` directly (`src/hirise_imagery.py:47,64`; `src/pds_labels.py:44-46`) or triggers
  `pds_labels`' idempotent side effect (`src/ctx_retrieve.py:182`, `src/validation_retrieve.py:76`).
  All 14 network-touching scripts/probes do the same. Zero gaps.
- **Full-res HiRISE JP2 reads.** The only two JP2 read paths are
  `read_full_footprint_decimated` (`out_shape`) and `read_native_window` (`window=`, clipped, with a
  loud `ValueError` on a fully-outside window at `src/hirise_imagery.py:242-250`). No unwindowed,
  undecimated JP2 read exists in `src/`, `scripts/*.py`, or `notebooks/_build_*.py`.
- **CTX mosaic `.read()` without `window=`.** All 20 `.read(` sites in `src/` were read: the
  unwindowed ones (`src/features.py:112-114`, `src/coregister.py:351-353`, `src/fgates.py:195,304`,
  `src/striping.py:36`, `scripts/f_h4_themis.py:90`, `scripts/f_region_stagec.py:257`) operate on
  already-cropped local products (Stage-2 window tifs, 160 m/px abundance rasters, the
  `--match-mosaic` THEMIS crop), never on a GB-scale Murray tile. The two live mosaic paths
  (`src/mapping.py:64`, `src/ctx_retrieve.py:433`) are windowed.
- **Torch/OpenMP import order beyond R05.** An AST sweep over `scripts/*.py`,
  `notebooks/_build_*.py` and `scripts/probes/*.py` — resolving transitively which `src.*` modules
  pull an MKL library — found only R05's two (`sweep.py:31/38`, `run_modeling_slim.py:36/45`) among
  non-probe scripts. Three probes also violate it (`_diag_compression_mechanism.py:26/29`,
  `_diag_compression_variants_smoke.py:9/11`, `_fm_reliability_validation.py:29/35`) — listed here
  rather than filed, since probes are not a reported surface and R05 already names the pattern.
  `scripts/map_region.py:48` and `train_deployable_head.py` get it right, with the reason in a
  comment.
- **Hardcoded observation IDs in `src/` bypassing the manifest.** The five hits are all deny-lists or
  special-case flags, not the image list: `src/features.py:91` / `scripts/run_stage4.py:38`
  (`EXCLUDED_FROM_SWEEP`, duplicated but currently identical, and 4b/6a/6b all `import` the `src/`
  copy), `src/dataset.py:49` + `src/modeling/evaluate.py:50` (`EMPTY_TRUTH_OBS_ID`). Discovery is
  still `{ObsId}/*-mask-nms.shp` via `src/manifest.py:23,82`. The `EMPTY_TRUTH_OBS_ID` special case
  is also **not** a manifest bypass: `src/modeling/evaluate.py:334` ORs it with the general
  `np.unique(y_true).size < 2` test, so a new empty-truth image is caught without a code change.
- **Hardcoded Mars radii beyond `src/striping.py:118`.** `scripts/f_region_stageb.py:53` is already
  R21; `scripts/run_stage7c_features.py:78-93` and `scripts/probes/_stage7_feasibility.py:81` embed
  the target-CRS WKT as a literal instead of reading `config.yaml::target_crs`, but the literal is
  byte-equivalent to the config value (`Mars_2000_IAU_IAG, 3396190`), so there is no live error —
  and the geo-crs review already established that the sphere-vs-oblate distinction is a no-op here.
  `notebooks/_build_24.py` hardcodes `R = 3396190.0` in five metres→degrees conversions, correct for
  the clon_0 equirectangular mosaic (`x = R·λ`, standard parallel 0).
- **`set_crs` where `to_crs` was meant.** Five hits, all correct: `src/detections.py:86`
  (`allow_override=True`, the deliberate SP1 override), `src/striping.py:180-183` (assigns only when
  `g.crs is None`, else `to_crs`), `scripts/f_leg_b_frame_list.py:79` (same guard),
  and two Stage-7 probes mirroring `detections.py`.
- **Presence AUC leaking into a new reported surface.** Beyond R02/R25, the only remaining sites are
  historical: `notebooks/_build_10..13` (pre-retirement; DECISIONS.md:2750 retires the metric on
  2026-06-10) and one non-headline column in a 5-column table at `docs/modeling_results.md:1407` /
  `PROMOTION_QUEUE.md:513`. Nothing post-retirement reports it as a headline.
- **A `_build` edited but never regenerated (stale committed notebook).** Checked all 21 pairs by
  last-commit timestamp *and* by exact cell-source comparison against a fresh regeneration: no
  `_build` is newer than its `.ipynb`, and 19 of 21 regenerate byte-identically. Only 17 and 20
  drift, both filed above.
- **`target_crs: from_ctx_tile` sentinel unhandled.** It is handled
  (`src/ctx_retrieve.py:145-152`).
- **Hardcoded cohort size (`n == 38`) baked into `src/`.** Grep for `== 38 / == 39 / N_IMAGES /
  EXPECTED_` returns one docstring mention in `scripts/bank_calibration.py:3` and no code.
- **Hardcoded absolute paths in tracked Python.** Only two Sherlock activation lines inside
  docstrings (`scripts/f_leg_b_extract.py:18`, `f_pilot_extract_crop.py:8`), three PowerShell
  invocation examples in `_build_14/15/16`, and three broken markdown links to the user's private
  memory directory in `_build_12.py:288,705,860`. Cosmetic; no code path depends on them.

## Verified clean

- **Invariant 1 (per-image local-radius CRS)** — no radius or datum is hardcoded on any read path;
  the only literals are the CTX *target* CRS (a fixed, verified constant) and the two already-filed
  sites. Independently corroborates `geo-crs.md`'s finding of 6 distinct local radii in the cohort.
- **Invariant 3 (SP1)** — the override is applied on every path that needs it: the shapefile
  (`src/detections.py:80-90`), the RED JP2 (`src/hirise_imagery.py:179-200, 229-253`), the coverage
  mask and the coregistration reference (both via the decimated read), and the COLOR JP2
  (`src/colour.py:73-87`, `scripts/run_stage7c_features.py:127`). `_crs_equal`'s literal-SP1
  comparison correctly defeats pyproj's canonicalisation, so a pre-fix cache is rebuilt rather than
  silently reused.
- **Invariant 4 (windowed / decimated reads)** — see the two refutations above; every GB-scale
  access is windowed and every HiRISE full-footprint access is decimated.
- **Invariant 7 (manifest-driven)** — `src/manifest.py:23,74-90` globs `{ObsId}/*-mask-nms.shp` and
  refuses to guess on 0 or >1 matches; all `--all` runners iterate `df["ObsId"]`; no code path
  enumerates images. `map_region.py`'s `BLOCK_TILES` / `EXPANSION_TILES` are *map* tiles, not cohort
  images, and are internally consistent (26 / 19, matching their comments).
- **Invariant 9 (environment)** — `truststore` coverage is complete (above); `src/modeling/__init__.py`
  sets `KMP_DUPLICATE_LIB_OK` and the Windows `add_dll_directory` + `shm.dll` preload before any
  torch import, guarded by `os.name == "nt"` so Sherlock is unaffected.
- **Notebook regeneration mechanics** — all 21 `_build_NN.py` scripts run standalone with no repo
  side effects beyond writing their own `.ipynb`; none reads data or imports torch at module level
  (the `import numpy` / `import rasterio` lines that a naive grep surfaces are inside cell strings).

## Coverage note

**Read in full:** `src/config.py`, `src/hirise_imagery.py`, `src/mapping.py`,
`src/modeling/__init__.py`, `notebooks/_build_20.py` §7 and `_build_17.py` §§3-4, `config.yaml`
(lines 1-60), and the drifting cells of notebooks 17 and 20 in both forms.
**Read in relevant part:** `src/ctx_retrieve.py` (`build_hirise_coverage_mask`, `stage2_one_image`,
`_target_crs_wkt`), `src/colour.py:55-165`, `src/features.py:85-155`, `src/striping.py:95-200`,
`src/modeling/evaluate.py:315-405`, `src/detections.py:29-95`, `scripts/map_region.py:37-95`,
`scripts/run_stage2.py`, `scripts/run_stage7c_features.py:70-150`, `scripts/f_h4_buildprep.py`,
`scripts/f_region_stagec.py`, `scripts/f_leg_b_embed.py`, `notebooks/18_w1_error_atlas.ipynb`.
**Grepped only:** the remaining `scripts/probes/*` (229 files — sampled the 5 import-order hits and
the `truststore` hits, did not read them); `DECISIONS.md` by term (`hirise_decimation_mpp`,
`decimation`, `presence_auc`, `0.034`, `OR = 23`, `from_ctx_tile`, `SP1`, `nbconvert`).

**Measurements I ran** (read-only w.r.t. the repo; everything written went to the scratchpad): the
21-way `_build` → `.ipynb` regeneration and cell-source diff; an executed-cell census over all 28
notebooks; an AST import-order sweep over 350+ scripts/notebooks/tests with transitive `src.*` → MKL
resolution; `git log`/`git show` on `486af93`, `177f731`, `6e3b9f1`, `a003d33`, `478293c`. No
imagery, no network, no notebook execution, no training.

**Could NOT check:** (1) whether the *outputs* of the 19 byte-identical notebooks were actually
produced by the sources they now sit beside — an unexecuted-but-source-matching regeneration is
undetectable without re-running them, which the rules of engagement forbid, so my drift result is a
lower bound; (2) whether the numbers narrated in unexecuted notebooks 12 and 13 agree with what
their code would produce today (same reason) — the `_diag_*` probes committed alongside are the
place to verify; (3) the `.ipynb`-vs-`_build` question for notebooks 01–06 and 18, which have no
source to compare against; (4) whether any `scripts/probes/*.py` violates an invariant in a way that
fed a DECISIONS number — I only swept them by grep, per §6's note that all 229 remain unopened.
