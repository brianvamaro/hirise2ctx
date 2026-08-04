# Review area: other-scripts

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-01
- **Verification:** self-refuted (single-agent pass) + **second-pass line-level re-verification
  2026-08-01 at the same commit `da884c7`** by an independent reviewer agent. All six findings were
  re-confirmed against the cited lines; no finding was withdrawn, no new finding was added. What the
  second pass re-read directly: `src/dataset.py:390-401` vs `scripts/run_stage6a_repackage.py:46-60`
  (the `"seed"` key is present in the script copy and absent from the canonical function — **-1**
  confirmed); a repo-wide grep for `include_cnn` returning only `scripts/sweep.py:13` and `:133`
  (**-2** confirmed, no third reference anywhere); `scripts/train_gbm.py:74-85` vs
  `scripts/sweep.py:73-84` side by side, plus a grep showing `dataset_dir` occurs **zero** times in
  `train_gbm.py` and `train_binary.py` while `sweep.py:77` hashes it (**-3** confirmed);
  `scripts/build_vclaire_manifest.py:263-267`, where the ternary parses as
  `w_lon = (w_lon + 360.0 if w_lon < e_lon else w_lon)`, so for the seam-crossing case (w=359.95,
  e=0.05) the condition is False and the branch is a no-op → mean 180.0 (**-5** confirmed);
  `run_stage7c_features.py:64,241-246` sharing the `--out` default with `--only`, and
  `run_stage4.py:129` `return 0` against `run_stage1.py:75` `return 0 if not bad else 1` (**-6**
  confirmed); and a grep showing `meaningful_threshold` occurs **zero** times in `train_gbm.py`,
  `train_cnn.py` and `train_binary.py` against the `1e-2` defaults at
  `src/modeling/evaluate.py:319,567` and the caller-warning docstring at `:578-585` (**-4**
  confirmed). The second pass did not re-run the artifact arithmetic (the split-hash recomputation
  over the 14 split JSONs, the 220-snapshot enumeration, the 26-run threshold cross-check, the
  `hirise_40_vclaire.csv` inspection) — those numeric claims rest on the first pass alone.

Scope as briefed: `scripts/run_stage1..7*.py`, `train_gbm.py`, `train_cnn.py`, `train_binary.py`,
`train_deployable_head.py`, `sweep.py`, `sweep_binary.py`, `sweep_cnn.py`, `sweep_stage2.py`,
`sweep_within_image.py`, `run_modeling_slim.py`, `build_vclaire_manifest.py`, `parity_check.py`
(+ `bank_calibration.py` for the R05 follow-up question).

## Findings

### other-scripts-1 — The two repackage drivers' copy of the split hash has drifted from `src/dataset.py`, so 7 committed split JSONs carry a `split_hash` the canonical function cannot reproduce
- **Severity:** medium
- **Liveness:** live-shipped (the `dataset_v2/` production splits `loio_nfold_ctx_illum`, `loio_nfold_nbr_s5`)
- **Confidence:** high (measured on every committed split JSON)
- **Where:** `scripts/run_stage6a_repackage.py:46-60` (+ call at `:122`),
  `scripts/run_stage6b_repackage.py:46-55` (+ call at `:107`) vs `src/dataset.py:390-401`

Both repackage scripts inline a private helper and say in the docstring that it "Mirrors
`src.dataset._split_metadata_hash`". It does not: the copies add `"seed"` to the hashed key tuple.
`DECISIONS.md:705` states the intent explicitly — "stable `split_hash` **over the assignment**" — i.e.
the hash identifies the fold assignment, not the RNG input that produced it. So every scheme built by
`build_split` is hashed one way and every scheme produced by a repackage driver is hashed another,
under a name that asserts they are the same function.

- **Failure scenario:** exactly the consumer-side guard **R04** prescribes ("add a guard in
  `loaders.load_metadata`/`load_fold` comparing the packaged `split_hash` to
  `dataset*/splits/{scheme}.json`'s and raising on mismatch"). Any such guard that re-derives the hash
  with the canonical `_split_metadata_hash` will report **7 of 14** committed split JSONs as corrupt —
  including the two production `dataset_v2` schemes that back every Stage-6a/6b LOIO comparison —
  while the actually-stale case it exists to catch stays invisible. Equally, a future session that
  "de-duplicates" the copies back onto the `src/` function silently invalidates those 7 recorded
  hashes.
- **Evidence:**
  ```
  src/dataset.py:392-396
      keys = (
          "name", "kind", "n_folds", "stratification", "manifest_obs_ids", "folds",
          # within-image specific:
          "n_folds_per_image", "buffer_tiles", "excluded_obs_ids",
      )

  scripts/run_stage6a_repackage.py:49-56
      """Mirrors src.dataset._split_metadata_hash (private; duplicated here so the
      repackage script doesn't depend on a private symbol).
      """
      keys = (
          "name", "kind", "n_folds", "stratification", "seed", "manifest_obs_ids", "folds",
          "n_folds_per_image", "buffer_tiles", "excluded_obs_ids",
      )
  ```
  Recomputed both formulas over every `dataset*/splits/*.json` (read-only, committed artifacts).
  The split is perfectly 1:1 with `repackaged_from_scheme`:

  | split JSON | stored hash reproduced by `_split_metadata_hash` | by the script copy |
  |---|---|---|
  | `dataset/{loio_3fold_balanced, loio_9fold, within_image_4fold}` | ✅ | ❌ |
  | `dataset_v2/{loio_nfold, within_image_4fold}` | ✅ | ❌ |
  | `dataset_v2_dev/{loio_nfold, within_image_4fold}` | ✅ | ❌ |
  | **`dataset_v2/loio_nfold_ctx_illum`** | ❌ | ✅ |
  | **`dataset_v2/loio_nfold_nbr_s5`** | ❌ | ✅ |
  | `dataset_v2_dev/loio_nfold_ctx_illum` | ❌ | ✅ |
  | `dataset_v2_dev/within_image_4fold_ctx_illum` | ❌ | ✅ |
  | `dataset_v2_dev/within_image_4fold_nbr` | ❌ | ✅ |
  | `dataset_v2_dev/within_image_4fold_nbr_max` | ❌ | ✅ |
  | `dataset_v2_dev/within_image_4fold_nbr_s5` | ❌ | ✅ |

- **Self-refutation attempted:** (a) I checked whether the difference is inert because `name` is also
  hashed and always differs — it is not inert: the *stored* value is what a re-derivation is compared
  against, and the two functions disagree on the same input dict. (b) I checked for an existing
  consumer: `notebooks/_build_09.py:298` is the only one, and it compares
  `meta["split_hash"] == pkg["split_hash"]` — which `src/dataset.py:711` copies verbatim from
  `metadata`, so that QA line is a tautology and cannot detect this (or anything). (c) I grepped
  `DECISIONS.md` for `split_hash` / `run_stage6*_repackage`: one hit (`:706`) and it states the
  assignment-only intent, i.e. the seed inclusion is a deviation, not a recorded decision. (d) Tests
  (`tests/test_splits.py:192,266`, `tests/test_within_image_split.py:264,360`) only assert
  self-consistency of the `src/` function; nothing exercises the script copies.
- **Fix:** export `_split_metadata_hash` (or add a thin public `split_metadata_hash`) from
  `src/dataset.py` and call it from both repackage scripts; then re-emit the 7 affected split JSONs
  (a metadata-only rewrite — the folds are unchanged) or record in DECISIONS that those 7 hashes
  predate the unification.

### other-scripts-2 — `sweep.py --include-cnn` is accepted, documented and never read, so the CNN arm silently does not run and notebook 10 reports a stale CNN row instead
- **Severity:** medium
- **Liveness:** live-shipped (the v1/v2 GBM sweep driver documented at `README.md:227,232`)
- **Confidence:** high
- **Where:** `scripts/sweep.py:133-134` (declaration), `scripts/sweep.py:13` (docstring),
  `scripts/sweep.py:126-207` (`main`, which never references `args.include_cnn`);
  downstream consumer `notebooks/_build_10.py:415-427`

`--include-cnn` is declared with `action="store_true"` and advertised in the module docstring as
"also CNN at S32 and S64 (delegates to `scripts/train_cnn.py`)", but `main()` never reads it.
Repo-wide grep for `include_cnn` returns exactly two hits — the `add_argument` and the docstring
line — so the flag is a no-op that argparse accepts without complaint and the script exits 0.

- **Failure scenario:** an operator runs `python scripts/sweep.py --include-cnn --dataset-dir
  dataset_v2 --scheme loio_nfold`, sees a clean run and a full aggregate table, and concludes the CNN
  baseline was re-evaluated on v2. It was not. Notebook 10's CNN block does **not** read the sweep
  output at all — it globs `models/cnn_log1p_huber_S{32,64}/*` and takes `runs[-1]` with no snapshot
  check — so the model-comparison table still shows a CNN row, sourced from whatever CNN artifact
  happens to be on disk (today: the v1 `loio_9fold` runs at
  `models/cnn_log1p_huber_S{32,64}/*/scale_S{32,64}_P{32,64}`, whose `snapshot.json` has **no**
  `dataset_dir` key). The GBM rows are v2, the CNN row is v1, and nothing in the notebook or the
  sweep output says so.
- **Evidence:**
  ```
  scripts/sweep.py:13
      python scripts/sweep.py --include-cnn        # also CNN at S32 and S64

  scripts/sweep.py:133-134
      ap.add_argument("--include-cnn", action="store_true",
                      help="Also run CNN at S32 and S64 (delegates to scripts/train_cnn.py).")

  notebooks/_build_10.py:418-424
      runs = sorted((MODELS_ROOT / name).glob('*'))
      if not runs:
          continue
      tile_size = patch_size
      scale_dirs = sorted(runs[-1].glob(f'scale_S{tile_size}_P{patch_size}'))
  ```
- **Self-refutation attempted:** I grepped `scripts/`, `notebooks/`, `README.md`, `SHERLOCK_RUN.md`
  and `docs/*.md` for `include_cnn`/`include-cnn` — only the two `sweep.py` lines. I also checked
  whether some wrapper shells out to `train_cnn.py` after `sweep.py` (there is no Makefile, no
  `.github/`, and the `.sbatch` files under `scripts/` do not call `sweep.py`). And I checked whether
  the flag is *harmlessly* vestigial: it is not, because the docstring is the script's `--help` text
  (`ArgumentParser(description=__doc__)`), so `--help` actively advertises the behaviour.
- **Fix:** either implement the delegation (call `train_cnn.py`'s `main` for `(P=32, scale 2)` and
  `(P=64, scale 3)` with the run's `--dataset-dir`/`--scheme`) or delete the flag and the docstring
  line. Separately, make `_build_10`'s CNN cell use `_build_11.py:385-395`'s snapshot-matching
  selector rather than `runs[-1]`.

### other-scripts-3 — `train_gbm.py` / `train_binary.py` cannot select a dataset root, omit it from their provenance, and write into the same `*/scale_S{n}` glob namespace the sweeps use — while claiming to be interchangeable with them
- **Severity:** medium
- **Liveness:** live-shipped (single-run drivers; `DECISIONS.md:1018` lists `train_gbm.py` as the
  single-variant LOIO driver)
- **Confidence:** high on the mechanism; the contaminating run has not happened yet (latent)
- **Where:** `scripts/train_gbm.py:7-8` (claim), `:74-85` (snapshot + out_dir), `:94-100`, `:109`;
  `scripts/train_binary.py:13-15` (claim), `:80-98`, `:103-110`, `:119`; contrast
  `scripts/sweep.py:73-84` and `scripts/sweep_within_image.py:71-82`

Three coupled defects. (1) The docstrings assert "same params + target + scale produce the same hash,
so train_binary and sweep_binary write to identical paths and are interchangeable". They cannot: the
sweeps put `"dataset_dir"` into the hashed snapshot and the single-run scripts do not, so the SHA-256
inputs differ for every configuration; the defaults also differ (`n_estimators` 500 vs 400,
`early_stopping_rounds` 50 vs 40). (2) Neither script has a `--dataset-dir` flag and both call
`run_loio(...)` / `iter_loio_folds(...)` without one, so `src/modeling/loaders.py:26`'s
`DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"` (v1) is always used — but `--scheme` is free text, and
`within_image_4fold` exists in `dataset/`, `dataset_v2/` **and** `dataset_v2_dev/`. (3) The output
leaf is `scale_S{tile}` (`train_gbm.py:85`) / `scale_S{tile}_t{target}` (`train_binary.py:95-98`)
with no scheme marker, whereas `sweep_within_image.py:82,123-126` deliberately appends `_within` for
exactly this reason.

- **Failure scenario:** working on v2, an operator re-runs one cell with
  `python scripts/train_gbm.py lightgbm_two_stage --scale-idx 3 --scheme within_image_4fold`. It
  silently trains on the **v1 9-image** packaged split (no error — the path exists), and writes
  `models/lightgbm_two_stage/<new hash>/scale_S64/`. `notebooks/_build_10.py:255-256` then resolves
  the "LOIO two-stage @ S=64" panel with
  `sorted((MODELS_ROOT/variant).glob('*/scale_S64'), key=mtime)[-1]`, which is now that directory —
  so the notebook plots v1 *within-image* predictions in a v2 *LOIO* panel. Nothing on disk
  contradicts it: the artifact's `snapshot.json` has no `dataset_dir` key at all (verified: 35 of 220
  committed `models/*/*/*/snapshot.json` lack it, all v1-era). Conversely `_build_11.py:387-393`'s
  "robust artifact selector" requires `s.get('dataset_dir') == want`, so it can never match a
  `train_gbm.py`/`train_binary.py` artifact and silently returns `None`.
- **Evidence:**
  ```
  scripts/train_gbm.py:7-8
      Same params -> same config_hash as `scripts/train_gbm.py`, so the two scripts   [sweep.py:7-8]
      write to identical paths and are interchangeable.

  scripts/train_gbm.py:74-85              scripts/sweep.py:73-84
      snapshot = {                            snapshot = {
          "variant": args.variant,                "variant": variant,
          "target_col": args.target_col,          "target_col": TARGET_COL,
          "scheme": args.scheme,                  "scheme": scheme,
          #  <-- no dataset_dir                   "dataset_dir": dataset_dir or "dataset",
          "scale_idx": args.scale_idx,            "scale_idx": scale_idx,
          ...
      out_dir = MODELS_ROOT / args.variant / cfg_hash / f"scale_S{TILE_SIZE_FOR_SCALE[...]}"

  scripts/sweep_within_image.py:82
      out_dir = MODELS_ROOT / "lightgbm_two_stage" / cfg_hash / f"scale_S{tile_size}_within"
  ```
- **Self-refutation attempted:** (a) I checked whether the v2 scheme name saves it — it does for
  `loio_nfold` (`dataset/packaged/loio_nfold` does not exist, so it fails loudly), but **not** for
  `within_image_4fold`, which is present in all three roots (`ls dataset*/packaged`). (b) I checked
  whether README/SHERLOCK document these scripts at all — they do not (only the four `sweep*` entries
  at `README.md:227-234`), which lowers the likelihood but leaves the docstrings as the only
  guidance, and they are wrong. (c) I checked whether any committed artifact already shows the
  collision: no — every existing `scale_S{n}` leaf under `lightgbm_two_stage` is `loio_9fold`, and
  every within-image one carries `_within`. So this is latent, not realised. (d) `DECISIONS.md` has
  no entry on interchangeability or on omitting `dataset_dir` from the hash.
- **Fix:** add `--dataset-dir` to both scripts and thread it into `run_loio`/`iter_loio_folds` **and**
  into the hashed snapshot (matching `sweep.py:78`); append a scheme marker to the leaf when the
  scheme is not the LOIO one (or simply put `scheme` in the leaf); and correct or delete the
  "interchangeable" sentence in both docstrings — with the defaults aligned (400/40) if
  interchangeability is actually wanted.

### other-scripts-4 — `--target-col` on `train_gbm.py` and `train_cnn.py` silently degrades the mandated rich/poor metrics to presence metrics
- **Severity:** low (latent: no committed artifact was produced this way)
- **Liveness:** live-shipped scripts, dormant flag
- **Confidence:** high
- **Where:** `scripts/train_gbm.py:57` (`--target-col`) vs `:94-100` (`run_loio` call);
  `scripts/train_cnn.py:87` vs `:159` (`per_fold_metrics` call); defaults at
  `src/modeling/evaluate.py:567` and `:319`

`run_loio`'s own docstring (`src/modeling/evaluate.py:578-585`) warns that `meaningful_threshold`
"is TARGET-DEPENDENT and must be set by the caller for any target other than `fractional_area`:
… applied to a `boulder_count` target it would collapse to `count > 0.01` == presence (count >= 1),
which is degenerate and not the scientific question." Both scripts expose `--target-col` as a
first-class CLI flag and neither passes `meaningful_threshold`, so `meaningful_auc`, `pr_auc`,
`normalised_lift` and `precision_at_top_{1,5,10}pct` would be computed at 1e-2 — presence metrics
under the mandated key names (invariant 8). Neither script records the threshold in `snapshot.json`
either, so the artifact cannot be distinguished after the fact.

- **Failure scenario:** `python scripts/train_gbm.py lightgbm_two_stage --scale-idx 3 --target-col
  boulder_count` writes a `metrics.json` whose `meaningful_auc_mean` is presence AUC and a
  `snapshot.json` with `"target_col": "boulder_count"` and no threshold field. Every existing
  count-target artifact on disk uses the correct 50.0 (checked), so a reader comparing the new run to
  them compares two different statistics.
- **Evidence:**
  ```
  scripts/train_gbm.py:57
      ap.add_argument("--target-col", default="fractional_area")
  scripts/train_gbm.py:94-100
      result = run_loio(
          factory,
          target_col=args.target_col,
          scheme=args.scheme,
          scale_idx=args.scale_idx,
          snapshot=snapshot,
      )                                  # no meaningful_threshold=

  scripts/train_cnn.py:159
      m = per_fold_metrics(y_test, y_pred, held_out_obs_ids=fold.held_out_obs_ids)
  ```
- **Self-refutation attempted:** I scanned all 220 committed `models/*/*/*/snapshot.json` for a
  non-`fractional_area` `target_col` (26 hits) and cross-checked each against its
  `metrics.json`'s `per_fold[0]["meaningful_threshold"]`. All 26 are correct (50.0 for counts, 256/1024
  for areas, log-space equivalents for the log targets) — including the six `models/fang_tier2/
  tier2_*_boulder_count_S32/` runs whose *snapshots* omit the key but whose *metrics* record 50.0.
  So nothing banked is wrong; the defect is a live footgun on two documented flags. It is the same
  class as **R35 `modeling-heads-3`** (which scoped five count-target *probes*); I file it because
  these are two top-level scripts in this area with user-facing `--target-col` flags, which that
  finding does not cover.
- **Fix:** in both scripts, derive the threshold from the target (the probes' `_meaningful_threshold`
  helper) or require an explicit `--meaningful-threshold` when `--target-col != fractional_area`, and
  write it into `snapshot.json`.

### other-scripts-5 — `build_vclaire_manifest.py`'s antimeridian guard is a no-op in the only case it exists for, putting the derived image centre ~180° from truth
- **Severity:** low (latent; never fired for the 39-row cohort)
- **Liveness:** live-shipped manifest builder — invariant-7 ("adding a manifest row must flow
  end-to-end") hazard
- **Confidence:** high
- **Where:** `scripts/build_vclaire_manifest.py:263-267`

The PDS footprint gives `EASTERNMOST_LONGITUDE` / `WESTERNMOST_LONGITUDE`. For a swath straddling
lon 0/360 the westernmost value is the *larger* number (e.g. w=359.95, e=0.05), which is precisely
when `abs(e_lon - w_lon) > 180` fires. But the correction only adds 360 to `w_lon` when
`w_lon < e_lon`, which is false in that case — so the branch executes and changes nothing, and the
mean of 0.05 and 359.95 is **180.0**, the antipode of the truth.

- **Failure scenario:** a future manifest row for an observation crossing the prime meridian gets
  `CenterLon_360 ≈ 180`, `ctx_tile_name` returns `E180_N{lat}` (a real Murray tile), Stage 2 fetches
  it, and the Stage-1 detection bbox lies ~10 000 km outside that tile's extent — the exact
  overhanging-window path R31 shows is silently mis-georeferenced rather than raised. The manifest
  itself gives no warning: `CenterSource` still reads `pds_footprint`.
- **Evidence:**
  ```
  scripts/build_vclaire_manifest.py:263-267
      e_lon, w_lon = fp["east_lon_deg"] % 360.0, fp["west_lon_deg"] % 360.0
      # Narrow HiRISE swaths don't wrap the antimeridian; guard anyway.
      if abs(e_lon - w_lon) > 180.0:
          e_lon, w_lon = e_lon, w_lon + 360.0 if w_lon < e_lon else w_lon
      center_lon_360 = ((e_lon + w_lon) / 2.0) % 360.0
  ```
- **Self-refutation attempted:** I checked the committed `hirise_40_vclaire.csv`: all 39 rows have
  `CenterSource == pds_footprint` (so the fetch path, not the fallback, produced every centre) and
  the closest approach to the seam is `CenterLon_360 = 0.747` (`E000_N40`) — ~44 km, against a
  ~6 km HiRISE swath (~0.1°). So the branch has never been taken and no existing tile assignment is
  wrong. I also verified the comment's own premise: the guard is aimed at the **prime meridian** in
  the `%360` frame, not the antimeridian the comment names, so the comment is also misleading. I
  confirmed the `_get`/`%` precedence at `:278` is *not* a bug (`corner_lon % 360.0` binds first, as
  intended).
- **Fix:** `if e_lon < w_lon: e_lon += 360.0` before the mean (then the existing `% 360.0` closes it),
  and drop the dead conditional.

### other-scripts-6 — `run_stage7c_features.py --only` overwrites the full-cohort colour features with the sanity subset, and four `--all` drivers exit 0 after failing images
- **Severity:** low
- **Liveness:** live module, PARKED Stage-7 programme (`--only`); live-shipped (`--all` exit codes)
- **Confidence:** high
- **Where:** `scripts/run_stage7c_features.py:64` (`DEFAULT_OUT`), `:241-242` (`--only`), `:279`;
  `scripts/run_stage3.py:117`, `scripts/run_stage4.py:128`, `scripts/run_stage4b.py:118`,
  `scripts/run_stage7a_fetch.py:179`

`--only` is documented as a "sanity-run subset" but shares `--out`'s default, so a two-image sanity
run replaces the 37-image `dataset_v2/features_colour.parquet` that
`scripts/run_stage7d_pooled.py:46` reads by default. Stage 7d then runs its "pooled cross-image" test
over two images and writes `dataset_v2/stage7d_pooled.parquet` with the same filename and no
`n_images` field in the output — the only signal is a console line
(`run_stage7d_pooled.py:74`) nobody has to read.

Separately, and as the other half of **R04**: `run_stage3.py --all`, `run_stage4.py --all`,
`run_stage4b.py --all` and `run_stage7a_fetch.py` all count failures, print them, and then
`return 0`. `run_stage1.py:75` already does the right thing (`return 0 if not bad else 1`), so the
inconsistency is within the same family of drivers.

- **Failure scenario:** operator sanity-checks Stage 7c on `--only ESP_042964_2160`, then re-runs
  Stage 7d without re-running the full Stage 7c. `docs/compositional.md`'s pooled effect sizes are
  regenerated over one image and land in the same parquet path, with the previous full-cohort file
  gone. For the exit codes: a Stage-4 run in which 5 of 38 images fail reports "Skipped: …" and exits
  0, so any wrapper (or a future Slurm chain) treats the stage as complete.
- **Evidence:**
  ```
  scripts/run_stage7c_features.py:64
      DEFAULT_OUT = Path("dataset_v2/features_colour.parquet")
  scripts/run_stage7c_features.py:241-246
      ap.add_argument("--only", nargs="*", default=None,
                      help="Process only these obs_ids (sanity-run subset).")
      ...
      ap.add_argument("--out", type=Path, default=DEFAULT_OUT, ...)

  scripts/run_stage4.py:117-128
      print(f"\nSolved {len(solved)} / {len(rows)}; skipped {len(skipped)}", flush=True)
      ...
      return 0
  ```
- **Self-refutation attempted:** I checked whether Stage 7d would notice: `s7d.load_joined` does an
  inner join and `run_stage7d_pooled.py:95-97` prints `eligible images for {rule}`, but the emitted
  parquet has no image-count column and `s7d.eligible_images(min_per_class=5)` would not raise on a
  1–2 image pool. I also checked whether `--only` is used anywhere automated — it is not (no caller
  in `scripts/`, `notebooks/` or the `.sbatch` files), so this requires a human mistake. On the exit
  codes I confirmed there is no Makefile, no `.github/`, and no in-repo caller reading these codes
  today — the same "impact today: nil" finding R04 records for Stage 5.
- **Fix:** make `--only` require an explicit `--out` (or default it to
  `features_colour_subset_{n}.parquet`); and have the three `--all` drivers return
  `0 if not skipped else 1`, matching `run_stage1.py:75`.

## Refuted by my own check

- **`models/fang_tier2/tier2_*_boulder_count_S32/`'s `meaningful_auc` is presence AUC.** Their
  `snapshot.json` has no `meaningful_threshold` key, which looked like the invariant-8 failure. Read
  the metrics: `per_fold[0]["meaningful_threshold"] == 50.0` in all six. The threshold was passed as a
  kwarg and only the snapshot omits it. Consistent with `modeling-heads.md`'s own audit.
- **`sweep_binary.py --skip-fa-gt-1e-2-s8` is misspelled against its dest.** argparse maps
  `--skip-fa-gt-1e-2-s8` → `skip_fa_gt_1e_2_s8`, which is exactly what `:173` reads. Correct.
- **`run_stage6a.py`'s `--stats` / `--stencil-size` do not enter the output path, so two Stage-6a
  configurations collide onto `features_nbr/`.** Real in principle, but the `--output-suffix` escape
  (`:129-133`) was actually used: `dataset_v2_dev/` holds `features_nbr`, `features_nbr_max` and
  `features_nbr_s5` side by side, and `run_stage6a_repackage.py --features-suffix` propagates the
  suffix into both the scheme name and the packaged dir, so the downstream `config_hash` is distinct
  per variant. Mitigated in practice; not filed.
- **`sweep.py:191,203` and `sweep_within_image.py:270-271` print/persist `presence_auc_mean` as the
  headline AUC.** True, and `aggregate.parquet` carries the column — but this is inside R02's stated
  blast radius ("any consumer … that reads `presence_auc_mean` from a run's artifact"), so not
  re-filed.
- **`train_cnn.py` / `sweep_cnn.py` hand-roll `predictions.parquet` + `metrics.json` +
  `snapshot.json` instead of calling `write_run_artifacts`.** Duplication, but I diffed the three
  writes against `src/modeling/evaluate.py:722-753` and they are structurally identical (same three
  filenames, same `{"per_fold", "aggregate"}` shape, same `default=float`). No drift, so no finding.
- **`sweep.py` / `sweep_binary.py` / `train_binary.py`'s per-fold refit pass could use a different
  inner-validation image than the harness.** It does not: `unique_train[fold.fold_idx %
  unique_train.size]` in the scripts is character-for-character `run_loio`'s rule at
  `src/modeling/evaluate.py:615-619`.
- **`run_modeling_slim.py:136` fits without `eval_set`/`groups` while `LGBMParams.early_stopping_rounds
  = 50`.** Deliberate and harmless: with no eval set LightGBM simply trains all 500 trees; the slim
  variant is defined as "default hyperparameters, no early stopping" and the docstring says so.
- **`run_modeling_slim.py:85` uses `fa >= 1e-2` where the project standard is `fa > 1e-2`.**
  `fractional_area` is a continuous area ratio; exact equality with 1e-2 has measure zero. Immaterial.
- **`build_vclaire_manifest.py` does a network fetch with no `truststore.inject_into_ssl()`.** The
  injection lives in `src/pds_labels.py:32-49` (module-import time), which is the only network path
  the script uses. Correct.
- **`run_stage6a_repackage.py:127` writes the new split JSON before `package_split` runs**, so a crash
  leaves a split JSON pointing at a non-existent packaged dir. Real but strictly weaker than R04's
  staleness finding and with the same nil impact today; folded into the coverage note rather than
  filed.

## Verified clean

- **R05 follow-up answered:** `scripts/bank_calibration.py` does **not** touch torch. Its only
  first-party import is `src.calibration`, whose imports are `json`, `pathlib`, `numpy`, `pandas`
  (`src/calibration.py:27-31`). No LightGBM either. The numpy-before-`src.modeling` order at
  `:19-20` is benign, as R05 suspected.
- **No other script in this area violates the torch/OpenMP import order.** `sweep_binary.py:32`,
  `sweep_within_image.py:38`, `sweep_cnn.py:39`, `train_binary.py:29`, `train_cnn.py:36`,
  `train_deployable_head.py:34` and `parity_check.py:30` all `import src.modeling` before numpy;
  `train_gbm.py:28`'s first third-party import is `src.modeling.evaluate`, which triggers the package
  init. `run_stage1..7*`, `sweep_stage2.py` and `build_vclaire_manifest.py` never load torch.
- **No network call in this area lacks `truststore`.** `run_stage7a_audit.py:28-30` and
  `run_stage7a_fetch.py:27-29` inject it at import; `build_vclaire_manifest.py` inherits it via
  `src/pds_labels.py`. `run_stage7a_fetch.py:64-100`'s retry/backoff, size verification against the
  HEAD-declared `Content-Length`, and `.partial` → `os`-level `replace` are all correct (this is the
  atomic-write discipline R14 finds missing in `map_region.py`).
- **`snapshot_params` covers the whole hyperparameter set** (`src/modeling/gbm.py:716-722` →
  `asdict(params)`), so no two `LGBMParams` can collide onto one `config_hash` through the model
  block.
- **`sweep_cnn.py:62-78`'s inner-validation pool is whole-image and never touches the held-out fold**
  (`np.isin(train_codes, …)` over `fold.groups_train` only), and its rotation
  `(fold_idx * k + i) % n` is deterministic and collision-free for `k ≤ n-1`.
- **`sweep_within_image.py:82,123-126` correctly namespaces its artifacts** (`_within`,
  `_within_t{target}`), which is why no within-image artifact currently collides with a LOIO one.
- **`run_modeling_slim.py:125` selects slim features by name** (`fold.feature_names.index(f)`), not by
  position, so a column reordering upstream cannot silently permute the 5-feature matrix.
- **`run_stage6b.py:166-173`'s `_bbox_from_window`** uses `ymin = ymax + h * t.e` with the negative
  `e`, i.e. the sign convention is right.
- **The `--all` drivers are manifest/inventory-driven, not hardcoded** (`run_stage1..4b` iterate
  `M.load_manifest(cfg.manifest_path)`; `run_stage6a/6b` glob `{dataset}/features/*.parquet`). The
  only hardcoded ObsIds are the two documented exclusions
  (`run_stage4.py:31-38` / `src.features.EXCLUDED_FROM_SWEEP`), kept in sync with a comment saying so.

## Coverage note

**Read in full:** `run_stage1.py`, `run_stage2.py`, `run_stage3.py`, `run_stage4.py`,
`run_stage4b.py`, `run_stage5.py`, `run_stage6a.py`, `run_stage6a_repackage.py`, `run_stage6b.py`,
`run_stage6b_repackage.py`, `run_stage7a_audit.py`, `run_stage7a_fetch.py`,
`run_stage7c_features.py`, `run_stage7d_pooled.py`, `sweep.py`, `sweep_binary.py`, `sweep_cnn.py`,
`sweep_stage2.py`, `sweep_within_image.py`, `train_gbm.py`, `train_binary.py`, `train_cnn.py`,
`train_deployable_head.py`, `run_modeling_slim.py`, `build_vclaire_manifest.py`, `parity_check.py`,
`bank_calibration.py`.

**Read in part (as callees, to check the scripts against them):** `src/dataset.py`
(`_split_metadata_hash`, `build_split`, `package_split`), `src/modeling/evaluate.py` (`run_loio`,
`per_fold_metrics`, `write_run_artifacts`), `src/modeling/loaders.py` (`package_dir`, `load_fold`,
`iter_loio_folds`), `src/modeling/gbm.py` (`LGBMParams`, `snapshot_params`), `src/calibration.py`
(imports only), `src/pds_labels.py` (TLS + `image_footprint`), `notebooks/_build_09.py:296-300`,
`_build_10.py:250-260,310-320,415-430,590-606`, `_build_11.py:378-400`.

**Executed (read-only, on committed artifacts):** (1) recomputed both split-hash formulas over all 14
`dataset*/splits/*.json`; (2) enumerated `dataset_dir` / `scheme` / `variant` / `target_id` across all
220 `models/*/*/*/snapshot.json`; (3) cross-checked `snapshot["target_col"]` against
`metrics["per_fold"][0]["meaningful_threshold"]` for the 26 non-`fractional_area` runs; (4) inspected
`hirise_40_vclaire.csv`'s `CenterSource` / `CenterLon_360` / `MapPixel_mpp` columns.

**Not covered / could not check:**
- `scripts/parity_check.py`'s substantive question (what the gate pins and what
  `resolve_model_dir`'s `hits[-1]` glob can silently select when two heads exist under
  `models/deployable/`) is assigned to the `fm-embeddings` brief and partly filed as R35
  `fm-embeddings-4`; I did not duplicate it. I note only that `parity_ref.npz` records the window but
  **not** the model dir / `model_hash` / calibration file it was produced with, so a mismatched head
  produces a FAIL rather than a silent pass — safe, but uninformative.
- The `f_*` / `striping_*` / `map_*` scripts are other areas' (they are covered in §4/§4b/§4c).
- I did not run any script; all claims are static reads plus arithmetic on committed artifacts.
- `scripts/probes/` (229 files) remains unopened by anyone, including me — several `train_*`/`sweep_*`
  behaviours are mirrored there and could have drifted independently.
- `run_stage6a_repackage.py:127`'s write-before-package ordering (orphan split JSON on a mid-run
  crash) is real but unfiled; likewise `run_stage6a.py`'s sidecar records no `config_hash`, so a
  Stage-6a output cannot be tied to the config that produced its inputs.
- I did not verify that `docs/modeling_slim.md`'s reported numbers match
  `dataset_v2/modeling_slim_summary.parquet` — that belongs to `docs-consistency`.
