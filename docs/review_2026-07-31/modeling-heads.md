# Review area: modeling-heads

- **Reviewed at commit:** da884c7
- **Date:** 2026-07-31
- **Verification:** self-refuted (single-agent pass; not independently verified)

## Findings

### modeling-heads-1 — `LightGBMClassification` puts `auc` in the early-stopping monitor, the exact opposite of its own docstring and PLAN_Stage5b §4; the banked Tier-1 reference classifier consequently ships **1-tree** boosters on 11 of 38 LOIO folds

- **Severity:** high
- **Liveness:** live-shipped (this is the Tier-1 reference head the whole FM/Fang programme was benchmarked against: `scripts/probes/_w2_fang_heads.py:293`, `_w2_fang_probe.py:236`, `_fm_freeze_window.py:255`, plus `scripts/sweep_binary.py` / `train_binary.py` / `sweep_within_image.py`)
- **Confidence:** high (mechanism confirmed against the installed LightGBM 4.6.0 source + reproduced synthetically; impact measured on the banked boosters)
- **Where:** `src/modeling/gbm.py:419` (`metric` list), `:436` (`lgb.early_stopping(...)` with default `first_metric_only=False`), contradicting `src/modeling/gbm.py:381-384` and `PLAN_Stage5b.md:136-138`

`fit` sets `kw["metric"] = ["binary_logloss", "auc"]` and then attaches `lgb.early_stopping(rounds)`
with the default `first_metric_only=False`. LightGBM's callback loops over **every** (dataset, metric)
pair and raises `EarlyStopException(self.best_iter[i], ...)` for the **first** pair that has not
improved for `stopping_rounds` — so valid-set AUC co-governs both the stop *and* the selected
`best_iteration`. The class docstring and the plan both state the opposite ("Early-stopping metric is
`binary_logloss`, not AUC — AUC is non-decomposable and noisier as an early-stop signal on small
inner-validation sets"). Because `model_to_string()` truncates to `best_iteration`, whichever metric
stalls first physically determines how many trees are saved and used for every prediction.

- **Failure scenario:** measured on the two banked `fa_gt_1e-2` Tier-1 reference runs (`n_estimators=400`, `early_stopping_rounds=40`), counting `^Tree=` in each `fold_*/classifier.txt`:
  * `models/lightgbm_classification/99de85c1ad2a72e6/scale_S64_tfa_gt_1e-2` — 38 folds, trees `min=1 p25=1 median=13 p75=40 max=400`; **42 % of folds ship ≤5 trees and 11 folds ship exactly 1 tree**. Mean per-fold AUC 0.649 on the ≤10-tree folds vs 0.661 on the rest.
  * `.../2d046f48c722f0a5/scale_S32_tfa_gt_1e-2` — trees `min=1 p25=1 median=7 max=249`; 42 % ≤5 trees, 13 folds with 1 tree. Mean AUC 0.643 (≤10 trees) vs **0.683** (>10 trees).

  A 1-tree booster at `num_leaves=63` is a single tree — effectively a constant-ish predictor. The
  Tier-1 baseline AUC that the frozen recipe's 0.7865 median AUC was declared to beat is therefore an
  average over a cohort in which ~40 % of the folds were truncated to a handful of trees by a noisy
  metric the plan told the code not to use.
- **Evidence:**
  ```
  src/modeling/gbm.py:381-384   (docstring claim)
      Early-stopping metric is `binary_logloss`, not AUC -- AUC is
      non-decomposable and noisier as an early-stop signal on small
      inner-validation sets (PLAN_Stage5b.md §4).

  src/modeling/gbm.py:419
      kw["metric"] = ["binary_logloss", "auc"]
  src/modeling/gbm.py:436
      callbacks.append(lgb.early_stopping(self.params.early_stopping_rounds, verbose=False))

  PLAN_Stage5b.md:136-138
      The classifier's early-stopping metric is `binary_logloss`, not
      `auc`, because AUC is non-decomposable and produces noisier early-stopping
      behaviour on small inner-validation sets.

  lightgbm/callback.py:414-437 (installed 4.6.0) — loops over ALL eval results:
      for i in range(len(env.evaluation_result_list)):
          ...
          if self.first_metric_only and self.first_metric != metric_name:
              continue
          if self._is_train_set(...):  continue
          elif env.iteration - self.best_iter[i] >= self.stopping_rounds:
              raise EarlyStopException(self.best_iter[i], self.best_score_list[i])
  ```
  Synthetic reproduction (8 seeds, weak signal, small noisy valid set, no `scale_pos_weight`): the
  shipped `["binary_logloss","auc"]` config selected a **strictly earlier** `best_iteration` than a
  logloss-only config in 4/8 seeds (11 vs 28, 13 vs 23, 29 vs 48, 34 vs 40) and never a later one.
- **Self-refutation attempted:** (a) I checked whether passing `train_set` as `valid_sets[0]` makes the
  *training* metric drive early stopping — it does not; `lgb.train` detects `valid_data is train_set`
  (`engine.py:255-261`), names it, and `_is_train_set` skips it. That pattern is clean. (b) I checked
  whether `best_iteration` is lost on save/load and found it is not (`model_to_string()` defaults
  `num_iteration=self.best_iteration`, so the saved string is already truncated and the reload's
  `best_iteration=-1` → "all trees" is equivalent). (c) I checked whether `tests/test_modeling_gbm.py`
  pins the behaviour — every GBM test passes `early_stopping_rounds=0`, which disables the callback
  entirely (`_should_enable_early_stopping`), so **no test exercises early stopping at all**. (d) I
  checked whether `scale_pos_weight` alone explains the short fits: the `_balanced` variant (no weight,
  same metric list) still shows 24 % of folds ≤5 trees and 4 folds at 1 tree, so the AUC monitor is a
  genuine co-driver, not a red herring. (e) `DECISIONS.md` has no entry acknowledging AUC in the
  early-stopping monitor (grep `is_unbalance`, `scale_pos_weight`, `best_iteration`).
- **Fix:** either drop `"auc"` from `kw["metric"]` (keep it out of the monitored set) or pass
  `lgb.early_stopping(rounds, first_metric_only=True)`. Independently, add an assertion/log of
  `booster.best_iteration` per fold so a 1-tree fold is visible instead of silent, and consider
  `min_delta` > 0 given how coarse AUC is on a single held-out image.

### modeling-heads-2 — the two classification variants can be trained as *regression* with no error: `fit` validates `y` **after** `astype(np.int8)`, so continuous `fractional_area` truncates to all-zeros and passes the "y must be binary 0/1" check

- **Severity:** medium
- **Liveness:** live-shipped (still reachable; `scripts/sweep.py`'s **default** `--variants` is all 9 variants including both classifiers, and this already happened — 4 junk artifact dirs + 4 rows are banked in the v2 regression sweep)
- **Confidence:** high (measured on committed artifacts)
- **Where:** `src/modeling/gbm.py:401-406` (validation after the cast), `:412` (`n_pos > 0` guard), with `scripts/sweep.py:43,128` and `scripts/train_gbm.py:43` as the entry points that never filter on `gbm.CLASSIFICATION_VARIANTS`

`y_bin = y.astype(np.int8, copy=False)` runs *before* the binary check, and the check explicitly
accepts a single-class y (`np.array_equal(unique, [0])`) to tolerate the LOIO empty-truth fold. A
`fractional_area` vector lives in `[0, 1)`, so the int8 cast maps it to all zeros and the guard passes.
LightGBM then trains an all-negative binary booster and `predict` returns ~1e-15 everywhere, which
`run_loio(task="regression")` records as a normal regression result. `gbm.py` already exports
`CLASSIFICATION_VARIANTS` for exactly this purpose (`tests/test_modeling_gbm.py:22` and
`scripts/probes/_sweep_w0.py:50` use it) but the two main GBM entry points never adopted it.

- **Failure scenario:** it happened. `models/lightgbm_classification/{257740081fc901d5/scale_S8,
  c6c71d5ef90080bc/scale_S16, 97a465cf5eb593a8/scale_S32, 3078a6f2e423a544/scale_S64}` all carry
  `snapshot.json` with `task="regression"`, `target_col="fractional_area"`,
  `variant="lightgbm_classification"` (written 2026-05-29T07:36–07:57Z by `sweep.py`), and
  `predictions.parquet` with **`y_pred` constant at 1e-15 across 2,700,653 / 665,794 / 161,005 /
  37,315 tiles** (`nunique == 1`). Their `metrics.json` looks legitimate — `n_real_folds: 38`,
  `n_specificity_folds: 0`, `presence_auc_mean: 0.5` — and `rmse_log1p_mean = 0.022118` at S=8 is the
  **best of all four variants** at that scale (tweedie 0.022383, log1p_huber 0.022567, two_stage
  0.022458), because a near-zero constant minimises RMSE on a zero-inflated target. All four rows are
  in the currently-selected v2 aggregate (`models/_sweep/20260529T061553Z/aggregate.parquet`, 16 rows).
  The only thing keeping them out of notebook 11's regression table is a hand-maintained allowlist.
- **Evidence:**
  ```
  src/modeling/gbm.py:401-406
      y_bin = y.astype(np.int8, copy=False)          # fa in [0,1)  ->  all zeros
      if y_bin.ndim != 1: raise ValueError(...)
      unique = np.unique(y_bin)
      if not np.array_equal(unique, [0, 1]) and not np.array_equal(unique, [0]) and not np.array_equal(unique, [1]):
          raise ValueError(f"y must be binary 0/1, got unique values {unique.tolist()}")

  scripts/sweep.py:43   ALL_GBM_VARIANTS = list(VARIANT_CONSTRUCTORS)   # includes both classifiers
  scripts/sweep.py:128  ap.add_argument("--variants", nargs="+", default=ALL_GBM_VARIANTS, ...)
  scripts/sweep.py:11   #   python scripts/sweep.py       # all GBM variants, all scales

  notebooks/_build_11.py:111-113   (the only thing that hides the junk)
      REG_VARIANTS = ['lightgbm_tweedie', 'lightgbm_log1p_huber', 'lightgbm_two_stage']
      aggregate = aggregate[aggregate['variant'].isin(REG_VARIANTS)].reset_index(drop=True)
  ```
- **Self-refutation attempted:** (a) I checked whether any reported figure consumes the junk rows —
  `_build_11.py` filters (above), `_diag_per_image_breakdown.py:41` and `_diag_within_image_deltas.py`
  filter to a single variant, and the committed `11_modeling_qa_v2.ipynb` `reg-table` output shows only
  12 rows. So no *published* number is contaminated today; that is why this is medium rather than high.
  (b) I checked whether the run was a deliberate experiment: `snapshot.json` says `task="regression"`,
  and `run_loio` was called with the default `task`, so it is not a classification run mislabelled — it
  is a regression run of a classifier. (c) I checked `DECISIONS.md` for a note about these dirs — none.
  (d) The single-class acceptance itself is *intentional* (the LOIO empty-truth fold), so the bug is the
  cast-then-validate ordering, not the tolerance.
- **Fix:** validate before casting — `u = np.unique(y); if not np.isin(u, (0, 1)).all(): raise` — and
  make `sweep.py` / `train_gbm.py` reject `variant in gbm.CLASSIFICATION_VARIANTS` (or route them to
  `task="classification"` with a required `--target-id`).

### modeling-heads-3 — commit 61184fd silently neutralised the `meaningful_threshold` monkeypatch in five count-target sweep probes; re-running any of them re-scores counts as presence (invariant 8) while `snapshot.json` still claims the right threshold

- **Severity:** medium
- **Liveness:** live-active (latent regression: all currently-banked artifacts are correct, but the plumbing is broken for any future run)
- **Confidence:** high
- **Where:** `src/modeling/evaluate.py:647-648` (the explicit kwarg that shadows the patch) × `scripts/probes/_sweep_w0.py:104-112`, `_sweep_target_reformulation.py:130-141`, `_sweep_stage6a.py:118-124`, `_sweep_stage6b.py:122-128`, `_sweep_perimage_std.py:88-95`

Five probes inject their target-appropriate threshold by monkeypatching
`src.modeling.evaluate.per_fold_metrics` with a wrapper whose **default argument** carries the
threshold — a technique that only works while `run_loio` calls `per_fold_metrics` *without* that
kwarg. Commit `61184fd` (2026-06-13, "thread meaningful_threshold through run_loio (count target was
scored as presence)") made `run_loio` pass `meaningful_threshold=meaningful_threshold` explicitly,
where `meaningful_threshold` is `run_loio`'s own `1e-2` default because none of the five probes passes
it. The explicit argument shadows the patched default, so the wrapper now forwards `1e-2` — exactly the
bug the commit was written to fix.

- **Failure scenario:** `python scripts/probes/_sweep_w0.py ... --targets boulder_count` today writes
  `snapshot.json` with `"meaningful_threshold": 50.0` (line 90, taken from `_meaningful_threshold`)
  while `metrics.json`'s `per_fold[*]["meaningful_threshold"]` is `0.01`. For a count target
  `count > 0.01` ⇔ `count >= 1`, so `meaningful_auc` / `pr_auc` / `precision_at_top_5pct` become
  **presence** metrics — the one statistic `CLAUDE.md` forbids reporting — inside an artifact that
  advertises the correct threshold. The same applies to `_sweep_stage6a/6b` (Stage 6 verdicts) and
  `_sweep_perimage_std` (bet-1 verdict). Nothing asserts snapshot/metrics agreement.
- **Evidence:**
  ```
  src/modeling/evaluate.py:647-648
      m = per_fold_metrics(y_test, y_pred, held_out_obs_ids=fold.held_out_obs_ids,
                           meaningful_threshold=meaningful_threshold)   # = run_loio's 1e-2 default

  scripts/probes/_sweep_w0.py:102-112   (stale comment + shadowed default)
      # Plumb meaningful_threshold into per_fold_metrics (same monkeypatch route
      # as _sweep_target_reformulation.py -- run_loio has no threshold kwarg).
      def _patched(y_true, y_pred, *, held_out_obs_ids, meaningful_threshold=mt):
          return orig_per_fold_metrics(y_true, y_pred, held_out_obs_ids=held_out_obs_ids,
                                       meaningful_threshold=meaningful_threshold)
      _ev.per_fold_metrics = _patched
  ```
- **Self-refutation attempted:** (a) I verified the patch *is* picked up (module-global lookup at call
  time), so the failure is purely the shadowed default, not an ineffective patch. (b) I audited all 40
  committed non-`fractional_area` runs, comparing `snapshot["meaningful_threshold"]` to
  `metrics["per_fold"][0]["meaningful_threshold"]`: **zero mismatches** — every count/area run predates
  61184fd (latest 2026-06-11) or uses the post-fix kwarg route (`fang_tier2/*`, metrics 50.0). So no
  banked number is wrong; the defect is a live trap, which is why it is medium not high. (c) I confirmed
  none of the five probes passes `meaningful_threshold=` to `run_loio` (read all five call sites).
  (d) `DECISIONS.md:3552-3559` records the original fix and its test
  (`tests/test_evaluate_meaningful_threshold.py`), but that test only covers direct `run_loio` callers,
  not the monkeypatch route.
- **Fix:** delete the monkeypatch in all five probes and pass
  `meaningful_threshold=_meaningful_threshold(...)` to `run_loio`; add
  `assert metrics["per_fold"][0]["meaningful_threshold"] == snapshot["meaningful_threshold"]` in
  `write_run_artifacts` (or in each probe) so the two can never disagree silently again.

### modeling-heads-4 — the four two-stage cousins save their per-fold artifacts into a **directory named `booster.txt`**, because both entry points route on `variant == "lightgbm_two_stage"` while `_TwoStageBase`'s docstring claims a `startswith` prefix check

- **Severity:** low
- **Liveness:** live-shipped but latent (no `fold_*` artifacts exist for the cousins on disk, so it has never actually been triggered)
- **Confidence:** high (reproduced)
- **Where:** `src/modeling/gbm.py:504-506` (the false claim), `:340-346` (`save` = `mkdir` + two files), `scripts/sweep.py:118-121`, `scripts/train_gbm.py:132-135`, pinned by `tests/test_modeling_gbm.py:81-88`

`_TwoStageBase`'s docstring asserts that "sweep.py's 'is two-stage' path keys on the booster save shape,
so a name-prefix check (`startswith("lightgbm_two_stage")`) routes all four new variants correctly."
Neither caller does that: both compare with `==`, so `lightgbm_two_stage_balanced` / `_weighted` /
`_gamma` / `_combined` take the single-booster branch `model.save(fold_out / "booster.txt")` — and the
inherited `LightGBMTwoStage.save` does `Path(path).mkdir(parents=True, exist_ok=True)`, creating a
*directory* called `booster.txt` containing `presence.txt` + `magnitude.txt`. The unit test's own `else`
branch does exactly this and passes, so the layout bug is pinned as correct.

- **Failure scenario:** `python scripts/sweep.py --variants lightgbm_two_stage_balanced --scales 3`
  (a documented, argparse-accepted invocation for the P1-promoted variant) writes
  `models/lightgbm_two_stage_balanced/<hash>/scale_S64/fold_<obs>/booster.txt/{presence,magnitude}.txt`.
  Every consumer of the documented layout then breaks or misreads: `notebooks/_build_10.py:380`,
  `_build_11.py:290` and `scripts/probes/_summarize_modeling_results.py:144` glob
  `fold_*/booster.txt` and call `bp.read_text(...)` on the match — on a directory that raises
  `PermissionError`/`IsADirectoryError` (in `_build_10` it is swallowed by a bare
  `except Exception` and printed as "feature importance skipped"). Verified directly:
  `lightgbm_two_stage_balanced` → `booster.txt is_dir=True contents=['magnitude.txt','presence.txt']`;
  `lightgbm_tweedie` → `is_dir=False`.
- **Evidence:**
  ```
  src/modeling/gbm.py:504-506
      identical to `LightGBMTwoStage`; sweep.py's "is two-stage" path keys on the
      booster save shape, so a name-prefix check (`startswith("lightgbm_two_stage")`)
      routes all four new variants correctly.

  scripts/sweep.py:118-121
      if variant == "lightgbm_two_stage":
          model.save(fold_out)
      else:
          model.save(fold_out / "booster.txt")

  src/modeling/gbm.py:340-346
      def save(self, path): path = Path(path); path.mkdir(parents=True, exist_ok=True)
          (path / "presence.txt").write_text(...)

  tests/test_modeling_gbm.py:81-88   (pins the wrong layout for the 4 cousins)
      if variant == "lightgbm_two_stage": save_path = tmp_path / "two_stage"
      else:                               save_path = tmp_path / "booster.txt"
  ```
- **Self-refutation attempted:** (a) I searched the filesystem: `models/lightgbm_two_stage_balanced|
  _combined|_gamma|_weighted/**` contain **no** `fold_*` directories at all — every banked cousin run
  came from a probe sweep that skips booster persistence — so nothing on disk is currently mislaid.
  (b) I confirmed the three `booster.txt` globs all target `lightgbm_tweedie` only, so no committed
  figure is affected. (c) `DECISIONS.md` has no `booster.txt` entry, so this is not a recorded choice.
- **Fix:** replace `variant == "lightgbm_two_stage"` with
  `variant.startswith("lightgbm_two_stage")` in both scripts (matching the docstring), and change the
  test to derive the path the same way so it can fail.

### modeling-heads-5 — CNN `load()` leaves the network on CPU while `predict()` moves inputs to `params.device` (CUDA by default), so any reload-then-predict crashes on the GPU box

- **Severity:** low
- **Liveness:** dead-closed (PLAN_CNN closed; CNN rejected) but the code is reachable from `models/cnn_*/**/fold_*/state_dict.pt`
- **Confidence:** high
- **Where:** `src/modeling/cnn.py:413-416` and `:570-573` (`load`), vs `:385-396` and `:546-555` (`predict`), with `:82` (`device: str = field(default_factory=_default_device)` → `"cuda"` when available)

`load()` constructs `SmallCNN(...)` and `load_state_dict(...)` but never `.to(self.params.device)`.
`predict()` then does `device = torch.device(self.params.device)` and `xb.to(device)` while the module's
parameters are still on CPU → `RuntimeError: Expected all tensors to be on the same device`. `fit()`
does move the net (`:310`, `:493`), so the bug only bites the reload path, which is precisely the path
used for post-hoc re-scoring of banked folds (e.g. `scripts/probes/_w2_adabn.py`).

- **Failure scenario:** on this machine (`local_gpu_available`: RTX 5070, CUDA torch installed),
  `m = SmallCNNClassifier(params=CNNParams(patch_size_px=64)); m.load("models/cnn_bce_S64/<hash>/<suffix>/fold_ESP_.../state_dict.pt"); m.bind_predict_data(keys); m.predict(X)` raises immediately.
  On a CPU-only box (Sherlock without GPU) it silently works, so the failure is environment-dependent.
- **Evidence:**
  ```
  src/modeling/cnn.py:413-416
      def load(self, path: str | Path) -> None:
          self._state_blob = Path(path).read_bytes()
          self._net = SmallCNN(self.params.patch_size_px, dropout=self.params.dropout)
          self._net.load_state_dict(torch.load(io.BytesIO(self._state_blob)))
  src/modeling/cnn.py:385-386
          device = torch.device(self.params.device)
          self._net.eval()
  src/modeling/cnn.py:396
                  xb = xb.to(device, non_blocking=True)
  ```
- **Self-refutation attempted:** (a) I checked the tests: `tests/test_modeling_cnn.py` never calls
  `load()` — every test goes fit→predict in one object, so nothing covers the reload path. (b) I checked
  whether a caller compensates: `scripts/probes/_w2_adabn.py:73` binds predict data but does not `.to()`
  the net either. (c) The GBM equivalents have no device concept, so this is CNN-only.
- **Fix:** `self._net = SmallCNN(...).to(torch.device(self.params.device))` in both `load()` methods
  (or move the net at the top of `predict`).

### modeling-heads-6 — the CNN's brightness jitter is ±15 % of the **full 0–255 DN range**, not "±15 % of the per-tile intensity range" as PLAN_modeling §4 and the module docstring both specify — measured 2.1× the intended magnitude

- **Severity:** low
- **Liveness:** dead-closed (W2 augmentation grid; "H-B photometric augmentation REFUTED", DECISIONS 2026-06-11) — but it is a confounder on that refutation
- **Confidence:** high (measured on real S=64 context patches)
- **Where:** `src/modeling/cnn.py:194-195`, contradicting `src/modeling/cnn.py:14` and `PLAN_modeling.md:214`

The spec (`PLAN_modeling.md:214`: "Random brightness jitter ±15 % of the **per-tile** intensity range")
and `cnn.py`'s own docstring and inline comment both say per-tile; the code uses a constant `* 255`.

- **Failure scenario:** on `dataset_v2/context_patches/ESP_017355_2260_S64.npy` (304,428 patches, first
  4,000 sampled) the median per-tile DN range is 123 and the median per-tile DN sd is 19.9. The spec's
  jitter is therefore ±18.4 DN; the code applies ±38 DN — **2.1× stronger, ≈1.9 per-tile sd of pure
  additive offset**. Clipping is negligible (0.01 % of pixels at ±38 DN), so this is not destructive,
  but the photometric cell (C/E) that the W2 grid declared "REFUTED cohort-level" was run with a
  brightness perturbation twice the designed strength, i.e. the refutation is of a stronger
  augmentation than the one specified.
- **Evidence:**
  ```
  src/modeling/cnn.py:14      - brightness jitter +-15% of the per-tile intensity range,
  src/modeling/cnn.py:194-195
              # brightness jitter +-15% of the tile's range
              rng_brightness = self.rng.uniform(-0.15, 0.15) * 255
  PLAN_modeling.md:214        - Random brightness jitter ±15% of the per-tile intensity range.
  ```
- **Self-refutation attempted:** (a) I looked for a DECISIONS entry redefining the magnitude — `grep -i
  brightness DECISIONS.md PLAN_CNN.md` turns up no magnitude spec, only the gamma range
  `[0.8, 1.25]` (DECISIONS 2900) which *is* implemented as documented. (b) I checked whether clipping
  makes it destructive (it does not — 0.01 %), which is why this is low and not medium. (c) I checked
  the tests: `test_cell_geometric_preserves_pixel_multiset` and `test_cell_none_is_identity_div255`
  cover the geometric and no-aug cells; **no test asserts anything about the photometric magnitudes**,
  so nothing pins the current value as intended.
- **Fix:** `rng_brightness = self.rng.uniform(-0.15, 0.15) * float(img.max() - img.min())` (or amend
  the docstring and `PLAN_modeling.md:214` to say "of the 0–255 range" if the global form is wanted) —
  and note in DECISIONS that the W2 grid ran with the global form.

## Refuted by my own check

- **`predict(num_iteration=best_iteration)` breaking on reload.** I expected `lgb.Booster(model_str=...)`
  to reset `best_iteration = -1` (`basic.py:3614`) and so predict with *all* trees, diverging from the
  fitted model's early-stopped predictions. It does not: `model_to_string()` defaults
  `num_iteration = self.best_iteration` (`basic.py:4584-4585`), so `save()` already truncates and
  `-1` → "all of the truncated trees" is equivalent. Save/load round-trip and `model_hash` are exact.
- **Training set in `valid_sets` corrupting early stopping.** All GBM variants pass `train_set` as
  `valid_sets[0]` named `"train"`. LightGBM detects `valid_data is train_set`
  (`engine.py:255-261`), sets `_train_data_name`, and `_EarlyStoppingCallback._is_train_set`
  `continue`s over it (`callback.py:425-429`). The train metric is displayed but never monitored.
- **`num_iteration=0` when early stopping never fires.** `lgb.train` leaves `best_iteration = 0`, and
  `num_iteration <= 0` means "no limit" in the C API, so `predict` correctly uses all trees.
- **Weight normalisation asymmetry in `_TwoStageBase`.** Train and valid magnitude weights are each
  divided by *their own* mean (`gbm.py:565-568` vs `:613-616`). Harmless: LightGBM's metrics are
  `Σ w·l / Σ w`, so a uniform rescale cancels; early stopping is unaffected.
- **`n_estimators` inside `to_lgb_kwargs()` fighting `num_boost_round`.** `lgb.train`'s
  `_choose_param_value("num_iterations", ...)` prefers the alias in `params`, but both are
  `self.params.n_estimators`, so they always agree.
- **`expm1(Huber-on-log1p)` as a biased mean estimator.** Real, but explicitly diagnosed in-repo:
  `gbm.py:481-485` ("magnitude head shrunk to log-positive median: log1p+Huber-on-positives fits the
  geometric median") and `DECISIONS.md:1397`. Acknowledged, not a finding.
- **`is_unbalance` / `scale_pos_weight` inflating probabilities without an inverse at predict.** Also
  real and also explicitly diagnosed — it is the entire motivation for `LightGBMTwoStageBalanced`
  (`gbm.py:487-489`, `DECISIONS.md:1393-1405`) and `LightGBMClassificationBalanced`
  (`gbm.py:677-682`, `DECISIONS.md:2497`).
- **`LightGBMTwoStageCombined`'s gamma + `weight ∝ y` estimating `E[y²]/E[y]` rather than `E[y|x]`.**
  Mathematically true, but that tail emphasis is the stated intent (`gbm.py:489-493`) and the variant
  was never promoted (P1 promoted `_balanced` only), so there is no reported number to correct.
- **DataLoader `shuffle` on an eval loader desyncing CNN predictions.** Both eval/predict loaders are
  `shuffle=False` (`cnn.py:304-307`, `:390-391`, `:489-490`, `:550`) and `preds` are filled by a
  monotone cursor before `out[valid_rows] = preds`, so row alignment holds.
- **Validation `_PatchDataset` inheriting `geometric=True/photometric=True` defaults**
  (`cnn.py:302-303`, `:487-488`). Inert: `__getitem__` gates on `self.augment`, which is `False`.
- **`sweep_select.pick_sweep` mis-selecting a sweep.** Ran it for all 3 kinds × 3 dataset dirs: every
  call returns the correct tagged dir, `Path.glob("*/")` works on Python 3.14.3, and the legacy
  untagged fallback resolves to the two genuine v1 dirs.
- **`src/modeling/inference.py` being stale dead code** (its docstring says off-HiRISE inference is
  "deferred" while `src/mapping.py` shipped it). Real, but already noted in
  `docs/review_2026-07-31/features.md:277`, and nothing imports the module.

## Verified clean

- **The inner-validation rotation in `run_loio`** (`evaluate.py:610-623`): `unique_train` is taken from
  `groups_train`, which by construction excludes the held-out image, and the
  `assert inner_val_code not in held_codes` is a real (if unreachable) belt. `train_gbm.py:113-116`,
  `sweep.py:102-106`, `train_binary.py`, `sweep_binary.py` and `train_cnn.py:58-69` all reproduce the
  same rule, so the persisted per-fold boosters match the boosters that produced `predictions.parquet`.
  `sweep_cnn.py:62-78` uses a 4-image group-aware pool drawn from the same training-only code set.
  **No GBM `eval_set` site in `src/` or top-level `scripts/` uses the held-out image.** The one
  exception is `scripts/probes/_smoke_gbm_one_fold.py:25` (`eval_set=(f.X_test, y_test)`), which is a
  smoke probe that writes no artifact.
- **Two-stage hurdle composition.** `predict` = `p_pos × back_transform(mag)` with each head using its
  own `best_iteration`; the `< 10 positives` guard returns a zero magnitude rather than raising
  (`gbm.py:292-296`, `:548-550`), and `predict` handles `_magnitude is None`
  (`gbm.py:332-333`, `:644-645`). The `_mag_back_transform` closure is correctly reconstructed from
  `magnitude_loss` after `load()` (`gbm.py:635-643`). Gamma's log link means `clip(p, 0, None)` is the
  right (identity) back-transform.
- **Dataclass inheritance for the 4 cousins + `LightGBMClassificationBalanced`**: field re-declaration
  keeps positional order, `make_factory` uses `cls(params=p)` (keyword), and each subclass differs in
  exactly the one knob its docstring names.
- **`binary_target.py`**: `binarize` raises `KeyError` on a missing source column; the `>` / `>=`
  comparisons match the documented labels; `fa_gt_1e-2` uses strict `>` consistent with the project's
  reporting standard.
- **LightGBM seeding**: `seed`, `bagging_seed`, `feature_fraction_seed`, `data_random_seed` and
  `deterministic=True` are all set (`gbm.py:78-93`), and `test_model_hash_is_stable` confirms
  bit-identical boosters across two fits.
- **`_standardize_matrix_per_group` / `standardize_fold_per_image` / `augment_fold_with_fang`**
  (`loaders.py:189-382`): train and test statistics are each computed from their own rows, so no
  statistic crosses the split boundary; the Fang join is `validate="one_to_one"` with a
  no-miss assertion.
- **`gather_patches` row alignment** (`loaders.py:406-433`): `valid_rows` are positional indices into
  the (reset-index) keys frame and are used consistently to subset `y` in `fit` and to scatter
  predictions in `predict`.

## Coverage note

**Read in full:** `src/modeling/gbm.py` (722), `src/modeling/cnn.py` (578),
`src/modeling/binary_target.py`, `base.py`, `inference.py`, `sweep_select.py`, `__init__.py`,
`src/modeling/loaders.py` (433), `src/modeling/evaluate.py:560-753` (the LOIO runner + artifact
writer), `scripts/train_gbm.py`, `scripts/sweep.py`, `scripts/train_cnn.py`, `scripts/sweep_cnn.py`,
`tests/test_modeling_gbm.py`, `tests/test_modeling_cnn.py`, and the relevant parts of the installed
`lightgbm/callback.py` + `basic.py` + `engine.py` (4.6.0) to pin early-stopping and save semantics.

**Grepped / spot-read only:** `src/modeling/mlp_head.py` (assigned to `fm-embeddings`; I only checked
its `eval_set` provenance at `:401`), `scripts/sweep_binary.py`, `train_binary.py`,
`sweep_within_image.py`, `run_modeling_slim.py` (assigned to `other-scripts`; checked their
`eval_set` and `early_stopping_rounds` only), the five `scripts/probes/_sweep_*.py` files (read the
monkeypatch + `run_loio` call sites), `notebooks/_build_10/11.py` (only the `booster.txt` and
`aggregate.parquet` consumers), `PLAN_Stage5b.md`, `PLAN_modeling.md`, `PLAN_CNN.md`.
`DECISIONS.md` grepped by term (`two_stage_balanced`, `is_unbalance`, `scale_pos_weight`,
`best_iteration`, `booster.txt`, `expm1`, `geometric median`, `brightness`, `photometric`,
`optimistic`, `selection bias`) — never read linearly.

**Measured (read-only, over committed/derived artifacts):** tree counts in every
`fold_*/classifier.txt` of the two `fa_gt_1e-2` Tier-1 runs and the `_balanced` counterpart;
snapshot-vs-metrics `meaningful_threshold` across all 40 non-`fractional_area` runs under `models/`;
the four `lightgbm_classification` regression artifact dirs and `models/_sweep/*/aggregate.parquet`;
per-tile DN statistics of one `dataset_v2/context_patches/*_S64.npy`. Three tiny synthetic LightGBM
fits (≤6 k rows) were used to confirm the early-stopping mechanism and the `save()`-creates-a-directory
behaviour; no project training, sweep, notebook or map build was run, and no CTX/HiRISE imagery or
network was touched.

**Could not check:** the exact AUC delta attributable to finding 1 — isolating it requires re-running
the 38-fold binary sweep with and without `"auc"` in the metric list, which is a training run and out
of scope here. I therefore quantified the *consequence* (tree counts, and the ≤10-tree vs >10-tree AUC
gap of 0.643 vs 0.683 at S=32) rather than the counterfactual. Also not checked: whether any Sherlock
run reproduced these variants with different LightGBM versions (`SHERLOCK_RUN.md` not read), and the
CNN's CUDA determinism (`torch.backends.cudnn.deterministic` / `use_deterministic_algorithms` are never
set, so CNN `model_hash()` is not reproducible on GPU — a low-severity reproducibility gap I did not
file separately because the CNN programme is closed).
