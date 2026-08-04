# Review area: notebooks

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-01
- **Verification:** self-refuted (single-agent pass; not independently verified)
- **Passes:** two. Findings **notebooks-1..3** are from pass 1 (already carried into
  `docs/CODE_REVIEW_2026-07-31.md` §4d/R42). Findings **notebooks-4..6** are from a second,
  independent pass over the gaps pass 1's own *Coverage note* left open — chiefly the
  notebook-internal **statistics** in `_build_10`/`_build_11`/`_build_12`. Numbering is
  append-only so the register's citations stay valid; **by severity the order is
  notebooks-4 (high) > notebooks-1 = notebooks-2 = notebooks-5 (medium) > notebooks-6 =
  notebooks-3 (low)**.

Scope note: the `invariants` reviewer got to a large slice of this brief first and filed
**invariants-1** (notebook 17 drift), **invariants-2** (notebook 20's `.ipynb`-only banner),
**invariants-3** (notebooks 12/13 committed unexecuted), **invariants-5** (no `_build_18.py`;
CLAUDE.md's "notebooks are generated" false for 7 of 28), plus the hardcoded-absolute-path sweep and
a full 21-way `_build` → `.ipynb` regeneration diff. I re-confirmed their drift census independently
(every commit touching a `_build_NN.py` also touched its `.ipynb`; no `_build` is newer than its
notebook) and did **not** re-file any of it. Their coverage note explicitly leaves *notebook-internal
computation* to this area, and `numerics.md:363` and `calibration.md:519` defer to it too — so that is
where I spent the pass: which notebook computes a reported number, from which artifact, and whether
the artifact is the one the surrounding prose claims.

## Findings

### notebooks-1 — Notebook 10 is pinned to the v1 sweep for its tables but resolves model artifacts by *most-recently-modified* directory, so three of the figures `docs/modeling_results.md` publishes as the v1 baseline are the v2 runs

- **Severity:** medium
- **Liveness:** live-shipped (the figures are embedded in two published writeups; the glob pattern is also live in the v2 notebook)
- **Confidence:** high
- **Where:** `notebooks/_build_10.py:256-257`, `:314-315`, `:377-378`, `:596-597`, `:995-996`
  (mtime globs) vs `:110-114`, `:478-479`, `:722-723` (the pinned sweeps); caption at `:390`;
  same glob at `notebooks/_build_11.py:288-290` and `:385-386`; consumers
  `docs/modeling_results.md:208`, `:319`, `:334`, `docs/modeling.md:380`

Notebook 10 is `README.md:175`'s "v1 priority10 modeling QA (frozen baseline)" and pins its three
sweep tables to explicit timestamped directories precisely so they stay v1. But five other cells
resolve `models/<variant>/<config_hash>/scale_S<n>` by globbing across **all** config hashes and
taking `sorted(..., key=st_mtime)[-1]`. v1 (`scheme: loio_9fold`, 9 folds) and v2
(`scheme: loio_nfold`, `dataset_dir: dataset_v2`, 38 folds) write into the *same* variant tree, and
v2 ran on 2026-05-28/29 — before the notebook was executed and committed on 2026-05-29T07:59
(`c8d68cd`). So the pinned tables are v1 and the figures beside them are v2, with no signal to the
reader. The x-axis caption is hardcoded `'…across 9 folds'` while the run averaged has 38.

- **Failure scenario:** `docs/modeling_results.md:195-205` prints the v1 named top-10 (`shadow_fraction`
  16.1 % at S=8) and then embeds `10_feature_importance_tweedie_S8.png` at `:208` as its
  visualisation, followed by "The single most informative feature is the per-tile shadow fraction."
  The embedded figure is the v2 run: its top bar is `Column_2` at gain 2.9e5 (25.2 % share), not the
  v1 run's `Column_16` at 9.7e3 (**16.1 % share — exactly the doc's number**). A reader checking the
  claim against the figure finds a different feature at a different share, and cannot tell which is
  right. More seriously, re-running the notebook (the documented refresh step) after *any* later GBM
  run silently re-points five panels at that run, so the "frozen baseline" figures drift with the
  models directory.
- **Evidence:**
  ```
  notebooks/_build_10.py:110  # PINNED to the v1 (priority10) regression sweep documented in docs/modeling_results.md
  notebooks/_build_10.py:114  SWEEP_DIR = MODELS_ROOT / '_sweep' / '20260524T071830Z'

  notebooks/_build_10.py:377      scale_dirs = sorted((MODELS_ROOT / 'lightgbm_tweedie').glob('*/scale_S8'),
  notebooks/_build_10.py:378                          key=lambda p: p.stat().st_mtime)
  notebooks/_build_10.py:380      booster_paths = sorted(scale_dirs[-1].glob('fold_*/booster.txt'))
  notebooks/_build_10.py:390      ax.set_xlabel('Mean split-gain importance across 9 folds')
  ```
  The committed notebook's **own output** names a v2 hash in the section labelled v1
  (`10_modeling_qa.ipynb`, cell 38, `execution_count: 22`):
  ```
  regression  preds: models\lightgbm_two_stage\629276139c22da68\scale_S64\predictions.parquet
  ```
  and `629276139c22da68` is v2 — `_build_12.py:120-121,136` documents that exact directory as
  "The full-v2 `lightgbm_two_stage` regression at S=64". Snapshots confirm:
  ```
  models/lightgbm_two_stage/629276139c22da68/scale_S64/snapshot.json
      {'scheme': 'loio_nfold', 'dataset_dir': 'dataset_v2', 'written_at_iso': '2026-05-29T07:24:21Z'}   38 folds
  models/lightgbm_two_stage/8ce4b88b0aad10e9/scale_S64/snapshot.json
      {'scheme': 'loio_9fold',                             'written_at_iso': '2026-05-24T16:45:46Z'}    9 folds
  models/lightgbm_tweedie/0660b50a0abd27ce/scale_S8   loio_nfold / dataset_v2  38 folds  (dir mtime 2026-05-28 23:29)
  models/lightgbm_tweedie/3c2a470e21a8cf80/scale_S8   loio_9fold               9 folds  (dir mtime 2026-05-26 10:16)
  ```
  Mean split-gain over each dir's folds (recomputed from the committed boosters):
  ```
  3c2a470e21a8cf80 (v1, 9 folds):  top Column_16  9676.2   share 0.161
  0660b50a0abd27ce (v2, 38 folds): top Column_2 290495.6   share 0.252
  ```
  The committed PNG `reports/figures/10_feature_importance_tweedie_S8.png` (same commit `c8d68cd`,
  2026-05-29T07:59) shows `Column_2` topping the chart at ~290,000 on a 0–300,000 axis — the v2
  numbers — under the label "across 9 folds". The v1 fold directories are the priority10 ObsIds
  (`ESP_039820_1750`, `ESP_054857_2270`, …); the directory actually used holds the 38 vClaire ObsIds.
- **Self-refutation attempted:** (a) I checked whether the v2 dirs post-date the notebook execution —
  they do not: v2 tweedie S=8 was written 2026-05-29T06:22Z (2026-05-28 23:22 local), the notebook and
  figures were committed 2026-05-29T07:59 local. (b) I checked whether v1 and v2 give the same picture
  so the mix-up is harmless — they do not: different top feature, different share, 30× different gain
  scale. (c) I checked whether the docs' 16.1 % might itself be the v2 number — it is not; 0.161 is
  the v1 run's top share to three decimals, so the table is v1 and only the figure moved. (d) I
  checked whether the mtime pick is documented as intentional — the in-code comments
  (`:253-254`, `:375-376`, `:991-992`) justify globbing *across hashes* (which the
  `sweep_vs_train_gbm_artifacts` memory note endorses) but say nothing about disambiguating v1 from
  v2; the pinning comments three cells earlier show the author's intent was the opposite. (e) I
  grepped `DECISIONS.md` for `629276139c22da68`, `st_mtime` and `config_hash` — no entry records this
  as a deliberate cross-dataset comparison. (f) `reports/map_region/` and the two other pinned sweeps
  are unaffected, so no live map or frozen-recipe number is wrong — this is a documentation-integrity
  defect, which caps it at medium.
- **Fix:** replace the five `st_mtime` picks with an explicit hash (or a `snapshot.json` filter on
  `scheme`/`dataset_dir` matching the pinned sweep), derive the fold count from
  `len(booster_paths)` instead of hardcoding 9, and regenerate the three figures. Same change in
  `_build_11.py:288-290`/`:385-386`, where the identical glob makes notebooks 10 and 11 render the
  same directory under "v1" and "v2" labels.

### notebooks-2 — The Stage-7.0 GO statistic (the dust-confound partial correlation) exists only inside notebook 14, its declared producer never computes it, and the published writeup calls it a Spearman correlation when the code is Pearson

- **Severity:** medium
- **Liveness:** dead-closed (compositional programme) — but it is the number that authorised the whole Stage 7a–7e build and it is quoted in a published methods doc
- **Confidence:** high on the mechanism and the mislabel; medium on how much the number would move
- **Where:** `notebooks/_build_14.py:331-342` (`partial_corr`), `:372-373` (the only call sites),
  `:383` (the only writer of `dust_summary.parquet`);
  `scripts/probes/_stage7_feasibility.py:3`, `:12-13`, `:21` (claims it);
  `docs/compositional.md:226-227`; `DECISIONS.md:1669`, `:1694`

`PLAN_Compositional.md` §5.2's decisive step — does the boulder-rich↔`IR/BG` signal survive control for
`dust_index` — is implemented as a 12-line helper inside the notebook. Its declared producer
`scripts/probes/_stage7_feasibility.py` advertises "dust-confound discriminator" in its docstring and
lists `dust_summary.parquet` among its outputs, but contains no partial-correlation code and no such
`to_parquet` — a grep for `partial_corr` / `dust_summary` across the repo returns only `_build_14.py`
and that docstring. This is invariant 10 inverted for the one statistic in the compositional arc that
carried a go/no-go.

Worse, the two places that report it describe a different statistic than the code computes.
`docs/compositional.md:226-227` says `ESP_055253_2245` "passed the partial-correlation dust
discriminator with a **Spearman** partial correlation of +0.16, p = 0.037". `partial_corr` is
`np.polyfit` residualisation followed by `scipy.stats.pearsonr` — a *Pearson* point-biserial partial
correlation on raw `IR/BG` and `RED/BG` ratios, which are heavy-tailed I/F quotients and exactly the
case where Pearson and Spearman diverge. Its p-value also comes from `pearsonr`, i.e. df = n − 2, not
the n − 3 a one-covariate partial correlation needs; and the finite-value mask is `~np.isnan(...)`
only, so an `inf` from a zero `BG` denominator would poison `np.polyfit` into all-NaN rather than be
excluded.

- **Failure scenario:** a reader (or reviewer) of `docs/compositional.md` re-derives the discriminator
  with `scipy.stats.spearmanr` on residuals as the doc specifies, gets a different value on a
  heavy-tailed ratio, and cannot reconcile it — or, worse, accepts +0.16 / p = 0.037 as a rank-based
  (outlier-robust) result when it is a parametric one sitting just inside p < 0.05. Since this is the
  single "survives dust control" observation in a 3-image trio and `DECISIONS.md:1673` records
  "**Final verdict: PASS (a)**" on it, a materially different rank-based value would have changed the
  Stage-7 go decision.
- **Evidence:**
  ```
  notebooks/_build_14.py:331  def partial_corr(x, y, z):
  notebooks/_build_14.py:332      """Partial correlation of x and y controlling for z (Pearson on residuals)."""
  notebooks/_build_14.py:334      m = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
  notebooks/_build_14.py:339      bx = np.polyfit(z, x, 1); rx = x - np.polyval(bx, z)
  notebooks/_build_14.py:341      r, p = sst.pearsonr(rx, ry)
  notebooks/_build_14.py:372      pr_irbg, pp_irbg = partial_corr(sub["rich"], sub["IR_over_BG"], sub["dust_index"])
  notebooks/_build_14.py:383  dust_df.to_parquet(STAGE7 / "dust_summary.parquet")

  scripts/probes/_stage7_feasibility.py:21    - dust_summary.parquet        -- dust discriminator outputs
  # (no `partial`, no `dust_summary` write, no `pearsonr`/`spearmanr` anywhere in that file)

  docs/compositional.md:226   passed the partial-correlation dust discriminator with a Spearman
  docs/compositional.md:227   partial correlation of +0.16, p = 0.037, after controlling for
  ```
- **Self-refutation attempted:** (a) I looked for a second implementation that *is* Spearman —
  `src/stage7d_pooled.py`'s `mann_whitney_partial_dust` (`docs/compositional.md:314`) is the later
  Stage-7d per-image residualisation on a different statistic and post-dates this decision; there is
  no other partial-correlation code in the repo. (b) I checked whether the project already knows the
  code lives in the notebook — `DECISIONS.md:1694` says "partial correlation done in-notebook", so
  the *location* is acknowledged and I have not filed it as the headline; the **Spearman-vs-Pearson
  mislabel** and the probe's phantom output are not recorded anywhere. (c) I checked whether the
  notebook is unexecuted (which would make the number unverifiable, as with notebooks 12/13 in
  invariants-3) — it is not; all 7 code cells carry outputs. (d) I checked whether `inf` actually
  occurs: the reported values are finite, so the `isnan`-only mask has not bitten yet — it is latent,
  and I report it only as a secondary note. (e) `docs/index.md` lists `compositional.md` as a
  reader-facing writeup, so the mislabel is in the surface a non-coder reads.
- **Fix:** move `partial_corr` into `src/colour.py` (or `src/stage7d_pooled.py`) with a test, have
  `scripts/probes/_stage7_feasibility.py` actually call it and write `dust_summary.parquet` as its
  docstring promises, switch the finite mask to `np.isfinite`, correct the df to n − 3, and either
  change `docs/compositional.md:226` to say "Pearson partial correlation" or recompute it as Spearman
  and restate the value.

### notebooks-3 — `.gitignore` excludes an 18 MB per-tile LOIO prediction dump by name while 267 MB of identical-class dumps from the aborted F programme are tracked, alongside an 86 MB notebook

- **Severity:** low
- **Liveness:** live (repo hygiene; the excluded/included split is asymmetric today)
- **Confidence:** high
- **Where:** `.gitignore:50-52` vs `git ls-files reports/figures/f_leg_b_loio_preds*.csv` (15 files);
  `notebooks/05_coregistration_qa.ipynb` (85.9 MB)

`.gitignore:50-52` carves out one file by name with an explicit rationale — "Per-tile LOIO prediction
dump (18 MB; regenerable via `scripts/striping_a1_loio.py`…)". The F programme's fifteen
`reports/figures/f_leg_b_loio_preds*.csv` are the same artifact class at 17.1–19.5 MB each
(**267 MB total**, 39 % of the 692 MB tracked tree), are regenerable by
`scripts/f_leg_b_loio.py:183` from the same kind of inputs, and belong to a **CLOSED** programme —
and none of them is excluded. The abort commit `41a6f26` added nine `.gitignore` lines to keep
709 MB of Stage-B logits and 3.6 GB of variant rasters out, but left these in. Separately,
`notebooks/05_coregistration_qa.ipynb` is 85.9 MB of embedded output (notebooks total 149 MB).
Working tree 692 MB, `.git` 491 MB.

- **Failure scenario:** a collaborator (or a Sherlock deploy, which `SHERLOCK_RUN.md:102-141` does by
  `git`-less tarball precisely because of this) clones the repo and pulls ~0.5 GB, of which ~40 % is
  dead-programme intermediate CSVs; and because the same class of file is excluded elsewhere, the
  next person adding a prediction dump has no consistent rule to follow. `f_leg_b_loio_preds_*.csv`
  at 19.5 MB is also within 5× of GitHub's 100 MB hard per-file limit, which the 86 MB notebook is
  within 15 % of.
- **Evidence:**
  ```
  .gitignore:50  # Per-tile LOIO prediction dump (18 MB; regenerable via scripts/striping_a1_loio.py;
  .gitignore:51  # the small striping_a1_loio_summary.csv IS tracked)
  .gitignore:52  reports/figures/striping_a1_loio_preds.csv

  $ git ls-files | xargs -I{} stat -c '%s {}' | sort -rn | head -3
  85902961 notebooks/05_coregistration_qa.ipynb
  19547038 reports/figures/f_leg_b_loio_preds_minnaert_center_h2_k4.csv
  19541950 reports/figures/f_leg_b_loio_preds_minnaert_center.csv
  # tracked total 692 MB; notebooks 149 MB; loio_preds csvs 267 MB
  ```
- **Self-refutation attempted:** (a) I checked whether the F dumps are still consumed — they are, by
  `scripts/bank_calibration_f.py:48`, `scripts/f_h4_legb.py:158`, `notebooks/_build_27.py:75` and four
  probes, so they cannot simply be deleted; but that is equally true of the excluded A1 dump
  (`striping_a1_loio.py` regenerates it on demand), which is the point of the asymmetry. (b) I
  checked whether they are needed for a **reported** number that could not be recomputed — the
  numbers all live in the small `f_leg_b_loio_summary_*.csv` siblings, which are tracked and stay
  tracked under any fix. (c) I checked whether `41a6f26` deliberately kept them — its message
  enumerates what it gitignored and what it kept (Stage-B logits, Stage-C cache, cohort join parquet)
  and never mentions the leg-B prediction dumps, so this looks like omission rather than a decision.
  (d) I checked whether `models/` being ignored is a comparable defect — it is not: `SHERLOCK_RUN.md`
  :102-141 documents the out-of-band transfer of `models/deployable/{86c51a5dca220f63,
  calibration.npz,parity_ref.npz}`, so the shipped head's absence from git is a known workflow, not
  an oversight (see *Refuted*).
- **Fix:** either gitignore `reports/figures/f_leg_b_loio_preds*.csv` with the same rationale as line
  50 (they regenerate from `scripts/f_leg_b_loio.py`), or delete line 52 and state that per-tile
  dumps are tracked — one rule either way, recorded in DECISIONS. Strip outputs from
  `05_coregistration_qa.ipynb` or downsample its embedded figures.

### notebooks-4 — The within-image-vs-LOIO diagnostic pairs a **quadrant** AUC against a **whole-image** AUC, which handicaps the within-image arm by a systematic ~0.02 at every scale — the same size as the effect being tested; matching the populations doubles the deltas and turns two of the four "no p < 0.05" cells into p = 0.014 / p = 0.002

- **Severity:** high
- **Liveness:** live-shipped — this is the *only* quantitative instrument behind the H5
  "5 m/px per-tile texture floor" verdict, which is still asserted in two reader-facing docs and
  frames the whole project's ceiling story
- **Confidence:** high on the mis-specification and its direction; medium on the exact magnitudes
  (see the reproduction caveat below)
- **Where:** `notebooks/_build_10.py:789-823` (`per_image_within_minus_loio`), specifically
  `:791-802` (within arm = mean of 4 **quadrant** AUCs) vs `:803-808` (LOIO arm = the
  **whole-image** fold AUC), `:810`; framing at `:769-781` and `:872-879`; the pre-declared
  decision rule at `:700-706`. Same logic, duplicated, in
  `scripts/probes/_diag_within_image_deltas.py:3-6`, `:50-59`, `:61-66`, `:80`.
  Reported surface: `docs/modeling_results.md:961-985` (the v2 §9.4 table + its verdict),
  `:999`, `:1206`; `docs/modeling.md:478`, `:606`.

`_diag_within_image_deltas.py:4-5` states the design plainly: *"Aggregate the 4 within-image
quadrant folds per image -> 1 AUC per image. Pair with the corresponding LOIO fold AUC for the
same image."* But those two AUCs are computed over **different tile populations**: the within-image
arm scores each fold on one quadrant (~¼ of the image's tiles, one corner of its abundance field),
while the LOIO arm scores one AUC over the **whole** image. A whole-image AUC gets credit for the
coarse between-quadrant abundance contrast that a within-quadrant AUC structurally cannot see, so
the LOIO arm is measured on the easier task. `_build_10.py:873-879` nonetheless frames the pair as
commensurable — *"the within-image AUC (averaged across its 4 spatial quadrant folds) alongside the
LOIO AUC for that same image … do bars systematically lift above the LOIO baseline"* — and
`docs/modeling_results.md:981-985` draws the verdict from it.

I measured the bias by restricting the **LOIO model's own** predictions to the **same** quadrants
(re-deriving them from the code's own predicate `2*(ti>=ti_mid) + (tj>=tj_mid)` and the
`quadrant_definitions` banked in `dataset_v2/splits/within_image_4fold.json`) and averaging the
four quadrant AUCs — i.e. the exactly-matched counterpart of the within-image arm. On v2 the
whole-image LOIO AUC is higher than its own quadrant-restricted value at **every** scale, by
**−0.017 to −0.023**, which is the same order as the +0.004…+0.030 deltas the table reports.

- **Failure scenario:** `docs/modeling_results.md:981-983` concludes **"every CI brackets zero and
  no Wilcoxon p < 0.05: training and testing on the *same* image does not meaningfully beat the
  cross-image LOIO baseline"**, and `:999` promotes that to *"the remaining presence-AUC ceiling
  (~0.6) is a per-tile signal limit at 5 m/px, not a data-quantity or generalisation limit"* — the
  H5 verdict, restated at `:1206`, `docs/modeling.md:478` and `:606`. That "no p < 0.05" claim does
  not survive matching the populations. Recomputed on the banked v2 predictions
  (`models/_sweep_within_image/20260529T142227Z` + `models/_sweep/20260529T061553Z`,
  `lightgbm_two_stage`, presence AUC, paired Wilcoxon, same `n` as the published table):

  | S | n | published-style Δ (quad within − **whole** LOIO) | p | matched Δ (quad within − **quad** LOIO) | p |
  |---|--:|--:|--:|--:|--:|
  | 8  | 38 | +0.0040 | 0.63 | **+0.0246** | **0.014** |
  | 16 | 38 | +0.0142 | 0.46 | **+0.0375** | **0.002** |
  | 32 | 37 | +0.0145 | 0.77 | +0.0332 | 0.18 |
  | 64 | 25 | +0.0295 | 0.07 | +0.0461 | 0.08 |

  (LOIO quadrant-AUC vs whole-image AUC: 0.5385/0.5591, 0.5452/0.5684, 0.5542/0.5730,
  0.5626/0.5792 — the systematic shift.) The v1 arm behaves the same way and the sign flips there:
  at S=8 the published +0.0137 becomes **−0.0055** under matching. So the instrument's error runs
  *toward the null that was adopted*, and the headline sentence is an artifact of the mismatch, not
  a finding about CTX.
- **Evidence:**
  ```
  scripts/probes/_diag_within_image_deltas.py:4-5
        1. Aggregate the 4 within-image quadrant folds per image -> 1 AUC per image.
        2. Pair with the corresponding LOIO fold AUC for the same image.

  notebooks/_build_10.py:799-802     # within arm: MEAN OF QUADRANT AUCs
      w = sub.groupby('held_out_obs_id').agg(
          within_auc=(auc_col, 'mean'),
          n_real_folds=('fold_idx', 'count'),
      ).reset_index()
  notebooks/_build_10.py:803-808     # LOIO arm: WHOLE-IMAGE AUC
      # LOIO baseline: one AUC per image (one fold per image).
      lo = loio_baseline[...][['held_out_obs_id', 'auc']].rename(columns={'auc': 'loio_auc'})
  notebooks/_build_10.py:810
      paired['delta'] = paired['within_auc'] - paired['loio_auc']

  src/dataset.py:240-242             # the within-image test set really is one quadrant
      q_arr, keep = _quadrant_array_for_image(df, quadrant_definitions, buffer_tiles=buffer_tiles)
      test_mask = (q_arr == quadrant_idx) & keep

  docs/modeling_results.md:981-983
      As in v1, **every CI brackets zero and no Wilcoxon p < 0.05**: training and
      testing on the *same* image does not meaningfully beat the cross-image LOIO
      baseline.
  ```
- **Self-refutation attempted:** (a) **The obvious alternative fix is worse, and I checked it.**
  Pooling the four quadrant folds' predictions into one whole-image within-image AUC gives
  *lower* values than the quadrant mean (v1 S=8 0.478 vs 0.524; S=32 0.474 vs 0.550) because the
  four quadrants come from four different boosters with different output scales — so pooling is
  *not* the right matched statistic and I did not use it. Restricting the LOIO arm to the same
  quadrants is the only comparison that holds both the tiles and the aggregation fixed. (b) **Does
  the verdict actually die?** No, and I have not claimed it does: `_build_10.py:704-706`
  pre-declared the decision rule as *"Within-image AUC >> LOIO AUC (>= 0.7) → per-image
  generalisation is the binding constraint"*, and the within-image AUC is 0.54–0.61 at every
  scale — nowhere near 0.7. So H5 survives on the *pre-declared* rule; what fails is the
  significance claim the docs actually print. (c) **The matched delta has its own upward bias.**
  `within_image_4fold` runs with `buffer_tiles = 0` (`src/dataset.py:184`, `:212`), so
  test-quadrant tiles are spatially adjacent to training tiles; `leakage-1` documents the
  consequence for the Stage-6a arm. That inflates my matched Δ, so +0.025…+0.046 is an **upper
  bound**. The honest net statement is therefore *neither* number establishes the null: the
  published one is biased toward it by ~0.02, the matched one away from it by an unmeasured
  boundary term. (d) I grepped `DECISIONS.md` for `within-image`, `quadrant`, `within ≈ LOIO` and
  `_diag_within_image_deltas` — the design is described repeatedly, the population mismatch never.
  `PLAN_Stage5c.md` §1/§3 specifies the 2×2 quadrant partition and the LOIO comparison but never
  addresses commensurability. (e) The refuted-list entry *"`buffer_tiles: 0` invalidates
  within-image CV → closed dev-only work"* does not cover this: §9.4 is not dev-only (it is the
  live H5 evidence), and the mechanism here is population mismatch, not adjacency. (f)
  **Reproduction caveat, stated so nobody over-reads the numbers:** my published-style column
  (+0.0040/+0.0142/+0.0145/+0.0295) is close to but not identical with the doc's
  (+0.008/+0.019/+0.018/+0.028) because I recompute presence AUC from each run's
  `predictions.parquet` via `mannwhitneyu` and re-derive quadrants from the predicate, whereas the
  probe reads the banked per-fold `presence_auc` and groups by `fold_idx`. The `n` column matches
  the doc exactly (38/38/37/25) and both columns are computed the same way, so the **difference**
  between the two columns — the thing the finding is about — is unaffected.
- **Fix:** in `_diag_within_image_deltas.py`, compute the LOIO arm on the *same* quadrants
  (restrict the LOIO run's `predictions.parquet` with the split JSON's `quadrant_definitions` and
  average the four quadrant AUCs) and pair that against the within-image quadrant mean; report
  both the matched Δ and the whole-image LOIO AUC beside it. Regenerate
  `docs/modeling_results.md` §7.1/§9.4 and restate the verdict against the pre-declared ≥ 0.7
  rule rather than against a significance test. Re-run `within_image_4fold` with
  `buffer_tiles ≥ 1` before quoting the matched Δ as an effect size. Delete the duplicate in
  `_build_10.py:789-823` and have the notebook read `delta_vs_loio.parquet` (invariant 10).

### notebooks-5 — Notebook 12's target-reframing evidence compares raw `lift@top-K` across two targets whose base rates differ 2.7×, so most of the "1.43 vs 1.02" gap is the lift ceiling moving, not ranking; the project implemented `normalised_lift_at_top_k` in the same session to fix exactly this and never applied it here

- **Severity:** medium
- **Liveness:** live document + live reporting rule — this is the stated dev evidence for
  PROMOTION_QUEUE **P4** (retire `bc_ge_1`), i.e. for CLAUDE.md invariant 8's threshold
- **Confidence:** high (recomputed from the named sweep artifact)
- **Where:** `notebooks/_build_12.py:439-461` (§6.1, which *defines* the correction),
  `:473-483` (the table cell), `:485-497` (the readout that draws the conclusion), esp. `:488-490`;
  `docs/modeling_results.md:1174-1176` (the published table), `:1182-1186`;
  `PROMOTION_QUEUE.md:211`, `:423-427`, `:31`; the unused correction at
  `src/modeling/evaluate.py:271-285`, and its provenance header at `:246-253`

`_build_12.py:458-461` states the rule correctly and in the author's own words: *"For a
common-positive image (80 % boulder-rich) the max-possible lift is only 1.25, so even a perfect
model 'looks weak' by raw lift — **normalized lift = lift × base_rate** corrects for that."* Twenty
lines later `:488-490` compares **raw** lift across two targets with radically different base
rates and reads the difference as ranking quality. At S=64 in the named sweep
(`models/_sweep_binary/20260529T075754Z`) the scored-fold mean base rate is **0.909** for
`bc_ge_1` and **0.339** for `fa_gt_1e-2`, so the two raw numbers are drawn from lift scales whose
ceilings are **1.115** and **24.2** (medians 1.056 and 4.02). `bc_ge_1`'s 1.016 is therefore near
the top of its available range; `fa_gt_1e-2`'s 1.430 is near the bottom of its. `src/modeling/
evaluate.py:271-285` implements `normalised_lift_at_top_k` with a docstring that says precisely
this (*"Lift saturates at `1 / base_rate` … Comparable across images with different base rates"*),
and `:246-253` records that it was added in the same 2026-05-29 session **citing
`docs/modeling_results.md` §11.4** — the very section that still carries the un-normalised
comparison.

- **Failure scenario:** `_build_12.py:488-490` asserts *"The boulder-rich classifier already finds
  **40 % more true positives in its top-K** than the any-boulder classifier."* That sentence is
  false as written: precision@K = lift × base_rate is **0.921** for `bc_ge_1` and **0.388** for
  `fa_gt_1e-2`, so `bc_ge_1`'s top-K is 92 % true positives against 39 % — the boulder-rich
  classifier finds *fewer* true positives in its top-K, not 40 % more. The 40 % is entirely
  relative-to-own-base-rate. Worse, the trend the doc reads off the table across scales is the
  base rate moving: `bc_ge_1`'s scored-fold base rate climbs 0.490 → 0.680 → 0.829 → 0.909 while
  `fa_gt_1e-2`'s is flat at 0.303 → 0.339, so `bc_ge_1`'s lift **ceiling** collapses from 2.04 to
  1.10 over exactly the scales where its reported lift "declines" 1.09 → 1.02. On the base-rate-free
  statistic — fraction of achievable headroom captured, `(lift − 1)/(1/base_rate − 1)`, per fold —
  the two targets are near-identical at fine scale and separate only ~2× at coarse scale:

  | S | `bc_ge_1` raw lift / base rate / headroom | `fa_gt_1e-2` raw lift / base rate / headroom |
  |---|---|---|
  | 8  | 1.089 / 0.490 / **+0.051** | 1.205 / 0.303 / **+0.055** |
  | 16 | 1.041 / 0.680 / **+0.054** | 1.229 / 0.323 / **+0.060** |
  | 32 | 1.027 / 0.829 / **+0.056** | 1.246 / 0.331 / **+0.080** |
  | 64 | 1.016 / 0.909 / **+0.056** | 1.430 / 0.339 / **+0.104** |

  So the real effect is "up to ~2× more of the achievable headroom at coarse scale", not
  "1.43 vs 1.02 / 40 % more true positives". `PROMOTION_QUEUE.md:423-427` names this number as the
  **"Dev evidence"** for P4, and `:211` as the evidence for Problem 5 (metric framing).
- **Evidence:**
  ```
  notebooks/_build_12.py:458-461
    For a rare-positive image (say, 1.3 % boulder-rich), perfect-lift can reach ~77x; lift = 9x is
    a real, operationally useful signal. For a common-positive image (80 % boulder-rich) the
    max-possible lift is only 1.25, so even a perfect model "looks weak" by raw lift --
    **normalized lift = lift x base_rate** corrects for that.

  notebooks/_build_12.py:488-490
    - **`fa_gt_1e-2` at S=64 lifts 1.43x** vs `bc_ge_1`'s 1.02. The boulder-rich classifier already
      finds **40 % more true positives in its top-K** than the any-boulder classifier -- and 1.43x
      more than random.

  src/modeling/evaluate.py:271-278
      def normalised_lift_at_top_k(...):
          """Lift @ top-K normalised by max-possible-lift = `lift x base_rate`.
          Lift saturates at `1 / base_rate`, so high-base-rate images can never reach
          raw lift above ~1.3 even with a perfect classifier. ...  Comparable
          across images with different base rates."""

  PROMOTION_QUEUE.md:423-425
    **Dev evidence**: per [`docs/modeling_results.md`](docs/modeling_results.md) §11.4: the
    existing v2 binary sweep at S=64 had lift@top-K = 1.43 for `fa_gt_1e-2` vs 1.02 for
    `bc_ge_1`
  ```
  Recomputed (scored folds only, `~is_specificity_only & auc.notna()`): `bc_ge_1` S=64 n = 26/38,
  base rate 0.909, lift 1.016, precision@K 0.921; `fa_gt_1e-2` S=64 n = 37/38, base rate 0.339,
  lift 1.430, precision@K 0.388. The doc's 1.09/1.04/1.03/1.02 and 1.20/1.23/1.25/1.43 reproduce
  exactly, so the artifact is the right one.
- **Self-refutation attempted:** (a) **Is the P4 decision wrong?** No — and I am not claiming that.
  `bc_ge_1` at a 0.909 base rate *is* operationally meaningless ("≥ 1 boulder in a 320 m tile" is
  true of 91 % of tiles), which is an independent and sufficient argument, made in the same
  paragraph. The defect is that the *quantitative* evidence offered alongside it does not support
  it and its literal claim is false. (b) **Does normalisation reverse the conclusion?** It depends
  which correction you use, which is itself the point: `lift × base_rate` favours `bc_ge_1`
  (0.921 vs 0.388), headroom-fraction favours `fa_gt_1e-2` (+0.056 vs +0.104), raw lift favours
  `fa_gt_1e-2` by 40 %. Three answers from one pair of runs — raw lift is simply not comparable
  across base rates. I report the headroom version as the fairest and it still supports the
  direction, just at half the claimed strength. (c) **Fold-set mismatch — and it runs the safe
  way.** `bc_ge_1` S=64 drops 12 of 38 folds as `is_specificity_only` (base rate ≈ 1) against
  1 of 38 for `fa_gt_1e-2`, so the two means are over different cohorts; the dropped folds are the
  *highest*-base-rate (lowest-possible-lift) images, so the drop **raises** `bc_ge_1`'s mean lift
  and works against the doc's conclusion. Noted for completeness, not as the finding. (d) I checked
  whether R26 already covers it — R26 is `precision@5%`'s base-rate cap in
  `src/modeling/evaluate.py`; this is `lift_at_top_k` in a notebook and two live docs, and R26's
  own "Where" cites `normalised_lift_at_top_k` as the *existing fix*, so the two are complementary,
  not duplicate. (e) I grepped `DECISIONS.md` for `lift`, `normalised lift`, `base_rate` and `P4`:
  the P4 ruling is recorded, the base-rate non-comparability of the number cited for it is not.
- **Fix:** replace `lift@top-K` with `normalised_lift_at_top_k` (or the headroom fraction) in
  `docs/modeling_results.md:1174-1176` and `PROMOTION_QUEUE.md:423-427`, print the per-target base
  rate in the same table, delete the "40 % more true positives" sentence at `_build_12.py:488-490`,
  and re-state P4's justification on the operational-meaninglessness argument (which stands on its
  own).

### notebooks-6 — Notebook 12's per-image summary hard-codes "7 of 25 / 4 of 25 folds"; the artifact it names says 8 of 37 / 6 of 37, and the notebook's own cell would have printed the right denominator had it ever been executed

- **Severity:** low
- **Liveness:** dead-closed leg, but the numbers are restated in a live doc
- **Confidence:** high
- **Where:** `notebooks/_build_12.py:503-515` (the hand-written markdown), `:520-533` (the code
  cell that computes the same thing); `docs/modeling_results.md:1178-1181`

`_build_12.py:506-507` asserts *"7 of 25 folds AUC > 0.70 (genuinely strong) · 4 of 25 folds
AUC < 0.50 (anti-signal)"*. The named sweep `models/_sweep_binary/20260529T075754Z` at
`fa_gt_1e-2` S=64 has **38** folds, **37** scored (1 `is_specificity_only`), of which **8** exceed
0.70 and **6** fall below 0.50. Every *other* statistic in the same bullet reproduces exactly from
those 37 folds (median 0.607→"0.61", max 0.911→"0.91", min 0.398→"0.40", σ 0.115→"0.12", and both
named exemplars: ESP_042964_2160 AUC 0.911 lift 5.35, ESP_055978_2270 AUC 0.759 lift 9.07), so the
denominator is not a different subset — it is wrong. 25 is the `n` of a *different* table, the
`two_stage` S=64 row of `docs/modeling_results.md:1172` (§9.4), which is a regression
presence-AUC pairing over 25 images. The immediately following code cell (`:526`) prints
`f'{len(sub)} held-out images at fa_gt_1e-2 S=64'` — it would have printed 37 and exposed the
contradiction, but notebook 12 is committed with **zero executed cells** (`invariants-3`), so
nothing checked it.

- **Failure scenario:** the "usable fraction" of the cohort is overstated: 7/25 = 28 % of images
  with AUC > 0.70 versus the true 8/37 = 22 %, while the anti-signal count is understated 4 → 6.
  `docs/modeling_results.md:1178-1181` repeats the counts as "~7 … ~4 … the rest near chance" and
  `:1184-1186` concludes the v2 dataset *"already supports a usable boulder-rich classifier on a
  meaningful subset of held-out images"* — the sentence P4 and P3 lean on. `_build_13.py:44-45`
  carries a third variant of the same census ("~7 strong winners … ~3 anti-signal failures").
- **Evidence:**
  ```
  notebooks/_build_12.py:505-507
    - median AUC **0.61**, max **0.91**, min 0.40, sigma 0.12
    - 7 of 25 folds AUC > 0.70 (genuinely strong)
    - 4 of 25 folds AUC < 0.50 (anti-signal)

  notebooks/_build_12.py:523-528     # the cell that would have contradicted it
      sub = binsf[(binsf['target_id'] == 'fa_gt_1e-2') & (binsf['scale_idx'] == 3)
                  & ~binsf['is_specificity_only'].astype(bool)].copy()
      sub = sub.dropna(subset=['auc'])
      print(f'{len(sub)} held-out images at fa_gt_1e-2 S=64')
  ```
  Measured on `models/_sweep_binary/20260529T075754Z/summary.parquet`: 38 folds, 37 scored,
  count(auc > 0.70) = 8, count(auc < 0.50) = 6, median 0.607, σ 0.115, min 0.398, max 0.911.
- **Self-refutation attempted:** (a) I checked the four other v2 binary sweeps in case the
  markdown quotes a different run — 20260611T002603Z gives 7 and 5 (median 0.590, max 0.937,
  lift 4.86) and the two 20260611 successors give 15 and 5/6; none matches "7 of 25", and none
  reproduces the max 0.91 / lift 9.07 exemplars that the same bullet names, so the cited run is
  the right one and only the counts are wrong. (b) I checked whether a 25-fold subset could give
  min 0.398 — it is the global min over all 37, so any 25-fold subset containing it would have to
  be an arbitrary selection, and no filter in the notebook produces one. (c) `is_specificity_only`
  removes exactly 1 fold at this cell, not 13, so the drop cannot explain 25. (d) I confirmed the
  notebook is unexecuted (0 outputs, `execution_count: null` throughout), which is *why* this
  survived — that part is `invariants-3`, not re-filed; only the wrong number is new.
- **Fix:** regenerate notebook 12 (`python notebooks/_build_12.py` + `nbconvert --execute
  --inplace`) so the cell's own output stands beside the prose, replace the hard-coded counts with
  values read from `summary.parquet`, and correct `docs/modeling_results.md:1178-1181` to
  8 / 6 of 37.

## Correction to an earlier finding in this file

- **notebooks-1's Fix line over-reaches on `_build_11.py`.** It says the same `st_mtime` glob should
  be changed at "`_build_11.py:288-290`/`:385-386`". Only **`:288-290`** is unguarded (the
  `11_feature_importance_tweedie_S8.png` cell, which additionally derives its fold count from
  `len(booster_paths)` at `:299` — so notebook 11's caption cannot go stale the way notebook 10's
  hardcoded "across 9 folds" does). **`:385-386`** is the glob *inside* `artifact_dir`
  (`:384-396`), which then filters every candidate on `snapshot.json`'s `dataset_dir`, `scheme`,
  `tile_size_px` and `target_id` before returning, with the comment *"models/ is shared across
  v1/v2/dev, so match each run's snapshot.json … rather than trusting mtime."* That is the correct
  pattern and is in fact the model for notebooks-1's own fix — it should be cited as the remedy,
  not as a second instance of the bug. The head of the finding (notebook 10's five mtime picks) is
  unaffected, and this is corroborating evidence that the author knew the hazard.

## Refuted by my own check

*(pass 2 additions first, then pass 1's list unchanged)*

- **Notebook 28's FINAL VERDICT banner table might not be reproducible from the committed
  artifacts.** It is, to 3 decimals. Recomputed from `reports/figures/fbuild_abort_*`: population
  `sd(log10 pred/label)` = 0.1702 / 0.3282 / 0.3710 / 0.5318 (banner 0.170 / 0.328 / 0.371 /
  0.532); median ratios 0.890 / 2.225 / 1.921 / 1.349 (banner 0.89 / 2.22 / 1.92 / 1.35); spreads
  5.1× / 29.4× / 32.5× / 189.6× (exact); the "within-tile striping" row is the **median** of
  `*_fm_cv` = 0.8266 / 0.6462 / 0.5750 / 0.5362 (banner 0.827 / 0.646 / 0.575 / 0.536). The
  "1.13× / 1.62× / 1.26×" prose is resid/pfree/**full** over h1only (1.130 / 1.620 / 1.255) even
  though the table's columns are h1only/resid/pfree — an ordering ambiguity, not an error. The
  `full` arm's 0.412 is over **19** of 21 observations (two `full_ratio == 0` → `log10` undefined)
  with a median ratio of 15.0, which is **R33**, already filed.
- **`_build_28.py`'s H1/H2 skill deltas might be mis-signed or mis-sourced.** They are not:
  `f_leg_b_loio_summary_minnaert_center.csv` gives 0.77215 − 0.78600 = **−0.01385** → the prose's
  "−0.0139" at `:362`; the H2 rows give −0.0026 / −0.0510 / −0.1223 → "−0.003 / −0.051 / −0.122"
  at `:431`. (`:400`'s `s.iloc[0], s.iloc[1]` positional base/treatment unpack is brittle — the
  `store` column is literally identical in both rows of the H2 CSVs, only the filename encodes
  `k` — but the row order is consistent across all four files, so no number is wrong today.)
- **`fbuild_abort_per_obs_skill.csv` might contain a mosaic-vs-F skill comparison that **R39** says
  does not exist.** It does not: its columns are `ap_{h1only,full,resid,pfree}` with **no**
  `ap_mosaic`. It is also blind by construction for 6 of 21 observations, whose `sd_off_*` is
  ~1e-7 (a constant within-obs offset ⇒ per-obs AP exactly unchanged, `d_ap = 0.0`). Both points
  strengthen **R39**/**R36** rather than adding anything new.
- **`_build_23.py` rebinds `y`, `lab`, `iso` and `rows` across cells** (`:108` vs `:249`; `:165`
  vs `:422`; `:117` vs `:238`; `:120` vs `:240`). Real shadowing, but every consumer runs before
  the rebind in top-to-bottom execution order, so no committed number is affected; it only breaks
  out-of-order re-execution.
- **`_build_23.py:110-114`'s `ece_split` conditions its low/high split on the *calibrated* `p`, so
  the three calibrators are scored on different subsets.** True for isotonic/beta in principle,
  but temperature scaling maps 0.5 → 0.5 exactly, and the narrative claim it supports
  ("temperature trades lows for highs") is about temperature. Not load-bearing.
- **`_build_20.py:126`'s scale detection could mislabel an `emb_only` run at S=32 as S=64.** The
  test is `"gem32" in label or "gem96" in label or "_S32" in vj.parts[-3]`, so a hypothetical
  `emb_only` S=32 probe run would land in the S=64 group. No such run exists on disk, so it is
  latent only. Same file, `:261`: `win[rw-64:rw+128, ...]` silently wraps on a tile within 64 px of
  the window origin — a visual-only cell, and the chosen tiles are not edge tiles.
- **`_build_11.py:306-307`'s bare `except Exception` around the feature-importance cell would
  silently leave a stale committed PNG in place.** It would, but the figure is regenerated in the
  same commit as the notebook in every instance I checked, so it has not bitten. Low enough that I
  did not file it.
- **Notebook 09's group-leak assertion could be false assurance.** It is not: it asserts
  train∩test = ∅ *and* that the materialised `X_*_fold{k}.parquet` obs-id sets equal the split
  metadata's (`:290-295`), which is a real check. Its limits are that it compares
  `meta['split_hash']` to `pkg['split_hash']` rather than recomputing either (so it cannot catch
  **R04**'s stale-package case at tile level), and that it runs on `config.yaml` only — the
  go-forward **v2** dataset has no equivalent QA notebook. Recorded as a coverage gap, not a defect.
- **Notebook-referenced artifact paths might no longer exist (broken regeneration).** I extracted
  every literal repo-relative path and every `FIG/LEGB/BO/PROBE/MODELS_ROOT/FIG_DIR / "…"`
  fragment from all 21 `_build_*.py` and existence-checked them: **zero** missing in the working
  tree. (They resolve only because `models/`, `dataset*/` and `cache*/` are present locally and
  gitignored — the fresh-clone limitation is the already-refuted `models/` item.)
- **`.gitignore`'s global `*.tif` / `*.tiff` / `*.img` rules might be excluding something needed.**
  `git status --ignored` over `tests/` and `reports/figures/` returns only `__pycache__` and the two
  deliberately-named exclusions, so nothing load-bearing is silently dropped.
- **`models/` being gitignored means the shipped deliverable is unversioned.** True in fact
  (`models/deployable/{86c51a5dca220f63,calibration.npz,parity_ref.npz}` is untracked while the
  *aborted* `models/deployable_f_center/` was force-added at `131e6e1`), but `SHERLOCK_RUN.md:102`,
  `:112`, `:133-135` document the tar/rsync transfer as the intended workflow. Not a defect; noted so
  a future session does not re-file it.
- **Notebook 24 §2b overlays the probability/binary panels on the abundance mosaic's `ext`.** If the
  `*_prob.tif` and `*_abundance.tif` tile sets differed, the panels would misregister — checked on
  disk: 26 abundance / 26 prob / 26 prob_raw, identical sets.
- **`_build_10.py:793` uses `presence_auc` for the within-image arm and `auc` for the LOIO arm.**
  Looks like an apples-to-oranges paired delta, but `:754` renames `presence_auc` → `auc` when
  building the LOIO baseline, so both sides are the same statistic. (The use of presence AUC at all
  is R02/`invariants`' historical-notebook carve-out, not re-filed.)
- **`_build_19.py:140-143` re-implements pooled PR-AUC and precision@5 % instead of reading the
  artifact.** It does (`int(0.05*n)` + `argsort` vs `src/modeling/evaluate.py:303,307`'s
  `int(round(k_frac*n))` + `argpartition`), and the W2 gate at `:202` compares this recomputation
  against an artifact-computed `pooled_pr_auc`. But both call `average_precision_score`, and the `k`
  difference is at most one tile out of tens of thousands — not enough to move a ±0.03 gate.
- **`_build_12.py:149-155` re-implements the abundance binning.** Right-closed `lo < y <= hi` with a
  `y <= 0` zero bin; measure-zero disagreement with `src`'s edges on continuous `fractional_area`, and
  the notebook's top-bin fallback is actually *safer* than the `src` version `evaluate-4` flags.
- **`_build_23.py:164-167` uses different bin edges from `ABUNDANCE_BIN_LABELS`.** Deliberate — it is
  a compression diagnostic with its own `3e-2` split, and it is stated as such in the surrounding
  markdown; nothing downstream joins the two binnings.
- **Notebook 28's η² target moves from "≲ 0.03" (`_build_28.py:243`, `:253`) to "0.05"
  (`:343`, `:361-369`).** Real, but it mirrors `PLAN_StripingArtifact.md:129/149` → `:200/:242` and
  `PLAN_FBuild.md:46`, which records the 0.05 reopening bar as a Brian ruling on 2026-07-09b; and the
  related "0.0505 crosses the 0.05 bar" reading is already covered by `stats-fallacies-3` (every
  tolerance in this programme is ±0.02 with no sampling uncertainty) and by that file's
  rotation-null refutation at its line 368.
- **`reports/figures/fbuild_stagec_{attribution,graph,lambda,missing_frames}.csv` were not
  regenerated in the abort commit while `scripts/f_region_stagec.py` changed 66 lines.** Checked the
  diff: `41a6f26` *did* rewrite `fbuild_stagec_{offsets,watchlist,lean_guards}.csv` and
  `fbuild_trend_guard.csv` in the same commit, so the script demonstrably ran; the four unchanged
  files are outputs the `lcv`→`pfree` swap does not touch.
- **`_build_21.py:219`'s `map_pilot_*_east.png` glob can never pick the calibrated pilot render.**
  True, but notebook 21 §3 is explicitly about the raw `P(rich)` map; the calibrated pilot is
  notebook 23's subject.
- **Unseeded RNG behind a reported number.** Every RNG in `notebooks/_build_*.py` is seeded
  (`_build_08.py:582,883,1006`, `_build_10.py:272,790`); the only nondeterminism left is tie-breaking
  inside `argsort`/`argpartition`, which is deterministic for a fixed input.
- **Committed `reports/figures/*.csv|json` whose producer changed afterwards.** I ran the whole
  131-artifact matrix (artifact last-commit date vs. every tracked `.py` that names it). Every hit
  was either a *consumer* misidentified as a producer (`_build_28.py` reads the F CSVs), an f-string
  filename my grep could not see, or R12's already-filed `fbuild_abort_*`. No genuine
  producer-changed-after-artifact case survived.

## Verified clean

*(pass 2 additions first, then pass 1's list unchanged)*

- **Notebook 28's abort banner arithmetic** — every one of the 20 numbers in the FINAL VERDICT
  table, plus the "1.9–3.1× less stable" and "1.13/1.62/1.26×" ratios, reproduces from
  `reports/figures/fbuild_abort_{level_vs_labels,level_per_tile}.csv` (see *Refuted*). The banner is
  a faithful summary of an artifact with no producer (**R12**), not a mis-summary.
- **`_build_20.py`** (the Fang-ViT probe / frozen-recipe notebook) reads `verdict.json` artifacts
  only; its one hardcoded Tier-1 reference is a **pinned** config hash
  (`models/lightgbm_classification/99de85c1ad2a72e6/scale_S64_tfa_gt_1e-2`, `:251-252`), not an
  mtime pick — so notebooks-1's mechanism does not reach the recipe-freeze evidence.
- **`_build_11.py`'s `artifact_dir`** (`:384-396`) — the snapshot-filtered artifact selector — is
  correct and is the pattern notebooks-1 asks for (see *Correction* above).
- **`_build_08.py`** is a good invariant-10 citizen: every quantity is a column read from
  `src.features.load_features` / `src.labeling.load_labels`; no feature math is re-implemented, and
  its documented shadow thresholds (`mode − 20` / `mode − 35` / `mode + 30`, `:375-380`) match
  `config.yaml:129-131` and `src/features.py:156-158` exactly.
- **`_build_13.py`'s manifest read** (`:175`, `pd.read_csv('hirise_40_vclaire.csv')`) is the same
  file `config_v2.yaml:6` designates, and its two sweep directories (`:177`, `:186`) are pinned
  timestamps — no mtime resolution anywhere in notebook 13.
- **`_build_23.py`'s Stage-2 statistics** — every table is a paired per-image Spearman + Wilcoxon
  over `obs_id` groups with the `nunique() > 1` degeneracy guard applied on both sides
  (`:315-317`, `:376-378`, `:425-429`, `:476-478`, `:524-531`), and the two places where a
  median-of-medians and a paired test disagree are called out *in the prose* (`:462-463`,
  `:576-577`) rather than quoting the favourable one.
- **`_build` ↔ `.ipynb` commit coupling.** For all 21 pairs, every commit that touched a `_build_NN.py`
  also touched its `.ipynb` in the same commit, and no `_build` has a later commit date than its
  notebook. (Independently reproduces `invariants.md`'s stronger cell-by-cell regeneration result.)
- **Notebook execution census.** 26 of 28 notebooks carry full outputs with monotonically increasing
  `execution_count` and **zero** `output_type: "error"` cells; the two exceptions are notebooks 12 and
  13 (invariants-3).
- **`_build_25.py`** is a model citizen for invariant 10: every quantity comes from `src/striping.py`
  or from `scripts/striping_frame_blocks.py`'s committed CSVs; the notebook only plots and reads.
  Likewise `_build_15`/`_build_16` (`src.stage7d_pooled`), `_build_22` (banked verdicts only, and
  `evaluate.md:283` independently checked its `per_bin_curve` call sites), `_build_26`/`_build_28`
  (CSV readers).
- **`_build_24.py`'s metres→degrees conversions.** `R = 3396190.0; dpm = 180/(π·R)` at `:117`, `:201`,
  `:275`, `:337`, `:483`, `:562` — correct for the clon_0 equirectangular mosaic (`x = R·λ` at
  standard parallel 0) and consistent across all six sites; `tile_to_box` at `:81-86` parses both
  `E…`/`W…` and negative-east ids correctly.
- **`_build_24.py`'s `_coarsen`** (`:570-572`) is a genuine nodata-aware block mean (`np.nanmean` over
  a reshaped view) with the trailing partial blocks correctly dropped.
- **Seeds and determinism** — see *Refuted*.
- **Hardcoded absolute paths** — only the three PowerShell invocation examples in `_build_14/15/16`
  and three dead markdown links to the user's private memory directory in `_build_12.py:288,705,860`
  (also found by `invariants`); no code path depends on them.

## Coverage note

### Pass 2 (this session)

**Read in full:** `notebooks/_build_09.py`, `_build_20.py`, `_build_23.py`, `_build_28.py`,
`scripts/probes/_diag_within_image_deltas.py`, `.gitignore`, and the four sibling area files that
defer to this one (`invariants.md`, `evaluate.md`, `stats-fallacies.md`, `numerics.md` — the
relevant sections).
**Read in relevant part:** `_build_10.py:690-900` (the within-image section, line by line),
`_build_11.py:40-500`, `_build_12.py:400-540`, `_build_13.py:36-560`, `_build_08.py:81-400`,
`src/dataset.py:120-320` and `:400-460` (quadrant construction), `scripts/sweep_within_image.py`,
`src/modeling/evaluate.py:230-300`, `docs/modeling_results.md` §§9.4–9.5, §11.4–11.6,
`PROMOTION_QUEUE.md` Problem 5 + P4.
**Grepped only:** `DECISIONS.md` by term (`within-image`, `quadrant`, `within ≈ LOIO`,
`_diag_within_image_deltas`, `lift`, `normalised lift`, `base_rate`, `P4`, `bc_ge_1`),
`PLAN_Stage5c.md`, `PLAN_ModelImprovement.md`.

**Measurements I ran** (read-only w.r.t. the repo; scratch files only, `conda run` with a temp
script per invariant 9): recomputation of all 20 abort-banner statistics from
`fbuild_abort_{level_vs_labels,level_per_tile,per_obs_skill}.csv`; H1/H2 skill deltas from the five
`f_leg_b_loio_summary_*.csv`; presence-AUC recomputation from `predictions.parquet` for the v1 and
v2 `within_image_4fold` and LOIO `lightgbm_two_stage` runs at S = 8/16/32/64 (quadrant-restricted,
quadrant-pooled and whole-image variants, ~1.1 M tile rows), with quadrants re-derived from
`dataset*/splits/within_image_4fold.json`; per-fold base rates, lift ceilings, precision@K and
headroom fractions for `bc_ge_1` vs `fa_gt_1e-2` across all four scales and all five v2 binary
sweeps; a notebook-path existence sweep over all 21 `_build_*.py`; `git status --ignored` over
`tests/`, `reports/figures/`, and each `reports/*` subdir. No notebook execution, no training, no
sweep, no map build, no network, no CTX/HiRISE imagery.

**Two low-severity items I did not file as findings** (below the bar, recorded so they are not
re-hunted): `_build_08.py:353-355` draws a dotted reference line from `(0, 0)` to
`(q99.9(grad_mag_mean), q99.9(grad_mag_p99))` and legends it **`'y=x'`** — since `p99 ≥ mean`
pointwise the two quantiles differ substantially, so the line's slope is ≫ 1 and points that lie
*above* true `y = x` render below the labelled line, in a figure whose caption reads "p99 catches
what mean misses". And `_build_11.py:306-307`'s bare `except Exception` around the
feature-importance cell (see *Refuted*).

**Could NOT check in pass 2:** (1) whether the matched within-vs-LOIO Δ in notebooks-4 survives at
`buffer_tiles ≥ 1` — that needs a re-run of `within_image_4fold`, which is outside the rules of
engagement, so the matched Δ stands only as an upper bound; (2) the `classification`/`bc_ge_1` half
of the §9.4 table (I measured `two_stage` only — the mechanism is identical but the numbers are
not); (3) `scripts/probes/*` beyond `_diag_within_image_deltas.py` and `_stage7_feasibility.py`,
still the largest unopened surface in the repo and the origin of most DECISIONS numbers;
(4) notebooks 01–06 and 18, which have no `_build` source.

### Pass 1

**Read in full:** `notebooks/_build_28.py`, `_build_25.py`, `_build_24.py`, `_build_22.py`,
`_build_21.py` (§2-4), `.gitignore`, and `docs/review_2026-07-31/invariants.md`.
**Read in relevant part:** `_build_10.py` (`:96-130`, `:240-420`, `:470-620`, `:686-900`, `:985-1010`),
`_build_11.py` (setup + the two mtime globs), `_build_12.py:110-240`, `_build_13.py:280-600`,
`_build_14.py:88-390`, `_build_19.py:88-230`, `_build_23.py:40-240`,
`scripts/probes/_stage7_feasibility.py` (docstring + all `to_parquet` sites),
`src/modeling/evaluate.py:286-312`, `src/pds_labels.py` (signatures),
`docs/modeling_results.md` §§3.1-3.2 and the feature-importance section, `docs/modeling.md:370-395`,
`docs/compositional.md:218-235`, `SHERLOCK_RUN.md:95-145`.
**Grepped only:** `DECISIONS.md` by term (`partial`, `dust_summary`, `notebook 12/13`,
`629276139c22da68`, `config_hash`, `st_mtime`, `qmatch`), `PLAN_StripingArtifact.md`,
`PLAN_Compositional.md`, `PROMOTION_QUEUE.md`; `_build_07/08/09/16/17/20/26/27` (imports, savefig
targets, statistical calls).

**Measurements I ran** (read-only w.r.t. the repo; scratch files only): an execution-state census over
all 28 `.ipynb`; a `_build`↔`.ipynb` commit-coupling census; a 131-artifact producer-staleness matrix
over `reports/**/*.{csv,json,npz}`; `lgb.Booster` split-gain aggregation over the 9 v1 and 38 v2
`lightgbm_tweedie` S=8 boosters; snapshot/fold-count reads on four `models/**` run directories; a
tracked-file size census; `git log`/`git show` on `41a6f26`, `c8d68cd`, `131e6e1`, `478293c`. No
notebook execution, no training, no sweep, no map build, no network, no CTX/HiRISE imagery.

**Could NOT check:** (1) whether the *other* four mtime-resolved panels in notebook 10
(`10_pred_vs_true_loglog.png`, `10_per_bin_rmse.png`, `10_binary_calibration.png`, the spatial
truth-vs-pred rows) show v1 or v2 content — the mechanism and the dated directories say v2 for at
least the two-stage S=8/S=64 rows, but I proved it pixel-for-pixel only for the feature-importance
figure; (2) whether recomputing notebook 14's discriminator as a true Spearman partial correlation
still gives p < 0.05 — that needs `cache_v2/stage7/test_b_per_tile_*.parquet`, which I did not open
(it is derived from HiRISE COLOR imagery and outside the read budget I set myself); (3) notebooks
01-06 and 18, which have no `_build` source and whose logic I only skimmed — notebook 05 in particular
is 86 MB and I read only its metadata; (4) whether the 26 byte-identical `_build`→`.ipynb`
regenerations were actually *executed* from the sources they sit beside (the same lower-bound caveat
`invariants.md` records); (5) `scripts/probes/*` beyond `_stage7_feasibility.py`, which remain the
largest unopened surface in the repo.
