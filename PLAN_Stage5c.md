# PLAN — Stage 5c: within-image k-fold cross-validation

**Status:** **shipped 2026-05-27**. Sweep at
`models/_sweep_within_image/20260527T175437Z/`. Diagnostic answer:
**signal-floor branch** — within-image AUC is statistically
indistinguishable from LOIO at all 8 (variant, scale) cells (mean Δ
ranges −0.005 to +0.037, 95 % CI always brackets zero, Wilcoxon
*p* ≥ 0.31). See `DECISIONS.md` 2026-05-27 Stage 5c entry and
`docs/modeling_results.md` §7 for the full evidence and the
recommendation update. Three AskUserQuestion answers pinned (§10 +
the multi-scale-quadrant resolution).

This plan extends [PLAN_Stage5](PLAN_Stage5.md) — Stage 5 packaged the LOIO
split that measures *between-image* generalisation. Stage 5c adds a parallel
**within-image k-fold spatial split** that measures *within-image*
generalisation — train on three spatial quadrants of an image, test on the
fourth, rotate. The two schemes live side by side in `dataset/packaged/`;
modeling code selects which to consume at runtime.

**Why this matters — this is a diagnostic experiment, not a model
improvement.** Both Week 3 modeling sweeps ([modeling_results.md](docs/modeling_results.md))
converged on the same ceiling: regression mean AUC +0.526 (12/12 cells above
chance), binary classification mean AUC +0.514 (7/12 cells above chance),
best individual cell ~0.56. Two independent target framings (regression with
three losses, binary classification at three thresholds) gave statistically
equivalent answers, which is consistent with the data-quantity hypothesis
(modeling_results.md §6.5: "the 9-image LOIO dataset is at its information
ceiling"). Stage 5c is the single test that falsifies or confirms that
hypothesis:

- **If within-image AUC stays ≈0.55** like LOIO — the signal floor is real
  at 5 m / pixel CTX texture. More HiRISE images are not the unlock.
- **If within-image AUC reaches ≈0.7+** — per-image generalisation is the
  binding constraint. More HiRISE images (especially geographically diverse
  ones) become a strong recommendation, and the existing modeling stack is
  fundamentally sound.

This is a binary diagnostic; the deliverable is one number per (model,
scale) plus a 1-paragraph addition to modeling_results.md.

**Scope** — Stage 5c is a *split-scheme* extension. The Stage 4b feature
extraction, the Stage 5 packaging algorithm, the Stage 5b modeling stack,
and every existing test all remain unchanged. What changes:

- a new `within_image_kfold` scheme in [src/dataset.py](src/dataset.py)
  alongside the existing `loio_9fold` and `loio_3fold_balanced`,
- a new `kind: "within-image"` value in the split-metadata JSON schema
  (alongside `"leave-image-out"`),
- a small extension to [src/modeling/loaders.py](src/modeling/loaders.py) so
  the LOIO harness can also iterate within-image folds (the per-fold object
  shape stays the same),
- a new fan-out script `scripts/sweep_within_image.py` that runs the same
  two models we already have evidence for (`lightgbm_two_stage` regression
  + `lightgbm_classification` at `bc_ge_1`) at the matched scales,
- a new section in [notebooks/10_modeling_qa.ipynb](notebooks/10_modeling_qa.ipynb)
  rendering the within-image vs LOIO comparison,
- a new section in [docs/modeling_results.md](docs/modeling_results.md)
  reporting the headline number and updating the priority of "more HiRISE
  images" based on the answer.

---

## 1. Inputs

| Input | Path | Used for |
|---|---|---|
| Per-image labels | `dataset/labels/{ObsId}.parquet` | Tile-key columns (`ti`, `tj`, scale_idx) for spatial partition |
| Per-image features | `dataset/features/{ObsId}.parquet` | Same as Stage 5 packaging |
| Manifest | `hirise_priority10.csv` | Restrict to non-empty-truth images (§3) |
| Per-image label provenance | `dataset/labels/{ObsId}.json` | Inventory metadata reused from Stage 5 |

No new data dependencies. ESP_065711_1545 (empty truth) is excluded — within-
image CV needs positives in the test set to be meaningful.

## 2. Outputs

```
dataset/splits/within_image_4fold.json     # split-scheme JSON, kind="within-image"
dataset/packaged/within_image_4fold/
    X_train_fold{k}.parquet                # k indexes (ObsId, quadrant) pairs
    y_train_fold{k}.parquet
    X_test_fold{k}.parquet                 # one image's one quadrant
    y_test_fold{k}.parquet
    groups_train_fold{k}.npy
    groups_test_fold{k}.npy
    metadata.json                          # ObsId-and-quadrant -> fold_idx map
```

Total fold count at the default `n_folds_per_image=4`:
- 8 images (9 priority10 minus empty-truth) × 4 quadrants = **32 folds**.

`models/_sweep_within_image/{ts}/{summary,aggregate}.parquet` for the
cross-(model, scale, image) aggregate; per-(model, scale) artifacts under
`models/<variant>/<config_hash>/scale_S{n}_within/` (parallel to the
existing `scale_S{n}/` LOIO artifacts).

## 3. Spatial partition algorithm

Each image's tile grid is partitioned into `n_folds_per_image=4` non-
overlapping quadrants by (ti, tj) median split. For an image with
`ti ∈ [ti_min, ti_max]` and `tj ∈ [tj_min, tj_max]`:

```
ti_mid = median(ti)              # computed per-image, per-scale
tj_mid = median(tj)
quadrant(ti, tj) = 2 * (ti >= ti_mid) + (tj >= tj_mid)   # values 0..3
```

Median split (rather than midpoint of range) gives roughly equal tile counts
across quadrants, which matters at S=64 where each image has only
~700–1500 tiles. At S=8, quadrants have ~12k–18k tiles each (plenty).

**Buffer zones (optional, default off).** A 1-tile buffer at quadrant
boundaries — i.e., drop tiles whose (ti, tj) is within 1 of the median in
either dimension — provides extra protection against adjacency leakage at
the cost of ~5–10 % of training tiles per fold. Defaults to off because the
texture/illumination signal at a single CTX pixel (~5 m) is much smaller
than the smallest tile (S=8 = 40 m), so 1-tile-distant tiles are already
roughly independent. Configurable via `buffer_tiles: int = 0` in the
scheme params for sensitivity analysis.

**Multi-scale handling.** The partition is computed *per-scale* — at S=8 the
median (ti, tj) is computed over S=8 tiles only, etc. This guarantees that
the four spatial quadrants of S=64 are coherent with the spatial coverage
of the 64 (= 8×8) S=8 tiles they contain.

**Pure within-image.** Training data for a fold is the OTHER THREE
quadrants of the SAME image — no cross-image data. This is what makes the
experiment diagnostic: a model that does well here cannot be relying on
inter-image transfer.

## 4. Scheme metadata + `src/dataset.py` extension

The existing `build_split` dispatches on a `stratification` string to
choose between LOIO and balanced-k-fold. The new scheme adds a third
strategy with a different `kind` field in the metadata:

```python
# new helper alongside the existing _assign_loio_9fold / _assign_size_balanced_kfold
def _assign_within_image_kfold(
    inventory: pd.DataFrame,
    *,
    n_folds_per_image: int,        # default 4 -> 2x2 spatial partition
    buffer_tiles: int,             # default 0
    labels_dir: Path,              # need to read per-image (ti, tj) to compute medians
    excluded_obs_ids: list[str],   # default [EMPTY_TRUTH_OBS_ID]
) -> list[dict]:
    """Return a flat list of folds. Each fold dict carries:
        - test_obs_id (single ObsId; the image being tested)
        - test_quadrant (int in 0..n_folds_per_image-1)
        - test_tile_predicate (a serialised description of which (ti, tj) tiles are
          in this fold's test set, per scale -- written to metadata so package_split
          can reproduce it deterministically without re-reading the labels parquet)
        - train_obs_ids (always == [test_obs_id]; the other 3 quadrants of the SAME image)
    """
```

Split-metadata JSON schema gains:

```json
{
  "name": "within_image_4fold",
  "kind": "within-image",
  "n_folds_per_image": 4,
  "buffer_tiles": 0,
  "manifest_obs_ids": [...],
  "excluded_obs_ids": ["ESP_065711_1545"],
  "folds": [
    {
      "fold_idx": 0,
      "test_obs_id": "ESP_039820_1750",
      "test_quadrant": 0,
      "quadrant_definitions": {
        "8":  {"ti_mid": 134, "tj_mid": 89},   # one median per scale
        "16": {"ti_mid": 67,  "tj_mid": 45},
        ...
      },
      "n_test_tiles_per_scale": {"8": 12453, "16": 3204, "32": 803, "64": 198},
      "n_train_tiles_per_scale": {...},
      "test_summary": {"BoulderLabel": "unknown", "frac_mean_S8": 0.0008},
    },
    ...
  ]
}
```

`package_split` is extended to handle `kind: "within-image"`. The main
change: for each fold, instead of selecting whole ObsIds for train/test,
filter the single ObsId's rows by the per-scale `(ti, tj)` median split
recorded in `quadrant_definitions`. The existing per-fold parquet write
path is unchanged.

## 5. Sweep driver

`scripts/sweep_within_image.py` (new, ~150 LOC) mirrors `scripts/sweep.py` /
`scripts/sweep_binary.py`:

```
Usage:
    python scripts/sweep_within_image.py                         # both models, all scales
    python scripts/sweep_within_image.py --variants lightgbm_two_stage
    python scripts/sweep_within_image.py --scales 2 3            # S=32 + S=64 only
```

Fixed: this sweep is diagnostic. Only the two models we have strongest
evidence for are run by default:
- `lightgbm_two_stage` (best regression Spearman at S=64)
- `lightgbm_classification` at `bc_ge_1` (best binary AUC at S=32, S=64)

Both at scales 0, 1, 2, 3 = 8 (variant × scale) configurations. Each runs
32 folds (8 images × 4 quadrants). Total fits per sweep: 8 × 32 = 256.
LightGBM fits are seconds at this dataset size; expected total sweep time
~10 minutes.

Writes to `models/_sweep_within_image/{ts}/{summary,aggregate}.parquet`
(one summary row per fold, one aggregate row per (variant, scale, ObsId)).
Per-(variant, scale, ObsId, quadrant) artifacts under
`models/<variant>/<config_hash>/scale_S{n}_within/`.

## 6. Headline metrics

The per-fold metrics are the same as the existing LOIO sweep (regression:
Spearman, AUC, RMSE; classification: AUC, Brier, lift), computed
identically. The aggregation differs:

- **Per-image AUC**: average the 4 within-image folds for one image.
- **Per-class average**: mean per-image AUC across the 6 Boulder-rich
  images. This is the headline number directly comparable to LOIO.
- **LOIO baseline overlay**: for each (model, scale), report
  `within_image_AUC − LOIO_AUC` per image. A positive delta is the
  experimental finding "data quantity is the bottleneck."

Single-fold-level metrics on a within-image quadrant test set are noisier
than LOIO (fewer tiles per test fold — 25 % of an image vs all of an
image). The 4-fold aggregation per image recovers most of that.

## 7. Notebook integration

New section appended to `notebooks/10_modeling_qa.ipynb` (built by
`notebooks/_build_10.py`):

- **Per-image AUC table** — 8 rows (one per image), 2 model columns × 4
  scale columns. Cells coloured by AUC; LOIO baseline annotated.
- **Per-image AUC bar chart** — grouped bar showing within-image vs LOIO
  for each (image, model, scale) cell. Visual answer to "did within-image
  CV produce a stronger signal?"
- **"Lift over LOIO" summary** — single table reporting
  `mean(within_image_AUC − LOIO_AUC)` per (model, scale), with a bootstrap
  95 % CI on the delta. The headline number.

Existing LOIO + binary sections unchanged.

## 8. Tests

New `tests/test_within_image_split.py` (~150 LOC):

- `test_within_image_4fold_partitions_each_image_into_4_quadrants` — on a
  synthetic 100×100 tile image, each fold's test set has ~25 % of the
  tiles and the union is the full image.
- `test_within_image_quadrants_dont_overlap` — pairwise intersection of
  every fold's test set is empty.
- `test_within_image_train_is_same_image_only` — `train_obs_ids ==
  [test_obs_id]`; no cross-image rows in train.
- `test_within_image_excludes_empty_truth_image` — ESP_065711_1545 is not
  in the manifest_obs_ids of the generated scheme.
- `test_within_image_per_scale_quadrant_coherence` — median (ti, tj) at
  S=8 must place every S=8 tile in the same quadrant as the S=64 tile
  containing it. (Spatial-coherence invariant required for multi-scale
  consistency.)
- `test_within_image_buffer_drops_boundary_tiles` — synthetic image with
  `buffer_tiles=1`, assert that no test-fold row has `ti == ti_mid` or
  `tj == tj_mid` (or within ±buffer).
- `test_within_image_handles_image_with_zero_positives_in_one_quadrant` —
  the model should still fit (binary classifier on all-zero train would
  produce a constant-zero predictor); the harness should tag the fold
  as `is_specificity_only` if test y is constant.
- `test_within_image_split_reproducibility_with_seed` — same inventory +
  seed → identical fold assignment.
- `test_within_image_metadata_records_quadrant_definitions` — the
  metadata JSON contains `quadrant_definitions` with per-scale medians
  enabling deterministic packaging.
- `test_within_image_packaged_round_trip` — build split → package → load
  one fold → row counts match the metadata's `n_test_tiles_per_scale`.

Plus integration tests against the real packaged dataset (slow-marked):

- `test_within_image_4fold_on_priority10_yields_32_folds`.
- `test_run_loio_works_on_within_image_scheme` — confirm the existing
  LOIO runner accepts `scheme="within_image_4fold"` (since the per-fold
  shape is identical).

Target: +10–12 tests; pytest 191 → ~203.

## 9. Sequencing relative to Stage 5 / 5b

Stage 5 is shipped (commit `aa6cd74`). Stage 5b is shipped (commits
`ba7c776`, `5b31171`, `b8ae68a`). Stage 5c is purely additive:

- No Stage 5 code is modified; `src/dataset.py` gains a new branch in the
  existing `build_split` dispatcher.
- No Stage 5b code is modified; `lightgbm_classification` and the
  classification harness run unchanged against the new scheme.
- The existing LOIO sweeps and their artifacts remain valid; nothing in
  the regression or binary sections of `modeling_results.md` changes.

Stage 5c can be reverted in isolation without affecting any prior result.

## 10. Decisions already pinned (AskUserQuestion 2026-05-27)

| Question | Decision | Rationale |
|---|---|---|
| Save the plan as a PLAN doc? | **Yes, save as `PLAN_Stage5c.md`** | Mirrors PLAN_Stage5b.md naming; gives a second-read checkpoint before implementation. |
| Implementation scope | **Proper path — extend `src/dataset.py` with a new split scheme** | Makes within-image a permanent capability alongside LOIO rather than a one-off probe. If the diagnostic answer warrants more experiments (e.g. comparing buffer-zone variants, applying within-image to future image cohorts), the infrastructure is already in place. |

## 11. Open questions to resolve at implementation time

1. **`n_folds_per_image` default.** 4 (2×2 quadrants) is the simplest;
   3×3 = 9 spatial blocks would give finer-grained variance estimates per
   image at the cost of smaller training sets per fold. Recommendation:
   start at 4; document as easily configurable; revisit if 4-fold per-image
   variance is too high to interpret.
2. **Should Boulder-poor and unknown images be included?** Two Boulder-poor
   + one unknown have so few positives per image that some quadrants will
   contain zero positives in test → specificity-only folds. Argument for
   inclusion: the LOIO sweep found Boulder-poor folds had the *highest*
   AUC (0.59), so within-image on the same images is an interesting
   comparison. Argument against: noisy. Recommendation: include all 8
   non-empty images by default; tag the quadrants of Boulder-poor images
   that go specificity-only rather than dropping them.
3. **Comparison statistic.** Per-image `(within_image_AUC − LOIO_AUC)`
   gives 8 paired differences (one per non-empty image). A paired t-test
   or Wilcoxon signed-rank test on those 8 deltas is the natural
   significance check. Recommendation: report both the mean delta and a
   Wilcoxon p-value, plus the bootstrap CI of the mean.
4. **CNN inclusion.** Out of scope (PLAN_Stage5b.md §12: CNN binary loss
   is a separate fix). The CNN's failure on LOIO is structural; running
   it on within-image without first fixing the loss would not be
   diagnostic.
5. **`buffer_tiles` sensitivity analysis.** Default is 0; PLAN §3 argues
   adjacency leakage at S=8 is already weak. A follow-up sensitivity
   sweep at `buffer_tiles=1, 2, 4` would confirm the leakage is small.
   Out of scope for the headline experiment; can be added as a
   secondary sweep if reviewers ask.

## 12. Out of scope (carry forward)

- **More HiRISE images.** This experiment determines whether that
  recommendation is justified; it does not itself add images.
- **Cross-image transfer with shared spatial structure.** A more elaborate
  variant would train on three quadrants of every image (24 quadrants
  total) and test on one quadrant of one image (the image's spatial
  structure is partially "shared" through having other images contribute
  to training). Useful for studying inter-image transfer in detail; not
  needed for the diagnostic.
- **THEMIS validation.** Same status as in Stage 5b — CLAUDE.md §10
  future work, deferred.
- **Calibration recalibration.** The Stage 5b binary classifier has poor
  calibration (modeling_results.md §6.4) that an isotonic-or-Platt-scaling
  layer would fix. Worth doing but orthogonal to Stage 5c.

## 13. Time estimate

| Task | Est. |
|---|---|
| `src/dataset.py` extension (split + package + tests for both) | 45 min |
| Helper: per-image (ti, tj) median computation + buffer handling | 15 min |
| `scripts/sweep_within_image.py` + run | 25 min |
| Run the 256-fit sweep | 10 min |
| Notebook 10 extension + execution | 30 min |
| `docs/modeling_results.md` update (new §6.5 or §7) | 20 min |
| Tests written alongside each component (~10 tests) | 30 min |
| **Total** | **~175 min** |

Implementation order: `src/dataset.py` extension → tests for the splitter
→ run the sweep on one (model, scale) cell to smoke-test the harness →
`scripts/sweep_within_image.py` → full sweep → notebook → docs update.
Tests written alongside each component.
