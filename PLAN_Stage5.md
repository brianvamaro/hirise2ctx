# PLAN — Stage 5: leave-image-out splits + dataset packaging

**Status:** scoped (not yet implemented). Reads `dataset/labels/{ObsId}.parquet` + (eventually) `dataset/features/{ObsId}.parquet`; emits split metadata + a single packaged dataset view.

**Why this matters** — CLAUDE.md §4 acceptance #5 says "splits must be by image (group-aware), never random tiles." This is the single methodological decision that, if gotten wrong, invalidates every downstream evaluation number. Random per-tile splits leak across tiles within an image because tiles only a few hundred metres apart share illumination, surface composition, and BoulderNet detector behavior. The model would learn the boilerplate of each image instead of generalising.

**Scope already pinned in CLAUDE.md §4 Stage 5** — group-aware leave-image-out, stratify so high-abundance images and the 2 boulder-poor images are spread across folds, generic for a growing set. This plan fills in *how*: stratification rule, fold count, output format, and how the splitter handles edge cases (empty-shapefile image, scale-specific splits, growing manifest).

---

## 1. Inputs

| Input | Path | Used for |
|---|---|---|
| Per-image labels | `dataset/labels/{ObsId}.parquet` | Group structure (one parquet per image) |
| Per-image features (optional) | `dataset/features/{ObsId}.parquet` | Not strictly required for splitting, but consumed by the same packaging step |
| Manifest | `hirise_priority10.csv` | `BoulderLabel` for stratification, `CenterLat/Lon` for optional geographic-stratification extensions |
| Per-image label provenance | `dataset/labels/{ObsId}.json` | Per-image stats (eligible tile counts, n_polygons, coreg shift) — input to stratification |

## 2. Output

`dataset/splits/{name}.json` — JSON describing one named split scheme (e.g. `loio_5fold_balanced.json`). Multiple split schemes can coexist; the modeler picks one by name at training time.

Schema:

```json
{
  "name": "loio_5fold_balanced",
  "kind": "leave-image-out",
  "n_folds": 5,
  "manifest_obs_ids": ["ESP_055714_2270", "ESP_054857_2270", ...],
  "stratification": {
    "method": "boulder_label_balanced",
    "params": {"target_labels_per_fold": {"Boulder rich": 1, "Boulder poor": 1, "unknown": 1}}
  },
  "folds": [
    {
      "fold_idx": 0,
      "test_obs_ids": ["ESP_069669_2220"],
      "train_obs_ids": ["...the other 8..."],
      "test_summary": {"n_tiles_S8": 72821, "BoulderLabel": "Boulder rich", "frac_mean_S8": 0.0008}
    },
    ...
  ],
  "config_hash": "...",
  "written_at_iso": "..."
}
```

`dataset/packaged/{name}/` — a packaged view materialised from the split + labels + features, ready for training:
```
dataset/packaged/{name}/
  X_train_fold{k}.parquet     # features only, with obs_id and tile key columns
  y_train_fold{k}.parquet     # labels only (label columns), same row order
  X_test_fold{k}.parquet
  y_test_fold{k}.parquet
  groups_train_fold{k}.npy    # obs_id-as-int array for the rare case the modeler wants intra-train group structure
  metadata.json               # provenance: scale_idx filter applied, feature set hash, label config hash
```

Optionally a single `dataset/packaged/{name}/all.parquet` with all rows + a `fold_idx` + `split` (train/test) column for ad-hoc analysis. Recommendation: emit this — saves repeated joins downstream.

## 3. Split-construction algorithm (default: `loio_5fold_balanced`)

With 9 ObsIds, the natural fold count is **9 (true leave-one-image-out)** or a **stratified k-fold (k=3 or 5)** that puts each test fold's group composition close to the overall composition.

Recommendation: **k=9 (LOIO)** as the default. With this few images, k-fold doesn't save much compute, and per-fold variance is meaningful information — a fold where the test image is the only boulder-poor case is a different generalisation question than a fold where the test image is boulder-rich. The user / modeler can see this explicitly.

But also emit a **secondary `loio_3fold_balanced` scheme** that bundles 3 ObsIds per fold using a stratification objective (each fold has ~1 rich, ~0.7 poor, ~0.7 unknown). This is for the variance-smoothing case where the user wants more samples per fold.

Implementation: greedy stratified assignment:
1. Group images by `BoulderLabel` (rich / poor / unknown).
2. Round-robin assign images from each group to folds, starting from the largest group.
3. Within a group, shuffle deterministically using `seed: 0` (recorded in metadata).

Edge case: when the manifest grows past 9 images, the splitter takes a `n_folds` config and re-runs the same algorithm. No code change needed.

## 4. Empty-shapefile image (ESP_065711_1545)

ESP_065711_1545 has zero polygons (true zero-truth tiles, valid signal — see DECISIONS.md 2026-05-23). It should appear in folds like any other image. Don't filter it out — its 25,221 finest tiles are real "boulder absent" examples the model needs to learn from. But flag in the split metadata that its `BoulderLabel == "unknown"` and that the modeler may want to treat its evaluation specially (e.g. measure false-positive rate on it alone).

## 5. Per-scale splits

Stage 4 emits 4 scales (40/80/160/320 m). Two paths:

- **One split scheme covering all scales** (recommended) — `(obs_id, scale_idx, ti, tj)` rows from all scales share the same train/test assignment based on `obs_id` alone. Modeler picks which scale(s) to load via a row filter.
- **Per-scale split schemes** — separate fold structure per scale. Strictly more flexible but adds 4x split files and creates a footgun (training at scale S on fold k of S-split and evaluating at S on fold k of S'-split would be wrong).

Going with the first. The split is over images, not tiles, so the scale dimension is orthogonal.

## 6. Module + file layout

```
src/dataset.py               # new — split construction + packaging
scripts/run_stage5.py        # new — driver: build named splits + materialise packages
tests/test_splits.py         # new
notebooks/08_splits_qa.ipynb # new (or whatever the next notebook number is)
```

`src/dataset.py` interface:

```python
def build_split(
    manifest_path,
    cache_dir,             # where labels live (cfg.output_dir / 'labels')
    *,
    name: str,
    n_folds: int,
    stratification_method: str,
    seed: int = 0,
) -> dict:
    """Return the split-metadata dict above. Side-effect: write {name}.json."""

def package_split(
    split_metadata,
    labels_dir,
    features_dir,
    out_dir,
    *,
    scale_filter: list[int] | None = None,
    config_hash: str,
) -> Path:
    """Materialise per-fold parquets ready for training."""
```

Splitter is pure-Python + pandas — fast and easy to test on synthetic manifests.

## 7. Config

Add `splits` block to `config.yaml`:

```yaml
splits:
  default_scheme: loio_9fold
  schemes:
    loio_9fold:
      n_folds: 9
      stratification: none           # 1 image per fold; stratification doesn't apply
    loio_3fold_balanced:
      n_folds: 3
      stratification: boulder_label_balanced
      seed: 0
  scale_filter: null                # null = include all 4 scales; list to restrict
```

## 8. Tests

- `test_loio_9fold_uses_each_image_exactly_once_in_test` — synthetic 9-image manifest.
- `test_stratified_3fold_balances_boulder_labels` — verify the rich/poor/unknown distribution across folds.
- `test_split_reproducibility_with_seed` — same seed → same fold assignment.
- `test_split_grows_with_manifest` — synthetic 12-image manifest; sanity that nothing hardcoded to 9.
- `test_no_obs_id_in_both_train_and_test_in_any_fold` — the only correctness property that actually matters.
- `test_package_split_round_trip` — build split → package → reload → row counts match expectation.
- `test_packaged_metadata_records_provenance` — config_hash + split name + scale_filter are recorded.

Target: +8-10 tests; pytest ~88 → ~96-100.

## 9. QA notebook

- Bar chart: per-image tile count + boulder label, ordered by fold assignment under each scheme.
- For each scheme, a per-fold panel showing the train+test breakdown by `BoulderLabel`.
- Per-fold target distribution (`fractional_area` at finest scale) for train vs test — sanity check that no fold is pathological (e.g. test set is 100% zeros).
- One-line summary: total tiles, train/test ratio under each scheme, group-leak check (assert no overlap).

## 10. Key decisions to surface via AskUserQuestion at execution time

1. **Default fold count** — 9 (LOIO, recommended for honest variance reporting on a small dataset) vs 5 vs 3 (smoother metrics, fewer fold-level edge cases).
2. **Whether to emit the consolidated `all.parquet`** — yes (recommended, easier ad-hoc analysis) vs no (just per-fold files, smaller dataset/).
3. **What to do with ESP_065711_1545** — include in folds with `BoulderLabel == "unknown"` (recommended; its zero-tiles are real signal) vs hold out as a permanent eval-only set (treat as the cleanest false-positive-rate test) vs exclude from training entirely.
4. **Whether the `Boulder rich` images need finer-grained stratification by `frac_mean_S8`** — i.e. do we further split the 6 boulder-rich images by their actual measured abundance instead of trusting the manifest label? Argument for: the manifest label is qualitative; the measured distribution is what matters for modeling. Argument against: with only 6 rich images, slicing further produces folds with 0 or 1 example per bin. Recommendation: punt to Week 3 unless distribution analysis (notebook 06's per-image histograms) shows a clear bimodality among the "rich" group.

## 11. Sequencing relative to Stage 4b

See PLAN_Stage4b.md §10 — recommended order is 4b then 5 so the QA notebook here can show per-fold feature stability sanity checks. Either order is fine functionally.

## 11b. Scaling to a many-images dataset (50-200+ ObsIds)

The manifest will grow. Pinning concrete switchover points so the splitter doesn't silently become wrong:

- **Fold count default flips by manifest size.** ≤15 images: `n_folds = n_images` (true LOIO, cheap and gives per-image variance). 16-50 images: `n_folds = 10` stratified by `BoulderLabel`. 50+ images: `n_folds = 5` plus a permanent held-out evaluation set of 3-5 images chosen for terrain diversity. Encode this as `splits.n_folds: auto` with a documented decision table; user can override.
- **Memory: avoid concatenating all parquets at once.** At 9 images we can load every `dataset/labels/{ObsId}.parquet` into one DataFrame (~50 MB). At 100 images that's ~500 MB labels + several GB features. Switch the loader in `src/dataset.py` to a streaming/chunked pattern — yield `(X, y, group)` per ObsId, never materialize the whole dataset in memory. Implement this now (cheap with 9 images, makes the growth case painless).
- **Stratification becomes more meaningful.** With 50+ images, stratify on (a) `BoulderLabel`, (b) measured `frac_mean_S8` quintiles, (c) lat-band quartiles (so geographic generalization is testable). Add as `splits.stratification.method: multi_factor` once needed.
- **Per-fold packaging.** At 9 images the materialized `dataset/packaged/{name}/` is small; at 100 images it's gigabytes per fold. Switch packaging to **on-demand lazy iteration** (a `Split` object exposes `iter_train_batches()` / `iter_test_batches()` reading the source parquets directly) rather than writing per-fold parquets to disk. Keep the existing `all.parquet` only when manifest size warrants.
- **Sanity check that doesn't change with size:** `test_no_obs_id_in_both_train_and_test_in_any_fold` remains the one correctness property. Add `test_split_handles_100_image_synthetic_manifest` so the growth case has a regression test before it's needed.

The splitter API stays the same across all sizes — only the defaults and internal loading strategy change.

## 12. Open questions (carry forward to PLAN_modeling.md)

- Does the modeler want **leave-one-out CV (n_folds = n_images)** for the headline metric and **stratified k-fold** for hyperparameter selection? Common ML practice, doubles the runtime.
- Should packaging emit the **count_density** column separately as a "compositionality" target alongside `fractional_area`? Cheap and the modeler may want to try predicting density as a stabler target than fractional area in dense regions.
