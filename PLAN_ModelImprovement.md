# PLAN — Improving the HiRISE→CTX rock-abundance model (v2)

**Status:** planned 2026-05-29 (approved). Follows the vClaire v2 modeling A/B
(commit `c8d68cd`, [`docs/modeling_results.md`](docs/modeling_results.md) §9).
Mirrored in the session plan file; this is the in-repo, ROADMAP-indexed copy.

## Context

v2 is built end-to-end and the A/B vs v1 is committed. Result: denser labels lifted
regression **Spearman ~3–10×** (v1 ≈0 → v2 +0.10..+0.17) but presence/binary **AUC only
modestly** (0.52–0.55 → 0.55–0.62), and the within-image diagnostic still shows within ≈ LOIO
— a 5 m/px texture floor still binds the *presence* ceiling. v2 is the go-forward dataset.

We want to **improve the model**, in three thrusts: **(A)** analyze & visualize current v2
results to flag concrete problems (thresholds, over/under-estimation, where it hits/misses);
**(B)** improve the CNN, prioritizing task/loss design (the documented cause of v1's
below-chance CNN); **(C)** a tile-scale case study extending the ladder to S128 (640 m), since
v2 metrics rise monotonically to S64.

**Hard constraint:** full v2 (38 images, 3.5 M tiles) is too big to iterate quickly. Build a
**5-image dev set** for fast loops; promote validated winners to full v2.

Reuse, don't rebuild:
- 4 tile scales already swept (S8/16/32/64) in `sweep.py` + `sweep_binary.py`.
- The CNN ([`src/modeling/cnn.py`](src/modeling/cnn.py) `SmallCNNRegressor`) already consumes
  a P×P **context patch** (P=32/64 = 160/320 m) — it *does* use outside-of-tile context — but
  ran on **v1 only** and below chance; `modeling_results.md` §3.3 blames **loss design**
  (log1p+Huber collapses to ~0 on 98%-zero v1 truth). Context patches are **off** in
  [`config_v2.yaml`](config_v2.yaml) (`features.context_patch.enabled: false`).
- v1 prediction diagnostics already in [`notebooks/_build_10.py`](notebooks/_build_10.py)
  (`render_image_row`, `_grid_for_image`, `_render_panel`, per-bin mean_true/mean_pred,
  binary calibration deciles); notebook 11 has none yet.

---

## Phase 0 — Fast-iteration dev harness (prereq for B & C)

Stage 5 discovers images from `dataset_*/labels/` (not the manifest rows —
[`scripts/run_stage5.py`](scripts/run_stage5.py) `discover_obs_ids`), so a dev set is just a
labels/features dir holding 5 images.

**Dev images (density-span + v1 overlap; from Stage-1 valid-polygon counts):**

| ObsId | polygons | role |
|---|---:|---|
| `ESP_055978_2270` | 9.6k | sparsest |
| `ESP_069669_2220` | 35k | v1-overlap, low-mid |
| `ESP_064510_2260` | 81k | mid |
| `ESP_071093_2210` | 245k | v1-overlap, high-mid |
| `ESP_068483_2280` | 727k | densest |

1. `hirise_5_dev.csv` — those 5 rows copied from `hirise_40_vclaire.csv`.
2. `config_v2_dev.yaml` — copy of `config_v2.yaml` with `manifest: hirise_5_dev.csv`,
   `output_dir: ./dataset_v2_dev`, `cache_dir: ./cache_v2_dev`, `splits` → `loio_nfold`
   n_folds=5 / `within_image_4fold` n_folds=20.
3. Pre-seed: junction `cache_v2_dev/*` → `cache_v2/*` (all per-image caches exist for these 5).
   Copy the 5 `dataset_v2/labels/{obs}.parquet` + `dataset_v2/features/{obs}.parquet` into
   `dataset_v2_dev/`.
4. `run_stage5.py --all --config config_v2_dev.yaml` → 5-fold LOIO + 20-fold within-image, in
   seconds. **No Stage 1–4 re-run** (reused).

Dev modeling uses existing flags: `sweep.py --dataset-dir dataset_v2_dev --scheme loio_nfold`.

---

## Phase A — Analyze & visualize current v2 results (flag problems)

Read-only over the **existing full-v2 sweeps** (no retrain). Extend
[`notebooks/_build_11.py`](notebooks/_build_11.py), porting v1 diagnostics from `_build_10.py`
**minus the dense-polygon overlay** (v2 gpkgs ≤727k polygons — outlines would hang; rasterized
grids only).

1. **Example/spatial predictions** — truth `fractional_area` heatmap vs `lightgbm_two_stage`
   pred vs `lightgbm_classification` (bc_ge_1) probability, S=64 and S=32, ~4 images. Reuse
   `render_image_row` with polygon overlay removed.
2. **Over/under-estimation** — pred-vs-true log-log scatter; per-bin `mean_true` vs
   `mean_pred` (from each `metrics.json` `per_bin_rmse`); binary calibration deciles + ECE.
3. **Threshold / operating point** — PR curve + threshold sweep for bc_ge_1; surface the
   **coarse-scale saturation** (S=64 v2 ~93% positive → whole images single-class → AUC on
   25–26 of 38). Likely motivates a *scale-dependent* "boulder-rich" threshold vs fixed
   `fa_gt_1e-2`.
4. Execute notebook 11 headless; figs → `reports/figures/11_*`. Add a "what's wrong / working"
   read; update `modeling_results.md` §9 if it changes the interpretation.

---

## Phase B — CNN improvements (task/loss design first)

Iterate on **v2-dev**, promote to full v2.

1. **Thread `--dataset-dir` into [`scripts/train_cnn.py`](scripts/train_cnn.py)** (mirror the
   sweep scripts; `iter_loio_folds` + `gather_patches` already accept `dataset_dir`).
2. **Enable context patches on dev:** `config_v2_dev.yaml`
   `features.context_patch.enabled: true` (sizes [32,64]); re-run
   `run_stage4b.py --all --config config_v2_dev.yaml`.
3. **Baseline:** existing `SmallCNNRegressor` on v2-dev — does lower zero-inflation lift it off
   below-chance?
4. **The fix — classification CNN** (primary lever): add `SmallCNNClassifier` to
   `src/modeling/cnn.py` — same backbone, `BCEWithLogitsLoss` + `pos_weight`, `predict()` →
   sigmoid prob, `predict_presence_prob()` → same. Drive via
   `run_loio(task="classification", binarize=...)` ([`src/modeling/evaluate.py`](src/modeling/evaluate.py))
   with `bc_ge_1` ([`src/modeling/binary_target.py`](src/modeling/binary_target.py)).
   **Head-to-head vs the GBM classifier** at matched scale — CNN > GBM AUC ⇒ recoverable
   spatial signal the hand-crafted features miss (decides whether a Stage-4c feature push is
   worth it).
5. **If promising:** two-stage CNN (presence+magnitude) and/or larger context patch **P=128
   (640 m)** (`features.context_patch.sizes_px: [32,64,128]`, re-run Stage 4b dev). Winner →
   full v2, compare to §9.3.

**Dev outcome (2026-05-29) — CNN NOT promoted (Brian's call).** Within-image dev (20 folds)
head-to-head, bc_ge_1 AUC: GBM **0.541** (S32) / **0.538** (S64) vs CNN **0.474** (S32 P32) /
**0.503** (S32 **P128 wide context**) / **0.546** (S64 P64). The classification reframing fixes
v1's below-chance *collapse* (the regression CNN still collapses, ρ≈0), but the CNN **does not
beat the GBM** at any patch size, and wider outside-of-tile context (P=128) helps only a little
(0.474→0.503), not past the GBM — consistent with the §9.4 texture-floor finding. Dev error
bars are wide (±0.09–0.20), so this is "no evidence the CNN wins," not a hard inequality.
Decision: keep the new `SmallCNNClassifier` + smoke test, document the negative result, do **not**
spend the full-v2 CNN (~60 GB patches + hours).

**DEFERRED — more CNN variants to try later** (Brian wants these revisited): two-stage CNN
(presence + magnitude heads), a larger/deeper architecture (the current net is ~30k params),
more epochs / LR schedule, a multi-scale two-patch input (tile + wide context together rather
than one fixed P), and possibly attention. Revisit on dev before any full-v2 commitment.

---

## Phase C — Tile-scale case study (extend ladder to S128)

1. On **dev**: `labeling.tile_sizes_px: [8,16,32,64,128]` (×2 ladder, S128 = 640 m). Re-run
   Stage 4 → 4b → 5 dev (reads cached detections + windows; fast).
2. Re-sweep GBM regression + binary (+ best CNN) across all 5 scales; plot **metric-vs-scale**
   (Spearman, AUC) + the resolution-vs-signal trade-off (note S=64 is near image-saturation).
3. **Decision:** if S128 keeps lifting, add `128` to full `config_v2.yaml`, re-run Stage
   4/4b/5 + sweeps on full v2, update splits `n_folds` + `modeling_results.md`. Else document
   S64 as the operating scale.

**Dev outcome (2026-05-29) — S128 lifts ranking; HELD as dev-only (Brian's call).** Within-image
dev (20 folds): two_stage Spearman climbs 0.118→0.130→0.187→0.263→**0.406** (S8→S128); bc_ge_1
AUC stays flat ~0.54–0.57. So coarser tiles markedly improve abundance *ranking* (not the
presence ceiling). Brian held promotion — full `config_v2.yaml` stays `[8,16,32,64]`. The
scale-extension plumbing is in place (`SCALE_TO_FACTOR_FROM_FINEST` +16, sweep `SCALE_TILE_PX`
+4:128, GLCM 128 entry in `config_v2_dev.yaml`), so a future full-v2 S128 confirmation is a
config flip + re-run. See `docs/modeling_results.md` §10.2.

---

## Critical files

**New:** `config_v2_dev.yaml`, `hirise_5_dev.csv`, `SmallCNNClassifier` in `src/modeling/cnn.py`.
**Modify:** `notebooks/_build_11.py` (Phase A), `scripts/train_cnn.py` (`--dataset-dir`),
`config_v2.yaml`/`config_v2_dev.yaml` (context patches, tile_sizes_px — dev first, full on promotion).
**Reuse:** `render_image_row`/`_grid_for_image`/`_render_panel` (`notebooks/_build_10.py`);
`gather_patches`/`iter_loio_folds`/`load_fold` (dataset_dir) (`src/modeling/loaders.py`);
`run_loio(task="classification")` (`src/modeling/evaluate.py`); `BINARY_TARGETS`
(`src/modeling/binary_target.py`); `build_split`/`package_split`/`discover_obs_ids`
(`src/dataset.py`); `pick_sweep` (`src/modeling/sweep_select.py`); `--dataset-dir`/`--scheme`
on `scripts/sweep*.py`; `_diag_target_dist_v1v2.py`, `_diag_within_image_deltas.py`.

## Verification

- **Phase 0:** Stage 5 dev prints 5 LOIO + 20 within folds; dev sweeps run in minutes.
- **Phase A:** notebook 11 executes headless; figs in `reports/figures/11_*`.
- **Phase B:** CNN classification AUC on v2-dev vs GBM bc_ge_1; promote winner to full v2,
  compare to §9.3.
- **Phase C:** metric-vs-scale curve across [8,16,32,64,128]; S128 decision.
- **Tests:** `pytest tests/ -q` green; light smoke test if `SmallCNNClassifier`/dev path added.
- **Promotion rule:** full-v2 (expensive) only after dev validates. Commit per phase — **git
  gated** (`.claude/settings.local.json` allows only geospatial conda python/jupyter), so the
  user reviews + commits.
