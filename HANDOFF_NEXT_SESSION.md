# Handoff prompt — next session

> ⚠️ **STALE (frozen at 2026-06-15; banner added 2026-07-02).** This file is no longer maintained;
> live session state = the `project_state_*` memory notes, current phase = [ROADMAP.md](ROADMAP.md),
> running log = [DECISIONS.md](DECISIONS.md). Kept only as a historical snapshot.

**Last updated 2026-06-15 — Calibration/de-compression workstream opened + Stage 0
+ prototypes DONE (committed b96b8f4). Everything below is COMMITTED on branch
`fm-deployable-head-and-map-pilot` (recent: b96b8f4 cal-prototypes, 63fd8b7 cal-Stage0,
3a66f53 figure-rich report, fd38a19 §2.7).** [PLAN_FM.md](PLAN_FM.md) is the active
program; [PLAN_Calibration.md](PLAN_Calibration.md) is the new de-compression plan.

**DONE & committed:** §2.1 freeze, §2.2 productize, §2.4 Tier-2, §2.6 deployable head
+ off-HiRISE map, §2.7 reliability (validated NEGATIVE at n=38 → overlay DEFERRED),
§2.5 model-evidence report (figure-rich: gap-fill headline + basis + gallery + Tier-2
map; Tier-2 deep-dive §8), and **PLAN_Calibration Stage 0 + prototypes**.

**The critical-path bottleneck is still on Brian's side:** any "confirmed" claim
(§2.3) needs the 23 expansion ObsIds (`cohort_expansion_candidates.csv`) run through
BoulderNet — not yet done. The next BUILDABLE pieces (no expansion data), suggested order:

- **Calibration Stage 1 — RECOMMENDED next build** (no GPU). Productize a
  `CalibrationLayer` in `src/`: **isotonic** for Tier-1 P(rich), **quantile-match**
  for Tier-2 abundance (both fit LOIO/all-38, rank-preserving), and wire into
  `predict_window` / the map renderer. Drafts already exist
  (`model_evidence_{gapfill_map,tier2_map}_calibrated*.png`); this makes them a real
  layer. Gates: Tier-1 ECE≤0.05 + global-AUC±0.005; Tier-2 top-bin∈[0.8,1.2] +
  Spearman±0.01. PLAN_Calibration §5 Stage 1.
- **§2.3 declaration** — writable now (pre-data): confirm-then-absorb gates/baseline/
  protocol (PLAN_FM §2.3). Unlocks the held-out stamp for §2.5's headline row.
- **Calibration Stage 2 (L1/L2)** — the cheap L1 swaps (log1p, count-Poisson) are
  RULED OUT (2026-06-15); the remaining levers are **HL-Gauss/quantile head (L1)** and
  **coarser-scale / `min_confidence` label-noise (L2)** — the only lever that raises
  the ranking ceiling. GPU, minutes. PLAN_Calibration §3.
- **§2.5 finish** — the §4 ViT→GeM→MLP schematic figure + fill the `[held-out: pending]`
  headline row once §2.3 lands.

**Calibration findings (committed):** Tier-1 already well-calibrated (ECE 0.06);
ISOTONIC fixes both ends best (→0.014, AUC-exact at deployment; beta = smooth
fallback). Tier-2 compression is TWO-SIDED + intrinsic (aleatoric floor); quantile-
matching de-compresses the marginal (top-bin 0.71→0.87, ranking preserved); cheap L1
swaps don't help. `src/calibration.py` (+9 tests), notebook 23. The map currently has
NO calibration layer wired (drafts only) and NO reliability overlay (deferred).

## What landed in the §2.7 session (2026-06-14b; DECISIONS.md 2026-06-14b)

- `src/reliability.py` — `MahalanobisNovelty` (PCA-whitened top-256) + `KNNNovelty`
  (cosine k=50, subsampled ref); NaN-safe, deterministic; `aggregate_per_image`.
- `scripts/probes/_fm_reliability_validation.py` (LOIO novelty vs frozen per-image
  AUC, writes `reports/figures/27_reliability_validation.png` +
  `reports/reliability/per_image_novelty.csv`); `_fm_reliability_inspect.py`
  (confound diagnosis); `_fm_reliability_smoke.py`.
- `tests/test_reliability.py` (+10, all pass).
- Verdict: NEGATIVE at n=38 → overlay deferred (see top summary + DECISIONS).
- Store-location gotcha confirmed: P96 embeddings live in
  `dataset_v2/fang_embeddings`, pass `dataset_dir=REPO/"dataset_v2"`.

## What landed earlier (§2.6; 2026-06-14; DECISIONS.md 2026-06-14)

**§2.6.A deployable head — `src/modeling/mlp_head.py` (NEW).** The frozen
`mlp_ens3` classifier was only inside the LOIO probe harness; now productized:
`FeatureScaler` (median-impute+zscore parity), `MLPClassifierHead`
(Model-protocol — also drops into the LOIO harness), `DeployableHead` (3-seed
ensemble, train-on-all). `fit(X,y,groups)` rotates one inner-val image PER SEED
for early stopping (every image in-training for ≥2 of 3 seeds); `predict` = mean
seed sigmoid; `save`/`load` persist 3 state-dicts + scalers + `recipe.json`
(frozen cell id, LOIO numbers, config-only `recipe_hash`). **Perf fix applied**
(batch 4096 + train tensor on device once; batch is NOT on the frozen recipe
card → impl-only, LOIO 0.7832 stands). Trainer `scripts/train_deployable_head.py`
builds the all-38 emb-only S=32 matrix by unioning the `loio_nfold` per-fold TEST
slices. **Banked `models/deployable/86c51a5dca220f63/`** (38 imgs, 161k tiles,
76 s; in-sample sanity AUC 0.966 — NOT validation; round-trip 2e-7).

**§2.6 B-E map pilot — `src/mapping.py` + `scripts/map_pilot.py` (NEW).** First
off-HiRISE inference. Murray tiles are 4°×4° (~237 km) vs ~6 km footprints and
the tile zips are STILL cached in `cache_v2/ctx_tiles/`, **so NO download was
needed**. Pilot windows a cohort tile beyond its footprint → `read_tile_window`
→ `FangEmbedder.embed_window` → `DeployableHead.predict` → `tiles_to_raster` →
GeoTIFF+PNG. Result E4_N44 east of ESP_055253_2245 (15 km, 8281 tiles, 21 s):
mean P(rich) 0.117, ≥0.5 share 0.001 (honest "mostly poor plains"), heatmap
patches track CTX texture. Outputs:
`reports/figures/map_pilot_E4_N44_ESP_055253_2245_east.png` + GeoTIFF/JSON in
`reports/map_pilot/`. **Georef bug found by a post-render check + fixed**: `(ti,
tj)` are PARENT-TILE-anchored, but `predict_window` passed the WINDOW affine
(already offset) into `coarsened_transform` → offset double-counted (~108 km off);
fix = `tile_origin_transform` rebuilds the tile origin first; regression-tested.

**Tests +18** (11 `test_deployable_head.py`, 7 `test_mapping.py`); **full fast
suite 312 passed**.

**QA notebooks (+2, executed clean).** `notebooks/21_map_pilot.ipynb` (deployable
card + honest held-out truth-vs-model at S=32 + the beyond-coverage map) and
`notebooks/22_freeze_and_tier2.ipynb` (head bake-off → freeze → Tier-2 +
compression curve) — both built via `_build_{21,22}.py`. A §7 notebook audit found
target-distribution already covered (08/09/10/11/12); these close the FM-program
gaps. README updated with the new commands + notebook list.

## Earlier-session context (still true)

**FROZEN RECIPE (Brian sign-off; DECISIONS.md "Freeze window CLOSED"):**
`mlp_ens3` (3-seed MLP 768-256-64-1, dropout 0.2, BCE pos_weight, AdamW
lr1e-3/wd1e-4, ES patience 8 on rotated inner-val, mean of 3) on the **S=32 96-px
3×3-context GeM(p=3) 768-dim emb-only** matrix, target `fa_gt_1e-2`. Banked LOIO
`models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/`: pooled PR-AUC
**0.7832** / prec@5% **0.948** / med per-image AUC **0.7865** / dAUC(v) +0.120 /
win 0.96, both gates PASS. npz naming: **P96 = S=32 3×3 context** (the frozen one).

**§2.4 Tier-2 (DONE):** single-stage `mlp_reg` wins (Spearman 0.431 fa); hurdle
DROPPED; meaningful_auc 0.78 ≈ classifier. Tier-2 freeze/productize + a
tail-compression calibration layer remain future work (fold into the deployable
path when needed).

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`

## Key tooling (all on disk; UNCOMMITTED this session's additions)

- `src/modeling/mlp_head.py` — productized head + `DeployableHead` (save/load).
- `scripts/train_deployable_head.py` — train-on-all; banks `models/deployable/`.
- `src/mapping.py` — windowed read, own-tile nodata, (ti,tj)→raster,
  `tile_origin_transform`/`coarsened_transform`, `predict_window`, `write_geotiff`.
- `scripts/map_pilot.py` — one-tile off-HiRISE map (auto-places window beyond a
  footprint; nodata-aware candidate search; renders 3-panel PNG + GeoTIFF).
- `src/fm_embeddings.py` (§2.2) — ViT + GeM + `embed_window`; `src/modeling/loaders.py`
  fang cached-store join; `scripts/probes/_fm_freeze_window.py` (freeze runner).

## Next-session queue — PLAN_FM.md §2 is authoritative

DONE: §2.1 freeze, §2.2 productization, §2.4 Tier-2, **§2.6 deployable head +
map pilot**, **§2.7 reliability validation (negative → overlay deferred)**,
**§2.5 model-evidence report (DRAFTED — prose complete, held-out row + schematic
pending)**. Remaining, suggested order:

1. **§2.3 pre-declared confirmation — RECOMMENDED next build** — write the
   declaration now (gates/baseline/protocol, pre-data); execution waits on Brian's
   BoulderNet runs on the 23 expansion ObsIds. Confirm-then-absorb.
2. **§2.5 finish** — add the §3 ViT→GeM→MLP schematic figure and fill the
   `[held-out: pending]` headline row once §2.3 lands.
3. **§2.7 reliability overlay re-run (post-expansion)** — `src/reliability.py` +
   `scripts/probes/_fm_reliability_validation.py` are built and tested; the n=38
   validation came back negative (FM decoupled novelty from skill). Re-run the
   SAME validation once the cohort expands; only wire the per-tile overlay into
   `predict_window` if it clears the bar then. PLAN_FM §2.7 / DECISIONS 2026-06-14b.
4. Optional/gated: Tier-2 freeze + tail-compression calibration; full-Murray-tile
   map scale-out (the combine pattern needs the Murray-tile id carried alongside
   `(ti,tj)` — see DECISIONS.md 2026-06-14); MOMO disjoint-corpus probe; ViT
   fine-tune (decide after §2.3).

## Discipline now binding (PLAN_FM §3)

**No more recipe shopping on the 38 images.** The recipe is frozen; the next
number that touches it is the §2.3 pre-declared confirmation on held-out
expansion images. Misses recorded as declared. The deployable head trains the
SAME recipe on all data — not a new dev cell.

## Critical gotchas (carry forward)

- **Don't `cd` in the Bash tool** — it persists and poisons subsequent relative
  paths (cost ~20 min this session chasing phantom "missing data"). Use absolute
  paths; the data in `cache_v2/` is all present (24 tile zips, 39 windows).
- `conda run` needs `--no-capture-output` + `python -u`; multi-line `python -c`
  FAILS on Windows. Bare `python` not on PATH (only inside the env).
- `import src.modeling` BEFORE numpy/pandas in torch-adjacent scripts.
- Map grid is **PARENT-TILE-anchored** (not global-mosaic), so georef must go
  through `tile_origin_transform`; cross-tile combine needs the Murray-tile id.
- `EmbeddingBank`/store join keyed on (obs_id, ti, tj), validate="one_to_one".
- Group-aware LOIO always; inference features must be CTX-derivable (embeddings
  are, mosaic-global). Per-image AUC ±0.1-0.2 fold-ripple error bars.
- **AskUserQuestion before: commits, expensive sweeps, env mutation.**
- Run only ONE GPU job at a time. The deployable train is ONE run (~76 s); the
  map pilot is ONE GPU run (~21 s) — neither is a sweep.
- Fast pytest 312 (was 265 pre-FM) + slow CNN/checkpoint tests deselected.

## Reporting protocol

1. DECISIONS.md — entry per item with numbers (2026-06-14 entry exists).
2. Memory — `project_state_2026-06-14.md` is CURRENT.
3. This file — rewrite based on what actually lands.
