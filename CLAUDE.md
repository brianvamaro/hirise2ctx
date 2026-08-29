# HiRISE → CTX Rock Abundance Pipeline — operating manual

Always-loaded operating manual: **orientation + the invariants that must fire every session +
pointers**. The full original build spec is preserved verbatim in
[docs/build_spec.md](docs/build_spec.md).

> **Orientation.** Goal: learn 5 m/px **CTX** texture/shadow → per-tile **rock abundance**, trained
> on BoulderNet HiRISE detections, to map abundance across the CTX mosaic where HiRISE is absent.
> The Weeks-1–2 data pipeline (in build_spec) is **built**. Work has since progressed: modeling →
> foundation-model recipe (frozen `mlp_ens3` on Fang-ViT embeddings) → deployable head + calibration
> → regional circum-Chryse map → the CTX **source-frame striping artifact**, whose cause is solved
> and for which **no mitigation survives** (A1 demoted 2026-08-25; the artifact ships as a caveat). For
> **current phase + plan index see [ROADMAP.md](ROADMAP.md)**; running log = [DECISIONS.md](DECISIONS.md);
> live session state = the `project_state_*` memory notes.

---

## Invariants & gotchas (load-bearing — keep these in mind every session)

> **Mutation safety (2026-08-06, test-side gate CLOSED):** the test suite can no longer write a live
> artifact. `tests/live_artifact_guard.py` is installed session-wide and **refuses** any write under
> `cache*/`, `dataset*/`, `models/`, `reports/` (the `cache_v2_dev` junction included); a static AST
> scan fails if a test hands a producer a live root; `read_only_cache` copies mutable derived artifacts
> and links only `{tile}.zip` / `{obs}_RED.JP2`. **Still true and still load-bearing:** the producers
> have no dry-run mode, so *scripts and notebooks* — which the guard does not cover — must be given
> explicit absolute scratch roots. A copied YAML is not isolation (`Config` resolves relative paths
> against `REPO_ROOT`) and `cache_v2_dev` is a junction to the live `cache_v2`. A verified backup exists
> (`D:\HiRISE2CTX Backup`, 125.55 GB, 8/8 roots verified 2026-08-18) — it makes a mistake
> recoverable, it does not make one safe. The rebuild itself is **done** (all 12 steps, 2026-08-20 → 25).

- **Per-image local-radius CRS (the #1 gotcha).** Detections are equirectangular (`Equidistant_Cylindrical`,
  central meridian 180°) **on a sphere whose radius is the local Mars radius at that image's center
  latitude** — it differs image-to-image (e.g. `3393833.26 m`, not the standard `3396190 m`). **Read
  each shapefile's own `.prj`; reproject per-image into the common CTX CRS. Never hardcode a radius
  or assume a shared datum.** The CTX mosaic CRS is Mars_2015 equirectangular **clon_0** (sphere
  3396190 m), verified at runtime.
- **CRS sanity check:** after correct reprojection the residual HiRISE↔CTX offset is **O(200 m)**,
  not km. If it comes out in km, the CRS handling is wrong — **fail loudly**.
- **HiRISE PDS SP1 bug:** the pipeline auto-corrects the upstream `Standard_Parallel_1=0` bug via the
  PDS `.LBL` (both polygon and JP2 sides). Don't "fix" the override thinking it's wrong.
- **CTX tiles are GB-scale → windowed reads only.** **HiRISE RED → read decimated (~5 m/px), never
  full-res.** Prefer `/vsicurl/` range requests.
- **Target is heavily zero-inflated / right-skewed** — preserve raw base stats so the modeling stage
  can choose log1p / two-stage / stratified handling.
- **Splits are group-aware leave-image-out, never random tiles** (tiles within an image are
  spatially correlated).
- **Manifest-driven:** adding a manifest row + its detection folder must flow end-to-end with no code
  changes. Don't hardcode the image list. Discover detection shapefiles by glob (`{ObsId}/*-mask-nms.shp`).
- **VERIFY AT RUNTIME:** read unknowns from the data/source; record the answer in DECISIONS.

## Environment & invocation (operational — easy to get wrong)

- **OS Windows; conda env `geospatial`.** Invoke via `C:\Users\brian\anaconda3\Scripts\conda.exe run
  -n geospatial python ...` — conda is **not** on PATH; do **not** call the env's `python.exe`
  directly (memory: `conda_location`).
- **OpenMP/MKL:** any script that uses torch must `import src.modeling` **before** numpy/pandas
  (memory: `torch_windows_openmp_fix`).
- **SSL:** stdlib `urllib` fails `CERTIFICATE_VERIFY_FAILED` here; `truststore.inject_into_ssl()` is
  the project-wide fix (already installed; memory: `conda_windows_ssl`). For GDAL `/vsicurl/` use it
  too (or `GDAL_HTTP_UNSAFESSL=YES`).
- **Notebooks must not re-produce a shipped artifact.** `scripts/map_mosaics.py` is the sole
  producer of `regional_{layer}_mosaic.tif` **in a per-arm product** (`map_region` / `map_a1` /
  `map_extended`) — it alone carries the `SIZE_FLOOR_*`/`MOSAIC_*` tags and the closed-footprint
  gate. `scripts/map_union.py` is the sole producer of the same filenames **in `reports/map_union`**
  and nowhere else (it refuses an `--out` that is also a `--source`). Consumers call
  `src.mapping.load_regional_mosaic` or `src.map_validation.load_union`, which read and never write.
  Notebook 24 §2 used to rebuild them over the top — rewired 2026-08-28.
- **Notebooks are generated:** edit the source-of-truth `notebooks/_build_NN.py`, not the `.ipynb`;
  regenerate with `python notebooks/_build_NN.py` then `nbconvert --execute --inplace`. **Never run
  two notebooks (or two CTX-heavy jobs) at once** (memory: `feedback_collaboration`).
- **Logic lives in importable `src/` modules**; notebooks and tests *call* it (nothing important
  lives only in a notebook). Loop: `pytest -m "not slow"`. The full suite was verified
  **non-mutating** at the 2026-08-06 audit (path/size/mtime manifest over all six artifact
  roots, bit-identical before and after). **`slow` is not the safety control** (20 non-slow
  tests call a producer); the runtime guard and the static AST scan are.
- For repeat visual analyses, **download JP2s** rather than `/vsicurl/` each time.

## Reporting standards (project-specific)

- **Never report presence AUC** (`y_true>0` / "any boulder"). Use the meaningful rich/poor threshold
  **fa > 1e-2**: `meaningful_auc` / `pr_auc@1e-2` / `precision@5%` + Spearman ρ + per-bin RMSE
  (memory: `feedback_no_presence_auc`).
- **Hyperlink every citation** to its canonical DOI in docs/notebooks (memory: `feedback_hyperlink_citations`).
- **Collaboration:** surface genuinely open decisions via `AskUserQuestion` before acting on them.

## Pointers

- **Full build spec (Weeks 1–2, verbatim):** [docs/build_spec.md](docs/build_spec.md)
- **Current phase + plan index:** [ROADMAP.md](ROADMAP.md) — active vs closed plans, supersession chains
- **Running decision log:** [DECISIONS.md](DECISIONS.md) — every VERIFY-AT-RUNTIME answer + deviations
- **Setup + how to run each stage / sweep / map / striping:** [README.md](README.md)
- **Config:** `config.yaml` (+ `config_v2.yaml` for the vClaire v2 dataset)
- **Output column dictionary:** `dataset/DATA_DICTIONARY.md`
- **Methods writeups (for non-coders):** [docs/methods.md](docs/methods.md), [docs/index.md](docs/index.md)
- **The gating code-review audit — now a *record*, not a handoff** (all five safety criteria
  and the mapping gate read CLOSED):
  [docs/CODE_REVIEW_AUDIT_2026-08-06.md](docs/CODE_REVIEW_AUDIT_2026-08-06.md)
- **The rebuild:** [PLAN_Rebuild.md](PLAN_Rebuild.md) — ✅ **COMPLETE. All 12 steps executed and
  verified (2026-08-20 → 25).** `docs/PENDING_REBUILD.md` is now a *record*, not a plan: row 1
  discharged, rows 2–3 open by ruling (FM-path only). **The canonical maps are
  `reports/map_region` (baseline) and `reports/map_a1` (A1)**; the displaced pre-R01 product is
  archived at `reports/map_region_g1` and **must not be quoted** — it has 26 distinct sub-cell
  lattice phases. Re-check the shipped product any time with `scripts/verify_map_download.py`
  (sha256 vs each sidecar's own `rasters[]`), `scripts/verify_arm_parity.py` (one lattice,
  cell-for-cell co-registration, one size-floor basis) and `scripts/map_sidecar_qa.py` (12 gates
  over all 52 sidecars). `scripts/rebuild_map_manifest.py` repairs a damaged manifest index from
  the sidecars, with no GPU and no re-render.
  - ⚠ **Two reporting traps the QA tooling exists to prevent.** (1) The sidecars come in **three
    schema generations**; a missing `overlap` key is an *absence of measurement*, not a zero — 28
    of 52 tiles are `unknown_on_gate_layer`. (2) A missing `device` field means "predates the
    field", and what hardware that implies is **arm-conditional** (baseline = 2080 Ti, A1 = Pascal),
    known from the run logs, not from the sidecar.
  - ⚠⚠ **A1 IS NOT THE PRODUCT (ruled 2026-08-25, DECISIONS 2026-08-25k).** `reports/map_region`
    (baseline) is the deliverable; `reports/map_a1` is a **sensitivity arm** kept for differencing.
    A1 was demoted because it wins only on *raw* η² while its ratio to its own rotation null does
    not improve, it **fails** the Tier-1 ECE gate the baseline passes (0.0523 vs 0.0204,
    force-banked), it is no better thermally, and it **manufactures** frame-shaped blocks on 9/26
    tiles — predictably from its own per-frame gain `A1_REF_IQR/frame_IQR` (ρ +0.490, p 1.4e-4).
    The source-frame artifact therefore **ships unmitigated**, as a documented caveat: treat any
    abundance reading in low-contrast terrain as carrying frame structure. A **gain-capped** A1 is
    the one untried lever, logged as v3.
  - ⚠ **A1's η², re-derived 2026-08-25:** raw η² fell (window median 0.1444 → 0.1145; like-for-like
    pilot crop 0.2327 → 0.1298, −44%) at a **−0.0024** skill cost and no THEMIS-ρ cost — **but η²
    relative to its own rotation null did not improve at all** (median ratio 1.599 → 1.639, better
    on only 106/234 windows), and **9 of 26 tiles get worse** on raw η². A1 narrows the *bulk* of
    the field (`prob_raw` IQR ratio 0.85 — but its sd *rises* 3%, so it is **not** a uniform
    compression: the tails widen), which lowers the geological floor along with the artifact.
    **Quote the raw reduction only alongside the ratio.** The banked 0.196→0.141 / −0.024 pair is
    superseded and not comparable.
- **Growing the map beyond circum-Chryse** (opened 2026-08-28, DECISIONS 2026-08-28b, runbook
  [SHERLOCK_RUN.md](SHERLOCK_RUN.md) §C5). The extension goes to a **separate, growable product
  `reports/map_extended`** — `map_region`/`map_a1` stay frozen at 26 tiles so their footprint gate,
  12-gate sidecar QA and cell-for-cell arm parity keep passing. Same global R01 lattice, so the two
  are mergeable by construction. The kit is **plan-driven**, so a new box edits no code:
  `scripts/plan_map_extent.py` (box → tiles + measured GPU-h + verified download GB → `plan.json`),
  then `scripts/adopt_map_tiles.py` (copy already-rendered overlap, verified both ends),
  `scripts/fetch_ctx_tiles.py`, `run_map_extended_array.sbatch`, `verify_map_download.py --plan`.
  - ⚠ **Murray URLs are zero-padded** (`E-024_N28`), not the bare tile id (`E-24_N28`) — every
    western tile 404s on the bare form. Anything that talks to the mosaic must try both, as
    `ensure_tile_cached` does.
  - ⚠ **Map wall-clock is GPU-conditional**: 17–22 s/window on a 2080 Ti, ~202 s/window on Pascal
    (which is what timed out in rebuild step 11). Read the job's own `nvidia-smi` line first.
  - ⚠ **Truth coverage thins fast outside circum-Chryse** — 23 of the 39-image cohort sit inside the
    shipped 26-tile map; only 1 inside the new southern block. Say so in captions.
  - ⚠ **Anything new written into a map-output directory must join `MANIFEST_NAMES`.**
    `src.map_manifest.tile_sidecars` is a denylist *on purpose* — `tile_result_rows` has to index
    whatever footprint is on disk, so a tile-name pattern would reintroduce the
    hardcoded-tile-list assumption. An unlisted JSON reads as a corrupt tile on a second lattice.
  - **Comparing the shipped maps:** [notebooks/29_map_comparison.ipynb](notebooks/29_map_comparison.ipynb)
    — old vs new and baseline vs A1. ⚠ The archived `map_region_g1` is **not co-registered** with the
    promoted product, so it can only be compared by world coordinates or by distribution, never by
    array index. Old→new turns out to be *the same field, moved*: 95% of the difference is
    high-frequency with the gradient signature of a pure translation, over a small (4.9%-of-variance)
    genuine re-levelling. A1's effect is **33.7%** regional by contrast.
- **Validating the map against independent data** (PLAN_MapValidation, opened 2026-08-28; step 1
  + **notebook 30 done** 2026-08-29, DECISIONS 2026-08-29a/b). Five notebooks 30–34 read **one**
  deduplicated surface: `reports/map_union`, produced only by `scripts/map_union.py`. **The union is
  122 tiles** since round 2 rendered (26 `map_region` + 104 `map_extended` − **8 shared**; it was 53
  on the first build, and the plan as written said 54 — an arithmetic slip). The 8 shared tiles are
  sha256-identical on all three layers, so dedup **asserts equality** and a mismatch is a hard
  failure, not a merge. **Never hardcode the tile count** — read it from the product
  (`meta["n_union_tiles"]`); it has changed twice in two days. Notebooks call
  `src/map_validation.py`, never the arms.
  - ⚠ **Reading an arm mosaic instead of the union is a silent 50%-coverage bug** — it loads, it
    computes, and the answer is about half the footprint. `load_union` refuses a mosaic with no
    `UNION_N_TILES` tag; override only deliberately.
  - ⚠⚠ **The striping artifact is deliberately NOT controlled for in 30–34** (Brian's ruling: he is
    not confident in how it has been done so far). **Every contrast is an upper bound on the
    geologic signal**; notebook 32 is the entry point to that separate investigation.
  - ⚠ **Significance never from the pixel count.** `cluster_bootstrap_ci` resamples polygons /
    craters / CTX source frames and reports `n_groups` *and* `n_cells`. `frame_effective_n`
    deduplicates `PRODUCT_ID` **across tiles** — frames straddle tile boundaries.
  - ⚠ **A median is not the summary for a pooled zonal read**: `abundance` is so zero-inflated that
    the median over a multi-million-cell region is exactly 0.0 with a zero-width CI (measured). Ruled
    2026-08-29: the **headline statistic is the rich fraction** (`prob >= 0.5`) with mean abundance
    beside it, and the reportability floor is **`mv.MIN_CELLS_UNIT` = 50,000 cells** (1,280 km²) —
    below it a unit is *flagged*, never silently dropped.
  - ⚠⚠ **SIM3292 (Tanaka geology) cannot be reprojected naively — and it fails SILENTLY.** All 1311
    polygons are valid in the source Robinson CRS, but the **inverse** Robinson overflows to `inf`
    for **62** of them, `make_valid` then **crashes**, and `.intersects()` on a non-finite geometry
    returns *garbage* behind only a `RuntimeWarning`. Use `mv.load_geology`, which selects and clips
    **in Robinson first**. The plan's planning-stage "67 polygons / 16 units" was measured the naive
    way and is superseded by **75 polygons / 14 units**.
  - ⚠ **Cell-weighted ≠ per-polygon ranking** (notebook 30, measured): the two agree only at
    ρ +0.427, and `lNh` moves from rank 3 to rank 12 because its cell-weighted value is **282×** its
    typical polygon. Before writing "unit X is boulder-rich", check its per-polygon spread.
- **Live session state:** the `project_state_*` memory notes (not the stale `HANDOFF_NEXT_SESSION.md`)

When reality diverges from a doc, update DECISIONS (and the relevant PLAN/ROADMAP) in the same change —
don't let docs silently drift.
