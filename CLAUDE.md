# HiRISE → CTX Rock Abundance Pipeline — operating manual

Always-loaded operating manual: **orientation + the invariants that must fire every session +
pointers**. The full original build spec is preserved verbatim in
[docs/build_spec.md](docs/build_spec.md).

> **Orientation.** Goal: learn 5 m/px **CTX** texture/shadow → per-tile **rock abundance**, trained
> on BoulderNet HiRISE detections, to map abundance across the CTX mosaic where HiRISE is absent.
> The Weeks-1–2 data pipeline (in build_spec) is **built**. Work has since progressed: modeling →
> foundation-model recipe (frozen `mlp_ens3` on Fang-ViT embeddings) → deployable head + calibration
> → regional circum-Chryse map → the CTX **source-frame striping artifact** + A1 mitigation. For
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
> against `REPO_ROOT`) and `cache_v2_dev` is a junction to the live `cache_v2`. Do not start a rebuild:
> the remaining gates are in [docs/CODE_REVIEW_AUDIT_2026-08-06.md](docs/CODE_REVIEW_AUDIT_2026-08-06.md).

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
- **Notebooks are generated:** edit the source-of-truth `notebooks/_build_NN.py`, not the `.ipynb`;
  regenerate with `python notebooks/_build_NN.py` then `nbconvert --execute --inplace`. **Never run
  two notebooks (or two CTX-heavy jobs) at once** (memory: `feedback_collaboration`).
- **Logic lives in importable `src/` modules**; notebooks and tests *call* it (nothing important
  lives only in a notebook). Loop: `pytest -m "not slow"` (512 passed / 21 deselected, 2026-08-06).
  The slow suite is no longer structurally unsafe, but its four producer tests have not been run since
  the staging fixture changed — ask before running it. **`slow` is not the safety control** (20
  non-slow tests call a producer); the guard and the static scan are.
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
- **Current code-review/fixing handoff:**
  [docs/CODE_REVIEW_AUDIT_2026-08-06.md](docs/CODE_REVIEW_AUDIT_2026-08-06.md)
- **Live session state:** the `project_state_*` memory notes (not the stale `HANDOFF_NEXT_SESSION.md`)

When reality diverges from a doc, update DECISIONS (and the relevant PLAN/ROADMAP) in the same change —
don't let docs silently drift.
