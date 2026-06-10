# Handoff prompt — next session

**Last updated 2026-06-10 — Project direction PIVOTED from report-writing to
model usability.** The slimmer report writeups are submitted (bylines committed
`cd291f0`); compositional work (dust index, Tier 3) is explicitly OFF the
docket. The active program is **[PLAN_ModelUsability.md](PLAN_ModelUsability.md)**:
make the rock-abundance / classification model actually usable over CTX.

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`
(see memory `conda_run_no_capture_output`).

## Read in this order before starting

1. **[PLAN_ModelUsability.md](PLAN_ModelUsability.md)** — the active program:
   root-cause synthesis, two product tiers, workstreams W0–W5.
2. Memory `project_state_2026-06-10.md` (CURRENT) — scoping decisions.
3. [PROMOTION_QUEUE.md](PROMOTION_QUEUE.md) — P1/P2/P5 + Stage 6a details that
   W0 promotes.
4. [docs/modeling_results.md](docs/modeling_results.md) §11–14 — the evidence
   the plan is built on.

## Brian's scoping decisions (2026-06-10)

- **Usable = two tiers**: Tier 1 binary boulder-rich map (FIRST), Tier 2
  calibrated abundance map (after).
- **Cohort fixed at 38 v2 images** — no new BoulderNet runs for now (W5
  documents data expansion as the deferred unlock).
- **CUDA torch approved** for the `geospatial` env (GPU present; torch
  currently CPU-only build — see memory `local_gpu_available`).
- **Start with W0 — bank the wins.**

## Next session = W0 (~1 day)

1. **P1+P2 full-v2 LOIO promotion** — P1+P2 numbers already exist as the
   Stage 6b sweep baseline (Spearman 0.1431 / PR-AUC 0.5431 @ S=64,
   `models/_sweep_stage6b/20260531T020308Z/`); formalize the delta vs the
   `fractional_area` baseline + DECISIONS.md entry.
2. **Single-stage vs two-stage test** — memory `modeling-single-stage-future`;
   if within noise, drop the hurdle.
3. **Genuine binary classifier + P5 calibration fix** —
   `LightGBMClassificationBalanced` on `fa_gt_1e-2` LOIO; becomes the Tier 1
   reference model.
4. **Stage 6a 5×5 @ S=32 full-v2 confirmation** (only strict dev PASS in the
   Stage 6 family).

Deliverable: a "recipe table" in DECISIONS.md naming the promoted baseline +
per-image metric distribution. Everything later compares against it.

## Critical gotchas (carry forward)

- **`conda run` swallows subprocess stdout** unless `--no-capture-output`;
  combine with `python -u` + flushed prints.
- **`conda run python -c "..."` can't take multi-line strings** on Windows —
  write a tempfile.
- **`import src.modeling` BEFORE numpy/pandas** in any torch-adjacent script
  (Windows MKL OpenMP fix).
- **Inference-time scope**: model features must be derivable from CTX alone
  (PROMOTION_QUEUE.md top section). HiRISE metadata is diagnosis-only.
- **Group-aware LOIO splits always** — never tile-random, including CNN.
- **Sweep artifact layout**: each (variant, scale) gets its own config_hash
  dir — glob `*/scale_S{n}`, not `runs[-1]/scale_S{n}` (memory
  `sweep_vs_train_gbm_artifacts`).
- **AskUserQuestion before**: expensive sweeps, git commits, destructive ops
  on cached artefacts.
- **281 pytest pass baseline** — run `pytest tests/ -q` before any promotion.

## Reporting protocol (carry forward)

1. **PROMOTION_QUEUE.md**: move promoted items to the bottom "Promoted"
   section with full-v2 numbers.
2. **DECISIONS.md**: one entry per promotion/test with numbers.
3. **Memory**: write a new `project_state_*` file; mark it CURRENT in
   MEMORY.md and supersede the old one.
4. **HANDOFF_NEXT_SESSION.md**: this file — rewrite based on what actually
   lands.
