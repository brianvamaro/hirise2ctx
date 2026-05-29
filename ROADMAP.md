# ROADMAP

Index of phase-level planning documents for hirise2ctx, drafted 2026-05-23.

`CLAUDE.md` is the project spec; `DECISIONS.md` is the running log of runtime
decisions. The `PLAN_*.md` files below sit between them: architecture-level
plans for phases that have started and phases that haven't.

## Phases

| Phase | Status | Plan | Notes |
|---|---|---|---|
| Stage 0–1 — manifest + reproject detections | shipped | (covered in CLAUDE.md §3-4) | Commit `0770e2a` |
| Stage 2 — CTX windowed retrieval + HiRISE coverage mask | shipped | (covered in CLAUDE.md §4) | Commit `8e8645d` |
| Stage 3 — HiRISE↔CTX co-registration | shipped | (covered in CLAUDE.md §4) | Commit `ed9003e` |
| Stage 4 — label generation on nested ×2 grid | shipped | (covered in CLAUDE.md §4) | Commit `896cdef`; results in DECISIONS.md 2026-05-23 |
| Stage 4b — per-tile CTX texture features | shipped | [PLAN_Stage4b.md](PLAN_Stage4b.md) | Commit `014f645`; results in DECISIONS.md 2026-05-23. 9 feature families, 643k feature rows, 3.3 GB context patches. |
| Stage 5 — leave-image-out splits + packaging | shipped | [PLAN_Stage5.md](PLAN_Stage5.md) | Commit `aa6cd74`; results in DECISIONS.md 2026-05-25. Two schemes (`loio_9fold` + `loio_3fold_balanced`); group-leak assertion in QA notebook. |
| **Week 3 modeling — GBM baseline + two-stage** | shipped (v1) | [PLAN_modeling.md](PLAN_modeling.md) | LightGBM + Tweedie, LOIO CV, Spearman ρ primary metric. Results in [docs/modeling_results.md](docs/modeling_results.md). CNN explicitly punted (dead-end, §3.3). |
| **vClaire v2 dataset — denser 40-image detection set** | in progress | [PLAN_NewDetections.md](PLAN_NewDetections.md) | Parallel dataset on the far-denser vClaire BoulderNet run (`config_v2.yaml` → `cache_v2/`+`dataset_v2/`, both gitignored). Stages 1–5 done on 38 images; modeling A/B (does denser labels lift the v1 AUC≈0.55 ceiling? §9) underway. Decisions in DECISIONS.md 2026-05-28. |
| THEMIS validation | future work | not yet planned | CLAUDE.md §10. Coarse-scale independent check using THEMIS rock-abundance map. |
| Compositional analysis | future work | not yet planned | CLAUDE.md §10. Thermal / CRISM spectra in boulder-rich vs boulder-poor areas (instructor's extra goal). |

## How to read the plans

- **Architecture depth**: modules, interfaces, file layout, key decisions to
  surface via `AskUserQuestion` at execution time. No pseudocode.
- **Each plan is self-contained.** A future session can pick up a single
  phase by reading its `PLAN_*.md` + the relevant DECISIONS.md entries
  without needing the others.
- **Sequencing**: Stage 4b and Stage 5 are independent (either order works,
  4b-then-5 slightly preferred for QA — see PLAN_Stage4b.md §10). We landed
  4b first then 5 in that order.
  Week 3 modeling needs both 4b and 5 to land first — both are done as of
  commit `aa6cd74` (2026-05-25).
- **Open questions** in each plan's §10 (or near-end) are the things
  execution should NOT pre-decide — surface them via `AskUserQuestion` per
  the project's `[[feedback-collaboration]]` rule #1.

## When plans go stale

If reality diverges from a plan during execution, update the plan in the
same commit that diverges, with a short note at the top of the affected
section. Don't silently let the plan drift — that's the whole point of
having them checked in rather than living only in memory.

`CLAUDE.md` remains authoritative for the overall spec; per-phase plans
flesh out specific *how* questions inside that spec. When `CLAUDE.md`
itself changes (rare), update the affected plans.
