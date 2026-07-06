# ROADMAP — plan index for hirise2ctx

Single map of every `PLAN_*.md` and meta-doc, with current status and the supersession chain.
`CLAUDE.md` is the project spec; `DECISIONS.md` is the authoritative running log (exact dates,
commits, numbers). The `PLAN_*.md` files sit between them — architecture-level plans for each phase.
*Last reorganized 2026-06-19; statuses refreshed 2026-07-05.*

## 🟢 ACTIVE plans (current work)

| Plan | What it is | Status |
|---|---|---|
| [PLAN_RegionalMap.md](PLAN_RegionalMap.md) | Regional circum-Chryse abundance map + thermal/Rodriguez-2016 validation legs (first real off-HiRISE deployment of the frozen head + CalibrationLayer) | **ACTIVE** — map shipped (26 tiles, Sherlock); MOLA leg done; THEMIS night-IR leg-1 done but weak (ρ ≈ +0.07); remaining thermal legs wait for the **final (post-mitigation) map** |
| [PLAN_StripingArtifact.md](PLAN_StripingArtifact.md) | The regional-map rectangular-block artifact: cause + mitigation | **ACTIVE — PHASE 2 docket (2026-07-05d)** — cause SOLVED 2026-06-18d (CTX source-frame radiometry); A1 partial (28% eta² ↓ / −0.024); **F input-mapping leg closed** (5 mappings; log-minnaert = skill-gate PASS +0.0067 but η² 0.179, blocks visible — notebooks 26–28, DECISIONS 2026-07-05b/c). **Review 2026-07-05d amended the verdict**: post-minnaert overlaps agree to 4% (not 10%) — the embedder (5–20× amplifier, no cross-frame loss term) + one anomalous frame are the real floor → opened the Brian-approved **invariance & leveling docket H1–H6** (PLAN "PHASE 2" section: H1 log-median centering ~1 h → H2 nuisance-subspace removal → H3 consistency-regularized head → H4 overlap-constrained leveling of per-frame predictions (F-mode; ≠ the ruled-out D); decision rule η² ≲ 0.05 at skill ≥ −0.02 reopens the 907-frame build). **Next action = H1; nothing run yet** |

## ✅ CLOSED / SHIPPED plans (chronological program arc)

| Plan | Phase | Status / outcome |
|---|---|---|
| (CLAUDE.md §3–4) | Stages 0–3 — manifest, reproject detections, CTX retrieval, co-registration | shipped (commits `0770e2a`/`8e8645d`/`ed9003e`) |
| (CLAUDE.md §4) | Stage 4 — label gen on nested ×2 grid | shipped (`896cdef`; DECISIONS 2026-05-23) |
| [PLAN_Stage4b.md](PLAN_Stage4b.md) | Stage 4b — per-tile CTX texture features | shipped (`014f645`; 9 feature families, 60 cols) |
| [PLAN_Stage5.md](PLAN_Stage5.md) | Stage 5 — leave-image-out splits + packaging | shipped (`aa6cd74`; `loio_9fold` + `loio_3fold_balanced`) |
| [PLAN_Stage5b.md](PLAN_Stage5b.md) | Stage 5b — binary rich/poor reframing | shipped (folded into the modeling track) |
| [PLAN_Stage5c.md](PLAN_Stage5c.md) | Stage 5c — within-image k-fold CV | shipped 2026-05-27 (signal-floor branch confirmed) |
| [PLAN_modeling.md](PLAN_modeling.md) | Week 3 modeling — GBM baseline + two-stage | shipped v1; results [docs/modeling_results.md](docs/modeling_results.md) §1–8 |
| [PLAN_NewDetections.md](PLAN_NewDetections.md) | vClaire v2 — denser 40-img detection set | shipped → `dataset_v2/` (38 images); the dataset all later work uses |
| [PLAN_ModelImprovement.md](PLAN_ModelImprovement.md) | Model-improvement Phases A/B/C | shipped & **historical** (compression diagnosis, hurdle variants) |
| [PROMOTION_QUEUE.md](PROMOTION_QUEUE.md) Part B | Stage 6 — feature augmentation / per-image standardization | done (6a dev-PASS deferred; 6b strict-FAIL but mechanism validated; 6c soft-PASS) |
| [PLAN_Compositional.md](PLAN_Compositional.md) | Stage 7 — compositional study (HiRISE 3 bands) | Stage 7.0/7a/7c/7d + Tier 1/2 done; **PARKED** (compositional taken off the docket at the 2026-06-10 usability pivot); [docs/compositional.md](docs/compositional.md) |
| [PLAN_CNN.md](PLAN_CNN.md) | W2 — CNN on context patches | **CLOSED 2026-06-12 → superseded by PLAN_FM** (the Fang-ViT probe beat the CNN) |
| [PLAN_FM.md](PLAN_FM.md) | Post-foundation-model program | recipe **FROZEN 2026-06-12** (`fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2`) + productized → `DeployableHead`; closed |
| [PLAN_ModelUsability.md](PLAN_ModelUsability.md) | Make the model usable (Tier-1 binary map → calibrated abundance) | the 2026-06-10 pivot umbrella; **delivered via PLAN_Calibration + PLAN_RegionalMap** |
| [PLAN_Calibration.md](PLAN_Calibration.md) | De-compress / calibrate the abundance outputs | Stage 0 + Stage 1 **SHIPPED** (`CalibrationLayer`, a09f06b); Stage 2 **CLOSED** (retraining ceiling = 5 m/px CTX floor) |

## 📋 Meta / reference docs (not plans)

- [CLAUDE.md](CLAUDE.md) — the build spec (authoritative for scope).
- [DECISIONS.md](DECISIONS.md) — the running log; authoritative for dates/commits/numbers.
- [README.md](README.md) — setup + how to run each stage + sweeps.
- [PROMOTION_QUEUE.md](PROMOTION_QUEUE.md) — forward-looking docket of dev-validated changes awaiting full-v2 confirmation.
- [SHERLOCK_RUN.md](SHERLOCK_RUN.md) — operational runbook for the Sherlock GPU regional runs (active reference).
- [HANDOFF_NEXT_SESSION.md](HANDOFF_NEXT_SESSION.md) — ⚠️ stale (last 2026-06-15); the live session state lives in the memory `project_state_*` notes, not here.

## Conventions

- **Each plan is self-contained**: a future session can pick up a phase from its `PLAN_*.md` + the
  relevant DECISIONS entries without the others.
- **When a plan goes stale**, update it in the same change that diverges (a dated note at the top of
  the affected section); don't let it silently drift. Mark superseded plans with a top banner pointing
  to the successor (see PLAN_CNN.md, PLAN_StripingArtifact.md §1–6 for the pattern).
- **Open questions** in a plan's §10/end are the things execution should NOT pre-decide — surface them
  via `AskUserQuestion` (collaboration rule #1).
- This file is the index; keep it current when a plan opens, closes, or is superseded.
