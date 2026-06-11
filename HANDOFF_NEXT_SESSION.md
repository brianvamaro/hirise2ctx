# Handoff prompt — next session

**Last updated 2026-06-10 (late) — W0 "bank the wins" DONE.** The active
program is [PLAN_ModelUsability.md](PLAN_ModelUsability.md); next session is
**W1: error atlas as differential diagnosis**.

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`

## Read in this order before starting

1. [PLAN_ModelUsability.md](PLAN_ModelUsability.md) — the program. W0 marked
   done; W1 section has the full diagnosis-ladder spec.
2. Memory `project_state_2026-06-10.md` (CURRENT).
3. [DECISIONS.md](DECISIONS.md) 2026-06-10 entry — W0 verdicts + the
   promoted recipe table.

## W0 outcome (what W1 builds on)

**Promoted baseline recipe** (everything compares against this):
`lightgbm_two_stage_balanced` × `boulder_count` @ S=64, 51 Stage-4b
features, full-v2 LOIO — ρ +0.1431, presence AUC 0.6149, PR-AUC 0.5431,
prec@top-5% 0.5679. Per-image meaningful-AUC: median 0.594 / max 0.979 /
23.7% > 0.70 / **28.9% < 0.50**. Predictions parquet for W1 error maps:
`models/lightgbm_two_stage_balanced/<hash>/scale_S64_target_boulder_count/predictions.parquet`
(find via `models/_sweep_w0/20260610T221932Z/aggregate.parquet` →
`artifact_dir` column).

Verdicts: P2 promoted (PR-AUC +0.162 p<1e-4); P1 null at LOIO; P5 null
(ECE unchanged 0.26 — miscalibration is between-image shift); single-stage
rejected (hurdle wins per-image meaningful-AUC +0.022, p=0.008); Stage 6a
S=32 strict FAIL / partial carry (Δρ +0.072 PASS, ΔPR-AUC +0.017 FAIL).

**Meta-finding for W1**: three dev wins (P1, P5, + the historic Stage 6c
gate) all evaporate at LOIO the same way — per-image distribution shift is
the binding constraint. The error atlas should treat this as its prior.

## W1 — error atlas (next session)

Work the ladder mundane → fundamental (Brian directive; full spec in plan):

1. **Rung 1 — label geometry**: reprojection QA overlays for the worst
   images; coreg shift quality vs per-image AUC; **the ±1-tile label-shift
   rescore test on an anti-signal image** (cheap, decisive — if AUC
   recovers, it was geometry).
2. **Rung 2 — join/pipeline integrity** per image.
3. **Rung 3 — BoulderNet label content** on anti-signal images (full-res
   HiRISE visual sampling).
4. **Rung 4 — feature/CTX content**: tile-level error maps + SeamMap seam
   overlay + terrain-unit join (global geologic map).
5. **Rung 5 — genuine limits** by exclusion only.

Outputs: notebook 18 + per-image dossier (38 rows) + Tier 1 reliability-flag
definition + go/no-go on the native-CTX pivot (plan §1.1 takeaway 2).
AUC < 0.5 favours artifact causes — absent signal gives 0.5; inversion
needs a mechanism (shifted labels over a coherent field qualifies).

Anti-signal images to start from (notebook 13 §6 + Stage 6b winners):
ESP_054000_2255, ESP_064510_2260; check the W0 summary parquet for the
current < 0.50 list (11 images).

## Critical gotchas (carry forward)

- `conda run` needs `--no-capture-output` + `python -u`; multi-line
  `python -c` fails on Windows — write a tempfile/probe script.
- `import src.modeling` BEFORE numpy/pandas in torch-adjacent scripts.
- Inference features must be CTX-derivable (PROMOTION_QUEUE.md top).
- Group-aware LOIO always, including CNN (W2).
- Sweep artifact layout: glob `*/scale_S{n}`, not `runs[-1]/...`.
- W0 metric caveat: boulder_count cells use bc ≥ 50 positives,
  fractional_area cells fa > 1e-2 — designed equivalent, not identical.
- AskUserQuestion before: expensive sweeps, git commits, destructive ops.
- **283 pytest pass baseline** (was 281; +2 for the balanced classifier).

## Reporting protocol (carry forward)

1. PROMOTION_QUEUE.md — 2026-06-10 status banner added; queue is now
   mostly resolved, new Stage-6-style items go to the plan instead.
2. DECISIONS.md — one entry per W-item with numbers.
3. Memory — supersede `project_state_2026-06-10.md` when W1 lands.
4. This file — rewrite based on what actually lands.
