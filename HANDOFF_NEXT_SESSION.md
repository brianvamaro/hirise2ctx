# Handoff prompt — next session

**Last updated 2026-06-10 (night) — W1 session 1: rung 1 DONE, found and
fixed the coreg sign bug, baseline re-banked on corrected labels.** The
active program is [PLAN_ModelUsability.md](PLAN_ModelUsability.md); next
session continues **W1 rungs 2–5** on the clean predictions.

Working dir: `c:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise2ctx`.
Conda: `C:\Users\brian\anaconda3\Scripts\conda.exe run --no-capture-output -n geospatial python -u ...`

## Read in this order before starting

1. [PLAN_ModelUsability.md](PLAN_ModelUsability.md) — W1 section status
   banner + the rung 2–5 specs.
2. Memory `project_state_2026-06-10c.md` (CURRENT).
3. [DECISIONS.md](DECISIONS.md) 2026-06-10 "W1 rung 1" entry — the bug, the
   evidence chain, the re-banked numbers.

## What happened (session 1 of W1)

**The rung-1 geometry audit found a cause-0 bug**: the coreg y-shift was
applied with inverted sign to all 38 v2 labels (missing row→world-y flip);
every label field sat 2×|dy| ≈ 360 m ≈ 1.1 S=64 tiles SOUTH of its CTX
texture. Fixed (`src/coregister.py::shift_px_to_world_m` + 2 regression
tests; fast suite 265), 48 cached coreg JSONs migrated in place
(`y_sign_fix_applied` marker), Stage 4 re-run 38/38, Stage 5 repackaged,
baseline re-banked. Validation: post-fix label-vs-CTX displacement is
sub-pixel; cohort rescore surface peaks at (0,0).

**Re-banked baseline (compare everything against THIS)**:
`lightgbm_two_stage_balanced` × `boulder_count` @ S=64 —
ρ +0.1878, presence AUC 0.6149, meaningful AUC 0.6243, PR-AUC 0.5616,
prec@top-5% 0.5859. Per-image meaningful-AUC: median 0.603 / >0.70: 34.2% /
<0.50: 21.1% (8 images). Sweep `models/_sweep_w0/20260611T013810Z`
(artifact dirs overwritten in place; old predictions are GONE — the pre-fix
rescore grid survives in `scripts/probes/_w1_shift_rescore.parquet`).
Tier 1 reference classifier: AUC 0.655 ± 0.129, lift 1.845, ECE 0.254
(`models/_sweep_binary/20260611T042543Z`). All W0 verdicts re-verified on
corrected labels — P2 promoted, P1 null, hurdle retained.

## W1 remaining work (rungs 2–5 + synthesis)

Targets = the 8 surviving anti-signal images (worst first):
**ESP_076499_1160** (AUC 0.224, ρ −0.51 — WORSE post-fix, strongly
inverted, ~64°S geographic outlier, "unknown" cohort label),
**ESP_055978_2270** (0.245), ESP_054000_2255, ESP_046328_2180,
ESP_064510_2260, ESP_047976_2020, ESP_049242_2115, ESP_059686_2235 (only
8 negatives — validity-suspect, may not be real anti-signal).

- **Rung 2 — join/pipeline integrity** per image (row counts, key
  uniqueness, NaN fractions, (ti,tj)↔pixel-block spot checks).
- **Rung 3 — BoulderNet label content** on the anti-signal images
  (full-res HiRISE visual sampling; detection score/size distributions
  vs cohort).
- **Rung 4 — feature/CTX content**: tile-level error maps + SeamMap seam
  polygons + per-tile n_sources / dominant_source_fraction + terrain-unit
  join (Tanaka global geologic map; Serrano takeaway 1).
- **Rung 5 — genuine limits** by exclusion only.
- **Synthesis**: notebook 18 + 38-row dossier (include n_pos/n_neg
  validity columns — per-image AUC is meaningless on near-saturated
  images like ESP_054622_2240 with 4 negatives) + Tier 1 reliability-flag
  definition + native-CTX go/no-go.

Reusable probes: `_w1_shift_rescore.py` (--artifact-dir/--summary/--tag),
`_w1_label_ctx_displacement.py` (model-free geometry check),
`_w1_antisignal_list.py <sweep_dir>`.

## Critical gotchas (carry forward)

- **All pre-fix v2 numbers are stale** (Stage 5c/6a/6b/6c, notebook 13
  taxonomy, slim/slimmer model numbers). Don't quote them as current.
  Slimmer docs are submitted — erratum decision is Brian's, not yet made.
- Stage 6a nbr artifacts (`dataset_v2/features_nbr_s5`,
  `packaged/loio_nfold_nbr_s5`) were NOT regenerated post-fix.
- `conda run` needs `--no-capture-output` + `python -u`; multi-line
  `python -c` fails on Windows — write a probe script.
- `import src.modeling` BEFORE numpy/pandas in torch-adjacent scripts.
- Group-aware LOIO always; inference features must be CTX-derivable.
- Sweep artifact layout: glob `*/scale_S{n}`, not `runs[-1]/...`.
- bc ≥ 50 vs fa > 1e-2 positive-definition caveat on cross-target deltas.
- AskUserQuestion before: expensive sweeps, git commits, destructive ops.
- **Fast pytest baseline 265 (full 285)** after the 2 coreg sign tests.

## Reporting protocol (carry forward)

1. DECISIONS.md — one entry per W-item with numbers.
2. Memory — supersede `project_state_2026-06-10c.md` when rungs 2–5 land.
3. This file — rewrite based on what actually lands.
