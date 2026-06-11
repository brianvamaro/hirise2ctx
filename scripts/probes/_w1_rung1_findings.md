# W1 Rung 1 — FINDING: coreg y-shift applied with inverted sign to all v2 labels

**Date:** 2026-06-10 (W1 session). Probes: `_w1_shift_rescore.py`,
`_w1_coreg_vs_auc.py`, `_w1_shift_surface.py`, `_w1_sign_error_check.py`,
`_w1_label_ctx_displacement.py`.

## The bug

`src/coregister.py:383` converts the phase-correlation row shift to metres as
`dy_m = dy_px * px_y`, **omitting the row→world-y sign flip** (rows increase
southward, world y northward; `Δy = e·Δrow` with `e < 0`). The array-space
convention itself is verified inside `phase_correlate_translation` (it applies
`nd_shift(moving, (dy, dx))` and reports Pearson peaks of 0.6–0.9), so
`dy_px` is right and the metre conversion is what's wrong. The x-component
(`dx_m = dx_px * px_x`) is correct. `labeling._apply_coreg_shift` then
translates the polygons by `(xoff=dx_m, yoff=dy_m)` — so every v2 label was
moved **south** when it needed to move **north**.

All 38 cohort images have `dy_px < 0` (HiRISE sits 6–285 m, median ~180 m,
north of the Murray mosaic), so after "correction" every label field sits
**~2·|dy| ≈ 12–570 m (median ~360 m ≈ 1.1 S=64 tiles) south** of the CTX
texture it is paired with.

## Evidence chain

1. **Label-shift rescore** (`_w1_shift_rescore.md`): cohort-mean AUC over all
   38 images peaks at label-read offset (di=+1, dj=0) — 0.616 vs 0.598 at
   center, monotone gradient along di, symmetric in dj. Mean Δ at (+1,0) =
   +0.017 vs −0.022 at (−1,0). A global row-direction misalignment of ~1 tile,
   no column misalignment — matching 2×median|dy| = 360 m ≈ 1.13 tiles and
   correct dx application.
2. **Direct displacement measurement** (`_w1_label_ctx_displacement.csv`):
   phase-correlating boulder-density rasters against CTX texture energy:
   - ESP_042964_2160 (peak 0.44): nominal labels (−36.2, +21.0) px ≈ cached
     HiRISE shift (−35.9, +21.3); as-applied labels (−72.2, −0.1) = **2× in
     dy, 0 in dx**.
   - ESP_066634_2210 (peak 0.49): nominal (−44.8, +8.0) ≈ cached (−44.5,
     +9.1); as-applied (−89.3, −1.2) = 2× dy, 0 dx.
   - ESP_069763_2235 (weak peak): same 2× pattern (−15.9 → −31.6).
3. **Coreg solve quality is NOT the problem** (`_w1_coreg_vs_auc.md`): peak /
   block-MAD / confident-fraction all uncorrelated with per-image AUC; the
   solves are good — the application is what's wrong.

## Collateral findings

- **ESP_054622_2240's anti-signal AUC is statistically meaningless**: 3
  negative tiles at bc>50; its 5×5 rescore surface swings 0.25↔0.85 between
  adjacent cells. Same caution applies to other near-saturated images
  (ESP_069763_2235 7 neg, ESP_059686_2235 10 neg, ESP_068483_2280 21 neg,
  ESP_045550_2180 26 neg). The dossier needs an n_neg column and a validity
  floor before per-image AUC is interpreted.
- Per-image best rescore offset tracks the predicted residual
  `round(2|dy_m|/320)` within ±1 tile for 79% of images
  (`_w1_sign_error_check.py`); the residue is argmax noise + coexisting
  failure modes (e.g. ESP_076499_1160 and ESP_046328_2180 peak at negative
  di — still unexplained, candidates for rungs 3–5 after relabeling).
- v1 is likely affected too if any v1 dataset was built with
  `apply_coreg_shift=True` (9 solves exist in `cache/coregistration`); v1
  modeling predated the 2026-05-28 "apply shift in Stage 4" change — verify
  before touching v1 conclusions.

## Implications

Every v2 label parquet, every model trained on it (including the W0 banked
recipe), and the per-image failure taxonomy were computed on labels displaced
~1 tile south. The W0 promoted baseline understates achievable performance.
Required sequence: fix sign → regenerate Stage 4/5 (cheap by design;
features are label-independent) → re-bank the W0 baseline → resume the W1
ladder on clean predictions.
