# Review area: calibration

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-01
- **Verification:** self-refuted (single-agent pass; not independently verified)

Scope read in full: `src/calibration.py` (385), `src/reliability.py` (161),
`scripts/bank_calibration.py`, `scripts/bank_calibration_f.py`, `tests/test_calibration.py`,
`tests/test_reliability.py`, plus every consumer of `CalibrationLayer`
(`src/mapping.py:200-289`, `scripts/map_region.py`, `scripts/map_pilot.py`,
`scripts/striping_a1_map.py`, `scripts/f_region_staged.py`, `scripts/parity_check.py`,
`src/fgates.py:270-296`).

All numeric claims below were recomputed from committed artifacts
(`reports/figures/fbuild_*`, `reports/map_fbuild/*.json`) plus the two banked
`calibration.npz` files and `dataset_v2/labels/*.parquet`. Where a published number is quoted I
reproduced it exactly before criticising it.

---

## Findings

### calibration-1 — The abort table's `full` row measures `np.interp` clamping, not abundance level, and silently drops its 2 worst observations
- **Severity:** high
- **Liveness:** dead-closed (F build) — corrects the abort *record*, does not overturn the abort
- **Confidence:** high (reproduced the published row exactly, then decomposed it)
- **Where:** `src/calibration.py:365-369` (`calibrate_abundance` = clamped `np.interp`);
  `scripts/f_region_staged.py:232,247-248`; `DECISIONS.md:5532-5541`;
  `reports/map_fbuild/README.md:14-20`; `reports/figures/fbuild_abort_level_vs_labels.csv`;
  `reports/figures/fbuild_gate6_abundance.csv` (`F_full` rows)

`calibrate_abundance` is `np.interp` on the quantile-match knots, so **any** probability above
`t2_x[-1] = 0.99991630` returns the constant `t2_y[-1] = 0.29324219`, and any probability at/below
`t2_x[752] = 0.064311` returns exactly 0. The `full` variant rails (documented: "rails 51.8 % of
co-located tiles, |o|max 21.31 logits vs the model's own ±9.21 range",
`reports/map_fbuild/README.md:32`), so **37.07 % of the 89,145 labelled tiles scored for `full` sit
above the calibrator's entire reference range and receive the clamp constant**, and a further
8.02 % receive exactly 0 — 45 % of the scored population is a constant. For 6 of 21 observations
≥ 73.9 % of finite tiles are clamped (up to 100 %), so their published "over-prediction ratio" is
just `ceiling / mean(label)`. Two further observations are 100 % at the zero floor, giving ratio 0;
those two are **excluded** from the published `full` row without any note, so `full` is reported over
19 observations while every other row is over 21.

- **Failure scenario:** `DECISIONS.md:5538` and `reports/map_fbuild/README.md:20` publish
  `full` = median 15.54 / max 380.28 / max-min 81.3× / sd(log₁₀) 0.412 as a measurement of F's
  between-place level instability. It is not a measurement: for ESP_055978_2270 the published
  380.28 versus `ceiling/label_mean = 380.47` (98.76 % of its tiles clamped); ESP_045983_2270 60.21
  vs 60.24 (99.07 % clamped); ESP_064510_2260 34.88 vs 35.32 (84.97 %); ESP_017355_2260 published
  0.29324219 = the ceiling to 8 significant figures because **100 % of its finite tiles are
  clamped**. The row is bounded above by the calibrator, so it also *understates* how badly `full`
  fails, while the two obs with a genuinely null prediction (ratio 0) are dropped and never
  mentioned. Any future reader who resurrects the F record and compares `full` 0.412 to `pfree`
  0.532 will conclude `full` is the more level-stable variant; it is the least measurable one.
- **Evidence:**
  ```
  src/calibration.py:365-369
      def calibrate_abundance(self, abundance_input) -> np.ndarray:
          """Tier-2 abundance product: input → ``fractional_area``. ..."""
          self._require_fit()
          return np.interp(np.asarray(abundance_input, dtype=np.float64), self._t2[0], self._t2[1])

  # models/deployable_f_center/calibration.npz
  #   t2_x in [7.165e-06, 0.99991630];  t2_y in [0, 0.29324219];  t2_y == 0 for the first 753 knots

  # recomputed from reports/figures/fbuild_cohort_join.parquet (89,145 finite labelled tiles)
  #   variant   p >= t2_x[-1]   ab == ceiling   ab == 0
  #   h1only        0.000%          0.000%       2.793%
  #   full         37.073%         37.073%       8.023%
  #   resid         0.000%          0.000%       7.346%
  #   pfree         0.000%          0.000%      10.833%

  # sd(log10 ratio), ddof=0, recomputed from reports/figures/fbuild_abort_level_vs_labels.csv
  #   h1only 0.328 (n=21)  resid 0.371 (n=21)  pfree 0.532 (n=21)   <- match DECISIONS exactly
  #   full   0.412 requires n=19; over all 21 obs it is undefined (two ratios are 0)

  reports/map_fbuild/*.json already record the diagnostic and were never consulted:
      "abundance_saturated_frac": 0.9974  (E16_N44, full)
      "abundance_saturated_frac": 0.6368  (E12_N44, full)
      ... 0 for h1only on all 26 tiles; <=0.0085 for resid; <=3e-6 for pfree
  ```
- **Self-refutation attempted:** (1) Maybe the mean coinciding with the ceiling is chance — no: it
  matches to 8 significant figures over 6,996 finite tiles, and 100 % of that obs's finite
  `ab_full` cells equal the ceiling. (2) Maybe the level table used raw `prob_raw`, not the
  calibrated layer — no: `f_region_staged.py:6-8` writes `prob_raw` separately and calls it "the
  gate-scoring layer", the abort text says "predicted **abundance**", and `full_pred` lands exactly
  on the calibrator's `t2_y[-1]`, which only the calibrated layer can produce. (3) Maybe it is
  already flagged — grepped `DECISIONS.md` for `saturat`, `ceiling`, `clamp`, `interp`: the
  saturation guard exists only in `bank_calibration_f.py:166-167` (fit cohort, 6.5e-06) and in the
  per-tile sidecars; neither the gate tables, `fbuild_gates.json`, nor the abort entry mentions it.
  (4) Does it overturn the abort? **No** — `h1only`/`resid`/`pfree` have zero ceiling clamping, so
  the rows that carry the argument are clean; this is a record-correctness defect. (5) Not covered
  by R10 (which decomposes head/training-set/input-radiometry, not the calibrator) or R12 (missing
  producer).
- **Fix:** in `fgates.abundance_fidelity` and in whatever produced
  `fbuild_abort_level_vs_labels.csv`, refuse to report `top_ratio` / level ratio when the clamped
  fraction (both ends) exceeds a small threshold, and carry `abundance_saturated_frac` +
  `abundance_at_floor_frac` into the gate table; annotate the `full` row in `DECISIONS.md:5538`
  and `reports/map_fbuild/README.md:20` as "n=19 of 21; 37 % of tiles at the calibrator clamp — not
  a level measurement".

---

### calibration-2 — The F Tier-2 calibrator is fitted on the *un-levelled* per-frame path and reused unchanged for all four variants; DECISIONS retracts exactly this on a false factual ground
- **Severity:** high
- **Liveness:** dead-closed (F build) — bears on the *disqualifying* evidence, not on the verdict
- **Confidence:** high on the facts; medium on the size of the residual distortion
- **Where:** `scripts/bank_calibration_f.py:48,52,193-198`; `scripts/f_region_staged.py:230-234,
  260-264,341-347`; `models/deployable_f_center/calibration.npz` (`meta`);
  `DECISIONS.md:5560-5561` (retraction #3); `DECISIONS.md:5532-5541`

The F Tier-2 map is fitted from `reports/figures/f_leg_b_loio_preds_minnaert_center.csv`, store
`fang_embeddings_f_minnaert_center` — the H1-centred **per-frame, pre-levelling** LOIO predictions.
`f_region_staged.py` then applies that single fixed map to every variant's **levelled composite**
(`h1only`, `full`, `resid`, `pfree`). Quantile matching is a marginal-transfer map, so its validity
depends on the input marginal matching `t2_x`; `bank_calibration_f.py:1-8` makes precisely this
argument to justify not reusing the mosaic layer ("Tier-2 is a **quantile-match** — a
marginal-transfer map — so reusing the mosaic-path knots on F-path P(rich) is a train/deploy
mismatch of exactly the class that killed F pilot leg A ... Measured shift ... CDF L1 0.0358"), and
then never makes it for the per-frame→composite shift, which I measure at CDF-L1 **0.069
(pfree) / 0.088 (resid) / 0.123 (h1only) / 0.460 (full)** — 2–13× the shift that was deemed
disqualifying. `h1only` is by construction the nearest-domain variant (its composite is a median
over the very per-frame probabilities the map was fitted on), so gate 6 and the level table compare
variants through a fixed, strongly convex-at-the-top map whose fidelity is ordered *against*
levelling.

Separately: `DECISIONS.md:5560-5561` retracts "the claim that the calibrator was banked on the
H1-only path and structurally favoured h1only" on the stated ground "*There are no metadata JSONs
in `models/deployable{,_f_center}/`; unsubstantiated*". The provenance does exist — the banked npz
carries it, the fit source is a tracked CSV, and the banking script hardcodes it — and the factual
half of the retracted claim is **correct**.

- **Failure scenario:** the abort's decisive evidence is a *value*-based statistic
  (`mean(predicted abundance) / mean(labelled fa)`) computed through a fixed map fitted on a
  different distribution, with the mismatch growing from `h1only` → `resid`/`pfree` → `full`. The
  published reading — "The spread grows monotonically with the strength of levelling applied ... so
  the instability is attributable to F" (`DECISIONS.md:5545-5546`) — has a competing explanation
  (monotonically growing calibrator-domain mismatch × a convex map) that was raised, and then
  dismissed on a premise that is factually wrong. The confound-free version of the same statistic
  is available for free: `*_prob_raw.tif` exists for all four variants, and
  `fbuild_cohort_join.parquet` carries `p_<variant>` per labelled tile.
- **Evidence:**
  ```
  scripts/bank_calibration_f.py:48,52
      PREDS   = REPO / "reports" / "figures" / "f_leg_b_loio_preds_minnaert_center.csv"
      F_STORE = "fang_embeddings_f_minnaert_center"

  scripts/f_region_staged.py:230-234   (one cal_f, applied to every variant v)
          if cal_f is not None:
              layers[f"{v}_prob"]      = _apply(cal_f.calibrate_prob, p, fin)
              layers[f"{v}_abundance"] = _apply(cal_f.calibrate_abundance, p, fin)

  models/deployable_f_center/calibration.npz  meta  (the "missing" provenance):
      {"n": 153663, "recipe": "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2", "scale": "S32",
       "mode": "one_model", "fit": "pooled_loio_36_f_minnaert_center",
       "head": "deployable_f_center/86c51a5dca220f63",
       "store": "fang_embeddings_f_minnaert_center",
       "fa_marginal": "dataset_v2/labels tile_size_px==32"}

  DECISIONS.md:5560-5561
      3. **Retracted:** the claim that the calibrator was banked on the H1-only path and structurally
         favoured h1only. There are no metadata JSONs in `models/deployable{,_f_center}/`; unsubstantiated.

  # CDF-L1 between each variant's labelled-tile p marginal and t2_x (the fitted reference marginal):
  #   pfree 0.0685   resid 0.0884   h1only 0.1225   full 0.4604      (cf. the 0.0358 that was disqualifying)
  ```
- **Self-refutation attempted:** (1) Is a per-variant re-bank even possible? No — there are no
  per-tile composite LOIO predictions, which is *why* the leg-B CSV was used; so the mismatch is
  structural, and the defect is that it is unquantified and undisclosed while the analogous mosaic
  mismatch was quantified. (2) Does the calibrator change *ranking*? No — it is monotone, so
  gates 1/2/3/5 (all on `prob_raw`) and gate 6's Spearman are unaffected; only gate 6's
  `top_ratio`/`marginal_l1`/`rich_bin_rmse` and the level table are. I have scoped the claim to
  those. (3) Does the retraction have a second ground? Read `DECISIONS.md:5552-5566` in full — it
  gives only the metadata ground. (4) Is this R10? No — R10 lists head / training set / input
  radiometry and *accepts* the retraction as merely "partly substantiated"; it does not identify
  the calibrator mechanism, the fit-source path, or the npz `meta`. (5) Does it overturn the abort?
  No: gate 5 (`prob_raw`, uncalibrated) and the `h1only`-vs-`resid` decomposition stand.
- **Fix:** recompute the level statistic on `prob_raw` (or on abundance under a per-variant
  re-bank) before any future comparison; correct `DECISIONS.md:5560-5561` to cite
  `bank_calibration_f.py:48,52` and the npz `meta`, and record the per-variant CDF-L1 shift beside
  the gate-6 table.

---

### calibration-3 — A banked calibrator is bound to no head: `load()` ignores the provenance it stores, `--model` and `--calibration` are independent, and the map records neither
- **Severity:** medium
- **Liveness:** live-shipped (the circum-Chryse abundance layer)
- **Confidence:** high (code-level; no known mis-paired artifact)
- **Where:** `src/calibration.py:378-385`; `scripts/map_region.py:56-57,86-92,239-247,268-269,
  313-317`; `scripts/map_pilot.py:102-103`; `scripts/parity_check.py:39,92`;
  `scripts/bank_calibration_f.py:180-183`

`CalibrationLayer.load` reads `meta` and validates nothing. `map_region.py` takes `--model` and
`--calibration` as unrelated flags whose defaults are only *conventionally* paired
(`DEFAULT_CALIBRATION = models/deployable/calibration.npz`, while `--model` defaults to
`resolve_model_dir(None)` = the lexicographically **last** subdirectory of `models/deployable`, not
the newest). `bank_calibration_f.py:180-183` hard-errors on writing over the mosaic calibrator, but
nothing guards *reading* the wrong one — the direction that actually corrupts a map. Compounding
it: `models/deployable/calibration.npz`'s meta has **no** `head` key at all
(`{"n": 161005, "abundance_source": "p_rich", "recipe": ..., "scale": "S32", "mode": "one_model",
"fit": "pooled_loio_38"}`), and the shipped per-tile sidecars + `region_manifest.json` record only
`"calibrated": true` / `"isotonic": true` — never which `.npz` produced the abundance layer.

- **Failure scenario:** run `map_region.py --model models/deployable_f_center/86c51a5dca220f63`
  (or retrain a head so `resolve_model_dir` picks a different directory) and omit
  `--calibration`. The mosaic knots are applied to a different head's probabilities — the exact
  train/deploy mismatch `bank_calibration_f.py` was written to prevent, measured there at CDF-L1
  0.0358 — with no error, no warning, and no record in the output. R09 shows both heads carry the
  identical `recipe_hash` `86c51a5dca220f63`, so even a recipe-hash check would not catch it, and
  the sidecar's `recipe_hash` field cannot distinguish the two. Every abundance number on the map
  would be silently wrong while `prob_raw` looked fine.
- **Evidence:**
  ```
  src/calibration.py:378-385
      @classmethod
      def load(cls, path: str | Path) -> "CalibrationLayer":
          ...
          d = np.load(path, allow_pickle=False)
          meta = json.loads(str(d["meta"]))
          return cls((d["t1_x"], d["t1_y"]), (d["t2_x"], d["t2_y"]), meta)   # meta never checked

  scripts/map_region.py:56-57,267-269
      DEFAULT_MODEL_PARENT = REPO_ROOT / "models" / "deployable"
      DEFAULT_CALIBRATION  = DEFAULT_MODEL_PARENT / "calibration.npz"
      ap.add_argument("--model", default=None, help="deployable head dir (default: latest)")
      ap.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), ...)

  scripts/map_region.py:86-92
      hits = sorted(p for p in DEFAULT_MODEL_PARENT.glob("*") if (p / "recipe.json").exists())
      return hits[-1]                      # lexicographic, not newest

  reports/map_region/E-12_N32.json        (no calibrator identity anywhere)
      "calibrated": true, "isotonic": true, "abundance_mean": 0.0005571906245347298
  ```
- **Self-refutation attempted:** (1) Is there a guard elsewhere? Grepped every
  `CalibrationLayer.load` call site: `map_region.py:297`, `map_pilot.py:155`,
  `striping_a1_map.py:222`, `f_region_staged.py:345-346`, `parity_check.py:67` — none validates
  meta; `f_region_staged.py:342-344` errors only if the file is *missing*. (2) Does
  `parity_check.py` pin the pairing? It re-runs the same `(model, calibration)` pair it is given,
  so it detects drift in a *fixed* pair, not a wrong pair (and R09 notes `deployable_f_center` has
  no `parity_ref.npz` at all). (3) Deliberate? Grepped `DECISIONS.md` and `PLAN_Calibration.md` for
  `head`/`bind`/`keyed` — the F re-bank ruling (`DECISIONS.md:5040-5048`) treats head↔calibrator
  pairing as *load-bearing*, which argues for the check rather than against it. (4) Hypothetical? Yes
  — no shipped artifact is known to be mis-paired, which is why this is medium and not high.
- **Fix:** write `head` (and the head's `model_hash`, not `recipe_hash`) into the meta of both
  banked layers; have `CalibrationLayer.load` accept an optional `expect_head=` and raise on
  mismatch; make `map_region.py`/`map_pilot.py` pass the resolved head and copy the calibrator's
  filename + meta into the per-tile sidecar and `region_manifest.json`.

---

### calibration-4 — `bank_calibration.py` writes the shipped calibrator *before* it computes any gate, exits 0 whatever the result, and never evaluates 4 of the 6 declared §6 metrics
- **Severity:** medium
- **Liveness:** live-shipped (this script produced `models/deployable/calibration.npz`)
- **Confidence:** high on the code path; I verified the *current* artifact would have passed
- **Where:** `scripts/bank_calibration.py:42,60-74`; `PLAN_Calibration.md:355-359`

`layer.save(OUT)` runs at line 42; the ECE and `top_ratio` gates are computed at lines 60-67 and
merely *printed* as `PASS`/`FAIL`; `main()` returns 0 unconditionally at line 74. So a
gate-failing calibrator is already on the shipped path before anyone can see the verdict, and no
caller or CI can detect it. The round-trip check (lines 69-73) is run on only the first 4,096 rows
and prints `MISMATCH` rather than raising — `bank_calibration_f.py:238-244` does it correctly
(writes last, `--dry-run`, `raise SystemExit` on any knot mismatch), so the correct pattern exists
in the sibling script. `PLAN_Calibration.md:355-359` declares six Tier-1/Tier-2 metrics; the
script evaluates two. `AUC within ±0.005`, `Brier`, `near-zero pred within ±3 pts of truth`,
`marginal-L1 ↓ ≥ 50 %` and `Spearman & NDCG@5 % within ±0.01 of the uncalibrated recipe (the hard
constraint)` are never compared to anything — three of them are printed as bare values with no
baseline, and AUC/Brier/NDCG are not computed at all.

- **Failure scenario:** re-bank after any upstream change (new labels, a re-trained head, a
  different `predictions.parquet`); if the Tier-2 `top_ratio` drifts out of [0.8, 1.2] or the
  monotone map's zero-floor ties cost more Spearman than the declared ±0.01, the artifact is
  already written, the process exits 0, and the next `map_region.py` run renders an
  out-of-specification abundance map. Nothing on disk records that a gate failed.
- **Evidence:**
  ```
  scripts/bank_calibration.py:39-42
      layer = CalibrationLayer.from_loio_predictions(df, meta={...})
      layer.save(OUT)                                   # <- written before any gate is computed

  scripts/bank_calibration.py:62-67
      print(f"  [LOIO bound] Tier-1 ECE {ece_loio:.3f}  (gate <=0.05: "
            f"{'PASS' if ece_loio <= 0.05 else 'FAIL'})", flush=True)
      ...  f"(gate top in [0.8,1.2]: {'PASS' if 0.8 <= m_loio['top_ratio'] <= 1.2 else 'FAIL'})"

  scripts/bank_calibration.py:72-74
      print(f"  save/load round-trip max |d| = {d:.2e} ({'OK' if d < 1e-9 else 'MISMATCH'}) ...")
      return 0                                          # <- 0 regardless of PASS/FAIL

  PLAN_Calibration.md:355-359 (declared, never evaluated by the banking script)
      - **Tier-1:** ECE <= 0.05; AUC within +/-0.005; reliability diagonal; Brier.
      - **Tier-2 point:** top-bin ratio in [0.8,1.2]; near-zero pred within +/-3 pts of
        truth; marginal-L1 down >= 50 %; **Spearman & NDCG@5 % within +/-0.01** of the
        uncalibrated recipe (the hard constraint).
  ```
- **Self-refutation attempted:** (1) Is the current artifact actually out of spec? I recomputed the
  two unevaluated constraints that could plausibly bite on the shipped layer: pooled
  Spearman(fa, raw p) 0.629745 → Spearman(fa, qmatch abundance) 0.631862 (**Δ +0.0021**, inside
  ±0.01; per-image paired median Δ −0.0000, worst −0.0190), and pooled AUC 0.848371 → isotonic
  0.848671 (**Δ +0.0003**, inside ±0.005 — an exact reproduction of `DECISIONS.md:3847`). So the
  shipped calibrator passes; the defect is that nothing would have stopped it if it had not. That
  caps this at medium. (2) Is the ordering deliberate so the artifact exists for the LOIO step? No
  — the LOIO bound at lines 56-61 refits from `df`, never from `OUT`. (3) Is a human meant to read
  the printout? The docstring gives a bare `Usage:` line; README/SHERLOCK_RUN reference only
  `bank_calibration_f.py`, never this script, so there is no documented "check the gates" step.
- **Fix:** move `layer.save(OUT)` after the gate block, gate it on both PASS conditions (or add
  `--force`), `return 1` on FAIL, raise on a round-trip mismatch over the full vector, and add the
  four declared-but-unevaluated comparisons (they are 4 lines: `roc_auc_score`, `brier_score_loss`,
  `spearmanr` vs raw, and the uncalibrated `marginal_l1` baseline).

---

### calibration-5 — The shipped abundance product has an undocumented hard ceiling of 0.2932, and its whole top decade rests on 39 label tiles
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high (knots + label distribution measured; live exposure measured)
- **Where:** `src/calibration.py:198-231,365-369`; `models/deployable/calibration.npz`;
  `scripts/map_region.py:239-247`; `PLAN_Calibration.md:262-263,355-359`

The one-model Tier-2 map is `np.interp` on 4,000 knot pairs whose `y` side is the cohort
`fractional_area` quantiles. Consequences never stated anywhere: (a) the map **cannot** output
more than `t2_y[-1] = 0.29324219`, the single richest tile in the 38-image cohort, so off-HiRISE
terrain rockier than anything in the cohort reads a hard 29 %; (b) the last knot pair spans
`t2_y` 0.16234 → 0.29324 over `t2_x` 0.999748 → 0.999912, so everything the map can say above
fa ≈ 0.162 is estimated from the **39 tiles of 161,005** with fa > 0.162 (and the ceiling from
exactly 1 tile); (c) nothing on the deployment path measures how often the clamp is hit —
`bank_calibration_f.py:166-167` computes `ceiling`/`saturated_frac` for the F path, and
`f_region_staged.py:247-248` records `abundance_saturated_frac` per tile, but `map_region.py`'s
sidecar records only `abundance_mean` and `bank_calibration.py` prints no ceiling at all.
`PLAN_Calibration.md:262-263` flags the generic risk ("Marginal-match assumes in-cohort") but not
the clamp or the 39-tile top.

- **Failure scenario:** extend the map beyond circum-Chryse (PLAN_RegionalMap's stated direction is
  the full Murray index) into terrain where the head saturates. Those tiles all read exactly
  0.2932 with no flag, an abundance ceiling is reported as a measurement, and any regional mean or
  THEMIS-comparison statistic is biased low in the rockiest terrain — the terrain the product most
  wants to identify.
- **Evidence:**
  ```
  # models/deployable/calibration.npz
  #   t2_x last 5: [0.999477 0.999558 0.999643 0.999742 0.999912]
  #   t2_y last 5: [0.139102 0.142860 0.148668 0.161680 0.293242]
  # dataset_v2/labels/*.parquet, tile_size_px == 32, n = 161,005:
  #   tiles with fa > 0.139102 : 161      tiles with fa > 0.162341 : 39      max fa = 0.2932421875

  # measured live exposure over all 26 reports/map_region/*_abundance.tif (56,870,060 finite cells):
  #   cells at the ceiling            :   1  (1.8e-06 %)     [E16_N44]
  #   cells above the top-cliff knee  : 296  (5.2e-04 %)
  ```
- **Self-refutation attempted:** I tried to kill this by measuring the actual shipped exposure, and
  it *nearly* dies — 1 clamped cell in 56.9 M means the current 26-tile map is essentially
  unaffected, which is why this is low and not high. It survives as (a) an undocumented hard bound
  on a product described in absolute `fractional_area` units, (b) a top-of-scale estimated from 39
  tiles, and (c) a missing diagnostic that the F path already implements — i.e. it is cheap to fix
  and expensive to discover later. Checked `README.md`, `PLAN_Calibration.md`,
  `PLAN_RegionalMap.md`, `docs/*.md`, `dataset/DATA_DICTIONARY.md` for `0.293` / `ceiling` /
  `clamp`: nothing.
- **Fix:** record `abundance_saturated_frac` (and the fraction in the top knot interval) in
  `map_region.py`'s per-tile sidecar exactly as `f_region_staged.py:247-248` does, and state the
  ceiling + the 39-tile support of the top decade in `PLAN_Calibration.md` §6 and
  `dataset/DATA_DICTIONARY.md`.

---

### calibration-6 — `near_zero_pred` (< 1e-4) is documented and reported as comparable to `near_zero_true` (== 0); the truth has 0.54 % of tiles strictly between them
- **Severity:** low
- **Liveness:** live-shipped (reported metric) + dead-closed (F gate 6)
- **Confidence:** high (measured)
- **Where:** `src/calibration.py:257-259,273-274`; `tests/test_calibration.py:160`;
  `scripts/bank_calibration.py:64-65`; `DECISIONS.md:3795`;
  `reports/figures/fbuild_calibration_layer_compare.csv`

`near_zero_pred = mean(yp < 1e-4)` and `near_zero_true = mean(yt <= 0)` are different events, and
the docstring instructs the reader to compare them ("share of predictions < 1e-4 (compare to the
true exact-zero share)"). Because `min_size_m = 1.4105` puts the smallest single-boulder tile at
fa = 3.90625e-05, the labels have **0.541 % of tiles in (0, 1e-4)**, so a *perfectly*
marginal-matched calibrator necessarily reports `near_zero_pred` ≈ `near_zero_true` + 0.0054. The
reported residual is therefore a definition artifact, not a calibration shortfall — and the unit
test hides it with `abs=0.03`, 5× the size of the effect.

- **Failure scenario:** `DECISIONS.md:3795` reads "near-zero pred 1.8 %→**18.6 %** (= truth)" —
  18.6 % is `P(fa < 1e-4) = 0.18547`, while the truth it is equated to is
  `P(fa == 0) = 0.18006`. `reports/figures/fbuild_calibration_layer_compare.csv` records
  `near_zero_pred 0.19370` vs `near_zero_true 0.18807` for the re-banked F layer, a gap of 0.00563
  that is exactly the F cohort's (0, 1e-4) mass. `PLAN_Calibration.md:356-357` declares a gate
  "near-zero pred within ±3 pts of truth" against this mismatched pair, so the gate has a built-in
  0.5 pt offset — harmless at ±3 pts, wrong if anyone tightens it or reads the residual as a
  physical over-prediction of zeros.
- **Evidence:**
  ```
  src/calibration.py:257-258,273-274
      - ``near_zero_pred`` : share of predictions < 1e-4 (compare to the true exact-zero share)
      - ``near_zero_true`` : share of truth exactly zero
        "near_zero_pred": float(np.mean(yp < 1e-4)),
        "near_zero_true": float(np.mean(zero)),          # zero = yt <= 0

  tests/test_calibration.py:160
      assert np.mean(qm < 1e-4) == pytest.approx(np.mean(df.fractional_area <= 0), abs=0.03)

  # dataset_v2/labels/*.parquet, tile_size_px == 32, n = 161,005
  #   P(fa == 0)          = 0.180063
  #   P(fa < 1e-4)        = 0.185473        <- what near_zero_pred converges to after qmatch
  #   P(0 < fa < 1e-4)    = 0.005410
  #   min positive fa     = 3.90625e-05
  ```
- **Self-refutation attempted:** (1) Maybe no label sits below 1e-4, making the two events
  identical — measured: 871 of 161,005 tiles do, min positive fa 3.9e-05. (2) Maybe the 1e-4 is a
  deliberate "effectively zero" tolerance — plausible for a *prediction*, but then it must not be
  compared to an exact-zero truth share, and both the docstring and `PLAN_Calibration.md:356` do
  exactly that. (3) Deliberate per DECISIONS? Grepped `near_zero` / `near-zero`: every mention
  treats the two as the same quantity. (4) Consequence is 0.5 pt, hence low.
- **Fix:** define `near_zero_true` with the same threshold (`mean(yt < 1e-4)`), or add
  `exact_zero_pred`/`exact_zero_true` alongside, and tighten the test tolerance below 0.005 so the
  distinction cannot silently reappear.

---

## Refuted by my own check

- **`compression_metrics["low_over"]` is degenerate** (`yt[zero].mean()` is identically 0, so the
  `max(..., 1e-9)` floor makes it `1e9 × mean(pred)` rather than a ratio). Already known and
  handled: `src/fgates.py:272-274,289` documents it ("it reads ~2e6") and drops the key, and
  `tests/test_fgates.py:319` pins the exclusion. Only the docstring
  (`src/calibration.py:257`) and a dead probe (`scripts/probes/_diag_calibration_preview.py:52,57`)
  still present it as a ratio.
- **"All calibrators are monotone ... so AUC / Spearman / NDCG are invariant by construction"**
  (`src/calibration.py:10-11`) is false for isotonic and qmatch, which are only *non-strictly*
  monotone and therefore create ties. But the project measured the real effect and recorded it
  (`DECISIONS.md:3847`: "a single GLOBAL calibrator is AUC-exact (isotonic +0.0003, beta +0.0000),
  and ties are harmless at n=161k"), and I reproduced +0.000300 exactly. Docstring overstatement
  with a measured, negligible consequence — not a defect.
- **`compare_layers` in `bank_calibration_f.py` scores the re-banked layer in-sample and the
  mosaic layer out-of-domain** (so its `marginal_l1` 5.4e-06 vs 2.1e-03, quoted at
  `DECISIONS.md:5045-5046`, overstates the domain-shift advantage ~65×: the honest LOIO number
  from the same run is 3.5e-04). Survives as a real asymmetry but does **not** clear the bar for a
  finding slot: `bank_calibration.py:44-48` spells out the identical caveat for the mosaic layer
  ("isotonic fits its training ECE to ~0 by construction; qmatch matches its training marginal
  exactly"), the honest LOIO bound is computed and banked in the same CSV, and the number
  supported the conservative decision (re-bank) anyway.
- **`BetaCalibrator`'s monotonicity fallback can produce a constant map** (drop a feature, refit,
  then `max(coef, 0)` can zero the survivor, leaving `sigmoid(c)` and AUC 0.5). Beta lost the
  Tier-1 bake-off (`DECISIONS.md:3843-3847`); the only caller is
  `scripts/probes/_diag_tier1_beta.py`; and the branch needs the refit coefficient to flip sign,
  which I could not trigger. Dead + implausible.
- **`QuantileMatcher.fit` degenerates for `len(ref_pred) == 1`** (`np.linspace(0, 1, 1)` → one
  knot → constant map) **and `n_quantiles` is chosen from `len(ref_pred)` only**, ignoring
  `len(ref_true)`. Unreachable: every call site passes ≥ 10⁵ rows, and the fit sorts each side
  independently so unequal lengths are handled by `np.quantile`.
- **Ensemble averaging space could mismatch the calibrator's fit distribution** (probability vs
  logit). Checked: `scripts/probes/_fm_freeze_window.py:240` builds `predictions.parquet` as
  `base[[f"p{s}"...]].mean(axis=1)` over sigmoid probabilities, and
  `src/modeling/mlp_head.py:407-414` averages member `predict()` (also sigmoid) outputs. Same
  space; refuted.
- **`bank_calibration.py` might violate the OpenMP import order (invariant 9).** It imports only
  numpy/pandas/scipy/sklearn via `src.calibration`; no torch anywhere on that path, so
  `import src.modeling` is not required. (`bank_calibration_f.py:40` does it correctly.)
- **The `bank_calibration.py` label join could duplicate or drop rows.** Verified 1:1:
  161,005 prediction rows ⋈ 161,005 label rows at `tile_size_px == 32` → 161,005 merged rows over
  38 images, zero loss.
- **`reliability_curve` could silently absorb NaN predictions into the top bin.** True
  (`np.digitize(nan)` → last bin), but the NaN then propagates through `acc - conf` and ECE
  returns NaN, which is loud. No call site passes NaN.
- **The live regional map could be materially clamped by the qmatch ceiling.** Measured over all
  26 shipped `*_abundance.tif`: 1 cell of 56,870,060. Refuted as a live numerical error (retained
  only as the documentation/diagnostic gap in calibration-5).

## Verified clean

- **The LOIO-honesty protocol.** `loio_calibrate` (`src/calibration.py:284-300`) fits on `~held`
  and scores `held`, per group, with a NaN-initialised output; `bank_calibration.py:56-59` uses it
  for both tiers; `bank_calibration_f.py:106-119` hand-rolls the same split correctly (`keep =
  ~held`, `fa_ret` excludes the held obs). The in-sample-vs-LOIO labelling in both scripts is
  accurate, and the in-sample ECE of 5.16e-18 in `fbuild_calibration_f_summary.csv` is *exactly*
  zero by construction (ECE bins on the calibrated value, each isotonic block is constant so it
  falls entirely in one bin, and within a block mean(y) == the fitted value) — the comment at
  `bank_calibration.py:46-48` says precisely this.
- **Isotonic serialisation.** `IsotonicCalibrator.knots()` → `np.interp` reproduces sklearn's
  `predict` exactly: sklearn clips to `[X_min_, X_max_]`, and `X_thresholds_[0] == X_min_` /
  `[-1] == X_max_` because `_build_y`'s `keep_data` always retains both endpoints.
  `tests/test_calibration.py:166-170` pins it; verified on the banked knots (236 / 232 knots,
  `t1_y ∈ [0, 1]`, monotone).
- **Monotonicity of both banked maps.** `np.all(np.diff(...) >= 0)` is True for `t1_x`, `t1_y`,
  `t2_x`, `t2_y` in both `calibration.npz` files, so no ranking metric computed after calibration
  can be re-ordered.
- **The zero atom.** `t2_y` is exactly 0 for its first 721 knots in the mosaic layer, and
  721/4000 = 0.18025 matches the measured label zero share 0.180063 — the qmatch reproduces the
  true-zero mass without inventing abundance below it, and `predict` cannot emit a negative value.
- **`CalibrationLayer.save`/`load` round trip.** `np.savez` of four float64 knot arrays plus a
  0-d JSON string, `allow_pickle=False` on load; byte-exact (`tests/test_calibration.py:197-206`
  uses `np.array_equal`, and `bank_calibration_f.py:240-244` raises on any knot difference > 1e-12).
  A missing key raises `KeyError` — loud.
- **NaN handling on the map path.** `src/mapping.py:259-278` and `f_region_staged.py:286-292`
  calibrate only finite cells and keep nodata as NaN; `write_geotiff` uses `nodata=np.nan`, so the
  legitimate zero-abundance value is never confused with nodata (and `fgates.read_layer`'s
  `np.isfinite(nd)` guard means it never rewrites real zeros).
- **`src/reliability.py`** (dead-closed, PLAN_FM §2.7 deferred): `MahalanobisNovelty` is a correct
  truncated-PCA Mahalanobis distance (`proj @ components.T`, scaled by `1/sqrt(explained_variance +
  eps)`), `KNNNovelty`'s reference subsample is seeded and its scored set is disjoint from the
  reference in the LOIO use, NaN rows score NaN in both, and `aggregate_per_image` drops all-NaN
  images. No defect found.
- **`compression_metrics` selection convention.** `top = yt > 1e-2` (selection on truth, the
  project's rich/poor cut, strict `>`), `zero = yt <= 0`, predictions clipped at 0 — consistent
  with `src/modeling/binary_target`'s strict threshold, and `bank_calibration_f.py:114-119`'s
  set-based numerator/denominator is order-independent so the missing `ti`/`tj` join does not
  bias it.
- **The pooled-vs-per-image disclosure in `bank_calibration_f.py:121-128,225-228`** is honest: it
  reports both, states which is the declared gate quantity and why the per-image median is
  harsher, and `DECISIONS.md:5049-5053` records the mid-run correction.

## Coverage note

**Read in full:** `src/calibration.py`, `src/reliability.py`, `scripts/bank_calibration.py`,
`scripts/bank_calibration_f.py`, `tests/test_calibration.py`, `tests/test_reliability.py`,
`src/mapping.py:150-290`, `src/fgates.py:246-310`, `scripts/f_region_staged.py:200-300`,
`PLAN_Calibration.md` §§3-7, and the `DECISIONS.md` entries for 2026-06-14c / -15 / -16 /
2026-07-28 / 2026-07-30b (grepped by term, never read linearly).

**Grepped only:** `scripts/probes/_diag_tier*`/`_evidence_*` (dead Stage-0/2 diagnostics; I read
`_diag_calibration_preview.py` because it still reports `low_over`), `notebooks/_build_23.py`
(read for its AUC-exactness and rank-preservation claims only — notebook-internal metric logic is
the `notebooks` area), `scripts/map_pilot.py`, `scripts/striping_a1_map.py`,
`scripts/parity_check.py` (calibration call sites only).

**Numerical checks I ran** (read-only, over committed artifacts + two `.npz` + the label
parquets): reproduced the abort level table's four `sd(log₁₀)` values exactly (ddof=0), which is
how the `full` row's n=19 was established; measured the per-variant clamp fractions and
composite-vs-reference CDF-L1 from `fbuild_cohort_join.parquet`; measured the label `fa`
distribution near 0 and at the top; measured the shipped map's ceiling exposure over all 26
`*_abundance.tif`; recomputed the isotonic AUC delta (+0.000300) and the qmatch Spearman delta
(+0.002117). Scratch scripts were written outside the repo.

**Could not check:** (1) whether the *deployed* `DeployableHead`'s P(rich) marginal matches the
pooled-LOIO marginal that `t2_x` was fitted from — the analogous mosaic-vs-F shift was measured at
CDF-L1 0.0358 and treated as disqualifying, but reproducing it for the mosaic path needs a GPU
inference pass over the cohort embeddings, which the rules of engagement exclude; the cheap control
is to run the existing head over `fang_embeddings` and compare `np.quantile` against `t2_x`.
(2) the producer of `fbuild_abort_level_vs_labels.csv` — it does not exist in the repo (R12), so I
could verify only the artifact's contents, not the code that made it (I confirmed *which* layer it
used by showing its values land exactly on `t2_y[-1]`). (3) `models/deployable/` is entirely
untracked, so I could not check the shipped calibrator's history with `git log`; the code itself
notes it "has no versioning" (`bank_calibration_f.py:34-35`). (4) I did not run pytest, any
notebook, or any banking script.
