# Review area: probes-stage7

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-02
- **Verification:** self-refuted (single-agent pass; not independently verified). Every number below was
  **recomputed from the on-disk artifacts the docs name** (`cache_v2/stage7/*.parquet`,
  `cache_v2/hirise_color/*_COLOR.LBL`, `dataset_v2/features_colour*.parquet`,
  `dataset_v2/stage7d_attribution_shadow_*.parquet`, `dataset_v2/terrain_classification_v2.parquet`);
  no probe was executed and no imagery or network was touched. Snippet inputs are quoted so a verifier
  can re-run them.

Scope: the 18 probes listed for this area in `_prompts_probes.md` §2. Triage was by citation first
(`DECISIONS.md`, `docs/*.md`, `PLAN_*.md`, `PROMOTION_QUEUE.md`, `README.md`, `reports/figures/`,
`notebooks/_build_*.py`), then a statistic-level audit of the load-bearing ones.

---

## Findings

### probes-stage7-1 — The Stage-7.0 feasibility probe computes every band ratio on **raw uint16 DN**, not I/F, while its own docstring and the reader-facing writeup say I/F. Applying the PDS `I/F = DN·SCALING_FACTOR + OFFSET` conversion the *same session* used elsewhere **flips the sign** of the headline Test-A result and **kills the dust discriminator that carried the GO** (+0.159, p = 0.037 → **+0.018, p = 0.81**)

- **Severity:** high
- **Liveness:** dead-closed programme (Stage 7 is PARKED), but the numbers are the recorded **GO
  decision** for the whole Stage-7a–7e build and they sit in `docs/compositional.md`, a reader-facing
  writeup routed from `docs/index.md`
- **Confidence:** high (reproduced from the committed-in-spirit cached parquets + the PDS labels; the
  conversion is stated verbatim in the LBL and applied by the project's own Stage-7c code)
- **Where:** `scripts/probes/_stage7_feasibility.py:150-151` and `:210` (the only pixel-value reads),
  `:275-280` (the docstring that asserts I/F), `:283-289` (Test-A ratio features), `:317-323` (Test-B
  ratio features); `src/colour.py:168-193` (`region_means` returns raw band means) vs
  `src/colour.py:19-20` (the conversion it documents but never applies); consumers
  `DECISIONS.md:1654-1673` (the verdict table + "Final verdict: PASS (a)"), `:1676-1684`
  ("opposite-direction compositional shifts", "Supports H_local"), `docs/compositional.md:213-227`
  (§3.1, "extract the per-band mean **I/F**"), `notebooks/_build_14.py:346-373` (the partial
  correlation, which reads the same DN parquet)

`region_means` returns `arr[b][valid].mean()` where `arr = ds.read(...)` on the COLOR.JP2 — a 10-bit
DN array (`SAMPLE_BIT_MASK = 2#0000001111111111#`). The PDS label gives
`I/F = DN·SCALING_FACTOR + OFFSET` with a **large additive** `OFFSET` (0.0316–0.0458 I/F, i.e. 22–30 %
of the product's full I/F range), so a DN ratio is **not** an I/F ratio: it is
`(I/F_a − O)/(I/F_b − O)`, which is strongly brightness-dependent. That is fatal for exactly the two
things Test A was built to do — compare a *darker* boulder interior against a *brighter* ring, and
compare band ratios across images at different incidence.

- **Failure scenario:** `DECISIONS.md:1654-1673` records the trio table and "**Final verdict: PASS (a)
  — composition signal detected (dust-controlled)**", and `docs/compositional.md:224-229` tells the
  reader `ESP_055253_2245` "passed the partial-correlation dust discriminator with a … partial
  correlation of +0.16, p = 0.037 … motivating the full Stage 7a–7e build". Recomputed on the same
  cached per-polygon / per-tile parquets with the LBL conversion applied:

  **Test A (paired, interior − ring), published DN vs corrected I/F:**

  | ObsId | feature | DN: mean diff / d / p (published) | I/F: mean diff / d / p |
  |---|---|---|---|
  | ESP_042964_2160 | IR/BG | **+0.398 / +0.943 / 2.5e-34** | **−0.0710 / −1.858 / 7.4e-44** ← sign flip |
  | ESP_042964_2160 | dust_index RED/BG | **+0.266 / +0.767 / 1.2e-27** | **−0.0816 / −2.234 / 3.1e-44** ← sign flip |
  | ESP_042964_2160 | IR/RED | +0.0172 / +1.275 / 3.4e-38 | +0.0081 / +0.972 / 2.4e-31 |
  | ESP_054000_2255 | IR/BG | +0.0227 / +0.046 / **0.82 (null)** | −0.0548 / −0.585 / **3.7e-22** |
  | ESP_054000_2255 | dust_index RED/BG | +0.0355 / +0.086 / **0.65 (null)** | −0.0458 / −0.690 / **1.7e-28** |
  | ESP_055253_2245 | IR/BG | −0.158 / −0.341 / 1.8e-12 | −0.0141 / −0.431 / 5.7e-15 |
  | ESP_055253_2245 | dust_index RED/BG | −0.0921 / −0.222 / **3.4e-7** | −0.0023 / −0.076 / **0.15 (null)** |

  **Test B / the dust discriminator (the GO statistic):**

  | ObsId | statistic | DN (published) | I/F |
  |---|---|---|---|
  | ESP_055253_2245 | marginal r(rich, IR/BG) | +0.252, p = 8.3e-4 | +0.123, p = 0.108 |
  | **ESP_055253_2245** | **partial r(rich, IR/BG \| dust)** | **+0.159, p = 0.0368** | **+0.018, p = 0.811** |
  | ESP_042964_2160 | partial r(rich, IR/BG \| dust) | +0.070, p = 0.404 | −0.046, p = 0.581 |
  | ESP_054000_2255 | dust_index d, rich vs poor | **+0.309, p = 0.0117** | **−0.081, p = 0.834** |

  Three recorded conclusions do not survive: (i) "The two boulder populations have *opposite-direction*
  compositional shifts (042964 redder …, 055253 bluer). Suggests different source / transport
  histories" (`DECISIONS.md:1676-1678`) — in I/F **all three images shift the same way** (interior
  IR/BG *below* ring); (ii) "ESP_054000_2255 … shows no compositional signal at the per-polygon scale …
  Supports H_local for that image" (`:1682-1684`) — in I/F it is d = −0.59, p = 3.7e-22; (iii) the
  "(dust-controlled)" half of the final verdict — no image's IR/BG↔rich association survives dust
  control in I/F. Pass criterion (a) itself (`p<0.05 ∧ |d|>0.3` anywhere) still passes, so the *GO* is
  not overturned, but the reason recorded for it is.

  **Two corollaries, both provable:**
  - `_stage7_feasibility.py:277-280` and `docs/compositional.md:184-191` both assert the Lambertian
    correction "cancels … in **all band ratios** (Test A and the partial-dust discriminator are
    Lambertian-invariant)". With a nonzero `OFFSET` that is false for DN: if `I/F = A·cos i` then
    `DN_a/DN_b = (A_a·cos i − O)/(A_b·cos i − O)`, which depends on `cos i`. The trio spans
    cos(i) = 0.759 / 0.700 / 0.496, so the *cross-image* DN-ratio comparison in the verdict table is
    illumination-confounded as well.
  - The published IR/BG values are physically impossible for Mars. Test A reports interior/ring IR/BG
    of **4.77–6.44**; the correct I/F values are **1.76–2.54**, and the project's own Stage-7c cohort
    number is `dust_index` p50 = 1.95 (`DECISIONS.md:1869`). A 2.4× discrepancy in the same named
    quantity sat in two DECISIONS entries a month apart.

- **Evidence:**
  ```python
  # scripts/probes/_stage7_feasibility.py:150-151 -- the only pixel reads (Test A)
  interior = colour.region_means(arr, interior_mask, min_pixels=MIN_POLYGON_PIXELS)
  ring     = colour.region_means(arr, ring_mask,     min_pixels=MIN_RING_PIXELS)

  # :277-280 (summarise_test_a docstring)
  """... Lambertian correction is multiplicative (1/cos(i)) and cancels in interior-ring
  differences AND in band ratios -- so we work on raw I/F and just record cos(i) for
  reproducibility."""            # <- it is raw DN, not raw I/F

  # src/colour.py:188-193 -- returns the mean of the DN array, unconverted
  return {"n_pixels": n,
          "IR":  float(arr[0][valid].mean()),
          "RED": float(arr[1][valid].mean()),
          "BG":  float(arr[2][valid].mean())}
  ```
  ```python
  # scripts/run_stage7c_features.py:186-192 -- the SAME region_means output, done right
  # COLOR.JP2 stores raw uint16 DN. Convert mean DN -> I/F via the
  # COLOR.LBL's per-image scaling: I/F = DN * scaling_factor + offset.
  ir_iof  = means["IR"]  * lbl.scaling_factor + lbl.offset
  red_iof = means["RED"] * lbl.scaling_factor + lbl.offset
  bg_iof  = means["BG"]  * lbl.scaling_factor + lbl.offset
  ```
  ```
  cache_v2/hirise_color/ESP_042964_2160_COLOR.LBL
      /* I/F = (DN * SCALING_FACTOR) + OFFSET */
      SCALING_FACTOR = 1.54838931378537e-04
      OFFSET         = 0.045827398528133          <- 22 % of the product's full I/F range
      SAMPLE_BIT_MASK = 2#0000001111111111#       <- 10-bit DN, 0..1023
  ```
  ```
  cache_v2/stage7/test_a_summary.parquet, ESP_042964_2160
      IR_over_BG   mean_interior 6.4379  mean_ring 6.0399  d +0.943  p 2.5e-34   (DN)
      -> I/F       mean_interior 2.4662  mean_ring 2.5372  d -1.858  p 7.4e-44
  ```

- **How it survived — the cross-check that never compared.** `scripts/probes/_verify_stage7c_trio.py`
  exists for exactly this: its docstring is *"Cross-check Stage 7c trio output against Stage 7.0 Test B
  findings … compares to the Stage 7.0 verdict table in DECISIONS.md 2026-05-31"*, and it lists the
  three verdicts it is checking. But `main()` (`:36-62`) only **prints** the Stage-7c side — it never
  loads `cache_v2/stage7/test_b_per_tile.parquet`, never computes a delta, and has no assertion. The
  two artifacts cover the **identical 542 tiles**, so the comparison was one merge away:
  median IR/BG **2.53 vs 5.95** (ESP_042964_2160), **2.43 vs 4.80** (ESP_054000_2255), **1.77 vs 4.79**
  (ESP_055253_2245); and ESP_054000_2255's `dust_index` Cohen's d is **−0.081 (7c) vs +0.309 (7.0)** —
  the very number the docstring names. The probe printed −0.081 beside a docstring asserting +0.309 and
  nobody noticed, because nothing compared them. Same lesson as R23's `_diag_vclaire_source_nulls.py`:
  the diagnostic asked the right question and stopped one column short.
- **Second-order:** the GO statistic is also thin on power. `_stage7_feasibility.py:242` refuses a
  Mann-Whitney below n = 10 per group, and `DECISIONS.md:1712-1713` duly caveats "too few boulder-rich
  tiles (n=5, 46, 8) for stable Mann-Whitney on two of three images" — but `_build_14.py:331-342`'s
  `partial_corr` has no such guard, so the decisive `+0.16, p = 0.037` is a point-biserial partial
  correlation with **8 rich tiles of 173**. The project declared n = 8 too few for a rank test and then
  rested the GO on the same 8 tiles.
- **Self-refutation attempted:**
  (i) *"rasterio applied the scale/offset, so the parquet is already I/F."* No. The stored means are
  47–883, i.e. squarely inside the 10-bit DN range, and `run_stage7c_features.py:190` applies the
  conversion to the output of the **same** `region_means` call. Settled by the project's own code.
  (ii) *"Maybe `SCALING_FACTOR`/`OFFSET` are per-band, so the single-value conversion is also wrong."*
  The LBL carries one pair per IMAGE object (all 3 bands). Even if per-band values existed, the probe
  applies **no** conversion at all, and the single-pair conversion is what Stage 7c ships and what
  produces physically sensible Mars ratios (IR/BG ≈ 1.8–2.5) where DN gives 4.8–6.4.
  (iii) *"Is this just `notebooks-2`?"* No. `notebooks-2` is about *where* `partial_corr` lives, its
  phantom `dust_summary.parquet` declaration, and Pearson-vs-Spearman **mislabelling**. It explicitly
  says "medium on how much the number would move". This finding says the number moves from
  **p = 0.037 to p = 0.81** for a reason orthogonal to Pearson-vs-Spearman, and it flips two Test-A
  signs that `notebooks-2` never touched.
  (iv) *"Does the record already know?"* The opposite, in a way that makes it worse:
  `docs/compositional.md:177-182` states the DN gotcha explicitly and says "This bug was caught during
  the Stage 7c trio sanity-run and **fixed before the cohort run**", and `DECISIONS.md:1843-1846` frames
  the conversion as "mandatory for **Stage 7d cross-image pooling**". Both framings treat it as a
  cross-image pooling issue only, so §3.1 — written *before* the cohort run — was never revisited, and
  `docs/compositional.md:213-221` still describes Test A/B as extracting "per-band mean **I/F**".
  (v) *"Does it reach the shipped map or the Stage-7d headline?"* No. `run_stage7c_features.py` converts
  correctly, so `dataset_v2/features_colour.parquet` and everything Stage 7d/7e computes on it are
  sound; the CTX rock-abundance pipeline never touches colour. That is why this is `high` and not
  `blocker`. The blast radius is `DECISIONS.md` 2026-05-31, `docs/compositional.md` §3.1, notebook 14,
  and `cache_v2/stage7/*`.
- **Fix:** (1) In `_stage7_feasibility.py`, convert immediately after `region_means`
  (`v * lbl.scaling_factor + lbl.offset`, exactly as `run_stage7c_features.py:190-192`) — or better,
  add the conversion to `src.colour.region_means` behind a `lbl=` argument so no caller can forget it,
  with a test. (2) Re-emit `test_a_summary.parquet` / `test_b_summary.parquet` / `dust_summary.parquet`
  from the **cached raw parquets** (no JP2 re-read needed — the DN band means are already stored, the
  correction is a two-line affine map, and the whole recomputation takes seconds). (3) Correct
  `DECISIONS.md:1654-1684` — the verdict table, the "opposite-direction / different source and
  transport histories" bullet, the "Supports H_local" bullet, and the "(dust-controlled)" qualifier on
  the final verdict — and `docs/compositional.md:224-229`. (4) Strike or requalify the
  "Test A … Lambertian-invariant" claim at `_stage7_feasibility.py:277-280` and
  `docs/compositional.md:184-191`. (5) Give `_verify_stage7c_trio.py` the merge and the assertion its
  docstring promises, or delete it — a cross-check with no comparison is worse than none.

---

### probes-stage7-2 — The published Tier-1 provenance result (**Fisher's exact OR = 23.0, p = 0.018**) is the single significant cell of **twelve** analysis choices the two terrain probes compute, it is **not** the doc's own declared headline partition (which gives p = 0.059), and both its inputs are untracked — one of them lives in `C:/Users/brian/Downloads/`

- **Severity:** medium
- **Liveness:** dead-closed programme, but the number is the **only positive empirical evidence** in
  `docs/compositional.md`'s answer to the instructor's Q3 (transported vs locally-sourced provenance),
  restated three times including in the executive summary
- **Confidence:** high (all four published p-values reproduce to 4 dp; the other eight cells recomputed
  from the same artifacts)
- **Where:** `scripts/probes/_terrain_stats.py:20` (`for rule in ("P4_area", "P2_count")`), `:52-58`
  (the `is_comp_resid` Fisher call), `scripts/probes/_terrain_stats_honest.py:19`, `:26-47` (the
  impute-vs-exclude fork), `scripts/probes/_terrain_classify.py:38` (`"deposit_flag": r"\bdeposit\!"`),
  `:89-91` (writes the exposure table), `:107` (hardcodes the T=0.10 attribution file); consumers
  `docs/compositional.md:553-575` (the table), `:618-620`, `:870-877`, `:948-955`; `DECISIONS.md:2338`

`_terrain_stats.py` runs the Fisher test over both `partition_rule`s; `_terrain_stats_honest.py` runs
it again under both missing-data policies; and three shadow-threshold attribution tables exist on disk
(`stage7d_attribution_shadow_{0.05,0.10,0.20}.parquet`). That is 3 × 2 × 2 = **12** analysis cells for
one hypothesis. Recomputed:

| T | partition | missing data | n | OR | p |
|---|---|---|---|---:|---:|
| 0.05 | P2_count | impute | 30 | 1.00 | 1.000 |
| 0.05 | P2_count | honest | 28 | 1.27 | 1.000 |
| 0.05 | P4_area | impute | 24 | 8.50 | 0.143 |
| 0.05 | P4_area | honest | 22 | inf | 0.065 |
| 0.10 | P2_count | impute | 32 | 12.0 | 0.034 |
| **0.10** | **P2_count** | **honest** | **30** | **23.0** | **0.018** ← the published number |
| 0.10 | P4_area | impute | 26 | 6.38 | 0.101 |
| 0.10 | P4_area | honest | 24 | 12.0 | 0.059 ← the doc's own declared headline |
| 0.20 | P2_count | impute | 33 | 2.88 | 0.295 |
| 0.20 | P2_count | honest | 31 | 3.67 | 0.241 |
| 0.20 | P4_area | impute | 27 | 6.75 | 0.091 |
| 0.20 | P4_area | honest | 25 | 12.8 | 0.053 |

- **Failure scenario:** `docs/compositional.md:845` fixes the headline as "**T=0.10 / P4_area**" and
  every per-image attribution statement, the attribution figure and §5's counts use it. The Tier-1
  test at that headline is **p = 0.059 — not significant**. But `:618-620`, `:874-877` and `:952-955`
  all quote the **P2_count** cell — "Fisher's exact OR = 23.0, p = 0.018 … deposit-flagged /
  streamlined-shapes images are an order of magnitude enriched in `composition_residual`" — without
  saying it is off-headline, and the executive summary presents it as one of the two legs that
  "**disfavour locally-sourced-from-crater-ejecta and favour transported-with-deposit-character**"
  (the other leg being a null). The doc shows only the T = 0.10 row of the table above; it never
  mentions that the same test at the *stricter* shadow threshold the doc elsewhere treats as the
  robustness check (T = 0.05) gives **p = 1.000, OR = 1.27**, i.e. the association vanishes entirely.
  `docs/compositional.md:494` tells the reader "the composition narrative **strengthens** under shadow
  control"; for the Tier-1 test the opposite is true.
- **Evidence:**
  ```python
  # scripts/probes/_terrain_stats.py:20  -- fork 1: both partition rules, both reported
  for rule in ("P4_area", "P2_count"):
  # :52-58
  sub["is_comp"] = (sub["attribution"] == "composition_residual")
  ct2 = pd.crosstab(sub["transport_indicator"], sub["is_comp"], margins=False)
  odds, p = stats.fisher_exact(ct2.values, alternative="two-sided")

  # scripts/probes/_terrain_stats_honest.py:26-47 -- fork 2: impute vs exclude, both printed
  print("\n-- A. Original approach: impute missing terrain as transport_indicator=False --")
  print("\n-- B. Honest approach: exclude images missing terrain annotations --")

  # scripts/probes/_terrain_classify.py:107 -- fork 3 is silent: only T=0.10 is ever read
  attr = pd.read_parquet(ROOT / "dataset_v2" / "stage7d_attribution_shadow_0.10.parquet")
  ```
  Reproduction of the four published values from
  `dataset_v2/{terrain_classification_v2,stage7d_attribution_shadow_0.10}.parquet`: honest P2 OR 23.0
  p = 0.01806 (doc: 23.0 / 0.018); honest P4 OR 12.0 p = 0.05929 (doc: 12.0 / 0.059); imputed P2
  p = 0.03424 (doc: "earlier drafts … P2 p = 0.034"); imputed P4 p = 0.1014 (doc: "P4 p = 0.10"). The
  probes are unambiguously the producers. Cell counts also match: honest P2 = 3/6 exposed vs 1/24
  unexposed.
- **Two aggravating factors:**
  1. **The exposure variable is a literal-string parse with a defensible alternative that weakens the
     result.** `_terrain_classify.py:38` sets `deposit_flag` from `r"\bdeposit\!"` — the exclamation
     mark is required. `ESP_052576_2250`'s note is *"lots of good boulders. **deposit** on left side."*
     — a depositional annotation that the doc's own stated semantics ("a geological annotation for
     depositional features") covers, but the regex misses. Adding it (7 exposed instead of 6; it is
     `no_signal` at all three thresholds) moves the headline cell to **OR = 16.5, p = 0.031** and the
     P4 headline to p = 0.091. Still significant, but it shows one free character in a regex is worth
     a factor of 1.7 in p on a 2×2 table with 6 exposed images and 4 outcomes.
  2. **Neither input is committed.** `dataset_v2/` is gitignored in full (`.gitignore:19`), so
     `terrain_classification_v2.parquet` and `stage7d_attribution_shadow_0.10.parquet` are not in the
     repo, and the *source* of the exposure variable is
     `C:/Users/brian/Downloads/Mapping_Images_33_36.xlsx` (`_terrain_classify.py:25`) — outside the
     repository entirely. A reader of `docs/compositional.md` §4.7 cannot reproduce OR = 23.0 from
     anything in the project. Compare **R12**.
- **Self-refutation attempted:**
  (i) *"The doc discloses the P4 row, so the reader can see both."* It prints both T = 0.10 rows in the
  §4.7 table and calls P4 "marginal at 0.059" — but the three *summary* restatements (`:620`, `:874`,
  `:952`), which are what a reader of the abstract/conclusions sees, quote only P2, and none of them
  flags that P4 is the declared headline everywhere else in the document. It never discloses that
  T = 0.05 and T = 0.20 were computed at all.
  (ii) *"Is the 'honest exclusion' itself the defect?"* No — excluding two unannotated images is
  defensible and the doc argues it well. I checked the direction: exclusion raises significance in 5 of
  6 cells, so it is a systematic, not a cherry-picked, choice. The defect is the *unreported* T and
  partition forks, not the missing-data policy.
  (iii) *"Is this already `stats-fallacies-4`?"* No. That finding is about spatial autocorrelation in
  Stage-7d's **pooled per-tile** tests. This is an **image-level** 2×2 Fisher test with n ≈ 30 and a
  multiplicity/selective-reporting problem; grepping `docs/review_2026-07-31/*.md` for `Fisher`,
  `OR = 23`, `terrain` returns nothing.
  (iv) *"Is the conclusion wrong?"* Not necessarily — the direction is positive in 11 of 12 cells. The
  defect is that the stated evidence (a single p = 0.018 presented as the Tier-1 result) is much
  stronger than what the grid supports, on a claim about *provenance* that the doc calls the
  instructor's extra goal.
- **Fix:** report the full 12-cell grid (or at minimum the 3 shadow thresholds × 2 partitions at the
  honest policy) in `docs/compositional.md` §4.7; quote the **declared headline** (T = 0.10 / P4_area,
  OR = 12.0, p = 0.059) as *the* Tier-1 result and P2 as a secondary; downgrade the Q3 summary at
  `:618-620` / `:874-877` / `:952-955` from "we see it at p = 0.018" to "direction positive across the
  grid, significant in 1 of 12 pre-existing analysis cells"; add the `deposit`-without-`!` case to the
  regex (or state the literal-`Deposit!` rule as a pre-registered definition and show the sensitivity);
  and commit `terrain_classification_v2.parquet` (39 rows, kilobytes) plus a CSV export of the
  spreadsheet columns actually used, so the test has provenance inside the repo.

---

### probes-stage7-3 — The Stage-7.0 verdict table compares three images whose surviving boulder populations differ **4.4× in median area**, because the 0.25/0.50 m/px cohort split (R03) plus a fixed 8-pixel floor cut each image at a different physical size

- **Severity:** low
- **Liveness:** dead-closed; affects the same `DECISIONS.md:1654-1684` cross-image reading as finding 1
- **Confidence:** high on the measurement, medium on how much of it is the probe's own doing
- **Where:** `scripts/probes/_stage7_feasibility.py:68-69` (`MIN_POLYGON_PIXELS = 8`,
  `MIN_RING_PIXELS = 16`), `:59-63` (the trio); `src/colour.py:174` (`min_pixels` applied to a
  DN-pixel count, not an area); consumers `DECISIONS.md:1676-1684` and `:1708-1711`

The trio mixes map scales — `ESP_042964_2160` and `ESP_055253_2245` are 0.25 m/px, `ESP_054000_2255` is
0.50 m/px (`cache_v2/hirise_color/*_COLOR.LBL`, `MAP_SCALE`). The retention floor is expressed in
**pixels**, so it is 0.5 m² for the two fine images and 2.0 m² for the coarse one; on top of that
BoulderNet's own detection floor scales with pixel size (**R03**). Measured on
`cache_v2/stage7/test_a_per_polygon.parquet` (the surviving polygons):

| ObsId | m/px | n kept | min area m² | median area m² | median ring px |
|---|---|---:|---:|---:|---:|
| ESP_042964_2160 | 0.25 | 259 | 1.44 | 3.59 | 5720 |
| ESP_054000_2255 | **0.50** | 354 | **4.21** | **8.83** | 1565 |
| ESP_055253_2245 | 0.25 | 362 | 0.88 | 2.02 | 5489 |

- **Failure scenario:** `DECISIONS.md:1676-1684` reads the three images against each other — "the two
  boulder populations have *opposite-direction* compositional shifts … suggests different source /
  transport histories" and "ESP_054000_2255 … boulders there are spectrally indistinguishable from
  surroundings … Supports H_local for that image". A 4.4× difference in median boulder area between the
  compared populations is an alternative explanation for any per-image difference in a
  boulder-vs-surroundings spectral contrast (bigger boulders → more resolved interior, less mixing with
  regolith, different shadow fraction), and it is nowhere in the caveat list at `:1706-1719`. Combined
  with finding 1 (which flips two of the three signs), nothing in the cross-image reading of that table
  is safe.
- **Self-refutation attempted:** (i) *Is the 8-px floor actually what binds?* Mostly not — the observed
  minima (23 px, 17 px, 14 px) are all above 8, so the size cut comes principally from BoulderNet's
  pixel-scale-dependent detection floor, i.e. **R03**, which is already filed. I am filing this as a
  *new consequence* of R03 in a place R03's entry does not reach (the Stage-7 record), and the probe's
  own pixel-denominated floor is a second, image-dependent cut layered on it. (ii) *Does this change any
  Stage-7c/7d number?* No — Stage 7c is per-tile, not per-polygon, so the 36-image cohort analysis is
  untouched. (iii) *Was the trio's scale mix known?* `DECISIONS.md:1630` records "Resolution typically
  0.25 m/px but can be 0.5 m/px (verified for `ESP_054000_2255`)" — the fact is in the record; its
  consequence for the cross-image comparison is not.
- **Fix:** state the floor in metres, not pixels (`min_area_m2` converted per image via
  `lbl.map_scale_mpp`), and restrict any cross-image Test-A comparison to a common size band; add the
  scale mix to the caveat list at `DECISIONS.md:1706-1719`.

---

### probes-stage7-4 — "Cohort I/F medians IR = 0.169 / RED = 0.165 / BG = 0.077, all inside the expected 0.05–0.30 range for Mars" are **Lambert albedos, not I/F**; the true I/F medians are 0.086 / 0.087 / 0.048

- **Severity:** low
- **Liveness:** live document (`docs/compositional.md` is reader-facing) on a dead-closed programme
- **Confidence:** high (reproduced exactly)
- **Where:** `scripts/probes/_summarise_stage7c.py:31-32` (`cohort I/F medians: IR={fc.IR_iof.median()}`),
  `scripts/run_stage7c_features.py:193-198` (the column named `IR_iof` is written **after**
  `/ cos_i`); consumers `DECISIONS.md:1867-1868`, `docs/compositional.md:274-278`

Stage 7c computes `ir_iof = DN·S + O` and then overwrites it with `ir = ir_iof / cos_i` before emitting
it under the key `IR_iof`. `_summarise_stage7c.py` medians that column and prints it as "cohort I/F
medians"; the record then checks the value against a literature **I/F** range. Recomputed from
`dataset_v2/features_colour.parquet` (9,860 rows / 36 images — matching the doc):
post-Lambert medians 0.1691 / 0.1648 / 0.0771 (exactly the published triple), true I/F medians
`IR_iof × cos_incidence` = **0.0862 / 0.0869 / 0.0477**, cos(i) 0.302–0.763.

- **Failure scenario:** the sanity check that would catch a units error in the colour pipeline is itself
  computed on the wrong units — the same class of error as finding 1, in the one place the project
  validates its colour values against physical expectation. A reader comparing 0.169 against published
  HiRISE I/F for dusty equatorial regolith gets a ~2× discrepancy, and the "all inside the expected
  0.05–0.30 range" reassurance is about a quantity (Lambert albedo) with a different expected range.
  Nothing downstream breaks — cos(i) cancels in every ratio, and the Stage-7d standardised tests are
  per-image standardised — so this is purely a labelling defect.
- **Self-refutation attempted:** (i) *Maybe `IR_iof` really is pre-Lambert and the division happens
  later.* No — `run_stage7c_features.py:195-198` divides and then writes `"IR_iof": float(ir)`, and
  `docs/compositional.md:264-270` lists steps 4 (convert) and 5 (Lambert) before the emit. (ii) *Does
  the sanity check fail under the correct units?* No — 0.086/0.087/0.048 is still broadly plausible
  (BG dips just under 0.05). The claim is not falsified, it is unsupported as stated. That is why this
  is `low`.
- **Fix:** rename the emitted columns `IR_lambert` / `RED_lambert` / `BG_lambert` (or keep `_iof` and
  stop dividing there), update `dataset/DATA_DICTIONARY.md` if it lists them, and change
  `_summarise_stage7c.py:31-32` to print both the raw I/F and the Lambert-corrected medians; correct
  `DECISIONS.md:1867` and `docs/compositional.md:274-278`.

---

## Refuted by my own check

- **`_stage7_feasibility.py` violates the per-image local-radius CRS invariant** (`_load_polygons`
  uses `set_crs(..., allow_override=True)`, an assign not a reproject). It does not: the override is
  the documented SP1 correction (`src/colour.py:73-87`, invariant 3), the COLOR.JP2's own pixel
  transform is under the same corrected CRS, and Test B reprojects properly with
  `pyproj.Transformer.from_crs(CTX_CRS, corrected_crs)` (`:180`). `_stage7a_sanity.py` even exists to
  assert every colour-covered ObsId has the Stage-1 sidecar and `sys.exit(1)`s if not.
- **The Test-A "ring" contains other boulders, inflating the contrast.** It dilutes it, not inflates —
  a 2–10 m annulus around a ~6 m² polygon is ~370 m² (median 4,155–5,720 valid pixels) and will contain
  neighbouring detections in a boulder-rich field, biasing interior−ring **toward zero**. Conservative,
  and the direction is wrong for a false positive. Worth one sentence in the caveats, not a finding.
- **`_compositional_slim_polygons_overlay.py` (Figure 1 of the published `compositional_slimmer.pdf`)
  has the units bug too.** It does not — `:81` applies `* lbl.scaling_factor + lbl.offset`, and `:76-79`
  explicitly reasons about the offset shifting true-zero nodata upward, computing the valid mask on raw
  DN first. It also reprojects polygons properly (`polys.to_crs(crs)`, `:135`). Cleanest probe in the
  area.
- **`_compositional_slimmer_attribution_bars.py` fabricates the "0 inconclusive" claim.** It renders the
  `inconclusive` category and skips it when the count is zero (`:59-67`), with a comment stating the
  category "is currently always 0". That is **R15** (structurally unreachable), already filed against
  `src/stage7d_pooled.py:462-468` and `notebooks/_build_16.py`; this probe is a third site of the same
  claim, noted here only so a future session sees the full surface.
- **`_fetch_color.py` fetches over the network without the SSL fix.** It has it —
  `import truststore; truststore.inject_into_ssl()` at module top, with the memory note cited. Downloads
  atomically via `.partial` + `replace`. Clean.
- **The Tier-1 `has_signal` test (`_terrain_stats.py:28-49`) is another undisclosed fork.** It is
  computed and printed, but no doc quotes it — `docs/compositional.md` reports only the
  `is_comp_resid` version. Not load-bearing; excluded from the 12-cell count for that reason.
- **`_terrain_classify.py`'s `dominant_terrain` priority order silently drives the published crosstab.**
  The crosstabs it prints appear in notebook 17 as context, not as a tested statistic; no p-value or
  effect size depends on the priority order. Low, uncited.

## Verified clean

- **`_summarise_stage7c.py`'s cohort numbers reproduce exactly** from
  `dataset_v2/features_colour.parquet`: 9,860 rows / 36 images; dust_index p5/p50/p95 =
  1.642/1.947/2.353 (doc: 1.64/1.95/2.35); cos(i) 0.302–0.763 (doc: 0.30–0.76). Only the *label* on the
  band medians is wrong (finding 4).
- **`_terrain_stats.py` / `_terrain_stats_honest.py` arithmetic** — all four published Fisher values
  (OR 23.0/p 0.018, OR 12.0/p 0.059, imputed p 0.034 and p 0.10) reproduce to 4 dp, and the published
  cell counts (3/6 vs 1/24) match. The two probes are a deliberate pair: `_honest` supersedes
  `_terrain_stats`'s missing-data handling and is the one whose numbers the doc publishes; the doc
  states the supersession at `:571-575`. Nothing is wrong with either computation *per se* — finding 2
  is about which of their outputs was selected for the record.
- **`_terrain_classify.py`'s keyword parse** — checked every note against a plain `deposit` / `stream`
  substring test: exactly **one** disagreement (`ESP_052576_2250`, discussed in finding 2). The other
  seven flagged images all carry the literal `Deposit!` / `Streamlined`.
- **`region_means`'s nodata handling** — `valid = mask & (arr[b] > 1e-9)` for all three bands drops the
  COLOR.JP2's `CORE_NULL = 0` pad correctly, and the `min_pixels` floor is applied after masking, so no
  region is summarised from pad.
- **Test A's ratio-of-means convention** — both the probe and my recomputation take
  `mean(band_a) / mean(band_b)` per region, so a single saturated pixel (`CORE_LOW_REPR_SATURATION = 1`)
  cannot blow up a per-polygon ratio.
- **Cohen's d and Wilcoxon are invariant to the DN→I/F affine map for *single bands*** — only ratios
  change. This is why all the absolute-band rows of the verdict table are correct as published and only
  the ratio rows move in finding 1.
- **`_stage7a_sanity.py`** — genuinely asserts (exit 1 on any missing Stage-1 sidecar), which is more
  than most probes here do.
- **`_dump_attribution.py`, `_dump_browse_terrain.py`, `_dump_terrain_excel.py`, `_terrain_join_v2.py`,
  `_stage7_check_labels.py`, `_stage7_inspect.py`, `_stage7_verdict.py`,
  `_inspect_terrain_for_evidence.py`** — read-and-print only; no statistic, no artifact, nothing quoted
  as a number in any doc. (`_stage7_inspect.py:58-59` is notable only because it *does* apply the
  DN→I/F conversion, in the same session as finding 1's probe.)

## Coverage note

**Read in full:** `_stage7_feasibility.py` (405), `_verify_stage7c_trio.py`, `_summarise_stage7c.py`,
`_terrain_classify.py`, `_terrain_stats.py`, `_terrain_stats_honest.py`,
`_compositional_slim_polygons_overlay.py`, `_compositional_slimmer_attribution_bars.py`,
`_fetch_color.py`, `_stage7_inspect.py`, `_stage7a_sanity.py`, `_stage7_check_labels.py`,
`_stage7_verdict.py`, `_dump_attribution.py`, `_dump_browse_terrain.py`, `_dump_terrain_excel.py`,
`_terrain_join_v2.py`, `_inspect_terrain_for_evidence.py` — i.e. all 18 files in the area. Also read in
full: `src/colour.py` (268), `scripts/run_stage7c_features.py:175-205`, `notebooks/_build_14.py:320-400`,
`docs/compositional.md` §§2.3, 3.1, 3.3, 4.7, 6.1-6.2, and `DECISIONS.md:1624-1720`, `:1840-1915`,
`:2160-2340`.

**Reproduced numerically** (pandas/scipy over on-disk artifacts; no probe executed, no imagery, no
network): the full Test-A summary table in both DN and I/F from
`cache_v2/stage7/test_a_per_polygon.parquet` + the three `cache_v2/hirise_color/*_COLOR.LBL`; the Test-B
point-biserial / partial-correlation / dust-discriminator table in both units from
`cache_v2/stage7/test_b_per_tile.parquet` (matching the banked `dust_summary.parquet` exactly in the DN
column); the 542-tile merge of `dataset_v2/features_colour_trio.parquet` against
`cache_v2/stage7/test_b_per_tile.parquet`; the 12-cell Tier-1 Fisher grid plus the exposure-definition
sensitivity from `dataset_v2/terrain_classification_v2.parquet` and the three
`stage7d_attribution_shadow_*.parquet`; and the Stage-7c cohort summary from
`dataset_v2/features_colour.parquet`.

**Not checked, and why:**
- **`src/stage7d_pooled.py` and the Stage-7d headline results** — out of area (`stats-fallacies-4` and
  R15 cover them). I verified only that Stage 7c feeds it correctly-converted I/F, which bounds
  finding 1.
- **Whether re-running the Stage-7.0 verdict in I/F changes the *pass criterion*** beyond the three
  conclusions listed — criterion (a) plainly still passes, but I did not re-derive the full
  PLAN_Compositional §3.1 decision tree.
- **`reports/figures/stage7_tier1_terrain_attribution.png` / `stage7_tier2_crater_distance.png` /
  `compositional_slimmer_attribution_bars.png`** — I did not open the PNGs to confirm which partition
  each renders; `_compositional_slimmer_attribution_bars.py:56` hardcodes `P4_area`, and
  `_terrain_classify.py`'s crosstab is `P4_area` too, so the Tier-1 *figure*'s producer is not in this
  area's file list and I could not confirm whether it plots P2 (the quoted result) or P4 (the headline).
  Worth one grep by whoever fixes finding 2.
- **Tier 2 (`_crater_distance.py`)** — assigned to `probes-w1-geospatial`, not this area. Its Kruskal-
  Wallis null is the other leg of the Q3 argument and deserves the same forking-paths check
  (4 diameter thresholds × 2 partitions × center-vs-rim distance = 16 cells, all reported as "null").
- **The 2 unannotated ObsIds' browse images** — `ESP_017355_2260` is both the image the "honest
  exclusion" removes *and* one of **R23**'s two score-truncated label images; whether its
  `composition_residual` attribution is itself an artifact of the truncated label basis is a real
  question I could not answer without re-running Stage 7d.
- Per the rules of engagement I executed no probe and touched no imagery or network. Note that
  `cache_v2/` and `dataset_v2/` are **gitignored**, so every artifact I read is local-only: a verifier
  on a fresh clone cannot reproduce findings 1–4 without re-fetching 9.1 GB of COLOR.JP2s and re-running
  Stages 7.0/7a/7c/7d. That is itself worth recording (compare **R12**).

## Load-bearing map

| probe | cited by | number it produced | verdict |
|---|---|---|---|
| `_stage7_feasibility.py` | `DECISIONS.md:1654` (verdict table), `:1673` (final verdict), `:1692`, `:1809`; `docs/compositional.md:213-229` (§3.1); `notebooks/_build_14.py:3,59`; `cache_v2/stage7/*.parquet` | the whole Stage-7.0 trio table: Test A `d`/`p` per band + ratio, Test B `d`/`p`, and (via notebook 14) the GO statistic `partial r = +0.16, p = 0.037` | **WRONG in every ratio row — finding 1** (DN, not I/F: two sign flips, one null→significant, one significant→null, GO statistic → p = 0.81) |
| `_verify_stage7c_trio.py` | `DECISIONS.md:1902` ("cross-checks the trio output's within-image direction-of-effect against the Stage 7.0 verdict table") | none — prints Stage-7c d values only | **a cross-check with no comparison and no assertion**; it printed the contradicting value (−0.081 vs the recorded +0.309) without noticing — finding 1 |
| `_terrain_stats_honest.py` | (unattributed) `docs/compositional.md:553-575`, `:618-620`, `:874-877`, `:952-955` | **Fisher's exact OR = 23.0, p = 0.018** (P2_count) and OR = 12.0, p = 0.059 (P4_area) | arithmetic **REPRODUCED exactly**; the *selection* is 1 significant cell of 12 and off the declared headline — finding 2 |
| `_terrain_stats.py` | `DECISIONS.md:2338` ("Fisher's exact (Tier 1)") | the superseded imputed-missing numbers P2 p = 0.034 / P4 p = 0.10, plus an uncited `has_signal` variant | **REPRODUCED**; superseded by `_terrain_stats_honest.py`, and the doc says so (`:571-575`) |
| `_terrain_classify.py` | `DECISIONS.md:2218`, `:2337`; `docs/compositional.md:541`; `notebooks/_build_17.py:11,124`; writes `dataset_v2/terrain_classification_v2.parquet` | the Tier-1 **exposure variable** (`deposit_flag`, `streamlined_flag`, `terrain_category`) for all 39 ObsIds | parse is faithful in 38 of 39 notes; the 1 miss (`deposit` without `!`) moves the headline p from 0.018 to 0.031 — finding 2. Source spreadsheet lives outside the repo |
| `_summarise_stage7c.py` | `DECISIONS.md:1854-1875`, `:1904`; `docs/compositional.md:271-278` | 9,860 rows / 36 images; retention 24–31 %; "cohort I/F medians 0.169/0.165/0.077"; dust_index p5/p50/p95 1.64/1.95/2.35; cos(i) 0.30–0.76 | **REPRODUCED exactly**; the band medians are Lambert albedos mislabelled as I/F — finding 4 |
| `_fetch_color.py` | `DECISIONS.md:1690`; `notebooks/_build_14.py:75` | the trio's COLOR.JP2/LBL cache (~842 MB) | clean (truststore, atomic writes, `[cached]` short-circuit) |
| `_compositional_slim_polygons_overlay.py` | `docs/compositional_slimmer.md:345` ("Figure 1 builder"); writes `reports/figures/compositional_slim_polygons_on_color.png` | the published IRB false-colour + polygon-centroid figure | **clean** — applies the DN→I/F conversion and reasons explicitly about the offset-vs-nodata interaction |
| `_compositional_slimmer_attribution_bars.py` | `docs/compositional_slimmer.md:346` ("Figure 2 builder"); writes `reports/figures/compositional_slimmer_attribution_bars.png` | per-image attribution counts across 4 shadow thresholds (P4_area) | counts read straight from the attribution parquets — sound; renders the unreachable `inconclusive` category (**R15**, third site) |
| `_dump_attribution.py` | `DECISIONS.md:2171` | the per-image attribution table transcribed into the writeup (T=0.10 / P4_area) | pure dump of `stage7d_attribution_shadow_0.10.parquet`; no derived statistic |
| `_terrain_join_v2.py` | `DECISIONS.md:2336` | "37 of 39 v2 ObsIds in the spreadsheet" (the coverage claim at `docs/compositional.md:537`) | reproduces (`in_spreadsheet` sums to 37 in the parquet) |
| `_dump_browse_terrain.py` / `_dump_terrain_excel.py` | `DECISIONS.md:2334-2335` | manifest / spreadsheet dumps used to build the terrain notes | no derived number |
| `_stage7a_sanity.py` | — | Stage-1 sidecar presence for all 37 colour-covered ObsIds | uncited but the only probe here that **asserts** (exit 1); clean |
| `_stage7_inspect.py`, `_stage7_check_labels.py`, `_stage7_verdict.py`, `_inspect_terrain_for_evidence.py` | — | smoke/inspection prints only | not cited, write nothing |
</content>
</invoke>
