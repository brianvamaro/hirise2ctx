# Review area: probes-fbuild

- **Reviewed at commit:** `da884c7`
- **Date:** 2026-08-03
- **Verification:** self-refuted (single-agent pass; not independently verified). Every number below
  was re-derived read-only from committed artifacts — `reports/f_leg_b/{diag_uint8_contrast,
  diag_per_image,variant_summary,mapping_compare_per_image,review_overlap_residual,frame_incidence}.csv`,
  `reports/figures/f_pilot_overlap_pairs.csv`, `reports/f_timing/frame_list.csv` — with small
  pandas/scipy snippets. **No probe was run**; nothing that touches CTX/HiRISE imagery, ISIS or the
  network was executed.

**Headline.** All 20 files read in full (1 242 lines). The probes' *arithmetic* is almost entirely
right: I reproduced the 10.2 % → 4.0 % overlap headline, the 20.4 % prediction disagreement, the
18/18 win-loss split, the +0.0067 Δ-median PASS, the "mosaic windows vary 19–57" range and the F02
z-score to the digits quoted in `DECISIONS.md`. What does not survive is **two joins and two
interpretations**:

1. the 2026-07-04b **mechanism** claim ("dim = high-incidence → illumination → next iteration =
   minnaert") is contradicted by the incidence table sitting in the same directory — actual
   incidence is null (ρ = +0.06, p = 0.74) and partialling it out *strengthens* the brightness
   signal, i.e. the live axis is the part of brightness `cos^k(i)` cannot touch;
2. the 2026-07-05d amended verdict's "ρ = −0.33 anti-correlation with Δincidence" is an artifact of
   a **truncated 3-character join key** that collides two frames; fixed, it is **ρ = −0.03**;
3. the ESP_053989 diagnosis in the record is arithmetically wrong and refuted by the probe's own
   committed table;
4. the "post-minnaert 4.0 %" is algebraically a per-pair *constant* rescale, so it measures only what
   a per-frame DC gain removes — and H5 ("stronger physics") was downgraded on it.

None of this overturns the ABORT (which rests on the Stage-C/D level table and gates 5/6, not on
leg B), and the shipped mosaic map does not depend on any of it.

---

## Findings

### probes-fbuild-1 — Leg B's "surviving correlate = illumination, exactly A0's cos-i axis" is a brightness/incidence confound: the ΔAUC signal is the part of scene brightness that `cos^k(i)` cannot touch, and it was harvested from six untested-for-multiplicity correlations

- **Severity:** high (record correctness; it selected the next experiment and is in a committed, executed notebook + two committed figures)
- **Liveness:** dead-closed programme, but the claim is stated as mechanism in the committed
  `notebooks/27_f_pilot_legb.ipynb` §3, `reports/figures/f_leg_b_diag_scatter.png` and
  `reports/figures/27_f_legb_median_scatter.png`
- **Confidence:** high for the statistics (all from committed CSVs); medium for how much the mapping
  choice actually cost
- **Where:** [scripts/probes/_f_leg_b_diag.py:47-58](../../scripts/probes/_f_leg_b_diag.py#L47-L58)
  (`if_median` / `if_iqr`), [:80-83](../../scripts/probes/_f_leg_b_diag.py#L80-L83) (the six-way
  correlation loop), [scripts/probes/_f_leg_b_figures.py:67-71](../../scripts/probes/_f_leg_b_figures.py#L67-L71)
  (the figure axis + title); consumers `DECISIONS.md:5654-5656`, `notebooks/_build_27.py:156-159`,
  `:177-179`

`_f_leg_b_diag.py` prints Spearman ΔAUC against five candidates, `_f_leg_b_uint8_contrast.py` adds a
sixth, and the record keeps the largest one:

> `DECISIONS.md:5654-5656` — "**The surviving correlate is the composite I/F median: ρ = +0.35 — DIM
> (high-incidence) scenes collapse, bright scenes improve** — illumination again, exactly A0's cos-i
> axis."
> `notebooks/_build_27.py:156-157` — "**Survivor: the composite I/F median (ρ = +0.35) …** That is an
> illumination axis: leg A0 measured per-frame I/F median ↔ cos(incidence) r = +0.83."

Three defects, in increasing order of consequence.

**(a) `if_median` is not the composite median.** `_f_leg_b_diag.py:42-50` reads *every* crop of an
obs, appends `arr[fin]` to a list and takes `np.median(np.concatenate(vals))`. The F window is a
**last-write-wins** composite, so a pixel covered by two crops contributes **twice** to `if_median`
and only once to what the embedder saw. The sibling probe's own docstring flags exactly this defect —
[`_f_leg_b_uint8_contrast.py:8`](../../scripts/probes/_f_leg_b_uint8_contrast.py#L8): "*the earlier
diag `if_iqr` concatenated crops, double-counting overlaps*" — fixes it for the **IQR** (`f_iqr`) and
never re-computes the **median** on the real composite. `overlap` (fraction of the window written by
2+ crops) runs from 0.00 to 0.9993 across the 36 images, so the weighting error is not uniform.
`DECISIONS.md:5654` and the notebook both call it "the composite I/F median".

**(b) The interpretation was never tested against incidence, although a sibling probe joins the
incidence table.** `_f_leg_b_mapping_compare.py:51-56` joins `frame_incidence.csv` (PDS truth) onto
`obs_frame_map.csv` and banks `inc_min`/`inc_max` per obs in
`reports/f_leg_b/mapping_compare_per_image.csv`. Merging that with `diag_uint8_contrast.csv` (36
images, both committed):

| statistic | ρ | p |
|---|---|---|
| ΔAUC vs `if_median` (the recorded survivor) | **+0.347** | 0.038 |
| ΔAUC vs `inc_min` | +0.065 | 0.705 |
| ΔAUC vs `inc_max` | +0.005 | 0.979 |
| ΔAUC vs `(inc_min+inc_max)/2` | +0.058 | 0.737 |
| ΔAUC vs `cos^0.58(inc_mean)` | −0.058 | 0.737 |
| `if_median` vs `inc_mean` | −0.502 | 0.0018 |
| **partial** ΔAUC vs `if_median` \| `inc_mean` | **+0.436** | 0.0078 |
| **partial** ΔAUC vs `inc_mean` \| `if_median` | +0.287 (wrong sign for the claim) | 0.090 |

So brightness does predict ΔAUC, and brightness *is* partly an incidence proxy (ρ = −0.50) — but
controlling for incidence **strengthens** the brightness relation and leaves incidence itself null
with the sign reversed. The live axis is therefore the **incidence-orthogonal** component of
brightness (intrinsic albedo / dust / atmospheric epoch), which is exactly the component a
`cos^k(i)` divisor cannot remove by construction. Rewriting "DIM (high-incidence)" as "DIM
(intrinsically dark)" flips the recommendation the entry makes.

**(c) ρ = +0.35 is the maximum of six correlations with no multiplicity correction.** p = 0.038 → a
Holm/Bonferroni-corrected p ≈ 0.23 over the six the same probe printed (`coverage`, `overlap`,
`n_crops`, `if_iqr`, `if_median`, `iqr_ratio`), and the record's own next-strongest is `if_iqr` at
+0.242. This is the R41 tolerance/no-uncertainty theme applied to a correlation instead of a gate.

- **Failure scenario:** a reader (or the next session) concludes from `DECISIONS.md:5654-5656`,
  notebook 27 §3 and the two committed figures that the leg-B collapse is an **illumination-geometry**
  failure curable by a photometric divisor, and prioritises `cos^k(i)` work. The programme's own
  sequel says otherwise: `cos^k(i)` over plain global bought **Δmedian AUC +0.0046** (global −0.0387 →
  minnaert p2–98 −0.0341, `DECISIONS.md:5593-5595`), while switching the I/F→uint8 *domain* to log
  bought **+0.030** (−0.0236 → +0.0067, `DECISIONS.md:5347-5349`), and `DECISIONS.md:5361` itself
  concludes "the I/F→uint8 *domain* (linear vs log) is" the lever. The correct reading of (b) predicts
  precisely that ordering; the recorded reading predicts the opposite.
- **Evidence:**
  ```
  scripts/probes/_f_leg_b_diag.py:47-57
      vals.append(arr[fin])                       # per crop, overlaps counted once each
      v = np.concatenate(vals) if vals else np.array([0.0])
      q75, q25 = np.percentile(v, [75, 25]) if v.size > 1 else (0, 0)
      return dict(..., if_iqr=float(q75 - q25), if_median=float(np.median(v)))

  scripts/probes/_f_leg_b_diag.py:80-83
      print("\nSpearman correlations with d_auc:")
      for c in ("coverage", "overlap", "n_crops", "if_iqr", "if_median"):
          rho = df["d_auc"].corr(df[c], method="spearman")

  scripts/probes/_f_leg_b_figures.py:69-70
      axes[1].set_title("DIM scenes collapse (ρ=+0.35) — illumination is the live correlate;\n"
                        "post-norm uint8 contrast is pinned at IQR≈27.7 for all (ratio ρ=+0.09, null)")
  ```
- **Self-refutation attempted:**
  (i) *Maybe `inc_min`/`inc_max` are the wrong incidence proxy* — a coverage-weighted mean incidence
  would sit between them, and **all three endpoints plus the `cos^k` transform are null** (+0.005 to
  +0.065). (ii) *Maybe the multi-frame composites muddy it* — on the **8 single-crop** images, where
  incidence is exact and unambiguous, ρ(ΔAUC, incidence) = **−0.503 (p = 0.204)**, the sign the record
  wants; that is the one piece of support for the recorded reading and I report it honestly, but n = 8,
  p = 0.20, `if_median` there is only +0.238, and on the **28 multi-crop** images the sign **reverses
  to +0.322 (p = 0.095)** — a sign-unstable, never-significant axis. (iii) *Maybe "illumination" was
  meant loosely as "level"* — no: both the DECISIONS entry and the notebook name "A0's cos-i axis" and
  "cos(incidence) r = +0.83" explicitly, and the entry's stated conclusion is to try the cos-i
  correction next. (iv) *Maybe A0's r = +0.83 licenses it* — A0 measured 7 frames **inside one Murray
  tile** (`DECISIONS.md:5672-5677`), i.e. the same ground, where brightness variation really is mostly
  geometry; transporting that to 36 obs spread across circum-Chryse is where albedo enters, and the
  measurement above is what that transport costs. (v) *Already filed?* — no: R48/R49/R26/R50 are the
  prevalence family in Stage 6 / targets; `_f_leg_b_diag.py` appears in no register entry; §5 has
  nothing on leg B's diagnostics.
- **Fix:** (1) re-state `DECISIONS.md:5654-5656` and `notebooks/_build_27.py:156-159` as
  "ΔAUC tracks *scene brightness*; the incidence component of brightness is **not** the driver
  (ρ = +0.06, n = 36) — the live axis is the residual albedo/atmosphere component, which a photometric
  divisor cannot address", and note that the log-domain result later confirmed this; (2) report the
  base statistic beside it (partial ρ controlling for incidence) — the same rule the register's
  Pattern B already adopts; (3) if the number is ever reused, recompute `if_median` from
  `fe.composite_crops` output rather than the concatenation, as `f_iqr` already does.

---

### probes-fbuild-2 — A truncated 3-character pair key collides two frames, so the amended verdict's "prediction disagreement anti-correlates with Δincidence (ρ = −0.33)" is really ρ = −0.03

- **Severity:** medium (record correctness on the entry that opened the H1–H6 docket)
- **Liveness:** dead-closed, but the entry is quoted in `ROADMAP.md:19` and
  `PLAN_StripingArtifact.md:191-192`, `:206-211`
- **Confidence:** high (reproduced exactly from two committed CSVs; the collision is visible in the
  banked file as two rows sharing a label)
- **Where:** [scripts/probes/_f_review_overlap_residual.py:67](../../scripts/probes/_f_review_overlap_residual.py#L67)
  (`pair = f"{pids[i][:3]}~{pids[j][:3]}"`),
  [:54](../../scripts/probes/_f_review_overlap_residual.py#L54) (`dict(zip(...))`),
  [:74](../../scripts/probes/_f_review_overlap_residual.py#L74),
  [:84-85](../../scripts/probes/_f_review_overlap_residual.py#L84-L85); artifact
  `reports/f_leg_b/review_overlap_residual.csv`; consumer `DECISIONS.md:5235-5236`

The E8_N44 pilot has **7** frames but only **6** distinct 3-character prefixes:
`reports/f_timing/frame_list.csv` shows both `P22_009549_2289_XN_48N351W` and
`P22_009694_2267_XN_46N350W`. Truncating to `pids[i][:3]` makes `P21~P22` and `B03~P22` each name
**two different frame pairs**, and `pred_by_pair = dict(zip(pred_pairs["pair"], ...))` silently keeps
the **last** value, so both members of a colliding group receive the same `pred_absdiff`. This is
visible in the banked artifact: rows 4/6 (`P21~P22`, n = 1 357 and 15 199) both carry 0.2050, and rows
12/14 (`B03~P22`, n = 5 646 and 6 325) both carry 0.0904. The upstream
`reports/figures/f_pilot_overlap_pairs.csv` has the two real values — `B03~P22` = **0.3941**
(n = 6 415) and **0.0904** (n = 5 714) — so one row is wrong by **4.4×**.

Re-joining by within-label n-rank (the only unambiguous mapping the artifacts allow: IF n
5 646/6 325 ↔ pred n 5 714/6 415):

| Spearman `pred_absdiff` vs | as banked | join fixed |
|---|---|---|
| **`d_inc`** | **−0.329** (p = 0.231) | **−0.032** (p = 0.909) |
| `d_orbit` | +0.200 | +0.068 |
| `raw_absratio` | −0.134 | −0.004 |
| `minn_absratio` | +0.408 | +0.304 |

`DECISIONS.md:5236` reports the banked value as a finding: "*prediction disagreement anti-correlates
with Δincidence (ρ = −0.33)*". One mis-joined row out of 15 produces the entire effect.

- **Failure scenario:** the amended verdict argues that the residual is *not* photometric because
  prediction disagreement fails to grow with Δincidence. The stated evidence is a **negative** ρ, i.e.
  an active structure (worse agreement at *low* Δi). There is no such structure: the truth is a clean
  null. A future session revisiting F would try to explain a −0.33 that does not exist. Symmetrically,
  the mis-join **hides** the strongest instance of the same entry's own embedder-amplification claim:
  the corrected row is 1.1 % corrected I/F disagreement → **39.4 %** prediction disagreement, the
  second-largest in the table.
- **Evidence:**
  ```
  scripts/probes/_f_review_overlap_residual.py:51-54
      pred_pairs = pd.read_csv(FIG / "f_pilot_overlap_pairs.csv")
      ...
      pred_by_pair = dict(zip(pred_pairs["pair"], pred_pairs["median_absdiff"]))   # last wins

  scripts/probes/_f_review_overlap_residual.py:67,74
      pair = f"{pids[i][:3]}~{pids[j][:3]}"      # 7 frames -> 6 distinct keys
      pred_absdiff=pred_by_pair.get(pair, np.nan),

  reports/figures/f_pilot_overlap_pairs.csv:6-7   (the two B03~P22 pred rows)
      minnaert_log,f_wl,pred,B03~P22,6415,0.3941,0.084
      minnaert_log,f_wl,pred,B03~P22,5714,0.0904,0.36
  ```
- **Self-refutation attempted:** (i) *Do the two P22 frames really both appear?* Yes — the d_inc
  column resolves each row uniquely: 57.859 − 45.939 = 11.92 and 57.859 − 46.469 = 11.39 are both
  present, i.e. `B03` vs each `P22`. (ii) *Does the headline change?* No — `median pred_absdiff` is
  0.2044 both before and after, so the quoted "predictions disagree **20.4 %** median" stands, as do
  the 4.0 % / 10.2 % / "0.7–4 %" figures (which never touch the join) and the two per-pair examples
  quoted at `DECISIONS.md:5235` (both resolve to correctly-joined rows). Only the ρ is wrong. (iii)
  *Does the conclusion flip?* No — a null and a negative both refute "disagreement grows with Δi", so
  the direction of the argument survives; the record simply asserts a structure that is not there.
  (iv) *Is the ρ load-bearing?* It is one of three evidentiary clauses in the sentence that amended the
  verdict, so its severity is "the record", not "the verdict". (v) Not in the register: R11/R12/R33/R34
  /R36 cover Stage C/D and H4, none touches this probe.
- **Fix:** key the join on the full `PRODUCT_ID` pair (or on `(pair, n)`), and assert
  `pred_pairs["pair"].is_unique` before `dict(zip(...))`. Then either correct or delete the ρ = −0.33
  clause in `DECISIONS.md:5236`. Note additionally that with 15 pairs drawn from 7 frames each frame
  appears in ~5 pairs, so the pairs are pseudo-replicates; even the banked ρ had p = 0.23 and should
  never have been reported as a finding.

---

### probes-fbuild-3 — The record's diagnosis of the ESP_053989 minnaert inversion is arithmetically wrong and refuted by the same probe's committed table: 20 of 36 images get a *larger* cos^k step and none inverts

- **Severity:** medium (record correctness; the proposed fix targets a non-cause and is PLAN_FBuild
  P4's declared fallback)
- **Liveness:** dead-closed; P4 passed on 2026-07-14 so the fallback was never applied
- **Confidence:** high (the divisor arithmetic and the cohort ranking are both from committed CSVs)
- **Where:** producer of the falsifying data =
  [scripts/probes/_f_leg_b_mapping_compare.py:51-56](../../scripts/probes/_f_leg_b_mapping_compare.py#L51-L56)
  → `reports/f_leg_b/mapping_compare_per_image.csv`; and
  [scripts/probes/_f_leg_b_variant_summary.py](../../scripts/probes/_f_leg_b_variant_summary.py) →
  `reports/f_leg_b/variant_summary.csv`. Claim at `DECISIONS.md:5364-5369`; referenced as the fallback
  fix list by `PLAN_FBuild.md:55` and `PLAN_StripingArtifact.md:180`

> `DECISIONS.md:5366-5369` — "Its two frames (i = 42.76°, 46.32°) get cos^0.58 divisors **0.847 vs
> 0.826 — a ~2.5 % step** that global doesn't apply. So the illumination correction itself breaks this
> one image … (candidate: per-composite single divisor, or drop cos^k when a composite spans a Δi
> step)."

Two problems. **The arithmetic:** `cos(42.76°)^0.58 = 0.8359` and `cos(46.32°)^0.58 = 0.8068` — a
**3.61 %** step, not 2.5 %, and neither quoted divisor is right. **The cohort:** computing the same
step for all 36 obs from `mapping_compare_per_image.csv`, ESP_053989's 3.61 % ranks **21st of 36**;
**20 images receive a larger step**, up to **67.6 %** (ESP_051943_2270, i = 42.66°–72.44°), and **none
of the 20 inverts** — ESP_051943 *improves* (Δ minnaert +0.015, Δ log-minnaert +0.023). So a
within-composite Δi step cannot be the mechanism: it is present 20× more strongly elsewhere with no
effect.

The record already contains the mechanism that does survive, 236 lines earlier:
`DECISIONS.md:5602-5604` — ESP_053989 "*is the cohort's dimmest scene (I/F median 0.083) and after
÷cos^k ≈ 0.101 ≈ the stretch floor lo = 0.1011, so ~half its pixels clip to black*". That is
consistent with the eventual cure: P4 records the inversion **gone** under `minnaert_center`
(per-image AUC 0.884 vs mosaic 0.873, `PLAN_FBuild.md:55`), and H1's per-frame ln-median centering is
exactly a *level* operator that moves a scene off the stretch floor — it does nothing at all to a
within-composite Δi step.

- **Failure scenario:** had P4 failed, `PLAN_FBuild.md:55` sends the build session to "candidate fixes
  in DECISIONS 2026-07-05b caveat 1" — i.e. "per-composite single divisor, or drop cos^k when a
  composite spans a Δi step". Dropping the per-frame divisor on the 20 composites that span 7–68 %
  steps would have removed the correction from most of the cohort to fix an image whose step is
  below-median, at the cost of the one thing `cos^k(i)` does buy.
- **Evidence:**
  ```
  reports/f_leg_b/mapping_compare_per_image.csv  (cos^0.58 step, largest first; d_minnaert)
      ESP_051943_2270  42.658  72.438   67.6%   +0.015
      ESP_068402_2240  43.929  66.530   41.0%   -0.034
      ESP_054000_2255  46.640  65.670   34.5%   +0.081
      ...
      ESP_053989_2260  42.760  46.320    3.6%   -0.599   <-- the only inversion, 21st of 36
  n with step > ESP_053989's: 20   of those with d_minnaert < -0.3: 0
  ```
- **Self-refutation attempted:** (i) *Maybe the divisors quoted use a different k or incidence source*
  — solving `cos(i)^0.58 = 0.847 / 0.826` gives i = 41.3° / 44.0°, matching neither the quoted
  incidences nor any value in `frame_incidence.csv` for that obs; and `inc_min = 42.76` is exactly the
  PDS `OVERRIDES` value for `P20_008839`, so the incidences the record quotes *are* the ones in the
  banked table. (ii) *Maybe the step matters only when combined with dimness* — that is a different
  (and plausible) claim, and it is the stretch-floor mechanism at `:5602`, not the one at `:5366`;
  filing this is asking the log to keep the mechanism it verified rather than the one it did not.
  (iii) *Maybe the wide stretch already fixed the clipping, leaving the step* — no: the wide stretch's
  floor is 0.0965 and ESP_053989's post-divisor median ≈ 0.1017, still hard against it, and it stayed
  inverted (0.187 linear / 0.167 log). (iv) Not in the register.
- **Fix:** correct `DECISIONS.md:5366-5367` (divisors 0.836/0.807, 3.6 %), record that 20 images take a
  larger step without inverting, and point caveat 1's fix list at the stretch-floor mechanism the log
  already verified at `:5602-5604` (per-scene stretch headroom / the centering that P4 confirmed
  works) — not at the divisor.

---

### probes-fbuild-4 — The "post-minnaert 4.0 %" that opened the H1–H6 docket is algebraically a per-pair *constant* rescale, so it cannot distinguish "photometrically correctable" from "information-level" — and H5 (stronger physics) was downgraded on it

- **Severity:** medium (record correctness + a mis-scoped priority call on a closed docket item)
- **Liveness:** dead-closed; quoted in `ROADMAP.md:19`, `PLAN_StripingArtifact.md:191-192`,
  `:206-211`, `:323`, `:353`, `notebooks/_build_28.py:92-98`
- **Confidence:** high on the algebra (two lines of code); medium on how much the conclusion should
  move
- **Where:** [scripts/probes/_f_review_overlap_residual.py:63-66](../../scripts/probes/_f_review_overlap_residual.py#L63-L66),
  docstring [:1-11](../../scripts/probes/_f_review_overlap_residual.py#L1-L11); consumers
  `DECISIONS.md:5230-5244`, `:5285-5286`

The probe's stated purpose is to decide whether "*the residual floor is photometric-correctable
(corrected disagreement still large, tracks Δi) or information-level (corrected disagreement small;
predictions disagree anyway)*". As implemented, the correction is

```
raw  = np.median(np.abs(a[both] / b[both] - 1))
ac = a[both] / (cos_i[pids[i]] ** K);  bc = b[both] / (cos_i[pids[j]] ** K)
corr = np.median(np.abs(ac / bc - 1))
```

and `ac/bc = (a/b) · (cos_i_b / cos_i_a)^K` — a **single scalar per pair**, identical for every
co-located cell. The corrected ratio field is therefore the raw ratio field multiplied by a constant:
the operator can only remove a per-frame **multiplicative DC** term and is mathematically incapable of
changing the spatial *shape* of the disagreement. Consequences for the record:

- "*the photometric level correction largely WORKS*" (`DECISIONS.md:5233-5234`) and "*its measured
  reach: level residual 10.2 % → 4.0 %*" (`PLAN_StripingArtifact.md:323`) credit the Minnaert **form**
  for a reduction that **any** per-frame gain of roughly the right size achieves — which is precisely
  what the data-driven H1 log-median centering that actually shipped does, and H1 beat minnaert-only
  on every gate (`DECISIONS.md:5198-5210`).
- The 4.0 % is a residual **after** DC removal, so it is blind to any **additive** term. Atmospheric
  haze is additive in I/F — and additive contamination is exactly the mechanism the same entry
  hypothesises for F02 ("*atmosphere (dust/haze) or calibration-epoch offset*",
  `DECISIONS.md:5215-5222`). The visible symptom is in the banked table: `B03~F02` goes **0.0413 →
  0.1492**, i.e. the multiplicative correction makes that pair **3.6× worse**, the only pair it hurts.
  A multiplicative-only instrument classifies an additive term as "information-level" by construction.
- `DECISIONS.md:5285-5286` then downgrades **H5 (Hapke / atmospheric EPF)** to LOW priority —
  "*headroom now known small (4 % residual, mostly anomalous frames that H1 fixes empirically)*" — on
  a number produced by an instrument that cannot see the additive/atmospheric class H5 targets.

- **Failure scenario:** a session revisiting F reads "post-minnaert overlaps agree to 4 %" as
  "physical photometry is essentially solved; only the embedder and one bad frame remain", and skips
  both the additive/atmospheric leg (H5) and any *shape*-sensitive photometric test. The scope error
  compounds: the 4 % was measured on a ~1.3° crop where within-frame incidence is ~constant, and
  `PLAN_FBuild.md:163-174` later measured a **3–5 % geometry-predicted within-frame ramp** on real
  907-frame-scale frames — a photometric term of the same size as the entire quoted "remaining
  headroom", invisible to this instrument and to η².
- **Evidence:** the two lines above, plus
  ```
  reports/f_leg_b/review_overlap_residual.csv:10
      B03~F02,9495,6.951,26056,0.0413,0.1492,0.5749     # raw 4.1% -> "corrected" 14.9%
  ```
- **Self-refutation attempted:** (i) *Is k in-sample, making the improvement guaranteed?* No —
  `PLAN_FBuild.md:175-176` records k = 0.580 as the **training-cohort** fit while "*the pilot's own 7
  frames fit 0.694*", so the constant is genuinely out-of-sample for these frames and the 10.2 % → 4.0 %
  drop is a real (if DC-only) result. That kills the strongest version of this finding and is why it is
  medium, not high. (ii) *Is the DC framing unfair?* The overlap disagreement genuinely was mostly DC —
  that is the finding. The defect is the *attribution* ("photometric correction works", "H5 headroom is
  small") of a DC-only measurement. (iii) *Did the project catch the crop-width scope issue?* Partly —
  `PLAN_FBuild.md:163` flags the within-frame ramp as a **build** risk, but never revisits the 4 %
  headroom claim or H5's priority in its light. (iv) *Already filed?* R41 covers "no gated statistic
  has a sampling spread"; this is a different defect (instrument scope), and §5's refuted list does not
  contain it. Also note `|a/b − 1|` is asymmetric in pair order (`|r−1|` vs `|1/r−1| = |r−1|/r`), so
  both headline percentages depend on the alphabetical `pids` ordering by up to ~1/r; at r ≈ 0.86 that
  is ~16 % relative on the raw figure — immaterial to the verdict, worth a footnote.
- **Fix:** state in `DECISIONS.md:5230-5244` that the correction applied was a per-pair scalar, so
  "4.0 %" is the **post-DC-removal** residual (equivalently: the floor for any per-frame gain,
  including H1), and that it therefore bounds neither additive/atmospheric nor within-frame
  photometric error. Re-state H5's priority against a shape-sensitive instrument (e.g. residual
  disagreement after per-frame DC removal, decomposed against Δphase / Δemission / within-frame
  latitude), not against this number.

---

### probes-fbuild-5 — The "over-stretch REFUTED (ρ = +0.09)" test could not have produced anything else: `f_iqr` is a two-valued constant, so the "F/mosaic contrast ratio" is the mosaic baseline's own contrast, inverted

- **Severity:** low (the conclusion is right and the record discloses the pinning; the *statistic*
  offered as the refutation carries no F-side information) — logged because it is another instance of
  the register's **Pattern A**
- **Liveness:** dead-closed
- **Confidence:** high (measured)
- **Where:** [scripts/probes/_f_leg_b_uint8_contrast.py:53](../../scripts/probes/_f_leg_b_uint8_contrast.py#L53)
  (`df["iqr_ratio"] = df["f_iqr"] / df["mosaic_iqr"]`),
  [:58-59](../../scripts/probes/_f_leg_b_uint8_contrast.py#L58-L59); consumers `DECISIONS.md:5651-5654`,
  `notebooks/_build_27.py:152-154`, `reports/figures/f_leg_b_diag_scatter.png`

Across all 36 images `f_iqr` takes exactly **two** values, 27 and 28 (sd 0.494), because the perframe
mapping pins the composite IQR to 27.7 DN. `mosaic_iqr` spans 19–57 (sd 10.6). Hence
`Spearman(iqr_ratio, mosaic_iqr) = −0.9974` and `Spearman(iqr_ratio, 1/mosaic_iqr) = +0.9974`: the
"F/mosaic contrast ratio" is a monotone transform of the **mosaic** window's own contrast. The reported
`ρ(ΔAUC, iqr_ratio) = +0.091` is, up to the tie structure, just `−ρ(ΔAUC, mosaic_iqr) = +0.075`. The
over-stretch hypothesis is in fact refuted by the **pinning itself** (there is no F-side variance to
correlate), which `DECISIONS.md:5652-5653` correctly states — so the ρ adds nothing and, being a
baseline-side statistic, could not have gone the other way.

- **Failure scenario:** the same construction re-used on a mapping whose IQR is *not* pinned (A2
  histogram matching, or the F build's fixed-stretch variants) would be read as an F-side test when it
  is a baseline-side one; more immediately, a reader treats "ρ = +0.09, null" as independent evidence
  when it is a restatement of the by-construction pinning in the previous clause.
- **Evidence:**
  ```
  reports/f_leg_b/diag_uint8_contrast.csv:  f_iqr in {27.0, 28.0} for all 36 rows
  spearman(iqr_ratio, mosaic_iqr) = -0.9974  (p = 1.7e-40)
  spearman(d_auc, iqr_ratio) = +0.091 ;  spearman(d_auc, mosaic_iqr) = -0.075
  ```
- **Self-refutation attempted:** the record *does* name the pinning, so this is not a hidden
  tautology — which is why it is `low`. It survives as a finding only because the ρ is presented as
  the refutation ("*and the F/mosaic contrast ratio is null vs ΔAUC (ρ = +0.09)*") in DECISIONS, the
  notebook and a committed figure title, and because the Pattern-A census in
  `docs/CODE_REVIEW_2026-07-31.md:156-168` is explicitly collecting instances of this shape. The
  probe's *other* output — "mosaic windows vary 19–57" — is correct and is the native 5 m/px Stage-2
  window IQR (`win_iqr` on `sc["ctx_window_tif"]`, zeros excluded), so anything relying on that range
  is safe.
- **Fix:** report `sd(f_iqr)` beside the ratio ρ (the same "report the treatment's magnitude beside the
  metric delta" rule the register's Pattern-A section proposes), and re-word the DECISIONS clause so
  the pinning, not the correlation, is the refutation.

---

### probes-fbuild-6 — `DECISIONS` credits DOI verification to two probes that resolve no DOIs; the script that does verify checks 4 of ~13 hyperlinked citations, is cited nowhere, and banks no log

- **Severity:** low (record provenance, against a standing CLAUDE.md rule)
- **Liveness:** dead-closed entry, live project rule ("hyperlink every citation to its canonical DOI")
- **Confidence:** high for the code; the DOIs themselves were **not** checked (network out of scope)
- **Where:** `DECISIONS.md:5254` ("**Literature anchors (verified DOIs, `_f_litreview_queries*.py`)**")
  vs [scripts/probes/_f_litreview_queries.py:39-46](../../scripts/probes/_f_litreview_queries.py#L39-L46)
  and [_f_litreview_queries2.py:32-37](../../scripts/probes/_f_litreview_queries2.py#L32-L37) (both
  keyword `search=` only) vs
  [_f_litreview_verify.py:17-22](../../scripts/probes/_f_litreview_verify.py#L17-L22) (the real DOI
  resolver)

`DECISIONS.md:5254-5269` hyperlinks ~13 DOIs under the heading "verified DOIs,
`_f_litreview_queries*.py`". Neither named script resolves a DOI: both issue OpenAlex
`works?search=<phrase>` keyword queries and print the top hits. The only script that resolves DOIs is
`_f_litreview_verify.py`, which checks **exactly four** — Claverie 2018, Roy 2016, Bickel 2020 and a
paper it labels "Mars-from-Moon DA 2022" (`10.1109/jstars.2022.3156371`), which the record renders as
"[Lagain-adjacent JSTARS 2022]" — and it is cited in no document. `_f_litreview_queries2.py:63` also
fetches Dickson 2024's abstract by DOI. No stdout is banked anywhere under `reports/`, so the
remaining ~8 citations (Canty & Nielsen 2004/2007, Du–Teillet–Cihlar 2002, Edwards 2011, Li 2022, Tuia
2016, Deep CORAL, Fernando 2012) have **no** verification trail, and the "Lagain-adjacent" hedge in
the record suggests at least one DOI↔author binding was never pinned down.

- **Failure scenario:** a DOI that resolves to a different paper than the claim it supports propagates
  into a thesis chapter or a paper, on a project whose operating manual makes DOI hyperlinking a
  standing rule. The specific candidate is the "planetary DA precedent" whose own probe label
  ("Mars-from-Moon DA 2022") does not match the citation text ("Lagain-adjacent").
- **Self-refutation attempted:** DOIs printed by `queries*.py` come *from* OpenAlex alongside their
  titles, so any citation lifted from that output has a correct DOI↔title binding at the moment it was
  read — the defect is that nothing was persisted, so it is unauditable rather than demonstrably
  wrong. I could not check any DOI directly (network is out of scope for this review), so this is a
  provenance finding, not a factual one.
- **Fix:** point `DECISIONS.md:5254` at `_f_litreview_verify.py`, extend its `DOIS` dict to all 13
  cited DOIs, and commit its output as `reports/f_leg_b/litreview_dois.txt` (title + year + DOI per
  citation) so the anchors are auditable.

---

## Refuted by my own check

- **"`_f_review_overlap_residual.py` used the untrusted SeamMap incidence while the production leg-B
  path used PDS truth."** True as stated —
  [`:45-46`](../../scripts/probes/_f_review_overlap_residual.py#L45-L46) reads
  `frame_table(TILE)["INCIDENCE"]`, which `src/ctx_edr.py:38` sources from
  `src.striping.load_frames` (the SeamMap), whereas `scripts/f_leg_b_embed.py:53` reads the
  PDS-corrected `frame_incidence.csv`. But it is immaterial here: none of the 7 E8_N44 pilot frames is
  in the `OVERRIDES` table, and `reports/f_timing/frame_list.csv` (SeamMap-derived) agrees with
  `frame_incidence.csv` on both E8_N44 frames present in both (P21_009338 = 44.28, and the PDS table
  carries no other pilot frame). `DECISIONS.md:4859` separately records "SeamMap-vs-PDS incidence: 0
  disagreements > 1°" over the 907-frame cohort, so the decimal-shift class is confined to
  P20_008839. Downgraded to a note.
- **"The `_f_leg_b_crop_stats.py` n = 6 gallery sample is too small to 'rule out' between-frame
  illumination mismatch inside a composite."** The claim (`DECISIONS.md:5647-5649`,
  `notebooks/_build_27.py:149-151`) is honestly labelled as the 6-image sample, and it **holds** on
  the full cohort: using the within-composite incidence span from `mapping_compare_per_image.csv` as
  the mismatch proxy, ρ(ΔAUC, span) = +0.033 (n = 36) and +0.338 (n = 28 multi-crop) — same
  (anti-)direction as the record, never adverse. Not a defect.
- **"The F02 z-score is a deflated, in-sample statistic."** `_f02_diagnose.py:56-58` fits a 2-parameter
  line through 7 points and z-scores the residuals with `ddof=0`, so the sd is deflated by
  √(7/5) = 1.18 (z = −2.23 → −1.88 with ddof = 2), the `− mean` is a no-op for an OLS-with-intercept
  residual, F02 has leverage on its own fit, and with n = 7 the maximum attainable |z| is √6 = 2.449 —
  so "z = −2.23" sits at 91 % of its structural ceiling and should not be read as p ≈ 0.026. But the
  F02 anomaly is independently corroborated (its corrected residual stays 10–15 % against **all four**
  partners in `review_overlap_residual.csv`; it is the darkest frame even minnaert-corrected;
  `B03~F02` is the one pair the correction worsens) and every quoted number reproduces. Not filed.
- **"`_f_leg_b_pds_incidence.py` silently drops frames missing from the PDS volume index."** True
  ([`:60-62`](../../scripts/probes/_f_leg_b_pds_incidence.py#L60-L62) `continue`s without appending),
  but no frame was dropped in the banked run: `cohort_frame_list.csv` and `frame_incidence.csv` both
  carry **81** rows. Related and also not filed: the probe **overwrites its own input in place**
  (`OUT` is both the "current SeamMap" table it diffs against at `:47` and its output at `:73`), so
  the SeamMap-vs-PDS delta report is not re-runnable — a second run compares PDS against PDS and
  prints zero deltas. Same "baseline silently overwritten in place" shape as **R37**, but with no
  quoted number depending on it.
- **"Δ median per-image AUC is a difference of medians, not the median of paired deltas."** True, and
  the two disagree materially: `minn_wide_LOG` is **+0.0067** as a difference of medians but
  **−0.0006** as the paired median (18 wins / 18 losses), so the headline "F now EXCEEDS the mosaic
  baseline (0.786 → 0.793)" is a paired *tie*. But this is the **pre-registered** gate — the shipped
  `scripts/f_leg_b_loio.py:192,198` computes `f_s["median_auc"] - b_s["median_auc"]` — so the probes
  (`_f_leg_b_variant_summary.py:68`, `_f_leg_b_mapping_compare.py:64`) faithfully reproduce the
  declared statistic. Both readings clear the −0.02 bar. Recorded here so it is not re-filed; it
  belongs with **R41**, not as a probe defect.
- **"`_f_leg_b_diag.py` computes presence AUC."** No — `reports/figures/f_leg_b_loio_preds*.csv`
  carries `y` already binarised by `scripts/f_leg_b_loio.py:42` `TARGET = "fa_gt_1e-2"`, so every
  per-image AUC in this area is the mandated rich/poor AUC. Invariant 8 is respected throughout
  `probes-fbuild`.
- **"`_f_leg_b_quant_check.py:34` renames pivot columns positionally."** `auc.columns = ["base",
  "f_w"]` relies on `"fang_embeddings" < "fang_embeddings_f_minnaert_w"` alphabetically. Correct as it
  stands, fragile in principle, cited nowhere, writes an uncited CSV → not worth a finding.
- **"`_f_leg_b_mapping_compare.py:38-47` takes its baseline from the first file only."** It does
  (`if base is None: base = b`), so `d_global`/`d_minnaert` are measured against the *perframe* run's
  baseline. Checked: `variant_summary.csv`'s independently-derived `baseline` column matches
  `mapping_compare_per_image.csv`'s to full precision for all 36 obs, i.e. the baseline store is
  identical across runs. No effect.

## Verified clean

- **The 2026-07-05d headline numbers.** `median raw_absratio = 0.1019` (→ "10.2 % raw"),
  `median minn_absratio = 0.0395` (→ "**4.0 %**"), the five highest-Δi pairs at 0.73–4.68 % (→ "worst
  high-Δi pairs drop to 0.7–4 %"), `median pred_absdiff = 0.2044` (→ "**20.4 %**"), and the two
  per-pair examples quoted at `DECISIONS.md:5235` (`P21~P22` 1.0 % → 20.5 %, `B03~P21` 0.8 % → 20.4 %)
  all reproduce exactly from `reports/f_leg_b/review_overlap_residual.csv`, and the 20.4 % median is
  **unchanged** by the join bug in `probes-fbuild-2`. F02's "corrected residual stays 10–15 % vs every
  partner" reproduces (0.1076–0.1536).
- **The leg-B gate table** (`DECISIONS.md:5344-5349`): `minn_wide_LOG` 18 wins / 18 losses,
  Δmedian +0.00672, mean 0.747, 3 images below 0.5; `perframe` −0.0499 / 8 below 0.5 / 11 improvers;
  ESP_053989 = 0.16687 and ESP_068483 = 0.84579 — all reproduce from
  `reports/f_leg_b/variant_summary.csv`.
- **`_f_leg_b_diag.py`'s "composite mechanics ruled out"**: coverage ≥ 0.999 on every image, and
  ρ(ΔAUC, overlap) = −0.059, ρ(ΔAUC, n_crops) = −0.063 — matches the record's "|ρ| < 0.07".
- **"mosaic windows vary 19–57"** is the native 5 m/px Stage-2 CTX window uint8 IQR with zeros
  (nodata) excluded — correct as described, and safe for anything downstream that quotes the range.
- **Invariant 8 throughout** — every AUC in this area is on `fa_gt_1e-2`, never presence.
- **Invariant 9 throughout** — every probe that imports torch-dependent code puts
  `import src.modeling` before numpy/pandas (`_f02_diagnose:20`, `_f_leg_b_diag:19`,
  `_f_leg_b_blur_check:22`, `_f_leg_b_crop_stats:9`, `_f_leg_b_figures:24`,
  `_f_leg_b_mapping_compare:12`, `_f_leg_b_quant_check:16`, `_f_leg_b_uint8_contrast:18`,
  `_f_leg_b_variant_summary:13`, `_f_review_overlap_residual:18`), and every probe that fetches
  calls `truststore.inject_into_ssl()` before use (`_f_leg_b_pds_incidence:18-20`,
  `_f_leg_b_fetch_true_incidence:10-12`, `_f_edr_url_verify:1-3`, all three `_f_litreview_*`).
- **`coarse()`'s nodata handling** (`_f_review_overlap_residual.py:32-40`): `np.nanmean` over 32×32
  blocks with `frac < 0.9 → NaN`, then `np.isfinite(a) & np.isfinite(b)` and a ≥200-cell minimum
  before any statistic — no nodata sentinel enters the ratio.
- **`_f_leg_b_blur_check.py`'s stretch-invariance argument** is sound: `var(∇)/var(x)` and
  `var(Lap)/var(x)` are invariant to `x → αx + β`, so comparing mosaic uint8 against F float I/F is
  legitimate. Its uint8-quantization floor is ~1.7/var(x) ≈ 0.001–0.007, negligible against the ~40 %
  deficit reported; and the probe is cited nowhere anyway.
- **`_f_edr_url_verify.py`** samples 3 frames × 4 tiles ordered by `VOLUME_ID` (first/middle/last), so
  the "12/12 mission-spanning" claim in `src/ctx_edr.py:1-10` is fairly constructed.

## Coverage note

**Read in full (all 20 files, 1 242 lines):** `_f02_diagnose.py`, `_f_edr_url_verify.py`,
`_f_leg_b_blur_check.py`, `_f_leg_b_crop_stats.py`, `_f_leg_b_diag.py`,
`_f_leg_b_fetch_true_incidence.py`, `_f_leg_b_figures.py`, `_f_leg_b_incidence_check.py`,
`_f_leg_b_mapping_compare.py`, `_f_leg_b_pds_incidence.py`, `_f_leg_b_quant_check.py`,
`_f_leg_b_uint8_contrast.py`, `_f_leg_b_variant_summary.py`, `_f_litreview_queries.py`,
`_f_litreview_queries2.py`, `_f_litreview_verify.py`, `_f_pilot_bounds.py`,
`_f_review_overlap_residual.py`, `_f_seammap_probe.py`, `_inspect_seammap_E12_N44.py`.

**Also read:** `DECISIONS.md` 2026-07-03b / -04b / -05 / -05b / -05c / -05d / -07 / -13 / -14 (by
grep + context), `PLAN_FBuild.md` §0/§3/§4, `PLAN_StripingArtifact.md` PHASE 2 + §A-meta rows,
`ROADMAP.md:19`, `notebooks/_build_27.py` (§2–§4) and `_build_28.py:61,92-98,258`,
`scripts/f_leg_b_loio.py` (gate definition), `src/ctx_edr.py`, `src/striping.py:145-190`
(`load_frames`).

**Reproduced numerically** from committed CSVs: the 2026-07-05d overlap table and every percentage
quoted from it; the corrected pair join; the six leg-B diagnostic correlations plus partials against
PDS incidence; the cos^0.58 divisor step for all 36 obs; the variant win/loss table and both
Δ-median conventions.

**Could NOT check (and why):**
- Anything requiring the actual rasters — `reports/f_timing/pilot_work/aligned/*.npy`,
  `reports/f_leg_b/obs_crops/*_ifcrop.tif`, `dataset_v2/labels/*.json`'s `ctx_window_tif` — is
  imagery, excluded by the rules of engagement. So I could not recompute `if_median` on the true
  last-write-wins composite (`probes-fbuild-1a`), re-derive `f_iqr`/`mosaic_iqr`, re-derive the
  blur-check HF ratios, or measure how close the 4.0 % residual sits to the best-achievable per-pair
  constant (`probes-fbuild-4`). All four are analytic/citation arguments plus committed-CSV
  statistics, not re-measurements.
- **No DOI was resolved** — the network is out of scope, so `probes-fbuild-6` is a provenance finding
  only; the ~9 unverified citations in `DECISIONS.md:5254-5269` remain genuinely unchecked, including
  the "Lagain-adjacent JSTARS 2022" label.
- **`reports/f_leg_b/*.log`** (h1/h2/h3/h4 run logs, 11 files) were not read line-by-line; I grepped
  them for the numbers I needed. `h1_f02_diagnose.log` in particular is the banked stdout of
  `_f02_diagnose.py` and would let a future session confirm the fitted k and the seven residuals
  without re-running anything.
- **`_f_pilot_bounds.py`, `_f_seammap_probe.py`, `_inspect_seammap_E12_N44.py`, `_f_edr_url_verify.py`**
  are one-shot column/geometry/URL prints; read but not audited numerically (nothing to audit — no
  derived statistic reaches a doc beyond the "12/12" URL check and the "SeamMap carries INCIDENCE"
  fact, both correct).

## Load-bearing map

| probe | cited by | number it produced | verdict |
|---|---|---|---|
| `_f_review_overlap_residual.py` (92) | `DECISIONS.md:5230-5236`; `PLAN_StripingArtifact.md:191-192,206-211,323,353`; `ROADMAP.md:19`; `notebooks/_build_28.py:92-98`; writes `reports/f_leg_b/review_overlap_residual.csv` | the 2026-07-05d amended verdict: raw 10.2 % → **post-minnaert 4.0 %**, worst high-Δi pairs 0.7–4 %, prediction \|diff\| 20.4 %, F02 residual 10–15 %, **ρ(pred,Δi) = −0.33** | percentages all reproduce; **ρ = −0.33 → −0.03 once a truncated 3-char join key is fixed** (`-2`); the "minnaert correction" is a per-pair **constant**, so 4.0 % bounds only per-frame DC error, yet H5 was downgraded on it (`-4`) |
| `_f_leg_b_diag.py` (91) | `DECISIONS.md:5638-5656`; `notebooks/_build_27.py:144-183`; `PLAN_StripingArtifact.md:163`; writes `reports/f_leg_b/diag_per_image.csv` | per-image ΔAUC (bimodal, 11 improve / 8 below 0.5); coverage/overlap/n_crops null (\|ρ\|<0.07); **`if_median` ρ = +0.35 → "illumination is the live correlate"** | ΔAUC + null correlates reproduce; **the +0.35 mechanism claim is a brightness/albedo confound — incidence itself is ρ = +0.06, p = 0.74** (`-1`); `if_median` is a crop-concatenation, not the composite median |
| `_f_leg_b_uint8_contrast.py` (66) | `DECISIONS.md:5651-5654`; `notebooks/_build_27.py:152-154`; `PLAN_StripingArtifact.md:163`; writes `reports/f_leg_b/diag_uint8_contrast.csv` | "F windows pinned at uint8 IQR 27–28, **mosaic windows vary 19–57**"; over-stretch REFUTED at **ρ = +0.09** | 19–57 range **correct** (native 5 m/px window, zeros excluded); `f_iqr ∈ {27,28}` so `iqr_ratio` is Spearman −0.997 with `mosaic_iqr` → the ρ is a baseline-side statistic that could not have differed (`-5`) |
| `_f02_diagnose.py` (79) | `DECISIONS.md:5215-5226`; `notebooks/_build_28.py:219`; banks `reports/f_leg_b/h1_f02_diagnose.log` | F02 is **−0.114 in log (z = −2.23)** below the cos-i photometric line; minnaert I/F 0.116 vs 0.130–0.138; frame-mean P(rich) 0.222 vs ≤0.173 | numbers consistent and the conclusion is independently corroborated; the **z is in-sample with `ddof=0` on a 2-param fit at n = 7** (max attainable \|z\| = 2.449, so −2.23 ≈ 91 % of ceiling) → do not read it as p ≈ 0.026. Not filed |
| `_f_leg_b_crop_stats.py` (28) | `DECISIONS.md:5647-5649`; `notebooks/_build_27.py:149-151`; `PLAN_StripingArtifact.md` (leg-B rows) | improvers' frame-median ratio **1.43–1.58×** vs collapsed 1.02–1.30× → "composite illumination mismatch ruled out" | n = 6 hardcoded gallery sample, but the conclusion **holds** on all 36 via the banked incidence span (ρ = +0.03 / +0.34, never adverse). Clean |
| `_f_leg_b_pds_incidence.py` (78) | `DECISIONS.md:5375`; `PLAN_FBuild.md:160`; `PLAN_StripingArtifact.md:222`; **overwrites** `reports/f_leg_b/frame_incidence.csv` | the 81-frame PDS incidence table (37.9–67.4°) that every minnaert variant and the H1 recipe use | table is complete (81/81, no silent drops) and is PDS truth as claimed; **overwrites its own diff baseline in place**, so the SeamMap-vs-PDS delta report is not re-runnable (R37-shaped; not filed) |
| `_f_leg_b_fetch_true_incidence.py` (29) | `DECISIONS.md:5599`; `_f_leg_b_incidence_check.py:14` | P20_008839 SeamMap **4.2759° = decimal shift of true 42.76°** | clean — single-purpose PDS index lookup; the bug and the fix are both real and propagated |
| `_f_leg_b_incidence_check.py` (64) | `DECISIONS.md:5601`; `PLAN_FBuild.md:160` (the `OVERRIDES` table) | the pre-PDS `frame_incidence.csv` + the `OVERRIDES` mechanism | clean; superseded in place by `_f_leg_b_pds_incidence.py` |
| `_f_leg_b_figures.py` (137) | `DECISIONS.md:5656-5659`; `notebooks/_build_27.py:186-203`; writes committed `reports/figures/f_leg_b_diag_{scatter,gallery}.png` | the two published leg-B diagnostic figures, incl. the axis title "DIM scenes collapse (ρ=+0.35) — illumination is the live correlate" | plotting is faithful to `diag_per_image.csv`; the **caption asserts the confounded mechanism** (`-1`) on a committed figure |
| `_f_litreview_queries.py` (62) | `DECISIONS.md:5254` (as the source of "verified DOIs") | keyword search hits only — **no DOI resolution** | the record's verification attribution is wrong (`-6`) |
| `_f_litreview_verify.py` (49) | nowhere | resolves **4** DOIs (Claverie 2018, Roy 2016, Bickel 2020, JSTARS 2022) | the only actual verifier, uncited, no banked output (`-6`) |
| `_f_litreview_queries2.py` (69) | nowhere | 11 more keyword searches + Dickson 2024 abstract by DOI | throwaway (`-6`) |
| `_f_edr_url_verify.py` (42) | `src/ctx_edr.py:1-10`; `DECISIONS.md` 2026-07-02 | "**12/12** resolve to live PDS3 EDRs via the `mro/ctx` template" | clean — sampling is mission-spanning as claimed |
| `_f_leg_b_variant_summary.py` (73) | nowhere in docs; writes committed `reports/f_leg_b/variant_summary.csv` | the 6-variant × 36-image per-image AUC table behind `DECISIONS.md:5344-5349` | arithmetic reproduces exactly; Δmedian is a difference-of-medians (the pre-registered gate) but the **paired** median for the PASSING variant is −0.0006, not +0.0067 — R41 territory, not re-filed |
| `_f_leg_b_mapping_compare.py` (73) | nowhere in docs; writes committed `reports/f_leg_b/mapping_compare_per_image.csv` | per-obs baseline/perframe/global/minnaert AUC + `inc_min`/`inc_max` | clean, and it is the artifact that **falsifies** the record's ESP_053989 diagnosis (`-3`) and supplies the incidence axis for `-1` |
| `_f_leg_b_quant_check.py` (57) | nowhere; writes `reports/f_leg_b/quant_check.csv` | per-image post-minnaert DN IQR vs ΔAUC | uncited; positional pivot rename is fragile but correct. Nothing depends on it |
| `_f_leg_b_blur_check.py` (98) | nowhere by name (the "≈40 % HF deficit" it produced is quoted at `DECISIONS.md:5360`); writes `reports/f_leg_b/blur_check.csv` | HF / gradient-energy ratio F vs mosaic | the stretch-invariance argument is sound; note `DECISIONS.md:5358-5361` closes the blur hypothesis on a **0.0034** AUC difference (cubic −0.0270 vs linear −0.0236), i.e. inside the noise band **R41** describes |
| `_f_pilot_bounds.py` (11) | `scripts/f_pilot_extract_crop.py` (bounds it produced) | world bounds of the E8_N44 A1-payoff crop | trivial, correct |
| `_f_seammap_probe.py` (10) | nowhere | SeamMap column listing | throwaway |
| `_inspect_seammap_E12_N44.py` (34) | nowhere | "the SeamMap embeds per-source illumination angles" (corrects notebook 13) | throwaway, but the fact it established is right and is what `src/ctx_edr.frame_table` relies on |
