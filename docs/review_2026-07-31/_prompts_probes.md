# Reviewer briefs — `scripts/probes/` (184 files, 17.7k LOC)

The largest blind spot left by the 2026-07-31 review: no reviewer opened `scripts/probes/`, and it is
the origin of many numbers quoted in `DECISIONS.md`, the `PLAN_*.md` files and `docs/`. Same protocol
as `_prompts.md`: **an area is done iff `docs/review_2026-07-31/<area>.md` exists.** Run 3–4 at a time;
each agent writes its own file as its final action.

Remaining areas: `probes-fm-recipe`, `probes-tier2-calibration`, `probes-stage6`,
`probes-compression-targets`, `probes-stage7`, `probes-w1-geospatial`, `probes-fbuild`,
`probes-utility`.

---

## 1. Shared brief

Read **§1 of [`_prompts.md`](_prompts.md)** first — project description, the 10 load-bearing
invariants, the environment rules, and the rules of engagement. All of it applies. Then read
**`docs/CODE_REVIEW_2026-07-31.md`** §3 (priority), §4/§4b/§4c/§4d/§4e (findings **R01–R47**) and §5
(refuted / verified-clean). **Do not re-report anything already in the register**, and do not re-file
anything in §5.

### What makes a probe finding valuable

A probe is throwaway analysis code, so ordinary code smells in one are *not* worth reporting. The
question that matters is:

> **Did a number this probe computed reach `DECISIONS.md`, a `PLAN_*.md`, `docs/`, `PROMOTION_QUEUE.md`,
> a committed `reports/figures/*` artifact, or a notebook — and is that number right?**

So the workflow for each probe is:
1. **Is it load-bearing?** `grep -n "<probe_name>" DECISIONS.md docs/*.md PLAN_*.md README.md
   PROMOTION_QUEUE.md ROADMAP.md notebooks/_build_*.py`, and check whether it writes into
   `reports/` or `models/`. A probe cited nowhere and writing nothing is **low priority** — note it and
   move on.
2. **If it is load-bearing, audit the statistic it produced**, not its style: the metric definition,
   the population it was computed over, the comparison it licensed, and whether the conclusion recorded
   in the doc actually follows. Reproduce the number from committed artifacts where you can.
3. **Check it against the invariants** — especially **never presence AUC** (invariant 8), group-aware
   LOIO (6), and per-image local-radius CRS (1).

### Failure modes already found elsewhere in this codebase — look for more of the same

- **A gate that could not fail** (R36, R11, R43, `leakage-3`): the statistic is mathematically pinned by
  the construction it is meant to test. This is the single most productive pattern in this repo.
- **Silent population change**: a metric averaged over a different n than the doc claims (R24: a mean
  over 5 of 20 folds reported as 20; R12/R33: rows dropped without a note).
- **Prevalence / ceiling dependence**: `precision@k` and raw `lift@k` compared across populations with
  different base rates (R26, `notebooks-5`).
- **Two implementations of one headline**: `_w2_fang_probe.verdict()` recomputes `pooled_pr_auc` rather
  than reading the artifact (R25). Where else does a probe reimplement a `src/` metric, and do the two
  agree?
- **Presence AUC under another name** (R02, R25, `other-scripts-4`).
- **A doc quoting a number the producing code no longer computes** (R43, `docs-consistency-3`,
  `notebooks-6`).
- **Transductive estimation** presented as deployable (`fm-embeddings-1`).

### Rules of engagement (in addition to `_prompts.md` §1)

- **READ-ONLY except your own output file.** Do not run the probes — many touch CTX/HiRISE imagery, the
  network, or GPUs. You may read committed `reports/figures/*.csv|json` and `models/**/metrics.json`
  with small pandas snippets to reproduce a number.
- Cite `path:line`, quote the offending lines, and **self-refute before reporting**.
- At most 6 findings, ranked most-severe first. Most of these areas will legitimately yield 2–4.
- **Severity is about the record, not the probe.** A bug in a probe whose number nobody cited is `low`.
  A correct-looking probe whose number is quoted as a verdict in `ROADMAP.md` and is wrong is `high`.

### Output

Write `docs/review_2026-07-31/<area>.md` using the **§3 template in `_prompts.md`**, as your FINAL
action, even if you found nothing. Add one extra section before *Coverage note*:

```markdown
## Load-bearing map
| probe | cited by | number it produced | verdict |
|---|---|---|---|
```
listing every probe in your area that is cited anywhere or writes a committed artifact, so a future
session can see at a glance which probes matter.

---

## 2. Per-area briefs

### `probes-fm-recipe`
**The origin of the frozen recipe — highest-value area.**
Files: `_w2_fang_probe.py` (255), `_w2_fang_embed.py` (291), `_w2_fang_heads.py` (329),
`_fm_freeze_window.py` (389), `_w2_cnn_verdict.py` (158), `_w2_fang_azimuth.py`,
`_w2_fang_ckpt_keys.py`, `_w2_fang_head_pairs.py`, `_w2_fang_inspect.py`, `_w2_fang_patch_visual.py`,
`_w2_adabn.py`, `_w2_azimuth_spread.py`, `_w2_fusion.py`, `_w2_midgrid_diag.py`,
`_w2_photonly_read.py`, `_w2_s32_confirm.py`, `_w2_seed_ensemble.py`, `_fm_parity_check.py`,
`_fm_count_dist.py`.

Priorities: (a) **`_w2_fang_probe.verdict()` computes the headline `pooled_pr_auc 0.7832` /
`precision@5% 0.948` / `med per-image AUC 0.7865`** that is baked into `models/*/recipe.json`,
`ROADMAP.md`, `PLAN_FM.md` and `docs/`. Audit that function line by line: the pooling (does it
concatenate 38 per-fold models' outputs, and does that mix output scales?), the `k = max(1, int(0.05n))`
cutoff, tie-breaking, the `fa > 1e-2` binarisation, and whether any fold is dropped. Reproduce it from
`models/fang_probe/**/predictions.parquet` if you can. (b) **`_w2_fang_embed.py` is the training-time
embedding extractor** — it is the other half of the train/deploy parity question (`R07` found the A1
variant inverted; verify the plain path). (c) `_fm_freeze_window.py` is what *closed* the freeze
decision — check the bake-off is apples-to-apples across heads. (d) `_w2_cnn_verdict.py` decided
PLAN_CNN was superseded; check the CNN arm was given a fair comparison (cf. `modeling-heads-6`, the
brightness-jitter confounder on the augmentation refutation).

### `probes-tier2-calibration`
Files: `_diag_tier1_accuracy.py`, `_diag_tier1_beta.py`, `_diag_tier1_isotonic.py`,
`_diag_tier2_compression_direction.py`, `_diag_tier2_l1_bakeoff.py` (378), `_diag_tier2_minconf_sweep.py`,
`_diag_tier2_objectives.py`, `_diag_tier2_reweight.py`, `_diag_tier2_scale_sweep.py`,
`_diag_tier2_variant_compression.py`, `_fm_tier2_ceiling.py`, `_fm_tier2_collect.py`,
`_fm_tier2_regression.py` (331), `_fm_reliability_inspect.py`, `_fm_reliability_smoke.py`,
`_fm_reliability_validation.py`, `_diag_calibration_preview.py`.

These produced the Stage-0/1/2 calibration decisions (`PLAN_Calibration.md`): "isotonic for Tier-1,
qmatch de-compresses Tier-2", the Stage-2 "retraining ceiling = the 5 m/px CTX floor" closure, and the
`§2.7 reliability overlay LOIO-NEGATIVE at n=38` deferral. Audit: is each comparison LOIO-honest (the
`calibration` area verified `src/`'s protocol — check the *probes* match it); is `top_ratio` computed
pooled or per-image and does the doc say which (cf. R26, and the `DECISIONS.md:5049-5053` ruling); does
`_fm_tier2_ceiling.py` establish a *ceiling* or just an observed maximum; and does the reliability leg's
"LOIO-NEGATIVE at n=38" verdict have the power to support a deferral.

### `probes-stage6`
Files: `_sweep_stage6a.py` (303), `_sweep_stage6b.py` (330), `_diag_stage6a_fold_variance.py`,
`_diag_stage6a_followup_compare.py`, `_diag_stage6b_h3_check.py` (235), `_stage6c_gate.py` (661),
`_stage6c_gate_v2.py` (381), `_sweep_perimage_std.py`, `_diag_within_image_deltas.py`,
`_diag_within_image_smoke.py`, `_inspect_stage6b_output.py`.

`_diag_stage6b_h3_check` and `_stage6c_gate*` are the most-cited probes in the whole directory (8 and 12
citations). They produced "6a dev-PASS deferred; 6b strict-FAIL but mechanism validated; 6c soft-PASS
(ridge Strategy B +0.056)". Audit each verdict: what exactly was the gate, was it pre-declared, and is
`+0.056` inside the noise (cf. **R41**)? **Cross-check `leakage-1`**, which found Stage-6a neighbour
features are computed across the within-image quadrant cut so only the treatment arm carries
training-fold values — does that invalidate the dev-PASS these probes recorded? Also check whether
`_stage6c_gate_v2.py` supersedes `_stage6c_gate.py` and whether the docs cite the right one.

### `probes-compression-targets`
Files: `_diag_compression_mechanism.py` (292), `_diag_compression_sweep_figure.py`,
`_diag_compression_sweep_table.py`, `_diag_compression_variants_smoke.py`,
`_sweep_compression_fixes.py` (237), `_sweep_target_reformulation.py` (308), `_diag_target_dist_v1v2.py`,
`_diag_target_reformulation_figure.py`, `_sweep_w0.py` (261), `_w0_paired_deltas.py`,
`_pick_binary_thresholds.py` (204), `_diag_v2_binary_per_image.py`, `_diag_v2_binary_thresholds.py`,
`_summarize_binary_results.py`, `_summarize_modeling_results.py` (210), `_diag_per_image_breakdown.py`,
`_diag_topk_confusion_map.py`.

These produced the compression diagnosis, the `boulder_count` target lift ("+22% dev PR-AUC"), the W0
promotion decisions ("P2 promoted, single-stage rejected", "dev wins die at LOIO"), and the binary
thresholds. Audit: **`_pick_binary_thresholds.py`** — was any threshold chosen on the test fold
(a leakage channel the `leakage` area flagged as unchecked)? **`_sweep_target_reformulation.py`** — is
the cross-target comparison base-rate-corrected (cf. R26 and `notebooks-5`, which found exactly this
defect in the notebook that reports these numbers)? **`_w0_paired_deltas.py`** — are the deltas paired
on the same folds? And does `_summarize_modeling_results.py` read `presence_auc_mean` (R02)?

### `probes-stage7`
Files: `_stage7_check_labels.py`, `_stage7_feasibility.py` (405), `_stage7_inspect.py`,
`_stage7_verdict.py`, `_stage7a_sanity.py`, `_summarise_stage7c.py`, `_verify_stage7c_trio.py`,
`_terrain_classify.py`, `_terrain_join_v2.py`, `_terrain_stats.py`, `_terrain_stats_honest.py`,
`_dump_attribution.py`, `_dump_browse_terrain.py`, `_dump_terrain_excel.py`,
`_compositional_slim_polygons_overlay.py`, `_compositional_slimmer_attribution_bars.py`,
`_fetch_color.py`, `_inspect_terrain_for_evidence.py`.

The compositional programme is PARKED but `docs/compositional.md` is a **reader-facing writeup**, and
two findings already landed here: **R15** (`classify_image` can never return `inconclusive`, yet two docs
report "0 inconclusive" as a result) and **`stats-fallacies-4`** (pooled tests treat spatially
autocorrelated tiles as independent; p-values overstate by ~12 orders of magnitude and the strongest
feature fails an image-level test). Audit `_stage7_feasibility.py` — it produced the **GO** decision for
the whole Stage-7 build; **`notebooks-2`** found the GO statistic lives only in notebook 14 and the
writeup mislabels it. Is `_stage7_feasibility.py` the declared producer that never computes it? Also
note the pair `_terrain_stats.py` / `_terrain_stats_honest.py` — why are there two, which is cited, and
what does "honest" fix?

### `probes-w1-geospatial`
Files: all 24 `_w1_*.py`, plus `_boulder_size_audit.py`, `_crater_distance.py`,
`_diag_boulder_localization.py`, `_diag_boulder_localization_fullres.py`, `_diag_block_shift_field.py`,
`_diag_tocrs_displacement.py`, `_diag_crs_names.py`, `_diag_lbl_center.py`, `_check_decimated_sp1.py`,
`_probe_jp2_crs.py`, `_probe_murray_url_variants.py`, `_probe_pyproj_sp1.py`, `_probe_sp1_regex.py`,
`_verify_sp1_fix.py`, `_diag_vclaire_detections.py`, `_diag_vclaire_geom_validity.py`,
`_diag_vclaire_sizes.py`, `_diag_vclaire_source_nulls.py`.

Largest area (~45 files) — triage hard by citation. Priorities:
**(a) `_diag_vclaire_source_nulls.py` is directly relevant to R23** (independently confirmed: three
exports' null-geometry rows are the entire low-score tail, so two cohort images are labelled at a
0.41/0.62 confidence floor while 36 are at 0.10, and `DECISIONS.md:1194` records the drop as benign
density hygiene). Did this probe look at the score distribution, and if so what did it report? If it
did and the score dependence was visible, the record's "benign" framing is worse than an oversight.
**(b) `_w1_latitude_distortion.py`** produced the "true min-size floor 0.94 m vs 1.16–1.36, carry as a
known systematic" entry — check that arithmetic; it interacts with **R03**.
**(c) `_w1_migrate_coreg_sign.py` / `_w1_sign_error_check.py` / `_w1_shift_rescore.py`** relate to the
coregistration **sign error** that made v2 labels ~360 m south (a real historical bug) — verify the fix
was complete and that everything re-derived after it.
**(d) the SP1 cluster** (`_probe_sp1_regex`, `_probe_pyproj_sp1`, `_verify_sp1_fix`,
`_check_decimated_sp1`) — cross-check against **`geo-crs-3`** (the SP1 correction is silently skipped
for low-latitude images and the tolerance is in degrees of latitude while the ground error scales with
longitude distance from the 180° meridian).
**(e) `_w1_check4_presence_and_check1_deadfeat.py`** — "presence" in the name; check invariant 8, and
whether the dead-feature audit it ran covers the canny family (**R28** found it does not).

### `probes-fbuild`
Files: `_f02_diagnose.py`, `_f_edr_url_verify.py`, `_f_leg_b_blur_check.py`, `_f_leg_b_crop_stats.py`,
`_f_leg_b_diag.py`, `_f_leg_b_fetch_true_incidence.py`, `_f_leg_b_figures.py`,
`_f_leg_b_incidence_check.py`, `_f_leg_b_mapping_compare.py`, `_f_leg_b_pds_incidence.py`,
`_f_leg_b_quant_check.py`, `_f_leg_b_uint8_contrast.py`, `_f_leg_b_variant_summary.py`,
`_f_litreview_queries.py`, `_f_litreview_queries2.py`, `_f_litreview_verify.py`, `_f_pilot_bounds.py`,
`_f_review_overlap_residual.py`, `_f_seammap_probe.py`, `_inspect_seammap_E12_N44.py`.

The F programme is closed and hard-aborted, so severity is mostly *record correctness* — but the
register already contains four findings showing the F record has real errors (**R11**, **R33**, **R34**,
**R36**), so this area is about completing that picture. Priorities: **`_f_review_overlap_residual.py`**
produced the 2026-07-05d amended verdict ("post-minnaert overlaps agree to 4%, not 10% — the embedder is
the real floor") that **opened the entire H1–H6 docket**; audit it. **`_f_leg_b_uint8_contrast.py`**
produced the native-window IQR range 19–57 that **R07** relies on. **`_f_leg_b_pds_incidence.py` /
`_f_leg_b_incidence_check.py` / `_f_leg_b_fetch_true_incidence.py`** — DECISIONS records a "SeamMap
incidence typo" and a bogus-incidence run; check the corrected values propagated everywhere.
`_f_litreview_*.py` are literature searches — check any DOI/citation claim they produced that reached a
doc (project rule: hyperlink every citation to its canonical DOI).

### `probes-utility`
Files: `_evidence_basis_figure.py`, `_evidence_gapfill_map.py` (248), `_evidence_prediction_gallery.py`,
`_evidence_select_exemplars.py`, `_evidence_tier2_map.py`, `_evidence_tier2_map_calibrated.py`,
`_modeling_slim_boulders_overlay.py`, `_modeling_slim_figures.py`, `_modeling_slim_panels.py` (188),
`_modeling_slim_resolution_gap.py`, `_add_marker_cells.py`, `_check_modeling_deps.py`,
`_check_per_bin_rmse.py`, `_check_prediction_range.py`, `_diag_extract_nb13_results.py`,
`_diag_fallback_explore.py`, `_diag_missing_label_tiles.py`, `_diag_nb13_correlations.py`,
`_diag_read_mapping_xlsx.py`, `_diag_torch_import.py`, `_diag_torch_via_kmp_env.py`,
`_diag_torch_via_package_init.py`, `_extract_allowlist_candidates.py`, `_extract_pdf_hirise_ids.py`,
`_fetch_cumindex.py`, `_fetch_missing_labels.py`, `_inspect_footprint_shape.py`,
`_inspect_nb15_outputs.py`, `_inspect_stage4_nb_output.py`, `_inspect_stage4_ti.py`,
`_setup_dev_dataset.py`, `_show_feature_columns.py`, `_smoke_cnn_one_fold.py`,
`_smoke_cnn_v2_one_fold.py`, `_smoke_gbm_one_fold.py`, `_smoke_loio_one_variant.py`,
`_smoke_modeling_loaders.py`, `_smoke_stage4b.py`, `_smoke_stage5.py`, `_summarize_stage4b.py`.

Mostly figure generation and smoke tests — expect a thin yield, and say so if that is what you find.
Priorities: the `_evidence_*` and `_modeling_slim_*` probes **produce figures embedded in published
writeups** (`docs/model_evidence.md`, `docs/modeling_slim.md`, `docs/classification_slimmer.pdf`), so
check they read the artifact they claim to (**`notebooks-1`** found notebook 10 resolves model
directories by *most-recently-modified*, so three "v1 baseline" figures are actually v2 — do these
probes have the same glob bug?). `_check_per_bin_rmse.py` and `_check_prediction_range.py` are
assertions-as-scripts: do they still pass, and does anything run them? `_setup_dev_dataset.py` builds
`dataset_v2_dev` / the `cache_v2_dev` symlink, which no reviewer has examined.
