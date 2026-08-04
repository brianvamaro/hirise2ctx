# Review area: features-deep

- **Reviewed at commit:** da884c7 (clean tree apart from the untracked review docs)
- **Date:** 2026-08-03
- **Verification:** self-refuted (single-agent pass; not independently verified). Every number below
  was measured read-only from committed/derived artifacts (`dataset*/features*/**.parquet|json`,
  `dataset_v2/packaged/**`, `reports/figures/*.csv`, `models/_sweep_*/**`, and one cached SeamMap
  vector file) with small pandas/geopandas snippets. No notebook, sweep, training, map build, ISIS
  step or network fetch was run; no CTX/HiRISE imagery pixels were decoded.
- **Scope note:** this is the *second* pass on `src/features.py` / `src/spatial_features.py` /
  `src/colour.py` / `src/ctx_source_illumination.py`. `features-1 … features-6` (→ **R27**, **R28**)
  and everything in [features.md](features.md)'s *Refuted* / *Verified clean* lists are **not**
  re-filed. Two findings below (`features-deep-2`, `features-deep-4`) materially extend register
  entries (**R04**, and the question `probes-stage6` explicitly punted to this area); I say so
  explicitly in each.

## Findings

### features-deep-1 — The per-source-frame nuisance variance of the hand-crafted features was never measured; it is 2–2.5× the embedding η² that launched the whole striping/F programme, and it is concentrated exactly in the families whose code uses *absolute* DN
- **Severity:** high
- **Liveness:** dead-closed for the shipped map (the frozen recipe is emb-only) · **live** for the Tier-1 reference classifier that the FM headline is measured against, for every W1 attribution, and for the "per-image heterogeneity is the binding constraint" / "5 m/px is the ceiling" conclusions
- **Confidence:** high (mechanism, from the code) · medium (the magnitude is confounded by terrain — see self-refutation)
- **Where:** [src/features.py:430-436](../../src/features.py#L430-L436) (`_quantize_for_glcm`),
  [:199-208](../../src/features.py#L199-L208) (`_compute_canny_window`),
  [:13-16](../../src/features.py#L13-L16) (the docstring claim),
  `dataset/DATA_DICTIONARY.md:230-245`; contrast `src/striping.py:320` (`eta2`) + `:310` (`frame_label_map`), which is the machinery that was only ever pointed at the *embedding* path

The striping programme's whole diagnosis is "per-frame radiometry × a fixed-scale reader"
(`regional_map_rectangular_artifact` memory; DECISIONS `:5240-5241` even names "per-frame
PSF/sharpness/haze" as a residual component). Every mitigation it built — A1, H1 log-median
centering, H4 leveling — operates on the **embedding** path. Nobody ever ran the same measurement on
the hand-crafted feature columns, even though `_quantize_for_glcm` bins **absolute** DN with a fixed
width (`256 // levels`) and `_compute_canny_window` uses skimage's absolute default gradient cut
(R28), so both are gain-sensitive *by construction*.

I measured it. Using `ctx_incidence_mean` as an exact per-source-frame key (it is constant inside
each SeamMap polygon), restricted to single-source tiles (`ctx_n_sources == 1`) at S=64, for the 19
of 38 images that contain ≥2 source frames with ≥50 tiles each, the fraction of **within-image**
feature variance explained by source-frame identity (the same `ss_between / ss_total` as
`src/striping.py:320`) has median:

| feature | frame η² | | feature | frame η² |
|---|---:|---|---|---:|
| `glcm_dissimilarity_d1` | **0.460** | | `grad_mag_mean` | 0.106 |
| `glcm_homogeneity_d1` | **0.456** | | `intensity_std` | 0.077 |
| `glcm_homogeneity_d2` | **0.442** | | `intensity_p50` | 0.068 |
| `glcm_dissimilarity_d2` | **0.429** | | `intensity_mean` | 0.064 |
| `glcm_contrast_d1` | **0.338** | | `bright_cap_fraction` | 0.040 |
| `edge_density` | **0.330** | | `shadow_fraction` | 0.030 |
| `glcm_homogeneity_d3` | 0.321 | | `intensity_subtile_var` | 0.018 |
| `lbp_hist_1` | 0.221 | | `lacunarity_shadow_b2` | 0.006 |

For scale: the F programme's gates were fought over composite η² of **0.196** (mosaic raw) →
**0.141** (A1) → **0.081** (H1) (`DECISIONS.md:5201-5202`). The fine-scale GLCM columns carry 2–2.5×
the *worst* of those numbers, on the input side, unmitigated.

The ordering is the tell, and it is not what a terrain explanation predicts. The families the code
computes on absolute DN (GLCM at d=1, Canny) are at the top; the families the code computes against a
**per-image** reference (shadow/strict/bright cuts, lacunarity on the shadow mask) are at the bottom,
0.006–0.040 — even though `shadow_fraction` is the project's *strongest* terrain/boulder signal
(`docs/methods.md:1016-1023`), i.e. the family with the most real geology in it shows the least
frame structure. Pooling all 40 frame-level points (per-image z-scored frame means), the frame's
feature value tracks the frame's own DN contrast:

| frame-mean feature | Spearman vs frame-mean `intensity_std` |
|---|---:|
| `grad_mag_mean` | **+0.730** |
| `edge_density` | **+0.671** |
| `glcm_dissimilarity_d1` | **+0.573** |
| `glcm_homogeneity_d1` | **−0.551** |
| `glcm_contrast_d1` | **+0.540** |

which is R28's ρ = 0.894 (per-*image* `edge_density` vs `intensity_std`) reproduced one level down, at
the source-frame scale, and extended to the GLCM family that R28 did not cover.

Finally, `src/features.py:13-16` asserts "CTX pixels are NEVER spatially downsampled. The only
information-discarding step is GLCM intensity quantization" — false three times over
(`_compute_gradient_window` applies `gaussian_filter(sigma=1.0)`; `canny` applies its own σ=1;
`_compute_lbp_window` collapses to 10 labels; the shadow/bright masks binarise) — and neither the
docstring nor `DATA_DICTIONARY.md:230-245` records that GLCM's bins are *absolute DN* and therefore
gain-carrying. The dictionary documents that quantisation is scale-dependent (`:233`) but not that
`glcm_contrast_d1` is denominated in (fixed-width DN bin)², so the same physical texture yields a
value that scales with the frame's stretch.

- **Failure scenario:** any revival of the feature-based head (the Stage-6a "artifacts kept" path,
  `DECISIONS.md:2514`; the Tier-1 reference classifier; a feature+embedding fusion) inherits an input
  whose fine-scale texture columns are ~35–46 % per-source-frame variance, with no mitigation and no
  measurement. Concretely: the FM-vs-Tier-1 margin (`0.5651 → 0.7637` pooled PR-AUC,
  `DECISIONS.md:3149`) is a comparison against a baseline whose inputs carry a large, *fixable*
  nuisance axis that was never fixed, and the conclusions "per-image heterogeneity is the binding
  constraint" (`DECISIONS.md:2472-2476`) and "the ~0.43 per-image ceiling is the 5 m/px CTX floor"
  (**R55**) were reached without that measurement. A mundane cause was left unexamined while the
  programme chased the embedding path.
- **Evidence:**
  ```
  src/features.py:430-436
      def _quantize_for_glcm(arr: np.ndarray, levels: int) -> np.ndarray:
          """Linearly quantize uint8 [0, 255] -> [0, levels-1] for GLCM."""
          bin_width = 256 // levels
          out = (arr // bin_width).astype(np.uint8)          # <- ABSOLUTE DN bins

  src/features.py:13-16
      - **Resolution-preservation**: CTX pixels are NEVER spatially downsampled. The only
        information-discarding step is GLCM intensity quantization, and even that is
        scale-aware ...

  dataset/DATA_DICTIONARY.md:240
      | `glcm_contrast_d{1,2,3}` | float | GLCM contrast `Σ_{i,j} (i-j)² P(i,j)` — angle-averaged; ...

  src/striping.py:320-336   (the eta2 that was never pointed at the features)
      def eta2(values, labels, finite) -> float:
          """Fraction of variance of ``values`` explained by group ``labels`` (between/total).
          The spatial analogue: how much of the abundance variance is organised *between* CTX
          source frames."""
  ```
- **Self-refutation attempted:** (a) **The terrain confound is real and I state it**: different source
  frames cover different ground inside a window, so part of every η² above is geology. Three things
  keep the finding alive: the *ordering* (absolute-DN families top, per-image-adaptive families
  bottom, with the strongest geological feature at the bottom — the opposite of a terrain
  explanation); the frame-level ρ = +0.54…+0.73 against the frame's own `intensity_std`; and the code,
  which is decisive on the mechanism regardless of the magnitude. (b) **Is it a different statistic
  from the programme's η²?** Yes, and I flag it: the gated 0.196/0.141/0.081 are η² on the *predicted*
  abundance composite, mine is on the *input* feature. They are not interchangeable — but
  `DECISIONS.md:5240-5241` measures the amplification as 5–20× (a 1–4 % input residual → ~20 %
  prediction difference), so a 35–46 % input variance share is not a smaller problem than the one the
  programme gated on. (c) **Was it already treated?** `DECISIONS.md:2809-2848` ("Bet 1, per-image
  feature standardization") tried per-**image** z-score/rank/robust standardisation and it FAILED all
  four cells — that treats *between-image* level, not *within-image between-frame* structure, and
  cannot remove it. Grepping `DECISIONS.md` for `per-frame` ∧ {feature, handcraft, glcm, shadow}
  returns exactly one hit (`:5241`), in the embedding context. No per-frame feature normalisation was
  ever tried or measured. (d) **Does a test pin it?** No; `tests/test_features.py` exercises the GLCM
  arithmetic on synthetic tiles, never the radiometric invariance. (e) `features.md` verified
  `_quantize_for_glcm` is *arithmetically* exact (8/16/32 divide 256) — true, and orthogonal: exact
  arithmetic on the wrong scale.
- **Fix:** (i) record the mechanism in `DATA_DICTIONARY.md` and drop the false "only
  information-discarding step" sentence; (ii) before any feature-based head is revived, run
  `st.eta2` with `st.frame_label_map` over the feature rasters and report it beside the embedding η²
  — the machinery already exists; (iii) the cheap treatments are per-frame DN normalisation of the
  window *before* the per-image artifacts are computed (that is exactly
  `src.striping.a1_normalize_per_frame`, already written), and/or quantising GLCM on per-tile
  percentile bins instead of fixed absolute bins, plus `use_quantiles=True` for canny (R28's fix).

---

### features-deep-2 — Both Stage-6 derived feature caches and their packaged splits are two generations stale: they carry the pre-DN-clip-fix dead shadow features and pre-coreg-sign-fix labels, the built-in staleness detector mismatches 38/38, and nothing validates it
- **Severity:** medium
- **Liveness:** live-on-disk hazard on a path `DECISIONS.md:2514` explicitly keeps open ("Artifacts kept"); the numbers it already fed are closed
- **Confidence:** high (measured; byte hashes + value diffs + label diffs)
- **Where:** [scripts/run_stage6a.py:86](../../scripts/run_stage6a.py#L86) and
  [scripts/run_stage6b.py:135](../../scripts/run_stage6b.py#L135) (write `source_sha256_short`);
  no reader anywhere (`grep -rn source_sha256_short` hits only those two writers);
  artifacts `dataset_v2/features_nbr_s5/`, `dataset_v2/features_ctx_illum/`,
  `dataset_v2/packaged/loio_nfold_nbr_s5/`, `dataset_v2/packaged/loio_nfold_ctx_illum/`

Both Stage-6 drivers hash their input parquet into the sidecar — a staleness detector. I recomputed
the hash of every current `dataset_v2/features/{obs}.parquet` and compared:
**38/38 mismatch for `features_nbr_s5`, 38/38 for `features_ctx_illum`.** Nothing reads the field, so
the mismatch is invisible at read time.

Two substantive consequences, both measured:

1. **The DN-clip shadow fix never reached them.** Diffing every base feature column between
   `dataset_v2/features/{obs}.parquet` and `features_nbr_s5/{obs}.parquet` over all 38 images:
   36 images differ in **0** columns; the two that differ are exactly the DN-clip-fix images, in
   exactly the 5 shadow-family columns.

   | image | rows | `shadow_fraction` differing | `bright_cap_fraction` differing | `lacunarity_shadow_b{2,4}` differing (max Δ) |
   |---|---:|---:|---:|---:|
   | `ESP_046328_2180` | 41,190 | 32,161 (78 %) | 36,861 (89 %) | 2,115 (1984 / 930) |
   | `ESP_064510_2260` | 98,489 | 43,685 (44 %) | 80,946 (82 %) | 3,474 (3969 / 3721) |

   In the stale caches those two images still have `shadow_fraction ≡ 0`, `shadow_fraction_strict ≡ 0`
   and `lacunarity_shadow_b2 ≡ 0` on **100 %** of S≥32 tiles, with `bright_cap_fraction ≥ 0.999` on
   87 % / 74 % — i.e. the exact regression `DECISIONS.md:2772-2790` records as fixed and worth
   **+0.249 / +0.127** meaningful AUC. (This also means R27's lacunarity sentinel is at 100 %, not
   21 %, for those two images in the promoted `features_nbr_s5` cache.)
2. **Both packaged splits carry pre-coreg-sign-fix labels.** `dataset_v2/labels/*.json` were written
   2026-06-11T01:19–01:26 (the sign-fix regeneration); `packaged/loio_nfold_nbr_s5` was written
   2026-06-10T22:25 and `packaged/loio_nfold_ctx_illum` 2026-05-31T02:02. Joining their `y_test_*`
   to the current `loio_nfold`'s on `(obs_id, tile_size_px, ti, tj)` at S=32 (161,005 rows matched,
   both schemes identical): **`fractional_area` differs on 141,916 rows (88.1 %, max |Δ| 0.1995)**,
   `boulder_count` on 135,099 (83.9 %, max |Δ| 503), and **`binary_by_area` is *flipped* on 24,140
   rows (15.0 %)**. Tile geometry (`xmin/ymin/xmax/ymax`, `tile_area`) is identical, so this is the
   ~360 m label shift, not a different grid.

- **Failure scenario:** the documented revisit path. `DECISIONS.md:2514` keeps
  `dataset_v2/features_nbr_s5/` + `packaged/loio_nfold_nbr_s5/` specifically so the Stage-6a S=32
  "partial carry" can be reopened. A future session that re-runs that comparison against the
  **current** `loio_nfold` (rewritten 2026-06-11T21:28, post-sign-fix, post-DN-fix) gets a
  three-factor comparison — different labels on 88 % of rows, a flipped positive class on 15 %, and
  two images with dead shadow features on the treatment side only — and nothing warns it. The
  banked Δ (`rho +0.0945 → +0.1665`, `models/_sweep_w0/20260610T223114Z` vs `…223410Z`) is also no
  longer reproducible from the artifacts on disk.
- **Evidence:**
  ```
  scripts/run_stage6a.py:84-86     (and run_stage6b.py:133-135, identical)
      "source_features_parquet": str(in_parquet),
      "source_sha256_short": _file_sha256(in_parquet),

  dataset_v2/features_nbr_s5/ESP_046328_2180.json
      "source_sha256_short": "d592c3c51af4383a"      # recorded
      actual sha256 of dataset_v2/features/ESP_046328_2180.parquet today: 9aaa71f8f8797c64
      "written_at_iso": "2026-06-10T22:20:08"        # Stage 4b re-ran 2026-06-11T19:05

  dataset_v2/features_nbr_s5/ESP_046328_2180.parquet  S=32:  shadow_fraction == 0 on 100.00 %
  dataset_v2/features/ESP_046328_2180.parquet         S=32:  shadow_fraction == 0 on   9.26 %
  ```
- **Self-refutation attempted:** (a) **Is the non-refresh recorded?** Partly, and this is the strongest
  objection: `DECISIONS.md:2871-2873` says "NOTE: the stage-6 side schemes `loio_nfold_ctx_illum` /
  `loio_nfold_nbr_s5` were **NOT** refreshed (built by their own repackage scripts; not in the
  recipe)." But that note is about a *missing join column* (`X_cols 53 -> 55`, the context-patch
  indices) and frames the schemes as merely out-of-recipe; nothing anywhere records that they also
  hold pre-sign-fix labels or the two images' dead shadow features, and the DN-clip entry
  (`:2779`, "only the two affected images were regenerated ... Stage 5 repackaged") does not mention
  the stage-6 schemes at all. So the *fact* of non-refresh is logged; its *content* is not, and the
  detector that would surface it is inert. That is why this is medium, not high. (b) **Was the banked
  Stage-6a comparison itself confounded?** No — I checked the two sweep runs
  (`_sweep_w0/20260610T223114Z` baseline, `…223410Z` nbr_s5, 3 minutes apart, both pre-sign-fix), so
  the 2026-06-10 verdict was internally consistent. The defect is the trap left behind, plus the fact
  that the record's S=32 baseline (`rho +0.0945`) was never re-measured post-fix
  (`_sweep_w0/20260611T215447Z` has no S=32 row). (c) **Does the hash mismatch merely reflect a
  byte-level parquet rewrite?** For 36 of 38 images, yes — which is why I measured values and labels
  rather than resting on the hash. (d) **Are the stale `shadow_fraction` columns load-bearing in any
  probe that reads `features_ctx_illum`?** I checked all seven consumers: `_w1_build_dossier.py:55-58`
  and `_w1_reliability_proxy.py:33-38` read only `ctx_n_sources`/`ctx_dominant_source_fraction` from
  it and take shadow features from the post-fix `dataset_v2/features`; `_stage6c_gate.py:143` uses
  only `ctx_*`; the two azimuth probes only `ctx_subsolar_az_mean`/`ctx_incidence_mean`. So no
  *reported* probe number moves — only the packaged splits do. (e) **Extends R04**: R04 says stale
  packaged splits are undetectable and rates "Impact today: **nil**". That is wrong — 2 of the 4
  packaged v2 schemes on disk are two generations behind, quantified above.
- **Fix:** validate at read time — `run_stage6{a,b}_repackage.py` (and ideally
  `loaders.load_metadata`, per R04) should recompute `source_sha256_short` / compare `split_hash`
  and raise. Then either regenerate the two stage-6 caches and their packaged splits, or move them to
  a `_stale_2026-06-10/` directory with a README naming both drifts, so the "artifacts kept" line in
  `DECISIONS.md:2514` cannot be taken at face value.

---

### features-deep-3 — The shadow/bright detector's zero point is a single per-image constant, but the DN zero point moves by up to 47 DN *between source frames inside one window* — 2.4× the 20 DN offset — and the docstring calls the cut "stable across tiles within an image"
- **Severity:** medium
- **Liveness:** dead-closed for the shipped map; live for every GBM/Tier-1/W1 number off `dataset*/features/` and for the W1 cause attribution
- **Confidence:** high (mechanism + measured spreads) · medium (the variance *share* is small — see below)
- **Where:** [src/features.py:26-31](../../src/features.py#L26-L31) (the design claim),
  [:154-158](../../src/features.py#L154-L158) (one `bincount` per image),
  [:305-307](../../src/features.py#L305-L307) (three absolute masks over the whole window);
  `dataset/DATA_DICTIONARY.md:255-264`

`_compute_dn_thresholds` takes one modal DN per **image** and `_shadow_bright_per_tile` applies
`arr < mode−20`, `arr < mode−35`, `arr > mode+30` across the whole window. A HiRISE-sized CTX window
spans 1–3+ Murray SeamMap source frames (`ctx_source_illumination` docstring: "typically 56 sources
reduce to 5-15 inside one HiRISE-sized window"), and the striping programme's own table
`reports/figures/striping_frame_radiometry.csv` measures the per-frame median-DN spread inside a
single Murray tile at **60–134 DN** (9/9 tiles > 35 DN; median 79 DN). Inside the HiRISE windows
themselves, grouping single-source S=64 tiles by source frame, the spread of frame-median
`intensity_p50` reaches:

| image | frames | between-frame median-DN spread | frame-mean `shadow_fraction` spread ÷ its mean |
|---|---:|---:|---:|
| `ESP_069669_2220` | 2 | **47.3 DN** | 0.40 |
| `ESP_064510_2260` | 2 | **45.3 DN** | — (feature dead pre-fix) |
| `ESP_076499_1160` | 2 | **43.6 DN** | **1.98** |
| `ESP_054000_2255` | 2 | **43.1 DN** | **1.56** |
| `ESP_059421_2170` | 3 | **37.6 DN** | **1.47** |
| `ESP_054134_2265` | 2 | 29.6 DN | **1.83** |

Pooled over all 40 frame-level points, frame-mean `shadow_fraction` vs frame-mean `intensity_p50`
gives Spearman **−0.718** — a darker frame simply puts more of its pixels below the image-wide cut.
So `shadow_fraction` is partly "which source frame is this tile in", and the same physical shadow
depth maps to different feature values in different parts of the same image. `src/features.py:29`
presents the opposite as a design virtue ("Stable across tiles within an image"), and
`DATA_DICTIONARY.md:262-264` documents the arithmetic correctly while labelling the columns
"shadows" / "sunlit boulder tops".

The second half is **threshold starvation the 2026-06-10 audit could not catch**. That audit's
criterion was "identically zero across the entirety of the image" (`DECISIONS.md:2718-2721`), so it
found exactly 2 images. On the current post-fix `dataset_v2/features` at S=64, features that are
*exactly* 0 on most of an image's tiles are far more common:

| image | `shadow_fraction_strict` == 0 | `shadow_fraction` == 0 | `bright_cap_fraction` == 0 | `edge_density` == 0 |
|---|---:|---:|---:|---:|
| `ESP_068402_2240` | **92.7 %** | 65.9 % | 49.3 % | 33.8 % |
| `ESP_055978_2270` | **79.7 %** | 40.2 % | 23.6 % | 1.0 % |
| `ESP_045878_2235` | **76.6 %** | 32.1 % | 44.7 % | 16.9 % |
| `ESP_063429_2240` | **75.6 %** | 40.3 % | 24.2 % | 8.1 % |
| `ESP_076499_1160` | **74.0 %** | 54.8 % | 21.5 % | 2.8 % |
| `ESP_055690_2200` | 13.9 % | 2.3 % | **70.4 %** | 0.1 % |
| `ESP_045139_2270` | 6.5 % | 1.6 % | **68.4 %** | 0.1 % |
| `ESP_076723_2265` | 17.5 % | 0.7 % | **68.7 %** | 0.0 % |

`lacunarity_shadow_b2 == 0` matches `shadow_fraction == 0` to the row in every image, which
independently re-confirms R27's mechanism.

- **Failure scenario:** `ESP_068402_2240`'s `shadow_fraction_strict` is a constant 0 on 93 % of its
  tiles and `edge_density` on 34 % (R28's example image), so three of the four features
  `_w1_build_dossier.py:32` uses for its `texture_rho_med` cause attribution are near-degenerate
  there; its W1 cause is recorded as `ok` on a `texture_rho_med` of 0.298 computed partly over
  constants. More generally, any per-image AUC attributed to `texture_decorrelated` /
  `distribution_shift` (6 of 38 images) may be threshold starvation — a mundane, fixable cause —
  rather than a fundamental one, and the audit that would have caught it tested only the
  all-tiles-zero extreme.
- **Evidence:**
  ```
  src/features.py:26-31
    - **Shadow detector** = DN-mode-derived absolute threshold ... one bincount per image finds
      the modal DN of HiRISE-covered pixels; thresholds are `mode - shadow_offset_dn` (normal) ...
      Stable across tiles within an image;

  src/features.py:305-307
      shadow_mask = (arr < thresholds["shadow"]).astype(np.uint8)
      strict_mask = (arr < thresholds["shadow_strict"]).astype(np.uint8)
      bright_mask = (arr > thresholds["bright"]).astype(np.uint8)

  config.yaml features.shadow_fraction:  shadow_offset_dn: 20, strict_offset_dn: 35,
                                         bright_offset_dn: 30
  reports/figures/striping_frame_radiometry.csv: per-Murray-tile spread of per-frame median DN
      deciles [59.9, 63.9, 79.2, 89.7, 119.7] DN, max 133.8; 9/9 tiles above the 35 DN strict offset
  ```
- **Self-refutation attempted:** (a) **The variance share is small and I say so**: frame η² for
  `shadow_fraction` is only 0.030 median (features-deep-1's table), because the *within*-frame
  variance is the real signal and dominates. The frame dependence shows up in the frame **means**
  (ρ = −0.718, relative spread up to 1.98×), not in the variance share — so this is a level/offset
  defect, not a dominant one, which is why it is medium and not high. (b) **Is the per-image cut a
  recorded deliberate choice?** Yes — `AskUserQuestion 2026-05-23`, cited at `src/features.py:26`.
  The choice is defensible; the *claim* attached to it ("stable across tiles within an image") is what
  the data contradicts, and the striping programme later established the mechanism that breaks it. So
  the finding is a doc-vs-reality defect plus an unmeasured cost, not a reversal of a decision.
  (c) **Is the starvation table just R28 restated?** No: R28 is `edge_density` and its absolute
  *gradient* cut; this is the shadow/bright family and its absolute *DN* cut, a different mechanism,
  different columns, and `shadow_fraction_strict` (74–93 % zeros on 5 images) is the worst case in the
  whole feature set and appears nowhere in the register. (d) **Does DECISIONS already record the
  starvation?** Grep for `DN-clip`/`clip floor`/`shadowfeat` returns only the 2-image episode
  (`:2712-2731`, `:2772-2790`); the "next-lowest whole-image shadow fraction is 2.1 %" line at
  `:2727` is a whole-image statistic and is exactly the criterion that misses a 93 %-of-tiles case.
  (e) **Tests?** `tests/test_features.py:279-291` exercises the *second* percentile fallback only;
  nothing asserts a minimum firing rate.
- **Fix:** derive the three DN cuts **per CTX source frame** rather than per image — the frame labels
  are already available from `ctx_source_illumination.rasterize_seam_map_window` / `striping.SeamMap`
  — or express them as percentiles of each frame's own DN distribution. Cheap interim: record, in the
  Stage-4b sidecar next to `dn_thresholds`, the share of tiles for which each DN-cut feature is
  exactly 0 or exactly 1, so starvation is visible without a bespoke audit; and re-run the
  dead-feature audit with an "exactly 0 on > X % of tiles" criterion instead of "all tiles".

---

### features-deep-4 — Stage 6b propagates a physically impossible illumination geometry from one corrupt SeamMap row to all 115,878 tiles of `ESP_068483_2280`; no angle is range-checked (answers the question `probes-stage6` punted here)
- **Severity:** medium
- **Liveness:** dead-closed (Stage 6b / W2 closed) but the columns are on disk, and one queued work item is motivated by the error
- **Confidence:** high (root cause read out of the cached SeamMap)
- **Where:** [src/ctx_source_illumination.py:185-199](../../src/ctx_source_illumination.py#L185-L199)
  (angle rasterisation — only `np.isfinite(v)` is checked),
  [:220-229](../../src/ctx_source_illumination.py#L220-L229) (the *only* validation in the module),
  [:114-122](../../src/ctx_source_illumination.py#L114-L122) (`load_seam_map` checks column
  *presence*, not values)

`probes-stage6` flagged `ESP_068483_2280`'s `mean_ctx_incidence = 4.276°` as "physically implausible
and worth a look ... owned by the `features` area" and could not check it. I checked it. Reading the
cached SeamMap for the tile the Stage-6b sidecar records (`murray_tile: E-48_N44`, 472 polygons, CRS
`Mars_2015_Ocentric_Equirectangular_clon_0` — identical to the window's, so `features-4`'s CRS
hypothesis is **not** the cause here), exactly **one** polygon intersects the window bbox, and it is
the single row of 472 with `INCIDENCE < 20`:

```
PRODUCT_ID                  INCIDENCE  EMISSION   PHASE   SB_SLR_AZ
P20_008839_2269_XI_46N046W     4.2759    4.3498   4.7088     1.7353     <- the whole image
P21_009406_2268_XI_46N047W    44.7100    1.1000  45.8100   174.2500     <- next polygon, 334 m away
tile summary (n=472):        mean 51.60   3.90    52.47    164.79
```

All four angles are impossible for the scene: the frame is at 46.9°N and Mars's subsolar latitude
never exceeds ±25.2°, so the minimum possible solar incidence there is ≈21.7°; and a sub-solar
azimuth of 1.7° puts the Sun due **north** at 47°N. Every other cohort image is 39.4–64.3° incidence
and 142–229° azimuth (`reports/figures/region_frame_incidence.csv`: 907 regional frames span
37.1–80.8° incidence). So the row is corrupt upstream — and `src/ctx_source_illumination.py` copies
it verbatim into `ctx_incidence_mean` / `_emission_mean` / `_phase_mean` / `_subsolar_az_mean` for all
115,878 tiles of that image. The module's only guard is the `SB_SLR_AZ` *spread* warning at
`:220-229`, which cannot fire because the spread inside this window is 0.

Impact, measured:

- **`_w2_azimuth_spread.py`** was written to answer a direct question ("how consistent is the
  illumination direction the CNN patches see?"). Its shipped answer is cohort azimuth
  **min 1.7° / max 228.6° / sd 30.1°**; with the corrupt row removed it is **min 142.1° / max 228.6° /
  sd 13.1°** — the error more than doubles the reported cohort illumination-direction spread.
  (`DECISIONS.md:2949` quotes the more careful "142-186 deg for 36/38 images", so the log is not
  wrong, but the probe's own headline is.)
- **`_w2_fang_azimuth.py`** hardcodes this image as one of two `AZ_OUTLIERS` and treats it as a real
  illumination outlier. Its Q1 statistics: `dAUC~incidence` ρ = −0.058 shipped vs −0.076 corrected;
  `dAUC~|az−median|` ρ = **+0.160 shipped vs +0.219 corrected** (p 0.344 → 0.201). Both directions
  are toward the null, so the recorded "illumination caveat present-but-harmless" conclusion
  (`DECISIONS.md:3277-3287`) survives — but its Q2 LOO-ridge recovery of `sin_az`/`cos_az` is fitted
  against one garbage target, and `docs/w2_litreview.md:236` proposes azimuth-canonical patch
  orientation on the stated ground that it "mainly protects the 2 azimuth outliers", one of which
  does not exist.
- **Refuted**: the Stage-6b pre-declared H3 test and Stage 6c are barely affected —
  ρ(`pr_auc`, `mean_ctx_incidence`) = +0.059 shipped / +0.097 corrected / +0.119 dropped (FAIL in all
  three), and Stage 6c uses `std_ctx_incidence` = mean(`ctx_incidence_std`), which is ~0 for this
  single-source image regardless of the value. So `probes-stage6`'s "it moves nothing" was
  substantively right for the statistics *it* examined.

- **Failure scenario:** any future per-tile use of the SeamMap angles — the azimuth-canonical
  augmentation in `docs/w2_litreview.md:236`, a photometric correction, an incidence covariate —
  silently applies a 40°-wrong incidence and a 170°-wrong sun azimuth to one whole image, and the
  module raises nothing. The generic version: `load_seam_map` validates that the four angle columns
  *exist* and never that their values are physical, against CLAUDE.md's VERIFY-AT-RUNTIME rule for
  exactly this class of external-data assumption.
- **Evidence:**
  ```
  src/ctx_source_illumination.py:185-190     (the only per-value check is finiteness)
      for _suffix, col in ANGLE_COLUMNS:
          shapes = [
              (g, float(v))
              for g, v in zip(subset.geometry, subset[col])
              if g is not None and not g.is_empty and np.isfinite(v)
          ]

  dataset_v2/features_ctx_illum/ESP_068483_2280.parquet, S=64 (1248 tiles), one unique combo:
      ctx_incidence_mean 4.276 | ctx_emission_mean 4.350 | ctx_phase_mean 4.709
      ctx_subsolar_az_mean 1.735 | ctx_n_sources 1 | ctx_dominant_source_fraction 1.0

  scripts/probes/_w2_fang_azimuth.py:49
      AZ_OUTLIERS = ("ESP_076499_1160", "ESP_068483_2280")
  ```
- **Self-refutation attempted:** (a) **Is it my join that is wrong, not the data?** No — window CRS
  and SeamMap CRS are the same WKT string, the window bbox lies inside `gdf.total_bounds`, the
  offending polygon's own area (5.41e9 m²) is 20× the window (2.59e8 m²) so it genuinely covers it,
  and the next-nearest polygon is 334 m away. The value is in the shapefile. (b) **Could 4.3° be
  real?** Not at 46.9°N: `region_frame_incidence.csv` has subsolar_lat ∈ [−25.45, 25.45] over 907
  frames, so |lat| − 25.45 = 21.5° is a hard floor; and the sub-solar azimuth of 1.7° is impossible
  in the northern hemisphere. (c) **Is the triple internally inconsistent?** No — |i−e| ≤ phase ≤ i+e
  holds, which is precisely why a phase/emission cross-check alone would not catch it and a
  latitude-aware incidence floor is needed. (d) **Already filed?** `features-4` covers the *missing
  CRS check* on the same join and is explicitly a different mechanism (and is refuted as the cause
  here); `probes-stage6-4` covers the `std_ctx_incidence` naming collision; `probes-stage6` records
  this outlier as "noted, not filed" and hands it to this area. (e) **Tests?** No test in
  `tests/test_ctx_source_illumination.py` uses an out-of-range angle.
- **Fix:** range-check the four angle columns in `load_seam_map` and record the result in the Stage-6b
  sidecar: `0 < INCIDENCE < 90` **and** `INCIDENCE >= |center_lat| − 25.5` (the hard Mars geometry
  floor), `0 <= EMISSION < 90`, `|i−e| <= PHASE <= i+e`, and `SB_SLR_AZ` in the hemisphere-appropriate
  half — raise (or drop the polygon and warn) rather than rasterising it. Then correct
  `docs/w2_litreview.md:236` and `_w2_fang_azimuth.py:49` to name one azimuth outlier, not two, and
  re-state `_w2_azimuth_spread.py`'s cohort answer as 142.1–228.6° / sd 13.1°.

---

### features-deep-5 — `docs/methods.md` §7.4 reads five GLCM entries as converging evidence; three of them are one statistic (ρ = 1.0000 between `energy` and `ASM`, −0.9998 between `homogeneity` and `dissimilarity`), and one appears in both the positive and the negative table
- **Severity:** low
- **Liveness:** live — reader-facing (`docs/methods.md` is the document README sends external readers to, cf. **R44**)
- **Confidence:** high (table reproduced exactly from the committed v1 caches)
- **Where:** `docs/methods.md:1016-1030` (the two tables) and `:1037-1041` (the prose reading);
  producers [src/features.py:472-482](../../src/features.py#L472-L482),
  `dataset/DATA_DICTIONARY.md:243,245`

skimage's `graycoprops` defines `energy = sqrt(ASM)`, so `glcm_energy_d{k}` is an exact monotone
function of `glcm_ASM_d{k}` — `DATA_DICTIONARY.md:245` even says so ("energy²") — and
`contrast`/`dissimilarity`/`homogeneity` are near-exact monotone transforms of each other on real
tiles. Measured over the 488,554 finest-scale v1 tiles the table is computed on (I reproduce every
published ρ to 3 dp, and the 643,910-row and 97.9 %-zero figures exactly):

| | contrast_d1 | dissim_d1 | homog_d1 | ASM_d1 | energy_d1 | corr_d1 |
|---|---:|---:|---:|---:|---:|---:|
| contrast_d1 | 1 | **+0.9975** | **−0.9959** | −0.9279 | −0.9279 | −0.048 |
| dissim_d1 | | 1 | **−0.9998** | −0.9296 | −0.9295 | −0.060 |
| ASM_d1 | | | | 1 | **+1.0000** | −0.165 |

So the published pair of tables contains `glcm_contrast_d1 +0.033` and `glcm_dissimilarity_d1 +0.033`
in the *positive* list and `glcm_homogeneity_d1 −0.033` in the *negative* list — the same rank
ordering, sign-flipped (ρ = −0.9959 / −0.9998) — plus `glcm_ASM_d1 −0.024` and `glcm_energy_d1 −0.024`,
which are rank-**identical** (ρ = 1.0000). The prose then reads exactly this as corroboration: "GLCM
homogeneity-type features (energy / ASM / homogeneity / correlation, which all rise on uniform
textures) lead in the negative direction". Of those four, two are one number and a third is the
sign-flip of an entry the same section lists as a *positive* finding; only `glcm_correlation_d1`
(ρ ≤ 0.17 against the others) is independent.

- **Failure scenario:** an external reader (the audience `docs/methods.md` is written for) takes five
  independently-measured GLCM columns pointing the same way as five-fold corroboration of the
  shape-from-shading reading, when the effective evidence is two numbers: one contrast/homogeneity
  axis and one energy/ASM axis. The same redundancy inflates every nominal feature count built on the
  18 GLCM columns (the "206 cols" Stage-6a matrix, `DECISIONS.md:2511`) and splits importance credit
  between exact duplicates in any per-feature ranking.
- **Evidence:**
  ```
  docs/methods.md:1025-1030
      | Top negative Spearman ρ | Value |
      | `bright_cap_fraction` | −0.040 |
      | `glcm_homogeneity_d1` | −0.033 |
      | `glcm_ASM_d1` | −0.024 |
      | `glcm_energy_d1` | −0.024 |
      | `glcm_correlation_d1` | −0.023 |

  docs/methods.md:1037-1041
      ... and GLCM homogeneity-type features (energy / ASM / homogeneity / correlation, which all
      rise on uniform textures) lead in the negative direction.

  measured on dataset/features + dataset/labels, S=8, n=488,554:
      spearman(glcm_energy_d1, glcm_ASM_d1)          = 1.0000   (max |e² − ASM| = 6.8e-3)
      spearman(glcm_dissimilarity_d1, glcm_homogeneity_d1) = −0.9998
  ```
- **Self-refutation attempted:** (a) **Is it flagged anywhere?** `grep -rn "glcm_energy\|glcm_ASM"` over
  `docs/review_2026-07-31/*.md` returns nothing, and the register's `methods.md` items are R44
  (half-migrated to v2) and R59 (size-audit undercount) — neither touches §7.4. (b) **Is the
  redundancy harmless because LightGBM ignores duplicates?** For the model, yes (and I do not file it
  as a modelling defect); the defect is that a reader-facing table and its prose present it as
  independent converging evidence. (c) **Is `energy = sqrt(ASM)` version-dependent?** I verified on
  the installed skimage output in the committed parquets, not from documentation: `e² == ASM` to
  6.8e-3 over 304,428 rows and rank-identical to 4 dp. (d) **Do the published numbers themselves
  reproduce?** Yes — all 13 rows to 3 dp, plus the 643,910 row count and the 97.89 % zero share, so
  this is a reading defect, not an arithmetic one.
- **Fix:** in `docs/methods.md` §7.4, collapse the rank-equivalent entries to one row each with a
  footnote giving the pairwise |ρ| ≥ 0.996, and rewrite the interpretive sentence so the
  positive-table `glcm_contrast_d1` and the negative-table `glcm_homogeneity_d1` are named as one
  observation. Optionally drop `glcm_energy_d*` from the emitted schema (or note in
  `DATA_DICTIONARY.md:243` that it is a redundant monotone transform of `glcm_ASM_d*`).

---

## Why the first pass found little

**The area is not under-reviewed at the code level — it is under-reviewed at the artifact and
semantics level.** The first pass read all four modules in full, and its refuted list is long,
specific and largely correct: I independently re-confirmed the nodata refutation (min DN = 1 over all
labelled tiles), the `energy`/`ASM`/`correlation` NaN-fill analysis, the column-order determinism, the
`_quantize_for_glcm` exactness, the `mosaic_origin_pixels` parity, and the per-scale NaN-padding
schema. Nothing it filed needed retracting. So the low yield was not carelessness.

What it missed is a *class of question it never asked*. Its method was: read each function, ask "is
the arithmetic right, and does it match the docstring?", then measure **how many rows hold a
suspicious constant**. That method finds R27 (21.2 % of rows are `0.0`) and it finds R28 (a threshold
that is not what the config says). It cannot find:

1. **"Is the artifact on disk the one this code would produce today?"** The first pass did check for
   staleness — but it checked `dataset_v2/features/*.json` **sidecars** for the post-fix
   `dn_thresholds`, concluded "all 38 are post-fix ... No reported number is stale", and stopped.
   The stale data is one directory over, in the *derived* caches (`features_nbr_s5`,
   `features_ctx_illum`) and their packaged splits, where the values are pre-fix even though a
   `source_sha256_short` field exists specifically to detect that (38/38 mismatch, never read).
   Checking it costs one `hashlib` loop. This is `features-deep-2`.
2. **"What nuisance variable does this statistic actually track?"** Every empirical check in the first
   pass was a *prevalence-of-a-sentinel* measurement. Not one was a *correlation* measurement — with
   the single exception of R28's ρ = 0.894, which is precisely the finding that generalises. Asking
   the same question of the other families (frame η² 0.46 / 0.34 / 0.33 for
   `glcm_dissimilarity_d1` / `glcm_contrast_d1` / `edge_density`; ρ = −0.718 for
   `shadow_fraction` against frame DN level) is `features-deep-1` and `features-deep-3`, and it is the
   audit the striping programme owed the feature path and never paid. Note the shape of the gap:
   the first pass verified `_quantize_for_glcm` is *arithmetically* exact and moved on — but "exact"
   and "measures what the dictionary says it measures" are different claims, and only the first was
   tested.
3. **"Is the external input physically possible?"** The first pass correctly identified that the
   SeamMap join has no CRS check (`features-4`) and then verified the *outputs* were finite and
   non-zero — which is exactly the check that a 4.3° incidence at 47°N passes. `features-deep-4` is
   what that finding looks like when the plausibility question is asked instead of the finiteness
   question. (Credit where due: the first pass's `features-4` had the right instinct and the wrong
   failure mode; the CRS is in fact identical, and the corruption is upstream.)

Two things the first pass **correctly skipped**, which I re-checked and can close: the cross-scale
GLCM quantisation hazard (`levels_per_scale = {8:8, 16:16, 32:16, 64:32}` makes `glcm_contrast_d1`
incomparable across scales) has **no consumer** — every one of the ~80 model runs on disk is
scale-filtered into its own `scale_S{n}` directory, and `loaders.load_fold`'s `scale_idx=None`
all-scale path is never exercised by any script; and the all-NaN-columns-at-S=8 hazard never reaches
a NaN-intolerant model, because every MLP/ridge run is S=32 or S=64.

**Bottom line:** the code in these four modules is, at the level of "does the arithmetic do what the
function name says", genuinely sound — the first pass established that and I did not overturn it. The
defects live at the two seams it did not look at: between the code and the artifacts it already
wrote, and between the statistic and the physical quantity it is presented as measuring.

## Refuted by my own check

- **GLCM's scale-dependent quantisation makes columns incomparable across scales, and something
  compares them.** Nothing does. `find models -name "scale_S*"` shows every run is scale-filtered;
  `docs/methods.md:1002-1006` even flags the 8-level finest quantisation explicitly; Stage 6a
  aggregates per `(obs_id, scale_idx)`. Documented at `DATA_DICTIONARY.md:233-236`. Latent only.
- **All-NaN feature columns at S=8/S=16 (canny, lacunarity, subtile-var, `glcm_*_d2/d3`) reaching a
  NaN-intolerant head via `FeatureScaler`.** Real in principle; unreachable in practice — all MLP
  variants under `models/fang_probe|fang_tier2|deployable*` are S=32/S=64 only, where every family is
  computed. LightGBM handles the S=8/S=16 NaNs natively.
- **`load_fold(scale_idx=None)` pooling four scales into one X matrix** (mixed GLCM quantisation,
  intensity stats over 64×-different pixel counts, NaN columns). The parameter exists
  (`loaders.py:113-116`) and `Fold.scale_idx` documents "None = all scales concatenated", but no
  script or probe calls it without a scale. Dormant API.
- **`ctx_incidence_std` is a seam-crossing indicator, not an illumination-variance signal.** True —
  the rasterised INCIDENCE field is piecewise constant per polygon, so a single-source tile gets
  4e-6 (float32 round-off) and only seam-straddling tiles get a real value. But the module docstring
  already says so (`:20-21`, "probe within-tile geometry mixing (mosaic-seam-like effects)"), and
  `probes-stage6-4` owns the naming collision downstream. No new defect.
- **`_aggregate_per_tile`'s docstring says tiles "fully outside the window get NaN" while the code
  NaNs *partially* outside ones too** (`in_bounds` requires all four edges inside). Docstring-only;
  0 of 3,564,767 rows are affected (all `ctx_*` values finite), per the first pass's measurement,
  which I re-confirmed.
- **The angle rasterisation drops polygons with a non-finite angle but keeps them in `SOURCE_ID`**
  (`:185-190` filters `np.isfinite(v)`; `:206-210` does not), so `ctx_n_sources` could count a source
  that contributes no angle. Zero rows affected (no cohort row has `ctx_n_sources == 0` or a
  partially-finite angle set).
- **`_GLCM_NAN_FILL` gives `NaN` and `0.0` two different meanings in one column** (NaN = "distance not
  computed at this scale", 0.0 = "computed but non-finite"), and the sidecar records
  `"nan_fill": 0.0` as if it were the only convention. Documentation-only: the first pass proved the
  0.0 branch is effectively dead (skimage returns `correlation = 1` on constant tiles) and I did not
  re-open it.
- **The Stage-6a `nbr_*` stencil leaks across the within-image fold boundary at `buffer_tiles: 0`.**
  It pools *features*, never labels, and §5 already refutes the within-image-CV version of this.
- **`ESP_076499_1160`'s azimuth 228.6° is corrupt too.** No — it is at 63.7 °S, so a
  southern-hemisphere sun azimuth is legitimate; it is a real outlier and the record treats it as one.
- **The Stage-6a S=32 verdict was computed as a confounded pre-fix-vs-post-fix comparison.** No: the
  baseline (`_sweep_w0/20260610T223114Z`) and the nbr_s5 arm (`…223410Z`) ran 3 minutes apart, both
  pre-fix. The confound is a trap for the *next* comparison, not a defect in that one.

## Verified clean

- `_lacunarity_per_tile`'s integral-image algebra, `_subtile_variance_per_tile`'s
  `reshape(n,2,sub,2,sub)` decomposition, `_lbp_hist_per_tile`'s normalisation and `int8` safety,
  `_glcm_per_tile`'s distance↔column mapping — re-read and re-confirmed (as the first pass found).
- `src/spatial_features.py` end to end: the NaN-as-gap semantics, the `count_win > 0` mean guard, the
  `-inf` max sentinel (correct even for features whose valid values are negative, e.g.
  `intensity_kurtosis`, `glcm_correlation`), the `ex2 − mean²` population-variance identity with its
  `np.clip(..., 0, None)` floor, and the positional row-order preservation via
  `df.reset_index(drop=True)` + `groupby(sort=False)`.
- `src/colour.py`'s `lambertian_correct` raises on `cos(i) <= 0` (`:127-128`) — no negative-power or
  divide-by-zero path; `ColorLBL.cos_incidence` and `lambertian_correct` both convert degrees→radians
  exactly once (`np.deg2rad`); `region_means` shares one valid mask across the three bands so the
  ratios are internally consistent. Stage 6b performs no trigonometry at all (it averages degrees),
  so there is no degrees/radians hazard there.
- `ctx_source_illumination.mosaic_origin_pixels` vs `labeling._compute_grid_alignment`: identical
  arithmetic including the `round()` and the `e`-sign convention (re-derived; and I independently
  re-confirmed the 38/38 origin match the first pass measured).
- `_compute_dn_thresholds`'s main path: the `covered > _DN_CLIP_FLOOR` filter and the
  `shadow <= _DN_CLIP_FLOOR` percentile fallback both behave as documented; all 38 v2 sidecars report
  `method: dn_mode_offset` with `mode ∈ [71, 169]`, `shadow ∈ [51, 149]`.
- Feature column-order determinism between train and inference: `loaders._feature_columns` derives
  the list from the *train* frame and indexes the test frame with it (`loaders.py:143-147`), and
  `TILE_KEY_COLUMNS` correctly excludes the string `config_hash` from X via `src/dataset.py`'s
  packaging (the packaged `X_*.parquet` carry 60 numeric+key columns and no `config_hash`).
- `reports/figures/striping_frame_radiometry.csv`, `frame_tile_map.csv` and
  `region_frame_incidence.csv` are internally consistent with each other and with the SeamMap I read
  (subsolar_lat ∈ [−25.45, 25.45] over 907 frames; incidence 37.1–80.8°).
- The `docs/methods.md` §7.4 numbers themselves: all 13 Spearman values reproduce to 3 dp, the
  643,910 row count and the 97.89 % finest-scale zero share reproduce exactly.

## Coverage note

**Read in full (again, independently):** `src/features.py` (872), `src/spatial_features.py` (213),
`src/colour.py` (267), `src/ctx_source_illumination.py` (373), `scripts/run_stage6a.py`,
`scripts/run_stage6b.py`, `src/modeling/loaders.py:1-175`, `scripts/probes/_w1_build_dossier.py`,
`scripts/probes/_w1_reliability_proxy.py:1-60`, `scripts/probes/_w2_azimuth_spread.py`,
`scripts/probes/_w2_fang_azimuth.py:40-110`, `src/striping.py:310-350`,
`dataset/DATA_DICTIONARY.md` §Stage 4b (`:195-333`), `docs/methods.md:1000-1050`,
[features.md](features.md) in full (to avoid re-filing), and the register's §3–§4h + §5.

**Grepped, not read in full:** `DECISIONS.md` by term (`DN-clip`, `clip floor`, `shadowfeat`,
`azimuth`, `features_nbr`, `Artifacts kept`, `per-frame`+feature, `eta2`+feature, `were NOT
refreshed`, `2026-06-1[0-4]` headings — then read `:2449-2530`, `:2712-2760`, `:2772-2848`,
`:2849-2900`, `:5230-5250`); `docs/w2_litreview.md`, `docs/model_evidence.md`,
`scripts/probes/_stage6c_gate.py`, `scripts/run_stage6{a,b}_repackage.py`, `PROMOTION_QUEUE.md`.

**Measurements run (all read-only):** per-image/per-scale saturation of the DN-cut features over all
38 `dataset_v2/features/*.parquet`; per-source-frame η² and frame-mean correlations over all 38
`dataset_v2/features_ctx_illum/*.parquet` (S=32 and S=64, `ctx_n_sources == 1`, frames ≥50 tiles);
sha256 of 38 feature parquets vs both stage-6 sidecars; full column-by-column value diff between
`features/` and `features_nbr_s5/` for all 38 images (3.56M rows); label diff across the four
`dataset_v2/packaged/` schemes at S=32 (161,005 rows); `reports/figures/striping_frame_radiometry.csv`
per-tile frame-DN spreads; `models/_sweep_w0/*/aggregate.parquet` S=32 rows; reproduction of
`docs/methods.md` §7.4 from `dataset/features` + `dataset/labels`; one cached SeamMap read
(`cache_v2/ctx_tiles/E-48_N44.zip`, attributes + geometry only) and the *headers* of one cached CTX
window GeoTIFF (`transform`/`height`/`width`; no pixel read).

**Could NOT check:** (1) whether any reported metric actually moves once the GLCM/canny quantisation
becomes frame-invariant or the DN cuts become per-frame — that needs a re-sweep, out of scope;
(2) the true illumination geometry of `P20_008839_2269_XI_46N046W` (would need the PDS CTX CUMINDEX or
the frame's own label — network); (3) whether the corrupt SeamMap row also affects Murray tiles
outside the v2 cohort (I read only `E-48_N44`); (4) whether re-running Stage 6a/6b on the current
`dataset_v2/features` changes the Stage-6a S=32 Δ — needs a sweep; (5) the two `@pytest.mark.slow`
integration tests in `tests/test_features.py`; (6) whether the per-frame η² above survives a
terrain control (would need a per-frame terrain covariate at tile resolution — the closest available,
`dataset_v2/terrain_classification_v2.parquet`, is per *image*, not per frame).
