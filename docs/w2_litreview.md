# W2 literature review — CNN improvement hypotheses in context (2026-06-11)

Compiled mid-Phase-1 (while the augmentation grid ran), motivated by the
mid-grid diagnostics ([`_w2_midgrid_diag.py`](../scripts/probes/_w2_midgrid_diag.py)):
the no-aug CNN beats the tabular baselines *within* images but mis-levels
images (per-image mean-score ↔ base-rate rank-corr +0.22 vs Tier 1's +0.41),
geometric augmentation damages the (empirically consistent, 142–186°)
sun-azimuth shadow prior, and the `texture_decorrelated` images unexpectedly
respond to raw pixels. Each section ends with the implication for this project.

## 1. Multi-domain learning in planetary ML — Bickel, Mandrake & Doran 2021

[Analyzing multi-domain learning for enhanced rockfall mapping in known and
unknown planetary domains](https://doi.org/10.1016/j.isprsjprs.2021.09.018),
ISPRS J. Photogramm. Remote Sens. 182 (provided by Brian, read in full).

- Mixing Moon+Mars rockfall labels at constant label budget improves home-domain
  AP up to +6% and unknown-domain (Ceres) AP up to +16%; 90% foreign + 10% home
  labels ≥ 100% home labels.
- **Experiment 4 (key for us):** widening the *scale* range alone (down-sampled
  Mars labels) consistently *hurts* (AP 0.32 → 0.25 at 90% down-sampled).
  The multi-domain benefit comes from illumination/background/appearance
  diversity, NOT resolution diversity.
- Their augmentation (flips/rots/up-down-sampling/brightness/contrast) is the
  v1 recipe — but for *resolved object detection* (boulder+track shape), where
  rotation preserves the cue. Our sub-resolution texture task keys on
  illumination-locked shadow statistics, where cell B showed rotation hurts.

**Implications:** (a) keep S=32 and S=64 models separate (validates the §4.2b
design; do NOT pool scales as augmentation); (b) cohort *diversity* (more
latitude bands / azimuths / terrains) is the literature-backed long-term fix
for the distribution_shift class — clever single-cohort tricks are second
order; (c) label economics support pretrain-then-finetune (their 10%-home
result is the detection analog).

## 2. The structural twin: sparse high-res supervision → dense coarse-sensor regression

Earth observation has industrialized exactly our setup (high-quality sparse
reference from one sensor supervising dense prediction on a coarser one):

- [Lang et al., country-wide vegetation height from Sentinel-2](https://doi.org/10.1016/j.rse.2019.111347)
  and the [global 10 m canopy height map](https://langnico.github.io/globalcanopyheight/)
  — **ensemble of CNNs, probabilistic regression head (mean + variance),
  sparse GEDI lidar supervision**, deliverable = global map.
- [THREASURE-Net (2025)](https://arxiv.org/html/2512.11524v1) — joint tree-height
  regression + super-resolution from Sentinel-2 time series supervised by
  airborne lidar.
- [Bias-corrected ICESat-2–GEDI fusion for tropical canopy height](https://www.mdpi.com/2072-4292/17/12/1968)
  — explicit treatment of reference-data bias (our analog: BoulderNet
  detection bias / min_confidence filtering).

**Implications:** (a) the program is sound at planetary scale — this is the
strongest external validation of the whole HiRISE→CTX design; (b) when the
Tier 2 regression head comes, copy the field's choices: **probabilistic head
(predict mean and variance) + small ensemble** — which also serves the W4
reliability/honesty goals directly; (c) reference-bias correction (BoulderNet
score thresholds) is a first-class concern in this literature, supporting the
still-untested `min_confidence` label filter experiment.

## 3. Counting objects below the ground sample distance

- [Rodriguez & Wegner, "Counting the Uncountable: deep semantic density
  estimation from space"](https://arxiv.org/pdf/1809.07091), GCPR 2018
  (**full PDF provided by Brian and read 2026-06-11**) — counts objects
  **1/4–1/3 of a pixel in area** (trees, cars in 10 m Sentinel-2; ratio
  area_object/area_pixel 0.26–0.39) by casting counting as density
  regression; detection is explicitly infeasible, density is not. Validated
  over >200 km² / >1.6 M instances. The structural match to us is closer
  than the abstract suggested:
  - **Ground-truth pipeline is ours exactly:** high-res detector
    (Faster R-CNN on ~1 m imagery) → manual cleansing → Gaussian smoothing
    (σ = K/π, K = resolution ratio) → K×K mean-pool to the coarse grid.
    BoulderNet → tile `boulder_count` is the same construction (our K ≈ 10
    too: 0.5 m HiRISE → 5 m CTX). Their σ = K/π is a concrete default for
    the §5.4 3×3-context *smoothing control*.
  - **Architecture finding (their headline):** a shallow 6-block
    **stride-1, no-downsampling** ResNet beats DeepLab v2/v3 on every
    dataset — "any down-sampling operation inside the network risks losing
    precious details"; they *shrink* the receptive field deliberately
    because context matters less for sub-pixel objects. Direct support for
    SmallCNN over deep pretrained nets, and a concrete capacity-scaling
    direction: widen at stride 1 / drop pooling before GAP, don't deepen.
  - **Multi-task helps:** joint CE (presence segmentation) + MSE (density)
    loss, shared trunk, two 1-conv heads — the CNN analog of our two-stage
    hurdle; supports a joint binary+count head at Tier 2.
  - **Known failure mode = ours:** systematic *underestimation of
    high-density pockets* surrounded by low density (global undercount
    −2.7…−4.6%, MSE ≫ MAE from those outliers) — the literature echo of
    the W0 dynamic-range-compression finding.
  - **Texture-only regime works:** for cars (no spectral signature) RGB
    texture alone gave IoU 0.93; spectral bands only mattered for
    vegetation. Single-band CTX puts us in the cars regime — workable,
    just count-noisier (their car-count error was ~2× the tree error at
    5× the density).
- [Density-map vehicle counting at limited resolution](https://doi.org/10.1016/j.isprsjprs.2022.04.011),
  [regional aggregation layer for coarse-supervision density](https://arxiv.org/pdf/1810.09528).

**Implications:** our `boulder_count` target (P2-promoted) is the planetary
instance of sub-GSD density estimation, with the unusual luxury of dense
HiRISE-derived supervision. The literature's success here raises the prior
that a *regression/density* head can work at 5 m/px even where per-tile
binary classification is marginal. New concrete items: (a) a **stride-1 /
no-pool SmallCNN variant** is the literature-backed capacity experiment;
(b) σ = K/π smoothing for the context-label control; (c) joint
presence+density head when Tier 2 starts.

## 4. Test-time adaptation — the per-image shift treatment

- [AdaBN — Li et al. 2016](https://ar5iv.labs.arxiv.org/html/1603.04779):
  domain identity lives in BatchNorm statistics, task knowledge in weights;
  recompute BN stats on the target domain, no labels, no retraining.
- [Prediction-time BN — Nado et al. 2020](https://arxiv.org/pdf/2006.10963),
  [Test-time BN](https://arxiv.org/pdf/2205.10210),
  [TTN (ICLR 2023)](https://openreview.net/forum?id=EQfeudmWLQ) — refinements
  (interpolating source/target stats; known failure mode = small/non-iid test
  batches).
- [TENT — Wang et al., ICLR 2021](https://arxiv.org/abs/2006.10726): one step
  further; minimize prediction entropy on the test batch, updating only BN
  affine parameters. Source-free, one epoch.

**Implications:** our deployment unit is a whole CTX window (~10³–10⁴ patches
from ONE coherent domain), which is the *best case* for these methods — the
small-batch/non-iid caveat doesn't bind. AdaBN is the architecture-level
analog of the bet-1 zscore rescue (which causally rescued all 3
distribution_shift images but cost raw-feature images); the open question it
answers cheaply: does adapting *normalization only* keep the rescue without
the cost? **Evaluable post-hoc on the already-saved per-fold state_dicts.**

## 5. Photometric/style augmentation beyond brightness jitter

- [FDA — Yang & Soatto, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.pdf):
  swap the **low-frequency amplitude spectrum** of a source image with a
  random target image's; phase carries semantics, amplitude carries "style".
  Training-free image translation; widely replicated.
- [Randomized Histogram Matching — Yaras et al.](https://www.researchgate.net/publication/376348367_Randomized_Histogram_Matching_A_Simple_Augmentation_for_Unsupervised_Domain_Adaptation_in_Overhead_Imagery):
  match each training image's histogram to a random other training image;
  competitive with full UDA pipelines on overhead imagery.
- [Hybrid OBA+histogram matching for cross-domain building segmentation](https://www.mdpi.com/2076-3417/16/1/543),
  [style randomization (Jackson et al. 2018)](https://arxiv.org/abs/1809.05375)
  (cited by Bickel et al. as the alternative they did not pursue).

**Implications:** both FDA and RHM are *cross-image* photometric augmentations
— they expose the net to "this texture, under that image's radiometry", which
is the actual LOIO failure axis. Our cell C jitter (brightness/contrast/gamma/
noise) only explores a ball around each patch's own radiometry. On 64×64
uint8 patches both are a few lines (FFT swap / histogram map) in
`_PatchDataset`. Strong candidates for a follow-up cell; physically safer than
geometric augmentation because they leave shadow *orientation* untouched.

## 6. Mars-specific self-supervised pretraining (Phase 2 upgrade)

- [MOMO — Mars Orbital Model foundation model (2026)](https://arxiv.org/abs/2604.02719):
  first multi-sensor Mars foundation model, pretrained on ~12M samples from
  **HiRISE + CTX + THEMIS** (0.25–100 m/px), per-sensor pretraining merged via
  task arithmetic (Equal Validation Loss checkpoint alignment); evaluated on
  9 Mars-Bench tasks (strongest on segmentation); **weights, pretraining data,
  and code public**.
- [Fang et al. 2026, "A domain-specific vision foundation model for Mars"](https://doi.org/10.1029/2025JH000827)
  (JGR MLC; **full PDF provided by Brian and read 2026-06-11**). The single
  most relevant FM for us:
  - Pretrained on **3.91M crops of the Murray Lab CTX Mosaic V1** — the
    *identical data product* our pipeline windows, so zero sensor/processing
    domain gap. Crops at 512/1024/2048 px resized to 224; strict spatial
    holdout vs DoMars16k.
  - ViT-Base, **single-channel** patch projector (1×16×16→768), MAE 75% mask
    + DINO-style self-distillation refinement. ~690 GPU-h total.
  - Results: DoMars16k k=1 kNN 92.8% (beats ViT-G/DINOv2 at 2× embed dim);
    best on 4/5 MarsBench tasks, largest gains on spatially demanding ones
    (cone detection mAP 0.339, crater segmentation IoU 0.373). EO FMs
    (Prithvi, Clay) score *below* Internet baselines — "remote sensing"
    pretraining is not transferable to Mars; in-domain is.
  - **Weights public: [Zenodo 18180801](https://doi.org/10.5281/zenodo.18180801).**
  - Two self-flagged limitations that bear directly on us: (a) MAE
    reconstruction "subdues very fine ripple detail, signaling a limitation
    for tasks that depend on **sub-pixel roughness**" (§4.1 Fig 3d) — their
    words, our exact task; (b) CBIR failure mode: high-incidence scenes
    produce **shadow-dominated embeddings that match by illumination
    geometry rather than morphology** (§4.4) — the embedding retains shadow
    texture (our signal!) but also embeds illumination shift (our disease).
  - Scale caveat: pretraining crops span 2.5–10 km scenes; our 64-px tiles
    (320 m) are below the smallest pretraining crop. The 3×3-context patch
    (960 m) or larger windows sit closer to its input distribution.
- [Purohit et al., "Investigating the benefits of foundation models for Mars"](https://openreview.net/pdf?id=u6IvidncOj):
  pretraining on ~1M raw CTX patches **beats ImageNet pretraining** for Mars
  tasks — independent confirmation that in-domain SSL pays at CTX scale.
- [Mars-Bench (NeurIPS-adjacent 2025)](https://arxiv.org/abs/2510.24010)
  ([repo](https://github.com/kerner-lab/Mars-Bench), [site](https://mars-bench.github.io/)):
  20 standardized Mars datasets (classification/segmentation/detection,
  orbital + rover) **including boulder tasks**; baselines = natural-image,
  EO-pretrained, and VLM models; finding: Mars-specific FMs likely beat
  general-domain ones. Data on Zenodo/HuggingFace. TODO: check what its
  boulder dataset labels are (sensor/resolution) — possible auxiliary
  supervision or eval set.
- [Mars terrain segmentation with less labels (2022)](https://arxiv.org/pdf/2202.00791).

**Implications:** Phase 2 item 1 ("SimCLR/MAE-small from scratch") is
re-specced: **frozen-embedding probe of public CTX-pretrained backbones
first** — Fang et al.'s weights are on Zenodo and were pretrained on our
exact mosaic product (zero domain gap). Concrete first experiment: extract
GeM-pooled ViT-B embeddings for the S=64 tiles (37k tiles ≈ minutes on the
5070), at both 64-px and 192-px (3×3-context, closer to the pretraining
scale) inputs, append as LightGBM feature columns in the standard LOIO
harness. Fine-tuning only if the probe shows signal. Mars-Bench's boulder
tasks may also provide a "foreign domain" label source in the Bickel
sense (§1).

## 7. Thermal rock abundance (validation track, unchanged)

[Nowicki & Christensen 2007](https://doi.org/10.1029/2006JE002798) (TES rock
abundance), [Golombek et al. 2003 rock SFD](https://doi.org/10.1029/2002JE002035),
[InSight-era SFD update](https://doi.org/10.1029/2021EA001959) — the
independent coarse-scale check planned in CLAUDE.md §10 remains the right
external validation once a CTX-scale map exists; thermal "rock" (>1250 TIU,
≥0.1–0.15 m) and BoulderNet boulders (≥1.4 m) measure different but
overlapping populations — a future comparison must mind that definition gap.

## Follow-up queue (updated 2026-06-11 after the free tests ran)

Executed same session:

| item | outcome |
|---|---|
| "CNN ranks / GBM scales" fusion (`_w2_fusion.py`) | **STRONG +**: F1 = cnn_rank × t1_image_mean → pooled PR-AUC **0.5932** (Tier 1 0.5651, CNN raw 0.5095), pooled prec@5% **0.914** (Tier 1 0.771), per-image AUC preserved by construction. Diagnosis (within-image skill + cross-image mis-leveling) confirmed causally. Single-seed caveat. |
| AdaBN post-hoc (`_w2_adabn.py`) | cohort NULL (paired Δ median −0.017, p=0.40; pooled PR-AUC drops) but **class-specific rescue replicates a third time**: ESP_076499_1160 +0.315 — same image rescued by tabular zscore, photometric aug, AND BN re-estimation. Catastrophic losses elsewhere (ESP_055978_2270 −0.448). NEW IDEA from the result: base-vs-adapted **prediction disagreement as an inference-time reliability flag** for the shift class (no labels needed; W1 found no other warning signal for the ~13% silent failures). |
| 3-seed replication of cell A (`sweep_cnn.py` seeds 1, 2) | **gate does NOT replicate per-seed**: median paired ΔAUC vs Tier 1 = +0.066 (p=0.016) / +0.038 (p=0.059) / +0.005 (p=0.66); pooled PR-AUC swings 0.51/0.56/0.49. Per-image medians stay 0.69–0.71 — the *skill* is stable, single-seed *score* calibration is not. |
| 3-seed ensemble + fusion (`_w2_seed_ensemble.py`) | **STRONG +, the W2 candidate recipe**: mean-of-3-seeds alone passes the per-image gate (Δ median +0.052, win 0.70, p=0.0065); **F1(ens) = within-image quantile × Tier-1 image mean → pooled PR-AUC 0.5955 (+0.030, at the pooled gate), prec@5% 0.887**; F3(ens) rank-average → strongest per-image stats (Δ median +0.058, win 0.85, p=0.0001, median AUC 0.714). Caveat: recipe assembled after seeing per-seed results → **S=32 replication is the held-out confirmation**. |

Still queued, in order:

| rank | item | cost | what it tests |
|---|---|---|---|
| 0 | S=32 cell-A 3-seed + ensemble-fusion replication | next GPU job | **held-out confirmation of the candidate recipe** (assembled post-hoc at S=64) |
| 1 | photometric-only cell | running (GPU chain) | de-confound cell C (was it the photometric part that helped the shift class?) |
| 2 | Fang-ViT frozen-embedding probe (64-px + 192-px inputs) → LightGBM | ~1 h | FM representation carries signal beyond handcrafted + SmallCNN |
| 3 | FDA / RHM augmentation cell | 1 cell + ~50 LOC | cross-image radiometry invariance (leaves shadow orientation intact) |
| 4 | azimuth-canonical orientation (rotate patches to fixed sun direction; per-tile azimuth from SeamMap) | small loader change | geometric augmentation without destroying the shadow prior; mainly protects the 2 azimuth outliers |
| 5 | illumination conditioning (ctx_subsolar_az / ctx_incidence scalars into the head) | small model change | geometry × pixels interaction (GBM H3 was null, but the shadowless ESP_068483_2280 case is CNN-specific) |
| 6 | AdaBN-disagreement reliability flag | free probe | inference-time warning for the shift class |
| 7 | TENT (BN affine, entropy min.) | small driver change | adaptation beyond statistics (only if 6 shows promise) |
| 8 | min_confidence=0.5 label filter | Stage-4 re-run | label-noise lever (helps every model or none) |
| 9 | SmallCNN 32-d GAP embedding → LightGBM columns | cheap (reuse state_dicts) | formal feature-level fusion (vs the score-level F1/F3) |
| 10 | probabilistic head + small ensemble (Tier 2, canopy-height style) | Phase 2 | W4 reliability honesty |
| 11 | stride-1 / no-pool SmallCNN variant (Rodriguez & Wegner §3) | 1 model variant | does in-network downsampling discard sub-pixel texture? literature-backed capacity direction |

Cohort diversification (new images at other latitudes/azimuths) remains the
literature's first-order answer (Bickel) but is a data-acquisition decision,
not a modeling one.
