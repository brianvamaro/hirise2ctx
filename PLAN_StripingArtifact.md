# PLAN_StripingArtifact — investigate the rectangular/striping artifact in the regional map

**Created 2026-06-18 (Brian-flagged; next-session first step).** The 26-tile circum-Chryse
abundance map (notebook 24 §2) shows **rectangular / vertical-striping** patterns. This plan is the
focused investigation to characterize the cause, decide a mitigation, and re-test. Context +
ruled-out causes live in [[regional_map_rectangular_artifact]] (memory) and DECISIONS 2026-06-18.

## 0. What we already know (don't re-do)
- **NOT** map_region read-window seams (4096px boundaries don't align with the structure).
- **NOT** per-tile radiometric offsets (per-tile mean abundance is a smooth N→S geology gradient).
- **NOT** HiRISE footprints (overlaid; abundance doesn't trace them — Brian agrees).
- Abundance has a **weak non-monotonic dependence on CTX brightness** (peaks mid-DN, r≈0.07).
- **Leading hypothesis (Brian): CTX mosaic stitching** — the Murray global CTX mosaic is a patchwork
  of individual CTX frames / pushbroom orbital tracks with imperfect radiometric matching; those
  ~rectangular **frame/track seams** are brightness steps the Fang ViT + head key on.

## 1. Characterize the artifact (cheap, no re-inference)
1a. **Geometry/scale** — FFT / directional autocorrelation of one tile's abundance (and prob_raw):
   dominant orientation (vertical = N–S = CTX orbital tracks?) and wavelength (km). Compare prob_raw
   vs calibrated abundance (is it in the raw model output or introduced by qmatch?). → expect raw.
1b. **Co-locate with CTX brightness discontinuities** — edge-detect the coarsened CTX mosaic
   (Sobel/Canny) and the abundance; quantify whether abundance edges coincide with CTX brightness
   edges (vs random). The decisive test of the stitching hypothesis.

## 2. Source the CTX frame provenance (the key unknown)
- Does **Murray Lab** publish a per-pixel **source-frame index / CTX EDR footprint layer** for the
  V01 mosaic? (Check the Murray Lab site + the mosaic's ancillary products.) If yes → overlay the
  actual frame boundaries on the abundance map and test alignment directly (gold-standard test).
- If no provenance layer: fall back to §1b (detect CTX seams from the imagery itself).

## 3. Quantify the model's susceptibility (synthetic, one tile)
- Take CTX patches; apply a **brightness gain/offset step** (simulating a frame seam) and measure
  how much predicted abundance shifts. Establishes how strongly the embedder/head respond to a
  radiometric seam vs true texture. (Ties to the r≈0.07 finding — is it gain, offset, or contrast?)

## 4. Prototype mitigations (on a test tile; only re-run inference there)
Pick based on §1–§3. Candidates, cheapest first:
- **Local contrast normalization / high-pass** on CTX before the Fang embedding (removes large-scale
  brightness/frame steps, keeps boulder-scale texture). Re-embed + predict one tile; does striping
  drop?
- **Per-frame radiometric normalization (destriping)** if §2 gives frame provenance.
- **Per-image/per-track standardization** (the deferred W1 bet) at inference time.
- **Post-hoc notch/high-pass on the output abundance** (cosmetic, least principled — last resort).

## 5. Validate any mitigation against the things that matter (don't just make it look nicer)
A mitigation is only worth adopting if it does **not** cost skill and ideally helps the real tests:
- **LOIO skill preserved/improved** — re-score the held-out CV with the normalized CTX input
  (per-image AUC ≈ 0.43 baseline must not drop).
- **Thermal correlation improves** — if the stripes are noise, removing them should *raise* the
  leg-1/leg-2 Spearman ρ (currently leg-1 ≈ +0.07). This is the objective adjudicator.
- Re-render the regional map; visually confirm the striping is gone without washing out real signal.

## 6. Decision
- If a mitigation clears §5 → adopt it, **re-run the 26-tile regional inference on Sherlock** (same
  job-array kit) and regenerate the validation figures.
- If nothing helps without costing skill → **document as a known 5 m/px CTX-mosaic-floor limitation**
  (the stripes are a data artifact the model can't fully escape), disclose it in the validation
  write-up, and lean on the coarse-scale/thermal results that are robust to it.

## 7. Deliverable
`reports/figures/striping_*` (the characterization + edge-coincidence + before/after) and a
DECISIONS entry with the verdict. Update notebook 24 / the regional map if a mitigation is adopted.
