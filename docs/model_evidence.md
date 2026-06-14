# Model Evidence — CTX boulder-abundance from foundation-model embeddings


> **DRAFT SKELETON.** Structure + figure list locked; prose and final numbers to
> be filled. Headline numbers marked `[held-out: pending]` await the §2.3
> confirmation read on the expansion cohort; all other numbers are group-aware
> LOIO on the current 38-image v2 set. Companion to
> [classification_slimmer.md](classification_slimmer.md) (Part 1, the Tier-1
> handcrafted detector this supersedes).

## 0. Headline

_One paragraph: a frozen Mars-pretrained vision foundation model turns 5 m/px CTX
texture into a boulder-rich/poor call at 160 m, near-globally — substantially
better than the handcrafted-feature detector, at a 4× finer resolution._

| recipe (held-out CV) | resolution | pooled PR-AUC | prec@5% | med per-image AUC |
|---|---|---|---|---|
| **FM recipe (this work)** | 160 m (S=32) | **0.7832** | **0.948** | **0.7865** |
| Tier-1 handcrafted (Part 1) | 320 m (S=64) | 0.5651 | 0.771 | 0.681 |
| held-out confirmation | — | `[held-out: pending]` | `[pending]` | `[pending]` |

<!-- money figure: the truth-vs-model map for one strong image + the reliability overlay -->

## 1. The question, and the honest way to answer it

- What is claimed: a per-tile boulder **rich/poor** call (`fractional_area > 1e-2`)
  that holds on terrain the model never trained on.
- Why **group-aware leave-image-out** CV is the honest test (tiles within an image
  are spatially correlated; random tile splits leak background) — and why dev-set
  numbers are not reported.
- The two honest qualifiers, stated up front: the recipe was *selected* on these
  38 images (selection caveat → the pre-registered confirmation removes it), and
  the foundation model was pretrained on the CTX mosaic itself (transductive →
  §6).

## 2. Reading the numbers (plain language)

_Only the metrics the project actually uses — each with a one-line operational meaning._

- **pooled PR-AUC** vs the ~0.35 base rate: what 0.78 means against a chance line
  of 0.35.
- **precision@5%**: the top 5% of map tiles by score are ~95% truly boulder-rich —
  the operational "where do I look first" number.
- **per-image AUC** and its **±0.1–0.2 fold-ripple error bars** (carry n_pos/n_neg);
  why the median across images is the summary.
- the **rich/poor threshold** (`fractional_area > 1e-2`) and why that is the
  scientifically meaningful cut.

## 3. The model

_The frozen recipe, explained for a reader who hasn't followed the build._

- **Inputs:** a 160 m (S=32) CTX tile plus its 3×3-tile neighbourhood (96 px),
  nothing else — no handcrafted texture features at map time.
- **The foundation model:** a ViT-B/16 self-supervised (MAE + DINO) on 3.9M crops
  of the Murray Lab CTX global mosaic
  ([Fang et al. 2026](https://doi.org/10.1029/2025JH000827); weights
  [Zenodo 18180801](https://doi.org/10.5281/zenodo.18180801)) — the *same* imagery
  product we predict on. Frozen; we never fine-tune it.
- **The embedding:** GeM(p=3) pooling of the patch tokens → one 768-dim vector per
  tile (why GeM over mean/CLS: it weights the few boulder-bearing patches without
  collapsing to a single max).
- **The head:** a small 3-seed MLP ensemble on the 768-dim vector → rich/poor
  probability. Deterministic-modulo-seed; no fusion, no second model.
- **Why it works:** the binding constraint was *representation*, not the model —
  handcrafted CTX texture hit a feature-set floor; the FM supplies a representation
  that clears it (the embeddings carry the signal alone). _(1-line, link supplement.)_

<!-- figure: schematic — CTX window -> ViT -> GeM -> 768-d -> MLP -> rich/poor -->

## 4. What the predictions look like

_The visual proof — truth vs model, reusing [notebook 20](../notebooks/20_fang_vit_probe.ipynb)._

- **Truth-vs-model maps** at matched budget for 2–3 images, including the
  formerly-failing classes: the old anti-signal exemplar **ESP_046328_2180**
  (slim 0.344 → FM ~0.79) and the azimuth outlier **ESP_076499_1160** (FM the
  cohort's biggest single-image win).
- **Top-k tile strips:** the highest-scoring tiles shown as CTX chips with their
  true labels — the FM's top picks are nearly all true positives.

<!-- figures: 19_w2_fang_* + the notebook-20 truth-vs-model panels -->

## 5. What a map user gets

- The deliverable: a near-global **rich/poor** boulder map at 160 m, with a
  per-tile reliability flag.
- **Reliability overlay** (§2.7): an embedding-space novelty score that flags tiles
  far from anything the model trained on — *where on the map to trust it*.
- Operational reading: prec@5% as the "first places to look" guarantee; the 4×
  resolution gain over the Part-1 320 m map.

## 6. Honest caveats

- **Transductive pretraining:** the FM saw test *pixels* (never labels) during
  self-supervision. Why this is acceptable for the deployment estimand
  (Murray-mosaic inference is in-corpus everywhere) + the optional disjoint-corpus
  (MOMO) bound.
- **Confirmation status:** the held-out numbers are pre-registered and pending the
  expansion cohort (§2.3 confirm-then-absorb); the LOIO numbers carry the selection
  caveat until then.
- **Label noise:** BoulderNet detection limits propagate into the labels.

## 7. Calibrated abundance (Tier-2)

Beyond rich/poor, can we predict *how much*? A single-stage 3-seed MLP regressor
on the same frozen emb-only S=32 features (no hurdle/two-stage needed — it was
tested and dropped) gives, LOIO over the 38 images (DECISIONS.md 2026-06-13):

- **Rank skill** — per-image Spearman ρ ≈ **0.43** (fractional area), **~2× the
  handcrafted-feature baseline** (0.22). Magnitude ranking from 5 m/px imagery is
  intrinsically hard (label noise + meter-scale signal), so this is moderate in
  absolute terms but a large relative gain.
- **Rich/poor for free** — the regressor's rich/poor `meaningful_auc` is **0.78**,
  matching the dedicated classifier (§0). So you get a continuous abundance value
  *and* classifier-level rich/poor detection from one model.
- **Top-tile ranking** — NDCG@5% (ranking quality normalized against the ideal
  ordering, so the label-distribution ceiling is built in) is **0.50** vs 0.35 for
  handcrafted.

**Honest limit (tested, not assumed):** we checked whether zero-inflation caps the
Spearman — it does not (only ~16% of tiles are exactly zero at this scale, and
removing them changes ρ by ~0.01). The wall is the intrinsic difficulty of ranking
magnitude among boulder-bearing tiles. The one real caveat is **dynamic-range
compression**: the model under-predicts the high-abundance tail by ~30% (and the
FM compresses *less* than handcrafted), so the map's *ordering* is reliable but its
absolute high-end *values* are squashed — a calibration layer is future work.

<!-- figure: calibration curve (mean_pred vs mean_true per abundance bin) -->

_Status: Tier-2 candidate identified, not yet frozen/productized — the deployable
head + calibration come with the map pilot._

---

## Supplement S1 — How we got here (the trajectory)

_For readers who want the journey and the dead-ends ruled out; not needed to
trust the result._

| step | held-out skill | what it ruled out |
|---|---|---|
| slim 5-feature (shadow+roughness) | per-image AUC med ~0.57 | minimal handcrafted floor |
| Tier-1 handcrafted (52 feat.) | pooled 0.5651 / med 0.681 | the handcrafted ceiling |
| CNN / conditional-leveler fusion (W2) | F1 0.5955 | learned CTX features ≈ handcrafted (representation floor is real) |
| **FM recipe** | **0.7832 / 0.7865** | the floor was a *feature-set* floor, not a sensor floor |

- one paragraph each on the key negative results (augmentation refuted, fusion
  obsolete, head bake-off: trees are the wrong reader) — links to DECISIONS.md.

## Supplement S2 — Recipe spec & reproduction

- exact recipe (head/pool/scale/target/features), checkpoint provenance, the
  `src/fm_embeddings.py` inference path, and the commands to reproduce.
