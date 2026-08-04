# Review area: fm-embeddings

- **Reviewed at commit:** da884c7
- **Date:** 2026-07-31
- **Verification:** self-refuted (single-agent pass; not independently verified)

## Findings

### fm-embeddings-1 — The H2 nuisance basis and the H3 consistency pairs are estimated on a pool that contains every LOIO fold's held-out image, so both "skill Δ" columns are transductive, not deployable
- **Severity:** medium
- **Liveness:** dead-closed (H2/H3 legs of PLAN_StripingArtifact PHASE 2; but the numbers are quoted in `DECISIONS.md` and in a committed CSV)
- **Confidence:** high (measured)
- **Where:** `scripts/f_leg_b_loio.py:135-147` (basis/pairs loaded **once**, outside the fold loop), `:69-87` (`run_store` hands the same objects to every fold), `src/modeling/mlp_head.py:369-405` (`DeployableHead.fit` has no fold-aware hook for either), `scripts/f_h2_nuisance.py:49-57` + `:116` (basis pool = every multi-crop obs), `scripts/f_h3_pairs.py:64` (pairs pool = `h2.obs_frames()`, the same set)

`f_h2_nuisance.py` PCAs co-located frame-difference vectors over **all** obs with ≥2 crops, and `f_h3_pairs.py` draws its 40 000 pairs from the same pool. `f_leg_b_loio.py` then loads that one basis / one pair set before the fold loop and passes it into every fold's `DeployableHead`, so in each fold the operator applied to the held-out image (H2) — or the unlabelled penalty term the head is trained against (H3) — was estimated using that same held-out image's own embeddings. I measured the overlap: **28 of the 36 LOIO-evaluated images contributed to both the basis and the pairs; zero contributing images are outside the evaluated set.** The bias direction is optimistic for the F arms, so the FAIL verdicts stand — but the H2 table's only skill-gate survivor, `k=4` at Δ −0.0026 ("even a hair better than H1", pooled PR +0.039, `DECISIONS.md:4396-4405`), and every `skill_delta` in the committed `reports/figures/f_h3_pareto.csv`, are contaminated numbers presented as LOIO.

- **Failure scenario:** a future session revisits H2 (k=4 is the documented near-miss) or reuses `--nuisance-basis` / `--consistency-pairs` for a *passing* arm. Because the basis/pairs are fit once over the whole cohort, the reported Δ median AUC overstates what a deployable pipeline achieves — the reopening gate (−0.02) would be cleared on a number that cannot be reproduced out of sample. The 0.05 η² bar is unaffected (its test set really is the 7 disjoint pilot frames), so a stack could be declared PASS on a half-honest pair of gates.
- **Evidence:**
  ```
  scripts/f_leg_b_loio.py:135-144
      basis = None
      if args.nuisance_basis and args.nuisance_k > 0:
          basis = np.load(args.nuisance_basis)["basis"][:, :args.nuisance_k]
      ...
      pairs = None
      if args.consistency_pairs and args.lambda_consistency > 0:
          z = np.load(args.consistency_pairs)
          pairs = (z["ea"], z["eb"])

  scripts/f_leg_b_loio.py:75-87   # ... then reused for EVERY fold, unfiltered
      for fold in iter_loio_folds(SCHEME, scale_idx=SCALE_IDX, dataset_dir=DATASET_DIR):
          fold = restrict_fold(fold, avail)
          ...
          head = DeployableHead(recipe=dict(target_id=TARGET),
                                nuisance_basis=nuisance_basis,
                                lambda_consistency=lambda_consistency)
          head.fit(f.X_train, ytr, groups=f.groups_train, obs_to_int=f.obs_to_int,
                   consistency_pairs=consistency_pairs, verbose=False)

  scripts/f_h2_nuisance.py:9-10   # the non-circularity claim, scoped only to the eta2 test set
      Basis source = the 28 multi-crop TRAINING obs, NOT the 7 E8_N44 pilot frames the eta2 test
      scores - so N is learned independently of the artifact-reduction test set (no circularity).
  ```
  Measured (`reports/f_leg_b/h2_nuisance_basis.npz`, `h3_consistency_pairs.npz`, store globs):
  ```
  basis obs 28   pairs obs 28
  baseline store 38   f_minnaert_center store 36   avail (LOIO folds evaluated) 36
  LOIO-evaluated images that ALSO contributed to the basis: 28
  LOIO-evaluated images that ALSO contributed to the H3 pairs: 28
  basis-only (not evaluated): []
  ```
- **Self-refutation attempted:** (a) I checked the disclaimer — `f_h2_nuisance.py:9-10` and `DECISIONS.md:4374-4376` / `:4423-4426` both say "no circularity", but both scope it explicitly to the **η² test set** (the 7 E8_N44 pilot frames, which really are in a different crops dir: `reports/f_timing/pilot_crops` vs `reports/f_leg_b/obs_crops` — verified). Neither entry mentions the skill gate's folds. (b) I checked whether the leak is label-bearing — it is not; both channels are unsupervised, which is why I rank this medium not high. (c) I checked whether the verdicts could flip — they cannot: the bias is optimistic and both arms still FAILED (H2 on η², H3 on the never-overlapping Pareto), so this corrects the record rather than overturning it. (d) I checked whether the basis is even a valid projector (a non-orthonormal `N` would break `_project`): `max|NᵀN − I| = 1.1e-8` over the top 64 columns — clean.
- **Fix:** either (i) recompute the basis / re-subsample the pairs per fold from the training obs only — both are cheap given `reports/f_leg_b/h2_frame_emb/` is already a per-obs cache — or (ii) leave the code and annotate the two DECISIONS tables and `f_h3_pareto.csv` header that the skill column is transductive (basis/pairs fit over all 28 obs incl. each held-out image), with the bias direction stated.

### fm-embeddings-2 — The cross-machine parity gate exercises zero masked tiles and runs a different masking threshold than production, so the shipped map's nodata/NaN path is outside the only numerical guard
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high (measured from the committed reference)
- **Where:** `scripts/parity_check.py:69-70` (no `max_zero_fraction` passed → `predict_window`'s default 0.5), `:72-78` (`keep = np.isfinite(pred.prob)`), `:122-137` (what is gated); `scripts/map_region.py:265-266` (production default **0.3**); `src/mapping.py:229-261`; `src/fm_embeddings.py:203-220`

`parity_ref.npz` is the only thing standing between the laptop and the Sherlock GPU box for the shipped regional map. Its window is a 512² interior block of `E4_N44`; `tile_grid_for_window((512,512), 20000, 20000, 32)` enumerates exactly 14×14 = **196** tiles, and the reference stores **196** tiles — i.e. `keep` dropped nothing, so `own_tile_zero_fraction`, the `usable` mask, the NaN row placement in `embed_window`, and `tiles_to_raster`'s `fill=np.nan` are all never exercised by the gate. Separately, `run_window` never forwards `max_zero_fraction`, so the reference was produced at 0.5 while `map_region.py` runs at 0.3 — the gate validates a configuration the production job does not use. The reference also records no model or calibration identity, which matters here because 11 different heads on disk all live in a directory named `86c51a5dca220f63` (see R09).

- **Failure scenario:** the L-shaped 26-tile block has genuine mosaic nodata (two corners, plus interior swath gaps). Any machine-dependent difference confined to the masking path — a rasterio/GDAL version returning a different nodata representation for a partially-empty window read, a numpy change in `(box == 0).mean()` on an empty slice, a float32-vs-float64 NaN fill — passes `[PASS] faithful quantities … match` and then silently changes which tiles the regional map claims coverage for. `n_masked_nodata` in the sidecar would move with it, and nothing compares it to a reference.
- **Evidence:**
  ```
  scripts/parity_check.py:69-71
      pred = predict_window(window, embedder, head, tile_px=TILE_PX, batch=batch,
                            calibrator=calibrator, apply_isotonic=True)
      keep = np.isfinite(pred.prob)
  scripts/map_region.py:265-266
      ap.add_argument("--max-zero-fraction", type=float, default=0.3,
                      help="mask a tile whose own CTX is more than this share mosaic nodata")
  ```
  Measured from `models/deployable/parity_ref.npz`:
  ```
  ti (196,) int32 range 626..639     tj (196,) int32 range 626..639
  -> 14 x 14 = 196 enumerated tiles, 196 kept  => zero tiles masked in the reference
  prob_raw range 0.0386..0.9970   abundance range 0.00039..0.0944
  ```
- **Self-refutation attempted:** I checked whether a wrong head/checkpoint/calibrator would still be caught — it would (predictions diverge far beyond `atol=2e-3`, and `[FAIL] tile grid (ti,tj) differs` fires on a grid change), so the gate is sound for its stated purpose and the threshold mismatch is *self-consistent* across machines (it only changes which tiles are in the set, identically on both). I also checked whether the isotonic exclusion is a hole — it is documented and correct (`parity_check.py:126-130`; isotonic is a step function, so gating it would produce false alarms). What survives is purely a **coverage** claim: measured, the gate's tile set has no masked tile at all, so an entire branch of the shipped inference path has no cross-machine reference.
- **Fix:** pass `max_zero_fraction=0.3` in `run_window` (or make it a CLI flag defaulted to map_region's value), pick a reference window that straddles a mosaic gap so `keep.sum() < ti.size`, and store `n_valid` / `n_masked_nodata` / the head's `model_hash` in the reference npz and compare them.

### fm-embeddings-3 — Embedding stores carry no build provenance and resume on filename existence, so a store can silently become a mixture of two parameterizations
- **Severity:** low (latent — measured **not** to have fired)
- **Liveness:** live-shipped (the frozen `dataset_v2/fang_embeddings` store is the input to every reported FM number) + dead-closed (the F stores are the abort's input)
- **Confidence:** high
- **Where:** `scripts/probes/_w2_fang_embed.py:238-243` (`np.savez` writes only `ti/tj/valid/cls/mean/gem`), `:280-283` (resume keys on file existence only), `:250-261` (`--norm` and `--out-suffix` are independent flags); `scripts/f_leg_b_embed.py:243-245` (`if out_path.exists(): cached`), `:309-324` (`--stretch-pcts` / `--stretch-scale` / `--crops-dir` / `--store-suffix` all change the pixels without changing the store name)

Nothing in an `{obs}_P{px}.npz` records which pixel mapping produced it (verified: the frozen store's keys are exactly `['cls','gem','mean','ti','tj','valid']`; the F stores' are `['gem','ti','tj','valid']`). Both writers resume on `out_path.exists()`, so a rerun with different constants fills in only the *missing* images and leaves the rest — producing a store that is a mixture of two radiometric parameterizations with no detectable trace. `_w2_fang_embed.py --norm a1` without `--out-suffix _a1` writes A1-normalized embeddings straight into `dataset_v2/fang_embeddings`, the store behind pooled PR-AUC 0.7832. The project already knows the constants live nowhere: `f_h2_nuisance.py:102-105` warns `--stretch-pcts … MUST match the store build (H1 fang_embeddings_f_minnaert_center used 0.5 99.5)` — a hand-maintained invariant.

- **Failure scenario:** a rerun of `f_leg_b_embed.py --mapping minnaert` with different `--stretch-pcts` after a partial failure yields `dataset_v2/fang_embeddings_f_minnaert/` half at one stretch and half at another. `f_leg_b_loio.py --f-store fang_embeddings_f_minnaert` and `train_deployable_head.py --store-name …` both accept it without complaint, and the resulting per-image AUC spread — the exact statistic the striping programme was reading — is then partly an artifact of the mixed store.
- **Evidence:**
  ```
  scripts/probes/_w2_fang_embed.py:280-283
      done = OUT_DIR / f"{obs_id}_P{CONTEXT_PX}.npz"
      if done.exists() and (OUT_DIR / f"{obs_id}_P{TILE_PX}.npz").exists():
          print(f"  {obs_id}: cached, skipping", flush=True)
          continue
  scripts/probes/_w2_fang_embed.py:241-243
      np.savez(OUT_DIR / f"{obs_id}_P{CONTEXT_PX}.npz",
               ti=ti.astype(np.int32), tj=tj.astype(np.int32),
               valid=valid192, **emb192)          # <- no norm / mapping / stretch fields
  scripts/f_leg_b_embed.py:243-245
      out_path = out_dir / f"{obs_id}_P96.npz"
      if out_path.exists():
          print(f"  {obs_id}: cached", flush=True); return True
  ```
- **Self-refutation attempted:** I tried to show it had already happened, by checking whether any store's npz mtimes are split across separate sessions. Every one of the nine stores is a single contiguous run (`fang_embeddings` 38 files over 0.21 h, max internal gap 0.02 h; `fang_embeddings_f_minnaert_center` 36 files over 0.09 h; all others similar) — so **no store on disk is mixed**, which is why this is low and not medium. The hazard is real and undetectable, but the discipline has held.
- **Fix:** write the mapping/norm name, stretch percentiles, stretch scale, crops-dir and `k_minnaert` into every npz (a handful of scalars), have `load_fang_store` assert those fields are identical across the files it concatenates, and key the resume check on them.

### fm-embeddings-4 — `DeployableHead.load` verifies nothing, although the recipe card already carries a `model_hash` that is exactly recomputable at load time
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high (measured)
- **Where:** `src/modeling/mlp_head.py:456-475` (`load`), `:429-433` (`model_hash`), `:443-453` (the card that stores it), `:461` (`basis = np.load(basis_path) if basis_path.exists() else None`); `scripts/train_deployable_head.py:152-158` (the round-trip check prints `MISMATCH` and still `return 0`)

`save` writes `model_hash` into `recipe.json`; `load` never recomputes it. I verified the hash **is** bit-reproducible at load: for all 11 heads under `models/deployable*/`, `DeployableHead.load(d).model_hash()` equals the card's value exactly — so a two-line check would turn every artifact-integrity failure into a loud one, and today there is none. Worse, `nuisance_k` is recorded on the card but never cross-checked against `nuisance_basis.npy`: a **missing** `.npy` silently yields an unprojected head (H2 weights applied to unprojected embeddings), and a **stray** `.npy` silently applies a projection the weights were never trained with. `train_deployable_head.py`'s own round-trip guard also exits 0 on mismatch.

- **Failure scenario:** any head dir transferred with a filter that drops `.npy` (or assembled by copying `seed*/` from a sibling — plausible because all 11 heads share the directory name `86c51a5dca220f63`) loads without a word and predicts confidently wrong probabilities. On Sherlock `parity_check.py` would catch it *if run*, but nothing on the laptop, in `map_pilot.py`, in `f_h2_eta2.py` or in `f_h4_*.py` does.
- **Evidence:**
  ```
  src/modeling/mlp_head.py:459-467
      card = json.loads((path / "recipe.json").read_text(encoding="utf-8"))
      basis_path = path / "nuisance_basis.npy"
      basis = np.load(basis_path) if basis_path.exists() else None      # card["nuisance_k"] ignored
      head = cls(seeds=tuple(card["seeds"]), ...)
      # ... no comparison against card["model_hash"] anywhere below
  scripts/train_deployable_head.py:155-158
      max_diff = float(np.abs(p2 - p[:2048]).max())
      print(f"  save/load round-trip max |dp| = {max_diff:.2e} "
            f"({'OK' if max_diff < 1e-6 else 'MISMATCH'})")
      return 0
  ```
  Measured over all 11 heads (recomputing at load):
  ```
  deployable            mh_match=True rh_match=True nuis_card=None nuis_loaded=None n_train=38
  deployable_f_h2_k4    mh_match=True rh_match=True nuis_card=4    nuis_loaded=4    n_train=36
  deployable_f_h3_lam10 mh_match=True rh_match=True nuis_card=None nuis_loaded=None lam=10.0
  ... (all 11 consistent)
  ```
- **Self-refutation attempted:** (a) Is the hash actually stable enough to check? Yes — measured exact match for all 11, so the check would not produce false alarms. (b) Is a downstream guard already covering it? Only `parity_check.py`, only on the Sherlock leg, only for the mosaic head (R09 notes `deployable_f_center` has no `parity_ref.npz`). (c) Is the documented transfer path safe? `SHERLOCK_RUN.md:102` / `:133-135` tar/rsync the whole directory, so the missing-`.npy` scenario is not the documented workflow — hence low, not medium. (d) Would a mismatched member shape fail loudly anyway? Only if `d_in`/`hidden` differ; same-architecture swaps load silently.
- **Fix:** in `load`, recompute `model_hash()` and raise on mismatch with the card; assert `(card.get("nuisance_k") is None) == (basis is None)` and `basis.shape[1] == card["nuisance_k"]`; make `train_deployable_head.py` return non-zero on `MISMATCH`.

### fm-embeddings-5 — The H3 consistency penalty is evaluated with the network in `train()` mode, so independent dropout masks contaminate the objective it is documented to optimize
- **Severity:** low
- **Liveness:** dead-closed (H3)
- **Confidence:** high (measured)
- **Where:** `src/modeling/mlp_head.py:196-208`

`DECISIONS.md:4417-4418` and the method docstring (`mlp_head.py:152-158`) define the penalty as `λ·mean((sigmoid(net(e_i)) − sigmoid(net(e_j)))²)` for the deterministic head. As implemented, both forward passes happen inside the epoch's `self._net.train()` block, so each draws its **own** dropout masks; the term therefore equals the frame-disagreement plus an irreducible dropout-variance floor that is present even when `e_i == e_j`. Measured at convergence on the real 40 000 pairs: for the λ=0 base head (`deployable_f_center`, member 0) the same-input train-mode term is 0.00696 against an as-implemented 0.08060 (**8.6 %**); for the λ=10 head it is 0.00151 against 0.01070 (**14.1 %**). So the quantity minimized is "be insensitive to dropout **and** to frame", not the stated objective, and part of the skill cost attributed to frame-invariance is generic dropout shrinkage.

- **Failure scenario:** anyone re-running the H3 sweep, or reusing `lambda_consistency` for the co-located-overlap objective in a future lever, optimizes a penalty whose minimum is not at frame-invariance; the λ that reaches a given η² is inflated, and the accompanying skill loss over-charged.
- **Evidence:**
  ```
  src/modeling/mlp_head.py:196-208
      for _epoch in range(self.epochs):
          self._net.train()                      # <- dropout ON for the lines below
          ...
                  if Ca is not None:
                      m = Ca.shape[0]
                      cidx = torch.randint(0, m, (min(self.batch, m),), device=device)
                      pa = torch.sigmoid(self._net(Ca[cidx]).squeeze(-1))
                      pb = torch.sigmoid(self._net(Cb[cidx]).squeeze(-1))
                      loss = loss + self.lambda_consistency * ((pa - pb) ** 2).mean()
  ```
  Measured (member 0, real pairs, scaler applied):
  ```
  deployable_f_center   E[(dp)^2] eval/frame-only=0.07655  train same-input (pure dropout)=0.00696  as-implemented=0.08060  -> 8.6%
  deployable_f_h3_lam10 E[(dp)^2] eval/frame-only=0.00929  train same-input (pure dropout)=0.00151  as-implemented=0.01070  -> 14.1%
  ```
- **Self-refutation attempted:** Two things argue the H3 verdict is safe. (1) Magnitude: 9–14 % of the penalty, not dominant. (2) Direction: `DECISIONS.md:4437-4440` identifies the observed mechanism as compression of the head's dynamic range toward the middle (in-sample p|pos 0.785→0.631, p|neg 0.207→0.396), but sigmoid output variance under logit noise is *maximal* near p = 0.5, so minimizing the dropout term would push predictions **away** from the middle — the opposite of the observed mechanism. So the contamination did not drive the trade-off and the FAIL stands. It is nonetheless an implementation infidelity to the documented formula, and I could not find any note acknowledging it. I also checked that `test_lambda_zero_ignores_pairs_exactly` (`tests/test_deployable_head.py:232-240`) is not pinning this behaviour — it only exercises λ=0, where `Ca is None`.
- **Fix:** compute the consistency term with dropout disabled — e.g. `self._net.eval()` around the two pair forwards then back to `train()`, or a shared-mask forward on `cat([Ca[cidx], Cb[cidx]])` under a functional-dropout override — and re-run the λ sweep if H3 is ever reopened.

## Refuted by my own check

- **The 3 ensemble members are not genuinely distinct.** They are: `MLPClassifierHead.fit:163` calls `torch.manual_seed(self.seed)` before `build_mlp`, so init, shuffling and dropout masks all differ per seed, *and* `DeployableHead.fit:392-393` gives each seed a different inner-val image (`unique[s % unique.size]`). Verified by hash: the three `seed*/state.pt` produce three distinct `model_hash` contributions.
- **Ensemble averaging space is inconsistent with the calibration layer.** It is consistent: the validated LOIO `mlp_ens3` is `base[[f"p{s}"]].mean(axis=1)` over sigmoid probabilities (`scripts/probes/_fm_freeze_window.py:240`), the deployed head is `acc += head.predict(X); return acc/len(members)` over sigmoid probabilities (`mlp_head.py:407-414`), and `CalibrationLayer` was fit on exactly those LOIO mean-probabilities. Both are probability-space means — no logit/probability mismatch.
- **`np.interp` in `calibrate_abundance` clamps the deployed head's out-of-range probabilities, saturating the abundance map.** Checked the banked knots: `t2_x` spans 1.2e-6 … 0.99991 over 4000 quantiles (and `meta["n"] = 161005`, which exactly equals the 161 005 tiles in the frozen P96 store — a clean cross-check). Nothing a sigmoid can emit lies outside that range, so clamping is unreachable.
- **The frozen store's NaN margin rows mean the head was trained on median-imputed garbage features paired with real labels.** Measured: **all 38 `*_P96.npz` files have zero invalid rows** (161 005/161 005 valid). The `FeatureScaler` median-impute path is never exercised by the frozen recipe. (It *is* exercised by the F/H2/H3 stores, where `valid` has false entries.)
- **The H2 basis is not orthonormal, so `_project` is not a projection.** It is: `f_h2_nuisance.py:151-154` takes `np.linalg.eigh` eigenvectors; measured `max|NᵀN − I| = 1.1e-8` over the top 64 columns, and `_project` is applied identically in `fit:371` and `predict:410`.
- **The η² test set is circular with the H2 basis.** It is not: the basis pool comes from `reports/f_leg_b/obs_crops/` (73 crops, 28 obs) while the η² pilot frames come from `reports/f_timing/pilot_crops/` (`f_pilot_crop.py:63`) — genuinely disjoint directories, exactly as claimed.
- **`load_fold` masks `y_*_fold{k}.parquet` positionally by `x_df["scale_idx"]`, which would mis-join labels if the two files were not row-parallel.** They are row-parallel by construction: `src/dataset.py:663-676` writes `X`, `y` and `groups` from the *same* `train_df` via column subsets. Safe.
- **The training embedding store and the map path read differently-scaled rasters.** `extract_ctx_window` (`src/ctx_retrieve.py:404-455`) preserves dtype and per-pixel size and only changes the affine origin/shape, so the Stage-2 `ctx_window_tif` is a byte-identical crop of the same `/vsizip/` inner TIFF the map windows. `_fm_parity_check.py` independently confirms the productized slicer reproduces the cached store (validity mask equality + per-row cosine > 0.999).
- **The grid anchor drifted between the training extractor and `src/fm_embeddings.slice_context_boxes`.** Byte-for-byte identical arithmetic (`_w2_fang_embed.py:213-217` vs `fm_embeddings.py:193-195`), and `tile_grid_for_window`'s `ceil`/`floor` bounds are the exact inverse of the validity predicate (`tests/test_fm_embeddings.py:122-128` pins the agreement).
- **`fang_columns_for_keys` could silently mis-join or reuse a wrong row.** `validate="one_to_one"` raises on duplicate keys on either side (including the real case of `scale_idx=None`, where S=32 and S=64 share `(ti,tj)` values), and `np.isnan` on the int64 `row` column is well-defined, so the missing-tile assert fires as intended.
- **The deployed head's raw P(rich) distribution may not match the LOIO distribution the qmatch was fit on.** Real concern but unfalsifiable here (the deployed head is out-of-sample on new terrain, so an in-sample comparison proves nothing) and already acknowledged in `CalibrationLayer`'s docstring (`src/calibration.py:320-325`: "only mis-scale where the head itself is fooled by out-of-distribution texture — an off-cohort / global-map concern handled later by the (deferred) novelty hook").
- **`slice_context_boxes` silently truncates a non-uint8 window** (`boxes = np.empty(..., dtype=np.uint8)` then assigns from `window`, so float or int16 input wraps/truncates with no error). Every caller already produces uint8 (`read_tile_window`, `map_uint8`, `composite_crops`, `a1_apply`), so latent only — not filed.
- **Any script in this area importing torch before `src.modeling`.** All four checked files bootstrap correctly (`_w2_fang_embed.py:45`, `parity_check.py:30`, `_fm_parity_check.py:18`, `train_deployable_head.py:34`); `src/fm_embeddings.py` and `src/modeling/mlp_head.py` import torch lazily inside methods, as documented.

## Verified clean

- `load_timm_state_dict` is genuinely strict: the dict comprehension `KeyError`s on any unexpected source key and `load_state_dict(..., strict=True)` raises on missing/unexpected targets, so a layout drift cannot leave random weights.
- `FeatureScaler` is train-only in every path: `fit` on the inner-train rows; `apply` on the eval_set (`mlp_head.py:192`), on the H3 pairs (`:186-187`), and at predict. `to_arrays`/`from_arrays` round-trips exactly (`tests/test_deployable_head.py:57-62`).
- `DeployableHead._project` is applied identically in `fit` (before the scaler) and `predict`, travels via `save`/`load`, is folded into `model_hash`, and passes all-NaN rows through untouched.
- `embed_window`'s NaN-row bookkeeping is row-parallel: `slice_context_boxes` emits boxes in ascending `np.where(valid)[0]` order and `emb[np.where(valid)[0]] = emb_valid` writes back in the same order.
- `predict_window` really does mask before predicting (`usable = valid & (zero_frac <= max_zero_fraction)`; `head.predict(emb[usable])`), so no imputed/invalid row reaches the raster.
- `f_region_stageb.py`'s `keep`-then-`embed_window` ordering keeps `ti/tj/valid/prob` aligned, and its overlap dedup (`np.add.reduceat` with `np.unique(..., return_index=True)`) is arithmetically correct including the final segment.
- GeM parity between `gem_pool_np` and `_pool_tokens`, and the normalize-then-resize preprocessing parity between `FangEmbedder.preprocess` and `_w2_fang_embed.embed_batches` (re-confirmed independently).
- The frozen recipe really is emb-only 3×3-context: `_fm_freeze_window.MATRICES["emb"] = ("ctx",)` → a single 768-column P96 block, matching `FROZEN_RECIPE["input_px"] = 96` and the shipped head's `d_in = 768`.
- `build_all_image_matrix` (`train_deployable_head.py:50-84`) is a true union of per-fold test slices (each image once), and correctly skips folds whose held-out image is absent from a partial F store without tripping the all-present assert.
- `standardize_fold_per_image` / `augment_fold_with_per_image` compute statistics within each group from that group's own rows only — no cross-split statistics.

## Coverage note

**Read in full:** `src/fm_embeddings.py`, `src/modeling/mlp_head.py`, `src/modeling/loaders.py`, `scripts/probes/_w2_fang_embed.py`, `scripts/parity_check.py`, `scripts/probes/_fm_parity_check.py`, `scripts/train_deployable_head.py`, `src/mapping.py`, `src/calibration.py`, `tests/test_fm_embeddings.py`, `tests/test_deployable_head.py`, `scripts/f_leg_b_loio.py`, `scripts/f_h2_nuisance.py`. **Read in part:** `scripts/probes/_w2_fang_heads.py` (the `MLPHead`/`_StandardizedHead` that produced 0.7832 — compared line-by-line against the productized head; the only deltas are `batch` 512→4096 and `randperm` on the CUDA vs CPU generator, both documented as implementation choices), `scripts/probes/_fm_freeze_window.py`, `scripts/probes/_w2_fang_probe.py`, `src/modeling/evaluate.py:600-700`, `scripts/map_region.py`, `scripts/f_region_stageb.py`, `scripts/f_leg_b_embed.py`, `scripts/f_h3_pareto.py`, `src/ctx_retrieve.py:404-455`, `src/dataset.py:604-690`.

**Measurements run** (read-only, on committed/local artifacts — no imagery, no network): parity-reference and calibration-knot ranges; invalid-row counts across all 38 frozen `*_P96.npz`; `model_hash`/`recipe_hash`/`nuisance_k` recomputation for all 11 heads under `models/deployable*/`; basis/pair-pool vs LOIO-fold overlap; basis orthonormality; the H3 dropout-contamination decomposition on the real 40 000 pairs; per-store npz mtime spans.

**Could not check:** anything requiring the 341 MB Fang checkpoint forward pass on two devices (so I could not independently measure GPU-fp16 vs CPU-fp32 embedding drift — `parity_check.py`'s `atol=2e-3` claim of "~1e-3 on probabilities" is taken on trust); whether the `.ipynb` for notebooks 26–28 reproduces the H2/H3/H4 tables (notebooks are out of scope here and belong to the `notebooks` area); whether `dataset_v2/fang_embeddings` was in fact produced by the committed `_w2_fang_embed.py` rather than an earlier revision (the store predates the current file's `--norm` flag by commits I did not diff); and R09's provenance surface beyond what I extended above (I confirmed the collision quantitatively — all 11 heads share directory name `86c51a5dca220f63`, and on one fixed random 256×768 batch their mean P(rich) is 0.238 / 0.047 / 0.018 for `deployable` / `deployable_f_center` / `deployable_f_h2_k4`, so a mis-pointed `--model` is a large, silent error — but did not re-file it). Also unchecked: whether `map_region.py`'s per-tile sidecar should record model/calibration identity (it records none; the only model provenance is a stdout line naming the shared hash) — flagged here for the `other-scripts`/`docs-consistency` areas rather than filed as a finding.
