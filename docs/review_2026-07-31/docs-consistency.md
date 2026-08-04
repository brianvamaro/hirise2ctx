# Review area: docs-consistency

- **Reviewed at commit:** da884c7
- **Date:** 2026-08-01
- **Verification:** self-refuted (single-agent; PASS 1 + an independent PASS 2 by a second agent — neither independently verified by a third party)

> **Note on structure.** This file now carries **two** passes. `docs-consistency-1..-6` are PASS 1
> (2026-08-01) and are already folded into the register as **R37** (`-1/-2/-3`) and **R42**
> (`-4/-5/-6`), which cite them **by ID** — so their numbering is frozen and they are reproduced
> unchanged. `docs-consistency-7..-10` are PASS 2: a second independent agent re-ran the area
> against the gaps PASS 1's own coverage note declared unchecked (prose-level numerical agreement
> inside `docs/methods.md` / `docs/model_evidence.md`, and `PLAN_RegionalMap.md` outside §10). PASS 2
> spot-checked several PASS-1 findings and did not contradict any of them.
>
> **Severity ranking across both passes:** `-7` (high) · `-1` (high) · `-8` (medium) · `-9` (medium) ·
> `-2` (medium) · `-3` (medium) · `-4` (medium) · `-5` (medium) · `-10` (low) · `-6` (low).

## Findings

### docs-consistency-1 — README's Status + "Next priorities" and SHERLOCK_RUN Part J are entirely pre-abort: they instruct the next session to run the 907-frame F build that was hard-aborted
- **Severity:** high
- **Liveness:** live-shipped (these are the two entry-point operating docs)
- **Confidence:** high
- **Where:** `README.md:14`, `README.md:24-36`, `README.md:94-104`, `README.md:370-387`;
  `SHERLOCK_RUN.md:597-601`, `SHERLOCK_RUN.md:623-650`

`git log` shows `README.md` and `SHERLOCK_RUN.md` were last touched at `458168f` (2026-07-29);
the abort landed at `41a6f26` (2026-07-30) and touched only `DECISIONS.md`, `PLAN_FBuild.md`,
`ROADMAP.md` and `reports/`. So both docs still present the F build as the live next step, with a
concrete cost estimate and a numbered runbook. README is the doc `CLAUDE.md` points at for "Setup +
how to run each stage", and `ROADMAP.md:43` calls SHERLOCK_RUN the "operational runbook … (active
reference)". A fresh session that follows either will spend real GPU/laptop hours re-deriving a
verdict that already exists — and README:383-386 routes it into `scripts/striping_a1_map.py`, the
path **R07** shows is train/deploy-inconsistent. This is broader than **R06** (which is only about
`reports/map_a1/` not existing) and than **R10** (which is only about ROADMAP:18's causal
attribution).

- **Failure scenario:** a session reads `README.md:94` → "PLAN_FBuild is fully built and now needs
  *running*" → follows `SHERLOCK_RUN.md:597` Part J → transfers 2 GB of logits, re-runs Stage C/D +
  gates, then commits ~6–8 GPU-h to `striping_a1_map.py` + the A1 LOIO re-run for the §5.1 A1
  column. All of it re-derives an ABORTED verdict, and the A1 half is generated through the R07
  defect.
- **Evidence:**
  ```
  README.md:14   **Current phase (2026-07): regional deployment + striping-artifact Phase 2 (invariance & leveling).**
  README.md:34   ... **H1+H4 is the first stack to reach the
  README.md:35   reopening bar** (eta² ≲ 0.05 at skill ≥ −0.02); the 907-frame build is planned in
  README.md:36   [PLAN_FBuild.md](PLAN_FBuild.md), gated on its §0 checklist.
  README.md:94   **Next priorities:** **PLAN_FBuild** is fully built and now needs *running*. The reopening call landed
  README.md:97   ... The only thing between here and the F-build verdict is the ~2 GB npz
  README.md:98   transfer plus ~30 min of laptop compute — see **SHERLOCK_RUN Part J** for the exact order.
  README.md:100  ... If the gates fail → fall back to
  README.md:101  shipping the A1 map + caveat + H6 provenance.
  README.md:370  Stage D writes into **`reports/map_fbuild/`** — never `reports/map_region/`, which stays on disk as the
  README.md:371  comparison object.
  SHERLOCK_RUN.md:599  **Stage B is COMPLETE: 906/907 npzs** ... Everything downstream is a laptop step ... so the whole
  SHERLOCK_RUN.md:601  remaining build is a transfer plus ~30 minutes of local compute.
  ```
- **Self-refutation attempted:** (a) *CLAUDE.md redirects to ROADMAP for current phase* — true
  (`CLAUDE.md:10-13`), but ROADMAP's own ACTIVE row is also stale (finding 2), and README:94 is an
  explicit imperative, not a status line. (b) *Maybe it's deliberate history* — no: the project's
  own convention (`ROADMAP.md:50-51` and CLAUDE.md's closing line) requires updating the affected
  doc "in the same change that diverges", and `ROADMAP.md`/`PLAN_FBuild.md` **were** updated at
  `41a6f26`, so README/SHERLOCK were simply missed. (c) *Maybe SHERLOCK Part J is marked historical*
  — it is not; it is written in the imperative with numbered steps and "already banked" claims.
- **Fix:** in the same edit: (i) replace README's Status block and "Next priorities" with
  "F build HARD-ABORTED 2026-07-30 (`41a6f26`); the **mosaic** map in `reports/map_region/` is the
  deliverable; open work = PLAN_RegionalMap's thermal legs"; (ii) put a dated `**CLOSED — F build
  aborted 2026-07-30; retained as the record of how it was run**` banner at the top of
  SHERLOCK_RUN Parts E–J; (iii) correct README:370-371's polarity (`map_region` is the product,
  `map_fbuild` the abandoned arm).

### docs-consistency-2 — The only ACTIVE plan is still documented as blocked on the F map, contradicting the commit message of the change that made it the only active plan
- **Severity:** medium
- **Liveness:** live-active-plan
- **Confidence:** high
- **Where:** `ROADMAP.md:12`; `PLAN_RegionalMap.md:323-324`, `PLAN_RegionalMap.md:327-330`

`da884c7`'s commit message states outright: *"PLAN_RegionalMap is now the only ACTIVE plan, and its
thermal / Rodriguez-2016 legs are **unblocked**: they were explicitly waiting on 'the final
(post-mitigation) map', which is settled as A1/mosaic."* But `git show da884c7 -- ROADMAP.md`
changed only the two rows it moved into the CLOSED table — the ACTIVE row was left verbatim, and
`PLAN_RegionalMap.md` was not touched at all (last modified `efe545e`, 2026-07-13). So the intent
recorded in the commit never landed in either document, and the repo's sole open workstream reads as
waiting on an artifact that will never be produced.

- **Failure scenario:** the next session opens ROADMAP (the file CLAUDE.md designates as the
  authority for "current phase"), reads the one ACTIVE row as "remaining thermal legs wait for the
  final (post-mitigation) map", cross-checks `PLAN_RegionalMap.md:327` which says the legs resume
  "on it" (the F map from PLAN_FBuild), and concludes there is **no** runnable work — the exact
  mis-read the commit message says it was preventing.
- **Evidence:**
  ```
  ROADMAP.md:12   | [PLAN_RegionalMap.md] ... | **ACTIVE** — map shipped (26 tiles, Sherlock); MOLA leg done;
                    THEMIS night-IR leg-1 done but weak (ρ ≈ +0.07); remaining thermal legs wait for the
                    **final (post-mitigation) map** |
  PLAN_RegionalMap.md:323  **The remaining quantitative thermal legs resume on whichever final map the mitigation decision
  PLAN_RegionalMap.md:324  produces.**
  PLAN_RegionalMap.md:327  **UPDATE 2026-07-13 — validation-leg relaunch staged for the post-mitigation map.** The mitigation
  PLAN_RegionalMap.md:328  arc has converged: PHASE-2 H1+H4 ... is the first stack to
  PLAN_RegionalMap.md:329  reach the reopening bar, and the 907-frame per-frame build is planned in
  PLAN_RegionalMap.md:330  [PLAN_FBuild.md](PLAN_FBuild.md) ... When that map ships, the parked legs resume **on it**
  ```
- **Self-refutation attempted:** (a) *Is the commit message maybe describing a follow-up?* — no
  follow-up commit exists; `da884c7` is HEAD. (b) *Is the ACTIVE row's wording still true under a
  charitable reading ("final map" = the mosaic)?* — `PLAN_RegionalMap.md:330` pins "it" explicitly to
  PLAN_FBuild's map, so the charitable reading is closed off by the plan the row links to. (c) *Does
  a `project_state_*` memory note carry it?* — the 2026-07-30 note says "thermal legs unblocked", but
  memory notes are session state, not the repo's plan index; ROADMAP.md:55 says "keep it current when
  a plan opens, closes, or is superseded".
- **Fix:** amend `ROADMAP.md:12` to "remaining thermal legs **unblocked 2026-07-30** — they run on
  the **mosaic** map (`reports/map_region/`), the settled deliverable", and add a dated
  `UPDATE 2026-07-30` to `PLAN_RegionalMap.md` §10 replacing the 2026-07-13 relaunch note's premise
  (also re-pointing its pre-declared leg-1 "not degraded" gate, which was written against a leveled
  map that does not exist).

### docs-consistency-3 — Gate 1's "banked mosaic baseline" was silently overwritten in place; three docs still cite numbers that the file they name no longer contains
- **Severity:** medium
- **Liveness:** live-shipped (the quoted statistic characterises the **shipped mosaic** map's artifact)
- **Confidence:** high
- **Where:** `DECISIONS.md:5022`, `README.md:382-383`, `SHERLOCK_RUN.md:639-640` vs
  `reports/figures/fbuild_gate1_summary.csv` (mosaic row) and `DECISIONS.md:5505`

All three docs quote the mosaic gate-1 row as **η² 0.1222 / null p95 0.0676 / ratio 1.65**, and
`DECISIONS.md:5022` names `fbuild_gate1_summary.csv` as the evidence. That file was written at
`afe6fce` with exactly those values, then **rewritten by `41a6f26`** with **0.120535 / 0.07001 /
ratio 1.5276** (tile-scale row likewise 0.184986/1.5081 → 0.180379/1.5368). The cause is legitimate —
`41a6f26` added the common-footprint mask to `scripts/f_region_gates.py:78-85`, so the mosaic is now
scored on the F rows' restricted footprint — but nothing records that the baseline moved. As a result
`DECISIONS.md` carries **two** mosaic windowed-η² values (0.1222 at :5022, 0.121 in the abort table at
:5505) and two artifact ratios (1.65 and 1.528), only the second of which is on disk.

- **Failure scenario:** anyone quoting "the mosaic map's striping artifact is 1.65× its own rotation
  null" (the phrasing at `SHERLOCK_RUN.md:640`, "ratio 1.65 — the artifact") cites a number that no
  committed artifact supports; re-running `f_region_gates.py` to check reproduces 1.528 and looks
  like a regression. Separately, README:99 / SHERLOCK:640 tell the operator "the F rows are what this
  run adds" / "the mosaic baseline is already banked", which is false — the run recomputes the mosaic
  row into the same file, so there is no frozen pre-registered baseline to compare against.
- **Evidence:**
  ```
  DECISIONS.md:5022   **Mosaic baseline banked** (`fbuild_gate1_summary.csv`): median-window η² 0.1222, null p95 0.0676,
  DECISIONS.md:5023   excess +0.0719, ratio 1.65, 21.4% of windows already under 0.05, `passes_bar` False.
  SHERLOCK_RUN.md:639 Gate 1's mosaic baseline is already banked (median-window partition η² **0.1222** against its own
  SHERLOCK_RUN.md:640 rotation-null p95 **0.0676**, ratio 1.65 — the artifact). The F rows are what this run adds.
  README.md:382       The mosaic baseline is banked at median-window η² **0.1222** vs null p95 **0.0676**.

  $ git show afe6fce:reports/figures/fbuild_gate1_summary.csv
  mosaic,234,0.12215999999999999,0.2885480000000003,0.04327,0.06764,0.07188,1.6496550409843305,0.21367521367521367,False,0.1849861007546681,0.09418773531480569,1.5081329304529536
  $ cat reports/figures/fbuild_gate1_summary.csv          # HEAD (da884c7), written by 41a6f26
  mosaic,234,0.120535,0.27523600000000004,0.042225,0.07001,0.07080000000000002,1.527551104572662,0.21794871794871795,False,0.18037860889434482,0.09005304717357696,1.5367735594812735
  ```
- **Self-refutation attempted:** (a) *Rounding?* — no: 0.1222 vs 0.1205 and 1.650 vs 1.528 (7.7 %)
  are different measurements, and `frac_windows_below_bar` moved 0.2137 → 0.2179. (b) *Does the abort
  entry reconcile them?* — `DECISIONS.md:5495-5560` never mentions that the mosaic row was re-scored;
  its "Corrections" list (4 items) does not include it. (c) *Does the change alter a verdict?* — no,
  every row fails the 0.05 bar either way, which is why this is medium and not high; but the mosaic's
  artifact magnitude is a **live** number about the shipped map. (d) *Is this R33 or the register's
  "gate 1's common-footprint fix … verified clean"?* — no: the register verified the fix is correct
  and non-differential; nobody noticed it silently superseded the banked baseline the docs cite.
- **Fix:** in `DECISIONS.md`, add a line to the 2026-07-30b entry: "the banked 0.1222/0.0676/1.65
  baseline was measured on the mosaic's **own** footprint at `afe6fce`; the common-footprint fix
  (`41a6f26`) re-scores it to 0.1205/0.0700/1.528, which is what `fbuild_gate1_summary.csv` now
  holds". Update `README.md:382` and `SHERLOCK_RUN.md:639-640` to the on-disk pair and drop "the F
  rows are what this run adds".

### docs-consistency-4 — `docs/modeling.md` §11 "Reproducibility" gives a command that cannot run and names an artifact directory that does not exist
- **Severity:** medium
- **Liveness:** live-shipped (this is the reproducibility statement for every v2 LOIO number in `modeling_results.md`)
- **Confidence:** high
- **Where:** `docs/modeling.md:634`, `docs/modeling.md:643`; `scripts/sweep.py:135-138`,
  `src/modeling/loaders.py:69-71`

`--dataset-dir` is the **dataset root**; `package_dir()` appends `packaged/{scheme}` itself, and the
scheme comes from the separate `--scheme` flag (default `loio_9fold`). So the documented command
resolves to `dataset_v2/packaged/loio_38fold/packaged/loio_9fold`. On top of that, `loio_38fold` is
not a scheme name — the v2 scheme is `loio_nfold` (`dataset_v2/packaged/` contains
`loio_nfold`, `loio_nfold_ctx_illum`, `loio_nfold_nbr_s5`, `within_image_4fold`), and the string
`loio_38fold` appears nowhere else in the repo. `models/_sweep_smallcnn/` likewise appears nowhere
else; the CNN sweeps live in `models/_sweep_cnn/`.

- **Failure scenario:** a reviewer or committee member trying to reproduce the v2 LOIO headline runs
  the documented line and gets a `FileNotFoundError` on a path with `packaged` twice in it, with no
  hint that both the flag semantics and the scheme name are wrong; and looking for the CNN
  provenance directory named in the same section finds nothing.
- **Evidence:**
  ```
  docs/modeling.md:634  - A CNN run under `models/cnn_*` or `models/_sweep_smallcnn/`.
  docs/modeling.md:643  Re-running `python scripts/sweep.py --dataset-dir dataset_v2/packaged/loio_38fold`
  docs/modeling.md:644  against the unchanged packaged data reproduces the v2 LOIO numbers.

  scripts/sweep.py:135      ap.add_argument("--dataset-dir", default=None,
  scripts/sweep.py:136          help="Packaged dataset root (default: ./dataset = v1). Use dataset_v2 for the vClaire A/B.")
  scripts/sweep.py:137      ap.add_argument("--scheme", default=DEFAULT_SCHEME,   # DEFAULT_SCHEME = "loio_9fold"
  src/modeling/loaders.py:70    base = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
  src/modeling/loaders.py:71    return base / PACKAGED_SUBDIR / scheme
  ```
  README.md:232 has the correct invocation for comparison: `--dataset-dir dataset_v2 --scheme loio_nfold`.
- **Self-refutation attempted:** (a) *Was `loio_38fold` ever a real scheme that has since been
  renamed?* — `grep -rn loio_38fold` over `.md`, `.py`, `.yaml` returns only this one line, and
  `dataset_v2/splits/loio_nfold.json` records `n_folds: 38, n obs: 38`, so the "38" was descriptive
  prose, never a name. (b) *Does `--dataset-dir` accept a scheme dir by some fallback?* — no,
  `package_dir` unconditionally appends `packaged/{scheme}`. (c) *Is `docs/modeling.md` marked
  historical?* — `docs/index.md:16` lists it as the current methods companion to `modeling_results.md`.
- **Fix:** `docs/modeling.md:643` → ``python scripts/sweep.py --dataset-dir dataset_v2 --scheme
  loio_nfold``; `:634` → `models/_sweep_cnn/`.

### docs-consistency-5 — `DATA_DICTIONARY` states the detection filters are null "the current default"; both configs set `min_size_m: 1.4105` and it demonstrably drops polygons
- **Severity:** medium
- **Liveness:** live-shipped (this is the schema doc for the label basis of every reported number)
- **Confidence:** high
- **Where:** `dataset/DATA_DICTIONARY.md:176`; `config.yaml:88`, `config_v2.yaml:105`

The dictionary's gloss on `n_polygons_after_filter` tells the reader the size/confidence filters are
inactive. They are not, and have not been since 2026-05-26 (`config.yaml:82-88` records the
`AskUserQuestion` that set them). Measured on the sidecars on disk: v2 `ESP_045550_2180` 326,636 →
320,706 (−1.8 %), `ESP_042964_2160` 34,237 → 34,222; v1 `ESP_047976_2020` 1,346 → 1,324. This is the
same `min_size_m: 1.4105` global floor that **R03** shows is a 0.25-vs-0.50 m/px label confound — so
the one schema document a reader consults to understand the label basis actively says the confound
is not there.

- **Failure scenario:** someone auditing the label provenance (exactly what R03 requires) reads
  `DATA_DICTIONARY:176`, concludes no detection filter is applied, and does not look for the global
  size floor or its pixel-scale asymmetry; or reads a sidecar where
  `n_polygons_after_filter == n_polygons_stage1` (true for many images, e.g. `ESP_017355_2260`) and
  takes it as confirmation, when it only means that image had no sub-threshold polygons.
- **Evidence:**
  ```
  dataset/DATA_DICTIONARY.md:176
  | `n_polygons_after_filter` | int | Polygon count after applying `detection_filters.min_confidence` /
    `min_size_m` (equal to `n_polygons_stage1` when both are null, the current default) |

  config.yaml:88       min_size_m: 1.4105
  config_v2.yaml:105   min_size_m: 1.4105

  dataset_v2/labels/ESP_045550_2180.json -> 326636 320706 {'min_confidence': None, 'min_size_m': 1.4105}
  dataset/labels/ESP_047976_2020.json    ->   1346   1324 {'min_confidence': None, 'min_size_m': 1.4105}
  ```
- **Self-refutation attempted:** (a) *Maybe the dictionary describes only v1 `dataset/`* — checked:
  `config.yaml` (v1) also sets 1.4105 and v1 sidecars show the drop. (b) *Maybe the sentence is a
  conditional, not an assertion* — the trailing clause "the current default" is an assertion about
  the shipped configs, and it is false in both. (c) *Is this just R03 re-filed?* — no: R03 is about
  the scientific consequence of the single global floor; this is that the schema doc denies the
  floor exists at all, which is why R03's confound went unexamined for so long.
- **Fix:** `dataset/DATA_DICTIONARY.md:176` → "…(`min_confidence: null`, `min_size_m: 1.4105` in both
  shipped configs — a single global equivalent-circle-diameter floor set to the 0.25 m/px design
  floor; see `config.yaml:83-88` and DECISIONS 2026-05-26)".

### docs-consistency-6 — `docs/index.md`, defined as the docs index, omits 4 of the 11 documents in `docs/` — including `model_evidence.md`, the headline evidence writeup
- **Severity:** low
- **Liveness:** live-shipped
- **Confidence:** high
- **Where:** `docs/index.md:11-19` (the index table), `docs/index.md:23-24` (planned-docs list);
  `README.md:42-48` (deliverables table)

`docs/` contains `build_spec, classification_slimmer, compositional, compositional_slim,
compositional_slimmer, index, methods, model_evidence, modeling, modeling_results, modeling_slim,
w2_litreview`. The index table lists 7 of them. Missing: `model_evidence.md` (the document that
carries the frozen recipe's pooled 0.7832 / prec@5% 0.948 / med AUC 0.7865 and the gap-fill figure),
`classification_slimmer.md` (its declared Part 1 companion), `compositional_slimmer.md`, and
`w2_litreview.md`. `README.md:42-48`'s deliverables table also omits `model_evidence.md`. Relatedly,
`docs/index.md:24` still gates the planned `striping_artifact.md` on "*write once the Phase-2 docket
verdict lands — the ending decides which map ships*"; the verdict landed 2026-07-30, so that
document is now due and the gating note reads as if it were still pending.

- **Failure scenario:** a reviewer routed through `docs/index.md` ("You want to understand the rock-
  abundance modelling stage: read modeling.md then modeling_results.md") lands on the 2026-06-02
  LightGBM-era writeups and never sees `model_evidence.md`, the only document that presents the
  foundation-model recipe as evidence — so the project's strongest result is invisible on its own
  index. `model_evidence.md` also carries the `[held-out: pending]` placeholders, which nobody will
  chase if the document is unindexed.
- **Evidence:**
  ```
  docs/index.md:12-18   | build_spec.md | methods.md | modeling.md | modeling_slim.md |
                          modeling_results.md | compositional.md | compositional_slim.md |   (7 rows)
  docs/index.md:24      - `striping_artifact.md` — *(queued 2026-07-09; **write once the Phase-2 docket verdict
                          lands** — the ending decides which map ships)*
  $ ls docs/*.md   -> 12 files (index.md + 11 documents)
  ```
- **Self-refutation attempted:** (a) *Maybe the missing four are deliberately internal* — no:
  `docs/index.md:5-7` defines the folder as "written for **readers**, including reviewers,
  collaborators, and the project's advising committee", and `model_evidence.md:3` names
  `classification_slimmer.md` as its Part 1, so both are reader-facing by their own text. (b) *Maybe
  README's table covers them* — it does not (`README.md:42-48` lists 5, none of them
  `model_evidence.md`). (c) *Consequence?* — discoverability only, hence low.
- **Fix:** add the four missing rows to `docs/index.md`'s table and `model_evidence.md` to
  `README.md:42-48`; change the `striping_artifact.md` gating note to "verdict landed 2026-07-30
  (F ABORTED, mosaic ships) — **now writable**".

---

## Findings — PASS 2 (second independent agent, 2026-08-01)

### docs-consistency-7 — `docs/methods.md`, the reader-facing pipeline Methods document, was half-migrated to the v2 cohort: §5 reports 39 vClaire images while §2/§6/§7/§8 report the superseded 9-image v1 sweep as "the current dataset", and its target-distribution statistics are wrong by 2×–70×
- **Severity:** high
- **Liveness:** live-shipped (`README.md:8-10` and `docs/index.md:32` route reviewers, collaborators and the advising committee here for "how the dataset was produced")
- **Confidence:** high
- **Where:** `docs/methods.md:3-4` (the self-pin), `:57`, `:120`, `:596-600`, `:646-647`,
  `:777-785`, `:790`, `:853`, `:1010-1036`, `:1097-1118`, `:1159-1160`

The document's banner pins it to "commit `b9bc82a`, 2026-05-25", which would make its v1 scope an
honest snapshot. But it was edited afterwards at `479688d` (2026-05-28), which **added §5.2–5.6** and
migrated only §5 to the vClaire v2 cohort ("Across the 39 retained vClaire images…", "38 of 39 images
proceed"), leaving §2.1 asserting "Ten HiRISE Observation IDs make up **the current dataset**" and
§§6.7 / 7.4 / 8.2 / 8.4 on the 9-image priority10 sweep. So the pin is false, the document
contradicts itself about which cohort it describes, and the reader has no way to tell which numbers
belong to which. Measured on disk, the gap is not cosmetic: §6.7's "**97.88 %** of finest tiles have
`boulder_area == 0`" reproduces v1 exactly (0.9789) but the actual cohort is **50.20 %**; mean
`fractional_area` is **1.54 × 10⁻²**, not the stated 2.2 × 10⁻⁴ (**70×**); P99 is **1.30 × 10⁻¹**, not
6.25 × 10⁻³ (**21×**); max is 0.436, not 0.269; and the sweep is **3,564,767** tiles over 38 images,
not the stated 643,910 over 9. §7.4 then builds a *causal* explanation on the wrong number
("correlations are all weak … **because** … 97.9 % of finest tiles have `fractional_area = 0`, so the
Spearman statistic is dominated by ties at zero") — the same framing `DECISIONS.md:3530-3536`
later **empirically refutes** ("removing the zeros LOWERS Spearman by ~0.01 … the earlier framing is
empirically refuted").

- **Failure scenario:** a committee member or external reviewer reads the document the README sends
  them to, and takes away that the training set is 10 images / 643,910 tiles with a 97.9 %-zero
  target — then reads `docs/model_evidence.md` (38 images, base rate 0.36) or `modeling_results.md`
  §9+ and cannot reconcile them. Concretely: anyone reasoning about the zero-inflation invariant
  (CLAUDE.md invariant 5) from §6.7/§7.4 will design for a 98 %-zero target when the shipped labels
  are 50 % zero at the finest scale and ~16 % at S=32, and will inherit the "weak-ρ-because-of-ties"
  explanation the project has since retracted.
- **Evidence:**
  ```
  docs/methods.md:3-4    > ... as it stands at the end of CLAUDE.md Week 1-2 scope (commit `b9bc82a`,
                         > 2026-05-25).
  docs/methods.md:120    Ten HiRISE Observation IDs make up the current dataset.
  docs/methods.md:596    ### 5.4 Results (vClaire 40-image dataset)
  docs/methods.md:598    Across the 39 retained vClaire images, solved shift magnitudes range from 80 m to
  docs/methods.md:777    Across the nine retained manifest images, Stage 4 emits **488,554 finest tiles**,
  docs/methods.md:778    **119,944 tiles at S=16**, **28,825 tiles at S=32**, and **6,587 tiles at S=64**
  docs/methods.md:779    (total 643,910 tiles across all scales).
  docs/methods.md:782    - 97.88 % of finest tiles have `boulder_area == 0` ...
  docs/methods.md:783-4  - ... mean `fractional_area` = 2.2 × 10⁻⁴, median = 0, P90 = 0,
                           P99 = 6.25 × 10⁻³, max = 0.269
  docs/methods.md:1035-6 The correlations are all weak in absolute terms (|ρ| ≤ 0.083) because of the
                         heavy zero-inflation: 97.9 % of finest tiles have `fractional_area = 0`, ...

  $ git log --format="%h %ad" --date=short -1 b9bc82a          -> b9bc82a 2026-05-22   (banner says 2026-05-25)
  $ git show 479688d -- docs/methods.md | grep '^+### '        -> +### 5.2 / 5.3 / 5.4 (vClaire 40-image dataset) / 5.5 / 5.6
  $ git log --format="%h %ad" --date=short -1 479688d           -> 479688d 2026-05-28   (AFTER the pinned commit)

  # measured from the on-disk label sidecars + parquets (finest scale, tile_size_px == 8)
  dataset     images: 9   {8: 488554, 16: 119944, 32: 28825, 64: 6587}   total 643910
  dataset_v2  images: 38  {8: 2700653, 16: 665794, 32: 161005, 64: 37315} total 3564767
  dataset     S=8 n=488554   zero_frac=0.9789  max_fa=0.2687  p99=6.250e-03  mean=2.179e-04
  dataset_v2  S=8 n=2700653  zero_frac=0.5020  max_fa=0.4362  p99=1.300e-01  mean=1.543e-02
  ```
- **Self-refutation attempted:** (a) *The banner discloses the snapshot, so the numbers are honestly
  dated* — this is the strongest objection and it fails on its own terms: the banner names a commit
  (`b9bc82a`, 2026-05-22) that **predates** content the document contains (`479688d`, 2026-05-28), so
  the pin is factually wrong and the "snapshot" is a mixture. (b) *`docs/index.md:42` has a blanket
  disclaimer* — "Commit references in these documents point at the commit at the time of writing …
  if a figure or value disagrees with the current cache, the current cache is authoritative." That
  covers value drift, but not a document asserting the wrong **cohort size** as "the current
  dataset", and it does not survive §5 having been migrated while §§6–8 were not. (c) *Is a v2
  successor planned, making this deliberately frozen?* — `docs/index.md:23-25` lists three planned
  documents (`data_release.md`, `fm_deployment.md`, `striping_artifact.md`); none is a v2 pipeline
  Methods doc, and `grep -n 'methods\.md' DECISIONS.md` returns only the 2026-05-22/23/28 update
  entries — nothing declares it frozen. (d) *Is this PASS 1's coverage note, not a finding?* — PASS 1
  explicitly listed this as **not checked** ("prose-level numerical agreement inside
  `docs/methods.md` … left for others"); this is that check, done. (e) *Is it R03?* — no; R03 is the
  0.25/0.50 m/px pixel-scale confound in the labels themselves.
- **Fix:** replace the banner with "**Scope: the v1 `priority10` 9-image sweep**, except §5
  (co-registration), which was updated to the vClaire v2 39-image cohort on 2026-05-28 (`479688d`).
  All results reported since 2026-05-29 use the v2 38-image cohort — see `docs/model_evidence.md`";
  then either re-run §6.7 / §7.4 / §8.2 / §8.4 against `dataset_v2` (the numbers above are the v2
  values) or label each of those sections `(v1 priority10)` in its heading. Drop or correct §7.4's
  zero-inflation causal claim, citing the `DECISIONS.md` 2026-06-13 refutation.

### docs-consistency-8 — "per-image AUC ≈ 0.43" is the Tier-2 abundance Spearman ρ mislabelled as an AUC, and it is the stated skill baseline in `DECISIONS`, in the only ACTIVE plan (twice), and in notebook 24 — read as an AUC it is *below chance*, and the real value is 0.79
- **Severity:** medium
- **Liveness:** live-active-plan (it is the pre-declared instrument for leg 4 of `PLAN_RegionalMap`, the only ACTIVE plan) + the record of the striping-mitigation adjudication rule
- **Confidence:** high
- **Where:** `DECISIONS.md:4101`; `PLAN_RegionalMap.md:75`, `PLAN_RegionalMap.md:192`;
  `notebooks/_build_24.py:531` and its generated `notebooks/24_regional_map.ipynb:804`.
  Contradicted 23 lines later by `DECISIONS.md:4124`.

`DECISIONS.md:4101` (the 2026-06-18d striping-cause verdict) declares the mitigation adjudication
rule as "**LOIO skill preserved (per-image AUC≈0.43)**". The very next entry measures that quantity:
"baseline **median per-image AUC 0.790** / pooled PR 0.777" (`:4124`). 0.43 is not any per-image AUC
in the repo — it is the Tier-2 **abundance Spearman ρ** (`DECISIONS.md:3491` `mlp_reg` fa **0.431**;
`:3744` "rho~0.43 med"; `:3929` "the ~0.43 per-image ceiling is the 5 m/px CTX **magnitude** floor").
Commit `5016275` then propagated the mislabel verbatim into `PLAN_RegionalMap.md` (§2 leg-4 caveat and
§5 figure 4) and into notebook 24, where leg 4 — the *only* honest truth anchor for the shipped
regional map — is specified as "held-out predicted abundance vs BoulderNet detections … (per-image
AUC ≈ 0.43, pooled ρ)". Listing "pooled ρ" separately in the same parenthesis makes clear the writer
did not mean ρ by "AUC".

- **Failure scenario:** the session that finally runs leg 4 computes a per-image AUC on the held-out
  LOIO predictions, gets ≈0.79, and either (a) concludes the map massively over-performs its
  pre-declared expectation, or (b) concludes the pre-declared number was measured differently and
  goes hunting for a 0.43-AUC configuration that never existed. Either way the pre-registration is
  useless. Separately, an external reader of `PLAN_RegionalMap.md:75` — the paragraph whose whole
  purpose is to state the *honest* skill anchor — reads that the held-out model scores 0.43 AUC, i.e.
  **worse than a coin flip**, flatly contradicting the project's own headline median per-image AUC of
  0.7865 (`docs/model_evidence.md:28`).
- **Evidence:**
  ```
  DECISIONS.md:4101   step = prototype on one tile, re-score, and adjudicate by **LOIO skill preserved (per-image AUC≈0.43)
  DECISIONS.md:4102   + THEMIS/TES thermal ρ ideally up**.
  DECISIONS.md:4124   baseline median per-image AUC **0.790** / pooled PR **0.777**; A1 **0.766** / **0.771**. **Δ median

  PLAN_RegionalMap.md:75    > by a model trained without it; per-image AUC ≈ 0.43, pooled ρ). Leg 4 therefore reuses those
  PLAN_RegionalMap.md:192   cohort (per-image AUC ≈ 0.43, pooled ρ). NOT the all-data map at a training site (in-sample;
  notebooks/_build_24.py:531   cohort (per-image AUC ≈ 0.43). *NOT* the all-data map at a cohort site like `ESP_017355_2260`:

  DECISIONS.md:3491   | **mlp_reg (1-stage)** | **0.431** | **0.386** | 0.223 | 0.202 |     <- per-image Spearman rho
  DECISIONS.md:3929   ... The ~0.43 per-image ceiling is the 5 m/px CTX magnitude floor ...
  ```
- **Self-refutation attempted:** (a) *Is there some real per-image AUC ≈ 0.43?* — searched: the
  closest reported per-image AUCs are v1 Tier-1 median 0.61 (`docs/modeling_results.md:1178`) and the
  frozen recipe's 0.7865 / A1's 0.766 / baseline 0.790; `grep -n '0\.43' DECISIONS.md` returns only
  Spearman/η²/CV values. Individual *images* dip below 0.5 (ESP_046328_2180 at 0.344 pre-FM), but no
  cohort-level baseline is 0.43. (b) *Did the mislabel change any measurement?* — no: the A1 gate at
  `:4113-4127` and every later skill gate use 0.790 / pooled PR, so no reported number is wrong. That
  caps this at medium rather than high. (c) *Is leg 4 perhaps about the abundance regressor, so ρ is
  the right metric and only the word "AUC" is wrong?* — yes, and that is exactly the claim: the value
  is right for ρ and the metric name is wrong, in four places, one of which is a pre-registration.
  (d) *Already filed?* — no: R37 covers README/SHERLOCK pre-abort staleness; `grep -rn '0\.43'` over
  `docs/review_2026-07-31/*.md` and `docs/CODE_REVIEW_2026-07-31.md` finds no related entry.
- **Fix:** one edit across four files: `per-image AUC ≈ 0.43` → `per-image Spearman ρ ≈ 0.43 (the
  Tier-2 abundance rank metric; the classifier's median per-image AUC is 0.79)`. Regenerate notebook
  24 from `_build_24.py` in the same change so the committed `.ipynb` matches.

### docs-consistency-9 — `docs/model_evidence.md` §8 still lists the calibration layer and the abundance product as future work; both shipped 2026-06-16/17, and the shipped abundance raster is produced by a *different* architecture than the one §8 describes
- **Severity:** medium
- **Liveness:** live-shipped (the reader-facing evidence writeup; `PLAN_Calibration.md:350` lists updating it as a deliverable of a stage `ROADMAP.md:35` marks SHIPPED)
- **Confidence:** high
- **Where:** `docs/model_evidence.md:267-275`, `:298-320`, `:340-345`; vs
  `PLAN_Calibration.md:324-328`, `:350`, `:390-391`; `scripts/map_region.py:27-30`;
  `src/mapping.py:202-211`; `ROADMAP.md:35`

`docs/model_evidence.md` was last written 2026-06-14 (`3a66f53`). Two days later `a09f06b`
("Calibration Stage 1: CalibrationLayer + one-model abundance, wired into the map") shipped the
calibration layer, and `PLAN_Calibration.md:324-328` / `:390-391` resolved the architecture the
opposite way from §8: **"One-model simplification is the default abundance path. No separate Tier-2
head: the abundance map is `quantile_match(P(rich) → fa)`."** The 26-tile regional map then shipped
`<tile>_abundance.tif` on that path. §8 still tells the reader the opposite — that Tier-2 is a
`mlp_reg` regressor, that "absolute area-fractions **await** that calibration layer", and that
"what remains" includes "(ii) a two-sided de-compression calibration layer". `PLAN_Calibration.md:350`
explicitly listed "update `docs/model_evidence.md` §8" as a Stage-1 deliverable; Stage 1 is marked
SHIPPED and that item was never done.

- **Failure scenario:** a reviewer reads §8 and concludes (i) the project has no calibrated absolute-
  abundance product, and (ii) any abundance figure they see comes from a regression model with
  per-image ρ ≈ 0.43. Both are wrong for the shipped map: the calibrated abundance raster exists, and
  it is a **monotone quantile-match of the rich/poor classifier probability**, so it carries exactly
  the classifier's ranking and no independent magnitude estimate — a materially different claim about
  what the numbers on the map mean, and the one caveat §8 does not state.
- **Evidence:**
  ```
  docs/model_evidence.md:268-9  Tier-2 has a clear winning candidate but
                                is **not yet frozen or productised**: a single-stage three-seed MLP *regressor* ...
  docs/model_evidence.md:319-20 ... **use Tier-2 for relative
                                abundance and ranking today; absolute area-fractions await that calibration layer.**
  docs/model_evidence.md:340-2  **What remains (the to-do).** (i) Freeze + productise the single-stage `mlp_reg`
                                into a deployable regressor head; (ii) a two-sided de-compression calibration layer
                                (isotonic / quantile remap, or a tail-weighted loss); (iii) THEMIS overlap validation.

  PLAN_Calibration.md:324-6   3. **One-model simplification is the default abundance path.** No separate Tier-2 head:
                                 the abundance map is `quantile_match(P(rich) -> fa)` ...
  PLAN_Calibration.md:350        layer; re-render the regional/headline maps raw-vs-calibrated; update `docs/model_evidence.md`
  PLAN_Calibration.md:351        §8 + `DATA_DICTIONARY` if output columns change.
  ROADMAP.md:35                  ... Stage 0 + Stage 1 **SHIPPED** (`CalibrationLayer`, a09f06b) ...
  scripts/map_region.py:29       <tile>_abundance.tif  fractional_area (qmatch)               (omitted with --raw)

  $ git log --date=short --format='%h %ad' -1 -- docs/model_evidence.md   -> 3a66f53 2026-06-14
  $ git log --date=short --format='%h %ad %s' -1 a09f06b                  -> a09f06b 2026-06-16 Calibration Stage 1 ...
  ```
- **Self-refutation attempted:** (a) *Is §8's regressor statement still literally true?* — yes, and I
  scoped the finding accordingly: `mlp_reg` was never productised and
  `PLAN_Calibration.md:327-328` keeps banking a dedicated head as a documented option. The false
  parts are the calibration layer being future work, and the absence of any mention that a shipped
  abundance product exists on the qmatch path. (b) *Maybe the doc is dated so drift is expected* — it
  carries no date banner at all (unlike `docs/methods.md`), and `docs/index.md:42`'s blanket
  disclaimer is about values disagreeing with the cache, not about a to-do list whose items shipped.
  (c) *Is this R09?* — no; R09 is the `recipe_hash` collision and copied metrics on
  `deployable_f_center`. (d) *Is this PASS 1's `-6`?* — no; `-6` is that `docs/index.md` omits the
  file, not that its §8 is stale.
- **Fix:** in §8 replace "not yet frozen or productised" / "await that calibration layer" with the
  shipped state: "The **`CalibrationLayer`** shipped 2026-06-16 (`a09f06b`) and the 26-tile regional
  map emits a calibrated `*_abundance.tif`. Its abundance is **not** the `mlp_reg` regressor: the
  default path is a global quantile-match of the *classifier's* `P(rich)` onto the `fractional_area`
  marginal (`PLAN_Calibration.md` §Stage-1 pt 3), which fixes the marginal at a ~0.02 ranking cost and
  carries no magnitude information beyond the classifier's ranking. Banking a dedicated regressor head
  remains an option." Then reduce "what remains" to (i) the regressor option and (iii) quantitative
  THEMIS TI (leg 1 is done and weak, ρ ≈ +0.07).

### docs-consistency-10 — The only ACTIVE plan's final deliverable is named two different things and neither exists; its declared figure prefix does not match the figures actually produced
- **Severity:** low
- **Liveness:** live-active-plan
- **Confidence:** high
- **Where:** `PLAN_RegionalMap.md:197`, `PLAN_RegionalMap.md:262` (phase 6) vs `docs/index.md:24`;
  `reports/figures/`

`PLAN_RegionalMap.md:197` declares the plan's output as "`docs/regional_validation.md` +
`reports/figures/regional_*` (committed PNGs)", and §9 phase 6 is "the 5 figures +
`docs/regional_validation.md`". `docs/index.md:24` instead queues the same writeup under a different
name, `fm_deployment.md`, described as covering "the 26-tile circum-Chryse regional map with its
MOLA/THEMIS validation legs (PLAN_RegionalMap.md, notebook 24)". Neither file exists, and the figures
that *have* been produced for those legs are committed under the `24_*` notebook prefix
(`24_leg1_colocation.png`, `24_region_context_mola.png`, `24_region_products.png`, …) — there is no
`reports/figures/regional_*` at all.

- **Failure scenario:** the session that executes phase 6 writes `docs/regional_validation.md`,
  leaving `docs/index.md`'s queued `fm_deployment.md` entry permanently unresolvable (or vice versa),
  and looks for `reports/figures/regional_*` inputs that were never produced under that name — a
  small but real dead end at the last step of the repo's only open workstream.
- **Evidence:**
  ```
  PLAN_RegionalMap.md:197  Output: `docs/regional_validation.md` + `reports/figures/regional_*` (committed PNGs).
  PLAN_RegionalMap.md:262  | **6** | the 5 figures + `docs/regional_validation.md` | — |
  docs/index.md:24         - `fm_deployment.md` — *(queued 2026-07-09)* Methods + results for everything after the
                             2026-06-04 writeups ... and the 26-tile circum-Chryse regional map with its MOLA/THEMIS
                             validation legs ([PLAN_RegionalMap.md](../PLAN_RegionalMap.md), notebook 24).

  $ ls docs/*.md | grep regional   -> (none)
  $ ls reports/figures | grep -i regional -> (none)
  $ ls reports/figures | grep '^24_'      -> 24_coverage_planning.png 24_leg1_colocation.png
                                             24_region_abund_vs_ctx.png 24_region_binary_on_mola.png
                                             24_region_context_mola.png 24_region_ctx_raw.png
                                             24_region_extent.png 24_region_mosaic.png 24_region_products.png
  ```
- **Self-refutation attempted:** (a) *The document is simply not written yet (phase 6 is open), so
  its absence is not a defect* — agreed, and that is why this is low; the defect is the **two names**
  for one deliverable plus the falsified `regional_*` prefix, not the absence. (b) *Maybe
  `fm_deployment.md` is broader and `regional_validation.md` is a subset* — `docs/index.md:24`
  explicitly folds "the 26-tile circum-Chryse regional map with its MOLA/THEMIS validation legs" into
  `fm_deployment.md`, i.e. the same content, and no doc reconciles the two names. (c) *Already
  filed?* — `grep -rn regional_validation docs/review_2026-07-31/ docs/CODE_REVIEW_2026-07-31.md`
  returns nothing.
- **Fix:** pick one name in `docs/index.md:24` and `PLAN_RegionalMap.md:197,262` (the index's
  `fm_deployment.md` is the broader and later-dated choice), and change §5's figure clause to
  "`reports/figures/24_*` (committed PNGs, notebook 24)".

## Refuted by my own check

**PASS 1:**

- **"README/SHERLOCK cite scripts or flags that don't exist."** Refuted by an automated pass over
  every tracked `.md`: each `scripts/*.py` and `scripts/probes/*.py` invocation was matched against
  that script's own `add_argument` set. `README.md` and `SHERLOCK_RUN.md` are **100 % clean**. Only
  two hits repo-wide, both in closed plans: `PLAN_NewDetections.md:369` uses
  `run_stage2.py --all` (`run_stage2.py` takes a positional `obs_id` + `--config` only — the same
  line's own comment says "or sweep_stage2.py"), and `PLAN_Stage5b.md:181` uses
  `sweep_binary.py --thresholds` (the flag is `--targets`). `PLAN_modeling.md:401-403,482` and
  `PLAN_ModelUsability.md:316` name `train_baseline.py` / `src/evaluate.py` / `src/models/*.py` /
  `infer_ctx_region.py`, which are pre-implementation plan sketches, not drift.
- **CLAUDE.md:23-24 "The CTX mosaic CRS is Mars_2015 equirectangular clon_0 (sphere 3396190 m)."**
  Looks like it contradicts `README.md:540-544`, `dataset/DATA_DICTIONARY.md:65` and
  `docs/methods.md:349-353`, which all say the Murray Lab mosaic is **oblate** (1/f = 169.894). But
  `src/labeling.py:461-467` records the runtime verification: PROJ's `eqc` uses the shared
  semi-major radius, so sphere and oblate definitions give **0.000 m** displacement at our
  coordinates (DECISIONS 2026-05-28). Loose phrasing, zero numerical consequence, and the precise
  statement is in three other docs. Not filed.
- **"9 feature families" (README:59, README:414, ROADMAP:23, PLAN_Stage4b.md:4, DECISIONS:584) vs 8
  entries in `features.enabled`.** Real count mismatch, but self-correcting: `README.md:414`
  enumerates the 8 on the same line, and the "9th" is `PLAN_Stage4b.md` §3.5.1 higher-order moments,
  which was folded into `intensity_stats` rather than made its own family. No consequence.
  *(PASS 2 correction to the reasoning, not the verdict: `docs/methods.md:858-995` resolves it
  differently — it enumerates the nine as (a)–(i) where **(i) is "Context patches"**, which is the
  9th "family" and is configured under `features.context_patch`, not `features.enabled`. Verdict
  unchanged: no consequence.)*
- **ROADMAP:20 "`lv.solve_offsets*` stays in the codebase unused."** Not literally true —
  `scripts/f_region_stagec.py:207,416`, `src/fgates.py:170,172`, `src/leveling.py:455,481` and 8
  tests call it. It is true in the intended sense (no live-shipped caller; every caller is F-build
  machinery from a closed plan). Not worth a finding.
- **ROADMAP:18's abort numbers.** Recomputed from `reports/figures/fbuild_abort_level_vs_labels.csv`:
  sd(log₁₀) with `ddof=0` gives mosaic **0.1702**, h1only 0.3281, resid **0.3710**, pfree **0.5317**,
  full 0.4119 → 0.170 / 0.371 / 0.532 and the "1.13×/1.62×/1.26× worse than unleveled" ratios all
  reproduce exactly, as do the 5.1× / 32.5× / 189.6× spreads. The published numbers are right.
- **The frozen-recipe headline (0.7832 / 0.948 / 0.7865)** is stated identically in
  `DECISIONS.md:3430,3457-3458`, `docs/model_evidence.md:28,107,111,115` and both
  `models/deployable*/recipe.json` cards. No contradiction (the *card-copying* problem is R09).
- **Cohort counts.** `hirise_40_vclaire.csv` = 39 rows; `loio_nfold` = 38 folds / 38 obs; 38 label +
  38 feature parquets + 38 `_P96.npz` stores; `recipe.json n_train_images` = 38 — all consistent with
  `README.md:55,60,88-92`.
- **`PLAN_RegionalMap.md:281` cites `cache_v2/thermal/mola_dem_region.tif`; the real path is
  `cache_v2/validation/`.** Real, but it sits inside a dated historical `UPDATE 2026-06-17b` note and
  `src/validation_retrieve.py:34` (`VALIDATION_SUBDIR = "validation"`) plus
  `scripts/fetch_validation_data.py:66` are unambiguous. Path-only drift, no consequence.
- **`SHERLOCK_RUN.md:5` and `:661` still say "7-tile block"; `scripts/map_region.py:5` too.**
  Superseded by the same doc's §C4 and by `BLOCK_TILES` (26). Self-correcting within one read.
- **README test counts** — `README.md:88` "366 pytest pass" and `README.md:433` "tests/ # 125 tests"
  vs `pytest --collect-only -q` = **511 collected**. Stale but consequence-free, and two different
  stale numbers in one file make it obvious.
- **`README.md`'s `src/` Layout block (`:398-424`) lists 12 modules; `src/` has ~20** (missing
  `mapping.py`, `striping.py`, `calibration.py`, `fgates.py`, `reliability.py`,
  `spatial_features.py`, `colour.py`, `ctx_source_illumination.py`, `ctx_edr.py`,
  `validation_retrieve.py`, `stage7d_pooled.py`). Incompleteness of an illustrative tree, not a false
  claim.

**PASS 2:**

- **`.gitignore:49` excludes `reports/map_region/` — the shipped deliverable's 26 per-tile provenance
  JSONs are untracked while 30 JSONs of the **aborted** `reports/map_fbuild/` are tracked.** PASS 1
  left this as a pointer for the `notebooks` reviewer; closing it here. The exclusion is **deliberate
  and documented in the file itself**: `.gitignore:46-48` reads "Regional-map rasters + run
  checkpoints (475 MB; regenerable on Sherlock — the `*.tif` rule already blocks the rasters, **this
  also keeps the per-tile .json checkpoints out**)". The asymmetry is only that `reports/map_fbuild/`
  was never added; since that arm is aborted, its 30 tracked JSONs are harmless clutter. Not a
  finding. (`models/` being gitignored was already refuted in `notebooks.md:227`.)
- **`docs/model_evidence.md:294` "rich/poor `meaningful_auc` = 0.78 (matching the dedicated Tier-1
  classifier)" looked like a PR-AUC-vs-ROC-AUC conflation** (the Tier-1 headline 0.7832 is a *pooled
  PR-AUC*). It is not: `DECISIONS.md:3517-3521` states the comparison explicitly against "the frozen
  Tier-1 classifier's **per-image AUC 0.7865**", and `src/modeling/evaluate.py:366` computes
  `meaningful_auc` as an ROC-AUC on `y_true > 1e-2` — same family. The only slack is mean-of-folds
  (0.784) vs median-of-folds (0.7865), which is R26's territory. Not filed.
- **`docs/model_evidence.md:29` Tier-1 row (0.5651 / 0.771 / 0.681) vs the banked
  `tier1_ref` (0.4840 / 0.607 / 0.6631) in
  `models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/verdict.json`.** Not a contradiction: the
  doc's row is the S=64 Tier-1 *product* (`DECISIONS.md:3189` = 0.5651 / 0.771 / 0.6806), the
  verdict's is the same-scale S=32 reference (`DECISIONS.md:3110`, and `:3250-3258` explains the
  S=32 collapse). The doc labels its row "320 m (S=64)". Correct as written.
- **`coregistration.search_radius_m: 400` is in all three configs and read by no code** (0 hits in
  `src/`+`scripts/`), so no shift-magnitude bound is enforced. Real, but **already filed by the
  `geo-crs` reviewer** (`geo-crs.md:84,95,131-140`). Not re-filed.
- **`dataset/DATA_DICTIONARY.md:411` documents `split_hash` as SHA256 over
  `{name, kind, n_folds, stratification, manifest_obs_ids, folds}`, but
  `src/dataset.py:392-396` also hashes `n_folds_per_image`, `buffer_tiles`, `excluded_obs_ids`.**
  Real omission, but those three keys only exist on within-image schemes, so the documented formula
  reproduces the hash for every LOIO scheme; the drift that matters is `other-scripts-1`. Not filed.
- **`CLAUDE.md`'s invariant 4 "Prefer `/vsicurl/` range requests" vs the shipped
  `ctx_retrieve.mode: download_then_window`.** The config comment at `config.yaml:45-49` records the
  2026-05-22 decision and its reason (~140× slowdown), `config.yaml:40` marks `ctx_read` DEPRECATED,
  and `README.md:551-554` documents the real mode. CLAUDE.md says "prefer", i.e. advice inherited
  verbatim from `docs/build_spec.md:86-88` ("try `/vsicurl/` first; fall back to
  download-then-window"). Loose, not false.
- **`docs/methods.md:1146-1147` "y_*_fold{k}.parquet — the label side (12 columns …)" vs the 17
  columns on disk.** 4 transforms + 3 base stats + 5 bounds = 12, so the count is coherent and the
  error is only the trailing "and the row key" (the 5 key columns are additional). Subsumed by
  `docs-consistency-7`; not filed separately.
- **`docs/build_spec.md` vs the code.** Read in full. It is scrupulously annotated as the verbatim
  historical spec, and every place where reality diverged carries a pointer (§5 "the live config is
  `config.yaml` … the original spec's illustrative block"; §8's note that `src/` has grown; §10/§11
  marked DONE/RESOLVED). Found nothing to file.
- **The `H1`–`H6` label collision** between `docs/modeling_results.md:1189-1208` ("H1 metric, H2
  target, H3 per-image heterogeneity, H4 multiplicative hurdle, H5 texture floor") and the striping
  PHASE-2 docket's `H1`–`H6` (centering / nuisance subspace / consistency head / leveling). Genuinely
  ambiguous across documents, but each is scoped inside its own plan and neither is cited from the
  other; too close to a naming nit to file.

## Verified clean

**PASS 1:**

- **Every `scripts/*.py` and `scripts/probes/*.py` invocation in every tracked `.md`, checked
  flag-by-flag against the script's `argparse`.** README and SHERLOCK_RUN are clean; see the refuted
  section for the two closed-plan hits.
- **Every repo-relative path referenced in every tracked `.md`** (`reports/`, `models/`, `dataset*/`,
  `cache*/`, `notebooks/`, `src/`, `scripts/`, `tests/`) was existence-checked. **All doc-referenced
  figures under `reports/figures/` exist.** The only misses are listed above (plus already-filed
  **R06** `reports/map_a1/`).
- **`dataset/DATA_DICTIONARY.md` feature schema vs the emitted parquet.** Programmatic diff against
  `dataset_v2/features/ESP_017355_2260.parquet` (60 columns): **zero** emitted-but-undocumented and
  zero documented-but-missing columns. Label parquet (18 cols) and `y_*_fold{k}.parquet` (17 cols)
  also match §"Stage 4" / §"Stage 5". (One nit not worth filing: the packaged X carries
  `config_hash_feat`, while `:422-424` says `config_hash` is "kept as a join key".)
- **`SHERLOCK_RUN.md` step-2 snippet** (`cfg["ctx_mosaic"]["url_template"]`, `cfg.cache_dir`) works —
  `src/config.py:76-90` provides both accessors, matching `scripts/run_stage2.py`'s usage.
- **`run_region_array.sbatch`'s hardcoded `TILES` array == `scripts/map_region.py:76-83`
  `EXPANSION_TILES` == the 19-tile list pasted at `SHERLOCK_RUN.md:285-289`** — all three in sync,
  and `BLOCK_TILES` is 26, matching README:313's "`--all` # the 26-tile block".
- **Shapefile discovery** (`CLAUDE.md` invariant 7) — `src/manifest.py:23` `SHAPEFILE_GLOB =
  "*-mask-nms.shp"`, resolved by glob at `:82`, exactly as documented.
- **README's Stage-4 eligibility gotcha (`:545-550`)** matches `src/labeling.py:65-67,276-279`
  (`ELIGIBILITY_RULE = "coverage_equals_one"`, `eligible = (mask_min == 1)`), and README's
  `download_then_window` claim (`:551-554`) matches `config.yaml:50` / `config_v2.yaml:75`.

**PASS 2 (independent additions):**

- **`PLAN_RegionalMap.md:167-169`'s output contract vs the code.** "single-band float32, 160 m/px,
  `NaN`=nodata/masked", `*_prob.tif` (calibrated P(rich)), `*_abundance.tif` (fa qmatch),
  `*_prob_raw.tif` (uncalibrated, QA) — matches `scripts/map_region.py:27-30,142,176-180,234-237`
  and `src/mapping.py:181-193` (`dtype="float32"`, `nodata=np.nan`) exactly, including the
  `--raw` omission semantics.
- **`src/modeling/evaluate.py`'s three pointers into `docs/modeling_results.md`** (`:250` and `:327`
  → §11.4; `src/modeling/gbm.py:378` → §3.1; `src/modeling/cnn.py:428` → §3.3). §11.4 exists at
  `docs/modeling_results.md:1165` and does say what the docstrings claim (the cross-image-mean-AUC
  critique and the `fa_gt_1e-2` reframing). Not stale.
- **`dataset/DATA_DICTIONARY.md:165-166`** `binary_by_area` / `binary_by_count` documented as `>=`
  → `src/labeling.py:390-391` uses `frac >= binary_area_threshold` / `bc >= binary_count_threshold`.
  Thresholds 0.005 / 1 match both configs. Correct.
- **`DATA_DICTIONARY:262-264`** shadow / strict / bright DN offsets (mode −20 / −35 / +30) match
  `config.yaml`'s `shadow_offset_dn: 20`, `strict_offset_dn: 35`, `bright_offset_dn: 30`, and
  `docs/methods.md:922-926` states the same three. `:303`'s "up to log(8) ≈ 2.08" is right.
- **`DATA_DICTIONARY:230-236` GLCM provenance** (8/16/16/32 levels; d=[1] at S=8 else [1,2,3]; four
  angles rotation-averaged; six properties) matches `config.yaml`'s `levels_per_scale`,
  `distances_per_scale`, `angles`, `angle_average`, `properties` exactly, and
  `docs/methods.md:879-887` restates it consistently.
- **Config-key liveness sweep.** Every key in `config.yaml` was grepped against `src/` + `scripts/`.
  All are read except `coregistration.search_radius_m` (0 hits — already filed by `geo-crs`),
  `hirise_decimation_mpp` (1 hit, the required-key check — already filed as `invariants-4`) and the
  two keys the file itself marks DEPRECATED (`ctx_read`, `labeling.features`/`context_patch_px`).
  No other documented knob is dead.
- **The frozen-recipe metric identity.** `pooled 0.7832` is genuinely a **pooled PR-AUC**
  (`DECISIONS.md:3378,3455-3457` table header; `models/fang_probe/fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2/verdict.json`
  key `pooled_pr_auc = 0.7832132925969041`, `prec_at_5 = 0.9475776`, `med_auc = 0.7865094`,
  `pos_rate = 0.3598273`). `docs/model_evidence.md:22,26-28,107`'s "pooled PR-AUC 0.78 against a 0.36
  base rate" is exactly right.
- **`ROADMAP.md:19`'s H1/H2/H3/H4 η² chain (0.179 → 0.081 / 0.277 → 0.128; A1 0.141; H2 0.131; H3
  λ=100 0.035; H4 0.128 → 0.0505; edge-CV 0.074 → 0.035)** cross-checks line-for-line against
  `PLAN_StripingArtifact.md:89,129,183-185,219-220,231-237,249-254`, `PLAN_H4_Leveling.md:141`,
  `README.md:28-36` and `PLAN_FBuild.md:389` (which itself flags the pilot-crop-vs-build scale mixing).
  No contradiction between docs.

## Coverage note

**PASS 1 — read in full:** `README.md`, `ROADMAP.md`, `CLAUDE.md`, `dataset/DATA_DICTIONARY.md`,
`docs/index.md`, `PLAN_RegionalMap.md` §10, `SHERLOCK_RUN.md` Parts A–D + Part J (rest skimmed by
heading), `docs/modeling.md` §11, `run_region_array.sbatch`, `.gitignore`, and the argparse blocks of
every script a doc invokes.

**PASS 1 — automated (whole-repo):** doc→script flag validation and doc→path existence validation over
all 34 tracked `.md` files (scripts written to the scratchpad, read-only). `DECISIONS.md` was grepped
by term only (`0.1222`, `0.087`, `0.827`, `min_size_m`, `loio_38fold`, `prec@5%`, `0.7832`,
`9 feature famil`, `solve_offsets`), never read linearly. Numerical checks were run only against
committed `reports/figures/*.csv|json` and `dataset*/**/*.parquet|json`.

**PASS 2 — read in full:** `docs/model_evidence.md` (391), `docs/build_spec.md` (336),
`dataset/DATA_DICTIONARY.md` (471), `PLAN_RegionalMap.md` (345, all sections), `docs/index.md`,
`ROADMAP.md`, `config.yaml`, `docs/methods.md` §§1, 6.7–6.8, 7, 8.2–8.4 (rest by heading + grep),
`PLAN_Calibration.md` §Stage-1 design + §6–7, `docs/modeling_results.md` §11.4–11.6,
`src/modeling/evaluate.py:250-430`, `src/mapping.py:181-215`, `scripts/map_region.py` (output path),
`.gitignore`.

**PASS 2 — measured (read-only, over committed data):** v1-vs-v2 tile counts from all 47
`dataset*/labels/*.json` sidecars, and finest-scale zero fraction / mean / P99 / max of
`fractional_area` from all 47 `dataset*/labels/*.parquet` (the basis of `docs-consistency-7`);
packaged X/y column counts for all 7 packaged schemes; `models/fang_probe/…/verdict.json`.
`DECISIONS.md` grepped by term only (`0.43`, `0.5651`, `0.771`, `0.484`, `0.081`, `0.128`, `0.141`,
`0.179`, `methods.md`, `model_evidence`, `search_radius`, `qmatch`, `isotonic`, `ECE`), never read
linearly.

**Could NOT check / left for others:**
- Prose-level numerical agreement inside `docs/compositional*.md`, `docs/classification_slimmer.md`,
  `docs/modeling_slim.md` and `docs/w2_litreview.md` — PASS 2 closed the `docs/methods.md` and
  `docs/model_evidence.md` half of PASS 1's declared gap, but not these four. Their quantities are
  produced by notebooks/probes (the `notebooks` area) and their inferential validity is covered by
  `stats-fallacies-4`.
- `SHERLOCK_RUN.md` Parts E–I (39 KB doc) were skimmed by heading only in both passes; **R37** covers
  their pre-abort framing but not their internal command-level detail beyond PASS 1's automated
  flag/path validation.
- `.ipynb`-vs-`_build_NN.py` drift and stale committed notebook outputs — the `notebooks` area's
  remit. One concrete hand-off: `docs-consistency-8`'s mislabel is present in **both**
  `notebooks/_build_24.py:531` and the committed `notebooks/24_regional_map.ipynb:804`, so fixing it
  requires a regenerate-and-execute, not just a source edit.
- **Extends R29, not re-filed:** `docs/methods.md:483-489` asserts "The distinction between 'boulder
  absent' (observed and empty) and 'boulder unobserved' (no HiRISE coverage) … we propagate it
  through the pipeline explicitly" — R29 shows the Stage-3 shift moves the polygons but not the
  coverage mask, so a ~1-tile strip inside every swath edge is zero by construction. Whoever fixes
  R29 should fix that sentence in the same change.
- **Extends R22, not re-filed:** `dataset/DATA_DICTIONARY.md:417-419` recommends the streaming
  `iter_train_batches` / `iter_test_batches` API "for the 50-200+ image case"; R22 shows it lacks
  `package_split`'s kind dispatch and has zero callers. `docs/methods.md:1155-1160` makes the same
  recommendation ("the recommended path once the manifest grows past ~50 images").
