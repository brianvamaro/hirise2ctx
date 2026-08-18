# Artifact recovery plan — what survives losing the gitignored trees

**Status: this is a stopgap for isolation criterion 5, not a substitute for it.** Brian is getting an
external drive; until then this records exactly what could be reconstructed, what it would cost, and —
the part that matters — **what could not be reconstructed at all**.

Written 2026-08-06, all sizes measured. `git` tracks none of this: `cache*/`, `dataset*/`, `models/`,
`reports/` and the detection roots are all ignored.

---

## Tier 0 — NOT RECOVERABLE. No plan covers these.

| what | size | why it cannot be regenerated |
|---|---|---|
| `../hirise_priority10_detections/` + `../hirise_40_vClaire/` | 0.01 + **4.17 GB** | **The ground truth.** BoulderNet detection shapefiles — the input every label, model, metric and map in this project derives from. They live *outside the repo* (`detections_root` in both configs) and were outside every artifact manifest taken during this review, so nothing here has ever been checking them. Re-running BoulderNet is a separate project, and the vClaire set came with a specific detector config (`vclaire_40img_ct010_ss256_downscaled_2026-05-28`). |
| ~~`dataset/` (v1)~~ | ~~5.0 GB~~ | **EXPENDABLE — Brian, 2026-08-06.** Not backed up, not rebuilt, and losing it is accepted. It is non-reproducible (pre-2026-06-10 y-sign fix; re-running Stage 4 yields *different* labels, which was the point of keeping it), so the cost is that every v1 measurement becomes **unre-verifiable** — including R81's 236–493 m label offset, R92/R97's "v1 matches a step-8 recompute 8/8", and the 2026-08-04 incident's `max\|Δfa\|` 0.115. Those *conclusions* survive in `DECISIONS.md` and the review register, which git tracks; what dies is the ability to re-derive them. Accepted deliberately: v1 is superseded and nothing current reads it. |
| `reports/f_build`, `f_leg_b`, `f_timing`, `f_region_logits`, `f_stagec` | **20.2 GB** | Output of the aborted 907-frame F build (~333 CPU-h on Sherlock). The programme is CLOSED, so nobody will re-run it — meaning this is the only surviving evidence for the HARD ABORT verdict. `f_timing` (10.4 GB) is the most droppable: its conclusion (22 min/frame) is already in DECISIONS. |
| `reports/figures`, `reports/map_region` | 0.98 GB | Regenerable in principle, but from artifacts that are themselves about to change in the rebuild — so the *current* figures are the record of the current claims. |
| `dataset_v2/packaged/loio_nfold_{ctx_illum,nbr_s5}` | (within the 78 GB) | Pre-sign-fix targets kept deliberately as documented drift (R82). Regenerating them destroys what they document. |

**≈25 GB after dropping v1**, of which **~6.3 GB is small and precious**: the detections (4.18 GB),
trained models minus `pretrained` (1.1 GB), and current figures + `map_region` (0.98 GB).

## Tier 1 — RE-DOWNLOADABLE from external sources. ≈64 GB.

| what | size | source |
|---|---|---|
| `cache*/ctx_tiles/*.zip` (24 tiles) | 41.4 GB | Murray Lab, `https://murray-lab.caltech.edu/CTX/V01/tiles/MurrayLab_GlobalCTXMosaic_V01_{tile}.zip` — the template is `ctx_mosaic.url_template` in both configs. `ctx_retrieve.ensure_tile_cached` refetches on a cache miss, including the zero-padded-name 404 retry. **Sidecars (`{tile}.json`) rebuild automatically** from the zip header on first use. |
| `cache*/hirise_jp2/*_RED.JP2` (46) | 19.8 GB | PDS. Per-image URLs are the `JP2_URL` column of `hirise_priority10.csv` / `hirise_40_vclaire.csv` (both tracked in git). `hirise_imagery.ensure_jp2_local` downloads on demand. |
| `cache_v2/validation/` | 2.4 GB | `planetarymaps.usgs.gov` MOLA DEM + THEMIS night IR; URLs in `config_v2.yaml` `validation_rasters`. `scripts/fetch_validation_data.py`. |
| `cache_v2/craters/` | 0.07 GB | Robbins crater database. |
| `models/pretrained/` | 0.34 GB | Fang ViT checkpoint. **Verify the exact provenance before relying on this line** — the download URL is not recorded in the repo. |
| `cache*/pds_index`, `pds_labels` | 0.03 GB | PDS; `src/pds_labels.py`. |

**Cost:** bandwidth and a day of wall clock, no compute. This tier is genuinely covered by a plan.

## Tier 2 — REGENERABLE by re-running the pipeline. ≈84 GB.

Needs Tier 0 (detections) and Tier 1 (imagery) intact, plus the code at the right commit.

| what | size | how |
|---|---|---|
| `cache*/reprojected_detections`, `ctx_windows`, `hirise_decimated`, `coregistration` | 4.3 GB | Stages 1–3. Hours. |
| `dataset_v2/labels`, `features`, `context_patches` | ~19 GB | Stages 4 / 4b. |
| `dataset_v2/fang_embeddings*` | ~9 GB | GPU embedding runs. |
| `dataset_v2/packaged` | 52.5 GB | Stage 5. Bulky but cheap and deterministic. |
| `models/` trained heads and sweeps | ~1.1 GB | Training runs. |

**Caveat that makes this weaker than it looks:** re-running today's code does **not** reproduce today's
artifacts. R74, R27, R28 and R97 have all changed producer behaviour since these were written — see
[PENDING_REBUILD.md](PENDING_REBUILD.md). Tier 2 recovers *a* dataset, not *this* dataset. For anything
whose exact values are cited, treat Tier 2 as Tier 0.

---

## The honest summary

A re-download plan covers Tier 1 well and Tier 2 with an asterisk. It covers **Tier 0 not at all** —
and Tier 0 includes the detections the whole project rests on, which no manifest in this review has
ever been watching.

## When the drive arrives

> ✅ **THE DRIVE ARRIVED AND THE BACKUP IS DONE — 2026-08-18.** `D:\HiRISE2CTX Backup`, an independent
> USB device: **11,260 files / 125.55 GB**, verified 8/8 roots at 0 missing / 0 extra / 0 size
> mismatch. See `scripts/backup_artifacts.ps1` and DECISIONS 2026-08-18.
>
> **The prioritised subset below was not needed** — with 1,012 GB free there was no reason to triage,
> so the whole irreplaceable set went in one pass, including everything Tier 0 names. In particular
> the **4.18 GB of detections (`hirise_40_vClaire` + `hirise_priority10_detections`) are backed up**;
> they were the item this document called highest-priority and "no manifest in this review has ever
> been watching", and they are now covered under `external\` in the snapshot.
>
> This section is retained because the triage order remains the right one if a future snapshot is ever
> space-constrained.

**Deferred by Brian on 2026-08-06 — no backup runs until then.** *(Superseded — see above.)* The set
to copy, with v1 dropped:

```
detections (both roots)        4.18 GB   <- highest priority; nothing else can be rebuilt without it
models/ minus pretrained       1.10 GB
reports/figures + map_region   0.98 GB
                              ~6.3 GB    small Tier-0 set
reports/f_* (F programme)     20.2 GB    judgment call; f_timing 10.4 GB is the most droppable
dataset_v2                    78.3 GB    Tier 2, but see the asterisk above
                             ~105 GB     total
```

`dataset/` (5.0 GB) is **excluded by decision** — see Tier 0.

**Exclude `ctx_tiles` and `hirise_jp2` explicitly.** `cache_v2` reaches them by junction, so a naive
recursive copy of `cache/` follows it and duplicates 61 GB of re-downloadable archives.

C: has 600 GB free if an interim same-volume copy of the 6.3 GB set is ever wanted. It is **not**
disaster protection — one disk failure takes both — but it does defend against the failure that
actually happened here twice (2026-06-10 and 2026-08-04: a producer silently overwriting a live
artifact).

## Before the rebuild

Criterion 5 is not closed by this document. The rebuild overwrites Tier 2 and can touch Tier 0 if a
producer is pointed wrongly; the runtime write guard is **test-only** and does not cover scripts or
notebooks. Do not start it on the strength of a recovery plan.
