# docs

Long-form documentation that complements (rather than duplicates) the in-repo
README + DECISIONS + DATA_DICTIONARY. The repo-root files are written for
**code users**; the documents under `docs/` are written for **readers**,
including reviewers, collaborators, and the project's advising committee, who
care about how the results were produced but won't necessarily touch the code.

## Index

| Document | Audience | Scope |
|---|---|---|
| [build_spec.md](build_spec.md) | Code users / maintainers | The original Weeks-1–2 build specification, preserved verbatim when [CLAUDE.md](../CLAUDE.md) was streamlined into an operating manual (2026-06-20). Authoritative for the data-pipeline stage definitions; the project has since moved on (see [ROADMAP.md](../ROADMAP.md)). |
| [methods.md](methods.md) | Mixed / general scientific reader | Full data-pipeline Methods section in paper-Methods style: inputs → coordinate handling → co-registration → labels → features → cross-validation. Stops at the point where the modeler receives a packaged train/test dataset. ~20 pages of narrative + 9 embedded figures + 5 quantitative tables. |
| [modeling.md](modeling.md) | Mixed / general scientific reader | Methods companion to `modeling_results.md` — describes the modelling stage on top of the packaged dataset: targets, features (incl. Stage 6a/6b/6c extensions), variants, CV design, evaluation, reproducibility. Written 2026-06-02 at project wrap-up. |
| [modeling_slim.md](modeling_slim.md) | Mixed / general scientific reader | A simplified, reportable LightGBM model for predicting per-tile boulder abundance: 5 physically motivated features (shadow fraction + roughness), 36-image cohort, LOIO cross-validation. Used as the model writeup for the project report. Written 2026-06-03. |
| [modeling_results.md](modeling_results.md) | Mixed / general scientific reader | First-pass assessment of the Week 3 LightGBM + CNN baselines: what the headline numbers do and do not say, what signal exists in the model and where it falls short, and a short list of next experiments that would either resolve or close out the "is this approach working?" question. Embeds the five notebook 10 figures. Updated through Stage 6c (2026-05-31). |
| [compositional.md](compositional.md) | Mixed / general scientific reader | Stage 7 wrap-up — does boulder-rich HiRISE tile colour differ from boulder-poor surroundings, and if so, is the difference composition or dust? Methods + Results + Discussion for Stages 7.0 / 7a / 7c / 7d (including the shadow-masking refinement and per-image attribution table). Written 2026-06-02. |
| [compositional_slim.md](compositional_slim.md) | Mixed / general scientific reader | A higher-level reportable writeup of the compositional study: same data and numbers as `compositional.md`, but no Stage 7 terminology, no implementation gotchas, no crater-catalog cross-reference. Used as the science writeup for the project report. Written 2026-06-04. |

Planned future documents (not yet written):

- `data_release.md` — A short, citeable description of the released dataset version (manifest, ObsId list, pipeline commit hash, schema reference) for use when the data is shared externally.

## When to read what

- **You want to run or modify the code:** start at the repo-root [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md).
- **You want a chronological log of decisions and the reasons behind them:** read [DECISIONS.md](../DECISIONS.md).
- **You want the per-column output schemas:** read [dataset/DATA_DICTIONARY.md](../dataset/DATA_DICTIONARY.md).
- **You want to understand how the dataset was produced, in paper-Methods style, without reading code:** read [methods.md](methods.md).
- **You want the per-feature literature trail:** read [notebooks/08_features_explained.ipynb](../notebooks/08_features_explained.ipynb).
- **You want to understand the rock-abundance modelling stage:** read [modeling.md](modeling.md) (methods) then [modeling_results.md](modeling_results.md) (results).
- **You want the Stage 7 compositional-analysis conclusion:** read [compositional.md](compositional.md).

## Style conventions for this folder

- **No code blocks.** Documents here describe what was done in natural language. Pointers to code live in the repo-root files.
- **Figures live in `reports/figures/`** and are linked into the documents via relative paths. Documents in this folder do not duplicate figure files.
- **The repo-root [CLAUDE.md](../CLAUDE.md) is the build spec and stays stable.** When pipeline reality diverges, [DECISIONS.md](../DECISIONS.md) records the divergence. When the divergence is consequential, the affected document under `docs/` is updated in the same commit.
- **Commit references** in these documents point at the commit at the time of writing. They are not auto-updated; if a figure or value disagrees with the current cache, the current cache is authoritative.
