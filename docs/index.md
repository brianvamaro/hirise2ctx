# docs

Long-form documentation that complements (rather than duplicates) the in-repo
README + DECISIONS + DATA_DICTIONARY. The repo-root files are written for
**code users**; the documents under `docs/` are written for **readers**,
including reviewers, collaborators, and the project's advising committee, who
care about how the results were produced but won't necessarily touch the code.

## Index

| Document | Audience | Scope |
|---|---|---|
| [methods.md](methods.md) | Mixed / general scientific reader | Full data-pipeline Methods section in paper-Methods style: inputs → coordinate handling → co-registration → labels → features → cross-validation. Stops at the point where the modeler receives a packaged train/test dataset. ~20 pages of narrative + 9 embedded figures + 5 quantitative tables. |

Planned future documents (not yet written):

- `modeling.md` — Methods for the Week 3 modeling stage, written in the same style. Will cite back to `methods.md` for the data-pipeline details.
- `data_release.md` — A short, citeable description of the released dataset version (manifest, ObsId list, pipeline commit hash, schema reference) for use when the data is shared externally.

## When to read what

- **You want to run or modify the code:** start at the repo-root [README.md](../README.md) and [CLAUDE.md](../CLAUDE.md).
- **You want a chronological log of decisions and the reasons behind them:** read [DECISIONS.md](../DECISIONS.md).
- **You want the per-column output schemas:** read [dataset/DATA_DICTIONARY.md](../dataset/DATA_DICTIONARY.md).
- **You want to understand how the dataset was produced, in paper-Methods style, without reading code:** read [methods.md](methods.md).
- **You want the per-feature literature trail:** read [notebooks/08_features_explained.ipynb](../notebooks/08_features_explained.ipynb).

## Style conventions for this folder

- **No code blocks.** Documents here describe what was done in natural language. Pointers to code live in the repo-root files.
- **Figures live in `reports/figures/`** and are linked into the documents via relative paths. Documents in this folder do not duplicate figure files.
- **The repo-root [CLAUDE.md](../CLAUDE.md) is the build spec and stays stable.** When pipeline reality diverges, [DECISIONS.md](../DECISIONS.md) records the divergence. When the divergence is consequential, the affected document under `docs/` is updated in the same commit.
- **Commit references** in these documents point at the commit at the time of writing. They are not auto-updated; if a figure or value disagrees with the current cache, the current cache is authoritative.
