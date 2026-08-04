# Verification protocol — adversarially check a register finding

Only **9 of 64** register findings were checked by anyone other than their author. When R01–R22 were put
through a dedicated adversarial verifier it **killed 8 of 20 candidates** and corrected several
severities downward — a ~40 % kill rate. Applying that rate to the 55 unverified findings suggests
**~20 may not survive**. This protocol closes that gap for the findings that would change what we do.

**Idempotency contract:** a finding is verified iff `docs/review_2026-07-31/verify/<Rxx>.md` exists.
Run 4–5 at a time; each agent writes its own verdict file as its final action.

## Scope — the high-severity findings on the live path

These are the ones whose truth changes an action. In priority order:

| | finding | one-line claim |
|---|---|---|
| R60 | submitted PDF on pre-sign-fix labels | rescoring moves "usable" 14 % → 26 % |
| R61 | ">90 % agreement" vs chance | worst possible ranking scores 74.6 %, random 79.7 % |
| R54 | shipped abundance calibration | pooled `top_ratio 0.86` PASS; per image 11 of 37 in band |
| R32 | Tier-1 reference classifier | early-stops on AUC; 1-tree boosters on 11 of 38 folds |
| R56 | "`min_confidence` is harmful" | two-factor comparison; blocks R23's fix |
| R24 | S=128 Spearman | mean over 5 of 20 folds, reported as 20 |
| R31 | `extract_ctx_window` | cropped read stamped with the un-cropped transform |
| R36 | H4 leg-B skill gate | offsets applied were a near-constant; gate could not fail |
| R03 | HiRISE pixel-scale confound | 15.8 % of the mosaic's level-error variance |
| R48 | Stage-6b "validated mechanism" | prevalence confound; 10 of 12 significant cells die |
| R47 | v2 splits untested | already directly confirmed — re-verify only the coverage claim |
| R51 | `modeling_results.md` bottom line | sign test over 12 correlated re-analyses of 8 images |
| R44 | `docs/methods.md` half-migrated | README sends external readers here |
| R45 | within-image vs LOIO diagnostic | quadrant AUC paired against whole-image AUC |
| R37 | README / SHERLOCK_RUN pre-abort | instruct the next session to run the aborted build |
| R38 | A1 clip floor = nodata sentinel | dark valid pixels become "nodata" |

## Your task

You are verifying **one** finding. Read it in `docs/CODE_REVIEW_2026-07-31.md` (find `### <Rxx> —`) and
in its linked area file, then **try to kill it**.

1. **Do not trust the finding.** Open every file it cites and read enough surrounding code and callers to
   judge it yourself. Re-derive every number it quotes from committed artifacts
   (`reports/figures/*.csv|json`, `models/**/metrics.json`, `dataset*/`, the `.dbf`/GPKG caches) with
   small read-only pandas/numpy snippets. If a number does not reproduce, say so with your value.
2. **Decide which it is:**
   - **(a) factually wrong** about what the code does;
   - **(b) unreachable** / already guarded by a caller, a validation step, or a config default;
   - **(c) deliberate and documented** — grep `DECISIONS.md` and the relevant `PLAN_*.md` for the term
     before concluding this;
   - **(d) pinned by a passing test** as intended behaviour;
   - **(e) real but mis-stated** — give the corrected claim;
   - **(f) real and correctly stated** — say what you tried in order to kill it and why it survived.
3. **Default to REFUTED when you cannot positively confirm the defect by reading the code.** But do NOT
   refute a real defect merely because it sits in closed code or has small impact — **downgrade the
   severity instead**.
4. **Severity calibration:** `blocker` = invalidates a shipped number or a scientific verdict, or crashes
   the live path · `high` = wrong results in a plausible scenario · `medium` = wrong results in a narrow
   scenario, or a real protocol defect with bounded impact · `low` = hygiene with teeth.
5. Also judge the finding's **liveness** tag and its stated **impact** independently — several PASS-1
   findings were real in mechanism but wrong about blast radius, and that correction was the most useful
   part of the verdict.

READ-ONLY apart from your own verdict file. Do not run notebooks, sweeps, training, map builds, ISIS, or
anything touching the network or CTX/HiRISE imagery. Reading committed CSV/JSON/parquet artifacts and
cached vector files is expected and encouraged.

## Output — write `docs/review_2026-07-31/verify/<Rxx>.md`, as your FINAL action

```markdown
# Verification: <Rxx> — <short title>

- **Verdict:** CONFIRMED | CONFIRMED-BUT-MIS-STATED | REFUTED
- **Corrected severity:** blocker | high | medium | low
- **Corrected liveness:** live-shipped | live-active-plan | dead-closed | unclear
- **Verified at commit:** <git rev-parse --short HEAD>
- **Date:** <YYYY-MM-DD>

## What I checked
<The files and artifacts you opened, and the commands/snippets you ran.>

## Numbers re-derived
| quantity | finding claims | I measured | agrees? |
|---|---|---|---|

## Verdict reasoning
<Why it survives or dies. Cite path:line. If you refuted it, be explicit about which of (a)-(d) applies
and give the evidence, because this verdict is what stops a future session re-filing it.>

## Corrected claim
<If (e): the accurate statement. If the finding was right, write "as stated".>

## Fix assessment
<Is the register's proposed fix correct and sufficient? If not, what is?>
```

Write the file even if your verdict is REFUTED — a refutation is the most valuable output this protocol
produces, because it removes work.
