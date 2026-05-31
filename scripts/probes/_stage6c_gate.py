"""Stage 6c — anti-signal image gate.

Builds the per-image predictor table from Stage 6b features, joins with per-image
baseline LOIO labels, runs leave-one-image-out CV of three gate models
(logistic regression, LightGBM regressor, simple threshold rule), and applies
the gate to the existing full-v2 LOIO summary to evaluate Strategy A
(headline-exclusion) against the strict acceptance criteria.

Inputs (all on disk, no re-sweep):
  - dataset_v2/features_ctx_illum/*.parquet     (38 images, Stage 6b features)
  - models/_sweep_stage6b/20260531T020308Z/summary.parquet  (per-fold metrics)

Outputs:
  - cache/stage6c/predictor_table.parquet       (38 rows x features+labels)
  - cache/stage6c/gate_cv.parquet               (per-image out-of-fold preds)
  - scripts/probes/_stage6c_gate.md             (persistent writeup)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "dataset_v2" / "features_ctx_illum"
SWEEP_DIR = ROOT / "models" / "_sweep_stage6b" / "20260531T020308Z"
SUMMARY = SWEEP_DIR / "summary.parquet"
AGGREGATE = SWEEP_DIR / "aggregate.parquet"

CACHE = ROOT / "cache" / "stage6c"
CACHE.mkdir(parents=True, exist_ok=True)

WRITEUP = ROOT / "scripts" / "probes" / "_stage6c_gate.md"

S64_IDX = 3  # tile_size_px == 64

# Baseline scheme to gate (un-augmented model).
BASELINE_SCHEME = "loio_nfold"

# Per-tile predictions for the baseline scheme (Strategy B / pooled-global eval).
BASELINE_TILE_DIR = (
    SWEEP_DIR / "loio_nfold" / "8c7523615964f5cb" / "scale_S64"
)
PER_TILE_PREDICTIONS = BASELINE_TILE_DIR / "predictions.parquet"

# Meaningful-positive cutoff used by the sweep (see DECISIONS for context):
# y_true >= 50 means "operationally meaningful boulder presence". Read from
# summary.parquet so we stay consistent if it ever changes.
MEANINGFUL_THRESHOLD = 50.0

# Binary "bad image" cutoff. The full-set baseline PR-AUC mean is 0.543 (from
# aggregate.parquet); we call an image "bad" if held-out PR-AUC is below
# this average. Threshold tunable via __main__.
BAD_PR_AUC_THRESHOLD = 0.55


# ---------------------------------------------------------------------------
# Step 1: per-image predictor table
# ---------------------------------------------------------------------------


def aggregate_per_image_features(obs_parquet: Path, scale_idx: int = S64_IDX) -> dict:
    """Aggregate Stage 6b ctx_* features for one image at the given scale."""
    df = pd.read_parquet(obs_parquet)
    df = df[df["scale_idx"] == scale_idx]
    if df.empty:
        raise ValueError(f"{obs_parquet.stem}: no tiles at scale_idx={scale_idx}")
    # Only valid tiles (drop NaN ctx features — these arise where the tile sits
    # off the SeamMap, which is rare but happens at footprint edges).
    sub = df.dropna(subset=["ctx_n_sources", "ctx_incidence_std",
                            "ctx_dominant_source_fraction"])
    obs_id = df["obs_id"].iloc[0]
    return {
        "obs_id": obs_id,
        "n_tiles_s64": int(len(df)),
        # Mechanism features (validated p < 0.05 in Stage 6b H3 check):
        "mean_n_sources": float(sub["ctx_n_sources"].mean()),
        "max_n_sources": float(sub["ctx_n_sources"].max()),
        "std_ctx_incidence": float(sub["ctx_incidence_std"].mean()),
        "mean_dominant_source_fraction": float(sub["ctx_dominant_source_fraction"].mean()),
        # Sanity covariates (not used as predictors, useful for the writeup):
        "mean_ctx_incidence": float(sub["ctx_incidence_mean"].mean()),
        "n_tiles_with_seammap": int(len(sub)),
    }


def build_predictor_table(features_dir: Path = FEATURES_DIR) -> pd.DataFrame:
    rows = []
    for f in sorted(features_dir.glob("*.parquet")):
        rows.append(aggregate_per_image_features(f))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Step 2: per-image baseline labels
# ---------------------------------------------------------------------------


def load_per_image_labels(
    summary_path: Path = SUMMARY,
    scheme: str = BASELINE_SCHEME,
    scale_idx: int = S64_IDX,
) -> pd.DataFrame:
    """One row per image, columns = held-out LOIO metrics for the baseline model."""
    s = pd.read_parquet(summary_path)
    s = s[(s["scheme"] == scheme) & (s["scale_idx"] == scale_idx)].copy()
    # Rename for clarity in the joined table.
    return s.rename(columns={"held_out_obs_id": "obs_id"})[
        [
            "obs_id",
            "n_tiles",
            "pr_auc",
            "spearman_rho",
            "presence_auc",
            "precision_at_top_5pct",
            "normalised_lift_meaningful",
            "meaningful_base_rate",
        ]
    ].rename(columns={
        "pr_auc": "baseline_pr_auc",
        "spearman_rho": "baseline_spearman",
        "presence_auc": "baseline_presence_auc",
        "precision_at_top_5pct": "baseline_p5pct",
        "normalised_lift_meaningful": "baseline_norm_lift",
        "meaningful_base_rate": "baseline_base_rate",
    })


# ---------------------------------------------------------------------------
# Step 3: gate models (LOIO CV)
# ---------------------------------------------------------------------------


FEATURES = ["mean_n_sources", "std_ctx_incidence", "mean_dominant_source_fraction"]


@dataclass
class GateCV:
    name: str
    y_true_binary: np.ndarray          # 1 = bad image (gate should flag)
    y_true_continuous: np.ndarray      # baseline PR-AUC
    p_bad: np.ndarray                  # gate's predicted prob of "bad"
    pred_continuous: np.ndarray | None # only for regressors
    roc_auc: float
    spearman_to_pr_auc: float


def loio_cv_logreg(table: pd.DataFrame, bad_threshold: float) -> GateCV:
    """Logistic regression with standardised features, LOIO CV."""
    X = table[FEATURES].to_numpy()
    y_cont = table["baseline_pr_auc"].to_numpy()
    y_bin = (y_cont < bad_threshold).astype(int)
    p_bad = np.zeros(len(X))

    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        # If the training fold has only one class, fall back to base rate.
        if len(np.unique(y_bin[tr])) < 2:
            p_bad[te] = float(y_bin[tr].mean())
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
        clf.fit(sc.transform(X[tr]), y_bin[tr])
        p_bad[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]

    return GateCV(
        name="logreg",
        y_true_binary=y_bin,
        y_true_continuous=y_cont,
        p_bad=p_bad,
        pred_continuous=None,
        roc_auc=float(roc_auc_score(y_bin, p_bad)) if len(np.unique(y_bin)) == 2 else float("nan"),
        spearman_to_pr_auc=float(stats.spearmanr(p_bad, y_cont).correlation),
    )


def loio_cv_ridge(table: pd.DataFrame, bad_threshold: float) -> GateCV:
    """Ridge regression on PR-AUC, then derive p_bad = score(below threshold).

    LightGBM would be overkill for n=38, 3 features — Ridge gives a smooth
    monotone predictor and avoids fitting noise.
    """
    X = table[FEATURES].to_numpy()
    y_cont = table["baseline_pr_auc"].to_numpy()
    y_bin = (y_cont < bad_threshold).astype(int)
    pred_cont = np.zeros(len(X))

    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        sc = StandardScaler().fit(X[tr])
        m = Ridge(alpha=1.0)
        m.fit(sc.transform(X[tr]), y_cont[tr])
        pred_cont[te] = m.predict(sc.transform(X[te]))

    # Convert continuous prediction to "p_bad" = how far below threshold.
    # Use a logistic on (threshold - pred) / scale; scale = std of pred.
    s = float(pred_cont.std()) or 1.0
    p_bad = 1.0 / (1.0 + np.exp(-(bad_threshold - pred_cont) / (s / 2)))

    return GateCV(
        name="ridge",
        y_true_binary=y_bin,
        y_true_continuous=y_cont,
        p_bad=p_bad,
        pred_continuous=pred_cont,
        roc_auc=float(roc_auc_score(y_bin, p_bad)) if len(np.unique(y_bin)) == 2 else float("nan"),
        spearman_to_pr_auc=float(stats.spearmanr(pred_cont, y_cont).correlation),
    )


def threshold_rule(table: pd.DataFrame, bad_threshold: float) -> GateCV:
    """Simple feature-based rule: flag bad if mean_n_sources > median(train).

    Sign is fixed from the Stage 6b H3 check: higher n_sources -> worse model.
    We pick the threshold *per LOIO fold* on the training set (median).
    """
    X = table[["mean_n_sources"]].to_numpy().ravel()
    y_cont = table["baseline_pr_auc"].to_numpy()
    y_bin = (y_cont < bad_threshold).astype(int)
    p_bad = np.zeros(len(X))
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        tau = float(np.median(X[tr]))
        p_bad[te] = 1.0 if X[te[0]] > tau else 0.0
    return GateCV(
        name="rule_n_sources_gt_median",
        y_true_binary=y_bin,
        y_true_continuous=y_cont,
        p_bad=p_bad,
        pred_continuous=None,
        roc_auc=float(roc_auc_score(y_bin, p_bad)) if len(np.unique(y_bin)) == 2 else float("nan"),
        spearman_to_pr_auc=float(stats.spearmanr(p_bad, y_cont).correlation),
    )


# ---------------------------------------------------------------------------
# Step 4: Strategy A (headline exclusion) evaluation
# ---------------------------------------------------------------------------


def evaluate_strategy_a(
    table: pd.DataFrame,
    gate: GateCV,
    tau_sweep: list[float],
) -> pd.DataFrame:
    """For each gate threshold tau, compute retained-set aggregate metrics."""
    rows = []
    total_tiles = int(table["n_tiles"].sum())
    full_pr_auc_mean = float(table["baseline_pr_auc"].mean())
    full_lift_mean = float(table["baseline_norm_lift"].mean())
    full_p5_mean = float(table["baseline_p5pct"].mean())
    full_spearman_mean = float(table["baseline_spearman"].mean())

    for tau in tau_sweep:
        keep = gate.p_bad <= tau
        n_kept = int(keep.sum())
        sub = table.loc[keep]
        kept_tiles = int(sub["n_tiles"].sum())
        rows.append({
            "tau": tau,
            "n_kept": n_kept,
            "n_total": len(table),
            "image_kept_frac": n_kept / len(table),
            "tile_kept_frac": kept_tiles / total_tiles if total_tiles else float("nan"),
            "kept_pr_auc_mean": float(sub["baseline_pr_auc"].mean()) if n_kept else float("nan"),
            "kept_norm_lift_mean": float(sub["baseline_norm_lift"].mean()) if n_kept else float("nan"),
            "kept_p5pct_mean": float(sub["baseline_p5pct"].mean()) if n_kept else float("nan"),
            "kept_spearman_mean": float(sub["baseline_spearman"].mean()) if n_kept else float("nan"),
            "delta_pr_auc": float(sub["baseline_pr_auc"].mean()) - full_pr_auc_mean if n_kept else float("nan"),
            "delta_norm_lift": float(sub["baseline_norm_lift"].mean()) - full_lift_mean if n_kept else float("nan"),
            "delta_p5pct": float(sub["baseline_p5pct"].mean()) - full_p5_mean if n_kept else float("nan"),
            "delta_spearman": float(sub["baseline_spearman"].mean()) - full_spearman_mean if n_kept else float("nan"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-K-worst exclusion sweep (finer-grained than the probability threshold;
# guaranteed monotone in retained-image count)
# ---------------------------------------------------------------------------


def evaluate_top_k_worst(
    table: pd.DataFrame,
    gate: GateCV,
    k_sweep: list[int],
) -> pd.DataFrame:
    """For each K, drop the K images with the highest predicted p_bad."""
    order = np.argsort(-gate.p_bad)  # highest p_bad first
    total_tiles = int(table["n_tiles"].sum())
    full_pr_auc_mean = float(table["baseline_pr_auc"].mean())
    full_lift_mean = float(table["baseline_norm_lift"].mean())
    rows = []
    for k in k_sweep:
        if k < 0 or k >= len(table):
            continue
        drop = order[:k] if k > 0 else np.array([], dtype=int)
        keep_mask = np.ones(len(table), dtype=bool)
        keep_mask[drop] = False
        sub = table.loc[keep_mask]
        kept_tiles = int(sub["n_tiles"].sum())
        rows.append({
            "k_dropped": k,
            "n_kept": len(sub),
            "tile_kept_frac": kept_tiles / total_tiles,
            "kept_pr_auc_mean": float(sub["baseline_pr_auc"].mean()),
            "kept_norm_lift_mean": float(sub["baseline_norm_lift"].mean()),
            "kept_p5pct_mean": float(sub["baseline_p5pct"].mean()),
            "kept_spearman_mean": float(sub["baseline_spearman"].mean()),
            "delta_pr_auc": float(sub["baseline_pr_auc"].mean()) - full_pr_auc_mean,
            "delta_norm_lift": float(sub["baseline_norm_lift"].mean()) - full_lift_mean,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pooled-global PR-AUC (Strategy B / C operate here, not at per-fold mean)
# ---------------------------------------------------------------------------


def pooled_global_pr_auc(
    per_tile: pd.DataFrame,
    meaningful_threshold: float = MEANINGFUL_THRESHOLD,
) -> dict:
    """Pool all 38 folds' held-out tiles into one ranking; compute global PR-AUC."""
    from sklearn.metrics import average_precision_score
    y_true_bin = (per_tile["y_true"].to_numpy() >= meaningful_threshold).astype(int)
    y_pred = per_tile["y_pred"].to_numpy()
    n_pos = int(y_true_bin.sum())
    return {
        "n_tiles": int(len(per_tile)),
        "n_positive": n_pos,
        "base_rate": float(n_pos / len(per_tile)),
        "pr_auc_global": float(average_precision_score(y_true_bin, y_pred)),
    }


def pooled_global_with_strategy_b(
    per_tile: pd.DataFrame,
    table: pd.DataFrame,
    gate: GateCV,
    meaningful_threshold: float = MEANINGFUL_THRESHOLD,
) -> dict:
    """Pool tiles, but multiply y_pred by (1 - p_bad) for tiles in flagged images.

    p_bad is the gate's LOIO out-of-fold prediction for the tile's held-out image.
    """
    from sklearn.metrics import average_precision_score
    p_bad_by_obs = dict(zip(table["obs_id"], gate.p_bad))
    # Use the fold's held-out obs_id as the key (this is where the gate would
    # have flagged the image at inference time).
    p_bad_per_tile = per_tile["fold_held_out_obs_id"].map(p_bad_by_obs).to_numpy()
    p_bad_per_tile = np.nan_to_num(p_bad_per_tile, nan=0.0)
    y_pred_adj = per_tile["y_pred"].to_numpy() * (1.0 - p_bad_per_tile)
    y_true_bin = (per_tile["y_true"].to_numpy() >= meaningful_threshold).astype(int)
    return {
        "pr_auc_global": float(average_precision_score(y_true_bin, y_pred_adj)),
        "mean_weight_on_flagged_tiles": float(
            (1.0 - p_bad_per_tile)[(p_bad_per_tile > 0.5)].mean()
            if (p_bad_per_tile > 0.5).any() else float("nan")
        ),
    }


def pooled_global_with_strategy_a(
    per_tile: pd.DataFrame,
    table: pd.DataFrame,
    gate: GateCV,
    tau: float,
    meaningful_threshold: float = MEANINGFUL_THRESHOLD,
) -> dict:
    """Pool tiles, dropping any tile whose held-out fold image was flagged (p_bad > tau)."""
    from sklearn.metrics import average_precision_score
    flagged = set(table.loc[gate.p_bad > tau, "obs_id"])
    keep_mask = ~per_tile["fold_held_out_obs_id"].isin(flagged)
    sub = per_tile.loc[keep_mask]
    if sub.empty:
        return {"pr_auc_global": float("nan"), "n_tiles_kept": 0,
                "tile_kept_frac": 0.0, "n_images_kept": 0}
    y_true_bin = (sub["y_true"].to_numpy() >= meaningful_threshold).astype(int)
    if y_true_bin.sum() == 0 or y_true_bin.sum() == len(y_true_bin):
        return {"pr_auc_global": float("nan"), "n_tiles_kept": int(len(sub)),
                "tile_kept_frac": float(len(sub) / len(per_tile)),
                "n_images_kept": int(len(table) - len(flagged))}
    return {
        "pr_auc_global": float(average_precision_score(y_true_bin, sub["y_pred"].to_numpy())),
        "n_tiles_kept": int(len(sub)),
        "tile_kept_frac": float(len(sub) / len(per_tile)),
        "n_images_kept": int(len(table) - len(flagged)),
    }


# ---------------------------------------------------------------------------
# Acceptance check
# ---------------------------------------------------------------------------


STRICT = {
    "kept_pr_auc_mean": 0.65,
    "tile_kept_frac": 0.70,
    "delta_norm_lift": 0.10,
}


def acceptance(row: pd.Series) -> dict:
    return {
        "pr_auc_ok": row["kept_pr_auc_mean"] >= STRICT["kept_pr_auc_mean"],
        "tile_frac_ok": row["tile_kept_frac"] >= STRICT["tile_kept_frac"],
        "lift_ok": row["delta_norm_lift"] >= STRICT["delta_norm_lift"],
    }


# ---------------------------------------------------------------------------
# Writeup
# ---------------------------------------------------------------------------


def fmt_pct(x: float) -> str:
    return f"{x*100:.1f} %"


def df_to_md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Minimal markdown table emitter (avoids the tabulate dep)."""
    cols = list(df.columns)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"

    def fmt(v: object) -> str:
        if isinstance(v, bool):
            return "✓" if v else "✗"
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "—"
            return format(float(v), floatfmt)
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return str(v)

    rows = ["| " + " | ".join(fmt(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, sep, *rows])


def write_markdown(
    table: pd.DataFrame,
    gate_cvs: list[GateCV],
    strategies: dict[str, pd.DataFrame],
    top_k: dict[str, pd.DataFrame],
    pooled: dict[str, dict],
    out: Path,
) -> None:
    lines = []
    lines.append("# Stage 6c — anti-signal image gate (probe results)")
    lines.append("")
    lines.append(
        f"Generated by [`_stage6c_gate.py`](./_stage6c_gate.py). "
        f"Inputs: 38 per-image predictor rows from "
        f"`dataset_v2/features_ctx_illum/*.parquet` at S=64, joined to baseline "
        f"per-image LOIO metrics from "
        f"`{SUMMARY.relative_to(ROOT).as_posix()}`."
    )
    lines.append("")
    lines.append(f"**Bad-image cutoff** for binary target: `baseline_pr_auc < {BAD_PR_AUC_THRESHOLD}` "
                 f"(matches the 0.54 full-set mean baseline PR-AUC; "
                 f"{int((table['baseline_pr_auc'] < BAD_PR_AUC_THRESHOLD).sum())}/{len(table)} images flagged as bad).")
    lines.append("")

    lines.append("## 1. Per-image predictor table (head)")
    lines.append("")
    cols = ["obs_id", "mean_n_sources", "std_ctx_incidence",
            "mean_dominant_source_fraction", "n_tiles",
            "baseline_pr_auc", "baseline_norm_lift", "baseline_p5pct",
            "baseline_spearman"]
    show = table[cols].copy()
    show = show.sort_values("baseline_pr_auc")
    lines.append(show.head(10).pipe(df_to_md))
    lines.append("")
    lines.append("Full table cached at `cache/stage6c/predictor_table.parquet`.")
    lines.append("")

    lines.append("## 2. Univariate feature ↔ baseline-PR-AUC correlations (n = 38)")
    lines.append("")
    lines.append("| feature | Spearman ρ | p | sign expected? |")
    lines.append("|---|---:|---:|---|")
    for f in FEATURES:
        r = stats.spearmanr(table[f], table["baseline_pr_auc"])
        # From Stage 6b H3 check: mean_n_sources & std_ctx_incidence are negative
        # vs per-image Spearman; mean_dominant_source_fraction is positive.
        expected_pos = f.startswith("mean_dominant")
        sign_ok = (r.correlation > 0) if expected_pos else (r.correlation < 0)
        lines.append(
            f"| `{f}` | {r.correlation:+.3f} | {r.pvalue:.3f} | "
            f"{'yes' if sign_ok else 'no'} |"
        )
    lines.append("")

    lines.append("## 3. Gate model LOIO cross-validation")
    lines.append("")
    lines.append("| model | ROC-AUC (binary) | Spearman(p_bad, baseline_pr_auc) |")
    lines.append("|---|---:|---:|")
    for g in gate_cvs:
        lines.append(f"| {g.name} | {g.roc_auc:.3f} | {g.spearman_to_pr_auc:+.3f} |")
    lines.append("")

    lines.append("## 4. Strategy A (headline exclusion) — per-threshold sweep")
    lines.append("")
    for name, df in strategies.items():
        lines.append(f"### Gate model = `{name}`")
        lines.append("")
        # Add acceptance flags
        flag = df.apply(acceptance, axis=1, result_type="expand")
        out_df = pd.concat([df, flag], axis=1)
        cols = [
            "tau", "n_kept", "tile_kept_frac", "kept_pr_auc_mean",
            "kept_norm_lift_mean", "delta_pr_auc", "delta_norm_lift",
            "pr_auc_ok", "tile_frac_ok", "lift_ok",
        ]
        lines.append(out_df[cols].pipe(df_to_md))
        lines.append("")
        # Identify any row passing strict criteria.
        passes = out_df[
            (out_df["kept_pr_auc_mean"] >= STRICT["kept_pr_auc_mean"])
            & (out_df["tile_kept_frac"] >= STRICT["tile_kept_frac"])
            & (out_df["delta_norm_lift"] >= STRICT["delta_norm_lift"])
        ]
        if not passes.empty:
            lines.append(f"**Strict acceptance PASS** at τ = {passes['tau'].tolist()}")
        else:
            # Soft: at least pr_auc + tile_frac (without lift).
            soft = out_df[
                (out_df["kept_pr_auc_mean"] >= STRICT["kept_pr_auc_mean"])
                & (out_df["tile_kept_frac"] >= STRICT["tile_kept_frac"])
            ]
            if not soft.empty:
                lines.append(f"**Strict FAIL** but PR-AUC+tile-fraction pass at τ = {soft['tau'].tolist()} "
                             f"(normalised lift below +0.10 over full baseline).")
            else:
                lines.append("**Strict FAIL** at all swept τ.")
        lines.append("")

    lines.append("## 5. Strategy A — top-K-worst exclusion (finer than probability threshold)")
    lines.append("")
    for name, df in top_k.items():
        lines.append(f"### Gate model = `{name}`")
        lines.append("")
        cols = ["k_dropped", "n_kept", "tile_kept_frac", "kept_pr_auc_mean",
                "kept_norm_lift_mean", "delta_pr_auc", "delta_norm_lift"]
        lines.append(df_to_md(df[cols]))
        lines.append("")

    lines.append("## 6. Pooled-global PR-AUC (Strategy B is meaningful here)")
    lines.append("")
    lines.append("Per-fold PR-AUC is **rank-invariant** within a single held-out image, so "
                 "Strategy B (down-weighting predictions on flagged images) and Strategy C "
                 "(per-image normalisation) leave the per-image PR-AUC unchanged. Their "
                 "effect shows up only in a **global pooled** ranking across all 38 folds' "
                 "tiles — that is the metric reported here.")
    lines.append("")
    # All gate models share the same baseline pooled metric; pick the first.
    first = next(iter(pooled.values()))
    base = first["baseline"]
    lines.append(
        f"**Baseline pooled PR-AUC** = {base['pr_auc_global']:.4f} "
        f"(n_tiles = {base['n_tiles']}, n_positive = {base['n_positive']}, "
        f"global base rate = {base['base_rate']:.3f})."
    )
    lines.append("")
    lines.append("| gate model | A@τ=0.5: PR-AUC | A@τ=0.5: tiles kept | B (down-weight) PR-AUC | Δ vs baseline (B) |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, res in pooled.items():
        a = res["strategy_a_tau_0.5"]
        b = res["strategy_b_continuous_downweight"]
        delta_b = b["pr_auc_global"] - base["pr_auc_global"]
        lines.append(
            f"| {name} | {a['pr_auc_global']:.4f} | {a['tile_kept_frac']*100:.1f} % | "
            f"{b['pr_auc_global']:.4f} | {delta_b:+.4f} |"
        )
    lines.append("")

    lines.append("## 7. Acceptance criteria (strict)")
    lines.append("")
    lines.append("- Retained-image mean PR-AUC ≥ **0.65** (vs 0.54 full-set baseline)")
    lines.append("- Retained-tile fraction ≥ **70 %** (don't drop > 30 % of tiles)")
    lines.append("- Retained-set normalised lift @ top-K ≥ **+0.10** over full baseline (0.528)")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("[1/5] Build per-image predictor table …")
    preds = build_predictor_table(FEATURES_DIR)
    print(f"  -> {len(preds)} rows")

    print("[2/5] Load per-image baseline labels …")
    labels = load_per_image_labels()
    table = preds.merge(labels, on="obs_id", how="inner")
    assert len(table) == len(preds), f"merge dropped rows: {len(table)} vs {len(preds)}"
    table.to_parquet(CACHE / "predictor_table.parquet", index=False)

    print("[3/5] LOIO CV gate models …")
    gates = [
        loio_cv_logreg(table, BAD_PR_AUC_THRESHOLD),
        loio_cv_ridge(table, BAD_PR_AUC_THRESHOLD),
        threshold_rule(table, BAD_PR_AUC_THRESHOLD),
    ]
    cv_df = pd.DataFrame({
        "obs_id": table["obs_id"].to_numpy(),
        "baseline_pr_auc": table["baseline_pr_auc"].to_numpy(),
        "y_bad": gates[0].y_true_binary,
        **{f"p_bad_{g.name}": g.p_bad for g in gates},
    })
    if gates[1].pred_continuous is not None:
        cv_df["pred_pr_auc_ridge"] = gates[1].pred_continuous
    cv_df.to_parquet(CACHE / "gate_cv.parquet", index=False)

    tau_sweep = [0.30, 0.40, 0.50, 0.60, 0.70]
    strategies = {g.name: evaluate_strategy_a(table, g, tau_sweep) for g in gates}

    k_sweep = [0, 2, 4, 6, 8, 10, 12, 15, 19]
    top_k_results = {g.name: evaluate_top_k_worst(table, g, k_sweep) for g in gates}

    print("[4/5] Pooled-global PR-AUC under Strategies A and B …")
    per_tile = pd.read_parquet(PER_TILE_PREDICTIONS)
    baseline_pooled = pooled_global_pr_auc(per_tile)
    print(f"  baseline pooled PR-AUC = {baseline_pooled['pr_auc_global']:.4f} "
          f"(n_tiles={baseline_pooled['n_tiles']}, base_rate={baseline_pooled['base_rate']:.3f})")
    pooled_results = {}
    for g in gates:
        b = pooled_global_with_strategy_b(per_tile, table, g)
        a_at_05 = pooled_global_with_strategy_a(per_tile, table, g, tau=0.5)
        pooled_results[g.name] = {
            "baseline": baseline_pooled,
            "strategy_a_tau_0.5": a_at_05,
            "strategy_b_continuous_downweight": b,
        }
        print(f"  {g.name}: A@τ=0.5 PR-AUC={a_at_05['pr_auc_global']:.4f} "
              f"(kept {a_at_05['tile_kept_frac']*100:.1f}% tiles); "
              f"B PR-AUC={b['pr_auc_global']:.4f}")

    print("[5/5] Write markdown writeup …")
    write_markdown(table, gates, strategies, top_k_results, pooled_results, WRITEUP)
    print(f"  -> {WRITEUP}")


if __name__ == "__main__":
    main()
