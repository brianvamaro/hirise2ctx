"""Bank the Stage-1 CalibrationLayer (PLAN_Calibration §5 Stage 1).

Fits the deployment calibration on the POOLED LOIO predictions of all 38 images
(deployment-honest: out-of-fold preds, no further holdout) and saves it next to the
DeployableHead. **One-model default:** Tier-1 isotonic on `P(rich)` + Tier-2 *global*
quantile-match of the SAME `P(rich)` onto the `fractional_area` marginal (no separate
Tier-2 head). The in-cohort metrics printed (ECE, top_ratio) are the conservative bound
the deployed layer inherits — off-HiRISE terrain has no truth.

**2026-08-06 (audit isolation criterion 4 + the calibration gate).** Three defects fixed:

- every path was hard-coded, so a scratch rebuild could not run without writing the live
  `models/` tree. All three are now flags, and `--out` may point anywhere.
- the layer was **saved before the gates were evaluated**, so a run that printed `FAIL`
  still overwrote the banked calibrator, and `main` returned 0 regardless. Gates are now
  computed first and a failing run writes nothing and exits 1 (override with `--force`).
- the predictions↔labels merge was a bare `how="inner"`, which silently drops any key that
  does not join. It now asserts a complete one-to-one join and reports the anti-join.

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/bank_calibration.py
    ... --predictions <preds.parquet> --labels-dir <dir> --out <calibration.npz>
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.calibration import (CalibrationLayer, IsotonicCalibrator, quantile_match,
                             loio_calibrate, expected_calibration_error, compression_metrics)

DEFAULT_RECIPE = "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2"
DEFAULT_PREDS = REPO / "models/fang_probe" / DEFAULT_RECIPE / "predictions.parquet"
DEFAULT_LABELS_DIR = REPO / "dataset_v2/labels"
DEFAULT_OUT = REPO / "models/deployable/calibration.npz"

JOIN_KEYS = ["obs_id", "ti", "tj"]

# Promotion gates (PLAN_Calibration §5). Evaluated on the LOIO bound, never in-sample.
ECE_MAX = 0.05
TOP_RATIO_RANGE = (0.8, 1.2)


def _load_labels(labels_dir: Path, scale_px: int) -> pd.DataFrame:
    paths = sorted(Path(labels_dir).glob("*.parquet"))
    if not paths:
        raise SystemExit(f"no label parquets in {labels_dir}")
    parts = []
    for p in paths:
        d = pd.read_parquet(p)
        parts.append(d[d.tile_size_px == scale_px][JOIN_KEYS + ["fractional_area"]])
    return pd.concat(parts, ignore_index=True)


def _complete_one_to_one_join(t1: pd.DataFrame, lab: pd.DataFrame) -> pd.DataFrame:
    """Inner-join predictions to labels, refusing to proceed on anything but 1:1 complete.

    A bare `how="inner"` turns "these keys disappeared" into "fewer rows", which is exactly
    how a recovered-or-missing tile leaves the calibration pool without anyone noticing.
    """
    for name, df in (("predictions", t1), ("labels", lab)):
        dup = df.duplicated(subset=JOIN_KEYS).sum()
        if dup:
            raise SystemExit(
                f"{name}: {dup} duplicate {JOIN_KEYS} rows — the join would not be 1:1."
            )
    merged = t1.merge(lab, on=JOIN_KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(t1):
        missing = t1.merge(lab, on=JOIN_KEYS, how="left", indicator=True)
        orphans = missing[missing["_merge"] == "left_only"]
        raise SystemExit(
            f"incomplete join: {len(t1) - len(merged)} of {len(t1)} prediction rows have no "
            f"label at this scale (e.g. {orphans[JOIN_KEYS].head(3).to_dict('records')}). "
            "The prediction artifact and the labels are different generations — re-run LOIO "
            "against the current labels rather than calibrating on the intersection."
        )
    return merged


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Fit and bank the deployment CalibrationLayer")
    ap.add_argument("--predictions", type=Path, default=DEFAULT_PREDS,
                    help="pooled LOIO predictions parquet (obs_id, ti, tj, y_true, y_pred)")
    ap.add_argument("--labels-dir", type=Path, default=DEFAULT_LABELS_DIR,
                    help="Stage 4 labels directory supplying fractional_area")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="where to write calibration.npz (may be any scratch path)")
    ap.add_argument("--scale-px", type=int, default=32, help="tile_size_px to calibrate on")
    ap.add_argument("--recipe", default=DEFAULT_RECIPE, help="recipe id recorded in the layer meta")
    ap.add_argument("--force", action="store_true",
                    help="write the layer even if a promotion gate fails (records it in meta)")
    args = ap.parse_args(argv)

    t1 = pd.read_parquet(args.predictions).rename(
        columns={"y_true": "y_binary", "y_pred": "p_rich"})
    lab = _load_labels(args.labels_dir, args.scale_px)
    df = _complete_one_to_one_join(t1, lab)
    print(f"fit on {len(df)} tiles / {df.obs_id.nunique()} images (pooled LOIO, one-model)",
          flush=True)
    print(f"  predictions: {args.predictions}", flush=True)
    print(f"  labels:      {args.labels_dir}", flush=True)

    yb = df.y_binary.to_numpy(); pr = df.p_rich.to_numpy(); fa = df.fractional_area.to_numpy()

    # ---- Gates first. Nothing is written until they pass. -------------------------
    # LOIO deployment bound (honest): fit the map on the other 37 images, apply to the
    # held-out one — what off-cohort-like terrain inherits. This is the number to trust.
    iso_loio = loio_calibrate(df.rename(columns={"p_rich": "y_pred", "y_binary": "y_true"}),
                              lambda rp, rt, hp: IsotonicCalibrator().fit(rp, rt).predict(hp))
    ab_loio = loio_calibrate(df.rename(columns={"p_rich": "y_pred", "fractional_area": "y_true"}),
                             lambda rp, rt, hp: quantile_match(hp, rp, rt))
    ece_loio = expected_calibration_error(yb, iso_loio)
    m_loio = compression_metrics(fa, ab_loio)
    ece_pass = ece_loio <= ECE_MAX
    top_pass = TOP_RATIO_RANGE[0] <= m_loio["top_ratio"] <= TOP_RATIO_RANGE[1]
    print(f"  [LOIO bound] Tier-1 ECE {ece_loio:.3f}  (gate <={ECE_MAX}: "
          f"{'PASS' if ece_pass else 'FAIL'})", flush=True)
    print(f"  [LOIO bound] Tier-2 top_ratio {m_loio['top_ratio']:.2f}  "
          f"near0 {m_loio['near_zero_pred']:.1%} (true {m_loio['near_zero_true']:.1%})  "
          f"marginal_L1 {m_loio['marginal_l1']:.4f}  spearman {m_loio['spearman']:.3f}  "
          f"(gate top in {list(TOP_RATIO_RANGE)}: {'PASS' if top_pass else 'FAIL'})", flush=True)

    if not (ece_pass and top_pass) and not args.force:
        print(
            "\nGATE FAILURE — nothing written. The previously banked calibration at "
            f"{args.out} is unchanged and still belongs to an earlier fit. Re-run with "
            "--force only if you intend to bank a layer that failed its promotion gates.",
            flush=True,
        )
        return 1

    layer = CalibrationLayer.from_loio_predictions(
        df, meta={"recipe": args.recipe, "scale": f"S{args.scale_px}",
                  "mode": "one_model", "fit": f"pooled_loio_{df.obs_id.nunique()}",
                  "n_tiles": int(len(df)),
                  "predictions_path": str(args.predictions),
                  "labels_dir": str(args.labels_dir),
                  "loio_ece": float(ece_loio),
                  "loio_top_ratio": float(m_loio["top_ratio"]),
                  "gates_passed": bool(ece_pass and top_pass),
                  "forced": bool(args.force and not (ece_pass and top_pass))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    layer.save(args.out)

    # In-sample (the banked global map vs its own fit data) — a SANITY check that the
    # fit reproduces the training marginal, NOT a deployment estimate (isotonic fits its
    # training ECE to ~0 by construction; qmatch matches its training marginal exactly).
    ece_in = expected_calibration_error(yb, layer.calibrate_prob(pr))
    m_in = compression_metrics(fa, layer.calibrate_abundance(pr))
    print(f"  [in-sample sanity] Tier-1 ECE {expected_calibration_error(yb, pr):.3f} -> {ece_in:.3f}; "
          f"Tier-2 top_ratio {m_in['top_ratio']:.2f}, marginal_L1 {m_in['marginal_l1']:.4f}",
          flush=True)

    back = CalibrationLayer.load(args.out)
    d = max(float(np.abs(back.calibrate_prob(pr[:4096]) - layer.calibrate_prob(pr[:4096])).max()),
            float(np.abs(back.calibrate_abundance(pr[:4096]) - layer.calibrate_abundance(pr[:4096])).max()))
    ok = d < 1e-9
    print(f"  save/load round-trip max |d| = {d:.2e} ({'OK' if ok else 'MISMATCH'}) "
          f"-> {args.out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
