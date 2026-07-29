"""Bank a CalibrationLayer for the F-BUILD head (PLAN_FBuild Stage D / gate 6).

`models/deployable/calibration.npz` was fitted on the MOSAIC-path head's pooled LOIO predictions.
The F build runs a different head (`models/deployable_f_center/86c51a5dca220f63`, H1-centered
per-frame inputs, 36 train images), and Tier-2 is a **quantile-match** — a marginal-transfer map — so
reusing the mosaic-path knots on F-path P(rich) is a train/deploy mismatch of exactly the class that
killed F pilot leg A (DECISIONS 2026-07-04). Measured shift between the two heads' pooled P(rich)
marginals: median 0.2513 -> 0.3218, CDF L1 0.0358.

Brian ruled 2026-07-28: **re-bank on the F path and ship both** — F abundance uses this layer, and
gate 6 reports under this layer AND the reused mosaic-path layer so the domain-shift cost is
quantified rather than assumed.

Fit source: `reports/figures/f_leg_b_loio_preds_minnaert_center.csv`, store
`fang_embeddings_f_minnaert_center` — 153,663 out-of-fold tiles over the 36 common images.
  Tier-1 (isotonic)      needs PAIRED (p, y) — present in the CSV (`y` is fa > 1e-2, strict).
  Tier-2 (quantile-match) needs only the two MARGINALS (`QuantileMatcher.fit` sorts each side
                          independently), so the fractional_area side comes from
                          `dataset_v2/labels/{obs}.parquet` at tile_size_px == 32. VERIFIED
                          2026-07-28: per-obs tile counts match the CSV for all 36 obs
                          (153,663 == 153,663), so both marginals are over the SAME tile
                          population even though the CSV carries no ti/tj to join on.

Writes `models/deployable_f_center/calibration.npz` — never the mosaic-path artifact (which has no
versioning and whose knots the shipped mosaic map on disk depends on).

Run (laptop, ~1 min, CPU):
  conda run --no-capture-output -n geospatial python -u scripts/bank_calibration_f.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/pandas

import numpy as np
import pandas as pd

from src.calibration import (CalibrationLayer, IsotonicCalibrator, compression_metrics,
                             expected_calibration_error, quantile_match)

PREDS = REPO / "reports" / "figures" / "f_leg_b_loio_preds_minnaert_center.csv"
LABELS = REPO / "dataset_v2" / "labels"
MOSAIC_CAL = REPO / "models" / "deployable" / "calibration.npz"
OUT = REPO / "models" / "deployable_f_center" / "calibration.npz"
F_STORE = "fang_embeddings_f_minnaert_center"
FA_THRESH = 1e-2                 # the project's rich/poor cut, STRICT '>' (src/modeling/binary_target)
ECE_GATE = 0.05                  # PLAN_Calibration §6 Tier-1 gate
TOP_RATIO_BAND = (0.8, 1.2)      # PLAN_Calibration §6 Tier-2 gate


def load_fit_inputs(preds: Path):
    """(per-row p, per-row y, per-obs fa marginals) with the tile-population check enforced."""
    df = pd.read_csv(preds)
    f = df[df.store == F_STORE].copy()
    if f.empty:
        raise SystemExit(f"store {F_STORE!r} not in {preds}")
    obs_ids = sorted(f.obs_id.unique())
    fa_by_obs, n_bad = {}, []
    for obs in obs_ids:
        p = LABELS / f"{obs}.parquet"
        if not p.exists():
            n_bad.append((obs, int((f.obs_id == obs).sum()), -1))
            continue
        lab = pd.read_parquet(p, columns=["tile_size_px", "fractional_area"])
        fa = lab.loc[lab.tile_size_px == 32, "fractional_area"].to_numpy(float)
        fa_by_obs[obs] = fa
        if fa.size != int((f.obs_id == obs).sum()):
            n_bad.append((obs, int((f.obs_id == obs).sum()), int(fa.size)))
    if n_bad:
        raise SystemExit(
            "tile-population mismatch between the LOIO preds and dataset_v2/labels — the Tier-2\n"
            "quantile-match target would be a DIFFERENT tile set than the predictions.\n"
            f"  (obs, n_pred, n_label32): {n_bad[:8]}")
    print(f"fit inputs: {len(f):,} out-of-fold tiles over {len(obs_ids)} images "
          f"(pos rate {f.y.mean():.4f}); label marginal matched per obs for all {len(obs_ids)}",
          flush=True)
    return f, fa_by_obs, obs_ids


def marginal_loio_bound(f: pd.DataFrame, fa_by_obs: dict, obs_ids: list[str]) -> dict:
    """Honest deployment bound: fit on the other 35 images, score the held-out one.

    Tier-1 is a normal paired LOIO. Tier-2 has no per-tile (p, fa) pairing available in this CSV, so
    it is bounded at the MARGINAL level: qmatch is fitted on the retained images' two marginals and
    applied to the held-out image's p, then scored as
      marginal_l1  = mean |quantile(fa_held) - quantile(abundance_pred)|   (needs no pairing)
      top_ratio    = mean(pred abundance | y == 1) / mean(fa_held | fa_held > 1e-2)
    The numerator uses the CSV's paired binary y (which IS fa > 1e-2), the denominator the label
    marginal — both over the same tiles, so the ratio is well defined without a tile join.
    per-bin RMSE is NOT computable here (it needs per-tile truth); it is scored in gate 6 instead,
    where the cohort labels are joined to the composite through the global tile grid.
    """
    q = np.linspace(0, 1, 101)
    t1_pred = np.full(len(f), np.nan)
    ab_loio = np.full(len(f), np.nan)
    rows = []
    p_all, y_all = f.p.to_numpy(float), f.y.to_numpy(int)
    obs_col = f.obs_id.to_numpy()
    for obs in obs_ids:
        held = obs_col == obs
        keep = ~held
        t1_pred[held] = IsotonicCalibrator().fit(p_all[keep], y_all[keep]).predict(p_all[held])
        fa_ret = np.concatenate([fa_by_obs[o] for o in obs_ids if o != obs])
        ab_held = quantile_match(p_all[held], p_all[keep], fa_ret)
        ab_loio[held] = ab_held
        fa_held = fa_by_obs[obs]
        rich = fa_held > FA_THRESH
        num = float(ab_held[y_all[held] == 1].mean()) if (y_all[held] == 1).any() else np.nan
        den = float(fa_held[rich].mean()) if rich.any() else np.nan
        rows.append({"obs_id": obs, "n": int(held.sum()),
                     "marginal_l1": float(np.abs(np.quantile(fa_held, q) - np.quantile(ab_held, q)).mean()),
                     "top_ratio": num / den if (np.isfinite(num) and den) else np.nan})
    per_obs = pd.DataFrame(rows)
    # POOLED bound — this is the statistic comparable to the mosaic layer's number of record
    # (scripts/bank_calibration.py reports top_ratio over the pooled LOIO vector, 0.8573). The
    # per-image MEDIAN below is a strictly harsher statistic (each image's tail is predicted from
    # the other 35, so an unusually rocky image is under-predicted); reporting only the median
    # would be comparing a different quantity to the declared band.
    fa_pooled = np.concatenate([fa_by_obs[o] for o in obs_ids])
    rich_p = fa_pooled > FA_THRESH
    pooled_top = (float(ab_loio[y_all == 1].mean()) / float(fa_pooled[rich_p].mean())
                  if rich_p.any() and (y_all == 1).any() else np.nan)
    pooled_l1 = float(np.abs(np.quantile(fa_pooled, q) - np.quantile(ab_loio, q)).mean())
    ece = expected_calibration_error(y_all, t1_pred)
    return {"tier1_loio_ece": float(ece),
            "tier2_loio_top_ratio_pooled": float(pooled_top),
            "tier2_loio_marginal_l1_pooled": pooled_l1,
            "tier2_loio_top_ratio_img_median": float(per_obs.top_ratio.median()),
            "tier2_loio_top_ratio_img_p10": float(per_obs.top_ratio.quantile(0.10)),
            "tier2_loio_marginal_l1_img_median": float(per_obs.marginal_l1.median()),
            "per_obs": per_obs}


def compare_layers(f: pd.DataFrame, fa_by_obs: dict, obs_ids: list[str],
                   f_layer: CalibrationLayer, mos_layer: CalibrationLayer | None) -> pd.DataFrame:
    """Gate-6 evidence: what does REUSING the mosaic-path layer on F-path P(rich) actually cost?

    Scored at the marginal level over the pooled 36 images (compression_metrics needs pairing for
    top_ratio/spearman, so only its pairing-free parts are meaningful here — the paired version is
    gate 6 proper). Reported for both layers side by side.
    """
    q = np.linspace(0, 1, 101)
    fa = np.concatenate([fa_by_obs[o] for o in obs_ids])
    p = f.p.to_numpy(float)
    y = f.y.to_numpy(int)
    rows = []
    for name, layer in (("rebanked_f", f_layer), ("reused_mosaic", mos_layer)):
        if layer is None:
            continue
        ab = layer.calibrate_abundance(p)
        rich_num = float(ab[y == 1].mean()) if (y == 1).any() else np.nan
        rich_den = float(fa[fa > FA_THRESH].mean())
        rows.append({
            "layer": name,
            "marginal_l1": float(np.abs(np.quantile(fa, q) - np.quantile(ab, q)).mean()),
            "top_ratio": rich_num / rich_den if rich_den else np.nan,
            "near_zero_pred": float((ab < 1e-4).mean()), "near_zero_true": float((fa <= 0).mean()),
            "mean_pred": float(ab.mean()), "mean_true": float(fa.mean()),
            "max_pred": float(ab.max()), "ceiling": float(layer._t2[1][-1]),
            "saturated_frac": float((ab >= layer._t2[1][-1] - 1e-12).mean()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=str(PREDS))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-loio", action="store_true", help="skip the 36-fold deployment bound")
    ap.add_argument("--dry-run", action="store_true", help="fit + report, write nothing")
    args = ap.parse_args()

    out = Path(args.out)
    if out.resolve() == MOSAIC_CAL.resolve():
        raise SystemExit(f"refusing to overwrite the mosaic-path calibrator {MOSAIC_CAL} — the "
                         f"shipped reports/map_region map depends on its knots")

    f, fa_by_obs, obs_ids = load_fit_inputs(Path(args.preds))
    fa = np.concatenate([fa_by_obs[o] for o in obs_ids])
    p, y = f.p.to_numpy(float), f.y.to_numpy(int)
    # sanity: the CSV's binary y must BE fa > 1e-2 on this tile population (marginal check only,
    # since there is no per-tile join — the positive RATES must agree)
    print(f"binary check: pos rate y {y.mean():.6f} vs (fa > {FA_THRESH:g}) "
          f"{(fa > FA_THRESH).mean():.6f}", flush=True)

    layer = CalibrationLayer.fit(
        p_rich=p, y_binary=y, abundance_input=p, y_fractional_area=fa,
        meta={"recipe": "fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2", "scale": "S32", "mode": "one_model",
              "fit": f"pooled_loio_{len(obs_ids)}_f_minnaert_center",
              "head": "deployable_f_center/86c51a5dca220f63",
              "store": F_STORE, "fa_marginal": "dataset_v2/labels tile_size_px==32"})
    print(f"\nfitted: t1 {layer._t1[0].size} knots -> y in "
          f"[{layer._t1[1][0]:.4f}, {layer._t1[1][-1]:.4f}]; "
          f"t2 {layer._t2[0].size} knots -> abundance in "
          f"[{layer._t2[1][0]:.6f}, {layer._t2[1][-1]:.6f}]", flush=True)

    mos = CalibrationLayer.load(MOSAIC_CAL) if MOSAIC_CAL.exists() else None
    cmp_df = compare_layers(f, fa_by_obs, obs_ids, layer, mos)
    print("\n=== reused mosaic-path layer vs re-banked F layer (pooled, marginal level) ===")
    print(cmp_df.to_string(index=False), flush=True)

    summary = {"n_tiles": int(p.size), "n_images": len(obs_ids),
               "t1_knots": int(layer._t1[0].size), "t2_knots": int(layer._t2[0].size),
               "abundance_ceiling": float(layer._t2[1][-1]),
               "insample_ece": float(expected_calibration_error(y, layer.calibrate_prob(p)))}
    if not args.skip_loio:
        print("\n=== LOIO deployment bound (fit on 35, score the held-out image) ===", flush=True)
        bound = marginal_loio_bound(f, fa_by_obs, obs_ids)
        per_obs = bound.pop("per_obs")
        summary.update(bound)
        print(f"  Tier-1 ECE                {bound['tier1_loio_ece']:.4f}   "
              f"(gate <= {ECE_GATE}: {'PASS' if bound['tier1_loio_ece'] <= ECE_GATE else 'FAIL'})")
        tr = bound["tier2_loio_top_ratio_pooled"]
        print(f"  Tier-2 top_ratio POOLED   {tr:.4f}   (gate {TOP_RATIO_BAND}: "
              f"{'PASS' if TOP_RATIO_BAND[0] <= tr <= TOP_RATIO_BAND[1] else 'FAIL'})"
              f"   [mosaic layer on record: 0.8573]")
        print(f"  Tier-2 marginal_l1 pooled {bound['tier2_loio_marginal_l1_pooled']:.6f}")
        print(f"  per-image top_ratio       median {bound['tier2_loio_top_ratio_img_median']:.4f}, "
              f"p10 {bound['tier2_loio_top_ratio_img_p10']:.4f}  "
              f"(harsher statistic — each image's tail predicted from the other 35; "
              f"NOT the declared band's quantity)")
        fig = REPO / "reports" / "figures"
        fig.mkdir(parents=True, exist_ok=True)
        per_obs.to_csv(fig / "fbuild_calibration_f_loio.csv", index=False)
        cmp_df.to_csv(fig / "fbuild_calibration_layer_compare.csv", index=False)
        pd.DataFrame([summary]).to_csv(fig / "fbuild_calibration_f_summary.csv", index=False)

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    layer.save(out)
    back = CalibrationLayer.load(out)
    for name, (a, b) in (("t1_x", (layer._t1[0], back._t1[0])), ("t1_y", (layer._t1[1], back._t1[1])),
                         ("t2_x", (layer._t2[0], back._t2[0])), ("t2_y", (layer._t2[1], back._t2[1]))):
        d = float(np.abs(np.asarray(a) - np.asarray(b)).max())
        if d > 1e-12:
            raise SystemExit(f"save/load round-trip differs on {name} by {d}")
    print(f"\nwrote {out} (round-trip exact). Stage D ships F abundance under THIS layer; "
          f"gate 6 reports both.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
