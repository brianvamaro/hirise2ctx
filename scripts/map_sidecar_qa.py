"""PLAN_Rebuild step 12 -- the per-tile sidecar QA table for both shipped map arms.

One row per (arm, tile), plus the aggregate gates that a step-12 promotion rests on. The
point of the exercise is to state what the 52 sidecars actually support **without conflating
their three schema generations** -- see ``src.map_qa`` for why that is load-bearing rather
than pedantic. Two traps this exists to avoid:

* **A missing ``overlap`` key is not a zero.** 21 baseline + 7 A1 tiles predate the key and
  carry only the old scalar, which was counted on the *calibrated* ``prob`` layer where
  isotonic collapses raw fp16 disagreements onto shared knots. Their 0 is an absence of
  measurement. They are reported as ``unknown_on_gate_layer``, and the aggregate says so with
  its own denominator instead of averaging a 0 in.
* **A missing ``device`` field does not identify the hardware by itself.** It is absent on the
  21 oldest baseline tiles *and* the 7 oldest A1 tiles, which ran on different cards (2080 Ti
  vs Pascal). The attribution comes from the run logs, and every inferred row is flagged
  ``device_inferred``.

Run (laptop, seconds):

    C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run --no-capture-output -n geospatial \\
        python -u scripts/map_sidecar_qa.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402
import rasterio                                                             # noqa: E402

from src import map_qa                                                      # noqa: E402
from src.mapping import COARSE_GRID_ID                                      # noqa: E402
from src.striping import A1_ARM, A1_VALID_FLOOR                             # noqa: E402

A1_FALLBACK_PIXEL_MAX = 1e-3   # share of a tile's pixels allowed to carry the tile-level
                               # fallback statistic instead of a per-frame one (measured max 1.6e-4)

ARMS = {"baseline": REPO / "reports" / "map_region", "a1": REPO / "reports" / "map_a1"}
FIG = REPO / "reports" / "figures"


def tile_row(arm: str, tile: str, sc: dict, map_dir: Path) -> dict:
    run = sc.get("run") or {}
    ov = map_qa.overlap_status(sc)
    dev = map_qa.device_status(sc, arm=arm)
    rasters = map_qa.raster_records(sc)
    row = {"arm": arm, "tile": tile, "generation": ov["generation"],
           "grid_id": sc.get("grid_id"), "cell_row0": sc.get("cell_row0"),
           "cell_col0": sc.get("cell_col0"),
           "raster_shape": "x".join(str(v) for v in (sc.get("raster_shape") or [])),
           "n_layers": len(rasters), "n_predicted_tiles": sc.get("n_predicted_tiles"),
           "n_unique_cells": sc.get("n_unique_cells"),
           "abundance_mean": sc.get("abundance_mean"), "prob_mean": sc.get("prob_mean"),
           "rich_share_at_0p5": sc.get("rich_share_at_0p5"),
           "head_digest": sc.get("head_digest") or run.get("head_digest"),
           "calibration_digest": sc.get("calibration_digest") or run.get("calibration_digest"),
           "n_windows": run.get("n_windows"), "elapsed_s": run.get("elapsed_s"),
           "cost_s": run.get("cost_s"),
           "overlap_verdict": ov["verdict"], "overlap_gate_layer": ov["gate_layer"],
           "overlap_n_dup": ov["n_dup"], "overlap_n_disagree": ov["n_disagree"],
           "overlap_fraction": ov["fraction"], "overlap_max_abs": ov["max_abs"],
           "overlap_scalar_legacy": ov["scalar_overlap_disagreements"],
           "overlap_note": ov["note"],
           "device": dev["device"], "device_inferred": dev["device_inferred"],
           "device_evidence": dev["device_evidence"]}
    for kind in map_qa.LAYERS:
        r = rasters.get(kind) or {}
        row[f"{kind}_n_finite"] = r.get("n_finite")
        row[f"{kind}_sha256_present"] = bool(r.get("sha256"))
    nd = sc.get("nodata_gate") or {}
    row["n_masked_nodata"] = nd.get("n_masked_nodata")
    row["n_masked_context_nodata"] = nd.get("n_masked_context_nodata")

    # The nodata accounting, reconciled against the raster rather than asserted. The gate
    # masks a coarse cell when its own CTX window is too empty (`max_zero_fraction`) or its
    # context is (`max_context_zero_fraction`); every masked cell must then be NaN in the
    # product, and every NaN must be a masked cell. `n_unique_cells` would give the same
    # answer but exists only on the baseline sidecars, so the raster is the common basis.
    with rasterio.open(map_dir / f"{tile}_abundance.tif") as ds:
        a = ds.read(1)
        tile_cells = int(a.size)
        raster_nodata = tile_cells - int(np.isfinite(a).sum())
    masked = (row["n_masked_nodata"] or 0) + (row["n_masked_context_nodata"] or 0)
    row["tile_cells"] = tile_cells
    row["raster_nodata"] = raster_nodata
    row["nodata_reconciles"] = bool(masked == raster_nodata)
    row["n_unique_cells_matches_raster"] = (
        None if sc.get("n_unique_cells") is None
        else bool(sc["n_unique_cells"] == tile_cells - raster_nodata))

    # A1-arm specifics. The versioned arm string is the one that killed all six tasks of the
    # first step-11 array when a literal `a1` was recorded instead (PLAN_Rebuild §3 step 8).
    row["a1_arm"] = sc.get("a1_a1_arm")
    row["a1_clip_floor"] = sc.get("a1_clip_floor")
    row["a1_clip_fraction"] = sc.get("a1_clip_fraction")
    row["a1_n_frames"] = sc.get("a1_n_frames")
    row["a1_n_frames_too_small"] = sc.get("a1_n_frames_too_small")
    row["a1_min_frame_px"] = sc.get("a1_min_frame_px")
    row["a1_fallback_pixel_fraction"] = sc.get("a1_fallback_pixel_fraction")
    return row


def gates(df: pd.DataFrame) -> list[dict]:
    """The aggregate assertions a promotion rests on. Each returns pass/fail plus its basis."""
    out = []

    def add(name, ok, detail):
        out.append({"gate": name, "verdict": "PASS" if ok else "FAIL", "detail": detail})

    gids = sorted(set(df.grid_id.dropna()))
    add("one grid_id across both arms",
        gids == [COARSE_GRID_ID], f"{len(gids)} distinct: {gids}")

    add("every tile carries all 3 layers with a sha256",
        bool((df.n_layers == 3).all()
             and df[[f"{k}_sha256_present" for k in map_qa.LAYERS]].all().all()),
        f"n_layers min {df.n_layers.min()}, all sha256 present "
        f"{df[[f'{k}_sha256_present' for k in map_qa.LAYERS]].all().all()}")

    heads = df.groupby("arm").head_digest.nunique().to_dict()
    per_arm = df.groupby("arm").head_digest.agg(lambda s: sorted(set(s.dropna()))[0]).to_dict()
    add("exactly one head per arm, and the two arms DIFFER (R07: norm_arm is in the "
        "recipe hash, so a shared digest means the fix is not in effect)",
        all(v == 1 for v in heads.values()) and len(set(per_arm.values())) == len(per_arm),
        f"digests per arm {heads}; " + ", ".join(f"{a}={d[:12]}…" for a, d in per_arm.items()))

    cals = df.groupby("arm").calibration_digest.nunique().to_dict()
    add("exactly one calibration layer per arm", all(v == 1 for v in cals.values()),
        f"distinct per arm {cals}")

    n_fail = int((df.overlap_verdict == "fail").sum())
    n_unknown = int((df.overlap_verdict == "unknown_on_gate_layer").sum())
    n_pass = int((df.overlap_verdict == "pass").sum())
    add("no tile FAILS the overlap-agreement gate (unknown is reported, never counted as pass)",
        n_fail == 0,
        f"{n_pass} pass / {n_unknown} unknown_on_gate_layer / {n_fail} fail, of {len(df)}")

    shapes = sorted(set(df.raster_shape))
    add("one raster shape across every tile", len(shapes) == 1, f"{shapes}")

    per_arm_tiles = df.groupby("arm").tile.nunique().to_dict()
    add("both arms carry the same 26 tiles",
        len(set(per_arm_tiles.values())) == 1
        and len(set(df[df.arm == "baseline"].tile)) == len(set(df[df.arm == "a1"].tile))
        and set(df[df.arm == "baseline"].tile) == set(df[df.arm == "a1"].tile),
        f"{per_arm_tiles}")

    # NOT "no cell was masked" -- 6 of the 26 tiles legitimately mask cells where the CTX
    # mosaic has no coverage (102-2,781 each, 7,940 total per arm). The gate is that the
    # accounting CLOSES: masked-by-the-gate == NaN-in-the-raster, tile by tile.
    n_masking = int((df.raster_nodata > 0).sum())
    add("nodata accounting reconciles per tile "
        "(n_masked_nodata + n_masked_context_nodata == raster NaN count)",
        bool(df.nodata_reconciles.all()),
        f"{int(df.nodata_reconciles.sum())}/{len(df)} reconcile; {n_masking} row(s) mask any "
        f"cell at all, {int(df.raster_nodata.sum())} cells total across both arms "
        f"(max {int(df.raster_nodata.max())} on one tile)")

    chk = df.n_unique_cells_matches_raster.dropna()
    add("where the sidecar records n_unique_cells, it matches the raster's finite count "
        "(A1 sidecars do not carry the key, so this covers the baseline arm only)",
        bool(chk.all()) if len(chk) else True, f"{int(chk.sum())}/{len(chk)} checked")

    a1 = df[df.arm == "a1"]
    arms_seen = sorted(set(a1.a1_arm.dropna()))
    add(f"every A1 tile records the versioned arm '{A1_ARM}' (a literal 'a1' here is what "
        "killed all six tasks of the first step-11 array)",
        arms_seen == [A1_ARM], f"{arms_seen}")
    floors = sorted(set(a1.a1_clip_floor.dropna()))
    add(f"every A1 tile records R38's clip floor A1_VALID_FLOOR={A1_VALID_FLOOR} "
        "(so DN 0 means nodata only)", floors == [A1_VALID_FLOOR], f"{floors}")
    # NOT "no frame was dropped". Frames under `a1_min_frame_px` (50) fall back to the
    # tile-level median/IQR *by design* -- R08's population is exactly the small frames where
    # the robust IQR is unstable. 9 of 1,371 frames fall back across the arm. What has to be
    # small is the share of PIXELS carrying the fallback statistic rather than a per-frame one,
    # and that number also covers pixels no SeamMap frame owns, so it is the honest quantity.
    add(f"A1 fallback statistic covers < {A1_FALLBACK_PIXEL_MAX:.1%} of pixels on every tile "
        "(frames below a1_min_frame_px fall back by design, R08 -- the count is reported, "
        "the pixel share is gated)",
        bool((a1.a1_fallback_pixel_fraction < A1_FALLBACK_PIXEL_MAX).all()),
        f"max fallback pixel fraction {a1.a1_fallback_pixel_fraction.max():.3e}, median "
        f"{a1.a1_fallback_pixel_fraction.median():.3e}; "
        f"{int(a1.a1_n_frames_too_small.fillna(0).sum())} of "
        f"{int(a1.a1_n_frames.sum())} frames fell back on "
        f"{int((a1.a1_n_frames_too_small.fillna(0) > 0).sum())} tile(s); frames/tile "
        f"{int(a1.a1_n_frames.min())}-{int(a1.a1_n_frames.max())}; "
        f"max clip fraction {a1.a1_clip_fraction.max():.3g}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(ARMS["baseline"]))
    ap.add_argument("--a1", default=str(ARMS["a1"]))
    args = ap.parse_args()

    dirs = {"baseline": Path(args.baseline), "a1": Path(args.a1)}
    rows = []
    for arm, d in dirs.items():
        arm_sidecars = map_qa.load_arm(d)
        if not arm_sidecars:
            raise SystemExit(f"no per-tile sidecars under {d}")
        for tile, sc in arm_sidecars.items():
            rows.append(tile_row(arm, tile, sc, d))
    df = pd.DataFrame(rows).sort_values(["arm", "tile"]).reset_index(drop=True)

    print("=== sidecar schema generations (they must not be conflated) ===")
    gen = df.pivot_table(index="arm", columns="generation", values="tile",
                         aggfunc="count", fill_value=0)
    print(gen.to_string())
    print("\n  g1_scalar_only  = no `overlap` key; legacy scalar was counted on the CALIBRATED")
    print("                    prob layer, so its 0 is an ABSENCE OF MEASUREMENT, not a zero")
    print("  g2_raw_fraction = overlap.prob_raw.fraction, pre-1e-6-floor: an UPPER BOUND")
    print("  g3_floored      = post-floor gate quantity -- NO shipped tile is g3")

    print("\n=== render hardware (absence of `device` is inferred, and arm-conditional) ===")
    print(df.groupby(["arm", "device", "device_inferred"]).tile.count().to_string())

    print("\n=== overlap-agreement gate ===")
    for arm in dirs:
        s = df[df.arm == arm]
        c = Counter(s.overlap_verdict)
        known = s[s.overlap_verdict != "unknown_on_gate_layer"]
        line = (f"  {arm:9s} " + " / ".join(f"{v} {k}" for k, v in sorted(c.items())))
        if len(known):
            line += (f"  |  measured fraction max {known.overlap_fraction.max():.5f} "
                     f"(gate {map_qa.OVERLAP_FRACTION_GATE}), "
                     f"max |Δ| {known.overlap_max_abs.max():.3g}, "
                     f"dup cells {int(known.overlap_n_dup.min())}-"
                     f"{int(known.overlap_n_dup.max())}/tile")
        print(line)

    print("\n=== GATES ===")
    g = gates(df)
    for row in g:
        print(f"  [{row['verdict']}] {row['gate']}")
        print(f"          {row['detail']}")

    FIG.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIG / "step12_sidecar_qa.csv", index=False)
    (FIG / "step12_sidecar_qa.json").write_text(json.dumps({
        "generations": gen.to_dict(),
        "devices": {f"{a}|{d}|inferred={i}": int(n) for (a, d, i), n
                    in df.groupby(["arm", "device", "device_inferred"]).tile.count().items()},
        "gates": g,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {(FIG / 'step12_sidecar_qa.csv').relative_to(REPO)} (+ .json)")
    return 0 if all(r["verdict"] == "PASS" for r in g) else 1


if __name__ == "__main__":
    raise SystemExit(main())
