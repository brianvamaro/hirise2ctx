"""Run Stage 4 (label generation) for one ObsId or the full manifest.

Usage:
    # Single image:
    conda run -n geospatial python scripts/run_stage4.py ESP_069669_2220

    # All ObsIds in the manifest (auto-skips Stage 2/3 misses and ESP_057469_2215):
    conda run -n geospatial python scripts/run_stage4.py --all

ESP_057469_2215 is excluded from --all: its Stage 2 window only covers 0.1% of the HiRISE
swath because its polygons straddle the W004_N40 / E000_N40 tile boundary (see DECISIONS.md
2026-05-22 tile-straddle entry; user decision 2026-05-23 = "drop from Stage 4 dataset").
Pass it explicitly as the positional argument to override.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import manifest as M
from src.config import load_config
from src.ctx_retrieve import CTX_WINDOWS_SUBDIR
from src.labeling import LABELS_SUBDIR, stage4_one_image

EXCLUDED_FROM_SWEEP = {"ESP_057469_2215"}


def _stage2_ready(cache_dir: Path, obs_id: str) -> bool:
    return (
        (cache_dir / CTX_WINDOWS_SUBDIR / f"{obs_id}.tif").exists()
        and (cache_dir / CTX_WINDOWS_SUBDIR / f"{obs_id}_hirise_mask.tif").exists()
        and (cache_dir / "reprojected_detections" / f"{obs_id}.gpkg").exists()
    )


def _label_one(cfg, obs_id: str, row, apply_coreg_shift: bool) -> dict | None:
    cache_dir = cfg.cache_dir
    if not _stage2_ready(cache_dir, obs_id):
        print(f"  {obs_id}: SKIP (Stage 2 outputs missing)", flush=True)
        return None
    t0 = time.monotonic()
    try:
        prov = stage4_one_image(
            obs_id,
            cache_dir=cache_dir,
            output_dir=cfg.output_dir,
            manifest_row=row,
            target_crs=cfg["target_crs"],
            labeling_cfg=cfg["labeling"],
            config_hash=cfg.hash,
            apply_coreg_shift=apply_coreg_shift,
        )
    except (RuntimeError, ValueError) as e:
        print(f"  {obs_id}: FAILED ({e})", flush=True)
        return None
    dt = time.monotonic() - t0
    eligible = prov["eligible_tiles_per_scale"]
    shift_note = (
        f"shift=({prov['coreg_shift_m']['dx']:+.1f}, {prov['coreg_shift_m']['dy']:+.1f}) m  "
        f"peak={prov['coreg_peak_correlation']:.3f}"
        if prov["coreg_shift_applied"] and prov["coreg_shift_m"]
        else "no shift"
    )
    eligible_str = "  ".join(f"S={s}:{n}" for s, n in sorted(eligible.items()))
    print(
        f"  {obs_id}: n_polys={prov['n_polygons_after_filter']:5d}  "
        f"eligible {eligible_str}  {shift_note}  [{dt:.1f}s]",
        flush=True,
    )
    return prov


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4 label-generation driver")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("obs_id", nargs="?", default=None, help="HiRISE Observation ID")
    g.add_argument("--all", action="store_true", help="Generate labels for the full manifest")
    parser.add_argument(
        "--no-coreg-shift", action="store_true",
        help="Leave grid on nominal geolocation (don't apply Stage 3 shift)",
    )
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    df = M.load_manifest(cfg.manifest_path)
    apply_coreg = not args.no_coreg_shift

    if args.all:
        rows = [
            (str(row["ObsId"]), row) for _, row in df.iterrows()
            if str(row["ObsId"]) not in EXCLUDED_FROM_SWEEP
        ]
        print(
            f"Stage 4 :: {len(rows)} of {len(df)} manifest rows "
            f"(excluding {sorted(EXCLUDED_FROM_SWEEP)})", flush=True,
        )
        print(f"  apply_coreg_shift = {apply_coreg}", flush=True)
        results: list[tuple[str, dict | None]] = []
        for obs, row in rows:
            results.append((obs, _label_one(cfg, obs, row, apply_coreg)))
        solved = [(o, p) for o, p in results if p is not None]
        skipped = [o for o, p in results if p is None]
        print(f"\nSolved {len(solved)} / {len(rows)}; skipped {len(skipped)}", flush=True)
        if skipped:
            print(f"  Skipped: {', '.join(skipped)}", flush=True)
        if solved:
            total_eligible = 0
            for _, p in solved:
                total_eligible += sum(p["eligible_tiles_per_scale"].values())
            print(
                f"\nTotal eligible tiles across all images and scales: {total_eligible}",
                flush=True,
            )
        return 0

    obs = args.obs_id
    if obs not in df["ObsId"].values:
        print(f"ObsId {obs!r} not in manifest")
        return 2
    row = df.set_index("ObsId").loc[obs]
    print(f"Stage 4 :: {obs}  (apply_coreg_shift={apply_coreg})", flush=True)
    prov = _label_one(cfg, obs, row, apply_coreg)
    if prov is None:
        return 1
    out_parquet = cfg.output_dir / LABELS_SUBDIR / f"{obs}.parquet"
    out_json = cfg.output_dir / LABELS_SUBDIR / f"{obs}.json"
    print(f"  wrote {out_parquet}", flush=True)
    print(f"        {out_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
