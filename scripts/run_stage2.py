"""Run Stage 2 (CTX retrieval) for one ObsId. Headless driver for manual / notebook use.

Usage:
    conda run -n geospatial python scripts/run_stage2.py ESP_069669_2220
    conda run -n geospatial python scripts/run_stage2.py ESP_017355_2260 --config config_v2.yaml
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
from src.ctx_retrieve import DEFAULT_MAX_INTERIOR_HOLE_PX, stage2_one_image


def _make_progress(prefix: str):
    last_pct = [-10]
    started = time.monotonic()

    def cb(downloaded: int, total: int) -> None:
        if total <= 0:
            return
        pct = int(downloaded * 100 / total)
        if pct >= last_pct[0] + 5:
            elapsed = time.monotonic() - started
            mb = downloaded / (1024 * 1024)
            mb_per_s = mb / elapsed if elapsed > 0 else 0.0
            print(
                f"  [{prefix}] {pct:3d}% ({mb:7.1f} MB) {mb_per_s:5.1f} MB/s",
                flush=True,
            )
            last_pct[0] = pct

    return cb


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 CTX-retrieval driver (one ObsId)")
    parser.add_argument("obs_id", help="HiRISE Observation ID")
    parser.add_argument("--config", default="config.yaml", help="Path to the pipeline config YAML")
    args = parser.parse_args()
    obs_id = args.obs_id

    cfg = load_config(args.config)
    df = M.load_manifest(cfg.manifest_path)
    if obs_id not in df["ObsId"].values:
        print(f"ObsId {obs_id!r} not in manifest")
        return 2
    row = df.set_index("ObsId").loc[obs_id]
    cfg_retrieve = cfg["ctx_retrieve"]

    t0 = time.monotonic()
    print(f"Stage 2 :: {obs_id}", flush=True)
    print(f"  CTX_TileName (manifest): {row['CTX_TileName']}", flush=True)

    prov = stage2_one_image(
        obs_id,
        cache_dir=cfg.cache_dir,
        manifest_row=row,
        target_crs=cfg["target_crs"],
        url_template=cfg["ctx_mosaic"]["url_template"],
        buffer_m=float(cfg_retrieve["buffer_m"]),
        nominal_width_m=float(cfg_retrieve["nominal_hirise_width_m"]),
        nominal_length_m=float(cfg_retrieve["nominal_hirise_length_m"]),
        config_hash=cfg.hash,
        # R74: explicit, config-driven, and recorded in the sidecar rather than left to a
        # default nobody can see from the artifact.
        max_interior_hole_px=int(
            cfg_retrieve.get("max_interior_hole_px", DEFAULT_MAX_INTERIOR_HOLE_PX)
        ),
        on_progress=_make_progress(obs_id),
    )

    dt = time.monotonic() - t0
    print(f"Done in {dt:.1f}s", flush=True)
    print(f"  source_murray_tile = {prov['source_murray_tile']}", flush=True)
    print(f"  footprint_source   = {prov['footprint_source']}", flush=True)
    print(f"  actual_shape       = {prov['actual_shape']} (rows x cols)", flush=True)
    print(f"  actual_bounds      = {prov['actual_bounds_target_crs']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
