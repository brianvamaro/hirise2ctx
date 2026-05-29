"""Run Stage 1 (detection ingest + reproject to the common CTX CRS) headless.

Stage 1 reads each ObsId's BoulderNet `*-mask-nms.shp`, corrects the buggy
`Standard_Parallel_1=0` / `D_unnamed` .prj via the PDS `.LBL` `CENTER_LATITUDE`
when detected, reprojects to `target_crs`, and caches the result to
`cache/reprojected_detections/{ObsId}.gpkg` (+ provenance sidecar). It is the
prerequisite for Stage 2 (window bounds) and Stage 4 (label rasterization),
both of which only *load* that cache.

Usage:
    conda run -n geospatial python scripts/run_stage1.py ESP_069669_2220
    conda run -n geospatial python scripts/run_stage1.py --all
    conda run -n geospatial python scripts/run_stage1.py --all --config config_v2.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import detections as det
from src import manifest as M
from src.config import load_config


def _reproject_one(cfg, obs_id: str, row) -> dict | None:
    t0 = time.monotonic()
    try:
        gdf_t, gpkg, correction = det.stage1_one_image(
            obs_id,
            detections_root=cfg.detections_root,
            target_crs=cfg["target_crs"],
            cache_dir=cfg.cache_dir,
            config_hash=cfg.hash,
            manifest_row=row,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"  {obs_id}: FAILED ({e})", flush=True)
        return None
    dt = time.monotonic() - t0
    status = correction.get("status", "trusted_prj")
    print(f"  {obs_id}: n_polys={len(gdf_t):>9}  {status}  [{dt:.1f}s]  -> {gpkg.name}", flush=True)
    return {"obs_id": obs_id, "n_polygons": len(gdf_t), "correction": correction}


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 1 detection-ingest + reproject driver")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("obs_id", nargs="?", default=None, help="HiRISE Observation ID")
    g.add_argument("--all", action="store_true", help="Reproject every manifest row")
    parser.add_argument("--config", default="config.yaml", help="Path to the pipeline config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = M.load_manifest(cfg.manifest_path)
    print(f"Stage 1 :: detections_root = {cfg.detections_root}", flush=True)
    print(f"           target_crs cached to {cfg.cache_dir / det.CACHE_SUBDIR}", flush=True)

    if args.all:
        rows = [(str(r["ObsId"]), r) for _, r in df.iterrows()]
        print(f"Stage 1 :: {len(rows)} manifest rows", flush=True)
        results = [(o, _reproject_one(cfg, o, r)) for o, r in rows]
        ok = [o for o, p in results if p is not None]
        bad = [o for o, p in results if p is None]
        print(f"\nReprojected {len(ok)} / {len(rows)}; failed {len(bad)}", flush=True)
        if bad:
            print(f"  Failed: {', '.join(bad)}", flush=True)
        corrected = [o for o, p in results if p and p["correction"].get("status") != "trusted_prj"]
        print(f"  SP1-corrected via PDS LBL: {len(corrected)}", flush=True)
        return 0 if not bad else 1

    obs = args.obs_id
    if obs not in df["ObsId"].values:
        print(f"ObsId {obs!r} not in manifest")
        return 2
    row = df.set_index("ObsId").loc[obs]
    print(f"Stage 1 :: {obs}", flush=True)
    prov = _reproject_one(cfg, obs, row)
    return 0 if prov is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
