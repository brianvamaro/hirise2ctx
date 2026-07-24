"""PLAN_FBuild Stage-0 — build the 907-frame region frame list + frame→tile map.

The first executable step of the approved 907-frame F build (DECISIONS 2026-07-23b). Emits the
two manifests every later stage consumes:

  region_frame_list.csv  — one row per UNIQUE source frame: PRODUCT_ID, VOLUME_ID, edr_url
                           (deterministic PDS URL via src.ctx_edr, no network), the SeamMap
                           INCIDENCE/EMISSION/IMAGE_TIME (see caveat), n_tiles, tiles.
  frame_tile_map.csv     — long form, one row per (frame, tile) it must be rendered into.

Purely local + fast: `frame_table(tile)` reads the per-tile cached SeamMap gpkg (already built by
f_h4_buildprep 2026-07-11) and `edr_url()` is a string template — no CTX/GDAL heavy I/O, no network.
Unique frames = union of per-tile PRODUCT_IDs (no cross-tile geometry dissolve needed for the list).

CAVEAT (PLAN_FBuild §3 / V2): the SeamMap INCIDENCE has a known decimal-shift bug class
(e.g. P20_008839), so it is carried here only for reference/flagging — Stage B must resolve
incidence from the PDS volume indexes (scripts/probes/_f_leg_b_pds_incidence.py machinery) and
fail loudly on gaps. This script does NOT trust or fix SeamMap incidence.

Run (CPU, ~1-2 min; conda not on PATH):
  C:\\Users\\brian\\anaconda3\\Scripts\\conda.exe run --no-capture-output -n geospatial \
      python -u scripts/f_build_framelist.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/pandas

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.map_region import BLOCK_TILES
from src.ctx_edr import frame_table

OUT = REPO / "reports" / "figures"
META_COLS = ["VOLUME_ID", "EMISSION", "INCIDENCE", "IMAGE_TIME", "edr_url"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames: dict[str, dict] = {}   # pid -> metadata (first occurrence)
    tiles_of: dict[str, list[str]] = {}   # pid -> [tiles]
    long_rows = []                 # (pid, tile) pairs
    vol_conflicts = []

    n_frag = 0
    for tile in BLOCK_TILES:
        g = frame_table(tile)
        n_frag += len(g)
        print(f"  {tile}: {len(g)} frames", flush=True)
        for _, r in g.iterrows():
            pid = r["PRODUCT_ID"]
            long_rows.append({"PRODUCT_ID": pid, "tile": tile})
            tiles_of.setdefault(pid, []).append(tile)
            meta = {c: (r[c] if c in g.columns else np.nan) for c in META_COLS}
            if pid not in frames:
                frames[pid] = meta
            elif str(frames[pid]["VOLUME_ID"]) != str(meta["VOLUME_ID"]):
                vol_conflicts.append((pid, frames[pid]["VOLUME_ID"], meta["VOLUME_ID"]))

    pids = sorted(frames)
    frame_rows = []
    for pid in pids:
        m = frames[pid]
        tl = sorted(tiles_of[pid])
        frame_rows.append({
            "PRODUCT_ID": pid,
            "VOLUME_ID": m["VOLUME_ID"],
            "edr_url": m["edr_url"],
            "incidence_seammap": m["INCIDENCE"],   # UNTRUSTED — V2 resolves from PDS
            "emission_seammap": m["EMISSION"],
            "image_time": str(m["IMAGE_TIME"]),
            "n_tiles": len(tl),
            "tiles": ";".join(tl),
        })

    fdf = pd.DataFrame(frame_rows)
    ldf = pd.DataFrame(long_rows).sort_values(["PRODUCT_ID", "tile"]).reset_index(drop=True)
    fdf.to_csv(OUT / "region_frame_list.csv", index=False)
    ldf.to_csv(OUT / "frame_tile_map.csv", index=False)

    # ---- summary + sanity checks (plan expects ~907 unique / ~1,371 footprints) ----
    print(f"\n=== FRAME LIST ===", flush=True)
    print(f"unique frames        : {len(fdf)}  (plan expected ~907)")
    print(f"frame x tile rows    : {len(ldf)}  (plan expected ~1,371)")
    print(f"per-tile footprints  : {n_frag}")
    print(f"frames spanning >1 tile: {(fdf['n_tiles'] > 1).sum()}  "
          f"(max tiles for one frame: {fdf['n_tiles'].max()})")
    miss_vol = fdf["VOLUME_ID"].isna().sum()
    miss_url = fdf["edr_url"].isna().sum() + (fdf["edr_url"] == "").sum()
    print(f"missing VOLUME_ID    : {miss_vol}")
    print(f"missing edr_url      : {miss_url}")
    if vol_conflicts:
        print(f"⚠ VOLUME_ID conflicts across tiles: {len(vol_conflicts)} "
              f"(e.g. {vol_conflicts[:3]})")
    else:
        print("VOLUME_ID consistent across tiles: OK")
    inc_bad = fdf["incidence_seammap"].isna().sum()
    print(f"SeamMap incidence present: {len(fdf) - inc_bad}/{len(fdf)} "
          f"(UNTRUSTED — Stage B/V2 resolves from PDS volume indexes; do not use this column raw)")
    print(f"\nwrote {OUT / 'region_frame_list.csv'}")
    print(f"wrote {OUT / 'frame_tile_map.csv'}")
    if miss_vol or miss_url:
        raise SystemExit("FAIL: missing VOLUME_ID / edr_url — EDR resolution incomplete")


if __name__ == "__main__":
    main()
