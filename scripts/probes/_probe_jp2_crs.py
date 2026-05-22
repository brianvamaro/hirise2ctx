"""One-off probe: compare JP2 embedded CRS vs Stage 1 corrected CRS for cached JP2s.

Tells us whether the upstream HiRISE SP1=0 bug also poisons the JP2 metadata (which
would affect Stage 2 mask building + Stage 3 phase correlation).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rasterio


def sp1_value(wkt: str) -> float | None:
    if not wkt:
        return None
    m = re.search(r"standard_parallel_1[\"']?\s*,\s*(-?\d+\.?\d*)", wkt, re.IGNORECASE)
    return float(m.group(1)) if m else None


def main():
    jp2_dir = REPO_ROOT / "cache" / "hirise_jp2"
    decimated_dir = REPO_ROOT / "cache" / "hirise_decimated"
    stage1_dir = REPO_ROOT / "cache" / "reprojected_detections"
    for jp2 in sorted(jp2_dir.glob("*_RED.JP2")):
        obs = jp2.stem.replace("_RED", "")
        with rasterio.open(jp2) as ds:
            jp2_wkt = ds.crs.to_wkt() if ds.crs else None
        jp2_sp1 = sp1_value(jp2_wkt)
        sidecar_path = stage1_dir / f"{obs}.json"
        s1_sp1 = None
        status = "?"
        s1_wkt = ""
        if sidecar_path.exists():
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            s1_wkt = sidecar["source_crs_wkt"]
            s1_sp1 = sp1_value(s1_wkt)
            status = sidecar.get("correction", {}).get("status", "?")
        dec_path = decimated_dir / f"{obs}_5mpp_full.tif"
        dec_sp1 = None
        if dec_path.exists():
            with rasterio.open(dec_path) as ds:
                dec_wkt = ds.crs.to_wkt() if ds.crs else ""
            dec_sp1 = sp1_value(dec_wkt)
        print(
            f"{obs}: Stage1={status!r}  JP2_SP1={jp2_sp1}  decimated_SP1={dec_sp1}  Stage1_SP1={s1_sp1}"
        )
        if s1_wkt:
            # Show the Stage 1 corrected projection latitude as it appears in the WKT
            # (pyproj may have normalized the keyword name).
            m = re.search(r"latitude_of_origin[\"']?\s*,\s*(-?\d+\.?\d*)", s1_wkt, re.IGNORECASE)
            if m:
                print(f"    Stage1 latitude_of_origin = {m.group(1)}")
            m = re.search(r"Latitude of 1st standard parallel[\"']?[^,]*,\s*(-?\d+\.?\d*)", s1_wkt, re.IGNORECASE)
            if m:
                print(f"    Stage1 1st-standard-parallel parameter = {m.group(1)}")


if __name__ == "__main__":
    main()
