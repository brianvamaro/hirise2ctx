"""Stage 7.0: fetch the HiRISE COLOR.JP2 + COLOR.LBL for the feasibility trio.

PDS publishes a single 3-band COLOR.JP2 per observation (IR/RED/BG, I/F units, 0.25 m/px,
~1-3 km central swath). The PLAN_Compositional.md §2.1 sentence about separate IRB+RGB
products was wrong; the verified PDS layout is one COLOR.JP2 alongside the panchromatic
RED.JP2.

Run via:
    conda run -n geospatial python scripts/probes/_fetch_color.py
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

import truststore  # SSL fix per memory note conda_windows_ssl

truststore.inject_into_ssl()

CACHE_DIR = Path("cache_v2/hirise_color")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# (ObsId, orbit-range-dir, role) tuples. URL convention:
#   https://hirise.lpl.arizona.edu/PDS/RDR/ESP/ORB_{orbit}/{ObsId}/{ObsId}_COLOR.JP2
TRIO = [
    ("ESP_042964_2160", "ORB_042900_042999", "high-density positive (AUC 0.91)"),
    ("ESP_054000_2255", "ORB_054000_054099", "anti-signal #1 (AUC 0.40)"),
    ("ESP_055253_2245", "ORB_055200_055299", "anti-signal #2 (AUC 0.42)"),
]


def _download(url: str, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 1_000:
        print(f"  [cached] {out_path.name} ({out_path.stat().st_size:,} bytes)")
        return
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    req = urllib.request.Request(url, headers={"User-Agent": "hirise2ctx/0.1"})
    print(f"  [fetching] {url}")
    with urllib.request.urlopen(req, timeout=300) as resp, tmp.open("wb") as f:
        shutil.copyfileobj(resp, f, length=1 << 20)
    tmp.replace(out_path)
    print(f"  [done] {out_path.name} ({out_path.stat().st_size:,} bytes)")


def main() -> int:
    for obs_id, orbit, role in TRIO:
        print(f"\n{obs_id} -- {role}")
        base = f"https://hirise.lpl.arizona.edu/PDS/RDR/ESP/{orbit}/{obs_id}"
        for ext in ("LBL", "JP2"):
            _download(f"{base}/{obs_id}_COLOR.{ext}", CACHE_DIR / f"{obs_id}_COLOR.{ext}")
    print("\nAll COLOR.JP2 + COLOR.LBL files cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
