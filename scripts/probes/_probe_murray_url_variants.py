"""For the western-longitude / southern-latitude tiles in the priority10 manifest,
probe candidate URL formats to discover the one Murray Lab actually serves."""
from __future__ import annotations

import sys
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import pds_labels  # noqa: F401  (truststore side effect)

BASE = "https://murray-lab.caltech.edu/CTX/V01/tiles/MurrayLab_GlobalCTXMosaic_V01_{}.zip"

# (manifest_form, candidate URL substrings to try)
TILES = {
    "W040_N20": [
        # Various positive/negative/padded combinations
        "E-040_N20",
        "E-40_N20",
        "W040_N20",
        "W40_N20",
        "E-040_N020",
    ],
    "E152_S08": [
        "E152_N-08",
        "E152_N-8",
        "E152_S08",
        "E152_S8",
    ],
    "E000_S28": [
        "E000_N-28",
        "E0_N-28",
        "E000_S28",
    ],
    "W052_N36": [
        "E-052_N36",
        "E-52_N36",
        "W052_N36",
    ],
    "W024_N28": [
        "E-024_N28",
        "E-24_N28",
        "W024_N28",
    ],
}


def head(url: str, timeout: float = 15.0) -> int:
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "hirise2ctx/0.1 probe"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def main():
    for manifest, candidates in TILES.items():
        print(f"\n--- {manifest} ---")
        for c in candidates:
            url = BASE.format(c)
            code = head(url)
            tag = "OK" if code == 200 else f"  {code}"
            print(f"  {tag}  {c:>14}  {url}")
            if code == 200:
                break  # found the live one; move to next manifest tile


if __name__ == "__main__":
    main()
