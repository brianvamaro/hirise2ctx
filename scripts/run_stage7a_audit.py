"""Stage 7a audit -- HEAD-probe COLOR.JP2 + COLOR.LBL availability across all v2 ObsIds.

For each row in `hirise_40_vclaire.csv` we derive the PDS COLOR product URL from the
panchromatic `JP2_URL` (substituting `_RED.JP2` -> `_COLOR.JP2`), then issue HTTP HEAD
requests against PDS to see what's actually there. HEAD is much cheaper than parsing
the directory autoindex and gives the same answer (200 vs 404) plus Content-Length.

Writes `cache_v2/hirise_color/coverage.parquet` with one row per ObsId:
  obs_id, color_jp2_url, color_jp2_status, color_jp2_bytes,
  color_lbl_url, color_lbl_status, color_lbl_bytes,
  has_color (bool), audited_at_iso, error

Idempotent -- re-running re-audits and overwrites. Runtime ~40-60 s (40 HEADs + 40 HEADs).

Run via:
    conda run --no-capture-output -n geospatial python -u scripts/run_stage7a_audit.py
"""
from __future__ import annotations

import datetime as _dt
import functools
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import truststore

truststore.inject_into_ssl()

print = functools.partial(print, flush=True)

MANIFEST = Path("hirise_40_vclaire.csv")
CACHE_DIR = Path("cache_v2/hirise_color")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT = CACHE_DIR / "coverage.parquet"


def _head(url: str, timeout: float = 30) -> tuple[int, int]:
    """Return (HTTP status, Content-Length). status == -1 on transport error.

    PDS returns Content-Length for both JP2s and LBLs. For 404s the status is 404
    and Content-Length is whatever the error body is -- we report -1 then.
    """
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "hirise2ctx/0.1 (stage7a-audit)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), int(resp.headers.get("Content-Length", -1))
    except urllib.error.HTTPError as e:
        return int(e.code), -1
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  [transport-error] {type(e).__name__}: {e} -- {url}")
        return -1, -1


def _color_urls(jp2_url: str) -> tuple[str, str]:
    """`{ObsId}_RED.JP2` -> (`{ObsId}_COLOR.JP2`, `{ObsId}_COLOR.LBL`)."""
    color_jp2 = jp2_url.replace("_RED.JP2", "_COLOR.JP2")
    color_lbl = color_jp2[:-3] + "LBL"
    return color_jp2, color_lbl


def main() -> int:
    manifest = pd.read_csv(MANIFEST)
    print(f"Manifest: {len(manifest)} ObsIds")
    rows = []
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    for i, r in manifest.iterrows():
        obs_id = r["ObsId"]
        jp2 = r["JP2_URL"]
        color_jp2, color_lbl = _color_urls(jp2)
        s_jp2, sz_jp2 = _head(color_jp2)
        if s_jp2 == 200:
            s_lbl, sz_lbl = _head(color_lbl)
        else:
            s_lbl, sz_lbl = -1, -1
        has = s_jp2 == 200
        rows.append({
            "obs_id": obs_id,
            "color_jp2_url": color_jp2,
            "color_jp2_status": s_jp2,
            "color_jp2_bytes": sz_jp2,
            "color_lbl_url": color_lbl,
            "color_lbl_status": s_lbl,
            "color_lbl_bytes": sz_lbl,
            "has_color": has,
            "audited_at_iso": now,
        })
        flag = "OK" if has else f"MISS ({s_jp2})"
        print(f"  [{i+1:>2}/{len(manifest)}] {obs_id}  {flag:<8}  "
              f"{sz_jp2 if has else '-':>12}")
    df = pd.DataFrame(rows)
    df.to_parquet(OUT)
    n_ok = int(df["has_color"].sum())
    print(f"\nWrote {OUT}")
    print(f"Coverage: {n_ok}/{len(df)} ({100*n_ok/len(df):.1f}%)")
    print(f"Total COLOR.JP2 fetch size if all kept: "
          f"{df.loc[df['has_color'], 'color_jp2_bytes'].sum()/1024/1024:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
