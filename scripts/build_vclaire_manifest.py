"""Integrity pre-flight on the 40 vClaire detection folders + build the v2 manifest.

Two phases:

1. INTEGRITY — for each `{root}/{ObsId}/*-mask-nms.shp`: confirm the glob resolves,
   parse the .dbf/.shx headers to detect truncation (the ESP_028537_2270 failure mode),
   read the layer metadata (feature count, fields, CRS) without loading geometry, and
   record the SP1-bug status from the .prj. No network.

2. MANIFEST — for each ObsId: template the PDS RDR URLs from the orbit number, fetch the
   .LBL (authoritative CENTER_LATITUDE/LONGITUDE + sphere radius; also validates the URL),
   derive CTX_TileName via a floor-to-4-degree rule that is first *validated against the
   existing hirise_priority10.csv*, and pull BoulderLabel + quality/notes from the
   Mapping spreadsheet. Falls back to the spreadsheet corner coords (flagged) if a .LBL
   fetch fails.

Run:
    conda run -n geospatial python scripts/build_vclaire_manifest.py
"""
from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import pyogrio

from src import detections
from src import manifest as M
from src import pds_labels

DET_ROOT = Path(r"C:\Users\brian\Documents\PhD\HiRiseToCTXBoulders\hirise_40_vClaire")
XLSX = Path(r"C:\Users\brian\Downloads\Mapping_Images_33_36.xlsx")
EXISTING_MANIFEST = REPO_ROOT / "hirise_priority10.csv"
CACHE_DIR = REPO_ROOT / "cache"            # shared pds_labels cache (LBLs are detection-independent)
OUT_MANIFEST = REPO_ROOT / "hirise_40_vclaire.csv"

OUT_COLUMNS = [
    "ObsId", "ProductId", "LabelSource", "BoulderLabel", "TerrainNote",
    "CenterLat", "CenterLon_360", "CenterLon_180", "IncidenceAngle", "EmissionAngle",
    "CTX_TileName", "BrowseURL", "JP2_URL", "LabelURL", "QualityNote", "OriginalNote",
    # provenance / cross-check extras (not in REQUIRED_COLUMNS but harmless):
    "MapPixel_mpp", "CenterSource", "NPolygons", "PrjSP1Corrected", "IntegrityOK",
]


# ---------------------------------------------------------------------------
# URL + tile-name derivation
# ---------------------------------------------------------------------------

def orbit_folder(obs_id: str) -> str:
    """ESP_055714_2270 -> ORB_055700_055799."""
    orbit = int(obs_id.split("_")[1])
    lo = (orbit // 100) * 100
    return f"ORB_{lo:06d}_{lo + 99:06d}"


def pds_urls(obs_id: str) -> dict[str, str]:
    base = f"https://hirise.lpl.arizona.edu/PDS/RDR/ESP/{orbit_folder(obs_id)}/{obs_id}/{obs_id}_RED"
    return {
        "BrowseURL": f"https://www.uahirise.org/{obs_id}",
        "JP2_URL": f"{base}.JP2",
        "LabelURL": f"{base}.LBL",
    }


def ctx_tile_name(center_lon_360: float, center_lat: float) -> str:
    """Floor center lon/lat to the 4-degree Murray Lab tile, in the manifest E/W,N/S form.

    Validated against hirise_priority10.csv before use (see validate_tile_formula)."""
    lon180 = center_lon_360 if center_lon_360 <= 180 else center_lon_360 - 360.0
    lon_tile = int(math.floor(lon180 / 4.0) * 4)
    lat_tile = int(math.floor(center_lat / 4.0) * 4)
    lon_part = f"E{lon_tile:03d}" if lon_tile >= 0 else f"W{abs(lon_tile):03d}"
    lat_part = f"N{lat_tile:02d}" if lat_tile >= 0 else f"S{abs(lat_tile):02d}"
    return f"{lon_part}_{lat_part}"


def validate_tile_formula() -> None:
    """Reproduce every CTX_TileName in the existing manifest; abort if any mismatch."""
    df = pd.read_csv(EXISTING_MANIFEST)
    bad = []
    for _, r in df.iterrows():
        got = ctx_tile_name(float(r["CenterLon_360"]), float(r["CenterLat"]))
        if got != str(r["CTX_TileName"]):
            bad.append((r["ObsId"], r["CTX_TileName"], got))
    if bad:
        print("CTX_TileName formula MISMATCH against existing manifest:")
        for o, exp, got in bad:
            print(f"  {o}: manifest={exp} derived={got}")
        raise SystemExit("Refusing to build a manifest with an unvalidated tile formula.")
    print(f"CTX_TileName formula validated against {len(df)} existing rows.")


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def dbf_header(dbf_path: Path) -> tuple[int, int, int]:
    """Return (n_records, header_len, record_len) from the DBF's 32-byte header."""
    with open(dbf_path, "rb") as f:
        hdr = f.read(32)
    n_records = struct.unpack("<I", hdr[4:8])[0]
    header_len = struct.unpack("<H", hdr[8:10])[0]
    record_len = struct.unpack("<H", hdr[10:12])[0]
    return n_records, header_len, record_len


def integrity_check(obs_id: str) -> dict:
    """Per-folder integrity: glob, shx/dbf consistency, layer metadata, SP1 status."""
    out: dict = {"ObsId": obs_id, "ok": False}
    try:
        shp = M.find_shapefile(obs_id, DET_ROOT)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"find_shapefile: {e}"
        return out
    out["shp_name"] = shp.name
    shx = shp.with_suffix(".shx")
    dbf = shp.with_suffix(".dbf")
    # .shx implies the record count: (filesize - 100 header) / 8 bytes per record.
    shx_records = (shx.stat().st_size - 100) // 8 if shx.exists() else -1
    out["shx_records"] = shx_records
    # .dbf header declares its own record count + sizes; check the file isn't truncated.
    dbf_ok = True
    try:
        n_dbf, hlen, rlen = dbf_header(dbf)
        expected = hlen + n_dbf * rlen + 1  # +1 for the 0x1A EOF byte
        actual = dbf.stat().st_size
        out["dbf_records_declared"] = n_dbf
        out["dbf_record_len"] = rlen
        out["dbf_expected_bytes"] = expected
        out["dbf_actual_bytes"] = actual
        # allow a tiny slack for the trailing EOF byte
        dbf_ok = abs(actual - expected) <= 1 and n_dbf == shx_records
    except Exception as e:  # noqa: BLE001
        out["dbf_error"] = str(e)
        dbf_ok = False
    out["dbf_ok"] = dbf_ok
    # --- .shp byte-completeness (R23, added 2026-08-06) -------------------------------
    # The three checks above are all satisfied by a shapefile whose .shp is truncated but
    # whose .dbf and .shx are intact: the record count agrees, the .dbf is self-consistent,
    # and pyogrio happily reports every feature because the .shx says they exist. That is
    # exactly how ESP_017355_2260 / ESP_046803_2325 / ESP_068483_2280 passed this gate
    # while missing 354/132/173 MB of polygon bytes, surfacing downstream only as "null
    # geometry" (R23; DECISIONS 2026-08-06o). ESP_028537_2270 was caught only because its
    # .dbf was short too. Ask the .shp about itself.
    integrity = detections.inspect_shapefile_integrity(shp)
    out["shp_status"] = integrity.get("status")
    out["shp_declared_bytes"] = integrity.get("declared_bytes")
    out["shp_actual_bytes"] = integrity.get("actual_bytes")
    out["shp_missing_bytes"] = integrity.get("missing_bytes")
    out["shp_records_present"] = integrity.get("n_records_present")
    shp_ok = integrity.get("status") == "complete"
    out["shp_ok"] = shp_ok
    if not shp_ok:
        out["shp_error"] = (
            f"{integrity.get('status')}: declares {integrity.get('declared_bytes')} bytes, "
            f"{integrity.get('actual_bytes')} present "
            f"({(integrity.get('missing_bytes') or 0) / 1e6:.1f} MB missing)"
        )
    # Layer metadata without loading geometry.
    try:
        info = pyogrio.read_info(shp)
        out["info_features"] = int(info["features"])
        out["fields"] = list(info["fields"])
        out["crs_name"] = info.get("crs")
    except Exception as e:  # noqa: BLE001
        out["info_error"] = str(e)
    # SP1 bug fingerprint from the .prj text.
    prj = shp.with_suffix(".prj").read_text(encoding="latin-1")
    out["prj_d_unnamed"] = "D_unnamed" in prj
    import re
    m = re.search(r'Standard_Parallel_1",([-\d.eE]+)', prj)
    out["prj_sp1"] = float(m.group(1)) if m else None
    # `ok` gates INGESTION -- "can this folder be read at all". A byte-truncated .shp is
    # readable (GDAL returns every .shx-indexed record, with null geometry for the missing
    # tail), and the three affected images are RETAINED by decision (Brian, 2026-08-06:
    # retain + document, temporary pending the v3 re-detection). So `shp_ok` is reported
    # and printed loudly but deliberately NOT folded into `ok` -- doing so would silently
    # shrink the manifest 39 -> 36 rows and invert that decision. See DECISIONS 2026-08-06o.
    out["ok"] = bool(
        dbf_ok and "info_features" in out and out.get("info_features", -1) == shx_records
    )
    return out


# ---------------------------------------------------------------------------
# Spreadsheet lookup
# ---------------------------------------------------------------------------

def load_spreadsheet_lookup() -> dict[str, dict]:
    """ObsId -> {BoulderLabel, quality, notes, corner_lat, corner_lon_180/360, map_pixel}.

    Union across all sheets; first non-null wins. The 'Overall... ' column carries the
    Boulder rich/poor label where present; otherwise we fall back to the per-sheet
    quality column.
    """
    xl = pd.ExcelFile(XLSX)
    lut: dict[str, dict] = {}
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if "Image Name" not in df.columns:
            continue
        cols = {c.strip(): c for c in df.columns}  # tolerate trailing spaces
        for _, r in df.iterrows():
            obs = str(r["Image Name"]).strip()
            if not obs.startswith("ESP_"):
                continue
            rec = lut.setdefault(obs, {})
            def _get(name):
                col = cols.get(name)
                return r[col] if col is not None and pd.notna(r[col]) else None
            overall = _get("Overall...")
            quality = _get("Quality of boulders") or _get("Boulders?")
            label = overall if isinstance(overall, str) and "Boulder" in overall else rec.get("BoulderLabel")
            rec.setdefault("BoulderLabel", label)
            if rec.get("BoulderLabel") is None and label:
                rec["BoulderLabel"] = label
            rec.setdefault("QualityNote", quality)
            rec.setdefault("OriginalNote", _get("Notes") or _get("Notes "))
            lat = _get("corner1_latitude")
            lon = _get("corner1_longitude")
            if lat is not None and "corner_lat" not in rec:
                rec["corner_lat"] = float(lat)
            if lon is not None and "corner_lon" not in rec:
                rec["corner_lon"] = float(lon)
            mp = _get("map_pixel")
            if mp is not None and "map_pixel" not in rec:
                rec["map_pixel"] = float(mp)
    return lut


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    obs_ids = sorted(d.name for d in DET_ROOT.iterdir() if d.is_dir())
    print(f"vClaire folders: {len(obs_ids)}\n")

    validate_tile_formula()
    existing_obs = set(pd.read_csv(EXISTING_MANIFEST)["ObsId"])

    print("\n========== PHASE 1: integrity ==========")
    integ: dict[str, dict] = {}
    for obs in obs_ids:
        r = integrity_check(obs)
        integ[obs] = r
        flag = "OK " if r["ok"] else "BAD"
        feats = r.get("info_features", r.get("shx_records", "?"))
        extra = ""
        if not r["ok"]:
            extra = (
                r.get("error") or r.get("info_error") or r.get("dbf_error")
                or r.get("shp_error") or "dbf/shx mismatch"
            )
            extra = f"  <-- {extra}"
        sp1 = "SP1bug" if r.get("prj_d_unnamed") else "prjOK"
        if r.get("shp_status") == "truncated" and r["ok"]:
            extra += (
                f"  <-- .shp TRUNCATED, {(r.get('shp_missing_bytes') or 0) / 1e6:.1f} MB missing "
                f"({r.get('shp_records_present')}/{r.get('shx_records')} records present) "
                "[RETAINED per DECISIONS 2026-08-06o]"
            )
        print(f"  [{flag}] {obs}  n_polys={feats:<9} {sp1}{extra}")
    n_bad = sum(1 for r in integ.values() if not r["ok"])
    print(f"\nIntegrity: {len(obs_ids) - n_bad}/{len(obs_ids)} OK, {n_bad} need attention.")

    # R23: a truncated .shp is READABLE, so it does not fail `ok` and is not excluded --
    # but it silently truncates that image's label basis at a high confidence floor, so it
    # must never pass unremarked. See DECISIONS 2026-08-06o.
    truncated = [o for o in obs_ids if integ[o].get("shp_status") == "truncated"]
    if truncated:
        print(
            f"\n  !! {len(truncated)} .shp file(s) are BYTE-TRUNCATED (readable, retained, "
            "label basis affected):"
        )
        for o in truncated:
            r = integ[o]
            print(
                f"       {o}: {(r.get('shp_missing_bytes') or 0) / 1e6:7.1f} MB missing of "
                f"{(r.get('shp_declared_bytes') or 0) / 1e6:7.1f} MB; "
                f"{r.get('shp_records_present')}/{r.get('shx_records')} records present"
            )
        print(
            "     Records are stored score-descending, so the survivors are the "
            "highest-scoring\n     detections and these images' abundance LEVEL is biased "
            "low. Rank-only use is safe.\n     Remedy decided: retain + document, temporary "
            "pending v3 (DECISIONS 2026-08-06o)."
        )

    print("\n========== PHASE 2: manifest ==========")
    lut = load_spreadsheet_lookup()
    in_xlsx = sum(1 for o in obs_ids if o in lut)
    print(f"Spreadsheet rows matched: {in_xlsx}/{len(obs_ids)}")

    # Integrity-failed folders can't be ingested (truncated/corrupt or missing the
    # mask-nms variant). Exclude them from the written manifest so the pipeline never
    # tries to read them; report which were dropped.
    excluded = [o for o in obs_ids if not integ[o]["ok"]]
    manifest_obs = [o for o in obs_ids if integ[o]["ok"]]
    if excluded:
        print(f"Excluding {len(excluded)} integrity-failed ObsId(s) from the manifest: {excluded}")

    rows = []
    for obs in manifest_obs:
        u = pds_urls(obs)
        rec = lut.get(obs, {})
        center_source = "pds_footprint"
        center_lat = center_lon_360 = None
        # Authoritative image center = midpoint of the PDS footprint extents
        # (MIN/MAX_LATITUDE, EASTERNMOST/WESTERNMOST_LONGITUDE). NOT projection_origin,
        # which is the map projection's central meridian / standard parallel (rounded;
        # used only for the SP1 .prj fix). Verified against the spreadsheet corners +
        # the v1 manifest tiles (scripts/probes/_diag_lbl_center.py).
        try:
            pds_labels.fetch_label(obs, u["LabelURL"], CACHE_DIR)
            fp = pds_labels.image_footprint(obs, CACHE_DIR)
            center_lat = (fp["max_lat_deg"] + fp["min_lat_deg"]) / 2.0
            e_lon, w_lon = fp["east_lon_deg"] % 360.0, fp["west_lon_deg"] % 360.0
            # Narrow HiRISE swaths don't wrap the antimeridian; guard anyway.
            if abs(e_lon - w_lon) > 180.0:
                e_lon, w_lon = e_lon, w_lon + 360.0 if w_lon < e_lon else w_lon
            center_lon_360 = ((e_lon + w_lon) / 2.0) % 360.0
        except Exception as e:  # noqa: BLE001
            center_source = f"xlsx_corner({type(e).__name__})"
            if rec.get("corner_lat") is not None and rec.get("corner_lon") is not None:
                center_lat = rec["corner_lat"]
                center_lon_360 = rec["corner_lon"] % 360.0
            else:
                print(f"  !! {obs}: LBL fetch failed AND no spreadsheet coords ({e})")
        # Cross-check against the spreadsheet corner (informational; corner != center).
        if center_lat is not None and rec.get("corner_lat") is not None:
            dlat = abs(center_lat - rec["corner_lat"])
            dlon = abs(((center_lon_360 - rec["corner_lon"] % 360.0) + 180) % 360 - 180)
            if dlat > 1.0 or dlon > 1.0:
                print(f"  ~ {obs}: footprint-center vs spreadsheet-corner differ "
                      f"(dlat={dlat:.2f} dlon={dlon:.2f})")
        ctx_tile = ctx_tile_name(center_lon_360, center_lat) if center_lat is not None else ""
        lon180 = (center_lon_360 - 360.0) if (center_lon_360 is not None and center_lon_360 > 180) else center_lon_360
        bl = rec.get("BoulderLabel")
        bl = bl if isinstance(bl, str) and "Boulder" in bl else "unknown"
        ig = integ.get(obs, {})
        rows.append({
            "ObsId": obs,
            "ProductId": f"{obs}_RED",
            "LabelSource": "spreadsheet" if obs in lut else "none",
            "BoulderLabel": bl,
            "TerrainNote": "",
            "CenterLat": round(center_lat, 4) if center_lat is not None else "",
            "CenterLon_360": round(center_lon_360, 4) if center_lon_360 is not None else "",
            "CenterLon_180": round(lon180, 4) if lon180 is not None else "",
            "IncidenceAngle": "",
            "EmissionAngle": "",
            "CTX_TileName": ctx_tile,
            "BrowseURL": u["BrowseURL"],
            "JP2_URL": u["JP2_URL"],
            "LabelURL": u["LabelURL"],
            "QualityNote": rec.get("QualityNote") or "",
            "OriginalNote": rec.get("OriginalNote") or "",
            "MapPixel_mpp": rec.get("map_pixel") or "",
            "CenterSource": center_source,
            "NPolygons": ig.get("info_features", ig.get("shx_records", "")),
            "PrjSP1Corrected": bool(ig.get("prj_d_unnamed")),
            "IntegrityOK": bool(ig.get("ok")),
        })

    df = pd.DataFrame(rows)[OUT_COLUMNS]
    df.to_csv(OUT_MANIFEST, index=False)
    print(f"\nWrote {OUT_MANIFEST}  ({len(df)} rows)")
    # Summaries the user will want.
    print("\nBoulderLabel counts:", df["BoulderLabel"].value_counts().to_dict())
    print("CenterSource counts:", df["CenterSource"].value_counts().to_dict())
    overlap = sorted(set(obs_ids) & existing_obs)
    print(f"Overlap with existing priority10 ({len(overlap)}): {overlap}")
    missing_xlsx = [o for o in obs_ids if o not in lut]
    if missing_xlsx:
        print(f"NOT found in spreadsheet ({len(missing_xlsx)}): {missing_xlsx}")
    bad = [o for o, r in integ.items() if not r["ok"]]
    if bad:
        print(f"Integrity FAILURES ({len(bad)}): {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
