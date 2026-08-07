"""Stage 1 — ingest BoulderNet detections, reproject to a common CTX CRS, cache.

The reprojection is intentionally boring: geopandas + pyproj read each shapefile's own
`.prj` (which carries the per-image local-Mars-radius equirectangular CRS) and project
to the target CTX CRS. The whole point of this stage is to NEVER bypass the source CRS
or hardcode a sphere radius — see CLAUDE.md §3.3.

Exception: 4 of 10 BoulderNet `.prj` files in the priority10 manifest are mis-labelled
with `Standard_Parallel_1=0` (datum `D_unnamed`) even though the geometry was actually
generated with the PDS-declared projection latitude. We detect that case and override
SP1 with the authoritative value from the HiRISE `.LBL` (CENTER_LATITUDE). See
`DECISIONS.md` 2026-05-20 entries.
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
from pyproj import CRS

from . import manifest as manifest_mod
from . import pds_labels

CACHE_SUBDIR = "reprojected_detections"

# Matches ESRI-WKT1-style `PARAMETER["Standard_Parallel_1",<num>]`
_SP1_PATTERN = re.compile(r'PARAMETER\["Standard_Parallel_1",([-\d.eE]+)\]')
# Bug fingerprint: BoulderNet's mis-labelled exports use the `D_unnamed` / `unnamed`
# placeholder strings instead of the canonical `D_MARS` / `MARS_localRadius`.
_BAD_DATUM_FINGERPRINT = re.compile(r'DATUM\["D_unnamed"', re.IGNORECASE)
# Tolerance: if the .prj's SP1 is within this many degrees of the manifest CenterLat,
# trust the .prj. The buggy files are off by tens of degrees; the good files are within
# ~5°. 15° is a generous margin that cleanly separates the two regimes.
_SP1_TOLERANCE_DEG = 15.0

# Minimum dropped-row count before `describe_null_geometry_drop` will assert a rank
# truncation. Below it, "every dropped row scores at or below every kept row" is satisfied
# by chance (and always by tied scores), which would raise a false "LEVEL is biased low"
# alarm. The four real vClaire cases drop 291k-875k rows, so this is nowhere near binding.
_MIN_DROPPED_FOR_RANK_VERDICT = 100


def _suspect_sp1(prj_text: str, image_lat_deg: float) -> tuple[bool, float | None]:
    """Return `(is_buggy, current_sp1)`. `is_buggy` is True iff the .prj looks like a
    BoulderNet mis-labelled export AND its SP1 disagrees with `image_lat_deg` by more
    than `_SP1_TOLERANCE_DEG`.
    """
    sp1_match = _SP1_PATTERN.search(prj_text)
    if not sp1_match:
        return False, None
    current_sp1 = float(sp1_match.group(1))
    has_bad_datum = bool(_BAD_DATUM_FINGERPRINT.search(prj_text))
    far_from_image = abs(current_sp1 - image_lat_deg) > _SP1_TOLERANCE_DEG
    return (has_bad_datum and far_from_image), current_sp1


def _override_sp1(prj_text: str, new_sp1_deg: float) -> str:
    """Return `prj_text` with `Standard_Parallel_1` set to `new_sp1_deg`."""
    return _SP1_PATTERN.sub(f'PARAMETER["Standard_Parallel_1",{new_sp1_deg}]', prj_text)


def read_detection_shapefile(
    obs_id: str,
    detections_root: str | Path,
    *,
    manifest_row=None,
    cache_dir: str | Path | None = None,
) -> tuple[gpd.GeoDataFrame, dict]:
    """Read the BoulderNet shapefile for `obs_id`, returning the GeoDataFrame in its
    native CRS (after correction, if needed) plus a small `correction` dict for provenance.

    When `manifest_row` and `cache_dir` are given, the function checks for the
    BoulderNet `.prj` SP1 mis-labelling bug and, if found, fetches the PDS `.LBL` for
    `obs_id` and overrides SP1 with the authoritative `CENTER_LATITUDE` from the label.
    """
    shp = manifest_mod.find_shapefile(obs_id, detections_root)
    prj_path = shp.with_suffix(".prj")
    original_prj = prj_path.read_text(encoding="latin-1")
    correction: dict = {"status": "trusted_prj"}

    if manifest_row is not None and cache_dir is not None:
        image_lat = float(manifest_row["CenterLat"])
        is_buggy, current_sp1 = _suspect_sp1(original_prj, image_lat)
        if is_buggy:
            pds_labels.fetch_label(obs_id, manifest_row["LabelURL"], cache_dir)
            origin = pds_labels.projection_origin(obs_id, cache_dir)
            corrected_prj = _override_sp1(original_prj, origin["center_lat_deg"])
            gdf = gpd.read_file(shp)
            gdf = gdf.set_crs(corrected_prj, allow_override=True)
            correction = {
                "status": "sp1_corrected_from_pds_label",
                "original_sp1_deg": current_sp1,
                "corrected_sp1_deg": float(origin["center_lat_deg"]),
                "pds_center_lat_deg": float(origin["center_lat_deg"]),
                "pds_center_lon_deg": float(origin["center_lon_deg"]),
                "pds_a_axis_km": float(origin["a_axis_km"]),
            }
            return gdf, correction

    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        raise RuntimeError(
            f"{shp}: shapefile has no CRS (.prj missing or unreadable). "
            "Reprojection requires a known source CRS — refusing to guess."
        )
    return gdf, correction


def reproject_to_target(gdf: gpd.GeoDataFrame, target_crs: str | CRS) -> gpd.GeoDataFrame:
    """Reproject `gdf` to `target_crs`. Source CRS comes from the GeoDataFrame, not args."""
    target = CRS.from_user_input(target_crs)
    return gdf.to_crs(target)


def inspect_shapefile_integrity(shp: str | Path) -> dict:
    """Is this `.shp` byte-complete? Compare its own header-declared length to disk.

    A shapefile's 100-byte header stores the total file length at bytes 24-27 as a
    big-endian count of 16-bit words, so a partially-copied `.shp` is *self-diagnosing*:
    the declared length exceeds the bytes actually present. The `.shx` index and `.dbf`
    table are small and typically finish copying, which is why a truncated `.shp` reads
    without error and simply yields NULL geometry for every record whose bytes are
    missing -- exactly the R23 signature (DECISIONS 2026-08-06o): three vClaire exports
    were short by 354/132/173 MB, and because BoulderNet writes records in
    score-DESCENDING order, the surviving prefix is the highest-scoring detections and
    the label basis is silently truncated at a per-image confidence floor.

    Returns a provenance dict; `status` is one of "complete", "truncated", or
    "unreadable". Never raises on a malformed/missing file -- Stage 1 records the finding
    rather than failing, because the truncated cohort is retained by decision (Brian,
    2026-08-06) pending the v3 re-detection.
    """
    shp = Path(shp)
    out: dict = {"status": "unreadable", "shp_path": str(shp)}
    try:
        with open(shp, "rb") as fh:
            header = fh.read(100)
        if len(header) < 100 or int.from_bytes(header[0:4], "big") != 9994:
            out["note"] = "not a shapefile header (magic != 9994)"
            return out
        declared = int.from_bytes(header[24:28], "big") * 2
        actual = shp.stat().st_size
        if declared < 100:
            # A writer stamps the true length in at close; a length below the header size
            # means the header itself is bogus/unwritten. Never certify that as complete.
            status = "suspect_header"
        elif actual < declared:
            status = "truncated"
        elif actual > declared:
            # Not the R23 failure mode (a partial copy is always short), but a stale or
            # unwritten length field must not be silently reported as complete.
            status = "length_mismatch"
        else:
            status = "complete"
        out.update(
            status=status,
            declared_bytes=int(declared),
            actual_bytes=int(actual),
            missing_bytes=int(max(0, declared - actual)),
        )
        # The .shx is a flat table of (offset, length) pairs in 16-bit words. It lets us
        # say exactly how many records survived, without parsing geometry. The .shx has the
        # same self-describing header, so check it too -- a copy can stop early in the
        # index as well (ESP_028537_2270's .dbf did), and a short .shx would make
        # `n_records_present` an undercount rather than an exact answer.
        shx = shp.with_suffix(".shx")
        if shx.exists():
            raw_all = shx.read_bytes()
            raw = raw_all[100:]
            n_index = len(raw) // 8
            out["n_records_index"] = int(n_index)
            shx_declared = (
                int.from_bytes(raw_all[24:28], "big") * 2 if len(raw_all) >= 100 else 0
            )
            out["shx_status"] = (
                "complete" if len(raw_all) >= shx_declared >= 100 else "truncated"
            )
            if status == "truncated" and n_index:
                import numpy as _np

                idx = _np.frombuffer(raw[: n_index * 8], dtype=">i4").reshape(-1, 2)
                ends = idx[:, 0].astype("int64") * 2 + idx[:, 1].astype("int64") * 2 + 8
                out["n_records_present"] = int((ends <= actual).sum())
                if out["shx_status"] != "complete":
                    # The index is short, so records past its end are uncounted.
                    out["n_records_present_is_lower_bound"] = True
    except OSError as exc:  # unreadable/missing -- record, don't crash Stage 1
        out["note"] = f"{type(exc).__name__}: {exc}"
    return out


def describe_null_geometry_drop(gdf: gpd.GeoDataFrame, *, score_col: str = "score") -> dict | None:
    """Characterise the population `drop_null_geometries` is about to remove.

    R23's lesson, recorded verbatim in the register: *when a filter drops a third to
    two-thirds of the rows, characterise the dropped population on every available
    column, not just the one you suspect.* The original diagnostic broke the nulls down
    by `is_at_edge` and never looked at `score`, so a rank truncation was recorded as
    benign density hygiene for two months.

    Returns None when nothing is dropped. Otherwise a dict carrying the kept/dropped
    `score` distributions and `is_rank_truncation` -- True when every dropped row scores
    at or below every kept row, which is the fingerprint of a score-ordered truncation
    rather than sparse export noise.
    """
    if len(gdf) == 0:
        return None
    null_mask = (gdf.geometry.isna() | gdf.geometry.is_empty).to_numpy()
    n_dropped = int(null_mask.sum())
    if n_dropped == 0:
        return None

    out: dict = {
        "n_rows": int(len(gdf)),
        "n_dropped": n_dropped,
        "n_kept": int(len(gdf) - n_dropped),
        "dropped_fraction": float(n_dropped / len(gdf)),
    }
    if score_col not in gdf.columns:
        out["note"] = f"no {score_col!r} column; dropped population not characterised"
        return out

    import numpy as _np
    import pandas as _pd

    # Never raise: this runs inside the Stage-1 producer, and a manifest row whose export
    # stores `score` as a DBF character field must still ingest (the manifest-driven
    # invariant). Record the finding instead.
    scores = _pd.to_numeric(gdf[score_col], errors="coerce").to_numpy(dtype=float)
    if not _np.isfinite(scores).any():
        out["note"] = f"{score_col!r} is not numeric; dropped population not characterised"
        return out
    dropped, kept = scores[null_mask], scores[~null_mask]
    finite_d, finite_k = dropped[_np.isfinite(dropped)], kept[_np.isfinite(kept)]

    def _summary(a):
        if a.size == 0:
            return None
        q = _np.percentile(a, [1, 25, 50, 75, 99])
        return {
            "n": int(a.size), "min": float(a.min()), "max": float(a.max()),
            "mean": float(a.mean()), "p1": float(q[0]), "p25": float(q[1]),
            "median": float(q[2]), "p75": float(q[3]), "p99": float(q[4]),
        }

    out["score_column"] = score_col
    out["dropped_score"] = _summary(finite_d)
    out["kept_score"] = _summary(finite_k)
    if finite_d.size and finite_k.size:
        out["realised_score_floor"] = float(finite_k.min())
        out["dropped_score_max"] = float(finite_d.max())
        # Record the separation so a reader can see a zero-width (tie/degenerate) split.
        out["kept_minus_dropped_gap"] = float(finite_k.min() - finite_d.max())
        if n_dropped < _MIN_DROPPED_FOR_RANK_VERDICT:
            # With a handful of dropped rows, "every dropped row scores at or below every
            # kept row" happens by chance (and always when all scores tie), so the loud
            # "LEVEL is biased low" alarm would fire on clean images. Withhold the verdict
            # rather than guess. The real cases drop 291k-745k rows.
            out["is_rank_truncation"] = None
            out["note"] = (
                f"only {n_dropped} dropped row(s) (< {_MIN_DROPPED_FOR_RANK_VERDICT}); too "
                "few to distinguish a rank truncation from chance"
            )
        else:
            # <= (not <) because the truncation cut lands *between* two records that may
            # share a score to float precision; the measured vClaire gap is +1e-6.
            out["is_rank_truncation"] = bool(finite_d.max() <= finite_k.min())
    return out


def drop_null_geometries(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, int]:
    """Drop rows with null or empty geometry. Returns (cleaned_gdf, n_dropped).

    A record can have a DBF row (score, id, ...) but no polygon. Such rows cannot be
    rasterized or centroid-counted, so Stage 4 would error/miscount; we drop them at
    ingest so the cached GPKG and its `n_polygons` reflect only real boulder outlines.

    **Treat ANY null geometry as suspicious until `inspect_shapefile_integrity` says
    otherwise.** This docstring used to describe up to ~67% null-geometry as a normal
    property of dense vClaire exports -- that mental model is what let R23 sit in the
    record as "benign density hygiene" for two months. Measured 2026-08-06: **36 of 39
    readable vClaire exports drop exactly zero rows**, and every null-geometry row in the
    cohort came from a **byte-truncated `.shp`** whose polygon bytes were never copied.
    Because BoulderNet writes records score-descending, dropping them truncates that
    image's label basis at a high confidence floor. See DECISIONS 2026-08-06o.

    This function's behaviour is unchanged and correct either way -- the diagnosis belongs
    to `inspect_shapefile_integrity` / `describe_null_geometry_drop`, which Stage 1 calls.
    """
    if len(gdf) == 0:
        return gdf, 0
    valid = ~(gdf.geometry.isna() | gdf.geometry.is_empty)
    n_dropped = int((~valid).sum())
    if n_dropped == 0:
        return gdf, 0
    return gdf.loc[valid].reset_index(drop=True), n_dropped


def cache_reprojected(
    gdf: gpd.GeoDataFrame,
    obs_id: str,
    cache_dir: str | Path,
    *,
    source_wkt: str,
    target_wkt: str,
    config_hash: str,
    source_path: str | Path,
    correction: dict | None = None,
    n_polygons_raw: int | None = None,
    n_dropped_null: int = 0,
    source_integrity: dict | None = None,
    null_geometry_basis: dict | None = None,
) -> Path:
    """Write reprojected GeoDataFrame to `cache_dir/reprojected_detections/{obs_id}.gpkg`
    plus a sidecar `{obs_id}.json` provenance record. Returns the GPKG path.
    """
    out_dir = Path(cache_dir) / CACHE_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    gpkg = out_dir / f"{obs_id}.gpkg"
    sidecar = out_dir / f"{obs_id}.json"

    gdf.to_file(gpkg, driver="GPKG", layer="detections")

    sidecar.write_text(
        json.dumps(
            {
                "obs_id": obs_id,
                "n_polygons": int(len(gdf)),
                "n_polygons_raw": int(n_polygons_raw) if n_polygons_raw is not None else int(len(gdf)),
                "n_dropped_null_geometry": int(n_dropped_null),
                # R23 provenance (2026-08-06). `source_integrity.status == "truncated"`
                # means the .shp was never fully copied; `null_geometry_basis
                # .is_rank_truncation` means the dropped rows are the whole low-score
                # tail, so this image's labels sit at `realised_score_floor` while an
                # unaffected image sits at ~0.10. Consumers making per-image LEVEL claims
                # must check these. See DECISIONS 2026-08-06o.
                "source_integrity": source_integrity,
                "null_geometry_basis": null_geometry_basis,
                "source_path": str(source_path),
                "source_mtime_iso": _dt.datetime.fromtimestamp(
                    Path(source_path).stat().st_mtime, tz=_dt.timezone.utc
                ).isoformat(),
                "source_crs_wkt": source_wkt,
                "target_crs_wkt": target_wkt,
                "config_hash": config_hash,
                "correction": correction or {"status": "trusted_prj"},
                "written_at_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return gpkg


def load_reprojected(obs_id: str, cache_dir: str | Path) -> gpd.GeoDataFrame:
    """Load a previously cached reprojected GeoDataFrame."""
    gpkg = Path(cache_dir) / CACHE_SUBDIR / f"{obs_id}.gpkg"
    return gpd.read_file(gpkg, layer="detections")


def stage1_one_image(
    obs_id: str,
    *,
    detections_root: str | Path,
    target_crs: str,
    cache_dir: str | Path,
    config_hash: str,
    manifest_row=None,
) -> tuple[gpd.GeoDataFrame, Path, dict]:
    """Run Stage 1 end-to-end for one ObsId: read, reproject (correcting buggy .prj
    if `manifest_row` is provided), cache. Returns the reprojected GeoDataFrame, the
    cache GPKG path, and the `correction` provenance dict.
    """
    shp = manifest_mod.find_shapefile(obs_id, detections_root)
    gdf, correction = read_detection_shapefile(
        obs_id, detections_root, manifest_row=manifest_row, cache_dir=cache_dir,
    )
    source_wkt = gdf.crs.to_wkt()
    n_raw = len(gdf)

    # R23 (DECISIONS 2026-08-06o): characterise the dropped population BEFORE dropping it,
    # and ask the file itself whether it is byte-complete. Both are recorded, and a
    # truncated source or a rank truncation warns loudly -- it does not raise, because the
    # three affected vClaire images are retained by decision pending the v3 re-detection.
    integrity = inspect_shapefile_integrity(shp)
    basis = describe_null_geometry_drop(gdf)
    if integrity.get("status") == "truncated":
        warnings.warn(
            f"{obs_id}: source shapefile is BYTE-TRUNCATED -- {shp.name} declares "
            f"{integrity['declared_bytes']:,} bytes but only {integrity['actual_bytes']:,} "
            f"are present ({integrity['missing_bytes'] / 1e6:.1f} MB missing). Records whose "
            "polygon bytes are absent read as null geometry and are dropped, so this image's "
            "label basis is a high-confidence subset. See DECISIONS 2026-08-06o.",
            RuntimeWarning,
            stacklevel=2,
        )
    if basis and basis.get("is_rank_truncation"):
        warnings.warn(
            f"{obs_id}: null-geometry rows are a SCORE-RANK TRUNCATION, not sparse export "
            f"noise -- every one of the {basis['n_dropped']:,} dropped rows "
            f"({basis['dropped_fraction']:.1%}) scores at or below every kept row. This "
            f"image's realised confidence floor is {basis['realised_score_floor']:.6f}, "
            "against ~0.10 for an unaffected image; its per-image abundance LEVEL is "
            "biased low. See DECISIONS 2026-08-06o.",
            RuntimeWarning,
            stacklevel=2,
        )

    gdf, n_dropped = drop_null_geometries(gdf)
    gdf_t = reproject_to_target(gdf, target_crs)
    target_wkt = gdf_t.crs.to_wkt()
    gpkg = cache_reprojected(
        gdf_t,
        obs_id,
        cache_dir,
        source_wkt=source_wkt,
        target_wkt=target_wkt,
        config_hash=config_hash,
        source_path=shp,
        correction=correction,
        n_polygons_raw=n_raw,
        n_dropped_null=n_dropped,
        source_integrity=integrity,
        null_geometry_basis=basis,
    )
    return gdf_t, gpkg, correction
