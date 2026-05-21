"""Read the manifest CSV and resolve per-ObsId detection shapefiles by glob."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

REQUIRED_COLUMNS = (
    "ObsId",
    "ProductId",
    "BoulderLabel",
    "CenterLat",
    "CenterLon_360",
    "CenterLon_180",
    "CTX_TileName",
    "BrowseURL",
    "JP2_URL",
    "LabelURL",
)

SHAPEFILE_GLOB = "*-mask-nms.shp"


@dataclass(frozen=True)
class ManifestRow:
    obs_id: str
    product_id: str
    boulder_label: str
    center_lat: float
    center_lon_360: float
    center_lon_180: float
    ctx_tile_name: str
    jp2_url: str
    label_url: str
    raw: dict

    @classmethod
    def from_series(cls, row: pd.Series) -> "ManifestRow":
        return cls(
            obs_id=str(row["ObsId"]),
            product_id=str(row["ProductId"]),
            boulder_label=str(row["BoulderLabel"]),
            center_lat=float(row["CenterLat"]),
            center_lon_360=float(row["CenterLon_360"]),
            center_lon_180=float(row["CenterLon_180"]),
            ctx_tile_name=str(row["CTX_TileName"]),
            jp2_url=str(row["JP2_URL"]),
            label_url=str(row["LabelURL"]),
            raw=row.to_dict(),
        )


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load the manifest CSV and validate required columns and ObsId uniqueness."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: manifest missing required columns: {missing}")
    dups = df["ObsId"][df["ObsId"].duplicated()].tolist()
    if dups:
        raise ValueError(f"{path}: duplicate ObsId rows: {dups}")
    return df


def iter_rows(df: pd.DataFrame) -> Iterator[ManifestRow]:
    """Iterate manifest rows as `ManifestRow` records."""
    for _, row in df.iterrows():
        yield ManifestRow.from_series(row)


def find_shapefile(obs_id: str, detections_root: str | Path) -> Path:
    """Resolve the single BoulderNet `*-mask-nms.shp` for an ObsId.

    Fails loudly if 0 or >1 shapefiles match, so a future BoulderNet param-suffix change
    (the `ct-010-ss-512-...` substring may differ) doesn't silently pick the wrong file.
    """
    folder = Path(detections_root) / obs_id
    if not folder.is_dir():
        raise FileNotFoundError(f"detections folder missing for {obs_id}: {folder}")
    matches = sorted(folder.glob(SHAPEFILE_GLOB))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"no shapefile matching {SHAPEFILE_GLOB!r} under {folder}; "
            f"folder contents: {[p.name for p in folder.iterdir()]}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"multiple shapefiles match {SHAPEFILE_GLOB!r} under {folder}: "
            f"{[p.name for p in matches]} — refusing to guess"
        )
    return matches[0]


def resolve_all_shapefiles(df: pd.DataFrame, detections_root: str | Path) -> dict[str, Path]:
    """Resolve every manifest ObsId to its shapefile path. Fails loudly on the first miss."""
    return {row["ObsId"]: find_shapefile(row["ObsId"], detections_root) for _, row in df.iterrows()}
