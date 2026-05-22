"""Load and validate the pipeline YAML config; produce a stable config hash for provenance."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TOP_LEVEL = {
    "manifest",
    "detections_root",
    "cache_dir",
    "output_dir",
    "target_crs",
    "ctx_mosaic",
    "ctx_read",
    "ctx_retrieve",
    "hirise_decimation_mpp",
    "coregistration",
    "labeling",
    "sanity",
}

REQUIRED_CTX_MOSAIC = {"catalog_url", "url_template", "probe_tile"}
REQUIRED_CTX_RETRIEVE = {
    "mode",
    "buffer_m",
    "nominal_hirise_width_m",
    "nominal_hirise_length_m",
}
SUPPORTED_CTX_RETRIEVE_MODES = {"download_then_window"}
REQUIRED_LABELING = {
    "grid_anchor",
    "tile_sizes_px",
    "label_type",
    "binary_area_threshold",
    "binary_count_threshold",
    "categorical_bins",
    "detection_filters",
    "context_patch_px",
    "features",
}
REQUIRED_SANITY = {"centroid_max_km"}


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path
    root: Path = field(default_factory=lambda: REPO_ROOT)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def resolve(self, key: str) -> Path:
        """Resolve a path-valued config key against the repo root."""
        return (self.root / self.raw[key]).resolve()

    @property
    def cache_dir(self) -> Path:
        p = self.resolve("cache_dir")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir(self) -> Path:
        p = self.resolve("output_dir")
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def manifest_path(self) -> Path:
        return self.resolve("manifest")

    @property
    def detections_root(self) -> Path:
        return Path(self.raw["detections_root"]).resolve()

    @property
    def hash(self) -> str:
        return config_hash(self.raw)


def _validate(cfg: dict[str, Any], path: Path) -> None:
    missing = REQUIRED_TOP_LEVEL - cfg.keys()
    if missing:
        raise ValueError(f"{path}: missing required top-level keys: {sorted(missing)}")

    missing = REQUIRED_CTX_MOSAIC - cfg["ctx_mosaic"].keys()
    if missing:
        raise ValueError(f"{path}: ctx_mosaic missing keys: {sorted(missing)}")

    missing = REQUIRED_CTX_RETRIEVE - cfg["ctx_retrieve"].keys()
    if missing:
        raise ValueError(f"{path}: ctx_retrieve missing keys: {sorted(missing)}")
    mode = cfg["ctx_retrieve"]["mode"]
    if mode not in SUPPORTED_CTX_RETRIEVE_MODES:
        raise ValueError(
            f"{path}: ctx_retrieve.mode={mode!r} not in {sorted(SUPPORTED_CTX_RETRIEVE_MODES)}"
        )

    missing = REQUIRED_LABELING - cfg["labeling"].keys()
    if missing:
        raise ValueError(f"{path}: labeling missing keys: {sorted(missing)}")

    missing = REQUIRED_SANITY - cfg["sanity"].keys()
    if missing:
        raise ValueError(f"{path}: sanity missing keys: {sorted(missing)}")

    sizes = cfg["labeling"]["tile_sizes_px"]
    if not (isinstance(sizes, list) and len(sizes) >= 1 and all(isinstance(s, int) and s > 0 for s in sizes)):
        raise ValueError(f"{path}: labeling.tile_sizes_px must be a non-empty list of positive ints")
    # x2 ladder check: each next size = previous * 2
    for a, b in zip(sizes, sizes[1:]):
        if b != a * 2:
            raise ValueError(
                f"{path}: labeling.tile_sizes_px must be a x2 ladder (got {sizes}); "
                "this is required for nested-grid sum-up consistency (CLAUDE.md Stage 4)."
            )


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load and validate the pipeline config from YAML."""
    p = Path(path)
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: config root must be a mapping, got {type(raw).__name__}")
    _validate(raw, p)
    return Config(raw=raw, path=p)


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable SHA256 over the config's sorted-keys JSON serialization.

    Used as provenance in per-image caches and dataset rows so downstream stages can
    detect when their inputs were generated under a different config.
    """
    canonical = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
