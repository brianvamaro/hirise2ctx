"""Pick the right ``models/_sweep*`` run for a given dataset version.

The sweep output dirs are named only by UTC timestamp (``models/_sweep/{ts}``),
so once both a v1 (``dataset``) and a v2 (``dataset_v2``) sweep exist, "the latest
dir" is ambiguous. ``scripts/sweep*.py`` now drop a ``sweep_meta.json`` recording
``dataset_dir`` + ``scheme``; this helper selects the newest run whose meta matches
the requested ``dataset_dir``, with a fallback for legacy (pre-meta) dirs that are
all v1.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_ROOT = REPO_ROOT / "models"

KIND_SUBDIR = {
    "regression": "_sweep",
    "binary": "_sweep_binary",
    "within_image": "_sweep_within_image",
}


def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def pick_sweep(
    kind: str,
    dataset_dir: str = "dataset",
    *,
    models_root: Path | str | None = None,
) -> Path:
    """Newest sweep dir of ``kind`` whose ``sweep_meta.json`` matches ``dataset_dir``.

    ``kind`` is one of ``regression`` / ``binary`` / ``within_image``. Falls back to
    the newest *untagged* (legacy) dir when ``dataset_dir == "dataset"`` and nothing
    is tagged yet, preserving the pre-``sweep_meta.json`` behaviour. Raises
    ``FileNotFoundError`` when no matching run exists.
    """
    if kind not in KIND_SUBDIR:
        raise ValueError(f"unknown sweep kind {kind!r}; expected {list(KIND_SUBDIR)}")
    root = Path(models_root) if models_root is not None else DEFAULT_MODELS_ROOT
    base = root / KIND_SUBDIR[kind]
    candidates = sorted((p for p in base.glob("*/") if p.is_dir()), key=_mtime, reverse=True)

    for c in candidates:
        meta_path = c / "sweep_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("dataset_dir") == dataset_dir:
                return c

    # Legacy fallback: untagged dirs predate sweep_meta.json and are all v1.
    if dataset_dir == "dataset":
        for c in candidates:
            if not (c / "sweep_meta.json").exists():
                return c

    raise FileNotFoundError(
        f"no {kind} sweep found for dataset_dir={dataset_dir!r} under {base} "
        f"(checked {len(candidates)} dirs)"
    )
