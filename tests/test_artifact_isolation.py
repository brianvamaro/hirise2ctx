"""R77 (residual): the test suite must not be able to write a live repository artifact.

Three independent layers, because each catches what the others cannot:

1. **Runtime guard** (`tests/live_artifact_guard.py`, installed session-wide by an autouse
   fixture in `conftest.py`) — refuses the syscall for any write under `cache*/`,
   `dataset*/`, `models/` or `reports/`. Catches paths nobody predicted, including the
   `cache_v2_dev` junction and a copied YAML whose relative paths still resolve against
   `REPO_ROOT`.
2. **Static scan** — the guard only fires on code that actually runs, and the producer
   tests all `skip` when their caches are absent. The AST scan below fails even when the
   call is skipped.
3. **Staging discipline** (`conftest.read_only_cache`) — a hard link is a second name for
   a live inode and lives outside every guarded prefix, so no path-based guard can see a
   write through it. Mutable derived artifacts are therefore copied, and only large
   immutable source archives are linked.

The end-to-end regression at the bottom exercises `read_full_footprint_decimated`'s
cache-invalidation branch — the branch the 2026-08-05 511-pass checksum run never took,
which is why that run was not evidence of isolation. It runs against two entirely
temporary roots and never touches a repository cache.
"""
from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pyproj
import pytest
import rasterio
from rasterio.transform import Affine

from src.config import REPO_ROOT
from src.hirise_imagery import _sp1_literal, read_full_footprint_decimated

from . import live_artifact_guard
from .conftest import LINKABLE_ARCHIVE_SUFFIXES, LINKABLE_IMMUTABLE_SUBDIRS
from .live_artifact_guard import ARTIFACT_ROOT_NAMES, LiveArtifactWriteError

TESTS_DIR = Path(__file__).resolve().parent


# ===========================================================================
# 1. Runtime guard
# ===========================================================================

def test_guard_is_installed_for_the_whole_session():
    assert live_artifact_guard.is_installed(), (
        "the autouse `_no_live_artifact_writes` fixture in conftest.py did not install "
        "the guard — every producer test is unprotected"
    )


@pytest.mark.parametrize("root_name", ARTIFACT_ROOT_NAMES)
def test_guard_blocks_builtin_writes_into_every_artifact_root(root_name):
    target = REPO_ROOT / root_name / "__r77_guard_probe__.tmp"
    with pytest.raises(LiveArtifactWriteError):
        open(target, "w").close()
    assert not target.exists(), "the guard must refuse before the file is created"


def test_guard_blocks_the_write_apis_src_actually_uses(tmp_path):
    """`rasterio`, `pyarrow` and `pyogrio` write through C layers that never reach
    `builtins.open`, so each needs its own patch."""
    victim = REPO_ROOT / "cache" / "__r77_guard_probe__.tif"
    with pytest.raises(LiveArtifactWriteError):
        rasterio.open(victim, "w", driver="GTiff", height=1, width=1, count=1, dtype="uint8")
    # `r+` is the in-place update mode that genuinely writes through a hard link.
    with pytest.raises(LiveArtifactWriteError):
        rasterio.open(REPO_ROOT / "cache_v2" / "anything.tif", "r+")

    import pandas as pd

    with pytest.raises(LiveArtifactWriteError):
        pd.DataFrame({"a": [1]}).to_parquet(REPO_ROOT / "dataset_v2" / "__probe__.parquet")

    with pytest.raises(LiveArtifactWriteError):
        np.save(REPO_ROOT / "dataset" / "__probe__.npy", np.zeros(3))

    for victim_path in (REPO_ROOT / "models" / "x.pkl", REPO_ROOT / "reports" / "y.csv"):
        with pytest.raises(LiveArtifactWriteError):
            Path(victim_path).write_text("nope", encoding="utf-8")

    assert not (REPO_ROOT / "cache" / "__r77_guard_probe__.tif").exists()
    assert not (REPO_ROOT / "dataset_v2" / "__probe__.parquet").exists()


def test_guard_blocks_deletion_and_replacement_of_live_artifacts(tmp_path):
    donor = tmp_path / "donor.bin"
    donor.write_bytes(b"x")
    with pytest.raises(LiveArtifactWriteError):
        os.remove(REPO_ROOT / "dataset" / "labels")
    with pytest.raises(LiveArtifactWriteError):
        os.replace(donor, REPO_ROOT / "dataset" / "labels" / "anything.parquet")
    with pytest.raises(LiveArtifactWriteError):
        os.link(donor, REPO_ROOT / "cache" / "__probe_link__")


def test_guard_leaves_reads_and_temporary_writes_alone(tmp_path):
    (tmp_path / "fine.txt").write_text("ok", encoding="utf-8")
    assert (tmp_path / "fine.txt").read_text(encoding="utf-8") == "ok"
    # Reading a live artifact root must stay legal — that is the whole point of staging.
    assert (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
    if (REPO_ROOT / "cache").is_dir():
        list((REPO_ROOT / "cache").iterdir())


def test_cache_v2_dev_is_a_junction_to_the_live_cache_and_is_guarded():
    """`config_v2_dev.yaml` looks like an isolated development cache and is not one."""
    dev = REPO_ROOT / "cache_v2_dev"
    if not dev.exists():
        pytest.skip("cache_v2_dev is not present in this checkout")
    real = Path(os.path.realpath(dev))
    assert real == (REPO_ROOT / "cache_v2").resolve(), (
        f"cache_v2_dev resolves to {real}; the isolation notes assume it is a junction to "
        "the live cache_v2. Re-check config_v2_dev.yaml before trusting it."
    )
    with pytest.raises(LiveArtifactWriteError):
        (dev / "__probe__.txt").write_text("nope", encoding="utf-8")


def test_a_copied_config_with_relative_paths_is_not_isolation(tmp_path):
    """`Config.resolve` joins against REPO_ROOT, not against the YAML's own directory."""
    from src.config import load_config

    src_yaml = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
    copied = tmp_path / "config.yaml"
    copied.write_text(src_yaml, encoding="utf-8")
    cfg = load_config(copied)
    assert cfg.resolve("cache_dir") == (REPO_ROOT / "cache").resolve(), (
        "copying the YAML into a temp dir did not redirect cache_dir — use an explicit "
        "absolute temporary root instead"
    )


def test_absolute_roots_in_a_config_really_do_redirect(tmp_path):
    """The isolation recipe a scratch rebuild must use, pinned so it cannot silently rot.

    `Config.resolve` is `(self.root / raw).resolve()`, and `Path.__truediv__` with an
    absolute right-hand side discards the left — so *absolute* `cache_dir`/`output_dir`
    values redirect completely, while relative ones do not (previous test). This is the
    difference between a rebuild that writes to scratch and one that overwrites the live
    tree, so it is worth an assertion rather than a comment.
    """
    import yaml

    from src.config import load_config

    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text(encoding="utf-8"))
    scratch_cache = tmp_path / "scratch_cache"
    scratch_out = tmp_path / "scratch_dataset"
    raw["cache_dir"] = str(scratch_cache)
    raw["output_dir"] = str(scratch_out)
    copied = tmp_path / "config.yaml"
    copied.write_text(yaml.safe_dump(raw), encoding="utf-8")

    cfg = load_config(copied)
    assert cfg.cache_dir == scratch_cache.resolve()
    assert cfg.output_dir == scratch_out.resolve()
    for root in (cfg.cache_dir, cfg.output_dir):
        assert live_artifact_guard.offending_path(root) is None, (
            f"{root} still resolves inside a repository artifact root"
        )


# ===========================================================================
# 2. Staging discipline
# ===========================================================================

# Directories a Stage 1-4b producer writes into. None of these may ever be hard-linked.
PRODUCER_WRITE_SUBDIRS = frozenset({
    "reprojected_detections",   # Stage 1
    "ctx_windows",              # Stage 2 (window, coverage mask, sidecar)
    "hirise_decimated",         # rebuilt by read_full_footprint_decimated on stale CRS
    "coregistration",           # Stage 3
})


def test_linkable_allowlist_excludes_every_producer_write_target():
    overlap = LINKABLE_IMMUTABLE_SUBDIRS & PRODUCER_WRITE_SUBDIRS
    assert not overlap, (
        f"{sorted(overlap)} are written by a producer and must be copied, not linked. "
        "This is the residual R77 hole the 2026-08-06 audit found."
    )


def _fake_cache(root: Path) -> Path:
    """A miniature stand-in for `cache/` with one mutable and one immutable subdir."""
    (root / "hirise_decimated").mkdir(parents=True)
    (root / "hirise_decimated" / "ESP_000001_2000_5mpp_full.tif").write_bytes(b"derived")
    (root / "hirise_decimated" / "ESP_999999_2000_5mpp_full.tif").write_bytes(b"other obs")
    (root / "ctx_tiles").mkdir(parents=True)
    (root / "ctx_tiles" / "E0_N40.zip").write_bytes(b"archive")
    (root / "ctx_tiles" / "E0_N40.json").write_bytes(b'{"inner_transform": []}')
    return root


def test_read_only_cache_copies_mutable_subdirs_and_links_only_the_allowlist(
    tmp_path, read_only_cache
):
    source = _fake_cache(tmp_path / "src_cache")
    staged = read_only_cache(source, ["hirise_decimated", "ctx_tiles"])

    derived = staged / "hirise_decimated" / "ESP_000001_2000_5mpp_full.tif"
    archive = staged / "ctx_tiles" / "E0_N40.zip"
    assert derived.exists() and archive.exists()
    assert not derived.samefile(source / "hirise_decimated" / "ESP_000001_2000_5mpp_full.tif"), (
        "a mutable derived artifact was hard-linked; an in-place writer would reach the "
        "live inode"
    )
    assert archive.samefile(source / "ctx_tiles" / "E0_N40.zip"), (
        "the immutable archive should still be linked — copying 44 GB of tile zips per "
        "test is not viable"
    )


def test_read_only_cache_only_filter_copies_just_the_requested_obs(tmp_path, read_only_cache):
    source = _fake_cache(tmp_path / "src_cache")
    staged = read_only_cache(source, ["hirise_decimated", "ctx_tiles"], only="ESP_000001_2000")
    names = {p.name for p in (staged / "hirise_decimated").iterdir()}
    assert names == {"ESP_000001_2000_5mpp_full.tif"}
    # Linked archives ignore the filter: their names carry a Murray tile, not an ObsId.
    assert (staged / "ctx_tiles" / "E0_N40.zip").exists()
    # ...but a sidecar beside an archive is derived, so it is filtered like everything else
    # and callers must ask for it by name.
    assert not (staged / "ctx_tiles" / "E0_N40.json").exists()
    staged2 = read_only_cache(source, ["ctx_tiles"], only=["ESP_000001_2000", "E0_N40"])
    assert (staged2 / "ctx_tiles" / "E0_N40.json").exists()


def test_sidecars_beside_an_archive_are_copied_not_linked(tmp_path, read_only_cache):
    """A `{tile}.json` or GDAL `.aux.xml` is derived and GDAL rewrites PAM sidecars in
    place — the immutability argument covers the archive, not its neighbours."""
    source = _fake_cache(tmp_path / "src_cache")
    staged = read_only_cache(source, ["ctx_tiles"])
    assert (staged / "ctx_tiles" / "E0_N40.zip").samefile(source / "ctx_tiles" / "E0_N40.zip")
    assert not (staged / "ctx_tiles" / "E0_N40.json").samefile(
        source / "ctx_tiles" / "E0_N40.json"
    )
    assert ".json" not in LINKABLE_ARCHIVE_SUFFIXES


def test_hard_link_write_through_is_real_for_in_place_writers(tmp_path):
    """Why a mutable derived artifact may not be linked, independent of any one library.

    Measured 2026-08-06 (rasterio 1.5.0 / GDAL 3.11.4, NTFS): `rasterio.open(p, "w")`
    happens to delete-then-create, which breaks the link and spares the source — so the
    audit's *specific* `read_full_footprint_decimated` truncation mechanism does not fire
    on this version. Every in-place writer below does write through, and that is a
    property of hard links rather than of a library version, so the staging rule cannot
    rest on rasterio's current create path.

    Wholly synthetic: both names live under `tmp_path`.
    """
    source = tmp_path / "live.bin"
    source.write_bytes(b"original")
    link = tmp_path / "staged.bin"
    os.link(source, link)

    link.write_text("clobbered", encoding="utf-8")
    assert source.read_bytes() == b"clobbered", (
        "expected a hard link to write through; if this ever stops being true the "
        "staging rule is still correct, but this rationale needs rewriting"
    )
    assert source.samefile(link)


# ===========================================================================
# 3. End-to-end: the cache-invalidation branch, on two temporary roots
# ===========================================================================

# Same shape as the WKT `src/detections.py` writes into the Stage 1 sidecar. SP1=20 is
# the PDS-correct projection latitude; the JP2 (and any cache built before the JP2-side
# SP1 fix) carries the buggy SP1=0.
_CORRECTED_WKT = pyproj.CRS.from_user_input(
    'PROJCS["Equirectangular_MARS",'
    'GEOGCS["GCS_MARS",DATUM["D_MARS",'
    'SPHEROID["MARS_localRadius",3393833.2607584,0.0]],'
    'PRIMEM["Reference_Meridian",0.0],'
    'UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Equidistant_Cylindrical"],'
    'PARAMETER["False_Easting",0.0],PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",180.0],'
    'PARAMETER["Standard_Parallel_1",20.0],'
    'UNIT["Meter",1.0]]'
).to_wkt()
_BUGGY_WKT = _CORRECTED_WKT.replace(
    '"Latitude of 1st standard parallel",20', '"Latitude of 1st standard parallel",0'
)

_OBS = "ESP_000001_2000"
_STALE_FILL = 7


def _synthetic_source_cache(root: Path) -> Path:
    """A cache tree holding a >1 MB stand-in JP2, a Stage 1 sidecar, and a STALE-CRS
    decimated cache. Nothing here is a repository path."""
    assert _BUGGY_WKT != _CORRECTED_WKT, "SP1 replace did not fire — fixture is degenerate"

    (root / "hirise_jp2").mkdir(parents=True)
    (root / "hirise_decimated").mkdir(parents=True)
    (root / "reprojected_detections").mkdir(parents=True)

    # `_open_source` accepts any GDAL-readable file at this path once it exceeds 1 MB;
    # 900x900 uint16 uncompressed is ~1.6 MB. 1 m/px so target_mpp=5 decimates by 5.
    jp2 = root / "hirise_jp2" / f"{_OBS}_RED.JP2"
    with rasterio.open(
        jp2, "w", driver="GTiff", height=900, width=900, count=1, dtype="uint16",
        crs=rasterio.crs.CRS.from_wkt(_BUGGY_WKT), transform=Affine(1.0, 0, 5_000.0, 0, -1.0, 9_000.0),
    ) as ds:
        ds.write(np.arange(900 * 900, dtype=np.uint16).reshape(900, 900) % 4096, 1)
    assert jp2.stat().st_size > 1_000_000, "stand-in JP2 must clear the 1 MB local-source gate"

    (root / "reprojected_detections" / f"{_OBS}.json").write_text(
        json.dumps({"obs_id": _OBS, "source_crs_wkt": _CORRECTED_WKT}), encoding="utf-8"
    )

    stale = root / "hirise_decimated" / f"{_OBS}_5mpp_full.tif"
    with rasterio.open(
        stale, "w", driver="GTiff", height=180, width=180, count=1, dtype="uint16",
        crs=rasterio.crs.CRS.from_wkt(_BUGGY_WKT), transform=Affine(5.0, 0, 5_000.0, 0, -5.0, 9_000.0),
    ) as ds:
        ds.write(np.full((180, 180), _STALE_FILL, dtype=np.uint16), 1)
    return root


def test_stale_crs_cache_rebuild_never_reaches_the_source_tree(tmp_path, read_only_cache):
    """The invalidation branch the 511-pass checksum run never entered.

    A cached decimated GeoTIFF whose CRS predates the JP2-side SP1 fix must be rebuilt,
    and with copy-staging that rebuild cannot reach the source tree. Two temporary roots;
    no repository cache is opened.
    """
    source = _synthetic_source_cache(tmp_path / "source_cache")
    stale = source / "hirise_decimated" / f"{_OBS}_5mpp_full.tif"
    before = (stale.read_bytes(), stale.stat().st_mtime_ns)

    staged = read_only_cache(
        source, ["hirise_jp2", "hirise_decimated", "reprojected_detections"], only=_OBS
    )
    staged_tif = staged / "hirise_decimated" / f"{_OBS}_5mpp_full.tif"
    assert not staged_tif.samefile(stale), "staged derived TIFF must not be the source inode"

    arr, _transform, crs = read_full_footprint_decimated(
        _OBS, "http://unused.invalid/never-fetched.JP2", staged, target_mpp=5.0
    )

    # The rebuild fired: the returned CRS is the corrected one, and the pixels come from
    # the stand-in source rather than the stale fill.
    assert _sp1_literal(crs) == 20.0, "stale SP1=0 cache was accepted instead of rebuilt"
    assert not np.all(arr == _STALE_FILL), "the stale cache contents were returned unchanged"
    with rasterio.open(staged_tif) as ds:
        assert _sp1_literal(ds.crs) == 20.0, "the staged cache was not rewritten"

    assert (stale.read_bytes(), stale.stat().st_mtime_ns) == before, (
        "R77: the cache rebuild reached the source tree. If the staged copy had been a "
        "hard link this is where the live hirise_decimated GeoTIFF would change."
    )


def test_stale_crs_rebuild_converges_on_the_second_call(tmp_path, read_only_cache):
    """A corrected cache must be accepted, or every producer call rewrites it forever."""
    source = _synthetic_source_cache(tmp_path / "source_cache")
    staged = read_only_cache(
        source, ["hirise_jp2", "hirise_decimated", "reprojected_detections"], only=_OBS
    )
    staged_tif = staged / "hirise_decimated" / f"{_OBS}_5mpp_full.tif"

    read_full_footprint_decimated(_OBS, "http://unused.invalid/x.JP2", staged, target_mpp=5.0)
    fingerprint = (staged_tif.read_bytes(), staged_tif.stat().st_mtime_ns)
    read_full_footprint_decimated(_OBS, "http://unused.invalid/x.JP2", staged, target_mpp=5.0)

    assert (staged_tif.read_bytes(), staged_tif.stat().st_mtime_ns) == fingerprint, (
        "the second call rebuilt an already-corrected cache — the SP1 round trip through "
        "GeoTIFF does not survive, so every Stage 2/3 run would rewrite hirise_decimated"
    )


# ===========================================================================
# 4. Static scan: producers must never receive a live artifact root to write
# ===========================================================================

# Producer entry point -> the argument(s) it WRITES through. Read-only roots (e.g. Stage
# 4/4b's `cache_dir`) are deliberately absent: staging those is what `read_only_cache` is
# for. `positions` maps 0-based positional index -> argument name for the non-kw-only
# signatures.
PRODUCER_WRITE_ARGS: dict[str, dict] = {
    "stage1_one_image":              {"kwargs": {"cache_dir"},              "positions": {}},
    "stage2_one_image":              {"kwargs": {"cache_dir"},              "positions": {}},
    "stage3_one_image":              {"kwargs": {"cache_dir"},              "positions": {}},
    "stage4_one_image":              {"kwargs": {"output_dir"},             "positions": {}},
    "stage4b_one_image":             {"kwargs": {"output_dir"},             "positions": {}},
    "ensure_tile_cached":            {"kwargs": {"cache_dir"},              "positions": {}},
    "build_hirise_coverage_mask":    {"kwargs": {"cache_dir", "out_path"},  "positions": {}},
    "extract_ctx_window":            {"kwargs": {"out_path"},               "positions": {3: "out_path"}},
    "ensure_jp2_local":              {"kwargs": {"cache_dir"},              "positions": {2: "cache_dir"}},
    "read_full_footprint_decimated": {"kwargs": {"cache_dir"},              "positions": {2: "cache_dir"}},
    "read_native_window":            {"kwargs": {"cache_dir"},              "positions": {3: "cache_dir"}},
    "package_split":                 {"kwargs": {"output_dir"},             "positions": {}},
    "write_split_metadata":          {"kwargs": {"output_dir"},             "positions": {1: "output_dir"}},
}

_LIVE_ROOT_EXPR = re.compile(
    r"\bcfg\.(?:cache_dir|output_dir)\b"
    r"|\bcfg\.resolve\(\s*['\"](?:cache_dir|output_dir)['\"]\s*\)"
    r"|\b(?:repo_root|REPO_ROOT)\s*/\s*['\"](?:" + "|".join(ARTIFACT_ROOT_NAMES) + r")['\"]"
)


def _is_live_root_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return live_artifact_guard.offending_path(node.value) is not None
    return bool(_LIVE_ROOT_EXPR.search(ast.unparse(node)))


def _offending_producer_calls() -> list[str]:
    problems: list[str] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            spec = PRODUCER_WRITE_ARGS.get(name or "")
            if spec is None:
                continue
            checked: list[tuple[str, ast.AST]] = [
                (kw.arg, kw.value) for kw in node.keywords
                if kw.arg in spec["kwargs"]
            ]
            for index, arg_name in spec["positions"].items():
                if len(node.args) > index:
                    checked.append((arg_name, node.args[index]))
            for arg_name, value in checked:
                if _is_live_root_expression(value):
                    problems.append(
                        f"{path.name}:{node.lineno} {name}({arg_name}={ast.unparse(value)})"
                    )
    return problems


def test_no_test_hands_a_producer_a_live_artifact_root_to_write():
    """Static, so it fails even when the producer test would `skip` for a missing cache.

    R77 bit twice because the call site *looked* harmless: a producer takes one root and
    uses it for both input and output, with no dry-run mode.
    """
    problems = _offending_producer_calls()
    assert not problems, (
        "R77: these tests pass a live gitignored artifact root as a producer's WRITE "
        "argument. git cannot restore those trees. Use tmp_path (or `read_only_cache` "
        "for the read side):\n  " + "\n  ".join(problems)
    )


def test_producer_inventory_has_not_rotted():
    """A renamed producer would silently empty the scan above."""
    import importlib

    modules = [
        importlib.import_module(f"src.{m}")
        for m in ("detections", "ctx_retrieve", "coregister", "labeling", "features",
                  "hirise_imagery", "dataset")
    ]
    for name in PRODUCER_WRITE_ARGS:
        assert any(hasattr(mod, name) for mod in modules), (
            f"{name} is in PRODUCER_WRITE_ARGS but no longer exists in src/ — the static "
            "scan is checking a function nobody calls"
        )
