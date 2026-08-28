"""Unit tests for the map-extent planner (`scripts/plan_map_extent.py`).

The planner's output is what a submission decision rests on, so the two things that can
quietly make it wrong get exact assertions:

* **The tile naming is corner-relative, not centre-relative.** Murray names a tile by its
  lower-left corner, so a box's last row starts at `lat1 - 4`. An off-by-one-row planner
  produces a tile list that is entirely valid-looking and covers the wrong ground.
* **The timing comes from measured runs, not a constant.** A manifest entry with no
  `elapsed_s`, or a resumed task that skipped every tile in seconds, must not be averaged in
  as if it were a rate.

Pure functions only -- no network, no artifact reads. `--verify-urls` is exercised against
the real host by running the script, not here.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from plan_map_extent import (WINDOWS_PER_TILE, snap_box, tiles_in_box,  # noqa: E402
                             window_seconds)


def test_snap_box_expands_outward_to_whole_tiles():
    assert snap_box(-25, -5, 20, 35) == (-28, -4, 20, 36)


def test_snap_box_is_identity_on_an_aligned_box():
    assert snap_box(-24, -4, 20, 36) == (-24, -4, 20, 36)


def test_snap_box_accepts_reversed_bounds():
    assert snap_box(-5, -25, 35, 20) == snap_box(-25, -5, 20, 35)


def test_tiles_in_box_names_tiles_by_their_LOWER_LEFT_corner():
    """`E-12_N32` spans lon [-12,-8] lat [32,36]; the box's top row is `lat1 - 4`."""
    tiles = tiles_in_box(-12, -4, 32, 40)
    assert tiles == ["E-12_N36", "E-8_N36", "E-12_N32", "E-8_N32"]
    assert "E-12_N40" not in tiles, "included a row starting at lat1 -- one row too far north"
    assert "E-4_N32" not in tiles, "included a column starting at lon1 -- one column too far east"


def test_tiles_in_box_reproduces_the_shipped_circum_chryse_block():
    """The 24 box tiles of the shipped map (the 2 NE tabs are additions, not box members)."""
    from map_region import BLOCK_TILES

    box = set(tiles_in_box(-12, 12, 32, 48))
    assert len(box) == 24
    assert box == set(BLOCK_TILES) - {"E12_N44", "E16_N44"}


def test_tiles_in_box_counts_the_requested_region():
    assert len(tiles_in_box(*snap_box(-25, -5, 20, 35))) == 24
    assert len(tiles_in_box(-24, -4, 20, 36)) == 20


def _manifest(tmp_path, runs):
    p = tmp_path / "region_manifest.json"
    p.write_text(json.dumps({"runs": runs}), encoding="utf-8")
    return p


def test_window_seconds_divides_by_tiles_times_windows(tmp_path):
    per_window = 20.0
    p = _manifest(tmp_path, [{"tiles": ["a", "b", "c"],
                              "elapsed_s": 3 * WINDOWS_PER_TILE * per_window}])
    assert window_seconds(p)["median"] == pytest.approx(per_window)


def test_window_seconds_reproduces_the_shipped_rate():
    """The real manifest: ~17-22 s/window on the 2080 Ti, which is what the estimate quotes."""
    rates = window_seconds(REPO / "reports" / "map_region" / "region_manifest.json")
    if rates["median"] is None:
        pytest.skip("no shipped manifest in this checkout")
    assert 15.0 < rates["median"] < 25.0
    assert rates["min"] <= rates["median"] <= rates["max"]
    assert rates["n_runs"] >= 4


def test_window_seconds_drops_a_resumed_task_that_did_no_work(tmp_path):
    """A 5.4 s run over 1 'tile' is the skip path; averaging it in halves the estimate."""
    p = _manifest(tmp_path, [{"tiles": ["a"], "elapsed_s": WINDOWS_PER_TILE * 18.0},
                             {"tiles": ["b"], "elapsed_s": 5.4}])
    rates = window_seconds(p)
    assert rates["n_runs"] == 1
    assert rates["median"] == pytest.approx(18.0)


def test_window_seconds_drops_entries_with_no_timing(tmp_path):
    p = _manifest(tmp_path, [{"tiles": ["a"], "elapsed_s": WINDOWS_PER_TILE * 18.0},
                             {"tiles": ["b"]},
                             {"tiles": [], "elapsed_s": 999.0}])
    assert window_seconds(p)["n_runs"] == 1


def test_window_seconds_is_silent_when_there_is_no_manifest(tmp_path):
    rates = window_seconds(tmp_path / "absent.json")
    assert rates["median"] is None and rates["n_runs"] == 0


def test_the_padded_url_form_is_what_the_region_actually_needs():
    """Every western tile 404s on the bare id; the planner must resolve the padded form.

    Not a network test -- it pins the name transform the URL check depends on. The first
    run of the planner skipped it and declared all 22 published tiles missing.
    """
    from src.ctx_retrieve import _padded_manifest_form

    assert _padded_manifest_form("E-24_N28") == "E-024_N28"
    assert _padded_manifest_form("E-8_N20") == "E-008_N20"
    for t in tiles_in_box(*snap_box(-25, -5, 20, 35)):
        assert _padded_manifest_form(t) is not None, f"{t} has no padded form to fall back to"
