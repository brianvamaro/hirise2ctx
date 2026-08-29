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


# ---------------------------------------------------------------------------
# expect_digests — the head pin, recorded where the rendered tiles actually live
# ---------------------------------------------------------------------------
#
# The Sherlock job's preflight compares the head it resolved against this block. It used to
# read a product directory instead, and on the cluster `reports/map_region` is a symlink to an
# empty $SCRATCH dir — so on the first real submission (job 41110268) it found no basis, printed
# "skipping the match check", and let the job run ungated. The plan travels with the repo.

def _sidecar(d, tile, *, head="H", calib="C"):
    import json
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tile}.json").write_text(
        json.dumps({"murray_tile": tile, "head_digest": head, "calibration_digest": calib}),
        encoding="utf-8")


def test_expected_digests_reads_the_pin_from_the_rendered_tiles(tmp_path):
    from plan_map_extent import expected_digests

    _sidecar(tmp_path / "prod", "E-12_N32")
    _sidecar(tmp_path / "prod", "E-8_N32")
    exp = expected_digests([tmp_path / "prod"], ["E-12_N32", "E-8_N32", "E-24_N20"])
    assert exp["head_digest"] == "H" and exp["calibration_digest"] == "C"
    assert exp["measured_from"] == ["E-12_N32", "E-8_N32"]


def test_expected_digests_does_not_double_count_an_adopted_tile(tmp_path):
    """An adopted tile is in BOTH the source and the destination product."""
    from plan_map_extent import expected_digests

    for sub in ("map_region", "map_extended"):
        _sidecar(tmp_path / sub, "E-12_N32")
    exp = expected_digests([tmp_path / "map_region", tmp_path / "map_extended"], ["E-12_N32"])
    assert exp["measured_from"] == ["E-12_N32"]


def test_expected_digests_is_empty_for_a_product_with_no_predecessor(tmp_path):
    """A genuinely fresh product has nothing to match, and must say so rather than invent one."""
    from plan_map_extent import expected_digests

    assert expected_digests([tmp_path], ["E-24_N20"]) == {}


def test_expected_digests_refuses_a_mixed_product(tmp_path):
    """Two heads already in the product: no single expectation would be honest."""
    from plan_map_extent import expected_digests

    _sidecar(tmp_path / "prod", "E-12_N32", head="H1")
    _sidecar(tmp_path / "prod", "E-8_N32", head="H2")
    with pytest.raises(SystemExit, match="MIXED"):
        expected_digests([tmp_path / "prod"], ["E-12_N32", "E-8_N32"])


def test_expected_digests_ignores_the_plan_file_itself(tmp_path):
    """`plan.json` lives in the product dir; it is not a tile (MANIFEST_NAMES)."""
    from plan_map_extent import expected_digests

    _sidecar(tmp_path / "prod", "E-12_N32")
    (tmp_path / "prod" / "plan.json").write_text('{"tiles": []}', encoding="utf-8")
    assert expected_digests([tmp_path / "prod"], ["E-12_N32"])["head_digest"] == "H"


def test_the_shipped_plan_pins_the_rebuild_head():
    """The real plan must carry the g2 digests, not the legacy head's."""
    import json

    plan_path = REPO / "reports" / "map_extended" / "plan.json"
    if not plan_path.exists():
        pytest.skip("no extension plan in this checkout")
    exp = json.loads(plan_path.read_text(encoding="utf-8")).get("expect_digests")
    assert exp, "the plan carries no head pin -- the Sherlock preflight would refuse to start"
    assert exp["head_digest"] == (
        "29e833be74e5cc151d1382caa9b5d7d7e2abf8d62597f648c6de5da71a34db2e")
    assert exp["calibration_digest"] == (
        "290a86614f190ced416606689e33533ec55e32a9d349484c51626313c897a61d")
    # `measured_from` names the already-rendered tiles the digests were read off. This used
    # to assert `== 8`, the round-1 count, and went stale the moment round 2 was planned over
    # a 35-tile product (60413f9) -- a snapshot of one planning round masquerading as an
    # invariant. What must hold for any round is that the pin came from real rendered tiles.
    measured = exp["measured_from"]
    assert measured, "the digests were read off no tile -- the pin is unsourced"
    prod = plan_path.parent
    assert all((prod / f"{t}.json").exists() for t in measured), (
        f"measured_from names tiles with no sidecar in {prod.name}: "
        f"{[t for t in measured if not (prod / f'{t}.json').exists()]}")
