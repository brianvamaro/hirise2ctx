"""URL padding fallback used by `ctx_retrieve.ensure_tile_cached`.

The Murray Lab CTX mosaic actually serves a zero-padded canonical filename for every
tile (verified 2026-05-22). `manifest_to_murray` produces a *bare* signed-int form that
404s for many tiles; `_padded_manifest_form` rewrites it into the canonical form.
"""
from __future__ import annotations

import pytest

from src.ctx_retrieve import _padded_manifest_form


@pytest.mark.parametrize(
    "murray_bare, expected_padded",
    [
        # Verified successful Murray Lab URLs (2026-05-22 probe):
        ("E0_N40", "E000_N40"),       # ESP_069669_2220 etc.
        ("E12_N44", "E012_N44"),      # ESP_054857_2270
        ("E16_N44", "E016_N44"),      # ESP_055714_2270
        ("E-40_N20", "E-040_N20"),    # ESP_047976_2020  (the case that broke the May sweep)
        ("E-52_N36", "E-052_N36"),    # ESP_056165_2200
        ("E-24_N28", "E-024_N28"),    # ESP_075577_2105
        ("E152_N-8", "E152_N-08"),    # ESP_039820_1750
        ("E0_N-28", "E000_N-28"),     # ESP_065711_1545
        # Tile already in canonical form -> no padding needed (return None to signal "don't retry").
        ("E160_N-20", None),          # 3-digit lon, 2-digit lat both already padded
        ("E000_N40", None),
        ("E-040_N20", None),
    ],
)
def test_padded_manifest_form(murray_bare: str, expected_padded: str | None):
    assert _padded_manifest_form(murray_bare) == expected_padded


def test_padded_manifest_form_returns_none_for_non_murray_input():
    assert _padded_manifest_form("not_a_tile") is None
    assert _padded_manifest_form("W040_N20") is None  # W/S prefixes are not Murray form
