"""Unit tests for the manifest <-> Murray Lab tile-name translator.

Covers the four sign quadrants, the zero-edge cases that actually appear in the
priority10 manifest, and the URL template substitution.
"""
from __future__ import annotations

import pytest

from src.ctx_tiles import build_tile_url, manifest_to_murray


@pytest.mark.parametrize(
    "manifest_name, expected_murray",
    [
        # Verified examples from DECISIONS.md 2026-05-20:
        ("W040_N20", "E-40_N20"),
        ("E152_S08", "E152_N-8"),
        # Other quadrants present in the manifest (ESP_065711_1545 is E000_S28):
        ("E016_N44", "E16_N44"),
        ("E012_N44", "E12_N44"),
        ("E000_N40", "E0_N40"),   # zero-longitude edge; padding fallback handled by retriever
        ("E000_S28", "E0_N-28"),  # zero-longitude + south
        ("W052_N36", "E-52_N36"),
        ("W024_N28", "E-24_N28"),
        # Reference URL example noted in DECISIONS.md (E160_N-20.zip exists upstream):
        ("E160_S20", "E160_N-20"),
    ],
)
def test_manifest_to_murray_all_quadrants(manifest_name, expected_murray):
    assert manifest_to_murray(manifest_name) == expected_murray


@pytest.mark.parametrize(
    "bad_name",
    [
        "",
        "X040_N20",         # unknown hemisphere letter
        "W040N20",          # missing underscore
        "W040_X20",         # bad lat hemisphere
        "W040_N",           # missing lat digits
        "w040_n20",         # lowercase — manifest is always uppercase
    ],
)
def test_manifest_to_murray_rejects_malformed(bad_name):
    with pytest.raises(ValueError, match="unrecognized manifest tile name"):
        manifest_to_murray(bad_name)


def test_build_tile_url_inserts_tile_name():
    template = (
        "https://murray-lab.caltech.edu/CTX/V01/tiles/"
        "MurrayLab_GlobalCTXMosaic_V01_{tile_name}.zip"
    )
    url = build_tile_url(template, "E-40_N20")
    assert url == (
        "https://murray-lab.caltech.edu/CTX/V01/tiles/"
        "MurrayLab_GlobalCTXMosaic_V01_E-40_N20.zip"
    )


def test_build_tile_url_rejects_template_without_token():
    with pytest.raises(ValueError, match="missing required '{tile_name}'"):
        build_tile_url("https://example.com/no/token/here.zip", "E-40_N20")
