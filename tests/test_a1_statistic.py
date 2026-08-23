"""R07 — the ONE A1 preprocessing statistic, and the arm versioning that pins it.

R07 measured two independent train/deploy mismatches and one provenance gap:

  * resolution — training used a native 5 m statistic, deployment a 160 m area-averaged one,
    and the deploy gain was applied to native DN anyway. Over all 39 Stage-2 windows the gain
    error was 1.35x median / 2.15x max, so the frozen ViT received inputs with an IQR of 37.3
    median (max 59.6) against the 27.7 it was trained on, clipping ~10x more pixels;
  * unit — training took ONE statistic per observation window, deployment one per SeamMap
    source frame, and only 10 of 38 windows lie inside a single frame;
  * versioning — eleven heads shared `recipe_hash` 86c51a5dca220f63 and none recorded which
    preprocessing arm it expects.

These tests pin the shared definition and the guard. Synthetic and read-only.
"""
import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy/torch

import json
from pathlib import Path

import numpy as np
import pytest

from src.modeling.mlp_head import (A1_NORM_ARM, NO_NORM_ARM, DeployableHead, infer_norm_arm,
                                   require_norm_arm)
from src.striping import (A1_ARM, A1_MIN_FRAME_PX, a1_normalize_native, a1_stats,
                          a1_stats_from_hist)

REPO = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ the statistic

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_histogram_percentiles_are_exact_not_approximate(seed):
    """A 2.2-Gpx tile cannot hold its values, but uint8 has only 256 of them — so the
    streamed statistic must be the *exact* percentile, not a binned estimate."""
    rng = np.random.default_rng(seed)
    v = rng.integers(1, 256, size=20000).astype(np.uint8)
    hist = np.bincount(v, minlength=256)
    med_h, iqr_h = a1_stats_from_hist(hist)
    med_d, iqr_d = a1_stats(v)
    assert med_h == pytest.approx(med_d, abs=1e-9)
    assert iqr_h == pytest.approx(iqr_d, abs=1e-9)


def test_histogram_statistic_excludes_the_nodata_sentinel():
    """DN 0 is the Murray mosaic nodata sentinel; counting it would drag every frame's
    median toward zero exactly where coverage is patchy."""
    v = np.concatenate([np.zeros(5000, np.uint8), np.full(1000, 100, np.uint8),
                        np.full(1000, 140, np.uint8)])
    assert a1_stats_from_hist(np.bincount(v, minlength=256)) == a1_stats(v)


def test_histogram_statistic_refuses_a_frame_below_the_pixel_floor():
    hist = np.zeros(256, dtype=np.int64)
    hist[120] = A1_MIN_FRAME_PX - 1
    assert all(np.isnan(x) for x in a1_stats_from_hist(hist))


# ------------------------------------------------------------------ no raw DN survives

def _two_frame_scene():
    arr = np.zeros((40, 40), dtype=np.uint8)
    arr[:20] = 100          # frame 0
    arr[20:] = 180          # frame 1
    arr[:, :4] = 0          # a nodata gutter
    labels = np.full((40, 40), -1, dtype=np.int32)
    labels[:20] = 0
    labels[20:35] = 1
    # rows 35-40 deliberately left unlabelled -> the R08 population
    return arr, labels


def test_no_valid_pixel_is_left_at_raw_dn():
    """R08, structurally. `a1_normalize_per_frame` returned unlabelled and small-frame pixels
    at raw DN, putting two radiometric scales in one array and handing the mixture to a frozen
    embedder that cannot tell them apart."""
    arr, labels = _two_frame_scene()
    stats = {0: (100.0, 10.0), 1: (180.0, 10.0)}
    out = a1_normalize_native(arr, labels, stats, fallback=(150.0, 20.0))
    unlabelled = (labels < 0) & (arr > 0)
    assert unlabelled.any(), "fixture must exercise the unlabelled case"
    assert not np.array_equal(out[unlabelled], arr[unlabelled]), "unlabelled kept raw DN"
    assert (out[arr == 0] == 0).all(), "nodata must stay the sentinel"
    assert (out[arr > 0] > 0).all(), "no valid pixel may become nodata here"


def test_a_frame_lands_on_the_reference_and_frames_become_comparable():
    """The whole point of A1: two frames at different DN levels must come out at the same
    level, which is what makes the frozen embedder see them alike."""
    arr, labels = _two_frame_scene()
    stats = {0: (100.0, 10.0), 1: (180.0, 10.0)}
    out = a1_normalize_native(arr, labels, stats, fallback=(150.0, 20.0))
    assert out[5, 20] == out[25, 20], "two frames at their own medians must map together"


def test_missing_fallback_raises_rather_than_silently_returning_raw():
    arr, labels = _two_frame_scene()
    with pytest.raises(ValueError, match="R08"):
        a1_normalize_native(arr, labels, {0: (100.0, 10.0)}, fallback=None)


def test_unlabelled_pixels_are_normalized_not_dropped_the_ratified_r08_contract():
    """**R08's contract, RATIFIED 2026-08-10 (Brian) — and this test is the ratification.**

    The open question was whether an unlabelled pixel should take the tile-wide fallback
    statistic (an approximation) or be dropped as nodata (exact, but it removes real ground).
    Measured on three whole cached Murray tiles, dropping is catastrophically the wrong trade:

      * the population is **isolated single pixels** — horizontal run length median 1, p90 2,
        max 15 — scattered over 21k-30k of a tile's 2.19 M blocks. They are rasterization
        precision gaps inside the dissolved SeamMap polygons, not real coverage holes;
      * they are 0.0058-0.0108 % of valid pixels;
      * but dropping them makes each one nodata, and R13's zero-tolerance context gate then
        masks every coarse cell whose 96-px box touches one. Cost: **3.11 %, 3.37 % and 4.38 %
        of the tile** (E8_N44, E4_N44, E-8_N32) against 0.00 %, 0.00 % and 0.072 % today —
        a 400-530x amplification in cell-equivalents.

    So: **normalize them, never drop them.** Trading 3-4 % of the map to avoid a 1e-4
    radiometric approximation is the wrong direction by three orders of magnitude. This test
    fails if anyone later "tightens" the contract by masking the fallback population.
    """
    arr, labels = _two_frame_scene()
    stats = {0: (100.0, 10.0), 1: (180.0, 10.0)}
    out = a1_normalize_native(arr, labels, stats, fallback=(150.0, 20.0))

    unlabelled = (labels < 0) & (arr > 0)
    assert unlabelled.any(), "fixture must exercise the unlabelled case"
    # the contract, in one line: an unlabelled VALID pixel stays valid
    assert (out[unlabelled] > 0).all(), (
        "unlabelled valid pixels were dropped to the nodata sentinel; R08 was ratified the "
        "other way — see the measured 3-4 %-of-tile cost in this test's docstring")
    # ... and it is genuinely the FALLBACK statistic they carry, not a frame's
    from src.striping import a1_apply
    assert np.array_equal(out[unlabelled], a1_apply(arr, 150.0, 20.0)[unlabelled])


def test_the_small_frame_floor_is_a_tripwire_not_a_tuning_knob():
    """R08's second half, answered by measurement rather than by choosing a number.

    Across four real tiles / 214 dissolved SeamMap frames, **exactly one** frame fell below
    `A1_MIN_FRAME_PX` (E-12_N36, 1 of 81; the other three tiles were 0 of 54, 0 of 48, 0 of 31).
    The floor is therefore not a knob whose value trades anything off on real data — it is a
    guard against a degenerate frame, and 50 px is comfortably below any real one. What it must
    keep doing is *route* such a frame to the fallback rather than admit it.
    """
    assert A1_MIN_FRAME_PX == 50
    tiny = np.full(A1_MIN_FRAME_PX - 1, 120, dtype=np.uint8)
    assert all(np.isnan(v) for v in a1_stats(tiny)), "a sub-floor frame must not get a statistic"


# ------------------------------------------------------------------ arm versioning

def test_the_two_arm_literals_agree():
    """`src.striping` owns the definition and `src.modeling.mlp_head` repeats the literal to
    avoid an import cycle; if they ever drift, every arm check silently stops matching."""
    assert A1_ARM == A1_NORM_ARM


@pytest.mark.parametrize("store,expect", [
    ("fang_embeddings", NO_NORM_ARM),
    ("fang_embeddings_a1", A1_NORM_ARM),
    ("fang_embeddings_a1_scratch", A1_NORM_ARM),
    (None, NO_NORM_ARM),
])
def test_the_arm_is_inferred_from_the_store_name(store, expect):
    assert infer_norm_arm(store) == expect


class _Head:
    def __init__(self, arm):
        self.norm_arm = arm


def test_a_matching_arm_passes_and_a_contradicting_one_raises():
    assert require_norm_arm(_Head(A1_ARM), A1_ARM) == A1_ARM
    with pytest.raises(ValueError, match="different input distributions"):
        require_norm_arm(_Head(A1_ARM), NO_NORM_ARM, where="map_region")
    with pytest.raises(ValueError, match="different input distributions"):
        require_norm_arm(_Head(NO_NORM_ARM), A1_ARM, where="striping_a1_map")


def test_an_unversioned_head_is_refused_on_the_a1_path_but_only_warned_on_the_baseline():
    """Asymmetric on purpose: feeding the A1 head the wrong DN is the dangerous direction and
    that head must be retrained for R07 anyway, whereas unversioned + raw DN is exactly the
    pre-R07 status quo and blocking the baseline re-render on it would buy no safety."""
    with pytest.raises(ValueError, match="predates R07 arm versioning"):
        require_norm_arm(_Head(None), A1_ARM, strict=True)
    with pytest.warns(RuntimeWarning, match="predates R07 arm versioning"):
        assert require_norm_arm(_Head(None), NO_NORM_ARM, strict=False) is None


def test_declaring_an_arm_changes_the_recipe_hash_and_omitting_it_does_not():
    """Eleven heads shared one recipe_hash. Two heads that expect different input
    distributions are not the same recipe — but pre-R07 hashes must not move."""
    plain = DeployableHead()
    base = DeployableHead(norm_arm=NO_NORM_ARM)
    a1 = DeployableHead(norm_arm=A1_ARM)
    assert plain.recipe_hash() != base.recipe_hash() != a1.recipe_hash()
    assert base.recipe_hash() != a1.recipe_hash()
    # an undeclared arm reproduces the historical hash exactly, so existing dirs still resolve
    assert plain.recipe_hash() == DeployableHead(norm_arm=None).recipe_hash()


def test_banked_heads_split_into_pre_and_post_R07_generations():
    """R07 on disk. **Updated 2026-08-23** — the v2 rebuild trained the first armed heads.

    This test used to assert `armed == 0`, pinning the pre-fix state R07 found: eleven heads
    sharing one `recipe_hash` with the arm recorded nowhere, so the ONLY thing distinguishing
    them was the parent directory name. It carried its own instruction to be updated once a
    head declared an arm. That has now happened, so it pins the *invariant* instead of the
    snapshot:

      * the legacy collision is still on disk and still documents what R07 found;
      * every head that declares an arm has a hash unique to that arm.

    The second clause is the part that must hold forever. If two heads with DIFFERENT arms
    ever share a `recipe_hash`, R07 has regressed and a raster can be rendered with the wrong
    preprocessing and no error.
    """
    cards = sorted((REPO / "models").glob("deployable*/*/recipe.json"))
    if not cards:
        pytest.skip("no banked heads on disk")
    by_hash: dict[str, list] = {}
    for c in cards:
        d = json.loads(c.read_text(encoding="utf-8"))
        by_hash.setdefault(d.get("recipe_hash"), []).append(
            (c.parent.parent.name, d.get("norm_arm")))

    unversioned = {h: v for h, v in by_hash.items() if all(a is None for _, a in v)}
    armed = {h: v for h, v in by_hash.items() if any(a is not None for _, a in v)}

    assert unversioned, "expected the pre-R07 unversioned heads to still be on disk"
    assert max((len(v) for v in unversioned.values()), default=0) > 1, (
        "expected the pre-R07 recipe_hash collision among the legacy heads")

    # THE INVARIANT: one hash never spans two arms.
    for h, members in armed.items():
        arms = {a for _, a in members}
        assert len(arms) == 1, (
            f"recipe_hash {h} spans arms {sorted(arms)} across {[n for n, _ in members]} -- "
            f"R07 has regressed; a head can now be fed the wrong preprocessing silently")

    # and an undeclared arm must still reproduce the historical hash, so legacy dirs resolve
    assert DeployableHead(norm_arm=None).recipe_hash() != DeployableHead(
        norm_arm="a1").recipe_hash()
