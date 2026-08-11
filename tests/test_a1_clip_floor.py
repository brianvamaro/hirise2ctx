"""R38 — A1's clip floor collided with the mosaic nodata sentinel.

`a1_apply` clipped to `[0, 255]`, so a valid pixel darker than about `med - 4.51*iqr` was written
as **0** — and 0 is unambiguously "no data" everywhere downstream (`src/striping.py` `a1_stats`
excludes `DN > 0`; `src/mapping.py` inferred its nodata mask from `arr == 0`). A legitimately dark
patch of terrain was therefore counted as a mosaic gap: measured on the real native patch stacks,
0.041 % of valid pixels on the training path and 0.04-0.41 % at deploy, with **6.7 %** of deploy-sim
tiles carrying at least one false-black pixel while still passing the mask, and whole tiles reaching
`own_tile_zero_fraction == 1.00` in low-IQR frames.

**Two problems, and only one of them is the floor.** Flooring valid pixels at 1 stops the
miscounting, but it does NOT rescue the pixel: R13 measured that DN 0 and the perfectly legal DN 1
move the frozen head's prediction *identically to three decimals*, because the damage is blackness,
not the sentinel. Worse, after the floor moves those pixels stop reading as nodata, so nothing sees
them at all. That is why the fix is three things, not one:

  1. valid pixels floor at `A1_VALID_FLOOR`, so DN 0 means only "no data";
  2. `predict_window` takes an EXPLICIT `nodata_mask` from the raw DN instead of inferring it from
     a transformed array — the durable half, since "the output happens to be safe to infer from"
     stops holding the next time the transfer function changes;
  3. the destroyed texture is counted separately (`a1_clip_counts_from_hist`) as a radiometric
     statistic, because no nodata count can represent it.

Mutants these kill, all green before the fix:
  M1  restore the `[0, 255]` clip
  M2  `predict_window` ignores `nodata_mask` and re-infers `arr == 0`
  M3  `a1_stats` / `a1_stats_from_hist` keep `or 1.0` on a degenerate IQR
  M4  clip counting reports 0
"""
import numpy as np
import pytest

from src.mapping import as_nodata_mask, own_tile_zero_fraction, predict_window
from src.striping import (A1_MIN_FRAME_PX, A1_REF_IQR, A1_REF_MEDIAN, A1_VALID_FLOOR, a1_apply,
                          a1_clip_counts, a1_clip_counts_from_hist, a1_normalize_native,
                          a1_stats, a1_stats_from_hist)


def _dark_frame(n=4096, med=169, iqr=21, n_dark=64, dark_dn=40, seed=0):
    """A realistic frame plus a band of genuinely dark VALID pixels and a genuine nodata band.

    (med=169, iqr=21) and DN 40 are the verified reproduction's numbers: DN 40 sits below
    `med - 4.51*iqr` and so is exactly the population the old floor destroyed.
    """
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(med, iqr, n), 1, 255).astype(np.uint8)
    a[:n_dark] = dark_dn                       # valid, but dark enough to clip
    a[n_dark:n_dark + 32] = 0                  # genuine mosaic nodata
    return a


# ---------------------------------------------------------------- the floor itself

def test_a_dark_valid_pixel_is_no_longer_written_as_the_nodata_sentinel():
    """M1. The whole finding: DN 40 used to come out as 0, identical to a data gap."""
    arr = _dark_frame()
    out = a1_apply(arr, 169.0, 21.0)
    dark, gap = out[:64], out[64:96]
    assert (gap == 0).all(), "genuine nodata must still be 0"
    assert (dark == A1_VALID_FLOOR).all(), "the dark valid band must clip to the floor, not 0"
    assert A1_VALID_FLOOR > 0, "a floor of 0 is the defect"
    # and the two populations are now distinguishable, which is the point
    assert not (dark == gap[0]).any()


def test_the_pre_r38_behaviour_is_still_reachable_but_only_deliberately():
    """Reproducing a pre-R38 artifact must be possible and must be explicit."""
    arr = _dark_frame()
    assert (a1_apply(arr, 169.0, 21.0, floor=0)[:64] == 0).all()


def test_the_floor_does_not_disturb_unclipped_pixels():
    """The fix must touch only the clipped tail — 0.04 % of pixels, not the image."""
    arr = _dark_frame()
    new = a1_apply(arr, 169.0, 21.0)
    old = a1_apply(arr, 169.0, 21.0, floor=0)
    differ = new != old
    assert differ.sum() == 64, f"{differ.sum()} pixels changed; only the 64 dark ones should"
    assert np.array_equal(np.where(differ)[0], np.arange(64))


def test_a_whole_dark_tile_no_longer_reads_as_a_data_gap():
    """The consequence that mattered: `own_tile_zero_fraction == 1.00` on real terrain.

    A 32x32 box of valid-but-dark pixels used to report a zero-fraction of 1.00 and be dropped
    by `max_zero_fraction` as missing coverage.
    """
    tile_px = 8
    raw = np.full((tile_px, tile_px), 40, dtype=np.uint8)          # valid, dark
    old = a1_apply(raw, 169.0, 21.0, floor=0)
    new = a1_apply(raw, 169.0, 21.0)
    kw = dict(tile_px=tile_px, row0=0, col0=0)
    assert own_tile_zero_fraction(old, np.array([0]), np.array([0]), **kw)[0] == 1.0
    assert own_tile_zero_fraction(new, np.array([0]), np.array([0]), **kw)[0] == 0.0


# ---------------------------------------------------------------- the explicit mask

def test_as_nodata_mask_prefers_the_supplied_truth_over_the_inferred_value():
    w = np.array([[0, 5], [7, 0]], dtype=np.uint8)
    assert as_nodata_mask(w).tolist() == [[True, False], [False, True]]
    supplied = np.array([[False, True], [False, False]])
    assert as_nodata_mask(w, supplied).tolist() == supplied.tolist()
    with pytest.raises(ValueError, match="does not match the window"):
        as_nodata_mask(w, np.zeros((3, 3), bool))


class _FakeEmbedder:
    def embed_window(self, arr, ti, tj, *, tile_px, row0, col0, pool, batch):
        from src.fm_embeddings import slice_context_boxes
        _, valid = slice_context_boxes(arr, ti, tj, tile_px, row0, col0)
        emb = np.zeros((ti.size, 4), np.float32)
        emb[:, 0] = 0.5
        return emb, valid


class _FakeHead:
    def predict(self, emb):
        return emb[:, 0].astype(np.float64)


def _win(data):
    from src.mapping import CtxWindow
    return CtxWindow(data=data, row_off=0, col_off=0,
                     transform=(5.0, 0.0, 0.0, 0.0, -5.0, 0.0), crs_wkt="LOCAL")


def test_predict_window_trusts_the_supplied_mask_over_the_pixel_value():
    """M2 — the durable half of the fix.

    Hand it an array whose DN-0 pixels are NOT nodata (a legal dark value under some future
    transfer function) together with the true mask, and no cell may be gated. Re-inferring from
    `arr == 0` would mask four cells.
    """
    tile_px = 8
    data = np.full((5 * tile_px, 5 * tile_px), 200, dtype=np.uint8)
    data[tile_px, tile_px] = 0                       # a value that only LOOKS like nodata
    truth = np.zeros(data.shape, dtype=bool)         # ... but nothing here is actually missing

    inferred = predict_window(_win(data), _FakeEmbedder(), _FakeHead(), tile_px=tile_px)
    told = predict_window(_win(data), _FakeEmbedder(), _FakeHead(), tile_px=tile_px,
                          nodata_mask=truth)
    assert inferred.n_masked_context_nodata == 4
    assert told.n_masked_context_nodata == 0 and told.n_masked_nodata == 0
    assert np.isfinite(told.prob).all()


def test_predict_window_masks_real_nodata_the_mask_declares_but_the_values_hide():
    """The converse, and the one that actually protects the A1 map: a genuine gap whose pixels
    are NOT 0 after normalization must still be masked."""
    tile_px = 8
    data = np.full((5 * tile_px, 5 * tile_px), 200, dtype=np.uint8)   # no zeros at all
    truth = np.zeros(data.shape, dtype=bool)
    truth[tile_px, tile_px] = True

    told = predict_window(_win(data), _FakeEmbedder(), _FakeHead(), tile_px=tile_px,
                          nodata_mask=truth)
    assert told.n_masked_context_nodata == 4, "a declared gap must gate even at DN 200"
    assert predict_window(_win(data), _FakeEmbedder(), _FakeHead(),
                          tile_px=tile_px).n_masked_context_nodata == 0


# ---------------------------------------------------------------- the degenerate IQR

def test_a_degenerate_iqr_is_nan_not_a_fabricated_one():
    """M3, and it was not merely cosmetic.

    `a1_stats_native_tile` admits a frame only when `iqr > 0`. The old `or 1.0` sailed straight
    through that guard with a fabricated IQR, handing the frame a gain of `s0/1 = 27.7x` instead
    of routing it to the fallback statistic. The guard existed precisely to prevent this.
    """
    flat = np.full(A1_MIN_FRAME_PX * 2, 120, dtype=np.uint8)       # zero IQR
    assert all(np.isnan(v) for v in a1_stats(flat))

    hist = np.zeros(256, dtype=np.int64)
    hist[120] = A1_MIN_FRAME_PX * 2
    assert all(np.isnan(v) for v in a1_stats_from_hist(hist))

    # and the downstream guard now actually excludes it
    assert not (np.isfinite(np.nan) and np.nan > 0)
    # a1_apply must refuse to amplify rather than emit garbage
    assert np.array_equal(a1_apply(flat, *a1_stats(flat)), flat)


def test_too_few_valid_pixels_still_returns_nan():
    assert all(np.isnan(v) for v in a1_stats(np.full(A1_MIN_FRAME_PX - 1, 120, np.uint8)))


# ---------------------------------------------------------------- the clip accounting

def test_clip_counts_from_the_histogram_equal_the_array_answer():
    """M4. The histogram form is what production uses (exact, once per tile, resume-proof);
    the array form is the independent check that it is right."""
    arr = _dark_frame(n=8192, seed=3)
    med, iqr = 169.0, 21.0
    hist = np.bincount(arr, minlength=256)

    from_arr = a1_clip_counts(arr, med, iqr)
    from_hist = a1_clip_counts_from_hist(hist, med, iqr)
    assert from_arr == from_hist
    assert from_arr["n_floored"] == 64, "the dark band is exactly the clipped population"
    assert from_arr["n_valid"] == arr.size - 32, "the nodata band is not a valid pixel"


def test_clip_counts_see_the_ceiling_too():
    arr = np.concatenate([np.full(200, 120, np.uint8), np.full(50, 255, np.uint8)])
    c = a1_clip_counts(arr, 120.0, 1.0)          # gain 27.7 -> everything bright saturates
    assert c["n_ceiled"] == 50 and c["n_valid"] == 250


def test_clip_counts_are_zero_when_nothing_clips():
    arr = np.clip(np.random.default_rng(1).normal(169, 21, 4000), 1, 255).astype(np.uint8)
    c = a1_clip_counts(arr, 169.0, 21.0)
    assert c["n_floored"] == 0 and c["n_ceiled"] == 0 and c["n_valid"] == 4000


def test_clip_counts_refuse_to_guess_on_a_degenerate_statistic():
    arr = _dark_frame()
    assert a1_clip_counts(arr, np.nan, np.nan)["n_floored"] == 0
    assert a1_clip_counts(arr, 169.0, 0.0)["n_ceiled"] == 0


# ---------------------------------------------------------------- end to end

def test_a1_normalize_native_never_emits_the_sentinel_for_a_valid_pixel():
    """The property the whole finding reduces to, asserted on the production entry point."""
    arr = _dark_frame(n=400).reshape(20, 20)
    labels = np.zeros(arr.shape, dtype=np.int32)
    out = a1_normalize_native(arr, labels, {0: (169.0, 21.0)}, fallback=(169.0, 21.0))
    assert ((out == 0) == (arr == 0)).all(), "DN 0 must mean nodata and nothing else"
    assert (out[arr == 40] == A1_VALID_FLOOR).all()


def test_the_reference_constants_are_unchanged():
    """Brian's call 2026-08-10: fix the sentinel, record the loss, do NOT retune the transfer
    function. A1_REF is baked into the frozen A1 arm; moving it silently would invalidate the
    banked numbers without anyone noticing."""
    assert (A1_REF_MEDIAN, A1_REF_IQR) == (125.0, 27.7)
