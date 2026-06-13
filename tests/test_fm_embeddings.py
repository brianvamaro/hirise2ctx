"""Unit tests for the productized Fang-ViT embedding path (PLAN_FM §2.2).

Synthetic data only (random-weight ViT for the forward/pooling contracts, a
hand-built window for the slicing geometry, a tmp npz store for the loader
join) — no checkpoint or dataset on disk. The one checkpoint-dependent test is
marked `slow` and skips when the 341 MB weights are absent.
"""
import src.modeling  # noqa: F401 -- Windows DLL bootstrap; must precede numpy/torch

import numpy as np
import pandas as pd
import pytest
import torch

from src import fm_embeddings as fm
from src.fm_embeddings import (
    EMBED_DIM, FangEmbedder, build_vit_b16, gem_pool_np, load_timm_state_dict,
    slice_context_boxes, tile_grid_for_window,
)


# ----------------------------------------------------------------------------
# Pooling
# ----------------------------------------------------------------------------


def test_gem_pool_p1_equals_mean():
    rng = np.random.default_rng(0)
    x = rng.random((5, 17, EMBED_DIM)).astype(np.float32)
    np.testing.assert_allclose(gem_pool_np(x, p=1.0), x.mean(axis=1), rtol=1e-5, atol=1e-5)


def test_gem_pool_between_mean_and_max():
    # GeM(p=3) of a positive stack lies between the mean and the max per channel.
    rng = np.random.default_rng(1)
    x = rng.random((3, 40, EMBED_DIM)).astype(np.float32) + 0.1
    g = gem_pool_np(x, p=3.0)
    assert np.all(g >= x.mean(axis=1) - 1e-5)
    assert np.all(g <= x.max(axis=1) + 1e-5)


def test_gem_pool_matches_torch_path():
    # The numpy reference and the torch _pool_tokens GeM must agree.
    rng = np.random.default_rng(2)
    patch = rng.random((4, 196, EMBED_DIM)).astype(np.float32)
    tokens = torch.from_numpy(np.concatenate([np.zeros((4, 1, EMBED_DIM), np.float32), patch], axis=1))
    torch_gem = fm._pool_tokens(tokens)["gem"].numpy()
    np.testing.assert_allclose(gem_pool_np(patch), torch_gem, rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------------------------
# ViT encoder + embedder (random weights, CPU)
# ----------------------------------------------------------------------------


def _cpu_embedder() -> FangEmbedder:
    torch.manual_seed(0)
    return FangEmbedder(build_vit_b16().eval(), torch.device("cpu"))


def test_vit_forward_token_shape():
    model = build_vit_b16().eval()
    with torch.no_grad():
        tokens = model(torch.zeros(2, 1, fm.MODEL_INPUT, fm.MODEL_INPUT))
    assert tokens.shape == (2, 197, EMBED_DIM)  # 14*14 patches + cls


def test_embed_patches_shape_and_determinism():
    emb = _cpu_embedder()
    rng = np.random.default_rng(3)
    patches = rng.integers(0, 256, size=(7, 96, 96), dtype=np.uint8)
    a = emb.embed_patches(patches, pool="gem")
    b = emb.embed_patches(patches, pool="gem")
    assert a.shape == (7, EMBED_DIM) and a.dtype == np.float32
    np.testing.assert_array_equal(a, b)  # frozen weights -> deterministic


def test_embed_patches_empty():
    assert _cpu_embedder().embed_patches(np.zeros((0, 96, 96), np.uint8)).shape == (0, EMBED_DIM)


def test_embed_patches_rejects_bad_pool():
    with pytest.raises(ValueError):
        _cpu_embedder().embed_patches(np.zeros((1, 96, 96), np.uint8), pool="bogus")


# ----------------------------------------------------------------------------
# Geometry: 3×3-context slicing on the mosaic-anchored grid
# ----------------------------------------------------------------------------


def test_slice_context_center_equals_own_tile():
    # Window encodes mosaic position so each box's center must equal the own tile.
    tile_px, row0, col0 = 4, 10, 7
    H = W = 60
    rr, cc = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    window = (((rr + row0) * 31 + (cc + col0) * 17) % 256).astype(np.uint8)

    ti = np.array([4, 5, 6])
    tj = np.array([3, 4, 5])
    boxes, valid = slice_context_boxes(window, ti, tj, tile_px, row0, col0)
    assert valid.all()
    for out_row, (t_i, t_j) in enumerate(zip(ti, tj)):
        r_win = t_i * tile_px - row0
        c_win = t_j * tile_px - col0
        own = window[r_win: r_win + tile_px, c_win: c_win + tile_px]
        center = boxes[out_row, tile_px: 2 * tile_px, tile_px: 2 * tile_px]
        np.testing.assert_array_equal(center, own)


def test_slice_context_marks_edge_tiles_invalid():
    tile_px, row0, col0 = 4, 0, 0
    window = np.zeros((40, 40), dtype=np.uint8)
    # ti=0 -> box starts at row -4 (spills); ti=5 -> box rows [16,28) fits.
    ti = np.array([0, 5])
    tj = np.array([5, 5])
    boxes, valid = slice_context_boxes(window, ti, tj, tile_px, row0, col0)
    assert list(valid) == [False, True]
    assert boxes.shape == (1, 3 * tile_px, 3 * tile_px)


def test_tile_grid_matches_slicing_validity():
    tile_px, row0, col0 = 4, 10, 7
    shape = (60, 60)
    ti, tj = tile_grid_for_window(shape, row0, col0, tile_px)
    window = np.zeros(shape, dtype=np.uint8)
    _, valid = slice_context_boxes(window, ti, tj, tile_px, row0, col0)
    assert valid.all() and ti.size > 0  # every enumerated tile has full context


def test_embed_window_nan_rows_for_invalid():
    emb = _cpu_embedder()
    tile_px, row0, col0 = 4, 0, 0
    window = np.random.default_rng(4).integers(0, 256, size=(40, 40), dtype=np.uint8)
    ti = np.array([0, 5])   # first invalid, second valid
    tj = np.array([5, 5])
    out, valid = emb.embed_window(window, ti, tj, tile_px=tile_px, row0=row0, col0=col0)
    assert out.shape == (2, EMBED_DIM)
    assert np.isnan(out[0]).all() and not np.isnan(out[1]).any()
    assert list(valid) == [False, True]


# ----------------------------------------------------------------------------
# Loader feature source (cached npz store join)
# ----------------------------------------------------------------------------


def _write_store(tmp_path, obs_rows: dict[str, int], px: int = 96):
    fdir = tmp_path / "fang_embeddings"
    fdir.mkdir(parents=True)
    rng = np.random.default_rng(5)
    for obs, n in obs_rows.items():
        np.savez(
            fdir / f"{obs}_P{px}.npz",
            ti=np.arange(n, dtype=np.int32),
            tj=np.zeros(n, dtype=np.int32),
            valid=np.array([True] * n, dtype=bool),
            cls=rng.random((n, EMBED_DIM)).astype(np.float32),
            mean=rng.random((n, EMBED_DIM)).astype(np.float32),
            gem=rng.random((n, EMBED_DIM)).astype(np.float32),
        )
    return tmp_path


def test_fang_columns_keyed_lookup(tmp_path):
    from src.modeling.loaders import fang_columns_for_keys, load_fang_store

    _write_store(tmp_path, {"ESP_A": 4, "ESP_B": 3})
    index, matrix = load_fang_store(96, pool="gem", dataset_dir=tmp_path)
    # Request rows out of store order; the join must follow the keys, not the store.
    keys = pd.DataFrame({"obs_id": ["ESP_B", "ESP_A"], "ti": [2, 0], "tj": [0, 0]})
    cols, names = fang_columns_for_keys(keys, 96, pool="gem", dataset_dir=tmp_path)
    assert cols.shape == (2, EMBED_DIM)
    assert names[0] == "fang_gem96_000" and len(names) == EMBED_DIM
    # Row 0 of the result is ESP_B/ti=2; verify against the store directly.
    b_row = index.query("obs_id == 'ESP_B' and ti == 2")["row"].iloc[0]
    np.testing.assert_array_equal(cols[0], matrix[b_row])


def test_fang_columns_missing_tile_raises(tmp_path):
    from src.modeling.loaders import fang_columns_for_keys

    _write_store(tmp_path, {"ESP_A": 2})
    keys = pd.DataFrame({"obs_id": ["ESP_A"], "ti": [99], "tj": [0]})  # not in store
    with pytest.raises(AssertionError):
        fang_columns_for_keys(keys, 96, pool="gem", dataset_dir=tmp_path)


def test_augment_fold_replace_vs_concat(tmp_path):
    from src.modeling.loaders import Fold, augment_fold_with_fang

    _write_store(tmp_path, {"ESP_A": 5})
    keys = pd.DataFrame({"obs_id": ["ESP_A"] * 5, "ti": np.arange(5), "tj": np.zeros(5, int)})
    X = np.ones((5, 3), dtype=np.float32)
    fold = Fold(
        fold_idx=0, scheme="t", scale_idx=2,
        X_train=X, y_train=pd.DataFrame({"y": np.zeros(5)}),
        groups_train=np.zeros(5, np.int32), keys_train=keys,
        X_test=X[:2], y_test=pd.DataFrame({"y": np.zeros(2)}),
        groups_test=np.zeros(2, np.int32), keys_test=keys.iloc[:2].reset_index(drop=True),
        feature_names=["a", "b", "c"], obs_to_int={"ESP_A": 0}, held_out_obs_ids=["ESP_A"],
    )
    concat = augment_fold_with_fang(fold, px=96, dataset_dir=tmp_path, replace=False)
    assert concat.X_train.shape == (5, 3 + EMBED_DIM)
    assert concat.feature_names[:3] == ["a", "b", "c"]

    repl = augment_fold_with_fang(fold, px=96, dataset_dir=tmp_path, replace=True)
    assert repl.X_train.shape == (5, EMBED_DIM)
    assert repl.X_test.shape == (2, EMBED_DIM)
    assert len(repl.feature_names) == EMBED_DIM


# ----------------------------------------------------------------------------
# Checkpoint-dependent (slow; skipped when weights absent)
# ----------------------------------------------------------------------------


@pytest.mark.slow
def test_real_checkpoint_loads_strict_and_embeds():
    if not fm.DEFAULT_CKPT.exists():
        pytest.skip("Fang checkpoint absent (Zenodo 18180801, untracked 341 MB)")
    emb = FangEmbedder.load(device="cpu")
    out = emb.embed_patches(
        np.random.default_rng(0).integers(0, 256, size=(2, 96, 96), dtype=np.uint8))
    assert out.shape == (2, EMBED_DIM) and np.isfinite(out).all()
