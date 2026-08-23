"""The A1 LOIO artifact must retain (obs_id, ti, tj) or step 9 cannot calibrate that arm.

DECISIONS 2026-08-21. `striping_a1_loio.py` emitted only `obs_id, y, p`, while
`scripts/bank_calibration.py` merges on `(obs_id, ti, tj)` with `validate="one_to_one"`.
The audit's own rebuild-DAG bullet requires the keys; its gate table wrongly listed the
requirement as already enforced.

Read-only: everything is built in tmp_path.
"""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]

# what scripts/bank_calibration.py needs to join predictions to labels
REQUIRED = ["obs_id", "ti", "tj", "y_true", "y_pred"]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "_striping_a1_loio", REPO / "scripts" / "striping_a1_loio.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _preds(store: str, n: int, *, ti0: int = 0):
    return pd.DataFrame({
        "obs_id": [f"ESP_{i:06d}_2000" for i in range(n)],
        "ti": range(ti0, ti0 + n), "tj": range(100, 100 + n),
        "y": [0, 1] * (n // 2), "p": [0.1 * i for i in range(n)], "store": store,
    })


def test_both_arms_written_in_the_calibration_schema(tmp_path):
    m = _mod()
    allrows = pd.concat([_preds("fang_embeddings", 6), _preds("fang_embeddings_a1", 6)],
                        ignore_index=True)
    written = m.write_arm_predictions(allrows, tmp_path)
    assert set(written) == {"fang_embeddings", "fang_embeddings_a1"}
    for store, path in written.items():
        d = pd.read_parquet(path)
        assert list(d.columns) == REQUIRED, f"{store} schema drifted: {list(d.columns)}"
        assert len(d) == 6
        assert d[["obs_id", "ti", "tj"]].duplicated().sum() == 0


def test_tile_keys_survive_the_roundtrip(tmp_path):
    """The point of the change: the exact keys in must be the keys out."""
    m = _mod()
    src = _preds("fang_embeddings_a1", 4, ti0=77)
    path = m.write_arm_predictions(src, tmp_path)["fang_embeddings_a1"]
    got = pd.read_parquet(path)
    assert set(zip(got.ti, got.tj)) == set(zip(src.ti, src.tj))
    assert got.y_pred.tolist() == src.p.tolist()
    assert got.y_true.tolist() == src.y.tolist()


def test_duplicate_tile_keys_are_a_hard_error(tmp_path):
    """A LOIO artifact has one row per tile by construction; a duplicate means folds
    overlapped or a store was concatenated twice. That must not reach the calibrator."""
    m = _mod()
    dup = pd.concat([_preds("fang_embeddings_a1", 4)] * 2, ignore_index=True)
    with pytest.raises(SystemExit, match="duplicate"):
        m.write_arm_predictions(dup, tmp_path)


def test_tag_keeps_a_restricted_run_from_overwriting_the_record(tmp_path):
    """--restrict-store changes the numbers, so its artifacts must land elsewhere."""
    m = _mod()
    src = _preds("fang_embeddings_a1", 4)
    m.write_arm_predictions(src, tmp_path)
    m.write_arm_predictions(src, tmp_path, tag="_36")
    assert (tmp_path / "loio_fang_embeddings_a1" / "predictions.parquet").is_file()
    assert (tmp_path / "loio_fang_embeddings_a1_36" / "predictions.parquet").is_file()
