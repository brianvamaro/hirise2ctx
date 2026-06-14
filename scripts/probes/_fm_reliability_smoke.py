"""Smoke test for src.reliability novelty scorers on synthetic near/far data."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.reliability import MahalanobisNovelty, KNNNovelty, aggregate_per_image

rng = np.random.default_rng(0)
train = rng.normal(size=(2000, 768)).astype("float32")
near = rng.normal(size=(50, 768)).astype("float32")
far = (rng.normal(size=(50, 768)) + 8).astype("float32")
X = np.vstack([near, far])
X[0] = np.nan  # margin tile

for M in [MahalanobisNovelty(n_components=64), KNNNovelty(k=20, max_reference=1000)]:
    M.fit(train)
    s = M.score(X)
    near_med = np.nanmedian(s[1:50])
    far_med = np.nanmedian(s[50:])
    print(type(M).__name__, "nan_row", bool(np.isnan(s[0])),
          "near_med", round(float(near_med), 2), "far_med", round(float(far_med), 2),
          "far>near", bool(far_med > near_med))

obs = np.array(["a"] * 50 + ["b"] * 50)
agg = aggregate_per_image(obs, MahalanobisNovelty(n_components=64).fit(train).score(X), how="median")
print("agg", {k: round(v, 2) for k, v in agg.items()})
print("SMOKE OK")
