"""Pre-flight for the Fang-ViT probe: dossier classes, S=64 tile counts, patch-idx coverage."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd

dossier = pd.read_parquet(REPO_ROOT / "dataset_v2/w1_dossier.parquet")
print("dossier columns:", list(dossier.columns))
print("dossier index name:", dossier.index.name, " n:", len(dossier))
if "failure_class" in dossier.columns:
    print(dossier["failure_class"].value_counts(dropna=False))
print("validity_ok:", int(dossier["validity_ok"].sum()), "of", len(dossier))

t1 = pd.read_parquet(
    REPO_ROOT / "models/lightgbm_classification/99de85c1ad2a72e6/scale_S64_tfa_gt_1e-2/predictions.parquet")
print("\nT1 S=64 predictions:", t1.shape, "images:", t1["obs_id"].nunique())
print("columns:", list(t1.columns))

# Patch-idx coverage on the packaged v2 dataset at scale_idx 3 (S=64)
from src.modeling.loaders import load_fold
fold = load_fold("loio_nfold", 0, scale_idx=3, dataset_dir=REPO_ROOT / "dataset_v2")
keys_all = pd.concat([fold.keys_train, fold.keys_test], ignore_index=True)
print("\nfold0 S=64 rows train+test:", len(keys_all))
print("keys columns:", list(keys_all.columns))
print("patch_idx_S64 == -1:", int((keys_all["patch_idx_S64"] < 0).sum()))
print("n features:", len(fold.feature_names))
print("first 10 features:", fold.feature_names[:10])

# Neighbor availability for a 3x3 stitch at S=64: for each tile, do all 8 neighbors
# exist (same obs, ti+-1, tj+-1) with a valid patch_idx_S64?
k = keys_all[keys_all["patch_idx_S64"] >= 0][["obs_id", "ti", "tj", "patch_idx_S64"]]
idx = {(o, a, b): p for o, a, b, p in k.itertuples(index=False)}
n_full = 0
for o, a, b, _p in k.itertuples(index=False):
    if all((o, a + da, b + db) in idx for da in (-1, 0, 1) for db in (-1, 0, 1)):
        n_full += 1
print(f"\n3x3-complete tiles: {n_full} / {len(k)} ({n_full/len(k):.1%})")
