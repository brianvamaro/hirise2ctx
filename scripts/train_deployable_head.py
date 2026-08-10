"""Train the deployable frozen head on ALL images (PLAN_FM §2.6.A).

The frozen recipe (`fw_emb_mlp_ens3_gem96_S32_fa_gt_1e-2`, DECISIONS.md
2026-06-12) was validated under LOIO (a fresh head per fold). For the map pilot
we need ONE head trained on every image. This script assembles the all-image
emb-only S=32 matrix and fits `src.modeling.mlp_head.DeployableHead`.

Matrix assembly reuses the exact harness join: across the `loio_nfold` folds,
each image appears once as the held-out TEST set, so concatenating the
fang-augmented (emb-only, P96 / GeM) test slices yields every image's tiles
exactly once -- identical embeddings + labels + group codes to what the LOIO
runs consumed (no re-derivation, no leakage concern: this is just a union, and
the deployable head trains on all of it by design).

Output: `models/deployable/<recipe_hash>/` with the seed state-dicts, scalers,
and a recipe card (`recipe.json`).

Usage:
    conda run --no-capture-output -n geospatial python -u \
        scripts/train_deployable_head.py
    # options: --target fa_gt_1e-2 --batch 4096 --out models/deployable
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np

from src.modeling.binary_target import get_target
from src.modeling.loaders import (fang_columns_for_keys, iter_loio_folds,
                                   load_fang_store)
from src.modeling.mlp_head import (A1_NORM_ARM as A1_ARM, FROZEN_RECIPE, NO_NORM_ARM,
                                   DeployableHead, infer_norm_arm)

DATASET_DIR = REPO_ROOT / "dataset_v2"
SCHEME = "loio_nfold"
SCALE_IDX = 2          # S=32
INPUT_PX = 96          # 3x3-context box side (frozen)
POOL = "gem"


def build_all_image_matrix(target_id: str, store_name: str = "fang_embeddings",
                           dataset_dir: str | Path | None = None):
    """Assemble (X_emb, y_binary, groups, obs_to_int) over every image at S=32.

    Each image is the test set of exactly one LOIO fold; the union of test slices
    is the full cohort, fang-augmented emb-only (P96 / GeM) so X matches the frozen
    matrix the recipe was validated on. ``store_name`` selects the embedding store
    (e.g. ``fang_embeddings_a1`` for the A1 striping-mitigation variant).
    """
    # Isolation criterion 4: the packaged-dataset + embedding-store root is an argument, so
    # a scratch rebuild can train a head without reading the live dataset_v2 tree.
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else DATASET_DIR
    target = get_target(target_id)
    store = load_fang_store(INPUT_PX, pool=POOL, dataset_dir=dataset_dir, store_name=store_name)
    store_obs = set(store[0]["obs_id"].unique())   # some F stores omit images (failed frames)
    Xs, ys, gs = [], [], []
    obs_to_int: dict[str, int] = {}
    skipped = []
    # Union of per-fold TEST slices = each image once. We look up ONLY the test keys
    # (not augment_fold_with_fang, which also joins the train side — every fold's train
    # set contains the store-missing images and would trip the all-present assert).
    for fold in iter_loio_folds(SCHEME, scale_idx=SCALE_IDX, dataset_dir=dataset_dir):
        held = set(fold.keys_test["obs_id"].unique())
        if not held <= store_obs:                  # test image absent from this store
            skipped.extend(sorted(held - store_obs))
            continue
        emb_test, _ = fang_columns_for_keys(fold.keys_test, INPUT_PX, pool=POOL,
                                            dataset_dir=dataset_dir, store=store)
        Xs.append(emb_test)
        ys.append(target.binarize(fold.y_test))
        gs.append(fold.groups_test)
        obs_to_int = fold.obs_to_int  # consistent across folds
    if skipped:
        print(f"  skipped {len(skipped)} image(s) absent from {store_name}: "
              f"{', '.join(skipped)}", flush=True)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    groups = np.concatenate(gs, axis=0)
    return X, y, groups, obs_to_int


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="fa_gt_1e-2")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=str(REPO_ROOT / "models" / "deployable"))
    ap.add_argument("--store-name", default="fang_embeddings",
                    help="embedding store dir name (e.g. fang_embeddings_a1 for the A1 variant)")
    ap.add_argument("--dataset-dir", default=str(DATASET_DIR),
                    help="packaged-dataset + embedding-store root (isolation criterion 4)")
    ap.add_argument("--nuisance-basis", default=None,
                    help="H2: npz with a 'basis' (768, N) array; its first --nuisance-k "
                         "columns are projected out of every embedding before the scaler")
    ap.add_argument("--nuisance-k", type=int, default=0,
                    help="H2: number of nuisance directions to remove (0 = none)")
    ap.add_argument("--consistency-pairs", default=None,
                    help="H3: npz from f_h3_pairs.py with 'ea'/'eb' co-located "
                         "overlap embedding pairs for the consistency penalty")
    ap.add_argument("--lambda-consistency", type=float, default=0.0,
                    help="H3: weight on the co-located overlap MSE penalty (0 = off)")
    # R07: the arm was always knowable at train time (it IS --store-name) and was recorded
    # nowhere, so eleven heads shared one recipe_hash with no way to tell them apart.
    ap.add_argument("--norm-arm", default=None,
                    help="CTX preprocessing arm these embeddings came from, stamped into "
                         f"recipe.json and the recipe hash (e.g. {NO_NORM_ARM!r} for raw "
                         f"Murray DN, {A1_ARM!r} for R07 A1). Default: inferred from "
                         "--store-name.")
    args = ap.parse_args()

    norm_arm = args.norm_arm or infer_norm_arm(args.store_name)
    print(f"  norm_arm={norm_arm}  (store={args.store_name})", flush=True)

    print(f"=== train deployable head ({args.target}, batch={args.batch}) ===", flush=True)
    t0 = time.monotonic()
    X, y, groups, obs_to_int = build_all_image_matrix(
        args.target, store_name=args.store_name, dataset_dir=args.dataset_dir)
    n_img = np.unique(groups).size
    print(f"  matrix: X={X.shape}  pos_rate={float(y.mean()):.4f}  images={n_img}  "
          f"nan_rows={int(np.isnan(X).any(axis=1).sum())}", flush=True)

    basis = None
    if args.nuisance_basis and args.nuisance_k > 0:
        basis = np.load(args.nuisance_basis)["basis"][:, :args.nuisance_k]
        print(f"  H2: removing top-{args.nuisance_k} nuisance directions "
              f"({Path(args.nuisance_basis).name})", flush=True)

    pairs = None
    if args.consistency_pairs and args.lambda_consistency > 0:
        z = np.load(args.consistency_pairs)
        pairs = (z["ea"], z["eb"])
        print(f"  H3: consistency λ={args.lambda_consistency} on {pairs[0].shape[0]} "
              f"co-located overlap pairs ({Path(args.consistency_pairs).name})", flush=True)

    recipe = dict(FROZEN_RECIPE, target_id=args.target)
    head = DeployableHead(seeds=tuple(args.seeds), batch=args.batch, recipe=recipe,
                          nuisance_basis=basis, norm_arm=norm_arm,
                          lambda_consistency=args.lambda_consistency)
    head.fit(X, y, groups=groups, obs_to_int=obs_to_int,
             consistency_pairs=pairs, verbose=True)

    out_dir = Path(args.out) / head.recipe_hash()
    head.save(out_dir)

    # In-sample sanity (NOT a validation number -- LOIO is the honest estimate):
    # confirm the saved head separates rich from poor on the training cohort.
    p = head.predict(X)
    from src.modeling.evaluate import presence_auc
    auc = presence_auc(y.astype(bool), p)
    print(f"\n  in-sample AUC (sanity only) = {auc:.4f}   "
          f"mean p|pos={p[y == 1].mean():.3f}  mean p|neg={p[y == 0].mean():.3f}")
    print(f"  recipe_hash={head.recipe_hash()}  model_hash={head.model_hash()[:16]}")
    try:
        shown = out_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = out_dir.resolve()
    print(f"  [done] {time.monotonic() - t0:.0f} s -> {shown}")

    # Round-trip check: a freshly loaded head must reproduce predictions exactly.
    reloaded = DeployableHead.load(out_dir)
    p2 = reloaded.predict(X[:2048])
    max_diff = float(np.abs(p2 - p[:2048]).max())
    print(f"  save/load round-trip max |dp| = {max_diff:.2e} "
          f"({'OK' if max_diff < 1e-6 else 'MISMATCH'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
