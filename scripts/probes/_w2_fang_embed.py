"""W2 Phase 2 lead bet (PLAN_CNN.md 5.1): extract frozen Fang-ViT embeddings for S=64 tiles.

Fang et al. 2026 (doi:10.1029/2025JH000827) ViT-B/16 pretrained MAE+DINO on 3.9M crops of
the Murray Lab CTX mosaic -- the identical product our pipeline windows. Checkpoint
(Zenodo 18180801, models/pretrained/mars-mae-dino-vit-base-v1.pth) is a standard
timm-layout ViT-B/16 state dict (in_chans=1, 224 px, no head), so the encoder forward is
hand-rolled here in plain torch -- no timm/torchvision dependency.

Per tile (scale_idx 3, S=64) two inputs, both bicubic-resized to 224 and normalized
(x/255 - 0.5)/0.5 per the model card:
  - P=64:  the tile's own cached context patch (dataset_v2/context_patches/{obs}_S64.npy);
  - P=192: a 3x3-tile window sliced directly from the cached Stage 2 CTX window
           (NOT stitched from neighbor patches: only 71% of tiles have all 8 neighbors
           emitted, while the window buffer covers most tiles' 192-px surround).
           Tiles whose 192-px box exceeds the window are recorded invalid.

Poolings banked per input: cls token, mean of patch tokens, GeM(p=3) of patch tokens
(probe uses GeM per the plan; the others are free to bank).

Output: dataset_v2/fang_embeddings/{obs_id}_P{px}.npz with
  ti, tj (int32), valid (bool), cls, mean, gem (n, 768) float32 -- row-parallel to the
  S=64 keys for that image. Geometry self-check per image: center 64x64 of each valid
  192-px patch must equal the tile's own cached S64 patch exactly.

Usage:
    conda run --no-capture-output -n geospatial python -u scripts/probes/_w2_fang_embed.py
    ... [--tile-px 32]   # S=32 read: 32-px own tile + 96-px 3x3 context (scale_idx 2)

npz naming encodes the INPUT size only ({obs}_P{px}.npz): P64/P192 are the S=64 tile
inputs, P32/P96 the S=32 ones -- unambiguous because each tile scale only ever gets its
own-tile and 3x3 input sizes.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CKPT = REPO_ROOT / "models/pretrained/mars-mae-dino-vit-base-v1.pth"
DATASET_DIR = REPO_ROOT / "dataset_v2"
OUT_DIR = DATASET_DIR / "fang_embeddings"
NORM = "none"   # set from --norm in main(); "a1" = striping-mitigation CTX normalization
TILE_PX = 64           # set from --tile-px in main(); 64 (scale_idx 3) or 32 (scale_idx 2)
CONTEXT_PX = 192       # always 3 * TILE_PX
SCALE_IDX_BY_TILE = {64: 3, 32: 2}
MODEL_INPUT = 224
BATCH = 96
GEM_P = 3.0


# ============================================================================
# Plain-torch ViT-B/16 (timm key layout)
# ============================================================================


class _Block(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.heads = heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v)
        a = a.transpose(1, 2).reshape(B, N, C)
        x = x + self.proj(a)
        h = self.norm2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class ViTB16(nn.Module):
    """Minimal ViT-B/16 encoder matching the timm state-dict layout of the Fang checkpoint."""

    def __init__(self, img_size: int = 224, patch: int = 16, dim: int = 768, depth: int = 12):
        super().__init__()
        self.patch_embed_proj = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        n_tokens = (img_size // patch) ** 2 + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 224, 224) -> tokens (B, 197, 768) after final norm
        B = x.shape[0]
        t = self.patch_embed_proj(x).flatten(2).transpose(1, 2)  # (B, 196, 768)
        t = torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1)
        t = t + self.pos_embed
        for blk in self.blocks:
            t = blk(t)
        return self.norm(t)

    def load_timm_state_dict(self, sd: dict) -> None:
        """Map timm vit_base_patch16_224 keys onto this module; strict (all 150 tensors)."""
        remap = {
            "cls_token": "cls_token",
            "pos_embed": "pos_embed",
            "patch_embed.proj.weight": "patch_embed_proj.weight",
            "patch_embed.proj.bias": "patch_embed_proj.bias",
            "norm.weight": "norm.weight",
            "norm.bias": "norm.bias",
        }
        for i in range(len(self.blocks)):
            for a, b in (
                (f"blocks.{i}.norm1", f"blocks.{i}.norm1"),
                (f"blocks.{i}.attn.qkv", f"blocks.{i}.qkv"),
                (f"blocks.{i}.attn.proj", f"blocks.{i}.proj"),
                (f"blocks.{i}.norm2", f"blocks.{i}.norm2"),
                (f"blocks.{i}.mlp.fc1", f"blocks.{i}.fc1"),
                (f"blocks.{i}.mlp.fc2", f"blocks.{i}.fc2"),
            ):
                remap[a + ".weight"] = b + ".weight"
                remap[a + ".bias"] = b + ".bias"
        mapped = {remap[k]: v for k, v in sd.items()}  # KeyError = unexpected key
        missing, unexpected = self.load_state_dict(mapped, strict=True)
        assert not missing and not unexpected


# ============================================================================
# Pooling
# ============================================================================


def pool_tokens(tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    """cls / mean / GeM(p=3) pooled embeddings from (B, 197, 768) tokens."""
    cls = tokens[:, 0]
    patch = tokens[:, 1:]
    mean = patch.mean(dim=1)
    gem = patch.clamp(min=1e-6).pow(GEM_P).mean(dim=1).pow(1.0 / GEM_P)
    return {"cls": cls, "mean": mean, "gem": gem}


# ============================================================================
# Per-image extraction
# ============================================================================


def _load_ctx_window(obs_id: str) -> tuple[np.ndarray, int, int]:
    import rasterio

    sidecar = json.loads((DATASET_DIR / "labels" / f"{obs_id}.json").read_text(encoding="utf-8"))
    with rasterio.open(sidecar["ctx_window_tif"]) as src:
        arr = src.read(1).astype(np.uint8, copy=False)
    return arr, int(sidecar["mosaic_row_origin"]), int(sidecar["mosaic_col_origin"])


@torch.no_grad()
def embed_batches(model: ViTB16, patches: np.ndarray, device: torch.device) -> dict[str, np.ndarray]:
    """Forward a (n, P, P) uint8 stack; returns float32 (n, 768) per pooling."""
    outs: dict[str, list[np.ndarray]] = {"cls": [], "mean": [], "gem": []}
    for i in range(0, patches.shape[0], BATCH):
        chunk = torch.from_numpy(patches[i: i + BATCH]).to(device)
        x = chunk.float().unsqueeze(1) / 255.0
        x = (x - 0.5) / 0.5
        x = F.interpolate(x, size=(MODEL_INPUT, MODEL_INPUT), mode="bicubic", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            tokens = model(x)
        for k, v in pool_tokens(tokens.float()).items():
            outs[k].append(v.cpu().numpy().astype(np.float32))
    return {k: (np.concatenate(v, axis=0) if v else np.zeros((0, 768), np.float32))
            for k, v in outs.items()}


_A1_PROV: dict[str, dict] = {}
_A1_TILE_CACHE: dict[str, tuple] = {}


def a1_train_frame_context(obs_id: str, shape):
    """R07: the per-frame native A1 statistic + frame labels for one Stage-2 training window.

    The statistic is the SAME one deployment uses (`src.striping.a1_stats_native_tile`): per
    SeamMap source frame, native 5 m DN, over the frame's extent in the **parent Murray tile**
    rather than only the part inside this window. Streaming a tile is the expensive step, so it
    is cached — the 39 windows share 20 tiles.
    """
    import rasterio

    from src.striping import a1_stats_native_tile, frame_labels_on, load_frames

    # the window's own sidecar sits beside the tif the label sidecar points at
    label_side = json.loads(
        (DATASET_DIR / "labels" / f"{obs_id}.json").read_text(encoding="utf-8"))
    win_tif = Path(label_side["ctx_window_tif"])
    side = json.loads(win_tif.with_suffix(".json").read_text(encoding="utf-8"))
    tile = side["source_murray_tile"]
    if tile not in _A1_TILE_CACHE:
        frames = load_frames(tile)
        print(f"    A1: streaming {tile} for per-frame native statistics ...", flush=True)
        stats, fallback, prov = a1_stats_native_tile(tile, frames)
        print(f"    A1: {tile} {prov['n_frames_with_stats']}/{prov['n_frames']} frames, "
              f"fallback {prov['fallback_pixel_fraction']:.4%} of valid px", flush=True)
        _A1_TILE_CACHE[tile] = (stats, fallback, prov, frames)
    stats, fallback, prov, frames = _A1_TILE_CACHE[tile]
    with rasterio.open(win_tif) as ds:                 # the window's true affine, not a copy
        transform = ds.transform
    labels = frame_labels_on(transform, shape, frames)
    return stats, fallback, labels, dict(prov, murray_tile=tile)


def _a1_own_patches(own, arr_norm, r_win, c_win, fallback):
    """Re-slice each own-tile patch from the NORMALIZED window.

    Normalizing the cached patch separately would give a value that only approximately matches
    the centre of the 192-px box, silently weakening the geometry self-check from exact equality
    to nearly-equal. Tiles whose own box falls outside the cached window have no window pixels
    to re-slice, so they take the tile-wide fallback statistic — never raw DN (R08).
    """
    from src.striping import a1_apply

    H, W = arr_norm.shape
    out = own.copy()
    inside = (r_win >= 0) & (c_win >= 0) & (r_win + TILE_PX <= H) & (c_win + TILE_PX <= W)
    for i in np.where(inside)[0]:
        out[i] = arr_norm[r_win[i]: r_win[i] + TILE_PX, c_win[i]: c_win[i] + TILE_PX]
    outside = np.where(~inside)[0]
    for i in outside:
        out[i] = a1_apply(own[i], fallback[0], fallback[1])
    return out, int(outside.size)


def extract_one(model: ViTB16, obs_id: str, keys: pd.DataFrame, device: torch.device) -> None:
    """Write {obs_id}_P64.npz + {obs_id}_P192.npz, row-parallel to `keys` (S=64 tiles)."""
    from src.modeling.loaders import load_context_patch_stack

    ti = keys["ti"].to_numpy(np.int64)
    tj = keys["tj"].to_numpy(np.int64)
    n = len(keys)

    # ---- P=64: the tile's own cached patch ----
    stack = load_context_patch_stack(obs_id, TILE_PX, dataset_dir=DATASET_DIR)
    pidx = keys[f"patch_idx_S{TILE_PX}"].to_numpy(np.int64)
    assert (pidx >= 0).all(), f"{obs_id}: unexpected -1 patch_idx_S64 rows"
    own64 = np.ascontiguousarray(stack[pidx])

    # ---- P=192: slice 3x3-tile boxes from the cached CTX window ----
    arr, row0, col0 = _load_ctx_window(obs_id)

    H, W = arr.shape
    r_win = ti * TILE_PX - row0
    c_win = tj * TILE_PX - col0

    # A1 striping mitigation. **R07:** this used to take ONE `a1_stats(arr)` for the whole
    # window, on the stated grounds that "each training window is ~one CTX source frame".
    # Measured against the cached SeamMaps, that is false: only 10 of 38 windows lie in a
    # single frame; 22 span two, 3 span three, max four, and the dominant frame's share is a
    # median 81% (min 48%). So training removed between-WINDOW scale while deployment removed
    # between-FRAME scale -- two different normalizations, which is why the A1 payoff number
    # and the A1 skill number were never comparable.
    #
    # Both sides now call one definition (src.striping.A1_ARM): the per-frame NATIVE statistic
    # over the frame's extent in the parent Murray tile. Own patches are re-sliced from the
    # normalized window rather than normalized separately, which is what keeps the geometry
    # self-check below an exact-equality check instead of an approximate one.
    #
    # **R38 rides along, deliberately.** `a1_apply` now floors valid pixels at
    # `A1_VALID_FLOOR = 1` so DN 0 means only nodata, and because BOTH sides call it, the
    # training input changes in the same commit as the deploy input -- which is the whole point
    # (a floor changed on one side only would re-open R07's train/deploy mismatch on a second
    # axis). It also means `dataset_v2/fang_embeddings_a1` and `models/deployable_a1` were baked
    # under the old floor and must be re-made; that is already row 7 of docs/PENDING_REBUILD.md.
    # Measured cost of the floor itself: 0.041 % of training pixels.
    if NORM == "a1":
        from src.striping import a1_normalize_native
        stats, fallback, labels, prov = a1_train_frame_context(obs_id, arr.shape)
        arr = a1_normalize_native(arr, labels, stats, fallback)
        own64, n_outside = _a1_own_patches(own64, arr, r_win, c_win, fallback)
        prov = dict(prov, own_patches_outside_window=n_outside)
        _A1_PROV[obs_id] = prov

    r0 = r_win - TILE_PX
    c0 = c_win - TILE_PX
    valid192 = (r0 >= 0) & (c0 >= 0) & (r0 + CONTEXT_PX <= H) & (c0 + CONTEXT_PX <= W)
    big = np.zeros((int(valid192.sum()), CONTEXT_PX, CONTEXT_PX), dtype=np.uint8)
    vi = np.where(valid192)[0]
    for out_row, i in enumerate(vi):
        big[out_row] = arr[r0[i]: r0[i] + CONTEXT_PX, c0[i]: c0[i] + CONTEXT_PX]

    # Geometry self-check: center 64x64 of the 192-px box == the tile's own cached patch.
    if vi.size:
        probe = vi[:: max(1, vi.size // 16)]
        for out_row, i in [(np.searchsorted(vi, p), p) for p in probe]:
            center = big[out_row, TILE_PX: 2 * TILE_PX, TILE_PX: 2 * TILE_PX]
            assert np.array_equal(center, own64[i]), (
                f"{obs_id}: 192-px center mismatch at row {i} -- grid/origin handling is wrong")

    emb64 = embed_batches(model, own64, device)
    emb192_v = embed_batches(model, big, device)
    emb192 = {k: np.full((n, 768), np.nan, dtype=np.float32) for k in emb192_v}
    for k in emb192:
        emb192[k][vi] = emb192_v[k]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / f"{obs_id}_P{TILE_PX}.npz",
             ti=ti.astype(np.int32), tj=tj.astype(np.int32),
             valid=np.ones(n, dtype=bool), **emb64)
    np.savez(OUT_DIR / f"{obs_id}_P{CONTEXT_PX}.npz",
             ti=ti.astype(np.int32), tj=tj.astype(np.int32),
             valid=valid192, **emb192)
    print(f"  {obs_id}: n={n}  192-valid={int(valid192.sum())} ({valid192.mean():.1%})", flush=True)


def _cache_is_stale(obs_id: str, keys) -> str | None:
    """`None` if the cached store for `obs_id` matches `keys` exactly, else why it doesn't.

    **DECISIONS 2026-08-20k.** The resume used to be `if npz.exists(): skip`, with no comparison
    of any kind. On the v2 rebuild that made the whole of step 6 a **4-second silent no-op**: the
    cached stores held the pre-rebuild pool (161,005 tiles) while the fresh labels held 164,644,
    every one of the 38 images had a different (ti, tj) set, and 7,390 new tiles had no embedding
    at all. Nothing downstream would have noticed — the npz records no provenance whatsoever, so
    a stale store and a fresh one are indistinguishable on disk.

    The key set IS the provenance we have. Comparing it makes the skip safe rather than merely
    bypassable, which is why this is a check and not a `--force` flag (`--force` exists too, but
    it should never be the thing standing between you and a correct rebuild).
    """
    ctx = OUT_DIR / f"{obs_id}_P{CONTEXT_PX}.npz"
    own = OUT_DIR / f"{obs_id}_P{TILE_PX}.npz"
    if not (ctx.is_file() and own.is_file()):
        return "absent"
    try:
        with np.load(ctx) as z:
            cached = set(zip(z["ti"].tolist(), z["tj"].tolist()))
    except Exception as exc:                                   # noqa: BLE001
        return f"cached store is unreadable ({type(exc).__name__})"
    want = set(zip(keys["ti"].tolist(), keys["tj"].tolist()))
    if cached == want:
        return None
    return (f"key set differs: cached {len(cached)} vs labels {len(want)} "
            f"({len(want - cached)} new tiles missing, {len(cached - want)} cached tiles gone)")


def main() -> int:
    from src.modeling.loaders import load_fold

    global TILE_PX, CONTEXT_PX, NORM, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tile-px", type=int, default=64, choices=sorted(SCALE_IDX_BY_TILE))
    ap.add_argument("--norm", choices=["none", "a1"], default="none",
                    help="a1 = per-window robust offset+gain CTX normalization (striping mitigation)")
    ap.add_argument("--out-suffix", default="",
                    help="suffix on the output store dir, e.g. _a1 -> dataset_v2/fang_embeddings_a1")
    ap.add_argument("--force", action="store_true",
                    help="recompute every image even if its cached key set matches")
    args = ap.parse_args()
    TILE_PX = args.tile_px
    CONTEXT_PX = 3 * TILE_PX
    NORM = args.norm
    OUT_DIR = DATASET_DIR / f"fang_embeddings{args.out_suffix}"
    scale_idx = SCALE_IDX_BY_TILE[TILE_PX]
    print(f"norm={NORM}  out_dir={OUT_DIR}")

    t_start = time.monotonic()
    obj = torch.load(CKPT, map_location="cpu", weights_only=False)
    print(f"checkpoint meta: timm_name={obj.get('timm_name')} arch={obj.get('arch')} "
          f"img_size={obj.get('img_size')} in_chans={obj.get('in_chans')} source={obj.get('source')}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTB16().to(device).eval()
    model.load_timm_state_dict(obj["state_dict"])
    print(f"ViT-B/16 loaded strict OK on {device}")

    # Fold 0 train+test covers all 38 images' rows at this scale exactly once.
    fold = load_fold("loio_nfold", 0, scale_idx=scale_idx, dataset_dir=DATASET_DIR)
    keys_all = pd.concat([fold.keys_train, fold.keys_test], ignore_index=True)
    print(f"S={TILE_PX} tiles: {len(keys_all)} across {keys_all['obs_id'].nunique()} images\n")

    n_reused = n_stale = 0
    for obs_id, g in keys_all.groupby("obs_id", sort=True):
        g = g.reset_index(drop=True)
        why = "--force" if args.force else _cache_is_stale(obs_id, g)
        if why is None:
            print(f"  {obs_id}: cached and key-set matches, skipping", flush=True)
            n_reused += 1
            continue
        if why != "absent":
            print(f"  {obs_id}: RECOMPUTING -- {why}", flush=True)
            n_stale += 1
        extract_one(model, obs_id, g, device)

    print(f"\nreused {n_reused} cached / recomputed {len(keys_all.groupby('obs_id')) - n_reused} "
          f"({n_stale} of them because the cached key set was stale)")

    print(f"\ndone in {time.monotonic() - t_start:.0f} s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
