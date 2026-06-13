"""Frozen Fang-ViT embeddings — productized extraction + inference path.

Factors the probe-tier `scripts/probes/_w2_fang_embed.py` into `src/`: the
plain-torch ViT-B/16 encoder, the GeM-pooled embedding primitive, and the
CTX-window inference path (embed an arbitrary mosaic window on the tile grid →
768-dim vectors per tile). The cached-store *join* used by the LOIO training
harness lives in `src.modeling.loaders` (numpy-only) so the modeling loader
stays torch-free; this module is the half that needs torch.

Frozen recipe (DECISIONS.md 2026-06-12 "Freeze window CLOSED"): `mlp_ens3` on
the **S=32 96-px 3×3-context GeM(p=3)** emb-only matrix, target `fa_gt_1e-2`.
This module produces the embedding half of that recipe's inference path — given
a CTX window beyond HiRISE coverage (the map pilot), it emits the per-tile
768-dim GeM vectors the head consumes.

The model is the Fang et al. 2026 ([doi:10.1029/2025JH000827]) ViT-B/16,
self-supervised (MAE+DINO) on 3.9M crops of the Murray Lab CTX mosaic — the
identical product our pipeline windows. Checkpoint: Zenodo 18180801,
`models/pretrained/mars-mae-dino-vit-base-v1.pth` (untracked, 341 MB; a
standard timm-layout state dict, in_chans=1, 224 px, no head).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO_ROOT / "models" / "pretrained" / "mars-mae-dino-vit-base-v1.pth"

MODEL_INPUT = 224       # ViT-B/16 expects 224-px input; tile boxes are bicubic-resized
GEM_P = 3.0             # frozen pooling exponent (1d ablation: GeM(3) > mean > cls)
EMBED_DIM = 768
DEFAULT_BATCH = 96
POOLINGS = ("cls", "mean", "gem")


# ============================================================================
# Plain-torch ViT-B/16 (timm key layout) — single source of truth for the encoder
# ============================================================================


def _build_block(dim: int = EMBED_DIM, heads: int = 12, mlp_ratio: float = 4.0):
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(dim, eps=1e-6)
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.norm2 = nn.LayerNorm(dim, eps=1e-6)
            self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
            self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
            self.heads = heads

        def forward(self, x):
            import torch.nn.functional as F

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

    return _Block()


def build_vit_b16(img_size: int = MODEL_INPUT, patch: int = 16,
                  dim: int = EMBED_DIM, depth: int = 12):
    """Construct the (untrained) ViT-B/16 encoder matching the Fang checkpoint layout.

    Returns an `nn.Module` whose `forward((B,1,224,224)) -> (B,197,768)` final-norm
    tokens. Kept as a factory (not a module-level class) so importing this module
    does not require torch until an embedder is actually built.
    """
    import torch
    import torch.nn as nn

    class ViTB16(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed_proj = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
            n_tokens = (img_size // patch) ** 2 + 1
            self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
            self.blocks = nn.ModuleList([_build_block(dim) for _ in range(depth)])
            self.norm = nn.LayerNorm(dim, eps=1e-6)

        def forward(self, x):
            B = x.shape[0]
            t = self.patch_embed_proj(x).flatten(2).transpose(1, 2)  # (B, 196, 768)
            t = torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1)
            t = t + self.pos_embed
            for blk in self.blocks:
                t = blk(t)
            return self.norm(t)

    return ViTB16()


def _timm_remap(depth: int = 12) -> dict[str, str]:
    """timm `vit_base_patch16_224` state-dict keys → this module's attribute names."""
    remap = {
        "cls_token": "cls_token",
        "pos_embed": "pos_embed",
        "patch_embed.proj.weight": "patch_embed_proj.weight",
        "patch_embed.proj.bias": "patch_embed_proj.bias",
        "norm.weight": "norm.weight",
        "norm.bias": "norm.bias",
    }
    for i in range(depth):
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
    return remap


def load_timm_state_dict(model, sd: dict) -> None:
    """Strict-load a timm ViT-B/16 state dict onto a `build_vit_b16()` module.

    KeyError on an unexpected source key; asserts no missing/unexpected after the
    remap (all 150 tensors present), so a layout drift fails loudly rather than
    silently leaving random weights.
    """
    remap = _timm_remap(len(model.blocks))
    mapped = {remap[k]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(mapped, strict=True)
    assert not missing and not unexpected, (missing, unexpected)


# ============================================================================
# Pooling
# ============================================================================


def gem_pool_np(patch_tokens: np.ndarray, p: float = GEM_P) -> np.ndarray:
    """GeM(p) over the patch-token axis of a (n, n_patch, 768) float array.

    Reference implementation (numpy) used by tests and any non-torch caller:
    ``(mean_i clamp(x_i, 1e-6)^p)^(1/p)`` per channel. p=1 → mean, p→∞ → max.
    """
    x = np.clip(patch_tokens, 1e-6, None).astype(np.float64)
    return (np.power(x, p).mean(axis=1) ** (1.0 / p)).astype(np.float32)


def _pool_tokens(tokens, p: float = GEM_P) -> dict:
    """cls / mean / GeM(p) pooled embeddings from (B, 197, 768) torch tokens."""
    cls = tokens[:, 0]
    patch = tokens[:, 1:]
    mean = patch.mean(dim=1)
    gem = patch.clamp(min=1e-6).pow(p).mean(dim=1).pow(1.0 / p)
    return {"cls": cls, "mean": mean, "gem": gem}


# ============================================================================
# Geometry: 3×3-context boxes from a mosaic window on the tile grid
# ============================================================================


def slice_context_boxes(
    window: np.ndarray, ti: np.ndarray, tj: np.ndarray, tile_px: int,
    row0: int, col0: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice the (3·tile_px)² context box for each (ti, tj) tile from a mosaic window.

    The grid is anchored to the **mosaic pixel origin** (Stage 4 convention): tile
    (ti, tj) covers mosaic rows [ti·tile_px, ti·tile_px+tile_px). Its 3×3-context box
    is the surrounding [−tile_px, +2·tile_px) ring, so its center tile_px² equals the
    tile itself. `window` is that mosaic region with its top-left at (row0, col0).

    Returns ``(boxes, valid)``: `boxes` is (n_valid, P, P) uint8 in input order of the
    valid tiles, `valid` is the (n,) bool mask of tiles whose full box fits inside the
    window. Tiles whose context spills past the window edge are dropped (recorded
    invalid) rather than zero-padded — matching the extraction-time convention.
    """
    ti = np.asarray(ti, dtype=np.int64)
    tj = np.asarray(tj, dtype=np.int64)
    context_px = 3 * tile_px
    H, W = window.shape
    r0 = ti * tile_px - row0 - tile_px
    c0 = tj * tile_px - col0 - tile_px
    valid = (r0 >= 0) & (c0 >= 0) & (r0 + context_px <= H) & (c0 + context_px <= W)
    vi = np.where(valid)[0]
    boxes = np.empty((vi.size, context_px, context_px), dtype=np.uint8)
    for out_row, i in enumerate(vi):
        boxes[out_row] = window[r0[i]: r0[i] + context_px, c0[i]: c0[i] + context_px]
    return boxes, valid


def tile_grid_for_window(
    window_shape: tuple[int, int], row0: int, col0: int, tile_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Enumerate (ti, tj) for every tile whose full 3×3-context box fits in the window.

    For the map-pilot inference path: a CTX window with mosaic origin (row0, col0)
    and shape `window_shape` is gridded on the same mosaic-anchored tile_px lattice
    used in training, returning the global (ti, tj) indices of tiles with full context.
    """
    H, W = window_shape
    ti_lo = int(np.ceil((row0 + tile_px) / tile_px))
    ti_hi = int(np.floor((row0 + H - 2 * tile_px) / tile_px))
    tj_lo = int(np.ceil((col0 + tile_px) / tile_px))
    tj_hi = int(np.floor((col0 + W - 2 * tile_px) / tile_px))
    tis = np.arange(ti_lo, ti_hi + 1, dtype=np.int64)
    tjs = np.arange(tj_lo, tj_hi + 1, dtype=np.int64)
    gi, gj = np.meshgrid(tis, tjs, indexing="ij")
    return gi.ravel(), gj.ravel()


# ============================================================================
# Embedder
# ============================================================================


class FangEmbedder:
    """Frozen Fang-ViT wrapped for inference: uint8 patches/windows → 768-dim vectors.

    Build via `FangEmbedder.load(...)` (strict checkpoint load). All forwards run
    under `no_grad` + eval; CUDA autocast(fp16) when on GPU. The path is
    deterministic given the frozen weights — no seed protocol.
    """

    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device

    @classmethod
    def load(cls, ckpt_path: Path | str = DEFAULT_CKPT, device: str | None = None) -> "FangEmbedder":
        import torch

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Fang checkpoint not found at {ckpt_path}. Re-download from Zenodo 18180801 "
                f"(mars-mae-dino-vit-base-v1.pth, 341 MB; untracked).")
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_vit_b16().to(dev).eval()
        load_timm_state_dict(model, obj["state_dict"])
        return cls(model, dev)

    def preprocess(self, patches: np.ndarray):
        """uint8 (n, P, P) → normalized (n, 1, 224, 224) float tensor on the device.

        Per the model card: x/255, (x − 0.5)/0.5, then bicubic-resize to 224.
        """
        import torch
        import torch.nn.functional as F

        x = torch.from_numpy(np.ascontiguousarray(patches)).to(self.device)
        x = x.float().unsqueeze(1) / 255.0
        x = (x - 0.5) / 0.5
        return F.interpolate(x, size=(MODEL_INPUT, MODEL_INPUT), mode="bicubic", align_corners=False)

    def embed_patches(self, patches: np.ndarray, *, pool: str = "gem",
                      batch: int = DEFAULT_BATCH) -> np.ndarray:
        """Embed a (n, P, P) uint8 stack → (n, 768) float32 with the requested pooling.

        `pool` in {"cls","mean","gem"}; the frozen recipe uses "gem". Empty input
        returns a (0, 768) array. Batched to bound GPU memory.
        """
        import torch

        if pool not in POOLINGS:
            raise ValueError(f"unknown pool {pool!r}; pick from {POOLINGS}")
        if patches.shape[0] == 0:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        out: list[np.ndarray] = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, patches.shape[0], batch):
                x = self.preprocess(patches[i: i + batch])
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.type == "cuda"):
                    tokens = self.model(x)
                vec = _pool_tokens(tokens.float())[pool]
                out.append(vec.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)

    def embed_window(
        self, window: np.ndarray, ti: np.ndarray, tj: np.ndarray, *,
        tile_px: int, row0: int, col0: int, pool: str = "gem",
        batch: int = DEFAULT_BATCH,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inference path: embed the 3×3-context box of each (ti, tj) tile in a window.

        Returns ``(emb, valid)``: `emb` is (n, 768) float32 with NaN rows where the
        tile's context box spilled past the window edge (so the matrix stays
        row-parallel to the input keys); `valid` is the (n,) bool mask. The frozen
        recipe embeds the 96-px (3×32) context at tile_px=32.
        """
        boxes, valid = slice_context_boxes(window, ti, tj, tile_px, row0, col0)
        emb_valid = self.embed_patches(boxes, pool=pool, batch=batch)
        emb = np.full((valid.size, EMBED_DIM), np.nan, dtype=np.float32)
        emb[np.where(valid)[0]] = emb_valid
        return emb, valid
