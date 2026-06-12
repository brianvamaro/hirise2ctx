"""Inspect the Fang et al. ViT-B checkpoint: top-level structure + state-dict key families.

Decides whether timm is needed or a plain-torch ViT-B/16 forward can consume it directly.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401  -- Windows DLL bootstrap; must precede numpy

import torch

CKPT = REPO_ROOT / "models/pretrained/mars-mae-dino-vit-base-v1.pth"

obj = torch.load(CKPT, map_location="cpu", weights_only=False)
print("top-level type:", type(obj).__name__)
if isinstance(obj, dict):
    print("top-level keys:", list(obj.keys())[:20])
    sd = None
    for k in ("model", "state_dict", "model_state_dict", "teacher", "encoder"):
        if k in obj and isinstance(obj[k], dict):
            sd = obj[k]
            print(f"-> using obj[{k!r}] as state_dict")
            break
    if sd is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
        sd = obj
        print("-> top-level dict IS the state_dict")
else:
    sd = obj.state_dict() if hasattr(obj, "state_dict") else None

assert sd is not None, "could not locate a state dict"
keys = list(sd.keys())
print(f"\nn tensors: {len(keys)}")
prefixes = {}
for k in keys:
    p = k.split(".")[0]
    prefixes[p] = prefixes.get(p, 0) + 1
print("prefix families:", prefixes)
for k in keys[:12]:
    print(f"  {k:<50s} {tuple(sd[k].shape)}")
print("  ...")
for k in keys[-6:]:
    print(f"  {k:<50s} {tuple(sd[k].shape)}")

# The shapes that pin the architecture
for probe in ("patch_embed.proj.weight", "pos_embed", "cls_token", "norm.weight",
              "fc_norm.weight", "blocks.0.attn.qkv.weight"):
    for cand in (probe, "model." + probe, "module." + probe, "backbone." + probe):
        if cand in sd:
            print(f"{cand}: {tuple(sd[cand].shape)}")
            break
