"""One-off smoke test for src.features.stage4b_one_image on ESP_069669_2220."""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.features import stage4b_one_image, load_features


def main() -> int:
    cfg = load_config("config.yaml")
    obs = "ESP_069669_2220"
    t0 = time.monotonic()
    prov = stage4b_one_image(
        obs,
        cache_dir=cfg.cache_dir,
        output_dir=cfg.output_dir,
        features_cfg=cfg["features"],
        config_hash=cfg.hash,
    )
    total = time.monotonic() - t0
    print(f"total: {total:.1f} s")
    print(f"n_tiles_total: {prov['n_tiles_total']}")
    print(f"per_scale: {prov['per_scale_tile_counts']}")
    print(f"dn_thresholds: {prov['dn_thresholds']}")
    print("timings per image (s):")
    for k, v in prov["timings_per_image_seconds"].items():
        print(f"  {k}: {v:.2f}")
    print("timings per scale (s):")
    for s, t in prov["timings_per_scale_seconds"].items():
        joined = "  ".join(f"{k}={v:.2f}" for k, v in t.items())
        print(f"  S={s}: {joined}")
    if prov["context_patch"]["enabled"]:
        print(f"patch_counts: {prov['context_patch']['patch_counts']}")
        print(f"patch_bytes: {prov['context_patch']['patch_bytes_estimate']}")
    df = load_features(obs, cfg.output_dir)
    print(f"\nfeatures_df shape: {df.shape}")
    print(f"columns ({len(df.columns)}): {list(df.columns)}")
    print(df.head(3).T)
    # Verify alignment with labels parquet.
    from src.labeling import load_labels
    labels = load_labels(obs, cfg.output_dir)
    print(f"\nlabels rows: {len(labels)}; features rows: {len(df)}")
    join = labels.merge(df, on=["scale_idx", "tile_size_px", "ti", "tj"], how="inner",
                        suffixes=("_lbl", "_feat"))
    print(f"join rows: {len(join)} (must equal both)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
