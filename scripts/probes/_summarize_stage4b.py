"""Summarize Stage 4b sweep outputs for the DECISIONS.md write-up."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import manifest as M
from src.config import load_config
from src.features import FEATURES_SUBDIR, load_features, load_features_provenance


def main() -> int:
    cfg = load_config(REPO_ROOT / "config.yaml")
    manifest = M.load_manifest(cfg.manifest_path)
    obs_ids = [str(obs) for obs in manifest["ObsId"]
               if (cfg.output_dir / FEATURES_SUBDIR / f"{obs}.parquet").exists()]
    print(f"{len(obs_ids)} ObsIds with Stage 4b outputs")

    rows = []
    total_glcm = 0.0
    patch_count_32 = 0
    patch_count_64 = 0
    patch_bytes_32 = 0
    patch_bytes_64 = 0
    for obs in obs_ids:
        prov = load_features_provenance(obs, cfg.output_dir)
        df = load_features(obs, cfg.output_dir)
        scale_t = prov["timings_per_scale_seconds"]
        glcm_total = sum(scale_t.get(s, {}).get("glcm", 0.0) for s in scale_t)
        total_glcm += glcm_total
        cp = prov.get("context_patch", {})
        if cp.get("enabled"):
            patch_count_32 += cp["patch_counts"].get(32, 0)
            patch_count_64 += cp["patch_counts"].get(64, 0)
            patch_bytes_32 += cp["patch_bytes_estimate"].get(32, 0)
            patch_bytes_64 += cp["patch_bytes_estimate"].get(64, 0)
        rows.append({
            "obs_id": obs,
            "label": str(manifest.set_index("ObsId").loc[obs, "BoulderLabel"]),
            "n_tiles_total": int(prov["n_tiles_total"]),
            "dn_mode": int(prov["dn_thresholds"]["mode"]),
            "dn_shadow": int(prov["dn_thresholds"]["shadow"]),
            "dn_bright": int(prov["dn_thresholds"]["bright"]),
            "glcm_total_s": round(glcm_total, 1),
            "shadow_frac_mean_S8_pct": round(
                float(df[df["tile_size_px"] == 8]["shadow_fraction"].mean() * 100), 2,
            ),
            "bright_frac_mean_S8_pct": round(
                float(df[df["tile_size_px"] == 8]["bright_cap_fraction"].mean() * 100), 2,
            ),
            "glcm_contrast_d1_mean_S8": round(
                float(df[df["tile_size_px"] == 8]["glcm_contrast_d1"].mean()), 3,
            ),
            "intensity_mean_S8": round(
                float(df[df["tile_size_px"] == 8]["intensity_mean"].mean()), 1,
            ),
        })

    summary = pd.DataFrame(rows).set_index("obs_id")
    print("\nPer-image Stage 4b sweep summary:")
    print(summary.to_string())

    print(f"\nTotal Stage 4b feature rows across the sweep: {sum(r['n_tiles_total'] for r in rows):,}")
    print(f"Total GLCM compute time: {total_glcm:.1f} s")
    print(f"Context patches S=32: {patch_count_32:,} patches, {patch_bytes_32/1e9:.2f} GB")
    print(f"Context patches S=64: {patch_count_64:,} patches, {patch_bytes_64/1e9:.2f} GB")
    print(f"Total patch disk: {(patch_bytes_32 + patch_bytes_64)/1e9:.2f} GB")

    # Feature->target correlation: top + bottom 8 features ranked by Spearman.
    all_parts = []
    sample = load_features(obs_ids[0], cfg.output_dir)
    exclude = {"obs_id", "scale_idx", "tile_size_px", "ti", "tj", "config_hash",
               "patch_idx_S32", "patch_idx_S64"}
    feature_cols = [c for c in sample.columns if c not in exclude]
    for obs in obs_ids:
        feats = load_features(obs, cfg.output_dir)
        from src.labeling import load_labels
        labels = load_labels(obs, cfg.output_dir)
        df = labels[["obs_id", "scale_idx", "tile_size_px", "ti", "tj", "fractional_area"]].merge(
            feats[["obs_id", "scale_idx", "tile_size_px", "ti", "tj"] + feature_cols],
            on=["obs_id", "scale_idx", "tile_size_px", "ti", "tj"],
        )
        all_parts.append(df[df["tile_size_px"] == df["tile_size_px"].min()])
    big = pd.concat(all_parts, ignore_index=True)
    finite_cols = [c for c in feature_cols if big[c].dropna().shape[0] > 100]
    corr = big[finite_cols + ["fractional_area"]].corr(method="spearman")
    target_corr = corr["fractional_area"].drop("fractional_area").sort_values(ascending=False)
    print("\nTop 8 positive Spearman correlations with fractional_area (finest tiles):")
    print(target_corr.head(8).round(4).to_string())
    print("\nTop 8 negative Spearman correlations:")
    print(target_corr.tail(8).round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
