"""Cohort-wide CTX-source sub-solar azimuth spread (Brian question, 2026-06-11).

Reads the Stage 6b ctx_illum per-tile features and reports each image's
S=64 mean/std of ctx_subsolar_az_mean + ctx_incidence_mean, to answer:
how consistent is the illumination direction the CNN patches see?
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import src.modeling  # noqa: F401

import numpy as np
import pandas as pd

rows = []
for p in sorted((REPO_ROOT / "dataset_v2/features_ctx_illum").glob("*.parquet")):
    df = pd.read_parquet(p, columns=["scale_idx", "ctx_subsolar_az_mean",
                                     "ctx_incidence_mean"])
    df = df[df.scale_idx == 3]
    rows.append({
        "obs_id": p.stem,
        "az_mean": float(df.ctx_subsolar_az_mean.mean()),
        "az_std_within": float(df.ctx_subsolar_az_mean.std()),
        "inc_mean": float(df.ctx_incidence_mean.mean()),
    })
t = pd.DataFrame(rows).sort_values("az_mean")
print(t.to_string(index=False, float_format=lambda v: f"{v:8.1f}"))
print(f"\ncohort az_mean: min {t.az_mean.min():.1f}  max {t.az_mean.max():.1f}  "
      f"std {t.az_mean.std():.1f} deg")
print(f"cohort inc_mean: min {t.inc_mean.min():.1f}  max {t.inc_mean.max():.1f} deg")
