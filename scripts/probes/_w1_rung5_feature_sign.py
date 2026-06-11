"""W1 Rung 5 — is the within-image feature->label relationship INVERTED on
anti-signal images?

For each image at S=64, Spearman correlation of each headline feature with
boulder_count. If anti-signal images carry opposite-sign correlations vs the
cohort majority, the model's LOIO inversion has a concrete mechanism
(terrain/illumination-conditional texture inversion, Serrano's geomorphic-unit
mediation) -- and 'anti-signal' is real signal with a flipped sign, not noise.

Writes _w1_rung5_feature_sign.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = Path("dataset_v2/labels")
FEATURES = Path("dataset_v2/features")
OUT_MD = Path("scripts/probes/_w1_rung5_feature_sign.md")
ANTI = {
    "ESP_076499_1160", "ESP_055978_2270", "ESP_054000_2255", "ESP_046328_2180",
    "ESP_064510_2260", "ESP_047976_2020", "ESP_049242_2115", "ESP_059686_2235",
}
FEATS = ["shadow_fraction", "grad_mag_mean", "glcm_contrast_d1", "glcm_energy_d1",
         "intensity_std", "edge_density", "intensity_mean"]

rows = []
for lf in sorted(LABELS.glob("*.parquet")):
    obs = lf.stem
    lab = pd.read_parquet(lf)
    lab = lab[lab.scale_idx == 3][["ti", "tj", "boulder_count"]]
    feat = pd.read_parquet(FEATURES / f"{obs}.parquet")
    feat = feat[feat.scale_idx == 3]
    m = lab.merge(feat, on=["ti", "tj"], validate="one_to_one")
    r = {"obs_id": obs, "anti": obs in ANTI, "n": len(m)}
    for c in FEATS:
        r[c] = float(m[c].corr(m.boulder_count, method="spearman"))
    rows.append(r)

df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda v: f"{v:+.2f}"))

lines = []
for c in FEATS:
    a = df.loc[df.anti, c]
    h = df.loc[~df.anti, c]
    lines.append(
        f"- `{c}`: cohort-majority sign {'+' if h.median() > 0 else '-'} "
        f"(healthy median {h.median():+.3f}, {len(h)} imgs); anti median {a.median():+.3f}, "
        f"sign-flipped in {(np.sign(a) != np.sign(h.median())).sum()}/8 anti images"
    )
    print(lines[-1])

OUT_MD.write_text(
    "# W1 Rung 5 — within-image feature-label correlation signs\n\n"
    "```\n" + df.to_string(index=False, float_format=lambda v: f"{v:+.2f}") + "\n```\n\n"
    + "\n".join(lines) + "\n",
    encoding="utf-8",
)
print(f"wrote {OUT_MD}")
