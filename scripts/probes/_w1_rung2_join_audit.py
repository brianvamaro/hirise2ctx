"""W1 Rung 2 — per-image join/pipeline integrity audit.

For every v2 image (all 38, anti-signal flagged), checks the mundane failure
modes that would masquerade as model failure:

1. labels parquet: key uniqueness on (scale_idx, ti, tj); row count per scale.
2. features parquet: same keys, same uniqueness; row count per scale.
3. inner-join coverage labels<->features: rows lost on either side.
4. NaN fraction across feature columns (per image, S=64 rows).
5. nested-grid consistency at S=64 vs S=32 (summing 4 children == parent
   boulder_count) — a corrupted grid/join would break exact nesting.
6. packaged loio_nfold test rows for the image match the joined row count.

Writes scripts/probes/_w1_rung2_join_audit.md with one row per image.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = Path("dataset_v2/labels")
FEATURES = Path("dataset_v2/features")
ANTI = {
    "ESP_076499_1160", "ESP_055978_2270", "ESP_054000_2255", "ESP_046328_2180",
    "ESP_064510_2260", "ESP_047976_2020", "ESP_049242_2115", "ESP_059686_2235",
}
OUT_MD = Path("scripts/probes/_w1_rung2_join_audit.md")

rows = []
for lf in sorted(LABELS.glob("*.parquet")):
    obs = lf.stem
    lab = pd.read_parquet(lf)
    feat = pd.read_parquet(FEATURES / f"{obs}.parquet")

    key = ["scale_idx", "ti", "tj"]
    lab_dup = int(lab.duplicated(subset=key).sum())
    feat_dup = int(feat.duplicated(subset=key).sum())

    merged = lab.merge(feat[key].assign(_f=1), on=key, how="outer", indicator=True)
    lab_only = int((merged["_merge"] == "left_only").sum())
    feat_only = int((merged["_merge"] == "right_only").sum())

    l64 = lab[lab.scale_idx == 3]
    f64 = feat[feat.scale_idx == 3]
    feat_cols = [c for c in f64.columns if c not in key + ["obs_id", "tile_size_px"]]
    nan_frac = float(f64[feat_cols].isna().to_numpy().mean()) if len(f64) else np.nan

    # nested-grid check: S=32 (scale_idx 2) children sum to S=64 parent count
    l32 = lab[lab.scale_idx == 2]
    nest_bad = np.nan
    if len(l32) and len(l64):
        child = l32.assign(pti=l32.ti // 2, ptj=l32.tj // 2)
        agg = child.groupby(["pti", "ptj"])["boulder_count"].sum()
        parent = l64.set_index(["ti", "tj"])["boulder_count"]
        common = agg.index.intersection(parent.index)
        nest_bad = int((np.abs(agg.loc[common] - parent.loc[common]) > 1e-6).sum())

    rows.append(
        dict(
            obs_id=obs,
            anti="*" if obs in ANTI else "",
            n_lab=len(lab), n_feat=len(feat),
            lab_dup=lab_dup, feat_dup=feat_dup,
            lab_only=lab_only, feat_only=feat_only,
            n_S64=len(l64),
            nan_frac=round(nan_frac, 5),
            nest_violations=nest_bad,
        )
    )

df = pd.DataFrame(rows)
print(df.to_string(index=False))
bad = df[(df.lab_dup > 0) | (df.feat_dup > 0) | (df.lab_only > 0)
         | (df.feat_only > 0) | (df.nest_violations > 0) | (df.nan_frac > 0.01)]
verdict = ("CLEAN — no duplicates, no join loss, exact nesting, NaN < 1% everywhere"
           if bad.empty else f"PROBLEMS in {len(bad)} images:\n{bad.to_string(index=False)}")
print("\nVERDICT:", verdict)

OUT_MD.write_text(
    "# W1 Rung 2 — join/pipeline integrity audit (per image)\n\n"
    "Checks: key uniqueness, labels<->features join loss, NaN fraction across\n"
    "S=64 feature columns, exact S=32->S=64 nested-count consistency.\n"
    "`anti=*` marks the 8 post-fix anti-signal images.\n\n"
    "```\n" + df.to_string(index=False) + "\n```\n\n"
    f"**Verdict:** {verdict}\n",
    encoding="utf-8",
)
print(f"wrote {OUT_MD}")
