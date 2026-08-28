"""Build notebooks/24_regional_map.ipynb from Python source.

RE-EXECUTED 2026-08-28 on the PROMOTED baseline map (`reports/map_region`), discharging
    the 2026-08-27 staleness flag. The previous execution (2026-06-19) came from the map
    step 12 archived as `reports/map_region_g1`, so every stored output and every
    reports/figures/24_*.png has been replaced. Read the notebook's first cell for what the
    outputs now describe -- in particular that the striping artifact SHIPS UNMITIGATED
    (A1 demoted 2026-08-25).

§2 IS NOW READ-ONLY ON THE MOSAICS (rewired 2026-08-28). It used to call
    `mosaic_geotiffs(..., out_path=MAP_DIR/"regional_{abundance,prob}_mosaic.tif")` -- the
    exact files `scripts/map_mosaics.py` produces WITH provenance tags (SIZE_FLOOR_*,
    MOSAIC_*) and a closed-footprint gate, so executing this notebook would have silently
    replaced the tagged product with an untagged look-alike (notebooks are NOT covered by
    the test-side write guard, CLAUDE.md). It now calls
    `src.mapping.load_regional_mosaic`, which reads the tagged mosaic and only merges in
    memory -- never writing -- if the file is genuinely absent. Keep it that way:
    `scripts/map_mosaics.py` is the sole producer.

    Re-executing is worth doing, and not only cosmetically: THEMIS has been re-fetched
    onto the corrected lattice (`assert_coregistered` dx=dy=0), so leg 1 -- which
    index-compares -- would finally run co-registered. That is the first of
    PLAN_RegionalMap's unblocked thermal legs. See DECISIONS 2026-08-25k/2026-08-27.

Documentation + analysis home for [PLAN_RegionalMap.md](../PLAN_RegionalMap.md): the
regional rock-abundance map over the circum-Chryse highland-lowland boundary and its
validation against Rodriguez et al. 2016 + THEMIS/TES thermal inertia.

§1 draws the 26-Murray-tile regional map (what `scripts/map_region.py --all` covers) with the
cohort HiRISE footprints colour-coded rich/poor; §1a records how the box was chosen (the first
run mapped 7 eastern tiles, then it was widened to 26); §1b is the MOLA shaded-relief context;
§2 stitches the returned GeoTIFFs into the abundance/probability/binary mosaics; §3 is the 5
validation legs (filled as thermal data lands).

Figures: reports/figures/24_region_{extent,coverage_planning,context_mola,mosaic,products,abund_vs_ctx}.png.
To regenerate: `python notebooks/_build_24.py` then nbconvert --execute --inplace.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent / "24_regional_map.ipynb"


def md(text, cid):
    return {"cell_type": "markdown", "id": cid, "metadata": {},
            "source": text.splitlines(keepends=True)}


def code(text, cid):
    return {"cell_type": "code", "id": cid, "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


# Provenance banner. Its predecessor was a STALE warning: the notebook had last executed
# 2026-06-19 against the map now archived as `reports/map_region_g1`, while its code read the
# promoted `reports/map_region`. Both halves are now discharged -- §2 reads the step-12 mosaics
# instead of rebuilding them, and the notebook has been re-executed on the promoted product --
# so this states what the outputs ARE rather than warning about what they were.
BANNER = '<div style="border-left:6px solid #2e7d32;background:#f2fbf3;padding:10px 14px">\n\n## Provenance — re-executed 2026-08-28 on the PROMOTED baseline map\n\nEvery figure and number below was produced from **`reports/map_region`**, the deliverable regional product promoted by PLAN_Rebuild step 12 (R01 global lattice, v2 labels at rich prevalence 0.373272, the frozen `mlp_ens3` head, the banked calibrator). It supersedes this notebook\'s previous execution (2026-06-19), which came from the map now **archived** as `reports/map_region_g1` — a different lattice, different labels, an older head. Old-vs-new is largely *the same field, moved*, over a small genuine re-levelling; the like-for-like accounting is in **[notebook 29](29_map_comparison.ipynb)**.\n\n**Two things to carry into every read below.**\n\n1. **The CTX source-frame striping artifact ships UNMITIGATED** (ruled 2026-08-25, DECISIONS 2026-08-25k). A1 was demoted from shipped mitigation to a *sensitivity arm*, so the map here carries frame-shaped structure — window-median η² **0.1444**, ratio **1.599** over its own rotation null. Treat any abundance reading in low-contrast terrain accordingly. §2d is the qualitative view of it; notebook 25 is the diagnosis.\n2. **§2 reads the mosaics, it does not build them.** `scripts/map_mosaics.py` is the sole producer of `regional_{layer}_mosaic.tif` and the only thing that stamps their `SIZE_FLOOR_*` / `MOSAIC_*` provenance and gates the footprint. Notebooks are **not** covered by the test-side write guard, so a cell that re-merged and wrote here would silently replace the shipped product with an untagged look-alike.\n\n**Leg 1 is now co-registered.** THEMIS was re-fetched onto the corrected lattice (`assert_coregistered` dx=dy=0), so the §3.1 index-comparison finally correlates the same ground rather than displaced cells — the first of PLAN_RegionalMap\'s unblocked thermal legs.\n\n</div>'

cells = []

cells.append(md(BANNER, "provenance-banner-24"))

cells.append(md(
    """# 24 — Regional rock-abundance map (circum-Chryse)

Documentation + analysis for [PLAN_RegionalMap.md](../PLAN_RegionalMap.md): the first
real **regional deployment** of the frozen head + Stage-1 `CalibrationLayer`, tested
against the boulder-rich Late-Hesperian tsunami deposits of
[Rodriguez et al. 2016, *Sci. Rep.* 6:25106](https://doi.org/10.1038/srep25106) and
independent **THEMIS / TES thermal inertia**.

**This notebook, §1:** *what region are we mapping?* It draws the **26-tile circum-Chryse
regional map** that `scripts/map_region.py --all` runs on Sherlock (the box
`lon[-10,10] lat[32,46]` snapped to whole 4° Murray tiles, plus the 2 original NE tiles
`E12_N44`, `E16_N44`) over the circum-Chryse / NW Arabia Terra highland-lowland boundary —
with the cohort HiRISE footprints that train/anchor the model overlaid. The key realisation
(PLAN §0): boulder-rich cohort images sit **on this boundary**, and the paper's Fig-2C image
**ESP_017355_2260** *is* a cohort image, so the region is in-distribution where we lead and
OOD in the distal plains / southern highlands where we stress-test. §1a records how the box
was chosen (the first run mapped only the 7 eastern tiles; the map was then widened).

Later sections: the calibrated abundance mosaic (§2), then the 5 validation legs (§3) —
abundance↔thermal-inertia correlation, the shoreline-distance profile, the cohort
truth-anchor, and the OOD-honesty panel.
""", "intro"))

cells.append(code(
    """import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists())
sys.path.insert(0, str(REPO))
FIG = REPO / "reports" / "figures"; FIG.mkdir(parents=True, exist_ok=True)

# The 26-tile circum-Chryse regional map (must match scripts/map_region.BLOCK_TILES):
# box lon[-10,10] lat[32,46] snapped to whole tiles (24) + the 2 original NE tiles.
BLOCK_TILES = [
    "E-12_N32", "E-12_N36", "E-12_N40", "E-12_N44", "E-8_N32", "E-8_N36", "E-8_N40", "E-8_N44",
    "E-4_N32", "E-4_N36", "E-4_N40", "E-4_N44", "E0_N32", "E0_N36", "E0_N40", "E0_N44",
    "E4_N32", "E4_N36", "E4_N40", "E4_N44", "E8_N32", "E8_N36", "E8_N40", "E8_N44",
    "E12_N44", "E16_N44",
]
TILE_DEG = 4  # Murray Lab tiles are 4 deg x 4 deg

def tile_to_box(name):
    \"\"\"Murray tile id 'E16_N44' / 'W008_N32' -> (lon0, lat0) lower-left corner in deg.\"\"\"
    e, n = name.split("_")
    lon = int(e[1:]) * (1 if e[0] == "E" else -1)
    lat = int(n[1:]) * (1 if n[0] == "N" else -1)
    return lon, lat

boxes = {t: tile_to_box(t) for t in BLOCK_TILES}
lons = [lo for lo, _ in boxes.values()]; lats = [la for _, la in boxes.values()]
block_extent = (min(lons), max(lons) + TILE_DEG, min(lats), max(lats) + TILE_DEG)
print("map tiles (%d):" % len(BLOCK_TILES), " ".join(BLOCK_TILES))
print("map extent  lon[%g, %g]  lat[%g, %g] deg" % block_extent)
print("map area ~ %d tiles x (4 deg)^2; full-res S=32 ~ %.0fM tiles"
      % (len(BLOCK_TILES), len(BLOCK_TILES) * 2.2))""",
    "setup"))

cells.append(code(
    """# Cohort HiRISE footprints (centres) + rich/poor label.
cohort = pd.read_csv(REPO / "hirise_40_vclaire.csv")
cohort = cohort[["ObsId", "BoulderLabel", "CenterLat", "CenterLon_180"]].copy()
COL = {"Boulder rich": "#d62728", "Boulder poor": "#1f77b4", "unknown": "0.55"}
cohort["color"] = cohort.BoulderLabel.map(COL).fillna("0.55")

lo0, lo1, la0, la1 = block_extent
in_block = cohort[(cohort.CenterLon_180.between(lo0, lo1)) & (cohort.CenterLat.between(la0, la1))]
print("cohort total:", len(cohort))
print("cohort within the 26-tile map:", len(in_block))
print(in_block.BoulderLabel.value_counts().to_string())
PAPER_IMG = "ESP_017355_2260"  # Rodriguez+2016 Fig 2C; in our cohort

# Real HiRISE footprint boxes (the cached Stage-2 CTX-window bounds = footprint bbox + 1km buffer,
# in the pipeline target_crs). That CRS and the Murray map CRS are both equirectangular on
# R=3396190, so the same metres->degrees factor (dpm below) places them on the map. Used to CHECK
# that abundance structure does NOT coincide with HiRISE coverage (inference is pure CTX -- HiRISE
# never enters it -- so any coincidence would be a red flag).
import json as _json
_RDEG = 180.0 / (np.pi * 3396190.0)   # clon_0 / IAU-eqc metres -> degrees
footprints = {}   # ObsId -> (lon0, lat0, lon1, lat1) in degrees
for _, r in cohort.iterrows():
    j = REPO / "cache_v2" / "ctx_windows" / f"{r.ObsId}.json"
    if j.exists():
        b = _json.loads(j.read_text()).get("actual_bounds_target_crs")
        if b:
            footprints[r.ObsId] = (b[0] * _RDEG, b[1] * _RDEG, b[2] * _RDEG, b[3] * _RDEG)
print("HiRISE footprint boxes loaded:", len(footprints))
print("paper Fig-2C image in cohort:", PAPER_IMG in set(cohort.ObsId))""",
    "cohort"))

cells.append(code(
    """fig, ax = plt.subplots(figsize=(12, 6.4))

# 1) the 26 Murray tiles = the regional map.
for t, (lo, la) in boxes.items():
    ax.add_patch(Rectangle((lo, la), TILE_DEG, TILE_DEG, facecolor="#fff2cc",
                           edgecolor="#b8860b", lw=1.4, zorder=1))
    ax.text(lo + TILE_DEG / 2, la + TILE_DEG / 2, t, ha="center", va="center",
            fontsize=8, color="#6b5300", zorder=2)

# 2) cohort footprints WITHIN the map, coloured by label (count only what's drawn).
for lab, c in COL.items():
    sub = in_block[in_block.BoulderLabel == lab]
    if len(sub):
        ax.scatter(sub.CenterLon_180, sub.CenterLat, s=46, c=c, edgecolor="k",
                   lw=0.4, label=f"{lab} (n={len(sub)} in map)", zorder=4)

# 3) the paper's Fig-2C image — the parity / truth-anchor site.
pi = cohort[cohort.ObsId == PAPER_IMG]
if len(pi):
    ax.scatter(pi.CenterLon_180, pi.CenterLat, s=320, marker="*",
               facecolor="gold", edgecolor="k", lw=0.8, zorder=5,
               label=f"{PAPER_IMG}\\n(Rodriguez+2016 Fig 2C)")

ax.set_xlim(lo0 - 2, lo1 + 2); ax.set_ylim(la0 - 2, la1 + 2)
ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)")
ax.set_title("Regional map — 26 Murray CTX tiles over the circum-Chryse\\n"
             "highland-lowland boundary (lHl1 tsunami-deposit zone), with cohort footprints")
ax.grid(True, ls=":", alpha=0.4); ax.set_aspect("equal")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
ax.annotate("southern highlands\\n(specificity check)", (lo0 + 1.0, la0 + 0.5),
            fontsize=8, style="italic", color="0.4")
ax.annotate(f"{len(in_block)} of {len(cohort)} cohort images fall in this map",
            (lo1 + 1.8, la0 - 1.2), ha="right", fontsize=8, color="0.3")
fig.tight_layout()
out = FIG / "24_region_extent.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out.relative_to(REPO)); plt.show()""",
    "fig_extent"))

cells.append(md(
    """## 1a. Coverage planning — the 26-tile expansion (PLAN §10 decision #5)

The 39-image cohort spans **lon −54→+22°E, lat −64→+52°N** and its centers touch **20** distinct
4° Murray tiles — but the *first* regional run mapped only **7** (the eastern block). To give the
validation real reach the map was widened to a box **lon[-10,10] lat[32,46]**, snapped to whole
tiles (→ lon[-12,12] lat[32,48] = 24 tiles) **plus** the 2 original NE tiles (`E12_N44`,
`E16_N44`) = **26 tiles**, all now mapped. This panel records the choice: orange = the **19
expansion** tiles (added this round), green = the **7 original**. The box deliberately reaches
south into the highlands (lat 32–40, above the −3795 m boundary) as a specificity check — terrain
the model should read *poor*. Wide MOLA basemap is fetched once by `scripts/fetch_wide_basemap.py`
(→ `cache_v2/validation/mola_dem_wide.tif`).
""", "plan_md"))

cells.append(code(
    """import math, rasterio
from src.validation_retrieve import hillshade

WIDE = REPO / "cache_v2" / "validation" / "mola_dem_wide.tif"
BOX = dict(lonmin=-10.0, lonmax=10.0, latmin=32.0, latmax=46.0)
MAP_TILES = [(lo, la) for lo in range(-12, 12, 4) for la in range(32, 48, 4)] + [(12, 44), (16, 44)]
CURRENT = {(0, 40), (4, 40), (4, 44), (8, 40), (8, 44), (12, 44), (16, 44)}
counts = {(int(np.floor(r.CenterLon_180 / 4) * 4), int(np.floor(r.CenterLat / 4) * 4)): 0
          for _, r in cohort.iterrows()}
for _, r in cohort.iterrows():
    counts[(int(np.floor(r.CenterLon_180 / 4) * 4), int(np.floor(r.CenterLat / 4) * 4))] += 1

if not WIDE.exists():
    print("No wide MOLA basemap yet:", WIDE.relative_to(REPO))
    print("Make it once:  python scripts/fetch_wide_basemap.py  (downloads the 2 GB global MOLA, caches a coarse reprojection)")
else:
    with rasterio.open(WIDE) as ds:
        dem = ds.read(1); T = ds.transform; res_m = abs(T.a)
    R = 3396190.0; dpm = 180.0 / (math.pi * R)
    hh, ww = dem.shape
    we = [T.c * dpm, (T.c + ww * T.a) * dpm, (T.f + hh * T.e) * dpm, T.f * dpm]
    Lo = np.linspace(we[0], we[1], ww); La = np.linspace(we[3], we[2], hh)
    Log, Lag = np.meshgrid(Lo, La)
    hs = hillshade(dem, res_m=res_m, azimuth_deg=315, altitude_deg=45)

    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    ax.imshow(hs, cmap="gray", extent=we, origin="upper", aspect="equal", zorder=0)
    ax.imshow(np.ma.masked_invalid(dem), cmap="terrain", extent=we, origin="upper",
              aspect="equal", alpha=0.38, zorder=1)
    cs = ax.contour(Log, Lag, dem, levels=[-3795], colors="k", linewidths=1.1, zorder=2)
    ax.clabel(cs, fmt={-3795: "-3795 m"}, fontsize=7)
    for (lo, la) in MAP_TILES:
        new = (lo, la) not in CURRENT
        ax.add_patch(Rectangle((lo, la), TILE_DEG, TILE_DEG,
                               facecolor=("#ff7f0e" if new else "#2ca02c"),
                               edgecolor=("#cc5500" if new else "#1a6b1a"),
                               lw=1.2, alpha=0.42, zorder=3))
        n = counts.get((lo, la), 0)
        if n:
            ax.text(lo + 3.3, la + 3.3, f"{n}", ha="center", fontsize=8, fontweight="bold",
                    color="#063", zorder=5)
    ax.add_patch(Rectangle((BOX["lonmin"], BOX["latmin"]), BOX["lonmax"] - BOX["lonmin"],
                           BOX["latmax"] - BOX["latmin"], fill=False, edgecolor="red",
                           lw=2.0, ls="--", zorder=6))
    for lab, c in {"Boulder rich": "#d62728", "unknown": "0.45"}.items():
        sub = cohort[(cohort.BoulderLabel == lab) & cohort.CenterLon_180.between(we[0], we[1])
                     & cohort.CenterLat.between(we[2], we[3])]
        ax.scatter(sub.CenterLon_180, sub.CenterLat, s=28, c=c, edgecolor="k", lw=0.3,
                   zorder=7, label=f"{lab} (n={len(sub)})")
    from matplotlib.patches import Patch
    h2, _ = ax.get_legend_handles_labels()
    ax.legend(handles=[Patch(fc="#ff7f0e", alpha=0.42, ec="#cc5500", label="expansion (19)"),
                       Patch(fc="#2ca02c", alpha=0.42, ec="#1a6b1a", label="original (7)")] + h2,
              loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    ax.set_xlim(-22, 24); ax.set_ylim(26, 52)
    ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)")
    ax.set_title("Coverage planning — 26-tile circum-Chryse map  (red dashed = box lon[-10,10] lat[32,46])\\n"
                 "orange = 19 expansion tiles;  green = 7 original;  number = cohort imgs in tile")
    fig.tight_layout()
    out = FIG / "24_coverage_planning.png"; fig.savefig(out, dpi=145, bbox_inches="tight")
    print("wrote", out.relative_to(REPO)); plt.show()""",
    "fig_plan"))

cells.append(md(
    """## 1b. Regional context — MOLA shaded-relief underlay (PLAN §5 fig 1)

The independent geological frame for the map: **MOLA MEGDM topography** (463 m/px, USGS),
fetched + reprojected onto the CTX `clon_0` CRS by `src/validation_retrieve.py`
(`python scripts/fetch_validation_data.py --product mola_dem`) so it co-registers with the abundance
mosaic. Shaded relief underlay + the paper's paleoshoreline elevations as context contours
(**−3795 m lHl1**, **−4100 m lHl2**; [Rodriguez et al. 2016](https://doi.org/10.1038/srep25106)).
The −3795 m contour separates the bouldery highland–lowland run-up zone (PLAN §1 prediction 1)
from the distal Chryse plains; the boulder-rich cohort sites should cluster on the highland
side of it. *(Sanity: the eastern boundary block's median elevation came back −3794 m — right on
the lHl1 shoreline — and the contour threads through the whole map, with the southern tiles
(lat 32–40) climbing above it into the highlands and the NW lowering toward the lowland plains.)*
""", "ctx_md"))

cells.append(code(
    """import math
import rasterio
from src.validation_retrieve import hillshade

MOLA = REPO / "cache_v2" / "validation" / "mola_dem_region.tif"
if not MOLA.exists():
    print("No MOLA raster yet:", MOLA.relative_to(REPO))
    print("Fetch it:  python scripts/fetch_validation_data.py --product mola_dem")
else:
    with rasterio.open(MOLA) as ds:
        dem = ds.read(1)
        T = ds.transform
        res_m = abs(T.a)
    R = 3396190.0; dpm = 180.0 / (math.pi * R)           # clon_0 metres -> degrees
    h, w = dem.shape
    lon0, lat1 = T.c * dpm, T.f * dpm                     # top-left
    lon1, lat0 = (T.c + w * T.a) * dpm, (T.f + h * T.e) * dpm
    mext = [lon0, lon1, lat0, lat1]
    lon = np.linspace(lon0, lon1, w); lat = np.linspace(lat1, lat0, h)
    LON, LAT = np.meshgrid(lon, lat)
    hs = hillshade(dem, res_m=res_m, azimuth_deg=315, altitude_deg=45)

    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.imshow(hs, cmap="gray", extent=mext, origin="upper", aspect="equal", zorder=0)
    im = ax.imshow(np.ma.masked_invalid(dem), cmap="terrain", extent=mext, origin="upper",
                   aspect="equal", alpha=0.45, zorder=1)
    cs = ax.contour(LON, LAT, dem, levels=[-4100, -3795], colors=["#2b2b2b", "k"],
                    linewidths=[0.9, 1.6], linestyles=["--", "-"], zorder=3)
    ax.clabel(cs, fmt={-4100: "-4100 m (lHl2)", -3795: "-3795 m (lHl1)"}, fontsize=7)
    for t, (lo, la) in boxes.items():                    # the 7 inference tiles
        ax.add_patch(Rectangle((lo, la), TILE_DEG, TILE_DEG, fill=False,
                               edgecolor="#b8860b", lw=0.8, alpha=0.7, zorder=2))
    for lab, c in COL.items():                           # cohort footprints
        sub = in_block[in_block.BoulderLabel == lab]
        if len(sub):
            ax.scatter(sub.CenterLon_180, sub.CenterLat, s=40, c=c, edgecolor="k", lw=0.4,
                       label=f"{lab} (n={len(sub)})", zorder=5)
    pi = cohort[cohort.ObsId == PAPER_IMG]
    if len(pi):
        ax.scatter(pi.CenterLon_180, pi.CenterLat, s=260, marker="*", facecolor="gold",
                   edgecolor="k", lw=0.7, zorder=6, label=f"{PAPER_IMG} (Rodriguez+2016)")
    ax.set_xlim(mext[0], mext[1]); ax.set_ylim(mext[2], mext[3])
    ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)")
    ax.set_title("Regional context — MOLA shaded relief + paleoshoreline contours\\n"
                 "circum-Chryse highland–lowland boundary (cohort footprints overlaid)")
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, label="elevation (m, MOLA)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.13, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIG / "24_region_context_mola.png"; fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out.relative_to(REPO)); plt.show()""",
    "fig_ctx"))

cells.append(md(
    """## 2. Stitched regional abundance mosaic

`scripts/map_region.py --all` writes per Murray tile `<tile>_{prob,abundance,prob_raw}.tif`
(160 m/px) to `reports/map_region/`. All 26 share the Murray `clon_0` equirectangular CRS, so
they merge into **one** georeferenced raster with no reprojection; the map is the lon[-12,12]
box plus a NE tab (`E12/E16_N44`), so the SE corner of the bounding rectangle (lon 12–20,
lat 32–44) is nodata.

⚠ **This section READS the mosaic; it does not build it.** `scripts/map_mosaics.py`
(PLAN_Rebuild step 12) is the sole producer of `regional_{layer}_mosaic.tif`: it carries each
tile's `SIZE_FLOOR_*` basis forward, stamps `MOSAIC_*` provenance, and gates the footprint to a
closed account (`n_finite == 26×1479² − 7,940` intra-tile nodata). An earlier version of this
cell called `mosaic_geotiffs(..., out_path=MAP_DIR/"regional_abundance_mosaic.tif")`, which would
overwrite that tagged product with an untagged look-alike — and notebooks are **not** covered by
the test-side write guard. `src.mapping.load_regional_mosaic` reads it instead, and merges in
memory (no write, no tags) only if the file is genuinely absent.

Below is the calibrated `fractional_area` abundance with the cohort footprints overlaid:
high-abundance terrain should track the boulder-rich cohort sites along the highland–lowland
boundary (the qualitative form of validation leg 4, ahead of the quantitative legs in §3).
""", "ab_md"))

cells.append(code(
    """import math
from src.mapping import load_regional_mosaic
MAP_DIR = REPO / "reports" / "map_region"
ab_tifs = sorted(p for p in MAP_DIR.glob("*_abundance.tif")
                 if not p.name.startswith("regional_")) if MAP_DIR.exists() else []
if not ab_tifs:
    print("No abundance GeoTIFFs yet under reports/map_region/.")
    print("Run on Sherlock:  python scripts/map_region.py --all  -> download *.tif into reports/map_region/.")
else:
    # READ the step-12 mosaic (see the note above); never re-write it from this notebook.
    arr, transform, _, mmeta = load_regional_mosaic(MAP_DIR, "abundance", dtype="float32")
    print(f"mosaic source: {mmeta['source']}  tiles={mmeta['n_tiles']}  "
          f"built_by={mmeta['tags'].get('MOSAIC_BUILT_BY', '(untagged)')}")
    print("  size-floor basis:", {k: v for k, v in mmeta["tags"].items()
                                  if k.startswith("SIZE_FLOOR_")} or "(none — untagged merge)")
    R = 3396190.0; dpm = 180.0 / (math.pi * R)   # clon_0 equirectangular metres -> degrees
    h, w = arr.shape
    left, top = transform.c, transform.f
    right, bottom = left + w * transform.a, top + h * transform.e
    ext = [left * dpm, right * dpm, bottom * dpm, top * dpm]   # lon0, lon1, lat0, lat1
    vmax = float(np.nanpercentile(arr, 99)) or 1e-3

    fig, ax = plt.subplots(figsize=(11, 6.6))
    im = ax.imshow(np.ma.masked_invalid(arr), cmap="turbo", vmin=0, vmax=vmax, extent=ext,
                   origin="upper", interpolation="nearest", aspect="equal")
    for t, (lo, la) in boxes.items():                      # tile outlines
        ax.add_patch(Rectangle((lo, la), TILE_DEG, TILE_DEG, fill=False,
                               edgecolor="white", lw=0.3, alpha=0.12))  # faint: the grid is not data
    for lab, c in COL.items():                             # cohort centres (label colour)
        sub = in_block[in_block.BoulderLabel == lab]
        if len(sub):
            ax.scatter(sub.CenterLon_180, sub.CenterLat, s=30, c=c, edgecolor="k", lw=0.4,
                       label=f"{lab} (n={len(sub)})", zorder=4)
    # HiRISE footprint outlines (real Stage-2 window bounds) — QA: abundance structure must NOT
    # trace these boxes (the map is off-HiRISE CTX inference; HiRISE never enters it).
    fp_drawn = 0
    for oid, (flo0, fla0, flo1, fla1) in footprints.items():
        if flo1 < ext[0] or flo0 > ext[1] or fla1 < ext[2] or fla0 > ext[3]:
            continue
        ax.add_patch(Rectangle((flo0, fla0), flo1 - flo0, fla1 - fla0, fill=False,
                               edgecolor="lime", lw=0.9, zorder=6,
                               label="HiRISE footprint" if fp_drawn == 0 else None))
        fp_drawn += 1
    pi = cohort[cohort.ObsId == PAPER_IMG]
    if len(pi):
        ax.scatter(pi.CenterLon_180, pi.CenterLat, s=210, marker="*", facecolor="gold",
                   edgecolor="k", lw=0.7, zorder=7, label=f"{PAPER_IMG} (Rodriguez+2016)")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)")
    ax.set_title("Regional rock-abundance mosaic — calibrated fractional_area @160 m/px\\n"
                 "circum-Chryse boundary (26 CTX tiles) — green = HiRISE footprints (QA overlay)")
    fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02, label=f"fractional_area (vmax=p99={vmax:.3f})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    out = FIG / "24_region_mosaic.png"; fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out.relative_to(REPO), "(read, did not write, the abundance mosaic)")
    plt.show()""",
    "fig_ab"))

cells.append(md(
    """### 2b. All three products side by side

The same stitch applied to the **calibrated P(boulder-rich)** tiles, plus the **binary
rich/poor** map (threshold P ≥ 0.5). Abundance (continuous regression target) and
probability (the classifier) carry the same spatial structure; the binary panel is what a
"where are the boulder fields" map looks like. Boulder-rich cohort sites in cyan.
""", "products_md"))

cells.append(code(
    """if ab_tifs:
    prob_arr, _, _, _ = load_regional_mosaic(MAP_DIR, "prob", dtype="float32")
    binary = np.where(np.isfinite(prob_arr), (prob_arr >= 0.5).astype(float), np.nan)
    rich = in_block[in_block.BoulderLabel == "Boulder rich"]

    panels = [("abundance  (fractional_area)", arr,      "turbo",    0.0, vmax),
              ("P(boulder-rich)  calibrated", prob_arr,  "magma",    0.0, 1.0),
              ("binary rich / poor  (P>=0.5)", binary,   "RdYlBu_r", 0.0, 1.0)]
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 13.5))   # vertical stack (region is wide)
    for ax, (title, data, cmap, vmin, vmx) in zip(axes, panels):
        im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=vmin, vmax=vmx, extent=ext,
                       origin="upper", aspect="equal", interpolation="nearest")
        ax.scatter(rich.CenterLon_180, rich.CenterLat, s=16, c="cyan", edgecolor="k",
                   lw=0.3, zorder=4)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_title(title, fontsize=10); ax.set_ylabel("lat (deg N)")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    axes[-1].set_xlabel("lon (deg E)")
    fig.suptitle("Regional map products — circum-Chryse  (boulder-rich cohort sites in cyan)",
                 fontsize=12)
    fig.tight_layout()
    out = FIG / "24_region_products.png"; fig.savefig(out, dpi=145, bbox_inches="tight")
    print("wrote", out.relative_to(REPO)); plt.show()""",
    "fig_products"))

cells.append(md(
    """### 2c. Binary boulder map on the MOLA terrain (where are the boulder fields?)

The deliverable read of the map: predicted **boulder-rich (P ≥ 0.5)** tiles overlaid in red on
the MOLA shaded-relief background, predicted-poor dimmed, with the **−3795 m lHl1 paleoshoreline**
and the cohort boulder-rich truth sites (yellow triangles). This is the qualitative form of
PLAN legs 1/3/4 in one frame: the predicted boulder fields should hug the highland side of the
contour (the run-up zone) and coincide with the cohort truth, while the distal lowland plains
stay poor. Co-registration is exact — both layers are in the CTX `clon_0` CRS.
""", "binmola_md"))

cells.append(code(
    """from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
if ab_tifs and MOLA.exists():
    rich_overlay = np.where(binary == 1, 1.0, np.nan)   # predicted boulder-rich
    poor_overlay = np.where(binary == 0, 1.0, np.nan)   # predicted boulder-poor

    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.imshow(hs, cmap="gray", extent=mext, origin="upper", aspect="equal", zorder=0)
    ax.imshow(np.ma.masked_invalid(poor_overlay), cmap=ListedColormap(["#1f77b4"]), extent=ext,
              origin="upper", aspect="equal", alpha=0.18, zorder=1, interpolation="nearest")
    ax.imshow(np.ma.masked_invalid(rich_overlay), cmap=ListedColormap(["#d62728"]), extent=ext,
              origin="upper", aspect="equal", alpha=0.55, zorder=2, interpolation="nearest")
    cs = ax.contour(LON, LAT, dem, levels=[-3795], colors="k", linewidths=1.4, zorder=3)
    ax.clabel(cs, fmt={-3795: "-3795 m (lHl1)"}, fontsize=7)
    for t, (lo, la) in boxes.items():
        ax.add_patch(Rectangle((lo, la), TILE_DEG, TILE_DEG, fill=False, edgecolor="#b8860b",
                               lw=0.4, alpha=0.22, zorder=4))
    ax.scatter(rich.CenterLon_180, rich.CenterLat, s=48, marker="^", facecolor="yellow",
               edgecolor="k", lw=0.5, zorder=6)

    ax.set_xlim(mext[0], mext[1]); ax.set_ylim(mext[2], mext[3])
    ax.set_xlabel("longitude (deg E)"); ax.set_ylabel("latitude (deg N)")
    ax.set_title("Predicted boulder-rich tiles on MOLA terrain — circum-Chryse boundary\\n"
                 "(red = P(rich)>=0.5; blue = poor; black = -3795 m paleoshoreline)")
    handles = [Patch(facecolor="#d62728", alpha=0.55, label="predicted boulder-rich (P>=0.5)"),
               Patch(facecolor="#1f77b4", alpha=0.30, label="predicted boulder-poor"),
               plt.Line2D([], [], marker="^", ls="", mfc="yellow", mec="k",
                          label="cohort boulder-rich (truth)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8,
              frameon=False)
    fig.tight_layout()
    out = FIG / "24_region_binary_on_mola.png"; fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out.relative_to(REPO)); plt.show()""",
    "fig_binmola"))

cells.append(md(
    """### 2d. Abundance vs. raw CTX — are the rectangular blocks in the source data?

The abundance map shows **high-amplitude rectangular blocks** (notebook 25). To test whether they
originate in the **CTX mosaic itself** (radiometric differences between source frames) rather than in
the model, here are the **predicted abundance (left) and the raw CTX brightness (right) side by side**,
on the same 160 m/px grid and the same extent, for the tiles whose Murray CTX zip is cached locally,
with HiRISE footprints overlaid. If the abundance blocks coincide with brightness/seam structure in
the raw CTX, the artifact is a property of the 5 m/px source data (per-frame radiometry) that the
model faithfully tracks. (Each Murray tile is itself a patchwork of CTX source frames — notebook 25
quantifies the per-frame effect.)
""", "ctx_md"))

cells.append(code(
    """from src.striping import mosaic_tiles, CTX_ZIP_DIR
ctx_tiles = [t for t in BLOCK_TILES if (CTX_ZIP_DIR / f"{t}.zip").exists()]
print(f"abundance+CTX available for {len(ctx_tiles)} of {len(BLOCK_TILES)} tiles:", ctx_tiles)
if ctx_tiles:
    abm, ctxm, ctr, _ = mosaic_tiles(ctx_tiles, "abundance", with_ctx=True)
    R = 3396190.0; dpm = 180.0 / (math.pi * R)
    ch, cw = ctxm.shape
    cext = [ctr.c * dpm, (ctr.c + cw * ctr.a) * dpm, (ctr.f + ch * ctr.e) * dpm, ctr.f * dpm]
    avmax = float(np.nanpercentile(abm, 99)) or 1e-3
    cvlo, cvhi = np.nanpercentile(ctxm, [2, 98])

    def overlay(ax):
        for t, (lo, la) in boxes.items():
            ax.add_patch(Rectangle((lo, la), TILE_DEG, TILE_DEG, fill=False,
                                   edgecolor="cyan", lw=0.4, alpha=0.25))
        fp = 0
        for oid, (a0, b0, a1, b1) in footprints.items():
            if a1 < ext[0] or a0 > ext[1] or b1 < ext[2] or b0 > ext[3]:
                continue
            ax.add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0, fill=False, edgecolor="lime",
                                   lw=0.9, zorder=6, label="HiRISE footprint" if fp == 0 else None))
            fp += 1
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("longitude (deg E)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), sharex=True, sharey=True)
    i0 = axes[0].imshow(np.ma.masked_invalid(abm), cmap="turbo", vmin=0, vmax=avmax, extent=cext,
                        origin="upper", interpolation="nearest", aspect="equal")
    overlay(axes[0]); axes[0].set_ylabel("latitude (deg N)")
    axes[0].set_title("predicted rock abundance (fractional_area)")
    fig.colorbar(i0, ax=axes[0], fraction=0.030, pad=0.02, label=f"abundance (vmax=p99={avmax:.3f})")
    i1 = axes[1].imshow(np.ma.masked_invalid(ctxm), cmap="gray", vmin=cvlo, vmax=cvhi, extent=cext,
                        origin="upper", interpolation="nearest", aspect="equal")
    overlay(axes[1]); axes[1].set_title("raw CTX brightness")
    fig.colorbar(i1, ax=axes[1], fraction=0.030, pad=0.02, label="CTX DN")
    axes[1].legend(loc="upper right", fontsize=8, frameon=False)
    fig.suptitle("Abundance vs raw CTX @160 m/px (cached tiles) — same grid & scale\\n"
                 "abundance blocks should coincide with CTX brightness/seam structure (green = HiRISE)",
                 fontsize=12)
    fig.tight_layout()
    out = FIG / "24_region_abund_vs_ctx.png"; fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out.relative_to(REPO)); plt.show()""",
    "fig_ctx"))

cells.append(md(
    """## 3. Validation legs (PLAN §2)

1. **Spatial co-location** *(below)* — abundance band ↔ THEMIS night-IR thermal-bright.
2. **Thermal-inertia correlation** — rank-corr(abundance, THEMIS *quantitative* TI / Fergason
   2006). *(TES `nmap2003` was an RGB render — unusable; switched to physical THEMIS TI, DECISIONS
   2026-06-18b. Needs the two `.cub` tiles → to come.)*
3. **Shoreline-distance profile** — abundance vs distance from the −3795 m MOLA contour.
4. **LOIO truth anchor** — **held-out** predicted abundance vs BoulderNet detections across the
   cohort (per-image AUC ≈ 0.43). *NOT* the all-data map at a cohort site like `ESP_017355_2260`:
   that's **in-sample** (the deployed head trained on it) → circular. The honest anchor reuses the
   modeling LOIO cross-validation; a map-at-cohort panel, if shown, is labelled in-sample.
5. **Generalisation** — does the band continue along un-imaged boundary segments?

### 3.1 Leg 1 — spatial co-location (abundance ↔ THEMIS night-IR)

The paper's exact proxy: in **THEMIS nighttime IR**, rocky/high-thermal-inertia surfaces stay warm
(bright), dust/fines go cold (dark). If our CTX-texture abundance is recovering real rockiness, the
high-abundance band should co-locate with the thermal-bright terrain — an **independent** check
(THEMIS never enters the model). THEMIS night-IR (100 m/px, USGS v14) is fetched + reprojected onto
the **abundance grid** (`fetch_validation_data.py --product themis_night_ir --match-mosaic`; the
region crosses the mosaic's lon-0 seam, auto-split by `validation_retrieve`). Brightness here is
8-bit DN (a relative proxy), so the correlation is reported as **Spearman ρ** (rank), not absolute TI.
""", "legs_md"))

cells.append(code(
    """from scipy.stats import spearmanr
import rasterio
THERMAL = REPO / "cache_v2" / "validation" / "themis_night_ir_region.tif"
AB = MAP_DIR / "regional_abundance_mosaic.tif"
if not THERMAL.exists() or not AB.exists():
    print("Missing inputs for leg 1:")
    print("  abundance mosaic:", AB.exists(), AB.relative_to(REPO))
    print("  THEMIS night-IR :", THERMAL.exists(), THERMAL.relative_to(REPO))
    print("Fetch THEMIS:  python scripts/fetch_validation_data.py --product themis_night_ir --match-mosaic")
else:
    with rasterio.open(AB) as da:
        ab = da.read(1); T = da.transform; H, W = ab.shape
    with rasterio.open(THERMAL) as dt:
        ti = dt.read(1); T_ti = dt.transform
    # R01: leg 1 correlates these two arrays BY INDEX. The corrected mosaic keeps the shipped
    # shape but moves +100 m E / -80 m S, and this THEMIS crop was fetched --match-mosaic
    # against the OLD transform -- same shape, different ground position, no error. Fail loud.
    from src.mapping import assert_coregistered
    assert_coregistered(T, T_ti, shape_a=ab.shape, shape_b=ti.shape,
                        name_a="abundance mosaic", name_b="THEMIS night-IR")
    R = 3396190.0; dpm = 180.0 / (math.pi * R)
    e3 = [T.c * dpm, (T.c + W * T.a) * dpm, (T.f + H * T.e) * dpm, T.f * dpm]
    good = np.isfinite(ab) & np.isfinite(ti) & (ti > 0)
    rho, p = spearmanr(ab[good], ti[good])
    print(f"co-registered pixels: {int(good.sum()):,}   pixel-level Spearman rho = {rho:+.3f} (p={p:.1e})")

    # The hypothesis is about the BAND, not 160 m pixels -- coarsen (block-mean, nodata-aware) and
    # re-correlate. Pixel noise + the ~200 m co-reg slack + crater rings wash out at the band scale.
    def _coarsen(a, f):
        h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
        return np.nanmean(a[:h, :w].reshape(h // f, f, w // f, f), axis=(1, 3))
    abm = np.where(good, ab, np.nan); tim = np.where(good, ti, np.nan)
    rho_band = rho
    for f in (8, 32, 64):
        ac, tc = _coarsen(abm, f), _coarsen(tim, f)
        m = np.isfinite(ac) & np.isfinite(tc)
        if m.sum() > 20:
            rc, pc = spearmanr(ac[m], tc[m])
            if f == 32:
                rho_band = rc
            print(f"  coarsened x{f:>2} (~{f*160/1000:.1f} km cells, n={int(m.sum()):,}): Spearman rho = {rc:+.3f} (p={pc:.1e})")

    fig, axes = plt.subplots(2, 1, figsize=(11, 11))
    panels = [("calibrated abundance (fractional_area)", ab, "turbo", np.nanpercentile(ab, 99)),
              ("THEMIS night-IR brightness (rocky=bright, proxy for thermal inertia)", ti, "inferno", None)]
    for axi, (title, dat, cmap, vmx) in zip(axes, panels):
        im = axi.imshow(np.ma.masked_invalid(dat), cmap=cmap, extent=e3, origin="upper",
                        aspect="equal", vmin=0, vmax=vmx, interpolation="nearest")
        try:
            axi.contour(LON, LAT, dem, levels=[-3795], colors="cyan", linewidths=0.9)
        except Exception:
            pass
        rich = in_block[in_block.BoulderLabel == "Boulder rich"]
        axi.scatter(rich.CenterLon_180, rich.CenterLat, s=16, c="cyan", edgecolor="k", lw=0.3, zorder=5)
        axi.set_xlim(e3[0], e3[1]); axi.set_ylim(e3[2], e3[3]); axi.set_title(title, fontsize=10)
        axi.set_ylabel("lat (deg N)"); fig.colorbar(im, ax=axi, fraction=0.025, pad=0.02)
    axes[-1].set_xlabel("lon (deg E)")
    fig.suptitle(f"Leg 1 — abundance vs THEMIS night-IR  (Spearman rho = {rho:+.3f} pixel / {rho_band:+.3f} @5km;"
                 f"  cyan = -3795 m + cohort)", fontsize=12)
    fig.tight_layout()
    out = FIG / "24_leg1_colocation.png"; fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out.relative_to(REPO)); plt.show()""",
    "fig_leg1"))

cells.append(md(
    """**Leg-1 read (honest), now on co-registered layers.** The independent thermal proxy
co-locates with predicted abundance in the **right direction but weakly**: Spearman ρ =
**+0.052 at the pixel** (n = 56.7 M), rising to **+0.066 at ~1.3 km** and flattening to
**+0.063–0.064** at 5–10 km — highly significant (n is huge) but a small effect. So this
corroborates rather than confirms.

⚠ **Misregistration was not what was holding leg 1 down.** This run is the first where the
comparison is genuinely cell-for-cell: the map is on the R01 global lattice and THEMIS was
re-fetched onto it (`assert_coregistered` dx = dy = 0), whereas the 2026-06-19 execution
correlated the pre-R01 map — 26 distinct sub-cell phases, median 140 m displacement — against a
THEMIS crop matched to *that* transform. It read ρ ≈ +0.06 pixel / ≈ +0.07 at band scale. The
corrected comparison is **marginally lower, not higher**. Two honest consequences: the June
number was not inflated by the error in any way that mattered, and the +0.0741 → +0.0821 lift
R01 measured *per tile* does not carry to the pooled mosaic. Leg 1 is a genuinely weak
corroboration, and it is now weak for reasons other than geometry.

Several reasons it's weak, none fatal: (i) THEMIS **night-IR brightness is a crude 8-bit relative
proxy**, not calibrated thermal inertia, and responds to *all* rocky/bedrock/dust/slope variation,
not specifically meter-boulders — **leg 2's physical THEMIS TI (Fergason 2006) is the cleaner
test**; (ii) the abundance map carries **unmitigated CTX source-frame structure** (A1 demoted
2026-08-25; window-median η² 0.1444 at ratio 1.599 over its own rotation null), which is variance
THEMIS cannot possibly track; (iii) ~200 m co-registration slack in the *labels* plus THEMIS's own
100 m posting. The visual band-to-bright correspondence along the boundary is the qualitative form
of the claim; the quantitative weight should rest on leg 2 + the cohort truth-anchor (leg 4).
""", "leg1_read"))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", NB_PATH)
