"""Stage 7.0 feasibility probe -- compositional analysis on the 3-image trio.

Runs Tests A + B + dust-confound discriminator per PLAN_Compositional.md §3.1, using
*truth* BoulderNet polygons (not model predictions).

  Test A -- per-polygon: extract HiRISE COLOR.JP2 pixel spectra INSIDE each boulder
            polygon and in a 2-10 m outward buffer ring. Paired Wilcoxon test per band
            (IR / RED / BG) and per band ratio (IR/BG, IR/RED, dust_index = RED/BG).
  Test B -- per-tile @ S=64: extract mean spectra inside each 320 m CTX tile,
            partition tiles by `fractional_area >= 1e-2` (truth boulder-rich/poor
            label), Mann-Whitney U + Cohen's d per band and per band ratio.
  Dust   -- compare dust_index between boulder vs surroundings; partial-correlation
            controlling for dust_index to discriminate dust-attributable from
            composition-attributable signals.

Outputs (under cache_v2/stage7/):
  - test_a_per_polygon.parquet  -- raw paired spectra per polygon
  - test_b_per_tile.parquet     -- raw tile spectra + truth labels
  - test_a_summary.parquet      -- per-image stats (Wilcoxon p, Cohen's d) per band
  - test_b_summary.parquet      -- per-image stats (Mann-Whitney p, Cohen's d) per band
  - dust_summary.parquet        -- dust discriminator outputs
  - _stage7_feasibility.md      -- human-readable summary + go/no-go decision (written
                                   separately by the notebook author from these parquets)

Run via:
    conda run -n geospatial python scripts/probes/_stage7_feasibility.py
Typical runtime: 5-15 min (dominated by JP2 windowed reads).
"""
from __future__ import annotations

import functools
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.crs import CRS as RioCRS
from scipy import stats as sst
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src import colour  # noqa: E402

# Force flush on every print -- `conda run` buffers subprocess stdout. Combined with
# --no-capture-output on the invocation, this guarantees per-line progress visibility.
print = functools.partial(print, flush=True)  # noqa: A001

# ----------------------------------------------------------------------- config
CACHE = Path("cache_v2")
OUT = CACHE / "stage7"
OUT.mkdir(parents=True, exist_ok=True)
DETECTIONS = Path("C:/Users/brian/Documents/PhD/HiRiseToCTXBoulders/hirise_40_vClaire")
LABELS_DIR = Path("dataset_v2/labels")

TRIO = [
    ("ESP_042964_2160", "high-density positive (AUC 0.91)"),
    ("ESP_054000_2255", "anti-signal #1 (AUC 0.40)"),
    ("ESP_055253_2245", "anti-signal #2 (AUC 0.42)"),
]

# Test A
POLYGON_BUFFER_INNER_M = 2.0
POLYGON_BUFFER_OUTER_M = 10.0
MIN_POLYGON_PIXELS = 8
MIN_RING_PIXELS = 16
POLYGON_SAMPLE_N = 800  # cap per image for feasibility; full pop if smaller
RNG = np.random.default_rng(20260531)

# Test B
S64_SCALE_IDX = 3  # tile_size_px=64 @ 5 m/px CTX = 320 m
MIN_TILE_PIXELS = 64
BINARY_AREA_THRESHOLD = 1e-2  # fa_gt_1e-2 per P4 promotion

# CTX target CRS (from Stage 1 sidecar target_crs_wkt -- Mars_2000 IAU at 3396190 m).
CTX_CRS = RioCRS.from_wkt(
    'PROJCRS["Mars_2000_Equidistant_Cylindrical",BASEGEOGCRS["GCS_Mars_2000",'
    'DATUM["D_Mars_2000",ELLIPSOID["Mars_2000_IAU_IAG",3396190,0,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Reference_Meridian",0,ANGLEUNIT["Degree",0.0174532925199433]]],'
    'CONVERSION["Equidistant Cylindrical (Spherical)",'
    'METHOD["Equidistant Cylindrical (Spherical)",ID["EPSG",1029]],'
    'PARAMETER["Latitude of 1st standard parallel",0,'
    'ANGLEUNIT["Degree",0.0174532925199433]],'
    'PARAMETER["Longitude of natural origin",0,'
    'ANGLEUNIT["Degree",0.0174532925199433]],'
    'PARAMETER["False easting",0,LENGTHUNIT["metre",1]],'
    'PARAMETER["False northing",0,LENGTHUNIT["metre",1]]],'
    'CS[Cartesian,2],AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1]],'
    'AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1]]]'
)


# ----------------------------------------------------------------------- helpers
def _detection_shp(obs_id: str) -> Path:
    candidates = list((DETECTIONS / obs_id).glob("*-mask-nms.shp"))
    if not candidates:
        raise FileNotFoundError(f"no detection shp for {obs_id}")
    return candidates[0]


def _load_polygons(obs_id: str, corrected_crs) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_detection_shp(obs_id))
    return gdf.set_crs(corrected_crs, allow_override=True)


def _filter_polygons_in_swath(gdf: gpd.GeoDataFrame, jp2_bounds: tuple) -> gpd.GeoDataFrame:
    swath = box(*jp2_bounds)
    return gdf[gdf.intersects(swath)].copy().reset_index(drop=True)


# ----------------------------------------------------------------------- Test A
def run_test_a(
    obs_id: str,
    jp2_path: Path,
    lbl: colour.ColorLBL,
    gdf_inside: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Per-polygon paired spectra: interior + 2-10 m outward ring."""
    # Sample if too many
    n = len(gdf_inside)
    if n > POLYGON_SAMPLE_N:
        idx = RNG.choice(n, POLYGON_SAMPLE_N, replace=False)
        gdf_inside = gdf_inside.iloc[idx].reset_index(drop=True)
    print(f"  Test A: {len(gdf_inside)} polygons (of {n} inside swath)")

    records = []
    n_skipped_window = n_skipped_pixels = 0
    t0 = time.time()
    with rasterio.open(jp2_path) as ds:
        pad = POLYGON_BUFFER_OUTER_M + ds.res[0]
        for poly_idx, poly in enumerate(gdf_inside.geometry):
            if poly_idx and poly_idx % 100 == 0:
                rate = poly_idx / max(0.01, time.time() - t0)
                print(f"           [{obs_id}] Test A progress: {poly_idx}/{len(gdf_inside)} "
                      f"polygons ({rate:.1f}/s, kept={len(records)})")
            xmin, ymin, xmax, ymax = poly.bounds
            arr, win_t = colour.read_color_window(
                ds, (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
            )
            if arr is None:
                n_skipped_window += 1
                continue
            interior_mask, ring_mask = colour.polygon_masks(
                poly, POLYGON_BUFFER_INNER_M, POLYGON_BUFFER_OUTER_M,
                window_transform=win_t, window_shape=arr.shape[1:],
            )
            interior = colour.region_means(arr, interior_mask, min_pixels=MIN_POLYGON_PIXELS)
            ring = colour.region_means(arr, ring_mask, min_pixels=MIN_RING_PIXELS)
            if interior is None or ring is None:
                n_skipped_pixels += 1
                continue
            records.append({
                "obs_id": obs_id,
                "polygon_idx": poly_idx,
                "area_m2": float(poly.area),
                "IR_in": interior["IR"], "RED_in": interior["RED"], "BG_in": interior["BG"],
                "IR_ring": ring["IR"], "RED_ring": ring["RED"], "BG_ring": ring["BG"],
                "n_pixels_in": interior["n_pixels"], "n_pixels_ring": ring["n_pixels"],
            })
    print(f"           kept {len(records)} (window-skip {n_skipped_window}, "
          f"pixel-skip {n_skipped_pixels})")
    return pd.DataFrame(records)


# ----------------------------------------------------------------------- Test B
def run_test_b(
    obs_id: str,
    jp2_path: Path,
    lbl: colour.ColorLBL,
    corrected_crs,
) -> pd.DataFrame:
    """Per-tile @ S=64 spectra, joined to truth fractional_area + boulder_count."""
    labels = pd.read_parquet(LABELS_DIR / f"{obs_id}.parquet")
    tiles = labels[labels["scale_idx"] == S64_SCALE_IDX].copy().reset_index(drop=True)
    print(f"  Test B: {len(tiles)} S=64 tiles total for this image")

    transformer = pyproj.Transformer.from_crs(CTX_CRS, corrected_crs, always_xy=True)

    records = []
    n_outside = n_pixel_skip = 0
    t0 = time.time()
    with rasterio.open(jp2_path) as ds:
        jp2_bounds = ds.bounds
        ones_cache: dict[tuple[int, int], np.ndarray] = {}
        for tile_idx, row in tiles.iterrows():
            if tile_idx and tile_idx % 500 == 0:
                rate = tile_idx / max(0.01, time.time() - t0)
                print(f"           [{obs_id}] Test B progress: {tile_idx}/{len(tiles)} "
                      f"tiles ({rate:.1f}/s, kept={len(records)})")
            xs = [row.xmin, row.xmin, row.xmax, row.xmax]
            ys = [row.ymin, row.ymax, row.ymin, row.ymax]
            x2s, y2s = transformer.transform(xs, ys)
            src_bounds = (min(x2s), min(y2s), max(x2s), max(y2s))
            if (src_bounds[2] < jp2_bounds.left or src_bounds[0] > jp2_bounds.right
                    or src_bounds[3] < jp2_bounds.bottom or src_bounds[1] > jp2_bounds.top):
                n_outside += 1
                continue
            arr, _ = colour.read_color_window(ds, src_bounds)
            if arr is None:
                n_outside += 1
                continue
            shape = arr.shape[1:]
            ones = ones_cache.get(shape)
            if ones is None:
                ones = np.ones(shape, dtype=bool)
                ones_cache[shape] = ones
            means = colour.region_means(arr, ones, min_pixels=MIN_TILE_PIXELS)
            if means is None:
                n_pixel_skip += 1
                continue
            records.append({
                "obs_id": obs_id, "ti": int(row.ti), "tj": int(row.tj),
                "fractional_area": float(row.fractional_area),
                "boulder_count": int(row.boulder_count),
                "tile_area": float(row.tile_area),
                "IR": means["IR"], "RED": means["RED"], "BG": means["BG"],
                "n_pixels": means["n_pixels"],
            })
    print(f"           kept {len(records)} tiles (outside-swath {n_outside}, "
          f"pixel-skip {n_pixel_skip})")
    return pd.DataFrame(records)


# ----------------------------------------------------------------------- stats
def _wilcoxon(diffs: np.ndarray) -> tuple[float, float]:
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 30 or (diffs == 0).all():
        return float("nan"), float("nan")
    try:
        w, p = sst.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        return float(w), float(p)
    except ValueError:
        return float("nan"), float("nan")


def _mannwhitney(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return float("nan"), float("nan")
    u, p = sst.mannwhitneyu(a, b, alternative="two-sided")
    return float(u), float(p)


def _cohens_d_paired(diffs: np.ndarray) -> float:
    diffs = diffs[~np.isnan(diffs)]
    if len(diffs) < 2 or diffs.std(ddof=1) == 0:
        return float("nan")
    return float(diffs.mean() / diffs.std(ddof=1))


def _cohens_d_unpaired(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def _resolve(df: pd.DataFrame, expr: str) -> np.ndarray:
    """`expr` is a column name or a single-slash ratio of column names."""
    if "/" in expr:
        num, den = (s.strip() for s in expr.split("/", 1))
        return (df[num] / df[den]).to_numpy()
    return df[expr].to_numpy()


def summarise_test_a(df: pd.DataFrame, lbl: colour.ColorLBL) -> pd.DataFrame:
    """Paired statistics on (interior - ring) per band and per band ratio.

    Lambertian correction is multiplicative (1/cos(i)) and cancels in interior-ring
    differences AND in band ratios -- so we work on raw I/F and just record cos(i) for
    reproducibility."""
    if df.empty:
        return pd.DataFrame()
    feats = {
        "IR": ("IR_in", "IR_ring"),
        "RED": ("RED_in", "RED_ring"),
        "BG": ("BG_in", "BG_ring"),
        "IR_over_BG": ("IR_in/BG_in", "IR_ring/BG_ring"),
        "IR_over_RED": ("IR_in/RED_in", "IR_ring/RED_ring"),
        "dust_index_RED_over_BG": ("RED_in/BG_in", "RED_ring/BG_ring"),
    }
    rows = []
    for feat, (lhs, rhs) in feats.items():
        a = _resolve(df, lhs)
        b = _resolve(df, rhs)
        diffs = a - b
        _, p = _wilcoxon(diffs)
        d = _cohens_d_paired(diffs)
        rows.append({
            "feature": feat, "n_pairs": int((~np.isnan(diffs)).sum()),
            "mean_interior": float(np.nanmean(a)),
            "mean_ring": float(np.nanmean(b)),
            "mean_diff": float(np.nanmean(diffs)),
            "wilcoxon_p": p,
            "cohens_d_paired": d,
            "cos_incidence": lbl.cos_incidence,
        })
    return pd.DataFrame(rows)


def summarise_test_b(df: pd.DataFrame, lbl: colour.ColorLBL) -> pd.DataFrame:
    """Unpaired stats on boulder-rich vs boulder-poor tiles, per band/ratio."""
    if df.empty:
        return pd.DataFrame()
    rich = df["fractional_area"] >= BINARY_AREA_THRESHOLD
    rich_df = df[rich]
    poor_df = df[~rich]
    feats = {
        "IR": "IR",
        "RED": "RED",
        "BG": "BG",
        "IR_over_BG": ("IR", "BG"),
        "IR_over_RED": ("IR", "RED"),
        "dust_index_RED_over_BG": ("RED", "BG"),
    }
    rows = []
    for feat, spec in feats.items():
        if isinstance(spec, str):
            a = rich_df[spec].to_numpy(); b = poor_df[spec].to_numpy()
        else:
            num, den = spec
            a = (rich_df[num] / rich_df[den]).to_numpy()
            b = (poor_df[num] / poor_df[den]).to_numpy()
        _, p = _mannwhitney(a, b)
        d = _cohens_d_unpaired(a, b)
        rows.append({
            "feature": feat, "n_rich": int(rich.sum()), "n_poor": int((~rich).sum()),
            "mean_rich": float(np.nanmean(a)) if len(a) else float("nan"),
            "mean_poor": float(np.nanmean(b)) if len(b) else float("nan"),
            "mannwhitney_p": p,
            "cohens_d": d,
            "cos_incidence": lbl.cos_incidence,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- main
def main() -> int:
    test_a_dfs: list[pd.DataFrame] = []
    test_b_dfs: list[pd.DataFrame] = []
    summary_a_rows: list[pd.DataFrame] = []
    summary_b_rows: list[pd.DataFrame] = []

    for obs_id, role in TRIO:
        print(f"\n=== {obs_id} -- {role} ===")
        lbl = colour.parse_color_lbl(colour.color_lbl_path(CACHE, obs_id))
        corrected_crs = colour.corrected_source_crs(obs_id, CACHE)
        jp2_path = colour.color_jp2_path(CACHE, obs_id)
        if corrected_crs is None:
            print(f"  SKIP: no Stage 1 corrected CRS for {obs_id}")
            continue

        gdf = _load_polygons(obs_id, corrected_crs)
        with rasterio.open(jp2_path) as ds:
            jp2_bounds = tuple(ds.bounds)
        gdf_inside = _filter_polygons_in_swath(gdf, jp2_bounds)
        print(f"  swath polygons: {len(gdf_inside)}/{len(gdf)} "
              f"({100*len(gdf_inside)/max(1,len(gdf)):.1f}%)  "
              f"cos(i)={lbl.cos_incidence:.4f}")

        df_a = run_test_a(obs_id, jp2_path, lbl, gdf_inside)
        df_a.to_parquet(OUT / f"test_a_per_polygon_{obs_id}.parquet")
        df_b = run_test_b(obs_id, jp2_path, lbl, corrected_crs)
        df_b.to_parquet(OUT / f"test_b_per_tile_{obs_id}.parquet")
        test_a_dfs.append(df_a); test_b_dfs.append(df_b)

        sum_a = summarise_test_a(df_a, lbl); sum_a["obs_id"] = obs_id
        sum_b = summarise_test_b(df_b, lbl); sum_b["obs_id"] = obs_id
        summary_a_rows.append(sum_a); summary_b_rows.append(sum_b)

        if not sum_a.empty:
            print("  -- Test A per-band paired diffs (interior - ring):")
            for _, r in sum_a.iterrows():
                print(f"     {r.feature:24s} mean_diff={r.mean_diff:+.4f}  "
                      f"d={r.cohens_d_paired:+.3f}  p={r.wilcoxon_p:.2e}  n={r.n_pairs}")
        if not sum_b.empty:
            print("  -- Test B boulder-rich vs boulder-poor tile means:")
            for _, r in sum_b.iterrows():
                print(f"     {r.feature:24s} mean_rich={r.mean_rich:+.4f}  "
                      f"mean_poor={r.mean_poor:+.4f}  d={r.cohens_d:+.3f}  "
                      f"p={r.mannwhitney_p:.2e}  n_rich={r.n_rich}/n_poor={r.n_poor}")

    # Persist
    if test_a_dfs:
        pd.concat(test_a_dfs, ignore_index=True).to_parquet(OUT / "test_a_per_polygon.parquet")
        pd.concat(summary_a_rows, ignore_index=True).to_parquet(OUT / "test_a_summary.parquet")
    if test_b_dfs:
        pd.concat(test_b_dfs, ignore_index=True).to_parquet(OUT / "test_b_per_tile.parquet")
        pd.concat(summary_b_rows, ignore_index=True).to_parquet(OUT / "test_b_summary.parquet")

    print(f"\nWrote outputs to {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
