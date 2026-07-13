"""H4 build-prep verify items (PLAN_H4_Leveling §5) — the two pre-build checks.

A. OVERLAP GRAPH AT SCALE. H4's leveling needs a connected frame-overlap graph — each
   disconnected component carries its own gauge (offsets are only relative within a
   component). Build the graph for the full 26-tile / ~907-frame region from the SeamMap
   footprints (dissolved per PRODUCT_ID, unioned across tiles) and count components.
   NOTE the SeamMap is a *partition* (fragments touch at seams, never overlap), so edges
   are buffered-adjacency: two frames are linked if their dissolved footprints come within
   2·buffer of each other. Adjacent-in-partition frames physically overlap (the seam line
   is imaged by both), so this is the right proxy; true overlap *widths* are only known
   once the frames are ISIS-projected at build time. Sensitivity across buffers reported.

B. H1 DEPLOY-TIME CENTERING-STATISTIC STABILITY. H1 centers each window by its OWN
   median I/F; training used obs-crop windows, deploy uses whole frames. If the median is
   spatially stable across a frame, per-frame ≈ per-crop and nothing changes; if it
   drifts, centering must be per-window. Full frames are not local, so two local probes:
     B1: split each 75 km pilot aligned array into a 3x3 grid of ~25 km sub-windows;
         spread of ln(median I/F) across sub-windows = within-frame drift over a large
         support.
     B2: frames appearing in >=2 independent local crops (multi-obs frames + pilot
         overlaps): range of ln(median) across those windows.
   Yardsticks: the between-frame ln-median spread H1 exists to remove (~0.22, the 24.9%
   leg-A0 spread) and the H1 log-stretch width ln(1.1170/0.8400) = 0.285.

Run (CPU, minutes):
  conda run --no-capture-output -n geospatial python -u scripts/f_h4_buildprep.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import src.modeling  # noqa: F401  OpenMP bootstrap; must precede numpy/pandas

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import rasterio
from scripts.map_region import BLOCK_TILES
from src.striping import load_frames

FIG = REPO / "reports" / "figures"
ALIGNED = REPO / "reports" / "f_timing" / "pilot_work" / "aligned"
OBS_CROPS = REPO / "reports" / "f_leg_b" / "obs_crops"
BUFFERS_M = [0.0, 250.0, 1000.0]
STEP = 8            # subsample stride for medians (5 m/px -> 40 m sampling)
MIN_VALID = 0.05    # min finite fraction for a sub-window median to count


# ------------------------------------------------------------------ A: overlap graph
def union_find_components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    comp = defaultdict(list)
    for x in range(n):
        comp[find(x)].append(x)
    return sorted(comp.values(), key=len, reverse=True)


def graph_at_scale():
    import geopandas as gpd

    print("=== A. overlap graph at scale (26 tiles) ===", flush=True)
    parts, n_frag = [], 0
    for tile in BLOCK_TILES:
        g = load_frames(tile)                       # dissolved per PRODUCT_ID within tile
        n_frag += len(g)
        parts.append(g[["PRODUCT_ID", "geometry"]].assign(tile=tile))
        print(f"  {tile}: {len(g)} frames", flush=True)
    allg = pd.concat(parts, ignore_index=True)
    crs = parts[0].crs
    # union each frame's per-tile pieces (frames span tiles)
    merged = gpd.GeoDataFrame(allg, crs=crs).dissolve(by="PRODUCT_ID")
    pids = list(merged.index)
    n = len(pids)
    print(f"\n{n} unique frames from {n_frag} per-tile footprints "
          f"(plan expected ~907 / 1,371)", flush=True)
    # SeamMap fragments are pixel-resolution — buffer() on the raw dissolved multipolygons
    # is quadratic in vertices (the 2026-07-11 run burned 5.7 CPU-h there). 50 m tolerance
    # (10 native px) is harmless for >=250 m adjacency and makes the sweep seconds.
    simplified = merged.geometry.simplify(50.0)

    rows = []
    comp_rows = []
    for buf in BUFFERS_M:
        geo = simplified.buffer(buf / 2.0) if buf else merged.geometry
        sidx = geo.sindex
        edges = set()
        for i, gm in enumerate(geo.values):
            for j in sidx.query(gm, predicate="intersects"):
                if j > i:
                    edges.add((i, int(j)))
        comps = union_find_components(n, edges)
        deg = np.zeros(n, int)
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        iso = int((deg == 0).sum())
        rows.append({"buffer_m": buf, "n_frames": n, "n_edges": len(edges),
                     "n_components": len(comps), "largest_comp": len(comps[0]),
                     "largest_frac": round(len(comps[0]) / n, 4),
                     "isolated_frames": iso,
                     "median_degree": float(np.median(deg)), "max_degree": int(deg.max())})
        print(f"  buffer {buf:>6.0f} m: {len(edges)} edges, {len(comps)} components, "
              f"largest {len(comps[0])}/{n} ({len(comps[0])/n:.1%}), isolated {iso}, "
              f"median degree {np.median(deg):.0f}", flush=True)
        if buf == BUFFERS_M[0]:
            for rank, c in enumerate(comps):
                if len(c) == 1 and rank >= 20:
                    break
                comp_rows.append({"rank": rank, "n_frames": len(c),
                                  "example_pids": ";".join(pids[k] for k in c[:3])})
        # write incrementally — a stalled later sweep must not lose earlier rows
        pd.DataFrame(rows).to_csv(FIG / "f_h4_buildprep_graph.csv", index=False)
        pd.DataFrame(comp_rows).to_csv(FIG / "f_h4_buildprep_components.csv", index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ B: median stability
def ln_median(arr) -> float | None:
    a = arr[np.isfinite(arr)]
    a = a[a > 0]
    if a.size < MIN_VALID * arr.size:
        return None
    return float(np.log(np.median(a)))


def pilot_subwindow_drift():
    """B1: 3x3 sub-window ln(median) spread per pilot frame (75 km support)."""
    rows = []
    for p in sorted(ALIGNED.glob("*.npy")):
        a = np.load(p, mmap_mode="r")[::STEP, ::STEP]
        h, w = a.shape
        meds = []
        for r in range(3):
            for c in range(3):
                m = ln_median(np.asarray(a[r * h // 3:(r + 1) * h // 3,
                                           c * w // 3:(c + 1) * w // 3]))
                if m is not None:
                    meds.append(m)
        if len(meds) < 2:
            continue
        rows.append({"PRODUCT_ID": p.stem, "kind": "pilot_3x3", "n_windows": len(meds),
                     "ln_med_range": round(max(meds) - min(meds), 4),
                     "ln_med_std": round(float(np.std(meds)), 4),
                     "frame_ln_med": round(float(np.mean(meds)), 4)})
    return rows


def cross_crop_drift():
    """B2: same frame seen in >=2 independent crop windows (multi-obs + pilot x obs)."""
    by_pid = defaultdict(list)   # pid -> [(label, ln_median)]
    for p in sorted(OBS_CROPS.glob("*_ifcrop.tif")):
        parts = p.name.replace("_ifcrop.tif", "").split("_")
        obs_id, pid = "_".join(parts[:3]), "_".join(parts[3:])
        with rasterio.open(p) as ds:
            a = ds.read(1)[::4, ::4]
        m = ln_median(a)
        if m is not None:
            by_pid[pid].append((obs_id, m))
    for p in sorted(ALIGNED.glob("*.npy")):   # pilot windows are additional supports
        a = np.load(p, mmap_mode="r")[::STEP, ::STEP]
        m = ln_median(np.asarray(a))
        if m is not None:
            by_pid[p.stem].append(("pilot_E8_N44", m))
    rows = []
    for pid, vals in by_pid.items():
        if len(vals) < 2:
            continue
        ms = [v for _, v in vals]
        rows.append({"PRODUCT_ID": pid, "kind": "cross_crop", "n_windows": len(vals),
                     "ln_med_range": round(max(ms) - min(ms), 4),
                     "ln_med_std": round(float(np.std(ms)), 4),
                     "frame_ln_med": round(float(np.mean(ms)), 4),
                     "windows": ";".join(l for l, _ in vals)})
    return rows


def centering_stability():
    print("\n=== B. H1 centering-statistic stability ===", flush=True)
    rows = pilot_subwindow_drift() + cross_crop_drift()
    df = pd.DataFrame(rows)
    df.to_csv(FIG / "f_h4_buildprep_median_stability.csv", index=False)
    # yardstick: between-frame spread on the pilot (what H1 removes)
    pil = df[df.kind == "pilot_3x3"]
    between = float(pil["frame_ln_med"].max() - pil["frame_ln_med"].min())
    stretch = float(np.log(1.1170 / 0.8400))
    print(df.to_string(index=False))
    print(f"\nyardsticks: between-frame ln-median spread (pilot) = {between:.3f}; "
          f"H1 log-stretch width = {stretch:.3f}")
    for kind, g in df.groupby("kind"):
        print(f"  {kind}: median within-frame range {g.ln_med_range.median():.4f}, "
              f"worst {g.ln_med_range.max():.4f} "
              f"({g.ln_med_range.max()/between:.0%} of between-frame spread)")
    worst = df.ln_med_range.max()
    print("\nVERDICT:", "STABLE — per-frame median at deploy ≈ per-crop training statistic"
          if worst < 0.25 * between else
          "DRIFTS — center per-window (or per-frame-with-latitude-band) at deploy")
    return df


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    centering_stability()   # cheap local file reads — run first so a graph stall can't block it
    graph_at_scale()


if __name__ == "__main__":
    main()
