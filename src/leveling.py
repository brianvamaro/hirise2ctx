"""H4 overlap-constrained leveling at BUILD scale — the reusable core of PLAN_FBuild Stage C.

The pilot solver (`scripts/f_h4_level.py`) worked on a 7-frame co-registered raster *stack*: every
frame was a dense array on one shared grid, so "co-located" was just an elementwise mask. At 907
frames that representation is impossible (the block is ~10^8 tiles and each frame touches a thin
swath of it), so Stage B emits per-frame **sparse tile lists** keyed to a GLOBAL 160 m grid
(`{TI, TJ, prob}` npz). This module is the same mathematics on that sparse representation:

    edges      per frame PAIR, the sufficient statistics (δ̄_ij, W_ij) over co-located global tiles
    solve      907-unknown weighted LS  Σ_edges W·[(o_i − o_j) − δ̄_ij]²  + λ·Σ o²,  gauge median(o)=0
    CV         held-out-edge |Δp| (PLAN_FBuild §4: a random 5% edge sample at build scale)
    guard      weighted lon/lat surface fit + block-permutation significance (§4.2)

Sign convention (identical to the pilot, verified by `tests/test_leveling.py`): with
δ̄_ij = mean(ℓ_j − ℓ_i) over co-located tiles, the LS wants **o_i − o_j ≈ δ̄_ij**, so a frame whose
logits sit HIGH gets a NEGATIVE offset and ℓ_f + o_f agrees across seams.

Everything here is pure numpy/scipy — no rasterio, no torch, no GPU — so Stage C runs identically on
a Sherlock login node (next to the npzs) or on the laptop after a tar-and-transfer.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS = 1e-4                      # logit clip, matches scripts/f_h4_level._logit
TILE_M = 160.0                  # global tile pitch = S=32 * 5 m/px (scripts/f_region_stageb.GLOBAL_M)
KEY_SHIFT = 1 << 20             # (TI,TJ) -> int64 key; needs |TJ| < 524_288 (grid spans ±66_700)


# --------------------------------------------------------------------------- keys & links
def pack_key(ti, tj) -> np.ndarray:
    """(TI, TJ) global tile indices -> a single monotone int64 key (lexicographic in TI, TJ)."""
    ti = np.asarray(ti, dtype=np.int64)
    tj = np.asarray(tj, dtype=np.int64)
    if tj.size and int(np.abs(tj).max()) >= KEY_SHIFT // 2:
        raise ValueError(f"|TJ| >= {KEY_SHIFT // 2}: key packing would alias (max {np.abs(tj).max()})")
    return ti * KEY_SHIFT + tj


def unpack_key(key) -> tuple[np.ndarray, np.ndarray]:
    key = np.asarray(key, dtype=np.int64)
    tj = ((key + KEY_SHIFT // 2) % KEY_SHIFT) - KEY_SHIFT // 2
    return (key - tj) // KEY_SHIFT, tj


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def logit(p):
    q = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(q) - np.log(1.0 - q)


def intersect_sorted(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indices (ia, ib) of the common values of two SORTED, UNIQUE int64 arrays.

    searchsorted (O(n log m)) rather than np.intersect1d (concatenate + full sort): at 907 frames
    the pair loop runs ~10^4 times over ~10^5-element arrays, where the constant matters.
    """
    if a.size == 0 or b.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    if a.size > b.size:                                  # probe the smaller into the larger
        ib, ia = intersect_sorted(b, a)
        return ia, ib
    pos = np.clip(np.searchsorted(b, a), 0, b.size - 1)
    hit = b[pos] == a
    return np.flatnonzero(hit), pos[hit]


def components(ei, ej, n: int) -> np.ndarray:
    """Connected-component label per frame (union-find). Isolated frames get their own label."""
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return int(x)

    for i, j in zip(np.asarray(ei).tolist(), np.asarray(ej).tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([find(x) for x in range(n)])
    _, lab = np.unique(roots, return_inverse=True)
    return lab.astype(np.int32)


# --------------------------------------------------------------------------- edge construction
@dataclass
class EdgeSet:
    """Overlap graph + the per-edge statistics the solve and the CV need.

    `dbar`/`w` are the exact sufficient statistics over ALL co-located tiles. `samp_*` keeps a
    bounded random SUBSAMPLE of the raw co-located logit pairs per edge, because the held-out CV
    metric (median |p_i − p_j|) is a nonlinear function of the tile pairs and cannot be recovered
    from (δ̄, W). 1,000 pairs/edge fixes the median to ~1e-3 while keeping the cache ~100 MB.
    """
    pids: list[str]
    ei: np.ndarray            # int32 frame index (lower)
    ej: np.ndarray            # int32 frame index (upper)
    dbar: np.ndarray          # float64 mean(ℓ_j − ℓ_i) over co-located tiles
    w: np.ndarray             # float64 co-located tile count
    samp_off: np.ndarray      # int64, len n_edges+1 — CSR-style offsets into samp_i/samp_j
    samp_i: np.ndarray        # float32 ℓ_i on sampled co-located tiles
    samp_j: np.ndarray        # float32 ℓ_j on the same tiles

    @property
    def n_edges(self) -> int:
        return int(self.ei.size)

    def degrees(self, n: int) -> np.ndarray:
        deg = np.zeros(n, dtype=np.int64)
        np.add.at(deg, self.ei, 1)
        np.add.at(deg, self.ej, 1)
        return deg

    def weight_sum(self, n: int) -> np.ndarray:
        ws = np.zeros(n, dtype=np.float64)
        np.add.at(ws, self.ei, self.w)
        np.add.at(ws, self.ej, self.w)
        return ws

    def filter(self, keep: np.ndarray) -> "EdgeSet":
        """Sub-EdgeSet over a boolean/integer edge selection (the min-shared-tiles sensitivity)."""
        idx = np.flatnonzero(keep) if np.asarray(keep).dtype == bool else np.asarray(keep, np.int64)
        off = np.concatenate([[0], np.cumsum([self.samp_off[e + 1] - self.samp_off[e] for e in idx])])
        return EdgeSet(
            pids=self.pids, ei=self.ei[idx], ej=self.ej[idx], dbar=self.dbar[idx], w=self.w[idx],
            samp_off=off.astype(np.int64),
            samp_i=np.concatenate([self.samp_i[self.samp_off[e]:self.samp_off[e + 1]] for e in idx])
            if idx.size else np.empty(0, np.float32),
            samp_j=np.concatenate([self.samp_j[self.samp_off[e]:self.samp_off[e + 1]] for e in idx])
            if idx.size else np.empty(0, np.float32),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, pids=np.array(self.pids, dtype=object), ei=self.ei, ej=self.ej,
                            dbar=self.dbar, w=self.w, samp_off=self.samp_off,
                            samp_i=self.samp_i, samp_j=self.samp_j)

    @classmethod
    def load(cls, path: str | Path) -> "EdgeSet":
        z = np.load(path, allow_pickle=True)
        return cls(pids=[str(p) for p in z["pids"]], ei=z["ei"], ej=z["ej"], dbar=z["dbar"],
                   w=z["w"], samp_off=z["samp_off"], samp_i=z["samp_i"], samp_j=z["samp_j"])


def candidate_pairs(keys: list[np.ndarray], cell_tiles: int = 64) -> list[tuple[int, int]]:
    """Frame pairs that could overlap, by shared coarse cell (64 tiles = 10.24 km).

    A bbox prescreen is too loose for CTX's long N–S swaths (two parallel strips 50 km apart share a
    bbox); co-occurrence in a 10 km cell is tight enough that almost every candidate survives the
    exact intersection, and cheap enough to run over all 907 frames.
    """
    cell2f: dict[int, list[int]] = defaultdict(list)
    for f, k in enumerate(keys):
        if k.size == 0:
            continue
        ti, tj = unpack_key(k)
        for c in np.unique(pack_key(ti // cell_tiles, tj // cell_tiles)).tolist():
            cell2f[c].append(f)
    pairs: set[tuple[int, int]] = set()
    for fs in cell2f.values():
        for a in range(len(fs)):
            for b in range(a + 1, len(fs)):
                pairs.add((fs[a], fs[b]))
    return sorted(pairs)


def build_edges(pids: list[str], keys: list[np.ndarray], logits: list[np.ndarray],
                min_tiles: int = 200, dp_sample: int = 1000, seed: int = 0,
                cell_tiles: int = 64, pairs=None, progress: int = 0) -> EdgeSet:
    """Exact per-edge (δ̄, W) over co-located global tiles + a bounded logit-pair subsample."""
    rng = np.random.default_rng(seed)
    if pairs is None:
        pairs = candidate_pairs(keys, cell_tiles=cell_tiles)
    ei, ej, dbar, w = [], [], [], []
    si, sj, off = [], [], [0]
    for k, (a, b) in enumerate(pairs):
        ia, ib = intersect_sorted(keys[a], keys[b])
        if ia.size < min_tiles:
            continue
        la = np.asarray(logits[a][ia], dtype=np.float64)
        lb = np.asarray(logits[b][ib], dtype=np.float64)
        ei.append(a)
        ej.append(b)
        dbar.append(float((lb - la).mean()))
        w.append(float(ia.size))
        if ia.size > dp_sample:
            sel = rng.choice(ia.size, dp_sample, replace=False)
            la, lb = la[sel], lb[sel]
        si.append(la.astype(np.float32))
        sj.append(lb.astype(np.float32))
        off.append(off[-1] + la.size)
        if progress and len(ei) % progress == 0:
            print(f"    {len(ei)} edges from {k + 1}/{len(pairs)} candidate pairs", flush=True)
    return EdgeSet(
        pids=list(pids),
        ei=np.asarray(ei, np.int32), ej=np.asarray(ej, np.int32),
        dbar=np.asarray(dbar, np.float64), w=np.asarray(w, np.float64),
        samp_off=np.asarray(off, np.int64),
        samp_i=np.concatenate(si) if si else np.empty(0, np.float32),
        samp_j=np.concatenate(sj) if sj else np.empty(0, np.float32),
    )


# --------------------------------------------------------------------------- the solve
def regauge(o: np.ndarray, comp: np.ndarray | None = None) -> np.ndarray:
    """Gauge fix median(o) = 0 — per connected component (offsets are only relative within one)."""
    o = np.asarray(o, dtype=np.float64).copy()
    if comp is None:
        return o - np.median(o)
    for c in np.unique(comp):
        m = comp == c
        o[m] -= np.median(o[m])
    return o


def solve_offsets(es: EdgeSet, lam: float, n: int, comp: np.ndarray | None = None,
                  edge_mask: np.ndarray | None = None) -> np.ndarray:
    """Weighted LS + λ·Tikhonov for the per-frame additive logit offsets (PLAN_H4 §2).

    Minimises Σ W_ij·[(o_i − o_j) − δ̄_ij]² + λ·Σ o². Dense normal equations: n=907 ⇒ a 6.6 MB
    matrix and a millisecond Cholesky, so there is no reason to reach for sparse machinery.
    λ>0 makes the Laplacian SPD (and pins isolated / dropped-out frames at 0, which is what
    leave-one-frame-out needs); λ=0 falls back to the min-norm lstsq the pilot used.
    """
    ei, ej, dbar, w = es.ei, es.ej, es.dbar, es.w
    if edge_mask is not None:
        ei, ej, dbar, w = ei[edge_mask], ej[edge_mask], dbar[edge_mask], w[edge_mask]
    ata = np.zeros((n, n), dtype=np.float64)
    atb = np.zeros(n, dtype=np.float64)
    np.add.at(ata, (ei, ei), w)
    np.add.at(ata, (ej, ej), w)
    np.add.at(ata, (ei, ej), -w)
    np.add.at(ata, (ej, ei), -w)
    np.add.at(atb, ei, w * dbar)
    np.add.at(atb, ej, -w * dbar)
    ata[np.diag_indices(n)] += lam
    if lam > 0:
        try:
            from scipy.linalg import cho_factor, cho_solve
            o = cho_solve(cho_factor(ata, lower=True, check_finite=False), atb, check_finite=False)
        except Exception:                                  # pragma: no cover - numerical fallback
            o = np.linalg.lstsq(ata, atb, rcond=None)[0]
    else:
        o = np.linalg.lstsq(ata, atb, rcond=None)[0]
    return regauge(o, comp)


def _edge_stat(es: EdgeSet, o: np.ndarray, idx, fn) -> np.ndarray:
    """Per-edge reduction `fn(ℓ_i + o_i, ℓ_j + o_j)` over the edge's sampled co-located tiles."""
    idx = np.arange(es.n_edges) if idx is None else np.asarray(idx, dtype=np.int64)
    out = np.empty(idx.size, dtype=np.float64)
    for k, e in enumerate(idx.tolist()):
        s, t = int(es.samp_off[e]), int(es.samp_off[e + 1])
        li = es.samp_i[s:t].astype(np.float64) + o[es.ei[e]]
        lj = es.samp_j[s:t].astype(np.float64) + o[es.ej[e]]
        out[k] = fn(li, lj)
    return out


def edge_dp(es: EdgeSet, o: np.ndarray, idx=None) -> np.ndarray:
    """Per-edge median |p_i − p_j| over its sampled co-located tiles, after applying offsets o.

    Pre-registered as gate 2's metric, so it is still computed and reported — but see
    `edge_dlogit`: in probability space this statistic is minimised by SATURATING the sigmoid, so it
    must not be the only instrument at build scale.
    """
    return _edge_stat(es, o, idx, lambda li, lj: np.median(np.abs(sigmoid(li) - sigmoid(lj))))


def edge_dlogit(es: EdgeSet, o: np.ndarray, idx=None) -> np.ndarray:
    """Per-edge median |ℓ_i − ℓ_j| on co-located tiles — the SATURATION-IMMUNE twin of `edge_dp`.

    `edge_dp` measures disagreement after sigmoid(), which compresses hard near the rails: an offset
    big enough to push both frames' tiles to p≈1 drives |p_i − p_j| → 0 while the underlying logit
    disagreement is untouched. At 906-frame scale that is not hypothetical — the λ=0 solve saturates
    51.8% of co-located tiles, corr(saturated fraction, median |Δp|) = −0.997, and |Δp| therefore
    ranks λ by how hard it saturates rather than by how well frames agree (DECISIONS 2026-07-29:
    med|Δp| 0.1622→0.0112 but med|Δlogit| only 1.1731→1.0859). Logit space is affine in the offsets,
    so it cannot be gamed this way; λ is selected on THIS metric as of 2026-07-29.
    """
    return _edge_stat(es, o, idx, lambda li, lj: np.median(np.abs(li - lj)))


EDGE_METRICS = {"dp": edge_dp, "dlogit": edge_dlogit}


def edge_saturated_frac(es: EdgeSet, o: np.ndarray, idx=None, rail: float = 0.01) -> float:
    """Share of sampled co-located tile probabilities pushed onto a rail (p<rail or p>1−rail).

    The diagnostic that makes a suspiciously good `edge_dp` readable: a solve that "agrees" because
    everything saturated reports a high value here (0.518 at λ*=0 vs 0.017 unleveled).
    """
    idx = np.arange(es.n_edges) if idx is None else np.asarray(idx, dtype=np.int64)
    sat = tot = 0
    for e in idx.tolist():
        s, t = int(es.samp_off[e]), int(es.samp_off[e + 1])
        for p in (sigmoid(es.samp_i[s:t].astype(np.float64) + o[es.ei[e]]),
                  sigmoid(es.samp_j[s:t].astype(np.float64) + o[es.ej[e]])):
            sat += int(((p < rail) | (p > 1.0 - rail)).sum())
            tot += p.size
    return sat / tot if tot else float("nan")


def heldout_edge_cv(es: EdgeSet, lam: float, n: int, frac: float = 0.05, repeats: int = 4,
                    seed: int = 0, metric: str = "dp") -> tuple[float, int]:
    """Held-out-edge CV (PLAN_FBuild §4: 5% edge sample, not the pilot's all-edges loop).

    Fit the offsets WITHOUT a random `frac` of the edges, then score `metric` on exactly those
    edges. Held-out edges whose endpoints fall in different components of the *fit* graph have no
    common gauge and are skipped (counted, and reported so a silent drop can't hide).

    `metric` is "dp" (the pre-registered probability-space statistic) or "dlogit" (saturation-immune;
    what λ* is selected on since 2026-07-29 — see `edge_dlogit`).
    """
    stat = EDGE_METRICS[metric]
    rng = np.random.default_rng(seed)
    m = es.n_edges
    k = max(1, int(round(frac * m)))
    dps, skipped = [], 0
    for _ in range(repeats):
        hold = rng.choice(m, size=min(k, m), replace=False)
        mask = np.ones(m, dtype=bool)
        mask[hold] = False
        comp = components(es.ei[mask], es.ej[mask], n)
        o = solve_offsets(es, lam, n, comp=comp, edge_mask=mask)
        ok = comp[es.ei[hold]] == comp[es.ej[hold]]
        skipped += int((~ok).sum())
        if ok.any():
            dps.append(stat(es, o, hold[ok]))
    return (float(np.median(np.concatenate(dps))) if dps else float("nan")), skipped


def lofo_offsets(es: EdgeSet, o_full: np.ndarray, lam: float, n: int,
                 frames=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-FRAME-out: predict each frame's offset from its overlaps to the retained frames.

    The generalization instrument the 2026-07-15 review asked for (`scripts/f_h4_lofo.py` at pilot
    scale), which doubles as the §0.1 guard-3 "under-pinned offset" detector: a large |o| whose LOFO
    prediction error is also large is an offset the graph does not actually constrain.

    Returns (o_hat, pred_err, n_edges_used); frames not in `frames` get NaN.
    """
    frames = np.arange(n) if frames is None else np.asarray(frames, dtype=np.int64)
    o_hat = np.full(n, np.nan)
    n_used = np.zeros(n, dtype=np.int64)
    for f in frames.tolist():
        touch = (es.ei == f) | (es.ej == f)
        if not touch.any():
            continue
        comp = components(es.ei[~touch], es.ej[~touch], n)
        o_ret = solve_offsets(es, lam, n, comp=comp, edge_mask=~touch)
        # edge (f,j): o_f − o_j ≈ δ̄  ⇒  ô_f = o_j + δ̄ ;  edge (i,f): o_i − o_f ≈ δ̄  ⇒  ô_f = o_i − δ̄
        idx = np.flatnonzero(touch)
        est = np.where(es.ei[idx] == f, o_ret[es.ej[idx]] + es.dbar[idx],
                       o_ret[es.ei[idx]] - es.dbar[idx])
        wt = es.w[idx]
        o_hat[f] = float((wt * est).sum() / wt.sum())
        n_used[f] = idx.size
    return o_hat, np.abs(o_hat - np.asarray(o_full, dtype=np.float64)), n_used


# --------------------------------------------------------------------------- graph holes (§4 end)
def idw_predict(lon, lat, o, known: np.ndarray, target: np.ndarray, k: int = 6,
                power: float = 2.0) -> np.ndarray:
    """Inverse-distance offset prediction from the k nearest KNOWN frames (PLAN_FBuild §4 end).

    P1 verified the 907-frame graph is a single component, so this should never fire; it exists
    because Stage-A/B holes can strand a frame, and a stranded frame must get a flagged estimate
    rather than a silent o=0 (which is a real level error, not a neutral choice).
    """
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)
    o = np.asarray(o, float)
    out = np.full(lon.size, np.nan)
    ki = np.flatnonzero(known)
    if ki.size == 0:
        return out
    for t in np.flatnonzero(target).tolist():
        d = np.hypot(lon[ki] - lon[t], lat[ki] - lat[t])
        if not np.isfinite(d).any():
            continue
        near = ki[np.argsort(d)[:k]]
        dd = np.hypot(lon[near] - lon[t], lat[near] - lat[t])
        if (dd < 1e-9).any():
            out[t] = float(o[near[np.argmin(dd)]])
            continue
        w = dd ** (-power)
        out[t] = float((w * o[near]).sum() / w.sum())
    return out


def patch_graph_holes(o: np.ndarray, comp: np.ndarray, deg: np.ndarray, lon, lat,
                      k: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Make offsets comparable across components; returns (offsets, provenance labels).

    `solved`            in the largest component — the gauge everything else is expressed against.
    `component_gauged`  in a smaller component: the component's internal offsets are trustworthy but
                        its gauge is free, so shift the WHOLE component by the median IDW residual
                        against the main component (never re-solve; that would mix gauges).
    `interpolated`      isolated frame with no overlap at all: pure IDW.
    """
    o = np.asarray(o, float).copy()
    src = np.array(["solved"] * o.size, dtype=object)
    sizes = np.bincount(comp)
    main = int(np.argmax(sizes))
    known = (comp == main) & (deg > 0)
    for c in np.unique(comp):
        if c == main:
            continue
        m = comp == c
        pred = idw_predict(lon, lat, o, known, m, k=k)
        if m.sum() == 1 and deg[m][0] == 0:
            o[m] = np.where(np.isfinite(pred[m]), pred[m], 0.0)
            src[m] = "interpolated"
        else:
            shift = np.nanmedian(pred[m] - o[m])
            o[m] += 0.0 if not np.isfinite(shift) else shift
            src[m] = "component_gauged"
    return o, src


# --------------------------------------------------------------------------- trend guard (§4.2)
def design_matrix(x, y, order: int = 1) -> np.ndarray:
    """[1, x, y] (order 1) or + [x², xy, y²] (order 2) on CENTERED coordinates.

    R² of a least-squares surface fit is invariant to any affine rescaling of (x, y), so degrees vs
    metres is a presentation choice; centering just conditions the normal equations.
    """
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    cols = [np.ones_like(x), x, y]
    if order >= 2:
        cols += [x * x, x * y, y * y]
    return np.column_stack(cols)


def weighted_fit(A: np.ndarray, z: np.ndarray, w=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted LS fit; returns (coef, fitted, weighted R²)."""
    z = np.asarray(z, dtype=np.float64)
    w = np.ones_like(z) if w is None else np.asarray(w, dtype=np.float64)
    sw = np.sqrt(w)
    coef = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)[0]
    fitted = A @ coef
    zbar = float((w * z).sum() / w.sum())
    ss_tot = float((w * (z - zbar) ** 2).sum())
    ss_res = float((w * (z - fitted) ** 2).sum())
    return coef, fitted, (1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)


def block_labels(lon, lat, cell_deg: float = 4.0) -> np.ndarray:
    """~cell_deg spatial blocks for the permutation null (PLAN_FBuild §4.2: ~4° cells)."""
    li = np.floor(np.asarray(lon, dtype=np.float64) / cell_deg).astype(np.int64)
    lj = np.floor(np.asarray(lat, dtype=np.float64) / cell_deg).astype(np.int64)
    _, lab = np.unique(li * 100_000 + lj, return_inverse=True)
    return lab.astype(np.int32)


def block_permute(o: np.ndarray, blocks: np.ndarray, rng) -> np.ndarray:
    """One block-permutation draw: relocate whole ~4° blocks of offsets to other blocks' frames.

    Plain (unrestricted) permutation would destroy the SHORT-range autocorrelation too and give a
    null that any spatially smooth field beats — a trend test that always fires. Moving blocks
    wholesale keeps within-block structure and only kills the large-scale gradient, which is the
    thing under test. Unequal block sizes are handled by resampling the source block's values
    (without replacement where it fits).
    """
    members = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    perm = rng.permutation(len(members))
    out = np.empty_like(np.asarray(o, dtype=np.float64))
    for t, src in enumerate(perm.tolist()):
        dst, pool = members[t], np.asarray(o, dtype=np.float64)[members[src]]
        out[dst] = rng.choice(pool, size=dst.size, replace=pool.size < dst.size)
    return out


def trend_significance(o, lon, lat, w=None, order: int = 1, cell_deg: float = 4.0,
                       n_draws: int = 1000, seed: int = 0) -> dict:
    """Fit a lon/lat surface to the offsets and test its R² against a block-permutation null."""
    rng = np.random.default_rng(seed)
    A = design_matrix(lon, lat, order=order)
    coef, fitted, r2 = weighted_fit(A, o, w)
    blocks = block_labels(lon, lat, cell_deg=cell_deg)
    null = np.array([weighted_fit(A, block_permute(o, blocks, rng), w)[2] for _ in range(n_draws)])
    p = float((1 + int((null >= r2).sum())) / (1 + n_draws))       # +1: never report p = 0
    return {"order": order, "r2": float(r2), "p_value": p, "n_blocks": int(np.unique(blocks).size),
            "null_median_r2": float(np.median(null)), "null_p95_r2": float(np.percentile(null, 95)),
            "coef": coef, "fitted": fitted}


def attribution(smooth, axes: dict[str, np.ndarray], lon, lat, w=None, cell_deg: float = 4.0,
                n_draws: int = 1000, seed: int = 0) -> dict[str, dict]:
    """Weighted R² of the smooth offset field against each candidate axis, vs a block-permuted null.

    Both metadata (epoch/incidence) and geology proxies (MOLA/THEMIS) are themselves spatially
    smooth, so a naive correlation p-value is meaningless here — every axis "wins". The block
    permutation is what makes §4.3's attribution able to BIND: it asks whether an axis explains the
    smooth field better than a spatially-equivalent random relabelling does.
    """
    rng = np.random.default_rng(seed)
    smooth = np.asarray(smooth, dtype=np.float64)
    blocks = block_labels(lon, lat, cell_deg=cell_deg)
    nulls = [block_permute(smooth, blocks, rng) for _ in range(n_draws)]
    out = {}
    for name, v in axes.items():
        v = np.asarray(v, dtype=np.float64)
        ok = np.isfinite(v) & np.isfinite(smooth)
        if ok.sum() < 10:
            out[name] = {"r2": np.nan, "p_value": np.nan, "n": int(ok.sum())}
            continue
        A = np.column_stack([np.ones(int(ok.sum())), (v[ok] - v[ok].mean())])
        ww = None if w is None else np.asarray(w, dtype=np.float64)[ok]
        r2 = weighted_fit(A, smooth[ok], ww)[2]
        null = np.array([weighted_fit(A, nz[ok], ww)[2] for nz in nulls])
        out[name] = {"r2": float(r2), "p_value": float((1 + int((null >= r2).sum())) / (1 + n_draws)),
                     "n": int(ok.sum()), "null_p95_r2": float(np.percentile(null, 95))}
    return out


def group_r2(smooth, axes: dict[str, np.ndarray], names: list[str], lon, lat, w=None,
             cell_deg: float = 4.0, n_draws: int = 1000, seed: int = 0) -> dict:
    """Joint weighted R² of a GROUP of axes (metadata-side vs geology-side) + permutation p."""
    rng = np.random.default_rng(seed)
    smooth = np.asarray(smooth, dtype=np.float64)
    cols, used = [], []
    for nm in names:
        v = np.asarray(axes[nm], dtype=np.float64)
        if np.isfinite(v).sum() >= 10:
            cols.append(np.where(np.isfinite(v), v, np.nanmean(v)))
            used.append(nm)
    if not cols:
        return {"r2": np.nan, "p_value": np.nan, "axes": []}
    A = np.column_stack([np.ones(smooth.size)] + [c - c.mean() for c in cols])
    r2 = weighted_fit(A, smooth, w)[2]
    blocks = block_labels(lon, lat, cell_deg=cell_deg)
    null = np.array([weighted_fit(A, block_permute(smooth, blocks, rng), w)[2] for _ in range(n_draws)])
    return {"r2": float(r2), "p_value": float((1 + int((null >= r2).sum())) / (1 + n_draws)),
            "axes": used, "null_p95_r2": float(np.percentile(null, 95))}


# --- the §4.3 verdict, encoded BEFORE any build offsets are seen ---------------------------------
ATTR_ALPHA = 0.05      # permutation significance for "the smooth field tracks this axis"
ATTR_MARGIN = 0.05     # R² margin one side must win by for the verdict to be called (not ambiguous)


def trend_verdict(trend: dict, meta: dict, geo: dict, alpha: float = ATTR_ALPHA,
                  margin: float = ATTR_MARGIN) -> dict:
    """PLAN_FBuild §4.3 + §0.1 guard 1, as a rule table (no post-hoc discretion).

    NO_TREND        smooth surface not significant vs the block-permutation null → full offsets.
    FULL            smooth field tracks epoch/incidence METADATA and beats geology by `margin`
                    → artifact; apply full offsets (the pilot's F02 lesson).
    RESIDUAL_ONLY   smooth field tracks GEOLOGY proxies and beats metadata by `margin` → this is
                    §0.1 HARD-ABORT guard 1: mandated fallback to residual-only offsets.
    AMBIGUOUS       neither side wins by `margin`. §4.3 says default to full offsets + ship the
                    smooth field as an H6 diagnostic; §0.1 guard 1 says an ambiguous verdict must
                    NOT silently become full offsets. Both are honoured by refusing to auto-apply:
                    Stage D emits full AND residual-only composites and the call goes to Brian
                    (PLAN_FBuild §7 Q3) with the dense-graph evidence attached.

    **The winning side must beat the other by `margin` on R², regardless of whether the loser cleared
    `alpha`** (Brian, 2026-07-29). Until then the rule also fired on `not other_sig`, which let a
    side win while holding the LOWER R²: the 906-frame run returned FULL on metadata R²=0.108 vs
    geology R²=0.142 purely because geology's permutation p landed at 0.0579 instead of ≤0.05 — a
    margin of 8 draws in 1000, and 19-FULL/1-AMBIGUOUS across 20 seeds. Requiring the margin makes
    the verdict seed-stable and matches PLAN_FBuild §4.3's table as written. A side whose R² is NaN
    is treated as *unavailable* (proxy rasters missing), not as "lost": that cannot trigger guard 1,
    so an unavailable geology side leaves a significant metadata side applying full offsets.
    """
    if trend["p_value"] > alpha:
        return {"verdict": "NO_TREND", "apply": "full", "needs_ruling": False,
                "why": f"smooth surface R²={trend['r2']:.3f} not significant "
                       f"(p={trend['p_value']:.3f} > {alpha}) — no trend to guard against"}
    m_sig = np.isfinite(meta.get("p_value", np.nan)) and meta["p_value"] <= alpha
    g_sig = np.isfinite(geo.get("p_value", np.nan)) and geo["p_value"] <= alpha
    m_r2, g_r2 = float(meta.get("r2", np.nan)), float(geo.get("r2", np.nan))
    m_ok, g_ok = np.isfinite(m_r2), np.isfinite(g_r2)
    tail = (f"metadata R²={m_r2:.3f} (p={meta.get('p_value', float('nan')):.3f}) vs "
            f"geology R²={g_r2:.3f} (p={geo.get('p_value', float('nan')):.3f}); "
            f"smooth surface R²={trend['r2']:.3f} (p={trend['p_value']:.3f})")
    ambiguous = {"verdict": "AMBIGUOUS", "apply": "full_pending_ruling", "needs_ruling": True,
                 "why": f"neither side wins by {margin} R² — Brian's call (§7 Q3); "
                        f"ship both composites + the smooth field as an H6 diagnostic. {tail}"}
    if not m_ok and not g_ok:                       # neither side measurable → nothing to attribute
        return {**ambiguous, "why": f"no attribution axes available — cannot bind §4.3. {tail}"}
    if not g_ok:                                    # geology unavailable → guard 1 cannot trigger
        return ({"verdict": "FULL", "apply": "full", "needs_ruling": False,
                 "why": f"METADATA-dominant, geology proxies unavailable. {tail}"} if m_sig
                else ambiguous)
    if not m_ok:
        return ({"verdict": "RESIDUAL_ONLY", "apply": "residual", "needs_ruling": False,
                 "why": f"GEOLOGY-dominant, metadata axes unavailable — §0.1 guard 1. {tail}"}
                if g_sig else ambiguous)
    if g_sig and g_r2 > m_r2 + margin:
        return {"verdict": "RESIDUAL_ONLY", "apply": "residual", "needs_ruling": False,
                "why": f"GEOLOGY-dominant — §0.1 guard 1 mandates residual-only. {tail}"}
    if m_sig and m_r2 > g_r2 + margin:
        return {"verdict": "FULL", "apply": "full", "needs_ruling": False,
                "why": f"METADATA-dominant — smooth field is artifact-side. {tail}"}
    return ambiguous
