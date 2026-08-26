#!/usr/bin/env python3
"""Per-point neighborhood-spread arrays for the compare page.

For every exported arm: r10[i] = 2D radius (in that map) containing the
point's 10 nearest high-D true neighbors (from the rung's saved kNN),
normalized by the map's robust radius (p90 distance from median center —
same convention as the collapse statistic). Saved as
~/.agent/basemap-maps/compare/data/<rung>/<arm>.spread.bin, float16[N].

Between two same-rung maps, log2(r10_B[i]/r10_A[i]) is then a per-point
spread(+)/compaction(-) score computable client-side: the systematic version
of "this cluster got squeezed into an island".

2m-knobs has no raw knn_indices (only the fuzzy edge npz) — a top-15
neighbor table is derived from it once and cached beside the sandbox rung.
Idempotent by coordinates mtime.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SANDBOX = Path("/data/latent-basemap/sandbox")
DATA = Path.home() / ".agent/basemap-maps/compare/data"
R0216 = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/"
             "artifacts/minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
K = 10


def knn_for_rung(rung: str):
    f = SANDBOX / rung / "knn_indices.npy"
    if f.exists():
        return np.load(f, mmap_mode="r")
    if rung == "2m-knobs":
        d = SANDBOX / rung / "knn_indices_derived.npy"
        if not d.exists():
            npz = np.load(R0216 / "edges-k15-fuzzy.npz")
            src, dst, w = npz["sources"], npz["targets"], npz["weights"]
            order = np.lexsort((-w, src))
            src, dst = src[order], dst[order]
            n = 2_000_000
            knn = np.full((n, 15), -1, dtype=np.int32)
            fill = np.zeros(n, dtype=np.int8)
            for s, t in zip(src, dst):
                f_ = fill[s]
                if f_ < 15:
                    knn[s, f_] = t
                    fill[s] = f_ + 1
            np.save(d, knn)
        return np.load(d, mmap_mode="r")
    return None


def r10_norm(xy: np.ndarray, knn) -> np.ndarray:
    n = xy.shape[0]
    out = np.empty(n, dtype=np.float32)
    center = np.median(xy, axis=0)
    R = np.percentile(np.linalg.norm(xy - center, axis=1), 90)
    for i0 in range(0, n, 200_000):
        i1 = min(n, i0 + 200_000)
        nb = np.asarray(knn[i0:i1], dtype=np.int64)
        valid = nb >= 0
        nb = np.where(valid, nb, 0)
        d = np.linalg.norm(xy[nb] - xy[i0:i1, None, :], axis=2)
        d[~valid] = np.inf
        d.sort(axis=1)
        kk = np.minimum(K - 1, valid.sum(axis=1) - 1).clip(min=0)
        out[i0:i1] = d[np.arange(i1 - i0), kk]
    return (out / max(R, 1e-9)).astype(np.float16)


def main() -> int:
    catalog = json.loads((DATA / "catalog.json").read_text())
    knn_cache: dict = {}
    n_new = 0
    for e in catalog["maps"]:
        rung, arm = e["rung"], e["arm"]
        cf = SANDBOX / rung / arm / "coordinates.npy"
        sp = DATA / rung / f"{arm}.spread.bin"
        if not cf.exists():
            continue
        if sp.exists() and sp.stat().st_mtime >= cf.stat().st_mtime:
            continue
        if rung not in knn_cache:
            knn_cache[rung] = knn_for_rung(rung)
        knn = knn_cache[rung]
        if knn is None or knn.shape[0] != e["n"]:
            continue
        xy = np.asarray(np.load(cf, mmap_mode="r"), dtype=np.float32)
        if xy.ndim != 2 or xy.shape[1] != 2:
            continue
        r10_norm(xy, knn).tofile(sp)
        n_new += 1
        print(f"{rung}/{arm}: spread written", flush=True)
    print(f"{n_new} spread arrays updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
