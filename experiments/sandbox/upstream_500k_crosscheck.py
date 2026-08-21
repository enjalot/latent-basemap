#!/usr/bin/env python3
"""500K upstream cross-check (owner request 2026-08-21): umap-learn 0.6dev on
CPU vs our 2M-trained maps, same rows, same truth, same instrument.

  subset : every 4th row of the 2M substrate -> exactly 500K, mix preserved.
  truth  : the sealed 2M exact-k15 fuzzy edges INDUCED on the subset (both
           endpoints sampled, remapped) — identical for every contender.
  contenders:
    upstream-06dev      umap-learn 0.6.0 (the 0.6dev clone), CPU, defaults
    r0265-seed42-slice  our sealed promoted 2M map, subset rows sliced
    md005-fneg-slice    the aesthetic-cross winner, subset rows sliced
  score  : quick-FFR at 0.1% of 500K (disc=500) against the induced truth,
           + a density render per contender. Cards land on the review page
           under "500K upstream cross-check".

Phases (separate envs): `prep` + `score` run in the latent-basemap .venv;
`upstream` runs in /data/latent-basemap/umap06dev-env (CPU; safe while the
GPU window owns the card). Asymmetry stated on every card: our maps saw all
2M rows in training; upstream sees only the 500K (parametric atlas vs
transductive embedder — the comparison the paper needs).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SUB = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
           "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
OUT = Path("/data/latent-basemap/sandbox/500k-crosscheck")
STRIDE = 4
ROWS_2M = 2_000_000

SLICES = {
    "r0265-seed42-slice": Path(
        "/data/latent-basemap/runs/round-0265/queue-correction-3/artifacts/"
        "minilm-mixed-2m-fneg-x4-md000-seed42-r0265-v1/coordinates.npy"),
    "md005-fneg-slice": Path(
        "/data/latent-basemap/sandbox/2m-knobs/umap-md005-x2-fneg10/"
        "coordinates.npy"),
}


def prep() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    idx = np.arange(0, ROWS_2M, STRIDE)
    n = len(idx)
    np.save(OUT / "subset_rows.npy", idx)

    X = np.load(SUB / "substrate.f32.npy", mmap_mode="r")
    np.save(OUT / "subset_x.npy", np.ascontiguousarray(X[idx]))

    npz = np.load(SUB / "edges-k15-fuzzy.npz")
    src, dst = npz["sources"], npz["targets"]
    w = npz["weights"] if "weights" in npz.files else None
    in_sub = np.zeros(ROWS_2M, dtype=bool)
    in_sub[idx] = True
    keep = in_sub[src] & in_sub[dst]
    remap = np.full(ROWS_2M, -1, dtype=np.int64)
    remap[idx] = np.arange(n)
    out = {"sources": remap[src[keep]].astype(np.int32),
           "targets": remap[dst[keep]].astype(np.int32),
           "n_nodes": np.int64(n)}
    if w is not None:
        out["weights"] = w[keep].astype(np.float32)
    np.savez(OUT / "edges-induced.npz", **out)
    print(f"subset {n:,} rows; induced edges {int(keep.sum()):,} "
          f"({keep.sum()/n:.2f} per node)")
    return 0


def upstream() -> int:
    import umap

    X = np.load(OUT / "subset_x.npy")
    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, verbose=True,
                        random_state=42)
    xy = reducer.fit_transform(X)
    wall = time.time() - t0
    d = OUT / "upstream-06dev"
    d.mkdir(exist_ok=True)
    np.save(d / "coordinates.npy", np.asarray(xy, dtype=np.float32))
    (d / "fit_info.json").write_text(json.dumps({
        "upstream": f"umap-learn {umap.__version__} (0.6dev clone, HEAD 67ca365)",
        "settings": {"n_neighbors": 15, "min_dist": 0.0, "random_state": 42,
                     "everything_else": "upstream defaults"},
        "device": "cpu",
        "wall_s": wall,
    }, indent=1))
    print(f"upstream 0.6dev: {wall/60:.1f} min")
    return 0


def score() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from knobs_2m import quick_ffr
    from map_renders import binned_counts, render_png, robust_extent

    idx = np.load(OUT / "subset_rows.npy")
    n = len(idx)
    edges = OUT / "edges-induced.npz"

    contenders: dict[str, np.ndarray] = {}
    up = OUT / "upstream-06dev/coordinates.npy"
    if up.exists():
        contenders["upstream-06dev"] = np.load(up)
    for name, path in SLICES.items():
        if path.exists():
            contenders[name] = np.asarray(
                np.load(path, mmap_mode="r")[idx], dtype=np.float32)

    for name, xy in contenders.items():
        d = OUT / name
        d.mkdir(exist_ok=True)
        ffr = quick_ffr(xy, edges, n)
        render_png(binned_counts(xy, robust_extent(xy)), d / "density.png")
        note = ("saw ONLY the 500k rows (transductive)" if name.startswith("upstream")
                else "sliced from a map trained on ALL 2M rows (parametric atlas)")
        extra = {}
        fi = d / "fit_info.json"
        if fi.exists():
            extra["wall_s"] = json.loads(fi.read_text()).get("wall_s")
        (d / "summary.json").write_text(json.dumps({
            "arm": name,
            "rung": "500k-crosscheck",
            "overrides": {"crosscheck": note},
            "seed": 42,
            "quick_ffr_at_0.1pct": float(ffr),
            **extra,
            "substrate": str(OUT / "subset_x.npy"),
            "edges": str(edges),
            "note": "500K upstream cross-check; induced 2M-exact-graph truth; "
                    "sandbox read, no sealed claim.",
        }, indent=1))
        print(f"{name}: quick-FFR {ffr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit({"prep": prep, "upstream": upstream, "score": score}[sys.argv[1]]())
