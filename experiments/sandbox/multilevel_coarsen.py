#!/usr/bin/env python3
"""Multilevel graph coarsening — the faithful port of upstream 0.6dev's
recursive init, stage 1 (owner order 2026-08-22: build the real thing, no
proxies).

Upstream (umap/label_prop.py): hub-seeded label propagation -> fuzzy-union
graph coarsening (ratio ~4) -> recursive coarse layouts with weakened kernels
-> expand. Attribution: random init drops upstream 0.4798 -> 0.3938, the
biggest single component.

This tool builds the level pyramid from the sealed 2M artifact:
  level 0 = the full substrate/graph; level k+1 = ratio-4 coarsening of k
  (seeds = top weighted-degree hubs, 4 label-prop sweeps, leftovers become
  singletons); supernode FEATURE = L2-normalized mean of member features (the
  parametric deviation: upstream coarsens positions, an encoder needs coarse
  INPUTS); coarse edges = weight-summed collapsed edges, self-edges dropped.
  Stops when a level has <= 40K nodes.

Output: /data/latent-basemap/substrates/multilevel-2m/level<k>/{substrate.f32.npy,
edges.npz, mapping.npy (level k-1 node -> level k supernode), meta.json}.
CPU-only (~minutes); deterministic.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

SRC = Path("/data/latent-basemap/runs/round-0216/queue-correction-3/artifacts/"
           "minilm-mixed-2m-substrate-and-exact-k15-graph-v1")
OUT = Path("/data/latent-basemap/substrates/multilevel-2m")
RATIO = 4
SWEEPS = 4
MIN_NODES = 40_000


def coarsen_once(x, src, dst, w, seed_frac=1.0 / RATIO):
    n = x.shape[0]
    deg_w = np.zeros(n)
    np.add.at(deg_w, src, w)
    n_seeds = max(1, int(round(n * seed_frac)))
    seeds = np.argpartition(-deg_w, n_seeds - 1)[:n_seeds]
    label = np.full(n, -1, dtype=np.int64)
    label[seeds] = np.arange(n_seeds)
    is_seed = np.zeros(n, dtype=bool)
    is_seed[seeds] = True

    for _ in range(SWEEPS):
        m = label[src] >= 0
        t, lab, ww = dst[m], label[src[m]], w[m]
        key = t.astype(np.int64) * n_seeds + lab
        order = np.argsort(key, kind="stable")
        key_s, w_s = key[order], ww[order]
        uniq, start = np.unique(key_s, return_index=True)
        sums = np.add.reduceat(w_s, start)
        tt = (uniq // n_seeds).astype(np.int64)
        ll = (uniq % n_seeds).astype(np.int64)
        # per target node, the label with max summed weight
        ord2 = np.lexsort((-sums, tt))
        uniq_t, first = np.unique(tt[ord2], return_index=True)
        best = ll[ord2][first]
        upd = ~is_seed[uniq_t]           # hubs keep their own label
        label[uniq_t[upd]] = best[upd]

    orphan = label < 0
    label[orphan] = n_seeds + np.arange(int(orphan.sum()))
    # compress to dense ids
    uniq_labels, label = np.unique(label, return_inverse=True)
    L = len(uniq_labels)

    # supernode features: L2-normalized mean of members
    feats = np.zeros((L, x.shape[1]), dtype=np.float64)
    np.add.at(feats, label, x.astype(np.float64))
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    feats = (feats / norms).astype(np.float32)

    # coarse edges: collapse, drop self, sum weights
    cs, ct = label[src], label[dst]
    keep = cs != ct
    key = cs[keep].astype(np.int64) * L + ct[keep]
    order = np.argsort(key, kind="stable")
    key_s, w_s = key[order], w[keep][order]
    uniq, start = np.unique(key_s, return_index=True)
    sums = np.add.reduceat(w_s, start)
    return (feats, (uniq // L).astype(np.int32), (uniq % L).astype(np.int32),
            sums.astype(np.float32), label)


def main() -> int:
    x = np.array(np.load(SRC / "substrate.f32.npy", mmap_mode="r"))
    npz = np.load(SRC / "edges-k15-fuzzy.npz")
    src, dst, w = npz["sources"].astype(np.int64), npz["targets"].astype(np.int64), \
        npz["weights"].astype(np.float64)
    level = 0
    while x.shape[0] > MIN_NODES:
        t0 = time.time()
        feats, cs, ct, cw, mapping = coarsen_once(x, src, dst, w)
        level += 1
        d = OUT / f"level{level}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "substrate.f32.npy", feats)
        np.savez(d / "edges.npz", sources=cs, targets=ct, weights=cw,
                 n_nodes=np.int64(feats.shape[0]))
        np.save(d / "mapping.npy", mapping.astype(np.int32))
        (d / "meta.json").write_text(json.dumps({
            "level": level, "nodes": int(feats.shape[0]),
            "directed_edges": int(len(cs)),
            "from_nodes": int(x.shape[0]),
            "wall_s": time.time() - t0,
        }, indent=1))
        print(f"level {level}: {x.shape[0]:,} -> {feats.shape[0]:,} nodes, "
              f"{len(cs):,} edges ({time.time()-t0:.0f}s)", flush=True)
        x, src, dst, w = feats, cs.astype(np.int64), ct.astype(np.int64), \
            cw.astype(np.float64)
    print("pyramid complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
