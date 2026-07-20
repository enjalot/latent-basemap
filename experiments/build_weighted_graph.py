"""Upgrade a uniform-weight neighbor graph to real UMAP fuzzy (weighted) edges.

The basemap trainer's single best quality lever is *weighted edge sampling*:
drawing neighbor-graph edges in proportion to UMAP's fuzzy membership weights
instead of uniformly. But the ship-scale graphs (``edges_30m_k15.npz``,
``edges_150m_k15.npz``) carry constant 1/k weights (built on Modal with FAISS
IVF_PQ topology + uniform weights). This tool takes such a uniform graph plus
the embeddings it indexes and rebuilds it with genuine fuzzy weights, on ONE
RTX 5090, streamed and resumable, at 30M / 150M / 405M scale.

Design (Path A — weights on the existing topology):

  Phase A  (GPU)  For each node's k neighbors (already in the npz, source-sorted,
                  exactly k per node), gather both endpoint vectors from the fp16
                  shards, recompute EXACT cosine distances on GPU, sort each
                  node's neighbors ascending, prepend a self-column (index=i,
                  dist=0). Emit per-node-chunk forward edges to staged .npz files.
  Phase B  (CPU)  Partition the forward directed edges by a symmetric hash of the
                  unordered pair {i,j} into P on-disk buckets. Both (i,j) and
                  (j,i) land in the same bucket, so the t-conorm join is
                  bucket-local.
  Phase C  (CPU)  Per bucket, apply the probabilistic t-conorm
                  W = W + Wᵀ − W∘Wᵀ (umap set_op_mix_ratio=1.0), emitting BOTH
                  directed edges for every pair with a membership in either
                  direction — exactly what ``fuzzy_simplicial_set(...).tocoo()``
                  produces.
  Phase D  (CPU)  Assemble: a single ``.npz`` (30M) or sharded ``part-*.npz`` +
                  ``index.json`` (150M), schema-identical to the input so the
                  trainer needs zero changes, plus a provenance manifest.

Correctness: the per-row math (rho / sigma / membership) is umap-learn's own
``smooth_knn_dist`` and ``compute_membership_strengths`` applied per node-chunk,
so it matches the ``fuzzy_simplicial_set`` oracle by construction. The custom
code — GPU distance recompute, self-column assembly, and the partitioned
symmetrization join — is what the V1 battery validates end-to-end.

The existing topology has k=15 REAL neighbors per node (self excluded). UMAP's
convention is n_neighbors columns INCLUDING self, so the faithful reproduction
prepends a self-column and uses n_neighbors = k+1 = 16 (target = log2(16)); this
matches how the k=50 reference artifact was built (fuzzy_simplicial_set with
n_neighbors=50 -> 49 real neighbors). See docs/WEIGHTED_GRAPH_VALIDATION.md.

Usage:
    uv run python experiments/build_weighted_graph.py build \
        --edges /data/checkpoints/pumap/edges_30m_k15.npz \
        --embeddings-list <sorted shard list or dir globs> \
        --workdir /data/checkpoints/pumap/_wg_30m \
        --out /data/checkpoints/pumap/edges_30m_k15_fuzzy.npz

    uv run python experiments/build_weighted_graph.py validate-v1 --n 100000 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.output_safety import (atomic_save_new_npz, atomic_write_new_json,
                                    refuse_existing)

# umap-learn's own row-independent kernels — the correctness oracle itself.
from umap.umap_ import smooth_knn_dist, compute_membership_strengths

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_weighted_graph")

# splitmix64 finalizer constants — a good integer hash for pair-bucketing.
_M1 = np.uint64(0xff51afd7ed558ccd)
_M2 = np.uint64(0xc4ceb9fe1a85ec53)
_S = np.uint64(33)


# --------------------------------------------------------------------------- #
# Embeddings: a deterministic, memmap-backed, cross-shard gather.
# --------------------------------------------------------------------------- #
def open_shard_2d(path: str, dim: int, raw_dtype=None):
    """Open a shard as a 2-D (rows, dim) memmap. Auto-detects a real ``.npy``
    (with NUMPY magic) vs a RAW headerless float buffer — the on-disk 405M
    corpora are raw normalised fp32 buffers stored with a ``.npy`` extension."""
    with open(path, "rb") as fh:
        magic = fh.read(6)
    if magic == b"\x93NUMPY":
        mm = np.load(path, mmap_mode="r")
        if mm.ndim != 2 or mm.shape[1] != dim:
            raise ValueError(f"npy shard {path} shape {mm.shape} != (*, {dim})")
        return mm
    dt = np.dtype(raw_dtype or "<f4")
    nbytes = os.path.getsize(path)
    per_row = dim * dt.itemsize
    if nbytes % per_row != 0:
        raise ValueError(f"raw shard {path} size {nbytes} not divisible by "
                         f"{dim}*{dt.itemsize} — wrong dtype/dim?")
    return np.memmap(path, dtype=dt, mode="r").reshape(-1, dim)


class ShardedEmbeddings:
    """Row-addressable view over an ordered list of 2-D shards.

    Row indices are global and contiguous across shards in the given order —
    that order MUST equal the graph's node-id order (row i of the graph indexes
    row i here). Nothing >= a shard is ever materialised: ``gather`` reads only
    the requested rows via memmap fancy-indexing, grouped per shard. Shards may
    be capped (``take`` < full length) so a corpus can contribute exactly its
    first N rows (the 150M mix uses the first 50M rows of each corpus)."""

    def __init__(self, paths: Optional[List[str]] = None,
                 expected_dim: Optional[int] = None, raw_dtype=None,
                 specs=None):
        if specs is None:
            if not paths:
                raise ValueError("no embedding shards given")
            dim = expected_dim
            specs = []
            for p in paths:
                mm = open_shard_2d(p, dim if dim else 384, raw_dtype)
                if dim is None:
                    dim = int(mm.shape[1])
                specs.append((mm, int(mm.shape[0])))
        self.memmaps = [mm for mm, _ in specs]
        takes = [int(t) for _, t in specs]
        self.dim = int(self.memmaps[0].shape[1])
        if expected_dim is not None and self.dim != expected_dim:
            raise ValueError(f"shard dim {self.dim} != {expected_dim}")
        self.sizes = np.asarray(takes, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)]).astype(np.int64)
        self.n = int(self.offsets[-1])
        self.dtype = self.memmaps[0].dtype

    @classmethod
    def from_corpora(cls, corpora, dim=384, raw_dtype="<f4"):
        """Build a view over the first ``n_take`` rows of each corpus dir, in the
        given corpus order. ``corpora`` = list of ``(dir, n_take)``; within each
        dir the ``*.npy`` shards are used in sorted order (deterministic)."""
        specs = []
        for d, n_take in corpora:
            files = sorted(glob.glob(os.path.join(d, "*.npy")))
            if not files:
                raise ValueError(f"no shards in {d}")
            remaining = int(n_take)
            for f in files:
                if remaining <= 0:
                    break
                mm = open_shard_2d(f, dim, raw_dtype)
                take = min(int(mm.shape[0]), remaining)
                specs.append((mm, take))
                remaining -= take
            if remaining > 0:
                raise ValueError(f"{d} has fewer than {n_take} rows "
                                 f"(short by {remaining})")
        return cls(specs=specs, expected_dim=dim)

    def __len__(self):
        return self.n

    def gather(self, ids: np.ndarray, out_dtype=np.float16) -> np.ndarray:
        """Return rows for arbitrary global ids as an (len(ids), dim) array.

        Preserves the order of ``ids``. Groups reads by shard so each shard's
        memmap is fancy-indexed once.
        """
        ids = np.asarray(ids, dtype=np.int64)
        out = np.empty((ids.shape[0], self.dim), dtype=out_dtype)
        shard = np.searchsorted(self.offsets, ids, side="right") - 1
        if shard.min() < 0 or shard.max() >= len(self.memmaps):
            raise IndexError("global id out of range for the shard set")
        order = np.argsort(shard, kind="stable")
        shard_sorted = shard[order]
        ids_sorted = ids[order]
        # boundaries between shard groups in the sorted order
        bounds = np.flatnonzero(np.diff(shard_sorted)) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(order)]])
        for a, b in zip(starts, ends):
            si = int(shard_sorted[a])
            local = ids_sorted[a:b] - self.offsets[si]
            # Read in ascending local-row order so the memmap is touched
            # near-sequentially (NVMe sequential ≫ random 4K) — the dominant
            # cost when the working set exceeds RAM (150M/405M). Unscatter after.
            iloc = np.argsort(local, kind="stable")
            rows = np.asarray(self.memmaps[si][local[iloc]])
            dst = order[a:b][iloc]
            out[dst] = rows.astype(out_dtype, copy=False)
        return out


def resolve_shard_paths(embeddings_list: Optional[List[str]],
                        embeddings_dirs: Optional[List[str]]) -> List[str]:
    """Build the ORDERED shard list. Explicit --embeddings-list wins (order as
    given); otherwise sort files within each --embeddings-dir and concatenate
    dirs in the order given (deterministic — glob order is arbitrary)."""
    if embeddings_list:
        paths = []
        for entry in embeddings_list:
            if any(ch in entry for ch in "*?["):
                paths.extend(sorted(glob.glob(entry)))
            else:
                paths.append(entry)
        return paths
    if embeddings_dirs:
        paths = []
        for d in embeddings_dirs:
            files = sorted(glob.glob(os.path.join(d, "*.npy")))
            if not files:
                raise ValueError(f"no .npy shards in {d}")
            paths.extend(files)
        return paths
    raise ValueError("provide --embeddings-list or --embeddings-dir")


# --------------------------------------------------------------------------- #
# Topology loading + validation (the reshape assumption).
# --------------------------------------------------------------------------- #
def load_topology(edges_path: str):
    """Return (neighbors[n,k] int32, n_nodes, k, nprobe). Requires the npz to be
    source-sorted with EXACTLY k directed edges per node (verified), so the
    target column reshapes cleanly to per-node neighbor lists."""
    npz = np.load(edges_path, mmap_mode="r")
    sources = npz["sources"]
    targets = npz["targets"]
    n_nodes = int(np.asarray(npz["n_nodes"]))
    k = int(np.asarray(npz["k"])) if "k" in npz.files else None
    nprobe = int(np.asarray(npz["nprobe"])) if "nprobe" in npz.files else None
    n_edges = int(sources.shape[0])
    if k is None:
        if n_edges % n_nodes != 0:
            raise ValueError("cannot infer k: n_edges not divisible by n_nodes")
        k = n_edges // n_nodes
    if n_edges != n_nodes * k:
        raise ValueError(f"expected {n_nodes*k} edges (n*k), found {n_edges}")
    # Verify the reshape assumption on a sample of node-rows spread across the
    # array (checking all n rows would touch the whole array; a spread sample
    # catches any block that is not exactly-k-per-source-sorted).
    S = np.asarray(sources)
    probe = np.unique(np.concatenate([
        np.arange(0, min(2_000_000, n_edges)),
        np.linspace(0, n_edges - 1, 200_000).astype(np.int64),
    ]))
    node_of = probe // k
    if not np.array_equal(S[probe], node_of.astype(S.dtype)):
        raise ValueError("topology is NOT source-sorted exactly-k-per-node; "
                         "reshape(n,k) invalid — this builder needs that layout")
    neighbors = np.asarray(targets).reshape(n_nodes, k)
    return neighbors, n_nodes, k, nprobe


# --------------------------------------------------------------------------- #
# GPU: exact cosine distances for a chunk of nodes, sorted, self prepended.
# --------------------------------------------------------------------------- #
def chunk_knn_with_self(emb: ShardedEmbeddings, neighbors: np.ndarray,
                        node_lo: int, node_hi: int, device, torch):
    """Build (knn_indices, knn_dists) of shape (m, k+1) for nodes [lo,hi):
    column 0 = self (index=node, dist=0), columns 1..k = the node's k neighbors
    sorted ascending by EXACT cosine distance. Distances computed on ``device``."""
    import torch.nn.functional as F
    m = node_hi - node_lo
    k = neighbors.shape[1]
    nbr_block = neighbors[node_lo:node_hi]                       # (m, k) int32
    self_ids = np.arange(node_lo, node_hi, dtype=np.int64)

    self_vec = emb.gather(self_ids, out_dtype=np.float16)        # (m, d)
    nbr_vec = emb.gather(nbr_block.reshape(-1), out_dtype=np.float16)  # (m*k, d)

    sv = torch.from_numpy(self_vec).to(device).float()
    nv = torch.from_numpy(nbr_vec).to(device).float().view(m, k, -1)
    sv = F.normalize(sv, dim=1)
    nv = F.normalize(nv, dim=2)
    cos = (sv.unsqueeze(1) * nv).sum(dim=2)                      # (m, k)
    dist = (1.0 - cos).clamp_(min=0.0)                          # cosine distance

    order = torch.argsort(dist, dim=1)                          # ascending
    dist_sorted = torch.gather(dist, 1, order).cpu().numpy().astype(np.float32)
    nbr_t = torch.from_numpy(nbr_block.astype(np.int64)).to(device)
    nbr_sorted = torch.gather(nbr_t, 1, order).cpu().numpy().astype(np.int32)

    knn_dists = np.zeros((m, k + 1), dtype=np.float32)
    knn_dists[:, 1:] = dist_sorted
    knn_indices = np.empty((m, k + 1), dtype=np.int32)
    knn_indices[:, 0] = self_ids.astype(np.int32)
    knn_indices[:, 1:] = nbr_sorted
    return knn_indices, knn_dists


def fuzzy_directed_from_knn(knn_indices: np.ndarray, knn_dists: np.ndarray,
                            n_neighbors: float, local_connectivity: float = 1.0):
    """Directed membership edges (self/zeros dropped) for a chunk of nodes whose
    GLOBAL ids sit in ``knn_indices[:, 0]`` (the self-column).

    sigma/rho come from umap-learn's own ``smooth_knn_dist`` (id-independent —
    it reads distances only and treats column 0 as the smallest, which our self
    column with distance 0 satisfies). The membership emission is done here
    (not via umap's ``compute_membership_strengths``) because that function
    derives the source id and the self-mask from the POSITIONAL row index, which
    is only the global id when a chunk starts at node 0. We instead key both off
    the real global id in ``knn_indices[:, 0]`` so any chunk offset is correct.
    The per-value formula and edge-case order (self → 0; d≤0 or sigma==0 → 1;
    else exp(-(d)/sigma); missing (-1) → 0; eliminate zeros) match umap exactly;
    validated against ``compute_membership_strengths`` in the tests and V1."""
    knn_dists = knn_dists.astype(np.float32)
    sigmas, rhos = smooth_knn_dist(knn_dists, float(n_neighbors),
                                   local_connectivity=float(local_connectivity))
    ki = knn_indices.astype(np.int64)
    row_ids = ki[:, 0]                                      # global self id/row
    d = knn_dists.astype(np.float64) - rhos[:, None].astype(np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        safe_sig = np.where(sigmas[:, None] == 0.0, 1.0,
                            sigmas[:, None]).astype(np.float64)
        val = np.exp(-(d / safe_sig))
    val = np.where(d <= 0.0, 1.0, val)                     # d≤0 → 1
    val = np.where((sigmas[:, None] == 0.0) & (d > 0.0), 1.0, val)  # sigma==0 → 1
    val = np.where(ki == -1, 0.0, val)                     # missing neighbor
    val = np.where(ki == row_ids[:, None], 0.0, val)       # self → 0
    val = val.astype(np.float32)
    rows = np.repeat(row_ids, ki.shape[1])
    cols = ki.reshape(-1)
    vals = val.reshape(-1)
    mask = vals > 0.0                                       # umap eliminate_zeros
    n_rho_zero = int((rhos == 0.0).sum())
    return (rows[mask].astype(np.int32), cols[mask].astype(np.int32),
            vals[mask].astype(np.float32), sigmas, rhos, n_rho_zero)


# --------------------------------------------------------------------------- #
# Phase A — staged forward directed edges (GPU).
# --------------------------------------------------------------------------- #
def phase_a_forward_edges(emb, neighbors, n_nodes, k, n_neighbors, workdir,
                          chunk_size, device_str, yield_seconds=None):
    import torch
    device = torch.device(device_str)
    fwd_dir = os.path.join(workdir, "fwd")
    os.makedirs(fwd_dir, exist_ok=True)
    n_chunks = (n_nodes + chunk_size - 1) // chunk_size
    stats = {"weight_sum": 0.0, "weight_sq": 0.0, "n_edges": 0,
             "w_min": np.inf, "w_max": -np.inf, "n_rho_zero": 0,
             "sigma_sum": 0.0, "rho_sum": 0.0, "n_rows": 0}
    t_yield = time.time()
    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n_nodes)
        out_path = os.path.join(fwd_dir, f"chunk-{ci:05d}.npz")
        done = out_path + ".done"
        if os.path.exists(done):
            continue
        t0 = time.time()
        knn_i, knn_d = chunk_knn_with_self(emb, neighbors, lo, hi, device, torch)
        rows, cols, vals, sig, rho, nrz = fuzzy_directed_from_knn(
            knn_i, knn_d, n_neighbors)
        # stage a complete file, then mark done (crash-safe / resumable).
        # NB: write through a handle — np.savez appends '.npz' to bare paths.
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, sources=rows, targets=cols, weights=vals)
        os.replace(tmp, out_path)
        with open(done, "w") as fh:
            fh.write(f"{len(rows)}\n")
        stats["n_edges"] += int(len(vals))
        stats["weight_sum"] += float(vals.astype(np.float64).sum())
        stats["weight_sq"] += float((vals.astype(np.float64) ** 2).sum())
        stats["w_min"] = min(stats["w_min"], float(vals.min()) if len(vals) else np.inf)
        stats["w_max"] = max(stats["w_max"], float(vals.max()) if len(vals) else -np.inf)
        stats["n_rho_zero"] += nrz
        stats["sigma_sum"] += float(sig.astype(np.float64).sum())
        stats["rho_sum"] += float(rho.astype(np.float64).sum())
        stats["n_rows"] += (hi - lo)
        if ci % 10 == 0 or ci == n_chunks - 1:
            log.info("phase A chunk %d/%d nodes[%d:%d] %d fwd edges %.2fs",
                     ci, n_chunks, lo, hi, len(vals), time.time() - t0)
        if yield_seconds and (time.time() - t_yield) > yield_seconds:
            log.info("phase A: %.0fs elapsed since last yield checkpoint "
                     "(resumable at chunk %d)", time.time() - t_yield, ci + 1)
            t_yield = time.time()
    with open(os.path.join(workdir, "phase_a_stats.json"), "w") as fh:
        json.dump(stats, fh)
    log.info("phase A done: %d forward directed edges staged", stats["n_edges"])
    return fwd_dir, stats


# --------------------------------------------------------------------------- #
# Phase B — partition forward edges by a symmetric pair-hash to disk.
# --------------------------------------------------------------------------- #
_REC = np.dtype([("s", "<i4"), ("t", "<i4"), ("w", "<f4")])


def _pair_bucket(a: np.ndarray, b: np.ndarray, n_nodes: int, P: int) -> np.ndarray:
    """splitmix64 hash of the canonical (min,max) pair key -> bucket in [0,P)."""
    key = a.astype(np.uint64) * np.uint64(n_nodes) + b.astype(np.uint64)
    x = key ^ (key >> _S)
    x = x * _M1
    x = x ^ (x >> _S)
    x = x * _M2
    x = x ^ (x >> _S)
    return (x % np.uint64(P)).astype(np.int64)


def phase_b_partition(fwd_dir, workdir, n_nodes, P):
    part_final = os.path.join(workdir, "parts")
    if os.path.exists(os.path.join(part_final, "_DONE")):
        log.info("phase B already complete")
        return part_final
    part_tmp = os.path.join(workdir, "parts.tmp")
    if os.path.exists(part_tmp):
        import shutil
        shutil.rmtree(part_tmp)
    os.makedirs(part_tmp)
    handles = [open(os.path.join(part_tmp, f"p{p:04d}.bin"), "wb") for p in range(P)]
    try:
        chunk_files = sorted(glob.glob(os.path.join(fwd_dir, "chunk-*.npz")))
        for fi, cf in enumerate(chunk_files):
            z = np.load(cf)
            s = z["sources"]; t = z["targets"]; w = z["weights"]
            a = np.minimum(s, t); b = np.maximum(s, t)
            buckets = _pair_bucket(a, b, n_nodes, P)
            order = np.argsort(buckets, kind="stable")
            bs = buckets[order]
            rec = np.empty(len(order), dtype=_REC)
            rec["s"] = s[order]; rec["t"] = t[order]; rec["w"] = w[order]
            edges = np.flatnonzero(np.diff(bs)) + 1
            starts = np.concatenate([[0], edges])
            ends = np.concatenate([edges, [len(order)]])
            for a0, b0 in zip(starts, ends):
                p = int(bs[a0])
                handles[p].write(rec[a0:b0].tobytes())
            if fi % 20 == 0:
                log.info("phase B: partitioned %d/%d chunk files", fi, len(chunk_files))
    finally:
        for h in handles:
            h.close()
    os.replace(part_tmp, part_final)
    with open(os.path.join(part_final, "_DONE"), "w") as fh:
        fh.write(f"P={P}\n")
    log.info("phase B done: %d buckets", P)
    return part_final


# --------------------------------------------------------------------------- #
# Phase C — per-bucket probabilistic t-conorm symmetrization.
# --------------------------------------------------------------------------- #
def symmetrize_bucket(s, t, w, n_nodes):
    """W + Wᵀ − W∘Wᵀ, restricted to this bucket's pairs. Emits BOTH directed
    edges for every pair with a membership in either direction, in (src,dst)
    ascending order. Vectorised."""
    N = np.uint64(n_nodes)
    fkey = s.astype(np.uint64) * N + t.astype(np.uint64)
    order = np.argsort(fkey, kind="stable")
    fkey_s = fkey[order]
    w_s = w[order].astype(np.float64)
    uniq = np.empty(len(fkey_s), dtype=bool)
    uniq[0] = True
    uniq[1:] = fkey_s[1:] != fkey_s[:-1]
    fkey_u = fkey_s[uniq]
    w_u = w_s[uniq]
    rev = t.astype(np.uint64) * N + s.astype(np.uint64)
    U = np.union1d(fkey_u, np.unique(rev))          # all directed keys in sym graph
    ui = (U // N).astype(np.int64)
    uj = (U % N).astype(np.int64)

    def lookup(keys):
        idx = np.searchsorted(fkey_u, keys)
        idx_c = np.clip(idx, 0, len(fkey_u) - 1)
        hit = fkey_u[idx_c] == keys
        return np.where(hit, w_u[idx_c], 0.0)

    w_ij = lookup(U)
    w_ji = lookup(uj.astype(np.uint64) * N + ui.astype(np.uint64))
    sym = w_ij + w_ji - w_ij * w_ji
    return ui.astype(np.int32), uj.astype(np.int32), sym.astype(np.float32)


def _join_one_bucket(job):
    """Join a single bucket -> output shard + .done meta. Module-level so it is
    picklable for the multiprocessing pool. Skips work if already done."""
    p, part_dir, out_dir, n_nodes = job
    out_path = os.path.join(out_dir, f"part-{p:04d}.npz")
    done = out_path + ".done"
    if os.path.exists(done):
        with open(done) as fh:
            return json.load(fh)
    rec = np.fromfile(os.path.join(part_dir, f"p{p:04d}.bin"), dtype=_REC)
    if len(rec) == 0:
        ui = np.empty(0, np.int32); uj = np.empty(0, np.int32)
        sym = np.empty(0, np.float32)
    else:
        ui, uj, sym = symmetrize_bucket(rec["s"], rec["t"], rec["w"], n_nodes)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, sources=ui, targets=uj, weights=sym)
    os.replace(tmp, out_path)
    m = {"n": int(len(sym)),
         "wsum": float(sym.astype(np.float64).sum()) if len(sym) else 0.0,
         "wsq": float((sym.astype(np.float64) ** 2).sum()) if len(sym) else 0.0,
         "wmin": float(sym.min()) if len(sym) else np.inf,
         "wmax": float(sym.max()) if len(sym) else -np.inf}
    with open(done, "w") as fh:
        json.dump(m, fh)
    return m


def phase_c_join(part_dir, workdir, n_nodes, P, workers=1):
    out_dir = os.path.join(workdir, "out_parts")
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(p, part_dir, out_dir, n_nodes) for p in range(P)]
    metas = [None] * P
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(min(workers, P)) as pool:
            for i, m in enumerate(pool.imap_unordered(_join_one_bucket, jobs)):
                metas[i] = m
                if i % 16 == 0 or i == P - 1:
                    log.info("phase C: %d/%d buckets joined (%d workers)",
                             i + 1, P, workers)
    else:
        for p in range(P):
            metas[p] = _join_one_bucket(jobs[p])
            if p % 16 == 0 or p == P - 1:
                log.info("phase C: joined bucket %d/%d (%d out edges)",
                         p, P, metas[p]["n"])
    counts = [m["n"] for m in metas]
    wsum = float(sum(m["wsum"] for m in metas))
    wsq = float(sum(m["wsq"] for m in metas))
    wmin = min(m["wmin"] for m in metas)
    wmax = max(m["wmax"] for m in metas)
    total = int(sum(counts))
    summary = {"n_edges": total, "weight_sum": wsum, "weight_sq": wsq,
               "w_min": wmin, "w_max": wmax, "counts": counts}
    with open(os.path.join(workdir, "phase_c_stats.json"), "w") as fh:
        json.dump(summary, fh)
    log.info("phase C done: %d symmetrized directed edges", total)
    return out_dir, summary


# --------------------------------------------------------------------------- #
# Phase D — assemble the final artifact + manifest.
# --------------------------------------------------------------------------- #
def _weight_summary(w):
    w = np.asarray(w, dtype=np.float64)
    return {
        "n": int(w.size),
        "min": float(w.min()), "max": float(w.max()),
        "mean": float(w.mean()), "median": float(np.median(w)),
        "p10": float(np.percentile(w, 10)), "p90": float(np.percentile(w, 90)),
    }


def _builder_commit():
    try:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        h = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        d = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                          capture_output=True, text=True)
        return h.stdout.strip() or None, bool(d.stdout.strip())
    except Exception:
        return None, None


def assemble_single(out_dir, P, out_path, n_nodes, k, meta, sort_output=True):
    """Concatenate bucket outputs into one npz (schema == input). Globally
    (src,dst)-sorted when sort_output (deterministic, node-contiguous)."""
    from basemap.artifact_identity import ordered_array_sha256
    srcs, tgts, wts = [], [], []
    for p in range(P):
        z = np.load(os.path.join(out_dir, f"part-{p:04d}.npz"))
        srcs.append(z["sources"]); tgts.append(z["targets"]); wts.append(z["weights"])
    sources = np.concatenate(srcs); del srcs
    targets = np.concatenate(tgts); del tgts
    weights = np.concatenate(wts); del wts
    if sort_output:
        key = sources.astype(np.uint64) * np.uint64(n_nodes) + targets.astype(np.uint64)
        order = np.argsort(key, kind="stable")
        del key
        sources = sources[order]; targets = targets[order]; weights = weights[order]
        del order
    log.info("assembling single npz: %d edges -> %s", len(weights), out_path)
    atomic_save_new_npz(out_path, compressed=True,
                        sources=sources, targets=targets, weights=weights,
                        n_nodes=np.int64(n_nodes), k=np.int64(k),
                        nprobe=np.int64(meta.get("nprobe") or -1))
    manifest = build_manifest(out_path, sources, targets, weights, n_nodes, k, meta,
                              ordered_array_sha256)
    atomic_write_new_json(out_path + ".manifest.json", manifest)
    return manifest


def assemble_sharded(out_dir, P, out_root, n_nodes, k, meta):
    """Ship bucket outputs as the artifact's shards + an index.json. Each shard
    is a valid npz (sources/targets/weights) so load_sharded_edges concatenates
    them for the trainer; format matches the single-file schema (proven by the
    30M single-file V3)."""
    from basemap.artifact_identity import sha256_file
    refuse_existing(out_root, label="sharded artifact root")
    os.makedirs(out_root)
    shards = []
    total = 0
    wsum = 0.0
    wmin = np.inf
    wmax = -np.inf
    for p in range(P):
        src = os.path.join(out_dir, f"part-{p:04d}.npz")
        z = np.load(src)
        w = z["weights"]
        dst = os.path.join(out_root, f"part-{p:04d}.npz")
        # atomic copy into the published root
        import shutil
        tmp = dst + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        shards.append({"path": f"part-{p:04d}.npz", "n_edges": int(len(w)),
                       "sha256": sha256_file(dst)})
        total += int(len(w))
        if len(w):
            wsum += float(w.astype(np.float64).sum())
            wmin = min(wmin, float(w.min())); wmax = max(wmax, float(w.max()))
    index = {"schema": "sharded_edge_list.v1", "n_nodes": int(n_nodes),
             "k": int(k), "n_shards": P, "n_edges": total,
             "weight_mean": (wsum / total if total else 0.0),
             "weight_min": wmin, "weight_max": wmax,
             "shards": shards,
             "note": "each part-*.npz has sources/targets/weights (int32/int32/"
                     "float32); concatenate in shard order for a single edge list. "
                     "load via experiments.build_weighted_graph.load_sharded_edges"}
    atomic_write_new_json(os.path.join(out_root, "index.json"), index)
    commit, dirty = _builder_commit()
    manifest = {"schema": "weighted_graph_manifest.v1", "artifact": out_root,
                "sharded": True, "n_nodes": int(n_nodes), "k": int(k),
                "metric": "cosine", "n_neighbors_param": int(k) + 1,
                "target": float(np.log2(int(k) + 1)),
                "smooth_k_tolerance": 1e-5, "max_iter": 64,
                "nprobe_inherited": meta.get("nprobe"),
                "input_edges_sha256": meta.get("input_edges_sha256"),
                "embedding_shards": meta.get("embedding_shards"),
                "n_edges": total, "weight_summary_note": "see index.json",
                "builder_commit": commit, "builder_dirty": dirty,
                "resources": meta.get("resources")}
    atomic_write_new_json(out_root + ".manifest.json", manifest)
    log.info("assembled sharded artifact: %d edges across %d shards -> %s",
             total, P, out_root)
    return index, manifest


def build_manifest(out_path, sources, targets, weights, n_nodes, k, meta,
                   ordered_array_sha256):
    commit, dirty = _builder_commit()
    return {
        "schema": "weighted_graph_manifest.v1",
        "artifact": os.path.basename(out_path),
        "sharded": False,
        "n_nodes": int(n_nodes), "k": int(k), "metric": "cosine",
        "n_neighbors_param": int(k) + 1,           # self-column prepended
        "target": float(np.log2(int(k) + 1)),
        "smooth_k_tolerance": 1e-5, "max_iter": 64, "local_connectivity": 1.0,
        "set_op_mix_ratio": 1.0,
        "nprobe_inherited": meta.get("nprobe"),
        "input_edges_path": meta.get("input_edges_path"),
        "input_edges_sha256": meta.get("input_edges_sha256"),
        "embedding_shards": meta.get("embedding_shards"),
        "n_edges": int(len(weights)),
        "weight_summary": _weight_summary(weights),
        "sha256_sources": ordered_array_sha256(sources),
        "sha256_targets": ordered_array_sha256(targets),
        "sha256_weights": ordered_array_sha256(weights),
        "builder_commit": commit, "builder_dirty": dirty,
        "resources": meta.get("resources"),
    }


def load_sharded_edges(out_root):
    """Trivial loader glue: concatenate a sharded artifact into
    (sources, targets, weights, n_nodes) — the load_edge_arrays contract."""
    with open(os.path.join(out_root, "index.json")) as fh:
        index = json.load(fh)
    srcs, tgts, wts = [], [], []
    for shard in index["shards"]:
        z = np.load(os.path.join(out_root, shard["path"]))
        srcs.append(z["sources"]); tgts.append(z["targets"]); wts.append(z["weights"])
    return (np.concatenate(srcs), np.concatenate(tgts), np.concatenate(wts),
            int(index["n_nodes"]))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def gpu_guard(require_free=True):
    """Refuse to start GPU work if another compute process holds the card."""
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True)
        procs = [l for l in out.stdout.strip().splitlines() if l.strip()]
        if procs and require_free:
            raise SystemExit(f"GPU busy ({len(procs)} compute procs): {procs}. "
                             "Yield to the research queue; re-run to resume.")
    except FileNotFoundError:
        pass


def cmd_build(args):
    t_start = time.time()
    os.makedirs(args.workdir, exist_ok=True)
    neighbors, n_nodes, k, nprobe = load_topology(args.edges)
    log.info("topology: n_nodes=%d k=%d nprobe=%s edges=%d",
             n_nodes, k, nprobe, n_nodes * k)
    from basemap.artifact_identity import sha256_file
    if args.corpus:
        corpora = []
        for spec in args.corpus:
            d, _, nt = spec.rpartition(":")
            corpora.append((d, int(nt)))
        emb = ShardedEmbeddings.from_corpora(corpora, dim=args.dim,
                                             raw_dtype=args.raw_dtype)
        shard_names = [f"{os.path.basename(d.rstrip('/'))}:{nt}" for d, nt in corpora]
    else:
        shard_paths = resolve_shard_paths(args.embeddings_list, args.embeddings_dir)
        emb = ShardedEmbeddings(shard_paths, expected_dim=args.dim,
                                raw_dtype=args.raw_dtype)
        shard_names = [os.path.join(os.path.basename(os.path.dirname(p)),
                                    os.path.basename(p)) for p in shard_paths]
    if len(emb) != n_nodes:
        raise SystemExit(f"embedding rows {len(emb)} != graph n_nodes {n_nodes} "
                         "— wrong shard set or order")
    log.info("embeddings: %d rows x %d dim across %d shards",
             len(emb), emb.dim, len(emb.memmaps))
    n_neighbors = args.target_neighbors or (k + 1)

    meta = {"nprobe": nprobe, "input_edges_path": os.path.abspath(args.edges),
            "input_edges_sha256": sha256_file(args.edges),
            "embedding_shards": shard_names}

    if not args.skip_gpu:
        gpu_guard(require_free=not args.force_gpu)
        _, a_stats = phase_a_forward_edges(
            emb, neighbors, n_nodes, k, n_neighbors, args.workdir,
            args.chunk_size, args.device, yield_seconds=args.yield_seconds)
    fwd_dir = os.path.join(args.workdir, "fwd")
    part_dir = phase_b_partition(fwd_dir, args.workdir, n_nodes, args.partitions)
    out_dir, c_stats = phase_c_join(part_dir, args.workdir, n_nodes, args.partitions,
                                    workers=args.phase_c_workers)

    peak_vram = 0.0
    try:
        import torch
        peak_vram = torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    import resource
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # GB (kb)
    meta["resources"] = {"wall_seconds": round(time.time() - t_start, 1),
                         "peak_rss_gb": round(peak_rss, 2),
                         "peak_vram_gb": round(peak_vram, 2)}

    if args.sharded:
        assemble_sharded(out_dir, args.partitions, args.out, n_nodes, k, meta)
    else:
        assemble_single(out_dir, args.partitions, args.out, n_nodes, k, meta,
                        sort_output=not args.no_sort)
    log.info("BUILD COMPLETE in %.1fs", time.time() - t_start)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build weighted graph from uniform topology")
    b.add_argument("--edges", required=True)
    b.add_argument("--embeddings-list", nargs="+", default=None,
                   help="ordered shard paths or globs (node-id order)")
    b.add_argument("--embeddings-dir", nargs="+", default=None,
                   help="dirs; *.npy sorted within each, dirs concatenated in order")
    b.add_argument("--corpus", nargs="+", default=None,
                   help="corpus mix as 'dir:n_take' (first n_take rows of each, "
                        "in order) — for the 150M fineweb/redpajama/pile mix")
    b.add_argument("--raw-dtype", default=None,
                   help="dtype for raw headerless buffers, e.g. '<f4' (405M corpora)")
    b.add_argument("--out", required=True, help="output npz (or dir if --sharded)")
    b.add_argument("--workdir", required=True, help="staging dir on /data")
    b.add_argument("--dim", type=int, default=384)
    b.add_argument("--chunk-size", type=int, default=150_000)
    b.add_argument("--partitions", type=int, default=64)
    b.add_argument("--phase-c-workers", type=int, default=1,
                   help="parallel processes for the symmetrization join (150M)")
    b.add_argument("--target-neighbors", type=int, default=None,
                   help="n_neighbors for log2 target; default k+1 (self-column)")
    b.add_argument("--device", default="cuda")
    b.add_argument("--sharded", action="store_true", help="emit part-*.npz + index")
    b.add_argument("--no-sort", action="store_true",
                   help="skip global (src,dst) sort of single-file output")
    b.add_argument("--skip-gpu", action="store_true",
                   help="resume from staged forward edges (phases B-D only)")
    b.add_argument("--force-gpu", action="store_true",
                   help="run GPU phase even if a compute process is present")
    b.add_argument("--yield-seconds", type=float, default=600.0)
    b.set_defaults(func=cmd_build)

    # validation subcommands are registered by validate module import
    from experiments import weighted_graph_validate as V
    V.register(sub)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
