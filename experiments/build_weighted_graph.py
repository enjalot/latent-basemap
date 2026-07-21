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
                  exactly k per node), gather both endpoint vectors without
                  narrowing their stored precision, recompute EXACT cosine
                  distances on GPU, sort each
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
  Phase D  (CPU)  Assemble: a single trainer-admissible ``.npz`` (30M), or an
                  experimental sharded ``part-*.npz`` + ``index.json`` for
                  validation and future consumer work, plus provenance.

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

from basemap.artifact_identity import (canonical_json, ordered_array_sha256,
                                       sha256_bytes, sha256_file)
from basemap.output_safety import (atomic_save_new_npz, atomic_write_new_json,
                                    refuse_existing)

# umap-learn's own row-independent kernels — the correctness oracle itself.
from umap.umap_ import smooth_knn_dist

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_weighted_graph")

# splitmix64 finalizer constants — a good integer hash for pair-bucketing.
_M1 = np.uint64(0xff51afd7ed558ccd)
_M2 = np.uint64(0xc4ceb9fe1a85ec53)
_S = np.uint64(33)


def _atomic_replace_json(path: str, value: dict) -> None:
    """Crash-safe replace for mutable summaries (stage receipts are write-once)."""
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(value, fh, indent=2, sort_keys=True)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_identity(value: dict) -> str:
    return sha256_bytes(canonical_json(value))


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _validate_file_identity(path: str, *, expected_bytes: int,
                            expected_sha256: str, label: str) -> None:
    if not os.path.isfile(path) or os.path.islink(path):
        raise RuntimeError(f"{label} is missing, linked, or not a regular file: {path}")
    got_bytes = os.path.getsize(path)
    if got_bytes != int(expected_bytes):
        raise RuntimeError(
            f"{label} byte count {got_bytes} != receipt {expected_bytes}: {path}")
    got_sha = sha256_file(path)
    if got_sha != expected_sha256:
        raise RuntimeError(
            f"{label} sha256 {got_sha} != receipt {expected_sha256}: {path}")


def _publish_stage_npz(path: str, **arrays) -> None:
    """Atomically publish a fresh stage archive."""
    atomic_save_new_npz(path, compressed=False, **arrays)


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
        for i, (mm, take) in enumerate(specs):
            if mm.ndim != 2 or int(mm.shape[1]) != self.dim:
                raise ValueError(f"embedding shard {i} shape {mm.shape} is incompatible")
            if np.dtype(mm.dtype) != np.dtype(self.dtype):
                raise ValueError(
                    f"embedding shard {i} dtype {mm.dtype} != first shard {self.dtype}")
            if take < 0 or take > int(mm.shape[0]):
                raise ValueError(f"embedding shard {i} take={take} outside [0,{len(mm)}]")

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

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            index = int(key)
            if index < 0:
                index += self.n
            return self.gather(np.asarray([index], dtype=np.int64))[0]
        if isinstance(key, slice):
            start, stop, step = key.indices(self.n)
            return self.gather(np.arange(start, stop, step, dtype=np.int64))
        return self.gather(np.asarray(key, dtype=np.int64))

    def gather(self, ids: np.ndarray, out_dtype=None) -> np.ndarray:
        """Return rows for arbitrary global ids as an (len(ids), dim) array.

        Preserves the order of ``ids``. Groups reads by shard so each shard's
        memmap is fancy-indexed once. By default, preserves the stored dtype;
        callers must opt in to a narrowing conversion.
        """
        ids = np.asarray(ids, dtype=np.int64)
        if ids.ndim != 1:
            raise ValueError("embedding ids must be one-dimensional")
        if out_dtype is None:
            out_dtype = self.dtype
        out = np.empty((ids.shape[0], self.dim), dtype=out_dtype)
        if not len(ids):
            return out
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
            # Monotonic local ids avoid backward seeks, but sparse requested
            # pages remain random I/O above page-cache scale. Unscatter after.
            iloc = np.argsort(local, kind="stable")
            rows = np.asarray(self.memmaps[si][local[iloc]])
            dst = order[a:b][iloc]
            out[dst] = rows.astype(out_dtype, copy=False)
        return out

    def content_records(self) -> list[dict]:
        """Full ordered member identities used by resume and trainer admission."""
        records = []
        for ordinal, (mm, take) in enumerate(zip(self.memmaps, self.sizes)):
            path = os.path.realpath(os.fspath(mm.filename))
            records.append({
                "ordinal": ordinal,
                "canonical_path": path,
                "bytes": int(os.path.getsize(path)),
                "sha256": sha256_file(path),
                "full_rows": int(mm.shape[0]),
                "take_rows": int(take),
                "dimension": int(mm.shape[1]),
                "dtype": np.dtype(mm.dtype).str,
            })
        return records


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
def load_topology(edges_path: str, *, validation_chunk_rows: int = 250_000,
                  return_stats: bool = False):
    """Return (neighbors[n,k] int32, n_nodes, k, nprobe). Requires the npz to be
    source-sorted with EXACTLY k distinct, in-range targets per node (verified
    over the entire topology), so the target column reshapes cleanly."""
    npz = np.load(edges_path, mmap_mode="r", allow_pickle=False)
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
    if sources.ndim != 1 or targets.ndim != 1 or len(targets) != n_edges:
        raise ValueError("topology sources/targets must be equal-length 1-D arrays")
    neighbors = np.asarray(targets).reshape(n_nodes, k)
    self_slots = 0
    for lo in range(0, n_nodes, validation_chunk_rows):
        hi = min(lo + validation_chunk_rows, n_nodes)
        edge_lo, edge_hi = lo * k, hi * k
        source_chunk = np.asarray(sources[edge_lo:edge_hi])
        expected = np.repeat(np.arange(lo, hi, dtype=np.int64), k)
        if not np.array_equal(source_chunk.astype(np.int64, copy=False), expected):
            bad = int(np.flatnonzero(
                source_chunk.astype(np.int64, copy=False) != expected)[0])
            absolute = edge_lo + bad
            raise ValueError(
                "topology is not source-sorted exactly-k-per-node: "
                f"edge {absolute} has source {int(source_chunk[bad])}, "
                f"expected {int(expected[bad])}")
        block = np.asarray(neighbors[lo:hi])
        if block.dtype.kind not in "iu":
            raise ValueError(f"topology targets must be integers, got {block.dtype}")
        if block.size:
            target_min, target_max = int(block.min()), int(block.max())
            if target_min < 0 or target_max >= n_nodes:
                raise ValueError(
                    f"topology targets outside [0,{n_nodes}): "
                    f"chunk rows [{lo},{hi}) min={target_min} max={target_max}")
            sorted_targets = np.sort(block.astype(np.int64, copy=False), axis=1)
            duplicate_rows = np.flatnonzero(
                np.any(sorted_targets[:, 1:] == sorted_targets[:, :-1], axis=1))
            if len(duplicate_rows):
                row = lo + int(duplicate_rows[0])
                raise ValueError(
                    f"topology row {row} contains duplicate target ids; kNN rows "
                    "must contain k distinct neighbors")
            self_slots += int(np.count_nonzero(
                block == np.arange(lo, hi, dtype=block.dtype)[:, None]))
    stats = {
        "validation": "full_scan",
        "source_layout_valid": True,
        "target_bounds_valid": True,
        "duplicate_target_rows": 0,
        "self_slots": self_slots,
        "self_slot_fraction": self_slots / n_edges if n_edges else 0.0,
        "nodes_with_self_fraction_upper_bound": self_slots / n_nodes if n_nodes else 0.0,
    }
    result = (neighbors, n_nodes, k, nprobe)
    return (*result, stats) if return_stats else result


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

    # Promote stored values to fp32 without first narrowing fp32 corpora to fp16.
    # For materialized fp16 inputs this is exact with respect to the stored bytes.
    self_vec = emb.gather(self_ids, out_dtype=np.float32)        # (m, d)
    nbr_vec = emb.gather(nbr_block.reshape(-1), out_dtype=np.float32)  # (m*k, d)

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
def _phase_a_meta(path: str, *, contract_sha256: str, chunk_index: int,
                  node_lo: int, node_hi: int, vals, sig, rho,
                  n_rho_zero: int) -> dict:
    return {
        "schema": "weighted-graph-phase-a-receipt-v2",
        "contract_sha256": contract_sha256,
        "chunk_index": int(chunk_index),
        "node_lo": int(node_lo),
        "node_hi": int(node_hi),
        "output": {
            "path": os.path.basename(path),
            "bytes": int(os.path.getsize(path)),
            "sha256": sha256_file(path),
        },
        "stats": {
            "n_edges": int(len(vals)),
            "weight_sum": float(vals.astype(np.float64).sum()),
            "weight_sq": float((vals.astype(np.float64) ** 2).sum()),
            "w_min": float(vals.min()) if len(vals) else None,
            "w_max": float(vals.max()) if len(vals) else None,
            "n_rho_zero": int(n_rho_zero),
            "sigma_sum": float(sig.astype(np.float64).sum()),
            "rho_sum": float(rho.astype(np.float64).sum()),
            "n_rows": int(node_hi - node_lo),
        },
    }


def _validate_phase_a_meta(path: str, receipt_path: str, *,
                           contract_sha256: str, chunk_index: int,
                           node_lo: int, node_hi: int) -> dict:
    if os.path.exists(path) != os.path.exists(receipt_path):
        raise RuntimeError(
            f"incomplete phase-A checkpoint (artifact/receipt pair required): {path}")
    if not os.path.exists(path):
        return None
    receipt = _load_json(receipt_path)
    expected = {
        "schema": "weighted-graph-phase-a-receipt-v2",
        "contract_sha256": contract_sha256,
        "chunk_index": int(chunk_index),
        "node_lo": int(node_lo),
        "node_hi": int(node_hi),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"phase-A receipt {receipt_path} has {key}={receipt.get(key)!r}, "
                f"expected {value!r}")
    ident = receipt.get("output") or {}
    if ident.get("path") != os.path.basename(path):
        raise RuntimeError(f"phase-A receipt names the wrong output: {receipt_path}")
    _validate_file_identity(
        path, expected_bytes=ident.get("bytes", -1),
        expected_sha256=ident.get("sha256", ""), label="phase-A chunk")
    stats = receipt.get("stats") or {}
    if stats.get("n_rows") != node_hi - node_lo:
        raise RuntimeError(f"phase-A receipt has the wrong row count: {receipt_path}")
    return receipt


def _empty_phase_a_stats() -> dict:
    return {"weight_sum": 0.0, "weight_sq": 0.0, "n_edges": 0,
            "w_min": None, "w_max": None, "n_rho_zero": 0,
            "sigma_sum": 0.0, "rho_sum": 0.0, "n_rows": 0}


def _accumulate_phase_a_stats(total: dict, item: dict) -> None:
    for key in ("weight_sum", "weight_sq", "n_edges", "n_rho_zero",
                "sigma_sum", "rho_sum", "n_rows"):
        total[key] += item[key]
    if item["w_min"] is not None:
        total["w_min"] = (item["w_min"] if total["w_min"] is None
                          else min(total["w_min"], item["w_min"]))
    if item["w_max"] is not None:
        total["w_max"] = (item["w_max"] if total["w_max"] is None
                          else max(total["w_max"], item["w_max"]))


def phase_a_forward_edges(emb, neighbors, n_nodes, k, n_neighbors, workdir,
                          chunk_size, device_str, contract_sha256,
                          yield_seconds=None):
    import torch
    device = torch.device(device_str)
    fwd_dir = os.path.join(workdir, "fwd")
    os.makedirs(fwd_dir, exist_ok=True)
    n_chunks = (n_nodes + chunk_size - 1) // chunk_size
    stats = _empty_phase_a_stats()
    t_yield = time.time()
    for ci in range(n_chunks):
        lo = ci * chunk_size
        hi = min(lo + chunk_size, n_nodes)
        out_path = os.path.join(fwd_dir, f"chunk-{ci:05d}.npz")
        receipt_path = out_path + ".receipt.json"
        prior = _validate_phase_a_meta(
            out_path, receipt_path, contract_sha256=contract_sha256,
            chunk_index=ci, node_lo=lo, node_hi=hi)
        if prior is not None:
            _accumulate_phase_a_stats(stats, prior["stats"])
            continue
        t0 = time.time()
        knn_i, knn_d = chunk_knn_with_self(emb, neighbors, lo, hi, device, torch)
        rows, cols, vals, sig, rho, nrz = fuzzy_directed_from_knn(
            knn_i, knn_d, n_neighbors)
        _publish_stage_npz(out_path, sources=rows, targets=cols, weights=vals)
        receipt = _phase_a_meta(
            out_path, contract_sha256=contract_sha256, chunk_index=ci,
            node_lo=lo, node_hi=hi, vals=vals, sig=sig, rho=rho,
            n_rho_zero=nrz)
        atomic_write_new_json(receipt_path, receipt)
        _accumulate_phase_a_stats(stats, receipt["stats"])
        if ci % 10 == 0 or ci == n_chunks - 1:
            log.info("phase A chunk %d/%d nodes[%d:%d] %d fwd edges %.2fs",
                     ci, n_chunks, lo, hi, len(vals), time.time() - t0)
        if yield_seconds and (time.time() - t_yield) > yield_seconds:
            log.info("phase A: %.0fs elapsed since last yield checkpoint "
                     "(resumable at chunk %d)", time.time() - t_yield, ci + 1)
            t_yield = time.time()
    stats["contract_sha256"] = contract_sha256
    _atomic_replace_json(os.path.join(workdir, "phase_a_stats.json"), stats)
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


def phase_a_closure(fwd_dir: str, *, n_nodes: int, chunk_size: int,
                    contract_sha256: str) -> dict:
    """Validate every phase-A artifact/receipt and bind their ordered closure."""
    receipts = []
    n_chunks = (n_nodes + chunk_size - 1) // chunk_size
    for ci in range(n_chunks):
        lo, hi = ci * chunk_size, min((ci + 1) * chunk_size, n_nodes)
        path = os.path.join(fwd_dir, f"chunk-{ci:05d}.npz")
        receipt = _validate_phase_a_meta(
            path, path + ".receipt.json", contract_sha256=contract_sha256,
            chunk_index=ci, node_lo=lo, node_hi=hi)
        if receipt is None:
            raise RuntimeError(f"phase A is incomplete at chunk {ci}: {path}")
        receipts.append({
            "chunk_index": ci,
            "node_lo": lo,
            "node_hi": hi,
            "output": receipt["output"],
            "receipt_sha256": sha256_file(path + ".receipt.json"),
        })
    body = {
        "schema": "weighted-graph-phase-a-closure-v2",
        "contract_sha256": contract_sha256,
        "n_chunks": n_chunks,
        "members": receipts,
    }
    return {**body, "closure_sha256": _json_identity(body)}


def _validate_phase_b_receipt(part_final: str, *, P: int,
                              contract_sha256: str,
                              phase_a_closure_sha256: str) -> Optional[dict]:
    receipt_path = os.path.join(part_final, "_DONE.json")
    if not os.path.exists(part_final):
        return None
    if not os.path.isdir(part_final) or os.path.islink(part_final):
        raise RuntimeError(f"phase-B output is not a regular directory: {part_final}")
    if not os.path.exists(receipt_path):
        raise RuntimeError(
            f"phase-B directory exists without a v2 content receipt: {part_final}")
    receipt = _load_json(receipt_path)
    expected = {
        "schema": "weighted-graph-phase-b-receipt-v2",
        "contract_sha256": contract_sha256,
        "phase_a_closure_sha256": phase_a_closure_sha256,
        "partitions": int(P),
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"phase-B receipt {key}={receipt.get(key)!r}, expected {value!r}")
    buckets = receipt.get("buckets") or []
    if len(buckets) != P:
        raise RuntimeError(f"phase-B receipt has {len(buckets)} buckets, expected {P}")
    for p, member in enumerate(buckets):
        expected_name = f"p{p:04d}.bin"
        if member.get("path") != expected_name:
            raise RuntimeError(f"phase-B receipt bucket {p} names {member.get('path')}")
        _validate_file_identity(
            os.path.join(part_final, expected_name),
            expected_bytes=member.get("bytes", -1),
            expected_sha256=member.get("sha256", ""),
            label=f"phase-B bucket {p}")
    return receipt


def phase_b_partition(fwd_dir, workdir, n_nodes, P, *, contract_sha256,
                      phase_a_closure_sha256):
    part_final = os.path.join(workdir, "parts")
    prior = _validate_phase_b_receipt(
        part_final, P=P, contract_sha256=contract_sha256,
        phase_a_closure_sha256=phase_a_closure_sha256)
    if prior is not None:
        log.info("phase B already complete and content-verified")
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
    buckets = []
    for p in range(P):
        path = os.path.join(part_final, f"p{p:04d}.bin")
        buckets.append({
            "path": os.path.basename(path),
            "bytes": int(os.path.getsize(path)),
            "sha256": sha256_file(path),
            "records": int(os.path.getsize(path) // _REC.itemsize),
        })
    receipt = {
        "schema": "weighted-graph-phase-b-receipt-v2",
        "contract_sha256": contract_sha256,
        "phase_a_closure_sha256": phase_a_closure_sha256,
        "partitions": int(P),
        "buckets": buckets,
    }
    atomic_write_new_json(os.path.join(part_final, "_DONE.json"), receipt)
    log.info("phase B done: %d buckets", P)
    return part_final


# --------------------------------------------------------------------------- #
# Phase C — per-bucket probabilistic t-conorm symmetrization.
# --------------------------------------------------------------------------- #
def symmetrize_bucket(s, t, w, n_nodes):
    """W + Wᵀ − W∘Wᵀ, restricted to this bucket's pairs. Emits BOTH directed
    edges for every pair with a membership in either direction, in (src,dst)
    ascending order. Vectorised."""
    if not len(s):
        return (np.empty(0, np.int32), np.empty(0, np.int32),
                np.empty(0, np.float32))
    N = np.uint64(n_nodes)
    fkey = s.astype(np.uint64) * N + t.astype(np.uint64)
    order = np.argsort(fkey, kind="stable")
    fkey_s = fkey[order]
    w_s = w[order].astype(np.float64)
    # scipy COO->CSR (the UMAP oracle path) sums duplicate directed keys before
    # applying the t-conorm. Selecting the first duplicate silently underweights
    # a graph whose topology repeats a target.
    starts = np.r_[0, np.flatnonzero(fkey_s[1:] != fkey_s[:-1]) + 1]
    fkey_u = fkey_s[starts]
    w_u = np.add.reduceat(w_s, starts)
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
    p, part_dir, out_dir, n_nodes, contract_sha256, input_identity = job
    out_path = os.path.join(out_dir, f"part-{p:04d}.npz")
    receipt_path = out_path + ".receipt.json"
    if os.path.exists(out_path) != os.path.exists(receipt_path):
        raise RuntimeError(
            f"incomplete phase-C checkpoint (artifact/receipt pair required): {out_path}")
    if os.path.exists(receipt_path):
        prior = _load_json(receipt_path)
        expected = {
            "schema": "weighted-graph-phase-c-receipt-v2",
            "contract_sha256": contract_sha256,
            "partition": int(p),
            "input": input_identity,
        }
        for key, value in expected.items():
            if prior.get(key) != value:
                raise RuntimeError(
                    f"phase-C receipt {receipt_path} has wrong {key}")
        ident = prior.get("output") or {}
        _validate_file_identity(
            out_path, expected_bytes=ident.get("bytes", -1),
            expected_sha256=ident.get("sha256", ""),
            label=f"phase-C partition {p}")
        return prior
    input_path = os.path.join(part_dir, f"p{p:04d}.bin")
    _validate_file_identity(
        input_path, expected_bytes=input_identity["bytes"],
        expected_sha256=input_identity["sha256"],
        label=f"phase-B input bucket {p}")
    rec = np.fromfile(input_path, dtype=_REC)
    if len(rec) == 0:
        ui = np.empty(0, np.int32); uj = np.empty(0, np.int32)
        sym = np.empty(0, np.float32)
    else:
        ui, uj, sym = symmetrize_bucket(rec["s"], rec["t"], rec["w"], n_nodes)
    _publish_stage_npz(out_path, sources=ui, targets=uj, weights=sym)
    m = {
        "schema": "weighted-graph-phase-c-receipt-v2",
        "contract_sha256": contract_sha256,
        "partition": int(p),
        "input": input_identity,
        "output": {
            "path": os.path.basename(out_path),
            "bytes": int(os.path.getsize(out_path)),
            "sha256": sha256_file(out_path),
        },
        "stats": {
            "n": int(len(sym)),
            "wsum": float(sym.astype(np.float64).sum()) if len(sym) else 0.0,
            "wsq": float((sym.astype(np.float64) ** 2).sum()) if len(sym) else 0.0,
            "wmin": float(sym.min()) if len(sym) else None,
            "wmax": float(sym.max()) if len(sym) else None,
        },
    }
    atomic_write_new_json(receipt_path, m)
    return m


def phase_c_join(part_dir, workdir, n_nodes, P, *, contract_sha256, workers=1):
    out_dir = os.path.join(workdir, "out_parts")
    os.makedirs(out_dir, exist_ok=True)
    phase_b_receipt = _load_json(os.path.join(part_dir, "_DONE.json"))
    inputs = phase_b_receipt.get("buckets") or []
    if len(inputs) != P:
        raise RuntimeError("phase-B closure is incomplete before phase C")
    jobs = [(p, part_dir, out_dir, n_nodes, contract_sha256, inputs[p])
            for p in range(P)]
    metas = [None] * P
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(min(workers, P)) as pool:
            for i, m in enumerate(pool.imap_unordered(_join_one_bucket, jobs)):
                metas[int(m["partition"])] = m
                if i % 16 == 0 or i == P - 1:
                    log.info("phase C: %d/%d buckets joined (%d workers)",
                             i + 1, P, workers)
    else:
        for p in range(P):
            metas[p] = _join_one_bucket(jobs[p])
            if p % 16 == 0 or p == P - 1:
                log.info("phase C: joined bucket %d/%d (%d out edges)",
                         p, P, metas[p]["stats"]["n"])
    stage_stats = [m["stats"] for m in metas]
    counts = [m["n"] for m in stage_stats]
    wsum = float(sum(m["wsum"] for m in stage_stats))
    wsq = float(sum(m["wsq"] for m in stage_stats))
    minima = [m["wmin"] for m in stage_stats if m["wmin"] is not None]
    maxima = [m["wmax"] for m in stage_stats if m["wmax"] is not None]
    wmin = min(minima) if minima else None
    wmax = max(maxima) if maxima else None
    total = int(sum(counts))
    summary = {"n_edges": total, "weight_sum": wsum, "weight_sq": wsq,
               "w_min": wmin, "w_max": wmax, "counts": counts}
    summary["contract_sha256"] = contract_sha256
    summary["part_receipt_sha256"] = [
        sha256_file(os.path.join(out_dir, f"part-{p:04d}.npz.receipt.json"))
        for p in range(P)]
    _atomic_replace_json(os.path.join(workdir, "phase_c_stats.json"), summary)
    log.info("phase C done: %d symmetrized directed edges", total)
    return out_dir, summary


# --------------------------------------------------------------------------- #
# Phase D — assemble the final artifact + manifest.
# --------------------------------------------------------------------------- #
def _weight_summary(w, quantile_sample=1_000_000):
    """Exact count/range/mean plus bounded deterministic quantile estimates."""
    w = np.asarray(w)
    n = int(w.size)
    if not n:
        return {"n": 0, "min": None, "max": None, "mean": None,
                "quantile_sample_n": 0, "sample_median": None,
                "sample_p10": None, "sample_p90": None}
    ids = np.unique(np.linspace(0, n - 1, min(n, quantile_sample)).astype(np.int64))
    sample = w[ids].astype(np.float64, copy=False)
    return {
        "n": n,
        "min": float(w.min()), "max": float(w.max()),
        "mean": float(np.mean(w, dtype=np.float64)),
        "quantile_sample_n": int(len(sample)),
        "quantile_sample_method": "evenly-spaced-edge-indices",
        "sample_median": float(np.median(sample)),
        "sample_p10": float(np.percentile(sample, 10)),
        "sample_p90": float(np.percentile(sample, 90)),
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


def validate_weighted_output_arrays(sources, targets, weights, n_nodes,
                                    *, require_sorted: bool) -> dict:
    """Full structural scan of the final directed weighted edge list."""
    s = np.asarray(sources); t = np.asarray(targets); w = np.asarray(weights)
    if s.ndim != 1 or t.ndim != 1 or w.ndim != 1 or not (len(s) == len(t) == len(w)):
        raise ValueError("weighted output arrays must be equal-length and one-dimensional")
    if s.dtype != np.dtype("int32") or t.dtype != np.dtype("int32"):
        raise ValueError(f"weighted output endpoints must be int32, got {s.dtype}/{t.dtype}")
    if w.dtype != np.dtype("float32"):
        raise ValueError(f"weighted output weights must be float32, got {w.dtype}")
    if len(w):
        if int(s.min()) < 0 or int(t.min()) < 0 or int(s.max()) >= n_nodes or int(t.max()) >= n_nodes:
            raise ValueError("weighted output contains an out-of-range endpoint")
        if np.any(s == t):
            raise ValueError("weighted output contains self edges")
        if not np.all(np.isfinite(w)) or np.any(w <= 0) or np.any(w > 1.0 + 1e-6):
            raise ValueError("weighted output weights must be finite and in (0,1]")
    order_violations = 0
    duplicate_keys = 0
    if require_sorted and len(s) > 1:
        prior_s, prior_t = int(s[0]), int(t[0])
        chunk = 5_000_000
        for lo in range(1, len(s), chunk):
            hi = min(lo + chunk, len(s))
            cs, ct = s[lo:hi], t[lo:hi]
            ps = np.r_[np.int32(prior_s), cs[:-1]]
            pt = np.r_[np.int32(prior_t), ct[:-1]]
            order_violations += int(np.count_nonzero(
                (cs < ps) | ((cs == ps) & (ct < pt))))
            duplicate_keys += int(np.count_nonzero((cs == ps) & (ct == pt)))
            prior_s, prior_t = int(cs[-1]), int(ct[-1])
        if order_violations or duplicate_keys:
            raise ValueError(
                f"weighted output is not strictly (source,target)-sorted: "
                f"order_violations={order_violations}, duplicates={duplicate_keys}")
    return {
        "validation": "full_scan",
        "n_edges": int(len(w)),
        "endpoint_bounds_valid": True,
        "self_edges": 0,
        "weight_domain_valid": True,
        "sorted_required": bool(require_sorted),
        "order_violations": order_violations,
        "duplicate_keys": duplicate_keys,
    }


def assemble_single(out_dir, P, out_path, n_nodes, k, meta, sort_output=True):
    """Concatenate bucket outputs into one npz (schema == input). Globally
    (src,dst)-sorted when sort_output (deterministic, node-contiguous)."""
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
    meta["output_validation"] = validate_weighted_output_arrays(
        sources, targets, weights, n_nodes, require_sorted=sort_output)
    log.info("assembling single npz: %d edges -> %s", len(weights), out_path)
    atomic_save_new_npz(out_path, compressed=True,
                        sources=sources, targets=targets, weights=weights,
                        n_nodes=np.int64(n_nodes), k=np.int64(k),
                        nprobe=np.int64(-1 if meta.get("nprobe") is None
                                       else meta["nprobe"]))
    meta["output_sorted"] = bool(sort_output)
    manifest = build_manifest(out_path, sources, targets, weights, n_nodes, k, meta,
                              ordered_array_sha256)
    atomic_write_new_json(out_path + ".manifest.json", manifest)
    return manifest


def assemble_sharded(out_dir, P, out_root, n_nodes, k, meta):
    """Publish an experimental sharded artifact.

    This is a validated interchange format, not a production-scale trainer
    consumer: the current loader materializes every shard into one process.
    """
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
                       "bytes": int(os.path.getsize(dst)),
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
    commit, dirty = meta.get("builder_commit"), meta.get("builder_dirty")
    manifest = {"schema": "weighted_graph_sharded_manifest.v2", "artifact": out_root,
                "sharded": True, "n_nodes": int(n_nodes), "k": int(k),
                "metric": "cosine", "n_neighbors_param": meta["n_neighbors_param"],
                "target": float(np.log2(meta["n_neighbors_param"])),
                "smooth_k_tolerance": 1e-5, "max_iter": 64,
                "nprobe_inherited": meta.get("nprobe"),
                "input_edges_sha256": meta.get("input_edges_sha256"),
                "data_shard_records": meta.get("data_shard_records"),
                "n_edges": total, "weight_summary_note": "see index.json",
                "builder_commit": commit, "builder_dirty": dirty,
                "build_contract_sha256": meta.get("build_contract_sha256"),
                "production_trainer_ready": False,
                "resources": meta.get("resources")}
    atomic_write_new_json(out_root + ".manifest.json", manifest)
    log.info("assembled sharded artifact: %d edges across %d shards -> %s",
             total, P, out_root)
    return index, manifest


def build_manifest(out_path, sources, targets, weights, n_nodes, k, meta,
                   ordered_array_sha256):
    commit, dirty = meta.get("builder_commit"), meta.get("builder_dirty")
    source_min = int(sources.min()) if len(sources) else None
    source_max = int(sources.max()) if len(sources) else None
    target_min = int(targets.min()) if len(targets) else None
    target_max = int(targets.max()) if len(targets) else None
    return {
        "schema": "graph_manifest.v2",
        "graph_kind": "weighted_fuzzy_existing_topology.v2",
        "artifact": os.path.basename(out_path),
        "graph_path": os.path.basename(out_path),
        "graph_bytes": int(os.path.getsize(out_path)),
        "graph_sha256": sha256_file(out_path),
        "sharded": False,
        "n_nodes": int(n_nodes), "k": int(k), "metric": "cosine",
        "n_edges": int(len(weights)),
        "source_min": source_min, "source_max": source_max,
        "target_min": target_min, "target_max": target_max,
        "node_namespace": "contiguous_0..n_nodes",
        "directed": True,
        "weight_semantics": "umap-fuzzy-tconorm; sampled-proportional-to-weight",
        "n_neighbors_param": int(meta["n_neighbors_param"]),
        "target": float(np.log2(meta["n_neighbors_param"])),
        "smooth_k_tolerance": 1e-5, "max_iter": 64, "local_connectivity": 1.0,
        "set_op_mix_ratio": 1.0,
        "distance_compute_dtype": "float32",
        "embedding_storage_dtype": meta.get("embedding_storage_dtype"),
        "nprobe_inherited": meta.get("nprobe"),
        "input_edges_path": meta.get("input_edges_path"),
        "input_edges_sha256": meta.get("input_edges_sha256"),
        "input_topology_validation": meta.get("input_topology_validation"),
        "output_validation": meta.get("output_validation"),
        "data_len": int(meta["data_len"]),
        "data_fingerprint": meta["data_fingerprint"],
        "data_fingerprint_n": int(meta["data_fingerprint_n"]),
        "data_shard_records": meta.get("data_shard_records"),
        "weight_summary": _weight_summary(weights),
        "sha256_sources": ordered_array_sha256(sources),
        "sha256_targets": ordered_array_sha256(targets),
        "sha256_weights": ordered_array_sha256(weights),
        "builder_commit": commit, "builder_dirty": dirty,
        "builder_source_sha256": meta.get("builder_source_sha256"),
        "builder_runtime": meta.get("builder_runtime"),
        "build_contract_sha256": meta.get("build_contract_sha256"),
        "output_sorted_by_source_target": bool(meta.get("output_sorted")),
        "production_trainer_ready": bool(meta.get("output_sorted") and dirty is False),
        "resources": meta.get("resources"),
    }


def load_sharded_edges(out_root, *, allow_materialize=False):
    """Validate and concatenate a sharded artifact for bounded diagnostics.

    This is intentionally not advertised as a scale trainer path: it allocates
    all arrays. Callers must acknowledge that with ``allow_materialize=True``.
    """
    if not allow_materialize:
        raise RuntimeError(
            "sharded weighted graphs have no streaming trainer consumer yet; "
            "pass allow_materialize=True only for bounded validation")
    with open(os.path.join(out_root, "index.json")) as fh:
        index = json.load(fh)
    srcs, tgts, wts = [], [], []
    for shard in index["shards"]:
        path = os.path.join(out_root, shard["path"])
        _validate_file_identity(
            path, expected_bytes=shard["bytes"],
            expected_sha256=shard["sha256"], label="sharded graph member")
        z = np.load(path, allow_pickle=False)
        srcs.append(z["sources"]); tgts.append(z["targets"]); wts.append(z["weights"])
    return (np.concatenate(srcs), np.concatenate(tgts), np.concatenate(wts),
            int(index["n_nodes"]))


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _build_contract(args, *, n_nodes: int, k: int, nprobe: Optional[int],
                    n_neighbors: int, topology_sha256: str,
                    topology_stats: dict, embedding_records: list[dict]) -> dict:
    import importlib.metadata
    source_path = os.path.realpath(__file__)
    umap_source_path = os.path.realpath(sys.modules["umap.umap_"].__file__)
    commit, dirty = _builder_commit()
    body = {
        "schema": "weighted-graph-build-contract-v2",
        "input_topology": {
            "canonical_path": os.path.realpath(args.edges),
            "bytes": int(os.path.getsize(args.edges)),
            "sha256": topology_sha256,
            "n_nodes": int(n_nodes), "k": int(k), "nprobe": nprobe,
            "validation": topology_stats,
        },
        "embedding_members": embedding_records,
        "parameters": {
            "dimension": int(args.dim),
            "raw_dtype": args.raw_dtype,
            "compute_device": str(args.device),
            "n_neighbors": int(n_neighbors),
            "local_connectivity": 1.0,
            "metric": "cosine",
            "distance_compute_dtype": "float32",
            "chunk_size": int(args.chunk_size),
            "partitions": int(args.partitions),
            "output_sharded": bool(args.sharded),
            "output_sorted": not bool(args.no_sort),
            "output_path": os.path.abspath(args.out),
        },
        "builder": {
            "commit": commit,
            "dirty": dirty,
            "source_path": source_path,
            "source_sha256": sha256_file(source_path),
        },
        "runtime": {
            "python": sys.version,
            "numpy": importlib.metadata.version("numpy"),
            "torch": importlib.metadata.version("torch"),
            "umap_learn": importlib.metadata.version("umap-learn"),
            "umap_source_path": umap_source_path,
            "umap_source_sha256": sha256_file(umap_source_path),
        },
    }
    return {**body, "contract_sha256": _json_identity(body)}


def _admit_workdir(workdir: str, contract: dict) -> str:
    path = os.path.abspath(workdir)
    if os.path.islink(path):
        raise RuntimeError(f"weighted-graph workdir cannot be a symlink: {path}")
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        raise RuntimeError(f"weighted-graph workdir is not a directory: {path}")
    contract_path = os.path.join(path, "build-contract.json")
    entries = os.listdir(path)
    if not os.path.exists(contract_path):
        if entries:
            raise RuntimeError(
                f"nonempty workdir has no v2 build contract and is not resumable: {path}. "
                "Use a fresh versioned workdir.")
        atomic_write_new_json(contract_path, contract)
    else:
        observed = _load_json(contract_path)
        if canonical_json(observed) != canonical_json(contract):
            raise RuntimeError(
                f"workdir contract differs from this invocation: {path}. "
                "Changed input bytes, shard order, parameters, output, or builder code "
                "requires a fresh workdir.")
    return path


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
    refuse_existing(args.out, label="weighted graph output")
    refuse_existing(args.out + ".manifest.json", label="weighted graph manifest")
    neighbors, n_nodes, k, nprobe, topology_stats = load_topology(
        args.edges, return_stats=True)
    log.info("topology: n_nodes=%d k=%d nprobe=%s edges=%d",
             n_nodes, k, nprobe, n_nodes * k)
    log.info("topology full scan: self slots=%d (%.6f%% of slots)",
             topology_stats["self_slots"],
             100 * topology_stats["self_slot_fraction"])
    if args.corpus:
        corpora = []
        for spec in args.corpus:
            d, _, nt = spec.rpartition(":")
            corpora.append((d, int(nt)))
        emb = ShardedEmbeddings.from_corpora(corpora, dim=args.dim,
                                             raw_dtype=args.raw_dtype)
    else:
        shard_paths = resolve_shard_paths(args.embeddings_list, args.embeddings_dir)
        emb = ShardedEmbeddings(shard_paths, expected_dim=args.dim,
                                raw_dtype=args.raw_dtype)
    if len(emb) != n_nodes:
        raise SystemExit(f"embedding rows {len(emb)} != graph n_nodes {n_nodes} "
                         "— wrong shard set or order")
    log.info("embeddings: %d rows x %d dim across %d shards",
             len(emb), emb.dim, len(emb.memmaps))
    n_neighbors = args.target_neighbors or (k + 1)
    if n_neighbors <= 1:
        raise SystemExit("--target-neighbors must be greater than one")

    topology_sha256 = sha256_file(args.edges)
    embedding_records = emb.content_records()
    from basemap.graph_validation import data_fingerprint
    fp_ids, fingerprint = data_fingerprint(emb)
    contract = _build_contract(
        args, n_nodes=n_nodes, k=k, nprobe=nprobe,
        n_neighbors=n_neighbors, topology_sha256=topology_sha256,
        topology_stats=topology_stats, embedding_records=embedding_records)
    args.workdir = _admit_workdir(args.workdir, contract)
    log.info("build contract: %s", contract["contract_sha256"])

    meta = {"nprobe": nprobe, "input_edges_path": os.path.abspath(args.edges),
            "input_edges_sha256": topology_sha256,
            "input_topology_validation": topology_stats,
            "data_shard_records": embedding_records,
            "data_len": len(emb), "data_fingerprint": fingerprint,
            "data_fingerprint_n": len(fp_ids),
            "embedding_storage_dtype": np.dtype(emb.dtype).str,
            "n_neighbors_param": int(n_neighbors),
            "builder_commit": contract["builder"]["commit"],
            "builder_dirty": contract["builder"]["dirty"],
            "builder_source_sha256": contract["builder"]["source_sha256"],
            "builder_runtime": contract["runtime"],
            "build_contract_sha256": contract["contract_sha256"]}

    if not args.skip_gpu:
        gpu_guard(require_free=not args.force_gpu)
        _, a_stats = phase_a_forward_edges(
            emb, neighbors, n_nodes, k, n_neighbors, args.workdir,
            args.chunk_size, args.device, contract["contract_sha256"],
            yield_seconds=args.yield_seconds)
    fwd_dir = os.path.join(args.workdir, "fwd")
    closure = phase_a_closure(
        fwd_dir, n_nodes=n_nodes, chunk_size=args.chunk_size,
        contract_sha256=contract["contract_sha256"])
    _atomic_replace_json(os.path.join(args.workdir, "phase_a_closure.json"), closure)
    part_dir = phase_b_partition(
        fwd_dir, args.workdir, n_nodes, args.partitions,
        contract_sha256=contract["contract_sha256"],
        phase_a_closure_sha256=closure["closure_sha256"])
    out_dir, c_stats = phase_c_join(
        part_dir, args.workdir, n_nodes, args.partitions,
        contract_sha256=contract["contract_sha256"],
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
