"""Validation battery (V1-V4) for build_weighted_graph.py.

V1  exactness at 100k  — my rho/sigma/membership/symmetrization vs
                          umap.fuzzy_simplicial_set on identical exact-kNN topology.
V2  topology honesty   — IVF_PQ recall@k vs exact GPU kNN at 3M, fuzzy-weight
                          agreement, and measured top-C exact-rerank coverage.
V3  CPU admission      — full graph/data manifest admission plus bounded host and
                          device weighted-CDF equivalence. This is not the real
                          production GPU canary.
V4  spot physical check— per-node monotonicity and exact t-conorm recomputation,
                          with true mutual and one-way pairs reported separately.
"""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np

log = logging.getLogger("weighted_graph_validate")


# --------------------------------------------------------------------------- #
# Exact cosine kNN on GPU (tiled). Reference topology for V1/V2.
# --------------------------------------------------------------------------- #
def build_exact_knn_gpu(X, k, device="cuda", tile=4096, base=None,
                        query_ids=None):
    """Exact cosine kNN of the rows of ``X`` against ``base`` (default X itself).
    Returns (neighbors[n,k] int32, dists[n,k] float32 cosine distance), self
    excluded, sorted ascending. For a separate base that contains the queries,
    pass each query's explicit base-row id; rank position is not a safe self
    test when exact-vector ties exist. Everything stays on GPU except outputs."""
    import torch
    import torch.nn.functional as F
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    Xt = F.normalize(Xt, dim=1)
    if base is None:
        Bt = Xt
        query_ids = np.arange(len(X), dtype=np.int64)
    else:
        Bt = F.normalize(torch.from_numpy(np.asarray(base, dtype=np.float32)).to(device), dim=1)
        if query_ids is not None:
            query_ids = np.asarray(query_ids, dtype=np.int64)
            if query_ids.shape != (len(X),):
                raise ValueError("query_ids must have one base-row id per query")
            if np.any(query_ids < 0) or np.any(query_ids >= len(base)):
                raise ValueError("query_ids contain ids outside the base")
    n = Xt.shape[0]
    neighbors = np.empty((n, k), dtype=np.int32)
    dists = np.empty((n, k), dtype=np.float32)
    if k > int(Bt.shape[0]) - (1 if query_ids is not None else 0):
        raise ValueError("k exceeds the number of available non-self base rows")
    for s in range(0, n, tile):
        e = min(s + tile, n)
        sim = Xt[s:e] @ Bt.T                      # (tile, base)
        if query_ids is not None:
            row = torch.arange(e - s, device=sim.device)
            col = torch.as_tensor(query_ids[s:e], dtype=torch.long,
                                  device=sim.device)
            sim[row, col] = -torch.inf
        top = torch.topk(sim, k, dim=1, largest=True)
        idx_k = top.indices.cpu().numpy()
        val_k = top.values.cpu().numpy()
        neighbors[s:e] = idx_k.astype(np.int32)
        dists[s:e] = (1.0 - val_k).astype(np.float32)
    # topk already descending by sim -> ascending by distance
    return neighbors, dists


def _knn_to_16col(neighbors, dists, node_offset=0):
    """Prepend a self-column (index=row, dist=0) -> (n, k+1) knn arrays."""
    n, k = neighbors.shape
    knn_i = np.empty((n, k + 1), dtype=np.int32)
    knn_d = np.zeros((n, k + 1), dtype=np.float32)
    knn_i[:, 0] = np.arange(node_offset, node_offset + n, dtype=np.int32)
    knn_i[:, 1:] = neighbors
    knn_d[:, 1:] = dists
    return knn_i, knn_d


# --------------------------------------------------------------------------- #
# V1 — exactness at 100k
# --------------------------------------------------------------------------- #
def v1(args):
    from experiments.build_weighted_graph import (
        ShardedEmbeddings, resolve_shard_paths, fuzzy_directed_from_knn,
        symmetrize_bucket)
    from umap.umap_ import fuzzy_simplicial_set
    import scipy.sparse as sp

    n = args.n
    k = args.k
    shard_paths = resolve_shard_paths(args.embeddings_list, args.embeddings_dir)
    emb = ShardedEmbeddings(shard_paths, expected_dim=args.dim)
    X = np.asarray(emb.gather(np.arange(n), out_dtype=np.float32))
    log.info("V1: loaded %d x %d slice", *X.shape)

    t0 = time.time()
    neighbors, dists = build_exact_knn_gpu(X, k, device=args.device)
    log.info("V1: exact GPU kNN in %.1fs", time.time() - t0)

    # cross-check the GPU kNN against sklearn on a subset (independent oracle)
    from sklearn.neighbors import NearestNeighbors
    sub = min(2000, n)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    _, skl = nn.kneighbors(X[:sub])
    jac = []
    for i in range(sub):
        a = set(neighbors[i].tolist())
        bset = set(skl[i].tolist()) - {i}
        jac.append(len(a & bset) / len(a | bset))
    knn_jaccard = float(np.mean(jac))
    log.info("V1: GPU-vs-sklearn kNN mean Jaccard@%d = %.4f", k, knn_jaccard)

    knn_i, knn_d = _knn_to_16col(neighbors, dists)
    n_neighbors = args.target_neighbors or (k + 1)

    # MY pipeline: directed membership -> my partitioned-style symmetrization
    rows, cols, vals, sig, rho, nrz = fuzzy_directed_from_knn(knn_i, knn_d, n_neighbors)
    mi, mj, mw = symmetrize_bucket(rows, cols, vals, n)
    my = {(int(a), int(b)): float(w) for a, b, w in zip(mi, mj, mw)}

    # ORACLE: umap on identical precomputed topology
    graph, sig_o, rho_o = fuzzy_simplicial_set(
        X, n_neighbors=n_neighbors, random_state=42, metric="cosine",
        knn_indices=knn_i.astype(np.int64), knn_dists=knn_d.astype(np.float64))
    coo = graph.tocoo()
    oracle = {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}

    my_set = set(my); or_set = set(oracle)
    only_mine = len(my_set - or_set)
    only_oracle = len(or_set - my_set)
    shared = my_set & or_set
    max_abs = max((abs(my[e] - oracle[e]) for e in shared), default=0.0)
    sig_diff = float(np.max(np.abs(sig - sig_o)))
    rho_diff = float(np.max(np.abs(rho - rho_o)))

    result = {
        "test": "V1_exactness_100k", "n": n, "k": k,
        "n_neighbors_param": n_neighbors, "target_log2": float(np.log2(n_neighbors)),
        "knn_gpu_vs_sklearn_jaccard": round(knn_jaccard, 4),
        "my_edges": len(my), "oracle_edges": len(oracle),
        "edges_only_mine": only_mine, "edges_only_oracle": only_oracle,
        "edge_set_equal": (only_mine == 0 and only_oracle == 0),
        "weight_max_abs_diff": max_abs,
        "sigma_max_abs_diff": sig_diff, "rho_max_abs_diff": rho_diff,
        "n_rho_zero": nrz,
        "PASS": (only_mine == 0 and only_oracle == 0 and max_abs <= 1e-4),
    }
    print(json.dumps(result, indent=2))
    _dump(args, result)
    return result


# --------------------------------------------------------------------------- #
# V2 — topology honesty at 3M
# --------------------------------------------------------------------------- #
def v2(args):
    import faiss
    from experiments.build_weighted_graph import (
        ShardedEmbeddings, resolve_shard_paths, fuzzy_directed_from_knn)

    k = args.k
    shard_paths = resolve_shard_paths(args.embeddings_list, args.embeddings_dir)
    emb = ShardedEmbeddings(shard_paths, expected_dim=args.dim)
    n_base = args.n_base or len(emb)
    n_base = min(n_base, len(emb))
    log.info("V2: base = %d rows; loading for exact kNN", n_base)
    base = np.asarray(emb.gather(np.arange(n_base), out_dtype=np.float32))

    index = faiss.read_index(args.index)
    if int(index.ntotal) != int(n_base):
        raise ValueError(
            f"index/base universe mismatch: index has {index.ntotal} rows, "
            f"exact base has {n_base}")
    if args.nprobe:
        try:
            faiss.ParameterSpace().set_index_parameter(index, "nprobe", args.nprobe)
        except Exception:
            index.nprobe = args.nprobe
    log.info("V2: index ntotal=%d nprobe=%s", index.ntotal, args.nprobe)

    rng = np.random.RandomState(0)
    m = min(args.n_sample, n_base)
    sample = np.sort(rng.choice(n_base, m, replace=False))
    q = np.ascontiguousarray(base[sample].astype(np.float32))

    # IVF_PQ topology and the wider candidate set used by Path B exact rerank.
    requested_widths = getattr(args, "candidate_widths", None) or []
    candidate_widths = sorted({int(args.candidate_k), *map(int, requested_widths)})
    if any(width < k for width in candidate_widths):
        raise ValueError("every Path-B candidate width must be at least k")
    candidate_k = max(candidate_widths)
    t_ann = time.time()
    _, approx = index.search(q, candidate_k + 1)
    ann_seconds = time.time() - t_ann
    candidates = np.empty((m, candidate_k), dtype=np.int64)
    for i in range(m):
        row = approx[i][(approx[i] >= 0) & (approx[i] != sample[i])]
        # Stable unique is required: duplicate candidate ids otherwise inflate
        # coverage and can occupy multiple reranked slots.
        _, first = np.unique(row, return_index=True)
        row = row[np.sort(first)][:candidate_k]
        if len(row) != candidate_k:
            raise RuntimeError(
                f"ANN returned only {len(row)} distinct non-self candidates for "
                f"query {int(sample[i])}; requested {candidate_k}")
        candidates[i] = row
    if int(candidates.max()) >= n_base:
        raise ValueError(
            f"ANN candidate id {int(candidates.max())} is outside the exact base "
            f"of {n_base}; use a matching index/base universe")
    approx_clean = candidates[:, :k].copy()

    # Upload and normalize the base/query exactly once.  The earlier validator
    # repeated this 4.6 GB upload for exact kNN, ANN-distance reconstruction,
    # and Path-B reranking.  Reusing the tensors makes the diagnostic both
    # faster and a cleaner comparison of the two candidate policies.
    import torch
    import torch.nn.functional as F
    Bt = F.normalize(torch.from_numpy(base).to(args.device), dim=1)
    Qn = F.normalize(torch.from_numpy(q).to(args.device), dim=1)

    # Exact neighbors of the sample vs the base. Size the query tile so the
    # (tile, n_base) similarity block stays well under VRAM (~4 GB fp32).
    q_tile = max(64, int(4e9 / (4 * max(1, n_base))))
    t_exact = time.time()
    exact_n = np.empty((m, k), dtype=np.int32)
    exact_d = np.empty((m, k), dtype=np.float32)
    exact_boundary_tie = np.zeros(m, dtype=bool)
    with torch.no_grad():
        for i0 in range(0, m, q_tile):
            i1 = min(i0 + q_tile, m)
            sim = Qn[i0:i1] @ Bt.T
            rows = torch.arange(i1 - i0, device=sim.device)
            cols = torch.as_tensor(sample[i0:i1], dtype=torch.long,
                                   device=sim.device)
            sim[rows, cols] = -torch.inf
            # k+1 exposes exact-score ties at the evaluation boundary. Raw row
            # recall is not identifiable for those queries (common with exact
            # duplicate embeddings), so report them separately rather than
            # treating an arbitrary tied row id as ground truth.
            top = torch.topk(sim, k + 1, dim=1, largest=True)
            exact_n[i0:i1] = top.indices[:, :k].cpu().numpy().astype(np.int32)
            exact_d[i0:i1] = (
                1.0 - top.values[:, :k]).cpu().numpy().astype(np.float32)
            exact_boundary_tie[i0:i1] = (
                torch.abs(top.values[:, k - 1] - top.values[:, k]) <= 1e-7
            ).cpu().numpy()
    exact_seconds = time.time() - t_exact

    recalls = np.asarray([
        len(set(approx_clean[i]) & set(exact_n[i].tolist())) / k
        for i in range(m)
    ], dtype=np.float64)
    unambiguous = ~exact_boundary_tie
    if not np.any(unambiguous):
        raise RuntimeError("every exact query has a tied top-k boundary")
    recall_at_k = float(np.mean(recalls))
    log.info("V2: recall@%d (IVF_PQ vs exact) = %.4f over %d nodes", k, recall_at_k, m)

    # fuzzy weights on approx vs exact topology (need distances for both)
    def dist_of(neigh):
        width = int(neigh.shape[1])
        d = np.empty((m, width), dtype=np.float32)
        for i0 in range(0, m, 4096):
            i1 = min(i0 + 4096, m)
            nb = torch.from_numpy(neigh[i0:i1].astype(np.int64)).to(args.device)
            nv = Bt[nb.reshape(-1)].view(i1 - i0, width, -1)
            cos = (Qn[i0:i1].unsqueeze(1) * nv).sum(2)
            d[i0:i1] = (1 - cos).clamp_(min=0).cpu().numpy()
        return d

    approx_d = dist_of(approx_clean)
    # sort each row ascending (IVF_PQ order may not be exact-distance order)
    ao = np.argsort(approx_d, axis=1)
    approx_sorted = np.take_along_axis(approx_clean, ao, 1)
    approx_d_sorted = np.take_along_axis(approx_d, ao, 1)

    # Path B diagnostic: coverage of exact top-k within the wider ANN candidate
    # set, followed by exact fp32 cosine reranking of those candidates.
    t_rerank = time.time()
    candidate_d = dist_of(candidates)
    path_b = {}
    for width in candidate_widths:
        width_candidates = candidates[:, :width]
        width_distances = candidate_d[:, :width]
        width_order = np.argsort(width_distances, axis=1, kind="stable")
        width_reranked = np.take_along_axis(
            width_candidates, width_order[:, :k], axis=1)
        candidate_recalls = np.asarray([
            len(set(width_candidates[i].tolist()) & set(exact_n[i].tolist())) / k
            for i in range(m)
        ], dtype=np.float64)
        rerank_recalls = np.asarray([
            len(set(width_reranked[i].tolist()) & set(exact_n[i].tolist())) / k
            for i in range(m)
        ], dtype=np.float64)
        path_b[str(width)] = {
            "candidate_recall_at_k": round(float(np.mean(candidate_recalls)), 4),
            "candidate_recall_p10": round(
                float(np.percentile(candidate_recalls, 10)), 4),
            "exact_rerank_recall_at_k": round(float(np.mean(rerank_recalls)), 4),
            "exact_rerank_recall_p10": round(
                float(np.percentile(rerank_recalls, 10)), 4),
            "candidate_recall_at_k_unambiguous": round(
                float(np.mean(candidate_recalls[unambiguous])), 4),
            "exact_rerank_recall_at_k_unambiguous": round(
                float(np.mean(rerank_recalls[unambiguous])), 4),
        }
    rerank_seconds = time.time() - t_rerank
    widest = path_b[str(candidate_k)]
    candidate_recall = float(widest["candidate_recall_at_k"])
    reranked_recall = float(widest["exact_rerank_recall_at_k"])

    n_neighbors = args.target_neighbors or (k + 1)

    def directed(neigh, dist):
        knn_i = np.empty((m, k + 1), dtype=np.int32)
        knn_d = np.zeros((m, k + 1), dtype=np.float32)
        knn_i[:, 0] = sample.astype(np.int32)
        knn_i[:, 1:] = neigh
        knn_d[:, 1:] = dist
        r, c, v, _, _, _ = fuzzy_directed_from_knn(knn_i, knn_d, n_neighbors)
        return {(int(a), int(b)): float(w) for a, b, w in zip(r, c, v)}

    w_ex = directed(exact_n, exact_d)
    w_ap = directed(approx_sorted, approx_d_sorted)
    shared = set(w_ex) & set(w_ap)
    agree = [abs(w_ex[e] - w_ap[e]) for e in shared]

    result = {
        "test": "V2_topology_honesty", "index": os.path.basename(args.index),
        "n_base": n_base, "n_sample": m, "k": k, "nprobe": args.nprobe,
        "candidate_k": candidate_k, "candidate_widths": candidate_widths,
        "recall_at_k": round(recall_at_k, 4),
        "recall_p10": round(float(np.percentile(recalls, 10)), 4),
        "recall_at_k_unambiguous": round(
            float(np.mean(recalls[unambiguous])), 4),
        "exact_boundary_tie_count": int(exact_boundary_tie.sum()),
        "exact_boundary_tie_fraction": round(
            float(exact_boundary_tie.mean()), 6),
        "unambiguous_query_count": int(unambiguous.sum()),
        "weight_shared_edges": len(shared),
        "weight_mean_abs_diff_shared": float(np.mean(agree)) if agree else 0.0,
        "weight_median_exact": float(np.median(list(w_ex.values()))),
        "weight_median_approx": float(np.median(list(w_ap.values()))),
        "path_a_trustworthy": float(np.mean(recalls[unambiguous])) >= 0.90,
        "path_b_candidate_recall_at_k": round(candidate_recall, 4),
        "path_b_candidate_recall_p10": widest["candidate_recall_p10"],
        "path_b_reranked_recall_at_k": round(reranked_recall, 4),
        "path_b_reranked_recall_p10": widest["exact_rerank_recall_p10"],
        "path_b_by_candidate_width": path_b,
        "ann_search_seconds": round(ann_seconds, 3),
        "exact_reference_seconds": round(exact_seconds, 3),
        "candidate_exact_rerank_seconds": round(rerank_seconds, 3),
        "path_b_measured": True,
        "base_uploaded_and_normalized_once": True,
    }
    result["PASS"] = bool(
        np.isfinite([
            result["recall_at_k"], result["path_b_candidate_recall_at_k"],
            result["path_b_reranked_recall_at_k"],
        ]).all()
        and result["path_b_candidate_recall_at_k"] + 1e-4 >= result["recall_at_k"]
        and result["path_b_reranked_recall_at_k"]
        <= result["path_b_candidate_recall_at_k"] + 1e-4
        and result["unambiguous_query_count"] > 0
    )
    print(json.dumps(result, indent=2))
    _dump(args, result)
    return result


# --------------------------------------------------------------------------- #
# V3 — consumer contract at 30M
# --------------------------------------------------------------------------- #
class _StubDataset:
    def __init__(self, device):
        import torch
        self.device = device
        self._t = torch
    def index_select(self, idx):
        return self._t.zeros((idx.shape[0], 1), device=self.device)


def v3(args):
    import resource
    import torch
    from basemap.graph_validation import (
        validate_against_manifest, validate_graph_content,
        validate_graph_data_pair)
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays, DeviceEdgeSampler, HostStreamEdgeSampler)
    from experiments.build_weighted_graph import (
        ShardedEmbeddings, resolve_shard_paths)

    t0 = time.time()
    if os.path.isdir(args.artifact):
        raise RuntimeError(
            "V3 production admission requires a single trainer-consumable graph; "
            "the sharded format is validation-only until a streaming consumer exists")
    sources, targets, weights, n_nodes = load_edge_arrays(args.artifact)
    sources = np.asarray(sources); targets = np.asarray(targets)
    weights = np.asarray(weights)
    load_s = time.time() - t0

    manifest_path = args.artifact + ".manifest.json"
    if not os.path.exists(manifest_path):
        raise RuntimeError(f"V3 requires the sibling production manifest: {manifest_path}")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    if not manifest.get("production_trainer_ready", False):
        raise RuntimeError("weighted graph manifest does not claim production_trainer_ready")
    if args.embeddings_list or args.embeddings_dir:
        shard_paths = resolve_shard_paths(args.embeddings_list, args.embeddings_dir)
    else:
        records = manifest.get("data_shard_records") or []
        shard_paths = [item["canonical_path"] for item in records]
    if not shard_paths:
        raise RuntimeError(
            "V3 needs ordered embedding paths (manifest data_shard_records or CLI)")
    emb = ShardedEmbeddings(shard_paths, expected_dim=args.dim)
    validate_against_manifest(emb, manifest)
    trusted = validate_graph_content(
        args.artifact, manifest, shard_paths=shard_paths,
        require_manifest_sha=True)
    validate_graph_data_pair(sources, targets, n_nodes, len(emb))

    if weights is None:
        raise RuntimeError("weighted artifact did not load a weights array")
    if (not np.all(np.isfinite(weights)) or np.any(weights < 0)
            or not float(np.sum(weights, dtype=np.float64)) > 0):
        raise RuntimeError("weighted artifact has invalid or unsamplable weights")

    # Exercise both sampler implementations on a deterministic bounded slice.
    # This verifies the host-hybrid CDF used at 30M without allocating a second
    # full 738M-edge CDF during a CPU-side admission check.
    sample_n = min(int(args.sampler_edges), len(weights))
    sample_ids = np.unique(np.linspace(0, len(weights) - 1, sample_n).astype(np.int64))
    sample_s = sources[sample_ids]
    sample_t = targets[sample_ids]
    sample_w = weights[sample_ids]

    # capture logging to detect the "constant weights" warning
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        device_sampler = DeviceEdgeSampler(
            _StubDataset("cpu"), sample_s, sample_t, sample_w, n_nodes,
            weighted_edge_sampling=True, batch_size=4096, device="cpu")
        host_sampler = HostStreamEdgeSampler(
            _StubDataset("cpu"), sample_s, sample_t, sample_w, n_nodes,
            weighted_edge_sampling=True, batch_size=4096, device="cpu",
            n_workers=0)
    finally:
        root.removeHandler(handler)
    constant_warning = any("constant weights" in m for m in records)

    assert device_sampler.sample_cdf is not None, "device CDF not built"
    assert host_sampler._cdf_h is not None, "host CDF not built"
    cdf = host_sampler._cdf_h
    device_cdf = device_sampler.sample_cdf.cpu().numpy()
    cdf_impl_max_abs_diff = float(np.max(np.abs(cdf - device_cdf)))
    # GPU torch.cumsum uses a parallel scan whose float reassociation can make
    # the CDF non-monotone by ~1e-16 — harmless for searchsorted sampling. Assert
    # effective monotonicity (no drop beyond fp noise) and record the worst dip.
    dcdf = np.diff(cdf)
    max_backstep = float(-dcdf.min()) if dcdf.min() < 0 else 0.0
    assert max_backstep < 1e-9, f"CDF not monotone (max backstep {max_backstep})"
    cdf_endpoint = float(cdf[-1])

    # Draw from the host sampler's exact inverse-CDF rule and aggregate by weight
    # decile. No per-edge histogram is allocated (which previously added 5.9 GB).
    n_draw = args.n_draw
    w = sample_w.astype(np.float64)
    order = np.argsort(w, kind="stable")
    bins = np.array_split(order, 10)
    bin_of_edge = np.empty(len(w), dtype=np.int8)
    expected_mass = []
    bin_w = []
    total_w = float(w.sum())
    for bi, members in enumerate(bins):
        bin_of_edge[members] = bi
        expected_mass.append(float(w[members].sum() / total_w))
        bin_w.append(float(w[members].mean()))
    draw_by_bin = np.zeros(10, dtype=np.int64)
    rng = np.random.default_rng(0)
    per = 1_000_000
    got = 0
    while got < n_draw:
        m = min(per, n_draw - got)
        idx = np.searchsorted(cdf, rng.random(m), side="left")
        np.clip(idx, 0, len(w) - 1, out=idx)
        draw_by_bin += np.bincount(bin_of_edge[idx], minlength=10)
        got += m
    bin_freq = (draw_by_bin / got).tolist()
    bin_exp = expected_mass
    # Per-edge corr is Poisson-noise-limited (10M draws over ~1e9 edges → most
    # edges drawn 0×), so the meaningful test is that the AGGREGATE draw mass per
    # weight-decile matches the expected mass. corr on the decile aggregates and
    # the max per-decile abs error both capture "draws ∝ weight".
    decile_corr = float(np.corrcoef(bin_freq, bin_exp)[0, 1])
    decile_max_abs_diff = float(max(abs(a - b) for a, b in zip(bin_freq, bin_exp)))

    host_sampler.close()
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    result = {
        "test": "V3_consumer_contract", "artifact": os.path.basename(args.artifact.rstrip("/")),
        "n_edges": int(len(weights)), "n_nodes": int(n_nodes),
        "load_seconds": round(load_s, 1), "peak_rss_gb": round(rss_gb, 2),
        "peak_rss_measurement_point": "after_manifest_validation_and_bounded_samplers",
        "manifest_admitted": True, "trusted_content": trusted,
        "production_expected_pipeline": "hybrid",
        "production_expected_sampler_class": "HostStreamEdgeSampler",
        "production_gpu_canary_executed": False,
        "sampler_test_scope_edges": int(len(sample_w)),
        "cdf_built": True, "cdf_max_backstep": max_backstep,
        "cdf_endpoint": cdf_endpoint,
        "device_vs_host_cdf_max_abs_diff": cdf_impl_max_abs_diff,
        "constant_weights_warning": constant_warning,
        "sample_weight_median": float(np.median(w)),
        "full_weight_mean": float(np.mean(weights, dtype=np.float64)),
        "full_weight_min": float(weights.min()), "full_weight_max": float(weights.max()),
        "n_drawn": int(got),
        "decile_draw_vs_expected_corr": round(decile_corr, 6),
        "decile_max_abs_diff": round(decile_max_abs_diff, 6),
        "decile_mean_weight": [round(x, 5) for x in bin_w],
        "decile_draw_freq": [round(x, 5) for x in bin_freq],
        "decile_expected_freq": [round(x, 5) for x in bin_exp],
        "PASS": (not constant_warning and decile_corr > 0.999
                 and decile_max_abs_diff < 0.005
                 and cdf_impl_max_abs_diff < 1e-9
                 and float(np.median(w)) < 0.9),
    }
    print(json.dumps(result, indent=2))
    _dump(args, result)
    return result


# --------------------------------------------------------------------------- #
# V4 — spot physical check
# --------------------------------------------------------------------------- #
def v4(args):
    import gc
    from experiments.build_weighted_graph import (
        ShardedEmbeddings, resolve_shard_paths,
        fuzzy_directed_from_knn, chunk_knn_with_self)
    from basemap.artifact_identity import sha256_file
    import torch

    # The production manifest binds the complete topology and records the
    # builder's full source/target scan. Reopening all 450M source ids here
    # would duplicate that admission and adds ~1.8 GB only to establish
    # ``source == row`` again. V4 needs the neighbor matrix plus the exact
    # content binding, so authenticate the file and consume targets directly.
    manifest_path = args.artifact + ".manifest.json"
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    topology_validation = manifest.get("input_topology_validation") or {}
    if (manifest.get("input_edges_sha256") != sha256_file(args.edges)
            or topology_validation.get("validation") != "full_scan"
            or topology_validation.get("source_layout_valid") is not True
            or topology_validation.get("target_bounds_valid") is not True
            or topology_validation.get("duplicate_target_rows") != 0):
        raise RuntimeError(
            "V4 requires a manifest-bound topology with a passing full scan")
    with np.load(args.edges, allow_pickle=False) as topology:
        n_nodes = int(np.asarray(topology["n_nodes"]))
        k = int(np.asarray(topology["k"]))
        targets = np.asarray(topology["targets"])
    if (targets.ndim != 1 or len(targets) != n_nodes * k
            or targets.dtype.kind not in "iu"
            or manifest.get("n_nodes") != n_nodes or manifest.get("k") != k):
        raise RuntimeError("V4 topology geometry differs from its graph manifest")
    neighbors = targets.reshape(n_nodes, k)
    shard_paths = resolve_shard_paths(args.embeddings_list, args.embeddings_dir)
    emb = ShardedEmbeddings(shard_paths, expected_dim=args.dim)
    n_neighbors = args.target_neighbors or (k + 1)
    rng = np.random.RandomState(args.seed)
    nodes = np.sort(rng.choice(n_nodes, args.n_nodes, replace=False))

    # load the artifact's symmetrized weights, indexed by (src,dst)
    if os.path.isdir(args.artifact):
        from experiments.build_weighted_graph import load_sharded_edges
        s, t, w, _ = load_sharded_edges(args.artifact, allow_materialize=True)
        left = np.searchsorted(s, nodes, side="left")
        right = np.searchsorted(s, nodes, side="right")
        if any(hi > lo and not np.all(s[lo:hi] == node)
               for node, lo, hi in zip(nodes, left, right)):
            raise RuntimeError("weighted artifact source slices are not contiguous")
        target_slices = [np.array(t[lo:hi], copy=True) for lo, hi in zip(left, right)]
        weight_slices = [np.array(w[lo:hi], copy=True) for lo, hi in zip(left, right)]
        del s, t, w
    else:
        # A compressed production NPZ cannot be memmapped. Load its three
        # multi-gigabyte members one at a time and retain only the sampled
        # source slices; V4 otherwise held ~8.9 GB of graph arrays at once for
        # sixty scalar comparisons.
        with np.load(args.artifact, allow_pickle=False) as z:
            s = np.asarray(z["sources"])
            left = np.searchsorted(s, nodes, side="left")
            right = np.searchsorted(s, nodes, side="right")
            if any(hi > lo and not np.all(s[lo:hi] == node)
                   for node, lo, hi in zip(nodes, left, right)):
                raise RuntimeError("weighted artifact source slices are not contiguous")
            del s
            gc.collect()
            t = np.asarray(z["targets"])
            target_slices = [
                np.array(t[lo:hi], copy=True) for lo, hi in zip(left, right)]
            del t
            gc.collect()
            w = np.asarray(z["weights"])
            weight_slices = [
                np.array(w[lo:hi], copy=True) for lo, hi in zip(left, right)]
            del w
        gc.collect()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checks = {"nearest_has_max_membership": 0, "weight_monotone_decay": 0,
              "sym_weight_matches": 0, "pairs_tested": 0,
              "actual_mutual_pairs": 0, "one_way_pairs": 0,
              "nodes": int(len(nodes))}
    details = []
    pair_details = []
    missing_artifact_pairs = []
    expected_pairs = 0
    absolute_tolerance = 2e-5
    # The production artifact is source-sorted.  Use bounded source slices for
    # the sampled nodes instead of allocating an E-length ``np.isin`` mask
    # (about 738 MB at 30M) and scanning all 738M rows a second time.
    art = {}
    for node, targets, weights in zip(nodes, target_slices, weight_slices):
        for b, ww in zip(targets, weights):
            art[(int(node), int(b))] = float(ww)

    for node in nodes:
        knn_i, knn_d = chunk_knn_with_self(emb, neighbors, int(node), int(node) + 1,
                                           device, torch)
        r, c, v, sig, rho, _ = fuzzy_directed_from_knn(knn_i, knn_d, n_neighbors)
        if len(v) == 0:
            continue
        # neighbors sorted ascending by distance in knn_d[0,1:]
        nbr = knn_i[0, 1:]; dst = knn_d[0, 1:]
        # directed membership per neighbor (align to c)
        memb = {int(cc): float(vv) for cc, vv in zip(c, v)}
        ordered = [(int(nbr[j]), float(dst[j]), memb.get(int(nbr[j]))) for j in range(k)
                   if int(nbr[j]) in memb and int(nbr[j]) != int(node)]
        if len(ordered) < 2:
            continue
        # rho can make several nearest memberships exactly 1.0; test the value,
        # not an arbitrary neighbor id selected from a tie.
        best_w = max(ordered, key=lambda x: x[2])
        min_d = min(ordered, key=lambda x: x[1])
        if min_d[2] >= best_w[2] - 1e-6:
            checks["nearest_has_max_membership"] += 1
        # weights decay with distance (Spearman-ish: sorted by dist -> nonincreasing w)
        ws = [o[2] for o in sorted(ordered, key=lambda x: x[1])]
        if all(ws[i] >= ws[i + 1] - 1e-6 for i in range(len(ws) - 1)):
            checks["weight_monotone_decay"] += 1
        # Recompute the reverse directed membership. Only a pair with a positive
        # reverse membership is mutual; compare the artifact to the exact
        # t-conorm for both true-mutual and one-way cases.
        expected_pairs += min(3, len(ordered))
        for (nb, dd, mw) in ordered[:3]:
            key_fwd = (int(node), int(nb))
            if key_fwd not in art:
                missing_artifact_pairs.append({
                    "source": int(node), "target": int(nb)})
                continue
            rev_i, rev_d = chunk_knn_with_self(
                emb, neighbors, int(nb), int(nb) + 1, device, torch)
            _, rev_c, rev_v, _, _, _ = fuzzy_directed_from_knn(
                rev_i, rev_d, n_neighbors)
            reverse = {int(cc): float(vv) for cc, vv in zip(rev_c, rev_v)}
            reverse_m = reverse.get(int(node), 0.0)
            expected_sym = mw + reverse_m - mw * reverse_m
            observed_sym = art[key_fwd]
            absolute_delta = abs(observed_sym - expected_sym)
            relative_delta = absolute_delta / max(abs(expected_sym), 1e-45)
            within_tolerance = absolute_delta <= absolute_tolerance
            checks["pairs_tested"] += 1
            if reverse_m > 0:
                checks["actual_mutual_pairs"] += 1
            else:
                checks["one_way_pairs"] += 1
            if within_tolerance:
                checks["sym_weight_matches"] += 1
            pair_details.append({
                "source": int(node), "target": int(nb),
                "distance": float(dd),
                "forward_membership": float(mw),
                "reverse_membership": float(reverse_m),
                "mutual": bool(reverse_m > 0),
                "artifact_weight": float(observed_sym),
                "expected_tconorm": float(expected_sym),
                "absolute_delta": float(absolute_delta),
                "relative_delta": float(relative_delta),
                "within_tolerance": bool(within_tolerance),
            })
        details.append({"node": int(node), "best_w_nbr": best_w[0],
                        "min_d_nbr": min_d[0]})

    result = {
        "test": "V4_spot_physical", "n_nodes": checks["nodes"], "k": k,
        "nearest_has_max_membership": checks["nearest_has_max_membership"],
        "weight_monotone_decay": checks["weight_monotone_decay"],
        "pairs_tested": checks["pairs_tested"],
        "actual_mutual_pairs": checks["actual_mutual_pairs"],
        "one_way_pairs": checks["one_way_pairs"],
        "sym_weight_matches": checks["sym_weight_matches"],
        "expected_pairs": int(expected_pairs),
        "missing_artifact_pairs": missing_artifact_pairs,
        "absolute_tolerance": absolute_tolerance,
        "compute_device": str(device),
        "pair_details": pair_details,
        "PASS": (checks["weight_monotone_decay"] >= 0.9 * checks["nodes"]
                 and checks["nearest_has_max_membership"] >= 0.9 * checks["nodes"]
                 and checks["pairs_tested"] == expected_pairs
                 and not missing_artifact_pairs
                 and checks["sym_weight_matches"] == checks["pairs_tested"]),
    }
    print(json.dumps(result, indent=2))
    _dump(args, result)
    return result


def _dump(args, result):
    if getattr(args, "json_out", None):
        with open(args.json_out, "w") as fh:
            json.dump(result, fh, indent=2)
        log.info("wrote %s", args.json_out)


def register(sub):
    def common(p):
        p.add_argument("--embeddings-list", nargs="+", default=None)
        p.add_argument("--embeddings-dir", nargs="+", default=None)
        p.add_argument("--dim", type=int, default=384)
        p.add_argument("--device", default="cuda")
        p.add_argument("--target-neighbors", type=int, default=None)
        p.add_argument("--json-out", default=None)

    p1 = sub.add_parser("validate-v1", help="exactness at 100k vs fuzzy_simplicial_set")
    common(p1); p1.add_argument("--n", type=int, default=100_000)
    p1.add_argument("--k", type=int, default=15); p1.set_defaults(func=v1)

    p2 = sub.add_parser("validate-v2", help="IVF_PQ topology honesty at 3M")
    common(p2); p2.add_argument("--index", required=True)
    p2.add_argument("--n-base", type=int, default=None)
    p2.add_argument("--n-sample", type=int, default=50_000)
    p2.add_argument("--nprobe", type=int, default=64)
    p2.add_argument("--candidate-k", type=int, default=64,
                    help="ANN candidate width for the Path-B exact-rerank diagnostic")
    p2.add_argument("--candidate-widths", type=int, nargs="+", default=None,
                    help="report multiple widths from one widest ANN/exact pass")
    p2.add_argument("--k", type=int, default=15); p2.set_defaults(func=v2)

    p3 = sub.add_parser("validate-v3", help="trainer weighted-sampler contract")
    common(p3); p3.add_argument("--artifact", required=True)
    p3.add_argument("--sampler-edges", type=int, default=1_000_000,
                    help="bounded deterministic edge sample for CPU sampler checks")
    p3.add_argument("--n-draw", type=int, default=10_000_000); p3.set_defaults(func=v3)

    p4 = sub.add_parser("validate-v4", help="spot physical check")
    common(p4); p4.add_argument("--edges", required=True)
    p4.add_argument("--artifact", required=True)
    p4.add_argument("--n-nodes", type=int, default=20)
    p4.add_argument("--k", type=int, default=15)
    p4.add_argument("--seed", type=int, default=0); p4.set_defaults(func=v4)
