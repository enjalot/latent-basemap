"""Validation battery (V1-V4) for build_weighted_graph.py.

V1  exactness at 100k  — my rho/sigma/membership/symmetrization vs
                          umap.fuzzy_simplicial_set on identical exact-kNN topology.
V2  topology honesty   — IVF_PQ recall@k vs exact GPU kNN at 3M, and fuzzy-weight
                          agreement approx-vs-exact topology.
V3  consumer contract  — the trainer's weighted edge sampler on the 30M artifact:
                          CDF builds, no constant-weights warning, draw ∝ weight.
V4  spot physical check— per-node weight/distance monotonicity + mutual-pair
                          symmetrized weight dominance.
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
def build_exact_knn_gpu(X, k, device="cuda", tile=4096, base=None):
    """Exact cosine kNN of the rows of ``X`` against ``base`` (default X itself).
    Returns (neighbors[n,k] int32, dists[n,k] float32 cosine distance), self
    excluded, sorted ascending. Everything stays on GPU except the outputs."""
    import torch
    import torch.nn.functional as F
    Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(device)
    Xt = F.normalize(Xt, dim=1)
    if base is None:
        Bt = Xt
        self_present = True
    else:
        Bt = F.normalize(torch.from_numpy(np.asarray(base, dtype=np.float32)).to(device), dim=1)
        self_present = False
    n = Xt.shape[0]
    neighbors = np.empty((n, k), dtype=np.int32)
    dists = np.empty((n, k), dtype=np.float32)
    # When querying against a separate base that CONTAINS the query rows (V2:
    # q = base[sample]), the query's own row is the rank-0 hit (sim=1) — treat
    # column 0 as self and fetch k+1 so k real neighbors survive.
    drop_col0 = self_present or (base is not None)
    kk = k + 1 if drop_col0 else k
    for s in range(0, n, tile):
        e = min(s + tile, n)
        sim = Xt[s:e] @ Bt.T                      # (tile, base)
        top = torch.topk(sim, kk, dim=1, largest=True)
        idx = top.indices.cpu().numpy()
        val = top.values.cpu().numpy()
        if self_present:
            # drop the self column (its own index) from each row
            qids = np.arange(s, e)[:, None]
            keep = np.ones_like(idx, dtype=bool)
            selfpos = idx == qids
            # if self not found (shouldn't happen), drop the last column
            no_self = ~selfpos.any(axis=1)
            for r in np.flatnonzero(no_self):
                selfpos[r, -1] = True
            keep[selfpos] = False
            idx_k = idx[keep].reshape(e - s, k)
            val_k = val[keep].reshape(e - s, k)
        elif base is not None:
            # drop the rank-0 self hit (query is a base row)
            idx_k = idx[:, 1:k + 1]
            val_k = val[:, 1:k + 1]
        else:
            idx_k = idx[:, :k]
            val_k = val[:, :k]
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

    # IVF_PQ (approximate) neighbors
    _, approx = index.search(q, k + 1)
    approx_clean = np.empty((m, k), dtype=np.int64)
    for i in range(m):
        row = approx[i][approx[i] != sample[i]][:k]
        if len(row) < k:
            row = np.concatenate([row, np.full(k - len(row), row[-1] if len(row) else 0)])
        approx_clean[i] = row

    # exact neighbors of the sample vs the base. Size the query tile so the
    # (tile, n_base) similarity block stays well under VRAM (~4 GB fp32).
    q_tile = max(64, int(4e9 / (4 * max(1, n_base))))
    exact_n, exact_d = build_exact_knn_gpu(q, k, device=args.device, base=base,
                                           tile=q_tile)
    # drop self from exact (sample id) if present
    for i in range(m):
        if sample[i] in exact_n[i]:
            keep = exact_n[i] != sample[i]
            en = exact_n[i][keep]
            if len(en) < k:  # pad
                en = np.concatenate([en, [en[-1]]])
            exact_n[i, :k] = en[:k]

    recalls = [len(set(approx_clean[i]) & set(exact_n[i].tolist())) / k for i in range(m)]
    recall_at_k = float(np.mean(recalls))
    log.info("V2: recall@%d (IVF_PQ vs exact) = %.4f over %d nodes", k, recall_at_k, m)

    # fuzzy weights on approx vs exact topology (need distances for both)
    def dist_of(neigh):
        d = np.empty((m, k), dtype=np.float32)
        import torch, torch.nn.functional as F
        Bt = F.normalize(torch.from_numpy(base).to(args.device), dim=1)
        Qn = F.normalize(torch.from_numpy(q).to(args.device), dim=1)
        for i0 in range(0, m, 4096):
            i1 = min(i0 + 4096, m)
            nb = torch.from_numpy(neigh[i0:i1].astype(np.int64)).to(args.device)
            nv = Bt[nb.reshape(-1)].view(i1 - i0, k, -1)
            cos = (Qn[i0:i1].unsqueeze(1) * nv).sum(2)
            d[i0:i1] = (1 - cos).clamp_(min=0).cpu().numpy()
        return d

    approx_d = dist_of(approx_clean)
    # sort each row ascending (IVF_PQ order may not be exact-distance order)
    ao = np.argsort(approx_d, axis=1)
    approx_sorted = np.take_along_axis(approx_clean, ao, 1)
    approx_d_sorted = np.take_along_axis(approx_d, ao, 1)

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
        "recall_at_k": round(recall_at_k, 4),
        "recall_p10": round(float(np.percentile(recalls, 10)), 4),
        "weight_shared_edges": len(shared),
        "weight_mean_abs_diff_shared": float(np.mean(agree)) if agree else 0.0,
        "weight_median_exact": float(np.median(list(w_ex.values()))),
        "weight_median_approx": float(np.median(list(w_ap.values()))),
        "path_a_trustworthy": recall_at_k >= 0.90,
    }
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
    import torch
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays, DeviceEdgeSampler)

    t0 = time.time()
    if os.path.isdir(args.artifact):
        from experiments.build_weighted_graph import load_sharded_edges
        sources, targets, weights, n_nodes = load_sharded_edges(args.artifact)
    else:
        sources, targets, weights, n_nodes = load_edge_arrays(args.artifact)
        sources = np.asarray(sources); targets = np.asarray(targets)
        weights = np.asarray(weights)
    load_s = time.time() - t0
    import resource
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6

    # capture logging to detect the "constant weights" warning
    records = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())
    root = logging.getLogger()
    root.addHandler(handler)
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    try:
        sampler = DeviceEdgeSampler(
            _StubDataset(device), sources, targets, weights, n_nodes,
            weighted_edge_sampling=True, batch_size=4096, device=device)
    finally:
        root.removeHandler(handler)
    constant_warning = any("constant weights" in m for m in records)

    assert sampler.sample_cdf is not None, "CDF not built"
    cdf = sampler.sample_cdf.cpu().numpy()
    # GPU torch.cumsum uses a parallel scan whose float reassociation can make
    # the CDF non-monotone by ~1e-16 — harmless for searchsorted sampling. Assert
    # effective monotonicity (no drop beyond fp noise) and record the worst dip.
    dcdf = np.diff(cdf)
    max_backstep = float(-dcdf.min()) if dcdf.min() < 0 else 0.0
    assert max_backstep < 1e-9, f"CDF not monotone (max backstep {max_backstep})"
    cdf_endpoint = float(cdf[-1])

    # draw >=10M edge indices via the sampler's OWN routine, histogram by weight
    n_draw = args.n_draw
    drawn_counts = np.zeros(len(weights), dtype=np.int64)
    per = 5_000_000
    got = 0
    while got < n_draw:
        m = min(per, n_draw - got)
        idx = sampler._draw_idx(m).cpu().numpy()
        np.add.at(drawn_counts, idx, 1)
        got += m
    # correlation between draw frequency and weight (should be ~ linear)
    w = weights.astype(np.float64)
    freq = drawn_counts / drawn_counts.sum()
    expected = w / w.sum()
    # bin by weight decile to show monotone increase in draw rate
    order = np.argsort(w)
    bins = np.array_split(order, 10)
    bin_w = [float(w[b].mean()) for b in bins]
    bin_freq = [float(freq[b].sum()) for b in bins]
    bin_exp = [float(expected[b].sum()) for b in bins]
    # Per-edge corr is Poisson-noise-limited (10M draws over ~1e9 edges → most
    # edges drawn 0×), so the meaningful test is that the AGGREGATE draw mass per
    # weight-decile matches the expected mass. corr on the decile aggregates and
    # the max per-decile abs error both capture "draws ∝ weight".
    decile_corr = float(np.corrcoef(bin_freq, bin_exp)[0, 1])
    decile_max_abs_diff = float(max(abs(a - b) for a, b in zip(bin_freq, bin_exp)))

    ws = w
    result = {
        "test": "V3_consumer_contract", "artifact": os.path.basename(args.artifact.rstrip("/")),
        "n_edges": int(len(weights)), "n_nodes": int(n_nodes),
        "load_seconds": round(load_s, 1), "peak_rss_gb": round(rss_gb, 2),
        "cdf_built": True, "cdf_max_backstep": max_backstep,
        "cdf_endpoint": cdf_endpoint,
        "constant_weights_warning": constant_warning,
        "weight_median": float(np.median(ws)), "weight_mean": float(ws.mean()),
        "weight_min": float(ws.min()), "weight_max": float(ws.max()),
        "n_drawn": int(got),
        "decile_draw_vs_expected_corr": round(decile_corr, 6),
        "decile_max_abs_diff": round(decile_max_abs_diff, 6),
        "decile_mean_weight": [round(x, 5) for x in bin_w],
        "decile_draw_freq": [round(x, 5) for x in bin_freq],
        "decile_expected_freq": [round(x, 5) for x in bin_exp],
        "PASS": (not constant_warning and decile_corr > 0.999
                 and decile_max_abs_diff < 0.005
                 and float(np.median(ws)) < 0.9),
    }
    print(json.dumps(result, indent=2))
    _dump(args, result)
    return result


# --------------------------------------------------------------------------- #
# V4 — spot physical check
# --------------------------------------------------------------------------- #
def v4(args):
    from experiments.build_weighted_graph import (
        ShardedEmbeddings, resolve_shard_paths, load_topology,
        fuzzy_directed_from_knn, chunk_knn_with_self)
    import torch

    neighbors, n_nodes, k, nprobe = load_topology(args.edges)
    shard_paths = resolve_shard_paths(args.embeddings_list, args.embeddings_dir)
    emb = ShardedEmbeddings(shard_paths, expected_dim=args.dim)
    n_neighbors = args.target_neighbors or (k + 1)
    rng = np.random.RandomState(args.seed)
    nodes = np.sort(rng.choice(n_nodes, args.n_nodes, replace=False))

    # load the artifact's symmetrized weights, indexed by (src,dst)
    if os.path.isdir(args.artifact):
        from experiments.build_weighted_graph import load_sharded_edges
        s, t, w, _ = load_sharded_edges(args.artifact)
    else:
        z = np.load(args.artifact)
        s, t, w = np.asarray(z["sources"]), np.asarray(z["targets"]), np.asarray(z["weights"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checks = {"weight_matches_min_dist": 0, "weight_monotone_decay": 0,
              "mutual_dominates": 0, "mutual_pairs_tested": 0, "nodes": int(len(nodes))}
    details = []
    # build a lookup for the sampled nodes' out-edges from the artifact
    node_set = set(int(x) for x in nodes)
    sel = np.isin(s, list(node_set))
    sub_s, sub_t, sub_w = s[sel], t[sel], w[sel]
    art = {}
    for a, b, ww in zip(sub_s, sub_t, sub_w):
        art[(int(a), int(b))] = float(ww)

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
        # highest-weight neighbor == smallest-distance neighbor?
        best_w = max(ordered, key=lambda x: x[2])
        min_d = min(ordered, key=lambda x: x[1])
        if best_w[0] == min_d[0]:
            checks["weight_matches_min_dist"] += 1
        # weights decay with distance (Spearman-ish: sorted by dist -> nonincreasing w)
        ws = [o[2] for o in sorted(ordered, key=lambda x: x[1])]
        if all(ws[i] >= ws[i + 1] - 1e-6 for i in range(len(ws) - 1)):
            checks["weight_monotone_decay"] += 1
        # mutual-pair dominance using the artifact's symmetrized weights
        for (nb, dd, mw) in ordered[:3]:
            key_fwd = (int(node), int(nb))
            if key_fwd in art:
                checks["mutual_pairs_tested"] += 1
                if art[key_fwd] >= mw - 1e-6:
                    checks["mutual_dominates"] += 1
        details.append({"node": int(node), "best_w_nbr": best_w[0],
                        "min_d_nbr": min_d[0]})

    result = {
        "test": "V4_spot_physical", "n_nodes": checks["nodes"], "k": k,
        "weight_matches_min_dist": checks["weight_matches_min_dist"],
        "weight_monotone_decay": checks["weight_monotone_decay"],
        "mutual_pairs_tested": checks["mutual_pairs_tested"],
        "mutual_dominates": checks["mutual_dominates"],
        "PASS": (checks["weight_monotone_decay"] >= 0.9 * checks["nodes"]
                 and checks["mutual_dominates"] == checks["mutual_pairs_tested"]),
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
    p2.add_argument("--k", type=int, default=15); p2.set_defaults(func=v2)

    p3 = sub.add_parser("validate-v3", help="trainer weighted-sampler contract")
    common(p3); p3.add_argument("--artifact", required=True)
    p3.add_argument("--n-draw", type=int, default=10_000_000); p3.set_defaults(func=v3)

    p4 = sub.add_parser("validate-v4", help="spot physical check")
    common(p4); p4.add_argument("--edges", required=True)
    p4.add_argument("--artifact", required=True)
    p4.add_argument("--n-nodes", type=int, default=20)
    p4.add_argument("--k", type=int, default=15)
    p4.add_argument("--seed", type=int, default=0); p4.set_defaults(func=v4)
