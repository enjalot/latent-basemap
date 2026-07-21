"""Focused tests for experiments/build_weighted_graph.py.

Small synthetic checks that (a) the symmetrization t-conorm join reproduces
scipy's, (b) partitioning is join-invariant, (c) the full per-chunk pipeline
matches umap.fuzzy_simplicial_set on identical topology, and (d) the sharded
gather is order-preserving.
"""
import os
import sys
import ast
from argparse import Namespace

import numpy as np
import pytest
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.build_weighted_graph import (
    ShardedEmbeddings, symmetrize_bucket, _pair_bucket,
    fuzzy_directed_from_knn, load_topology, phase_a_forward_edges,
    phase_a_closure, phase_b_partition, phase_c_join, chunk_knn_with_self,
    cmd_build)
from experiments.weighted_graph_validate import build_exact_knn_gpu, _knn_to_16col


def _scipy_tconorm(rows, cols, vals, n):
    P = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    P.eliminate_zeros()
    T = P.transpose()
    S = P + T - P.multiply(T)
    S.eliminate_zeros()
    return S.tocoo()


def test_symmetrize_bucket_matches_scipy():
    # hand-crafted directed membership with mutual + one-directional edges
    rows = np.array([0, 0, 1, 2, 3], dtype=np.int32)
    cols = np.array([1, 2, 0, 3, 2], dtype=np.int32)
    vals = np.array([0.8, 0.5, 0.4, 0.9, 0.3], dtype=np.float32)
    n = 4
    ui, uj, uw = symmetrize_bucket(rows, cols, vals, n)
    mine = {(int(a), int(b)): float(w) for a, b, w in zip(ui, uj, uw)}
    coo = _scipy_tconorm(rows, cols, vals, n)
    ref = {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}
    assert set(mine) == set(ref)
    for e in ref:
        assert abs(mine[e] - ref[e]) < 1e-6, (e, mine[e], ref[e])
    # mutual pair (0,1)/(1,0): sym = .8 + .4 - .8*.4 = 0.88 both directions
    assert abs(mine[(0, 1)] - 0.88) < 1e-6
    assert abs(mine[(1, 0)] - 0.88) < 1e-6
    # one-directional (0,2)/(2,0)=(only 0->2 .5): sym=.5 both directions, and
    # (2,3)/(3,2) mutual = .9+.3-.27=0.93
    assert abs(mine[(2, 0)] - 0.5) < 1e-6
    assert abs(mine[(2, 3)] - 0.93) < 1e-6


def test_symmetrize_bucket_coalesces_duplicate_directed_keys_like_scipy():
    rows = np.array([0, 0, 1], dtype=np.int32)
    cols = np.array([1, 1, 0], dtype=np.int32)
    vals = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    ui, uj, uw = symmetrize_bucket(rows, cols, vals, 2)
    mine = {(int(a), int(b)): float(w) for a, b, w in zip(ui, uj, uw)}
    ref_coo = _scipy_tconorm(rows, cols, vals, 2)
    ref = {(int(a), int(b)): float(w)
           for a, b, w in zip(ref_coo.row, ref_coo.col, ref_coo.data)}
    assert set(mine) == set(ref)
    assert mine[(0, 1)] == pytest.approx(0.7, abs=1e-6)
    assert mine == pytest.approx(ref, abs=1e-6)


def test_partition_join_invariance():
    rng = np.random.RandomState(0)
    n = 500
    k = 10
    src = np.repeat(np.arange(n), k).astype(np.int32)
    dst = rng.randint(0, n, size=n * k).astype(np.int32)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    w = rng.uniform(0.01, 1.0, size=len(src)).astype(np.float32)
    # single-bucket reference
    ui0, uj0, uw0 = symmetrize_bucket(src, dst, w, n)
    ref = {(int(a), int(b)): float(x) for a, b, x in zip(ui0, uj0, uw0)}
    # partitioned
    P = 8
    a = np.minimum(src, dst); b = np.maximum(src, dst)
    buckets = _pair_bucket(a, b, n, P)
    merged = {}
    for p in range(P):
        m = buckets == p
        if not m.any():
            continue
        ui, uj, uw = symmetrize_bucket(src[m], dst[m], w[m], n)
        for aa, bb, xx in zip(ui, uj, uw):
            merged[(int(aa), int(bb))] = float(xx)
    assert set(merged) == set(ref)
    for e in ref:
        assert abs(merged[e] - ref[e]) < 1e-6


def test_full_pipeline_matches_umap():
    pytest.importorskip("torch")
    from umap.umap_ import fuzzy_simplicial_set
    rng = np.random.RandomState(1)
    n = 800
    k = 15
    X = rng.randn(n, 48).astype(np.float32)
    neighbors, dists = build_exact_knn_gpu(X, k, device="cpu")
    knn_i, knn_d = _knn_to_16col(neighbors, dists)
    n_neighbors = k + 1
    rows, cols, vals, _, _, _ = fuzzy_directed_from_knn(knn_i, knn_d, n_neighbors)
    ui, uj, uw = symmetrize_bucket(rows, cols, vals, n)
    mine = {(int(a), int(b)): float(x) for a, b, x in zip(ui, uj, uw)}
    graph, _, _ = fuzzy_simplicial_set(
        X, n_neighbors=n_neighbors, random_state=42, metric="cosine",
        knn_indices=knn_i.astype(np.int64), knn_dists=knn_d.astype(np.float64))
    coo = graph.tocoo()
    ref = {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data)}
    assert set(mine) == set(ref), (len(mine), len(ref))
    maxdiff = max(abs(mine[e] - ref[e]) for e in ref)
    assert maxdiff < 1e-4, maxdiff


def test_membership_matches_umap_full_and_offset():
    """fuzzy_directed_from_knn must (a) equal umap.compute_membership_strengths on
    a full 0..n block and (b) emit correct GLOBAL source ids for a sub-block that
    does NOT start at 0 — the bug that corrupted the first 30M build."""
    pytest.importorskip("torch")
    from umap.umap_ import smooth_knn_dist, compute_membership_strengths
    rng = np.random.RandomState(7)
    n, k = 600, 15
    X = rng.randn(n, 32).astype(np.float32)
    neighbors, dists = build_exact_knn_gpu(X, k, device="cpu")
    knn_i, knn_d = _knn_to_16col(neighbors, dists)  # col0 = global self id
    nn = k + 1

    # (a) full block vs umap's own kernel
    sig, rho = smooth_knn_dist(knn_d.astype(np.float32), float(nn), local_connectivity=1.0)
    ur, uc, uv, _ = compute_membership_strengths(knn_i, knn_d.astype(np.float32), sig, rho, False)
    umap_full = {(int(r), int(c)): float(v)
                 for r, c, v in zip(ur, uc, uv) if v > 0}
    mr, mc, mv, _, _, _ = fuzzy_directed_from_knn(knn_i, knn_d, nn)
    mine_full = {(int(a), int(b)): float(w) for a, b, w in zip(mr, mc, mv)}
    assert set(mine_full) == set(umap_full)
    assert max(abs(mine_full[e] - umap_full[e]) for e in umap_full) < 1e-4

    # (b) offset sub-block [200:260): sources must be the GLOBAL ids 200..259 and
    #     equal the full-block edges restricted to those sources.
    lo, hi = 200, 260
    sr, sc, sv, _, _, _ = fuzzy_directed_from_knn(knn_i[lo:hi], knn_d[lo:hi], nn)
    sub = {(int(a), int(b)): float(w) for a, b, w in zip(sr, sc, sv)}
    assert set(int(a) for a in sr) <= set(range(lo, hi))
    assert min(int(a) for a in sr) >= lo and max(int(a) for a in sr) < hi
    expected = {e: w for e, w in umap_full.items() if lo <= e[0] < hi}
    assert set(sub) == set(expected), (len(sub), len(expected))
    assert max(abs(sub[e] - expected[e]) for e in expected) < 1e-4


def test_sharded_gather_order(tmp_path):
    a = np.arange(20 * 4, dtype=np.float16).reshape(20, 4)
    p0 = tmp_path / "s0.npy"; p1 = tmp_path / "s1.npy"
    np.save(p0, a[:12]); np.save(p1, a[12:])
    emb = ShardedEmbeddings([str(p0), str(p1)], expected_dim=4)
    assert len(emb) == 20
    ids = np.array([15, 0, 11, 12, 3])
    got = emb.gather(ids, out_dtype=np.float32)
    assert np.array_equal(got, a[ids].astype(np.float32))


def _write_topology(path, targets, *, n, k, sources=None):
    if sources is None:
        sources = np.repeat(np.arange(n), k).astype(np.int32)
    np.savez(path, sources=sources, targets=np.asarray(targets, np.int32),
             n_nodes=np.int64(n), k=np.int64(k), nprobe=np.int64(8))


def test_load_topology_full_scan_and_self_denominators(tmp_path):
    n, k = 8, 3
    targets = np.array([
        0, 1, 2,  0, 1, 2,  0, 1, 2,  0, 1, 2,
        0, 1, 2,  0, 1, 2,  0, 1, 2,  0, 1, 2], dtype=np.int32)
    path = tmp_path / "topology.npz"
    _write_topology(path, targets, n=n, k=k)
    _, got_n, got_k, _, stats = load_topology(
        str(path), validation_chunk_rows=2, return_stats=True)
    assert (got_n, got_k) == (n, k)
    assert stats["validation"] == "full_scan"
    assert stats["self_slots"] == 3
    assert stats["self_slot_fraction"] == pytest.approx(3 / (n * k))
    assert stats["nodes_with_self_fraction_upper_bound"] == pytest.approx(3 / n)

    bad_sources = np.repeat(np.arange(n), k).astype(np.int32)
    bad_sources[-1] = 0  # corruption beyond the old leading sample
    bad = tmp_path / "bad-source.npz"
    _write_topology(bad, targets, n=n, k=k, sources=bad_sources)
    with pytest.raises(ValueError, match="not source-sorted"):
        load_topology(str(bad), validation_chunk_rows=2)


def test_load_topology_rejects_duplicate_targets(tmp_path):
    n, k = 4, 3
    targets = np.array([1, 1, 2, 0, 2, 3, 0, 1, 3, 0, 1, 2], np.int32)
    path = tmp_path / "duplicate.npz"
    _write_topology(path, targets, n=n, k=k)
    with pytest.raises(ValueError, match="duplicate target"):
        load_topology(str(path), validation_chunk_rows=2)


def test_exact_knn_explicit_query_ids_survive_exact_vector_ties():
    pytest.importorskip("torch")
    base = np.array([[1, 0], [1, 0], [0, 1], [-1, 0]], dtype=np.float32)
    query = base[[1]]
    neighbors, _ = build_exact_knn_gpu(
        query, 2, device="cpu", base=base, query_ids=np.array([1]))
    assert 1 not in neighbors[0]
    assert 0 in neighbors[0]


def test_chunk_distance_gather_does_not_narrow_fp32_storage(tmp_path):
    pytest.importorskip("torch")
    X = np.random.RandomState(12).randn(6, 5).astype(np.float32)
    path = tmp_path / "fp32.npy"
    np.save(path, X)

    class RecordingEmbeddings(ShardedEmbeddings):
        dtypes = []

        def gather(self, ids, out_dtype=None):
            self.dtypes.append(np.dtype(out_dtype) if out_dtype is not None else None)
            return super().gather(ids, out_dtype=out_dtype)

    emb = RecordingEmbeddings([str(path)], expected_dim=5)
    neighbors = np.array([[1, 2, 3], [0, 2, 3]], dtype=np.int32)
    import torch
    chunk_knn_with_self(emb, neighbors, 0, 2, torch.device("cpu"), torch)
    assert emb.dtypes == [np.dtype("float32"), np.dtype("float32")]


def test_phase_checkpoints_are_content_bound_and_resumable(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.RandomState(9)
    n, k, dim = 40, 4, 8
    X = rng.randn(n, dim).astype(np.float32)
    xp = tmp_path / "x.npy"
    np.save(xp, X)
    emb = ShardedEmbeddings([str(xp)], expected_dim=dim)
    neighbors = np.empty((n, k), dtype=np.int32)
    for i in range(n):
        neighbors[i] = [(i + j + 1) % n for j in range(k)]
    workdir = tmp_path / "work"
    workdir.mkdir()
    contract = "a" * 64
    _, first = phase_a_forward_edges(
        emb, neighbors, n, k, k + 1, str(workdir), 13, "cpu", contract)
    closure = phase_a_closure(
        str(workdir / "fwd"), n_nodes=n, chunk_size=13,
        contract_sha256=contract)
    part_dir = phase_b_partition(
        str(workdir / "fwd"), str(workdir), n, 4,
        contract_sha256=contract,
        phase_a_closure_sha256=closure["closure_sha256"])
    _, joined = phase_c_join(
        part_dir, str(workdir), n, 4, contract_sha256=contract)

    # Every stage should validate and reuse its exact artifacts on a second pass.
    _, second = phase_a_forward_edges(
        emb, neighbors, n, k, k + 1, str(workdir), 13, "cpu", contract)
    assert second == first
    assert phase_b_partition(
        str(workdir / "fwd"), str(workdir), n, 4,
        contract_sha256=contract,
        phase_a_closure_sha256=closure["closure_sha256"]) == part_dir
    _, joined_again = phase_c_join(
        part_dir, str(workdir), n, 4, contract_sha256=contract)
    assert joined_again == joined

    # A changed contract or changed staged byte can never be mistaken for done.
    with pytest.raises(RuntimeError, match="contract_sha256"):
        phase_a_closure(
            str(workdir / "fwd"), n_nodes=n, chunk_size=13,
            contract_sha256="b" * 64)
    chunk = workdir / "fwd" / "chunk-00000.npz"
    with open(chunk, "ab") as fh:
        fh.write(b"changed")
    with pytest.raises(RuntimeError, match="byte count|sha256"):
        phase_a_closure(
            str(workdir / "fwd"), n_nodes=n, chunk_size=13,
            contract_sha256=contract)


def test_cpu_end_to_end_build_emits_admissible_content_manifest(tmp_path):
    pytest.importorskip("torch")
    from basemap.graph_validation import (
        validate_against_manifest, validate_graph_content)

    rng = np.random.RandomState(13)
    n, k, dim = 32, 4, 7
    X = rng.randn(n, dim).astype(np.float32)
    xp = tmp_path / "embeddings.npy"; np.save(xp, X)
    targets = np.asarray([
        (i + j + 1) % n for i in range(n) for j in range(k)], dtype=np.int32)
    edges = tmp_path / "uniform.npz"
    _write_topology(edges, targets, n=n, k=k)
    out = tmp_path / "weighted.npz"
    args = Namespace(
        edges=str(edges), embeddings_list=[str(xp)], embeddings_dir=None,
        corpus=None, raw_dtype=None, out=str(out), workdir=str(tmp_path / "work"),
        dim=dim, chunk_size=11, partitions=4, phase_c_workers=1,
        target_neighbors=None, device="cpu", sharded=False, no_sort=False,
        skip_gpu=False, force_gpu=True, yield_seconds=None)
    cmd_build(args)

    manifest_path = str(out) + ".manifest.json"
    with open(manifest_path) as fh:
        manifest = __import__("json").load(fh)
    assert manifest["schema"] == "graph_manifest.v2"
    assert manifest["production_trainer_ready"] is (not manifest["builder_dirty"])
    assert manifest["distance_compute_dtype"] == "float32"
    assert manifest["n_neighbors_param"] == k + 1
    assert len(manifest["graph_sha256"]) == 64
    assert len(manifest["build_contract_sha256"]) == 64
    assert manifest["input_topology_validation"]["validation"] == "full_scan"
    emb = ShardedEmbeddings([str(xp)], expected_dim=dim)
    validate_against_manifest(emb, manifest)
    trusted = validate_graph_content(
        str(out), manifest, shard_paths=[str(xp)], require_manifest_sha=True)
    assert trusted["graph_sha256"] == manifest["graph_sha256"]


def test_weighted_canary_is_no_update_and_uses_real_admission_path():
    source_path = os.path.join(
        os.path.dirname(__file__), "..", "experiments", "weighted_graph_canary.py")
    tree = ast.parse(open(source_path, encoding="utf-8").read())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    attributes = [node.func.attr for node in calls
                  if isinstance(node.func, ast.Attribute)]
    assert "_prepare_edge_list_training" in attributes
    assert "fit" not in attributes and "_init_model" not in attributes
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Dict)]
    assert any(any(isinstance(key, ast.Constant) and key.value == "training_performed"
                   for key in item.keys) for item in assignments)
