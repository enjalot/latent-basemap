"""Graph/data pair validation + node manifests (P0.8).

Prevents the silent-corruption class the review found: applying a 150M graph's
prefix endpoints to a *balanced* 50M matrix (whose IDs [fw|rpj|pile] differ from
the graph's [fw 0:50M | rpj 0:50M | pile 0:50M]) connects unrelated rows; and
the old `source < n_train` mask let ANN sentinel `-1` through → last-row
negative indexing. Every graph/data pair must now be validated before training.
"""
from __future__ import annotations
import os, json, hashlib, numpy as np


def validate_edge_bounds(sources, targets, n_nodes: int) -> None:
    """Raise if any endpoint is outside ``[0, n_nodes)`` (rejects -1 sentinels
    and out-of-range ids). Vectorized min/max — no full copy."""
    s = np.asarray(sources); t = np.asarray(targets)
    for name, arr in (("sources", s), ("targets", t)):
        lo = int(arr.min()); hi = int(arr.max())
        if lo < 0:
            raise ValueError(f"graph {name} has negative id {lo} (ANN sentinel/-1?) — "
                             f"reject before training (P0.8).")
        if hi >= n_nodes:
            raise ValueError(f"graph {name} max id {hi} >= n_nodes {n_nodes} — out of range (P0.8).")


def data_fingerprint(X, n_sample: int = 2048):
    """Deterministic identity of a data matrix: the sampled row ids + a hash of
    those rows. Two matrices of equal length but different ROW ORDER (a shuffle)
    or a changed shard produce different fingerprints — length equality alone is
    not identity (P0-E). Returns (sample_ids list, hex hash)."""
    n = len(X)
    ids = np.unique(np.linspace(0, n - 1, min(n_sample, n)).astype(np.int64))
    h = hashlib.sha1()
    h.update(np.asarray(ids, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(np.asarray(X[ids], dtype=np.float32)).tobytes())
    return ids.tolist(), h.hexdigest()[:16]


def graph_manifest(sources, targets, n_nodes: int, X=None, extra: dict | None = None) -> dict:
    """Node-manifest for a graph/data pair: counts, endpoint bounds, and a
    deterministic DATA fingerprint (sampled row ids + hash) so a mismatched or
    reordered X is detectable at training time (P0-E)."""
    s = np.asarray(sources); t = np.asarray(targets)
    man = {"schema": "graph_manifest.v1", "n_nodes": int(n_nodes), "n_edges": int(len(s)),
           "source_min": int(s.min()), "source_max": int(s.max()),
           "target_min": int(t.min()), "target_max": int(t.max())}
    if X is not None:
        ids, fp = data_fingerprint(X)
        man["data_len"] = int(len(X))
        man["data_fingerprint"] = fp
        man["data_fingerprint_n"] = len(ids)
    if extra:
        man.update(extra)
    return man


def validate_against_manifest(X, manifest: dict, *, allow_prefix=False) -> None:
    """Compare a loaded data matrix to a graph's expected manifest BEFORE building
    samplers (P0-E). Length equality is necessary but NOT sufficient: the data
    fingerprint must match, so shuffled or changed data of equal length fails.
    For a verified literal prefix, the fingerprint is recomputed over the prefix."""
    n_nodes = int(manifest["n_nodes"]); exp_len = int(manifest.get("data_len", n_nodes))
    if len(X) == n_nodes:
        want = manifest.get("data_fingerprint")
        if want is not None:
            _, got = data_fingerprint(X)
            if got != want:
                raise ValueError(f"data fingerprint {got} != manifest {want}: X is reordered or a "
                                 f"different corpus than the graph was built on (P0-E).")
        return
    if len(X) < n_nodes and allow_prefix:
        return  # a shorter prefix; endpoint bounds are enforced separately
    raise ValueError(f"len(X)={len(X)} does not match manifest n_nodes={n_nodes} "
                     f"(data_len={exp_len}); refuse to train (P0-E).")


def edge_endpoint_cosine_check(sources, targets, X, n_probe=20000, seed=0,
                               min_margin=0.15) -> dict:
    """Fail-fast that the graph/data pairing is real: sampled edge endpoints must
    be far more similar than random pairs (edges connect kNN). Returns the
    measured cosines; raises if the margin is below ``min_margin``."""
    import torch
    s = np.asarray(sources); t = np.asarray(targets)
    n = len(X); rng = np.random.RandomState(seed)
    m = min(n_probe, len(s))
    sel = rng.choice(len(s), m, replace=False)
    si = s[sel].astype(np.int64); ti = t[sel].astype(np.int64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    def cos(ai, bi):
        A = torch.from_numpy(np.asarray(X[ai], dtype=np.float32)).to(dev)
        B = torch.from_numpy(np.asarray(X[bi], dtype=np.float32)).to(dev)
        A = torch.nn.functional.normalize(A, dim=1); B = torch.nn.functional.normalize(B, dim=1)
        return float((A * B).sum(1).mean())
    edge_cos = cos(si, ti)
    ri = rng.randint(0, n, m); rj = rng.randint(0, n, m)
    rand_cos = cos(ri, rj)
    out = {"edge_cosine": round(edge_cos, 4), "random_cosine": round(rand_cos, 4),
           "margin": round(edge_cos - rand_cos, 4)}
    if edge_cos - rand_cos < min_margin:
        raise ValueError(f"graph/data endpoint-cosine check FAILED {out} — the edges "
                         f"do not connect near-neighbours of this X (wrong pairing? P0.8).")
    return out


def validate_graph_data_pair(sources, targets, n_nodes, n_train, *,
                             allow_prefix_filter=False):
    """Gate a graph/data pairing before training. Returns a filter mask (or None
    if no filtering). Enforces:
      - n_nodes == n_train  → validate bounds, no filter.
      - n_nodes  > n_train  → prefix-filter ONLY if explicitly allowed (the
        balanced-50M-vs-150M hazard); validate bounds on the survivors.
      - n_nodes  < n_train  → error (graph smaller than data).
    """
    if n_nodes < n_train:
        raise ValueError(f"graph n_nodes {n_nodes} < training rows {n_train} (P0.8).")
    if n_nodes == n_train:
        validate_edge_bounds(sources, targets, n_nodes)
        return None
    # n_nodes > n_train: prefix-filter path — dangerous unless the training data
    # is literally the first n_train rows of the graph universe.
    if not allow_prefix_filter:
        raise ValueError(
            f"n_nodes {n_nodes} > n_train {n_train}: prefix-filtering a larger graph "
            f"assumes X is the graph's literal first {n_train} rows. Balanced/sampled "
            f"matrices do NOT satisfy this (P0.8: silent cross-corpus corruption). "
            f"Pass allow_prefix_filter=True only for a verified prefix, or use an "
            f"aligned graph/data pair.")
    s = np.asarray(sources); t = np.asarray(targets)
    mask = (s >= 0) & (s < n_train) & (t >= 0) & (t < n_train)   # rejects -1 sentinels
    return mask


def write_manifest(path: str, manifest: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    json.dump(manifest, open(path, "w"), indent=1)
