"""Cohort-correct neighbour metrics with mandatory self-exclusion."""
from __future__ import annotations

import numpy as np

from .experiment_contract import ids_identity


def validate_cohorts(cohorts: dict[str, np.ndarray], universe_ids=None) -> dict:
    normalized = {}
    for name, values in cohorts.items():
        ids = np.asarray(values)
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError(f"cohort {name} must be a one-dimensional integer array")
        ids = ids.astype(np.int64, copy=False)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"cohort {name} contains duplicate IDs")
        normalized[name] = ids
    names = sorted(normalized)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            overlap = np.intersect1d(normalized[left], normalized[right], assume_unique=True)
            if len(overlap):
                raise ValueError(f"cohorts {left}/{right} overlap by {len(overlap)} IDs")
    union = np.concatenate(list(normalized.values())) if normalized else np.array([], np.int64)
    if universe_ids is not None and not np.array_equal(
            np.sort(union), np.sort(np.asarray(universe_ids, dtype=np.int64))):
        raise ValueError("cohorts are not exhaustive for the declared universe")
    return {name: {"count": len(ids), "ids_sha256": ids_identity(ids)}
            for name, ids in sorted(normalized.items())}


def exclude_self(neighbours: np.ndarray, query_ids: np.ndarray, k: int) -> np.ndarray:
    neighbours = np.asarray(neighbours, dtype=np.int64)
    query_ids = np.asarray(query_ids, dtype=np.int64)
    if neighbours.ndim != 2 or len(neighbours) != len(query_ids):
        raise ValueError("neighbour/query shape mismatch")
    out = np.empty((len(query_ids), k), dtype=np.int64)
    for row, query_id in enumerate(query_ids):
        kept = neighbours[row][neighbours[row] != query_id]
        if len(np.unique(kept)) != len(kept):
            raise ValueError(f"query {query_id} has duplicate neighbour IDs")
        if len(kept) < k:
            raise ValueError(f"query {query_id} has only {len(kept)} self-excluded neighbours < {k}")
        out[row] = kept[:k]
    return out


def retention_and_jaccard(left: np.ndarray, right: np.ndarray, *, query_ids: np.ndarray,
                          k: int) -> dict:
    """Compare matched neighbourhoods after enforcing self-exclusion internally."""
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("matched neighbour matrices with at least k columns are required")
    left = exclude_self(left, query_ids, k)
    right = exclude_self(right, query_ids, k)
    ret, jac = [], []
    for a, b in zip(left[:, :k], right[:, :k]):
        sa, sb = set(a.tolist()), set(b.tolist())
        intersection = len(sa & sb)
        ret.append(intersection / k)
        jac.append(intersection / len(sa | sb))
    return {"retention_at_k": float(np.mean(ret)), "true_jaccard_at_k": float(np.mean(jac)),
            "k": int(k), "self_excluded": True,
            "query_ids_sha256": ids_identity(np.asarray(query_ids, dtype=np.int64))}
