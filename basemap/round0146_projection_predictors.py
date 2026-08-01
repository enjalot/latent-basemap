"""Frozen, CPU-only predictor calculations for campaign Track B2.

The estimand is deliberately exploratory: which high-dimensional probe
properties co-vary with R0142's observed projection-retention ratios?  These
helpers do not train or transform a map and do not alter any quality gate.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, ordered_array_sha256, sha256_bytes
from .round0142_jina_universality import MAP_ORDER, PROBE_ORDER


ROUND_ID = "0146"
CAPABILITY = "jina-diverse-projection-loss-predictors-v1"
DIMENSION = 768
GEOMETRY_SAMPLE_ROWS = 2_048
TRAINING_SUPPORT_ROWS = 8_192
HUBNESS_K = 10
BLAS_THREADS = 12
QUERY_ID_OFFSET = 1_000_000_000
PRIMARY_OUTCOME = "ffr_retention"
SECONDARY_OUTCOME = "recall10_retention"
PREDICTOR_ORDER = (
    "twonn_intrinsic_dimension",
    "hubness_k10_skew",
    "anisotropy_eigen_ratio",
    "support_cosine_distance_p50",
    "support_cosine_distance_p90",
)


class Round0146Error(RuntimeError):
    """The preregistered R0146 predictor contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0146Error(f"{label} identity seal is invalid")


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256(
        ("round0146:" + ":".join(parts)).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF


def systematic_positions(length: int, count: int, *, seed: int) -> np.ndarray:
    """Select one deterministic position from each equal-width order stratum."""
    if length <= 0 or count <= 0 or count > length:
        raise Round0146Error("systematic sample geometry is invalid")
    rng = np.random.RandomState(seed)
    offset = float(rng.uniform(0.0, 1.0))
    positions = np.floor(
        (np.arange(count, dtype=np.float64) + offset) * length / count
    ).astype(np.int64)
    if (
        positions.shape != (count,)
        or positions[0] < 0
        or positions[-1] >= length
        or np.any(positions[1:] <= positions[:-1])
    ):
        raise Round0146Error("systematic sample did not produce unique positions")
    return positions


def _unit_rows(values: np.ndarray, *, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != DIMENSION or len(array) < 3:
        raise Round0146Error(f"{label} embedding geometry changed")
    if not np.isfinite(array).all():
        raise Round0146Error(f"{label} contains nonfinite embeddings")
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms < 0.90) or np.any(norms > 1.10):
        raise Round0146Error(f"{label} normalization guard failed")
    normalized = np.ascontiguousarray(array / norms[:, None], dtype=np.float32)
    return normalized, {
        "rows": int(len(array)),
        "input_norm_minimum": float(norms.min()),
        "input_norm_median": float(np.median(norms)),
        "input_norm_maximum": float(norms.max()),
    }


def _hubness_skew(counts: np.ndarray) -> float:
    values = np.asarray(counts, dtype=np.float64)
    centered = values - values.mean()
    scale = float(np.sqrt(np.mean(centered * centered)))
    if not scale > 0.0:
        return 0.0
    return float(np.mean((centered / scale) ** 3))


def geometry_predictors(
    values: np.ndarray,
    *,
    source_row_ids: np.ndarray,
    label: str,
) -> dict[str, Any]:
    """Compute exact sample-level TwoNN, hubness, and anisotropy predictors."""
    source_ids = np.asarray(source_row_ids, dtype=np.int64)
    if source_ids.ndim != 1 or len(source_ids) != len(values):
        raise Round0146Error(f"{label} source-row IDs changed")
    rows = min(GEOMETRY_SAMPLE_ROWS, len(source_ids))
    positions = systematic_positions(
        len(source_ids), rows, seed=stable_seed(label, "geometry")
    )
    sample_ids = source_ids[positions]
    sample, norm_guard = _unit_rows(
        np.asarray(values)[positions], label=f"{label} geometry sample"
    )

    similarity = np.asarray(sample @ sample.T, dtype=np.float32)
    np.fill_diagonal(similarity, -np.inf)
    nearest = np.argpartition(similarity, -HUBNESS_K, axis=1)[:, -HUBNESS_K:]
    nearest_similarity = np.take_along_axis(similarity, nearest, axis=1)
    order = np.argsort(nearest_similarity, axis=1)[:, ::-1]
    nearest = np.take_along_axis(nearest, order, axis=1)
    nearest_similarity = np.take_along_axis(nearest_similarity, order, axis=1)

    r1 = np.sqrt(np.maximum(2.0 - 2.0 * nearest_similarity[:, 0], 1e-12))
    r2 = np.sqrt(np.maximum(2.0 - 2.0 * nearest_similarity[:, 1], 1e-12))
    log_ratio = np.log(r2 / r1)
    valid = np.isfinite(log_ratio) & (log_ratio > 1e-12)
    if int(valid.sum()) < max(32, int(0.90 * rows)):
        raise Round0146Error(f"{label} TwoNN valid-row guard failed")
    twonn = float(1.0 / np.mean(log_ratio[valid]))

    occurrences = np.bincount(nearest.reshape(-1), minlength=rows)
    centered = np.asarray(sample - sample.mean(axis=0), dtype=np.float64)
    covariance = centered.T @ centered / max(1, rows - 1)
    eigenvalues = np.linalg.eigvalsh(covariance)
    positive = eigenvalues[eigenvalues > max(float(eigenvalues[-1]), 1.0) * 1e-12]
    if len(positive) < 2 or not np.isfinite(positive).all():
        raise Round0146Error(f"{label} covariance spectrum collapsed")
    eigen_ratio = float(positive[-1] / positive.mean())

    return {
        "sample": {
            "policy": "one seeded systematic row per equal-width corpus-order stratum",
            "seed": stable_seed(label, "geometry"),
            "input_rows": int(len(source_ids)),
            "sample_rows": rows,
            "sample_positions_sha256": ordered_array_sha256(positions),
            "source_row_ids_sha256": ordered_array_sha256(sample_ids),
            "normalization_guard": norm_guard,
        },
        "twonn": {
            "metric": "1 / mean(log(r2 / r1)) on exact Euclidean neighbours of unit vectors",
            "intrinsic_dimension": twonn,
            "valid_rows": int(valid.sum()),
            "r1_median": float(np.median(r1[valid])),
            "r2_median": float(np.median(r2[valid])),
        },
        "hubness": {
            "metric": "standardized third moment of exact k10 occurrence counts",
            "k": HUBNESS_K,
            "skew": _hubness_skew(occurrences),
            "mean_occurrence": float(occurrences.mean()),
            "maximum_occurrence": int(occurrences.max()),
            "zero_occurrence_fraction": float(np.mean(occurrences == 0)),
            "occurrence_counts_sha256": ordered_array_sha256(
                occurrences.astype(np.int64, copy=False)
            ),
        },
        "anisotropy": {
            "metric": "largest positive covariance eigenvalue / positive-eigenvalue mean",
            "eigen_ratio": eigen_ratio,
            "top_eigenvalue_fraction": float(positive[-1] / positive.sum()),
            "positive_eigenvalues": int(len(positive)),
            "eigenvalues_sha256": ordered_array_sha256(eigenvalues),
        },
    }


def support_distance_predictor(
    queries: np.ndarray,
    support: np.ndarray,
    *,
    label: str,
    block_rows: int = 128,
) -> dict[str, Any]:
    """Exact nearest sampled-training-support cosine distances for B1 queries."""
    query, query_guard = _unit_rows(queries, label=f"{label} queries")
    support_rows, support_guard = _unit_rows(support, label=f"{label} support")
    nearest = np.full(len(query), -np.inf, dtype=np.float32)
    for start in range(0, len(query), block_rows):
        stop = min(start + block_rows, len(query))
        nearest[start:stop] = np.max(query[start:stop] @ support_rows.T, axis=1)
    distance = np.asarray(1.0 - nearest, dtype=np.float64)
    if np.any(distance < -1e-5) or not np.isfinite(distance).all():
        raise Round0146Error(f"{label} support distances are invalid")
    distance = np.maximum(distance, 0.0)
    return {
        "metric": "one minus exact maximum cosine to deterministic sampled train support",
        "query_rows": int(len(query)),
        "support_rows": int(len(support_rows)),
        "minimum": float(distance.min()),
        "p50": float(np.percentile(distance, 50)),
        "p90": float(np.percentile(distance, 90)),
        "p99": float(np.percentile(distance, 99)),
        "maximum": float(distance.max()),
        "distances_sha256": ordered_array_sha256(distance),
        "query_normalization_guard": query_guard,
        "support_normalization_guard": support_guard,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise Round0146Error("rank input must be one-dimensional and finite")
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    cursor = 0
    while cursor < len(array):
        stop = cursor + 1
        while stop < len(array) and array[order[stop]] == array[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * (cursor + stop - 1)
        cursor = stop
    return ranks


def spearman_rho(left: Sequence[float], right: Sequence[float]) -> float:
    x = _average_ranks(np.asarray(left, dtype=np.float64))
    y = _average_ranks(np.asarray(right, dtype=np.float64))
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    if not denominator > 0:
        raise Round0146Error("Spearman rank variance collapsed")
    result = float(np.dot(x, y) / denominator)
    if not math.isfinite(result):
        raise Round0146Error("Spearman correlation is nonfinite")
    return result


def correlation_table(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Report preregistered per-map and descriptive pooled rank correlations."""
    rows = [dict(item) for item in cells]
    expected = {(map_key, probe) for map_key in MAP_ORDER for probe in PROBE_ORDER}
    observed = {(str(item["map"]), str(item["probe"])) for item in rows}
    if observed != expected or len(rows) != len(expected):
        raise Round0146Error("predictor cell matrix is incomplete")
    output: list[dict[str, Any]] = []
    for outcome in (PRIMARY_OUTCOME, SECONDARY_OUTCOME):
        for scope in (*MAP_ORDER, "pooled-descriptive"):
            selected = rows if scope == "pooled-descriptive" else [
                item for item in rows if item["map"] == scope
            ]
            selected = [item for item in selected if item.get(outcome) is not None]
            for predictor in PREDICTOR_ORDER:
                rho = spearman_rho(
                    [float(item[predictor]) for item in selected],
                    [float(item[outcome]) for item in selected],
                )
                output.append({
                    "outcome": outcome,
                    "scope": scope,
                    "predictor": predictor,
                    "cells": len(selected),
                    "spearman_rho": rho,
                    "hypothesized_direction": "negative",
                    "direction_consistent": rho < 0,
                    "independence_note": (
                        "pooled rows repeat probe-level geometry across two maps"
                        if scope == "pooled-descriptive"
                        else "eleven probes; exploratory, not a significance test"
                    ),
                })
    return output

