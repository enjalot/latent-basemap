"""Frozen contracts for the R0167 prompted OOD universality panel.

R0167 reuses the exact row selections from accepted R0142, but re-embeds every
probe and its FineWeb control with the literal ``Document: `` convention.  It
then scores two accepted 2M prompted maps and the R0166 8M prompted map.  The
round is diagnostic and cannot change any map-quality gate.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .artifact_identity import canonical_json, sha256_bytes
from .round0142_jina_universality import PROBE_ORDER
from .round0146_projection_predictors import spearman_rho


ROUND_ID = "0167"
CAPABILITY = "jina-prompted-universality-panel-v1"
DIMENSION = 768
PROMPT_PREFIX = "Document: "
QUERY_ID_OFFSET = 1_000_000_000
CONTROL_QUERY_ID_OFFSET = 1_500_000_000
SOURCE_MAP = "r0107-25m-seed42"
PROMPTED_MAP_ORDER = (
    "r0115-prompted-2m-seed42",
    "r0117-prompted-2m-seed43",
    "r0166-prompted-8m-seed42",
)
PASS_RETENTION = 0.70
FAIL_RETENTION = 0.50
EMBED_CHUNK_ROWS = 5_000
EMBED_MINIMUM_ROWS_PER_S = 120.0


class Round0167Error(RuntimeError):
    """The preregistered R0167 contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0167Error(f"{label} identity seal is invalid")


def source_rows_from_coordinate_archive(
    corpus_ids: np.ndarray,
    query_ids: np.ndarray,
    *,
    label: str,
    separate_sources: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover R0142 source rows from its immutable coordinate row IDs."""
    corpus = np.asarray(corpus_ids, dtype=np.int64)
    queries = np.asarray(query_ids, dtype=np.int64)
    if (
        corpus.ndim != 1
        or queries.ndim != 1
        or len(corpus) < 50
        or len(queries) < 10
        or np.any(corpus < 0)
        or np.any(queries < QUERY_ID_OFFSET)
        or np.any(queries >= CONTROL_QUERY_ID_OFFSET)
        or len(np.unique(corpus)) != len(corpus)
        or len(np.unique(queries)) != len(queries)
    ):
        raise Round0167Error(f"{label} R0142 row IDs changed")
    query_rows = queries - QUERY_ID_OFFSET
    if (
        not separate_sources
        and np.intersect1d(corpus, query_rows, assume_unique=False).size
    ):
        raise Round0167Error(f"{label} source rows overlap across splits")
    return corpus, query_rows


def control_rows_from_coordinate_archive(
    corpus_ids: np.ndarray,
    query_ids: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    corpus = np.asarray(corpus_ids, dtype=np.int64)
    queries = np.asarray(query_ids, dtype=np.int64)
    if (
        corpus.ndim != 1
        or queries.ndim != 1
        or len(corpus) < 50
        or len(queries) < 10
        or np.any(corpus < 0)
        or np.any(corpus >= 60_000)
        or np.any(queries < CONTROL_QUERY_ID_OFFSET)
        or len(np.unique(corpus)) != len(corpus)
        or len(np.unique(queries)) != len(queries)
    ):
        raise Round0167Error(f"{label} R0142 control row IDs changed")
    query_rows = queries - CONTROL_QUERY_ID_OFFSET
    if np.any(query_rows >= 60_000) or np.intersect1d(corpus, query_rows).size:
        raise Round0167Error(f"{label} control rows overlap or exceed 60K")
    return corpus, query_rows


def retention_verdict(value: float) -> str:
    if not np.isfinite(value) or value < 0:
        raise Round0167Error("retention must be finite and nonnegative")
    if value >= PASS_RETENTION:
        return "pass"
    if value < FAIL_RETENTION:
        return "named-failure"
    return "amber"


def twonn_correlations(cells: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Frozen per-map and descriptive pooled TwoNN association table."""
    rows = [dict(item) for item in cells]
    expected = {
        (map_key, probe) for map_key in PROMPTED_MAP_ORDER for probe in PROBE_ORDER
    }
    observed = {(str(item["map"]), str(item["probe"])) for item in rows}
    if observed != expected or len(rows) != len(expected):
        raise Round0167Error("prompted universality cell matrix is incomplete")
    output: list[dict[str, Any]] = []
    for outcome in ("ffr_retention", "recall10_retention"):
        for scope in (*PROMPTED_MAP_ORDER, "pooled-descriptive"):
            selected = rows if scope == "pooled-descriptive" else [
                item for item in rows if item["map"] == scope
            ]
            selected = [item for item in selected if item.get(outcome) is not None]
            rho = spearman_rho(
                [float(item["twonn_intrinsic_dimension"]) for item in selected],
                [float(item[outcome]) for item in selected],
            )
            output.append({
                "outcome": outcome,
                "scope": scope,
                "predictor": "twonn_intrinsic_dimension",
                "cells": len(selected),
                "spearman_rho": rho,
                "raw_r0146_hypothesized_direction": "negative",
                "direction_consistent": rho < 0,
                "independence_note": (
                    "pooled rows repeat probe geometry across three maps"
                    if scope == "pooled-descriptive"
                    else "eleven probes; exploratory, not a significance test"
                ),
            })
    return output
