"""Shared contract for the split, quality-qualified 150M graph build."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from .round0086_program import (
    FILTER_RECEIPT_SCHEMA,
    QUALIFICATION_SCHEMA,
    ROUND_ID as STAGING_ROUND_ID,
    ROW_COUNT,
    Round0086Error,
    select_cell,
    validate_substrate,
)


ROUND_BY_CORPUS = {
    "fineweb": "0088",
    "redpajama": "0089",
    "pile": "0090",
}
CORPUS_BY_ROUND = {value: key for key, value in ROUND_BY_CORPUS.items()}
ASSEMBLY_ROUND_ID = "0091"
CORPUS_SPECS = {
    "fineweb": {
        "start": 0,
        "stop": 50_000_000,
        "excluded_rows": 1_470_724,
        "retained_rows": 48_529_276,
        "r0078_wall_us_per_retained_source": 244.41254876772425,
    },
    "redpajama": {
        "start": 50_000_000,
        "stop": 100_000_000,
        "excluded_rows": 432_547,
        "retained_rows": 49_567_453,
        "r0078_wall_us_per_retained_source": 262.7569978795701,
    },
    "pile": {
        "start": 100_000_000,
        "stop": 150_000_000,
        "excluded_rows": 874_972,
        "retained_rows": 49_125_028,
        "r0078_wall_us_per_retained_source": 275.74060818070257,
    },
}
R0081_SELECTED_SEARCH_SECONDS_PER_QUERY = 2.1966617880389094 / 10_000
R0081_SELECTED_RERANK_SECONDS_PER_QUERY = 1.2999102319590747 / 10_000
PART_SCHEMA = "round0088-split-150m-graph-part-v1"
ASSEMBLED_GRAPH_SCHEMA = "minilm-canonical-source-major-k15-v1"
ASSEMBLY_RECEIPT_SCHEMA = "round0091-balanced-150m-graph-assembly-v1"


class Round0088Error(RuntimeError):
    """The split 150M graph contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def corpus_spec(corpus: str) -> dict[str, Any]:
    try:
        return dict(CORPUS_SPECS[corpus])
    except KeyError as exc:
        raise Round0088Error(f"unknown 150M graph corpus {corpus!r}") from exc


def validate_qualification(
    path: str,
    *,
    expected_sha256: str,
    substrate_signature: Mapping[str, Any],
    filtered_index_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the positive R0086 search qualification and selection."""
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0088Error("R0086 qualification bytes changed")
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    selected = select_cell(receipt)
    if (
        receipt.get("schema") != QUALIFICATION_SCHEMA
        or receipt.get("round_id") != STAGING_ROUND_ID
        or receipt.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or receipt.get("substrate") != dict(substrate_signature)
        or receipt.get("filtered_index") != dict(filtered_index_signature)
        or not isinstance(selected, dict)
        or receipt.get("selected") != selected
    ):
        raise Round0088Error("R0086 did not release one valid passing policy")
    return {"signature": signature, "receipt": receipt, "selected": selected}


def validate_filter_receipt(
    path: str,
    *,
    expected_sha256: str,
    substrate_signature: Mapping[str, Any],
    filtered_index_signature: Mapping[str, Any],
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0088Error("R0086 filter receipt bytes changed")
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    if (
        receipt.get("schema") != FILTER_RECEIPT_SCHEMA
        or receipt.get("round_id") != STAGING_ROUND_ID
        or receipt.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or receipt.get("substrate") != dict(substrate_signature)
        or receipt.get("filtered_index") != dict(filtered_index_signature)
    ):
        raise Round0088Error("R0086 filtered-index receipt changed")
    return {"signature": signature, "receipt": receipt}


def projected_corpus_wall_seconds(
    corpus: str,
    *,
    selected_search_seconds_per_query: float,
    selected_rerank_seconds_per_query: float | None = None,
    safety_factor: float = 1.25,
    fixed_seconds: float = 300.0,
) -> float:
    """Scale R0078 wall by the slower R0086 search/rerank cost ratio."""
    spec = corpus_spec(corpus)
    if selected_search_seconds_per_query <= 0:
        raise Round0088Error("selected search time must be positive")
    search_ratio = (
        selected_search_seconds_per_query
        / R0081_SELECTED_SEARCH_SECONDS_PER_QUERY
    )
    rerank_ratio = 1.0
    if selected_rerank_seconds_per_query is not None:
        if selected_rerank_seconds_per_query <= 0:
            raise Round0088Error("selected rerank time must be positive")
        rerank_ratio = (
            selected_rerank_seconds_per_query
            / R0081_SELECTED_RERANK_SECONDS_PER_QUERY
        )
    ratio = max(search_ratio, rerank_ratio)
    observed = (
        spec["retained_rows"]
        * spec["r0078_wall_us_per_retained_source"]
        / 1_000_000
    )
    return float(observed * ratio * safety_factor + fixed_seconds)


def selected_benchmark_seconds_per_query(
    selected: Mapping[str, Any],
) -> tuple[float, float]:
    benchmark = selected.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise Round0088Error("R0086 selected cell lacks a benchmark")
    rows = int(benchmark.get("rows", 0))
    search = float(benchmark.get("median_search_seconds", 0.0))
    rerank = float(benchmark.get("median_rerank_seconds", 0.0))
    if rows <= 0 or search <= 0 or rerank <= 0:
        raise Round0088Error("R0086 selected benchmark is invalid")
    return search / rows, rerank / rows


def validate_staged_substrate(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    try:
        return validate_substrate(path, expected_sha256=expected_sha256)
    except Round0086Error as exc:
        raise Round0088Error(str(exc)) from exc


def validate_part_receipt(
    path: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if expected_sha256 is not None and signature["sha256"] != expected_sha256:
        raise Round0088Error("150M graph part receipt bytes changed")
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    corpus = str(receipt.get("corpus"))
    spec = corpus_spec(corpus)
    quality = receipt.get("quality") or {}
    if (
        receipt.get("schema") != PART_SCHEMA
        or receipt.get("round_id") != ROUND_BY_CORPUS[corpus]
        or receipt.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or int(receipt.get("start", -1)) != spec["start"]
        or int(receipt.get("stop", -1)) != spec["stop"]
        or int(receipt.get("retained_sources", -1)) != spec["retained_rows"]
        or int(receipt.get("excluded_sources", -1)) != spec["excluded_rows"]
        or int(receipt.get("valid_edges", -1)) != spec["retained_rows"] * 15
        or float(quality.get("floor", -1.0)) != 0.90
        or float(quality.get("mean_recall_at_15_unambiguous", -1.0)) < 0.90
        or int(quality.get("qualification_sample_rows", -1)) != 4_096
        or int(quality.get("qualification_sample_seed", -1)) != 86
    ):
        raise Round0088Error("150M graph part receipt contract changed")
    return {"signature": signature, "receipt": receipt}
