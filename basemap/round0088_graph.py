"""Shared contract for the R0097-R0100 IVF32768 150M graph build.

The filename is retained because the graph kernel was originally prepared for
the withdrawn R0088-R0091 chain.  This module now authenticates the corrected
R0096 evidence and emits only the replacement R0097-R0100 schemas.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from .round0086_program import Round0086Error, validate_substrate
from .round0096_larger_nlist import (
    DECISION_SCHEMA,
    DIMENSION,
    GLOBAL_MEAN_FLOOR,
    INDEX_SCHEMA,
    NLIST,
    PER_CORPUS_MEAN_FLOOR,
    POLICY_GRID,
    PQ_BITS,
    PQ_M,
    QUALITY_ROWS,
    QUALITY_SAMPLE_SHA256,
    QUALITY_SEED,
    QUALIFICATION_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID as SEARCH_ROUND_ID,
    ROW_COUNT,
    select_cell,
)


ROUND_BY_CORPUS = {
    "fineweb": "0097",
    "redpajama": "0098",
    "pile": "0099",
}
CORPUS_BY_ROUND = {value: key for key, value in ROUND_BY_CORPUS.items()}
ASSEMBLY_ROUND_ID = "0100"
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
R0081_SELECTED_TOTAL_SECONDS_PER_QUERY = (
    R0081_SELECTED_SEARCH_SECONDS_PER_QUERY
    + R0081_SELECTED_RERANK_SECONDS_PER_QUERY
)
PART_SCHEMA = "round0097-balanced-150m-ivf32768-graph-part-v1"
ASSEMBLED_GRAPH_SCHEMA = "minilm-canonical-source-major-k15-v1"
ASSEMBLY_RECEIPT_SCHEMA = "round0100-balanced-150m-graph-assembly-v1"


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


def _load_sealed_json(
    path: str,
    *,
    expected_sha256: str,
    schema: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0088Error(f"{schema} bytes changed")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    body = {
        key: item
        for key, item in value.items()
        if key != "identity_sha256"
    }
    if (
        value.get("schema") != schema
        or value.get("identity_sha256") != sha256_bytes(canonical_json(body))
    ):
        raise Round0088Error(f"{schema} content seal is invalid")
    return {"signature": signature, "receipt": value}


def validate_index_receipt(
    path: str,
    *,
    expected_sha256: str,
    substrate_signature: Mapping[str, Any],
    index_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the exact R0096 retained IVF32768 index receipt."""
    loaded = _load_sealed_json(
        path, expected_sha256=expected_sha256, schema=INDEX_SCHEMA,
    )
    receipt = loaded["receipt"]
    geometry = receipt.get("geometry") or {}
    validation = receipt.get("id_validation") or {}
    if (
        receipt.get("round_id") != SEARCH_ROUND_ID
        or receipt.get("substrate") != dict(substrate_signature)
        or receipt.get("index") != dict(index_signature)
        or geometry.get("class") != "IndexIVFPQ"
        or int(geometry.get("dimension", -1)) != DIMENSION
        or int(geometry.get("nlist", -1)) != NLIST
        or int(geometry.get("ntotal", -1)) != RETAINED_ROWS
        or int(geometry.get("code_size", -1)) != PQ_M
        or int(geometry.get("pq_m", -1)) != PQ_M
        or int(geometry.get("pq_bits", -1)) != PQ_BITS
        or int(geometry.get("metric_type", -1)) != 0
        or validation.get("global_ids_unique") is not True
        or validation.get("excluded_rows_absent") is not True
        or int(validation.get("seen_retained_rows", -1)) != RETAINED_ROWS
        or receipt.get("training_performed") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
    ):
        raise Round0088Error("R0096 retained IVF32768 index contract changed")
    return loaded


def validate_qualification(
    path: str,
    *,
    expected_sha256: str,
    substrate_signature: Mapping[str, Any],
    index_signature: Mapping[str, Any],
    index_receipt_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the positive, corrected R0096 policy qualification."""
    loaded = _load_sealed_json(
        path, expected_sha256=expected_sha256, schema=QUALIFICATION_SCHEMA,
    )
    receipt = loaded["receipt"]
    cells = receipt.get("cells")
    selected = select_cell(cells) if isinstance(cells, Mapping) else None
    quality = receipt.get("quality") or {}
    checks = receipt.get("checks") or {}
    geometry = receipt.get("geometry") or {}
    if (
        receipt.get("round_id") != SEARCH_ROUND_ID
        or receipt.get("substrate") != dict(substrate_signature)
        or receipt.get("index") != dict(index_signature)
        or receipt.get("index_receipt") != dict(index_receipt_signature)
        or receipt.get("validity_passed") is not True
        or receipt.get("training_performed") is not False
        or receipt.get("scale_decision_made") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
        or not isinstance(cells, Mapping)
        or len(cells) != len(POLICY_GRID)
        or not isinstance(selected, dict)
        or receipt.get("selected") != selected
        or int(selected.get("nprobe", -1)) != 128
        or int(selected.get("shortlist_width", -1)) != 512
        or selected.get("passes_global_floor") is not True
        or selected.get("passes_every_corpus_floor") is not True
        or float(selected.get("mean_recall_at_15_unambiguous", -1.0))
        < GLOBAL_MEAN_FLOOR
        or int(quality.get("sample_rows", -1)) != QUALITY_ROWS
        or int(quality.get("sample_seed", -1)) != QUALITY_SEED
        or quality.get("sample_sha256") != QUALITY_SAMPLE_SHA256
        or float(quality.get("global_mean_floor", -1.0))
        != GLOBAL_MEAN_FLOOR
        or float(quality.get("per_corpus_mean_floor", -1.0))
        != PER_CORPUS_MEAN_FLOOR
        or any(value is not True for value in checks.values())
        or receipt.get("failed_checks") != []
        or int(geometry.get("nlist", -1)) != NLIST
        or int(geometry.get("ntotal", -1)) != RETAINED_ROWS
    ):
        raise Round0088Error(
            "R0096 did not release the exact corrected passing policy"
        )
    return {**loaded, "selected": selected}


def validate_decision(
    path: str,
    *,
    expected_sha256: str,
    qualification_signature: Mapping[str, Any],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    loaded = _load_sealed_json(
        path, expected_sha256=expected_sha256, schema=DECISION_SCHEMA,
    )
    receipt = loaded["receipt"]
    if (
        receipt.get("round_id") != SEARCH_ROUND_ID
        or receipt.get("qualification") != dict(qualification_signature)
        or receipt.get("selected") != dict(selected)
        or receipt.get("outcome") != "qualified"
        or receipt.get("validity_passed") is not True
        or receipt.get("graph_build_released") is not True
        or receipt.get("training_performed") is not False
        or receipt.get("scale_decision_made") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
    ):
        raise Round0088Error("R0096 search decision does not release graph build")
    return loaded


def projected_corpus_wall_seconds(
    corpus: str,
    *,
    selected_total_seconds_per_query: float,
    safety_factor: float = 1.25,
    fixed_seconds: float = 300.0,
) -> float:
    """Scale R0078 wall by R0096/R0081 combined cost, never a speedup."""
    spec = corpus_spec(corpus)
    if selected_total_seconds_per_query <= 0:
        raise Round0088Error("selected combined time must be positive")
    ratio = max(
        1.0,
        selected_total_seconds_per_query
        / R0081_SELECTED_TOTAL_SECONDS_PER_QUERY,
    )
    observed = (
        spec["retained_rows"]
        * spec["r0078_wall_us_per_retained_source"]
        / 1_000_000
    )
    return float(observed * ratio * safety_factor + fixed_seconds)


def selected_benchmark_seconds_per_query(
    selected: Mapping[str, Any],
) -> float:
    benchmark = selected.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise Round0088Error("R0096 selected cell lacks a benchmark")
    value = float(benchmark.get("median_wall_seconds_per_query", 0.0))
    queries = int(benchmark.get("queries", 0))
    if queries <= 0 or value <= 0:
        raise Round0088Error("R0096 selected combined benchmark is invalid")
    return value


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
    selected_by_corpus = quality.get("selected_by_corpus") or {}
    corpus_quality = selected_by_corpus.get(corpus) or {}
    if (
        receipt.get("schema") != PART_SCHEMA
        or receipt.get("round_id") != ROUND_BY_CORPUS[corpus]
        or receipt.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or int(receipt.get("start", -1)) != spec["start"]
        or int(receipt.get("stop", -1)) != spec["stop"]
        or int(receipt.get("retained_sources", -1)) != spec["retained_rows"]
        or int(receipt.get("excluded_sources", -1)) != spec["excluded_rows"]
        or int(receipt.get("valid_edges", -1)) != spec["retained_rows"] * 15
        or float(quality.get("global_mean_floor", -1.0))
        != GLOBAL_MEAN_FLOOR
        or float(quality.get("per_corpus_mean_floor", -1.0))
        != PER_CORPUS_MEAN_FLOOR
        or float(quality.get("selected_global_mean_recall", -1.0))
        < GLOBAL_MEAN_FLOOR
        or float(corpus_quality.get("mean_recall_at_15_unambiguous", -1.0))
        < PER_CORPUS_MEAN_FLOOR
        or int(quality.get("qualification_sample_rows", -1)) != QUALITY_ROWS
        or int(quality.get("qualification_sample_seed", -1)) != QUALITY_SEED
        or quality.get("qualification_sample_sha256")
        != QUALITY_SAMPLE_SHA256
    ):
        raise Round0088Error("150M graph part receipt contract changed")
    return {"signature": signature, "receipt": receipt}
