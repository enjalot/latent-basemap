"""Contract helpers for the retained diverse-Jina fuzzy graph."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0105_search import (
    DECISION_SCHEMA as SEARCH_DECISION_SCHEMA,
    DIMENSION,
    GROUPS,
    INDEX_SCHEMA as SEARCH_INDEX_SCHEMA,
    K,
    NLIST,
    PQ_BITS,
    PQ_M,
    QUALIFICATION_SCHEMA as SEARCH_QUALIFICATION_SCHEMA,
    RETAINED_ROWS,
    ROW_COUNT,
    select_cell,
)


ROUND_ID = "0106"
N_NEIGHBORS = K + 1
LOCAL_CONNECTIVITY = 1.0
SHARD_ROWS = 100_000
SEARCH_BATCH_ROWS = 4_096
RERANK_BATCH_ROWS = 512
PAIR_BUCKETS = 128
PERFORMANCE_WARMUP_SHARDS = 1
MINIMUM_SHARD_SOURCES_PER_SECOND = 500.0
PERFORMANCE_SUBFLOOR_PATIENCE = 2

PART_SCHEMA = "round0106-jina-diverse-25m-fuzzy-graph-part-v1"
GRAPH_SCHEMA = "round0106-jina-diverse-25m-fuzzy-graph-v1"
SHARD_SCHEMA = "round0106-jina-diverse-25m-fuzzy-graph-shard-v1"

PARTS = {
    "english": {
        "global_start": 0,
        "global_stop": 9_126_376,
        "compact_start": 0,
        "compact_stop": 9_079_287,
        "retained_rows": 9_079_287,
        "excluded_rows": 47_089,
        "group_start": 0,
        "group_stop": 3,
    },
    "languages-a": {
        "global_start": 9_126_376,
        "global_stop": 16_645_462,
        "compact_start": 9_079_287,
        "compact_stop": 16_596_284,
        "retained_rows": 7_516_997,
        "excluded_rows": 2_089,
        "group_start": 3,
        "group_stop": 12,
    },
    "languages-b": {
        "global_start": 16_645_462,
        "global_stop": 25_000_000,
        "compact_start": 16_596_284,
        "compact_stop": 24_948_663,
        "retained_rows": 8_352_379,
        "excluded_rows": 2_159,
        "group_start": 12,
        "group_stop": 22,
    },
}


class Round0106Error(RuntimeError):
    """The R0106 graph contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def part_spec(name: str) -> dict[str, int]:
    try:
        return dict(PARTS[name])
    except KeyError as exc:
        raise Round0106Error(f"unknown R0106 graph part {name!r}") from exc


def update_performance_streak(
    current: int,
    *,
    completed_new_shards: int,
    sources_per_second: float,
) -> int:
    """Track consecutive grossly infeasible shard rates after warmup."""
    rate = float(sources_per_second)
    if not np.isfinite(rate) or rate <= 0:
        raise Round0106Error("R0106 shard throughput is nonfinite/nonpositive")
    if completed_new_shards <= PERFORMANCE_WARMUP_SHARDS:
        return 0
    return (
        current + 1
        if rate < MINIMUM_SHARD_SOURCES_PER_SECOND
        else 0
    )


def membership(sorted_rows: np.ndarray, values: np.ndarray) -> np.ndarray:
    rows = np.asarray(sorted_rows, dtype=np.int64)
    query = np.asarray(values, dtype=np.int64)
    if not len(rows):
        return np.zeros(query.shape, dtype=bool)
    positions = np.searchsorted(rows, query, side="left")
    bounded = positions < len(rows)
    result = np.zeros(query.shape, dtype=bool)
    result[bounded] = rows[positions[bounded]] == query[bounded]
    return result


def global_to_compact(
    global_rows: np.ndarray,
    excluded_rows: np.ndarray,
) -> np.ndarray:
    """Map retained original IDs to the contiguous retained universe."""
    rows = np.asarray(global_rows, dtype=np.int64)
    excluded = np.asarray(excluded_rows, dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= ROW_COUNT):
        raise Round0106Error("original row outside the diverse-25M universe")
    if np.any(membership(excluded, rows)):
        raise Round0106Error("excluded original row cannot become a compact ID")
    compact = rows - np.searchsorted(excluded, rows, side="left")
    if np.any(compact < 0) or np.any(compact >= RETAINED_ROWS):
        raise Round0106Error("global-to-compact mapping escaped retained bounds")
    return compact.astype(np.int64, copy=False)


def compact_to_global(
    compact_rows: np.ndarray,
    excluded_rows: np.ndarray,
) -> np.ndarray:
    """Map compact retained IDs to original IDs without a 25M-row lookup."""
    compact = np.asarray(compact_rows, dtype=np.int64)
    excluded = np.asarray(excluded_rows, dtype=np.int64)
    if np.any(compact < 0) or np.any(compact >= RETAINED_ROWS):
        raise Round0106Error("compact row outside the retained universe")
    # Removing excluded original row e shifts every later compact ID left by
    # one.  In compact coordinates its insertion point is therefore
    # e - number_of_prior_exclusions.  Counting insertion points <= c gives
    # the exact number of skipped original IDs, including long contiguous
    # excluded runs without a fixed-point loop.
    insertion_points = excluded - np.arange(len(excluded), dtype=np.int64)
    result = compact + np.searchsorted(
        insertion_points, compact, side="right"
    )
    if (
        np.any(result < 0)
        or np.any(result >= ROW_COUNT)
        or np.any(membership(excluded, result))
        or not np.array_equal(global_to_compact(result, excluded), compact)
    ):
        raise Round0106Error("compact-to-global mapping failed round trip")
    return result


def validate_part_specs() -> None:
    names = list(PARTS)
    if names != ["english", "languages-a", "languages-b"]:
        raise Round0106Error("R0106 part order changed")
    prior_global = 0
    prior_compact = 0
    retained = 0
    excluded = 0
    groups = 0
    for name in names:
        spec = part_spec(name)
        if (
            spec["global_start"] != prior_global
            or spec["compact_start"] != prior_compact
            or spec["global_stop"] - spec["global_start"]
            != spec["retained_rows"] + spec["excluded_rows"]
            or spec["compact_stop"] - spec["compact_start"]
            != spec["retained_rows"]
            or spec["group_start"] != groups
        ):
            raise Round0106Error(f"R0106 part {name} does not close")
        prior_global = spec["global_stop"]
        prior_compact = spec["compact_stop"]
        groups = spec["group_stop"]
        retained += spec["retained_rows"]
        excluded += spec["excluded_rows"]
    if (
        prior_global != ROW_COUNT
        or prior_compact != RETAINED_ROWS
        or retained != RETAINED_ROWS
        or excluded != ROW_COUNT - RETAINED_ROWS
        or groups != len(GROUPS)
    ):
        raise Round0106Error("R0106 parts do not cover the fixed universe")


def _load_sealed(
    path: str,
    *,
    expected_sha256: str,
    schema: str,
    round_id: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0106Error(f"{label} bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if (
        value.get("schema") != schema
        or value.get("round_id") != round_id
        or value.get("identity_sha256") != sha256_bytes(canonical_json(body))
    ):
        raise Round0106Error(f"{label} seal changed")
    return value, signature


def validate_search_artifacts(
    *,
    index_path: str,
    index_sha256: str,
    index_receipt_path: str,
    index_receipt_sha256: str,
    qualification_path: str,
    qualification_sha256: str,
    decision_path: str,
    decision_sha256: str,
    substrate_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the exact positive R0105 search release."""
    index_signature = expected_input_signature(index_path)
    if index_signature["sha256"] != index_sha256:
        raise Round0106Error("R0105 retained index bytes changed")
    index_receipt, index_receipt_signature = _load_sealed(
        index_receipt_path,
        expected_sha256=index_receipt_sha256,
        schema=SEARCH_INDEX_SCHEMA,
        round_id="0105",
        label="R0105 index receipt",
    )
    geometry = index_receipt.get("geometry") or {}
    ids = index_receipt.get("id_validation") or {}
    if (
        index_receipt.get("substrate") != dict(substrate_signature)
        or index_receipt.get("index") != index_signature
        or int(geometry.get("dimension", -1)) != DIMENSION
        or int(geometry.get("ntotal", -1)) != RETAINED_ROWS
        or int(geometry.get("nlist", -1)) != NLIST
        or int(geometry.get("pq_m", -1)) != PQ_M
        or int(geometry.get("pq_bits", -1)) != PQ_BITS
        or ids.get("global_ids_unique") is not True
        or ids.get("excluded_rows_absent") is not True
    ):
        raise Round0106Error("R0105 index receipt contract changed")
    qualification, qualification_signature = _load_sealed(
        qualification_path,
        expected_sha256=qualification_sha256,
        schema=SEARCH_QUALIFICATION_SCHEMA,
        round_id="0105",
        label="R0105 qualification",
    )
    selected = select_cell(qualification.get("cells") or {})
    if (
        qualification.get("substrate") != dict(substrate_signature)
        or qualification.get("index") != index_signature
        or qualification.get("index_receipt") != index_receipt_signature
        or qualification.get("validity_passed") is not True
        or qualification.get("failed_checks") != []
        or not isinstance(selected, dict)
        or qualification.get("selected") != selected
        or selected.get("passes_global_floor") is not True
        or selected.get("passes_every_group_floor") is not True
    ):
        raise Round0106Error("R0105 qualification did not release a policy")
    decision, decision_signature = _load_sealed(
        decision_path,
        expected_sha256=decision_sha256,
        schema=SEARCH_DECISION_SCHEMA,
        round_id="0105",
        label="R0105 decision",
    )
    if (
        decision.get("qualification") != qualification_signature
        or decision.get("selected") != selected
        or decision.get("outcome") != "qualified"
        or decision.get("graph_build_released") is not True
        or decision.get("validity_passed") is not True
    ):
        raise Round0106Error("R0105 decision does not release graph building")
    return {
        "index": index_signature,
        "index_receipt": index_receipt_signature,
        "qualification": qualification_signature,
        "decision": decision_signature,
        "selected": selected,
    }


validate_part_specs()
