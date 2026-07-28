"""Quality and performance contract for sharded 150M candidate search."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)


ROUND_ID = "0094"
TIER = "150m"
ROW_COUNT = 150_000_000
RETAINED_ROWS = 147_221_757
MEAN_RECALL_FLOOR = 0.84
MAX_MEDIAN_SECONDS_PER_QUERY = 0.001
SHARD_SPECS = {
    "fineweb": {
        "start": 0,
        "stop": 50_000_000,
        "retained_rows": 48_529_276,
        "excluded_rows": 1_470_724,
    },
    "redpajama": {
        "start": 50_000_000,
        "stop": 100_000_000,
        "retained_rows": 49_567_453,
        "excluded_rows": 432_547,
    },
    "pile": {
        "start": 100_000_000,
        "stop": 150_000_000,
        "retained_rows": 49_125_028,
        "excluded_rows": 874_972,
    },
}
POLICY_GRID = tuple(
    (nprobe, width_per_shard)
    for width_per_shard in (64, 128, 256)
    for nprobe in (32, 40, 64, 96)
)
SPLIT_SCHEMA = "round0094-balanced-150m-corpus-index-shards-v1"
QUALIFICATION_SCHEMA = (
    "round0094-balanced-150m-sharded-search-qualification-v1"
)
DECISION_SCHEMA = "round0094-balanced-150m-sharded-search-decision-v1"


class Round0094Error(RuntimeError):
    """The registered sharded-search contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {
        **value,
        "identity_sha256": sha256_bytes(canonical_json(value)),
    }


def cell_key(nprobe: int, width_per_shard: int) -> str:
    return f"nprobe-{nprobe}-width-per-shard-{width_per_shard}"


def select_cell(receipt: Mapping[str, Any]) -> dict[str, Any] | None:
    """Select the fastest quality- and performance-passing registered cell."""
    cells = receipt.get("cells") or {}
    passing = [
        cells.get(cell_key(nprobe, width))
        for nprobe, width in POLICY_GRID
    ]
    passing = [
        cell
        for cell in passing
        if isinstance(cell, dict)
        and cell.get("passes_mean_floor") is True
        and cell.get("passes_performance_ceiling") is True
        and isinstance(cell.get("benchmark"), dict)
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda cell: (
            float(cell["benchmark"]["median_wall_seconds_per_query"]),
            int(cell["total_shortlist_width"]),
            int(cell["nprobe_per_shard"]),
        ),
    )


def load_split_receipt(
    path: str,
    *,
    expected_source: Mapping[str, Any],
    expected_release_sha: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    shards = receipt.get("shards") or {}
    if (
        receipt.get("schema") != SPLIT_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("release_sha") != expected_release_sha
        or receipt.get("source_index") != dict(expected_source)
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or receipt.get("global_ids_preserved") is not True
        or receipt.get("disjoint_complete_id_ranges") is not True
        or receipt.get("training_performed") is not False
        or set(shards) != set(SHARD_SPECS)
        or sum(
            int((value or {}).get("ntotal", -1))
            for value in shards.values()
        )
        != RETAINED_ROWS
    ):
        raise Round0094Error("sharded-index receipt changed")
    for name, spec in SHARD_SPECS.items():
        shard = shards[name]
        if (
            int(shard.get("start", -1)) != spec["start"]
            or int(shard.get("stop", -1)) != spec["stop"]
            or int(shard.get("ntotal", -1)) != spec["retained_rows"]
            or not isinstance(shard.get("index"), dict)
        ):
            raise Round0094Error(f"{name} index-shard evidence changed")
    return {"receipt": receipt, "signature": signature}


def load_decision(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0094Error("R0094 decision bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    selected = receipt.get("selected") or {}
    benchmark = selected.get("benchmark") or {}
    if (
        receipt.get("schema") != DECISION_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("validity_passed") is not True
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or float(receipt.get("registered_mean_recall_floor", -1.0))
        != MEAN_RECALL_FLOOR
        or selected.get("passes_mean_floor") is not True
        or selected.get("passes_performance_ceiling") is not True
        or float(benchmark.get("median_wall_seconds_per_query", float("inf")))
        > MAX_MEDIAN_SECONDS_PER_QUERY
        or receipt.get("training_performed") is not False
    ):
        raise Round0094Error("R0094 sharded-search decision changed")
    return {"receipt": receipt, "signature": signature}
