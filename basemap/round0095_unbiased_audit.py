"""Contract helpers for the unbiased 150M search-policy replay."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0094_sharded_search import QUALIFICATION_SCHEMA as R0094_SCHEMA


ROUND_ID = "0095"
ROW_COUNT = 150_000_000
RETAINED_ROWS = 147_221_757
SAMPLE_ROWS = 4_096
SAMPLE_SEED = 86
SAMPLE_SHA256 = (
    "fab1613919b657a8116931b0fc336678576ea25ac3ce875b00576f860fa413fe"
)
MEAN_RECALL_FLOOR = 0.84
CORPUS_RANGES = {
    "fineweb": (0, 50_000_000),
    "redpajama": (50_000_000, 100_000_000),
    "pile": (100_000_000, 150_000_000),
}
MONOLITHIC_POLICIES = (
    ("r0093_selected", 256, 1_536),
    ("r0093_highest_recall", 512, 1_536),
)
SHARDED_POLICIES = (
    ("r0094_strongest_registered", 96, 256),
)
AUDIT_SCHEMA = "round0095-balanced-150m-unbiased-search-audit-v1"
DECISION_SCHEMA = "round0095-balanced-150m-search-correction-decision-v1"


class Round0095Error(RuntimeError):
    """The corrected sampling/audit contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {
        **value,
        "identity_sha256": sha256_bytes(canonical_json(value)),
    }


def load_r0094_negative(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0095Error("R0094 qualification bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    if (
        receipt.get("schema") != R0094_SCHEMA
        or receipt.get("round_id") != "0094"
        or receipt.get("validity_passed") is not False
        or receipt.get("selected") is not None
        or receipt.get("failed_checks")
        != ["passing_quality_and_performance_policy_selected"]
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or receipt.get("training_performed") is not False
    ):
        raise Round0095Error("R0094 negative qualification changed")
    return {"receipt": receipt, "signature": signature}


def sample_corpus_counts(rows: Any) -> dict[str, int]:
    return {
        name: int(((rows >= start) & (rows < stop)).sum())
        for name, (start, stop) in CORPUS_RANGES.items()
    }
