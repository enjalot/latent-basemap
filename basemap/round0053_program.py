"""Matched balanced-30M int8 substrate primitives for Round 0053."""
from __future__ import annotations

import json
from typing import Any, Mapping

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0049_program import (
    DIMENSION,
    INDEX_PATH,
    INDEX_SHA256,
    Round0049Error,
    compact_to_global,
    global_to_compact,
    validate_substrate_manifest,
)


ROUND_ID = "0053"
SOURCE_ROWS = 60_000_000
ROW_COUNT = 30_000_000
K = 15
EXPECTED_EXCLUDED_ROWS = 218_246
EXPECTED_RETAINED_ROWS = 29_781_754
SOURCE_INTERVALS = (
    (0, 10_000_000),
    (20_000_000, 30_000_000),
    (40_000_000, 50_000_000),
)
GLOBAL_150M_INTERVALS = (
    (0, 10_000_000),
    (50_000_000, 60_000_000),
    (100_000_000, 110_000_000),
)
SOURCE_SUBSTRATE_MANIFEST = (
    "/data/latent-basemap/runs/round-0049/queue/artifacts/"
    "balanced-60m-substrate/balanced-60m-substrate-v1.json"
)
SUBSTRATE_SCHEMA = "round0053-balanced-30m-int8-substrate-v1"


class Round0053Error(Round0049Error):
    """The matched balanced-30M control contract was violated."""


def compact30_to_source60(rows: Any):
    return compact_to_global(
        rows,
        intervals=SOURCE_INTERVALS,
        source_rows=SOURCE_ROWS,
    )


def source60_to_compact30(rows: Any):
    return global_to_compact(
        rows,
        intervals=SOURCE_INTERVALS,
        source_rows=SOURCE_ROWS,
    )


def compact30_to_global150(rows: Any):
    return compact_to_global(
        rows,
        intervals=GLOBAL_150M_INTERVALS,
        source_rows=150_000_000,
    )


def global150_to_compact30(rows: Any):
    return global_to_compact(
        rows,
        intervals=GLOBAL_150M_INTERVALS,
        source_rows=150_000_000,
    )


def validate_control_substrate(
    path: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if expected_sha256 is not None and signature["sha256"] != expected_sha256:
        raise Round0053Error("balanced-30M substrate bytes changed")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    body = {
        key: value for key, value in manifest.items()
        if key != "identity_sha256"
    }
    if (
        manifest.get("schema") != SUBSTRATE_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or manifest.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or int(manifest.get("row_count", -1)) != ROW_COUNT
        or int(manifest.get("dimension", -1)) != DIMENSION
        or manifest.get("source_60m_intervals")
        != [list(value) for value in SOURCE_INTERVALS]
        or manifest.get("global_150m_intervals")
        != [list(value) for value in GLOBAL_150M_INTERVALS]
    ):
        raise Round0053Error("balanced-30M substrate identity changed")
    outputs = manifest.get("outputs") or {}
    for key, expected_bytes in (
        ("int8", ROW_COUNT * DIMENSION),
        ("scales", ROW_COUNT * 2),
    ):
        observed = expected_input_signature(
            outputs.get(key, {}).get("canonical_path", "")
        )
        if (
            observed != outputs.get(key)
            or observed["bytes"] != expected_bytes
        ):
            raise Round0053Error(
                f"balanced-30M {key} bytes changed"
            )
    eligibility = expected_input_signature(
        outputs.get("eligibility", {}).get("canonical_path", "")
    )
    if eligibility != outputs.get("eligibility"):
        raise Round0053Error(
            "balanced-30M eligibility bytes changed"
        )
    source = validate_substrate_manifest(
        manifest["source_60m_substrate"]["canonical_path"],
        expected_sha256=manifest["source_60m_substrate"]["sha256"],
    )
    if source["signature"] != manifest["source_60m_substrate"]:
        raise Round0053Error("source 60M substrate changed")
    return {
        "manifest": manifest,
        "signature": signature,
    }
