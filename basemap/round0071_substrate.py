"""Exact balanced-90M MiniLM substrate contract."""
from __future__ import annotations

import json
from typing import Any

from .artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from .int8_eligibility import load_int8_eligibility
from .round0049_program import DIMENSION, SOURCE_ROWS, Round0049Error


ROUND_ID = "0071"
TIER = "90m"
ROW_COUNT = 90_000_000
ROWS_PER_CORPUS = 30_000_000
INTERVALS = (
    (0, 30_000_000),
    (50_000_000, 80_000_000),
    (100_000_000, 130_000_000),
)
SUBSTRATE_SCHEMA = "round0071-balanced-90m-substrate-v1"
ELIGIBILITY_SUMMARY = {
    "zero_row_count": 3_128,
    "exact_nonzero_family_count": 788_159,
    "rows_in_exact_nonzero_families": 1_839_718,
    "duplicate_copy_rows_excluded": 1_051_559,
    "excluded_row_count": 1_054_687,
    "retained_row_count": 88_945_313,
    "unique_nonzero_rows": 88_157_154,
}


class Round0071Error(Round0049Error):
    """The balanced-90M substrate contract was violated."""


def seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def validate_substrate(
    path: str,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if expected_sha256 and signature["sha256"] != expected_sha256:
        raise Round0071Error("balanced-90M substrate bytes changed")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    body = {
        key: value
        for key, value in manifest.items()
        if key != "identity_sha256"
    }
    if (
        manifest.get("schema") != SUBSTRATE_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or manifest.get("tier") != TIER
        or manifest.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or manifest.get("row_count") != ROW_COUNT
        or manifest.get("dimension") != DIMENSION
        or manifest.get("global_150m_intervals")
        != [list(value) for value in INTERVALS]
        or manifest.get("eligibility_summary") != (
            manifest.get("eligibility_metadata") or {}
        ).get("summary")
    ):
        raise Round0071Error("balanced-90M substrate identity changed")
    outputs = manifest.get("outputs") or {}
    for key, size in (
        ("int8", ROW_COUNT * DIMENSION),
        ("scales", ROW_COUNT * 2),
    ):
        expected = outputs.get(key)
        observed = expected_input_signature(
            (expected or {}).get("canonical_path", "")
        )
        if observed != expected or observed["bytes"] != size:
            raise Round0071Error(f"balanced-90M {key} bytes changed")
    eligibility_signature = outputs.get("eligibility") or {}
    eligibility = load_int8_eligibility(
        eligibility_signature.get("canonical_path", ""),
        expected_sha256=eligibility_signature.get("sha256"),
        row_count=ROW_COUNT,
    )
    summary = eligibility["metadata"]["summary"]
    if any(
        int(summary.get(key, -1)) != value
        for key, value in ELIGIBILITY_SUMMARY.items()
    ):
        raise Round0071Error("balanced-90M eligibility accounting changed")
    if ROW_COUNT > SOURCE_ROWS:
        raise Round0071Error("balanced-90M exceeds source universe")
    return {
        "manifest": manifest,
        "signature": signature,
        "eligibility": eligibility,
    }
