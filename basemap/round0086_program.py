"""Content-bound 150M substrate and fixed search qualification contract."""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .int8_eligibility import load_int8_eligibility
from .round0049_program import DIMENSION


ROUND_ID = "0086"
TIER = "150m"
ROW_COUNT = 150_000_000
RETAINED_ROWS = 147_221_757
EXCLUDED_ROWS = 2_778_243
ZERO_ROWS = 235_469
DUPLICATE_COPY_ROWS = 2_542_774
SUBSTRATE_SCHEMA = "round0086-balanced-150m-substrate-v1"
QUALIFICATION_SCHEMA = (
    "round0086-balanced-150m-gpu-ivfpq-policy-qualification-v1"
)
FILTER_RECEIPT_SCHEMA = "round0086-filtered-150m-index-v1"
MEAN_RECALL_FLOOR = 0.90
POLICY_GRID = (
    (128, 256),
    (192, 256),
    (256, 256),
    (128, 384),
    (192, 384),
    (256, 384),
    (128, 512),
    (192, 512),
    (256, 512),
)
SPEC = {
    "row_count": ROW_COUNT,
    "intervals": ((0, ROW_COUNT),),
    "eligibility_summary": {
        "zero_row_count": ZERO_ROWS,
        "duplicate_copy_rows_excluded": DUPLICATE_COPY_ROWS,
        "excluded_row_count": EXCLUDED_ROWS,
        "retained_row_count": RETAINED_ROWS,
    },
}


class Round0086Error(RuntimeError):
    """The registered 150M staging/search contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_substrate(
    path: str,
    *,
    tier: str = TIER,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate the reference-only manifest over reviewed 150M bytes."""
    if tier != TIER:
        raise Round0086Error(f"unknown R0086 tier {tier!r}")
    signature = expected_input_signature(path)
    if (
        expected_sha256 is not None
        and signature["sha256"] != expected_sha256
    ):
        raise Round0086Error("balanced-150m substrate bytes changed")
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
        or manifest.get("row_count") != ROW_COUNT
        or manifest.get("dimension") != DIMENSION
        or manifest.get("global_150m_intervals") != [[0, ROW_COUNT]]
        or manifest.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
    ):
        raise Round0086Error("balanced-150m substrate identity changed")
    outputs = manifest.get("outputs") or {}
    for key, size in (
        ("int8", ROW_COUNT * DIMENSION),
        ("scales", ROW_COUNT * 2),
    ):
        expected = outputs.get(key) or {}
        observed = expected_input_signature(
            str(expected.get("canonical_path", ""))
        )
        if observed != expected or observed["bytes"] != size:
            raise Round0086Error(f"balanced-150m {key} bytes changed")
    eligibility_signature = outputs.get("eligibility") or {}
    eligibility = load_int8_eligibility(
        str(eligibility_signature.get("canonical_path", "")),
        expected_sha256=str(eligibility_signature.get("sha256", "")),
        row_count=ROW_COUNT,
    )
    summary = eligibility["metadata"]["summary"]
    expected_summary = SPEC["eligibility_summary"]
    if any(
        int(summary.get(key, -1)) != value
        for key, value in expected_summary.items()
    ):
        raise Round0086Error("balanced-150m eligibility accounting changed")
    if manifest.get("eligibility_summary") != summary:
        raise Round0086Error("balanced-150m eligibility summary changed")
    return {
        "manifest": manifest,
        "signature": signature,
        "eligibility": eligibility,
    }


def select_cell(receipt: Mapping[str, Any]) -> dict[str, Any] | None:
    """Select the fastest measured passing policy, with fixed tie-breaks."""
    cells = receipt.get("cells") or {}
    passing = [
        cells.get(f"nprobe-{nprobe}-width-{width}")
        for nprobe, width in POLICY_GRID
    ]
    passing = [
        value
        for value in passing
        if isinstance(value, dict)
        and value.get("passes_mean_floor") is True
        and isinstance(value.get("benchmark"), dict)
    ]
    if not passing:
        return None
    return min(
        passing,
        key=lambda value: (
            float(value["benchmark"]["median_wall_seconds_per_query"]),
            int(value["shortlist_width"]),
            int(value["nprobe"]),
        ),
    )
