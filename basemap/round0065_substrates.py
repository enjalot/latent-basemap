"""Decision-ready MiniLM substrates for the next scale-ladder step.

Round 0064 decides whether the balanced MiniLM ladder should bisect at 45M or
advance to 120M.  This module registers both compact row universes so the
mechanical substrate build can be prepared without anticipating that decision.
It does not select a rung or make a geometry claim.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .int8_eligibility import load_int8_eligibility
from .round0049_program import DIMENSION, SOURCE_ROWS, Round0049Error


ROUND_ID = "0065"
SUBSTRATE_SCHEMA = "round0065-balanced-scale-substrate-v1"

SUBSETS: dict[str, dict[str, Any]] = {
    "45m": {
        "row_count": 45_000_000,
        "first_rows_per_corpus": 15_000_000,
        "intervals": (
            (0, 15_000_000),
            (50_000_000, 65_000_000),
            (100_000_000, 115_000_000),
        ),
        "eligibility_summary": {
            "zero_row_count": 0,
            "exact_nonzero_family_count": 273_858,
            "rows_in_exact_nonzero_families": 675_498,
            "duplicate_copy_rows_excluded": 401_640,
            "excluded_row_count": 401_640,
            "retained_row_count": 44_598_360,
            "unique_nonzero_rows": 44_324_502,
        },
    },
    "120m": {
        "row_count": 120_000_000,
        "first_rows_per_corpus": 40_000_000,
        "intervals": (
            (0, 40_000_000),
            (50_000_000, 90_000_000),
            (100_000_000, 140_000_000),
        ),
        "eligibility_summary": {
            "zero_row_count": 232_422,
            "exact_nonzero_family_count": 1_322_383,
            "rows_in_exact_nonzero_families": 3_022_469,
            "duplicate_copy_rows_excluded": 1_700_086,
            "excluded_row_count": 1_932_508,
            "retained_row_count": 118_067_492,
            "unique_nonzero_rows": 116_745_109,
        },
    },
}


class Round0065Error(Round0049Error):
    """The registered decision-ready substrate contract was violated."""


def subset_spec(tier: str) -> dict[str, Any]:
    """Return one immutable-by-convention registered subset specification."""
    try:
        return SUBSETS[tier]
    except KeyError as exc:
        raise Round0065Error(f"unknown scale substrate {tier!r}") from exc


def validate_scale_substrate(
    path: str,
    *,
    tier: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Authenticate a finished R0065 substrate and its materialized outputs."""
    spec = subset_spec(tier)
    signature = expected_input_signature(path)
    if (
        expected_sha256 is not None
        and signature["sha256"] != expected_sha256
    ):
        raise Round0065Error(f"balanced-{tier} substrate bytes changed")
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
        or manifest.get("tier") != tier
        or manifest.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or int(manifest.get("row_count", -1)) != spec["row_count"]
        or int(manifest.get("dimension", -1)) != DIMENSION
        or manifest.get("global_150m_intervals")
        != [list(value) for value in spec["intervals"]]
    ):
        raise Round0065Error(f"balanced-{tier} substrate identity changed")
    outputs = manifest.get("outputs") or {}
    for key, size in (
        ("int8", spec["row_count"] * DIMENSION),
        ("scales", spec["row_count"] * 2),
    ):
        expected = outputs.get(key)
        observed = expected_input_signature(
            (expected or {}).get("canonical_path", "")
        )
        if observed != expected or observed["bytes"] != size:
            raise Round0065Error(f"balanced-{tier} {key} bytes changed")
    eligibility_signature = outputs.get("eligibility")
    eligibility = load_int8_eligibility(
        (eligibility_signature or {}).get("canonical_path", ""),
        expected_sha256=(eligibility_signature or {}).get("sha256"),
        row_count=spec["row_count"],
    )
    summary = eligibility["metadata"]["summary"]
    if any(
        int(summary.get(key, -1)) != value
        for key, value in spec["eligibility_summary"].items()
    ):
        raise Round0065Error(
            f"balanced-{tier} eligibility accounting changed"
        )
    if (
        manifest.get("eligibility_summary") != summary
        or expected_input_signature(
            eligibility_signature["canonical_path"]
        )
        != eligibility_signature
    ):
        raise Round0065Error(
            f"balanced-{tier} eligibility bytes changed"
        )
    if spec["row_count"] > SOURCE_ROWS:
        raise Round0065Error("subset exceeds the registered source universe")
    return {
        "manifest": manifest,
        "signature": signature,
        "eligibility": eligibility,
    }
