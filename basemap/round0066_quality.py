"""Conditional next-rung selection and receipt validation for Round 0066."""
from __future__ import annotations

import json
from typing import Any

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)


ROUND_ID = "0066"
DECISION_SCHEMA = "round0064-scale-geometry-comparison-v1"
QUALIFICATION_SCHEMA = "round0066-next-rung-gpu-ivfpq-qualification-v1"
NPROBE_GRID = (16, 24, 32, 40, 48, 64, 96)


class Round0066Error(RuntimeError):
    """The conditional next-rung quality contract was violated."""


def load_scale_decision(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Authenticate R0064's preregistered branch and return its exact tier."""
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0066Error("R0064 scale-comparison bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    decision = receipt.get("decision") or {}
    advance = decision.get("advance_to_120m_scale_rung")
    bisect = decision.get("bisect_at_45m_if_false")
    if (
        receipt.get("schema") != DECISION_SCHEMA
        or receipt.get("round_id") != "0064"
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or not isinstance(advance, bool)
        or not isinstance(bisect, bool)
        or advance == bisect
    ):
        raise Round0066Error("R0064 scale decision is invalid")
    return {
        "tier": "120m" if advance else "45m",
        "receipt": receipt,
        "signature": signature,
    }
