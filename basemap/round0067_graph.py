"""Validation of the selected next-rung GPU qualification for Round 0067."""
from __future__ import annotations

import json
from typing import Any, Mapping

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0066_quality import (
    NPROBE_GRID,
    QUALIFICATION_SCHEMA,
    Round0066Error,
)


ROUND_ID = "0067"
GRAPH_RECEIPT_SCHEMA = "round0067-next-rung-gpu-graph-receipt-v1"


class Round0067Error(Round0066Error):
    """The selected next-rung graph contract was violated."""


def load_gpu_qualification(
    path: str,
    *,
    expected_sha256: str,
    tier: str,
    substrate_signature: Mapping[str, Any],
    eligibility_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the exact R0066 policy and filtered-index capability."""
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0067Error("R0066 qualification bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    candidate = receipt.get("candidate_universe") or {}
    quality = receipt.get("quality") or {}
    selected = quality.get("selected") or {}
    checks = receipt.get("checks") or {}
    nprobe = receipt.get("selected_nprobe")
    if (
        receipt.get("schema") != QUALIFICATION_SCHEMA
        or receipt.get("round_id") != "0066"
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or receipt.get("validity_passed") is not True
        or receipt.get("training_performed") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
        or receipt.get("tier") != tier
        or receipt.get("substrate") != substrate_signature
        or receipt.get("eligibility") != eligibility_signature
        or nprobe not in NPROBE_GRID
        or int(selected.get("nprobe", -1)) != nprobe
        or selected.get("passes_mean_floor") is not True
        or float(
            selected.get("mean_recall_at_15_unambiguous", -1.0)
        )
        < 0.90
        or not checks
        or any(value is not True for value in checks.values())
        or not (candidate.get("filtered_index") or {}).get("sha256")
    ):
        raise Round0067Error("R0066 qualification identity changed")
    return {"receipt": receipt, "signature": signature}
