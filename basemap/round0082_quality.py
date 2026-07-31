"""Independent holdout contract for the selected balanced-120M search policy."""
from __future__ import annotations

import json
from typing import Any

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)


ROUND_ID = "0082"
CONFIRMATION_SCHEMA = (
    "round0082-balanced-120m-gpu-ivfpq-policy-confirmation-v1"
)
MEAN_RECALL_FLOOR = 0.90
EXPECTED_NPROBE = 128
EXPECTED_SHORTLIST_WIDTH = 256
SOURCE_QUALIFICATION_SHA256 = (
    "ec062f5ce0fc30c3ee10e1a8f2839a7e26b1a38529fa3ae526da2d1f3796d787"
)
SOURCE_QUALIFICATION_IDENTITY = (
    "fe6a36446742298e182929ed848cf9e3c18b7e22629a055f0e05890e2e3b79b2"
)


class Round0082Error(RuntimeError):
    """The registered independent policy-confirmation contract was violated."""


def seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def load_policy_confirmation(
    path: str,
    *,
    expected_sha256: str,
    source_qualification_signature: dict[str, Any],
    substrate_signature: dict[str, Any],
    eligibility_signature: dict[str, Any],
    filtered_index_signature: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the fresh-sample confirmation for downstream training."""
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0082Error("R0082 confirmation bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    policy = receipt.get("selected_policy") or {}
    quality = receipt.get("quality") or {}
    checks = receipt.get("checks") or {}
    if (
        receipt.get("schema") != CONFIRMATION_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or receipt.get("validity_passed") is not True
        or receipt.get("training_performed") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
        or receipt.get("scale_decision_made") is not False
        or receipt.get("source_qualification")
        != source_qualification_signature
        or receipt.get("source_qualification_identity")
        != SOURCE_QUALIFICATION_IDENTITY
        or receipt.get("substrate") != substrate_signature
        or receipt.get("eligibility") != eligibility_signature
        or receipt.get("filtered_index") != filtered_index_signature
        or int(policy.get("nprobe", -1)) != EXPECTED_NPROBE
        or int(policy.get("shortlist_width", -1))
        != EXPECTED_SHORTLIST_WIDTH
        or float(quality.get("mean_recall_at_15_unambiguous", -1.0))
        < MEAN_RECALL_FLOOR
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise Round0082Error("R0082 confirmation identity changed")
    return {"receipt": receipt, "signature": signature}
