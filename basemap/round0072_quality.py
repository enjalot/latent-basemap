"""Fixed balanced-90M GPU candidate-search qualification contract."""
from __future__ import annotations

import json
from typing import Any

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)


ROUND_ID = "0072"
QUALIFICATION_SCHEMA = (
    "round0072-balanced-90m-gpu-ivfpq-qualification-v1"
)
NPROBE_GRID = (16, 24, 32, 40, 48, 64, 96)
MEAN_RECALL_FLOOR = 0.90


class Round0072Error(RuntimeError):
    """The balanced-90M search-qualification contract was violated."""


def seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def load_gpu_qualification(
    path: str,
    *,
    expected_sha256: str,
    substrate_signature: dict[str, Any],
    eligibility_signature: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the exact R0072 policy and filtered-index capability."""
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0072Error("R0072 qualification bytes changed")
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
    rows = receipt.get("rows_by_nprobe") or {}
    checks = receipt.get("checks") or {}
    nprobe = receipt.get("selected_nprobe")
    passing = [
        value
        for value in NPROBE_GRID
        if (rows.get(str(value)) or {}).get("passes_mean_floor") is True
    ]
    if (
        receipt.get("schema") != QUALIFICATION_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or receipt.get("validity_passed") is not True
        or receipt.get("training_performed") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
        or receipt.get("tier") != "90m"
        or receipt.get("substrate") != substrate_signature
        or receipt.get("eligibility") != eligibility_signature
        or nprobe not in NPROBE_GRID
        or not passing
        or nprobe != passing[0]
        or int(selected.get("nprobe", -1)) != nprobe
        or selected.get("passes_mean_floor") is not True
        or float(
            selected.get("mean_recall_at_15_unambiguous", -1.0)
        )
        < MEAN_RECALL_FLOOR
        or not checks
        or any(value is not True for value in checks.values())
        or not (candidate.get("filtered_index") or {}).get("sha256")
    ):
        raise Round0072Error("R0072 qualification identity changed")
    return {"receipt": receipt, "signature": signature}
