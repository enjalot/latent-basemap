"""Registered 120M IVF-PQ search-policy recovery contract."""
from __future__ import annotations

import json
from typing import Any

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)


ROUND_ID = "0081"
QUALIFICATION_SCHEMA = (
    "round0081-balanced-120m-gpu-ivfpq-policy-qualification-v1"
)
MEAN_RECALL_FLOOR = 0.90
POLICY_GRID = (
    (128, 128),
    (192, 128),
    (256, 128),
    (384, 128),
    (512, 128),
    (128, 256),
    (192, 256),
    (256, 256),
    (384, 256),
    (512, 256),
    (128, 512),
    (192, 512),
    (256, 512),
    (384, 512),
)


class Round0081Error(RuntimeError):
    """The registered 120M search-policy contract was violated."""


def cell_key(nprobe: int, shortlist_width: int) -> str:
    return f"nprobe-{int(nprobe)}-width-{int(shortlist_width)}"


def seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _selected_cell(receipt: dict[str, Any]) -> dict[str, Any] | None:
    cells = receipt.get("cells") or {}
    passing = [
        cells.get(cell_key(nprobe, width))
        for nprobe, width in POLICY_GRID
        if (cells.get(cell_key(nprobe, width)) or {}).get(
            "passes_mean_floor"
        )
        is True
    ]
    passing = [
        value
        for value in passing
        if isinstance(value, dict)
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


def load_gpu_policy_qualification(
    path: str,
    *,
    expected_sha256: str,
    substrate_signature: dict[str, Any],
    eligibility_signature: dict[str, Any],
    filtered_index_signature: dict[str, Any],
) -> dict[str, Any]:
    """Authenticate the selected 120M search policy and its inputs."""
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0081Error("R0081 qualification bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    selected = receipt.get("selected") or {}
    independently_selected = _selected_cell(receipt)
    checks = receipt.get("checks") or {}
    if (
        receipt.get("schema") != QUALIFICATION_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
        or receipt.get("validity_passed") is not True
        or receipt.get("training_performed") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
        or receipt.get("scale_decision_made") is not False
        or receipt.get("substrate") != substrate_signature
        or receipt.get("eligibility") != eligibility_signature
        or receipt.get("filtered_index") != filtered_index_signature
        or independently_selected is None
        or selected != independently_selected
        or float(selected.get("mean_recall_at_15_unambiguous", -1.0))
        < MEAN_RECALL_FLOOR
        or selected.get("passes_mean_floor") is not True
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise Round0081Error("R0081 qualification identity changed")
    return {"receipt": receipt, "signature": signature}
