"""Execute R0161's CPU-only prompted quality-gate registration."""
from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0160_prompted_seed_family import CAPABILITY as FAMILY_CAPABILITY
from basemap.round0161_prompted_gate_registration import (
    CAPABILITY,
    ROUND_ID,
    Round0161Error,
    register_prompted_gates,
)


def _read_sealed(path: str, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0161Error(f"{label} is not a JSON object")
    validate_seal(value, label=label)
    return value, signature


def run_registration(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0161Error("R0161 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0161Error("R0161 is CPU-only")
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0161 prompted gate registration"
    )
    family, family_signature = _read_sealed(
        str(job["family_evidence"]), label="accepted R0160 prompted seed family"
    )
    if (
        family.get("round_id") != "0160"
        or family.get("capability") != FAMILY_CAPABILITY
        or family.get("gate_registered") is not False
        or family.get("raw_floor_changed") is not False
    ):
        raise Round0161Error("R0160 prompted family contract changed")
    registration = register_prompted_gates(family["cells"])
    receipt = seal({
        **registration,
        "release_sha": active["manifest"]["release_sha"],
        "family_evidence": family_signature,
        "accepted_review": dict(job["accepted_review"]),
        "decision": {
            "outcome": "new-prompted-universe-gates-registered",
            "applies_to": "future commensurate Document:-prompted maps only",
            "does_not_apply_to": "retired raw row-policy or raw-universe maps",
        },
    })
    atomic_write_new_json(
        os.path.join(output, "prompted-quality-gates.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "register_prompted_quality_gates":
        raise Round0161Error("unknown R0161 action")
    run_registration(active, job)
