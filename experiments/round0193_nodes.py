"""Execute R0193's CPU-only mixed-quarter quality-gate registration."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0192_quarter_seed_family import CAPABILITY as FAMILY_CAPABILITY
from basemap.round0193_mixed_gate_registration import (
    CAPABILITY,
    ROUND_ID,
    Round0193Error,
    register_mixed_gates,
)
from basemap import round0113_prompt_contrast as prompt_contract


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if (
        str(job.get("action") or "") != "register_mixed_quality_gates"
        or active.get("manifest", {}).get("round_id") != ROUND_ID
    ):
        raise Round0193Error("unknown R0193 action or queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0193Error("R0193 is CPU-only")
    family_path = str(job["family_evidence"])
    family_signature = expected_input_signature(family_path)
    family_receipt = prompt_contract.read_sealed(
        family_path, label="accepted R0192 quarter seed family"
    )
    if (
        family_receipt.get("schema")
        != "round0192-mixed-quarter-three-seed-family-v1"
        or family_receipt.get("round_id") != "0192"
        or family_receipt.get("capabilities") != [FAMILY_CAPABILITY]
        or (family_receipt.get("scope") or {}).get("gate_registration_performed")
        is not False
    ):
        raise Round0193Error("accepted R0192 family contract changed")
    registration = register_mixed_gates(family_receipt["family"])
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0193 mixed gate registration"
    )
    receipt = prompt_contract.seal({
        **registration,
        "release_sha": active["manifest"]["release_sha"],
        "family_evidence": family_signature,
        "accepted_review": dict(job["accepted_review"]),
        "decision": {
            "outcome": "mixed-english-quarter-gates-registered",
            "applies_to": (
                "future byte-commensurate maps of the R0187 mixed-English "
                "quarter universe and registered recipe"
            ),
            "does_not_apply_to": (
                "FineWeb-only, raw, differently composed, differently prompted, "
                "or differently evaluated maps"
            ),
        },
    })
    atomic_write_new_json(
        os.path.join(output, "mixed-quality-gates.json"), receipt, immutable=True
    )


__all__ = ["run_job"]
