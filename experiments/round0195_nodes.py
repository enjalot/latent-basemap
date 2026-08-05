"""Execute R0195's CPU-only v0 release proposal synthesis."""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0108_evaluation import seal
from basemap.round0113_prompt_contrast import read_sealed
from basemap.round0195_release_proposal import (
    CAPABILITY,
    ROUND_ID,
    Round0195Error,
    build_proposal,
)


def _read(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0195Error(f"{label} bytes changed")
    return read_sealed(actual["canonical_path"], label=label)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if (
        str(job.get("action") or "") != "assemble_release_proposal"
        or active.get("manifest", {}).get("round_id") != ROUND_ID
    ):
        raise Round0195Error("unknown R0195 action or queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0195Error("R0195 is CPU-only")
    proposal = build_proposal(
        _read(job["family"], label="accepted R0160 family"),
        _read(job["gates"], label="accepted R0161 gates"),
        _read(job["universality"], label="accepted R0182 OOD packet"),
        _read(job["methods"], label="accepted R0183 method table"),
        _read(job["scale"], label="accepted R0190 scale synthesis"),
    )
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0195 v0 proposal"
    )
    receipt = seal({
        **proposal,
        "release_sha": active["manifest"]["release_sha"],
        "sources": {
            key: dict(job[key])
            for key in ("family", "gates", "universality", "methods", "scale")
        },
        "accepted_reviews": [dict(value) for value in job["accepted_reviews"]],
    })
    atomic_write_new_json(
        os.path.join(output, "release-proposal.json"), receipt, immutable=True
    )


__all__ = ["run_job"]
