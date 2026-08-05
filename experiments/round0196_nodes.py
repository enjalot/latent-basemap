"""Execute R0196's bounded CPU-only GrEASE inference diagnosis."""
from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping
from typing import Any

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0108_evaluation import seal
from basemap.round0196_grease_batch_stable import (
    CAPABILITY,
    ROUND_ID,
    Round0196Error,
    diagnose_execution,
)


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0196Error(f"{label} bytes changed")
    return actual


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if (
        str(job.get("action") or "") != "diagnose_grease_batch_stability"
        or active.get("manifest", {}).get("round_id") != ROUND_ID
    ):
        raise Round0196Error("unknown R0196 action or queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0196Error("R0196 F1 is CPU-only")
    checkpoint = _signature(job["checkpoint"], label="R0181 failed checkpoint")
    queries = _signature(job["queries"], label="R0181 held queries")
    _signature(job["reference_script"], label="R0196 reference adapter")
    _signature(
        job["toolchain_python"]["resolved_interpreter"],
        label="R0196 toolchain interpreter",
    )

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0196 GrEASE diagnosis"
    )
    reference_output = os.path.join(output, "reference")
    stdout_path = os.path.join(output, "reference.stdout.log")
    stderr_path = os.path.join(output, "reference.stderr.log")
    command = [
        job["toolchain_python"]["invocation_path"],
        job["reference_script"]["canonical_path"],
        "--checkpoint",
        checkpoint["canonical_path"],
        "--queries",
        queries["canonical_path"],
        "--output",
        reference_output,
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONHASHSEED": "42",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": os.path.join(output, "mplconfig"),
    }
    started = time.monotonic()
    with open(stdout_path, "x", encoding="utf-8") as stdout_handle, open(
        stderr_path, "x", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=active["manifest"]["repo_root"],
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            timeout=7_200,
            check=False,
        )
    if completed.returncode != 0:
        with open(stderr_path, encoding="utf-8", errors="replace") as handle:
            tail = handle.read()[-8_000:]
        raise Round0196Error(
            f"R0196 reference diagnosis failed with {completed.returncode}: {tail}"
        )
    execution_path = os.path.join(reference_output, "execution.json")
    with open(execution_path, encoding="utf-8") as handle:
        execution = json.load(handle)
    decision = diagnose_execution(execution)
    receipt = seal({
        "schema": "round0196-grease-batch-stability-diagnosis-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY],
        "branch_capabilities_releasable": decision["capabilities_releasable"],
        "source_checkpoint": checkpoint,
        "held_queries": queries,
        "reference_execution": expected_input_signature(execution_path),
        "reference_stdout": expected_input_signature(stdout_path),
        "reference_stderr": expected_input_signature(stderr_path),
        "accepted_r0181_review": dict(job["accepted_r0181_review"]),
        "decision": decision,
        "wall_seconds": time.monotonic() - started,
        "scope": {
            "gpu_used": False,
            "training_performed": False,
            "f2_authorized_only_on_positive_review": True,
            "f3_is_terminal_on_negative_review": True,
            "f4_authorized": False,
            "numap_toy_fit": {
                "inspected": True,
                "fixed": False,
                "source_round": "0175",
                "root_cause": (
                    "spectral-plus-input feature/encoder dimension mismatch"
                ),
                "shares_r0196_batch_geometry_cause": False,
                "numap_stays_killed": True,
            },
        },
    })
    atomic_write_new_json(
        os.path.join(output, "diagnosis.json"), receipt, immutable=True
    )


__all__ = ["run_job"]
