#!/usr/bin/env python3
"""Prepare, but never launch, conditional R0197 after positive Review 0196."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0175_aumap_baseline import ROWS, SCALES
from basemap.round0196_grease_batch_stable import PATCH_CAPABILITY
from basemap.round0197_grease_baseline import (
    CAPABILITY,
    ROUND_ID,
    SELECTED_PATCHES,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter
from experiments.prepare_round0175_queue import _source_signatures
from experiments.prepare_round0181_queue import _package_files
from experiments.round0175_nodes import TESTBED_ROOTS
from experiments.round0179_nodes import TOOLCHAIN_PYTHON, TOOLCHAIN_ROOT


ROUND_ROOT = "/data/latent-basemap/runs/round-0197"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0197-2026-08-05.md")
R0196_RESULT = os.path.join(LAB_ROOT, "result-0196-2026-08-05.md")
R0196_REVIEW = os.path.join(LAB_ROOT, "review-0196-2026-08-05.md")
R0196_DIAGNOSIS = (
    "/data/latent-basemap/runs/round-0196/queue/artifacts/"
    "jina-grease-batch-stability-diagnosis-v1/diagnosis.json"
)
R0183_TABLE = (
    "/data/latent-basemap/runs/round-0183/queue/artifacts/"
    "jina-heldout-projection-method-table-v1/table.json"
)
REFERENCE_SCRIPT = os.path.join(
    os.path.dirname(__file__), "round0197_grease_reference.py"
)
GPU_HOURS_MAXIMUM = 0.5
P90_SECONDS = {"200k": 210.0, "500k": 360.0, "2m": 1_220.0}


def _issued_round(release_sha: str) -> dict[str, Any]:
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0197 is not issued for this exact release")
    return expected_input_signature(ROUND_FILE)


def _positive_r0196() -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    result = _frontmatter(R0196_RESULT)
    review = _frontmatter(R0196_REVIEW)
    result_signature = expected_input_signature(R0196_RESULT)
    if (
        result.get("round_id") != "0196"
        or result.get("status") != "complete"
        or review.get("round_id") != "0196"
        or review.get("status") != "accepted"
        or review.get("result_sha256") != result_signature["sha256"]
        or f"capability:{PATCH_CAPABILITY}" not in (review.get("releases") or [])
    ):
        raise RuntimeError("R0197 requires positive accepted Review 0196")
    diagnosis_signature = expected_input_signature(R0196_DIAGNOSIS)
    with open(R0196_DIAGNOSIS, encoding="utf-8") as handle:
        diagnosis = json.load(handle)
    validate_seal(diagnosis, label="R0196 diagnosis")
    decision = diagnosis.get("decision") or {}
    selected_patch = str(decision.get("selected_patch") or "")
    if (
        diagnosis.get("schema")
        != "round0196-grease-batch-stability-diagnosis-v1"
        or decision.get("passed") is not True
        or decision.get("f2_gpu_baseline_activated") is not True
        or selected_patch not in SELECTED_PATCHES
        or PATCH_CAPABILITY not in (diagnosis.get("branch_capabilities_releasable") or [])
    ):
        raise RuntimeError("R0196 diagnosis did not activate F2")
    return (
        [
            result_signature,
            expected_input_signature(R0196_REVIEW),
            diagnosis_signature,
        ],
        selected_patch,
        diagnosis,
    )


def _release_cpu_smoke(
    *, release_sha: str, selected_patch: str, preflight: str
) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0197 release checkout differs from requested release")
    output = os.path.join(preflight, "grease-reference-smoke")
    stdout_path = os.path.join(preflight, "grease-reference-smoke.stdout.log")
    stderr_path = os.path.join(preflight, "grease-reference-smoke.stderr.log")
    command = [
        TOOLCHAIN_PYTHON,
        REFERENCE_SCRIPT,
        "--smoke",
        "--scale", "smoke",
        "--selected-patch", selected_patch,
        "--output", output,
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": os.path.join(preflight, "mplconfig"),
    }
    started = time.monotonic()
    with open(stdout_path, "x", encoding="utf-8") as stdout_handle, open(
        stderr_path, "x", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=RELEASE_ROOT,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            timeout=180,
            check=False,
        )
    execution_path = os.path.join(output, "fit", "execution.json")
    if completed.returncode != 0 or not os.path.isfile(execution_path):
        raise RuntimeError("R0197 exact train/seal/reload CPU smoke failed")
    with open(execution_path, encoding="utf-8") as handle:
        execution = json.load(handle)
    base = execution.get("base_execution") or {}
    checkpoint = base.get("checkpoint") or {}
    if (
        execution.get("schema")
        != "round0197-grease-batch-stable-reference-execution-v1"
        or execution.get("mode") != "smoke"
        or execution.get("cuda_available") is not False
        or (execution.get("inference_patch") or {}).get("selected_patch")
        != selected_patch
        or float(checkpoint.get("reload_full_max_abs_error", 1.0)) > 1.0e-4
        or float(checkpoint.get("reload_batch_max_abs_error", 1.0)) > 1.0e-4
    ):
        raise RuntimeError("R0197 CPU smoke did not close the exact patched path")
    return seal({
        "schema": "round0197-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "selected_patch": selected_patch,
        "command": command,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "execution": expected_input_signature(execution_path),
        "checkpoint": checkpoint,
        "stdout": expected_input_signature(stdout_path),
        "stderr": expected_input_signature(stderr_path),
        "path_exercised": "fit -> patched transform -> seal -> reload -> full/small guard",
    })


def prepare_round0197(*, release_sha: str, queue_root: str = QUEUE_ROOT) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0197 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    r0196_lineage, selected_patch, _diagnosis = _positive_r0196()
    source_manifest, source_shards = _source_signatures()
    prior_table = expected_input_signature(R0183_TABLE)
    reference_script = expected_input_signature(REFERENCE_SCRIPT)
    package_files = [expected_input_signature(path) for path in _package_files()]
    toolchain_python = {
        "invocation_path": TOOLCHAIN_PYTHON,
        "resolved_interpreter": expected_input_signature(os.path.realpath(TOOLCHAIN_PYTHON)),
        "pyvenv_config": expected_input_signature(os.path.join(TOOLCHAIN_ROOT, "pyvenv.cfg")),
    }
    scale_inputs = {
        scale: {
            "testbed_embeddings": expected_input_signature(
                os.path.join(TESTBED_ROOTS[scale], "train", "data-00000.npy")
            ),
            "sample_indices": expected_input_signature(
                os.path.join(TESTBED_ROOTS[scale], "sample_indices.npy")
            ),
        }
        for scale in SCALES
    }

    queue_root = create_fresh_directory(queue_root, label="R0197 GrEASE queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path,
        _release_cpu_smoke(
            release_sha=release_sha,
            selected_patch=selected_patch,
            preflight=preflight,
        ),
        immutable=True,
    )
    expected_inputs = _dedupe([
        round_signature,
        *r0196_lineage,
        prior_table,
        source_manifest,
        *source_shards,
        reference_script,
        toolchain_python["resolved_interpreter"],
        toolchain_python["pyvenv_config"],
        *package_files,
        expected_input_signature(smoke_path),
        *[item for inputs in scale_inputs.values() for item in inputs.values()],
    ])
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    scale_outputs = {
        scale: os.path.join(artifacts, f"grease-batch-stable-{scale}")
        for scale in SCALES
    }
    jobs: list[dict[str, Any]] = []
    previous: str | None = None
    wall_maxima = {"200k": 900, "500k": 1_200, "2m": 3_600}
    for scale in SCALES:
        job_id = f"fit_and_score_grease_{scale}"
        jobs.append({
            "id": job_id,
            "action": "scale",
            "handler_module": "experiments.round0197_nodes",
            "handler_callable": "run_job",
            "deps": [previous] if previous else [],
            "scale": scale,
            "selected_patch": selected_patch,
            **scale_inputs[scale],
            "source_manifest": source_manifest,
            "source_shards": source_shards,
            "reference_script": reference_script,
            "toolchain_python": toolchain_python,
            "package_files": package_files,
            "active_wall_seconds_maximum": wall_maxima[scale],
            "outputs": [scale_outputs[scale]],
            "done_marker": os.path.join(artifacts, f"fit-and-score-{scale}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": P90_SECONDS[scale],
            "node_policy": {"gpu_required": True, "training_performed": True},
        })
        previous = job_id
    jobs.append({
        "id": "synthesize_grease_baseline",
        "action": "synthesis",
        "handler_module": "experiments.round0197_nodes",
        "handler_callable": "run_job",
        "deps": [previous],
        "selected_patch": selected_patch,
        "scale_outputs": scale_outputs,
        "prior_method_table": prior_table,
        "accepted_r0196_review": r0196_lineage[1],
        "outputs": [os.path.join(artifacts, CAPABILITY)],
        "done_marker": os.path.join(artifacts, "synthesize-grease.done.json"),
        "expected_inputs": expected_inputs,
        "p90_wall_s": 30.0,
        "node_policy": {"gpu_required": False, "training_performed": False},
    })
    queue = _base_manifest(
        round_id=ROUND_ID,
        release_sha=release_sha,
        round_file=ROUND_FILE,
        queue_root=queue_root,
        gpu_hours_cap=GPU_HOURS_MAXIMUM,
        execution_authority="autonomous-gpu",
        gpu=True,
    )
    queue.update({
        "schema": "round0197-grease-batch-stable-oos-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": ["0196"],
        "capability_dependencies": [PATCH_CAPABILITY],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {**P90_SECONDS, "total": sum(P90_SECONDS.values())},
        "scientific_contract": {
            "question": "can the accepted batch-stable GrEASE path fill the 200k/500k/2m parametric baseline cells?",
            "scales": list(SCALES),
            "rows": ROWS,
            "selected_patch": selected_patch,
            "only_treatment_relative_to_r0181": "accepted R0196 fixed-chunk inference patch plus registered scale",
            "gpu_hours_maximum": GPU_HOURS_MAXIMUM,
            "one_attempt": True,
            "extends_reviewed_r0183_table": True,
            "numap_default_toy_fit_repaired": False,
            "numap_revival_authorized": False,
            "additional_retry_or_f4_authorized": False,
            "quality_role": "diagnostic only; no method-winner selector",
            "release_cpu_smoke": expected_input_signature(smoke_path),
            "map_registry_state_changed": False,
        },
    })
    path = os.path.join(queue_root, "queue.json")
    atomic_write_new_json(path, queue, immutable=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=QUEUE_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps({"queue_manifest": prepare_round0197(**vars(args))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
