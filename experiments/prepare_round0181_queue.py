#!/usr/bin/env python3
"""Prepare, but never launch, the bounded R0181 fixed-normalization queue."""
from __future__ import annotations

import argparse
import glob
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
from basemap.round0108_evaluation import seal
from basemap.round0181_fixed_normalization import (
    CAPABILITY,
    HELD_HASH,
    NORMALIZATION_POLICY,
    N_QUERIES,
    ROUND_ID,
    ROWS,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _accepted_review, _frontmatter
from experiments.prepare_round0175_queue import _source_signatures
from experiments.prepare_round0179_queue import AUMAP_SYNTHESIS
from experiments.round0179_nodes import TESTBED_ROOT, TOOLCHAIN_PYTHON, TOOLCHAIN_ROOT
from experiments.round0181_nodes import REFERENCE_SCRIPT


ROUND_ROOT = "/data/latent-basemap/runs/round-0181"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0181-2026-08-03.md")
R0179_RESULT = os.path.join(LAB_ROOT, "result-0179-2026-08-03.md")
R0179_REVIEW = os.path.join(LAB_ROOT, "review-0179-2026-08-03.md")
R0179_DIAGNOSIS = (
    "/home/enjalot/code/latent-labs/logs/process/"
    "2026-08-03_r0179-numap-batch-conditioned-projection.md"
)
GPU_HOURS_MAXIMUM = 0.75
GPU_P90_SECONDS = 2_400.0


def _issued_round(release_sha: str) -> dict[str, Any]:
    if not os.path.isfile(ROUND_FILE):
        raise RuntimeError("R0181 issued round file is absent")
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0181 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def _r0179_failure_inputs() -> list[dict[str, Any]]:
    result = _frontmatter(R0179_RESULT)
    review = _frontmatter(R0179_REVIEW)
    if (
        result.get("round_id") != "0179"
        or result.get("status") != "failed"
        or review.get("round_id") != "0179"
        or review.get("status") != "accepted"
    ):
        raise RuntimeError("R0179 accepted failure lineage changed")
    return [
        expected_input_signature(R0179_RESULT),
        expected_input_signature(R0179_REVIEW),
        expected_input_signature(R0179_DIAGNOSIS),
    ]


def _package_files() -> list[str]:
    site = os.path.join(TOOLCHAIN_ROOT, "lib", "python3.12", "site-packages")
    paths = sorted(
        glob.glob(os.path.join(site, "numap", "**", "*.py"), recursive=True)
        + glob.glob(os.path.join(site, "grease", "**", "*.py"), recursive=True)
    )
    for distribution in ("numap-0.2.3.dist-info", "grease_embeddings-0.1.5.dist-info"):
        paths.extend(
            os.path.join(site, distribution, name) for name in ("METADATA", "RECORD")
        )
    if len(paths) != 28 or not all(os.path.isfile(path) for path in paths):
        raise RuntimeError(f"R0181 package source closure changed: {len(paths)} files")
    return paths


def _release_cpu_smoke(release_sha: str, preflight: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0181 release checkout differs from requested release")
    smoke_root = os.path.join(preflight, "fixed-normalization-smoke")
    stdout_path = os.path.join(preflight, "fixed-normalization-smoke.stdout.log")
    stderr_path = os.path.join(preflight, "fixed-normalization-smoke.stderr.log")
    command = [
        TOOLCHAIN_PYTHON,
        REFERENCE_SCRIPT,
        "--smoke",
        "--output",
        smoke_root,
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": os.path.join(preflight, "mplconfig"),
    })
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
            timeout=120,
            check=False,
        )
    execution_path = os.path.join(smoke_root, "fit", "execution.json")
    if completed.returncode != 0 or not os.path.isfile(execution_path):
        raise RuntimeError("R0181 fixed-normalization CPU smoke failed")
    with open(execution_path, encoding="utf-8") as handle:
        execution = json.load(handle)
    normalization = execution.get("normalization") or {}
    checkpoint = execution.get("checkpoint") or {}
    if (
        execution.get("schema")
        != "round0181-numap-fixed-normalization-execution-v1"
        or execution.get("mode") != "smoke"
        or execution.get("cuda_available") is not False
        or normalization.get("policy") != NORMALIZATION_POLICY
        or normalization.get("statistics_stored_in_checkpoint") is not True
        or int(normalization.get("batch_composition_probe_rows", 0)) <= 0
        or float(checkpoint.get("reload_full_max_abs_error", 1.0)) > 1.0e-4
        or float(checkpoint.get("reload_batch_max_abs_error", 1.0)) > 1.0e-4
    ):
        raise RuntimeError("R0181 fixed-normalization CPU smoke did not close")
    return seal({
        "schema": "round0181-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "execution": expected_input_signature(execution_path),
        "checkpoint": checkpoint,
        "normalization": normalization,
        "stdout": expected_input_signature(stdout_path),
        "stderr": expected_input_signature(stderr_path),
        "path_exercised": (
            "fit -> stored train statistics -> full transform -> checkpoint -> "
            "reload -> full and smaller-batch transform invariance"
        ),
    })


def prepare_round0181(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0181 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    review_inputs = _accepted_review("0175", "jina-aumap-oos-baseline-v1")
    r0179_inputs = _r0179_failure_inputs()
    source_manifest, source_shards = _source_signatures()
    testbed_embeddings = expected_input_signature(
        os.path.join(TESTBED_ROOT, "train", "data-00000.npy")
    )
    sample_indices = expected_input_signature(
        os.path.join(TESTBED_ROOT, "sample_indices.npy")
    )
    if testbed_embeddings["bytes"] != ROWS * 768 * 4 + 128:
        raise RuntimeError("R0181 200k embedding matrix size changed")
    reference_script = expected_input_signature(REFERENCE_SCRIPT)
    package_files = [expected_input_signature(path) for path in _package_files()]
    toolchain_python = {
        "invocation_path": TOOLCHAIN_PYTHON,
        "resolved_interpreter": expected_input_signature(os.path.realpath(TOOLCHAIN_PYTHON)),
        "pyvenv_config": expected_input_signature(os.path.join(TOOLCHAIN_ROOT, "pyvenv.cfg")),
    }
    aumap_synthesis = expected_input_signature(AUMAP_SYNTHESIS)

    queue_root = create_fresh_directory(queue_root, label="R0181 NUMAP queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path,
        _release_cpu_smoke(release_sha, preflight),
        immutable=True,
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe([
        round_signature,
        *review_inputs,
        *r0179_inputs,
        source_manifest,
        *source_shards,
        testbed_embeddings,
        sample_indices,
        reference_script,
        toolchain_python["resolved_interpreter"],
        toolchain_python["pyvenv_config"],
        *package_files,
        aumap_synthesis,
        expected_input_signature(smoke_path),
    ])

    cell_output = os.path.join(artifacts, "numap-fixed-normalization-200k")
    synthesis_output = os.path.join(artifacts, CAPABILITY)
    jobs: list[dict[str, Any]] = [
        {
            "id": "fit_and_score_numap_fixed_normalization_200k",
            "action": "numap_cell",
            "handler_module": "experiments.round0181_nodes",
            "handler_callable": "run_job",
            "deps": [],
            "testbed_embeddings": testbed_embeddings,
            "sample_indices": sample_indices,
            "source_manifest": source_manifest,
            "source_shards": source_shards,
            "reference_script": reference_script,
            "toolchain_python": toolchain_python,
            "package_files": package_files,
            "outputs": [cell_output],
            "done_marker": os.path.join(artifacts, "fit-and-score-numap.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": GPU_P90_SECONDS,
            "node_policy": {"gpu_required": True, "training_performed": True},
        },
        {
            "id": "synthesize_numap_fixed_normalization_baseline",
            "action": "synthesis",
            "handler_module": "experiments.round0181_nodes",
            "handler_callable": "run_job",
            "deps": ["fit_and_score_numap_fixed_normalization_200k"],
            "cell_output": cell_output,
            "aumap_synthesis": aumap_synthesis,
            "outputs": [synthesis_output],
            "done_marker": os.path.join(artifacts, "synthesize-numap.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": 60.0,
            "node_policy": {"gpu_required": False, "training_performed": False},
        },
    ]
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
        "schema": "round0181-numap-fixed-normalization-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": ["0175", "0179"],
        "capability_dependencies": ["jina-aumap-oos-baseline-v1"],
        "capabilities_produced": [CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            "fit_and_score_numap_fixed_normalization_200k": GPU_P90_SECONDS,
            "total": GPU_P90_SECONDS,
        },
        "scientific_contract": {
            "question": (
                "does storing and reusing GrEASE train-time normalization make "
                "NUMAP a stable OOS baseline on the frozen 200k testbed?"
            ),
            "rows": ROWS,
            "queries": N_QUERIES,
            "held_hash": HELD_HASH,
            "only_treatment_relative_to_r0179": NORMALIZATION_POLICY,
            "frozen": (
                "numap==0.2.3, grease-embeddings==0.1.5, all model/config/seed "
                "values, testbed rows, held IDs, high truth, and metric formulas"
            ),
            "reload_guards": {
                "full_query_max_abs": 1.0e-4,
                "first_256_alone_max_abs": 1.0e-4,
            },
            "comparison": "reviewed R0175 aUMAP 200k; diagnostic only",
            "one_attempt_dependency_kill_rule": True,
            "quality_role": "diagnostic only; no method-winner branch",
            "memory_basis": (
                "R0179 same-shape scientific attempt completed fit/reload in "
                "208.44 GPU-s; fixed statistics add three 768-element arrays"
            ),
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
    path = prepare_round0181(release_sha=args.release_sha, queue_root=args.queue_root)
    print(json.dumps({"path": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
