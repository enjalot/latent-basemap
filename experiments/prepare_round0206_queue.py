#!/usr/bin/env python3
"""Prepare, but never launch, the final fresh-train GrEASE queue."""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Mapping
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0108_evaluation import seal
from basemap.round0175_aumap_baseline import ROWS, SCALES
from basemap.round0206_grease_fresh import (
    NEGATIVE_CAPABILITY,
    POSITIVE_CAPABILITY,
    ROUND_ID,
    validate_reference,
)
from experiments.prepare_round0020_0022_queues import LAB_ROOT, _base_manifest, _dedupe
from experiments.prepare_round0138_queue import _frontmatter, _frontmatter_list
from experiments.prepare_round0175_queue import _source_signatures
from experiments.prepare_round0181_queue import _package_files
from experiments.round0175_nodes import TESTBED_ROOTS
from experiments.round0179_nodes import TOOLCHAIN_PYTHON, TOOLCHAIN_ROOT


ROUND_ROOT = "/data/latent-basemap/runs/round-0206"
QUEUE_ROOT = os.path.join(ROUND_ROOT, "queue")
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE = os.path.join(LAB_ROOT, "round-0206-2026-08-06.md")
CAMPAIGN_FILE = os.path.join(
    LAB_ROOT, "campaign-2026-08-06-width-scaling-and-v0-ship.md"
)
REFERENCE_SCRIPT = os.path.join(
    RELEASE_ROOT, "experiments", "round0206_grease_fresh_reference.py"
)
PRIOR_METHOD_TABLE = (
    "/data/latent-basemap/runs/round-0183/queue/artifacts/"
    "jina-heldout-projection-method-table-v1/table.json"
)
GPU_HOURS_MAXIMUM = 0.75
P90_BY_SCALE = {"200k": 240.0, "500k": 450.0, "2m": 1_500.0}
TIMEOUT_BY_SCALE = {"200k": 480, "500k": 900, "2m": 2_700}


def _issued_round(release_sha: str) -> dict[str, Any]:
    if not os.path.isfile(ROUND_FILE):
        raise RuntimeError("R0206 issued round file is absent")
    frontmatter = _frontmatter(ROUND_FILE)
    if (
        frontmatter.get("round_id") != ROUND_ID
        or frontmatter.get("status") != "issued"
        or frontmatter.get("base_commit") != release_sha
    ):
        raise RuntimeError("R0206 issued round binding changed")
    return expected_input_signature(ROUND_FILE)


def _accepted_inputs(
    round_id: str, *, required_release: str | None = None
) -> list[dict[str, Any]]:
    matches: list[list[dict[str, Any]]] = []
    for review_path in sorted(
        glob.glob(os.path.join(LAB_ROOT, f"review-{round_id}-*.md"))
    ):
        review = _frontmatter(review_path)
        if review.get("round_id") != round_id or review.get("status") != "accepted":
            continue
        releases = _frontmatter_list(review, "releases")
        if required_release is not None and required_release not in releases:
            continue
        round_path = os.path.join(LAB_ROOT, review.get("round") or "")
        result_path = os.path.join(LAB_ROOT, review.get("result") or "")
        round_signature = expected_input_signature(round_path)
        result_signature = expected_input_signature(result_path)
        if (
            round_signature["sha256"] != review.get("round_sha256")
            or result_signature["sha256"] != review.get("result_sha256")
            or _frontmatter(result_path).get("release_commit")
            != review.get("verified_release_commit")
        ):
            raise RuntimeError(f"accepted Review {round_id} binding changed")
        matches.append([
            round_signature,
            result_signature,
            expected_input_signature(review_path),
        ])
    if len(matches) != 1:
        raise RuntimeError(
            f"R0206 requires one accepted Review {round_id}; found {len(matches)}"
        )
    return matches[0]


def _toolchain() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    python = {
        "invocation_path": TOOLCHAIN_PYTHON,
        "resolved_interpreter": expected_input_signature(
            os.path.realpath(TOOLCHAIN_PYTHON)
        ),
        "pyvenv_config": expected_input_signature(
            os.path.join(TOOLCHAIN_ROOT, "pyvenv.cfg")
        ),
    }
    packages = [expected_input_signature(path) for path in _package_files()]
    return python, packages


def _release_cpu_smoke(
    *, release_sha: str, preflight: str, toolchain_python: Mapping[str, Any]
) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0206 release checkout differs from requested release")
    smoke_root = os.path.join(preflight, "fresh-grease-smoke")
    stdout_path = os.path.join(preflight, "fresh-grease-smoke.stdout.log")
    stderr_path = os.path.join(preflight, "fresh-grease-smoke.stderr.log")
    command = [
        str(toolchain_python["invocation_path"]),
        REFERENCE_SCRIPT,
        "--smoke",
        "--scale",
        "smoke",
        "--output",
        smoke_root,
    ]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLCONFIGDIR": os.path.join(preflight, "mplconfig"),
        "NUMBA_CACHE_DIR": os.path.join(preflight, "numba-cache"),
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
        raise RuntimeError("R0206 fresh-train CPU smoke failed")
    with open(execution_path, encoding="utf-8") as handle:
        execution = json.load(handle)
    if not validate_reference(execution, scale="smoke", smoke=True):
        raise RuntimeError("R0206 CPU smoke was not batch stable")
    return seal({
        "schema": "round0206-release-cpu-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cwd": RELEASE_ROOT,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "wall_seconds": time.monotonic() - started,
        "execution": expected_input_signature(execution_path),
        "batch_stability": execution["batch_stability"],
        "train_accounting": execution["train_accounting"],
        "checkpoint_restore_performed": False,
        "dill_or_pickle_object_written": False,
        "stdout": expected_input_signature(stdout_path),
        "stderr": expected_input_signature(stderr_path),
        "path_exercised": (
            "fresh fit -> train-time normalization -> full and fixed-chunk "
            "GrEASE -> full and fixed-chunk final NUMAP -> receipt"
        ),
    })


def prepare_round0206(
    *, release_sha: str, queue_root: str = QUEUE_ROOT
) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", release_sha):
        raise ValueError("R0206 release SHA must be one full commit")
    round_signature = _issued_round(release_sha)
    campaign_signature = expected_input_signature(CAMPAIGN_FILE)
    accepted_inputs = _dedupe([
        *_accepted_inputs(
            "0175", required_release="capability:jina-aumap-oos-baseline-v1"
        ),
        *_accepted_inputs("0179"),
        *_accepted_inputs("0181"),
        *_accepted_inputs(
            "0183",
            required_release="capability:jina-heldout-projection-method-table-v1",
        ),
        *_accepted_inputs(
            "0200",
            required_release="capability:jina-grease-batch-stability-negative-v1",
        ),
    ])
    source_manifest, source_shards = _source_signatures()
    reference_script = expected_input_signature(REFERENCE_SCRIPT)
    toolchain_python, package_files = _toolchain()
    prior_method_table = expected_input_signature(PRIOR_METHOD_TABLE)

    testbeds: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        root = TESTBED_ROOTS[scale]
        embeddings = expected_input_signature(
            os.path.join(root, "train", "data-00000.npy")
        )
        sample_indices = expected_input_signature(
            os.path.join(root, "sample_indices.npy")
        )
        if embeddings["bytes"] != ROWS[scale] * 768 * 4 + 128:
            raise RuntimeError(f"R0206 {scale} embedding matrix size changed")
        testbeds[scale] = {
            "testbed_embeddings": embeddings,
            "sample_indices": sample_indices,
        }

    queue_root = create_fresh_directory(queue_root, label="R0206 fresh GrEASE queue")
    preflight = ensure_data_directory(os.path.join(queue_root, "preflight"))
    smoke_path = os.path.join(preflight, "release-cpu-smoke.json")
    atomic_write_new_json(
        smoke_path,
        _release_cpu_smoke(
            release_sha=release_sha,
            preflight=preflight,
            toolchain_python=toolchain_python,
        ),
        immutable=True,
    )
    artifacts = ensure_data_directory(os.path.join(queue_root, "artifacts"))
    expected_inputs = _dedupe([
        round_signature,
        campaign_signature,
        *accepted_inputs,
        source_manifest,
        *source_shards,
        reference_script,
        toolchain_python["resolved_interpreter"],
        toolchain_python["pyvenv_config"],
        *package_files,
        prior_method_table,
        expected_input_signature(smoke_path),
        *[
            signature
            for testbed in testbeds.values()
            for signature in testbed.values()
        ],
    ])

    jobs: list[dict[str, Any]] = []
    scale_outputs: dict[str, str] = {}
    previous_job: str | None = None
    previous_output: str | None = None
    for scale in SCALES:
        job_id = f"fit_and_score_fresh_grease_{scale}"
        output = os.path.join(artifacts, f"fresh-grease-{scale}")
        scale_outputs[scale] = output
        jobs.append({
            "id": job_id,
            "action": "scale",
            "scale": scale,
            "handler_module": "experiments.round0206_nodes",
            "handler_callable": "run_job",
            "deps": [] if previous_job is None else [previous_job],
            "prior_output": previous_output,
            **testbeds[scale],
            "source_manifest": source_manifest,
            "source_shards": source_shards,
            "reference_script": reference_script,
            "toolchain_python": toolchain_python,
            "package_files": package_files,
            "reference_timeout_s": TIMEOUT_BY_SCALE[scale],
            "outputs": [output],
            "done_marker": os.path.join(artifacts, f"{job_id}.done.json"),
            "expected_inputs": expected_inputs,
            "p90_wall_s": P90_BY_SCALE[scale],
            "node_policy": {"gpu_required": True, "training_performed": True},
        })
        previous_job = job_id
        previous_output = output

    synthesis_output = os.path.join(artifacts, "fresh-grease-synthesis")
    jobs.append({
        "id": "synthesize_fresh_grease_baseline",
        "action": "synthesis",
        "handler_module": "experiments.round0206_nodes",
        "handler_callable": "run_job",
        "deps": [f"fit_and_score_fresh_grease_{scale}" for scale in SCALES],
        "scale_outputs": scale_outputs,
        "prior_method_table": prior_method_table,
        "outputs": [synthesis_output],
        "done_marker": os.path.join(
            artifacts, "synthesize-fresh-grease-baseline.done.json"
        ),
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
        "schema": "round0206-grease-fresh-queue-v1",
        "repo_root": RELEASE_ROOT,
        "queue_class": "gpu-training",
        "required_reviews": ["0175", "0179", "0181", "0183", "0200"],
        "capability_dependencies": [
            "jina-aumap-oos-baseline-v1",
            "jina-heldout-projection-method-table-v1",
        ],
        "capabilities_produced": [POSITIVE_CAPABILITY, NEGATIVE_CAPABILITY],
        "training_performed": True,
        "jobs": jobs,
        "p90_gpu_seconds": {
            **{
                f"fit_and_score_fresh_grease_{scale}": P90_BY_SCALE[scale]
                for scale in SCALES
            },
            "total": sum(P90_BY_SCALE.values()),
        },
        "scientific_contract": {
            "question": (
                "is fresh same-process GrEASE/NUMAP inference batch-stable, and "
                "if so what is its held-out projection curve at 200k/500k/2m?"
            ),
            "scales": list(SCALES),
            "queries_per_scale": 20_000,
            "seed": 42,
            "batch_tolerance": 1.0e-4,
            "fixed_inference_chunk_rows": 256,
            "no_checkpoint_restore": True,
            "no_dill_or_pickle_model": True,
            "first_cell_is_admission_canary": True,
            "stop_after_first_batch_instability": True,
            "positive_branch": POSITIVE_CAPABILITY,
            "negative_branch": NEGATIVE_CAPABILITY,
            "extends_reviewed_round": "0183",
            "quality_role": "diagnostic baseline; no method winner or quality gate",
            "thread_closed_after_this_attempt": True,
            "release_cpu_smoke": expected_input_signature(smoke_path),
            "map_registry_state_changed": False,
            "production_or_publishing": False,
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
    path = prepare_round0206(
        release_sha=args.release_sha,
        queue_root=args.queue_root,
    )
    print(json.dumps({"path": path}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
