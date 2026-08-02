#!/usr/bin/env python3
"""Prepare, but never launch, the R0156 12.5M scale-native treatment."""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import json
import os
import subprocess
import sys
import time
from typing import Any, Iterator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import expected_input_signature
from basemap.round0156_scale_rescue import (
    CAPABILITY,
    DECISION_SCHEMA,
    DENSITY_FLOOR,
    FUNCTIONAL_SCHEMA,
    GRAPH_DEGREE,
    GRAPH_K,
    GRAPH_PART_SCHEMA,
    GRAPH_SCHEMA,
    GRAPH_SHARD_SCHEMA,
    INDEX_SCHEMA,
    NATIVE_SCHEMA,
    N_NEIGHBORS,
    OOD_RETENTION,
    OOD_SCHEMA,
    PARENT_CAPABILITY,
    PARENT_ROUND_ID,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    PRODUCTION_CONFIG_SCHEMA,
    QUALIFICATION_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID,
    SEED,
    SUBSET_SCHEMA,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    UPDATE_RULE,
)
from experiments import prepare_round0152_queue as base


ROUND_ROOT = "/data/latent-basemap/runs/round-0156"
RELEASE_ROOT = "/home/enjalot/code/latent-basemap-run"
ROUND_FILE_GLOB = os.path.join(base.LAB_ROOT, "round-0156-*.md")
PARENT_OUTPUT = os.path.join(
    "/data/latent-basemap/runs/round-0155/queue/artifacts", PARENT_CAPABILITY
)
PARENT_CENSUS = os.path.join(PARENT_OUTPUT, "census.json")
PARENT_MAPPING = os.path.join(PARENT_OUTPUT, "compact-to-global.i64.npy")
PARENT_GROUP_IDS = os.path.join(PARENT_OUTPUT, "compact-group-ids.u8.npy")
ISSUED_BASE_COMMIT = "a53d266b04bfea6589d7e5a9879b8f713b11a021"
_BASE_ISSUED_ROUND = base._issued_round
_MECHANICAL_CORRECTION_FILES = {
    "basemap/round0108_evaluation.py",
    "basemap/round0107_training.py",
    "experiments/prepare_round0156_queue.py",
    "experiments/round0106_nodes.py",
    "experiments/round0132_nodes.py",
    "tests/test_round0156_scale_rescue.py",
}

GPU_HOURS_MINIMUM = 2.10
GPU_HOURS_EXPECTED = 2.55
GPU_HOURS_P90 = 3.45
GPU_HOURS_MAXIMUM = 5.0
P90_GPU_SECONDS = {
    "build_search_index": 240.0,
    "qualify_fixed_search": 300.0,
    "graph_part": 700.0,
    "train_map": 8_000.0,
    "transform_map": 300.0,
    "score_matched_native": 600.0,
    "score_matched_ood": 430.0,
    "score_functional_density": 450.0,
}

REVIEW_CAPABILITIES = {
    "0087": "jina-diverse-25m-inventory-v1",
    "0103": "jina-diverse-25m-full768-int8-substrate-v1",
    "0105": "jina-diverse-25m-full768-search-qualified-v1",
    "0106": "jina-diverse-25m-full768-fuzzy-graph-v1",
    "0107": "jina-diverse-25m-full768-trained-map-seed42-v1",
    "0108": "jina-diverse-25m-map-registry-v1",
    "0119": "jina-density-failure-localization-v1",
    "0132": "jina-diverse-12p5m-25m-scale-policy-geometry-v1",
    "0140": "jina-2m-subsystem-bisection-v1",
    "0155": PARENT_CAPABILITY,
}


def _issued_round(release_sha: str) -> tuple[str, dict[str, Any]]:
    """Bind the issued contract while admitting only this setup correction."""
    path, signature = _BASE_ISSUED_ROUND(ISSUED_BASE_COMMIT)
    if release_sha == ISSUED_BASE_COMMIT:
        return path, signature
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            RELEASE_ROOT,
            "merge-base",
            "--is-ancestor",
            ISSUED_BASE_COMMIT,
            release_sha,
        ],
        check=False,
        timeout=10,
    )
    changed = subprocess.run(
        [
            "git",
            "-C",
            RELEASE_ROOT,
            "diff",
            "--name-only",
            f"{ISSUED_BASE_COMMIT}..{release_sha}",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.splitlines()
    if ancestor.returncode != 0 or not set(changed) <= _MECHANICAL_CORRECTION_FILES:
        raise RuntimeError("R0156 release exceeds the authorized setup correction")
    return path, signature


def _pytest_receipt(release_sha: str) -> dict[str, Any]:
    observed = subprocess.run(
        ["git", "-C", RELEASE_ROOT, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if observed != release_sha:
        raise RuntimeError("R0156 run checkout is not at the requested release")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "tests/test_round0156_scale_rescue.py",
        "tests/test_round0152_scale_rescue.py",
        "tests/test_round0155_scale_census.py",
        "tests/test_round0151_scale_census.py",
        "tests/test_round0132_scale_bridge.py::test_cpu_train_seal_reload_transform_panel_smoke",
        "tests/test_round0132_scale_bridge.py::test_actual_r0132_train_contract_seals_reloads_and_scores_on_cpu",
        "tests/test_round0107_training.py",
    ]
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"})
    started = time.monotonic()
    completed = subprocess.run(
        command,
        cwd=RELEASE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    receipt = base.seal({
        "schema": "round0156-release-pytest-and-cpu-path-smoke-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "command": command,
        "cuda_visible_devices": "",
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
    })
    if completed.returncode != 0:
        raise RuntimeError(
            f"R0156 release smoke failed:\n{completed.stdout}\n{completed.stderr}"
        )
    return receipt


@contextmanager
def _configured() -> Iterator[None]:
    values = {
        "ROUND_ROOT": ROUND_ROOT,
        "RELEASE_ROOT": RELEASE_ROOT,
        "ROUND_FILE_GLOB": ROUND_FILE_GLOB,
        "_issued_round": _issued_round,
        "HANDLER_MODULE": "experiments.round0156_nodes",
        "QUEUE_SCHEMA": "round0156-historical-prefix-rescue-queue-v1",
        "PARENT_ROUND_ID": PARENT_ROUND_ID,
        "R0151_OUTPUT": PARENT_OUTPUT,
        "R0151_CENSUS": PARENT_CENSUS,
        "R0151_MAPPING": PARENT_MAPPING,
        "R0151_GROUP_IDS": PARENT_GROUP_IDS,
        "R0151_CAPABILITY": PARENT_CAPABILITY,
        "ROUND_ID": ROUND_ID,
        "CAPABILITY": CAPABILITY,
        "RETAINED_ROWS": RETAINED_ROWS,
        "GRAPH_K": GRAPH_K,
        "N_NEIGHBORS": N_NEIGHBORS,
        "SEED": SEED,
        "SUBSET_SCHEMA": SUBSET_SCHEMA,
        "INDEX_SCHEMA": INDEX_SCHEMA,
        "QUALIFICATION_SCHEMA": QUALIFICATION_SCHEMA,
        "GRAPH_SHARD_SCHEMA": GRAPH_SHARD_SCHEMA,
        "GRAPH_PART_SCHEMA": GRAPH_PART_SCHEMA,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "TRAIN_CONFIG_SCHEMA": TRAIN_CONFIG_SCHEMA,
        "PRODUCTION_CONFIG_SCHEMA": PRODUCTION_CONFIG_SCHEMA,
        "TRAIN_RECEIPT_SCHEMA": TRAIN_RECEIPT_SCHEMA,
        "NATIVE_SCHEMA": NATIVE_SCHEMA,
        "OOD_SCHEMA": OOD_SCHEMA,
        "FUNCTIONAL_SCHEMA": FUNCTIONAL_SCHEMA,
        "DECISION_SCHEMA": DECISION_SCHEMA,
        "PIPELINE": PIPELINE,
        "PIPELINE_SCHEMA": PIPELINE_SCHEMA,
        "POSITIVE_DESTINATION_POLICY": POSITIVE_DESTINATION_POLICY,
        "UPDATE_RULE": UPDATE_RULE,
        "GRAPH_DEGREE": GRAPH_DEGREE,
        "DENSITY_FLOOR": DENSITY_FLOOR,
        "OOD_RETENTION": OOD_RETENTION,
        "GPU_HOURS_MINIMUM": GPU_HOURS_MINIMUM,
        "GPU_HOURS_EXPECTED": GPU_HOURS_EXPECTED,
        "GPU_HOURS_P90": GPU_HOURS_P90,
        "GPU_HOURS_MAXIMUM": GPU_HOURS_MAXIMUM,
        "P90_GPU_SECONDS": P90_GPU_SECONDS,
        "REVIEW_CAPABILITIES": REVIEW_CAPABILITIES,
        "SCIENTIFIC_QUESTION": (
            "does the R0153 density-restoring historical-prefix population "
            "transfer under the scale-native 12.5M recipe?"
        ),
        "SCIENTIFIC_ESTIMAND": (
            "historical-prefix population under the same k15 scale graph and "
            "coverage-aligned dose as R0132; not a unique duplicate cause"
        ),
        "POSITIVE_BRANCH": (
            "release a reviewed shippable-v1 candidate for a separate immutable "
            "registry-promotion decision"
        ),
    }
    values["_pytest_receipt"] = _pytest_receipt
    snapshots = {
        name: (hasattr(base, name), getattr(base, name, None)) for name in values
    }
    try:
        for name, value in values.items():
            setattr(base, name, value)
        yield
    finally:
        for name, (present, value) in snapshots.items():
            if present:
                setattr(base, name, value)
            else:
                delattr(base, name)


def prepare_round0156(
    *, release_sha: str, queue_root: str = os.path.join(ROUND_ROOT, "queue")
) -> str:
    with _configured():
        return base.prepare_round0152(release_sha=release_sha, queue_root=queue_root)


_POSTTRAIN_JOB_IDS = (
    "transform_map",
    "score_matched_native",
    "score_matched_ood",
    "score_functional_density",
    "decide_rescue",
)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _prior_posttrain_attempts(
    prior_queue_roots: list[str], *, release_sha: str
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    float,
    list[str],
    str,
    list[str],
    list[str],
]:
    """Authenticate every failed attempt and the final reusable train prefix."""
    if not prior_queue_roots:
        raise RuntimeError("R0156 posttrain continuation requires prior queues")
    inputs: list[dict[str, Any]] = []
    total_gpu_wall_s = 0.0
    attempts: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    prior_releases: list[str] = []
    reused_posttrain: list[str] = []
    seen: set[str] = set()
    for raw_root in prior_queue_roots:
        root = os.path.realpath(raw_root)
        if root in seen:
            raise RuntimeError("R0156 prior queue roots must be unique")
        seen.add(root)
        queue_path = os.path.join(root, "queue.json")
        terminal_path = os.path.join(root, "runner-terminal.json")
        queue_signature = expected_input_signature(queue_path)
        terminal_signature = expected_input_signature(terminal_path)
        queue = _read_json(queue_path)
        terminal = _read_json(terminal_path)
        if (
            queue.get("round_id") != ROUND_ID
            or terminal.get("round_id") != ROUND_ID
            or terminal.get("verdict") != "failed"
            or terminal.get("queue_manifest_sha256") != queue_signature["sha256"]
            or terminal.get("queue_manifest_unchanged") is not True
            or terminal.get("release_checkout_unchanged") is not True
            or terminal.get("gpu_wall_accounting_complete") is not True
            or terminal.get("boundary_problems") != []
        ):
            raise RuntimeError(f"R0156 prior attempt is not reusable: {root}")
        prior_release = str(queue.get("release_sha") or "")
        ancestor = subprocess.run(
            [
                "git", "-C", RELEASE_ROOT, "merge-base", "--is-ancestor",
                prior_release, release_sha,
            ],
            check=False,
            timeout=10,
        )
        if ancestor.returncode != 0:
            raise RuntimeError("R0156 prior release is not an ancestor of correction")
        prior_releases.append(prior_release)
        wall = float(terminal.get("gpu_wall_s") or -1.0)
        if not (0.0 < wall < GPU_HOURS_MAXIMUM * 3600.0):
            raise RuntimeError("R0156 prior GPU accounting is invalid")
        total_gpu_wall_s += wall
        inputs.extend([queue_signature, terminal_signature])
        for name in sorted(os.listdir(os.path.join(root, "artifacts"))):
            if name.endswith(".failed.json") or name.endswith(".done.json"):
                inputs.append(expected_input_signature(os.path.join(root, "artifacts", name)))
        jobs_by_id = {str(job["id"]): job for job in queue.get("jobs", [])}
        for node in terminal.get("completed_jobs") or []:
            if node not in _POSTTRAIN_JOB_IDS or node not in jobs_by_id:
                continue
            reused_posttrain.append(node)
            for output in jobs_by_id[node].get("outputs") or []:
                if os.path.isdir(output):
                    for directory, _, files in os.walk(output):
                        for name in sorted(files):
                            inputs.append(expected_input_signature(os.path.join(directory, name)))
                elif os.path.isfile(output):
                    inputs.append(expected_input_signature(output))
        attempts.append((root, queue, terminal))

    latest_root, template_queue, latest_terminal = attempts[-1]
    completed_posttrain = [
        node
        for node in (latest_terminal.get("completed_jobs") or [])
        if node in _POSTTRAIN_JOB_IDS
    ]
    if completed_posttrain != list(_POSTTRAIN_JOB_IDS[: len(completed_posttrain)]):
        raise RuntimeError("R0156 posttrain completion is not a reusable prefix")
    remaining_jobs = list(_POSTTRAIN_JOB_IDS[len(completed_posttrain) :])
    if not remaining_jobs or not str(latest_terminal.get("stop_reason") or "").startswith(
        f"node {remaining_jobs[0]} exited 1"
    ):
        raise RuntimeError("R0156 latest attempt does not stop at the next suffix node")

    required_prefix = [
        "materialize_prefix_drop_subset",
        "build_search_index",
        "qualify_fixed_search",
        "build_graph_part_groups-a",
        "build_graph_part_groups-b",
        "build_graph_part_groups-c",
        "assemble_graph",
        "train_map",
    ]
    train_attempts = [
        (root, queue, terminal)
        for root, queue, terminal in attempts
        if terminal.get("completed_jobs") == required_prefix
        and "transform_map" in {
            str(job["id"]): job for job in queue.get("jobs", [])
        }
    ]
    if not train_attempts:
        raise RuntimeError("R0156 has no authenticated sealed train prefix")
    _train_root, train_queue, _train_terminal = train_attempts[-1]
    train_jobs = {str(job["id"]): job for job in train_queue.get("jobs", [])}
    template_jobs = {
        str(job["id"]): job for job in template_queue.get("jobs", [])
    }
    if not all(node in template_jobs for node in remaining_jobs):
        raise RuntimeError("R0156 latest queue lacks its remaining suffix")
    train_output = str(train_jobs["transform_map"]["train_output"])
    graph_manifest = str(train_jobs["transform_map"]["graph_manifest"])
    train_receipt_path = os.path.join(train_output, "train-receipt.json")
    train_receipt = _read_json(train_receipt_path)
    base.validate_seal(train_receipt, label="R0156 prior train receipt")
    accounting = train_receipt.get("train_accounting") or {}
    if (
        train_receipt.get("round_id") != ROUND_ID
        or train_receipt.get("release_sha") != train_queue.get("release_sha")
        or accounting.get("optimizer_steps_succeeded") != 722_186
        or accounting.get("positive_lr_optimizer_steps") != 722_186
        or accounting.get("amp_overflow_skips") != 0
        or accounting.get("nonfinite_loss_skips") != 0
        or accounting.get("nonfinite_gradient_skips") != 0
    ):
        raise RuntimeError("R0156 prior train accounting cannot be adopted")
    inputs.extend([
        expected_input_signature(train_receipt_path),
        expected_input_signature(os.path.join(train_output, "model.pt")),
        expected_input_signature(os.path.join(train_output, "production-config.json")),
        expected_input_signature(graph_manifest),
    ])
    graph = _read_json(graph_manifest)
    graph_outputs = graph.get("outputs")
    if not isinstance(graph_outputs, dict):
        raise RuntimeError("R0156 prior graph output index is incomplete")
    for signature in [graph.get("compact_mapping"), *graph_outputs.values()]:
        if not isinstance(signature, dict):
            raise RuntimeError("R0156 prior graph seal is incomplete")
        observed = expected_input_signature(str(signature["canonical_path"]))
        if observed != signature:
            raise RuntimeError("R0156 prior graph payload changed")
        inputs.append(observed)
    subset_output = str(train_jobs["score_matched_native"]["subset_output"])
    for name in sorted(os.listdir(subset_output)):
        path = os.path.join(subset_output, name)
        if os.path.isfile(path):
            inputs.append(expected_input_signature(path))
    if total_gpu_wall_s >= GPU_HOURS_MAXIMUM * 3600.0:
        raise RuntimeError("R0156 failed attempts exhausted the registered GPU cap")
    reused_jobs = [*required_prefix, *dict.fromkeys(reused_posttrain)]
    return (
        template_queue,
        base._dedupe(inputs),
        total_gpu_wall_s,
        remaining_jobs,
        str(train_receipt["release_sha"]),
        prior_releases,
        reused_jobs,
    )


def prepare_round0156_posttrain_continuation(
    *,
    release_sha: str,
    queue_root: str,
    prior_queue_roots: list[str],
) -> str:
    """Prepare only the immutable train's transform/panel/decision suffix."""
    with _configured():
        round_path, _round_signature = _issued_round(release_sha)
        (
            source_queue,
            prior_inputs,
            prior_gpu_wall_s,
            remaining_jobs,
            train_release_sha,
            prior_releases,
            reused_jobs,
        ) = _prior_posttrain_attempts(prior_queue_roots, release_sha=release_sha)
        queue_root = base.create_fresh_directory(
            queue_root, label="R0156 posttrain continuation queue"
        )
        artifacts = base.ensure_data_directory(os.path.join(queue_root, "artifacts"))
        preflight = base.ensure_data_directory(os.path.join(queue_root, "preflight"))
        smoke_path = os.path.join(preflight, "release-pytest-and-cpu-path-smoke.json")
        base.atomic_write_new_json(
            smoke_path, _pytest_receipt(release_sha), immutable=True
        )
        smoke = expected_input_signature(smoke_path)
        continuation_inputs = base._dedupe([*prior_inputs, smoke])
        source_jobs = {
            str(job["id"]): job for job in source_queue.get("jobs", [])
        }
        new_outputs = {
            node: os.path.join(
                artifacts, os.path.basename(str(source_jobs[node]["outputs"][0]))
            )
            for node in remaining_jobs
        }
        jobs: list[dict[str, Any]] = []
        for node in remaining_jobs:
            job = copy.deepcopy(source_jobs[node])
            job["outputs"] = [new_outputs[node]]
            job["done_marker"] = os.path.join(artifacts, f"{node}.done.json")
            job["train_release_sha"] = train_release_sha
            job["expected_inputs"] = base._dedupe([
                *job.get("expected_inputs", []), *continuation_inputs
            ])
            job["deps"] = [
                dep for dep in job.get("deps", []) if dep in remaining_jobs
            ]
            if node == "score_matched_native" and "transform_map" in new_outputs:
                job["transform_output"] = new_outputs["transform_map"]
            if node == "decide_rescue":
                if "score_matched_native" in new_outputs:
                    job["native_output"] = new_outputs["score_matched_native"]
                if "score_matched_ood" in new_outputs:
                    job["ood_output"] = new_outputs["score_matched_ood"]
                if "score_functional_density" in new_outputs:
                    job["functional_output"] = new_outputs["score_functional_density"]
            jobs.append(job)

        remaining_cap = GPU_HOURS_MAXIMUM - prior_gpu_wall_s / 3600.0
        manifest = base._base_manifest(
            round_id=ROUND_ID,
            release_sha=release_sha,
            round_file=round_path,
            queue_root=queue_root,
            gpu_hours_cap=remaining_cap,
            execution_authority="autonomous-gpu",
            gpu=True,
        )
        scientific_contract = copy.deepcopy(source_queue["scientific_contract"])
        scientific_contract["release_smoke"] = smoke
        scientific_contract["setup_retry"] = {
            "prior_queue_roots": [os.path.realpath(path) for path in prior_queue_roots],
            "prior_releases": prior_releases,
            "prior_gpu_wall_s": prior_gpu_wall_s,
            "successful_prefix_reused_without_reexecution": reused_jobs,
            "continuation_jobs": remaining_jobs,
            "remaining_gpu_hours_cap": remaining_cap,
            "science_contract_changed": False,
            "training_performed_in_continuation": False,
        }
        p90_by_node = {
            node: P90_GPU_SECONDS[node]
            for node in remaining_jobs
            if node in P90_GPU_SECONDS
        }
        remaining_p90_s = sum(p90_by_node.values())
        manifest.update({
            "schema": "round0156-historical-prefix-posttrain-continuation-queue-v1",
            "repo_root": RELEASE_ROOT,
            "queue_class": "gpu-research",
            "required_reviews": list(REVIEW_CAPABILITIES),
            "capability_dependencies": list(REVIEW_CAPABILITIES.values()),
            "capabilities_produced": [CAPABILITY],
            "training_performed": False,
            "scientific_contract": scientific_contract,
            "gpu_hours": {
                "minimum": 0.0 if remaining_p90_s == 0 else 0.01,
                "expected": min(0.35, remaining_p90_s / 7200.0),
                "p90": remaining_p90_s / 3600.0,
                "maximum": remaining_cap,
                "prior_attempt_gpu_wall_s": prior_gpu_wall_s,
            },
            "p90_gpu_seconds": {
                **p90_by_node,
                "total": remaining_p90_s,
                "prior_attempt_gpu_wall_s": prior_gpu_wall_s,
            },
            "jobs": jobs,
        })
        path = os.path.join(queue_root, "queue.json")
        base.atomic_write_new_json(path, manifest, immutable=True)
        return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--queue-root", default=os.path.join(ROUND_ROOT, "queue"))
    parser.add_argument("--prior-queue-root", action="append", default=[])
    args = parser.parse_args(argv)
    queue_manifest = (
        prepare_round0156_posttrain_continuation(
            release_sha=args.release_sha,
            queue_root=args.queue_root,
            prior_queue_roots=args.prior_queue_root,
        )
        if args.prior_queue_root
        else prepare_round0156(
            release_sha=args.release_sha, queue_root=args.queue_root
        )
    )
    print(json.dumps({"queue_manifest": queue_manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
