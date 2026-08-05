"""Execute the conditional R0197 three-scale GrEASE baseline."""
from __future__ import annotations

import json
import os
import resource
import subprocess
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.panel_v2 import _ids_hash
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0175_aumap_baseline import (
    HELD_HASHES,
    K_HIT,
    N_QUERIES,
    ROWS,
    SCALES,
    projection_metrics,
)
from basemap.round0197_grease_baseline import (
    CAPABILITY,
    ROUND_ID,
    Round0197Error,
    build_synthesis,
    validate_execution,
)
from experiments.round0175_nodes import (
    TESTBED_ROOTS,
    _exact_cosine_neighbors,
    _exact_low_neighbors,
    _heldout_queries,
)


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0197Error(f"{label} bytes changed")
    return actual


def _read_sealed(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0197Error(f"{label} is not an object")
    validate_seal(value, label=label)
    return value


def _freeze_tree(root: str) -> None:
    for current, directories, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(os.path.join(current, name), 0o444)
        for name in directories:
            os.chmod(os.path.join(current, name), 0o555)
    os.chmod(root, 0o555)


def run_scale(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    scale = str(job.get("scale") or "")
    selected_patch = str(job.get("selected_patch") or "")
    if scale not in SCALES:
        raise Round0197Error(f"unknown R0197 scale {scale!r}")
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0197 {scale} GrEASE cell"
    )
    _signature(job["testbed_embeddings"], label=f"R0197 {scale} embeddings")
    _signature(job["sample_indices"], label=f"R0197 {scale} sample indices")
    _signature(job["reference_script"], label="R0197 reference adapter")
    _signature(job["toolchain_python"]["resolved_interpreter"], label="R0197 interpreter")
    for index, expected in enumerate(job["package_files"]):
        _signature(expected, label=f"R0197 package file {index}")

    started = time.monotonic()
    _source, held, queries = _heldout_queries(scale)
    if len(held) != N_QUERIES or _ids_hash(held) != HELD_HASHES[scale]:
        raise Round0197Error(f"R0197 {scale} held-out selector changed")
    held_path = os.path.join(output, "held-source-row-ids.npy")
    query_path = os.path.join(output, "held-query-embeddings.npy")
    atomic_save_new_npy(held_path, held, immutable=True)
    atomic_save_new_npy(query_path, queries, immutable=True)

    reference_output = os.path.join(output, "reference")
    os.mkdir(reference_output)
    stdout_path = os.path.join(output, "reference.stdout.log")
    stderr_path = os.path.join(output, "reference.stderr.log")
    command = [
        job["toolchain_python"]["invocation_path"],
        job["reference_script"]["canonical_path"],
        "--matrix", job["testbed_embeddings"]["canonical_path"],
        "--queries", query_path,
        "--rows", str(ROWS[scale]),
        "--scale", scale,
        "--selected-patch", selected_patch,
        "--output", reference_output,
    ]
    reference_started = time.monotonic()
    with open(stdout_path, "x", encoding="utf-8") as stdout_handle, open(
        stderr_path, "x", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=active["manifest"]["repo_root"],
            env={
                **os.environ,
                "PYTHONHASHSEED": "42",
                "MPLCONFIGDIR": os.path.join(output, "mplconfig"),
            },
            stdout=stdout_handle,
            stderr=stderr_handle,
            timeout=int(job["active_wall_seconds_maximum"]),
            check=False,
        )
    reference_seconds = time.monotonic() - reference_started
    if completed.returncode != 0:
        with open(stderr_path, encoding="utf-8", errors="replace") as handle:
            tail = handle.read()[-8_000:]
        raise Round0197Error(
            f"R0197 {scale} reference failed with {completed.returncode}: {tail}"
        )
    execution_path = os.path.join(reference_output, "execution.json")
    with open(execution_path, encoding="utf-8") as handle:
        execution = json.load(handle)
    validate_execution(execution, scale=scale, selected_patch=selected_patch)
    base_execution = execution["base_execution"]
    base_root = os.path.join(reference_output, execution["base_output"])
    train_path = os.path.join(base_root, base_execution["paths"]["train_coordinates"])
    projected_path = os.path.join(base_root, base_execution["paths"]["query_coordinates"])
    checkpoint_path = os.path.join(base_root, base_execution["paths"]["checkpoint"])
    train_coordinates = np.load(train_path, mmap_mode="r", allow_pickle=False)
    projected = np.load(projected_path, mmap_mode="r", allow_pickle=False)
    if (
        train_coordinates.shape != (ROWS[scale], 2)
        or projected.shape != (N_QUERIES, 2)
        or not np.isfinite(train_coordinates).all()
        or not np.isfinite(projected).all()
    ):
        raise Round0197Error(f"R0197 {scale} coordinates are malformed")

    evaluation_started = time.monotonic()
    corpus = np.load(
        os.path.join(TESTBED_ROOTS[scale], "train", "data-00000.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    high_ids, high_distances, high_performance = _exact_cosine_neighbors(
        corpus, queries
    )
    low_k = max(K_HIT, int(np.ceil(0.001 * ROWS[scale])))
    low_ids = _exact_low_neighbors(
        np.asarray(train_coordinates), np.asarray(projected), low_k
    )
    metrics = projection_metrics(high_ids, low_ids)
    evaluation_seconds = time.monotonic() - evaluation_started
    high_ids_path = os.path.join(output, "high-neighbor-ids.npy")
    high_distances_path = os.path.join(output, "high-cosine-distances.npy")
    low_ids_path = os.path.join(output, "low-neighbor-ids.npy")
    atomic_save_new_npy(high_ids_path, high_ids, immutable=True)
    atomic_save_new_npy(high_distances_path, high_distances, immutable=True)
    atomic_save_new_npy(low_ids_path, low_ids, immutable=True)
    artifacts = {
        "held_ids": held_path,
        "held_queries": query_path,
        "train_coordinates": train_path,
        "query_coordinates": projected_path,
        "model_checkpoint": checkpoint_path,
        "reference_execution": execution_path,
        "reference_stdout": stdout_path,
        "reference_stderr": stderr_path,
        "high_neighbor_ids": high_ids_path,
        "high_neighbor_distances": high_distances_path,
        "low_neighbor_ids": low_ids_path,
    }
    receipt = seal({
        "schema": "round0197-grease-batch-stable-cell-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "scale": scale,
        "rows": ROWS[scale],
        "dimension": 768,
        "n_queries": N_QUERIES,
        "held_hash": HELD_HASHES[scale],
        "selected_patch": selected_patch,
        "execution": execution,
        "heldout_projection": metrics,
        "projection_panel": {
            "k_hit": K_HIT,
            "low_fraction": 0.001,
            "low_k": low_k,
            "high_truth": "GPU exact fp32 cosine IndexFlatIP k15",
            "low_search": "GPU exact fp32 L2 IndexFlatL2 over fitted train coordinates",
        },
        "high_search_performance": high_performance,
        "performance": {
            "reference_seconds": reference_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_node_seconds": time.monotonic() - started,
            "peak_parent_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        },
        "inputs": {
            "testbed_embeddings": dict(job["testbed_embeddings"]),
            "sample_indices": dict(job["sample_indices"]),
        },
        "artifacts": {
            key: expected_input_signature(path) for key, path in artifacts.items()
        },
        "guards_passed": True,
        "quality_role": "diagnostic baseline; no quality floor or method winner",
    })
    atomic_write_new_json(os.path.join(output, "cell.json"), receipt, immutable=True)
    _freeze_tree(output)


def run_synthesis(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0197 GrEASE synthesis"
    )
    cells: dict[str, dict[str, Any]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    for scale, root in job["scale_outputs"].items():
        signature = expected_input_signature(os.path.join(root, "cell.json"))
        signatures[scale] = signature
        cells[scale] = _read_sealed(signature, label=f"R0197 {scale} cell")
    prior_table = _read_sealed(job["prior_method_table"], label="R0183 method table")
    synthesis = build_synthesis(
        cells=cells,
        prior_table=prior_table,
        selected_patch=str(job["selected_patch"]),
    )
    science_identity = synthesis.pop("identity_sha256")
    receipt = seal({
        **synthesis,
        "science_identity_sha256": science_identity,
        "release_sha": active["manifest"]["release_sha"],
        "cell_receipts": signatures,
        "prior_method_table": dict(job["prior_method_table"]),
        "accepted_r0196_review": dict(job["accepted_r0196_review"]),
    })
    atomic_write_new_json(os.path.join(output, "synthesis.json"), receipt, immutable=True)


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0197Error("R0197 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "scale":
        return run_scale(active, job)
    if action == "synthesis":
        return run_synthesis(active, job)
    raise Round0197Error(f"unknown R0197 action {action!r}")


__all__ = ["run_job"]
