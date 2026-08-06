"""Execute the final fresh-train GrEASE stability and baseline queue."""
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
    FRAC,
    HELD_HASHES,
    K_HIT,
    N_QUERIES,
    ROWS,
    SCALES,
    projection_metrics,
)
from basemap.round0206_grease_fresh import (
    CELL_SCHEMA,
    NEGATIVE_CAPABILITY,
    POSITIVE_CAPABILITY,
    ROUND_ID,
    Round0206Error,
    build_synthesis,
    validate_reference,
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
        raise Round0206Error(f"{label} bytes changed")
    return actual


def _read_sealed(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0206Error(f"{label} is not an object")
    validate_seal(value, label=label)
    return value


def _freeze_tree(root: str) -> None:
    for current, directories, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(os.path.join(current, name), 0o444)
        for name in directories:
            os.chmod(os.path.join(current, name), 0o555)
    os.chmod(root, 0o555)


def _prior_failure(job: Mapping[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    prior = job.get("prior_output")
    if not prior:
        return None, None
    path = os.path.join(str(prior), "cell.json")
    cell = _read_sealed(path, label="prior R0206 cell")
    if cell.get("batch_stability_passed") is False:
        return str(cell.get("scale")), expected_input_signature(path)
    if cell.get("status") == "skipped-prior-batch-instability":
        return str(cell.get("prior_failure_scale")), expected_input_signature(path)
    return None, expected_input_signature(path)


def _write_skip(
    active: Mapping[str, Any], job: Mapping[str, Any], *, scale: str,
    prior_failure: str, prior_signature: Mapping[str, Any], output: str,
) -> None:
    receipt = seal({
        "schema": CELL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "scale": scale,
        "rows": ROWS[scale],
        "held_hash": HELD_HASHES[scale],
        "status": "skipped-prior-batch-instability",
        "prior_failure_scale": prior_failure,
        "prior_cell": dict(prior_signature),
        "batch_stability_passed": None,
        "heldout_projection": None,
        "training_performed": False,
        "gpu_allocation_performed": False,
    })
    atomic_write_new_json(os.path.join(output, "cell.json"), receipt, immutable=True)
    _freeze_tree(output)


def run_scale(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    scale = str(job.get("scale") or "")
    if scale not in SCALES:
        raise Round0206Error(f"unknown R0206 scale {scale!r}")
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0206 {scale} fresh GrEASE cell"
    )
    prior_failure, prior_signature = _prior_failure(job)
    if prior_failure is not None:
        if prior_signature is None:
            raise Round0206Error("R0206 skip lacks prior cell identity")
        return _write_skip(
            active,
            job,
            scale=scale,
            prior_failure=prior_failure,
            prior_signature=prior_signature,
            output=output,
        )

    _signature(job["testbed_embeddings"], label=f"R0206 {scale} embeddings")
    _signature(job["sample_indices"], label=f"R0206 {scale} sample indices")
    _signature(job["source_manifest"], label="R0206 source manifest")
    for index, expected in enumerate(job["source_shards"]):
        _signature(expected, label=f"R0206 source shard {index}")
    _signature(job["reference_script"], label="R0206 reference adapter")
    _signature(
        job["toolchain_python"]["resolved_interpreter"],
        label="R0206 toolchain interpreter",
    )
    for index, expected in enumerate(job["package_files"]):
        _signature(expected, label=f"R0206 package file {index}")

    started = time.monotonic()
    _source, held, queries = _heldout_queries(scale)
    if len(held) != N_QUERIES or _ids_hash(held) != HELD_HASHES[scale]:
        raise Round0206Error(f"R0206 {scale} held-out selector changed")
    held_path = os.path.join(output, "held-source-row-ids.npy")
    query_path = os.path.join(output, "held-query-embeddings.npy")
    atomic_save_new_npy(held_path, held, immutable=True)
    atomic_save_new_npy(query_path, queries, immutable=True)

    reference_output = os.path.join(output, "reference")
    stdout_path = os.path.join(output, "reference.stdout.log")
    stderr_path = os.path.join(output, "reference.stderr.log")
    command = [
        job["toolchain_python"]["invocation_path"],
        job["reference_script"]["canonical_path"],
        "--matrix", job["testbed_embeddings"]["canonical_path"],
        "--queries", query_path,
        "--rows", str(ROWS[scale]),
        "--scale", scale,
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
            timeout=int(job["reference_timeout_s"]),
            check=False,
        )
    reference_seconds = time.monotonic() - reference_started
    if completed.returncode != 0:
        with open(stderr_path, encoding="utf-8", errors="replace") as handle:
            tail = handle.read()[-8_000:]
        raise Round0206Error(
            f"R0206 {scale} fresh reference failed with {completed.returncode}: {tail}"
        )
    execution_path = os.path.join(reference_output, "execution.json")
    with open(execution_path, encoding="utf-8") as handle:
        execution = json.load(handle)
    stable = validate_reference(execution, scale=scale)
    artifacts: dict[str, str] = {
        "held_ids": held_path,
        "held_queries": query_path,
        "reference_execution": execution_path,
        "reference_stdout": stdout_path,
        "reference_stderr": stderr_path,
    }
    for key, relative in (execution.get("paths") or {}).items():
        relative = str(relative)
        path = os.path.abspath(os.path.join(reference_output, relative))
        if (
            os.path.commonpath([reference_output, path]) != reference_output
            or relative != os.path.normpath(relative)
            or os.path.isabs(relative)
            or not os.path.isfile(path)
        ):
            raise Round0206Error(f"R0206 {scale} reference path changed at {key}")
        artifacts[key] = path

    metrics = None
    high_performance = None
    evaluation_seconds = 0.0
    if stable:
        train_path = artifacts["train_coordinates"]
        projected_path = artifacts["query_coordinates"]
        train_coordinates = np.load(train_path, mmap_mode="r", allow_pickle=False)
        projected = np.load(projected_path, mmap_mode="r", allow_pickle=False)
        if (
            train_coordinates.shape != (ROWS[scale], 2)
            or projected.shape != (N_QUERIES, 2)
            or not np.isfinite(train_coordinates).all()
            or not np.isfinite(projected).all()
        ):
            raise Round0206Error(f"R0206 {scale} stable coordinates changed")
        evaluation_started = time.monotonic()
        corpus = np.load(
            os.path.join(TESTBED_ROOTS[scale], "train", "data-00000.npy"),
            mmap_mode="r",
            allow_pickle=False,
        )
        high_ids, high_distances, high_performance = _exact_cosine_neighbors(
            corpus, queries
        )
        low_k = max(K_HIT, int(np.ceil(FRAC * ROWS[scale])))
        low_ids = _exact_low_neighbors(
            np.asarray(train_coordinates), np.asarray(projected), low_k
        )
        metrics = projection_metrics(high_ids, low_ids)
        evaluation_seconds = time.monotonic() - evaluation_started
        for name, value in {
            "high-neighbor-ids.npy": high_ids,
            "high-cosine-distances.npy": high_distances,
            "low-neighbor-ids.npy": low_ids,
        }.items():
            path = os.path.join(output, name)
            atomic_save_new_npy(path, value, immutable=True)
            artifacts[name.removesuffix(".npy").replace("-", "_")] = path

    receipt = seal({
        "schema": CELL_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "scale": scale,
        "rows": ROWS[scale],
        "dimension": 768,
        "n_queries": N_QUERIES,
        "held_hash": HELD_HASHES[scale],
        "status": (
            "stable-baseline-measured" if stable else "batch-instability-measured"
        ),
        "batch_stability_passed": stable,
        "execution": execution,
        "heldout_projection": metrics,
        "projection_panel": (
            {
                "k_hit": K_HIT,
                "low_fraction": FRAC,
                "low_k": max(K_HIT, int(np.ceil(FRAC * ROWS[scale]))),
                "high_truth": "GPU exact fp32 cosine IndexFlatIP k15",
                "low_search": "GPU exact fp32 L2 over fresh GrEASE/NUMAP coordinates",
            }
            if stable else None
        ),
        "high_search_performance": high_performance,
        "performance": {
            "reference_seconds": reference_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_node_seconds": time.monotonic() - started,
            "peak_parent_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
        },
        "inputs": {
            "source_manifest": dict(job["source_manifest"]),
            "source_shards": [dict(item) for item in job["source_shards"]],
            "testbed_embeddings": dict(job["testbed_embeddings"]),
            "sample_indices": dict(job["sample_indices"]),
        },
        "artifacts": {
            key: expected_input_signature(path) for key, path in artifacts.items()
        },
        "training_performed": True,
        "checkpoint_restore_performed": False,
        "quality_role": "diagnostic baseline; no quality floor or method winner",
    })
    atomic_write_new_json(os.path.join(output, "cell.json"), receipt, immutable=True)
    _freeze_tree(output)


def run_synthesis(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0206 fresh GrEASE synthesis"
    )
    cells: dict[str, dict[str, Any]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    for scale, root in job["scale_outputs"].items():
        path = os.path.join(str(root), "cell.json")
        signatures[scale] = expected_input_signature(path)
        cells[scale] = _read_sealed(path, label=f"R0206 {scale} cell")
    prior = _read_sealed(
        str(job["prior_method_table"]["canonical_path"]),
        label="accepted R0183 method table",
    )
    synthesis = build_synthesis(cells=cells, prior_table=prior)
    capability = str(synthesis["capability"])
    if capability not in {POSITIVE_CAPABILITY, NEGATIVE_CAPABILITY}:
        raise Round0206Error("R0206 synthesis capability changed")
    science_identity = synthesis.pop("identity_sha256")
    receipt = seal({
        **synthesis,
        "science_identity_sha256": science_identity,
        "release_sha": active["manifest"]["release_sha"],
        "cell_receipts": signatures,
        "prior_method_table": dict(job["prior_method_table"]),
    })
    atomic_write_new_json(os.path.join(output, "synthesis.json"), receipt, immutable=True)


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0206Error("R0206 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "scale":
        return run_scale(active, job)
    if action == "synthesis":
        return run_synthesis(active, job)
    raise Round0206Error(f"unknown R0206 action {action!r}")


__all__ = ["run_job"]
