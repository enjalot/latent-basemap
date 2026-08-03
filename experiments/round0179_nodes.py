"""Execute the R0179 unmodified NUMAP 200k OOS baseline."""
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
from basemap.round0175_aumap_baseline import projection_metrics
from basemap.round0179_numap_baseline import (
    CAPABILITY,
    HELD_HASH,
    K_HIT,
    LOW_FRACTION,
    N_QUERIES,
    ROUND_ID,
    ROWS,
    Round0179Error,
    build_synthesis,
    validate_execution,
)
from experiments.round0175_nodes import (
    TESTBED_ROOTS,
    _exact_cosine_neighbors,
    _exact_low_neighbors,
    _heldout_queries,
)


TOOLCHAIN_ROOT = "/data/latent-basemap/toolchains/numap-v0.2.3-py312-r0179"
TOOLCHAIN_PYTHON = os.path.join(TOOLCHAIN_ROOT, "bin", "python")
REFERENCE_SCRIPT = os.path.join(
    os.path.dirname(__file__), "round0179_numap_reference.py"
)
TESTBED_ROOT = TESTBED_ROOTS["200k"]


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0179Error(f"{label} bytes changed")
    return actual


def _read_json(path: str, *, label: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0179Error(f"{label} is not an object")
    return value


def _read_sealed(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    value = _read_json(signature["canonical_path"], label=label)
    validate_seal(value, label=label)
    return value


def _freeze_tree(root: str) -> None:
    for current, directories, files in os.walk(root, topdown=False):
        for name in files:
            os.chmod(os.path.join(current, name), 0o444)
        for name in directories:
            os.chmod(os.path.join(current, name), 0o555)
    os.chmod(root, 0o555)


def run_numap_cell(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0179 NUMAP 200k cell"
    )
    _signature(job["testbed_embeddings"], label="R0179 200k embeddings")
    _signature(job["sample_indices"], label="R0179 200k sample indices")
    for index, expected in enumerate(job["source_shards"]):
        _signature(expected, label=f"R0179 source shard {index}")
    _signature(job["source_manifest"], label="R0179 source manifest")
    _signature(job["reference_script"], label="R0179 reference adapter")
    _signature(job["toolchain_python"]["resolved_interpreter"], label="R0179 interpreter")
    _signature(job["toolchain_python"]["pyvenv_config"], label="R0179 pyvenv config")
    for index, expected in enumerate(job["package_files"]):
        _signature(expected, label=f"R0179 package file {index}")

    started = time.monotonic()
    _source, held, queries = _heldout_queries("200k")
    if len(held) != N_QUERIES or _ids_hash(held) != HELD_HASH:
        raise Round0179Error("R0179 held-out selector changed")
    held_path = os.path.join(output, "held-source-row-ids.npy")
    query_path = os.path.join(output, "held-query-embeddings.npy")
    atomic_save_new_npy(held_path, held, immutable=True)
    atomic_save_new_npy(query_path, queries, immutable=True)

    reference_output = os.path.join(output, "reference")
    os.mkdir(reference_output)
    stdout_path = os.path.join(output, "numap-reference.stdout.log")
    stderr_path = os.path.join(output, "numap-reference.stderr.log")
    reference_started = time.monotonic()
    with open(stdout_path, "x", encoding="utf-8") as stdout_handle, open(
        stderr_path, "x", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            [
                TOOLCHAIN_PYTHON,
                REFERENCE_SCRIPT,
                "--matrix",
                job["testbed_embeddings"]["canonical_path"],
                "--queries",
                query_path,
                "--rows",
                str(ROWS),
                "--output",
                reference_output,
            ],
            cwd=reference_output,
            stdout=stdout_handle,
            stderr=stderr_handle,
            timeout=9_000,
            check=False,
            env={
                **os.environ,
                "PYTHONHASHSEED": "42",
                "MPLCONFIGDIR": os.path.join(reference_output, ".mplconfig"),
            },
        )
    reference_seconds = time.monotonic() - reference_started
    if completed.returncode != 0:
        with open(stderr_path, encoding="utf-8", errors="replace") as handle:
            stderr_tail = handle.read()[-8_000:]
        raise Round0179Error(
            f"unmodified NUMAP reference failed with {completed.returncode}: {stderr_tail}"
        )
    execution_path = os.path.join(reference_output, "execution.json")
    execution = _read_json(execution_path, label="R0179 NUMAP execution")
    validate_execution(execution)
    train_coordinates_path = os.path.join(
        reference_output, execution["paths"]["train_coordinates"]
    )
    query_coordinates_path = os.path.join(
        reference_output, execution["paths"]["query_coordinates"]
    )
    checkpoint_path = os.path.join(
        reference_output, execution["paths"]["checkpoint"]
    )
    train_coordinates = np.load(
        train_coordinates_path, mmap_mode="r", allow_pickle=False
    )
    query_coordinates = np.load(
        query_coordinates_path, mmap_mode="r", allow_pickle=False
    )
    if (
        train_coordinates.shape != (ROWS, 2)
        or query_coordinates.shape != (N_QUERIES, 2)
        or not np.isfinite(train_coordinates).all()
        or not np.isfinite(query_coordinates).all()
    ):
        raise Round0179Error("R0179 NUMAP coordinate files are malformed")

    evaluation_started = time.monotonic()
    corpus = np.load(
        os.path.join(TESTBED_ROOT, "train", "data-00000.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    high_ids, high_distances, high_performance = _exact_cosine_neighbors(
        corpus, queries
    )
    low_k = max(K_HIT, int(np.ceil(LOW_FRACTION * ROWS)))
    low_ids = _exact_low_neighbors(
        np.asarray(train_coordinates), np.asarray(query_coordinates), low_k
    )
    metrics = projection_metrics(high_ids, low_ids)
    evaluation_seconds = time.monotonic() - evaluation_started
    high_ids_path = os.path.join(output, "high-neighbor-ids.npy")
    high_distances_path = os.path.join(output, "high-cosine-distances.npy")
    low_ids_path = os.path.join(output, "low-neighbor-ids.npy")
    atomic_save_new_npy(high_ids_path, high_ids, immutable=True)
    atomic_save_new_npy(high_distances_path, high_distances, immutable=True)
    atomic_save_new_npy(low_ids_path, low_ids, immutable=True)

    artifact_paths = {
        "held_ids": held_path,
        "held_queries": query_path,
        "train_coordinates": train_coordinates_path,
        "query_coordinates": query_coordinates_path,
        "model_checkpoint": checkpoint_path,
        "reference_execution": execution_path,
        "reference_stdout": stdout_path,
        "reference_stderr": stderr_path,
        "high_neighbor_ids": high_ids_path,
        "high_neighbor_distances": high_distances_path,
        "low_neighbor_ids": low_ids_path,
    }
    receipt = seal({
        "schema": "round0179-numap-cell-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "rows": ROWS,
        "dimension": 768,
        "n_queries": N_QUERIES,
        "held_hash": HELD_HASH,
        "held_selection": (
            "the exact accepted R0175 200k sorted RandomState(123) held-source IDs"
        ),
        "execution": execution,
        "heldout_projection": metrics,
        "projection_panel": {
            "k_hit": K_HIT,
            "low_fraction": LOW_FRACTION,
            "low_k": low_k,
            "ffr_formula": "canonical panel_v2 ffr_from_neighbors",
            "recall_formula": "canonical panel_v2 recall_at_k_from_neighbors",
            "high_truth": "GPU exact fp32 cosine IndexFlatIP k15",
            "low_search": "GPU exact fp32 L2 IndexFlatL2 over NUMAP train coordinates",
        },
        "high_search_performance": high_performance,
        "performance": {
            "reference_seconds": reference_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_node_seconds": time.monotonic() - started,
            "peak_parent_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
            "peak_reference_rss_gib": execution["performance"]["peak_rss_gib"],
        },
        "inputs": {
            "testbed_embeddings": job["testbed_embeddings"],
            "sample_indices": job["sample_indices"],
            "source_manifest": job["source_manifest"],
        },
        "artifacts": {
            key: expected_input_signature(path) for key, path in artifact_paths.items()
        },
        "guards_passed": True,
        "quality_role": "diagnostic baseline; no quality floor or method winner",
    })
    atomic_write_new_json(os.path.join(output, "cell.json"), receipt, immutable=True)
    _freeze_tree(output)


def run_synthesis(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0179 NUMAP synthesis"
    )
    cell_signature = expected_input_signature(
        os.path.join(str(job["cell_output"]), "cell.json")
    )
    cell = _read_sealed(cell_signature, label="R0179 NUMAP cell")
    aumap = _read_sealed(job["aumap_synthesis"], label="accepted R0175 synthesis")
    synthesis = build_synthesis(cell=cell, aumap_context=aumap)
    receipt = seal({
        **synthesis,
        "release_sha": active["manifest"]["release_sha"],
        "cell_receipt": cell_signature,
        "aumap_context_receipt": job["aumap_synthesis"],
        "capability": CAPABILITY,
    })
    atomic_write_new_json(
        os.path.join(output, "synthesis.json"), receipt, immutable=True
    )


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0179Error("R0179 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "numap_cell":
        return run_numap_cell(active, job)
    if action == "synthesis":
        return run_synthesis(active, job)
    raise Round0179Error(f"unknown R0179 action: {action!r}")
