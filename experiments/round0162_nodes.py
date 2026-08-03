"""Execute R0162's CPU-only content-addressed prompted-English staging."""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from basemap.output_safety import atomic_write_new_json, create_fresh_directory, ensure_data_directory
from basemap.round0108_evaluation import seal
from basemap.round0162_prompted_english_staging import (
    CAPABILITY,
    DATASETS,
    DIMENSION,
    DTYPE,
    ROUND_ID,
    TOTAL_ROWS,
    VIEW_CAPABILITY,
    VIEW_ROWS,
    Round0162Error,
    first_view,
    layout_identity,
    ordered_chunks,
)


def _read_and_validate_manifest(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = expected_input_signature(str(expected.get("canonical_path") or ""))
    if signature != dict(expected):
        raise Round0162Error(f"{label} bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0162Error(f"{label} is not a JSON object")
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0162Error(f"{label} identity seal changed")
    return value


def _hardlink(source: Mapping[str, Any], destination: str) -> dict[str, Any]:
    observed = expected_input_signature(str(source["canonical_path"]))
    if observed != dict(source):
        raise Round0162Error("source prompted chunk bytes changed")
    if os.path.lexists(destination):
        existing = expected_input_signature(destination)
        if existing["bytes"] != observed["bytes"] or existing["sha256"] != observed["sha256"]:
            raise Round0162Error("content-addressed chunk collision")
    else:
        os.link(observed["canonical_path"], destination)
        os.chmod(destination, 0o444)
    linked = expected_input_signature(destination)
    if linked["bytes"] != observed["bytes"] or linked["sha256"] != observed["sha256"]:
        raise Round0162Error("staged hardlink payload changed")
    return linked


def run_staging(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0162Error("R0162 handler received another queue")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
        raise Round0162Error("R0162 is CPU-only")
    started = time.monotonic()
    r0116 = _read_and_validate_manifest(job["r0116_manifest"], label="accepted R0116 manifest")
    r0120 = _read_and_validate_manifest(job["r0120_manifest"], label="accepted R0120 manifest")
    chunks = ordered_chunks(r0116, r0120)
    expected_layout = layout_identity(
        r0116_signature=job["r0116_manifest"],
        r0120_signature=job["r0120_manifest"],
        chunks=chunks,
    )
    if str(job.get("layout_identity") or "") != expected_layout:
        raise Round0162Error("R0162 layout identity changed after issuance")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0162 canonical layout")
    chunk_root = ensure_data_directory(os.path.join(output, "chunks"))
    manifest_root = ensure_data_directory(os.path.join(output, "source-manifests"))

    preserved_manifests = {}
    for round_id, expected in (("0116", job["r0116_manifest"]), ("0120", job["r0120_manifest"])):
        destination = os.path.join(manifest_root, f"sha256-{expected['sha256']}.json")
        preserved_manifests[round_id] = _hardlink(expected, destination)

    staged_chunks: list[dict[str, Any]] = []
    content_paths: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        sha = str(chunk["source_output"]["sha256"])
        destination = os.path.join(chunk_root, f"sha256-{sha}.f16.npy")
        linked = content_paths.get(sha)
        if linked is None:
            linked = _hardlink(chunk["source_output"], destination)
            content_paths[sha] = linked
        staged_chunks.append({
            **{key: value for key, value in chunk.items() if key != "source_output"},
            "source_output": dict(chunk["source_output"]),
            "staged_output": linked,
        })

    view = first_view(chunks)
    payload = hashlib.sha256()
    finite = True
    maximum_norm_error = 0.0
    for item in view["slices"]:
        chunk = staged_chunks[int(item["chunk_position"])]
        values = np.load(chunk["staged_output"]["canonical_path"], mmap_mode="r", allow_pickle=False)
        start, stop = (int(value) for value in item["source_array_row_slice"])
        if values.dtype != np.dtype(DTYPE) or values.shape != tuple(chunk["output_shape"]):
            raise Round0162Error("staged prompted chunk geometry changed")
        selected = np.asarray(values[start:stop], dtype=np.float16)
        payload.update(memoryview(np.ascontiguousarray(selected)).cast("B"))
        finite = finite and bool(np.isfinite(selected).all())
        for row_start in range(0, len(selected), 4096):
            block = np.asarray(selected[row_start:row_start + 4096], dtype=np.float32)
            maximum_norm_error = max(
                maximum_norm_error,
                float(np.max(np.abs(np.linalg.norm(block, axis=1) - 1.0))),
            )
    if not finite or maximum_norm_error > 0.002:
        raise Round0162Error("first-8M prompted view normalization changed")
    view.update({
        "schema": "jina-document-english-first8m-view-v1",
        "capability": VIEW_CAPABILITY,
        "layout_identity": expected_layout,
        "ordered_embedding_payload_sha256": payload.hexdigest(),
        "payload_hash_semantics": "SHA-256 over concatenated little-endian fp16 row payloads without NPY headers",
        "all_finite": finite,
        "maximum_fp16_norm_absolute_error": maximum_norm_error,
        "staged_slices": [
            {
                **item,
                "staged_output": staged_chunks[int(item["chunk_position"])]["staged_output"],
            }
            for item in view["slices"]
        ],
    })
    view_receipt = seal(view)
    view_path = os.path.join(output, "first-8m-view.json")
    atomic_write_new_json(view_path, view_receipt, immutable=True)

    dataset_ranges = {
        dataset: [
            min(chunk["canonical_row_range"][0] for chunk in staged_chunks if chunk["dataset"] == dataset),
            max(chunk["canonical_row_range"][1] for chunk in staged_chunks if chunk["dataset"] == dataset),
        ]
        for dataset in DATASETS
    }
    receipt = seal({
        "schema": "jina-document-english-9p126m-canonical-layout-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CAPABILITY,
        "layout_identity": expected_layout,
        "rows": TOTAL_ROWS,
        "dimension": DIMENSION,
        "dtype": DTYPE,
        "embedding_convention": "Document: ",
        "source_order": list(DATASETS),
        "dataset_canonical_row_ranges": dataset_ranges,
        "source_manifests": preserved_manifests,
        "chunks": staged_chunks,
        "unique_content_files": len(content_paths),
        "hardlinked_immutable_copy": True,
        "symlinks": False,
        "first8m_view": expected_input_signature(view_path),
        "ordered_selection_sha256": view_receipt["ordered_selection_sha256"],
        "ordered_embedding_payload_sha256": view_receipt["ordered_embedding_payload_sha256"],
        "coverage": {
            "gap_free": True,
            "overlap_free": True,
            "canonical_source_order_preserved": True,
            "rows": TOTAL_ROWS,
            "first8m_rows": VIEW_ROWS,
        },
        "training_performed": False,
        "graph_built": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(os.path.join(output, "canonical-layout.json"), receipt, immutable=True)


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if str(job.get("action") or "") != "stage_prompted_english_corpus":
        raise Round0162Error("unknown R0162 action")
    run_staging(active, job)
