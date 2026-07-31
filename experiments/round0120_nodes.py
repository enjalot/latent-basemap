"""GPU embedding nodes and CPU coverage finalizer for R0120."""
from __future__ import annotations

import json
import math
import os
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    ordered_array_sha256,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0112_prompt_substrate import (
    OUTPUT_DTYPE,
    ordered_text_sha256,
)
from basemap.round0120_prompted_pile import (
    BATCH_SIZE,
    CAPABILITY,
    CHUNK_ROWS,
    COMPUTE_DTYPE,
    CORPUS_ROWS,
    CORPUS_SCHEMA,
    DATASET,
    DIMENSION,
    EMBED_MINIMUM_ROWS_PER_S,
    EMBED_WARNING_ROWS_PER_S,
    NODE_SCHEMA,
    PERFORMANCE_GUARD_ROWS,
    PROMPT_PREFIX,
    R0087_PILE_GLOBAL_OFFSET,
    R0087_PILE_GLOBAL_STOP,
    ROUND_ID,
    WORK_RANGES,
    Round0120Error,
    clip_layout,
    expected_work_range,
    load_inventory_manifest,
    load_r0114_model_prompt_closure,
    model_contract,
    seal,
    validate_coverage,
    validate_environment_freeze,
    validate_node_receipt,
)
from experiments.round0116_nodes import (
    _SequentialTextReader,
    _encode_document,
    _float32_norm_guard,
    _load_document_model,
    _prompt_equivalence,
    _stored_array_guard,
    _verify_source_files,
)


def _encode_pile_document(
    model: Any,
    texts: list[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use R0120's registered safe batch, not R0116's module default."""
    return _encode_document(
        model,
        texts,
        requested_batch_size=BATCH_SIZE,
    )


def _verify_bound_closures(job: Mapping[str, Any]) -> dict[str, Any]:
    _, inventory_signature = load_inventory_manifest(
        str(job.get("inventory_manifest_path") or "")
    )
    if inventory_signature != job.get("inventory_manifest_signature"):
        raise Round0120Error("R0087 inventory binding changed at node boundary")
    _, r0114_signature = load_r0114_model_prompt_closure(
        str(job.get("r0114_manifest_path") or "")
    )
    if r0114_signature != job.get("r0114_manifest_signature"):
        raise Round0120Error("R0114 model/prompt binding changed at boundary")
    return {
        "inventory_manifest": inventory_signature,
        "r0114_model_prompt_manifest": r0114_signature,
    }


def _validate_node_layout(
    layout: list[dict[str, Any]], *, start: int, stop: int
) -> None:
    if (
        not layout
        or int(layout[0].get("dataset_row_start", -1)) != start
        or int(layout[-1].get("dataset_row_stop", -1)) != stop
        or any(item.get("dataset") != DATASET for item in layout)
    ):
        raise Round0120Error("R0120 node source layout changed")
    cursor = start
    for item in layout:
        item_start = int(item.get("dataset_row_start", -1))
        item_stop = int(item.get("dataset_row_stop", -1))
        accepted = item.get("accepted_raw_embedding") or {}
        if (
            item_start != cursor
            or item_stop <= item_start
            or int(item.get("corpus_global_row_start", -1)) != item_start
            or int(item.get("corpus_global_row_stop", -1)) != item_stop
            or int(item.get("r0087_global_row_start", -1))
            != R0087_PILE_GLOBAL_OFFSET + item_start
            or int(item.get("r0087_global_row_stop", -1))
            != R0087_PILE_GLOBAL_OFFSET + item_stop
            or len(str(accepted.get("sha256") or "")) != 64
        ):
            raise Round0120Error("R0120 source-row identity changed")
        cursor = item_stop
    if cursor != stop:
        raise Round0120Error("R0120 node source layout did not close")


def run_embed_document_rows(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    node_id = str(job.get("id") or "")
    expected_start, expected_stop = expected_work_range(node_id)
    start = int(job.get("dataset_row_start", -1))
    stop = int(job.get("dataset_row_stop", -1))
    if (
        job.get("dataset") != DATASET
        or (start, stop) != (expected_start, expected_stop)
        or len(job.get("outputs") or []) != 1
        or job.get("corpus_global_row_start") != start
        or job.get("corpus_global_row_stop") != stop
        or job.get("r0087_global_row_start")
        != R0087_PILE_GLOBAL_OFFSET + start
        or job.get("r0087_global_row_stop")
        != R0087_PILE_GLOBAL_OFFSET + stop
    ):
        raise Round0120Error(f"R0120 node {node_id} is malformed")

    boundary_closures = _verify_bound_closures(job)
    source_layout = [
        dict(item) for item in (job.get("authenticated_source_layout") or [])
    ]
    _validate_node_layout(source_layout, start=start, stop=stop)
    verified_source_files = _verify_source_files(source_layout)
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"R0120 {node_id} embedding output",
    )
    chunks_root = create_fresh_directory(
        os.path.join(output, "chunks"),
        label=f"R0120 {node_id} atomic chunks",
    )
    started = time.monotonic()
    reader = _SequentialTextReader(source_layout)
    model, runtime_semantics, model_members = _load_document_model()
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    chunks: list[dict[str, Any]] = []
    prompt_equivalence: dict[str, Any] | None = None
    encode_wall_total = 0.0
    oom_retries = 0

    for index, chunk_start in enumerate(range(start, stop, CHUNK_ROWS)):
        chunk_stop = min(chunk_start + CHUNK_ROWS, stop)
        chunk_started = time.monotonic()
        raw_texts = reader.read(chunk_start, chunk_stop)
        if prompt_equivalence is None:
            prompt_equivalence = _prompt_equivalence(model, raw_texts)
        document_texts = [PROMPT_PREFIX + text for text in raw_texts]
        source_hash = ordered_text_sha256(raw_texts)
        document_hash = ordered_text_sha256(document_texts)
        encode_started = time.monotonic()
        values, telemetry = _encode_pile_document(model, document_texts)
        encode_wall = time.monotonic() - encode_started
        encode_wall_total += encode_wall
        oom_retries += int(telemetry["oom_retries"])
        float32_norm = _float32_norm_guard(
            values, label=f"{node_id} chunk {index}"
        )
        path = os.path.join(
            chunks_root,
            f"document-{chunk_start:07d}-{chunk_stop:07d}.f16.npy",
        )
        atomic_save_new_npy(
            path, values.astype(OUTPUT_DTYPE), immutable=True
        )
        stored_norm = _stored_array_guard(
            path, expected_rows=chunk_stop - chunk_start
        )
        output_signature = expected_input_signature(path)
        chunks.append(
            {
                "chunk_index": index,
                "dataset": DATASET,
                "dataset_row_range": [chunk_start, chunk_stop],
                "corpus_global_row_range": [chunk_start, chunk_stop],
                "r0087_global_row_range": [
                    R0087_PILE_GLOBAL_OFFSET + chunk_start,
                    R0087_PILE_GLOBAL_OFFSET + chunk_stop,
                ],
                "source_row_count": chunk_stop - chunk_start,
                "source_ids_ordered_sha256": ordered_array_sha256(
                    np.arange(chunk_start, chunk_stop, dtype=np.int64)
                ),
                "source_text_ordered_sha256": source_hash,
                "document_text_ordered_sha256": document_hash,
                "output": output_signature,
                "output_shape": [chunk_stop - chunk_start, DIMENSION],
                "output_dtype": OUTPUT_DTYPE.str,
                "float32_norm": float32_norm,
                "stored_norm": stored_norm,
                "embedding": telemetry,
                "encode_wall_s": encode_wall,
                "document_rows_per_s": (
                    (chunk_stop - chunk_start) / max(encode_wall, 1e-12)
                ),
                "wall_s": time.monotonic() - chunk_started,
            }
        )
        completed_rows = chunk_stop - start
        cumulative_rate = completed_rows / max(encode_wall_total, 1e-12)
        if (
            completed_rows >= PERFORMANCE_GUARD_ROWS
            and cumulative_rate < EMBED_MINIMUM_ROWS_PER_S
        ):
            raise Round0120Error(
                "R0120 document embedding throughput regressed below "
                f"{EMBED_MINIMUM_ROWS_PER_S:.1f} rows/s after "
                f"{completed_rows:,} rows: {cumulative_rate:.1f}"
            )
        if (
            completed_rows >= PERFORMANCE_GUARD_ROWS
            and cumulative_rate < EMBED_WARNING_ROWS_PER_S
        ):
            print(
                "[round0120] WARNING cumulative document rate "
                f"{cumulative_rate:.1f} rows/s is below "
                f"{EMBED_WARNING_ROWS_PER_S:.1f}",
                flush=True,
            )
        print(
            f"[round0120] {node_id} chunk {index + 1}/"
            f"{math.ceil((stop - start) / CHUNK_ROWS)} "
            f"{chunks[-1]['document_rows_per_s']:.1f} rows/s",
            flush=True,
        )

    wall = time.monotonic() - started
    try:
        import torch

        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    except Exception:
        peak_allocated = 0
        peak_reserved = 0
    body = {
        "schema": NODE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "node_id": node_id,
        "dataset": DATASET,
        "dataset_row_range": [start, stop],
        "corpus_global_row_range": [start, stop],
        "r0087_global_row_range": [
            R0087_PILE_GLOBAL_OFFSET + start,
            R0087_PILE_GLOBAL_OFFSET + stop,
        ],
        "source_layout": source_layout,
        "source_files_rehashed_at_node_boundary": verified_source_files,
        "boundary_closures": boundary_closures,
        "environment_freeze": job["environment_freeze"],
        "model": {
            **model_contract(),
            "members": model_members,
            "runtime_semantics": runtime_semantics,
        },
        "dimension": DIMENSION,
        "compute_dtype": COMPUTE_DTYPE,
        "output_dtype": OUTPUT_DTYPE.str,
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
        "prompt_name_equivalence": prompt_equivalence,
        "prompt_name_equivalence_passed": bool(
            prompt_equivalence
            and prompt_equivalence["array_equal"] is True
        ),
        "atomic_chunk_rows_maximum": CHUNK_ROWS,
        "chunks": chunks,
        "training_performed": False,
        "optimizer_updates": 0,
        "performance": {
            "wall_s": wall,
            "encode_wall_s": encode_wall_total,
            "document_rows_per_s": (stop - start) / max(wall, 1e-12),
            "encode_document_rows_per_s": (
                (stop - start) / max(encode_wall_total, 1e-12)
            ),
            "oom_retries": oom_retries,
            "requested_batch_size": BATCH_SIZE,
            "minimum_rows_per_s": EMBED_MINIMUM_ROWS_PER_S,
            "warning_rows_per_s": EMBED_WARNING_ROWS_PER_S,
            "peak_cuda_allocated_bytes": peak_allocated,
            "peak_cuda_reserved_bytes": peak_reserved,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            ),
        },
    }
    receipt = seal(body)
    receipt_path = os.path.join(output, "node-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    validate_node_receipt(receipt, node_id=node_id)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _scan_bound_chunk(
    signature: Mapping[str, Any], *, rows: int
) -> dict[str, Any]:
    actual = expected_input_signature(str(signature["canonical_path"]))
    expected = {
        key: signature[key]
        for key in ("kind", "canonical_path", "bytes", "sha256")
    }
    if actual != expected:
        raise Round0120Error("finalizer found changed embedding bytes")
    guard = _stored_array_guard(actual["canonical_path"], expected_rows=rows)
    return {"signature": actual, "norm": guard}


def run_finalize(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    output = create_fresh_directory(
        job["outputs"][0], label="R0120 canonical prompted Pile corpus"
    )
    final_boundary_closures = _verify_bound_closures(job)
    source_layout = [
        dict(item) for item in (job.get("canonical_source_layout") or [])
    ]
    _validate_node_layout(source_layout, start=0, stop=CORPUS_ROWS)
    final_source_signatures = _verify_source_files(source_layout)

    receipt_paths = [str(path) for path in job.get("node_receipts") or []]
    if len(receipt_paths) != len(WORK_RANGES):
        raise Round0120Error("R0120 finalizer has wrong node receipt count")
    receipts: list[dict[str, Any]] = []
    receipt_signatures: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    for path in receipt_paths:
        signature = expected_input_signature(path)
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        node_id = str(value.get("node_id") or "")
        receipt = validate_node_receipt(value, node_id=node_id)
        if (
            node_id in seen_nodes
            or receipt.get("release_sha")
            != active["manifest"]["release_sha"]
            or receipt.get("environment_freeze") != job["environment_freeze"]
        ):
            raise Round0120Error(
                f"R0120 node {node_id} finalizer binding changed"
            )
        expected_layout = clip_layout(
            source_layout,
            start=int(receipt["dataset_row_range"][0]),
            stop=int(receipt["dataset_row_range"][1]),
        )
        if receipt.get("source_layout") != expected_layout:
            raise Round0120Error(
                f"R0120 node {node_id} canonical source mapping changed"
            )
        seen_nodes.add(node_id)
        receipts.append(receipt)
        receipt_signatures.append(signature)
    if seen_nodes != {item[0] for item in WORK_RANGES}:
        raise Round0120Error("R0120 finalizer is missing a work node")

    combined: list[dict[str, Any]] = []
    scanned_rows = 0
    scanned_bytes = 0
    maximum_norm_error = 0.0
    for receipt in receipts:
        for chunk in receipt["chunks"]:
            start, stop = [
                int(item) for item in chunk["dataset_row_range"]
            ]
            scan = _scan_bound_chunk(chunk["output"], rows=stop - start)
            maximum_norm_error = max(
                maximum_norm_error,
                float(scan["norm"]["maximum_absolute_error"]),
            )
            scanned_rows += stop - start
            scanned_bytes += int(scan["signature"]["bytes"])
            combined.append(
                {
                    "dataset": DATASET,
                    "dataset_row_range": [start, stop],
                    "corpus_global_row_range": [start, stop],
                    "r0087_global_row_range": list(
                        chunk["r0087_global_row_range"]
                    ),
                    "source_text_ordered_sha256": chunk[
                        "source_text_ordered_sha256"
                    ],
                    "document_text_ordered_sha256": chunk[
                        "document_text_ordered_sha256"
                    ],
                    "output": scan["signature"],
                    "output_shape": [stop - start, DIMENSION],
                    "output_dtype": OUTPUT_DTYPE.str,
                    "provenance": {
                        "kind": "new-r0120-embedding",
                        "round_id": ROUND_ID,
                        "node_id": receipt["node_id"],
                        "node_receipt_identity_sha256": receipt[
                            "identity_sha256"
                        ],
                    },
                }
            )

    validate_coverage(combined)
    if scanned_rows != CORPUS_ROWS:
        raise Round0120Error("R0120 finalizer did not scan every Pile row")
    ordered_chunks = sorted(
        combined, key=lambda item: int(item["dataset_row_range"][0])
    )
    body = {
        "schema": CORPUS_SCHEMA,
        "capability": CAPABILITY,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "row_count": CORPUS_ROWS,
        "dimension": DIMENSION,
        "dtype": OUTPUT_DTYPE.str,
        "source_order": [DATASET],
        "r0087_selected_global_row_range": [
            R0087_PILE_GLOBAL_OFFSET,
            R0087_PILE_GLOBAL_STOP,
        ],
        "model": model_contract(),
        "convention": {
            "prompt_policy": "literal-document-prefix",
            "prompt_prefix": PROMPT_PREFIX,
            "prompt_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
            "native_prompt_name_equivalent": "document",
        },
        "source_layout": source_layout,
        "source_files_rehashed_at_finalizer": final_source_signatures,
        "input_closures_reverified_at_finalizer": final_boundary_closures,
        "environment_freeze": job["environment_freeze"],
        "dataset": {
            "name": DATASET,
            "row_count": CORPUS_ROWS,
            "corpus_global_row_range": [0, CORPUS_ROWS],
            "r0087_global_row_range": [
                R0087_PILE_GLOBAL_OFFSET,
                R0087_PILE_GLOBAL_STOP,
            ],
            "chunk_count": len(ordered_chunks),
            "chunks": ordered_chunks,
        },
        "new_embedding": {
            "rows": CORPUS_ROWS,
            "node_receipts": receipt_signatures,
            "node_receipt_identities": [
                receipt["identity_sha256"] for receipt in receipts
            ],
        },
        "coverage_validation": {
            "gap_free": True,
            "overlap_free": True,
            "one_to_one_source_output_rows": True,
            "canonical_r0087_source_order_preserved": True,
            "scanned_rows": scanned_rows,
            "scanned_embedding_file_bytes": scanned_bytes,
            "maximum_fp16_norm_absolute_error": maximum_norm_error,
            "all_finite": True,
            "all_normalized": True,
            "chunk_rows_maximum": CHUNK_ROWS,
        },
        "consumer_scope": {
            "basemap_view_allowed_after_review": True,
            "sae_view_allowed_after_review": True,
            "views_must_preserve_source_row_mapping": True,
            "complete_sae_training_corpus": False,
        },
        "claims": {
            "canonical_prompted_embedding_tranche_complete": True,
            "graph_built": False,
            "map_trained": False,
            "map_quality_estimated": False,
            "prompt_quality_effect_estimated": False,
            "production_promoted": False,
            "complete_sae_training_corpus": False,
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "performance": {
            "finalizer_wall_s": time.monotonic() - started,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
            ),
            "embedding_nodes": [
                receipt["performance"] for receipt in receipts
            ],
            "aggregate_document_rows_per_s": (
                CORPUS_ROWS
                / sum(
                    float(receipt["performance"]["wall_s"])
                    for receipt in receipts
                )
            ),
            "aggregate_oom_retries": sum(
                int(receipt["performance"]["oom_retries"])
                for receipt in receipts
            ),
        },
    }
    manifest = seal(body)
    path = os.path.join(output, f"{CORPUS_SCHEMA}.json")
    atomic_write_new_json(path, manifest, immutable=True)
    return {**manifest, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0120Error("R0120 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    validate_environment_freeze(selected.get("environment_freeze") or {})
    action = selected.get("action")
    if action == "embed_document_rows":
        return run_embed_document_rows(active, selected)
    if action == "finalize_canonical_prompted_pile":
        return run_finalize(active, selected)
    raise Round0120Error(f"unknown R0120 action: {action!r}")
