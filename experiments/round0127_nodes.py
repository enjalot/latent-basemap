"""Independent GPU embedding nodes and CPU finalizer for R0127."""
from __future__ import annotations

import json
import math
import os
import resource
import time
from collections import Counter
from collections.abc import Mapping, Sequence
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
from basemap.round0127_prompted_multilingual import (
    BATCH_SIZE,
    CAPABILITY,
    CHUNK_ROWS,
    COMPUTE_DTYPE,
    CORPUS_ROWS,
    CORPUS_SCHEMA,
    DIMENSION,
    EMBED_MINIMUM_ROWS_PER_S,
    EMBED_WARNING_ROWS_PER_S,
    LANGUAGE_TRANCHES,
    NODE_SCHEMA,
    PERFORMANCE_GUARD_ROWS,
    PROMPT_PREFIX,
    R0087_GLOBAL_START,
    R0087_GLOBAL_STOP,
    ROUND_ID,
    ROWS_PER_LANGUAGE,
    Round0127Error,
    load_inventory_manifest,
    load_r0114_model_prompt_closure,
    model_contract,
    seal,
    source_for_node,
    tranche_for_node,
    validate_coverage,
    validate_environment_freeze,
    validate_node_receipt,
    validate_source_layout,
)
from experiments.round0116_nodes import (
    _SequentialTextReader,
    _encode_document,
    _float32_norm_guard,
    _load_document_model,
    _stored_array_guard,
)


_SIGNATURE_KEYS = ("kind", "canonical_path", "bytes", "sha256")
_SINGLETON_BOUNDARY_ROLES = {
    "round",
    "review-0087",
    "review-0114",
    "inventory",
    "model-prompt-manifest",
}
_BOUNDARY_ROLES = _SINGLETON_BOUNDARY_ROLES | {
    "model-member",
    "source-parquet",
    "raw-embedding",
}


def _signature_only(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in _SIGNATURE_KEYS}


def _rehash_boundary_inputs(
    bindings: Sequence[Mapping[str, Any]],
    *,
    require_all_sources: bool,
) -> list[dict[str, Any]]:
    """Rehash every declared byte binding inside the executing process."""
    if not bindings:
        raise Round0127Error("R0127 job has no authenticated boundary inputs")
    roles = Counter(str(item.get("role") or "") for item in bindings)
    required = set(_SINGLETON_BOUNDARY_ROLES)
    required.update({"model-member", "source-parquet", "raw-embedding"})
    if (
        any(roles[role] != 1 for role in _SINGLETON_BOUNDARY_ROLES)
        or any(roles[role] < 1 for role in required - _SINGLETON_BOUNDARY_ROLES)
        or set(roles) != _BOUNDARY_ROLES
        or (require_all_sources and roles["source-parquet"] != len(LANGUAGE_TRANCHES))
        or (require_all_sources and roles["raw-embedding"] != len(LANGUAGE_TRANCHES))
        or (not require_all_sources and roles["source-parquet"] != 1)
        or (not require_all_sources and roles["raw-embedding"] != 1)
    ):
        raise Round0127Error("R0127 boundary input role closure is incomplete")
    observed: list[dict[str, Any]] = []
    for binding in bindings:
        role = str(binding["role"])
        expected = _signature_only(binding["signature"])
        actual = expected_input_signature(str(expected["canonical_path"]))
        if actual != expected:
            raise Round0127Error(
                f"R0127 {role} bytes changed at job boundary: "
                f"{expected['canonical_path']}"
            )
        observed.append({"role": role, "signature": actual})
    return observed


def _roles(
    bindings: Sequence[Mapping[str, Any]], role: str
) -> list[dict[str, Any]]:
    return [
        _signature_only(item["signature"])
        for item in bindings
        if item.get("role") == role
    ]


def _verify_semantic_closures(
    job: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _, inventory_signature = load_inventory_manifest(
        str(job.get("inventory_manifest_path") or "")
    )
    if inventory_signature != job.get("inventory_manifest_signature"):
        raise Round0127Error("R0087 inventory binding changed at job boundary")
    _, r0114_signature = load_r0114_model_prompt_closure(
        str(job.get("r0114_manifest_path") or "")
    )
    if r0114_signature != job.get("r0114_manifest_signature"):
        raise Round0127Error("R0114 model/prompt binding changed at job boundary")
    return {
        "inventory_manifest": inventory_signature,
        "r0114_model_prompt_manifest": r0114_signature,
    }


def _verify_loaded_model_members(
    members: Sequence[Mapping[str, Any]],
    bindings: Sequence[Mapping[str, Any]],
) -> None:
    loaded = sorted(
        (_signature_only(item) for item in members),
        key=lambda item: item["canonical_path"],
    )
    bound = sorted(
        _roles(bindings, "model-member"),
        key=lambda item: item["canonical_path"],
    )
    if loaded != bound:
        raise Round0127Error("loaded Jina model differs from the queue binding")


def _encode_multilingual_document(
    model: Any, texts: Sequence[str]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the registered safe batch without mutating shared module state."""
    return _encode_document(
        model,
        texts,
        requested_batch_size=BATCH_SIZE,
    )


def _prompt_equivalence(
    model: Any, raw_texts: Sequence[str]
) -> dict[str, Any]:
    sample = list(raw_texts[:8])
    literal, literal_telemetry = _encode_multilingual_document(
        model, [PROMPT_PREFIX + text for text in sample]
    )
    named = np.asarray(
        model.encode(
            sample,
            prompt_name="document",
            batch_size=BATCH_SIZE,
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    exact = bool(np.array_equal(literal, named))
    maximum = float(np.max(np.abs(literal - named)))
    if not exact:
        raise Round0127Error(
            "literal Document prefix differs from prompt_name=document"
        )
    return {
        "rows": len(sample),
        "literal_prefix": PROMPT_PREFIX,
        "literal_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
        "native_prompt_name": "document",
        "array_equal": True,
        "maximum_absolute_difference": maximum,
        "literal_telemetry": literal_telemetry,
    }


def _validate_node_source(source: Mapping[str, Any], *, node_id: str) -> None:
    expected = tranche_for_node(node_id)
    if (
        source.get("node_id") != node_id
        or source.get("language") != expected["language"]
        or source.get("dataset") != expected["dataset"]
        or source.get("dataset_row_range") != [0, ROWS_PER_LANGUAGE]
        or source.get("corpus_global_row_range")
        != expected["corpus_global_row_range"]
        or source.get("r0087_global_row_range")
        != expected["r0087_global_row_range"]
    ):
        raise Round0127Error(f"R0127 source-row identity changed for {node_id}")


def run_embed_document_rows(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    job_started = time.monotonic()
    node_id = str(job.get("id") or "")
    expected = tranche_for_node(node_id)
    if (
        job.get("language") != expected["language"]
        or job.get("dataset") != expected["dataset"]
        or job.get("dataset_row_range") != [0, ROWS_PER_LANGUAGE]
        or job.get("corpus_global_row_range")
        != expected["corpus_global_row_range"]
        or job.get("r0087_global_row_range")
        != expected["r0087_global_row_range"]
        or len(job.get("outputs") or []) != 1
    ):
        raise Round0127Error(f"R0127 node {node_id} is malformed")

    bindings = [
        dict(item) for item in (job.get("authenticated_boundary_inputs") or [])
    ]
    boundary_rehash = _rehash_boundary_inputs(
        bindings, require_all_sources=False
    )
    closures = _verify_semantic_closures(job)
    source = dict(job.get("authenticated_source") or {})
    _validate_node_source(source, node_id=node_id)
    if _roles(bindings, "source-parquet") != [_signature_only(source["text"])]:
        raise Round0127Error("source parquet boundary binding changed")
    if _roles(bindings, "raw-embedding") != [
        _signature_only(source["accepted_raw_embedding"])
    ]:
        raise Round0127Error("raw embedding boundary binding changed")

    model, runtime_semantics, model_members = _load_document_model()
    _verify_loaded_model_members(model_members, bindings)
    output = create_fresh_directory(
        job["outputs"][0], label=f"R0127 {node_id} embedding output"
    )
    chunks_root = create_fresh_directory(
        os.path.join(output, "chunks"),
        label=f"R0127 {node_id} atomic chunks",
    )
    reader = _SequentialTextReader([source])
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass

    chunks: list[dict[str, Any]] = []
    prompt_equivalence: dict[str, Any] | None = None
    encode_wall_total = 0.0
    oom_retries = 0
    corpus_offset = int(expected["corpus_global_row_range"][0])
    accepted_offset = int(expected["r0087_global_row_range"][0])
    for index, chunk_start in enumerate(
        range(0, ROWS_PER_LANGUAGE, CHUNK_ROWS)
    ):
        chunk_stop = min(chunk_start + CHUNK_ROWS, ROWS_PER_LANGUAGE)
        chunk_started = time.monotonic()
        raw_texts = reader.read(chunk_start, chunk_stop)
        if prompt_equivalence is None:
            prompt_equivalence = _prompt_equivalence(model, raw_texts)
        document_texts = [PROMPT_PREFIX + text for text in raw_texts]
        source_hash = ordered_text_sha256(raw_texts)
        document_hash = ordered_text_sha256(document_texts)
        encode_started = time.monotonic()
        values, telemetry = _encode_multilingual_document(model, document_texts)
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
        atomic_save_new_npy(path, values.astype(OUTPUT_DTYPE), immutable=True)
        stored_norm = _stored_array_guard(
            path, expected_rows=chunk_stop - chunk_start
        )
        output_signature = expected_input_signature(path)
        chunks.append(
            {
                "chunk_index": index,
                "language": expected["language"],
                "dataset": expected["dataset"],
                "dataset_row_range": [chunk_start, chunk_stop],
                "corpus_global_row_range": [
                    corpus_offset + chunk_start,
                    corpus_offset + chunk_stop,
                ],
                "r0087_global_row_range": [
                    accepted_offset + chunk_start,
                    accepted_offset + chunk_stop,
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
        completed_rows = chunk_stop
        cumulative_rate = completed_rows / max(encode_wall_total, 1e-12)
        if (
            completed_rows >= PERFORMANCE_GUARD_ROWS
            and cumulative_rate < EMBED_MINIMUM_ROWS_PER_S
        ):
            raise Round0127Error(
                "R0127 document embedding throughput regressed below "
                f"{EMBED_MINIMUM_ROWS_PER_S:.1f} rows/s after "
                f"{completed_rows:,} rows: {cumulative_rate:.1f}"
            )
        if (
            completed_rows >= PERFORMANCE_GUARD_ROWS
            and cumulative_rate < EMBED_WARNING_ROWS_PER_S
        ):
            print(
                "[round0127] WARNING cumulative document rate "
                f"{cumulative_rate:.1f} rows/s is below "
                f"{EMBED_WARNING_ROWS_PER_S:.1f}",
                flush=True,
            )
        print(
            f"[round0127] {node_id} chunk {index + 1}/"
            f"{math.ceil(ROWS_PER_LANGUAGE / CHUNK_ROWS)} "
            f"{chunks[-1]['document_rows_per_s']:.1f} rows/s",
            flush=True,
        )

    wall = time.monotonic() - job_started
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
        "language": expected["language"],
        "dataset": expected["dataset"],
        "dataset_row_range": [0, ROWS_PER_LANGUAGE],
        "corpus_global_row_range": list(expected["corpus_global_row_range"]),
        "r0087_global_row_range": list(expected["r0087_global_row_range"]),
        "source_layout": source,
        "job_boundary_rehash": boundary_rehash,
        "input_closures": closures,
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
            prompt_equivalence and prompt_equivalence["array_equal"] is True
        ),
        "atomic_chunk_rows_maximum": CHUNK_ROWS,
        "chunks": chunks,
        "training_performed": False,
        "optimizer_updates": 0,
        "performance": {
            "wall_s": wall,
            "encode_wall_s": encode_wall_total,
            "document_rows_per_s": ROWS_PER_LANGUAGE / max(wall, 1e-12),
            "encode_document_rows_per_s": (
                ROWS_PER_LANGUAGE / max(encode_wall_total, 1e-12)
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
    expected = _signature_only(signature)
    actual = expected_input_signature(str(expected["canonical_path"]))
    if actual != expected:
        raise Round0127Error("finalizer found changed embedding bytes")
    guard = _stored_array_guard(actual["canonical_path"], expected_rows=rows)
    return {"signature": actual, "norm": guard}


def run_finalize(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    started = time.monotonic()
    bindings = [
        dict(item) for item in (job.get("authenticated_boundary_inputs") or [])
    ]
    final_boundary_rehash = _rehash_boundary_inputs(
        bindings, require_all_sources=True
    )
    closures = _verify_semantic_closures(job)
    source_layout = [
        dict(item) for item in (job.get("canonical_source_layout") or [])
    ]
    validate_source_layout(source_layout)
    expected_source_signatures = sorted(
        (_signature_only(item["text"]) for item in source_layout),
        key=lambda item: item["canonical_path"],
    )
    expected_raw_signatures = sorted(
        (
            _signature_only(item["accepted_raw_embedding"])
            for item in source_layout
        ),
        key=lambda item: item["canonical_path"],
    )
    if sorted(
        _roles(bindings, "source-parquet"),
        key=lambda item: item["canonical_path"],
    ) != expected_source_signatures or sorted(
        _roles(bindings, "raw-embedding"),
        key=lambda item: item["canonical_path"],
    ) != expected_raw_signatures:
        raise Round0127Error("finalizer source-byte bindings changed")

    receipt_paths = [str(path) for path in job.get("node_receipts") or []]
    if len(receipt_paths) != len(LANGUAGE_TRANCHES):
        raise Round0127Error("R0127 finalizer has wrong node receipt count")
    receipts: list[dict[str, Any]] = []
    receipt_signatures: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    for path in receipt_paths:
        signature = expected_input_signature(path)
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        node_id = str(value.get("node_id") or "")
        receipt = validate_node_receipt(value, node_id=node_id)
        expected_source = source_for_node(source_layout, node_id)
        if (
            node_id in seen_nodes
            or receipt.get("release_sha") != active["manifest"]["release_sha"]
            or receipt.get("environment_freeze") != job["environment_freeze"]
            or receipt.get("source_layout") != expected_source
        ):
            raise Round0127Error(
                f"R0127 node {node_id} finalizer binding changed"
            )
        seen_nodes.add(node_id)
        receipts.append(receipt)
        receipt_signatures.append(signature)
    if seen_nodes != {str(item["node_id"]) for item in LANGUAGE_TRANCHES}:
        raise Round0127Error("R0127 finalizer is missing a work node")

    combined: list[dict[str, Any]] = []
    scanned_rows = 0
    scanned_bytes = 0
    maximum_norm_error = 0.0
    for receipt in receipts:
        for chunk in receipt["chunks"]:
            start, stop = [int(item) for item in chunk["dataset_row_range"]]
            scan = _scan_bound_chunk(chunk["output"], rows=stop - start)
            maximum_norm_error = max(
                maximum_norm_error,
                float(scan["norm"]["maximum_absolute_error"]),
            )
            scanned_rows += stop - start
            scanned_bytes += int(scan["signature"]["bytes"])
            combined.append(
                {
                    "language": receipt["language"],
                    "dataset": receipt["dataset"],
                    "dataset_row_range": [start, stop],
                    "corpus_global_row_range": list(
                        chunk["corpus_global_row_range"]
                    ),
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
                        "kind": "new-r0127-embedding",
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
        raise Round0127Error("R0127 finalizer did not scan every row")

    output = create_fresh_directory(
        job["outputs"][0], label="R0127 multilingual prompted corpus"
    )
    ordered_chunks = sorted(
        combined, key=lambda item: int(item["corpus_global_row_range"][0])
    )
    languages: dict[str, dict[str, Any]] = {}
    for tranche in LANGUAGE_TRANCHES:
        language = str(tranche["language"])
        selected = [item for item in ordered_chunks if item["language"] == language]
        languages[language] = {
            "dataset": tranche["dataset"],
            "row_count": ROWS_PER_LANGUAGE,
            "dataset_row_range": [0, ROWS_PER_LANGUAGE],
            "corpus_global_row_range": tranche["corpus_global_row_range"],
            "r0087_global_row_range": tranche["r0087_global_row_range"],
            "chunk_count": len(selected),
            "chunks": selected,
        }
    body = {
        "schema": CORPUS_SCHEMA,
        "capability": CAPABILITY,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "row_count": CORPUS_ROWS,
        "dimension": DIMENSION,
        "dtype": OUTPUT_DTYPE.str,
        "source_order": [str(item["language"]) for item in LANGUAGE_TRANCHES],
        "r0087_selected_global_row_range": [
            R0087_GLOBAL_START,
            R0087_GLOBAL_STOP,
        ],
        "model": model_contract(),
        "convention": {
            "prompt_policy": "literal-document-prefix",
            "prompt_prefix": PROMPT_PREFIX,
            "prompt_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
            "native_prompt_name_equivalent": "document",
        },
        "source_layout": source_layout,
        "job_boundary_rehash": final_boundary_rehash,
        "input_closures": closures,
        "environment_freeze": job["environment_freeze"],
        "languages": languages,
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
            "embedding_nodes": [receipt["performance"] for receipt in receipts],
            "aggregate_document_rows_per_s": (
                CORPUS_ROWS
                / sum(float(receipt["performance"]["wall_s"]) for receipt in receipts)
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
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0127Error("R0127 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    validate_environment_freeze(selected.get("environment_freeze") or {})
    action = selected.get("action")
    if action == "embed_document_rows":
        return run_embed_document_rows(active, selected)
    if action == "finalize_prompted_multilingual_tranche":
        return run_finalize(active, selected)
    raise Round0127Error(f"unknown R0127 action: {action!r}")

