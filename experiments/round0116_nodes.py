"""GPU embedding nodes and CPU coverage finalizer for R0116."""
from __future__ import annotations

import json
import math
import os
import resource
import time
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
    MODEL_ROOT,
    model_member_signatures,
    ordered_text_sha256,
)
from basemap.round0116_prompted_corpus import (
    BATCH_SIZE,
    CAPABILITY,
    CHUNK_ROWS,
    COMPUTE_DTYPE,
    CORPUS_ROWS,
    DATASET_GLOBAL_OFFSETS,
    DATASET_ROWS,
    DIMENSION,
    EMBED_MINIMUM_ROWS_PER_S,
    EMBED_WARNING_ROWS_PER_S,
    FINEWEB,
    NATIVE_MAX_SEQ_LENGTH,
    NEW_ROWS,
    NODE_SCHEMA,
    OUTPUT_DTYPE,
    PERFORMANCE_GUARD_ROWS,
    PROMPT_PREFIX,
    R0114_IDENTITY_SHA256,
    REDPAJAMA,
    REUSED_FINEWEB_ROWS,
    ROUND_ID,
    CORPUS_SCHEMA,
    Round0116Error,
    clip_layout,
    expected_work_range,
    validate_environment_freeze,
    load_reused_manifest,
    model_contract,
    seal,
    validate_coverage,
    validate_node_receipt,
    validate_reused_mapping,
)


def _load_document_model():
    """Load the exact accepted native-8192 float32 model closure."""
    os.environ.setdefault("HF_HOME", "/data/hf")
    import torch
    from sentence_transformers import SentenceTransformer

    members = model_member_signatures()
    model = SentenceTransformer(
        MODEL_ROOT,
        trust_remote_code=True,
        device="cuda",
        model_kwargs={"torch_dtype": torch.float32},
        local_files_only=True,
    )
    if int(getattr(model, "max_seq_length", -1)) != NATIVE_MAX_SEQ_LENGTH:
        raise Round0116Error(
            "loaded SentenceTransformer does not resolve native max length 8192"
        )
    from experiments.embed_prompted_200k import inspect_loaded_jina_model

    runtime = inspect_loaded_jina_model(model)
    runtime["resolved_sentence_transformers_max_seq_length"] = int(
        model.max_seq_length
    )
    return model, runtime, members


def _encode_document(
    model: Any,
    texts: Sequence[str],
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    requested = BATCH_SIZE
    batch_size = requested
    retries = 0
    while True:
        try:
            values = np.asarray(
                model.encode(
                    list(texts),
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            return values, {
                "requested_batch_size": requested,
                "effective_batch_size": batch_size,
                "oom_retries": retries,
            }
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 8:
                raise
            retries += 1
            batch_size = max(8, batch_size // 2)
            torch.cuda.empty_cache()
            print(
                f"[round0116] CUDA OOM; retrying batch_size={batch_size}",
                flush=True,
            )


def _prompt_equivalence(
    model: Any,
    raw_texts: Sequence[str],
) -> dict[str, Any]:
    sample = list(raw_texts[:8])
    literal, literal_telemetry = _encode_document(
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
        raise Round0116Error(
            "literal Document prefix differs from prompt_name=document"
        )
    return {
        "rows": len(sample),
        "literal_prefix": PROMPT_PREFIX,
        "literal_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
        "native_prompt_name": "document",
        "array_equal": exact,
        "maximum_absolute_difference": maximum,
        "literal_telemetry": literal_telemetry,
    }


def _float32_norm_guard(
    values: np.ndarray,
    *,
    label: str,
) -> dict[str, Any]:
    if (
        values.ndim != 2
        or values.shape[1] != DIMENSION
        or values.dtype != np.dtype("float32")
        or not np.isfinite(values).all()
    ):
        raise Round0116Error(f"{label} embedding geometry is invalid")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    maximum = float(np.max(np.abs(norms - 1.0)))
    if maximum > 1e-3:
        raise Round0116Error(
            f"{label} float32 embeddings are not normalized: {maximum}"
        )
    return {
        "rows": len(values),
        "mean": float(np.mean(norms)),
        "minimum": float(np.min(norms)),
        "maximum": float(np.max(norms)),
        "maximum_absolute_error": maximum,
        "maximum_absolute_error_floor": 1e-3,
        "passed": True,
    }


def _stored_array_guard(
    path: str,
    *,
    expected_rows: int,
) -> dict[str, Any]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        values.shape != (expected_rows, DIMENSION)
        or values.dtype != OUTPUT_DTYPE
        or not np.isfinite(values).all()
    ):
        raise Round0116Error(f"stored embedding geometry changed: {path}")
    norms = np.linalg.norm(np.asarray(values, dtype=np.float32), axis=1)
    maximum = float(np.max(np.abs(norms - 1.0)))
    if maximum > 0.002:
        raise Round0116Error(
            f"stored fp16 embeddings are not normalized: {path}"
        )
    return {
        "rows": expected_rows,
        "mean": float(np.mean(norms)),
        "minimum": float(np.min(norms)),
        "maximum": float(np.max(norms)),
        "maximum_absolute_error": maximum,
        "maximum_absolute_error_floor": 0.002,
        "passed": True,
    }


class _SequentialTextReader:
    """Read contiguous rows while holding at most one parquet column."""

    def __init__(self, layout: Sequence[Mapping[str, Any]]) -> None:
        self.layout = [dict(item) for item in layout]
        self._cached_path: str | None = None
        self._cached_column: Any | None = None

    def _column(self, item: Mapping[str, Any]):
        path = str((item.get("text") or {})["canonical_path"])
        if path != self._cached_path:
            import pyarrow.parquet as pq

            column = pq.read_table(
                path,
                columns=[str(item["text_column"])],
            ).column(str(item["text_column"]))
            if len(column) != int(item["shard_rows"]):
                raise Round0116Error(
                    "source parquet changed after queue preparation"
                )
            self._cached_path = path
            self._cached_column = column
        return self._cached_column

    def read(self, start: int, stop: int) -> list[str]:
        if start < 0 or stop <= start:
            raise Round0116Error("invalid sequential text interval")
        values: list[str] = []
        cursor = start
        for item in self.layout:
            item_start = int(item["dataset_row_start"])
            item_stop = int(item["dataset_row_stop"])
            if item_stop <= start:
                continue
            if item_start >= stop:
                break
            take_start = max(start, item_start)
            take_stop = min(stop, item_stop)
            if take_start != cursor:
                raise Round0116Error("sequential text layout has a gap")
            column = self._column(item)
            local_start = int(item["shard_row_start"]) + (
                take_start - item_start
            )
            count = take_stop - take_start
            part = column.slice(local_start, count).to_pylist()
            if (
                len(part) != count
                or not all(isinstance(text, str) for text in part)
            ):
                raise Round0116Error("source text read is incomplete")
            values.extend(part)
            cursor = take_stop
        if cursor != stop or len(values) != stop - start:
            raise Round0116Error("source text interval did not close")
        return values


def _verify_source_files(
    layout: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rehash every source parquet at the GPU-node boundary."""
    expected_by_path: dict[str, dict[str, Any]] = {}
    for item in layout:
        expected = dict(item.get("text") or {})
        path = str(expected.get("canonical_path") or "")
        prior = expected_by_path.setdefault(path, expected)
        if prior != expected:
            raise Round0116Error(
                "source layout has conflicting signatures for one parquet"
            )
    observed = []
    for path in sorted(expected_by_path):
        expected = expected_by_path[path]
        actual = expected_input_signature(path)
        if actual != expected:
            raise Round0116Error(
                f"source parquet changed after queue preparation: {path}"
            )
        observed.append(actual)
    return observed


def run_embed_document_rows(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    node_id = str(job.get("id") or "")
    expected_dataset, expected_start, expected_stop = expected_work_range(
        node_id
    )
    dataset = str(job.get("dataset") or "")
    start = int(job.get("dataset_row_start", -1))
    stop = int(job.get("dataset_row_stop", -1))
    if (
        dataset != expected_dataset
        or (start, stop) != (expected_start, expected_stop)
        or len(job.get("outputs") or []) != 1
        or job.get("corpus_global_row_start")
        != DATASET_GLOBAL_OFFSETS[dataset] + start
        or job.get("corpus_global_row_stop")
        != DATASET_GLOBAL_OFFSETS[dataset] + stop
    ):
        raise Round0116Error(f"R0116 node {node_id} is malformed")

    source_layout = [
        dict(item)
        for item in (job.get("authenticated_source_layout") or [])
    ]
    if (
        not source_layout
        or int(source_layout[0]["dataset_row_start"]) != start
        or int(source_layout[-1]["dataset_row_stop"]) != stop
        or any(item.get("dataset") != dataset for item in source_layout)
    ):
        raise Round0116Error(f"R0116 node {node_id} source layout changed")
    verified_source_files = _verify_source_files(source_layout)
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"R0116 {node_id} embedding output",
    )
    chunks_root = create_fresh_directory(
        os.path.join(output, "chunks"),
        label=f"R0116 {node_id} atomic chunks",
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
        values, telemetry = _encode_document(model, document_texts)
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
            path,
            values.astype(OUTPUT_DTYPE),
            immutable=True,
        )
        stored_norm = _stored_array_guard(
            path, expected_rows=chunk_stop - chunk_start
        )
        output_signature = expected_input_signature(path)
        global_start = DATASET_GLOBAL_OFFSETS[dataset] + chunk_start
        global_stop = DATASET_GLOBAL_OFFSETS[dataset] + chunk_stop
        chunks.append(
            {
                "chunk_index": index,
                "dataset": dataset,
                "dataset_row_range": [chunk_start, chunk_stop],
                "corpus_global_row_range": [global_start, global_stop],
                "source_row_count": chunk_stop - chunk_start,
                "source_ids_ordered_sha256": ordered_array_sha256(
                    np.arange(
                        chunk_start,
                        chunk_stop,
                        dtype=np.int64,
                    )
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
                    (chunk_stop - chunk_start)
                    / max(encode_wall, 1e-12)
                ),
                "wall_s": time.monotonic() - chunk_started,
            }
        )
        completed_rows = chunk_stop - start
        cumulative_rate = completed_rows / max(
            encode_wall_total, 1e-12
        )
        if (
            completed_rows >= PERFORMANCE_GUARD_ROWS
            and cumulative_rate < EMBED_MINIMUM_ROWS_PER_S
        ):
            raise Round0116Error(
                "R0116 document embedding throughput regressed below "
                f"{EMBED_MINIMUM_ROWS_PER_S:.1f} rows/s after "
                f"{completed_rows:,} rows: {cumulative_rate:.1f}"
            )
        if (
            completed_rows >= PERFORMANCE_GUARD_ROWS
            and cumulative_rate < EMBED_WARNING_ROWS_PER_S
        ):
            print(
                "[round0116] WARNING cumulative document rate "
                f"{cumulative_rate:.1f} rows/s is below "
                f"{EMBED_WARNING_ROWS_PER_S:.1f}",
                flush=True,
            )
        print(
            f"[round0116] {node_id} chunk {index + 1}/"
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
        "dataset": dataset,
        "dataset_row_range": [start, stop],
        "corpus_global_row_range": [
            DATASET_GLOBAL_OFFSETS[dataset] + start,
            DATASET_GLOBAL_OFFSETS[dataset] + stop,
        ],
        "source_layout": source_layout,
        "source_files_rehashed_at_node_boundary": verified_source_files,
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
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024**2
            ),
        },
    }
    receipt = seal(body)
    receipt_path = os.path.join(output, "node-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    validate_node_receipt(receipt, node_id=node_id)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _scan_bound_chunk(
    signature: Mapping[str, Any],
    *,
    rows: int,
) -> dict[str, Any]:
    actual = expected_input_signature(str(signature["canonical_path"]))
    expected = {
        key: signature[key]
        for key in ("kind", "canonical_path", "bytes", "sha256")
    }
    if actual != expected:
        raise Round0116Error("finalizer found changed embedding bytes")
    guard = _stored_array_guard(
        actual["canonical_path"],
        expected_rows=rows,
    )
    return {
        "signature": actual,
        "norm": guard,
    }


def run_finalize(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0116 canonical prompted English corpus",
    )
    source_layout = [
        dict(item) for item in (job.get("canonical_source_layout") or [])
    ]
    reused_path = str(job.get("reused_manifest") or "")
    reused, reused_signature = load_reused_manifest(reused_path)
    lineage = job.get("r0114_source_lineage")
    reused_mapping = validate_reused_mapping(
        source_layout,
        reused,
        r0114_source_lineage=(
            list(lineage) if lineage is not None else None
        ),
    )
    if reused_mapping != job.get("reused_prefix_mapping"):
        raise Round0116Error(
            "R0116 reused-prefix mapping changed after queue preparation"
        )

    receipt_paths = [str(path) for path in job.get("node_receipts") or []]
    if len(receipt_paths) != 4:
        raise Round0116Error("R0116 finalizer requires four node receipts")
    receipts: list[dict[str, Any]] = []
    receipt_signatures: list[dict[str, Any]] = []
    for path in receipt_paths:
        signature = expected_input_signature(path)
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        node_id = str(value.get("node_id") or "")
        receipt = validate_node_receipt(value, node_id=node_id)
        if receipt.get("release_sha") != active["manifest"]["release_sha"]:
            raise Round0116Error(
                f"R0116 node {node_id} release binding changed"
            )
        if receipt.get("environment_freeze") != job["environment_freeze"]:
            raise Round0116Error(
                f"R0116 node {node_id} environment binding changed"
            )
        expected_layout = clip_layout(
            source_layout,
            dataset=receipt["dataset"],
            start=int(receipt["dataset_row_range"][0]),
            stop=int(receipt["dataset_row_range"][1]),
        )
        if receipt.get("source_layout") != expected_layout:
            raise Round0116Error(
                f"R0116 node {node_id} canonical source mapping changed"
            )
        receipts.append(receipt)
        receipt_signatures.append(signature)

    combined: list[dict[str, Any]] = []
    scanned_rows = 0
    scanned_bytes = 0
    maximum_norm_error = 0.0
    reused_chunks = list(
        reused["conventions"]["document"]["chunks"]
    )
    reused_text = list(reused["chunk_text_receipts"])
    for chunk, text in zip(reused_chunks, reused_text, strict=True):
        start, stop = [int(item) for item in text["source_row_range"]]
        scan = _scan_bound_chunk(chunk, rows=stop - start)
        maximum_norm_error = max(
            maximum_norm_error,
            float(scan["norm"]["maximum_absolute_error"]),
        )
        scanned_rows += stop - start
        scanned_bytes += int(scan["signature"]["bytes"])
        combined.append(
            {
                "dataset": FINEWEB,
                "dataset_row_range": [start, stop],
                "corpus_global_row_range": [start, stop],
                "source_text_ordered_sha256": text[
                    "source_text_ordered_sha256"
                ],
                "document_text_ordered_sha256": text[
                    "document_text_ordered_sha256"
                ],
                "output": scan["signature"],
                "output_shape": [stop - start, DIMENSION],
                "output_dtype": OUTPUT_DTYPE.str,
                "provenance": {
                    "kind": "reviewed-reuse",
                    "round_id": "0114",
                    "manifest_identity_sha256": R0114_IDENTITY_SHA256,
                },
            }
        )

    for receipt in receipts:
        for chunk in receipt["chunks"]:
            start, stop = [
                int(item) for item in chunk["dataset_row_range"]
            ]
            scan = _scan_bound_chunk(
                chunk["output"], rows=stop - start
            )
            maximum_norm_error = max(
                maximum_norm_error,
                float(scan["norm"]["maximum_absolute_error"]),
            )
            scanned_rows += stop - start
            scanned_bytes += int(scan["signature"]["bytes"])
            combined.append(
                {
                    "dataset": receipt["dataset"],
                    "dataset_row_range": [start, stop],
                    "corpus_global_row_range": list(
                        chunk["corpus_global_row_range"]
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
                        "kind": "new-r0116-embedding",
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
        raise Round0116Error("R0116 finalizer did not scan every corpus row")
    new_rows = sum(
        stop - start
        for receipt in receipts
        for start, stop in (
            [
                int(item)
                for item in chunk["dataset_row_range"]
            ]
            for chunk in receipt["chunks"]
        )
    )
    if new_rows != NEW_ROWS:
        raise Round0116Error("R0116 newly embedded row count changed")

    ordered_chunks = sorted(
        combined,
        key=lambda item: int(item["corpus_global_row_range"][0]),
    )
    per_dataset = {}
    for dataset in (FINEWEB, REDPAJAMA):
        selected = [
            item for item in ordered_chunks if item["dataset"] == dataset
        ]
        per_dataset[dataset] = {
            "row_count": DATASET_ROWS[dataset],
            "corpus_global_row_range": [
                DATASET_GLOBAL_OFFSETS[dataset],
                DATASET_GLOBAL_OFFSETS[dataset] + DATASET_ROWS[dataset],
            ],
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
        "source_order": [FINEWEB, REDPAJAMA],
        "model": model_contract(),
        "convention": {
            "prompt_policy": "literal-document-prefix",
            "prompt_prefix": PROMPT_PREFIX,
            "prompt_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
            "native_prompt_name_equivalent": "document",
        },
        "source_layout": source_layout,
        "environment_freeze": job["environment_freeze"],
        "datasets": per_dataset,
        "reuse": {
            "reviewed_manifest": reused_signature,
            "mapping": reused_mapping,
            "rows": REUSED_FINEWEB_ROWS,
            "bytes_copied": 0,
        },
        "new_embedding": {
            "rows": NEW_ROWS,
            "node_receipts": receipt_signatures,
            "node_receipt_identities": [
                receipt["identity_sha256"] for receipt in receipts
            ],
        },
        "coverage_validation": {
            "gap_free": True,
            "overlap_free": True,
            "one_to_one_source_output_rows": True,
            "canonical_source_order_preserved": True,
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
            "canonical_prompted_embedding_data_complete": True,
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
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024**2
            ),
            "embedding_nodes": [
                receipt["performance"] for receipt in receipts
            ],
            "aggregate_new_document_rows_per_s": (
                NEW_ROWS
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
        raise Round0116Error("R0116 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    validate_environment_freeze(selected.get("environment_freeze") or {})
    action = selected.get("action")
    if action == "embed_document_rows":
        return run_embed_document_rows(active, selected)
    if action == "finalize_canonical_prompted_english":
        return run_finalize(active, selected)
    raise Round0116Error(f"unknown R0116 action: {action!r}")
