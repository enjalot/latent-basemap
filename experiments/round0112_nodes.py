"""GPU embedding and CPU finalization nodes for Round 0112."""
from __future__ import annotations

import json
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
from basemap.round0005_staging import (
    ROUND0005_MODEL_ID,
    ROUND0005_MODEL_REVISION,
    ROUND0005_NORMALIZATION,
    ROUND0005_POOLING,
)
from basemap.round0104_training import InventoryFp16Array
from basemap.round0112_prompt_substrate import (
    BATCH_SIZE,
    CHUNK_ROWS,
    COMPUTE_DTYPE,
    CONVENTIONS,
    DIMENSION,
    EMBED_MINIMUM_PAIRED_ROWS_PER_S,
    EMBED_WARNING_PAIRED_ROWS_PER_S,
    HISTORICAL_RAW_MEAN_COSINE_FLOOR,
    HISTORICAL_RAW_MIN_COSINE_FLOOR,
    MODEL_ROOT,
    OUTPUT_DTYPE,
    PROMPT_PREFIX,
    ROUND_ID,
    ROWS,
    SLICE_SCHEMA,
    SUBSTRATE_SCHEMA,
    Round0112Error,
    aggregate_slice_receipts,
    first2m_layout,
    load_eligibility_prefix,
    model_member_signatures,
    ordered_text_sha256,
    seal,
    source_contract,
)


FAITHFULNESS_ROWS_PER_SLICE = 64


def _load_model():
    """Load only the authenticated local snapshot at float32 compute."""
    model_members = model_member_signatures()
    os.environ.setdefault("HF_HOME", "/data/hf")
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        MODEL_ROOT,
        trust_remote_code=True,
        device="cuda",
        model_kwargs={"torch_dtype": torch.float32},
        local_files_only=True,
    )
    # Reuse the already-reviewed semantic inspector, not its retired launcher
    # or its old controller-specific encode wrapper.
    from experiments.embed_prompted_200k import inspect_loaded_jina_model

    runtime = inspect_loaded_jina_model(model)
    return model, runtime, model_members


def _encode(
    model: Any,
    texts: Sequence[str],
    *,
    prompt_name: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode one convention with bounded OOM backoff."""
    import torch

    batch_size = BATCH_SIZE
    oom_retries = 0
    while True:
        try:
            kwargs: dict[str, Any] = {}
            if prompt_name is not None:
                kwargs["prompt_name"] = prompt_name
            values = np.asarray(
                model.encode(
                    list(texts),
                    batch_size=batch_size,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    **kwargs,
                ),
                dtype=np.float32,
            )
            return values, {
                "requested_batch_size": BATCH_SIZE,
                "effective_batch_size": batch_size,
                "oom_retries": oom_retries,
            }
        except torch.cuda.OutOfMemoryError:
            if batch_size <= 8:
                raise
            oom_retries += 1
            batch_size = max(8, batch_size // 2)
            torch.cuda.empty_cache()
            print(
                f"[round0112] CUDA OOM; retrying batch_size={batch_size}",
                flush=True,
            )


class _TextReader:
    """Read ordered contiguous text ranges while caching each parquet once."""

    def __init__(self, layout: Sequence[Mapping[str, Any]]) -> None:
        self.layout = [dict(item) for item in layout]
        self._columns: dict[str, Any] = {}

    def read(self, start: int, stop: int) -> list[str]:
        import pyarrow.parquet as pq

        if not 0 <= start < stop <= ROWS:
            raise Round0112Error("R0112 text interval is out of range")
        values: list[str] = []
        cursor = start
        for item in self.layout:
            global_start = int(item["global_row_start"])
            global_stop = int(item["global_row_stop"])
            if global_stop <= start:
                continue
            if global_start >= stop:
                break
            take_start = max(start, global_start)
            take_stop = min(stop, global_stop)
            if take_start != cursor:
                raise Round0112Error("R0112 text layout is not contiguous")
            path = str(item["text_path"])
            column = self._columns.get(path)
            if column is None:
                column = pq.read_table(path, columns=["chunk_text"]).column(
                    "chunk_text"
                )
                if len(column) != int(item["shard_rows"]):
                    raise Round0112Error("R0112 text shard changed after preflight")
                self._columns[path] = column
            local_start = take_start - global_start
            count = take_stop - take_start
            part = column.slice(local_start, count).to_pylist()
            if len(part) != count or not all(isinstance(text, str) for text in part):
                raise Round0112Error("R0112 source text read is incomplete")
            values.extend(part)
            cursor = take_stop
        if cursor != stop or len(values) != stop - start:
            raise Round0112Error("R0112 text interval did not close")
        return values


def _normalized_guard(values: np.ndarray, *, label: str) -> dict[str, float]:
    if (
        values.shape[1:] != (DIMENSION,)
        or values.dtype != np.dtype("float32")
        or not np.isfinite(values).all()
    ):
        raise Round0112Error(f"{label} embeddings have invalid geometry")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    maximum_error = float(np.max(np.abs(norms - 1.0)))
    if maximum_error > 1e-3:
        raise Round0112Error(
            f"{label} output is not L2-normalized: max error {maximum_error}"
        )
    return {
        "mean": float(np.mean(norms)),
        "min": float(np.min(norms)),
        "max": float(np.max(norms)),
        "maximum_absolute_error": maximum_error,
    }


def _prompt_equivalence(model: Any, texts: Sequence[str]) -> dict[str, Any]:
    sample = list(texts[:8])
    explicit, explicit_telemetry = _encode(
        model, [PROMPT_PREFIX + text for text in sample]
    )
    named, named_telemetry = _encode(model, sample, prompt_name="document")
    exact = bool(np.array_equal(explicit, named))
    maximum = float(np.max(np.abs(explicit - named)))
    if not exact:
        raise Round0112Error(
            "literal Document prefix is not byte-identical to prompt_name=document"
        )
    return {
        "rows": len(sample),
        "literal_prefix": PROMPT_PREFIX,
        "native_prompt_name": "document",
        "array_equal": exact,
        "maximum_absolute_difference": maximum,
        "explicit_telemetry": explicit_telemetry,
        "native_telemetry": named_telemetry,
    }


def _faithfulness_positions(start: int, stop: int) -> np.ndarray:
    return np.sort(
        np.random.default_rng(11_200 + start).choice(
            stop - start,
            size=FAITHFULNESS_ROWS_PER_SLICE,
            replace=False,
        ).astype(np.int64)
    )


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Copy because callers persist the exact SentenceTransformer outputs; a
    # diagnostic cosine helper must not silently renormalize those arrays.
    a = np.array(left, dtype=np.float32, copy=True)
    b = np.array(right, dtype=np.float32, copy=True)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", a, b, dtype=np.float64)


def run_embed_slice(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    start = int(job["source_row_start"])
    stop = int(job["source_row_stop"])
    if (
        len(job.get("outputs") or []) != 1
        or stop - start != 500_000
        or start % 500_000
    ):
        raise Round0112Error("R0112 paired slice job is malformed")
    output = create_fresh_directory(
        job["outputs"][0],
        label=f"R0112 paired embedding slice {start}:{stop}",
    )
    arm_roots = {
        arm: create_fresh_directory(
            os.path.join(output, arm),
            label=f"R0112 {arm} embedding chunks",
        )
        for arm in CONVENTIONS
    }
    started = time.monotonic()
    layout = first2m_layout()
    authenticated_layout = list(job.get("authenticated_source_layout") or [])
    runtime_layout = [
        item
        for item in layout
        if int(item["global_row_stop"]) > start
        and int(item["global_row_start"]) < stop
    ]
    if len(authenticated_layout) != len(runtime_layout):
        raise Round0112Error("R0112 authenticated source layout is missing")
    for runtime, authenticated in zip(
        runtime_layout, authenticated_layout, strict=True
    ):
        if (
            runtime
            != {
                key: value
                for key, value in authenticated.items()
                if key != "text"
            }
            or os.path.realpath(str(runtime["text_path"]))
            != (authenticated.get("text") or {}).get("canonical_path")
            or os.path.getsize(str(runtime["text_path"]))
            != int((authenticated.get("text") or {}).get("bytes", -1))
        ):
            raise Round0112Error(
                "R0112 text/embedding layout changed after queue preparation"
            )
    reader = _TextReader(layout)
    model, runtime_semantics, model_members = _load_model()
    sample_positions = _faithfulness_positions(start, stop)
    historical_source = InventoryFp16Array(start, stop)
    sample_fresh: dict[int, np.ndarray] = {}
    chunks: list[dict[str, Any]] = []
    prompt_equivalence: dict[str, Any] | None = None
    oom_retries = 0
    cumulative_encode_wall = 0.0

    for chunk_index, chunk_start in enumerate(range(start, stop, CHUNK_ROWS)):
        chunk_stop = min(chunk_start + CHUNK_ROWS, stop)
        chunk_started = time.monotonic()
        texts = reader.read(chunk_start, chunk_stop)
        if prompt_equivalence is None:
            prompt_equivalence = _prompt_equivalence(model, texts)
        text_hash = ordered_text_sha256(texts)
        prompted = [PROMPT_PREFIX + text for text in texts]
        prompted_hash = ordered_text_sha256(prompted)

        encode_started = time.monotonic()
        raw, raw_telemetry = _encode(model, texts)
        document, document_telemetry = _encode(model, prompted)
        encode_wall = time.monotonic() - encode_started
        cumulative_encode_wall += encode_wall
        oom_retries += int(raw_telemetry["oom_retries"])
        oom_retries += int(document_telemetry["oom_retries"])
        raw_norm = _normalized_guard(raw, label="raw")
        document_norm = _normalized_guard(document, label="document")
        paired_cosine = _cosine_rows(raw, document)

        local_sample = sample_positions[
            (sample_positions >= chunk_start - start)
            & (sample_positions < chunk_stop - start)
        ]
        for position in local_sample.tolist():
            sample_fresh[int(position)] = raw[int(position - (chunk_start - start))]

        outputs: dict[str, dict[str, Any]] = {}
        for arm, values in (("raw", raw), ("document", document)):
            path = os.path.join(
                arm_roots[arm], f"data-{chunk_index:05d}.npy"
            )
            stored = values.astype(OUTPUT_DTYPE)
            atomic_save_new_npy(path, stored, immutable=True)
            outputs[arm] = expected_input_signature(path)

        chunks.append(
            {
                "chunk_index": chunk_index,
                "source_row_range": [chunk_start, chunk_stop],
                "source_row_count": chunk_stop - chunk_start,
                "source_ids_ordered_sha256": ordered_array_sha256(
                    np.arange(chunk_start, chunk_stop, dtype=np.int64)
                ),
                "source_text_ordered_sha256": text_hash,
                "document_text_ordered_sha256": prompted_hash,
                "outputs": outputs,
                "output_shape": [chunk_stop - chunk_start, DIMENSION],
                "output_dtype": OUTPUT_DTYPE.str,
                "raw_norm": raw_norm,
                "document_norm": document_norm,
                "paired_raw_document_cosine_mean": float(
                    np.mean(paired_cosine)
                ),
                "paired_raw_document_cosine_p01": float(
                    np.quantile(paired_cosine, 0.01)
                ),
                "raw_embedding": raw_telemetry,
                "document_embedding": document_telemetry,
                "encode_wall_s": encode_wall,
                "paired_source_rows_per_s": (
                    (chunk_stop - chunk_start) / max(encode_wall, 1e-12)
                ),
                "convention_rows_per_s": (
                    2 * (chunk_stop - chunk_start) / max(encode_wall, 1e-12)
                ),
                "wall_s": time.monotonic() - chunk_started,
            }
        )
        cumulative_source_rows = len(chunks) * CHUNK_ROWS
        cumulative_paired_rate = (
            cumulative_source_rows / max(cumulative_encode_wall, 1e-12)
        )
        if len(chunks) >= 2 and (
            cumulative_paired_rate < EMBED_MINIMUM_PAIRED_ROWS_PER_S
        ):
            raise Round0112Error(
                "R0112 embedding throughput regressed below the registered "
                f"{EMBED_MINIMUM_PAIRED_ROWS_PER_S:.1f} paired rows/s floor "
                f"after {cumulative_source_rows:,} rows: "
                f"{cumulative_paired_rate:.1f}"
            )
        if len(chunks) >= 2 and (
            cumulative_paired_rate < EMBED_WARNING_PAIRED_ROWS_PER_S
        ):
            print(
                "[round0112] WARNING cumulative embedding rate "
                f"{cumulative_paired_rate:.1f} paired rows/s is below "
                f"{EMBED_WARNING_PAIRED_ROWS_PER_S:.1f}",
                flush=True,
            )
        print(
            f"[round0112] {start}:{stop} chunk {chunk_index + 1:02d}/"
            f"{(stop - start) // CHUNK_ROWS:02d} "
            f"{chunks[-1]['paired_source_rows_per_s']:.1f} paired rows/s",
            flush=True,
        )

    if set(sample_fresh) != set(sample_positions.tolist()):
        raise Round0112Error("R0112 historical-faithfulness sample is incomplete")
    fresh = np.stack([sample_fresh[int(row)] for row in sample_positions])
    historical = np.asarray(historical_source[sample_positions], dtype=np.float32)
    historical_cosines = _cosine_rows(fresh, historical)
    faithfulness_passed = bool(
        float(np.mean(historical_cosines))
        >= HISTORICAL_RAW_MEAN_COSINE_FLOOR
        and float(np.min(historical_cosines))
        >= HISTORICAL_RAW_MIN_COSINE_FLOOR
    )
    if not faithfulness_passed:
        raise Round0112Error(
            "fresh raw local embeddings failed the historical alignment guard"
        )
    wall = time.monotonic() - started
    body = {
        "schema": SLICE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "source_row_range": [start, stop],
        "source_contract": source_contract(),
        "source_layout": authenticated_layout,
        "model_id": ROUND0005_MODEL_ID,
        "model_revision": ROUND0005_MODEL_REVISION,
        "model_members": model_members,
        "runtime_model_semantics": runtime_semantics,
        "pooling": ROUND0005_POOLING,
        "normalization": ROUND0005_NORMALIZATION,
        "compute_dtype": COMPUTE_DTYPE,
        "output_dtype": OUTPUT_DTYPE.str,
        "conventions": list(CONVENTIONS),
        "prompt_prefix": PROMPT_PREFIX,
        "prompt_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
        "prompt_name_equivalence": prompt_equivalence,
        "prompt_name_equivalence_passed": bool(
            prompt_equivalence and prompt_equivalence["array_equal"]
        ),
        "chunks": chunks,
        "historical_raw_sample_positions": sample_positions.tolist(),
        "historical_raw_cosines": historical_cosines.tolist(),
        "historical_raw_faithfulness": {
            "mean": float(np.mean(historical_cosines)),
            "min": float(np.min(historical_cosines)),
            "mean_floor": HISTORICAL_RAW_MEAN_COSINE_FLOOR,
            "min_floor": HISTORICAL_RAW_MIN_COSINE_FLOOR,
        },
        "historical_raw_faithfulness_passed": faithfulness_passed,
        "training_performed": False,
        "optimizer_updates": 0,
        "performance": {
            "wall_s": wall,
            "paired_source_rows_per_s": (stop - start) / wall,
            "convention_rows_per_s": 2 * (stop - start) / wall,
            "oom_retries": oom_retries,
            "minimum_paired_rows_per_s": (
                EMBED_MINIMUM_PAIRED_ROWS_PER_S
            ),
            "warning_paired_rows_per_s": (
                EMBED_WARNING_PAIRED_ROWS_PER_S
            ),
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    receipt = seal(body)
    receipt_path = os.path.join(output, "slice-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def run_finalize(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0112 dual-prompt embedding substrate",
    )
    receipt_paths = [str(path) for path in job["slice_receipts"]]
    receipts: list[dict[str, Any]] = []
    receipt_signatures: list[dict[str, Any]] = []
    for path in receipt_paths:
        receipt_signatures.append(expected_input_signature(path))
        with open(path, encoding="utf-8") as handle:
            receipts.append(json.load(handle))
    aggregate = aggregate_slice_receipts(receipts)
    excluded, eligibility_signature, eligibility_report = (
        load_eligibility_prefix()
    )
    selector_path = os.path.join(output, "duplicate-excluded-rows.i64.npy")
    atomic_save_new_npy(selector_path, excluded.astype("<i8"), immutable=True)
    selector_signature = expected_input_signature(selector_path)

    output_chunks: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in CONVENTIONS
    }
    for receipt in aggregate["slices"]:
        for chunk in receipt["chunks"]:
            for arm in CONVENTIONS:
                output_chunks[arm].append(dict(chunk["outputs"][arm]))
    expected_chunks = ROWS // CHUNK_ROWS
    if any(len(output_chunks[arm]) != expected_chunks for arm in CONVENTIONS):
        raise Round0112Error("R0112 aggregate output chunk count changed")
    body = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "row_count": ROWS,
        "dimension": DIMENSION,
        "source_contract": source_contract(),
        "model": {
            "id": ROUND0005_MODEL_ID,
            "revision": ROUND0005_MODEL_REVISION,
            "root": MODEL_ROOT,
            "pooling": ROUND0005_POOLING,
            "normalization": ROUND0005_NORMALIZATION,
            "compute_dtype": COMPUTE_DTYPE,
            "runtime_semantics": aggregate["slices"][0][
                "runtime_model_semantics"
            ],
        },
        "paired_invariant": {
            "same_ordered_text_rows": True,
            "same_model_bytes": True,
            "same_sentence_transformers_pipeline": True,
            "same_compute_dtype": True,
            "same_output_dtype": True,
            "only_varying_factor": (
                "literal UTF-8 Document: prefix prepended to treatment texts"
            ),
        },
        "conventions": {
            "raw": {
                "prompt_prefix": "",
                "prompt_applied": False,
                "chunks": output_chunks["raw"],
            },
            "document": {
                "prompt_prefix": PROMPT_PREFIX,
                "prompt_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
                "prompt_applied": True,
                "sentence_transformers_prompt_name_equivalent": "document",
                "chunks": output_chunks["document"],
            },
        },
        "slice_receipts": receipt_signatures,
        "diagnostics": {
            "historical_raw_faithfulness": aggregate[
                "historical_raw_cosine"
            ],
            "raw_document_shift": aggregate[
                "paired_raw_document_chunk_mean_cosine"
            ],
        },
        "duplicate_control": {
            "source": eligibility_signature,
            "selector": selector_signature,
            "cohort_reconciliation": eligibility_report,
            "excluded_exact_copy_rows": int(len(excluded)),
            "retained_representative_rows": ROWS - int(len(excluded)),
            "map_training_policy_for_successor": (
                "apply this identical representative-only selector to both "
                "arms; retain an in-cohort representative when R0087's global "
                "representative lies outside 2M; do not multiplicity-weight "
                "exact copies"
            ),
            "embedding_storage_policy": (
                "retain all rows in both conventions for causal identity and "
                "future SAE-corpus reuse"
            ),
        },
        "sae_reuse_scope": {
            "usable_first_tranche": True,
            "corpus": "FineWeb first 2M ordered rows",
            "document_prompt_available": True,
            "complete_sae_training_corpus": False,
            "note": (
                "future SAE training requires the intended broader corpus to "
                "be embedded under the same reviewed convention"
            ),
        },
        "claims": {
            "embedding_substrate_complete": True,
            "graph_built": False,
            "map_trained": False,
            "prompt_quality_effect_estimated": False,
            "production_prompt_transfer_resolved": False,
        },
        "training_performed": False,
        "optimizer_updates": 0,
    }
    manifest = seal(body)
    manifest_path = os.path.join(
        output, "jina-fineweb-2m-dual-prompt-embedding-substrate-v1.json"
    )
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    return {**manifest, "receipt": expected_input_signature(manifest_path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0112Error("R0112 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    action = selected.get("action")
    if action == "embed_paired_slice":
        return run_embed_slice(active, selected)
    if action == "finalize_dual_prompt_substrate":
        return run_finalize(active, selected)
    raise Round0112Error(f"unknown R0112 action: {action!r}")
