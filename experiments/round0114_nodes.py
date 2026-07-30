"""CPU-only validation and resealing of R0112's completed paired bytes."""
from __future__ import annotations

import json
import os
import resource
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
from basemap.round0104_training import InventoryFp16Array
from basemap.round0112_prompt_substrate import (
    CHUNK_ROWS,
    CONVENTIONS,
    DIMENSION,
    MODEL_ROOT,
    OUTPUT_DTYPE,
    PROMPT_PREFIX,
    ROWS,
)
from basemap.round0112_prompt_substrate import (
    first2m_layout,
    load_eligibility_prefix,
    model_member_signatures,
    ordered_text_sha256,
    source_contract,
)
from basemap.round0114_prompt_recovery import (
    COMPARABLE_MEAN_FLOOR,
    COMPARABLE_MINIMUM_FLOOR,
    HISTORICAL_MAX_SEQ_LENGTH,
    HISTORICAL_OVERALL_MEAN_FLOOR,
    NATIVE_MAX_SEQ_LENGTH,
    RECOVERY_SCHEMA,
    ROUND_ID,
    ROW_IDENTITY_RADIUS,
    SOURCE_FAILED_PATH,
    SOURCE_RELEASE_SHA,
    SOURCE_TERMINAL_PATH,
    Round0114Error,
    seal,
    source_chunk_path,
    source_sample_positions,
    validate_source_failure,
    validate_source_terminal,
)
from experiments.round0112_nodes import _TextReader


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _validate_model_contract() -> dict[str, Any]:
    model_config_path = os.path.join(MODEL_ROOT, "config.json")
    tokenizer_config_path = os.path.join(MODEL_ROOT, "tokenizer_config.json")
    sentence_config_path = os.path.join(
        MODEL_ROOT,
        "config_sentence_transformers.json",
    )
    model_config = _read_json(model_config_path)
    tokenizer_config = _read_json(tokenizer_config_path)
    sentence_config = _read_json(sentence_config_path)
    tokenizer_limit = int(tokenizer_config.get("model_max_length", -1))
    if (
        int(model_config.get("max_position_embeddings", -1))
        != NATIVE_MAX_SEQ_LENGTH
        or tokenizer_limit < NATIVE_MAX_SEQ_LENGTH
        or sentence_config.get("prompts")
        != {"query": "Query: ", "document": PROMPT_PREFIX}
        or sentence_config.get("default_prompt_name") is not None
    ):
        raise Round0114Error("R0114 native model/prompt contract changed")
    return {
        "resolved_sentence_transformers_max_seq_length": min(
            NATIVE_MAX_SEQ_LENGTH,
            tokenizer_limit,
        ),
        "model_config": expected_input_signature(model_config_path),
        "tokenizer_config": expected_input_signature(tokenizer_config_path),
        "sentence_transformers_config": expected_input_signature(
            sentence_config_path
        ),
    }


def _cosine_rows(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.einsum("ij,ij->i", a, b, dtype=np.float64)


def _validate_chunk(path: str, *, label: str) -> np.ndarray:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (CHUNK_ROWS, DIMENSION) or values.dtype != OUTPUT_DTYPE:
        raise Round0114Error(f"{label} geometry changed")
    # R0112 checked float32 outputs before storage.  Recheck every persisted
    # fp16 value here; norm drift is allowed only at fp16 rounding scale.
    if not np.isfinite(values).all():
        raise Round0114Error(f"{label} contains nonfinite values")
    norms = np.linalg.norm(np.asarray(values, dtype=np.float32), axis=1)
    maximum_error = float(np.max(np.abs(norms - 1.0)))
    if maximum_error > 0.002:
        raise Round0114Error(f"{label} is not unit-normalized fp16 output")
    return values


def _fresh_rows(positions: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for position in positions.tolist():
        chunk, local = divmod(int(position), CHUNK_ROWS)
        values = np.load(
            source_chunk_path("raw", chunk),
            mmap_mode="r",
            allow_pickle=False,
        )
        rows.append(np.asarray(values[local], dtype=np.float32))
    return np.stack(rows)


def _historical_diagnostic(
    positions: np.ndarray,
    texts: Mapping[int, str],
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    fresh = _fresh_rows(positions)
    historical_source = InventoryFp16Array(0, ROWS)
    historical = np.asarray(historical_source[positions], dtype=np.float32)
    cosines = _cosine_rows(fresh, historical)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        trust_remote_code=True,
    )
    token_counts = np.asarray(
        [
            len(
                tokenizer(
                    texts[int(position)],
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            for position in positions.tolist()
        ],
        dtype=np.int64,
    )
    comparable = token_counts <= HISTORICAL_MAX_SEQ_LENGTH
    if not np.any(comparable) or not np.any(~comparable):
        raise Round0114Error("R0114 truncation strata are unexpectedly empty")

    top1 = 0
    margins: list[float] = []
    for index, position in enumerate(positions.tolist()):
        start = max(0, position - ROW_IDENTITY_RADIUS)
        stop = min(ROWS, position + ROW_IDENTITY_RADIUS + 1)
        candidates = np.asarray(
            InventoryFp16Array(start, stop)[:],
            dtype=np.float32,
        )
        candidate_cosines = _cosine_rows(
            np.repeat(fresh[index : index + 1], len(candidates), axis=0),
            candidates,
        )
        own = position - start
        best = int(np.argmax(candidate_cosines))
        if best == own:
            top1 += 1
        competitor = float(np.max(np.delete(candidate_cosines, own)))
        margins.append(float(candidate_cosines[own] - competitor))

    overall_mean = float(np.mean(cosines))
    comparable_cosines = cosines[comparable]
    passed = bool(
        overall_mean >= HISTORICAL_OVERALL_MEAN_FLOOR
        and float(np.mean(comparable_cosines)) >= COMPARABLE_MEAN_FLOOR
        and float(np.min(comparable_cosines)) >= COMPARABLE_MINIMUM_FLOOR
        and top1 == len(positions)
        and min(margins) > 0.0
    )
    if not passed:
        raise Round0114Error("R0114 corrected historical row-identity proof failed")
    return {
        "sample_positions": positions.tolist(),
        "sample_rows": int(len(positions)),
        "historical_pipeline_max_seq_length": HISTORICAL_MAX_SEQ_LENGTH,
        "fresh_pipeline_max_seq_length": NATIVE_MAX_SEQ_LENGTH,
        "all_rows": {
            "mean_cosine": overall_mean,
            "minimum_cosine": float(np.min(cosines)),
            "mean_floor": HISTORICAL_OVERALL_MEAN_FLOOR,
            "below_original_0p95_count": int(np.sum(cosines < 0.95)),
        },
        "historically_comparable_rows": {
            "definition": "unpadded tokenizer length including specials <= 512",
            "rows": int(np.sum(comparable)),
            "mean_cosine": float(np.mean(comparable_cosines)),
            "minimum_cosine": float(np.min(comparable_cosines)),
            "mean_floor": COMPARABLE_MEAN_FLOOR,
            "minimum_floor": COMPARABLE_MINIMUM_FLOOR,
        },
        "native_long_context_rows": {
            "definition": "unpadded tokenizer length including specials > 512",
            "rows": int(np.sum(~comparable)),
            "mean_cosine": float(np.mean(cosines[~comparable])),
            "minimum_cosine": float(np.min(cosines[~comparable])),
            "role": "expected truncation-semantic diagnostic; no minimum gate",
        },
        "same_row_top1_count": top1,
        "same_row_top1_denominator": int(len(positions)),
        "candidate_window_radius": ROW_IDENTITY_RADIUS,
        "minimum_same_row_margin": float(min(margins)),
        "passed": passed,
    }


def _sample_texts(positions: np.ndarray) -> dict[int, str]:
    values: dict[int, str] = {}
    for item in first2m_layout():
        selected = positions[
            (positions >= int(item["global_row_start"]))
            & (positions < int(item["global_row_stop"]))
        ]
        if not len(selected):
            continue
        import pyarrow.parquet as pq

        column = pq.read_table(
            str(item["text_path"]),
            columns=["chunk_text"],
        ).column("chunk_text")
        for position in selected.tolist():
            local = position - int(item["global_row_start"])
            text = column[local].as_py()
            if not isinstance(text, str):
                raise Round0114Error("R0114 sampled source text is not a string")
            values[int(position)] = text
    if set(values) != set(positions.tolist()):
        raise Round0114Error("R0114 sampled source text did not close")
    return values


def run_recover(active: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    terminal = _read_json(SOURCE_TERMINAL_PATH)
    failure = _read_json(SOURCE_FAILED_PATH)
    validate_source_terminal(terminal)
    validate_source_failure(failure)
    model_contract = _validate_model_contract()

    output = create_fresh_directory(
        job["outputs"][0],
        label="R0114 recovered native-8192 paired substrate",
    )
    chunks: dict[str, list[dict[str, Any]]] = {
        arm: [] for arm in CONVENTIONS
    }
    reader = _TextReader(first2m_layout())
    chunk_text_receipts: list[dict[str, Any]] = []
    for global_chunk in range(ROWS // CHUNK_ROWS):
        start = global_chunk * CHUNK_ROWS
        stop = start + CHUNK_ROWS
        texts = reader.read(start, stop)
        chunk_text_receipts.append(
            {
                "source_row_range": [start, stop],
                "source_text_ordered_sha256": ordered_text_sha256(texts),
                "document_text_ordered_sha256": ordered_text_sha256(
                    [PROMPT_PREFIX + text for text in texts]
                ),
            }
        )
        for arm in CONVENTIONS:
            path = source_chunk_path(arm, global_chunk)
            _validate_chunk(path, label=f"R0112 {arm} chunk {global_chunk}")
            chunks[arm].append(expected_input_signature(path))

    positions = np.asarray(source_sample_positions(), dtype=np.int64)
    historical = _historical_diagnostic(positions, _sample_texts(positions))
    excluded, eligibility_signature, eligibility_report = (
        load_eligibility_prefix()
    )
    selector_path = os.path.join(output, "duplicate-excluded-rows.i64.npy")
    atomic_save_new_npy(selector_path, excluded.astype("<i8"), immutable=True)
    selector_signature = expected_input_signature(selector_path)

    body = {
        "schema": RECOVERY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "row_count": ROWS,
        "dimension": DIMENSION,
        "source_contract": source_contract(),
        "source_round": {
            "round_id": "0112",
            "release_sha": SOURCE_RELEASE_SHA,
            "terminal": expected_input_signature(SOURCE_TERMINAL_PATH),
            "failure": expected_input_signature(SOURCE_FAILED_PATH),
            "verdict_preserved": "failed",
            "gpu_wall_s_preserved": float(terminal["gpu_wall_s"]),
        },
        "model": {
            "root": os.path.realpath(MODEL_ROOT),
            "members": model_member_signatures(),
            "native_max_seq_length": NATIVE_MAX_SEQ_LENGTH,
            "historical_max_seq_length": HISTORICAL_MAX_SEQ_LENGTH,
            "output_dtype": OUTPUT_DTYPE.str,
            "runtime_resolution": model_contract,
        },
        "paired_invariant": {
            "same_ordered_text_rows": True,
            "same_model_bytes": True,
            "same_sentence_transformers_execution": True,
            "same_native_max_seq_length": NATIVE_MAX_SEQ_LENGTH,
            "only_varying_factor": (
                "literal UTF-8 Document: prefix prepended to treatment texts"
            ),
        },
        "conventions": {
            "raw": {
                "prompt_prefix": "",
                "prompt_applied": False,
                "chunks": chunks["raw"],
            },
            "document": {
                "prompt_prefix": PROMPT_PREFIX,
                "prompt_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
                "prompt_applied": True,
                "chunks": chunks["document"],
            },
        },
        "chunk_text_receipts": chunk_text_receipts,
        "diagnostics": {
            "historical_row_identity": historical,
            "original_r0112_guard": {
                "passed": False,
                "reason": (
                    "one >512-token row compared native-8192 fresh semantics "
                    "against historical explicit-512 truncation"
                ),
                "acceptance_rewritten": False,
                "recovery_registered_as_new_round": True,
            },
        },
        "duplicate_control": {
            "source": eligibility_signature,
            "selector": selector_signature,
            "cohort_reconciliation": eligibility_report,
            "excluded_exact_copy_rows": int(len(excluded)),
            "retained_representative_rows": ROWS - int(len(excluded)),
        },
        "claims": {
            "native8192_embedding_substrate_complete": True,
            "r0112_accepted": False,
            "graph_built": False,
            "map_trained": False,
            "prompt_quality_effect_estimated": False,
            "complete_sae_training_corpus": False,
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "performance": {
            "wall_s": time.monotonic() - started,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
            ),
        },
    }
    manifest = seal(body)
    path = os.path.join(
        output,
        "jina-fineweb-2m-dual-prompt-native8192-substrate-v2.json",
    )
    atomic_write_new_json(path, manifest, immutable=True)
    return {**manifest, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0114Error("R0114 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "recover_native8192_substrate":
        raise Round0114Error(f"unknown R0114 action: {selected.get('action')!r}")
    return run_recover(active, selected)
