"""Prompted Jina embeddings for R0087's first multilingual tranches.

R0127 is deliberately a data-production capability.  It preserves the exact
R0087 row order for German, Greek, and French while replacing the historical
raw convention with the literal ``Document: `` convention accepted in R0114.
It does not build a graph, train a map, estimate quality, or promote a model.
"""
from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from .round0005_staging import (
    ROUND0005_DIMENSIONS,
    ROUND0005_MODEL_ID,
    ROUND0005_MODEL_REVISION,
)
from .round0112_prompt_substrate import MODEL_ROOT, OUTPUT_DTYPE, PROMPT_PREFIX
from .round0114_prompt_recovery import NATIVE_MAX_SEQ_LENGTH
from .round0116_prompted_corpus import (
    environment_freeze_receipt as _environment_freeze_receipt,
    validate_environment_freeze as _validate_environment_freeze,
)
from .round0120_prompted_pile import (
    R0114_MANIFEST_PATH,
    R0114_MANIFEST_SHA256,
    load_r0114_model_prompt_closure,
)

R0087_REVIEW_SHA256 = (
    "61ab9268899c2edc47519bdbe4efeea65a54f0c9fda52bd89e7cad0dafd9d483"
)
R0114_REVIEW_SHA256 = (
    "610a9abb93f3fb6908a018d855f81feecc1045e261c007a3ca13ad8379eec4b9"
)


ROUND_ID = "0127"
DIMENSION = ROUND0005_DIMENSIONS
CHUNK_ROWS = 25_000
BATCH_SIZE = 16
COMPUTE_DTYPE = "float32"
EMBED_MINIMUM_ROWS_PER_S = 150.0
EMBED_WARNING_ROWS_PER_S = 180.0
PERFORMANCE_GUARD_ROWS = 50_000
GPU_HOURS_CAP = 5.0
P90_BUDGET_ROWS_PER_S = 190.0
P90_FIXED_SECONDS_PER_NODE = 300.0
EXPECTED_ROWS_PER_S = 240.0
EXPECTED_FIXED_SECONDS_PER_NODE = 240.0

ROWS_PER_LANGUAGE = 835_454
LANGUAGE_TRANCHES = (
    {
        "node_id": "embed_deu_Latn",
        "language": "deu_Latn",
        "dataset": "fineweb2-deu_Latn-chunked-500-jina-v5-nano",
        "corpus_global_row_range": [0, ROWS_PER_LANGUAGE],
        "r0087_global_row_range": [11_632_738, 12_468_192],
    },
    {
        "node_id": "embed_ell_Grek",
        "language": "ell_Grek",
        "dataset": "fineweb2-ell_Grek-chunked-500-jina-v5-nano",
        "corpus_global_row_range": [ROWS_PER_LANGUAGE, 2 * ROWS_PER_LANGUAGE],
        "r0087_global_row_range": [12_468_192, 13_303_646],
    },
    {
        "node_id": "embed_fra_Latn",
        "language": "fra_Latn",
        "dataset": "fineweb2-fra_Latn-chunked-500-jina-v5-nano",
        "corpus_global_row_range": [2 * ROWS_PER_LANGUAGE, 3 * ROWS_PER_LANGUAGE],
        "r0087_global_row_range": [13_303_646, 14_139_100],
    },
)
CORPUS_ROWS = 3 * ROWS_PER_LANGUAGE
R0087_GLOBAL_START = 11_632_738
R0087_GLOBAL_STOP = 14_139_100

INVENTORY_MANIFEST_PATH = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-inventory-v1.json"
)
INVENTORY_MANIFEST_SHA256 = (
    "364aaa2f7a5e886f9cacdb96d3ffef1bbe697148e1babe10eee7817af0fc7163"
)
INVENTORY_IDENTITY_SHA256 = (
    "6c73f781208d16a84fc9e619e66c89f8fe56375dad77f7eda75e795f85cfec9b"
)
TEXT_ROOT = "/data/chunks"

NODE_SCHEMA = "round0127-canonical-document-multilingual-node-v1"
CORPUS_SCHEMA = "jina-document-multilingual-deu-ell-fra-2p506m-v1"
CAPABILITY = CORPUS_SCHEMA


class Round0127Error(RuntimeError):
    """The exact R0127 data-production contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {
        key: item for key, item in value.items() if key != "identity_sha256"
    }
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0127Error(f"{label} identity seal is invalid")


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_inventory_manifest(
    path: str = INVENTORY_MANIFEST_PATH,
    *,
    expected_sha256: str = INVENTORY_MANIFEST_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0127Error("R0087 inventory manifest bytes changed")
    value = _read_json(path)
    body = {
        key: item for key, item in value.items() if key != "identity_sha256"
    }
    selection = value.get("selection") or {}
    budgets = selection.get("budgets") or {}
    datasets = [str(item["dataset"]) for item in LANGUAGE_TRANCHES]
    source_order = list(selection.get("source_order") or [])
    try:
        positions = [source_order.index(dataset) for dataset in datasets]
    except ValueError as error:
        raise Round0127Error(
            "R0087 multilingual source order is incomplete"
        ) from error
    if (
        value.get("schema") != "jina-diverse-25m-inventory-v1"
        or value.get("identity_sha256") != INVENTORY_IDENTITY_SHA256
        or value.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or value.get("capability_ready") is not True
        or selection.get("complete") is not True
        or int(selection.get("selected_rows", -1)) != 25_000_000
        or positions != [6, 7, 8]
        or any(
            int(budgets.get(dataset, -1)) != ROWS_PER_LANGUAGE
            for dataset in datasets
        )
    ):
        raise Round0127Error("R0087 multilingual inventory capability is not exact")
    return value, signature


def _parquet_row_count_and_column(path: str) -> tuple[int, str]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    try:
        field = schema.field("chunk_text")
    except KeyError as error:
        raise Round0127Error(
            f"canonical source parquet lacks chunk_text: {path}"
        ) from error
    if not (
        pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ):
        raise Round0127Error(
            f"canonical source chunk_text is not UTF-8 text: {path}"
        )
    return int(parquet.metadata.num_rows), str(field.type)


def _npy_shape_and_dtype(path: str) -> tuple[tuple[int, ...], str]:
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return tuple(int(item) for item in values.shape), values.dtype.str


def source_layout_from_inventory(
    inventory: Mapping[str, Any],
    *,
    text_root: str = TEXT_ROOT,
    signature_fn: Callable[[str], Mapping[str, Any]] = expected_input_signature,
    parquet_inspector: Callable[[str], tuple[int, str]] = (
        _parquet_row_count_and_column
    ),
    npy_inspector: Callable[[str], tuple[tuple[int, ...], str]] = (
        _npy_shape_and_dtype
    ),
) -> list[dict[str, Any]]:
    """Map the exact three R0087 ranges onto authenticated source text."""
    all_ranges = list((inventory.get("selection") or {}).get("ranges") or [])
    result: list[dict[str, Any]] = []
    for tranche in LANGUAGE_TRANCHES:
        dataset = str(tranche["dataset"])
        matches = [item for item in all_ranges if item.get("dataset") == dataset]
        if len(matches) != 1:
            raise Round0127Error(
                f"R0087 must contain exactly one selected range for {dataset}"
            )
        selected = matches[0]
        shard = dict(selected.get("shard") or {})
        embedding_path = os.path.realpath(str(shard.get("canonical_path") or ""))
        expected_global = list(tranche["r0087_global_row_range"])
        if (
            int(selected.get("dataset_row_start", -1)) != 0
            or int(selected.get("dataset_row_stop", -1)) != ROWS_PER_LANGUAGE
            or int(selected.get("shard_row_start", -1)) != 0
            or int(selected.get("shard_row_stop", -1)) != ROWS_PER_LANGUAGE
            or [
                int(selected.get("global_row_start", -1)),
                int(selected.get("global_row_stop", -1)),
            ]
            != expected_global
            or selected.get("language") != tranche["language"]
            or int(shard.get("rows", -1)) != 2_000_000
            or int(shard.get("bytes", -1)) <= 0
            or len(str(shard.get("sha256") or "")) != 64
            or not embedding_path.endswith("/train/000_00000.npy")
        ):
            raise Round0127Error(f"R0087 range is malformed for {dataset}")

        raw_signature = dict(signature_fn(embedding_path))
        inventory_raw_signature = {
            "kind": "file",
            "canonical_path": embedding_path,
            "bytes": int(shard["bytes"]),
            "sha256": str(shard["sha256"]),
        }
        if raw_signature != inventory_raw_signature:
            raise Round0127Error(
                f"accepted raw embedding bytes changed for {dataset}"
            )
        raw_shape, raw_dtype = npy_inspector(embedding_path)
        if raw_shape != (2_000_000, DIMENSION) or raw_dtype != OUTPUT_DTYPE.str:
            raise Round0127Error(
                f"accepted raw embedding geometry changed for {dataset}"
            )

        source_name = dataset.removesuffix("-jina-v5-nano")
        text_path = os.path.realpath(
            os.path.join(text_root, source_name, "train", "000_00000.parquet")
        )
        text_rows, text_type = parquet_inspector(text_path)
        if text_rows != 2_000_000:
            raise Round0127Error(
                f"source parquet row count changed for {dataset}"
            )
        text_signature = dict(signature_fn(text_path))
        if (
            text_signature.get("canonical_path") != text_path
            or int(text_signature.get("bytes", -1)) <= 0
            or len(str(text_signature.get("sha256") or "")) != 64
        ):
            raise Round0127Error(
                f"source parquet signature is malformed for {dataset}"
            )
        result.append(
            {
                "node_id": tranche["node_id"],
                "language": tranche["language"],
                "dataset": dataset,
                "dataset_row_range": [0, ROWS_PER_LANGUAGE],
                "dataset_row_start": 0,
                "dataset_row_stop": ROWS_PER_LANGUAGE,
                "corpus_global_row_range": list(
                    tranche["corpus_global_row_range"]
                ),
                "r0087_global_row_range": expected_global,
                "shard_row_range": [0, ROWS_PER_LANGUAGE],
                "shard_row_start": 0,
                "shard_row_stop": ROWS_PER_LANGUAGE,
                "shard_rows": text_rows,
                "text_column": "chunk_text",
                "text_column_type": text_type,
                "text": text_signature,
                "accepted_raw_embedding": {
                    **raw_signature,
                    "rows": 2_000_000,
                    "dimension": DIMENSION,
                    "dtype": raw_dtype,
                    "selected_row_range": [0, ROWS_PER_LANGUAGE],
                },
            }
        )
    validate_source_layout(result)
    return result


def canonical_source_layout() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory, signature = load_inventory_manifest()
    return source_layout_from_inventory(inventory), signature


def tranche_for_node(node_id: str) -> dict[str, Any]:
    matches = [item for item in LANGUAGE_TRANCHES if item["node_id"] == node_id]
    if len(matches) != 1:
        raise Round0127Error(f"unknown R0127 work node {node_id!r}")
    return dict(matches[0])


def source_for_node(
    layout: Sequence[Mapping[str, Any]], node_id: str
) -> dict[str, Any]:
    matches = [dict(item) for item in layout if item.get("node_id") == node_id]
    if len(matches) != 1:
        raise Round0127Error(f"source layout does not close for {node_id}")
    return matches[0]


def validate_source_layout(layout: Sequence[Mapping[str, Any]]) -> None:
    if len(layout) != len(LANGUAGE_TRANCHES):
        raise Round0127Error("multilingual source layout has wrong tranche count")
    corpus_cursor = 0
    accepted_cursor = R0087_GLOBAL_START
    for expected, observed in zip(LANGUAGE_TRANCHES, layout, strict=True):
        dataset_rows = list(observed.get("dataset_row_range") or [])
        corpus_rows = list(observed.get("corpus_global_row_range") or [])
        accepted_rows = list(observed.get("r0087_global_row_range") or [])
        shard_rows = list(observed.get("shard_row_range") or [])
        text = observed.get("text") or {}
        raw = observed.get("accepted_raw_embedding") or {}
        if (
            observed.get("node_id") != expected["node_id"]
            or observed.get("language") != expected["language"]
            or observed.get("dataset") != expected["dataset"]
            or dataset_rows != [0, ROWS_PER_LANGUAGE]
            or int(observed.get("dataset_row_start", -1)) != 0
            or int(observed.get("dataset_row_stop", -1)) != ROWS_PER_LANGUAGE
            or corpus_rows != expected["corpus_global_row_range"]
            or accepted_rows != expected["r0087_global_row_range"]
            or shard_rows != [0, ROWS_PER_LANGUAGE]
            or int(observed.get("shard_row_start", -1)) != 0
            or int(observed.get("shard_row_stop", -1)) != ROWS_PER_LANGUAGE
            or corpus_rows[0] != corpus_cursor
            or accepted_rows[0] != accepted_cursor
            or int(observed.get("shard_rows", -1)) != 2_000_000
            or observed.get("text_column") != "chunk_text"
            or not str(text.get("canonical_path") or "")
            or len(str(text.get("sha256") or "")) != 64
            or not str(raw.get("canonical_path") or "")
            or len(str(raw.get("sha256") or "")) != 64
            or int(raw.get("rows", -1)) != 2_000_000
            or int(raw.get("dimension", -1)) != DIMENSION
            or raw.get("dtype") != OUTPUT_DTYPE.str
        ):
            raise Round0127Error("multilingual source layout identity changed")
        corpus_cursor = int(corpus_rows[1])
        accepted_cursor = int(accepted_rows[1])
    if corpus_cursor != CORPUS_ROWS or accepted_cursor != R0087_GLOBAL_STOP:
        raise Round0127Error("multilingual source layout did not close")


def model_contract() -> dict[str, Any]:
    return {
        "id": ROUND0005_MODEL_ID,
        "revision": ROUND0005_MODEL_REVISION,
        "root": MODEL_ROOT,
        "native_max_seq_length": NATIVE_MAX_SEQ_LENGTH,
        "pooling": "lasttoken",
        "normalization": "l2",
        "compute_dtype": COMPUTE_DTYPE,
        "output_dtype": OUTPUT_DTYPE.str,
        "dimension": DIMENSION,
        "literal_document_prefix": PROMPT_PREFIX,
        "literal_document_prefix_hex": PROMPT_PREFIX.encode("utf-8").hex(),
    }


def environment_freeze_receipt() -> dict[str, Any]:
    return _environment_freeze_receipt()


def validate_environment_freeze(expected: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_environment_freeze(expected)


def validate_node_receipt(
    value: Mapping[str, Any], *, node_id: str
) -> dict[str, Any]:
    receipt = dict(value)
    validate_seal(receipt, label=f"R0127 node {node_id}")
    expected = tranche_for_node(node_id)
    chunks = list(receipt.get("chunks") or [])
    source_layout = receipt.get("source_layout") or {}
    model = receipt.get("model") or {}
    runtime = model.get("runtime_semantics") or {}
    boundary = list(receipt.get("job_boundary_rehash") or [])
    boundary_roles = [str(item.get("role") or "") for item in boundary]
    closures = receipt.get("input_closures") or {}
    expected_chunks = math.ceil(ROWS_PER_LANGUAGE / CHUNK_ROWS)
    if (
        receipt.get("schema") != NODE_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("node_id") != node_id
        or receipt.get("language") != expected["language"]
        or receipt.get("dataset") != expected["dataset"]
        or receipt.get("dataset_row_range") != [0, ROWS_PER_LANGUAGE]
        or receipt.get("corpus_global_row_range")
        != expected["corpus_global_row_range"]
        or receipt.get("r0087_global_row_range")
        != expected["r0087_global_row_range"]
        or source_layout.get("node_id") != node_id
        or int(receipt.get("dimension", -1)) != DIMENSION
        or receipt.get("compute_dtype") != COMPUTE_DTYPE
        or receipt.get("output_dtype") != OUTPUT_DTYPE.str
        or receipt.get("prompt_prefix") != PROMPT_PREFIX
        or receipt.get("prompt_name_equivalence_passed") is not True
        or model.get("id") != ROUND0005_MODEL_ID
        or model.get("revision") != ROUND0005_MODEL_REVISION
        or model.get("root") != MODEL_ROOT
        or int(model.get("native_max_seq_length", -1)) != NATIVE_MAX_SEQ_LENGTH
        or model.get("pooling") != "lasttoken"
        or model.get("normalization") != "l2"
        or runtime.get("resolved_sentence_transformers_max_seq_length")
        != NATIVE_MAX_SEQ_LENGTH
        or receipt.get("training_performed") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
        or len(chunks) != expected_chunks
        or any(
            boundary_roles.count(role) != 1
            for role in (
                "round",
                "review-0087",
                "review-0114",
                "inventory",
                "model-prompt-manifest",
                "source-parquet",
                "raw-embedding",
            )
        )
        or boundary_roles.count("model-member") < 1
        or set(boundary_roles)
        != {
            "round",
            "review-0087",
            "review-0114",
            "inventory",
            "model-prompt-manifest",
            "model-member",
            "source-parquet",
            "raw-embedding",
        }
    ):
        raise Round0127Error(f"R0127 node {node_id} contract changed")

    def bound(role: str) -> list[dict[str, Any]]:
        return [
            {
                key: item["signature"][key]
                for key in ("kind", "canonical_path", "bytes", "sha256")
            }
            for item in boundary
            if item.get("role") == role
        ]

    inventory_bound = bound("inventory")[0]
    model_prompt_bound = bound("model-prompt-manifest")[0]
    source_bound = bound("source-parquet")[0]
    raw_bound = bound("raw-embedding")[0]
    model_members_bound = sorted(
        bound("model-member"), key=lambda item: item["canonical_path"]
    )
    model_members_receipt = sorted(
        (
            {
                key: item[key]
                for key in ("kind", "canonical_path", "bytes", "sha256")
            }
            for item in (model.get("members") or [])
        ),
        key=lambda item: item["canonical_path"],
    )
    if (
        bound("review-0087")[0]["sha256"] != R0087_REVIEW_SHA256
        or bound("review-0114")[0]["sha256"] != R0114_REVIEW_SHA256
        or inventory_bound.get("sha256") != INVENTORY_MANIFEST_SHA256
        or model_prompt_bound.get("sha256") != R0114_MANIFEST_SHA256
        or source_bound
        != {
            key: source_layout["text"][key]
            for key in ("kind", "canonical_path", "bytes", "sha256")
        }
        or raw_bound
        != {
            key: source_layout["accepted_raw_embedding"][key]
            for key in ("kind", "canonical_path", "bytes", "sha256")
        }
        or model_members_bound != model_members_receipt
        or closures.get("inventory_manifest") != inventory_bound
        or closures.get("r0114_model_prompt_manifest") != model_prompt_bound
    ):
        raise Round0127Error(f"R0127 node {node_id} byte closure changed")

    cursor = 0
    corpus_offset = int(expected["corpus_global_row_range"][0])
    accepted_offset = int(expected["r0087_global_row_range"][0])
    for chunk in chunks:
        start, stop = [int(item) for item in chunk["dataset_row_range"]]
        rows = stop - start
        output = chunk.get("output") or {}
        if (
            chunk.get("language") != expected["language"]
            or chunk.get("dataset") != expected["dataset"]
            or start != cursor
            or not 0 < rows <= CHUNK_ROWS
            or chunk.get("corpus_global_row_range")
            != [corpus_offset + start, corpus_offset + stop]
            or chunk.get("r0087_global_row_range")
            != [accepted_offset + start, accepted_offset + stop]
            or int(chunk.get("source_row_count", -1)) != rows
            or chunk.get("source_ids_ordered_sha256")
            != ordered_array_sha256(np.arange(start, stop, dtype=np.int64))
            or chunk.get("output_shape") != [rows, DIMENSION]
            or chunk.get("output_dtype") != OUTPUT_DTYPE.str
            or output.get("kind") != "file"
            or int(output.get("bytes", -1)) < rows * DIMENSION * OUTPUT_DTYPE.itemsize
            or int(output.get("bytes", -1))
            > rows * DIMENSION * OUTPUT_DTYPE.itemsize + 4_096
            or len(str(output.get("sha256") or "")) != 64
            or len(str(chunk.get("source_text_ordered_sha256") or "")) != 64
            or len(str(chunk.get("document_text_ordered_sha256") or "")) != 64
            or (chunk.get("stored_norm") or {}).get("passed") is not True
        ):
            raise Round0127Error(f"R0127 node {node_id} chunk coverage changed")
        cursor = stop
    performance = receipt.get("performance") or {}
    if (
        cursor != ROWS_PER_LANGUAGE
        or not math.isfinite(float(performance.get("wall_s", math.nan)))
        or float(performance.get("wall_s", 0.0)) <= 0.0
        or not math.isfinite(
            float(performance.get("document_rows_per_s", math.nan))
        )
        or float(performance.get("document_rows_per_s", 0.0)) <= 0.0
        or int(performance.get("oom_retries", -1)) < 0
        or int(performance.get("requested_batch_size", -1)) != BATCH_SIZE
    ):
        raise Round0127Error(f"R0127 node {node_id} telemetry is invalid")
    return receipt


def validate_coverage(chunks: Sequence[Mapping[str, Any]]) -> None:
    output_paths: set[str] = set()
    corpus_cursor = 0
    accepted_cursor = R0087_GLOBAL_START
    by_language: dict[str, int] = {}
    for chunk in sorted(
        chunks, key=lambda item: int(item["corpus_global_row_range"][0])
    ):
        local_start, local_stop = [
            int(item) for item in chunk["dataset_row_range"]
        ]
        corpus_start, corpus_stop = [
            int(item) for item in chunk["corpus_global_row_range"]
        ]
        accepted_start, accepted_stop = [
            int(item) for item in chunk["r0087_global_row_range"]
        ]
        language = str(chunk.get("language") or "")
        expected = next(
            (item for item in LANGUAGE_TRANCHES if item["language"] == language),
            None,
        )
        output_path = str((chunk.get("output") or {}).get("canonical_path") or "")
        if (
            expected is None
            or chunk.get("dataset") != expected["dataset"]
            or corpus_start != corpus_cursor
            or accepted_start != accepted_cursor
            or corpus_stop - corpus_start != local_stop - local_start
            or accepted_stop - accepted_start != local_stop - local_start
            or not 0 < local_stop - local_start <= CHUNK_ROWS
            or not output_path
            or output_path in output_paths
        ):
            raise Round0127Error(
                "multilingual corpus has a gap, overlap, repeated output, "
                "or row-order change"
            )
        prior = by_language.get(language, 0)
        if local_start != prior:
            raise Round0127Error("multilingual per-language row order changed")
        by_language[language] = local_stop
        output_paths.add(output_path)
        corpus_cursor = corpus_stop
        accepted_cursor = accepted_stop
    expected_languages = {str(item["language"]) for item in LANGUAGE_TRANCHES}
    if (
        corpus_cursor != CORPUS_ROWS
        or accepted_cursor != R0087_GLOBAL_STOP
        or set(by_language) != expected_languages
        or any(rows != ROWS_PER_LANGUAGE for rows in by_language.values())
    ):
        raise Round0127Error("canonical multilingual corpus does not close")


def production_payload_bytes() -> int:
    return CORPUS_ROWS * DIMENSION * OUTPUT_DTYPE.itemsize


def required_free_bytes() -> int:
    payload = production_payload_bytes()
    return int(math.ceil(payload * 1.25)) + 2 * 1024**3


def node_p90_seconds(rows: int = ROWS_PER_LANGUAGE) -> float:
    return float(rows / P90_BUDGET_ROWS_PER_S + P90_FIXED_SECONDS_PER_NODE)


def expected_gpu_seconds() -> float:
    return float(
        CORPUS_ROWS / EXPECTED_ROWS_PER_S
        + len(LANGUAGE_TRANCHES) * EXPECTED_FIXED_SECONDS_PER_NODE
    )


def worst_passing_gpu_seconds() -> float:
    return float(
        CORPUS_ROWS / EMBED_MINIMUM_ROWS_PER_S
        + len(LANGUAGE_TRANCHES) * P90_FIXED_SECONDS_PER_NODE
    )


def source_manifest_summary(
    layout: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_source_layout(layout)
    return {
        "rows": CORPUS_ROWS,
        "languages": [str(item["language"]) for item in layout],
        "source_parquet_bytes": sum(int(item["text"]["bytes"]) for item in layout),
        "source_parquet_sha256s": [item["text"]["sha256"] for item in layout],
        "accepted_raw_embedding_bytes": sum(
            int(item["accepted_raw_embedding"]["bytes"]) for item in layout
        ),
        "accepted_raw_embedding_sha256s": [
            item["accepted_raw_embedding"]["sha256"] for item in layout
        ],
        "r0087_global_row_range": [R0087_GLOBAL_START, R0087_GLOBAL_STOP],
    }
