"""Canonical native-8192 ``Document: `` Jina corpus contracts for R0116.

R0116 is data production only.  It reuses the exact reviewed R0114 document
embeddings for the first two million FineWeb rows, embeds the remaining
FineWeb and RedPajama rows in canonical R0087 order, and seals one manifest
over the combined coverage.  It does not train or evaluate a map.
"""
from __future__ import annotations

import json
import math
import os
import platform
import sys
from importlib import metadata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0005_staging import (
    ROUND0005_DIMENSIONS,
    ROUND0005_MODEL_ID,
    ROUND0005_MODEL_REVISION,
)
from .round0087_inventory import FINEWEB, REDPAJAMA
from .round0112_prompt_substrate import (
    MODEL_ROOT,
    OUTPUT_DTYPE,
    PROMPT_PREFIX,
)
from .round0114_prompt_recovery import (
    NATIVE_MAX_SEQ_LENGTH,
    RECOVERY_SCHEMA,
)


ROUND_ID = "0116"
DIMENSION = ROUND0005_DIMENSIONS
CHUNK_ROWS = 25_000
BATCH_SIZE = 256
COMPUTE_DTYPE = "float32"
EMBED_MINIMUM_ROWS_PER_S = 120.0
EMBED_WARNING_ROWS_PER_S = 180.0
PERFORMANCE_GUARD_ROWS = 50_000

FINEWEB_ROWS = 2_890_362
REDPAJAMA_ROWS = 2_836_978
REUSED_FINEWEB_ROWS = 2_000_000
CORPUS_ROWS = FINEWEB_ROWS + REDPAJAMA_ROWS
NEW_ROWS = CORPUS_ROWS - REUSED_FINEWEB_ROWS

DATASET_ROWS = {
    FINEWEB: FINEWEB_ROWS,
    REDPAJAMA: REDPAJAMA_ROWS,
}
DATASET_GLOBAL_OFFSETS = {
    FINEWEB: 0,
    REDPAJAMA: FINEWEB_ROWS,
}
TEXT_ROOTS = {
    FINEWEB: (
        "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train"
    ),
    REDPAJAMA: (
        "/data/chunks/"
        "RedPajama-Data-V2-sample-10B-chunked-500/train"
    ),
}

# These are independently retryable queue nodes.  A failed node may be
# archived and retried without invalidating any earlier done marker.
WORK_RANGES = (
    ("embed_fineweb_tail", FINEWEB, 2_000_000, 2_890_362),
    ("embed_redpajama_00", REDPAJAMA, 0, 1_000_000),
    ("embed_redpajama_01", REDPAJAMA, 1_000_000, 2_000_000),
    ("embed_redpajama_02", REDPAJAMA, 2_000_000, 2_836_978),
)

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
R0114_MANIFEST_PATH = (
    "/data/latent-basemap/runs/round-0114/queue/artifacts/"
    "jina-fineweb-2m-dual-prompt-native8192-substrate/"
    "jina-fineweb-2m-dual-prompt-native8192-substrate-v2.json"
)
R0114_MANIFEST_SHA256 = (
    "0c32f75c115fec194c0833c6d946081c37d319581bb14e372a994d7d47e4044a"
)
R0114_IDENTITY_SHA256 = (
    "c77d96f1a1f0b80751127b22bbb353f85293f6a508deebabd68f3feb70b9f5a1"
)

NODE_SCHEMA = "round0116-canonical-document-embedding-node-v1"
CORPUS_SCHEMA = "jina-document-english-fineweb-rpj-5p727m-v1"
CAPABILITY = CORPUS_SCHEMA


class Round0116Error(RuntimeError):
    """The exact R0116 data-production contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {
        key: item
        for key, item in value.items()
        if key != "identity_sha256"
    }
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0116Error(f"{label} identity seal is invalid")


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
        raise Round0116Error("R0087 inventory manifest bytes changed")
    value = _read_json(path)
    body = {
        key: item
        for key, item in value.items()
        if key != "identity_sha256"
    }
    if (
        value.get("schema") != "jina-diverse-25m-inventory-v1"
        or value.get("identity_sha256") != INVENTORY_IDENTITY_SHA256
        or value.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or value.get("capability_ready") is not True
        or (value.get("selection") or {}).get("complete") is not True
        or int((value.get("selection") or {}).get("selected_rows", -1))
        != 25_000_000
    ):
        raise Round0116Error("R0087 inventory capability is not exact")
    return value, signature


def _parquet_row_count_and_column(path: str) -> tuple[int, str]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    try:
        field = schema.field("chunk_text")
    except KeyError as error:
        raise Round0116Error(
            f"canonical source parquet lacks chunk_text: {path}"
        ) from error
    if not (
        pa.types.is_string(field.type)
        or pa.types.is_large_string(field.type)
    ):
        raise Round0116Error(
            f"canonical source chunk_text is not UTF-8 text: {path}"
        )
    return int(parquet.metadata.num_rows), str(field.type)


def source_layout_from_inventory(
    inventory: Mapping[str, Any],
    *,
    text_roots: Mapping[str, str] = TEXT_ROOTS,
    signature_fn: Callable[[str], Mapping[str, Any]] = (
        expected_input_signature
    ),
    parquet_inspector: Callable[[str], tuple[int, str]] = (
        _parquet_row_count_and_column
    ),
) -> list[dict[str, Any]]:
    """Map accepted R0087 rows one-to-one onto canonical source parquets."""
    selection = inventory.get("selection") or {}
    ranges = list(selection.get("ranges") or [])
    result: list[dict[str, Any]] = []
    corpus_cursor = 0
    for dataset in (FINEWEB, REDPAJAMA):
        expected_rows = DATASET_ROWS[dataset]
        dataset_cursor = 0
        selected = [item for item in ranges if item.get("dataset") == dataset]
        if not selected:
            raise Round0116Error(f"R0087 has no selected rows for {dataset}")
        for item in selected:
            shard = dict(item.get("shard") or {})
            embedding_path = os.path.realpath(
                str(shard.get("canonical_path") or "")
            )
            name = os.path.basename(embedding_path)
            start = int(item.get("dataset_row_start", -1))
            stop = int(item.get("dataset_row_stop", -1))
            shard_start = int(item.get("shard_row_start", -1))
            shard_stop = int(item.get("shard_row_stop", -1))
            global_start = int(item.get("global_row_start", -1))
            global_stop = int(item.get("global_row_stop", -1))
            if (
                start != dataset_cursor
                or global_start != corpus_cursor
                or stop <= start
                or stop - start != shard_stop - shard_start
                or shard_start != 0
                or global_stop - global_start != stop - start
                or not name.endswith(".npy")
                or int(shard.get("rows", -1)) < shard_stop
                or int(shard.get("bytes", -1)) <= 0
                or len(str(shard.get("sha256") or "")) != 64
            ):
                raise Round0116Error(
                    f"R0087 canonical range is malformed for {dataset}"
                )
            text_path = os.path.realpath(
                os.path.join(text_roots[dataset], name[:-4] + ".parquet")
            )
            text_rows, text_type = parquet_inspector(text_path)
            if (
                text_rows != int(shard["rows"])
                or shard_stop > text_rows
            ):
                raise Round0116Error(
                    f"text/embedding row mapping changed for {name}"
                )
            text_signature = dict(signature_fn(text_path))
            if (
                text_signature.get("canonical_path") != text_path
                or int(text_signature.get("bytes", -1)) <= 0
                or len(str(text_signature.get("sha256") or "")) != 64
            ):
                raise Round0116Error(
                    f"source parquet signature is malformed for {name}"
                )
            result.append(
                {
                    "dataset": dataset,
                    "dataset_row_start": start,
                    "dataset_row_stop": stop,
                    "corpus_global_row_start": global_start,
                    "corpus_global_row_stop": global_stop,
                    "shard_row_start": shard_start,
                    "shard_row_stop": shard_stop,
                    "shard_rows": text_rows,
                    "text_column": "chunk_text",
                    "text_column_type": text_type,
                    "text": text_signature,
                    "accepted_raw_embedding": {
                        "kind": "file",
                        "canonical_path": embedding_path,
                        "bytes": int(shard["bytes"]),
                        "sha256": str(shard["sha256"]),
                        "rows": int(shard["rows"]),
                    },
                }
            )
            dataset_cursor = stop
            corpus_cursor = global_stop
        if dataset_cursor != expected_rows:
            raise Round0116Error(
                f"canonical {dataset} rows do not close at {expected_rows}"
            )
    if corpus_cursor != CORPUS_ROWS:
        raise Round0116Error("canonical English corpus rows do not close")
    return result


def canonical_source_layout() -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    inventory, signature = load_inventory_manifest()
    return source_layout_from_inventory(inventory), signature


def clip_layout(
    layout: Sequence[Mapping[str, Any]],
    *,
    dataset: str,
    start: int,
    stop: int,
) -> list[dict[str, Any]]:
    """Return the exact shard slices intersecting one dataset-row interval."""
    if (
        dataset not in DATASET_ROWS
        or not 0 <= start < stop <= DATASET_ROWS[dataset]
    ):
        raise Round0116Error("requested source interval is outside the corpus")
    result: list[dict[str, Any]] = []
    cursor = start
    for original in layout:
        if original.get("dataset") != dataset:
            continue
        source_start = int(original["dataset_row_start"])
        source_stop = int(original["dataset_row_stop"])
        if source_stop <= start:
            continue
        if source_start >= stop:
            break
        take_start = max(start, source_start)
        take_stop = min(stop, source_stop)
        if take_start != cursor:
            raise Round0116Error("source layout interval has a gap or overlap")
        offset = take_start - source_start
        item = dict(original)
        item["dataset_row_start"] = take_start
        item["dataset_row_stop"] = take_stop
        item["corpus_global_row_start"] = (
            DATASET_GLOBAL_OFFSETS[dataset] + take_start
        )
        item["corpus_global_row_stop"] = (
            DATASET_GLOBAL_OFFSETS[dataset] + take_stop
        )
        item["shard_row_start"] = int(original["shard_row_start"]) + offset
        item["shard_row_stop"] = (
            int(item["shard_row_start"]) + take_stop - take_start
        )
        result.append(item)
        cursor = take_stop
    if cursor != stop:
        raise Round0116Error("source layout interval did not close")
    return result


def expected_work_range(node_id: str) -> tuple[str, int, int]:
    matches = [
        (dataset, start, stop)
        for candidate, dataset, start, stop in WORK_RANGES
        if candidate == node_id
    ]
    if len(matches) != 1:
        raise Round0116Error(f"unknown R0116 work node {node_id!r}")
    return matches[0]


def load_reused_manifest(
    path: str = R0114_MANIFEST_PATH,
    *,
    expected_sha256: str = R0114_MANIFEST_SHA256,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0116Error("accepted R0114 manifest bytes changed")
    value = _read_json(path)
    validate_seal(value, label="accepted R0114 manifest")
    source = value.get("source_contract") or {}
    model = value.get("model") or {}
    document = (value.get("conventions") or {}).get("document") or {}
    chunks = list(document.get("chunks") or [])
    text_receipts = list(value.get("chunk_text_receipts") or [])
    expected_chunks = math.ceil(REUSED_FINEWEB_ROWS / CHUNK_ROWS)
    if (
        value.get("schema") != RECOVERY_SCHEMA
        or value.get("identity_sha256") != R0114_IDENTITY_SHA256
        or int(value.get("row_count", -1)) != REUSED_FINEWEB_ROWS
        or int(value.get("dimension", -1)) != DIMENSION
        or source.get("r0087_inventory_identity_sha256")
        != INVENTORY_IDENTITY_SHA256
        or source.get("source_dataset") != FINEWEB
        or source.get("source_global_rows")
        != [0, REUSED_FINEWEB_ROWS]
        or int(model.get("native_max_seq_length", -1))
        != NATIVE_MAX_SEQ_LENGTH
        or model.get("output_dtype") != OUTPUT_DTYPE.str
        or document.get("prompt_prefix") != PROMPT_PREFIX
        or document.get("prompt_applied") is not True
        or len(chunks) != expected_chunks
        or len(text_receipts) != expected_chunks
    ):
        raise Round0116Error("accepted R0114 reuse contract changed")
    cursor = 0
    for index, (chunk, text) in enumerate(
        zip(chunks, text_receipts, strict=True)
    ):
        start, stop = [int(item) for item in text["source_row_range"]]
        expected_stop = min(start + CHUNK_ROWS, REUSED_FINEWEB_ROWS)
        if (
            start != cursor
            or stop != expected_stop
            or int(chunk.get("bytes", -1))
            < (stop - start) * DIMENSION * OUTPUT_DTYPE.itemsize
            or len(str(chunk.get("sha256") or "")) != 64
            or len(str(text.get("source_text_ordered_sha256") or "")) != 64
            or len(str(text.get("document_text_ordered_sha256") or "")) != 64
        ):
            raise Round0116Error(
                f"accepted R0114 chunk {index} mapping is malformed"
            )
        cursor = stop
    if cursor != REUSED_FINEWEB_ROWS:
        raise Round0116Error("accepted R0114 prefix coverage did not close")
    return value, signature


def validate_reused_mapping(
    layout: Sequence[Mapping[str, Any]],
    reused: Mapping[str, Any],
    *,
    r0114_source_lineage: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind the reviewed first 2M substrate into canonical FineWeb order."""
    prefix = clip_layout(
        layout,
        dataset=FINEWEB,
        start=0,
        stop=REUSED_FINEWEB_ROWS,
    )
    source = reused.get("source_contract") or {}
    if (
        source.get("r0087_inventory_identity_sha256")
        != INVENTORY_IDENTITY_SHA256
        or source.get("source_dataset") != FINEWEB
        or source.get("source_global_rows")
        != [0, REUSED_FINEWEB_ROWS]
        or prefix[0]["dataset_row_start"] != 0
        or prefix[-1]["dataset_row_stop"] != REUSED_FINEWEB_ROWS
    ):
        raise Round0116Error(
            "R0114 rows do not map into the canonical FineWeb prefix"
        )
    proof = {
        "dataset": FINEWEB,
        "dataset_row_range": [0, REUSED_FINEWEB_ROWS],
        "corpus_global_row_range": [0, REUSED_FINEWEB_ROWS],
        "canonical_source_slices": prefix,
        "r0087_inventory_identity_sha256": INVENTORY_IDENTITY_SHA256,
        "r0114_manifest_identity_sha256": R0114_IDENTITY_SHA256,
        "mapping_exact": True,
    }
    if r0114_source_lineage is not None:
        lineage = [dict(item) for item in r0114_source_lineage]
        if len(lineage) != len(prefix):
            raise Round0116Error(
                "R0114 source lineage has a different shard count"
            )
        for canonical, recovered in zip(prefix, lineage, strict=True):
            embedding = recovered.get("embedding") or {}
            if (
                (
                    int(canonical["dataset_row_start"]),
                    int(canonical["dataset_row_stop"]),
                )
                != (
                    int(recovered.get("global_row_start", -1)),
                    int(recovered.get("global_row_stop", -1)),
                )
                or (
                    int(canonical["shard_row_start"]),
                    int(canonical["shard_row_stop"]),
                )
                != (
                    int(recovered.get("shard_row_start", -1)),
                    int(recovered.get("shard_row_stop", -1)),
                )
                or canonical["text"]["canonical_path"]
                != os.path.realpath(str(recovered.get("text_path") or ""))
                or canonical["accepted_raw_embedding"]["canonical_path"]
                != os.path.realpath(
                    str(embedding.get("canonical_path") or "")
                )
                or canonical["accepted_raw_embedding"]["sha256"]
                != embedding.get("sha256")
                or canonical["accepted_raw_embedding"]["bytes"]
                != int(embedding.get("bytes", -1))
            ):
                raise Round0116Error(
                    "R0114 source lineage does not equal the canonical "
                    "FineWeb prefix"
                )
        proof["r0114_source_lineage_slices"] = len(lineage)
        proof["r0114_source_lineage_matches_canonical_prefix"] = True
    return proof


def validate_node_receipt(
    value: Mapping[str, Any],
    *,
    node_id: str,
) -> dict[str, Any]:
    receipt = dict(value)
    validate_seal(receipt, label=f"R0116 node {node_id}")
    dataset, expected_start, expected_stop = expected_work_range(node_id)
    chunks = list(receipt.get("chunks") or [])
    source_layout = list(receipt.get("source_layout") or [])
    rehashed_sources = list(
        receipt.get("source_files_rehashed_at_node_boundary") or []
    )
    model = receipt.get("model") or {}
    runtime = model.get("runtime_semantics") or {}
    expected_chunk_count = math.ceil(
        (expected_stop - expected_start) / CHUNK_ROWS
    )
    if (
        receipt.get("schema") != NODE_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("node_id") != node_id
        or receipt.get("dataset") != dataset
        or receipt.get("dataset_row_range")
        != [expected_start, expected_stop]
        or receipt.get("corpus_global_row_range")
        != [
            DATASET_GLOBAL_OFFSETS[dataset] + expected_start,
            DATASET_GLOBAL_OFFSETS[dataset] + expected_stop,
        ]
        or int(receipt.get("dimension", -1)) != DIMENSION
        or receipt.get("compute_dtype") != COMPUTE_DTYPE
        or receipt.get("output_dtype") != OUTPUT_DTYPE.str
        or receipt.get("prompt_prefix") != PROMPT_PREFIX
        or receipt.get("prompt_name_equivalence_passed") is not True
        or model.get("id") != ROUND0005_MODEL_ID
        or model.get("revision") != ROUND0005_MODEL_REVISION
        or model.get("root") != MODEL_ROOT
        or int(model.get("native_max_seq_length", -1))
        != NATIVE_MAX_SEQ_LENGTH
        or model.get("pooling") != "lasttoken"
        or model.get("normalization") != "l2"
        or runtime.get(
            "resolved_sentence_transformers_max_seq_length"
        )
        != NATIVE_MAX_SEQ_LENGTH
        or receipt.get("training_performed") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
        or len(chunks) != expected_chunk_count
        or not source_layout
        or int(source_layout[0].get("dataset_row_start", -1))
        != expected_start
        or int(source_layout[-1].get("dataset_row_stop", -1))
        != expected_stop
        or any(item.get("dataset") != dataset for item in source_layout)
    ):
        raise Round0116Error(f"R0116 node {node_id} contract changed")
    source_cursor = expected_start
    expected_sources: dict[str, dict[str, Any]] = {}
    for item in source_layout:
        source_start = int(item.get("dataset_row_start", -1))
        source_stop = int(item.get("dataset_row_stop", -1))
        text = dict(item.get("text") or {})
        path = str(text.get("canonical_path") or "")
        if (
            source_start != source_cursor
            or source_stop <= source_start
            or not path
            or len(str(text.get("sha256") or "")) != 64
        ):
            raise Round0116Error(
                f"R0116 node {node_id} source coverage changed"
            )
        prior = expected_sources.setdefault(path, text)
        if prior != text:
            raise Round0116Error(
                f"R0116 node {node_id} source signatures conflict"
            )
        source_cursor = source_stop
    if (
        source_cursor != expected_stop
        or rehashed_sources
        != [expected_sources[path] for path in sorted(expected_sources)]
    ):
        raise Round0116Error(
            f"R0116 node {node_id} source-boundary proof changed"
        )
    cursor = expected_start
    for chunk in chunks:
        start, stop = [int(item) for item in chunk["dataset_row_range"]]
        global_start, global_stop = [
            int(item) for item in chunk["corpus_global_row_range"]
        ]
        rows = stop - start
        output = chunk.get("output") or {}
        if (
            start != cursor
            or not 0 < rows <= CHUNK_ROWS
            or global_start
            != DATASET_GLOBAL_OFFSETS[dataset] + start
            or global_stop - global_start != rows
            or chunk.get("output_shape") != [rows, DIMENSION]
            or chunk.get("output_dtype") != OUTPUT_DTYPE.str
            or int(output.get("bytes", -1))
            < rows * DIMENSION * OUTPUT_DTYPE.itemsize
            or int(output.get("bytes", -1))
            > rows * DIMENSION * OUTPUT_DTYPE.itemsize + 4_096
            or len(str(output.get("sha256") or "")) != 64
            or len(str(chunk.get("source_text_ordered_sha256") or ""))
            != 64
            or len(str(chunk.get("document_text_ordered_sha256") or ""))
            != 64
            or (chunk.get("stored_norm") or {}).get("passed") is not True
        ):
            raise Round0116Error(
                f"R0116 node {node_id} chunk coverage changed"
            )
        cursor = stop
    if cursor != expected_stop:
        raise Round0116Error(f"R0116 node {node_id} did not close")
    performance = receipt.get("performance") or {}
    if (
        not math.isfinite(float(performance.get("wall_s", math.nan)))
        or float(performance.get("wall_s", 0.0)) <= 0.0
        or not math.isfinite(
            float(performance.get("document_rows_per_s", math.nan))
        )
        or float(performance.get("document_rows_per_s", 0.0)) <= 0.0
        or int(performance.get("oom_retries", -1)) < 0
    ):
        raise Round0116Error(f"R0116 node {node_id} telemetry is invalid")
    return receipt


def validate_coverage(chunks: Sequence[Mapping[str, Any]]) -> None:
    """Require one gap-free, overlap-free row cover over both source pools."""
    by_dataset = {
        FINEWEB: [],
        REDPAJAMA: [],
    }
    output_paths: set[str] = set()
    for chunk in chunks:
        dataset = str(chunk.get("dataset") or "")
        if dataset not in by_dataset:
            raise Round0116Error("combined corpus contains an unknown dataset")
        output_path = str(
            (chunk.get("output") or {}).get("canonical_path") or ""
        )
        if not output_path or output_path in output_paths:
            raise Round0116Error(
                "combined corpus contains a missing or repeated output chunk"
            )
        output_paths.add(output_path)
        by_dataset[dataset].append(chunk)
    corpus_cursor = 0
    for dataset in (FINEWEB, REDPAJAMA):
        cursor = 0
        ordered = sorted(
            by_dataset[dataset],
            key=lambda item: int(item["dataset_row_range"][0]),
        )
        for chunk in ordered:
            start, stop = [
                int(item) for item in chunk["dataset_row_range"]
            ]
            global_start, global_stop = [
                int(item) for item in chunk["corpus_global_row_range"]
            ]
            if (
                start != cursor
                or stop <= start
                or stop - start > CHUNK_ROWS
                or global_start != DATASET_GLOBAL_OFFSETS[dataset] + start
                or global_stop != DATASET_GLOBAL_OFFSETS[dataset] + stop
                or global_start != corpus_cursor
            ):
                raise Round0116Error(
                    "combined corpus has a gap, overlap, or row-order change"
                )
            cursor = stop
            corpus_cursor = global_stop
        if cursor != DATASET_ROWS[dataset]:
            raise Round0116Error(
                f"combined corpus does not cover all {dataset} rows"
            )
    if corpus_cursor != CORPUS_ROWS:
        raise Round0116Error("combined canonical corpus does not close")


def production_payload_bytes() -> int:
    return NEW_ROWS * DIMENSION * OUTPUT_DTYPE.itemsize


def required_free_bytes() -> int:
    # Payload plus one atomic chunk, filesystem/receipt overhead, and a fixed
    # 2 GiB safety reserve.  Existing R0114 prefix bytes are referenced, not
    # copied.
    payload = production_payload_bytes()
    return int(math.ceil(payload * 1.25)) + 2 * 1024**3


def source_manifest_summary(
    layout: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset in (FINEWEB, REDPAJAMA):
        selected = [item for item in layout if item["dataset"] == dataset]
        result[dataset] = {
            "rows": DATASET_ROWS[dataset],
            "source_shards": len(selected),
            "source_parquet_bytes": sum(
                int(item["text"]["bytes"]) for item in selected
            ),
            "source_parquet_sha256s": [
                item["text"]["sha256"] for item in selected
            ],
            "accepted_raw_embedding_sha256s": [
                item["accepted_raw_embedding"]["sha256"]
                for item in selected
            ],
        }
    return result


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
    """Hash the exact package freeze used to prepare or execute a node.

    The run checkout owns a dedicated read-only virtualenv.  Recomputing this
    small receipt in every child still detects an accidental package mutation
    between independently queued embedding nodes.
    """
    packages = []
    for distribution in metadata.distributions():
        name = str(distribution.metadata.get("Name") or "").strip()
        version = str(distribution.version or "").strip()
        if not name or not version:
            raise Round0116Error(
                "installed distribution lacks a stable name or version"
            )
        direct_url = distribution.read_text("direct_url.json")
        record = distribution.read_text("RECORD")
        packages.append(
            {
                "name": name.lower().replace("_", "-"),
                "version": version,
                "direct_url_sha256": (
                    sha256_bytes(direct_url.encode("utf-8"))
                    if direct_url is not None else None
                ),
                "record_sha256": (
                    sha256_bytes(record.encode("utf-8"))
                    if record is not None else None
                ),
            }
        )
    packages.sort(
        key=lambda item: (
            item["name"],
            item["version"],
            str(item["direct_url_sha256"]),
            str(item["record_sha256"]),
        )
    )
    body = {
        "schema": "round0116-python-environment-freeze-v1",
        "python_executable": os.path.abspath(sys.executable),
        "python_prefix": os.path.abspath(sys.prefix),
        "python_version": platform.python_version(),
        "packages": packages,
    }
    return {
        **body,
        "freeze_sha256": sha256_bytes(canonical_json(body)),
    }


def validate_environment_freeze(
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    observed = environment_freeze_receipt()
    if observed != dict(expected):
        raise Round0116Error(
            "R0116 execution environment changed after queue preparation"
        )
    return observed
