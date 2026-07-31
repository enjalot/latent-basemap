"""Canonical literal-``Document: `` Jina embeddings for R0087's Pile pool.

R0120 is a data-production follow-on to R0116.  It embeds exactly the
3,399,036 Pile rows selected by the accepted R0087 inventory, using the model
and prompt closure accepted in R0114.  It does not build a graph or train,
evaluate, or promote a map.
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
from .round0087_inventory import FINEWEB, PILE, REDPAJAMA
from .round0112_prompt_substrate import MODEL_ROOT, OUTPUT_DTYPE, PROMPT_PREFIX
from .round0114_prompt_recovery import NATIVE_MAX_SEQ_LENGTH
from .round0116_prompted_corpus import (
    environment_freeze_receipt as _environment_freeze_receipt,
    load_reused_manifest,
    validate_environment_freeze as _validate_environment_freeze,
)


ROUND_ID = "0120"
DIMENSION = ROUND0005_DIMENSIONS
CHUNK_ROWS = 25_000
BATCH_SIZE = 256
COMPUTE_DTYPE = "float32"
# At this floor, all four nodes plus their registered fixed overhead remain
# inside the 6.5 GPU-hour queue cap.  A lower passing floor could let a slow
# run finish over budget because the slim runner enforces the cap at node
# boundaries.
EMBED_MINIMUM_ROWS_PER_S = 160.0
EMBED_WARNING_ROWS_PER_S = 180.0
PERFORMANCE_GUARD_ROWS = 50_000

DATASET = PILE
CORPUS_ROWS = 3_399_036
R0087_PILE_GLOBAL_OFFSET = 2_890_362 + 2_836_978
R0087_PILE_GLOBAL_STOP = R0087_PILE_GLOBAL_OFFSET + CORPUS_ROWS
TEXT_ROOT = "/data/chunks/pile-uncopyrighted-chunked-500/train"

# Four equal-size independently retryable nodes.  Each is about 56 minutes of
# encoding at the accepted 253 rows/s receipt rate, before fixed overhead.
WORK_RANGES = (
    ("embed_pile_00", 0, 850_000),
    ("embed_pile_01", 850_000, 1_700_000),
    ("embed_pile_02", 1_700_000, 2_550_000),
    ("embed_pile_03", 2_550_000, CORPUS_ROWS),
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

NODE_SCHEMA = "round0120-canonical-document-pile-node-v1"
CORPUS_SCHEMA = "jina-document-pile-english-3p399m-v1"
CAPABILITY = CORPUS_SCHEMA


class Round0120Error(RuntimeError):
    """The exact R0120 data-production contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {
        key: item for key, item in value.items() if key != "identity_sha256"
    }
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0120Error(f"{label} identity seal is invalid")


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
        raise Round0120Error("R0087 inventory manifest bytes changed")
    value = _read_json(path)
    body = {
        key: item for key, item in value.items() if key != "identity_sha256"
    }
    selection = value.get("selection") or {}
    budgets = selection.get("budgets") or {}
    source_order = list(selection.get("source_order") or [])
    if (
        value.get("schema") != "jina-diverse-25m-inventory-v1"
        or value.get("identity_sha256") != INVENTORY_IDENTITY_SHA256
        or value.get("identity_sha256") != sha256_bytes(canonical_json(body))
        or value.get("capability_ready") is not True
        or selection.get("complete") is not True
        or int(selection.get("selected_rows", -1)) != 25_000_000
        or int(budgets.get(DATASET, -1)) != CORPUS_ROWS
        or source_order[:3] != [FINEWEB, REDPAJAMA, PILE]
    ):
        raise Round0120Error("R0087 Pile inventory capability is not exact")
    return value, signature


def load_r0114_model_prompt_closure(
    path: str = R0114_MANIFEST_PATH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the accepted model/prompt closure without reusing its rows."""
    value, signature = load_reused_manifest(
        path, expected_sha256=R0114_MANIFEST_SHA256
    )
    model = value.get("model") or {}
    document = (value.get("conventions") or {}).get("document") or {}
    if (
        int(model.get("native_max_seq_length", -1))
        != NATIVE_MAX_SEQ_LENGTH
        or model.get("output_dtype") != OUTPUT_DTYPE.str
        or document.get("prompt_prefix") != PROMPT_PREFIX
        or document.get("prompt_applied") is not True
    ):
        raise Round0120Error("R0114 model/prompt closure changed")
    return value, signature


def _parquet_row_count_and_column(path: str) -> tuple[int, str]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    try:
        field = schema.field("chunk_text")
    except KeyError as error:
        raise Round0120Error(
            f"canonical source parquet lacks chunk_text: {path}"
        ) from error
    if not (
        pa.types.is_string(field.type) or pa.types.is_large_string(field.type)
    ):
        raise Round0120Error(
            f"canonical source chunk_text is not UTF-8 text: {path}"
        )
    return int(parquet.metadata.num_rows), str(field.type)


def source_layout_from_inventory(
    inventory: Mapping[str, Any],
    *,
    text_root: str = TEXT_ROOT,
    signature_fn: Callable[[str], Mapping[str, Any]] = (
        expected_input_signature
    ),
    parquet_inspector: Callable[[str], tuple[int, str]] = (
        _parquet_row_count_and_column
    ),
) -> list[dict[str, Any]]:
    """Map R0087's exact Pile row range onto authenticated source text."""
    ranges = [
        item
        for item in ((inventory.get("selection") or {}).get("ranges") or [])
        if item.get("dataset") == DATASET
    ]
    if not ranges:
        raise Round0120Error("R0087 has no selected Pile rows")
    result: list[dict[str, Any]] = []
    dataset_cursor = 0
    accepted_global_cursor = R0087_PILE_GLOBAL_OFFSET
    for item in ranges:
        shard = dict(item.get("shard") or {})
        embedding_path = os.path.realpath(
            str(shard.get("canonical_path") or "")
        )
        name = os.path.basename(embedding_path)
        start = int(item.get("dataset_row_start", -1))
        stop = int(item.get("dataset_row_stop", -1))
        shard_start = int(item.get("shard_row_start", -1))
        shard_stop = int(item.get("shard_row_stop", -1))
        accepted_start = int(item.get("global_row_start", -1))
        accepted_stop = int(item.get("global_row_stop", -1))
        if (
            start != dataset_cursor
            or accepted_start != accepted_global_cursor
            or stop <= start
            or stop - start != shard_stop - shard_start
            or shard_start != 0
            or accepted_stop - accepted_start != stop - start
            or not name.endswith(".npy")
            or int(shard.get("rows", -1)) < shard_stop
            or int(shard.get("bytes", -1)) <= 0
            or len(str(shard.get("sha256") or "")) != 64
        ):
            raise Round0120Error("R0087 canonical Pile range is malformed")
        text_path = os.path.realpath(
            os.path.join(text_root, name[:-4] + ".parquet")
        )
        text_rows, text_type = parquet_inspector(text_path)
        if text_rows != int(shard["rows"]) or shard_stop > text_rows:
            raise Round0120Error(
                f"text/embedding row mapping changed for {name}"
            )
        text_signature = dict(signature_fn(text_path))
        if (
            text_signature.get("canonical_path") != text_path
            or int(text_signature.get("bytes", -1)) <= 0
            or len(str(text_signature.get("sha256") or "")) != 64
        ):
            raise Round0120Error(
                f"source parquet signature is malformed for {name}"
            )
        result.append(
            {
                "dataset": DATASET,
                "dataset_row_start": start,
                "dataset_row_stop": stop,
                "corpus_global_row_start": start,
                "corpus_global_row_stop": stop,
                "r0087_global_row_start": accepted_start,
                "r0087_global_row_stop": accepted_stop,
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
        accepted_global_cursor = accepted_stop
    if (
        dataset_cursor != CORPUS_ROWS
        or accepted_global_cursor != R0087_PILE_GLOBAL_STOP
    ):
        raise Round0120Error("canonical R0087 Pile rows do not close")
    return result


def canonical_source_layout() -> tuple[
    list[dict[str, Any]], dict[str, Any]
]:
    inventory, signature = load_inventory_manifest()
    return source_layout_from_inventory(inventory), signature


def clip_layout(
    layout: Sequence[Mapping[str, Any]],
    *,
    start: int,
    stop: int,
) -> list[dict[str, Any]]:
    if not 0 <= start < stop <= CORPUS_ROWS:
        raise Round0120Error("requested source interval is outside Pile")
    result: list[dict[str, Any]] = []
    cursor = start
    for original in layout:
        source_start = int(original["dataset_row_start"])
        source_stop = int(original["dataset_row_stop"])
        if source_stop <= start:
            continue
        if source_start >= stop:
            break
        take_start = max(start, source_start)
        take_stop = min(stop, source_stop)
        if take_start != cursor:
            raise Round0120Error("source layout interval has a gap or overlap")
        offset = take_start - source_start
        item = dict(original)
        item["dataset_row_start"] = take_start
        item["dataset_row_stop"] = take_stop
        item["corpus_global_row_start"] = take_start
        item["corpus_global_row_stop"] = take_stop
        item["r0087_global_row_start"] = (
            int(original["r0087_global_row_start"]) + offset
        )
        item["r0087_global_row_stop"] = (
            int(item["r0087_global_row_start"]) + take_stop - take_start
        )
        item["shard_row_start"] = int(original["shard_row_start"]) + offset
        item["shard_row_stop"] = (
            int(item["shard_row_start"]) + take_stop - take_start
        )
        result.append(item)
        cursor = take_stop
    if cursor != stop:
        raise Round0120Error("source layout interval did not close")
    return result


def expected_work_range(node_id: str) -> tuple[int, int]:
    matches = [
        (start, stop)
        for candidate, start, stop in WORK_RANGES
        if candidate == node_id
    ]
    if len(matches) != 1:
        raise Round0120Error(f"unknown R0120 work node {node_id!r}")
    return matches[0]


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
    """Use the exact R0116 environment closure shared by this follow-on."""
    return _environment_freeze_receipt()


def validate_environment_freeze(
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Recheck the shared environment closure at every job boundary."""
    return _validate_environment_freeze(expected)


def validate_node_receipt(
    value: Mapping[str, Any], *, node_id: str
) -> dict[str, Any]:
    receipt = dict(value)
    validate_seal(receipt, label=f"R0120 node {node_id}")
    expected_start, expected_stop = expected_work_range(node_id)
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
        or receipt.get("dataset") != DATASET
        or receipt.get("dataset_row_range")
        != [expected_start, expected_stop]
        or receipt.get("corpus_global_row_range")
        != [expected_start, expected_stop]
        or receipt.get("r0087_global_row_range")
        != [
            R0087_PILE_GLOBAL_OFFSET + expected_start,
            R0087_PILE_GLOBAL_OFFSET + expected_stop,
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
        or runtime.get("resolved_sentence_transformers_max_seq_length")
        != NATIVE_MAX_SEQ_LENGTH
        or receipt.get("training_performed") is not False
        or int(receipt.get("optimizer_updates", -1)) != 0
        or len(chunks) != expected_chunk_count
        or not source_layout
        or int(source_layout[0].get("dataset_row_start", -1))
        != expected_start
        or int(source_layout[-1].get("dataset_row_stop", -1))
        != expected_stop
        or any(item.get("dataset") != DATASET for item in source_layout)
    ):
        raise Round0120Error(f"R0120 node {node_id} contract changed")
    expected_sources: dict[str, dict[str, Any]] = {}
    cursor = expected_start
    for item in source_layout:
        start = int(item.get("dataset_row_start", -1))
        stop = int(item.get("dataset_row_stop", -1))
        text = dict(item.get("text") or {})
        accepted = dict(item.get("accepted_raw_embedding") or {})
        path = str(text.get("canonical_path") or "")
        if (
            start != cursor
            or stop <= start
            or int(item.get("corpus_global_row_start", -1)) != start
            or int(item.get("corpus_global_row_stop", -1)) != stop
            or int(item.get("r0087_global_row_start", -1))
            != R0087_PILE_GLOBAL_OFFSET + start
            or int(item.get("r0087_global_row_stop", -1))
            != R0087_PILE_GLOBAL_OFFSET + stop
            or not path
            or len(str(text.get("sha256") or "")) != 64
            or len(str(accepted.get("sha256") or "")) != 64
        ):
            raise Round0120Error(
                f"R0120 node {node_id} source coverage changed"
            )
        prior = expected_sources.setdefault(path, text)
        if prior != text:
            raise Round0120Error(
                f"R0120 node {node_id} source signatures conflict"
            )
        cursor = stop
    if (
        cursor != expected_stop
        or rehashed_sources
        != [expected_sources[path] for path in sorted(expected_sources)]
    ):
        raise Round0120Error(
            f"R0120 node {node_id} source-boundary proof changed"
        )
    cursor = expected_start
    for chunk in chunks:
        start, stop = [int(item) for item in chunk["dataset_row_range"]]
        local_start, local_stop = [
            int(item) for item in chunk["corpus_global_row_range"]
        ]
        accepted_start, accepted_stop = [
            int(item) for item in chunk["r0087_global_row_range"]
        ]
        rows = stop - start
        output = chunk.get("output") or {}
        if (
            chunk.get("dataset") != DATASET
            or start != cursor
            or not 0 < rows <= CHUNK_ROWS
            or (local_start, local_stop) != (start, stop)
            or accepted_start != R0087_PILE_GLOBAL_OFFSET + start
            or accepted_stop != R0087_PILE_GLOBAL_OFFSET + stop
            or int(chunk.get("source_row_count", -1)) != rows
            or chunk.get("source_ids_ordered_sha256")
            != ordered_array_sha256(
                np.arange(start, stop, dtype=np.int64)
            )
            or chunk.get("output_shape") != [rows, DIMENSION]
            or chunk.get("output_dtype") != OUTPUT_DTYPE.str
            or output.get("kind") != "file"
            or not str(output.get("canonical_path") or "")
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
            raise Round0120Error(
                f"R0120 node {node_id} chunk coverage changed"
            )
        cursor = stop
    if cursor != expected_stop:
        raise Round0120Error(f"R0120 node {node_id} did not close")
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
        raise Round0120Error(f"R0120 node {node_id} telemetry is invalid")
    return receipt


def validate_coverage(chunks: Sequence[Mapping[str, Any]]) -> None:
    output_paths: set[str] = set()
    cursor = 0
    for chunk in sorted(
        chunks, key=lambda item: int(item["dataset_row_range"][0])
    ):
        start, stop = [int(item) for item in chunk["dataset_row_range"]]
        local_start, local_stop = [
            int(item) for item in chunk["corpus_global_row_range"]
        ]
        accepted_start, accepted_stop = [
            int(item) for item in chunk["r0087_global_row_range"]
        ]
        output_path = str(
            (chunk.get("output") or {}).get("canonical_path") or ""
        )
        if (
            chunk.get("dataset") != DATASET
            or start != cursor
            or stop <= start
            or stop - start > CHUNK_ROWS
            or (local_start, local_stop) != (start, stop)
            or accepted_start != R0087_PILE_GLOBAL_OFFSET + start
            or accepted_stop != R0087_PILE_GLOBAL_OFFSET + stop
            or not output_path
            or output_path in output_paths
        ):
            raise Round0120Error(
                "Pile corpus has a gap, overlap, repeated output, or "
                "row-order change"
            )
        output_paths.add(output_path)
        cursor = stop
    if cursor != CORPUS_ROWS:
        raise Round0120Error("canonical Pile corpus does not close")


def production_payload_bytes() -> int:
    return CORPUS_ROWS * DIMENSION * OUTPUT_DTYPE.itemsize


def required_free_bytes() -> int:
    payload = production_payload_bytes()
    return int(math.ceil(payload * 1.25)) + 2 * 1024**3


def source_manifest_summary(
    layout: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "dataset": DATASET,
        "rows": CORPUS_ROWS,
        "r0087_global_row_range": [
            R0087_PILE_GLOBAL_OFFSET,
            R0087_PILE_GLOBAL_STOP,
        ],
        "source_shards": len(layout),
        "source_parquet_bytes": sum(
            int(item["text"]["bytes"]) for item in layout
        ),
        "source_parquet_sha256s": [
            item["text"]["sha256"] for item in layout
        ],
        "accepted_raw_embedding_sha256s": [
            item["accepted_raw_embedding"]["sha256"] for item in layout
        ],
    }
