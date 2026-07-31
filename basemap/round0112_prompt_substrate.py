"""Paired raw/document Jina-v5 embedding substrate for Round 0112.

R0112 removes the implementation confound from the historical prompt
comparison.  Both arms are freshly embedded from the exact same ordered first
two million FineWeb texts, through one pinned local SentenceTransformer
pipeline.  The only arm-level difference is the literal ``"Document: "``
prefix.

The round intentionally stops before graph construction or map training.  Its
fp16 shards are a durable input for the paired map contrast and a reusable
first tranche for future prompted-SAE corpus work.
"""
from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from .round0005_staging import (
    ROUND0005_DIMENSIONS,
    ROUND0005_JINA_MODEL_CLOSURE,
    ROUND0005_MODEL_ID,
    ROUND0005_MODEL_REVISION,
)
from .round0104_training import (
    R0087_INVENTORY_IDENTITY_SHA256,
    SOURCE_FIRST2M_PAYLOAD_SHA256,
    source_segments,
)


ROUND_ID = "0112"
ROWS = 2_000_000
DIMENSION = ROUND0005_DIMENSIONS
SLICE_ROWS = 500_000
CHUNK_ROWS = 25_000
SLICE_COUNT = ROWS // SLICE_ROWS
BATCH_SIZE = 256
EMBED_MINIMUM_PAIRED_ROWS_PER_S = 110.0
EMBED_WARNING_PAIRED_ROWS_PER_S = 130.0
PROMPT_PREFIX = "Document: "
CONVENTIONS = ("raw", "document")
COMPUTE_DTYPE = "float32"
OUTPUT_DTYPE = np.dtype("<f2")

TEXT_ROOT = "/data/chunks/fineweb-edu-sample-10BT-chunked-500/train"
EMBEDDING_ROOT = (
    "/data/embeddings/"
    "fineweb-edu-sample-10BT-chunked-500-jina-v5-nano/train"
)
MODEL_ROOT = (
    "/data/hf/hub/models--jinaai--"
    "jina-embeddings-v5-text-nano-retrieval/snapshots/"
    f"{ROUND0005_MODEL_REVISION}"
)
ELIGIBILITY_PATH = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-eligibility-v1.npz"
)
ELIGIBILITY_SHA256 = (
    "11a9c197f0e20cb1e5d6968bc9ec3a9e2c89fa66c711d252f864959016eac274"
)

HISTORICAL_RAW_MEAN_COSINE_FLOOR = 0.98
HISTORICAL_RAW_MIN_COSINE_FLOOR = 0.95
SLICE_SCHEMA = "round0112-paired-jina-embedding-slice-v1"
SUBSTRATE_SCHEMA = "jina-fineweb-2m-dual-prompt-embedding-substrate-v1"


class Round0112Error(RuntimeError):
    """The registered R0112 prompt-substrate contract was violated."""


def seal(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "identity_sha256": sha256_bytes(canonical_json(value))}


def validate_seal(value: Mapping[str, Any], *, label: str) -> None:
    body = {key: item for key, item in value.items() if key != "identity_sha256"}
    if value.get("identity_sha256") != sha256_bytes(canonical_json(body)):
        raise Round0112Error(f"{label} identity seal is invalid")


def ordered_text_sha256(texts: Sequence[str]) -> str:
    """Hash UTF-8 texts in order with an unambiguous length prefix."""
    digest = hashlib.sha256()
    for text in texts:
        if not isinstance(text, str):
            raise Round0112Error("source text is not a string")
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def build_offsets(sizes: Sequence[int]) -> np.ndarray:
    values = np.asarray(sizes, dtype=np.int64)
    if (
        values.ndim != 1
        or not len(values)
        or np.any(values <= 0)
    ):
        raise Round0112Error("source shard sizes are malformed")
    return np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(values, dtype=np.int64))
    )


def locate_rows(
    rows: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    requested = np.asarray(rows, dtype=np.int64)
    boundaries = np.asarray(offsets, dtype=np.int64)
    if (
        requested.ndim != 1
        or boundaries.ndim != 1
        or len(boundaries) < 2
        or boundaries[0] != 0
        or np.any(boundaries[1:] <= boundaries[:-1])
        or (
            len(requested)
            and (requested[0] < 0 or requested[-1] >= boundaries[-1])
        )
        or (len(requested) > 1 and np.any(requested[1:] <= requested[:-1]))
    ):
        raise Round0112Error("ordered source rows or shard offsets are malformed")
    shard = np.searchsorted(boundaries, requested, side="right") - 1
    return shard, requested - boundaries[shard]


def expected_slice_ranges() -> list[tuple[int, int]]:
    return [
        (start, min(start + SLICE_ROWS, ROWS))
        for start in range(0, ROWS, SLICE_ROWS)
    ]


def first2m_layout() -> list[dict[str, Any]]:
    """Resolve and verify the exact embedding/text shard alignment.

    R0104 authenticates the fp16 embedding ordering.  Here each corresponding
    text parquet is independently checked for the same full-shard row count;
    the final selected shard may be consumed only through its registered
    prefix.
    """
    import pyarrow.parquet as pq

    segments = source_segments(0, ROWS)
    layout: list[dict[str, Any]] = []
    cursor = 0
    for item in segments:
        start = int(item["global_row_start"])
        stop = int(item["global_row_stop"])
        shard = dict(item["shard"])
        embedding_path = os.path.realpath(str(shard["canonical_path"]))
        if (
            start != cursor
            or stop <= start
            or item.get("dataset")
            != "fineweb-edu-sample-10BT-chunked-500-jina-v5-nano"
            or int(item["shard_row_start"]) != 0
        ):
            raise Round0112Error("R0104 first-2M FineWeb layout changed")
        name = os.path.basename(embedding_path)
        if not name.endswith(".npy"):
            raise Round0112Error("R0104 source shard is not an NPY embedding")
        text_path = os.path.realpath(
            os.path.join(TEXT_ROOT, name[:-4] + ".parquet")
        )
        if not os.path.isfile(text_path):
            raise Round0112Error(f"aligned text shard is missing: {text_path}")
        text_rows = int(pq.ParquetFile(text_path).metadata.num_rows)
        if text_rows != int(item["shard_rows"]):
            raise Round0112Error(
                f"text/embedding row-count mismatch for {name}: "
                f"{text_rows} != {item['shard_rows']}"
            )
        layout.append(
            {
                "global_row_start": start,
                "global_row_stop": stop,
                "shard_row_start": 0,
                "shard_row_stop": stop - start,
                "shard_rows": text_rows,
                "embedding": {
                    "canonical_path": embedding_path,
                    "kind": "file",
                    "bytes": int(shard["bytes"]),
                    "sha256": str(shard["sha256"]),
                },
                "text_path": text_path,
            }
        )
        cursor = stop
    if cursor != ROWS:
        raise Round0112Error("first-2M FineWeb source ranges do not close")
    return layout


def model_member_signatures() -> list[dict[str, Any]]:
    """Authenticate the complete pinned local model closure."""
    if not os.path.isdir(MODEL_ROOT):
        raise Round0112Error(f"pinned local model snapshot is missing: {MODEL_ROOT}")
    signatures: list[dict[str, Any]] = []
    for relative, (size, sha256) in sorted(
        ROUND0005_JINA_MODEL_CLOSURE.items()
    ):
        requested = os.path.join(MODEL_ROOT, relative)
        if not os.path.isfile(requested):
            raise Round0112Error(f"pinned model member is missing: {relative}")
        signature = expected_input_signature(os.path.realpath(requested))
        if (
            signature["bytes"] != int(size)
            or signature["sha256"] != str(sha256)
        ):
            raise Round0112Error(f"pinned model member changed: {relative}")
        signatures.append({**signature, "model_relative_path": relative})
    return signatures


def validate_slice_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_start: int,
    expected_stop: int,
) -> dict[str, Any]:
    value = dict(receipt)
    validate_seal(value, label="R0112 slice receipt")
    chunks = list(value.get("chunks") or [])
    if (
        value.get("schema") != SLICE_SCHEMA
        or value.get("round_id") != ROUND_ID
        or value.get("model_id") != ROUND0005_MODEL_ID
        or value.get("model_revision") != ROUND0005_MODEL_REVISION
        or value.get("prompt_prefix") != PROMPT_PREFIX
        or value.get("conventions") != list(CONVENTIONS)
        or value.get("source_row_range") != [expected_start, expected_stop]
        or value.get("compute_dtype") != COMPUTE_DTYPE
        or value.get("output_dtype") != OUTPUT_DTYPE.str
        or len(chunks) != (expected_stop - expected_start) // CHUNK_ROWS
        or value.get("prompt_name_equivalence_passed") is not True
        or value.get("historical_raw_faithfulness_passed") is not True
    ):
        raise Round0112Error("R0112 slice receipt contract changed")
    cursor = expected_start
    for chunk in chunks:
        start, stop = [int(item) for item in chunk["source_row_range"]]
        outputs = chunk.get("outputs") or {}
        if (
            start != cursor
            or stop - start != CHUNK_ROWS
            or set(outputs) != set(CONVENTIONS)
            or any(
                int(outputs[arm].get("bytes", -1))
                < CHUNK_ROWS * DIMENSION * OUTPUT_DTYPE.itemsize
                for arm in CONVENTIONS
            )
            or chunk.get("output_shape") != [CHUNK_ROWS, DIMENSION]
            or chunk.get("output_dtype") != OUTPUT_DTYPE.str
        ):
            raise Round0112Error("R0112 slice chunk geometry changed")
        cursor = stop
    if cursor != expected_stop:
        raise Round0112Error("R0112 slice chunks do not close")
    return value


def aggregate_slice_receipts(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate four slice receipts and aggregate their paired diagnostics."""
    ranges = expected_slice_ranges()
    if len(receipts) != len(ranges):
        raise Round0112Error("R0112 requires exactly four paired slice receipts")
    validated = [
        validate_slice_receipt(value, expected_start=start, expected_stop=stop)
        for value, (start, stop) in zip(receipts, ranges, strict=True)
    ]
    historical = np.concatenate(
        [
            np.asarray(value["historical_raw_cosines"], dtype=np.float64)
            for value in validated
        ]
    )
    shift = np.concatenate(
        [
            np.asarray(
                [chunk["paired_raw_document_cosine_mean"] for chunk in value["chunks"]],
                dtype=np.float64,
            )
            for value in validated
        ]
    )
    if (
        not len(historical)
        or not np.isfinite(historical).all()
        or not np.isfinite(shift).all()
        or float(np.mean(historical)) < HISTORICAL_RAW_MEAN_COSINE_FLOOR
        or float(np.min(historical)) < HISTORICAL_RAW_MIN_COSINE_FLOOR
    ):
        raise Round0112Error("R0112 aggregate faithfulness guard failed")
    return {
        "slices": validated,
        "historical_raw_cosine": {
            "sample_rows": int(len(historical)),
            "mean": float(np.mean(historical)),
            "min": float(np.min(historical)),
            "max": float(np.max(historical)),
            "mean_floor": HISTORICAL_RAW_MEAN_COSINE_FLOOR,
            "min_floor": HISTORICAL_RAW_MIN_COSINE_FLOOR,
            "passed": True,
        },
        "paired_raw_document_chunk_mean_cosine": {
            "chunks": int(len(shift)),
            "mean": float(np.mean(shift)),
            "min": float(np.min(shift)),
            "max": float(np.max(shift)),
            "role": "substrate diagnostic; map-quality effect is not inferred here",
        },
    }


def load_eligibility_prefix(
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Derive a representative-only selector for the *2M cohort*.

    R0087 chose one representative over the complete 25M inventory.  A few
    FineWeb members have their chosen representative later in another source,
    outside this experiment.  Slicing the global exclusion mask would
    therefore erase those families.  Preserve the accepted representative
    when it is in-cohort; otherwise deterministically retain the lowest
    in-cohort member.
    """
    signature = expected_input_signature(ELIGIBILITY_PATH)
    if signature["sha256"] != ELIGIBILITY_SHA256:
        raise Round0112Error("R0087 duplicate eligibility bytes changed")
    with np.load(ELIGIBILITY_PATH, allow_pickle=False) as archive:
        global_excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)
        representatives = np.asarray(
            archive["representative_rows"], dtype=np.int64
        )
        offsets = np.asarray(archive["family_offsets"], dtype=np.int64)
        members = np.asarray(archive["member_rows"], dtype=np.int64)
    if (
        representatives.ndim != 1
        or offsets.shape != (len(representatives) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(members)
        or np.any(offsets[1:] <= offsets[:-1])
    ):
        raise Round0112Error("R0087 duplicate family table is malformed")
    excluded_parts: list[np.ndarray] = []
    rebound: list[tuple[int, int]] = []
    in_cohort_families = 0
    duplicate_families = 0
    for family_index, representative in enumerate(representatives.tolist()):
        family = np.asarray(
            members[offsets[family_index] : offsets[family_index + 1]],
            dtype=np.int64,
        )
        cohort = np.sort(family[family < ROWS])
        if not len(cohort):
            continue
        in_cohort_families += 1
        if len(cohort) == 1:
            # A global duplicate family is a singleton in this cohort.  It
            # must remain represented even when R0087 chose an outside row.
            if representative >= ROWS:
                rebound.append((int(cohort[0]), int(representative)))
            continue
        duplicate_families += 1
        keep = (
            int(representative)
            if representative < ROWS and np.any(cohort == representative)
            else int(cohort[0])
        )
        if representative >= ROWS:
            rebound.append((keep, int(representative)))
        excluded_parts.append(cohort[cohort != keep])
    prefix = (
        np.sort(np.concatenate(excluded_parts)).astype(np.int64)
        if excluded_parts
        else np.empty(0, dtype=np.int64)
    )
    if (
        len(prefix)
        and (
            prefix[0] < 0
            or prefix[-1] >= ROWS
            or np.any(prefix[1:] <= prefix[:-1])
        )
    ):
        raise Round0112Error("R0112 cohort-local duplicate selector is malformed")
    global_prefix = global_excluded[global_excluded < ROWS]
    restored = np.setdiff1d(global_prefix, prefix, assume_unique=True)
    newly_excluded = np.setdiff1d(prefix, global_prefix, assume_unique=True)
    if len(newly_excluded) or len(restored) != len(rebound):
        raise Round0112Error(
            "R0112 cohort-local duplicate-selector reconciliation failed"
        )
    report = {
        "global_family_count": int(len(representatives)),
        "families_touching_cohort": int(in_cohort_families),
        "duplicate_families_within_cohort": int(duplicate_families),
        "global_prefix_excluded_rows": int(len(global_prefix)),
        "cohort_local_excluded_rows": int(len(prefix)),
        "outside_representative_rows_restored": int(len(restored)),
        "restored_rows": restored.tolist(),
        "outside_representatives": [outside for _inside, outside in rebound],
        "newly_excluded_rows": int(len(newly_excluded)),
        "selection_rule": (
            "keep accepted R0087 representative when inside cohort; otherwise "
            "keep lowest global row in the cohort; exclude other cohort members"
        ),
    }
    return prefix, signature, report


def source_contract() -> dict[str, Any]:
    """Stable scientific identity shared by queue preparation and receipts."""
    return {
        "rows": ROWS,
        "dimension": DIMENSION,
        "source_global_rows": [0, ROWS],
        "source_dataset": (
            "fineweb-edu-sample-10BT-chunked-500-jina-v5-nano"
        ),
        "source_historical_fp16_payload_sha256": (
            SOURCE_FIRST2M_PAYLOAD_SHA256
        ),
        "r0087_inventory_identity_sha256": (
            R0087_INVENTORY_IDENTITY_SHA256
        ),
        "row_order": "R0087/R0103 contiguous global order; exact rows 0:2000000",
        "text_alignment": (
            "same-numbered FineWeb parquet/embedding shards with exact "
            "full-shard row-count equality"
        ),
    }
