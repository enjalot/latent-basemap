"""Canonical prompted-English staging and 8M selection for Round 0162."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from basemap.artifact_identity import canonical_json, sha256_bytes


ROUND_ID = "0162"
CAPABILITY = "jina-document-english-9p126m-canonical-layout-v1"
VIEW_CAPABILITY = "jina-document-english-first8m-view-v1"
FINEWEB = "fineweb-edu-sample-10BT-chunked-500-jina-v5-nano"
REDPAJAMA = "RedPajama-Data-V2-sample-10B-chunked-500-jina-v5-nano"
PILE = "pile-uncopyrighted-chunked-500-jina-v5-nano"
DATASETS = (FINEWEB, REDPAJAMA, PILE)
DATASET_ROWS = {FINEWEB: 2_890_362, REDPAJAMA: 2_836_978, PILE: 3_399_036}
DATASET_OFFSETS = {
    FINEWEB: 0,
    REDPAJAMA: DATASET_ROWS[FINEWEB],
    PILE: DATASET_ROWS[FINEWEB] + DATASET_ROWS[REDPAJAMA],
}
TOTAL_ROWS = sum(DATASET_ROWS.values())
VIEW_ROWS = 8_000_000
DIMENSION = 768
DTYPE = "<f2"


class Round0162Error(RuntimeError):
    """Raised when accepted prompted-English tranche lineage changes."""


def _signature_shape(signature: Mapping[str, Any]) -> None:
    if (
        signature.get("kind") != "file"
        or not isinstance(signature.get("canonical_path"), str)
        or not isinstance(signature.get("bytes"), int)
        or int(signature["bytes"]) <= 0
        or not isinstance(signature.get("sha256"), str)
        or len(str(signature["sha256"])) != 64
    ):
        raise Round0162Error("prompted chunk signature is malformed")


def ordered_chunks(
    r0116: Mapping[str, Any], r0120: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Normalize both reviewed tranche schemas into one gap-free order."""
    if (
        r0116.get("schema") != "jina-document-english-fineweb-rpj-5p727m-v1"
        or r0116.get("round_id") != "0116"
        or int(r0116.get("row_count", -1)) != DATASET_OFFSETS[PILE]
        or r0116.get("training_performed") is not False
        or r0120.get("schema") != "jina-document-pile-english-3p399m-v1"
        or r0120.get("round_id") != "0120"
        or int(r0120.get("row_count", -1)) != DATASET_ROWS[PILE]
        or r0120.get("training_performed") is not False
    ):
        raise Round0162Error("reviewed prompted tranche contract changed")
    r0116_datasets = r0116.get("datasets")
    r0120_dataset = r0120.get("dataset")
    if not isinstance(r0116_datasets, Mapping) or not isinstance(r0120_dataset, Mapping):
        raise Round0162Error("prompted tranche datasets are missing")

    sources = {
        FINEWEB: r0116_datasets.get(FINEWEB),
        REDPAJAMA: r0116_datasets.get(REDPAJAMA),
        PILE: r0120_dataset,
    }
    output: list[dict[str, Any]] = []
    cursor = 0
    for dataset in DATASETS:
        description = sources[dataset]
        if (
            not isinstance(description, Mapping)
            or int(description.get("row_count", -1)) != DATASET_ROWS[dataset]
            or not isinstance(description.get("chunks"), list)
            or int(description.get("chunk_count", -1)) != len(description["chunks"])
        ):
            raise Round0162Error(f"prompted {dataset} dataset contract changed")
        dataset_cursor = 0
        for chunk in description["chunks"]:
            dataset_range = list(chunk.get("dataset_row_range") or [])
            shape = list(chunk.get("output_shape") or [])
            signature = chunk.get("output")
            if (
                len(dataset_range) != 2
                or dataset_range[0] != dataset_cursor
                or dataset_range[1] <= dataset_range[0]
                or shape != [dataset_range[1] - dataset_range[0], DIMENSION]
                or chunk.get("output_dtype") != DTYPE
                or not isinstance(signature, Mapping)
            ):
                raise Round0162Error(f"prompted {dataset} chunk order changed")
            _signature_shape(signature)
            start = DATASET_OFFSETS[dataset] + dataset_range[0]
            stop = DATASET_OFFSETS[dataset] + dataset_range[1]
            if start != cursor:
                raise Round0162Error("prompted English chunks are not gap-free")
            output.append({
                "position": len(output),
                "dataset": dataset,
                "dataset_row_range": dataset_range,
                "canonical_row_range": [start, stop],
                "output_shape": shape,
                "output_dtype": DTYPE,
                "source_output": dict(signature),
                "source_round": "0120" if dataset == PILE else "0116",
                "source_text_ordered_sha256": str(chunk["source_text_ordered_sha256"]),
                "document_text_ordered_sha256": str(chunk["document_text_ordered_sha256"]),
            })
            dataset_cursor = dataset_range[1]
            cursor = stop
        if dataset_cursor != DATASET_ROWS[dataset]:
            raise Round0162Error(f"prompted {dataset} rows do not close")
    if cursor != TOTAL_ROWS:
        raise Round0162Error("prompted English row count does not close")
    return output


def first_view(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    slices: list[dict[str, Any]] = []
    cursor = 0
    for chunk in chunks:
        start, stop = (int(value) for value in chunk["canonical_row_range"])
        if start >= VIEW_ROWS:
            break
        selected_stop = min(stop, VIEW_ROWS)
        selected_rows = selected_stop - start
        if start != cursor or selected_rows <= 0:
            raise Round0162Error("first-8M selection is not gap-free")
        slices.append({
            "position": len(slices),
            "chunk_position": int(chunk["position"]),
            "dataset": str(chunk["dataset"]),
            "dataset_row_range": [
                int(chunk["dataset_row_range"][0]),
                int(chunk["dataset_row_range"][0]) + selected_rows,
            ],
            "canonical_row_range": [start, selected_stop],
            "source_array_row_slice": [0, selected_rows],
            "source_output": dict(chunk["source_output"]),
        })
        cursor = selected_stop
    if cursor != VIEW_ROWS:
        raise Round0162Error("first-8M selection does not close")
    dataset_ranges = {
        FINEWEB: [0, DATASET_ROWS[FINEWEB]],
        REDPAJAMA: [DATASET_OFFSETS[REDPAJAMA], DATASET_OFFSETS[PILE]],
        PILE: [DATASET_OFFSETS[PILE], VIEW_ROWS],
    }
    body = {
        "schema": "round0162-first8m-selection-v1",
        "rows": VIEW_ROWS,
        "dimension": DIMENSION,
        "dtype": DTYPE,
        "source_order": list(DATASETS),
        "dataset_canonical_row_ranges": dataset_ranges,
        "slices": slices,
    }
    return {**body, "ordered_selection_sha256": sha256_bytes(canonical_json(body))}


def layout_identity(
    *, r0116_signature: Mapping[str, Any], r0120_signature: Mapping[str, Any], chunks: Sequence[Mapping[str, Any]]
) -> str:
    body = {
        "schema": "round0162-prompted-english-layout-identity-v1",
        "source_manifests": [dict(r0116_signature), dict(r0120_signature)],
        "rows": TOTAL_ROWS,
        "dimension": DIMENSION,
        "dtype": DTYPE,
        "source_order": list(DATASETS),
        "chunks": [
            {
                "position": int(chunk["position"]),
                "dataset": str(chunk["dataset"]),
                "dataset_row_range": list(chunk["dataset_row_range"]),
                "canonical_row_range": list(chunk["canonical_row_range"]),
                "bytes": int(chunk["source_output"]["bytes"]),
                "sha256": str(chunk["source_output"]["sha256"]),
            }
            for chunk in chunks
        ],
    }
    return sha256_bytes(canonical_json(body))
