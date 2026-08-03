"""Pure layout tests for R0162 prompted-English staging."""
from __future__ import annotations

from basemap.round0162_prompted_english_staging import (
    DATASET_OFFSETS,
    DATASET_ROWS,
    DATASETS,
    DIMENSION,
    DTYPE,
    PILE,
    TOTAL_ROWS,
    VIEW_ROWS,
    first_view,
    layout_identity,
    ordered_chunks,
)


def _chunk(dataset: str, rows: int, byte: str, *, pile: bool = False) -> dict:
    value = {
        "dataset": dataset,
        "dataset_row_range": [0, rows],
        "corpus_global_row_range": [0, rows] if pile else [DATASET_OFFSETS[dataset], DATASET_OFFSETS[dataset] + rows],
        "output": {"canonical_path": f"/data/{dataset}.npy", "kind": "file", "bytes": rows * DIMENSION * 2 + 128, "sha256": byte * 64},
        "output_dtype": DTYPE,
        "output_shape": [rows, DIMENSION],
        "source_text_ordered_sha256": "c" * 64,
        "document_text_ordered_sha256": "d" * 64,
    }
    if pile:
        value["r0087_global_row_range"] = [DATASET_OFFSETS[PILE], TOTAL_ROWS]
    return value


def _manifests() -> tuple[dict, dict]:
    r0116_datasets = {
        dataset: {"row_count": DATASET_ROWS[dataset], "chunk_count": 1, "chunks": [_chunk(dataset, DATASET_ROWS[dataset], byte)]}
        for dataset, byte in zip(DATASETS[:2], ("a", "b"), strict=True)
    }
    r0116 = {
        "schema": "jina-document-english-fineweb-rpj-5p727m-v1",
        "round_id": "0116",
        "row_count": DATASET_OFFSETS[PILE],
        "training_performed": False,
        "datasets": r0116_datasets,
    }
    r0120 = {
        "schema": "jina-document-pile-english-3p399m-v1",
        "round_id": "0120",
        "row_count": DATASET_ROWS[PILE],
        "training_performed": False,
        "dataset": {"row_count": DATASET_ROWS[PILE], "chunk_count": 1, "chunks": [_chunk(PILE, DATASET_ROWS[PILE], "e", pile=True)]},
    }
    return r0116, r0120


def test_normalized_layout_and_first8m_view_close_exactly() -> None:
    chunks = ordered_chunks(*_manifests())
    assert len(chunks) == 3
    assert chunks[0]["canonical_row_range"] == [0, DATASET_ROWS[DATASETS[0]]]
    assert chunks[-1]["canonical_row_range"] == [DATASET_OFFSETS[PILE], TOTAL_ROWS]
    view = first_view(chunks)
    assert view["rows"] == VIEW_ROWS
    assert view["slices"][-1]["canonical_row_range"] == [DATASET_OFFSETS[PILE], VIEW_ROWS]
    assert view["slices"][-1]["dataset_row_range"] == [0, VIEW_ROWS - DATASET_OFFSETS[PILE]]
    assert len(view["ordered_selection_sha256"]) == 64


def test_layout_identity_binds_source_manifest_signatures() -> None:
    chunks = ordered_chunks(*_manifests())
    first = {"canonical_path": "/a", "kind": "file", "bytes": 1, "sha256": "1" * 64}
    second = {"canonical_path": "/b", "kind": "file", "bytes": 1, "sha256": "2" * 64}
    baseline = layout_identity(r0116_signature=first, r0120_signature=second, chunks=chunks)
    changed = dict(second, sha256="3" * 64)
    assert baseline != layout_identity(r0116_signature=first, r0120_signature=changed, chunks=chunks)
