from __future__ import annotations

import copy

import numpy as np
import pytest

from basemap.round0112_prompt_substrate import (
    CHUNK_ROWS,
    CONVENTIONS,
    DIMENSION,
    OUTPUT_DTYPE,
    PROMPT_PREFIX,
    ROUND_ID,
    SLICE_SCHEMA,
    Round0112Error,
    aggregate_slice_receipts,
    build_offsets,
    expected_slice_ranges,
    first2m_layout,
    load_eligibility_prefix,
    locate_rows,
    seal,
    source_contract,
    validate_slice_receipt,
)
from experiments.round0112_nodes import _cosine_rows


def _slice_receipt(start: int, stop: int) -> dict:
    chunks = []
    for chunk_index, chunk_start in enumerate(range(start, stop, CHUNK_ROWS)):
        chunk_stop = chunk_start + CHUNK_ROWS
        chunks.append(
            {
                "chunk_index": chunk_index,
                "source_row_range": [chunk_start, chunk_stop],
                "outputs": {
                    arm: {
                        "canonical_path": f"/data/{arm}/{chunk_start}.npy",
                        "kind": "file",
                        "bytes": CHUNK_ROWS
                        * DIMENSION
                        * OUTPUT_DTYPE.itemsize
                        + 128,
                        "sha256": f"{chunk_index + 1:064x}",
                    }
                    for arm in CONVENTIONS
                },
                "output_shape": [CHUNK_ROWS, DIMENSION],
                "output_dtype": OUTPUT_DTYPE.str,
                "paired_raw_document_cosine_mean": 0.8,
            }
        )
    return seal(
        {
            "schema": SLICE_SCHEMA,
            "round_id": ROUND_ID,
            "model_id": "jinaai/jina-embeddings-v5-text-nano-retrieval",
            "model_revision": "ac5d898c8d382b17167c33e5c8af644a3519b47d",
            "prompt_prefix": PROMPT_PREFIX,
            "conventions": list(CONVENTIONS),
            "source_row_range": [start, stop],
            "compute_dtype": "float32",
            "output_dtype": OUTPUT_DTYPE.str,
            "prompt_name_equivalence_passed": True,
            "historical_raw_faithfulness_passed": True,
            "historical_raw_cosines": [0.99, 0.995],
            "chunks": chunks,
        }
    )


def test_slice_ranges_are_four_contiguous_500k_units():
    assert expected_slice_ranges() == [
        (0, 500_000),
        (500_000, 1_000_000),
        (1_000_000, 1_500_000),
        (1_500_000, 2_000_000),
    ]


def test_row_location_preserves_exact_order():
    offsets = build_offsets([3, 4, 2])
    shard, local = locate_rows(np.array([0, 2, 3, 6, 8]), offsets)
    assert shard.tolist() == [0, 0, 1, 1, 2]
    assert local.tolist() == [0, 2, 0, 3, 1]
    with pytest.raises(Round0112Error):
        locate_rows(np.array([2, 1]), offsets)


def test_slice_receipt_closes_exact_chunk_geometry():
    receipt = _slice_receipt(0, 500_000)
    assert validate_slice_receipt(
        receipt, expected_start=0, expected_stop=500_000
    )["prompt_name_equivalence_passed"]
    bad = copy.deepcopy(receipt)
    bad["chunks"][0]["source_row_range"][1] -= 1
    bad = seal({key: value for key, value in bad.items() if key != "identity_sha256"})
    with pytest.raises(Round0112Error):
        validate_slice_receipt(
            bad, expected_start=0, expected_stop=500_000
        )


def test_aggregate_requires_all_four_paired_slices():
    receipts = [_slice_receipt(start, stop) for start, stop in expected_slice_ranges()]
    aggregate = aggregate_slice_receipts(receipts)
    assert aggregate["historical_raw_cosine"]["passed"] is True
    assert aggregate["paired_raw_document_chunk_mean_cosine"]["chunks"] == 80
    with pytest.raises(Round0112Error):
        aggregate_slice_receipts(receipts[:3])


def test_cosine_rows_is_scale_invariant():
    left = np.asarray([[1.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    right = np.asarray([[2.0, 0.0], [0.0, -2.0]], dtype=np.float32)
    assert np.allclose(_cosine_rows(left, right), [1.0, -1.0])


def test_real_first2m_text_layout_matches_authenticated_embedding_order():
    layout = first2m_layout()
    assert layout[0]["global_row_start"] == 0
    assert layout[-1]["global_row_stop"] == 2_000_000
    assert len(layout) == 8
    assert all(item["shard_row_start"] == 0 for item in layout)
    assert source_contract()["source_global_rows"] == [0, 2_000_000]


def test_cohort_local_duplicate_selector_does_not_erase_outside_rep_families():
    excluded, _signature, report = load_eligibility_prefix()
    assert len(excluded) == report["cohort_local_excluded_rows"]
    assert report["global_prefix_excluded_rows"] - len(excluded) == 11
    assert report["outside_representative_rows_restored"] == 11
    assert report["newly_excluded_rows"] == 0
