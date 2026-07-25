from __future__ import annotations

import numpy as np

from basemap.round0049_program import (
    compact_to_global,
    derive_subset_eligibility_arrays,
    global_to_compact,
)
from experiments.round0049_nodes import (
    _clean_search,
    _exact_rerank_shortlist,
    _warm_page_cache,
)


def test_balanced_mapping_round_trip_and_outside_marker():
    intervals = ((0, 3), (5, 8))
    compact = np.arange(6, dtype=np.int64)
    global_rows = compact_to_global(
        compact,
        intervals=intervals,
        source_rows=10,
    )
    assert global_rows.tolist() == [0, 1, 2, 5, 6, 7]
    np.testing.assert_array_equal(
        global_to_compact(
            global_rows,
            intervals=intervals,
            source_rows=10,
        ),
        compact,
    )
    assert global_to_compact(
        np.asarray([3, 4, 8, 9]),
        intervals=intervals,
        source_rows=10,
    ).tolist() == [-1, -1, -1, -1]


def test_subset_families_choose_an_in_subset_representative():
    source = {
        "family_counts": np.asarray([2, 3, 3], dtype=np.int64),
        "family_offsets": np.asarray([0, 2, 5, 8], dtype=np.int64),
        "member_rows": np.asarray(
            [3, 4, 1, 2, 6, 5, 7, 9],
            dtype=np.int64,
        ),
        "zero_rows": np.asarray([0, 8], dtype=np.int64),
    }
    arrays = derive_subset_eligibility_arrays(
        source,
        intervals=((0, 3), (5, 8)),
        source_rows=10,
    )
    # Family [3,4] disappears. [1,2,6] becomes compact [1,2,4].
    # [5,7,9] becomes compact [3,5]. Row 0 is a selected zero.
    assert arrays["representative_rows"].tolist() == [1, 3]
    assert arrays["family_counts"].tolist() == [3, 2]
    assert arrays["family_offsets"].tolist() == [0, 3, 5]
    assert arrays["member_rows"].tolist() == [1, 2, 4, 3, 5]
    assert arrays["duplicate_excluded_rows"].tolist() == [2, 4, 5]
    assert arrays["duplicate_representative_rows"].tolist() == [1, 1, 3]
    assert arrays["zero_rows"].tolist() == [0]
    assert arrays["excluded_rows"].tolist() == [0, 2, 4, 5]


def test_subset_families_reorder_by_the_new_compact_representative():
    source = {
        # Full-universe representatives 3 then 5 are sorted. After row 3 is
        # removed by the subset, the first family's representative moves past
        # the second family in compact order.
        "family_counts": np.asarray([3, 2], dtype=np.int64),
        "family_offsets": np.asarray([0, 3, 5], dtype=np.int64),
        "member_rows": np.asarray([3, 7, 8, 5, 6], dtype=np.int64),
        "zero_rows": np.empty(0, dtype=np.int64),
    }
    arrays = derive_subset_eligibility_arrays(
        source,
        intervals=((0, 3), (5, 9)),
        source_rows=12,
    )
    assert arrays["representative_rows"].tolist() == [3, 5]
    assert arrays["family_counts"].tolist() == [2, 2]
    assert arrays["member_rows"].tolist() == [3, 4, 5, 6]
    assert arrays["duplicate_excluded_rows"].tolist() == [4, 6]
    assert arrays["duplicate_representative_rows"].tolist() == [3, 5]


def test_search_cleanup_removes_self_and_preserves_rank_order():
    source = np.asarray([0], dtype=np.int64)
    raw = np.asarray(
        [[0, *range(1, 20), 50_000_000, 100_000_000]],
        dtype=np.int64,
    )
    compact, self_seen = _clean_search(
        raw,
        global_sources=source,
    )
    assert self_seen == 1
    assert compact.shape == (1, 15)
    assert compact[0].tolist() == list(range(1, 16))


def test_search_cleanup_keeps_full_r0047_shortlist_after_self():
    raw = np.asarray(
        [[0, *range(1, 129)]],
        dtype=np.int64,
    )
    compact, self_seen = _clean_search(
        raw,
        global_sources=np.asarray([0], dtype=np.int64),
        candidate_count=128,
    )
    assert self_seen == 1
    assert compact.shape == (1, 128)
    assert compact[0, [0, -1]].tolist() == [1, 128]


def test_exact_rerank_uses_dequantized_cosine_not_pq_order():
    encoded = np.asarray(
        [
            [1, 0, 0],
            [0, 1, 0],
            [2, 2, 0],
            [3, 0, 0],
        ],
        dtype=np.int8,
    )
    padded = np.zeros((4, 384), dtype=np.int8)
    padded[:, :3] = encoded
    scales = np.asarray([1, 2, 3, 4], dtype="<f2")
    query = np.zeros((1, 384), dtype=np.float32)
    query[0, 0] = 1
    selected, receipt = _exact_rerank_shortlist(
        queries=query,
        shortlist=np.asarray([[1, 2, 3]], dtype=np.int32),
        encoded=padded,
        scales=scales,
        k=2,
        batch_rows=1,
    )
    assert selected.tolist() == [[3, 2]]
    assert receipt["shortlist_width"] == 3
    assert receipt["vector_source"].startswith("int8-plus-fp16")


def test_page_cache_warm_reads_every_byte(tmp_path):
    path = tmp_path / "vectors.i8"
    path.write_bytes(b"registered-vector-bytes")
    receipt = _warm_page_cache(str(path))
    assert receipt["bytes"] == path.stat().st_size
    assert receipt["wall_seconds"] >= 0
