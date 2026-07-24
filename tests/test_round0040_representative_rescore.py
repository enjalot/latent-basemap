import os

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.panel_v2 import _require_score_panel_scale_admission
from basemap.round0040_program import (
    CachedShardedArray,
    RepresentativeArrayView,
    RepresentativeRowSelector,
    _exact_families_from_fingerprints,
    _write_jina_fingerprints,
)
from experiments.prepare_round0040_queue import _jobs
from experiments.round0040_nodes import _scan_minilm_invalid_rows


def _member(path, start, stop):
    return {
        "path": str(path),
        "global_row_start": start,
        "global_row_stop": stop,
        "signature": expected_input_signature(str(path)),
    }


def test_exact_fingerprint_census_byte_verifies_families_and_invalid_rows(
    tmp_path,
):
    array = np.asarray([
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [0, 0, 0],
        [4, 5, 6],
        [4, 5, 6],
        [7, 8, 9],
    ], dtype=np.float16)
    path = tmp_path / "fingerprints.bin"
    fingerprint = _write_jina_fingerprints(
        array, str(path), block_rows=3
    )
    families = _exact_families_from_fingerprints(array, str(path))

    assert fingerprint["zero_rows"].tolist() == [3]
    assert fingerprint["nonfinite_rows"].tolist() == []
    assert families["representative_rows"].tolist() == [0, 1]
    assert families["family_counts"].tolist() == [2, 3]
    assert families["family_offsets"].tolist() == [0, 2, 5]
    assert families["member_rows"].tolist() == [0, 2, 1, 4, 5]
    assert families["hash_collision_splits"] == 0


def test_cached_shards_and_representative_view_preserve_compact_order(
    tmp_path,
):
    first = np.arange(18, dtype=np.float16).reshape(6, 3)
    second = np.arange(18, 36, dtype=np.float16).reshape(6, 3)
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    np.save(first_path, first)
    np.save(second_path, second)
    base = CachedShardedArray(
        [
            _member(first_path, 0, 6),
            _member(second_path, 6, 12),
        ],
        row_count=12,
        dimension=3,
        dtype="<f2",
    )
    source = expected_input_signature(str(first_path))
    selector = RepresentativeRowSelector(
        np.asarray([1, 4, 9], dtype=np.int64),
        row_count=12,
        source=source,
        policy="synthetic exact-family representatives",
    )
    view = RepresentativeArrayView(base, selector)
    full = np.concatenate([first, second])
    expected_rows = np.asarray([0, 2, 3, 5, 6, 7, 8, 10, 11])

    assert len(view) == 9
    np.testing.assert_array_equal(view[:], full[expected_rows])
    np.testing.assert_array_equal(
        view[np.asarray([[8, 0], [5, 2]])],
        full[expected_rows[np.asarray([[8, 0], [5, 2]])]],
    )
    np.testing.assert_array_equal(
        selector.global_to_compact(np.asarray([0, 8, 11])),
        np.asarray([0, 6, 8]),
    )


def test_invalid_row_scan_uses_full_sharded_matrix(tmp_path):
    first = np.asarray([[1, 2], [0, 0], [3, 4]], dtype=np.float16)
    second = np.asarray(
        [[5, 6], [np.inf, 1], [7, 8]], dtype=np.float16
    )
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    np.save(first_path, first)
    np.save(second_path, second)
    base = CachedShardedArray(
        [
            _member(first_path, 0, 3),
            _member(second_path, 3, 6),
        ],
        row_count=6,
        dimension=2,
        dtype="<f2",
    )

    zero, nonfinite = _scan_minilm_invalid_rows(base)
    assert zero.tolist() == [1]
    assert nonfinite.tolist() == [4]


def test_representative_scale_identity_is_self_contained(tmp_path):
    first = np.arange(24, dtype=np.float16).reshape(8, 3)
    path = tmp_path / "source.npy"
    np.save(path, first)
    base = CachedShardedArray(
        [_member(path, 0, 8)],
        row_count=8,
        dimension=3,
        dtype="<f2",
    )
    selector = RepresentativeRowSelector(
        np.asarray([2, 6], dtype=np.int64),
        row_count=8,
        source=expected_input_signature(str(path)),
        policy="synthetic exact-family representatives",
    )
    view = RepresentativeArrayView(base, selector)
    identity = view.scale_admission_identity()

    assert identity["schema"] == "representative-row-scale-input-v1"
    assert identity["row_count"] == 6
    assert identity["base"]["shape"] == [8, 3]
    assert identity["selector"]["representative_count"] == 6

    # Exercise the >=8M branch without allocating an 8M-row fixture.
    original_shape = view.shape
    original_base_shape = base.shape
    original_count = selector.retained_count
    original_rows = selector.row_count
    try:
        view.shape = (8_000_000, 3)
        base.shape = (8_000_002, 3)
        selector.retained_count = 8_000_000
        selector.row_count = 8_000_002
        admitted = _require_score_panel_scale_admission(view, None)
        assert admitted["schema"] == "representative-row-scale-input-v1"
    finally:
        view.shape = original_shape
        base.shape = original_base_shape
        selector.retained_count = original_count
        selector.row_count = original_rows


def test_round0040_module_does_not_mutate_cuda_visibility():
    before = os.environ.get("CUDA_VISIBLE_DEVICES")
    __import__("experiments.round0040_nodes")
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


def test_round0040_queue_is_no_training_and_bounded():
    jobs = _jobs(artifacts="/data/test-r0040", inputs=[])
    assert [job["id"] for job in jobs] == [
        "jina_census",
        "jina_representative_rescore",
        "minilm_representative_reference",
        "minilm_representative_rescore",
        "duplicate_controlled_comparison",
    ]
    assert all(
        job["node_policy"]["training_performed"] is False for job in jobs
    )
    assert sum(
        job["p90_wall_s"]
        for job in jobs if job["node_policy"]["gpu_required"]
    ) == 6_300.0
