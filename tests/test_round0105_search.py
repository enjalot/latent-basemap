from __future__ import annotations

import hashlib
import json

import numpy as np

from basemap.round0105_search import (
    EVERY_GROUP_MEAN_FLOOR,
    GLOBAL_MEAN_FLOOR,
    GROUPS,
    INDEX_TRAIN_ROWS,
    INDEX_TRAIN_SAMPLE_SHA256,
    NLIST,
    POLICY_GRID,
    QUALITY_GROUP_IDS_SHA256,
    QUALITY_ROWS,
    QUALITY_ROWS_PER_GROUP,
    QUALITY_SAMPLE_SHA256,
    SUBSTRATE_MANIFEST_PATH,
    group_ranges,
    membership,
    sample_retained_rows,
    sample_stratified_rows,
    select_cell,
)
from experiments.round0105_nodes import (
    _clean_search,
    _gpu_options,
    _normalized_rows,
    _policy_metrics,
    _retained_batch,
)


ELIGIBILITY_PATH = (
    "/data/latent-basemap/runs/round-0087/queue/artifacts/"
    "jina-diverse-25m-inventory/jina-diverse-25m-eligibility-v1.npz"
)


def test_registered_geometry_and_grid_are_fixed() -> None:
    assert INDEX_TRAIN_ROWS == 40 * NLIST
    assert len(GROUPS) == 22
    assert QUALITY_ROWS == 22 * QUALITY_ROWS_PER_GROUP == 5_632
    assert POLICY_GRID == (
        (64, 128),
        (64, 256),
        (64, 512),
        (128, 128),
        (128, 256),
        (128, 512),
        (192, 128),
        (192, 256),
        (192, 512),
    )
    assert GLOBAL_MEAN_FLOOR == 0.90
    assert EVERY_GROUP_MEAN_FLOOR == 0.84


def test_gpu_clone_options_support_registered_pq96_on_5090() -> None:
    options = _gpu_options()
    assert options.indicesOptions != 0
    # GpuClonerOptions.useFloat16 is the actual SWIG-backed field that the
    # IndexIVFPQ cloner maps to GpuIndexIVFPQConfig.useFloat16LookupTables.
    assert options.useFloat16 is True
    assert options.usePrecomputed is True


def test_registered_samples_reproduce_before_issue() -> None:
    manifest = json.load(open(SUBSTRATE_MANIFEST_PATH, encoding="utf-8"))
    with np.load(ELIGIBILITY_PATH, allow_pickle=False) as archive:
        excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)
    train = sample_retained_rows(
        excluded, count=INDEX_TRAIN_ROWS, seed=105
    )
    sample, group_ids = sample_stratified_rows(
        excluded, group_ranges(manifest)
    )
    assert hashlib.sha256(train.tobytes()).hexdigest() == (
        INDEX_TRAIN_SAMPLE_SHA256
    )
    assert hashlib.sha256(sample.tobytes()).hexdigest() == (
        QUALITY_SAMPLE_SHA256
    )
    assert hashlib.sha256(group_ids.tobytes()).hexdigest() == (
        QUALITY_GROUP_IDS_SHA256
    )
    assert np.bincount(group_ids).tolist() == [256] * 22
    assert not membership(excluded, sample).any()


def test_membership_and_retained_batch_preserve_only_eligible_ids() -> None:
    excluded = np.asarray([2, 5, 9, 10, 18], dtype=np.int64)
    values = np.asarray([[1, 2, 5], [8, 10, 20]], dtype=np.int64)
    assert membership(excluded, values).tolist() == [
        [False, True, True],
        [False, True, False],
    ]
    assert _retained_batch(excluded, start=4, stop=12).tolist() == [
        4, 6, 7, 8, 11,
    ]


def test_native_int8_plus_scale_normalization_is_unit_length() -> None:
    encoded = np.zeros((3, 768), dtype=np.int8)
    encoded[0, :2] = [3, 4]
    encoded[1, :2] = [-5, 12]
    encoded[2, :3] = [1, 1, 1]
    scales = np.asarray([0.5, 0.25, 2.0], dtype="<f2")
    values = _normalized_rows(
        encoded, scales, np.asarray([2, 0, 1], dtype=np.int64)
    )
    assert values.dtype == np.float32
    np.testing.assert_allclose(
        np.linalg.norm(values, axis=1), np.ones(3), atol=1e-6
    )


def test_candidate_cleanup_removes_self_and_rejects_excluded_rows() -> None:
    raw = np.asarray(
        [[4, 8, 7, 6], [9, 10, 11, 12]],
        dtype=np.int64,
    )
    selected, self_seen = _clean_search(
        raw,
        sources=np.asarray([4, 10], dtype=np.int64),
        width=3,
        excluded=np.asarray([100], dtype=np.int64),
    )
    assert selected.tolist() == [[8, 7, 6], [9, 11, 12]]
    assert self_seen == 2


def test_per_group_floor_cannot_hide_behind_global_mean() -> None:
    exact = np.tile(np.arange(15, dtype=np.int64), (QUALITY_ROWS, 1))
    selected = exact.copy()
    selected[:QUALITY_ROWS_PER_GROUP, :4] = np.arange(100, 104)
    group_ids = np.repeat(
        np.arange(len(GROUPS), dtype=np.uint8),
        QUALITY_ROWS_PER_GROUP,
    )
    metrics = _policy_metrics(
        selected,
        exact,
        group_ids=group_ids,
        unambiguous=np.ones(QUALITY_ROWS, dtype=bool),
    )
    assert metrics["passes_global_floor"] is True
    assert metrics["by_group"][GROUPS[0]]["passes_floor"] is False
    assert metrics["passes_every_group_floor"] is False
    assert metrics["all_rows_complete"] is True


def test_selector_uses_only_complete_dual_floor_passing_cells() -> None:
    cells = {
        "incomplete": {
            "nprobe": 64,
            "shortlist_width": 128,
            "passes_global_floor": True,
            "passes_every_group_floor": True,
            "all_rows_complete": False,
            "benchmark": {"median_wall_seconds_per_query": 0.001},
        },
        "slow": {
            "nprobe": 128,
            "shortlist_width": 256,
            "passes_global_floor": True,
            "passes_every_group_floor": True,
            "all_rows_complete": True,
            "benchmark": {"median_wall_seconds_per_query": 0.003},
        },
        "fast": {
            "nprobe": 192,
            "shortlist_width": 128,
            "passes_global_floor": True,
            "passes_every_group_floor": True,
            "all_rows_complete": True,
            "benchmark": {"median_wall_seconds_per_query": 0.002},
        },
    }
    assert select_cell(cells) is cells["fast"]
