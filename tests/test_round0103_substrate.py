from __future__ import annotations

import inspect

import numpy as np
import pytest

from basemap.round0087_inventory import (
    FINEWEB,
    PILE,
    REDPAJAMA,
)
from basemap.round0103_substrate import (
    EXCLUDED_ROWS,
    INVENTORY_IDENTITY,
    RECONSTRUCTION_COSINE_P01_FLOOR,
    RETAINED_ROWS,
    Round0103Error,
    build_label_arrays,
    quantize_block,
    retained_sample_rows,
    validate_inventory,
)
from experiments import prepare_round0103_queue, round0103_nodes


def test_accepted_inventory_closes_exact_registered_geometry() -> None:
    value = validate_inventory()
    assert value["manifest"]["identity_sha256"] == INVENTORY_IDENTITY
    assert value["selection"]["selected_rows"] == 25_000_000
    summary = value["manifest"]["duplicate_control"]["summary"]
    assert summary["retained_row_count"] == RETAINED_ROWS
    assert summary["excluded_row_count"] == EXCLUDED_ROWS


def test_row_local_fp16_scale_quantizer_has_high_cosine() -> None:
    rng = np.random.default_rng(103)
    source = rng.normal(size=(257, 768)).astype("<f2")
    encoded, scales = quantize_block(source)
    reconstructed = (
        encoded.astype(np.float32)
        * scales.astype(np.float32)[:, None]
    )
    original = source.astype(np.float32)
    cosine = np.einsum(
        "ij,ij->i",
        original,
        reconstructed,
        dtype=np.float64,
    ) / (
        np.linalg.norm(original, axis=1).astype(np.float64)
        * np.linalg.norm(reconstructed, axis=1).astype(np.float64)
    )
    assert encoded.dtype == np.dtype("int8")
    assert scales.dtype == np.dtype("<f2")
    assert float(np.quantile(cosine, 0.01)) >= (
        RECONSTRUCTION_COSINE_P01_FLOOR
    )


def test_quantizer_rejects_zero_and_nonfinite_rows() -> None:
    zero = np.zeros((1, 4), dtype="<f2")
    with pytest.raises(Round0103Error, match="nonfinite or zero"):
        quantize_block(zero)
    nonfinite = np.ones((1, 4), dtype="<f2")
    nonfinite[0, 2] = np.inf
    with pytest.raises(Round0103Error, match="nonfinite or zero"):
        quantize_block(nonfinite)


def test_streaming_writers_preserve_registered_range_order(
    tmp_path,
    monkeypatch,
) -> None:
    first = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
        [4.0, 5.0, 6.0, 7.0],
    ], dtype="<f2")
    second = np.array([
        [-1.0, -2.0, -3.0, -4.0],
        [-2.0, -3.0, -4.0, -5.0],
        [-3.0, -4.0, -5.0, -6.0],
    ], dtype="<f2")
    first_path = tmp_path / "first.npy"
    second_path = tmp_path / "second.npy"
    np.save(first_path, first)
    np.save(second_path, second)
    selection = {
        "ranges": [
            {
                "global_row_start": 0,
                "global_row_stop": 3,
                "shard_row_start": 1,
                "shard_row_stop": 4,
                "shard": {"canonical_path": str(first_path)},
            },
            {
                "global_row_start": 3,
                "global_row_stop": 5,
                "shard_row_start": 0,
                "shard_row_stop": 2,
                "shard": {"canonical_path": str(second_path)},
            },
        ],
    }
    monkeypatch.setattr(round0103_nodes, "TARGET_ROWS", 5)
    monkeypatch.setattr(round0103_nodes, "DIMENSION", 4)
    monkeypatch.setattr(round0103_nodes, "BLOCK_ROWS", 2)
    scales_path = tmp_path / "scales.f16"
    int8_path = tmp_path / "embeddings.i8"
    round0103_nodes._write_scales(str(scales_path), selection)
    round0103_nodes._write_int8(
        str(int8_path),
        selection,
        str(scales_path),
    )
    expected_source = np.concatenate([first[1:4], second[0:2]])
    expected_int8, expected_scales = quantize_block(expected_source)
    actual_scales = np.memmap(
        scales_path,
        mode="r",
        dtype="<f2",
        shape=(5,),
    )
    actual_int8 = np.memmap(
        int8_path,
        mode="r",
        dtype="int8",
        shape=(5, 4),
    )
    assert np.array_equal(actual_scales, expected_scales)
    assert np.array_equal(actual_int8, expected_int8)


def test_retained_sample_is_ordered_replayable_and_excludes_rows() -> None:
    excluded = np.array([0, 2, 3, 19, 20, 21, 98], dtype=np.int64)
    first = retained_sample_rows(
        excluded,
        row_count=100,
        sample_count=50,
        seed=103,
    )
    second = retained_sample_rows(
        excluded,
        row_count=100,
        sample_count=50,
        seed=103,
    )
    assert np.array_equal(first, second)
    assert np.all(first[1:] > first[:-1])
    assert not np.intersect1d(first, excluded).size


def test_compact_labels_preserve_dataset_english_and_language_identity() -> None:
    language = "fineweb2-arb_Arab-chunked-500-jina-v5-nano"
    source_order = [FINEWEB, REDPAJAMA, PILE, language]
    ranges = []
    for index, dataset in enumerate(source_order):
        ranges.append({
            "dataset": dataset,
            "language": None if index < 3 else "arb_Arab",
            "global_row_start": index * 2,
            "global_row_stop": index * 2 + 2,
        })
    labels = build_label_arrays(
        {"source_order": source_order, "ranges": ranges},
        row_count=8,
    )
    arrays = labels["arrays"]
    assert arrays["dataset_id"].tolist() == [
        0, 0, 1, 1, 2, 2, 3, 3,
    ]
    assert arrays["english_corpus_id"].tolist() == [
        1, 1, 2, 2, 3, 3, 0, 0,
    ]
    assert arrays["language_id"].tolist() == [
        0, 0, 0, 0, 0, 0, 1, 1,
    ]
    assert labels["vocabulary"]["language"] == [
        "eng_Latn",
        "arb_Arab",
    ]


def test_queue_is_one_cpu_only_nonoverlapping_stage() -> None:
    source = inspect.getsource(
        prepare_round0103_queue.prepare_round0103
    )
    assert source.count('"action": "stage_full768_int8"') == 1
    assert 'gpu_hours_cap=0.0' in source
    assert 'execution_authority="autonomous-cpu"' in source
    assert '"may_overlap_host_int8_training": False' in source
    assert '"jina-mrl-two-seed-decision-v1"' in source
    assert '"jina-diverse-25m-inventory-v1"' in source
    assert '"jina-diverse-25m-full768-int8-substrate-v1"' in source


def test_node_rehashes_sources_and_emits_full768_raw_payload() -> None:
    source = inspect.getsource(round0103_nodes)
    assert "expected_input_signature(path)" in source
    assert "TARGET_ROWS * DIMENSION" in source
    assert "dimension_truncated" in source
    assert "normalization_applied" in source
    assert "prompt_applied" in source
    assert '"optimizer_updates": 0' in source
    assert '"gpu_used": False' in source
