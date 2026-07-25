from __future__ import annotations

import importlib
import inspect

import numpy as np

from basemap.round0053_program import (
    compact30_to_global150,
    compact30_to_source60,
    global150_to_compact30,
    source60_to_compact30,
)
from experiments.round0049_nodes import _clean_search


def test_balanced_30m_mappings_are_not_contiguous_prefixes() -> None:
    compact = np.asarray(
        [0, 9_999_999, 10_000_000, 19_999_999, 20_000_000, 29_999_999],
        dtype=np.int64,
    )
    source = compact30_to_source60(compact)
    assert source.tolist() == [
        0,
        9_999_999,
        20_000_000,
        29_999_999,
        40_000_000,
        49_999_999,
    ]
    np.testing.assert_array_equal(
        source60_to_compact30(source),
        compact,
    )
    global_rows = compact30_to_global150(compact)
    assert global_rows.tolist() == [
        0,
        9_999_999,
        50_000_000,
        59_999_999,
        100_000_000,
        109_999_999,
    ]
    np.testing.assert_array_equal(
        global150_to_compact30(global_rows),
        compact,
    )


def test_clean_search_accepts_matched_control_mapping() -> None:
    source = np.asarray([0], dtype=np.int64)
    global_source = compact30_to_global150(source)
    raw = np.asarray(
        [[0, *range(1, 20), 50_000_000, 100_000_000]],
        dtype=np.int64,
    )
    compact, self_seen = _clean_search(
        raw,
        global_sources=global_source,
        global_to_compact_fn=global150_to_compact30,
    )
    assert self_seen == 1
    assert compact.shape == (1, 15)
    assert compact[0].tolist() == list(range(1, 16))


def test_generalized_subset_writer_keeps_r0049_defaults() -> None:
    from basemap import round0049_program

    signature = inspect.signature(
        round0049_program.write_subset_eligibility
    )
    assert signature.parameters["round_id"].default == "0049"
    assert signature.parameters["universe"].default == (
        "minilm-int8-balanced-60m"
    )
    assert signature.parameters["source_input_key"].default == (
        "r0033_eligibility"
    )


def test_round0053_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0053_program")
    importlib.import_module("experiments.round0053_nodes")
    importlib.import_module("experiments.prepare_round0053_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"


def test_r0053_is_quality_only_and_bounded() -> None:
    from experiments import prepare_round0053_queue as queue_prep

    source = inspect.getsource(queue_prep.prepare_round0053)
    assert "review-0049-2026-07-26.md" in source
    assert "review-0049-2026-07-25.md" not in source
    assert "gpu_hours_cap=0.5" in source
    assert '"total": 1_200.0' in source
    assert '"action": "validate_candidate_quality"' in source
    assert '"training_performed"] = False' in source
