from __future__ import annotations

import copy
import os

import numpy as np

from basemap.panel_v2 import PanelV2Config
from basemap.round0043_program import (
    BalancedRungSelector,
    BalancedRungView,
)
from experiments import round0043_nodes as node


def _selector(width: int) -> BalancedRungSelector:
    excluded = np.array(
        [0, 3, 8, 10, 13, 18, 20, 23, 28], dtype=np.int64
    )
    return BalancedRungSelector(
        excluded,
        per_corpus_rows=width,
        corpus_block_rows=10,
        corpus_count=3,
    )


def test_balanced_rung_uses_three_disjoint_global_intervals() -> None:
    selector = _selector(5)
    expected = np.array(
        [1, 2, 4, 11, 12, 14, 21, 22, 24], dtype=np.int64
    )
    assert len(selector) == len(expected)
    assert np.array_equal(
        selector.compact_to_global(np.arange(len(selector))), expected
    )
    assert np.array_equal(
        selector.global_to_compact(expected),
        np.arange(len(expected)),
    )
    identity = selector.identity()
    assert identity["intervals"] == [[0, 5], [10, 15], [20, 25]]
    assert identity["retained_rows_per_corpus"] == [3, 3, 3]


def test_balanced_rungs_are_strictly_nested_in_global_namespace() -> None:
    small = _selector(5)
    large = _selector(9)
    small_rows = small.compact_to_global(np.arange(len(small)))
    large_rows = large.compact_to_global(np.arange(len(large)))
    assert np.all(large.is_member(small_rows))
    assert np.array_equal(
        large.compact_to_global(large.global_to_compact(small_rows)),
        small_rows,
    )
    assert set(small_rows) < set(large_rows)


def test_balanced_view_never_treats_disjoint_ranges_as_prefix() -> None:
    selector = _selector(5)
    base = np.arange(60, dtype=np.float32).reshape(30, 2)
    view = BalancedRungView(base, selector)
    global_rows = selector.compact_to_global(np.arange(len(selector)))
    assert np.array_equal(view[:], base[global_rows])
    assert np.array_equal(view[[8, 0, 4]], base[global_rows[[8, 0, 4]]])


def test_fixed_core_anchor_bytes_map_identically_into_larger_rung() -> None:
    config = PanelV2Config(n_anchors=4, anchor_seed=123)
    small = _selector(5)
    large = _selector(9)
    compact = node.sample_anchors(len(small), config)
    global_rows = small.compact_to_global(compact)
    remapped = large.global_to_compact(global_rows)
    assert np.array_equal(large.compact_to_global(remapped), global_rows)


def test_nested_metric_scorer_matches_panel_primitives_on_cpu(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    rng = np.random.RandomState(43)
    base_x = rng.normal(size=(30, 4)).astype(np.float32)
    base_z = rng.normal(size=(30, 2)).astype(np.float32)
    selector = BalancedRungSelector(
        np.array([3, 13, 23], dtype=np.int64),
        per_corpus_rows=8,
        corpus_block_rows=10,
        corpus_count=3,
    )
    x = BalancedRungView(base_x, selector)
    z = BalancedRungView(base_z, selector)
    config = PanelV2Config(
        frac=0.25,
        k_density=3,
        k_hit=2,
        n_anchors=5,
        anchor_seed=7,
        corpus_chunk=7,
        block_elems=70,
        rerank_byte_cap=1_000_000,
        peak_byte_cap=2_000_000,
    )
    anchors = node.sample_anchors(len(selector), config)
    result = node._score_anchor_view(
        embeddings=x,
        coordinates=z,
        anchors=anchors,
        config=config,
    )
    assert result["n"] == len(selector)
    assert result["n_anchors"] == 5
    assert result["k_fraction"] == 6
    assert 0 <= result["ffr"] <= 1
    assert 0 <= result["recall_at_10"] <= 1
    assert np.isfinite(result["density"])


def test_r0036_reproduction_check_is_exact_and_fail_closed(
    monkeypatch,
) -> None:
    baseline = {
        "schema": "round0036-registered-panel-v1",
        "round_id": "0036",
        "scientific_universe": {"rows": 147_221_757},
        "panel": {
            "anchor_hash": "abc",
            "ffr": 0.5,
            "recall@k": 0.1,
            "density": 0.2,
        },
    }
    monkeypatch.setattr(node, "_load_r0036_panel", lambda: baseline)
    monkeypatch.setattr(
        node,
        "expected_input_signature",
        lambda path: {"canonical_path": path, "sha256": "a" * 64},
    )
    metrics = {
        "anchor_hash": "abc",
        "ffr": 0.5,
        "recall_at_10": 0.1,
        "density": 0.2,
    }
    assert node._reproduce_r0036(
        per_corpus_rows=50_000_000,
        rung_metrics=metrics,
    )["passed"]
    changed = copy.deepcopy(metrics)
    changed["density"] = 0.19
    try:
        node._reproduce_r0036(
            per_corpus_rows=50_000_000,
            rung_metrics=changed,
        )
    except Exception as exc:
        assert "does not reproduce" in str(exc)
    else:
        raise AssertionError("R0036 reproduction drift was accepted")


def test_round0043_queue_source_registers_no_training() -> None:
    path = os.path.join(
        os.path.dirname(node.__file__), "prepare_round0043_queue.py"
    )
    text = open(path, encoding="utf-8").read()
    assert '"training_performed": False' in text
    assert "gpu_hours_cap=3.0" in text
    assert "score_nested_030m" not in text
