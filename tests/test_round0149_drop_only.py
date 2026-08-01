from __future__ import annotations

import numpy as np
import pytest

from basemap.round0104_training import (
    PIPELINE,
    PairedHostWeightedJinaSampler,
    Round0104TrainingInput,
    negative_sampling_stamp,
)
from basemap.round0140_subsystem_bisection import (
    CURRENT_GRAPH_CURRENT_HOST,
    RESTORATION_FLOORS,
)
from basemap.round0147_row_policy import TREATMENT as SIZE_PRESERVING_TREATMENT
from basemap.round0149_drop_only import (
    RAW_PREFIX_EXCLUDED_ROWS,
    RAW_PREFIX_ROWS,
    ROWS,
    ROW_UNIVERSE,
    TREATMENT,
    Round0149Error,
    build_decision,
    derive_drop_only_selection,
    treatment_train_config,
)


def _signature(path: str, character: str) -> dict:
    return {
        "canonical_path": path,
        "kind": "file",
        "bytes": 1,
        "sha256": character * 64,
    }


def _cell(offset: float) -> dict:
    values = {key: floor + offset for key, floor in RESTORATION_FLOORS.items()}
    return {
        "panel": {
            "ffr": values["ffr"],
            "purity": {
                "k256": values["purity_fidelity_k256"],
                "k1024": values["purity_fidelity_k1024"],
            },
        },
        "projection": {
            "ffr": values["projection_ffr"],
            "recall_at_10": values["ood_recall_at_10"],
        },
    }


def _summary() -> dict:
    return {
        "target_rows": ROWS,
        "raw_prefix_rows": RAW_PREFIX_ROWS,
        "raw_prefix_excluded_rows": RAW_PREFIX_EXCLUDED_ROWS,
        "eligible_rows_retained": ROWS,
        "replacement_rows_beyond_raw_prefix": 0,
        "historical_position_start": 0,
        "historical_position_stop_exclusive": RAW_PREFIX_ROWS,
        "historical_order_preserved": True,
        "size_preserving": False,
        "parent_selection_target_rows": RAW_PREFIX_ROWS,
    }


def test_drop_only_train_config_stamps_real_row_universe() -> None:
    config, digest = treatment_train_config(
        graph_signature=_signature("/tmp/graph.npz", "1"),
        graph_manifest_signature=_signature("/tmp/manifest.json", "2"),
        graph_edges=123_456,
        source_sha256="3" * 64,
        selection_sha256="4" * 64,
    )
    expected = config["execution"]["expected_pipeline_stamp"]
    assert config["paired_invariant"]["rows"] == ROWS
    assert config["arm"] == TREATMENT
    assert expected["negative_sampling"] == negative_sampling_stamp(ROWS)
    assert expected["row_universe"] == ROW_UNIVERSE
    assert config["input_preprocessing"]["replacement_rows"] == 0
    assert len(digest) == 64


def test_drop_only_selector_branches_without_causal_overclaim() -> None:
    cells = {
        CURRENT_GRAPH_CURRENT_HOST: _cell(0.02),
        SIZE_PRESERVING_TREATMENT: _cell(-0.001),
        TREATMENT: _cell(0.0),
    }
    decision = build_decision(cells, selection_summary=_summary())
    assert decision["outcome"] == "drop-only-historical-row-policy-restores"
    assert decision["drop_only_compatible_with_restoration"] is True
    assert decision["unique_causal_factor_claimed"] is False

    cells[TREATMENT] = _cell(-0.001)
    decision = build_decision(cells, selection_summary=_summary())
    assert decision["outcome"] == "drop-only-historical-row-policy-does-not-restore"
    assert decision["drop_only_compatible_with_restoration"] is False


def test_size_preserving_parent_must_remain_negative() -> None:
    with pytest.raises(Round0149Error, match="unexpectedly restores"):
        build_decision(
            {
                CURRENT_GRAPH_CURRENT_HOST: _cell(0.02),
                SIZE_PRESERVING_TREATMENT: _cell(0.01),
                TREATMENT: _cell(0.01),
            },
            selection_summary=_summary(),
        )


def test_drop_only_selection_is_exact_parent_prefix() -> None:
    excluded = np.arange(100, 100 + RAW_PREFIX_EXCLUDED_ROWS, dtype=np.int64)
    historical = np.concatenate((
        np.setdiff1d(
            np.arange(RAW_PREFIX_ROWS, dtype=np.int64),
            excluded,
            assume_unique=True,
        ),
        np.arange(
            RAW_PREFIX_ROWS,
            RAW_PREFIX_ROWS + RAW_PREFIX_EXCLUDED_ROWS,
            dtype=np.int64,
        ),
    ))
    arrays = {
        "historical_positions": historical,
        "pre_shuffle_positions": historical.copy(),
        "corpus_ids": np.zeros(RAW_PREFIX_ROWS, dtype=np.int8),
        "dataset_rows": historical.copy(),
        "global_rows": historical.copy(),
    }
    selected, summary = derive_drop_only_selection(
        arrays,
        parent_summary={
            "target_rows": RAW_PREFIX_ROWS,
            "raw_prefix_excluded_rows": RAW_PREFIX_EXCLUDED_ROWS,
        },
    )
    assert all(len(value) == ROWS for value in selected.values())
    assert selected["historical_positions"][-1] < RAW_PREFIX_ROWS
    assert summary["replacement_rows_beyond_raw_prefix"] == 0
    assert summary["first_parent_replacement_historical_position"] == RAW_PREFIX_ROWS


class _TinyDataset:
    shape = (7, 768)
    device = "cpu"

    def __len__(self) -> int:
        return 7

    def execution_stamp(self) -> dict:
        return {"source_representation": "fp16-control"}


def test_host_sampler_reports_dynamic_negative_universe() -> None:
    dataset = _TinyDataset()
    graph = {
        "signature": _signature("/tmp/graph.npz", "1"),
        "manifest_signature": _signature("/tmp/manifest.json", "2"),
        "sources": np.asarray([0, 1, 2], dtype=np.int32),
        "targets": np.asarray([1, 2, 3], dtype=np.int32),
        "weights": np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
        "n_nodes": 7,
    }
    wrapper = Round0104TrainingInput(
        dataset,
        graph,
        arm="fp16_control",
        required_pipeline=PIPELINE,
        expected_rows=7,
    )
    sampler = PairedHostWeightedJinaSampler(
        dataset,
        sources=graph["sources"],
        targets=graph["targets"],
        weights=graph["weights"],
        n_nodes=7,
        batch_size=20,
        pos_ratio=0.1,
        random_state=42,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        arm="fp16_control",
    )
    assert wrapper.shape == (7, 768)
    assert sampler.execution_stamp()["negative_sampling"] == (
        "uniform-7-row-universe-nonself"
    )
    left, right = sampler._rows()
    assert np.all(left < 7)
    assert np.all(right < 7)
