from __future__ import annotations

import numpy as np
import pytest

import basemap.round0107_training as training
from basemap.round0107_training import (
    CompactMappedInt8Array,
    PIPELINE,
    POSITIVE_ROWS_PER_UPDATE,
    DiverseWeightedJinaSampler,
    Round0107Error,
    Round0107TrainingInput,
    performance_windows,
    successful_update_target,
    train_config,
)


class _FakeDataset:
    def __init__(self, rows: int):
        self.shape = (rows, 768)
        self.device = "cpu"
        self._slots = [{}, {}]

    def __len__(self):
        return self.shape[0]

    def fill_pair_slot(self, slot, source, destination):
        self._slots[slot] = (
            np.asarray(source, dtype=np.int64),
            np.asarray(destination, dtype=np.int64),
        )
        return len(source)

    def transfer_pair_slot(self, slot, count):
        source, destination = self._slots[slot]
        return source[:count], destination[:count]

    def index_select(self, rows):
        return np.asarray(rows)

    def execution_stamp(self):
        return {
            "source_representation": "int8-treatment",
            "feature_residency": "synthetic",
            "endpoint_gather_calls": 0,
            "source_rows_gathered": 0,
            "destination_rows_gathered": 0,
            "host_prefetch_batches_filled": 0,
            "host_prefetch_source_rows_filled": 0,
            "host_prefetch_destination_rows_filled": 0,
        }


def _sampler(seed: int = 42):
    return DiverseWeightedJinaSampler(
        _FakeDataset(2),
        sources=np.asarray([0, 1], dtype=np.int32),
        targets=np.asarray([1, 0], dtype=np.int32),
        weights=np.asarray([0.1, 0.9], dtype=np.float32),
        n_nodes=2,
        batch_size=10,
        pos_ratio=0.5,
        random_state=seed,
        graph_signatures={"synthetic": True},
    )


def test_update_horizon_is_exact_integer_ceiling():
    assert POSITIVE_ROWS_PER_UPDATE == 409
    assert successful_update_target(409) == 1
    assert successful_update_target(410) == 2
    assert successful_update_target(750_000_000) == 1_833_741
    with pytest.raises(Round0107Error):
        successful_update_target(0)


def test_performance_window_count_tracks_dynamic_horizon():
    assert performance_windows(200) == 1
    assert performance_windows(2_700) == 1
    assert performance_windows(2_701) == 2


def test_uniform_envelope_rejection_is_weight_proportional():
    sampler = _sampler()
    draws = sampler._draw_weighted_edge_ids(100_000)
    fraction_heavy = float(np.mean(draws == 1))
    assert fraction_heavy == pytest.approx(0.9, abs=0.01)
    stamp = sampler.execution_stamp()
    assert stamp["weight_sampler"] == "uniform-envelope-rejection-max-weight-one"
    assert 0 < stamp["weight_acceptance_rate"] < 1
    assert stamp["weight_emitted_draws"] == len(draws)
    assert (
        stamp["weight_acceptances"]
        == stamp["weight_emitted_draws"] + stamp["weight_buffered_draws"]
    )


def test_weighted_rejection_reuses_surplus_acceptances():
    sampler = _sampler()
    first = sampler._draw_weighted_edge_ids(5)
    after_first = sampler.execution_stamp()
    proposals = after_first["weight_proposals"]
    second = sampler._draw_weighted_edge_ids(5)
    after_second = sampler.execution_stamp()
    assert len(first) == len(second) == 5
    assert after_first["weight_buffered_draws"] > 0
    assert after_second["weight_proposals"] == proposals
    assert after_second["weight_emitted_draws"] == 10
    assert (
        after_second["weight_acceptances"]
        == after_second["weight_emitted_draws"]
        + after_second["weight_buffered_draws"]
    )


def test_weighted_rejection_is_seed_deterministic():
    left = _sampler(seed=7)._draw_weighted_edge_ids(5_000)
    right = _sampler(seed=7)._draw_weighted_edge_ids(5_000)
    np.testing.assert_array_equal(left, right)


def test_sampler_horizon_uses_edge_count_not_weight_sum():
    sampler = _sampler()
    assert len(sampler) == successful_update_target(2)


def test_compact_feature_view_maps_ids_lazily(monkeypatch):
    monkeypatch.setattr(training, "RETAINED_ROWS", 3)
    source = np.arange(5 * 768, dtype=np.int16).astype(np.int8).reshape(5, 768)
    mapping = np.asarray([4, 1, 3], dtype=np.int64)
    view = CompactMappedInt8Array(source, mapping)
    np.testing.assert_array_equal(view[[0, 2]], source[[4, 3]])


def test_train_config_binds_topology_derived_horizon():
    graph = {
        "directed_edge_count": 1_000,
        "compact_mapping": {"sha256": "a" * 64},
        "outputs": {
            "sources": {"sha256": "b" * 64},
            "targets": {"sha256": "c" * 64},
            "weights": {"sha256": "d" * 64},
        },
    }
    config, digest = train_config(
        graph_manifest=graph,
        graph_signature={
            "kind": "file",
            "canonical_path": "/data/synthetic-graph.json",
            "bytes": 1,
            "sha256": "e" * 64,
        },
    )
    assert config["optimizer"]["successful_positive_lr_updates"] == 3
    assert config["execution"]["required_pipeline"] == PIPELINE
    assert len(digest) == 64


def test_adapter_rejects_uniform_sampling_request():
    dataset = _FakeDataset(2)
    graph = {
        "signature": {
            "canonical_path": "/data/synthetic-graph.json",
        },
        "graph_signatures": {},
        "mapping_signature": {},
        "sources": np.asarray([0, 1], dtype=np.int32),
        "targets": np.asarray([1, 0], dtype=np.int32),
        "weights": np.asarray([0.5, 0.5], dtype=np.float32),
    }
    wrapper = object.__new__(Round0107TrainingInput)
    wrapper.dataset = dataset
    wrapper.graph = graph
    wrapper.required_pipeline = PIPELINE
    with pytest.raises(Round0107Error, match="pipeline request"):
        wrapper.prepare_round0034_training(
            edges_path="/data/synthetic-graph.json",
            batch_size=10,
            pos_ratio=0.5,
            random_state=42,
            positive_target_mode="binary",
            weighted_edge_sampling=False,
            reject_neighbors=False,
            required_input_pipeline=PIPELINE,
        )
