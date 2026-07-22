from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0034_pipeline import (
    DEFAULT_K,
    ELIGIBILITY_SCHEMA,
    HostInt8CanonicalSampler,
    HostInt8MaterializedArray,
    Round0034PipelineError,
    Round0034TrainingInput,
    build_canonical_graph,
    coverage_aligned_successful_updates,
    load_canonical_graph,
)
from basemap.round0034_program import train_config_from_capabilities


def _eligibility(row_count: int) -> dict:
    excluded = np.asarray([0, 2], dtype=np.int64)
    return {
        "signature": {
            "canonical_path": "/synthetic/eligibility.npz",
            "kind": "file",
            "bytes": 1,
            "sha256": "e" * 64,
        },
        "metadata": {
            "schema": ELIGIBILITY_SCHEMA,
            "row_count": row_count,
            "summary": {
                "excluded_row_count": 2,
                "retained_row_count": row_count - 2,
            },
        },
        "zero_rows": np.asarray([0], dtype=np.int64),
        "excluded_rows": excluded,
        "duplicate_excluded_rows": np.asarray([2], dtype=np.int64),
        "duplicate_representative_rows": np.asarray([1], dtype=np.int64),
    }


def _graph_arrays(row_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources = np.repeat(np.arange(row_count, dtype=np.int32), DEFAULT_K)
    targets = np.empty((row_count, DEFAULT_K), dtype=np.int32)
    for row in range(row_count):
        targets[row] = (row + 1 + np.arange(DEFAULT_K)) % row_count
    # Row 1 exercises every destination-side operation. Duplicate row 2 maps
    # to representative 1 and becomes self; row 0 is zero; row 3 repeats.
    targets[1, :6] = np.asarray([2, 0, 3, 3, 4, 5], dtype=np.int32)
    weights = np.full(row_count * DEFAULT_K, 1 / DEFAULT_K, dtype=np.float32)
    return sources, targets.reshape(-1), weights


def _write_graph(path: Path, row_count: int, *, bad_sources: bool = False,
                 bad_weights: bool = False) -> None:
    sources, targets, weights = _graph_arrays(row_count)
    if bad_sources:
        sources[DEFAULT_K + 2] = 5
    if bad_weights:
        weights[-1] *= 0.5
    np.savez_compressed(
        path,
        sources=sources,
        targets=targets,
        weights=weights,
        n_nodes=np.asarray(row_count, dtype=np.int64),
        k=np.asarray(DEFAULT_K, dtype=np.int64),
        nprobe=np.asarray(32, dtype=np.int64),
    )


def test_canonical_graph_streams_source_weight_validation_and_destination_policy(
        tmp_path):
    row_count = 7
    graph_path = tmp_path / "edges.npz"
    _write_graph(graph_path, row_count)
    signature = expected_input_signature(graph_path)

    result = build_canonical_graph(
        graph_path=str(graph_path),
        expected_graph_sha256=signature["sha256"],
        eligibility=_eligibility(row_count),
        output_root=str(tmp_path / "canonical"),
        row_count=row_count,
        k=DEFAULT_K,
        block_rows=2,
    )

    loaded = load_canonical_graph(
        result["manifest"]["canonical_path"],
        expected_sha256=result["manifest"]["sha256"],
        expected_eligibility_sha256="e" * 64,
        row_count=row_count,
    )
    degrees = np.asarray(loaded["degrees"])
    targets = np.asarray(loaded["targets"])
    assert degrees[0] == 0 and degrees[2] == 0
    assert targets[0].tolist() == [-1] * DEFAULT_K
    assert targets[2].tolist() == [-1] * DEFAULT_K
    assert targets[1, :3].tolist() == [3, 4, 5]
    assert len(set(targets[1, :degrees[1]].tolist())) == int(degrees[1])
    assert all(row not in targets[row, :degrees[row]] for row in range(row_count))
    assert result["summary"]["zero_destinations_dropped"] > 0
    assert result["summary"]["self_destinations_dropped"] > 0
    assert result["summary"]["repeated_canonical_destinations_dropped"] > 0
    assert result["summary"]["duplicate_destinations_mapped"] > 0
    assert result["summary"]["valid_canonical_edge_count"] == int(degrees.sum())
    assert result["summary"]["retained_positive_source_count"] == int(
        np.count_nonzero(degrees)
    )
    for name in ("canonical-targets.i32", "valid-degrees.u8",
                 "canonical-graph-v1.json"):
        assert os.stat(tmp_path / "canonical" / name).st_mode & 0o222 == 0


@pytest.mark.parametrize("failure", ["sources", "weights"])
def test_canonical_graph_fails_closed_on_noncanonical_input(tmp_path, failure):
    graph_path = tmp_path / "edges.npz"
    _write_graph(
        graph_path,
        7,
        bad_sources=failure == "sources",
        bad_weights=failure == "weights",
    )
    signature = expected_input_signature(graph_path)
    with pytest.raises(Round0034PipelineError):
        build_canonical_graph(
            graph_path=str(graph_path),
            expected_graph_sha256=signature["sha256"],
            eligibility=_eligibility(7),
            output_root=str(tmp_path / "canonical"),
            row_count=7,
            k=DEFAULT_K,
            block_rows=2,
        )


def test_host_int8_sampler_gathers_both_endpoints_and_exact_scales():
    torch = pytest.importorskip("torch")
    row_count = 7
    encoded = np.zeros((row_count, 4), dtype=np.int8)
    encoded[:, 0] = np.arange(1, row_count + 1, dtype=np.int8)
    scales = np.asarray([0.5, 1, 2, 3, 4, 5, 6], dtype="<f2")
    dataset = HostInt8MaterializedArray(
        encoded, scales, device="cpu", buffer_rows=4
    )
    targets = np.full((row_count, DEFAULT_K), -1, dtype=np.int32)
    degrees = np.zeros(row_count, dtype=np.uint8)
    for source, destination in ((1, 3), (3, 4), (4, 5), (5, 6), (6, 1)):
        targets[source, 0] = destination
        degrees[source] = 1
    sampler = HostInt8CanonicalSampler(
        dataset,
        targets=targets,
        degrees=degrees,
        excluded_rows=np.asarray([0, 2], dtype=np.int64),
        positive_source_count=5,
        valid_edge_count=5,
        batch_size=4,
        pos_ratio=0.5,
        random_state=34,
        graph_signature={"sha256": "g" * 64},
        eligibility_signature={"sha256": "e" * 64},
    )

    source, destination, labels = next(iter(sampler))

    assert source.shape == destination.shape == (4, 4)
    assert source.dtype == destination.dtype == torch.float32
    assert labels.tolist() == [1.0, 1.0, 0.0, 0.0]
    # Every first coordinate identifies exact int8[row] * fp16_scale[row].
    valid_values = {
        float((row + 1) * np.float32(scales[row])) for row in (1, 3, 4, 5, 6)
    }
    assert set(source[:, 0].tolist()).issubset(valid_values)
    assert set(destination[:, 0].tolist()).issubset(valid_values)
    stamp = sampler.execution_stamp()
    assert stamp["x_residency"] == "host_int8_materialized"
    assert stamp["endpoint_gather_calls"] == 1
    assert stamp["source_rows_gathered"] == 4
    assert stamp["destination_rows_gathered"] == 4


def test_training_input_dispatch_and_coverage_formula(tmp_path):
    assert coverage_aligned_successful_updates(29_989_838) == 500_000
    assert coverage_aligned_successful_updates(5 * 29_989_838) == 2_500_000
    assert coverage_aligned_successful_updates(29_989_839) == 500_001

    row_count = 7
    encoded = np.ones((row_count, 4), dtype=np.int8)
    scales = np.ones(row_count, dtype="<f2")
    dataset = HostInt8MaterializedArray(
        encoded, scales, device="cpu", buffer_rows=4
    )
    targets = np.full((row_count, DEFAULT_K), -1, dtype=np.int32)
    degrees = np.zeros(row_count, dtype=np.uint8)
    for source, destination in ((1, 3), (3, 4), (4, 5), (5, 6), (6, 1)):
        targets[source, 0] = destination
        degrees[source] = 1
    manifest_path = tmp_path / "canonical.json"
    manifest_path.write_text("{}")
    graph = {
        "signature": {
            "canonical_path": str(manifest_path.resolve()),
            "sha256": "g" * 64,
        },
        "manifest": {
            "summary": {
                "retained_positive_source_count": 5,
                "valid_canonical_edge_count": 5,
                "eligibility_retained_row_count": 5,
                "eligibility_excluded_source_count": 2,
            },
            "outputs": {
                "targets": {"sha256": "t" * 64},
                "degrees": {"sha256": "d" * 64},
            },
        },
        "targets": targets,
        "degrees": degrees,
    }
    wrapper = Round0034TrainingInput(dataset, graph, _eligibility(row_count))
    _dataset, sampler, n_pos, stamp, verified = wrapper.prepare_round0034_training(
        edges_path=str(manifest_path),
        batch_size=4,
        pos_ratio=0.5,
        random_state=42,
        positive_target_mode="binary",
        weighted_edge_sampling=False,
        reject_neighbors=False,
        required_input_pipeline="host_int8_canonical",
    )
    assert n_pos == 5
    assert stamp["sampler_class"] == "HostInt8CanonicalSampler"
    assert verified["eligibility"]["sha256"] == "e" * 64
    with pytest.raises(Round0034PipelineError):
        wrapper.prepare_round0034_training(
            edges_path=str(manifest_path), batch_size=4, pos_ratio=0.5,
            random_state=42, positive_target_mode="binary",
            weighted_edge_sampling=True, reject_neighbors=False,
            required_input_pipeline="host_int8_canonical",
        )


def test_dynamic_program_and_trainer_core_dispatch(tmp_path):
    from basemap.pumap.parametric_umap import ParametricUMAP

    manifest = {
        "schema": "minilm-canonical-source-major-k15-v1",
        "row_count": 150_000_000,
        "input_k": 15,
        "inputs": {"eligibility": {"sha256": "e" * 64}},
        "summary": {
            "retained_positive_source_count": 149_000_001,
            "eligibility_retained_row_count": 149_000_001,
            "zero_degree_retained_source_count": 0,
            "zero_degree_retained_source_fraction": 0.0,
            "valid_canonical_edge_count": 2_100_000_000,
        },
    }
    config, digest = train_config_from_capabilities(
        manifest,
        canonical_graph_manifest_path="/data/synthetic/canonical.json",
        canonical_graph_manifest_sha256="g" * 64,
        eligibility_sha256="e" * 64,
    )
    assert config["optimizer"]["successful_positive_lr_updates"] == (
        500_000 * 149_000_001 + 29_989_838 - 1
    ) // 29_989_838
    assert config["execution"]["required_pipeline"] == "host_int8_canonical"
    assert len(digest) == 64

    # Focused dispatch proof on a small structural capability.
    row_count = 7
    encoded = np.ones((row_count, 4), dtype=np.int8)
    scales = np.ones(row_count, dtype="<f2")
    dataset = HostInt8MaterializedArray(encoded, scales, device="cpu", buffer_rows=4)
    targets = np.full((row_count, DEFAULT_K), -1, dtype=np.int32)
    degrees = np.zeros(row_count, dtype=np.uint8)
    for source, destination in ((1, 3), (3, 4), (4, 5), (5, 6), (6, 1)):
        targets[source, 0] = destination
        degrees[source] = 1
    graph_path = tmp_path / "canonical.json"
    graph_path.write_text("{}")
    graph = {
        "signature": {
            "canonical_path": str(graph_path.resolve()), "sha256": "g" * 64,
        },
        "manifest": {
            "summary": {
                "retained_positive_source_count": 5,
                "valid_canonical_edge_count": 5,
                "eligibility_retained_row_count": 5,
                "eligibility_excluded_source_count": 2,
            },
            "outputs": {
                "targets": {"sha256": "t" * 64},
                "degrees": {"sha256": "d" * 64},
            },
        },
        "targets": targets,
        "degrees": degrees,
    }
    wrapper = Round0034TrainingInput(dataset, graph, _eligibility(row_count))
    model = ParametricUMAP(
        device="cpu", batch_size=4, pos_ratio=0.5,
        positive_target_mode="binary", weighted_edge_sampling=False,
        reject_neighbors=False, required_input_pipeline="host_int8_canonical",
    )
    prepared_dataset, sampler, n_pos = model._prepare_edge_list_training(
        wrapper, str(graph_path), row_count, True, 42
    )
    assert prepared_dataset is wrapper
    assert isinstance(sampler, HostInt8CanonicalSampler)
    assert n_pos == 5
    assert model._fast_device_path is True
    assert model._pipeline_info["x_residency"] == "host_int8_materialized"


def test_tiny_round0034_pipeline_runs_through_real_trainer(tmp_path):
    from basemap.pumap.parametric_umap import ParametricUMAP

    row_count = 7
    encoded = np.arange(row_count * 4, dtype=np.int8).reshape(row_count, 4)
    scales = np.ones(row_count, dtype="<f2")
    dataset = HostInt8MaterializedArray(encoded, scales, device="cpu", buffer_rows=4)
    targets = np.full((row_count, DEFAULT_K), -1, dtype=np.int32)
    degrees = np.zeros(row_count, dtype=np.uint8)
    for source, destination in ((1, 3), (3, 4), (4, 5), (5, 6), (6, 1)):
        targets[source, 0] = destination
        degrees[source] = 1
    graph_path = tmp_path / "canonical.json"
    graph_path.write_text("{}")
    graph = {
        "signature": {
            "canonical_path": str(graph_path.resolve()), "sha256": "g" * 64,
        },
        "manifest": {
            "summary": {
                "retained_positive_source_count": 5,
                "valid_canonical_edge_count": 5,
                "eligibility_retained_row_count": 5,
                "eligibility_excluded_source_count": 2,
            },
            "outputs": {
                "targets": {"sha256": "t" * 64},
                "degrees": {"sha256": "d" * 64},
            },
        },
        "targets": targets,
        "degrees": degrees,
    }
    wrapper = Round0034TrainingInput(dataset, graph, _eligibility(row_count))
    model = ParametricUMAP(
        architecture="residual_bottleneck", hidden_dim=8, n_layers=2,
        n_components=2, a=1.0, b=1.0, correlation_weight=0.0,
        learning_rate=0.001, n_epochs=1, batch_size=4, pos_ratio=0.5,
        device="cpu", use_amp=False, positive_target_mode="binary",
        lr_schedule="cosine", warmup_steps=1, total_steps_estimate=2,
        required_input_pipeline="host_int8_canonical",
        weighted_edge_sampling=False, reject_neighbors=False,
    )
    model.fit(
        wrapper, precomputed_edges_path=str(graph_path),
        random_state=42, low_memory=True, verbose=False,
    )
    assert model._train_stats["budget_satisfied"] is True
    assert model._train_stats["positive_lr_optimizer_steps"] == 2
    runtime = model._train_stats["pipeline_runtime"]
    assert runtime["source_rows_gathered"] == 8
    assert runtime["destination_rows_gathered"] == 8


def test_queue_builder_refuses_a_draft_round(tmp_path):
    from experiments.prepare_round0034_queue import _assert_issued_round

    draft = tmp_path / "round.md"
    draft.write_text("---\nstatus: draft\n---\n")
    with pytest.raises(RuntimeError, match="remains draft"):
        _assert_issued_round(str(draft))
    issued = tmp_path / "issued.md"
    issued.write_text("---\nstatus: issued\n---\n")
    _assert_issued_round(str(issued))


def test_training_config_rejects_zero_degree_planning_alert():
    retained = 1_000_000
    zero_degree = 101
    manifest = {
        "schema": "minilm-canonical-source-major-k15-v1",
        "row_count": 150_000_000,
        "input_k": 15,
        "inputs": {"eligibility": {"sha256": "e" * 64}},
        "summary": {
            "retained_positive_source_count": retained - zero_degree,
            "eligibility_retained_row_count": retained,
            "zero_degree_retained_source_count": zero_degree,
            "zero_degree_retained_source_fraction": zero_degree / retained,
            "valid_canonical_edge_count": retained,
        },
    }
    with pytest.raises(ValueError, match="canonical graph capability"):
        train_config_from_capabilities(
            manifest,
            canonical_graph_manifest_path="/data/synthetic/canonical.json",
            canonical_graph_manifest_sha256="g" * 64,
            eligibility_sha256="e" * 64,
        )
