from __future__ import annotations

import os

import numpy as np
import pytest

from basemap.round0131_runtime_factorial import (
    ARMS,
    MATCHED_DENSITY_FLOOR,
    PIPELINES,
    POSITIVE_R0125_OUTCOMES,
    RESIDENT_FUSED,
    RESIDENT_SEPARATE,
    Round0131Error,
    Round0131TrainingInput,
    classify_interval,
    select_outcome,
    train_config,
)
from experiments.smoke_round0131_cpu import run_smoke


POSITIVE = "device-path-restores-density-without-native-regression-at-seed42"


def _graph(tmp_path, *, rows: int = 23):
    sources = np.concatenate((np.arange(rows), np.arange(rows))).astype(np.int32)
    targets = np.concatenate(
        ((np.arange(rows) + 1) % rows, (np.arange(rows) + 3) % rows)
    ).astype(np.int32)
    weights = np.linspace(0.1, 1.0, len(sources), dtype=np.float32)
    path = tmp_path / "edges.npz"
    np.savez(path, sources=sources, targets=targets, weights=weights, n_nodes=rows)
    signature = {
        "canonical_path": str(path),
        "kind": "file",
        "bytes": os.path.getsize(path),
        "sha256": "a" * 64,
    }
    return {
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": rows,
        "signature": signature,
        "manifest_signature": signature,
    }


def _selector(**updates):
    values = {
        "r0125_outcome": POSITIVE,
        "correlations": {
            "host_control": 0.15,
            RESIDENT_FUSED: 0.16,
            RESIDENT_SEPARATE: 0.17,
            "device_treatment": 0.20,
        },
        "adjacent_ci99": {
            "residency": (-0.01, 0.03),
            "endpoint_forward": (-0.01, 0.03),
            "sampler_rng_epoch": (0.01, 0.05),
        },
        "execution_valid": True,
    }
    values.update(updates)
    return select_outcome(**values)


def test_registered_positive_r0125_outcomes_are_exact():
    assert POSITIVE_R0125_OUTCOMES == {
        "device-path-restores-density-without-native-regression-at-seed42",
        "device-path-restores-density-but-regresses-native-panel-at-seed42",
    }


def test_train_configs_change_only_forward_arm_execution_fields(tmp_path):
    graph = _graph(tmp_path)
    configs = {
        arm: train_config(
            arm,
            graph_signature=graph["signature"],
            graph_manifest_signature=graph["manifest_signature"],
            graph_edges=len(graph["sources"]),
        )[0]
        for arm in ARMS
    }
    assert configs[RESIDENT_FUSED]["causal_invariant"] == configs[RESIDENT_SEPARATE]["causal_invariant"]
    assert configs[RESIDENT_FUSED]["execution"]["required_pipeline"] == PIPELINES[RESIDENT_FUSED]
    assert configs[RESIDENT_SEPARATE]["execution"]["required_pipeline"] == PIPELINES[RESIDENT_SEPARATE]
    fused = configs[RESIDENT_FUSED]["execution"]["expected_pipeline_stamp"]
    separate = configs[RESIDENT_SEPARATE]["execution"]["expected_pipeline_stamp"]
    differing = {key for key in fused if fused[key] != separate[key]}
    assert differing == {"pipeline", "component_arm", "endpoint_forward"}


def test_two_intermediate_adapters_emit_same_bounded_stream(tmp_path):
    rows = 23
    graph = _graph(tmp_path, rows=rows)
    source = np.random.default_rng(7).normal(size=(rows, 768)).astype(np.float32)
    receipts = {}
    samplers = []
    for arm in ARMS:
        training = Round0131TrainingInput(
            source, graph, arm=arm, device="cpu", expected_rows=rows
        )
        _dataset, sampler, n_pos, _runtime, _verified = (
            training.prepare_round0034_training(
                edges_path=str(tmp_path / "edges.npz"),
                batch_size=16,
                pos_ratio=0.25,
                random_state=42,
                positive_target_mode="binary",
                weighted_edge_sampling=True,
                reject_neighbors=False,
                required_input_pipeline=PIPELINES[arm],
            )
        )
        assert n_pos == 46
        batches = []
        iterator = iter(sampler)
        for _ in range(3):
            left, right, labels = next(iterator)
            batches.append((left.numpy(), right.numpy(), labels.numpy()))
        receipts[arm] = sampler.execution_stamp()
        samplers.append(sampler)
        if arm == RESIDENT_FUSED:
            reference = batches
        else:
            for observed, expected in zip(batches, reference, strict=True):
                for observed_array, expected_array in zip(observed, expected, strict=True):
                    assert np.array_equal(observed_array, expected_array)
    for sampler in samplers:
        sampler.close()
    assert receipts[RESIDENT_FUSED]["stream_trace"] == receipts[RESIDENT_SEPARATE]["stream_trace"]
    assert receipts[RESIDENT_FUSED]["endpoint_forward"] == "fused-source-destination"
    assert receipts[RESIDENT_SEPARATE]["endpoint_forward"] == "separate-source-destination"


@pytest.mark.parametrize(
    ("interval", "expected"),
    [((0.001, 0.1), "reliably-positive"), ((-0.1, 0.0), "nonpositive"), ((-0.1, 0.1), "unresolved")],
)
def test_interval_classification(interval, expected):
    assert classify_interval(interval) == expected


def test_selector_localizes_first_resolved_residency_transition():
    result = _selector(
        correlations={
            "host_control": 0.15,
            RESIDENT_FUSED: MATCHED_DENSITY_FLOOR + 0.01,
            RESIDENT_SEPARATE: 0.19,
            "device_treatment": 0.20,
        },
        adjacent_ci99={
            "residency": (0.01, 0.05),
            "endpoint_forward": (-0.01, 0.02),
            "sampler_rng_epoch": (-0.01, 0.02),
        },
    )
    assert result["outcome"] == "residency-transition-is-first-resolved-restoration"


def test_selector_localizes_first_resolved_forward_transition():
    result = _selector(
        correlations={
            "host_control": 0.15,
            RESIDENT_FUSED: 0.16,
            RESIDENT_SEPARATE: MATCHED_DENSITY_FLOOR + 0.01,
            "device_treatment": 0.20,
        },
        adjacent_ci99={
            "residency": (-0.01, 0.03),
            "endpoint_forward": (0.01, 0.05),
            "sampler_rng_epoch": (-0.01, 0.03),
        },
    )
    assert result["outcome"] == "endpoint-forward-transition-is-first-resolved-restoration"


def test_selector_localizes_first_resolved_sampler_transition():
    result = _selector()
    assert result["outcome"] == "sampler-rng-transition-is-first-resolved-restoration"


def test_selector_preserves_inconclusive_and_invalid_outcomes():
    inconclusive = _selector(
        adjacent_ci99={
            "residency": (-0.01, 0.03),
            "endpoint_forward": (-0.01, 0.03),
            "sampler_rng_epoch": (-0.01, 0.05),
        }
    )
    assert inconclusive["outcome"] == "runtime-component-localization-inconclusive"
    assert _selector(execution_valid=False)["outcome"] == "invalid-execution"


def test_selector_rejects_nonpositive_r0125_branch():
    with pytest.raises(Round0131Error, match="positive branch"):
        _selector(r0125_outcome="device-path-not-sufficient-at-seed42")


def test_cuda_hidden_train_seal_reload_panel_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "smoke.json"
    receipt = run_smoke(release_sha="f" * 40, output_path=str(output))
    assert receipt["outcome"] == "passed"
    assert all(receipt["checks"].values())
    assert output.is_file()

