from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0104_training import (
    ROWS,
    PairedHostWeightedJinaSampler,
    Round0104Error,
)
from basemap.round0125_runtime_bridge import (
    CAPABILITY,
    DEVICE_ARM,
    HOST_ARM,
    R0104_ACCEPTED_METRICS,
    Round0125DeviceTrainingInput,
    Round0125Error,
    expected_device_endpoint_accounting,
    select_outcome,
    train_config,
    validate_environment_freeze,
    validate_seal,
)
from experiments import prepare_round0125_queue
from experiments import round0125_nodes
from experiments.smoke_round0125_cpu import run_smoke


def _signature(path: str, marker: str) -> dict[str, object]:
    return {
        "kind": "file",
        "canonical_path": path,
        "bytes": 1,
        "sha256": marker * 64,
    }


def test_production_nondivisible_device_epoch_accounting_is_exact() -> None:
    accounting = expected_device_endpoint_accounting()
    assert accounting == {
        "positive_rows": 204_499_774,
        "negative_rows": 3_891_500_000,
        "endpoint_rows_per_side": 4_095_999_774,
        "batches_per_epoch": 369_690,
        "short_last_batch_positive_rows": 183,
        "shortfall_per_completed_epoch": 226,
        "completed_epoch_boundaries": 1,
    }


def test_arm_configs_differ_only_in_registered_execution_bundle() -> None:
    graph = _signature("/smoke/graph.npz", "a")
    manifest = _signature("/smoke/graph.manifest.json", "b")
    device, _ = train_config(
        DEVICE_ARM,
        graph_signature=graph,
        graph_manifest_signature=manifest,
    )
    host, _ = train_config(
        HOST_ARM,
        graph_signature=graph,
        graph_manifest_signature=manifest,
    )
    assert device["causal_invariant"] == host["causal_invariant"]
    assert device["causal_invariant_sha256"] == host["causal_invariant_sha256"]
    assert device["model"] == host["model"]
    assert device["optimizer"] == host["optimizer"]
    assert device["arm"] != host["arm"]
    assert device["execution"]["required_pipeline"] == "device"
    assert host["execution"]["required_pipeline"] == "host_weighted_jina_paired"
    assert device["execution"]["expected_pipeline_stamp"]["sampler_class"] == (
        "DeviceEdgeSampler"
    )
    assert host["execution"]["expected_pipeline_stamp"]["sampler_class"] == (
        "PairedHostWeightedJinaSampler"
    )


def test_issued_capability_and_train_receipt_contract_are_exact() -> None:
    expected = "jina-fineweb-2m-runtime-path-density-bridge-v1"
    assert CAPABILITY == expected
    assert prepare_round0125_queue.CAPABILITY == expected
    assert round0125_nodes.CAPABILITY == expected
    release = "a" * 40
    valid = {
        "schema": "round0125-runtime-arm-train-receipt-v1",
        "round_id": "0125",
        "arm": DEVICE_ARM,
        "release_sha": release,
        "train_checks": {key: True for key in round0125_nodes.TRAIN_CHECK_KEYS},
    }
    assert round0125_nodes._train_receipt_contract(
        valid, arm=DEVICE_ARM, release_sha=release
    )
    assert not round0125_nodes._train_receipt_contract(
        {**valid, "train_checks": {}},
        arm=DEVICE_ARM,
        release_sha=release,
    )
    assert not round0125_nodes._train_receipt_contract(
        {**valid, "release_sha": "b" * 40},
        arm=DEVICE_ARM,
        release_sha=release,
    )


def test_matched_coordinate_geometry_fails_closed_on_collapse() -> None:
    healthy = np.column_stack(
        (np.linspace(-1.0, 1.0, 32), np.linspace(1.0, 3.0, 32) ** 2)
    ).astype(np.float32)
    observed = round0125_nodes._matched_axis_standard_deviation(healthy)
    assert observed.shape == (2,)
    assert np.all(observed > round0125_nodes.MATCHED_COORDINATE_STD_MIN)
    with pytest.raises(Round0125Error, match="coordinates collapsed"):
        round0125_nodes._matched_axis_standard_deviation(
            np.zeros((32, 2), dtype=np.float32)
        )


@pytest.mark.parametrize(
    ("host_density", "device_density", "ci", "execution", "scale", "outcome"),
    [
        (
            0.15, 0.19, (0.01, 0.06), True, 1.0,
            "device-path-restores-density-without-native-regression-at-seed42",
        ),
        (
            0.15, 0.19, (0.01, 0.06), True, 0.90,
            "device-path-restores-density-but-regresses-native-panel-at-seed42",
        ),
        (
            0.15, 0.16, (-0.01, 0.03), True, 1.0,
            "device-path-not-sufficient-at-seed42",
        ),
        (
            0.18, 0.19, (0.001, 0.03), True, 1.0,
            "historical-host-baseline-not-reproduced",
        ),
        (
            0.15, 0.19, (-0.01, 0.06), True, 1.0,
            "device-path-effect-inconclusive-at-seed42",
        ),
        (
            0.15, 0.19, (0.01, 0.06), False, 1.0,
            "invalid-execution",
        ),
    ],
)
def test_selector_branch_order(
    host_density: float,
    device_density: float,
    ci: tuple[float, float],
    execution: bool,
    scale: float,
    outcome: str,
) -> None:
    host = dict(R0104_ACCEPTED_METRICS)
    device = {key: value * scale for key, value in host.items()}
    result = select_outcome(
        host_metrics=host,
        device_metrics=device,
        host_matched_density=host_density,
        device_matched_density=device_density,
        paired_delta_ci99=ci,
        execution_valid=execution,
    )
    assert result["outcome"] == outcome
    assert result["single_seed_path_bundle_only"] is True


class _LengthOnlyDataset:
    device = "cpu"

    def __init__(self, rows: int) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return self.rows


def test_r0104_sampler_keeps_production_default_but_allows_bounded_smoke() -> None:
    kwargs = {
        "sources": np.asarray([0, 1, 2], dtype=np.int32),
        "targets": np.asarray([1, 2, 3], dtype=np.int32),
        "weights": np.asarray([1.0, 0.5, 0.25], dtype=np.float32),
        "n_nodes": 4,
        "batch_size": 8,
        "pos_ratio": 0.25,
        "random_state": 42,
        "graph_signature": _signature("/smoke/graph", "c"),
        "graph_manifest_signature": _signature("/smoke/manifest", "d"),
        "arm": "fp16_control",
    }
    with pytest.raises(Round0104Error, match="graph/dataset is invalid"):
        PairedHostWeightedJinaSampler(_LengthOnlyDataset(4), **kwargs)
    sampler = PairedHostWeightedJinaSampler(
        _LengthOnlyDataset(4), expected_rows=4, **kwargs
    )
    assert sampler.expected_rows == 4
    assert ROWS == 2_000_000


def test_device_adapter_uses_actual_sampler_and_rejects_threshold_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.arange(32, dtype=np.float32).reshape(8, 4)
    graph = {
        "sources": np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        "targets": np.asarray([1, 2, 3, 4, 5], dtype=np.int32),
        "weights": np.linspace(0.2, 1.0, 5, dtype=np.float32),
        "n_nodes": 8,
        "signature": _signature("/smoke/graph", "e"),
        "manifest_signature": _signature("/smoke/manifest", "f"),
    }
    wrapper = Round0125DeviceTrainingInput(
        source, graph, device="cpu", expected_rows=8
    )
    _dataset, sampler, _edges, runtime, _verified = (
        wrapper.prepare_round0034_training(
            edges_path="/smoke/graph",
            batch_size=8,
            pos_ratio=0.25,
            random_state=42,
            positive_target_mode="binary",
            weighted_edge_sampling=True,
            reject_neighbors=False,
            required_input_pipeline="device",
        )
    )
    assert type(sampler).__name__ == "DeviceEdgeSampler"
    assert sampler._per_batch is False
    assert runtime["full_epoch_weighted_draw"] is True

    monkeypatch.setenv("PER_BATCH_EDGE_THRESHOLD", "1")
    changed = Round0125DeviceTrainingInput(
        source, graph, device="cpu", expected_rows=8
    )
    with pytest.raises(Round0125Error, match="must be unset"):
        changed.prepare_round0034_training(
            edges_path="/smoke/graph",
            batch_size=8,
            pos_ratio=0.25,
            random_state=42,
            positive_target_mode="binary",
            weighted_edge_sampling=True,
            reject_neighbors=False,
            required_input_pipeline="device",
        )


def test_environment_freeze_fails_closed_on_job_boundary_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {"freeze_sha256": "a" * 64}
    monkeypatch.setattr(
        "basemap.round0125_runtime_bridge.environment_freeze_receipt",
        lambda: {"freeze_sha256": "b" * 64},
    )
    with pytest.raises(Round0125Error, match="environment changed"):
        validate_environment_freeze(expected)


def test_queue_preparation_refuses_wrong_python_before_writes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prepare_round0125_queue, "RUN_PYTHON", "/wrong/python")
    with pytest.raises(RuntimeError, match="dedicated run environment"):
        prepare_round0125_queue._require_dedicated_run_environment()


def test_cuda_hidden_train_seal_reload_mini_panels_and_preflight_binding(
    tmp_path: Path,
) -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    release = "a" * 40
    receipt_path = run_smoke(
        release_sha=release, output_root=str(tmp_path / "preflight")
    )
    signature = expected_input_signature(receipt_path)
    with open(receipt_path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    validate_seal(receipt, label="R0125 test preflight")
    assert receipt["outcome"] == "passed"
    assert all(receipt["checks"].values())
    observed, sources = prepare_round0125_queue._cpu_smoke_receipt(
        receipt_path, signature["sha256"], release_sha=release
    )
    assert observed == signature
    assert len(sources) == 5
