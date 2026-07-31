from __future__ import annotations

import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
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


def test_historical_query_truth_requires_exact_explicit_producer_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import basemap.panel_v2 as panel_v2

    def historical_cross_knn(
        queries, corpus, k, config, hi_dim=True, q_tile=4096, exact=True
    ):
        return np.tile(np.arange(k, dtype=np.int64), (len(queries), 1))

    monkeypatch.setattr(panel_v2, "cross_knn", historical_cross_knn)
    historical_sha = sha256_bytes(
        inspect.getsource(historical_cross_knn).encode("utf-8")
    )
    config = panel_v2.PanelV2Config(k_hit=3, overselect=2)
    key, parts = panel_v2.query_truth_key(
        corpus_identity={"sha256": "a" * 64},
        query_identity={"sha256": "b" * 64},
        cfg=config,
        k=3,
        corpus_cardinality=8,
        query_rows=2,
        dimensions=4,
        candidate_compute_backend="cuda",
    )
    neighbors = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    truth_path = panel_v2.save_query_truth(
        {
            "schema": panel_v2.QUERY_TRUTH_SCHEMA,
            "key": key,
            "key_parts": parts,
            "k": 3,
            "query_rows": 2,
            "corpus_cardinality": 8,
            "neighbors": neighbors,
            "payload_sha256": ordered_array_sha256(neighbors),
            "build_wall_s": 0.0,
        },
        str(tmp_path / "historical-truth.npz"),
    )

    def current_cross_knn(
        queries, corpus, k, config, hi_dim=True, q_tile=4096, exact=True
    ):
        return np.tile(np.arange(k - 1, -1, -1, dtype=np.int64), (len(queries), 1))

    monkeypatch.setattr(panel_v2, "cross_knn", current_cross_knn)
    with pytest.raises(ValueError, match="implementation/backend identity"):
        panel_v2.load_query_truth(
            truth_path, expected_key=key,
            expected_candidate_compute_backend="cuda",
        )
    with pytest.raises(ValueError, match="malformed"):
        panel_v2.load_query_truth(
            truth_path,
            expected_candidate_compute_backend="cuda",
            expected_producer_implementation_sha256=42,
        )
    with pytest.raises(ValueError, match="requires an explicit"):
        panel_v2.load_query_truth(
            truth_path,
            expected_producer_implementation_sha256=historical_sha,
        )
    with pytest.raises(ValueError, match="implementation/backend identity"):
        panel_v2.load_query_truth(
            truth_path,
            expected_candidate_compute_backend="cuda",
            expected_producer_implementation_sha256="f" * 64,
        )
    loaded = panel_v2.load_query_truth(
        truth_path,
        expected_key=key,
        expected_key_parts=parts,
        expected_candidate_compute_backend="cuda",
        expected_producer_implementation_sha256=historical_sha,
    )
    assert np.array_equal(loaded["neighbors"], neighbors)
    assert sha256_bytes(canonical_json(loaded["key_parts"])) == key
    with np.load(truth_path, allow_pickle=False) as archive:
        corrupted = {
            name: np.array(archive[name], copy=True) for name in archive.files
        }
    corrupted["neighbors"][0, 0] = 7
    os.chmod(truth_path, 0o644)
    with open(truth_path, "wb") as handle:
        np.savez(handle, **corrupted)
    with pytest.raises(ValueError, match="payload SHA-256 mismatch"):
        panel_v2.load_query_truth(
            truth_path,
            expected_key=key,
            expected_candidate_compute_backend="cuda",
            expected_producer_implementation_sha256=historical_sha,
        )


def test_prior_r0125_artifacts_can_name_only_the_original_release() -> None:
    active = {"manifest": {"release_sha": "c" * 40}}
    assert round0125_nodes._prior_artifact_release_sha(
        active, {}, field="train_release_sha"
    ) == "c" * 40
    assert round0125_nodes._prior_artifact_release_sha(
        active,
        {"train_release_sha": "ff5dfcde5632257aac355008a70bc330bab26bee"},
        field="train_release_sha",
    ) == "ff5dfcde5632257aac355008a70bc330bab26bee"
    with pytest.raises(Round0125Error, match="exact original release"):
        round0125_nodes._prior_artifact_release_sha(
            active,
            {"train_release_sha": "d" * 40},
            field="train_release_sha",
        )


def test_r0125_query_truth_wrapper_never_infers_historical_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import basemap.panel_v2 as panel_v2
    from basemap.round0125_runtime_bridge import (
        R0104_QUERY_TRUTH_KEY,
        R0104_QUERY_TRUTH_PRODUCER_BACKEND,
        R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256,
    )

    artifact = tmp_path / "truth.npz"
    artifact.write_bytes(b"representative truth bytes")
    signature = expected_input_signature(str(artifact))
    observed: dict[str, object] = {}

    def fake_load(path: str, **kwargs):
        observed.update({"path": path, **kwargs})
        return {
            "key_parts": {
                "policy": {
                    "implementation_sha256": (
                        R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256
                    ),
                    "candidate_compute_backend": R0104_QUERY_TRUTH_PRODUCER_BACKEND,
                }
            }
        }

    monkeypatch.setattr(panel_v2, "load_query_truth", fake_load)
    round0125_nodes._load_accepted_r0104_query_truth({
        "query_truth": signature,
        "query_truth_key": R0104_QUERY_TRUTH_KEY,
    })
    assert observed == {
        "path": str(artifact),
        "expected_key": R0104_QUERY_TRUTH_KEY,
        "expected_candidate_compute_backend": R0104_QUERY_TRUTH_PRODUCER_BACKEND,
        "expected_producer_implementation_sha256": (
            R0104_QUERY_TRUTH_PRODUCER_IMPLEMENTATION_SHA256
        ),
    }

    def wrong_policy(_path: str, **_kwargs):
        return {
            "key_parts": {
                "policy": {
                    "implementation_sha256": "f" * 64,
                    "candidate_compute_backend": R0104_QUERY_TRUTH_PRODUCER_BACKEND,
                }
            }
        }

    monkeypatch.setattr(panel_v2, "load_query_truth", wrong_policy)
    with pytest.raises(Round0125Error, match="producer changed"):
        round0125_nodes._load_accepted_r0104_query_truth({
            "query_truth": signature,
            "query_truth_key": R0104_QUERY_TRUTH_KEY,
        })


def test_correction_queue_reserves_only_the_original_cap_residual() -> None:
    from experiments.prepare_round0125_correction_queue import (
        PRIOR_GPU_WALL_S,
        RESIDUAL_GPU_CAP_HOURS,
        ROUND_GPU_CAP_S,
    )

    assert PRIOR_GPU_WALL_S == 9_246.523752104957
    assert RESIDUAL_GPU_CAP_HOURS > 900.0 / 3_600.0
    assert math.isclose(
        PRIOR_GPU_WALL_S + RESIDUAL_GPU_CAP_HOURS * 3_600.0,
        ROUND_GPU_CAP_S,
        rel_tol=0.0,
        abs_tol=1e-9,
    )


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
