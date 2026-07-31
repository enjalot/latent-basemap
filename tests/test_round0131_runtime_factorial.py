from __future__ import annotations

import copy
import json
import os

import numpy as np
import pytest

from basemap.artifact_identity import canonical_json, expected_input_signature, sha256_bytes
from basemap.round0125_runtime_bridge import (
    BATCH_SIZE,
    DEVICE_ARM,
    GRAPH_EDGES,
    HOST_ARM,
    STREAM_TRACE_BATCHES,
    SUCCESSFUL_UPDATES,
    expected_device_endpoint_accounting,
    train_config as r0125_train_config,
)
from basemap.round0131_runtime_factorial import (
    ARMS,
    MATCHED_DENSITY_FLOOR,
    PIPELINES,
    POSITIVE_R0125_OUTCOMES,
    RESIDENT_FUSED,
    RESIDENT_SEPARATE,
    Round0131Error,
    Round0131TrainingInput,
    causal_execution_checks,
    classify_interval,
    select_outcome,
    train_config,
)
from experiments import round0131_nodes
from experiments.smoke_round0131_cpu import run_smoke


POSITIVE = "device-path-restores-density-without-native-regression-at-seed42"


def _signature(path: str, marker: str) -> dict[str, object]:
    return {
        "kind": "file",
        "canonical_path": path,
        "bytes": 1,
        "sha256": marker * 64,
    }


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
        for _ in range(STREAM_TRACE_BATCHES):
            left, right, labels = next(iterator)
            batches.append((left.numpy(), right.numpy(), labels.numpy()))
        # Production closes (and therefore drains) the async producer before
        # sealing the runtime stamp.  Doing the same here makes the bounded
        # trace independent of whether the producer has filled its next slot
        # at the instant this thread reaches the assertion.
        sampler.close()
        receipts[arm] = sampler.execution_stamp()
        if arm == RESIDENT_FUSED:
            reference = batches
        else:
            for observed, expected in zip(batches, reference, strict=True):
                for observed_array, expected_array in zip(observed, expected, strict=True):
                    assert np.array_equal(observed_array, expected_array)
    assert all(
        receipt["stream_trace"]["batches_hashed"] == STREAM_TRACE_BATCHES
        for receipt in receipts.values()
    )
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


def _causal_fixture():
    graph = _signature("/sealed/graph.npz", "a")
    manifest = _signature("/sealed/graph.manifest.json", "b")
    configs = {
        HOST_ARM: r0125_train_config(
            HOST_ARM,
            graph_signature=graph,
            graph_manifest_signature=manifest,
        )[0],
        RESIDENT_FUSED: train_config(
            RESIDENT_FUSED,
            graph_signature=graph,
            graph_manifest_signature=manifest,
        )[0],
        RESIDENT_SEPARATE: train_config(
            RESIDENT_SEPARATE,
            graph_signature=graph,
            graph_manifest_signature=manifest,
        )[0],
        DEVICE_ARM: r0125_train_config(
            DEVICE_ARM,
            graph_signature=graph,
            graph_manifest_signature=manifest,
        )[0],
    }
    accounting = {
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": GRAPH_EDGES,
    }
    numpy_trace = {
        "schema": "round0125-first-batches-endpoint-stream-v1",
        "requested_batches": STREAM_TRACE_BATCHES,
        "batches_hashed": STREAM_TRACE_BATCHES,
        "source_endpoint_ids_sha256": "c" * 64,
        "destination_endpoint_ids_sha256": "d" * 64,
    }
    path_rows = {
        HOST_ARM: SUCCESSFUL_UPDATES * BATCH_SIZE,
        RESIDENT_FUSED: SUCCESSFUL_UPDATES * BATCH_SIZE,
        RESIDENT_SEPARATE: SUCCESSFUL_UPDATES * BATCH_SIZE,
        DEVICE_ARM: expected_device_endpoint_accounting()["endpoint_rows_per_side"],
    }
    trains = {}
    for arm, config in configs.items():
        runtime = {
            **config["execution"]["expected_pipeline_stamp"],
            "graph": graph,
            "graph_manifest": manifest,
            "source_rows_gathered": path_rows[arm],
            "destination_rows_gathered": path_rows[arm],
            "stream_trace": (
                copy.deepcopy(numpy_trace)
                if arm != DEVICE_ARM
                else {
                    **numpy_trace,
                    "source_endpoint_ids_sha256": "e" * 64,
                    "destination_endpoint_ids_sha256": "f" * 64,
                }
            ),
        }
        trains[arm] = {
            "initial_model_state_sha256": "1" * 64,
            "environment_freeze_sha256": "2" * 64,
            "causal_invariant_sha256": sha256_bytes(
                canonical_json(config["causal_invariant"])
            ),
            "graph": graph,
            "graph_manifest": manifest,
            "train_accounting": copy.deepcopy(accounting),
            "exact_execution_receipt": runtime,
        }
    return trains, configs


def test_all_four_causal_execution_contract_is_valid():
    trains, configs = _causal_fixture()
    checks = causal_execution_checks(
        active_environment_sha256="2" * 64,
        trains=trains,
        configs=configs,
    )
    assert all(checks.values()), checks


@pytest.mark.parametrize(
    ("drift", "failed_check"),
    [
        ("update0", "all_four_update0_model_hashes_equal"),
        ("environment", "cross_round_environment_equal"),
        ("graph", "normalized_graph_invariant_equal"),
        ("model", "normalized_model_invariant_equal"),
        ("optimizer", "normalized_optimizer_invariant_equal"),
        ("dose", "normalized_registered_and_observed_dose_equal"),
        ("numpy_stream", "h_r_f_first8_numpy_endpoint_streams_equal"),
        (
            "device_same_stream",
            "d_first8_device_endpoint_stream_is_valid_and_distinct",
        ),
        (
            "device_invalid_stream",
            "d_first8_device_endpoint_stream_is_valid_and_distinct",
        ),
        ("path", "registered_h_r_f_d_path_shape"),
        ("pipeline", "observed_pipeline_stamps_match_configs"),
        ("endpoint_rows", "endpoint_rows_match_registered_path"),
    ],
)
def test_each_cross_round_causal_drift_fails_closed(drift, failed_check):
    trains, configs = _causal_fixture()
    if drift == "update0":
        trains[DEVICE_ARM]["initial_model_state_sha256"] = "3" * 64
    elif drift == "environment":
        trains[HOST_ARM]["environment_freeze_sha256"] = "4" * 64
    elif drift == "graph":
        changed = _signature("/sealed/other-graph.npz", "9")
        config = configs[DEVICE_ARM]
        config["causal_invariant"]["graph"] = changed
        trains[DEVICE_ARM]["causal_invariant_sha256"] = sha256_bytes(
            canonical_json(config["causal_invariant"])
        )
        trains[DEVICE_ARM]["graph"] = changed
        trains[DEVICE_ARM]["exact_execution_receipt"]["graph"] = changed
    elif drift == "model":
        configs[DEVICE_ARM]["model"]["hidden_dimension"] = 1024
    elif drift == "optimizer":
        configs[DEVICE_ARM]["optimizer"]["learning_rate"] = 0.002
    elif drift == "dose":
        trains[RESIDENT_FUSED]["train_accounting"][
            "positive_lr_optimizer_steps"
        ] -= 1
    elif drift == "numpy_stream":
        trains[RESIDENT_FUSED]["exact_execution_receipt"]["stream_trace"][
            "source_endpoint_ids_sha256"
        ] = "8" * 64
    elif drift == "device_same_stream":
        trains[DEVICE_ARM]["exact_execution_receipt"]["stream_trace"] = (
            copy.deepcopy(
                trains[HOST_ARM]["exact_execution_receipt"]["stream_trace"]
            )
        )
    elif drift == "device_invalid_stream":
        trains[DEVICE_ARM]["exact_execution_receipt"]["stream_trace"][
            "destination_endpoint_ids_sha256"
        ] = "not-a-sha256"
    elif drift == "path":
        trains[RESIDENT_FUSED]["exact_execution_receipt"]["endpoint_forward"] = (
            "separate-source-destination"
        )
    elif drift == "pipeline":
        trains[HOST_ARM]["exact_execution_receipt"]["positive_sampling"] = (
            "uniform_with_replacement"
        )
    elif drift == "endpoint_rows":
        trains[RESIDENT_SEPARATE]["exact_execution_receipt"][
            "source_rows_gathered"
        ] -= 1
    checks = causal_execution_checks(
        active_environment_sha256="2" * 64,
        trains=trains,
        configs=configs,
    )
    assert checks[failed_check] is False
    assert not all(checks.values())


def _write_sealed(path, body):
    receipt = round0131_nodes._seal(body)
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return expected_input_signature(str(path))


def _endpoint_chain(tmp_path, monkeypatch):
    graph = _signature("/sealed/graph.npz", "a")
    manifest = _signature("/sealed/graph.manifest.json", "b")
    shared_signature = _signature("/sealed/shared.json", "c")
    environment = "2" * 64
    train_signatures = {}
    config_signatures = {}
    for arm in (HOST_ARM, DEVICE_ARM):
        config, config_sha = r0125_train_config(
            arm,
            graph_signature=graph,
            graph_manifest_signature=manifest,
        )
        config_path = tmp_path / f"{arm}-config.json"
        config_signatures[arm] = _write_sealed(
            config_path,
            {
                "schema": "round0125-production-config-v1",
                "round_id": "0125",
                "arm": arm,
                "config": config,
                "config_sha256": config_sha,
            },
        )
        train_path = tmp_path / f"{arm}-train.json"
        train_signatures[arm] = _write_sealed(
            train_path,
            {
                "schema": "round0125-runtime-arm-train-receipt-v1",
                "round_id": "0125",
                "arm": arm,
                "release_sha": round0131_nodes.R0125_RELEASE_SHA,
                "production_config": config_signatures[arm],
                "production_config_sha256": config_sha,
                "causal_invariant_sha256": config["causal_invariant_sha256"],
                "initial_model_state_sha256": "1" * 64,
                "shared_evidence": shared_signature,
                "graph": graph,
                "graph_manifest": manifest,
                "environment_freeze_sha256": environment,
                "train_checks": {
                    key: True for key in round0131_nodes.R0125_TRAIN_CHECK_KEYS
                },
            },
        )
    native_signatures = {}
    for arm in (HOST_ARM, DEVICE_ARM):
        score_path = tmp_path / f"{arm}-score.json"
        native_signatures[arm] = _write_sealed(
            score_path,
            {
                "schema": "round0125-native-runtime-arm-score-v1",
                "round_id": "0125",
                "arm": arm,
                "release_sha": round0131_nodes.R0125_RELEASE_SHA,
                "train_receipt": train_signatures[arm],
                "shared_evidence": shared_signature,
                "execution_gates": {
                    key: True for key in round0131_nodes.R0125_NATIVE_GATE_KEYS
                },
            },
        )
    panel_path = tmp_path / "panel.json"
    panel_signature = _write_sealed(
        panel_path,
        {
            "schema": "round0125-matched-runtime-density-panel-v1",
            "round_id": "0125",
            "release_sha": round0131_nodes.R0125_RELEASE_SHA,
            "cells": {
                arm: {"train_receipt": train_signatures[arm]}
                for arm in (HOST_ARM, DEVICE_ARM)
            },
            "train_receipts": train_signatures,
        },
    )
    decision = {
        "native_scores": native_signatures,
        "matched_density_panel": panel_signature,
        "outcome": POSITIVE,
    }
    monkeypatch.setattr(
        round0131_nodes,
        "_validate_positive_trigger",
        lambda _job: (decision, _signature("/sealed/decision.json", "d")),
    )
    monkeypatch.setattr(
        round0131_nodes,
        "_load_shared_exact",
        lambda _job: (
            {
                "graph": graph,
                "graph_manifest": manifest,
                "graph_edges": GRAPH_EDGES,
            },
            shared_signature,
        ),
    )
    active = {"manifest": {"environment_freeze": {"freeze_sha256": environment}}}
    job = {
        "r0125_native_scores": native_signatures,
        "r0125_panel": panel_signature,
        "r0125_train_receipts": train_signatures,
        "r0125_train_configs": config_signatures,
    }
    return active, job, decision


def test_endpoint_train_receipts_are_bound_through_scores_and_panel(
    tmp_path, monkeypatch
):
    active, job, _decision = _endpoint_chain(tmp_path, monkeypatch)
    trains, configs, scores, panel = round0131_nodes._authenticate_r0125_endpoints(
        active, job
    )
    assert set(trains) == {HOST_ARM, DEVICE_ARM}
    assert set(configs) == {HOST_ARM, DEVICE_ARM}
    assert set(scores) == {HOST_ARM, DEVICE_ARM}
    assert panel["train_receipts"] == job["r0125_train_receipts"]


def test_native_score_cannot_rebind_an_endpoint_train(tmp_path, monkeypatch):
    active, job, decision = _endpoint_chain(tmp_path, monkeypatch)
    score_path = tmp_path / "host_control-score-rebound.json"
    job["r0125_native_scores"][HOST_ARM] = _write_sealed(
        score_path,
        {
            "schema": "round0125-native-runtime-arm-score-v1",
            "round_id": "0125",
            "arm": HOST_ARM,
            "release_sha": round0131_nodes.R0125_RELEASE_SHA,
            "train_receipt": job["r0125_train_receipts"][DEVICE_ARM],
            "shared_evidence": _signature("/sealed/shared.json", "c"),
            "execution_gates": {
                key: True for key in round0131_nodes.R0125_NATIVE_GATE_KEYS
            },
        },
    )
    decision["native_scores"] = job["r0125_native_scores"]
    with pytest.raises(Round0131Error, match="native score lineage changed"):
        round0131_nodes._authenticate_r0125_endpoints(active, job)


def test_matched_panel_cannot_rebind_an_endpoint_train(tmp_path, monkeypatch):
    active, job, decision = _endpoint_chain(tmp_path, monkeypatch)
    panel_path = tmp_path / "panel-rebound.json"
    rebound = copy.deepcopy(job["r0125_train_receipts"])
    rebound[HOST_ARM] = rebound[DEVICE_ARM]
    panel_signature = _write_sealed(
        panel_path,
        {
            "schema": "round0125-matched-runtime-density-panel-v1",
            "round_id": "0125",
            "release_sha": round0131_nodes.R0125_RELEASE_SHA,
            "cells": {
                arm: {"train_receipt": rebound[arm]}
                for arm in (HOST_ARM, DEVICE_ARM)
            },
            "train_receipts": rebound,
        },
    )
    job["r0125_panel"] = panel_signature
    decision["matched_density_panel"] = panel_signature
    with pytest.raises(Round0131Error, match="matched panel no longer binds"):
        round0131_nodes._authenticate_r0125_endpoints(active, job)


def test_cuda_hidden_train_seal_reload_panel_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    output = tmp_path / "smoke.json"
    receipt = run_smoke(release_sha="f" * 40, output_path=str(output))
    assert receipt["outcome"] == "passed"
    assert all(receipt["checks"].values())
    assert output.is_file()
