"""CUDA-hidden target tests for the bounded Round 0016 wrapper."""
from __future__ import annotations

import ast
import copy
import datetime
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from basemap import gate_preparation
from basemap.artifact_identity import (canonical_json,
                                       expected_input_signature, sha256_bytes)
from basemap import round0016_service as service
from basemap.round0014_program import (NODES as ROUND0014_NODES,
                                       TRAIN_CONFIG as ROUND0014_CONFIG)
from basemap.round0016_program import (
    BASE_COMMIT, BASE_TREE, NODES, PROGRAM_INPUT_ROLES, ROUND_FILE,
    ROUND_SHA256, SEQUENCED_REVIEW_FILE, SEQUENCED_REVIEW_SHA256,
    TRAIN_CONFIG, derived_node_policy, program_policy,
    round0016_release_chain,
)
from basemap.run_controller import (_round0015_release_with_receipt,
                                    _round0016_terminal_identity)
from basemap.source_closure import (ROUND0015_RUNTIME_ENTRYPOINTS,
                                    ROUND0016_RUNTIME_ENTRYPOINTS,
                                    runtime_source_closure)


ROOT = Path(__file__).resolve().parents[1]


def test_round0016_exact_contract_and_one_commit_parent_are_bound():
    assert hashlib.sha256(Path(ROUND_FILE).read_bytes()).hexdigest() == \
        ROUND_SHA256
    assert hashlib.sha256(Path(SEQUENCED_REVIEW_FILE).read_bytes()).hexdigest() == \
        SEQUENCED_REVIEW_SHA256
    assert BASE_COMMIT == "836687a81aa8f94798098d5edfdd72e264e29d77"
    assert BASE_TREE == "3b0366e5032408c93a0eeb2cdcae480d8eaebd2d"
    release = "c" * 40
    evidence = round0016_release_chain(release)
    assert evidence["implementation_commits"] == [release]
    assert evidence["ancestry"] == [BASE_COMMIT, release]
    assert evidence["commits_after_base"] == 1


def test_round0016_science_order_canary_and_training_are_unchanged():
    predecessor = copy.deepcopy(ROUND0014_CONFIG)
    successor = copy.deepcopy(TRAIN_CONFIG)
    assert predecessor.pop("schema") == "round0014-production-config-v1"
    assert successor.pop("schema") == "round0016-production-config-v1"
    assert successor == predecessor
    assert [(n.node_id, n.dependency, n.predicted_wall_s, n.p90_wall_s,
             n.training_performed, n.output_name) for n in NODES] == [
        (n.node_id, n.dependency, n.predicted_wall_s, n.p90_wall_s,
         n.training_performed, n.output_name) for n in ROUND0014_NODES]
    assert NODES[0].node_id == "no_training_seal_canary"
    assert NODES[0].p90_wall_s == 300.0
    assert NODES[0].training_performed is False
    assert NODES[1].node_id == "train_seed42_30m"
    assert NODES[1].dependency == NODES[0].node_id
    assert NODES[1].training_performed is True
    assert TRAIN_CONFIG["optimizer"]["seed"] == 42
    assert TRAIN_CONFIG["optimizer"]["successful_positive_lr_updates"] == 500_000
    assert all(derived_node_policy(node)["retry_count"] == 0 for node in NODES)
    assert all(derived_node_policy(node)["gpu_memory_cap_mb"] == 31_488
               for node in NODES)
    assert all(derived_node_policy(node)["required_free_gb"] == 29.0
               for node in NODES)


def test_round0016_dynamic_dispatch_preserves_historical_closure_cardinality():
    historical = runtime_source_closure(str(ROOT), ROUND0015_RUNTIME_ENTRYPOINTS)
    current = runtime_source_closure(str(ROOT), ROUND0016_RUNTIME_ENTRYPOINTS)
    assert len(historical) == 47
    assert {
        "basemap/round0016_admission.py", "basemap/round0016_program.py",
        "basemap/round0016_service.py", "basemap/round0016_staging.py",
        "experiments/run_round0016_node.py",
    }.issubset(current)
    assert not any("round0016" in path for path in historical)
    tree = ast.parse((ROOT / "experiments" / "run_round0016_node.py").read_text())
    guarded = next(node for node in tree.body if isinstance(node, ast.If))
    calls = [node for node in ast.walk(guarded) if isinstance(node, ast.Call)]
    configure = next(node for node in calls if isinstance(node.func, ast.Name) and
                     node.func.id == "configure_round0016")
    main = next(node for node in calls if isinstance(node.func, ast.Name) and
                node.func.id == "main")
    assert configure.lineno < main.lineno


def _environment(tmp_path: Path) -> str:
    path = tmp_path / "environment.json"
    path.write_text(json.dumps({
        "freeze_sha256": "1" * 64,
        "identity_sha256": "2" * 64,
        "gpu_uuid": "GPU-fixture",
        "gpu_name": service.EXPECTED_GPU_NAME,
        "gpu_driver": "fixture-driver",
        "venv_path": "/fixture/.venv",
        "freeze_file": "/fixture/freeze.txt",
    }), encoding="utf-8")
    return str(path)


def _decision(tmp_path: Path) -> tuple[dict, str]:
    environment = _environment(tmp_path)
    observer = expected_input_signature("/usr/bin/nvidia-smi")
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="microseconds")
    snapshot = {
        "at": now, "gpu": "fixture-gpu-row", "compute_apps": [],
        "compute_app_records": [], "compute_pids": [],
        "free_mb": 32_597.0, "used_mb": 10.0, "total_mb": 32_607.0,
        "n_co_tenants": 0, "observer": observer,
        "gpu_uuid": "GPU-fixture", "gpu_name": service.EXPECTED_GPU_NAME,
        "gpu_driver": "fixture-driver",
    }
    body = {
        "schema": "round0016-exclusive-gpu-decision-v1",
        "captured_utc": now,
        "policy": service.POLICY,
        "environment_manifest": expected_input_signature(environment),
        "environment_identity_sha256": "2" * 64,
        "gpu": {
            "uuid": "GPU-fixture", "name": service.EXPECTED_GPU_NAME,
            "driver": "fixture-driver", "total_mib": 32_607,
            "used_mib": 10.0, "free_mib": 32_597.0,
            "compute_app_records": [], "observer": observer,
        },
        "allowed_processes": [], "service_marker": None,
        "service_reservation_mib": 0,
        "memory_reservation": {
            "job_cap_mib": service.JOB_CAP_MIB,
            "required_free_mib": service.REQUIRED_FREE_MIB,
            "device_total_mib": 32_607,
            "within_device_total": True, "free_floor_met": True,
        },
        "unknown_compute_processes": [],
        "allowed_validation": {
            "expected": [], "observed": [], "gpu_snapshot": snapshot,
        },
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}, environment


def _reseal(value: dict) -> dict:
    value.pop("identity_sha256", None)
    value["identity_sha256"] = sha256_bytes(canonical_json(value))
    return value


def test_empty_snapshot_and_empty_allowlist_pass(tmp_path):
    decision, environment = _decision(tmp_path)
    assert service.validate_exclusive_decision(
        decision, environment_manifest=environment,
        require_current=False) == decision


@pytest.mark.parametrize("mutation", [
    "compute-process", "pid-record", "co-tenant", "nonempty-allowlist",
    "service-marker", "identity-drift", "capacity", "free-floor",
])
def test_exclusive_construction_rejects_every_forbidden_variant(
        tmp_path, mutation):
    decision, environment = _decision(tmp_path)
    snapshot = decision["allowed_validation"]["gpu_snapshot"]
    if mutation == "compute-process":
        record = {"gpu_uuid": "GPU-fixture", "pid": 9001,
                  "used_memory_mb": 1.0}
        decision["gpu"]["compute_app_records"] = [copy.deepcopy(record)]
        snapshot["compute_app_records"] = [record]
    elif mutation == "pid-record":
        snapshot["compute_pids"] = [9001]
    elif mutation == "co-tenant":
        snapshot["n_co_tenants"] = 1
    elif mutation == "nonempty-allowlist":
        decision["allowed_processes"] = [{"pid": 9001}]
    elif mutation == "service-marker":
        decision["service_marker"] = "ls-serve"
    elif mutation == "identity-drift":
        decision["gpu"]["uuid"] = "GPU-other"
    elif mutation == "capacity":
        decision["gpu"]["total_mib"] = 31_487
        snapshot["total_mb"] = 31_487.0
        decision["memory_reservation"]["device_total_mib"] = 31_487
    elif mutation == "free-floor":
        decision["gpu"]["free_mib"] = service.REQUIRED_FREE_MIB - 1.0
        snapshot["free_mb"] = service.REQUIRED_FREE_MIB - 1.0
    with pytest.raises((ValueError, RuntimeError)):
        service.validate_exclusive_decision(
            _reseal(decision), environment_manifest=environment,
            require_current=False)


def test_stale_snapshot_fails_current_revalidation(tmp_path, monkeypatch):
    decision, environment = _decision(tmp_path)
    decision["captured_utc"] = "2026-07-18T00:00:00+00:00"
    _reseal(decision)
    monkeypatch.setattr(service, "_capture", lambda **_kwargs: decision)
    with pytest.raises(RuntimeError, match="stale"):
        service.validate_exclusive_decision(
            decision, environment_manifest=environment, require_current=True)


def test_gate_binding_revalidates_empty_snapshot(tmp_path, monkeypatch):
    decision, environment = _decision(tmp_path)
    path = tmp_path / "exclusive-gpu-decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    monkeypatch.setattr(service, "_capture", lambda **_kwargs: copy.deepcopy(decision))
    manifest = {
        "program_inputs": [{
            "role": "exclusive_gpu_decision",
            "signature": expected_input_signature(str(path)),
        }],
        "environment_manifest": environment,
        "allowed_processes": [],
    }
    binding = gate_preparation._round0016_exclusive_binding(
        manifest, require_current=True)
    assert binding["policy"] == service.POLICY
    assert binding["allowed_processes"] == []
    assert binding["current_identity_revalidated"] is True


def test_program_policy_counts_are_derived_and_release_path_is_unchanged():
    policy = program_policy(registered_global_inputs=151,
                            source_closure_members=52)
    assert policy["registered_global_inputs"] == 151
    assert policy["source_closure_members"] == 52
    assert policy["program_input_roles"] == len(PROGRAM_INPUT_ROLES)
    assert policy["gpu_policy"] == service.POLICY
    assert policy["allowed_processes"] == []
    source = inspect.getsource(_round0015_release_with_receipt)
    assert "_round0016" in source
    terminal = _round0016_terminal_identity({
        "controller_id": "fixture", "controller_pid": 1,
        "controller_starttime_ticks": 2,
        "queue_manifest_path": "/fixture/queue.json",
        "queue_manifest_sha256": "a" * 64,
        "queue_release_sha": "b" * 40,
        "started": "start", "finished": "finish",
        "terminal_verdict": "passed", "stop_reason": "passed",
        "required_jobs": [node.node_id for node in NODES],
        "completed_jobs": [node.node_id for node in NODES],
        "gpu_elapsed_s": 0.0,
    })
    assert terminal["schema"] == "round0016-controller-terminal-identity-v1"
