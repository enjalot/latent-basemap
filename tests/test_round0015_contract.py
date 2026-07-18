"""Targeted CUDA-hidden regressions for the bounded Round 0015 correction."""
from __future__ import annotations

import ast
import base64
import copy
import json
import os
from pathlib import Path

import pytest

from basemap.artifact_identity import (canonical_json,
                                       expected_input_signature, sha256_bytes)
from basemap import gate_preparation
from basemap import round0015_service as service
from basemap.round0014_program import (NODES as ROUND0014_NODES,
                                       TRAIN_CONFIG as ROUND0014_CONFIG)
from basemap.round0015_program import (
    GPU_HOURS_CAP, NODES, TRAIN_CONFIG, derived_node_policy, program_policy,
)
from basemap.run_controller import (
    GpuLease, _round0015_release_with_receipt,
    _round0015_terminal_identity,
)
from basemap.source_closure import (ROUND0015_RUNTIME_ENTRYPOINTS,
                                    runtime_source_closure)


ROOT = Path(__file__).resolve().parents[1]


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
    raw_cmdline = b"/fixture/python\x00/fixture/ls-serve\x00/data\x00"
    allowed = {
        "pid": 4242,
        "proc_starttime_ticks": 10101,
        "cmdline_sha256": sha256_bytes(raw_cmdline),
        "service_identity": service.MARKER,
        "marker": service.MARKER,
        "gpu_memory_budget_mb": service.SERVICE_CAP_MIB,
    }
    observer = expected_input_signature("/usr/bin/nvidia-smi")
    records = [{
        "gpu_uuid": "GPU-fixture", "pid": 4242,
        "used_memory_mb": 738.0,
    }]
    snapshot = {
        "at": "2026-07-18T00:00:00+00:00",
        "gpu": "fixture-gpu-row", "compute_apps": ["fixture-app-row"],
        "compute_app_records": copy.deepcopy(records),
        "compute_pids": [4242], "free_mb": 31_373.0,
        "used_mb": 748.0, "total_mb": 32_607.0,
        "n_co_tenants": 1, "observer": observer,
        "gpu_uuid": "GPU-fixture", "gpu_name": service.EXPECTED_GPU_NAME,
        "gpu_driver": "fixture-driver",
    }
    body = {
        "schema": "round0015-service-decision-v1",
        "captured_utc": "2026-07-18T00:00:00+00:00",
        "policy": service.POLICY,
        "environment_manifest": expected_input_signature(environment),
        "environment_identity_sha256": "2" * 64,
        "gpu": {
            "uuid": "GPU-fixture", "name": service.EXPECTED_GPU_NAME,
            "driver": "fixture-driver", "total_mib": 32_607,
            "used_mib": 748.0, "free_mib": 31_373.0,
            "compute_app_records": copy.deepcopy(records),
            "observer": observer,
        },
        "declared_services": [{
            **allowed,
            "raw_cmdline_base64": base64.b64encode(raw_cmdline).decode("ascii"),
            "observed_vram_mib": 738.0, "gpu_uuid": "GPU-fixture",
        }],
        "allowed_processes": [copy.deepcopy(allowed)],
        "memory_reservation": {
            "service_cap_mib": service.SERVICE_CAP_MIB,
            "job_cap_mib": service.JOB_CAP_MIB,
            "combined_cap_mib": service.COMBINED_CAP_MIB,
            "device_total_mib": 32_607, "within_device_total": True,
        },
        "unknown_compute_processes": [],
        "allowed_validation": {
            "expected": [copy.deepcopy(allowed)],
            "observed": [copy.deepcopy(allowed)],
            "gpu_snapshot": snapshot,
        },
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}, environment


def _reseal(value: dict) -> dict:
    value.pop("identity_sha256", None)
    value["identity_sha256"] = sha256_bytes(canonical_json(value))
    return value


def test_round0015_is_fresh_wiring_for_the_exact_round0014_science():
    predecessor = copy.deepcopy(ROUND0014_CONFIG)
    successor = copy.deepcopy(TRAIN_CONFIG)
    assert predecessor.pop("schema") == "round0014-production-config-v1"
    assert successor.pop("schema") == "round0015-production-config-v1"
    assert successor == predecessor
    assert [(n.node_id, n.dependency, n.predicted_wall_s, n.p90_wall_s,
             n.training_performed, n.output_name) for n in NODES] == [
        (n.node_id, n.dependency, n.predicted_wall_s, n.p90_wall_s,
         n.training_performed, n.output_name) for n in ROUND0014_NODES]
    assert [node.node_id for node in NODES] == [
        "no_training_seal_canary", "train_seed42_30m", "transform_30m",
        "high_d_reference", "registered_panel", "semantic_renders",
    ]
    assert sum(node.p90_wall_s * 1.15 for node in NODES) <= \
        GPU_HOURS_CAP * 3600
    assert all(derived_node_policy(node)["gpu_memory_cap_mb"] == 31_488
               for node in NODES)
    assert all(derived_node_policy(node)["required_free_gb"] == 29.0
               for node in NODES)
    policy = program_policy()
    assert policy["service_policy"] == "allow_exact_service"
    assert policy["terminal_lease_release_required"] is True
    assert policy["retry_count"] == 0


def test_round0015_closure_and_wrapper_are_target_specific():
    closure = runtime_source_closure(str(ROOT), ROUND0015_RUNTIME_ENTRYPOINTS)
    assert {
        "basemap/round0015_admission.py", "basemap/round0015_program.py",
        "basemap/round0015_service.py", "basemap/run_controller.py",
        "basemap/panel_v2.py", "experiments/run_round0015_node.py",
    }.issubset(closure)
    tree = ast.parse((ROOT / "experiments" / "run_round0015_node.py").read_text())
    guarded = next(node for node in tree.body if isinstance(node, ast.If))
    calls = [node for node in ast.walk(guarded) if isinstance(node, ast.Call)]
    configure = next(node for node in calls if isinstance(node.func, ast.Name) and
                     node.func.id == "configure_round0015")
    main = next(node for node in calls if isinstance(node.func, ast.Name) and
                node.func.id == "main")
    assert configure.lineno < main.lineno


def test_exact_service_decision_accepts_its_complete_snapshot(tmp_path):
    decision, environment = _decision(tmp_path)
    assert service.validate_service_decision(
        decision, environment_manifest=environment,
        require_current=False) == decision


@pytest.mark.parametrize("mutation", [
    "empty-allowlist", "mismatched-allowlist", "missing-snapshot",
    "alternate-marker", "over-budget", "unknown-extra", "capacity-overflow",
])
def test_service_construction_rejects_every_demonstrated_variant(
        tmp_path, mutation):
    decision, environment = _decision(tmp_path)
    if mutation == "empty-allowlist":
        decision["allowed_processes"] = []
    elif mutation == "mismatched-allowlist":
        decision["allowed_processes"][0]["cmdline_sha256"] = "f" * 64
    elif mutation == "missing-snapshot":
        decision.pop("gpu")
    elif mutation == "alternate-marker":
        for record in (decision["declared_services"][0],
                       decision["allowed_processes"][0],
                       decision["allowed_validation"]["expected"][0],
                       decision["allowed_validation"]["observed"][0]):
            record["marker"] = "moonshine-web"
            record["service_identity"] = "moonshine-web"
    elif mutation == "over-budget":
        decision["declared_services"][0]["observed_vram_mib"] = 1025.0
        decision["gpu"]["compute_app_records"][0]["used_memory_mb"] = 1025.0
        decision["allowed_validation"]["gpu_snapshot"][
            "compute_app_records"][0]["used_memory_mb"] = 1025.0
    elif mutation == "unknown-extra":
        extra = {"gpu_uuid": "GPU-fixture", "pid": 9001,
                 "used_memory_mb": 1.0}
        decision["gpu"]["compute_app_records"].append(copy.deepcopy(extra))
        snapshot = decision["allowed_validation"]["gpu_snapshot"]
        snapshot["compute_app_records"].append(extra)
        snapshot["compute_pids"].append(9001)
        snapshot["n_co_tenants"] = 2
    elif mutation == "capacity-overflow":
        decision["memory_reservation"]["device_total_mib"] = 32_511
        decision["gpu"]["total_mib"] = 32_511
        decision["allowed_validation"]["gpu_snapshot"]["total_mb"] = 32_511.0
    with pytest.raises((ValueError, RuntimeError)):
        service.validate_service_decision(
            _reseal(decision), environment_manifest=environment,
            require_current=False)


def test_service_construction_rejects_stale_live_identity(tmp_path, monkeypatch):
    decision, environment = _decision(tmp_path)
    stale = copy.deepcopy(decision["allowed_processes"])
    stale[0]["proc_starttime_ticks"] += 1
    monkeypatch.setattr(service, "_capture", lambda **_kwargs: {
        "allowed_processes": stale})
    with pytest.raises(RuntimeError, match="stale or changed"):
        service.validate_service_decision(
            decision, environment_manifest=environment, require_current=True)


def test_gate_construction_binds_and_revalidates_the_one_snapshot(
        tmp_path, monkeypatch):
    decision, environment = _decision(tmp_path)
    path = tmp_path / "service-decision.json"
    path.write_text(json.dumps(decision), encoding="utf-8")
    monkeypatch.setattr(service, "_capture", lambda **_kwargs: {
        "allowed_processes": copy.deepcopy(decision["allowed_processes"])})
    manifest = {
        "program_inputs": [{
            "role": "service_decision",
            "signature": expected_input_signature(str(path)),
        }],
        "environment_manifest": environment,
        "allowed_processes": decision["allowed_processes"],
    }
    binding = gate_preparation._round0015_service_binding(
        manifest, require_current=True)
    assert binding["policy"] == service.POLICY
    assert binding["decision_identity_sha256"] == decision["identity_sha256"]
    assert binding["current_identity_revalidated"] is True
    assert binding["allowed_processes"] == decision["allowed_processes"]


def _terminal(controller_id: str, verdict: str) -> dict:
    summary = {
        "controller_id": controller_id, "controller_pid": os.getpid(),
        "controller_starttime_ticks": 12345,
        "queue_manifest_path": "/data/fixture/queue.json",
        "queue_manifest_sha256": "a" * 64, "queue_release_sha": "b" * 40,
        "started": "2026-07-18T00:00:00+00:00",
        "finished": "2026-07-18T00:00:01+00:00",
        "terminal_verdict": verdict,
        "stop_reason": ("every required manifest node succeeded"
                        if verdict == "passed" else "fixture node failed"),
        "required_jobs": ["fixture"],
        "completed_jobs": (["fixture"] if verdict == "passed" else []),
        "gpu_elapsed_s": 0.0,
    }
    return _round0015_terminal_identity(summary)


@pytest.mark.parametrize("verdict", ["passed", "failed"])
def test_round0015_terminal_paths_neutralize_release_and_prove_both_locks(
        tmp_path, verdict):
    lease_path = str(tmp_path / f"{verdict}.lease")
    receipt_path = str(tmp_path / f"{verdict}-release.json")
    lease = GpuLease(
        path=lease_path, timeout=0,
        controller_id=f"round0015-{verdict}").acquire()
    before = os.pread(lease.fileno(), 4096, 0)
    terminal = _terminal(f"round0015-{verdict}", verdict)
    receipt = _round0015_release_with_receipt(
        lease, receipt_path=receipt_path, terminal_identity=terminal,
        _fixture_only=True)
    body = {key: receipt[key] for key in receipt if key != "identity_sha256"}
    assert receipt["status"] == "passed"
    assert receipt["identity_sha256"] == sha256_bytes(canonical_json(body))
    assert receipt["payload_before_sha256"] == sha256_bytes(before)
    assert receipt["payload_after"]["state"] == "released"
    assert receipt["payload_after"]["terminal_identity_sha256"] == \
        terminal["identity_sha256"]
    assert receipt["lock_proof"][
        "parent_serialization_lock_nonblocking_acquired"] is True
    assert receipt["lock_proof"][
        "lease_file_ofd_lock_nonblocking_acquired"] is True
    GpuLease(path=lease_path, timeout=0).acquire().release()


def test_round0015_release_failure_still_closes_and_seals_failed_receipt(
        tmp_path, monkeypatch):
    lease_path = str(tmp_path / "write-failure.lease")
    receipt_path = str(tmp_path / "write-failure-release.json")
    lease = GpuLease(
        path=lease_path, timeout=0,
        controller_id="round0015-write-failure").acquire()
    real_pwrite = os.pwrite

    def fail_pwrite(fd, value, offset):
        if fd == lease.fileno():
            raise OSError("fixture neutralization write failure")
        return real_pwrite(fd, value, offset)

    monkeypatch.setattr(os, "pwrite", fail_pwrite)
    with pytest.raises(RuntimeError, match="terminal lease release proof failed"):
        _round0015_release_with_receipt(
            lease, receipt_path=receipt_path,
            terminal_identity=_terminal("round0015-write-failure", "failed"),
            _fixture_only=True)
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    assert receipt["status"] == "failed"
    assert "neutralization write failure" in receipt["release_error"]
    assert receipt["lock_proof"][
        "parent_serialization_lock_nonblocking_acquired"] is True
    assert receipt["lock_proof"][
        "lease_file_ofd_lock_nonblocking_acquired"] is True
    monkeypatch.setattr(os, "pwrite", real_pwrite)
    GpuLease(path=lease_path, timeout=0).acquire().release()
