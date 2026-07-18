"""CUDA-uninitialized exclusive-GPU decision for the one Round 0016 queue."""
from __future__ import annotations

import datetime
import json
import os
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes)
from .output_safety import atomic_write_new_json, refuse_existing
from .run_controller import gpu_snapshot, validate_allowed_processes


POLICY = "exclusive_empty_gpu"
JOB_CAP_MIB = 31_488
REQUIRED_FREE_GIB = 29.0
REQUIRED_FREE_MIB = int(REQUIRED_FREE_GIB * 1024)
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"
MAX_SNAPSHOT_AGE_SECONDS = 300.0


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="microseconds")


def _environment(path: str) -> dict[str, Any]:
    canonical = os.path.realpath(path)
    if canonical != path or not os.path.isfile(path) or os.path.islink(path):
        raise ValueError("Round 0016 environment manifest is not one canonical file")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    required = {
        "freeze_sha256", "identity_sha256", "gpu_uuid", "gpu_name",
        "gpu_driver", "venv_path", "freeze_file",
    }
    if (not isinstance(value, dict) or not required.issubset(value) or
            any(not isinstance(value[key], str) or not value[key]
                for key in required) or
            value["gpu_name"] != EXPECTED_GPU_NAME):
        raise ValueError("Round 0016 environment is not the sealed RTX 5090")
    return value


def _capture(*, environment_manifest: str) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Round 0016 capture requires CUDA_VISIBLE_DEVICES=''")
    environment = _environment(environment_manifest)
    snapshot = gpu_snapshot()
    records = snapshot.get("compute_app_records")
    if (snapshot.get("gpu_uuid") != environment["gpu_uuid"] or
            snapshot.get("gpu_name") != environment["gpu_name"] or
            snapshot.get("gpu_driver") != environment["gpu_driver"] or
            not isinstance(snapshot.get("total_mb"), (int, float)) or
            not isinstance(snapshot.get("free_mb"), (int, float)) or
            not isinstance(snapshot.get("used_mb"), (int, float)) or
            not isinstance(records, list)):
        raise RuntimeError("Round 0016 live GPU identity/memory differs from the seal")
    allowed = validate_allowed_processes([], snapshot=snapshot)
    if (records != [] or snapshot.get("compute_pids") != [] or
            snapshot.get("n_co_tenants") != 0):
        raise RuntimeError("Round 0016 requires zero GPU compute processes")
    device_total = int(snapshot["total_mb"])
    free_mib = float(snapshot["free_mb"])
    if JOB_CAP_MIB > device_total:
        raise RuntimeError("Round 0016 job reservation exceeds device total")
    if free_mib < REQUIRED_FREE_MIB:
        raise RuntimeError("Round 0016 live free memory is below the 29 GiB floor")
    body = {
        "schema": "round0016-exclusive-gpu-decision-v1",
        "captured_utc": _utcnow(),
        "policy": POLICY,
        "environment_manifest": expected_input_signature(environment_manifest),
        "environment_identity_sha256": environment["identity_sha256"],
        "gpu": {
            "uuid": snapshot["gpu_uuid"],
            "name": snapshot["gpu_name"],
            "driver": snapshot["gpu_driver"],
            "total_mib": device_total,
            "used_mib": float(snapshot["used_mb"]),
            "free_mib": free_mib,
            "compute_app_records": records,
            "observer": snapshot["observer"],
        },
        "allowed_processes": [],
        "service_marker": None,
        "service_reservation_mib": 0,
        "memory_reservation": {
            "job_cap_mib": JOB_CAP_MIB,
            "required_free_mib": REQUIRED_FREE_MIB,
            "device_total_mib": device_total,
            "within_device_total": True,
            "free_floor_met": True,
        },
        "unknown_compute_processes": [],
        "allowed_validation": allowed,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def capture_exclusive_decision(*, environment_manifest: str) -> dict[str, Any]:
    return _capture(environment_manifest=environment_manifest)


def write_exclusive_decision(path: str, *,
                             environment_manifest: str) -> dict[str, Any]:
    refuse_existing(path, label="Round 0016 exclusive GPU decision")
    value = _capture(environment_manifest=environment_manifest)
    atomic_write_new_json(path, value, immutable=True)
    return value


def validate_exclusive_decision(value: dict[str, Any], *,
                                environment_manifest: str,
                                require_current: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Round 0016 exclusive decision must be an object")
    identity = value.get("identity_sha256")
    body = {key: value[key] for key in value if key != "identity_sha256"}
    if identity != sha256_bytes(canonical_json(body)):
        raise ValueError("Round 0016 exclusive decision seal changed")
    expected_fields = {
        "schema", "captured_utc", "policy", "environment_manifest",
        "environment_identity_sha256", "gpu", "allowed_processes",
        "service_marker", "service_reservation_mib", "memory_reservation",
        "unknown_compute_processes", "allowed_validation", "identity_sha256",
    }
    gpu_fields = {
        "uuid", "name", "driver", "total_mib", "used_mib", "free_mib",
        "compute_app_records", "observer",
    }
    snapshot_fields = {
        "at", "gpu", "compute_apps", "compute_app_records", "compute_pids",
        "free_mb", "used_mb", "total_mb", "n_co_tenants", "observer",
        "gpu_uuid", "gpu_name", "gpu_driver",
    }
    memory_fields = {
        "job_cap_mib", "required_free_mib", "device_total_mib",
        "within_device_total", "free_floor_met",
    }
    environment = _environment(environment_manifest)
    gpu = value.get("gpu")
    memory = value.get("memory_reservation")
    allowed = value.get("allowed_validation")
    if (set(value) != expected_fields or
            value.get("schema") != "round0016-exclusive-gpu-decision-v1" or
            value.get("policy") != POLICY or
            value.get("allowed_processes") != [] or
            value.get("service_marker") is not None or
            value.get("service_reservation_mib") != 0 or
            value.get("unknown_compute_processes") != [] or
            value.get("environment_manifest") !=
            expected_input_signature(environment_manifest) or
            value.get("environment_identity_sha256") !=
            environment["identity_sha256"] or
            not isinstance(gpu, dict) or set(gpu) != gpu_fields or
            gpu.get("uuid") != environment["gpu_uuid"] or
            gpu.get("name") != environment["gpu_name"] or
            gpu.get("driver") != environment["gpu_driver"] or
            gpu.get("compute_app_records") != [] or
            not isinstance(gpu.get("observer"), dict) or
            set(gpu["observer"]) != {"canonical_path", "kind", "bytes", "sha256"} or
            expected_input_signature(gpu["observer"]["canonical_path"]) !=
            gpu["observer"] or
            not isinstance(memory, dict) or set(memory) != memory_fields or
            memory.get("job_cap_mib") != JOB_CAP_MIB or
            memory.get("required_free_mib") != REQUIRED_FREE_MIB or
            memory.get("within_device_total") is not True or
            memory.get("free_floor_met") is not True or
            not isinstance(allowed, dict) or
            set(allowed) != {"expected", "observed", "gpu_snapshot"} or
            allowed.get("expected") != [] or allowed.get("observed") != [] or
            not isinstance(allowed.get("gpu_snapshot"), dict) or
            set(allowed["gpu_snapshot"]) != snapshot_fields):
        raise ValueError("Round 0016 exclusive policy/snapshot is incomplete")
    snapshot = allowed["gpu_snapshot"]
    try:
        captured = datetime.datetime.fromisoformat(
            value["captured_utc"].replace("Z", "+00:00"))
        numeric_consistent = (
            int(gpu["total_mib"]) == int(snapshot["total_mb"]) ==
            int(memory["device_total_mib"]) and
            float(gpu["used_mib"]) == float(snapshot["used_mb"]) and
            float(gpu["free_mib"]) == float(snapshot["free_mb"]) and
            int(memory["job_cap_mib"]) <= int(memory["device_total_mib"]) and
            float(gpu["free_mib"]) >= int(memory["required_free_mib"]))
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError("Round 0016 exclusive snapshot values are malformed") from exc
    if (captured.tzinfo is None or not numeric_consistent or
            snapshot["gpu_uuid"] != gpu["uuid"] or
            snapshot["gpu_name"] != gpu["name"] or
            snapshot["gpu_driver"] != gpu["driver"] or
            snapshot["observer"] != gpu["observer"] or
            snapshot["compute_app_records"] != [] or
            snapshot["compute_pids"] != [] or
            snapshot["n_co_tenants"] != 0):
        raise ValueError("Round 0016 snapshot is not the empty sealed device")
    if require_current:
        now = datetime.datetime.now(datetime.timezone.utc)
        age = (now - captured).total_seconds()
        if age < 0 or age > MAX_SNAPSHOT_AGE_SECONDS:
            raise RuntimeError("Round 0016 exclusive snapshot is stale")
        current = _capture(environment_manifest=environment_manifest)
        if (current["allowed_processes"] != [] or
                current["gpu"]["compute_app_records"] != [] or
                current["gpu"]["uuid"] != gpu["uuid"] or
                current["gpu"]["name"] != gpu["name"] or
                current["gpu"]["driver"] != gpu["driver"] or
                current["gpu"]["observer"] != gpu["observer"]):
            raise RuntimeError("Round 0016 live exclusive GPU identity changed")
    return value


def load_exclusive_decision(path: str, *, environment_manifest: str,
                            require_current: bool) -> dict[str, Any]:
    canonical = os.path.realpath(path)
    if canonical != path or not os.path.isfile(path) or os.path.islink(path):
        raise ValueError("Round 0016 exclusive decision path is not canonical")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    return validate_exclusive_decision(
        value, environment_manifest=environment_manifest,
        require_current=require_current)
