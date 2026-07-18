"""Exact known-service decision for the one Round 0015 queue.

This is deliberately target-specific.  It binds the already observed
``ls-serve`` co-tenant to one construction snapshot and reopens that decision
immediately before the owner gate is sealed.  It is not a general process
policy or a service-discovery mechanism.
"""
from __future__ import annotations

import base64
import datetime
import json
import os
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes)
from .output_safety import atomic_write_new_json, refuse_existing
from .run_controller import (gpu_snapshot, process_identity,
                             validate_allowed_processes)


POLICY = "allow_exact_service"
MARKER = "ls-serve"
SERVICE_CAP_MIB = 1024
JOB_CAP_MIB = 31_488
COMBINED_CAP_MIB = 32_512
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 5090"


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="microseconds")


def _environment(path: str) -> dict[str, Any]:
    canonical = os.path.realpath(path)
    if canonical != path or not os.path.isfile(path) or os.path.islink(path):
        raise ValueError("Round 0015 environment manifest is not one canonical file")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    required = {
        "freeze_sha256", "identity_sha256", "gpu_uuid", "gpu_name",
        "gpu_driver", "venv_path", "freeze_file",
    }
    if (not isinstance(value, dict) or not required.issubset(value) or
            any(not isinstance(value[key], str) or not value[key]
                for key in required)):
        raise ValueError("Round 0015 environment manifest is incomplete")
    if value["gpu_name"] != EXPECTED_GPU_NAME:
        raise ValueError("Round 0015 environment is not the one sealed RTX 5090")
    return value


def _capture(*, pid: int, environment_manifest: str) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Round 0015 service capture requires CUDA_VISIBLE_DEVICES='' ")
    environment = _environment(environment_manifest)
    service = process_identity(
        int(pid), marker=MARKER, gpu_memory_budget_mb=SERVICE_CAP_MIB,
        service_identity=MARKER)
    raw_cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
    # CUDA stays hidden; the target-specific field checks below turn this
    # nvidia-smi observation into the exact one-device construction snapshot.
    snapshot = gpu_snapshot()
    if (snapshot.get("gpu_uuid") != environment["gpu_uuid"] or
            snapshot.get("gpu_name") != environment["gpu_name"] or
            snapshot.get("gpu_driver") != environment["gpu_driver"] or
            not isinstance(snapshot.get("total_mb"), (int, float)) or
            snapshot["total_mb"] <= 0):
        raise RuntimeError("Round 0015 live GPU identity/total differs from the seal")
    allowed = validate_allowed_processes([service], snapshot=snapshot)
    records = snapshot.get("compute_app_records")
    if (not isinstance(records, list) or len(records) != 1 or
            records[0].get("pid") != int(pid) or
            records[0].get("gpu_uuid") != environment["gpu_uuid"]):
        raise RuntimeError("Round 0015 requires exactly the one declared ls-serve process")
    observed_vram = float(records[0]["used_memory_mb"])
    if observed_vram > SERVICE_CAP_MIB:
        raise RuntimeError("Round 0015 ls-serve is above its 1,024 MiB ceiling")
    device_total = int(snapshot["total_mb"])
    if COMBINED_CAP_MIB > device_total:
        raise RuntimeError("Round 0015 service plus job reservation exceeds device total")
    body = {
        "schema": "round0015-service-decision-v1",
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
            "free_mib": float(snapshot["free_mb"]),
            "compute_app_records": records,
            "observer": snapshot["observer"],
        },
        "declared_services": [{
            **service,
            "raw_cmdline_base64": base64.b64encode(raw_cmdline).decode("ascii"),
            "observed_vram_mib": observed_vram,
            "gpu_uuid": snapshot["gpu_uuid"],
        }],
        "allowed_processes": [service],
        "memory_reservation": {
            "service_cap_mib": SERVICE_CAP_MIB,
            "job_cap_mib": JOB_CAP_MIB,
            "combined_cap_mib": COMBINED_CAP_MIB,
            "device_total_mib": device_total,
            "within_device_total": True,
        },
        "unknown_compute_processes": [],
        "allowed_validation": allowed,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def capture_service_decision(*, pid: int, environment_manifest: str) -> dict[str, Any]:
    """Capture one live exact-service decision without writing anything."""
    return _capture(pid=pid, environment_manifest=environment_manifest)


def write_service_decision(path: str, *, pid: int,
                           environment_manifest: str) -> dict[str, Any]:
    refuse_existing(path, label="Round 0015 service decision")
    value = _capture(pid=pid, environment_manifest=environment_manifest)
    atomic_write_new_json(path, value, immutable=True)
    return value


def validate_service_decision(value: dict[str, Any], *,
                              environment_manifest: str,
                              require_current: bool) -> dict[str, Any]:
    """Reject every policy/list/snapshot mismatch authorized by Round 0015."""
    if not isinstance(value, dict):
        raise ValueError("Round 0015 service decision must be an object")
    identity = value.get("identity_sha256")
    body = {key: value[key] for key in value if key != "identity_sha256"}
    if identity != sha256_bytes(canonical_json(body)):
        raise ValueError("Round 0015 service decision seal changed")
    services = value.get("declared_services")
    allowed = value.get("allowed_processes")
    memory = value.get("memory_reservation")
    gpu = value.get("gpu")
    allowed_validation = value.get("allowed_validation")
    environment = _environment(environment_manifest)
    expected_fields = {
        "schema", "captured_utc", "policy", "environment_manifest",
        "environment_identity_sha256", "gpu", "declared_services",
        "allowed_processes", "memory_reservation",
        "unknown_compute_processes", "allowed_validation", "identity_sha256",
    }
    service_fields = {
        "pid", "proc_starttime_ticks", "cmdline_sha256", "service_identity",
        "marker", "gpu_memory_budget_mb", "raw_cmdline_base64",
        "observed_vram_mib", "gpu_uuid",
    }
    allowed_fields = {
        "pid", "proc_starttime_ticks", "cmdline_sha256", "service_identity",
        "marker", "gpu_memory_budget_mb",
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
        "service_cap_mib", "job_cap_mib", "combined_cap_mib",
        "device_total_mib", "within_device_total",
    }
    if (set(value) != expected_fields or
            value.get("schema") != "round0015-service-decision-v1" or
            value.get("policy") != POLICY or
            not isinstance(services, list) or len(services) != 1 or
            not isinstance(allowed, list) or len(allowed) != 1 or
            set(services[0]) != service_fields or
            set(allowed[0]) != allowed_fields or
            services[0].get("marker") != MARKER or
            services[0].get("service_identity") != MARKER or
            services[0].get("gpu_memory_budget_mb") != SERVICE_CAP_MIB or
            {key: services[0][key] for key in (
                "pid", "proc_starttime_ticks", "cmdline_sha256",
                "service_identity", "marker", "gpu_memory_budget_mb")}
            != allowed[0] or
            not isinstance(memory, dict) or set(memory) != memory_fields or
            memory.get("service_cap_mib") != SERVICE_CAP_MIB or
            memory.get("job_cap_mib") != JOB_CAP_MIB or
            memory.get("combined_cap_mib") != COMBINED_CAP_MIB or
            memory.get("within_device_total") is not True or
            COMBINED_CAP_MIB > int(memory.get("device_total_mib", 0)) or
            value.get("unknown_compute_processes") != [] or
            value.get("environment_manifest") !=
            expected_input_signature(environment_manifest) or
            value.get("environment_identity_sha256") !=
            environment["identity_sha256"] or
            not isinstance(gpu, dict) or set(gpu) != gpu_fields or
            gpu.get("uuid") != environment["gpu_uuid"] or
            gpu.get("name") != environment["gpu_name"] or
            gpu.get("driver") != environment["gpu_driver"] or
            not isinstance(gpu.get("observer"), dict) or
            set(gpu["observer"]) != {"canonical_path", "kind", "bytes", "sha256"} or
            not isinstance(allowed_validation, dict) or
            set(allowed_validation) != {"expected", "observed", "gpu_snapshot"} or
            allowed_validation.get("expected") != allowed or
            allowed_validation.get("observed") != allowed or
            not isinstance(allowed_validation.get("gpu_snapshot"), dict) or
            set(allowed_validation["gpu_snapshot"]) != snapshot_fields):
        raise ValueError("Round 0015 exact-service policy/snapshot is incomplete or mismatched")
    snapshot = allowed_validation["gpu_snapshot"]
    records = snapshot["compute_app_records"]
    try:
        raw_cmdline = base64.b64decode(
            services[0]["raw_cmdline_base64"], validate=True)
        captured_at = datetime.datetime.fromisoformat(
            value["captured_utc"].replace("Z", "+00:00"))
        numeric_consistent = (
            int(gpu["total_mib"]) == int(snapshot["total_mb"]) ==
            int(memory["device_total_mib"]) and
            float(gpu["used_mib"]) == float(snapshot["used_mb"]) and
            float(gpu["free_mib"]) == float(snapshot["free_mb"]) and
            int(memory["combined_cap_mib"]) ==
            int(memory["service_cap_mib"]) + int(memory["job_cap_mib"]))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("Round 0015 service snapshot values are malformed") from exc
    if (captured_at.tzinfo is None or
            base64.b64encode(raw_cmdline).decode("ascii") !=
            services[0]["raw_cmdline_base64"] or
            sha256_bytes(raw_cmdline) != allowed[0]["cmdline_sha256"] or
            snapshot["gpu_uuid"] != gpu["uuid"] or
            snapshot["gpu_name"] != gpu["name"] or
            snapshot["gpu_driver"] != gpu["driver"] or
            snapshot["observer"] != gpu["observer"] or
            expected_input_signature(gpu["observer"]["canonical_path"]) !=
            gpu["observer"] or
            snapshot["n_co_tenants"] != 1 or
            snapshot["compute_pids"] != [allowed[0]["pid"]] or
            not isinstance(records, list) or len(records) != 1 or
            set(records[0]) != {"gpu_uuid", "pid", "used_memory_mb"} or
            records != gpu["compute_app_records"] or
            records[0].get("pid") != allowed[0]["pid"] or
            records[0].get("gpu_uuid") != gpu["uuid"] or
            float(records[0].get("used_memory_mb", SERVICE_CAP_MIB + 1)) !=
            float(services[0]["observed_vram_mib"]) or
            services[0]["gpu_uuid"] != gpu["uuid"] or
            not numeric_consistent or
            float(services[0].get("observed_vram_mib", SERVICE_CAP_MIB + 1)) >
            SERVICE_CAP_MIB):
        raise ValueError("Round 0015 construction snapshot is over service budget")
    if require_current:
        current = _capture(
            pid=int(allowed[0]["pid"]), environment_manifest=environment_manifest)
        if current["allowed_processes"] != allowed:
            raise RuntimeError("Round 0015 service identity is stale or changed")
        return value
    return value


def load_service_decision(path: str, *, environment_manifest: str,
                          require_current: bool) -> dict[str, Any]:
    canonical = os.path.realpath(path)
    if canonical != path or not os.path.isfile(path) or os.path.islink(path):
        raise ValueError("Round 0015 service decision path is not canonical")
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    return validate_service_decision(
        value, environment_manifest=environment_manifest,
        require_current=require_current)
