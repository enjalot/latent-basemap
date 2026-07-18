"""CUDA-uninitialized validation for the exact Round 0017 queue manifest."""
from __future__ import annotations

import json
import os
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes)
from .release_preflight import validate_release_preflight_receipt
from .round0014_admission import (
    CACHE_KEYS, CHILD_KEYS, HASH64, FULL_SHA, _canonical, _deadline,
    _overlap, _static_environment, _validate_signature,
)
from .round0017_program import (
    ACCEPTED_CAPABILITY_SHA256, ACCEPTED_MANIFEST_FILE_SHA256,
    ACCEPTED_MANIFEST_RECEIPT_SHA256, GPU_HOURS_CAP, GPU_LEASE_PATH,
    JOB_FIELDS, PROGRAM, ROUND_ID, ROUND_SHA256, validate_exact_program,
)
from .round0016_service import (JOB_CAP_MIB, POLICY, REQUIRED_FREE_MIB)
from .roundwatch_gate import validate_roundwatch_binding


QUEUE_FIELDS = {
    "schema_version", "program", "round_id", "round_sha256", "release_sha",
    "execution_authority", "required_reviews", "environment_freeze_sha",
    "environment_identity_sha", "gpu_hours_cap", "queue_class",
    "training_performed", "deadline_utc", "environment_manifest",
    "cache_environment", "child_environment", "gate_receipts_dir",
    "controller_checkpoints_dir", "controller_terminal_summary", "repo_root",
    "lease_path", "lease_release_receipt", "allowed_processes", "jobs",
    "input_staging", "fixture_identity", "program_policy", "program_inputs",
    "global_input_registry", "source_closure", "roundwatch_binding",
    "release_preflight_identity", "gate_preparation_receipt",
}


def validate_round0017_queue_manifest(data: dict[str, Any],
                                      path: str) -> dict[str, Any]:
    if not isinstance(data, dict) or set(data) != QUEUE_FIELDS:
        raise ValueError("Round 0017 queue manifest fields changed")
    if (data["schema_version"] != 1 or data["program"] != PROGRAM or
            data["round_id"] != ROUND_ID or data["round_sha256"] != ROUND_SHA256):
        raise ValueError("queue does not identify issued basemap-100m Round 0017")
    if (data["execution_authority"] != "autonomous-gpu" or
            data["required_reviews"] != ["0013"] or
            float(data["gpu_hours_cap"]) != GPU_HOURS_CAP or
            data["queue_class"] != "research" or
            data["training_performed"] is not True):
        raise ValueError("Round 0017 authority/review/cap policy changed")
    if not FULL_SHA.fullmatch(str(data["release_sha"])) or any(
            not HASH64.fullmatch(str(data[key])) for key in
            ("environment_freeze_sha", "environment_identity_sha")):
        raise ValueError("Round 0017 release/environment identity is malformed")
    _deadline(data["deadline_utc"])
    manifest_path = _canonical(os.path.realpath(path), label="queue manifest")
    if os.path.realpath(path) != path:
        raise ValueError("Round 0017 queue manifest path is not canonical")
    repo_root = os.path.realpath(data["repo_root"])
    if repo_root != data["repo_root"] or not os.path.isdir(repo_root):
        raise ValueError("Round 0017 run checkout is not canonical")
    for name in ("gate_receipts_dir", "controller_checkpoints_dir"):
        current = _canonical(data[name], label=name)
        if not os.path.isdir(current) or os.path.islink(current) or os.listdir(current):
            raise ValueError(f"Round 0017 {name} must be a fresh empty directory")
    for name in ("controller_terminal_summary", "gate_preparation_receipt",
                 "lease_release_receipt", "environment_manifest"):
        _canonical(data[name], label=name)
    for name in ("controller_terminal_summary", "lease_release_receipt"):
        if os.path.lexists(data[name]):
            raise FileExistsError(f"Round 0017 {name} must be absent before execution")
    if data["lease_path"] != GPU_LEASE_PATH:
        raise ValueError("Round 0017 GPU lease path changed")
    if data["allowed_processes"] != []:
        raise ValueError("Round 0017 requires allowed_processes=[]")

    cache = data["cache_environment"]
    if (not isinstance(cache, dict) or
            set(cache) != {"PYTHONDONTWRITEBYTECODE", *CACHE_KEYS} or
            cache["PYTHONDONTWRITEBYTECODE"] != "1"):
        raise ValueError("Round 0017 cache environment fields changed")
    roots = []
    for key in CACHE_KEYS:
        root = _canonical(cache[key], label=f"cache {key}")
        if not os.path.isdir(root) or os.path.islink(root) or os.listdir(root):
            raise ValueError(f"Round 0017 cache {key} is not fresh and empty")
        roots.append(root)
    if any(_overlap(left, right) for position, left in enumerate(roots)
           for right in roots[position + 1:]):
        raise ValueError("Round 0017 cache roots overlap")
    child = data["child_environment"]
    if (not isinstance(child, dict) or set(child) != CHILD_KEYS or
            any(not isinstance(value, str) for value in child.values()) or
            any(child[key] != cache[key] for key in cache) or
            not child["CUDA_VISIBLE_DEVICES"] or
            "," in child["CUDA_VISIBLE_DEVICES"] or
            child["PYTHONDONTWRITEBYTECODE"] != "1" or
            child["PYTHONNOUSERSITE"] != "1" or
            child["PYTHONHASHSEED"] != "0" or
            child["TOKENIZERS_PARALLELISM"].lower() not in {"false", "0"} or
            child["LANG"] != "C.UTF-8" or child["LC_ALL"] != "C.UTF-8"):
        raise ValueError("Round 0017 child/cache environment changed")
    environment = _static_environment(data["environment_manifest"])
    if (environment["freeze_sha256"] != data["environment_freeze_sha"] or
            environment["identity_sha256"] != data["environment_identity_sha"] or
            environment["gpu_uuid"] != child["CUDA_VISIBLE_DEVICES"]):
        raise ValueError("Round 0017 environment differs from child binding")

    staging = data["input_staging"]
    fields = {
        "schema", "reference_manifest", "accepted_manifest_sha256",
        "manifest_receipt_sha256", "capability_sha256", "registered_file_count",
        "payloads_copied", "identity_sha256",
    }
    if not isinstance(staging, dict) or set(staging) != fields:
        raise ValueError("Round 0017 input-staging fields changed")
    body = {key: staging[key] for key in staging if key != "identity_sha256"}
    if (staging["schema"] != "round0017-input-staging-v1" or
            staging["identity_sha256"] != sha256_bytes(canonical_json(body)) or
            staging["accepted_manifest_sha256"] != ACCEPTED_MANIFEST_FILE_SHA256 or
            staging["manifest_receipt_sha256"] !=
            ACCEPTED_MANIFEST_RECEIPT_SHA256 or
            staging["capability_sha256"] != ACCEPTED_CAPABILITY_SHA256 or
            not isinstance(staging["registered_file_count"], int) or
            staging["registered_file_count"] <= 0 or
            staging["payloads_copied"] is not False):
        raise ValueError("Round 0017 reference-only staging policy changed")
    _validate_signature(staging["reference_manifest"], label="reference manifest")
    fixture = data["fixture_identity"]
    if (not isinstance(fixture, dict) or set(fixture) != {
            "schema", "canonical_path", "sha256", "identity_sha256"} or
            fixture["schema"] != "round0017-canary-derivation-v1" or
            any(not HASH64.fullmatch(str(fixture[key])) for key in
                ("sha256", "identity_sha256")) or
            expected_input_signature(fixture["canonical_path"])["sha256"] !=
            fixture["sha256"]):
        raise ValueError("Round 0017 canary identity changed")
    validate_roundwatch_binding(data["roundwatch_binding"])

    registry = data["global_input_registry"]
    if not isinstance(registry, list) or not registry:
        raise ValueError("Round 0017 global input registry is empty")
    registry_paths = []
    for position, signature in enumerate(registry):
        _validate_signature(signature, label=f"global input {position}")
        registry_paths.append(signature["canonical_path"])
    if registry_paths != sorted(registry_paths) or len(registry_paths) != len(
            set(registry_paths)):
        raise ValueError("Round 0017 global input registry is not sorted/unique")

    context = validate_exact_program(
        data, manifest_path=manifest_path, repo_root=repo_root)
    if staging["registered_file_count"] != context[
            "reference_manifest"]["reference_count"]:
        raise ValueError("Round 0017 staged reference count changed")
    exclusive = context["exclusive_gpu_decision"]
    memory = exclusive["memory_reservation"]
    if (exclusive["policy"] != POLICY or exclusive["allowed_processes"] != [] or
            exclusive["service_marker"] is not None or
            exclusive["service_reservation_mib"] != 0 or
            memory["job_cap_mib"] != JOB_CAP_MIB or
            memory["required_free_mib"] != REQUIRED_FREE_MIB or
            memory["within_device_total"] is not True or
            memory["free_floor_met"] is not True):
        raise ValueError("Round 0017 exclusive GPU capacity decision changed")
    release_roles = [item["signature"] for item in data["program_inputs"]
                     if item["role"] == "release_preflight_receipt"]
    if len(release_roles) != 1:
        raise ValueError("Round 0017 release-preflight role is not unique")
    release = validate_release_preflight_receipt(
        release_roles[0]["canonical_path"],
        expected_identity_sha256=data["release_preflight_identity"],
        expected_signature=release_roles[0])
    if release["release_sha"] != data["release_sha"]:
        raise ValueError("Round 0017 release preflight differs from queue")

    controls = [
        ("controller terminal", data["controller_terminal_summary"], "absent"),
        ("lease release", data["lease_release_receipt"], "absent"),
        ("gate preparation", data["gate_preparation_receipt"], "optional-file"),
        ("controller checkpoints", data["controller_checkpoints_dir"], "directory"),
        ("gate receipts", data["gate_receipts_dir"], "directory"),
    ]
    for job in data["jobs"]:
        if set(job) != JOB_FIELDS or job["node_policy"]["gpu_memory_cap_mb"] != \
                JOB_CAP_MIB:
            raise ValueError("Round 0017 job fields/memory cap changed")
        controls.extend((f"{job['id']} output", value, "absent")
                        for value in job["outputs"])
        controls.extend((f"{job['id']} {key}", job[key], "absent")
                        for key in ("done_marker", "log", "manifest"))
    normalized = []
    for label, value, state in controls:
        canonical = _canonical(value, label=label)
        if state == "absent" and os.path.lexists(canonical):
            raise FileExistsError(f"Round 0017 refuses existing {label}")
        if state == "directory" and (
                not os.path.isdir(canonical) or os.path.islink(canonical)):
            raise ValueError(f"Round 0017 {label} must remain a real directory")
        if state == "optional-file" and os.path.lexists(canonical) and (
                not os.path.isfile(canonical) or os.path.islink(canonical)):
            raise ValueError(f"Round 0017 {label} must be absent or a real file")
        normalized.append((label, canonical))
    for position, (left_label, left) in enumerate(normalized):
        for right_label, right in normalized[position + 1:]:
            if _overlap(left, right):
                raise ValueError(
                    f"Round 0017 control/output alias: {left_label}/{right_label}")
        if any(_overlap(left, item) for item in registry_paths):
            raise ValueError("Round 0017 output/control overlaps a signed input")
        if any(_overlap(left, item) for item in roots):
            raise ValueError("Round 0017 output/control overlaps a cache")
    return context
