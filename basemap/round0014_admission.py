"""CUDA-uninitialized validation for the exact Round 0014 queue manifest."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes)
from .release_preflight import validate_release_preflight_receipt
from .round0014_program import (
    ACCEPTED_CAPABILITY_SHA256, ACCEPTED_MANIFEST_FILE_SHA256,
    ACCEPTED_MANIFEST_RECEIPT_SHA256, GPU_HOURS_CAP, GPU_LEASE_PATH, JOB_FIELDS,
    PROGRAM, ROUND_ID, ROUND_SHA256, SEQUENCED_REVIEW_SHA256,
    validate_exact_program,
)
from .roundwatch_gate import validate_roundwatch_binding


HASH64 = re.compile(r"[0-9a-f]{64}")
FULL_SHA = re.compile(r"[0-9a-f]{40}")
QUEUE_FIELDS = {
    "schema_version", "program", "round_id", "round_sha256", "release_sha",
    "execution_authority", "required_reviews", "environment_freeze_sha",
    "environment_identity_sha", "gpu_hours_cap", "queue_class",
    "training_performed", "deadline_utc", "environment_manifest",
    "cache_environment", "child_environment", "gate_receipts_dir",
    "controller_checkpoints_dir", "controller_terminal_summary", "repo_root",
    "lease_path", "allowed_processes", "jobs", "input_staging",
    "fixture_identity", "program_policy", "program_inputs",
    "global_input_registry", "source_closure", "roundwatch_binding",
    "release_preflight_identity", "gate_preparation_receipt",
}
CACHE_KEYS = ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR",
              "PYTHONPYCACHEPREFIX", "NUMBA_CACHE_DIR", "MPLCONFIGDIR")
CHILD_KEYS = {
    "CUDA_VISIBLE_DEVICES", "PATH", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
    "PYTHONHASHSEED", "TOKENIZERS_PARALLELISM", "LANG", "LC_ALL", *CACHE_KEYS,
}


def _contained(path: str) -> bool:
    try:
        return os.path.commonpath(["/data", os.path.abspath(path)]) == "/data"
    except (TypeError, ValueError):
        return False


def _canonical(path: str, *, label: str) -> str:
    if not isinstance(path, str) or not os.path.isabs(path) or not _contained(path):
        raise ValueError(f"Round 0014 {label} must be an absolute /data path")
    if os.path.realpath(path) != path:
        raise ValueError(f"Round 0014 {label} traverses a symlink")
    current = "/data"
    for part in os.path.relpath(path, "/data").split(os.sep):
        current = os.path.join(current, part)
        if not os.path.lexists(current):
            break
        if os.path.islink(current):
            raise ValueError(f"Round 0014 {label} has symlinked ancestor {current}")
    return path


def _overlap(left: str, right: str) -> bool:
    try:
        return os.path.commonpath([left, right]) in {left, right}
    except ValueError:
        return False


def _deadline(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Round 0014 deadline must be RFC3339 text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Round 0014 deadline must carry a UTC offset")
    return parsed.astimezone(timezone.utc)


def _static_environment(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        environment = json.load(handle)
    fields = (
        "freeze_sha256", "python", "torch", "torch_cuda", "gpu_driver",
        "gpu_name", "gpu_uuid",
    )
    location_fields = ("freeze_file", "venv_path")
    if not isinstance(environment, dict) or any(
            not isinstance(environment.get(key), str) or not environment[key]
            for key in (*fields, *location_fields)):
        raise ValueError("Round 0014 sealed environment identity is incomplete")
    body = {key: environment[key] for key in fields}
    if (environment.get("identity_sha256") != sha256_bytes(canonical_json(body)) or
            environment.get("gpu_name") != "NVIDIA GeForce RTX 5090" or
            not re.fullmatch(r"GPU-[A-Za-z0-9][A-Za-z0-9-]{3,127}",
                             environment.get("gpu_uuid", ""))):
        raise ValueError("Round 0014 sealed environment identity is invalid")
    freeze = environment.get("freeze_file")
    if not isinstance(freeze, str) or os.path.realpath(freeze) != freeze:
        raise ValueError("Round 0014 freeze path is not canonical")
    with open(freeze, encoding="utf-8") as handle:
        lines = sorted(line.strip() for line in handle if line.strip())
    observed_freeze = sha256_bytes(
        "".join(line + "\n" for line in lines).encode("utf-8"))
    if observed_freeze != environment["freeze_sha256"]:
        raise ValueError("Round 0014 package freeze changed")
    python = os.path.realpath(os.path.join(environment["venv_path"], "bin", "python"))
    if not os.path.isfile(python):
        raise ValueError("Round 0014 sealed interpreter is missing")
    return {**environment, "resolved_python": python}


def _validate_signature(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or "canonical_path" not in value:
        raise ValueError(f"Round 0014 {label} lacks a content signature")
    if expected_input_signature(value["canonical_path"]) != value:
        raise ValueError(f"Round 0014 {label} bytes changed")
    return value


def validate_round0014_queue_manifest(data: dict[str, Any], path: str) -> dict[str, Any]:
    """Reopen the complete CPU-visible queue state without observing a GPU."""
    if not isinstance(data, dict) or set(data) != QUEUE_FIELDS:
        raise ValueError("Round 0014 queue manifest fields changed")
    if (data["schema_version"] != 1 or data["program"] != PROGRAM or
            data["round_id"] != ROUND_ID or data["round_sha256"] != ROUND_SHA256):
        raise ValueError("queue does not identify the issued basemap-100m Round 0014")
    if (data["execution_authority"] != "owner-gpu" or
            data["required_reviews"] != ["0013"] or
            float(data["gpu_hours_cap"]) != GPU_HOURS_CAP or
            data["queue_class"] != "research" or
            data["training_performed"] is not True):
        raise ValueError("Round 0014 authority/review/cap/training policy changed")
    if not FULL_SHA.fullmatch(str(data["release_sha"])):
        raise ValueError("Round 0014 release SHA is malformed")
    if any(not HASH64.fullmatch(str(data[key])) for key in
           ("environment_freeze_sha", "environment_identity_sha")):
        raise ValueError("Round 0014 environment identities are malformed")
    _deadline(data["deadline_utc"])
    manifest_path = _canonical(os.path.realpath(path), label="queue manifest")
    if os.path.realpath(path) != path:
        raise ValueError("Round 0014 queue manifest path is not canonical")
    repo_root = os.path.realpath(data["repo_root"])
    if repo_root != data["repo_root"] or not os.path.isdir(repo_root):
        raise ValueError("Round 0014 run checkout is not canonical")
    for name in ("gate_receipts_dir", "controller_checkpoints_dir"):
        current = _canonical(data[name], label=name)
        if not os.path.isdir(current) or os.path.islink(current) or os.listdir(current):
            raise ValueError(f"Round 0014 {name} must be a fresh empty directory")
    for name in ("controller_terminal_summary", "gate_preparation_receipt",
                 "environment_manifest"):
        _canonical(data[name], label=name)
    if os.path.lexists(data["controller_terminal_summary"]):
        raise FileExistsError("Round 0014 terminal summary must be absent before execution")
    if data["lease_path"] != GPU_LEASE_PATH:
        raise ValueError("Round 0014 GPU lease path changed")
    if not isinstance(data["allowed_processes"], list):
        raise ValueError("Round 0014 allowed-process records must be a list")
    from .queue_admission import _validate_allowed_process
    for position, process in enumerate(data["allowed_processes"]):
        _validate_allowed_process(process, position=position)
    if len({item["pid"] for item in data["allowed_processes"]}) != len(
            data["allowed_processes"]):
        raise ValueError("Round 0014 allowed-process PID identity is duplicated")

    cache = data["cache_environment"]
    if not isinstance(cache, dict) or set(cache) != {"PYTHONDONTWRITEBYTECODE", *CACHE_KEYS} \
            or cache["PYTHONDONTWRITEBYTECODE"] != "1":
        raise ValueError("Round 0014 cache environment fields changed")
    roots = []
    for key in CACHE_KEYS:
        root = _canonical(cache[key], label=f"cache {key}")
        if not os.path.isdir(root) or os.path.islink(root) or os.listdir(root):
            raise ValueError(f"Round 0014 cache {key} is not fresh and empty")
        roots.append(root)
    if any(_overlap(left, right) for position, left in enumerate(roots)
           for right in roots[position + 1:]):
        raise ValueError("Round 0014 cache roots overlap")
    child = data["child_environment"]
    if not isinstance(child, dict) or set(child) != CHILD_KEYS or \
            any(not isinstance(value, str) for value in child.values()):
        raise ValueError("Round 0014 child environment fields changed")
    if any(child[key] != cache[key] for key in cache):
        raise ValueError("Round 0014 child/cache environment differs")
    if (not child["CUDA_VISIBLE_DEVICES"] or "," in child["CUDA_VISIBLE_DEVICES"] or
            child["PYTHONDONTWRITEBYTECODE"] != "1" or
            child["PYTHONNOUSERSITE"] != "1" or child["PYTHONHASHSEED"] != "0" or
            child["TOKENIZERS_PARALLELISM"].lower() not in {"false", "0"} or
            child["LANG"] != "C.UTF-8" or child["LC_ALL"] != "C.UTF-8"):
        raise ValueError("Round 0014 child environment does not expose one sealed GPU")
    environment = _static_environment(data["environment_manifest"])
    if (environment["freeze_sha256"] != data["environment_freeze_sha"] or
            environment["identity_sha256"] != data["environment_identity_sha"] or
            environment["gpu_uuid"] != child["CUDA_VISIBLE_DEVICES"]):
        raise ValueError("Round 0014 environment manifest differs from child binding")

    staging = data["input_staging"]
    fields = {
        "schema", "reference_manifest", "accepted_manifest_sha256",
        "manifest_receipt_sha256", "capability_sha256", "registered_file_count",
        "payloads_copied", "identity_sha256",
    }
    if not isinstance(staging, dict) or set(staging) != fields:
        raise ValueError("Round 0014 input-staging fields changed")
    staging_body = {key: staging[key] for key in staging if key != "identity_sha256"}
    if (staging["schema"] != "round0014-input-staging-v1" or
            staging["identity_sha256"] != sha256_bytes(canonical_json(staging_body)) or
            staging["accepted_manifest_sha256"] != ACCEPTED_MANIFEST_FILE_SHA256 or
            staging["manifest_receipt_sha256"] != ACCEPTED_MANIFEST_RECEIPT_SHA256 or
            staging["capability_sha256"] != ACCEPTED_CAPABILITY_SHA256 or
            staging["registered_file_count"] != 77 or
            staging["payloads_copied"] is not False):
        raise ValueError("Round 0014 reference-only input-staging policy changed")
    _validate_signature(staging["reference_manifest"], label="reference manifest")
    fixture = data["fixture_identity"]
    if not isinstance(fixture, dict) or set(fixture) != {
            "schema", "canonical_path", "sha256", "identity_sha256"} or \
            fixture["schema"] != "round0014-canary-derivation-v1" or \
            any(not HASH64.fullmatch(str(fixture[key])) for key in
                ("sha256", "identity_sha256")):
        raise ValueError("Round 0014 deterministic canary identity changed")
    if expected_input_signature(fixture["canonical_path"])["sha256"] != fixture["sha256"]:
        raise ValueError("Round 0014 deterministic canary record changed")
    validate_roundwatch_binding(data["roundwatch_binding"])

    registry = data["global_input_registry"]
    if not isinstance(registry, list) or not registry:
        raise ValueError("Round 0014 global input registry is empty")
    registry_paths = []
    for position, signature in enumerate(registry):
        _validate_signature(signature, label=f"global input {position}")
        registry_paths.append(signature["canonical_path"])
    if registry_paths != sorted(registry_paths) or len(registry_paths) != len(
            set(registry_paths)):
        raise ValueError("Round 0014 global input registry is not sorted/unique")

    context = validate_exact_program(
        data, manifest_path=manifest_path, repo_root=repo_root)
    release_role = [item["signature"] for item in data["program_inputs"]
                    if item["role"] == "release_preflight_receipt"]
    if len(release_role) != 1:
        raise ValueError("Round 0014 release-preflight role is not unique")
    release = validate_release_preflight_receipt(
        release_role[0]["canonical_path"],
        expected_identity_sha256=data["release_preflight_identity"],
        expected_signature=release_role[0])
    if release["release_sha"] != data["release_sha"]:
        raise ValueError("Round 0014 release preflight differs from queue")

    controls = [
        ("controller terminal", data["controller_terminal_summary"], "absent"),
        ("gate preparation", data["gate_preparation_receipt"], "optional-file"),
        ("controller checkpoints", data["controller_checkpoints_dir"], "directory"),
        ("gate receipts", data["gate_receipts_dir"], "directory"),
    ]
    for job in data["jobs"]:
        if set(job) != JOB_FIELDS:
            raise ValueError("Round 0014 job fields changed")
        controls.extend((f"{job['id']} output", value, "absent")
                        for value in job["outputs"])
        controls.extend((f"{job['id']} {key}", job[key], "absent")
                        for key in ("done_marker", "log", "manifest"))
    normalized = []
    for label, value, state in controls:
        canonical = _canonical(value, label=label)
        if state == "absent" and os.path.lexists(canonical):
            raise FileExistsError(f"Round 0014 refuses existing {label}: {canonical}")
        if state == "directory" and (
                not os.path.isdir(canonical) or os.path.islink(canonical)):
            raise ValueError(f"Round 0014 {label} must remain a real directory")
        if state == "optional-file" and os.path.lexists(canonical) and (
                not os.path.isfile(canonical) or os.path.islink(canonical)):
            raise ValueError(f"Round 0014 {label} must be absent or a real file")
        normalized.append((label, canonical))
    for position, (left_label, left) in enumerate(normalized):
        for right_label, right in normalized[position + 1:]:
            if _overlap(left, right):
                raise ValueError(f"Round 0014 control/output alias: {left_label}/{right_label}")
        if any(_overlap(left, item) for item in registry_paths):
            raise ValueError(f"Round 0014 output/control overlaps a signed input: {left}")
        if any(_overlap(left, item) for item in roots):
            raise ValueError(f"Round 0014 output/control overlaps a cache: {left}")
    return context
