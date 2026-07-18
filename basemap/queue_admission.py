"""Fail-closed admission for the one issued Round 0005 production program."""
from __future__ import annotations

import csv
import json
import os
import re
import secrets
import subprocess
import threading
import uuid
import weakref
from datetime import datetime, timezone
from typing import Any, Callable

from .artifact_identity import (canonical_json, expected_input_signature,
                                git_checkout_state, path_signature, sha256_bytes,
                                sha256_file)
from .output_safety import atomic_write_new_json, canonical_data_path
from .release_preflight import validate_release_preflight_receipt
from .round0005_program import (ROUND0005_JOB_FIELDS, ROUND0005_NODES,
                               validate_exact_program)
from .roundwatch_gate import (RoundwatchGateAuthority,
                              validate_roundwatch_binding)
from .source_closure import validate_source_closure_receipt

FULL_SHA = re.compile(r"[0-9a-f]{40}")
HASH64 = re.compile(r"[0-9a-f]{64}")
CACHE_KEYS = ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR",
              "PYTHONPYCACHEPREFIX", "NUMBA_CACHE_DIR", "MPLCONFIGDIR")
ROUND0005_LEASE_PATH = "/data/latent-basemap/.gpu_lease"
NVIDIA_SMI_EXECUTABLE = "/usr/bin/nvidia-smi"
ROUND0005_GSV_GPU_NAME = "NVIDIA GeForce RTX 5090"
NVIDIA_SMI_ENVIRONMENT = {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
ROUND0005_CHILD_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES", "PATH", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE",
    "PYTHONHASHSEED", "TOKENIZERS_PARALLELISM", "LANG", "LC_ALL", *CACHE_KEYS,
}
ROUND0005_QUEUE_FIELDS = {
    "schema_version", "program", "round_id", "round_sha256", "release_sha",
    "execution_authority", "required_reviews",
    "environment_freeze_sha", "environment_identity_sha", "gpu_hours_cap",
    "queue_class", "training_performed", "deadline_utc", "environment_manifest",
    "cache_environment", "child_environment", "gate_receipts_dir",
    "controller_checkpoints_dir", "controller_terminal_summary", "repo_root",
    "lease_path", "allowed_processes", "jobs", "input_staging", "fixture_identity",
    "program_policy", "program_inputs", "global_input_registry", "source_closure",
    "roundwatch_binding", "release_preflight_identity", "gate_preparation_receipt",
}
ROUND0005_INPUT_STAGING_FIELDS = {
    "maps_seal", "model_seal", "testbed_seal", "data_closure_identity_sha256",
    "model_revision",
}
ALLOWED_SERVICE_MARKERS = {"ls-serve", "moonshine-web"}
MUTATION_WINDOWS = (
    "capture-to-manifest-publication",
    "manifest-publication-to-gate-preparation",
    "gate-preparation-to-admission",
    "gate-response-to-Popen",
)

_ADMISSION_REGISTRY_LOCK = threading.Lock()
_ADMISSION_REGISTRY: dict[int, tuple[weakref.ReferenceType, str]] = {}


def _controller_process_record(pid: int) -> dict[str, Any]:
    """Capture the exact live process that consumes the one-shot admission."""
    raw_stat = open(f"/proc/{pid}/stat", encoding="utf-8").read()
    fields = raw_stat[raw_stat.rfind(")") + 2:].split()
    cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
    argv = [part.decode(errors="surrogateescape")
            for part in cmdline.rstrip(b"\0").split(b"\0")]
    executable = os.path.realpath(f"/proc/{pid}/exe")
    return {
        "pid": int(pid),
        "proc_starttime_ticks": int(fields[19]),
        "argv": argv,
        "cmdline_sha256": sha256_bytes(cmdline),
        "executable": expected_input_signature(executable),
    }


def observe_round0005_cache_policy(manifest: dict[str, Any]) -> dict[str, Any]:
    """Reopen every cache root and bind the bytecode directory's history.

    Python bytecode is disabled for the admitted program, so its dedicated
    no-follow directory must remain both empty and untouched.  Binding mtime and
    ctime catches a create-then-unlink injection that an emptiness-only check
    would miss; ctime cannot be restored with ``utime``.
    """
    observed: dict[str, Any] = {}
    for key in CACHE_KEYS:
        path = manifest["cache_environment"][key]
        flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                 getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
        fd = os.open(path, flags)
        try:
            status = os.fstat(fd)
            entries = sorted(os.listdir(fd))
        finally:
            os.close(fd)
        if os.path.realpath(path) != path or os.path.islink(path):
            raise RuntimeError(f"cache {key} is not a canonical no-follow directory")
        if key == "PYTHONPYCACHEPREFIX" and entries:
            raise RuntimeError(
                f"PYTHONPYCACHEPREFIX must remain empty; observed {entries!r}")
        observed[key] = {
            "path": path, "device": int(status.st_dev), "inode": int(status.st_ino),
            "mode": int(status.st_mode), "links": int(status.st_nlink),
            "kind": "directory", "pycache_empty": (
                not entries if key == "PYTHONPYCACHEPREFIX" else None),
            "pycache_mtime_ns": (
                int(status.st_mtime_ns) if key == "PYTHONPYCACHEPREFIX" else None),
            "pycache_ctime_ns": (
                int(status.st_ctime_ns) if key == "PYTHONPYCACHEPREFIX" else None),
        }
    return observed


def _observe_canonical_gpu() -> dict[str, Any]:
    """Read immutable GPU identities without importing Torch or calling CUDA."""
    if (os.path.realpath(NVIDIA_SMI_EXECUTABLE) != NVIDIA_SMI_EXECUTABLE or
            not os.path.isfile(NVIDIA_SMI_EXECUTABLE) or
            os.path.islink(NVIDIA_SMI_EXECUTABLE)):
        raise RuntimeError("canonical NVIDIA observer is missing, non-regular, or symlinked")
    observer = expected_input_signature(NVIDIA_SMI_EXECUTABLE)
    proc = subprocess.run(
        [NVIDIA_SMI_EXECUTABLE,
         "--query-gpu=uuid,name,driver_version",
         "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=dict(NVIDIA_SMI_ENVIRONMENT),
        close_fds=True, timeout=20)
    if expected_input_signature(NVIDIA_SMI_EXECUTABLE) != observer:
        raise RuntimeError("canonical NVIDIA observer identity changed during observation")
    if proc.returncode:
        raise RuntimeError(f"canonical GPU identity query failed: {proc.stderr.strip()}")
    rows = []
    for raw in csv.reader(proc.stdout.splitlines(), skipinitialspace=True):
        values = [value.strip() for value in raw]
        if len(values) != 3:
            raise RuntimeError(f"canonical GPU identity query is malformed: {raw!r}")
        uuid_value, name, driver = values
        if (not re.fullmatch(r"GPU-[A-Za-z0-9][A-Za-z0-9-]{3,127}", uuid_value) or
                not name or not driver):
            raise RuntimeError(f"canonical GPU identity row is invalid: {raw!r}")
        rows.append({"gpu_uuid": uuid_value, "gpu_name": name,
                     "gpu_driver": driver})
    if len(rows) != 1:
        raise RuntimeError(
            "Round 0005 requires the one registered GSV GPU; "
            f"observed {len(rows)} inventory rows")
    return {
        "schema": "round0005_live_gpu_identity.v1",
        "observer": observer,
        "gpus": rows,
        "inventory_sha256": sha256_bytes(canonical_json(rows)),
    }


def validate_canonical_gpu_environment(environment: dict[str, Any]) -> dict[str, Any]:
    """Recompute the sealed environment identity and match one live GPU UUID."""
    identity_fields = (
        "freeze_sha256", "python", "torch", "torch_cuda", "gpu_driver",
        "gpu_name", "gpu_uuid",
    )
    if (not isinstance(environment, dict) or
            any(not isinstance(environment.get(key), str) or not environment[key]
                for key in identity_fields)):
        raise RuntimeError("sealed production environment identity fields are incomplete")
    identity_body = {key: environment[key] for key in identity_fields}
    if (not HASH64.fullmatch(str(environment.get("identity_sha256", ""))) or
            environment["identity_sha256"] !=
            sha256_bytes(canonical_json(identity_body))):
        raise RuntimeError("sealed production environment identity hash is invalid")
    if environment["gpu_name"] != ROUND0005_GSV_GPU_NAME:
        raise RuntimeError(
            "Round 0005 environment is not the registered GSV RTX-5090")
    observation = _observe_canonical_gpu()
    if (observation.get("observer") != expected_input_signature(
            NVIDIA_SMI_EXECUTABLE) or
            observation.get("gpus") is None or len(observation["gpus"]) != 1):
        raise RuntimeError("canonical single-device observation is stale or incomplete")
    matches = [value for value in observation["gpus"]
               if value["gpu_uuid"] == environment["gpu_uuid"]]
    if (len(matches) != 1 or matches[0]["gpu_name"] != environment["gpu_name"] or
            matches[0]["gpu_driver"] != environment["gpu_driver"]):
        raise RuntimeError("sealed GPU UUID/model/driver differs from live canonical identity")
    return {"environment_identity_sha256": environment["identity_sha256"],
            "sealed_gpu": matches[0], "observation": observation}


class IntegrityMismatch(RuntimeError):
    """Carry exact expected/observed state into an automatic rejection receipt."""

    def __init__(self, message: str, *, expected: Any, observed: Any, phase: str):
        super().__init__(message)
        self.expected = expected
        self.observed = observed
        self.phase = phase


class ControllerEntryBoundaryFailure(RuntimeError):
    """A genuine one-shot admission failed its live boundary before the lease."""

    def __init__(self, job_name: str, original: Exception):
        super().__init__(f"{type(original).__name__}: {original}")
        self.job_name = job_name
        self.original = original


# Compatibility name used by older callers; the semantics now cover every
# integrity identity rather than only current-job input bytes.
ExpectedInputMismatch = IntegrityMismatch


def _parse_deadline(value: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("queue deadline_utc must be a nonempty RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("queue deadline_utc must be a valid RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("queue deadline_utc must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _contained_by_data(path: str) -> bool:
    try:
        return os.path.commonpath(["/data", os.path.abspath(path)]) == "/data"
    except (TypeError, ValueError):
        return False


def _canonical_fresh_path(value: str, *, label: str) -> str:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError(f"{label} must be a nonempty absolute path")
    return canonical_data_path(value, label=label)


def _paths_overlap(left: str, right: str) -> bool:
    try:
        return os.path.commonpath([left, right]) in {left, right}
    except ValueError:
        return False


def _validate_expected_signature(signature: dict, *, label: str) -> None:
    if not isinstance(signature, dict):
        raise ValueError(f"{label} signature must be an object")
    required = {"canonical_path", "kind", "bytes", "sha256"}
    allowed = required | ({"members"} if signature.get("kind") == "directory" else set())
    if set(signature) != allowed:
        raise ValueError(
            f"{label} signature fields mismatch: expected={sorted(allowed)} "
            f"observed={sorted(signature)}")
    path = signature["canonical_path"]
    if (not isinstance(path, str) or not os.path.isabs(path) or
            os.path.realpath(path) != path):
        raise ValueError(f"{label} signature path is not absolute/canonical")
    if signature["kind"] not in {"file", "directory"}:
        raise ValueError(f"{label} signature kind is unsupported")
    if (not isinstance(signature["bytes"], int) or isinstance(signature["bytes"], bool) or
            signature["bytes"] < 0 or not HASH64.fullmatch(str(signature["sha256"]))):
        raise ValueError(f"{label} signature size/hash is invalid")
    if signature["kind"] == "directory":
        members = signature["members"]
        if not isinstance(members, list):
            raise ValueError(f"{label} directory signature members are missing")
        names = []
        total = 0
        directory_names = set()
        for member in members:
            if not isinstance(member, dict) or set(member) != {
                    "relative_path", "kind", "bytes", "sha256"}:
                raise ValueError(f"{label} directory member fields are invalid")
            name = member["relative_path"]
            if (not isinstance(name, str) or not name or os.path.isabs(name) or
                    os.path.normpath(name) != name or name.startswith("..") or
                    member["kind"] not in {"file", "directory"} or
                    not isinstance(member["bytes"], int) or isinstance(member["bytes"], bool) or
                    member["bytes"] < 0 or
                    not HASH64.fullmatch(str(member["sha256"]))):
                raise ValueError(f"{label} directory member is invalid")
            if member["kind"] == "directory":
                if member["bytes"] != 0:
                    raise ValueError(f"{label} directory member size must be zero")
                directory_names.add(name)
            names.append(name)
            if member["kind"] == "file":
                total += member["bytes"]
        if names != sorted(names) or len(names) != len(set(names)) or total != signature["bytes"]:
            raise ValueError(f"{label} directory members are not sorted/unique/size-bound")
        for member in members:
            parent = os.path.dirname(member["relative_path"])
            if parent and parent not in directory_names:
                raise ValueError(f"{label} directory member has an unbound parent")


def _validate_allowed_process(value: dict, *, position: int) -> None:
    fields = {"pid", "proc_starttime_ticks", "cmdline_sha256", "service_identity",
              "marker", "gpu_memory_budget_mb"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"allowed process {position} fields mismatch")
    for field in ("pid", "proc_starttime_ticks", "gpu_memory_budget_mb"):
        minimum = 0 if field == "gpu_memory_budget_mb" else 1
        if (not isinstance(value[field], int) or isinstance(value[field], bool) or
                value[field] < minimum):
            raise ValueError(f"allowed process {position} {field} is invalid")
    if (not HASH64.fullmatch(str(value["cmdline_sha256"])) or
            value["marker"] not in ALLOWED_SERVICE_MARKERS or
            value["service_identity"] != value["marker"]):
        raise ValueError(f"allowed process {position} identity is invalid")


def _validate_parent_chain(path: str, *, label: str) -> None:
    """Reject a symlink in any existing ancestor without creating anything."""
    absolute = os.path.abspath(path)
    current = "/"
    for piece in [part for part in absolute.split(os.sep) if part]:
        current = os.path.join(current, piece)
        if not os.path.lexists(current):
            break
        if os.path.islink(current):
            raise ValueError(f"{label} has a symlinked existing ancestor: {current}")
        if current != absolute and not os.path.isdir(current):
            raise ValueError(f"{label} has a nondirectory existing ancestor: {current}")


def validate_queue_manifest(data: dict, path: str) -> dict[str, Any]:
    """Validate the exact production program before gate, lease, or output work."""
    if isinstance(data, dict) and data.get("round_id") == "0015":
        from .round0015_admission import validate_round0015_queue_manifest

        return validate_round0015_queue_manifest(data, path)
    if isinstance(data, dict) and data.get("round_id") == "0014":
        # The Round-0005 seal remains byte-for-byte useful, but its scientific
        # program is not promoted.  Round 0014 has one target-specific static
        # validator that deliberately does not query or initialize a GPU.
        from .round0014_admission import validate_round0014_queue_manifest

        return validate_round0014_queue_manifest(data, path)
    if not isinstance(data, dict):
        raise ValueError("queue manifest must be a JSON object")
    missing = sorted(ROUND0005_QUEUE_FIELDS - set(data))
    extra = sorted(set(data) - ROUND0005_QUEUE_FIELDS)
    if missing or extra:
        raise ValueError(f"queue manifest fields mismatch: missing={missing} extra={extra}")
    if data["schema_version"] != 1:
        raise ValueError("queue schema_version must equal Roundwatch-compatible version 1")
    if data["program"] != "basemap-100m" or data["round_id"] != "0005":
        raise ValueError("queue must identify basemap-100m Round 0005 exactly")
    if (data["execution_authority"] != "planner-gpu" or
            data["required_reviews"] != []):
        raise ValueError("Round 0005 requires planner-gpu authority and no reviews")
    if not HASH64.fullmatch(str(data["round_sha256"])):
        raise ValueError("queue round_sha256 must be a full lowercase SHA-256")
    if not FULL_SHA.fullmatch(str(data["release_sha"])):
        raise ValueError("queue release_sha must be a full lowercase Git SHA")
    if any(not HASH64.fullmatch(str(data[key])) for key in
           ("environment_freeze_sha", "environment_identity_sha")):
        raise ValueError("queue environment identities must be full lowercase SHA-256 values")
    if float(data["gpu_hours_cap"]) != 0.75:
        raise ValueError("Round 0005 gpu_hours_cap must equal 0.75")
    if data["queue_class"] != "research":
        raise ValueError("Round 0005 is an exact research queue")
    # This field is recorded, never trusted: exact-program validation below
    # derives no-training from each canonical script and policy.
    if data["training_performed"] is not False:
        raise ValueError("the issued Round 0005 scripts perform no training")
    _parse_deadline(data["deadline_utc"])
    manifest_path = _canonical_fresh_path(path, label="queue manifest")
    _validate_parent_chain(manifest_path, label="queue manifest")
    repo_root = os.path.realpath(data["repo_root"])
    if repo_root != data["repo_root"] or not os.path.isdir(repo_root):
        raise ValueError("queue repo_root must be an existing canonical directory")
    for key in ("gate_receipts_dir", "controller_checkpoints_dir",
                "controller_terminal_summary", "gate_preparation_receipt",
                "environment_manifest"):
        if not isinstance(data[key], str) or not _contained_by_data(data[key]):
            raise ValueError(f"queue {key} must be contained by /data")
        _validate_parent_chain(data[key], label=f"queue {key}")
    for key in ("gate_receipts_dir", "controller_checkpoints_dir"):
        if (not os.path.isdir(data[key]) or os.path.islink(data[key]) or
                os.listdir(data[key])):
            raise ValueError(f"queue {key} must be a fresh regular directory")
    _canonical_fresh_path(data["controller_terminal_summary"],
                          label="controller terminal summary")
    if os.path.lexists(data["controller_terminal_summary"]):
        raise FileExistsError("controller terminal summary must be absent before gate")
    _canonical_fresh_path(data["gate_preparation_receipt"],
                          label="gate preparation receipt")
    if data["lease_path"] != ROUND0005_LEASE_PATH:
        raise ValueError(f"Round 0005 lease_path must equal {ROUND0005_LEASE_PATH}")
    _validate_parent_chain(data["lease_path"], label="GPU lease")
    if not isinstance(data["allowed_processes"], list):
        raise ValueError("queue allowed_processes must be a list")
    for position, value in enumerate(data["allowed_processes"]):
        _validate_allowed_process(value, position=position)
    if len({value["pid"] for value in data["allowed_processes"]}) != len(
            data["allowed_processes"]):
        raise ValueError("queue allowed process PID identities must be unique")

    cache = data["cache_environment"]
    if not isinstance(cache, dict) or set(cache) != {"PYTHONDONTWRITEBYTECODE", *CACHE_KEYS}:
        raise ValueError("queue cache_environment fields must be exact")
    if cache["PYTHONDONTWRITEBYTECODE"] != "1":
        raise ValueError("queue must disable Python bytecode")
    for key in CACHE_KEYS:
        if not isinstance(cache[key], str) or not _contained_by_data(cache[key]):
            raise ValueError(f"queue cache {key} must be contained by /data")
        _validate_parent_chain(cache[key], label=f"queue cache {key}")
        if (os.path.realpath(cache[key]) != cache[key] or os.path.islink(cache[key]) or
                not os.path.isdir(cache[key]) or os.listdir(cache[key])):
            raise ValueError(f"queue cache {key} must be a fresh no-follow directory")
    cache_roots = [cache[key] for key in CACHE_KEYS]
    for position, left in enumerate(cache_roots):
        for right in cache_roots[position + 1:]:
            if _paths_overlap(left, right):
                raise ValueError(f"queue cache roots alias or contain each other: {left} {right}")
    child = data["child_environment"]
    if not isinstance(child, dict) or set(child) != ROUND0005_CHILD_ENV_KEYS:
        raise ValueError("queue child environment fields must match the exact allow-list")
    if any(not isinstance(value, str) for value in child.values()):
        raise ValueError("queue child environment values must all be strings")
    if any(child[key] != cache[key] for key in cache):
        raise ValueError("queue child environment differs from cache environment")
    if (not child["CUDA_VISIBLE_DEVICES"] or "," in child["CUDA_VISIBLE_DEVICES"] or
            child["PYTHONNOUSERSITE"] != "1" or child["PYTHONHASHSEED"] != "0" or
            child["LANG"] != "C.UTF-8" or child["LC_ALL"] != "C.UTF-8" or
            child["TOKENIZERS_PARALLELISM"].lower() not in {"false", "0"}):
        raise ValueError("queue child environment does not expose exactly one sealed GPU")
    with open(data["environment_manifest"], encoding="utf-8") as handle:
        sealed_environment = json.load(handle)
    validate_canonical_gpu_environment(sealed_environment)
    sealed_gpu_uuid = sealed_environment.get("gpu_uuid") \
        if isinstance(sealed_environment, dict) else None
    if (not isinstance(sealed_gpu_uuid, str) or
            not re.fullmatch(r"GPU-[A-Za-z0-9][A-Za-z0-9-]{3,127}", sealed_gpu_uuid) or
            child["CUDA_VISIBLE_DEVICES"] != sealed_gpu_uuid):
        raise ValueError("queue CUDA visibility differs from the canonical sealed GPU UUID")

    staging = data["input_staging"]
    if not isinstance(staging, dict) or set(staging) != ROUND0005_INPUT_STAGING_FIELDS:
        raise ValueError("queue input_staging fields are incomplete or unknown")
    for name in ("maps_seal", "model_seal", "testbed_seal"):
        _validate_expected_signature(staging[name], label=f"input staging {name}")
        if expected_input_signature(staging[name]["canonical_path"]) != staging[name]:
            raise ValueError(f"queue staged {name} changed before gate")
    if (not HASH64.fullmatch(str(staging["data_closure_identity_sha256"])) or
            not FULL_SHA.fullmatch(str(staging["model_revision"]))):
        raise ValueError("queue staged data/model identities are malformed")
    fixture = data["fixture_identity"]
    if (not isinstance(fixture, dict) or set(fixture) != {
            "schema", "canonical_path", "sha256", "identity_sha256"} or
            fixture["schema"] != "round0005_all_node_fixture.v3" or
            not HASH64.fullmatch(str(fixture["sha256"])) or
            not HASH64.fullmatch(str(fixture["identity_sha256"]))):
        raise ValueError("queue fixture identity fields are invalid")
    if not isinstance(data["release_preflight_identity"], str) or not HASH64.fullmatch(
            data["release_preflight_identity"]):
        raise ValueError("queue release preflight identity is malformed")
    validate_roundwatch_binding(data["roundwatch_binding"])

    registry = data["global_input_registry"]
    if not isinstance(registry, list) or not registry:
        raise ValueError("queue global input registry must be nonempty")
    paths = []
    for position, signature in enumerate(registry):
        _validate_expected_signature(signature, label=f"global input {position}")
        paths.append(signature["canonical_path"])
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("queue global input registry must be sorted and canonical-unique")

    # The deep call reopens every seal and derives the exact command/resource
    # policy.  It is deliberately before all output alias checks and before any
    # caller can acquire the GPU lease.
    context = validate_exact_program(
        data, manifest_path=manifest_path, repo_root=repo_root)

    all_outputs: list[tuple[str, str]] = []
    for job in data["jobs"]:
        for output in job["outputs"]:
            canonical = _canonical_fresh_path(output, label=f"{job['id']} output")
            _validate_parent_chain(canonical, label=f"{job['id']} output")
            if os.path.lexists(canonical):
                raise FileExistsError(f"queue job {job['id']} refuses existing output: {canonical}")
            all_outputs.append((f"{job['id']}:output", canonical))
        for field in ("done_marker", "log", "manifest"):
            canonical = _canonical_fresh_path(job[field], label=f"{job['id']} {field}")
            _validate_parent_chain(canonical, label=f"{job['id']} {field}")
            if os.path.lexists(canonical):
                raise FileExistsError(f"queue job {job['id']} refuses existing {field}")
            all_outputs.append((f"{job['id']}:{field}", canonical))
    controller_paths = [
        ("controller:terminal", os.path.realpath(data["controller_terminal_summary"])),
        ("controller:gate-preparation",
         os.path.realpath(data["gate_preparation_receipt"])),
        ("controller:checkpoints", os.path.realpath(data["controller_checkpoints_dir"])),
        ("controller:gate-receipts", os.path.realpath(data["gate_receipts_dir"])),
    ]
    for position, (left_label, left) in enumerate([*all_outputs, *controller_paths]):
        for right_label, right in [*all_outputs, *controller_paths][position + 1:]:
            if _paths_overlap(left, right):
                raise ValueError(
                    f"queue output/control path alias: {left_label}={left} overlaps "
                    f"{right_label}={right}")
        for input_path in paths:
            if _paths_overlap(left, input_path):
                raise ValueError(
                    f"queue output/control path {left_label}={left} overlaps signed input "
                    f"{input_path}")
        for key in CACHE_KEYS:
            cache_path = cache[key]
            if _paths_overlap(left, cache_path):
                raise ValueError(
                    f"queue output/control path {left_label} aliases cache {key}")
    for key in CACHE_KEYS:
        for input_path in paths:
            if _paths_overlap(cache[key], input_path):
                raise ValueError(f"queue cache {key} aliases signed input {input_path}")
    return context


def canonical_manifest_job(job: dict) -> dict:
    return {field: job[field] for field in sorted(ROUND0005_JOB_FIELDS)}


def _emit_component_rejection_receipt(
        *, receipts_dir: str, phase: str, manifest_path: str,
        original_manifest_sha256: str, job: dict, expected: Any, observed: Any,
        error: str) -> str:
    """Component-only helper for the three pre-controller mutation windows."""
    if phase not in MUTATION_WINDOWS or phase == "gate-response-to-Popen":
        raise ValueError(f"component rejection phase is invalid: {phase}")
    if expected == observed:
        raise ValueError("automatic rejection receipt requires actual identity drift")
    if (not os.path.isdir(receipts_dir) or os.path.islink(receipts_dir) or
            not _contained_by_data(receipts_dir)):
        raise RuntimeError("automatic rejection receipt root is not a regular /data directory")
    sentinel_output = job["outputs"][0]
    if os.path.lexists(sentinel_output):
        raise RuntimeError("would-be child output exists; cannot prove pre-Popen rejection")
    body = {
        "schema": "round0005_integrity_receipt.v3",
        "phase": phase,
        "status": "rejected",
        "job": job["id"],
        "original_manifest_path": os.path.realpath(manifest_path),
        "original_manifest_sha256": original_manifest_sha256,
        "current_manifest_sha256": (sha256_file(manifest_path)
                                     if os.path.isfile(manifest_path) else None),
        "expected": expected,
        "observed": observed,
        "error": error,
        "sentinel_argv": list(job["argv"]),
        "sentinel_output": sentinel_output,
        "child_pid": None,
        "no_child_pid_created": True,
        "sentinel_output_absent": True,
    }
    receipt = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    destination = os.path.join(
        receipts_dir, f"{phase}-{uuid.uuid4().hex}.json")
    atomic_write_new_json(destination, receipt, immutable=True)
    return destination


def _best_effort_observation(manifest: dict) -> list[dict]:
    observed = []
    for signature in manifest.get("global_input_registry", []) \
            if isinstance(manifest, dict) else []:
        if not isinstance(signature, dict):
            continue
        path = signature.get("canonical_path")
        value = {"canonical_path": path}
        try:
            value["signature"] = expected_input_signature(path)
        except Exception as exc:
            value["error"] = f"{type(exc).__name__}: {exc}"
        observed.append(value)
    return observed


class QueueAdmission:
    """Validated immutable queue plus comprehensive integrity boundaries."""

    def __init__(self, manifest_path: str, repo_root: str):
        if type(self) is not QueueAdmission:
            raise RuntimeError("QueueAdmission subclasses are not production capabilities")
        self._initialize(
            manifest_path, repo_root, validator=validate_queue_manifest,
            gate_authority=RoundwatchGateAuthority(), fixture_only=False)

    @classmethod
    def _for_fixture(cls, manifest_path: str, repo_root: str, *, validator: Callable,
                     gate_authority):
        """Private fixture-only constructor; unreachable from the production CLI."""
        if getattr(gate_authority, "fixture_only", False) is not True:
            raise RuntimeError("fixture admission requires an explicit fixture-only authority")
        value = cls.__new__(cls)
        value._initialize(
            manifest_path, repo_root, validator=validator,
            gate_authority=gate_authority, fixture_only=True)
        return value

    def _initialize(self, manifest_path: str, repo_root: str, *, validator: Callable,
                    gate_authority, fixture_only: bool) -> None:
        self.manifest_path = os.path.realpath(manifest_path)
        self.repo_root = os.path.realpath(repo_root)
        self.gate_authority = gate_authority
        self.fixture_only = fixture_only
        self.manifest: dict[str, Any] = {}
        self._program_validated = False
        self._gate_identity: dict[str, Any] | None = None
        with open(self.manifest_path, encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        try:
            self.program_context = validator(self.manifest, self.manifest_path)
        except Exception as exc:
            # A valid gate-preparation sidecar establishes that this was a
            # previously admitted program rather than an arbitrary caller JSON.
            # Only that case is entitled to an automatic window-3 receipt.
            prep_path = self.manifest.get("gate_preparation_receipt")
            try:
                with open(prep_path, encoding="utf-8") as handle:
                    prep = json.load(handle)
                prep_body = {key: prep[key] for key in prep if key != "identity_sha256"}
                trusted = (
                    prep.get("schema") == "round0005_gate_preparation_receipt.v1" and
                    prep.get("manifest_path") == self.manifest_path and
                    HASH64.fullmatch(str(prep.get("manifest_sha256", ""))) and
                    sha256_bytes(canonical_json(prep_body)) == prep.get("identity_sha256") and
                    isinstance(self.manifest.get("jobs"), list) and
                    bool(self.manifest["jobs"]) and
                    os.path.isdir(self.manifest.get("gate_receipts_dir", "")))
                if trusted:
                    expected = {
                        "manifest_sha256": prep["manifest_sha256"],
                        "global_inputs": self.manifest.get("global_input_registry"),
                    }
                    observed = {
                        "manifest_sha256": sha256_file(self.manifest_path),
                        "global_inputs": _best_effort_observation(self.manifest),
                    }
                    receipt = _emit_component_rejection_receipt(
                        receipts_dir=self.manifest["gate_receipts_dir"],
                        phase="gate-preparation-to-admission",
                        manifest_path=self.manifest_path,
                        original_manifest_sha256=prep["manifest_sha256"],
                        job=self.manifest["jobs"][0], expected=expected,
                        observed=observed,
                        error=f"{type(exc).__name__}: {exc}")
                    try:
                        exc.add_note(f"automatic admission rejection receipt: {receipt}")
                    except AttributeError:
                        pass
            except Exception as receipt_exc:
                try:
                    exc.add_note(
                        f"window-3 receipt unavailable because gate sidecar was invalid: "
                        f"{receipt_exc}")
                except AttributeError:
                    pass
            raise
        self._program_validated = True
        if self.repo_root != os.path.realpath(self.manifest["repo_root"]):
            raise RuntimeError("caller repo_root differs from gate-hashed queue repo_root")
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.original_manifest_signature = expected_input_signature(self.manifest_path)
        from .gate_preparation import validate_gate_preparation_receipt
        self.gate_preparation = validate_gate_preparation_receipt(
            self.manifest["gate_preparation_receipt"],
            manifest_path=self.manifest_path, manifest=self.manifest)
        if self.gate_preparation["manifest_sha256"] != self.manifest_sha256:
            raise RuntimeError("published manifest differs from prepared gate receipt")
        self.gate_preparation_signature = expected_input_signature(
            self.manifest["gate_preparation_receipt"])
        self.launch_checkout = self._verify_checkout()
        self.initial_environment = self._verify_environment_binding()
        self.resolved_venv = self.initial_environment["resolved_venv"]
        self.initial_cache_policy = self._verify_cache_policy()
        self.release_receipt_path = self._release_receipt_path()
        self.release_receipt_signature = self._release_receipt_signature()
        self.initial_release = validate_release_preflight_receipt(
            self.release_receipt_path,
            expected_identity_sha256=self.manifest["release_preflight_identity"],
            expected_signature=self.release_receipt_signature)
        if self.initial_release["identity_sha256"] != self.manifest["release_preflight_identity"]:
            raise RuntimeError("queue release-preflight identity differs from immutable receipt")
        self.expected_inputs = self._expected_input_registry()
        self.initial_integrity = self._integrity_state(include_output_absence=False)
        self.construction_receipt_path = self._write_receipt(
            phase="admission-construction", status="matched", job=None,
            expected=self.initial_integrity, observed=self.initial_integrity,
            error=None, child_pid=None)
        self.__auth_nonce = secrets.token_hex(32)
        self.__controller_claimed = False
        self.__manifest_object_sha256 = sha256_bytes(canonical_json(self.manifest))
        self.__gate_authority_object = self.gate_authority
        self.__gate_authority_check = type(self.gate_authority).check
        with _ADMISSION_REGISTRY_LOCK:
            _ADMISSION_REGISTRY[id(self)] = (weakref.ref(self), self.__auth_nonce)

    def __copy__(self):
        raise RuntimeError("QueueAdmission capabilities cannot be copied")

    def __deepcopy__(self, _memo):
        raise RuntimeError("QueueAdmission capabilities cannot be copied")

    def _is_authentic_capability(self, *, claimed: bool | None) -> bool:
        """Authenticate object identity and its still-sealed in-memory state."""
        if type(self) is not QueueAdmission:
            return False
        try:
            with _ADMISSION_REGISTRY_LOCK:
                registered = _ADMISSION_REGISTRY.get(id(self))
                registered_ok = (
                    registered is not None and registered[0]() is self and
                    registered[1] == self.__auth_nonce)
            state_ok = (
                self._program_validated is True and
                self.gate_authority is self.__gate_authority_object and
                type(self.gate_authority).check is self.__gate_authority_check and
                (self.fixture_only or (
                    type(self.gate_authority) is RoundwatchGateAuthority and
                    "check" not in vars(self.gate_authority))) and
                sha256_bytes(canonical_json(self.manifest)) ==
                self.__manifest_object_sha256 and
                sha256_file(self.manifest_path) == self.manifest_sha256 and
                self.repo_root == os.path.realpath(self.manifest["repo_root"]) and
                self.manifest_path == os.path.realpath(self.manifest_path))
            claimed_ok = (claimed is None or self.__controller_claimed is claimed)
            return bool(registered_ok and state_ok and claimed_ok)
        except Exception:
            return False

    def _claim_controller(self, *, fixture_only: bool, jobs: list | None,
                          controller_id: str) -> tuple[list, dict]:
        """Consume the one controller entry capability after a live boundary."""
        if type(self) is not QueueAdmission:
            raise RuntimeError("controller requires the exact QueueAdmission type")
        if not self._is_authentic_capability(claimed=False):
            raise RuntimeError("QueueAdmission capability is copied or unauthentic")
        with _ADMISSION_REGISTRY_LOCK:
            registered = _ADMISSION_REGISTRY.get(id(self))
            if (registered is None or registered[0]() is not self or
                    registered[1] != getattr(self, "_QueueAdmission__auth_nonce", None)):
                raise RuntimeError("QueueAdmission capability is copied or unauthentic")
            if self.__controller_claimed:
                raise RuntimeError("QueueAdmission controller capability was already consumed")
            if self.fixture_only is not fixture_only:
                lane = "fixture" if fixture_only else "production"
                raise RuntimeError(f"{lane} controller received the wrong admission class")
            if not self._program_validated or not self.manifest_sha256:
                raise RuntimeError("QueueAdmission is partially initialized")
            self.__controller_claimed = True
        runtime_jobs = QueueAdmission.runtime_jobs(self) if jobs is None else jobs
        QueueAdmission.assert_runtime_jobs(self, runtime_jobs)
        if not runtime_jobs:
            raise RuntimeError("QueueAdmission contains no controller jobs")
        try:
            entry = QueueAdmission.boundary(self, runtime_jobs[0].name)
        except Exception as exc:
            raise ControllerEntryBoundaryFailure(runtime_jobs[0].name, exc) from exc
        process = _controller_process_record(os.getpid())
        claim_body = {
            "schema": "round0005_queue_controller_claim.v1",
            "admission_id": id(self),
            "admission_nonce": self.__auth_nonce,
            "claim_nonce": secrets.token_hex(32),
            "controller_id": controller_id,
            "controller_pid": os.getpid(),
            "controller_starttime_ticks": process["proc_starttime_ticks"],
            "controller_process": process,
            "fixture_only": fixture_only,
            "manifest": expected_input_signature(self.manifest_path),
            "construction_receipt": expected_input_signature(
                self.construction_receipt_path),
            "ordered_job_ids": [job.name for job in runtime_jobs],
            "entry_gate_sha256": sha256_bytes(canonical_json(entry)),
        }
        controller_claim = {
            **claim_body,
            "identity_sha256": sha256_bytes(canonical_json(claim_body)),
        }
        capability = {
            "admission_id": id(self), "nonce": self.__auth_nonce,
            "manifest_sha256": self.manifest_sha256,
            "fixture_only": fixture_only, "entry_gate": entry,
            "controller_claim": controller_claim,
        }
        return runtime_jobs, capability

    def _verify_controller_capability(self, capability: dict, *, fixture_only: bool) -> None:
        valid = self._is_authentic_capability(claimed=True)
        expected = {
            "admission_id": id(self), "nonce": self.__auth_nonce,
            "manifest_sha256": self.manifest_sha256,
            "fixture_only": fixture_only,
        }
        if (not valid or not isinstance(capability, dict) or
                any(capability.get(key) != value for key, value in expected.items()) or
                not isinstance(capability.get("entry_gate"), dict) or
                not isinstance(capability.get("controller_claim"), dict)):
            raise RuntimeError("controller capability is not authentic/current")
        claim = capability["controller_claim"]
        claim_body = {key: claim[key] for key in claim if key != "identity_sha256"}
        if (claim.get("schema") != "round0005_queue_controller_claim.v1" or
                claim.get("admission_id") != id(self) or
                claim.get("admission_nonce") != self.__auth_nonce or
                claim.get("fixture_only") is not fixture_only or
                claim.get("manifest") != expected_input_signature(self.manifest_path) or
                claim.get("construction_receipt") != expected_input_signature(
                    self.construction_receipt_path) or
                claim.get("entry_gate_sha256") != sha256_bytes(canonical_json(
                    capability["entry_gate"])) or
                claim.get("identity_sha256") != sha256_bytes(canonical_json(claim_body))):
            raise RuntimeError("controller claim is not the issued QueueAdmission claim")

    def _release_receipt_path(self) -> str:
        if self.fixture_only and "release_preflight_receipt" in self.manifest:
            return self.manifest["release_preflight_receipt"]
        for entry in self.manifest["program_inputs"]:
            if entry["role"] == "release_preflight_receipt":
                return entry["signature"]["canonical_path"]
        raise RuntimeError("queue has no release preflight receipt role")

    def _release_receipt_signature(self) -> dict[str, Any]:
        matches = [signature for signature in self.manifest["global_input_registry"]
                   if signature.get("canonical_path") == self.release_receipt_path]
        if len(matches) != 1:
            raise RuntimeError("queue has no unique sealed release preflight signature")
        return matches[0]

    def _receipt_dir(self) -> str:
        value = self.manifest["gate_receipts_dir"]
        if (not _contained_by_data(value) or not os.path.isdir(value) or
                os.path.islink(value)):
            raise RuntimeError("gate receipt root is not a regular /data directory")
        return os.path.realpath(value)

    def _write_receipt(self, *, phase: str, status: str, job: str | None,
                       expected: Any, observed: Any, error: str | None,
                       child_pid: int | None) -> str:
        job_entry = None
        if job is not None:
            matches = [value for value in self.manifest.get("jobs", [])
                       if value.get("id") == job]
            job_entry = matches[0] if len(matches) == 1 else None
        sentinel_argv = list(job_entry["argv"]) if job_entry else None
        sentinel_output = (job_entry["outputs"][0]
                           if job_entry and job_entry.get("outputs") else None)
        body = {
            "schema": "round0005_integrity_receipt.v3",
            "phase": phase,
            "status": status,
            "job": job,
            "original_manifest_path": self.manifest_path,
            "original_manifest_sha256": getattr(self, "manifest_sha256", None),
            "current_manifest_sha256": (sha256_file(self.manifest_path)
                                         if os.path.isfile(self.manifest_path) else None),
            "expected": expected,
            "observed": observed,
            "error": error,
            "sentinel_argv": sentinel_argv,
            "sentinel_output": sentinel_output,
            "child_pid": child_pid,
            "no_child_pid_created": child_pid is None,
            "sentinel_output_absent": (sentinel_output is None or
                                       not os.path.lexists(sentinel_output)),
        }
        receipt = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
        path = os.path.join(
            self._receipt_dir(), f"{phase}-{uuid.uuid4().hex}.json")
        atomic_write_new_json(path, receipt, immutable=True)
        return path

    def _verify_checkout(self) -> dict:
        state = git_checkout_state(self.repo_root)
        if (state["head"] != self.manifest["release_sha"] or
                state["detached"] is not True or state["clean"] is not True):
            raise RuntimeError(
                f"queue checkout is not the clean detached release: {state!r}")
        return state

    def _verify_environment_binding(self) -> dict:
        path = os.path.realpath(self.manifest["environment_manifest"])
        with open(path, encoding="utf-8") as handle:
            environment = json.load(handle)
        if (environment.get("freeze_sha256") != self.manifest["environment_freeze_sha"] or
                environment.get("identity_sha256") !=
                self.manifest["environment_identity_sha"]):
            raise RuntimeError("queue environment identities differ from sealed manifest")
        freeze = os.path.realpath(environment["freeze_file"])
        with open(freeze, encoding="utf-8") as handle:
            lines = sorted(line.strip() for line in handle if line.strip())
        freeze_sha = sha256_bytes(
            "".join(line + "\n" for line in lines).encode("utf-8"))
        if freeze_sha != self.manifest["environment_freeze_sha"]:
            raise RuntimeError("installed-package freeze changed after sealing")
        venv = os.path.realpath(environment["venv_path"])
        python = os.path.realpath(os.path.join(venv, "bin", "python"))
        if not os.path.isdir(venv) or not os.path.isfile(python):
            raise RuntimeError("sealed venv/Python executable is missing")
        gpu_uuid = environment.get("gpu_uuid")
        gpu_observation = None
        if not self.fixture_only and (
                not isinstance(gpu_uuid, str) or
                self.manifest["child_environment"].get("CUDA_VISIBLE_DEVICES") != gpu_uuid):
            raise RuntimeError("sealed GPU UUID differs from the exact child policy")
        if not self.fixture_only:
            gpu_observation = validate_canonical_gpu_environment(environment)
        return {
            "manifest": expected_input_signature(path),
            "freeze": expected_input_signature(freeze),
            "freeze_sha256": freeze_sha,
            "identity_sha256": environment["identity_sha256"],
            "resolved_venv": venv,
            "python_executable": expected_input_signature(python),
            "gpu_uuid": gpu_uuid,
            "gpu_observation": gpu_observation,
        }

    def _verify_cache_policy(self) -> dict[str, Any]:
        """Bind no-follow cache directory inodes and keep pycache empty forever."""
        return observe_round0005_cache_policy(self.manifest)

    def _expected_input_registry(self) -> list[dict]:
        return list(self.manifest["global_input_registry"])

    def _observed_inputs(self) -> list[dict]:
        observed = []
        for signature in self.expected_inputs:
            path = signature["canonical_path"]
            try:
                observed.append(expected_input_signature(path))
            except Exception as exc:
                observed.append({"canonical_path": path,
                                 "error": f"{type(exc).__name__}: {exc}"})
        return observed

    def _output_absence(self, job_name: str) -> dict[str, bool]:
        job = self._job_entry(job_name)
        paths = [*job["outputs"], job["done_marker"], job["log"], job["manifest"]]
        return {path: not os.path.lexists(path) for path in paths}

    def _integrity_state(self, *, job_name: str | None = None,
                         include_output_absence: bool) -> dict[str, Any]:
        if self.fixture_only:
            source_closure = self.manifest["source_closure"]
        elif self.manifest.get("round_id") == "0015":
            from .source_closure import validate_round0015_source_closure_receipt

            source_closure = validate_round0015_source_closure_receipt(
                self.manifest["source_closure"], repo_root=self.repo_root)
        elif self.manifest.get("round_id") == "0014":
            from .source_closure import validate_round0014_source_closure_receipt

            source_closure = validate_round0014_source_closure_receipt(
                self.manifest["source_closure"], repo_root=self.repo_root)
        else:
            source_closure = validate_source_closure_receipt(
                self.manifest["source_closure"], repo_root=self.repo_root)
        state: dict[str, Any] = {
            "manifest": (expected_input_signature(self.manifest_path)
                         if os.path.isfile(self.manifest_path) else None),
            "checkout": self._verify_checkout(),
            "environment": self._verify_environment_binding(),
            "release_preflight": validate_release_preflight_receipt(
                self.release_receipt_path,
                expected_identity_sha256=self.manifest["release_preflight_identity"],
                expected_signature=self.release_receipt_signature),
            "gate_preparation": expected_input_signature(
                self.manifest["gate_preparation_receipt"]),
            "roundwatch_binding": validate_roundwatch_binding(
                self.manifest["roundwatch_binding"]),
            "source_closure": source_closure,
            "cache_policy": self._verify_cache_policy(),
            "global_inputs": self._observed_inputs(),
        }
        if include_output_absence:
            if job_name is None:
                raise ValueError("output-absence integrity needs a job name")
            state["output_absence"] = self._output_absence(job_name)
        return state

    def _expected_integrity(self, *, job_name: str | None,
                            include_output_absence: bool) -> dict[str, Any]:
        expected = {
            "manifest": self.original_manifest_signature,
            "checkout": self.launch_checkout,
            "environment": self.initial_environment,
            "release_preflight": self.initial_release,
            "gate_preparation": self.gate_preparation_signature,
            "roundwatch_binding": self.manifest["roundwatch_binding"],
            "source_closure": self.manifest["source_closure"],
            "cache_policy": self.initial_cache_policy,
            "global_inputs": self.expected_inputs,
        }
        if include_output_absence:
            expected["output_absence"] = {
                path: True for path in self._output_absence(job_name)
            }
        return expected

    @staticmethod
    def _require_integrity(expected: dict, observed: dict, *, phase: str) -> None:
        if expected != observed:
            raise IntegrityMismatch(
                f"Round 0005 comprehensive integrity changed during {phase}: "
                f"expected={expected!r} observed={observed!r}",
                expected=expected, observed=observed, phase=phase)

    def comprehensive_integrity_boundary(
            self, job_name: str, *, phase: str, include_output_absence: bool,
            after_comparison_hook=None, runtime_expected=None,
            runtime_probe=None) -> dict[str, Any]:
        """Pure comparison used immediately before Popen and after child exit."""
        if not self._is_authentic_capability(claimed=None):
            raise RuntimeError("QueueAdmission capability state changed or is unauthentic")
        expected = self._expected_integrity(
            job_name=job_name, include_output_absence=include_output_absence)
        observed = self._integrity_state(
            job_name=job_name, include_output_absence=include_output_absence)
        if runtime_probe is not None:
            expected["runtime"] = runtime_expected
            observed["runtime"] = runtime_probe()
        self._require_integrity(expected, observed, phase=phase)
        if after_comparison_hook is not None:
            if not self.fixture_only:
                raise RuntimeError("production admission exposes no adversarial hook")
            after_comparison_hook({"phase": phase, "expected": expected,
                                   "observed": observed})
            observed = self._integrity_state(
                job_name=job_name, include_output_absence=include_output_absence)
            if runtime_probe is not None:
                observed["runtime"] = runtime_probe()
            self._require_integrity(expected, observed, phase=phase)
        return {"phase": phase, "expected": expected, "observed": observed,
                "integrity_match": True}

    def _current_gate_boundary(self, job_name: str, *, phase: str,
                               include_output_absence: bool) -> dict[str, Any]:
        """Validate local state and the stable current control/gate identity."""
        self._job_entry(job_name)
        local = self.comprehensive_integrity_boundary(
            job_name, phase=phase,
            include_output_absence=include_output_absence)
        receipt_phase = ("gate-boundary" if include_output_absence else
                         "terminal-gate-boundary")
        try:
            gate = self.__gate_authority_check(
                self.gate_authority,
                manifest=self.manifest, manifest_path=self.manifest_path,
                manifest_sha256=self.manifest_sha256)
            identity = gate["event_identity"]
            stable = {
                "instance_id": identity["instance_id"],
                "gate_id": identity["gate_id"],
                "authority": identity["authority"],
                "round_event": identity["round_event"],
                "gate_prepared_event": identity["gate_prepared_event"],
                "control_event": identity["control_event"],
            }
            if self._gate_identity is None:
                self._gate_identity = stable
            elif stable != self._gate_identity:
                raise RuntimeError(
                    f"Roundwatch current gate/event identity changed: "
                    f"expected={self._gate_identity!r} observed={stable!r}")
        except Exception as exc:
            path = self._write_receipt(
                phase=receipt_phase, status="rejected", job=job_name,
                expected=local["expected"], observed=local["observed"],
                error=f"{type(exc).__name__}: {exc}", child_pid=None)
            try:
                exc.add_note(f"automatic gate rejection receipt: {path}")
            except AttributeError:
                pass
            raise
        receipt = self._write_receipt(
            phase=receipt_phase, status="matched", job=job_name,
            expected=local["expected"], observed=local["observed"],
            error=None, child_pid=None)
        return {"receipt_path": receipt, "gate": gate,
                "comprehensive_integrity": local}

    def boundary(self, job_name: str) -> dict[str, Any]:
        """Prelaunch gate boundary; this job's outputs must not yet exist."""
        return self._current_gate_boundary(
            job_name, phase=f"pre-gate:{job_name}",
            include_output_absence=True)

    def terminal_boundary(self, job_name: str) -> dict[str, Any]:
        """Final current-control boundary after every signed output exists."""
        return self._current_gate_boundary(
            job_name, phase=f"terminal-gate:{job_name}",
            include_output_absence=False)

    def record_integrity_failure(self, job_name: str, exc: Exception, *,
                                 phase: str, child_pid: int | None) -> str:
        expected = (exc.expected if isinstance(exc, IntegrityMismatch) else
                    self._expected_integrity(
                        job_name=job_name,
                        include_output_absence=child_pid is None))
        observed = (exc.observed if isinstance(exc, IntegrityMismatch) else
                    _best_effort_observation(self.manifest))
        return self._write_receipt(
            phase=phase, status="rejected", job=job_name,
            expected=expected, observed=observed,
            error=f"{type(exc).__name__}: {exc}", child_pid=child_pid)

    # Compatibility names retain strict comprehensive behavior.
    def final_expected_input_comparison(self, job_name: str,
                                        after_comparison_hook=None) -> dict:
        return self.comprehensive_integrity_boundary(
            job_name, phase=f"final-child-launch:{job_name}",
            include_output_absence=True,
            after_comparison_hook=after_comparison_hook)

    def record_final_comparison_failure(self, job_name: str, exc: Exception) -> str:
        return self.record_integrity_failure(
            job_name, exc, phase="final-child-launch", child_pid=None)

    def prelaunch(self, job_name: str) -> dict:
        return self.final_expected_input_comparison(job_name)

    def _job_entry(self, job_name: str) -> dict:
        matches = [job for job in self.manifest["jobs"] if job["id"] == job_name]
        if len(matches) != 1:
            raise RuntimeError(f"queue must contain exactly one job {job_name!r}")
        return matches[0]

    def manifest_job_contracts(self) -> list[dict]:
        return [canonical_manifest_job(job) for job in self.manifest["jobs"]]

    def runtime_jobs(self) -> list:
        from .run_controller import Job

        jobs = []
        for raw in self.manifest["jobs"]:
            policy = raw["node_policy"]
            jobs.append(Job(
                name=raw["id"], argv=list(raw["argv"]), outputs=list(raw["outputs"]),
                done_marker=raw["done_marker"], deps=list(raw["deps"]), cwd=raw["cwd"],
                log=raw["log"], manifest=raw["manifest"],
                required_free_gb=float(policy["required_free_gb"]),
                input_paths=list(raw["inputs"]), expected_inputs=list(raw["expected_inputs"]),
                predicted_wall_s=float(raw["predicted_wall_s"]),
                p90_wall_s=float(raw["p90_wall_s"]),
                scientific_rows=int(policy["scientific_rows"]),
                gpu_memory_cap_mb=int(policy["gpu_memory_cap_mb"]),
                node_policy=dict(policy),
            ))
        self.assert_runtime_jobs(jobs)
        return jobs

    def assert_runtime_jobs(self, jobs: list) -> None:
        observed = []
        for job in jobs:
            if not hasattr(job, "manifest_contract"):
                raise RuntimeError("runtime Job lacks exact manifest serialization")
            observed.append(job.manifest_contract())
        expected = self.manifest_job_contracts()
        if observed != expected:
            raise RuntimeError(
                "runtime jobs differ from exact ordered gate-hashed manifest: "
                f"expected={expected!r} observed={observed!r}")


def validate_mutation_window_receipt(path: str, *, expected_window: str) -> dict:
    """Validate an automatically-emitted production-path integration receipt."""
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    if receipt.get("schema") != "round0005_integrity_receipt.v3":
        raise ValueError("mutation receipt schema is unsupported")
    if receipt.get("phase") != expected_window or receipt.get("status") != "rejected":
        raise ValueError("mutation receipt phase/status mismatch")
    identity = receipt.get("identity_sha256")
    body = {key: receipt[key] for key in receipt if key != "identity_sha256"}
    if (not HASH64.fullmatch(str(identity)) or
            sha256_bytes(canonical_json(body)) != identity or
            receipt.get("expected") == receipt.get("observed") or
            receipt.get("no_child_pid_created") is not True or
            receipt.get("child_pid") is not None or
            receipt.get("sentinel_output_absent") is not True or
            (receipt.get("sentinel_output") and
             os.path.lexists(receipt["sentinel_output"]))):
        raise ValueError("mutation receipt is forged or did not prove pre-Popen rejection")
    return receipt
