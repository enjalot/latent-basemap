"""Exact CUDA-hidden integration fixture for the Round 0005 production path."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone

from .artifact_identity import (canonical_json, expected_input_signature,
                                git_checkout_state, sha256_bytes)
from .queue_admission import (MUTATION_WINDOWS,
                              validate_mutation_window_receipt)
from .release_preflight import validate_release_preflight_receipt
from .round0005_program import ROUND0005_JOB_FIELDS
from .roundwatch_gate import validate_roundwatch_binding
from .source_closure import (ROUND0005_RUNTIME_ENTRYPOINTS,
                             _source_closure_receipt_fixture_only,
                             _validate_source_closure_receipt_fixture_only)

FIXTURE_SCHEMA = "round0005_all_node_fixture.v3"
FIXTURE_QUEUE_SCHEMA = "round0005_fixture_queue.v2"
FULL_SHA = re.compile(r"[0-9a-f]{40}")
HASH64 = re.compile(r"[0-9a-f]{64}")
SIX_NODE_IDS = (
    "expected-input-contract", "semantic-render-contract", "query-cache-contract",
    "chunk-resume-contract", "all-node-contract", "controller-sentinel-contract",
)
FIXTURE_SOURCE_ENTRYPOINTS = tuple(dict.fromkeys((
    *ROUND0005_RUNTIME_ENTRYPOINTS,
    "basemap/round0005_fixture.py",
    "experiments/run_round0005_fixture.py",
    "experiments/run_round0005_fixture_node.py",
)))
FIXTURE_QUEUE_FIELDS = {
    "schema", "program", "round_id", "round_sha256", "release_sha",
    "environment_freeze_sha", "environment_identity_sha", "gpu_hours_cap",
    "environment_manifest", "deadline_utc", "repo_root", "queue_root", "cache_environment",
    "child_environment", "gate_receipts_dir", "controller_checkpoints_dir",
    "controller_terminal_summary", "gate_preparation_receipt", "lease_path",
    "allowed_processes", "jobs", "global_input_registry", "fixture_input",
    "source_closure", "roundwatch_binding", "release_preflight_identity",
    "release_preflight_receipt",
}


def fixture_source_closure(repo_root: str) -> dict:
    return _source_closure_receipt_fixture_only(
        repo_root, entrypoints=FIXTURE_SOURCE_ENTRYPOINTS)


def fixture_identity_payload(report: dict) -> dict:
    return {key: report[key] for key in sorted(report) if key != "identity_sha256"}


def _under(root: str, path: str) -> bool:
    try:
        return os.path.commonpath([root, path]) == root
    except (TypeError, ValueError):
        return False


def _future(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return False
    return parsed.tzinfo is not None and parsed.astimezone(timezone.utc) > \
        datetime.now(timezone.utc)


def _environment_state(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        environment = json.load(handle)
    freeze = os.path.realpath(environment["freeze_file"])
    with open(freeze, encoding="utf-8") as handle:
        freeze_sha = sha256_bytes(
            "".join(line.strip() + "\n" for line in sorted(
                line.strip() for line in handle if line.strip())).encode("utf-8"))
    python = os.path.realpath(
        os.path.join(os.path.realpath(environment["venv_path"]), "bin", "python"))
    if not os.path.isfile(python) or os.path.islink(python):
        raise RuntimeError("fixture sealed Python executable is missing or symlinked")
    return {
        "manifest": expected_input_signature(path),
        "freeze": expected_input_signature(freeze),
        "freeze_sha256": freeze_sha,
        "identity_sha256": environment["identity_sha256"],
        "venv_path": os.path.realpath(environment["venv_path"]),
        "python": expected_input_signature(python),
    }


def validate_fixture_queue(data: dict, path: str) -> dict:
    """Fixture-only exact program validator used by builder, gate, and admission."""
    if not isinstance(data, dict) or set(data) != FIXTURE_QUEUE_FIELDS:
        raise ValueError("fixture queue fields are incomplete or unknown")
    queue_path = os.path.abspath(path)
    queue_root = os.path.realpath(data["queue_root"])
    repo_root = os.path.realpath(data["repo_root"])
    if (data["schema"] != FIXTURE_QUEUE_SCHEMA or data["program"] != "basemap-100m" or
            data["round_id"] != "0005" or not HASH64.fullmatch(data["round_sha256"]) or
            not FULL_SHA.fullmatch(data["release_sha"]) or
            queue_path != os.path.join(queue_root, "queue.json") or
            not _under("/data", queue_root) or os.path.realpath(queue_path) != queue_path or
            not os.path.isdir(repo_root) or not _future(data["deadline_utc"]) or
            float(data["gpu_hours_cap"]) != 0.75):
        raise ValueError("fixture queue identity/root/deadline is invalid")
    for key in ("gate_receipts_dir", "controller_checkpoints_dir"):
        value = data[key]
        if (not _under(queue_root, value) or os.path.realpath(value) != value or
                not os.path.isdir(value) or os.path.islink(value)):
            raise ValueError(f"fixture queue {key} is not a regular contained directory")
    for key in ("controller_terminal_summary", "gate_preparation_receipt", "lease_path"):
        value = data[key]
        if not _under(queue_root, value) or os.path.realpath(value) != value:
            raise ValueError(f"fixture queue {key} escapes its root")
    if data["allowed_processes"] != []:
        raise ValueError("CUDA-hidden fixture cannot allow GPU service processes")

    release_signatures = [
        signature for signature in data.get("global_input_registry", [])
        if isinstance(signature, dict) and signature.get("canonical_path") ==
        data["release_preflight_receipt"]]
    if len(release_signatures) != 1:
        raise ValueError("fixture queue lacks one sealed release preflight signature")
    release = validate_release_preflight_receipt(
        data["release_preflight_receipt"],
        expected_identity_sha256=data["release_preflight_identity"],
        expected_signature=release_signatures[0])
    if (release["identity_sha256"] != data["release_preflight_identity"] or
            release["release_sha"] != data["release_sha"] or
            release["run_checkout_path"] != repo_root or
            release["cache_environment"] != data["cache_environment"]):
        raise ValueError("fixture release preflight binding changed")
    environment = _environment_state(release["environment_manifest_path"])
    if os.path.realpath(data["environment_manifest"]) != release["environment_manifest_path"]:
        raise ValueError("fixture environment manifest differs from release receipt")
    if (environment["freeze_sha256"] != data["environment_freeze_sha"] or
            environment["identity_sha256"] != data["environment_identity_sha"]):
        raise ValueError("fixture environment identity changed")
    cache = data["cache_environment"]
    if (not isinstance(cache, dict) or cache.get("PYTHONDONTWRITEBYTECODE") != "1" or
            set(cache) != set(release["cache_environment"])):
        raise ValueError("fixture cache environment fields are invalid")
    for key, value in cache.items():
        if key != "PYTHONDONTWRITEBYTECODE" and (
                not _under(queue_root, value) or not os.path.isdir(value)):
            raise ValueError(f"fixture cache {key} is not a fresh contained directory")
    child = data["child_environment"]
    expected_child = {
        **cache, "CUDA_VISIBLE_DEVICES": "",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "TOKENIZERS_PARALLELISM": "false",
    }
    if child != expected_child:
        raise ValueError("fixture children are not exactly CUDA-hidden and cache-isolated")

    validate_roundwatch_binding(data["roundwatch_binding"])
    _validate_source_closure_receipt_fixture_only(
        data["source_closure"], repo_root=repo_root,
        entrypoints=FIXTURE_SOURCE_ENTRYPOINTS)
    fixture_input = data["fixture_input"]
    if expected_input_signature(fixture_input["canonical_path"]) != fixture_input:
        raise ValueError("fixture dependency changed")
    registry = data["global_input_registry"]
    if not isinstance(registry, list) or not registry:
        raise ValueError("fixture queue global input registry is empty")
    paths = [value.get("canonical_path") for value in registry
             if isinstance(value, dict)]
    if len(paths) != len(registry) or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("fixture global input registry is not sorted and unique")
    for signature in registry:
        if expected_input_signature(signature["canonical_path"]) != signature:
            raise ValueError(f"fixture global input changed: {signature['canonical_path']}")
    required_inputs = {
        fixture_input["canonical_path"], data["release_preflight_receipt"],
        environment["manifest"]["canonical_path"],
        environment["freeze"]["canonical_path"], environment["python"]["canonical_path"],
        data["roundwatch_binding"]["cli"]["path"],
        data["roundwatch_binding"]["interpreter"]["canonical_path"],
        data["roundwatch_binding"]["git"]["canonical_path"],
        *[entry["canonical_path"]
          for entry in data["roundwatch_binding"]["import_closure"]],
        *[entry["signature"]["canonical_path"]
          for entry in data["source_closure"]["members"]],
    }
    if set(paths) != required_inputs:
        raise ValueError("fixture global registry is not the full fixture/runtime closure")

    jobs = data["jobs"]
    if (not isinstance(jobs, list) or len(jobs) != 6 or
            [job.get("id") for job in jobs if isinstance(job, dict)] != list(SIX_NODE_IDS)):
        raise ValueError("fixture queue must contain the exact six ordered nodes")
    helper = os.path.join(repo_root, "experiments", "run_round0005_fixture_node.py")
    python = release["python_invocation_path"]
    output_paths = []
    for position, (node, job) in enumerate(zip(SIX_NODE_IDS, jobs)):
        if set(job) != ROUND0005_JOB_FIELDS:
            raise ValueError(f"fixture job {node} fields differ from runtime serialization")
        output = os.path.join(queue_root, "node-outputs", f"{position:02d}-{node}.json")
        expected_argv = [python, helper, "--node", node, "--out", output,
                         "--repo-root", repo_root]
        policy = {
            "schema": "round0005_fixture_node_policy.v1", "node_id": node,
            "canonical_script": "experiments/run_round0005_fixture_node.py",
            "training_performed": False, "gpu_required": False,
            "cuda_device_count": 0, "scientific_rows": 1,
            "required_free_gb": 0.0, "gpu_memory_cap_mb": 0,
            "scale_certificate_required": False,
        }
        expected_deps = [] if position == 0 else [SIX_NODE_IDS[position - 1]]
        if (job["argv"] != expected_argv or job["outputs"] != [output] or
                job["deps"] != expected_deps or job["cwd"] != repo_root or
                job["inputs"] != paths or job["expected_inputs"] != registry or
                job["node_policy"] != policy or job["predicted_wall_s"] != 10.0 or
                job["p90_wall_s"] != 30.0):
            raise ValueError(f"fixture job {node} differs from its exact derived contract")
        for value in [*job["outputs"], job["done_marker"], job["log"], job["manifest"]]:
            if not _under(queue_root, value) or os.path.realpath(value) != value:
                raise ValueError(f"fixture job {node} output/control escapes queue root")
            output_paths.append(value)
    if len(output_paths) != len(set(output_paths)) or set(output_paths) & set(paths):
        raise ValueError("fixture graph aliases outputs/controls with each other or inputs")
    return {"fixture_only": True, "release": release, "environment": environment}


def validate_round0005_fixture(path: str, *, repo_root: str, release_sha: str,
                               environment_manifest: str) -> dict:
    """Re-probe release, environment, sources, six Popen children, and windows."""
    if not FULL_SHA.fullmatch(release_sha):
        raise ValueError("fixture release SHA must be a full immutable Git SHA")
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    required = {
        "schema", "passed", "release_sha", "round_id", "round_sha256", "checkout",
        "environment", "python", "cuda_hidden", "source_closure", "six_node_queue",
        "mutation_windows", "identity_sha256",
    }
    if not isinstance(report, dict) or set(report) != required:
        raise ValueError("Round 0005 fixture fields are incomplete or unknown")
    if (report["schema"] != FIXTURE_SCHEMA or report["passed"] is not True or
            report["round_id"] != "0005" or report["release_sha"] != release_sha or
            not HASH64.fullmatch(report["round_sha256"]) or
            sha256_bytes(canonical_json(fixture_identity_payload(report))) !=
            report["identity_sha256"]):
        raise ValueError("Round 0005 fixture identity/status is invalid")
    checkout = git_checkout_state(repo_root)
    if (checkout != report["checkout"] or checkout["head"] != release_sha or
            checkout["clean"] is not True or checkout["detached"] is not True):
        raise ValueError("Round 0005 fixture is stale, dirty, or not the detached release")
    environment = _environment_state(os.path.realpath(environment_manifest))
    expected_environment = {key: environment[key] for key in
                            ("manifest", "freeze", "freeze_sha256",
                             "identity_sha256", "venv_path")}
    if report["environment"] != expected_environment or report["python"] != {
            "canonical_path": environment["python"]["canonical_path"],
            "signature": environment["python"]}:
        raise ValueError("Round 0005 fixture environment/Python identity changed")
    if report["cuda_hidden"] != {
            "CUDA_VISIBLE_DEVICES": "", "all_children_proved_hidden": True,
            "cuda_api_calls": 0}:
        raise ValueError("Round 0005 fixture did not prove CUDA was hidden and untouched")
    _validate_source_closure_receipt_fixture_only(
        report["source_closure"], repo_root=repo_root,
        entrypoints=FIXTURE_SOURCE_ENTRYPOINTS)

    queue = report["six_node_queue"]
    queue_fields = {
        "node_ids", "queue_manifest", "queue_signature", "terminal_summary",
        "terminal_signature", "popen_count", "child_pids", "children",
    }
    if (not isinstance(queue, dict) or set(queue) != queue_fields or
            tuple(queue["node_ids"]) != SIX_NODE_IDS or queue["popen_count"] != 6 or
            len(queue["children"]) != 6 or len(queue["child_pids"]) != 6 or
            len(set(queue["child_pids"])) != 6):
        raise ValueError("Round 0005 fixture did not execute exactly six unique children")
    if (expected_input_signature(queue["queue_manifest"]) != queue["queue_signature"] or
            expected_input_signature(queue["terminal_summary"]) !=
            queue["terminal_signature"]):
        raise ValueError("Round 0005 fixture queue/terminal bytes changed")
    with open(queue["queue_manifest"], encoding="utf-8") as handle:
        fixture_queue = json.load(handle)
    validate_fixture_queue(fixture_queue, queue["queue_manifest"])
    with open(queue["terminal_summary"], encoding="utf-8") as handle:
        terminal = json.load(handle)
    records = terminal.get("jobs") or []
    if (terminal.get("terminal_verdict") != "passed" or
            terminal.get("required_jobs") != list(SIX_NODE_IDS) or
            terminal.get("completed_jobs") != list(SIX_NODE_IDS) or
            [entry.get("child_pid") for entry in records] != queue["child_pids"] or
            any(entry.get("status") != "ok" for entry in records)):
        raise ValueError("Round 0005 six-node controller terminal evidence is incomplete")
    for position, child in enumerate(queue["children"]):
        fields = {"id", "pid", "output", "output_signature", "cuda_visible_devices",
                  "cuda_probe_performed"}
        if (not isinstance(child, dict) or set(child) != fields or
                child["id"] != SIX_NODE_IDS[position] or
                child["pid"] != queue["child_pids"][position] or
                child["cuda_visible_devices"] != "" or
                child["cuda_probe_performed"] is not False or
                expected_input_signature(child["output"]) != child["output_signature"]):
            raise ValueError(f"Round 0005 fixture child {position} evidence is stale")

    windows = report["mutation_windows"]
    if not isinstance(windows, list) or len(windows) != len(MUTATION_WINDOWS):
        raise ValueError("Round 0005 fixture mutation matrix is incomplete")
    for expected_window, entry in zip(MUTATION_WINDOWS, windows):
        if (not isinstance(entry, dict) or
                set(entry) != {"window", "receipt", "signature"} or
                entry["window"] != expected_window or
                expected_input_signature(entry["receipt"]) != entry["signature"]):
            raise ValueError("Round 0005 fixture mutation receipt binding is invalid")
        validate_mutation_window_receipt(entry["receipt"],
                                         expected_window=expected_window)
    return report
