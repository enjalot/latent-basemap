"""Run the clean-release six-Popen fixture and four real mutation windows."""
from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       git_checkout_state, sha256_bytes)
from basemap.gate_preparation import prepare_gate
from basemap.output_safety import (atomic_write_new_json, create_fresh_directory,
                                   refuse_existing)
from basemap.queue_admission import (MUTATION_WINDOWS, QueueAdmission,
                                     validate_mutation_window_receipt)
from basemap.release_preflight import issue_release_preflight_receipt
from basemap.round0005_fixture import (FIXTURE_QUEUE_SCHEMA, FIXTURE_SCHEMA,
                                       SIX_NODE_IDS, fixture_source_closure,
                                       validate_fixture_queue,
                                       validate_round0005_fixture)
from basemap.roundwatch_gate import (canonical_roundwatch_binding,
                                     _new_fixture_roundwatch_authority)
from basemap.run_controller import _run_admitted_queue_fixture_only
from experiments.prepare_round0005_queue import (
    _FIXTURE_BUILDER_CAPABILITY, _publish_fixture_queue,
)


class EnvironmentValidationError(RuntimeError):
    """Fail-closed validation error for an incomplete fixture environment."""


def _cache_environment(root: str) -> dict:
    cache_root = create_fresh_directory(
        os.path.join(root, "cache"), label="fixture cache root")
    result = {"PYTHONDONTWRITEBYTECODE": "1"}
    for key in ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR",
                "PYTHONPYCACHEPREFIX", "NUMBA_CACHE_DIR", "MPLCONFIGDIR"):
        result[key] = create_fresh_directory(
            os.path.join(cache_root, key.lower()), label=f"fixture cache {key}")
    return result


def _rewrite_fixture_input(path: str, *, version: int) -> None:
    os.chmod(path, 0o644)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"schema": "round0005_fixture_dependency.v1", "version": version}, handle)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _environment_report(environment_path: str) -> tuple[dict, str]:
    with open(environment_path, encoding="utf-8") as handle:
        environment = json.load(handle)
    required = ("freeze_file", "freeze_sha256", "identity_sha256", "venv_path")
    if (not isinstance(environment, dict) or
            any(not isinstance(environment.get(key), str) or not environment[key]
                for key in required)):
        raise EnvironmentValidationError(
            "fixture environment manifest is incomplete; required string fields: "
            + ", ".join(required))
    freeze_path = os.path.realpath(environment["freeze_file"])
    python = os.path.realpath(
        os.path.join(os.path.realpath(environment["venv_path"]), "bin", "python"))
    with open(freeze_path, encoding="utf-8") as handle:
        freeze_sha = sha256_bytes(
            "".join(line + "\n" for line in sorted(
                line.strip() for line in handle if line.strip())).encode("utf-8"))
    report = {
        "manifest": expected_input_signature(environment_path),
        "freeze": expected_input_signature(freeze_path),
        "freeze_sha256": freeze_sha,
        "identity_sha256": environment["identity_sha256"],
        "venv_path": os.path.realpath(environment["venv_path"]),
    }
    return report, python


def _captured_fixture_manifest(*, root: str, repo_root: str, release_sha: str,
                               round_sha256: str, environment_manifest: str,
                               integration_repo: str, implementation_commits: list[str],
                               pushed_ref: str) -> tuple[dict, str, str]:
    for name in ("node-outputs", "node-controls", "gate-receipts", "checkpoints"):
        create_fresh_directory(os.path.join(root, name), label=f"fixture {name}")
    cache = _cache_environment(root)
    environment, resolved_python = _environment_report(environment_manifest)
    release_path = os.path.join(root, "release-preflight.json")
    release = issue_release_preflight_receipt(
        release_path, integration_repo=integration_repo, release_sha=release_sha,
        implementation_commits=implementation_commits, pushed_ref=pushed_ref,
        run_checkout=repo_root, environment_manifest=environment_manifest,
        cache_environment=cache)
    python = release["python_invocation_path"]
    source_closure = fixture_source_closure(repo_root)
    roundwatch = canonical_roundwatch_binding()
    fixture_input = os.path.join(root, "fixture-input.json")
    atomic_write_new_json(
        fixture_input,
        {"schema": "round0005_fixture_dependency.v1", "version": 1}, immutable=True)
    signature_paths = {
        fixture_input, release_path, environment["manifest"]["canonical_path"],
        environment["freeze"]["canonical_path"], resolved_python,
        roundwatch["cli"]["path"],
        roundwatch["interpreter"]["canonical_path"],
        roundwatch["git"]["canonical_path"],
        *[entry["canonical_path"] for entry in roundwatch["import_closure"]],
        *[entry["signature"]["canonical_path"] for entry in source_closure["members"]],
    }
    registry = [expected_input_signature(path) for path in sorted(signature_paths)]
    registry_paths = [value["canonical_path"] for value in registry]
    helper = os.path.join(repo_root, "experiments", "run_round0005_fixture_node.py")
    jobs = []
    for position, node in enumerate(SIX_NODE_IDS):
        output = os.path.join(root, "node-outputs", f"{position:02d}-{node}.json")
        controls = os.path.join(root, "node-controls")
        policy = {
            "schema": "round0005_fixture_node_policy.v1", "node_id": node,
            "canonical_script": "experiments/run_round0005_fixture_node.py",
            "training_performed": False, "gpu_required": False,
            "cuda_device_count": 0, "scientific_rows": 1,
            "required_free_gb": 0.0, "gpu_memory_cap_mb": 0,
            "scale_certificate_required": False,
        }
        jobs.append({
            "id": node,
            "argv": [python, helper, "--node", node, "--out", output,
                     "--repo-root", repo_root],
            "inputs": registry_paths, "expected_inputs": registry,
            "outputs": [output],
            "done_marker": os.path.join(controls, f"{node}.done.json"),
            "log": os.path.join(controls, f"{node}.log"),
            "manifest": os.path.join(controls, f"{node}.controller.json"),
            "cwd": repo_root, "predicted_wall_s": 10.0, "p90_wall_s": 30.0,
            "deps": [] if position == 0 else [SIX_NODE_IDS[position - 1]],
            "node_policy": policy,
        })
    child_environment = {
        **cache, "CUDA_VISIBLE_DEVICES": "",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "TOKENIZERS_PARALLELISM": "false",
    }
    manifest = {
        "schema": FIXTURE_QUEUE_SCHEMA, "program": "basemap-100m", "round_id": "0005",
        "round_sha256": round_sha256, "release_sha": release_sha,
        "environment_freeze_sha": environment["freeze_sha256"],
        "environment_identity_sha": environment["identity_sha256"],
        "environment_manifest": os.path.realpath(environment_manifest),
        "gpu_hours_cap": 0.75,
        "deadline_utc": (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat(),
        "repo_root": repo_root, "queue_root": root,
        "cache_environment": cache, "child_environment": child_environment,
        "gate_receipts_dir": os.path.join(root, "gate-receipts"),
        "controller_checkpoints_dir": os.path.join(root, "checkpoints"),
        "controller_terminal_summary": os.path.join(root, "controller-terminal.json"),
        "gate_preparation_receipt": os.path.join(root, "gate-preparation.json"),
        "lease_path": os.path.join(root, "fixture.gpu-lease"),
        "allowed_processes": [], "jobs": jobs, "global_input_registry": registry,
        "fixture_input": expected_input_signature(fixture_input),
        "source_closure": source_closure, "roundwatch_binding": roundwatch,
        "release_preflight_identity": release["identity_sha256"],
        "release_preflight_receipt": release_path,
    }
    return manifest, os.path.join(root, "queue.json"), fixture_input


def _publish_case(*, parent: str, label: str, args,
                  publication_hook=None) -> tuple[dict, str, str]:
    root = create_fresh_directory(
        os.path.join(parent, label), label=f"fixture case {label}")
    manifest, path, fixture_input = _captured_fixture_manifest(
        root=root, repo_root=args.repo_root, release_sha=args.release_sha,
        round_sha256=args.round_sha256, environment_manifest=args.environment_manifest,
        integration_repo=args.integration_repo,
        implementation_commits=args.implementation_commit,
        pushed_ref=args.pushed_ref)
    _publish_fixture_queue(
        manifest=manifest, out=path, validator=validate_fixture_queue,
        phase_hook=publication_hook, capability=_FIXTURE_BUILDER_CAPABILITY)
    return manifest, path, fixture_input


def _prepared_admission(*, parent: str, label: str, args,
                        gate_hook=None):
    manifest, path, fixture_input = _publish_case(
        parent=parent, label=label, args=args)
    authority = _new_fixture_roundwatch_authority()
    prepare_gate(
        path, _fixture_authority=authority, _fixture_phase_hook=gate_hook,
        _fixture_validator=validate_fixture_queue)
    admission = QueueAdmission._for_fixture(
        path, args.repo_root, validator=validate_fixture_queue,
        gate_authority=authority)
    return manifest, path, fixture_input, admission


def _only_receipt(root: str, phase: str) -> str:
    directory = os.path.join(root, "gate-receipts")
    matches = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        try:
            receipt = validate_mutation_window_receipt(path, expected_window=phase)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if receipt["phase"] == phase:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"fixture expected one automatic {phase} receipt, got {matches}")
    return matches[0]


def _mutation_integrations(parent: str, args) -> list[dict]:
    results = []

    # Window 1: the shared builder rejects after capture and before publication.
    label = "mutation-0-capture"
    root = os.path.join(parent, label)
    try:
        _publish_case(
            parent=parent, label=label, args=args,
            publication_hook=lambda _phase, manifest: _rewrite_fixture_input(
                manifest["fixture_input"]["canonical_path"], version=2))
        raise AssertionError("capture mutation unexpectedly published a queue")
    except RuntimeError as exc:
        if "before publication" not in str(exc):
            raise
    receipt = _only_receipt(root, MUTATION_WINDOWS[0])
    results.append({"window": MUTATION_WINDOWS[0], "receipt": receipt,
                    "signature": expected_input_signature(receipt)})

    # Window 2: real gate preparation reopens and rejects the published queue.
    label = "mutation-1-publication"
    manifest, path, fixture_input = _publish_case(parent=parent, label=label, args=args)
    authority = _new_fixture_roundwatch_authority()
    try:
        prepare_gate(
            path, _fixture_authority=authority,
            _fixture_phase_hook=lambda _phase, _manifest: _rewrite_fixture_input(
                fixture_input, version=2),
            _fixture_validator=validate_fixture_queue)
        raise AssertionError("gate preparation mutation unexpectedly passed")
    except ValueError as exc:
        if "fixture" not in str(exc):
            raise
    receipt = _only_receipt(os.path.dirname(path), MUTATION_WINDOWS[1])
    results.append({"window": MUTATION_WINDOWS[1], "receipt": receipt,
                    "signature": expected_input_signature(receipt)})

    # Window 3: mutation occurs after gate sidecar publication; admission rejects.
    label = "mutation-2-pre-admission"
    manifest, path, fixture_input = _publish_case(parent=parent, label=label, args=args)
    authority = _new_fixture_roundwatch_authority()
    prepare_gate(
        path, _fixture_authority=authority,
        _fixture_phase_hook=lambda phase, _manifest: (
            _rewrite_fixture_input(fixture_input, version=2)
            if phase == MUTATION_WINDOWS[2] else None),
        _fixture_validator=validate_fixture_queue)
    try:
        QueueAdmission._for_fixture(
            path, args.repo_root, validator=validate_fixture_queue,
            gate_authority=authority)
        raise AssertionError("pre-admission mutation unexpectedly passed")
    except ValueError as exc:
        if "fixture" not in str(exc):
            raise
    receipt = _only_receipt(os.path.dirname(path), MUTATION_WINDOWS[2])
    results.append({"window": MUTATION_WINDOWS[2], "receipt": receipt,
                    "signature": expected_input_signature(receipt)})

    # Window 4: controller obtains a real bound gate response; its launch-edge
    # hook mutates the global input, the repeated comparison rejects before Popen.
    label = "mutation-3-launch-edge"
    manifest, path, fixture_input, admission = _prepared_admission(
        parent=parent, label=label, args=args)
    controller = _run_admitted_queue_fixture_only(
        admission=admission,
        launch_edge_hook=lambda _job: _rewrite_fixture_input(fixture_input, version=2),
        telemetry_interval_s=0.05)
    if controller["terminal_verdict"] != "failed" or any(
            entry.get("child_pid") for entry in controller["jobs"]):
        raise RuntimeError("launch-edge mutation did not fail before child PID creation")
    receipt = controller["jobs"][0].get("integrity_receipt")
    validate_mutation_window_receipt(receipt, expected_window=MUTATION_WINDOWS[3])
    results.append({"window": MUTATION_WINDOWS[3], "receipt": receipt,
                    "signature": expected_input_signature(receipt)})
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--implementation-commit", action="append", required=True)
    parser.add_argument("--integration-repo", required=True)
    parser.add_argument("--pushed-ref", required=True)
    parser.add_argument("--round-sha256", required=True)
    parser.add_argument("--environment-manifest", required=True)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    args.out = os.path.abspath(args.out)
    args.repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args.environment_manifest = os.path.realpath(args.environment_manifest)
    args.integration_repo = os.path.realpath(args.integration_repo)
    if not args.out.startswith("/data/"):
        raise ValueError("fixture report must live under /data")
    refuse_existing(args.out, label="Round 0005 fixture report")
    checkout = git_checkout_state(args.repo_root)
    if (checkout["head"] != args.release_sha or checkout["clean"] is not True or
            checkout["detached"] is not True):
        raise RuntimeError("fixture requires the exact clean detached queue release")
    environment, python = _environment_report(args.environment_manifest)
    if os.path.realpath(args.python) != python or not os.path.isfile(python):
        raise RuntimeError("fixture Python is not the sealed environment executable")

    parent = create_fresh_directory(
        f"{args.out}.runtime-{uuid.uuid4().hex}",
        label="Round 0005 fixture integration root")
    mutation_windows = _mutation_integrations(parent, args)
    manifest, queue_path, _fixture_input, admission = _prepared_admission(
        parent=parent, label="six-node-success", args=args)
    controller = _run_admitted_queue_fixture_only(
        admission=admission, telemetry_interval_s=0.05)
    if controller.get("terminal_verdict") != "passed":
        raise RuntimeError(f"Round 0005 six-node fixture failed: {controller}")
    child_pids = [entry.get("child_pid") for entry in controller["jobs"]]
    if len(child_pids) != 6 or None in child_pids or len(set(child_pids)) != 6:
        raise RuntimeError("fixture controller did not create exactly six unique Popen children")
    children = []
    for node, job, child_pid in zip(SIX_NODE_IDS, admission.runtime_jobs(), child_pids):
        with open(job.outputs[0], encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("pid") != child_pid or payload.get("nested_child_processes") != 0:
            raise RuntimeError(f"fixture node {node} process identity is not exact")
        children.append({
            "id": node, "pid": child_pid, "output": job.outputs[0],
            "output_signature": expected_input_signature(job.outputs[0]),
            "cuda_visible_devices": payload["cuda_visible_devices"],
            "cuda_probe_performed": payload["cuda_probe_performed"],
        })
    report = {
        "schema": FIXTURE_SCHEMA, "passed": True, "release_sha": args.release_sha,
        "round_id": "0005", "round_sha256": args.round_sha256, "checkout": checkout,
        "environment": environment,
        "python": {"canonical_path": python, "signature": expected_input_signature(python)},
        "cuda_hidden": {"CUDA_VISIBLE_DEVICES": "", "all_children_proved_hidden": True,
                        "cuda_api_calls": 0},
        "source_closure": manifest["source_closure"],
        "six_node_queue": {
            "node_ids": list(SIX_NODE_IDS), "queue_manifest": queue_path,
            "queue_signature": expected_input_signature(queue_path),
            "terminal_summary": manifest["controller_terminal_summary"],
            "terminal_signature": expected_input_signature(
                manifest["controller_terminal_summary"]),
            "popen_count": 6, "child_pids": child_pids, "children": children,
        },
        "mutation_windows": mutation_windows,
    }
    body = canonical_json(report)
    report["identity_sha256"] = sha256_bytes(body)
    atomic_write_new_json(args.out, report, immutable=True)
    validate_round0005_fixture(
        args.out, repo_root=args.repo_root, release_sha=args.release_sha,
        environment_manifest=args.environment_manifest)
    print(json.dumps({"passed": True, "fixture": args.out,
                      "identity_sha256": report["identity_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
