"""Build the immutable pre-gate queue for basemap-100m Round 0014."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       sha256_bytes)
from basemap.output_safety import (atomic_write_new_json, canonical_data_path,
                                   create_fresh_directory, refuse_existing)
from basemap.queue_admission import CACHE_KEYS, validate_queue_manifest
from basemap.release_preflight import (issue_release_preflight_receipt,
                                       release_reports_equivalent,
                                       verify_release)
from basemap.round0014_program import (
    ACCEPTED_CAPABILITY_SHA256, ACCEPTED_MANIFEST_FILE_SHA256,
    ACCEPTED_MANIFEST_RECEIPT_SHA256, GPU_HOURS_CAP, GPU_LEASE_PATH, NODES,
    PROGRAM, PROGRAM_INPUT_ROLES, ROUND_FILE, ROUND_ID, ROUND_SHA256,
    SEQUENCED_REVIEW_FILE, derived_node_policy, expected_argv, expected_outputs,
    program_policy,
)
from basemap.round0014_staging import validate_input_reference_manifest
from basemap.roundwatch_gate import canonical_roundwatch_binding
from basemap.run_controller import process_identity
from basemap.source_closure import round0014_source_closure_receipt


ROUND_ROOT = "/data/latent-basemap/runs/round-0014"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-sha", required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--integration-repo", required=True)
    parser.add_argument("--pushed-ref", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--round-root", default=ROUND_ROOT)
    parser.add_argument("--staging-manifest", required=True)
    parser.add_argument("--production-config", required=True)
    parser.add_argument("--canary-derivation", required=True)
    parser.add_argument("--transform-spec-template", required=True)
    parser.add_argument("--environment-manifest",
                        default="/data/latent-basemap/envs/run-env.json")
    parser.add_argument("--allowed-service", action="append", default=[],
                        metavar="PID:MARKER:VRAM_MB")
    parser.add_argument("--deadline-utc")
    parser.add_argument("--out", required=True)
    return parser


def _load_environment(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    required = ("venv_path", "freeze_file", "freeze_sha256", "identity_sha256",
                "gpu_uuid", "gpu_name")
    if not isinstance(value, dict) or any(
            not isinstance(value.get(key), str) or not value[key] for key in required):
        raise ValueError("Round 0014 environment manifest is incomplete")
    return value


def build_queue(args) -> dict:
    if args.implementation_commit != args.release_sha:
        raise ValueError("Round 0014 has exactly one implementation commit/release")
    round_root = canonical_data_path(args.round_root, label="Round 0014 root")
    if round_root != ROUND_ROOT or not os.path.isdir(round_root):
        raise ValueError(f"Round 0014 root must already exist at {ROUND_ROOT}")
    run_root = os.path.realpath(args.run_root)
    out = canonical_data_path(args.out, label="Round 0014 queue manifest")
    queue_root = os.path.dirname(out)
    if out != os.path.join(round_root, "queue", "queue.json"):
        raise ValueError("Round 0014 queue manifest must be round-root/queue/queue.json")
    refuse_existing(queue_root, label="Round 0014 queue root")
    environment = _load_environment(args.environment_manifest)
    python = os.path.realpath(os.path.join(environment["venv_path"], "bin", "python"))
    if not os.path.isfile(python):
        raise FileNotFoundError("Round 0014 sealed environment interpreter is missing")

    cache_root = os.path.join(queue_root, "cache")
    cache_environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    for key in CACHE_KEYS:
        cache_environment[key] = os.path.join(cache_root, key.lower())
    child_environment = {
        **cache_environment,
        "CUDA_VISIBLE_DEVICES": environment["gpu_uuid"],
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
    }

    preflight = verify_release(
        integration_repo=args.integration_repo, release_sha=args.release_sha,
        implementation_commits=[args.implementation_commit],
        pushed_ref=args.pushed_ref, run_checkout=run_root,
        environment_manifest=args.environment_manifest,
        cache_environment=cache_environment)
    if not preflight["passed"]:
        raise RuntimeError("Round 0014 release preflight rejected: " +
                           "; ".join(preflight["errors"]))
    with open(args.staging_manifest, encoding="utf-8") as handle:
        staged = validate_input_reference_manifest(json.load(handle), full_hash=True)

    allowed_processes = []
    for raw in args.allowed_service:
        try:
            pid, marker, budget = raw.split(":", 2)
            allowed_processes.append(process_identity(
                int(pid), marker=marker, gpu_memory_budget_mb=int(budget)))
        except Exception as exc:
            raise ValueError(
                f"invalid --allowed-service {raw!r}; expected PID:MARKER:VRAM_MB") from exc
    if len({item["pid"] for item in allowed_processes}) != len(allowed_processes):
        raise ValueError("Round 0014 allowed-service PID is duplicated")
    sources = round0014_source_closure_receipt(
        run_root, require_tracked=True)
    roundwatch = canonical_roundwatch_binding()

    create_fresh_directory(queue_root, label="Round 0014 queue root")
    artifacts = create_fresh_directory(
        os.path.join(queue_root, "artifacts"), label="Round 0014 artifact parent")
    receipts = create_fresh_directory(
        os.path.join(queue_root, "gate-receipts"), label="Round 0014 gate receipts")
    checkpoints = create_fresh_directory(
        os.path.join(queue_root, "controller-checkpoints"),
        label="Round 0014 controller checkpoints")
    create_fresh_directory(cache_root, label="Round 0014 cache parent")
    for key in CACHE_KEYS:
        create_fresh_directory(cache_environment[key], label=f"Round 0014 cache {key}")
    terminal = os.path.join(queue_root, "controller-terminal.json")
    gate_preparation = os.path.join(queue_root, "gate-preparation.json")
    release_receipt_path = os.path.join(queue_root, "release-preflight.json")
    for value in (terminal, gate_preparation, release_receipt_path):
        refuse_existing(value, label="Round 0014 queue control")
    issued = issue_release_preflight_receipt(
        release_receipt_path, integration_repo=args.integration_repo,
        release_sha=args.release_sha,
        implementation_commits=[args.implementation_commit],
        pushed_ref=args.pushed_ref, run_checkout=run_root,
        environment_manifest=args.environment_manifest,
        cache_environment=cache_environment)
    if not release_reports_equivalent(issued, preflight):
        raise RuntimeError("Round 0014 release changed while queue scaffolding was created")

    signature_cache: dict[str, dict] = {}

    def signature(path: str) -> dict:
        canonical = os.path.realpath(path)
        observed = expected_input_signature(canonical)
        prior = signature_cache.get(canonical)
        if prior is not None and prior != observed:
            raise RuntimeError(f"Round 0014 input changed during capture: {canonical}")
        signature_cache[canonical] = observed
        return observed

    for reference in staged["references"]:
        observed = signature(reference["canonical_path"])
        expected = {key: reference[key] for key in
                    ("canonical_path", "kind", "bytes", "sha256")}
        if observed != expected:
            raise RuntimeError(
                f"Round 0014 staged input changed before queue capture: {reference['role']}")
    role_paths = {
        "accepted_pack_manifest": staged["accepted_tuple"]["manifest_path"],
        "canary_derivation": os.path.realpath(args.canary_derivation),
        "environment_manifest": os.path.realpath(args.environment_manifest),
        "input_reference_manifest": os.path.realpath(args.staging_manifest),
        "production_config": os.path.realpath(args.production_config),
        "release_preflight_receipt": os.path.realpath(release_receipt_path),
        "round_file": ROUND_FILE,
        "sequenced_review0013": SEQUENCED_REVIEW_FILE,
        "transform_spec_template": os.path.realpath(args.transform_spec_template),
    }
    if tuple(sorted(role_paths)) != PROGRAM_INPUT_ROLES:
        raise AssertionError("Round 0014 program-input role implementation drift")
    extra_paths = [
        *role_paths.values(), environment["freeze_file"], python,
        roundwatch["cli"]["path"], roundwatch["interpreter"]["canonical_path"],
        roundwatch["git"]["canonical_path"],
        *[item["canonical_path"] for item in roundwatch["import_closure"]],
        *[item["signature"]["canonical_path"] for item in sources["members"]],
    ]
    for path in extra_paths:
        signature(path)
    registry = [signature_cache[path] for path in sorted(signature_cache)]
    program_inputs = [
        {"role": role, "signature": signature_cache[role_paths[role]]}
        for role in sorted(role_paths)
    ]
    with open(args.canary_derivation, encoding="utf-8") as handle:
        canary = json.load(handle)
    staging_body = {
        "schema": "round0014-input-staging-v1",
        "reference_manifest": signature_cache[os.path.realpath(args.staging_manifest)],
        "accepted_manifest_sha256": ACCEPTED_MANIFEST_FILE_SHA256,
        "manifest_receipt_sha256": ACCEPTED_MANIFEST_RECEIPT_SHA256,
        "capability_sha256": ACCEPTED_CAPABILITY_SHA256,
        "registered_file_count": staged["reference_count"],
        "payloads_copied": False,
    }
    input_staging = {
        **staging_body, "identity_sha256": sha256_bytes(canonical_json(staging_body))}
    deadline = args.deadline_utc or (
        datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(timespec="seconds")
    manifest = {
        "schema_version": 1, "program": PROGRAM, "round_id": ROUND_ID,
        "round_sha256": ROUND_SHA256, "release_sha": args.release_sha,
        "execution_authority": "owner-gpu", "required_reviews": ["0013"],
        "environment_freeze_sha": environment["freeze_sha256"],
        "environment_identity_sha": environment["identity_sha256"],
        "gpu_hours_cap": GPU_HOURS_CAP, "queue_class": "research",
        "training_performed": True, "deadline_utc": deadline,
        "environment_manifest": os.path.realpath(args.environment_manifest),
        "cache_environment": cache_environment,
        "child_environment": child_environment,
        "gate_receipts_dir": receipts,
        "controller_checkpoints_dir": checkpoints,
        "controller_terminal_summary": terminal,
        "repo_root": run_root, "lease_path": GPU_LEASE_PATH,
        "allowed_processes": allowed_processes,
        "jobs": [], "input_staging": input_staging,
        "fixture_identity": {
            "schema": canary["schema"],
            "canonical_path": os.path.realpath(args.canary_derivation),
            "sha256": signature_cache[os.path.realpath(args.canary_derivation)]["sha256"],
            "identity_sha256": canary["identity_sha256"],
        },
        "program_policy": program_policy(),
        "program_inputs": program_inputs,
        "global_input_registry": registry,
        "source_closure": sources,
        "roundwatch_binding": roundwatch,
        "release_preflight_identity": issued["identity_sha256"],
        "gate_preparation_receipt": gate_preparation,
    }
    for spec in NODES:
        controls = {
            "done_marker": os.path.join(artifacts, f"{spec.node_id}.done.json"),
            "log": os.path.join(artifacts, f"{spec.node_id}.log"),
            "manifest": os.path.join(artifacts, f"{spec.node_id}.controller.json"),
        }
        outputs = expected_outputs(spec, queue_root=queue_root)
        for value in (*outputs, *controls.values()):
            refuse_existing(value, label=f"Round 0014 {spec.node_id} output/control")
        manifest["jobs"].append({
            "id": spec.node_id,
            "argv": expected_argv(spec, manifest=manifest, manifest_path=out),
            "inputs": [item["canonical_path"] for item in registry],
            "expected_inputs": registry,
            "outputs": outputs, **controls, "cwd": run_root,
            "predicted_wall_s": spec.predicted_wall_s,
            "p90_wall_s": spec.p90_wall_s,
            "deps": [] if spec.dependency is None else [spec.dependency],
            "node_policy": derived_node_policy(spec),
        })
    validate_queue_manifest(manifest, out)
    atomic_write_new_json(out, manifest, immutable=True)
    return manifest


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_queue(args)
    print(json.dumps({
        "queue_manifest": os.path.realpath(args.out),
        "round_id": manifest["round_id"],
        "release_sha": manifest["release_sha"],
        "jobs": [item["id"] for item in manifest["jobs"]],
        "gpu_hours_cap": manifest["gpu_hours_cap"],
        "payloads_copied": manifest["input_staging"]["payloads_copied"],
        "registered_pre_gate_inputs": len(manifest["global_input_registry"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
