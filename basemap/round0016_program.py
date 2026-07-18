"""Exact CPU-visible contract for the one Round 0016 production queue."""
from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes)
from .release_preflight import validate_release_preflight_receipt
from .round0014_program import (
    ACCEPTED_CAPABILITY_SHA256, ACCEPTED_MANIFEST,
    ACCEPTED_MANIFEST_FILE_SHA256, ACCEPTED_MANIFEST_RECEIPT_SHA256,
    CENTROIDS_K1024_PATH, CENTROIDS_K1024_SHA256,
    CENTROIDS_K256_PATH, CENTROIDS_K256_SHA256, DIMENSION, GRAPH_PATH,
    GRAPH_SHA256, INDEX_PATH, INDEX_SHA256, QUERY_PROVENANCE_PATH,
    QUERY_PROVENANCE_SHA256, QUERIES_PATH, QUERIES_SHA256,
    Round0014MaterializedArray, TOTAL_ROWS,
    accepted_reference_records as _round0014_reference_records,
    raw_source_map,
)
from .round0014_program import TRAIN_CONFIG as _ROUND0014_TRAIN_CONFIG
from .round0016_service import (JOB_CAP_MIB, POLICY, REQUIRED_FREE_GIB,
                                load_exclusive_decision)
from .source_closure import validate_round0016_source_closure_receipt


ROUND_ID = "0016"
PROGRAM = "basemap-100m"
BASE_COMMIT = "836687a81aa8f94798098d5edfdd72e264e29d77"
BASE_TREE = "3b0366e5032408c93a0eeb2cdcae480d8eaebd2d"
ROUND_FILE = "/home/enjalot/code/latent-labs/basemap-100m/round-0016-2026-07-18.md"
ROUND_SHA256 = "8b359bff4db6bf9eeeb91c5d9163b5d901deb4c388ee4244c304605a82ef4594"
SEQUENCED_REVIEW_FILE = (
    "/home/enjalot/code/latent-labs/basemap-100m/review-0013-2026-07-18-01.md")
SEQUENCED_REVIEW_SHA256 = (
    "362c9fa55d9e15db55f7074882200e68c637c3d8b6f24b24854f9048997ffc17")
GPU_HOURS_CAP = 5.5
GPU_LEASE_PATH = "/data/latent-basemap/.gpu_lease"
HASH64 = re.compile(r"[0-9a-f]{64}")
FULL_SHA = re.compile(r"[0-9a-f]{40}")

PROGRAM_INPUT_ROLES = (
    "accepted_pack_manifest",
    "canary_derivation",
    "environment_manifest",
    "exclusive_gpu_decision",
    "input_reference_manifest",
    "production_config",
    "release_preflight_receipt",
    "round_file",
    "sequenced_review0013",
    "transform_spec_template",
)
REQUIRED_PROGRAM_INPUT_ROLES = len(PROGRAM_INPUT_ROLES)
JOB_FIELDS = {
    "id", "argv", "inputs", "expected_inputs", "outputs", "done_marker",
    "log", "manifest", "cwd", "predicted_wall_s", "p90_wall_s", "deps",
    "node_policy",
}


TRAIN_CONFIG: dict[str, Any] = copy.deepcopy(_ROUND0014_TRAIN_CONFIG)
TRAIN_CONFIG["schema"] = "round0016-production-config-v1"
TRAIN_CONFIG_SHA256 = sha256_bytes(canonical_json(TRAIN_CONFIG))
Round0016MaterializedArray = Round0014MaterializedArray


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    dependency: str | None
    predicted_wall_s: float
    p90_wall_s: float
    training_performed: bool
    output_name: str


NODES = (
    NodeSpec("no_training_seal_canary", None, 180.0, 300.0, False, "canary"),
    NodeSpec("train_seed42_30m", "no_training_seal_canary", 8460.0, 10800.0,
             True, "train"),
    NodeSpec("transform_30m", "train_seed42_30m", 900.0, 1800.0, False,
             "coordinates"),
    NodeSpec("high_d_reference", "transform_30m", 600.0, 900.0, False,
             "high-d-reference"),
    NodeSpec("registered_panel", "high_d_reference", 1800.0, 1800.0, False,
             "panel"),
    NodeSpec("semantic_renders", "registered_panel", 300.0, 300.0, False,
             "semantic-renders"),
)
NODE_BY_ID = {item.node_id: item for item in NODES}


def round0016_release_chain(release_sha: str) -> dict[str, Any]:
    if (not isinstance(release_sha, str) or
            not FULL_SHA.fullmatch(release_sha) or release_sha == BASE_COMMIT):
        raise ValueError("Round 0016 release SHA is invalid")
    return {
        "scientific_base": BASE_COMMIT,
        "scientific_base_tree": BASE_TREE,
        "execution_release": release_sha,
        "implementation_commits": [release_sha],
        "ancestry": [BASE_COMMIT, release_sha],
        "commits_after_base": 1,
    }


def accepted_reference_records(*, full_hash: bool) -> list[dict[str, Any]]:
    return _round0014_reference_records(full_hash=full_hash)


def derived_node_policy(spec: NodeSpec) -> dict[str, Any]:
    body = {
        "schema": "round0016-derived-node-policy-v1",
        "node_id": spec.node_id,
        "canonical_script": "experiments/run_round0016_node.py",
        "training_performed": spec.training_performed,
        "gpu_required": True,
        "cuda_device_count": 1,
        "scientific_rows": TOTAL_ROWS,
        "required_free_gb": REQUIRED_FREE_GIB,
        "gpu_memory_cap_mb": JOB_CAP_MIB,
        "scale_certificate_required": False,
        "canary_predecessor": (None if spec.dependency is None else
                               NODES[0].node_id),
        "one_use": True,
        "retry_count": 0,
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _role_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    entries = manifest.get("program_inputs")
    if not isinstance(entries, list):
        raise ValueError("Round 0016 program inputs must be an ordered list")
    roles: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"role", "signature"}:
            raise ValueError("Round 0016 program input fields changed")
        if entry["role"] in roles:
            raise ValueError("Round 0016 program input role is duplicated")
        roles[entry["role"]] = entry["signature"]
    if tuple(sorted(roles)) != PROGRAM_INPUT_ROLES:
        raise ValueError("Round 0016 program input roles changed")
    return roles


def _validate_bound_release(manifest: dict[str, Any]) -> dict[str, Any]:
    signature = _role_map(manifest)["release_preflight_receipt"]
    release = validate_release_preflight_receipt(
        signature["canonical_path"],
        expected_identity_sha256=manifest["release_preflight_identity"],
        expected_signature=signature)
    expected = round0016_release_chain(release["release_sha"])
    if release["implementation_commits"] != expected["implementation_commits"]:
        raise ValueError("Round 0016 release receipt changed")
    return release


def validate_global_input_registry(
        registry: Any, *, required_paths: set[str]) -> list[str]:
    paths = [item.get("canonical_path") for item in registry] \
        if isinstance(registry, list) else []
    if not paths or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Round 0016 global input registry must be nonempty, sorted, and unique")
    if not required_paths.issubset(set(paths)):
        raise ValueError("Round 0016 global registry omits a required source/input")
    return paths


def expected_argv(spec: NodeSpec, *, manifest: dict[str, Any],
                  manifest_path: str) -> list[str]:
    release = _validate_bound_release(manifest)
    return [
        release["python_invocation_path"],
        os.path.join(manifest["repo_root"], "experiments/run_round0016_node.py"),
        "--queue-manifest", os.path.realpath(manifest_path),
        "--node", spec.node_id,
    ]


def expected_outputs(spec: NodeSpec, *, queue_root: str) -> list[str]:
    return [os.path.join(queue_root, "artifacts", spec.output_name)]


def derive_program_context(manifest: dict[str, Any], *,
                           repo_root: str) -> dict[str, Any]:
    roles = _role_map(manifest)
    paths: dict[str, str] = {}
    for role in PROGRAM_INPUT_ROLES:
        signature = roles[role]
        path = signature.get("canonical_path") if isinstance(signature, dict) else None
        if (not isinstance(path, str) or not os.path.isabs(path) or
                expected_input_signature(path) != signature):
            raise ValueError(f"Round 0016 program input changed: {role}")
        paths[role] = path
    fixed = {
        "round_file": (ROUND_FILE, ROUND_SHA256),
        "sequenced_review0013": (SEQUENCED_REVIEW_FILE,
                                 SEQUENCED_REVIEW_SHA256),
        "accepted_pack_manifest": (ACCEPTED_MANIFEST,
                                   ACCEPTED_MANIFEST_FILE_SHA256),
    }
    for role, (path, digest) in fixed.items():
        if paths[role] != path or roles[role]["sha256"] != digest:
            raise ValueError(f"Round 0016 fixed {role} binding changed")
    release = _validate_bound_release(manifest)
    if (release["release_sha"] != manifest["release_sha"] or
            release["run_checkout_path"] != os.path.realpath(repo_root)):
        raise ValueError("Round 0016 release receipt differs from queue checkout")
    validate_round0016_source_closure_receipt(
        manifest["source_closure"], repo_root=repo_root)
    with open(paths["production_config"], encoding="utf-8") as handle:
        config_record = json.load(handle)
    if config_record != {
            "schema": "round0016-production-config-receipt-v1",
            "config": TRAIN_CONFIG, "config_sha256": TRAIN_CONFIG_SHA256}:
        raise ValueError("Round 0016 production config changed")
    with open(paths["input_reference_manifest"], encoding="utf-8") as handle:
        reference_manifest = json.load(handle)
    from .round0016_staging import validate_input_reference_manifest
    validate_input_reference_manifest(reference_manifest, full_hash=False)
    from .round0014_transform import validate_transform_template
    transform_template = validate_transform_template(
        paths["transform_spec_template"], release_root=repo_root,
        release_sha=manifest["release_sha"])
    exclusive = load_exclusive_decision(
        paths["exclusive_gpu_decision"],
        environment_manifest=paths["environment_manifest"],
        require_current=False)
    if exclusive["allowed_processes"] != manifest.get("allowed_processes") != []:
        raise ValueError("Round 0016 exclusive decision differs from queue")
    return {
        "paths": paths, "release": release,
        "reference_manifest": reference_manifest,
        "transform_template": transform_template,
        "exclusive_gpu_decision": exclusive,
    }


def validate_exact_program(manifest: dict[str, Any], *, manifest_path: str,
                           repo_root: str) -> dict[str, Any]:
    context = derive_program_context(manifest, repo_root=repo_root)
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or [item.get("id") for item in jobs] != [
            item.node_id for item in NODES]:
        raise ValueError("Round 0016 queue must contain its exact six nodes in order")
    required = {
        entry["signature"]["canonical_path"] for entry in manifest["program_inputs"]
    } | {
        entry["signature"]["canonical_path"]
        for entry in manifest["source_closure"]["members"]
    } | {
        item["canonical_path"] for item in context["reference_manifest"]["references"]
    }
    registry = manifest.get("global_input_registry")
    paths = validate_global_input_registry(registry, required_paths=required)
    release_evidence = context["reference_manifest"]["release_evidence"]
    expected_chain = round0016_release_chain(manifest["release_sha"])
    if any(release_evidence.get(key) != value
           for key, value in expected_chain.items()):
        raise ValueError("Round 0016 exact one-commit ancestry changed")
    queue_root = os.path.dirname(os.path.realpath(manifest_path))
    for position, (job, spec) in enumerate(zip(jobs, NODES)):
        dependency = [] if spec.dependency is None else [spec.dependency]
        controls = {
            "done_marker": os.path.join(
                queue_root, "artifacts", f"{spec.node_id}.done.json"),
            "log": os.path.join(queue_root, "artifacts", f"{spec.node_id}.log"),
            "manifest": os.path.join(
                queue_root, "artifacts", f"{spec.node_id}.controller.json"),
        }
        if (set(job) != JOB_FIELDS or job["deps"] != dependency or
                (position and spec.dependency != jobs[position - 1]["id"]) or
                job["argv"] != expected_argv(
                    spec, manifest=manifest, manifest_path=manifest_path) or
                job["inputs"] != paths or job["expected_inputs"] != registry or
                job["outputs"] != expected_outputs(spec, queue_root=queue_root) or
                any(job[key] != value for key, value in controls.items()) or
                job["cwd"] != repo_root or
                float(job["predicted_wall_s"]) != spec.predicted_wall_s or
                float(job["p90_wall_s"]) != spec.p90_wall_s or
                job["node_policy"] != derived_node_policy(spec)):
            raise ValueError(f"Round 0016 exact node wiring changed: {spec.node_id}")
    if sum(item.p90_wall_s * 1.15 for item in NODES) > GPU_HOURS_CAP * 3600:
        raise AssertionError("Round 0016 registered p90+15% exceeds 5.5 hours")
    if manifest.get("lease_release_receipt") != os.path.join(
            queue_root, "lease-release.json"):
        raise ValueError("Round 0016 lease-release destination changed")
    expected_policy = program_policy(
        registered_global_inputs=len(paths),
        source_closure_members=len(manifest["source_closure"]["members"]))
    if manifest.get("program_policy") != expected_policy:
        raise ValueError("Round 0016 program policy changed")
    return context


def program_policy(*, registered_global_inputs: int,
                   source_closure_members: int) -> dict[str, Any]:
    body = {
        "schema": "round0016-program-policy-v1",
        "node_ids": [item.node_id for item in NODES],
        "training_nodes": [item.node_id for item in NODES
                           if item.training_performed],
        "one_no_training_canary": True,
        "one_seed42_treatment": True,
        "exact_ordered_fail_stop_dag": True,
        "gpu_policy": POLICY,
        "allowed_processes": [],
        "service_marker": None,
        "service_reservation_mib": 0,
        "job_gpu_memory_cap_mib": JOB_CAP_MIB,
        "required_free_gib": REQUIRED_FREE_GIB,
        "terminal_lease_release_required": True,
        "registered_global_inputs": int(registered_global_inputs),
        "source_closure_members": int(source_closure_members),
        "program_input_roles": len(PROGRAM_INPUT_ROLES),
        "retry_count": 0,
        "registered_p90_plus_margin_seconds": sum(
            item.p90_wall_s * 1.15 for item in NODES),
    }
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
