"""Reference-only staging for the immutable Round 0015 production inputs."""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes)
from .output_safety import (atomic_write_new_json, create_fresh_directory,
                            refuse_existing)
from .round0015_program import (
    ACCEPTED_CAPABILITY_SHA256, ACCEPTED_MANIFEST,
    ACCEPTED_MANIFEST_FILE_SHA256, ACCEPTED_MANIFEST_RECEIPT_SHA256,
    FIRST_IMPLEMENTATION_COMMIT, FIRST_RELEASE_TREE, ISSUED_BASE, ROUND_ID,
    SEQUENCED_REVIEW_FILE, SEQUENCED_REVIEW_SHA256, TRAIN_CONFIG,
    TRAIN_CONFIG_SHA256, accepted_reference_records, round0015_release_chain,
)


REFERENCE_SCHEMA = "round0015-input-reference-manifest-v1"
ROUND_ROOT = "/data/latent-basemap/runs/round-0015"


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _git(repo: str, *args: str) -> str:
    process = subprocess.run(
        ["/usr/bin/git", "-C", repo, *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
             "PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1",
             "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_OPTIONAL_LOCKS": "0"})
    if process.returncode:
        raise RuntimeError(f"Git {' '.join(args)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def verify_release_chain(release_root: str, release_sha: str) -> dict[str, Any]:
    root = os.path.realpath(release_root)
    if root != release_root or _git(root, "rev-parse", "HEAD") != release_sha:
        raise RuntimeError("Round 0015 staging release checkout/commit changed")
    expected = round0015_release_chain(release_sha)
    if (_git(root, "rev-parse", "HEAD^") != FIRST_IMPLEMENTATION_COMMIT or
            _git(root, "rev-parse", "HEAD^^") != ISSUED_BASE or
            _git(root, "rev-parse", f"{FIRST_IMPLEMENTATION_COMMIT}^") !=
            ISSUED_BASE or
            _git(root, "rev-parse", f"{FIRST_IMPLEMENTATION_COMMIT}^{{tree}}") !=
            FIRST_RELEASE_TREE or
            _git(root, "rev-list", "--ancestry-path", "--reverse",
                 f"{ISSUED_BASE}..{release_sha}").splitlines() !=
            expected["implementation_commits"]):
        raise RuntimeError("Round 0015 release is not the exact reset-authorized chain")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Round 0015 staging release checkout is dirty")
    if subprocess.run(
            ["/usr/bin/git", "-C", root, "symbolic-ref", "-q", "HEAD"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("Round 0015 staging requires the detached run checkout")
    body = {
        "schema": "round0015-two-commit-release-evidence-v1",
        "release_root": root,
        **expected,
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "parent": FIRST_IMPLEMENTATION_COMMIT,
        "grandparent": ISSUED_BASE,
        "detached": True,
        "clean": True,
    }
    return _seal(body)


def validate_release_evidence(value: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "release_root", "reviewed_base",
        "first_implementation_commit", "first_release_tree",
        "corrected_release", "implementation_commits", "ancestry",
        "commits_after_issued_base", "tree", "parent", "grandparent",
        "detached", "clean", "identity_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Round 0015 release evidence fields changed")
    body = {key: value[key] for key in value if key != "identity_sha256"}
    expected = round0015_release_chain(value.get("corrected_release"))
    if (value["identity_sha256"] != sha256_bytes(canonical_json(body)) or
            value["schema"] != "round0015-two-commit-release-evidence-v1" or
            any(value.get(key) != expected_value
                for key, expected_value in expected.items()) or
            value["parent"] != FIRST_IMPLEMENTATION_COMMIT or
            value["grandparent"] != ISSUED_BASE or
            value["detached"] is not True or value["clean"] is not True or
            not isinstance(value["release_root"], str) or
            not os.path.isabs(value["release_root"])):
        raise ValueError("Round 0015 exact release evidence changed")
    return value


def _expected_references(*, full_hash: bool) -> list[dict[str, Any]]:
    references = accepted_reference_records(full_hash=full_hash)
    accepted = expected_input_signature(ACCEPTED_MANIFEST)
    if accepted["sha256"] != ACCEPTED_MANIFEST_FILE_SHA256:
        raise RuntimeError("accepted manifest changed while staging Round 0015")
    references.append({"role": "accepted_pack_manifest_file", **accepted})
    return sorted(references, key=lambda item: item["canonical_path"])


def validate_input_reference_manifest(value: dict[str, Any], *,
                                      full_hash: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Round 0015 input-reference manifest must be an object")
    identity = value.get("identity_sha256")
    body = {key: value[key] for key in value if key != "identity_sha256"}
    required = {
        "schema", "round_id", "payloads_copied", "accepted_tuple",
        "sequenced_review", "release_evidence", "reference_count",
        "references", "identity_sha256",
    }
    if (identity != sha256_bytes(canonical_json(body)) or set(value) != required or
            value["schema"] != REFERENCE_SCHEMA or value["round_id"] != ROUND_ID or
            value["payloads_copied"] is not False):
        raise ValueError("Round 0015 input-reference fields/seal changed")
    if value["accepted_tuple"] != {
        "manifest_path": ACCEPTED_MANIFEST,
        "manifest_file_sha256": ACCEPTED_MANIFEST_FILE_SHA256,
        "manifest_receipt_sha256": ACCEPTED_MANIFEST_RECEIPT_SHA256,
        "capability_sha256": ACCEPTED_CAPABILITY_SHA256,
        "implementation_release_commit":
            "7dd55bf79d73a894f7e1354c803ec725fc9a7579",
    }:
        raise ValueError("Round 0015 accepted input-pack tuple changed")
    if value["sequenced_review"] != {
        "path": SEQUENCED_REVIEW_FILE, "sha256": SEQUENCED_REVIEW_SHA256,
        "status": "accepted", "releases": ["0015"], "blocks": ["0006"],
    }:
        raise ValueError("Round 0015 sequenced review binding changed")
    validate_release_evidence(value["release_evidence"])
    expected = _expected_references(full_hash=full_hash)
    if value["reference_count"] != 77 or value["references"] != expected:
        raise ValueError("Round 0015 reference-only closure changed")
    return value


def stage_input_references(*, round_root: str, release_root: str,
                           release_sha: str) -> dict[str, Any]:
    root = os.path.realpath(round_root)
    if root != ROUND_ROOT or not os.path.isdir(root) or os.path.islink(root):
        raise ValueError(f"Round 0015 staging root must be existing {ROUND_ROOT}")
    release = verify_release_chain(release_root, release_sha)
    inputs = os.path.join(root, "inputs")
    create_fresh_directory(inputs, label="Round 0015 reference staging root")
    manifest_path = os.path.join(inputs, "immutable-input-references.json")
    config_path = os.path.join(inputs, "production-config.json")
    canary_path = os.path.join(inputs, "canary-derivation.json")
    transform_path = os.path.join(inputs, "transform-spec-template.json")
    for path in (manifest_path, config_path, canary_path, transform_path):
        refuse_existing(path, label="Round 0015 staged reference record")
    references = _expected_references(full_hash=True)
    body = {
        "schema": REFERENCE_SCHEMA, "round_id": ROUND_ID,
        "payloads_copied": False,
        "accepted_tuple": {
            "manifest_path": ACCEPTED_MANIFEST,
            "manifest_file_sha256": ACCEPTED_MANIFEST_FILE_SHA256,
            "manifest_receipt_sha256": ACCEPTED_MANIFEST_RECEIPT_SHA256,
            "capability_sha256": ACCEPTED_CAPABILITY_SHA256,
            "implementation_release_commit":
                "7dd55bf79d73a894f7e1354c803ec725fc9a7579",
        },
        "sequenced_review": {
            "path": SEQUENCED_REVIEW_FILE, "sha256": SEQUENCED_REVIEW_SHA256,
            "status": "accepted", "releases": ["0015"], "blocks": ["0006"],
        },
        "release_evidence": release,
        "reference_count": len(references), "references": references,
    }
    manifest = _seal(body)
    validate_input_reference_manifest(manifest, full_hash=False)
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    atomic_write_new_json(config_path, {
        "schema": "round0015-production-config-receipt-v1",
        "config": TRAIN_CONFIG, "config_sha256": TRAIN_CONFIG_SHA256,
    }, immutable=True)
    canary = {
        "schema": "round0015-canary-derivation-v1", "round_id": ROUND_ID,
        "release_sha": release_sha,
        "production_config_sha256": TRAIN_CONFIG_SHA256,
        "optimizer_updates": 0, "required_pipeline": "device_uniform",
        "sampling": "uniform-over-directed-edges",
        "minimum_post_setup_headroom_gib": 1.5,
        "scorer_cache_modes": ["off", "on-build", "on-hit"],
        "scalar_equivalence": "every registered scalar and numerator bitwise equal",
        "semantic_id_alignment": "same deterministic ID universe and gathered rows",
        "full_queue_p90_margin": 1.15, "gpu_hours_cap": 5.5,
        "retry_count": 0,
    }
    atomic_write_new_json(canary_path, _seal(canary), immutable=True)
    from .round0014_transform import build_transform_template
    transform = build_transform_template(
        release_root=release_root, release_sha=release_sha,
        train_output_relative_path="artifacts/train/model.pt")
    atomic_write_new_json(transform_path, transform, immutable=True)
    return {
        "schema": "round0015-reference-staging-v1", "release": release,
        "input_reference_manifest": expected_input_signature(manifest_path),
        "production_config": expected_input_signature(config_path),
        "canary_derivation": expected_input_signature(canary_path),
        "transform_spec_template": expected_input_signature(transform_path),
        "payloads_copied": False, "registered_references": 77,
    }
