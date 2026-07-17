"""Reference-only staging for the immutable Round 0014 production inputs."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                sha256_bytes)
from .output_safety import (atomic_write_new_json, create_fresh_directory,
                            refuse_existing)
from .round0014_program import (
    ACCEPTED_CAPABILITY_SHA256, ACCEPTED_MANIFEST,
    ACCEPTED_MANIFEST_FILE_SHA256, ACCEPTED_MANIFEST_RECEIPT_SHA256, ISSUED_BASE,
    ROUND_ID, SEQUENCED_REVIEW_FILE, SEQUENCED_REVIEW_SHA256, TRAIN_CONFIG,
    TRAIN_CONFIG_SHA256, accepted_reference_records,
)


REFERENCE_SCHEMA = "round0014-input-reference-manifest-v1"
ROUND_ROOT = "/data/latent-basemap/runs/round-0014"


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


def _verify_release(release_root: str, release_sha: str) -> dict[str, Any]:
    root = os.path.realpath(release_root)
    if root != release_root or _git(root, "rev-parse", "HEAD") != release_sha:
        raise RuntimeError("Round 0014 staging release checkout/commit changed")
    if _git(root, "rev-parse", "HEAD^") != ISSUED_BASE:
        raise RuntimeError("Round 0014 staging release is not the issued base's direct child")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Round 0014 staging release checkout is dirty")
    if subprocess.run(
            ["/usr/bin/git", "-C", root, "symbolic-ref", "-q", "HEAD"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError("Round 0014 staging requires the detached run checkout")
    return {"release_root": root, "release_sha": release_sha,
            "parent": ISSUED_BASE, "tree": _git(root, "rev-parse", "HEAD^{tree}"),
            "detached": True, "clean": True}


def _expected_references(*, full_hash: bool) -> list[dict[str, Any]]:
    references = accepted_reference_records(full_hash=full_hash)
    accepted = expected_input_signature(ACCEPTED_MANIFEST)
    if accepted["sha256"] != ACCEPTED_MANIFEST_FILE_SHA256:
        raise RuntimeError("accepted manifest changed while staging references")
    references.append({"role": "accepted_pack_manifest_file", **accepted})
    return sorted(references, key=lambda item: item["canonical_path"])


def validate_input_reference_manifest(value: dict[str, Any], *,
                                      full_hash: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Round 0014 input-reference manifest must be an object")
    identity = value.get("identity_sha256")
    body = {key: value[key] for key in value if key != "identity_sha256"}
    if identity != sha256_bytes(canonical_json(body)):
        raise ValueError("Round 0014 input-reference manifest seal changed")
    required = {
        "schema", "round_id", "payloads_copied", "accepted_tuple",
        "sequenced_review", "reference_count", "references", "identity_sha256",
    }
    if set(value) != required or value["schema"] != REFERENCE_SCHEMA \
            or value["round_id"] != ROUND_ID or value["payloads_copied"] is not False:
        raise ValueError("Round 0014 input-reference manifest fields/policy changed")
    if value["accepted_tuple"] != {
        "manifest_path": ACCEPTED_MANIFEST,
        "manifest_file_sha256": ACCEPTED_MANIFEST_FILE_SHA256,
        "manifest_receipt_sha256": ACCEPTED_MANIFEST_RECEIPT_SHA256,
        "capability_sha256": ACCEPTED_CAPABILITY_SHA256,
        "implementation_release_commit": ISSUED_BASE,
    }:
        raise ValueError("Round 0014 accepted input-pack tuple changed")
    if value["sequenced_review"] != {
        "path": SEQUENCED_REVIEW_FILE, "sha256": SEQUENCED_REVIEW_SHA256,
        "status": "accepted", "releases": ["0014"], "blocks": ["0006"],
    }:
        raise ValueError("Round 0014 sequenced review binding changed")
    expected = _expected_references(full_hash=full_hash)
    if value["reference_count"] != 77 or value["references"] != expected:
        raise ValueError("Round 0014 reference-only closure changed")
    return value


def stage_input_references(*, round_root: str, release_root: str,
                           release_sha: str) -> dict[str, Any]:
    """Hash/reopen in place and publish only small path/hash records."""
    root = os.path.realpath(round_root)
    if root != ROUND_ROOT or not os.path.isdir(root) or os.path.islink(root):
        raise ValueError(f"Round 0014 staging root must be existing {ROUND_ROOT}")
    release = _verify_release(release_root, release_sha)
    inputs = os.path.join(root, "inputs")
    create_fresh_directory(inputs, label="Round 0014 reference staging root")
    manifest_path = os.path.join(inputs, "immutable-input-references.json")
    config_path = os.path.join(inputs, "production-config.json")
    canary_path = os.path.join(inputs, "canary-derivation.json")
    transform_path = os.path.join(inputs, "transform-spec-template.json")
    for path in (manifest_path, config_path, canary_path, transform_path):
        refuse_existing(path, label="Round 0014 staged reference record")

    references = _expected_references(full_hash=True)
    body = {
        "schema": REFERENCE_SCHEMA,
        "round_id": ROUND_ID,
        "payloads_copied": False,
        "accepted_tuple": {
            "manifest_path": ACCEPTED_MANIFEST,
            "manifest_file_sha256": ACCEPTED_MANIFEST_FILE_SHA256,
            "manifest_receipt_sha256": ACCEPTED_MANIFEST_RECEIPT_SHA256,
            "capability_sha256": ACCEPTED_CAPABILITY_SHA256,
            "implementation_release_commit": ISSUED_BASE,
        },
        "sequenced_review": {
            "path": SEQUENCED_REVIEW_FILE, "sha256": SEQUENCED_REVIEW_SHA256,
            "status": "accepted", "releases": ["0014"], "blocks": ["0006"],
        },
        "reference_count": len(references),
        "references": references,
    }
    reference_manifest = _seal(body)
    validate_input_reference_manifest(reference_manifest, full_hash=False)
    atomic_write_new_json(manifest_path, reference_manifest, immutable=True)

    config_record = {
        "schema": "round0014-production-config-receipt-v1",
        "config": TRAIN_CONFIG,
        "config_sha256": TRAIN_CONFIG_SHA256,
    }
    atomic_write_new_json(config_path, config_record, immutable=True)
    canary_body = {
        "schema": "round0014-canary-derivation-v1",
        "round_id": ROUND_ID,
        "release_sha": release_sha,
        "production_config_sha256": TRAIN_CONFIG_SHA256,
        "optimizer_updates": 0,
        "required_pipeline": "device_uniform",
        "sampling": "uniform-over-directed-edges",
        "minimum_post_setup_headroom_gib": 1.5,
        "scorer_cache_modes": ["off", "on-build", "on-hit"],
        "scalar_equivalence": "every registered scalar and numerator bitwise equal",
        "semantic_id_alignment": "same deterministic ID universe and gathered rows",
        "full_queue_p90_margin": 1.15,
        "gpu_hours_cap": 5.5,
        "retry_count": 0,
    }
    atomic_write_new_json(canary_path, _seal(canary_body), immutable=True)

    from .round0014_transform import build_transform_template
    transform = build_transform_template(
        release_root=release_root, release_sha=release_sha,
        train_output_relative_path="artifacts/train/model.pt")
    atomic_write_new_json(transform_path, transform, immutable=True)
    return {
        "schema": "round0014-reference-staging-v1",
        "release": release,
        "input_reference_manifest": expected_input_signature(manifest_path),
        "production_config": expected_input_signature(config_path),
        "canary_derivation": expected_input_signature(canary_path),
        "transform_spec_template": expected_input_signature(transform_path),
        "payloads_copied": False,
        "registered_references": 77,
    }
