"""Authoritative manifest-publication to Roundwatch gate-preparation boundary."""
from __future__ import annotations

import json
import os
from typing import Any

from .artifact_identity import canonical_json, expected_input_signature, sha256_bytes, sha256_file
from .output_safety import atomic_write_new_json, refuse_existing
from .queue_admission import _emit_component_rejection_receipt, validate_queue_manifest
from .release_preflight import validate_release_preflight_receipt
from .roundwatch_gate import (RoundwatchGateAuthority,
                              validate_roundwatch_binding)


def _release_receipt_path(manifest: dict[str, Any], *, fixture_only: bool = False) -> str:
    if fixture_only:
        value = manifest.get("release_preflight_receipt")
        if isinstance(value, str):
            return value
    matches = [entry["signature"]["canonical_path"]
               for entry in manifest.get("program_inputs", [])
               if entry.get("role") == "release_preflight_receipt"]
    if len(matches) != 1:
        raise RuntimeError("queue has no unique release-preflight role")
    return matches[0]


def _validate_bound_release(manifest: dict[str, Any], *, fixture_only: bool) -> dict:
    path = _release_receipt_path(manifest, fixture_only=fixture_only)
    matches = [signature for signature in manifest.get("global_input_registry", [])
               if isinstance(signature, dict) and
               signature.get("canonical_path") == path]
    if len(matches) != 1:
        raise RuntimeError("queue has no unique sealed release-preflight signature")
    return validate_release_preflight_receipt(
        path, expected_identity_sha256=manifest["release_preflight_identity"],
        expected_signature=matches[0])


def validate_gate_preparation_receipt(path: str, *, manifest_path: str,
                                      manifest: dict[str, Any]) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        receipt = json.load(handle)
    identity = receipt.get("identity_sha256") if isinstance(receipt, dict) else None
    body = {key: receipt[key] for key in receipt if key != "identity_sha256"} \
        if isinstance(receipt, dict) else {}
    expected_fields = {
        "schema", "manifest_path", "manifest_sha256", "round_sha256",
        "release_sha", "release_preflight_identity", "gate", "identity_sha256",
    }
    gate = receipt.get("gate") if isinstance(receipt, dict) else None
    fixture_only = manifest.get("schema") == "round0005_fixture_queue.v2"
    release = _validate_bound_release(manifest, fixture_only=fixture_only)
    validate_roundwatch_binding(manifest["roundwatch_binding"])
    try:
        gpu_hours_match = (float(gate.get("gpu_hours", -1)) ==
                           float(manifest["gpu_hours_cap"]))
    except (AttributeError, TypeError, ValueError):
        gpu_hours_match = False
    gate_common = (
        isinstance(gate, dict) and isinstance(gate.get("id"), int) and
        not isinstance(gate.get("id"), bool) and gate["id"] > 0 and
        gate.get("program") == manifest["program"] and
        gate.get("round_id") == manifest["round_id"] and
        gate.get("round_sha") == manifest["round_sha256"] and
        gate.get("release_sha") == manifest["release_sha"] and
        gate.get("env_sha") == manifest["environment_freeze_sha"] and
        gate.get("env_identity_sha") == manifest["environment_identity_sha"] and
        gate.get("queue_manifest_path") == os.path.realpath(manifest_path) and
        gate.get("queue_manifest_sha") == sha256_file(manifest_path) and
        gpu_hours_match)
    expected_status = "pending"
    fixture_approval = (not fixture_only or (
        isinstance(gate, dict) and gate.get("approval") is None))
    if (not isinstance(receipt, dict) or set(receipt) != expected_fields or
            os.path.realpath(path) != os.path.realpath(
                manifest["gate_preparation_receipt"]) or
            receipt.get("schema") != "round0005_gate_preparation_receipt.v1" or
            receipt.get("manifest_path") != os.path.realpath(manifest_path) or
            receipt.get("manifest_sha256") != sha256_file(manifest_path) or
            receipt.get("release_sha") != manifest["release_sha"] or
            receipt.get("round_sha256") != manifest["round_sha256"] or
            receipt.get("release_preflight_identity") !=
            manifest["release_preflight_identity"] or
            receipt.get("release_preflight_identity") != release["identity_sha256"] or
            receipt.get("manifest_sha256") != receipt.get("gate", {}).get(
                "queue_manifest_sha") or
            receipt.get("manifest_path") != receipt.get("gate", {}).get(
                "queue_manifest_path") or
            not gate_common or gate.get("status") != expected_status or
            not fixture_approval or
            not isinstance(identity, str) or
            sha256_bytes(canonical_json(body)) != identity):
        raise RuntimeError("gate preparation receipt is forged or does not bind this queue")
    return receipt


def prepare_gate(manifest_path: str, *, _fixture_authority=None,
                 _fixture_phase_hook=None, _fixture_validator=None) -> dict[str, Any]:
    """Revalidate a published queue and prepare its exact canonical gate."""
    path = os.path.realpath(manifest_path)
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    authority = _fixture_authority or RoundwatchGateAuthority()
    fixture_only = getattr(authority, "fixture_only", False) is True
    if (_fixture_authority is not None or _fixture_phase_hook is not None or
            _fixture_validator is not None) and not fixture_only:
        raise RuntimeError("fixture gate interfaces require explicit fixture-only authority")
    if fixture_only and _fixture_validator is None:
        raise RuntimeError("fixture gate preparation requires its exact fixture validator")
    validator = _fixture_validator if fixture_only else validate_queue_manifest
    validator(manifest, path)
    manifest_sha = sha256_file(path)
    release = _validate_bound_release(manifest, fixture_only=fixture_only)
    if release["identity_sha256"] != manifest["release_preflight_identity"]:
        raise RuntimeError("release-preflight receipt changed before gate preparation")
    receipt_path = manifest["gate_preparation_receipt"]
    refuse_existing(receipt_path, label="gate preparation receipt")
    if _fixture_phase_hook is not None:
        _fixture_phase_hook("manifest-publication-to-gate-preparation", manifest)
    try:
        if sha256_file(path) != manifest_sha:
            raise RuntimeError("published queue manifest changed before gate preparation")
        validator(manifest, path)
    except Exception as exc:
        observed = {
            "manifest_sha256": sha256_file(path) if os.path.isfile(path) else None,
            "global_inputs": [
                (expected_input_signature(value["canonical_path"])
                 if os.path.exists(value["canonical_path"]) else
                 {"canonical_path": value["canonical_path"], "missing": True})
                for value in manifest.get("global_input_registry", [])
            ],
        }
        expected = {"manifest_sha256": manifest_sha,
                    "global_inputs": manifest.get("global_input_registry")}
        receipt = _emit_component_rejection_receipt(
            receipts_dir=manifest["gate_receipts_dir"],
            phase="manifest-publication-to-gate-preparation",
            manifest_path=path, original_manifest_sha256=manifest_sha,
            job=manifest["jobs"][0], expected=expected, observed=observed,
            error=f"{type(exc).__name__}: {exc}")
        try:
            exc.add_note(f"automatic gate-preparation rejection receipt: {receipt}")
        except AttributeError:
            pass
        raise

    gate = authority.prepare(
        manifest=manifest, manifest_path=path, manifest_sha256=manifest_sha)
    body = {
        "schema": "round0005_gate_preparation_receipt.v1",
        "manifest_path": path,
        "manifest_sha256": manifest_sha,
        "round_sha256": manifest["round_sha256"],
        "release_sha": manifest["release_sha"],
        "release_preflight_identity": release["identity_sha256"],
        "gate": gate,
    }
    receipt = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    if _fixture_phase_hook is not None:
        _fixture_phase_hook("gate-preparation-to-admission", manifest)
    return receipt
