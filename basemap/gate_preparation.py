"""Authoritative manifest-publication to Roundwatch gate-preparation boundary."""
from __future__ import annotations

import importlib
import json
import os
import re
import time
from datetime import datetime, timezone
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


def _round0015_service_binding(manifest: dict[str, Any], *,
                               require_current: bool) -> dict[str, Any]:
    """Bind the single construction snapshot and re-probe its exact service."""
    matches = [entry["signature"] for entry in manifest.get("program_inputs", [])
               if entry.get("role") == "service_decision"]
    if len(matches) != 1:
        raise RuntimeError("Round 0015 gate has no unique service decision")
    signature = matches[0]
    from .round0015_service import load_service_decision
    decision = load_service_decision(
        signature["canonical_path"],
        environment_manifest=manifest["environment_manifest"],
        require_current=require_current)
    snapshot_body = {
        "gpu": decision["gpu"],
        "declared_services": decision["declared_services"],
        "memory_reservation": decision["memory_reservation"],
        "allowed_validation": decision["allowed_validation"],
    }
    return {
        "schema": "round0015-service-construction-binding-v1",
        "policy": decision["policy"],
        "decision_signature": signature,
        "decision_identity_sha256": decision["identity_sha256"],
        "construction_snapshot_sha256": sha256_bytes(
            canonical_json(snapshot_body)),
        "allowed_processes": decision["allowed_processes"],
        "current_identity_revalidated": bool(require_current),
        "revalidated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="microseconds"),
    }


def _validate_round0015_service_binding(value: Any,
                                        manifest: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {
            "schema", "policy", "decision_signature",
            "decision_identity_sha256", "construction_snapshot_sha256",
            "allowed_processes", "current_identity_revalidated",
            "revalidated_at_utc"}:
        raise RuntimeError("Round 0015 gate service binding fields changed")
    expected = _round0015_service_binding(manifest, require_current=False)
    stable = set(expected) - {"revalidated_at_utc", "current_identity_revalidated"}
    try:
        checked = datetime.fromisoformat(
            value["revalidated_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Round 0015 gate service revalidation time is invalid") from exc
    if (checked.tzinfo is None or value["current_identity_revalidated"] is not True or
            any(value[key] != expected[key] for key in stable) or
            value["allowed_processes"] != manifest.get("allowed_processes")):
        raise RuntimeError("Round 0015 gate service snapshot/decision binding changed")


def _round0016_exclusive_binding(manifest: dict[str, Any], *,
                                 require_current: bool) -> dict[str, Any]:
    """Bind and revalidate the target-specific empty-GPU construction."""
    matches = [entry["signature"] for entry in manifest.get("program_inputs", [])
               if entry.get("role") == "exclusive_gpu_decision"]
    if len(matches) != 1:
        raise RuntimeError("Round 0016 gate has no unique exclusive decision")
    signature = matches[0]
    loader = importlib.import_module(
        ".round0016_service", __package__).load_exclusive_decision
    decision = loader(
        signature["canonical_path"],
        environment_manifest=manifest["environment_manifest"],
        require_current=require_current)
    snapshot_body = {
        "gpu": decision["gpu"],
        "memory_reservation": decision["memory_reservation"],
        "allowed_validation": decision["allowed_validation"],
        "service_marker": decision["service_marker"],
        "service_reservation_mib": decision["service_reservation_mib"],
    }
    return {
        "schema": "round0016-exclusive-gpu-construction-binding-v1",
        "policy": decision["policy"],
        "decision_signature": signature,
        "decision_identity_sha256": decision["identity_sha256"],
        "construction_snapshot_sha256": sha256_bytes(
            canonical_json(snapshot_body)),
        "allowed_processes": [],
        "current_identity_revalidated": bool(require_current),
        "revalidated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="microseconds"),
    }


def _validate_round0016_exclusive_binding(value: Any,
                                          manifest: dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {
            "schema", "policy", "decision_signature",
            "decision_identity_sha256", "construction_snapshot_sha256",
            "allowed_processes", "current_identity_revalidated",
            "revalidated_at_utc"}:
        raise RuntimeError("Round 0016 gate exclusive binding fields changed")
    expected = _round0016_exclusive_binding(manifest, require_current=False)
    stable = set(expected) - {"revalidated_at_utc", "current_identity_revalidated"}
    try:
        checked = datetime.fromisoformat(
            value["revalidated_at_utc"].replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Round 0016 gate revalidation time is invalid") from exc
    if (checked.tzinfo is None or value["current_identity_revalidated"] is not True or
            any(value[key] != expected[key] for key in stable) or
            value["allowed_processes"] != manifest.get("allowed_processes") or
            value["allowed_processes"] != []):
        raise RuntimeError("Round 0016 exclusive snapshot/decision binding changed")


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
    round0015 = manifest.get("round_id") == "0015"
    round0016 = manifest.get("round_id") == "0016"
    if round0015:
        expected_fields.add("service_construction")
    if round0016:
        expected_fields.add("exclusive_gpu_construction")
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
    if round0015:
        _validate_round0015_service_binding(
            receipt.get("service_construction") if isinstance(receipt, dict) else None,
            manifest)
    if round0016:
        _validate_round0016_exclusive_binding(
            receipt.get("exclusive_gpu_construction")
            if isinstance(receipt, dict) else None, manifest)
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


def seal_exact_round0014_prepared_gate(
        manifest_path: str, gate: dict[str, Any]) -> dict[str, Any]:
    """Seal the result of the one externally-invoked exact prepare-gate CLI.

    This function never contacts Roundwatch.  The supervised CPU runner invokes
    the mandated CLI exactly once, then passes that returned pending record
    here so the later controller can reopen the same immutable sidecar.
    """
    path = os.path.realpath(manifest_path)
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("round_id") != "0014":
        raise RuntimeError("external prepared-gate sealing is Round 0014 only")
    validate_queue_manifest(manifest, path)
    manifest_sha = sha256_file(path)
    release = _validate_bound_release(manifest, fixture_only=False)
    receipt_path = manifest["gate_preparation_receipt"]
    refuse_existing(receipt_path, label="Round 0014 gate preparation receipt")
    try:
        gpu_hours = float(gate.get("gpu_hours", -1))
        expires_at = float(gate.get("expires_at", 0))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Round 0014 prepared gate has malformed cap/expiry") from exc
    if (not isinstance(gate, dict) or
            not isinstance(gate.get("id"), int) or
            isinstance(gate.get("id"), bool) or gate["id"] <= 0 or
            gate.get("program") != manifest["program"] or
            gate.get("round_id") != "0014" or
            gate.get("round_sha") != manifest["round_sha256"] or
            gate.get("release_sha") != manifest["release_sha"] or
            gate.get("env_sha") != manifest["environment_freeze_sha"] or
            gate.get("env_identity_sha") != manifest["environment_identity_sha"] or
            not re.fullmatch(r"[0-9a-f]{64}", str(gate.get("reviews_sha", ""))) or
            gate.get("queue_manifest_path") != path or
            gate.get("queue_manifest_sha") != manifest_sha or
            gpu_hours != float(manifest["gpu_hours_cap"]) or
            gate.get("status") != "pending" or gate.get("approval") is not None or
            expires_at <= time.time()):
        raise RuntimeError("Round 0014 CLI did not return the exact pending owner gate")
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
    return validate_gate_preparation_receipt(
        receipt_path, manifest_path=path, manifest=manifest)


def seal_exact_round0015_prepared_gate(
        manifest_path: str, gate: dict[str, Any]) -> dict[str, Any]:
    """Seal the sole externally invoked Round-0015 pending owner gate."""
    path = os.path.realpath(manifest_path)
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("round_id") != "0015":
        raise RuntimeError("external prepared-gate sealing is Round 0015 only")
    validate_queue_manifest(manifest, path)
    manifest_sha = sha256_file(path)
    release = _validate_bound_release(manifest, fixture_only=False)
    receipt_path = manifest["gate_preparation_receipt"]
    refuse_existing(receipt_path, label="Round 0015 gate preparation receipt")
    # This is the gate-preparation re-probe required by allow_exact_service.
    service_construction = _round0015_service_binding(
        manifest, require_current=True)
    try:
        gpu_hours = float(gate.get("gpu_hours", -1))
        expires_at = float(gate.get("expires_at", 0))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Round 0015 prepared gate has malformed cap/expiry") from exc
    if (not isinstance(gate, dict) or
            not isinstance(gate.get("id"), int) or
            isinstance(gate.get("id"), bool) or gate["id"] <= 0 or
            gate.get("program") != manifest["program"] or
            gate.get("round_id") != "0015" or
            gate.get("round_sha") != manifest["round_sha256"] or
            gate.get("release_sha") != manifest["release_sha"] or
            gate.get("env_sha") != manifest["environment_freeze_sha"] or
            gate.get("env_identity_sha") != manifest["environment_identity_sha"] or
            not re.fullmatch(r"[0-9a-f]{64}", str(gate.get("reviews_sha", ""))) or
            gate.get("queue_manifest_path") != path or
            gate.get("queue_manifest_sha") != manifest_sha or
            gpu_hours != float(manifest["gpu_hours_cap"]) or
            gate.get("status") != "pending" or gate.get("approval") is not None or
            expires_at <= time.time()):
        raise RuntimeError("Round 0015 CLI did not return the exact pending owner gate")
    body = {
        "schema": "round0005_gate_preparation_receipt.v1",
        "manifest_path": path,
        "manifest_sha256": manifest_sha,
        "round_sha256": manifest["round_sha256"],
        "release_sha": manifest["release_sha"],
        "release_preflight_identity": release["identity_sha256"],
        "service_construction": service_construction,
        "gate": gate,
    }
    receipt = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return validate_gate_preparation_receipt(
        receipt_path, manifest_path=path, manifest=manifest)


def seal_exact_round0016_prepared_gate(
        manifest_path: str, gate: dict[str, Any]) -> dict[str, Any]:
    """Seal the sole externally invoked Round-0016 pending owner gate."""
    path = os.path.realpath(manifest_path)
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("round_id") != "0016":
        raise RuntimeError("external prepared-gate sealing is Round 0016 only")
    validate_queue_manifest(manifest, path)
    manifest_sha = sha256_file(path)
    release = _validate_bound_release(manifest, fixture_only=False)
    receipt_path = manifest["gate_preparation_receipt"]
    refuse_existing(receipt_path, label="Round 0016 gate preparation receipt")
    exclusive = _round0016_exclusive_binding(manifest, require_current=True)
    try:
        gpu_hours = float(gate.get("gpu_hours", -1))
        expires_at = float(gate.get("expires_at", 0))
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("Round 0016 prepared gate has malformed cap/expiry") from exc
    if (not isinstance(gate, dict) or
            not isinstance(gate.get("id"), int) or
            isinstance(gate.get("id"), bool) or gate["id"] <= 0 or
            gate.get("program") != manifest["program"] or
            gate.get("round_id") != "0016" or
            gate.get("round_sha") != manifest["round_sha256"] or
            gate.get("release_sha") != manifest["release_sha"] or
            gate.get("env_sha") != manifest["environment_freeze_sha"] or
            gate.get("env_identity_sha") != manifest["environment_identity_sha"] or
            not re.fullmatch(r"[0-9a-f]{64}", str(gate.get("reviews_sha", ""))) or
            gate.get("queue_manifest_path") != path or
            gate.get("queue_manifest_sha") != manifest_sha or
            gpu_hours != float(manifest["gpu_hours_cap"]) or
            gate.get("status") != "pending" or gate.get("approval") is not None or
            expires_at <= time.time()):
        raise RuntimeError("Round 0016 CLI did not return the exact pending owner gate")
    body = {
        "schema": "round0005_gate_preparation_receipt.v1",
        "manifest_path": path,
        "manifest_sha256": manifest_sha,
        "round_sha256": manifest["round_sha256"],
        "release_sha": manifest["release_sha"],
        "release_preflight_identity": release["identity_sha256"],
        "exclusive_gpu_construction": exclusive,
        "gate": gate,
    }
    receipt = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return validate_gate_preparation_receipt(
        receipt_path, manifest_path=path, manifest=manifest)
