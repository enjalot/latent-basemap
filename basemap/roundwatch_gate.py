"""Canonical Roundwatch authority binding for Round 0005.

Production always executes the shared CLI from the workshop checkout.  There is
no caller-provided binary or URL: the subprocess environment removes Roundwatch
endpoint overrides and verifies that the checkout is at the required commit or
a descendant before every call.  A private in-memory authority exists only for
CUDA-hidden integration fixtures and is not selectable by the production CLI.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import time
from typing import Any

from .artifact_identity import (canonical_json, expected_input_signature,
                                path_signature, sha256_bytes, sha256_file)

WORKSHOP_ROOT = "/home/enjalot/code/workshop"
ROUNDWATCH_CLI = "/home/enjalot/code/workshop/bin/roundwatch"
ROUNDWATCH_CLI_PY = "/home/enjalot/code/workshop/rounds/cli.py"
ROUNDWATCH_CORE_PY = "/home/enjalot/code/workshop/rounds/core.py"
ROUNDWATCH_ROUNDS_DIR = "/home/enjalot/code/workshop/rounds"
ROUNDWATCH_PYTHON = "/usr/bin/python3.12"
ROUNDWATCH_GIT = "/usr/bin/git"
ROUNDWATCH_URL = "http://127.0.0.1:8710"
ROUNDWATCH_STATE_DIR = "/home/enjalot/.agent/rounds"
MINIMUM_WORKSHOP_COMMIT = "14a338d43d130338b1056451437d33070d7d57d5"
ISSUED_PROTOCOL_COMMIT = "d9bc73d6823bb05fabd1b5e0722a9b2f2de3ad11"
ROUNDWATCH_AUTHORITY_VALUES = {"planner-no-training", "planner-gpu", "owner-gpu"}
ROUNDWATCH_ISOLATED_BOOTSTRAP = (
    "import runpy,sys\n"
    f"sys.path.insert(0, {ROUNDWATCH_ROUNDS_DIR!r})\n"
    f"sys.argv = [{ROUNDWATCH_CLI_PY!r}, *sys.argv[1:]]\n"
    f"runpy.run_path({ROUNDWATCH_CLI_PY!r}, run_name='__main__')\n"
)


def _closed_process_environment() -> dict[str, str]:
    """Environment shared by the canonical Python and Git authority boundaries."""
    return {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}


def _git(*args: str) -> str:
    proc = subprocess.run(
        [ROUNDWATCH_GIT, "-C", WORKSHOP_ROOT, *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=_closed_process_environment())
    if proc.returncode:
        raise RuntimeError(f"workshop git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def canonical_roundwatch_binding() -> dict[str, Any]:
    root = os.path.realpath(WORKSHOP_ROOT)
    paths = (ROUNDWATCH_CLI, ROUNDWATCH_CLI_PY, ROUNDWATCH_CORE_PY,
             ROUNDWATCH_PYTHON, ROUNDWATCH_GIT)
    if root != WORKSHOP_ROOT or any(os.path.realpath(value) != value for value in paths):
        raise RuntimeError("canonical workshop/Roundwatch path traverses a symlink")
    if any(not os.path.isfile(value) or os.path.islink(value) for value in paths):
        raise RuntimeError("canonical Roundwatch interpreter/import closure is missing or symlinked")
    head = _git("rev-parse", "HEAD")
    if subprocess.run(
            [ROUNDWATCH_GIT, "-C", root, "merge-base", "--is-ancestor",
             MINIMUM_WORKSHOP_COMMIT, head], env=_closed_process_environment(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise RuntimeError(
            f"workshop Roundwatch {head} predates required {MINIMUM_WORKSHOP_COMMIT}")
    if subprocess.run(
            [ROUNDWATCH_GIT, "-C", root, "merge-base", "--is-ancestor",
             ISSUED_PROTOCOL_COMMIT, head], env=_closed_process_environment(),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise RuntimeError(
            f"workshop Roundwatch {head} lacks issued planner-gpu protocol "
            f"{ISSUED_PROTOCOL_COMMIT}")
    state = subprocess.run(
        [ROUNDWATCH_GIT, "-C", root, "status", "--porcelain=v1",
         "--untracked-files=all"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=_closed_process_environment())
    if state.returncode:
        raise RuntimeError(f"workshop tracked-state check failed: {state.stderr.strip()}")
    return {
        "schema": "round0005_roundwatch_binding.v2",
        "workshop_root": root,
        "minimum_commit": MINIMUM_WORKSHOP_COMMIT,
        "issued_protocol_commit": ISSUED_PROTOCOL_COMMIT,
        "workshop_commit": head,
        "tracked_worktree_clean": state.stdout == "",
        "worktree_porcelain_sha256": sha256_bytes(state.stdout.encode("utf-8")),
        "url": ROUNDWATCH_URL,
        "state_dir": ROUNDWATCH_STATE_DIR,
        "isolated_bootstrap_sha256": sha256_bytes(
            ROUNDWATCH_ISOLATED_BOOTSTRAP.encode("utf-8")),
        "interpreter": expected_input_signature(ROUNDWATCH_PYTHON),
        "git": expected_input_signature(ROUNDWATCH_GIT),
        "cli": path_signature(ROUNDWATCH_CLI),
        "import_closure": [
            expected_input_signature(ROUNDWATCH_CLI_PY),
            expected_input_signature(ROUNDWATCH_CORE_PY),
        ],
    }


def validate_roundwatch_binding(recorded: dict[str, Any]) -> dict[str, Any]:
    current = canonical_roundwatch_binding()
    if current != recorded:
        raise RuntimeError(
            "canonical Roundwatch implementation changed after queue publication: "
            f"expected={recorded!r} observed={current!r}")
    if current["tracked_worktree_clean"] is not True:
        raise RuntimeError("canonical Roundwatch workshop checkout has tracked modifications")
    return current


def _declared_authority(manifest: dict[str, Any]) -> str:
    value = manifest.get("execution_authority")
    if value is None and manifest.get("round_id") == "0005":
        value = "planner-gpu"
    if value not in ROUNDWATCH_AUTHORITY_VALUES:
        raise RuntimeError("queue has no valid declared Roundwatch execution authority")
    return value


def _round_file_signature(manifest: dict[str, Any]) -> dict[str, Any]:
    matches = [
        entry.get("signature") for entry in manifest.get("program_inputs", [])
        if isinstance(entry, dict) and entry.get("role") == "round_file"
    ]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise RuntimeError("queue has no unique signed full round file")
    signature = matches[0]
    if (expected_input_signature(signature.get("canonical_path", "")) != signature or
            signature.get("kind") != "file" or
            signature.get("sha256") != manifest.get("round_sha256")):
        raise RuntimeError("queue full round-file bytes do not match its issued hash")
    return signature


def _validate_gate_payload(service: dict[str, Any], events: dict[str, Any], *,
                           manifest: dict[str, Any], manifest_path: str,
                           manifest_sha256: str) -> dict[str, Any]:
    if not isinstance(service, dict) or service.get("approved") is not True:
        raise RuntimeError(f"Roundwatch gate is not approved: {service!r}")
    required = {
        "approved", "round_id", "release_sha", "env_sha", "env_identity_sha",
        "authority", "required_reviews_receipt", "reasons", "warnings",
        "gate_request", "run_repo", "checked_at",
    }
    missing = sorted(required - set(service))
    if missing:
        raise RuntimeError(f"Roundwatch gate response is minimal/incomplete: {missing}")
    authority = _declared_authority(manifest)
    if (service["round_id"] != manifest["round_id"] or
            service["release_sha"] != manifest["release_sha"] or
            service["env_sha"] != manifest["environment_freeze_sha"] or
            service["env_identity_sha"] != manifest["environment_identity_sha"] or
            service["authority"] != authority or service["reasons"] != [] or
            not isinstance(service.get("warnings"), list) or
            not isinstance(service.get("checked_at"), str)):
        raise RuntimeError("Roundwatch gate response does not bind queue/round/environment")
    gate = service["gate_request"]
    if not isinstance(gate, dict):
        raise RuntimeError("Roundwatch gate response has no current gate request")
    approval = gate.get("approval")
    status = gate.get("status")
    if authority == "owner-gpu":
        authority_status_valid = (
            status == "approved" and isinstance(approval, dict) and
            approval.get("decision") == "approved" and
            approval.get("revoked_at") is None)
    else:
        authority_status_valid = status in {"pending", "approved"}
        if status == "pending":
            authority_status_valid = authority_status_valid and approval is None
        else:
            authority_status_valid = authority_status_valid and (
                isinstance(approval, dict) and approval.get("decision") == "approved" and
                approval.get("revoked_at") is None)
    if (not isinstance(gate.get("id"), int) or gate.get("program") != manifest["program"] or
            gate.get("round_id") != manifest["round_id"] or
            gate.get("round_sha") != manifest["round_sha256"] or
            gate.get("release_sha") != manifest["release_sha"] or
            gate.get("env_sha") != manifest["environment_freeze_sha"] or
            gate.get("env_identity_sha") != manifest["environment_identity_sha"] or
            gate.get("queue_manifest_path") != os.path.realpath(manifest_path) or
            gate.get("queue_manifest_sha") != manifest_sha256 or
            float(gate.get("gpu_hours", -1)) != float(manifest["gpu_hours_cap"]) or
            not authority_status_valid or
            not gate.get("expires_at") or float(gate["expires_at"]) <= time.time()):
        raise RuntimeError("Roundwatch current gate request is stale or incompletely bound")
    reviews = service["required_reviews_receipt"]
    declared_reviews = manifest.get("required_reviews", [])
    observed_reviews = reviews.get("reviews") if isinstance(reviews, dict) else None
    observed_ids = ([entry.get("round_id") for entry in observed_reviews]
                    if isinstance(observed_reviews, list) and
                    all(isinstance(entry, dict) for entry in observed_reviews) else None)
    if (not isinstance(reviews, dict) or set(reviews) != {"sha256", "reviews"} or
            not isinstance(observed_reviews, list) or
            reviews.get("sha256") != sha256_bytes(canonical_json(observed_reviews)) or
            gate.get("reviews_sha") != reviews.get("sha256") or
            observed_ids != sorted(declared_reviews)):
        raise RuntimeError("Roundwatch gate required-review identity is stale")
    repo = service["run_repo"]
    if (not isinstance(repo, dict) or repo.get("head") != manifest["release_sha"] or
            repo.get("dirty") is not False or repo.get("detached") is not True or
            os.path.realpath(repo.get("path", "")) != os.path.realpath(manifest["repo_root"])):
        raise RuntimeError("Roundwatch gate run-checkout evidence is stale")

    round_signature = _round_file_signature(manifest)

    if (not isinstance(events, dict) or not isinstance(events.get("instance_id"), str) or
            not events["instance_id"] or not isinstance(events.get("latest"), int) or
            not isinstance(events.get("events"), list)):
        raise RuntimeError("Roundwatch event identity response is incomplete")
    round_events = [
        event for event in events["events"]
        if isinstance(event, dict) and event.get("program") == manifest["program"] and
        event.get("round_id") == manifest["round_id"] and
        event.get("type") in {"round.new", "round.changed"}
    ]
    if not round_events:
        raise RuntimeError("Roundwatch has no durable event for the current round file")
    round_event = max(round_events, key=lambda value: int(value.get("id", -1)))
    if (set(round_event.get("payload") or {}) != {"path", "sha256", "status"} or
            round_event["payload"].get("path") !=
            os.path.basename(round_signature["canonical_path"]) or
            round_event["payload"].get("sha256") != manifest["round_sha256"] or
            round_event["payload"].get("status") != "issued"):
        raise RuntimeError("Roundwatch current round event differs from the full round file")

    relevant = [
        event for event in events["events"]
        if isinstance(event, dict) and event.get("program") == manifest["program"] and
        event.get("round_id") == manifest["round_id"] and
        event.get("type") in {"gate.prepared", "owner.approved", "owner.rejected",
                              "owner.revoked"}
    ]
    prepared = [event for event in relevant
                if event.get("type") == "gate.prepared" and
                isinstance(event.get("payload"), dict) and
                event["payload"].get("id") == gate["id"]]
    if len(prepared) != 1:
        raise RuntimeError("Roundwatch has no unique durable gate.prepared event")
    prepared_event = prepared[0]
    prepared_payload = prepared_event["payload"]
    common_gate_fields = set(gate) - {"status", "approval"}
    if (prepared_payload.get("status") != "pending" or
            prepared_payload.get("approval") is not None or
            any(prepared_payload.get(key) != gate.get(key) for key in common_gate_fields)):
        raise RuntimeError("Roundwatch gate.prepared payload differs from current gate identity")
    if status == "approved":
        approved = [event for event in relevant
                    if event.get("type") == "owner.approved" and
                    isinstance(event.get("payload"), dict) and
                    event["payload"].get("id") == gate["id"]]
        if len(approved) != 1 or approved[0].get("payload") != gate:
            raise RuntimeError("Roundwatch has no exact durable approval for this gate")
        control_event = approved[0]
    else:
        control_event = prepared_event
    if any(int(event.get("id", -1)) > int(control_event["id"]) for event in relevant):
        raise RuntimeError("Roundwatch gate event is not the current control event")
    if events["latest"] < int(control_event["id"]):
        raise RuntimeError("Roundwatch latest-event identity predates gate control")
    return {
        "schema": "round0005_gate_evidence.v3",
        "service": service,
        "event_identity": {
            "instance_id": events["instance_id"],
            "latest_event_id": events["latest"],
            "round_event": round_event,
            "gate_prepared_event": prepared_event,
            "control_event": control_event,
            "gate_id": gate["id"],
            "authority": authority,
        },
        "manifest_sha256": manifest_sha256,
        "round_file": round_signature,
        "required_reviews_receipt": reviews,
        "round_file_sha256": manifest["round_sha256"],
        "release_sha": manifest["release_sha"],
        "environment_freeze_sha": manifest["environment_freeze_sha"],
        "environment_identity_sha": manifest["environment_identity_sha"],
    }


class RoundwatchGateAuthority:
    """Production-only authority backed by the fixed shared Roundwatch CLI."""

    fixture_only = False

    @staticmethod
    def _run_json(arguments: list[str], *, expected_binding: dict[str, Any]) -> dict[str, Any]:
        if canonical_roundwatch_binding() != expected_binding:
            raise RuntimeError("Roundwatch command/import identity changed before subprocess")
        command = [
            ROUNDWATCH_PYTHON, "-I", "-S", "-B", "-c", ROUNDWATCH_ISOLATED_BOOTSTRAP,
            "--url", ROUNDWATCH_URL, "--state-dir", ROUNDWATCH_STATE_DIR,
            *arguments,
        ]
        proc = subprocess.run(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=40, env=_closed_process_environment(), close_fds=True)
        if canonical_roundwatch_binding() != expected_binding:
            raise RuntimeError("Roundwatch command/import identity changed during subprocess")
        try:
            value = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"canonical Roundwatch returned non-JSON output: stdout={proc.stdout!r} "
                f"stderr={proc.stderr!r}") from exc
        if proc.returncode:
            raise RuntimeError(
                f"canonical Roundwatch rejected {arguments[0]}: {value!r}; "
                f"stderr={proc.stderr!r}")
        if not isinstance(value, dict):
            raise RuntimeError("canonical Roundwatch JSON response must be an object")
        return value

    def check(self, *, manifest: dict[str, Any], manifest_path: str,
              manifest_sha256: str) -> dict[str, Any]:
        binding = validate_roundwatch_binding(manifest["roundwatch_binding"])
        service = self._run_json([
            "gate-check", manifest["round_id"], "--program", manifest["program"],
            "--release-sha", manifest["release_sha"],
        ], expected_binding=binding)
        all_events: list[dict[str, Any]] = []
        after = 0
        instance_id = None
        latest = None
        while True:
            page = self._run_json(
                ["events", "--after", str(after)], expected_binding=binding)
            if instance_id is None:
                instance_id = page.get("instance_id")
                latest = page.get("latest")
            elif page.get("instance_id") != instance_id:
                raise RuntimeError("Roundwatch service instance changed during gate evidence read")
            events = page.get("events")
            if not isinstance(events, list):
                raise RuntimeError("Roundwatch events response is incomplete")
            all_events.extend(events)
            if not events or int(events[-1]["id"]) >= int(page.get("latest", -1)):
                latest = page.get("latest")
                break
            after = int(events[-1]["id"])
        evidence = _validate_gate_payload(
            service, {"instance_id": instance_id, "latest": latest, "events": all_events},
            manifest=manifest, manifest_path=manifest_path,
            manifest_sha256=manifest_sha256)
        # Recheck after the event snapshot.  ``checked_at`` is intentionally
        # fresh; every authority-bearing field and the current gate must remain
        # identical across the read boundary.
        final_service = self._run_json([
            "gate-check", manifest["round_id"], "--program", manifest["program"],
            "--release-sha", manifest["release_sha"],
        ], expected_binding=binding)
        final_evidence = _validate_gate_payload(
            final_service,
            {"instance_id": instance_id, "latest": latest, "events": all_events},
            manifest=manifest, manifest_path=manifest_path,
            manifest_sha256=manifest_sha256)
        unstable = {"checked_at"}
        first_stable = {key: value for key, value in evidence["service"].items()
                        if key not in unstable}
        final_stable = {key: value for key, value in final_evidence["service"].items()
                        if key not in unstable}
        if (first_stable != final_stable or
                evidence["event_identity"] != final_evidence["event_identity"]):
            raise RuntimeError("Roundwatch gate/control identity changed during authority read")
        return final_evidence

    def prepare(self, *, manifest: dict[str, Any], manifest_path: str,
                manifest_sha256: str) -> dict[str, Any]:
        """Prepare the fixed manifest with the canonical shared CLI."""
        binding = validate_roundwatch_binding(manifest["roundwatch_binding"])
        if sha256_file(manifest_path) != manifest_sha256:
            raise RuntimeError("queue manifest changed before Roundwatch gate preparation")
        gate = self._run_json([
            "prepare-gate", manifest["round_id"], "--program", manifest["program"],
            "--release-sha", manifest["release_sha"], "--gpu-hours",
            str(manifest["gpu_hours_cap"]), "--queue-manifest",
            os.path.realpath(manifest_path),
        ], expected_binding=binding)
        if (not isinstance(gate.get("id"), int) or gate.get("program") != manifest["program"] or
                gate.get("round_id") != manifest["round_id"] or
                gate.get("round_sha") != manifest["round_sha256"] or
                gate.get("release_sha") != manifest["release_sha"] or
                gate.get("env_sha") != manifest["environment_freeze_sha"] or
                gate.get("env_identity_sha") != manifest["environment_identity_sha"] or
                gate.get("queue_manifest_path") != os.path.realpath(manifest_path) or
                gate.get("queue_manifest_sha") != manifest_sha256 or
                float(gate.get("gpu_hours", -1)) != float(manifest["gpu_hours_cap"]) or
                gate.get("status") != "pending" or gate.get("approval") is not None):
            raise RuntimeError("Roundwatch prepared gate does not bind the exact queue")
        return gate


_FIXTURE_AUTHORITY_CAPABILITY = object()


class _FixtureRoundwatchAuthority:
    """Explicit CUDA-hidden test double; never constructed by production CLI."""

    fixture_only = True

    def __init__(self, *, capability, event_id: int = 1000):
        if capability is not _FIXTURE_AUTHORITY_CAPABILITY:
            raise RuntimeError("fixture Roundwatch authority capability is invalid")
        self.event_id = int(event_id)
        self.gate_id = 1
        self.instance_id = "round0005-fixture-instance"
        self.prepared: dict[str, Any] | None = None

    def prepare(self, *, manifest: dict[str, Any], manifest_path: str,
                manifest_sha256: str) -> dict[str, Any]:
        if sha256_file(manifest_path) != manifest_sha256:
            raise RuntimeError("fixture gate preparation saw a changed manifest")
        gate = {
            "id": self.gate_id, "program": manifest["program"],
            "round_id": manifest["round_id"], "round_sha": manifest["round_sha256"],
            "release_sha": manifest["release_sha"],
            "env_sha": manifest["environment_freeze_sha"],
            "env_identity_sha": manifest["environment_identity_sha"],
            "reviews_sha": sha256_bytes(canonical_json([])),
            "queue_manifest_path": os.path.realpath(manifest_path),
            "queue_manifest_sha": manifest_sha256, "gpu_hours": manifest["gpu_hours_cap"],
            "created_at": "2098-01-01T00:00:00+00:00", "expires_at": time.time() + 3600,
            "status": "pending", "approval": None,
        }
        self.prepared = gate
        return copy.deepcopy(gate)

    def check(self, *, manifest: dict[str, Any], manifest_path: str,
              manifest_sha256: str) -> dict[str, Any]:
        if self.prepared is None:
            self.prepare(manifest=manifest, manifest_path=manifest_path,
                         manifest_sha256=manifest_sha256)
        gate = copy.deepcopy(self.prepared)
        service = {
            "approved": True, "round_id": manifest["round_id"],
            "release_sha": manifest["release_sha"],
            "env_sha": manifest["environment_freeze_sha"],
            "env_identity_sha": manifest["environment_identity_sha"],
            "authority": "planner-gpu",
            "required_reviews_receipt": {"sha256": gate["reviews_sha"], "reviews": []},
            "reasons": [], "warnings": [], "gate_request": gate,
            "run_repo": {"head": manifest["release_sha"], "dirty": False,
                         "detached": True},
            "checked_at": "2098-01-01T00:00:02+00:00",
        }
        prepared_event = {
            "id": self.event_id, "created_at": "2098-01-01T00:00:01+00:00",
            "type": "gate.prepared", "program": manifest["program"],
            "round_id": manifest["round_id"], "payload": copy.deepcopy(gate),
        }
        return {
            "schema": "round0005_fixture_gate_evidence.v1",
            "service": service,
            "event_identity": {
                "instance_id": self.instance_id,
                "latest_event_id": self.event_id,
                "round_event": {
                    "id": self.event_id - 1, "type": "round.new",
                    "program": manifest["program"], "round_id": manifest["round_id"],
                    "payload": {"fixture_only": True},
                },
                "gate_prepared_event": prepared_event,
                "control_event": prepared_event,
                "gate_id": gate["id"], "authority": "planner-gpu",
            },
            "manifest_sha256": manifest_sha256,
            "round_file_sha256": manifest["round_sha256"],
            "release_sha": manifest["release_sha"],
            "environment_freeze_sha": manifest["environment_freeze_sha"],
            "environment_identity_sha": manifest["environment_identity_sha"],
            "fixture_only": True,
        }


def _new_fixture_roundwatch_authority() -> _FixtureRoundwatchAuthority:
    return _FixtureRoundwatchAuthority(capability=_FIXTURE_AUTHORITY_CAPABILITY)
