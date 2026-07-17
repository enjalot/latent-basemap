"""Adversarial Round 0005 release/gate/admission/controller integration tests."""
from __future__ import annotations

import copy
import datetime
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.gate_preparation import validate_gate_preparation_receipt
from basemap.output_safety import (atomic_write_new_json, canonical_data_path,
                                   create_fresh_directory)
from basemap.queue_admission import (MUTATION_WINDOWS, QueueAdmission,
                                     validate_mutation_window_receipt)
from basemap.release_preflight import (_canonical_freeze_sha,
                                       validate_release_preflight_receipt,
                                       verify_release)
from basemap.round0005_fixture import (SIX_NODE_IDS, validate_fixture_queue)
from basemap.roundwatch_gate import (_validate_gate_payload,
                                     canonical_roundwatch_binding)
from basemap.run_controller import (Job, RuntimeGpuAccountant,
                                    RuntimeGpuViolation,
                                    _build_round0005_child_contract,
                                    _cache_policy_from_comprehensive_boundary,
                                    _require_budget_fit,
                                    _run_admitted_queue_fixture_only, GpuLease,
                                    gpu_snapshot, require_active_lease, run_jobs)
from basemap.source_closure import (ROUND0005_REQUIRED_DYNAMIC_SOURCES,
                                    ROUND0005_RUNTIME_ENTRYPOINTS,
                                    runtime_source_closure,
                                    source_closure_receipt,
                                    validate_source_closure_receipt)
from experiments.run_round0005_fixture import (_mutation_integrations,
                                                _prepared_admission,
                                                _publish_case)


ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def clean_fixture_release(tmp_path, fresh_data_root):
    repo = tmp_path / "clean-release"
    shutil.copytree(
        ROOT, repo, symlinks=True,
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc"))
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email",
                           "fixture@example.invalid"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Fixture"])
    subprocess.check_call(["git", "-C", str(repo), "add", "-f", "-A"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "fixture release"])
    release = _git(repo, "rev-parse", "HEAD")
    remote = tmp_path / "clean-release-remote.git"
    subprocess.check_call(["git", "init", "--bare", "-q", str(remote)])
    subprocess.check_call(["git", "-C", str(repo), "remote", "add", "origin", str(remote)])
    subprocess.check_call([
        "git", "-C", str(repo), "push", "-q", "origin", f"{release}:refs/heads/main"])
    subprocess.check_call(["git", "-C", str(repo), "fetch", "-q", "origin", "main"])
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", "-q", release])

    evidence_root = Path(fresh_data_root) / "fixture-release"
    evidence_root.mkdir()
    freeze = evidence_root / "freeze.txt"
    freeze.write_text("fixture-package==1\n")
    environment = evidence_root / "environment.json"
    environment.write_text(json.dumps({
        "freeze_file": str(freeze),
        "freeze_sha256": _canonical_freeze_sha(str(freeze)),
        "identity_sha256": "e" * 64,
        "venv_path": sys.prefix,
    }) + "\n")
    args = SimpleNamespace(
        repo_root=str(repo), integration_repo=str(repo), release_sha=release,
        implementation_commit=[release], pushed_ref="refs/remotes/origin/main",
        round_sha256="a" * 64, environment_manifest=str(environment),
    )
    return args, evidence_root


def test_fixture_manifest_is_exact_and_every_runtime_field_is_bound(
        clean_fixture_release):
    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "exact"), label="exact fixture parent")
    manifest, path, _ = _publish_case(parent=parent, label="queue", args=args)
    validate_fixture_queue(manifest, path)
    assert [job["id"] for job in manifest["jobs"]] == list(SIX_NODE_IDS)
    for field in ("argv", "deps", "node_policy", "expected_inputs", "p90_wall_s"):
        changed = copy.deepcopy(manifest)
        if field in {"argv", "deps", "expected_inputs"}:
            changed["jobs"][0][field] = ["lie"]
        elif field == "node_policy":
            changed["jobs"][0][field]["scientific_rows"] = 0
        else:
            changed["jobs"][0][field] = 0.0
        with pytest.raises(ValueError, match="exact derived contract"):
            validate_fixture_queue(changed, path)


def test_four_windows_use_real_components_and_automatic_receipts(
        clean_fixture_release):
    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "windows"), label="mutation parent")
    receipts = _mutation_integrations(parent, args)
    assert [entry["window"] for entry in receipts] == list(MUTATION_WINDOWS)
    for entry in receipts:
        payload = validate_mutation_window_receipt(
            entry["receipt"], expected_window=entry["window"])
        assert expected_input_signature(entry["receipt"]) == entry["signature"]
        assert payload["sentinel_argv"]
        assert payload["sentinel_output"] == payload["sentinel_argv"][
            payload["sentinel_argv"].index("--out") + 1]
        assert payload["child_pid"] is None


def test_real_six_node_fixture_has_exactly_six_popen_children_and_telemetry(
        clean_fixture_release):
    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "success"), label="success parent")
    manifest, _path, _input, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    result = _run_admitted_queue_fixture_only(
        admission=admission, telemetry_interval_s=0.01)
    assert result["terminal_verdict"] == "passed", result
    assert result["required_jobs"] == list(SIX_NODE_IDS)
    pids = [entry["child_pid"] for entry in result["jobs"]]
    assert len(pids) == len(set(pids)) == 6
    for entry in result["jobs"]:
        assert entry["status"] == "ok"
        assert Path(entry["output_signatures"].popitem()[0]).exists()
    checkpoints = Path(manifest["controller_checkpoints_dir"])
    telemetry = list(checkpoints.glob("*gpu-telemetry*.json"))
    assert telemetry
    assert all((value.stat().st_mode & 0o777) == 0o444 for value in telemetry)


def test_launch_edge_and_post_child_hooks_recompare_comprehensive_state(
        clean_fixture_release):
    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "hooks"), label="hooks parent")
    manifest, _path, fixture_input, admission = _prepared_admission(
        parent=parent, label="launch", args=args)
    Path(fixture_input).chmod(0o644)
    launch = _run_admitted_queue_fixture_only(
        admission=admission,
        launch_edge_hook=lambda _job: Path(fixture_input).write_text('{"drift":1}\n'))
    assert launch["terminal_verdict"] == "failed"
    assert launch["jobs"][0].get("child_pid") is None
    assert not Path(manifest["jobs"][0]["outputs"][0]).exists()
    receipt = validate_mutation_window_receipt(
        launch["jobs"][0]["integrity_receipt"],
        expected_window="gate-response-to-Popen")
    assert receipt["expected"] != receipt["observed"]

    manifest, _path, fixture_input, admission = _prepared_admission(
        parent=parent, label="post-child", args=args)
    Path(fixture_input).chmod(0o644)
    post = _run_admitted_queue_fixture_only(
        admission=admission,
        post_child_integrity_hook=lambda _job: Path(fixture_input).write_text(
            '{"post_child_drift":1}\n'), telemetry_interval_s=0.01)
    assert post["terminal_verdict"] == "failed"
    first = post["jobs"][0]
    assert first["child_pid"] is not None
    assert Path(manifest["jobs"][0]["outputs"][0]).exists()
    assert not Path(manifest["jobs"][0]["done_marker"]).exists()
    payload = json.loads(Path(first["integrity_receipt"]).read_text())
    assert payload["phase"] == "post-child-integrity"
    assert payload["expected"] != payload["observed"]
    assert payload["child_pid"] == first["child_pid"]


def test_release_receipt_rejects_changed_remote_tip_and_changed_bytes(
        clean_fixture_release):
    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "release"), label="release parent")
    manifest, _path, _ = _publish_case(parent=parent, label="queue", args=args)
    receipt = manifest["release_preflight_receipt"]
    receipt_signature = next(
        value for value in manifest["global_input_registry"]
        if value["canonical_path"] == receipt)
    validation = {
        "expected_identity_sha256": manifest["release_preflight_identity"],
        "expected_signature": receipt_signature,
    }
    validate_release_preflight_receipt(receipt, **validation)
    repo = Path(args.integration_repo)
    subprocess.check_call(["git", "-C", str(repo), "switch", "-q", "--detach"])
    (repo / "tip-change").write_text("changed\n")
    subprocess.check_call(["git", "-C", str(repo), "add", "tip-change"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "tip change"])
    changed_tip = _git(repo, "rev-parse", "HEAD")
    subprocess.check_call([
        "git", "-C", str(repo), "update-ref", "refs/remotes/origin/main", changed_tip])
    with pytest.raises(RuntimeError, match="changed after publication"):
        validate_release_preflight_receipt(receipt, **validation)


def test_clean_unpushed_release_and_self_rehashed_receipt_are_rejected(
        clean_release_evidence):
    evidence = clean_release_evidence
    repo = Path(evidence["repo"])
    (repo / "unpushed.txt").write_text("local only\n")
    subprocess.check_call(["git", "-C", str(repo), "add", "unpushed.txt"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "local only"])
    unpushed = _git(repo, "rev-parse", "HEAD")
    recorded = evidence["receipt"]
    report = verify_release(
        integration_repo=str(repo), release_sha=unpushed,
        implementation_commits=[recorded["release_sha"]],
        pushed_ref=recorded["pushed_ref"], run_checkout=str(repo),
        environment_manifest=recorded["environment_manifest_path"],
        cache_environment=recorded["cache_environment"])
    assert not report["passed"]
    assert any("not reachable from pushed ref" in error for error in report["errors"])

    receipt_path = Path(evidence["receipt_path"])
    receipt_signature = expected_input_signature(str(receipt_path))
    changed = copy.deepcopy(recorded)
    from basemap.artifact_identity import canonical_json, sha256_bytes
    observation = changed["remote_observation"]
    observation["observed_at"] = datetime.datetime.now(
        datetime.timezone.utc).isoformat(timespec="microseconds")
    observation_body = {
        key: observation[key] for key in observation if key != "identity_sha256"}
    observation["identity_sha256"] = sha256_bytes(canonical_json(observation_body))
    body = {key: changed[key] for key in changed if key != "identity_sha256"}
    changed["identity_sha256"] = sha256_bytes(canonical_json(body))
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(changed) + "\n")
    receipt_path.chmod(0o444)
    with pytest.raises(RuntimeError, match="gate-hashed signature"):
        validate_release_preflight_receipt(
            str(receipt_path),
            expected_identity_sha256=recorded["identity_sha256"],
            expected_signature=receipt_signature)


def _release_kwargs(evidence):
    recorded = evidence["receipt"]
    return {
        "integration_repo": evidence["repo"],
        "release_sha": recorded["release_sha"],
        "implementation_commits": recorded["implementation_commits"],
        "pushed_ref": recorded["pushed_ref"],
        "run_checkout": evidence["repo"],
        "environment_manifest": recorded["environment_manifest_path"],
        "cache_environment": recorded["cache_environment"],
    }


def test_remote_authority_rejects_missing_remote_url_drift_tip_drift_and_offline(
        clean_release_evidence, tmp_path):
    evidence = clean_release_evidence
    repo = Path(evidence["repo"])
    kwargs = _release_kwargs(evidence)

    subprocess.check_call(["git", "-C", str(repo), "remote", "remove", "origin"])
    missing = verify_release(**kwargs)
    assert not missing["passed"]
    assert any("server remote observation failed" in value for value in missing["errors"])

    original_remote = Path(evidence["receipt"]["remote_url"])
    alternate = tmp_path / "alternate.git"
    subprocess.check_call(["git", "clone", "--bare", "-q", str(original_remote),
                           str(alternate)])
    subprocess.check_call(["git", "-C", str(repo), "remote", "add", "origin",
                           str(alternate)])
    with pytest.raises(RuntimeError, match="changed after publication"):
        validate_release_preflight_receipt(
            evidence["receipt_path"],
            expected_identity_sha256=evidence["receipt"]["identity_sha256"],
            expected_signature=expected_input_signature(evidence["receipt_path"]))

    subprocess.check_call(["git", "-C", str(repo), "remote", "set-url", "origin",
                           str(original_remote)])
    subprocess.check_call(["git", "-C", str(repo), "switch", "-q", "-c", "server-drift"])
    (repo / "server-drift.txt").write_text("server drift\n")
    subprocess.check_call(["git", "-C", str(repo), "add", "server-drift.txt"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "server drift"])
    drift = _git(repo, "rev-parse", "HEAD")
    subprocess.check_call(["git", "-C", str(repo), "push", "-q", "origin",
                           f"{drift}:refs/heads/main"])
    subprocess.check_call(["git", "-C", str(repo), "update-ref",
                           "refs/remotes/origin/main", evidence["release_sha"]])
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", "-q",
                           evidence["release_sha"]])
    tip_drift = verify_release(**kwargs)
    assert not tip_drift["passed"]
    assert any("differs from observed server tip" in value for value in tip_drift["errors"])

    subprocess.check_call(["git", "-C", str(repo), "remote", "set-url", "origin",
                           str(tmp_path / "offline-missing.git")])
    offline = verify_release(**kwargs)
    assert not offline["passed"]
    assert any("remote observation failed closed" in value for value in offline["errors"])


def test_remote_authority_rejects_ambiguous_ls_remote_observation(
        clean_release_evidence, monkeypatch):
    import basemap.release_preflight as preflight

    original_run = preflight.subprocess.run

    def ambiguous(command, *args, **kwargs):
        if isinstance(command, list) and "ls-remote" in command:
            sha = clean_release_evidence["release_sha"]
            return subprocess.CompletedProcess(
                command, 0,
                f"{sha}\trefs/heads/main\n{sha}\trefs/heads/main\n", "")
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(preflight.subprocess, "run", ambiguous)
    report = verify_release(**_release_kwargs(clean_release_evidence))
    assert not report["passed"]
    assert any("absent/ambiguous" in value for value in report["errors"])


def test_minimal_roundwatch_approval_and_caller_selected_binary_fail_closed():
    with pytest.raises(RuntimeError, match="minimal/incomplete"):
        _validate_gate_payload(
            {"approved": True}, {}, manifest={"round_id": "0005"},
            manifest_path="/data/unused", manifest_sha256="a" * 64)
    with pytest.raises(TypeError):
        QueueAdmission("/data/unused", "/data/unused", roundwatch_bin="/tmp/fake")


def test_canonical_core_planner_pending_is_approved_but_owner_requires_approval(
        tmp_path):
    core_path = "/home/enjalot/code/workshop/rounds/core.py"
    spec = importlib.util.spec_from_file_location("roundwatch_issued_core_test", core_path)
    core = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    database = core.StateDB(tmp_path / "roundwatch.sqlite")
    queue = tmp_path / "queue.json"
    queue.write_text("{}\n")
    round_file = Path(
        "/home/enjalot/code/latent-labs/basemap-100m/round-0005-2026-07-16.md")
    round_signature = expected_input_signature(round_file)
    release = "1" * 40
    freeze = "2" * 64
    environment = "3" * 64
    reviews_sha = core.sha256_bytes(b"[]")

    def target(authority):
        return {
            "id": "0005", "authority": authority, "required_reviews": [],
            "contract_errors": [], "superseded_by": [],
            "estimate": {"maximum": 0.75},
            "round": {"sha256": round_signature["sha256"]},
        }

    planner = target("planner-gpu")
    database.event(
        "round.new", "basemap-100m", "0005",
        {"path": round_file.name, "sha256": round_signature["sha256"],
         "status": "issued"})
    gate = database.prepare_gate(
        "basemap-100m", "0005", planner["round"]["sha256"], release,
        freeze, environment, reviews_sha, str(queue), core.sha256_file(queue),
        0.75, core.time.time() + 3600)
    planner_result = core.evaluate_gate(
        "basemap-100m", planner, {"0005": planner}, database,
        release, freeze, environment, None)
    assert gate["status"] == "pending" and gate["approval"] is None
    assert planner_result["approved"] is True
    planner_result["run_repo"] = {
        "head": release, "dirty": False, "detached": True,
        "path": str(tmp_path),
    }
    manifest = {
        "program": "basemap-100m", "round_id": "0005",
        "execution_authority": "planner-gpu", "required_reviews": [],
        "round_sha256": round_signature["sha256"], "release_sha": release,
        "environment_freeze_sha": freeze,
        "environment_identity_sha": environment, "gpu_hours_cap": 0.75,
        "repo_root": str(tmp_path),
        "program_inputs": [{"role": "round_file", "signature": round_signature}],
    }
    events = database.events()
    canonical_evidence = _validate_gate_payload(
        planner_result,
        {"instance_id": "canonical-core-test",
         "latest": database.latest_event_id(), "events": events},
        manifest=manifest, manifest_path=str(queue),
        manifest_sha256=core.sha256_file(queue))
    assert canonical_evidence["event_identity"]["authority"] == "planner-gpu"
    assert canonical_evidence["event_identity"]["control_event"]["type"] == \
        "gate.prepared"

    owner = target("owner-gpu")
    owner_pending = core.evaluate_gate(
        "basemap-100m", owner, {"0005": owner}, database,
        release, freeze, environment, None)
    assert owner_pending["approved"] is False
    database.decide_gate(gate["id"], "approved", "test owner approval")
    owner_approved = core.evaluate_gate(
        "basemap-100m", owner, {"0005": owner}, database,
        release, freeze, environment, None)
    assert owner_approved["approved"] is True


def test_absolute_isolated_roundwatch_cli_ignores_ambient_startup_and_loader_env(
        tmp_path, monkeypatch):
    import basemap.roundwatch_gate as authority

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, payload):
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            requests.append(("GET", self.path))
            if self.path.startswith("/api/events"):
                self._reply({"instance_id": "isolated", "latest": 0, "events": []})
            else:
                self._reply({"approved": True, "source": "isolated"})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            requests.append(("POST", self.path, body))
            self._reply({"id": 7, "status": "pending", "approval": None})

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state = tmp_path / "state"
    state.mkdir()
    (state / "agent-token").write_text("isolated-token\n")
    hostile = tmp_path / "hostile-startup"
    hostile.write_text(f"touch {tmp_path / 'startup-ran'}\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "python3.12").write_text("#!/bin/sh\nexit 99\n")
    os.chmod(fake_bin / "python3.12", 0o755)
    fake_modules = tmp_path / "modules"
    fake_modules.mkdir()
    (fake_modules / "core.py").write_text("raise RuntimeError('ambient core loaded')\n")
    for key, value in {
        "BASH_ENV": str(hostile), "ENV": str(hostile), "PATH": str(fake_bin),
        "PYTHONPATH": str(fake_modules), "PYTHONHOME": str(tmp_path / "bad-home"),
        "LD_PRELOAD": str(tmp_path / "bad-preload.so"),
        "LD_AUDIT": str(tmp_path / "bad-audit.so"),
        "DYLD_INSERT_LIBRARIES": str(tmp_path / "bad-dylib"),
        "ROUNDWATCH_URL": "http://127.0.0.1:1",
        "ROUNDWATCH_STATE_DIR": str(tmp_path / "wrong-state"),
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(authority, "ROUNDWATCH_URL",
                        f"http://127.0.0.1:{server.server_port}")
    monkeypatch.setattr(authority, "ROUNDWATCH_STATE_DIR", str(state))
    try:
        binding = canonical_roundwatch_binding()
        assert authority.RoundwatchGateAuthority._run_json(
            ["events", "--after", "0"], expected_binding=binding)["events"] == []
        assert authority.RoundwatchGateAuthority._run_json(
            ["gate-check", "0005", "--program", "basemap-100m",
             "--release-sha", "1" * 40], expected_binding=binding)["approved"] is True
        prepared = authority.RoundwatchGateAuthority._run_json(
            ["prepare-gate", "0005", "--program", "basemap-100m",
             "--release-sha", "1" * 40, "--gpu-hours", "0.75",
             "--queue-manifest", "/data/isolated-queue.json"],
            expected_binding=binding)
        assert prepared["status"] == "pending"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not (tmp_path / "startup-ran").exists()
    assert {item[0] for item in requests} == {"GET", "POST"}


def test_gate_receipt_and_current_event_identity_reject_self_rehashed_drift(
        clean_fixture_release):
    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "gate-binding"), label="gate parent")
    manifest, path, _input, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    receipt_path = Path(manifest["gate_preparation_receipt"])
    receipt = json.loads(receipt_path.read_text())
    receipt["gate"]["env_identity_sha"] = "0" * 64
    body = {key: receipt[key] for key in receipt if key != "identity_sha256"}
    from basemap.artifact_identity import canonical_json, sha256_bytes
    receipt["identity_sha256"] = sha256_bytes(canonical_json(body))
    receipt_path.chmod(0o644)
    receipt_path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(RuntimeError, match="forged|bind"):
        validate_gate_preparation_receipt(
            str(receipt_path), manifest_path=path, manifest=manifest)

    manifest, _path, _input, admission = _prepared_admission(
        parent=parent, label="event", args=args)
    shadow_calls = []
    admission.gate_authority.check = lambda **_kwargs: shadow_calls.append(True) or {}
    admission.boundary(SIX_NODE_IDS[0])
    assert shadow_calls == []
    admission.gate_authority.event_id += 1
    with pytest.raises(RuntimeError, match="gate/event identity changed"):
        admission.boundary(SIX_NODE_IDS[1])


@pytest.mark.parametrize("control", ["not-yet-approved", "paused", "revoked"])
def test_actual_child_boundary_rechecks_live_control_before_capability_ack(
        monkeypatch, control):
    import basemap.gate_preparation as preparation
    import basemap.round0005_program as program
    import basemap.roundwatch_gate as authority
    import basemap.run_controller as controller

    monkeypatch.setattr(program, "validate_exact_program", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        preparation, "validate_gate_preparation_receipt",
        lambda *_args, **_kwargs: {})

    def reject(_self, **_kwargs):
        raise RuntimeError(f"live Roundwatch control is {control}")

    monkeypatch.setattr(authority.RoundwatchGateAuthority, "check", reject)
    manifest = {
        "repo_root": "/data/child-boundary-fixture",
        "gate_preparation_receipt": "/data/not-opened-after-live-rejection",
    }
    with pytest.raises(RuntimeError, match=control):
        controller._validate_child_live_authority(
            manifest, "/data/queue.json", "a" * 64,
            {"gate_identity": {}})


def test_actual_child_boundary_rejects_out_of_order_missing_predecessor_evidence():
    import basemap.run_controller as controller

    first = {
        "id": "first", "deps": [], "outputs": ["/data/missing-first-output"],
        "manifest": "/data/missing-first-controller.json",
        "done_marker": "/data/missing-first-done.json",
        "log": "/data/missing-first.log",
    }
    second = {
        "id": "second", "deps": ["first"], "outputs": ["/data/missing-second"],
        "manifest": "/data/missing-second-controller.json",
        "done_marker": "/data/missing-second-done.json",
        "log": "/data/missing-second.log",
    }
    response = {
        "job_index": 1, "node_id": "second", "deps": ["first"],
        "controller_id": "self-parent", "controller_pid": os.getpid(),
        "manifest_sha256": "a" * 64, "delegated": False,
        "delegated_ordinal": None,
        "controller_claim": {"identity_sha256": "c" * 64},
    }
    with pytest.raises(FileNotFoundError):
        controller._validate_completed_child_predecessors(
            {"jobs": [first, second]}, response)


def test_public_run_jobs_rejects_arbitrary_python_c_before_output(fresh_data_root):
    output = os.path.join(fresh_data_root, "must-not-run")
    job = Job("arbitrary", [sys.executable, "-c", f"open({output!r},'x').write('x')"],
              [output], os.path.join(fresh_data_root, "done"))
    with pytest.raises(RuntimeError, match="requires an exact Round 0005 QueueAdmission"):
        run_jobs([job])
    assert not os.path.lexists(output)


def test_public_controller_rejects_duck_subclass_partial_and_copied_admissions_before_popen(
        clean_fixture_release, monkeypatch):
    import basemap.run_controller as controller

    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "authenticity"), label="authenticity parent")
    manifest, _path, _input, authentic = _prepared_admission(
        parent=parent, label="queue", args=args)
    popen_calls = []
    monkeypatch.setattr(
        controller.subprocess, "Popen",
        lambda *values, **kwargs: popen_calls.append((values, kwargs)))

    class Duck:
        fixture_only = False

        def runtime_jobs(self):
            return []

        def assert_runtime_jobs(self, _jobs):
            return None

        def boundary(self, _job):
            return {}

        def comprehensive_integrity_boundary(self, *_args, **_kwargs):
            return {}

    with pytest.raises(RuntimeError, match="exact Round 0005 QueueAdmission"):
        run_jobs(admission=Duck())

    class Subclass(QueueAdmission):
        pass

    with pytest.raises(RuntimeError, match="subclasses"):
        Subclass("/data/unused", "/data/unused")

    partial = QueueAdmission.__new__(QueueAdmission)
    partial.fixture_only = False
    with pytest.raises(RuntimeError, match="copied or unauthentic"):
        run_jobs(admission=partial)

    with pytest.raises(RuntimeError, match="cannot be copied"):
        copy.copy(authentic)
    copied = QueueAdmission.__new__(QueueAdmission)
    copied.__dict__.update(authentic.__dict__)
    copied.fixture_only = False
    with pytest.raises(RuntimeError, match="copied or unauthentic"):
        run_jobs(admission=copied)
    assert popen_calls == []
    assert not any(Path(path).exists() for path in manifest["jobs"][0]["outputs"])


@pytest.mark.parametrize("mutation", ["rewrite", "unlink"])
def test_completed_predecessor_registry_detects_rewrite_or_unlink_during_next_child(
        clean_fixture_release, mutation):
    args, root = clean_fixture_release
    parent = create_fresh_directory(
        str(root / f"cumulative-{mutation}"), label="cumulative parent")
    manifest, _path, _input, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    predecessor = Path(manifest["jobs"][0]["outputs"][0])

    def mutate(job):
        if job.name != SIX_NODE_IDS[1]:
            return
        if mutation == "unlink":
            predecessor.unlink()
        else:
            predecessor.chmod(0o644)
            predecessor.write_text('{"rewritten":true}\n')

    result = _run_admitted_queue_fixture_only(
        admission=admission, post_child_integrity_hook=mutate,
        telemetry_interval_s=0.01)
    assert result["terminal_verdict"] == "failed"
    assert "predecessor" in result["stop_reason"] or "cumulative" in result["stop_reason"]
    assert not Path(manifest["controller_terminal_summary"]).read_text().find(
        '"terminal_verdict": "passed"') >= 0


def test_terminal_boundary_reopens_all_predecessors_after_sixth_done(
        clean_fixture_release):
    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "terminal-registry"),
                                    label="terminal registry parent")
    manifest, _path, _input, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    first_done = Path(manifest["jobs"][0]["done_marker"])

    def mutate(_job):
        first_done.chmod(0o644)
        first_done.write_text('{"forged":"after-sixth"}\n')

    result = _run_admitted_queue_fixture_only(
        admission=admission, terminal_integrity_hook=mutate,
        telemetry_interval_s=0.01)
    assert result["terminal_verdict"] == "failed"
    assert len(result["completed_jobs"]) == 6
    assert "cumulative" in result["stop_reason"] or "predecessor" in result["stop_reason"]


def test_injected_pycache_is_rejected_at_launch_edge_before_popen(
        clean_fixture_release):
    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "pycache-injection"),
                                    label="pycache injection parent")
    manifest, _path, _input, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    injected = Path(manifest["cache_environment"]["PYTHONPYCACHEPREFIX"]) / "evil.pyc"
    result = _run_admitted_queue_fixture_only(
        admission=admission,
        launch_edge_hook=lambda _job: injected.write_bytes(b"forged bytecode"))
    assert result["terminal_verdict"] == "failed"
    assert result["jobs"][0].get("child_pid") is None
    assert "PYTHONPYCACHEPREFIX" in result["stop_reason"]


@pytest.mark.parametrize("phase", ["post-child", "terminal"])
def test_transient_pycache_create_unlink_is_rejected_at_later_boundaries(
        clean_fixture_release, phase):
    args, root = clean_fixture_release
    parent = create_fresh_directory(
        str(root / f"pycache-transient-{phase}"),
        label="transient pycache parent")
    manifest, _path, _input, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    injected = Path(
        manifest["cache_environment"]["PYTHONPYCACHEPREFIX"]) / "removed.pyc"

    def create_then_unlink(_job):
        injected.write_bytes(b"transient forged bytecode")
        injected.unlink()

    kwargs = ({"post_child_integrity_hook": create_then_unlink}
              if phase == "post-child" else
              {"terminal_integrity_hook": create_then_unlink})
    result = _run_admitted_queue_fixture_only(
        admission=admission, telemetry_interval_s=0.01, **kwargs)
    assert result["terminal_verdict"] == "failed"
    assert "cache_policy" in result["stop_reason"] or \
        "PYTHONPYCACHEPREFIX" in result["stop_reason"]


def test_sealed_gpu_uuid_is_recomputed_with_absolute_closed_observer(monkeypatch):
    import basemap.queue_admission as queue
    from basemap.artifact_identity import canonical_json, sha256_bytes

    calls = []

    def observe(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0,
            "GPU-round0005-test, NVIDIA GeForce RTX 5090, fixture-driver\n", "")

    monkeypatch.setattr(queue.subprocess, "run", observe)
    environment = {
        "freeze_sha256": "f" * 64, "python": "3.12.3",
        "torch": "fixture+cu128", "torch_cuda": "12.8",
        "gpu_driver": "fixture-driver", "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": "GPU-round0005-test",
    }
    environment["identity_sha256"] = sha256_bytes(canonical_json(environment))
    result = queue.validate_canonical_gpu_environment(environment)
    assert result["sealed_gpu"]["gpu_uuid"] == "GPU-round0005-test"
    command, kwargs = calls[0]
    assert command[0] == "/usr/bin/nvidia-smi"
    assert kwargs["env"] == {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    forged = dict(environment)
    forged["gpu_uuid"] = "GPU-another-device"
    body = {key: forged[key] for key in environment if key != "identity_sha256"}
    forged["identity_sha256"] = sha256_bytes(canonical_json(body))
    with pytest.raises(RuntimeError, match="differs from live"):
        queue.validate_canonical_gpu_environment(forged)


def test_canonical_gpu_observer_rejects_multiple_rows_and_revalidates_binary(
        monkeypatch):
    import basemap.queue_admission as queue

    signatures = []
    real_signature = queue.expected_input_signature

    def signature(path):
        value = real_signature(path)
        signatures.append(value)
        return value

    monkeypatch.setattr(queue, "expected_input_signature", signature)
    monkeypatch.setattr(queue.subprocess, "run", lambda command, **kwargs:
        subprocess.CompletedProcess(
            command, 0,
            "GPU-a111, NVIDIA GeForce RTX 5090, driver\n"
            "GPU-b222, NVIDIA GeForce RTX 5090, driver\n", ""))
    with pytest.raises(RuntimeError, match="one registered GSV GPU"):
        queue._observe_canonical_gpu()
    assert len(signatures) == 2


def test_active_lease_returns_exact_single_verified_record(fresh_data_root):
    lease_path = os.path.join(fresh_data_root, "return-contract.lease")
    with GpuLease(path=lease_path, timeout=0,
                  controller_id="lease-return-test") as lease:
        expected = lease.verify_current()
        observed = require_active_lease(lease_path)
        assert observed == expected
        assert observed["token"] == lease.token


def test_production_cache_policy_comes_from_observed_comprehensive_boundary():
    policy = {"PYTHONPYCACHEPREFIX": {"path": "/data/cache", "inode": 7}}
    state = {"cache_policy": policy, "runtime": {"lease": {"token": "held"}}}
    boundary = {
        "phase": "child-capability-release", "expected": copy.deepcopy(state),
        "observed": copy.deepcopy(state), "integrity_match": True,
    }
    assert _cache_policy_from_comprehensive_boundary(boundary) == policy
    forged = copy.deepcopy(boundary)
    forged["observed"]["cache_policy"] = {}
    with pytest.raises(RuntimeError, match="boundary|cache policy"):
        _cache_policy_from_comprehensive_boundary(forged)


def test_nonfixture_contract_builder_consumes_observed_cache_policy():
    policy = {"PYTHONPYCACHEPREFIX": {"path": "/data/cache", "inode": 17}}
    boundary = {
        "phase": "child-capability-release",
        "expected": {"cache_policy": copy.deepcopy(policy)},
        "observed": {"cache_policy": copy.deepcopy(policy)},
        "integrity_match": True,
    }
    job = Job(
        "cached_nine_map", [sys.executable, str(ROOT / "fixture_child.py")],
        ["/data/out"], "/data/done", deps=[], p90_wall_s=10.0,
        gpu_memory_cap_mb=1024,
        node_policy={"canonical_script": "fixture_child.py"})
    contract = _build_round0005_child_contract(
        launch_nonce="a" * 64, controller_id="ctl-production-shape",
        child_pid=12345, job=job, jobs=[job], child_environment={"SEALED": "1"},
        manifest={"deadline_utc": "2099-01-01T00:00:00+00:00",
                  "gpu_hours_cap": 0.75, "repo_root": str(ROOT)},
        manifest_sha256="b" * 64, gate_identity={"gate": "current"},
        lease_identity={"token": "held"}, telemetry_interval_s=5.0,
        watchdog_pid=12346, watchdog_nonce="c" * 32,
        launch_integrity=boundary,
        controller_claim={"identity_sha256": "d" * 64})
    assert contract["cache_policy"] == policy
    assert contract["delegated"] is False
    assert contract["controller_claim"]["identity_sha256"] == "d" * 64


def test_owned_lease_fake_controller_claim_rejected_before_output_or_probe(
        fresh_data_root):
    import basemap.run_controller as controller
    from basemap.artifact_identity import canonical_json, sha256_bytes

    manifest_path = os.path.join(fresh_data_root, "fake-parent-queue.json")
    construction_path = os.path.join(fresh_data_root, "construction.json")
    environment_path = os.path.join(fresh_data_root, "environment.json")
    Path(construction_path).write_text('{"fixture":"construction"}\n')
    Path(environment_path).write_text(json.dumps({
        "venv_path": str(Path(sys.executable).resolve().parents[1]),
    }) + "\n")
    manifest = {
        "environment_manifest": environment_path,
        "jobs": [{"id": "fresh_uncached_2m"}],
    }
    Path(manifest_path).write_text(json.dumps(manifest) + "\n")
    process = controller._controller_process_record(os.getpid())
    body = {
        "schema": "round0005_queue_controller_claim.v1",
        "admission_id": 123, "admission_nonce": "a" * 64,
        "claim_nonce": "b" * 64, "controller_id": "fake-owned-parent",
        "controller_pid": os.getpid(),
        "controller_starttime_ticks": process["proc_starttime_ticks"],
        "controller_process": process, "fixture_only": False,
        "manifest": expected_input_signature(manifest_path),
        "construction_receipt": expected_input_signature(construction_path),
        "ordered_job_ids": ["fresh_uncached_2m"],
        "entry_gate_sha256": "c" * 64,
    }
    claim = {**body, "identity_sha256": sha256_bytes(canonical_json(body))}
    response = {
        "controller_claim": claim, "controller_id": "fake-owned-parent",
        "controller_pid": os.getpid(),
        "controller_starttime_ticks": process["proc_starttime_ticks"],
    }
    output = os.path.join(fresh_data_root, "must-not-exist")
    lease_path = os.path.join(fresh_data_root, "fake-parent.lease")
    with GpuLease(path=lease_path, timeout=0, controller_id="fake-owned-parent"):
        with pytest.raises(RuntimeError, match="canonical queue controller CLI"):
            controller._validate_controller_claim(manifest, manifest_path, response)
    assert not os.path.lexists(output)


def test_runtime_gpu_snapshot_uses_absolute_closed_observer_and_manifest_uuid(
        fresh_data_root, monkeypatch):
    import basemap.run_controller as controller

    environment_path = os.path.join(fresh_data_root, "runtime-gpu-environment.json")
    Path(environment_path).write_text(json.dumps({
        "gpu_uuid": "GPU-round0005-test",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_driver": "fixture-driver",
    }) + "\n")
    manifest = {
        "environment_manifest": environment_path,
        "child_environment": {"CUDA_VISIBLE_DEVICES": "GPU-round0005-test"},
    }
    calls = []
    signature = {"canonical_path": "/usr/bin/nvidia-smi", "kind": "file",
                 "bytes": 1, "sha256": "e" * 64}
    monkeypatch.setattr(controller, "expected_input_signature", lambda path: signature)

    def observe(command, **kwargs):
        calls.append((command, kwargs))
        if command[1].startswith("--query-gpu"):
            return ("GPU-round0005-test, NVIDIA GeForce RTX 5090, fixture-driver, "
                    "30000, 2000, 32000, 5, 100\n")
        return "GPU-round0005-test, 4242, 256\n"

    monkeypatch.setattr(controller.subprocess, "check_output", observe)
    monkeypatch.setenv("PATH", os.path.join(fresh_data_root, "hostile-bin"))
    monkeypatch.setenv("LD_PRELOAD", os.path.join(fresh_data_root, "hostile.so"))
    snapshot = gpu_snapshot(strict=True, manifest=manifest)
    assert snapshot["gpu_uuid"] == "GPU-round0005-test"
    assert snapshot["compute_app_records"] == [{
        "gpu_uuid": "GPU-round0005-test", "pid": 4242,
        "used_memory_mb": 256.0,
    }]
    assert all(command[0] == "/usr/bin/nvidia-smi" for command, _ in calls)
    assert all(kwargs["env"] == {"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
               for _, kwargs in calls)


def test_watchdog_emergency_verdict_cannot_coexist_with_job_or_terminal_success(
        clean_fixture_release, monkeypatch):
    import basemap.run_controller as controller

    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "watchdog-emergency"),
                                    label="watchdog emergency parent")
    manifest, _path, _input, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    real_stop = controller._stop_watchdog

    def emergency(handle, *, require_clean=True):
        verdict = real_stop(handle, require_clean=False)
        verdict["status"] = "emergency"
        verdict["error"] = "authenticated fixture emergency"
        body = {key: verdict[key] for key in verdict if key != "identity_sha256"}
        from basemap.artifact_identity import canonical_json, sha256_bytes
        verdict["identity_sha256"] = sha256_bytes(canonical_json(body))
        if require_clean:
            return verdict
        return verdict

    monkeypatch.setattr(controller, "_stop_watchdog", emergency)
    result = _run_admitted_queue_fixture_only(
        admission=admission, telemetry_interval_s=0.01)
    assert result["terminal_verdict"] == "failed"
    assert result["jobs"][0]["status"] != "ok"
    assert not Path(manifest["jobs"][0]["done_marker"]).exists()


@pytest.mark.parametrize("mode", ["clean", "controller-loss", "deadline"])
def test_exec_spawn_watchdog_with_background_thread_preserves_terminal_behavior(
        mode, fresh_data_root):
    import basemap.run_controller as controller

    checkpoint_root = create_fresh_directory(
        os.path.join(fresh_data_root, f"spawn-watchdog-{mode}"),
        label="spawn watchdog root")
    environment_path = os.path.join(fresh_data_root, f"watchdog-env-{mode}.json")
    Path(environment_path).write_text("{}\n")
    manifest = {
        "repo_root": str(ROOT), "allowed_processes": [],
        "environment_manifest": environment_path,
        "child_environment": dict(os.environ),
    }
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.1 if %r == 'clean' else 60)" % mode],
        start_new_session=True, env=dict(os.environ))
    job = Job(
        f"watchdog-{mode}", [sys.executable, "-c", "pass"], [],
        os.path.join(fresh_data_root, f"watchdog-{mode}.done"),
        node_policy={"gpu_required": False})
    stop = threading.Event()
    background = threading.Thread(target=lambda: stop.wait(30), daemon=True)
    background.start()
    lease_path = os.path.join(fresh_data_root, f"watchdog-{mode}.lease")
    handle = None
    try:
        with GpuLease(path=lease_path, timeout=0,
                      controller_id=f"watchdog-{mode}") as lease:
            now = time.time()
            read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
            handle = controller._start_watchdog(
                read_fd=read_fd, write_fd=write_fd,
                checkpoint_root=checkpoint_root, manifest=manifest, job=job,
                child_pid=child.pid, lease=lease,
                deadline_epoch=(now + 0.2 if mode == "deadline" else now + 10),
                runtime_deadline_epoch=now + 10,
                gpu_cap_deadline_epoch=now + 10,
                result_nonce="f" * 32,
                controller_claim_sha256="e" * 64)
            assert background.is_alive()
            if mode == "clean":
                assert child.wait(timeout=5) == 0
                verdict = controller._stop_watchdog(handle)
            else:
                if mode == "controller-loss":
                    os.close(handle.control_fd)
                    handle.control_fd = -1
                child.wait(timeout=5)
                verdict = controller._stop_watchdog(handle, require_clean=False)
            handle = None
        assert verdict["status"] == ("clean" if mode == "clean" else "emergency")
        if mode != "clean":
            assert "controller process died" in verdict["error"] or \
                "deadline" in verdict["error"]
    finally:
        stop.set()
        background.join(timeout=2)
        if handle is not None:
            try:
                controller._stop_watchdog(handle, require_clean=False)
            except Exception:
                pass
        if child.poll() is None:
            os.killpg(child.pid, 9)
            child.wait(timeout=5)


class _Lease:
    def verify_current(self):
        return {"token": "fixture", "inode": 1}


def _gpu_snapshot(records, *, total=1000.0):
    return {
        "compute_app_records": records,
        "compute_pids": [value["pid"] for value in records],
        "free_mb": total - sum(value["used_memory_mb"] for value in records),
        "used_mb": sum(value["used_memory_mb"] for value in records),
        "total_mb": total, "at": "fixture", "gpu": "fixture",
        "compute_apps": [], "n_co_tenants": len(records),
    }


def test_runtime_gpu_accountant_rejects_unknown_over_budget_and_pid_drift(monkeypatch):
    import basemap.run_controller as controller

    tree = [{"pid": 100, "ppid": 1, "proc_starttime_ticks": 10,
             "cmdline_sha256": "a" * 64}]
    monkeypatch.setattr("basemap.run_controller._process_tree", lambda _pid: copy.deepcopy(tree))
    job = Job("gpu", ["fixture"], ["out"], "done", gpu_memory_cap_mb=100)
    accountant = RuntimeGpuAccountant(
        manifest={"allowed_processes": []}, job=job, root_pid=100, lease=_Lease())
    with pytest.raises(controller.RuntimeGpuViolation, match="unknown"):
        accountant.snapshot(supplied_gpu=_gpu_snapshot([
            {"pid": 999, "process_name": "intruder", "used_memory_mb": 1.0}]))
    with pytest.raises(controller.RuntimeGpuViolation, match="exceeds cap"):
        accountant.snapshot(supplied_gpu=_gpu_snapshot([
            {"pid": 100, "process_name": "child", "used_memory_mb": 101.0}]))
    accountant.snapshot(supplied_gpu=_gpu_snapshot([]))
    tree[0]["proc_starttime_ticks"] = 11
    with pytest.raises(controller.RuntimeGpuViolation, match="identity drift"):
        accountant.snapshot(supplied_gpu=_gpu_snapshot([]))

    tree[0]["proc_starttime_ticks"] = 10
    service = {
        "pid": 200, "proc_starttime_ticks": 20, "cmdline_sha256": "b" * 64,
        "service_identity": "ls-serve", "marker": "ls-serve",
        "gpu_memory_budget_mb": 600,
    }
    monkeypatch.setattr(
        "basemap.run_controller.validate_allowed_processes",
        lambda expected, *, snapshot, allow_pids: {
            "expected": expected, "observed": expected, "gpu_snapshot": snapshot})
    cumulative = RuntimeGpuAccountant(
        manifest={"allowed_processes": [service]},
        job=Job("gpu", ["fixture"], ["out"], "done", gpu_memory_cap_mb=600),
        root_pid=100, lease=_Lease())
    with pytest.raises(
            controller.RuntimeGpuViolation,
            match="reserved memory|cumulative allocated"):
        cumulative.snapshot(supplied_gpu=_gpu_snapshot([
            {"pid": 100, "process_name": "child", "used_memory_mb": 550.0},
            {"pid": 200, "process_name": "ls-serve", "used_memory_mb": 550.0},
        ], total=1000.0))


def test_registered_p90_must_fit_deadline_and_cumulative_gpu_hour_cap():
    now = datetime.datetime.now(datetime.timezone.utc)
    job = Job(
        "gpu", ["fixture"], ["out"], "done", required_free_gb=1.0,
        predicted_wall_s=10.0, p90_wall_s=20.0)
    with pytest.raises(RuntimeError, match="deadline blocks gpu"):
        _require_budget_fit(
            {"deadline_utc": (now + datetime.timedelta(seconds=2)).isoformat(),
             "gpu_hours_cap": 1.0},
            job, gpu_elapsed_s=0.0, phase="adversarial-deadline")
    with pytest.raises(RuntimeError, match="cumulative GPU-hour cap blocks gpu"):
        _require_budget_fit(
            {"deadline_utc": (now + datetime.timedelta(hours=1)).isoformat(),
             "gpu_hours_cap": 0.01},
            job, gpu_elapsed_s=20.0, phase="adversarial-cap")


def test_generated_source_closure_contains_required_dynamic_modules(clean_fixture_release):
    args, _ = clean_fixture_release
    closure = runtime_source_closure(args.repo_root)
    assert set(ROUND0005_REQUIRED_DYNAMIC_SOURCES) <= set(closure)
    assert "basemap/pumap/parametric_umap/models/mlp.py" in closure
    assert "basemap/pumap/parametric_umap/utils/data_prefetcher.py" in closure


def test_production_source_closure_rejects_reduced_reordered_extra_and_each_member(
        clean_fixture_release):
    args, _ = clean_fixture_release
    receipt = source_closure_receipt(args.repo_root, require_tracked=True)
    validate_source_closure_receipt(receipt, repo_root=args.repo_root)
    required = list(ROUND0005_RUNTIME_ENTRYPOINTS)
    reduced = tuple(value for value in required if value not in {
        "basemap/run_controller.py", "experiments/score_complete_panel.py",
        "experiments/round0005_performance_gate.py",
    })
    with pytest.raises(RuntimeError, match="must equal"):
        source_closure_receipt(
            args.repo_root, require_tracked=True, entrypoints=reduced)
    for position, member in enumerate(required):
        changed = copy.deepcopy(receipt)
        changed["entrypoints"].pop(position)
        with pytest.raises(RuntimeError, match="entrypoints differ"):
            validate_source_closure_receipt(changed, repo_root=args.repo_root)
        assert member not in changed["entrypoints"]
    reordered = copy.deepcopy(receipt)
    reordered["entrypoints"][0], reordered["entrypoints"][1] = (
        reordered["entrypoints"][1], reordered["entrypoints"][0])
    with pytest.raises(RuntimeError, match="entrypoints differ"):
        validate_source_closure_receipt(reordered, repo_root=args.repo_root)
    extra = copy.deepcopy(receipt)
    extra["entrypoints"].append("basemap/__init__.py")
    with pytest.raises(RuntimeError, match="entrypoints differ"):
        validate_source_closure_receipt(extra, repo_root=args.repo_root)


def test_approved_source_closure_mutation_blocks_launch_with_receipt(
        clean_fixture_release):
    args, root = clean_fixture_release
    parent = create_fresh_directory(str(root / "source-drift"), label="source parent")
    manifest, _path, _input, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    source = Path(args.repo_root) / "basemap/pumap/parametric_umap/models/mlp.py"
    source.write_text(source.read_text() + "\n# post-approval drift\n")
    result = _run_admitted_queue_fixture_only(admission=admission)
    assert result["terminal_verdict"] == "failed"
    first = result["jobs"][0]
    assert first.get("child_pid") is None
    assert not Path(manifest["jobs"][0]["outputs"][0]).exists()
    receipt = json.loads(Path(first["integrity_receipt"]).read_text())
    assert receipt["status"] == "rejected"
    assert receipt["sentinel_output_absent"] is True


def test_data_path_rejects_symlinked_parent_before_creation(fresh_data_root):
    root = Path(fresh_data_root) / "path"
    root.mkdir()
    target = root / "target"; target.mkdir()
    link = root / "link"; link.symlink_to(target, target_is_directory=True)
    destination = link / "new" / "output.json"
    with pytest.raises(ValueError, match="symlinked ancestor"):
        canonical_data_path(str(destination), leaf_may_exist=False)
    assert not (target / "new").exists()


def test_atomic_publication_rejects_parent_swap_and_never_writes_escape(
        fresh_data_root, monkeypatch):
    import basemap.output_safety as safety

    root = Path(fresh_data_root) / "atomic-parent-swap"
    root.mkdir()
    parent = root / "parent"; parent.mkdir()
    held = root / "held"
    attacker = root / "attacker"; attacker.mkdir()
    original_open = safety._open_data_directory
    swapped = False

    def swap_before_open(path, *, create, mode=0o755):
        nonlocal swapped
        if os.path.abspath(path) == str(parent) and not swapped:
            parent.rename(held)
            parent.symlink_to(attacker, target_is_directory=True)
            swapped = True
        return original_open(path, create=create, mode=mode)

    monkeypatch.setattr(safety, "_open_data_directory", swap_before_open)
    destination = parent / "must-not-publish.json"
    with pytest.raises((OSError, ValueError)):
        atomic_write_new_json(str(destination), {"unsafe": True}, immutable=True)
    assert not (attacker / destination.name).exists()
    assert not (held / destination.name).exists()
