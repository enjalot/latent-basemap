"""Round 0001 content-complete admission, status, render, and drift fixtures."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from basemap import evidence_status as es
from basemap.artifact_identity import (expected_input_signature,
                                       expected_input_signatures,
                                       ordered_array_sha256)
from basemap.cohort_metrics import retention_and_jaccard, validate_cohorts
from basemap.experiment_contract import validate_contract
from basemap.queue_admission import (CACHE_KEYS, ROUND0005_LEASE_PATH,
                                     QueueAdmission, validate_queue_manifest)
from basemap.release_preflight import verify_release
from basemap.run_controller import Job, run_jobs
from basemap.round0005_staging import SEMANTIC_NAMESPACE_SCHEMA
from experiments.render_fixed_comparisons import render
from experiments.score_complete_panel import load_sample_indices


REGISTRY = "experiments/evidence/status-registry.json"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return str(path)


def _write_ids(path, values):
    np.save(path, np.asarray(values, dtype=np.int64))
    return str(path)


def test_status_registry_accepts_only_reviewed_evidence_by_default():
    found = es.discover_evidence(REGISTRY)
    assert {item["evidence_id"] for item in found} == {
        "g1.stdcurve-decision", "kernel.2m-three-seed-backfill"
    }
    with pytest.raises(PermissionError, match="invalidated"):
        es.resolve_evidence(REGISTRY, "o1.prompted-projection")
    with pytest.raises(PermissionError, match="diagnostic-only"):
        es.resolve_evidence(REGISTRY, "o2.engineering-maps",
                            allowed_status=("provisional",))


def test_no_model_does_not_require_sample_indices_and_model_path_does(tmp_path):
    assert load_sample_indices(str(tmp_path), no_model=True) is None
    with pytest.raises(FileNotFoundError, match="held-out projection"):
        load_sample_indices(str(tmp_path), no_model=False)


def _contract_fixture(tmp_path):
    ids = _write_ids(tmp_path / "rows.npy", np.arange(20))
    config_a = _write_json(tmp_path / "a.json", {"train": {"updates": 10}, "arm": "raw"})
    config_b = _write_json(tmp_path / "b.json", {"train": {"updates": 10}, "arm": "prompted"})
    query = _write_ids(tmp_path / "queries.npy", np.arange(100, 105))
    score = _write_json(tmp_path / "score.json", {"ok": True})
    return {
        "schema": "basemap_experiment_contract.v1",
        "experiment": "tiny-o1",
        "arms": [
            {"id": "raw", "config": config_a, "row_ids": ids, "seed": 42, "updates": 10},
            {"id": "prompted", "config": config_b, "row_ids": ids, "seed": 42, "updates": 10},
        ],
        "comparisons": [{
            "id": "prompt-only", "arms": ["raw", "prompted"],
            "allowed_config_differences": ["arm"],
            "required_config_differences": ["arm"],
        }],
        "jobs": [
            {"id": "embed", "produces": ["queries"], "consumes": []},
            {"id": "score", "produces": ["decision"], "consumes": ["queries"]},
        ],
        "artifacts": [
            {"id": "queries", "kind": "query", "path": query, "producer": "embed",
             "consumers": ["score"], "scientific": True,
             "metadata": {"prompt_bytes_hex": "446f63756d656e743a20", "pooling": "last_token"}},
            {"id": "decision", "kind": "decision", "path": score, "producer": "score",
             "consumers": [], "scientific": True, "terminal": True},
        ],
        "convention_checks": [{"artifact": "queries", "equals": {
            "prompt_bytes_hex": "446f63756d656e743a20", "pooling": "last_token"}}],
    }


def test_contract_fixture_consumes_query_and_enforces_allowlist(tmp_path):
    contract = _contract_fixture(tmp_path)
    report = validate_contract(contract, contract_path=str(tmp_path / "contract.json"))
    assert report["passed"], report
    contract["jobs"][1]["consumes"] = []
    contract["artifacts"][0]["consumers"] = []
    report = validate_contract(contract, contract_path=str(tmp_path / "contract.json"))
    assert not report["passed"]
    assert any("produced-but-unconsumed query" in error for error in report["errors"])


def test_contract_landmark_dose_and_cohort_overlap_fail_closed(tmp_path):
    contract = _contract_fixture(tmp_path)
    contract["landmarks"] = {
        "ids": _write_ids(tmp_path / "active.npy", np.arange(5)),
        "old_ids": _write_ids(tmp_path / "old.npy", np.arange(100)),
        "exact_count": 5,
        "dose_of_old": 0.05,
    }
    contract["cohorts"] = [
        {"id": "active", "ids": _write_ids(tmp_path / "ca.npy", np.arange(5))},
        {"id": "old_unpinned", "ids": _write_ids(tmp_path / "cu.npy", np.arange(5, 100))},
        {"id": "new", "ids": _write_ids(tmp_path / "cn.npy", np.arange(100, 200))},
    ]
    contract["cohort_universe"] = _write_ids(tmp_path / "universe.npy", np.arange(200))
    assert validate_contract(contract, contract_path=str(tmp_path / "c.json"))["passed"]
    _write_ids(tmp_path / "cn.npy", np.arange(99, 200))
    report = validate_contract(contract, contract_path=str(tmp_path / "c.json"))
    assert not report["passed"] and any("cohort overlap" in e for e in report["errors"])


@pytest.mark.parametrize("mutation,token", [
    ("config", "outside allow-list"),
    ("seed", "seed parity"),
    ("row_order", "row correspondence"),
    ("teacher_student", "teacher/student"),
    ("landmark_dose", "landmark dose"),
    ("convention", "convention pooling"),
    ("consumer_edge", "consumer edge mismatch"),
])
def test_contract_design_mutations_fail_closed(tmp_path, mutation, token):
    root = tmp_path / mutation
    root.mkdir()
    contract = _contract_fixture(root)
    if mutation == "config":
        _write_json(root / "b.json", {"train": {"updates": 11}, "arm": "prompted"})
    elif mutation == "seed":
        contract["arms"][1]["seed"] = 43
    elif mutation == "row_order":
        contract["arms"][1]["row_ids"] = _write_ids(root / "rows-b.npy", np.arange(20)[::-1])
    elif mutation == "teacher_student":
        contract["teacher_student_correspondence"] = {
            "teacher_ids": _write_ids(root / "teacher.npy", np.arange(20)),
            "student_old_ids": _write_ids(root / "student.npy", np.arange(1, 21)),
        }
    elif mutation == "landmark_dose":
        contract["landmarks"] = {
            "ids": _write_ids(root / "active.npy", np.arange(5)),
            "old_ids": _write_ids(root / "old.npy", np.arange(100)),
            "exact_count": 5, "dose_of_old": 0.04,
        }
    elif mutation == "convention":
        contract["artifacts"][0]["metadata"]["pooling"] = "mean"
    elif mutation == "consumer_edge":
        contract["artifacts"][0]["consumers"] = []
    report = validate_contract(contract, contract_path=str(root / "contract.json"))
    assert not report["passed"]
    assert any(token in error for error in report["errors"]), report


def test_selector_cohorts_self_exclusion_retention_and_true_jaccard():
    validate_cohorts({"old": np.array([0, 1]), "new": np.array([2, 3])},
                     universe_ids=np.arange(4))
    query_ids = np.array([0, 1])
    a = np.array([[0, 1, 2, 3], [1, 0, 2, 3]])
    b = np.array([[0, 1, 3, 4], [1, 0, 3, 4]])
    metrics = retention_and_jaccard(a, b, query_ids=query_ids, k=3)
    assert metrics["self_excluded"] is True
    assert metrics["retention_at_k"] == pytest.approx(2 / 3)
    assert metrics["true_jaccard_at_k"] == pytest.approx(1 / 2)


def test_fixed_render_uses_one_sample_and_identical_axes(tmp_path, fresh_data_root):
    n = 100
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pd.DataFrame({"x": np.arange(n), "y": np.arange(n), "ls_index": np.arange(n)}).to_parquet(p1)
    pd.DataFrame({"x": np.arange(n) * 2, "y": -np.arange(n), "ls_index": np.arange(n)}).to_parquet(p2)
    out = os.path.join(fresh_data_root, "round0001-render")
    ids = np.arange(n, dtype=np.int64)
    namespace = {
        "schema": SEMANTIC_NAMESPACE_SCHEMA,
        "name": "round0001/tiny-coordinate-row-id",
        "kind": "coordinate_semantic_id",
        "corpus_identity_sha256": "b" * 64,
        "universe_sha256": ordered_array_sha256(ids),
        "row_count": n,
    }
    spec_path = tmp_path / "render.json"
    spec = {"output_dir": out, "sample_size": 20, "comparisons": [{
        "id": "pair", "substrate": "tiny", "semantic_id_namespace": namespace,
        "maps": [
            {"label": "a", "coords": str(p1), "semantic_id_namespace": namespace},
            {"label": "b", "coords": str(p2),
             "semantic_id_namespace": namespace}]}]}
    _write_json(spec_path, spec)
    manifest = render(spec, spec_path=str(spec_path))
    entry = manifest["comparisons"][0]
    assert entry["sample_count"] == 20
    assert entry["axis_policy"] == "shared union extent; no per-map normalization"
    assert len({entry["sample_ids_sha256"]}) == 1
    assert os.path.isfile(entry["image"]["path"])


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def test_release_preflight_requires_ancestor_pushed_clean_detached(tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    remote = tmp_path / "remote.git"
    subprocess.check_call(["git", "init", "--bare", "-q", str(remote)])
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email", "test@example.com"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
    (repo / "x").write_text("x")
    subprocess.check_call(["git", "-C", str(repo), "add", "x"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "base"])
    release = _git(repo, "rev-parse", "HEAD")
    pushed_ref = "refs/remotes/origin/main"
    subprocess.check_call(["git", "-C", str(repo), "remote", "add", "origin",
                           str(remote)])
    subprocess.check_call(["git", "-C", str(repo), "push", "-q", "origin",
                           f"{release}:refs/heads/main"])
    subprocess.check_call(["git", "-C", str(repo), "fetch", "-q", "origin",
                           "+refs/heads/main:refs/remotes/origin/main"])
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", "-q", release])
    venv = tmp_path / "venv"; (venv / "bin").mkdir(parents=True)
    os.symlink(sys.executable, venv / "bin" / "python")
    freeze = tmp_path / "freeze.txt"; freeze.write_text("b==2\na==1\n")
    from basemap.release_preflight import _canonical_freeze_sha
    env = _write_json(tmp_path / "env.json", {
        "freeze_file": str(freeze), "freeze_sha256": _canonical_freeze_sha(str(freeze)),
        "identity_sha256": "f" * 64, "venv_path": str(venv)})
    cache = {"PYTHONDONTWRITEBYTECODE": "1", **{
        key: f"/data/latent-basemap/runs/round-0001/test-cache/{key}"
        for key in CACHE_KEYS}}
    report = verify_release(integration_repo=str(repo), release_sha=release,
                            implementation_commits=[release], pushed_ref=pushed_ref,
                            run_checkout=str(repo), environment_manifest=env,
                            cache_environment=cache)
    assert report["passed"], report
    (repo / "x").write_text("dirty")
    assert not verify_release(integration_repo=str(repo), release_sha=release,
                              implementation_commits=[release], pushed_ref=pushed_ref,
                              run_checkout=str(repo), environment_manifest=env,
                              cache_environment=cache)["passed"]


def _strict_queue_fixture(tmp_path, fresh_data_root, suffix):
    repo = tmp_path / f"repo-{suffix}"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email",
                           "test@example.com"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
    (repo / "tracked").write_text("tracked")
    subprocess.check_call(["git", "-C", str(repo), "add", "tracked"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "base"])
    release = _git(repo, "rev-parse", "HEAD")
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", "-q",
                           release])

    runtime = os.path.join(fresh_data_root, f"strict-{suffix}")
    os.mkdir(runtime)
    freeze = os.path.join(runtime, "freeze.txt")
    with open(freeze, "w", encoding="utf-8") as handle:
        handle.write("b==2\na==1\n")
    from basemap.release_preflight import _canonical_freeze_sha
    freeze_sha = _canonical_freeze_sha(freeze)
    env_path = os.path.join(runtime, "env.json")
    with open(env_path, "w", encoding="utf-8") as handle:
        json.dump({"freeze_file": freeze, "freeze_sha256": freeze_sha,
                   "identity_sha256": "f" * 64,
                   "venv_path": os.path.dirname(os.path.dirname(sys.executable))}, handle)
    external_input = os.path.join(runtime, "input.json")
    with open(external_input, "w", encoding="utf-8") as handle:
        json.dump({"version": 1}, handle)
    cache = {"PYTHONDONTWRITEBYTECODE": "1"}
    for key in CACHE_KEYS:
        value = os.path.join(runtime, "cache", key.lower())
        os.makedirs(value)
        cache[key] = value
    receipts = os.path.join(runtime, "receipts"); os.mkdir(receipts)
    checkpoints = os.path.join(runtime, "checkpoints"); os.mkdir(checkpoints)
    fixture = os.path.join(runtime, "fixture.json")
    with open(fixture, "w", encoding="utf-8") as handle:
        handle.write("{}\n")
    inputs = [external_input, "tracked"]
    job = {
        "id": "canary", "argv": [sys.executable, "-c", "pass"],
        "inputs": inputs,
        "expected_inputs": expected_input_signatures(inputs, repo_root=str(repo)),
        "outputs": [os.path.join(runtime, "output.json")],
        "done_marker": os.path.join(runtime, "done.json"),
        "log": os.path.join(runtime, "job.log"),
        "manifest": os.path.join(runtime, "job.json"), "cwd": str(repo),
        "required_free_gb": 0, "predicted_wall_s": 1, "p90_wall_s": 2,
        "deps": [], "continue_on_failure": False, "certifying": True,
        "canary_dep": None, "require_passing_verdict": None,
        "scientific_rows": 0, "performance_gate_path": None,
        "scale_policy": {"schema": "round0005_scale_policy.v1",
                         "scientific_rows": 0, "row_evidence": None,
                         "certificate": None},
    }
    manifest = {
        "schema_version": 1, "program": "basemap-100m", "round_id": "0005",
        "round_sha256": "a" * 64, "release_sha": release,
        "environment_freeze_sha": freeze_sha,
        "environment_identity_sha": "f" * 64, "gpu_hours_cap": 0.75,
        "queue_class": "research", "training_performed": False,
        "deadline_utc": "2099-01-01T00:00:00+00:00",
        "environment_manifest": env_path, "cache_environment": cache,
        "child_environment": {
            **cache, "CUDA_VISIBLE_DEVICES": "GPU-test", "PATH": "/usr/bin:/bin",
            "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "gate_receipts_dir": receipts,
        "controller_checkpoints_dir": checkpoints,
        "controller_terminal_summary": os.path.join(runtime, "terminal.json"),
        "repo_root": str(repo), "lease_path": ROUND0005_LEASE_PATH,
        "allowed_processes": [], "jobs": [job],
        "input_staging": {"maps_seal": {}, "model_seal": {}, "testbed_seal": {},
                          "data_closure_identity_sha256": "d" * 64,
                          "model_revision": "1" * 40},
        "fixture_identity": {"schema": "round0005_all_node_fixture.v2",
                             "canonical_path": fixture, "sha256": "e" * 64,
                             "identity_sha256": "b" * 64},
    }
    return repo, runtime, external_input, os.path.join(runtime, "queue.json"), manifest


def test_queue_manifest_content_complete_schema(tmp_path, fresh_data_root):
    _, _, _, path, data = _strict_queue_fixture(
        tmp_path, fresh_data_root, "schema")
    sentinel = data["jobs"][0]["outputs"][0]
    with pytest.raises(ValueError, match="fields mismatch"):
        validate_queue_manifest(data, path)
    assert not os.path.lexists(sentinel)


@pytest.mark.parametrize("mutation,token", [
    ("manifest", "manifest changed"),
    ("environment", "environment manifest"),
    ("input", "differs from gate-bound expectation"),
    ("release", "release SHA"),
])
def test_queue_admission_rejects_runtime_drift_before_roundwatch(
        tmp_path, fresh_data_root, monkeypatch, mutation, token):
    repo, runtime, external_input, manifest_path, manifest = _strict_queue_fixture(
        tmp_path, fresh_data_root, mutation)
    env_path = manifest["environment_manifest"]
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    called = False
    def guarded(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("Roundwatch must not be called after local drift")
    monkeypatch.setattr(
        "basemap.roundwatch_gate.RoundwatchGateAuthority.check", guarded)
    # Superseded arbitrary queues now fail at exact-program admission, before
    # any mutation-specific runtime path or Roundwatch contact is possible.
    with pytest.raises(ValueError, match="fields mismatch"):
        QueueAdmission(manifest_path, str(repo))
    assert called is False


@pytest.mark.parametrize("reason", [
    "service missing", "program paused", "gate expired", "manifest drift",
    "review drift", "environment drift", "release drift", "owner approval absent",
    "checkout drift",
])
def test_controller_boundary_rejection_stops_sentinel(
        fresh_data_root, reason):
    root = os.path.join(fresh_data_root, f"boundary-{reason.replace(' ', '-')}")
    os.mkdir(root)
    checkpoints = os.path.join(root, "checkpoints"); os.mkdir(checkpoints)
    receipts = os.path.join(root, "receipts"); os.mkdir(receipts)
    source = os.path.join(root, "input.json")
    with open(source, "w", encoding="utf-8") as handle:
        handle.write("{}\n")
    sentinel = os.path.join(root, "sentinel")
    job = Job(
        "sentinel",
        [sys.executable, "-c", f"open({sentinel!r}, 'x').write('launched')"],
        [sentinel], os.path.join(root, "done"), cwd=root,
        log=os.path.join(root, "job.log"),
        manifest=os.path.join(root, "job.manifest.json"),
        input_paths=[source], expected_inputs=[expected_input_signature(source)],
        predicted_wall_s=1, p90_wall_s=2,
        scale_policy={"schema": "round0005_scale_policy.v1",
                      "scientific_rows": 0, "row_evidence": None,
                      "certificate": None})
    manifest = {
        "release_sha": "a" * 40,
        "deadline_utc": (datetime.now(timezone.utc) +
                         timedelta(minutes=5)).isoformat(),
        "gpu_hours_cap": 0.75,
        "lease_path": os.path.join(root, "temporary-fixture.lease"),
        "controller_checkpoints_dir": checkpoints,
        "controller_terminal_summary": os.path.join(root, "terminal.json"),
        "gate_receipts_dir": receipts,
        "child_environment": {"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin",
                              "PYTHONDONTWRITEBYTECODE": "1",
                              "PYTHONPYCACHEPREFIX": os.path.join(root, "pycache")},
        "allowed_processes": [],
    }
    with pytest.raises(RuntimeError, match="requires an exact Round 0005 QueueAdmission"):
        run_jobs([job])
    assert not os.path.exists(sentinel)
    assert not any("launch" in name for name in os.listdir(checkpoints))
