"""Round 0001 content-complete admission, status, render, and drift fixtures."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from basemap import evidence_status as es
from basemap.cohort_metrics import retention_and_jaccard, validate_cohorts
from basemap.experiment_contract import validate_contract
from basemap.queue_admission import QueueAdmission, validate_queue_manifest
from basemap.release_preflight import verify_release
from basemap.run_controller import Job, run_jobs
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


def test_fixed_render_uses_one_sample_and_identical_axes(tmp_path):
    n = 100
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pd.DataFrame({"x": np.arange(n), "y": np.arange(n), "ls_index": np.arange(n)}).to_parquet(p1)
    pd.DataFrame({"x": np.arange(n) * 2, "y": -np.arange(n), "ls_index": np.arange(n)}).to_parquet(p2)
    out = "/data/latent-basemap/runs/round-0001/test-render"
    os.makedirs(out, exist_ok=True)
    spec_path = tmp_path / "render.json"
    spec = {"output_dir": out, "sample_size": 20, "comparisons": [{
        "id": "pair", "substrate": "tiny", "maps": [
            {"label": "a", "coords": str(p1)}, {"label": "b", "coords": str(p2)}]}]}
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
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email", "test@example.com"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
    (repo / "x").write_text("x")
    subprocess.check_call(["git", "-C", str(repo), "add", "x"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "base"])
    release = _git(repo, "rev-parse", "HEAD")
    subprocess.check_call(["git", "-C", str(repo), "branch", "pushed", release])
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", "-q", release])
    venv = tmp_path / "venv"; venv.mkdir()
    freeze = tmp_path / "freeze.txt"; freeze.write_text("b==2\na==1\n")
    from basemap.release_preflight import _canonical_freeze_sha
    env = _write_json(tmp_path / "env.json", {
        "freeze_file": str(freeze), "freeze_sha256": _canonical_freeze_sha(str(freeze)),
        "identity_sha256": "f" * 64, "venv_path": str(venv)})
    cache = {"PYTHONDONTWRITEBYTECODE": "1", **{
        key: f"/data/latent-basemap/runs/round-0001/test-cache/{key}"
        for key in ("XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR",
                    "PYTHONPYCACHEPREFIX")}}
    report = verify_release(integration_repo=str(repo), release_sha=release,
                            implementation_commits=[release], pushed_ref="pushed",
                            run_checkout=str(repo), environment_manifest=env,
                            cache_environment=cache)
    assert report["passed"], report
    (repo / "x").write_text("dirty")
    assert not verify_release(integration_repo=str(repo), release_sha=release,
                              implementation_commits=[release], pushed_ref="pushed",
                              run_checkout=str(repo), environment_manifest=env,
                              cache_environment=cache)["passed"]


def test_queue_manifest_content_complete_schema(tmp_path):
    path = "/data/latent-basemap/runs/round-0001/test-queue.json"
    data = {"schema_version": 1, "program": "basemap-100m", "round_id": "0001",
            "release_sha": "a" * 40, "environment_freeze_sha": "b" * 64,
            "environment_identity_sha": "c" * 64, "gpu_hours_cap": 0.1,
            "environment_manifest": "/data/env.json",
            "repo_root": "/tmp/run", "allowed_pids": [],
            "cache_environment": {"PYTHONDONTWRITEBYTECODE": "1",
                                  **{key: f"/data/cache/{key}" for key in (
                                      "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME",
                                      "TRITON_CACHE_DIR", "PYTHONPYCACHEPREFIX",
                                      "NUMBA_CACHE_DIR", "MPLCONFIGDIR")}},
            "gate_receipts_dir": "/data/receipts",
            "jobs": [{"id": "canary", "argv": ["true"], "inputs": ["input"],
                      "outputs": ["/data/output"], "done_marker": "/data/done",
                      "log": "/data/log", "manifest": "/data/manifest",
                      "cwd": "/tmp/run", "required_free_gb": 1,
                      "predicted_wall_s": 30}]}
    validate_queue_manifest(data, path)
    data["jobs"] = []
    with pytest.raises(ValueError, match="nonempty"):
        validate_queue_manifest(data, path)


@pytest.mark.parametrize("mutation,token", [
    ("manifest", "manifest changed"),
    ("environment", "environment manifest"),
    ("input", "input changed"),
    ("release", "release SHA"),
])
def test_queue_admission_rejects_runtime_drift_before_roundwatch(tmp_path, monkeypatch,
                                                                mutation, token):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email", "test@example.com"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
    (repo / "tracked").write_text("tracked")
    subprocess.check_call(["git", "-C", str(repo), "add", "tracked"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "base"])
    release = _git(repo, "rev-parse", "HEAD")
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", "-q", release])

    runtime = os.path.join("/data/latent-basemap/runs/round-0001/pytest", tmp_path.name,
                           mutation)
    os.makedirs(runtime, exist_ok=True)
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
    cache = {"PYTHONDONTWRITEBYTECODE": "1", **{
        key: os.path.join(runtime, "cache", key) for key in (
            "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR",
            "PYTHONPYCACHEPREFIX", "NUMBA_CACHE_DIR", "MPLCONFIGDIR")}}
    manifest_path = os.path.join(runtime, "queue.json")
    manifest = {
        "schema_version": 1, "program": "basemap-100m", "round_id": "0001",
        "release_sha": release, "environment_freeze_sha": freeze_sha,
        "environment_identity_sha": "f" * 64, "gpu_hours_cap": 0.1,
        "environment_manifest": env_path, "cache_environment": cache,
        "gate_receipts_dir": os.path.join(runtime, "receipts"),
        "repo_root": str(repo), "allowed_pids": [],
        "jobs": [{"id": "canary", "argv": [sys.executable, "-c", "pass"],
                  "inputs": [external_input, "tracked"],
                  "outputs": [os.path.join(runtime, "output.json")],
                  "done_marker": os.path.join(runtime, "done.json"),
                  "log": os.path.join(runtime, "job.log"),
                  "manifest": os.path.join(runtime, "job.json"), "cwd": str(repo),
                  "required_free_gb": 0, "predicted_wall_s": 1}],
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    admission = QueueAdmission(manifest_path, str(repo))
    called = False
    real_run = subprocess.run
    def guarded(command, *args, **kwargs):
        nonlocal called
        if command and os.path.basename(command[0]) == "roundwatch":
            called = True
            raise AssertionError("Roundwatch must not be called after local drift")
        return real_run(command, *args, **kwargs)
    monkeypatch.setattr("basemap.queue_admission.subprocess.run", guarded)
    if mutation == "manifest":
        with open(manifest_path, "a", encoding="utf-8") as handle:
            handle.write("\n")
    elif mutation == "environment":
        with open(env_path, "r+", encoding="utf-8") as handle:
            value = json.load(handle); value["mutation"] = True
            handle.seek(0); json.dump(value, handle); handle.truncate()
    elif mutation == "input":
        with open(external_input, "w", encoding="utf-8") as handle:
            json.dump({"version": 2}, handle)
    elif mutation == "release":
        (repo / "other").write_text("other")
        subprocess.check_call(["git", "-C", str(repo), "add", "other"])
        subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "other"])
    with pytest.raises(RuntimeError, match=token):
        admission.boundary("canary")
    assert called is False


@pytest.mark.parametrize("reason", [
    "service missing", "program paused", "gate expired", "manifest drift",
    "review drift", "environment drift", "release drift", "owner approval absent",
    "checkout drift",
])
def test_controller_boundary_rejection_stops_sentinel(tmp_path, monkeypatch, reason):
    monkeypatch.setenv("BASEMAP_GPU_LEASE", str(tmp_path / "lease"))
    sentinel = tmp_path / "sentinel"
    class Reject:
        manifest_path = "/data/queue.json"; manifest_sha256 = "x"
        manifest = {"release_sha": "a" * 40, "cache_environment": {}}
        initial_inputs = {}
        def boundary(self, name):
            raise RuntimeError(reason)
    job = Job("sentinel", ["touch", str(sentinel)], [str(sentinel)], str(tmp_path / "done"))
    summary = run_jobs([job], admission=Reject())
    assert summary["jobs"][0]["status"] == "boundary_rejected"
    assert not sentinel.exists()
