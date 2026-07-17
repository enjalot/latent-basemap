"""Round 0005 adversarial admission, staging, scorer, and calibration fixtures."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       expected_input_signatures, ordered_array_sha256,
                                       path_signature, sha256_bytes)
from basemap.output_safety import atomic_write_new_json
from basemap.panel_v2 import (PanelV2Config, QueryTruthCache, build_hiD_reference,
                              sample_anchors, save_hiD_reference, score_panel)
from basemap.query_artifact import (_identity as query_identity, _normalization_proof,
                                    build_query_artifact, load_query_artifact)
from basemap.queue_admission import QueueAdmission, validate_queue_manifest
from basemap.release_preflight import _canonical_freeze_sha
from basemap.round0005_fixture import (SIX_NODE_IDS, validate_fixture_queue)
from basemap.round0005_staging import (
    MAP_EXPECTATIONS, ROUND0005_MODEL_REVISION, SEMANTIC_NAMESPACE_SCHEMA,
    stage_regular_model_snapshot, stage_round0005_maps, validate_staged_map_seal,
    validate_staged_model_seal,
)
from basemap.run_controller import (Job, _run_jobs_fixture_only,
                                    _run_admitted_queue_fixture_only,
                                    require_scale_performance_gate, run_jobs)
from experiments.build_prompted_graph import neighbor_overlap_report
from experiments.build_round0005_scorer_fixture import build_fixture as build_scorer_fixture
from experiments.calibrate_jina_embedding import certify_with_model, validate_inventory
from experiments.compare_panel_cache import load_fixture as load_scorer_fixture, run_equivalence
from experiments.embed_prompted_200k import embed_outer_chunks, embed_texts
from experiments.prepare_jina_calibration import prepare as prepare_calibration_inventory
from experiments.prepare_round0005_queue import (ROUND_SHA256 as ROUND0005_PLAN_SHA,
                                                  main as prepare_queue)
from experiments.render_fixed_comparisons import render
from experiments.round0005_performance_gate import evaluate_panel, run_synthetic_regression
from experiments.run_round0005_fixture import (
    _mutation_integrations, _prepared_admission, _publish_case,
    main as run_acceptance_fixture,
)
from experiments.score_complete_panel import score_query_bundle

TEST_REVISION = "0123456789abcdef0123456789abcdef01234567"
MUTATED_REVISION = "89abcdef0123456789abcdef0123456789abcdef"


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _queue_fixture(tmp_path, fresh_data_root, *, two_jobs=False):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-q", str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.email", "test@example.com"])
    subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
    (repo / "tracked").write_text("tracked\n")
    subprocess.check_call(["git", "-C", str(repo), "add", "tracked"])
    subprocess.check_call(["git", "-C", str(repo), "commit", "-qm", "base"])
    release = _git(repo, "rev-parse", "HEAD")
    subprocess.check_call(["git", "-C", str(repo), "checkout", "--detach", "-q", release])
    runtime = os.path.join(fresh_data_root, "queue")
    os.mkdir(runtime)
    freeze = os.path.join(runtime, "freeze.txt")
    Path(freeze).write_text("a==1\nb==2\n")
    freeze_sha = _canonical_freeze_sha(freeze)
    env_path = os.path.join(runtime, "env.json")
    Path(env_path).write_text(json.dumps({
        "freeze_file": freeze, "freeze_sha256": freeze_sha,
        "identity_sha256": "f" * 64,
        "venv_path": os.path.dirname(os.path.dirname(sys.executable)),
    }))
    external = os.path.join(runtime, "input.json")
    Path(external).write_text(json.dumps({"version": 1}))
    cache = {"PYTHONDONTWRITEBYTECODE": "1", **{
        key: os.path.join(runtime, "cache", key) for key in (
            "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "TRITON_CACHE_DIR",
            "PYTHONPYCACHEPREFIX", "NUMBA_CACHE_DIR", "MPLCONFIGDIR")}}
    for value in cache.values():
        if isinstance(value, str) and value.startswith("/data/"):
            os.makedirs(value, exist_ok=True)
    receipts = os.path.join(runtime, "receipts"); os.mkdir(receipts)
    checkpoints = os.path.join(runtime, "checkpoints"); os.mkdir(checkpoints)
    fixture_path = os.path.join(runtime, "fixture.json")
    Path(fixture_path).write_text("{}\n")

    def job(job_id, deps=()):
        inputs = [external, "tracked"]
        return {
            "id": job_id, "argv": [sys.executable, "-c", "pass"], "inputs": inputs,
            "expected_inputs": expected_input_signatures(inputs, repo_root=str(repo)),
            "outputs": [os.path.join(runtime, f"{job_id}.output")],
            "done_marker": os.path.join(runtime, f"{job_id}.done"),
            "log": os.path.join(runtime, f"{job_id}.log"),
            "manifest": os.path.join(runtime, f"{job_id}.job.json"),
            "cwd": str(repo), "required_free_gb": 0, "predicted_wall_s": 1,
            "p90_wall_s": 2, "deps": list(deps), "continue_on_failure": False,
            "certifying": True, "canary_dep": None,
            "require_passing_verdict": None, "scientific_rows": 0,
            "performance_gate_path": None,
            "scale_policy": {"schema": "round0005_scale_policy.v1",
                             "scientific_rows": 0, "row_evidence": None,
                             "certificate": None},
        }

    jobs = [job("seal")]
    if two_jobs:
        jobs.append(job("repeat", deps=("seal",)))
    manifest = {
        "schema_version": 1, "program": "basemap-100m", "round_id": "0005",
        "round_sha256": "a" * 64,
        "release_sha": release, "environment_freeze_sha": freeze_sha,
        "environment_identity_sha": "f" * 64, "gpu_hours_cap": 0.75,
        "queue_class": "research", "training_performed": False,
        "deadline_utc": "2099-01-01T00:00:00+00:00",
        "environment_manifest": env_path, "cache_environment": cache,
        "child_environment": {**cache, "CUDA_VISIBLE_DEVICES": "GPU-test",
                              "PATH": "/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
                              "PYTHONHASHSEED": "0", "TOKENIZERS_PARALLELISM": "false"},
        "gate_receipts_dir": receipts, "controller_checkpoints_dir": checkpoints,
        "controller_terminal_summary": os.path.join(runtime, "terminal.json"),
        "repo_root": str(repo), "lease_path": "/data/latent-basemap/.gpu_lease",
        "allowed_processes": [], "jobs": jobs,
        "input_staging": {"maps_seal": {}, "model_seal": {}, "testbed_seal": {},
                          "data_closure_identity_sha256": "d" * 64,
                          "model_revision": "1" * 40},
        "fixture_identity": {"schema": "round0005_all_node_fixture.v2",
                             "canonical_path": fixture_path, "sha256": "e" * 64,
                             "identity_sha256": "b" * 64},
    }
    manifest_path = os.path.join(runtime, "queue.json")
    return repo, runtime, external, manifest_path, manifest


def _write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest, indent=2))


def test_manifest_rejects_unsigned_extra_duplicate_missing_unsupported_and_consumes(
        round0005_clean_release):
    args, evidence = round0005_clean_release
    parent = os.path.join(evidence, "manifest-contracts")
    os.mkdir(parent)
    manifest, path, _ = _publish_case(parent=parent, label="queue", args=args)
    validate_fixture_queue(manifest, path)
    unsigned = copy.deepcopy(manifest)
    unsigned["jobs"][0]["expected_inputs"] = unsigned["jobs"][0]["expected_inputs"][:-1]
    with pytest.raises(ValueError, match="exact derived contract"):
        validate_fixture_queue(unsigned, path)
    extra = copy.deepcopy(manifest)
    extra["jobs"][0]["expected_inputs"] = list(extra["jobs"][0]["expected_inputs"])
    extra["jobs"][0]["expected_inputs"].append(
        copy.deepcopy(extra["jobs"][0]["expected_inputs"][0]))
    extra["jobs"][0]["expected_inputs"][-1]["canonical_path"] += ".extra"
    with pytest.raises(ValueError, match="exact derived contract"):
        validate_fixture_queue(extra, path)
    duplicate = copy.deepcopy(manifest)
    duplicate["jobs"][0]["expected_inputs"] = list(
        duplicate["jobs"][0]["expected_inputs"])
    duplicate["jobs"][0]["inputs"].append(duplicate["jobs"][0]["inputs"][0])
    duplicate["jobs"][0]["expected_inputs"].append(
        copy.deepcopy(duplicate["jobs"][0]["expected_inputs"][0]))
    with pytest.raises(ValueError, match="exact derived contract"):
        validate_fixture_queue(duplicate, path)
    missing = copy.deepcopy(manifest)
    missing["jobs"][0]["inputs"][0] += ".missing"
    with pytest.raises(ValueError, match="exact derived contract"):
        validate_fixture_queue(missing, path)
    unsupported = copy.deepcopy(manifest)
    unsupported["jobs"][0]["expected_inputs"] = copy.deepcopy(
        unsupported["jobs"][0]["expected_inputs"])
    unsupported["jobs"][0]["expected_inputs"][0]["kind"] = "socket"
    with pytest.raises(ValueError, match="exact derived contract"):
        validate_fixture_queue(unsupported, path)
    consumes = copy.deepcopy(manifest)
    consumes["jobs"][0]["consumes"] = []
    with pytest.raises(ValueError, match="fields differ"):
        validate_fixture_queue(consumes, path)


def test_manifest_global_equal_repeats_pass_and_unequal_repeats_fail(
        round0005_clean_release):
    args, evidence = round0005_clean_release
    parent = os.path.join(evidence, "global-registry")
    os.mkdir(parent)
    manifest, path, _ = _publish_case(parent=parent, label="queue", args=args)
    validate_fixture_queue(manifest, path)
    assert all(job["expected_inputs"] == manifest["global_input_registry"]
               for job in manifest["jobs"])
    conflict = copy.deepcopy(manifest)
    conflict["jobs"][1]["expected_inputs"] = copy.deepcopy(
        conflict["jobs"][1]["expected_inputs"])
    conflict["jobs"][1]["expected_inputs"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="exact derived contract"):
        validate_fixture_queue(conflict, path)


def test_admission_construction_failure_writes_atomic_full_receipt(
        round0005_clean_release):
    args, evidence = round0005_clean_release
    parent = os.path.join(evidence, "construction-window")
    os.mkdir(parent)
    receipts = _mutation_integrations(parent, args)
    entry = next(value for value in receipts
                 if value["window"] == "gate-preparation-to-admission")
    receipt_path = Path(entry["receipt"])
    assert oct(receipt_path.stat().st_mode & 0o777) == "0o444"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "rejected"
    assert receipt["expected"] != receipt["observed"]
    assert receipt["child_pid"] is None


def test_mutation_after_admission_before_boundary_fails_with_receipt(
        round0005_clean_release):
    args, evidence = round0005_clean_release
    parent = os.path.join(evidence, "boundary-window")
    os.mkdir(parent)
    manifest, path, external, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    Path(external).chmod(0o644)
    Path(external).write_text(json.dumps({"version": 2}) + "\n")
    summary = _run_admitted_queue_fixture_only(admission=admission)
    assert summary["terminal_verdict"] == "failed"
    first = summary["jobs"][0]
    assert first["status"] == "exception" and first.get("child_pid") is None
    receipt = json.loads(Path(first["integrity_receipt"]).read_text())
    assert receipt["status"] == "rejected"
    assert receipt["expected"] != receipt["observed"]


def test_launch_edge_mutation_stops_sentinel_and_comparison_is_adjacent_to_popen(
        round0005_clean_release):
    args, evidence = round0005_clean_release
    parent = os.path.join(evidence, "launch-edge")
    os.mkdir(parent)
    manifest, _path, external, admission = _prepared_admission(
        parent=parent, label="queue", args=args)
    Path(external).chmod(0o644)
    sentinel = manifest["jobs"][0]["outputs"][0]
    hook_ran = []

    def mutate(_job):
        hook_ran.append(True)
        Path(external).write_text(json.dumps({"version": 9}) + "\n")

    summary = _run_admitted_queue_fixture_only(
        admission=admission, launch_edge_hook=mutate)
    assert summary["terminal_verdict"] == "failed"
    assert summary["jobs"][0]["status"] == "exception"
    assert "integrity" in summary["jobs"][0]["error"].lower()
    assert hook_ran == [True]
    assert not os.path.exists(sentinel)
    assert summary["jobs"][0].get("child_pid") is None


def _map_source(root: Path, label: str, ids: np.ndarray):
    expectation = MAP_EXPECTATIONS[label]
    path = root / label
    path.mkdir(parents=True)
    pd.DataFrame({"x": ids.astype(float), "y": -ids.astype(float),
                  "ls_index": ids}).to_parquet(path / "coords.parquet")
    (path / "model.pt").write_bytes(f"model:{label}".encode())
    (path / "config.yaml").write_text(f"name: {label}\n")
    (path / "manifest.json").write_text(json.dumps({
        "low_dim_kernel": expectation["kernel"], "complete": True}))
    (path / "results.json").write_text(json.dumps({
        "data": {"n_train": len(ids)},
        "config": {"data": {"random_seed": expectation["seed"]},
                   "model": {"low_dim_kernel": expectation["kernel"],
                             "a": expectation["a"], "b": expectation["b"]}},
    }))
    (path / "extra-proof.txt").write_text("preserved whole directory\n")
    return str(path)


def test_nine_map_staging_validates_semantic_ids_copies_whole_dirs_and_refuses_reuse(
        tmp_path, fresh_data_root):
    ids = np.arange(8, dtype=np.int64)
    source_root = tmp_path / "map-sources"
    sources = {label: _map_source(source_root, label, ids)
               for label in MAP_EXPECTATIONS}
    corpus = tmp_path / "corpus.identity"
    corpus.write_bytes(b"corpus")
    destination = os.path.join(fresh_data_root, "inputs", "maps")
    seal = os.path.join(fresh_data_root, "inputs", "maps-seal.json")
    report = stage_round0005_maps(
        sources=sources, destination_root=destination, seal_path=seal,
        corpus_identity_path=str(corpus), expected_rows=len(ids))
    assert report["expected_rows"] == 8
    assert len(report["maps"]) == 9
    assert validate_staged_map_seal(seal)["identity_sha256"] == report["identity_sha256"]
    assert all(os.path.isfile(os.path.join(entry["staged_dir"], "extra-proof.txt"))
               for entry in report["maps"])
    tampered = json.loads(Path(seal).read_text())
    tampered["maps"][0]["coordinate_rows"] -= 1
    tampered["identity_sha256"] = sha256_bytes(canonical_json({
        key: value for key, value in tampered.items() if key != "identity_sha256"}))
    tampered_seal = os.path.join(fresh_data_root, "inputs", "tampered-maps-seal.json")
    Path(tampered_seal).write_text(json.dumps(tampered) + "\n")
    with pytest.raises(ValueError, match="recorded coordinate_rows mismatch"):
        validate_staged_map_seal(tampered_seal)
    with pytest.raises(FileExistsError, match="map staging root"):
        stage_round0005_maps(
            sources=sources, destination_root=destination,
            seal_path=os.path.join(fresh_data_root, "another-seal.json"),
            corpus_identity_path=str(corpus), expected_rows=len(ids))


def test_map_staging_rejects_same_length_wrong_universe_before_destination(
        tmp_path, fresh_data_root):
    ids = np.arange(8, dtype=np.int64)
    source_root = tmp_path / "bad-map-sources"
    sources = {}
    for label in MAP_EXPECTATIONS:
        candidate = ids + 1 if label == "umap_stdcurve_s44" else ids
        sources[label] = _map_source(source_root, label, candidate)
    corpus = tmp_path / "corpus.identity"
    corpus.write_bytes(b"corpus")
    destination = os.path.join(fresh_data_root, "bad-maps")
    with pytest.raises(ValueError, match="semantic-ID universe differs"):
        stage_round0005_maps(
            sources=sources, destination_root=destination,
            seal_path=os.path.join(fresh_data_root, "bad-maps-seal.json"),
            corpus_identity_path=str(corpus), expected_rows=len(ids))
    assert not os.path.exists(destination)


def test_symlink_model_snapshot_stages_deterministic_regular_files_and_seals(
        tmp_path, fresh_data_root):
    revision = TEST_REVISION
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "config").write_bytes(b"{\"hidden\": 4}\n")
    (blobs / "weights").write_bytes(b"weights")
    snapshot = tmp_path / revision
    (snapshot / "nested").mkdir(parents=True)
    os.symlink(blobs / "config", snapshot / "config.json")
    os.symlink(blobs / "weights", snapshot / "nested" / "model.safetensors")
    destination = os.path.join(fresh_data_root, "model")
    seal = os.path.join(fresh_data_root, "model-seal.json")
    report = stage_regular_model_snapshot(
        source_snapshot=str(snapshot), destination_root=destination,
        seal_path=seal, model_id="example/jina", expected_revision=revision)
    assert validate_staged_model_seal(
        seal, expected_root=destination, expected_revision=revision)["regular_files_only"]
    assert not any(path.is_symlink() for path in Path(destination).rglob("*"))
    second = stage_regular_model_snapshot(
        source_snapshot=str(snapshot),
        destination_root=os.path.join(fresh_data_root, "model-second"),
        seal_path=os.path.join(fresh_data_root, "model-second-seal.json"),
        model_id="example/jina", expected_revision=revision)
    assert second["staged_signature"]["sha256"] == report["staged_signature"]["sha256"]
    staged_config = os.path.join(destination, "config.json")
    os.chmod(staged_config, 0o644)
    Path(staged_config).write_bytes(b"mutated")
    with pytest.raises(ValueError, match="staged-model bytes changed"):
        validate_staged_model_seal(seal, expected_root=destination)
    with pytest.raises(FileExistsError, match="model staging root"):
        stage_regular_model_snapshot(
            source_snapshot=str(snapshot), destination_root=destination,
            seal_path=os.path.join(fresh_data_root, "unused-seal.json"),
            model_id="example/jina", expected_revision=revision)


def _namespace(ids, *, corpus_sha="b" * 64, name="tiny/coordinate-row-id",
               kind="coordinate_semantic_id"):
    return {"schema": SEMANTIC_NAMESPACE_SCHEMA, "name": name, "kind": kind,
            "corpus_identity_sha256": corpus_sha,
            "universe_sha256": ordered_array_sha256(np.sort(np.asarray(ids, np.int64))),
            "row_count": len(ids)}


def _parquet(path, ids, *, with_ids=True, scale=1.0):
    data = {"x": np.asarray(ids) * scale, "y": -np.asarray(ids) * scale}
    if with_ids:
        data["ls_index"] = np.asarray(ids)
    pd.DataFrame(data).to_parquet(path)


def test_render_realigns_permuted_ids_persists_rows_and_refuses_output_root(
        tmp_path, fresh_data_root):
    ids = np.arange(100, dtype=np.int64)
    namespace = _namespace(ids)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _parquet(a, ids)
    _parquet(b, ids[::-1])
    out = os.path.join(fresh_data_root, "render")
    spec_path = tmp_path / "spec.json"
    maps = [{"label": "ordered", "coords": str(a), "semantic_id_namespace": namespace},
            {"label": "permuted", "coords": str(b), "semantic_id_namespace": namespace}]
    spec = {"output_dir": out, "sample_size": 25, "comparisons": [{
        "id": "pair", "substrate": "tiny", "semantic_id_namespace": namespace,
        "maps": maps}]}
    spec_path.write_text(json.dumps(spec))
    result = render(spec, spec_path=str(spec_path))["comparisons"][0]
    first = np.load(result["maps"][0]["gathered_row_positions"]["path"])
    second = np.load(result["maps"][1]["gathered_row_positions"]["path"])
    sampled = np.load(result["sample_ids_path"])
    assert np.array_equal(ids[first], sampled)
    assert np.array_equal(ids[::-1][second], sampled)
    assert not np.array_equal(first, second)
    with pytest.raises(FileExistsError, match="render output root"):
        render(spec, spec_path=str(spec_path))


@pytest.mark.parametrize("mutation", ["same_length_wrong", "duplicate", "namespace"])
def test_render_rejects_wrong_duplicate_and_namespace_mutations(
        tmp_path, fresh_data_root, mutation):
    ids = np.arange(20, dtype=np.int64)
    namespace = _namespace(ids)
    wrong = ids + 1 if mutation == "same_length_wrong" else (
        np.r_[ids[:-1], ids[-2]] if mutation == "duplicate" else ids)
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    _parquet(a, ids)
    _parquet(b, wrong)
    item_namespace = _namespace(ids, name="wrong/namespace") if mutation == "namespace" else namespace
    spec = {"output_dir": os.path.join(fresh_data_root, f"render-{mutation}"),
            "comparisons": [{"id": "pair", "substrate": "tiny",
                             "semantic_id_namespace": namespace, "maps": [
                {"label": "a", "coords": str(a), "semantic_id_namespace": namespace},
                {"label": "b", "coords": str(b),
                 "semantic_id_namespace": item_namespace}]}]}
    spec_path = tmp_path / f"{mutation}.json"
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="duplicate|universe|namespace"):
        render(spec, spec_path=str(spec_path))


@pytest.mark.parametrize("mutation", ["valid", "coordinate_bytes", "namespace", "universe"])
def test_render_positional_declaration_is_fully_content_bound(
        tmp_path, fresh_data_root, mutation):
    ids = np.arange(12, dtype=np.int64)
    namespace = _namespace(ids, kind="row_position", name="tiny/row-position")
    coords = tmp_path / f"pos-{mutation}.parquet"
    _parquet(coords, ids, with_ids=False)
    declaration = {"kind": "row_position", "namespace": namespace,
                   "row_count": len(ids),
                   "coordinate_sha256": path_signature(coords)["sha256"],
                   "universe_sha256": ordered_array_sha256(ids)}
    if mutation == "coordinate_bytes":
        declaration["coordinate_sha256"] = "0" * 64
    elif mutation == "namespace":
        declaration["namespace"] = _namespace(
            ids, kind="row_position", name="wrong/row-position")
    elif mutation == "universe":
        declaration["universe_sha256"] = "0" * 64
    spec = {"output_dir": os.path.join(fresh_data_root, f"positional-{mutation}"),
            "comparisons": [{"id": "pos", "substrate": "tiny",
                             "semantic_id_namespace": namespace, "maps": [{
                "label": "pos", "coords": str(coords),
                "positional_identity": declaration}]}]}
    spec_path = tmp_path / f"pos-{mutation}.json"
    spec_path.write_text(json.dumps(spec))
    if mutation == "valid":
        assert render(spec, spec_path=str(spec_path))["comparisons"][0]["sample_count"] == 12
    else:
        with pytest.raises(ValueError, match="coordinate|namespace|universe"):
            render(spec, spec_path=str(spec_path))


def _query_fixture(fresh_data_root):
    runtime = os.path.join(fresh_data_root, "query")
    os.mkdir(runtime)
    testbed = os.path.join(runtime, "testbed")
    source = os.path.join(runtime, "source")
    os.makedirs(os.path.join(testbed, "train"))
    os.mkdir(source)
    rng = np.random.RandomState(0)
    src = rng.randn(40, 4).astype(np.float32)
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    train_ids = np.arange(20, dtype=np.int64)
    np.save(os.path.join(source, "data-00000.npy"), src)
    np.save(os.path.join(testbed, "train", "data-00000.npy"), src[train_ids])
    np.save(os.path.join(testbed, "sample_indices.npy"), train_ids)
    convention = {"model_id": "example/jina", "model_revision": TEST_REVISION,
                  "prompt_bytes_hex": "446f63756d656e743a20", "pooling": "lasttoken",
                  "dtype": "float32", "normalization": "l2", "dimensions": 4}
    report = build_query_artifact(
        testbed=testbed, source=source, out_dir=os.path.join(runtime, "artifact"),
        dim=4, n_holdout=8, seed=7, convention=convention)
    return testbed, source, convention, report["manifest_path"]


def _rewrite_manifest(path, payload):
    os.chmod(path, 0o644)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def _rewrite_npy(path, value):
    os.chmod(path, 0o644)
    with open(path, "wb") as handle:
        np.save(handle, value)


@pytest.mark.parametrize("mutation", [
    "identity", "selection_order", "ids", "id_dtype", "dtype", "norm", "nonfinite",
    "model_revision", "prompt", "pooling", "embedding_bytes", "corpus_bytes",
])
def test_query_artifact_mutation_matrix(tmp_path, fresh_data_root, mutation):
    testbed, source, convention, manifest_path = _query_fixture(fresh_data_root)
    payload = json.loads(Path(manifest_path).read_text())
    emb_path = payload["embeddings"]["canonical_path"]
    ids_path = payload["ids"]["canonical_path"]
    if mutation == "identity":
        payload["identity_sha256"] = "0" * 64
        _rewrite_manifest(manifest_path, payload)
    elif mutation == "selection_order":
        ids = np.load(ids_path)[::-1].copy()
        embeddings = np.load(emb_path)[::-1].copy()
        _rewrite_npy(ids_path, ids)
        _rewrite_npy(emb_path, embeddings)
        payload["ids"] = expected_input_signature(ids_path)
        payload["embeddings"] = expected_input_signature(emb_path)
        payload["ordered_ids_sha256"] = ordered_array_sha256(ids)
        payload["ordered_embeddings_sha256"] = ordered_array_sha256(embeddings)
        payload["query_selection"]["query_ids_ordered_sha256"] = ordered_array_sha256(ids)
        payload["identity_sha256"] = query_identity(payload)
        _rewrite_manifest(manifest_path, payload)
    elif mutation == "ids":
        ids = np.load(ids_path)
        replacement = next(value for value in range(20, 40) if value not in set(ids.tolist()))
        ids[0] = replacement
        ids = np.sort(ids.astype(np.int64))
        _rewrite_npy(ids_path, ids)
        payload["ids"] = expected_input_signature(ids_path)
        payload["ordered_ids_sha256"] = ordered_array_sha256(ids)
        payload["query_selection"]["query_ids_ordered_sha256"] = ordered_array_sha256(ids)
        payload["identity_sha256"] = query_identity(payload)
        _rewrite_manifest(manifest_path, payload)
    elif mutation == "id_dtype":
        ids = np.load(ids_path).astype(np.int32)
        _rewrite_npy(ids_path, ids)
        payload["ids"] = expected_input_signature(ids_path)
        payload["ordered_ids_sha256"] = ordered_array_sha256(ids)
        payload["query_selection"]["query_ids_ordered_sha256"] = ordered_array_sha256(ids)
        payload["identity_sha256"] = query_identity(payload)
        _rewrite_manifest(manifest_path, payload)
    elif mutation in {"dtype", "norm", "nonfinite"}:
        values = np.load(emb_path)
        if mutation == "dtype":
            values = values.astype(np.float64)
        elif mutation == "norm":
            values[0] *= 2.0
        else:
            values[0, 0] = np.nan
        _rewrite_npy(emb_path, values)
        payload["embeddings"] = expected_input_signature(emb_path)
        payload["ordered_embeddings_sha256"] = ordered_array_sha256(values)
        payload["identity_sha256"] = query_identity(payload)
        _rewrite_manifest(manifest_path, payload)
    elif mutation == "model_revision":
        payload["convention"]["model_revision"] = MUTATED_REVISION
        payload["identity_sha256"] = query_identity(payload)
        _rewrite_manifest(manifest_path, payload)
    elif mutation == "prompt":
        payload["convention"]["prompt_bytes_hex"] = ""
        payload["identity_sha256"] = query_identity(payload)
        _rewrite_manifest(manifest_path, payload)
    elif mutation == "pooling":
        payload["convention"]["pooling"] = "mean"
        payload["identity_sha256"] = query_identity(payload)
        _rewrite_manifest(manifest_path, payload)
    elif mutation == "embedding_bytes":
        values = np.load(emb_path)
        values[0, 0] += np.float32(1e-5)
        _rewrite_npy(emb_path, values)
    elif mutation == "corpus_bytes":
        path = os.path.join(testbed, "train", "data-00000.npy")
        values = np.load(path)
        values[0, 0] += np.float32(1e-5)
        _rewrite_npy(path, values)
    with pytest.raises(ValueError):
        load_query_artifact(
            manifest_path, testbed=testbed, expected_convention=convention)


def test_query_artifact_uses_normalized_nonplaceholder_fixture_and_refuses_root(
        fresh_data_root):
    testbed, source, convention, manifest = _query_fixture(fresh_data_root)
    loaded = load_query_artifact(manifest, testbed=testbed, expected_convention=convention)
    assert np.max(np.abs(np.linalg.norm(loaded["Xq"], axis=1) - 1.0)) < 1e-4
    assert convention["model_revision"] != "revision"
    with pytest.raises(FileExistsError, match="query artifact root"):
        build_query_artifact(
            testbed=testbed, source=source, out_dir=os.path.dirname(manifest), dim=4,
            n_holdout=8, seed=7, convention=convention)


def _truth_cache(fresh_data_root, monkeypatch):
    import basemap.panel_v2 as pv
    calls = []

    def fake_cross(Q, corpus, k, cfg, hi_dim=True, q_tile=4096, exact=True):
        calls.append(k)
        return np.tile(np.arange(k, dtype=np.int64), (len(Q), 1))

    monkeypatch.setattr(pv, "cross_knn", fake_cross)
    cfg = pv.PanelV2Config(k_hit=10)
    cache_dir = os.path.join(fresh_data_root, "truth")
    cache = pv.QueryTruthCache(cache_dir=cache_dir, enabled=True)
    cache.get_or_build(
        np.zeros((4, 3), np.float32), np.zeros((20, 3), np.float32), cfg=cfg,
        corpus_identity={"x": 1}, query_identity={"q": 1}, k=15)
    return pv, cfg, cache, calls, cache.path


def test_query_truth_builds_k15_once_slices_k10_and_verifies_persisted_payload(
        fresh_data_root, monkeypatch):
    pv, cfg, cache, calls, path = _truth_cache(fresh_data_root, monkeypatch)
    for name, k in (("parametric", 10), ("knn-build", 15),
                    ("knn-score", 10), ("random", 10)):
        assert cache.use(name, k=k).shape == (4, k)
    telemetry = cache.telemetry()
    assert calls == [15]
    assert telemetry["build_count"] == 1 and telemetry["consumer_count"] == 4
    loaded = pv.QueryTruthCache(cache_dir=os.path.dirname(path), enabled=True)
    loaded.get_or_build(
        np.zeros((4, 3), np.float32), np.zeros((20, 3), np.float32), cfg=cfg,
        corpus_identity={"x": 1}, query_identity={"q": 1}, k=15)
    assert loaded.telemetry()["disk_load_count"] == 1


def test_one_maximum_k15_truth_is_shared_by_all_nine_maps_and_baselines(
        fresh_data_root):
    fixture_path = os.path.join(fresh_data_root, "nine-map-scorer.npz")
    build_scorer_fixture(fixture_path, rows=96, query_rows=6, dimensions=5)
    fixture = load_scorer_fixture(fixture_path)
    cfg = PanelV2Config(**fixture["meta"]["panel_config"])
    cache = QueryTruthCache(
        cache_dir=os.path.join(fresh_data_root, "nine-map-truth"), enabled=True)
    cache.get_or_build(
        fixture["Xq"], fixture["X"], cfg=cfg,
        corpus_identity={"sha256": ordered_array_sha256(fixture["X"])},
        query_identity={"sha256": ordered_array_sha256(fixture["Xq"])}, k=15)
    for label in sorted(MAP_EXPECTATIONS):
        score_query_bundle(
            X=fixture["X"], Z=fixture["Z"], Xq=fixture["Xq"], Zq=fixture["Zq"],
            cfg=cfg, truth_cache=cache, label=label, random_seed=20260716)
    telemetry = cache.telemetry()
    assert telemetry["build_count"] == 1
    assert telemetry["maximum_k"] == 15
    assert telemetry["consumer_count"] == 9 * 4
    assert {entry["k"] for entry in telemetry["consumers"]} == {10, 15}


@pytest.mark.parametrize("mutation", ["payload", "collision", "policy", "shape", "dtype",
                                      "bounds", "duplicate", "k", "cardinality"])
def test_query_truth_corruption_collision_and_shape_contract(
        fresh_data_root, monkeypatch, mutation):
    pv, cfg, cache, _, path = _truth_cache(fresh_data_root, monkeypatch)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    if mutation == "payload":
        values["neighbors"][0, 0] = 19
    elif mutation == "collision":
        values["key"] = np.array("0" * 64)
    elif mutation == "policy":
        meta = json.loads(str(values["meta"]))
        candidate = meta["key_parts"]["policy"]["candidate_selection"]
        candidate["overselect"] += 1
        candidate["candidate_count"] += 1
        values["meta"] = np.array(json.dumps(
            meta, sort_keys=True, separators=(",", ":")))
        values["key"] = np.array(sha256_bytes(canonical_json(meta["key_parts"])))
    elif mutation == "shape":
        values["neighbors"] = values["neighbors"][:-1]
    elif mutation == "dtype":
        values["neighbors"] = values["neighbors"].astype(np.int32)
    elif mutation == "bounds":
        values["neighbors"][0, 0] = 20
    elif mutation == "duplicate":
        values["neighbors"][0, 1] = values["neighbors"][0, 0]
    elif mutation == "k":
        values["k"] = np.array(14)
    elif mutation == "cardinality":
        values["corpus_cardinality"] = np.array(10)
    os.chmod(path, 0o644)
    with open(path, "wb") as handle:
        np.savez(handle, **values)
    if mutation == "policy":
        expected_cache = pv.QueryTruthCache(cache_dir=os.path.dirname(path), enabled=True)
        with pytest.raises(ValueError, match="key mismatch|policy/identity mismatch"):
            expected_cache.get_or_build(
                np.zeros((4, 3), np.float32), np.zeros((20, 3), np.float32), cfg=cfg,
                corpus_identity={"x": 1}, query_identity={"q": 1}, k=15)
    else:
        with pytest.raises(ValueError):
            pv.load_query_truth(path)


def test_query_truth_policy_mutation_cannot_collide_with_nonempty_cache(
        fresh_data_root, monkeypatch):
    pv, cfg, cache, _, _ = _truth_cache(fresh_data_root, monkeypatch)
    changed = copy.copy(cfg)
    changed.overselect += 1
    second = pv.QueryTruthCache(cache_dir=cache.cache_dir, enabled=True)
    with pytest.raises(FileExistsError, match="nonempty query truth cache"):
        second.get_or_build(
            np.zeros((4, 3), np.float32), np.zeros((20, 3), np.float32), cfg=changed,
            corpus_identity={"x": 1}, query_identity={"q": 1}, k=15)


def test_actual_cache_equivalence_and_real_slowed_performance_path(
        fresh_data_root):
    fixture = os.path.join(fresh_data_root, "scorer-fixture.npz")
    build_scorer_fixture(fixture, rows=128, query_rows=8, dimensions=6)
    with pytest.raises(FileExistsError, match="scorer fixture"):
        build_scorer_fixture(fixture, rows=128, query_rows=8, dimensions=6)
    equivalence_root = os.path.join(fresh_data_root, "equivalence")
    report = run_equivalence(fixture_path=fixture, out_root=equivalence_root)
    assert report["passed"] and report["checks"]["persisted_scalars_identical"]
    with pytest.raises(FileExistsError, match="cache equivalence output root"):
        run_equivalence(fixture_path=fixture, out_root=equivalence_root)
    regression = run_synthetic_regression(
        fixture_path=fixture, out_root=os.path.join(fresh_data_root, "regression"),
        baseline_phase_delay_s=0.01)
    assert regression["passed"]
    assert regression["slowed"]["wall_s"] > regression["baseline"]["wall_s"]
    assert regression["allows_scale_launch"] is False


def test_measured_performance_gate_consumes_actual_nine_map_scorer_outputs(
        fresh_data_root):
    fixture_path = os.path.join(fresh_data_root, "performance-scorer.npz")
    build_scorer_fixture(fixture_path, rows=64, query_rows=4, dimensions=4)
    fixture = load_scorer_fixture(fixture_path)
    cfg = PanelV2Config(**fixture["meta"]["panel_config"])
    cache = QueryTruthCache(
        cache_dir=os.path.join(fresh_data_root, "performance-truth"), enabled=True)
    cache.get_or_build(
        fixture["Xq"], fixture["X"], cfg=cfg,
        corpus_identity={"sha256": ordered_array_sha256(fixture["X"])},
        query_identity={"sha256": ordered_array_sha256(fixture["Xq"])}, k=15)
    anchors = sample_anchors(len(fixture["X"]), cfg)
    reference = build_hiD_reference(fixture["X"], anchors, cfg)
    reference_path = os.path.join(fresh_data_root, "private-hiD-reference.npz")
    save_hiD_reference(reference, reference_path)
    reference_receipt = {
        "schema": "round0005_private_hiD_reference_receipt.v1",
        "reference": expected_input_signature(reference_path),
        "identity_key": reference["key"],
        "content_sha256": reference["content_sha256"],
        "key_parts": reference["key_parts"],
        "built_and_reloaded_in_same_scorer": True,
        "pre_gate_reference_consumed": False,
    }
    receipt_path = os.path.join(fresh_data_root, "private-hiD-reference-receipt.json")
    atomic_write_new_json(receipt_path, reference_receipt, immutable=True)
    runs = {}
    started = time.monotonic()
    for label in sorted(MAP_EXPECTATIONS):
        full = score_panel(
            fixture["X"], fixture["Z"], config=cfg,
            z_ids=np.arange(len(fixture["X"]), dtype=np.int64),
            hiD_reference=reference,
            provenance={"scorer": "complete_panel", "run": label})
        query = score_query_bundle(
            X=fixture["X"], Z=fixture["Z"], Xq=fixture["Xq"], Zq=fixture["Zq"],
            cfg=cfg, truth_cache=cache, label=label, random_seed=20260716)
        runs[label] = {
            "ffr": full["ffr"], "recall@k": full["recall@k"],
            "density": full["density"], "proj_ffr": query["proj_ffr"],
            "proj_knn_regressor_ffr": query["proj_knn_regressor_ffr"],
            "proj_random_floor_ffr": query["proj_random_floor_ffr"],
            "hiD_reference_key": reference["key"],
            "panel_full": full,
        }
    panel = {
        "n": len(fixture["X"]), "formula_version": cfg.formula_version, "runs": runs,
        "query_truth_cache": cache.telemetry(),
        "total_wall_s": time.monotonic() - started,
        # CUDA-hidden contract fixture; production fills these fields from the
        # scorer process's allocated and reserved CUDA peak counters.
        "peak_gpu_gb": 0.0,
        "process_cuda_peak": {
            "schema": "process_cuda_peak.v1", "available": True,
            "allocated_bytes": 0, "reserved_bytes": 0,
            "allocated_gib": 0.0, "reserved_gib": 0.0, "maximum_gib": 0.0,
        },
        "hiD_reference_content_sha256": reference["content_sha256"],
        "hiD_reference_receipt": expected_input_signature(receipt_path),
        "pre_gate_reference_consumed": False,
    }
    report = evaluate_panel(panel, wall_max_s=120, peak_max_gb=26)
    assert report["passed"], report
    assert report["allows_scale_launch"] is False
    handcrafted = copy.deepcopy(panel)
    handcrafted["runs"][next(iter(handcrafted["runs"]))]["panel_full"] = {
        "schema": "panel_v2", "formula_version": cfg.formula_version}
    assert not evaluate_panel(handcrafted)["passed"]


@pytest.mark.parametrize("scientific_rows", [8_000_000, 30_000_000])
def test_rejected_performance_fixture_hard_blocks_scale_child(
        fresh_data_root, monkeypatch, scientific_rows):
    gate = os.path.join(fresh_data_root, "reject-scale.json")
    atomic_write_new_json(gate, {"passed": True, "allows_scale_launch": False})
    with pytest.raises(RuntimeError, match="requires actual-input row derivation"):
        require_scale_performance_gate(gate, scientific_rows=scientific_rows)
    monkeypatch.setenv("BASEMAP_GPU_LEASE", os.path.join(fresh_data_root, "lease"))
    sentinel = os.path.join(fresh_data_root, "scale-launched")
    job = Job(
        "scale", [sys.executable, "-c", f"open({sentinel!r}, 'x').write('bad')"],
        [sentinel], os.path.join(fresh_data_root, "scale.done"), input_paths=[gate],
        scientific_rows=scientific_rows, performance_gate_path=gate)
    with pytest.raises(RuntimeError, match="requires an exact Round 0005 QueueAdmission"):
        run_jobs([job])
    summary = _run_jobs_fixture_only([job])
    assert summary["jobs"][0]["status"] == "blocked:performance_gate"
    assert not os.path.exists(sentinel)


def test_queue_builder_is_self_contained_signed_and_never_consumes_prior_outputs(
        fresh_data_root, monkeypatch, round0005_clean_release):
    release_args, _ = round0005_clean_release
    repo = release_args.repo_root
    round_root = os.path.join(fresh_data_root, "production-round")
    maps_root = os.path.join(round_root, "inputs", "maps")
    os.makedirs(maps_root)
    map_entries = []
    for label in sorted(MAP_EXPECTATIONS):
        staged = os.path.join(maps_root, label)
        os.mkdir(staged)
        Path(staged, "coords.parquet").write_bytes(f"coords:{label}".encode())
        Path(staged, "model.pt").write_bytes(f"model:{label}".encode())
        map_entries.append({"label": label, "staged_dir": staged})
    maps_seal = os.path.join(fresh_data_root, "maps-seal.json")
    Path(maps_seal).write_text("{}\n")
    def validate_maps(_path, *, expected_testbed_seal, require_round0005):
        assert expected_testbed_seal == testbed_seal
        assert require_round0005 is True
        return {"expected_rows": 2_000_000,
                "destination_root": maps_root, "maps": map_entries}

    monkeypatch.setattr(
        "experiments.prepare_round0005_queue.validate_staged_map_seal",
        validate_maps)
    monkeypatch.setattr(
        "basemap.round0005_staging.validate_staged_map_seal", validate_maps)

    model_root = os.path.join(round_root, "inputs", "model")
    os.makedirs(model_root)
    Path(model_root, "config.json").write_text("{}\n")
    model_seal = os.path.join(fresh_data_root, "model-seal.json")
    Path(model_seal).write_text("{}\n")
    def validate_model(_path, *, expected_root, expected_model_id,
                       expected_revision, expected_testbed_seal,
                       require_round0005):
        assert expected_root == model_root
        assert expected_revision == ROUND0005_MODEL_REVISION
        assert expected_testbed_seal == testbed_seal
        assert require_round0005 is True
        return {"staged_root": expected_root,
                "model_revision": ROUND0005_MODEL_REVISION}

    monkeypatch.setattr(
        "experiments.prepare_round0005_queue.validate_staged_model_seal",
        validate_model)
    monkeypatch.setattr(
        "basemap.round0005_staging.validate_staged_model_seal", validate_model)

    fixture = os.path.join(fresh_data_root, "fixture.json")
    Path(fixture).write_text("{}\n")
    monkeypatch.setattr(
        "experiments.prepare_round0005_queue.validate_round0005_fixture",
        lambda _path, *, repo_root, release_sha, environment_manifest: {
            "schema": "round0005_all_node_fixture.v3",
            "identity_sha256": "4" * 64,
        })
    monkeypatch.setattr(
        "basemap.round0005_fixture.validate_round0005_fixture",
        lambda _path, *, repo_root, release_sha, environment_manifest: {
            "schema": "round0005_all_node_fixture.v3",
            "identity_sha256": "4" * 64,
        })
    scorer_fixture = os.path.join(fresh_data_root, "scorer.npz")
    build_scorer_fixture(scorer_fixture, rows=64, query_rows=4, dimensions=4)
    testbed = os.path.join(fresh_data_root, "testbed")
    os.makedirs(os.path.join(testbed, "train"))
    rng = np.random.RandomState(11)
    source_values = rng.standard_normal((16, 4)).astype(np.float32)
    source_values /= np.linalg.norm(source_values, axis=1, keepdims=True)
    train_ids = np.arange(4, dtype=np.int64)
    np.save(os.path.join(testbed, "train", "data.npy"), source_values[train_ids])
    sample_indices = np.lib.format.open_memmap(
        os.path.join(testbed, "sample_indices.npy"), mode="w+",
        dtype=np.int64, shape=(2_000_000,))
    sample_indices.flush()
    del sample_indices
    np.save(os.path.join(testbed, "centroids_k256.npy"), np.zeros((2, 4), np.float32))
    np.save(os.path.join(testbed, "centroids_k1024.npy"), np.zeros((3, 4), np.float32))
    testbed_seal = os.path.join(fresh_data_root, "testbed-seal.json")
    Path(testbed_seal).write_text("{}\n")
    monkeypatch.setattr(
        "experiments.prepare_round0005_queue.validate_round0005_testbed_seal",
        lambda _path, *, require_round0005: {"testbed_root": testbed})
    monkeypatch.setattr(
        "basemap.round0005_staging.validate_round0005_testbed_seal",
        lambda _path, *, require_round0005: {"testbed_root": testbed})

    query_root = os.path.join(fresh_data_root, "query-input")
    os.mkdir(query_root)
    query_embeddings = os.path.join(query_root, "embeddings.npy")
    query_ids = os.path.join(query_root, "ids.npy")
    source_embeddings = os.path.join(query_root, "source.npy")
    np.save(query_embeddings, source_values[:4].astype(np.float32))
    np.save(query_ids, np.arange(4, dtype=np.int64))
    np.save(source_embeddings, source_values.astype(np.float16))
    query_artifact = os.path.join(query_root, "manifest.json")
    query_manifest = {
        "corpus": {"testbed": testbed},
        "embeddings": {"canonical_path": query_embeddings},
        "ids": {"canonical_path": query_ids},
        "source_embeddings": {"canonical_path": source_embeddings},
    }
    Path(query_artifact).write_text(json.dumps(query_manifest) + "\n")
    query_expectation = os.path.join(fresh_data_root, "query-expectation.json")
    Path(query_expectation).write_text("{}\n")
    monkeypatch.setattr(
        "experiments.prepare_round0005_queue.validate_convention", lambda value: value)
    monkeypatch.setattr("basemap.query_artifact.validate_convention", lambda value: value)
    monkeypatch.setattr(
        "experiments.prepare_round0005_queue.load_query_artifact",
        lambda _path, *, testbed, expected_convention, expected_testbed_seal,
        require_round0005: {"manifest": query_manifest})
    monkeypatch.setattr(
        "basemap.query_artifact.load_query_artifact",
        lambda _path, *, testbed, expected_convention, expected_testbed_seal,
        require_round0005: {"manifest": query_manifest})

    inventory = os.path.join(fresh_data_root, "inventory.parquet")
    pd.DataFrame({"row": np.arange(50_000, dtype=np.int32)}).to_parquet(
        inventory, index=False)
    calibration_sample = os.path.join(fresh_data_root, "calibration-sample.npy")
    calibration_text = os.path.join(fresh_data_root, "text-shard.jsonl")
    calibration_embeddings = os.path.join(fresh_data_root, "embedding-shard.npy")
    np.save(calibration_sample, np.arange(4, dtype=np.int64))
    Path(calibration_text).write_text("{\"text\": \"fixture\"}\n")
    np.save(calibration_embeddings, source_values.astype(np.float16))
    inventory_manifest = os.path.join(fresh_data_root, "inventory.json")
    calibration_manifest = {
        "inventory": {"canonical_path": inventory},
        "source_sample_indices": {"canonical_path": calibration_sample},
        "source_text_shards": [{"canonical_path": calibration_text}],
        "source_embedding_shards": [{"canonical_path": calibration_embeddings}],
    }
    Path(inventory_manifest).write_text(json.dumps(calibration_manifest) + "\n")
    monkeypatch.setattr(
        "experiments.prepare_round0005_queue.validate_inventory",
        lambda _path, *, expected_testbed_seal, require_round0005:
        (calibration_manifest, pd.DataFrame()))
    monkeypatch.setattr(
        "experiments.calibrate_jina_embedding.validate_inventory",
        lambda _path, *, expected_testbed_seal, require_round0005:
        (calibration_manifest, pd.DataFrame()))
    monkeypatch.setattr(
        "experiments.prepare_round0005_queue.cross_check_round0005_data_identity",
        lambda **kwargs: {"identity_sha256": "5" * 64})
    monkeypatch.setattr(
        "basemap.round0005_staging.cross_check_round0005_data_identity",
        lambda **kwargs: {"identity_sha256": "5" * 64})

    environment = release_args.environment_manifest
    environment_value = json.loads(Path(environment).read_text())
    environment_value.update({
        "python": "3.12.3", "torch": "fixture-torch+cu128",
        "torch_cuda": "12.8", "gpu_driver": "fixture-driver",
        "gpu_name": "NVIDIA GeForce RTX 5090", "gpu_uuid": "GPU-round0005-test",
    })
    identity_fields = (
        "freeze_sha256", "python", "torch", "torch_cuda", "gpu_driver",
        "gpu_name", "gpu_uuid",
    )
    environment_value["identity_sha256"] = sha256_bytes(canonical_json(
        {key: environment_value[key] for key in identity_fields}))
    Path(environment).write_text(json.dumps(environment_value) + "\n")
    monkeypatch.setattr(
        "basemap.queue_admission._observe_canonical_gpu",
        lambda: {
            "schema": "round0005_live_gpu_identity.v1",
            "observer": expected_input_signature("/usr/bin/nvidia-smi"),
            "gpus": [{"gpu_uuid": "GPU-round0005-test",
                      "gpu_name": "NVIDIA GeForce RTX 5090",
                      "gpu_driver": "fixture-driver"}],
            "inventory_sha256": sha256_bytes(canonical_json([
                {"gpu_uuid": "GPU-round0005-test",
                 "gpu_name": "NVIDIA GeForce RTX 5090",
                 "gpu_driver": "fixture-driver"},
            ])),
        })
    queue_root = os.path.join(fresh_data_root, "queue")
    queue_path = os.path.join(queue_root, "queue.json")
    argv = [
        "--release-sha", release_args.release_sha,
        "--implementation-commit", release_args.release_sha,
        "--integration-repo", release_args.integration_repo,
        "--pushed-ref", release_args.pushed_ref,
        "--round-sha256", ROUND0005_PLAN_SHA,
        "--run-root", repo, "--round-root", round_root,
        "--queue-root", queue_root, "--fixture", fixture,
        "--scorer-fixture", scorer_fixture, "--query-artifact", query_artifact,
        "--query-expectation", query_expectation, "--maps-seal", maps_seal,
        "--calibration-inventory", inventory_manifest, "--model-path", model_root,
        "--model-seal", model_seal, "--testbed-seal", testbed_seal,
        "--environment-manifest", environment, "--out", queue_path,
    ]
    assert prepare_queue(argv) == 0
    manifest = json.loads(Path(queue_path).read_text())
    assert oct(Path(queue_path).stat().st_mode & 0o777) == "0o444"
    produced = set()
    for job in manifest["jobs"]:
        assert "consumes" not in job
        assert not produced.intersection(job["inputs"])
        assert all(os.path.exists(path if os.path.isabs(path) else os.path.join(repo, path))
                   for path in job["inputs"])
        assert len(job["expected_inputs"]) == len(job["inputs"])
        produced.update(job["outputs"])
    jobs = {job["id"]: job for job in manifest["jobs"]}
    registry_paths = [value["canonical_path"]
                      for value in manifest["global_input_registry"]]
    assert all(job["inputs"] == registry_paths for job in manifest["jobs"])
    assert scorer_fixture in jobs["scalar_equivalence"]["inputs"]
    assert maps_root + os.sep + "legacy_a1b1_s42" in jobs["scalar_equivalence"]["inputs"]
    assert query_artifact in jobs["scalar_equivalence"]["inputs"]
    assert scorer_fixture in jobs["synthetic_4x_regression"]["inputs"]
    assert testbed_seal in jobs["fresh_uncached_2m"]["inputs"]
    assert "/usr/bin/nvidia-smi" in registry_paths
    assert not any("reference.npz" in value for job in manifest["jobs"]
                   for value in job["inputs"])
    for job_id in ("fresh_uncached_2m", "cached_nine_map"):
        assert any(value.endswith("hiD-reference.npz") for value in jobs[job_id]["outputs"])
    for job_id in ("fresh_uncached_2m", "cached_nine_map", "scalar_equivalence"):
        argv_runs = [value for value in jobs[job_id]["argv"] if "=" in value and
                     value.split("=", 1)[0] in MAP_EXPECTATIONS]
        assert len(argv_runs) == 9
    assert not any("uncached-one-map" in value or "cached-nine-map" in value
                   for value in jobs["no_training_seal_canary"]["inputs"])
    with pytest.raises(FileExistsError, match="queue root"):
        prepare_queue(argv)

    def reject(mutator):
        changed = copy.deepcopy(manifest)
        mutator(changed)
        with pytest.raises((RuntimeError, ValueError)):
            validate_queue_manifest(changed, queue_path)

    reject(lambda value: value["jobs"].reverse())
    reject(lambda value: value["jobs"].pop())
    reject(lambda value: value["jobs"].append(copy.deepcopy(value["jobs"][-1])))
    reject(lambda value: value["jobs"][0].__setitem__(
        "argv", [sys.executable, "-c", "pass"]))
    reject(lambda value: value["jobs"][0]["argv"].__setitem__(
        1, os.path.join(repo, "experiments", "run_experiment.py")))
    reject(lambda value: value["jobs"][0]["argv"].append("--train-config"))
    reject(lambda value: value["jobs"][0]["node_policy"].__setitem__(
        "training_performed", True))
    reject(lambda value: value["jobs"][0]["node_policy"].__setitem__(
        "scientific_rows", 0))
    reject(lambda value: value["jobs"][0]["node_policy"].__setitem__(
        "required_free_gb", 0))
    reject(lambda value: value["jobs"][0]["node_policy"].__setitem__(
        "cuda_device_count", 0))
    reject(lambda value: value["jobs"][0].__setitem__("p90_wall_s", 0))
    reject(lambda value: value["global_input_registry"].pop())
    def alias_cache_roots(value):
        value["cache_environment"]["HF_HOME"] = \
            value["cache_environment"]["XDG_CACHE_HOME"]
        value["child_environment"]["HF_HOME"] = \
            value["cache_environment"]["XDG_CACHE_HOME"]
    reject(alias_cache_roots)
    Path(maps_seal).write_text("")
    with pytest.raises(ValueError, match="changed"):
        validate_queue_manifest(manifest, queue_path)


def test_acceptance_fixture_requires_clean_detached_release_and_refuses_reuse(
        fresh_data_root):
    out = os.path.join(fresh_data_root, "acceptance-fixture.json")
    repo = str(Path(__file__).resolve().parents[1])
    release = _git(repo, "rev-parse", "HEAD")
    environment = os.path.join(fresh_data_root, "fixture-environment.json")
    Path(environment).write_text("{}\n")
    argv = [
        "--out", out, "--python", sys.executable, "--release-sha", release,
        "--implementation-commit", release, "--integration-repo", repo,
        "--pushed-ref", "refs/remotes/origin/main",
        "--round-sha256", ROUND0005_PLAN_SHA,
        "--environment-manifest", environment,
    ]
    with pytest.raises(RuntimeError, match="clean detached queue release"):
        run_acceptance_fixture(argv)
    assert not os.path.lexists(out)
    Path(out).write_text("user-owned artifact\n")
    with pytest.raises(FileExistsError, match="fixture report"):
        run_acceptance_fixture(argv)
    assert Path(out).read_text() == "user-owned artifact\n"


def _normalized_embed(_model, texts, batch_size, show_progress, return_telemetry):
    values = np.tile(np.array([[0.6, 0.8]], dtype=np.float32), (len(texts), 1))
    return values, {"requested_batch_size": batch_size,
                    "final_batch_size": batch_size, "oom_retries": 0}


def test_atomic_chunks_resume_only_valid_and_corruption_fails_without_reembedding(
        fresh_data_root):
    root = os.path.join(fresh_data_root, "chunks")
    calls = []

    def fetch(ids, *_):
        return [f"text-{int(value)}" for value in ids]

    def embed(*args, **kwargs):
        calls.append(len(args[1]))
        return _normalized_embed(*args, **kwargs)

    kwargs = dict(
        model=object(), sample_indices=np.arange(6),
        out_train=os.path.join(root, "train"), receipt_dir=os.path.join(root, "receipts"),
        text_dir="x", text_shards=[], offsets=np.array([0, 6]),
        model_commit=TEST_REVISION, compute_dtype="float32", batch_size=256,
        chunk_rows=3, fetch_fn=fetch, embed_fn=embed)
    first = embed_outer_chunks(**kwargs)
    assert first["new_chunks"] == 2 and calls == [3, 3]
    calls.clear()
    second = embed_outer_chunks(**kwargs)
    assert second["resumed_chunks"] == 2 and calls == []
    path = os.path.join(root, "train", "data-00000.npy")
    values = np.load(path)
    values[0, 0] += 0.01
    _rewrite_npy(path, values)
    with pytest.raises(RuntimeError, match="corrupt"):
        embed_outer_chunks(**kwargs)
    assert calls == []


def test_corrupt_chunk_receipt_fails_closed_and_is_not_reembedded(fresh_data_root):
    root = os.path.join(fresh_data_root, "receipt-corrupt")
    calls = []

    def fetch(ids, *_):
        return [f"text-{int(value)}" for value in ids]

    def embed(*args, **kwargs):
        calls.append(1)
        return _normalized_embed(*args, **kwargs)

    kwargs = dict(
        model=object(), sample_indices=np.arange(3),
        out_train=os.path.join(root, "train"), receipt_dir=os.path.join(root, "receipts"),
        text_dir="x", text_shards=[], offsets=np.array([0, 3]),
        model_commit=TEST_REVISION, compute_dtype="float32", batch_size=256,
        chunk_rows=3, fetch_fn=fetch, embed_fn=embed)
    embed_outer_chunks(**kwargs)
    calls.clear()
    receipt = os.path.join(root, "receipts", "chunk-00000.json")
    os.chmod(receipt, 0o644)
    Path(receipt).write_text("{broken")
    with pytest.raises(RuntimeError, match="corrupt existing embedding chunk receipt"):
        embed_outer_chunks(**kwargs)
    assert calls == []


def test_embedding_chunks_refuse_unrecognized_nonempty_roots(fresh_data_root):
    root = os.path.join(fresh_data_root, "chunk-junk")
    train = os.path.join(root, "train")
    receipts = os.path.join(root, "receipts")
    os.makedirs(train)
    os.mkdir(receipts)
    Path(train, "stale.bin").write_bytes(b"do not overwrite")
    with pytest.raises(FileExistsError, match="unrecognized files"):
        embed_outer_chunks(
            object(), sample_indices=np.arange(2), out_train=train,
            receipt_dir=receipts, text_dir="x", text_shards=[],
            offsets=np.array([0, 2]), model_commit=TEST_REVISION,
            compute_dtype="float32", batch_size=256, chunk_rows=2,
            fetch_fn=lambda ids, *_: [str(value) for value in ids],
            embed_fn=_normalized_embed)
    assert Path(train, "stale.bin").read_bytes() == b"do not overwrite"


def test_outer_chunk_oom_retries_only_current_chunk_and_keeps_completed_chunk(
        fresh_data_root):
    import torch

    class Model:
        def __init__(self):
            self.calls = []

        def encode(self, texts, *, batch_size, **_):
            self.calls.append((texts[0], batch_size))
            if len(self.calls) == 2:
                raise torch.cuda.OutOfMemoryError("synthetic second-chunk OOM")
            return np.tile(np.array([[0.6, 0.8]], np.float32), (len(texts), 1))

    model = Model()
    report = embed_outer_chunks(
        model, sample_indices=np.arange(4),
        out_train=os.path.join(fresh_data_root, "oom-chunks", "train"),
        receipt_dir=os.path.join(fresh_data_root, "oom-chunks", "receipts"),
        text_dir="x", text_shards=[], offsets=np.array([0, 4]),
        model_commit=TEST_REVISION, compute_dtype="float32", batch_size=256,
        chunk_rows=2, fetch_fn=lambda ids, *_: [f"text-{value}" for value in ids],
        embed_fn=embed_texts)
    assert [batch for _, batch in model.calls] == [256, 256, 128]
    assert [text for text, _ in model.calls] == [
        "Document: text-0", "Document: text-2", "Document: text-2"]
    assert report["oom_retries"] == 1 and report["new_chunks"] == 2
    assert report["chunks"][0]["embedding"]["oom_retries"] == 0
    assert report["chunks"][1]["embedding"]["oom_retries"] == 1


def test_oom_retry_reduces_only_current_chunk_batch():
    import torch

    class Model:
        def __init__(self):
            self.calls = []

        def encode(self, texts, *, batch_size, **_):
            self.calls.append(batch_size)
            if len(self.calls) == 1:
                raise torch.cuda.OutOfMemoryError("synthetic")
            return np.tile(np.array([[1.0, 0.0]], np.float32), (len(texts), 1))

    model = Model()
    values, telemetry = embed_texts(
        model, ["a", "b"], batch_size=256, return_telemetry=True)
    assert values.shape == (2, 2) and model.calls == [256, 128]
    assert telemetry == {"requested_batch_size": 256,
                         "final_batch_size": 128, "oom_retries": 1}


class _FakeTokenizer:
    model_max_length = 8

    def __init__(self):
        self.seen_prompt = False

    def __call__(self, texts, *, truncation, max_length, **_):
        assert truncation is True and max_length == 8
        self.seen_prompt = self.seen_prompt or all(text.startswith("Document: ") for text in texts)
        return {"input_ids": [list(range(min(max_length, 2 + len(text.split()))))
                              for text in texts]}


class _FakeModel:
    max_seq_length = 8

    def __init__(self):
        self.tokenizer = _FakeTokenizer()


def _calibration_frame():
    n = 50_000
    return pd.DataFrame({
        "phase": ["calibration"] * 25_000 + ["heldout"] * 25_000,
        "source_position": np.arange(n, dtype=np.int64),
        "global_id": np.arange(n, dtype=np.int64),
        "text": [f"calibration text {i} with tokens" for i in range(25_000)] +
                [f"heldout text {i} with tokens" for i in range(25_000)],
        "character_length": np.full(n, 32, dtype=np.int64),
    })


def _calibration_embed(*, calibration_sleep=0.15, heldout_sleep=0.15, oom=0):
    def embed(_model, texts, batch_size, show_progress, return_telemetry):
        if texts and "heldout" in texts[0]:
            time.sleep(heldout_sleep)
        else:
            time.sleep(calibration_sleep)
        values = np.tile(np.array([[1.0, 0.0]], np.float32), (len(texts), 1))
        return values, {"requested_batch_size": batch_size,
                        "final_batch_size": batch_size, "oom_retries": oom}
    return embed


def _calibration_convention():
    return {"model_id": "jinaai/jina-embeddings-v5-text-nano-retrieval",
            "model_revision": TEST_REVISION,
            "prompt_bytes_hex": "446f63756d656e743a20", "pooling": "lasttoken",
            "dtype": "float32", "normalization": "l2"}


def test_cuda_hidden_fake_calibration_profiles_prompted_truncated_tokens_and_two_chunks(
        fresh_data_root):
    model = _FakeModel()
    report = certify_with_model(
        frame=_calibration_frame(), model=model, model_revision=TEST_REVISION,
        convention=_calibration_convention(),
        out_dir=os.path.join(fresh_data_root, "calibration-pass"),
        embed_fn=_calibration_embed())
    assert report["passed"]
    assert report["checks"]["exactly_two_new_atomic_25k_chunks"]
    assert report["chunks"]["new_chunks"] == 2
    assert model.tokenizer.seen_prompt
    assert report["token_lengths"]["calibration"]["token_length_quantiles"]["maximum"] <= 8


@pytest.mark.parametrize("failure", ["oom", "prediction"])
def test_cuda_hidden_fake_calibration_rejects_oom_and_heldout_error(
        fresh_data_root, failure):
    embed = (_calibration_embed(oom=1) if failure == "oom" else
             _calibration_embed(calibration_sleep=0.01, heldout_sleep=0.20))
    report = certify_with_model(
        frame=_calibration_frame(), model=_FakeModel(), model_revision=TEST_REVISION,
        convention=_calibration_convention(),
        out_dir=os.path.join(fresh_data_root, f"calibration-{failure}"), embed_fn=embed)
    assert not report["passed"]
    key = "zero_oom" if failure == "oom" else "heldout_prediction_error"
    assert report["checks"][key] is False


def test_calibration_refuses_existing_output_root(fresh_data_root):
    out = os.path.join(fresh_data_root, "existing-calibration")
    os.mkdir(out)
    with pytest.raises(FileExistsError, match="certifying calibration output root"):
        certify_with_model(
            frame=_calibration_frame(), model=_FakeModel(), model_revision=TEST_REVISION,
            convention=_calibration_convention(), out_dir=out,
            embed_fn=_calibration_embed())


def test_calibration_inventory_builder_preserves_source_order_and_refuses_outputs(
        tmp_path, fresh_data_root):
    testbed = tmp_path / "testbed"
    text_dir = tmp_path / "text"
    embed_dir = tmp_path / "embed"
    testbed.mkdir()
    text_dir.mkdir()
    embed_dir.mkdir()
    np.save(testbed / "sample_indices.npy", np.arange(50_000, dtype=np.int64))
    pd.DataFrame({"chunk_text": [f"text {i}" for i in range(50_000)]}).to_parquet(
        text_dir / "part-000.parquet")
    np.save(embed_dir / "data-000.npy", np.zeros((50_000, 2), np.float32))
    out_parquet = os.path.join(fresh_data_root, "inventory.parquet")
    out_manifest = os.path.join(fresh_data_root, "inventory.json")
    report = prepare_calibration_inventory(
        testbed=str(testbed), text_dir=str(text_dir), embed_dir=str(embed_dir),
        out_parquet=out_parquet, out_manifest=out_manifest,
        model_revision=TEST_REVISION, seed=0)
    frame = pd.read_parquet(out_parquet)
    validated, validated_frame = validate_inventory(out_manifest)
    assert validated["identity_sha256"] == report["identity_sha256"]
    assert len(validated_frame) == 50_000
    assert np.array_equal(frame.source_position.to_numpy(), np.arange(50_000))
    assert "length rank" not in report["selection"]
    with pytest.raises(FileExistsError, match="calibration inventory output"):
        prepare_calibration_inventory(
            testbed=str(testbed), text_dir=str(text_dir), embed_dir=str(embed_dir),
            out_parquet=out_parquet,
            out_manifest=os.path.join(fresh_data_root, "unused-inventory.json"),
            model_revision=TEST_REVISION, seed=0)


def test_neighbor_overlap_report_emits_retention_and_true_jaccard_with_self_exclusion(
        monkeypatch):
    calls = []
    outputs = [
        np.array([[1, 2, 3], [0, 2, 3]], dtype=np.int32),
        np.array([[1, 2, 4], [0, 2, 4]], dtype=np.int32),
    ]

    def fake_topk(Xq, Xc, k, device, chunk, exclude_self_ids):
        calls.append(np.asarray(exclude_self_ids).copy())
        return outputs[len(calls) - 1], np.zeros((2, 3), np.float32)

    monkeypatch.setattr("experiments.build_prompted_graph.topk_neighbors", fake_topk)
    report = neighbor_overlap_report(
        np.zeros((5, 2), np.float32), np.ones((5, 2), np.float32),
        np.array([0, 1], np.int64), k=3, device="cpu")
    assert report["schema"] == "neighbor_overlap_metrics.v2"
    assert report["self_excluded"] is True
    assert report["retention"]["mean"] == pytest.approx(2 / 3)
    assert report["true_jaccard"]["mean"] == pytest.approx(1 / 2)
    assert all(np.array_equal(call, [0, 1]) for call in calls)
