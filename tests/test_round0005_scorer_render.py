"""Focused Round 0005 scorer/reference/performance/render corrections."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from basemap.artifact_identity import (canonical_json, expected_input_signature,
                                       ordered_array_sha256, path_signature,
                                       sha256_bytes)
from basemap.output_safety import atomic_write_new_json
from basemap.panel_v2 import (FORMULA_VERSION, PanelV2Config, QueryTruthCache,
                              align_x_to_z, build_hiD_reference,
                              load_hiD_reference, process_cuda_peak,
                              query_truth_key, save_query_truth,
                              reset_process_cuda_peak, sample_anchors,
                              save_hiD_reference, validate_hiD_reference)
from basemap.round0005_staging import MAP_EXPECTATIONS, SEMANTIC_NAMESPACE_SCHEMA
from experiments.build_round0005_scorer_fixture import build_fixture
from experiments.compare_panel_cache import (complete_panel_child_output_contract,
                                               compare,
                                               _inherited_lease_pass_fds)
from experiments.render_fixed_comparisons import (ROUND0005_FIXED_COMPARISONS,
                                                   build_round0005_fixed_spec,
                                                   render,
                                                   validate_round0005_fixed_spec)
from experiments.round0005_performance_gate import (
    PERFORMANCE_CERTIFICATE_SCHEMA, SCALE_POLICY_SCHEMA, derive_scale_rows,
    _controller_wall_matches, _validate_complete_panel_report,
    _validate_complete_report_structure, _validate_controller_checkpoint_journal,
    _validate_query_truth_evidence,
    issue_scale_performance_certificate, require_scale_performance_gate,
    run_synthetic_regression, validate_regression_certificate,
    validate_scale_performance_certificate,
)
from experiments.score_complete_panel import (expected_query_consumers,
                                               parse_run_pairs,
                                               validate_query_truth_consumers)


def test_scalar_equivalence_distinguishes_missing_leaf_from_persisted_null():
    uncached = {"persisted_scalars": {"metric": {"leaf": None, "other": 1.0}}}
    cached = {"persisted_scalars": {"metric": {"other": 1.0}}}
    report = compare(uncached, cached, label="missing-versus-null")
    assert report["passed"] is False
    difference = report["differences"]["metric.leaf"]
    assert difference == {
        "uncached_present": True, "cached_present": False,
        "uncached": None, "cached": None,
    }


def _tiny_reference():
    rng = np.random.RandomState(20260716)
    X = rng.standard_normal((64, 6)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    cfg = PanelV2Config(
        frac=0.1, k_hit=3, k_density=3, n_anchors=6, corpus_chunk=64,
        overselect=4, block_elems=100_000, rerank_byte_cap=8_000_000,
        peak_byte_cap=16_000_000)
    anchors = sample_anchors(len(X), cfg)
    return X, cfg, build_hiD_reference(X, anchors, cfg)


def test_identity_arange_alignment_keeps_memmap_zero_copy(fresh_data_root):
    path = os.path.join(fresh_data_root, "x.npy")
    np.save(path, np.arange(40, dtype=np.float32).reshape(20, 2))
    X = np.load(path, mmap_mode="r")
    Z = np.zeros((20, 2), dtype=np.float32)
    ids = np.arange(20, dtype=np.int64)
    aligned, aligned_ids, note = align_x_to_z(X, Z, None, ids)
    assert aligned is X
    assert np.array_equal(aligned_ids, ids)
    assert note == "semantic_identity_arange_zero_copy"
    permutation = ids[::-1].copy()
    gathered, _, note = align_x_to_z(X, Z, None, permutation)
    assert note == "gather_X_by_z_ids"
    assert np.array_equal(gathered, np.asarray(X)[::-1])


def test_hid_reference_v3_binds_every_payload_and_rejects_archive_extras(
        fresh_data_root):
    _, _, reference = _tiny_reference()
    path = save_hiD_reference(reference, os.path.join(fresh_data_root, "reference.npz"))
    loaded = load_hiD_reference(
        path, expected_key=reference["key"], expected_key_parts=reference["key_parts"])
    assert loaded["schema"] == "hiD_reference.v3"
    assert loaded["content_sha256"] == reference["content_sha256"]

    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    values["hi_hit"][0, 0] = values["anchor_ids"][0]
    corrupted = os.path.join(fresh_data_root, "corrupted-payload.npz")
    np.savez(corrupted, **values)
    with pytest.raises(ValueError, match="self exclusion|payload content"):
        load_hiD_reference(corrupted)

    values["hi_hit"] = reference["hi_hit"]
    values["unexpected"] = np.array(1)
    extra = os.path.join(fresh_data_root, "extra-field.npz")
    np.savez(extra, **values)
    with pytest.raises(ValueError, match="missing/extra"):
        load_hiD_reference(extra)

    forged = copy.deepcopy(reference)
    forged["guard_hit"]["overselect"] += 1
    forged["content_sha256"] = sha256_bytes(canonical_json({"forged": True}))
    with pytest.raises(ValueError, match="guard_hit"):
        validate_hiD_reference(forged)


class _FakeCuda:
    def __init__(self):
        self.resets = 0

    def is_available(self):
        return True

    def reset_peak_memory_stats(self):
        self.resets += 1

    def max_memory_allocated(self):
        return 1 * 1024 ** 3

    def max_memory_reserved(self):
        return 3 * 1024 ** 3


def test_process_global_cuda_peak_counts_allocated_and_reserved():
    cuda = _FakeCuda()
    assert reset_process_cuda_peak(cuda) is True
    assert cuda.resets == 1
    peak = process_cuda_peak(cuda)
    assert peak == {
        "schema": "process_cuda_peak.v1", "available": True,
        "allocated_bytes": 1024 ** 3, "reserved_bytes": 3 * 1024 ** 3,
        "allocated_gib": 1.0, "reserved_gib": 3.0, "maximum_gib": 3.0,
    }


def test_exact_consumers_duplicate_labels_lease_fd_and_child_siblings(
        fresh_data_root, monkeypatch):
    labels = sorted(MAP_EXPECTATIONS)
    consumers = expected_query_consumers(labels, k_hit=10)
    telemetry = {"build_count": 1, "maximum_k": 15,
                 "consumer_count": len(consumers), "consumers": consumers}
    assert validate_query_truth_consumers(
        telemetry, labels, k_hit=10, expected_builds=1)["passed"]
    duplicated = copy.deepcopy(telemetry)
    duplicated["consumers"][-1] = copy.deepcopy(duplicated["consumers"][0])
    assert not validate_query_truth_consumers(
        duplicated, labels, k_hit=10, expected_builds=1)["passed"]
    with pytest.raises(ValueError, match="duplicate map label"):
        parse_run_pairs(["same=/a", "same=/b"])

    cache = QueryTruthCache(cache_dir=None, enabled=False)
    cache.truth = {"k": 15, "neighbors": np.zeros((2, 15), dtype=np.int64)}
    cache.use("one", k=10)
    with pytest.raises(ValueError, match="duplicate query truth consumer"):
        cache.use("one", k=10)

    read_fd, write_fd = os.pipe()
    try:
        monkeypatch.setenv("BASEMAP_GPU_LEASE_FD", str(read_fd))
        assert _inherited_lease_pass_fds() == (read_fd,)
    finally:
        os.close(read_fd)
        os.close(write_fd)
    with pytest.raises(RuntimeError, match="stale/closed"):
        _inherited_lease_pass_fds()

    contract = complete_panel_child_output_contract(
        os.path.join(fresh_data_root, "child"), "on")
    assert set(contract) == {
        "report", "hiD_reference", "hiD_reference_receipt",
        "child_process_receipt", "query_truth_cache",
    }
    assert len({value for value in contract.values() if value is not None}) == 5
    assert contract["report"].endswith("/report.json")
    assert contract["hiD_reference"].endswith("/hiD-reference.npz")


def _complete_report(metric_value=2):
    label = "map"
    consumers = expected_query_consumers([label], k_hit=10)
    return {
        "runs": {label: {
            "wall_s": 1.0, "ffr": 0.5,
            "panel_full": {
                "schema": "panel_v2", "formula_version": FORMULA_VERSION,
                "ffr": 0.5, "nested": {"numerator": metric_value,
                                        "guard_vector": [1, 2]},
                "provenance": {"runtime": "ignored"},
            },
        }},
        "query_truth_cache": {
            "build_count": 1, "maximum_k": 15, "consumer_count": 4,
            "consumers": consumers,
        },
    }


def test_cache_equivalence_covers_nested_scientific_scalars():
    uncached = _complete_report()
    cached = copy.deepcopy(uncached)
    cached["runs"]["map"]["wall_s"] = 99
    cached["runs"]["map"]["panel_full"]["provenance"] = {"runtime": "changed"}
    assert compare(uncached, cached)["passed"]
    cached["runs"]["map"]["panel_full"]["nested"]["numerator"] = 3
    result = compare(uncached, cached)
    assert not result["passed"]
    assert any(key.endswith("nested.numerator") for key in result["differences"])
    cached = copy.deepcopy(uncached)
    cached["runs"]["map"]["panel_full"]["nested"]["guard_vector"][1] = 3
    result = compare(uncached, cached)
    assert not result["passed"]
    assert any(key.endswith("nested.guard_vector[1]") for key in result["differences"])


def _write_private_reference_receipt(root: str, *, rows: int,
                                     dimensions: int) -> tuple[dict, str, str]:
    rng = np.random.RandomState(20260717)
    matrix = rng.standard_normal((rows, dimensions)).astype(np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    cfg = PanelV2Config(
        frac=0.1, k_hit=3, k_density=3, n_anchors=6, corpus_chunk=64,
        overselect=4, block_elems=100_000, rerank_byte_cap=8_000_000,
        peak_byte_cap=16_000_000)
    reference = build_hiD_reference(matrix, sample_anchors(rows, cfg), cfg)
    reference_path = os.path.join(root, "private-reference.npz")
    save_hiD_reference(reference, reference_path)
    receipt = {
        "schema": "round0005_private_hiD_reference_receipt.v1",
        "reference": expected_input_signature(reference_path),
        "identity_key": reference["key"],
        "content_sha256": reference["content_sha256"],
        "key_parts": reference["key_parts"],
        "built_and_reloaded_in_same_scorer": True,
        "pre_gate_reference_consumed": False,
    }
    receipt_path = os.path.join(root, "private-reference-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return (expected_input_signature(receipt_path), reference["key"],
            reference["content_sha256"])


def _measured_panel(root: str, *, rows: int = 64) -> dict:
    labels = sorted(MAP_EXPECTATIONS)
    receipt_signature, reference_key, reference_content = (
        _write_private_reference_receipt(root, rows=rows, dimensions=6))
    runs = {}
    density = {"spearman": 0.8}
    for label in labels:
        runs[label] = {
            "ffr": 0.5, "recall@k": 0.25, "density": density,
            "proj_ffr": 0.4, "proj_knn_regressor_ffr": 0.3,
            "proj_random_floor_ffr": 0.1, "hiD_reference_key": reference_key,
            "panel_full": {
                "schema": "panel_v2", "formula_version": FORMULA_VERSION,
                "ffr": 0.5, "recall@k": 0.25, "density": density,
                "provenance": {"scorer": "complete_panel", "run": label},
            },
        }
    consumers = expected_query_consumers(labels, k_hit=10)
    return {
        "n": rows, "formula_version": FORMULA_VERSION, "runs": runs,
        "query_truth_cache": {
            "build_count": 1, "maximum_k": 15,
            "consumer_count": len(consumers), "consumers": consumers,
        },
        "total_wall_s": 20.0, "peak_gpu_gb": 3.0,
        "process_cuda_peak": {
            "schema": "process_cuda_peak.v1", "available": True,
            "allocated_bytes": 2 * 1024 ** 3, "reserved_bytes": 3 * 1024 ** 3,
            "allocated_gib": 2.0, "reserved_gib": 3.0, "maximum_gib": 3.0,
        },
        "hiD_reference_content_sha256": reference_content,
        "hiD_reference_receipt": receipt_signature,
        "pre_gate_reference_consumed": False,
    }


def test_small_hand_authored_panel_cannot_issue_production_scale_certificate(
        fresh_data_root, clean_release_evidence):
    fixture_path = os.path.join(fresh_data_root, "scorer-fixture.npz")
    build_fixture(fixture_path, rows=96, query_rows=6, dimensions=6)
    regression_root = os.path.join(fresh_data_root, "regression")
    regression = run_synthetic_regression(
        fixture_path=fixture_path, out_root=regression_root,
        baseline_phase_delay_s=0.05, slowdown_injection_factor=64.0)
    validation = validate_regression_certificate(regression)
    assert validation["passed"], validation

    panel = _measured_panel(fresh_data_root)
    panel_path = os.path.join(fresh_data_root, "panel.json")
    atomic_write_new_json(panel_path, panel, immutable=True)
    regression_path = os.path.join(regression_root, "regression.json")

    # A sparse 8M x 1 .npy gives the row derivation a real file header/shape
    # without allocating the matrix in RAM.
    scale_path = os.path.join(fresh_data_root, "scale-8m.npy")
    scale = np.lib.format.open_memmap(
        scale_path, mode="w+", dtype=np.float32, shape=(8_000_000, 1))
    scale.flush()
    del scale
    row_derivation = derive_scale_rows(scale_path, dimensions=1)
    with pytest.raises(ValueError, match="standalone panel/regression evidence"):
        issue_scale_performance_certificate(
            panel_path=panel_path, regression_path=regression_path,
            release_preflight_receipt=clean_release_evidence["receipt_path"],
            row_derivation=row_derivation)
    legacy = {
        "schema": "round0005_performance_certificate.v2", "passed": True,
        "allows_scale_launch": True,
        "release_sha": clean_release_evidence["release_sha"],
        "scientific_rows": 8_000_000, "row_derivation": row_derivation,
    }
    rejected = validate_scale_performance_certificate(
        legacy, expected_release_sha=clean_release_evidence["release_sha"],
        scientific_rows=8_000_000, row_derivation=row_derivation)
    assert not rejected["passed"]
    assert not rejected["checks"]["exact_fields"]


def test_production_certificate_cli_accepts_queue_terminal_and_immediately_reopens(
        fresh_data_root, monkeypatch):
    import experiments.round0005_performance_gate as gate

    scale_path = os.path.join(fresh_data_root, "cli-scale-8m.npy")
    scale = np.lib.format.open_memmap(
        scale_path, mode="w+", dtype=np.float32, shape=(8_000_000, 1))
    scale.flush(); del scale
    row_derivation = derive_scale_rows(scale_path, dimensions=1)
    row_path = os.path.join(fresh_data_root, "row-derivation.json")
    Path(row_path).write_text(json.dumps(row_derivation) + "\n")

    def artifact(name, payload="fixture\n"):
        path = os.path.join(fresh_data_root, name)
        Path(path).write_text(payload)
        return path

    release_path = artifact("release.json", "{}\n")
    maps_path = artifact("maps-seal.json", "{}\n")
    testbed_path = artifact("testbed-seal.json", "{}\n")
    jobs = []
    for node_id in gate._CERTIFYING_JOB_IDS:
        controller_path = artifact(f"{node_id}.controller.json", "{}\n")
        done_path = artifact(f"{node_id}.done.json", "{}\n")
        log_path = artifact(f"{node_id}.log")
        output_path = artifact(f"{node_id}.output.json", "{}\n")
        jobs.append({
            "id": node_id, "manifest": controller_path,
            "done_marker": done_path, "log": log_path,
            "outputs": [output_path],
        })
    queue_path = os.path.join(fresh_data_root, "production-queue.json")
    terminal_path = artifact("production-terminal.json", "{}\n")
    manifest = {
        "release_preflight_identity": "d" * 64,
        "program_inputs": [
            {"role": "release_preflight_receipt",
             "signature": expected_input_signature(release_path)},
            {"role": "maps_seal", "signature": expected_input_signature(maps_path)},
            {"role": "testbed_seal",
             "signature": expected_input_signature(testbed_path)},
        ],
        "gate_preparation_receipt": artifact("gate-preparation.json", "{}\n"),
        "jobs": jobs,
    }
    Path(queue_path).write_text(json.dumps(manifest) + "\n")
    release_sha = "1" * 40
    monkeypatch.setattr(gate, "validate_release_preflight_receipt", lambda *_a, **_k: {
        "release_sha": release_sha,
        "environment_manifest": {"path": "/data/environment.json"},
        "environment_freeze": {"path": "/data/freeze.txt"},
        "python_executable": {"path": "/data/venv/bin/python"},
    })
    derived = {
        "environment_binding": {"sealed": True},
        "controller_identity": {"queue_terminal": True},
        "benchmark_identity": {"production_shape": True},
        "measured_gate": {"passed": True},
        "measured_total_slowdown": 4.25,
    }
    monkeypatch.setattr(gate, "_reopen_production_scale_evidence",
                        lambda **_kwargs: copy.deepcopy(derived))
    monkeypatch.setattr(
        "basemap.run_controller.require_round0005_child_admission",
        lambda _script: (_ for _ in ()).throw(
            AssertionError("post-terminal issuer must not require a live child")))
    out_path = os.path.join(fresh_data_root, "production-certificate.json")
    assert gate.main([
        "--queue-manifest", queue_path,
        "--controller-terminal", terminal_path,
        "--scale-row-derivation", row_path, "--out", out_path,
    ]) == 0
    report = json.loads(Path(out_path).read_text())
    reopened = validate_scale_performance_certificate(
        report, expected_release_sha=release_sha, scientific_rows=8_000_000,
        row_derivation=row_derivation)
    assert reopened["passed"], reopened
    assert report["evidence"]["queue_manifest"]["canonical_path"] == queue_path
    assert report["evidence"]["controller_terminal"]["canonical_path"] == terminal_path


@pytest.mark.parametrize("mutation", ["reordered", "alternate"])
def test_production_benchmark_rejects_reordered_or_alternate_nine_maps(mutation):
    labels = sorted(MAP_EXPECTATIONS)
    if mutation == "reordered":
        observed = list(reversed(labels))
    else:
        observed = [*labels[:-1], "unsealed-alternate-map"]
    report = {"n": 2_000_000, "testbed": "/data/sealed-testbed",
              "runs": {label: {} for label in observed}}
    with pytest.raises(ValueError, match="ordered 2M nine-map"):
        _validate_complete_panel_report(
            report, context={"testbed": "/data/sealed-testbed"},
            cache_enabled=True, expected_wall_limit=120.0,
            expected_report_path="/data/report.json",
            expected_cache_path="/data/cache", label="adversarial benchmark")


def test_production_benchmark_requires_complete_schema_and_controller_timing():
    with pytest.raises(ValueError, match="top-level fields"):
        _validate_complete_report_structure(
            {"runs": {}}, context={}, cache_enabled=False,
            expected_report_path="/data/report.json", expected_cache_path=None,
            label="missing-leaf benchmark")
    controller = {"record": {"seconds": 100.0}}
    assert _controller_wall_matches(90.0, controller)
    assert not _controller_wall_matches(5.0, controller)
    assert not _controller_wall_matches(float("nan"), controller)


def test_production_certificate_recomputes_controller_gpu_telemetry(fresh_data_root):
    from experiments.round0005_performance_gate import (
        _validate_runtime_gpu_snapshot)

    raw_job = {
        "id": "cached_nine_map",
        "node_policy": {"gpu_memory_cap_mb": 26 * 1024},
    }
    accounting = {
        "job_allocated_accounted_mb": 4096.0,
        "job_reserved_cap_mb": float(26 * 1024),
        "service_allocated_accounted_mb": 512.0,
        "service_reserved_cap_mb": 1024.0,
        "cumulative_allocated_accounted_mb": 4608.0,
        "cumulative_reserved_cap_mb": float(27 * 1024),
        "device_total_mb": float(40 * 1024),
    }
    environment_path = os.path.join(fresh_data_root, "gpu-accounting-environment.json")
    Path(environment_path).write_text(json.dumps({
        "gpu_uuid": "GPU-round0005-test",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_driver": "fixture-driver",
    }) + "\n")
    service = {
        "pid": 200, "proc_starttime_ticks": 20,
        "cmdline_sha256": "b" * 64, "service_identity": "ls-serve",
        "marker": "ls-serve", "gpu_memory_budget_mb": 1024,
    }
    manifest = {
        "environment_manifest": environment_path,
        "child_environment": {"CUDA_VISIBLE_DEVICES": "GPU-round0005-test"},
        "allowed_processes": [service],
    }
    gpu = {
        "at": "fixture", "gpu": "fixture", "compute_apps": [],
        "compute_app_records": [
            {"gpu_uuid": "GPU-round0005-test", "pid": 123,
             "used_memory_mb": 4096.0},
            {"gpu_uuid": "GPU-round0005-test", "pid": 200,
             "used_memory_mb": 512.0},
        ],
        "compute_pids": [123, 200], "free_mb": float(40 * 1024 - 4608),
        "used_mb": 4608.0, "total_mb": float(40 * 1024), "n_co_tenants": 2,
        "observer": expected_input_signature("/usr/bin/nvidia-smi"),
        "gpu_uuid": "GPU-round0005-test",
        "gpu_name": "NVIDIA GeForce RTX 5090", "gpu_driver": "fixture-driver",
    }
    process = {"pid": 123, "ppid": 1, "proc_starttime_ticks": 10,
               "cmdline_sha256": "a" * 64}
    snapshot = {
        "schema": "round0005_runtime_gpu_telemetry.v2", "at": "fixture",
        "job": "cached_nine_map", "root_pid": 123,
        "lease_owner": {"token": "controller"},
        "process_tree": [process],
        "known_process_identities": [process],
        "allowed_services": {"expected": [service], "observed": [service],
                             "gpu_snapshot": gpu},
        "gpu_snapshot": gpu,
        "unknown_gpu_processes": [],
        "job_gpu_processes": [gpu["compute_app_records"][0]],
        "service_gpu_processes": [gpu["compute_app_records"][1]],
        "memory_accounting": accounting,
        "errors": [],
    }
    assert _validate_runtime_gpu_snapshot(
        snapshot, raw_job=raw_job, manifest=manifest,
        label="valid controller sample") == 4096.0
    forged = copy.deepcopy(snapshot)
    forged["memory_accounting"]["cumulative_allocated_accounted_mb"] = 1.0
    with pytest.raises(ValueError, match="does not recompute"):
        _validate_runtime_gpu_snapshot(
            forged, raw_job=raw_job, manifest=manifest,
            label="forged controller sample")
    missing = copy.deepcopy(snapshot)
    del missing["memory_accounting"]["job_allocated_accounted_mb"]
    with pytest.raises(ValueError, match="incomplete"):
        _validate_runtime_gpu_snapshot(
            missing, raw_job=raw_job, manifest=manifest,
            label="missing controller sample")
    unsupported = copy.deepcopy(snapshot)
    unsupported["job_gpu_processes"] = []
    with pytest.raises(ValueError, match="allocation|authenticated"):
        _validate_runtime_gpu_snapshot(
            unsupported, raw_job=raw_job, manifest=manifest,
            label="allocation without process evidence")


def test_production_query_truth_telemetry_reopens_exact_bound_npz(
        fresh_data_root, monkeypatch):
    import experiments.round0005_performance_gate as gate

    corpus = {"sealed": "ordered-corpus"}
    query_identity = {
        "artifact_identity_sha256": "a" * 64,
        "ordered_query_ids_sha256": "b" * 64,
        "ordered_query_embeddings_sha256": "c" * 64,
    }
    context = {"query": {
        "identity_sha256": "a" * 64,
        "manifest": {
            "ordered_ids_sha256": "b" * 64,
            "ordered_embeddings_sha256": "c" * 64,
        },
    }}
    monkeypatch.setattr(
        gate, "_sealed_reference_identities",
        lambda _context: ({"sealed": "data"}, {"256": {}}, corpus))
    cfg = PanelV2Config(frac=0.001, n_anchors=10_000, corpus_chunk=500_000)
    key, parts = query_truth_key(
        corpus_identity=corpus, query_identity=query_identity, cfg=cfg, k=15,
        corpus_cardinality=2_000_000, query_rows=20_000, dimensions=768,
        candidate_compute_backend="cuda")
    cache_root = os.path.join(fresh_data_root, "exact-query-cache")
    os.mkdir(cache_root)
    neighbors = np.tile(np.arange(15, dtype=np.int64), (20_000, 1))
    truth = {
        "schema": "heldout_query_truth.v2", "key": key, "key_parts": parts,
        "k": 15, "query_rows": 20_000, "corpus_cardinality": 2_000_000,
        "neighbors": neighbors,
        "payload_sha256": ordered_array_sha256(neighbors),
        "build_wall_s": 0.25,
    }
    npz_path = os.path.join(cache_root, f"{key}.npz")
    save_query_truth(truth, npz_path)
    telemetry = {
        "enabled": True, "key": key, "path": npz_path, "maximum_k": 15,
        "build_count": 1, "disk_load_count": 0, "consumer_count": 1,
        "hit_count": 0, "consumers": [{"consumer": "map:parametric", "k": 10}],
        "build_wall_s": 0.25,
    }
    reopened = _validate_query_truth_evidence(
        {"query_truth_cache": telemetry}, context=context, cache_enabled=True,
        expected_cache_path=cache_root, label="production-shaped cache")
    assert reopened["key"] == key

    partial = copy.deepcopy(telemetry)
    partial.pop("disk_load_count")
    with pytest.raises(ValueError, match="fields are not exact"):
        _validate_query_truth_evidence(
            {"query_truth_cache": partial}, context=context, cache_enabled=True,
            expected_cache_path=cache_root, label="partial cache")
    empty_root = os.path.join(fresh_data_root, "empty-query-cache")
    os.mkdir(empty_root)
    empty = {**telemetry, "path": os.path.join(empty_root, f"{key}.npz")}
    with pytest.raises(ValueError, match="empty|wrong|ambiguous"):
        _validate_query_truth_evidence(
            {"query_truth_cache": empty}, context=context, cache_enabled=True,
            expected_cache_path=empty_root, label="empty cache")
    monkeypatch.setattr(
        gate, "_sealed_reference_identities",
        lambda _context: ({"sealed": "data"}, {"256": {}},
                          {"sealed": "different-corpus"}))
    with pytest.raises(ValueError, match="relationships"):
        _validate_query_truth_evidence(
            {"query_truth_cache": telemetry}, context=context, cache_enabled=True,
            expected_cache_path=cache_root, label="corpus mismatch")


def test_hand_minted_terminal_without_checkpoint_journal_is_rejected(
        fresh_data_root):
    root = os.path.join(fresh_data_root, "empty-controller-journal")
    os.mkdir(root)
    with pytest.raises(ValueError, match="checkpoint journal"):
        _validate_controller_checkpoint_journal(
            terminal={"controller_id": "hand-minted", "jobs": []},
            manifest={"controller_checkpoints_dir": root}, registry={})


def _namespace(ids: np.ndarray, name: str) -> dict:
    return {
        "schema": SEMANTIC_NAMESPACE_SCHEMA, "name": name,
        "kind": "coordinate_semantic_id", "corpus_identity_sha256": "d" * 64,
        "universe_sha256": ordered_array_sha256(np.sort(ids)), "row_count": len(ids),
    }


def test_render_publishes_one_complete_root_and_cleans_up_on_toctou(
        tmp_path, fresh_data_root, monkeypatch):
    import experiments.render_fixed_comparisons as renderer

    ids = np.arange(30, dtype=np.int64)
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pd.DataFrame({"x": ids, "y": ids * 2, "ls_index": ids}).to_parquet(first)
    pd.DataFrame({"x": ids[::-1], "y": -ids, "ls_index": ids}).to_parquet(second)
    namespace = _namespace(ids, "round0005/test-render")

    def spec_for(output):
        return {
            "output_dir": output, "sample_size": 12,
            "comparisons": [{
                "id": "pair", "substrate": "tiny",
                "semantic_id_namespace": namespace,
                "maps": [
                    {"label": "first", "coords": str(first),
                     "semantic_id_namespace": namespace},
                    {"label": "second", "coords": str(second),
                     "semantic_id_namespace": namespace},
                ],
            }],
        }

    successful_out = os.path.join(fresh_data_root, "published")
    successful_spec = tmp_path / "successful.json"
    successful_spec.write_text(json.dumps(spec_for(successful_out)))
    manifest = render(spec_for(successful_out), spec_path=str(successful_spec))
    assert os.path.isfile(os.path.join(successful_out, "render-manifest.json"))
    assert os.path.isfile(manifest["comparisons"][0]["image"]["path"])
    assert oct(os.stat(successful_out).st_mode & 0o777) == "0o555"

    toctou_out = os.path.join(fresh_data_root, "toctou-rejected")
    toctou_spec = tmp_path / "toctou.json"
    toctou_spec.write_text(json.dumps(spec_for(toctou_out)))
    original = renderer._render_into

    def mutate_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        first.write_bytes(first.read_bytes() + b"drift")
        return result

    monkeypatch.setattr(renderer, "_render_into", mutate_after_read)
    with pytest.raises(ValueError, match="TOCTOU"):
        render(spec_for(toctou_out), spec_path=str(toctou_spec))
    assert not os.path.lexists(toctou_out)
    assert not list(Path(fresh_data_root).glob(".toctou-rejected.tmp.*"))

    pd.DataFrame({"x": ids, "y": ids * 2, "ls_index": ids}).to_parquet(first)
    race_out = os.path.join(fresh_data_root, "race-preserved")
    race_spec = tmp_path / "race.json"
    race_spec.write_text(json.dumps(spec_for(race_out)))

    def create_racing_root(*args, **kwargs):
        result = original(*args, **kwargs)
        os.mkdir(race_out)
        return result

    monkeypatch.setattr(renderer, "_render_into", create_racing_root)
    with pytest.raises(FileExistsError, match="racing render output root"):
        render(spec_for(race_out), spec_path=str(race_spec))
    assert os.path.isdir(race_out) and not os.listdir(race_out)
    assert not list(Path(fresh_data_root).glob(".race-preserved.tmp.*"))


def test_exact_three_group_render_spec_is_historical_and_content_bound(
        fresh_data_root):
    ids = np.arange(8, dtype=np.int64)
    registry = []
    historical_comparisons = []
    for index in range(3):
        run_dir = os.path.join(fresh_data_root, f"registered-{index}")
        os.mkdir(run_dir)
        coords = os.path.join(run_dir, "coords.parquet")
        config = os.path.join(run_dir, "config.yaml")
        results = os.path.join(run_dir, "results.json")
        pd.DataFrame({"x": ids + index, "y": -ids,
                      "ls_index": ids}).to_parquet(coords)
        Path(config).write_text(f"comparison: {index}\n")
        Path(results).write_text(json.dumps({"comparison": index}))
        comparison_id = f"registered-{index}"
        label = f"map-{index}"
        registry.append({
            "id": comparison_id, "title": f"Registered {index}",
            "substrate": f"substrate-{index}", "maps": ((label, run_dir),),
        })
        historical_comparisons.append({
            "comparison_id": comparison_id,
            "maps": [{
                "label": label, "coordinate_signature": path_signature(coords),
                "source_signatures": [path_signature(config), path_signature(results)],
            }],
        })
    historical_path = os.path.join(fresh_data_root, "historical-manifest.json")
    Path(historical_path).write_text(json.dumps({"comparisons": historical_comparisons}))
    # This historical helper intentionally validates a Round-0005 path.  Keep
    # it a path-only value: all actual input fixtures remain below the current
    # BASEMAP_TEST_DATA_PARENT and this location is never created or written.
    output = os.path.join(
        "/data/latent-basemap/runs/round-0005/round0014-path-only",
        os.path.basename(fresh_data_root), "fixed-render-output")
    assert not os.path.lexists(output)
    spec = build_round0005_fixed_spec(
        output_dir=output, registry=tuple(registry),
        historical_manifest_path=historical_path)
    assert validate_round0005_fixed_spec(
        spec, registry=tuple(registry), historical_manifest_path=historical_path) == spec
    assert [item["id"] for item in spec["comparisons"]] == [
        "registered-0", "registered-1", "registered-2"]
    mutated = copy.deepcopy(spec)
    mutated["comparisons"][1]["maps"][0]["label"] = "wrong"
    with pytest.raises(ValueError, match="registered map changed"):
        validate_round0005_fixed_spec(
            mutated, registry=tuple(registry), historical_manifest_path=historical_path)
    assert not os.path.lexists(output)

    assert [item["id"] for item in ROUND0005_FIXED_COMPARISONS] == [
        "g1_fixed_pair", "o1_fixed_six_map", "o2_fixed_controls_and_sparse"]
