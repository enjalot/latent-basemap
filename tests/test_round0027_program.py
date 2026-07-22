import copy

import numpy as np
import pytest


def _cells():
    from basemap.round0027_program import CELL_LABELS, parse_cell

    out = {}
    for label in CELL_LABELS:
        dimension, seed = parse_cell(label)
        out[label] = {
            "dimension": dimension,
            "seed": seed,
            "ffr": 0.60 if dimension == 768 else 0.59,
            "purity_k256": 0.9,
            "purity_k1024": 0.8,
            "density": 0.5,
            "oos_proj_ffr": 0.50 if dimension == 768 else 0.46,
            "updates_per_s": 300.0,
            "peak_reserved_bytes": 8_000_000_000,
        }
    out["d768_s43"]["ffr"] = 0.58
    return out


def test_query_ids_are_fixed_unique_and_strictly_outside_train_prefix():
    from basemap.round0027_program import QUERY_ROWS, ROWS, query_row_ids

    first = query_row_ids()
    second = query_row_ids()
    assert first.dtype == np.dtype("int64")
    assert first.shape == (QUERY_ROWS,)
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.unique(first))
    assert first[0] >= ROWS
    assert first[-1] < 4_000_000


def test_cosine_truth_view_renormalizes_all_768_dimensions():
    from basemap.round0027_program import cosine_truth_array

    truth = cosine_truth_array()
    rows = truth[[0, 1, 1_999_999]]
    assert rows.shape == (3, 768)
    assert np.allclose(np.linalg.norm(rows, axis=1), 1.0, atol=2e-6)
    assert truth.execution_preprocessing_stamp[
        "input_l2_renormalized"] is True


def test_payload_prefix_proof_excludes_different_npy_headers(tmp_path):
    from experiments.round0027_nodes import _npy_payload_sha256

    rng = np.random.RandomState(5)
    prefix = rng.normal(size=(2, 768)).astype(np.float16)
    larger = np.concatenate(
        [prefix, rng.normal(size=(2, 768)).astype(np.float16)], axis=0)
    small_path = tmp_path / "small.npy"
    large_path = tmp_path / "large.npy"
    np.save(small_path, prefix)
    np.save(large_path, larger)
    small = _npy_payload_sha256(str(small_path), rows=2)
    large = _npy_payload_sha256(str(large_path), rows=2)
    assert small["payload_sha256"] == large["payload_sha256"]
    assert small["file_sha256"] != large["file_sha256"]
    assert small["payload_bytes_hashed"] == 2 * 768 * 2


def test_downstream_signature_verification_rejects_mutated_artifact(tmp_path):
    from basemap.artifact_identity import expected_input_signature
    from experiments.round0027_nodes import _verified_signature_path

    path = tmp_path / "artifact.bin"
    path.write_bytes(b"first")
    signature = expected_input_signature(path)
    assert _verified_signature_path(signature, label="fixture") == str(path)
    path.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="content changed"):
        _verified_signature_path(signature, label="fixture")


def test_six_configs_bind_only_registered_dimension_seed_and_preprocessing():
    from basemap.round0027_program import (
        CELL_LABELS,
        GRAPH_PATH,
        GRAPH_SHA256,
        train_config_for_cell,
    )

    configs = {}
    for label in CELL_LABELS:
        config, digest = train_config_for_cell(
            label,
            graph_manifest_path=f"/data/fixture/{label[:4]}.json",
            graph_manifest_sha256="a" * 64,
        )
        assert len(digest) == 64
        assert config["graph"]["path"] == GRAPH_PATH
        assert config["graph"]["sha256"] == GRAPH_SHA256
        assert config["graph"]["directed_edges"] == 149_061_552
        assert config["optimizer"]["successful_positive_lr_updates"] == 500_000
        assert config["execution"]["expected_pipeline_stamp"][
            "weighted_effective"] is True
        assert config["execution"]["expected_pipeline_stamp"][
            "input_effective_dimension"] == config["model"]["input_dimension"]
        configs[label] = config
    assert configs["d768_s42"]["row_universe"]["input_preprocessing"][
        "l2_renormalized"] is False
    assert configs["d384_s42"]["row_universe"]["input_preprocessing"][
        "l2_renormalized"] is True
    assert configs["d256_s43"]["optimizer"]["seed"] == 43


def test_graph_adapter_separates_network_input_from_full_768d_truth():
    from basemap.round0027_program import (
        GRAPH_SHA256,
        TRAIN_SHA256,
        graph_manifest_for_dimension,
    )

    reduced = graph_manifest_for_dimension(256)
    truth = reduced["graph_construction_truth"]
    assert reduced["graph_sha256"] == GRAPH_SHA256
    assert reduced["metric"] == "cosine"
    assert reduced["metric_input"] == "full_768d_source"
    assert reduced["input_preprocessing"]["input_effective_dimension"] == 256
    assert reduced["model_input_alignment"]["data_fingerprint"] == \
        reduced["data_fingerprint"]
    assert truth["source_sha256"] == TRAIN_SHA256
    assert truth["source_dimension"] == 768
    assert truth["input_preprocessing"]["input_effective_dimension"] == 768
    assert truth["reduced_dimension_graph_rebuilt"] is False


def test_registered_decision_selects_smallest_qualifying_dimension():
    from basemap.round0027_program import build_registered_decision

    cells = _cells()
    decision = build_registered_decision(cells)
    assert decision["decision"] == "adopt-256d"
    assert decision["adopted_input_dimension"] == 256
    assert decision["control_768_ffr_seed_spread_max_minus_min"] == \
        pytest.approx(0.02)
    assert decision["qualification"]["256"]["qualified"] is True

    only_384 = copy.deepcopy(cells)
    only_384["d256_s42"]["ffr"] = 0.50
    only_384["d256_s43"]["ffr"] = 0.50
    assert build_registered_decision(only_384)[
        "adopted_input_dimension"] == 384

    neither = copy.deepcopy(only_384)
    neither["d384_s42"]["oos_proj_ffr"] = 0.40
    neither["d384_s43"]["oos_proj_ffr"] = 0.40
    assert build_registered_decision(neither)["decision"] == \
        "reject-all-reduced-dimensions"


def test_registered_decision_uses_control_max_minus_min_not_layout_metric():
    from basemap.round0027_program import build_registered_decision

    cells = _cells()
    # Control mean=.59, spread=.02, so the exact transductive floor is .57.
    for seed in (42, 43):
        cells[f"d256_s{seed}"]["ffr"] = 0.57
        cells[f"d256_s{seed}"]["oos_proj_ffr"] = 0.45
    decision = build_registered_decision(cells)
    assert decision["qualification"]["256"]["qualified"] is True
    assert decision["qualification"]["256"][
        "transductive_ffr_floor"] == pytest.approx(0.57)


def test_queue_is_one_canary_one_shared_reference_six_cells_and_cpu_decision(tmp_path):
    from basemap.round0027_program import DIMENSIONS, validate_job_cell
    from experiments.prepare_round0027_queue import _jobs

    manifests = {
        dimension: {
            "path": f"/data/fixture/graph-manifest-d{dimension}.json",
            "sha256": str(dimension).zfill(64),
        }
        for dimension in DIMENSIONS
    }
    jobs = _jobs(
        artifacts=str(tmp_path / "artifacts"),
        inputs=[],
        manifests=manifests,
        query_ids_path="/data/fixture/query-ids.npy",
    )
    assert len(jobs) == 21
    assert jobs[0]["handler"] == "round0027_sampler_canary"
    assert jobs[1]["handler"] == "round0027_shared_reference"
    assert sum(job["handler"] == "round0027_train" for job in jobs) == 6
    assert sum(job["handler"] == "round0027_transform" for job in jobs) == 6
    assert sum(job["handler"] == "round0027_score" for job in jobs) == 6
    assert jobs[-1]["handler"] == "round0027_decision"
    assert jobs[-1]["node_policy"]["gpu_required"] is False
    assert all(
        "shared_reference_output" in job
        for job in jobs if job["handler"] == "round0027_transform"
    )
    gpu_p90 = sum(
        job["p90_wall_s"] for job in jobs
        if job["node_policy"]["gpu_required"])
    assert gpu_p90 == 18_140.0
    assert all(
        job["handler_module"] == "experiments.round0027_nodes"
        for job in jobs
    )
    from experiments import round0027_nodes

    assert all(
        job["handler_callable"].startswith("run_")
        and callable(getattr(round0027_nodes, job["handler_callable"]))
        for job in jobs
    )
    assert all(
        validate_job_cell(job)["label"] == job["cell"]
        for job in jobs if "cell" in job
    )
