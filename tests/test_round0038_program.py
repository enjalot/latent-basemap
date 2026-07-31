import json
import os

import pytest


def _cells():
    from basemap.round0038_program import DECISION_CELL_LABELS

    cells = {}
    for label in DECISION_CELL_LABELS:
        dimension_text, seed_text = label.split("_")
        dimension = int(dimension_text[1:])
        seed = int(seed_text[1:])
        cells[label] = {
            "dimension": dimension,
            "seed": seed,
            "ffr": 0.60 if dimension == 768 else 0.58,
            "purity_k256": 0.9,
            "purity_k1024": 0.8,
            "density": 0.6,
            "oos_proj_ffr": 0.50 if dimension == 768 else 0.46,
            "updates_per_s": 109.0,
            "peak_reserved_bytes": 4_000_000_000,
        }
    return cells


def test_configs_bind_seed43_and_reviewed_performance_contract():
    from basemap.round0038_program import (
        CELL_LABELS,
        GRAPH_SHA256,
        train_config_for_cell,
    )

    assert CELL_LABELS == ("d768_s43", "d384_s43")
    for label in CELL_LABELS:
        config, digest = train_config_for_cell(
            label,
            graph_manifest_path=f"/data/fixture/{label}.json",
            graph_manifest_sha256="a" * 64,
        )
        assert len(digest) == 64
        assert config["graph"]["sha256"] == GRAPH_SHA256
        assert config["optimizer"]["seed"] == 43
        assert config["optimizer"]["successful_positive_lr_updates"] == 500_000
        assert config["execution"]["minimum_train_upd_s"] == 75.0
        assert config["execution"]["warning_train_upd_s"] == 90.0
        assert config["execution"]["expected_pipeline_stamp"][
            "weighted_effective"] is True


def test_registered_rule_adopts_384_at_equal_boundaries():
    from basemap.round0038_program import build_registered_decision

    cells = _cells()
    cells["d768_s42"]["ffr"] = 0.60
    cells["d768_s43"]["ffr"] = 0.58
    for seed in (42, 43):
        cells[f"d384_s{seed}"]["ffr"] = 0.57
        cells[f"d384_s{seed}"]["oos_proj_ffr"] = 0.45
    decision = build_registered_decision(cells)
    assert decision["control_768_ffr_seed_spread_max_minus_min"] == \
        pytest.approx(0.02)
    assert decision["qualification"]["384"]["checks"] == {
        "oos_at_least_90pct_control": True,
        "transductive_within_control_seed_spread": True,
    }
    assert decision["decision"] == "adopt-384d"
    assert decision["adopted_input_dimension"] == 384


def test_registered_rule_rejects_either_failed_check():
    from basemap.round0038_program import build_registered_decision

    cells = _cells()
    cells["d384_s42"]["oos_proj_ffr"] = 0.40
    cells["d384_s43"]["oos_proj_ffr"] = 0.40
    decision = build_registered_decision(cells)
    assert decision["decision"] == "reject-384d"
    assert decision["adopted_input_dimension"] is None

    cells = _cells()
    cells["d768_s42"]["ffr"] = 0.60
    cells["d768_s43"]["ffr"] = 0.60
    cells["d384_s42"]["ffr"] = 0.59
    cells["d384_s43"]["ffr"] = 0.59
    decision = build_registered_decision(cells)
    assert decision["qualification"]["384"]["checks"][
        "transductive_within_control_seed_spread"] is False


def test_queue_reuses_reference_and_has_two_training_cells(tmp_path):
    from basemap.round0038_program import (
        CANARY_MINIMUM_UPDATES_PER_S,
        CELL_LABELS,
        DIMENSIONS,
        validate_job_cell,
    )
    from experiments.prepare_round0038_queue import (
        GPU_P90_SECONDS,
        R0037_SHARED_ROOT,
        _jobs,
    )

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
    assert len(jobs) == 8
    assert jobs[0]["handler"] == "round0038_sampler_canary"
    assert jobs[0]["cell"] == "d768_s43"
    assert jobs[0]["minimum_updates_per_s"] == \
        CANARY_MINIMUM_UPDATES_PER_S
    assert not any("shared_reference" in job["handler"] for job in jobs)
    assert sum(job["handler"] == "round0038_train" for job in jobs) == 2
    assert sum(job["handler"] == "round0038_transform" for job in jobs) == 2
    assert sum(job["handler"] == "round0038_score" for job in jobs) == 2
    assert all(
        job.get("shared_reference_output", R0037_SHARED_ROOT)
        == R0037_SHARED_ROOT
        for job in jobs
    )
    assert jobs[-1]["handler"] == "round0038_decision"
    assert jobs[-1]["node_policy"]["gpu_required"] is False
    assert set(jobs[-1]["cell_outputs"]) == set(CELL_LABELS)
    gpu_p90 = sum(
        job["p90_wall_s"]
        for job in jobs
        if job["node_policy"]["gpu_required"]
    )
    assert gpu_p90 == GPU_P90_SECONDS
    assert gpu_p90 / 3600 == pytest.approx(3.1333333333)
    assert all(
        validate_job_cell(job)["label"] == job["cell"]
        for job in jobs
        if "cell" in job
    )


def test_queue_cells_survive_json_roundtrip(tmp_path):
    from basemap.round0038_program import DIMENSIONS, validate_job_cell
    from experiments.prepare_round0038_queue import _jobs

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
    roundtripped = json.loads(json.dumps(jobs, sort_keys=True))
    assert all(
        validate_job_cell(job)["label"] == job["cell"]
        for job in roundtripped
        if "cell" in job
    )


def test_handlers_bind_seed43_canary_and_resolve(tmp_path):
    from basemap.round0038_program import DIMENSIONS
    from experiments.prepare_round0038_queue import _jobs
    from experiments import round0038_nodes

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
    assert all(
        job["handler_module"] == "experiments.round0038_nodes"
        and callable(getattr(round0038_nodes, job["handler_callable"]))
        for job in jobs
    )
    round0038_nodes._configure_inherited_handlers()
    assert round0038_nodes.inherited.CANARY_CELL_LABEL == "d768_s43"


def test_queue_materialization_requires_issued_frontmatter(
    tmp_path, monkeypatch
):
    from experiments import prepare_round0038_queue as prepare

    round_file = tmp_path / "round.md"
    round_root = tmp_path / "round-root"
    round_file.write_text("---\nstatus: draft\n---\n# Draft\n")
    monkeypatch.setattr(prepare, "ROUND_FILE", str(round_file))
    monkeypatch.setattr(prepare, "ROUND_ROOT", str(round_root))
    with pytest.raises(RuntimeError, match="requires status: issued"):
        prepare.prepare_round0038("a" * 40)
    assert not round_root.exists()

    round_file.write_text("---\nstatus: issued\n---\n# Issued\n")
    prepare._require_issued_round(str(round_file))


def test_failed_canary_persists_sealed_verdict_before_reraising(
    tmp_path, monkeypatch
):
    from basemap.artifact_identity import canonical_json, sha256_bytes
    from experiments import round0038_nodes

    output = tmp_path / "canary"

    def fail_after_output(active, job):
        os.mkdir(job["outputs"][0])
        raise RuntimeError("synthetic seed43 canary regression")

    monkeypatch.setattr(
        round0038_nodes.inherited, "run_sampler_canary", fail_after_output)
    job = {
        "outputs": [str(output)],
        "cell": "d768_s43",
        "production_config_sha256": "b" * 64,
        "minimum_updates_per_s": 90.0,
    }
    active = {"manifest": {"round_id": "0038"}}
    with pytest.raises(RuntimeError, match="synthetic seed43 canary regression"):
        round0038_nodes.run_sampler_canary(active, job)

    verdict_path = output / "verdict.json"
    verdict = json.loads(verdict_path.read_text())
    body = {
        key: value for key, value in verdict.items()
        if key != "identity_sha256"
    }
    assert verdict["identity_sha256"] == sha256_bytes(canonical_json(body))
    assert verdict["passed"] is False
    assert verdict["round_id"] == "0038"
    assert verdict["cell"] == "d768_s43"
    assert verdict["exception_type"] == "RuntimeError"
