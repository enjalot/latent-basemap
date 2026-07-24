import json
import os

import pytest


def _seal(body):
    from basemap.artifact_identity import canonical_json, sha256_bytes

    return {**body, "identity_sha256": sha256_bytes(canonical_json(body))}


def _write_json(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_seal(body), sort_keys=True))


def _quality(density):
    return {
        "ffr": 0.46,
        "density": density,
        "purity_k256": 1.10,
        "purity_k1024": 0.91,
        "recall_at_10": 0.0037,
        "recall_at_50": 0.0045,
        "projection_ffr": 0.42,
    }


def _cell_files(root, label, *, updates, config_sha, graph_sha, density):
    from basemap.round0039_program import (
        CAP_SHA256,
        GRAPH_EFFECTIVE_EDGES,
        GRAPH_RESIDENT_EDGES,
        train_config_for_arm,
    )
    from basemap.round0030_program import (
        train_config_for_arm as control_config,
    )
    from basemap.artifact_identity import expected_input_signature

    if label == "u500k_control":
        config, _ = control_config("uniform")
    else:
        config, _ = train_config_for_arm(label)
    model = root / label / "model.pt"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(label.encode())
    stats = {
        "positive_lr_optimizer_steps": updates,
        "optimizer_steps_succeeded": updates,
        "scheduler_steps": updates,
        "lr_horizon": updates,
        "budget_satisfied": True,
        "n_pos_edges": GRAPH_EFFECTIVE_EDGES,
        "pipeline_multiplicity_cap_artifact_sha256": CAP_SHA256,
        "pipeline_multiplicity_positive_edges_resident": GRAPH_RESIDENT_EDGES,
        "pipeline_multiplicity_positive_edges_effective": GRAPH_EFFECTIVE_EDGES,
        "verified_hashes": {"graph_sha256": graph_sha},
        "updates_per_s": 115.0,
    }
    for key, value in config["execution"]["expected_pipeline_stamp"].items():
        stats[f"pipeline_{key}"] = value
    train = root / label / "train-receipt.json"
    _write_json(train, {
        "production_config_sha256": config_sha,
        "train_stats": stats,
        "model": expected_input_signature(str(model)),
    })
    quality = _quality(density)
    panel = root / label / "panel.json"
    _write_json(panel, {
        "production_config_sha256": config_sha,
        "panel": {
            "ffr": quality["ffr"],
            "density": quality["density"],
            "purity": {
                "k256": quality["purity_k256"],
                "k1024": quality["purity_k1024"],
            },
        },
        "recall@10": quality["recall_at_10"],
        "recall@50": quality["recall_at_50"],
        "projection": {"proj_ffr": quality["projection_ffr"]},
        "selector_passed": True,
    })
    return train, panel


def test_budget_configs_change_only_registered_horizon_fields():
    from basemap.round0039_program import (
        ARMS,
        UPDATES_BY_ARM,
        train_config_for_arm,
    )

    assert ARMS == ("u250k", "u1000k")
    values = {}
    for arm in ARMS:
        config, digest = train_config_for_arm(arm)
        assert len(digest) == 64
        assert config["optimizer"]["successful_positive_lr_updates"] == \
            UPDATES_BY_ARM[arm]
        assert config["optimizer"]["weighted_edge_sampling"] is False
        assert config["execution"]["round0039_budget_arm"] == arm
        assert config["execution"]["expected_pipeline_stamp"][
            "positive_sampling"] == "uniform"
        values[arm] = config
    left = json.loads(json.dumps(values["u250k"]))
    right = json.loads(json.dumps(values["u1000k"]))
    for config in (left, right):
        config["schema"] = None
        config["phrase"] = None
        config["optimizer"]["successful_positive_lr_updates"] = None
        config["execution"]["round0039_budget_arm"] = None
    assert left == right


def test_legacy_handler_reads_dynamic_horizon_and_preserves_r0030():
    from experiments import run_round0014_node as node

    node.configure_round0039(job={"arm": "u250k"})
    assert node._successful_update_horizon() == 250_000
    node.configure_round0039(job={"arm": "u1000k"})
    assert node._successful_update_horizon() == 1_000_000
    node.configure_round0030(job={"arm": "uniform"})
    assert node._successful_update_horizon() == 500_000


def test_queue_runs_both_trains_before_evaluation(tmp_path):
    from experiments.prepare_round0039_queue import (
        GPU_P90_SECONDS,
        _jobs,
    )

    jobs = _jobs(
        artifacts=str(tmp_path / "artifacts"),
        templates={
            "u250k": "/fixture/u250.json",
            "u1000k": "/fixture/u1000.json",
        },
        inputs=[],
    )
    assert len(jobs) == 10
    assert [job["id"] for job in jobs[:3]] == [
        "sampler_canary", "u250k_train_30m", "u1000k_train_30m"]
    assert jobs[0]["handler_callable"] == "run_sampler_canary"
    assert jobs[-1]["handler_callable"] == "run_budget_response"
    assert jobs[-1]["node_policy"]["gpu_required"] is False
    assert sum(
        job["node_policy"]["training_performed"] for job in jobs) == 2
    gpu_p90 = sum(
        job["p90_wall_s"]
        for job in jobs
        if job["node_policy"]["gpu_required"]
    )
    assert gpu_p90 == GPU_P90_SECONDS
    assert gpu_p90 / 3600 == pytest.approx(5.3611111111)


def test_budget_response_classifies_material_density_slope(
    tmp_path, monkeypatch
):
    from basemap.artifact_identity import expected_input_signature
    from basemap.round0030_program import (
        train_config_for_arm as control_config,
    )
    from basemap.round0039_program import train_config_for_arm
    from experiments import round0039_nodes as nodes

    graph = tmp_path / "graph.npz"
    graph.write_bytes(b"graph")
    graph_sha = expected_input_signature(str(graph))["sha256"]
    monkeypatch.setattr(nodes, "GRAPH_PATH", str(graph))
    monkeypatch.setattr(nodes, "GRAPH_SHA256", graph_sha)

    control_cfg, control_sha = control_config("uniform")
    del control_cfg
    control_train, control_panel = _cell_files(
        tmp_path, "u500k_control", updates=500_000,
        config_sha=control_sha, graph_sha=graph_sha, density=0.79)
    cell_paths = {}
    for arm, updates, density in (
        ("u250k", 250_000, 0.82),
        ("u1000k", 1_000_000, 0.74),
    ):
        _, config_sha = train_config_for_arm(arm)
        train, panel = _cell_files(
            tmp_path, arm, updates=updates, config_sha=config_sha,
            graph_sha=graph_sha, density=density)
        cell_paths[arm] = {
            "train": str(train.parent),
            "panel": str(panel.parent),
        }
    scale_panel = tmp_path / "r36-panel.json"
    _write_json(scale_panel, {
        "round_id": "0036",
        "panel": {"density": 0.0933},
    })
    response = nodes.build_budget_response(
        release_sha="a" * 40,
        cell_paths=cell_paths,
        control_paths={
            "train": str(control_train),
            "panel": str(control_panel),
        },
        scale_panel_path=str(scale_panel),
    )
    assert response["classification"] == \
        "monotone-density-degradation-with-update-budget"
    assert response["checks"]["budget_sensitive"] is True
    assert response["checks"][
        "u1000k_density_degradation_exceeds_band"] is True
    assert response["checks"][
        "u250k_density_improvement_exceeds_band"] is True


def test_queue_materialization_requires_issued_frontmatter(
    tmp_path, monkeypatch
):
    from experiments import prepare_round0039_queue as prepare

    round_file = tmp_path / "round.md"
    round_root = tmp_path / "round-root"
    round_file.write_text("---\nstatus: draft\n---\n# Draft\n")
    monkeypatch.setattr(prepare, "ROUND_FILE", str(round_file))
    monkeypatch.setattr(prepare, "ROUND_ROOT", str(round_root))
    with pytest.raises(RuntimeError, match="requires status: issued"):
        prepare.prepare_round0039("a" * 40)
    assert not round_root.exists()
