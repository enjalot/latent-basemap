from __future__ import annotations

import importlib
import inspect

from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0071_substrate import (
    ELIGIBILITY_SUMMARY,
    INTERVALS,
    ROW_COUNT,
    SUBSTRATE_SCHEMA,
)
from basemap.round0075_training import (
    PIPELINE_SCHEMA,
    SAMPLER_CLASS,
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities,
)
from experiments import (
    prepare_round0075_queue,
    round0075_nodes,
)


R0069_SCALE = (
    "/data/latent-basemap/runs/round-0069/queue/artifacts/"
    "scale-comparison/scale-comparison.json"
)
R0069_SCALE_SHA256 = (
    "5eec5ce7135c19bc75044c476b09591950b2d2bc951b79b480074646daa0f587"
)
R0074_ANCHOR = (
    "/data/latent-basemap/runs/round-0074/queue-attempt-2/artifacts/"
    "duplicate-anchor-leverage/duplicate-anchor-leverage.json"
)


def _capability_fixture():
    retained = ELIGIBILITY_SUMMARY["retained_row_count"]
    excluded = ELIGIBILITY_SUMMARY["excluded_row_count"]
    eligibility = {
        "canonical_path": "/data/90m-eligibility.npz",
        "bytes": 123,
        "sha256": "e" * 64,
    }
    substrate_signature = {
        "canonical_path": "/data/90m-substrate.json",
        "bytes": 456,
        "sha256": "d" * 64,
    }
    substrate = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": "0071",
        "tier": "90m",
        "row_count": ROW_COUNT,
        "dimension": 384,
        "global_150m_intervals": [list(value) for value in INTERVALS],
        "outputs": {
            "int8": {
                "canonical_path": "/data/90m.i8",
                "bytes": ROW_COUNT * 384,
                "sha256": "a" * 64,
            },
            "scales": {
                "canonical_path": "/data/90m.f16",
                "bytes": ROW_COUNT * 2,
                "sha256": "b" * 64,
            },
            "eligibility": eligibility,
        },
    }
    graph = {
        "schema": GRAPH_SCHEMA,
        "round_id": "0073",
        "tier": "90m",
        "row_count": ROW_COUNT,
        "input_k": 15,
        "inputs": {
            "eligibility": eligibility,
            "substrate": substrate_signature,
            "gpu_qualification": {"sha256": "f" * 64},
        },
        "summary": {
            "eligibility_excluded_source_count": excluded,
            "eligibility_retained_row_count": retained,
            "retained_positive_source_count": retained,
            "zero_degree_retained_source_count": 0,
            "valid_canonical_edge_count": retained * 15,
            "degree_histogram": {"0": excluded, "15": retained},
        },
        "quality": {"mean_recall_at_15_unambiguous": 0.91},
    }
    return substrate, substrate_signature, graph


def test_balanced_90m_updates_are_coverage_aligned() -> None:
    assert SUCCESSFUL_UPDATES == 1_493_293
    substrate, substrate_signature, graph = _capability_fixture()
    config, config_sha = train_config_from_capabilities(
        graph_manifest=graph,
        graph_manifest_path="/data/90m-graph.json",
        graph_manifest_sha256="c" * 64,
        substrate_manifest=substrate,
        substrate_manifest_path=substrate_signature["canonical_path"],
        substrate_manifest_sha256=substrate_signature["sha256"],
        scale_geometry_signature={"sha256": "1" * 64},
        anchor_leverage_signature={"sha256": "2" * 64},
    )
    assert len(config_sha) == 64
    assert (
        config["optimizer"]["successful_positive_lr_updates"]
        == SUCCESSFUL_UPDATES
    )
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert stamp["schema"] == PIPELINE_SCHEMA
    assert stamp["sampler_class"] == SAMPLER_CLASS
    assert stamp["positive_source_count"] == 88_945_313
    assert stamp["valid_canonical_edge_count"] == 1_334_179_695
    assert config["graph"]["weights_consumed"] is False
    assert config["decision_thresholds"][
        "geometry_claim_requires_downstream_evaluation"
    ] is True
    assert config["execution"]["scale_transition"][
        "density_floor_tuned"
    ] is False


def test_reviewed_scale_and_anchor_evidence_loads() -> None:
    from basemap.artifact_identity import expected_input_signature

    anchor_sha = expected_input_signature(R0074_ANCHOR)["sha256"]
    scale, anchor = round0075_nodes._load_evidence({
        "scale_geometry": R0069_SCALE,
        "scale_geometry_sha256": R0069_SCALE_SHA256,
        "anchor_leverage": R0074_ANCHOR,
        "anchor_leverage_sha256": anchor_sha,
    })
    assert scale["sha256"] == R0069_SCALE_SHA256
    assert anchor["sha256"] == anchor_sha


def test_round0075_is_one_bounded_training_job() -> None:
    source = inspect.getsource(prepare_round0075_queue.prepare_round0075)
    assert source.count('"action": "train_balanced_90m"') == 1
    assert "gpu_hours_cap=6.0" in source
    assert "p90_seconds = 18_900.0" in source
    assert '"standalone_canary": False' in source
    assert '"training_wall_only": True' in source
    assert '"geometry_claim_requires_successor_evaluation": True' in source
    assert (
        'manifest["required_reviews"] = ["0069", "0071", "0073", "0074"]'
        in source
    )


def test_round0075_receipts_exact_training_accounting() -> None:
    source = inspect.getsource(round0075_nodes.run_train)
    for field in (
        "lr_horizon",
        "positive_lr_optimizer_steps",
        "optimizer_steps_succeeded",
        "amp_overflow_skips",
        "nonfinite_loss_skips",
        "nonfinite_gradient_skips",
        "pipeline_runtime",
        "host_prefetch_consumer_batches",
        "scale_geometry",
        "anchor_leverage",
    ):
        assert field in source
    loader = inspect.getsource(round0075_nodes._load_pipeline)
    assert 'graph["manifest"].get("round_id") != "0073"' in loader
    handler = inspect.getsource(round0075_nodes.run_job)
    assert 'active.get("manifest", {}).get("round_id")' in handler


def test_round0075_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0075_training")
    importlib.import_module("experiments.round0075_nodes")
    importlib.import_module("experiments.prepare_round0075_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
