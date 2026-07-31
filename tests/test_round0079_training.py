from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0065_substrates import SUBSTRATE_SCHEMA
from basemap.round0079_training import (
    ELIGIBILITY_SUMMARY,
    INTERVALS,
    PERFORMANCE_WINDOWS,
    PERFORMANCE_WINDOW_UPDATES_MAX,
    PERFORMANCE_WARMUP_UPDATES,
    PIPELINE_SCHEMA,
    ROW_COUNT,
    SAMPLER_CLASS,
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities,
)
from experiments import (
    prepare_round0079_queue,
    round0079_nodes,
)


def _capability_fixture():
    retained = ELIGIBILITY_SUMMARY["retained_row_count"]
    excluded = ELIGIBILITY_SUMMARY["excluded_row_count"]
    eligibility = {
        "canonical_path": "/data/120m-eligibility.npz",
        "bytes": 123,
        "sha256": "e" * 64,
    }
    substrate_signature = {
        "canonical_path": "/data/120m-substrate.json",
        "bytes": 456,
        "sha256": "d" * 64,
    }
    substrate = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": "0065",
        "tier": "120m",
        "row_count": ROW_COUNT,
        "dimension": 384,
        "global_150m_intervals": [list(value) for value in INTERVALS],
        "outputs": {
            "int8": {
                "canonical_path": "/data/120m.i8",
                "bytes": ROW_COUNT * 384,
                "sha256": "a" * 64,
            },
            "scales": {
                "canonical_path": "/data/120m.f16",
                "bytes": ROW_COUNT * 2,
                "sha256": "b" * 64,
            },
            "eligibility": eligibility,
        },
    }
    graph = {
        "schema": GRAPH_SCHEMA,
        "round_id": "0078",
        "tier": "120m",
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


def test_balanced_120m_updates_are_coverage_aligned() -> None:
    assert SUCCESSFUL_UPDATES == 1_982_221
    assert PERFORMANCE_WARMUP_UPDATES == 200
    assert PERFORMANCE_WINDOW_UPDATES_MAX == 2_500
    assert PERFORMANCE_WINDOWS == 793
    assert (
        2 * PERFORMANCE_WINDOW_UPDATES_MAX / 80.0
        <= 63.0
    )
    substrate, substrate_signature, graph = _capability_fixture()
    config, config_sha = train_config_from_capabilities(
        graph_manifest=graph,
        graph_manifest_path="/data/120m-graph.json",
        graph_manifest_sha256="c" * 64,
        substrate_manifest=substrate,
        substrate_manifest_path=substrate_signature["canonical_path"],
        substrate_manifest_sha256=substrate_signature["sha256"],
        scale_geometry_signature={"sha256": "1" * 64},
        anchor_leverage_signature={"sha256": "2" * 64},
        policy_confirmation_signature={"sha256": "3" * 64},
    )
    assert len(config_sha) == 64
    assert (
        config["optimizer"]["successful_positive_lr_updates"]
        == SUCCESSFUL_UPDATES
    )
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert stamp["schema"] == PIPELINE_SCHEMA
    assert stamp["sampler_class"] == SAMPLER_CLASS
    assert stamp["positive_source_count"] == 118_067_492
    assert stamp["valid_canonical_edge_count"] == 1_771_012_380
    assert config["scorer"]["required"][0] == (
        "matched retained-90M scale comparison"
    )
    assert config["graph"]["weights_consumed"] is False
    assert config["graph"]["independent_policy_confirmation"] == {
        "sha256": "3" * 64
    }
    assert config["decision_thresholds"][
        "geometry_claim_requires_downstream_evaluation"
    ] is True
    assert config["execution"]["scale_transition"][
        "density_floor_tuned"
    ] is False


def test_reviewed_90m_scale_evidence_loads(
    tmp_path: Path,
) -> None:
    anchor = {
        "canonical_path": "/data/anchor.json",
        "bytes": 123,
        "sha256": "a" * 64,
    }
    body = {
        "schema": "round0076-scale-geometry-comparison-v1",
        "round_id": "0076",
        "same_row_30m_comparison": {"passed": True},
        "full_90m_non_density_checks_passed": True,
        "density_semantics": {
            "selector": "relative-noninferiority-only",
            "legacy_absolute_floor_used_for_decision": False,
            "threshold_calibrated": False,
            "anchor_leverage_evidence": anchor,
        },
        "decision": {
            "90m_supported_as_deliberate_ladder_rung": True,
            "prepare_120m_search_and_graph_if_true": True,
            "train_120m_without_separate_round": False,
        },
    }
    receipt = {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }
    path = tmp_path / "scale-comparison.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    from basemap.artifact_identity import expected_input_signature

    signature = expected_input_signature(str(path))
    scale, loaded_anchor = round0079_nodes._load_scale_evidence({
        "scale_geometry": str(path),
        "scale_geometry_sha256": signature["sha256"],
    })
    assert scale == signature
    assert loaded_anchor == anchor


def test_round0079_is_one_bounded_training_job() -> None:
    source = inspect.getsource(prepare_round0079_queue.prepare_round0079)
    assert source.count('"action": "train_balanced_120m"') == 1
    assert "gpu_hours_cap=7.0" in source
    assert "p90_seconds = 25_200.0" in source
    assert '"standalone_canary": False' in source
    assert '"training_wall_only": True' in source
    assert '"geometry_claim_requires_successor_evaluation": True' in source
    assert (
        'manifest["required_reviews"] = ["0065", "0076", "0078", "0082"]'
        in source
    )
    assert "minilm-balanced-120m-gpu-ivfpq-search-confirmed-v1" in source


def test_round0079_discovers_the_one_issued_dated_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    draft = tmp_path / "round-0079-2026-07-27.md"
    issued = tmp_path / "round-0079-2026-07-28.md"
    draft.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    issued.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    monkeypatch.setattr(
        prepare_round0079_queue,
        "ROUND_FILE_GLOB",
        str(tmp_path / "round-0079-*.md"),
    )
    assert prepare_round0079_queue._require_issued_round() == str(issued)
    draft.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    try:
        prepare_round0079_queue._require_issued_round()
    except RuntimeError as exc:
        assert "exactly one issued round document" in str(exc)
    else:
        raise AssertionError("multiple issued R0079 contracts were accepted")


def test_round0079_receipts_exact_training_accounting() -> None:
    source = inspect.getsource(round0079_nodes.run_train)
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
        "policy_confirmation",
    ):
        assert field in source
    loader = inspect.getsource(round0079_nodes._load_pipeline)
    assert 'graph["manifest"].get("round_id") != "0078"' in loader
    assert "load_policy_confirmation" in loader
    handler = inspect.getsource(round0079_nodes.run_job)
    assert 'active.get("manifest", {}).get("round_id")' in handler


def test_round0079_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0079_training")
    importlib.import_module("experiments.round0079_nodes")
    importlib.import_module("experiments.prepare_round0079_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
