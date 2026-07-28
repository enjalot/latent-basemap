from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0086_program import (
    EXCLUDED_ROWS,
    RETAINED_ROWS,
    ROW_COUNT,
    SUBSTRATE_SCHEMA,
)
from basemap.round0092_training import (
    MINIMUM_UPDATES_PER_SECOND,
    PERFORMANCE_WINDOWS,
    PERFORMANCE_WINDOW_UPDATES_MAX,
    PERFORMANCE_WARMUP_UPDATES,
    PIPELINE_SCHEMA,
    SAMPLER_CLASS,
    SUCCESSFUL_UPDATES,
    train_config_from_capabilities,
)
from experiments import prepare_round0092_queue, round0092_nodes


def _capability_fixture():
    eligibility = {
        "canonical_path": "/data/150m-eligibility.npz",
        "bytes": 123,
        "sha256": "e" * 64,
    }
    substrate_signature = {
        "canonical_path": "/data/150m-substrate.json",
        "bytes": 456,
        "sha256": "d" * 64,
    }
    substrate = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": "0086",
        "tier": "150m",
        "row_count": ROW_COUNT,
        "dimension": 384,
        "global_150m_intervals": [[0, ROW_COUNT]],
        "outputs": {
            "int8": {
                "canonical_path": "/data/150m.i8",
                "bytes": ROW_COUNT * 384,
                "sha256": "a" * 64,
            },
            "scales": {
                "canonical_path": "/data/150m.f16",
                "bytes": ROW_COUNT * 2,
                "sha256": "b" * 64,
            },
            "eligibility": eligibility,
        },
    }
    quality = {
        "mean_recall_at_15_unambiguous": 0.91,
        "floor": 0.90,
        "qualification_sample_rows": 4_096,
        "qualification_sample_seed": 86,
    }
    graph = {
        "schema": GRAPH_SCHEMA,
        "round_id": "0091",
        "row_count": ROW_COUNT,
        "input_k": 15,
        "inputs": {
            "eligibility": eligibility,
            "substrate": substrate_signature,
            "parts": {
                "fineweb": {"sha256": "1" * 64},
                "redpajama": {"sha256": "2" * 64},
                "pile": {"sha256": "3" * 64},
            },
        },
        "summary": {
            "eligibility_excluded_source_count": EXCLUDED_ROWS,
            "eligibility_retained_row_count": RETAINED_ROWS,
            "retained_positive_source_count": RETAINED_ROWS,
            "zero_degree_retained_source_count": 0,
            "valid_canonical_edge_count": RETAINED_ROWS * 15,
            "degree_histogram": {
                "0": EXCLUDED_ROWS,
                "15": RETAINED_ROWS,
            },
        },
        "quality": quality,
    }
    return substrate, substrate_signature, graph


def test_balanced_150m_updates_are_coverage_aligned_and_bounded() -> None:
    assert SUCCESSFUL_UPDATES == 2_471_689
    assert PERFORMANCE_WARMUP_UPDATES == 200
    assert PERFORMANCE_WINDOW_UPDATES_MAX == 2_500
    assert PERFORMANCE_WINDOWS == 989
    assert (
        2 * PERFORMANCE_WINDOW_UPDATES_MAX / MINIMUM_UPDATES_PER_SECOND
        <= 50.0
    )
    assert SUCCESSFUL_UPDATES / MINIMUM_UPDATES_PER_SECOND < 7 * 3600
    substrate, substrate_signature, graph = _capability_fixture()
    config, config_sha = train_config_from_capabilities(
        graph_manifest=graph,
        graph_manifest_path="/data/150m-graph.json",
        graph_manifest_sha256="c" * 64,
        substrate_manifest=substrate,
        substrate_manifest_path=substrate_signature["canonical_path"],
        substrate_manifest_sha256=substrate_signature["sha256"],
        scale_geometry_signature={"sha256": "4" * 64},
    )
    assert len(config_sha) == 64
    assert (
        config["optimizer"]["successful_positive_lr_updates"]
        == SUCCESSFUL_UPDATES
    )
    stamp = config["execution"]["expected_pipeline_stamp"]
    assert stamp["schema"] == PIPELINE_SCHEMA
    assert stamp["sampler_class"] == SAMPLER_CLASS
    assert stamp["positive_source_count"] == RETAINED_ROWS
    assert stamp["valid_canonical_edge_count"] == RETAINED_ROWS * 15
    assert config["execution"]["minimum_train_upd_s"] == 100.0
    assert config["graph"]["weights_consumed"] is False
    assert config["execution"]["duplicate_control"][
        "full_150m_family_census_reused"
    ] is True
    assert config["decision_thresholds"][
        "geometry_claim_requires_downstream_evaluation"
    ] is True


def test_graph_quality_and_complete_counts_fail_closed() -> None:
    substrate, substrate_signature, graph = _capability_fixture()
    graph["quality"]["mean_recall_at_15_unambiguous"] = 0.899
    with pytest.raises(RuntimeError, match="geometry changed"):
        train_config_from_capabilities(
            graph_manifest=graph,
            graph_manifest_path="/data/150m-graph.json",
            graph_manifest_sha256="c" * 64,
            substrate_manifest=substrate,
            substrate_manifest_path=substrate_signature["canonical_path"],
            substrate_manifest_sha256=substrate_signature["sha256"],
            scale_geometry_signature={"sha256": "4" * 64},
        )


def test_reviewed_120m_scale_evidence_loads(tmp_path: Path) -> None:
    body = {
        "schema": "round0080-scale-geometry-comparison-v1",
        "round_id": "0080",
        "same_row_90m_comparison": {"passed": True},
        "full_120m_non_density_checks_passed": True,
        "density_semantics": {
            "selector": "relative-noninferiority-only",
            "legacy_absolute_floor_used_for_decision": False,
            "threshold_calibrated": False,
        },
        "decision": {
            "120m_supported_as_deliberate_ladder_rung": True,
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
    loaded = round0092_nodes._load_scale_evidence({
        "scale_geometry": str(path),
        "scale_geometry_sha256": signature["sha256"],
    })
    assert loaded == signature


def test_round0092_is_one_bounded_training_job() -> None:
    source = inspect.getsource(prepare_round0092_queue.prepare_round0092)
    assert source.count('"action": "train_balanced_150m"') == 1
    assert "gpu_hours_cap=8.0" in source
    assert "p90_seconds = 27_000.0" in source
    assert '"standalone_canary": False' in source
    assert '"training_wall_only": True' in source
    assert '"geometry_claim_requires_successor_evaluation": True' in source
    assert (
        'manifest["required_reviews"] = ["0080", "0086", "0091"]'
        in source
    )


def test_round0092_discovers_one_issued_dated_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    draft = tmp_path / "round-0092-2026-07-28.md"
    issued = tmp_path / "round-0092-2026-07-29.md"
    draft.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    issued.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    monkeypatch.setattr(
        prepare_round0092_queue,
        "ROUND_FILE_GLOB",
        str(tmp_path / "round-0092-*.md"),
    )
    assert prepare_round0092_queue._require_issued_round() == str(issued)
    draft.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="exactly one issued"):
        prepare_round0092_queue._require_issued_round()


def test_round0092_receipts_exact_training_accounting() -> None:
    source = inspect.getsource(round0092_nodes.run_train)
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
    ):
        assert field in source
    loader = inspect.getsource(round0092_nodes._load_pipeline)
    assert 'graph["manifest"].get("round_id") != "0091"' in loader
    assert "HostInt8MaterializedArray.from_files" not in loader
    assert "validate_substrate(" in loader
    handler = inspect.getsource(round0092_nodes.run_job)
    assert 'active.get("manifest", {}).get("round_id")' in handler


def test_round0092_modules_do_not_mutate_cuda_visibility(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "sentinel")
    importlib.import_module("basemap.round0092_training")
    importlib.import_module("experiments.round0092_nodes")
    importlib.import_module("experiments.prepare_round0092_queue")
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "sentinel"
