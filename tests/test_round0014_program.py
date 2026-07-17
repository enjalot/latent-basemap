"""Target-specific CPU checks for the one Round 0014 production program."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest

from basemap.artifact_identity import canonical_json, sha256_bytes
from basemap.round0014_program import (
    ACCEPTED_CAPABILITY_SHA256,
    GPU_HOURS_CAP,
    NODES,
    Round0014MaterializedArray,
    TRAIN_CONFIG,
    TRAIN_CONFIG_SHA256,
    accepted_reference_records,
    program_policy,
    raw_source_map,
)
from basemap.round0014_transform import (
    build_transform_template,
    production_transform,
)
from basemap.source_closure import (
    ROUND0014_RUNTIME_ENTRYPOINTS,
    round0014_source_closure_receipt,
    runtime_source_closure,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact_treatment_input_tuple_and_ordered_six_node_dag():
    config = TRAIN_CONFIG
    assert TRAIN_CONFIG_SHA256 == sha256_bytes(canonical_json(config))
    assert config["phrase"] == "one 30M uniform-k15 MiniLM rung on one GSV RTX 5090"
    assert config["row_universe"] == {
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "rows": 30_000_000,
        "input_dimension": 384,
        "materialized_dtype": "<f2",
    }
    assert config["model"] == {
        "architecture": "residual_bottleneck",
        "input_dimension": 384,
        "hidden_dimension": 2048,
        "hidden_layers": 3,
        "output_dimension": 2,
        "use_batchnorm": False,
        "use_dropout": False,
        "low_dim_kernel": "legacy_lp",
        "a": 1.0,
        "b": 1.0,
    }
    optimizer = config["optimizer"]
    assert (optimizer["seed"], optimizer["successful_positive_lr_updates"],
            optimizer["schedule"]) == (42, 500_000, "cosine-v3-positive-budget")
    assert optimizer["positive_target_mode"] == "binary"
    assert optimizer["weighted_edge_sampling"] is False
    assert config["graph"]["sampling"] == "uniform-over-directed-edges"
    assert config["graph"]["with_replacement"] is True
    assert config["execution"]["required_pipeline"] == "device_uniform"

    ids = [node.node_id for node in NODES]
    assert ids == [
        "no_training_seal_canary", "train_seed42_30m", "transform_30m",
        "high_d_reference", "registered_panel", "semantic_renders",
    ]
    assert [node.dependency for node in NODES] == [None, *ids[:-1]]
    assert [node.node_id for node in NODES if node.training_performed] == [
        "train_seed42_30m"]
    assert sum(node.p90_wall_s * 1.15 for node in NODES) <= GPU_HOURS_CAP * 3600
    policy = program_policy()
    body = {key: policy[key] for key in policy if key != "identity_sha256"}
    assert policy["identity_sha256"] == sha256_bytes(canonical_json(body))

    references = accepted_reference_records(full_hash=False)
    assert len(references) == 76
    assert len({item["canonical_path"] for item in references}) == 76
    assert len({item["role"] for item in references}) == 76
    assert all(len(item["sha256"]) == 64 and item["bytes"] > 0
               for item in references)


def test_accepted_materialized_adapter_is_lazy_ordered_and_indexable():
    matrix = Round0014MaterializedArray()
    assert matrix.shape == (30_000_000, 384)
    assert matrix.dtype == np.dtype("<f2")
    assert isinstance(matrix.shard_paths, list) and len(matrix.shard_paths) == 30
    assert len(set(matrix.shard_paths)) == 30
    sample = matrix[np.array([0, 29_999_999], dtype=np.int64)]
    assert sample.shape == (2, 384) and sample.dtype == np.dtype("<f2")
    assert np.isfinite(sample).all()
    assert matrix[0, :3].shape == (3,)
    source = raw_source_map()
    assert source.total_rows == 30_000_000 and source.dimension == 384
    assert source.source_segments(0, 1)[0]["output_global_row_start"] == 0
    assert matrix.round0014_pack_seal["capability_sha256"] == \
        ACCEPTED_CAPABILITY_SHA256


def test_production_transform_is_plain_release_bound_and_bounded(monkeypatch):
    assert inspect.isfunction(production_transform)
    assert production_transform.__closure__ is None
    assert production_transform.__defaults__ is None
    assert production_transform.__kwdefaults__ is None
    assert production_transform.__qualname__ == "production_transform"
    template = build_transform_template(
        release_root=str(ROOT), release_sha="1" * 40,
        train_output_relative_path="artifacts/train/model.pt")
    spec = template["execution_spec"]
    assert spec["release_commit"] == "1" * 40
    assert spec["defining_artifact"]["relative_path"] == \
        "basemap/round0014_transform.py"
    config = template["template_config"]
    assert config["trained_model"] == {
        "producer_node": "train_seed42_30m",
        "controller_output_relative_path": "artifacts/train/model.pt",
        "content_binding": "runtime-full-sha256-before-transform-spec-seal",
        "pre_gate_hash_available": False,
    }
    assert (config["model_batch_rows"], config["read_block_rows"],
            config["rows_per_output_chunk"]) == (4096, 32768, 1_000_000)
    assert config["normalization"] == config["centering"] == "none"
    assert config["stochastic_options"] == []

    class Model:
        @staticmethod
        def transform(value, *, batch_size):
            assert batch_size == 4096
            return np.column_stack((value[:, 0], value[:, 1]))

    import basemap.round0014_transform as target
    monkeypatch.setattr(target, "_ACTIVE_MODEL", Model())
    monkeypatch.setattr(target, "_ACTIVE_CONFIG", {"model_batch_rows": 4096})
    block = np.zeros((3, 384), dtype="<f4")
    assert production_transform(block).shape == (3, 2)
    with pytest.raises(ValueError, match="bounded <f4"):
        production_transform(block.astype("<f8"))


def test_round0014_source_closure_binds_the_only_node_executable():
    closure = runtime_source_closure(str(ROOT), ROUND0014_RUNTIME_ENTRYPOINTS)
    required = {
        "experiments/run_round0014_node.py",
        "basemap/round0014_program.py",
        "basemap/round0014_admission.py",
        "basemap/round0014_transform.py",
        "basemap/run_controller.py",
        "basemap/pumap/parametric_umap/core.py",
        "experiments/score_complete_panel.py",
    }
    assert required.issubset(set(closure))
    receipt = round0014_source_closure_receipt(
        str(ROOT), require_tracked=False)
    assert receipt["schema"] == "round0014-runtime-source-closure-v1"
    assert tuple(receipt["entrypoints"]) == ROUND0014_RUNTIME_ENTRYPOINTS
    assert {item["relative_path"] for item in receipt["members"]} == set(closure)


def test_node_main_admits_before_parse_output_torch_or_cuda():
    source = (ROOT / "experiments" / "run_round0014_node.py").read_text()
    tree = ast.parse(source)
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and
                   ((isinstance(node, ast.Import) and any(
                       alias.name == "torch" for alias in node.names)) or
                    (isinstance(node, ast.ImportFrom) and node.module == "torch"))
                   for node in tree.body)
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
    calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
    admission = next(node for node in calls
                     if isinstance(node.func, ast.Name) and
                     node.func.id == "require_round0005_child_admission")
    parsing = next(node for node in calls
                   if isinstance(node.func, ast.Attribute) and
                   node.func.attr == "parse_args")
    assert admission.lineno < parsing.lineno
