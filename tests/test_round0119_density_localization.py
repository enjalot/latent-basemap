from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.round0108_evaluation import seal
from experiments import round0119_nodes
from experiments.prepare_round0119_queue import (
    REQUIRED_REVIEWS,
    RELEASE_ROOT,
    _accepted_review,
    _clean_terminal,
    _document,
    _frontmatter_list,
)


def _write_json(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return expected_input_signature(str(path))


def test_authenticate_model_binds_train_config_and_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = {
        "schema": "test-config-v1",
        "arm": "raw",
        "input": {
            "rows": 1_993_761,
            "representation": "fresh-local-raw-fp16",
        },
        "graph": {
            "k": 50,
            "sampling": "fuzzy-weight-proportional-with-replacement",
            "positive_target_mode": "binary",
        },
        "optimizer": {
            "seed": 43,
            "successful_positive_lr_updates": 500_000,
        },
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": 768,
            "hidden_dimension": 2048,
            "hidden_layers": 3,
            "output_dimension": 2,
            "use_batchnorm": False,
            "use_dropout": False,
            "low_dim_kernel": "legacy_lp",
            "a": 1.0,
            "b": 1.0,
        },
    }
    config_sha = sha256_bytes(canonical_json(config))
    config_path = tmp_path / "production-config.json"
    config_signature = _write_json(
        config_path,
        {
            "schema": "test-config-receipt-v1",
            "round_id": "0117",
            "config": config,
            "config_sha256": config_sha,
        },
    )
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"model")
    model_signature = expected_input_signature(str(model_path))
    train_path = tmp_path / "train-receipt.json"
    train_signature = _write_json(
        train_path,
        seal({
            "schema": "test-train-v1",
            "round_id": "0117",
            "arm": "raw",
            "training_seed": 43,
            "optimizer_updates": 500_000,
            "exact_execution_receipt": {
                "pipeline": "host_weighted_jina_prompt_contrast",
                "sampler_class": "PromptWeightedJinaSampler",
                "positive_sampling": (
                    "fuzzy_weight_proportional_with_replacement_via_exact_"
                    "uniform_envelope_rejection"
                ),
                "positive_with_replacement": True,
                "weighted_effective": True,
                "multiplicity_policy": (
                    "shared-source-raw-document-union-representative-only"
                ),
                "feature_residency": (
                    "host-contiguous-compact-fp16-memmap"
                ),
                "source_representation": "raw-fp16",
                "device_conversion": "device-fp32-from-exact-fp16",
            },
            "train_accounting": {
                "optimizer_steps_attempted": 500_000,
                "optimizer_steps_succeeded": 500_000,
                "pipeline_pipeline": (
                    "host_weighted_jina_prompt_contrast"
                ),
                "pipeline_sampler_class": "PromptWeightedJinaSampler",
                "pipeline_positive_sampling": (
                    "fuzzy_weight_proportional_with_replacement_via_exact_"
                    "uniform_envelope_rejection"
                ),
                "pipeline_multiplicity_policy": (
                    "shared-source-raw-document-union-representative-only"
                ),
                "pipeline_feature_residency": (
                    "host-contiguous-compact-fp16-memmap"
                ),
                "pipeline_source_representation": "raw-fp16",
                "pipeline_device_conversion": (
                    "device-fp32-from-exact-fp16"
                ),
                "pipeline_graph_degree": (
                    "variable-symmetric-fuzzy-k50-topology"
                ),
                "pipeline_compact_retained_rows": 1_993_761,
            },
            "train_checks": {
                "endpoint_rows_match_updates": True,
                "exact_update_closure": True,
                "no_pipeline_stamp_drift": True,
                "zero_numerical_skips": True,
            },
            "production_config": config_signature,
            "production_config_sha256": config_sha,
            "model": model_signature,
        }),
    )

    class FakeModel:
        architecture = "residual_bottleneck"
        input_dim = 768
        hidden_dim = 2048
        n_layers = 3
        n_components = 2
        use_batchnorm = False
        use_dropout = False
        low_dim_kernel = "legacy_lp"
        a = 1.0
        b = 1.0

    from basemap.pumap import parametric_umap

    monkeypatch.setattr(
        parametric_umap.ParametricUMAP,
        "load",
        lambda path, device: FakeModel(),
    )
    spec = {
        "key": "current_2m_seed43",
        "group": "current_2m",
        "round_id": "0117",
        "seed": 43,
        "arm": "raw",
        "train_schema": "test-train-v1",
        "config_receipt_schema": "test-config-receipt-v1",
        "config_receipt_round_id": "0117",
        "config_schema": "test-config-v1",
        "training_population": "population",
        "training_graph": "graph",
        "training_dose": "dose",
        "training_representation": "representation",
        "training_dequantization": "dequantization",
        "semantic_contract": {
            "population_rows": 1_993_761,
            "graph_neighbors": 50,
            "successful_updates": 500_000,
            "pipeline": "host_weighted_jina_prompt_contrast",
            "sampler_class": "PromptWeightedJinaSampler",
            "positive_sampling": (
                "fuzzy_weight_proportional_with_replacement_via_exact_"
                "uniform_envelope_rejection"
            ),
            "multiplicity_policy": (
                "shared-source-raw-document-union-representative-only"
            ),
            "feature_residency": (
                "host-contiguous-compact-fp16-memmap"
            ),
            "source_representation": "raw-fp16",
            "dequantization": "device-fp32-from-exact-fp16",
        },
        "train_receipt": train_signature,
        "production_config": config_signature,
        "model": model_signature,
    }
    bundle = round0119_nodes._authenticate_model(spec)
    assert bundle["train"] == train_signature
    assert bundle["production_config"] == config_signature
    assert bundle["model_signature"] == model_signature

    drifted = {
        **spec,
        "semantic_contract": {
            **spec["semantic_contract"],
            "graph_neighbors": 15,
        },
    }
    with pytest.raises(
        round0119_nodes.Round0119Error,
        match="actual pipeline semantics changed",
    ):
        round0119_nodes._authenticate_model(drifted)

    changed = json.loads(config_path.read_text(encoding="utf-8"))
    changed["config"]["optimizer"]["seed"] = 42
    config_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(round0119_nodes.Round0119Error, match="bytes changed"):
        round0119_nodes._authenticate_model(spec)


def test_clean_terminal_requires_r0117_completion(tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.json"
    release = "d" * 40
    queue = {
        "round_id": "0117",
        "release_sha": release,
        "repo_root": RELEASE_ROOT,
        "jobs": [{"id": key} for key in ("raw", "document", "decision")],
    }
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    queue_sha256 = expected_input_signature(str(queue_path))["sha256"]
    path = tmp_path / "runner-terminal.json"
    terminal = {
        "schema": "slim-runner-terminal-v3",
        "round_id": "0117",
        "verdict": "succeeded",
        "completed_jobs": ["raw", "document", "decision"],
        "required_jobs": ["raw", "document", "decision"],
        "gpu_wall_accounting_complete": True,
        "queue_manifest_sha256": queue_sha256,
        "queue_manifest_sha256_at_finish": queue_sha256,
        "release_checkout": {
            "repo_root": RELEASE_ROOT,
            "head": release,
            "detached": True,
            "dirty": False,
        },
        "release_checkout_at_finish": {
            "repo_root": RELEASE_ROOT,
            "head": release,
            "detached": True,
            "dirty": False,
        },
        "release_checkout_unchanged": True,
        "queue_manifest_unchanged": True,
        "boundary_problems": [],
        "nodes": [
            {
                "node": key,
                "returncode": 0,
                "validation_problems": [],
            }
            for key in ("raw", "document", "decision")
        ],
    }
    path.write_text(json.dumps(terminal), encoding="utf-8")
    assert _clean_terminal(
        str(queue_path),
        str(path),
        round_id="0117",
        expected_release_sha=release,
    )["terminal"]["bytes"] > 0
    with pytest.raises(RuntimeError, match="clean success"):
        _clean_terminal(
            str(queue_path),
            str(path),
            round_id="0117",
            expected_release_sha="f" * 40,
        )
    terminal["completed_jobs"] = ["raw"]
    path.write_text(json.dumps(terminal), encoding="utf-8")
    with pytest.raises(RuntimeError, match="clean success"):
        _clean_terminal(
            str(queue_path),
            str(path),
            round_id="0117",
            expected_release_sha=release,
        )


def test_accepted_review_closes_result_release_and_capability(
    tmp_path: Path,
) -> None:
    release = "e" * 40
    result_path = tmp_path / "result-0117-2026-07-31.md"
    result_path.write_text(
        "\n".join([
            "---",
            'round_id: "0117"',
            "status: complete",
            f'release_commit: "{release}"',
            "capabilities_produced:",
            "  - jina-fineweb-2m-prompt-map-seed43-contrast-v1",
            "---",
            "result",
            "",
        ]),
        encoding="utf-8",
    )
    result_sha256 = expected_input_signature(str(result_path))["sha256"]
    review_path = tmp_path / "review-0117-2026-07-31.md"
    review_path.write_text(
        "\n".join([
            "---",
            'round_id: "0117"',
            "status: accepted",
            f"result: {result_path.name}",
            f'result_sha256: "{result_sha256}"',
            f'verified_release_commit: "{release}"',
            (
                'releases: ["capability:jina-fineweb-2m-prompt-map-'
                'seed43-contrast-v1"]'
            ),
            "---",
            "review",
            "",
        ]),
        encoding="utf-8",
    )
    review_sha256 = expected_input_signature(str(review_path))["sha256"]
    evidence = _accepted_review(
        str(review_path), review_sha256, round_id="0117"
    )
    assert evidence["result"]["sha256"] == result_sha256
    result_path.write_text(
        result_path.read_text(encoding="utf-8") + "changed\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="does not close"):
        _accepted_review(str(review_path), review_sha256, round_id="0117")


def test_frontmatter_list_accepts_r0117_block_shape_and_inline_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result-0117-2026-07-31.md"
    path.write_text(
        "\n".join([
            "---",
            'round_id: "0117"',
            "status: complete",
            "capabilities_produced:",
            "  - jina-fineweb-2m-prompt-map-seed43-contrast-v1",
            "---",
            "result",
            "",
        ]),
        encoding="utf-8",
    )
    frontmatter, _ = _document(str(path))
    assert _frontmatter_list(
        frontmatter,
        "capabilities_produced",
        label="R0117 result",
    ) == ["jina-fineweb-2m-prompt-map-seed43-contrast-v1"]
    assert _frontmatter_list(
        {"releases": '["capability:one", "capability:two"]'},
        "releases",
        label="review",
    ) == ["capability:one", "capability:two"]
    assert _frontmatter_list(
        {"releases": "  - \"a string\"\n  - 'reviewer''s-release'"},
        "releases",
        label="review",
    ) == ["a string", "reviewer's-release"]


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "jina-capability",
        "  -",
        "  - valid\n    - nested",
        "  - valid\n - inconsistent",
        "  - valid\n  - 17",
        "  - valid\n  - true",
        "  - valid\n  - key: value",
        '  - valid\n  - ["nested"]',
        '  - "unterminated',
        "  - 'unterminated",
        '["valid", 17]',
        '["valid", null]',
        '["unterminated"',
        '{"not": "a list"}',
    ],
)
def test_frontmatter_list_rejects_malformed_or_non_string_items(
    raw: str,
) -> None:
    with pytest.raises(RuntimeError):
        _frontmatter_list(
            {"capabilities_produced": raw},
            "capabilities_produced",
            label="result",
        )


def test_registered_cell_and_group_order_is_frozen() -> None:
    assert round0119_nodes.TRANSFORM_BATCH_ROWS == 8_192
    assert round0119_nodes.CELL_ORDER == (
        "historical_2m_seed42",
        "historical_2m_seed43",
        "current_2m_seed42",
        "current_2m_seed43",
        "current_25m_seed42",
        "current_25m_seed43",
    )
    assert set(round0119_nodes.GROUPS) == {
        "historical_2m",
        "current_2m",
        "current_25m",
    }
    assert REQUIRED_REVIEWS == (
        "0037",
        "0038",
        "0107",
        "0108",
        "0109",
        "0115",
        "0117",
    )
    assert "0110" not in REQUIRED_REVIEWS
    assert "0111" not in REQUIRED_REVIEWS
    assert "0118" not in REQUIRED_REVIEWS
