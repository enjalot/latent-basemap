import json
from pathlib import Path

import pytest

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.round0064_evaluation import (
    Round0064Error,
    seal,
    validate_train_bundle,
)
from experiments import map_registry
from experiments import prepare_round0064_queue
from experiments.round0064_nodes import (
    MATCHED_NONINFERIORITY_MARGINS,
    run_comparison,
)


def _bundle_fixture(tmp_path: Path, *, label: str) -> dict:
    specs = {
        "r0061-30m": {
            "round": "0061",
            "receipt": "round0061-train-receipt-v1",
            "config": "round0055-production-config-v1",
            "rows": 30_000_000,
            "retained": 29_781_754,
            "updates": 500_003,
            "sampler": "HostInt8Balanced30mCanonicalSampler",
        },
        "r0063-60m": {
            "round": "0063",
            "receipt": "round0063-train-receipt-v1",
            "config": "round0052-production-config-v1",
            "rows": 60_000_000,
            "retained": 59_399_288,
            "updates": 997_248,
            "sampler": "HostInt8BalancedCanonicalSampler",
        },
    }
    spec = specs[label]
    model_path = tmp_path / f"{label}.pt"
    model_path.write_bytes(b"fixture-checkpoint")
    model_signature = expected_input_signature(str(model_path))
    stamp = {
        "pipeline": "host_int8_canonical",
        "sampler_class": spec["sampler"],
        "x_residency": "host_int8_materialized",
        "positive_sampling": "uniform-source-then-slot",
        "negative_sampling": "uniform-retained-nonself",
        "positive_source_count": spec["retained"],
        "valid_canonical_edge_count": spec["retained"] * 15,
    }
    config = {
        "schema": spec["config"],
        "row_universe": {
            "rows": spec["rows"],
            "input_dimension": 384,
        },
        "model": {
            "architecture": "residual_bottleneck",
            "input_dimension": 384,
            "hidden_dimension": 2048,
            "output_dimension": 2,
        },
        "execution": {"expected_pipeline_stamp": stamp},
    }
    updates = spec["updates"]
    body = {
        "schema": spec["receipt"],
        "round_id": spec["round"],
        "model": model_signature,
        "production_config": config,
        "production_config_sha256": sha256_bytes(canonical_json(config)),
        "train_accounting": {
            "budget_satisfied": True,
            "positive_lr_optimizer_steps": updates,
            "optimizer_steps_attempted": updates,
            "optimizer_steps_succeeded": updates,
            "amp_overflow_skips": 0,
            "nonfinite_loss_skips": 0,
            "nonfinite_gradient_skips": 0,
            "stop_reason": "lr_horizon",
        },
        "exact_execution_receipt": stamp,
        "retry_count": 0,
    }
    receipt = seal(body)
    receipt_path = tmp_path / f"{label}-receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    return {
        "label": label,
        "model_path": str(model_path),
        "model_sha256": model_signature["sha256"],
        "train_receipt_path": str(receipt_path),
        "train_receipt_sha256": expected_input_signature(
            str(receipt_path)
        )["sha256"],
    }


@pytest.mark.parametrize("label", ["r0061-30m", "r0063-60m"])
def test_train_bundle_authenticates_exact_successful_execution(
    tmp_path: Path,
    label: str,
) -> None:
    args = _bundle_fixture(tmp_path, label=label)
    bundle = validate_train_bundle(**args)
    assert bundle["label"] == label
    assert bundle["receipt"]["retry_count"] == 0

    receipt_path = Path(args["train_receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    receipt["train_accounting"]["nonfinite_loss_skips"] = 1
    mutated = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    receipt_path.write_text(json.dumps(seal(mutated)))
    args["train_receipt_sha256"] = expected_input_signature(
        str(receipt_path)
    )["sha256"]
    with pytest.raises(Round0064Error):
        validate_train_bundle(**args)


def test_scale_margins_are_bounded_and_predeclared() -> None:
    assert MATCHED_NONINFERIORITY_MARGINS == {
        "ffr": 0.02,
        "density": 0.05,
        "purity_k256": 0.05,
        "purity_k1024": 0.05,
        "projection_ffr": 0.02,
    }
    assert all(0 < value <= 0.05 for value in
               MATCHED_NONINFERIORITY_MARGINS.values())


def _panel_fixture(
    *,
    key: str,
    reference: str = "reference-30m",
    delta: float = 0.0,
) -> dict:
    body = {
        "schema": "round0064-registered-panel-v1",
        "round_id": "0064",
        "map_key": key,
        "eligibility": {"sha256": "e" * 64},
        "scientific_universe": {
            "rows": 29_781_754,
            "substrate": "balanced-30m",
        },
        "panel": {
            "n": 29_781_754,
            "anchor_hash": "anchors",
            "ffr": 0.50 + delta,
            "density": 0.70 + delta,
            "purity": {
                "k256": 0.90 + delta,
                "k1024": 0.85 + delta,
            },
            "recall@k": 0.03,
            "provenance": {"hiD_reference_key": reference},
        },
        "recall_at_10": 0.03,
        "recall_at_50": 0.08,
        "projection": {"proj_ffr": 0.45 + delta},
        "absolute_selector_passed": True,
    }
    return seal(body)


def test_scale_comparison_requires_one_exact_matched_reference(
    tmp_path: Path,
) -> None:
    roots = {
        "matched_control_panel": tmp_path / "control",
        "scaled_matched_panel": tmp_path / "matched",
        "scaled_full_panel": tmp_path / "full",
    }
    for root in roots.values():
        root.mkdir()
    (roots["matched_control_panel"] / "panel.json").write_text(json.dumps(
        _panel_fixture(key="r0061-30m-on-30m")
    ))
    (roots["scaled_matched_panel"] / "panel.json").write_text(json.dumps(
        _panel_fixture(key="r0063-60m-on-30m", delta=-0.01)
    ))
    full = _panel_fixture(
        key="r0063-60m-on-60m",
        reference="reference-60m",
    )
    full_body = {
        key: value
        for key, value in full.items()
        if key != "identity_sha256"
    }
    full_body["scientific_universe"] = {
        "rows": 59_399_288,
        "substrate": "balanced-60m",
    }
    full_body["panel"]["n"] = 59_399_288
    (roots["scaled_full_panel"] / "panel.json").write_text(
        json.dumps(seal(full_body))
    )
    output = tmp_path / "comparison"
    receipt = run_comparison(
        {},
        {
            "outputs": [str(output)],
            **{key: str(value) for key, value in roots.items()},
        },
    )
    assert receipt["decision"]["advance_to_120m_scale_rung"] is True

    mismatched = _panel_fixture(
        key="r0063-60m-on-30m",
        reference="different-reference",
    )
    (roots["scaled_matched_panel"] / "panel.json").write_text(
        json.dumps(mismatched)
    )
    with pytest.raises(
        Round0064Error,
        match="one exact row universe/reference",
    ):
        run_comparison(
            {},
            {
                "outputs": [str(tmp_path / "comparison-mismatch")],
                **{key: str(value) for key, value in roots.items()},
            },
        )


def test_round0064_queue_is_bounded_and_has_no_training() -> None:
    import inspect

    source = inspect.getsource(
        prepare_round0064_queue.prepare_round0064
    )
    assert "gpu_hours_cap=2.5" in source
    assert source.count('action="transform"') == 3
    assert source.count('action="panel"') == 3
    assert '"training_performed"] = False' in source
    assert 'action="train"' not in source
    assert "review-0061" not in source
    assert "review-0063" not in source


def test_registry_discovers_two_r0064_base_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    ledger = tmp_path / "ledger"
    checkpoints = tmp_path / "checkpoints"
    round_dir = runs / "round-0064"
    artifacts = round_dir / "queue/artifacts"
    ledger.mkdir()
    checkpoints.mkdir()
    (round_dir / "queue").mkdir(parents=True)
    (round_dir / "queue/queue.json").write_text(json.dumps({
        "release_sha": "a" * 40,
    }))
    (ledger / "round-0064-2026-07-26.md").write_text(
        "---\nround_id: \"0064\"\nstatus: issued\n---\n"
    )
    definitions = (
        (
            "r0061-30m-on-30m",
            "coordinates-r0061-30m",
            "panel-r0061-30m",
        ),
        (
            "r0063-60m-on-60m",
            "coordinates-r0063-60m",
            "panel-r0063-60m",
        ),
    )
    for key, coordinate_name, panel_name in definitions:
        coordinates = artifacts / coordinate_name
        panel_root = artifacts / panel_name
        chunk = coordinates / "chunk-00000"
        chunk.mkdir(parents=True)
        panel_root.mkdir(parents=True)
        (chunk / "coordinates.npy").write_bytes(b"coords")
        label = (
            "r0061-balanced-30m-seed42"
            if key.startswith("r0061")
            else "r0063-balanced-60m-seed42"
        )
        transform = {
            "schema": "round0036-transform-capability-v1",
            "map_key": key,
            "model": {"sha256": "b" * 64},
            "row_accounting": {
                "all_rows": 30_000_000 if key.startswith("r0061") else
                60_000_000,
                "retained_representatives": (
                    29_781_754 if key.startswith("r0061") else 59_399_288
                ),
            },
        }
        (coordinates / "actual-transform.json").write_text(
            json.dumps(transform)
        )
        panel = {
            "schema": "round0064-registered-panel-v1",
            "map_key": key,
            "map": {"label": label},
            "absolute_selector_passed": True,
            "panel": {
                "ffr": 0.5,
                "density": 0.7,
                "purity": {"k256": 1.0, "k1024": 1.0},
                "formula_version": "panel_v2.2-2026-07-15",
            },
            "projection": {"proj_ffr": 0.4},
        }
        (panel_root / "panel.json").write_text(json.dumps(panel))

    monkeypatch.setattr(map_registry, "RUNS_DIR", runs)
    monkeypatch.setattr(map_registry, "LEDGER_DIR", ledger)
    monkeypatch.setattr(map_registry, "CHECKPOINT_DIR", checkpoints)
    registry = map_registry.scan()
    entries = [
        item for item in registry["maps"]
        if item.get("round_id") == "0064"
        and item.get("kind") == "round-map"
    ]
    assert {item["map_label"] for item in entries} == {
        "r0061-balanced-30m-seed42",
        "r0063-balanced-60m-seed42",
    }
