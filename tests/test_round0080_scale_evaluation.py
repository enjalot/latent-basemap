import inspect
import json
from pathlib import Path

from basemap.round0064_evaluation import MODEL_SPECS, seal
from experiments import prepare_round0080_queue
from experiments.round0080_nodes import (
    MATCHED_NONINFERIORITY_MARGINS,
    _noninferiority,
    run_comparison,
)


def _checks(*, density_passed: bool = False) -> dict[str, bool]:
    return {
        "ffr_at_least_0_40": True,
        "density_at_least_0_60": density_passed,
        "purity_k256_at_least_0_50": True,
        "purity_k1024_at_least_0_50": True,
        "heldout_projection_beats_untrained_floor": True,
        "recall_at_50_exceeds_recall_at_10": True,
        "coords_finite": True,
        "coords_not_collapsed": True,
        "embeddings_finite": True,
        "eligible_embeddings_nonzero": True,
    }


def _panel(
    *,
    schema: str,
    key: str,
    rows: int,
    substrate: str,
    delta: float = 0.0,
) -> dict:
    return seal({
        "schema": schema,
        "round_id": "0076" if key.startswith("r0075") else "0080",
        "map_key": key,
        "eligibility": (
            {"sha256": "e" * 64}
            if rows < 100_000_000
            else {"sha256": "f" * 64}
        ),
        "scientific_universe": {
            "rows": rows,
            "substrate": substrate,
            "row_namespace": f"compact ascending {substrate} retained rows",
            "excluded_rows_in_scoring": False,
        },
        "panel": {
            "n": rows,
            "anchor_hash": (
                "shared-90m-anchors"
                if rows < 100_000_000
                else "full-120m-anchors"
            ),
            "ffr": 0.48 + delta,
            "density": 0.10 + delta,
            "purity": {
                "k256": 1.05 + delta,
                "k1024": 0.88 + delta,
            },
            "recall@k": 0.001,
            "provenance": {
                "hiD_reference_key": (
                    "shared-90m-reference"
                    if rows < 100_000_000
                    else "full-120m-reference"
                ),
            },
        },
        "recall_at_10": 0.001,
        "recall_at_50": 0.003,
        "projection": {"proj_ffr": 0.44 + delta},
        "decision_checks": _checks(),
        "absolute_selector_passed": False,
    })


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_r0079_model_identity_is_registered_exactly() -> None:
    assert MODEL_SPECS["r0079-120m"] == {
        "round_id": "0079",
        "receipt_schema": "round0079-train-receipt-v1",
        "config_schema": "round0079-production-config-v1",
        "rows": 120_000_000,
        "retained_rows": 118_067_492,
        "updates": 1_982_221,
        "sampler_class": "HostInt8Balanced120mCanonicalSampler",
    }


def test_noninferiority_uses_registered_relative_margins() -> None:
    control = {
        "ffr": 0.48,
        "density": 0.10,
        "purity_k256": 1.05,
        "purity_k1024": 0.88,
        "projection_ffr": 0.44,
        "recall_at_10": 0.001,
        "recall_at_50": 0.003,
    }
    treatment = dict(control)
    for metric, margin in MATCHED_NONINFERIORITY_MARGINS.items():
        treatment[metric] = control[metric] - margin
    comparison = _noninferiority(treatment, control)
    assert all(item["passed"] for item in comparison.values())
    # The old six-decimal decision rounding incorrectly accepted a real
    # 4e-7 miss. Reporting may round later, but acceptance uses raw metrics.
    treatment["ffr"] -= 0.0000004
    comparison = _noninferiority(treatment, control)
    assert comparison["ffr"]["passed"] is False


def test_comparison_uses_nearest_rung_and_not_absolute_density(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control/panel.json"
    matched = tmp_path / "matched/panel.json"
    full = tmp_path / "full/panel.json"
    _write_json(control, _panel(
        schema="round0076-registered-panel-v1",
        key="r0075-90m-on-90m",
        rows=88_945_313,
        substrate="balanced-90m",
    ))
    _write_json(matched, _panel(
        schema="round0080-registered-panel-v1",
        key="r0079-120m-on-90m",
        rows=88_945_313,
        substrate="balanced-90m",
        delta=0.002,
    ))
    _write_json(full, _panel(
        schema="round0080-registered-panel-v1",
        key="r0079-120m-on-120m",
        rows=118_067_492,
        substrate="balanced-120m",
        delta=0.002,
    ))
    receipt = run_comparison({}, {
        "outputs": [str(tmp_path / "output")],
        "control_panel": str(control),
        "matched_panel": str(matched),
        "full_panel": str(full),
    })
    assert receipt["decision"]["120m_supported_as_deliberate_ladder_rung"]
    assert receipt["same_row_90m_comparison"]["passed"]
    assert receipt["full_120m_non_density_checks_passed"]
    assert (
        receipt["density_semantics"][
            "legacy_absolute_floor_used_for_decision"
        ]
        is False
    )
    assert (
        receipt["density_semantics"]["legacy_absolute_floor_reported"]
        is False
    )


def test_round0080_queue_is_bounded_no_training_and_nearest_rung() -> None:
    source = inspect.getsource(prepare_round0080_queue.prepare_round0080)
    assert "gpu_hours_cap=3.0" in source
    assert source.count('action="transform"') == 2
    assert source.count('action="panel"') == 2
    assert source.count('action="high_d_reference"') == 1
    assert source.count('action="ood"') == 1
    assert 'action="train"' not in source
    assert '"required_reviews"] = ["0065", "0076", "0079"]' in source
    assert '"120m_noninferiority_control": "90m"' in source


def test_round0080_discovers_the_one_issued_dated_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    old = tmp_path / "round-0080-2026-07-27.md"
    current = tmp_path / "round-0080-2026-07-28.md"
    old.write_text("---\nstatus: draft\n---\n", encoding="utf-8")
    current.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    monkeypatch.setattr(
        prepare_round0080_queue,
        "ROUND_FILE_GLOB",
        str(tmp_path / "round-0080-*.md"),
    )
    assert prepare_round0080_queue._require_issued_round() == str(current)
    old.write_text("---\nstatus: issued\n---\n", encoding="utf-8")
    try:
        prepare_round0080_queue._require_issued_round()
    except RuntimeError as exc:
        assert "exactly one issued round document" in str(exc)
    else:
        raise AssertionError("multiple issued R0080 contracts were accepted")
