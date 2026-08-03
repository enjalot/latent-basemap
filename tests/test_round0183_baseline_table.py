from __future__ import annotations

import copy

import pytest

from basemap.round0183_baseline_table import Round0183Error, build_table, render_markdown


def _aumap() -> dict:
    scales = {}
    for scale, value in (("200k", 0.3), ("500k", 0.4), ("2m", 0.5)):
        historical = None
        if scale != "500k":
            historical = {
                "standard_curve_seed42": {
                    "projection_ffr": value + 0.1,
                    "projection_recall_at_10": 0.02,
                },
                "legacy_a1b1_seed42": {
                    "projection_ffr": value + 0.15,
                    "projection_recall_at_10": 0.01,
                },
                "evidence": {"canonical_path": f"/{scale}", "sha256": scale},
            }
        scales[scale] = {
            "aumap_inverse_distance": {"ffr": value, "recall_at_10": 0.03},
            "historical_parametric_context": historical,
        }
    return {
        "schema": "round0175-aumap-oos-synthesis-v1",
        "round_id": "0175",
        "outcome": "aumap-oos-baseline-measured",
        "scales": scales,
    }


def _numap() -> dict:
    return {
        "schema": "round0181-numap-fixed-normalization-synthesis-v1",
        "round_id": "0181",
        "outcome": "numap-grease-fixed-normalization-baseline-measured",
        "comparison_to_reviewed_r0175": {
            "numap_fixed_normalization": {"ffr": 0.35, "recall_at_10": 0.04},
            "aumap_inverse_distance": {"ffr": 0.3, "recall_at_10": 0.03},
            "comparability": "same rows",
        },
    }


def test_table_includes_numap_and_exposes_500k_gap() -> None:
    table = build_table(aumap=_aumap(), numap=_numap(), numap_terminal_status="measured")
    assert table["rows"]["500k"]["corrected_parametric_standard_curve_seed42"] is None
    assert table["rows"]["200k"]["corrected_parametric_minus_aumap"]["ffr"] == pytest.approx(0.1)
    assert table["numap_grease_fixed_normalization"]["minus_aumap"]["ffr"] == pytest.approx(0.05)
    rendered = render_markdown(table)
    assert "not measured" in rendered
    assert "NUMAP/GrEASE" in rendered


def test_terminal_numap_failure_is_a_valid_branch() -> None:
    table = build_table(
        aumap=_aumap(), numap=None, numap_terminal_status="terminal-retry-failed"
    )
    assert table["numap_grease_fixed_normalization"] is None
    assert "Unavailable" in render_markdown(table)


def test_numap_branch_and_payload_must_agree() -> None:
    with pytest.raises(Round0183Error, match="presence disagrees"):
        build_table(aumap=_aumap(), numap=_numap(), numap_terminal_status="terminal-retry-failed")


def test_500k_substitution_fails_closed() -> None:
    source = _aumap()
    broken = copy.deepcopy(source)
    broken["scales"]["500k"]["historical_parametric_context"] = {}
    with pytest.raises(Round0183Error, match="unexpectedly acquired"):
        build_table(aumap=broken, numap=None, numap_terminal_status="terminal-retry-failed")
