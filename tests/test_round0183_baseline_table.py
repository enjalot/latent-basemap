from __future__ import annotations

import copy
import hashlib

import pytest

from basemap.round0183_baseline_table import Round0183Error, build_table, render_markdown
from experiments import prepare_round0183_queue


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


def test_terminal_pair_discovers_cross_midnight_files_and_checks_hashes(
    tmp_path, monkeypatch
) -> None:
    round_path = tmp_path / "round-0181-2026-08-03.md"
    result_path = tmp_path / "result-0181-2026-08-04.md"
    review_path = tmp_path / "review-0181-2026-08-04.md"
    release = "a" * 40
    round_path.write_text(
        '---\nround_id: "0181"\nstatus: issued\n---\n', encoding="utf-8"
    )
    result_path.write_text(
        '---\nround_id: "0181"\nstatus: complete\n'
        f'release_commit: "{release}"\ncapabilities_produced: []\n---\n',
        encoding="utf-8",
    )
    round_sha = hashlib.sha256(round_path.read_bytes()).hexdigest()
    result_sha = hashlib.sha256(result_path.read_bytes()).hexdigest()
    review_path.write_text(
        '---\nround_id: "0181"\nstatus: accepted\n'
        f'round: {round_path.name}\nround_sha256: "{round_sha}"\n'
        f'result: {result_path.name}\nresult_sha256: "{result_sha}"\n'
        f'verified_release_commit: "{release}"\nreleases: []\n---\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(prepare_round0183_queue, "LAB_ROOT", str(tmp_path))
    monkeypatch.setattr(
        prepare_round0183_queue,
        "R0181_RESULT_GLOB",
        str(tmp_path / "result-0181-*.md"),
    )
    monkeypatch.setattr(
        prepare_round0183_queue,
        "R0181_REVIEW_GLOB",
        str(tmp_path / "review-0181-*.md"),
    )
    result, review = prepare_round0183_queue._r0181_terminal_pair()
    assert result["canonical_path"] == str(result_path)
    assert review["canonical_path"] == str(review_path)

    review_path.write_text(review_path.read_text().replace(result_sha, "0" * 64))
    with pytest.raises(RuntimeError, match="binding changed"):
        prepare_round0183_queue._r0181_terminal_pair()
