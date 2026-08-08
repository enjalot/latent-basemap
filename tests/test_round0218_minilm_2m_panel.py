"""Contract tests for the R0218 MiniLM 2M four-seed panel."""
from __future__ import annotations

import math
from typing import Any

import pytest

from basemap.round0216_minilm_2m_substrate import COMPOSITION
from basemap.round0217_minilm_2m_seed_family import SEEDS as MAP_SEEDS
from basemap.round0218_minilm_2m_panel import (
    CORPORA,
    CORPUS_SLUGS,
    DIAGNOSTIC_METRICS,
    GATE_REGISTERABLE_HERE,
    PANEL_METRICS,
    ROUND_ID,
    Round0218Error,
    SEEDS,
    build_family_panel_evidence,
    corpus_ffr_view,
    descriptive_summaries,
    map_capability,
    panel_execution_ok,
    panel_metric_view,
    purity_fidelity,
)


def _panel(
    *,
    density: float = 0.62,
    ffr: float = 0.41,
    k256: float = 0.98,
    k1024: float = 1.03,
    groups: dict[str, Any] | None = None,
    guards: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "density": density,
        "ffr": ffr,
        "purity": {"k256": k256, "k1024": k1024},
        "ffr_by_group": groups
        if groups is not None
        else {
            slug: {"anchors": 1_000, "ffr": round(ffr + 0.01 * index, 4)}
            for index, slug in enumerate(CORPUS_SLUGS)
        },
        "guards": guards
        if guards is not None
        else {
            "coords_finite": True,
            "coords_collapsed": False,
            "emb_finite": True,
            "emb_zero_rows": 0,
        },
        "provenance": {"hiD_reference_reused": True},
    }


def _cells(**overrides: Any) -> dict[int, dict[str, Any]]:
    cells = {}
    for index, seed in enumerate(SEEDS):
        panel = overrides.get(str(seed)) or _panel(
            density=0.60 + 0.01 * index, ffr=0.40 + 0.002 * index
        )
        cells[seed] = {
            "seed": seed,
            "capability": map_capability(seed),
            "panel": panel,
            "panel_metrics": panel_metric_view(panel),
            "corpus_ffr": corpus_ffr_view(panel),
        }
    return cells


def test_round0218_identity_is_stable() -> None:
    assert ROUND_ID == "0218"
    assert SEEDS == MAP_SEEDS == (42, 43, 44, 45)
    assert GATE_REGISTERABLE_HERE is False
    assert PANEL_METRICS == (
        "density_v2",
        "ffr",
        "purity_fidelity_k256",
        "purity_fidelity_k1024",
    )
    assert DIAGNOSTIC_METRICS == ("density_v2",)


def test_round0218_corpus_table_follows_r0216_composition() -> None:
    """The corpus ids are positions in R0216's registered COMPOSITION."""
    assert len(CORPORA) == len(COMPOSITION) == 4
    for (index, slug, name, rows), (registered_name, registered_rows) in zip(
        CORPORA, COMPOSITION
    ):
        assert name == registered_name
        assert rows == registered_rows
        assert CORPUS_SLUGS[index] == slug
    assert [rows for *_rest, rows in CORPORA] == [800_000, 500_000, 500_000, 200_000]


def test_round0218_purity_fidelity_is_symmetric_about_one() -> None:
    """A map that over-separates is penalised exactly like one that under-separates."""
    assert purity_fidelity(1.0) == 1.0
    assert math.isclose(purity_fidelity(1.25), purity_fidelity(1 / 1.25), rel_tol=1e-12)
    assert purity_fidelity(0.5) < purity_fidelity(0.9) < 1.0


def test_round0218_metric_view_reads_the_panel_payload() -> None:
    metrics = panel_metric_view(_panel(density=0.61, ffr=0.42, k256=0.9, k1024=1.1))
    assert metrics["density_v2"] == 0.61
    assert metrics["ffr"] == 0.42
    assert math.isclose(metrics["purity_fidelity_k256"], 0.9)
    assert math.isclose(metrics["purity_fidelity_k1024"], 1 / 1.1)


@pytest.mark.parametrize(
    "panel",
    [
        _panel(ffr=float("nan")),
        _panel(density=float("inf")),
        _panel(density=1.4),
        _panel(ffr=0.0),
        _panel(ffr=1.5),
    ],
)
def test_round0218_metric_view_rejects_inadmissible_numbers(panel: dict) -> None:
    with pytest.raises(Round0218Error):
        panel_metric_view(panel)


def test_round0218_metric_view_rejects_an_undefined_purity_ratio() -> None:
    with pytest.raises(Round0218Error):
        panel_metric_view(_panel(k256=None))


def test_round0218_corpus_slices_must_cover_every_corpus() -> None:
    slices = corpus_ffr_view(_panel())
    assert set(slices) == set(CORPUS_SLUGS) == {"fineweb", "redpajama", "pile", "code"}
    with pytest.raises(Round0218Error):
        corpus_ffr_view(
            _panel(groups={slug: {"anchors": 10, "ffr": 0.4} for slug in ("pile",)})
        )
    with pytest.raises(Round0218Error):
        corpus_ffr_view(
            _panel(
                groups={
                    slug: {"anchors": 0 if slug == "code" else 10, "ffr": 0.4}
                    for slug in CORPUS_SLUGS
                }
            )
        )


def test_round0218_panel_execution_guard_matches_the_accepted_shape() -> None:
    assert panel_execution_ok(_panel()) is True
    assert (
        panel_execution_ok(
            _panel(
                guards={
                    "coords_finite": True,
                    "coords_collapsed": True,
                    "emb_finite": True,
                    "emb_zero_rows": 0,
                }
            )
        )
        is False
    )
    assert panel_execution_ok({"guards": {}}) is False


def test_round0218_family_evidence_registers_no_gate() -> None:
    evidence = build_family_panel_evidence(_cells())
    assert evidence["gate_registered"] is False
    assert evidence["gate_registerable_here"] is False
    assert evidence["map_quality_claim_available"] is False
    assert evidence["training_performed"] is False
    assert evidence["density_v2_role"] == "diagnostic-only, transcribed"
    assert evidence["seeds"] == list(SEEDS)
    assert evidence["n"] == 4
    assert set(evidence["panel_metric_cells"]) == {str(seed) for seed in SEEDS}
    assert set(evidence["corpus_ffr_cells"]["42"]) == set(CORPUS_SLUGS)


def test_round0218_family_evidence_requires_all_four_cells() -> None:
    cells = _cells()
    cells.pop(45)
    with pytest.raises(Round0218Error):
        build_family_panel_evidence(cells)


def test_round0218_family_evidence_rejects_a_mislabelled_cell() -> None:
    cells = _cells()
    cells[44]["capability"] = map_capability(45)
    with pytest.raises(Round0218Error):
        build_family_panel_evidence(cells)


def test_round0218_summaries_are_descriptive_and_state_their_n() -> None:
    evidence = build_family_panel_evidence(_cells())
    summaries = evidence["descriptive_summaries"]
    assert set(summaries) == set(PANEL_METRICS)
    for metric, cell in summaries.items():
        assert cell["n"] == 4
        assert cell["seed_order"] == list(SEEDS)
        assert len(cell["values"]) == 4
        assert math.isclose(cell["mean"], sum(cell["values"]) / 4, rel_tol=1e-12)
        if metric in DIAGNOSTIC_METRICS:
            assert cell["role"] == "diagnostic-only, transcribed"
        else:
            assert "gate-eligible population" in cell["role"]
    # sample sd, not population sd: the four density values are an arithmetic
    # progression of step 0.01, whose ddof=1 standard deviation is exact.
    assert math.isclose(
        summaries["density_v2"]["sample_sd_ddof1"],
        math.sqrt(sum((x - 0.015) ** 2 for x in (0.0, 0.01, 0.02, 0.03)) / 3),
        rel_tol=1e-9,
    )


def test_round0218_summaries_reject_a_nonfinite_cell() -> None:
    with pytest.raises(Round0218Error):
        descriptive_summaries(
            {
                str(seed): {
                    "density_v2": float("nan") if seed == 43 else 0.6,
                    "ffr": 0.4,
                    "purity_fidelity_k256": 0.9,
                    "purity_fidelity_k1024": 0.9,
                }
                for seed in SEEDS
            }
        )


def test_round0218_map_capability_rejects_a_foreign_seed() -> None:
    assert map_capability(42) == "minilm-mixed-2m-map-seed42-low-dose-v1"
    with pytest.raises(Round0218Error):
        map_capability(99)
