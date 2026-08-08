"""Registered checks for the R0222 n=8 MiniLM gate registration.

Two properties carry this round and both are asserted structurally, so a later
edit that reintroduces R0219's defects fails rather than silently ships:

1. `n = 8`, and the cells-clearing count is only reported as informative where
   the identity bound `(n-1)/sqrt(n)` actually exceeds the multiplier.
2. **No metric this panel computes is withheld from the gate** — `density_v2`
   included — and the two accepted metrics that are missing are recorded as
   unavailable-by-panel, not excluded-by-judgement.
"""
from __future__ import annotations

import math
import statistics

import pytest

from basemap.round0218_minilm_2m_panel import PANEL_METRICS, SEEDS as R0218_SEEDS
from basemap.round0221_minilm_2m_seed_extension import SEEDS as R0221_SEEDS
from basemap.round0222_minilm_2m_gate_n8 import (
    ACCEPTED_SIX_METRIC_SET,
    EXCLUDED_BY_JUDGEMENT,
    GATE_METRICS,
    MULTIPLIER,
    N_REQUIRED,
    POOLED_SEEDS,
    PRECEDENT_CAPABILITIES,
    RETRACTED_CLAIM,
    Round0222Error,
    UNAVAILABLE_METRICS,
    assert_density_v2_is_gated_in_precedent,
    gate_cell,
    identity_bound,
    jackknife,
    register_minilm_gates_n8,
)


#: R0218's four published cells (result-0218-2026-08-08), used as the fixed part
#: of the pooled family in these tests. The R0221 half is synthetic here; the
#: real values come from the run.
R0218_CELLS = {
    "42": {
        "density_v2": 0.4377,
        "ffr": 0.3369,
        "purity_fidelity_k256": 0.9788566953797964,
        "purity_fidelity_k1024": 0.7326,
    },
    "43": {
        "density_v2": 0.4406,
        "ffr": 0.3382,
        "purity_fidelity_k256": 0.9941346058256287,
        "purity_fidelity_k1024": 0.7229,
    },
    "44": {
        "density_v2": 0.4387,
        "ffr": 0.3258,
        "purity_fidelity_k256": 0.9954210631096955,
        "purity_fidelity_k1024": 0.6980,
    },
    "45": {
        "density_v2": 0.4477,
        "ffr": 0.3227,
        "purity_fidelity_k256": 0.9929,
        "purity_fidelity_k1024": 0.6936,
    },
}

CORPUS_SLUGS_ORDER = ("fineweb", "redpajama", "pile", "code")


def _pooled_cells() -> dict:
    cells = {seed: dict(values) for seed, values in R0218_CELLS.items()}
    for index, seed in enumerate(R0221_SEEDS):
        cells[str(seed)] = {
            "density_v2": 0.4400 + 0.0010 * index,
            "ffr": 0.3300 + 0.0020 * index,
            "purity_fidelity_k256": 0.9880 + 0.0010 * index,
            "purity_fidelity_k1024": 0.7050 + 0.0030 * index,
        }
    return cells


def _corpus_cells() -> dict:
    return {
        str(seed): {
            slug: {"anchors": 400 + 100 * index, "ffr": 0.30 + 0.001 * (seed % 7)}
            for index, slug in enumerate(CORPUS_SLUGS_ORDER)
        }
        for seed in POOLED_SEEDS
    }


def _precedents(
    *, r0161_keys=None, r0193_keys=None, r0161_floor=0.19134355783912885
) -> dict:
    def artifact(capability, keys, floor):
        return {
            "capability": capability,
            "formula": "family mean - 2 * sample standard deviation (ddof=1)",
            "n": 4,
            "seed_family": [42, 43, 44, 45],
            "gates": {
                key: {"floor": floor, "mean": floor + 0.02, "sample_sd_ddof1": 0.01}
                for key in (keys or ACCEPTED_SIX_METRIC_SET)
            },
        }

    return {
        "0161": artifact(PRECEDENT_CAPABILITIES["0161"], r0161_keys, r0161_floor),
        "0193": artifact(
            PRECEDENT_CAPABILITIES["0193"], r0193_keys, 0.18616941334799972
        ),
    }


def test_round0222_pools_eight_cells() -> None:
    assert N_REQUIRED == 8
    assert POOLED_SEEDS == tuple(R0218_SEEDS) + tuple(R0221_SEEDS)
    assert len(POOLED_SEEDS) == 8


def test_round0222_identity_bound_is_what_makes_n8_informative() -> None:
    assert identity_bound(4) == pytest.approx(1.5)
    assert identity_bound(8) == pytest.approx(2.4748737341529163)
    assert identity_bound(4) < MULTIPLIER < identity_bound(8)


def test_round0222_gates_every_metric_the_panel_computes() -> None:
    available = tuple(m for m in PANEL_METRICS if m in ACCEPTED_SIX_METRIC_SET)
    assert tuple(GATE_METRICS) == available
    assert "density_v2" in GATE_METRICS
    assert EXCLUDED_BY_JUDGEMENT == {}
    assert set(UNAVAILABLE_METRICS) == set(ACCEPTED_SIX_METRIC_SET) - set(
        PANEL_METRICS
    )
    assert set(UNAVAILABLE_METRICS) == {"heldout_recall_at_10", "projection_ffr"}
    assert not set(GATE_METRICS) & set(UNAVAILABLE_METRICS)


def test_round0222_accepted_set_is_the_precedent_set() -> None:
    assert set(ACCEPTED_SIX_METRIC_SET) == {
        "density_v2",
        "ffr",
        "heldout_recall_at_10",
        "projection_ffr",
        "purity_fidelity_k1024",
        "purity_fidelity_k256",
    }


def test_round0222_gate_cell_arithmetic_and_clearing_count() -> None:
    values = [0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.50]
    cell = gate_cell("ffr", values, POOLED_SEEDS)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    assert cell["n"] == 8
    assert cell["mean"] == mean
    assert cell["sample_sd_ddof1"] == sd
    assert cell["floor"] == mean - 2.0 * sd
    assert cell["cells_clearing_floor"] + len(cell["seeds_below_floor"]) == 8
    assert cell["cells_clearing_is_informative"] is True
    assert cell["max_abs_z"] <= cell["identity_bound_on_max_abs_z"] + 1e-12


def test_round0222_a_defining_cell_can_fail_at_n8_but_never_at_n4() -> None:
    """The whole reason the round exists, as an executable statement."""
    outlier = [0.30, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34]
    eight = gate_cell("ffr", outlier, POOLED_SEEDS)
    assert eight["cells_clearing_floor"] < 8
    assert eight["seeds_below_floor"] == [42]
    for four in ([0.30, 0.34, 0.34, 0.34], [0.9, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.9]):
        cell = gate_cell("ffr", four, R0218_SEEDS)
        assert cell["cells_clearing_floor"] == 4
        assert cell["cells_clearing_is_informative"] is False
        assert cell["max_abs_z"] <= 1.5 + 1e-12


def test_round0222_density_v2_is_admissible_as_a_correlation() -> None:
    cell = gate_cell("density_v2", [0.44] * 7 + [0.43], POOLED_SEEDS)
    assert math.isfinite(cell["floor"])
    with pytest.raises(Round0222Error):
        gate_cell("density_v2", [1.4] + [0.4] * 7, POOLED_SEEDS)
    with pytest.raises(Round0222Error):
        gate_cell("ffr", [0.0] + [0.4] * 7, POOLED_SEEDS)


def test_round0222_jackknife_reports_single_cell_leverage() -> None:
    values = [0.30, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34, 0.34]
    jk = jackknife("ffr", values, POOLED_SEEDS)
    assert jk["n"] == 8 and jk["leave_one_out_n"] == 7
    assert set(jk["leave_one_out_floors"]) == {str(seed) for seed in POOLED_SEEDS}
    assert jk["loo_range"] > 0
    assert jk["largest_single_cell_shift"] >= 0
    with pytest.raises(Round0222Error):
        jackknife("ffr", [0.3, 0.4], [42, 43])


def test_round0222_retraction_is_read_from_the_precedent_artifacts() -> None:
    evidence = assert_density_v2_is_gated_in_precedent(_precedents())
    assert evidence["density_v2_gated_in_both_precedents"] is True
    assert evidence["retracted_claim"] == RETRACTED_CLAIM
    assert evidence["precedents"]["0161"]["density_v2_floor"] == pytest.approx(
        0.19134355783912885
    )
    assert evidence["precedents"]["0193"]["density_v2_floor"] == pytest.approx(
        0.18616941334799972
    )


def test_round0222_retraction_fails_closed_if_a_precedent_did_not_gate_density() -> None:
    """If R0219's sentence had been true, this round's premise would be wrong."""
    narrowed = tuple(m for m in ACCEPTED_SIX_METRIC_SET if m != "density_v2")
    with pytest.raises(Round0222Error):
        assert_density_v2_is_gated_in_precedent(_precedents(r0161_keys=narrowed))
    with pytest.raises(Round0222Error):
        assert_density_v2_is_gated_in_precedent(_precedents(r0193_keys=narrowed))
    with pytest.raises(Round0222Error):
        assert_density_v2_is_gated_in_precedent({"0161": _precedents()["0161"]})


def test_round0222_registration_reports_n4_and_n8_side_by_side() -> None:
    registration = register_minilm_gates_n8(
        pooled_cells=_pooled_cells(),
        corpus_cells=_corpus_cells(),
        precedents=_precedents(),
    )
    assert registration["n"] == 8
    assert registration["seed_family"] == list(POOLED_SEEDS)
    assert set(registration["gates"]) == set(GATE_METRICS)
    assert set(registration["n4_gates_for_comparison"]) == set(GATE_METRICS)
    assert set(registration["n4_vs_n8"]) == set(GATE_METRICS)
    assert set(registration["jackknife"]) == {"n4", "n8"}
    for metric in GATE_METRICS:
        row = registration["n4_vs_n8"][metric]
        assert row["n4_seeds"] == list(R0218_SEEDS)
        assert row["n8_seeds"] == list(POOLED_SEEDS)
        assert row["n4_cells_clearing"] == 4
        assert row["n4_cells_clearing_is_informative"] is False
        assert row["n8_cells_clearing_is_informative"] is True
        assert row["floor_shift_n4_to_n8"] == pytest.approx(
            row["n8_floor"] - row["n4_floor"]
        )
        # The n=4 floors must reproduce R0219's published values exactly.
        expected = statistics.fmean(
            [R0218_CELLS[str(seed)][metric] for seed in R0218_SEEDS]
        ) - 2.0 * statistics.stdev(
            [R0218_CELLS[str(seed)][metric] for seed in R0218_SEEDS]
        )
        assert row["n4_floor"] == expected
    assert registration["gates"]["density_v2"]["n"] == 8
    assert registration["excluded_by_judgement"] == {}
    assert registration["density_v2_role"].startswith("registered floor")


def test_round0222_reproduces_r0219_published_n4_floors() -> None:
    """A cross-check against numbers an accepted review already recomputed."""
    registration = register_minilm_gates_n8(
        pooled_cells=_pooled_cells(),
        corpus_cells=_corpus_cells(),
        precedents=_precedents(),
    )
    published = {
        "ffr": 0.31529914532255787,
        "purity_fidelity_k256": 0.9748949858216875,
        "purity_fidelity_k1024": 0.6738711303646536,
    }
    for metric, floor in published.items():
        assert registration["n4_vs_n8"][metric]["n4_floor"] == pytest.approx(
            floor, rel=0, abs=1e-15
        )
    # And the density_v2 floor R0219 would have registered had it not narrowed.
    assert registration["n4_vs_n8"]["density_v2"]["n4_floor"] == pytest.approx(
        0.4321485573636864, abs=1e-12
    )


def test_round0222_rejects_an_incomplete_or_foreign_family() -> None:
    cells = _pooled_cells()
    del cells["49"]
    with pytest.raises(Round0222Error):
        register_minilm_gates_n8(
            pooled_cells=cells,
            corpus_cells=_corpus_cells(),
            precedents=_precedents(),
        )
    cells = _pooled_cells()
    cells["42"] = {k: v for k, v in cells["42"].items() if k != "density_v2"}
    with pytest.raises(Round0222Error):
        register_minilm_gates_n8(
            pooled_cells=cells,
            corpus_cells=_corpus_cells(),
            precedents=_precedents(),
        )


def test_round0222_per_corpus_ffr_is_not_a_floor() -> None:
    registration = register_minilm_gates_n8(
        pooled_cells=_pooled_cells(),
        corpus_cells=_corpus_cells(),
        precedents=_precedents(),
    )
    for slug in CORPUS_SLUGS_ORDER:
        row = registration["per_corpus_ffr"][slug]
        assert row["registered_as_floor"] is False
        assert row["n"] == 8
        assert len(row["values"]) == 8
