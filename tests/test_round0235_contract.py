"""R0235 contract tests — nesting, reserves, span, the law, and the margin.

Every test here corresponds to a registered fail-closed check. The point of the
suite is that each one RAISES on the defect it exists to catch: review-0233-01
found the 25M/50M pricing wrong because a margin was applied in one place and
silently dropped in another, and found a device law called homogeneous that was
not, so both properties are tested directly rather than inspected.
"""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0235_rung2 import (
    COMPOSITION,
    CONTROL_CLUSTERS,
    C_MIN,
    GRAPH_DEGREE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_IMBALANCE_MARGIN,
    IMBALANCE_PROBE_CLUSTERS,
    INCREMENT_BY_CORPUS,
    INTERMEDIATE_GRAPH_DEGREE,
    LAW_RESIDUAL_MARGIN,
    MAX_ITERATIONS,
    PARENT_COMPOSITION,
    PARENT_ROWS,
    R0229_2M_S8_IMBALANCE_REFERENCE,
    RESERVE_ROWS,
    RESERVE_ROWS_PER_CORPUS,
    ROWS,
    Round0235Error,
    SELECTION_CANDIDATES,
    SPILL,
    admissible_max_cluster_rows,
    admit_law_point,
    assert_memmap_for_cuvs,
    assert_nesting,
    assert_no_signal_policy,
    assert_reserve_disjoint,
    fit_device_law,
    guard_decision,
    guarded_max_cluster_rows,
    imbalance_drift,
    provenance_keys,
    rung_derivation,
    select_clusters,
    validate_composition,
    validate_shard_span,
)

PROV_DTYPE = np.dtype([("corpus", "u1"), ("shard", "u2"), ("row", "i8")])


def _records(triples):
    out = np.empty(len(triples), dtype=PROV_DTYPE)
    for index, (corpus, shard, row) in enumerate(triples):
        out[index] = (corpus, shard, row)
    return out


def _law(slope=1678.267781, intercept=7.513210e9):
    return {
        "label": "test",
        "slope_bytes_per_max_cluster_row": slope,
        "intercept_bytes": intercept,
    }


# --------------------------------------------------------------------------- #
# composition and nesting
# --------------------------------------------------------------------------- #
def test_composition_is_the_registered_mix_and_doubles_rung_one():
    assert sum(rows for _n, rows in COMPOSITION) == ROWS == 2 * PARENT_ROWS
    for name, rows in COMPOSITION:
        assert rows == 2 * dict(PARENT_COMPOSITION)[name]
        assert INCREMENT_BY_CORPUS[name] == dict(PARENT_COMPOSITION)[name]
    observed = validate_composition({name: rows for name, rows in COMPOSITION})
    assert observed[COMPOSITION[0][0]]["rows"] == 5_000_000


def test_composition_refuses_a_short_corpus():
    counts = {name: rows for name, rows in COMPOSITION}
    counts[COMPOSITION[1][0]] -= 1
    with pytest.raises(Round0235Error):
        validate_composition(counts)


def test_nesting_holds_for_a_true_prefix():
    parent = _records([(0, 0, 5), (0, 1, 7), (1, 3, 11)])
    child = _records([(0, 0, 5), (0, 1, 7), (1, 3, 11), (0, 2, 9), (2, 4, 1)])
    record = assert_nesting(parent=parent, child=child)
    assert record["parent_rows_missing_from_child"] == 0
    assert record["positional_prefix"] is True


def test_nesting_raises_when_a_parent_row_is_absent():
    parent = _records([(0, 0, 5), (0, 1, 7)])
    child = _records([(0, 0, 5), (0, 2, 9)])
    with pytest.raises(Round0235Error, match="not nested"):
        assert_nesting(parent=parent, child=child)


def test_nesting_raises_when_the_prefix_is_permuted():
    parent = _records([(0, 0, 5), (0, 1, 7)])
    child = _records([(0, 1, 7), (0, 0, 5), (1, 2, 3)])
    with pytest.raises(Round0235Error, match="positional"):
        assert_nesting(parent=parent, child=child)


def test_nesting_raises_on_a_duplicated_source_row():
    parent = _records([(0, 0, 5)])
    child = _records([(0, 0, 5), (0, 0, 5)])
    with pytest.raises(Round0235Error, match="duplicated"):
        assert_nesting(parent=parent, child=child)


def test_provenance_keys_are_injective_over_the_registered_ranges():
    records = _records([(0, 0, 0), (0, 0, 1), (0, 1, 0), (3, 176, 227_613_719)])
    keys = provenance_keys(records)
    assert np.unique(keys).size == keys.size


# --------------------------------------------------------------------------- #
# the reserve
# --------------------------------------------------------------------------- #
def test_reserve_disjointness_passes_and_raises():
    training = _records([(index, 0, index) for index in range(4)])
    reserve = _records([(index, 0, 100 + index) for index in range(4)])
    record = assert_reserve_disjoint(training=training, reserve=reserve)
    assert record["global_intersection_rows"] == 0
    shared = _records([(0, 0, 0), (1, 0, 101), (2, 0, 102), (3, 0, 103)])
    with pytest.raises(Round0235Error):
        assert_reserve_disjoint(training=training, reserve=shared)


def test_reserve_sizes_are_r0233s_scheme_unchanged():
    assert RESERVE_ROWS_PER_CORPUS == 50_000
    assert RESERVE_ROWS == RESERVE_ROWS_PER_CORPUS * len(COMPOSITION) == 200_000


# --------------------------------------------------------------------------- #
# span
# --------------------------------------------------------------------------- #
def test_shard_span_raises_on_a_forced_prefix():
    validate_shard_span(
        corpus="c", shards_touched=98, shards_total=98, label="union"
    )
    with pytest.raises(Round0235Error, match="SPAN"):
        validate_shard_span(
            corpus="c", shards_touched=90, shards_total=98, label="increment"
        )


# --------------------------------------------------------------------------- #
# the device law — D3
# --------------------------------------------------------------------------- #
def test_law_refuses_a_point_from_another_nn_descent_setting():
    good = {
        "graph_degree": GRAPH_DEGREE,
        "intermediate_graph_degree": INTERMEDIATE_GRAPH_DEGREE,
        "max_iterations": MAX_ITERATIONS,
        "max_cluster_rows": 318_519, "device_bytes": 7_940_866_048.0,
    }
    # R0229 q6: the exact cell R0233 pulled in from a review's prose.
    q6 = {**good, "intermediate_graph_degree": 128, "max_iterations": 20,
          "device_bytes": 7_957_643_264.0}
    assert admit_law_point(good)["admitted"] is True
    refused = admit_law_point(q6)
    assert refused["admitted"] is False
    assert "not the registered" in refused["refusal_reasons"][0]


def test_law_fit_reproduces_the_sealed_six_point_line():
    points = [
        {"max_cluster_rows": rows, "device_bytes": device,
         "graph_degree": GRAPH_DEGREE,
         "intermediate_graph_degree": INTERMEDIATE_GRAPH_DEGREE,
         "max_iterations": MAX_ITERATIONS}
        for rows, device in (
            (170_504, 7_470_055_424.0),
            (318_519, 7_940_866_048.0),
            (532_626, 8_707_375_104.0),
            (1_248_823, 9_820_962_816.0),
            (2_496_850, 11_752_439_808.0),
            (3_656_227, 13_524_533_248.0),
        )
    ]
    law = fit_device_law(points, label="six")
    assert law["n_points"] == 6
    assert law["slope_bytes_per_max_cluster_row"] == pytest.approx(1678.2678, rel=1e-5)
    assert law["intercept_bytes"] == pytest.approx(7.513210e9, rel=1e-6)
    assert law["worst_absolute_relative_residual"] < 0.05
    # An off-setting point is dropped, not fitted, and the fit is unchanged.
    with_q6 = fit_device_law(
        points + [{"max_cluster_rows": 318_519, "device_bytes": 7_957_643_264.0,
                   "graph_degree": 64, "intermediate_graph_degree": 128,
                   "max_iterations": 20}],
        label="six-plus-q6",
    )
    assert with_q6["n_points"] == 6
    assert len(with_q6["points_refused"]) == 1


def test_law_needs_two_admissible_points():
    with pytest.raises(Round0235Error):
        fit_device_law([
            {"max_cluster_rows": 1, "device_bytes": 1.0, "graph_degree": 64,
             "intermediate_graph_degree": 256, "max_iterations": 40},
        ], label="one")


# --------------------------------------------------------------------------- #
# the margin reaches BOTH decisions — D1
# --------------------------------------------------------------------------- #
def test_guarded_max_cluster_applies_the_margin():
    plain = 12_500_000 * SPILL / 16 * 1.17
    assert guarded_max_cluster_rows(
        rows=12_500_000, clusters=16, imbalance=1.17
    ) == pytest.approx(plain * GUARD_IMBALANCE_MARGIN)


def test_rung_derivation_uses_the_same_margin_as_the_launch_guard():
    laws = [_law()]
    imbalance = {16: 1.16998304, 32: 1.59797632, 64: 1.59898624, 200: 2.130548}
    derived = rung_derivation(
        rung=12_500_000, imbalance_by_c=imbalance,
        imbalance_source="test", laws=laws,
    )
    guard = guard_decision(
        rows=12_500_000, clusters=int(derived["selected_clusters"]),
        imbalance=imbalance[int(derived["selected_clusters"])],
        imbalance_source="test", laws=laws,
    )
    assert derived["imbalance_margin_applied"] == GUARD_IMBALANCE_MARGIN
    assert derived["selection"]["guarded_max_cluster_rows"] == pytest.approx(
        guard["prediction"]["guarded_max_cluster_rows"]
    )
    assert derived["selection"]["device_bytes"] == pytest.approx(
        guard["prediction"]["predicted_device_bytes"]
    )


def test_the_margin_is_what_turns_25m_at_c32_infeasible():
    """R0233 priced 25M at c = 32; with its own margin that cell does not fit."""
    laws = [_law()]
    imbalance = {16: 1.16998304, 32: 1.59797632, 64: 1.59898624, 200: 2.130548}
    with_margin = rung_derivation(
        rung=25_000_000, imbalance_by_c=imbalance, imbalance_source="test",
        laws=laws, apply_margin=True,
    )
    without = rung_derivation(
        rung=25_000_000, imbalance_by_c=imbalance, imbalance_source="test",
        laws=laws, apply_margin=False,
    )
    assert without["selected_clusters"] == 32
    assert with_margin["selected_clusters"] == 64
    entry = next(
        item for item in with_margin["candidates_considered"]
        if item["clusters"] == 32
    )
    assert entry["admissible"] is False


def test_guard_refuses_a_cell_that_cannot_fit_and_records_the_reason():
    laws = [_law()]
    decision = guard_decision(
        rows=100_000_000, clusters=16, imbalance=1.17,
        imbalance_source="test", laws=laws,
    )
    assert decision["allowed"] is False
    assert decision["refused_a_priori"] is True
    assert decision["refusal_reasons"]


def test_admissible_max_cluster_is_the_budget_solved_for_rows():
    law = _law()
    admissible = admissible_max_cluster_rows([law])
    charged = (
        law["intercept_bytes"]
        + law["slope_bytes_per_max_cluster_row"] * admissible
    ) * (1.0 + LAW_RESIDUAL_MARGIN) + 1024 ** 3
    assert charged == pytest.approx(float(GUARD_DEVICE_BUDGET_BYTES), rel=1e-9)


def test_selection_takes_the_smallest_admissible_c_at_or_above_c_min():
    laws = [_law()]
    imbalance = {16: 1.16998304, 32: 1.59797632, 64: 1.59898624, 200: 2.130548}
    chosen = select_clusters(rows=12_500_000, measured_imbalance=imbalance, laws=laws)
    assert chosen["selected_clusters"] == 16
    assert C_MIN == 2 * SPILL == 16
    assert set(SELECTION_CANDIDATES) <= set(IMBALANCE_PROBE_CLUSTERS)
    assert set(CONTROL_CLUSTERS) <= set(IMBALANCE_PROBE_CLUSTERS)


def test_selection_refuses_a_c_with_no_measurement_at_this_n():
    laws = [_law()]
    chosen = select_clusters(
        rows=12_500_000, measured_imbalance={200: 2.130548}, laws=laws
    )
    assert chosen["selected_clusters"] == 200
    reasons = [
        entry.get("reason") for entry in chosen["candidates_considered"]
        if entry["clusters"] in (16, 32, 64)
    ]
    assert all("no imbalance measured" in str(value) for value in reasons)


# --------------------------------------------------------------------------- #
# the drift table — D2
# --------------------------------------------------------------------------- #
def test_drift_table_reports_absence_rather_than_interpolating():
    drift = imbalance_drift({
        2_000_000: R0229_2M_S8_IMBALANCE_REFERENCE,
        6_250_000: {16: 1.16998304, 32: 1.59797632, 64: 1.59898624,
                    200: 2.130548},
        12_500_000: {16: 1.17, 32: 1.60, 64: 1.61, 200: 2.13},
    })
    assert drift["by_clusters"]["32"]["by_rows"]["2000000"] is None
    assert drift["by_clusters"]["32"]["measured_at_rows"] == [6_250_000, 12_500_000]
    assert drift["by_clusters"]["64"]["drift_relative"] == pytest.approx(
        1.61 / 1.539004 - 1.0
    )
    assert drift["by_clusters"]["200"]["drift_relative"] < 0.01


# --------------------------------------------------------------------------- #
# safety preconditions, inherited unchanged
# --------------------------------------------------------------------------- #
def test_memmap_precondition_raises_on_an_anonymous_array(tmp_path):
    with pytest.raises(Exception):
        assert_memmap_for_cuvs(np.zeros((4, 4), dtype=np.float32), label="anon")
    path = tmp_path / "m.f32"
    np.memmap(path, dtype="<f4", mode="w+", shape=(4, 4)).flush()
    read_only = np.memmap(path, dtype="<f4", mode="r", shape=(4, 4))
    assert_memmap_for_cuvs(read_only, label="ok")
    writable = np.memmap(path, dtype="<f4", mode="r+", shape=(4, 4))
    with pytest.raises(Exception):
        assert_memmap_for_cuvs(writable, label="writable")


def test_signal_policy_raises_on_any_recorded_signal():
    assert_no_signal_policy([])
    assert_no_signal_policy(["cooperative-flag"])
    with pytest.raises(Exception):
        assert_no_signal_policy(["escalated to SIGTERM"])
    with pytest.raises(Exception):
        assert_no_signal_policy(["SIGKILL after grace"])
