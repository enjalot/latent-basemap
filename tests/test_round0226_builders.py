"""R0226 — the merge, the guard, the ladder, the flatness rule, the verdict.

These run CPU-only and are the pre-launch smoke. The merge tests are the load
bearing ones: R0209 shipped a merge that ranked FAISS's `-1` sentinel slots as
though they were neighbours, and candidate B reuses that code path's semantics.
"""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0226_graph_builders import (
    A_CLUSTER_CAPACITY_ROWS,
    A_MIN_CLUSTERS,
    A_SPILL,
    B_SHARD_ROWS,
    CANDIDATE_A,
    CANDIDATE_B,
    DEVICE_TOTAL_BYTES,
    FLATNESS_TOLERANCE,
    GRAPH_K,
    GUARD_DEVICE_BUDGET_BYTES,
    INSTRUMENTS,
    INSTRUMENT_APPLICABILITY,
    LADDER_ROWS,
    PHASE2_RUNGS,
    PROJECTION_ROWS,
    Round0226Error,
    a_cluster_count,
    a_spill_groups,
    b_shard_count,
    device_verdict_at_100m,
    flatness,
    guard_decision,
    ladder_settings,
    merge_into_topk,
    power_law,
    predict_footprint,
    project_wall,
    rung_recommendation,
)
from experiments.round0226_nodes import (
    refused_cell,
    run_ascending_ladder,
    skipped_cell,
)


def _empty(rows: int, k: int = 3):
    return (
        np.full((rows, k), -1, dtype=np.int32),
        np.full((rows, k), -np.inf, dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# the merge
# --------------------------------------------------------------------------- #
def test_merge_takes_the_exact_top_k():
    ids, cos = _empty(1)
    merge_into_topk(
        ids,
        cos,
        rows=np.array([0]),
        candidate_ids=np.array([[5, 7, 9, 11]]),
        candidate_cos=np.array([[0.1, 0.9, 0.5, 0.3]]),
        k=3,
    )
    assert ids[0].tolist() == [7, 9, 11]
    assert cos[0].tolist() == pytest.approx([0.9, 0.5, 0.3])


def test_merge_excludes_the_faiss_minus_one_sentinel():
    """R0209's bug: a -1 slot must never be ranked as a neighbour."""
    ids, cos = _empty(1)
    merge_into_topk(
        ids,
        cos,
        rows=np.array([0]),
        # FAISS pairs an unfilled slot with a huge sentinel score. If it were
        # ranked it would take the top position.
        candidate_ids=np.array([[-1, 4, -1, 6]]),
        candidate_cos=np.array([[3.4e38, 0.2, 3.4e38, 0.1]]),
        k=3,
    )
    assert ids[0].tolist() == [4, 6, -1]
    assert cos[0][2] == -np.inf


def test_merge_reemits_unfilled_slots_as_minus_one():
    ids, cos = _empty(1)
    merge_into_topk(
        ids,
        cos,
        rows=np.array([0]),
        candidate_ids=np.array([[8, -1, -1, -1]]),
        candidate_cos=np.array([[0.5, 0.0, 0.0, 0.0]]),
        k=3,
    )
    assert ids[0].tolist() == [8, -1, -1]


def test_merge_excludes_self_and_collapses_duplicates():
    ids, cos = _empty(3)
    merge_into_topk(
        ids,
        cos,
        rows=np.array([1]),
        candidate_ids=np.array([[1, 4, 4, 4]]),
        candidate_cos=np.array([[1.0, 0.6, 0.6, 0.6]]),
        k=3,
    )
    assert ids[1].tolist() == [4, -1, -1]


def test_merge_breaks_ties_on_lower_global_id():
    ids, cos = _empty(1, 2)
    merge_into_topk(
        ids,
        cos,
        rows=np.array([0]),
        candidate_ids=np.array([[9, 2, 5]]),
        candidate_cos=np.array([[0.5, 0.5, 0.5]]),
        k=2,
    )
    assert ids[0].tolist() == [2, 5]


def test_incremental_merge_equals_one_shot_merge():
    """Top-k over a union is associative; the sharded merge must prove it."""
    rng = np.random.default_rng(226)
    rows, width = 64, 8
    all_ids = rng.integers(0, 500, size=(rows, 4 * width)).astype(np.int64)
    all_cos = rng.random((rows, 4 * width))
    row_index = np.arange(rows, dtype=np.int64)

    one_ids, one_cos = _empty(rows, GRAPH_K)
    merge_into_topk(
        one_ids, one_cos, rows=row_index, candidate_ids=all_ids,
        candidate_cos=all_cos, k=GRAPH_K,
    )
    many_ids, many_cos = _empty(rows, GRAPH_K)
    for block in range(4):
        merge_into_topk(
            many_ids,
            many_cos,
            rows=row_index,
            candidate_ids=all_ids[:, block * width : (block + 1) * width],
            candidate_cos=all_cos[:, block * width : (block + 1) * width],
            k=GRAPH_K,
        )
    assert np.array_equal(one_ids, many_ids)
    assert np.allclose(one_cos, many_cos)


def test_merge_rejects_a_geometry_change():
    ids, cos = _empty(2)
    with pytest.raises(Round0226Error):
        merge_into_topk(
            ids, cos, rows=np.array([0, 1]),
            candidate_ids=np.zeros((1, 3), dtype=np.int64),
            candidate_cos=np.zeros((1, 3)), k=3,
        )


# --------------------------------------------------------------------------- #
# shape of the design
# --------------------------------------------------------------------------- #
def test_cluster_count_never_degenerates_to_the_spill_factor():
    for rows in LADDER_ROWS + (PROJECTION_ROWS,):
        assert a_cluster_count(rows) >= A_MIN_CLUSTERS > A_SPILL


def test_spill_groups_bound_the_scratch_and_grow_with_n():
    assert a_spill_groups(2_000_000) == 1
    assert a_spill_groups(PROJECTION_ROWS) > a_spill_groups(16_000_000)


def test_shard_count_is_ceil_of_rows_over_shard_rows():
    assert b_shard_count(B_SHARD_ROWS) == 1
    assert b_shard_count(B_SHARD_ROWS + 1) == 2
    assert b_shard_count(PROJECTION_ROWS) == PROJECTION_ROWS // B_SHARD_ROWS


def test_ladder_is_ascending_per_candidate_and_marks_the_recall_rung():
    settings = ladder_settings()
    assert len(settings) == len(LADDER_ROWS) * 2
    for candidate in (CANDIDATE_A, CANDIDATE_B):
        rows = [item["rows"] for item in settings if item["candidate"] == candidate]
        assert rows == sorted(rows)
        emitting = [
            item["rows"]
            for item in settings
            if item["candidate"] == candidate and item["emit_graph"]
        ]
        assert emitting == [2_000_000]


def test_every_registered_instrument_has_a_declared_applicability():
    assert set(INSTRUMENT_APPLICABILITY) == set(INSTRUMENTS)
    assert INSTRUMENT_APPLICABILITY["rmm_peak_bytes"] == CANDIDATE_A


# --------------------------------------------------------------------------- #
# the guard
# --------------------------------------------------------------------------- #
def test_device_prediction_does_not_grow_with_n():
    for candidate in (CANDIDATE_A, CANDIDATE_B):
        small = predict_footprint(candidate=candidate, rows=2_000_000)
        big = predict_footprint(candidate=candidate, rows=PROJECTION_ROWS)
        assert small["predicted_device_bytes"] == big["predicted_device_bytes"]


def test_host_anonymous_prediction_does_grow_with_n():
    for candidate in (CANDIDATE_A, CANDIDATE_B):
        small = predict_footprint(candidate=candidate, rows=2_000_000)
        big = predict_footprint(candidate=candidate, rows=PROJECTION_ROWS)
        assert big["predicted_host_anon_bytes"] > small["predicted_host_anon_bytes"]


def test_guard_allows_every_registered_cell_and_the_100m_projection():
    for candidate in (CANDIDATE_A, CANDIDATE_B):
        for rows in LADDER_ROWS + (PROJECTION_ROWS,):
            assert guard_decision(candidate=candidate, rows=rows)["allowed"]


def test_guard_refuses_a_cell_over_the_device_budget():
    decision = guard_decision(
        candidate=CANDIDATE_B, rows=4_000_000, device_budget_bytes=1024
    )
    assert decision["refused_a_priori"] and not decision["allowed"]
    assert decision["refusal_reasons"]


def test_guard_charges_the_largest_admissible_cluster_not_the_target():
    prediction = predict_footprint(candidate=CANDIDATE_A, rows=8_000_000)
    assert (
        prediction["terms"]["cluster_capacity_rows"] == A_CLUSTER_CAPACITY_ROWS
    )
    assert prediction["predicted_device_bytes"] <= GUARD_DEVICE_BUDGET_BYTES


def test_unknown_candidate_is_rejected():
    with pytest.raises(Round0226Error):
        predict_footprint(candidate="not-a-builder", rows=1000)


# --------------------------------------------------------------------------- #
# the ascending ladder
# --------------------------------------------------------------------------- #
def test_ladder_stops_a_candidate_at_its_first_failure_and_records_the_skips():
    settings = ladder_settings()
    attempted: list[str] = []

    def make_config(setting):
        return {
            "setting_id": setting["id"],
            "candidate": setting["candidate"],
            "rows": setting["rows"],
            "dimension": setting["dimension"],
            "k": setting["k"],
            "substrate": setting["substrate"],
        }

    def run_cell(config, _setting):
        attempted.append(config["setting_id"])
        fit = not (
            config["candidate"] == CANDIDATE_A and config["rows"] >= 4_000_000
        )
        return {
            "setting_id": config["setting_id"],
            "candidate": config["candidate"],
            "rows": config["rows"],
            "fit": fit,
            "error_type": None if fit else "SimulatedOOM",
        }

    records = run_ascending_ladder(
        settings=settings, make_config=make_config, run_cell=run_cell
    )
    assert len(records) == len(settings)
    # A stops after its 4M failure; B runs its whole ladder.
    assert attempted == [
        f"{CANDIDATE_A}-n2000000",
        f"{CANDIDATE_A}-n4000000",
    ] + [f"{CANDIDATE_B}-n{rows}" for rows in LADDER_ROWS]
    skipped = [
        item for item in records if item.get("skipped_after_failure_at_smaller_n")
    ]
    assert [item["rows"] for item in skipped] == [8_000_000, 16_000_000]


def test_refused_and_skipped_cells_carry_every_instrument_key():
    config = {
        "setting_id": "x",
        "candidate": CANDIDATE_B,
        "rows": 2_000_000,
        "dimension": 384,
        "k": GRAPH_K,
        "substrate": "/dev/null",
    }
    guard = guard_decision(
        candidate=CANDIDATE_B, rows=2_000_000, device_budget_bytes=1024
    )
    for record in (refused_cell(config, guard), skipped_cell(config, "because")):
        for instrument in INSTRUMENTS:
            assert instrument in record
        assert record["rmm_peak_bytes"] is None


# --------------------------------------------------------------------------- #
# flatness, verdict, projections
# --------------------------------------------------------------------------- #
def test_flatness_accepts_a_plateau_and_rejects_a_ramp():
    plateau = flatness([4.00e9, 4.01e9, 4.02e9, 4.00e9])
    assert plateau["flat"] and plateau["relative_spread"] <= FLATNESS_TOLERANCE
    ramp = flatness([2.0e9, 4.0e9, 8.0e9, 16.0e9])
    assert not ramp["flat"]


def test_flat_verdict_publishes_a_plateau_and_never_extrapolates():
    verdict = device_verdict_at_100m(
        candidate=CANDIDATE_B,
        rows=[2_000_000, 4_000_000, 8_000_000, 16_000_000],
        device_peaks=[4.00e9, 4.01e9, 4.02e9, 4.00e9],
    )
    assert verdict["is_extrapolation"] is False
    assert verdict["extrapolation_factor"] == 1.0
    assert verdict["device_bytes_at_100m"] == pytest.approx(4.02e9)
    assert verdict["fits_100m"] is True
    assert verdict["headroom_gib"] > 0


def test_non_flat_verdict_states_its_range_and_extrapolation_factor():
    verdict = device_verdict_at_100m(
        candidate=CANDIDATE_A,
        rows=[2_000_000, 4_000_000, 8_000_000],
        device_peaks=[2.096e9, 4.192e9, 8.384e9],
    )
    assert verdict["is_extrapolation"] is True
    assert verdict["bytes_per_row"] == pytest.approx(1048.0, rel=1e-6)
    assert verdict["extrapolation_factor"] == pytest.approx(12.5)
    assert verdict["fitted_range_rows"] == [2_000_000, 8_000_000]
    # 1048 B/row at 100M is 97.6 GiB, far over a 31.37 GiB card.
    assert verdict["fits_100m"] is False
    assert verdict["device_gib_at_100m"] == pytest.approx(97.6, abs=0.2)
    assert DEVICE_TOTAL_BYTES / 1024 ** 3 == pytest.approx(31.37)


def test_wall_projection_is_labelled_and_carries_its_own_fit():
    fit = power_law([2_000_000, 4_000_000, 8_000_000], [10.0, 40.0, 160.0])
    assert fit["exponent"] == pytest.approx(2.0, rel=1e-6)
    projection = project_wall(fit)
    assert projection["is_projection"] is True
    assert projection["extrapolation_factor"] == pytest.approx(12.5)
    assert projection["fitted_range_rows"] == [2_000_000, 8_000_000]


def test_power_law_rejects_a_single_point():
    with pytest.raises(Round0226Error):
        power_law([2_000_000], [10.0])


def test_recommendation_covers_every_phase2_rung_and_prefers_the_faster_builder():
    verdicts = {
        CANDIDATE_A: {"fits_100m": True, "device_gib_at_100m": 11.0,
                      "headroom_gib": 20.0, "method": "measured plateau"},
        CANDIDATE_B: {"fits_100m": True, "device_gib_at_100m": 4.0,
                      "headroom_gib": 27.0, "method": "measured plateau"},
    }
    recalls = {
        CANDIDATE_A: {"zero_degree_rows": 0, "clears_recall_floor": True,
                      "wall_seconds_at_largest_n": 100.0},
        CANDIDATE_B: {"zero_degree_rows": 0, "clears_recall_floor": True,
                      "wall_seconds_at_largest_n": 900.0},
    }
    out = rung_recommendation(verdicts=verdicts, recalls=recalls)
    assert set(out["rungs"]) == {str(rung) for rung in PHASE2_RUNGS}
    assert all(item["builder"] == CANDIDATE_A for item in out["rungs"].values())


def test_a_candidate_with_edgeless_rows_is_never_recommended():
    verdicts = {
        CANDIDATE_A: {"fits_100m": True, "headroom_gib": 20.0},
        CANDIDATE_B: {"fits_100m": True, "headroom_gib": 27.0},
    }
    recalls = {
        CANDIDATE_A: {"zero_degree_rows": 17, "clears_recall_floor": True,
                      "wall_seconds_at_largest_n": 1.0},
        CANDIDATE_B: {"zero_degree_rows": 0, "clears_recall_floor": True,
                      "wall_seconds_at_largest_n": 900.0},
    }
    out = rung_recommendation(verdicts=verdicts, recalls=recalls)
    assert out["eligible_candidates"] == [CANDIDATE_B]
    assert all(item["builder"] == CANDIDATE_B for item in out["rungs"].values())


def test_no_candidate_qualifying_is_reported_rather_than_papered_over():
    verdicts = {CANDIDATE_A: {"fits_100m": False}, CANDIDATE_B: {"fits_100m": False}}
    recalls = {
        CANDIDATE_A: {"zero_degree_rows": 0, "clears_recall_floor": True},
        CANDIDATE_B: {"zero_degree_rows": 0, "clears_recall_floor": False},
    }
    out = rung_recommendation(verdicts=verdicts, recalls=recalls)
    assert out["eligible_candidates"] == []
    assert all(item["builder"] is None for item in out["rungs"].values())


# --------------------------------------------------------------------------- #
# the watchdog's own lifecycle
# --------------------------------------------------------------------------- #
def test_watchdog_starts_and_halts_without_shadowing_thread_internals():
    """A regression test for a bug the undefined-name guard cannot catch.

    Naming the stop flag `_stop` shadows `threading.Thread._stop`, and `join()`
    then raises `TypeError: 'Event' object is not callable` *after* the build has
    already run — the most expensive place to discover it.
    """
    import os as _os

    from experiments.round0226_nodes import BuildWatchdog

    watchdog = BuildWatchdog(
        pid=_os.getpid(),
        poll_s=0.01,
        host_anon_budget_bytes=1 << 60,
        swap_growth_abort_bytes=1 << 60,
        device_baseline_bytes=0,
        swap_baseline_bytes=0,
    )
    watchdog.start()
    watchdog.halt()
    assert not watchdog.is_alive()
    readings = watchdog.readings()
    assert readings["watchdog_aborted"] is False
    assert readings["watchdog_escalations"] == []
    assert readings["watchdog_samples"] >= 1
    assert readings["host_anon_peak_bytes"] > 0
