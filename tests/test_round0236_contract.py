"""R0236 contract tests — nesting, the probe, replicates, tolerance, and I/O.

Every test here corresponds to a registered fail-closed check or to a number
review-0235-01 published, so the suite either RAISES on the defect it exists to
catch or reproduces a figure an independent reviewer already computed by hand.
"""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0236_rung3 import (
    COMPOSITION,
    C_BUILD_MIN,
    C_MIN,
    GRAPH_DEGREE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_IMBALANCE_MARGIN,
    IMBALANCE_PROBE_CLUSTERS,
    IMBALANCE_PROBE_ROWS,
    IMBALANCE_REPLICATE_SEEDS,
    INCREMENT_BY_CORPUS,
    INTERMEDIATE_GRAPH_DEGREE,
    LAW_RESIDUAL_MARGIN,
    MAX_ITERATIONS,
    PAGE_CACHE_BUDGET_BYTES,
    PARENT_COMPOSITION,
    PARENT_ROWS,
    PHASE2_RUNGS,
    PRIMARY_IMBALANCE_SEED,
    RESERVE_ROWS,
    RESERVE_ROWS_PER_CORPUS,
    ROWS,
    Round0236Error,
    SELECTION_CANDIDATES,
    SPILL,
    TRUTH_PROBE_ROWS,
    TRUTH_PROBE_SEED,
    admissible_max_cluster_rows,
    admit_law_point,
    architectural_io,
    assert_memmap_for_cuvs,
    assert_nesting,
    assert_no_signal_policy,
    assert_reserve_disjoint,
    carry_distance,
    fit_device_law,
    guard_decision,
    guarded_max_cluster_rows,
    imbalance_tolerance,
    io_hours,
    io_scaling_fit,
    physical_io_prediction,
    predicted_substrate_passes,
    provenance_keys,
    replicate_drift,
    replicate_summary,
    select_clusters,
    truth_probe_query_rows,
    validate_composition,
    validate_shard_span,
)

PROV_DTYPE = np.dtype([("corpus", "u1"), ("shard", "u2"), ("row", "i8")])

#: The eight sealed `gd 64 / igd 256 / it 40` points review-0235-01 verified, so
#: this suite reproduces its published rung table rather than asserting a new one.
R0235_SEALED_LAW_POINTS = (
    (170_504, 7_470_055_424.0),
    (318_519, 7_940_866_048.0),
    (532_626, 8_707_375_104.0),
    (1_248_823, 9_820_962_816.0),
    (2_496_850, 11_752_439_808.0),
    (2_576_003, 11_865_686_016.0),
    (3_656_227, 13_524_533_248.0),
    (7_275_244, 19_107_151_872.0),
)


def _records(triples):
    out = np.empty(len(triples), dtype=PROV_DTYPE)
    for index, (corpus, shard, row) in enumerate(triples):
        out[index] = (corpus, shard, row)
    return out


def _law():
    return [fit_device_law(
        [
            {"max_cluster_rows": rows, "device_bytes": device,
             "graph_degree": 64, "intermediate_graph_degree": 256,
             "max_iterations": 40}
            for rows, device in R0235_SEALED_LAW_POINTS
        ],
        label="all-sealed-gd64-igd256-it40",
    )]


# --------------------------------------------------------------------------- #
# composition and nesting
# --------------------------------------------------------------------------- #
def test_composition_is_exactly_twice_rung_2_at_the_confirmed_shares():
    assert sum(rows for _n, rows in COMPOSITION) == ROWS == 25_000_000
    assert sum(rows for _n, rows in PARENT_COMPOSITION) == PARENT_ROWS
    for name, rows in COMPOSITION:
        assert rows == 2 * dict(PARENT_COMPOSITION)[name]
        assert INCREMENT_BY_CORPUS[name] == dict(PARENT_COMPOSITION)[name]
    assert [rows for _n, rows in COMPOSITION] == [
        10_000_000, 6_250_000, 6_250_000, 2_500_000
    ]


def test_composition_raises_on_any_drift():
    counts = {name: rows for name, rows in COMPOSITION}
    assert validate_composition(counts)["fineweb-edu-sample-10BT-chunked-120-"
                                        "all-MiniLM-L6-v2"]["rows"] == 10_000_000
    short = dict(counts)
    short[COMPOSITION[0][0]] -= 1
    with pytest.raises(Round0236Error, match="registered 25000000"):
        validate_composition(short)
    # A shifted share with the right total is the failure that would otherwise
    # pass unnoticed: 40/25/25/10 is the owner-confirmed mix, not just the sum.
    shifted = dict(counts)
    shifted[COMPOSITION[0][0]] -= 1
    shifted[COMPOSITION[1][0]] += 1
    with pytest.raises(Round0236Error, match="assembled"):
        validate_composition(shifted)


def test_shard_span_raises_below_the_floor():
    assert validate_shard_span(
        corpus="c", shards_touched=98, shards_total=98, label="increment"
    )["coverage"] == 1.0
    with pytest.raises(Round0236Error, match="below the registered"):
        validate_shard_span(
            corpus="c", shards_touched=90, shards_total=98, label="increment"
        )


def test_nesting_requires_containment_order_and_distinctness():
    parent = _records([(0, 0, 1), (0, 0, 2), (1, 3, 7)])
    child = _records([(0, 0, 1), (0, 0, 2), (1, 3, 7), (2, 5, 9)])
    report = assert_nesting(parent=parent, child=child)
    assert report["parent_rows_missing_from_child"] == 0
    assert report["positional_prefix"] and report["child_rows_distinct"]

    with pytest.raises(Round0236Error, match="not nested"):
        assert_nesting(parent=parent, child=_records([(0, 0, 1), (2, 5, 9)]))
    permuted = _records([(0, 0, 2), (0, 0, 1), (1, 3, 7), (2, 5, 9)])
    with pytest.raises(Round0236Error, match="R0235's order"):
        assert_nesting(parent=parent, child=permuted)
    duplicated = _records([(0, 0, 1), (0, 0, 1), (1, 3, 7)])
    with pytest.raises(Round0236Error, match="duplicated source row"):
        assert_nesting(parent=parent, child=duplicated)


def test_reserve_disjointness_raises_on_one_shared_row():
    training = _records([(0, 0, 1), (1, 2, 5)])
    reserve = _records([(0, 0, 9), (1, 2, 9)])
    assert assert_reserve_disjoint(
        training=training, reserve=reserve
    )["global_intersection_rows"] == 0
    with pytest.raises(Round0236Error, match="overlaps"):
        assert_reserve_disjoint(
            training=training, reserve=_records([(0, 0, 1), (1, 2, 9)])
        )


def test_provenance_keys_refuse_an_out_of_range_row():
    with pytest.raises(Round0236Error, match="registered key"):
        provenance_keys(_records([(0, 0, 1 << 41)]))


def test_reserve_geometry_is_inherited_unchanged():
    assert RESERVE_ROWS == RESERVE_ROWS_PER_CORPUS * len(COMPOSITION) == 200_000


# --------------------------------------------------------------------------- #
# the registered truth probe
# --------------------------------------------------------------------------- #
def test_truth_probe_is_uniform_distinct_ascending_and_seed_reproducible():
    probe = truth_probe_query_rows()
    assert probe.shape == (TRUTH_PROBE_ROWS,)
    assert probe.dtype == np.int64
    assert int(np.unique(probe).size) == TRUTH_PROBE_ROWS
    assert np.array_equal(probe, np.sort(probe))
    assert 0 <= int(probe.min()) and int(probe.max()) < ROWS
    assert np.array_equal(probe, truth_probe_query_rows())
    # A uniform draw spreads over the whole range rather than prefixing it: the
    # defect R0216 shipped was exactly a leading prefix.
    for quarter in range(4):
        low, high = quarter * ROWS // 4, (quarter + 1) * ROWS // 4
        share = float(((probe >= low) & (probe < high)).mean())
        assert abs(share - 0.25) < 0.005
    assert float(probe.mean()) == pytest.approx(ROWS / 2, rel=0.01)


def test_truth_probe_refuses_an_undrawable_size():
    with pytest.raises(Round0236Error, match="not drawable"):
        truth_probe_query_rows(rows=10, size=11)


def test_truth_probe_seed_and_size_are_registered_before_the_substrate():
    assert TRUTH_PROBE_SEED == 236_000
    assert TRUTH_PROBE_ROWS == 1_000_000
    assert TRUTH_PROBE_ROWS / ROWS == 0.04


# --------------------------------------------------------------------------- #
# the replicate grid — review-0235-01 F3 / B3
# --------------------------------------------------------------------------- #
def test_replicate_grid_is_three_seeds_at_three_nested_n():
    assert len(IMBALANCE_REPLICATE_SEEDS) >= 3
    assert IMBALANCE_REPLICATE_SEEDS[0] == PRIMARY_IMBALANCE_SEED == 226
    assert IMBALANCE_PROBE_ROWS == (6_250_000, 12_500_000, ROWS)
    assert set(IMBALANCE_PROBE_CLUSTERS) >= {128, 200}


def test_replicate_summary_reports_spread_and_the_primary():
    summary = replicate_summary({226: 1.60, 236: 1.70, 1236: 1.65})
    assert summary["n"] == 3
    assert summary["primary"] == 1.60
    assert summary["mean"] == pytest.approx(1.65)
    assert summary["spread_absolute"] == pytest.approx(0.10)
    assert summary["spread_relative"] == pytest.approx(0.10 / 1.65)
    with pytest.raises(Round0236Error, match="no realisations"):
        replicate_summary({})


def test_replicate_drift_separates_the_n_channel_from_the_draw_channel():
    # Movement of 20% across N against a 1% within-N spread: attributable.
    attributable = replicate_drift({
        6_250_000: {64: {226: 1.00, 236: 1.005, 1236: 1.002}},
        25_000_000: {64: {226: 1.20, 236: 1.205, 1236: 1.202}},
    })["by_clusters"]["64"]
    assert attributable["drift_primary"]["drift_relative"] == pytest.approx(0.20)
    assert attributable["worst_within_n_spread_relative"] < 0.01
    assert attributable["drift_exceeds_spread"] is True

    # Movement of 1% against a 20% within-N spread: NOT attributable.
    noise = replicate_drift({
        6_250_000: {64: {226: 1.00, 236: 1.20, 1236: 1.10}},
        25_000_000: {64: {226: 1.01, 236: 1.21, 1236: 1.11}},
    })["by_clusters"]["64"]
    assert noise["drift_primary"]["drift_relative"] == pytest.approx(0.01)
    assert noise["drift_exceeds_spread"] is False


def test_replicate_drift_reports_an_inherited_single_realisation_as_such():
    table = replicate_drift(
        {25_000_000: {200: {226: 1.9, 236: 2.0, 1236: 1.95}}},
        inherited={2_000_000: {200: 2.1311125, 32: 1.5}},
    )
    at_2m = table["by_clusters"]["200"]["by_rows"]["2000000"]
    assert at_2m["replicated"] is False and at_2m["n"] == 1
    assert at_2m["spread_relative"] is None
    # A `c` that exists only at 2M is carried as an absence at the other N.
    assert table["by_clusters"]["32"]["by_rows"]["25000000"] is None


# --------------------------------------------------------------------------- #
# feasibility, its margin, and its tolerance — review-0235-01 F1 / B1
# --------------------------------------------------------------------------- #
def test_admissible_reproduces_the_reviewed_figure():
    assert admissible_max_cluster_rows(_law()) == pytest.approx(9_936_160, rel=1e-5)
    assert admissible_max_cluster_rows(
        _law(), residual_margin=0.0
    ) == pytest.approx(10_670_950, rel=1e-5)


@pytest.mark.parametrize(
    "rung, clusters, imbalance, guarded, tolerance",
    [
        (25_000_000, 64, 1.64868480, 6_001_645, 0.656),
        (50_000_000, 128, 1.71442944, 6_240_973, 0.592),
        (100_000_000, 200, 1.98499000, 9_249_132, 0.0743),
    ],
)
def test_tolerance_reproduces_review_0235s_published_table(
    rung, clusters, imbalance, guarded, tolerance
):
    report = imbalance_tolerance(
        rung=rung, clusters=clusters, imbalance=imbalance, laws=_law()
    )
    assert report["guarded_max_cluster_rows"] == pytest.approx(guarded, rel=1e-4)
    assert report["admissible"] is True
    assert report["tolerance_to_adverse_imbalance"] == pytest.approx(
        tolerance, rel=2e-3
    )
    assert report["imbalance_margin_applied"] == GUARD_IMBALANCE_MARGIN


def test_the_100m_rung_is_the_one_with_no_room():
    tolerances = {
        rung: imbalance_tolerance(
            rung=rung, clusters=clusters, imbalance=imbalance, laws=_law()
        )["tolerance_to_adverse_imbalance"]
        for rung, clusters, imbalance in (
            (25_000_000, 64, 1.64868480),
            (50_000_000, 128, 1.71442944),
            (100_000_000, 200, 1.98499000),
        )
    }
    # The largest single-doubling move the program has measured is 13.42%.
    assert tolerances[100_000_000] < 0.1342
    assert min(tolerances[25_000_000], tolerances[50_000_000]) > 0.5


def test_carry_distance_counts_doublings():
    assert carry_distance(
        measured_at_rows=ROWS, rung=ROWS
    )["measured_at_the_rung"] is True
    assert carry_distance(
        measured_at_rows=ROWS, rung=100_000_000
    )["doublings_carried"] == pytest.approx(2.0)
    with pytest.raises(Round0236Error, match="positive"):
        carry_distance(measured_at_rows=0, rung=ROWS)


def test_the_margin_reaches_the_guard_and_the_selection_identically():
    laws = _law()
    guarded = guarded_max_cluster_rows(
        rows=ROWS, clusters=64, imbalance=1.64868480
    )
    guard = guard_decision(
        rows=ROWS, clusters=64, imbalance=1.64868480,
        imbalance_source="test", laws=laws,
    )
    assert guard["prediction"]["guarded_max_cluster_rows"] == pytest.approx(guarded)
    assert guard["prediction"]["imbalance_margin"] == GUARD_IMBALANCE_MARGIN
    selection = select_clusters(
        rows=ROWS, measured_imbalance={64: 1.64868480, 200: 1.98499},
        laws=laws, candidates=SELECTION_CANDIDATES, c_min=C_BUILD_MIN,
    )
    assert selection["selected_clusters"] == 64
    assert selection["selection"]["guarded_max_cluster_rows"] == pytest.approx(
        guarded
    )


def test_the_build_candidate_set_excludes_c32_and_says_why():
    assert SELECTION_CANDIDATES == (64, 200)
    assert C_BUILD_MIN == 64 > C_MIN == 2 * SPILL
    # c = 32 at 25M under R0235's measurement is over the admissible bound, which
    # is why it is priced but not built.
    over = imbalance_tolerance(
        rung=ROWS, clusters=32, imbalance=1.38355776, laws=_law()
    )
    assert over["admissible"] is False


def test_no_admissible_c_raises_rather_than_silently_picking_one():
    with pytest.raises(Exception, match="admissible c"):
        select_clusters(
            rows=1_000_000_000, measured_imbalance={64: 2.0}, laws=_law(),
            candidates=(64,), c_min=C_BUILD_MIN,
        )


def test_device_law_refuses_a_heterogeneous_point():
    refused = admit_law_point({
        "cell": "q6", "max_cluster_rows": 1000, "device_bytes": 1e9,
        "graph_degree": 64, "intermediate_graph_degree": 128, "max_iterations": 20,
    })
    assert refused["admitted"] is False
    assert "is not the registered" in refused["refusal_reasons"][0]
    assert (GRAPH_DEGREE, INTERMEDIATE_GRAPH_DEGREE, MAX_ITERATIONS) == (64, 256, 40)


def test_law_residual_margin_and_budget_are_the_registered_ones():
    assert LAW_RESIDUAL_MARGIN == 0.05
    assert GUARD_DEVICE_BUDGET_BYTES == 24 * 1024 ** 3
    assert GUARD_IMBALANCE_MARGIN == pytest.approx(1.1648840)


# --------------------------------------------------------------------------- #
# the I/O model — review-0235-01 F6 / B4
# --------------------------------------------------------------------------- #
def test_architectural_substrate_reads_are_quadratic_in_n():
    passes = {
        rows: predicted_substrate_passes(
            rows=rows, clusters=64, imbalance=1.65
        )
        for rows in (6_250_000, 12_500_000, 25_000_000, 100_000_000)
    }
    # Passes grow linearly in N, so reads grow quadratically.
    assert passes[25_000_000] > passes[12_500_000] > passes[6_250_000]
    fit = io_scaling_fit([
        {"rows": rows,
         "substrate_read_bytes": architectural_io(
             rows=rows, substrate_passes=count
         )["substrate_read_bytes"]}
        for rows, count in passes.items()
    ])
    assert fit["structural_exponent"] == 2.0
    assert 1.8 < fit["exponent"] < 2.1
    with pytest.raises(Round0236Error, match="at least two points"):
        io_scaling_fit([{"rows": 1, "substrate_read_bytes": 1}])


def test_architectural_io_matches_the_sealed_12500k_cell():
    # R0235's selected cell sealed 8 passes, 153,600,000,000 substrate read bytes
    # and 153,600,000,000 spill write bytes at 12,500,000 rows.
    io = architectural_io(rows=12_500_000, substrate_passes=8)
    assert io["substrate_read_bytes"] == 153_600_000_000
    assert io["spill_write_bytes"] == 153_600_000_000
    assert io["substrate_bytes"] == 19_200_000_000


def test_the_two_regime_prediction_flips_between_25m_and_100m():
    at_25m = physical_io_prediction(rows=ROWS, substrate_passes=13)
    at_100m = physical_io_prediction(rows=100_000_000, substrate_passes=52)
    assert at_25m["substrate_fits_page_cache"] is True
    assert at_25m["regime"] == "page-cache-resident"
    assert at_25m["predicted_physical_substrate_read_bytes"] == at_25m[
        "substrate_bytes"
    ]
    assert at_100m["substrate_fits_page_cache"] is False
    assert at_100m["regime"] == "page-cache-thrashing"
    assert at_100m["predicted_physical_substrate_read_bytes"] == at_100m[
        "substrate_read_bytes"
    ]
    assert at_100m["substrate_bytes"] > PAGE_CACHE_BUDGET_BYTES
    # The reviewer's arithmetic: ~7.4 TB of substrate reads at 100M.
    assert 6e12 < at_100m["substrate_read_bytes"] < 9e12
    assert at_100m["spill_write_bytes"] == pytest.approx(1.2288e12)


def test_io_hours_reproduces_the_reviewers_bracket_at_100m():
    at_100m = physical_io_prediction(rows=100_000_000, substrate_passes=48)
    slow = io_hours(
        read_bytes=at_100m["total_read_bytes"],
        write_bytes=at_100m["total_write_bytes"],
        read_bytes_per_s=1.24e9, write_bytes_per_s=2.52e9,
    )
    fast = io_hours(
        read_bytes=at_100m["total_read_bytes"],
        write_bytes=at_100m["total_write_bytes"],
        read_bytes_per_s=4.86e9, write_bytes_per_s=2.52e9,
    )
    assert 1.8 < slow["io_hours"] < 2.4
    assert 0.5 < fast["io_hours"] < 0.8
    assert slow["io_hours"] > fast["io_hours"]
    with pytest.raises(Round0236Error, match="positive"):
        io_hours(read_bytes=1, write_bytes=1, read_bytes_per_s=0,
                 write_bytes_per_s=1)


# --------------------------------------------------------------------------- #
# safety
# --------------------------------------------------------------------------- #
def test_memmap_precondition_refuses_an_anonymous_array(tmp_path):
    with pytest.raises(Exception, match="memmap"):
        assert_memmap_for_cuvs(np.zeros((4, 4), dtype=np.float32), label="anon")
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((4, 4), dtype=np.float32))
    assert_memmap_for_cuvs(np.load(path, mmap_mode="r"), label="memmap")


def test_signal_policy_refuses_a_delivered_signal():
    assert_no_signal_policy(["cooperative-flag"])
    with pytest.raises(Exception, match="SIG"):
        assert_no_signal_policy(["SIGTERM"])


def test_every_phase2_rung_is_priced():
    assert PHASE2_RUNGS == (
        6_250_000, 12_500_000, 25_000_000, 50_000_000, 100_000_000
    )
    assert ROWS in PHASE2_RUNGS
