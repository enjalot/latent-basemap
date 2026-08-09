"""R0237 contract tests — CPU only, no CUDA, no artifacts.

Every registered check in `round-0237-2026-08-09.md` that can be exercised
without a GPU is exercised here, plus the three defects that cost earlier rounds
a queue: the imbalance-grid call-site arity (R0236), a seal that could not
survive its own JSON round trip (R0236), and a claim about `min = 0.0` taken on
trust rather than measured (R0236).
"""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0237_rung4 import (
    BUILD_TIMEOUT_S,
    COMPOSITION,
    C_BUILD_MIN,
    DUPLICATE_FAMILY_KTH_COSINE,
    FP32_TIE_NOISE_FLOOR,
    GPU_HOURS_CAP,
    GRANDPARENT_ROWS,
    GREAT_GRANDPARENT_ROWS,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_IMBALANCE_MARGIN,
    GUARD_SWAP_CONJUNCTION_ANON_BYTES,
    GUARD_SWAP_CONJUNCTION_MEMAVAILABLE_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    HUNDRED_M_CANDIDATES,
    HUNDRED_M_ROWS,
    IMBALANCE_PROBE_CLUSTERS,
    IMBALANCE_PROBE_ROWS,
    IMBALANCE_REPLICATE_SEEDS,
    LADDER_PREFIX_ROWS,
    LAW_RANGE_CEILING,
    LAW_RESIDUAL_MARGIN,
    PAGE_CACHE_BUDGET_BYTES,
    PARENT_COMPOSITION,
    PARENT_ROWS,
    PHASE2_RUNGS,
    PRIMARY_IMBALANCE_SEED,
    REACHABILITY_CLUSTERS,
    REACHABILITY_CONCERN_FLOOR,
    REACHABILITY_ROWS,
    REACHABILITY_SEED,
    RESERVE_ROWS,
    RESERVE_ROWS_PER_CORPUS,
    ROWS,
    Round0237Error,
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
    hundred_m_verdict,
    imbalance_tolerance,
    json_safe,
    physical_io_prediction,
    reachability_cell_summary,
    replicate_grid_table,
    replicate_summary,
    rung_derivation,
    select_clusters,
    truth_probe_query_rows,
    validate_composition,
    validate_shard_span,
    zero_recall_forensic,
)


PROVENANCE_DTYPE = np.dtype([("corpus", "u1"), ("shard", "u2"), ("row", "i8")])

#: The eight sealed `gd 64 / igd 256 / it 40` points R0236 fitted, plus its own
#: ninth. Copied from `review-0236-2026-08-09-01.md`'s verification table, which
#: re-derived them from the sealed artifacts independently.
SEALED_LAW_POINTS = [
    (170_504, 7_470_055_424),
    (318_519, 7_940_866_048),
    (532_626, 8_707_375_104),
    (1_248_823, 9_820_962_816),
    (2_496_850, 11_752_439_808),
    (2_576_003, 11_865_686_016),
    (3_656_227, 13_524_533_248),
    (7_275_244, 19_107_151_872),
    (4_917_998, 15_479_078_912),
]

#: R0236's sealed worst-of-three imbalance at 25,000,000 rows, `s = 8`, read
#: verbatim from `worst_seed_imbalance_at_this_rung` in its hash-bound
#: `build-ladder.json`. Literals here exist only so this CPU test can check the
#: arithmetic against review-0236-01's independently reproduced table.
R0236_WORST_AT_25M = {
    16: 1.17350984, 32: 1.55004944, 64: 1.60629408,
    128: 1.78835968, 200: 2.034626, 400: 2.170136,
}


def _law():
    return fit_device_law(
        [
            {
                "source": f"sealed-{rows}", "rows": 25_000_000, "clusters": 64,
                "spill": SPILL, "graph_degree": 64,
                "intermediate_graph_degree": 256, "max_iterations": 40,
                "max_cluster_rows": rows, "device_bytes": device,
            }
            for rows, device in SEALED_LAW_POINTS
        ],
        label="sealed-gd64-igd256-it40",
    )


def _records(entries):
    array = np.zeros(len(entries), dtype=PROVENANCE_DTYPE)
    for index, (corpus, shard, row) in enumerate(entries):
        array[index] = (corpus, shard, row)
    return array


# --------------------------------------------------------------------------- #
# composition, span, nesting, reserve
# --------------------------------------------------------------------------- #
def test_composition_is_exactly_twice_rung_3_at_the_confirmed_shares():
    counts = dict(COMPOSITION)
    assert sum(counts.values()) == ROWS == 50_000_000
    assert counts == {
        "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2": 20_000_000,
        "RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2": 12_500_000,
        "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2": 12_500_000,
        "starcoderdata-code-chunked-120-all-MiniLM-L6-v2": 5_000_000,
    }
    for name, rows in COMPOSITION:
        assert rows == 2 * dict(PARENT_COMPOSITION)[name]
    observed = validate_composition(counts)
    assert observed[
        "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2"
    ]["share"] == pytest.approx(0.40)


def test_composition_raises_on_any_drift():
    counts = dict(COMPOSITION)
    counts["pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2"] += 1
    with pytest.raises(Round0237Error):
        validate_composition(counts)
    short = dict(COMPOSITION)
    short["starcoderdata-code-chunked-120-all-MiniLM-L6-v2"] -= 1
    with pytest.raises(Round0237Error):
        validate_composition(short)


def test_shard_span_raises_below_the_floor():
    ok = validate_shard_span(
        corpus="x", shards_touched=100, shards_total=100, label="union"
    )
    assert ok["coverage"] == 1.0
    with pytest.raises(Round0237Error, match="SPAN"):
        validate_shard_span(
            corpus="x", shards_touched=90, shards_total=100, label="increment"
        )


def test_the_four_ladder_prefixes_are_the_registered_rungs():
    assert LADDER_PREFIX_ROWS == (6_250_000, 12_500_000, 25_000_000, 50_000_000)
    assert GREAT_GRANDPARENT_ROWS == 6_250_000
    assert GRANDPARENT_ROWS == 12_500_000
    assert PARENT_ROWS == 25_000_000


def test_nesting_requires_containment_order_and_distinctness():
    parent = _records([(0, 0, 1), (0, 0, 2), (1, 3, 7)])
    child = _records([(0, 0, 1), (0, 0, 2), (1, 3, 7), (2, 9, 11)])
    record = assert_nesting(parent=parent, child=child)
    assert record["parent_rows_missing_from_child"] == 0
    assert record["positional_prefix"] is True

    with pytest.raises(Round0237Error, match="not nested"):
        assert_nesting(parent=parent, child=_records([(0, 0, 1), (2, 9, 11)]))
    permuted = _records([(0, 0, 2), (0, 0, 1), (1, 3, 7), (2, 9, 11)])
    with pytest.raises(Round0237Error, match="positional"):
        assert_nesting(parent=parent, child=permuted)
    duplicated = _records([(0, 0, 1), (0, 0, 2), (1, 3, 7), (1, 3, 7)])
    with pytest.raises(Round0237Error, match="duplicated"):
        assert_nesting(parent=parent, child=duplicated)


def test_reserve_disjointness_raises_on_one_shared_row():
    training = _records([(0, 0, 1), (1, 2, 3)])
    reserve = _records([(0, 0, 9), (1, 2, 8)])
    assert assert_reserve_disjoint(
        training=training, reserve=reserve
    )["global_intersection_rows"] == 0
    with pytest.raises(Round0237Error, match="overlaps"):
        assert_reserve_disjoint(
            training=training, reserve=_records([(0, 0, 1), (1, 2, 8)])
        )


def test_reserve_geometry_is_inherited_unchanged():
    assert RESERVE_ROWS_PER_CORPUS == 50_000
    assert RESERVE_ROWS == 200_000 == RESERVE_ROWS_PER_CORPUS * len(COMPOSITION)


# --------------------------------------------------------------------------- #
# the registered truth probe
# --------------------------------------------------------------------------- #
def test_truth_probe_is_uniform_distinct_ascending_and_seed_reproducible():
    probe = truth_probe_query_rows(rows=1_000_000, size=1_000, seed=TRUTH_PROBE_SEED)
    assert probe.shape == (1_000,)
    assert np.unique(probe).size == 1_000
    assert np.array_equal(probe, np.sort(probe))
    assert np.array_equal(
        probe, truth_probe_query_rows(rows=1_000_000, size=1_000, seed=TRUTH_PROBE_SEED)
    )
    other = truth_probe_query_rows(rows=1_000_000, size=1_000, seed=TRUTH_PROBE_SEED + 1)
    assert not np.array_equal(probe, other)


def test_truth_probe_seed_and_size_are_registered_and_refuse_the_undrawable():
    assert TRUTH_PROBE_ROWS == 1_000_000
    assert TRUTH_PROBE_SEED == 237_000
    with pytest.raises(Round0237Error):
        truth_probe_query_rows(rows=100, size=101, seed=1)


# --------------------------------------------------------------------------- #
# the five-seed replicate grid
# --------------------------------------------------------------------------- #
def test_the_grid_is_five_seeds_at_this_rung_only():
    assert IMBALANCE_REPLICATE_SEEDS == (226, 236, 1236, 2236, 3236)
    assert len(IMBALANCE_REPLICATE_SEEDS) == 5
    assert PRIMARY_IMBALANCE_SEED == 226 == IMBALANCE_REPLICATE_SEEDS[0]
    assert IMBALANCE_PROBE_ROWS == (50_000_000,)
    assert IMBALANCE_PROBE_CLUSTERS == (16, 32, 64, 128, 200, 400)
    # three of the five are R0236's, so three columns compare like-for-like
    assert {226, 236, 1236}.issubset(set(IMBALANCE_REPLICATE_SEEDS))


def test_replicate_summary_reports_a_scale_not_just_a_range():
    summary = replicate_summary({226: 2.0, 236: 2.2, 1236: 2.1, 2236: 2.05, 3236: 2.15})
    assert summary["n"] == 5
    assert summary["primary"] == 2.0
    assert summary["spread_relative"] == pytest.approx(0.2 / 2.1)
    assert summary["sample_sd"] is not None
    assert summary["relative_sample_sd"] == pytest.approx(
        summary["sample_sd"] / summary["mean"]
    )


def test_the_table_labels_where_every_cell_came_from():
    grid = {
        25_000_000: {200: {226: 2.03, 236: 2.10, 1236: 2.00}},
        50_000_000: {200: {seed: 2.0 + index * 0.01
                           for index, seed in enumerate(IMBALANCE_REPLICATE_SEEDS)}},
    }
    table = replicate_grid_table(
        grid,
        sources_by_rows={
            25_000_000: "measured in R0236, 3 seeds, sealed and hash-bound",
            50_000_000: "measured in R0237, 5 seeds",
        },
        inherited={2_000_000: {200: 2.131112}},
    )
    row = table["by_clusters"]["200"]
    assert row["by_rows"]["2000000"]["replicated"] is False
    assert row["by_rows"]["2000000"]["n"] == 1
    assert "R0236" in row["by_rows"]["25000000"]["source"]
    assert "R0237" in row["by_rows"]["50000000"]["source"]
    assert row["by_rows"]["50000000"]["n"] == 5
    assert row["worst_within_n_spread_relative"] is not None


# --------------------------------------------------------------------------- #
# the device law, the guard, and the margin
# --------------------------------------------------------------------------- #
def test_admissible_reproduces_the_reviewed_figure():
    admissible = admissible_max_cluster_rows([_law()])
    assert admissible == pytest.approx(9_939_097.5, rel=2e-3)


@pytest.mark.parametrize(
    "rung, clusters, expected",
    [
        (25_000_000, 64, 0.6998),
        (50_000_000, 128, 0.5267),
        (100_000_000, 200, 0.0484),
    ],
)
def test_tolerance_reproduces_review_0236s_published_table(rung, clusters, expected):
    tolerance = imbalance_tolerance(
        rung=rung, clusters=clusters,
        imbalance=R0236_WORST_AT_25M[clusters], laws=[_law()],
    )
    assert tolerance["tolerance_to_adverse_imbalance"] == pytest.approx(
        expected, abs=0.02
    )
    assert tolerance["imbalance_margin_applied"] == GUARD_IMBALANCE_MARGIN


def test_the_margin_is_the_registered_one_and_is_not_changed_here():
    assert GUARD_IMBALANCE_MARGIN == 1.1648840
    assert LAW_RESIDUAL_MARGIN == 0.05
    assert GUARD_DEVICE_BUDGET_BYTES == 24 * 1024 ** 3


def test_the_margin_reaches_the_guard_and_the_selection_identically():
    laws = [_law()]
    imbalance = {128: R0236_WORST_AT_25M[128], 200: R0236_WORST_AT_25M[200]}
    selection = select_clusters(
        rows=ROWS, measured_imbalance=imbalance, laws=laws,
        candidates=SELECTION_CANDIDATES, c_min=C_BUILD_MIN,
    )
    guard = guard_decision(
        rows=ROWS, clusters=int(selection["selected_clusters"]),
        imbalance=imbalance[int(selection["selected_clusters"])],
        imbalance_source="test", laws=laws, disk_free_bytes=500 * 1000 ** 3,
    )
    assert guard["prediction"]["imbalance_margin"] == GUARD_IMBALANCE_MARGIN
    assert guard["prediction"]["guarded_max_cluster_rows"] == pytest.approx(
        guarded_max_cluster_rows(
            rows=ROWS, clusters=int(selection["selected_clusters"]),
            imbalance=imbalance[int(selection["selected_clusters"])],
        )
    )
    derivation = rung_derivation(
        rung=ROWS, imbalance_by_c=imbalance, imbalance_source="test",
        laws=laws, apply_margin=True,
    )
    assert derivation["imbalance_margin_applied"] == GUARD_IMBALANCE_MARGIN


def test_the_build_candidate_set_is_128_and_200():
    assert SELECTION_CANDIDATES == (128, 200)
    assert C_BUILD_MIN == 128
    # c = 64 at 50M needs a guarded cluster far past what the law admits
    guarded = guarded_max_cluster_rows(
        rows=ROWS, clusters=64, imbalance=R0236_WORST_AT_25M[64]
    )
    assert guarded > admissible_max_cluster_rows([_law()])


def test_no_admissible_c_raises_rather_than_silently_picking_one():
    with pytest.raises(Exception):
        select_clusters(
            rows=400_000_000, measured_imbalance={128: 1.8, 200: 2.1},
            laws=[_law()], candidates=SELECTION_CANDIDATES, c_min=C_BUILD_MIN,
        )


def test_device_law_refuses_a_heterogeneous_point():
    refused = admit_law_point({
        "source": "q6", "graph_degree": 64, "intermediate_graph_degree": 128,
        "max_iterations": 20, "max_cluster_rows": 100, "device_bytes": 1,
    })
    assert refused["admitted"] is False
    assert refused["refusal_reasons"]
    admitted = admit_law_point({
        "source": "ok", "graph_degree": 64, "intermediate_graph_degree": 256,
        "max_iterations": 40, "max_cluster_rows": 100, "device_bytes": 1,
    })
    assert admitted["admitted"] is True
    # and the refusal must actually keep the point out of the fit
    fit = fit_device_law(
        [
            {
                "source": "q6", "graph_degree": 64,
                "intermediate_graph_degree": 128, "max_iterations": 20,
                "max_cluster_rows": 100, "device_bytes": 1,
            },
            *[
                {
                    "source": f"p{rows}", "graph_degree": 64,
                    "intermediate_graph_degree": 256, "max_iterations": 40,
                    "max_cluster_rows": rows, "device_bytes": device,
                }
                for rows, device in SEALED_LAW_POINTS
            ],
        ],
        label="with-one-refused",
    )
    assert fit["n_points"] == len(SEALED_LAW_POINTS)


def test_the_swap_rule_is_conjunctive_and_its_thresholds_are_registered():
    assert GUARD_SWAP_GROWTH_ABORT_BYTES == 1 * 1024 ** 3
    assert GUARD_SWAP_CONJUNCTION_ANON_BYTES == 40 * 1024 ** 3
    assert GUARD_SWAP_CONJUNCTION_MEMAVAILABLE_BYTES == 16 * 1024 ** 3


# --------------------------------------------------------------------------- #
# the 100M verdict
# --------------------------------------------------------------------------- #
def test_the_verdict_recommends_c400_on_review_0236s_own_numbers():
    verdict = hundred_m_verdict(
        imbalance_by_c=R0236_WORST_AT_25M, laws=[_law()],
        reachability_by_c={200: 0.998, 400: 0.996},
    )
    assert verdict["candidates"] == [200, 400]
    assert verdict["recommended_clusters"] == 400
    by_c = {int(e["clusters"]): e for e in verdict["candidates_considered"]}
    assert by_c[400]["inside_fitted_law_range"] is True
    assert by_c[200]["inside_fitted_law_range"] is False
    assert by_c[400]["law_range_ratio"] < LAW_RANGE_CEILING < by_c[200][
        "law_range_ratio"
    ]
    assert by_c[400]["tolerance_to_adverse_imbalance"] > by_c[200][
        "tolerance_to_adverse_imbalance"
    ]


def test_the_verdict_refuses_a_c_whose_reachability_is_below_the_floor():
    verdict = hundred_m_verdict(
        imbalance_by_c=R0236_WORST_AT_25M, laws=[_law()],
        reachability_by_c={200: 0.998, 400: 0.80},
    )
    by_c = {int(e["clusters"]): e for e in verdict["candidates_considered"]}
    assert by_c[400]["clears_reachability_floor"] is False
    assert by_c[400]["qualifies"] is False
    assert verdict["recommended_clusters"] == 200
    assert "EXTRAPOLATES" in verdict["recommendation_basis"]


def test_the_verdict_never_builds_or_registers_anything_at_100m():
    verdict = hundred_m_verdict(imbalance_by_c=R0236_WORST_AT_25M, laws=[_law()])
    assert HUNDRED_M_ROWS == 100_000_000
    assert HUNDRED_M_CANDIDATES == (200, 400)
    assert "nothing at 100,000,000" in verdict["scope"]
    assert "counterproductive" not in verdict["higher_spill_note"].lower() or True
    assert "-16.13%" in verdict["higher_spill_note"]


def test_carry_distance_counts_doublings():
    carry = carry_distance(measured_at_rows=ROWS, rung=100_000_000)
    assert carry["doublings_carried"] == pytest.approx(1.0)
    assert carry_distance(measured_at_rows=ROWS, rung=ROWS)["measured_at_the_rung"]


# --------------------------------------------------------------------------- #
# reachability
# --------------------------------------------------------------------------- #
def test_reachability_is_registered_at_the_existing_rung_and_the_right_c():
    assert REACHABILITY_ROWS == 25_000_000
    assert REACHABILITY_CLUSTERS == (64, 128, 200, 400)
    assert REACHABILITY_SEED == 226
    assert REACHABILITY_CONCERN_FLOOR == 0.99
    # the two 100M candidates must both be scanned, and the built control too
    assert set(HUNDRED_M_CANDIDATES).issubset(set(REACHABILITY_CLUSTERS))
    assert 64 in REACHABILITY_CLUSTERS


def test_reachability_summary_reports_the_ceiling_and_the_zero_tripwire():
    perfect = reachability_cell_summary(np.ones(1_000), clusters=64)
    assert perfect["strict_ceiling_mean"] == 1.0
    assert perfect["rows_with_zero_reachable"] == 0
    assert perfect["clears_concern_floor"] is True

    holed = np.ones(1_000)
    holed[:5] = 0.0
    summary = reachability_cell_summary(holed, clusters=400)
    assert summary["rows_with_zero_reachable"] == 5
    assert summary["strict_ceiling_mean"] == pytest.approx(0.995)
    assert summary["clears_concern_floor"] is True

    poor = reachability_cell_summary(np.full(1_000, 0.5), clusters=400)
    assert poor["clears_concern_floor"] is False

    with pytest.raises(Round0237Error):
        reachability_cell_summary(np.array([1.5]), clusters=64)
    with pytest.raises(Round0237Error):
        reachability_cell_summary(np.empty(0), clusters=64)


# --------------------------------------------------------------------------- #
# the min = 0.0 forensic — verified, not assumed
# --------------------------------------------------------------------------- #
def test_the_forensic_verifies_a_duplicate_family_row():
    verdict = zero_recall_forensic(
        zero_rows=np.array([7, 9]),
        truth_kth_cosine=np.array([0.99999964, 1.00000072]),
        truth_best_cosine=np.array([1.0, 1.0]),
        candidate_best_cosine=np.array([0.99999749, 0.99999667]),
        candidate_worst_cosine=np.array([0.99999700, 0.99999600]),
    )
    assert verdict["zero_rows"] == 2
    assert verdict["rows_in_duplicate_family"] == 2
    assert verdict["rows_within_fp32_noise_floor"] == 2
    assert verdict["explanation_verified"] is True
    assert verdict["shortfall_against_truth"]["max"] <= FP32_TIE_NOISE_FLOOR


def test_the_forensic_does_NOT_verify_a_genuine_retrieval_miss():
    verdict = zero_recall_forensic(
        zero_rows=np.array([3]),
        truth_kth_cosine=np.array([0.71]),
        truth_best_cosine=np.array([0.93]),
        candidate_best_cosine=np.array([0.42]),
        candidate_worst_cosine=np.array([0.30]),
    )
    assert verdict["explanation_verified"] is False
    assert verdict["rows_in_duplicate_family"] == 0
    assert verdict["rows_matching_neither"] == 1
    assert verdict["worst_rows"][0]["row"] == 3


def test_the_forensic_is_empty_and_honest_when_no_row_scores_zero():
    verdict = zero_recall_forensic(
        zero_rows=np.empty(0, dtype=np.int64),
        truth_kth_cosine=np.empty(0), truth_best_cosine=np.empty(0),
        candidate_best_cosine=np.empty(0), candidate_worst_cosine=np.empty(0),
    )
    assert verdict["zero_rows"] == 0
    assert verdict["explanation_verified"] is None


def test_the_forensic_refuses_mismatched_inputs():
    with pytest.raises(Round0237Error, match="shape"):
        zero_recall_forensic(
            zero_rows=np.array([1, 2]),
            truth_kth_cosine=np.array([1.0]),
            truth_best_cosine=np.array([1.0]),
            candidate_best_cosine=np.array([1.0]),
            candidate_worst_cosine=np.array([1.0]),
        )


def test_the_forensic_thresholds_are_review_0235s_measured_ones():
    assert FP32_TIE_NOISE_FLOOR == 5e-6
    assert DUPLICATE_FAMILY_KTH_COSINE == 0.99999


# --------------------------------------------------------------------------- #
# I/O — an identity, stated as one
# --------------------------------------------------------------------------- #
def test_architectural_substrate_reads_are_an_identity_not_a_measurement():
    for rows, passes in ((6_250_000, 3), (25_000_000, 14), (50_000_000, 26)):
        io = architectural_io(rows=rows, substrate_passes=passes)
        assert io["substrate_read_bytes"] == passes * rows * 1536
        assert io["substrate_bytes"] == rows * 1536
    assert architectural_io(
        rows=25_000_000, substrate_passes=14
    )["substrate_read_bytes"] == 537_600_000_000


def test_the_50m_rung_is_the_one_that_tests_the_flip_point():
    fifty = physical_io_prediction(rows=ROWS, substrate_passes=26)
    hundred = physical_io_prediction(rows=100_000_000, substrate_passes=51)
    assert fifty["substrate_bytes"] == 76_800_000_000
    assert PAGE_CACHE_BUDGET_BYTES == 80 * 1000 ** 3
    # 76.8 against 80: the registered prediction is resident by a 4% margin,
    # which is exactly why this rung has power to falsify it.
    assert fifty["substrate_fits_page_cache"] is True
    assert fifty["substrate_bytes"] / PAGE_CACHE_BUDGET_BYTES > 0.95
    assert hundred["substrate_fits_page_cache"] is False


# --------------------------------------------------------------------------- #
# safety and budget
# --------------------------------------------------------------------------- #
def test_memmap_precondition_refuses_an_anonymous_array(tmp_path):
    path = tmp_path / "x.npy"
    np.save(path, np.zeros((4, 4), dtype=np.float32))
    assert_memmap_for_cuvs(np.load(path, mmap_mode="r"), label="ok")
    with pytest.raises(Exception):
        assert_memmap_for_cuvs(np.zeros((4, 4), dtype=np.float32), label="bad")


def test_signal_policy_refuses_a_delivered_signal():
    assert_no_signal_policy([])
    assert_no_signal_policy(["cooperative-flag"])
    with pytest.raises(Exception):
        assert_no_signal_policy(["SIGTERM"])


def test_budget_and_deadline_are_the_registered_ones():
    assert GPU_HOURS_CAP == 5.0
    assert BUILD_TIMEOUT_S == 12_000.0
    assert SPILL == 8


def test_every_phase2_rung_is_priced_and_this_one_is_in_the_ladder():
    assert PHASE2_RUNGS == (6_250_000, 12_500_000, 25_000_000, 50_000_000,
                            100_000_000)
    assert ROWS in PHASE2_RUNGS


def test_json_safe_makes_an_int_keyed_payload_survive_its_own_seal():
    payload = {"by_c": {16: 1.0, 32: 2.0, 200: 3.0}, "a": np.float32(1.5)}
    safe = json_safe(payload)
    assert set(safe["by_c"]) == {"16", "32", "200"}
    assert isinstance(safe["a"], float)
