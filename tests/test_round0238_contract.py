"""R0238 contract tests — CPU only, no CUDA, no artifacts.

Every registered check in `round-0238-2026-08-09.md` that can be exercised
without a GPU is exercised here, plus the three defects that cost earlier rounds
a queue: the imbalance-grid call-site arity (R0236), a seal that could not
survive its own JSON round trip (R0236), and a claim about `min = 0.0` taken on
trust rather than measured (R0236).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from basemap.round0238_rung5 import (
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
    Round0238Error,
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
import basemap.round0238_rung5 as rung5
import experiments.prepare_round0238_queue as prepare


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

#: R0237's sealed worst-of-FIVE imbalance at 50,000,000 rows, `s = 8`. This is
#: the figure the whole 100M prediction was priced from, so it is READ FROM THE
#: SEALED ARTIFACT rather than transcribed; the literals beside it are only a
#: tripwire that the artifact is the one this round registered against.
_R0237_LADDER_PATH = prepare.R0237_LADDER
if os.path.exists(_R0237_LADDER_PATH):
    with open(_R0237_LADDER_PATH, encoding="utf-8") as _handle:
        R0237_WORST_AT_50M = {
            int(key): float(value) for key, value in
            json.load(_handle)["worst_seed_imbalance_at_this_rung"].items()
        }
else:  # pragma: no cover - only when the sealed artifact is absent
    R0237_WORST_AT_50M = {
        16: 1.175083, 32: 1.491957, 64: 1.679772,
        128: 1.790741, 200: 2.061112, 400: 2.456543,
    }

#: The ten-point law, which is the nine above plus R0237's own 50M cell. This is
#: the law every tolerance in review-0237-01's table actually uses (its F3
#: corrected the round for pairing the ten-point law with the nine-point bound).
SEALED_LAW_POINT_R0237 = (5_421_826, 16_259_219_456)


def _law(points=None):
    return fit_device_law(
        [
            {
                "source": f"sealed-{rows}", "rows": 25_000_000, "clusters": 64,
                "spill": SPILL, "graph_degree": 64,
                "intermediate_graph_degree": 256, "max_iterations": 40,
                "max_cluster_rows": rows, "device_bytes": device,
            }
            for rows, device in (
                points if points is not None
                else [*SEALED_LAW_POINTS, SEALED_LAW_POINT_R0237]
            )
        ],
        label="sealed-gd64-igd256-it40",
    )


def test_the_binding_law_is_the_ten_point_one_review_0237_reproduced():
    """review-0237-01 F3: the ten-point law, not the nine-point bound."""
    ten = _law()
    assert ten["n_points"] == 10
    assert ten["slope_bytes_per_max_cluster_row"] == pytest.approx(
        1598.3515811373481, rel=1e-9
    )
    assert ten["intercept_bytes"] == pytest.approx(
        7_619_114_558.335568, rel=1e-9
    )
    assert ten["r_squared"] == pytest.approx(0.9972485950132305, rel=1e-9)
    assert admissible_max_cluster_rows([ten]) == pytest.approx(
        9_948_339.67145981, rel=1e-9
    )
    nine = _law(SEALED_LAW_POINTS)
    assert admissible_max_cluster_rows([nine]) == pytest.approx(
        9_939_097.54214061, rel=1e-9
    )


def _records(entries):
    array = np.zeros(len(entries), dtype=PROVENANCE_DTYPE)
    for index, (corpus, shard, row) in enumerate(entries):
        array[index] = (corpus, shard, row)
    return array


# --------------------------------------------------------------------------- #
# composition, span, nesting, reserve
# --------------------------------------------------------------------------- #
def test_composition_is_exactly_twice_rung_4_at_the_confirmed_shares():
    counts = dict(COMPOSITION)
    assert sum(counts.values()) == ROWS == 100_000_000
    assert counts == {
        "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2": 40_000_000,
        "RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2": 25_000_000,
        "pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2": 25_000_000,
        "starcoderdata-code-chunked-120-all-MiniLM-L6-v2": 10_000_000,
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
    with pytest.raises(Round0238Error):
        validate_composition(counts)
    short = dict(COMPOSITION)
    short["starcoderdata-code-chunked-120-all-MiniLM-L6-v2"] -= 1
    with pytest.raises(Round0238Error):
        validate_composition(short)


def test_shard_span_raises_below_the_floor():
    ok = validate_shard_span(
        corpus="x", shards_touched=100, shards_total=100, label="union"
    )
    assert ok["coverage"] == 1.0
    with pytest.raises(Round0238Error, match="SPAN"):
        validate_shard_span(
            corpus="x", shards_touched=90, shards_total=100, label="increment"
        )


def test_the_five_ladder_prefixes_are_the_registered_rungs():
    assert LADDER_PREFIX_ROWS == (
        6_250_000, 12_500_000, 25_000_000, 50_000_000, 100_000_000
    )
    assert rung5.GREAT2_GRANDPARENT_ROWS == 6_250_000
    assert GREAT_GRANDPARENT_ROWS == 12_500_000
    assert GRANDPARENT_ROWS == 25_000_000
    assert PARENT_ROWS == 50_000_000
    assert set(rung5.INHERITED_PREFIX_SHA256) == {
        6_250_000, 12_500_000, 25_000_000
    }
    assert all(
        len(value) == 64 for value in rung5.INHERITED_PREFIX_SHA256.values()
    )


def test_nesting_requires_containment_order_and_distinctness():
    parent = _records([(0, 0, 1), (0, 0, 2), (1, 3, 7)])
    child = _records([(0, 0, 1), (0, 0, 2), (1, 3, 7), (2, 9, 11)])
    record = assert_nesting(parent=parent, child=child)
    assert record["parent_rows_missing_from_child"] == 0
    assert record["positional_prefix"] is True

    with pytest.raises(Round0238Error, match="not nested"):
        assert_nesting(parent=parent, child=_records([(0, 0, 1), (2, 9, 11)]))
    permuted = _records([(0, 0, 2), (0, 0, 1), (1, 3, 7), (2, 9, 11)])
    with pytest.raises(Round0238Error, match="positional"):
        assert_nesting(parent=parent, child=permuted)
    duplicated = _records([(0, 0, 1), (0, 0, 2), (1, 3, 7), (1, 3, 7)])
    with pytest.raises(Round0238Error, match="duplicated"):
        assert_nesting(parent=parent, child=duplicated)


def test_reserve_disjointness_raises_on_one_shared_row():
    training = _records([(0, 0, 1), (1, 2, 3)])
    reserve = _records([(0, 0, 9), (1, 2, 8)])
    assert assert_reserve_disjoint(
        training=training, reserve=reserve
    )["global_intersection_rows"] == 0
    with pytest.raises(Round0238Error, match="overlaps"):
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
    assert TRUTH_PROBE_ROWS == 500_000
    assert TRUTH_PROBE_SEED == 238_000
    # Registered in the release BEFORE the substrate exists, reproducible from
    # the seed alone, uniform over all 100,000,000 ids, never a seed union.
    drawn = truth_probe_query_rows(
        rows=ROWS, size=TRUTH_PROBE_ROWS, seed=TRUTH_PROBE_SEED
    )
    again = truth_probe_query_rows(
        rows=ROWS, size=TRUTH_PROBE_ROWS, seed=TRUTH_PROBE_SEED
    )
    assert np.array_equal(drawn, again)
    assert drawn.shape == (TRUTH_PROBE_ROWS,)
    assert np.unique(drawn).size == TRUTH_PROBE_ROWS
    assert np.array_equal(drawn, np.sort(drawn))
    assert int(drawn.min()) >= 0 and int(drawn.max()) < ROWS
    with pytest.raises(Round0238Error):
        truth_probe_query_rows(rows=100, size=101, seed=1)


# --------------------------------------------------------------------------- #
# the five-seed replicate grid
# --------------------------------------------------------------------------- #
def test_the_grid_is_five_seeds_at_this_rung_only():
    assert IMBALANCE_REPLICATE_SEEDS == (226, 236, 1236, 2236, 3236)
    assert len(IMBALANCE_REPLICATE_SEEDS) == 5
    assert PRIMARY_IMBALANCE_SEED == 226 == IMBALANCE_REPLICATE_SEEDS[0]
    assert IMBALANCE_PROBE_ROWS == (100_000_000,)
    assert IMBALANCE_PROBE_CLUSTERS == (16, 32, 64, 128, 200, 400)
    # ALL FIVE are R0237's, so every column compares like-for-like across the
    # 50M -> 100M doubling, which is what review-0237-01 F5 wanted published.
    assert set(IMBALANCE_REPLICATE_SEEDS) == {226, 236, 1236, 2236, 3236}
    # the c this rung builds must be measured at this rung
    assert SELECTION_CANDIDATES[0] in IMBALANCE_PROBE_CLUSTERS


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
            50_000_000: "measured in R0238, 5 seeds",
        },
        inherited={2_000_000: {200: 2.131112}},
    )
    row = table["by_clusters"]["200"]
    assert row["by_rows"]["2000000"]["replicated"] is False
    assert row["by_rows"]["2000000"]["n"] == 1
    assert "R0236" in row["by_rows"]["25000000"]["source"]
    assert "R0238" in row["by_rows"]["50000000"]["source"]
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
    imbalance = {400: R0237_WORST_AT_50M[400]}
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


def test_the_build_candidate_set_is_c400_alone():
    assert SELECTION_CANDIDATES == (400,)
    assert C_BUILD_MIN == 400
    laws = [_law()]
    admissible = admissible_max_cluster_rows(laws)
    # c = 400 fits with the tolerance R0237 and both reviews priced.
    guarded = guarded_max_cluster_rows(
        rows=ROWS, clusters=400, imbalance=R0237_WORST_AT_50M[400]
    )
    assert guarded < admissible
    assert guarded == pytest.approx(5_723_175.272, rel=1e-6)
    tolerance = imbalance_tolerance(
        rung=ROWS, clusters=400, imbalance=R0237_WORST_AT_50M[400], laws=laws
    )
    assert tolerance["tolerance_to_adverse_imbalance"] == pytest.approx(
        rung5.PREDICTION_TOLERANCE_AT_C400, rel=1e-4
    )
    # c = 200 is admissible but EXTRAPOLATES the law, which is why the
    # registered rule refuses it in advance.
    guarded_200 = guarded_max_cluster_rows(
        rows=ROWS, clusters=200, imbalance=R0237_WORST_AT_50M[200]
    )
    assert guarded_200 < admissible
    fitted_max = max(
        float(point["max_cluster_rows"]) for point in _law()["points"]
    )
    assert guarded_200 / fitted_max > 1.0
    assert guarded / fitted_max < 1.0


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


def test_the_rule_selects_c400_and_this_round_is_the_100m_rung():
    verdict = hundred_m_verdict(
        imbalance_by_c=R0237_WORST_AT_50M, laws=[_law()],
        reachability_by_c=rung5.R0237_25M_S8_CEILING_REFERENCE,
    )
    assert HUNDRED_M_ROWS == 100_000_000 == ROWS
    assert HUNDRED_M_CANDIDATES == (200, 400)
    assert verdict["recommended_clusters"] == 400
    assert verdict["recommended_clusters"] == SELECTION_CANDIDATES[0]
    assert "NOT a recommendation for a future round" in verdict["scope"]
    assert "-16.13%" in verdict["higher_spill_note"]
    by_c = {
        int(entry["clusters"]): entry
        for entry in verdict["candidates_considered"]
    }
    assert by_c[200]["admissible"] is True
    assert by_c[200]["inside_fitted_law_range"] is False
    assert by_c[400]["inside_fitted_law_range"] is True
    assert by_c[400]["clears_reachability_floor"] is True
    # A candidate whose ceiling is below the floor is refused even when it fits.
    refused = hundred_m_verdict(
        imbalance_by_c=R0237_WORST_AT_50M, laws=[_law()],
        reachability_by_c={200: 0.999, 400: 0.5},
    )
    assert refused["recommended_clusters"] != 400


def test_carry_distance_counts_doublings():
    # This round measures imbalance AT its own rung, so nothing is carried.
    carry = carry_distance(measured_at_rows=ROWS, rung=100_000_000)
    assert carry["doublings_carried"] == pytest.approx(0.0)
    assert carry["measured_at_the_rung"]
    assert carry_distance(
        measured_at_rows=PARENT_ROWS, rung=ROWS
    )["doublings_carried"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# reachability
# --------------------------------------------------------------------------- #
def test_reachability_is_registered_at_this_rung_and_the_built_c():
    assert REACHABILITY_ROWS == ROWS == 100_000_000
    assert REACHABILITY_CLUSTERS == (400,)
    assert REACHABILITY_SEED == 226
    assert REACHABILITY_CONCERN_FLOOR == 0.99
    # the partition this round BUILDS is the one it scans, so the ceiling and
    # the realised recall are measured on the same probe at the same N
    assert set(REACHABILITY_CLUSTERS) == set(SELECTION_CANDIDATES)
    # R0237's sealed 25M ceilings are the registered trend reference
    assert rung5.R0237_25M_S8_CEILING_REFERENCE[400] == 0.9977319
    assert rung5.R0237_25M_S8_CEILING_REFERENCE[400] > REACHABILITY_CONCERN_FLOOR


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

    with pytest.raises(Round0238Error):
        reachability_cell_summary(np.array([1.5]), clusters=64)
    with pytest.raises(Round0238Error):
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
    with pytest.raises(Round0238Error, match="shape"):
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


def test_this_rung_is_on_the_far_side_of_the_flip_point():
    fifty = physical_io_prediction(rows=PARENT_ROWS, substrate_passes=27)
    hundred = physical_io_prediction(rows=ROWS, substrate_passes=51)
    assert fifty["substrate_bytes"] == 76_800_000_000
    assert hundred["substrate_bytes"] == 153_600_000_000
    assert PAGE_CACHE_BUDGET_BYTES == 80 * 1000 ** 3
    # R0237 sat at 96% of the registered budget and stayed (coarsely) resident.
    assert fifty["substrate_fits_page_cache"] is True
    # This rung is 1.92x the budget: the FIRST rung whose substrate cannot be
    # served from page cache, so its I/O term is a measurement, not a forecast.
    assert hundred["substrate_fits_page_cache"] is False
    assert hundred["substrate_bytes"] / PAGE_CACHE_BUDGET_BYTES == pytest.approx(
        1.92
    )


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
    assert GPU_HOURS_CAP == 6.0
    assert BUILD_TIMEOUT_S == 16_000.0
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
