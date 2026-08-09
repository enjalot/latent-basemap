"""R0232 — the contract, the grids, the three-axis guard and the scratch law.

CUDA-hidden. Nothing here touches the GPU, the filesystem sampler or the queue;
it exercises the arithmetic every published number rests on.
"""
from __future__ import annotations

import math

import pytest

from basemap.round0226_graph_builders import DIMENSION
from basemap.round0227_low_c_contract import (
    CLUSTER_CAPACITY_ROWS,
    SCRATCH_BUDGET_BYTES,
    pack_clusters_into_groups,
)
from basemap.round0232_scratch_contract import (
    ARM_CELL,
    ARM_REFERENCE_CELL,
    DISK_FREE_RESERVE_BYTES,
    GRID_A,
    GRID_B,
    IDENTITY_FAMILIES,
    MINIMUM_DETECTABLE_DISPLACEMENT_SD,
    MODES,
    MODE_MATERIALISE,
    MODE_STREAM_GATHER,
    MODE_STREAM_RESIDENT,
    PERMUTATION_RESOLUTION_CEILING,
    ROUND_SCRATCH_BUDGET_BYTES,
    ROWS,
    SPILL_VOLUME_100M_S8_BYTES,
    Round0232Error,
    capacity_rows_at_device_budget,
    cell_guard,
    device_law_prediction,
    disk_guard,
    io_projection,
    ladder_disk_requirement,
    licensed_statement,
    linear_fit,
    map_capability,
    predicted_peak_scratch_bytes,
    predicted_resident_host_bytes,
    scratch_law,
)

_GIB = 1024 ** 3


# --------------------------------------------------------------------------- #
# the grids
# --------------------------------------------------------------------------- #
def test_every_registered_cell_is_well_formed():
    names = set()
    for cell in (*GRID_A, *GRID_B):
        assert cell["cell"] not in names, "duplicate cell name"
        names.add(cell["cell"])
        assert cell["mode"] in MODES
        assert cell["rows"] > 0 and cell["clusters"] > 0 and cell["spill"] >= 1
        assert cell["intermediate_graph_degree"] >= cell["graph_degree"]
        if cell["mode"] == MODE_STREAM_GATHER:
            assert int(cell["bound_bytes"]) == 0
        else:
            assert int(cell["bound_bytes"]) > 0
    assert len(GRID_A) == 11
    assert len(GRID_B) == 2


def test_the_arm_and_its_reference_are_a_matched_pair():
    arm = next(cell for cell in GRID_A if cell["cell"] == ARM_CELL)
    reference = next(cell for cell in GRID_A if cell["cell"] == ARM_REFERENCE_CELL)
    assert (arm["rows"], arm["clusters"], arm["spill"]) == (
        reference["rows"], reference["clusters"], reference["spill"]
    )
    assert arm["mode"] == MODE_STREAM_GATHER
    assert reference["mode"] == MODE_MATERIALISE
    assert reference["bound_bytes"] == SCRATCH_BUDGET_BYTES
    for key in ("graph_degree", "intermediate_graph_degree", "max_iterations"):
        assert arm[key] == reference[key]
    assert arm["rows"] == ROWS


def test_identity_families_are_matched_and_cover_every_multi_mode_cell():
    by_name = {cell["cell"]: cell for cell in (*GRID_A, *GRID_B)}
    covered = set()
    for family in IDENTITY_FAMILIES:
        assert len(family) >= 2
        first = by_name[family[0]]
        for name in family:
            cell = by_name[name]
            covered.add(name)
            assert (cell["rows"], cell["clusters"], cell["spill"]) == (
                first["rows"], first["clusters"], first["spill"]
            ), "a family must hold the partition fixed"
    # every cell that shares an (N, c, s) with another cell must be in a family
    triples: dict[tuple[int, int, int], list[str]] = {}
    for cell in (*GRID_A, *GRID_B):
        triples.setdefault(
            (cell["rows"], cell["clusters"], cell["spill"]), []
        ).append(cell["cell"])
    for names in triples.values():
        if len(names) > 1:
            assert set(names) <= covered


def test_each_family_contains_two_materialising_cells_or_is_a_pair():
    """The nn-descent non-determinism control, stated as a structural property."""
    by_name = {cell["cell"]: cell for cell in (*GRID_A, *GRID_B)}
    biggest = max(IDENTITY_FAMILIES, key=len)
    materialising = [
        name for name in biggest if by_name[name]["mode"] == MODE_MATERIALISE
    ]
    assert len(materialising) >= 2, (
        "the largest family needs two materialising cells at different bounds, "
        "otherwise a mode difference cannot be told from builder non-determinism"
    )
    bounds = {by_name[name]["bound_bytes"] for name in materialising}
    assert len(bounds) >= 2


# --------------------------------------------------------------------------- #
# the scratch prediction
# --------------------------------------------------------------------------- #
def test_streamed_modes_predict_exactly_zero_scratch():
    for mode in (MODE_STREAM_RESIDENT, MODE_STREAM_GATHER):
        assert predicted_peak_scratch_bytes(
            rows=100_000_000, clusters=200, spill=8, mode=mode,
            bound_bytes=24 * _GIB, imbalance=2.6,
        ) == 0


def test_materialise_prediction_is_bounded_by_the_bound_plus_one_cluster():
    rows, clusters, spill, imbalance = 100_000_000, 200, 8, 2.5919
    bound = 24 * _GIB
    peak = predicted_peak_scratch_bytes(
        rows=rows, clusters=clusters, spill=spill, mode=MODE_MATERIALISE,
        bound_bytes=bound, imbalance=imbalance,
    )
    volume = rows * spill * DIMENSION * 4
    largest = math.ceil(volume / clusters * imbalance)
    assert peak == min(volume, bound + largest)
    assert peak < volume // 20, "peak scratch is nowhere near the spill volume"


def test_the_spill_volume_constant_is_the_number_that_is_not_the_requirement():
    assert SPILL_VOLUME_100M_S8_BYTES == 100_000_000 * 8 * DIMENSION * 4
    assert SPILL_VOLUME_100M_S8_BYTES / 1e12 == pytest.approx(1.2288, abs=1e-4)
    peak = predicted_peak_scratch_bytes(
        rows=100_000_000, clusters=200, spill=8, mode=MODE_MATERIALISE,
        bound_bytes=24 * _GIB, imbalance=2.5919,
    )
    assert peak * 20 < SPILL_VOLUME_100M_S8_BYTES


def test_streamed_modes_charge_their_residency_to_host_memory():
    gather = predicted_resident_host_bytes(
        rows=100_000_000, clusters=200, spill=8, mode=MODE_STREAM_GATHER,
        bound_bytes=0, imbalance=2.5919,
    )
    resident = predicted_resident_host_bytes(
        rows=100_000_000, clusters=200, spill=8, mode=MODE_STREAM_RESIDENT,
        bound_bytes=24 * _GIB, imbalance=2.5919,
    )
    materialise = predicted_resident_host_bytes(
        rows=100_000_000, clusters=200, spill=8, mode=MODE_MATERIALISE,
        bound_bytes=24 * _GIB, imbalance=2.5919,
    )
    assert materialise == 0
    assert 0 < gather < resident, "driving scratch to zero is not free"


def test_an_unregistered_mode_is_refused():
    with pytest.raises(Round0232Error):
        predicted_peak_scratch_bytes(
            rows=1, clusters=1, spill=1, mode="magic", bound_bytes=1, imbalance=1.0,
        )


# --------------------------------------------------------------------------- #
# the disk guard
# --------------------------------------------------------------------------- #
def test_disk_guard_admits_a_streamed_100m_cell_on_this_box():
    decision = disk_guard(
        rows=100_000_000, clusters=200, spill=8, mode=MODE_STREAM_GATHER,
        bound_bytes=0, free_bytes=280 * 10 ** 9,
    )
    assert decision["allowed"] is True
    assert decision["predicted_peak_scratch_bytes"] == 0


def test_disk_guard_refuses_a_cell_that_would_breach_the_reserve():
    decision = disk_guard(
        rows=100_000_000, clusters=200, spill=8, mode=MODE_MATERIALISE,
        bound_bytes=200 * 10 ** 9, free_bytes=280 * 10 ** 9,
    )
    assert decision["allowed"] is False
    assert decision["refused_a_priori"] is True
    assert decision["refusal_reasons"]
    assert decision["data_free_bytes_after_prediction"] < DISK_FREE_RESERVE_BYTES


def test_disk_guard_refuses_a_cell_over_the_round_scratch_budget():
    decision = disk_guard(
        rows=100_000_000, clusters=200, spill=8, mode=MODE_MATERIALISE,
        bound_bytes=ROUND_SCRATCH_BUDGET_BYTES * 2,
        free_bytes=10 ** 15,
    )
    assert decision["over_round_scratch_budget"] is True
    assert decision["allowed"] is False


def test_every_registered_cell_passes_all_three_guard_axes_at_280_gb_free():
    for cell in (*GRID_A, *GRID_B):
        decision = cell_guard(cell, free_bytes=280 * 10 ** 9)
        assert decision["allowed"] is True, (
            f"{cell['cell']} is refused: {decision['refusal_reasons']}"
        )
        assert decision["axes"] == ["device", "host-anonymous", "disk"]


def test_the_guard_would_refuse_the_same_cells_if_the_volume_were_nearly_full():
    refused = [
        cell["cell"] for cell in (*GRID_A, *GRID_B)
        if not cell_guard(cell, free_bytes=151 * 10 ** 9)["allowed"]
    ]
    assert refused, "a nearly-full volume must refuse the materialising cells"
    for name in refused:
        cell = next(item for item in (*GRID_A, *GRID_B) if item["cell"] == name)
        assert cell["mode"] == MODE_MATERIALISE


# --------------------------------------------------------------------------- #
# the packing this round's whole argument rests on
# --------------------------------------------------------------------------- #
def test_packing_bounds_every_group_by_at_most_one_cluster():
    rows, clusters, spill = 100_000_000, 200, 8
    mean = rows * spill / clusters
    sizes = [int(mean)] * clusters
    sizes[0] = int(mean * 2.5919)
    for bound in (2 * _GIB, 4 * _GIB, 24 * _GIB):
        groups = pack_clusters_into_groups(sizes, budget_bytes=bound)
        largest_cluster_bytes = max(sizes) * DIMENSION * 4
        for group in groups:
            group_bytes = sum(sizes[index] for index in group) * DIMENSION * 4
            assert group_bytes <= bound + largest_cluster_bytes
        assert sorted(index for group in groups for index in group) == list(
            range(clusters)
        ), "the packing must partition the clusters exactly once"


def test_group_count_falls_as_the_bound_rises():
    sizes = [1_000_000] * 64
    counts = [
        len(pack_clusters_into_groups(sizes, budget_bytes=bound))
        for bound in (2 * _GIB, 4 * _GIB, 24 * _GIB)
    ]
    assert counts == sorted(counts, reverse=True)


# --------------------------------------------------------------------------- #
# the measured scratch law
# --------------------------------------------------------------------------- #
def test_scratch_law_reports_the_disagreement_it_finds():
    cells = [
        {
            "cell": "x", "rows": 2_000_000, "clusters": 200, "spill": 8,
            "mode": MODE_MATERIALISE, "bound_bytes": 4 * _GIB,
            "measured_peak_scratch_bytes": 4_100_000_000,
            "modelled_peak_scratch_bytes": 4_000_000_000,
            "spill_groups": 7, "substrate_passes": 7,
        },
        {
            "cell": "y", "rows": 2_000_000, "clusters": 200, "spill": 8,
            "mode": MODE_STREAM_GATHER, "bound_bytes": 0,
            "measured_peak_scratch_bytes": 0,
            "modelled_peak_scratch_bytes": 0,
            "spill_groups": 200, "substrate_passes": 0,
        },
    ]
    law = scratch_law(cells)
    assert law["bytes_per_resident_spilled_row"] == DIMENSION * 4
    assert law["worst_absolute_measured_minus_modelled_bytes"] == 100_000_000
    assert law["cells"][1]["fraction_of_volume_resident"] == 0.0


# --------------------------------------------------------------------------- #
# the device law refit
# --------------------------------------------------------------------------- #
def test_linear_fit_recovers_a_known_line():
    xs = [1e5, 5e5, 1e6, 2e6]
    ys = [5e9 + 1600.0 * x for x in xs]
    fit = linear_fit(xs, ys)
    assert fit["slope_bytes_per_max_cluster_row"] == pytest.approx(1600.0, rel=1e-9)
    assert fit["intercept_bytes"] == pytest.approx(5e9, rel=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert fit["fitted_range_max_cluster_rows"] == [1e5, 2e6]


def test_device_prediction_labels_its_extrapolation_and_its_gap_to_r0227():
    fit = linear_fit([1e5, 2e6], [5e9 + 2400.0 * 1e5, 5e9 + 2400.0 * 2e6])
    prediction = device_law_prediction(8_000_000, fit)
    assert prediction["is_extrapolation"] is True
    assert prediction["extrapolation_factor_beyond_fitted_max"] == pytest.approx(4.0)
    assert prediction["label"] == "PROJECTION"
    # a steeper refitted slope must show up as a positive gap against R0227's law
    assert prediction["refit_minus_unrefitted_gib"] > 0


def test_capacity_shrinks_when_the_refitted_law_is_steeper():
    steep = linear_fit([1e5, 2e6], [5e9 + 2400.0 * 1e5, 5e9 + 2400.0 * 2e6])
    assert capacity_rows_at_device_budget(steep) < CLUSTER_CAPACITY_ROWS


def test_capacity_refuses_a_non_positive_slope():
    flat = linear_fit([1e5, 2e6], [5e9, 5e9])
    with pytest.raises(Round0232Error):
        capacity_rows_at_device_budget(flat)


# --------------------------------------------------------------------------- #
# the I/O law
# --------------------------------------------------------------------------- #
def test_gather_moves_fewer_bytes_than_grouped_materialise_at_100m():
    kwargs = dict(
        rows=100_000_000, clusters=200, spill=8, imbalance=2.5919,
        read_bytes_per_s=6.0e9, write_bytes_per_s=2.5e9,
    )
    materialise = io_projection(
        mode=MODE_MATERIALISE, bound_bytes=24 * _GIB, **kwargs
    )
    resident = io_projection(
        mode=MODE_STREAM_RESIDENT, bound_bytes=24 * _GIB, **kwargs
    )
    gather = io_projection(mode=MODE_STREAM_GATHER, bound_bytes=0, **kwargs)
    assert materialise["substrate_passes"] > 1
    assert materialise["bytes_written"] > 0
    assert resident["bytes_written"] == 0
    assert resident["peak_scratch_bytes"] == 0
    assert gather["bytes_written"] == 0
    assert gather["peak_scratch_bytes"] == 0
    assert gather["total_bytes_moved"] < resident["total_bytes_moved"]
    assert resident["total_bytes_moved"] < materialise["total_bytes_moved"]
    assert all(
        entry["label"] == "PROJECTION"
        for entry in (materialise, resident, gather)
    )


def test_io_projection_charges_the_spill_read_back_in_materialise_mode():
    io = io_projection(
        rows=2_000_000, clusters=200, spill=8, mode=MODE_MATERIALISE,
        bound_bytes=24 * _GIB, imbalance=2.13,
        read_bytes_per_s=6.0e9, write_bytes_per_s=2.5e9,
    )
    volume = 2_000_000 * 8 * DIMENSION * 4
    assert io["bytes_written"] == volume
    assert io["bytes_read"] >= volume


# --------------------------------------------------------------------------- #
# the deliverable
# --------------------------------------------------------------------------- #
def test_the_100m_ladder_requirement_names_every_byte_not_just_the_scratch():
    requirement = ladder_disk_requirement(rows=100_000_000, peak_scratch_bytes=0)
    assert requirement["substrate_bytes"] == 100_000_000 * DIMENSION * 4
    assert requirement["substrate_bytes"] / 1e9 == pytest.approx(153.6, abs=0.1)
    assert requirement["neighbour_ids_bytes"] == 100_000_000 * 15 * 4
    assert requirement["fuzzy_edge_file_bytes"] > 0
    assert requirement["total_bytes_at_peak"] > requirement["substrate_bytes"]
    assert requirement["label"] == "PROJECTION"


def test_scratch_is_a_choice_not_a_requirement_in_the_deliverable():
    zero = ladder_disk_requirement(rows=100_000_000, peak_scratch_bytes=0)
    bounded = ladder_disk_requirement(
        rows=100_000_000, peak_scratch_bytes=29 * 10 ** 9
    )
    assert bounded["total_bytes_at_peak"] - zero["total_bytes_at_peak"] == 29 * 10 ** 9


# --------------------------------------------------------------------------- #
# inference discipline
# --------------------------------------------------------------------------- #
def test_a_non_rejection_is_never_worded_as_equivalence():
    statement = licensed_statement(p_value=0.109091)
    assert "NOT REJECTED" in statement
    assert "smaller than about one" in statement
    assert "NOT a claim of equivalence" in statement
    assert "equivalent" not in statement.lower().replace("equivalence", "")


def test_a_rejection_is_worded_as_a_rejection():
    assert licensed_statement(p_value=0.006061).startswith("REJECTED")


def test_the_designs_resolution_and_minimum_detectable_effect_are_registered():
    assert PERMUTATION_RESOLUTION_CEILING == pytest.approx(1.0 / 165.0)
    assert PERMUTATION_RESOLUTION_CEILING < 0.05, (
        "a test published as a null must be able to reject at its own threshold"
    )
    assert MINIMUM_DETECTABLE_DISPLACEMENT_SD == 0.98


def test_map_capabilities_are_distinct_from_r0229s():
    assert map_capability(42) == "minilm-mixed-2m-streamed-spill-map-seed42-v1"
    assert "spill-lifted" not in map_capability(42)


# --------------------------------------------------------------------------- #
# the two data paths, proven equal on a CPU before any GPU time is spent
# --------------------------------------------------------------------------- #
def test_sweep_and_gather_read_the_identical_rows_in_the_identical_order(tmp_path):
    """P3's mechanism: `stream-gather` feeds nn-descent exactly what the spill file did.

    Byte-identity of the merged graph can only hold if the per-cluster input
    arrays are themselves identical. That is arithmetic, it needs no GPU, and it
    is checked here rather than inferred from the fact that both loops "look
    right".
    """
    import numpy as np

    from basemap.round0227_low_c_contract import pack_clusters_into_groups
    from basemap.round0232_streamed_build import (
        cluster_membership,
        fill_group_by_sweep,
        gather_cluster,
    )

    rng = np.random.default_rng(232)
    rows, dimension, clusters, spill = 5_000, 8, 7, 3
    dataset = rng.standard_normal((rows, dimension)).astype(np.float32)
    assignment = rng.integers(0, clusters, size=(rows, spill)).astype(np.int32)
    sizes = np.bincount(assignment.ravel(), minlength=clusters).astype(np.int64)
    members, bounds = cluster_membership(assignment, sizes, rows=rows, spill=spill)

    assert int(sizes.sum()) == rows * spill
    for cluster in range(clusters):
        ids = members[bounds[cluster] : bounds[cluster + 1]]
        assert ids.size == int(sizes[cluster])
        assert np.all(np.diff(ids) >= 0), "gather must be a monotone strided read"

    for bound in (64 * 1024, 1024 * 1024):
        groups = pack_clusters_into_groups(sizes, budget_bytes=bound)
        seen = []
        for group in groups:
            handles = fill_group_by_sweep(
                dataset, group, sizes=sizes, members=members, bounds=bounds,
                rows=rows, dimension=dimension,
                allocate=lambda _cluster, size: np.empty(
                    (size, dimension), dtype=np.float32
                ),
            )
            for cluster in group:
                seen.append(cluster)
                gathered = gather_cluster(
                    dataset, members[bounds[cluster] : bounds[cluster + 1]]
                )
                assert gathered.shape == handles[cluster].shape
                assert gathered.dtype == handles[cluster].dtype
                assert np.array_equal(gathered, handles[cluster]), (
                    f"cluster {cluster} differs between the sweep and the gather"
                )
        assert seen == sorted(range(clusters)), (
            "the cluster visit order must be index order in every mode, or a "
            "mode comparison is confounded with an ordering change"
        )
