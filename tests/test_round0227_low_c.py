"""R0227 — the checks that have to hold before nine GPU builds are launched.

Three of these exist because a prior round paid for them:

* the **thread-lifecycle meta-test**, because R0226 named a sampler's stop flag
  `_stop`, shadowed `threading.Thread._stop`, and broke `join()` *after* the
  build completed. Review-0226-01 pointed out that R0226's regression test
  covered the one class that did **not** break and guarded one name out of
  several, so this one enumerates every `Thread` subclass this round can reach
  and checks all of `dir(threading.Thread)`.
* the **ascending-ladder ordering** check, because the mandate's safety rule is
  that the resource axis is ascended and stopped at the first failure, and at
  low `c` the resource axis is `max_cluster_rows` rather than `N`.
* the **group-packing** check, because R0226's equal-split rule stops bounding
  peak scratch once clusters get large, and the group count is the term the 100M
  I/O projection is built from.
"""
from __future__ import annotations

import inspect
import threading

import numpy as np
import pytest

from basemap import round0226_cluster_spill_build
from basemap import round0227_cluster_spill_build
from basemap import round0227_reachability_probe
from basemap.round0220_cuvs_qualification import graph_validity
from basemap.round0227_concentration import (
    Round0227ConcentrationError,
    density_decile_recall,
    edge_precision,
    loss_concentration,
    neighbour_loss_autocorrelation,
)
from basemap.round0227_low_c_contract import (
    BUILD_CELLS,
    CLUSTER_CAPACITY_ROWS,
    C_MIN,
    DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW,
    DEVICE_LAW_INTERCEPT_BYTES,
    DIMENSION,
    GUARD_DEVICE_BUDGET_BYTES,
    IMBALANCE_AT_C8,
    PHASE2_RUNGS,
    SCRATCH_BUDGET_BYTES,
    SPILL,
    Round0227Error,
    cluster_settings,
    device_bytes_from_law,
    guard_decision,
    imbalance_model,
    law_agreement,
    linear_fit,
    pack_clusters_into_groups,
    power_fit,
    predicted_max_cluster_rows,
    project_100m,
    smallest_feasible_clusters,
    substrate_passes,
)
from experiments import round0227_nodes


# --------------------------------------------------------------------------- #
# the matrix and the ladder's ordering
# --------------------------------------------------------------------------- #
def test_every_registered_cell_is_above_the_cluster_floor():
    for cell in BUILD_CELLS:
        assert int(cell["clusters"]) >= C_MIN
        # c = s would put every row in every cluster and partition nothing.
        assert int(cell["clusters"]) > SPILL


def test_ladder_ascends_predicted_max_cluster_rows():
    settings = cluster_settings()
    predicted = [int(item["predicted_max_cluster_rows"]) for item in settings]
    assert predicted == sorted(predicted)
    assert len(settings) == len(BUILD_CELLS)


def test_n_ascends_within_each_fixed_cluster_count():
    by_clusters: dict[int, list[int]] = {}
    for item in cluster_settings():
        by_clusters.setdefault(int(item["clusters"]), []).append(int(item["rows"]))
    for rows in by_clusters.values():
        assert rows == sorted(rows)


def test_every_cell_has_a_registered_substrate_and_a_unique_id():
    settings = cluster_settings()
    ids = [item["id"] for item in settings]
    assert len(set(ids)) == len(ids)
    for item in settings:
        assert item["substrate"].endswith(".npy")


def test_at_least_one_cell_emits_a_graph_at_each_scored_population():
    emitting = {
        int(item["rows"]) for item in cluster_settings() if item["emit_graph"]
    }
    assert 2_000_000 in emitting
    assert 16_000_000 in emitting


# --------------------------------------------------------------------------- #
# the guard, in both branches
# --------------------------------------------------------------------------- #
def test_guard_allows_every_registered_cell():
    for item in cluster_settings():
        decision = guard_decision(rows=item["rows"], clusters=item["clusters"])
        assert decision["allowed"], (item["id"], decision["refusal_reasons"])
        assert decision["prediction"]["predicted_device_bytes"] <= GUARD_DEVICE_BUDGET_BYTES


def test_guard_refuses_a_cell_whose_largest_cluster_cannot_fit():
    # 100M rows in four clusters is a 50M-row cluster: far past the card.
    decision = guard_decision(rows=100_000_000, clusters=4)
    assert decision["refused_a_priori"] is True
    assert decision["allowed"] is False
    assert decision["refusal_reasons"]


def test_guard_refusal_carries_its_prediction_as_data():
    decision = guard_decision(rows=100_000_000, clusters=4)
    assert decision["prediction"]["predicted_max_cluster_rows"] > CLUSTER_CAPACITY_ROWS
    assert decision["prediction"]["predicted_device_gib"] > 24.0


def test_guard_is_monotone_in_cluster_count():
    """Fewer clusters can only cost more device memory, never less."""
    previous = None
    for clusters in (4, 8, 16, 32, 64):
        prediction = guard_decision(rows=16_000_000, clusters=clusters)["prediction"]
        current = prediction["predicted_device_bytes"]
        if previous is not None:
            assert current <= previous
        previous = current


def test_predicted_max_cluster_rows_rejects_degenerate_inputs():
    with pytest.raises(Round0227Error):
        predicted_max_cluster_rows(rows=0, clusters=8)
    with pytest.raises(Round0227Error):
        predicted_max_cluster_rows(rows=1_000, clusters=0)


# --------------------------------------------------------------------------- #
# the memory law
# --------------------------------------------------------------------------- #
def test_memory_law_reproduces_review_0226_points():
    """The law must reproduce review-0226-01's own published 100M figure.

    It does so only under the decimal-GB reading of the intercept: `4.65 GiB +
    1560.9 x 1.69e6` is `7.107 GiB`, while `4.6467e9 B + 1560.9 x 1.69e6` is
    `6.787 GiB`, which is the `6.79 GiB` the review publishes. This test is what
    caught the unit.
    """
    assert device_bytes_from_law(1_690_000) / 1024 ** 3 == pytest.approx(6.79, abs=0.01)


def test_memory_law_reproduces_r0226_measured_device_peaks():
    # R0226's three capacity-saturated rungs, measured on the budget instrument.
    for max_cluster, measured in (
        (1_162_911, 6_461_325_312),
        (1_237_403, 6_578_765_824),
        (1_395_191, 6_824_132_608),
    ):
        assert device_bytes_from_law(max_cluster) == pytest.approx(measured, rel=0.002)


def test_law_agreement_is_exact_on_synthetic_points_from_the_law():
    largest = [600_000, 1_200_000, 4_000_000, 9_000_000]
    measured = [device_bytes_from_law(value) for value in largest]
    agreement = law_agreement(measured_bytes=measured, max_cluster_rows=largest)
    assert agreement["worst_absolute_relative_error"] == pytest.approx(0.0, abs=1e-12)
    assert agreement["refit_on_this_round"]["slope"] == pytest.approx(
        DEVICE_LAW_BYTES_PER_MAX_CLUSTER_ROW, rel=1e-9
    )
    assert agreement["refit_on_this_round"]["intercept"] == pytest.approx(
        DEVICE_LAW_INTERCEPT_BYTES, rel=1e-9
    )


def test_law_agreement_reports_a_real_deviation_rather_than_hiding_it():
    largest = [600_000, 9_000_000]
    measured = [device_bytes_from_law(600_000), device_bytes_from_law(9_000_000) * 1.5]
    agreement = law_agreement(measured_bytes=measured, max_cluster_rows=largest)
    assert agreement["worst_absolute_relative_error"] > 0.4


def test_imbalance_model_reproduces_its_fitted_points():
    assert imbalance_model(8) == pytest.approx(IMBALANCE_AT_C8, abs=1e-12)
    assert imbalance_model(32) == pytest.approx(1.395, abs=0.001)
    assert imbalance_model(200) == pytest.approx(1.70, abs=0.01)


# --------------------------------------------------------------------------- #
# the cluster-count calculation
# --------------------------------------------------------------------------- #
def test_smallest_feasible_c_fits_the_budget_and_is_minimal():
    for rung in PHASE2_RUNGS:
        choice = smallest_feasible_clusters(rows=rung)
        assert choice["feasible"] is True
        assert choice["predicted_device_bytes"] <= GUARD_DEVICE_BUDGET_BYTES
        smaller = int(choice["clusters"]) - 1
        if smaller >= C_MIN:
            largest = rung * SPILL * imbalance_model(smaller) / smaller
            assert largest > choice["admissible_max_cluster_rows"]


def test_smallest_feasible_c_is_nondecreasing_in_rows():
    previous = 0
    for rung in PHASE2_RUNGS:
        clusters = int(smallest_feasible_clusters(rows=rung)["clusters"])
        assert clusters >= previous
        previous = clusters


def test_smallest_feasible_c_at_100m_is_far_below_the_r0226_law():
    """R0226's registered law puts 100M at c = 200. The point of this round."""
    choice = smallest_feasible_clusters(rows=100_000_000)
    assert choice["clusters"] < 100


def test_measured_imbalance_overrides_the_model_when_supplied():
    modelled = smallest_feasible_clusters(rows=50_000_000)
    measured = smallest_feasible_clusters(
        rows=50_000_000, imbalance={value: 1.0 for value in range(4, 200)}
    )
    assert measured["imbalance_used"] == 1.0
    assert int(measured["clusters"]) <= int(modelled["clusters"])


# --------------------------------------------------------------------------- #
# spill grouping and I/O
# --------------------------------------------------------------------------- #
def test_group_packing_respects_the_scratch_budget():
    row_bytes = DIMENSION * 4
    sizes = [3_000_000] * 20
    groups = pack_clusters_into_groups(sizes, budget_bytes=SCRATCH_BUDGET_BYTES)
    assert sum(len(group) for group in groups) == len(sizes)
    assert sorted(index for group in groups for index in group) == list(range(len(sizes)))
    for group in groups:
        assert sum(sizes[index] for index in group) * row_bytes <= SCRATCH_BUDGET_BYTES


def test_group_packing_gives_an_oversized_cluster_its_own_group():
    row_bytes = DIMENSION * 4
    huge = int(SCRATCH_BUDGET_BYTES / row_bytes) * 2
    groups = pack_clusters_into_groups([1_000, huge, 1_000])
    assert [len(group) for group in groups] == [1, 1, 1]


def test_group_packing_drops_empty_clusters_but_keeps_every_nonempty_one():
    groups = pack_clusters_into_groups([0, 10, 0, 20])
    assert sorted(index for group in groups for index in group) == [1, 3]


def test_substrate_passes_rise_as_clusters_fall_at_fixed_n():
    high = substrate_passes(100_000_000, 200)
    low = substrate_passes(100_000_000, 22)
    assert low >= high


# --------------------------------------------------------------------------- #
# the concentration statistics, against constructed graphs with known answers
# --------------------------------------------------------------------------- #
def test_density_deciles_recover_a_planted_monotone_gradient():
    density = np.linspace(0.1, 0.9, 10_000)
    recall = np.linspace(0.5, 1.0, 10_000)
    result = density_decile_recall(recall, density, deciles=10)
    assert result["monotone_nondecreasing"] is True
    assert result["sparsest_decile_mean"] < result["densest_decile_mean"]
    assert result["sparsest_to_densest_gap"] == pytest.approx(0.45, abs=0.01)


def test_density_deciles_are_flat_when_the_loss_is_uniform():
    rng = np.random.default_rng(0)
    density = rng.random(10_000)
    recall = np.full(10_000, 0.9)
    result = density_decile_recall(recall, density, deciles=10)
    assert result["sparsest_to_densest_gap"] == pytest.approx(0.0, abs=1e-12)


def test_autocorrelation_is_high_when_loss_is_spatially_clustered():
    rows = 4_000
    # Neighbours are the adjacent rows, and the loss is a slow ramp, so a row's
    # loss and its neighbours' losses move together by construction.
    truth = np.stack(
        [(np.arange(rows) + offset) % rows for offset in range(1, 16)], axis=1
    )
    recall = 1.0 - np.linspace(0.0, 0.5, rows)
    result = neighbour_loss_autocorrelation(recall, truth, seed=1)
    assert result["neighbour_loss_correlation"] > 0.95
    assert abs(result["shuffled_null_correlation"]) < 0.2


def test_autocorrelation_null_is_near_zero_when_loss_is_independent():
    rng = np.random.default_rng(7)
    rows = 4_000
    truth = rng.integers(0, rows, size=(rows, 15))
    recall = rng.random(rows)
    result = neighbour_loss_autocorrelation(recall, truth, seed=2)
    assert abs(result["neighbour_loss_correlation"]) < 0.1
    assert abs(result["shuffled_null_correlation"]) < 0.1


def test_loss_concentration_separates_uniform_from_concentrated():
    uniform = np.full(10_000, 0.9)
    concentrated = np.ones(10_000)
    concentrated[:100] = 0.0
    spread = loss_concentration(uniform)
    spike = loss_concentration(concentrated)
    assert spread["worst_1pct_share_of_loss"] == pytest.approx(0.01, abs=0.001)
    assert spread["rows_carrying_any_loss"] == pytest.approx(1.0)
    assert spike["worst_1pct_share_of_loss"] == pytest.approx(1.0, abs=1e-9)
    assert spike["rows_carrying_any_loss"] == pytest.approx(0.01, abs=1e-9)


def test_edge_precision_of_an_exact_graph_equals_truth():
    rng = np.random.default_rng(11)
    rows, width = 500, 15
    truth_ids = rng.integers(0, 10_000, size=(rows, width))
    truth_cos = rng.random((rows, width))
    result = edge_precision(
        candidate_ids=truth_ids,
        candidate_cosines=truth_cos,
        truth_ids=truth_ids,
        truth_cosines=truth_cos,
    )
    assert result["emitted_over_true_ratio"] == pytest.approx(1.0, rel=1e-9)
    assert result["substituted_edge_fraction"] == pytest.approx(0.0)
    assert result["mean_missed_true_edge_cosine"] is None


def test_edge_precision_sees_a_wholly_substituted_graph():
    rows, width = 200, 15
    truth_ids = np.arange(rows * width).reshape(rows, width)
    truth_cos = np.full((rows, width), 0.9)
    candidate_ids = truth_ids + rows * width
    candidate_cos = np.full((rows, width), 0.2)
    result = edge_precision(
        candidate_ids=candidate_ids,
        candidate_cosines=candidate_cos,
        truth_ids=truth_ids,
        truth_cosines=truth_cos,
    )
    assert result["substituted_edge_fraction"] == pytest.approx(1.0)
    assert result["mean_substituted_edge_cosine"] == pytest.approx(0.2)
    assert result["mean_missed_true_edge_cosine"] == pytest.approx(0.9)


def test_concentration_statistics_refuse_mismatched_inputs():
    with pytest.raises(Round0227ConcentrationError):
        density_decile_recall(np.zeros(10), np.zeros(9))
    with pytest.raises(Round0227ConcentrationError):
        neighbour_loss_autocorrelation(np.zeros(10), np.zeros((9, 15)))
    with pytest.raises(Round0227ConcentrationError):
        loss_concentration(np.zeros((10, 2)))


# --------------------------------------------------------------------------- #
# the chunked tripwire agrees with the reference implementation
# --------------------------------------------------------------------------- #
def test_chunked_graph_validity_matches_the_reference():
    rng = np.random.default_rng(3)
    rows, width = 2_500, 15
    ids = rng.integers(0, rows, size=(rows, width)).astype(np.int32)
    ids[7, 3] = 7            # a self loop
    ids[11, 0] = ids[11, 1]  # a duplicate
    ids[13, 2] = -1          # a sentinel
    reference = graph_validity(ids, rows=rows)
    chunked = round0227_nodes._graph_validity_chunked(ids, rows=rows)
    for key, value in reference.items():
        assert chunked[key] == value, key


def test_chunked_graph_validity_uses_global_row_ids_across_chunks():
    rows, width = 2_500_000, 15
    # Each row points at the next 15 rows, so no row is its own neighbour and
    # no row repeats one — a clean graph, into which one defect is planted past
    # the 1,000,000-row chunk boundary.
    ids = (
        (np.arange(rows, dtype=np.int64)[:, None] + np.arange(1, width + 1)[None, :])
        % rows
    ).astype(np.int32)
    clean = round0227_nodes._graph_validity_chunked(ids, rows=rows)
    assert clean["self_loop_entries"] == 0
    assert clean["duplicate_entries"] == 0
    assert clean["zero_degree_rows"] == 0
    ids[2_000_000, 0] = 2_000_000  # a self loop past the first chunk boundary
    chunked = round0227_nodes._graph_validity_chunked(ids, rows=rows)
    assert chunked["self_loop_entries"] == 1
    assert chunked["rows_with_self_loop"] == 1
    assert chunked["min_usable_degree"] == width - 1


# --------------------------------------------------------------------------- #
# the projection
# --------------------------------------------------------------------------- #
def _example_fits():
    cluster_rows = [125_000, 250_000, 500_000, 1_000_000, 2_000_000]
    seconds = [value / 400_000.0 for value in cluster_rows]
    return power_fit(cluster_rows, seconds)


def test_projection_terms_are_all_positive_and_sum_to_the_total():
    fit = _example_fits()
    projection = project_100m(
        clusters=22,
        per_cluster_nn_descent=fit,
        cosine_per_spilled_row_s=5e-7,
        merge_per_row_s=2e-6,
        kmeans_assign_seconds=30.0,
        imbalance=1.3,
    )
    terms = projection["terms_seconds"]
    assert all(value > 0 for value in terms.values())
    assert sum(terms.values()) == pytest.approx(projection["projected_seconds"])
    assert projection["is_projection"] is True
    assert projection["substrate_passes"] >= 1


def test_projection_charges_more_io_when_clusters_are_fewer():
    fit = _example_fits()
    many = project_100m(
        clusters=200, per_cluster_nn_descent=fit, cosine_per_spilled_row_s=5e-7,
        merge_per_row_s=2e-6, kmeans_assign_seconds=30.0, imbalance=1.7,
    )
    few = project_100m(
        clusters=22, per_cluster_nn_descent=fit, cosine_per_spilled_row_s=5e-7,
        merge_per_row_s=2e-6, kmeans_assign_seconds=30.0, imbalance=1.3,
    )
    assert few["spill_read_bytes"] >= many["spill_read_bytes"]
    assert few["terms_seconds"]["spill_io"] >= many["terms_seconds"]["spill_io"]


def test_projection_carries_a_basis_for_every_term():
    fit = _example_fits()
    projection = project_100m(
        clusters=22, per_cluster_nn_descent=fit, cosine_per_spilled_row_s=5e-7,
        merge_per_row_s=2e-6, kmeans_assign_seconds=30.0, imbalance=1.3,
    )
    assert set(projection["bases"]) == set(projection["terms_seconds"])


def test_fits_reject_degenerate_input():
    with pytest.raises(Round0227Error):
        linear_fit([1.0], [1.0])
    with pytest.raises(Round0227Error):
        power_fit([1.0, 2.0], [0.0, 1.0])


def test_interpolation_stays_inside_the_measured_bracket():
    points = {8: 0.98, 32: 0.94}
    value, basis = round0227_nodes._interpolate(points, 16)
    assert 0.94 <= value <= 0.98
    assert "interpolated" in basis
    clamped, basis_low = round0227_nodes._interpolate(points, 4)
    assert clamped == 0.98
    assert "clamped" in basis_low


# --------------------------------------------------------------------------- #
# the thread-lifecycle guard — the CLASS of defect, not the one instance
# --------------------------------------------------------------------------- #
THREAD_MODULES = (
    round0226_cluster_spill_build,
    round0227_cluster_spill_build,
    round0227_reachability_probe,
    round0227_nodes,
)


def _thread_subclasses():
    seen: dict[str, type] = {}
    for module in THREAD_MODULES:
        for name, value in vars(module).items():
            if (
                inspect.isclass(value)
                and issubclass(value, threading.Thread)
                and value is not threading.Thread
            ):
                seen[f"{value.__module__}.{value.__qualname__}"] = value
    return seen


def test_this_round_can_reach_at_least_one_thread_subclass():
    # A meta-test that enumerates nothing proves nothing.
    assert _thread_subclasses()


def _reserved_thread_names() -> set[str]:
    """Every name `threading.Thread` owns — class attributes AND instance ones.

    `dir(threading.Thread)` alone is not the answer, and assuming it is would
    have missed the original defect entirely: `_tstate_lock`, `_started`,
    `_target` and `_is_stopped` are assigned in `Thread.__init__` and live in the
    *instance* dict, so they never appear on the class. `_stop` — the name that
    actually broke R0226 — is a class-level method, but it was shadowed by an
    *instance* assignment. Both halves of that have to be in scope.
    """
    probe = threading.Thread()
    names = set(dir(threading.Thread)) | set(vars(probe))
    # `run` is the documented override point and the only exemption.
    return names - {"run"}


def _self_assigned_names(subclass: type) -> set[str]:
    """Names a class binds on `self`, found in its source rather than guessed.

    R0226's bug was `self._stop = threading.Event()` inside `__init__`. That
    lands in the instance dictionary, so a check over `vars(subclass)` — the
    class dictionary — cannot see it. Reading the assignments out of the source
    catches it without having to construct every class.
    """
    import ast
    import textwrap

    try:
        source = inspect.getsource(subclass)
    except (OSError, TypeError):  # pragma: no cover - source always available here
        return set()
    tree = ast.parse(textwrap.dedent(source))
    found: set[str] = set()
    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        for target in targets:
            for item in ast.walk(target):
                if (
                    isinstance(item, ast.Attribute)
                    and isinstance(item.value, ast.Name)
                    and item.value.id == "self"
                ):
                    found.add(item.attr)
    return found


def test_the_reserved_name_set_contains_the_names_that_have_bitten_us():
    reserved = _reserved_thread_names()
    for name in ("_stop", "join", "start", "_tstate_lock", "_started", "_target",
                 "_is_stopped", "_bootstrap", "name", "daemon", "ident"):
        assert name in reserved, name
    assert "run" not in reserved


def test_the_assignment_scanner_would_have_caught_the_r0226_defect():
    """The meta-test is only worth having if it fails on the original bug."""

    class _Regression(threading.Thread):
        def __init__(self) -> None:
            super().__init__(daemon=True)
            self._stop = threading.Event()  # noqa: SLF001 - the R0226 defect

    assert "_stop" in _self_assigned_names(_Regression)
    assert _self_assigned_names(_Regression) & _reserved_thread_names()


def test_no_thread_subclass_shadows_any_threading_thread_internal():
    """R0226 shadowed `Thread._stop` and broke `join()` after a build finished.

    Review-0226-01: the regression test covered the one class that did not
    break, and guarded one name out of several. This guards every name
    `threading.Thread` owns, class-level and instance-level, against both the
    class dictionary and every `self.<name> =` in the source, for every `Thread`
    subclass this round can reach.
    """
    reserved = _reserved_thread_names()
    for label, subclass in _thread_subclasses().items():
        bound = set(vars(subclass)) | _self_assigned_names(subclass)
        collisions = {
            name
            for name in bound
            if name in reserved
            and not (name.startswith("__") and name.endswith("__"))
        }
        assert not collisions, f"{label} shadows threading.Thread {sorted(collisions)}"


def test_thread_subclasses_expose_a_halt_and_can_be_joined():
    for label, subclass in _thread_subclasses().items():
        assert hasattr(subclass, "halt"), f"{label} has no cooperative halt"
        assert callable(subclass.halt)
        # `_stop` must remain the bound method the interpreter installed.
        assert callable(getattr(subclass, "_stop", None)), label


def test_build_watchdog_starts_halts_and_joins_cleanly():
    watchdog = round0227_nodes.BuildWatchdog(
        pid=0,
        poll_s=0.01,
        host_anon_budget_bytes=1 << 60,
        swap_growth_abort_bytes=1 << 60,
        device_baseline_bytes=0,
        swap_baseline_bytes=0,
    )
    watchdog.start()
    watchdog.halt()
    assert watchdog.is_alive() is False
    readings = watchdog.readings()
    assert readings["watchdog_aborted"] is False
    assert readings["watchdog_escalations"] == []


def test_sampler_starts_halts_and_joins_cleanly():
    """The class where the `_stop` bug actually fired, which R0226 never tested."""

    class _FakeRuntime:
        @staticmethod
        def memGetInfo():
            return (1, 2)

    class _FakeCuda:
        runtime = _FakeRuntime()

    class _FakeCupy:
        cuda = _FakeCuda()

    sampler = round0226_cluster_spill_build.Sampler(_FakeCupy(), 0.01)
    sampler.start()
    sampler.halt()
    assert sampler.is_alive() is False
    assert sampler.samples >= 1
    assert sampler.device_peak == 1


# --------------------------------------------------------------------------- #
# the ladder's stop rule
# --------------------------------------------------------------------------- #
def test_ladder_stops_at_the_first_cell_that_does_not_fit():
    settings = [
        {"id": "a", "rows": 1, "clusters": 4},
        {"id": "b", "rows": 2, "clusters": 4},
        {"id": "c", "rows": 3, "clusters": 4},
    ]
    attempted: list[str] = []

    def make_config(setting):
        return {
            "setting_id": setting["id"],
            "rows": setting["rows"],
            "clusters": setting["clusters"],
        }

    def run_cell(config, _setting):
        attempted.append(config["setting_id"])
        return {"fit": config["setting_id"] != "b", "error_type": "Boom"}

    records = round0227_nodes.run_ascending_ladder(
        settings=settings, make_config=make_config, run_cell=run_cell
    )
    assert attempted == ["a", "b"]
    assert len(records) == 3
    assert records[2]["skipped_after_failure_at_smaller_max_cluster"] is True
    assert records[2]["skip_reason"]


def test_skipped_cells_still_publish_every_instrument():
    settings = [
        {"id": "a", "rows": 1, "clusters": 4},
        {"id": "b", "rows": 2, "clusters": 4},
    ]

    def make_config(setting):
        return {
            "setting_id": setting["id"],
            "rows": setting["rows"],
            "clusters": setting["clusters"],
        }

    records = round0227_nodes.run_ascending_ladder(
        settings=settings,
        make_config=make_config,
        run_cell=lambda config, setting: {"fit": False, "error_type": "Boom"},
    )
    for instrument in round0227_nodes.LADDER_INSTRUMENTS:
        assert instrument in records[1]
