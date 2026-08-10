"""R0242 CPU smoke — every new guard gets a PLANTED DEFECT it must catch.

AGENT_STARTUP: "Any guard, detector, or tripwire the round adds must ship a
positive control - a test that plants the defect and proves the guard catches
it. A guard whose test suite contains no failing input is untested at its only
job." R0242 adds three: the post-canonicalization degree-zero tripwire, the
cluster-locality halt rule, and the canonicalization itself. Each one below is
run against an input that MUST fail it and an input that MUST pass it.

Everything here is CPU-only, allocates kilobytes, and touches no card.
"""
from __future__ import annotations

import numpy as np
import pytest

from basemap.round0220_cuvs_qualification import strict_containment_rows
from basemap.round0242_locality import (
    CLUSTERS,
    Round0242Error,
    StageGuard0242,
    assign_torch,
    canonical_undirected_degrees,
    kmeans_torch,
    cluster_locality_test,
    cluster_rate_table,
    in_degree_bin_table,
    locality_verdict,
    loss_decomposition,
    partition_agreement,
    partition_reachability,
    post_canonical_tripwire,
    reproduce_r0241_in_degree,
    reproduce_r0241_recall,
    spearman_permutation,
    symmetrised_degree_once,
    weight_distribution,
)


# --------------------------------------------------------------------------- #
# guard 1 — the post-canonicalization degree-zero tripwire
# --------------------------------------------------------------------------- #
def _ring(rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A symmetric ring: every row has two neighbours, nobody is isolated."""
    src = np.concatenate([np.arange(rows), np.arange(rows)]).astype(np.int32)
    dst = np.concatenate([
        (np.arange(rows) + 1) % rows, (np.arange(rows) - 1) % rows
    ]).astype(np.int32)
    wts = np.full(src.size, 0.5, dtype=np.float32)
    return src, dst, wts


def test_canonicalization_folds_mirrors_and_keeps_every_row() -> None:
    rows = 64
    src, dst, wts = _ring(rows)
    result = canonical_undirected_degrees(
        src=src, dst=dst, weights=wts, rows=rows, block=17
    )
    # A ring of 64 has 64 undirected edges; the directed list carries 128.
    assert result["directed_entries_scanned"] == 128
    assert result["canonical_undirected_edges"] == rows
    assert result["self_loop_entries_dropped"] == 0
    assert result["non_positive_weight_entries_dropped"] == 0
    tripwire = post_canonical_tripwire(degree=result["degree"], rows=rows)
    assert tripwire["holds"] is True
    assert tripwire["zero_degree_rows"] == 0
    assert tripwire["min_degree"] == 2 and tripwire["max_degree"] == 2


def test_post_canonical_tripwire_CATCHES_a_planted_edgeless_row() -> None:
    """The v1 defect, planted: a row whose every weight underflowed to zero.

    R0034 shipped `2,779,481` such rows at 150M and R0215 traced the clumps to
    them. A tripwire on the RAW directed graph cannot see this, because the row
    still has `k` out-entries; only a post-canonicalization degree can.
    """
    rows = 64
    src, dst, wts = _ring(rows)
    victim = 37
    wts = wts.copy()
    wts[(src == victim) | (dst == victim)] = 0.0
    result = canonical_undirected_degrees(
        src=src, dst=dst, weights=wts, rows=rows, block=17
    )
    assert result["non_positive_weight_entries_dropped"] == 4
    tripwire = post_canonical_tripwire(degree=result["degree"], rows=rows)
    assert tripwire["holds"] is False
    assert tripwire["zero_degree_rows"] == 1
    assert tripwire["min_degree"] == 0


def test_post_canonical_tripwire_CATCHES_a_row_left_with_only_a_self_loop() -> None:
    """The other v1 shape: every edge a row has is a self-loop."""
    rows = 8
    src = np.array([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0], dtype=np.int32)
    dst = np.array([1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7], dtype=np.int32)
    # Detach row 5 from the chain and give it only a self-loop.
    mask = ~((src == 5) | (dst == 5))
    src = np.concatenate([src[mask], np.array([5], dtype=np.int32)])
    dst = np.concatenate([dst[mask], np.array([5], dtype=np.int32)])
    wts = np.full(src.size, 0.25, dtype=np.float32)
    result = canonical_undirected_degrees(
        src=src, dst=dst, weights=wts, rows=rows, block=5
    )
    assert result["self_loop_entries_dropped"] == 1
    tripwire = post_canonical_tripwire(degree=result["degree"], rows=rows)
    assert tripwire["holds"] is False
    assert tripwire["zero_degree_rows"] == 1


def test_canonicalization_counts_out_of_range_rather_than_crashing() -> None:
    rows = 8
    src = np.array([0, 1, 2], dtype=np.int32)
    dst = np.array([1, 99, 3], dtype=np.int32)
    wts = np.full(3, 0.5, dtype=np.float32)
    result = canonical_undirected_degrees(
        src=src, dst=dst, weights=wts, rows=rows, block=2
    )
    assert result["out_of_range_entries"] == 1
    assert result["canonical_undirected_edges"] == 2


# --------------------------------------------------------------------------- #
# guard 2 — symmetrised degree, reported ONCE, identity asserted
# --------------------------------------------------------------------------- #
def test_symmetrised_degree_is_reported_once_and_the_identity_holds() -> None:
    rows = 64
    src, dst, wts = _ring(rows)
    summary = symmetrised_degree_once(
        src=src, dst=dst, rows=rows, sample=rows, seed=1, block=9
    )
    assert summary["reported"] == "once"
    assert summary["identity_cross_check"][
        "in_degree_equals_out_degree_on_every_sampled_row"
    ] is True
    assert summary["min"] == 2 and summary["max"] == 2
    assert summary["zero_degree_rows"] == 0
    assert "degree" in summary and summary["degree"].shape == (rows,)


def test_symmetrised_identity_cross_check_FAILS_on_an_asymmetric_list() -> None:
    """Planted defect: a directed list that is not symmetric must be caught."""
    rows = 16
    src = np.arange(rows, dtype=np.int32)
    dst = ((np.arange(rows) + 1) % rows).astype(np.int32)
    summary = symmetrised_degree_once(
        src=src, dst=dst, rows=rows, sample=rows, seed=1, block=5
    )
    # Out-degree is 1 everywhere and in-degree is 1 everywhere, so the ring is
    # a degenerate case; break it properly by doubling one direction.
    src = np.concatenate([src, np.array([0, 0], dtype=np.int32)])
    dst = np.concatenate([dst, np.array([5, 7], dtype=np.int32)])
    summary = symmetrised_degree_once(
        src=src, dst=dst, rows=rows, sample=rows, seed=1, block=5
    )
    assert summary["identity_cross_check"][
        "in_degree_equals_out_degree_on_every_sampled_row"
    ] is False


# --------------------------------------------------------------------------- #
# guard 3 — the cluster locality test and its halt rule
# --------------------------------------------------------------------------- #
def _uniform_case(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = 20_000
    labels = rng.integers(0, CLUSTERS, size=rows)
    missing = rng.binomial(15, 0.004, size=rows).astype(np.float64)
    exposure = np.full(rows, 15.0)
    return labels, missing, exposure


def test_locality_test_does_NOT_fire_on_uniform_loss() -> None:
    labels, missing, exposure = _uniform_case()
    result = cluster_locality_test(
        labels=labels, missing=missing, exposure=exposure, clusters=CLUSTERS,
        permutations=200, seed=1, top_m=20, population="uniform control",
    )
    assert result["chi_square"]["p_value"] > 0.01
    assert result["top_m_share_of_missing"]["observed"] < 0.25


def test_locality_test_CATCHES_a_planted_concentration() -> None:
    """Planted defect: all the loss piled into ten clusters."""
    rng = np.random.default_rng(11)
    rows = 20_000
    labels = rng.integers(0, CLUSTERS, size=rows)
    missing = np.zeros(rows, dtype=np.float64)
    hot = np.isin(labels, np.arange(10))
    missing[hot] = rng.binomial(15, 0.5, size=int(hot.sum()))
    exposure = np.full(rows, 15.0)
    result = cluster_locality_test(
        labels=labels, missing=missing, exposure=exposure, clusters=CLUSTERS,
        permutations=200, seed=2, top_m=20, population="planted concentration",
    )
    assert result["chi_square"]["p_value"] <= 1.0 / 201.0
    assert result["top_m_share_of_missing"]["observed"] == pytest.approx(1.0)
    assert result["top_m_share_of_missing"]["observed"] > (
        result["top_m_share_of_missing"]["null_p99_9"]
    )


def test_halt_rule_halts_on_concentration_and_not_on_diffuse_structure() -> None:
    labels, missing, exposure = _uniform_case()
    clean = cluster_locality_test(
        labels=labels, missing=missing, exposure=exposure, clusters=CLUSTERS,
        permutations=200, seed=3, top_m=20, population="clean",
    )
    rng = np.random.default_rng(13)
    rows = labels.size
    hot = np.isin(labels, np.arange(10))
    concentrated_missing = np.zeros(rows, dtype=np.float64)
    concentrated_missing[hot] = rng.binomial(15, 0.5, size=int(hot.sum()))
    dirty = cluster_locality_test(
        labels=labels, missing=concentrated_missing, exposure=exposure,
        clusters=CLUSTERS, permutations=200, seed=4, top_m=20,
        population="concentrated",
    )
    assert locality_verdict(
        builder_test=clean, partition_test=dirty, total_test=clean
    )["halt_part_b"] is False, (
        "the partition-limited population must NOT be able to halt Part B"
    )
    verdict = locality_verdict(
        builder_test=dirty, partition_test=clean, total_test=dirty
    )
    assert verdict["halt_part_b"] is True
    assert verdict["reading"] == "spatially concentrated"


def test_halt_rule_reads_significant_but_diffuse_without_halting() -> None:
    """Significance alone must not stop a product step at n = 500,000."""
    rng = np.random.default_rng(17)
    rows = 60_000
    labels = rng.integers(0, CLUSTERS, size=rows)
    # A weak, broad gradient: every cluster differs a little, none dominates.
    rate = 0.002 + 0.004 * (labels / float(CLUSTERS))
    missing = rng.binomial(15, rate).astype(np.float64)
    exposure = np.full(rows, 15.0)
    result = cluster_locality_test(
        labels=labels, missing=missing, exposure=exposure, clusters=CLUSTERS,
        permutations=400, seed=5, top_m=20, population="diffuse gradient",
    )
    verdict = locality_verdict(
        builder_test=result, partition_test=result, total_test=result
    )
    assert verdict["builder_top_m_share_of_missing"] < 0.25
    assert verdict["halt_part_b"] is False


def test_cluster_rate_table_suppresses_thin_clusters() -> None:
    labels, missing, exposure = _uniform_case()
    table = cluster_rate_table(
        labels=labels, missing=missing, exposure=exposure, clusters=CLUSTERS,
        report=5, minimum_rows=1000,
    )
    assert table["clusters_scored"] == 0 or table["clusters_scored"] < CLUSTERS
    table = cluster_rate_table(
        labels=labels, missing=missing, exposure=exposure, clusters=CLUSTERS,
        report=5, minimum_rows=1,
    )
    assert table["clusters_scored"] > 0
    assert len(table["highest_rate_clusters"]) == 5


def test_cluster_test_refuses_a_label_out_of_range() -> None:
    labels = np.array([0, 1, CLUSTERS], dtype=np.int64)
    with pytest.raises(Round0242Error):
        cluster_locality_test(
            labels=labels, missing=np.ones(3), exposure=np.full(3, 15.0),
            clusters=CLUSTERS, permutations=2, seed=1, top_m=2,
            population="bad labels",
        )


# --------------------------------------------------------------------------- #
# the decomposition, and the joins
# --------------------------------------------------------------------------- #
def test_loss_decomposition_splits_partition_from_builder() -> None:
    k = 15
    strict = np.array([1.0, 13 / 15, 10 / 15, 1.0])
    reach = np.array([1.0, 1.0, 12 / 15, 13 / 15])
    result = loss_decomposition(strict=strict, reachability=reach, k=k)
    assert result["total_missing_edges"] == 0 + 2 + 5 + 0
    # Row 1 loses two with everything reachable -> both are builder loss.
    # Row 2 loses five of which three were unreachable -> two are builder loss.
    # Row 3 loses none although two were unreachable -> it RECOVERED them.
    assert result["builder_missing_edges"] == 2 + 2
    assert result["partition_forced_missing_edges"] == 3
    assert result["rows_recovering_an_unreachable_neighbour"] == 1
    assert result["rows_builder_loss_with_truth_fully_reachable"] == 1
    assert result["rows_both"] == 1
    vectors = result["vectors"]
    assert list(vectors["exposure_builder"]) == [15, 15, 12, 13]


def test_loss_decomposition_refuses_mismatched_vectors() -> None:
    with pytest.raises(Round0242Error):
        loss_decomposition(strict=np.ones(4), reachability=np.ones(3))


def test_spearman_permutation_finds_a_planted_monotone_relation() -> None:
    rng = np.random.default_rng(3)
    x = rng.integers(0, 50, size=2_000)
    y = x + rng.normal(0, 1.0, size=2_000)
    result = spearman_permutation(x, y, permutations=200, seed=1, label="planted")
    assert result["spearman_rho"] > 0.9
    assert result["two_sided_p_value"] <= 1.0 / 201.0


def test_spearman_permutation_stays_null_on_independent_vectors() -> None:
    rng = np.random.default_rng(4)
    x = rng.integers(0, 50, size=2_000)
    y = rng.integers(0, 16, size=2_000)
    result = spearman_permutation(
        x, y, permutations=200, seed=2, label="independent"
    )
    assert result["two_sided_p_value"] > 0.01
    assert abs(result["spearman_rho"]) < 0.1


def test_in_degree_bin_table_isolates_the_zero_bucket() -> None:
    in_degree = np.array([0, 0, 3, 7, 25, 1500])
    lost = np.array([2, 1, 0, 0, 0, 3])
    builder = np.array([1, 1, 0, 0, 0, 3])
    exposure = np.full(6, 15)
    table = in_degree_bin_table(
        in_degree=in_degree, lost=lost, builder_lost=builder,
        exposure_builder=exposure, k=15,
    )
    zero_band = next(row for row in table if row["in_degree_low"] == 0)
    assert zero_band["in_degree_high"] == 0
    assert zero_band["probe_rows"] == 2
    assert zero_band["missing_edges"] == 3
    top = next(row for row in table if row["in_degree_low"] == 1000)
    assert top["in_degree_high"] is None and top["missing_edges"] == 3


# --------------------------------------------------------------------------- #
# reproduction gates — the round stops rather than measuring a different thing
# --------------------------------------------------------------------------- #
def test_recall_reproduction_gate_detects_a_disagreement() -> None:
    sealed = {
        "strict": {"mean": 0.9959034666666666},
        "tie_aware": {"mean": 0.9979422666666667},
        "rows_carrying_any_loss": 14_330,
        "missing_true_edges": 30_724,
        "tie_aware_rows_at_zero": 25,
    }
    assert reproduce_r0241_recall(measured=sealed, sealed=sealed)["agree"] is True
    drifted = dict(sealed, rows_carrying_any_loss=14_331)
    result = reproduce_r0241_recall(measured=drifted, sealed=sealed)
    assert result["agree"] is False
    assert result["disagreements"] == ["rows_carrying_any_loss"]


def test_in_degree_reproduction_gate_detects_a_disagreement() -> None:
    sealed = {
        "zero_rows": 4_424_010, "max": 44_530, "mean": 15.0,
        "rows_at_or_above_1000": 261,
        "share_of_edges_in_top_1_percent_of_rows": 0.091622168,
    }
    assert reproduce_r0241_in_degree(measured=sealed, sealed=sealed)["agree"] is True
    drifted = dict(sealed, max=44_531)
    assert reproduce_r0241_in_degree(
        measured=drifted, sealed=sealed
    )["agree"] is False


def test_partition_agreement_refuses_a_different_realisation() -> None:
    sealed = np.full(1_000, 1.0)
    sealed[0] = 0.0
    close = sealed.copy()
    result = partition_agreement(
        reproduced=close, sealed=sealed, sealed_mean=float(sealed.mean())
    )
    assert result["holds"] is True and result["fraction_identical"] == 1.0
    far = np.full(1_000, 0.5)
    assert partition_agreement(
        reproduced=far, sealed=sealed, sealed_mean=float(sealed.mean())
    )["holds"] is False


def test_partition_reachability_reproduces_a_hand_built_case() -> None:
    assignment = np.array([[0, 1], [0, 2], [3, 4], [5, 6]], dtype=np.int32)
    probe = np.array([0], dtype=np.int64)
    truth = np.array([[1, 2, 3]], dtype=np.int64)
    # Row 0 is in {0, 1}; row 1 is {0, 2} (shares 0), row 2 is {3, 4} (no),
    # row 3 is {5, 6} (no). So one of three truth neighbours is reachable.
    result = partition_reachability(
        assignment=assignment, probe_rows=probe, truth_ids=truth, block=1
    )
    assert result[0] == pytest.approx(1 / 3)


def test_strict_containment_still_agrees_with_the_reviewed_function() -> None:
    """The loss vector must come from the REVIEWED check, not a local copy."""
    candidates = np.array([[1, 2, 3], [4, 5, 6]])
    truth = np.array([[1, 2, 9], [7, 8, 9]])
    assert list(strict_containment_rows(candidates, truth)) == [2 / 3, 0.0]


# --------------------------------------------------------------------------- #
# the wall guard, with review-0241-01/F5's corrections
# --------------------------------------------------------------------------- #
def test_stage_guard_stamps_its_wall_at_stage_completion() -> None:
    ticks = iter([0.0, 1.0, 2.0, 3.0, 50.0, 60.0])
    guard = StageGuard0242(
        label="probe", units_total=4, budget_s=1_000.0, deadline_s=1_000.0,
        clock=lambda: next(ticks), calibration_units=2,
    )
    guard.units_done = 2
    guard.stop()
    receipt = guard.receipt()
    # The wall is the stamp, not whatever the clock says when the receipt is
    # written - which is exactly the R0241 defect review-0241-01/F5 found.
    assert receipt["wall_s"] == 1.0
    assert receipt["wall_is_stage_completion_stamp"] is True
    assert guard.receipt()["wall_s"] == 1.0


def test_stage_guard_deadline_reached_is_measured_not_a_literal() -> None:
    ticks = iter([0.0, 5.0, 5.0])
    guard = StageGuard0242(
        label="probe", units_total=1, budget_s=1_000.0, deadline_s=1.0,
        clock=lambda: next(ticks),
    )
    guard.stop()
    assert guard.receipt()["deadline_reached"] is True


def test_stage_guard_refuses_a_stage_it_cannot_fit() -> None:
    ticks = iter([0.0, 100.0, 100.0])
    guard = StageGuard0242(
        label="probe", units_total=1_000, budget_s=10.0, deadline_s=1e9,
        clock=lambda: next(ticks), calibration_units=1,
    )
    with pytest.raises(Exception):
        guard.unit_done("first")


# --------------------------------------------------------------------------- #
# the fuzzy weight scan
# --------------------------------------------------------------------------- #
def test_weight_distribution_flags_a_planted_zero_weight() -> None:
    weights = np.full(1_000, 0.5, dtype=np.float32)
    assert weight_distribution(
        weights, block=64, quantile_sample=100, seed=1
    )["valid"] is True
    weights[17] = 0.0
    scan = weight_distribution(weights, block=64, quantile_sample=100, seed=1)
    assert scan["valid"] is False
    assert scan["entries_at_or_below_zero"] == 1


def test_weight_distribution_flags_a_planted_out_of_range_weight() -> None:
    weights = np.full(1_000, 0.5, dtype=np.float32)
    weights[3] = 1.5
    scan = weight_distribution(weights, block=64, quantile_sample=100, seed=1)
    assert scan["valid"] is False
    assert scan["max"] == pytest.approx(1.5)


# --------------------------------------------------------------------------- #
# the partition backend — the torch transcription against a numpy reference
# --------------------------------------------------------------------------- #
def _reference_kmeans(
    dataset: np.ndarray, *, clusters: int, seed: int, subsample_rows: int,
    iterations: int, block: int = 100_000,
) -> np.ndarray:
    """R0226's `_kmeans`, transcribed to plain numpy from its own source.

    Written independently of `kmeans_torch` so that the two agreeing is
    evidence the torch path is the registered algorithm rather than evidence
    that one file was copied twice.
    """
    rows = int(dataset.shape[0])
    take = min(rows, int(subsample_rows))
    rng = np.random.default_rng(int(seed))
    sample_rows = np.sort(rng.choice(rows, size=take, replace=False))
    sample = np.ascontiguousarray(dataset[sample_rows], dtype=np.float32)
    start = np.sort(rng.choice(take, size=int(clusters), replace=False))
    centroids = sample[start].copy()
    for _ in range(int(iterations)):
        sums = np.zeros((int(clusters), sample.shape[1]), dtype=np.float32)
        counts = np.zeros(int(clusters), dtype=np.float32)
        for begin in range(0, take, int(block)):
            end = min(begin + int(block), take)
            chunk = sample[begin:end]
            nearest = np.argmax(chunk @ centroids.T, axis=1)
            np.add.at(sums, nearest, chunk)
            np.add.at(counts, nearest, np.ones(end - begin, dtype=np.float32))
        empty = counts == 0
        safe = np.where(empty, np.float32(1.0), counts)
        updated = sums / safe[:, None]
        centroids = np.where(empty[:, None], centroids, updated)
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / np.maximum(norms, np.float32(1e-12))
    return centroids


def _unit_rows(rows: int, dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((rows, dimension)).astype(np.float32)
    return (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)


def test_torch_kmeans_reproduces_the_numpy_reference_of_r0226() -> None:
    torch = pytest.importorskip("torch")
    data = _unit_rows(4_000, 16, seed=9)
    reference = _reference_kmeans(
        data, clusters=12, seed=226, subsample_rows=2_000, iterations=25,
        block=512,
    )
    produced = kmeans_torch(
        torch, data, clusters=12, seed=226, subsample_rows=2_000,
        iterations=25, device=torch.device("cpu"), block=512,
    ).numpy()
    assert produced.shape == reference.shape
    assert np.abs(produced - reference).max() < 1e-4


def test_torch_assign_reproduces_a_numpy_argsort_assignment() -> None:
    torch = pytest.importorskip("torch")
    data = _unit_rows(500, 16, seed=11)
    centroids = torch.as_tensor(_unit_rows(12, 16, seed=12))
    produced = assign_torch(
        torch, data, centroids, rows=500, spill=4, block=64,
        device=torch.device("cpu"),
    )
    scores = data @ centroids.numpy().T
    reference = np.argsort(-scores, axis=1)[:, :4]
    assert produced.shape == (500, 4)
    assert np.array_equal(produced, reference)


def test_torch_assign_polls_the_cooperative_flag_every_block() -> None:
    torch = pytest.importorskip("torch")
    data = _unit_rows(500, 8, seed=13)
    centroids = torch.as_tensor(_unit_rows(5, 8, seed=14))
    seen: list[str] = []
    assign_torch(
        torch, data, centroids, rows=500, spill=2, block=100,
        device=torch.device("cpu"), poll=seen.append,
    )
    assert len(seen) == 5
