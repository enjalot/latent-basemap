"""R0233 contract tests — the registered checks must actually fail closed."""
from __future__ import annotations

import numpy as np
import pytest

from basemap import round0233_substrate as contract


def test_composition_is_the_owner_confirmed_shares_at_this_rung():
    assert contract.ROWS == 6_250_000
    counts = dict(contract.COMPOSITION)
    assert sum(counts.values()) == contract.ROWS
    shares = {name: value / contract.ROWS for name, value in counts.items()}
    assert shares["fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2"] == 0.40
    assert shares["RedPajama-Data-V2-sample-10B-chunked-120-all-MiniLM-L6-v2"] == 0.25
    assert shares["pile-uncopyrighted-chunked-120-all-MiniLM-L6-v2"] == 0.25
    assert shares["starcoderdata-code-chunked-120-all-MiniLM-L6-v2"] == 0.10


def test_damaged_fineweb_shard_37_is_excluded():
    key = (
        "fineweb-edu-sample-10BT-chunked-120-all-MiniLM-L6-v2/train/"
        "data-00037-of-00099.npy"
    )
    assert key in contract.EXCLUDED_SHARDS


def test_composition_validation_fails_closed():
    good = dict(contract.COMPOSITION)
    assert contract.validate_composition(good)
    bad = dict(good)
    first = contract.COMPOSITION[0][0]
    bad[first] = bad[first] - 1
    with pytest.raises(contract.Round0233Error):
        contract.validate_composition(bad)


def test_shard_span_assertion_raises_on_a_prefix():
    assert contract.validate_shard_span(
        corpus="x", shards_touched=100, shards_total=100
    )["coverage"] == 1.0
    # R0216's executed defect: 94.16% of fineweb's shards.
    with pytest.raises(contract.Round0233Error):
        contract.validate_shard_span(corpus="x", shards_touched=94, shards_total=100)
    # A single missed shard out of 98 is already below the 99.9% floor.
    with pytest.raises(contract.Round0233Error):
        contract.validate_shard_span(corpus="x", shards_touched=97, shards_total=98)


def test_reserve_split_closes_and_is_deterministic():
    total = 0
    for index in range(len(contract.COMPOSITION)):
        corpus_side, query_side = contract.reserve_split(index)
        assert corpus_side.size == contract.RESERVE_CORPUS_ROWS
        assert query_side.size == contract.RESERVE_QUERY_ROWS
        assert np.intersect1d(corpus_side, query_side).size == 0
        union = np.union1d(corpus_side, query_side)
        assert union.size == contract.RESERVE_ROWS_PER_CORPUS
        again, _ = contract.reserve_split(index)
        assert np.array_equal(corpus_side, again)
        total += union.size
    assert total == contract.RESERVE_ROWS


def test_c_min_is_twice_the_spill_and_rejects_the_degenerate_cell():
    assert contract.SPILL == 8
    assert contract.C_MIN == 16
    # R0229's per-rung table selected c = 8 at (6.25M, s = 8), which is c = s:
    # every row lands in every cluster and the projected largest cluster
    # (7,596,838) exceeds N. It must not be selectable.
    decision = contract.select_clusters(
        rows=contract.ROWS,
        measured_imbalance={8: 1.0, 16: 1.1834, 32: 1.4237},
        candidates=(8, 16, 32),
    )
    assert decision["selected_clusters"] == 16
    degenerate = [
        item for item in decision["candidates_considered"]
        if item["clusters"] == 8
    ][0]
    assert degenerate["admissible"] is False
    assert "C_MIN" in degenerate["reason"]


def test_c_selection_takes_the_smallest_admissible_and_never_interpolates():
    decision = contract.select_clusters(
        rows=contract.ROWS,
        measured_imbalance={16: 1.1834, 32: 1.4237, 64: 1.5390, 200: 2.1311},
    )
    assert decision["selected_clusters"] == 16
    assert decision["selection"]["imbalance_source"].startswith("measured")
    # A c with no measured imbalance is refused, not modelled.
    with_gap = contract.select_clusters(
        rows=contract.ROWS,
        measured_imbalance={64: 1.5390},
        candidates=(16, 64),
    )
    assert with_gap["selected_clusters"] == 64
    missing = [
        item for item in with_gap["candidates_considered"]
        if item["clusters"] == 16
    ][0]
    assert missing["admissible"] is False
    assert "no imbalance measured" in missing["reason"]


def test_guard_refuses_a_cluster_the_device_budget_cannot_hold():
    ok = contract.guard_decision(
        rows=contract.ROWS, clusters=16, imbalance=1.1834,
        imbalance_source="measured",
    )
    assert ok["allowed"] is True
    assert ok["refused_a_priori"] is False
    # 100M at c = 16 is far past the card.
    refused = contract.guard_decision(
        rows=100_000_000, clusters=16, imbalance=1.1834,
        imbalance_source="measured",
    )
    assert refused["allowed"] is False
    assert refused["refused_a_priori"] is True
    assert refused["refusal_reasons"]


def test_guard_refuses_on_disk_as_well():
    refused = contract.guard_decision(
        rows=contract.ROWS, clusters=16, imbalance=1.1834,
        imbalance_source="measured", disk_free_bytes=1_000_000_000,
    )
    assert refused["disk_over_budget"] is True
    assert refused["allowed"] is False


def test_memmap_precondition_raises_on_an_anonymous_array(tmp_path):
    anonymous = np.zeros((8, contract.DIMENSION), dtype=np.float32)
    with pytest.raises(contract.Round0233Error) as excinfo:
        contract.assert_memmap_for_cuvs(anonymous, label="probe")
    assert "np.memmap" in str(excinfo.value)

    path = tmp_path / "block.npy"
    np.save(path, anonymous)
    writable = np.load(path, mmap_mode="r+")
    with pytest.raises(contract.Round0233Error):
        contract.assert_memmap_for_cuvs(writable, label="probe")
    del writable

    readonly = np.load(path, mmap_mode="r")
    contract.assert_memmap_for_cuvs(readonly, label="probe")


def test_signal_policy_raises_on_any_signal():
    contract.assert_no_signal_policy(["cooperative-flag"])
    with pytest.raises(contract.Round0233Error):
        contract.assert_no_signal_policy(["SIGTERM"])
    with pytest.raises(contract.Round0233Error):
        contract.assert_no_signal_policy(["cooperative-flag", "SIGKILL-last-resort"])


def test_device_law_refit_reports_range_and_residuals():
    rows = [170_504, 318_519, 533_000, 1_202_000, 2_224_000, 3_698_000]
    bytes_ = [7.47e9, 7.96e9, 8.6e9, 10.6e9, 14.0e9, 19.0e9]
    law = contract.refit_device_law(max_cluster_rows=rows, device_bytes=bytes_)
    assert law["refit"]["n_points"] == 6
    assert law["fitted_range_max_cluster_rows"] == [170_504, 3_698_000]
    assert len(law["residual_bytes"]) == 6
    assert len(law["prior_gd32_law_relative_error"]) == 6
    # The gd-32 prior under-predicts at gd 64, which is the defect R0229 left.
    assert law["prior_gd32_law_relative_error"][0] > 0.0


def test_rung_derivation_never_interpolates_and_respects_c_min():
    imbalance = {16: 1.1834, 32: 1.4237, 64: 1.5390, 200: 2.1311}
    small = contract.rung_derivation(
        rung=6_250_000, imbalance_by_c=imbalance, imbalance_source="measured",
        law_intercept_bytes=7.0e9, law_bytes_per_row=3400.0,
    )
    assert small["selected_clusters"] == 16
    big = contract.rung_derivation(
        rung=100_000_000, imbalance_by_c=imbalance, imbalance_source="measured",
        law_intercept_bytes=7.0e9, law_bytes_per_row=3400.0,
    )
    # Whatever it selects, it must never select below C_MIN and never invent a c.
    assert big["selected_clusters"] is None or big["selected_clusters"] >= 16
    for entry in big["candidates_considered"]:
        assert entry["clusters"] in imbalance


def test_io_term_uses_the_fragmented_rate_by_default():
    term = contract.io_projection(rows=contract.ROWS, substrate_passes=4)
    assert term["read_bytes_per_s"] == contract.DATA_READ_FRAGMENTED_BYTES_PER_S
    assert term["spill_write_bytes"] == contract.ROWS * contract.SPILL * 1536
    assert term["substrate_read_bytes"] == 4 * contract.ROWS * 1536
    faster = contract.io_projection(
        rows=contract.ROWS, substrate_passes=4,
        read_bytes_per_s=contract.DATA_READ_CONTIGUOUS_BYTES_PER_S,
    )
    assert faster["io_seconds"] < term["io_seconds"]


def test_scratch_peak_is_max_not_sum():
    # Review-0232's correction: pack_clusters_into_groups flushes a group before
    # admitting a cluster that would cross the budget, so peak scratch is
    # max(bound, largest single cluster), never bound + largest.
    sizes = [3_698_000] + [3_100_000] * 15
    groups = contract.pack_clusters_into_groups(sizes)
    for group in groups:
        total = sum(sizes[index] for index in group) * 1536
        assert total <= contract.SCRATCH_BUDGET_BYTES or len(group) == 1
    assert sum(len(group) for group in groups) == len(sizes)


def test_floors_and_tripwire_are_the_registered_ones():
    assert contract.RECALL_MEAN_FLOOR == 0.90
    assert contract.RECALL_P10_FLOOR == 0.80
    assert contract.MAX_ZERO_DEGREE_ROWS == 0
    assert "no seed set, no neighbour union" in contract.RECALL_POPULATION
    assert "byte-identity" in contract.DETERMINISM_NOTE


def test_nn_descent_setting_is_r0229s_adopted_arm():
    assert contract.GRAPH_DEGREE == 64
    assert contract.INTERMEDIATE_GRAPH_DEGREE == 256
    assert contract.MAX_ITERATIONS == 40
    assert contract.NN_DESCENT_SETTING == "nnd-gd64-igd256-it40"
