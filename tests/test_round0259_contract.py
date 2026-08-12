"""R0259 contract tests — the loader branch, the rung entry, the chunked sum.

Every guard this round ships has a positive control here that plants the defect
and is verified to fail against the **shipped** path. Nothing in this file needs
a GPU or the 100M artifacts except the two tests marked `real_artifacts`, which
skip when they are absent.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from basemap import round0259_hundred_m as rung
from basemap.round0258_graph_load import chunk_loop_polls
from basemap.pumap.parametric_umap.datasets.edge_list_dataset import load_edge_arrays


# --------------------------------------------------------------------------- #
# I1-I4: the rung registry
# --------------------------------------------------------------------------- #


def test_rung_applicability_holds_in_both_directions():
    report = rung.assert_rung_applicability_controls()
    assert report["manifests_that_behave_as_expected"] == report["manifests_checked"]
    assert report["manifests_the_100m_entry_accepts"] == ["the_real_100m_manifest"]


def test_a_2m_config_cannot_enter_the_100m_entry():
    two_m = {
        "schema": rung.RUNGS[rung.RUNG_2M]["schema"],
        "retained_rows": 1_993_761, "k": 50, "dimension": 768,
    }
    with pytest.raises(rung.Round0259RungError):
        rung.assert_rung(two_m, expected=rung.RUNG_100M, entry="run_train_100m")
    assert rung.rung_of(two_m) == rung.RUNG_2M


def test_a_matching_schema_with_the_wrong_rows_raises_rather_than_guessing():
    manifest = {
        "schema": rung.RUNGS[rung.RUNG_100M]["schema"], "rows": 1_993_761, "k": 15,
    }
    with pytest.raises(rung.Round0259RungError):
        rung.rung_of(manifest)


# --------------------------------------------------------------------------- #
# I5: the container branch
# --------------------------------------------------------------------------- #


def test_the_bulk_npz_path_is_unchanged(tmp_path):
    report = rung.bulk_npz_is_not_claimed(str(tmp_path / "bulk"))
    assert report["the_bulk_path_is_unchanged"]
    assert report["the_branch_claims_it"] is False


def test_wrong_shape_streamed_containers_fail_loudly(tmp_path):
    report = rung.assert_container_defect_controls(str(tmp_path / "containers"))
    assert report["defects_that_failed_loudly"] == report["defects_planted"] == 5
    assert report["an_honest_streamed_graph_loads"]
    assert report["an_honest_streamed_graph_returns_memmaps"]


def test_the_branch_returns_memmaps_and_never_materialises(tmp_path):
    directory = tmp_path / "graph"
    directory.mkdir()
    edges = 1024
    np.save(directory / rung.MEMBER_FILENAMES["sources"],
            np.arange(edges, dtype=np.int32))
    np.save(directory / rung.MEMBER_FILENAMES["targets"],
            (np.arange(edges, dtype=np.int32) + 1) % edges)
    np.save(directory / rung.MEMBER_FILENAMES["weights"],
            np.full(edges, 0.5, dtype=np.float32))
    np.savez(directory / rung.MEMBER_FILENAMES["header"],
             n_nodes=np.int64(edges), k=np.int64(15),
             directed_edges=np.int64(edges))
    sources, targets, weights, n_nodes = load_edge_arrays(str(directory))
    assert int(n_nodes) == edges
    assert rung.file_backed(sources) and rung.file_backed(targets)
    assert rung.file_backed(weights)
    sources_only, _, none_weights, _ = load_edge_arrays(
        str(directory), load_weights=False
    )
    assert none_weights is None
    assert int(sources_only.shape[0]) == edges


# --------------------------------------------------------------------------- #
# residency: review-0258-01 §H.3
# --------------------------------------------------------------------------- #


def test_the_isinstance_rule_misclassifies_an_ascontiguousarray_view(tmp_path):
    path = tmp_path / "member.npy"
    np.save(path, np.arange(64, dtype=np.int32))
    mapped = np.load(path, mmap_mode="r")
    view = np.ascontiguousarray(np.asarray(mapped), dtype=np.int32)
    # This is `HostStreamEdgeSampler._src_h` exactly.
    assert rung.file_backed(view) is True
    assert rung.r0230_isinstance_verdict(view) is False
    report = rung.endpoint_residency(view, label="_src_h")
    assert report["the_two_rules_agree"] is False
    with pytest.raises(rung.Round0259Error):
        rung.assert_resident(view, label="_src_h")
    resident = np.array(mapped, dtype=np.int32)
    assert rung.file_backed(resident) is False
    rung.assert_resident(resident, label="resident")


# --------------------------------------------------------------------------- #
# the chunked pairwise sum
# --------------------------------------------------------------------------- #


def test_the_pairwise_rule_reproduces_np_sum_bitwise():
    report = rung.assert_pairwise_rule()
    assert report["every_check_bitwise_identical"]
    assert report["checks_run"] >= 60
    assert report["numpy_version_observed"] == report["numpy_version_verified"], (
        "the pairwise split rule is a numpy implementation detail; a version "
        "change must be looked at, not assumed"
    )


def test_five_planted_sums_are_refused_bitwise_and_accepted_by_allclose():
    report = rung.pairwise_sum_controls()
    assert report["the_polled_sum_is_bitwise_identical"]
    assert report["defects_refused_by_the_bitwise_comparison"] == 5
    assert report["defects_accepted_by_an_allclose_comparison"] >= 2


def test_the_pairwise_sum_refuses_float32_and_a_tiny_leaf():
    values = np.random.default_rng(0).random(4096).astype(np.float32)
    with pytest.raises(rung.Round0259Error):
        rung.pairwise_sum_polled(values, poll=lambda _w: None)
    with pytest.raises(rung.Round0259Error):
        rung.pairwise_sum_polled(
            values.astype(np.float64), poll=lambda _w: None, leaf=64
        )


def test_the_fully_polled_cdf_matches_the_shipped_one_bitwise():
    from basemap.round0258_graph_load import weight_cdf_unpolled

    rng = np.random.default_rng(20260812)
    weights = (rng.random(300_017).astype(np.float32) + np.float32(1e-6))
    shipped, shipped_total = weight_cdf_unpolled(weights)

    seen: list[str] = []

    def poll(where):
        seen.append(where)

    ours, total, timings = rung.weight_cdf_fully_polled(
        weights, poll=poll, chunk=65_536, leaf=1 << 13
    )
    assert total == shipped_total
    assert np.array_equal(ours.view(np.uint8), shipped.view(np.uint8))
    assert len(seen) > 3 * (300_017 // 65_536)
    assert timings["sum_leaves"] >= 1


# --------------------------------------------------------------------------- #
# the strengthened chunk-loop guard
# --------------------------------------------------------------------------- #


def test_every_plant_passes_the_shipped_guard_and_fails_the_new_one():
    report = rung.assert_structural_defect_controls_v2()
    assert report["defects_planted"] == 6
    assert report["defects_accepted_by_the_shipped_r0258_guard"] == 6
    assert report["defects_refused_by_the_r0259_guard"] == 6
    assert report["an_honest_install_passes_both_guards"]
    assert report["an_honest_poll_is_effective"]
    assert report["chunk_bound_refuses_a_whole_array_chunk"]


@pytest.mark.parametrize("name", sorted(rung.STRUCTURAL_DEFECTS_V2))
def test_each_static_plant_individually(name):
    source = rung.STRUCTURAL_DEFECTS_V2[name]
    assert chunk_loop_polls(source)["the_chunk_loop_polls"] is True
    assert rung.chunk_loop_polls_v2(source)["the_chunk_loop_polls"] is False


def test_a_noop_poll_is_refused_at_runtime():
    assert rung.poll_is_effective(lambda where: None) is False
    assert rung.poll_is_effective(None) is False

    def real_poll(where):
        real_poll.seen = where

    assert rung.poll_is_effective(real_poll) is True
    with pytest.raises(rung.Round0259Error):
        rung.assert_poll_is_effective(lambda where: None, label="probe")


def test_every_shipped_polled_stage_passes_the_strengthened_guard():
    report = rung.assert_chunk_loops_poll_v2()
    assert report["functions_checked"] == report["functions_that_pass"] == 5


def test_the_chunk_bound_refuses_a_whole_array_chunk():
    with pytest.raises(rung.Round0259Error):
        rung.assert_chunk_bounded(2_511_103_254, 2_511_103_254)
    with pytest.raises(rung.Round0259Error):
        rung.assert_chunk_bounded(1024, 0)
    rung.assert_chunk_bounded(2_511_103_254, rung.MAX_CHUNK_ELEMENTS)


# --------------------------------------------------------------------------- #
# the real artifacts, when they are here
# --------------------------------------------------------------------------- #

_REAL = os.path.isfile(rung.R0243_FUZZY_MANIFEST)
real_artifacts = pytest.mark.skipif(not _REAL, reason="the 100M graph is absent")


@real_artifacts
def test_the_branch_opens_the_real_100m_graph_three_ways():
    directory = os.path.dirname(rung.R0243_FUZZY_MANIFEST)
    header = os.path.join(directory, rung.MEMBER_FILENAMES["header"])
    for path in (directory, header):
        sources, targets, weights, n_nodes = load_edge_arrays(path)
        assert int(n_nodes) == 100_000_000
        assert int(sources.shape[0]) == 2_511_103_254
        assert str(sources.dtype) == "int32" and str(weights.dtype) == "float32"
        assert rung.file_backed(sources)
        del sources, targets, weights


@real_artifacts
def test_the_2m_entry_still_refuses_the_real_100m_manifest():
    from basemap import round0113_prompt_contrast as r0113
    from basemap.artifact_identity import expected_input_signature

    signature = expected_input_signature(rung.R0243_FUZZY_MANIFEST)
    with pytest.raises(r0113.Round0113Error):
        r0113.load_graph(
            rung.R0243_FUZZY_MANIFEST,
            expected_sha256=signature["sha256"],
            arm=sorted(r0113.ARMS)[0],
        )


@real_artifacts
def test_the_substrate_carries_the_rungs_dimension():
    report = rung.assert_substrate_dimension()
    assert report["dimension"] == 384
    assert report["shape"] == [100_000_000, 384]
    assert report["bytes"] == rung.SUBSTRATE_100M_BYTES
