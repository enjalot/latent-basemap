"""R0258 contract — the polled graph-load stages, their guards, and the controls.

Every guard this round ships is exercised here with a FAILING input. A guard
whose test suite contains no failing input is untested at its only job
(`AGENT_STARTUP.md`), and the plants are run through the shipped predicate
rather than re-implemented in the test body (review-0253-01 §I).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from basemap import round0258_graph_load as gl
from basemap.round0254_dispatch import (
    SCOPE_MODULES,
    derived_entries,
    entry_install_audit,
    entry_tuples,
    gate_census,
)


@pytest.fixture(scope="module")
def weights() -> np.ndarray:
    rng = np.random.default_rng(20260811)
    return (rng.random(1_000_003, dtype=np.float64).astype(np.float32)
            + np.float32(1e-6))


@pytest.fixture(scope="module")
def endpoints() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 1_000_000, size=1_000_003).astype(np.int32)


def _null_poll(_where: str) -> None:
    return None


# --------------------------------------------------------------------------- #
# the polled stages reproduce the shipped ones BITWISE
# --------------------------------------------------------------------------- #


def test_polled_bounds_are_exact(endpoints):
    assert (gl.endpoint_bounds_polled(endpoints, poll=_null_poll, chunk=4096)
            == gl.endpoint_bounds_unpolled(endpoints))


def test_polled_validity_is_exact(weights):
    assert (gl.weight_validity_polled(weights, poll=_null_poll, chunk=4096)
            == gl.weight_validity_unpolled(weights))


def test_polled_validity_can_report_a_defect(weights):
    broken = weights.copy()
    broken[123] = np.float32(-1.0)
    broken[456] = np.float32(np.inf)
    assert (gl.weight_validity_polled(broken, poll=_null_poll, chunk=4096)
            == gl.weight_validity_unpolled(broken)
            == (False, True, True))


def test_polled_contiguous_copy_is_bitwise_identical(endpoints):
    polled = gl.contiguous_int32_polled(endpoints, poll=_null_poll, chunk=4096)
    assert gl.bitwise_identical(polled, gl.contiguous_int32_unpolled(endpoints))


def test_polled_cdf_is_bitwise_identical(weights):
    shipped, total = gl.weight_cdf_unpolled(weights)
    polled, polled_total, residue = gl.weight_cdf_polled(
        weights, poll=_null_poll, chunk=4096
    )
    assert total == polled_total
    assert gl.bitwise_identical(shipped, polled)
    assert residue >= 0.0


def test_polled_cdf_polls_at_least_once_per_chunk(weights):
    seen: list[str] = []
    gl.weight_cdf_polled(weights, poll=seen.append, chunk=4096)
    # three passes (convert, cumsum, divide) over ceil(n / chunk) chunks
    assert len(seen) == 3 * -(-len(weights) // 4096)
    assert set(seen) == {gl.ABORT_POLL_SITE_CDF}


# --------------------------------------------------------------------------- #
# the loader
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("o_direct", [False, True])
@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_polled_loader_reproduces_np_load(tmp_path, dtype, o_direct):
    path = os.path.join(str(tmp_path), "a.npy")
    array = (np.random.default_rng(3).random(500_003) * 1000).astype(dtype)
    np.save(path, array)
    seen: list[str] = []
    polled = gl.load_array_polled(
        path, poll=seen.append, chunk_bytes=1 << 20, o_direct=o_direct
    )
    assert gl.bitwise_identical(polled, gl.load_array_unpolled(path))
    assert seen and set(seen) == {gl.ABORT_POLL_SITE_LOAD}


def test_polled_loader_refuses_a_non_callable_poll(tmp_path):
    path = os.path.join(str(tmp_path), "a.npy")
    np.save(path, np.arange(16, dtype=np.int32))
    with pytest.raises(gl.Round0258GraphLoadError):
        gl.load_array_polled(path, poll=None)


def test_polled_loader_refuses_an_unaligned_chunk(tmp_path):
    path = os.path.join(str(tmp_path), "a.npy")
    np.save(path, np.arange(16, dtype=np.int32))
    with pytest.raises(gl.Round0258GraphLoadError):
        gl.load_array_polled(path, poll=_null_poll, chunk_bytes=1000)


def test_readonly_memmap_rule_is_asserted_on_the_object(tmp_path):
    path = os.path.join(str(tmp_path), "a.npy")
    np.save(path, np.arange(16, dtype=np.int32))
    array = gl.open_readonly_memmap(path, label="test")
    assert isinstance(array, np.memmap) and not array.flags.writeable


# --------------------------------------------------------------------------- #
# positive controls -- five structural plants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(gl.STRUCTURAL_DEFECTS))
def test_each_structural_plant_is_refused_by_the_shipped_guard(name):
    report = gl.chunk_loop_polls(gl.STRUCTURAL_DEFECTS[name])
    assert not report["the_chunk_loop_polls"], name


@pytest.mark.parametrize("name", sorted(gl.STRUCTURAL_DEFECTS))
def test_each_structural_plant_is_accepted_by_the_weaker_predicate(name):
    # The contrast is what proves each refusal is a behaviour CHANGE rather than
    # a guard that refuses everything (the R0254 standard).
    assert "poll(" in gl.STRUCTURAL_DEFECTS[name]


def test_structural_controls_assert_clean():
    report = gl.assert_structural_defect_controls()
    assert report["defects_refused_by_the_shipped_guard"] == len(
        gl.STRUCTURAL_DEFECTS
    )
    assert report["an_honest_install_passes_both"]


def test_every_shipped_polled_stage_polls_inside_its_chunk_loop():
    audit = gl.assert_chunk_loops_poll()
    assert audit["functions_checked"] == len(gl.POLLED_FUNCTIONS)
    assert all(
        report["the_chunk_loop_polls"] for report in audit["by_function"].values()
    )


def test_the_chunk_loop_audit_fails_when_a_shipped_stage_is_broken(monkeypatch):
    """The audit must be capable of failing on the shipped set, not only on a
    planted string. Swap one shipped stage for a poll-free one and require it."""

    def unpolled_stage(array, *, poll, chunk):
        out = np.empty(len(array))
        for start in range(0, len(array), chunk):
            out[start:start + chunk] = array[start:start + chunk]
        return out

    monkeypatch.setattr(
        gl, "POLLED_FUNCTIONS", gl.POLLED_FUNCTIONS + (unpolled_stage,)
    )
    with pytest.raises(gl.Round0258GraphLoadError):
        gl.assert_chunk_loops_poll()


# --------------------------------------------------------------------------- #
# positive controls -- five numeric plants
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", sorted(gl.NUMERIC_DEFECTS))
def test_each_numeric_plant_is_refused_by_the_bitwise_comparison(name, weights):
    shipped, _total = gl.weight_cdf_unpolled(weights)
    planted = gl.NUMERIC_DEFECTS[name](weights, chunk=4096)
    assert not gl.bitwise_identical(shipped, planted), name


def test_at_least_one_numeric_plant_survives_an_allclose_comparison():
    report = gl.bitwise_identity_controls()
    assert report["defects_accepted_by_an_allclose_comparison"] >= 1
    assert report["planted"]["carry_added_after_the_chunk"][
        "accepted_by_an_allclose_comparison"
    ]


def test_numeric_controls_assert_clean():
    report = gl.assert_bitwise_identity_controls()
    assert report["the_polled_path_is_bitwise_identical"]
    assert report["defects_refused_by_the_shipped_comparison"] == len(
        gl.NUMERIC_DEFECTS
    )
    assert report["tail_chunk_exercised"]


# --------------------------------------------------------------------------- #
# the install-without-gate gap this round closes
# --------------------------------------------------------------------------- #


def test_run_train_and_run_assemble_now_both_install_and_gate():
    entries = entry_tuples(derived_entries(SCOPE_MODULES))
    gates = gate_census(entries)
    both = {
        row["entry"] for row in gates["entries"]
        if row["constructs_a_gate"] and row["installs_effectively"]
    }
    assert "experiments.round0113_nodes.run_train" in both
    assert "experiments.round0238_nodes.run_assemble" in both
    assert gates["entries_that_both_install_and_gate"] == len(both)


def test_every_derived_entry_still_installs_effectively():
    entries = entry_tuples(derived_entries(SCOPE_MODULES))
    audit = entry_install_audit(entries)
    assert audit["every_entry_installs_effectively"], (
        audit["entries_without_an_effective_install"]
    )


def test_round0258_nodes_is_in_scope():
    assert "experiments.round0258_nodes" in SCOPE_MODULES


# --------------------------------------------------------------------------- #
# the artifact this round measures
# --------------------------------------------------------------------------- #


def test_the_registered_rung_is_r0243s():
    assert gl.ROWS == 100_000_000
    assert gl.K == 15
    assert gl.DIRECTED_EDGES == 2_511_103_254
    assert set(gl.EDGE_ARRAYS) == {"sources", "targets", "weights"}
    for spec in gl.EDGE_ARRAYS.values():
        assert spec["bytes"] == 10_044_413_144
        assert len(spec["sha256"]) == 64
