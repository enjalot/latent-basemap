"""R0262 contract tests.

Every guard this round adds ships a **positive control**: a test that plants the
defect and fails if the shipped path stops catching it. review-0260 downgraded
R0260 partly because its ordering guards shipped no positive control while 19
tests covered lesser claims, so the tests that matter here are the ones marked
`positive control` — they break the shipped code and require the break to be
seen.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from basemap.round0034_pipeline import HostInt8MaterializedArray
from basemap.round0262_host_int8_adapter import (
    INT8_DIVISOR,
    Round0262BudgetError,
    Round0262WiringError,
    assert_buffer_admits_batch,
    classify_host_backing,
    dequantise_block,
    host_memory_accounting,
    quantise_block_int8,
    quantise_substrate_to_int8,
)

torch = pytest.importorskip("torch")
CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="requires CUDA")


# --------------------------------------------------------------------------- #
# 1. the backing rule -- positive controls
# --------------------------------------------------------------------------- #

def test_ascontiguousarray_of_a_memmap_is_a_no_op_and_the_base_chain_sees_it(tmp_path):
    """POSITIVE CONTROL for the mandate's third named defect.

    Plants exactly what `HostStreamEdgeSampler.__init__` does. If
    `classify_host_backing` ever stopped walking `.base`, this test fails —
    which is the only way to know the rule is a rule.
    """
    path = str(tmp_path / "endpoints.i32.npy")
    np.save(path, np.arange(100_000, dtype=np.int32))
    backing = np.load(path, mmap_mode="r")
    planted = np.ascontiguousarray(np.asarray(backing), dtype=np.int32)

    assert np.shares_memory(planted, backing), "the plant failed: it copied"
    verdict = classify_host_backing(planted)
    assert verdict["file_backed"] is True
    assert verdict["backing"] == "file_backed_page_cache"
    assert verdict["base_chain"][-1] == "mmap"
    # The defect: the isinstance reader rule calls it host-resident.
    assert verdict["isinstance_np_memmap"] is False
    assert verdict["isinstance_rule_agrees_with_base_chain"] is False


def test_the_backing_rule_is_not_a_constant():
    """An anonymous array must be classified anonymous by BOTH rules.

    Without this, a `classify_host_backing` that always returned "file_backed"
    would pass the test above.
    """
    verdict = classify_host_backing(np.empty(100_000, dtype=np.int32))
    assert verdict["file_backed"] is False
    assert verdict["backing"] == "anonymous"
    assert verdict["isinstance_rule_agrees_with_base_chain"] is True


def test_map_shared_moves_rss_shmem_and_not_rss_anon():
    """POSITIVE CONTROL: MAP_SHARED is invisible to an anonymous-bytes guard."""
    import mmap as _mmap

    size = 256 * (1 << 20)
    before = host_memory_accounting()
    region = _mmap.mmap(-1, size, flags=_mmap.MAP_SHARED | _mmap.MAP_ANONYMOUS)
    view = np.frombuffer(region, dtype=np.uint8)
    view[::4096] = 1
    during = host_memory_accounting()
    del view
    region.close()

    assert during["RssShmem"] - before["RssShmem"] >= size // 2
    assert during["RssAnon"] - before["RssAnon"] < size // 2


# --------------------------------------------------------------------------- #
# 2. the int8 encoder
# --------------------------------------------------------------------------- #

def test_the_encoder_round_trips_within_its_own_quantisation_step():
    rng = np.random.default_rng(0)
    rows = rng.normal(size=(4096, 384)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    encoded, scales = quantise_block_int8(rows)

    assert encoded.dtype == np.int8 and scales.dtype == np.float16
    assert int(np.abs(encoded).max()) <= 127
    # Every component lands within half a quantisation step of its source.
    error = np.abs(dequantise_block(encoded, scales) - rows)
    assert (error <= 0.5001 * scales.astype(np.float32)[:, None]).all()


def test_the_encoder_never_emits_minus_128():
    """-128 has no positive counterpart; a symmetric code must not produce it."""
    rows = np.full((16, 384), -1.0, dtype=np.float32)
    encoded, _ = quantise_block_int8(rows)
    assert int(encoded.min()) == -127


def test_an_all_zero_row_gets_a_positive_scale_and_dequantises_to_zero():
    """HostInt8MaterializedArray refuses scales <= 0, so this must not be 0."""
    rows = np.zeros((4, 384), dtype=np.float32)
    encoded, scales = quantise_block_int8(rows)
    assert (scales > 0).all()
    assert np.all(dequantise_block(encoded, scales) == 0.0)
    # And the array it feeds must accept it.
    HostInt8MaterializedArray(encoded, scales, device="cpu", buffer_rows=4)


def test_the_quantiser_refuses_a_substrate_of_the_wrong_shape(tmp_path):
    source = str(tmp_path / "wrong.npy")
    np.save(source, np.zeros((10, 7), dtype=np.float32))
    with pytest.raises(Round0262WiringError):
        quantise_substrate_to_int8(
            source_path=source, expected_shape=(10, 384),
            int8_path=str(tmp_path / "o.i8"),
            scales_path=str(tmp_path / "o.f16"), rows_per_chunk=4)


def test_the_quantiser_polls_between_chunks(tmp_path):
    source = str(tmp_path / "s.npy")
    np.save(source, np.ones((40, 384), dtype=np.float32))
    seen: list[str] = []
    receipt = quantise_substrate_to_int8(
        source_path=source, expected_shape=(40, 384),
        int8_path=str(tmp_path / "o.i8"), scales_path=str(tmp_path / "o.f16"),
        rows_per_chunk=8, poll=seen.append)
    assert receipt["chunks"] == 5
    assert len(seen) == 5
    assert receipt["int8_bytes"] == 40 * 384
    assert receipt["scales_bytes"] == 40 * 2
    assert receipt["fidelity_over_every_row"]["rows"] == 40


def test_an_aborting_poll_stops_the_quantiser_between_chunks(tmp_path):
    """POSITIVE CONTROL: the abort flag must be honoured on the encode stage."""
    source = str(tmp_path / "s.npy")
    np.save(source, np.ones((40, 384), dtype=np.float32))

    calls = {"n": 0}

    def poll(_where):
        calls["n"] += 1
        if calls["n"] == 3:
            raise KeyboardInterrupt("cooperative abort")

    with pytest.raises(KeyboardInterrupt):
        quantise_substrate_to_int8(
            source_path=source, expected_shape=(40, 384),
            int8_path=str(tmp_path / "o.i8"),
            scales_path=str(tmp_path / "o.f16"),
            rows_per_chunk=8, poll=poll)
    assert calls["n"] == 3, "the stage ran past the abort"


# --------------------------------------------------------------------------- #
# 3. the buffer guard -- positive control
# --------------------------------------------------------------------------- #

def test_a_buffer_narrower_than_the_batch_is_refused_at_construction():
    """POSITIVE CONTROL. Without this the fit raises on its FIRST update.

    `HostInt8MaterializedArray._validated_pair_rows` refuses a gather wider than
    `buffer_rows`, and `HostStreamEdgeSampler` gathers `batch_size` rows per
    endpoint. This plants exactly that mismatch.
    """
    rows = np.zeros((64, 384), dtype=np.float32)
    encoded, scales = quantise_block_int8(rows)
    narrow = HostInt8MaterializedArray(
        encoded, scales, device="cpu", buffer_rows=32)

    with pytest.raises(Round0262WiringError):
        assert_buffer_admits_batch(narrow, 64)
    assert assert_buffer_admits_batch(narrow, 32)["admits"] is True

    # And prove the refusal is not decorative: the underlying gather really
    # does fail at that width.
    from basemap.round0034_pipeline import Round0034PipelineError
    with pytest.raises(Round0034PipelineError):
        narrow.gather_pairs(np.arange(64), np.arange(64))


def test_the_headroom_guard_refuses_an_allocation_larger_than_the_host():
    from basemap.round0262_host_int8_adapter import assert_host_headroom

    with pytest.raises(Round0262BudgetError):
        assert_host_headroom(1 << 60, margin_bytes=0, where="test")


# --------------------------------------------------------------------------- #
# 4. the wiring itself -- the round's central claim
# --------------------------------------------------------------------------- #

def _tiny_graph(tmp_path, n_nodes=4096, k=8):
    rng = np.random.default_rng(3)
    src = np.repeat(np.arange(n_nodes, dtype=np.int32), k)
    dst = rng.integers(0, n_nodes, size=n_nodes * k).astype(np.int32)
    wts = (rng.random(n_nodes * k) + 1e-3).astype(np.float32)
    path = str(tmp_path / "edges.npz")
    np.savez(path, sources=src, targets=dst, weights=wts,
             n_nodes=np.int64(n_nodes))
    return path, n_nodes


def _estimator(batch_size, **kwargs):
    from basemap.pumap.parametric_umap.core import ParametricUMAP

    estimator = ParametricUMAP(
        n_components=2, batch_size=batch_size, pos_ratio=0.05,
        weighted_edge_sampling=True, positive_target_mode="binary",
        device="cuda", n_epochs=1, **kwargs)
    estimator._allow_model_before_admission = True
    return estimator


@needs_cuda
def test_a_host_int8_x_selects_the_weighted_host_int8_pipeline(tmp_path):
    """The round's central claim, checked through the SHIPPED entry point."""
    edges, n_nodes = _tiny_graph(tmp_path)
    rng = np.random.default_rng(1)
    rows = rng.normal(size=(n_nodes, 384)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    encoded, scales = quantise_block_int8(rows)
    batch_size = 512
    host_x = HostInt8MaterializedArray(
        encoded, scales, device="cuda", buffer_rows=batch_size)

    estimator = _estimator(batch_size)
    dataset, loader, _ = estimator._prepare_edge_list_training(
        host_x, edges, n_nodes, low_memory=False, random_state=0)
    try:
        info = estimator._pipeline_info
        assert info["pipeline"] == "host_int8_hybrid"
        assert info["sampler_class"] == "HostStreamEdgeSampler"
        assert info["positive_sampling"] == "weighted_with_replacement"
        assert info["weighted_requested"] is True
        assert info["weighted_effective"] is True
        assert info["x_residency"] == "host_int8_plus_exact_fp16_row_scales"
        assert dataset is host_x
        assert type(loader).__name__ == "HostStreamEdgeSampler"

        src_feats, dst_feats, targets = next(iter(loader))
        assert src_feats.shape == (batch_size, 384)
        assert dst_feats.shape == (batch_size, 384)
        assert targets.shape == (batch_size,)
        assert src_feats.dtype == torch.float32
        assert src_feats.is_cuda
        # A full update gathers 2 x batch_size feature rows.
        assert src_feats.shape[0] + dst_feats.shape[0] == 2 * batch_size
    finally:
        loader.close()


@needs_cuda
def test_the_fused_pair_gather_returns_the_rows_that_were_asked_for(tmp_path):
    """POSITIVE CONTROL against a src/dst swap in the fused branch.

    The new `__next__` branch gathers both endpoints from one slot. A wiring bug
    that returned the source rows for both endpoints, or swapped them, would be
    invisible in timing and in shape. This asks for known rows and checks the
    values.
    """
    rng = np.random.default_rng(5)
    rows = rng.normal(size=(2048, 384)).astype(np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    encoded, scales = quantise_block_int8(rows)
    host_x = HostInt8MaterializedArray(
        encoded, scales, device="cuda", buffer_rows=256)
    expected = dequantise_block(encoded, scales)

    src_rows = rng.integers(0, 2048, 256)
    dst_rows = rng.integers(0, 2048, 256)
    got_src, got_dst = host_x.gather_pairs(src_rows, dst_rows)

    assert np.allclose(got_src.cpu().numpy(), expected[src_rows], atol=1e-5)
    assert np.allclose(got_dst.cpu().numpy(), expected[dst_rows], atol=1e-5)
    # And they are genuinely different gathers, not the same one twice.
    assert not np.allclose(expected[src_rows], expected[dst_rows])


@needs_cuda
def test_the_weighted_uniform_downgrade_gate_still_raises_on_the_legacy_path(tmp_path):
    """POSITIVE CONTROL: core.py:621 must keep refusing the silent downgrade.

    R0262 adds a weighted host path; it must not have opened a second door for
    the legacy uniform sampler.
    """
    edges, n_nodes = _tiny_graph(tmp_path)
    rows = np.random.default_rng(2).normal(size=(n_nodes, 384)).astype(np.float32)
    path = str(tmp_path / "x.npy")
    np.save(path, rows)
    memmapped = np.load(path, mmap_mode="r")

    estimator = _estimator(512, gpu_resident_data=False)
    with pytest.raises(RuntimeError, match="samples UNIFORMLY"):
        estimator._prepare_edge_list_training(
            memmapped, edges, n_nodes, low_memory=True, random_state=0)


@needs_cuda
def test_a_host_int8_x_with_a_batch_wider_than_its_buffer_is_refused(tmp_path):
    """POSITIVE CONTROL: the guard fires through the shipped entry, not just
    through the helper."""
    edges, n_nodes = _tiny_graph(tmp_path)
    rows = np.random.default_rng(4).normal(size=(n_nodes, 384)).astype(np.float32)
    encoded, scales = quantise_block_int8(rows)
    host_x = HostInt8MaterializedArray(
        encoded, scales, device="cuda", buffer_rows=128)

    estimator = _estimator(512)
    with pytest.raises(RuntimeError, match="rows per"):
        estimator._prepare_edge_list_training(
            host_x, edges, n_nodes, low_memory=False, random_state=0)


@needs_cuda
def test_the_device_path_is_unchanged_by_the_new_branch(tmp_path):
    """No existing rung may change pipeline, sampler or residency."""
    edges, n_nodes = _tiny_graph(tmp_path)
    rows = np.random.default_rng(6).normal(size=(n_nodes, 384)).astype(np.float32)

    estimator = _estimator(512)
    dataset, loader, _ = estimator._prepare_edge_list_training(
        rows, edges, n_nodes, low_memory=False, random_state=0)
    info = estimator._pipeline_info
    assert info["pipeline"] in ("device", "device_uniform")
    assert info["sampler_class"] == "DeviceEdgeSampler"
    assert info["x_residency"] == "device_fp16"
    assert type(dataset).__name__ == "DeviceArrayDataset"


def test_a_non_cuda_device_refuses_a_host_int8_x(tmp_path):
    """The dequantisation runs on the training device; CPU is refused loudly."""
    edges, n_nodes = _tiny_graph(tmp_path)
    rows = np.random.default_rng(8).normal(size=(n_nodes, 384)).astype(np.float32)
    encoded, scales = quantise_block_int8(rows)
    host_x = HostInt8MaterializedArray(
        encoded, scales, device="cpu", buffer_rows=512)

    from basemap.pumap.parametric_umap.core import ParametricUMAP

    estimator = ParametricUMAP(
        n_components=2, batch_size=512, pos_ratio=0.05,
        weighted_edge_sampling=True, device="cpu", n_epochs=1)
    estimator._allow_model_before_admission = True
    with pytest.raises(RuntimeError, match="CUDA device"):
        estimator._prepare_edge_list_training(
            host_x, edges, n_nodes, low_memory=False, random_state=0)


# --------------------------------------------------------------------------- #
# 5. the numpy pin, closed
# --------------------------------------------------------------------------- #

def test_the_pairwise_rule_runs_and_holds_on_this_numpy():
    """R0259 recorded the version and never ran this in its 100M node."""
    from basemap.round0259_hundred_m import assert_pairwise_rule

    receipt = assert_pairwise_rule(sizes=(1_000_003, 262_144), leaves=(1 << 20,))
    assert receipt["every_check_bitwise_identical"] is True
    assert receipt["numpy_version_observed"] == np.__version__
    assert receipt["numpy_version_matches_the_verified_one"] is True
    assert receipt["checks_run"] >= 6
