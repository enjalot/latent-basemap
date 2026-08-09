"""R0233 CUDA-hidden CPU smoke — reach the real code paths before the GPU does.

Covers the selection law on a synthetic corpus (including the prefix defect the
span assertion exists to catch), the reserve carve, R0216's fuzzy law applied to
a MEMMAPPED substrate exactly as the qualification node applies it, the recall
instruments, and the build child's fail-closed argument checks.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from basemap import round0233_substrate as contract
from basemap.round0220_cuvs_qualification import (
    graph_validity,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from experiments import round0233_nodes as nodes


def _shard(path: str, rows: int, seed: int, *, zero_rows: int = 0) -> None:
    rng = np.random.default_rng(seed)
    block = rng.standard_normal((rows, contract.DIMENSION)).astype(np.float32)
    if zero_rows:
        block[:zero_rows] = 0.0
    np.save(path, block)


def test_draw_spans_every_shard_and_replaces_degenerate_rows(tmp_path):
    shards = []
    for index in range(6):
        path = str(tmp_path / f"shard-{index}.npy")
        _shard(path, 4_000, seed=index, zero_rows=200 if index == 2 else 0)
        shards.append((path, 4_000, True))
    offsets = np.concatenate([[0], np.cumsum([rows for _p, rows, _n in shards])])
    offsets = offsets.astype(np.int64)
    total = int(offsets[-1])
    picked = np.zeros(total, dtype=bool)
    rng = np.random.RandomState(233)

    selected, vectors, dropped, rounds = nodes._draw(
        shards=shards, offsets=offsets, picked=picked, rng=rng,
        want=6_000, corpus="synthetic",
    )
    assert selected.size == 6_000
    assert vectors.shape == (6_000, contract.DIMENSION)
    assert dropped > 0            # the zeroed rows were rejected, not accepted
    assert rounds >= 2            # and replaced from the unpicked complement
    assert np.all(np.diff(selected) > 0)
    shard_of = np.searchsorted(offsets, selected, side="right") - 1
    touched = int(np.unique(shard_of).size)
    assert contract.validate_shard_span(
        corpus="synthetic", shards_touched=touched, shards_total=len(shards)
    )["coverage"] == 1.0
    norms = np.linalg.norm(vectors, axis=1)
    assert float(norms.min()) > 0
    assert np.isfinite(vectors).all()

    # The reserve draw comes from the complement of the training picks.
    reserve_rng = np.random.RandomState(233 + 500)
    reserve, _rv, _rd, _rr = nodes._draw(
        shards=shards, offsets=offsets, picked=picked, rng=reserve_rng,
        want=2_000, corpus="synthetic[reserve]",
    )
    assert np.intersect1d(selected, reserve).size == 0


def test_a_prefix_selection_is_caught_by_the_registered_assertion():
    # Exactly R0216's executed defect: a leading prefix of the corpus.
    with pytest.raises(contract.Round0233Error):
        contract.validate_shard_span(
            corpus="fineweb", shards_touched=92, shards_total=98
        )


def test_fuzzy_law_accepts_a_memmapped_substrate(tmp_path):
    """The qualification node hands `fuzzy_simplicial_set` a memmap, not a copy.

    A 6.25M x 384 anonymous copy would be 9.6 GB of swappable memory for no
    reason: with knn arrays supplied, only `X.shape[0]` is read.
    """
    import umap.umap_ as umap_api

    rows, k = 512, contract.GRAPH_K
    rng = np.random.default_rng(0)
    block = rng.standard_normal((rows, 16)).astype(np.float32)
    block /= np.linalg.norm(block, axis=1, keepdims=True)
    path = str(tmp_path / "substrate.f32.npy")
    np.save(path, block)
    host = np.load(path, mmap_mode="r")
    assert isinstance(host, np.memmap)

    cosines = block @ block.T
    np.fill_diagonal(cosines, -np.inf)
    order = np.argsort(-cosines, axis=1)[:, :k]
    ids = order.astype(np.int32)
    dists = np.maximum(
        1.0 - np.take_along_axis(cosines, order, axis=1), 0.0
    ).astype(np.float32)

    graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
        host, n_neighbors=k,
        random_state=np.random.RandomState(contract.FUZZY_RANDOM_STATE_SEED),
        metric="cosine", knn_indices=ids, knn_dists=dists,
    )
    coo = graph.tocoo()
    src = np.asarray(coo.row, dtype=np.int32)
    weights = np.asarray(coo.data, dtype=np.float32)
    assert np.isfinite(weights).all()
    assert weights.min() > 0 and weights.max() <= 1
    degree = np.bincount(src, minlength=rows)
    assert int((degree == 0).sum()) == contract.MAX_ZERO_DEGREE_ROWS


def test_recall_instruments_agree_with_a_perfect_and_a_damaged_graph():
    rows, k = 256, contract.GRAPH_K
    rng = np.random.default_rng(1)
    truth = np.stack([
        rng.choice(rows, size=k, replace=False) for _ in range(rows)
    ]).astype(np.int32)
    truth_cos = np.sort(
        rng.uniform(0.2, 0.9, size=(rows, k)).astype(np.float32), axis=1
    )[:, ::-1].copy()
    kth = truth_cos[:, k - 1].astype(np.float64)

    perfect = strict_containment_rows(truth, truth)
    assert float(perfect.mean()) == 1.0
    assert summarize(perfect, label="perfect")["p10"] == 1.0

    damaged = truth.copy()
    damaged[:, 0] = (damaged[:, 0] + 1) % rows
    hurt = strict_containment_rows(damaged, truth)
    assert float(hurt.mean()) < 1.0

    tie = tie_aware_rows(truth_cos.astype(np.float64), truth, kth)
    assert float(tie.mean()) == 1.0

    structural = graph_validity(truth, rows=rows)
    assert structural["out_of_range_entries"] == 0
    assert "zero_degree_rows" in structural


def test_build_child_imports_and_refuses_a_degenerate_configuration():
    """CUDA-hidden import plus the argument checks that run before any device work."""
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
    from basemap import round0233_build

    assert round0233_build.ABORT_FLAG_NAME == "abort.flag"
    assert callable(round0233_build.main)
    assert callable(round0233_build._check_abort)
    # The in-band abort path raises rather than depending on a signal handler.
    with pytest.raises(Exception) as excinfo:
        round0233_build._check_abort(__file__, where="smoke")
    assert "cooperative abort flag" in str(excinfo.value)
    round0233_build._check_abort(None, where="smoke")
    round0233_build._check_abort("/nonexistent/flag", where="smoke")
    assert round0233_build._anon_bytes() > 0


def test_watchdog_writes_a_flag_and_never_signals(tmp_path):
    flag = str(tmp_path / "abort.flag")
    watchdog = nodes.FlagWatchdog(
        flag_path=flag, pid=os.getpid(), poll_s=10.0,
        host_anon_budget_bytes=1 << 62, swap_growth_abort_bytes=1 << 62,
        device_baseline_bytes=0, swap_baseline_bytes=0,
    )
    watchdog._trip("smoke")
    assert os.path.exists(flag)
    readings = watchdog.readings()
    assert readings["watchdog_escalations"] == ["cooperative-flag"]
    contract.assert_no_signal_policy(readings["watchdog_escalations"])


def test_node_actions_are_exhaustive_and_refuse_anything_else():
    with pytest.raises(contract.Round0233Error):
        nodes.run_job({"manifest": {"round_id": "0233"}}, {"action": "train"})
