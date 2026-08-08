"""CUDA-hidden CPU smoke for R0220.

Reaches the code paths that only execute at the *end* of a GPU node — the exact
top-k merge with self-exclusion, the CSR view of R0216's edge file, the receipt
seal, and JSON serialisation of every measurement dict — so a late NameError or
a numpy scalar that will not serialise is caught in milliseconds instead of
after the exact 2M search.
"""
from __future__ import annotations

import json
import os
import textwrap

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.round0220_cuvs_build import _drop_self
from basemap.round0220_cuvs_qualification import (
    CAPABILITY,
    CUVS_METRIC,
    GRAPH_K,
    PROJECTION_ROWS,
    QUALIFICATION_SCHEMA,
    ROUND_ID,
    Round0220Error,
    SCALING_SETTING_ID,
    SWEEP,
    TRUTH_SCHEMA,
    graph_validity,
    power_law,
    project_cost,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from experiments.round0220_nodes import (
    QUALIFY_ACTION,
    TRUTH_ACTION,
    _child_environment,
    _csr_bounds,
    _exact_top_k,
)


def test_cuda_is_hidden():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}


def test_exact_top_k_matches_numpy_and_excludes_self():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(220)
    rows, dim, k = 257, 16, 5
    data = rng.standard_normal((rows, dim)).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    tensor = torch.from_numpy(data)
    ids, cosines = _exact_top_k(torch, tensor, rows=rows, k=k)
    ids = ids.numpy()
    cosines = cosines.numpy()
    assert ids.shape == (rows, k)
    assert not np.any(ids == np.arange(rows)[:, None])

    similarity = data @ data.T
    np.fill_diagonal(similarity, -np.inf)
    expected = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    assert strict_containment_rows(ids, expected).mean() == 1.0
    np.testing.assert_allclose(
        np.take_along_axis(similarity, ids.astype(np.int64), axis=1), cosines, atol=1e-6
    )


def test_exact_top_k_survives_exact_duplicate_clusters():
    """The substrate has a 1,377-row byte-identical cluster; self may not rank."""
    torch = pytest.importorskip("torch")
    rows, dim, k = 40, 8, 5
    data = np.tile(np.eye(1, dim, dtype=np.float32), (rows, 1))
    ids, _ = _exact_top_k(torch, torch.from_numpy(data), rows=rows, k=k)
    ids = ids.numpy()
    assert ids.shape == (rows, k)
    assert not np.any(ids == np.arange(rows)[:, None])
    assert graph_validity(ids, rows=rows)["zero_degree_rows"] == 0


def test_csr_bounds_rejects_unsorted_sources():
    sources = np.array([0, 0, 1, 2, 2], dtype=np.int32)
    bounds = _csr_bounds(sources, 3)
    assert bounds.tolist() == [0, 2, 3, 5]
    with pytest.raises(Round0220Error):
        _csr_bounds(np.array([1, 0], dtype=np.int32), 2)


def test_drop_self_keeps_k_neighbours():
    graph = np.array([[0, 5, 6], [9, 1, 8], [4, 5, 6]], dtype=np.uint32)
    got = _drop_self(graph, 2)
    assert got.shape == (3, 2)
    assert got[0].tolist() == [5, 6]
    assert got[1].tolist() == [9, 8]
    # No self present: the last column is dropped instead.
    assert got[2].tolist() == [4, 5]


def test_child_environment_gives_the_rapids_process_writable_caches():
    env = _child_environment("/tmp/round0220-cache")
    for key in ("HOME", "CUPY_CACHE_DIR", "XDG_CACHE_HOME", "NUMBA_CACHE_DIR"):
        assert env[key].startswith("/tmp/round0220-cache")


def test_actions_are_distinct():
    assert TRUTH_ACTION != QUALIFY_ACTION


def test_receipt_seals_and_serialises():
    """Every measurement dict must survive canonical JSON and the seal."""
    rng = np.random.default_rng(7)
    truth = rng.choice(np.arange(4_999), size=64 * GRAPH_K, replace=False).reshape(
        64, GRAPH_K
    )
    candidates = truth.copy()
    candidates[0, 0] = 4_999
    cosines = np.full((64, GRAPH_K), 0.9)
    kth = np.full(64, 0.8)
    fit = power_law([250_000, 500_000, 1_000_000, 2_000_000], [1.0, 2.1, 4.3, 8.9])
    receipt = prompt_contract.seal({
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "capabilities": [CAPABILITY],
        "metric": CUVS_METRIC,
        "sweep": [
            {
                "setting": dict(SWEEP[0]),
                "measurement": {
                    "strict_recall_at_15": summarize(
                        strict_containment_rows(candidates, truth), label="strict"
                    ),
                    "tie_aware_recall_at_15": summarize(
                        tie_aware_rows(cosines, candidates, kth), label="tie"
                    ),
                    "validity": graph_validity(candidates, rows=500),
                },
            }
        ],
        "scaling": {"setting_id": SCALING_SETTING_ID, "wall_fit": fit},
        "projection_100m": project_cost(fit, rows=PROJECTION_ROWS),
        "gate_registered": False,
        "map_quality_claim_available": False,
        "training_performed": False,
    })
    encoded = json.dumps(receipt, indent=2, sort_keys=True)
    prompt_contract.validate_seal(json.loads(encoded), label="R0220 smoke receipt")
    assert receipt["projection_100m"]["is_measurement"] is False


def test_truth_receipt_shape_seals():
    receipt = prompt_contract.seal({
        "schema": TRUTH_SCHEMA,
        "round_id": ROUND_ID,
        "probe": {
            "passed": True,
            "tie_aware": summarize(np.ones(16), label="tie"),
            "strict": summarize(np.ones(16), label="strict"),
        },
    })
    prompt_contract.validate_seal(receipt, label="R0220 truth smoke receipt")


def test_truth_cosines_are_never_rebound_in_run_truth():
    """R0220's first queue shadowed the truth cosines with a probe block."""
    import ast
    import inspect

    from experiments import round0220_nodes

    source = inspect.getsource(round0220_nodes.run_truth)
    tree = ast.parse(textwrap.dedent(source))
    stores = [
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]
    assert stores.count("cos_np") == 1, "truth cosines must be bound exactly once"
    assert stores.count("ids_np") == 1
