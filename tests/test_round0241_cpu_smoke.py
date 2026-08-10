"""R0241 CPU smoke — the identity that licenses skipping the 4.3 h product step.

The load-bearing claim of this round is that a row is isolated in the
symmetrised graph **iff** it has no usable out-edge and no usable in-edge in the
raw directed graph. If that is wrong, reporting in-degree without symmetrising
is wrong, and the tripwire this round runs is not the tripwire the plan
registers. So it is proved here against R0238's own `_fuzzy_symmetrise_blocked`
- the very function this round declines to run at scale - on synthetic graphs
that deliberately contain isolated rows.

Everything else here is fail-closed behaviour: the inheritance verification
against the LIVE sealed manifests, the probe view's refusal to be indexed by
anything but the registered draw, and the degree pass cross-checked against
R0220's reviewed `graph_validity`.

CPU only, no CUDA context, no child process. The identity tests do write a few
kilobytes of scratch under `/data/latent-basemap/tmp/`, because R0238's
`_fuzzy_symmetrise_blocked` refuses any output directory outside `/data` and
this test calls it unchanged; the scratch is removed afterwards.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import uuid

import numpy as np
import pytest

from basemap.artifact_identity import expected_input_signature
from basemap.round0238_rung5 import GRAPH_K, TRUTH_PROBE_ROWS, TRUTH_PROBE_SEED
from basemap.round0240_rung5 import (
    INHERITED_REACHABILITY_MANIFEST,
    INHERITED_SUBSTRATE_MANIFEST,
    INHERITED_TRUTH_MANIFEST,
    verify_inherited_reachability,
    verify_inherited_substrate,
    verify_inherited_truth,
)
from basemap.round0241_qualify import (
    INHERITED_GRAPH_COS,
    INHERITED_GRAPH_IDS,
    INHERITED_LADDER_RECEIPT,
    REGISTERED_GRAPH_ARRAY_BYTES,
    REGISTERED_GRAPH_COS_SHA256,
    REGISTERED_GRAPH_IDS_SHA256,
    REGISTERED_LADDER_RECEIPT_SHA256,
    ROWS,
    Round0241Error,
    cross_check_structural,
)
from experiments.round0238_nodes import _graph_validity_blocked
from experiments.round0241_nodes import (
    _ProbeRowView,
    _degree_pass,
    verify_inheritance,
)
from basemap.round0241_qualify import StageGuard


SMOKE_SCRATCH_ROOT = "/data/latent-basemap/tmp"


@pytest.fixture()
def scratch_dir():
    """A short-lived directory under /data, which the reviewed function requires."""
    path = os.path.join(SMOKE_SCRATCH_ROOT, f"round0241-smoke-{uuid.uuid4().hex}")
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _guard(units: int) -> StageGuard:
    return StageGuard(
        label="test", units_total=max(1, units), budget_s=1e9, deadline_s=1e9,
    )


# --------------------------------------------------------------------------- #
# the identity
# --------------------------------------------------------------------------- #
def _symmetrised_isolated_rows(
    ids: np.ndarray, *, rows: int, k: int, out_dir: str
) -> set[int]:
    """Rows with no edge after R0238's own blocked fuzzy symmetrisation."""
    import shutil

    import umap.umap_ as umap_api

    from experiments.round0238_nodes import _fuzzy_symmetrise_blocked

    # Distances must be nondecreasing per row for smooth_knn_dist, so build
    # them from a monotone rank: the law is exercised on a well-formed input.
    dists = np.tile(np.linspace(0.0, 0.5, k, dtype=np.float32), (rows, 1))
    fuzzy = _fuzzy_symmetrise_blocked(
        knn_indices=np.ascontiguousarray(ids.astype(np.int32)),
        knn_dists=np.ascontiguousarray(dists),
        rows=rows, k=k, umap_api=umap_api, out_dir=out_dir,
        stripe_rows=max(1, rows // 2),
    )
    touched = set(np.asarray(fuzzy["src"]).tolist()) | set(
        np.asarray(fuzzy["dst"]).tolist()
    )
    shutil.rmtree(fuzzy["scratch"], ignore_errors=True)
    return set(range(rows)) - touched


def _predicted_isolated(ids: np.ndarray, *, rows: int) -> set[int]:
    """This round's prediction, from the raw id array alone."""
    measured = _degree_pass(
        np.ascontiguousarray(ids.astype(np.int32)), guard=_guard(1)
    )
    out_zero = measured["out_degree"]["zero_rows"]
    isolated = measured["symmetrised"]["isolated_rows"]
    # recompute the explicit set for comparison
    from basemap.round0220_cuvs_qualification import _first_occurrence_mask

    stripe = ids.astype(np.int64)
    out_of_range = (stripe < 0) | (stripe >= rows)
    self_loops = (stripe == np.arange(rows, dtype=np.int64)[:, None]) & ~out_of_range
    usable = _first_occurrence_mask(stripe) & ~out_of_range & ~self_loops
    out_degree = usable.sum(axis=1)
    in_degree = np.bincount(stripe[usable], minlength=rows)
    predicted = set(np.flatnonzero((out_degree == 0) & (in_degree == 0)).tolist())
    assert len(predicted) == isolated
    assert int((out_degree == 0).sum()) == out_zero
    return predicted


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_symmetrised_isolation_identity_on_random_graphs(seed, scratch_dir):
    rows, k = 64, GRAPH_K
    rng = np.random.RandomState(seed)
    ids = rng.randint(0, rows, size=(rows, k)).astype(np.int32)
    assert _predicted_isolated(ids, rows=rows) == _symmetrised_isolated_rows(
        ids, rows=rows, k=k, out_dir=scratch_dir
    )


def test_symmetrised_isolation_identity_when_a_row_is_genuinely_isolated(scratch_dir):
    """The case that matters: a row whose only entries are self-loops.

    Row 0 points at nothing but itself and no other row points at it, so it
    must be isolated after symmetrisation. If the identity were wrong in this
    direction the tripwire would report a clean graph while the map carries the
    v1 defect.
    """
    rows, k = 32, GRAPH_K
    rng = np.random.RandomState(7)
    ids = rng.randint(1, rows, size=(rows, k)).astype(np.int32)
    ids[0, :] = 0                      # row 0: nothing but self-loops
    ids[1:, :] = np.where(ids[1:, :] == 0, 1, ids[1:, :])  # nobody points at 0
    predicted = _predicted_isolated(ids, rows=rows)
    assert 0 in predicted
    assert predicted == _symmetrised_isolated_rows(
        ids, rows=rows, k=k, out_dir=scratch_dir
    )


def test_symmetrised_isolation_identity_with_an_in_edge_only(scratch_dir):
    """A row with no usable OUT-edge but an in-edge is NOT isolated.

    This is the distinction the round file insists on: raw directed in-degree
    zero is descriptive, and only the conjunction gates.
    """
    rows, k = 24, GRAPH_K
    rng = np.random.RandomState(11)
    ids = rng.randint(1, rows, size=(rows, k)).astype(np.int32)
    ids[0, :] = 0                      # row 0 has no usable out-edge
    ids[5, 0] = 0                      # but row 5 points at it
    predicted = _predicted_isolated(ids, rows=rows)
    assert 0 not in predicted
    assert predicted == _symmetrised_isolated_rows(
        ids, rows=rows, k=k, out_dir=scratch_dir
    )


# --------------------------------------------------------------------------- #
# the degree pass against R0220's reviewed structural scan
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [3, 4])
def test_degree_pass_agrees_with_the_reviewed_graph_validity(seed):
    rows, k = 500, GRAPH_K
    rng = np.random.RandomState(seed)
    ids = rng.randint(-2, rows + 2, size=(rows, k)).astype(np.int32)
    ids[7, :] = 7                       # self-loops only
    ids[9, :] = ids[9, 0]               # all duplicates
    array = np.ascontiguousarray(ids)
    reviewed = _graph_validity_blocked(array, rows=rows, block=137)
    own = _degree_pass(array, guard=_guard(1))["own_structural"]
    assert cross_check_structural(reviewed=reviewed, own=own)["agree"] is True


def test_degree_pass_catches_a_seeded_edgeless_row():
    rows, k = 200, GRAPH_K
    rng = np.random.RandomState(5)
    ids = rng.randint(1, rows, size=(rows, k)).astype(np.int32)
    ids[0, :] = 0
    ids[1:, :] = np.where(ids[1:, :] == 0, 1, ids[1:, :])
    measured = _degree_pass(np.ascontiguousarray(ids), guard=_guard(1))
    assert measured["out_degree"]["zero_rows"] >= 1
    assert measured["symmetrised"]["isolated_rows"] >= 1
    assert measured["own_structural"]["zero_degree_rows"] >= 1


# --------------------------------------------------------------------------- #
# the probe view is fail-closed
# --------------------------------------------------------------------------- #
def test_probe_view_returns_the_block_for_the_registered_draw():
    rows = np.array([3, 9, 20], dtype=np.int64)
    values = np.arange(3 * GRAPH_K, dtype=np.float32).reshape(3, GRAPH_K)
    view = _ProbeRowView(rows=rows, values=values)
    assert np.array_equal(view[rows], values)
    assert view.shape == (3, GRAPH_K)


def test_probe_view_refuses_any_other_index():
    rows = np.array([3, 9, 20], dtype=np.int64)
    values = np.zeros((3, GRAPH_K), dtype=np.float32)
    view = _ProbeRowView(rows=rows, values=values)
    for bad in (np.array([3, 9, 21]), np.array([3, 9]), np.arange(3)):
        with pytest.raises(Round0241Error, match="registered uniform probe"):
            _ = view[bad]


def test_score_probe_through_the_view_matches_scoring_full_arrays():
    from experiments.round0238_nodes import _score_probe

    rng = np.random.RandomState(19)
    rows, probe = 400, 40
    probe_rows = np.sort(
        rng.choice(rows, size=probe, replace=False).astype(np.int64)
    )
    ids = rng.randint(0, rows, size=(rows, GRAPH_K)).astype(np.int32)
    cos = rng.uniform(0.5, 1.0, size=(rows, GRAPH_K)).astype(np.float32)
    truth_ids = ids[probe_rows].copy()
    kth = cos[probe_rows].min(axis=1).astype(np.float64)
    best = cos[probe_rows].max(axis=1).astype(np.float64)

    full = _score_probe(
        ids=ids, candidate_cos=cos, probe_rows=probe_rows,
        truth_ids=truth_ids, kth=kth, truth_best=best,
    )
    viewed = _score_probe(
        ids=_ProbeRowView(rows=probe_rows, values=ids[probe_rows]),
        candidate_cos=_ProbeRowView(rows=probe_rows, values=cos[probe_rows]),
        probe_rows=probe_rows, truth_ids=truth_ids, kth=kth, truth_best=best,
    )
    assert json.dumps(full, sort_keys=True, default=str) == json.dumps(
        viewed, sort_keys=True, default=str
    )


# --------------------------------------------------------------------------- #
# the inheritance is verified against the LIVE sealed artifacts
# --------------------------------------------------------------------------- #
def _sealed(path: str) -> dict:
    from basemap import round0113_prompt_contrast as prompt_contract

    return prompt_contract.read_sealed(path, label="R0241 smoke")


@pytest.mark.skipif(
    not os.path.exists(INHERITED_SUBSTRATE_MANIFEST),
    reason="the sealed R0238 substrate manifest is not on this box",
)
def test_live_manifests_reproduce_every_registered_literal():
    assert verify_inherited_substrate(
        _sealed(INHERITED_SUBSTRATE_MANIFEST)
    )["verified"] is True
    truth = verify_inherited_truth(_sealed(INHERITED_TRUTH_MANIFEST))
    assert truth["probe_rows"] == TRUTH_PROBE_ROWS
    assert truth["probe_seed"] == TRUTH_PROBE_SEED
    assert verify_inherited_reachability(
        _sealed(INHERITED_REACHABILITY_MANIFEST)
    )["verified"] is True


@pytest.mark.skipif(
    not os.path.exists(INHERITED_LADDER_RECEIPT),
    reason="R0240's sealed build ladder is not on this box",
)
def test_the_bound_graph_is_the_graph_r0240_built():
    ladder = _sealed(INHERITED_LADDER_RECEIPT)
    assert int(ladder["rows"]) == ROWS
    assert int(ladder["cluster_selection"]["selected_clusters"]) == 400
    assert expected_input_signature(INHERITED_LADDER_RECEIPT)["sha256"] == (
        REGISTERED_LADDER_RECEIPT_SHA256
    )
    # sizes only here; the 6 GB hashes are verified once, by
    # test_verify_inheritance_refuses_a_graph_that_is_not_the_registered_one
    for path in (INHERITED_GRAPH_IDS, INHERITED_GRAPH_COS):
        assert os.path.getsize(path) == REGISTERED_GRAPH_ARRAY_BYTES


@pytest.mark.skipif(
    not os.path.exists(INHERITED_LADDER_RECEIPT),
    reason="R0240's sealed build ladder is not on this box",
)
def test_verify_inheritance_refuses_a_graph_that_is_not_the_registered_one():
    job = {
        "substrate_manifest": expected_input_signature(
            INHERITED_SUBSTRATE_MANIFEST
        ),
        "truth_reference": expected_input_signature(INHERITED_TRUTH_MANIFEST),
        "reachability_reference": expected_input_signature(
            INHERITED_REACHABILITY_MANIFEST
        ),
        "ladder_reference": expected_input_signature(INHERITED_LADDER_RECEIPT),
        "graph_ids": expected_input_signature(INHERITED_GRAPH_IDS),
        "graph_cos": expected_input_signature(INHERITED_GRAPH_COS),
    }
    assert verify_inheritance(job)["graph"]["verified"] is True

    tampered = copy.deepcopy(job)
    tampered["graph_ids"] = dict(tampered["graph_ids"])
    tampered["graph_ids"]["sha256"] = "0" * 64
    with pytest.raises(Exception):
        verify_inheritance(tampered)

    no_hash = copy.deepcopy(job)
    no_hash["graph_ids"] = {
        "kind": "file", "canonical_path": INHERITED_GRAPH_IDS,
    }
    with pytest.raises(Round0241Error, match="full sha256"):
        verify_inheritance(no_hash)
