"""R0243 CPU smoke — BOTH node entry paths executed end to end on tiny inputs.

Three rounds in a row died on a defect no static check can see: R0216 on a
`NameError`, R0236 on an arity mismatch, R0242 attempt 1 on a third-party
module that does not exist in the release venv. Each of those lived on a code
path that nothing executed until the GPU node reached it, and each cost real
GPU time to discover. `check_undefined_names.py` cannot see any of them.

So this file does the only thing that can: it builds a complete miniature
queue - real `.npy` files, real sealed receipts, real signatures - and calls
`run_job` for BOTH actions, driving `run_residual` and `run_fuzzy` through
every stage they will run on the card, at `2,000` rows instead of
`100,000,000`.

Two seams are stubbed, and only two:

* `verify_inheritance` asserts the registered `sha256` of R0240's `6 GB` graph
  arrays. Those bytes cannot exist in a tmpdir. The real function runs in
  `prepare` and in the node, and its own contract is R0241's.
* `_cluster_assignment` requires a CUDA device. The real function is R0242's
  reviewed torch transcription, exercised by R0242's own tests.

Everything else - the binding, the seals, the decomposition, the tie-aware
join, the exposure profile, the magnitude arms, the verdict, the sorted
gather, the early write, the receipt, the sort, UMAP's own
`smooth_knn_dist`/`compute_membership_strengths`, the stripe-wise set
operation, the symmetrised-degree pass, canonicalization, the tripwire and
every artifact save - is the code that will run at `100,000,000`.
"""
from __future__ import annotations

import os
import shutil
import uuid

import numpy as np
import pytest

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_save_new_npy, atomic_write_new_json
from basemap.round0243_residual import (
    RESIDUAL_FILE,
    RESIDUAL_SCHEMA,
    loss_decomposition,
    observed_dispersion,
)
from experiments import round0243_nodes as nodes

TINY_ROWS = 2_000
TINY_DIM = 8
TINY_K = 15
TINY_CLUSTERS = 40
TINY_TOP_M = 3
TINY_SPILL = 2
TINY_PROBE = 400
RELEASE = "0" * 40


def _npy(path: str, array: np.ndarray) -> dict:
    atomic_save_new_npy(path, array)
    return expected_input_signature(path)


def _sealed(path: str, body: dict) -> dict:
    atomic_write_new_json(path, prompt_contract.seal(body))
    return expected_input_signature(path)


class _TinyWorld:
    """Every byte the two nodes read, at 2,000 rows."""

    def __init__(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        self.root = root
        rng = np.random.default_rng(2_430)

        substrate = rng.normal(size=(TINY_ROWS, TINY_DIM)).astype(np.float32)
        substrate /= np.linalg.norm(substrate, axis=1, keepdims=True)
        self.substrate = _npy(os.path.join(root, "substrate.f32.npy"), substrate)

        #: No self-loop and no duplicate within a row - the two properties
        #: R0241's degree-zero tripwire proved of the real graph over all
        #: 1,500,000,000 entries. Without them `A + A^T` sums duplicate (i, j)
        #: pairs and the weight ceiling of 1.0 stops being a property of the
        #: symmetrisation, which would make this smoke test's world unlike the
        #: one the node will actually see.
        ids = np.empty((TINY_ROWS, TINY_K), dtype=np.int32)
        for row in range(TINY_ROWS):
            pool = rng.permutation(TINY_ROWS)
            ids[row] = pool[pool != row][:TINY_K]
        cos = rng.uniform(0.1, 0.99, size=(TINY_ROWS, TINY_K)).astype(np.float32)
        self.ids = _npy(os.path.join(root, "graph-ids.i32.npy"), ids)
        self.cos = _npy(os.path.join(root, "graph-cos.f32.npy"), cos)

        from basemap.round0238_rung5 import TRUTH_PROBE_SEED, truth_probe_query_rows

        probe_rows = truth_probe_query_rows(
            rows=TINY_ROWS, size=TINY_PROBE, seed=TRUTH_PROBE_SEED
        )
        truth_ids = rng.integers(
            0, TINY_ROWS, size=(TINY_PROBE, TINY_K), dtype=np.int64
        ).astype(np.int32)
        truth_cos = np.sort(
            rng.uniform(0.2, 0.99, size=(TINY_PROBE, TINY_K)).astype(np.float32),
            axis=1,
        )[:, ::-1].copy()
        query_sig = _npy(os.path.join(root, "probe-rows.i64.npy"), probe_rows)
        truth_ids_sig = _npy(os.path.join(root, "truth-ids.i32.npy"), truth_ids)
        truth_cos_sig = _npy(os.path.join(root, "truth-cos.f32.npy"), truth_cos)
        self.truth = _sealed(os.path.join(root, "truth.json"), {
            "schema": "tiny-truth", "round_id": "0238",
            "outputs": {
                "query_rows": query_sig, "ids": truth_ids_sig,
                "cosines": truth_cos_sig,
            },
        })

        #: Reachability: most rows fully reachable, a tail below one.
        reach = np.ones(TINY_PROBE, dtype=np.float64)
        reach[:40] = (TINY_K - 1) / TINY_K
        reach[40:50] = (TINY_K - 3) / TINY_K
        reach_sig = _npy(os.path.join(root, "strict-c400.f64.npy"), reach)
        self.reachability = _sealed(os.path.join(root, "reachability.json"), {
            "schema": "tiny-reachability", "round_id": "0238",
            "cells": [{"clusters": TINY_CLUSTERS, "strict_vector": reach_sig}],
        })
        self.substrate_manifest = _sealed(
            os.path.join(root, "substrate-manifest.json"),
            {"schema": "tiny-substrate", "round_id": "0238",
             "substrate": self.substrate},
        )
        self.ladder = _sealed(os.path.join(root, "ladder.json"), {
            "schema": "tiny-ladder", "round_id": "0240", "rows": TINY_ROWS,
        })

        #: R0242's sealed per-row vectors. Strict loss is planted with one hot
        #: cell; the tie-aware vector forgives most of it, which is the shape
        #: review-0242-01/F3 measured at 100,000,000 rows.
        labels = rng.integers(0, TINY_CLUSTERS, size=TINY_PROBE).astype(np.int16)
        strict = np.ones(TINY_PROBE, dtype=np.float64)
        hot = np.flatnonzero(labels == 3)[:30]
        strict[hot] = (TINY_K - 4) / TINY_K
        cold = np.flatnonzero(labels != 3)[:25]
        strict[cold] = (TINY_K - 1) / TINY_K
        tie = strict.copy()
        tie[hot] = (TINY_K - 1) / TINY_K
        self.labels = labels
        self.strict = strict
        self.tie = tie

        vectors = {
            "r0242_probe_cluster": ("probe-cluster.i16.npy", labels),
            "r0242_probe_strict_recall": ("probe-strict.f64.npy", strict),
            "r0242_probe_tie_aware_recall": ("probe-tie.f64.npy", tie),
            "r0242_probe_missing_edges": (
                "probe-missing.i16.npy",
                np.rint((1.0 - strict) * TINY_K).astype(np.int16),
            ),
            "r0242_probe_builder_missing_edges": (
                "probe-builder-missing.i16.npy",
                np.rint((1.0 - strict) * TINY_K).astype(np.int16),
            ),
            "r0242_probe_in_degree": (
                "probe-in-degree.i32.npy",
                rng.integers(0, 40, size=TINY_PROBE).astype(np.int32),
            ),
            "r0242_primary_cluster": (
                "primary-cluster.i16.npy",
                rng.integers(0, TINY_CLUSTERS, size=TINY_ROWS).astype(np.int16),
            ),
        }
        self.vectors = {
            key: _npy(os.path.join(root, name), value)
            for key, (name, value) in vectors.items()
        }

        #: The R0242 receipt Part A's H0 gate checks itself against. It is
        #: built with the SAME imported functions, so a clean world reproduces
        #: and a perturbed one does not.
        decomposition = loss_decomposition(
            strict=strict, reachability=reach, k=TINY_K
        )
        split = decomposition.pop("vectors")
        populations = {
            "total_loss": (split["lost"], split["exposure_all"]),
            "partition_limited_loss": (
                split["partition_lost"], split["exposure_all"]
            ),
            "builder_loss_inside_partition": (
                split["builder_lost"], split["exposure_builder"]
            ),
        }
        tests = {}
        for name, (missing, exposure) in populations.items():
            seen = observed_dispersion(
                labels=labels.astype(np.int64), missing=missing,
                exposure=exposure, clusters=TINY_CLUSTERS,
                top_m=TINY_TOP_M,
            )
            tests[name] = {
                "observed": seen,
                "chi_square": {"observed": seen["chi_square"],
                               "p_value": 9.999e-05},
                "top_m_share_of_missing": {
                    "observed": seen["top_m_share_of_missing"]
                },
                "max_single_cluster_share_of_missing": {
                    "observed": seen["max_single_cluster_share_of_missing"]
                },
            }
        self.decomposition = decomposition
        self.locality = _sealed(os.path.join(root, "loss-locality.json"), {
            "schema": "round0242-minilm-mixed-100000k-k15-loss-locality-v1",
            "round_id": "0242",
            "decomposition": decomposition,
            "cluster_locality_tests": tests,
        })

    def inheritance(self) -> dict:
        return {
            "note": "tiny",
            "substrate": {"source": self.substrate_manifest},
            "truth": {"source": self.truth},
            "reachability": {"source": self.reachability},
            "ladder": {"source": self.ladder},
            "graph": {"ids": self.ids, "cosines": self.cos},
        }


def _patch(monkeypatch, world: _TinyWorld) -> None:
    monkeypatch.setattr(nodes, "ROWS", TINY_ROWS)
    monkeypatch.setattr(nodes, "DIMENSION", TINY_DIM)
    monkeypatch.setattr(nodes, "CLUSTERS", TINY_CLUSTERS)
    monkeypatch.setattr(nodes, "SPILL", TINY_SPILL)
    monkeypatch.setattr(nodes, "TRUTH_PROBE_ROWS", TINY_PROBE)
    monkeypatch.setattr(nodes, "PERMUTATIONS", 20)
    monkeypatch.setattr(nodes, "CONCENTRATION_TOP_M", TINY_TOP_M)
    monkeypatch.setattr(nodes, "SORTED_GATHER_ANCHORS", (50, 200))
    monkeypatch.setattr(nodes, "SORTED_GATHER_READ_BLOCK", 64)
    monkeypatch.setattr(nodes, "SORTED_GATHER_ID_BLOCK", 32)
    monkeypatch.setattr(nodes, "PRIMARY_LABEL_BLOCK", 512)
    monkeypatch.setattr(nodes, "SUBSTRATE_BYTES", TINY_ROWS * TINY_DIM * 4)
    monkeypatch.setattr(nodes, "RESIDUAL_DEADLINE_S", 600.0)
    monkeypatch.setattr(nodes, "FUZZY_DEADLINE_S", 600.0)
    monkeypatch.setattr(
        nodes, "R0242_LOCALITY_SHA256", world.locality["sha256"]
    )
    monkeypatch.setattr(
        nodes, "R0242_TIE_AWARE_VECTOR_SHA256",
        world.vectors["r0242_probe_tie_aware_recall"]["sha256"],
    )
    monkeypatch.setattr(
        nodes, "R0242_PRIMARY_CLUSTER_SHA256",
        world.vectors["r0242_primary_cluster"]["sha256"],
    )
    monkeypatch.setattr(
        nodes, "verify_inheritance", lambda job: world.inheritance()
    )

    def _assignment(*, substrate, clusters, seed, spill):
        rng = np.random.default_rng(int(seed))
        assignment = rng.integers(
            0, int(clusters), size=(int(substrate.shape[0]), int(spill))
        ).astype(np.int16)
        return assignment, {
            "clusters": int(clusters), "spill": int(spill), "seed": int(seed),
            "backend": "tiny-stub-in-test-only",
        }

    monkeypatch.setattr(nodes, "_cluster_assignment", _assignment)


def _job_a(world: _TinyWorld, output: str) -> dict:
    job = {
        "action": nodes.RESIDUAL_ACTION,
        "outputs": [output],
        "stage_budget_s": 600.0,
        "r0242_locality": world.locality,
    }
    job.update(world.vectors)
    return job


#: `output_safety` refuses every path outside `/data`, by design: that
#: containment rule is one of the guards this program relies on and a test must
#: not weaken it. So the miniature world lives in a unique directory under
#: `/data` and is removed when the test ends.
SMOKE_ROOT = "/data/latent-basemap/tests"


@pytest.fixture()
def scratch():
    os.makedirs(SMOKE_ROOT, exist_ok=True)
    root = os.path.join(SMOKE_ROOT, f"round0243-{uuid.uuid4().hex}")
    os.makedirs(root)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def world(scratch) -> _TinyWorld:
    return _TinyWorld(os.path.join(scratch, "world"))


def test_part_a_entry_path_runs_end_to_end(monkeypatch, scratch, world) -> None:
    """`run_job` -> `run_residual`, every stage, on 2,000 rows."""
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-residual")
    nodes.run_job(
        {"manifest": {"round_id": "0243", "release_sha": RELEASE}},
        _job_a(world, output),
    )
    receipt = prompt_contract.read_sealed(
        os.path.join(output, RESIDUAL_FILE), label="tiny residual"
    )
    assert receipt["schema"] == RESIDUAL_SCHEMA
    assert receipt["strict_reproduction_gate"]["agree"] is True
    assert receipt["residual_verdict"]["h0_strict_reproduction_agrees"] is True
    # The planted world is the R0242 shape: strict loss concentrated in one
    # cell, most of it tie-forgiven.
    forgiveness = receipt["tie_forgiveness"]
    assert (
        forgiveness["tie_aware_builder_missing_edges"]
        < forgiveness["strict_builder_missing_edges"]
    )
    assert receipt["tie_aware_hot_cell_scan"]["clusters"] == TINY_CLUSTERS
    assert receipt["exposure_profile"]["clusters"] == TINY_CLUSTERS
    assert len(receipt["sorted_gathers"]) == 2
    for priced in receipt["sorted_gathers"].values():
        assert priced["distinct_rows_touched"] > 0
        assert priced["wall_guard"]["units_done"] > 0
    assert receipt["full_sorted_gather_prediction"]["kind"] == "prediction"
    assert receipt["partition_agreement_with_r0238"]["rows"] == TINY_PROBE
    assert receipt["map_harm_assessment"]["probe_edges"] == TINY_PROBE * TINY_K
    assert receipt["partition_size_profile"]["rows_counted"] == TINY_ROWS
    assert receipt["cuvs_calls"] == 0
    assert receipt["child_processes_launched"] == 0
    assert receipt["signal_delivered"] is False
    for name in (
        "reproduced-strict-c400.f64.npy",
        "reproduced-primary-cluster-c400.i16.npy",
        "probe-tie-aware-missing-edges.i16.npy",
        "probe-tie-aware-builder-missing-edges.i16.npy",
    ):
        assert os.path.isfile(os.path.join(output, "vectors", name)), name
    # The early write exists and precedes the receipt.
    assert os.path.isfile(
        os.path.join(output, "tie-aware-locality-first-write.json")
    )


def test_part_a_halts_when_the_strict_gate_fails(
    monkeypatch, scratch, world
) -> None:
    """H0, exercised on the node path: a drifted sealed receipt stops the round."""
    _patch(monkeypatch, world)
    drifted_body = dict(
        prompt_contract.read_sealed(
            world.locality["canonical_path"], label="tiny locality"
        )
    )
    drifted_body.pop("identity_sha256", None)
    decomposition = dict(drifted_body["decomposition"])
    decomposition["builder_missing_edges"] = (
        int(decomposition["builder_missing_edges"]) + 1
    )
    drifted_body["decomposition"] = decomposition
    drifted_path = os.path.join(scratch, "drifted-locality.json")
    signature = _sealed(drifted_path, drifted_body)
    monkeypatch.setattr(nodes, "R0242_LOCALITY_SHA256", signature["sha256"])
    job = _job_a(world, os.path.join(scratch, "artifacts-drifted"))
    job["r0242_locality"] = signature
    with pytest.raises(nodes.Round0243Error, match="H0"):
        nodes.run_job(
            {"manifest": {"round_id": "0243", "release_sha": RELEASE}}, job
        )


def _residual_receipt(path: str, *, may_run: bool) -> str:
    atomic_write_new_json(path, prompt_contract.seal({
        "schema": RESIDUAL_SCHEMA,
        "round_id": "0243",
        "release_sha": RELEASE,
        "residual_verdict": {
            "part_b_may_run": may_run,
            "halt_part_b": not may_run,
            "h0_strict_reproduction_agrees": True,
            "h1_global_tie_aware_builder_rate": 0.0004,
            "h1_threshold": 0.01,
            "h2_cells_firing_all_three_arms": 0 if may_run else 1,
            "h2_firing_clusters": [] if may_run else [3],
        },
    }))
    return path


def test_part_b_entry_path_runs_end_to_end(monkeypatch, scratch, world) -> None:
    """`run_job` -> `run_fuzzy`: sort, UMAP's law, degree once, tripwire."""
    _patch(monkeypatch, world)
    reference = _residual_receipt(
        os.path.join(scratch, "residual.json"), may_run=True
    )
    output = os.path.join(scratch, "artifacts-fuzzy")
    nodes.run_job(
        {"manifest": {"round_id": "0243", "release_sha": RELEASE}},
        {
            "action": nodes.FUZZY_ACTION,
            "outputs": [output],
            "stage_budget_s": 600.0,
            "residual_reference": reference,
        },
    )
    receipt = prompt_contract.read_sealed(
        os.path.join(output, "fuzzy-graph.json"), label="tiny fuzzy"
    )
    assert receipt["rows"] == TINY_ROWS
    assert receipt["directed_edges"] > TINY_ROWS
    assert receipt["weight_distribution"]["valid"] is True
    assert receipt["symmetrised_degree"]["reported"] == "once"
    assert "degree" not in receipt["symmetrised_degree"]
    assert receipt["symmetrised_degree"]["identity_cross_check"][
        "in_degree_equals_out_degree_on_every_sampled_row"
    ] is True
    assert receipt["post_canonical_tripwire"]["rows"] == TINY_ROWS
    assert receipt["canonicalization"]["canonical_undirected_edges"] > 0
    assert receipt["io"]["gather_term_lives_in"].startswith("Part A")
    for name in (
        "graph-k15-ids.i32.npy", "edges-k15-fuzzy-src.i32.npy",
        "edges-k15-fuzzy-dst.i32.npy", "edges-k15-fuzzy-wts.f32.npy",
        "edges-k15-fuzzy-header.npz",
    ):
        assert os.path.isfile(os.path.join(output, name)), name
    assert not os.path.isdir(os.path.join(output, ".fuzzy")), (
        "the fuzzy scratch must be released once the edge list is published"
    )


def test_part_b_refuses_itself_on_a_halting_verdict(
    monkeypatch, scratch, world
) -> None:
    """The refusal is bound to the sealed verdict, not to job ordering."""
    _patch(monkeypatch, world)
    reference = _residual_receipt(
        os.path.join(scratch, "halted.json"), may_run=False
    )
    with pytest.raises(nodes.Round0243Error, match="STOP before Part B"):
        nodes.run_job(
            {"manifest": {"round_id": "0243", "release_sha": RELEASE}},
            {
                "action": nodes.FUZZY_ACTION,
                "outputs": [os.path.join(scratch, "artifacts-refused")],
                "stage_budget_s": 600.0,
                "residual_reference": reference,
            },
        )


def test_part_b_refuses_a_tampered_part_a_receipt(
    monkeypatch, scratch, world
) -> None:
    """`read_sealed` recomputes `identity_sha256`; a flipped verdict is refused."""
    _patch(monkeypatch, world)
    path = _residual_receipt(
        os.path.join(scratch, "tampered.json"), may_run=False
    )
    import json

    with open(path, encoding="utf-8") as handle:
        body = json.load(handle)
    body["residual_verdict"]["part_b_may_run"] = True
    tampered = os.path.join(scratch, "tampered-2.json")
    atomic_write_new_json(tampered, body)
    with pytest.raises(Exception, match="identity seal"):
        nodes.run_job(
            {"manifest": {"round_id": "0243", "release_sha": RELEASE}},
            {
                "action": nodes.FUZZY_ACTION,
                "outputs": [os.path.join(scratch, "artifacts-tampered")],
                "stage_budget_s": 600.0,
                "residual_reference": tampered,
            },
        )


def test_run_job_refuses_an_unregistered_action(world) -> None:
    with pytest.raises(nodes.Round0243Error, match="does not authorize"):
        nodes.run_job(
            {"manifest": {"round_id": "0243", "release_sha": RELEASE}},
            {"action": "train_a_map", "outputs": []},
        )


def test_run_job_refuses_another_rounds_queue(monkeypatch, scratch, world) -> None:
    _patch(monkeypatch, world)
    with pytest.raises(nodes.Round0243Error, match="another queue"):
        nodes.run_job(
            {"manifest": {"round_id": "0242", "release_sha": RELEASE}},
            _job_a(world, os.path.join(scratch, "artifacts-wrong")),
        )
