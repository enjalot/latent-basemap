"""R0244 CPU smoke — ALL FOUR node entry paths executed end to end.

R0216 died on a `NameError`, R0236 on an arity mismatch, R0242 attempt 1 on a
module that does not exist in the release venv. Each lived on a code path that
nothing executed until the node reached it, and `check_undefined_names.py` can
see none of them. So this file builds a complete miniature world — real
`.npy` files, a real parquet, a real provenance record array, real MiniLM
embeddings — and calls `run_job` for every one of the four actions at `40` rows
instead of `100,000,000`.

Exactly one seam is stubbed: `verify_inheritance`, which asserts the registered
`sha256` of R0240's `6 GB` graph arrays and of R0238's `153.6 GB` substrate.
Those bytes cannot exist in a scratch directory. Everything else — the threaded
watchdog and its positive control, UMAP's own `smooth_knn_dist` and
`compute_membership_strengths`, the stripe-wise set operation, the streaming
weight profile, the two-level draw, the fidelity check, the mis-sampler
control, the DiD registration and its row sets, the provenance-to-parquet
resolution, the re-embedding verification and every receipt seal — is the code
that will run at `100,000,000`.
"""
from __future__ import annotations

import os
import shutil
import uuid

import numpy as np
import pytest

from basemap import round0113_prompt_contrast as prompt_contract
from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import atomic_save_new_npy
from experiments import round0244_nodes as nodes

TINY_ROWS = 40
TINY_K = 15
#: Every substrate row is a probe row here, and the true 15th-best cosine is
#: laid out as ten groups of four so R0228's decile match always has an intact
#: companion for every row that carries loss. At 100,000,000 rows that is a
#: property of the data; at 40 it has to be built.
TINY_PROBE = 40
TINY_LOST_GROUP_STRIDE = 4
TINY_CORPUS = "tiny-chunked-120-all-MiniLM-L6-v2"
TINY_SHARD = "data-00000-of-00001"
RELEASE = "0" * 40
SMOKE_ROOT = "/data/latent-basemap/tests"

TEXTS = [
    "the independent jane austen and the freedom to choose in her novels",
    "photosynthesis converts light energy into chemical energy in plants",
    "def add(a, b):\n    return a + b\n",
    "the mitochondrion is the powerhouse of the eukaryotic cell",
    "a short note about the weather in the north atlantic this week",
    "gradient descent minimises a loss by stepping along the negative gradient",
    "terms of service: by using this site you agree to the following terms",
    "terms of service: by using this website you agree to the following terms",
    "the quick brown fox jumps over the lazy dog again and again",
    "quantum chromodynamics confines colour charge at low energies",
]


def _npy(path: str, array: np.ndarray) -> dict:
    atomic_save_new_npy(path, array)
    return expected_input_signature(path)


class _TinyWorld:
    """Every byte the four nodes read, at 40 rows."""

    def __init__(self, root: str) -> None:
        os.makedirs(root, exist_ok=True)
        self.root = root
        rng = np.random.default_rng(244_000)

        #: Real texts, embedded by the model that produced the real substrate,
        #: so the text node's binding check is exercised rather than bypassed.
        self.texts = [
            f"{TEXTS[index % len(TEXTS)]} [{index}]" for index in range(TINY_ROWS)
        ]
        os.environ.setdefault("HF_HOME", "/data/hf")
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(nodes.TEXT_MODEL, device="cpu")
        substrate = np.asarray(
            model.encode(
                self.texts, normalize_embeddings=True, batch_size=8,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        assert substrate.shape == (TINY_ROWS, 384)
        self.substrate = _npy(
            os.path.join(root, "substrate.f32.npy"), substrate
        )
        self.substrate_matrix = substrate

        provenance = np.zeros(
            TINY_ROWS, dtype=[("corpus", "u1"), ("shard", "<u2"), ("row", "<i8")]
        )
        provenance["row"] = np.arange(TINY_ROWS, dtype=np.int64)
        self.provenance = _npy(
            os.path.join(root, "provenance.npy"), provenance
        )

        #: The chunk corpus, laid out exactly as `/data/chunks` is.
        import pyarrow as pa
        import pyarrow.parquet as pq

        chunk_root = os.path.join(root, "chunks")
        embedding_root = os.path.join(root, "embeddings")
        chunk_dir = os.path.join(
            chunk_root, TINY_CORPUS[: -len(nodes.EMBEDDING_SUFFIX)], "train"
        )
        shard_dir = os.path.join(embedding_root, TINY_CORPUS, "train")
        os.makedirs(chunk_dir, exist_ok=True)
        os.makedirs(shard_dir, exist_ok=True)
        pq.write_table(
            pa.table({
                "chunk_text": self.texts,
                "chunk_index": list(range(TINY_ROWS)),
                "id": [f"doc-{index // 4}" for index in range(TINY_ROWS)],
            }),
            os.path.join(chunk_dir, f"{TINY_SHARD}.parquet"),
            row_group_size=8,
        )
        atomic_save_new_npy(
            os.path.join(shard_dir, f"{TINY_SHARD}.npy"), substrate
        )
        self.chunk_root = chunk_root
        self.embedding_root = embedding_root

        labels = np.full(TINY_ROWS, 7, dtype=np.int16)
        labels[:20] = nodes.CLUSTER_UNDER_TEST
        self.labels = _npy(os.path.join(root, "primary-cluster.i16.npy"), labels)

        probe_rows = np.arange(TINY_PROBE, dtype=np.int64)
        self.probe_rows = _npy(
            os.path.join(root, "probe-query-rows.i64.npy"), probe_rows
        )
        probe_cluster = np.asarray(labels[probe_rows], dtype=np.int16)
        self.probe_cluster = _npy(
            os.path.join(root, "probe-cluster.i16.npy"), probe_cluster
        )

        #: One row in each group of four carries loss. Groups 0-2 lose one edge
        #: that a tie-equivalent substitute covered; groups 3-5 lose two that
        #: nothing covered; groups 6-9 lose nothing.
        strict = np.zeros(TINY_PROBE, dtype=np.int16)
        tie = np.zeros(TINY_PROBE, dtype=np.int16)
        self.forgiven_probes = [0, 4, 8]
        self.genuine_probes = [12, 16, 20]
        for probe in self.forgiven_probes:
            strict[probe] = 1
        for probe in self.genuine_probes:
            strict[probe] = 2
            tie[probe] = 2
        self.strict_vector = strict
        self.tie_vector = tie
        self.strict_builder = _npy(
            os.path.join(root, "probe-builder-missing.i16.npy"), strict
        )
        self.tie_builder = _npy(
            os.path.join(root, "probe-tie-builder-missing.i16.npy"), tie
        )
        strict_recall = 1.0 - strict.astype(np.float64) / float(TINY_K)
        self.strict_recall = _npy(
            os.path.join(root, "probe-strict-recall.f64.npy"), strict_recall
        )
        self.reachability = _npy(
            os.path.join(root, "strict-c400.f64.npy"),
            np.ones(TINY_PROBE, dtype=np.float64),
        )

        #: Truth and builder rows arranged so that each tie-forgiven probe has
        #: exactly one missed true neighbour and exactly one substitute whose
        #: cosine equals it to the last bit.
        truth_ids = np.zeros((TINY_PROBE, TINY_K), dtype=np.int32)
        truth_cos = np.zeros((TINY_PROBE, TINY_K), dtype=np.float32)
        graph_ids = np.zeros((TINY_ROWS, TINY_K), dtype=np.int32)
        graph_cos = np.zeros((TINY_ROWS, TINY_K), dtype=np.float32)
        for row in range(TINY_ROWS):
            pool = rng.permutation(TINY_ROWS)
            graph_ids[row] = pool[pool != row][:TINY_K]
            graph_cos[row] = np.sort(
                rng.uniform(0.1, 0.95, size=TINY_K).astype(np.float32)
            )[::-1]
        for probe in range(TINY_PROBE):
            query = int(probe_rows[probe])
            pool = np.array(
                [row for row in range(TINY_ROWS) if row != query], dtype=np.int32
            )
            picked = rng.permutation(pool)[: TINY_K + 1]
            truth_ids[probe] = picked[:TINY_K]
            kth = 0.30 + 0.01 * (probe // TINY_LOST_GROUP_STRIDE)
            cosines = np.linspace(0.99, kth, TINY_K).astype(np.float32)
            truth_cos[probe] = cosines
            if probe in self.forgiven_probes:
                graph_ids[query, :TINY_K - 1] = picked[: TINY_K - 1]
                graph_ids[query, TINY_K - 1] = picked[TINY_K]
                graph_cos[query] = cosines
            else:
                graph_ids[query] = picked[:TINY_K]
                graph_cos[query] = cosines
        self.truth_ids = _npy(os.path.join(root, "truth-ids.i32.npy"), truth_ids)
        self.truth_cos = _npy(os.path.join(root, "truth-cos.f32.npy"), truth_cos)
        self.graph_ids_path = os.path.join(root, "graph-ids.i32.npy")
        self.graph_cos_path = os.path.join(root, "graph-cos.f32.npy")
        self.graph_ids = _npy(self.graph_ids_path, graph_ids)
        self.graph_cos = _npy(self.graph_cos_path, graph_cos)

        #: A miniature symmetrised edge list for the sampler node.
        edges = 30_000
        weights = rng.beta(0.7, 2.0, size=edges).astype(np.float32)
        weights = np.maximum(weights, np.float32(1e-6))
        weights[rng.integers(0, edges, size=200)] = np.float32(1.0)
        src = rng.integers(0, TINY_ROWS, size=edges, dtype=np.int64)
        dst = (src + 1 + rng.integers(0, TINY_ROWS - 1, size=edges)) % TINY_ROWS
        self.edge_weights = weights
        self.edges_src = _npy(
            os.path.join(root, "edges-src.i32.npy"), src.astype(np.int32)
        )
        self.edges_dst = _npy(
            os.path.join(root, "edges-dst.i32.npy"), dst.astype(np.int32)
        )
        self.edges_wts = _npy(os.path.join(root, "edges-wts.f32.npy"), weights)
        header_path = os.path.join(root, "edges-header.npz")
        np.savez(
            header_path,
            n_nodes=np.asarray(TINY_ROWS, dtype=np.int64),
            k=np.asarray(TINY_K, dtype=np.int64),
            directed_edges=np.asarray(edges, dtype=np.int64),
        )
        self.edges_header = expected_input_signature(header_path)
        self.edges = edges

    def inheritance(self) -> dict:
        return {
            "note": "tiny",
            "graph": {"ids": self.graph_ids, "cosines": self.graph_cos},
        }


def _patch(monkeypatch, world: _TinyWorld) -> None:
    monkeypatch.setattr(nodes, "ROWS", TINY_ROWS)
    monkeypatch.setattr(nodes, "TRUTH_PROBE_ROWS", TINY_PROBE)
    monkeypatch.setattr(nodes, "LABEL_BLOCK", 16)
    monkeypatch.setattr(nodes, "CHUNK_ROOT", world.chunk_root)
    monkeypatch.setattr(nodes, "EMBEDDING_ROOT", world.embedding_root)
    monkeypatch.setattr(nodes, "EXCLUDED_SHARDS", frozenset())
    monkeypatch.setattr(nodes, "COMPOSITION", ((TINY_CORPUS, TINY_ROWS),))
    monkeypatch.setattr(nodes, "TEXT_SAMPLE_PAIRS", 3)
    monkeypatch.setattr(nodes, "SAMPLER_BLOCK_EDGES", 4_096)
    monkeypatch.setattr(nodes, "SAMPLER_MIN_DRAWS_PER_S", 1.0)
    monkeypatch.setattr(nodes, "R0243_DIRECTED_EDGES", world.edges)
    weights = np.asarray(world.edge_weights, dtype=np.float64)
    monkeypatch.setattr(nodes, "R0243_TOTAL_WEIGHT", float(weights.sum()))
    monkeypatch.setattr(nodes, "R0243_WEIGHT_MAX", float(weights.max()))
    monkeypatch.setattr(nodes, "R0243_WEIGHT_MIN", float(weights.min()))
    monkeypatch.setattr(
        nodes, "R0243_ENTRIES_AT_OR_ABOVE_ONE",
        int(np.count_nonzero(weights >= 1.0)),
    )
    monkeypatch.setattr(
        nodes, "verify_inheritance", lambda job: world.inheritance()
    )


@pytest.fixture()
def scratch():
    os.makedirs(SMOKE_ROOT, exist_ok=True)
    root = os.path.join(SMOKE_ROOT, f"round0244-{uuid.uuid4().hex}")
    os.makedirs(root)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def world(scratch) -> _TinyWorld:
    return _TinyWorld(os.path.join(scratch, "world"))


ACTIVE = {"manifest": {"round_id": "0244", "release_sha": RELEASE}}


def _did_job(world: _TinyWorld, output: str) -> dict:
    return {
        "action": nodes.DID_ACTION,
        "outputs": [output],
        "r0242_probe_strict_recall": world.strict_recall,
        "r0238_strict_reachability": world.reachability,
        "r0242_probe_builder_missing_edges": world.strict_builder,
        "r0243_probe_tie_aware_builder_missing_edges": world.tie_builder,
        "truth_cos": world.truth_cos,
    }


def test_did_entry_path_runs_end_to_end(monkeypatch, scratch, world) -> None:
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-did")
    nodes.run_job(ACTIVE, _did_job(world, output))
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.DID_FILE), label="tiny did"
    )
    assert receipt["schema"] == nodes.DID_SCHEMA
    assert receipt["displacement_measured"] is False
    assert receipt["r0238_reachability_discharge"]["agrees"] is True
    assert receipt["registration"]["selected_design"]["labellings"] == 252
    assert receipt["requirement"]["fits_in_the_rung_cap"] is False
    assert receipt["populations"]["genuine_rows_total"] == 3
    assert receipt["populations"]["tie_forgiven_rows_total"] == 3
    assert os.path.exists(
        os.path.join(output, "vectors", "did-placebo-a-rows.i64.npy")
    )


def test_did_node_fails_closed_on_a_wrong_reachability_vector(
    monkeypatch, scratch, world
) -> None:
    """The discharge is a CONSUMPTION, not a declaration: a vector that does
    not reproduce R0242's sealed builder-missing bytes stops the round."""
    _patch(monkeypatch, world)
    wrong = np.full(TINY_PROBE, (TINY_K - 3) / TINY_K, dtype=np.float64)
    path = os.path.join(world.root, "wrong-reachability.f64.npy")
    signature = _npy(path, wrong)
    job = _did_job(world, os.path.join(scratch, "artifacts-did-bad"))
    job["r0238_strict_reachability"] = signature
    with pytest.raises(nodes.Round0244Error):
        nodes.run_job(ACTIVE, job)


def test_watchdog_entry_path_runs_end_to_end(monkeypatch, scratch, world) -> None:
    """Drives the positive control, the sort, UMAP's own law and the stripe
    loop, then compares the thread's peak against the boundary column."""
    _patch(monkeypatch, world)
    expected = _reference_symmetrisation(world, os.path.join(scratch, "ref"))
    monkeypatch.setattr(
        nodes, "R0243_SIGMAS_MEAN", expected["sigmas_mean"]
    )
    monkeypatch.setattr(nodes, "R0243_RHOS_MEAN", expected["rhos_mean"])
    monkeypatch.setattr(
        nodes, "R0243_DIRECTED_EDGES", expected["directed_edges"]
    )
    output = os.path.join(scratch, "artifacts-watchdog")
    nodes.run_job(ACTIVE, {"action": nodes.WATCHDOG_ACTION, "outputs": [output]})
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.WATCHDOG_FILE), label="tiny watchdog"
    )
    assert receipt["schema"] == nodes.WATCHDOG_SCHEMA
    assert receipt["positive_control"]["holds"] is True
    assert receipt["positive_control"]["watchdog_fired"] is True
    assert receipt["stage_reproduction"]["directed_edges_agree"] is True
    assert receipt["host_watchdog"]["sampling_thread"] is True
    assert receipt["host_watchdog"]["samples"] >= 1
    assert receipt["host_watchdog"]["fired"] is False
    assert receipt["edges_republished"] is False
    assert not os.path.exists(os.path.join(output, ".fuzzy"))


def test_watchdog_node_refuses_a_stage_it_did_not_reproduce(
    monkeypatch, scratch, world
) -> None:
    """A peak measured on a stage that is not R0243's stage measures nothing."""
    _patch(monkeypatch, world)
    monkeypatch.setattr(nodes, "R0243_DIRECTED_EDGES", 1)
    with pytest.raises(nodes.Round0244Error):
        nodes.run_job(
            ACTIVE,
            {
                "action": nodes.WATCHDOG_ACTION,
                "outputs": [os.path.join(scratch, "artifacts-watchdog-bad")],
            },
        )


def _reference_symmetrisation(world: _TinyWorld, out_dir: str) -> dict:
    """The same imported functions, run test-side, to learn the tiny world's
    own moments — the constants the node then has to reproduce."""
    import umap.umap_ as umap_api

    from experiments.round0238_nodes import (
        _blocked_descending_sort,
        _fuzzy_symmetrise_blocked,
    )

    os.makedirs(out_dir, exist_ok=True)
    ids = np.load(world.graph_ids_path, mmap_mode="r")
    cos = np.load(world.graph_cos_path, mmap_mode="r")
    ids_sorted, cos_sorted = _blocked_descending_sort(
        ids=ids, cosines=cos, rows=TINY_ROWS
    )
    dists = np.maximum((1.0 - cos_sorted).astype(np.float32), 0.0)
    fuzzy = _fuzzy_symmetrise_blocked(
        knn_indices=ids_sorted, knn_dists=dists, rows=TINY_ROWS, k=TINY_K,
        umap_api=umap_api, out_dir=out_dir,
    )
    out = {
        "directed_edges": int(fuzzy["directed_edges"]),
        "sigmas_mean": float(fuzzy["sigmas_mean"]),
        "rhos_mean": float(fuzzy["rhos_mean"]),
    }
    for key in ("src", "dst", "weights"):
        fuzzy.pop(key, None)
    shutil.rmtree(str(fuzzy["scratch"]), ignore_errors=True)
    return out


def test_sampler_entry_path_runs_end_to_end(monkeypatch, scratch, world) -> None:
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-sampler")
    nodes.run_job(ACTIVE, {
        "action": nodes.SAMPLER_ACTION,
        "outputs": [output],
        "edges_header": world.edges_header,
        "edges_src": world.edges_src,
        "edges_dst": world.edges_dst,
        "edges_wts": world.edges_wts,
        "draws": 200_000,
    })
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.SAMPLER_FILE), label="tiny sampler"
    )
    assert receipt["schema"] == nodes.SAMPLER_SCHEMA
    assert receipt["loads_as_a_sampling_distribution"] is True
    assert receipt["binding_to_r0243"]["is_r0243s_distribution"] is True
    assert receipt["fidelity"]["holds"] is True
    assert receipt["mis_sampler_control"]["rejected"] is True
    assert receipt["endpoint_gather"]["in_range"] is True


def test_sampler_node_refuses_a_header_from_another_graph(
    monkeypatch, scratch, world
) -> None:
    _patch(monkeypatch, world)
    monkeypatch.setattr(nodes, "R0243_DIRECTED_EDGES", world.edges + 1)
    with pytest.raises(nodes.Round0244Error):
        nodes.run_job(ACTIVE, {
            "action": nodes.SAMPLER_ACTION,
            "outputs": [os.path.join(scratch, "artifacts-sampler-bad")],
            "edges_header": world.edges_header,
            "edges_src": world.edges_src,
            "edges_dst": world.edges_dst,
            "edges_wts": world.edges_wts,
            "draws": 20_000,
        })


def _text_job(world: _TinyWorld, output: str) -> dict:
    return {
        "action": nodes.TEXT_ACTION,
        "outputs": [output],
        "substrate_array": world.substrate,
        "provenance": world.provenance,
        "r0242_primary_cluster": world.labels,
        "probe_query_rows": world.probe_rows,
        "r0242_probe_cluster": world.probe_cluster,
        "r0242_probe_builder_missing_edges": world.strict_builder,
        "r0243_probe_tie_aware_builder_missing_edges": world.tie_builder,
        "truth_ids": world.truth_ids,
        "truth_cos": world.truth_cos,
    }


def test_text_entry_path_runs_end_to_end(monkeypatch, scratch, world) -> None:
    """Resolves provenance to a parquet, reads the chunks and VERIFIES the
    binding by re-embedding — the check that makes the excerpts evidence."""
    _patch(monkeypatch, world)
    output = os.path.join(scratch, "artifacts-text")
    nodes.run_job(ACTIVE, _text_job(world, output))
    receipt = prompt_contract.read_sealed(
        os.path.join(output, nodes.TEXT_FILE), label="tiny text"
    )
    assert receipt["schema"] == nodes.TEXT_SCHEMA
    assert receipt["text_binding_holds"] is True
    assert receipt["minimum_re_embedded_cosine"] >= 0.999
    assert receipt["cluster_rows"] == 20
    assert receipt["resolved_pairs"] >= 1
    for pair in receipt["pairs"]:
        assert pair["cosine_gap_to_the_tie_threshold"] <= pair["tie_tolerance"]
        assert pair["missed_truth_excerpt"]
        assert pair["tie_substitute_excerpt"]
        assert "cosine_missed_to_substitute" in pair


def test_text_node_fails_closed_when_the_text_is_not_the_row(
    monkeypatch, scratch, world
) -> None:
    """A wrong row-to-text mapping must stop the node, not produce excerpts."""
    _patch(monkeypatch, world)
    import numpy as _np

    shifted = _np.roll(world.substrate_matrix, 3, axis=0)
    path = os.path.join(world.root, "shifted-substrate.f32.npy")
    signature = _npy(path, shifted)
    job = _text_job(world, os.path.join(scratch, "artifacts-text-bad"))
    job["substrate_array"] = signature
    with pytest.raises(nodes.Round0244Error):
        nodes.run_job(ACTIVE, job)


def test_chunk_parquet_resolves_by_name_then_by_position(
    monkeypatch, scratch, world
) -> None:
    """`starcoderdata` names its parquets `000_000500000` against
    `data-00000-of-00020.npy`, with the same count and order; the pile's
    parquet directory is a superset of its embedded shards, so name matching
    is the only correct rule there. Both paths are exercised, and a count
    disagreement is refused rather than guessed."""
    _patch(monkeypatch, world)
    shards = nodes._corpus_shards(TINY_CORPUS)
    by_name = nodes._chunk_parquet(TINY_CORPUS, shards[0], shard_index=0)
    assert by_name.endswith(f"{TINY_SHARD}.parquet")

    directory = os.path.dirname(by_name)
    renamed = os.path.join(directory, "000_000500000.parquet")
    os.rename(by_name, renamed)
    assert nodes._chunk_parquet(TINY_CORPUS, shards[0], shard_index=0) == renamed

    shutil.copyfile(renamed, os.path.join(directory, "001_001000000.parquet"))
    with pytest.raises(nodes.Round0244Error):
        nodes._chunk_parquet(TINY_CORPUS, shards[0], shard_index=0)


def test_text_node_refuses_a_parquet_whose_row_count_differs(
    monkeypatch, scratch, world
) -> None:
    """A positional fallback is an assumption; this is what stops it being one."""
    _patch(monkeypatch, world)
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = os.path.join(
        world.chunk_root, TINY_CORPUS[: -len(nodes.EMBEDDING_SUFFIX)], "train"
    )
    pq.write_table(
        pa.table({"chunk_text": world.texts[:-1]}),
        os.path.join(directory, f"{TINY_SHARD}.parquet"),
        row_group_size=8,
    )
    with pytest.raises(nodes.Round0244Error):
        nodes.run_job(
            ACTIVE, _text_job(world, os.path.join(scratch, "artifacts-text-rows"))
        )


def test_run_job_refuses_an_unknown_action(scratch) -> None:
    with pytest.raises(nodes.Round0244Error):
        nodes.run_job(ACTIVE, {"action": "not-a-round-0244-node"})
