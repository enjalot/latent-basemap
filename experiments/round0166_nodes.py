"""Execute the prompted-English 8M scale rung for Round 0166."""
from __future__ import annotations

import gc
import hashlib
import math
import os
import random
import resource
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0104_training import L2NormalizedArray
from basemap.round0108_evaluation import exact_cosine_topk, exact_reference_copy_mask
from basemap.round0165_frozen_prefix_population import (
    CAPABILITY as POPULATION_CAPABILITY,
    HOST_CAPABILITY as POPULATION_HOST_CAPABILITY,
    SCHEMA as POPULATION_SCHEMA,
)
from basemap.round0166_prompted_8m import (
    CAPABILITY,
    DIMENSION,
    GRAPH_K,
    GRAPH_MEAN_RECALL_FLOOR,
    GRAPH_NLIST,
    GRAPH_NPROBE,
    GRAPH_NPROBE_GRID,
    GRAPH_P10_RECALL_FLOOR,
    GRAPH_QUALITY_ROWS,
    GRAPH_QUALITY_SEED,
    GRAPH_TRAIN_ROWS,
    GRAPH_TRAIN_SEED,
    HOST_RSS_LIMIT_GIB,
    QUERY_CANDIDATES,
    QUERY_ROWS,
    ROUND_ID,
    SEED,
    SUCCESSFUL_UPDATES,
    Round0166Error,
    ScalePromptTrainingInput,
    scale_decision,
    scale_train_config,
)
from basemap.round0160_prompted_seed_family import (
    CAPABILITY as FAMILY_CAPABILITY,
    METRICS,
    ROWS as MATCHED_ROWS,
    metric_view,
)
from basemap.round0161_prompted_gate_registration import (
    CAPABILITY as GATE_CAPABILITY,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0113_nodes as prompt_nodes
from experiments.round0163_nodes import _iter_texts, _read_sealed, _validate_signature


GRAPH_SCHEMA = "round0166-prompted-8m-fuzzy-graph-v1"
QUERY_SCHEMA = "round0166-prompted-8m-heldout-query-v1"
TRAIN_SCHEMA = "round0166-prompted-8m-train-receipt-v1"
EVALUATION_SCHEMA = "round0166-prompted-8m-scale-evaluation-v1"
PRODUCTION_CONFIG_SCHEMA = "round0166-prompted-8m-production-config-v1"
GRAPH_INDEX_DESCRIPTION = "GPU IndexIVFFlat/IP fp32 vector storage"
GRAPH_REFERENCE_ROW_ORDER = "R0165 frozen-prefix prompted compact order"
GRAPH_REFERENCE_ANCHOR_NAMESPACE = "R0165 compact IDs"
GRAPH_SHARD_ROWS = 4_000_000
# Descendant rounds may consume an already-reviewed graph without pretending
# that they rebuilt it.  The defaults preserve every historical R0166 call.
GRAPH_SOURCE_ROUND_ID = ROUND_ID
GRAPH_BUILT_IN_ROUND = True
# Later scale rounds may bind a freshly derived population whose receipt cannot
# exist when the queue itself is sealed.  Keep the historical R0166 reader as
# the default, while allowing a round-local runtime reader to authenticate a
# dependency-produced receipt.
POPULATION_READER: Callable[[Mapping[str, Any]], tuple[dict[str, Any], dict[str, Any]]] | None = None
# R0166 deliberately rejected the 2M family.  Composition-controlled ladders
# include a 1.988M mixed rung, so descendants may lower this bound explicitly.
MIN_SCALE_ROWS_EXCLUSIVE = 2_000_000
# Optional graph-time references let descendants reuse the already-materialized
# normalized matrix instead of paying another multi-gigabyte materialization.
GRAPH_EXTRA_REFERENCE_BUILDER: Callable[..., dict[str, Any]] | None = None

REQUIRED_TRAIN_CHECKS = {
    "exact_update_closure",
    "zero_numerical_skips",
    "no_pipeline_stamp_drift",
    "endpoint_rows_match_updates",
    "weighted_rejection_accounting_closes",
}


def _train_checks_close(value: Any) -> bool:
    """Require the exact nonempty execution-check set; never pass vacuously."""
    return (
        isinstance(value, Mapping)
        and set(value) == REQUIRED_TRAIN_CHECKS
        and all(value[key] is True for key in REQUIRED_TRAIN_CHECKS)
    )


def _faiss_gpu_options(faiss: Any) -> Any:
    """Return the registered fp32 GPU-index storage options."""
    return prompt_nodes._faiss_gpu_options(faiss)


def _merge_ann_topk(
    left_sims: np.ndarray,
    left_ids: np.ndarray,
    right_sims: np.ndarray,
    right_ids: np.ndarray,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge two per-row ANN result tables with deterministic global-ID ties."""
    if (
        left_sims.shape != left_ids.shape
        or right_sims.shape != right_ids.shape
        or left_sims.shape[0] != right_sims.shape[0]
        or left_sims.shape[1] < k
        or right_sims.shape[1] < k
    ):
        raise Round0166Error("sharded ANN merge geometry changed")
    sims = np.concatenate((left_sims, right_sims), axis=1).astype(
        np.float32, copy=False
    )
    ids = np.concatenate((left_ids, right_ids), axis=1).astype(
        np.int64, copy=False
    )
    # FAISS returns id -1 with a sentinel score when a query's probed lists
    # hold fewer than k vectors *in this shard*.  On a homogeneous population
    # (R0166/R0171 English) that never happens, but on a population whose
    # compact order blocks by corpus and language a contiguous shard can hold
    # zero vectors in the lists a given query probes.  Such a slot carries no
    # candidate: it must be excluded from the merge, never ranked as one.
    # Every slot that does carry a candidate must still be finite.
    missing = ids < 0
    if not np.isfinite(sims[~missing]).all():
        raise Round0166Error("sharded ANN merge received invalid candidates")
    absent = np.iinfo(np.int64).max
    sims = np.where(missing, -np.inf, sims)
    ids = np.where(missing, absent, ids)
    # The merged table is only 2k wide (100 columns for the registered graph),
    # so sort the complete candidate set.  Besides being cheap at this width,
    # this avoids argpartition choosing an arbitrary member of a score tie at
    # the kth boundary.
    order = np.lexsort((ids, -sims), axis=1)[:, :k]
    merged_sims = np.take_along_axis(sims, order, axis=1)
    merged_ids = np.take_along_axis(ids, order, axis=1)
    # Re-emit unfilled slots as the -1 the caller's completeness guard checks,
    # so a row that never reaches k real candidates across every shard still
    # fails closed instead of carrying a sentinel neighbour.
    unfilled = merged_ids == absent
    if unfilled.any():
        merged_ids = np.where(unfilled, -1, merged_ids)
        merged_sims = np.where(unfilled, -np.inf, merged_sims)
    return merged_sims, merged_ids


def _signature(path: str, *, label: str) -> dict[str, Any]:
    try:
        return expected_input_signature(path)
    except Exception as error:
        raise Round0166Error(f"{label} is unavailable or changed") from error


def _read_population(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if POPULATION_READER is not None:
        return POPULATION_READER(job)
    signature = dict(job["population_receipt"])
    population = _read_sealed(signature, label="accepted R0165 population")
    capabilities = population.get("capabilities")
    proofs = population.get("proofs") or {}
    derivation = population.get("derivation") or {}
    if (
        population.get("schema") != POPULATION_SCHEMA
        or population.get("round_id") != "0165"
        or population.get("outcome") != "prompted-8m-frozen-prefix-population-qualified"
        or population.get("q2_population_released") is not True
        or capabilities != [POPULATION_CAPABILITY, POPULATION_HOST_CAPABILITY]
        or int(population.get("retained_rows", -1)) != 7_952_419
        or int(population.get("excluded_rows", -1)) != 47_581
        or int(population.get("source_rows", -1)) != 8_000_000
        or int(population.get("dimension", -1)) != DIMENSION
        or population.get("dtype") != "<f2"
        or int(derivation.get("dropped_prompted_only_prefix_rows", -1)) != 7
        or int(derivation.get("added_over_r0163_rows", -1)) != 205
        or proofs.get("prefix_byte_exact") is not True
        or proofs.get("mapping_is_r0164_subset") is not True
        or proofs.get("mapping_is_strict_r0163_superset") is not True
        or proofs.get("raw_unprompted_relation_used_for_extension") is not False
    ):
        raise Round0166Error("accepted R0165 population contract changed")
    for key in ("mapping", "document_compact", "source_text_hash_index"):
        prompt_contract.verify_signature(
            population[key], label=f"accepted R0165 {key}"
        )
    return population, signature


def _open_source(population: Mapping[str, Any]) -> np.memmap:
    rows = int(population["retained_rows"])
    signature = population["document_compact"]
    path = prompt_contract.verify_signature(signature, label="R0165 compact document source")
    source = np.memmap(path, mode="r", dtype="<f2", shape=(rows, DIMENSION))
    if source.nbytes != int(signature["bytes"]):
        raise Round0166Error("R0165 compact source byte count changed")
    return source


def _without_self(rows: np.ndarray, ids: np.ndarray, width: int) -> np.ndarray:
    return prompt_nodes._without_self(rows, ids, width)


def _recall_rows(observed: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return prompt_nodes._recall_rows(observed, truth)


def _data_identity(population: Mapping[str, Any]) -> dict[str, Any]:
    source = population["document_compact"]
    return {
        "kind": "ordered_shards",
        "shape": [int(population["retained_rows"]), DIMENSION],
        "dtype": np.dtype("<f2").str,
        "shards": [{
            "position": 0,
            "name": os.path.basename(str(source["canonical_path"])),
            "bytes": int(source["bytes"]),
            "sha256": str(source["sha256"]),
        }],
    }


def run_build_graph(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import faiss
    import torch
    import umap.umap_ as umap_api
    from basemap.panel_v2 import build_hiD_reference, sample_anchors, save_hiD_reference
    from experiments.score_complete_panel import frozen_centroids

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0166Error("R0166 graph handler received another queue")
    population, population_signature = _read_population(job)
    rows = int(population["retained_rows"])
    source = _open_source(population)
    output = create_fresh_directory(str(job["outputs"][0]), label="R0166 graph output")
    started = time.monotonic()
    materialize_started = time.monotonic()
    X = prompt_nodes._materialize_normalized(source)
    materialize_seconds = time.monotonic() - materialize_started
    del source
    gc.collect()

    quality_ids = np.sort(
        np.random.RandomState(GRAPH_QUALITY_SEED)
        .choice(rows, GRAPH_QUALITY_ROWS, replace=False)
        .astype(np.int64)
    )
    truth_raw, truth_receipt = exact_cosine_topk(
        np.ascontiguousarray(X[quality_ids]), X, k=GRAPH_K, candidate_block_rows=50_000
    )
    truth = _without_self(truth_raw, quality_ids, GRAPH_K - 1)
    torch.cuda.empty_cache()

    train_rows = np.sort(
        np.random.RandomState(GRAPH_TRAIN_SEED)
        .choice(rows, GRAPH_TRAIN_ROWS, replace=False)
        .astype(np.int64)
    )
    quantizer = faiss.IndexFlatIP(DIMENSION)
    cpu_index = faiss.IndexIVFFlat(
        quantizer, DIMENSION, GRAPH_NLIST, faiss.METRIC_INNER_PRODUCT
    )
    cpu_index.cp.seed = GRAPH_TRAIN_SEED
    cpu_index.cp.niter = 25
    cpu_index.cp.spherical = True
    gpu_resource = faiss.StandardGpuResources()
    gpu_resource.setTempMemory(256 << 20)
    index = faiss.index_cpu_to_gpu(
        gpu_resource, 0, cpu_index, _faiss_gpu_options(faiss)
    )
    train_started = time.monotonic()
    index.train(np.ascontiguousarray(X[train_rows]))
    train_seconds = time.monotonic() - train_started
    trained_cpu = faiss.index_gpu_to_cpu(index)
    if trained_cpu.is_trained is not True or int(trained_cpu.ntotal) != 0:
        raise Round0166Error("R0171 trained empty IVF template changed")
    del index, cpu_index, quantizer, gpu_resource
    torch.cuda.empty_cache()
    gc.collect()

    neighbors = np.full((rows, GRAPH_K), -1, dtype=np.int32)
    distances = np.full((rows, GRAPH_K), -np.inf, dtype=np.float32)
    quality_best: dict[int, tuple[np.ndarray, np.ndarray] | None] = {
        nprobe: None for nprobe in GRAPH_NPROBE_GRID
    }
    quality_wall = {nprobe: 0.0 for nprobe in GRAPH_NPROBE_GRID}
    add_seconds = 0.0
    search_seconds = 0.0
    shard_receipts: list[dict[str, Any]] = []
    quality_queries = np.ascontiguousarray(X[quality_ids])
    for shard_start in range(0, rows, GRAPH_SHARD_ROWS):
        shard_stop = min(shard_start + GRAPH_SHARD_ROWS, rows)
        gpu_resource = faiss.StandardGpuResources()
        gpu_resource.setTempMemory(256 << 20)
        index = faiss.index_cpu_to_gpu(
            gpu_resource, 0, trained_cpu, _faiss_gpu_options(faiss)
        )
        add_started = time.monotonic()
        for start in range(shard_start, shard_stop, 25_000):
            stop = min(start + 25_000, shard_stop)
            index.add_with_ids(
                np.ascontiguousarray(X[start:stop]),
                np.arange(start, stop, dtype=np.int64),
            )
        shard_add_s = time.monotonic() - add_started
        add_seconds += shard_add_s
        if int(index.ntotal) != shard_stop - shard_start:
            raise Round0166Error("R0171 sharded IVF row count changed")

        for nprobe in GRAPH_NPROBE_GRID:
            index.nprobe = nprobe
            cell_started = time.monotonic()
            shard_sims, shard_ids = index.search(quality_queries, GRAPH_K)
            quality_wall[nprobe] += time.monotonic() - cell_started
            current = quality_best[nprobe]
            quality_best[nprobe] = (
                (
                    shard_sims.astype(np.float32, copy=False),
                    shard_ids.astype(np.int64, copy=False),
                )
                if current is None
                else _merge_ann_topk(
                    current[0],
                    current[1],
                    shard_sims,
                    shard_ids,
                    k=GRAPH_K,
                )
            )

        index.nprobe = GRAPH_NPROBE
        shard_search_started = time.monotonic()
        for start in range(0, rows, 16_384):
            stop = min(start + 16_384, rows)
            shard_sims, shard_ids = index.search(
                np.ascontiguousarray(X[start:stop]), GRAPH_K
            )
            if shard_start == 0:
                distances[start:stop] = shard_sims.astype(np.float32, copy=False)
                neighbors[start:stop] = shard_ids.astype(np.int32, copy=False)
            else:
                merged_sims, merged_ids = _merge_ann_topk(
                    distances[start:stop],
                    neighbors[start:stop],
                    shard_sims,
                    shard_ids,
                    k=GRAPH_K,
                )
                distances[start:stop] = merged_sims
                neighbors[start:stop] = merged_ids.astype(np.int32, copy=False)
        shard_search_s = time.monotonic() - shard_search_started
        search_seconds += shard_search_s
        shard_receipts.append({
            "start": shard_start,
            "stop": shard_stop,
            "rows": shard_stop - shard_start,
            "add_s": shard_add_s,
            "full_search_s": shard_search_s,
            "ntotal": int(index.ntotal),
        })
        del index, gpu_resource
        torch.cuda.empty_cache()
        gc.collect()

    if np.any(neighbors < 0) or np.any(neighbors >= rows) or not np.isfinite(distances).all():
        raise Round0166Error("R0171 sharded full graph search returned invalid rows")
    np.maximum(0.0, 1.0 - distances, out=distances)

    cells: dict[str, Any] = {}
    selected_observed: np.ndarray | None = None
    for nprobe in GRAPH_NPROBE_GRID:
        merged = quality_best[nprobe]
        if merged is None:
            raise Round0166Error("R0171 sharded quality search is incomplete")
        observed = _without_self(merged[1], quality_ids, GRAPH_K - 1)
        recalls = _recall_rows(observed, truth)
        passed = bool(
            recalls.mean() >= GRAPH_MEAN_RECALL_FLOOR
            and np.percentile(recalls, 10) >= GRAPH_P10_RECALL_FLOOR
        )
        cell_wall = quality_wall[nprobe]
        cells[str(nprobe)] = {
            "mean_recall_at_49": float(recalls.mean()),
            "p10_recall_at_49": float(np.percentile(recalls, 10)),
            "wall_s": cell_wall,
            "queries_per_s": GRAPH_QUALITY_ROWS / cell_wall,
            "passed": passed,
        }
        if nprobe == GRAPH_NPROBE:
            selected_observed = observed.copy()
    fixed = cells[str(GRAPH_NPROBE)]
    if fixed["passed"] is not True or selected_observed is None:
        raise Round0166Error("R0171 fixed-nprobe sharded graph search did not qualify")

    del trained_cpu, quality_queries, quality_best
    torch.cuda.empty_cache()
    gc.collect()

    fuzzy_started = time.monotonic()
    graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
        X,
        n_neighbors=GRAPH_K,
        random_state=np.random.RandomState(SEED),
        metric="cosine",
        knn_indices=neighbors,
        knn_dists=distances,
    )
    coo = graph.tocoo()
    sources = np.asarray(coo.row, dtype=np.int32)
    targets = np.asarray(coo.col, dtype=np.int32)
    weights = np.asarray(coo.data, dtype=np.float32)
    fuzzy_seconds = time.monotonic() - fuzzy_started
    if (
        len(sources) <= rows * (GRAPH_K - 1)
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
        or np.any(weights > 1)
    ):
        raise Round0166Error("R0166 fuzzy graph arrays are invalid")
    directed_edge_count = int(len(sources))
    graph_path = os.path.join(output, "edges-k50-fuzzy.npz")
    atomic_save_new_npz(
        graph_path,
        immutable=True,
        compressed=False,
        sources=sources,
        targets=targets,
        weights=weights,
        n_nodes=np.asarray(rows, dtype=np.int64),
        k=np.asarray(GRAPH_K, dtype=np.int64),
    )
    topology_path = os.path.join(output, "topology-probe.npz")
    atomic_save_new_npz(
        topology_path,
        immutable=True,
        compressed=False,
        anchor_compact_ids=quality_ids,
        exact_neighbors=truth,
        qualified_ann_neighbors=selected_observed,
    )
    del neighbors, distances, graph, coo, sources, targets, weights
    gc.collect()
    centroid_root = create_fresh_directory(
        os.path.join(output, "centroids"), label="R0166 native centroids"
    )
    centroid_started = time.monotonic()
    centroids = frozen_centroids(X, (256, 1024), centroid_root, seed=0, iters=25)
    centroid_seconds = time.monotonic() - centroid_started
    centroid_signatures = {
        str(k): _signature(
            os.path.join(centroid_root, f"centroids_k{k}.npy"),
            label=f"R0166 native k{k} centroids",
        )
        for k in (256, 1024)
    }
    cfg = prompt_contract.panel_config()
    anchors = sample_anchors(rows, cfg)
    reference_identity = {
        "data_identity": _data_identity(population),
        "convention": {
            "row_order": GRAPH_REFERENCE_ROW_ORDER,
            "distance": "cosine via fp32-L2-normalized squared L2",
            "self_exclusion": True,
            "anchor_namespace": GRAPH_REFERENCE_ANCHOR_NAMESPACE,
            "embedding_prompt": "document",
        },
    }
    reference_started = time.monotonic()
    reference = build_hiD_reference(
        X,
        anchors,
        cfg,
        centroids_by_k=centroids,
        **reference_identity,
    )
    reference_path = os.path.join(output, "high-d-reference.npz")
    save_hiD_reference(reference, reference_path)
    reference_seconds = time.monotonic() - reference_started
    extra_references = (
        GRAPH_EXTRA_REFERENCE_BUILDER(
            output=output,
            X=X,
            population=population,
            population_signature=population_signature,
            config=cfg,
        )
        if GRAPH_EXTRA_REFERENCE_BUILDER is not None
        else {}
    )
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0166Error(
            f"R0166 graph peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    manifest = prompt_contract.seal({
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "retained_rows": rows,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "directed_edge_count": directed_edge_count,
        "graph": _signature(graph_path, label="R0166 fuzzy graph"),
        "population": population_signature,
        "compact_mapping": population["mapping"],
        "source": population["document_compact"],
        "search_qualification": {
            "index": GRAPH_INDEX_DESCRIPTION,
            "selected_nprobe": GRAPH_NPROBE,
            "execution": {
                "shard_rows": GRAPH_SHARD_ROWS,
                "shards": shard_receipts,
                "coarse_quantizer": "one shared trained IVF8192 template",
                "merge": (
                    "exact global top-k over every row-disjoint fp32-IVF "
                    "shard, ordered by similarity descending then global ID ascending"
                ),
            },
            "cells": cells,
            "training_rows_sha256": ordered_array_sha256(train_rows),
            "quality_rows_sha256": ordered_array_sha256(quality_ids),
            "exact_truth": truth_receipt,
            "mean_recall_floor": GRAPH_MEAN_RECALL_FLOOR,
            "p10_recall_floor": GRAPH_P10_RECALL_FLOOR,
        },
        "topology_probe": _signature(topology_path, label="R0166 topology probe"),
        "high_d_reference": _signature(
            reference_path, label="R0166 native high-D reference"
        ),
        "high_d_reference_key": reference["key"],
        "high_d_reference_content_sha256": reference["content_sha256"],
        "centroids": centroid_signatures,
        "reference_identity": reference_identity,
        "reference_recipe": {
            "panel_config": dict(vars(cfg)),
            "centroid_seed": 0,
            "centroid_iterations": 25,
            "centroid_ks": [256, 1024],
        },
        "comparison_references": extra_references,
        "graph_law": {
            "same_as_r0115": True,
            "search_neighbors_including_self": GRAPH_K,
            "nonself_degree": GRAPH_K - 1,
            "sharded_search_equivalence": (
                "same shared coarse quantizer and nprobe on every disjoint row "
                "shard; exact global top-k merge by similarity then global ID"
            ),
            "fuzzy_symmetrization": "UMAP fuzzy_simplicial_set",
            "positive_weight_semantics": "fuzzy membership strength",
        },
        "performance": {
            "materialize_s": materialize_seconds,
            "ivf_train_s": train_seconds,
            "ivf_add_s": add_seconds,
            "full_search_s": search_seconds,
            "fuzzy_s": fuzzy_seconds,
            "centroids_s": centroid_seconds,
            "high_d_reference_s": reference_seconds,
            "peak_rss_gib": peak_rss_gib,
            "total_wall_s": time.monotonic() - started,
        },
        "training_performed": False,
    })
    atomic_write_new_json(os.path.join(output, "graph-manifest.json"), manifest, immutable=True)
    del (
        X,
        centroids,
        reference,
    )
    gc.collect()


def _gather_canonical_document_rows(
    layout: Mapping[str, Any], rows: np.ndarray
) -> np.ndarray:
    selected = np.asarray(rows, dtype=np.int64)
    output = np.empty((len(selected), DIMENSION), dtype=np.float16)
    filled = np.zeros(len(selected), dtype=bool)
    for item in layout.get("chunks") or []:
        start, stop = (int(value) for value in item["canonical_row_range"])
        left = int(np.searchsorted(selected, start, side="left"))
        right = int(np.searchsorted(selected, stop, side="left"))
        if right <= left:
            continue
        signature = item["staged_output"]
        path = prompt_contract.verify_signature(signature, label="R0166 heldout chunk")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        local = selected[left:right] - start
        output[left:right] = values[local]
        filled[left:right] = True
    if not np.all(filled):
        raise Round0166Error("R0166 heldout document gather did not close")
    return output


def _full_text_layouts(
    r0116: Mapping[str, Any], r0120: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return the complete 9.126M text layout, not R0163's clipped 8M view."""
    output: list[dict[str, Any]] = []
    cursor = 0
    for round_id, manifest in (("0116", r0116), ("0120", r0120)):
        for item in manifest.get("source_layout") or []:
            if round_id == "0120":
                start = int(item["r0087_global_row_start"])
                stop = int(item["r0087_global_row_stop"])
            else:
                start = int(item["corpus_global_row_start"])
                stop = int(item["corpus_global_row_stop"])
            if start != cursor or stop <= start:
                raise Round0166Error("R0166 full text layout is not gap-free")
            output.append({
                "canonical_start": start,
                "canonical_stop": stop,
                "shard_start": int(item["shard_row_start"]),
                "shard_stop": int(item["shard_row_start"]) + stop - start,
                "shard_rows": int(item["shard_rows"]),
                "text": dict(item["text"]),
                "text_column": str(item["text_column"]),
            })
            cursor = stop
    if cursor != 9_126_376:
        raise Round0166Error("R0166 full text layout row count changed")
    return output


def _query_payload_inputs(
    layout: Mapping[str, Any], text_layout: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    start = 8_000_000
    stop = start + QUERY_CANDIDATES
    selected: list[dict[str, Any]] = []
    for item in layout.get("chunks") or []:
        left, right = (int(value) for value in item["canonical_row_range"])
        if max(left, start) < min(right, stop):
            selected.append(dict(item["staged_output"]))
    for item in text_layout:
        if max(int(item["canonical_start"]), start) < min(
            int(item["canonical_stop"]), stop
        ):
            selected.append(dict(item["text"]))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for item in selected:
        key = (str(item["canonical_path"]), int(item["bytes"]), str(item["sha256"]))
        if key not in seen:
            unique.append(item)
            seen.add(key)
    return unique


def _sorted_hash_membership(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(reference, queries)
    valid = positions < len(reference)
    result = np.zeros(len(queries), dtype=bool)
    result[valid] = reference[positions[valid]] == queries[valid]
    return result


def run_select_queries(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0166Error("R0166 query handler received another queue")
    population, population_signature = _read_population(job)
    layout = _read_sealed(job["canonical_layout"], label="accepted R0162 layout")
    r0116 = _read_sealed(job["r0116_manifest"], label="accepted R0116 manifest")
    r0120 = _read_sealed(job["r0120_manifest"], label="accepted R0120 manifest")
    text_layout = _full_text_layouts(r0116, r0120)
    observed_payloads = _query_payload_inputs(layout, text_layout)
    observed_keys = {
        (item["canonical_path"], int(item["bytes"]), item["sha256"])
        for item in observed_payloads
    }
    expected_keys = {
        (item["canonical_path"], int(item["bytes"]), item["sha256"])
        for item in job["payload_inputs"]
    }
    if observed_keys != expected_keys:
        raise Round0166Error("R0166 held-out payload set changed")
    for index, signature in enumerate(job["payload_inputs"]):
        _validate_signature(signature, label=f"R0166 held-out payload {index}")
    candidates = np.arange(8_000_000, 8_000_000 + QUERY_CANDIDATES, dtype=np.int64)
    values = _gather_canonical_document_rows(layout, candidates)
    text_hashes = np.empty(QUERY_CANDIDATES, dtype="V32")
    for index, (row, text) in enumerate(_iter_texts(text_layout, candidates)):
        if row != int(candidates[index]):
            raise Round0166Error("R0166 heldout text order changed")
        text_hashes[index] = hashlib.sha256(text.encode("utf-8")).digest()

    source = _open_source(population)
    embedding_copied, embedding_audit = exact_reference_copy_mask(source, values)
    training_hashes = np.load(
        prompt_contract.verify_signature(
            population["source_text_hash_index"], label="R0165 training text hashes"
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    text_copied = _sorted_hash_membership(training_hashes, text_hashes)
    clean = ~(embedding_copied | text_copied)
    selected: list[int] = []
    seen_text: set[bytes] = set()
    seen_embedding: set[bytes] = set()
    within_text_rejections = 0
    within_embedding_rejections = 0
    for position in np.flatnonzero(clean).tolist():
        text_key = np.asarray(text_hashes[position]).tobytes(order="C")
        embedding_key = np.asarray(values[position]).tobytes(order="C")
        duplicate_text = text_key in seen_text
        duplicate_embedding = embedding_key in seen_embedding
        if duplicate_text or duplicate_embedding:
            within_text_rejections += int(duplicate_text)
            within_embedding_rejections += int(duplicate_embedding)
            continue
        seen_text.add(text_key)
        seen_embedding.add(embedding_key)
        selected.append(position)
        if len(selected) == QUERY_ROWS:
            break
    positions = np.asarray(selected, dtype=np.int64)
    if positions.shape != (QUERY_ROWS,):
        raise Round0166Error("R0166 heldout query reserve is exhausted")
    query_values = np.asarray(values[positions], dtype=np.float16)
    query_rows = np.asarray(candidates[positions], dtype=np.int64)
    query_hashes = np.asarray(text_hashes[positions], dtype="V32")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0166 query output")
    values_path = os.path.join(output, "document-query-embeddings.f16.npy")
    rows_path = os.path.join(output, "canonical-query-rows.i64.npy")
    hashes_path = os.path.join(output, "source-text-sha256.v32.npy")
    atomic_save_new_npy(values_path, query_values, immutable=True)
    atomic_save_new_npy(rows_path, query_rows, immutable=True)
    atomic_save_new_npy(hashes_path, query_hashes, immutable=True)
    receipt = prompt_contract.seal({
        "schema": QUERY_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "population": population_signature,
        "candidate_canonical_range": [8_000_000, 8_000_000 + QUERY_CANDIDATES],
        "candidate_rows": QUERY_CANDIDATES,
        "selected_rows": QUERY_ROWS,
        "selection_rule": (
            "first ascending post-8M rows disjoint from training by complete source text "
            "and complete stored prompted-fp16 bytes, unique within panel by both"
        ),
        "queries": _signature(values_path, label="R0166 heldout embeddings"),
        "canonical_rows": _signature(rows_path, label="R0166 heldout rows"),
        "source_text_hashes": _signature(hashes_path, label="R0166 heldout hashes"),
        "ordered_canonical_rows_sha256": ordered_array_sha256(query_rows),
        "ordered_prompted_fp16_sha256": ordered_array_sha256(query_values),
        "training_copy_audit": {
            "embedding": embedding_audit,
            "source_text_copy_rows": int(text_copied.sum()),
            "union_rejected_rows": int((~clean).sum()),
            "selected_exact_training_identity_disjoint": True,
        },
        "within_reserve_text_duplicate_rejections": within_text_rejections,
        "within_reserve_embedding_duplicate_rejections": within_embedding_rejections,
        "selected_before_training": True,
        "training_performed": False,
    })
    atomic_write_new_json(os.path.join(output, "query-reserve.json"), receipt, immutable=True)


def _load_graph(path: str) -> dict[str, Any]:
    signature = _signature(path, label="R0166 graph manifest")
    manifest = prompt_contract.read_sealed(path, label="R0166 graph manifest")
    search = manifest.get("search_qualification") or {}
    fixed = (search.get("cells") or {}).get(str(GRAPH_NPROBE)) or {}
    rows = int(manifest.get("retained_rows", -1))
    if (
        manifest.get("schema") != GRAPH_SCHEMA
        or manifest.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or rows <= MIN_SCALE_ROWS_EXCLUSIVE
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or int(manifest.get("directed_edge_count", -1)) <= 0
        or int(search.get("selected_nprobe", -1)) != GRAPH_NPROBE
        or fixed.get("passed") is not True
    ):
        raise Round0166Error("R0166 graph contract changed")
    graph_path = prompt_contract.verify_signature(manifest["graph"], label="R0166 graph")
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import load_edge_arrays

    sources, targets, weights, n_nodes = load_edge_arrays(graph_path, load_weights=True)
    if (
        weights is None
        or int(n_nodes) != rows
        or len(sources) != int(manifest["directed_edge_count"])
        or targets.shape != sources.shape
        or weights.shape != sources.shape
    ):
        raise Round0166Error("R0166 graph arrays changed")
    return {
        "manifest": manifest,
        "manifest_signature": signature,
        "signature": dict(manifest["graph"]),
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": rows,
    }


def _weighted_rejection_accounting_mismatch(
    runtime: Mapping[str, Any], *, producer_delta: int
) -> dict[str, Any] | None:
    """Close sampler accounting against this scale round's dynamic horizon."""
    expected_emitted = (
        SUCCESSFUL_UPDATES + producer_delta
    ) * prompt_contract.POSITIVE_ROWS_PER_UPDATE
    if (
        int(runtime["weight_emitted_draws"]) != expected_emitted
        or int(runtime["weight_acceptances"])
        != int(runtime["weight_emitted_draws"])
        + int(runtime["weight_buffered_draws"])
        or int(runtime["weight_proposals"]) < int(runtime["weight_acceptances"])
        or not 0 < float(runtime["weight_acceptance_rate"]) <= 1
    ):
        return {
            "expected_emitted_positive_draws": expected_emitted,
            "expected_consumed_positive_draws": (
                SUCCESSFUL_UPDATES * prompt_contract.POSITIVE_ROWS_PER_UPDATE
            ),
            "producer_delta": producer_delta,
            "runtime": runtime,
        }
    return None


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0166Error("R0166 train handler received another queue")
    population, population_signature = _read_population(job)
    graph = _load_graph(str(job["graph_manifest"]))
    rows = int(population["retained_rows"])
    if (
        graph["n_nodes"] != rows
        or graph["manifest"]["population"] != population_signature
        or graph["manifest"]["compact_mapping"] != population["mapping"]
    ):
        raise Round0166Error("R0166 graph/population binding changed")
    config, config_sha = scale_train_config(
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        graph_edges=len(graph["sources"]),
        retained_rows=rows,
    )
    source = _open_source(population)
    dataset = prompt_contract.HostFp16EndpointArray(
        source,
        arm="document",
        source_signature=population["document_compact"],
        mapping_signature=population["mapping"],
        buffer_rows=prompt_contract.BATCH_SIZE,
    )
    wrapper = ScalePromptTrainingInput(dataset, graph, arm="document")
    output = create_fresh_directory(str(job["outputs"][0]), label="R0166 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "config": config,
            "config_sha256": config_sha,
        },
        immutable=True,
    )
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = prompt_nodes._new_model(config)
    model._max_train_steps = SUCCESSFUL_UPDATES
    model._bench_warmup = prompt_contract.PERFORMANCE_WARMUP_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = prompt_contract.PERFORMANCE_WINDOWS
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    model.fit(
        wrapper,
        low_memory=True,
        verbose=False,
        n_processes=6,
        random_state=SEED,
        resample_negatives=False,
        precomputed_edges_path=graph["signature"]["canonical_path"],
        use_wandb=False,
    )
    wall = time.monotonic() - started
    accounting = dict(model._train_stats)
    runtime = wrapper.runtime_stamp()
    expected_stamp = config["execution"]["expected_pipeline_stamp"]
    mismatches = {
        key: {"expected": value, "observed": runtime.get(key)}
        for key, value in expected_stamp.items()
        if runtime.get(key) != value
    }
    exact = {
        "lr_horizon": SUCCESSFUL_UPDATES,
        "positive_lr_optimizer_steps": SUCCESSFUL_UPDATES,
        "scheduler_steps": SUCCESSFUL_UPDATES,
        "attempted_batches": SUCCESSFUL_UPDATES,
        "finite_loss_batches": SUCCESSFUL_UPDATES,
        "optimizer_steps_attempted": SUCCESSFUL_UPDATES,
        "optimizer_steps_succeeded": SUCCESSFUL_UPDATES,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": len(graph["sources"]),
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    expected_rows = SUCCESSFUL_UPDATES * prompt_contract.BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        int(runtime["source_rows_gathered"]) != expected_rows
        or int(runtime["destination_rows_gathered"]) != expected_rows
        or int(runtime["host_prefetch_consumer_batches"]) != SUCCESSFUL_UPDATES
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": runtime,
        }
    weighted = _weighted_rejection_accounting_mismatch(
        runtime, producer_delta=producer_delta
    )
    if weighted is not None:
        mismatches["weighted_rejection_accounting"] = weighted
    if mismatches:
        raise Round0166Error(f"R0166 train accounting failed: {mismatches}")
    prompt_contract.synchronize_runtime_counters(accounting, runtime)
    # ParametricUMAP captures the sampler's generic R0113 loader stamp before
    # ScalePromptTrainingInput applies the bound prompted-population semantics.
    # Keep the nested accounting view identical to the authenticated wrapper
    # stamp instead of persisting that stale generic multiplicity label.
    accounting["pipeline_runtime"] = dict(runtime)
    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (SUCCESSFUL_UPDATES - prompt_contract.PERFORMANCE_WARMUP_UPDATES)
        / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if profiler.get("aborted") is not False or rate < config["execution"]["minimum_train_upd_s"]:
        raise Round0166Error("R0166 train performance admission failed")
    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0166Error(
            f"R0166 train peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "training_seed": SEED,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": _signature(config_path, label="R0166 production config"),
        "production_config_sha256": config_sha,
        "model": _signature(model_path, label="R0166 model"),
        "population": population_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "requested_positive_draws_per_edge": float(
            config["execution"].get(
                "target_positive_draws_per_edge",
                SUCCESSFUL_UPDATES
                * prompt_contract.POSITIVE_ROWS_PER_UPDATE
                / len(graph["sources"]),
            )
        ),
        "consumed_positive_draws": int(
            SUCCESSFUL_UPDATES * prompt_contract.POSITIVE_ROWS_PER_UPDATE
        ),
        "consumed_positive_draws_per_edge": float(
            SUCCESSFUL_UPDATES
            * prompt_contract.POSITIVE_ROWS_PER_UPDATE
            / len(graph["sources"])
        ),
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_s": wall,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
        },
        "memory": {
            "device_total_bytes": int(total_bytes),
            "post_train_free_bytes": int(free_bytes),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
        "training_performed": True,
        "optimizer_updates": SUCCESSFUL_UPDATES,
        "map_decision_made": False,
    })
    atomic_write_new_json(os.path.join(output, "train-receipt.json"), receipt, immutable=True)
    del model, wrapper, dataset, source, graph
    torch.cuda.empty_cache()
    gc.collect()


def _authenticate_model(
    job: Mapping[str, Any],
    population: Mapping[str, Any],
    population_signature: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    graph_path = str(job["graph_manifest"])
    graph_signature = _signature(graph_path, label="R0166 graph manifest")
    graph = prompt_contract.read_sealed(graph_path, label="R0166 graph manifest")
    rows = int(population["retained_rows"])
    fixed = ((graph.get("search_qualification") or {}).get("cells") or {}).get(
        str(GRAPH_NPROBE), {}
    )
    if (
        graph.get("schema") != GRAPH_SCHEMA
        or graph.get("round_id") != GRAPH_SOURCE_ROUND_ID
        or graph.get("population") != dict(population_signature)
        or int(graph.get("retained_rows", -1)) != rows
        or int(graph.get("directed_edge_count", -1)) <= 0
        or fixed.get("passed") is not True
        or set(graph.get("centroids") or {}) != {"256", "1024"}
        or (graph.get("search_qualification") or {}).get("index")
        != GRAPH_INDEX_DESCRIPTION
    ):
        raise Round0166Error("R0166 graph/evaluation binding changed")
    for key in ("graph", "high_d_reference"):
        prompt_contract.verify_signature(graph[key], label=f"R0166 graph {key}")
    for key, signature in (graph.get("centroids") or {}).items():
        prompt_contract.verify_signature(signature, label=f"R0166 graph centroid {key}")
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    train_signature = _signature(train_path, label="R0166 train receipt")
    train = prompt_contract.read_sealed(train_path, label="R0166 train receipt")
    config, config_sha = scale_train_config(
        graph_signature=graph["graph"],
        graph_manifest_signature=graph_signature,
        graph_edges=int(graph["directed_edge_count"]),
        retained_rows=rows,
    )
    if (
        train.get("schema") != TRAIN_SCHEMA
        or train.get("round_id") != ROUND_ID
        or int(train.get("training_seed", -1)) != SEED
        or train.get("population") != dict(population_signature)
        or train.get("graph_manifest") != graph_signature
        or train.get("production_config_sha256") != config_sha
        or int(train.get("optimizer_updates", -1)) != SUCCESSFUL_UPDATES
        or not _train_checks_close(train.get("train_checks"))
    ):
        raise Round0166Error("R0166 accepted train receipt changed")
    model_path = prompt_contract.verify_signature(train["model"], label="R0166 model")
    from basemap.pumap.parametric_umap import ParametricUMAP

    model = ParametricUMAP.load(model_path, device="cuda")
    expected = config["model"]
    observed = {
        "architecture": model.architecture,
        "input_dimension": model.input_dim,
        "hidden_dimension": model.hidden_dim,
        "hidden_layers": model.n_layers,
        "output_dimension": model.n_components,
        "use_batchnorm": model.use_batchnorm,
        "use_dropout": model.use_dropout,
        "low_dim_kernel": model.low_dim_kernel,
        "a": model.a,
        "b": model.b,
    }
    if observed != expected:
        raise Round0166Error("R0166 loaded model architecture changed")
    return model, train, train_signature, graph


def _read_family_and_gates(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    family_signature = dict(job["family_evidence"])
    family = _read_sealed(family_signature, label="accepted R0160 prompted family")
    cells = family.get("cells") or {}
    seed42 = cells.get("seed42") or {}
    if (
        family.get("schema") != "round0160-prompted-four-seed-family-evidence-v1"
        or family.get("round_id") != "0160"
        or family.get("capability") != FAMILY_CAPABILITY
        or family.get("gate_registered") is not False
        or set(cells) != {"seed42", "seed43", "seed44", "seed45"}
        or set(seed42.get("decision_metrics") or {}) != set(METRICS)
        or set(family.get("centroids") or {}) != {"256", "1024"}
    ):
        raise Round0166Error("accepted R0160 prompted family changed")
    gate_signature = dict(job["gate_registration"])
    gates = _read_sealed(gate_signature, label="accepted R0161 prompted gates")
    if (
        gates.get("schema") != "round0161-prompted-universe-quality-gates-v1"
        or gates.get("round_id") != "0161"
        or gates.get("capability") != GATE_CAPABILITY
        or gates.get("registered") is not True
        or set(gates.get("gates") or {}) != set(METRICS)
        or gates.get("raw_floor_changed") is not False
    ):
        raise Round0166Error("accepted R0161 prompted gates changed")
    floors = {
        metric: float(gates["gates"][metric]["floor"])
        for metric in METRICS
    }
    if not np.isfinite(tuple(floors.values())).all():
        raise Round0166Error("accepted prompted floors are nonfinite")
    return family, gates, floors


def _centroids(signatures: Mapping[str, Any], *, label: str) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    for key in ("256", "1024"):
        path = prompt_contract.verify_signature(
            signatures[key], label=f"{label} k{key} centroids"
        )
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (int(key), DIMENSION) or values.dtype != np.float32:
            raise Round0166Error(f"{label} k{key} centroid geometry changed")
        output[int(key)] = values
    return output


def _projection_metrics(
    *,
    high10: np.ndarray,
    query_coordinates: np.ndarray,
    coordinates: np.ndarray,
    cfg: Any,
    truth_signature: Mapping[str, Any],
    truth_row_range: Sequence[int],
) -> dict[str, Any]:
    from basemap.panel_v2 import cross_knn, ffr_from_neighbors, recall_at_k_from_neighbors

    k_fraction = max(cfg.k_hit, int(math.ceil(cfg.frac * len(coordinates))))
    low = cross_knn(
        query_coordinates, coordinates, k_fraction, cfg, hi_dim=False
    )
    low10 = low[:, : cfg.k_hit]
    return {
        "ffr": round(float(ffr_from_neighbors(high10, low, cfg.k_hit)), 4),
        "recall_at_10": round(
            float(recall_at_k_from_neighbors(high10, low10, cfg.k_hit)), 5
        ),
        "queries": len(query_coordinates),
        "k_fraction": k_fraction,
        "truth": dict(truth_signature),
        "truth_row_range": [int(value) for value in truth_row_range],
        "query_embedding_convention": "document",
    }


def _panel_execution_ok(panel: Mapping[str, Any]) -> bool:
    guards = panel.get("guards") or {}
    return bool(
        guards.get("coords_finite") is True
        and guards.get("coords_collapsed") is False
        and guards.get("emb_finite") is True
        and guards.get("emb_zero_rows") == 0
    )


def _finalize_scale_decision(
    metric_decision: Mapping[str, Any], execution_gates: Mapping[str, bool]
) -> dict[str, Any]:
    """Combine metric evidence with either gated or diagnostic release policy."""
    metric_gates_passed = bool(metric_decision["passed"])
    metric_gates_required = metric_decision.get(
        "metric_gates_required_for_capability", True
    )
    if type(metric_gates_required) is not bool:
        raise Round0166Error("metric gate release policy is invalid")
    execution_gates_passed = all(execution_gates.values())
    passed = bool(
        execution_gates_passed
        and (metric_gates_passed or not metric_gates_required)
    )
    if passed and metric_gates_required:
        outcome = "prompted-english-8m-scale-rung-qualified"
    elif passed:
        outcome = "prompted-english-8m-dose-readout-valid"
    elif not execution_gates_passed:
        outcome = "prompted-english-8m-execution-invalid"
    else:
        outcome = "prompted-english-8m-scale-rung-not-qualified"
    return {
        **metric_decision,
        "metric_gates_passed": metric_gates_passed,
        "metric_gates_required_for_capability": metric_gates_required,
        "execution_gates": dict(execution_gates),
        "execution_gates_passed": execution_gates_passed,
        "passed": passed,
        "outcome": outcome,
    }


def run_evaluate(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import (
        build_query_truth,
        load_hiD_reference,
        load_query_truth,
        save_query_truth,
        score_panel,
    )

    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0166Error("R0166 evaluation handler received another queue")
    population, population_signature = _read_population(job)
    family, gates, floors = _read_family_and_gates(job)
    model, train, train_signature, graph = _authenticate_model(
        job, population, population_signature
    )
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0166 prompted scale evaluation"
    )
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats("cuda")
    cfg = prompt_contract.panel_config()

    source_raw = _open_source(population)
    source = L2NormalizedArray(source_raw)
    native_coordinates = np.asarray(
        model.transform(source_raw, batch_size=8192), dtype=np.float32
    )
    query_path = os.path.join(str(job["query_output"]), "query-reserve.json")
    query_signature = _signature(query_path, label="R0166 held-out query receipt")
    query = prompt_contract.read_sealed(query_path, label="R0166 held-out query receipt")
    if (
        query.get("schema") != QUERY_SCHEMA
        or query.get("population") != population_signature
        or int(query.get("selected_rows", -1)) != QUERY_ROWS
        or query.get("selected_before_training") is not True
        or (query.get("training_copy_audit") or {}).get(
            "selected_exact_training_identity_disjoint"
        )
        is not True
    ):
        raise Round0166Error("R0166 held-out query contract changed")
    query_values = np.load(
        prompt_contract.verify_signature(query["queries"], label="R0166 held-out queries"),
        mmap_mode="r",
        allow_pickle=False,
    )
    if query_values.shape != (QUERY_ROWS, DIMENSION) or query_values.dtype != np.float16:
        raise Round0166Error("R0166 held-out query geometry changed")
    native_query_coordinates = np.asarray(
        model.transform(query_values, batch_size=8192), dtype=np.float32
    )
    if (
        native_coordinates.shape != (len(source_raw), 2)
        or native_query_coordinates.shape != (QUERY_ROWS, 2)
        or not np.isfinite(native_coordinates).all()
        or not np.isfinite(native_query_coordinates).all()
    ):
        raise Round0166Error("R0166 native transform output is invalid")
    native_coordinate_path = os.path.join(output, "native-8m-coordinates.npy")
    native_query_coordinate_path = os.path.join(
        output, "native-8m-heldout-query-coordinates.npy"
    )
    atomic_save_new_npy(native_coordinate_path, native_coordinates, immutable=True)
    atomic_save_new_npy(
        native_query_coordinate_path, native_query_coordinates, immutable=True
    )
    native_centroids = _centroids(graph["centroids"], label="R0166 native")
    native_reference = load_hiD_reference(
        prompt_contract.verify_signature(
            graph["high_d_reference"], label="R0166 native high-D reference"
        ),
        expected_key=str(graph["high_d_reference_key"]),
    )
    native_panel = score_panel(
        source,
        native_coordinates,
        config=cfg,
        centroids_by_k=native_centroids,
        hiD_reference=native_reference,
        reference_identity=graph["reference_identity"],
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "universe": "native-frozen-prefix-8m",
            "population": population_signature,
            "train_receipt": train_signature,
            "coordinates": _signature(
                native_coordinate_path, label="R0166 native coordinates"
            ),
        },
    )
    native_query_identity = {
        "schema": "round0166-native-heldout-query-identity-v1",
        "receipt": query_signature,
        "ordered_rows_sha256": query["ordered_canonical_rows_sha256"],
        "ordered_fp16_sha256": query["ordered_prompted_fp16_sha256"],
    }
    native_truth = build_query_truth(
        L2NormalizedArray(query_values),
        source,
        cfg=cfg,
        corpus_identity=_data_identity(population),
        query_identity=native_query_identity,
        k=cfg.k_hit,
    )
    native_truth_path = os.path.join(output, "native-query-truth-k10.npz")
    save_query_truth(native_truth, native_truth_path)
    native_truth_signature = _signature(
        native_truth_path, label="R0166 native query truth"
    )
    native_projection = _projection_metrics(
        high10=np.asarray(native_truth["neighbors"], dtype=np.int64),
        query_coordinates=native_query_coordinates,
        coordinates=native_coordinates,
        cfg=cfg,
        truth_signature=native_truth_signature,
        truth_row_range=(0, QUERY_ROWS),
    )
    native_metrics = metric_view(
        panel=native_panel, native_score={"projections": {"matched": native_projection}}
    )

    seed42 = family["cells"]["seed42"]
    baseline_metrics = {
        metric: float(seed42["decision_metrics"][metric]) for metric in METRICS
    }
    accepted_score_signature = dict(seed42["native_score"])
    accepted_score = _read_sealed(
        accepted_score_signature, label="accepted R0160 seed-42 native score"
    )
    if (
        accepted_score.get("round_id") != "0115"
        or accepted_score.get("arm") != "document"
        or int(accepted_score.get("training_seed", 42)) != 42
        or accepted_score.get("projection_ffr_role") != "diagnostic-only"
    ):
        raise Round0166Error("accepted prompted seed-42 score changed")
    matched_source_signature = dict(family["lineage"]["document_compact"])
    matched_source_path = prompt_contract.verify_signature(
        matched_source_signature, label="accepted R0113 prompted compact matrix"
    )
    matched_raw = np.memmap(
        matched_source_path,
        mode="r",
        dtype="<f2",
        shape=(MATCHED_ROWS, DIMENSION),
    )
    matched_source = L2NormalizedArray(matched_raw)
    matched_coordinates = np.asarray(
        model.transform(matched_raw, batch_size=8192), dtype=np.float32
    )
    accepted_query = _read_sealed(
        accepted_score["query_reserve"], label="accepted R0113 query reserve"
    )
    accepted_selection = _read_sealed(
        accepted_score["query_selection"], label="accepted seed-42 query selection"
    )
    positions = np.load(
        prompt_contract.verify_signature(
            accepted_selection["positions"], label="accepted query positions"
        ),
        allow_pickle=False,
    )
    reserve = np.load(
        prompt_contract.verify_signature(
            accepted_query["outputs"]["document"],
            label="accepted prompted query reserve",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )
    if (
        positions.shape != (QUERY_ROWS,)
        or positions.dtype != np.int64
        or np.any(positions[1:] <= positions[:-1])
        or reserve.shape != (QUERY_CANDIDATES, DIMENSION)
        or reserve.dtype != np.float16
        or np.any(positions < 0)
        or np.any(positions >= QUERY_CANDIDATES)
    ):
        raise Round0166Error("accepted matched query selection changed")
    matched_query_values = np.asarray(reserve[positions], dtype=np.float16)
    matched_query_coordinates = np.asarray(
        model.transform(matched_query_values, batch_size=8192), dtype=np.float32
    )
    if (
        matched_coordinates.shape != (MATCHED_ROWS, 2)
        or matched_query_coordinates.shape != (QUERY_ROWS, 2)
        or not np.isfinite(matched_coordinates).all()
        or not np.isfinite(matched_query_coordinates).all()
    ):
        raise Round0166Error("R0166 matched-2M transform output is invalid")
    matched_coordinate_path = os.path.join(output, "matched-2m-coordinates.npy")
    matched_query_coordinate_path = os.path.join(
        output, "matched-2m-query-coordinates.npy"
    )
    atomic_save_new_npy(matched_coordinate_path, matched_coordinates, immutable=True)
    atomic_save_new_npy(
        matched_query_coordinate_path, matched_query_coordinates, immutable=True
    )
    matched_centroids = _centroids(family["centroids"], label="accepted R0160")
    matched_reference = load_hiD_reference(
        prompt_contract.verify_signature(
            family["shared_prompted_reference"],
            label="accepted R0160 prompted high-D reference",
        )
    )
    assembly = _read_sealed(
        family["lineage"]["assembly"], label="accepted R0113 compact assembly"
    )
    matched_reference_identity = {
        "data_identity": prompt_nodes._data_identity(assembly, arm="document"),
        "convention": {
            "row_order": (
                "R0113 shared source/raw/document union-representative compact order"
            ),
            "distance": "cosine via fp32-L2-normalized squared L2",
            "self_exclusion": True,
            "anchor_namespace": "R0113 compact IDs",
            "embedding_prompt": "document",
        },
    }
    matched_panel = score_panel(
        matched_source,
        matched_coordinates,
        config=cfg,
        centroids_by_k=matched_centroids,
        hiD_reference=matched_reference,
        reference_identity=matched_reference_identity,
        scale_admission=None,
        provenance={
            "round_id": ROUND_ID,
            "universe": "accepted-r0113-matched-2m",
            "source": matched_source_signature,
            "train_receipt": train_signature,
            "coordinates": _signature(
                matched_coordinate_path, label="R0166 matched coordinates"
            ),
        },
    )
    accepted_truth_signature = dict(accepted_score["combined_query_truth"])
    accepted_truth = load_query_truth(
        prompt_contract.verify_signature(
            accepted_truth_signature, label="accepted R0115 combined query truth"
        )
    )
    truth_range = accepted_score["projections"]["matched"]["truth_row_range"]
    if truth_range != [0, QUERY_ROWS] or accepted_truth["corpus_cardinality"] != MATCHED_ROWS:
        raise Round0166Error("accepted matched query-truth range changed")
    matched_projection = _projection_metrics(
        high10=np.asarray(accepted_truth["neighbors"][:QUERY_ROWS], dtype=np.int64),
        query_coordinates=matched_query_coordinates,
        coordinates=matched_coordinates,
        cfg=cfg,
        truth_signature=accepted_truth_signature,
        truth_row_range=truth_range,
    )
    matched_metrics = metric_view(
        panel=matched_panel,
        native_score={"projections": {"matched": matched_projection}},
    )
    metric_decision = scale_decision(
        native=native_metrics,
        matched_2m=matched_metrics,
        baseline_2m=baseline_metrics,
        prompted_floors=floors,
    )
    execution_gates = {
        "train_receipt_closes": _train_checks_close(train.get("train_checks")),
        "graph_fixed_nprobe_qualified": (
            ((graph.get("search_qualification") or {}).get("cells") or {})
            .get(str(GRAPH_NPROBE), {})
            .get("passed")
            is True
        ),
        "graph_vector_storage_registered": (
            (graph.get("search_qualification") or {}).get("index")
            == GRAPH_INDEX_DESCRIPTION
        ),
        "heldout_queries_selected_before_training_and_disjoint": (
            query.get("selected_before_training") is True
            and (query.get("training_copy_audit") or {}).get(
                "selected_exact_training_identity_disjoint"
            )
            is True
        ),
        "native_panel_finite_noncollapsed": _panel_execution_ok(native_panel),
        "matched_panel_finite_noncollapsed": _panel_execution_ok(matched_panel),
    }
    decision = _finalize_scale_decision(metric_decision, execution_gates)
    passed = bool(decision["passed"])
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0166Error(
            f"R0166 evaluation peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        "schema": EVALUATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capabilities": [CAPABILITY] if passed else [],
        "decision": decision,
        "population": population_signature,
        "query_reserve": query_signature,
        "graph_manifest": _signature(
            str(job["graph_manifest"]), label="R0166 graph manifest"
        ),
        "train_receipt": train_signature,
        "prompted_gate_registration": dict(job["gate_registration"]),
        "prompted_seed_family": dict(job["family_evidence"]),
        "native_8m": {
            "coordinates": _signature(
                native_coordinate_path, label="R0166 native coordinates"
            ),
            "query_coordinates": _signature(
                native_query_coordinate_path,
                label="R0166 native query coordinates",
            ),
            "panel": native_panel,
            "projection": native_projection,
            "decision_metrics": native_metrics,
            "projection_metrics_role": "diagnostic-only on changed N/query universe",
        },
        "matched_2m": {
            "source": matched_source_signature,
            "coordinates": _signature(
                matched_coordinate_path, label="R0166 matched coordinates"
            ),
            "query_coordinates": _signature(
                matched_query_coordinate_path,
                label="R0166 matched query coordinates",
            ),
            "accepted_seed42_score": accepted_score_signature,
            "accepted_query_truth": accepted_truth_signature,
            "panel": matched_panel,
            "projection": matched_projection,
            "decision_metrics": matched_metrics,
            "baseline_seed42_metrics": baseline_metrics,
        },
        "prompted_floors": floors,
        "training_performed_in_round": True,
        "evaluation_node_training_performed": False,
        "graph_built_in_round": GRAPH_BUILT_IN_ROUND,
        "performance": {
            "evaluation_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "scale-evaluation.json"), receipt, immutable=True
    )
    del (
        model,
        source_raw,
        source,
        native_coordinates,
        native_query_coordinates,
        native_centroids,
        native_reference,
        matched_raw,
        matched_source,
        matched_coordinates,
        matched_query_coordinates,
        matched_centroids,
        matched_reference,
    )
    torch.cuda.empty_cache()
    gc.collect()


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    actions = {
        "select_heldout_queries": run_select_queries,
        "build_graph_and_reference": run_build_graph,
        "train_prompted_8m": run_train,
        "evaluate_prompted_8m": run_evaluate,
    }
    action = str(job.get("action") or "")
    if action not in actions:
        raise Round0166Error(f"unknown R0166 action {action!r}")
    actions[action](active, job)
