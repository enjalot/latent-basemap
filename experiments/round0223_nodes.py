"""Execute R0223 — train maps on the cuVS igd48 graph and score them.

Five nodes:

* `build_cuvs_igd48_fuzzy_graph` (GPU, once) turns R0220's sealed
  `nnd-gd32-igd48-it20` neighbour ids into a k15 fuzzy graph over R0216's sealed
  `queue-correction-3` substrate, under R0216's identical fuzzy law. Before it
  does, it re-measures the graph's recall against R0220's sealed exact truth
  arrays over all 2,000,000 rows and requires the published tie-aware and strict
  values back to `1e-6`. It then applies R0171's ANN floors and the R0215
  zero-degree tripwire. This is where the round can find out that a
  0.994-recall graph is structurally unusable, before any GPU-hour is spent
  training on it.
* `train_minilm_mixed_2m_cuvs_graph` (GPU, three cells, seeds 42/43/44) is
  R0221's train node with one input swapped. The config is built from R0217's
  own template and only the registered seed-bearing and graph-bearing paths are
  overwritten; the node recomputes the treatment-invariant digest and refuses to
  train unless it equals the template's.
* `compare_cuvs_graph_map_panel` (GPU, once) scores the three cells on R0218's
  **byte-identical** frozen high-D reference — loaded, signature-checked,
  content-key re-derived and anchor-compared exactly as R0222 did — then reports
  z-scores against R0222's sealed eight-cell exact-graph family and pass/fail
  against R0222's registered-not-released `n = 8` floors, read out of R0222's
  sealed artifact rather than typed.

No gate is registered and none is released.
"""
from __future__ import annotations

import gc
import math
import os
import random
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature, ordered_array_sha256
from basemap.output_safety import (
    atomic_build_new_file,
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0217_minilm_2m_seed_family import WARMUP_SUCCESSFUL_UPDATES
from basemap.round0218_minilm_2m_panel import (
    CORPUS_SLUGS,
    PANEL_METRICS,
    corpus_ffr_view,
    panel_execution_ok,
    panel_metric_view,
)
from basemap.round0220_cuvs_qualification import (
    GRAPH_K as R0220_GRAPH_K,
    TIE_TOLERANCE,
    TRUTH_SCHEMA as R0220_TRUTH_SCHEMA,
    graph_validity,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from basemap.round0223_cuvs_graph_map import (
    BATCH_SIZE,
    COMPARISON_CAPABILITY,
    COMPARISON_SCHEMA,
    CUVS_GRAPH_CAPABILITY,
    CUVS_GRAPH_DEGREE,
    CUVS_GRAPH_SCHEMA,
    CUVS_INTERMEDIATE_GRAPH_DEGREE,
    CUVS_MAX_ITERATIONS,
    CUVS_METRIC,
    CUVS_SETTING_ID,
    DIMENSION,
    EVIDENCE_LIMITS,
    FLOOR_STATUS,
    FULL_TRANSFORM_BATCH,
    FUZZY_LAW,
    FUZZY_RANDOM_STATE_SEED,
    GATE_REGISTERABLE_HERE,
    GATE_RELEASE_CLAIMED,
    GRAPH_K,
    HOST_RSS_LIMIT_GIB,
    MAP_CAPABILITIES,
    MIN_ADMISSIBLE_NEGATIVE_DISTANCE,
    R0216_EXACT_KERNEL_MIN_DISTANCE,
    R0216_EXACT_KERNEL_NEGATIVE_ENTRIES,
    PENDING_FLOOR_METRICS,
    PIPELINE_STAMP_LABEL_CARRYOVER,
    POSITIVE_ROWS_PER_UPDATE,
    PRODUCTION_CONFIG_SCHEMA,
    R0220_ROUND_ID,
    R0220_STRICT_RECALL,
    R0220_TIE_AWARE_RECALL,
    R0222_GATE_SCHEMA,
    R0222_POOLED_SEEDS,
    RECALL_CROSS_CHECK_TOLERANCE,
    ROUND_ID,
    ROWS,
    Round0223Error,
    SEEDS,
    TEMPLATE_SEED,
    TRAIN_SCHEMA,
    compare_to_exact_family,
    map_capability,
    performance_windows,
    successful_updates_for_edges,
    train_config,
    treatment_invariant_sha256,
    validate_cuvs_graph,
    validate_dose,
    validate_full_population_map,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0113_nodes as prompt_nodes
from experiments import round0218_nodes, round0221_nodes, round0222_nodes


GRAPH_ACTION = "build_cuvs_igd48_fuzzy_graph"
TRAIN_ACTION = "train_minilm_mixed_2m_cuvs_graph"
COMPARE_ACTION = "compare_cuvs_graph_map_panel"

EVAL_BLOCK = 16_384


# --------------------------------------------------------------------------- #
# node 1: the cuVS fuzzy graph
# --------------------------------------------------------------------------- #


def _cosines_for(torch: Any, tensor: Any, ids: np.ndarray) -> np.ndarray:
    """Exact fp32 cosines of each row against its k candidate ids."""
    rows, width = ids.shape
    out = np.empty((rows, width), dtype=np.float32)
    for start in range(0, rows, EVAL_BLOCK):
        stop = min(start + EVAL_BLOCK, rows)
        block = torch.from_numpy(ids[start:stop].astype(np.int64)).to(tensor.device)
        gathered = tensor[block.reshape(-1)].reshape(stop - start, width, DIMENSION)
        queries = tensor[start:stop].unsqueeze(2)
        out[start:stop] = (
            torch.bmm(gathered, queries).squeeze(2).to(torch.float32).cpu().numpy()
        )
    return out


def run_build_graph(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    sealed = round0218_nodes._sealed_substrate(job)
    substrate_path = prompt_contract.verify_signature(
        sealed["substrate_signature"], label="R0223 sealed R0216 substrate"
    )

    qualification_signature = dict(job["cuvs_qualification_signature"])
    qualification_path = prompt_contract.verify_signature(
        qualification_signature, label="R0220 cuVS qualification receipt"
    )
    qualification = prompt_contract.read_sealed(
        qualification_path, label="R0220 cuVS qualification receipt"
    )
    if (
        qualification.get("round_id") != R0220_ROUND_ID
        or int(qualification.get("rows_evaluated", -1)) != ROWS
        or int(qualification.get("k", -1)) != GRAPH_K
        or qualification.get("metric") != CUVS_METRIC
        or qualification.get("training_performed") is not False
    ):
        raise Round0223Error("R0220 cuVS qualification receipt contract changed")
    published = None
    for entry in qualification.get("sweep") or []:
        if str(entry["setting"]["id"]) == CUVS_SETTING_ID:
            published = entry
            break
    if published is None:
        raise Round0223Error(f"R0220 receipt has no setting {CUVS_SETTING_ID}")
    if (
        int(published["setting"]["graph_degree"]) != CUVS_GRAPH_DEGREE
        or int(published["setting"]["intermediate_graph_degree"])
        != CUVS_INTERMEDIATE_GRAPH_DEGREE
        or int(published["setting"]["max_iterations"]) != CUVS_MAX_ITERATIONS
    ):
        raise Round0223Error("R0220 setting parameters are not the registered ones")

    truth_signature = dict(job["truth_receipt_signature"])
    truth_path = prompt_contract.verify_signature(
        truth_signature, label="R0220 exact k15 truth receipt"
    )
    truth = prompt_contract.read_sealed(
        truth_path, label="R0220 exact k15 truth receipt"
    )
    if (
        truth.get("schema") != R0220_TRUTH_SCHEMA
        or truth.get("round_id") != R0220_ROUND_ID
        or int(truth.get("rows", -1)) != ROWS
        or int(truth.get("k", -1)) != R0220_GRAPH_K
        or not truth["probe"]["passed"]
    ):
        raise Round0223Error("R0220 truth receipt contract changed")
    truth_ids_path = prompt_contract.verify_signature(
        truth["outputs"]["ids"], label="R0220 truth ids"
    )
    truth_cos_path = prompt_contract.verify_signature(
        truth["outputs"]["cosines"], label="R0220 truth cosines"
    )

    graph_signature = dict(job["cuvs_graph_signature"])
    cuvs_graph_path = prompt_contract.verify_signature(
        graph_signature, label="R0220 cuVS igd48 neighbour ids"
    )

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0223 cuVS igd48 fuzzy graph"
    )
    started = time.monotonic()

    raw = np.load(cuvs_graph_path, allow_pickle=False)
    if raw.shape != (ROWS, CUVS_GRAPH_DEGREE) or raw.dtype != np.uint32:
        raise Round0223Error(
            f"R0220 cuVS graph is {raw.shape}/{raw.dtype}, expected "
            f"({ROWS}, {CUVS_GRAPH_DEGREE})/uint32"
        )
    leading = np.ascontiguousarray(raw[:, :GRAPH_K])
    del raw
    # Range-check before any array is used as an index: an out-of-range id would
    # silently gather the wrong row rather than fail.
    as_int = leading.astype(np.int64)
    if int(as_int.min()) < 0 or int(as_int.max()) >= ROWS:
        raise Round0223Error("R0220 cuVS graph carries out-of-range neighbour ids")
    del as_int

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    host = np.load(substrate_path, mmap_mode="r", allow_pickle=False)
    if host.shape != (ROWS, DIMENSION) or host.dtype != np.float32:
        raise Round0223Error("R0223 sealed substrate geometry changed")
    tensor = torch.from_numpy(
        np.array(host, dtype=np.float32, order="C", copy=True)
    ).to(device)

    cosine_started = time.monotonic()
    candidate_cos = _cosines_for(torch, tensor, leading)
    cosine_s = time.monotonic() - cosine_started

    truth_ids = np.load(truth_ids_path, allow_pickle=False)
    truth_cos = np.load(truth_cos_path, allow_pickle=False)
    if truth_ids.shape != (ROWS, GRAPH_K) or truth_cos.shape != (ROWS, GRAPH_K):
        raise Round0223Error("R0220 truth arrays have the wrong shape")
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)
    strict = strict_containment_rows(leading, truth_ids)
    tie = tie_aware_rows(candidate_cos.astype(np.float64), leading, kth)
    strict_summary = summarize(strict, label="R0223 strict recall@15")
    tie_summary = summarize(tie, label="R0223 tie-aware recall@15")
    cross_check = {
        "published_tie_aware_mean": R0220_TIE_AWARE_RECALL,
        "measured_tie_aware_mean": tie_summary["mean"],
        "published_strict_mean": R0220_STRICT_RECALL,
        "measured_strict_mean": strict_summary["mean"],
        "tolerance": RECALL_CROSS_CHECK_TOLERANCE,
        "tie_aware_delta": tie_summary["mean"] - R0220_TIE_AWARE_RECALL,
        "strict_delta": strict_summary["mean"] - R0220_STRICT_RECALL,
    }
    if (
        abs(cross_check["tie_aware_delta"]) > RECALL_CROSS_CHECK_TOLERANCE
        or abs(cross_check["strict_delta"]) > RECALL_CROSS_CHECK_TOLERANCE
    ):
        raise Round0223Error(
            f"R0223 re-measured cuVS recall does not reproduce R0220's published "
            f"values: {cross_check}"
        )
    cross_check["reproduces_r0220"] = True

    structural = graph_validity(leading, rows=ROWS)
    del truth_ids, truth_cos, strict, tie

    # UMAP's smooth-knn law assumes ascending distances. cuVS returns each row
    # sorted by increasing distance, so this normally reorders nothing; it is
    # applied unconditionally and the count of reordered rows is published, so
    # "cuVS returns sorted output" is a measurement here rather than R0220's
    # stated assumption.
    order = np.argsort(-candidate_cos, axis=1, kind="stable")
    already_sorted = int(
        (order == np.arange(GRAPH_K, dtype=order.dtype)[None, :]).all(axis=1).sum()
    )
    ids_sorted = np.take_along_axis(leading, order, axis=1).astype(np.int32)
    cos_sorted = np.take_along_axis(candidate_cos, order, axis=1)
    del order, leading, candidate_cos

    dists = (1.0 - cos_sorted).astype(np.float32)
    negative_mask = dists < 0.0
    negative = int(negative_mask.sum())
    most_negative = float(dists.min()) if negative else 0.0
    if most_negative < MIN_ADMISSIBLE_NEGATIVE_DISTANCE:
        raise Round0223Error(
            f"R0223 found a cosine distance of {most_negative!r}, below the "
            f"registered {MIN_ADMISSIBLE_NEGATIVE_DISTANCE} floor; that is not "
            "float32 rounding, it is a cosine that is not a cosine"
        )
    np.maximum(dists, 0.0, out=dists)
    if not np.isfinite(dists).all():
        raise Round0223Error("R0223 cuVS candidate distances are not finite")
    del cos_sorted

    del tensor
    torch.cuda.empty_cache()
    gc.collect()

    X = np.array(host, dtype=np.float32, order="C", copy=True)
    import umap.umap_ as umap_api

    fuzzy_started = time.monotonic()
    graph, _sigmas, _rhos = umap_api.fuzzy_simplicial_set(
        X,
        n_neighbors=GRAPH_K,
        random_state=np.random.RandomState(FUZZY_RANDOM_STATE_SEED),
        metric="cosine",
        knn_indices=ids_sorted,
        knn_dists=dists,
    )
    coo = graph.tocoo()
    src = np.asarray(coo.row, dtype=np.int32)
    dst = np.asarray(coo.col, dtype=np.int32)
    wts = np.asarray(coo.data, dtype=np.float32)
    fuzzy_s = time.monotonic() - fuzzy_started
    del X, graph, coo
    gc.collect()

    if not np.isfinite(wts).all() or wts.min() <= 0 or wts.max() > 1:
        raise Round0223Error("R0223 fuzzy weights are invalid")
    if np.any(np.diff(src) < 0):
        raise Round0223Error("R0223 fuzzy edge sources are not sorted")
    degree_counts = np.bincount(src, minlength=ROWS)
    degrees = {
        "zero_degree_rows": int((degree_counts == 0).sum()),
        "min": int(degree_counts.min()),
        "median": float(np.median(degree_counts)),
        "mean": float(degree_counts.mean()),
        "max": int(degree_counts.max()),
    }
    checks = validate_cuvs_graph(
        degrees=degrees,
        recall={
            "mean_recall_at_k": float(tie_summary["mean"]),
            "p10_recall_at_k": float(tie_summary["p10"]),
        },
        edges=int(len(src)),
        structural=structural,
    )

    ids_path = atomic_save_new_npy(
        os.path.join(output, "cuvs-k15-ids.i32.npy"), ids_sorted, immutable=True
    )
    graph_path = atomic_save_new_npz(
        os.path.join(output, "edges-k15-fuzzy.npz"),
        immutable=True,
        compressed=False,
        sources=src,
        targets=dst,
        weights=wts,
        n_nodes=np.asarray(ROWS, dtype=np.int64),
        k=np.asarray(GRAPH_K, dtype=np.int64),
    )
    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0223Error(
            f"R0223 graph build peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        "schema": CUVS_GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": CUVS_GRAPH_CAPABILITY,
        "capabilities": [CUVS_GRAPH_CAPABILITY],
        "rows": ROWS,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "builder": {
            "source_round": R0220_ROUND_ID,
            "setting_id": CUVS_SETTING_ID,
            "algo": "cuvs.neighbors.nn_descent",
            "graph_degree": CUVS_GRAPH_DEGREE,
            "intermediate_graph_degree": CUVS_INTERMEDIATE_GRAPH_DEGREE,
            "max_iterations": CUVS_MAX_ITERATIONS,
            "metric": CUVS_METRIC,
            "cuvs_version": qualification.get("cuvs", {}).get("version"),
            "neighbour_ids": graph_signature,
            "qualification_receipt": qualification_signature,
            "approximate": True,
        },
        "recall_against_r0220_exact_truth": {
            "truth_receipt": truth_signature,
            "rows_measured": ROWS,
            "tie_aware": tie_summary,
            "strict": strict_summary,
            "tie_tolerance": TIE_TOLERANCE,
            "cross_check": cross_check,
        },
        "fuzzy_law": FUZZY_LAW,
        "fuzzy_random_state_seed": FUZZY_RANDOM_STATE_SEED,
        "neighbour_ordering": {
            "law": "per-row descending exact cosine, stable",
            "rows_already_in_builder_order": already_sorted,
            "rows_reordered": ROWS - already_sorted,
        },
        "distances": {
            "law": "1 - exact fp32 cosine of the substrate rows",
            "negative_entries_clipped_to_zero": negative,
            "most_negative_distance": most_negative,
            "entries": int(ROWS * GRAPH_K),
            "min_admissible_negative_distance": MIN_ADMISSIBLE_NEGATIVE_DISTANCE,
            "check_shape": (
                "magnitude, not count: the number of tied entries is a property "
                "of the substrate's duplicate structure, not of the builder"
            ),
            "r0216_exact_kernel_negative_entries": (
                R0216_EXACT_KERNEL_NEGATIVE_ENTRIES
            ),
            "r0216_exact_kernel_min_distance": R0216_EXACT_KERNEL_MIN_DISTANCE,
        },
        "substrate": dict(sealed["substrate_signature"]),
        "provenance": dict(sealed["provenance_signature"]),
        "r0216_graph_manifest": dict(sealed["manifest_signature"]),
        "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        "cuvs_k15_ids": expected_input_signature(ids_path),
        "graph": expected_input_signature(graph_path),
        "graph_checks": checks,
        "degrees": degrees,
        "directed_edge_count": int(len(src)),
        "r0216_directed_edge_count": int(sealed["directed_edges"]),
        "performance": {
            "candidate_cosine_s": cosine_s,
            "fuzzy_s": fuzzy_s,
            "total_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_host_rss_gib": peak_rss_gib,
        },
        "training_performed": False,
        "gate_registered": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "map_quality_claim_available": False,
    })
    atomic_write_new_json(
        os.path.join(output, "cuvs-graph.json"), receipt, immutable=True
    )
    del ids_sorted, dists, src, dst, wts
    gc.collect()


# --------------------------------------------------------------------------- #
# node 2: one train cell
# --------------------------------------------------------------------------- #


def _seed(job: Mapping[str, Any]) -> int:
    seed = job.get("training_seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0223Error(f"R0223 job seed {seed!r} is not a registered cell")
    if str(job.get("capability") or "") != map_capability(seed):
        raise Round0223Error("R0223 job capability does not match its seed")
    return int(seed)


def _sealed_cuvs_graph(job: Mapping[str, Any]) -> dict[str, Any]:
    """Read R0223's own sealed cuVS graph receipt and load its edges."""
    manifest_signature = dict(job["cuvs_graph_manifest_signature"])
    manifest_path = prompt_contract.verify_signature(
        manifest_signature, label="R0223 sealed cuVS graph receipt"
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label="R0223 sealed cuVS graph receipt"
    )
    checks = manifest.get("graph_checks") or {}
    degrees = manifest.get("degrees") or {}
    if (
        manifest.get("schema") != CUVS_GRAPH_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or manifest.get("capability") != CUVS_GRAPH_CAPABILITY
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or manifest.get("training_performed") is not False
        or manifest["builder"].get("setting_id") != CUVS_SETTING_ID
        or int(manifest["builder"].get("intermediate_graph_degree", -1))
        != CUVS_INTERMEDIATE_GRAPH_DEGREE
        or manifest["recall_against_r0220_exact_truth"]["cross_check"].get(
            "reproduces_r0220"
        )
        is not True
    ):
        raise Round0223Error("R0223 sealed cuVS graph contract changed")
    if (
        int(checks.get("zero_degree_rows", -1)) != 0
        or int(degrees.get("zero_degree_rows", -1)) != 0
        or float(checks.get("mean_recall_at_k", 0.0))
        < float(checks.get("mean_recall_floor", 1.0))
        or float(checks.get("p10_recall_at_k", 0.0))
        < float(checks.get("p10_recall_floor", 1.0))
    ):
        raise Round0223Error(
            "R0223 requires the sealed cuVS graph to have passed its recall and "
            "zero-degree checks"
        )
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges <= 0:
        raise Round0223Error("R0223 sealed cuVS graph reports no directed edges")
    graph_signature = dict(manifest["graph"])
    graph_path = prompt_contract.verify_signature(
        graph_signature, label="R0223 sealed cuVS fuzzy graph"
    )
    from basemap.pumap.parametric_umap.datasets.edge_list_dataset import (
        load_edge_arrays,
    )

    sources, targets, weights, n_nodes = load_edge_arrays(graph_path, load_weights=True)
    if (
        weights is None
        or int(n_nodes) != ROWS
        or len(sources) != edges
        or targets.shape != sources.shape
        or weights.shape != sources.shape
        or sources.dtype != np.int32
        or targets.dtype != np.int32
        or weights.dtype != np.float32
    ):
        raise Round0223Error("R0223 sealed cuVS graph arrays changed")
    return {
        "manifest": manifest,
        "manifest_signature": manifest_signature,
        "signature": graph_signature,
        "sources": sources,
        "targets": targets,
        "weights": weights,
        "n_nodes": int(n_nodes),
        "directed_edges": edges,
    }


def _open_substrate(graph: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Serve the 3.07 GB sealed R0216 substrate lazily; never materialize it."""
    signature = dict(graph["manifest"]["substrate"])
    path = prompt_contract.verify_signature(
        signature, label="R0223 sealed R0216 substrate"
    )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, DIMENSION) or array.dtype != np.float32:
        raise Round0223Error("R0223 sealed R0216 substrate geometry changed")
    return array, signature


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    seed = _seed(job)
    graph = _sealed_cuvs_graph(job)
    edges = graph["directed_edges"]
    updates = successful_updates_for_edges(edges)
    dose = validate_dose(updates=updates, edge_count=edges)
    declared_bound = job.get("registered_dose_bound")
    if declared_bound is not None and updates > int(declared_bound):
        raise Round0223Error(
            "R0223 derived update horizon exceeds the registered round bound"
        )

    source, substrate_signature = _open_substrate(graph)

    config, config_sha, invariant = train_config(
        seed=seed,
        graph_signature=graph["signature"],
        graph_manifest_signature=graph["manifest_signature"],
        substrate_signature=substrate_signature,
        r0216_graph_signature=dict(job["r0216_graph_signature"]),
        r0216_graph_manifest_signature=dict(job["r0216_graph_manifest_signature"]),
        graph_edges=edges,
        rows=ROWS,
    )
    declared_invariant = str(job.get("treatment_invariant_sha256") or "")
    if not declared_invariant or invariant != declared_invariant:
        raise Round0223Error(
            "R0223 cell config is not R0217's treatment outside the seed and the "
            f"graph: {invariant} != {declared_invariant}"
        )
    if int(config["optimizer"]["successful_positive_lr_updates"]) != updates:
        raise Round0223Error("R0223 horizon did not reach the train config")

    output = create_fresh_directory(str(job["outputs"][0]), label="R0223 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "treatment_config_round_id": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "seed": seed,
            "capability": map_capability(seed),
            "treatment_invariant_sha256": invariant,
            "pipeline_stamp_label_carryover": PIPELINE_STAMP_LABEL_CARRYOVER,
            "config": config,
            "config_sha256": config_sha,
        },
        immutable=True,
    )

    from basemap.round0217_minilm_2m_pipeline import (
        MiniLMHostFp32EndpointArray,
        MiniLMMixedTrainingInput,
    )

    dataset = MiniLMHostFp32EndpointArray(
        source, source_signature=substrate_signature, buffer_rows=BATCH_SIZE
    )
    wrapper = MiniLMMixedTrainingInput(dataset, graph, seed=seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats("cuda")
    model = prompt_nodes._new_model(config)
    model._max_train_steps = updates
    model._bench_warmup = WARMUP_SUCCESSFUL_UPDATES
    model._perf_profile = True
    model._perf_floor = config["execution"]["minimum_train_upd_s"]
    model._perf_warn_rate = config["execution"]["warning_train_upd_s"]
    model._perf_subfloor_patience = 2
    model._perf_n_windows = performance_windows(updates)
    model._abort_on_first_nonfinite = True
    model._admission_artifact_path = os.path.join(output, "admission.json")
    started = time.monotonic()
    model.fit(
        wrapper,
        low_memory=True,
        verbose=False,
        n_processes=6,
        random_state=seed,
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
        "lr_horizon": updates,
        "positive_lr_optimizer_steps": updates,
        "scheduler_steps": updates,
        "attempted_batches": updates,
        "finite_loss_batches": updates,
        "optimizer_steps_attempted": updates,
        "optimizer_steps_succeeded": updates,
        "amp_overflow_skips": 0,
        "nonfinite_loss_skips": 0,
        "nonfinite_gradient_skips": 0,
        "stop_reason": "lr_horizon",
        "budget_satisfied": True,
        "n_pos_edges": edges,
    }
    mismatches.update({
        key: {"expected": value, "observed": accounting.get(key)}
        for key, value in exact.items()
        if accounting.get(key) != value
    })
    expected_rows = updates * BATCH_SIZE
    producer_delta = (
        int(runtime["host_prefetch_producer_batches"])
        - int(runtime["host_prefetch_consumer_batches"])
    )
    if (
        int(runtime["source_rows_gathered"]) != expected_rows
        or int(runtime["destination_rows_gathered"]) != expected_rows
        or int(runtime["host_prefetch_consumer_batches"]) != updates
        or producer_delta not in {0, 1}
    ):
        mismatches["endpoint_accounting"] = {
            "expected_rows": expected_rows,
            "runtime": runtime,
        }
    weighted = round0221_nodes._weighted_rejection_accounting_mismatch(
        runtime, producer_delta=producer_delta, updates=updates
    )
    if weighted is not None:
        mismatches["weighted_rejection_accounting"] = weighted
    if mismatches:
        raise Round0223Error(f"R0223 train accounting failed: {mismatches}")
    prompt_contract.synchronize_runtime_counters(accounting, runtime)
    accounting["pipeline_runtime"] = dict(runtime)

    profiler = model._canary_profiler.finalize(
        bench_seconds=model._bench_seconds,
        setup_seconds=getattr(model, "_setup_seconds", None),
    )
    rate = (
        (updates - WARMUP_SUCCESSFUL_UPDATES) / model._bench_seconds
        if model._bench_seconds
        else 0.0
    )
    if profiler.get("aborted") is not False or rate < config["execution"][
        "minimum_train_upd_s"
    ]:
        raise Round0223Error("R0223 train performance admission failed")

    model_path = os.path.join(output, "model.pt")
    atomic_build_new_file(model_path, model.save, immutable=True)
    free_bytes, total_bytes = torch.cuda.mem_get_info("cuda")
    memory = {
        "device_total_bytes": int(total_bytes),
        "post_train_free_bytes": int(free_bytes),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
    }
    del model, wrapper, dataset
    torch.cuda.empty_cache()
    gc.collect()

    from basemap.pumap.parametric_umap import ParametricUMAP

    reloaded = ParametricUMAP.load(model_path, device="cuda")
    coordinates = np.asarray(
        reloaded.transform(source, batch_size=FULL_TRANSFORM_BATCH), dtype=np.float32
    )
    published = validate_full_population_map(coordinates)
    published["model"] = expected_input_signature(model_path)
    del reloaded, coordinates
    torch.cuda.empty_cache()
    gc.collect()

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0223Error(
            f"R0223 train peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    memory["peak_host_rss_gib"] = peak_rss_gib

    receipt = prompt_contract.seal({
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "treatment_config_round_id": "0217",
        "treatment_template_seed": TEMPLATE_SEED,
        "capability": map_capability(seed),
        "capabilities": [map_capability(seed)],
        "training_seed": seed,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "treatment_invariant_sha256": invariant,
        "pipeline_stamp_label_carryover": PIPELINE_STAMP_LABEL_CARRYOVER,
        "model": published["model"],
        "substrate": substrate_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "graph_capability": CUVS_GRAPH_CAPABILITY,
        "graph_builder": dict(graph["manifest"]["builder"]),
        "graph_recall": dict(graph["manifest"]["recall_against_r0220_exact_truth"]),
        "rows": ROWS,
        "dimension": DIMENSION,
        "directed_edges": edges,
        "dose_registration": dose,
        "train_accounting": accounting,
        "exact_execution_receipt": runtime,
        "requested_positive_draws_per_edge": float(
            config["execution"]["target_positive_draws_per_edge"]
        ),
        "consumed_positive_draws": int(updates * POSITIVE_ROWS_PER_UPDATE),
        "consumed_positive_draws_per_edge": float(
            updates * POSITIVE_ROWS_PER_UPDATE / edges
        ),
        "performance_profile": profiler,
        "steady_updates_per_s": rate,
        "train_wall_s": wall,
        "published_map_check": published,
        "train_checks": {
            "exact_update_closure": True,
            "zero_numerical_skips": True,
            "no_pipeline_stamp_drift": True,
            "endpoint_rows_match_updates": True,
            "weighted_rejection_accounting_closes": True,
            "dose_derived_from_sealed_edge_count": True,
            "treatment_identical_to_r0217_except_seed_and_graph": True,
            "published_checkpoint_reloads_finite_and_uncollapsed": True,
            "all_2m_coordinates_finite": True,
        },
        "memory": memory,
        "training_performed": True,
        "optimizer_updates": updates,
        "map_decision_made": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    })
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), receipt, immutable=True
    )
    del source, graph
    gc.collect()


# --------------------------------------------------------------------------- #
# node 3: the panel comparison
# --------------------------------------------------------------------------- #


def _authenticate_cuvs_map(
    cell: Mapping[str, Any], sealed: Mapping[str, Any]
) -> tuple[int, dict[str, Any], dict[str, Any], str]:
    seed = cell.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise Round0223Error(f"R0223 cell seed {seed!r} is not a registered cell")
    capability = map_capability(seed)
    if str(cell.get("capability") or "") != capability:
        raise Round0223Error(f"R0223 seed-{seed} cell capability changed")
    receipt_signature = dict(cell["train_receipt"])
    receipt_path = prompt_contract.verify_signature(
        receipt_signature, label=f"R0223 seed-{seed} train receipt"
    )
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0223 seed-{seed} train receipt"
    )
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("treatment_config_round_id") != "0217"
        or receipt.get("capability") != capability
        or int(receipt.get("training_seed", -1)) != seed
        or receipt.get("training_performed") is not True
        or receipt.get("gate_registerable_here") is not False
        or receipt.get("map_decision_made") is not False
        or int(receipt.get("rows", -1)) != ROWS
        or int(receipt.get("dimension", -1)) != DIMENSION
        or receipt.get("graph_capability") != CUVS_GRAPH_CAPABILITY
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0223Error(f"R0223 seed-{seed} train receipt contract changed")
    if dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"]):
        raise Round0223Error(
            f"R0223 seed-{seed} was not trained on the substrate this panel scores"
        )
    model_path = prompt_contract.verify_signature(
        receipt["model"], label=f"R0223 seed-{seed} published map"
    )
    return seed, receipt, receipt_signature, model_path


def _sealed_gate_artifact(job: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """R0222's sealed n=8 registration: the family and the pending floors."""
    signature = dict(job["r0222_gate_signature"])
    path = prompt_contract.verify_signature(
        signature, label="R0222 sealed n=8 gate registration"
    )
    artifact = prompt_contract.read_sealed(
        path, label="R0222 sealed n=8 gate registration"
    )
    if (
        artifact.get("schema") != R0222_GATE_SCHEMA
        or artifact.get("round_id") != "0222"
        or int(artifact.get("n", -1)) != len(R0222_POOLED_SEEDS)
        or artifact.get("gate_registered") is not True
    ):
        raise Round0223Error("R0222 sealed gate artifact contract changed")
    return artifact, signature


def run_compare(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import reset_process_cuda_peak, sample_anchors, score_panel
    from basemap.pumap.parametric_umap import ParametricUMAP

    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0223Error("R0223 panel scoring requires CUDA")

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    corpus_of_row = round0218_nodes._corpus_of_row(sealed)
    panel_evidence, panel_signature = round0222_nodes._sealed_panel(job)
    gate_artifact, gate_signature = _sealed_gate_artifact(job)

    cells_in = job.get("cells")
    if not isinstance(cells_in, list) or {
        int(cell.get("seed", -1)) for cell in cells_in
    } != set(SEEDS):
        raise Round0223Error("R0223 cell input matrix changed")
    authenticated = {}
    for cell in cells_in:
        seed, receipt, receipt_signature, model_path = _authenticate_cuvs_map(
            cell, sealed
        )
        authenticated[seed] = {
            "receipt": receipt,
            "receipt_signature": receipt_signature,
            "model_path": model_path,
        }
    invariants = {
        str(authenticated[seed]["receipt"]["treatment_invariant_sha256"])
        for seed in SEEDS
    }
    if len(invariants) != 1:
        raise Round0223Error(
            "R0223 scored family is not commensurate: the cells carry "
            f"{len(invariants)} treatment-invariant config digests"
        )
    model_hashes = {
        str(authenticated[seed]["receipt"]["model"]["sha256"]) for seed in SEEDS
    }
    if len(model_hashes) != len(SEEDS):
        raise Round0223Error("R0223 scored family contains a duplicated checkpoint")
    graph_hashes = {
        str(authenticated[seed]["receipt"]["graph"]["sha256"]) for seed in SEEDS
    }
    if len(graph_hashes) != 1:
        raise Round0223Error("R0223 cells were not trained on one cuVS graph")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0223 cuVS-graph map comparison"
    )
    started = time.monotonic()
    reset_process_cuda_peak()

    cfg = prompt_contract.panel_config()
    centroids, centroid_signatures = round0222_nodes._load_centroids(panel_evidence)
    data_identity = {
        "kind": "ordered_array",
        "shape": [ROWS, DIMENSION],
        "dtype": np.dtype("<f4").str,
        "sha256": sealed["ordered_substrate_sha256"],
    }
    reference_identity = {
        "data_identity": data_identity,
        "convention": dict(round0218_nodes.REFERENCE_CONVENTION),
    }
    reference, reference_signature, reference_key_fn = round0222_nodes._load_reference(
        panel_evidence, reference_identity
    )
    anchors = sample_anchors(ROWS, cfg)
    if not np.array_equal(
        np.asarray(anchors, dtype=np.int64),
        np.asarray(reference["anchor_ids"], dtype=np.int64),
    ):
        raise Round0223Error(
            f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} anchor drift"
        )
    rederived_key, _parts = reference_key_fn(
        source, anchors, cfg, centroids, kf=int(reference["kf"]), **reference_identity
    )
    if str(rederived_key) != str(reference["key"]):
        raise Round0223Error(
            f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} re-derived key drift"
        )
    anchor_labels = round0218_nodes._anchor_corpus_labels(corpus_of_row, anchors)
    anchor_corpus_counts = {
        slug: int((anchor_labels == slug).sum()) for slug in CORPUS_SLUGS
    }
    if anchor_corpus_counts != dict(panel_evidence["anchor_corpus_counts"]):
        raise Round0223Error(
            f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} anchor corpus drift"
        )

    cells: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        entry = authenticated[seed]
        model = ParametricUMAP.load(entry["model_path"], device="cuda")
        coordinates = np.asarray(
            model.transform(source, batch_size=8192), dtype=np.float32
        )
        if coordinates.shape != (ROWS, 2):
            raise Round0223Error(
                f"R0223 seed-{seed} transform produced {coordinates.shape}, "
                f"expected ({ROWS}, 2)"
            )
        if not np.isfinite(coordinates).all():
            raise Round0223Error(
                f"R0223 seed-{seed} transform over {ROWS} rows is not finite"
            )
        coordinates_path = os.path.join(output, f"coordinates-seed{seed}.npy")
        atomic_save_new_npy(coordinates_path, coordinates, immutable=True)
        coordinates_signature = expected_input_signature(coordinates_path)
        panel = score_panel(
            source,
            coordinates,
            config=cfg,
            centroids_by_k=centroids,
            hiD_reference=reference,
            reference_identity=reference_identity,
            ffr_group_labels=anchor_labels,
            scale_admission=None,
            provenance={
                "round_id": ROUND_ID,
                "seed": seed,
                "capability": map_capability(seed),
                "universe": "R0216-queue-correction-3-minilm-mixed-2m",
                "graph": "R0223-cuvs-igd48-k15-fuzzy",
                "substrate": dict(sealed["substrate_signature"]),
                "provenance_array": dict(sealed["provenance_signature"]),
                "train_receipt": dict(entry["receipt_signature"]),
                "coordinates": coordinates_signature,
                "shared_high_d_reference": reference_signature,
                "reference_source_round": "0218",
            },
        )
        if not panel_execution_ok(panel):
            raise Round0223Error(f"R0223 seed-{seed} panel is collapsed or nonfinite")
        if not bool(panel["provenance"]["hiD_reference_reused"]):
            raise Round0223Error(
                f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} seed-{seed} recomputed"
            )
        cells[seed] = {
            "seed": seed,
            "capability": map_capability(seed),
            "train_receipt": dict(entry["receipt_signature"]),
            "model": dict(entry["receipt"]["model"]),
            "coordinates": coordinates_signature,
            "coordinates_ordered_sha256": ordered_array_sha256(coordinates),
            "panel": panel,
            "panel_metrics": panel_metric_view(panel),
            "corpus_ffr": corpus_ffr_view(panel),
            "panel_finite_noncollapsed": True,
            "transform_rows_finite": ROWS,
        }
        del model, coordinates
        torch.cuda.empty_cache()
        gc.collect()

    #: The high-D side of every ratio must be literally the same numbers R0218
    #: and R0222 scored against, or the comparison is between two panels.
    reference_hi_d = panel_evidence["cells"][str(R0222_POOLED_SEEDS[0])]["panel"][
        "purity_numerators"
    ]
    for seed in SEEDS:
        mine = cells[seed]["panel"]["purity_numerators"]
        for k in ("k256", "k1024"):
            if float(mine[k]["hi_D_agreement"]) != float(
                reference_hi_d[k]["hi_D_agreement"]
            ):
                raise Round0223Error(
                    f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} seed-{seed} "
                    f"hi-D agreement {k}"
                )

    exact_cells = {
        str(seed): {
            key: float(value)
            for key, value in gate_artifact["pooled_panel_metric_cells"][
                str(seed)
            ].items()
        }
        for seed in R0222_POOLED_SEEDS
    }
    pending_floors = {
        metric: float(gate_artifact["gates"][metric]["floor"])
        for metric in PENDING_FLOOR_METRICS
    }
    cuvs_cells = {str(seed): dict(cells[seed]["panel_metrics"]) for seed in SEEDS}
    # Raw purity ratios, read from the sealed panel payloads rather than
    # reconstructed from the folded fidelity — `exp(-|log r|)` is not invertible
    # in direction, which is exactly review-0222-01's point. Seeds 42-45 come
    # from R0218's sealed cells, 46-49 from R0222's sealed `new_cells`.
    exact_purity_ratios: dict[str, dict[str, float]] = {}
    for seed in R0222_POOLED_SEEDS:
        if str(seed) in panel_evidence["cells"]:
            purity = panel_evidence["cells"][str(seed)]["panel"]["purity"]
        elif str(seed) in (gate_artifact.get("new_cells") or {}):
            purity = gate_artifact["new_cells"][str(seed)]["panel"]["purity"]
        else:
            raise Round0223Error(
                f"R0223 cannot locate a sealed purity ratio for exact cell {seed}"
            )
        exact_purity_ratios[str(seed)] = {
            "k256": float(purity["k256"]),
            "k1024": float(purity["k1024"]),
        }
    cuvs_purity_ratios = {
        str(seed): {
            "k256": float(cells[seed]["panel"]["purity"]["k256"]),
            "k1024": float(cells[seed]["panel"]["purity"]["k1024"]),
        }
        for seed in SEEDS
    }
    comparison = compare_to_exact_family(
        cuvs_cells=cuvs_cells,
        exact_cells=exact_cells,
        pending_floors=pending_floors,
        cuvs_purity_ratios=cuvs_purity_ratios,
        exact_purity_ratios=exact_purity_ratios,
    )

    execution_checks = {
        "all_cells_scored": set(cells) == set(SEEDS),
        "every_metric_finite": all(
            math.isfinite(float(value))
            for cell in cells.values()
            for value in (
                *cell["panel_metrics"].values(),
                *(slice_["ffr"] for slice_ in cell["corpus_ffr"].values()),
            )
        ),
        "no_collapsed_panel": all(
            bool(cell["panel_finite_noncollapsed"]) for cell in cells.values()
        ),
        "map_transform_finite_over_all_rows": all(
            int(cell["transform_rows_finite"]) == ROWS for cell in cells.values()
        ),
        "per_corpus_ffr_slices_complete": all(
            set(cell["corpus_ffr"]) == set(CORPUS_SLUGS) for cell in cells.values()
        ),
        "family_commensurate_one_treatment_invariant_digest": len(invariants) == 1,
        "distinct_checkpoints": len(model_hashes) == len(SEEDS),
        "one_cuvs_graph_across_cells": len(graph_hashes) == 1,
        "reference_byte_identical_to_r0218": True,
        "shared_reference_reused_by_content_key": all(
            bool(cell["panel"]["provenance"]["hiD_reference_reused"])
            for cell in cells.values()
        ),
        "pending_floors_read_from_sealed_r0222_artifact": set(pending_floors)
        == set(PENDING_FLOOR_METRICS),
        "no_gate_released_here": comparison["gate_release_claimed"] is False,
        "no_equivalence_claimed": comparison["equivalence_claimed"] is False,
        "both_floor_families_reported": all(
            "tolerance_floor_95_95" in comparison["per_metric"][metric]
            and "registered_mean_minus_2sd_floor" in comparison["per_metric"][metric]
            for metric in PANEL_METRICS
        ),
        "purity_reported_unfolded_with_direction": all(
            "unfolded" in comparison["per_metric"][metric]
            for metric in ("purity_fidelity_k256", "purity_fidelity_k1024")
        ),
        "tolerance_factor_reproduces_review_0222": bool(
            comparison["tolerance_factor"].get("reproduces_review_0222")
        ),
    }
    if not all(execution_checks.values()):
        raise Round0223Error(f"R0223 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0223Error(
            f"R0223 comparison peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        "schema": COMPARISON_SCHEMA,
        "round_id": ROUND_ID,
        "capability": COMPARISON_CAPABILITY,
        "capabilities": [COMPARISON_CAPABILITY],
        "release_sha": active["manifest"]["release_sha"],
        "outcome": "minilm-mixed-2m-cuvs-graph-maps-scored-on-the-frozen-panel",
        "seeds": list(SEEDS),
        "n": len(SEEDS),
        "map_capabilities": {str(seed): map_capability(seed) for seed in SEEDS},
        "metrics": list(PANEL_METRICS),
        "cuvs_panel_metric_cells": cuvs_cells,
        "cuvs_purity_ratios": cuvs_purity_ratios,
        "exact_family_purity_ratios": exact_purity_ratios,
        "cuvs_corpus_ffr_cells": {
            str(seed): dict(cells[seed]["corpus_ffr"]) for seed in SEEDS
        },
        "exact_family_panel_metric_cells": exact_cells,
        "registered_n8_mean_minus_2sd_floors": pending_floors,
        "tolerance_floors_95_95": {
            metric: comparison["per_metric"][metric]["tolerance_floor_95_95"]
            for metric in PANEL_METRICS
        },
        "n8_floor_status": FLOOR_STATUS,
        "comparison": comparison,
        "evidence_limits": EVIDENCE_LIMITS,
        "cells": {str(seed): cells[seed] for seed in SEEDS},
        "graph_manifest": dict(job["cuvs_graph_manifest_signature"]),
        "r0222_gate_artifact": gate_signature,
        "panel_evidence": panel_signature,
        "lineage": {
            "graph_manifest": dict(sealed["manifest_signature"]),
            "substrate": dict(sealed["substrate_signature"]),
            "provenance": dict(sealed["provenance_signature"]),
            "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        },
        "panel_configuration": {
            "source": "accepted R0113 panel_config()",
            "frac": cfg.frac,
            "k_hit": cfg.k_hit,
            "k_density": cfg.k_density,
            "k_frac_effective": int(reference["kf"]),
            "n_anchors": int(len(anchors)),
            "anchor_seed": cfg.anchor_seed,
            "formula_version": cfg.formula_version,
            "centroid_source": "R0218 published frozen centroid arrays, loaded",
        },
        "shared_high_d_reference": reference_signature,
        "high_d_reference_key": str(reference["key"]),
        "high_d_reference_content_sha256": str(reference["content_sha256"]),
        "reference_convention": dict(round0218_nodes.REFERENCE_CONVENTION),
        "reference_source_round": "0218",
        "reference_byte_identical_to_r0218": True,
        "centroids": centroid_signatures,
        "anchor_corpus_counts": anchor_corpus_counts,
        "treatment_invariant_sha256": sorted(invariants)[0],
        "pipeline_stamp_label_carryover": PIPELINE_STAMP_LABEL_CARRYOVER,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "gate_registered": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "gate_release_claimed": GATE_RELEASE_CLAIMED,
        "map_decision_made": False,
        "production_or_publishing": False,
        "training_performed": False,
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated("cuda")),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved("cuda")),
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "cuvs-graph-map-comparison.json"), receipt, immutable=True
    )
    del source, corpus_of_row, reference, centroids
    gc.collect()


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0223Error("R0223 handler received another queue")
    action = str(job.get("action") or "")
    if action == GRAPH_ACTION:
        run_build_graph(active, job)
    elif action == TRAIN_ACTION:
        run_train(active, job)
    elif action == COMPARE_ACTION:
        run_compare(active, job)
    else:
        raise Round0223Error(f"unknown R0223 action {action!r}")


__all__ = [
    "COMPARE_ACTION",
    "GRAPH_ACTION",
    "MAP_CAPABILITIES",
    "TRAIN_ACTION",
    "run_build_graph",
    "run_compare",
    "run_job",
    "run_train",
]
