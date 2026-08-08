"""Execute R0228 — train maps on `cluster-spill-nnd` graphs and test the geometry.

Node graph (15 nodes, one queue):

* `build_cluster_spill_c4_2m` (GPU, once) runs **R0227's builder script,
  unmodified and imported**, under R0227's own guard and watchdog, for the one
  configuration R0227 never built at 2M: `c = 4`. `c = 8` and `c = 16` reuse
  R0227's sealed 2M neighbour ids read-only, which is the R0223 discipline —
  rebuilding would break the link to the recall level review-0227-01 verified,
  because neither cuVS nn-descent nor the k-means seeding is bit-reproducible
  across runs.
* `fuzzy_graph_c{4,8,16}` (GPU, three nodes) re-measure each graph's recall
  **over all 2,000,000 rows** against R0220's sealed exact truth — strict and
  tie-aware — apply R0171's floors and the R0215 zero-degree tripwire, record
  which rows lost which edges, and symmetrise through R0216's identical fuzzy
  law. A node per configuration, not one node for three, so a defect in the
  third does not discard the first two.
* `train_cluster_spill_c{c}_seed{s}` (GPU, nine cells) is R0223's train node with
  the graph swapped. Each cell rebuilds its config from R0217's own template and
  refuses to train unless the treatment-invariant digest equals the cross-round
  constant `c28cfd61...` that R0217, R0221 and R0223 all carry.
* `compare_cluster_spill_panel` (GPU, once) scores all nine cells on R0218's
  **byte-identical** frozen high-D reference and compares them to the eight-cell
  exact family, to R0223's three cuVS cells, and to R0225's released tolerance
  gates, with exact permutation tests.
* `probe_cluster_spill_geometry` (GPU lease, CPU work, once) runs the geometry
  battery the panel cannot see: R0215's clump detector on all twenty coordinate
  arrays, and the density-matched true-neighbour displacement of the rows that
  actually lost edges, against the exact-graph maps as the null arm.

No gate is registered, none is released, and no equivalence is claimed.
"""
from __future__ import annotations

import gc
import json
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
    ensure_data_directory,
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
    TIE_TOLERANCE,
    TRUTH_SCHEMA as R0220_TRUTH_SCHEMA,
    graph_validity,
    strict_containment_rows,
    summarize,
    tie_aware_rows,
)
from basemap.round0227_low_c_contract import (
    BUILD_SCHEMA as R0227_BUILD_SCHEMA,
    CLUSTER_CAPACITY_ROWS,
    GUARD_BUDGET_NOTE,
    GUARD_DEVICE_BUDGET_BYTES,
    GUARD_HOST_ANON_BUDGET_BYTES,
    GUARD_SWAP_GROWTH_ABORT_BYTES,
    LADDER_SCHEMA as R0227_LADDER_SCHEMA,
    SAMPLE_INTERVAL_S,
    guard_decision,
)
from basemap.round0228_geometry import (
    CLUMP_DEFINITION,
    NULL_ARM_NOTE,
    SCALE_DEFINITION,
    SCATTER_DEFINITION,
    SCATTER_SAMPLE_ROWS,
    SCATTER_SAMPLE_SEED,
    clump_profile,
    density_matched_control,
    displacement_summary,
    map_scale,
    true_neighbour_scatter,
)
from basemap.round0228_low_c_map import (
    ADOPTION_CLAIMED,
    BATCH_SIZE,
    BUILD_SCHEMA,
    CELLS,
    CLUSTERS_BUILT_HERE,
    CLUSTERS_FROM_R0227,
    CLUSTER_COUNTS,
    CLUSTER_SPILL_BUILDER,
    COMPARISON_CAPABILITY,
    COMPARISON_SCHEMA,
    CROSS_CHECKED_CLUSTERS,
    DENSITY_V2_STATUS,
    DIMENSION,
    EQUIVALENCE_CLAIMED,
    EVIDENCE_LIMITS,
    EXACT_FAMILY_SEEDS,
    FULL_TRANSFORM_BATCH,
    FUZZY_LAW,
    FUZZY_RANDOM_STATE_SEED,
    GATED_METRICS,
    GATE_REGISTERABLE_HERE,
    GATE_RELEASE_CLAIMED,
    GEOMETRY_CAPABILITY,
    GEOMETRY_SCHEMA,
    GRAPH_K,
    HOST_RSS_LIMIT_GIB,
    IDENTITY_BOUND_NOTE,
    MIN_ADMISSIBLE_NEGATIVE_DISTANCE,
    POSITIVE_ROWS_PER_UPDATE,
    PRODUCTION_CONFIG_SCHEMA,
    PURITY_RATIO_KEYS,
    R0216_EXACT_KERNEL_MIN_DISTANCE,
    R0216_EXACT_KERNEL_NEGATIVE_ENTRIES,
    R0223_COMPARISON_SCHEMA,
    R0223_CUVS_SEEDS,
    R0225_GATE_SCHEMA,
    R0227_TIE_AWARE_RECALL_BY_C,
    RECALL_CROSS_CHECK_TOLERANCE,
    RECALL_POPULATION,
    RECALL_POPULATION_NOTE,
    ROUND_ID,
    ROWS,
    Round0228Error,
    SEEDS,
    TEMPLATE_SEED,
    TRAIN_SCHEMA,
    assert_configuration_family,
    compare_to_families,
    graph_capability,
    graph_exactness,
    map_capability,
    performance_windows,
    successful_updates_for_edges,
    train_config,
    validate_cluster_spill_graph,
    validate_dose,
    validate_full_population_map,
)
from basemap import round0113_prompt_contrast as prompt_contract
from experiments import round0113_nodes as prompt_nodes
from experiments import (
    round0218_nodes,
    round0221_nodes,
    round0222_nodes,
    round0227_nodes,
)


CLUSTER_BUILD_ACTION = "build_cluster_spill_2m"
FUZZY_ACTION = "build_cluster_spill_fuzzy_graph"
TRAIN_ACTION = "train_cluster_spill_map"
COMPARE_ACTION = "compare_cluster_spill_panel"
GEOMETRY_ACTION = "probe_cluster_spill_geometry"

EVAL_BLOCK = 16_384
#: Rows per block when gathering coordinates for the clump histogram. The
#: coordinate arrays are 16 MB each, so this is comfort rather than necessity.
CLUMP_BLOCK = 1_000_000


# --------------------------------------------------------------------------- #
# node 1 — the one graph R0227 never built at 2M
# --------------------------------------------------------------------------- #
def run_cluster_build(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    """R0227's builder script, unmodified, under R0227's guard and watchdog."""
    clusters = int(job["clusters"])
    if clusters not in CLUSTERS_BUILT_HERE:
        raise Round0228Error(
            f"R0228 c={clusters} is not a configuration this round builds"
        )
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0228 cluster-spill c={clusters} build"
    )
    cache_root = ensure_data_directory(str(job["cuvs_cache_root"]))
    scratch_root = ensure_data_directory(str(job["scratch_root"]))
    repo_root = str(active["manifest"]["repo_root"])
    substrate_signature = dict(job["substrate_signature"])
    substrate_path = prompt_contract.verify_signature(
        substrate_signature, label="R0228 sealed R0216 substrate"
    )
    norm_check = round0227_nodes._assert_unit_norm(substrate_path, rows=ROWS)

    setting_id = f"low-c-n{ROWS}-c{clusters}"
    config = {
        "setting_id": setting_id,
        "candidate": CLUSTER_SPILL_BUILDER,
        "rows": ROWS,
        "clusters": clusters,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "substrate": substrate_path,
        "emit_graph": True,
        "scratch_root": scratch_root,
        "sample_interval_s": SAMPLE_INTERVAL_S,
    }
    guard = guard_decision(rows=ROWS, clusters=clusters)
    build_dir = os.path.join(output, setting_id)
    record = round0227_nodes._run_child(
        command=[
            round0227_nodes.CUML_LAUNCHER,
            os.path.join(repo_root, round0227_nodes.BUILD_SCRIPT),
        ],
        config=config,
        out_dir=build_dir,
        cache_root=cache_root,
        repo_root=repo_root,
        receipt_name="build-receipt.json",
        guard=guard,
    )
    # A refusal or an abort is data, but this round cannot proceed without the
    # graph, so it is recorded and then raised rather than silently skipped.
    ids_path = os.path.join(build_dir, "graph-k15-ids.i32.npy")
    checks = {
        "guard_allowed": bool(guard.get("allowed")),
        "cell_fit": bool(record.get("fit")),
        "not_refused_a_priori": not bool(record.get("refused_a_priori")),
        "not_aborted_by_watchdog": not bool(record.get("aborted_by_watchdog")),
        "not_timed_out": not bool(record.get("timed_out")),
        "no_process_sigkilled": "SIGKILL-last-resort"
        not in (record.get("watchdog_escalations") or []),
        "swap_growth_within_threshold": int(
            record.get("system_swap_growth_bytes") or 0
        )
        <= GUARD_SWAP_GROWTH_ABORT_BYTES,
        "realised_max_cluster_within_capacity": (
            int((record.get("cluster_sizes") or {}).get("max", CLUSTER_CAPACITY_ROWS + 1))
            <= CLUSTER_CAPACITY_ROWS
        ),
        "graph_emitted": os.path.exists(ids_path),
        "substrate_unit_normalised": True,
    }
    receipt = prompt_contract.seal({
        "schema": f"round0228-cluster-spill-c{clusters}-2m-build-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "clusters": clusters,
        "rows": ROWS,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "builder": CLUSTER_SPILL_BUILDER,
        "builder_source_rounds": ["0226", "0227"],
        "builder_script": round0227_nodes.BUILD_SCRIPT,
        "builder_note": (
            "R0227's build script and every scientific parameter inside it are "
            "imported and run unmodified; this round supplies only the cluster "
            "count and the output directory"
        ),
        "exactness": graph_exactness(clusters),
        "substrate": substrate_signature,
        "substrate_norm_check": norm_check,
        "guard": dict(guard),
        "guard_budget_note": GUARD_BUDGET_NOTE,
        "guard_device_budget_bytes": GUARD_DEVICE_BUDGET_BYTES,
        "guard_host_anon_budget_bytes": GUARD_HOST_ANON_BUDGET_BYTES,
        "swap_growth_abort_bytes": GUARD_SWAP_GROWTH_ABORT_BYTES,
        "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
        "build": record,
        "neighbour_ids": (
            expected_input_signature(ids_path) if os.path.exists(ids_path) else None
        ),
        "execution_checks": checks,
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": ADOPTION_CLAIMED,
    })
    atomic_write_new_json(
        os.path.join(output, "cluster-spill-build.json"), receipt, immutable=True
    )
    if not all(checks.values()):
        raise Round0228Error(f"R0228 c={clusters} build checks failed: {checks}")


# --------------------------------------------------------------------------- #
# node 2 — recall over the uniform population, then the fuzzy graph
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


def _sealed_truth(job: Mapping[str, Any]) -> dict[str, Any]:
    truth_signature = dict(job["truth_receipt_signature"])
    truth_path = prompt_contract.verify_signature(
        truth_signature, label="R0220 exact k15 truth receipt"
    )
    truth = prompt_contract.read_sealed(
        truth_path, label="R0220 exact k15 truth receipt"
    )
    if (
        truth.get("schema") != R0220_TRUTH_SCHEMA
        or truth.get("round_id") != "0220"
        or int(truth.get("rows", -1)) != ROWS
        or int(truth.get("k", -1)) != GRAPH_K
        or not truth["probe"]["passed"]
    ):
        raise Round0228Error("R0220 truth receipt contract changed")
    return {
        "signature": truth_signature,
        "ids_path": prompt_contract.verify_signature(
            truth["outputs"]["ids"], label="R0220 truth ids"
        ),
        "cosines_path": prompt_contract.verify_signature(
            truth["outputs"]["cosines"], label="R0220 truth cosines"
        ),
        "ids_signature": dict(truth["outputs"]["ids"]),
        "cosines_signature": dict(truth["outputs"]["cosines"]),
    }


def _neighbour_ids_source(job: Mapping[str, Any], clusters: int) -> dict[str, Any]:
    """Where this configuration's k15 ids come from, and proof of what they are."""
    reference = dict(job["neighbour_ids_reference"])
    path = str(reference["canonical_path"])
    if reference.get("sha256"):
        path = prompt_contract.verify_signature(
            reference, label=f"R0228 c={clusters} neighbour ids"
        )
        signature = dict(reference)
    else:
        if not os.path.exists(path):
            raise Round0228Error(f"R0228 c={clusters} neighbour ids absent at {path}")
        signature = expected_input_signature(path)
    provenance = dict(job["neighbour_ids_provenance"])
    receipt_reference = dict(job["source_build_receipt"])
    receipt_path = (
        prompt_contract.verify_signature(
            receipt_reference, label=f"R0228 c={clusters} source build receipt"
        )
        if receipt_reference.get("sha256")
        else str(receipt_reference["canonical_path"])
    )
    if not os.path.exists(receipt_path):
        raise Round0228Error(f"R0228 c={clusters} source build receipt is absent")
    with open(receipt_path, encoding="utf-8") as handle:
        child = json.load(handle)
    if (
        str(child.get("schema") or "") not in {R0227_BUILD_SCHEMA}
        or int(child.get("clusters", -1)) != clusters
        or int(child.get("rows", -1)) != ROWS
        or int(child.get("k", -1)) != GRAPH_K
        or child.get("fit") is not True
        or child.get("graph_emitted") is not True
        or int(child.get("zero_degree_rows", -1)) != 0
        or int(child.get("rows_below_k", -1)) != 0
    ):
        raise Round0228Error(
            f"R0228 c={clusters} source build receipt is not a fitted "
            f"cluster-spill build with a complete graph: {receipt_path}"
        )
    return {
        "path": path,
        "signature": signature,
        "provenance": provenance,
        "source_build_receipt": expected_input_signature(receipt_path),
        "source_build_child": {
            key: child.get(key)
            for key in (
                "setting_id",
                "clusters",
                "rows",
                "spill",
                "graph_degree",
                "intermediate_graph_degree",
                "max_iterations",
                "metric",
                "seed",
                "cuvs_version",
                "cluster_sizes",
                "spill_groups",
                "zero_degree_rows",
                "rows_below_k",
                "min_degree",
                "builder_seconds",
            )
        },
    }


def run_fuzzy_graph(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    clusters = int(job["clusters"])
    if clusters not in CLUSTER_COUNTS:
        raise Round0228Error(f"R0228 c={clusters} is not a registered configuration")
    sealed = round0218_nodes._sealed_substrate(job)
    substrate_path = prompt_contract.verify_signature(
        sealed["substrate_signature"], label="R0228 sealed R0216 substrate"
    )
    truth = _sealed_truth(job)
    source = _neighbour_ids_source(job, clusters)

    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0228 c={clusters} fuzzy graph"
    )
    started = time.monotonic()

    raw = np.load(source["path"], allow_pickle=False)
    if raw.shape != (ROWS, GRAPH_K):
        raise Round0228Error(
            f"R0228 c={clusters} neighbour ids are {raw.shape}, expected "
            f"({ROWS}, {GRAPH_K})"
        )
    leading = np.ascontiguousarray(raw.astype(np.int32))
    del raw
    as_int = leading.astype(np.int64)
    if int(as_int.min()) < 0 or int(as_int.max()) >= ROWS:
        raise Round0228Error(f"R0228 c={clusters} carries out-of-range neighbour ids")
    del as_int

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    host = np.load(substrate_path, mmap_mode="r", allow_pickle=False)
    if host.shape != (ROWS, DIMENSION) or host.dtype != np.float32:
        raise Round0228Error("R0228 sealed substrate geometry changed")
    tensor = torch.from_numpy(
        np.array(host, dtype=np.float32, order="C", copy=True)
    ).to(device)

    cosine_started = time.monotonic()
    candidate_cos = _cosines_for(torch, tensor, leading)
    cosine_s = time.monotonic() - cosine_started
    del tensor
    torch.cuda.empty_cache()
    gc.collect()

    truth_ids = np.load(truth["ids_path"], allow_pickle=False)
    truth_cos = np.load(truth["cosines_path"], allow_pickle=False)
    if truth_ids.shape != (ROWS, GRAPH_K) or truth_cos.shape != (ROWS, GRAPH_K):
        raise Round0228Error("R0220 truth arrays have the wrong shape")
    kth = truth_cos[:, GRAPH_K - 1].astype(np.float64)

    # ---- recall over the UNIFORM population: every one of the 2,000,000 rows.
    strict = strict_containment_rows(leading, truth_ids)
    tie = tie_aware_rows(candidate_cos.astype(np.float64), leading, kth)
    strict_summary = summarize(strict, label=f"R0228 c={clusters} strict recall@15")
    tie_summary = summarize(tie, label=f"R0228 c={clusters} tie-aware recall@15")

    cross_check: dict[str, Any] = {
        "population": RECALL_POPULATION,
        "population_note": RECALL_POPULATION_NOTE,
        "cross_checked": clusters in CROSS_CHECKED_CLUSTERS,
    }
    if clusters in CROSS_CHECKED_CLUSTERS:
        published = float(R0227_TIE_AWARE_RECALL_BY_C[clusters])
        delta = float(tie_summary["mean"]) - published
        cross_check.update({
            "published_tie_aware_mean": published,
            "measured_tie_aware_mean": float(tie_summary["mean"]),
            "tie_aware_delta": delta,
            "tolerance": RECALL_CROSS_CHECK_TOLERANCE,
        })
        if abs(delta) > RECALL_CROSS_CHECK_TOLERANCE:
            raise Round0228Error(
                f"R0228 c={clusters} re-measured tie-aware recall does not "
                f"reproduce R0227's published value: {cross_check}"
            )
        cross_check["reproduces_r0227"] = True
    elif clusters in CLUSTERS_FROM_R0227:
        cross_check["published_tie_aware_mean"] = float(
            R0227_TIE_AWARE_RECALL_BY_C[clusters]
        )
        cross_check["measured_tie_aware_mean"] = float(tie_summary["mean"])
        cross_check["tie_aware_delta"] = float(tie_summary["mean"]) - float(
            R0227_TIE_AWARE_RECALL_BY_C[clusters]
        )

    structural = graph_validity(leading, rows=ROWS)

    # ---- which rows actually lost edges, and how much. This is what the
    # geometry node needs, and it is a per-row array over the whole population.
    lost_edges_per_row = np.rint((1.0 - strict) * GRAPH_K).astype(np.int16)
    rows_carrying_loss = int((lost_edges_per_row > 0).sum())
    strict_path = atomic_save_new_npy(
        os.path.join(output, "strict-recall-per-row.f32.npy"),
        strict.astype(np.float32),
        immutable=True,
    )
    lost_path = atomic_save_new_npy(
        os.path.join(output, "lost-edges-per-row.i16.npy"),
        lost_edges_per_row,
        immutable=True,
    )
    del truth_ids, truth_cos

    # ---- density deciles of the loss, on the row's own true 15th cosine.
    order = np.argsort(kth, kind="stable")
    decile_recall: list[float] = []
    for index in range(10):
        lo = index * ROWS // 10
        hi = (index + 1) * ROWS // 10
        decile_recall.append(float(tie[order[lo:hi]].mean()))
    del order

    # ---- the fuzzy stage, R0216's law, only the ids differ.
    sort_order = np.argsort(-candidate_cos, axis=1, kind="stable")
    already_sorted = int(
        (sort_order == np.arange(GRAPH_K, dtype=sort_order.dtype)[None, :])
        .all(axis=1)
        .sum()
    )
    ids_sorted = np.take_along_axis(leading, sort_order, axis=1).astype(np.int32)
    cos_sorted = np.take_along_axis(candidate_cos, sort_order, axis=1)
    del sort_order, leading, candidate_cos, strict, tie

    dists = (1.0 - cos_sorted).astype(np.float32)
    negative = int((dists < 0.0).sum())
    most_negative = float(dists.min()) if negative else 0.0
    if most_negative < MIN_ADMISSIBLE_NEGATIVE_DISTANCE:
        raise Round0228Error(
            f"R0228 c={clusters} found a cosine distance of {most_negative!r}, "
            f"below the registered {MIN_ADMISSIBLE_NEGATIVE_DISTANCE} floor"
        )
    np.maximum(dists, 0.0, out=dists)
    if not np.isfinite(dists).all():
        raise Round0228Error(f"R0228 c={clusters} candidate distances are not finite")
    del cos_sorted

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
        raise Round0228Error(f"R0228 c={clusters} fuzzy weights are invalid")
    if np.any(np.diff(src) < 0):
        raise Round0228Error(f"R0228 c={clusters} fuzzy edge sources are not sorted")
    degree_counts = np.bincount(src, minlength=ROWS)
    degrees = {
        "zero_degree_rows": int((degree_counts == 0).sum()),
        "min": int(degree_counts.min()),
        "median": float(np.median(degree_counts)),
        "mean": float(degree_counts.mean()),
        "max": int(degree_counts.max()),
    }
    checks = validate_cluster_spill_graph(
        clusters=clusters,
        degrees=degrees,
        recall={
            "mean_recall_at_k": float(tie_summary["mean"]),
            "p10_recall_at_k": float(tie_summary["p10"]),
        },
        edges=int(len(src)),
        structural=structural,
    )

    ids_path = atomic_save_new_npy(
        os.path.join(output, "cluster-spill-k15-ids.i32.npy"),
        ids_sorted,
        immutable=True,
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
        raise Round0228Error(
            f"R0228 c={clusters} graph build peak RSS {peak_rss_gib:.2f} GiB "
            f"exceeds {HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        "schema": BUILD_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "capability": graph_capability(clusters),
        "capabilities": [graph_capability(clusters)],
        "clusters": clusters,
        "rows": ROWS,
        "dimension": DIMENSION,
        "k": GRAPH_K,
        "builder": {
            "name": CLUSTER_SPILL_BUILDER,
            "source_rounds": ["0226", "0227"],
            "neighbour_ids": source["signature"],
            "provenance": source["provenance"],
            "source_build_receipt": source["source_build_receipt"],
            "source_build": source["source_build_child"],
            "approximate": True,
            "exactness": graph_exactness(clusters),
        },
        "recall_against_r0220_exact_truth": {
            "truth_receipt": truth["signature"],
            "truth_ids": truth["ids_signature"],
            "truth_cosines": truth["cosines_signature"],
            "rows_measured": ROWS,
            "population": RECALL_POPULATION,
            "population_note": RECALL_POPULATION_NOTE,
            "tie_aware": tie_summary,
            "strict": strict_summary,
            "tie_tolerance": TIE_TOLERANCE,
            "cross_check": cross_check,
            "density_decile_tie_aware": decile_recall,
            "density_decile_definition": (
                "deciles of the row's own true 15th-best cosine; decile 0 is the "
                "sparsest local neighbourhood, decile 9 the densest"
            ),
            "sparsest_decile_mean": decile_recall[0],
            "densest_decile_mean": decile_recall[-1],
            "sparsest_to_densest_gap": decile_recall[-1] - decile_recall[0],
            "rows_carrying_any_loss": rows_carrying_loss,
            "rows_carrying_any_loss_fraction": rows_carrying_loss / ROWS,
            "total_missing_true_edges": int(lost_edges_per_row.sum()),
        },
        "loss_arrays": {
            "strict_recall_per_row": expected_input_signature(strict_path),
            "lost_edges_per_row": expected_input_signature(lost_path),
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
            "r0216_exact_kernel_negative_entries": R0216_EXACT_KERNEL_NEGATIVE_ENTRIES,
            "r0216_exact_kernel_min_distance": R0216_EXACT_KERNEL_MIN_DISTANCE,
        },
        "substrate": dict(sealed["substrate_signature"]),
        "provenance": dict(sealed["provenance_signature"]),
        "r0216_graph_manifest": dict(sealed["manifest_signature"]),
        "ordered_substrate_sha256": sealed["ordered_substrate_sha256"],
        "cluster_spill_k15_ids": expected_input_signature(ids_path),
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
        "adoption_claimed": ADOPTION_CLAIMED,
        "map_quality_claim_available": False,
    })
    atomic_write_new_json(
        os.path.join(output, "cluster-spill-graph.json"), receipt, immutable=True
    )
    del ids_sorted, dists, src, dst, wts
    gc.collect()


# --------------------------------------------------------------------------- #
# node 3 — one train cell
# --------------------------------------------------------------------------- #
def _intra_queue_signature(
    reference: Mapping[str, Any], *, label: str
) -> tuple[str, dict[str, Any]]:
    """Resolve a reference to an artifact produced earlier in THIS queue."""
    reference = dict(reference)
    if reference.get("sha256"):
        path = prompt_contract.verify_signature(reference, label=label)
        return path, reference
    path = str(reference["canonical_path"])
    if not os.path.exists(path):
        raise Round0228Error(f"{label} is absent at {path}")
    return path, expected_input_signature(path)


def _cell(job: Mapping[str, Any]) -> tuple[int, int]:
    clusters = job.get("clusters")
    seed = job.get("training_seed")
    for value in (clusters, seed):
        if isinstance(value, bool) or not isinstance(value, int):
            raise Round0228Error(f"R0228 job cell {(clusters, seed)!r} is malformed")
    if (int(clusters), int(seed)) not in CELLS:
        raise Round0228Error(f"R0228 job cell {(clusters, seed)!r} is not registered")
    if str(job.get("capability") or "") != map_capability(int(clusters), int(seed)):
        raise Round0228Error("R0228 job capability does not match its cell")
    return int(clusters), int(seed)


def _sealed_cluster_spill_graph(
    job: Mapping[str, Any], clusters: int
) -> dict[str, Any]:
    manifest_path, manifest_signature = _intra_queue_signature(
        job["graph_manifest_signature"],
        label=f"R0228 sealed c={clusters} graph receipt",
    )
    manifest = prompt_contract.read_sealed(
        manifest_path, label=f"R0228 sealed c={clusters} graph receipt"
    )
    checks = manifest.get("graph_checks") or {}
    degrees = manifest.get("degrees") or {}
    if (
        manifest.get("schema") != BUILD_SCHEMA
        or manifest.get("round_id") != ROUND_ID
        or manifest.get("capability") != graph_capability(clusters)
        or int(manifest.get("clusters", -1)) != clusters
        or int(manifest.get("rows", -1)) != ROWS
        or int(manifest.get("dimension", -1)) != DIMENSION
        or int(manifest.get("k", -1)) != GRAPH_K
        or manifest.get("training_performed") is not False
        or manifest["recall_against_r0220_exact_truth"]["population"]
        != RECALL_POPULATION
    ):
        raise Round0228Error(f"R0228 sealed c={clusters} graph contract changed")
    if (
        int(checks.get("zero_degree_rows", -1)) != 0
        or int(degrees.get("zero_degree_rows", -1)) != 0
        or float(checks.get("mean_recall_at_k", 0.0))
        < float(checks.get("mean_recall_floor", 1.0))
        or float(checks.get("p10_recall_at_k", 0.0))
        < float(checks.get("p10_recall_floor", 1.0))
    ):
        raise Round0228Error(
            f"R0228 requires the sealed c={clusters} graph to have passed its "
            "recall and zero-degree checks"
        )
    edges = int(manifest.get("directed_edge_count", 0)) or int(
        checks.get("directed_edges", 0)
    )
    if edges <= 0:
        raise Round0228Error(f"R0228 sealed c={clusters} graph reports no edges")
    graph_signature = dict(manifest["graph"])
    graph_path = prompt_contract.verify_signature(
        graph_signature, label=f"R0228 sealed c={clusters} fuzzy graph"
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
        raise Round0228Error(f"R0228 sealed c={clusters} graph arrays changed")
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
    """Serve the 3.07 GB sealed substrate lazily; never materialize it."""
    signature = dict(graph["manifest"]["substrate"])
    path = prompt_contract.verify_signature(
        signature, label="R0228 sealed R0216 substrate"
    )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, DIMENSION) or array.dtype != np.float32:
        raise Round0228Error("R0228 sealed R0216 substrate geometry changed")
    return array, signature


def run_train(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch

    clusters, seed = _cell(job)
    graph = _sealed_cluster_spill_graph(job, clusters)
    edges = graph["directed_edges"]
    updates = successful_updates_for_edges(edges)
    dose = validate_dose(updates=updates, edge_count=edges)
    declared_bound = job.get("registered_dose_bound")
    if declared_bound is not None and updates > int(declared_bound):
        raise Round0228Error(
            "R0228 derived update horizon exceeds the registered round bound"
        )

    source, substrate_signature = _open_substrate(graph)
    config, config_sha, invariant = train_config(
        clusters=clusters,
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
        raise Round0228Error(
            "R0228 cell config is not R0217's treatment outside the seed and the "
            f"graph: {invariant} != {declared_invariant}"
        )

    output = create_fresh_directory(str(job["outputs"][0]), label="R0228 train output")
    config_path = os.path.join(output, "production-config.json")
    atomic_write_new_json(
        config_path,
        {
            "schema": PRODUCTION_CONFIG_SCHEMA,
            "round_id": ROUND_ID,
            "treatment_config_round_id": "0217",
            "treatment_template_seed": TEMPLATE_SEED,
            "clusters": clusters,
            "seed": seed,
            "capability": map_capability(clusters, seed),
            "treatment_invariant_sha256": invariant,
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
        raise Round0228Error(f"R0228 train accounting failed: {mismatches}")
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
        raise Round0228Error("R0228 train performance admission failed")

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
        raise Round0228Error(
            f"R0228 train peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    memory["peak_host_rss_gib"] = peak_rss_gib

    receipt = prompt_contract.seal({
        "schema": TRAIN_SCHEMA,
        "round_id": ROUND_ID,
        "treatment_config_round_id": "0217",
        "treatment_template_seed": TEMPLATE_SEED,
        "capability": map_capability(clusters, seed),
        "capabilities": [map_capability(clusters, seed)],
        "clusters": clusters,
        "training_seed": seed,
        "release_sha": active["manifest"]["release_sha"],
        "production_config": expected_input_signature(config_path),
        "production_config_sha256": config_sha,
        "treatment_invariant_sha256": invariant,
        "model": published["model"],
        "substrate": substrate_signature,
        "graph_manifest": graph["manifest_signature"],
        "graph": graph["signature"],
        "graph_capability": graph_capability(clusters),
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
            "treatment_digest_equals_cross_round_constant": True,
            "published_checkpoint_reloads_finite_and_uncollapsed": True,
            "all_2m_coordinates_finite": True,
        },
        "memory": memory,
        "training_performed": True,
        "optimizer_updates": updates,
        "map_decision_made": False,
        "adoption_claimed": ADOPTION_CLAIMED,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
    })
    atomic_write_new_json(
        os.path.join(output, "train-receipt.json"), receipt, immutable=True
    )
    del source, graph
    gc.collect()


# --------------------------------------------------------------------------- #
# node 4 — the panel comparison
# --------------------------------------------------------------------------- #
def _authenticate_cell(
    cell: Mapping[str, Any], sealed: Mapping[str, Any]
) -> tuple[int, int, dict[str, Any], dict[str, Any], str]:
    clusters = cell.get("clusters")
    seed = cell.get("seed")
    if (int(clusters), int(seed)) not in CELLS:
        raise Round0228Error(f"R0228 cell {(clusters, seed)!r} is not registered")
    capability = map_capability(int(clusters), int(seed))
    if str(cell.get("capability") or "") != capability:
        raise Round0228Error(f"R0228 cell {capability} capability changed")
    receipt_path, receipt_signature = _intra_queue_signature(
        cell["train_receipt"], label=f"R0228 {capability} train receipt"
    )
    receipt = prompt_contract.read_sealed(
        receipt_path, label=f"R0228 {capability} train receipt"
    )
    train_checks = receipt.get("train_checks") or {}
    if (
        receipt.get("schema") != TRAIN_SCHEMA
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("treatment_config_round_id") != "0217"
        or receipt.get("capability") != capability
        or int(receipt.get("clusters", -1)) != int(clusters)
        or int(receipt.get("training_seed", -1)) != int(seed)
        or receipt.get("training_performed") is not True
        or receipt.get("map_decision_made") is not False
        or int(receipt.get("rows", -1)) != ROWS
        or int(receipt.get("dimension", -1)) != DIMENSION
        or receipt.get("graph_capability") != graph_capability(int(clusters))
        or not train_checks
        or not all(bool(value) for value in train_checks.values())
    ):
        raise Round0228Error(f"R0228 {capability} train receipt contract changed")
    if dict(receipt.get("substrate") or {}) != dict(sealed["substrate_signature"]):
        raise Round0228Error(
            f"R0228 {capability} was not trained on the substrate this panel scores"
        )
    model_path = prompt_contract.verify_signature(
        receipt["model"], label=f"R0228 {capability} published map"
    )
    return int(clusters), int(seed), receipt, receipt_signature, model_path


def _sealed_json(job: Mapping[str, Any], key: str, *, label: str) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    signature = dict(job[key])
    path = prompt_contract.verify_signature(signature, label=label)
    return prompt_contract.read_sealed(path, label=label), signature


def run_compare(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    import torch
    from basemap.panel_v2 import reset_process_cuda_peak, sample_anchors, score_panel
    from basemap.pumap.parametric_umap import ParametricUMAP

    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise Round0228Error("R0228 panel scoring requires CUDA")

    sealed = round0218_nodes._sealed_substrate(job)
    source = round0218_nodes._open_substrate(sealed)
    corpus_of_row = round0218_nodes._corpus_of_row(sealed)
    panel_evidence, panel_signature = round0222_nodes._sealed_panel(job)
    gate_artifact, gate_signature = _sealed_json(
        job, "r0222_gate_signature", label="R0222 sealed n=8 gate registration"
    )
    tolerance_artifact, tolerance_signature = _sealed_json(
        job, "r0225_gate_signature", label="R0225 sealed tolerance gates"
    )
    cuvs_artifact, cuvs_signature = _sealed_json(
        job, "r0223_comparison_signature", label="R0223 sealed cuVS map comparison"
    )
    if (
        tolerance_artifact.get("schema") != R0225_GATE_SCHEMA
        or tolerance_artifact.get("round_id") != "0225"
        or tolerance_artifact.get("gate_registered") is not True
    ):
        raise Round0228Error("R0225 sealed tolerance gate contract changed")
    if (
        cuvs_artifact.get("schema") != R0223_COMPARISON_SCHEMA
        or cuvs_artifact.get("round_id") != "0223"
        or [int(value) for value in cuvs_artifact.get("seeds") or []]
        != list(R0223_CUVS_SEEDS)
    ):
        raise Round0228Error("R0223 sealed comparison contract changed")
    tolerance_gates = dict(tolerance_artifact["gate"]["gates"])
    if set(tolerance_gates) != set(PANEL_METRICS):
        raise Round0228Error("R0225 tolerance gates do not cover the panel metrics")

    cells_in = job.get("cells")
    if not isinstance(cells_in, list) or {
        (int(cell.get("clusters", -1)), int(cell.get("seed", -1))) for cell in cells_in
    } != set(CELLS):
        raise Round0228Error("R0228 cell input matrix changed")
    authenticated: dict[tuple[int, int], dict[str, Any]] = {}
    for cell in cells_in:
        clusters, seed, receipt, receipt_signature, model_path = _authenticate_cell(
            cell, sealed
        )
        authenticated[(clusters, seed)] = {
            "receipt": receipt,
            "receipt_signature": receipt_signature,
            "model_path": model_path,
        }
    invariants = {
        str(authenticated[key]["receipt"]["treatment_invariant_sha256"])
        for key in CELLS
    }
    if len(invariants) != 1:
        raise Round0228Error(
            "R0228 scored family is not commensurate: the cells carry "
            f"{len(invariants)} treatment-invariant config digests"
        )
    model_hashes = {
        str(authenticated[key]["receipt"]["model"]["sha256"]) for key in CELLS
    }
    if len(model_hashes) != len(CELLS):
        raise Round0228Error("R0228 scored family contains a duplicated checkpoint")
    graph_hashes_by_c = {
        clusters: {
            str(authenticated[(clusters, seed)]["receipt"]["graph"]["sha256"])
            for seed in SEEDS
        }
        for clusters in CLUSTER_COUNTS
    }
    if any(len(values) != 1 for values in graph_hashes_by_c.values()):
        raise Round0228Error(
            "R0228 cells of one configuration were not trained on one graph"
        )
    if len({next(iter(values)) for values in graph_hashes_by_c.values()}) != len(
        CLUSTER_COUNTS
    ):
        raise Round0228Error("R0228 configurations share a graph")

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0228 cluster-spill map comparison"
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
        raise Round0228Error(
            f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} anchor drift"
        )
    rederived_key, _parts = reference_key_fn(
        source, anchors, cfg, centroids, kf=int(reference["kf"]), **reference_identity
    )
    if str(rederived_key) != str(reference["key"]):
        raise Round0228Error(
            f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} re-derived key drift"
        )
    anchor_labels = round0218_nodes._anchor_corpus_labels(corpus_of_row, anchors)
    anchor_corpus_counts = {
        slug: int((anchor_labels == slug).sum()) for slug in CORPUS_SLUGS
    }
    if anchor_corpus_counts != dict(panel_evidence["anchor_corpus_counts"]):
        raise Round0228Error(
            f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} anchor corpus drift"
        )

    cells: dict[tuple[int, int], dict[str, Any]] = {}
    for clusters, seed in CELLS:
        entry = authenticated[(clusters, seed)]
        capability = map_capability(clusters, seed)
        model = ParametricUMAP.load(entry["model_path"], device="cuda")
        coordinates = np.asarray(
            model.transform(source, batch_size=FULL_TRANSFORM_BATCH), dtype=np.float32
        )
        if coordinates.shape != (ROWS, 2) or not np.isfinite(coordinates).all():
            raise Round0228Error(f"R0228 {capability} transform is invalid")
        coordinates_path = os.path.join(
            output, f"coordinates-c{clusters}-seed{seed}.npy"
        )
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
                "clusters": clusters,
                "seed": seed,
                "capability": capability,
                "universe": "R0216-queue-correction-3-minilm-mixed-2m",
                "graph": f"R0228-cluster-spill-c{clusters}-k15-fuzzy",
                "substrate": dict(sealed["substrate_signature"]),
                "provenance_array": dict(sealed["provenance_signature"]),
                "train_receipt": dict(entry["receipt_signature"]),
                "coordinates": coordinates_signature,
                "shared_high_d_reference": reference_signature,
                "reference_source_round": "0218",
            },
        )
        if not panel_execution_ok(panel):
            raise Round0228Error(f"R0228 {capability} panel is collapsed or nonfinite")
        if not bool(panel["provenance"]["hiD_reference_reused"]):
            raise Round0228Error(
                f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} {capability} recomputed"
            )
        cells[(clusters, seed)] = {
            "clusters": clusters,
            "seed": seed,
            "capability": capability,
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

    reference_hi_d = panel_evidence["cells"][str(EXACT_FAMILY_SEEDS[0])]["panel"][
        "purity_numerators"
    ]
    for key in CELLS:
        mine = cells[key]["panel"]["purity_numerators"]
        for granularity in ("k256", "k1024"):
            if float(mine[granularity]["hi_D_agreement"]) != float(
                reference_hi_d[granularity]["hi_D_agreement"]
            ):
                raise Round0228Error(
                    f"{round0222_nodes.REFERENCE_MISMATCH_MESSAGE} {key} "
                    f"hi-D agreement {granularity}"
                )

    exact_cells = {
        str(seed): {
            metric: float(value)
            for metric, value in gate_artifact["pooled_panel_metric_cells"][
                str(seed)
            ].items()
        }
        for seed in EXACT_FAMILY_SEEDS
    }
    exact_purity_ratios: dict[str, dict[str, float]] = {}
    for seed in EXACT_FAMILY_SEEDS:
        if str(seed) in panel_evidence["cells"]:
            purity = panel_evidence["cells"][str(seed)]["panel"]["purity"]
        elif str(seed) in (gate_artifact.get("new_cells") or {}):
            purity = gate_artifact["new_cells"][str(seed)]["panel"]["purity"]
        else:
            raise Round0228Error(
                f"R0228 cannot locate a sealed purity ratio for exact cell {seed}"
            )
        exact_purity_ratios[str(seed)] = {
            "k256": float(purity["k256"]),
            "k1024": float(purity["k1024"]),
        }
    cuvs_cells = {
        str(seed): {
            metric: float(value)
            for metric, value in cuvs_artifact["cuvs_panel_metric_cells"][
                str(seed)
            ].items()
        }
        for seed in R0223_CUVS_SEEDS
    }
    candidate_cells = {
        str(clusters): {
            str(seed): dict(cells[(clusters, seed)]["panel_metrics"]) for seed in SEEDS
        }
        for clusters in CLUSTER_COUNTS
    }
    candidate_purity_ratios = {
        str(clusters): {
            str(seed): {
                "k256": float(cells[(clusters, seed)]["panel"]["purity"]["k256"]),
                "k1024": float(cells[(clusters, seed)]["panel"]["purity"]["k1024"]),
            }
            for seed in SEEDS
        }
        for clusters in CLUSTER_COUNTS
    }
    comparison = compare_to_families(
        candidate_cells=candidate_cells,
        exact_cells=exact_cells,
        cuvs_cells=cuvs_cells,
        tolerance_gates=tolerance_gates,
        candidate_purity_ratios=candidate_purity_ratios,
        exact_purity_ratios=exact_purity_ratios,
    )

    graph_recall = {
        str(clusters): dict(
            authenticated[(clusters, SEEDS[0])]["receipt"]["graph_recall"]
        )
        for clusters in CLUSTER_COUNTS
    }
    execution_checks = {
        "all_cells_scored": set(cells) == set(CELLS),
        "nine_cells": len(cells) == len(CELLS),
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
        "distinct_checkpoints": len(model_hashes) == len(CELLS),
        "one_graph_per_configuration": all(
            len(values) == 1 for values in graph_hashes_by_c.values()
        ),
        "configurations_use_distinct_graphs": len(
            {next(iter(values)) for values in graph_hashes_by_c.values()}
        )
        == len(CLUSTER_COUNTS),
        "reference_byte_identical_to_r0218": True,
        "shared_reference_reused_by_content_key": all(
            bool(cell["panel"]["provenance"]["hiD_reference_reused"])
            for cell in cells.values()
        ),
        "tolerance_gates_read_from_sealed_r0225_artifact": set(tolerance_gates)
        == set(PANEL_METRICS),
        "cuvs_arm_read_from_sealed_r0223_artifact": set(cuvs_cells)
        == {str(seed) for seed in R0223_CUVS_SEEDS},
        "density_v2_held_descriptive_only": "density_v2" not in GATED_METRICS,
        "purity_reported_unfolded_two_sided": all(
            "unfolded_two_sided" in comparison["per_metric"][metric]
            for metric in ("purity_fidelity_k256", "purity_fidelity_k1024")
        ),
        "every_metric_carries_a_trend_test": all(
            "trend_in_log2_c" in comparison["per_metric"][metric]
            for metric in PANEL_METRICS
        ),
        "every_configuration_carries_a_permutation_test": all(
            "permutation_vs_exact_family"
            in comparison["per_metric"][metric]["by_clusters"][str(clusters)]
            for metric in PANEL_METRICS
            for clusters in CLUSTER_COUNTS
        ),
        "recall_scored_on_the_uniform_population": all(
            graph_recall[str(clusters)]["population"] == RECALL_POPULATION
            for clusters in CLUSTER_COUNTS
        ),
        "no_gate_released_here": comparison["gate_release_claimed"] is False,
        "no_equivalence_claimed": comparison["equivalence_claimed"] is False,
        "no_adoption_claimed": comparison["adoption_claimed"] is False,
    }
    if not all(execution_checks.values()):
        raise Round0228Error(f"R0228 execution checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    if peak_rss_gib > HOST_RSS_LIMIT_GIB:
        raise Round0228Error(
            f"R0228 comparison peak RSS {peak_rss_gib:.2f} GiB exceeds "
            f"{HOST_RSS_LIMIT_GIB:.0f} GiB"
        )
    receipt = prompt_contract.seal({
        "schema": COMPARISON_SCHEMA,
        "round_id": ROUND_ID,
        "capability": COMPARISON_CAPABILITY,
        "capabilities": [COMPARISON_CAPABILITY],
        "release_sha": active["manifest"]["release_sha"],
        "outcome": "cluster-spill-graph-maps-scored-on-the-frozen-panel",
        "cluster_counts": list(CLUSTER_COUNTS),
        "seeds": list(SEEDS),
        "n_per_configuration": len(SEEDS),
        "n_total": len(CELLS),
        "map_capabilities": {
            f"{clusters}:{seed}": map_capability(clusters, seed)
            for clusters, seed in CELLS
        },
        "metrics": list(PANEL_METRICS),
        "candidate_panel_metric_cells": candidate_cells,
        "candidate_purity_ratios": candidate_purity_ratios,
        "candidate_corpus_ffr_cells": {
            str(clusters): {
                str(seed): dict(cells[(clusters, seed)]["corpus_ffr"])
                for seed in SEEDS
            }
            for clusters in CLUSTER_COUNTS
        },
        "exact_family_panel_metric_cells": exact_cells,
        "exact_family_purity_ratios": exact_purity_ratios,
        "r0223_cuvs_panel_metric_cells": cuvs_cells,
        "graph_recall_by_clusters": graph_recall,
        "recall_population": RECALL_POPULATION,
        "recall_population_note": RECALL_POPULATION_NOTE,
        "comparison": comparison,
        "density_v2_status": DENSITY_V2_STATUS,
        "identity_bound_note": IDENTITY_BOUND_NOTE,
        "evidence_limits": EVIDENCE_LIMITS,
        "cells": {
            f"{clusters}:{seed}": cells[(clusters, seed)] for clusters, seed in CELLS
        },
        "r0222_gate_artifact": gate_signature,
        "r0225_tolerance_gate_artifact": tolerance_signature,
        "r0223_comparison_artifact": cuvs_signature,
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
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "gate_registered": False,
        "gate_registerable_here": GATE_REGISTERABLE_HERE,
        "gate_release_claimed": GATE_RELEASE_CLAIMED,
        "adoption_claimed": ADOPTION_CLAIMED,
        "equivalence_claimed": EQUIVALENCE_CLAIMED,
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
        os.path.join(output, "cluster-spill-graph-map-comparison.json"),
        receipt,
        immutable=True,
    )
    del source, corpus_of_row, reference, centroids
    gc.collect()


# --------------------------------------------------------------------------- #
# node 5 — the geometry the panel cannot see
# --------------------------------------------------------------------------- #
def _load_coordinates(signature: Mapping[str, Any], *, label: str) -> np.ndarray:
    path = prompt_contract.verify_signature(dict(signature), label=label)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (ROWS, 2):
        raise Round0228Error(f"{label} is {array.shape}, expected ({ROWS}, 2)")
    return np.asarray(array, dtype=np.float32)


def run_geometry(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    comparison_artifact, comparison_signature = _sealed_json(
        job, "comparison_signature", label="R0228 sealed map comparison"
    )
    if (
        comparison_artifact.get("schema") != COMPARISON_SCHEMA
        or comparison_artifact.get("round_id") != ROUND_ID
    ):
        raise Round0228Error("R0228 sealed comparison contract changed")
    panel_evidence, panel_signature = round0222_nodes._sealed_panel(job)
    gate_artifact, gate_signature = _sealed_json(
        job, "r0222_gate_signature", label="R0222 sealed n=8 gate registration"
    )
    cuvs_artifact, cuvs_signature = _sealed_json(
        job, "r0223_comparison_signature", label="R0223 sealed cuVS map comparison"
    )
    truth = _sealed_truth(job)

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0228 map geometry"
    )
    started = time.monotonic()

    truth_ids = np.load(truth["ids_path"], mmap_mode="r", allow_pickle=False)
    truth_cos = np.load(truth["cosines_path"], mmap_mode="r", allow_pickle=False)
    if truth_ids.shape != (ROWS, GRAPH_K) or truth_cos.shape != (ROWS, GRAPH_K):
        raise Round0228Error("R0220 truth arrays have the wrong shape")
    kth_cosine = np.asarray(truth_cos[:, GRAPH_K - 1], dtype=np.float64)
    truth_ids = np.asarray(truth_ids, dtype=np.int32)

    # ---- every map this round can see, in three arms.
    coordinate_sources: dict[str, dict[str, Any]] = {}
    for clusters, seed in CELLS:
        key = f"cluster-spill-c{clusters}-seed{seed}"
        coordinate_sources[key] = {
            "arm": f"cluster-spill-c{clusters}",
            "clusters": clusters,
            "seed": seed,
            "signature": dict(
                comparison_artifact["cells"][f"{clusters}:{seed}"]["coordinates"]
            ),
        }
    for seed in EXACT_FAMILY_SEEDS:
        if str(seed) in panel_evidence["cells"]:
            signature = dict(panel_evidence["cells"][str(seed)]["coordinates"])
        else:
            signature = dict(gate_artifact["new_cells"][str(seed)]["coordinates"])
        coordinate_sources[f"exact-seed{seed}"] = {
            "arm": "exact",
            "clusters": None,
            "seed": seed,
            "signature": signature,
        }
    for seed in R0223_CUVS_SEEDS:
        coordinate_sources[f"r0223-cuvs-seed{seed}"] = {
            "arm": "r0223-cuvs",
            "clusters": None,
            "seed": seed,
            "signature": dict(cuvs_artifact["cells"][str(seed)]["coordinates"]),
        }

    # ---- the row populations, one per candidate configuration, read out of the
    # sealed graph receipts rather than declared in the queue: the zero-degree
    # count and the lost-edge array only exist after the graph node has run.
    populations: dict[str, Any] = {}
    zero_degree: dict[str, int] = {}
    graph_manifest_signatures: dict[str, Any] = {}
    for clusters in CLUSTER_COUNTS:
        manifest_path, manifest_signature = _intra_queue_signature(
            job["graph_manifests"][str(clusters)],
            label=f"R0228 sealed c={clusters} graph receipt",
        )
        manifest = prompt_contract.read_sealed(
            manifest_path, label=f"R0228 sealed c={clusters} graph receipt"
        )
        if (
            manifest.get("schema") != BUILD_SCHEMA
            or manifest.get("round_id") != ROUND_ID
            or int(manifest.get("clusters", -1)) != clusters
        ):
            raise Round0228Error(f"R0228 sealed c={clusters} graph contract changed")
        graph_manifest_signatures[str(clusters)] = manifest_signature
        zero_degree[str(clusters)] = int(
            manifest["graph_checks"]["zero_degree_rows"]
        )
        lost_path = prompt_contract.verify_signature(
            dict(manifest["loss_arrays"]["lost_edges_per_row"]),
            label=f"R0228 c={clusters} lost-edge counts",
        )
        lost = np.load(lost_path, allow_pickle=False)
        if lost.shape != (ROWS,):
            raise Round0228Error(f"R0228 c={clusters} lost-edge array has wrong shape")
        mask = np.asarray(lost > 0, dtype=bool)
        selection = density_matched_control(
            lost_mask=mask,
            kth_cosine=kth_cosine,
            sample_rows=SCATTER_SAMPLE_ROWS,
            deciles=10,
            seed=SCATTER_SAMPLE_SEED,
        )
        populations[str(clusters)] = selection

    # ---- clump profiles and scatter, one pass per map.
    clumps: dict[str, Any] = {}
    scales: dict[str, float] = {}
    lost_scatter: dict[int, dict[str, list[float]]] = {
        clusters: {} for clusters in CLUSTER_COUNTS
    }
    control_scatter: dict[int, dict[str, list[float]]] = {
        clusters: {} for clusters in CLUSTER_COUNTS
    }
    for name, entry in coordinate_sources.items():
        coordinates = _load_coordinates(entry["signature"], label=f"R0228 {name} map")
        scale = map_scale(coordinates)
        scales[name] = scale
        clumps[name] = {
            "arm": entry["arm"],
            "clusters": entry["clusters"],
            "seed": entry["seed"],
            "map_rms_radius": scale,
            **clump_profile(coordinates),
        }
        for clusters in CLUSTER_COUNTS:
            selection = populations[str(clusters)]
            lost_scatter[clusters][name] = [
                float(value)
                for value in true_neighbour_scatter(
                    coordinates, truth_ids, selection["lost_sample"], scale=scale
                )
            ]
            control_scatter[clusters][name] = [
                float(value)
                for value in true_neighbour_scatter(
                    coordinates, truth_ids, selection["control_sample"], scale=scale
                )
            ]
        del coordinates
        gc.collect()

    candidate_names_by_c = {
        clusters: [f"cluster-spill-c{clusters}-seed{seed}" for seed in SEEDS]
        for clusters in CLUSTER_COUNTS
    }
    exact_names = [f"exact-seed{seed}" for seed in EXACT_FAMILY_SEEDS]
    cuvs_names = [f"r0223-cuvs-seed{seed}" for seed in R0223_CUVS_SEEDS]

    displacement: dict[str, Any] = {}
    for clusters in CLUSTER_COUNTS:
        displacement[str(clusters)] = {
            "clusters": clusters,
            "rows_carrying_loss": int(populations[str(clusters)]["lost_rows_total"]),
            "rows_carrying_loss_fraction": float(
                populations[str(clusters)]["lost_rows_total"] / ROWS
            ),
            "sampled_lost_rows": int(len(populations[str(clusters)]["lost_sample"])),
            "sampled_control_rows": int(
                len(populations[str(clusters)]["control_sample"])
            ),
            "density_match": {
                "deciles": populations[str(clusters)]["deciles"],
                "decile_counts_lost": populations[str(clusters)]["decile_counts_lost"],
                "decile_counts_control": populations[str(clusters)][
                    "decile_counts_control"
                ],
                "control_shortfall_by_decile": populations[str(clusters)][
                    "control_shortfall_by_decile"
                ],
                "matched_exactly": populations[str(clusters)]["matched_exactly"],
                "seed": populations[str(clusters)]["seed"],
            },
            "vs_exact_family": displacement_summary(
                lost_scatter=lost_scatter[clusters],
                control_scatter=control_scatter[clusters],
                candidate_maps=candidate_names_by_c[clusters],
                exact_maps=exact_names,
            ),
            "r0223_cuvs_on_the_same_rows": displacement_summary(
                lost_scatter=lost_scatter[clusters],
                control_scatter=control_scatter[clusters],
                candidate_maps=cuvs_names,
                exact_maps=exact_names,
            ),
        }

    def _arm_stats(names: list[str], field: str) -> dict[str, float]:
        values = [float(clumps[name][field]) for name in names]
        return {
            "n": len(values),
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": values,
        }

    clump_fields = (
        "clump_components",
        "largest_component_bins",
        "largest_component_share_of_clumped_rows",
        "clumped_row_fraction",
        "max_bin_count",
        "occupied_bins",
    )
    clump_comparison: dict[str, Any] = {}
    for field in clump_fields:
        exact_stats = _arm_stats(exact_names, field)
        clump_comparison[field] = {
            "exact_family": exact_stats,
            "r0223_cuvs": _arm_stats(cuvs_names, field),
            "cluster_spill": {
                str(clusters): {
                    **_arm_stats(candidate_names_by_c[clusters], field),
                    "z_of_mean_vs_exact_family": (
                        (
                            float(
                                np.mean(
                                    [
                                        clumps[name][field]
                                        for name in candidate_names_by_c[clusters]
                                    ]
                                )
                            )
                            - exact_stats["mean"]
                        )
                        / exact_stats["sd"]
                        if exact_stats["sd"] > 0
                        else None
                    ),
                    "cells_inside_exact_family_range": int(
                        sum(
                            1
                            for name in candidate_names_by_c[clusters]
                            if exact_stats["min"]
                            <= float(clumps[name][field])
                            <= exact_stats["max"]
                        )
                    ),
                }
                for clusters in CLUSTER_COUNTS
            },
        }

    execution_checks = {
        "every_map_profiled": set(clumps) == set(coordinate_sources),
        "twenty_maps": len(clumps) == len(CELLS) + len(EXACT_FAMILY_SEEDS) + len(
            R0223_CUVS_SEEDS
        ),
        "every_configuration_has_a_lost_population": all(
            int(populations[str(clusters)]["lost_rows_total"]) > 0
            for clusters in CLUSTER_COUNTS
        ),
        "control_density_matched": all(
            bool(populations[str(clusters)]["matched_exactly"])
            for clusters in CLUSTER_COUNTS
        ),
        "scatter_is_dimensionless": all(value > 0 for value in scales.values()),
        "null_arm_is_the_exact_family": all(
            displacement[str(clusters)]["vs_exact_family"]["exact_maps"] == exact_names
            for clusters in CLUSTER_COUNTS
        ),
        "zero_degree_published_for_every_configuration": set(zero_degree)
        == {str(clusters) for clusters in CLUSTER_COUNTS},
        "no_zero_degree_rows": all(value == 0 for value in zero_degree.values()),
    }
    if not all(execution_checks.values()):
        raise Round0228Error(f"R0228 geometry checks failed: {execution_checks}")

    peak_rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    receipt = prompt_contract.seal({
        "schema": GEOMETRY_SCHEMA,
        "round_id": ROUND_ID,
        "capability": GEOMETRY_CAPABILITY,
        "capabilities": [GEOMETRY_CAPABILITY],
        "release_sha": active["manifest"]["release_sha"],
        "outcome": "cluster-spill-map-geometry-against-the-exact-graph-null-arm",
        "rows": ROWS,
        "maps_profiled": len(clumps),
        "arms": {
            "cluster_spill": {
                str(clusters): candidate_names_by_c[clusters]
                for clusters in CLUSTER_COUNTS
            },
            "exact": exact_names,
            "r0223_cuvs": cuvs_names,
        },
        "clump_definition": CLUMP_DEFINITION,
        "clump_profiles": clumps,
        "clump_comparison": clump_comparison,
        "scatter_definition": SCATTER_DEFINITION,
        "scale_definition": SCALE_DEFINITION,
        "null_arm_note": NULL_ARM_NOTE,
        "scatter_sample_rows": SCATTER_SAMPLE_ROWS,
        "scatter_sample_seed": SCATTER_SAMPLE_SEED,
        "displacement": displacement,
        "zero_degree_rows_by_clusters": zero_degree,
        "graph_manifests": graph_manifest_signatures,
        "map_rms_radius": scales,
        "truth_receipt": truth["signature"],
        "comparison_artifact": comparison_signature,
        "panel_evidence": panel_signature,
        "r0222_gate_artifact": gate_signature,
        "r0223_comparison_artifact": cuvs_signature,
        "execution_checks": execution_checks,
        "evaluation_performed": True,
        "training_performed": False,
        "gate_registered": False,
        "adoption_claimed": ADOPTION_CLAIMED,
        "equivalence_claimed": EQUIVALENCE_CLAIMED,
        "production_or_publishing": False,
        "performance": {
            "node_wall_s": time.monotonic() - started,
            "peak_host_rss_gib": peak_rss_gib,
        },
    })
    atomic_write_new_json(
        os.path.join(output, "cluster-spill-map-geometry.json"), receipt, immutable=True
    )


def run_job(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0228Error("R0228 handler received another queue")
    action = str(job.get("action") or "")
    if action == CLUSTER_BUILD_ACTION:
        run_cluster_build(active, job)
    elif action == FUZZY_ACTION:
        run_fuzzy_graph(active, job)
    elif action == TRAIN_ACTION:
        run_train(active, job)
    elif action == COMPARE_ACTION:
        run_compare(active, job)
    elif action == GEOMETRY_ACTION:
        run_geometry(active, job)
    else:
        raise Round0228Error(f"unknown R0228 action {action!r}")


__all__ = [
    "CLUSTER_BUILD_ACTION",
    "COMPARE_ACTION",
    "FUZZY_ACTION",
    "GEOMETRY_ACTION",
    "TRAIN_ACTION",
    "run_cluster_build",
    "run_compare",
    "run_fuzzy_graph",
    "run_geometry",
    "run_job",
    "run_train",
]
