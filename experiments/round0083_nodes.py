"""Graph-recall dose, matched training, and evaluation for Round 0083."""
from __future__ import annotations

import json
import math
import os
import resource
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0034_pipeline import GRAPH_SCHEMA
from basemap.round0049_program import (
    DIMENSION,
    K,
    SOURCE_ROWS as INDEX_ROWS,
    _seal,
)
from basemap.round0053_program import (
    EXPECTED_EXCLUDED_ROWS,
    EXPECTED_RETAINED_ROWS,
    ROW_COUNT,
    compact30_to_global150,
    global150_to_compact30,
    validate_control_substrate,
)
from basemap.round0055_program import SUCCESSFUL_UPDATES
from basemap.round0064_evaluation import (
    MODEL_SPECS,
    seal,
    validate_seal,
)
from basemap.round0083_program import (
    CONFIG_SCHEMA,
    NPROBES,
    PANEL_SCHEMA,
    ROUND_ID,
    TARGET_RECALL_BANDS,
    TRAIN_RECEIPT_SCHEMAS,
    Round0083ProgramError,
    train_config_from_graph,
)
from experiments import round0055_nodes as trainer
from experiments import round0064_nodes as evaluator
from experiments.round0049_nodes import (
    INDEX_SEARCH_WIDTH,
    SEARCH_BATCH_ROWS,
    SEARCH_WIDTH,
    SHARD_ROWS,
    _assemble_graph,
    _exact_representative_truth,
    _initialize_graph_output,
    _membership,
    _sample_retained_rows,
    _warm_page_cache,
    _write_shard,
)
from experiments.round0053_nodes import (
    QUALITY_SAMPLE_ROWS,
    QUALITY_SEED,
)
from experiments.round0054_nodes import _validate_quality
from experiments.round0059_nodes import (
    _GpuSearchAdapter,
    _load_sealed_json,
    _project_full_graph_hours,
    _runtime_stamp,
)
from experiments.round0060_nodes import (
    BENCHMARK_ROWS,
    GRAPH_RECEIPT_SCHEMA as R0060_GRAPH_RECEIPT_SCHEMA,
    QUALIFICATION_SCHEMA as R0060_QUALIFICATION_SCHEMA,
    _queries,
    _search_and_rerank,
)


QUALIFICATION_SCHEMA = "round0083-graph-recall-qualification-v1"
GRAPH_RECEIPT_SCHEMA = "round0083-graph-recall-receipt-v1"
MAP_KEYS = {
    16: "r0083-nprobe16-on-30m",
    32: "r0083-nprobe32-on-30m",
}
MODEL_LABELS = {
    16: "r0083-nprobe16",
    32: "r0083-nprobe32",
}
MAP_LABELS = {
    MAP_KEYS[nprobe]: (
        f"r0083-balanced-30m-seed42-nprobe{nprobe}"
    )
    for nprobe in NPROBES
}
BASELINE_RECALL = 0.9224609375000001
NONINFERIORITY_MARGINS = {
    "ffr": 0.02,
    "purity_k256": 0.05,
    "purity_k1024": 0.05,
    "projection_ffr": 0.02,
}


class Round0083Error(Round0083ProgramError):
    """The registered graph-recall sensitivity contract was violated."""


def _read_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0083Error(f"{path} is not a JSON object")
    return value


def _gpu_index(faiss: Any, index: Any, nprobe: int) -> tuple[Any, Any, float]:
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    started = time.monotonic()
    raw = faiss.index_cpu_to_gpu(resources, 0, index, options)
    clone_seconds = time.monotonic() - started
    raw.nprobe = int(nprobe)
    return resources, _GpuSearchAdapter(raw, int(nprobe)), clone_seconds


def run_qualification(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure both fixed nprobe treatments on one exact truth/sample."""
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0083 fixed graph-recall qualification",
    )
    started = time.monotonic()
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    substrate = validate_control_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    quality, quality_signature = _validate_quality(
        str(job["quality_receipt"]),
        expected_sha256=str(job["quality_receipt_sha256"]),
        nprobe=64,
    )
    baseline, baseline_signature = _load_sealed_json(
        str(job["baseline_qualification"]),
        expected_sha256=str(job["baseline_qualification_sha256"]),
        schema=R0060_QUALIFICATION_SCHEMA,
    )
    outputs = substrate["manifest"]["outputs"]
    eligibility = load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    excluded = np.asarray(eligibility["excluded_rows"], dtype=np.int64)
    if (
        len(excluded) != EXPECTED_EXCLUDED_ROWS
        or ROW_COUNT - len(excluded) != EXPECTED_RETAINED_ROWS
        or baseline.get("validity_passed") is not True
        or float(
            baseline["quality"]["mean_recall_at_15_unambiguous"]
        ) != BASELINE_RECALL
    ):
        raise Round0083Error("reviewed R0060 qualification changed")
    filtered_signature = expected_input_signature(
        str(job["filtered_index"])
    )
    if filtered_signature != baseline.get("filtered_index"):
        raise Round0083Error("reviewed filtered 30M index changed")

    encoded = np.memmap(
        outputs["int8"]["canonical_path"],
        dtype=np.int8,
        mode="r",
        shape=(ROW_COUNT, DIMENSION),
    )
    scales = np.memmap(
        outputs["scales"]["canonical_path"],
        dtype="<f2",
        mode="r",
        shape=(ROW_COUNT,),
    )
    sample = _sample_retained_rows(
        excluded,
        count=QUALITY_SAMPLE_ROWS,
        seed=QUALITY_SEED,
        row_count=ROW_COUNT,
    )
    exact, ties, exact_performance = _exact_representative_truth(
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        sample=sample,
        row_count=ROW_COUNT,
    )
    if (
        sha256_bytes(sample.tobytes()) != quality["sample"]["row_sha256"]
        or int(ties.sum()) != 0
    ):
        raise Round0083Error("R0053 exact-truth sample changed")
    sample_queries = _queries(encoded, scales, sample)
    first_rows = np.arange(100_000, dtype=np.int64)
    first_retained = first_rows[~_membership(excluded, first_rows)]
    benchmark_rows = first_retained[:BENCHMARK_ROWS]
    benchmark_queries = _queries(encoded, scales, benchmark_rows)
    filtered = faiss.read_index(filtered_signature["canonical_path"])
    if int(filtered.ntotal) != EXPECTED_RETAINED_ROWS:
        raise Round0083Error("filtered index retained count changed")

    rows: dict[str, Any] = {}
    resources: list[Any] = []
    adapters: list[Any] = []
    for nprobe in NPROBES:
        resource_handle, gpu, clone_seconds = _gpu_index(
            faiss, filtered, nprobe
        )
        resources.append(resource_handle)
        adapters.append(gpu)
        selected, performance = _search_and_rerank(
            index=gpu,
            queries=sample_queries,
            compact_sources=sample,
            encoded=encoded,
            scales=scales,
        )
        overlap = (
            selected[:, :, None] == exact[:, None, :]
        ).any(axis=2).sum(axis=1) / K
        unambiguous = ~ties
        mean_recall = float(overlap[unambiguous].mean())
        p10_recall = float(np.percentile(overlap[unambiguous], 10))
        _selected, benchmark = _search_and_rerank(
            index=gpu,
            queries=benchmark_queries,
            compact_sources=benchmark_rows,
            encoded=encoded,
            scales=scales,
        )
        projection = _project_full_graph_hours(
            row_count=ROW_COUNT,
            benchmark_rows=len(benchmark_rows),
            gpu_search_seconds=float(benchmark["search_seconds"]),
            gpu_rerank_seconds=float(
                benchmark["exact_rerank"]["wall_seconds"]
            ),
            clone_seconds=clone_seconds,
        )
        low, high = TARGET_RECALL_BANDS[nprobe]
        rows[str(nprobe)] = {
            "nprobe": nprobe,
            "mean_recall_at_15_unambiguous": mean_recall,
            "p10_recall_at_15_unambiguous": p10_recall,
            "target_band": [low, high],
            "inside_planning_band": low <= mean_recall <= high,
            "planning_band_is_not_an_admission_gate": True,
            "sample_search": performance,
            "benchmark": {
                **benchmark,
                "rows": len(benchmark_rows),
                "clone_seconds": clone_seconds,
                "projected_full_graph_hours": projection["total_hours"],
            },
        }
    observed = [rows[str(value)]["mean_recall_at_15_unambiguous"]
                for value in NPROBES]
    checks = {
        "runtime_matches": all(
            value is True for value in runtime["checks"].values()
        ),
        "sample_matches_r0053": True,
        "zero_boundary_ties": True,
        "fixed_grid_complete": set(rows) == {str(value) for value in NPROBES},
        "recall_monotone_with_nprobe": observed == sorted(observed),
        "both_treatments_below_baseline": max(observed) < BASELINE_RECALL,
        "no_training_performed": True,
    }
    if not all(value is True for value in checks.values()):
        raise Round0083Error(
            "R0083 qualification failed: "
            + ", ".join(key for key, value in checks.items() if not value)
        )
    body = {
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": True,
        "training_performed": False,
        "runtime": runtime,
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "filtered_index": filtered_signature,
        "r0053_quality": quality_signature,
        "r0060_baseline_qualification": baseline_signature,
        "sample": {
            "rows": len(sample),
            "seed": QUALITY_SEED,
            "row_sha256": sha256_bytes(sample.tobytes()),
            "boundary_ties": int(ties.sum()),
            "exact_truth": exact_performance,
        },
        "baseline": {
            "nprobe": 64,
            "mean_recall_at_15_unambiguous": BASELINE_RECALL,
        },
        "fixed_nprobe_grid": list(NPROBES),
        "rows_by_nprobe": rows,
        "checks": checks,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_gib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2)
        ),
    }
    receipt = _seal(body)
    path = os.path.join(output, "graph-recall-qualification.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del adapters, resources, filtered
    return {**receipt, "receipt": expected_input_signature(path)}


def _graph_overlap(
    *,
    treatment_path: str,
    baseline_path: str,
    excluded: np.ndarray,
) -> dict[str, Any]:
    treatment = np.memmap(
        treatment_path,
        dtype="<i4",
        mode="r",
        shape=(ROW_COUNT, K),
    )
    baseline = np.memmap(
        baseline_path,
        dtype="<i4",
        mode="r",
        shape=(ROW_COUNT, K),
    )
    histogram = np.zeros(K + 1, dtype=np.int64)
    retained = 0
    for start in range(0, ROW_COUNT, 100_000):
        stop = min(start + 100_000, ROW_COUNT)
        rows = np.arange(start, stop, dtype=np.int64)
        keep = ~_membership(excluded, rows)
        left = np.asarray(treatment[start:stop])[keep]
        right = np.asarray(baseline[start:stop])[keep]
        overlap = (
            left[:, :, None] == right[:, None, :]
        ).any(axis=2).sum(axis=1)
        histogram += np.bincount(overlap, minlength=K + 1)
        retained += len(left)
    if retained != EXPECTED_RETAINED_ROWS or int(histogram.sum()) != retained:
        raise Round0083Error("graph overlap source accounting changed")
    cumulative = np.cumsum(histogram)
    p10_count = int(np.searchsorted(
        cumulative,
        math.ceil(0.10 * retained),
        side="left",
    ))
    return {
        "sources": retained,
        "mean_neighbor_overlap_fraction": (
            sum(index * int(count) for index, count in enumerate(histogram))
            / (retained * K)
        ),
        "p10_neighbor_overlap_fraction": p10_count / K,
        "identical_neighbor_set_fraction": int(histogram[K]) / retained,
        "overlap_count_histogram": {
            str(index): int(count)
            for index, count in enumerate(histogram)
            if count
        },
    }


def run_build_graph(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one exact-reranked fixed-degree graph at a registered nprobe."""
    import faiss

    nprobe = int(job["nprobe"])
    if nprobe not in NPROBES:
        raise Round0083Error("unregistered R0083 nprobe")
    started = time.monotonic()
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    substrate = validate_control_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    qualification, qualification_signature = _load_sealed_json(
        str(job["qualification_receipt"]),
        expected_sha256=expected_input_signature(
            str(job["qualification_receipt"])
        )["sha256"],
        schema=QUALIFICATION_SCHEMA,
    )
    row = (qualification.get("rows_by_nprobe") or {}).get(str(nprobe))
    if (
        qualification.get("validity_passed") is not True
        or qualification.get("release_sha")
        != active["manifest"]["release_sha"]
        or qualification.get("substrate") != substrate["signature"]
        or not isinstance(row, dict)
    ):
        raise Round0083Error("R0083 qualification binding changed")
    filtered_signature = expected_input_signature(str(job["filtered_index"]))
    if filtered_signature != qualification.get("filtered_index"):
        raise Round0083Error("qualified filtered index changed")
    outputs = substrate["manifest"]["outputs"]
    eligibility = load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    excluded = np.asarray(eligibility["excluded_rows"], dtype=np.int64)
    if len(excluded) != EXPECTED_EXCLUDED_ROWS:
        raise Round0083Error("R0083 graph eligibility changed")
    output = str(job["outputs"][0])
    contract = {
        "schema": "round0083-graph-recall-build-contract-v1",
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "qualification": qualification_signature,
        "filtered_index": filtered_signature,
        "runtime_spec": expected_input_signature(str(job["runtime_spec"])),
        "nprobe": nprobe,
        "search_width": SEARCH_WIDTH,
        "index_search_width": INDEX_SEARCH_WIDTH,
        "exact_rerank": True,
        "k": K,
        "shard_rows": SHARD_ROWS,
        "search_batch_rows": SEARCH_BATCH_ROWS,
        "candidate_universe": (
            "first 10M rows per corpus AND NOT within-subset zero/copies"
        ),
    }
    shard_root = _initialize_graph_output(output, contract=contract)
    page_cache_warm = {
        "int8": _warm_page_cache(outputs["int8"]["canonical_path"]),
        "scales": _warm_page_cache(outputs["scales"]["canonical_path"]),
    }
    encoded = np.memmap(
        outputs["int8"]["canonical_path"],
        dtype=np.int8,
        mode="r",
        shape=(ROW_COUNT, DIMENSION),
    )
    scales = np.memmap(
        outputs["scales"]["canonical_path"],
        dtype="<f2",
        mode="r",
        shape=(ROW_COUNT,),
    )
    filtered = faiss.read_index(filtered_signature["canonical_path"])
    if int(filtered.ntotal) != EXPECTED_RETAINED_ROWS:
        raise Round0083Error("filtered index count changed")
    resources, gpu, clone_seconds = _gpu_index(faiss, filtered, nprobe)
    parameters = faiss.SearchParametersIVF()
    parameters.nprobe = nprobe
    resumed = 0
    shard_receipts: list[dict[str, Any]] = []
    for shard, start in enumerate(range(0, ROW_COUNT, SHARD_ROWS)):
        stop = min(start + SHARD_ROWS, ROW_COUNT)
        receipt = _write_shard(
            index=gpu,
            parameters=parameters,
            encoded=encoded,
            scales=scales,
            excluded=excluded,
            shard_root=shard_root,
            shard=shard,
            start=start,
            stop=stop,
            nprobe=nprobe,
            round_id=ROUND_ID,
            compact_to_global_fn=compact30_to_global150,
            global_to_compact_fn=global150_to_compact30,
            source_rows=INDEX_ROWS,
        )
        resumed += int(receipt["resumed"])
        shard_receipts.append(receipt)
        print(
            f"R0083 nprobe{nprobe} graph shard {shard + 1}/"
            f"{math.ceil(ROW_COUNT / SHARD_ROWS)} "
            f"rows[{start}:{stop}] {receipt['wall_seconds']:.2f}s"
            + (" resumed" if receipt["resumed"] else ""),
            flush=True,
        )
    target_signature, degree_signature = _assemble_graph(
        output=output,
        shard_root=shard_root,
        excluded=excluded,
        nprobe=nprobe,
        round_id=ROUND_ID,
        row_count=ROW_COUNT,
    )
    overlap = _graph_overlap(
        treatment_path=target_signature["canonical_path"],
        baseline_path=str(job["baseline_targets"]),
        excluded=excluded,
    )
    search_seconds = sum(
        float(value["search_seconds"]) for value in shard_receipts
    )
    rerank_seconds = sum(
        float(value["rerank_seconds"]) for value in shard_receipts
    )
    self_seen = sum(
        int(value["self_returned"]) for value in shard_receipts
    )
    graph_body = {
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "row_count": ROW_COUNT,
        "input_k": K,
        "source_semantics": (
            "compact balanced-30M int8 row; excluded rows degree zero"
        ),
        "destination_policy": (
            "native representative-only GPU IVF-PQ search in matched "
            "30M candidate universe; remove self"
        ),
        "sampling_semantics": (
            "uniform retained source, then uniform one of 15 destinations; "
            "source-uniform and edge-uniform are identical at fixed degree"
        ),
        "negative_policy": "uniform-eligibility-retained-rows-nonself",
        "weight_semantics": "unweighted directed k15",
        "inputs": {
            "eligibility": outputs["eligibility"],
            "substrate": substrate["signature"],
            "qualification": qualification_signature,
            "filtered_index": filtered_signature,
            "baseline_targets": expected_input_signature(
                str(job["baseline_targets"])
            ),
        },
        "outputs": {
            "targets": target_signature,
            "degrees": degree_signature,
        },
        "target_dtype": "<i4",
        "target_shape": [ROW_COUNT, K],
        "degree_dtype": "|u1",
        "degree_shape": [ROW_COUNT],
        "summary": {
            "eligibility_excluded_source_count": EXPECTED_EXCLUDED_ROWS,
            "eligibility_retained_row_count": EXPECTED_RETAINED_ROWS,
            "retained_positive_source_count": EXPECTED_RETAINED_ROWS,
            "zero_degree_retained_source_count": 0,
            "zero_degree_retained_source_fraction": 0.0,
            "valid_canonical_edge_count": EXPECTED_RETAINED_ROWS * K,
            "input_edge_count": EXPECTED_RETAINED_ROWS * K,
            "degree_histogram": {
                "0": EXPECTED_EXCLUDED_ROWS,
                str(K): EXPECTED_RETAINED_ROWS,
            },
            "self_returned_count": self_seen,
            "self_returned_fraction": self_seen / EXPECTED_RETAINED_ROWS,
        },
        "candidate_generator": {
            "index_type": "GpuIndexIVFPQ",
            "source_index_rows": INDEX_ROWS,
            "physically_filtered_index_rows": EXPECTED_RETAINED_ROWS,
            "nlist": 8192,
            "pq_m": 48,
            "pq_nbits": 8,
            "nprobe": nprobe,
            "search_width": SEARCH_WIDTH,
            "index_search_width": INDEX_SEARCH_WIDTH,
            "selected_neighbors": K,
            "exact_rerank": True,
            "rerank_vector_source": (
                "balanced-subset int8-plus-fp16-scale exact cosine"
            ),
            "candidate_universe": "first 10M rows per corpus, retained",
        },
        "quality": {
            "mean_recall_at_15_unambiguous": float(
                row["mean_recall_at_15_unambiguous"]
            ),
            "p10_recall_at_15_unambiguous": float(
                row["p10_recall_at_15_unambiguous"]
            ),
            "planning_target_band": list(TARGET_RECALL_BANDS[nprobe]),
            "planning_band_is_not_an_admission_gate": True,
            "baseline_nprobe64_recall": BASELINE_RECALL,
            "full_graph_neighbor_set_overlap_vs_r0060": overlap,
        },
        "runtime": runtime,
        "timing": {
            "page_cache_warm": page_cache_warm,
            "gpu_clone_seconds": clone_seconds,
            "search_seconds": search_seconds,
            "rerank_seconds": rerank_seconds,
            "total_seconds": time.monotonic() - started,
            "shard_count": len(shard_receipts),
            "resumed_shards": resumed,
        },
    }
    graph = _seal(graph_body)
    graph_path = os.path.join(output, "canonical-graph-v1.json")
    atomic_write_new_json(graph_path, graph, immutable=True)
    receipt_body = {
        "schema": GRAPH_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "nprobe": nprobe,
        "training_performed": False,
        "optimizer_updates": 0,
        "graph": expected_input_signature(graph_path),
        "substrate": substrate["signature"],
        "qualification": qualification_signature,
        "search": graph["candidate_generator"],
        "quality": graph["quality"],
        "summary": graph["summary"],
        "performance": {
            **graph["timing"],
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    receipt = _seal(receipt_body)
    receipt_path = os.path.join(output, "receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del resources, gpu, filtered
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _configure_evaluator() -> None:
    evaluator.ROUND_ID = ROUND_ID
    evaluator.MAP_LABELS = MAP_LABELS
    for nprobe in NPROBES:
        MODEL_SPECS[MODEL_LABELS[nprobe]] = {
            "round_id": ROUND_ID,
            "receipt_schema": TRAIN_RECEIPT_SCHEMAS[nprobe],
            "config_schema": CONFIG_SCHEMA,
            "rows": ROW_COUNT,
            "retained_rows": EXPECTED_RETAINED_ROWS,
            "updates": SUCCESSFUL_UPDATES,
            "sampler_class": "HostInt8Balanced30mCanonicalSampler",
        }


def run_train(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    """Reuse the exact R0061 trainer with the registered graph config builder."""
    nprobe = int(job["nprobe"])
    graph_path = str(job["canonical_graph_manifest"])
    graph_signature = expected_input_signature(graph_path)
    graph = _read_json(graph_path)
    graph_body = {
        key: value for key, value in graph.items()
        if key != "identity_sha256"
    }
    qualification_path = (
        graph.get("inputs", {}).get("qualification", {}).get(
            "canonical_path"
        )
    )
    qualification = (
        _read_json(str(qualification_path))
        if qualification_path else {}
    )
    if (
        nprobe not in NPROBES
        or graph.get("schema") != GRAPH_SCHEMA
        or graph.get("round_id") != ROUND_ID
        or graph.get("identity_sha256")
        != sha256_bytes(canonical_json(graph_body))
        or int(
            (graph.get("candidate_generator") or {}).get("nprobe", -1)
        ) != nprobe
        or qualification.get("release_sha")
        != active["manifest"]["release_sha"]
    ):
        raise Round0083Error("late-bound R0083 training graph changed")
    substrate = validate_control_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    _config, config_sha256 = train_config_from_graph(
        graph,
        graph_manifest_path=graph_signature["canonical_path"],
        graph_manifest_sha256=graph_signature["sha256"],
        substrate_manifest=substrate["manifest"],
        substrate_manifest_path=substrate["signature"]["canonical_path"],
        substrate_manifest_sha256=substrate["signature"]["sha256"],
    )
    bound_job = {
        **job,
        "canonical_graph_manifest_sha256": graph_signature["sha256"],
        "train_config_sha256": config_sha256,
    }
    previous = trainer.train_config_from_capabilities
    trainer.train_config_from_capabilities = train_config_from_graph
    try:
        return trainer.run_train(active, bound_job)
    finally:
        trainer.train_config_from_capabilities = previous


def _late_bind_model(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    nprobe = int(job["nprobe"])
    model_path = str(job["model_path"])
    receipt_path = str(job["train_receipt_path"])
    model = expected_input_signature(model_path)
    receipt_signature = expected_input_signature(receipt_path)
    receipt = _read_json(receipt_path)
    receipt_body = {
        key: value for key, value in receipt.items()
        if key != "identity_sha256"
    }
    treatment = (
        (receipt.get("production_config") or {})
        .get("execution", {})
        .get("graph_recall_treatment", {})
    )
    if (
        nprobe not in NPROBES
        or receipt.get("schema") != TRAIN_RECEIPT_SCHEMAS[nprobe]
        or receipt.get("round_id") != ROUND_ID
        or receipt.get("release_sha") != active["manifest"]["release_sha"]
        or receipt.get("model") != model
        or receipt.get("identity_sha256")
        != sha256_bytes(canonical_json(receipt_body))
        or int(treatment.get("nprobe", -1)) != nprobe
    ):
        raise Round0083Error("late-bound R0083 model/receipt changed")
    return {
        **job,
        "model_sha256": model["sha256"],
        "train_receipt_sha256": receipt_signature["sha256"],
    }


def _metrics(panel: Mapping[str, Any]) -> dict[str, float]:
    scientific = panel["panel"]
    purity = scientific["purity"]
    return {
        "ffr": float(scientific["ffr"]),
        "density_legacy_diagnostic": float(scientific["density"]),
        "purity_k256": float(purity["k256"]),
        "purity_k1024": float(purity["k1024"]),
        "projection_ffr": float(panel["projection"]["proj_ffr"]),
        "recall_at_10": float(panel["recall_at_10"]),
        "recall_at_50": float(panel["recall_at_50"]),
    }


def _noninferiority(
    treatment: Mapping[str, float],
    baseline: Mapping[str, float],
) -> dict[str, Any]:
    """Apply registered margins to raw values; rounding is reporting-only."""
    result: dict[str, Any] = {}
    for metric, margin in NONINFERIORITY_MARGINS.items():
        baseline_value = float(baseline[metric])
        treatment_value = float(treatment[metric])
        delta = treatment_value - baseline_value
        result[metric] = {
            "baseline": baseline_value,
            "treatment": treatment_value,
            "delta": delta,
            "maximum_allowed_decrease": margin,
            "passed": treatment_value >= baseline_value - margin,
        }
    return result


def _load_panel(
    path: str,
    *,
    schema: str,
    map_key: str,
) -> dict[str, Any]:
    panel = _read_json(path)
    validate_seal(panel, label=f"R0083 {map_key} panel")
    if panel.get("schema") != schema or panel.get("map_key") != map_key:
        raise Round0083Error(f"{map_key} panel identity changed")
    return panel


def classify_sensitivity(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(cells) != {str(value) for value in NPROBES}:
        raise Round0083Error("graph-recall decision cells are incomplete")
    by_recall = sorted(
        cells.values(),
        key=lambda value: float(
            value["candidate_recall_at_15_unambiguous"]
        ),
    )
    lower, upper = by_recall
    if lower["passed"] and upper["passed"]:
        verdict = "insensitive-through-lowest-tested-recall"
        lowest_passing = lower["candidate_recall_at_15_unambiguous"]
    elif not lower["passed"] and upper["passed"]:
        verdict = "sensitive-between-tested-recalls"
        lowest_passing = upper["candidate_recall_at_15_unambiguous"]
    elif not lower["passed"] and not upper["passed"]:
        verdict = "current-floor-load-bearing-within-tested-range"
        lowest_passing = None
    else:
        verdict = "nonmonotonic-map-outcome-requires-follow-up"
        lowest_passing = lower["candidate_recall_at_15_unambiguous"]
    return {
        "verdict": verdict,
        "lowest_passing_measured_recall": lowest_passing,
        "changes_frozen_floor_in_this_round": False,
        "future_floor_registration_requires_separate_round": True,
    }


def run_comparison(
    _active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        job["outputs"][0],
        label="R0083 graph-recall sensitivity comparison",
    )
    baseline = _load_panel(
        str(job["baseline_panel"]),
        schema="round0064-registered-panel-v1",
        map_key="r0061-30m-on-30m",
    )
    treatments = {
        nprobe: _load_panel(
            str(job[f"panel_nprobe{nprobe}"]),
            schema=PANEL_SCHEMA,
            map_key=MAP_KEYS[nprobe],
        )
        for nprobe in NPROBES
    }
    panels = [baseline, *treatments.values()]
    scientific = [panel["panel"] for panel in panels]
    if (
        any(
            panel.get("eligibility") != baseline.get("eligibility")
            for panel in treatments.values()
        )
        or any(
            panel.get("scientific_universe")
            != baseline.get("scientific_universe")
            for panel in treatments.values()
        )
        or len({panel.get("n") for panel in scientific}) != 1
        or len({panel.get("anchor_hash") for panel in scientific}) != 1
        or len({
            panel.get("provenance", {}).get("hiD_reference_key")
            for panel in scientific
        }) != 1
    ):
        raise Round0083Error("R0083 panels do not share one evaluation universe")
    qualification = _read_json(str(job["qualification_receipt"]))
    if qualification.get("schema") != QUALIFICATION_SCHEMA:
        raise Round0083Error("R0083 qualification schema changed")
    qualification_body = {
        key: value for key, value in qualification.items()
        if key != "identity_sha256"
    }
    if qualification.get("identity_sha256") != sha256_bytes(
        canonical_json(qualification_body)
    ):
        raise Round0083Error("R0083 qualification seal changed")
    baseline_metrics = _metrics(baseline)
    cells: dict[str, Any] = {}
    for nprobe in NPROBES:
        panel = treatments[nprobe]
        metrics = _metrics(panel)
        noninferiority = _noninferiority(metrics, baseline_metrics)
        checks = dict(panel.get("decision_checks") or {})
        non_density_checks = {
            key: value
            for key, value in checks.items()
            if key != "density_at_least_0_60"
        }
        expected_checks = {
            "ffr_at_least_0_40",
            "purity_k256_at_least_0_50",
            "purity_k1024_at_least_0_50",
            "heldout_projection_beats_untrained_floor",
            "recall_at_50_exceeds_recall_at_10",
            "coords_finite",
            "coords_not_collapsed",
            "embeddings_finite",
            "eligible_embeddings_nonzero",
        }
        if set(non_density_checks) != expected_checks:
            raise Round0083Error("R0083 full-universe check set changed")
        graph_receipt = _read_json(str(job[f"graph_receipt_nprobe{nprobe}"]))
        graph_receipt_body = {
            key: value for key, value in graph_receipt.items()
            if key != "identity_sha256"
        }
        if (
            graph_receipt.get("schema") != GRAPH_RECEIPT_SCHEMA
            or int(graph_receipt.get("nprobe", -1)) != nprobe
            or graph_receipt.get("identity_sha256")
            != sha256_bytes(canonical_json(graph_receipt_body))
        ):
            raise Round0083Error("R0083 graph receipt changed")
        passed = (
            all(item["passed"] for item in noninferiority.values())
            and all(value is True for value in non_density_checks.values())
        )
        cells[str(nprobe)] = {
            "nprobe": nprobe,
            "candidate_recall_at_15_unambiguous": float(
                qualification["rows_by_nprobe"][str(nprobe)][
                    "mean_recall_at_15_unambiguous"
                ]
            ),
            "graph_neighbor_set_overlap_vs_nprobe64": (
                graph_receipt["quality"][
                    "full_graph_neighbor_set_overlap_vs_r0060"
                ]
            ),
            "metrics": metrics,
            "noninferiority_vs_r0061": noninferiority,
            "full_30m_non_density_checks": non_density_checks,
            "legacy_density_absolute_check_reported_not_gating": checks.get(
                "density_at_least_0_60"
            ),
            "passed": passed,
        }
    decision = classify_sensitivity(cells)
    body = {
        "schema": "round0083-graph-recall-sensitivity-v1",
        "round_id": ROUND_ID,
        "baseline": {
            "nprobe": 64,
            "candidate_recall_at_15_unambiguous": BASELINE_RECALL,
            "panel": expected_input_signature(str(job["baseline_panel"])),
            "metrics": baseline_metrics,
        },
        "qualification": expected_input_signature(
            str(job["qualification_receipt"])
        ),
        "cells": cells,
        "decision": decision,
        "causal_contract": {
            "same_substrate": True,
            "same_seed": 42,
            "same_successful_updates": SUCCESSFUL_UPDATES,
            "same_sampler_law": (
                "uniform retained source then uniform one of 15 fixed-degree "
                "destinations, with replacement"
            ),
            "treatment": (
                "neighbor identities induced by nprobe before exact rerank"
            ),
            "legacy_density_is_diagnostic_only": True,
        },
        "training_performed": True,
    }
    receipt = seal(body)
    path = os.path.join(output, "graph-recall-sensitivity.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise RuntimeError("R0083 handler requires the exact round")
    if job is None:
        raise RuntimeError("R0083 handler requires the exact job")
    action = str(job.get("action"))
    if action == "qualify":
        return run_qualification(active, job)
    if action == "build_graph":
        return run_build_graph(active, job)
    if action == "train":
        return run_train(active, job)
    _configure_evaluator()
    if action == "transform":
        return evaluator.run_transform(active, _late_bind_model(active, job))
    if action == "panel":
        return evaluator.run_panel(active, _late_bind_model(active, job))
    if action == "comparison":
        return run_comparison(active, job)
    raise RuntimeError(f"unknown R0083 action {action!r}")
