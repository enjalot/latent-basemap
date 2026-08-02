"""Execute the matched 12.5M-to-25M diverse-Jina scale-policy bridge."""
from __future__ import annotations

import gc
import json
import math
import os
import resource
import time
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    ordered_array_sha256,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_save_new_npz,
    atomic_write_new_json,
    create_fresh_directory,
    ensure_data_directory,
)
from basemap.round0036_pipeline import (
    COORDINATE_SCHEMA,
    TRANSFORM_SCHEMA,
    CoordinateStream,
    seal as coordinate_seal,
)
from basemap.round0104_training import validate_substrate_manifest
from basemap.round0105_search import (
    BOUNDARY_TIE_ATOL,
    DIMENSION,
    ELIGIBILITY_PATH,
    GROUPS,
    NLIST,
    PQ_BITS,
    PQ_M,
    ROW_COUNT,
    group_ranges,
    sample_retained_rows,
    sample_stratified_rows,
)
from basemap.round0106_graph import (
    LOCAL_CONNECTIVITY,
    MINIMUM_SHARD_SOURCES_PER_SECOND,
    PAIR_BUCKETS,
    PERFORMANCE_SUBFLOOR_PATIENCE,
    SHARD_ROWS,
    update_performance_streak,
)
from basemap.round0108_evaluation import (
    FAMILY_SIZE_CUTOFF,
    FRACTION,
    HELDOUT_CORPUS_ROWS,
    HELDOUT_QUERY_ROWS,
    IN_MIX_LANGUAGES,
    K_DENSITY,
    K_HIT,
    K_LOW_MAX,
    POLISH,
    TRANSFORM_BATCH_ROWS,
    TRANSFORM_CHUNK_ROWS,
    CompactInt8DequantizedArray,
    exact_cosine_topk,
    exact_split_duplicate_diagnostics,
    load_reviewed_model,
    map_family_sizes,
    projection_metrics,
    recall_from_neighbors,
    read_sealed,
)
from basemap.round0132_scale_bridge import (
    DECISION_SCHEMA,
    DENSITY_BOOTSTRAP_DRAWS,
    GRAPH_K,
    GRAPH_PART_SCHEMA,
    GRAPH_SCHEMA,
    GRAPH_SHARD_SCHEMA,
    HALF_RETAINED_ROWS,
    INDEX_SCHEMA,
    INDEX_TRAIN_ROWS,
    INDEX_TRAIN_SEED,
    NATIVE_ANCHORS_PER_GROUP,
    NATIVE_ANCHOR_SEED,
    NATIVE_SCHEMA,
    N_NEIGHBORS,
    OOD_SCHEMA,
    PIPELINE,
    PIPELINE_SCHEMA,
    POSITIVE_DESTINATION_POLICY,
    PRODUCTION_CONFIG_SCHEMA,
    QUALIFICATION_SCHEMA,
    ROUND_ID,
    SAMPLER_CLASS,
    SEARCH_ANCHORS_PER_GROUP,
    SEARCH_NPROBE,
    SEARCH_SEED,
    SEARCH_SHORTLIST_WIDTH,
    SEED,
    SUBSET_NAMESPACE,
    SUBSET_SCHEMA,
    TRAIN_CONFIG_SCHEMA,
    TRAIN_RECEIPT_SCHEMA,
    Round0132Error,
    group_part_specs,
    largest_remainder_quotas,
    noninferiority_checks,
    paired_density_bootstrap,
    qualification_metrics,
    recall50_at_least_recall10,
    scale_policy_decision,
    seal,
    select_lowest_sha256_rank,
    validate_seal,
    validate_train_execution,
)
from experiments.round0105_nodes import (
    ADD_BATCH_ROWS,
    _exact_truth,
    _gpu_options,
    _normalized_rows,
    _require_geometry,
    _retained_batch,
    _search_and_rerank,
    _substrate_arrays,
    _write_index_new,
)
from experiments.round0106_nodes import (
    GraphNodeContract,
    _partition_forward_edges,
    _publish_memmaps,
    _shard_paths,
    _validate_joined_reciprocity,
    _write_shard,
)
from experiments.round0107_nodes import run_train_contract
from experiments.round0108_nodes import _family_arrays, _panel_config
from experiments.build_weighted_graph import phase_c_join


GRAPH_PART_NAMES = ("groups-a", "groups-b", "groups-c")
GRAPH_CONTRACT = GraphNodeContract(
    round_id=ROUND_ID,
    k=GRAPH_K,
    n_neighbors=N_NEIGHBORS,
    shard_schema=GRAPH_SHARD_SCHEMA,
    part_schema=GRAPH_PART_SCHEMA,
    graph_schema=GRAPH_SCHEMA,
)
FULL_GRAPH_SCHEMA = "round0106-jina-diverse-25m-fuzzy-graph-v1"
FULL_TRAIN_SCHEMA = "round0107-diverse-jina-train-receipt-v1"
FULL_PRODUCTION_SCHEMA = "round0107-production-config-v1"
TRANSFORM_SCHEMA_R0132 = TRANSFORM_SCHEMA
SEARCH_POSITIVE_OUTCOME = "qualified-fixed-r0105-policy-on-half-universe"
SEARCH_NEGATIVE_OUTCOME = "fixed-r0105-policy-failed-closed-on-half-universe"
SEARCH_EXACT_UNIVERSE_CHECK = "candidate_universe_is_exact_half_subset"
GRAPH_CANDIDATE_UNIVERSE = "exact R0132 half subset"
TRANSFORM_MAP_KEY = "r0132-diverse-jina-12p5m-seed42"
TRANSFORM_SCIENTIFIC_UNIVERSE = (
    "R0132 deterministic 12,474,331-row half subset"
)
TRANSFORM_ROW_ORDER = "R0132 half compact order"
NATIVE_TREATMENT_KEY = "treatment_25m_on_u12"
NATIVE_SHARED_ROWS_CHECK = "same_u12_rows"
NATIVE_GLOBAL_FFR_ROLE = "registered-noninferiority-gate"


@lru_cache(maxsize=256)
def _cached_signature(path: str) -> tuple[str, str, int, str]:
    """Memoize immutable external signatures within each node process."""
    value = expected_input_signature(path)
    return (
        str(value["kind"]),
        str(value["canonical_path"]),
        int(value["bytes"]),
        str(value["sha256"]),
    )


def _signature(
    path: str,
    expected_sha256: str | None = None,
    *,
    label: str,
) -> dict[str, Any]:
    kind, canonical_path, size, digest = _cached_signature(os.path.realpath(path))
    value = {
        "kind": kind,
        "canonical_path": canonical_path,
        "bytes": size,
        "sha256": digest,
    }
    if expected_sha256 is not None and digest != expected_sha256:
        raise Round0132Error(f"{label} bytes changed")
    return value


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0132Error(f"{path} is not a JSON object")
    return value


def _peak_rss_gib() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024 ** 2)


def _subset_paths(root: str) -> dict[str, str]:
    return {
        "manifest": os.path.join(root, "subset-manifest.json"),
        "mapping": os.path.join(root, "compact-to-global.i64.npy"),
        "group_ids": os.path.join(root, "compact-group-ids.u8.npy"),
        "excluded": os.path.join(root, "excluded-from-half.i64.npy"),
    }


def _load_subset(root: str) -> dict[str, Any]:
    paths = _subset_paths(root)
    manifest = _load_json(paths["manifest"])
    validate_seal(manifest, label="R0132 subset manifest")
    if manifest.get("schema") != SUBSET_SCHEMA or manifest.get("round_id") != ROUND_ID:
        raise Round0132Error("R0132 subset manifest contract changed")
    signatures = {
        key: _signature(paths[key], label=f"R0132 subset {key}")
        for key in ("mapping", "group_ids", "excluded")
    }
    if any(manifest.get(key) != signatures[key] for key in signatures):
        raise Round0132Error("R0132 subset output signatures changed")
    mapping = np.load(paths["mapping"], mmap_mode="r", allow_pickle=False)
    group_ids = np.load(paths["group_ids"], mmap_mode="r", allow_pickle=False)
    excluded = np.load(paths["excluded"], mmap_mode="r", allow_pickle=False)
    if (
        mapping.shape != (HALF_RETAINED_ROWS,)
        or mapping.dtype != np.int64
        or group_ids.shape != mapping.shape
        or group_ids.dtype != np.uint8
        or excluded.shape != (ROW_COUNT - HALF_RETAINED_ROWS,)
        or excluded.dtype != np.int64
        or np.any(mapping[1:] <= mapping[:-1])
        or np.any(excluded[1:] <= excluded[:-1])
        or int(mapping[0]) < 0
        or int(mapping[-1]) >= ROW_COUNT
        or int(excluded[0]) < 0
        or int(excluded[-1]) >= ROW_COUNT
        or np.intersect1d(mapping, excluded).size
    ):
        raise Round0132Error("R0132 subset arrays changed")
    quotas = {group: int(manifest["quotas"][group]) for group in GROUPS}
    if any(int(np.count_nonzero(group_ids == index)) != quotas[group]
           for index, group in enumerate(GROUPS)):
        raise Round0132Error("R0132 subset group quotas changed")
    return {
        "manifest": manifest,
        "manifest_signature": _signature(paths["manifest"], label="R0132 subset manifest"),
        "signatures": signatures,
        "mapping": mapping,
        "group_ids": group_ids,
        "excluded": excluded,
        "paths": paths,
        "parts": group_part_specs(quotas),
    }


def run_select_subset(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0132 deterministic half subset"
    )
    started = time.monotonic()
    substrate = validate_substrate_manifest(verify_payloads=False)
    eligibility = _signature(
        ELIGIBILITY_PATH,
        str(job["eligibility_sha256"]),
        label="R0087 eligibility",
    )
    with np.load(ELIGIBILITY_PATH, allow_pickle=False) as archive:
        original_excluded = np.asarray(archive["excluded_rows"], dtype=np.int64)
    ranges = group_ranges(substrate["manifest"])
    counts: dict[str, int] = {}
    for group in GROUPS:
        start, stop = ranges[group]
        left = int(np.searchsorted(original_excluded, start, side="left"))
        right = int(np.searchsorted(original_excluded, stop, side="left"))
        counts[group] = stop - start - (right - left)
    quotas = largest_remainder_quotas(counts)
    selected_groups: list[np.ndarray] = []
    selected_group_ids: list[np.ndarray] = []
    group_receipts: dict[str, Any] = {}
    for group_id, group in enumerate(GROUPS):
        start, stop = ranges[group]
        eligible = _retained_batch(original_excluded, start=start, stop=stop)
        chosen = select_lowest_sha256_rank(
            eligible,
            count=quotas[group],
            namespace=SUBSET_NAMESPACE + group.encode("utf-8") + b"\0",
        )
        selected_groups.append(chosen)
        selected_group_ids.append(np.full(len(chosen), group_id, dtype=np.uint8))
        group_receipts[group] = {
            "global_start": start,
            "global_stop": stop,
            "retained_rows": len(eligible),
            "selected_rows": len(chosen),
            "ordered_rows_sha256": ordered_array_sha256(chosen),
        }
    mapping = np.concatenate(selected_groups).astype(np.int64, copy=False)
    group_ids = np.concatenate(selected_group_ids)
    keep = np.zeros(ROW_COUNT, dtype=bool)
    keep[mapping] = True
    excluded = np.flatnonzero(~keep).astype(np.int64, copy=False)
    if (
        len(mapping) != HALF_RETAINED_ROWS
        or len(np.unique(mapping)) != HALF_RETAINED_ROWS
        or np.any(mapping[1:] <= mapping[:-1])
        or len(excluded) != ROW_COUNT - HALF_RETAINED_ROWS
    ):
        raise Round0132Error("R0132 deterministic subset did not close")
    paths = _subset_paths(output)
    atomic_save_new_npy(paths["mapping"], mapping, immutable=True)
    atomic_save_new_npy(paths["group_ids"], group_ids, immutable=True)
    atomic_save_new_npy(paths["excluded"], excluded, immutable=True)
    signatures = {
        key: expected_input_signature(paths[key])
        for key in ("mapping", "group_ids", "excluded")
    }
    manifest = seal({
        "schema": SUBSET_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "full_retained_rows": sum(counts.values()),
        "selected_rows": len(mapping),
        "selector": {
            "allocation": "integer-largest-remainder",
            "allocation_tie_break": "registered-GROUPS order",
            "within_group_rank": (
                "ascending SHA256(round0132-half-v1, group, global-row-id); "
                "global-row-id final tie break"
            ),
            "namespace_hex": SUBSET_NAMESPACE.hex(),
            "prefix_or_contiguous_selection": False,
            "map_outcomes_observed": False,
        },
        "group_counts": counts,
        "quotas": quotas,
        "groups": group_receipts,
        "eligibility": eligibility,
        "substrate": substrate["signature"],
        "mapping": signatures["mapping"],
        "group_ids": signatures["group_ids"],
        "excluded": signatures["excluded"],
        "parts": group_part_specs(quotas),
        "checks": {
            "exact_target": True,
            "every_group_present": True,
            "duplicate_control_inherited": True,
            "mapping_strictly_increasing": True,
            "mapping_and_excluded_partition_25m": True,
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "map_outcomes_observed": False,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(paths["manifest"], manifest, immutable=True)
    return {**manifest, "receipt": expected_input_signature(paths["manifest"])}


def _validate_subset_index_ids(index: Any, selected_rows: np.ndarray) -> dict[str, Any]:
    import faiss

    seen: list[np.ndarray] = []
    list_sizes = np.empty(NLIST, dtype=np.int64)
    for list_id in range(NLIST):
        size = int(index.invlists.list_size(list_id))
        list_sizes[list_id] = size
        if size:
            seen.append(np.array(
                faiss.rev_swig_ptr(index.invlists.get_ids(list_id), size),
                dtype=np.int64,
                copy=True,
            ))
    ids = np.sort(np.concatenate(seen)) if seen else np.empty(0, dtype=np.int64)
    if not np.array_equal(ids, np.asarray(selected_rows, dtype=np.int64)):
        raise Round0132Error("R0132 search index IDs differ from the half subset")
    return {
        "list_size_min": int(list_sizes.min()),
        "list_size_mean": float(list_sizes.mean()),
        "list_size_p90": float(np.percentile(list_sizes, 90)),
        "list_size_max": int(list_sizes.max()),
        "seen_selected_rows": len(ids),
        "only_selected_rows_present": True,
        "global_ids_unique": True,
    }


def run_build_index(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    import faiss
    import torch

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0132 half-universe search index"
    )
    started = time.monotonic()
    subset = _load_subset(str(job["subset_output"]))
    substrate, _original_excluded, encoded, scales = _substrate_arrays()
    sample = sample_retained_rows(
        subset["excluded"], count=INDEX_TRAIN_ROWS, seed=INDEX_TRAIN_SEED
    )
    vectors = _normalized_rows(encoded, scales, sample)
    cpu = faiss.IndexIVFPQ(
        faiss.IndexFlatIP(DIMENSION),
        DIMENSION,
        NLIST,
        PQ_M,
        PQ_BITS,
        faiss.METRIC_INNER_PRODUCT,
    )
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    gpu = faiss.index_cpu_to_gpu(resources, 0, cpu, _gpu_options())
    gpu.cp.seed = INDEX_TRAIN_SEED
    gpu.cp.niter = 25
    gpu.cp.spherical = True
    gpu.pq.cp.seed = INDEX_TRAIN_SEED
    gpu.pq.cp.niter = 25
    train_started = time.monotonic()
    gpu.train(vectors)
    train_seconds = time.monotonic() - train_started
    del vectors
    add_started = time.monotonic()
    added = 0
    for start in range(0, ROW_COUNT, ADD_BATCH_ROWS):
        stop = min(start + ADD_BATCH_ROWS, ROW_COUNT)
        rows = _retained_batch(subset["excluded"], start=start, stop=stop)
        if len(rows):
            gpu.add_with_ids(_normalized_rows(encoded, scales, rows), rows)
            added += len(rows)
    add_seconds = time.monotonic() - add_started
    if added != HALF_RETAINED_ROWS or int(gpu.ntotal) != HALF_RETAINED_ROWS:
        raise Round0132Error("R0132 half index count changed")
    assembled = faiss.index_gpu_to_cpu(gpu)
    _require_geometry(assembled, ntotal=HALF_RETAINED_ROWS)
    validation = _validate_subset_index_ids(assembled, subset["mapping"])
    index_path = os.path.join(output, "jina-diverse-12p5m.ivfpq")
    index_signature = _write_index_new(assembled, index_path)
    receipt = seal({
        "schema": INDEX_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "subset_manifest": subset["manifest_signature"],
        "mapping": subset["signatures"]["mapping"],
        "substrate": substrate["signature"],
        "index": index_signature,
        "geometry": {
            "dimension": DIMENSION,
            "ntotal": int(assembled.ntotal),
            "nlist": int(assembled.nlist),
            "pq_m": int(assembled.pq.M),
            "pq_bits": int(assembled.pq.nbits),
            "metric": "inner-product-on-normalized-native-vectors",
            "fp16_lookup_tables": True,
        },
        "id_validation": validation,
        "index_training": {
            "rows": len(sample),
            "seed": INDEX_TRAIN_SEED,
            "sample_ordered_sha256": ordered_array_sha256(sample),
            "coarse_iterations": 25,
            "pq_iterations": 25,
        },
        "runtime": {
            "one_local_cuda_device": (
                faiss.get_num_gpus() == 1 and torch.cuda.device_count() == 1
            ),
            "train_seconds": train_seconds,
            "add_seconds": add_seconds,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "training_performed": False,
        "optimizer_updates": 0,
    })
    receipt_path = os.path.join(output, "index-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del gpu, resources, assembled, cpu, encoded, scales
    gc.collect()
    torch.cuda.empty_cache()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _load_index_receipt(job: Mapping[str, Any], subset: Mapping[str, Any]) -> dict[str, Any]:
    path = os.path.join(str(job["index_output"]), "index-receipt.json")
    value = _load_json(path)
    validate_seal(value, label="R0132 index receipt")
    observed_index = _signature(str(job["index"]), label="R0132 half index")
    if (
        value.get("schema") != INDEX_SCHEMA
        or value.get("round_id") != ROUND_ID
        or value.get("subset_manifest") != subset["manifest_signature"]
        or value.get("mapping") != subset["signatures"]["mapping"]
        or value.get("index") != observed_index
    ):
        raise Round0132Error("R0132 index lineage changed")
    return value


def run_qualify_search(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    import faiss
    import torch

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0132 fixed-policy search qualification"
    )
    started = time.monotonic()
    subset = _load_subset(str(job["subset_output"]))
    index_receipt = _load_index_receipt(job, subset)
    substrate, _original_excluded, encoded, scales = _substrate_arrays()
    ranges = group_ranges(substrate["manifest"])
    sample, group_ids = sample_stratified_rows(
        subset["excluded"],
        ranges,
        rows_per_group=SEARCH_ANCHORS_PER_GROUP,
        seed=SEARCH_SEED,
    )
    exact, ties, margins, truth_timing = _exact_truth(
        encoded=encoded,
        scales=scales,
        excluded=subset["excluded"],
        sample=sample,
        k=GRAPH_K,
    )
    queries = _normalized_rows(encoded, scales, sample)
    cpu = faiss.read_index(str(job["index"]))
    _require_geometry(cpu, ntotal=HALF_RETAINED_ROWS)
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    gpu = faiss.index_cpu_to_gpu(resources, 0, cpu, _gpu_options())
    selected, execution = _search_and_rerank(
        gpu,
        nprobe=SEARCH_NPROBE,
        width=SEARCH_SHORTLIST_WIDTH,
        queries=queries,
        sample=sample,
        excluded=subset["excluded"],
        encoded=encoded,
        scales=scales,
        k=GRAPH_K,
    )
    metrics = qualification_metrics(
        selected, exact, group_ids=group_ids, unambiguous=~ties
    )
    arrays_path = os.path.join(output, "fixed-policy-truth.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        sample_rows=sample,
        group_ids=group_ids,
        exact_neighbors=exact,
        selected_neighbors=selected,
        boundary_ties=ties,
        boundary_margins=margins.astype(np.float32),
    )
    receipt = seal({
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "subset_manifest": subset["manifest_signature"],
        "index_receipt": _signature(
            os.path.join(str(job["index_output"]), "index-receipt.json"),
            label="R0132 index receipt",
        ),
        "index": index_receipt["index"],
        "fixed_policy": {
            "nprobe": SEARCH_NPROBE,
            "shortlist_width": SEARCH_SHORTLIST_WIDTH,
            "selected_neighbors": GRAPH_K,
            "candidate_universe": HALF_RETAINED_ROWS,
            "policy_sweep_or_widening_performed": False,
        },
        "sample": {
            "seed": SEARCH_SEED,
            "rows_per_group": SEARCH_ANCHORS_PER_GROUP,
            "sample_rows_ordered_sha256": ordered_array_sha256(sample),
            "group_ids_ordered_sha256": ordered_array_sha256(group_ids),
            "boundary_ties": int(ties.sum()),
            "boundary_tie_atol": BOUNDARY_TIE_ATOL,
        },
        "quality": metrics,
        "truth_arrays": expected_input_signature(arrays_path),
        "performance": {
            "truth": truth_timing,
            "search_and_rerank": execution,
            "wall_seconds": time.monotonic() - started,
        },
        "checks": {
            **metrics["checks"],
            "one_local_cuda_device": (
                faiss.get_num_gpus() == 1 and torch.cuda.device_count() == 1
            ),
            SEARCH_EXACT_UNIVERSE_CHECK: True,
            "no_graph_or_map_built": True,
        },
        "outcome": (
            SEARCH_POSITIVE_OUTCOME
            if metrics["passed"]
            else SEARCH_NEGATIVE_OUTCOME
        ),
        "graph_build_released": metrics["passed"],
        "training_performed": False,
        "optimizer_updates": 0,
        "map_decision_made": False,
    })
    receipt_path = os.path.join(output, "qualification.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    if not metrics["passed"]:
        raise Round0132Error("R0132 fixed R0105 search policy failed closed")
    del gpu, resources, cpu, encoded, scales
    gc.collect()
    torch.cuda.empty_cache()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _quality_admission(job: Mapping[str, Any], subset: Mapping[str, Any]) -> dict[str, Any]:
    path = os.path.join(str(job["qualification_output"]), "qualification.json")
    value = _load_json(path)
    validate_seal(value, label="R0132 search qualification")
    if (
        value.get("schema") != QUALIFICATION_SCHEMA
        or value.get("subset_manifest") != subset["manifest_signature"]
        or value.get("graph_build_released") is not True
        or value.get("outcome")
        != SEARCH_POSITIVE_OUTCOME
        or not all((value.get("checks") or {}).values())
    ):
        raise Round0132Error("R0132 graph search admission is not positive")
    return _signature(path, label="R0132 search qualification")


def _part_contract(
    *,
    part: str,
    spec: Mapping[str, Any],
    release_sha: str,
    subset: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> str:
    return sha256_bytes(canonical_json({
        "schema": GRAPH_PART_SCHEMA,
        "round_id": ROUND_ID,
        "part": part,
        "part_spec": dict(spec),
        "release_sha": release_sha,
        "subset_manifest": subset["manifest_signature"],
        "quality_admission": dict(quality),
        "k": GRAPH_K,
        "n_neighbors_including_self": N_NEIGHBORS,
        "shard_rows": SHARD_ROWS,
    }))


def run_graph_part(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    import faiss
    import torch

    part = str(job["part"])
    subset = _load_subset(str(job["subset_output"]))
    if part not in subset["parts"]:
        raise Round0132Error(f"unknown R0132 graph part {part!r}")
    spec = subset["parts"][part]
    quality = _quality_admission(job, subset)
    _load_index_receipt(job, subset)
    output = ensure_data_directory(
        str(job["outputs"][0]), label=f"R0132 {part} graph shards"
    )
    completed_path = os.path.join(output, "part-receipt.json")
    contract_sha = _part_contract(
        part=part,
        spec=spec,
        release_sha=active["manifest"]["release_sha"],
        subset=subset,
        quality=quality,
    )
    if os.path.exists(completed_path):
        completed = _load_json(completed_path)
        validate_seal(completed, label=f"R0132 {part} part receipt")
        if completed.get("contract_sha256") != contract_sha:
            raise Round0132Error(f"R0132 completed {part} contract changed")
        return {**completed, "receipt": _signature(completed_path, label=f"R0132 {part} receipt")}
    started = time.monotonic()
    _substrate, _original_excluded, encoded, scales = _substrate_arrays()
    index = faiss.read_index(str(job["index"]))
    _require_geometry(index, ntotal=HALF_RETAINED_ROWS)
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    clone_started = time.monotonic()
    gpu = faiss.index_cpu_to_gpu(resources, 0, index, _gpu_options())
    clone_seconds = time.monotonic() - clone_started
    gpu.nprobe = SEARCH_NPROBE
    receipts: list[dict[str, Any]] = []
    completed_new = 0
    subfloor_streak = 0
    for shard, start in enumerate(
        range(int(spec["compact_start"]), int(spec["compact_stop"]), SHARD_ROWS)
    ):
        stop = min(start + SHARD_ROWS, int(spec["compact_stop"]))
        receipt = _write_shard(
            gpu=gpu,
            part=part,
            shard=shard,
            compact_start=start,
            compact_stop=stop,
            width=SEARCH_SHORTLIST_WIDTH,
            excluded=subset["excluded"],
            encoded=encoded,
            scales=scales,
            output=output,
            contract_sha256=contract_sha,
            contract=GRAPH_CONTRACT,
            universe_rows=HALF_RETAINED_ROWS,
        )
        receipts.append(receipt)
        if receipt["resumed"] is not True:
            completed_new += 1
            subfloor_streak = update_performance_streak(
                subfloor_streak,
                completed_new_shards=completed_new,
                sources_per_second=float(receipt["performance"]["sources_per_second"]),
            )
        if subfloor_streak >= PERFORMANCE_SUBFLOOR_PATIENCE:
            raise Round0132Error(
                "R0132 graph throughput remained below the reviewed gross floor"
            )
        print(
            f"R0132 {part} {stop - int(spec['compact_start']):,}/"
            f"{int(spec['retained_rows']):,}",
            flush=True,
        )
    retained_sources = sum(int(value["retained_sources"]) for value in receipts)
    knn_edges = sum(int(value["knn_edges"]) for value in receipts)
    directed_edges = sum(int(value["directed_edges"]) for value in receipts)
    zero_edges = sum(int(value["zero_memberships_eliminated"]) for value in receipts)
    if (
        retained_sources != int(spec["retained_rows"])
        or knn_edges != retained_sources * GRAPH_K
        or directed_edges + zero_edges != knn_edges
    ):
        raise Round0132Error(f"R0132 {part} graph accounting did not close")
    receipt = seal({
        "schema": GRAPH_PART_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "part": part,
        "part_spec": spec,
        "contract_sha256": contract_sha,
        "subset_manifest": subset["manifest_signature"],
        "quality_admission": quality,
        "retained_sources": retained_sources,
        "knn_edges": knn_edges,
        "directed_edges": directed_edges,
        "zero_memberships_eliminated": zero_edges,
        "sources_with_eliminated_memberships": sum(
            int(value["sources_with_eliminated_memberships"]) for value in receipts
        ),
        "minimum_memberships_per_source": min(
            int(value["minimum_memberships_per_source"]) for value in receipts
        ),
        "shards": [
            {
                "shard": int(value["shard"]),
                "compact_start": int(value["compact_start"]),
                "compact_stop": int(value["compact_stop"]),
                "receipt": expected_input_signature(
                    _shard_paths(output, int(value["shard"]))[1]
                ),
                "artifact": value["artifact"],
            }
            for value in receipts
        ],
        "pipeline": {
            "candidate_universe": GRAPH_CANDIDATE_UNIVERSE,
            "nprobe": SEARCH_NPROBE,
            "shortlist_width": SEARCH_SHORTLIST_WIDTH,
            "exact_native_rerank": True,
            "selected_neighbors": GRAPH_K,
            "fuzzy_kernel": "reviewed R0106 fp32 UMAP kernel",
            "symmetrization_deferred": "a+b-a*b",
        },
        "performance": {
            "clone_seconds": clone_seconds,
            "wall_seconds": time.monotonic() - started,
            "ending_subfloor_streak": subfloor_streak,
            "minimum_sources_per_second": MINIMUM_SHARD_SOURCES_PER_SECOND,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "map_decision_made": False,
    })
    atomic_write_new_json(completed_path, receipt, immutable=True)
    del gpu, resources, index, encoded, scales
    gc.collect()
    torch.cuda.empty_cache()
    return {**receipt, "receipt": expected_input_signature(completed_path)}


def _validate_part(
    root: str,
    *,
    part: str,
    subset: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = os.path.join(root, "part-receipt.json")
    value = _load_json(path)
    validate_seal(value, label=f"R0132 {part} part")
    spec = subset["parts"][part]
    if (
        value.get("schema") != GRAPH_PART_SCHEMA
        or value.get("part") != part
        or value.get("part_spec") != spec
        or value.get("subset_manifest") != subset["manifest_signature"]
        or int(value.get("retained_sources", -1)) != int(spec["retained_rows"])
        or int(value.get("knn_edges", -1)) != int(spec["retained_rows"]) * GRAPH_K
        or int(value.get("directed_edges", -1))
        + int(value.get("zero_memberships_eliminated", -1))
        != int(value.get("knn_edges", -1))
    ):
        raise Round0132Error(f"R0132 {part} part receipt changed")
    for member in value.get("shards") or []:
        _signature(member["artifact"]["canonical_path"], member["artifact"]["sha256"], label=f"R0132 {part} shard")
        _signature(member["receipt"]["canonical_path"], member["receipt"]["sha256"], label=f"R0132 {part} shard receipt")
    return value, _signature(path, label=f"R0132 {part} part receipt")


def _subset_graph_diagnostics(
    *,
    sources_path: str,
    targets_path: str,
    weights_path: str,
    group_ids: np.ndarray,
) -> dict[str, Any]:
    sources = np.load(sources_path, mmap_mode="r", allow_pickle=False)
    targets = np.load(targets_path, mmap_mode="r", allow_pickle=False)
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if (
        sources.shape != targets.shape
        or sources.shape != weights.shape
        or sources.dtype != np.int32
        or targets.dtype != np.int32
        or weights.dtype != np.float32
    ):
        raise Round0132Error("R0132 final graph arrays are malformed")
    mixing = np.zeros((len(GROUPS), len(GROUPS)), dtype=np.int64)
    weight_sum = 0.0
    minimum = float("inf")
    maximum = 0.0
    for start in range(0, len(sources), 25_000_000):
        stop = min(start + 25_000_000, len(sources))
        source = np.asarray(sources[start:stop])
        target = np.asarray(targets[start:stop])
        weight = np.asarray(weights[start:stop])
        if (
            np.any(source < 0)
            or np.any(source >= HALF_RETAINED_ROWS)
            or np.any(target < 0)
            or np.any(target >= HALF_RETAINED_ROWS)
            or np.any(source == target)
            or not np.isfinite(weight).all()
            or np.any(weight <= 0)
            or np.any(weight > 1)
        ):
            raise Round0132Error("R0132 final graph structural scan failed")
        mixing += np.bincount(
            group_ids[source].astype(np.int64) * len(GROUPS)
            + group_ids[target].astype(np.int64),
            minlength=len(GROUPS) ** 2,
        ).reshape(len(GROUPS), len(GROUPS))
        weight_sum += float(weight.astype(np.float64).sum())
        minimum = min(minimum, float(weight.min()))
        maximum = max(maximum, float(weight.max()))
    degrees = np.bincount(sources, minlength=HALF_RETAINED_ROWS)
    if int(degrees.sum()) != len(sources) or np.any(degrees <= 0):
        raise Round0132Error("R0132 final graph degree accounting failed")
    group_rows = mixing.sum(axis=1)
    return {
        "groups": list(GROUPS),
        "mixing_matrix": mixing.tolist(),
        "within_group_fraction": {
            group: float(mixing[index, index] / group_rows[index])
            for index, group in enumerate(GROUPS)
        },
        "degree": {
            "minimum": int(degrees.min()),
            "p10": float(np.percentile(degrees, 10)),
            "median": float(np.median(degrees)),
            "p90": float(np.percentile(degrees, 90)),
            "p99": float(np.percentile(degrees, 99)),
            "maximum": int(degrees.max()),
            "maximum_hub_share": float(degrees.max() / len(sources)),
            "zero_degree_rows": int(np.count_nonzero(degrees == 0)),
        },
        "weights": {
            "minimum": minimum,
            "mean": weight_sum / len(weights),
            "maximum": maximum,
            "sum": weight_sum,
        },
        "structural_scan": {
            "directed_edges": len(sources),
            "endpoint_bounds_valid": True,
            "self_edges": 0,
            "weight_domain_valid": True,
        },
    }


def run_assemble_graph(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = ensure_data_directory(
        str(job["outputs"][0]), label="R0132 assembled half graph"
    )
    manifest_path = os.path.join(output, "graph-manifest.json")
    if os.path.exists(manifest_path):
        value = _load_json(manifest_path)
        validate_seal(value, label="R0132 graph manifest")
        return {**value, "receipt": _signature(manifest_path, label="R0132 graph manifest")}
    started = time.monotonic()
    subset = _load_subset(str(job["subset_output"]))
    parts: dict[str, tuple[str, Mapping[str, Any]]] = {}
    part_signatures: dict[str, Any] = {}
    for part in GRAPH_PART_NAMES:
        root = str(job["part_outputs"][part])
        value, signature = _validate_part(root, part=part, subset=subset)
        parts[part] = (root, value)
        part_signatures[part] = signature
    quality_values = {
        canonical_json(value["quality_admission"]): value["quality_admission"]
        for _root, value in parts.values()
    }
    if len(quality_values) != 1:
        raise Round0132Error("R0132 graph parts disagree on qualification")
    quality = dict(next(iter(quality_values.values())))
    knn_edges = sum(int(value["knn_edges"]) for _root, value in parts.values())
    forward_edges = sum(int(value["directed_edges"]) for _root, value in parts.values())
    zero_edges = sum(
        int(value["zero_memberships_eliminated"]) for _root, value in parts.values()
    )
    if knn_edges != HALF_RETAINED_ROWS * GRAPH_K or forward_edges + zero_edges != knn_edges:
        raise Round0132Error("R0132 pre-assembly graph accounting changed")
    contract = sha256_bytes(canonical_json({
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "subset_manifest": subset["manifest_signature"],
        "parts": part_signatures,
        "quality_admission": quality,
        "retained_rows": HALF_RETAINED_ROWS,
        "k": GRAPH_K,
        "pair_buckets": PAIR_BUCKETS,
        "symmetrization": "a+b-a*b-set-op-mix-ratio-1",
    }))
    buckets = _partition_forward_edges(
        output=output,
        parts=parts,
        contract_sha256=contract,
        part_order=GRAPH_PART_NAMES,
        universe_rows=HALF_RETAINED_ROWS,
    )
    joined, join_stats = phase_c_join(
        buckets,
        output,
        HALF_RETAINED_ROWS,
        PAIR_BUCKETS,
        contract_sha256=contract,
        workers=8,
    )
    counts = [int(value) for value in join_stats["counts"]]
    reciprocity = _validate_joined_reciprocity(
        joined=joined, counts=counts, universe_rows=HALF_RETAINED_ROWS
    )
    sources, targets, weights = _publish_memmaps(
        output=output, joined=joined, counts=counts
    )
    diagnostics = _subset_graph_diagnostics(
        sources_path=sources["canonical_path"],
        targets_path=targets["canonical_path"],
        weights_path=weights["canonical_path"],
        group_ids=subset["group_ids"],
    )
    manifest = seal({
        "schema": GRAPH_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "contract_sha256": contract,
        "row_count": ROW_COUNT,
        "retained_rows": HALF_RETAINED_ROWS,
        "dimension": DIMENSION,
        "k_real": GRAPH_K,
        "n_neighbors_including_self": N_NEIGHBORS,
        "local_connectivity": LOCAL_CONNECTIVITY,
        "subset_manifest": subset["manifest_signature"],
        "quality_admission": quality,
        "part_receipts": part_signatures,
        "compact_mapping": subset["signatures"]["mapping"],
        "compact_group_ids": subset["signatures"]["group_ids"],
        "outputs": {"sources": sources, "targets": targets, "weights": weights},
        "knn_topology": {
            "distinct_nonself_neighbors_per_source": GRAPH_K,
            "knn_edge_count": knn_edges,
            "source_coverage_complete": True,
        },
        "forward_memberships": {
            "positive_count": forward_edges,
            "zero_memberships_eliminated": zero_edges,
            "elimination_semantics": "umap-eliminate-zeros-after-fp32-cast",
        },
        "directed_edge_count": int(join_stats["n_edges"]),
        "weight_sum": float(join_stats["weight_sum"]),
        "symmetrization": {
            "method": "probabilistic-t-conorm",
            "formula": "a+b-a*b",
            "set_op_mix_ratio": 1.0,
            "unordered_pair_partitions": PAIR_BUCKETS,
            "both_orientations_emitted": True,
        },
        "diagnostics": diagnostics,
        "reciprocity_validation": reciprocity,
        "performance": {
            "wall_seconds": time.monotonic() - started,
            "assembly_workers": 8,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "training_performed": False,
        "optimizer_updates": 0,
        "map_decision_made": False,
    })
    atomic_write_new_json(manifest_path, manifest, immutable=True)
    return {**manifest, "receipt": expected_input_signature(manifest_path)}


def run_train(
    active: dict[str, Any], job: dict[str, Any]
) -> dict[str, Any]:
    selected = dict(job)
    graph_signature = _signature(
        str(job["graph_manifest"]), label="R0132 assembled graph"
    )
    graph = _load_json(str(job["graph_manifest"]))
    validate_seal(graph, label="R0132 assembled graph")
    if (
        graph.get("schema") != GRAPH_SCHEMA
        or graph.get("round_id") != ROUND_ID
        or graph.get("release_sha") != active["manifest"]["release_sha"]
    ):
        raise Round0132Error("R0132 assembled graph lineage changed")
    selected["graph_manifest_sha256"] = graph_signature["sha256"]
    selected["graph_release_sha"] = graph["release_sha"]
    return run_train_contract(
        active,
        selected,
        round_id=ROUND_ID,
        seed=SEED,
        train_config_schema=TRAIN_CONFIG_SCHEMA,
        production_config_schema=PRODUCTION_CONFIG_SCHEMA,
        train_receipt_schema=TRAIN_RECEIPT_SCHEMA,
        output_label="R0132 12.5M coverage-aligned train output",
        graph_load_kwargs={
            "expected_graph_schema": GRAPH_SCHEMA,
            "expected_graph_round_id": ROUND_ID,
            "expected_k_real": GRAPH_K,
            "expected_retained_rows": HALF_RETAINED_ROWS,
        },
        train_config_kwargs={
            "n_neighbors_including_self": N_NEIGHBORS,
            "compact_retained_rows": HALF_RETAINED_ROWS,
            "pipeline": PIPELINE,
            "pipeline_schema": PIPELINE_SCHEMA,
            "sampler_class": SAMPLER_CLASS,
            "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
            "graph_degree": "variable-symmetric-fuzzy-k15-topology",
            "update_rule": "ceil(actual-R0132-directed-fuzzy-edges/409)",
        },
        training_input_kwargs={
            "positive_destination_policy": POSITIVE_DESTINATION_POLICY,
            "graph_degree": "variable-symmetric-fuzzy-k15-topology",
        },
    )


def _load_model_bundle(
    *,
    train_output: str,
    graph_manifest: str,
    graph_sha256: str,
    half: bool,
) -> dict[str, Any]:
    return load_reviewed_model(
        train_output=train_output,
        graph_manifest_path=graph_manifest,
        graph_manifest_sha256=graph_sha256,
        expected_train_round_id=ROUND_ID if half else "0107",
        expected_train_receipt_schema=(TRAIN_RECEIPT_SCHEMA if half else FULL_TRAIN_SCHEMA),
        expected_production_config_schema=(
            PRODUCTION_CONFIG_SCHEMA if half else FULL_PRODUCTION_SCHEMA
        ),
        expected_seed=SEED,
        expected_graph_schema=GRAPH_SCHEMA if half else FULL_GRAPH_SCHEMA,
    )


class _SliceView:
    def __init__(self, source: Any, start: int, stop: int):
        self.source = source
        self.start = int(start)
        self.stop = int(stop)
        self.shape = (self.stop - self.start, source.shape[1])
        self.dtype = source.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return self.source[self.start + start : self.start + stop : step]
        rows = np.asarray(key, dtype=np.int64)
        return self.source[rows + self.start]


def run_transform(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0132 half coordinate stream"
    )
    started = time.monotonic()
    graph_signature = _signature(str(job["graph_manifest"]), label="R0132 graph manifest")
    bundle = _load_model_bundle(
        train_output=str(job["train_output"]),
        graph_manifest=str(job["graph_manifest"]),
        graph_sha256=graph_signature["sha256"],
        half=True,
    )
    source = CompactInt8DequantizedArray(bundle["mapping"])
    if len(source) != HALF_RETAINED_ROWS:
        raise Round0132Error("R0132 transform source universe changed")
    members: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, HALF_RETAINED_ROWS, TRANSFORM_CHUNK_ROWS)):
        stop = min(start + TRANSFORM_CHUNK_ROWS, HALF_RETAINED_ROWS)
        root = create_fresh_directory(
            os.path.join(output, f"chunk-{index:05d}"),
            label="R0132 coordinate chunk",
        )
        coordinates = np.asarray(
            bundle["model"].transform(
                _SliceView(source, start, stop), batch_size=TRANSFORM_BATCH_ROWS
            ),
            dtype=np.float32,
        )
        if coordinates.shape != (stop - start, 2) or not np.isfinite(coordinates).all():
            raise Round0132Error("R0132 transform emitted malformed coordinates")
        path = os.path.join(root, "coordinates.npy")
        atomic_save_new_npy(path, coordinates, immutable=True)
        signature = expected_input_signature(path)
        members.append({
            "chunk_index": index,
            "global_row_start": start,
            "global_row_stop": stop,
            "bytes": signature["bytes"],
            "sha256": signature["sha256"],
        })
        del coordinates
    receipt = coordinate_seal({
        "schema": TRANSFORM_SCHEMA_R0132,
        "round_id": ROUND_ID,
        "map_key": TRANSFORM_MAP_KEY,
        "model": bundle["train"]["model"],
        "train_receipt": bundle["train_signature"],
        "production_config": bundle["config_signature"],
        "graph_manifest": bundle["graph_signature"],
        "compact_mapping": bundle["graph"]["compact_mapping"],
        "substrate": source.substrate["signature"],
        "scientific_universe": TRANSFORM_SCIENTIFIC_UNIVERSE,
        "input_preprocessing": (
            "signed-int8 times exact fp16 row scale to device fp32; no L2 "
            "renormalization before model"
        ),
        "row_accounting": {
            "all_rows": HALF_RETAINED_ROWS,
            "retained_representatives": HALF_RETAINED_ROWS,
            "original_rows": ROW_COUNT,
            "not_selected_or_excluded_rows": ROW_COUNT - HALF_RETAINED_ROWS,
        },
        "coordinate_stream": {
            "schema": COORDINATE_SCHEMA,
            "row_count": HALF_RETAINED_ROWS,
            "dimension": 2,
            "dtype": "<f4",
            "row_order": TRANSFORM_ROW_ORDER,
            "ordered_chunks": members,
        },
        "inference": {
            "batch_rows": TRANSFORM_BATCH_ROWS,
            "chunk_rows": TRANSFORM_CHUNK_ROWS,
            "all_real_rows_projected": True,
        },
        "release_sha": active["manifest"]["release_sha"],
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    receipt_path = os.path.join(output, "actual-transform.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    del bundle["model"], source
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


class _MappedRows:
    """Lazy row view used to score accepted 25M coordinates on U12."""

    def __init__(self, source: Any, rows: np.ndarray):
        self.source = source
        self.rows = np.asarray(rows, dtype=np.int64)
        self.shape = (len(self.rows), int(source.shape[1]))
        self.dtype = source.dtype

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: Any) -> np.ndarray:
        if isinstance(key, tuple):
            rows, columns = key
            return self[rows][..., columns]
        return self.source[self.rows[key]]

    def _reduce(self, operation: Any, initial: float, axis: int | None):
        if axis not in (None, 0):
            raise ValueError("mapped coordinate reduction supports axis 0/all")
        value = np.full(self.shape[1], initial, dtype=np.float32)
        for start in range(0, len(self), 1_000_000):
            block = np.asarray(self[start : min(start + 1_000_000, len(self))])
            value = operation(value, operation.reduce(block, axis=0))
        return value if axis == 0 else operation.reduce(value)

    def min(self, axis: int | None = None):
        return self._reduce(np.minimum, np.inf, axis)

    def max(self, axis: int | None = None):
        return self._reduce(np.maximum, -np.inf, axis)


def _native_metrics(
    *,
    coordinates: Any,
    anchors: np.ndarray,
    high10: np.ndarray,
    config: Any,
) -> tuple[
    dict[str, float],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    from basemap.panel_v2 import _self_knn

    fraction_k = max(K_LOW_MAX, int(math.ceil(FRACTION * len(coordinates))))
    low, distances, guard = _self_knn(
        coordinates,
        anchors,
        fraction_k,
        config,
        hi_dim=False,
        want_dist=True,
        exact=True,
    )
    low = np.asarray(low, dtype=np.int64)
    distances = np.asarray(distances, dtype=np.float32)
    raw = projection_metrics(high10, low, fraction_k=fraction_k)
    metrics = {
        "global_ffr": float(raw["ffr_diagnostic"]),
        "global_recall_at_10": float(raw["recall_at_10"]),
        "global_recall_at_50_of_high10": float(raw["recall_at_50_of_high10"]),
    }
    ffr_truth_hits = _native_ffr_truth_hits(high10, low, fraction_k=fraction_k)
    if not math.isclose(
        metrics["global_ffr"],
        float(ffr_truth_hits.mean()),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise Round0132Error("R0132 native global FFR evidence disagrees")
    return (
        metrics,
        distances[:, :K_DENSITY].mean(1),
        low[:, :K_LOW_MAX],
        ffr_truth_hits,
        guard,
    )


def _native_ffr_truth_hits(
    high10: np.ndarray,
    low_neighbors: np.ndarray,
    *,
    fraction_k: int,
) -> np.ndarray:
    """Preserve the per-anchor truth membership needed to recompute FFR.

    The full ~0.1%-width neighbor matrix is unnecessarily large.  This exact
    boolean membership matrix is sufficient evidence for the registered global
    FFR scalar while remaining tiny and auditable at the terminal decision.
    """
    truth = np.asarray(high10, dtype=np.int64)
    low = np.asarray(low_neighbors, dtype=np.int64)
    if (
        truth.ndim != 2
        or truth.shape[1] != K_HIT
        or low.ndim != 2
        or low.shape[0] != truth.shape[0]
        or fraction_k < K_LOW_MAX
        or low.shape[1] < fraction_k
    ):
        raise Round0132Error("R0132 native global FFR inputs are malformed")
    hits = np.empty(truth.shape, dtype=bool)
    # Bound the temporary broadcast to roughly 32 MB at production width.
    block_rows = max(1, 32_000_000 // (K_HIT * fraction_k))
    for start in range(0, len(truth), block_rows):
        stop = min(start + block_rows, len(truth))
        hits[start:stop] = np.any(
            truth[start:stop, :, None]
            == low[start:stop, None, :fraction_k],
            axis=2,
        )
    return hits


def run_score_native(
    _active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    from basemap.panel_v2 import _self_knn

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0132 matched U12 native panel"
    )
    started = time.monotonic()
    subset = _load_subset(str(job["subset_output"]))
    ranges = group_ranges(validate_substrate_manifest(verify_payloads=False)["manifest"])
    global_anchors, group_ids = sample_stratified_rows(
        subset["excluded"],
        ranges,
        rows_per_group=NATIVE_ANCHORS_PER_GROUP,
        seed=NATIVE_ANCHOR_SEED,
    )
    compact_anchors = np.searchsorted(subset["mapping"], global_anchors)
    if not np.array_equal(subset["mapping"][compact_anchors], global_anchors):
        raise Round0132Error("R0132 native anchors are outside U12")
    half_graph_signature = _signature(str(job["graph_manifest"]), label="R0132 graph")
    half_bundle = _load_model_bundle(
        train_output=str(job["train_output"]),
        graph_manifest=str(job["graph_manifest"]),
        graph_sha256=half_graph_signature["sha256"],
        half=True,
    )
    source = CompactInt8DequantizedArray(half_bundle["mapping"])
    config = _panel_config(anchors=len(compact_anchors))
    high_neighbors, high_distances, high_guard = _self_knn(
        source,
        compact_anchors,
        GRAPH_K,
        config,
        hi_dim=True,
        want_dist=True,
        exact=True,
    )
    high_neighbors = np.asarray(high_neighbors, dtype=np.int64)
    high_distances = np.asarray(high_distances, dtype=np.float64)
    high_radius = high_distances[:, :K_DENSITY].mean(1)
    high10 = high_neighbors[:, :K_HIT]
    half_coordinates = CoordinateStream(str(job["transform_output"]))
    if len(half_coordinates) != HALF_RETAINED_ROWS:
        raise Round0132Error("R0132 half coordinate stream changed")
    (
        control_metrics,
        control_low_radius,
        control_low50,
        control_ffr_hits,
        control_guard,
    ) = _native_metrics(
        coordinates=half_coordinates,
        anchors=compact_anchors,
        high10=high10,
        config=config,
    )
    full_coordinates = CoordinateStream(
        str(job["full_transform_output"]),
        expected_receipt_sha256=str(job["full_transform_receipt_sha256"]),
    )
    full_mapping = np.load(str(job["full_mapping"]), mmap_mode="r", allow_pickle=False)
    _signature(str(job["full_mapping"]), str(job["full_mapping_sha256"]), label="R0106 full mapping")
    full_positions = np.searchsorted(full_mapping, subset["mapping"])
    if (
        np.any(full_positions >= len(full_mapping))
        or not np.array_equal(full_mapping[full_positions], subset["mapping"])
    ):
        raise Round0132Error("R0132 U12 is not a subset of accepted R0106 mapping")
    treatment_coordinates = _MappedRows(full_coordinates, full_positions)
    (
        treatment_metrics,
        treatment_low_radius,
        treatment_low50,
        treatment_ffr_hits,
        treatment_guard,
    ) = _native_metrics(
        coordinates=treatment_coordinates,
        anchors=compact_anchors,
        high10=high10,
        config=config,
    )
    representatives, family_counts = _family_arrays(str(job["eligibility"]))
    family_sizes = map_family_sizes(global_anchors, representatives, family_counts)
    eligible = family_sizes < FAMILY_SIZE_CUTOFF
    density = paired_density_bootstrap(
        high_radius,
        control_low_radius,
        treatment_low_radius,
        eligible=eligible,
    )
    deltas = np.asarray(density.pop("bootstrap_deltas"), dtype=np.float64)
    stale_calibration = read_sealed(
        str(job["stale_calibration"]),
        label="R0108 stale Jina density calibration",
    )
    stale_floor = (stale_calibration.get("floor_calibration") or {}).get(
        "registered_floor"
    )
    control_min = np.asarray(half_coordinates.min(axis=0), dtype=np.float64)
    control_max = np.asarray(half_coordinates.max(axis=0), dtype=np.float64)
    treatment_min = np.asarray(
        treatment_coordinates.min(axis=0), dtype=np.float64
    )
    treatment_max = np.asarray(
        treatment_coordinates.max(axis=0), dtype=np.float64
    )
    control_finite = bool(
        np.isfinite(control_min).all()
        and np.isfinite(control_max).all()
        and np.all(control_max - control_min > 1e-6)
    )
    treatment_finite = bool(
        np.isfinite(treatment_min).all()
        and np.isfinite(treatment_max).all()
        and np.all(treatment_max - treatment_min > 1e-6)
    )
    arrays_path = os.path.join(output, "matched-native-arrays.npz")
    atomic_save_new_npz(
        arrays_path,
        immutable=True,
        global_anchor_rows=global_anchors,
        compact_anchor_rows=compact_anchors,
        group_ids=group_ids,
        high_neighbors_top15=high_neighbors,
        high_radius=high_radius,
        control_low_radius=control_low_radius,
        treatment_low_radius=treatment_low_radius,
        control_low_neighbors_top50=control_low50,
        treatment_low_neighbors_top50=treatment_low50,
        native_fraction_k=np.asarray(
            max(K_LOW_MAX, int(math.ceil(FRACTION * HALF_RETAINED_ROWS))),
            dtype=np.int64,
        ),
        control_ffr_truth_hits=control_ffr_hits,
        treatment_ffr_truth_hits=treatment_ffr_hits,
        family_sizes=family_sizes,
        density_bootstrap_deltas=deltas,
    )
    receipt = seal({
        "schema": NATIVE_SCHEMA,
        "round_id": ROUND_ID,
        "subset_manifest": subset["manifest_signature"],
        "model_lineage": {
            "control_12p5m_train_receipt": half_bundle["train_signature"],
            "control_12p5m_production_config": half_bundle["config_signature"],
            "control_12p5m_graph": half_bundle["graph_signature"],
            "control_12p5m_transform": _signature(
                os.path.join(str(job["transform_output"]), "actual-transform.json"),
                label="R0132 transform receipt",
            ),
            "treatment_25m_transform": _signature(
                os.path.join(str(job["full_transform_output"]), "actual-transform.json"),
                str(job["full_transform_receipt_sha256"]),
                label="R0108 transform receipt",
            ),
        },
        "control_12p5m": {
            **control_metrics,
            "finite_noncollapsed": control_finite,
        },
        NATIVE_TREATMENT_KEY: {
            **treatment_metrics,
            "finite_noncollapsed": treatment_finite,
        },
        "density_selector": density,
        "stale_absolute_jina_floor": {
            "value": stale_floor,
            "role": "diagnostic-only; changed universe portability unresolved",
            "can_gate_or_rescue": False,
            "calibration": _signature(str(job["stale_calibration"]), label="R0108 calibration"),
        },
        "native_global_ffr_role": NATIVE_GLOBAL_FFR_ROLE,
        "ood_projection_ffr_role": "diagnostic-only",
        "truth": {
            "computed_once_and_shared_by_both_maps": True,
            "high_d_guard": high_guard,
            "anchor_seed": NATIVE_ANCHOR_SEED,
            "anchors_per_group": NATIVE_ANCHORS_PER_GROUP,
        },
        "low_d_guards": {"control": control_guard, "treatment": treatment_guard},
        "arrays": expected_input_signature(arrays_path),
        "checks": {
            NATIVE_SHARED_ROWS_CHECK: True,
            "same_high_d_truth_and_anchors": True,
            "control_finite_noncollapsed": control_finite,
            "treatment_finite_noncollapsed": treatment_finite,
            "control_recall50_at_least_recall10": (
                control_metrics["global_recall_at_50_of_high10"]
                >= control_metrics["global_recall_at_10"]
            ),
            "treatment_recall50_at_least_recall10": (
                treatment_metrics["global_recall_at_50_of_high10"]
                >= treatment_metrics["global_recall_at_10"]
            ),
            "paired_density_bootstrap_exactly_1000": len(deltas) == DENSITY_BOOTSTRAP_DRAWS,
            "stale_floor_diagnostic_only": True,
        },
        "training_performed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "matched-native.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del half_bundle["model"], source, full_coordinates, half_coordinates
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def _matched_probe(
    *,
    name: str,
    corpus: np.ndarray,
    queries: np.ndarray,
    control_model: Any,
    treatment_model: Any,
    duplicate_policy: str,
) -> dict[str, Any]:
    from basemap.panel_v2 import cross_knn

    duplicate = exact_split_duplicate_diagnostics(corpus, queries)
    if (
        duplicate_policy == "require-disjoint"
        and not duplicate["corpus_query_exact_family_disjoint"]
    ):
        raise Round0132Error(f"{name} exact family crosses corpus/query split")
    truth, truth_guard = exact_cosine_topk(queries, corpus, k=K_HIT)
    config = _panel_config(anchors=len(queries))
    fraction_k = max(K_LOW_MAX, int(math.ceil(FRACTION * len(corpus))))
    cells: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {"exact_high_d_top10": truth}
    for label, model in (
        ("control_12p5m", control_model),
        ("treatment_25m", treatment_model),
    ):
        corpus_coordinates = np.asarray(
            model.transform(corpus, batch_size=TRANSFORM_BATCH_ROWS), dtype=np.float32
        )
        query_coordinates = np.asarray(
            model.transform(queries, batch_size=TRANSFORM_BATCH_ROWS), dtype=np.float32
        )
        low = cross_knn(
            query_coordinates,
            corpus_coordinates,
            fraction_k,
            config,
            hi_dim=False,
            exact=True,
        )
        metrics = projection_metrics(truth, low, fraction_k=fraction_k)
        cells[label] = {
            "recall_at_10": float(metrics["recall_at_10"]),
            "recall_at_50_of_high10": float(metrics["recall_at_50_of_high10"]),
            "projection_ffr": float(metrics["ffr_diagnostic"]),
            "projection_ffr_role": "diagnostic-only",
            "finite_noncollapsed": bool(
                np.isfinite(corpus_coordinates).all()
                and np.isfinite(query_coordinates).all()
                and np.all(corpus_coordinates.std(axis=0) > 1e-8)
            ),
        }
        arrays[f"{label}_query_coordinates"] = query_coordinates
        arrays[f"{label}_low_neighbors_top50"] = np.asarray(
            low[:, :K_LOW_MAX], dtype=np.int64
        )
    return {
        "name": name,
        "corpus_rows": len(corpus),
        "query_rows": len(queries),
        "truth_computed_once_for_both_models": True,
        "truth_guard": truth_guard,
        "duplicate_control": duplicate,
        "cells": cells,
        "arrays": arrays,
    }


def _selected_source(
    source_signature: Mapping[str, Any],
    corpus_rows: np.ndarray,
    query_rows: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    _signature(
        str(source_signature["canonical_path"]),
        str(source_signature["sha256"]),
        label=label,
    )
    source = np.load(str(source_signature["canonical_path"]), mmap_mode="r", allow_pickle=False)
    if source.ndim != 2 or source.shape[1] != DIMENSION:
        raise Round0132Error(f"{label} source geometry changed")
    return np.asarray(source[corpus_rows]), np.asarray(source[query_rows])


def run_score_ood(
    _active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0132 matched held-out OOD panel"
    )
    started = time.monotonic()
    half_graph = _signature(str(job["graph_manifest"]), label="R0132 graph")
    control = _load_model_bundle(
        train_output=str(job["train_output"]),
        graph_manifest=str(job["graph_manifest"]),
        graph_sha256=half_graph["sha256"],
        half=True,
    )
    full_graph = _signature(
        str(job["full_graph_manifest"]),
        str(job["full_graph_manifest_sha256"]),
        label="R0106 graph",
    )
    treatment = _load_model_bundle(
        train_output=str(job["full_train_output"]),
        graph_manifest=str(job["full_graph_manifest"]),
        graph_sha256=full_graph["sha256"],
        half=False,
    )
    selection_signature = _signature(
        str(job["selection"]), str(job["selection_sha256"]), label="R0108 selectors"
    )
    probe_receipts: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    with np.load(str(job["selection"]), allow_pickle=False) as selected:
        for language in (*IN_MIX_LANGUAGES, POLISH):
            corpus_rows = np.asarray(selected[f"{language}__corpus"], dtype=np.int64)
            query_rows = np.asarray(selected[f"{language}__queries"], dtype=np.int64)
            if corpus_rows.shape != (HELDOUT_CORPUS_ROWS,) or query_rows.shape != (HELDOUT_QUERY_ROWS,):
                raise Round0132Error(f"{language} R0108 selector changed")
            corpus, queries = _selected_source(
                job["language_sources"][language],
                corpus_rows,
                query_rows,
                label=f"R0108 {language}",
            )
            report = _matched_probe(
                name=language,
                corpus=corpus,
                queries=queries,
                control_model=control["model"],
                treatment_model=treatment["model"],
                duplicate_policy="require-disjoint",
            )
            for key, value in report.pop("arrays").items():
                arrays[f"{language}__{key}"] = value
            probe_receipts[language] = report
            del corpus, queries

        fineweb_rows = np.asarray(selected["fineweb__corpus"], dtype=np.int64)
        fineweb_queries = np.asarray(selected["fineweb__queries"], dtype=np.int64)
        corpus, queries = _selected_source(
            job["diagnostic_sources"]["fineweb"],
            fineweb_rows,
            fineweb_queries,
            label="R0108 held-out FineWeb",
        )
        fineweb = _matched_probe(
            name="fineweb-heldout",
            corpus=corpus,
            queries=queries,
            control_model=control["model"],
            treatment_model=treatment["model"],
            duplicate_policy="diagnostic",
        )
        for key, value in fineweb.pop("arrays").items():
            arrays[f"fineweb__{key}"] = value
        probe_receipts["fineweb-heldout"] = fineweb

        dad_rows = np.asarray(selected["dadabase__corpus"], dtype=np.int64)
        dad_queries = np.asarray(selected["dadabase__queries"], dtype=np.int64)
        dad_corpus, dad_query = _selected_source(
            job["diagnostic_sources"]["dadabase"],
            dad_rows,
            dad_queries,
            label="R0108 Dadabase",
        )
        dadabase = _matched_probe(
            name="dadabase",
            corpus=dad_corpus,
            queries=dad_query,
            control_model=control["model"],
            treatment_model=treatment["model"],
            duplicate_policy="diagnostic",
        )
        for key, value in dadabase.pop("arrays").items():
            arrays[f"dadabase__{key}"] = value
        probe_receipts["dadabase"] = dadabase

    trec_corpus_spec = job["diagnostic_sources"]["trec_corpus"]
    trec_queries_spec = job["diagnostic_sources"]["trec_queries"]
    _signature(trec_corpus_spec["canonical_path"], trec_corpus_spec["sha256"], label="R0108 TREC corpus")
    _signature(trec_queries_spec["canonical_path"], trec_queries_spec["sha256"], label="R0108 TREC queries")
    trec = _matched_probe(
        name="trec-covid",
        corpus=np.load(trec_corpus_spec["canonical_path"], mmap_mode="r", allow_pickle=False),
        queries=np.load(trec_queries_spec["canonical_path"], mmap_mode="r", allow_pickle=False),
        control_model=control["model"],
        treatment_model=treatment["model"],
        duplicate_policy="diagnostic",
    )
    for key, value in trec.pop("arrays").items():
        arrays[f"trec__{key}"] = value
    probe_receipts["trec-covid"] = trec

    in_mix_control = np.asarray([
        probe_receipts[language]["cells"]["control_12p5m"]["recall_at_50_of_high10"]
        for language in IN_MIX_LANGUAGES
    ])
    in_mix_treatment = np.asarray([
        probe_receipts[language]["cells"]["treatment_25m"]["recall_at_50_of_high10"]
        for language in IN_MIX_LANGUAGES
    ])
    polish_control = probe_receipts[POLISH]["cells"]["control_12p5m"]
    polish_treatment = probe_receipts[POLISH]["cells"]["treatment_25m"]
    every_probe_cell = [
        cell
        for probe in probe_receipts.values()
        for cell in probe["cells"].values()
    ]
    control_summary = {
        "fineweb_recall_at_50_of_high10": probe_receipts["fineweb-heldout"]["cells"]["control_12p5m"]["recall_at_50_of_high10"],
        "polish_recall_at_50_of_high10": polish_control["recall_at_50_of_high10"],
        "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix_control)),
    }
    treatment_summary = {
        "fineweb_recall_at_50_of_high10": probe_receipts["fineweb-heldout"]["cells"]["treatment_25m"]["recall_at_50_of_high10"],
        "polish_recall_at_50_of_high10": polish_treatment["recall_at_50_of_high10"],
        "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix_treatment)),
    }
    arrays_path = os.path.join(output, "matched-ood-arrays.npz")
    atomic_save_new_npz(arrays_path, immutable=True, **arrays)
    receipt = seal({
        "schema": OOD_SCHEMA,
        "round_id": ROUND_ID,
        "selection": selection_signature,
        "model_lineage": {
            "control_12p5m_train_receipt": control["train_signature"],
            "control_12p5m_production_config": control["config_signature"],
            "control_12p5m_graph": control["graph_signature"],
            "treatment_25m_train_receipt": treatment["train_signature"],
            "treatment_25m_production_config": treatment["config_signature"],
            "treatment_25m_graph": treatment["graph_signature"],
        },
        "control_12p5m": control_summary,
        "treatment_25m": treatment_summary,
        "probes": probe_receipts,
        "checks": {
            "same_queries_corpora_and_truth_for_both_models": True,
            "polish_absent_from_registered_training_inventory": True,
            "every_cell_recall50_at_least_recall10": (
                recall50_at_least_recall10(every_probe_cell)
            ),
            "all_probe_coordinates_finite_noncollapsed": all(
                cell["finite_noncollapsed"] for cell in every_probe_cell
            ),
        },
        "roles": {
            "polish_and_in_mix_recall50": "matched noninferiority gates",
            "fineweb_recall50": "matched noninferiority gate",
            "projection_ffr": "diagnostic-only",
            "trec-covid": "diagnostic-only",
            "dadabase": "diagnostic-only",
        },
        "arrays": expected_input_signature(arrays_path),
        "training_performed": False,
        "universal_ood_claimed": False,
        "wall_seconds": time.monotonic() - started,
    })
    path = os.path.join(output, "matched-ood.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del control["model"], treatment["model"]
    gc.collect()
    return {**receipt, "receipt": expected_input_signature(path)}


def _authenticate_half_train(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    train_path = os.path.join(str(job["train_output"]), "train-receipt.json")
    config_path = os.path.join(str(job["train_output"]), "production-config.json")
    graph_path = str(job["graph_manifest"])
    train = _load_json(train_path)
    config = _load_json(config_path)
    graph = _load_json(graph_path)
    validate_seal(train, label="R0132 train receipt")
    validate_seal(graph, label="R0132 graph manifest")
    graph_signature = _signature(graph_path, label="R0132 graph manifest")
    if (
        train.get("release_sha") != active["manifest"]["release_sha"]
        or graph.get("release_sha") != active["manifest"]["release_sha"]
        or train.get("graph_manifest") != graph_signature
        or (config.get("config") or {}).get("graph", {}).get("manifest")
        != graph_signature
    ):
        raise Round0132Error("R0132 train release/graph lineage changed")
    accounting = validate_train_execution(
        train=train, config_receipt=config, graph=graph
    )
    model = train.get("model") or {}
    _signature(
        str(model.get("canonical_path") or ""),
        str(model.get("sha256") or ""),
        label="R0132 trained model",
    )
    quality_signature = graph.get("quality_admission") or {}
    quality_path = str(quality_signature.get("canonical_path") or "")
    _signature(
        quality_path,
        str(quality_signature.get("sha256") or ""),
        label="R0132 fixed search qualification",
    )
    qualification = _load_json(quality_path)
    validate_seal(qualification, label="R0132 fixed search qualification")
    fixed = qualification.get("fixed_policy") or {}
    if (
        qualification.get("schema") != QUALIFICATION_SCHEMA
        or qualification.get("graph_build_released") is not True
        or qualification.get("outcome")
        != SEARCH_POSITIVE_OUTCOME
        or fixed.get("nprobe") != SEARCH_NPROBE
        or fixed.get("shortlist_width") != SEARCH_SHORTLIST_WIDTH
        or fixed.get("policy_sweep_or_widening_performed") is not False
        or not all((qualification.get("checks") or {}).values())
    ):
        raise Round0132Error("R0132 fixed-search qualification changed")
    return {
        **accounting,
        "train_receipt": _signature(train_path, label="R0132 train receipt"),
        "production_config": _signature(config_path, label="R0132 production config"),
        "graph_manifest": graph_signature,
        "qualification": _signature(
            quality_path, label="R0132 fixed search qualification"
        ),
    }


def _same_float(left: Any, right: Any) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=1e-12, abs_tol=1e-12
        )
    except (TypeError, ValueError):
        return False


def _authenticate_native_selector(native: Mapping[str, Any]) -> dict[str, Any]:
    arrays_spec = native.get("arrays") or {}
    arrays_path = str(arrays_spec.get("canonical_path") or "")
    if _signature(arrays_path, label="R0132 native arrays") != arrays_spec:
        raise Round0132Error("R0132 native array signature changed")
    expected_rows = NATIVE_ANCHORS_PER_GROUP * len(GROUPS)
    with np.load(arrays_path, allow_pickle=False) as arrays:
        high = np.asarray(arrays["high_neighbors_top15"], dtype=np.int64)
        high_radius = np.asarray(arrays["high_radius"], dtype=np.float64)
        control_radius = np.asarray(arrays["control_low_radius"], dtype=np.float64)
        treatment_radius = np.asarray(
            arrays["treatment_low_radius"], dtype=np.float64
        )
        control_low = np.asarray(
            arrays["control_low_neighbors_top50"], dtype=np.int64
        )
        treatment_low = np.asarray(
            arrays["treatment_low_neighbors_top50"], dtype=np.int64
        )
        native_fraction_k_array = np.asarray(arrays["native_fraction_k"])
        control_ffr_hits = np.asarray(arrays["control_ffr_truth_hits"])
        treatment_ffr_hits = np.asarray(arrays["treatment_ffr_truth_hits"])
        family_sizes = np.asarray(arrays["family_sizes"], dtype=np.int64)
        stored_deltas = np.asarray(
            arrays["density_bootstrap_deltas"], dtype=np.float64
        )
    expected_fraction_k = max(
        K_LOW_MAX, int(math.ceil(FRACTION * HALF_RETAINED_ROWS))
    )
    if (
        high.shape != (expected_rows, GRAPH_K)
        or high_radius.shape != (expected_rows,)
        or control_radius.shape != high_radius.shape
        or treatment_radius.shape != high_radius.shape
        or control_low.shape != (expected_rows, K_LOW_MAX)
        or treatment_low.shape != control_low.shape
        or native_fraction_k_array.shape != ()
        or native_fraction_k_array.dtype.kind not in "iu"
        or int(native_fraction_k_array.item()) != expected_fraction_k
        or control_ffr_hits.shape != (expected_rows, K_HIT)
        or treatment_ffr_hits.shape != control_ffr_hits.shape
        or control_ffr_hits.dtype.kind not in "bu"
        or treatment_ffr_hits.dtype.kind not in "bu"
        or np.any((control_ffr_hits != 0) & (control_ffr_hits != 1))
        or np.any((treatment_ffr_hits != 0) & (treatment_ffr_hits != 1))
        or family_sizes.shape != high_radius.shape
        or stored_deltas.shape != (DENSITY_BOOTSTRAP_DRAWS,)
        or np.any(high < 0)
        or np.any(high >= HALF_RETAINED_ROWS)
        or np.any(control_low < 0)
        or np.any(control_low >= HALF_RETAINED_ROWS)
        or np.any(treatment_low < 0)
        or np.any(treatment_low >= HALF_RETAINED_ROWS)
        or np.any(family_sizes < 1)
    ):
        raise Round0132Error("R0132 native selector arrays changed")
    recomputed = paired_density_bootstrap(
        high_radius,
        control_radius,
        treatment_radius,
        eligible=family_sizes < FAMILY_SIZE_CUTOFF,
    )
    recomputed_deltas = np.asarray(
        recomputed.pop("bootstrap_deltas"), dtype=np.float64
    )
    registered = native.get("density_selector") or {}
    scalar_fields = (
        "control_12p5m_density",
        "treatment_25m_density",
        "treatment_minus_control",
        "paired_bootstrap_ci_level",
        "noninferiority_margin",
        "comparison_atol",
    )
    if (
        not np.array_equal(stored_deltas, recomputed_deltas)
        or any(
            not _same_float(registered.get(key), recomputed.get(key))
            for key in scalar_fields
        )
        or registered.get("paired_bootstrap_delta_ci") is None
        or len(registered["paired_bootstrap_delta_ci"]) != 2
        or any(
            not _same_float(left, right)
            for left, right in zip(
                registered["paired_bootstrap_delta_ci"],
                recomputed["paired_bootstrap_delta_ci"],
            )
        )
        or registered.get("paired_bootstrap_draws")
        != recomputed["paired_bootstrap_draws"]
        or registered.get("paired_bootstrap_seed")
        != recomputed["paired_bootstrap_seed"]
        or registered.get("classification") != recomputed["classification"]
    ):
        raise Round0132Error("R0132 density selector does not recompute")
    high10 = high[:, :K_HIT]
    for label, low, ffr_hits, receipt_key in (
        ("control", control_low, control_ffr_hits, "control_12p5m"),
        (
            "treatment",
            treatment_low,
            treatment_ffr_hits,
            NATIVE_TREATMENT_KEY,
        ),
    ):
        receipt = native.get(receipt_key) or {}
        recall10 = recall_from_neighbors(high10, low[:, :K_HIT])
        recall50 = recall_from_neighbors(high10, low[:, :K_LOW_MAX])
        top50_hits = np.any(
            high10[:, :, None] == low[:, None, :K_LOW_MAX], axis=2
        )
        global_ffr = float(np.asarray(ffr_hits, dtype=bool).mean())
        if (
            not _same_float(receipt.get("global_recall_at_10"), recall10)
            or not _same_float(
                receipt.get("global_recall_at_50_of_high10"), recall50
            )
            or not _same_float(receipt.get("global_ffr"), global_ffr)
            or np.any(top50_hits & ~np.asarray(ffr_hits, dtype=bool))
        ):
            raise Round0132Error(
                f"R0132 {label} native recall/FFR does not recompute"
            )
    return {
        "arrays": dict(arrays_spec),
        "density_selector_recomputed": True,
        "density_classification": recomputed["classification"],
        "native_recall10_and_recall50_recomputed": True,
        "native_global_ffr_recomputed_from_per_anchor_evidence": True,
        "native_fraction_k": expected_fraction_k,
        "eligible_density_anchors": int(np.count_nonzero(
            family_sizes < FAMILY_SIZE_CUTOFF
        )),
    }


def _authenticate_ood_metrics(ood: Mapping[str, Any]) -> dict[str, Any]:
    arrays_spec = ood.get("arrays") or {}
    arrays_path = str(arrays_spec.get("canonical_path") or "")
    if _signature(arrays_path, label="R0132 OOD arrays") != arrays_spec:
        raise Round0132Error("R0132 OOD array signature changed")
    probes = ood.get("probes") or {}
    expected = (*IN_MIX_LANGUAGES, POLISH, "fineweb-heldout", "dadabase", "trec-covid")
    if set(probes) != set(expected):
        raise Round0132Error("R0132 OOD probe set changed")
    prefixes = {
        **{language: language for language in (*IN_MIX_LANGUAGES, POLISH)},
        "fineweb-heldout": "fineweb",
        "dadabase": "dadabase",
        "trec-covid": "trec",
    }
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for probe in expected:
            prefix = prefixes[probe]
            truth = np.asarray(
                arrays[f"{prefix}__exact_high_d_top10"], dtype=np.int64
            )
            if truth.ndim != 2 or truth.shape[1] != K_HIT:
                raise Round0132Error(f"R0132 {probe} OOD truth changed")
            for label in ("control_12p5m", "treatment_25m"):
                low = np.asarray(
                    arrays[f"{prefix}__{label}_low_neighbors_top50"],
                    dtype=np.int64,
                )
                cell = (probes[probe].get("cells") or {}).get(label) or {}
                if (
                    low.shape != (len(truth), K_LOW_MAX)
                    or not _same_float(
                        cell.get("recall_at_10"),
                        recall_from_neighbors(truth, low[:, :K_HIT]),
                    )
                    or not _same_float(
                        cell.get("recall_at_50_of_high10"),
                        recall_from_neighbors(truth, low[:, :K_LOW_MAX]),
                    )
                ):
                    raise Round0132Error(
                        f"R0132 {probe} {label} OOD recall does not recompute"
                    )
    in_mix_control = np.asarray([
        probes[language]["cells"]["control_12p5m"]["recall_at_50_of_high10"]
        for language in IN_MIX_LANGUAGES
    ])
    in_mix_treatment = np.asarray([
        probes[language]["cells"]["treatment_25m"]["recall_at_50_of_high10"]
        for language in IN_MIX_LANGUAGES
    ])
    expected_summaries = {
        "control_12p5m": {
            "fineweb_recall_at_50_of_high10": probes["fineweb-heldout"]["cells"]["control_12p5m"]["recall_at_50_of_high10"],
            "polish_recall_at_50_of_high10": probes[POLISH]["cells"]["control_12p5m"]["recall_at_50_of_high10"],
            "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix_control)),
        },
        "treatment_25m": {
            "fineweb_recall_at_50_of_high10": probes["fineweb-heldout"]["cells"]["treatment_25m"]["recall_at_50_of_high10"],
            "polish_recall_at_50_of_high10": probes[POLISH]["cells"]["treatment_25m"]["recall_at_50_of_high10"],
            "in_mix_median_recall_at_50_of_high10": float(np.median(in_mix_treatment)),
        },
    }
    for key, expected_summary in expected_summaries.items():
        observed = ood.get(key) or {}
        if any(
            not _same_float(observed.get(metric), value)
            for metric, value in expected_summary.items()
        ):
            raise Round0132Error(f"R0132 {key} OOD summary changed")
    every_probe_cell = [
        cell
        for probe in probes.values()
        for cell in (probe.get("cells") or {}).values()
    ]
    expected_checks = {
        "same_queries_corpora_and_truth_for_both_models": True,
        "polish_absent_from_registered_training_inventory": True,
        "every_cell_recall50_at_least_recall10": recall50_at_least_recall10(
            every_probe_cell
        ),
        "all_probe_coordinates_finite_noncollapsed": all(
            cell.get("finite_noncollapsed") is True
            for cell in every_probe_cell
        ),
    }
    if ood.get("checks") != expected_checks:
        raise Round0132Error("R0132 OOD validity checks changed")
    return {
        "arrays": dict(arrays_spec),
        "probe_count": len(expected),
        "recall10_and_recall50_recomputed_for_both_models": True,
        "inclusive_recall_order_recomputed_for_every_cell": True,
        "gating_summaries_recomputed": True,
    }


def run_decision(
    active: Mapping[str, Any], job: Mapping[str, Any]
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0132 scale-policy decision"
    )
    native_path = os.path.join(str(job["native_output"]), "matched-native.json")
    ood_path = os.path.join(str(job["ood_output"]), "matched-ood.json")
    native = _load_json(native_path)
    ood = _load_json(ood_path)
    validate_seal(native, label="R0132 native panel")
    validate_seal(ood, label="R0132 OOD panel")
    if (
        native.get("schema") != NATIVE_SCHEMA
        or native.get("round_id") != ROUND_ID
        or ood.get("schema") != OOD_SCHEMA
        or ood.get("round_id") != ROUND_ID
    ):
        raise Round0132Error("R0132 decision inputs changed schema")
    authenticated_train = _authenticate_half_train(active, job)
    authenticated_native = _authenticate_native_selector(native)
    authenticated_ood = _authenticate_ood_metrics(ood)
    native_lineage = native.get("model_lineage") or {}
    ood_lineage = ood.get("model_lineage") or {}
    if (
        native_lineage.get("control_12p5m_train_receipt")
        != authenticated_train["train_receipt"]
        or ood_lineage.get("control_12p5m_train_receipt")
        != authenticated_train["train_receipt"]
        or native_lineage.get("control_12p5m_graph")
        != authenticated_train["graph_manifest"]
        or ood_lineage.get("control_12p5m_graph")
        != authenticated_train["graph_manifest"]
    ):
        raise Round0132Error("R0132 panel/train lineage disagrees")
    quality = noninferiority_checks(
        control_native=native["control_12p5m"],
        treatment_native=native[NATIVE_TREATMENT_KEY],
        control_ood=ood["control_12p5m"],
        treatment_ood=ood["treatment_25m"],
    )
    validity = {
        "native_panel_checks_pass": all(native["checks"].values()),
        "ood_panel_checks_pass": all(ood["checks"].values()),
        "train_accounting_authenticated": all(
            authenticated_train["checks"].values()
        ),
        "coverage_aligned_horizon_computed_from_actual_graph_before_launch": (
            authenticated_train["successful_updates"] > 0
        ),
        "fixed_search_policy_passed_without_widening": bool(
            authenticated_train["qualification"]
        ),
        "same_u12_and_ood_selectors": (
            authenticated_native["density_selector_recomputed"] is True
            and authenticated_ood["gating_summaries_recomputed"] is True
        ),
    }
    decision = scale_policy_decision(
        validity_checks=validity,
        density=native["density_selector"],
        quality=quality,
    )
    receipt = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "native_panel": _signature(native_path, label="R0132 native panel"),
        "ood_panel": _signature(ood_path, label="R0132 OOD panel"),
        "authenticated_train_execution": authenticated_train,
        "authenticated_native_selector": authenticated_native,
        "authenticated_ood_metrics": authenticated_ood,
        **decision,
        "training_performed": True,
        "map_registry_state_changed": False,
        "production_ready": False,
    })
    path = os.path.join(output, "decision.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any], job: dict[str, Any] | None = None
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0132Error("R0132 handler requires its exact round/job")
    action = str(job.get("action") or "")
    handlers = {
        "select_half_subset": run_select_subset,
        "build_half_search_index": run_build_index,
        "qualify_fixed_search": run_qualify_search,
        "build_half_graph_part": run_graph_part,
        "assemble_half_graph": run_assemble_graph,
        "train_half_map": run_train,
        "transform_half_map": run_transform,
        "score_matched_native": run_score_native,
        "score_matched_ood": run_score_ood,
        "decide_scale_policy": run_decision,
    }
    try:
        handler = handlers[action]
    except KeyError as exc:
        raise Round0132Error(f"unknown R0132 action: {action!r}") from exc
    return handler(active, job)
