"""Qualify GPU IVF-PQ search for the balanced-60M native graph build."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import time
from typing import Any, Mapping

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
from basemap.round0049_program import (
    CORPUS_INTERVALS,
    DIMENSION,
    INDEX_PATH,
    INDEX_SHA256,
    K,
    ROW_COUNT,
    _seal,
    compact_to_global,
    validate_substrate_manifest,
)
from experiments.round0049_nodes import (
    INDEX_SEARCH_WIDTH,
    MEAN_RECALL_FLOOR,
    SEARCH_WIDTH,
    _clean_search,
    _eligible_selector,
    _exact_representative_truth,
    _exact_rerank_shortlist,
    _membership,
    _sample_retained_rows,
    _write_shard,
)
from experiments.round0058_nodes import RECEIPT_SCHEMA as R0058_SCHEMA


ROUND_ID = "0059"
RECEIPT_SCHEMA = "round0059-gpu-ivfpq-qualification-v1"
BENCHMARK_ROWS = 10_000
QUALIFICATION_SHARD_ROWS = 100_000
MIN_ENGINE_OVERLAP = 0.98
MIN_SEARCH_SPEEDUP = 3.0
MAX_PROJECTED_SEARCH_HOURS = 8.0


class Round0059Error(RuntimeError):
    """The registered GPU IVF-PQ qualification was violated."""


def _load_sealed_json(
    path: str,
    *,
    expected_sha256: str,
    schema: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0059Error(f"{schema} bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    body = {
        key: item for key, item in value.items()
        if key != "identity_sha256"
    }
    if (
        value.get("schema") != schema
        or value.get("identity_sha256")
        != sha256_bytes(canonical_json(body))
    ):
        raise Round0059Error(f"{schema} seal is invalid")
    return value, signature


def _selected_nprobe(receipt: Mapping[str, Any]) -> int:
    selected = int(receipt.get("selected_nprobe", -1))
    row = (receipt.get("rows_by_nprobe") or {}).get(str(selected))
    if (
        receipt.get("validity_passed") is not True
        or row is None
        or row.get("passes_mean_floor") is not True
        or float(row.get("mean_recall_at_15_unambiguous", -1.0))
        < MEAN_RECALL_FLOOR
    ):
        raise Round0059Error("R0058 did not release a passing nprobe")
    return selected


def _runtime_stamp(spec_path: str, expected_sha256: str) -> dict[str, Any]:
    signature = expected_input_signature(spec_path)
    if signature["sha256"] != expected_sha256:
        raise Round0059Error("GPU runtime specification changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        spec = json.load(handle)
    if spec.get("schema") != "round0059-gpu-faiss-runtime-v1":
        raise Round0059Error("GPU runtime specification schema changed")

    import faiss
    import torch

    modules = {
        "faiss": faiss.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    distributions = {
        name: importlib.metadata.version(name)
        for name in spec["distributions"]
    }
    binaries = {
        "faiss._swigfaiss": expected_input_signature(
            faiss._swigfaiss.__file__
        ),
        "torch._C": expected_input_signature(torch._C.__file__),
    }
    checks = {
        "python": platform.python_version() == spec["python"],
        "modules": modules == spec["modules"],
        "distributions": distributions == spec["distributions"],
        "faiss_gpu_compiled": (
            hasattr(faiss, "StandardGpuResources")
            and "GPU" in str(faiss.get_compile_options()).split()
        ),
        "faiss_binary": (
            binaries["faiss._swigfaiss"]["sha256"]
            == spec["binary_sha256"]["faiss._swigfaiss"]
        ),
        "torch_binary": (
            binaries["torch._C"]["sha256"]
            == spec["binary_sha256"]["torch._C"]
        ),
        "cuda_available": bool(torch.cuda.is_available()),
        "one_visible_gpu": int(torch.cuda.device_count()) == 1,
    }
    if not all(value is True for value in checks.values()):
        raise Round0059Error(
            "GPU runtime does not match the registered environment: "
            + ", ".join(
                key for key, value in checks.items()
                if value is not True
            )
        )
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise Round0059Error("nvidia-smi runtime receipt failed") from exc
    return {
        "specification": signature,
        "modules": modules,
        "distributions": distributions,
        "binaries": binaries,
        "checks": checks,
        "faiss_compile_options": str(faiss.get_compile_options()),
        "nvidia_smi": smi,
        "python_executable": os.path.realpath(sys.executable),
    }


def _build_filtered_index(
    *,
    faiss: Any,
    source_path: str,
    destination_path: str,
    excluded_global: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    """Copy the three balanced ID ranges, then physically remove exclusions."""
    started = time.monotonic()
    source = faiss.read_index(source_path)
    if (
        source.ntotal != 150_000_000
        or source.d != DIMENSION
        or source.nlist != 8_192
        or source.code_size != 48
        or source.pq.M != 48
        or source.pq.nbits != 8
    ):
        raise Round0059Error("source IVF-PQ geometry changed")
    destination = faiss.clone_index(source)
    destination.reset()
    for start, stop in CORPUS_INTERVALS:
        source.copy_subset_to(
            destination,
            faiss.InvertedLists.SUBSET_TYPE_ID_RANGE,
            start,
            stop,
        )
    copied = int(destination.ntotal)
    if copied != ROW_COUNT:
        raise Round0059Error(
            f"balanced range copy produced {copied} rows, wanted {ROW_COUNT}"
        )
    selector = faiss.IDSelectorBatch(
        np.ascontiguousarray(excluded_global, dtype=np.int64)
    )
    removed = int(destination.remove_ids(selector))
    expected_retained = ROW_COUNT - len(excluded_global)
    if removed != len(excluded_global) or destination.ntotal != expected_retained:
        raise Round0059Error("physical eligibility filtering changed")

    temporary = destination_path + ".partial"
    if os.path.exists(temporary) or os.path.exists(destination_path):
        raise Round0059Error("filtered index output already exists")
    faiss.write_index(destination, temporary)
    os.replace(temporary, destination_path)
    os.chmod(destination_path, 0o444)
    signature = expected_input_signature(destination_path)
    performance = {
        "wall_seconds": time.monotonic() - started,
        "source_ntotal": int(source.ntotal),
        "balanced_range_rows": copied,
        "physically_removed_rows": removed,
        "filtered_ntotal": int(destination.ntotal),
        "nlist": int(destination.nlist),
        "code_size": int(destination.code_size),
        "pq_m": int(destination.pq.M),
        "pq_nbits": int(destination.pq.nbits),
        "index": signature,
    }
    del source, selector
    gc.collect()
    return destination, performance


def _queries(
    encoded: np.ndarray,
    scales: np.ndarray,
    compact_rows: np.ndarray,
) -> np.ndarray:
    query = (
        np.asarray(encoded[compact_rows], dtype=np.float32)
        * np.asarray(scales[compact_rows], dtype=np.float32)[:, None]
    )
    norms = np.linalg.norm(query, axis=1, keepdims=True)
    if (
        not np.isfinite(query).all()
        or not np.isfinite(norms).all()
        or np.any(norms <= 0)
    ):
        raise Round0059Error("benchmark query block is invalid")
    query /= norms
    return np.ascontiguousarray(query)


def _search_and_rerank(
    *,
    index: Any,
    queries: np.ndarray,
    global_sources: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
    params: Any | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.monotonic()
    if params is None:
        _distances, raw = index.search(queries, INDEX_SEARCH_WIDTH)
    else:
        _distances, raw = index.search(
            queries,
            INDEX_SEARCH_WIDTH,
            params=params,
        )
    search_seconds = time.monotonic() - started
    shortlist, self_seen = _clean_search(
        raw,
        global_sources=global_sources,
        candidate_count=SEARCH_WIDTH,
    )
    selected, rerank = _exact_rerank_shortlist(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
    )
    return selected, {
        "search_seconds": search_seconds,
        "queries": len(queries),
        "queries_per_second": len(queries) / search_seconds,
        "self_returned": self_seen,
        "exact_rerank": rerank,
    }


def _overlap(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 2:
        raise Round0059Error("engine comparison geometry changed")
    per_row = (
        left[:, :, None] == right[:, None, :]
    ).any(axis=2).sum(axis=1) / left.shape[1]
    return {
        "mean": float(per_row.mean()),
        "p10": float(np.percentile(per_row, 10)),
        "exact_row_fraction": float(
            np.all(left == right, axis=1).mean()
        ),
    }


def _project_full_graph_hours(
    *,
    row_count: int,
    benchmark_rows: int,
    gpu_search_seconds: float,
    gpu_rerank_seconds: float,
    clone_seconds: float,
    fixed_seconds: float = 600.0,
) -> dict[str, float]:
    """Scale measured search and rerank walls to one complete graph build."""
    values = {
        "row_count": float(row_count),
        "benchmark_rows": float(benchmark_rows),
        "gpu_search_seconds": float(gpu_search_seconds),
        "gpu_rerank_seconds": float(gpu_rerank_seconds),
        "clone_seconds": float(clone_seconds),
        "fixed_seconds": float(fixed_seconds),
    }
    if any(value <= 0.0 for value in values.values()):
        raise Round0059Error("graph projection inputs must be positive")
    scale = values["row_count"] / values["benchmark_rows"]
    search_hours = values["gpu_search_seconds"] * scale / 3600.0
    rerank_hours = values["gpu_rerank_seconds"] * scale / 3600.0
    total_hours = (
        search_hours
        + rerank_hours
        + values["clone_seconds"] / 3600.0
        + values["fixed_seconds"] / 3600.0
    )
    return {
        "search_hours": search_hours,
        "rerank_hours": rerank_hours,
        "total_hours": total_hours,
    }


class _GpuSearchAdapter:
    """Expose the CPU helper's search signature without GPU filter params."""

    def __init__(self, index: Any, nprobe: int) -> None:
        self.index = index
        self.nprobe = nprobe

    def search(
        self,
        queries: np.ndarray,
        width: int,
        *,
        params: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if params is not None and int(params.nprobe) != self.nprobe:
            raise Round0059Error("GPU adapter nprobe changed")
        return self.index.search(queries, width)


def run_qualification(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0059 GPU IVF-PQ qualification",
    )
    source_index_signature = expected_input_signature(INDEX_PATH)
    if source_index_signature["sha256"] != INDEX_SHA256:
        raise Round0059Error("registered source IVF-PQ index changed")
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    substrate = validate_substrate_manifest(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    outputs = substrate["manifest"]["outputs"]
    eligibility = load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    excluded = np.asarray(eligibility["excluded_rows"], dtype=np.int64)
    excluded_global = compact_to_global(excluded)
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
    sweep, sweep_signature = _load_sealed_json(
        str(job["nprobe_receipt"]),
        expected_sha256=str(job["nprobe_receipt_sha256"]),
        schema=R0058_SCHEMA,
    )
    nprobe = _selected_nprobe(sweep)
    if nprobe != int(job["selected_nprobe"]):
        raise Round0059Error("materialized nprobe differs from R0058")

    sample = _sample_retained_rows(excluded)
    exact, ties, exact_performance = _exact_representative_truth(
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        sample=sample,
    )
    unambiguous = ~ties

    filtered_path = os.path.join(
        output,
        "balanced-60m-retained.ivfpq",
    )
    filtered, filter_performance = _build_filtered_index(
        faiss=faiss,
        source_path=INDEX_PATH,
        destination_path=filtered_path,
        excluded_global=excluded_global,
    )

    selector, selector_keepalive, excluded_keepalive = _eligible_selector(
        excluded
    )
    source = faiss.read_index(
        INDEX_PATH,
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    cpu_params = faiss.SearchParametersIVF()
    cpu_params.nprobe = nprobe
    cpu_params.sel = selector
    retained_first_shard = np.arange(
        QUALIFICATION_SHARD_ROWS,
        dtype=np.int64,
    )
    retained_first_shard = retained_first_shard[
        ~_membership(excluded, retained_first_shard)
    ]
    benchmark_rows = retained_first_shard[:BENCHMARK_ROWS]
    benchmark_queries = _queries(encoded, scales, benchmark_rows)
    benchmark_global = compact_to_global(benchmark_rows)
    cpu_selected, cpu_performance = _search_and_rerank(
        index=source,
        queries=benchmark_queries,
        global_sources=benchmark_global,
        encoded=encoded,
        scales=scales,
        params=cpu_params,
    )
    del source, selector, selector_keepalive, excluded_keepalive
    gc.collect()

    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    clone_started = time.monotonic()
    gpu = faiss.index_cpu_to_gpu(resources, 0, filtered, options)
    clone_seconds = time.monotonic() - clone_started
    gpu.nprobe = nprobe
    adapter = _GpuSearchAdapter(gpu, nprobe)

    gpu_selected, gpu_performance = _search_and_rerank(
        index=adapter,
        queries=benchmark_queries,
        global_sources=benchmark_global,
        encoded=encoded,
        scales=scales,
        params=None,
    )
    engine_overlap = _overlap(cpu_selected, gpu_selected)

    sample_queries = _queries(encoded, scales, sample)
    gpu_quality_selected, gpu_quality_performance = _search_and_rerank(
        index=adapter,
        queries=sample_queries,
        global_sources=compact_to_global(sample),
        encoded=encoded,
        scales=scales,
        params=None,
    )
    quality_overlap = (
        gpu_quality_selected[:, :, None] == exact[:, None, :]
    ).any(axis=2).sum(axis=1) / K
    mean_quality = float(quality_overlap[unambiguous].mean())
    p10_quality = float(
        np.percentile(quality_overlap[unambiguous], 10)
    )
    selected_row = sweep["rows_by_nprobe"][str(nprobe)]

    shard_root = os.path.join(output, "qualification-shard")
    os.mkdir(shard_root)
    gpu_params = faiss.SearchParametersIVF()
    gpu_params.nprobe = nprobe
    shard_receipt = _write_shard(
        index=adapter,
        parameters=gpu_params,
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        shard_root=shard_root,
        shard=0,
        start=0,
        stop=QUALIFICATION_SHARD_ROWS,
        nprobe=nprobe,
        round_id=ROUND_ID,
    )

    speedup = (
        float(gpu_performance["queries_per_second"])
        / float(cpu_performance["queries_per_second"])
    )
    projection = _project_full_graph_hours(
        row_count=ROW_COUNT,
        benchmark_rows=len(benchmark_rows),
        gpu_search_seconds=float(gpu_performance["search_seconds"]),
        gpu_rerank_seconds=float(
            gpu_performance["exact_rerank"]["wall_seconds"]
        ),
        clone_seconds=clone_seconds,
    )
    projected_search_hours = projection["search_hours"]
    projected_rerank_hours = projection["rerank_hours"]
    projected_graph_hours = projection["total_hours"]
    checks = {
        "runtime_matches": all(
            value is True
            for value in runtime["checks"].values()
        ),
        "filtered_candidate_count": (
            filter_performance["filtered_ntotal"]
            == ROW_COUNT - len(excluded)
        ),
        "sample_identity_matches_r0058": (
            sha256_bytes(sample.tobytes())
            == sweep["sample"]["row_sha256"]
        ),
        "gpu_mean_recall_floor": mean_quality >= MEAN_RECALL_FLOOR,
        "engine_overlap_floor": (
            engine_overlap["mean"] >= MIN_ENGINE_OVERLAP
        ),
        "search_speedup_floor": speedup >= MIN_SEARCH_SPEEDUP,
        "projected_graph_wall": (
            projected_graph_hours <= MAX_PROJECTED_SEARCH_HOURS
        ),
        "qualification_shard_complete": (
            shard_receipt["retained_sources"]
            == len(retained_first_shard)
        ),
        "no_training_performed": True,
    }
    # Recompute CPU quality on the exact R0058 sample, rather than accepting
    # the benchmark-row shortcut above.
    source = faiss.read_index(
        INDEX_PATH,
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    selector, selector_keepalive, excluded_keepalive = _eligible_selector(
        excluded
    )
    cpu_quality_params = faiss.SearchParametersIVF()
    cpu_quality_params.nprobe = nprobe
    cpu_quality_params.sel = selector
    cpu_quality_selected, cpu_quality_performance = _search_and_rerank(
        index=source,
        queries=sample_queries,
        global_sources=compact_to_global(sample),
        encoded=encoded,
        scales=scales,
        params=cpu_quality_params,
    )
    cpu_quality_overlap = (
        cpu_quality_selected[:, :, None] == exact[:, None, :]
    ).any(axis=2).sum(axis=1) / K
    cpu_mean = float(cpu_quality_overlap[unambiguous].mean())
    cpu_p10 = float(
        np.percentile(cpu_quality_overlap[unambiguous], 10)
    )
    checks["cpu_mean_reproduces_r0058"] = (
        abs(
            cpu_mean
            - float(selected_row["mean_recall_at_15_unambiguous"])
        )
        <= 1e-12
    )
    checks["cpu_p10_reproduces_r0058"] = (
        abs(
            cpu_p10
            - float(selected_row["p10_recall_at_15_unambiguous"])
        )
        <= 1e-12
    )
    checks["gpu_vs_cpu_quality_overlap"] = (
        _overlap(cpu_quality_selected, gpu_quality_selected)["mean"]
        >= MIN_ENGINE_OVERLAP
    )
    del source, selector, selector_keepalive, excluded_keepalive

    passed = all(value is True for value in checks.values())
    body = {
        "schema": RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": passed,
        "failed_checks": sorted(
            key for key, value in checks.items()
            if value is not True
        ),
        "training_performed": False,
        "optimizer_updates": 0,
        "runtime": runtime,
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "source_index": source_index_signature,
        "nprobe_calibration": sweep_signature,
        "selected_nprobe": nprobe,
        "candidate_universe": {
            "balanced_intervals": [list(value) for value in CORPUS_INTERVALS],
            "physical_exclusions": len(excluded_global),
            "retained_rows": ROW_COUNT - len(excluded_global),
            "filtered_index": filter_performance["index"],
        },
        "gpu_index": {
            "implementation": "faiss-classic-GpuIndexIVFPQ",
            "indices_options": "INDICES_64_BIT",
            "use_float16": False,
            "use_precomputed": True,
            "temporary_memory_bytes": 1 << 30,
            "clone_seconds": clone_seconds,
        },
        "quality": {
            "sample_rows": len(sample),
            "sample_sha256": sha256_bytes(sample.tobytes()),
            "boundary_ties": int(ties.sum()),
            "cpu_mean_recall_at_15_unambiguous": cpu_mean,
            "cpu_p10_recall_at_15_unambiguous": cpu_p10,
            "gpu_mean_recall_at_15_unambiguous": mean_quality,
            "gpu_p10_recall_at_15_unambiguous": p10_quality,
            "engine_overlap": _overlap(
                cpu_quality_selected,
                gpu_quality_selected,
            ),
            "floor": MEAN_RECALL_FLOOR,
        },
        "benchmark": {
            "rows": len(benchmark_rows),
            "cpu": cpu_performance,
            "gpu": gpu_performance,
            "engine_overlap": engine_overlap,
            "search_speedup": speedup,
            "projected_full_60m_search_hours": projected_search_hours,
            "projected_full_60m_rerank_hours": projected_rerank_hours,
            "projected_full_60m_graph_hours": projected_graph_hours,
            "minimum_search_speedup": MIN_SEARCH_SPEEDUP,
            "maximum_projected_graph_hours": (
                MAX_PROJECTED_SEARCH_HOURS
            ),
        },
        "qualification_shard": {
            key: value for key, value in shard_receipt.items()
            if key != "resumed"
        },
        "performance": {
            "filtered_index_build": filter_performance,
            "exact_truth": exact_performance,
            "cpu_quality": cpu_quality_performance,
            "gpu_quality": gpu_quality_performance,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
        "checks": checks,
    }
    receipt = _seal(body)
    path = os.path.join(output, "gpu-ivfpq-qualification-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    if not passed:
        raise Round0059Error(
            "GPU IVF-PQ qualification failed: "
            + ", ".join(receipt["failed_checks"])
        )
    return {
        **receipt,
        "receipt": expected_input_signature(path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0059Error("R0059 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "qualify_gpu_ivfpq":
        raise Round0059Error("R0059 accepts only the GPU IVF-PQ qualification")
    return run_qualification(active, selected)
