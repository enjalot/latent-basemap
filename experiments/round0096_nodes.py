"""Build and qualify one balanced-150M IVF32768/PQ48x8 search index."""
from __future__ import annotations

import gc
import json
import os
import tempfile
import time
from functools import partial
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    canonical_json,
    expected_input_signature,
    sha256_bytes,
)
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0049_program import SOURCE_ROWS, global_to_compact
from basemap.round0086_program import validate_substrate
from basemap.round0096_larger_nlist import (
    CORPUS_RANGES,
    DECISION_SCHEMA,
    DIMENSION,
    GLOBAL_MEAN_FLOOR,
    INDEX_SCHEMA,
    K,
    NLIST,
    PER_CORPUS_MEAN_FLOOR,
    POLICY_GRID,
    PQ_BITS,
    PQ_M,
    QUALITY_ROWS,
    QUALITY_SAMPLE_SHA256,
    QUALITY_SEED,
    QUALIFICATION_SCHEMA,
    RETAINED_ROWS,
    ROUND_ID,
    ROW_COUNT,
    SHARD_SCHEMA,
    TEMPLATE_SCHEMA,
    TRAIN_ROWS,
    TRAIN_SAMPLE_SHA256,
    TRAIN_SEED,
    Round0096Error,
    seal,
    select_cell,
)
from experiments.round0049_nodes import (
    _clean_search,
    _exact_representative_truth,
    _exact_rerank_shortlist,
    _sample_retained_rows,
)
from experiments.round0059_nodes import _GpuSearchAdapter, _runtime_stamp
from experiments.round0094_nodes import _peak_rss_gib


INTERVALS = ((0, ROW_COUNT),)
ADD_BATCH_ROWS = 200_000
BENCHMARK_ROWS = 8_192
BENCHMARK_SEED = 97


def _load_sealed(
    path: str,
    *,
    expected_sha256: str,
    schema: str,
    round_id: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0096Error(f"{label} bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    body = {key: item for key, item in value.items()
            if key != "identity_sha256"}
    if (
        value.get("schema") != schema
        or value.get("round_id") != round_id
        or value.get("identity_sha256") != sha256_bytes(
            canonical_json(body)
        )
    ):
        raise Round0096Error(f"{label} seal changed")
    return value, signature


def _load_r0095_review(
    path: str,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0096Error("R0095 review bytes changed")
    text = open(signature["canonical_path"], encoding="utf-8").read()
    if (
        'round_id: "0095"' not in text
        or ("status: accepted" not in text and "status: partial" not in text)
        or "larger-nlist qualification" not in text
        or "minilm-balanced-150m-unbiased-search-audit-v1" not in text
    ):
        raise Round0096Error(
            "R0095 review does not release larger-nlist qualification"
        )
    return signature


def _substrate_arrays(
    job: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.memmap, np.memmap]:
    substrate = validate_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    outputs = substrate["manifest"]["outputs"]
    excluded = np.asarray(
        substrate["eligibility"]["excluded_rows"], dtype=np.int64
    )
    if (
        len(excluded) != ROW_COUNT - RETAINED_ROWS
        or np.any(excluded[1:] <= excluded[:-1])
    ):
        raise Round0096Error("balanced-150M eligibility changed")
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
    return substrate, excluded, encoded, scales


def _normalized_rows(
    encoded: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    values = np.asarray(encoded[rows], dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise Round0096Error("nonfinite or zero retained vector")
    values /= norms
    return np.ascontiguousarray(values)


def _gpu_options() -> Any:
    import faiss

    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    return options


def _write_index_new(index: Any, path: str) -> dict[str, Any]:
    """Publish a FAISS index through a same-directory no-clobber hard link."""
    import faiss

    if os.path.lexists(path):
        raise FileExistsError(f"refuse existing FAISS output: {path}")
    fd, temporary = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        faiss.write_index(index, temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.link(temporary, path, follow_symlinks=False)
        directory_fd = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return expected_input_signature(path)


def _index_geometry(index: Any) -> dict[str, Any]:
    return {
        "class": type(index).__name__,
        "dimension": int(index.d),
        "ntotal": int(index.ntotal),
        "nlist": int(index.nlist),
        "pq_m": int(index.pq.M),
        "pq_bits": int(index.pq.nbits),
        "code_size": int(index.code_size),
        "metric_type": int(index.metric_type),
        "is_trained": bool(index.is_trained),
    }


def _require_geometry(index: Any, *, ntotal: int) -> None:
    import faiss

    geometry = _index_geometry(index)
    if geometry != {
        "class": "IndexIVFPQ",
        "dimension": DIMENSION,
        "ntotal": ntotal,
        "nlist": NLIST,
        "pq_m": PQ_M,
        "pq_bits": PQ_BITS,
        "code_size": PQ_M,
        "metric_type": int(faiss.METRIC_INNER_PRODUCT),
        "is_trained": True,
    }:
        raise Round0096Error(f"unexpected IVF-PQ geometry: {geometry}")


def run_train_template(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0096 trained-index template",
    )
    started = time.monotonic()
    substrate, excluded, encoded, _scales = _substrate_arrays(job)
    runtime = _runtime_stamp(
        str(job["runtime_spec"]), str(job["runtime_spec_sha256"])
    )
    review = _load_r0095_review(
        str(job["r0095_review"]),
        expected_sha256=str(job["r0095_review_sha256"]),
    )
    sample = _sample_retained_rows(
        excluded,
        count=TRAIN_ROWS,
        seed=TRAIN_SEED,
        row_count=ROW_COUNT,
    )
    sample_sha = sha256_bytes(sample.tobytes())
    if sample_sha != TRAIN_SAMPLE_SHA256:
        raise Round0096Error("training sample changed")
    vectors = _normalized_rows(encoded, sample)
    cpu = faiss.IndexIVFPQ(
        faiss.IndexFlatIP(DIMENSION),
        DIMENSION,
        NLIST,
        PQ_M,
        PQ_BITS,
        faiss.METRIC_INNER_PRODUCT,
    )
    resource = faiss.StandardGpuResources()
    resource.setTempMemory(1 << 30)
    gpu = faiss.index_cpu_to_gpu(resource, 0, cpu, _gpu_options())
    gpu.cp.seed = TRAIN_SEED
    gpu.cp.niter = 25
    gpu.cp.spherical = True
    gpu.pq.cp.seed = TRAIN_SEED
    gpu.pq.cp.niter = 25
    train_started = time.monotonic()
    gpu.train(vectors)
    train_seconds = time.monotonic() - train_started
    trained = faiss.index_gpu_to_cpu(gpu)
    _require_geometry(trained, ntotal=0)
    template_path = os.path.join(output, "ivf32768-pq48x8-template.ivfpq")
    template = _write_index_new(trained, template_path)
    corpus_counts = {
        name: int(((sample >= start) & (sample < stop)).sum())
        for name, (start, stop, _retained) in CORPUS_RANGES.items()
    }
    receipt = seal({
        "schema": TEMPLATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "r0095_review": review,
        "runtime": runtime,
        "template": template,
        "geometry": _index_geometry(trained),
        "clustering": {
            "seed": TRAIN_SEED,
            "coarse_iterations": 25,
            "coarse_spherical": True,
            "pq_iterations": 25,
            "training_rows_per_coarse_centroid": TRAIN_ROWS // NLIST,
        },
        "training_sample": {
            "method": "uniform retained rows; random subset before final sort",
            "seed": TRAIN_SEED,
            "rows": len(sample),
            "sha256": sample_sha,
            "minimum_row_id": int(sample.min()),
            "maximum_row_id": int(sample.max()),
            "corpus_counts": corpus_counts,
        },
        "performance": {
            "train_seconds": train_seconds,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "training_performed": False,
        "optimizer_updates": 0,
    })
    receipt_path = os.path.join(output, "template-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _retained_batch(
    excluded: np.ndarray,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    rows = np.arange(start, stop, dtype=np.int64)
    left = int(np.searchsorted(excluded, start, side="left"))
    right = int(np.searchsorted(excluded, stop, side="left"))
    if right > left:
        positions = excluded[left:right] - start
        keep = np.ones(stop - start, dtype=bool)
        keep[positions] = False
        rows = rows[keep]
    return rows


def run_build_shard(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    corpus = str(job["corpus"])
    if corpus not in CORPUS_RANGES:
        raise Round0096Error("unknown corpus shard")
    start, stop, expected_retained = CORPUS_RANGES[corpus]
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0096 {corpus} index shard",
    )
    started = time.monotonic()
    substrate, excluded, encoded, _scales = _substrate_arrays(job)
    template_receipt_path = str(job["template_receipt"])
    template_receipt_observed = expected_input_signature(
        template_receipt_path
    )
    template_receipt, template_receipt_signature = _load_sealed(
        template_receipt_path,
        expected_sha256=template_receipt_observed["sha256"],
        schema=TEMPLATE_SCHEMA,
        round_id=ROUND_ID,
        label="R0096 template receipt",
    )
    template = expected_input_signature(str(job["template_index"]))
    if template_receipt.get("template") != template:
        raise Round0096Error("R0096 template lineage changed")
    cpu = faiss.read_index(template["canonical_path"])
    _require_geometry(cpu, ntotal=0)
    resource = faiss.StandardGpuResources()
    resource.setTempMemory(1 << 30)
    gpu = faiss.index_cpu_to_gpu(resource, 0, cpu, _gpu_options())
    added = 0
    add_started = time.monotonic()
    for batch_start in range(start, stop, ADD_BATCH_ROWS):
        batch_stop = min(stop, batch_start + ADD_BATCH_ROWS)
        rows = _retained_batch(
            excluded, start=batch_start, stop=batch_stop,
        )
        if len(rows):
            gpu.add_with_ids(_normalized_rows(encoded, rows), rows)
            added += len(rows)
    add_seconds = time.monotonic() - add_started
    if added != expected_retained or int(gpu.ntotal) != expected_retained:
        raise Round0096Error(
            f"{corpus} retained count changed: {added}"
        )
    shard = faiss.index_gpu_to_cpu(gpu)
    _require_geometry(shard, ntotal=expected_retained)
    index_path = os.path.join(output, f"{corpus}.ivfpq")
    index_signature = _write_index_new(shard, index_path)
    receipt = seal({
        "schema": SHARD_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "corpus": corpus,
        "start": start,
        "stop": stop,
        "retained_rows": expected_retained,
        "excluded_rows": stop - start - expected_retained,
        "substrate": substrate["signature"],
        "template_receipt": template_receipt_signature,
        "template": template,
        "index": index_signature,
        "geometry": _index_geometry(shard),
        "global_ids_preserved": True,
        "performance": {
            "add_seconds": add_seconds,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "training_performed": False,
        "optimizer_updates": 0,
    })
    receipt_path = os.path.join(output, "shard-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _validate_complete_ids(
    index: Any,
    excluded: np.ndarray,
) -> dict[str, Any]:
    import faiss

    seen = np.zeros(ROW_COUNT, dtype=bool)
    list_sizes = np.empty(NLIST, dtype=np.int64)
    for list_id in range(NLIST):
        size = int(index.invlists.list_size(list_id))
        list_sizes[list_id] = size
        if not size:
            continue
        ids = np.array(
            faiss.rev_swig_ptr(index.invlists.get_ids(list_id), size),
            dtype=np.int64,
            copy=True,
        )
        if (
            np.any(ids < 0)
            or np.any(ids >= ROW_COUNT)
            or len(np.unique(ids)) != size
            or np.any(seen[ids])
        ):
            raise Round0096Error("assembled index IDs are invalid/duplicated")
        seen[ids] = True
    if (
        int(seen.sum()) != RETAINED_ROWS
        or np.any(seen[excluded])
        or int(list_sizes.sum()) != RETAINED_ROWS
    ):
        raise Round0096Error("assembled index does not cover retained IDs")
    if int((~seen).sum()) != len(excluded):
        raise Round0096Error("assembled index missing-row count changed")
    return {
        "list_size_min": int(list_sizes.min()),
        "list_size_mean": float(list_sizes.mean()),
        "list_size_p90": float(np.percentile(list_sizes, 90)),
        "list_size_max": int(list_sizes.max()),
        "seen_retained_rows": int(seen.sum()),
        "excluded_rows_absent": True,
        "global_ids_unique": True,
    }


def run_assemble_index(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0096 assembled larger index",
    )
    started = time.monotonic()
    substrate, excluded, _encoded, _scales = _substrate_arrays(job)
    template_receipt_path = str(job["template_receipt"])
    template_receipt_observed = expected_input_signature(
        template_receipt_path
    )
    template_receipt, template_receipt_signature = _load_sealed(
        template_receipt_path,
        expected_sha256=template_receipt_observed["sha256"],
        schema=TEMPLATE_SCHEMA,
        round_id=ROUND_ID,
        label="R0096 template receipt",
    )
    template = expected_input_signature(str(job["template_index"]))
    if template_receipt.get("template") != template:
        raise Round0096Error("R0096 assembly template changed")
    assembled = faiss.read_index(template["canonical_path"])
    _require_geometry(assembled, ntotal=0)
    shard_receipts: dict[str, Any] = {}
    shard_signatures: dict[str, Any] = {}
    for corpus in CORPUS_RANGES:
        receipt_path = str(job[f"{corpus}_receipt"])
        receipt_observed = expected_input_signature(receipt_path)
        receipt, signature = _load_sealed(
            receipt_path,
            expected_sha256=receipt_observed["sha256"],
            schema=SHARD_SCHEMA,
            round_id=ROUND_ID,
            label=f"R0096 {corpus} shard receipt",
        )
        index_signature = expected_input_signature(
            str(job[f"{corpus}_index"])
        )
        if (
            receipt.get("corpus") != corpus
            or receipt.get("template") != template
            or receipt.get("index") != index_signature
        ):
            raise Round0096Error(f"R0096 {corpus} shard lineage changed")
        shard = faiss.read_index(index_signature["canonical_path"])
        _require_geometry(
            shard, ntotal=CORPUS_RANGES[corpus][2],
        )
        assembled.merge_from(shard, 0)
        shard_receipts[corpus] = signature
        shard_signatures[corpus] = index_signature
        del shard
        gc.collect()
    _require_geometry(assembled, ntotal=RETAINED_ROWS)
    id_validation = _validate_complete_ids(assembled, excluded)
    index_path = os.path.join(
        output, "balanced-150m-retained-ivf32768.ivfpq",
    )
    index_signature = _write_index_new(assembled, index_path)
    receipt = seal({
        "schema": INDEX_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "substrate": substrate["signature"],
        "template_receipt": template_receipt_signature,
        "template": template,
        "shard_receipts": shard_receipts,
        "shards": shard_signatures,
        "index": index_signature,
        "geometry": _index_geometry(assembled),
        "id_validation": id_validation,
        "performance": {
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "training_performed": False,
        "optimizer_updates": 0,
    })
    receipt_path = os.path.join(output, "index-receipt.json")
    atomic_write_new_json(receipt_path, receipt, immutable=True)
    return {**receipt, "receipt": expected_input_signature(receipt_path)}


def _policy_metrics(
    selected: np.ndarray,
    exact: np.ndarray,
    *,
    sample: np.ndarray,
    unambiguous: np.ndarray,
) -> dict[str, Any]:
    overlap = (
        selected[:, :, None] == exact[:, None, :]
    ).any(axis=2).sum(axis=1) / K
    clear = overlap[unambiguous]
    by_corpus: dict[str, Any] = {}
    corpus_passes = []
    for name, (start, stop, _retained) in CORPUS_RANGES.items():
        mask = (sample >= start) & (sample < stop) & unambiguous
        values = overlap[mask]
        mean = float(values.mean()) if len(values) else None
        passed = mean is not None and mean >= PER_CORPUS_MEAN_FLOOR
        corpus_passes.append(passed)
        by_corpus[name] = {
            "unambiguous_rows": int(mask.sum()),
            "mean_recall_at_15_unambiguous": mean,
            "passes_floor": passed,
        }
    mean = float(clear.mean())
    return {
        "mean_recall_at_15": float(overlap.mean()),
        "mean_recall_at_15_unambiguous": mean,
        "p10_recall_at_15_unambiguous": float(
            np.percentile(clear, 10)
        ),
        "passes_global_floor": mean >= GLOBAL_MEAN_FLOOR,
        "passes_every_corpus_floor": all(corpus_passes),
        "by_corpus": by_corpus,
    }


def _search_and_rerank(
    gpu: Any,
    *,
    nprobe: int,
    width: int,
    queries: np.ndarray,
    sample: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    gpu.nprobe = nprobe
    started = time.monotonic()
    _distances, raw = _GpuSearchAdapter(gpu, nprobe).search(
        queries, width + 1,
    )
    search_seconds = time.monotonic() - started
    shortlist, self_seen = _clean_search(
        raw,
        global_sources=sample,
        candidate_count=width,
        source_rows=SOURCE_ROWS,
        global_to_compact_fn=partial(
            global_to_compact, intervals=INTERVALS,
        ),
    )
    selected, rerank = _exact_rerank_shortlist(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
    )
    return selected, {
        "search_seconds": search_seconds,
        "self_returned": self_seen,
        "exact_rerank": rerank,
        "total_seconds": search_seconds + float(rerank["wall_seconds"]),
    }


def _benchmark(
    gpu: Any,
    *,
    nprobe: int,
    width: int,
    queries: np.ndarray,
    sample: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> dict[str, Any]:
    # One warmup and three complete search-plus-exact-rerank measurements.
    _search_and_rerank(
        gpu, nprobe=nprobe, width=width,
        queries=queries[:512], sample=sample[:512], encoded=encoded,
        scales=scales,
    )
    repeats = []
    for _ in range(3):
        _selected, timing = _search_and_rerank(
            gpu, nprobe=nprobe, width=width,
            queries=queries, sample=sample, encoded=encoded,
            scales=scales,
        )
        repeats.append(float(timing["total_seconds"]) / len(queries))
    return {
        "queries": len(queries),
        "warmup_queries": 512,
        "repeats_seconds_per_query": repeats,
        "median_wall_seconds_per_query": float(np.median(repeats)),
    }


def run_qualify_index(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0096 larger-index qualification",
    )
    started = time.monotonic()
    substrate, excluded, encoded, scales = _substrate_arrays(job)
    runtime = _runtime_stamp(
        str(job["runtime_spec"]), str(job["runtime_spec_sha256"])
    )
    review = _load_r0095_review(
        str(job["r0095_review"]),
        expected_sha256=str(job["r0095_review_sha256"]),
    )
    r0095_audit, r0095_audit_signature = _load_sealed(
        str(job["r0095_audit"]),
        expected_sha256=str(job["r0095_audit_sha256"]),
        schema="round0095-balanced-150m-unbiased-search-audit-v1",
        round_id="0095",
        label="R0095 unbiased audit",
    )
    r0095_decision, r0095_decision_signature = _load_sealed(
        str(job["r0095_decision"]),
        expected_sha256=str(job["r0095_decision_sha256"]),
        schema="round0095-balanced-150m-search-correction-decision-v1",
        round_id="0095",
        label="R0095 correction decision",
    )
    if (
        r0095_audit.get("validity_passed") is not True
        or r0095_decision.get("audit") != r0095_audit_signature
        or r0095_decision.get("larger_nlist_qualification_is_next")
        is not True
    ):
        raise Round0096Error("R0095 does not release larger-nlist work")
    index_receipt_path = str(job["index_receipt"])
    index_receipt_observed = expected_input_signature(index_receipt_path)
    index_receipt, index_receipt_signature = _load_sealed(
        index_receipt_path,
        expected_sha256=index_receipt_observed["sha256"],
        schema=INDEX_SCHEMA,
        round_id=ROUND_ID,
        label="R0096 assembled index receipt",
    )
    index_signature = expected_input_signature(str(job["index"]))
    if (
        index_receipt.get("index") != index_signature
        or index_receipt.get("substrate") != substrate["signature"]
    ):
        raise Round0096Error("R0096 assembled index lineage changed")

    sample = _sample_retained_rows(
        excluded,
        count=QUALITY_ROWS,
        seed=QUALITY_SEED,
        row_count=ROW_COUNT,
    )
    sample_sha = sha256_bytes(sample.tobytes())
    if sample_sha != QUALITY_SAMPLE_SHA256:
        raise Round0096Error("R0096 quality sample changed")
    exact, ties, exact_performance = _exact_representative_truth(
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        sample=sample,
        row_count=ROW_COUNT,
    )
    unambiguous = ~ties
    queries = _normalized_rows(encoded, sample)

    cpu = faiss.read_index(index_signature["canonical_path"])
    _require_geometry(cpu, ntotal=RETAINED_ROWS)
    resource = faiss.StandardGpuResources()
    resource.setTempMemory(1 << 30)
    gpu = faiss.index_cpu_to_gpu(resource, 0, cpu, _gpu_options())
    cells: dict[str, Any] = {}
    for nprobe, width in POLICY_GRID:
        selected, execution = _search_and_rerank(
            gpu,
            nprobe=nprobe,
            width=width,
            queries=queries,
            sample=sample,
            encoded=encoded,
            scales=scales,
        )
        cells[f"nprobe-{nprobe}-width-{width}"] = {
            "nprobe": nprobe,
            "shortlist_width": width,
            **_policy_metrics(
                selected,
                exact,
                sample=sample,
                unambiguous=unambiguous,
            ),
            "execution": execution,
            "benchmark": None,
        }

    benchmark_sample = _sample_retained_rows(
        excluded,
        count=BENCHMARK_ROWS,
        seed=BENCHMARK_SEED,
        row_count=ROW_COUNT,
    )
    benchmark_sample_sha = sha256_bytes(benchmark_sample.tobytes())
    benchmark_queries = _normalized_rows(encoded, benchmark_sample)
    for cell in cells.values():
        if (
            cell["passes_global_floor"]
            and cell["passes_every_corpus_floor"]
        ):
            cell["benchmark"] = _benchmark(
                gpu,
                nprobe=int(cell["nprobe"]),
                width=int(cell["shortlist_width"]),
                queries=benchmark_queries,
                sample=benchmark_sample,
                encoded=encoded,
                scales=scales,
            )
    selected = select_cell(cells)
    checks = {
        "quality_sample_sha_matches": sample_sha
        == QUALITY_SAMPLE_SHA256,
        "quality_sample_covers_every_corpus": all(
            np.any((sample >= start) & (sample < stop))
            for start, stop, _retained in CORPUS_RANGES.values()
        ),
        "unambiguous_fraction_at_least_0_90": float(
            unambiguous.mean()
        ) >= 0.90,
        "all_registered_cells_present": set(cells) == {
            f"nprobe-{nprobe}-width-{width}"
            for nprobe, width in POLICY_GRID
        },
        "runtime_matches": all(
            value is True for value in runtime["checks"].values()
        ),
        "no_graph_built": True,
        "no_training_performed": True,
        "no_scale_decision_made": True,
    }
    validity_passed = all(value is True for value in checks.values())
    qualification = seal({
        "schema": QUALIFICATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": validity_passed,
        "failed_checks": sorted(
            key for key, value in checks.items() if value is not True
        ),
        "substrate": substrate["signature"],
        "r0095_review": review,
        "r0095_audit": r0095_audit_signature,
        "r0095_decision": r0095_decision_signature,
        "index_receipt": index_receipt_signature,
        "index": index_signature,
        "geometry": _index_geometry(cpu),
        "quality": {
            "global_mean_floor": GLOBAL_MEAN_FLOOR,
            "per_corpus_mean_floor": PER_CORPUS_MEAN_FLOOR,
            "sample_seed": QUALITY_SEED,
            "sample_rows": len(sample),
            "sample_sha256": sample_sha,
            "corpus_counts": {
                name: int(((sample >= start) & (sample < stop)).sum())
                for name, (start, stop, _retained)
                in CORPUS_RANGES.items()
            },
            "boundary_ties": int(ties.sum()),
            "unambiguous_fraction": float(unambiguous.mean()),
        },
        "benchmark_sample": {
            "seed": BENCHMARK_SEED,
            "rows": len(benchmark_sample),
            "sha256": benchmark_sample_sha,
            "minimum_row_id": int(benchmark_sample.min()),
            "maximum_row_id": int(benchmark_sample.max()),
            "corpus_counts": {
                name: int(
                    (
                        (benchmark_sample >= start)
                        & (benchmark_sample < stop)
                    ).sum()
                )
                for name, (start, stop, _retained)
                in CORPUS_RANGES.items()
            },
        },
        "cells": cells,
        "selected": selected,
        "runtime": runtime,
        "performance": {
            "exact_truth": exact_performance,
            "wall_seconds": time.monotonic() - started,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "checks": checks,
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
    })
    qualification_path = os.path.join(
        output, "ivf32768-policy-qualification.json",
    )
    atomic_write_new_json(
        qualification_path, qualification, immutable=True,
    )
    decision = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": validity_passed,
        "qualification": expected_input_signature(qualification_path),
        "selected": selected,
        "outcome": (
            "qualified"
            if selected is not None
            else "valid-negative-no-registered-cell-passed"
        ),
        "graph_build_released": selected is not None,
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
    })
    decision_path = os.path.join(output, "search-policy-decision.json")
    atomic_write_new_json(decision_path, decision, immutable=True)
    if not validity_passed:
        raise Round0096Error(
            "R0096 qualification invalid: "
            + ", ".join(qualification["failed_checks"])
        )
    return {**decision, "receipt": expected_input_signature(decision_path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0096Error("R0096 handler requires its exact round/job")
    action = job.get("action")
    if action == "train_larger_index_template":
        return run_train_template(active, job)
    if action == "build_larger_index_shard":
        return run_build_shard(active, job)
    if action == "assemble_larger_index":
        return run_assemble_index(active, job)
    if action == "qualify_larger_index":
        return run_qualify_index(active, job)
    raise Round0096Error(f"unknown R0096 action: {action!r}")
