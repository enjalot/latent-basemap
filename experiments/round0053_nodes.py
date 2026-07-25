"""Fresh-process substrate and quality handlers for matched 30M control."""
from __future__ import annotations

import os
import resource
import time
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import (
    expected_input_signature,
    sha256_bytes,
)
from basemap.int8_eligibility import load_int8_eligibility
from basemap.output_safety import (
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.round0049_program import (
    DIMENSION,
    INDEX_PATH,
    INDEX_SHA256,
    K,
    _seal,
    validate_substrate_manifest,
    write_subset_eligibility,
)
from basemap.round0053_program import (
    EXPECTED_EXCLUDED_ROWS,
    EXPECTED_RETAINED_ROWS,
    GLOBAL_150M_INTERVALS,
    ROUND_ID,
    ROW_COUNT,
    SOURCE_INTERVALS,
    SOURCE_ROWS,
    SOURCE_SUBSTRATE_MANIFEST,
    SUBSTRATE_SCHEMA,
    Round0053Error,
    compact30_to_global150,
    global150_to_compact30,
    validate_control_substrate,
)
from experiments.round0049_nodes import (
    INDEX_SEARCH_WIDTH,
    MEAN_RECALL_FLOOR,
    SEARCH_WIDTH,
    _clean_search,
    _copy_intervals,
    _eligible_selector,
    _exact_rerank_shortlist,
    _exact_representative_truth,
    _membership,
    _sample_retained_rows,
)


QUALITY_RECEIPT_SCHEMA = (
    "round0053-balanced-30m-candidate-quality-v1"
)
QUALITY_SAMPLE_ROWS = 1_024
QUALITY_SEED = 53


def run_build_substrate(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0053 balanced-30M substrate",
    )
    started = time.monotonic()
    source = validate_substrate_manifest(
        SOURCE_SUBSTRATE_MANIFEST,
        expected_sha256=str(job["source_substrate_manifest_sha256"]),
    )
    source_outputs = source["manifest"]["outputs"]
    source_eligibility = load_int8_eligibility(
        source_outputs["eligibility"]["canonical_path"],
        expected_sha256=source_outputs["eligibility"]["sha256"],
        row_count=SOURCE_ROWS,
    )
    int8_path = os.path.join(output, "embeddings.i8")
    scales_path = os.path.join(output, "scales.f16")
    eligibility_path = os.path.join(
        output,
        "minilm-balanced-30m-int8-row-eligibility-v1.npz",
    )
    int8_copy = _copy_intervals(
        source_outputs["int8"]["canonical_path"],
        int8_path,
        row_bytes=DIMENSION,
        intervals=SOURCE_INTERVALS,
    )
    scales_copy = _copy_intervals(
        source_outputs["scales"]["canonical_path"],
        scales_path,
        row_bytes=2,
        intervals=SOURCE_INTERVALS,
    )
    eligibility = write_subset_eligibility(
        eligibility_path,
        source_path=source_outputs["eligibility"]["canonical_path"],
        source_sha256=source_outputs["eligibility"]["sha256"],
        intervals=SOURCE_INTERVALS,
        source_rows=SOURCE_ROWS,
        round_id=ROUND_ID,
        universe="minilm-int8-balanced-30m-matched-control",
        source_input_key="r0049_balanced_60m_eligibility",
    )
    summary = eligibility["metadata"]["summary"]
    if (
        int8_copy["bytes"] != ROW_COUNT * DIMENSION
        or scales_copy["bytes"] != ROW_COUNT * 2
        or summary["excluded_row_count"] != EXPECTED_EXCLUDED_ROWS
        or summary["retained_row_count"] != EXPECTED_RETAINED_ROWS
        or len(source_eligibility["excluded_rows"]) <= (
            EXPECTED_EXCLUDED_ROWS
        )
    ):
        raise Round0053Error(
            "balanced-30M substrate accounting changed"
        )
    body = {
        "schema": SUBSTRATE_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "row_count": ROW_COUNT,
        "dimension": DIMENSION,
        "corpus_order": ["fineweb", "redpajama", "pile"],
        "source_60m_intervals": [
            list(value) for value in SOURCE_INTERVALS
        ],
        "global_150m_intervals": [
            list(value) for value in GLOBAL_150M_INTERVALS
        ],
        "compact_row_policy": (
            "first 10M rows per corpus, compacted in corpus order"
        ),
        "source_60m_substrate": source["signature"],
        "quantization": (
            "byte-identical nested subset of R0049 int8 plus fp16 scale"
        ),
        "exact_family_policy": (
            "recompute representative membership after restriction from "
            "balanced 60M to balanced 30M"
        ),
        "outputs": {
            "int8": int8_copy["signature"],
            "scales": scales_copy["signature"],
            "eligibility": eligibility["signature"],
        },
        "eligibility_summary": summary,
        "matched_fp16_context": {
            "r0040_fp16_retained_rows": 29_781_758,
            "int8_retained_rows": EXPECTED_RETAINED_ROWS,
            "retained_count_delta": EXPECTED_RETAINED_ROWS - 29_781_758,
            "byte_identical_universe_claimed": False,
        },
        "timing": {
            "int8_copy_seconds": int8_copy["wall_seconds"],
            "scales_copy_seconds": scales_copy["wall_seconds"],
            "total_seconds": time.monotonic() - started,
        },
        "peak_rss_gib": (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (1024 ** 2)
        ),
    }
    manifest = _seal(body)
    path = os.path.join(
        output,
        "balanced-30m-int8-substrate-v1.json",
    )
    atomic_write_new_json(path, manifest, immutable=True)
    return {
        **manifest,
        "manifest": expected_input_signature(path),
    }


def run_validate_candidate_quality(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0053 balanced-30M candidate quality",
    )
    nprobe = int(job["nprobe"])
    substrate = validate_control_substrate(
        str(job["substrate_manifest"]),
    )
    outputs = substrate["manifest"]["outputs"]
    eligibility = load_int8_eligibility(
        outputs["eligibility"]["canonical_path"],
        expected_sha256=outputs["eligibility"]["sha256"],
        row_count=ROW_COUNT,
    )
    excluded = np.asarray(
        eligibility["excluded_rows"],
        dtype=np.int64,
    )
    if (
        len(excluded) != EXPECTED_EXCLUDED_ROWS
        or ROW_COUNT - len(excluded) != EXPECTED_RETAINED_ROWS
    ):
        raise Round0053Error("balanced-30M eligibility changed")
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
    selector, selector_keepalive, excluded_global = (
        _eligible_selector(
            excluded,
            intervals=GLOBAL_150M_INTERVALS,
            compact_to_global_fn=compact30_to_global150,
        )
    )
    index_signature = expected_input_signature(INDEX_PATH)
    if index_signature["sha256"] != INDEX_SHA256:
        raise Round0053Error("registered IVF-PQ index changed")
    index = faiss.read_index(
        INDEX_PATH,
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    parameters = faiss.SearchParametersIVF()
    parameters.nprobe = nprobe
    parameters.sel = selector
    queries = (
        np.asarray(encoded[sample], dtype=np.float32)
        * np.asarray(scales[sample], dtype=np.float32)[:, None]
    )
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if (
        not np.isfinite(norms).all()
        or np.any(norms <= 0)
    ):
        raise Round0053Error("balanced-30M quality queries invalid")
    queries /= norms
    search_started = time.monotonic()
    _distances, raw = index.search(
        np.ascontiguousarray(queries),
        INDEX_SEARCH_WIDTH,
        params=parameters,
    )
    search_seconds = time.monotonic() - search_started
    shortlist, self_seen = _clean_search(
        raw,
        global_sources=compact30_to_global150(sample),
        candidate_count=SEARCH_WIDTH,
        global_to_compact_fn=global150_to_compact30,
    )
    selected, rerank_performance = _exact_rerank_shortlist(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
    )
    overlap = (
        selected[:, :, None] == exact[:, None, :]
    ).any(axis=2).sum(axis=1) / K
    unambiguous = ~ties
    if not np.any(unambiguous):
        raise Round0053Error(
            "balanced-30M sample has no unambiguous rows"
        )
    clear_mean = float(overlap[unambiguous].mean())
    clear_p10 = float(np.percentile(overlap[unambiguous], 10))
    checks = {
        "sample_count_is_registered": (
            len(sample) == QUALITY_SAMPLE_ROWS
        ),
        "unambiguous_fraction_at_least_0_90": (
            float(unambiguous.mean()) >= 0.90
        ),
        "mean_recall_at_15_unambiguous_at_least_0_90": (
            clear_mean >= MEAN_RECALL_FLOOR
        ),
        "all_shortlist_candidates_are_subset_representatives": (
            not np.any(_membership(excluded, shortlist))
        ),
        "all_selected_candidates_are_subset_representatives": (
            not np.any(_membership(excluded, selected))
        ),
        "no_training_performed": True,
    }
    passed = all(value is True for value in checks.values())
    body = {
        "schema": QUALITY_RECEIPT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": passed,
        "failed_checks": sorted(
            key for key, value in checks.items()
            if value is not True
        ),
        "training_performed": False,
        "optimizer_updates": 0,
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "index": index_signature,
        "sample": {
            "seed": QUALITY_SEED,
            "rows": len(sample),
            "row_sha256": sha256_bytes(sample.tobytes()),
            "boundary_ties": int(ties.sum()),
            "unambiguous_fraction": float(unambiguous.mean()),
        },
        "candidate_generator": {
            "index_type": "IndexIVFPQ",
            "nprobe": nprobe,
            "search_width": SEARCH_WIDTH,
            "index_search_width": INDEX_SEARCH_WIDTH,
            "selected_neighbors": K,
            "exact_rerank": True,
            "rerank_vector_source": (
                "balanced-subset int8-plus-fp16-scale exact cosine"
            ),
            "native_representative_selector": True,
        },
        "recall": {
            "mean_recall_at_15": float(overlap.mean()),
            "p10_recall_at_15": float(np.percentile(overlap, 10)),
            "mean_recall_at_15_unambiguous": clear_mean,
            "p10_recall_at_15_unambiguous": clear_p10,
            "floor": MEAN_RECALL_FLOOR,
        },
        "self_returned_count": self_seen,
        "checks": checks,
        "performance": {
            "exact_truth": exact_performance,
            "ivfpq_search_seconds": search_seconds,
            "exact_rerank": rerank_performance,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    receipt = _seal(body)
    path = os.path.join(
        output,
        "balanced-30m-candidate-quality-v1.json",
    )
    atomic_write_new_json(path, receipt, immutable=True)
    del selector_keepalive, excluded_global
    if not passed:
        raise Round0053Error(
            "balanced-30M candidate-quality floor failed"
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
        raise Round0053Error("R0053 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    handler = {
        "build_substrate": run_build_substrate,
        "validate_candidate_quality": run_validate_candidate_quality,
    }.get(selected.get("action"))
    if handler is None:
        raise Round0053Error(
            f"unknown R0053 action: {selected.get('action')!r}"
        )
    return handler(active, selected)
