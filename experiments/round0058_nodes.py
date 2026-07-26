"""Bounded nprobe calibration on the reviewed balanced-60M substrate."""
from __future__ import annotations

import json
import os
import resource
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
    QUALITY_RECEIPT_SCHEMA,
    SEARCH_WIDTH,
    _clean_search,
    _eligible_selector,
    _exact_representative_truth,
    _exact_rerank_shortlist,
    _membership,
    _sample_retained_rows,
)


ROUND_ID = "0058"
NPROBES = (32, 40, 48, 56, 64)
RECEIPT_SCHEMA = "round0058-balanced-60m-nprobe-sweep-v1"


class Round0058Error(RuntimeError):
    """The registered balanced-60M nprobe sweep was violated."""


def _select_smallest_passing(rows: Mapping[str, Mapping[str, Any]]) -> int:
    passing = sorted(
        int(probe)
        for probe, row in rows.items()
        if (
            float(row["mean_recall_at_15_unambiguous"])
            >= MEAN_RECALL_FLOOR
        )
    )
    if not passing:
        raise Round0058Error("registered nprobe sweep has no passing arm")
    return passing[0]


def run_sweep(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    probes = tuple(int(value) for value in job["nprobes"])
    if probes != NPROBES:
        raise Round0058Error("nprobe grid changed")
    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0058 balanced-60M nprobe sweep",
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
    sample = _sample_retained_rows(excluded)
    sample_sha256 = sha256_bytes(sample.tobytes())
    exact, ties, exact_performance = _exact_representative_truth(
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        sample=sample,
    )
    unambiguous = ~ties
    if float(unambiguous.mean()) < 0.90:
        raise Round0058Error("exact-truth sample is too ambiguous")

    baseline_signature = expected_input_signature(
        str(job["baseline_quality_receipt"])
    )
    if baseline_signature["sha256"] != str(
        job["baseline_quality_receipt_sha256"]
    ):
        raise Round0058Error("R0049 quality receipt changed")
    with open(
        baseline_signature["canonical_path"],
        encoding="utf-8",
    ) as handle:
        baseline = json.load(handle)
    baseline_body = {
        key: value for key, value in baseline.items()
        if key != "identity_sha256"
    }
    if (
        baseline.get("schema") != QUALITY_RECEIPT_SCHEMA
        or baseline.get("identity_sha256")
        != sha256_bytes(canonical_json(baseline_body))
        or baseline.get("validity_passed") is not True
        or baseline.get("sample", {}).get("row_sha256")
        != sample_sha256
        or baseline.get("candidate_generator", {}).get("nprobe") != 64
        or baseline.get("candidate_generator", {}).get("search_width")
        != SEARCH_WIDTH
    ):
        raise Round0058Error("R0049 baseline policy changed")

    selector, selector_keepalive, excluded_global = _eligible_selector(
        excluded
    )
    index_signature = expected_input_signature(INDEX_PATH)
    if index_signature["sha256"] != INDEX_SHA256:
        raise Round0058Error("registered IVF-PQ index changed")
    index = faiss.read_index(
        INDEX_PATH,
        faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY,
    )
    queries = (
        np.asarray(encoded[sample], dtype=np.float32)
        * np.asarray(scales[sample], dtype=np.float32)[:, None]
    )
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or np.any(norms <= 0):
        raise Round0058Error("quality queries are invalid")
    queries /= norms

    rows: dict[str, dict[str, Any]] = {}
    for probe in probes:
        parameters = faiss.SearchParametersIVF()
        parameters.nprobe = probe
        parameters.sel = selector
        search_started = time.monotonic()
        _distances, raw = index.search(
            np.ascontiguousarray(queries),
            INDEX_SEARCH_WIDTH,
            params=parameters,
        )
        search_seconds = time.monotonic() - search_started
        shortlist, self_seen = _clean_search(
            raw,
            global_sources=compact_to_global(sample),
            candidate_count=SEARCH_WIDTH,
        )
        selected, rerank = _exact_rerank_shortlist(
            queries=queries,
            shortlist=shortlist,
            encoded=encoded,
            scales=scales,
        )
        if (
            np.any(_membership(excluded, shortlist))
            or np.any(_membership(excluded, selected))
        ):
            raise Round0058Error(
                f"nprobe {probe} returned an excluded row"
            )
        overlap = (
            selected[:, :, None] == exact[:, None, :]
        ).any(axis=2).sum(axis=1) / K
        rows[str(probe)] = {
            "mean_recall_at_15": float(overlap.mean()),
            "p10_recall_at_15": float(np.percentile(overlap, 10)),
            "mean_recall_at_15_unambiguous": float(
                overlap[unambiguous].mean()
            ),
            "p10_recall_at_15_unambiguous": float(
                np.percentile(overlap[unambiguous], 10)
            ),
            "passes_mean_floor": bool(
                float(overlap[unambiguous].mean())
                >= MEAN_RECALL_FLOOR
            ),
            "self_returned_count": self_seen,
            "ivfpq_search_seconds": search_seconds,
            "exact_rerank": rerank,
        }

    baseline_mean = float(
        baseline["recall"]["mean_recall_at_15_unambiguous"]
    )
    baseline_p10 = float(
        baseline["recall"]["p10_recall_at_15_unambiguous"]
    )
    reproduction = {
        "sample_sha256_matches": (
            baseline["sample"]["row_sha256"] == sample_sha256
        ),
        "nprobe64_mean_matches": (
            abs(
                rows["64"]["mean_recall_at_15_unambiguous"]
                - baseline_mean
            )
            <= 1e-12
        ),
        "nprobe64_p10_matches": (
            abs(
                rows["64"]["p10_recall_at_15_unambiguous"]
                - baseline_p10
            )
            <= 1e-12
        ),
    }
    selected_probe = _select_smallest_passing(rows)
    checks = {
        **reproduction,
        "registered_grid_complete": (
            tuple(map(int, rows.keys())) == probes
        ),
        "selected_arm_passes_frozen_floor": (
            rows[str(selected_probe)]["passes_mean_floor"] is True
        ),
        "all_rows_are_unambiguous_enough": (
            float(unambiguous.mean()) >= 0.90
        ),
        "no_training_performed": True,
    }
    passed = all(value is True for value in checks.values())
    baseline_search = rows["64"]["ivfpq_search_seconds"]
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
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "index": index_signature,
        "baseline_quality_receipt": baseline_signature,
        "sample": {
            "seed": int(baseline["sample"]["seed"]),
            "rows": len(sample),
            "row_sha256": sample_sha256,
            "boundary_ties": int(ties.sum()),
            "unambiguous_fraction": float(unambiguous.mean()),
        },
        "candidate_generator": {
            "index_type": "IndexIVFPQ",
            "nprobe_grid": list(probes),
            "search_width": SEARCH_WIDTH,
            "index_search_width": INDEX_SEARCH_WIDTH,
            "selected_neighbors": K,
            "exact_rerank": True,
            "rerank_vector_source": (
                "balanced-subset int8-plus-fp16-scale exact cosine"
            ),
            "native_representative_selector": True,
        },
        "floor": MEAN_RECALL_FLOOR,
        "rows_by_nprobe": rows,
        "selected_nprobe": selected_probe,
        "selection_rule": (
            "smallest registered nprobe with mean unambiguous recall@15 "
            "at least 0.90"
        ),
        "relative_search_time_vs_64": {
            key: float(row["ivfpq_search_seconds"]) / baseline_search
            for key, row in rows.items()
        },
        "checks": checks,
        "performance": {
            "exact_truth": exact_performance,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
    }
    receipt = _seal(body)
    path = os.path.join(output, "balanced-60m-nprobe-sweep-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del selector_keepalive, excluded_global
    if not passed:
        raise Round0058Error(
            "nprobe sweep failed baseline reproduction; receipt preserved"
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
        raise Round0058Error("R0058 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "sweep_nprobe":
        raise Round0058Error("R0058 accepts only the nprobe sweep")
    return run_sweep(active, selected)
