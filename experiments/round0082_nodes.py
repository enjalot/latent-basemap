"""Confirm the selected balanced-120M IVF-PQ policy on a fresh holdout."""
from __future__ import annotations

import os
import resource
import time
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import expected_input_signature, sha256_bytes
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0049_program import (
    DIMENSION,
    K,
)
from basemap.round0065_substrates import subset_spec, validate_scale_substrate
from basemap.round0081_quality import load_gpu_policy_qualification
from basemap.round0082_quality import (
    CONFIRMATION_SCHEMA,
    EXPECTED_NPROBE,
    EXPECTED_SHORTLIST_WIDTH,
    MEAN_RECALL_FLOOR,
    ROUND_ID,
    SOURCE_QUALIFICATION_IDENTITY,
    SOURCE_QUALIFICATION_SHA256,
    Round0082Error,
    seal,
)
from experiments.round0049_nodes import (
    _exact_representative_truth,
    _sample_retained_rows,
)
from experiments.round0059_nodes import _GpuSearchAdapter, _runtime_stamp
from experiments.round0081_nodes import _queries, _search_and_rerank


TIER = "120m"
SPEC = subset_spec(TIER)
ROW_COUNT = int(SPEC["row_count"])
INTERVALS = tuple(
    (int(start), int(stop))
    for start, stop in SPEC["intervals"]
)
ELIGIBILITY_SUMMARY = dict(SPEC["eligibility_summary"])
QUALITY_SAMPLE_ROWS = 8_192
QUALITY_SEED = 82
SOURCE_SAMPLE_ROWS = 4_096
SOURCE_SAMPLE_SEED = 81
RUNTIME_SPEC = os.path.join(
    os.path.dirname(__file__),
    "round0060_runtime.json",
)


def run_confirmation(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss
    import torch

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="Round 0082 balanced-120M policy holdout",
    )
    substrate = validate_scale_substrate(
        str(job["substrate_manifest"]),
        tier=TIER,
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    outputs = substrate["manifest"]["outputs"]
    eligibility = substrate["eligibility"]
    excluded = np.asarray(eligibility["excluded_rows"], dtype=np.int64)
    expected_excluded = int(ELIGIBILITY_SUMMARY["excluded_row_count"])
    expected_retained = int(ELIGIBILITY_SUMMARY["retained_row_count"])
    if (
        len(excluded) != expected_excluded
        or ROW_COUNT - len(excluded) != expected_retained
    ):
        raise Round0082Error("balanced-120M eligibility accounting changed")

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
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )
    filtered_signature = expected_input_signature(str(job["filtered_index"]))
    if filtered_signature["sha256"] != str(job["filtered_index_sha256"]):
        raise Round0082Error("reviewed R0077 filtered index changed")
    qualification_signature = expected_input_signature(
        str(job["source_qualification"])
    )
    if (
        qualification_signature["sha256"]
        != SOURCE_QUALIFICATION_SHA256
        or str(job["source_qualification_sha256"])
        != SOURCE_QUALIFICATION_SHA256
    ):
        raise Round0082Error("reviewed R0081 qualification changed")
    qualification = load_gpu_policy_qualification(
        qualification_signature["canonical_path"],
        expected_sha256=SOURCE_QUALIFICATION_SHA256,
        substrate_signature=substrate["signature"],
        eligibility_signature=outputs["eligibility"],
        filtered_index_signature=filtered_signature,
    )["receipt"]
    selected_policy = qualification.get("selected") or {}
    if (
        qualification.get("identity_sha256")
        != SOURCE_QUALIFICATION_IDENTITY
        or int(selected_policy.get("nprobe", -1)) != EXPECTED_NPROBE
        or int(selected_policy.get("shortlist_width", -1))
        != EXPECTED_SHORTLIST_WIDTH
    ):
        raise Round0082Error("R0081 selected policy changed")

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
    unambiguous = ~ties
    sample_queries = _queries(encoded, scales, sample)
    # Exact truth is complete before Faiss clone. Release its cached blocks so
    # the larger holdout and the 6.6 GB GPU index do not overlap needlessly.
    torch.cuda.empty_cache()

    filtered = faiss.read_index(filtered_signature["canonical_path"])
    if (
        type(filtered).__name__ != "IndexIVFPQ"
        or int(filtered.ntotal) != expected_retained
        or int(filtered.d) != DIMENSION
        or int(filtered.nlist) != 8_192
        or int(filtered.code_size) != 48
        or int(filtered.pq.M) != 48
        or int(filtered.pq.nbits) != 8
    ):
        raise Round0082Error("reviewed 120M filtered index geometry changed")

    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    clone_started = time.monotonic()
    gpu = faiss.index_cpu_to_gpu(resources, 0, filtered, options)
    clone_seconds = time.monotonic() - clone_started
    gpu.nprobe = EXPECTED_NPROBE
    selected, policy_performance = _search_and_rerank(
        index=_GpuSearchAdapter(gpu, EXPECTED_NPROBE),
        nprobe=EXPECTED_NPROBE,
        shortlist_width=EXPECTED_SHORTLIST_WIDTH,
        queries=sample_queries,
        compact_sources=sample,
        encoded=encoded,
        scales=scales,
    )
    overlap = (
        selected[:, :, None] == exact[:, None, :]
    ).any(axis=2).sum(axis=1) / K
    clear = overlap[unambiguous]
    clear_mean = float(clear.mean()) if len(clear) else 0.0
    clear_std = (
        float(clear.std(ddof=1))
        if len(clear) > 1
        else 0.0
    )
    clear_se = (
        float(clear_std / np.sqrt(len(clear)))
        if len(clear)
        else float("inf")
    )
    lower_95_one_sided = (
        float(clear_mean - NormalDist().inv_cdf(0.95) * clear_se)
        if len(clear)
        else float("-inf")
    )
    source_mean = float(
        selected_policy["mean_recall_at_15_unambiguous"]
    )
    pooled_mean = float((
        source_mean * SOURCE_SAMPLE_ROWS
        + clear_mean * len(clear)
    ) / (SOURCE_SAMPLE_ROWS + len(clear)))

    checks = {
        "runtime_matches": all(
            value is True for value in runtime["checks"].values()
        ),
        "fixed_registered_120m_universe": (
            substrate["manifest"]["tier"] == TIER
            and substrate["manifest"]["row_count"] == ROW_COUNT
        ),
        "filtered_candidate_count": int(filtered.ntotal) == expected_retained,
        "source_qualification_authenticated": (
            qualification["identity_sha256"]
            == SOURCE_QUALIFICATION_IDENTITY
        ),
        "selected_policy_unchanged": (
            int(selected_policy["nprobe"]) == EXPECTED_NPROBE
            and int(selected_policy["shortlist_width"])
            == EXPECTED_SHORTLIST_WIDTH
        ),
        "fresh_sample_is_distinct": (
            QUALITY_SEED != SOURCE_SAMPLE_SEED
            and sha256_bytes(sample.tobytes())
            != str(qualification["quality"]["sample_sha256"])
        ),
        "unambiguous_fraction_at_least_0_90": (
            float(unambiguous.mean()) >= 0.90
        ),
        "fresh_mean_recall_at_least_0_90": (
            clear_mean >= MEAN_RECALL_FLOOR
        ),
        "no_training_performed": True,
        "no_scale_decision_made": True,
    }
    passed = all(value is True for value in checks.values())
    body = {
        "schema": CONFIRMATION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": passed,
        "failed_checks": sorted(
            key
            for key, value in checks.items()
            if value is not True
        ),
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
        "tier": TIER,
        "substrate": substrate["signature"],
        "eligibility": outputs["eligibility"],
        "filtered_index": filtered_signature,
        "source_qualification": qualification_signature,
        "source_qualification_identity": SOURCE_QUALIFICATION_IDENTITY,
        "selected_policy": {
            "nprobe": EXPECTED_NPROBE,
            "shortlist_width": EXPECTED_SHORTLIST_WIDTH,
            "selected_neighbors": K,
            "exact_rerank": True,
        },
        "quality": {
            "sample_rows": len(sample),
            "sample_seed": QUALITY_SEED,
            "sample_sha256": sha256_bytes(sample.tobytes()),
            "boundary_ties": int(ties.sum()),
            "unambiguous_fraction": float(unambiguous.mean()),
            "mean_recall_at_15": float(overlap.mean()),
            "p10_recall_at_15": float(np.percentile(overlap, 10)),
            "mean_recall_at_15_unambiguous": clear_mean,
            "p10_recall_at_15_unambiguous": (
                float(np.percentile(clear, 10))
                if len(clear)
                else 0.0
            ),
            "sample_standard_deviation": clear_std,
            "mean_standard_error": clear_se,
            "normal_one_sided_95_lower_diagnostic": lower_95_one_sided,
            "floor": MEAN_RECALL_FLOOR,
            "source_sample_rows": SOURCE_SAMPLE_ROWS,
            "source_sample_seed": SOURCE_SAMPLE_SEED,
            "source_mean_recall_at_15_unambiguous": source_mean,
            "fresh_minus_source_mean": clear_mean - source_mean,
            "pooled_mean_recall_at_15_unambiguous": pooled_mean,
        },
        "performance": {
            "exact_truth": exact_performance,
            "selected_policy": policy_performance,
            "gpu_clone_seconds": clone_seconds,
            "peak_rss_gib": (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 ** 2)
            ),
        },
        "runtime": runtime,
        "checks": checks,
    }
    receipt = seal(body)
    path = os.path.join(output, "gpu-ivfpq-policy-confirmation-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    del resources, gpu, filtered
    if not passed:
        raise Round0082Error(
            "balanced-120M policy holdout failed: "
            + ", ".join(receipt["failed_checks"])
        )
    return {**receipt, "receipt": expected_input_signature(path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0082Error("R0082 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if selected.get("action") != "confirm_balanced_120m_gpu_ivfpq_policy":
        raise Round0082Error("R0082 accepts only policy confirmation")
    return run_confirmation(active, selected)
