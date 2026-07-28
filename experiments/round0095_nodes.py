"""Replay 150M search policies on the corrected uniform retained-row sample."""
from __future__ import annotations

import gc
import json
import os
from functools import partial
from typing import Any, Mapping

import numpy as np

from basemap.artifact_identity import expected_input_signature, sha256_bytes
from basemap.output_safety import atomic_write_new_json, create_fresh_directory
from basemap.round0049_program import (
    DIMENSION,
    K,
    SOURCE_ROWS,
    global_to_compact,
)
from basemap.round0086_program import validate_substrate
from basemap.round0093_policy import (
    QUALIFICATION_SCHEMA as R0093_QUALIFICATION_SCHEMA,
    load_decision as load_r0093_decision,
)
from basemap.round0094_sharded_search import load_split_receipt
from basemap.round0095_unbiased_audit import (
    AUDIT_SCHEMA,
    CORPUS_RANGES,
    DECISION_SCHEMA,
    MEAN_RECALL_FLOOR,
    MONOLITHIC_POLICIES,
    RETAINED_ROWS,
    ROUND_ID,
    ROW_COUNT,
    SAMPLE_ROWS,
    SAMPLE_SEED,
    SAMPLE_SHA256,
    SHARDED_POLICIES,
    Round0095Error,
    load_r0094_negative,
    sample_corpus_counts,
    seal,
)
from experiments.round0049_nodes import (
    _clean_search,
    _exact_representative_truth,
    _exact_rerank_shortlist,
    _sample_retained_rows,
)
from experiments.round0059_nodes import _GpuSearchAdapter, _runtime_stamp
from experiments.round0094_nodes import (
    _peak_rss_gib,
    _queries,
    _search_and_rerank as _sharded_search_and_rerank,
)


INTERVALS = ((0, ROW_COUNT),)


def _load_r0093_qualification(
    path: str,
    *,
    expected_sha256: str,
    expected_substrate: Mapping[str, Any],
    expected_index: Mapping[str, Any],
) -> dict[str, Any]:
    signature = expected_input_signature(path)
    if signature["sha256"] != expected_sha256:
        raise Round0095Error("R0093 qualification bytes changed")
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        receipt = json.load(handle)
    body = {
        key: value
        for key, value in receipt.items()
        if key != "identity_sha256"
    }
    cells = receipt.get("cells") or {}
    expected = {
        (name, nprobe, width)
        for name, nprobe, width in MONOLITHIC_POLICIES
    }
    registered = {
        (
            name,
            int(cells.get(f"nprobe-{nprobe}-width-{width}", {}).get(
                "nprobe", -1
            )),
            int(cells.get(f"nprobe-{nprobe}-width-{width}", {}).get(
                "shortlist_width", -1
            )),
        )
        for name, nprobe, width in MONOLITHIC_POLICIES
    }
    if (
        receipt.get("schema") != R0093_QUALIFICATION_SCHEMA
        or receipt.get("round_id") != "0093"
        or receipt.get("identity_sha256")
        != sha256_bytes(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode()
        )
        or receipt.get("substrate") != dict(expected_substrate)
        or receipt.get("filtered_index") != dict(expected_index)
        or registered != expected
    ):
        raise Round0095Error("R0093 qualification lineage changed")
    return {"receipt": receipt, "signature": signature}


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
    for name, (start, stop) in CORPUS_RANGES.items():
        mask = (sample >= start) & (sample < stop) & unambiguous
        values = overlap[mask]
        by_corpus[name] = {
            "unambiguous_rows": int(mask.sum()),
            "mean_recall_at_15_unambiguous": (
                float(values.mean()) if len(values) else None
            ),
        }
    mean = float(clear.mean()) if len(clear) else 0.0
    return {
        "mean_recall_at_15": float(overlap.mean()),
        "mean_recall_at_15_unambiguous": mean,
        "p10_recall_at_15_unambiguous": (
            float(np.percentile(clear, 10)) if len(clear) else 0.0
        ),
        "passes_registered_global_mean_floor": (
            mean >= MEAN_RECALL_FLOOR
        ),
        "by_corpus": by_corpus,
    }


def _monolithic_search(
    *,
    gpu: Any,
    nprobe: int,
    width: int,
    queries: np.ndarray,
    sample: np.ndarray,
    encoded: np.ndarray,
    scales: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    gpu.nprobe = nprobe
    adapter = _GpuSearchAdapter(gpu, nprobe)
    _distances, raw = adapter.search(queries, width + 1)
    shortlist, self_seen = _clean_search(
        raw,
        global_sources=sample,
        candidate_count=width,
        source_rows=SOURCE_ROWS,
        global_to_compact_fn=partial(
            global_to_compact,
            intervals=INTERVALS,
        ),
    )
    selected, rerank = _exact_rerank_shortlist(
        queries=queries,
        shortlist=shortlist,
        encoded=encoded,
        scales=scales,
    )
    return selected, {
        "nprobe": nprobe,
        "shortlist_width": width,
        "self_returned": self_seen,
        "exact_rerank": rerank,
    }


def run_audit(
    active: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any]:
    import faiss

    output = create_fresh_directory(
        str(job["outputs"][0]),
        label="R0095 unbiased search audit",
    )
    substrate = validate_substrate(
        str(job["substrate_manifest"]),
        expected_sha256=str(job["substrate_manifest_sha256"]),
    )
    filtered = expected_input_signature(str(job["filtered_index"]))
    if filtered["sha256"] != str(job["filtered_index_sha256"]):
        raise Round0095Error("R0095 filtered index changed")
    r0093_decision = load_r0093_decision(
        str(job["r0093_decision"]),
        expected_sha256=str(job["r0093_decision_sha256"]),
    )
    r0093_qualification = _load_r0093_qualification(
        str(job["r0093_qualification"]),
        expected_sha256=str(job["r0093_qualification_sha256"]),
        expected_substrate=substrate["signature"],
        expected_index=filtered,
    )
    if (
        r0093_decision["receipt"].get("qualification")
        != r0093_qualification["signature"]
    ):
        raise Round0095Error("R0093 decision/qualification changed")
    r0094 = load_r0094_negative(
        str(job["r0094_qualification"]),
        expected_sha256=str(job["r0094_qualification_sha256"]),
    )
    if r0094["receipt"].get("source_index") != filtered:
        raise Round0095Error("R0094 negative does not bind source index")
    split = load_split_receipt(
        str(job["r0094_split_receipt"]),
        expected_source=filtered,
        expected_release_sha=str(r0094["receipt"]["release_sha"]),
    )
    if r0094["receipt"].get("split_receipt") != split["signature"]:
        raise Round0095Error("R0094 negative does not bind index shards")
    runtime = _runtime_stamp(
        str(job["runtime_spec"]),
        str(job["runtime_spec_sha256"]),
    )

    outputs = substrate["manifest"]["outputs"]
    excluded = np.asarray(
        substrate["eligibility"]["excluded_rows"], dtype=np.int64
    )
    if ROW_COUNT - len(excluded) != RETAINED_ROWS:
        raise Round0095Error("R0095 retained universe changed")
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
        count=SAMPLE_ROWS,
        seed=SAMPLE_SEED,
        row_count=ROW_COUNT,
    )
    sample_sha = sha256_bytes(sample.tobytes())
    counts = sample_corpus_counts(sample)
    exact, ties, exact_performance = _exact_representative_truth(
        encoded=encoded,
        scales=scales,
        excluded=excluded,
        sample=sample,
        row_count=ROW_COUNT,
    )
    unambiguous = ~ties
    queries = _queries(encoded, scales, sample)

    options = faiss.GpuClonerOptions()
    options.indicesOptions = faiss.INDICES_64_BIT
    options.useFloat16 = False
    options.usePrecomputed = True
    monolithic_cpu = faiss.read_index(filtered["canonical_path"])
    if (
        type(monolithic_cpu).__name__ != "IndexIVFPQ"
        or int(monolithic_cpu.ntotal) != RETAINED_ROWS
        or int(monolithic_cpu.nlist) != 8_192
        or int(monolithic_cpu.code_size) != 48
    ):
        raise Round0095Error("R0095 monolithic index geometry changed")
    monolithic_resource = faiss.StandardGpuResources()
    monolithic_resource.setTempMemory(1 << 30)
    monolithic_gpu = faiss.index_cpu_to_gpu(
        monolithic_resource,
        0,
        monolithic_cpu,
        options,
    )
    policies: dict[str, Any] = {}
    for name, nprobe, width in MONOLITHIC_POLICIES:
        selected, performance = _monolithic_search(
            gpu=monolithic_gpu,
            nprobe=nprobe,
            width=width,
            queries=queries,
            sample=sample,
            encoded=encoded,
            scales=scales,
        )
        policies[name] = {
            "geometry": "monolithic-ivf8192-pq48x8",
            "nprobe": nprobe,
            "shortlist_width": width,
            **_policy_metrics(
                selected,
                exact,
                sample=sample,
                unambiguous=unambiguous,
            ),
            "execution": performance,
        }
    del monolithic_gpu, monolithic_resource, monolithic_cpu
    gc.collect()

    gpu_resources = []
    gpu_indices = []
    for shard_name in CORPUS_RANGES:
        declared = split["receipt"]["shards"][shard_name]["index"]
        actual = expected_input_signature(declared["canonical_path"])
        if actual != declared:
            raise Round0095Error(f"{shard_name} shard changed")
        cpu = faiss.read_index(actual["canonical_path"])
        gpu_resource = faiss.StandardGpuResources()
        gpu_resource.setTempMemory(1 << 29)
        gpu = faiss.index_cpu_to_gpu(gpu_resource, 0, cpu, options)
        gpu_resources.append(gpu_resource)
        gpu_indices.append(_GpuSearchAdapter(gpu, 96))
        del cpu
    for name, nprobe, width in SHARDED_POLICIES:
        selected, performance = _sharded_search_and_rerank(
            indices=gpu_indices,
            nprobe=nprobe,
            width_per_shard=width,
            queries=queries,
            sources=sample,
            encoded=encoded,
            scales=scales,
        )
        policies[name] = {
            "geometry": "three-corpus-sharded-ivf8192-pq48x8",
            "nprobe_per_shard": nprobe,
            "width_per_shard": width,
            "total_shortlist_width": width * len(CORPUS_RANGES),
            **_policy_metrics(
                selected,
                exact,
                sample=sample,
                unambiguous=unambiguous,
            ),
            "execution": performance,
        }

    checks = {
        "corrected_sample_sha_matches": sample_sha == SAMPLE_SHA256,
        "sample_count_matches": len(sample) == SAMPLE_ROWS,
        "sample_is_unique": len(np.unique(sample)) == SAMPLE_ROWS,
        "every_corpus_is_represented": all(
            value > 0 for value in counts.values()
        ),
        "runtime_matches": all(
            value is True for value in runtime["checks"].values()
        ),
        "unambiguous_fraction_at_least_0_90": (
            float(unambiguous.mean()) >= 0.90
        ),
        "all_registered_replays_present": (
            set(policies)
            == {
                name
                for name, _nprobe, _width in (
                    MONOLITHIC_POLICIES + SHARDED_POLICIES
                )
            }
        ),
        "no_training_performed": True,
        "no_graph_built": True,
        "no_scale_decision_made": True,
    }
    passed = all(value is True for value in checks.values())
    audit = seal({
        "schema": AUDIT_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": passed,
        "failed_checks": sorted(
            key for key, value in checks.items() if value is not True
        ),
        "training_performed": False,
        "optimizer_updates": 0,
        "scale_decision_made": False,
        "substrate": substrate["signature"],
        "filtered_index": filtered,
        "r0093_decision": r0093_decision["signature"],
        "r0093_qualification": r0093_qualification["signature"],
        "r0094_negative_qualification": r0094["signature"],
        "r0094_split_receipt": split["signature"],
        "sample": {
            "method": (
                "uniform without replacement over all retained rows; "
                "random subset before final sort"
            ),
            "seed": SAMPLE_SEED,
            "rows": len(sample),
            "sha256": sample_sha,
            "minimum_row_id": int(sample.min()),
            "maximum_row_id": int(sample.max()),
            "corpus_counts": counts,
            "boundary_ties": int(ties.sum()),
            "unambiguous_fraction": float(unambiguous.mean()),
        },
        "registered_mean_recall_floor": MEAN_RECALL_FLOOR,
        "policies": policies,
        "runtime": runtime,
        "performance": {
            "exact_truth": exact_performance,
            "peak_rss_gib": _peak_rss_gib(),
        },
        "checks": checks,
    })
    audit_path = os.path.join(output, "unbiased-search-audit.json")
    atomic_write_new_json(audit_path, audit, immutable=True)
    if not passed:
        raise Round0095Error(
            "R0095 unbiased audit invalid: "
            + ", ".join(audit["failed_checks"])
        )
    decision = seal({
        "schema": DECISION_SCHEMA,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "validity_passed": True,
        "audit": expected_input_signature(audit_path),
        "r0093_selected_passes_corrected_global_floor": policies[
            "r0093_selected"
        ]["passes_registered_global_mean_floor"],
        "r0093_highest_recall_passes_corrected_global_floor": policies[
            "r0093_highest_recall"
        ]["passes_registered_global_mean_floor"],
        "r0094_strongest_passes_corrected_global_floor": policies[
            "r0094_strongest_registered"
        ]["passes_registered_global_mean_floor"],
        "old_sample_had_zero_pile_rows": True,
        "larger_nlist_qualification_is_next": True,
        "graph_build_remains_blocked": True,
        "training_performed": False,
        "optimizer_updates": 0,
    })
    decision_path = os.path.join(
        output,
        "search-correction-decision.json",
    )
    atomic_write_new_json(decision_path, decision, immutable=True)
    return {**decision, "receipt": expected_input_signature(decision_path)}


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        active.get("manifest", {}).get("round_id") != ROUND_ID
        or job is None
        or job.get("action") != "audit_unbiased_150m_search"
    ):
        raise Round0095Error("R0095 handler requires its exact round/job")
    return run_audit(active, job)
