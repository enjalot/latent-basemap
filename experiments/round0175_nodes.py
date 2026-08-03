"""Execute the R0175 approximate-UMAP out-of-sample baseline."""
from __future__ import annotations

import gc
import json
import os
import resource
import subprocess
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from basemap.artifact_identity import expected_input_signature
from basemap.output_safety import (
    atomic_save_new_npy,
    atomic_write_new_json,
    create_fresh_directory,
)
from basemap.panel_v2 import _ids_hash, load_coords, load_embeddings
from basemap.round0108_evaluation import seal, validate_seal
from basemap.round0175_aumap_baseline import (
    CAPABILITY,
    FRAC,
    HELD_HASHES,
    K_HIT,
    NEIGHBORS,
    N_QUERIES,
    ROUND_ID,
    ROWS,
    SEED,
    SCALES,
    Round0175Error,
    build_synthesis,
    project_coordinates,
    projection_metrics,
)


SOURCE_ROOT = (
    "/data/embeddings/fineweb-edu-sample-10BT-chunked-500-"
    "jina-v5-nano/train"
)
TOOLCHAIN_PYTHON = (
    "/data/latent-basemap/toolchains/aumap-v0.2.0-py312/bin/python"
)
OFFICIAL_PROBE = os.path.join(
    os.path.dirname(__file__), "round0175_official_probe.py"
)
TESTBED_ROOTS = {
    scale: f"/data/latent-basemap/jina-en-{scale}" for scale in SCALES
}


def _signature(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    actual = expected_input_signature(str(expected.get("canonical_path") or ""))
    if actual != dict(expected):
        raise Round0175Error(f"{label} bytes changed")
    return actual


def _read_sealed(expected: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    signature = _signature(expected, label=label)
    with open(signature["canonical_path"], encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise Round0175Error(f"{label} is not an object")
    validate_seal(value, label=label)
    return value


def run_official_probe(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0175 official aUMAP probe"
    )
    raw_path = os.path.join(output, "official-probe-raw.json")
    started = time.monotonic()
    completed = subprocess.run(
        [TOOLCHAIN_PYTHON, OFFICIAL_PROBE, "--output", raw_path],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    if completed.returncode != 0:
        raise Round0175Error(
            "official approx-umap probe failed: "
            f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
        )
    with open(raw_path, encoding="utf-8") as handle:
        probe = json.load(handle)
    if (
        not isinstance(probe, dict)
        or probe.get("package") != "approx-umap==0.2.0"
        or probe.get("passed") is not True
        or float(probe.get("max_abs_error", 1.0)) > 1.0e-6
    ):
        raise Round0175Error("official approx-umap probe contract changed")
    os.chmod(raw_path, 0o444)
    receipt = seal({
        **probe,
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "raw_probe": expected_input_signature(raw_path),
        "toolchain_python": job["toolchain_python"],
        "package_files": job["package_files"],
        "probe_source": job["probe_source"],
        "cuda_visible_devices": "",
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "wall_seconds": time.monotonic() - started,
    })
    atomic_write_new_json(
        os.path.join(output, "official-probe.json"), receipt, immutable=True
    )


def _heldout_queries(scale: str) -> tuple[Any, np.ndarray, np.ndarray]:
    source = load_embeddings(SOURCE_ROOT, dim=768)
    sample_ids = np.load(
        os.path.join(TESTBED_ROOTS[scale], "sample_indices.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    if sample_ids.shape != (ROWS[scale],) or sample_ids.dtype.kind not in "iu":
        raise Round0175Error(f"R0175 {scale} sample-index shape changed")
    canonical = np.asarray(sorted(set(int(item) for item in sample_ids)), dtype=np.int64)
    if len(canonical) != ROWS[scale] or canonical.min() < 0 or canonical.max() >= len(source):
        raise Round0175Error(f"R0175 {scale} sample-index universe changed")
    candidates = np.setdiff1d(
        np.arange(len(source), dtype=np.int64), canonical, assume_unique=True
    )
    held = np.sort(
        np.random.RandomState(SEED).choice(candidates, N_QUERIES, replace=False)
    ).astype(np.int64, copy=False)
    if _ids_hash(held) != HELD_HASHES[scale]:
        raise Round0175Error(f"R0175 {scale} held-out selection changed")
    queries = np.asarray(source[held], dtype=np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    if not np.isfinite(queries).all() or np.any(norms <= 0):
        raise Round0175Error(f"R0175 {scale} held-out embeddings are invalid")
    queries /= norms
    return source, held, queries


def _exact_cosine_neighbors(
    corpus: np.ndarray, queries: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import faiss

    started = time.monotonic()
    resources = faiss.StandardGpuResources()
    resources.setTempMemory(1 << 30)
    options = faiss.GpuClonerOptions()
    options.useFloat16 = False
    index = faiss.index_cpu_to_gpu(
        resources, 0, faiss.IndexFlatIP(corpus.shape[1]), options
    )
    stage = time.monotonic()
    for start in range(0, len(corpus), 100_000):
        block = np.asarray(corpus[start : min(start + 100_000, len(corpus))], dtype=np.float32)
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        if not np.isfinite(block).all() or np.any(norms <= 0):
            raise Round0175Error("teacher corpus embeddings are invalid")
        block /= norms
        index.add(np.ascontiguousarray(block))
    add_seconds = time.monotonic() - stage
    if int(index.ntotal) != len(corpus):
        raise Round0175Error("exact cosine index row count changed")
    search_k = NEIGHBORS + 1
    scores = np.empty((len(queries), search_k), dtype=np.float32)
    ids = np.empty((len(queries), search_k), dtype=np.int64)
    stage = time.monotonic()
    for start in range(0, len(queries), 2_048):
        stop = min(start + 2_048, len(queries))
        observed_scores, observed_ids = index.search(
            np.ascontiguousarray(queries[start:stop]), search_k
        )
        for row in range(stop - start):
            order = np.lexsort((observed_ids[row], -observed_scores[row]))
            scores[start + row] = observed_scores[row, order]
            ids[start + row] = observed_ids[row, order]
    search_seconds = time.monotonic() - stage
    if np.any(ids < 0) or np.any(ids >= len(corpus)) or not np.isfinite(scores).all():
        raise Round0175Error("exact cosine search returned invalid neighbors")
    boundary_gap = scores[:, NEIGHBORS - 1] - scores[:, NEIGHBORS]
    selected_ids = ids[:, :NEIGHBORS].astype(np.int32, copy=False)
    distances = np.maximum(0.0, 1.0 - scores[:, :NEIGHBORS]).astype(
        np.float32, copy=False
    )
    performance = {
        "index_add_seconds": add_seconds,
        "query_seconds": search_seconds,
        "queries_per_second": len(queries) / search_seconds,
        "total_seconds": time.monotonic() - started,
        "search_k_for_boundary_guard": search_k,
        "minimum_rank15_to_rank16_similarity_gap": float(boundary_gap.min()),
        "zero_rank15_to_rank16_gap_rows": int(np.count_nonzero(boundary_gap == 0)),
        "index": "GPU IndexFlatIP exact fp32",
        "tie_order_within_returned_candidates": "similarity descending then global ID ascending",
    }
    del index, resources, ids, scores
    gc.collect()
    return selected_ids, distances, performance


def _exact_low_neighbors(
    teacher: np.ndarray, projected: np.ndarray, k: int
) -> np.ndarray:
    import faiss

    resources = faiss.StandardGpuResources()
    resources.setTempMemory(512 << 20)
    options = faiss.GpuClonerOptions()
    options.useFloat16 = False
    index = faiss.index_cpu_to_gpu(resources, 0, faiss.IndexFlatL2(2), options)
    index.add(np.ascontiguousarray(teacher, dtype=np.float32))
    output = np.empty((len(projected), k), dtype=np.int32)
    for start in range(0, len(projected), 2_048):
        stop = min(start + 2_048, len(projected))
        _distances, ids = index.search(
            np.ascontiguousarray(projected[start:stop], dtype=np.float32), k
        )
        output[start:stop] = ids.astype(np.int32, copy=False)
    if np.any(output < 0) or np.any(output >= len(teacher)):
        raise Round0175Error("low-dimensional search returned invalid neighbors")
    del index, resources
    gc.collect()
    return output


def run_scale(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    scale = str(job.get("scale") or "")
    if scale not in SCALES:
        raise Round0175Error(f"unknown R0175 scale: {scale!r}")
    output = create_fresh_directory(
        str(job["outputs"][0]), label=f"R0175 {scale} aUMAP baseline"
    )
    started = time.monotonic()
    root = TESTBED_ROOTS[scale]
    corpus = np.load(
        os.path.join(root, "train", "data-00000.npy"),
        mmap_mode="r",
        allow_pickle=False,
    )
    teacher, teacher_ids = load_coords(
        os.path.join(root, "ceiling_umaplearn_k50.parquet")
    )
    if (
        corpus.shape != (ROWS[scale], 768)
        or teacher.shape != (ROWS[scale], 2)
        or teacher_ids is None
        or not np.array_equal(teacher_ids, np.arange(ROWS[scale], dtype=np.int64))
        or not np.isfinite(teacher).all()
    ):
        raise Round0175Error(f"R0175 {scale} teacher alignment changed")
    _source, held, queries = _heldout_queries(scale)
    neighbor_ids, distances, high_performance = _exact_cosine_neighbors(
        corpus, queries
    )
    weighted = project_coordinates(
        teacher, neighbor_ids, distances, weighted=True
    )
    unweighted = project_coordinates(
        teacher, neighbor_ids, distances, weighted=False
    )
    k_fraction = max(K_HIT, int(np.ceil(FRAC * ROWS[scale])))
    stage = time.monotonic()
    weighted_low = _exact_low_neighbors(teacher, weighted, k_fraction)
    weighted_metrics = projection_metrics(neighbor_ids, weighted_low)
    del weighted_low
    unweighted_low = _exact_low_neighbors(teacher, unweighted, k_fraction)
    unweighted_metrics = projection_metrics(neighbor_ids, unweighted_low)
    del unweighted_low
    low_search_seconds = time.monotonic() - stage

    paths = {
        "held_ids": os.path.join(output, "held-source-row-ids.npy"),
        "neighbor_ids": os.path.join(output, "high-neighbor-ids.npy"),
        "neighbor_distances": os.path.join(output, "high-cosine-distances.npy"),
        "weighted_coordinates": os.path.join(output, "aumap-query-coordinates.npy"),
        "unweighted_coordinates": os.path.join(output, "unweighted-query-coordinates.npy"),
    }
    for key, value in (
        ("held_ids", held),
        ("neighbor_ids", neighbor_ids),
        ("neighbor_distances", distances),
        ("weighted_coordinates", weighted),
        ("unweighted_coordinates", unweighted),
    ):
        atomic_save_new_npy(paths[key], value, immutable=True)
    receipt = seal({
        "schema": "round0175-aumap-scale-cell-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "scale": scale,
        "rows": ROWS[scale],
        "dimension": 768,
        "n_queries": N_QUERIES,
        "held_hash": _ids_hash(held),
        "held_selection": (
            "sorted RandomState(123) choice without replacement from source rows "
            "not present in the exact testbed sample_indices"
        ),
        "neighbors": NEIGHBORS,
        "metric": "cosine over per-row fp32 L2-normalized embeddings",
        "teacher": job["teacher"],
        "testbed_embeddings": job["testbed_embeddings"],
        "sample_indices": job["sample_indices"],
        "source_manifest": job["source_manifest"],
        "source_shards": job["source_shards"],
        "artifacts": {
            key: expected_input_signature(path) for key, path in paths.items()
        },
        "aumap_inverse_distance": weighted_metrics,
        "unweighted_knn15": unweighted_metrics,
        "delta_weighted_minus_unweighted": {
            metric: weighted_metrics[metric] - unweighted_metrics[metric]
            for metric in ("ffr", "recall_at_10")
        },
        "projection_panel": {
            "k_hit": K_HIT,
            "frac": FRAC,
            "k_fraction": k_fraction,
            "ffr_formula": "canonical panel_v2 ffr_from_neighbors",
            "recall_formula": "canonical panel_v2 recall_at_k_from_neighbors",
        },
        "high_search_performance": high_performance,
        "low_search_seconds_both_methods": low_search_seconds,
        "total_wall_seconds": time.monotonic() - started,
        "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        "guards_passed": True,
        "quality_role": "diagnostic; no frozen map gate is changed",
    })
    atomic_write_new_json(
        os.path.join(output, "scale-cell.json"), receipt, immutable=True
    )


def _historical_context(job: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {"500k": None}
    for scale, key in (("200k", "evidence_200k"), ("2m", "evidence_2m")):
        signature = _signature(job[key], label=f"historical R1 {scale} evidence")
        with open(signature["canonical_path"], encoding="utf-8") as handle:
            evidence = json.load(handle)
        run = (evidence.get("runs") or {}).get("legacy_a1b1_s42")
        standard = (evidence.get("runs") or {}).get("umap_stdcurve_s42")
        if evidence.get("held_hash") != HELD_HASHES[scale] or not isinstance(run, Mapping):
            raise Round0175Error(f"historical R1 {scale} context changed")
        context[scale] = {
            "evidence": signature,
            "legacy_a1b1_seed42": {
                "projection_ffr": float(run["proj_ffr"]),
                "projection_recall_at_10": float(run["proj_recall@k"]),
                "its_own_unweighted_knn15_ffr": float(run["proj_knn_regressor_ffr"]),
            },
            "standard_curve_seed42": (
                {
                    "projection_ffr": float(standard["proj_ffr"]),
                    "projection_recall_at_10": float(standard["proj_recall@k"]),
                    "its_own_unweighted_knn15_ffr": float(
                        standard["proj_knn_regressor_ffr"]
                    ),
                }
                if isinstance(standard, Mapping)
                else None
            ),
            "comparability_note": (
                "same held-out source-row IDs, but a different transductive map; "
                "context only, not a paired selector"
            ),
        }
    return context


def run_synthesis(active: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    output = create_fresh_directory(
        str(job["outputs"][0]), label="R0175 aUMAP synthesis"
    )
    probe_signature = expected_input_signature(
        os.path.join(str(job["official_probe_output"]), "official-probe.json")
    )
    probe = _read_sealed(probe_signature, label="R0175 official formula probe")
    cell_signatures = {
        scale: expected_input_signature(
            os.path.join(str(job["scale_outputs"][scale]), "scale-cell.json")
        )
        for scale in SCALES
    }
    cells = {
        scale: _read_sealed(signature, label=f"R0175 {scale} cell")
        for scale, signature in cell_signatures.items()
    }
    synthesis = build_synthesis(
        official_probe=probe,
        cells=cells,
        historical_context=_historical_context(job),
    )
    receipt = seal({
        **synthesis,
        "release_sha": active["manifest"]["release_sha"],
        "official_probe_receipt": probe_signature,
        "scale_cell_receipts": cell_signatures,
        "capability": CAPABILITY,
    })
    atomic_write_new_json(
        os.path.join(output, "synthesis.json"), receipt, immutable=True
    )


def run_job(active: dict[str, Any], job: dict[str, Any] | None = None) -> None:
    if active.get("manifest", {}).get("round_id") != ROUND_ID or job is None:
        raise Round0175Error("R0175 handler requires its exact round/job")
    action = str(job.get("action") or "")
    if action == "official_probe":
        return run_official_probe(active, job)
    if action == "scale":
        return run_scale(active, job)
    if action == "synthesis":
        return run_synthesis(active, job)
    raise Round0175Error(f"unknown R0175 action: {action!r}")
