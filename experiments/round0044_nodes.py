"""Fresh-process handler for the R0044 graph-candidate quality sweep."""
from __future__ import annotations

import json
import math
import os
import resource
import time
from typing import Any, Iterable

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
from experiments.round0029_program import ordered_embedding_paths


ROUND_ID = "0044"
INDEX_PATH = "/data/checkpoints/pumap/faiss_ivf_pq_3m.index"
INDEX_SHA256 = (
    "72be184fd0720c6749204820f661affdbac62a69f4332a1ece2fec6bb2ab7590"
)
R0031_MEASUREMENT = (
    "/data/latent-basemap/runs/round-0031/queue/artifacts/path-b/"
    "path-b-candidate-rerank.json"
)
R0031_MEASUREMENT_SHA256 = (
    "aac8c92b52f3535fe41506fb90ec3f63b8a762ff38cd85d255a7240c7e0afa2f"
)
N_BASE = 3_000_000
N_SAMPLE = 50_000
K = 15
SEED = 0
COARSE_NPROBES = (1, 4, 16, 32, 64, 128, 256, 512, 1024, 1732)
PQ_NPROBES = (64, 256)
PQ_WIDTHS = (15, 128, 512, 2048, 8192)
MEAN_RECALL_FLOOR = 0.90


class Round0044Error(RuntimeError):
    """R0044 contract or execution failure."""


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    return {
        **body,
        "identity_sha256": sha256_bytes(canonical_json(body)),
    }


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _metric_row(
    per_query_recall: np.ndarray,
    *,
    unambiguous: np.ndarray,
) -> dict[str, Any]:
    if len(per_query_recall) != len(unambiguous) or not np.any(unambiguous):
        raise Round0044Error("invalid tie-aware metric cohort")
    raw = np.asarray(per_query_recall, dtype=np.float64)
    clear = raw[unambiguous]
    return {
        "mean_recall_at_15": round(float(raw.mean()), 6),
        "p10_recall_at_15": round(_percentile(raw, 10), 6),
        "mean_recall_at_15_unambiguous": round(
            float(clear.mean()), 6
        ),
        "p10_recall_at_15_unambiguous": round(
            _percentile(clear, 10), 6
        ),
    }


def _candidate_ranks(
    *,
    exact_neighbors: np.ndarray,
    candidates: np.ndarray,
    n_base: int,
) -> np.ndarray:
    """Return each exact neighbor's zero-based position in one ANN shortlist."""
    truth = np.asarray(exact_neighbors, dtype=np.int64)
    shortlist = np.asarray(candidates, dtype=np.int64)
    if (
        truth.ndim != 2
        or shortlist.ndim != 2
        or len(truth) != len(shortlist)
        or np.any(shortlist < 0)
        or np.any(shortlist >= n_base)
    ):
        raise Round0044Error("malformed candidate-rank inputs")
    missing = shortlist.shape[1] + 1
    scratch = np.full(n_base, missing, dtype=np.int32)
    ranks = np.empty_like(truth, dtype=np.int32)
    positions = np.arange(shortlist.shape[1], dtype=np.int32)
    for row in range(len(shortlist)):
        ids = shortlist[row]
        scratch[ids] = positions
        ranks[row] = scratch[truth[row]]
        scratch[ids] = missing
    return ranks


def _clean_search_rows(
    raw: np.ndarray,
    *,
    query_ids: np.ndarray,
    width: int,
    n_base: int,
) -> np.ndarray:
    """Remove explicit self IDs without assuming that self is rank zero."""
    values = np.asarray(raw, dtype=np.int64)
    queries = np.asarray(query_ids, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != len(queries):
        raise Round0044Error("malformed ANN search output")
    cleaned = np.empty((len(values), width), dtype=np.int64)
    for row in range(len(values)):
        ids = values[row]
        ids = ids[
            (ids >= 0)
            & (ids < n_base)
            & (ids != queries[row])
        ]
        if len(ids) < width:
            raise Round0044Error(
                f"ANN row {row} has {len(ids)} nonself IDs; need {width}"
            )
        cleaned[row] = ids[:width]
    return cleaned


def _extract_list_assignments(index: Any) -> tuple[np.ndarray, np.ndarray]:
    """Recover the unique IVF list for every row from the accepted index."""
    import faiss

    assignments = np.full(N_BASE, -1, dtype=np.int32)
    list_sizes = np.empty(int(index.nlist), dtype=np.int64)
    for list_id in range(int(index.nlist)):
        size = int(index.invlists.list_size(list_id))
        list_sizes[list_id] = size
        if not size:
            continue
        ids = np.array(
            faiss.rev_swig_ptr(
                index.invlists.get_ids(list_id),
                size,
            ),
            dtype=np.int64,
            copy=True,
        )
        if np.any(ids < 0) or np.any(ids >= N_BASE):
            raise Round0044Error("IVF list contains an out-of-universe row")
        if len(np.unique(ids)) != size:
            raise Round0044Error("IVF list contains a duplicate row ID")
        if np.any(assignments[ids] != -1):
            raise Round0044Error("IVF row ID occurs in multiple lists")
        assignments[ids] = list_id
    if int(list_sizes.sum()) != N_BASE or np.any(assignments < 0):
        raise Round0044Error("IVF lists do not cover the 3M universe once")
    return assignments, list_sizes


def _exact_truth(
    base: np.ndarray,
    sample: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    import torch
    import torch.nn.functional as functional

    if not torch.cuda.is_available():
        raise Round0044Error("R0044 requires CUDA for exact 3M truth")
    started = time.monotonic()
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    base_tensor = functional.normalize(
        torch.from_numpy(base).to(device),
        dim=1,
    )
    query_tensor = functional.normalize(
        torch.from_numpy(
            np.ascontiguousarray(base[sample])
        ).to(device),
        dim=1,
    )
    tile = max(64, int(4e9 / (4 * N_BASE)))
    neighbors = np.empty((len(sample), K), dtype=np.int32)
    ties = np.zeros(len(sample), dtype=bool)
    with torch.no_grad():
        for start in range(0, len(sample), tile):
            stop = min(start + tile, len(sample))
            similarity = query_tensor[start:stop] @ base_tensor.T
            rows = torch.arange(stop - start, device=device)
            columns = torch.as_tensor(
                sample[start:stop],
                dtype=torch.long,
                device=device,
            )
            similarity[rows, columns] = -torch.inf
            top = torch.topk(
                similarity,
                K + 1,
                dim=1,
                largest=True,
            )
            neighbors[start:stop] = (
                top.indices[:, :K].cpu().numpy().astype(np.int32)
            )
            ties[start:stop] = (
                torch.abs(
                    top.values[:, K - 1] - top.values[:, K]
                )
                <= 1e-7
            ).cpu().numpy()
    peak = {
        "allocated_gb": round(
            torch.cuda.max_memory_allocated(device) / 1e9,
            4,
        ),
        "reserved_gb": round(
            torch.cuda.max_memory_reserved(device) / 1e9,
            4,
        ),
    }
    del similarity, top, query_tensor, base_tensor
    torch.cuda.empty_cache()
    return neighbors, ties, time.monotonic() - started, peak


def _coarse_oracle(
    *,
    queries: np.ndarray,
    truth_lists: np.ndarray,
    query_lists: np.ndarray,
    centroids: np.ndarray,
    list_sizes: np.ndarray,
    ties: np.ndarray,
    nprobes: Iterable[int],
) -> tuple[dict[str, Any], np.ndarray, float, dict[str, float]]:
    import torch

    device = torch.device("cuda")
    probes = tuple(sorted(set(map(int, nprobes))))
    if probes[-1] != len(centroids):
        raise Round0044Error("coarse sweep must include every IVF list")
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    centroid_tensor = torch.from_numpy(
        np.ascontiguousarray(centroids, dtype=np.float32)
    ).to(device)
    centroid_norm = (centroid_tensor * centroid_tensor).sum(1)
    list_sizes_tensor = torch.from_numpy(list_sizes).to(device)
    truth_ranks = np.empty_like(truth_lists, dtype=np.int32)
    candidate_counts = {
        probe: np.empty(len(queries), dtype=np.int64)
        for probe in probes
    }
    own_list_ranks = np.empty(len(queries), dtype=np.int32)
    batch = 4096
    with torch.no_grad():
        for start in range(0, len(queries), batch):
            stop = min(start + batch, len(queries))
            query = torch.from_numpy(
                np.ascontiguousarray(
                    queries[start:stop],
                    dtype=np.float32,
                )
            ).to(device)
            distance = (
                (query * query).sum(1, keepdim=True)
                + centroid_norm.unsqueeze(0)
                - 2.0 * query @ centroid_tensor.T
            )
            order = torch.argsort(distance, dim=1)
            inverse = torch.empty_like(order)
            inverse.scatter_(
                1,
                order,
                torch.arange(
                    len(centroids),
                    device=device,
                ).expand(stop - start, -1),
            )
            truth_index = torch.from_numpy(
                truth_lists[start:stop].astype(np.int64)
            ).to(device)
            truth_ranks[start:stop] = (
                inverse.gather(1, truth_index)
                .cpu()
                .numpy()
                .astype(np.int32)
            )
            own_index = torch.from_numpy(
                query_lists[start:stop].astype(np.int64)
            ).to(device)
            own_list_ranks[start:stop] = (
                inverse.gather(1, own_index[:, None])[:, 0]
                .cpu()
                .numpy()
                .astype(np.int32)
            )
            cumulative = torch.cumsum(
                list_sizes_tensor[order],
                dim=1,
            )
            for probe in probes:
                candidate_counts[probe][start:stop] = (
                    cumulative[:, probe - 1].cpu().numpy()
                )
            del (
                query,
                distance,
                order,
                inverse,
                truth_index,
                own_index,
                cumulative,
            )
    rows: dict[str, Any] = {}
    unambiguous = ~ties
    for probe in probes:
        recall = (truth_ranks < probe).mean(axis=1)
        counts = candidate_counts[probe]
        rows[str(probe)] = {
            **_metric_row(recall, unambiguous=unambiguous),
            "candidate_rows_mean": round(float(counts.mean()), 2),
            "candidate_rows_min": int(counts.min()),
            "candidate_rows_p50": round(_percentile(counts, 50), 2),
            "candidate_rows_p90": round(_percentile(counts, 90), 2),
            "candidate_rows_max": int(counts.max()),
            "candidate_fraction_mean": round(
                float(counts.mean() / N_BASE),
                8,
            ),
        }
    peak = {
        "allocated_gb": round(
            torch.cuda.max_memory_allocated(device) / 1e9,
            4,
        ),
        "reserved_gb": round(
            torch.cuda.max_memory_reserved(device) / 1e9,
            4,
        ),
    }
    del centroid_tensor, centroid_norm, list_sizes_tensor
    torch.cuda.empty_cache()
    return rows, own_list_ranks, time.monotonic() - started, peak


def _pq_shortlist_sweep(
    *,
    index: Any,
    queries: np.ndarray,
    query_ids: np.ndarray,
    exact_neighbors: np.ndarray,
    ties: np.ndarray,
    candidate_min_by_probe: dict[int, int],
) -> tuple[
    dict[str, Any],
    dict[str, float],
    dict[str, Any],
]:
    rows: dict[str, Any] = {}
    timings: dict[str, float] = {}
    availability: dict[str, Any] = {}
    for probe in PQ_NPROBES:
        max_nonself = int(candidate_min_by_probe[probe]) - 1
        widths = tuple(
            width for width in PQ_WIDTHS if width <= max_nonself
        )
        if 128 not in widths:
            raise Round0044Error(
                f"nprobe {probe} cannot reproduce R0031 width 128"
            )
        max_width = max(widths)
        all_ranks, timings[str(probe)] = _search_candidate_ranks(
            index=index,
            queries=queries,
            query_ids=query_ids,
            exact_neighbors=exact_neighbors,
            nprobe=probe,
            width=max_width,
        )
        cells: dict[str, Any] = {}
        for width in widths:
            recall = (all_ranks < width).mean(axis=1)
            cells[str(width)] = _metric_row(
                recall,
                unambiguous=~ties,
            )
        rows[str(probe)] = cells
        availability[str(probe)] = {
            "candidate_rows_min_including_possible_self": int(
                candidate_min_by_probe[probe]
            ),
            "max_uniform_nonself_width": max_nonself,
            "requested_widths": list(PQ_WIDTHS),
            "evaluated_widths": list(widths),
            "omitted_widths": [
                width for width in PQ_WIDTHS if width not in widths
            ],
        }
    return rows, timings, availability


def _search_candidate_ranks(
    *,
    index: Any,
    queries: np.ndarray,
    query_ids: np.ndarray,
    exact_neighbors: np.ndarray,
    nprobe: int,
    width: int,
) -> tuple[np.ndarray, float]:
    index.nprobe = int(nprobe)
    all_ranks = np.empty_like(exact_neighbors, dtype=np.int32)
    started = time.monotonic()
    batch = 1024
    for start in range(0, len(queries), batch):
        stop = min(start + batch, len(queries))
        _, raw = index.search(
            np.ascontiguousarray(queries[start:stop]),
            width + 1,
        )
        clean = _clean_search_rows(
            raw,
            query_ids=query_ids[start:stop],
            width=width,
            n_base=N_BASE,
        )
        all_ranks[start:stop] = _candidate_ranks(
            exact_neighbors=exact_neighbors[start:stop],
            candidates=clean,
            n_base=N_BASE,
        )
    return all_ranks, time.monotonic() - started


def _choose_candidate_policy(
    coarse: dict[str, Any],
    pq: dict[str, Any],
) -> dict[str, Any]:
    qualifying_pq: list[tuple[int, int, float]] = []
    for probe, widths in pq.items():
        for width, metrics in widths.items():
            recall = float(
                metrics["mean_recall_at_15_unambiguous"]
            )
            if recall >= MEAN_RECALL_FLOOR:
                qualifying_pq.append((int(probe), int(width), recall))
    qualifying_pq.sort(key=lambda value: (value[1], value[0]))
    qualifying_coarse = [
        (int(probe), float(metrics["mean_recall_at_15_unambiguous"]))
        for probe, metrics in coarse.items()
        if float(metrics["mean_recall_at_15_unambiguous"])
        >= MEAN_RECALL_FLOOR
    ]
    qualifying_coarse.sort()
    if qualifying_pq:
        probe, width, recall = qualifying_pq[0]
        return {
            "classification": "ivfpq-shortlist-plus-exact-rerank",
            "selected_nprobe": probe,
            "selected_width": width,
            "selected_recall": recall,
            "requires_new_exact_vector_generator": False,
        }
    if qualifying_coarse:
        probe, recall = qualifying_coarse[0]
        return {
            "classification": "exact-vector-search-with-current-coarse-routing",
            "selected_nprobe": probe,
            "selected_width": None,
            "selected_recall": recall,
            "requires_new_exact_vector_generator": True,
        }
    return {
        "classification": "replace-coarse-routing",
        "selected_nprobe": None,
        "selected_width": None,
        "selected_recall": None,
        "requires_new_exact_vector_generator": True,
    }


def run_candidate_sweep(
    active: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    import faiss
    from experiments.build_weighted_graph import ShardedEmbeddings

    output = create_fresh_directory(
        job["outputs"][0],
        label="Round 0044 candidate-quality output",
    )
    total_started = time.monotonic()
    signatures = {
        "index": expected_input_signature(INDEX_PATH),
        "r0031_measurement": expected_input_signature(R0031_MEASUREMENT),
    }
    if signatures["index"]["sha256"] != INDEX_SHA256:
        raise Round0044Error("accepted 3M IVF-PQ index bytes changed")
    if (
        signatures["r0031_measurement"]["sha256"]
        != R0031_MEASUREMENT_SHA256
    ):
        raise Round0044Error("accepted R0031 measurement bytes changed")
    with open(R0031_MEASUREMENT, encoding="utf-8") as handle:
        r0031 = json.load(handle)
    if (
        r0031.get("n_base") != N_BASE
        or r0031.get("n_sample") != N_SAMPLE
        or r0031.get("k") != K
        or r0031.get("nprobe") != 64
    ):
        raise Round0044Error("R0031 comparison contract changed")

    load_started = time.monotonic()
    paths = ordered_embedding_paths()[:3]
    embeddings = ShardedEmbeddings(paths, expected_dim=384)
    if len(embeddings) != N_BASE:
        raise Round0044Error("R0044 embedding universe is not 3M")
    base = np.ascontiguousarray(
        embeddings.gather(
            np.arange(N_BASE, dtype=np.int64),
            out_dtype=np.float32,
        )
    )
    load_seconds = time.monotonic() - load_started
    rng = np.random.RandomState(SEED)
    sample = np.sort(
        rng.choice(N_BASE, N_SAMPLE, replace=False)
    ).astype(np.int64)
    queries = np.ascontiguousarray(base[sample])

    index = faiss.read_index(INDEX_PATH)
    if (
        type(index).__name__ != "IndexIVFPQ"
        or int(index.ntotal) != N_BASE
        or int(index.d) != 384
        or int(index.nlist) != COARSE_NPROBES[-1]
        or int(index.pq.M) != 48
        or int(index.pq.nbits) != 8
    ):
        raise Round0044Error("accepted IVF-PQ index geometry changed")
    assignment_started = time.monotonic()
    assignments, list_sizes = _extract_list_assignments(index)
    assignment_seconds = time.monotonic() - assignment_started
    try:
        centroids = np.asarray(
            index.quantizer.reconstruct_n(0, int(index.nlist)),
            dtype=np.float32,
        )
    except TypeError:
        centroids = np.empty((int(index.nlist), 384), dtype=np.float32)
        index.quantizer.reconstruct_n(
            0,
            int(index.nlist),
            centroids,
        )
    if centroids.shape != (int(index.nlist), 384):
        raise Round0044Error("could not recover IVF coarse centroids")

    exact_neighbors, ties, exact_seconds, exact_peak = _exact_truth(
        base,
        sample,
    )
    truth_lists = assignments[exact_neighbors]
    coarse, own_ranks, coarse_seconds, coarse_peak = _coarse_oracle(
        queries=queries,
        truth_lists=truth_lists,
        query_lists=assignments[sample],
        centroids=centroids,
        list_sizes=list_sizes,
        ties=ties,
        nprobes=COARSE_NPROBES,
    )
    r0031_ranks, r0031_search_seconds = _search_candidate_ranks(
        index=index,
        queries=queries,
        query_ids=sample,
        exact_neighbors=exact_neighbors,
        nprobe=64,
        width=128,
    )
    r0031_observed = {
        "width15": _metric_row(
            (r0031_ranks < 15).mean(axis=1),
            unambiguous=~ties,
        ),
        "width128": _metric_row(
            (r0031_ranks < 128).mean(axis=1),
            unambiguous=~ties,
        ),
    }
    pq, pq_timings, pq_availability = _pq_shortlist_sweep(
        index=index,
        queries=queries,
        query_ids=sample,
        exact_neighbors=exact_neighbors,
        ties=ties,
        candidate_min_by_probe={
            probe: int(coarse[str(probe)]["candidate_rows_min"])
            for probe in PQ_NPROBES
        },
    )

    reproduction = {
        "nprobe64_width15": (
            abs(
                float(
                    r0031_observed["width15"]["mean_recall_at_15"]
                )
                - float(r0031["recall_at_k"])
            )
            <= 0.002
        ),
        "nprobe64_width128_unambiguous": (
            abs(
                float(
                    r0031_observed["width128"][
                        "mean_recall_at_15_unambiguous"
                    ]
                )
                - float(
                    r0031["path_b_by_candidate_width"]["128"][
                        "candidate_recall_at_k_unambiguous"
                    ]
                )
            )
            <= 0.002
        ),
    }
    checks = {
        "r0031_reproduced": all(reproduction.values()),
        "all_rows_have_one_ivf_list": True,
        "query_own_list_is_top_two_centroids_at_least_0_99": (
            float(np.mean(own_ranks <= 1)) >= 0.99
        ),
        "full_coarse_probe_has_unit_recall": (
            float(
                coarse[str(index.nlist)][
                    "mean_recall_at_15_unambiguous"
                ]
            )
            == 1.0
        ),
        "unambiguous_fraction_at_least_0_90": (
            float((~ties).mean()) >= 0.90
        ),
        "no_training_performed": True,
    }
    if not all(checks.values()):
        raise Round0044Error("R0044 validity guard failed")

    body = {
        "schema": "round0044-candidate-quality-sweep-v1",
        "round_id": ROUND_ID,
        "release_sha": active["manifest"]["release_sha"],
        "training_performed": False,
        "optimizer_updates": 0,
        "inputs": {
            **signatures,
            "embedding_members": [
                expected_input_signature(path) for path in paths
            ],
        },
        "universe": {
            "rows": N_BASE,
            "dimensions": 384,
            "sample_rows": N_SAMPLE,
            "sample_seed": SEED,
            "k": K,
            "exact_boundary_tie_count": int(ties.sum()),
            "exact_boundary_tie_fraction": round(
                float(ties.mean()),
                8,
            ),
            "unambiguous_queries": int((~ties).sum()),
        },
        "index": {
            "type": type(index).__name__,
            "nlist": int(index.nlist),
            "pq_m": int(index.pq.M),
            "pq_nbits": int(index.pq.nbits),
            "list_size_min": int(list_sizes.min()),
            "list_size_mean": round(float(list_sizes.mean()), 4),
            "list_size_p90": round(_percentile(list_sizes, 90), 4),
            "list_size_max": int(list_sizes.max()),
            "query_own_list_rank_zero_fraction": round(
                float(np.mean(own_ranks == 0)),
                8,
            ),
            "query_own_list_rank_le_one_fraction": round(
                float(np.mean(own_ranks <= 1)),
                8,
            ),
        },
        "coarse_cell_oracle": coarse,
        "ivfpq_shortlist_coverage": pq,
        "ivfpq_shortlist_availability": pq_availability,
        "decision": _choose_candidate_policy(coarse, pq),
        "r0031_reproduction": reproduction,
        "r0031_reproduction_observed": r0031_observed,
        "checks": checks,
        "performance": {
            "load_seconds": round(load_seconds, 4),
            "list_assignment_seconds": round(assignment_seconds, 4),
            "exact_truth_seconds": round(exact_seconds, 4),
            "coarse_oracle_seconds": round(coarse_seconds, 4),
            "r0031_reproduction_search_seconds": round(
                r0031_search_seconds,
                4,
            ),
            "pq_search_seconds_by_nprobe": {
                key: round(value, 4)
                for key, value in pq_timings.items()
            },
            "wall_seconds": round(
                time.monotonic() - total_started,
                4,
            ),
            "exact_truth_gpu_peak": exact_peak,
            "coarse_oracle_gpu_peak": coarse_peak,
            "rss_peak_gb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024
                / 1e9,
                4,
            ),
        },
    }
    receipt = _seal(body)
    path = os.path.join(output, "candidate-quality-sweep-v1.json")
    atomic_write_new_json(path, receipt, immutable=True)
    return {
        **receipt,
        "receipt": expected_input_signature(path),
    }


def run_job(
    active: dict[str, Any],
    job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if active.get("manifest", {}).get("round_id") != ROUND_ID:
        raise Round0044Error("R0044 handler received another queue")
    selected = job if job is not None else active.get("job") or {}
    if (
        selected.get("action") != "candidate_quality_sweep"
        or len(selected.get("outputs") or []) != 1
    ):
        raise Round0044Error("R0044 job contract changed")
    return run_candidate_sweep(active, selected)
