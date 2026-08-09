#!/usr/bin/env python3
"""R0229 — R0227's `cluster-spill-nnd` builder with the nn-descent knobs freed.

Runs under the RAPIDS env (`/data/latent-basemap/cuml_py`), one fresh process per
cell, guarded and watched by the parent:

    cuml_py basemap/round0229_quality_build.py --config c.json --out d

This is R0227's build script with **three** changes, and nothing else moves:

1. **`graph_degree`, `intermediate_graph_degree` and `max_iterations` come from
   the config** instead of R0226's module constants. That is the round's whole
   subject: review-0227-01 recommended raising them and R0228 called it the
   single highest-value next experiment, but no round has ever varied them.
2. **`spill` comes from the config.** Every round since R0226 has held
   `A_SPILL = 2` as a constant. The reachability ceiling is a function of the
   partition, and the partition has two knobs, not one.
3. **The k-means assignment can be cached and reused.** The quality sweep must
   hold the partition *identical* across cells, or a recall difference between
   two nn-descent settings is confounded with a difference in what was reachable.
   The first cell writes the assignment; every later cell loads the same bytes
   and asserts its shape and cluster count. This is a fail-closed identity, not
   an optimisation: a cell that cannot bind the cached partition refuses.

Everything else is R0227's, imported and called unmodified: the k-means recipe
(Lloyd, seed 226, 25 iterations, 1,000,000-row subsample), the whole-cluster
spill-group packing against the scratch budget, the exact-cosine recompute, the
exact incremental global top-k merge, the capacity refusal, the cooperative
abort, and the OOM-as-measurement discipline.

An OOM, a capacity refusal and a cooperative abort are all measurements: they are
caught, written as `fit: false`, and exit is 0 so the ladder records where the
configuration stops.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemap.round0226_cluster_spill_build import (  # noqa: E402
    OOM_MARKERS,
    CooperativeAbort,
    Sampler,
    _assign,
    _install_sigterm_handler,
    _kmeans,
    _rss_bytes,
    _vmhwm_bytes,
)
from basemap.round0226_graph_builders import (  # noqa: E402
    A_ASSIGN_BLOCK,
    A_CLUSTER_STAGE_ROWS,
    A_KMEANS_ITERATIONS,
    A_KMEANS_SUBSAMPLE_ROWS,
    A_METRIC,
    A_SEED,
    GRAPH_K,
    merge_into_topk,
)
from basemap.round0227_low_c_contract import (  # noqa: E402
    CANDIDATE,
    CLUSTER_CAPACITY_ROWS,
    SCRATCH_BUDGET_BYTES,
    pack_clusters_into_groups,
)
from basemap.round0229_quality_contract import BUILD_SCHEMA  # noqa: E402


def _load_or_build_assignment(
    cp,
    cupyx,
    dataset,
    *,
    rows: int,
    clusters: int,
    spill: int,
    cache_path: str | None,
    phase: dict,
) -> tuple[np.ndarray, dict]:
    """The shared partition. Cached bytes bind; they are never regenerated."""
    provenance: dict = {
        "kmeans_seed": A_SEED,
        "kmeans_iterations": A_KMEANS_ITERATIONS,
        "kmeans_subsample_rows": A_KMEANS_SUBSAMPLE_ROWS,
        "assignment_cache_path": cache_path,
    }
    if cache_path and os.path.exists(cache_path):
        started = time.perf_counter()
        assignment = np.load(cache_path)
        phase["assignment_load_seconds"] = time.perf_counter() - started
        if assignment.shape != (rows, spill):
            raise SystemExit(
                f"cached assignment {assignment.shape} is not ({rows}, {spill})"
            )
        observed = int(assignment.max()) + 1
        if observed > clusters:
            raise SystemExit(
                f"cached assignment references {observed} clusters against {clusters}"
            )
        provenance.update({
            "assignment_source": "cache",
            "assignment_reused": True,
            "assignment_cache_bytes": int(os.path.getsize(cache_path)),
        })
        return assignment, provenance

    started = time.perf_counter()
    centroids = _kmeans(cp, cupyx, dataset, clusters=clusters, seed=A_SEED)
    phase["kmeans_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    assignment = _assign(cp, dataset, centroids, rows=rows, spill=spill)
    phase["assign_seconds"] = time.perf_counter() - started
    del centroids
    cp.get_default_memory_pool().free_all_blocks()
    provenance.update({"assignment_source": "computed", "assignment_reused": False})
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        # np.save appends ".npy" to a path that does not already end in it, so
        # the temporary is written through an open handle and renamed. Writing
        # through a temporary keeps a partial file from ever being bound as the
        # shared partition by a later cell.
        temporary = f"{cache_path}.partial"
        with open(temporary, "wb") as handle:
            np.save(handle, assignment)
        os.replace(temporary, cache_path)
        provenance["assignment_cache_bytes"] = int(os.path.getsize(cache_path))
        provenance["assignment_cache_written"] = True
    return assignment, provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    _install_sigterm_handler()
    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    os.makedirs(args.out, exist_ok=True)
    rows = int(config["rows"])
    clusters = int(config["clusters"])
    spill = int(config["spill"])
    dimension = int(config["dimension"])
    scratch_root = str(config["scratch_root"])
    interval = float(config.get("sample_interval_s", 0.005))
    graph_degree = int(config["graph_degree"])
    intermediate_degree = int(config["intermediate_graph_degree"])
    max_iterations = int(config["max_iterations"])
    cache_path = config.get("assignment_cache") or None
    if spill < 1:
        raise SystemExit(f"spill {spill} is not >= 1")
    if intermediate_degree < graph_degree:
        raise SystemExit(
            f"intermediate_graph_degree {intermediate_degree} is below "
            f"graph_degree {graph_degree}; cuVS requires intermediate >= graph"
        )

    import rmm
    import cupy as cp
    import cupyx

    statistics = rmm.mr.StatisticsResourceAdaptor(rmm.mr.CudaMemoryResource())
    rmm.mr.set_current_device_resource(statistics)
    resource_type = str(type(rmm.mr.get_current_device_resource()))

    import cuvs
    from cuvs.neighbors import nn_descent

    resource_after = str(type(rmm.mr.get_current_device_resource()))
    try:
        import cuvs.neighbors as _neighbors

        has_all_neighbors = hasattr(_neighbors, "all_neighbors")
    except Exception:  # noqa: BLE001
        has_all_neighbors = False

    # R0227's warm-up, at R0227's literals, so the warm-up cost is the same
    # constant in every cell and never varies with the cell's own setting.
    warm_started = time.perf_counter()
    nn_descent.build(
        nn_descent.IndexParams(
            metric=A_METRIC, graph_degree=32, intermediate_graph_degree=48,
            max_iterations=4,
        ),
        np.ascontiguousarray(
            np.random.default_rng(A_SEED).standard_normal((4096, 8)), dtype=np.float32
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    warmup_seconds = time.perf_counter() - warm_started

    free, total = cp.cuda.runtime.memGetInfo()
    receipt = {
        "schema": BUILD_SCHEMA,
        "candidate": CANDIDATE,
        "setting_id": str(config["setting_id"]),
        "cell": str(config.get("cell") or config["setting_id"]),
        "config": config,
        "rows": rows,
        "dimension": dimension,
        "k": GRAPH_K,
        "clusters": clusters,
        "spill": spill,
        "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
        "scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
        "kmeans_subsample_rows": A_KMEANS_SUBSAMPLE_ROWS,
        "kmeans_iterations": A_KMEANS_ITERATIONS,
        "graph_degree": graph_degree,
        "intermediate_graph_degree": intermediate_degree,
        "max_iterations": max_iterations,
        "metric": A_METRIC,
        "seed": A_SEED,
        "cuvs_version": str(cuvs.__version__),
        "cuvs_has_all_neighbors": bool(has_all_neighbors),
        "rmm_resource_type": resource_type,
        "rmm_resource_type_after_cuvs_import": resource_after,
        "rmm_resource_replaced_by_cuvs": bool(resource_after != resource_type),
        "warmup_seconds": warmup_seconds,
        "device_total_bytes": int(total),
        "device_baseline_bytes": int(total) - int(free),
        "host_rss_baseline_bytes": _rss_bytes(),
        "rmm_baseline_peak_bytes": int(statistics.allocation_counts.peak_bytes),
        "sample_interval_s": interval,
        "parameters_note": (
            "R0227's builder with graph_degree, intermediate_graph_degree, "
            "max_iterations and spill taken from the config, and the k-means "
            "assignment optionally bound to cached bytes so a quality sweep "
            "holds the partition identical across cells. Every other parameter "
            "is R0226's, imported and called unmodified."
        ),
        "igd_host_law_bytes_per_row": (
            2 * int(32 * -(-int(1.3 * intermediate_degree) // 32))
        ),
        "igd_host_law_note": (
            "the intermediate graph is host-resident and quantised as "
            "2 B/row x roundUp32(1.3 x igd) (plan-minilm-100m-v2, five points to "
            "<=1.4%); this is the predicted host term for THIS igd"
        ),
    }

    sampler = Sampler(cp, interval)
    sampler.start()
    scratch = os.path.join(scratch_root, str(config["setting_id"]))
    phase: dict[str, float] = {}
    try:
        if os.path.isdir(scratch):
            shutil.rmtree(scratch)
        os.makedirs(scratch, exist_ok=True)

        substrate = np.load(str(config["substrate"]), mmap_mode="r")
        if substrate.ndim != 2 or int(substrate.shape[1]) != dimension:
            raise SystemExit(f"substrate shape {substrate.shape} is not (*, {dimension})")
        if rows > int(substrate.shape[0]):
            raise SystemExit(f"requested {rows} rows, substrate has {substrate.shape[0]}")
        dataset = substrate[:rows]

        assignment, partition = _load_or_build_assignment(
            cp, cupyx, dataset, rows=rows, clusters=clusters, spill=spill,
            cache_path=cache_path, phase=phase,
        )
        receipt["partition"] = partition

        sizes = np.bincount(assignment.ravel(), minlength=clusters).astype(np.int64)
        largest = int(sizes.max())
        receipt.update({
            "cluster_sizes": {
                "min": int(sizes.min()),
                "max": largest,
                "mean": float(sizes.mean()),
                "median": float(np.median(sizes)),
                "empty_clusters": int((sizes == 0).sum()),
                "imbalance_max_over_mean": (
                    float(largest / sizes.mean()) if sizes.mean() > 0 else None
                ),
            },
            "cluster_sizes_all": [int(value) for value in sizes],
        })
        if largest > CLUSTER_CAPACITY_ROWS:
            receipt.update({
                "fit": False,
                "oom": False,
                "timed_out": False,
                "refused_after_assignment": True,
                "error_type": "ClusterCapacityExceeded",
                "error": (
                    f"largest realised cluster {largest} exceeds the registered "
                    f"capacity {CLUSTER_CAPACITY_ROWS}"
                ),
                "phases": phase,
                "rmm_peak_bytes": int(statistics.allocation_counts.peak_bytes),
                "child_device_peak_sampled_bytes": int(sampler.device_peak),
                "host_peak_sampled_bytes": int(sampler.host_peak),
                "host_vmhwm_bytes": _vmhwm_bytes(),
            })
            raise SystemExit(0)

        top_ids = np.full((rows, GRAPH_K), -1, dtype=np.int32)
        top_cos = np.full((rows, GRAPH_K), -np.inf, dtype=np.float32)

        flat_cluster = assignment.ravel().astype(np.int64)
        flat_row = np.repeat(np.arange(rows, dtype=np.int64), spill)
        order = np.argsort(flat_cluster, kind="stable")
        members = flat_row[order]
        bounds = np.zeros(clusters + 1, dtype=np.int64)
        np.cumsum(sizes, out=bounds[1:])
        del flat_cluster, flat_row, order, assignment

        groups = pack_clusters_into_groups(sizes, budget_bytes=SCRATCH_BUDGET_BYTES)
        group_bytes = [
            int(sum(int(sizes[index]) for index in group) * dimension * 4)
            for group in groups
        ]
        receipt.update({
            "spill_groups": len(groups),
            "spill_group_cluster_counts": [len(group) for group in groups],
            "spill_group_bytes": group_bytes,
            "peak_scratch_bytes": int(max(group_bytes) if group_bytes else 0),
            "substrate_passes": len(groups),
            "substrate_read_bytes": int(len(groups) * rows * dimension * 4),
            "spill_write_bytes": int(int(sizes.sum()) * dimension * 4),
            "scratch_budget_exceeded_by_single_cluster": bool(
                group_bytes and max(group_bytes) > SCRATCH_BUDGET_BYTES
            ),
        })

        build_seconds = 0.0
        merge_seconds = 0.0
        spill_seconds = 0.0
        cosine_seconds = 0.0
        cluster_receipts: list[dict[str, object]] = []

        for group in groups:
            started = time.perf_counter()
            handles: dict[int, np.ndarray] = {}
            cursors: dict[int, int] = {}
            paths: dict[int, str] = {}
            member_of: dict[int, np.ndarray] = {}
            position: dict[int, int] = {}
            for cluster in group:
                size = int(sizes[cluster])
                path = os.path.join(scratch, f"cluster-{cluster:05d}.f32.npy")
                paths[cluster] = path
                handles[cluster] = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(size, dimension)
                )
                cursors[cluster] = 0
                member_of[cluster] = members[bounds[cluster] : bounds[cluster + 1]]
                position[cluster] = 0
            for begin in range(0, rows, A_ASSIGN_BLOCK):
                end = min(begin + A_ASSIGN_BLOCK, rows)
                block = np.ascontiguousarray(dataset[begin:end], dtype=np.float32)
                for cluster in group:
                    ids = member_of[cluster]
                    start_at = position[cluster]
                    stop_at = int(np.searchsorted(ids, end, side="left"))
                    if stop_at <= start_at:
                        continue
                    take = ids[start_at:stop_at]
                    cursor = cursors[cluster]
                    handles[cluster][cursor : cursor + take.size] = block[take - begin]
                    cursors[cluster] = cursor + int(take.size)
                    position[cluster] = stop_at
                del block
            for cluster in list(handles):
                handles[cluster].flush()
                del handles[cluster]
            handles.clear()
            spill_seconds += time.perf_counter() - started

            for cluster in group:
                size = int(sizes[cluster])
                global_ids = members[bounds[cluster] : bounds[cluster + 1]]
                chunk = np.load(paths[cluster], mmap_mode="r")
                started = time.perf_counter()
                brute = size <= 2 * intermediate_degree
                if brute:
                    small = cp.asarray(np.ascontiguousarray(chunk, dtype=np.float32))
                    width = min(graph_degree, max(1, size - 1))
                    scores = small @ small.T
                    cp.fill_diagonal(scores, cp.float32(-np.inf))
                    local_graph = cp.asnumpy(
                        cp.argsort(-scores, axis=1)[:, :width]
                    ).astype(np.int64, copy=False)
                    del small, scores
                    cp.get_default_memory_pool().free_all_blocks()
                else:
                    index = nn_descent.build(
                        nn_descent.IndexParams(
                            metric=A_METRIC,
                            graph_degree=graph_degree,
                            intermediate_graph_degree=intermediate_degree,
                            max_iterations=max_iterations,
                        ),
                        chunk,
                    )
                    cp.cuda.runtime.deviceSynchronize()
                    local_graph = np.asarray(index.graph).astype(np.int64, copy=True)
                    del index
                cluster_build_seconds = time.perf_counter() - started
                build_seconds += cluster_build_seconds
                if local_graph.shape[0] != size:
                    raise RuntimeError(
                        f"cluster {cluster} local graph has {local_graph.shape[0]} "
                        f"rows against {size} members"
                    )
                if int(local_graph.min()) < 0 or int(local_graph.max()) >= size:
                    raise RuntimeError(
                        f"cluster {cluster} local graph holds an out-of-range id"
                    )

                started = time.perf_counter()
                vectors = cp.empty((size, dimension), dtype=cp.float32)
                for begin in range(0, size, A_CLUSTER_STAGE_ROWS):
                    end = min(begin + A_CLUSTER_STAGE_ROWS, size)
                    vectors[begin:end] = cp.asarray(
                        np.ascontiguousarray(chunk[begin:end], dtype=np.float32)
                    )
                cosines = np.empty(local_graph.shape, dtype=np.float32)
                edge_block = 65_536
                for begin in range(0, size, edge_block):
                    end = min(begin + edge_block, size)
                    neighbours = cp.asarray(local_graph[begin:end])
                    gathered = vectors[neighbours]
                    cosines[begin:end] = cp.asnumpy(
                        cp.einsum("bd,bkd->bk", vectors[begin:end], gathered)
                    )
                    del neighbours, gathered
                del vectors
                cp.get_default_memory_pool().free_all_blocks()
                cluster_cosine_seconds = time.perf_counter() - started
                cosine_seconds += cluster_cosine_seconds

                started = time.perf_counter()
                merge_block = 100_000
                for begin in range(0, size, merge_block):
                    end = min(begin + merge_block, size)
                    merge_into_topk(
                        top_ids,
                        top_cos,
                        rows=global_ids[begin:end],
                        candidate_ids=global_ids[local_graph[begin:end]],
                        candidate_cos=cosines[begin:end],
                        k=GRAPH_K,
                    )
                cluster_merge_seconds = time.perf_counter() - started
                merge_seconds += cluster_merge_seconds
                cluster_receipts.append({
                    "cluster": int(cluster),
                    "rows": size,
                    "brute_force": bool(brute),
                    "graph_shape": [int(value) for value in local_graph.shape],
                    "nn_descent_seconds": float(cluster_build_seconds),
                    "exact_cosine_seconds": float(cluster_cosine_seconds),
                    "merge_seconds": float(cluster_merge_seconds),
                })
                del chunk, local_graph, cosines
                os.remove(paths[cluster])
            for cluster in list(paths):
                if os.path.exists(paths[cluster]):
                    os.remove(paths[cluster])

        phase.update({
            "spill_write_seconds": spill_seconds,
            "nn_descent_seconds": build_seconds,
            "exact_cosine_seconds": cosine_seconds,
            "merge_seconds": merge_seconds,
        })
        degree = (top_ids >= 0).sum(axis=1)
        receipt.update({
            "fit": True,
            "oom": False,
            "timed_out": False,
            "refused_after_assignment": False,
            "builder_seconds": float(sum(phase.values())),
            "build_seconds": float(sum(phase.values())),
            "phases": phase,
            "clusters_built": len(cluster_receipts),
            "cluster_receipts": cluster_receipts,
            "graph_shape": [int(rows), GRAPH_K],
            "zero_degree_rows": int((degree == 0).sum()),
            "rows_below_k": int((degree < GRAPH_K).sum()),
            "min_degree": int(degree.min()),
            "rmm_peak_bytes": int(statistics.allocation_counts.peak_bytes),
            "child_device_peak_sampled_bytes": int(sampler.device_peak),
            "host_peak_sampled_bytes": int(sampler.host_peak),
            "host_vmhwm_bytes": _vmhwm_bytes(),
            "samples_taken": int(sampler.samples),
        })
        if bool(config.get("emit_graph")):
            np.save(os.path.join(args.out, "graph-k15-ids.i32.npy"), top_ids)
            np.save(
                os.path.join(args.out, "graph-k15-cos.f32.npy"),
                top_cos.astype(np.float32, copy=False),
            )
            receipt["graph_emitted"] = True
        else:
            receipt["graph_emitted"] = False
    except SystemExit:
        if "fit" not in receipt:
            raise
    except BaseException as exc:  # noqa: BLE001 - an OOM is a measurement here
        text = f"{type(exc).__name__}: {exc}".lower()
        is_oom = any(marker in text for marker in OOM_MARKERS)
        receipt.update({
            "fit": False,
            "oom": bool(is_oom),
            "timed_out": False,
            "refused_after_assignment": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback_tail": traceback.format_exc()[-2000:],
            "phases": phase,
            "rmm_peak_bytes": int(statistics.allocation_counts.peak_bytes),
            "child_device_peak_sampled_bytes": int(sampler.device_peak),
            "host_peak_sampled_bytes": int(sampler.host_peak),
            "host_vmhwm_bytes": _vmhwm_bytes(),
        })
        if not is_oom and not isinstance(exc, CooperativeAbort):
            sampler.halt()
            shutil.rmtree(scratch, ignore_errors=True)
            with open(
                os.path.join(args.out, "build-receipt.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True, default=str)
                handle.write("\n")
            raise
    finally:
        sampler.halt()
        shutil.rmtree(scratch, ignore_errors=True)

    with open(
        os.path.join(args.out, "build-receipt.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({
        "setting_id": receipt["setting_id"],
        "clusters": clusters,
        "spill": spill,
        "graph_degree": graph_degree,
        "intermediate_graph_degree": intermediate_degree,
        "max_iterations": max_iterations,
        "fit": receipt["fit"],
        "oom": receipt["oom"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
