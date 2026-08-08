#!/usr/bin/env python3
"""R0227 — R0226's `cluster-spill-nnd` builder with the cluster count set freely.

Runs under the RAPIDS env (`/data/latent-basemap/cuml_py`), one fresh process per
cell, guarded and watched by the parent:

    cuml_py basemap/round0227_cluster_spill_build.py --config c.json --out d

**Every scientific parameter is R0226's, imported rather than restated**: the
spill factor, the k-means recipe (Lloyd, seed 226, 25 iterations, 1,000,000-row
subsample), the nn-descent setting (`graph_degree 32`, `intermediate 48`,
`max_iterations 20`, `sqeuclidean`), the exact-cosine recompute and the exact
incremental global top-k merge. The k-means, assignment and sampler routines are
imported from R0226's build module and called unmodified.

Exactly two things differ, and both follow from review-0226-01:

1. **`clusters` comes from the config**, instead of `c = max(8, ceil(N*s/1e6))`.
   That law puts 100M at `c = 200`, where the review measured A's structural
   reachability ceiling at `0.867`; the whole point of this round is that the
   card has ~25 GiB spare and `c` is what spends it.
2. **Spill groups pack whole clusters against a byte budget** instead of
   splitting `c` into equal-sized groups. At low `c` the equal-split rule stops
   bounding anything — two 12.5M-row clusters in one group is 38 GB against a
   24 GiB budget — and the realised group count is the number of substrate
   re-reads, which is the term the 100M I/O projection is made of. It is
   reported per cell rather than assumed.

Per-cluster nn-descent seconds are recorded individually so the 100M wall
projection can be built from a cost curve in *cluster rows* rather than from a
whole-builder power law that never sampled the regime it extrapolates into.

An OOM, a capacity refusal and a cooperative abort are all measurements: they
are caught, written as `fit: false`, and exit is 0 so the ladder records where
the configuration stops.
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
    A_GRAPH_DEGREE,
    A_INTERMEDIATE_DEGREE,
    A_KMEANS_ITERATIONS,
    A_KMEANS_SUBSAMPLE_ROWS,
    A_MAX_ITERATIONS,
    A_METRIC,
    A_SEED,
    A_SPILL,
    GRAPH_K,
    merge_into_topk,
)
from basemap.round0227_low_c_contract import (  # noqa: E402
    BUILD_SCHEMA,
    CANDIDATE,
    CLUSTER_CAPACITY_ROWS,
    SCRATCH_BUDGET_BYTES,
    pack_clusters_into_groups,
)


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
    dimension = int(config["dimension"])
    scratch_root = str(config["scratch_root"])
    interval = float(config.get("sample_interval_s", 0.005))

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
        "config": config,
        "rows": rows,
        "dimension": dimension,
        "k": GRAPH_K,
        "clusters": clusters,
        "spill": A_SPILL,
        "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
        "scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
        "kmeans_subsample_rows": A_KMEANS_SUBSAMPLE_ROWS,
        "kmeans_iterations": A_KMEANS_ITERATIONS,
        "graph_degree": A_GRAPH_DEGREE,
        "intermediate_graph_degree": A_INTERMEDIATE_DEGREE,
        "max_iterations": A_MAX_ITERATIONS,
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
            "every parameter except the cluster count and the spill-group "
            "packing is R0226's, imported from its modules and called unmodified"
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

        started = time.perf_counter()
        centroids = _kmeans(cp, cupyx, dataset, clusters=clusters, seed=A_SEED)
        phase["kmeans_seconds"] = time.perf_counter() - started

        started = time.perf_counter()
        assignment = _assign(cp, dataset, centroids, rows=rows, spill=A_SPILL)
        phase["assign_seconds"] = time.perf_counter() - started
        del centroids
        cp.get_default_memory_pool().free_all_blocks()

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
            # Refused after assignment and before any per-cluster build: nothing
            # heavy has been launched, so nothing can swap or need terminating.
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
        flat_row = np.repeat(np.arange(rows, dtype=np.int64), A_SPILL)
        order = np.argsort(flat_cluster, kind="stable")
        members = flat_row[order]
        bounds = np.zeros(clusters + 1, dtype=np.int64)
        np.cumsum(sizes, out=bounds[1:])
        del flat_cluster, flat_row, order, assignment

        # Whole clusters packed against the scratch budget. The group count is
        # the number of substrate passes and it is an output of this round.
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
            # One pass over the substrate per group; rows for the group's
            # clusters are appended contiguously in row order.
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
                # A cluster too small for nn-descent's own working set is done
                # exactly by brute force; exactness is never traded for
                # uniformity of code path.
                brute = size <= 2 * A_INTERMEDIATE_DEGREE
                if brute:
                    small = cp.asarray(np.ascontiguousarray(chunk, dtype=np.float32))
                    width = min(A_GRAPH_DEGREE, max(1, size - 1))
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
                            graph_degree=A_GRAPH_DEGREE,
                            intermediate_graph_degree=A_INTERMEDIATE_DEGREE,
                            max_iterations=A_MAX_ITERATIONS,
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
        "fit": receipt["fit"],
        "oom": receipt["oom"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
