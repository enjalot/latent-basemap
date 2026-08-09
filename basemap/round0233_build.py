#!/usr/bin/env python3
"""R0233 — `cluster-spill-nnd` at `s = 8` with an IN-BAND cooperative abort.

Runs under the RAPIDS env (`/data/latent-basemap/cuml_py`), one fresh process per
cell, guarded and watched by the parent:

    cuml_py basemap/round0233_build.py --config c.json --out d

This is R0229's quality build script with exactly two changes, and nothing else
moves:

1. **The abort path is a flag file, not a signal.** R0229 (via R0226) installed a
   `SIGTERM` handler and the parent watchdog sent `SIGTERM` on a budget breach.
   Review-0232 located R0232's UVM deadlock in the **signal-triggered exit
   path**, so cooperative termination was itself sufficient to wedge the card.
   Here the parent writes `abort.flag` and this process polls it between spill
   groups and between clusters, then raises in-band so the interpreter unwinds
   and tears its own CUDA context down through the normal path. No signal
   handler is installed and the parent never sends one.
2. **Every buffer handed to cuVS is asserted to be a read-only `np.memmap`** —
   the substrate view and each per-cluster spill file, checked immediately before
   `nn_descent.build`. R0232 wedged the box with an anonymous `ndarray` and its
   round spec never forbade it, so this is a registered precondition rather than
   a convention.

Everything else is R0229's, imported and called unmodified: the k-means recipe
(Lloyd, seed 226, 25 iterations, 1,000,000-row subsample), the whole-cluster
spill-group packing against the scratch budget, the exact-cosine recompute, the
exact incremental global top-k merge, the realised-capacity refusal, and the
OOM-as-measurement discipline.

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
    SCRATCH_BUDGET_BYTES,
    pack_clusters_into_groups,
)
from basemap.round0233_substrate import (  # noqa: E402
    BUILD_SCHEMA,
    CANDIDATE,
    CLUSTER_CAPACITY_ROWS,
    assert_memmap_for_cuvs,
)

ABORT_FLAG_NAME = "abort.flag"


def _anon_bytes() -> int:
    """Anonymous (swappable) bytes. Never RSS — RSS counts clean page cache."""
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def _check_abort(flag_path: str | None, *, where: str) -> None:
    """Poll the parent's cooperative flag. In-band only; no signal is involved."""
    if flag_path and os.path.exists(flag_path):
        raise CooperativeAbort(
            f"parent set the cooperative abort flag (observed at {where}); "
            "unwinding in-band so this process tears its own CUDA context down"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

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
    abort_flag = config.get("abort_flag") or None
    emit_graph = bool(config.get("emit_graph"))
    if spill < 1:
        raise SystemExit(f"spill {spill} is not >= 1")
    if clusters <= spill:
        raise SystemExit(
            f"clusters {clusters} <= spill {spill}: every row would land in "
            "every cluster and nothing would be partitioned"
        )
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

    # R0227's warm-up at R0227's literals, so the warm-up cost is the same
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
        "rmm_resource_type": resource_type,
        "rmm_resource_type_after_cuvs_import": resource_after,
        "rmm_resource_replaced_by_cuvs": bool(resource_after != resource_type),
        "warmup_seconds": warmup_seconds,
        "device_total_bytes": int(total),
        "device_baseline_bytes": int(total) - int(free),
        "host_rss_baseline_bytes": _rss_bytes(),
        "host_anon_baseline_bytes": _anon_bytes(),
        "rmm_baseline_peak_bytes": int(statistics.allocation_counts.peak_bytes),
        "sample_interval_s": interval,
        "abort_mechanism": "in-band cooperative flag file; no signal handler",
        "signal_handler_installed": False,
        "cuvs_inputs_asserted_memmap": True,
        "igd_host_law_bytes_per_row": (
            2 * int(32 * -(-int(1.3 * intermediate_degree) // 32))
        ),
    }

    sampler = Sampler(cp, interval)
    sampler.start()
    scratch = os.path.join(scratch_root, str(config["setting_id"]))
    phase: dict[str, float] = {}
    anon_peak = 0
    try:
        if os.path.isdir(scratch):
            shutil.rmtree(scratch)
        os.makedirs(scratch, exist_ok=True)
        _check_abort(abort_flag, where="start")

        substrate = np.load(str(config["substrate"]), mmap_mode="r")
        if substrate.ndim != 2 or int(substrate.shape[1]) != dimension:
            raise SystemExit(f"substrate shape {substrate.shape} is not (*, {dimension})")
        if rows > int(substrate.shape[0]):
            raise SystemExit(f"requested {rows} rows, substrate has {substrate.shape[0]}")
        dataset = substrate[:rows]
        assert_memmap_for_cuvs(dataset, label="R0233 substrate view")

        started = time.perf_counter()
        centroids = _kmeans(cp, cupyx, dataset, clusters=clusters, seed=A_SEED)
        phase["kmeans_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        assignment = _assign(cp, dataset, centroids, rows=rows, spill=spill)
        phase["assign_seconds"] = time.perf_counter() - started
        del centroids
        cp.get_default_memory_pool().free_all_blocks()
        _check_abort(abort_flag, where="after assignment")

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
                "cooperatively_aborted": False,
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
                "host_anon_peak_bytes": max(anon_peak, _anon_bytes()),
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
            _check_abort(abort_flag, where="spill group start")
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
            anon_peak = max(anon_peak, _anon_bytes())

            for cluster in group:
                _check_abort(abort_flag, where=f"before cluster {cluster}")
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
                    # The registered safety precondition, checked on the exact
                    # object cuVS receives, immediately before it receives it.
                    assert_memmap_for_cuvs(
                        chunk, label=f"R0233 cluster {cluster} spill file"
                    )
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
                anon_peak = max(anon_peak, _anon_bytes())
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
            "cooperatively_aborted": False,
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
            "host_anon_peak_bytes": max(anon_peak, _anon_bytes()),
            "host_vmhwm_bytes": _vmhwm_bytes(),
            "samples_taken": int(sampler.samples),
        })
        if emit_graph:
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
        aborted = isinstance(exc, CooperativeAbort)
        receipt.update({
            "fit": False,
            "oom": bool(is_oom),
            "timed_out": False,
            "cooperatively_aborted": bool(aborted),
            "refused_after_assignment": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback_tail": traceback.format_exc()[-2000:],
            "phases": phase,
            "rmm_peak_bytes": int(statistics.allocation_counts.peak_bytes),
            "child_device_peak_sampled_bytes": int(sampler.device_peak),
            "host_peak_sampled_bytes": int(sampler.host_peak),
            "host_anon_peak_bytes": max(anon_peak, _anon_bytes()),
            "host_vmhwm_bytes": _vmhwm_bytes(),
        })
        if not is_oom and not aborted:
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
        "fit": receipt["fit"],
        "oom": receipt["oom"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
