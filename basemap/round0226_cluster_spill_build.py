#!/usr/bin/env python3
"""R0226 candidate A — cluster-spill nn-descent, out of core, exact merge.

Runs under the RAPIDS env (`/data/latent-basemap/cuml_py`), as a fresh process
per cell, guarded and watched by the parent.

    cuml_py basemap/round0226_cluster_spill_build.py --config c.json --out d

Paper 0197's out-of-core all-neighbours design, implemented by hand: local
`cuvs 25.02.01` ships **no** `cuvs.neighbors.all_neighbors` and **no**
`cuvs.cluster`, which this script re-verifies and records rather than assuming.

Phases, all sequential so the device peak is their maximum and not their sum:

1. **k-means** on a seeded uniform subsample (`cupy`, Lloyd, fixed iterations).
2. **Spilling assignment** — every row joins its `s` nearest centroids, computed
   in blocks straight off the substrate memmap. Neighbourhoods straddling a
   partition boundary survive in whichever cluster holds both endpoints.
3. **Capacity check** — if the largest realised cluster exceeds the registered
   capacity the cell refuses itself before any build starts. That is data.
4. **Per-cluster local graphs** — each cluster's rows are streamed into a real
   `.npy` on scratch and handed to `nn_descent` as a **memmap**, never as a
   resident array (R0224's lever: `0.63 GiB` non-RMM device against `3.38 GiB`
   at matched N = 2M). Scratch is written in groups so peak disk stays inside a
   registered budget; that is the part that has to generalise to 100M, where a
   single-pass spill would be `307 GB` against `292 GB` free.
5. **Exact cosines** for every local edge, recomputed on the device from the
   cluster's own vectors.
6. **Exact incremental global top-k merge**, id `-1` excluded and never ranked.

An OOM is a measurement: it is caught, written as `fit: false`, and exit is 0 so
the ladder can record where the builder stops.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import sys
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemap.round0226_graph_builders import (  # noqa: E402
    A_ASSIGN_BLOCK,
    A_CLUSTER_STAGE_ROWS,
    A_CLUSTER_CAPACITY_ROWS,
    A_GRAPH_DEGREE,
    A_INTERMEDIATE_DEGREE,
    A_KMEANS_ITERATIONS,
    A_KMEANS_SUBSAMPLE_ROWS,
    A_MAX_ITERATIONS,
    A_METRIC,
    A_SEED,
    A_SPILL,
    BUILD_SCHEMA,
    CANDIDATE_A,
    GRAPH_K,
    a_cluster_count,
    a_spill_groups,
    merge_into_topk,
)


class CooperativeAbort(RuntimeError):
    """The parent watchdog asked this build to stop."""


def _install_sigterm_handler() -> None:
    """Turn SIGTERM into a Python exception so CUDA can unwind normally.

    SIGKILL on a process inside a CUDA/UVM call is what deadlocked RCU, put PID
    1 into `D` state and cost a hard reboot in R0224's first attempt. The parent
    never sends one while this path exists.
    """

    def _handler(signum: int, _frame: object) -> None:
        raise CooperativeAbort(
            f"aborted by parent watchdog (signal {signum}); unwinding so CUDA "
            "can tear the context down cleanly"
        )

    signal.signal(signal.SIGTERM, _handler)


OOM_MARKERS = (
    "out_of_memory",
    "bad_alloc",
    "cudaerrormemoryallocation",
    "out of memory",
    "cannot allocate",
)


def _vmhwm_bytes() -> int:
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return -1


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return 0


class Sampler(threading.Thread):
    """In-child device and host peaks. Corroboration, not the budget instrument.

    `nn_descent.build` holds the GIL for the whole build, so this thread is
    starved during phase 4 exactly as it was in R0224 (1-2 samples per build).
    `samples_taken` is published so the starvation is visible rather than
    implied, and the budget number comes from the parent instead.
    """

    def __init__(self, cupy_module, interval_s: float) -> None:
        super().__init__(daemon=True)
        self._cp = cupy_module
        self._interval = float(interval_s)
        self._halt = threading.Event()
        self.device_peak = 0
        self.host_peak = 0
        self.samples = 0

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                free, total = self._cp.cuda.runtime.memGetInfo()
                self.device_peak = max(self.device_peak, int(total) - int(free))
            except Exception:  # noqa: BLE001 - a sampler must never kill a build
                pass
            self.host_peak = max(self.host_peak, _rss_bytes())
            self.samples += 1
            self._halt.wait(self._interval)

    def halt(self) -> None:
        self._halt.set()
        self.join(timeout=5)


def _kmeans(cp, cupyx, dataset: np.ndarray, *, clusters: int, seed: int):
    """Lloyd's algorithm on a seeded uniform subsample, on the device."""
    rows = int(dataset.shape[0])
    take = min(rows, A_KMEANS_SUBSAMPLE_ROWS)
    rng = np.random.default_rng(seed)
    sample_rows = np.sort(rng.choice(rows, size=take, replace=False))
    sample = cp.asarray(np.ascontiguousarray(dataset[sample_rows], dtype=np.float32))
    start = np.sort(rng.choice(take, size=clusters, replace=False))
    centroids = sample[cp.asarray(start)].copy()
    block = 100_000
    for _ in range(A_KMEANS_ITERATIONS):
        sums = cp.zeros((clusters, sample.shape[1]), dtype=cp.float32)
        counts = cp.zeros(clusters, dtype=cp.float32)
        for begin in range(0, take, block):
            end = min(begin + block, take)
            chunk = sample[begin:end]
            # Unit rows, so max inner product is min squared euclidean.
            nearest = cp.argmax(chunk @ centroids.T, axis=1)
            cupyx.scatter_add(sums, nearest, chunk)
            cupyx.scatter_add(counts, nearest, cp.ones(end - begin, dtype=cp.float32))
            del chunk, nearest
        empty = counts == 0
        safe = cp.where(empty, cp.float32(1.0), counts)
        updated = sums / safe[:, None]
        # An emptied centroid keeps its old position rather than collapsing.
        centroids = cp.where(empty[:, None], centroids, updated)
        norms = cp.linalg.norm(centroids, axis=1, keepdims=True)
        centroids = centroids / cp.maximum(norms, cp.float32(1e-12))
        del sums, counts, safe, updated, empty, norms
    del sample
    cp.get_default_memory_pool().free_all_blocks()
    return centroids


def _assign(cp, dataset: np.ndarray, centroids, *, rows: int, spill: int) -> np.ndarray:
    """Every row's `spill` nearest centroids, computed off the memmap in blocks."""
    out = np.empty((rows, spill), dtype=np.int32)
    for begin in range(0, rows, A_ASSIGN_BLOCK):
        end = min(begin + A_ASSIGN_BLOCK, rows)
        chunk = cp.asarray(np.ascontiguousarray(dataset[begin:end], dtype=np.float32))
        scores = chunk @ centroids.T
        order = cp.argsort(-scores, axis=1)[:, :spill]
        out[begin:end] = cp.asnumpy(order).astype(np.int32, copy=False)
        del chunk, scores, order
    cp.get_default_memory_pool().free_all_blocks()
    return out



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
    clusters = a_cluster_count(rows)
    groups = a_spill_groups(rows)
    receipt = {
        "schema": BUILD_SCHEMA,
        "candidate": CANDIDATE_A,
        "setting_id": str(config["setting_id"]),
        "config": config,
        "rows": rows,
        "dimension": dimension,
        "k": GRAPH_K,
        "clusters": clusters,
        "spill": A_SPILL,
        "spill_groups": groups,
        "cluster_capacity_rows": A_CLUSTER_CAPACITY_ROWS,
        "graph_degree": A_GRAPH_DEGREE,
        "intermediate_graph_degree": A_INTERMEDIATE_DEGREE,
        "max_iterations": A_MAX_ITERATIONS,
        "metric": A_METRIC,
        "cuvs_version": str(cuvs.__version__),
        "cuvs_has_all_neighbors": bool(has_all_neighbors),
        "cuvs_all_neighbors_note": (
            "cuvs.neighbors.all_neighbors is the out-of-core builder of paper "
            "0197. It is absent from this cuVS, so the design is implemented "
            "by hand here; the flag records the check rather than assuming it."
        ),
        "rmm_resource_type": resource_type,
        "rmm_resource_type_after_cuvs_import": resource_after,
        "rmm_resource_replaced_by_cuvs": bool(resource_after != resource_type),
        "warmup_seconds": warmup_seconds,
        "device_total_bytes": int(total),
        "device_baseline_bytes": int(total) - int(free),
        "host_rss_baseline_bytes": _rss_bytes(),
        "rmm_baseline_peak_bytes": int(statistics.allocation_counts.peak_bytes),
        "sample_interval_s": interval,
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
            },
        })
        if largest > A_CLUSTER_CAPACITY_ROWS:
            # Refused after assignment, before any per-cluster build. Nothing
            # heavy has been launched, so nothing can swap or need killing.
            receipt.update({
                "fit": False,
                "oom": False,
                "timed_out": False,
                "refused_after_assignment": True,
                "error_type": "ClusterCapacityExceeded",
                "error": (
                    f"largest realised cluster {largest} exceeds the registered "
                    f"capacity {A_CLUSTER_CAPACITY_ROWS}"
                ),
                "phases": phase,
                "rmm_peak_bytes": int(statistics.allocation_counts.peak_bytes),
                "host_vmhwm_bytes": _vmhwm_bytes(),
            })
            raise SystemExit(0)

        top_ids = np.full((rows, GRAPH_K), -1, dtype=np.int32)
        top_cos = np.full((rows, GRAPH_K), -np.inf, dtype=np.float32)

        # Cluster membership as a CSR-ish index, built once.
        flat_cluster = assignment.ravel().astype(np.int64)
        flat_row = np.repeat(np.arange(rows, dtype=np.int64), A_SPILL)
        order = np.argsort(flat_cluster, kind="stable")
        members = flat_row[order]
        bounds = np.zeros(clusters + 1, dtype=np.int64)
        np.cumsum(sizes, out=bounds[1:])
        del flat_cluster, flat_row, order, assignment

        per_group = int(np.ceil(clusters / float(groups)))
        build_seconds = 0.0
        merge_seconds = 0.0
        spill_seconds = 0.0
        cosine_seconds = 0.0
        cluster_receipts: list[dict[str, object]] = []

        for group_index in range(groups):
            first = group_index * per_group
            last = min(first + per_group, clusters)
            if first >= last:
                continue
            started = time.perf_counter()
            handles: dict[int, np.ndarray] = {}
            cursors: dict[int, int] = {}
            paths: dict[int, str] = {}
            for cluster in range(first, last):
                size = int(sizes[cluster])
                if size == 0:
                    continue
                path = os.path.join(scratch, f"cluster-{cluster:05d}.f32.npy")
                paths[cluster] = path
                handles[cluster] = np.lib.format.open_memmap(
                    path, mode="w+", dtype=np.float32, shape=(size, dimension)
                )
                cursors[cluster] = 0
            # One pass over the substrate per group; rows for the group's
            # clusters are appended contiguously in row order.
            member_of = {
                cluster: members[bounds[cluster] : bounds[cluster + 1]]
                for cluster in range(first, last)
            }
            position = {
                cluster: 0 for cluster in range(first, last)
            }
            for begin in range(0, rows, A_ASSIGN_BLOCK):
                end = min(begin + A_ASSIGN_BLOCK, rows)
                block = np.ascontiguousarray(dataset[begin:end], dtype=np.float32)
                for cluster in range(first, last):
                    if cluster not in handles:
                        continue
                    ids = member_of[cluster]
                    start_at = position[cluster]
                    stop_at = np.searchsorted(ids, end, side="left")
                    if stop_at <= start_at:
                        continue
                    take = ids[start_at:stop_at]
                    cursor = cursors[cluster]
                    handles[cluster][cursor : cursor + take.size] = block[take - begin]
                    cursors[cluster] = cursor + int(take.size)
                    position[cluster] = int(stop_at)
                del block
            for cluster in list(handles):
                handles[cluster].flush()
                del handles[cluster]
            handles.clear()
            spill_seconds += time.perf_counter() - started

            for cluster in range(first, last):
                if cluster not in paths:
                    continue
                size = int(sizes[cluster])
                global_ids = members[bounds[cluster] : bounds[cluster + 1]]
                chunk = np.load(paths[cluster], mmap_mode="r")
                started = time.perf_counter()
                # A cluster too small for nn-descent's own working set is done
                # exactly by brute force instead; it is cheap at this size and
                # exactness is never traded for uniformity of code path.
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
                build_seconds += time.perf_counter() - started
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
                cosine_seconds += time.perf_counter() - started

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
                merge_seconds += time.perf_counter() - started
                cluster_receipts.append({
                    "cluster": int(cluster),
                    "rows": size,
                    "graph_shape": [int(x) for x in local_graph.shape],
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
        "fit": receipt["fit"],
        "oom": receipt["oom"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
