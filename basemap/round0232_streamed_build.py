#!/usr/bin/env python3
"""R0232 — R0229's `cluster-spill-nnd` builder with the spill made a variable.

Runs under the RAPIDS env (`/data/latent-basemap/cuml_py`), one fresh process per
cell, guarded and watched by the parent:

    cuml_py basemap/round0232_streamed_build.py --config c.json --out d

This is R0229's build script with the scratch behaviour parameterised and
instrumented, and **nothing about the science moved**. The k-means recipe, the
assignment, the whole-cluster packing, the per-cluster nn-descent, the exact
fp32 cosine recompute, the exact incremental global top-k merge, the capacity
refusal, the cooperative abort and the OOM-as-measurement discipline are R0226's
and R0227's, imported and called unmodified.

Three changes:

1. **`mode` comes from the config.**

   * `materialise` — R0229's behaviour exactly: pack whole clusters into groups
     against a scratch bound, write one group's clusters to `/data`, consume and
     `os.remove` each the instant its local graph is merged.
   * `stream-resident` — the same grouping with the spill file replaced by a host
     buffer of the same bound. Nothing is written to disk.
   * `stream-gather` — no grouping at all. Each cluster's rows are taken by a
     direct ascending index gather from the substrate memmap, one cluster
     resident at a time. Nothing is written to disk and the substrate is never
     swept.

   In every mode the clusters are visited in the identical order and the merge
   consumes the identical bytes, so the merged graph should be byte-identical.
   Whether it actually is, is measured rather than assumed.

2. **`bound_bytes` comes from the config** instead of `SCRATCH_BUDGET_BYTES`, so
   the peak-scratch law can be measured against the bound rather than at one
   point.

3. **The scratch is measured, not modelled.** A sampler thread walks the cell's
   scratch directory every 50 ms and records the true allocated bytes
   (`st_blocks x 512`), `/proc/self/io` counters bracket the whole build, and the
   sampler cooperatively aborts the cell if measured scratch runs past its own
   bound by more than one cluster plus the registered slack. Every prior round
   published `peak_scratch_bytes` computed from `sizes` before a byte was
   written; none of them ever looked at the disk.

An OOM, a capacity refusal, a scratch breach and a cooperative abort are all
measurements: they are caught, written as `fit: false`, and exit is 0 so the grid
records where the configuration stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
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
from basemap.round0232_scratch_contract import (  # noqa: E402
    BUILD_SCHEMA,
    MODES,
    MODE_MATERIALISE,
    MODE_STREAM_GATHER,
    MODE_STREAM_RESIDENT,
    SCRATCH_ABORT_SLACK_BYTES,
    SCRATCH_SAMPLE_INTERVAL_S,
)


class ScratchBreach(RuntimeError):
    """Measured on-disk scratch ran past this cell's own bound."""


def _directory_allocated_bytes(root: str) -> int:
    """True allocated bytes under `root`, from `st_blocks`, never from `st_size`.

    `np.lib.format.open_memmap(mode="w+")` truncates the file to its full length
    immediately, so `st_size` would report the whole cluster as resident before a
    single byte had been written. `st_blocks` reports what the filesystem has
    actually allocated, which is the quantity a disk budget is about.
    """
    total = 0
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                            continue
                        total += int(entry.stat(follow_symlinks=False).st_blocks) * 512
                    except (FileNotFoundError, PermissionError):
                        continue
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
    return total


class ScratchSampler(threading.Thread):
    """The instrument no prior round had: the disk, watched while the build runs."""

    def __init__(self, root: str, *, interval_s: float, abort_above_bytes: int) -> None:
        super().__init__(daemon=True)
        self._root = root
        self._interval = float(interval_s)
        self._abort_above = int(abort_above_bytes)
        self._halt = threading.Event()
        self.peak_bytes = 0
        self.samples = 0
        self.breached = False
        self.breach_bytes = 0

    def run(self) -> None:
        while not self._halt.is_set():
            current = _directory_allocated_bytes(self._root)
            if current > self.peak_bytes:
                self.peak_bytes = current
            self.samples += 1
            if self._abort_above > 0 and current > self._abort_above and not self.breached:
                self.breached = True
                self.breach_bytes = current
            self._halt.wait(self._interval)

    def halt(self) -> None:
        self._halt.set()
        self.join(timeout=5)
        current = _directory_allocated_bytes(self._root)
        if current > self.peak_bytes:
            self.peak_bytes = current


def _proc_io() -> dict[str, int]:
    """Block-layer and syscall byte counters for this process."""
    out: dict[str, int] = {}
    try:
        with open("/proc/self/io", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                out[key.strip()] = int(value.strip())
    except OSError:
        pass
    return out


def _io_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {
        key: int(after.get(key, 0)) - int(before.get(key, 0))
        for key in sorted(set(before) | set(after))
    }


def _fadvise_dontneed(path: str) -> bool:
    """Evict this file's clean pages so a read measurement is a disk measurement."""
    try:
        handle = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        os.posix_fadvise(handle, 0, 0, os.POSIX_FADV_DONTNEED)
        return True
    except (AttributeError, OSError):
        return False
    finally:
        os.close(handle)


def _measure_data_throughput(
    substrate_path: str, probe_root: str, *, read_bytes: int, write_bytes: int
) -> dict[str, object]:
    """Cold sequential read and fsync'd sequential write on `/data`, measured here.

    review-0226-01's `5.53`/`6.36 GB/s` are carried in the codebase as prose. The
    I/O line of this round's projection is the whole tradeoff of a streaming
    design, so the rate it uses is measured on this box in this round.
    """
    os.makedirs(probe_root, exist_ok=True)
    evicted = _fadvise_dontneed(substrate_path)
    block = 64 * 1024 * 1024
    read_total = 0
    started = time.perf_counter()
    with open(substrate_path, "rb") as handle:
        while read_total < read_bytes:
            chunk = handle.read(min(block, read_bytes - read_total))
            if not chunk:
                break
            read_total += len(chunk)
    read_seconds = time.perf_counter() - started

    payload = os.urandom(block)
    target = os.path.join(probe_root, "throughput-probe.bin")
    written = 0
    started = time.perf_counter()
    with open(target, "wb") as handle:
        while written < write_bytes:
            take = min(block, write_bytes - written)
            handle.write(payload[:take])
            written += take
        handle.flush()
        os.fsync(handle.fileno())
    write_seconds = time.perf_counter() - started
    os.remove(target)
    try:
        os.rmdir(probe_root)
    except OSError:
        pass
    return {
        "substrate_pages_evicted_before_read": bool(evicted),
        "read_bytes": int(read_total),
        "read_seconds": read_seconds,
        "read_bytes_per_s": (read_total / read_seconds) if read_seconds > 0 else None,
        "write_bytes": int(written),
        "write_seconds": write_seconds,
        "write_bytes_per_s": (written / write_seconds) if write_seconds > 0 else None,
        "note": (
            "cold sequential read after posix_fadvise(DONTNEED) over the substrate, "
            "and an fsync'd sequential write into the cell's scratch root. Both "
            "measured on gsv:/data in this round rather than carried from "
            "review-0226-01's prose."
        ),
    }


def cluster_membership(
    assignment: np.ndarray, sizes: np.ndarray, *, rows: int, spill: int
) -> tuple[np.ndarray, np.ndarray]:
    """R0226's membership table: global row ids grouped by cluster, ascending.

    Factored out of the build loop so the two data paths below can be shown to
    read the identical rows in the identical order on a CPU, before any GPU time
    is spent on the claim.
    """
    flat_cluster = assignment.ravel().astype(np.int64)
    flat_row = np.repeat(np.arange(rows, dtype=np.int64), spill)
    order = np.argsort(flat_cluster, kind="stable")
    members = flat_row[order]
    bounds = np.zeros(int(sizes.shape[0]) + 1, dtype=np.int64)
    np.cumsum(sizes, out=bounds[1:])
    return members, bounds


def fill_group_by_sweep(
    dataset, group, *, sizes: np.ndarray, members: np.ndarray, bounds: np.ndarray,
    rows: int, dimension: int, allocate,
) -> dict[int, np.ndarray]:
    """One sequential block sweep over the substrate, scattering into buffers.

    `allocate(cluster, size)` returns the destination — a `w+` memmap under
    `materialise`, a host array under `stream-resident`. The sweep itself is
    R0226's, unchanged; only where it writes differs.
    """
    handles: dict[int, np.ndarray] = {}
    cursors: dict[int, int] = {}
    member_of: dict[int, np.ndarray] = {}
    position: dict[int, int] = {}
    for cluster in group:
        handles[cluster] = allocate(cluster, int(sizes[cluster]))
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
    return handles


def gather_cluster(dataset, member_ids: np.ndarray) -> np.ndarray:
    """The whole of `stream-gather`: one cluster's rows, straight off the memmap.

    `member_ids` is ascending within a cluster by construction of
    `cluster_membership`, so this is a monotone strided read rather than a random
    one. Nothing is written and only this cluster is resident.
    """
    return np.ascontiguousarray(dataset[member_ids], dtype=np.float32)


def _load_or_build_assignment(
    cp, cupyx, dataset, *, rows: int, clusters: int, spill: int,
    cache_path: str | None, phase: dict,
) -> tuple[np.ndarray, dict]:
    """R0229's shared partition, verbatim. Cached bytes bind; never regenerated."""
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
    mode = str(config["mode"])
    bound_bytes = int(config.get("bound_bytes") or 0)
    if mode not in MODES:
        raise SystemExit(f"mode {mode!r} is not one of {MODES}")
    if mode in (MODE_MATERIALISE, MODE_STREAM_RESIDENT) and bound_bytes <= 0:
        raise SystemExit(f"mode {mode!r} needs a positive bound_bytes")
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
        "mode": mode,
        "bound_bytes": bound_bytes,
        "cluster_capacity_rows": CLUSTER_CAPACITY_ROWS,
        "registered_scratch_budget_bytes": SCRATCH_BUDGET_BYTES,
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
        "rmm_baseline_peak_bytes": int(statistics.allocation_counts.peak_bytes),
        "sample_interval_s": interval,
        "scratch_sample_interval_s": SCRATCH_SAMPLE_INTERVAL_S,
        "spill_volume_bytes": int(rows) * int(spill) * dimension * 4,
        "parameters_note": (
            "R0229's builder with the spill parameterised: mode in "
            "{materialise, stream-resident, stream-gather} and the scratch bound "
            "from the config. Every other parameter is R0226's/R0227's, imported "
            "and called unmodified, and the cluster visit order is identical in "
            "all three modes."
        ),
    }

    substrate_path = str(config["substrate"])
    scratch = os.path.join(scratch_root, str(config["setting_id"]))
    if os.path.isdir(scratch):
        shutil.rmtree(scratch)
    os.makedirs(scratch, exist_ok=True)

    if bool(config.get("measure_data_throughput")):
        receipt["data_throughput"] = _measure_data_throughput(
            substrate_path,
            os.path.join(scratch_root, f"{config['setting_id']}-throughput"),
            read_bytes=int(config.get("throughput_read_bytes") or 4 * 1024 ** 3),
            write_bytes=int(config.get("throughput_write_bytes") or 2 * 1024 ** 3),
        )
    if bool(config.get("fadvise_dontneed_substrate")):
        receipt["substrate_pages_evicted"] = _fadvise_dontneed(substrate_path)

    io_before = _proc_io()
    sampler = Sampler(cp, interval)
    sampler.start()
    scratch_sampler: ScratchSampler | None = None
    phase: dict[str, float] = {}
    try:
        substrate = np.load(substrate_path, mmap_mode="r")
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
        largest_cluster_bytes = int(largest) * dimension * 4
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
            "largest_cluster_bytes": largest_cluster_bytes,
        })
        if largest > CLUSTER_CAPACITY_ROWS:
            receipt.update({
                "fit": False, "oom": False, "timed_out": False,
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
                "measured_peak_scratch_bytes": 0,
            })
            raise SystemExit(0)

        top_ids = np.full((rows, GRAPH_K), -1, dtype=np.int32)
        top_cos = np.full((rows, GRAPH_K), -np.inf, dtype=np.float32)

        members, bounds = cluster_membership(
            assignment, sizes, rows=rows, spill=spill
        )
        del assignment

        if mode == MODE_STREAM_GATHER:
            groups = [[index] for index, size in enumerate(sizes) if int(size) > 0]
        else:
            groups = pack_clusters_into_groups(sizes, budget_bytes=bound_bytes)
        group_bytes = [
            int(sum(int(sizes[index]) for index in group) * dimension * 4)
            for group in groups
        ]
        modelled_peak = (
            0 if mode in (MODE_STREAM_RESIDENT, MODE_STREAM_GATHER)
            else int(max(group_bytes) if group_bytes else 0)
        )
        modelled_resident_host = (
            0 if mode == MODE_MATERIALISE
            else int(max(group_bytes) if group_bytes else 0)
        )
        passes = 0 if mode == MODE_STREAM_GATHER else len(groups)
        receipt.update({
            "spill_groups": len(groups),
            "spill_group_cluster_counts": [len(group) for group in groups],
            "spill_group_bytes": group_bytes,
            "modelled_peak_scratch_bytes": modelled_peak,
            "modelled_peak_resident_host_bytes": modelled_resident_host,
            "substrate_passes": passes,
            "substrate_read_bytes": int(passes * rows * dimension * 4),
            "spill_write_bytes": (
                int(int(sizes.sum()) * dimension * 4)
                if mode == MODE_MATERIALISE else 0
            ),
            "gathered_row_bytes": (
                int(int(sizes.sum()) * dimension * 4)
                if mode == MODE_STREAM_GATHER else 0
            ),
            "bound_exceeded_by_single_cluster": bool(
                mode != MODE_STREAM_GATHER and group_bytes
                and max(group_bytes) > bound_bytes
            ),
        })

        abort_above = (
            0 if mode != MODE_MATERIALISE
            else bound_bytes + largest_cluster_bytes + SCRATCH_ABORT_SLACK_BYTES
        )
        if mode != MODE_MATERIALISE:
            # A streamed cell must not write at all. One MiB is not a tolerance
            # for spilling — it is the smallest threshold that cannot be tripped
            # by an incidental file — and the published check is `measured == 0`.
            abort_above = 1024 ** 2
        scratch_sampler = ScratchSampler(
            scratch, interval_s=SCRATCH_SAMPLE_INTERVAL_S,
            abort_above_bytes=abort_above,
        )
        scratch_sampler.start()
        receipt["scratch_abort_above_bytes"] = int(abort_above)

        build_seconds = 0.0
        merge_seconds = 0.0
        spill_seconds = 0.0
        cosine_seconds = 0.0
        gather_seconds = 0.0
        cluster_receipts: list[dict[str, object]] = []

        for group in groups:
            if scratch_sampler.breached:
                raise ScratchBreach(
                    f"measured scratch {scratch_sampler.breach_bytes} exceeded "
                    f"{abort_above} for mode {mode}"
                )
            paths: dict[int, str] = {}
            buffers: dict[int, np.ndarray] = {}
            if mode in (MODE_MATERIALISE, MODE_STREAM_RESIDENT):
                started = time.perf_counter()

                def allocate(cluster: int, size: int) -> np.ndarray:
                    if mode == MODE_MATERIALISE:
                        path = os.path.join(scratch, f"cluster-{cluster:05d}.f32.npy")
                        paths[cluster] = path
                        return np.lib.format.open_memmap(
                            path, mode="w+", dtype=np.float32, shape=(size, dimension)
                        )
                    return np.empty((size, dimension), dtype=np.float32)

                handles = fill_group_by_sweep(
                    dataset, group, sizes=sizes, members=members, bounds=bounds,
                    rows=rows, dimension=dimension, allocate=allocate,
                )
                if mode == MODE_MATERIALISE:
                    for cluster in list(handles):
                        handles[cluster].flush()
                        del handles[cluster]
                    handles.clear()
                else:
                    buffers = handles
                spill_seconds += time.perf_counter() - started

            for cluster in group:
                if scratch_sampler.breached:
                    raise ScratchBreach(
                        f"measured scratch {scratch_sampler.breach_bytes} exceeded "
                        f"{abort_above} for mode {mode}"
                    )
                size = int(sizes[cluster])
                global_ids = members[bounds[cluster] : bounds[cluster + 1]]
                if mode == MODE_MATERIALISE:
                    chunk = np.load(paths[cluster], mmap_mode="r")
                elif mode == MODE_STREAM_RESIDENT:
                    chunk = buffers[cluster]
                else:
                    started = time.perf_counter()
                    chunk = gather_cluster(dataset, global_ids)
                    gather_seconds += time.perf_counter() - started
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
                        top_ids, top_cos,
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
                if mode == MODE_MATERIALISE:
                    os.remove(paths[cluster])
                elif mode == MODE_STREAM_RESIDENT:
                    buffers.pop(cluster, None)
            for cluster in list(paths):
                if os.path.exists(paths[cluster]):
                    os.remove(paths[cluster])
            buffers.clear()

        phase.update({
            "spill_write_seconds": spill_seconds,
            "gather_seconds": gather_seconds,
            "nn_descent_seconds": build_seconds,
            "exact_cosine_seconds": cosine_seconds,
            "merge_seconds": merge_seconds,
        })
        degree = (top_ids >= 0).sum(axis=1)
        scratch_sampler.halt()
        receipt.update({
            "fit": True, "oom": False, "timed_out": False,
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
            "measured_peak_scratch_bytes": int(scratch_sampler.peak_bytes),
            "scratch_samples_taken": int(scratch_sampler.samples),
            "scratch_breached": bool(scratch_sampler.breached),
            "graph_ids_sha256": hashlib.sha256(top_ids.tobytes()).hexdigest(),
            "graph_cos_sha256": hashlib.sha256(top_cos.tobytes()).hexdigest(),
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
        if scratch_sampler is not None:
            scratch_sampler.halt()
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
            "measured_peak_scratch_bytes": (
                int(scratch_sampler.peak_bytes) if scratch_sampler is not None else 0
            ),
            "scratch_breached": bool(
                scratch_sampler.breached if scratch_sampler is not None else False
            ),
        })
        if not is_oom and not isinstance(exc, (CooperativeAbort, ScratchBreach)):
            sampler.halt()
            shutil.rmtree(scratch, ignore_errors=True)
            receipt["proc_io_delta"] = _io_delta(io_before, _proc_io())
            with open(
                os.path.join(args.out, "build-receipt.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True, default=str)
                handle.write("\n")
            raise
    finally:
        sampler.halt()
        if scratch_sampler is not None:
            scratch_sampler.halt()
        shutil.rmtree(scratch, ignore_errors=True)

    receipt["proc_io_delta"] = _io_delta(io_before, _proc_io())
    receipt["proc_io_note"] = (
        "read_bytes/write_bytes are block-layer transfers for this process; "
        "rchar/wchar are syscall bytes and include page-cache hits. The 2M "
        "substrate is 3.07 GB and sits in page cache, so read_bytes at 2M "
        "measures the cache, not the disk. The 8M cells evict the substrate with "
        "posix_fadvise(DONTNEED) first and are the disk measurement."
    )
    with open(
        os.path.join(args.out, "build-receipt.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({
        "setting_id": receipt["setting_id"],
        "mode": mode,
        "clusters": clusters,
        "spill": spill,
        "fit": receipt["fit"],
        "oom": receipt["oom"],
        "measured_peak_scratch_bytes": receipt.get("measured_peak_scratch_bytes"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
