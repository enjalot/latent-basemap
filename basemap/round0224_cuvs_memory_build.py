#!/usr/bin/env python3
"""Build one cuVS nn-descent graph under four memory instruments at once.

Runs under the RAPIDS env, not the release venv, as a **fresh process per
build** — that is what makes `VmHWM` a per-build measurement rather than a
running high-water mark contaminated by previous settings.

    /data/latent-basemap/cuml_py basemap/round0224_cuvs_memory_build.py \
        --config <config.json> --out <dir>

The four instruments, and what each can see:

1. **RMM statistics adaptor** — every allocation routed through RMM's current
   device resource. cuVS allocates through `rmm::mr::get_current_device_resource`,
   so if it allocates on the device at all, this sees it. Installed *before*
   `cuvs` is imported.
2. **Device-wide sampling** (`cudaMemGetInfo`, 5 ms) — total device bytes in
   use by anyone, RMM or not. Immune to allocator choice; this is the instrument
   that cannot be fooled by cuVS using its own pool.
3. **Host sampling** (`/proc/self/statm`, 5 ms) plus `VmHWM` — because
   RAFT/cuVS nn-descent holds its intermediate graph in **host** memory. If the
   intermediate degree costs anything, this is where it shows up.
4. The parent additionally polls `nvidia-smi` against this pid. That is R0220's
   instrument, kept deliberately as a **control**: the round's first job is to
   say which instruments are sensitive to intermediate degree and which are
   blind.

An out-of-memory failure is a *measurement*, not a crash: the script catches it,
writes `fit: false` with the failing instrument's last reading, and exits 0 so
the sweep can continue and report where the wall actually is.
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import traceback

import numpy as np


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


class Sampler:
    """Device-wide and host RSS peaks, sampled on a thread."""

    def __init__(self, cupy_module, interval_s: float) -> None:
        self._cp = cupy_module
        self._interval = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.device_peak = 0
        self.host_peak = 0
        self.samples = 0

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                free, total = self._cp.cuda.runtime.memGetInfo()
                self.device_peak = max(self.device_peak, int(total) - int(free))
            except Exception:  # noqa: BLE001 - a sampler must never kill the build
                pass
            self.host_peak = max(self.host_peak, _rss_bytes())
            self.samples += 1
            self._stop.wait(self._interval)

    def __enter__(self) -> "Sampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)


def _load(config: dict) -> tuple["np.ndarray", dict]:
    """Materialize the dataset in host RAM, or feed the memmap straight in."""
    rows = int(config["rows"])
    dimension = int(config["dimension"])
    mode = str(config.get("dataset_mode", "materialize"))
    memmap = np.load(str(config["dataset"]), mmap_mode="r")
    if memmap.ndim != 2 or int(memmap.shape[1]) != dimension:
        raise SystemExit(f"dataset shape {memmap.shape} is not (*, {dimension})")
    if rows > int(memmap.shape[0]):
        raise SystemExit(f"requested {rows} rows, dataset has {memmap.shape[0]}")
    started = time.perf_counter()
    if mode == "materialize":
        data = np.array(memmap[:rows], dtype=np.float32, order="C", copy=True)
    elif mode == "memmap":
        data = memmap[:rows]
    else:
        raise SystemExit(f"unknown dataset_mode {mode!r}")
    return data, {
        "dataset_mode": mode,
        "load_seconds": time.perf_counter() - started,
        "rss_after_load_bytes": _rss_bytes(),
        "materialized": mode == "materialize",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    os.makedirs(args.out, exist_ok=True)
    interval = float(config.get("sample_interval_s", 0.005))

    # --- instrument 1, installed before cuvs is imported --------------------
    import rmm
    import cupy as cp

    statistics = rmm.mr.StatisticsResourceAdaptor(rmm.mr.CudaMemoryResource())
    rmm.mr.set_current_device_resource(statistics)
    resource_type = str(type(rmm.mr.get_current_device_resource()))

    import cuvs
    from cuvs.neighbors import nn_descent

    # Warm the CUDA context and JIT before any measurement, so the baseline is
    # a real baseline and the build wall is a build wall.
    warm_started = time.perf_counter()
    nn_descent.build(
        nn_descent.IndexParams(
            metric=str(config["metric"]),
            graph_degree=32,
            intermediate_graph_degree=48,
            max_iterations=4,
        ),
        np.ascontiguousarray(
            np.random.default_rng(224).standard_normal((4096, 8)), dtype=np.float32
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    warmup_seconds = time.perf_counter() - warm_started

    free, total = cp.cuda.runtime.memGetInfo()
    device_total = int(total)
    device_baseline = int(total) - int(free)
    rss_baseline = _rss_bytes()
    rmm_baseline_peak = int(statistics.allocation_counts.peak_bytes)

    receipt = {
        "schema": "round0224-cuvs-memory-build-v1",
        "setting_id": str(config["setting_id"]),
        "config": config,
        "rows": int(config["rows"]),
        "dimension": int(config["dimension"]),
        "intermediate_graph_degree": int(config["intermediate_graph_degree"]),
        "graph_degree": int(config["graph_degree"]),
        "max_iterations": int(config["max_iterations"]),
        "metric": str(config["metric"]),
        "cuvs_version": str(cuvs.__version__),
        "rmm_resource_type": resource_type,
        "sample_interval_s": interval,
        "warmup_seconds": warmup_seconds,
        "device_total_bytes": device_total,
        "device_baseline_bytes": device_baseline,
        "host_rss_baseline_bytes": rss_baseline,
        "rmm_baseline_peak_bytes": rmm_baseline_peak,
    }

    try:
        dataset, load_stats = _load(config)
        receipt.update(load_stats)
        with Sampler(cp, interval) as sampler:
            started = time.perf_counter()
            index = nn_descent.build(
                nn_descent.IndexParams(
                    metric=str(config["metric"]),
                    graph_degree=int(config["graph_degree"]),
                    intermediate_graph_degree=int(
                        config["intermediate_graph_degree"]
                    ),
                    max_iterations=int(config["max_iterations"]),
                ),
                dataset,
            )
            cp.cuda.runtime.deviceSynchronize()
            build_seconds = time.perf_counter() - started
            graph = np.asarray(index.graph)
            graph_shape = [int(graph.shape[0]), int(graph.shape[1])]
            del graph, index
        receipt.update({
            "fit": True,
            "oom": False,
            "timed_out": False,
            "build_seconds": build_seconds,
            "builder_seconds": build_seconds,
            "graph_shape": graph_shape,
            "rmm_peak_bytes": int(statistics.allocation_counts.peak_bytes),
            "rmm_current_bytes": int(statistics.allocation_counts.current_bytes),
            "device_peak_sampled_bytes": int(sampler.device_peak),
            # The budget instrument: the sampler misses transient peaks and the
            # RMM counter cannot see allocations made outside RMM, so the maximum
            # of the two is used and is a LOWER BOUND on true device peak.
            "device_peak_bytes": int(
                max(sampler.device_peak, statistics.allocation_counts.peak_bytes)
            ),
            "device_peak_over_baseline_bytes": int(
                sampler.device_peak - device_baseline
            ),
            "host_peak_sampled_bytes": int(sampler.host_peak),
            "host_vmhwm_bytes": _vmhwm_bytes(),
            "host_peak_over_baseline_bytes": int(sampler.host_peak - rss_baseline),
            "samples_taken": int(sampler.samples),
        })
    except BaseException as exc:  # noqa: BLE001 - an OOM is a measurement here
        text = f"{type(exc).__name__}: {exc}".lower()
        is_oom = any(marker in text for marker in OOM_MARKERS)
        receipt.update({
            "fit": False,
            "oom": bool(is_oom),
            "timed_out": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:2000],
            "traceback_tail": traceback.format_exc()[-2000:],
            "rmm_peak_bytes": int(statistics.allocation_counts.peak_bytes),
            "host_vmhwm_bytes": _vmhwm_bytes(),
        })
        if not is_oom:
            with open(
                os.path.join(args.out, "build-receipt.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True)
                handle.write("\n")
            raise

    with open(
        os.path.join(args.out, "build-receipt.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "setting_id": receipt["setting_id"],
        "fit": receipt["fit"],
        "oom": receipt["oom"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
