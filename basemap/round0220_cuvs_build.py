#!/usr/bin/env python3
"""Build one cuVS k-NN graph. Runs under the RAPIDS env, not the release venv.

Invoked as a subprocess by `experiments/round0220_nodes.py`:

    /data/latent-basemap/cuml_py basemap/round0220_cuvs_build.py \
        --config <config.json> --out <dir>

Deliberately dependency-free apart from numpy + cuvs (+ cupy for CAGRA search),
so it never imports the `basemap` package and cannot drag the release venv's
torch into the RAPIDS process. Timings separate data load, CUDA warm-up, build
and search; only the build (and, for CAGRA, the search) is charged as builder
cost. Peak GPU memory is sampled by the parent from `nvidia-smi` against this
process's pid.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def _load(dataset: str, rows: int, dimension: int) -> "np.ndarray":
    memmap = np.load(dataset, mmap_mode="r")
    if memmap.ndim != 2 or int(memmap.shape[1]) != dimension:
        raise SystemExit(f"dataset shape {memmap.shape} is not (*, {dimension})")
    if rows > int(memmap.shape[0]):
        raise SystemExit(f"requested {rows} rows, dataset has {memmap.shape[0]}")
    # A real copy, so page faults are charged to load and never to the build.
    return np.array(memmap[:rows], dtype=np.float32, order="C", copy=True)


def _warmup(metric: str) -> float:
    """Create the CUDA context and JIT anything cuVS needs, before timing."""
    from cuvs.neighbors import nn_descent

    started = time.perf_counter()
    tiny = np.ascontiguousarray(
        np.random.default_rng(0).standard_normal((2048, 8)), dtype=np.float32
    )
    params = nn_descent.IndexParams(
        metric=metric, graph_degree=32, intermediate_graph_degree=48, max_iterations=4
    )
    index = nn_descent.build(params, tiny)
    np.asarray(index.graph)
    return time.perf_counter() - started


def _build_nn_descent(config: dict, dataset: "np.ndarray") -> tuple["np.ndarray", dict]:
    from cuvs.neighbors import nn_descent

    params = nn_descent.IndexParams(
        metric=str(config["metric"]),
        graph_degree=int(config["graph_degree"]),
        intermediate_graph_degree=int(config["intermediate_graph_degree"]),
        max_iterations=int(config["max_iterations"]),
    )
    started = time.perf_counter()
    index = nn_descent.build(params, dataset)
    build_seconds = time.perf_counter() - started
    extract_started = time.perf_counter()
    graph = np.ascontiguousarray(np.asarray(index.graph))
    extract_seconds = time.perf_counter() - extract_started
    return graph, {
        "build_seconds": build_seconds,
        "search_seconds": 0.0,
        "extract_seconds": extract_seconds,
        "builder_seconds": build_seconds,
    }


def _build_cagra(config: dict, dataset: "np.ndarray") -> tuple["np.ndarray", dict]:
    import cupy as cp
    from cuvs.neighbors import cagra

    k = int(config["k"]) + 1
    params = cagra.IndexParams(
        metric=str(config["metric"]),
        graph_degree=int(config["graph_degree"]),
        intermediate_graph_degree=int(config["intermediate_graph_degree"]),
        build_algo="nn_descent",
    )
    started = time.perf_counter()
    index = cagra.build(params, dataset)
    cp.cuda.runtime.deviceSynchronize()
    build_seconds = time.perf_counter() - started

    search_params = cagra.SearchParams(
        itopk_size=int(config["itopk_size"]), search_width=int(config["search_width"])
    )
    block = int(config.get("query_block", 250_000))
    rows = int(dataset.shape[0])
    graph = np.empty((rows, k), dtype=np.uint32)
    started = time.perf_counter()
    for start in range(0, rows, block):
        stop = min(start + block, rows)
        queries = cp.asarray(dataset[start:stop])
        _, neighbors = cagra.search(search_params, index, queries, k)
        graph[start:stop] = cp.asnumpy(cp.asarray(neighbors)).astype(np.uint32)
    cp.cuda.runtime.deviceSynchronize()
    search_seconds = time.perf_counter() - started
    return graph, {
        "build_seconds": build_seconds,
        "search_seconds": search_seconds,
        "extract_seconds": 0.0,
        "builder_seconds": build_seconds + search_seconds,
    }


def _drop_self(graph: "np.ndarray", k: int) -> "np.ndarray":
    """Remove the self column from a self-query result, keeping k neighbours."""
    rows = graph.shape[0]
    self_ids = np.arange(rows, dtype=np.int64)[:, None]
    is_self = graph.astype(np.int64) == self_ids
    first_self = np.where(is_self.any(axis=1), is_self.argmax(axis=1), graph.shape[1] - 1)
    keep = np.ones(graph.shape, dtype=bool)
    keep[np.arange(rows), first_self] = False
    return graph[keep].reshape(rows, graph.shape[1] - 1)[:, :k].astype(np.uint32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)
    os.makedirs(args.out, exist_ok=True)

    load_started = time.perf_counter()
    dataset = _load(str(config["dataset"]), int(config["rows"]), int(config["dimension"]))
    load_seconds = time.perf_counter() - load_started

    import cuvs
    from cuvs.neighbors import cagra as _cagra_module  # noqa: F401
    from cuvs.neighbors import nn_descent as _nnd_module  # noqa: F401
    import cuvs.neighbors as neighbors_package

    warmup_seconds = _warmup(str(config["metric"]))

    algo = str(config["algo"])
    if algo == "nn_descent":
        graph, timings = _build_nn_descent(config, dataset)
    elif algo == "cagra":
        graph, timings = _build_cagra(config, dataset)
        graph = _drop_self(graph, int(config["k"]))
    else:
        raise SystemExit(f"unknown R0220 algo {algo!r}")

    receipt = {
        "schema": "round0220-cuvs-build-v1",
        "setting_id": str(config["setting_id"]),
        "algo": algo,
        "config": config,
        "rows": int(dataset.shape[0]),
        "dimension": int(dataset.shape[1]),
        "graph_shape": [int(graph.shape[0]), int(graph.shape[1])],
        "graph_dtype": str(graph.dtype),
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "cuvs_version": str(cuvs.__version__),
        "cuvs_neighbors_modules": sorted(
            name for name in dir(neighbors_package) if not name.startswith("_")
        ),
        **timings,
    }
    if bool(config.get("save_graph", True)):
        graph_path = os.path.join(args.out, "graph.u32.npy")
        np.save(graph_path, graph)
        receipt["graph_path"] = graph_path
    else:
        receipt["graph_path"] = None
    with open(os.path.join(args.out, "build-receipt.json"), "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"setting_id": receipt["setting_id"], "builder_seconds": receipt["builder_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
