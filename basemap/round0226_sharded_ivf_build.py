#!/usr/bin/env python3
"""R0226 candidate B — the R0171 sharded fp32 IVF path, measured at the same N.

Runs under the release venv (FAISS 1.14.3, sm120), as a fresh process per cell,
guarded and watched by the parent exactly like candidate A.

    .venv/bin/python basemap/round0226_sharded_ivf_build.py --config c.json --out d

R0171's law, unchanged: one empty fp32 `IndexIVFFlat/IP` coarse quantizer,
trained once on a seeded subsample, cloned into **row-disjoint** GPU shards;
every query searched against **every** shard at the same nprobe; the global
top-k taken over the union of all shards' candidates. Searching row-disjoint
shards and merging is the same candidate operation as searching their union, so
no shard is sampled, routed or skipped — which is why B's sharding costs no
recall, unlike candidate A's clustering.

`shard_rows`, not `N`, sets the device footprint: a shard is `shard_rows x 1536`
resident fp32 bytes on the card, plus the coarse centroids, one query block and
FAISS's scratch arena. Held fixed across the ladder so flatness in `N` is a
measurement rather than an assertion.

The merge is R0166/R0209's corrected one, re-derived in
`basemap.round0226_graph_builders.merge_into_topk`: FAISS returns id `-1` when a
query's probed lists hold fewer than `k` vectors *in this shard*, and R0209's
original code ranked those sentinel slots as though they were neighbours. They
are excluded here and a slot that never fills is re-emitted as `-1`.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemap.round0226_graph_builders import (  # noqa: E402
    B_NLIST,
    B_NPROBE,
    B_QUERY_BLOCK,
    B_SEARCH_K,
    B_SEED,
    B_SHARD_ROWS,
    B_TEMP_MEMORY_BYTES,
    B_TRAIN_ROWS,
    BUILD_SCHEMA,
    CANDIDATE_B,
    GRAPH_K,
    b_shard_count,
    merge_into_topk,
)


class CooperativeAbort(RuntimeError):
    """The parent watchdog asked this build to stop."""


def _install_sigterm_handler() -> None:
    """SIGTERM becomes a Python exception so CUDA can unwind normally."""

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
    """In-child device and host peaks, via torch's driver query.

    FAISS releases the GIL inside `search`, so unlike candidate A's cuVS build
    this thread is not starved; `samples_taken` is published either way.
    """

    def __init__(self, torch_module, interval_s: float) -> None:
        super().__init__(daemon=True)
        self._torch = torch_module
        self._interval = float(interval_s)
        self._halt = threading.Event()
        self.device_peak = 0
        self.host_peak = 0
        self.samples = 0

    def run(self) -> None:
        while not self._halt.is_set():
            try:
                free, total = self._torch.cuda.mem_get_info()
                self.device_peak = max(self.device_peak, int(total) - int(free))
            except Exception:  # noqa: BLE001 - a sampler must never kill a build
                pass
            self.host_peak = max(self.host_peak, _rss_bytes())
            self.samples += 1
            self._halt.wait(self._interval)

    def halt(self) -> None:
        self._halt.set()
        self.join(timeout=5)


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
    interval = float(config.get("sample_interval_s", 0.005))

    import faiss
    import torch

    free, total = torch.cuda.mem_get_info()
    shards = b_shard_count(rows)
    receipt = {
        "schema": BUILD_SCHEMA,
        "candidate": CANDIDATE_B,
        "setting_id": str(config["setting_id"]),
        "config": config,
        "rows": rows,
        "dimension": dimension,
        "k": GRAPH_K,
        "shards": shards,
        "shard_rows": min(rows, B_SHARD_ROWS),
        "nlist": B_NLIST,
        "nprobe": B_NPROBE,
        "search_k": B_SEARCH_K,
        "query_block": B_QUERY_BLOCK,
        "faiss_version": str(faiss.__version__),
        "faiss_num_gpus": int(faiss.get_num_gpus()),
        "storage": "fp32 IndexIVFFlat/IP, useFloat16=False",
        "merge_rule": (
            "exact global top-k over the union of every shard's candidates; "
            "FAISS id -1 excluded and never ranked (R0209's corrected rule); "
            "ties break on lower global id"
        ),
        "device_total_bytes": int(total),
        "device_baseline_bytes": int(total) - int(free),
        "host_rss_baseline_bytes": _rss_bytes(),
        # RMM is a RAPIDS allocator; FAISS does not route through it. Published
        # as null rather than dropped, so the instrument table stays complete.
        "rmm_peak_bytes": None,
        "sample_interval_s": interval,
    }

    sampler = Sampler(torch, interval)
    sampler.start()
    phase: dict[str, float] = {}
    resources = None
    try:
        substrate = np.load(str(config["substrate"]), mmap_mode="r")
        if substrate.ndim != 2 or int(substrate.shape[1]) != dimension:
            raise SystemExit(f"substrate shape {substrate.shape} is not (*, {dimension})")
        if rows > int(substrate.shape[0]):
            raise SystemExit(f"requested {rows} rows, substrate has {substrate.shape[0]}")
        dataset = substrate[:rows]

        started = time.perf_counter()
        take = min(rows, B_TRAIN_ROWS)
        rng = np.random.default_rng(B_SEED)
        train_rows = np.sort(rng.choice(rows, size=take, replace=False))
        training = np.ascontiguousarray(dataset[train_rows], dtype=np.float32)
        quantizer = faiss.IndexFlatIP(dimension)
        template = faiss.IndexIVFFlat(
            quantizer, dimension, B_NLIST, faiss.METRIC_INNER_PRODUCT
        )
        template.train(training)
        del training
        phase["train_seconds"] = time.perf_counter() - started
        if not template.is_trained:
            raise RuntimeError("R0226 candidate B coarse quantizer did not train")

        resources = faiss.StandardGpuResources()
        resources.setTempMemory(int(B_TEMP_MEMORY_BYTES))
        options = faiss.GpuClonerOptions()
        # R0170's lesson, cited in R0171: useFloat16 does NOT change
        # GpuIndexIVFFlat storage, so fp32 is stated by leaving it off and the
        # storage claim rests on the index type, not on this flag.
        options.useFloat16 = False

        top_ids = np.full((rows, GRAPH_K), -1, dtype=np.int32)
        top_cos = np.full((rows, GRAPH_K), -np.inf, dtype=np.float32)

        add_seconds = 0.0
        search_seconds = 0.0
        merge_seconds = 0.0
        shard_receipts: list[dict[str, object]] = []

        for shard in range(shards):
            begin = shard * B_SHARD_ROWS
            end = min(begin + B_SHARD_ROWS, rows)
            started = time.perf_counter()
            host_index = faiss.clone_index(template)
            gpu_index = faiss.index_cpu_to_gpu(resources, 0, host_index, options)
            stage = 250_000
            for cursor in range(begin, end, stage):
                stop = min(cursor + stage, end)
                gpu_index.add_with_ids(
                    np.ascontiguousarray(dataset[cursor:stop], dtype=np.float32),
                    np.arange(cursor, stop, dtype=np.int64),
                )
            add_seconds += time.perf_counter() - started
            if int(gpu_index.ntotal) != end - begin:
                raise RuntimeError(
                    f"shard {shard} holds {int(gpu_index.ntotal)} vectors against "
                    f"{end - begin} assigned"
                )
            gpu_index.nprobe = B_NPROBE

            for query_start in range(0, rows, B_QUERY_BLOCK):
                query_stop = min(query_start + B_QUERY_BLOCK, rows)
                queries = np.ascontiguousarray(
                    dataset[query_start:query_stop], dtype=np.float32
                )
                started = time.perf_counter()
                shard_cos, shard_ids = gpu_index.search(queries, B_SEARCH_K)
                search_seconds += time.perf_counter() - started
                started = time.perf_counter()
                merge_into_topk(
                    top_ids,
                    top_cos,
                    rows=np.arange(query_start, query_stop, dtype=np.int64),
                    candidate_ids=shard_ids,
                    candidate_cos=shard_cos,
                    k=GRAPH_K,
                )
                merge_seconds += time.perf_counter() - started
                del queries, shard_cos, shard_ids
            shard_receipts.append({
                "shard": int(shard),
                "start": int(begin),
                "stop": int(end),
                "rows": int(end - begin),
                "ntotal": int(gpu_index.ntotal),
            })
            del gpu_index, host_index

        phase.update({
            "shard_add_seconds": add_seconds,
            "search_seconds": search_seconds,
            "merge_seconds": merge_seconds,
        })
        degree = (top_ids >= 0).sum(axis=1)
        receipt.update({
            "fit": True,
            "oom": False,
            "timed_out": False,
            "builder_seconds": float(sum(phase.values())),
            "build_seconds": float(sum(phase.values())),
            "phases": phase,
            "shard_receipts": shard_receipts,
            "graph_shape": [int(rows), GRAPH_K],
            "zero_degree_rows": int((degree == 0).sum()),
            "rows_below_k": int((degree < GRAPH_K).sum()),
            "min_degree": int(degree.min()),
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
            "phases": phase,
            "child_device_peak_sampled_bytes": int(sampler.device_peak),
            "host_peak_sampled_bytes": int(sampler.host_peak),
            "host_vmhwm_bytes": _vmhwm_bytes(),
        })
        if not is_oom and not isinstance(exc, CooperativeAbort):
            sampler.halt()
            with open(
                os.path.join(args.out, "build-receipt.json"), "w", encoding="utf-8"
            ) as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True, default=str)
                handle.write("\n")
            raise
    finally:
        sampler.halt()

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
