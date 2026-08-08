#!/usr/bin/env python3
"""R0227 — map candidate A's structural reachability ceiling against `c`.

Runs under the RAPIDS env (`/data/latent-basemap/cuml_py`), one fresh process for
the whole sweep, guarded and watched by the parent:

    cuml_py basemap/round0227_reachability_probe.py --config c.json --out d

Candidate A can only ever find a true neighbour `j` of row `i` if `j` lands in
one of the `s` clusters `i` was assigned to. That is a purely geometric ceiling
on recall — it holds before nn-descent is run and independently of how well
nn-descent works — so it can be measured directly, cheaply, and at every `c`.

Review-0226-01 measured this ceiling with its own CPU reimplementation of
k-means and got `0.985` at `c = 8` falling to `0.867` at `c = 200`, but its
`max/mean` cluster imbalance was `1.92-2.14` against the release's `1.216`, so
its absolute levels are expected to sit below the release's. **This probe calls
R0226's own `_kmeans` and `_assign` on R0216's sealed 2M substrate**, with the
registered seed, iteration count and subsample size, so the ceiling it reports is
the ceiling the release builder actually operates under.

Two ceilings are reported at every `c`, because neither may be dropped:

* **strict** — of a row's 15 exact-truth ids, the fraction that are co-clustered
  with it. Computed for **every** row of the substrate.
* **tie-aware** — the fraction of the row's 15 slots that could be filled by
  *some* co-clustered row whose exact cosine is at least the row's true 15th-best
  cosine less `1e-6`, capped at 15. The substrate provably contains
  exact-duplicate clusters, one with `1,377` members, so strict understates.
  This one needs a full similarity scan per query and runs on a seeded sample.

Plus the tripwire the mandate asks for at every `c`: rows with **zero** reachable
true neighbours, which is the structural analogue of R0215's degree-zero defect.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basemap.round0226_cluster_spill_build import (  # noqa: E402
    OOM_MARKERS,
    CooperativeAbort,
    _assign,
    _install_sigterm_handler,
    _kmeans,
    _rss_bytes,
    _vmhwm_bytes,
)
from basemap.round0226_graph_builders import A_SEED, A_SPILL, GRAPH_K  # noqa: E402
from basemap.round0227_low_c_contract import (  # noqa: E402
    REACHABILITY_SCHEMA,
    TIE_TOLERANCE,
)


#: Query rows per similarity block, and substrate rows per candidate block. Both
#: chosen so the transient device block stays near 1 GiB.
TIE_QUERY_BLOCK = 2_048
TIE_CANDIDATE_BLOCK = 131_072


def _co_clustered(assignment: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """`(rows, k)` mask: is truth neighbour `queries[i, j]` co-clustered with `i`?

    `s = 2`, so this is the four-way comparison written out. Host-side numpy: at
    2,000,000 rows the gathered table is 240 MB and the mask is 30 MB.
    """
    gathered = assignment[queries]
    mine = assignment[:, None, :]
    return (
        (gathered[:, :, 0] == mine[:, :, 0])
        | (gathered[:, :, 0] == mine[:, :, 1])
        | (gathered[:, :, 1] == mine[:, :, 0])
        | (gathered[:, :, 1] == mine[:, :, 1])
    )


def _tie_aware_ceiling(
    cp,
    *,
    device_rows,
    assignment: np.ndarray,
    query_rows: np.ndarray,
    thresholds: np.ndarray,
    k: int,
) -> np.ndarray:
    """Per-query count of co-clustered rows at or above the row's k-th cosine.

    Capped at `k` and divided by `k`, so it is directly comparable with the
    tie-aware containment the recall evaluation reports. Self is excluded.
    """
    total_rows = int(device_rows.shape[0])
    device_assignment = cp.asarray(assignment.astype(np.int32, copy=False))
    counts = np.zeros(query_rows.size, dtype=np.int64)
    for begin in range(0, query_rows.size, TIE_QUERY_BLOCK):
        end = min(begin + TIE_QUERY_BLOCK, query_rows.size)
        rows_here = query_rows[begin:end]
        device_index = cp.asarray(rows_here.astype(np.int64, copy=False))
        queries = device_rows[device_index]
        query_assign = device_assignment[device_index]
        limit = cp.asarray(
            thresholds[begin:end].astype(np.float32, copy=False)
        )[:, None] - np.float32(TIE_TOLERANCE)
        block_counts = cp.zeros(rows_here.size, dtype=cp.int64)
        for start in range(0, total_rows, TIE_CANDIDATE_BLOCK):
            stop = min(start + TIE_CANDIDATE_BLOCK, total_rows)
            scores = queries @ device_rows[start:stop].T
            candidate_assign = device_assignment[start:stop]
            co = (
                (query_assign[:, 0:1] == candidate_assign[None, :, 0])
                | (query_assign[:, 0:1] == candidate_assign[None, :, 1])
                | (query_assign[:, 1:2] == candidate_assign[None, :, 0])
                | (query_assign[:, 1:2] == candidate_assign[None, :, 1])
            )
            identity = device_index[:, None] == cp.arange(start, stop)[None, :]
            block_counts += ((scores >= limit) & co & ~identity).sum(axis=1)
            del scores, co, identity, candidate_assign
        counts[begin:end] = cp.asnumpy(block_counts)
        del queries, query_assign, limit, block_counts, device_index
        cp.get_default_memory_pool().free_all_blocks()
    del device_assignment
    cp.get_default_memory_pool().free_all_blocks()
    return np.minimum(counts, k) / float(k)


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
    cluster_counts = [int(value) for value in config["cluster_counts"]]
    tie_query_rows = int(config["tie_query_rows"])
    query_seed = int(config["query_seed"])

    import cupy as cp
    import cupyx

    receipt = {
        "schema": REACHABILITY_SCHEMA,
        "setting_id": str(config["setting_id"]),
        "config": config,
        "rows": rows,
        "dimension": dimension,
        "k": GRAPH_K,
        "spill": A_SPILL,
        "seed": A_SEED,
        "cluster_counts": cluster_counts,
        "tie_tolerance": TIE_TOLERANCE,
        "host_rss_baseline_bytes": _rss_bytes(),
        "kmeans_note": (
            "R0226's own _kmeans and _assign, imported and called unmodified "
            "with the registered seed, iteration count and subsample size, so "
            "these ceilings are the ceilings the release builder operates under"
        ),
    }
    sweep: list[dict[str, object]] = []
    try:
        substrate = np.load(str(config["substrate"]), mmap_mode="r")
        if substrate.ndim != 2 or int(substrate.shape[1]) != dimension:
            raise SystemExit(f"substrate shape {substrate.shape} is not (*, {dimension})")
        dataset = substrate[:rows]
        truth_ids = np.load(str(config["truth_ids"]))
        truth_cos = np.load(str(config["truth_cos"]))
        if truth_ids.shape != (rows, GRAPH_K) or truth_cos.shape != (rows, GRAPH_K):
            raise SystemExit("R0227 truth arrays have the wrong shape")

        truth_ids64 = truth_ids.astype(np.int64, copy=False)
        rng = np.random.default_rng(query_seed)
        query_rows = np.sort(
            rng.choice(rows, size=min(tie_query_rows, rows), replace=False)
        ).astype(np.int64)
        thresholds = truth_cos[query_rows, GRAPH_K - 1].astype(np.float64)
        receipt.update({
            "tie_query_rows": int(query_rows.size),
            "query_seed": query_seed,
        })

        started = time.perf_counter()
        device_rows = cp.asarray(np.ascontiguousarray(dataset, dtype=np.float32))
        residency_seconds = time.perf_counter() - started

        for clusters in cluster_counts:
            started = time.perf_counter()
            centroids = _kmeans(cp, cupyx, dataset, clusters=clusters, seed=A_SEED)
            kmeans_seconds = time.perf_counter() - started
            started = time.perf_counter()
            assignment = _assign(
                cp, dataset, centroids, rows=rows, spill=A_SPILL
            )
            assign_seconds = time.perf_counter() - started
            del centroids
            cp.get_default_memory_pool().free_all_blocks()

            sizes = np.bincount(assignment.ravel(), minlength=clusters).astype(np.int64)
            reachable = _co_clustered(assignment, truth_ids64)
            strict = reachable.sum(axis=1) / float(GRAPH_K)
            zero_reachable = int((strict == 0.0).sum())

            started = time.perf_counter()
            tie = _tie_aware_ceiling(
                cp,
                device_rows=device_rows,
                assignment=assignment,
                query_rows=query_rows,
                thresholds=thresholds,
                k=GRAPH_K,
            )
            tie_seconds = time.perf_counter() - started
            strict_on_queries = strict[query_rows]

            free, total = cp.cuda.runtime.memGetInfo()
            sweep.append({
                "clusters": int(clusters),
                "spill_over_clusters": float(A_SPILL) / float(clusters),
                "kmeans_seconds": float(kmeans_seconds),
                "assign_seconds": float(assign_seconds),
                "tie_scan_seconds": float(tie_seconds),
                "cluster_sizes": {
                    "min": int(sizes.min()),
                    "max": int(sizes.max()),
                    "mean": float(sizes.mean()),
                    "median": float(np.median(sizes)),
                    "empty_clusters": int((sizes == 0).sum()),
                    "imbalance_max_over_mean": float(sizes.max() / sizes.mean()),
                },
                "strict_ceiling_all_rows": {
                    "n": int(strict.size),
                    "mean": float(strict.mean()),
                    "p10": float(np.percentile(strict, 10)),
                    "min": float(strict.min()),
                    "fraction_fully_reachable": float(np.mean(strict >= 1.0)),
                },
                "strict_ceiling_on_query_sample": {
                    "n": int(strict_on_queries.size),
                    "mean": float(strict_on_queries.mean()),
                    "p10": float(np.percentile(strict_on_queries, 10)),
                    "min": float(strict_on_queries.min()),
                    "fraction_fully_reachable": float(np.mean(strict_on_queries >= 1.0)),
                },
                "tie_aware_ceiling_on_query_sample": {
                    "n": int(tie.size),
                    "mean": float(tie.mean()),
                    "p10": float(np.percentile(tie, 10)),
                    "min": float(tie.min()),
                    "fraction_fully_reachable": float(np.mean(tie >= 1.0)),
                },
                "zero_reachable_rows": zero_reachable,
                "zero_reachable_fraction": float(zero_reachable) / float(rows),
                "device_in_use_bytes_after_cell": int(total) - int(free),
            })
            del assignment, reachable, strict, tie
            cp.get_default_memory_pool().free_all_blocks()
            print(json.dumps({
                "clusters": int(clusters),
                "strict_mean": sweep[-1]["strict_ceiling_all_rows"]["mean"],
                "tie_mean": sweep[-1]["tie_aware_ceiling_on_query_sample"]["mean"],
                "zero_reachable_rows": zero_reachable,
            }), flush=True)

        del device_rows
        cp.get_default_memory_pool().free_all_blocks()
        receipt.update({
            "fit": True,
            "oom": False,
            "timed_out": False,
            "sweep": sweep,
            "substrate_residency_seconds": float(residency_seconds),
            "host_vmhwm_bytes": _vmhwm_bytes(),
        })
    except SystemExit:
        raise
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
            "sweep": sweep,
            "host_vmhwm_bytes": _vmhwm_bytes(),
        })
        if not is_oom and not isinstance(exc, CooperativeAbort):
            with open(
                os.path.join(args.out, "reachability-receipt.json"), "w",
                encoding="utf-8",
            ) as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True, default=str)
                handle.write("\n")
            raise

    with open(
        os.path.join(args.out, "reachability-receipt.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    print(json.dumps({"setting_id": receipt["setting_id"], "fit": receipt["fit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
