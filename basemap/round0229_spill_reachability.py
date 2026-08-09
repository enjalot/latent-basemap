#!/usr/bin/env python3
"""R0229 — the structural reachability ceiling against `(c, s)`, not just `c`.

Runs under the RAPIDS env (`/data/latent-basemap/cuml_py`), one fresh process for
the whole grid, guarded and watched by the parent:

    cuml_py basemap/round0229_spill_reachability.py --config c.json --out d

R0227 measured this ceiling against `c` at the constant `s = 2` that every round
since R0226 has held. The ceiling is a property of the **partition**, and the
partition has two knobs. The device cost of a build is set by the largest
cluster, so `(c, s)` pairs with the same `c / s` cost the same memory — which
makes "more, smaller clusters with higher spill" versus "fewer, larger clusters
with lower spill" a fair comparison at fixed budget, and the only comparison
that can tell Phase 2 whether `cluster-spill-nnd` has any path at 50M/100M.

This is R0227's probe with `s` generalised beyond 2, which forces two rewrites:

* `_co_clustered` was the four-way `s = 2` comparison written out. Here it is a
  cluster-membership inner product, chunked over rows, which is exact for any
  `s` and any `c`.
* the tie-aware scan's co-membership term was likewise four-way. Here it is a
  `(rows, c)` membership matrix on the device and a GEMM, which is both exact
  and far cheaper than `s x s` broadcast comparisons at `s = 8`.

`(16, 2)` and `(4, 2)` are controls: R0227's sealed strict ceilings for them are
`0.953250` and `0.991562` over all 2,000,000 rows, and reproducing them is what
binds this implementation to the instrument review-0227-01 released.
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
from basemap.round0226_graph_builders import A_SEED, GRAPH_K  # noqa: E402
from basemap.round0227_low_c_contract import TIE_TOLERANCE  # noqa: E402
from basemap.round0229_quality_contract import REACHABILITY_SCHEMA  # noqa: E402


#: Query rows per similarity block, and substrate rows per candidate block. Both
#: chosen so the transient device block stays near 1 GiB, as in R0227.
TIE_QUERY_BLOCK = 2_048
TIE_CANDIDATE_BLOCK = 131_072
#: Rows per block of the strict host-side coverage scan.
STRICT_ROW_BLOCK = 100_000


def _membership(cp, assignment: np.ndarray, clusters: int):
    """`(rows, c)` float16 membership on the device: 1 where the row is a member.

    Exact for any spill. At 2,000,000 rows and `c = 200` this is 800 MB, which
    is inside the registered 24 GiB device budget alongside the 3.07 GB resident
    substrate, and it turns every co-membership question into a GEMM.
    """
    rows = int(assignment.shape[0])
    table = cp.zeros((rows, clusters), dtype=cp.float16)
    index = cp.arange(rows, dtype=cp.int64)
    for column in range(int(assignment.shape[1])):
        table[index, cp.asarray(assignment[:, column].astype(np.int64, copy=False))] = (
            cp.float16(1.0)
        )
    del index
    return table


def _strict_ceiling(assignment: np.ndarray, truth_ids: np.ndarray) -> np.ndarray:
    """Fraction of each row's 15 exact-truth ids that share a cluster with it.

    Host-side and exact for any `s`: for every row block, compare the block's own
    `s` cluster ids against the `k x s` ids of its truth neighbours. At `s = 8`
    and a 100,000-row block the transient is 100,000 x 15 x 8 x 8 bools = 96 MB.
    """
    rows = int(assignment.shape[0])
    spill = int(assignment.shape[1])
    out = np.empty(rows, dtype=np.float64)
    for begin in range(0, rows, STRICT_ROW_BLOCK):
        end = min(begin + STRICT_ROW_BLOCK, rows)
        mine = assignment[begin:end]                     # (b, s)
        gathered = assignment[truth_ids[begin:end]]      # (b, k, s)
        shared = (
            gathered[:, :, :, None] == mine[:, None, None, :]
        ).any(axis=3).any(axis=2)                        # (b, k)
        out[begin:end] = shared.sum(axis=1) / float(truth_ids.shape[1])
        del mine, gathered, shared
    return out


def _tie_aware_ceiling(
    cp,
    *,
    device_rows,
    membership,
    query_rows: np.ndarray,
    thresholds: np.ndarray,
    k: int,
) -> np.ndarray:
    """Co-clustered rows at or above the query's k-th cosine, capped at k.

    Directly comparable with the tie-aware containment the recall evaluation
    reports. Self is excluded. Exact for any spill via the membership GEMM.
    """
    total_rows = int(device_rows.shape[0])
    counts = np.zeros(query_rows.size, dtype=np.int64)
    for begin in range(0, query_rows.size, TIE_QUERY_BLOCK):
        end = min(begin + TIE_QUERY_BLOCK, query_rows.size)
        rows_here = query_rows[begin:end]
        device_index = cp.asarray(rows_here.astype(np.int64, copy=False))
        queries = device_rows[device_index]
        query_membership = membership[device_index]
        limit = cp.asarray(
            thresholds[begin:end].astype(np.float32, copy=False)
        )[:, None] - np.float32(TIE_TOLERANCE)
        block_counts = cp.zeros(rows_here.size, dtype=cp.int64)
        for start in range(0, total_rows, TIE_CANDIDATE_BLOCK):
            stop = min(start + TIE_CANDIDATE_BLOCK, total_rows)
            scores = queries @ device_rows[start:stop].T
            co = (query_membership @ membership[start:stop].T) > cp.float16(0.0)
            identity = device_index[:, None] == cp.arange(start, stop)[None, :]
            block_counts += ((scores >= limit) & co & ~identity).sum(axis=1)
            del scores, co, identity
        counts[begin:end] = cp.asnumpy(block_counts)
        del queries, query_membership, limit, block_counts, device_index
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
    cells = [dict(cell) for cell in config["cells"]]
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
        "seed": A_SEED,
        "tie_tolerance": TIE_TOLERANCE,
        "host_rss_baseline_bytes": _rss_bytes(),
        "kmeans_note": (
            "R0226's own _kmeans and _assign, imported and called unmodified "
            "with the registered seed, iteration count and subsample size; only "
            "the spill argument varies, and it was already a keyword of _assign"
        ),
        "generalisation_note": (
            "R0227's co-clustering test was the s = 2 four-way comparison "
            "written out; here it is an exact cluster-membership test valid for "
            "any s, and (16, 2) and (4, 2) are controls against R0227's bytes"
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
            raise SystemExit("R0229 truth arrays have the wrong shape")

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

        for cell in cells:
            name = str(cell["cell"])
            clusters = int(cell["clusters"])
            spill = int(cell["spill"])
            started = time.perf_counter()
            centroids = _kmeans(cp, cupyx, dataset, clusters=clusters, seed=A_SEED)
            kmeans_seconds = time.perf_counter() - started
            started = time.perf_counter()
            assignment = _assign(cp, dataset, centroids, rows=rows, spill=spill)
            assign_seconds = time.perf_counter() - started
            del centroids
            cp.get_default_memory_pool().free_all_blocks()

            sizes = np.bincount(assignment.ravel(), minlength=clusters).astype(np.int64)
            started = time.perf_counter()
            strict = _strict_ceiling(assignment, truth_ids64)
            strict_seconds = time.perf_counter() - started
            zero_reachable = int((strict == 0.0).sum())

            membership = _membership(cp, assignment, clusters)
            started = time.perf_counter()
            tie = _tie_aware_ceiling(
                cp,
                device_rows=device_rows,
                membership=membership,
                query_rows=query_rows,
                thresholds=thresholds,
                k=GRAPH_K,
            )
            tie_seconds = time.perf_counter() - started
            del membership
            cp.get_default_memory_pool().free_all_blocks()
            strict_on_queries = strict[query_rows]

            free, total = cp.cuda.runtime.memGetInfo()
            sweep.append({
                "cell": name,
                "family": str(cell.get("family") or ""),
                "clusters": clusters,
                "spill": spill,
                "clusters_over_spill": float(clusters) / float(spill),
                "mean_cluster_rows": float(spill) * float(rows) / float(clusters),
                "kmeans_seconds": float(kmeans_seconds),
                "assign_seconds": float(assign_seconds),
                "strict_scan_seconds": float(strict_seconds),
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
            del assignment, strict, tie
            cp.get_default_memory_pool().free_all_blocks()
            print(json.dumps({
                "cell": name,
                "clusters": clusters,
                "spill": spill,
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
